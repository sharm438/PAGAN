"""
pagan_v3.py
===========
PAGANv3: Blended model-distance + embedding-distance affinity.

Design
------
v2 used EMA model distance alone for affinity. Two confirmed problems:
  1. Mutual attraction: strangers drifting physically close get high weight.
  2. Affinity is relative to the k sampled nodes, not the full N population.

v3 adds a second distance signal — embedding distance — and blends them
with a scheduled weight α(t):

    score(i, j) = α(t) × D_ema(i,j)  +  (1-α(t)) × D_emb(i,j)
    logit(i, j) = -score(i, j) / τ(t)
    W[i, :]     = softmax(logits)  over sampled k nodes + self

α schedule (linear):
    α = 1.0  for rnd < alpha_start              (pure model distance)
    α decays linearly from 1.0 → 0.0            (transition window)
    α = 0.0  for rnd >= alpha_end               (pure embedding distance)

Rationale:
    - Early rounds: embeddings not yet trained, model distance is reliable.
    - Late rounds: embeddings have learned cluster topology and are more
      resistant to contamination (trained on ranklist structure, not raw
      parameter proximity). As α→0, the feedback loop loses its fuel:
      a stranger drifting close in model space no longer gains affinity
      if the embedding correctly places them far.

Triplet loss weighting (new):
    Three schemes for how much each neighbor's ranklist contributes to
    the embedding gradient. All use rank position (0=closest):
      'flat'          : weight = 1.0   (v2 behavior, uniform)
      'inv_rank'      : weight = 1 / (rank + 1)
      'inv_sqrt_rank' : weight = 1 / sqrt(rank + 1)

    Closer neighbors' ranklists are more informative about i's local
    neighbourhood — they share more distributional overlap. Rank-weighted
    schemes give the gradient more signal per step.

Diagnostics:
    Soft tp/fp tracked separately for three variants each round:
      - model_only:  softmax(-D_ema / τ)
      - emb_only:    softmax(-D_emb / τ)
      - blended:     softmax(-score / τ)    [used for actual aggregation]
    This lets us watch each signal's cluster separation quality over time.

Everything else (EMA, τ schedule, 4-slot sampling, warmup) is identical to v2.
"""

import math
import random as _random
import numpy as np
import torch
from collections import defaultdict

from shadow_registry import ShadowRegistry


class PAGANv3:

    def __init__(self,
                 num_nodes: int,
                 device,
                 total_rounds: int = 500,
                 warmup_rounds: int = 25,
                 # EMA (unchanged from v2)
                 ema_lambda: float = 0.95,
                 # Temperature schedule (unchanged from v2)
                 tau_0: float = 2.0,
                 tau_min: float = 0.3,
                 tau_half_life: float = 200.0,
                 softmax_tau: float = 1.0,
                 shadow_window: int = 20,
                 # Ablation flags (preserved from v2)
                 freeze_ema: bool = False,
                 freeze_embeddings: bool = False,
                 # Diagnostics
                 node_to_cluster: np.ndarray = None,
                 eff_weight_thresh: float = 0.02,
                 debug_node: int = 0,
                 # ── v3-specific ──────────────────────────────────────────
                 # Blend schedule
                 alpha_start: int = None,    # round where α begins decaying
                                             # default: warmup_rounds
                 alpha_end: int = None,      # round where α reaches 0.0
                                             # default: total_rounds
                 # Triplet loss weighting scheme
                 triplet_weight_scheme: str = 'flat',   # 'flat' | 'inv_rank' | 'inv_sqrt_rank'
                 verbose: bool = True):

        # ── Base (identical to v2) ───────────────────────────────────
        self.N = num_nodes
        self.device = device
        self.total_rounds = total_rounds
        self.warmup_rounds = warmup_rounds
        self.ema_lambda = ema_lambda
        self.tau_0 = tau_0
        self.tau_min = tau_min
        self.tau_half_life = tau_half_life
        self.shadow_window = shadow_window
        self.freeze_ema = freeze_ema
        self.freeze_embeddings = freeze_embeddings
        self.node_to_cluster = node_to_cluster
        self.eff_thresh = eff_weight_thresh
        self.debug_node = debug_node
        self.verbose = verbose

        # ── v3 blend schedule ────────────────────────────────────────
        self.alpha_start = alpha_start if alpha_start is not None else warmup_rounds
        self.alpha_end   = alpha_end   if alpha_end   is not None else total_rounds
        assert self.alpha_end > self.alpha_start, \
            "alpha_end must be greater than alpha_start"

        # ── v3 triplet weighting ─────────────────────────────────────
        assert triplet_weight_scheme in ('flat', 'inv_rank', 'inv_sqrt_rank'), \
            f"Unknown scheme: {triplet_weight_scheme}"
        self.triplet_weight_scheme = triplet_weight_scheme

        # ── EMA registry (identical to v2) ───────────────────────────
        self.registry = ShadowRegistry(num_nodes, total_rounds,
                                       ema_lambda=ema_lambda,
                                       softmax_tau=softmax_tau)
        self._frozen_ema = None
        self.last_W = None

        # ── Diagnostic stores ────────────────────────────────────────
        # Three parallel weight matrices for the three signal variants
        self.last_W_model = None    # model-distance-only weights
        self.last_W_emb   = None    # embedding-distance-only weights
        self.last_W_blend = None    # blended weights (= last_W, used for aggregation)

        self.soft_metrics_log = []  # list of dicts, one per eval round

        if verbose:
            print(f"[PAGANv3] init: N={num_nodes}  warmup={warmup_rounds}  "
                  f"λ={ema_lambda}  τ: {tau_0}→{tau_min} (hl={tau_half_life})\n"
                  f"  blend: α=1.0 until rnd={self.alpha_start}, "
                  f"linear→0.0 by rnd={self.alpha_end}\n"
                  f"  triplet_weight_scheme='{triplet_weight_scheme}'")

    # ------------------------------------------------------------------
    # α(t) — blend coefficient: 1.0 = model only, 0.0 = embedding only
    # ------------------------------------------------------------------
    def alpha(self, rnd: int) -> float:
        if rnd <= self.alpha_start:
            return 1.0
        if rnd >= self.alpha_end:
            return 0.0
        progress = (rnd - self.alpha_start) / (self.alpha_end - self.alpha_start)
        return float(1.0 - progress)   # linear decay

    # ------------------------------------------------------------------
    # Temperature schedule (identical to v2)
    # ------------------------------------------------------------------
    def tau(self, rnd: int) -> float:
        t = max(0, rnd - self.warmup_rounds)
        lam = math.log(
            max((self.tau_0 - self.tau_min) / max(self.tau_min, 1e-8), 1.0 + 1e-8)
        ) / max(self.tau_half_life, 1.0)
        return max(self.tau_min, self.tau_0 * math.exp(-lam * t))

    # ------------------------------------------------------------------
    # Record (identical to v2 — EMA update only)
    # ------------------------------------------------------------------
    def record(self, rnd: int, ranked_neighbors: list, ranked_dists: list):
        self.registry.record(rnd, ranked_neighbors, ranked_dists)

    # ------------------------------------------------------------------
    # EMA row (identical to v2)
    # ------------------------------------------------------------------
    def _ema_row(self, node_i: int) -> np.ndarray:
        if self.freeze_ema and self._frozen_ema is not None:
            return self._frozen_ema[node_i]
        return self.registry.get_ema_dist_row(node_i)

    # ------------------------------------------------------------------
    # Sampling targets (identical to v2 — 4-slot quota)
    # ------------------------------------------------------------------
    def get_sampling_targets(self, rnd: int, node_i: int, k_total: int,
                              physical_rank_list: list, E_list_i) -> list:
        N = self.N
        targets = set()

        # Slot A — top-3 physical persistence
        for c in physical_rank_list[:3]:
            if c != node_i:
                targets.add(c)

        if rnd >= self.warmup_rounds:
            # Slot B — top-7 by EMA distance
            D = self._ema_row(node_i)
            observed = [(j, float(D[j])) for j in range(N)
                        if j != node_i and not np.isinf(D[j])
                        and j not in targets]
            observed.sort(key=lambda x: x[1])
            for j, _ in observed[:7]:
                targets.add(j)

            # Slot C — top-5 by embedding distance
            if len(targets) < k_total:
                my_emb = E_list_i[node_i].unsqueeze(0)
                emb_d = torch.norm(E_list_i - my_emb, dim=1)
                emb_sorted = torch.argsort(emb_d).tolist()
                added = 0
                for c in emb_sorted:
                    if c != node_i and c not in targets:
                        targets.add(c)
                        added += 1
                    if added >= 5 or len(targets) >= k_total:
                        break

        # Slot D — random fill
        pool = [x for x in range(N) if x != node_i and x not in targets]
        _random.shuffle(pool)
        for c in pool:
            if len(targets) >= k_total:
                break
            targets.add(c)

        return list(targets)

    # ------------------------------------------------------------------
    # Triplet loss evidence weights
    # ------------------------------------------------------------------
    def get_evidence_weights(self, evidence: list) -> list:
        """
        Returns parallel weight lists for the triplet loss, one list per node.
        Each weight corresponds to one evidence item (neighbor's ranklist).
        Weight depends on the rank of that neighbor in i's physical ranking
        (rank_idx=0 is i's closest sampled neighbor).

        Schemes:
          flat          : 1.0 for all  (v2 behavior)
          inv_rank      : 1 / (rank_idx + 1)
          inv_sqrt_rank : 1 / sqrt(rank_idx + 1)
        """
        scheme = self.triplet_weight_scheme
        weights = []
        for items_i in evidence:
            w_i = []
            for rank_idx, _ in enumerate(items_i):
                if scheme == 'flat':
                    w_i.append(1.0)
                elif scheme == 'inv_rank':
                    w_i.append(1.0 / (rank_idx + 1))
                else:   # inv_sqrt_rank
                    w_i.append(1.0 / math.sqrt(rank_idx + 1))
            weights.append(w_i)
        return weights

    # ------------------------------------------------------------------
    # Core: blended softmax aggregation
    # ------------------------------------------------------------------
    def _softmax_weights(self, d_vec: np.ndarray, tau: float) -> np.ndarray:
        """Numerically stable softmax over a distance vector."""
        logits = -d_vec / max(tau, 1e-8)
        logits -= logits.max()
        exp_w = np.exp(logits)
        return exp_w / (exp_w.sum() + 1e-12)

    def run_topology_and_aggregate(self, rnd: int,
                                   node_states: torch.Tensor,
                                   E_list, ranked_neighbors,
                                   prev_ranked_neighbors):
        """
        Warmup: pure isolation.
        Post-warmup: W = softmax(-score/τ) where
            score = α(t)×D_ema + (1-α(t))×D_emb
        Also computes model-only and emb-only weights for diagnostics.
        """
        if rnd < self.warmup_rounds:
            if rnd % 5 == 0:
                spread = self.registry.dist_spread[self.debug_node, rnd]
                s = f"{spread:.4f}" if not np.isnan(spread) else "n/a"
                print(f"[v3 warmup rnd {rnd:3d}] "
                      f"Node {self.debug_node} dist_spread={s}")
            return node_states.clone()

        if self.freeze_ema and self._frozen_ema is None:
            self._frozen_ema = self.registry.snapshot_ema_dist()

        tau = self.tau(rnd)
        a   = self.alpha(rnd)
        N, _ = node_states.shape

        W_blend = torch.zeros(N, N, device=self.device, dtype=node_states.dtype)
        W_model = torch.zeros(N, N, device=self.device, dtype=node_states.dtype)
        W_emb   = torch.zeros(N, N, device=self.device, dtype=node_states.dtype)

        for i in range(N):
            D_ema = self._ema_row(i)   # [N] EMA model distances
            sampled   = ranked_neighbors[i].cpu().tolist()
            all_nodes = sampled + [i]

            # ── Model distances (EMA) ───────────────────────────────
            d_model = np.array([
                0.0 if j == i else
                (float(D_ema[j]) if not np.isinf(D_ema[j]) else 999.0)
                for j in all_nodes
            ], dtype=np.float32)

            # ── Embedding distances (fresh, from i's perspective) ───
            with torch.no_grad():
                Ei       = E_list[i]                    # [N, embed_dim]
                self_emb = Ei[i].unsqueeze(0)           # [1, embed_dim]
                all_idx  = torch.tensor(all_nodes,
                                        device=Ei.device, dtype=torch.long)
                emb_vecs = Ei[all_idx]                  # [k+1, embed_dim]
                self_emb_exp = Ei[i].unsqueeze(0).expand_as(emb_vecs)
                d_emb_t  = torch.norm(emb_vecs - self_emb_exp, dim=1)
                d_emb    = d_emb_t.cpu().numpy().astype(np.float32)
                # Self embedding distance is 0 by construction
                self_idx_in_list = all_nodes.index(i)
                d_emb[self_idx_in_list] = 0.0

            # ── Blended score ────────────────────────────────────────
            d_blend = a * d_model + (1.0 - a) * d_emb

            # ── Softmax for each variant ─────────────────────────────
            w_blend = self._softmax_weights(d_blend, tau)
            w_model = self._softmax_weights(d_model, tau)
            w_emb   = self._softmax_weights(d_emb,   tau)

            for j, (wb, wm, we) in zip(all_nodes, zip(w_blend, w_model, w_emb)):
                W_blend[i, j] = float(wb)
                W_model[i, j] = float(wm)
                W_emb[i, j]   = float(we)

        self.last_W       = W_blend.detach()
        self.last_W_blend = W_blend.detach()
        self.last_W_model = W_model.detach()
        self.last_W_emb   = W_emb.detach()

        if self.verbose and (rnd % 10 == 0 or rnd == self.warmup_rounds):
            dn  = self.debug_node
            sw  = W_blend[dn, dn].item()
            top5_idx = torch.topk(W_blend[dn], min(6, N)).indices.tolist()
            top5 = [x for x in top5_idx if x != dn][:5]
            print(f"[v3 rnd {rnd:3d}] τ={tau:.3f}  α={a:.3f}  "
                  f"Node {dn}: self_w={sw:.3f}  top5={top5}")

        return torch.mm(W_blend, node_states).detach()

    # ------------------------------------------------------------------
    # Soft TP/FP — three variants
    # ------------------------------------------------------------------
    def _soft_metrics_from_W(self, W: np.ndarray, rnd: int,
                              label: str) -> dict:
        """Compute tp/fp/fn/tn/avg_self from a weight matrix."""
        cc = defaultdict(int)
        for c in self.node_to_cluster:
            cc[c] += 1
        tp = fp = fn = tn = 0
        for i in range(self.N):
            my_c  = self.node_to_cluster[i]
            ideal = cc[my_c] - 1
            eff   = [j for j in range(self.N)
                     if j != i and W[i, j] > self.eff_thresh]
            m  = sum(1 for j in eff if self.node_to_cluster[j] == my_c)
            mm = len(eff) - m
            tp += m;  fp += mm
            fn += max(0, ideal - m)
            tn += (self.N - 1 - ideal) - mm
        avg_self = float(np.mean([W[i, i] for i in range(self.N)]))
        return dict(label=label, round=rnd, tp=tp, fp=fp, fn=fn, tn=tn,
                    avg_self_weight=avg_self)

    def compute_soft_metrics(self, rnd: int) -> dict:
        """
        Returns a dict with three sub-dicts: model_only, emb_only, blended.
        Also appended to self.soft_metrics_log.
        """
        if self.last_W_blend is None or self.node_to_cluster is None:
            return {}

        W_b = self.last_W_blend.cpu().numpy()
        W_m = self.last_W_model.cpu().numpy()
        W_e = self.last_W_emb.cpu().numpy()

        result = dict(
            round   = rnd,
            alpha   = self.alpha(rnd),
            tau     = self.tau(rnd),
            blended = self._soft_metrics_from_W(W_b, rnd, 'blended'),
            model   = self._soft_metrics_from_W(W_m, rnd, 'model'),
            emb     = self._soft_metrics_from_W(W_e, rnd, 'emb'),
        )
        self.soft_metrics_log.append(result)
        return result

    def get_weight_heatmap(self) -> np.ndarray:
        return self.last_W.cpu().numpy() if self.last_W is not None else None

    def freeze_ema_now(self, rnd: int):
        self._frozen_ema = self.registry.snapshot_ema_dist()
        print(f"[PAGANv3] EMA frozen at round {rnd}.")