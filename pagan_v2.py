"""
pagan_v2.py  (v3)
=================
PAGANv2: Broad-to-Tight decentralised learning protocol.

Signal: EMA distance (τ-independent, stored in ShadowRegistry.ema_dist)
Aggregation: softmax(-D_ij / τ(t)) over sampled nodes + self, where
             τ(t) decays from τ_0 to τ_min over tau_half_life post-warmup rounds.

Why EMA distance + scheduled τ:
  - EMA damps transient closeness (caution against sudden movers).
  - Distance is absolute, τ-independent — history is clean regardless of schedule.
  - Scheduled τ amplifies persistent differences: a node must stay close over
    many rounds (EMA) AND be evaluated when τ is small (schedule) to get high weight.
  - Self has EMA distance 0 → always highest logit → self-weight autoscales.

Phases
------
Phase 1 — Warmup  (rounds 0 .. warmup_rounds-1)
    Pure isolation. Feedback sampler. EMA distances accumulate on clean signal.
Phase 2 — Broad-to-tight  (rounds warmup_rounds ..)
    Aggregation weights = softmax(-D / τ(t)) over sampled nodes + self.
    τ decays from τ_0 to τ_min over tau_half_life rounds post-warmup.

Flags (ablation)
----------------
freeze_ema        : snapshot EMA at warmup end, hold fixed thereafter.
freeze_embeddings : stop embedding updates at warmup end.
"""

import math, random as _random
import numpy as np
import torch
from collections import defaultdict

from shadow_registry import ShadowRegistry


class PAGANv2:
    def __init__(self,
                 num_nodes:          int,
                 device,
                 total_rounds:       int   = 500,
                 warmup_rounds:      int   = 20,
                 # EMA
                 ema_lambda:         float = 0.95,
                 # Temperature schedule (aggregation only)
                 tau_0:              float = 2.0,
                 tau_min:            float = 0.3,
                 tau_half_life:      float = 200.0,
                 # Shadow registry recording tau (diagnostic sw only)
                 softmax_tau:        float = 1.0,
                 # Sampling
                 shadow_window:      int   = 20,   # kept for legacy affinity plots
                 # Flags
                 freeze_ema:         bool  = False,
                 freeze_embeddings:  bool  = False,
                 # Diagnostics
                 node_to_cluster:    np.ndarray = None,
                 eff_weight_thresh:  float = 0.02,
                 debug_node:         int   = 0):

        self.N              = num_nodes
        self.device         = device
        self.total_rounds   = total_rounds
        self.warmup_rounds  = warmup_rounds
        self.ema_lambda     = ema_lambda
        self.tau_0          = tau_0
        self.tau_min        = tau_min
        self.tau_half_life  = tau_half_life
        self.shadow_window  = shadow_window
        self.freeze_ema     = freeze_ema
        self.freeze_embeddings = freeze_embeddings
        self.node_to_cluster = node_to_cluster
        self.eff_thresh     = eff_weight_thresh
        self.debug_node     = debug_node

        self.registry = ShadowRegistry(num_nodes, total_rounds,
                                        ema_lambda=ema_lambda,
                                        softmax_tau=softmax_tau)
        self._frozen_ema    = None   # [N, N] snapshot if freeze_ema=True
        self.last_W         = None
        self.soft_metrics_log = []

    # ------------------------------------------------------------------
    # Temperature schedule
    # ------------------------------------------------------------------
    def tau(self, rnd: int) -> float:
        t   = max(0, rnd - self.warmup_rounds)
        lam = math.log(max((self.tau_0 - self.tau_min) / max(self.tau_min, 1e-8),
                            1.0 + 1e-8)) / max(self.tau_half_life, 1.0)
        return max(self.tau_min, self.tau_0 * math.exp(-lam * t))

    # ------------------------------------------------------------------
    # Record
    # ------------------------------------------------------------------
    def record(self, rnd: int, ranked_neighbors: list, ranked_dists: list):
        self.registry.record(rnd, ranked_neighbors, ranked_dists)

    # ------------------------------------------------------------------
    # Freeze EMA snapshot
    # ------------------------------------------------------------------
    def freeze_ema_now(self, rnd: int):
        self._frozen_ema = self.registry.snapshot_ema_dist()
        print(f"[PAGANv2] EMA distances frozen at round {rnd}.")

    # ------------------------------------------------------------------
    # EMA distance lookup
    # ------------------------------------------------------------------
    def _ema_row(self, node_i: int) -> np.ndarray:
        """Returns [N] EMA distances for node_i. inf = never observed."""
        if self.freeze_ema and self._frozen_ema is not None:
            return self._frozen_ema[node_i]
        return self.registry.get_ema_dist_row(node_i)

    # ------------------------------------------------------------------
    # Sampling targets
    # ------------------------------------------------------------------
    def get_sampling_targets(self, rnd: int, node_i: int, k_total: int,
                              physical_rank_list: list, E_list_i) -> list:
        """
        Quota (k_total=20 default):
          Slot A — top-3 physical persistence   (all rounds)
          Slot B — top-7 by EMA distance        (post-warmup)
          Slot C — top-5 embedding              (post-warmup)
          Slot D — random fill
        """
        N       = self.N
        targets = set()

        # Slot A
        for c in physical_rank_list[:3]:
            if c != node_i:
                targets.add(c)

        if rnd >= self.warmup_rounds:
            # Slot B — closest by EMA distance (excluding self and already chosen)
            D = self._ema_row(node_i)
            observed = [(j, float(D[j])) for j in range(N)
                        if j != node_i and not np.isinf(D[j])
                        and j not in targets]
            observed.sort(key=lambda x: x[1])   # ascending: closer first
            for j, _ in observed[:7]:
                targets.add(j)

            # Slot C — embedding
            if len(targets) < k_total:
                my_emb    = E_list_i[node_i].unsqueeze(0)
                emb_d     = torch.norm(E_list_i - my_emb, dim=1)
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
    # Aggregation
    # ------------------------------------------------------------------
    def run_topology_and_aggregate(self, rnd: int,
                                   node_states: torch.Tensor,
                                   E_list, ranked_neighbors, prev_ranked_neighbors):
        """Returns new_node_states [N, D]."""

        # Warmup: no aggregation
        if rnd < self.warmup_rounds:
            if rnd % 5 == 0:
                spread = self.registry.dist_spread[self.debug_node, rnd]
                s = f"{spread:.4f}" if not np.isnan(spread) else "n/a"
                print(f"[v2 warmup rnd {rnd:3d}] "
                      f"Node {self.debug_node} dist_spread={s}")
            return node_states.clone()

        # Freeze EMA on first post-warmup round if requested
        if self.freeze_ema and self._frozen_ema is None:
            self.freeze_ema_now(rnd)

        tau   = self.tau(rnd)
        N, _  = node_states.shape
        W     = torch.zeros(N, N, device=self.device, dtype=node_states.dtype)

        for i in range(N):
            D = self._ema_row(i)   # [N] EMA distances, inf = never seen

            # sampled this round + self
            sampled    = ranked_neighbors[i].cpu().tolist()
            all_nodes  = sampled + [i]

            # EMA distances for these nodes (self = 0.0)
            d_vec = np.array([0.0 if j == i else
                               (float(D[j]) if not np.isinf(D[j])
                                else 999.0)           # never seen → very far
                               for j in all_nodes], dtype=np.float32)

            logits  = -d_vec / max(tau, 1e-8)
            logits -= logits.max()                    # numerical stability
            exp_w   = np.exp(logits)
            w_np    = exp_w / (exp_w.sum() + 1e-12)

            for idx, (j, w) in enumerate(zip(all_nodes, w_np)):
                W[i, j] = float(w)

        self.last_W = W.detach()

        # Debug
        if rnd % 10 == 0 or rnd == self.warmup_rounds:
            dn   = self.debug_node
            sw   = W[dn, dn].item()
            top5 = torch.topk(W[dn], min(6, N)).indices.tolist()
            top5 = [x for x in top5 if x != dn][:5]
            print(f"[v2 rnd {rnd:3d}] tau={tau:.3f}  "
                  f"Node {dn}: self_w={sw:.3f}  top5_peers={top5}")

        return torch.mm(W, node_states).detach()

    # ------------------------------------------------------------------
    # Soft TP/FP
    # ------------------------------------------------------------------
    def compute_soft_metrics(self, rnd: int) -> dict:
        if self.last_W is None or self.node_to_cluster is None:
            return {}
        W  = self.last_W.cpu().numpy()
        cc = defaultdict(int)
        for c in self.node_to_cluster: cc[c] += 1
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
        result   = dict(round=rnd, tp=tp, fp=fp, fn=fn, tn=tn,
                        avg_self_weight=avg_self, tau=tau if False else self.tau(rnd))
        self.soft_metrics_log.append(result)
        return result

    def get_weight_heatmap(self) -> np.ndarray:
        return self.last_W.cpu().numpy() if self.last_W is not None else None