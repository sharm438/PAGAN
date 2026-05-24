"""
pagan_v4.py
===========
PAGANv4: Trust-gated aggregation with warmup consistency anchoring.

Core design
-----------
v2 used EMA distance alone for affinity. Mutual attraction occurred because
strangers drifting physically close received high aggregation weight — the
EMA had no way to distinguish legitimate proximity from contamination.

v4 cleanly separates two concerns:

  (1) OBSERVATION  — EMA distance, running freely with a single global λ.
                     Tracks physical reality without defensive tuning.

  (2) TRUST GATE   — a scalar trust(i,j) ∈ [0,1] that determines how much
                     of that physical reality matters for aggregation and
                     embedding learning.

Trust sources
-------------
  Warmup consistency (frozen anchor):
    Was j close to i *before* any mixing? Computed from bucket mode,
    observation count, and bucket concentration across warmup rounds.
    Peers seen close and consistently score near 1.0. Peers never seen
    score 0.0.

  Vouching (fresh every round, zero extra cost):
    When j appears this round, look at j's own ranked list. Count how
    many of j's top-M neighbors are i's warmup friends. High overlap
    means j is embedded in i's trusted neighbourhood even if i never
    directly observed j in warmup.

  Conflict suppression (frozen):
    If warmup *did* observe j but placed j consistently far (bad bucket),
    and j now appears close, this is a known conflict. Trust is hard-capped
    regardless of vouching — warmup evidence of being a stranger overrides
    any post-mix proximity signal.

Combined trust formula
----------------------
  W_ij   = count_factor × bucket_score × concentration   (warmup anchor)
  vouch  = |j's top-M ∩ i's warmup friends| / M          (fresh vouching)
  trust  = W_ij + (1 - W_ij) × vouch                     (additive fill)
  trust  = min(trust, conflict_cap)   if conflict_mask[i,j]

Aggregation
-----------
  logit(i,j) = log(trust(i,j) + ε)  −  D_ij / τ(t)
  W = softmax(logits)  over sampled peers + self

  Self: trust=1, D=0 → always highest logit → self-weight autoscales as τ→0.
  Trusted close friend: high logit from both terms.
  Stranger drifted close: small D_ij suppressed by log(trust≈0) ≈ -∞.
  Vouched newcomer: moderate trust lifts logit proportionally.

Embedding evidence
------------------
  weight(i,j) = trust(i,j) / (rank_of_j_in_i's_list + 1)

  Combines two independent quality signals:
    - Trust: whose ranklist is worth learning from?
    - Rank:  how representative is this peer of i's local neighbourhood?

Parameters (new, beyond v2 base)
---------------------------------
  n_buckets              int   4       Quartile buckets (same as v3)
  warmup_friend_thresh   float 0.3     W_ij above this → i's warmup friend
  conflict_bucket_thresh int   2       warmup_bucket ≥ this → conflict pair
  conflict_cap           float 0.15    Max trust for conflict pairs
  vouch_top_M            int   5       j's top-M neighbours checked for vouch
  trust_log_eps          float 1e-3    ε in log(trust + ε), floor for logits

No per-pair lambda, no Case-2, no embedding maturity gate,
no trust recompute schedule. EMA runs with single global λ.
"""

import math
import random as _random
import numpy as np
import torch
from collections import defaultdict

from shadow_registry import ShadowRegistry


class PAGANv4:
    """
    PAGANv4 protocol. Drop-in alternative to PAGANv2/v3.
    Constructor signature: v2 base args + v4-specific knobs at the end.
    """

    def __init__(self,
                 num_nodes: int,
                 device,
                 total_rounds: int = 500,
                 warmup_rounds: int = 25,
                 # EMA — single global lambda, no per-pair tuning
                 ema_lambda: float = 0.95,
                 # Temperature schedule (identical to v2)
                 tau_0: float = 2.0,
                 tau_min: float = 0.5,
                 tau_half_life: float = 200.0,
                 # Shadow-registry diagnostic tau
                 softmax_tau: float = 1.0,
                 # Sampling
                 shadow_window: int = 20,
                 k_sample: int = 20,
                 # Ablation flags (preserved from v2)
                 freeze_ema: bool = False,
                 freeze_embeddings: bool = False,
                 # Diagnostics
                 node_to_cluster: np.ndarray = None,
                 eff_weight_thresh: float = 0.02,
                 debug_node: int = 0,
                 # ── v4-specific ──────────────────────────────────────
                 n_buckets: int = 4,
                 warmup_friend_threshold: float = 0.3,
                 conflict_bucket_threshold: int = 2,
                 conflict_cap: float = 0.15,
                 vouch_top_M: int = 5,
                 trust_log_eps: float = 1e-3,
                 verbose: bool = True):

        # ── Base state (mirrors v2) ──────────────────────────────────
        self.N = num_nodes
        self.device = device
        self.total_rounds = total_rounds
        self.warmup_rounds = warmup_rounds
        self.ema_lambda = ema_lambda
        self.tau_0 = tau_0
        self.tau_min = tau_min
        self.tau_half_life = tau_half_life
        self.shadow_window = shadow_window
        self.k_sample = k_sample
        self.freeze_ema = freeze_ema
        self.freeze_embeddings = freeze_embeddings
        self.node_to_cluster = node_to_cluster
        self.eff_thresh = eff_weight_thresh
        self.debug_node = debug_node
        self.verbose = verbose

        # ── v4 hyperparameters ───────────────────────────────────────
        self.n_buckets = n_buckets
        self.warmup_friend_threshold = warmup_friend_threshold
        self.conflict_bucket_threshold = conflict_bucket_threshold
        self.conflict_cap = conflict_cap
        self.vouch_top_M = vouch_top_M
        self.trust_log_eps = trust_log_eps

        # ── EMA registry (single global λ, same as v2) ───────────────
        self.registry = ShadowRegistry(num_nodes, total_rounds,
                                       ema_lambda=ema_lambda,
                                       softmax_tau=softmax_tau)
        self._frozen_ema = None
        self.last_W = None
        self.soft_metrics_log = []

        # ── Warmup bucket tracking (same as v3) ─────────────────────
        # [N, N, warmup_rounds], -1 = not sampled that round
        self.bucket_history_warmup = np.full(
            (num_nodes, num_nodes, warmup_rounds), -1, dtype=np.int8)

        # Frozen at warmup end:
        self.warmup_bucket = np.full((num_nodes, num_nodes), -1, dtype=np.int8)
        self.warmup_count = np.zeros((num_nodes, num_nodes), dtype=np.int32)
        self.warmup_bucket_concentration = np.zeros(
            (num_nodes, num_nodes), dtype=np.float32)
        self._d_warmup_snapshot = None      # [N, N], inf for never-observed

        # Derived anchors (computed once at warmup end, then frozen):
        self.W_warmup = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        # conflict_mask[i,j] = True if warmup saw j as consistently far
        self.conflict_mask = np.zeros((num_nodes, num_nodes), dtype=bool)
        # warmup_friends[i] = set of j where W_warmup[i,j] > threshold
        self.warmup_friends: list[set] = [set() for _ in range(num_nodes)]

        # ── Live trust matrix [N, N] ─────────────────────────────────
        # Updated each round for sampled pairs only.
        # Diagonal = 1.0 always. Off-diagonal initialised to 0, then to
        # W_warmup at warmup end, then updated via warmup+vouching each round.
        self.trust = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        np.fill_diagonal(self.trust, 1.0)

        # ── Diagnostic ───────────────────────────────────────────────
        self.trust_snapshots = []   # list of {round, mean, q10, q90}

        if verbose:
            print(f"[PAGANv4] init: N={num_nodes}  warmup={warmup_rounds}  "
                  f"λ={ema_lambda}  τ: {tau_0}→{tau_min} (hl={tau_half_life})  "
                  f"friend_thresh={warmup_friend_threshold}  "
                  f"conflict_bucket≥{conflict_bucket_threshold}  "
                  f"conflict_cap={conflict_cap}  vouch_M={vouch_top_M}")

    # ------------------------------------------------------------------
    # Temperature schedule (identical to v2)
    # ------------------------------------------------------------------
    def tau(self, rnd: int) -> float:
        t = max(0, rnd - self.warmup_rounds)
        lam = math.log(
            max((self.tau_0 - self.tau_min) / max(self.tau_min, 1e-8),
                1.0 + 1e-8)
        ) / max(self.tau_half_life, 1.0)
        return max(self.tau_min, self.tau_0 * math.exp(-lam * t))

    # ------------------------------------------------------------------
    # Bucket assignment (identical to v3)
    # ------------------------------------------------------------------
    def _bucket_of(self, rank_pos: int, n_sampled: int) -> int:
        """Maps rank position (0-indexed) to bucket in [0, n_buckets)."""
        if n_sampled <= 0:
            return 0
        bucket_size = max(1, n_sampled / self.n_buckets)
        return min(int(rank_pos // bucket_size), self.n_buckets - 1)

    # ------------------------------------------------------------------
    # Record — dispatches to warmup or post-warmup path
    # ------------------------------------------------------------------
    def record(self, rnd: int, ranked_neighbors: list, ranked_dists: list):
        """Call once per round after rank_neighbors_by_model_distance."""
        if rnd < self.warmup_rounds:
            self._record_warmup(rnd, ranked_neighbors, ranked_dists)
        else:
            if self._d_warmup_snapshot is None:
                self._finalize_warmup()
            self._record_postwarmup(rnd, ranked_neighbors, ranked_dists)

    # ------------------------------------------------------------------
    def _record_warmup(self, rnd: int, ranked_neighbors, ranked_dists):
        """Warmup: standard EMA recording + bucket position tracking."""
        self.registry.record(rnd, ranked_neighbors, ranked_dists)

        for i in range(self.N):
            nbrs = ranked_neighbors[i]
            if nbrs.numel() == 0:
                continue
            nbrs_np = nbrs.cpu().numpy().astype(int)
            n = len(nbrs_np)
            for rank_pos, j in enumerate(nbrs_np):
                b = self._bucket_of(rank_pos, n)
                self.bucket_history_warmup[i, j, rnd] = b

        if self.verbose and rnd % 5 == 0:
            spread = self.registry.dist_spread[self.debug_node, rnd]
            s = f"{spread:.4f}" if not np.isnan(spread) else "n/a"
            print(f"[v4 warmup rnd {rnd:3d}] "
                  f"Node {self.debug_node} dist_spread={s}")

    # ------------------------------------------------------------------
    def _finalize_warmup(self):
        """
        Called once on the first post-warmup round.
        Computes frozen anchors: W_warmup, conflict_mask, warmup_friends.
        Bootstraps trust from warmup consistency.
        """
        if self.verbose:
            print(f"[PAGANv4] Finalizing warmup state...")

        for i in range(self.N):
            for j in range(self.N):
                if i == j:
                    continue
                hist = self.bucket_history_warmup[i, j, :self.warmup_rounds]
                valid = hist[hist >= 0]

                if len(valid) == 0:
                    # Never observed in warmup
                    self.warmup_bucket[i, j] = -1
                    self.warmup_count[i, j] = 0
                    self.warmup_bucket_concentration[i, j] = 0.0
                    self.W_warmup[i, j] = 0.0
                    # No conflict possible (no warmup data)
                else:
                    cnt = len(valid)
                    self.warmup_count[i, j] = cnt
                    counts = np.bincount(valid, minlength=self.n_buckets)
                    mode_bucket = int(counts.argmax())
                    concentration = float(counts.max()) / cnt

                    self.warmup_bucket[i, j] = mode_bucket
                    self.warmup_bucket_concentration[i, j] = concentration

                    # W_warmup: how much to trust based on warmup alone
                    #   count_factor  — saturates at warmup_rounds observations
                    #   bucket_score  — bucket 0 (close) → 1.0, bucket 3 → 0.0
                    #   concentration — how consistently j held that bucket
                    count_factor = min(cnt / max(self.warmup_rounds, 1), 1.0)
                    bucket_score = 1.0 - mode_bucket / max(self.n_buckets - 1, 1)
                    self.W_warmup[i, j] = (count_factor
                                           * bucket_score
                                           * concentration)

                    # Conflict: warmup placed j in far half AND we have
                    # enough observations to be confident about it.
                    if (mode_bucket >= self.conflict_bucket_threshold
                            and cnt >= 3):
                        self.conflict_mask[i, j] = True

        # Warmup friends: nodes i has seen closely and consistently
        for i in range(self.N):
            self.warmup_friends[i] = {
                j for j in range(self.N)
                if j != i
                and self.W_warmup[i, j] > self.warmup_friend_threshold
            }

        # Snapshot EMA at warmup end (used by _affinity_row if needed)
        self._d_warmup_snapshot = self.registry.snapshot_ema_dist()

        # Bootstrap trust from warmup consistency
        for i in range(self.N):
            for j in range(self.N):
                if i == j:
                    self.trust[i, j] = 1.0
                    continue
                w = float(self.W_warmup[i, j])
                # No vouching yet (no post-warmup ranklists available);
                # conflict cap applied from the start.
                t = min(w, self.conflict_cap) if self.conflict_mask[i, j] else w
                self.trust[i, j] = float(np.clip(t, 0.0, 1.0))

        if self.verbose:
            seen = np.sum(self.warmup_count > 0) / (self.N * (self.N - 1))
            avg_friends = sum(len(fs) for fs in self.warmup_friends) / self.N
            n_conflict = self.conflict_mask.sum() / (self.N * (self.N - 1))
            off_diag = ~np.eye(self.N, dtype=bool)
            print(f"  warmup coverage={seen:.2%}  "
                  f"avg_friends/node={avg_friends:.1f}  "
                  f"conflict_pairs={n_conflict:.2%}")
            print(f"  trust@bootstrap: "
                  f"mean={self.trust[off_diag].mean():.3f}  "
                  f"q90={np.quantile(self.trust[off_diag], 0.90):.3f}")

    # ------------------------------------------------------------------
    def _compute_trust(self, i: int, j: int,
                       j_ranked_neighbors) -> float:
        """
        Compute trust(i, j) for one sampled pair using:
          - frozen warmup consistency W_warmup[i,j]
          - fresh vouching from j's current ranked list
          - conflict cap if warmup identified j as a far peer

        j_ranked_neighbors : LongTensor of j's neighbours this round,
                             sorted closest-first.
        """
        w_ij = float(self.W_warmup[i, j])

        # Vouching: overlap between j's top-M and i's warmup friends
        if (len(self.warmup_friends[i]) > 0
                and j_ranked_neighbors.numel() > 0):
            j_top_M = j_ranked_neighbors[:self.vouch_top_M].cpu().tolist()
            overlap = sum(1 for k in j_top_M
                          if k in self.warmup_friends[i])
            vouch = overlap / max(len(j_top_M), 1)
        else:
            vouch = 0.0

        # Combine: warmup is primary; vouching fills in for unseen peers
        raw = w_ij + (1.0 - w_ij) * vouch

        # Conflict cap: warmup evidence of being a stranger is not overridden
        if self.conflict_mask[i, j]:
            raw = min(raw, self.conflict_cap)

        return float(np.clip(raw, 0.0, 1.0))

    # ------------------------------------------------------------------
    def _record_postwarmup(self, rnd: int, ranked_neighbors, ranked_dists):
        """
        Post-warmup recording:
          1. EMA update via registry (single global λ, no lambda_matrix).
          2. Trust recomputation for all sampled pairs using fresh ranklists.
        """
        # Standard EMA update — free-running, no defensive lambda
        self.registry.record(rnd, ranked_neighbors, ranked_dists)

        # Recompute trust for every sampled pair this round
        for i in range(self.N):
            nbrs = ranked_neighbors[i]
            if nbrs.numel() == 0:
                continue
            for j_t in nbrs:
                j = int(j_t)
                self.trust[i, j] = self._compute_trust(
                    i, j, ranked_neighbors[j])

        # Periodic diagnostic snapshot
        if rnd % 50 == 0:
            off_diag = ~np.eye(self.N, dtype=bool)
            vals = self.trust[off_diag]
            snap = {
                'round': rnd,
                'mean': float(vals.mean()),
                'q10': float(np.quantile(vals, 0.10)),
                'q90': float(np.quantile(vals, 0.90)),
            }
            self.trust_snapshots.append(snap)
            if self.verbose:
                print(f"[v4 rnd {rnd:3d}] trust: "
                      f"mean={snap['mean']:.3f}  "
                      f"q10={snap['q10']:.3f}  "
                      f"q90={snap['q90']:.3f}")

    # ------------------------------------------------------------------
    # EMA distance row
    # ------------------------------------------------------------------
    def _ema_row(self, node_i: int) -> np.ndarray:
        """[N] EMA distances for node_i. inf = never observed."""
        if self.freeze_ema and self._frozen_ema is not None:
            return self._frozen_ema[node_i]
        return self.registry.get_ema_dist_row(node_i)

    # ------------------------------------------------------------------
    # Sampling targets — identical 4-slot quota to v2
    # ------------------------------------------------------------------
    def get_sampling_targets(self, rnd: int, node_i: int, k_total: int,
                              physical_rank_list: list, E_list_i) -> list:
        """
        Slot A — top-3 physical persistence   (all rounds)
        Slot B — top-7 by EMA distance        (post-warmup)
        Slot C — top-5 by embedding distance  (post-warmup)
        Slot D — random fill
        """
        N = self.N
        targets = set()

        # Slot A
        for c in physical_rank_list[:3]:
            if c != node_i:
                targets.add(c)

        if rnd >= self.warmup_rounds:
            # Slot B — EMA-closest (unfiltered — EMA is free-running)
            D = self._ema_row(node_i)
            observed = [
                (j, float(D[j]))
                for j in range(N)
                if j != node_i and not np.isinf(D[j]) and j not in targets
            ]
            observed.sort(key=lambda x: x[1])
            for j, _ in observed[:7]:
                targets.add(j)

            # Slot C — embedding-closest
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
    # Trust weights for embedding evidence
    # ------------------------------------------------------------------
    def get_trust_weights_for_evidence(self, evidence) -> list:
        """
        For each node i's evidence list (already ordered by physical rank),
        return parallel weights:

            weight(i, j, rank_idx) = trust(i, j) / (rank_idx + 1)

        This combines:
          - Source credibility: whose ranklist is worth learning from?
          - Rank proximity: how representative is this peer of i's neighbourhood?

        The two signals are independent and multiplicative, so the combined
        weight is zero iff trust is zero, and decays harmonically with rank.
        """
        trust_weights = []
        for i, items_i in enumerate(evidence):
            w_i = []
            for rank_idx, (j, _, _) in enumerate(items_i):
                if self._d_warmup_snapshot is None:
                    # Pre-finalization (shouldn't happen post-warmup, but safe)
                    w_i.append(1.0 / (rank_idx + 1))
                else:
                    t = max(float(self.trust[i, j]), 0.01)
                    w_i.append(t / (rank_idx + 1))
            trust_weights.append(w_i)
        return trust_weights

    # ------------------------------------------------------------------
    # Aggregation — trust-gated softmax
    # ------------------------------------------------------------------
    def run_topology_and_aggregate(self, rnd: int,
                                   node_states: torch.Tensor,
                                   E_list, ranked_neighbors,
                                   prev_ranked_neighbors):
        """
        Warmup: pure isolation (no aggregation).
        Post-warmup: W = softmax(log(trust) - D/τ) over sampled + self.
        """
        if rnd < self.warmup_rounds:
            return node_states.clone()

        if self.freeze_ema and self._frozen_ema is None:
            self._frozen_ema = self.registry.snapshot_ema_dist()

        tau = self.tau(rnd)
        N, _ = node_states.shape
        W = torch.zeros(N, N, device=self.device, dtype=node_states.dtype)
        eps = self.trust_log_eps

        for i in range(N):
            D = self._ema_row(i)
            sampled = ranked_neighbors[i].cpu().tolist()
            all_nodes = sampled + [i]   # self appended last

            logits = np.empty(len(all_nodes), dtype=np.float32)
            for idx, j in enumerate(all_nodes):
                if j == i:
                    # Self: trust=1, D=0 → logit = log(1) - 0 = 0
                    logits[idx] = 0.0
                else:
                    t_ij = max(float(self.trust[i, j]), eps)
                    d_ij = (float(D[j]) if not np.isinf(D[j]) else 999.0)
                    logits[idx] = math.log(t_ij) - d_ij / max(tau, 1e-8)

            logits -= logits.max()   # numerical stability
            exp_w = np.exp(logits)
            w_np = exp_w / (exp_w.sum() + 1e-12)

            for j, w in zip(all_nodes, w_np):
                W[i, j] = float(w)

        self.last_W = W.detach()

        # Debug printout
        if self.verbose and (rnd % 10 == 0 or rnd == self.warmup_rounds):
            dn = self.debug_node
            sw = W[dn, dn].item()
            top5_idx = torch.topk(W[dn], min(6, N)).indices.tolist()
            top5 = [x for x in top5_idx if x != dn][:5]
            print(f"[v4 rnd {rnd:3d}] tau={tau:.3f}  "
                  f"Node {dn}: self_w={sw:.3f}  top5={top5}  "
                  f"trust={[f'{self.trust[dn,j]:.2f}' for j in top5]}")

        return torch.mm(W, node_states).detach()

    # ------------------------------------------------------------------
    # Soft TP/FP metrics (identical to v2)
    # ------------------------------------------------------------------
    def compute_soft_metrics(self, rnd: int) -> dict:
        if self.last_W is None or self.node_to_cluster is None:
            return {}
        W = self.last_W.cpu().numpy()
        cc = defaultdict(int)
        for c in self.node_to_cluster:
            cc[c] += 1
        tp = fp = fn = tn = 0
        for i in range(self.N):
            my_c = self.node_to_cluster[i]
            ideal = cc[my_c] - 1
            eff = [j for j in range(self.N)
                   if j != i and W[i, j] > self.eff_thresh]
            m = sum(1 for j in eff if self.node_to_cluster[j] == my_c)
            mm = len(eff) - m
            tp += m
            fp += mm
            fn += max(0, ideal - m)
            tn += (self.N - 1 - ideal) - mm
        avg_self = float(np.mean([W[i, i] for i in range(self.N)]))
        result = dict(
            round=rnd, tp=tp, fp=fp, fn=fn, tn=tn,
            avg_self_weight=avg_self,
            tau=self.tau(rnd),
            trust_mean=float(
                self.trust[~np.eye(self.N, dtype=bool)].mean()),
        )
        self.soft_metrics_log.append(result)
        return result

    def get_weight_heatmap(self) -> np.ndarray:
        return self.last_W.cpu().numpy() if self.last_W is not None else None

    def get_trust_snapshot(self) -> np.ndarray:
        return self.trust.copy()