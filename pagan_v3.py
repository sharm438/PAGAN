"""
pagan_v3.py
===========
PAGANv3: Trust-aware decentralised learning with regime-switching affinity updates.

Core idea
---------
PAGANv2 uses EMA-smoothed model distance as the affinity signal, with a single
global lambda. Post-warmup mixing contaminates this signal — strangers can drift
close, and EMA absorbs the drift, eventually causing mutual-attraction failures
between true strangers.

PAGANv3 addresses this with three additions on top of v2:

(1) Per-pair adaptive lambda (asymmetric inertia).
    Most pair updates use λ_safe (fast). The one exception: a low-trust pair
    that appears to be improving uses λ_cautious (slow) — requiring confirmation
    over multiple rounds before its affinity rises.

(2) Trust-score-driven Case-2 detection and defense.
    For high-trust pairs, we track bucket-position consistency between current
    and warmup-era ranklists. A trusted pair drifting significantly worse
    triggers Case 2: aggregation uses max(d_live, d_warmup) instead of d_live,
    shielding the node from contamination contagion while still permitting
    mixing if d_live remains consistent with the prior.

(3) Trust-weighted embedding training.
    Each evidence pair in the ladder triplet loss is weighted by trust(i, j),
    so untrusted peers' ranklists contribute less to embedding training.

Trust is bootstrapped from warmup data alone at the end of warmup, then
recomputed every K rounds combining warmup statistics, peer vouching, and
(after a maturity gate) embedding alignment.

Phases
------
Phase 1 — Warmup  (rounds 0 .. warmup_rounds-1)
    Pure isolation. EMA distances accumulate with single lambda.
    Bucket positions of each (i, j) recorded per round.
Phase 2 — Post-warmup  (rounds warmup_rounds ..)
    Aggregation weights = softmax(-a_ij / τ(t)) over sampled + self.
    a_ij = max(d_live, d_warmup) for Case-2 pairs, else d_live.
    Per-pair lambdas based on trust × direction.
    Trust scores recomputed every trust_recompute_every rounds.
"""

import math
import random as _random
import numpy as np
import torch
from collections import defaultdict

from shadow_registry import ShadowRegistry


class PAGANv3:
    """
    PAGANv3 protocol. Drop-in alternative to PAGANv2 with trust-aware updates.

    Constructor signature mirrors PAGANv2 plus the v3-specific knobs at the end.
    """

    def __init__(self,
                 num_nodes: int,
                 device,
                 total_rounds: int = 500,
                 warmup_rounds: int = 25,
                 # EMA defaults (v2-compatible; v3 overrides per-pair)
                 ema_lambda: float = 0.95,
                 # Temperature schedule (unchanged from v2)
                 tau_0: float = 2.0,
                 tau_min: float = 0.5,
                 tau_half_life: float = 200.0,
                 # Shadow-registry diagnostic tau
                 softmax_tau: float = 1.0,
                 # Sampling
                 shadow_window: int = 20,
                 k_sample: int = 20,
                 # v2 ablation flags (preserved)
                 freeze_ema: bool = False,
                 freeze_embeddings: bool = False,
                 # Diagnostics
                 node_to_cluster: np.ndarray = None,
                 eff_weight_thresh: float = 0.02,
                 debug_node: int = 0,
                 # ── v3-specific ──────────────────────────────────────
                 lambda_safe: float = 0.85,
                 lambda_cautious: float = 0.97,
                 case2_trigger_K: int = 2,
                 case2_exit_K: int = 2,
                 n_buckets: int = 4,
                 bucket_drift_threshold: int = 2,
                 peer_trust_top_K: int = 5,
                 embedding_maturity_round: int = 200,
                 trust_recompute_every: int = 5,
                 trust_high_threshold: float = 0.5,
                 trust_weight_components: tuple = (1.0, 1.0, 1.0),
                 verbose: bool = True):

        # ── Inherited v2 state ────────────────────────────────────────
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

        # ── v3 hyperparameters ────────────────────────────────────────
        self.lambda_safe = lambda_safe
        self.lambda_cautious = lambda_cautious
        self.case2_trigger_K = case2_trigger_K
        self.case2_exit_K = case2_exit_K
        self.n_buckets = n_buckets
        self.bucket_drift_threshold = bucket_drift_threshold
        self.peer_trust_top_K = peer_trust_top_K
        self.embedding_maturity_round = embedding_maturity_round
        self.trust_recompute_every = trust_recompute_every
        self.trust_high_threshold = trust_high_threshold
        # Weights for (warmup_term, vouching_term, embedding_term) in trust score
        self.trust_w_warmup = trust_weight_components[0]
        self.trust_w_vouch = trust_weight_components[1]
        self.trust_w_embed = trust_weight_components[2]

        # ── Underlying registry ──────────────────────────────────────
        self.registry = ShadowRegistry(num_nodes, total_rounds,
                                        ema_lambda=ema_lambda,
                                        softmax_tau=softmax_tau)
        self._frozen_ema = None
        self.last_W = None
        self.soft_metrics_log = []

        # ── v3 per-pair state ────────────────────────────────────────
        N = num_nodes
        # Bucket history during warmup: [N, N, warmup_rounds], -1 = not sampled
        self.bucket_history_warmup = np.full(
            (N, N, warmup_rounds), -1, dtype=np.int8)
        # Frozen at warmup-end:
        self.warmup_bucket = np.full((N, N), -1, dtype=np.int8)
        self.warmup_count = np.zeros((N, N), dtype=np.int32)
        self.warmup_bucket_concentration = np.zeros((N, N), dtype=np.float32)
        self._d_warmup_snapshot = None   # [N, N], inf for unobserved pairs

        # Trust score (continuous in [0, 1]). Recomputed periodically.
        self.trust = np.zeros((N, N), dtype=np.float32)
        np.fill_diagonal(self.trust, 1.0)

        # Case-2 state
        self.case2_active = np.zeros((N, N), dtype=bool)
        self.case2_flag_counter = np.zeros((N, N), dtype=np.int16)
        self.case2_consistent_counter = np.zeros((N, N), dtype=np.int16)

        # Most recent observation of each pair (round index, bucket position).
        # Used so update_trust_scores / vouching can use latest info if it ever
        # extends beyond warmup state. Round -1 means never observed.
        self.last_observed_round = np.full((N, N), -1, dtype=np.int32)
        self.last_observed_bucket = np.full((N, N), -1, dtype=np.int8)

        # ── Diagnostic logs (light) ──────────────────────────────────
        self.case2_events = []    # list of (round, i, j, "enter"/"exit")
        self.trust_snapshots = [] # list of {round, mean, max, q90, q10}

        if self.verbose:
            print(f"[PAGANv3] init: N={N}, warmup={warmup_rounds}, "
                  f"λ_safe={lambda_safe}, λ_cautious={lambda_cautious}, "
                  f"trust_thresh={trust_high_threshold}, "
                  f"embed_mature@{embedding_maturity_round}")

    # ------------------------------------------------------------------
    # Temperature schedule (unchanged from v2)
    # ------------------------------------------------------------------
    def tau(self, rnd: int) -> float:
        t = max(0, rnd - self.warmup_rounds)
        lam = math.log(max((self.tau_0 - self.tau_min) / max(self.tau_min, 1e-8),
                            1.0 + 1e-8)) / max(self.tau_half_life, 1.0)
        return max(self.tau_min, self.tau_0 * math.exp(-lam * t))

    # ------------------------------------------------------------------
    # Bucket assignment
    # ------------------------------------------------------------------
    def _bucket_of(self, rank_pos: int, n_sampled: int) -> int:
        """Maps a rank position (0-indexed) into a discrete bucket in [0, n_buckets)."""
        if n_sampled <= 0:
            return 0
        bucket_size = max(1, n_sampled / self.n_buckets)
        return min(int(rank_pos // bucket_size), self.n_buckets - 1)

    # ------------------------------------------------------------------
    # Record — branches into warmup vs post-warmup logic
    # ------------------------------------------------------------------
    def record(self, rnd: int, ranked_neighbors: list, ranked_dists: list):
        """Call once per round, after rank_neighbors_by_model_distance."""
        if rnd < self.warmup_rounds:
            self._record_warmup(rnd, ranked_neighbors, ranked_dists)
        else:
            # Finalize warmup state on the first post-warmup call.
            if self._d_warmup_snapshot is None:
                self._finalize_warmup()
            self._record_postwarmup(rnd, ranked_neighbors, ranked_dists)

    # ------------------------------------------------------------------
    def _record_warmup(self, rnd, ranked_neighbors, ranked_dists):
        """Warmup: standard EMA update via registry, plus bucket tracking."""
        # Standard EMA recording (same as v2)
        self.registry.record(rnd, ranked_neighbors, ranked_dists)

        # Track bucket positions for warmup statistics
        for i in range(self.N):
            nbrs = ranked_neighbors[i]
            if nbrs.numel() == 0:
                continue
            nbrs_np = nbrs.cpu().numpy().astype(int)
            n = len(nbrs_np)
            for rank_pos, j in enumerate(nbrs_np):
                b = self._bucket_of(rank_pos, n)
                self.bucket_history_warmup[i, j, rnd] = b
                self.last_observed_round[i, j] = rnd
                self.last_observed_bucket[i, j] = b

    # ------------------------------------------------------------------
    def _record_postwarmup(self, rnd, ranked_neighbors, ranked_dists):
        """
        Post-warmup:
          1. Compute current bucket and bucket_change per sampled pair
          2. Determine per-pair lambda via asymmetric inertia rule
          3. Update Case-2 trigger/exit counters
          4. Call registry.record() with per-pair lambda matrix
        """
        # Build lambda matrix (default: safe lambda for all pairs)
        lambda_matrix = np.full(
            (self.N, self.N), self.lambda_safe, dtype=np.float32)

        # We also record (current_bucket, bucket_change) for active pairs
        # for diagnostic / Case-2 logic
        per_pair_changes = []   # list of (i, j, current_bucket, bucket_change)

        for i in range(self.N):
            nbrs = ranked_neighbors[i]
            if nbrs.numel() == 0:
                continue
            nbrs_np = nbrs.cpu().numpy().astype(int)
            n = len(nbrs_np)

            for rank_pos, j in enumerate(nbrs_np):
                current_bucket = self._bucket_of(rank_pos, n)
                warm_b = int(self.warmup_bucket[i, j])
                if warm_b >= 0:
                    bucket_change = current_bucket - warm_b
                else:
                    bucket_change = None

                # Asymmetric inertia rule:
                #   Default = λ_safe (most cases fine with fast update).
                #   EXCEPTION: low-trust pair appearing to improve → λ_cautious.
                #
                # "Improve" means current_bucket is closer than warmup_bucket
                # (negative bucket_change) OR no warmup baseline exists.
                if self.trust[i, j] < self.trust_high_threshold:
                    if bucket_change is not None and bucket_change < 0:
                        lambda_matrix[i, j] = self.lambda_cautious
                    elif bucket_change is None:
                        # No warmup info; be cautious by default
                        lambda_matrix[i, j] = self.lambda_cautious
                # else: high-trust pair → λ_safe (let live signal respond).

                per_pair_changes.append((i, j, current_bucket, bucket_change))

                # Update most-recent observation
                self.last_observed_round[i, j] = rnd
                self.last_observed_bucket[i, j] = current_bucket

        # Update Case-2 state based on bucket changes
        self._update_case2_state(rnd, per_pair_changes)

        # EMA update with per-pair lambdas
        self.registry.record(rnd, ranked_neighbors, ranked_dists,
                              lambda_matrix=lambda_matrix)

    # ------------------------------------------------------------------
    def _update_case2_state(self, rnd, per_pair_changes):
        """
        For each (i, j) where we have warmup baseline AND trust is high:
          - If bucket_change >= bucket_drift_threshold (worse): flag counter++
          - Else: consistent counter++
          - Trigger Case 2 if flag counter reaches case2_trigger_K
          - Exit Case 2 if consistent counter reaches case2_exit_K
        """
        for (i, j, current_bucket, bucket_change) in per_pair_changes:
            if bucket_change is None:
                continue
            if self.trust[i, j] < self.trust_high_threshold:
                # Only trusted pairs can enter Case 2
                continue

            if bucket_change >= self.bucket_drift_threshold:
                # Drift away from warmup expectation
                self.case2_flag_counter[i, j] += 1
                self.case2_consistent_counter[i, j] = 0
                if (not self.case2_active[i, j]
                        and self.case2_flag_counter[i, j] >= self.case2_trigger_K):
                    self.case2_active[i, j] = True
                    self.case2_events.append((rnd, int(i), int(j), 'enter'))
                    if self.verbose and i == self.debug_node:
                        print(f"[v3 rnd {rnd:3d}] Case 2 ENTER: ({i}, {j})  "
                              f"warmup_bucket={int(self.warmup_bucket[i,j])} "
                              f"current_bucket={current_bucket} "
                              f"trust={self.trust[i,j]:.3f}")
            else:
                # Consistent observation
                self.case2_consistent_counter[i, j] += 1
                self.case2_flag_counter[i, j] = 0
                if (self.case2_active[i, j]
                        and self.case2_consistent_counter[i, j] >= self.case2_exit_K):
                    self.case2_active[i, j] = False
                    self.case2_events.append((rnd, int(i), int(j), 'exit'))
                    if self.verbose and i == self.debug_node:
                        print(f"[v3 rnd {rnd:3d}] Case 2 EXIT:  ({i}, {j})  "
                              f"trust={self.trust[i,j]:.3f}")

    # ------------------------------------------------------------------
    # Finalize warmup — compute warmup_bucket, warmup_count, concentration,
    # d_warmup snapshot, and bootstrap trust scores.
    # ------------------------------------------------------------------
    def _finalize_warmup(self):
        if self.verbose:
            print(f"[PAGANv3] Finalizing warmup state...")

        for i in range(self.N):
            for j in range(self.N):
                if i == j:
                    continue
                hist = self.bucket_history_warmup[i, j, :self.warmup_rounds]
                valid = hist[hist >= 0]
                if len(valid) == 0:
                    self.warmup_bucket[i, j] = -1
                    self.warmup_count[i, j] = 0
                    self.warmup_bucket_concentration[i, j] = 0.0
                else:
                    self.warmup_count[i, j] = len(valid)
                    counts = np.bincount(valid, minlength=self.n_buckets)
                    self.warmup_bucket[i, j] = int(counts.argmax())
                    self.warmup_bucket_concentration[i, j] = (
                        float(counts.max()) / len(valid))

        # Snapshot d_warmup from registry's EMA
        self._d_warmup_snapshot = self.registry.snapshot_ema_dist()

        # Bootstrap trust scores
        self._bootstrap_trust()

        if self.verbose:
            mean_count = np.mean(self.warmup_count[self.warmup_count > 0])
            pairs_seen = np.sum(self.warmup_count > 0) / (self.N * (self.N - 1))
            print(f"  warmup_count: mean(non-zero)={mean_count:.2f}  "
                  f"coverage={pairs_seen:.2%}")
            print(f"  trust@bootstrap: mean={self.trust.mean():.3f}  "
                  f"high(>{self.trust_high_threshold})="
                  f"{(self.trust > self.trust_high_threshold).sum() / (self.N**2):.2%}")

    # ------------------------------------------------------------------
    def _bootstrap_trust(self):
        """
        Bootstrap trust from warmup data alone (no vouching, no embeddings).
        trust(i, j) = saturating_count * bucket_concentration.
        Diagonal = 1.0.
        """
        for i in range(self.N):
            for j in range(self.N):
                if i == j:
                    self.trust[i, j] = 1.0
                    continue
                count_factor = min(self.warmup_count[i, j] / 5.0, 1.0)
                concentration = self.warmup_bucket_concentration[i, j]
                # Normalize: this is just the warmup component, so it lives
                # in [0, 1]. We don't divide by 3 here because vouching and
                # embedding terms are 0 at bootstrap.
                self.trust[i, j] = count_factor * concentration

    # ------------------------------------------------------------------
    # Trust score recomputation — called from main.py every K rounds
    # ------------------------------------------------------------------
    def update_trust_scores(self, rnd: int, E_list=None):
        """
        Recompute trust scores using:
          - warmup_term: count × bucket_concentration (frozen)
          - vouching_term: for each i, query top-K trusted peers' warmup_bucket of j
          - embedding_term: post round embedding_maturity_round, cosine similarity

        Called externally from main.py after the round's record() and embedding update.
        """
        if rnd < self.warmup_rounds:
            return
        if self._d_warmup_snapshot is None:
            return

        # Identify trusted peers per node (top-K by current trust, excluding self
        # and Case-2-active pairs).
        eligible_peers = {}
        for i in range(self.N):
            scores = self.trust[i].copy()
            scores[i] = -1.0
            # Exclude pairs in Case 2 (drift detected)
            for j in range(self.N):
                if self.case2_active[i, j]:
                    scores[j] = -1.0
            top_peers = np.argsort(-scores)[:self.peer_trust_top_K]
            # Filter out anything with trust 0 or negative
            top_peers = [p for p in top_peers if scores[p] > 0]
            eligible_peers[i] = top_peers

        use_embedding = (rnd >= self.embedding_maturity_round and E_list is not None)

        new_trust = np.zeros_like(self.trust)
        for i in range(self.N):
            # Precompute i's own embedding row if needed
            if use_embedding:
                Ei = E_list[i].detach().cpu().float().numpy()
                my_pos = Ei[i]
                emb_dists = np.linalg.norm(Ei - my_pos, axis=1)
                # Normalize embedding distances to [0, 1] by max (excluding self)
                non_self = np.concatenate([emb_dists[:i], emb_dists[i+1:]])
                emb_max = max(non_self.max(), 1e-6) if len(non_self) else 1.0

            for j in range(self.N):
                if i == j:
                    new_trust[i, j] = 1.0
                    continue

                # Component 1: warmup term
                count_factor = min(self.warmup_count[i, j] / 5.0, 1.0)
                concentration = self.warmup_bucket_concentration[i, j]
                warmup_term = count_factor * concentration

                # Component 2: vouching term
                vouches = []
                for k in eligible_peers[i]:
                    if k == j:
                        continue
                    # Does k have a warmup-era opinion of j?
                    wb_kj = int(self.warmup_bucket[k, j])
                    if wb_kj >= 0:
                        # Bucket 0 (closest) -> score 1.0
                        # Bucket (n_buckets-1) (farthest) -> score 0
                        bucket_score = 1.0 - wb_kj / max(self.n_buckets - 1, 1)
                        # Weight vouch by i's trust in k
                        vouches.append(self.trust[i, k] * bucket_score)
                vouching_term = float(np.mean(vouches)) if vouches else 0.0

                # Component 3: embedding term (gated by maturity)
                if use_embedding:
                    # Closer in embedding space → higher trust contribution
                    embedding_term = max(0.0, 1.0 - emb_dists[j] / emb_max)
                else:
                    embedding_term = 0.0

                # Combine. Weight sum equals number of active components.
                if use_embedding:
                    total = (self.trust_w_warmup * warmup_term
                              + self.trust_w_vouch * vouching_term
                              + self.trust_w_embed * embedding_term)
                    w_sum = (self.trust_w_warmup + self.trust_w_vouch
                              + self.trust_w_embed)
                else:
                    total = (self.trust_w_warmup * warmup_term
                              + self.trust_w_vouch * vouching_term)
                    w_sum = self.trust_w_warmup + self.trust_w_vouch
                new_trust[i, j] = np.clip(total / max(w_sum, 1e-6), 0.0, 1.0)

        self.trust = new_trust

        # Diagnostic snapshot
        off_diag_mask = ~np.eye(self.N, dtype=bool)
        vals = self.trust[off_diag_mask]
        self.trust_snapshots.append({
            'round': rnd,
            'mean': float(vals.mean()),
            'q10': float(np.quantile(vals, 0.10)),
            'q90': float(np.quantile(vals, 0.90)),
            'high_frac': float((vals > self.trust_high_threshold).mean()),
            'case2_active_pairs': int(self.case2_active.sum()),
            'used_embedding': use_embedding,
        })

        if self.verbose and rnd % 50 == 0:
            print(f"[v3 rnd {rnd:3d}] trust mean={vals.mean():.3f}  "
                  f"high={vals.mean() > self.trust_high_threshold} "
                  f"case2_active={self.case2_active.sum()} "
                  f"embed={use_embedding}")

    # ------------------------------------------------------------------
    # Get trust weights for embedding training (one weight per evidence pair)
    # ------------------------------------------------------------------
    def get_trust_weights_for_evidence(self, evidence):
        """
        Given the evidence structure returned by gather_neighbor_ranklists_with_dists,
        compute parallel trust weights:  trust_weights[i] is a list of trust(i, j)
        values, one per evidence item (where j is the anchor for that item).

        Used by the modified update_embeddings_ladder_triplet.
        """
        trust_weights = []
        for i, items_i in enumerate(evidence):
            w_i = []
            for (j, _, _) in items_i:
                # If warmup not yet finalized, use 1.0 (back-compat behavior)
                if self._d_warmup_snapshot is None:
                    w_i.append(1.0)
                else:
                    # Floor at small positive to avoid zero contribution
                    w_i.append(max(float(self.trust[i, j]), 0.05))
            trust_weights.append(w_i)
        return trust_weights

    # ------------------------------------------------------------------
    # Affinity row for aggregation (applies Case-2 max rule)
    # ------------------------------------------------------------------
    def _affinity_row(self, i: int) -> np.ndarray:
        """
        Returns [N] affinity values for node i.
        For non-Case-2 pairs: d_live (from registry EMA).
        For Case-2 pairs: max(d_live, d_warmup).
        Self = 0.0. Never-seen = inf.
        """
        if self.freeze_ema and self._frozen_ema is not None:
            d_live = self._frozen_ema[i].copy()
        else:
            d_live = self.registry.get_ema_dist_row(i)

        if self._d_warmup_snapshot is None:
            return d_live

        d_warm = self._d_warmup_snapshot[i]
        result = d_live.copy()
        active_mask = self.case2_active[i]
        for j in range(self.N):
            if not active_mask[j]:
                continue
            # Use the farther of the two as the defensive affinity.
            # If d_warm is inf (never seen in warmup), Case 2 can't have triggered
            # (warmup_bucket would be -1, so high-trust impossible).
            if not np.isinf(d_warm[j]):
                if np.isinf(d_live[j]):
                    result[j] = d_warm[j]
                else:
                    result[j] = max(d_live[j], d_warm[j])
        return result

    # ------------------------------------------------------------------
    # Sampling targets — same 4-slot quota as v2
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
            # Slot B — top-7 by affinity (uses d_live with Case-2 override)
            D = self._affinity_row(node_i)
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
    # Aggregation — same softmax(-affinity / τ), but affinity is Case-2-aware
    # ------------------------------------------------------------------
    def run_topology_and_aggregate(self, rnd: int,
                                    node_states: torch.Tensor,
                                    E_list, ranked_neighbors,
                                    prev_ranked_neighbors):
        # Warmup: no aggregation
        if rnd < self.warmup_rounds:
            return node_states.clone()

        # Freeze on first post-warmup round if requested
        if self.freeze_ema and self._frozen_ema is None:
            self._frozen_ema = self.registry.snapshot_ema_dist()

        tau = self.tau(rnd)
        N, _ = node_states.shape
        W = torch.zeros(N, N, device=self.device, dtype=node_states.dtype)

        for i in range(N):
            D = self._affinity_row(i)   # uses Case-2 max rule
            sampled = ranked_neighbors[i].cpu().tolist()
            all_nodes = sampled + [i]

            d_vec = np.array([0.0 if j == i else
                                (float(D[j]) if not np.isinf(D[j])
                                 else 999.0)
                                for j in all_nodes], dtype=np.float32)

            logits = -d_vec / max(tau, 1e-8)
            logits -= logits.max()
            exp_w = np.exp(logits)
            w_np = exp_w / (exp_w.sum() + 1e-12)

            for idx, (j, w) in enumerate(zip(all_nodes, w_np)):
                W[i, j] = float(w)

        self.last_W = W.detach()

        if self.verbose and (rnd % 10 == 0 or rnd == self.warmup_rounds):
            dn = self.debug_node
            sw = W[dn, dn].item()
            top5 = torch.topk(W[dn], min(6, N)).indices.tolist()
            top5 = [x for x in top5 if x != dn][:5]
            n_case2 = int(self.case2_active[dn].sum())
            print(f"[v3 rnd {rnd:3d}] tau={tau:.3f}  "
                  f"Node {dn}: self_w={sw:.3f}  top5={top5}  "
                  f"case2_active(self)={n_case2}")

        return torch.mm(W, node_states).detach()

    # ------------------------------------------------------------------
    # Soft TP/FP (same as v2)
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
        result = dict(round=rnd, tp=tp, fp=fp, fn=fn, tn=tn,
                       avg_self_weight=avg_self,
                       tau=self.tau(rnd),
                       case2_total=int(self.case2_active.sum()),
                       trust_mean=float(self.trust[~np.eye(self.N, dtype=bool)].mean()))
        self.soft_metrics_log.append(result)
        return result

    def get_weight_heatmap(self) -> np.ndarray:
        return self.last_W.cpu().numpy() if self.last_W is not None else None

    # ------------------------------------------------------------------
    # Diagnostic exports
    # ------------------------------------------------------------------
    def get_trust_snapshot(self) -> np.ndarray:
        return self.trust.copy()

    def get_case2_state(self) -> dict:
        return {
            'active_pairs': int(self.case2_active.sum()),
            'events': list(self.case2_events),
        }