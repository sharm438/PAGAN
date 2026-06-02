"""
pagan_v1.py
===========
PAGANv1: Warmup-anchored global affinity with embedding-guided discovery.

Core design
-----------
v2 had two structural problems:
  1. Affinity was k-relative — softmax over k sampled nodes, not all N.
     The denominator was biased toward whoever happened to be sampled.
  2. EMA freshness was non-uniform — recently-sampled nodes had current
     estimates; rarely-sampled nodes had stale ones. Same λ for all was unfair.

v1 fixes both with a principled warmup anchor:

WARMUP (rounds 0 → confirm_start):
  Pure uniform random sampling — no bias, maximum coverage.
  Tent-weighted distance tracking: exponentially increasing weight toward
  round W so late-warmup observations (more differentiated models) dominate.
  Embeddings trained with configurable triplet weighting scheme.
  Quality monitored every 5 rounds; plateau detection flags optimal warmup length.

CONFIRMATION (rounds confirm_start → W):
  Targeted sampling: confirm best tentative friends, reduce uncertain pairs,
  fill coverage gaps. Still within warmup (no aggregation).

WARMUP FINALIZATION (at round W):
  Compute tent-weighted d_tent[N,N] and confidence[N,N].
  Scale embedding distances to match model distance scale (calibrated from
  high-confidence pairs).
  Combined score: eff_conf × d_tent + (1-eff_conf) × scaled_D_emb.
  Global W_base = softmax(-score / τ) over ALL N nodes — fair, unbiased.
  Warmup rank order computed for vouching.

POST-WARMUP:
  Sampling:
    Slot A (K_SLOT_A=5)  — top W_base: warmup friends, stable
    Slot B (K_SLOT_B=7)  — top D_emb: embedding discovery of new nodes
    Slot C (K_SLOT_C=5)  — low-confidence pairs: targeted exploration
    Slot D (remainder)   — uniform random fill
  Aggregation:
    W_base updated each round via current embeddings for low-conf pairs.
    High-confidence pairs anchored to frozen d_tent.
    Sampled-subset weights = slice of W_base, renormalised.
  Vouching:
    Low-confidence slot B arrivals checked: do j's top-M neighbours
    overlap with i's warmup friends? vouch = quality × coverage.
    Vouch score updates eff_conf, which feeds score and W_base.
  Embeddings:
    Continue training on clean, warmup-anchored aggregation signal.
    Protected from contamination because W_base suppresses untrusted nodes.

MONITORING:
  tp@5/10/15 and intra/inter separation ratio every 5 rounds during warmup.
  Plateau detection: flag round where tp@5 stops improving (>0.01 in 3 evals).
  Scout events: log when embedding discovers a low-warmup-confidence peer
  that passes vouching. Accumulated count reported at run end.

Parameters
----------
  warmup_rounds     int    25      Warmup duration. No aggregation.
  confirm_start     int    W-5     Round to switch from random to targeted sampling.
  alpha_fresh       float  0.1     Tent weight decay rate toward W (warmup side).
  beta_fresh        float  0.1     Tent weight decay rate away from W (post-warmup).
  count_threshold   int    3       Observation count for full confidence.
  vouch_M           int    5       Top-M of j's neighbours checked for vouching.
  low_conf_thresh   float  0.4     Below this → pair needs vouching.
  vouch_threshold   float  0.3     Vouch score needed to lift eff_conf.
  triplet_scheme    str    'flat'  'flat' | 'inv_rank' | 'inv_sqrt_rank'
  tau_0 / tau_min   float          Temperature schedule (same as v2).
  tau_half_life     float
"""

import math
import random as _random
import numpy as np
import torch
from collections import defaultdict

from shadow_registry import ShadowRegistry


class PAGANv1:

    # Post-warmup slot quotas
    K_SLOT_A = 5    # top W_base — warmup friends
    K_SLOT_B = 7    # top D_emb  — embedding discovery
    K_SLOT_C = 5    # low-confidence — targeted exploration
    # Slot D fills remainder

    # Confirmation phase slot quotas
    K_CONF_A = 5    # best seen friends (top d_tent)
    K_CONF_B = 5    # most uncertain seen pairs (obs_count 1–2)
    K_CONF_C = 5    # never-observed nodes (coverage gaps)
    # Slot D fills remainder

    def __init__(self,
                 num_nodes: int,
                 device,
                 total_rounds:      int   = 500,
                 warmup_rounds:     int   = 25,
                 # Temperature schedule (same as v2)
                 tau_0:             float = 2.0,
                 tau_min:           float = 0.3,
                 tau_half_life:     float = 200.0,
                 softmax_tau:       float = 1.0,  # shadow registry diagnostic only
                 # Tent weighting
                 alpha_fresh:       float = 0.1,  # recency weight toward W
                 beta_fresh:        float = 0.1,  # decay weight away from W
                 # Confidence
                 count_threshold:   int   = 3,    # obs count for full confidence
                 # Confirmation phase
                 confirm_start:     int   = None, # default: warmup_rounds - 5
                 # Vouching
                 vouch_M:           int   = 5,
                 low_conf_threshold: float = 0.4,
                 vouch_threshold:   float = 0.3,
                 # Triplet loss weighting
                 triplet_weight_scheme: str = 'flat',
                 # Diagnostics
                 node_to_cluster:   np.ndarray = None,
                 eff_weight_thresh: float = 0.02,
                 debug_node:        int   = 0,
                 verbose:           bool  = True):

        # ── Core ─────────────────────────────────────────────────────
        self.N               = num_nodes
        self.device          = device
        self.total_rounds    = total_rounds
        self.warmup_rounds   = warmup_rounds
        self.tau_0           = tau_0
        self.tau_min         = tau_min
        self.tau_half_life   = tau_half_life
        self.alpha_fresh     = alpha_fresh
        self.beta_fresh      = beta_fresh
        self.count_threshold = count_threshold
        self.confirm_start   = (confirm_start if confirm_start is not None
                                else max(0, warmup_rounds - 5))
        self.vouch_M             = vouch_M
        self.low_conf_threshold  = low_conf_threshold
        self.vouch_threshold     = vouch_threshold
        self.triplet_weight_scheme = triplet_weight_scheme
        self.node_to_cluster     = node_to_cluster
        self.eff_thresh          = eff_weight_thresh
        self.debug_node          = debug_node
        self.verbose             = verbose

        # ── Tent-weighted distance tracking [N, N] ────────────────────
        # Incremental running sums — cheap O(k) update per round
        self.obs_wsum  = np.zeros((num_nodes, num_nodes), dtype=np.float64)
        self.obs_wgt   = np.zeros((num_nodes, num_nodes), dtype=np.float64)
        self.obs_count = np.zeros((num_nodes, num_nodes), dtype=np.int32)
        self.obs_count_warmup = np.zeros((num_nodes, num_nodes), dtype=np.int32)
        np.fill_diagonal(self.obs_wgt, 1.0)   # self always has weight

        # ── Warmup anchors (computed at round W, then frozen) ──────────
        self.d_tent = np.full((num_nodes, num_nodes), np.inf, dtype=np.float32)
        np.fill_diagonal(self.d_tent, 0.0)
        self.confidence = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        np.fill_diagonal(self.confidence, 1.0)

        # ── Effective confidence (confidence lifted by vouching) ───────
        self.eff_conf = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        np.fill_diagonal(self.eff_conf, 1.0)

        # ── Warmup rank ordering (for vouching lookup) ─────────────────
        # warmup_rank_order[i] : list of nodes sorted by d_tent[i] asc
        # warmup_rank_dict[i]  : {j: rank}  O(1) lookup
        self.warmup_rank_order: list = [[] for _ in range(num_nodes)]
        self.warmup_rank_dict:  list = [{} for _ in range(num_nodes)]

        self.warmup_top_Q_sets: list = [set() for _ in range(num_nodes)]
        self.warmup_top_Q: int = max(5, round(num_nodes * 0.15))  # top 15% of N

        self.last_seen = np.full((num_nodes, num_nodes), -1, dtype=np.int32)
        np.fill_diagonal(self.last_seen, 0)

        # ── Embedding distance scale (calibrated at warmup end) ────────
        self.emb_scale = 1.0   # scalar: scaled D_emb ≈ d_tent scale

        # ── Global score and aggregation weights ───────────────────────
        self.score  = np.full((num_nodes, num_nodes), np.inf,  dtype=np.float32)
        np.fill_diagonal(self.score, 0.0)
        self.W_base = np.full((num_nodes, num_nodes),
                              1.0 / num_nodes, dtype=np.float32)

        self._warmup_finalised = False

        # ── Shadow registry (raw dist/rank store for plotting tools) ───
        self.registry = ShadowRegistry(num_nodes, total_rounds,
                                       ema_lambda=0.95,
                                       softmax_tau=softmax_tau)

        # ── Monitoring ────────────────────────────────────────────────
        self.emb_quality_log:  list = []   # {round, tp@5, tp@10, tp@15, sep_ratio,...}
        self.plateau_round:    int  = -1   # round where tp@5 plateaued (-1 = not yet)
        self._tp5_history:     list = []   # rolling tp@5 values for plateau detection
        self.scout_events:     list = []   # {round, node_i, node_j, warmup_conf, vouch}
        self.scout_count:      int  = 0    # total scout events

        # ── Output ────────────────────────────────────────────────────
        self.last_W = None
        self.soft_metrics_log: list = []

        if verbose:
            print(f"[PAGANv1] init: N={num_nodes}  W={warmup_rounds}  "
                  f"confirm_start={self.confirm_start}\n"
                  f"  tent: α={alpha_fresh}  β={beta_fresh}  "
                  f"count_thresh={count_threshold}\n"
                  f"  vouch: M={vouch_M}  low_conf_thresh={low_conf_threshold}  "
                  f"vouch_thresh={vouch_threshold}\n"
                  f"  τ: {tau_0}→{tau_min} (hl={tau_half_life})  "
                  f"triplet='{triplet_weight_scheme}'")

    # ------------------------------------------------------------------
    # Temperature schedule (identical to v2)
    # ------------------------------------------------------------------
    def tau(self, rnd: int) -> float:
        t   = max(0, rnd - self.warmup_rounds)
        lam = math.log(
            max((self.tau_0 - self.tau_min) / max(self.tau_min, 1e-8),
                1.0 + 1e-8)
        ) / max(self.tau_half_life, 1.0)
        return max(self.tau_min, self.tau_0 * math.exp(-lam * t))

    # ------------------------------------------------------------------
    # Tent weight for a given round
    # ------------------------------------------------------------------
    def _tent_weight(self, rnd: int) -> float:
        """
        w(t) = exp(α(t-W))   for t < W   (increases toward W)
        w(t) = exp(-β(t-W))  for t >= W  (decreases from W)
        Peak weight = 1.0 at t = W.
        """
        W = self.warmup_rounds
        if rnd < W:
            return math.exp(self.alpha_fresh * (rnd - W))
        else:
            return math.exp(-self.beta_fresh * (rnd - W))

    # ------------------------------------------------------------------
    # Record — call every round after rank_neighbors_by_model_distance
    # ------------------------------------------------------------------
    def record(self, rnd: int, ranked_neighbors: list, ranked_dists: list,
               E_list=None):
        """
        Updates:
          - shadow registry (diagnostic storage)
          - tent-weighted distance sums
          - warmup finalization on first post-warmup round
          - W_base (post-warmup, using current embeddings)
        """
        # Shadow registry (diagnostic plots)
        self.registry.record(rnd, ranked_neighbors, ranked_dists)

        # Tent-weighted distance update
        w_tent = self._tent_weight(rnd)
        for i in range(self.N):
            nbrs = ranked_neighbors[i]
            dsts = ranked_dists[i]
            if nbrs.numel() == 0:
                continue
            nbrs_np = nbrs.cpu().numpy().astype(int)
            l2_np   = np.sqrt(np.maximum(
                dsts.cpu().numpy().astype(np.float32), 0.0))
            for j, d in zip(nbrs_np, l2_np):
                self.obs_wsum[i, j]  += w_tent * d
                self.obs_wgt[i, j]   += w_tent
                self.obs_count[i, j] += 1
                self.last_seen[i, j]  = rnd    
                if rnd < self.warmup_rounds:
                    self.obs_count_warmup[i, j] += 1


        # Finalise warmup on first post-warmup round
        if rnd == self.warmup_rounds - 1 and not self._warmup_finalised:
            if E_list is not None:
                self._finalize_warmup(E_list)
            else:
                print("[PAGANv1] WARNING: E_list not passed to record() at "
                      "warmup finalization — W_base will be distance-only.")
                self._finalize_warmup(None)

        # Post-warmup: refresh W_base with current embeddings each round
        if rnd >= self.warmup_rounds and self._warmup_finalised and E_list is not None:
            self._update_W_base(E_list, rnd)

    # ------------------------------------------------------------------
    # Warmup finalization (called once at round W)
    # ------------------------------------------------------------------
    def _finalize_warmup(self, E_list):
        N = self.N
        if self.verbose:
            print(f"[PAGANv1] Finalising warmup state...")

        # ── Tent-weighted distance ─────────────────────────────────────
        with np.errstate(invalid='ignore', divide='ignore'):
            d_raw = np.where(self.obs_wgt > 0,
                             self.obs_wsum / self.obs_wgt,
                             np.inf).astype(np.float32)
        np.fill_diagonal(d_raw, 0.0)
        self.d_tent = d_raw

        # ── Confidence ────────────────────────────────────────────────
        conf = np.clip(self.obs_count_warmup / max(self.count_threshold, 1),
                       0.0, 1.0).astype(np.float32)
        np.fill_diagonal(conf, 1.0)
        self.confidence = conf
        self.eff_conf   = conf.copy()

        # ── Warmup rank order (for vouching) ──────────────────────────
        for i in range(N):
            row = self.d_tent[i].copy()
            row[i] = np.inf
            order = np.argsort(row).tolist()
            self.warmup_rank_order[i] = order
            self.warmup_rank_dict[i]  = {int(j): r for r, j in enumerate(order)}

        # Top-Q set for middle-ground vouching
        # A valid voucher must be in i's warmup top-Q AND currently close
        Q = self.warmup_top_Q
        for i in range(N):
            self.warmup_top_Q_sets[i] = {
                int(self.warmup_rank_order[i][r])
                for r in range(min(Q, len(self.warmup_rank_order[i])))
                if np.isfinite(self.d_tent[i, self.warmup_rank_order[i][r]])
            }

        # ── Embedding scale calibration ───────────────────────────────
        if E_list is not None:
            D_emb = self._compute_emb_dist_matrix(E_list)
            high_conf_mask = (conf > 0.7) & ~np.eye(N, dtype=bool)
            if high_conf_mask.sum() > 0:
                dt_vals  = self.d_tent[high_conf_mask]
                de_vals  = D_emb[high_conf_mask]
                valid    = np.isfinite(dt_vals) & (de_vals > 1e-8)
                if valid.sum() > 0:
                    self.emb_scale = float(
                        np.median(dt_vals[valid]) /
                        np.median(de_vals[valid]))
        else:
            D_emb = np.full((N, N), np.inf, dtype=np.float32)
            np.fill_diagonal(D_emb, 0.0)

        if self.verbose:
            seen     = (self.obs_count > 0).sum() / (N * (N - 1))
            avg_conf = conf[~np.eye(N, dtype=bool)].mean()
            print(f"  coverage={seen:.2%}  avg_conf={avg_conf:.3f}  "
                  f"emb_scale={self.emb_scale:.3f}")

        # Initial W_base
        self._update_W_base(E_list, rnd=self.warmup_rounds)
        self._warmup_finalised = True

    # ------------------------------------------------------------------
    # Embedding distance matrix [N, N] — each row from node i's perspective
    # ------------------------------------------------------------------
    def _compute_emb_dist_matrix(self, E_list) -> np.ndarray:
        """
        D[i,j] = ||E_list[i][i] - E_list[i][j]||
        Symmetrised: (D_ij + D_ji) / 2
        """
        N = self.N
        D = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            Ei       = E_list[i].detach().cpu().float().numpy()
            self_emb = Ei[i]
            d_row    = np.linalg.norm(Ei - self_emb, axis=1).astype(np.float32)
            d_row[i] = 0.0
            D[i]     = d_row
        D = (D + D.T) / 2.0
        np.fill_diagonal(D, 0.0)
        return D

    # ------------------------------------------------------------------
    # Update W_base — called every post-warmup round
    # ------------------------------------------------------------------
    def _update_W_base(self, E_list, rnd: int):
        """
        score[i,j] = eff_conf[i,j] × d_tent[i,j]
                   + (1 - eff_conf[i,j]) × emb_scale × D_emb[i,j]
        W_base[i,:] = softmax(-score[i,:] / τ)   over all N nodes
        Self gets score=0 → always highest logit → self_w autoscales with τ.
        """
        with np.errstate(invalid='ignore', divide='ignore'):
            d_tent_live = np.where(self.obs_wgt > 0,
                                self.obs_wsum / self.obs_wgt,
                                np.inf).astype(np.float32)
        np.fill_diagonal(d_tent_live, 0.0)
        
        N   = self.N
        tau = self.tau(rnd)

        if E_list is not None:
            D_emb        = self._compute_emb_dist_matrix(E_list)
            D_emb_scaled = (D_emb * self.emb_scale).astype(np.float32)
        else:
            D_emb_scaled = np.full((N, N), np.inf, dtype=np.float32)
            np.fill_diagonal(D_emb_scaled, 0.0)

        # Blend: high conf → use d_tent; low conf → use embedding
        ec         = self.eff_conf
        d_tent_safe = np.where(np.isfinite(d_tent_live),
                               d_tent_live, D_emb_scaled)
        self.score = (ec * d_tent_safe
                      + (1.0 - ec) * D_emb_scaled).astype(np.float32)
        np.fill_diagonal(self.score, 0.0)

        # Global softmax per row
        W = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            logits  = -self.score[i] / max(tau, 1e-8)
            logits -= logits.max()
            exp_w   = np.exp(logits)
            W[i]    = exp_w / (exp_w.sum() + 1e-12)
        self.W_base = W

    # ------------------------------------------------------------------
    # Vouching — quality × coverage formula
    # ------------------------------------------------------------------
    def _vouch_score(self, i: int, j: int, j_ranked_nbrs,
                    current_ranked_i=None) -> float:
        """
        Vouch score = quality × coverage.
        Valid voucher m must satisfy ALL of:
        1. i observed m during warmup
        2. m was in i's warmup top-Q (genuine warmup friend)
        3. m is currently in i's ranked list (still close now)
        Condition 3 prevents contamination chains via drifted nodes.
        Unseen nodes excluded from quality mean, reduce coverage instead.
        """
        if j_ranked_nbrs is None or j_ranked_nbrs.numel() == 0:
            return 0.0

        j_topM    = j_ranked_nbrs[:self.vouch_M].cpu().tolist()
        rank_dict = self.warmup_rank_dict[i]
        top_Q     = self.warmup_top_Q_sets[i]
        N         = self.N

        # Build current set for condition 3
        current_set = set()
        if current_ranked_i is not None:
            current_set = set(current_ranked_i.cpu().tolist())

        # Apply all three conditions
        if current_ranked_i is not None:
            seen = [(m, rank_dict[m]) for m in j_topM
                    if m in rank_dict       # cond 1: seen in warmup
                    and m in top_Q          # cond 2: was warmup friend
                    and m in current_set]   # cond 3: still close now
        else:
            # Fallback: conditions 1+2 only (no current neighbours available)
            seen = [(m, rank_dict[m]) for m in j_topM
                    if m in rank_dict and m in top_Q]

        if not seen:
            return 0.0

        mean_rank = float(np.mean([r for _, r in seen]))
        quality   = 1.0 - mean_rank / max(N - 1, 1)
        coverage  = len(seen) / max(len(j_topM), 1)
        return float(np.clip(quality * coverage, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Vouching update — called each post-warmup round for sampled pairs
    # ------------------------------------------------------------------
    def _update_vouching(self, rnd: int, ranked_neighbors):
        """
        For each sampled low-confidence pair (i,j):
          - compute vouch score
          - update eff_conf
          - log scout events
        """
        for i in range(self.N):
            nbrs = ranked_neighbors[i]
            if nbrs.numel() == 0:
                continue
            for j_t in nbrs:
                j = int(j_t)
                if self.confidence[i, j] >= self.low_conf_threshold:
                    continue   # high warmup confidence — no vouching needed

                v = self._vouch_score(i, j, ranked_neighbors[j], current_ranked_i=ranked_neighbors[i])
                new_eff = max(float(self.confidence[i, j]), v)
                #new_eff = max(float(self.eff_conf[i, j]), v)

                # Scout event: low warmup confidence, passes vouching
                if (v >= self.vouch_threshold
                        and self.eff_conf[i, j] < self.vouch_threshold):
                    self.scout_events.append({
                        'round':       rnd,
                        'node_i':      i,
                        'node_j':      j,
                        'warmup_conf': float(self.confidence[i, j]),
                        'vouch_score': float(v),
                    })
                    self.scout_count += 1

                self.eff_conf[i, j] = new_eff

    # ------------------------------------------------------------------
    # Sampling targets
    # ------------------------------------------------------------------
    def get_sampling_targets(self, rnd: int, node_i: int, k_total: int,
                              physical_rank_list: list, E_list_i) -> list:
        N = self.N

        # ── Warmup: pure uniform random (0 → confirm_start) ───────────
        if rnd < self.confirm_start:
            pool = [x for x in range(N) if x != node_i]
            _random.shuffle(pool)
            return pool[:k_total]

        # ── Confirmation phase (confirm_start → warmup_rounds) ─────────
        if rnd < self.warmup_rounds:
            return self._confirmation_targets(node_i, k_total)

        # ── Post-warmup ────────────────────────────────────────────────
        return self._postwarmup_targets(node_i, k_total, E_list_i)

    def _confirmation_targets(self, node_i: int, k_total: int) -> list:
        """
        Slot A: best seen friends (top d_tent).
        Slot B: most uncertain seen pairs (obs_count 1–2).
        Slot C: never-observed nodes (coverage gaps).
        Slot D: random fill.
        """
        N = self.N
        targets = set()

        # Slot A — confirm best known friends
        d_row = self.d_tent[node_i].copy()
        d_row[node_i] = np.inf
        order = np.argsort(d_row)
        added = 0
        for j in order:
            if int(j) != node_i and np.isfinite(d_row[j]):
                targets.add(int(j))
                added += 1
            if added >= self.K_CONF_A:
                break

        # Slot B — uncertain seen pairs (fewest observations)
        uncertain = sorted(
            [(j, self.obs_count[node_i, j])
             for j in range(N)
             if j != node_i
             and 0 < self.obs_count[node_i, j] <= 2
             and j not in targets],
            key=lambda x: x[1]
        )
        for j, _ in uncertain[:self.K_CONF_B]:
            targets.add(j)

        # Slot C — coverage gaps (never observed)
        unseen = [j for j in range(N)
                  if j != node_i
                  and self.obs_count[node_i, j] == 0
                  and j not in targets]
        _random.shuffle(unseen)
        for j in unseen[:self.K_CONF_C]:
            targets.add(j)

        # Slot D — random fill
        pool = [x for x in range(N) if x != node_i and x not in targets]
        _random.shuffle(pool)
        for c in pool:
            if len(targets) >= k_total:
                break
            targets.add(c)

        return list(targets)

    def _postwarmup_targets(self, node_i: int, k_total: int,
                             E_list_i) -> list:
        """
        Slot A: top W_base (warmup friends).
        Slot B: top D_emb (embedding discovery).
        Slot C: low-confidence pairs (targeted exploration).
        Slot D: random fill.
        """
        N = self.N
        targets = set()

        # Slot A — highest W_base peers
        wb_row = self.W_base[node_i].copy()
        wb_row[node_i] = -1.0
        top_wb = np.argsort(-wb_row)
        added = 0
        for j in top_wb:
            if int(j) != node_i:
                targets.add(int(j))
                added += 1
            if added >= self.K_SLOT_A:
                break

        # Slot B — highest embedding affinity (discovery)
        if E_list_i is not None:
            my_emb     = E_list_i[node_i].unsqueeze(0)
            emb_d      = torch.norm(E_list_i - my_emb, dim=1)
            emb_sorted = torch.argsort(emb_d).tolist()
            added = 0
            for j in emb_sorted:
                if j != node_i and j not in targets:
                    targets.add(j)
                    added += 1
                if added >= self.K_SLOT_B or len(targets) >= k_total:
                    break

        # Slot C — low-confidence pairs (explore uncertain)
        low_conf = sorted(
            [(j, float(self.eff_conf[node_i, j]))
             for j in range(N)
             if j != node_i
             and self.eff_conf[node_i, j] < self.low_conf_threshold
             and j not in targets],
            key=lambda x: x[1]   # lowest confidence first
        )
        for j, _ in low_conf[:self.K_SLOT_C]:
            targets.add(j)

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
                                   E_list, ranked_neighbors,
                                   prev_ranked_neighbors):
        # Warmup: pure isolation
        if rnd < self.warmup_rounds:
            if rnd % 5 == 0:
                spread = self.registry.dist_spread[self.debug_node, rnd]
                s = f"{spread:.4f}" if not np.isnan(spread) else "n/a"
                print(f"[v1 warmup rnd {rnd:3d}] "
                      f"Node {self.debug_node} dist_spread={s}")
            return node_states.clone()

        # Update vouching for sampled low-confidence pairs
        self._update_vouching(rnd, ranked_neighbors)

        # Build W matrix from W_base: slice sampled + self, renormalise
        tau = self.tau(rnd)
        N, _ = node_states.shape
        W    = torch.zeros(N, N, device=self.device, dtype=node_states.dtype)

        for i in range(N):
            sampled   = ranked_neighbors[i].cpu().tolist()
            all_nodes = sampled + [i]

            d_vec = np.array([
                0.0 if j == i else
                (float(self.score[i, j]) if np.isfinite(self.score[i, j]) else 999.0)
                for j in all_nodes
            ], dtype=np.float32)

            logits  = -d_vec / max(tau, 1e-8)
            logits -= logits.max()
            exp_w   = np.exp(logits)
            w_np    = exp_w / (exp_w.sum() + 1e-12)

            for j, w in zip(all_nodes, w_np):
                W[i, j] = float(w)


        self.last_W = W.detach()

        if self.verbose and (rnd % 10 == 0 or rnd == self.warmup_rounds):
            dn   = self.debug_node
            sw   = W[dn, dn].item()
            top5_idx = torch.topk(W[dn], min(6, N)).indices.tolist()
            top5 = [x for x in top5_idx if x != dn][:5]
            tau  = self.tau(rnd)
            print(f"[v1 rnd {rnd:3d}] τ={tau:.3f}  "
                  f"Node {dn}: self_w={sw:.3f}  top5={top5}  "
                  f"scouts={self.scout_count}")

        return torch.mm(W, node_states).detach()

    # ------------------------------------------------------------------
    # Embedding quality monitoring (call every 5 rounds during warmup)
    # ------------------------------------------------------------------
    def monitor_embeddings(self, rnd: int, E_list,
                            node_to_cluster: np.ndarray) -> dict:
        """
        Computes:
          - tp@5, tp@10, tp@15
          - mean intra-cluster and inter-cluster embedding distances
          - separation ratio = inter_mean / intra_mean
        Plateau detection: if tp@5 hasn't improved >0.01 in last 3 evals,
        flags self.plateau_round.
        """
        N   = self.N
        ntc = node_to_cluster

        tp5 = tp10 = tp15 = 0.0
        intra_list = []
        inter_list = []

        for i in range(N):
            Ei       = E_list[i].detach().cpu().float().numpy()
            self_emb = Ei[i]
            d_row    = np.linalg.norm(Ei - self_emb, axis=1)
            d_row[i] = np.inf
            order    = np.argsort(d_row)

            my_c  = ntc[i]
            tp5  += sum(1 for j in order[:5]  if ntc[j] == my_c) / 5
            tp10 += sum(1 for j in order[:10] if ntc[j] == my_c) / 10
            tp15 += sum(1 for j in order[:15] if ntc[j] == my_c) / 15

            for j in range(N):
                if j == i or np.isinf(d_row[j]):
                    continue
                if ntc[j] == my_c:
                    intra_list.append(float(d_row[j]))
                else:
                    inter_list.append(float(d_row[j]))

        tp5  /= N
        tp10 /= N
        tp15 /= N
        intra_mean = float(np.mean(intra_list)) if intra_list else 0.0
        inter_mean = float(np.mean(inter_list)) if inter_list else 0.0
        sep_ratio  = inter_mean / max(intra_mean, 1e-8)

        entry = dict(round=rnd, tp_at_5=tp5, tp_at_10=tp10, tp_at_15=tp15,
                     intra_mean=intra_mean, inter_mean=inter_mean,
                     sep_ratio=sep_ratio)
        self.emb_quality_log.append(entry)

        if self.verbose:
            print(f"  [v1 emb rnd {rnd:3d}] "
                  f"tp@5={tp5:.3f}  tp@10={tp10:.3f}  tp@15={tp15:.3f}  "
                  f"sep={sep_ratio:.2f}  "
                  f"intra={intra_mean:.4f}  inter={inter_mean:.4f}")

        # Plateau detection
        self._tp5_history.append(tp5)
        if (self.plateau_round < 0
                and len(self._tp5_history) >= 3
                and tp5 > 0.5):
            recent      = self._tp5_history[-3:]
            improvement = max(recent) - min(recent)
            if improvement < 0.01:
                self.plateau_round = rnd
                if self.verbose:
                    print(f"  [v1 emb] *** PLATEAU at round {rnd} "
                          f"(tp@5={tp5:.3f}) — warmup may not need "
                          f"to run longer ***")

        return entry

    # ------------------------------------------------------------------
    # Triplet loss evidence weights
    # ------------------------------------------------------------------
    def get_evidence_weights(self, evidence: list) -> list:
        """
        Returns parallel weight lists for the triplet loss.
        All schemes weight evidence item at rank_idx:
          flat          : 1.0
          inv_rank      : 1 / (rank_idx + 1)
          inv_sqrt_rank : 1 / sqrt(rank_idx + 1)
        """
        scheme  = self.triplet_weight_scheme
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
    # Soft TP/FP metrics
    # ------------------------------------------------------------------
    def compute_soft_metrics(self, rnd: int) -> dict:
        if self.last_W is None or self.node_to_cluster is None:
            return {}
        W  = self.last_W.cpu().numpy()
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
        result = dict(
            round=rnd, tp=tp, fp=fp, fn=fn, tn=tn,
            avg_self_weight=avg_self,
            tau=self.tau(rnd),
            scout_count=self.scout_count,
            eff_conf_vouched = int((self.eff_conf[~np.eye(self.N, dtype=bool)] >= self.vouch_threshold).sum())
        )
        self.soft_metrics_log.append(result)
        return result

    def get_weight_heatmap(self) -> np.ndarray:
        return self.last_W.cpu().numpy() if self.last_W is not None else None

    def get_summary(self) -> dict:
        """End-of-run summary: plateau round, scout count, embedding log."""
        return {
            'plateau_round': self.plateau_round,
            'scout_count':   self.scout_count,
            'emb_quality':   self.emb_quality_log,
            'scout_events':  self.scout_events[:100],  # first 100
        }

    def compute_scouts(self, min_lift: float = 0.1) -> dict:
        """
        Reports:
        warmup_confirmed : true-friend pairs with warmup_conf >= low_conf_threshold
        post_discovered  : true-friend pairs with low warmup_conf but
                            meaningfully lifted by vouching (lift > min_lift)
        missed           : true-friend pairs with low warmup_conf and no discovery
        Only counts unique directed pairs. No recounting.
        """
        N   = self.N
        ntc = self.node_to_cluster

        warmup_confirmed = []
        post_discovered  = []
        missed           = []
        non_friend_lifted = []   # lifted but not a true friend — interesting to log

        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                wc  = float(self.confidence[i, j])
                ec  = float(self.eff_conf[i, j])
                lift = ec - wc
                is_friend = (ntc is not None
                            and ntc[i] == ntc[j])

                if is_friend:
                    if wc >= self.low_conf_threshold:
                        warmup_confirmed.append({'node_i': i, 'node_j': j,
                                                'warmup_conf': wc})
                    elif lift > min_lift:
                        post_discovered.append({'node_i': i, 'node_j': j,
                                                'warmup_conf': wc,
                                                'final_eff_conf': ec,
                                                'lift': lift})
                    else:
                        missed.append({'node_i': i, 'node_j': j,
                                    'warmup_conf': wc,
                                    'final_eff_conf': ec})
                else:
                    # Non-friend that was vouched — potential false positive
                    if lift > min_lift:
                        non_friend_lifted.append({'node_i': i, 'node_j': j,
                                                'warmup_conf': wc,
                                                'final_eff_conf': ec,
                                                'lift': lift})

        return {
            'warmup_confirmed_count':  len(warmup_confirmed),
            'post_discovered_count':   len(post_discovered),
            'missed_count':            len(missed),
            'non_friend_lifted_count': len(non_friend_lifted),
            'post_discovered':         post_discovered[:50],
            'missed':                  missed[:50],
            'non_friend_lifted':       non_friend_lifted[:20],
        }
