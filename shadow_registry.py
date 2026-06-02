"""
ShadowRegistry  (v3)
====================
Stores per-round rank and L2 distance observations, plus an EMA of distances.

The EMA distance D_ij is the primary signal used by PAGANv2 for aggregation.
It is an absolute quantity (not softmax-processed), so it is τ-independent
and can be converted to aggregation weights at any temperature at mix time.

Layout
------
shadow_rank  [N, N, R]  float32  rank of j in i's list at round r. NaN = not sampled.
shadow_dist  [N, N, R]  float32  raw L2 distance i->j at round r.  NaN = not sampled.
ema_dist     [N, N]     float32  EMA-smoothed L2 distance.
                                   0.0 for self (always).
                                   -1.0 = never observed (cold state, treated as inf).
dist_spread  [N, R]     float32  std-dev of L2 distances to sampled peers each round.

EMA update rule (each round, for each sampled peer j of node i):
    if first observation:  D_ij = d_ij
    else:                  D_ij = λ * D_ij + (1-λ) * d_ij
    D_ii = 0  always (self distance).
    Unsampled nodes: D_ij unchanged (remembered from last observation).

shadow_sw is retained as a diagnostic store (for plot_shadow compatibility)
but is no longer used for aggregation.
"""

import numpy as np
import torch


class ShadowRegistry:
    def __init__(self, num_nodes: int, num_rounds: int,
                 ema_lambda: float = 0.95,
                 softmax_tau: float = 1.0):   # kept for diagnostic sw only
        self.N          = num_nodes
        self.R          = num_rounds
        self.lam        = ema_lambda
        self.tau        = softmax_tau          # used for diagnostic shadow_sw only

        # Primary diagnostic stores  [N, N, R]
        # self.shadow_rank = np.full((num_nodes, num_nodes, num_rounds),
        #                             np.nan, dtype=np.float32)
        self.shadow_dist = np.full((num_nodes, num_nodes, num_rounds),
                                    np.nan, dtype=np.float32)
        # Diagnostic softmax weights (not used in aggregation)
        # self.shadow_sw   = np.full((num_nodes, num_nodes, num_rounds),
        #                             np.nan, dtype=np.float32)
        self.dist_spread = np.full((num_nodes, num_rounds),
                                    np.nan, dtype=np.float32)

        # EMA distance  [N, N]
        # -1.0 = sentinel meaning "never observed"
        # 0.0  = self (always)
        self.ema_dist = np.full((num_nodes, num_nodes), -1.0, dtype=np.float32)
        np.fill_diagonal(self.ema_dist, 0.0)

    # ------------------------------------------------------------------
    def record(self, rnd: int, ranked_neighbors: list, ranked_dists: list,
               lambda_matrix: np.ndarray = None):
        """
        Call once per round after rank_neighbors_by_model_distance.
        ranked_neighbors[i] : LongTensor sorted closest-first.
        ranked_dists[i]     : FloatTensor of squared-L2, aligned.

        Parameters
        ----------
        lambda_matrix : np.ndarray of shape [N, N], optional
            Per-pair EMA decay. If None, uses self.lam (back-compat).
            Used by PAGANv3 to apply asymmetric inertia.
        """
        if rnd >= self.R:
            return

        for i in range(self.N):
            nbrs = ranked_neighbors[i]
            dsts = ranked_dists[i]
            if nbrs.numel() == 0:
                continue

            nbrs_np = nbrs.cpu().numpy().astype(int)
            l2_np   = np.sqrt(np.maximum(
                dsts.cpu().numpy().astype(np.float32), 0.0))

            # ── rank and raw distance (diagnostic) ────────────────────
            # for rank_pos, (j, d) in enumerate(zip(nbrs_np, l2_np)):
            #     self.shadow_rank[i, j, rnd] = float(rank_pos)
            #     self.shadow_dist[i, j, rnd] = d

            # Raw distances only — rank and sw reconstructable in post-processing
            for j, d in zip(nbrs_np, l2_np):
                self.shadow_dist[i, j, rnd] = d

            # ── EMA distance update (per-pair lambda if provided) ─────
            for j, d in zip(nbrs_np, l2_np):
                lam_ij = float(lambda_matrix[i, j]) if lambda_matrix is not None else self.lam
                if self.ema_dist[i, j] < 0.0:     # first observation
                    self.ema_dist[i, j] = d
                else:
                    self.ema_dist[i, j] = (lam_ij * self.ema_dist[i, j]
                                           + (1.0 - lam_ij) * d)

            # # ── Diagnostic softmax weights (includes self at d=0) ──────
            # all_ids = np.append(nbrs_np, i)
            # all_l2  = np.append(l2_np,  0.0)
            # logits  = -all_l2 / max(self.tau, 1e-8)
            # logits -= logits.max()
            # exp_w   = np.exp(logits)
            # sw      = exp_w / (exp_w.sum() + 1e-12)
            # for j, w in zip(all_ids, sw):
            #     self.shadow_sw[i, j, rnd] = float(w)

            # ── Distance spread (diagnostic) ───────────────────────────
            if len(l2_np) >= 2:
                self.dist_spread[i, rnd] = float(np.std(l2_np))

    # ------------------------------------------------------------------
    # EMA distance accessors
    # ------------------------------------------------------------------
    def get_ema_dist(self, observer: int, target: int) -> float:
        """
        Returns EMA distance. Returns np.inf if target never observed.
        Returns 0.0 for self.
        """
        v = float(self.ema_dist[observer, target])
        return np.inf if v < 0.0 else v

    def get_ema_dist_row(self, observer: int) -> np.ndarray:
        """
        Returns [N] float32 array of EMA distances from observer to all nodes.
        Never-observed nodes have value np.inf. Self = 0.0.
        """
        row = self.ema_dist[observer].copy()
        row[row < 0.0] = np.inf    # sentinel → inf (will get near-zero weight)
        return row

    def snapshot_ema_dist(self) -> np.ndarray:
        """Returns a copy of the full [N, N] ema_dist matrix (for freezing)."""
        snap = self.ema_dist.copy()
        snap[snap < 0.0] = np.inf
        return snap

    # ------------------------------------------------------------------
    # Legacy affinity (window-based, for plot_shadow compatibility)
    # ------------------------------------------------------------------
    def affinity(self, observer: int, target: int,
                 end_round: int = None, window: int = 10) -> float:
        end   = (self.R - 1) if end_round is None else min(end_round, self.R - 1)
        start = max(0, end - window + 1)
        sw    = self.shadow_dist[observer, target, start:end + 1]
        valid = sw[~np.isnan(sw)]
        return float(np.mean(valid)) if len(valid) > 0 else np.nan

    def affinity_vector(self, observer: int,
                        end_round: int = None, window: int = 10) -> np.ndarray:
        end   = (self.R - 1) if end_round is None else min(end_round, self.R - 1)
        start = max(0, end - window + 1)
        dists = self.shadow_dist[observer, :, start:end + 1]   # [N, W]
        with np.errstate(invalid='ignore', divide='ignore'):
            aff = np.nanmean(1.0 / (1.0 + dists), axis=1)     # [N]
        return aff.astype(np.float32)


    # ------------------------------------------------------------------
    # Embedding evaluation (unchanged)
    # ------------------------------------------------------------------
    @staticmethod
    def _emb_ranks(E_list, observer: int) -> np.ndarray:
        Ei  = E_list[observer].detach().cpu().float().numpy()
        my  = Ei[observer]
        d   = np.linalg.norm(Ei - my, axis=1)
        d[observer] = np.inf
        return np.argsort(d).argsort().astype(np.int32)

    def compute_embedding_spearman(self, E_list, ranked_neighbors, k=10):
        from scipy.stats import spearmanr
        N = self.N
        per_rho    = np.full(N, np.nan, dtype=np.float32)
        per_recall = np.full(N, np.nan, dtype=np.float32)
        emb_rank_of_phys = np.full((N, k), np.nan, dtype=np.float32)
        for i in range(N):
            phys = ranked_neighbors[i]
            if phys.numel() < 2: continue
            phys_topk = phys[:k].cpu().numpy().astype(int)
            actual_k  = len(phys_topk)
            e_ranks   = self._emb_ranks(E_list, i)
            emb_r     = e_ranks[phys_topk]
            rho, _    = spearmanr(np.arange(actual_k), emb_r)
            if not np.isnan(rho): per_rho[i] = float(rho)
            emb_topk_set = set(np.where(e_ranks < k)[0])
            per_recall[i] = sum(1 for j in phys_topk if j in emb_topk_set) / actual_k
            for pos, j in enumerate(phys_topk):
                emb_rank_of_phys[i, pos] = float(e_ranks[j])
        return {
            'mean_rho':              float(np.nanmean(per_rho)),
            'per_node_rho':          per_rho,
            'mean_recall_at_k':      float(np.nanmean(per_recall)),
            'per_node_recall':       per_recall,
            'emb_rank_of_phys_topk': emb_rank_of_phys,
        }

    # ------------------------------------------------------------------
    # Simple accessors
    # ------------------------------------------------------------------
    def rank_history(self, observer, target): raise AttributeError("shadow_rank removed — use dist_history() instead.")
    def dist_history(self, observer, target): return self.shadow_dist[observer, target, :]
    def sw_history(self, observer, target):   raise AttributeError("shadow_sw removed — use affinity() instead.")
    def spread_history(self, observer):       return self.dist_spread[observer, :]
    def appearance_count(self, observer, target):
        return int(np.sum(~np.isnan(self.shadow_dist[observer, target, :])))

    # ------------------------------------------------------------------
    def save(self, path: str):
        np.savez_compressed(
            path,
            #shadow_rank = self.shadow_rank,
            shadow_dist = self.shadow_dist,
            #shadow_sw   = self.shadow_sw,
            dist_spread = self.dist_spread,
            ema_dist    = self.ema_dist,
            ema_lambda  = np.array([self.lam],  dtype=np.float32),
            tau         = np.array([self.tau],  dtype=np.float32),
        )
        print(f"[ShadowRegistry] Saved -> {path}.npz")

    @classmethod
    def load(cls, path: str) -> 'ShadowRegistry':
        p   = path if path.endswith('.npz') else path + '.npz'
        d   = np.load(p)
        N, _, R = d['shadow_dist'].shape
        reg = cls.__new__(cls)
        reg.N           = N
        reg.R           = R
        reg.lam         = float(d['ema_lambda'][0]) if 'ema_lambda' in d else 0.95
        reg.tau         = float(d['tau'][0])        if 'tau'        in d else 1.0
        #reg.shadow_rank = d['shadow_rank']
        reg.shadow_dist = d['shadow_dist']
        reg.shadow_sw   = d['shadow_sw']   if 'shadow_sw' in d else \
                          np.full((N, N, R), np.nan, dtype=np.float32)
        reg.dist_spread = d['dist_spread']
        reg.ema_dist    = d['ema_dist']    if 'ema_dist'  in d else \
                          np.full((N, N), -1.0, dtype=np.float32)
        np.fill_diagonal(reg.ema_dist, 0.0)
        return reg