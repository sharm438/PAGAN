"""
fedspd.py
=========
FedSPD baseline implementation for the PAGAN codebase.

Reference: Lin et al., "FedSPD: A Soft-clustering Approach for
Personalized Decentralized Federated Learning", UAI 2025.

Key differences from the original paper's code:
  - Uses our data distribution (Dirichlet / patho) instead of rotation-based
  - Fixed k-regular-random graph (k=--k arg) instead of ER graph with N=25
  - No fine-tuning phase (excluded for fair comparison with PAGAN)
  - Batched re-clustering: one full forward pass per round, no DataLoader overhead
  - Model pool (S models) reused across N nodes in re-clustering for efficiency

New args added in main.py (all prefixed fedspd_):
  --fedspd_S      int   Cluster centers per node (default: 2)
  --fedspd_ckc    int   Label-switch alignment every N rounds (default: 10)
  --fedspd_init   str   'fedspd_style' | 'round_robin' (default: fedspd_style)

'fedspd_style' init mirrors the paper's code exactly:
  - Models: each (node, cluster) independently random-initialized
  - Data:   biased positional-half split via 'portion' formula

'round_robin' init is the cleaner EM bootstrap:
  - Models: all S centers initialized identically (same random weights)
  - Data:   sample j → cluster (j % S), breaking symmetry via gradient only
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

import utils
import models as models_module
import train_node


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _load_vec_into_model(vec, model):
    """In-place load of a flat parameter vector into an existing model."""
    idx = 0
    with torch.no_grad():
        for p in model.parameters():
            sz = p.numel()
            p.data.copy_(vec[idx: idx + sz].view(p.shape))
            idx += sz


# ---------------------------------------------------------------------------
# FedSPD class
# ---------------------------------------------------------------------------

class FedSPD:
    """
    FedSPD protocol.

    Per-node state:
      centers[i, s, :]  – flat parameter vector for cluster s at node i  [N, S, D]
      cluster_idx[i][s] – numpy int64 array of local data indices in cluster s
      u[i, s]           – fraction of node i's data assigned to cluster s  [N, S]

    Fixed graph:
      adj_list[i]       – list of k neighbor indices (directed, fixed at init)
    """

    def __init__(
        self,
        N, S, D, net_name, inp_dim, out_dim,
        device, k, seed,
        train_data, train_labels,
        lr, batch_size,
        num_local_iters=5,
        init_style='fedspd_style',
        ckc=10,
    ):
        """
        Parameters
        ----------
        N               : number of nodes
        S               : number of cluster centers per node
        D               : flattened model dimension
        net_name        : model architecture string (same as rest of codebase)
        inp_dim, out_dim: model input/output dims
        device          : torch device
        k               : degree of fixed k-regular-random graph
        seed            : random seed (graph uses seed+9999 to avoid collision)
        train_data      : list of N CPU tensors, each [n_i, C, H, W]
        train_labels    : list of N CPU tensors, each [n_i]
        lr              : learning rate for local SGD
        batch_size      : local mini-batch size
        num_local_iters : SGD steps per round (matches --num_local_iters)
        init_style      : 'fedspd_style' | 'round_robin'
        ckc             : label-switch alignment frequency (rounds)
        """
        self.N = N
        self.S = S
        self.D = D
        self.net_name = net_name
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.num_local_iters = num_local_iters
        self.ckc = ckc
        self.rng = np.random.RandomState(seed)

        # ── Fixed k-regular graph ──────────────────────────────────────
        # Mirrors agg.k_regular_random: each node independently samples k
        # neighbors (directed). Fixed once at init, never changes.
        rng_g = np.random.RandomState(seed + 9999)
        self.adj_list = []
        for i in range(N):
            pool = np.delete(np.arange(N), i)
            nbrs = rng_g.choice(pool, size=min(k, N - 1), replace=False).tolist()
            self.adj_list.append(nbrs)

        # ── Cluster centers: [N, S, D] ─────────────────────────────────
        # 'fedspd_style': each (node, cluster) independently random-initialized.
        # 'round_robin':  all S centers per node start identical.
        self.centers = torch.zeros(N, S, D, device=device)
        if init_style == 'fedspd_style':
            # Independent random init per cluster — exactly what FedSPD's code does
            for i in range(N):
                for s in range(S):
                    m = models_module.load_net(net_name, inp_dim, out_dim, device)
                    self.centers[i, s] = utils.model_to_vec(m).detach()
        else:
            # All S centers per node start identical; gradient breaks symmetry
            for i in range(N):
                m = models_module.load_net(net_name, inp_dim, out_dim, device)
                vec = utils.model_to_vec(m).detach()
                for s in range(S):
                    self.centers[i, s] = vec.clone()

        # ── Data assignment ────────────────────────────────────────────
        self.cluster_idx = [[None] * S for _ in range(N)]
        self.u = np.zeros((N, S), dtype=np.float64)

        if init_style == 'fedspd_style':
            self._init_data_fedspd_style(train_data)
        else:
            self._init_data_round_robin(train_data)

        # ── Round-robin position per (node, cluster) ───────────────────
        # Each (node, cluster) pair has its own pointer into its data slice.
        self.rr_pos = {(i, s): 0 for i in range(N) for s in range(S)}

        # ── Model pool for re-clustering ───────────────────────────────
        # S models reused across all N nodes, avoiding N*S instantiations/round.
        self._pool = [
            models_module.load_net(net_name, inp_dim, out_dim, device)
            for _ in range(S)
        ]
        for m in self._pool:
            m.eval()

        self._print_init_stats()

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _init_data_fedspd_style(self, train_data):
        """
        Exact mirror of FedSPD's data initialization in main.py.

        For node i with n local samples:
          - pool1 = positional first half of local indices  {0 .. n//2-1}
          - pool2 = positional second half                  {n//2 .. n-1}
          For cluster s in {0 .. S-2}:
            - take (s+1)*portion from pool1
            - take (S-s)*portion  from pool2
            - assign union to cluster s, remove from pools
          Cluster S-1 gets everything remaining.

        Note: 'positional first/second half' has no semantic meaning for our
        Dirichlet/patho distributions — it is deliberately arbitrary, exactly
        as in the paper's code. The EM loop corrects the assignment from round 2.
        """
        for i in range(self.N):
            n = len(train_data[i])
            clusterwise = max(1, n // self.S)
            portion = max(1, int(clusterwise / (self.S + 1)))

            pool1 = set(range(n // 2))
            pool2 = set(range(n // 2, n))

            for s in range(self.S - 1):
                take1 = min(int((s + 1) * portion), len(pool1))
                take2 = min(int((self.S - s) * portion), len(pool2))

                c1 = set(
                    self.rng.choice(sorted(pool1), take1, replace=False).tolist()
                ) if take1 > 0 else set()
                c2 = set(
                    self.rng.choice(sorted(pool2), take2, replace=False).tolist()
                ) if take2 > 0 else set()
                assigned = c1 | c2

                self.cluster_idx[i][s] = np.array(sorted(assigned), dtype=np.int64)
                pool1 -= assigned
                pool2 -= assigned

            # Last cluster gets everything remaining
            remaining = pool1 | pool2
            self.cluster_idx[i][self.S - 1] = np.array(sorted(remaining), dtype=np.int64)

            self._fix_empty_clusters(i, n)
            self._compute_u(i)

    def _init_data_round_robin(self, train_data):
        """Round-robin: sample j → cluster (j % S). Clean EM bootstrap."""
        for i in range(self.N):
            n = len(train_data[i])
            for s in range(self.S):
                self.cluster_idx[i][s] = np.arange(s, n, self.S, dtype=np.int64)
            self._fix_empty_clusters(i, n)
            self._compute_u(i)

    def _fix_empty_clusters(self, i, n):
        """Steal one sample from the largest cluster to fill any empty cluster."""
        for s in range(self.S):
            if len(self.cluster_idx[i][s]) == 0:
                src = max(range(self.S), key=lambda x: len(self.cluster_idx[i][x]))
                stolen = self.cluster_idx[i][src][-1:].copy()
                self.cluster_idx[i][src] = self.cluster_idx[i][src][:-1]
                self.cluster_idx[i][s] = stolen

    def _compute_u(self, i):
        total = sum(len(self.cluster_idx[i][s]) for s in range(self.S))
        for s in range(self.S):
            self.u[i, s] = len(self.cluster_idx[i][s]) / max(total, 1)

    def _print_init_stats(self):
        sizes = [len(self.cluster_idx[i][s])
                 for i in range(self.N) for s in range(self.S)]
        print(f"[FedSPD] N={self.N}  S={self.S}  D={self.D}  "
              f"k={len(self.adj_list[0])}  ckc={self.ckc}")
        print(f"[FedSPD] Init cluster sizes — "
              f"mean={np.mean(sizes):.1f}  "
              f"min={int(np.min(sizes))}  max={int(np.max(sizes))}")

    # ------------------------------------------------------------------
    # Main round
    # ------------------------------------------------------------------

    def run_round(self, rnd, train_data, train_labels):
        """
        Execute one complete FedSPD round.

        Steps:
          1. Each node selects the cluster to update (weighted by u_is)
          2. Local SGD on that cluster's data subset
          3. Average updated center with same-cluster neighbors from fixed graph
          4. Re-cluster all local data via batched loss comparison (Option A)
          5. Every ckc rounds: Hungarian label-switch alignment across nodes

        Returns
        -------
        personalized : [N, D] tensor
            Weighted sum of cluster centers (u_is * center_is), summed over s.
            Written into node_states for the standard evaluation path.
        """
        N, S = self.N, self.S

        # ── Step 1: cluster selection ──────────────────────────────────
        # Each node samples one cluster proportional to its data fraction u_is.
        selected = np.empty(N, dtype=np.int64)
        for i in range(N):
            p = self.u[i].copy()
            p /= max(p.sum(), 1e-9)
            selected[i] = self.rng.choice(S, p=p)

        # ── Step 2: local training ─────────────────────────────────────
        # Only the selected cluster's center is updated; others unchanged.
        updated = self.centers.clone()
        for i in range(N):
            s = int(selected[i])
            idx = self.cluster_idx[i][s]
            if len(idx) == 0:
                continue   # edge case: empty cluster, skip (fix_empty_clusters prevents this)

            # Slice only the data assigned to cluster s for this node.
            # train_data[i] is a CPU tensor; local_train_worker_inline
            # moves mini-batches to device internally.
            X_s = train_data[i][idx]
            y_s = train_labels[i][idx]

            # Route rr_indices through the per-(node, cluster) pointer.
            # local_train_worker_inline reads and writes rr_indices[node_id].
            tmp_rr = {i: self.rr_pos[(i, s)]}

            new_vec = train_node.local_train_worker_inline(
                i,
                self.centers[i, s].clone(),   # start from current cluster center
                X_s, y_s,
                self.inp_dim, self.out_dim, self.net_name,
                self.num_local_iters,
                self.batch_size,
                self.lr,
                self.device,
                'round_robin',
                tmp_rr,
            )
            self.rr_pos[(i, s)] = tmp_rr[i]
            updated[i, s] = new_vec.detach()

        # ── Step 3: aggregate with same-cluster neighbors ──────────────
        # Node i averages its updated center with neighbors j ∈ adj_list[i]
        # that selected the same cluster this round.
        # Nodes that selected different clusters: their centers are unchanged.
        new_centers = updated.clone()
        for i in range(N):
            s = int(selected[i])
            same = [j for j in self.adj_list[i] if selected[j] == s]
            if same:
                contributors = [i] + same
                stacked = torch.stack([updated[j, s] for j in contributors])
                new_centers[i, s] = stacked.mean(0)

        self.centers = new_centers

        # ── Step 4: re-cluster local data (batched, Option A) ──────────
        self._recluster(train_data, train_labels)

        # ── Step 5: label-switch alignment ─────────────────────────────
        if rnd > 0 and rnd % self.ckc == 0:
            self._fix_label_switching()

        return self._personalized_weights()

    # ------------------------------------------------------------------
    # Re-clustering
    # ------------------------------------------------------------------

    def _recluster(self, train_data, train_labels):
        """
        Re-assign each node's data samples to clusters by comparing per-sample
        cross-entropy losses across all S cluster models.

        For node i:
          losses[s, j] = CrossEntropy(center_is, sample j)
          assignment[j] = argmin_s losses[s, j]

        Uses the model pool (self._pool) so only S model objects exist at a time.
        Each node's entire local dataset is passed as one batch (Option A).
        """
        criterion = nn.CrossEntropyLoss(reduction='none')

        for i in range(self.N):
            X = train_data[i].to(self.device)    # [n_i, C, H, W]
            y = train_labels[i].to(self.device)  # [n_i]
            n = len(X)

            losses = torch.zeros(self.S, n, device=self.device)

            with torch.no_grad():
                for s in range(self.S):
                    # Load this cluster's center into the s-th pool model (in-place)
                    _load_vec_into_model(self.centers[i, s], self._pool[s])
                    losses[s] = criterion(self._pool[s](X), y)

            assignments = losses.argmin(dim=0).cpu().numpy()   # [n_i]

            new_idx = [[] for _ in range(self.S)]
            for j in range(n):
                new_idx[assignments[j]].append(j)

            for s in range(self.S):
                self.cluster_idx[i][s] = np.array(new_idx[s], dtype=np.int64)

            self._fix_empty_clusters(i, n)
            self._compute_u(i)

    # ------------------------------------------------------------------
    # Label-switch alignment
    # ------------------------------------------------------------------

    def _fix_label_switching(self):
        """
        Align cluster labels across all nodes using node 0 as reference.

        Cluster s at node i might have converged to the same distribution as
        cluster s' ≠ s at the reference node. The Hungarian algorithm finds the
        permutation that maximizes total cosine similarity and re-labels
        node i's clusters accordingly.
        """
        ref_norm = F.normalize(self.centers[0].float(), dim=1)   # [S, D]

        for i in range(1, self.N):
            node_norm = F.normalize(self.centers[i].float(), dim=1)  # [S, D]
            cos_sim = (ref_norm @ node_norm.T).cpu().numpy()           # [S, S]

            # linear_sum_assignment minimizes; negate to maximize similarity
            _, col_idx = linear_sum_assignment(-cos_sim)

            if not np.array_equal(col_idx, np.arange(self.S)):
                perm = list(col_idx)
                self.centers[i]     = self.centers[i][perm]
                self.cluster_idx[i] = [self.cluster_idx[i][perm[s]] for s in range(self.S)]
                self.u[i]           = self.u[i][perm]

    # ------------------------------------------------------------------
    # Personalization
    # ------------------------------------------------------------------

    def _personalized_weights(self):
        """
        Personalized model for node i = Σ_s u_is · center_is.

        Returns [N, D] float32 tensor for use in the existing eval path
        (written into node_states, evaluated via evaluate_node / evaluate_and_log).
        """
        u_t = torch.tensor(
            self.u, dtype=torch.float32, device=self.device
        ).unsqueeze(-1)                           # [N, S, 1]
        return (u_t * self.centers).sum(dim=1).detach()  # [N, D]