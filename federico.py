"""
federico.py
===========
FedeRiCo: Federating with the Right Collaborators.

Reference: Sui et al., "Find Your Friends: Personalized Federated Learning
with the Right Collaborators", FL-NeurIPS Workshop 2022. arXiv:2210.06597.

No public code released; implemented from the paper.

Core idea (EM formulation):
    Each client i's data is modelled as a mixture over all K clients'
    distributions. EM alternates between:
      E-step: compute w_ij = posterior probability that client j's model
              is useful for client i, using j's model loss on i's data.
      M-step: update client i's model using gradients from all clients j
              that sampled i, scaled by w_ji (how much j relies on i).

Key inversion vs L2C: the M-step updates φ_i based on who NEEDS i
(w_ji), not who i needs (w_ij). Popular/similar nodes get stronger
gradient signal; isolated nodes train mostly locally.

Self-weight: L_hat[i,i] is never updated (i never samples itself),
so it stays at 0. Since softmax(-0) > softmax(-positive_loss), node i
always retains a non-negligible self-weight, preventing full collapse
onto peers. This is implicit regularisation from the algorithm structure.

No validation set required (unlike L2C). All signal comes from evaluating
peer models on local training data.

Design choices for our setup:
  - N=100 nodes, M=20 peers per round (same communication budget as PAGAN)
  - ε-greedy sampling: ε fraction random, (1-ε) fraction highest-w peers
  - EMA momentum β on tracked losses (not raw cumulative loss)
  - E-step uses mean CE loss (numerically stable; equivalent to sum since
    all nodes have equal local dataset sizes)
  - M-step uses full local dataset of sampler j for gradient computation
  - Local training runs in main.py standard loop; this module handles
    only E-step + M-step in aggregate_and_update()

Args added in main.py:
    --federico_M        int     peers sampled per round       (default 20)
    --federico_epsilon  float   exploration probability       (default 0.3)
    --federico_beta     float   EMA momentum for loss         (default 0.6)
"""

import numpy as np
import torch
import torch.nn as nn

import utils
import models as models_module


class FedeRiCoProtocol:
    """
    FedeRiCo protocol.

    State
    -----
    L_hat : [N, N] numpy float64
        EMA-smoothed loss of client j's model evaluated on client i's data.
        L_hat[i, i] = 0 always (self never sampled; acts as implicit
        self-weight anchor in the softmax).
        Initialized to 0 → softmax(-0) = 1/N (uniform prior, as in paper).

    w : [N, N] numpy float64
        w[i, j] = softmax(-L_hat[i, :])_j
        Row i sums to 1. Represents how useful j's model is for client i.
    """

    def __init__(self, N, M, epsilon, beta, device, seed):
        """
        Parameters
        ----------
        N       : number of nodes
        M       : peers sampled per round (use args.k = 20)
        epsilon : exploration probability in ε-greedy sampling
        beta    : EMA momentum for loss update (paper: 0.6)
        device  : torch device
        seed    : random seed
        """
        self.N = N
        self.M = M
        self.epsilon = epsilon
        self.beta = beta
        self.device = device
        self.rng = np.random.RandomState(seed)

        # EMA loss matrix: L_hat[i,j] = EMA of loss(model_j, data_i)
        # Initialized to 0 → w = 1/N (uniform, eq. to paper's w^(0)=1/K)
        self.L_hat = np.zeros((N, N), dtype=np.float64)

        # Weight matrix recomputed from L_hat each round
        self.w = np.ones((N, N), dtype=np.float64) / N

    # ------------------------------------------------------------------
    # ε-greedy peer sampling
    # ------------------------------------------------------------------

    def _sample_peers(self, i):
        """
        ε-greedy sampling of M peers for node i (never includes self).

        n_random = round(M * ε) peers chosen uniformly at random.
        n_greedy = M - n_random peers chosen as top-w[i,:] (most trusted).
        """
        pool = np.delete(np.arange(self.N), i)   # exclude self

        n_random = max(0, round(self.M * self.epsilon))
        n_greedy = self.M - n_random

        if n_random >= len(pool):
            # Edge case: pool smaller than n_random (shouldn't happen at N=100)
            return pool.tolist()

        # Random component
        random_peers = self.rng.choice(pool, size=n_random, replace=False)

        # Greedy component: top n_greedy from remaining pool by w[i,j]
        remaining = np.setdiff1d(pool, random_peers)
        if n_greedy > 0 and len(remaining) > 0:
            scores = self.w[i, remaining]
            top_idx = np.argsort(scores)[::-1][:n_greedy]
            greedy_peers = remaining[top_idx].tolist()
        else:
            greedy_peers = []

        return random_peers.tolist() + greedy_peers

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def aggregate_and_update(
        self, rnd, node_states, train_data, train_labels,
        net_name, inp_dim, out_dim, lr
    ):
        """
        Execute one FedeRiCo E-step + M-step.

        Called AFTER main.py's standard local training has updated node_states.

        Parameters
        ----------
        rnd                        : current round (0-based)
        node_states                : [N, D] post-training weights on device
        train_data, train_labels   : list of N CPU tensors (full local datasets)
        net_name, inp_dim, out_dim : model architecture
        lr                         : learning rate for M-step gradient application

        Returns
        -------
        new_states : [N, D] updated weights on device
        """
        N = self.N
        criterion = nn.CrossEntropyLoss(reduction='mean')

        # ── Step 1: ε-greedy peer sampling ─────────────────────────────
        # peers_list[i] = list of M peer indices that node i observes
        peers_list = [self._sample_peers(i) for i in range(N)]

        # Reverse map: reverse_map[i] = list of nodes j that sampled i
        # Used in M-step to know which gradients node i must receive
        reverse_map = [[] for _ in range(N)]
        for i in range(N):
            for j in peers_list[i]:
                reverse_map[j].append(i)

        # ── Step 2: E-step ─────────────────────────────────────────────
        # For each node i: evaluate model of each sampled peer j on i's data.
        # Update EMA: L_hat[i,j] = (1-β)*L_hat[i,j] + β*loss(φ_j, D_i)
        # Unsampled peers: L_hat[i,j] unchanged (lazy evaluation, as in paper).
        # Self (j=i): never updated, stays at 0 → self-weight anchor.

        for i in range(N):
            X_i = train_data[i].to(self.device)     # [n_i, C, H, W]
            y_i = train_labels[i].to(self.device)   # [n_i]

            for j in peers_list[i]:
                # Load j's post-training model and evaluate on i's data
                model_j = utils.vec_to_model(
                    node_states[j], net_name, inp_dim, out_dim, self.device)
                model_j.eval()
                with torch.no_grad():
                    loss_val = criterion(model_j(X_i), y_i).item()
                del model_j

                # EMA update of L_hat[i, j]
                self.L_hat[i, j] = (
                    (1.0 - self.beta) * self.L_hat[i, j]
                    + self.beta * loss_val
                )

        # Recompute w[i, :] = softmax(-L_hat[i, :]) for all i
        # Numerically stable: subtract row max before exp
        for i in range(N):
            neg_l = -self.L_hat[i].copy()
            neg_l -= neg_l.max()
            exp_l = np.exp(neg_l)
            self.w[i] = exp_l / exp_l.sum()

        # ── Step 3: M-step ─────────────────────────────────────────────
        # For each node i: accumulate scaled gradients from all j that
        # sampled i. Gradient from j is computed on j's data, scaled by
        # w[j, i] (how much j currently relies on i's model).
        # Apply sum of scaled gradients to i's post-training model.
        #
        # If no one sampled i this round, i's model is unchanged (local only).

        new_states = node_states.clone()

        for i in range(N):
            samplers = reverse_map[i]   # nodes j that sampled i
            if len(samplers) == 0:
                continue

            # Load i's post-local-training model
            model_i = utils.vec_to_model(
                node_states[i], net_name, inp_dim, out_dim, self.device)
            model_i.train()

            # Accumulate scaled gradients across all samplers j
            # grad_sum[k] = Σ_j  w[j,i] * ∇_{φ_i} loss(φ_i, D_j)
            grad_sum = [
                torch.zeros_like(p.data) for p in model_i.parameters()
            ]

            for j in samplers:
                X_j = train_data[j].to(self.device)
                y_j = train_labels[j].to(self.device)
                scale = float(self.w[j, i])   # how much j relies on i

                model_i.zero_grad()
                loss = criterion(model_i(X_j), y_j)
                loss.backward()

                for k, p in enumerate(model_i.parameters()):
                    if p.grad is not None:
                        grad_sum[k] += scale * p.grad.data

            # Apply accumulated gradient: φ_i ← φ_i - η * grad_sum
            with torch.no_grad():
                for p, g in zip(model_i.parameters(), grad_sum):
                    p.data -= lr * g

            new_states[i] = utils.model_to_vec(model_i).detach()
            del model_i

        return new_states.detach()