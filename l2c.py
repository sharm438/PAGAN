"""
l2c.py
======
L2C: Learning to Collaborate in Decentralized Learning of Personalized Models.

Reference: Li et al., CVPR 2022.
No public code released; implemented from the paper.

Design choices for fair p2p comparison:
  - k=20 random peer sampling during warmup (same communication budget as PAGAN).
  - After warmup_rounds, freeze topology: keep top-k peers by learned alpha value.
  - Aggregation uses weighted model average (not per-round gradient injection).
    This equals the cumulative-diff-from-init formulation and is valid in p2p
    where models diverge. (See paper discussion in implementation notes.)
  - Full local val set used for alpha gradient every round (matches paper Sec 3.2).
  - No fine-tuning phase; meta-L2C excluded (requires AllReduce global sync).

Aggregation rule (Eq. 2, adapted):
    θ_i^{t+1} = Σ_{j ∈ {i} ∪ N(i)}  w_{ij} · θ_j^{t+1/2}
    w_i = softmax(α_i)

Alpha update — analytical gradient (Eq. 3):
    ∂L_val / ∂α_{ij}  =  w_{ij} · ( g_i · θ_j  −  Σ_m w_{im} · g_i · θ_m )
    where  g_i = ∂L_val(θ_agg_i) / ∂θ_agg_i   (flattened val-loss gradient)
    α_{ij} ← α_{ij} − beta · ∂L_val / ∂α_{ij}

Derivation of the gradient formula:
    θ_new = Σ_m w_m θ_m,  w_m = softmax(α)_m
    ∂θ_new / ∂α_j = w_j (θ_j − θ_new)          [softmax Jacobian]
    ∂L / ∂α_j     = g · ∂θ_new/∂α_j
                   = w_j · (g·θ_j − g·θ_new)
                   = w_j · (dots_j − Σ_m w_m · dots_m)
    where dots_m = g · θ_m  (scalar dot product)

Integration with main.py:
    - Local training runs in main.py's standard loop (node_states updated).
    - aggregate_and_update() is called in the aggregation block with
      post-training node_states and local val data.
    - Returns new [N, D] node_states for use in the evaluation path.

Args added in main.py:
    --l2c_beta     float   LR for alpha logits           (default 0.01)
    --l2c_warmup   int     rounds before topology freeze  (default 50)
"""

import numpy as np
import torch
import torch.nn as nn

import utils
import models as models_module


class L2CProtocol:
    """
    L2C mixing-weight protocol.

    State
    -----
    alpha : dict[(i,j) -> float]
        Sparse logit for node i trusting node j.
        Initialised to 0.0 on first encounter → uniform softmax initially.
    fixed_topology : list[list[int]] | None
        None during warmup.  Set once at round == warmup_rounds.
    encountered : list[set]
        Peers each node has seen during warmup (used for freeze ranking).
    """

    def __init__(self, N, k, warmup_rounds, device, seed):
        """
        Parameters
        ----------
        N             : number of nodes
        k             : peers sampled per round (same as --k everywhere else)
        warmup_rounds : rounds of random sampling before topology freeze
        beta          : learning rate for alpha logits
        device        : torch device
        seed          : random seed
        """
        self.N = N
        self.k = k
        self.warmup_rounds = warmup_rounds
        # Adam optimizer state for alpha logits (matches paper: lr=0.1, wd=0.01)
        self.adam_lr = 0.1
        self.adam_wd = 0.01
        self.adam_b1 = 0.9
        self.adam_b2 = 0.999
        self.adam_eps = 1e-8
        self.adam_m = {}   # first moment per (i,j)
        self.adam_v = {}   # second moment per (i,j)
        self.adam_t = {}   # step count per (i,j)

        self.device = device
        self.rng = np.random.RandomState(seed)

        self.alpha = {}                              # sparse logit dict
        self.fixed_topology = None                   # set at freeze
        self.encountered = [set() for _ in range(N)]  # warmup peer sets

    # ------------------------------------------------------------------
    # Peer selection
    # ------------------------------------------------------------------

    def _get_peers(self, node_id):
        """
        Return list of k peer indices for node_id.
        During warmup: random sample, updating encountered sets.
        Post-warmup:   fixed frozen topology.
        """
        if self.fixed_topology is not None:
            return self.fixed_topology[node_id]

        pool = np.delete(np.arange(self.N), node_id)
        peers = self.rng.choice(
            pool, size=min(self.k, self.N - 1), replace=False
        ).tolist()
        self.encountered[node_id].update(peers)
        return peers

    def _freeze_topology(self):
        """
        Freeze topology once, at round == warmup_rounds.

        For each node: rank all encountered peers by their current alpha
        value (descending = most trusted) and keep the top-k.
        Nodes that encountered fewer than k unique peers keep all of them.
        """
        self.fixed_topology = []
        for i in range(self.N):
            peers = list(self.encountered[i])
            if len(peers) <= self.k:
                self.fixed_topology.append(peers)
            else:
                scored = [(self.alpha.get((i, j), 0.0), j) for j in peers]
                scored.sort(reverse=True)
                self.fixed_topology.append([j for _, j in scored[: self.k]])

        avg_deg = float(np.mean([len(p) for p in self.fixed_topology]))
        print(f"[L2C] Topology frozen at round {self.warmup_rounds}.  "
              f"Avg neighbours per node: {avg_deg:.1f}")

    # ------------------------------------------------------------------
    # Main entry point (called from main.py aggregation block)
    # ------------------------------------------------------------------

    def aggregate_and_update(
        self, rnd, node_states, val_data, val_labels, net_name, inp_dim, out_dim
    ):
        """
        One L2C aggregation + alpha update.

        Called AFTER local training has already updated node_states.

        Parameters
        ----------
        rnd         : current round index (0-based)
        node_states : [N, D] post-training weights on device
        val_data    : list of N CPU tensors, each [n_val, C, H, W]
        val_labels  : list of N CPU tensors, each [n_val]
        net_name, inp_dim, out_dim : model architecture

        Returns
        -------
        new_states  : [N, D] aggregated weights on device
        """
        N = self.N

        # ── Freeze topology at warmup boundary ─────────────────────────
        if rnd == self.warmup_rounds and self.fixed_topology is None:
            self._freeze_topology()

        # ── Step 1: peer selection ─────────────────────────────────────
        peers_list = [self._get_peers(i) for i in range(N)]

        # ── Step 2: weighted model aggregation ─────────────────────────
        # group[i] = [i] + peers  (self is always the first entry)
        # w[i]     = softmax over group's alpha logits
        # new_states[i] = Σ_m w[m] * node_states[group[m]]
        new_states = torch.zeros_like(node_states)
        groups  = []   # list of lists, length N
        weights = []   # list of numpy float64 arrays, length N

        for i in range(N):
            group = [i] + peers_list[i]
            groups.append(group)

            # Initialise any previously unseen alpha entries to 0.0
            for j in group:
                if (i, j) not in self.alpha:
                    self.alpha[(i, j)] = 0.0

            # Numerically stable softmax over the group's logits
            a = np.array([self.alpha[(i, j)] for j in group], dtype=np.float64)
            a -= a.max()
            w = np.exp(a)
            w /= w.sum()
            weights.append(w)

            # Weighted average of post-training model vectors
            group_vecs = torch.stack([node_states[j] for j in group])  # [k+1, D]
            w_t = torch.tensor(w, dtype=node_states.dtype, device=self.device)
            new_states[i] = (w_t.unsqueeze(1) * group_vecs).sum(0)

        # ── Step 3: alpha update via analytical gradient ───────────────
        #
        # For each node i:
        #   g_i     = ∂L_val(new_states[i]) / ∂θ_agg   [D]
        #   dots[m] = g_i · node_states[group[m]]        [k+1]
        #   ∂L/∂α[j] = w[j] * (dots[j] − Σ_m w[m]*dots[m])
        #   α[i,j] -= beta * ∂L/∂α[j]
        #
        criterion = nn.CrossEntropyLoss()

        for i in range(N):
            X_val = val_data[i]
            y_val = val_labels[i]
            if len(X_val) == 0:
                continue   # no val data — skip alpha update for this node

            group = groups[i]      # list of k+1 indices
            w     = weights[i]     # numpy [k+1]

            # Build model from aggregated weights, run forward+backward on val
            model = utils.vec_to_model(
                new_states[i], net_name, inp_dim, out_dim, self.device)
            model.train()   # keep running stats active for GN layers

            X_dev = X_val.to(self.device)
            y_dev = y_val.to(self.device)

            logits = model(X_dev)
            loss   = criterion(logits, y_dev)
            model.zero_grad()
            loss.backward()

            # Flatten gradient into a single [D] vector
            g_flat = torch.cat([
                p.grad.view(-1) if p.grad is not None
                else torch.zeros(p.numel(), device=self.device)
                for p in model.parameters()
            ])   # [D]

            # dots[m] = g_flat · node_states[group[m]]
            # Vectorised: [k+1, D] @ [D] = [k+1]
            group_vecs = torch.stack([node_states[j] for j in group])  # [k+1, D]
            dots = (group_vecs @ g_flat).cpu().numpy()                  # [k+1]

            # Analytical softmax-Jacobian gradient
            weighted_avg_dot = float(np.dot(w, dots))
            dL_dalpha = w * (dots - weighted_avg_dot)   # [k+1]

            # Adam update on alpha logits (paper: lr=0.1, wd=0.01)
            for m_idx, j in enumerate(group):
                key = (i, j)
                grad = float(dL_dalpha[m_idx])

                # L2 weight decay folded into gradient
                grad += self.adam_wd * self.alpha.get(key, 0.0)

                # Initialise moments on first encounter
                if key not in self.adam_m:
                    self.adam_m[key] = 0.0
                    self.adam_v[key] = 0.0
                    self.adam_t[key] = 0

                self.adam_t[key] += 1
                t = self.adam_t[key]

                self.adam_m[key] = (self.adam_b1 * self.adam_m[key]
                                    + (1 - self.adam_b1) * grad)
                self.adam_v[key] = (self.adam_b2 * self.adam_v[key]
                                    + (1 - self.adam_b2) * grad ** 2)

                m_hat = self.adam_m[key] / (1 - self.adam_b1 ** t)
                v_hat = self.adam_v[key] / (1 - self.adam_b2 ** t)

                self.alpha[key] = (self.alpha.get(key, 0.0)
                                   - self.adam_lr * m_hat
                                   / (v_hat ** 0.5 + self.adam_eps))

            # Clean up to free VRAM before next node
            del model, logits, loss, g_flat, group_vecs

        return new_states.detach()