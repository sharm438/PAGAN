# PAGAN — Peer-Aligned Gossip Aggregation Network

> Personalized peer-to-peer federated learning via adaptive model-distance affinity and learned neighbourhood representations.

---

## The Problem

In standard federated learning, a central server aggregates gradients or model weights from all participating clients into a single global model. This works well when data is homogeneous — but in the real world, it rarely is. A hospital in one city sees different patient demographics than one in another. A phone keyboard in one language learns different typing patterns than one in another. Forcing these heterogeneous clients into a single global model degrades performance for everyone, particularly for clients whose local distribution is far from the average.

**Personalized federated learning** addresses this by allowing each client to converge toward a model that reflects its own data distribution rather than the global average. The challenge is doing this without a central server, without sharing raw data, and without requiring clients to know in advance who their "similar" peers are.

**PAGAN operates in a fully peer-to-peer (P2P) setting.** There is no parameter server. Each node communicates directly with a subset of peers, decides how much to learn from each one, and adapts its neighbourhood over time. The only shared information between nodes is model parameters — no labels, no data statistics, no ground truth cluster assignments.

The core difficulty in this setting is **data heterogeneity**: nodes with dissimilar data distributions should not mix their models, but nodes with similar distributions should collaborate as much as possible. Identifying similarity without access to each other's data is the central technical challenge.

---

## Existing Approaches to Personalization

Personalized federated learning has been approached from several directions, each with different assumptions and tradeoffs:

**Validation-data-based methods** (e.g. L2C — *Learning to Collaborate*) use a small held-out validation set at each client to score incoming model updates — keeping updates that improve local performance and discarding those that hurt. This is effective but requires each client to hold a clean, representative validation set, which is often unavailable in practice.

**Expectation-maximization-based methods** (e.g. *FEDERICO*) model the population of clients as a mixture of latent distributions and use EM to jointly infer cluster assignments and per-cluster models. These methods are theoretically grounded but typically require either a central coordinator or several rounds of global communication to converge on cluster structure.

**Clustering-based methods** (e.g. *FedSoft*, *IFCA*) explicitly partition clients into groups and train a separate model per group. The challenge is that cluster assignments are either fixed at initialization (brittle) or require expensive iterative refinement. They also tend to assume the number of clusters is known.

**Regularization-based methods** (e.g. *pFedMe*, *Ditto*) add a proximity term to the local objective that pulls the personalized model toward a global reference. These methods are simple and effective for mild heterogeneity but do not exploit peer-to-peer structure — every node is regularized toward the same global model regardless of distributional similarity.

**Graph-based methods** try to learn a communication topology where edges reflect distributional similarity. Most require either explicit data sharing to compute similarity or a centralized step to construct the graph.

---

## PAGAN's Approach

PAGAN avoids all of these assumptions. It requires no validation data, no central coordinator, no pre-specified cluster count, and no data sharing. Instead, it uses a simple but powerful observation:

> **If two nodes train from the same initialization on similar data, their models will remain close in parameter space. If they train on dissimilar data, their models will diverge.**

Model distance in parameter space is used as a proxy for distributional similarity. This proxy is available at zero communication overhead — every node already exchanges model weights with its peers for aggregation, so computing pairwise distances costs nothing extra.

From this proxy, PAGAN builds two complementary signals:

**Affinity** — a per-pair scalar that determines how much node `i` weights node `j`'s model when updating its own. Affinity is computed as a softmax over distance-based scores across currently sampled peers. Nodes with similar data naturally drift closer in model space and receive higher affinity weights, driving selective collaboration.

**Neighbourhood embeddings** — a learned low-dimensional representation for each node, trained via a ladder triplet loss over the ranked neighbour lists exchanged each round. Embeddings capture the *topology* of each node's neighbourhood — who is close to whom — rather than just raw distances, and serve as a second independent similarity signal that is resistant to the contamination effects that can corrupt raw model distances.

These two signals are blended with a scheduled coefficient α(t) that transitions from pure model-distance affinity early in training (when embeddings are not yet reliable) to a blend of model and embedding affinity later (when embeddings have learned to reflect distributional structure).

---

## Protocol: Warmup and Post-Warmup

### Warmup Phase (rounds 0 – W)

During warmup, each node trains in **complete isolation** — it takes no model updates from any peer. This produces a clean, uncontaminated window in which pairwise model distances reflect only the distributional differences between nodes' local data.

Each round, every node samples `k` peers from the network and measures its model distance to each sampled peer. These observations are accumulated in a **shadow registry** — a per-node store of distance history, EMA-smoothed distances, rank-position histories, and distance spread statistics. No aggregation occurs; the shadow registry is purely observational.

By the end of warmup, each node has a rough picture of its neighbourhood: who has been consistently close, who has been consistently far, how stable those rankings have been. This warmup snapshot serves as the anchor for all post-warmup decisions.

Embeddings are also trained during warmup using the ladder triplet loss over neighbour ranklists. A neighbour's ranklist — the ordered list of *their* closest peers — provides indirect evidence about the global topology that node `i` cannot directly observe. Training on this evidence produces embeddings that reflect cluster structure even before any mixing occurs.

### Post-Warmup Phase (rounds W – T)

After warmup, aggregation begins. Each round proceeds as follows:

**1. Sampling** — Each node samples `k` peers using a four-slot quota designed to balance exploitation and exploration:
- **Slot A** (3 peers): top-3 from the previous round's physical ranking — persistence, ensuring recently-close peers are always observed.
- **Slot B** (7 peers): top-7 by EMA distance across all N nodes — exploitation of the best-known neighbours, globally remembered.
- **Slot C** (5 peers): top-5 by embedding distance — discovery of structurally similar peers that may not yet be close in model space.
- **Slot D** (remainder): uniformly random — exploration, preventing starvation of unseen nodes.

**2. Distance computation** — For each sampled peer, node `i` computes the L2 distance in model parameter space and the L2 distance in embedding space (from `i`'s own embedding table).

**3. EMA update** — The shadow registry updates the EMA distance for each observed peer: `D_ij ← λ·D_ij + (1-λ)·d_observed`. The EMA provides a smoothed, persistent distance estimate that feeds slot B and stabilises aggregation against round-to-round noise.

**4. Blended affinity and aggregation** — Aggregation weights are computed by blending two independent softmax distributions:

```
W_model[i, j]  =  softmax( -D_ema(i,j) / τ_model )   over sampled k + self
W_emb[i, j]    =  softmax( -D_emb(i,j) / τ_emb   )   over sampled k + self
W[i, j]        =  α(t) · W_model[i, j]  +  (1-α(t)) · W_emb[i, j]
new_state[i]   =  Σ_j  W[i, j] · state[j]
```

The blend coefficient α(t) follows a schedule (linear, cosine, concave, convex, or floor) that starts at 1.0 (pure model distance) and decays toward 0.0 (pure embedding distance) as embeddings mature.

**5. Embedding update** — Embeddings are updated using the ladder triplet loss over neighbour ranklists. Each neighbour's ranklist is weighted by its rank position in `i`'s physical ranking: `weight = 1 / (rank + 1)` or `1 / √(rank + 1)`. Closer neighbours' ranklists are more informative about `i`'s true neighbourhood and receive proportionally more gradient weight.

---

## Installation

```bash
git clone https://github.com/<your-username>/pagan.git
cd pagan
pip install -r requirements.txt
```

Requires Python 3.9+, PyTorch 2.3+, CUDA 12.1+ (for GPU acceleration).

---

## Running Experiments

### Basic syntax

```bash
python main.py \
  --topo paganv3 \
  --dataset cifar10 \
  --dist_case dirichlet \
  --bias 0.1 \
  --num_spokes 100 \
  --num_rounds 500 \
  --k 20 \
  --exp my_experiment
```

### Reproducing baselines

```bash
# Isolated training (no collaboration)
python main.py --aggregation isolated --dataset cifar10 --dist_case dirichlet --bias 0.1

# FedAvg
python main.py --aggregation fedavg --dataset cifar10 --dist_case dirichlet --bias 0.1

# PAGANv2 (model-distance affinity, no embedding blend)
python main.py --topo paganv2 --dataset cifar10 --dist_case dirichlet --bias 0.1
```

---

## Arguments

### Core

| Argument | Type | Default | Description |
|---|---|---|---|
| `--exp` | str | `experiment` | Output filename prefix for saved metrics and registry files. |
| `--dataset` | str | `cifar10` | Dataset. Choices: `cifar10`, `mnist`, `femnist`, `agnews`, `tiny_imagenet`, `cifar100`. |
| `--dist_case` | str | `dirichlet` | Data heterogeneity distribution. `dirichlet` uses a Dirichlet concentration parameter (`--bias`). `patho_1/2/3` are pathological label-skew settings of increasing severity. |
| `--bias` | float | `0.1` | Dirichlet concentration α. Lower = more heterogeneous. `0.1` is strongly heterogeneous; `100.0` approaches IID. |
| `--aggregation` | str | `p2p` | Aggregation protocol. `p2p` enables topology-aware mixing; use with `--topo`. |
| `--topo` | str | — | Topology protocol. `paganv3` for the full protocol. `paganv2` for the model-distance-only baseline. |
| `--num_spokes` | int | `100` | Number of nodes in the federation. |
| `--num_rounds` | int | `500` | Total training rounds. |
| `--num_local_iters` | int | `5` | Local SGD steps per round per node. |
| `--k` | int | `20` | Number of peers sampled per node per round. |
| `--lr` | float | `0.1` | Local SGD learning rate. |
| `--batch_size` | int | — | Local mini-batch size. Default uses the full local dataset. |
| `--eval_time` | int | `10` | Evaluate accuracy every this many rounds. |
| `--embed_dim` | int | `8` | Dimensionality of the neighbourhood embedding per node. Each node maintains an embedding table of shape `[N, embed_dim]`. |
| `--seed` | int | `108` | Random seed for reproducibility. |
| `--gpu` | int | `0` | GPU index. |
| `--num_workers` | int | `10` | Parallel workers for evaluation. |
| `--local_test_frac` | float | `0.2` | Fraction of each node's local data held out as a local test set. |

### Warmup and EMA (shared by v2 and v3)

| Argument | Type | Default | Description |
|---|---|---|---|
| `--v2_warmup_rounds` | int | `25` | Number of isolated warmup rounds before aggregation begins. During warmup, nodes observe but do not mix. |
| `--v2_ema_lambda` | float | `0.95` | EMA decay coefficient for model distances in the shadow registry. Higher = slower update, more memory. Used for slot B sampling. |
| `--v2_tau_0` | float | `2.0` | Initial softmax temperature for model-distance affinity post-warmup. Higher = flatter distribution, more mixing. |
| `--v2_tau_min` | float | `0.5` | Minimum softmax temperature. The schedule decays from `tau_0` toward `tau_min` over `tau_half_life` rounds. |
| `--v2_tau_half_life` | float | `200.0` | Rounds post-warmup for temperature to reach the midpoint between `tau_0` and `tau_min`. |
| `--v2_shadow_window` | int | `20` | Rolling window size for rank history in the shadow registry. |
| `--v2_freeze_ema` | flag | off | Ablation: freeze EMA distances at the end of warmup. The affinity is fixed at the warmup snapshot for the rest of training. |
| `--v2_freeze_embeddings` | flag | off | Ablation: freeze embedding updates at the end of warmup. Embeddings do not train post-warmup. |
| `--v2_eff_thresh` | float | `0.02` | Weight threshold for soft TP/FP evaluation. Peers with aggregation weight above this are counted as effective. |

### Embedding training (shared by v2 and v3)

| Argument | Type | Default | Description |
|---|---|---|---|
| `--v2_emb_lr` | float | `0.1` | Embedding learning rate during warmup. |
| `--v2_emb_lr_post` | float | `0.01` | Embedding learning rate post-warmup. Switched to at round `warmup_rounds`. |
| `--v2_emb_steps` | int | `10` | Gradient steps per round for embedding training during warmup. Halved post-warmup. |

### PAGANv3 — blend schedule and triplet weighting

| Argument | Type | Default | Description |
|---|---|---|---|
| `--v3_alpha_start` | int | `warmup_rounds` | Round at which α begins decaying from 1.0. Before this round, aggregation is pure model-distance (α=1). |
| `--v3_alpha_end` | int | `num_rounds` | Round at which α reaches its minimum. After this round, α is fixed at its floor value. |
| `--v3_alpha_schedule` | str | `linear` | Schedule for α decay over `[alpha_start, alpha_end]`. Choices: `linear`, `cosine`, `concave` (stays high, drops late), `convex` (drops early, flattens), `floor` (never goes below `alpha_min`). |
| `--v3_alpha_min` | float | `0.3` | Minimum α value for the `floor` schedule. Ensures model-distance always contributes. |
| `--v3_triplet_scheme` | str | `flat` | Rank-weighting scheme for the triplet loss. `flat` gives equal weight to all neighbour ranklists (v2 behaviour). `inv_rank` weights the rank-`r` neighbour's ranklist by `1/(r+1)`. `inv_sqrt_rank` uses `1/√(r+1)`. |

---

## Evaluation Output

Each run saves:

- `<exp>_metrics.json` — full metrics log including per-round accuracy statistics (min, max, mean, percentile bands), soft TP/FP/FN/TN for model-only, embedding-only, and blended affinity signals, and embedding quality metrics (tp@5, tp@10, tp@15).
- `<exp>_shadow.npz` — shadow registry dump with full per-round rank and distance history, EMA distances, and distance spread.

### Accuracy statistics reported per round

| Key | Description |
|---|---|
| `Min` | Minimum local-test accuracy across all nodes |
| `Low20%` | 20th percentile across nodes |
| `Mid60%` | Mean of the middle 60% of nodes |
| `Top20%` | 80th percentile across nodes |
| `Max` | Maximum local-test accuracy across nodes |
| `Avg` | Mean across all nodes |

`Min` is the most informative single number — it reflects the worst-case node, typically those with the most heterogeneous data distribution. A protocol that improves `Avg` while degrading `Min` is sacrificing the hardest cases for the easy ones.

---

## Experimental Setups

Five heterogeneity configurations are used throughout evaluation:

| Name | `--dist_case` | `--bias` | Description |
|---|---|---|---|
| Dir0.1 | `dirichlet` | `0.1` | Strongly heterogeneous. Each node has high label skew; clusters are well-separated in model space. |
| Dir100 | `dirichlet` | `100.0` | Near-IID. Minimal heterogeneity; all nodes see similar distributions. |
| Patho1 | `patho_1` | — | Each node holds data from exactly 1 class. Maximum label skew. |
| Patho2 | `patho_2` | — | Each node holds data from exactly 2 classes. |
| Patho3 | `patho_3` | — | Each node holds data from exactly 3 classes. |

Dir100 serves as a sanity check — a good protocol should still converge without degrading accuracy when the data is nearly IID. The pathological settings stress-test the protocol's ability to identify similar peers under extreme label skew.
