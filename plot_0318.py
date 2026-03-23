
from plot_training import TrainingPlotter
tp = TrainingPlotter("outputs/v2_dir0.1_live_ema_aff_tau2.0_0.5_half100_metrics.json")
tp.accuracy_candles()
tp.affinity_snapshots(shadow_path="outputs/v2_dir0.1_live_ema_aff_tau2.0_0.5_half100_shadow.npz", anchor=68)



# import numpy as np, matplotlib.pyplot as plt
# import pdb
# d = np.load("outputs/v2_dir0.1_full_tau1.0_shadow.npz")
# dist = d['shadow_dist']   # [N, N, R]
# anchor = 68
# pdb.set_trace()
# # Fixed friends of node 0 (say nodes 5, 10, 15) and strangers (55, 68, 80)
# fig, ax = plt.subplots(figsize=(12, 4))
# for j, label, color in [(61,'friend 61','steelblue'),(69,'friend 69','blue'),
#                          (97,'stranger 97','tomato'),(91,'stranger 91','red')]:
#     ax.plot(dist[anchor, j, :], label=label, color=color, alpha=0.7)
# ax.set_xlabel("Round"); ax.set_ylabel("L2 distance"); ax.legend(); ax.grid(alpha=0.3)
# plt.savefig("outputs/raw_dist_evolution.png", dpi=150)


# import json, numpy as np

# # Load
# with open("outputs/v2_dir0.1_live_ema_aff_tau2.0_0.8_half200_metrics.json") as f:
#     m = json.load(f)

# ntc    = np.array(m['node_to_cluster'])
# rounds = np.array(m['round'])
# accs   = np.array(m['local_acc'], dtype=float)   # [n_eval, N]

# # ── Bottom nodes by mean acc ──────────────────────────────────────────
# mean_acc = np.nanmean(accs, axis=0)
# min_acc  = np.nanmin(accs,  axis=0)
# bottom10 = np.argsort(mean_acc)[:10]

# print("Bottom 10 nodes:")
# print(f"{'Node':>5}  {'Cluster':>7}  {'MeanAcc':>8}  {'MinAcc':>7}  {'MinRound':>9}")
# for n in bottom10:
#     min_r = int(rounds[np.nanargmin(accs[:, n])])
#     print(f"{n:>5}  {ntc[n]:>7}  {mean_acc[n]:>8.1f}  {min_acc[n]:>7.1f}  {min_r:>9}")

# # ── Consistency check: how often is each node in the bottom 10%? ──────
# N = accs.shape[1]
# bottom_thresh = int(N * 0.1)   # 10% = 10 nodes for N=100
# in_bottom_count = np.zeros(N, dtype=int)
# for row in accs:
#     ranked = np.argsort(row)
#     for n in ranked[:bottom_thresh]:
#         in_bottom_count[n] += 1

# n_evals = len(rounds)
# print(f"\nNodes consistently in bottom 10% (>{n_evals//2} of {n_evals} eval rounds):")
# consistent = np.where(in_bottom_count > n_evals // 2)[0]
# for n in consistent:
#     print(f"  Node {n} [c{ntc[n]}]: in bottom {in_bottom_count[n]}/{n_evals} rounds  "
#           f"mean={mean_acc[n]:.1f}  min={min_acc[n]:.1f}")

# # ── Cluster breakdown of bottom 10 ────────────────────────────────────
# print(f"\nCluster breakdown of bottom 10:")
# from collections import Counter
# c = Counter(ntc[n] for n in bottom10)
# for cluster, count in sorted(c.items()):
#     print(f"  Cluster {cluster}: {count} nodes")