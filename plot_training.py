"""
plot_training.py
================
Two complementary diagnostic views for PAGANv2 live-affinity experiments.

Plot 1 — Accuracy Candle Chart
  Shows min, low-20%, mid-60% band, high-20%, max across training rounds.
  One file per metrics JSON.  Overlay multiple experiments for comparison.

Plot 2 — Organic Tightening Dashboard (4-panel)
  Tracks whether live affinity is reducing stranger influence over time.
  Requires the _shadow.npz file saved by main_v2.py.

  Panel A: Rolling affinity gap (mean in-cluster affinity minus mean
           out-cluster affinity) for a set of anchor nodes — rising gap
           confirms organic tightening.
  Panel B: Effective stranger weight — sum of aggregation weight going to
           out-cluster nodes, estimated from shadow registry affinity.
           Should decrease over time if tightening is working.
  Panel C: Self-weight evolution from v2_soft_metrics log.
  Panel D: Soft TP / (TP+FP+FN) — precision of effective neighbourhood.

Usage
-----
from plot_training import TrainingPlotter

# Single experiment candle plot
tp = TrainingPlotter("outputs/v2_patho1_liveaff_tau1_metrics.json")
tp.accuracy_candles(save_name="candles_patho1_liveaff.png")

# Overlay two experiments
tp2 = TrainingPlotter("outputs/v2_patho1_frozen_tau2_metrics.json")
TrainingPlotter.overlay_candles(
    [tp, tp2],
    labels=["live-aff τ=1", "frozen τ=2"],
    save_name="overlay_patho1.png"
)

# Organic tightening dashboard (needs shadow file)
tp.tightening_dashboard(
    shadow_path="outputs/v2_patho1_liveaff_tau1_shadow.npz",
    node_to_cluster=tp.node_to_cluster,
    anchor_nodes=[0, 10, 50, 68],
    save_name="tightening_patho1.png"
)
"""

import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import matplotlib.cm as cm

from shadow_registry import ShadowRegistry


# -----------------------------------------------------------------------
class TrainingPlotter:
    def __init__(self, metrics_path: str):
        with open(metrics_path) as f:
            self.metrics = json.load(f)

        self.rounds      = np.array(self.metrics["round"])
        # local_acc[r] is a list of per-node local test accuracies at round r
        raw = self.metrics.get("local_acc", [])
        self.local_acc   = [np.array(r, dtype=float) for r in raw]

        ntc = self.metrics.get("node_to_cluster", [])
        self.node_to_cluster = np.array(ntc) if ntc else None
        self.soft_metrics    = self.metrics.get("v2_soft_metrics", [])
        self.exp_name        = os.path.basename(metrics_path).replace("_metrics.json","")

        os.makedirs("outputs", exist_ok=True)

    # ------------------------------------------------------------------
    # Candle statistics
    # ------------------------------------------------------------------
    def _candle_stats(self):
        stats = []
        for accs in self.local_acc:
            a = np.sort(accs[~np.isnan(accs)])
            if len(a) == 0:
                stats.append(dict(mn=np.nan,p20=np.nan,p40=np.nan,
                                   p60=np.nan,p80=np.nan,mx=np.nan,avg=np.nan))
                continue
            stats.append(dict(
                mn  = float(np.min(a)),
                p20 = float(np.percentile(a, 20)),
                p40 = float(np.percentile(a, 40)),
                p60 = float(np.percentile(a, 60)),
                p80 = float(np.percentile(a, 80)),
                mx  = float(np.max(a)),
                avg = float(np.mean(a)),
            ))
        return stats

    # ------------------------------------------------------------------
    # 1. Accuracy candle chart
    # ------------------------------------------------------------------
    def accuracy_candles(self, save_name: str = None,
                         title: str = None, ax=None):
        stats  = self._candle_stats()
        rnds   = self.rounds[:len(stats)]
        mn     = np.array([s['mn']  for s in stats])
        p20    = np.array([s['p20'] for s in stats])
        p40    = np.array([s['p40'] for s in stats])
        p60    = np.array([s['p60'] for s in stats])
        p80    = np.array([s['p80'] for s in stats])
        mx     = np.array([s['mx']  for s in stats])
        avg    = np.array([s['avg'] for s in stats])

        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=(13, 5))

        # Outer whiskers: min–max
        ax.fill_between(rnds, mn, mx,
                        alpha=0.12, color="#4C72B0", label="Min–Max")
        # Low 20%
        ax.fill_between(rnds, mn, p20,
                        alpha=0.25, color="#DD4444", label="Bottom 20%")
        # Mid 60% band
        ax.fill_between(rnds, p20, p80,
                        alpha=0.30, color="#4C72B0", label="Mid 60%")
        # Top 20%
        ax.fill_between(rnds, p80, mx,
                        alpha=0.25, color="#44AA44", label="Top 20%")
        # Lines
        ax.plot(rnds, mn,  color="#DD4444", lw=1.2, ls="--", alpha=0.9)
        ax.plot(rnds, mx,  color="#44AA44", lw=1.2, ls="--", alpha=0.9)
        ax.plot(rnds, avg, color="#222222", lw=1.8, label="Mean", alpha=0.9)
        ax.plot(rnds, p40, color="#4C72B0", lw=0.8, ls=":", alpha=0.6)  # median
        ax.plot(rnds, p60, color="#4C72B0", lw=0.8, ls=":", alpha=0.6)

        ax.set_xlabel("Round"); ax.set_ylabel("Local Test Accuracy (%)")
        ax.set_title(title or f"Accuracy distribution — {self.exp_name}")
        ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.25)
        ax.set_ylim(0, 105)

        if standalone:
            plt.tight_layout()
            path = f"outputs/{save_name or self.exp_name + '_candles.png'}"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"[plot] -> {path}")
            plt.close(fig)

    # ------------------------------------------------------------------
    # Overlay candles for multiple experiments
    # ------------------------------------------------------------------
    @staticmethod
    def overlay_candles(plotters: list, labels: list,
                        save_name: str = "overlay_candles.png"):
        n = len(plotters)
        colors = cm.get_cmap("tab10", n)
        fig, axes = plt.subplots(n, 1, figsize=(13, 4*n), sharex=True)
        if n == 1:
            axes = [axes]

        for i, (tp, label) in enumerate(zip(plotters, labels)):
            tp.accuracy_candles(ax=axes[i], title=label)

        plt.tight_layout()
        path = f"outputs/{save_name}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[plot] -> {path}")
        plt.close(fig)

    # ------------------------------------------------------------------
    # 2. Organic tightening dashboard
    # ------------------------------------------------------------------
    def tightening_dashboard(self,
                              shadow_path: str,
                              node_to_cluster: np.ndarray = None,
                              anchor_nodes: list = None,
                              save_name: str = None):
        """
        4-panel dashboard tracking whether live affinity is organically
        reducing stranger influence over training.
        """
        reg = ShadowRegistry.load(shadow_path)
        ntc = node_to_cluster if node_to_cluster is not None \
              else self.node_to_cluster
        if ntc is None:
            print("[tightening_dashboard] node_to_cluster required.")
            return

        N = reg.N
        R = reg.R
        rounds_all = np.arange(R)

        if anchor_nodes is None:
            anchor_nodes = list(range(min(5, N)))

        # ── Panel A: Affinity gap over time ──────────────────────────
        # For each round r, for each anchor, compute:
        #   mean A_ij (j in-cluster) minus mean A_ij (j out-cluster)
        # Using a rolling window of 10 for smoothness

        WINDOW = 10
        gaps_per_anchor = []
        for anchor in anchor_nodes:
            my_c    = ntc[anchor]
            in_idx  = [j for j in range(N) if j != anchor and ntc[j] == my_c]
            out_idx = [j for j in range(N) if j != anchor and ntc[j] != my_c]
            gap_series = []
            for r in range(R):
                A_in  = np.nanmean([reg.affinity(anchor, j,
                                    end_round=r, window=WINDOW)
                                    for j in in_idx])
                A_out = np.nanmean([reg.affinity(anchor, j,
                                    end_round=r, window=WINDOW)
                                    for j in out_idx])
                gap_series.append(A_in - A_out
                                   if (not np.isnan(A_in) and
                                       not np.isnan(A_out)) else np.nan)
            gaps_per_anchor.append(np.array(gap_series))

        # ── Panel B: Estimated stranger weight over time ──────────────
        # For each round r, linear-normalise affinities over ALL observed
        # nodes (including self at 1.0), compute fraction going to
        # out-cluster nodes. Average across all nodes.
        stranger_frac = []
        for r in range(R):
            fracs = []
            for i in range(N):
                my_c = ntc[i]
                A    = np.array([reg.affinity(i, j, end_round=r, window=WINDOW)
                                  for j in range(N)], dtype=np.float32)
                A[i] = 1.0   # self
                A    = np.where(np.isnan(A), 0.0, A)
                total = A.sum()
                if total <= 0:
                    continue
                out_mask = np.array([ntc[j] != my_c for j in range(N)],
                                     dtype=float)
                out_mask[i] = 0.0   # self not a stranger
                fracs.append(float((A * out_mask).sum() / total))
            stranger_frac.append(np.nanmean(fracs) if fracs else np.nan)
        stranger_frac = np.array(stranger_frac)

        # ── Panel C: Self-weight from soft_metrics ────────────────────
        sm_rnds   = np.array([s['round']           for s in self.soft_metrics])
        self_w    = np.array([s['avg_self_weight']  for s in self.soft_metrics])

        # ── Panel D: Soft precision TP/(TP+FP+FN) ────────────────────
        sm_prec   = np.array([
            s['tp'] / max(1, s['tp'] + s['fp'] + s['fn'])
            for s in self.soft_metrics
        ])

        # ── Plot ─────────────────────────────────────────────────────
        fig, axes = plt.subplots(4, 1, figsize=(13, 16), sharex=False)
        cmap_a = cm.get_cmap("tab10", len(anchor_nodes))

        # Panel A
        ax = axes[0]
        ax.set_title("A: Affinity gap (in-cluster − out-cluster) per anchor node\n"
                     "↑ Rising = organic tightening working")
        for k, (anchor, gap) in enumerate(zip(anchor_nodes, gaps_per_anchor)):
            ax.plot(rounds_all, gap, lw=1.4, alpha=0.8,
                    color=cmap_a(k),
                    label=f"Node {anchor} [c{ntc[anchor]}]")
        ax.axhline(0, color="black", lw=0.8, ls=":")
        ax.set_ylabel("Affinity gap"); ax.legend(fontsize=7); ax.grid(alpha=0.25)

        # Panel B
        ax = axes[1]
        ax.set_title("B: Network-average stranger weight\n"
                     "↓ Decreasing = strangers losing influence")
        ax.plot(rounds_all, stranger_frac * 100,
                color="#DD4444", lw=1.6)
        ax.fill_between(rounds_all, stranger_frac * 100,
                        alpha=0.15, color="#DD4444")
        ax.set_ylabel("Stranger weight (%)"); ax.grid(alpha=0.25)

        # Panel C
        ax = axes[2]
        ax.set_title("C: Average self-weight over training\n"
                     "↑ Rising = node trusting itself more (tightening)")
        if len(sm_rnds) > 0:
            ax.plot(sm_rnds, self_w * 100, color="#4C72B0", lw=1.6)
            ax.fill_between(sm_rnds, self_w * 100,
                            alpha=0.15, color="#4C72B0")
        ax.set_ylabel("Self-weight (%)"); ax.grid(alpha=0.25)

        # Panel D
        ax = axes[3]
        ax.set_title("D: Soft neighbourhood precision  TP / (TP+FP+FN)\n"
                     "↑ Rising = effective neighbourhood becoming purer")
        if len(sm_rnds) > 0:
            ax.plot(sm_rnds, sm_prec * 100, color="#44AA44", lw=1.6)
            ax.fill_between(sm_rnds, sm_prec * 100,
                            alpha=0.15, color="#44AA44")
        ax.set_ylabel("Precision (%)"); ax.set_xlabel("Round")
        ax.grid(alpha=0.25)

        plt.tight_layout()
        path = f"outputs/{save_name or self.exp_name + '_tightening.png'}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[plot] -> {path}")
        plt.close(fig)

    # ------------------------------------------------------------------
    # 3. Per-round affinity heatmap animation frames
    #    (produces one PNG per checkpoint round for inspection)
    # ------------------------------------------------------------------
    def affinity_snapshots(self,
                            shadow_path: str,
                            anchor: int,
                            node_to_cluster: np.ndarray = None,
                            rounds_to_plot: list = None,
                            window: int = 10,
                            save_prefix: str = None):
        """
        Bar-chart snapshots of A[anchor, :] at several training rounds.
        Shows how affinity distribution evolves post-warmup.
        Useful to confirm strangers lose affinity mass over time.
        """
        reg = ShadowRegistry.load(shadow_path)
        ntc = node_to_cluster if node_to_cluster is not None \
              else self.node_to_cluster
        if ntc is None:
            return

        N    = reg.N
        sort_idx = np.argsort(ntc)

        if rounds_to_plot is None:
            total = reg.R
            # rounds_to_plot = [total // 5, 2*total//5,
            #                     3*total//5, 4*total//5, total-1]
            rounds_to_plot = [20, 25, 30, 40, 50]
        num_clusters = int(ntc.max()) + 1
        cmap_c = cm.get_cmap("tab20", max(num_clusters, 2))

        fig, axes = plt.subplots(1, len(rounds_to_plot),
                                  figsize=(5*len(rounds_to_plot), 4),
                                  sharey=True)
        if len(rounds_to_plot) == 1:
            axes = [axes]

        for ax, r in zip(axes, rounds_to_plot):
            A  = reg.affinity_vector(anchor, end_round=r, window=window)
            A_sorted = A[sort_idx]
            colors   = [cmap_c(ntc[j]) for j in sort_idx]
            ax.bar(range(N), A_sorted, color=colors, width=1.0)
            ax.set_title(f"Round {r}")
            ax.set_xlabel("Node (sorted by cluster)")
            # cluster boundaries
            clusters = ntc[sort_idx]
            for b in np.where(np.diff(clusters))[0] + 1:
                ax.axvline(x=b - 0.5, color="black", lw=0.5, alpha=0.4)

        axes[0].set_ylabel(f"Affinity (window={window})")
        fig.suptitle(f"Affinity snapshots — Node {anchor} "
                     f"[cluster {ntc[anchor]}]", y=1.02)
        plt.tight_layout()
        prefix = save_prefix or f"{self.exp_name}_aff_node{anchor}"
        path   = f"outputs/{prefix}_snapshots.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[plot] -> {path}")
        plt.close(fig)