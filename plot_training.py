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
        self.soft_metrics    = (self.metrics.get("v1_soft_metrics", None) or self.metrics.get("v2_soft_metrics", []))
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
                            W_snap_path: str = None,
                            save_prefix: str = None):
        """
        Bar-chart snapshots of aggregation weight (or affinity proxy) for
        `anchor` node at several training rounds.

        Uses exact PAGANv1 aggregation weights (W_snapshots.npz) when available.
        Falls back to distance-based affinity proxy from shadow registry otherwise.

        Parameters
        ----------
        shadow_path   : path to _shadow.npz
        anchor        : node index whose outgoing weights to plot
        rounds_to_plot: list of 1-indexed round numbers to display.
                        If None, attempts to read from metrics summary
                        (v1_summary.snapshot_rounds), then falls back to
                        [25, 50, 100, 250, 500].
        window        : rolling window for fallback affinity proxy (ignored
                        when exact W snapshots are available)
        W_snap_path   : path to _W_snapshots.npz.  If None, auto-detected
                        by replacing '_shadow.npz' with '_W_snapshots.npz'.
        """
        reg = ShadowRegistry.load(shadow_path)
        ntc = node_to_cluster if node_to_cluster is not None \
            else self.node_to_cluster
        if ntc is None:
            print("[affinity_snapshots] node_to_cluster required.")
            return

        # ── Try to load exact aggregation weight snapshots ─────────────
        if W_snap_path is None:
            W_snap_path = shadow_path.replace('_shadow.npz', '_W_snapshots.npz')
        W_snaps = None
        if os.path.exists(W_snap_path):
            W_snaps = np.load(W_snap_path)
            print(f"[affinity_snapshots] Using exact W snapshots: {W_snap_path}")
        else:
            print(f"[affinity_snapshots] W_snapshots not found, "
                f"falling back to distance-based affinity proxy.")

        # ── Determine rounds to plot (1-indexed for display) ───────────
        if rounds_to_plot is None:
            # Try to read from saved summary first
            summary = self.metrics.get('v1_summary', {})
            saved_snaps = summary.get('snapshot_rounds', None)
            if saved_snaps is not None:
                # saved_snaps are 0-indexed rnd values → convert to 1-indexed
                all_display = sorted(r + 1 for r in saved_snaps)
            elif W_snaps is not None:
                # Infer from W_snaps keys: 'rnd_0', 'rnd_24', etc.
                all_display = sorted(
                    int(k.replace('rnd_', '')) + 1
                    for k in W_snaps.files
                    if k.startswith('rnd_')
                )
            else:
                all_display = [25, 50, 100, 250, 500]
            # Pick a manageable subset: up to 5 nicely spaced rounds
            if len(all_display) > 5:
                # Always include first and last, fill middle
                step = max(1, (len(all_display) - 2) // 3)
                rounds_to_plot = (
                    [all_display[0]]
                    + all_display[1:-1:step][:3]
                    + [all_display[-1]]
                )
                # Deduplicate while preserving order
                seen = set()
                rounds_to_plot = [r for r in rounds_to_plot
                                if not (r in seen or seen.add(r))]
            else:
                rounds_to_plot = all_display

        N        = reg.N
        sort_idx = np.argsort(ntc)
        n_panels = len(rounds_to_plot)

        num_clusters = int(ntc.max()) + 1
        cmap_c = cm.get_cmap("tab20", max(num_clusters, 2))

        fig, axes = plt.subplots(1, n_panels,
                                figsize=(5 * n_panels, 4),
                                sharey=True)
        if n_panels == 1:
            axes = [axes]

        using_exact = False   # track for y-label

        for ax, r_display in zip(axes, rounds_to_plot):
            r_internal = r_display - 1   # 0-indexed rnd

            # ── Get affinity / weight vector ──────────────────────────
            key = f'rnd_{r_internal}'
            if W_snaps is not None and key in W_snaps:
                # Exact aggregation weights — sum to 1
                A = W_snaps[key][anchor, :].astype(np.float32)
                using_exact = True
            else:
                # Fallback: distance-based proxy
                A = reg.affinity_vector(anchor,
                                        end_round=r_internal,
                                        window=window)

            # ── Plot bar chart sorted by cluster ──────────────────────
            A_sorted = A[sort_idx]
            # Replace NaN with 0 for plotting
            A_sorted = np.where(np.isnan(A_sorted), 0.0, A_sorted)
            colors = [cmap_c(ntc[j]) for j in sort_idx]

            ax.bar(range(N), A_sorted, color=colors, width=1.0)
            ax.set_title(f"Round {r_display}")
            ax.set_xlabel("Node (sorted by cluster)")

            # Cluster boundary lines
            clusters = ntc[sort_idx]
            for b in np.where(np.diff(clusters))[0] + 1:
                ax.axvline(x=b - 0.5, color="black", lw=0.5, alpha=0.4)

        # ── Y-axis label and title ─────────────────────────────────────
        y_label = ("Aggregation weight"
                if using_exact
                else f"Affinity proxy (window={window})")
        axes[0].set_ylabel(y_label)

        fig.suptitle(
            f"Affinity snapshots — Node {anchor} [cluster {ntc[anchor]}]",
            y=1.02)

        plt.tight_layout()
        prefix = save_prefix or f"{self.exp_name}_aff_node{anchor}"
        path   = f"outputs/{prefix}_snapshots.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[plot] -> {path}")
        plt.close(fig)

    def embedding_recall(self, m_values=(5, 10, 15, 20),
                        metric='recall',   # 'recall' | 'tp' | 'both'
                        save_name=None, title=None):
        emb_stats = (self.metrics.get('v1_emb_stats', None)
                    or self.metrics.get('v2_emb_stats', []))
        if not emb_stats:
            print("[embedding_recall] No emb_stats found.")
            return

        # Filter to entries that have recall keys
        stats = [e for e in emb_stats if f'recall_at_{m_values[0]}' in e]
        if not stats:
            print("[embedding_recall] No recall data — old tp@k format only.")
            return

        rnds   = np.array([e['round'] + 1 for e in stats])
        colors = {'5': '#DD4444', '10': '#4C72B0',
                '15': '#44AA44', '20': '#AA44AA'}

        fig, ax = plt.subplots(figsize=(13, 5))

        for m in m_values:
            c = colors.get(str(m), '#888888')
            if metric in ('recall', 'both'):
                vals = np.array([e.get(f'recall_at_{m}', np.nan)
                                for e in stats])
                ax.plot(rnds, vals, lw=2.0, color=c,
                        marker='o', ms=4,
                        label=f'recall@{m}',
                        ls='-')
            if metric in ('tp', 'both'):
                vals = np.array([e.get(f'tp_at_{m}', np.nan)
                                for e in stats])
                ax.plot(rnds, vals, lw=1.5, color=c,
                        marker='s', ms=3,
                        label=f'tp@{m}',
                        ls='--', alpha=0.7)

        # Warmup boundary
        warmup_end = (self.metrics.get('v2_warmup_rounds', None)
                    or self.metrics.get('v1_warmup_rounds', None))
        if warmup_end:
            ax.axvline(x=warmup_end, color='black', lw=1.2,
                    ls='--', alpha=0.5,
                    label=f'warmup end (r={warmup_end})')

        # Plateau
        summary = self.metrics.get('v1_summary', {})
        plateau = summary.get('plateau_round', -1)
        if plateau > 0:
            ax.axvline(x=plateau + 1, color='purple', lw=1.5,
                    ls=':', alpha=0.8,
                    label=f'plateau (r={plateau + 1})')

        ax.set_ylim(-0.02, 1.08)
        ax.set_xlabel('Round')
        y_label = {'recall': 'Recall (friends found / total friends)',
                'tp':     'Precision (friends found / m)',
                'both':   'Recall (—) / Precision (---)'}
        ax.set_ylabel(y_label.get(metric, 'Score'))
        ax.set_title(title or f"Embedding recall/precision — {self.exp_name}")
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(alpha=0.25)

        plt.tight_layout()
        path = f"outputs/{save_name or self.exp_name + '_emb_recall.png'}"
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"[plot] -> {path}")
        plt.close(fig)

    # ------------------------------------------------------------------
    # 4. Warmup quality dashboard
    # ------------------------------------------------------------------
    def warmup_quality(self,
                    v1_state_path: str = None,
                    save_name: str = None):

        # ── Load logs ─────────────────────────────────────────────────
        # Dense distance log → Panel B (every round during warmup + buffer)
        dist_log = self.metrics.get('v1_emb_dist_log', [])

        # Sparse recall/precision log → Panel A (every 5 rounds)
        recall_log = (self.metrics.get('v1_emb_stats', None)
                    or self.metrics.get('v2_emb_stats', []))

        warmup_rounds = (self.metrics.get('v2_warmup_rounds', None)
                        or self.metrics.get('v1_warmup_rounds', 25))
        confirm_start = self.metrics.get('v1_confirm_start', None)
        summary       = self.metrics.get('v1_summary', {})
        plateau_rnd   = summary.get('plateau_round', -1)

        # Filter each log to warmup window only
        # dist_log: include a few rounds post-warmup for continuity
        dist_warmup   = [e for e in dist_log
                        if e['round'] < warmup_rounds + 10]
        recall_warmup = [e for e in recall_log
                        if e['round'] < warmup_rounds]

        # ── Load v1 state ─────────────────────────────────────────────
        state = None
        if v1_state_path is None:
            candidate = f"outputs/{self.exp_name}_v1_state.npz"
            if os.path.exists(candidate):
                v1_state_path = candidate
        if v1_state_path and os.path.exists(v1_state_path):
            state = np.load(v1_state_path)

        ntc = self.node_to_cluster

        # ── Figure ────────────────────────────────────────────────────
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        ax_A, ax_B = axes[0, 0], axes[0, 1]
        ax_C, ax_D = axes[1, 0], axes[1, 1]

        # ── Panel A: Recall over warmup rounds ────────────────────────
        if recall_warmup:
            rnds_A = np.array([e['round'] + 1 for e in recall_warmup])

            has_recall = 'recall_at_5' in recall_warmup[0]
            has_tp     = 'tp_at_5'     in recall_warmup[0]

            if has_recall:
                for m, color, marker in [
                        (5,  '#DD4444', 'o'),
                        (10, '#4C72B0', 's'),
                        (20, '#44AA44', '^')]:
                    vals = np.array([e.get(f'recall_at_{m}', np.nan)
                                    for e in recall_warmup])
                    ax_A.plot(rnds_A, vals, lw=2.0, color=color,
                            marker=marker, ms=5,
                            label=f'recall@{m}')
                ax_A.set_ylabel('Recall (friends found / total friends)')
                ax_A.set_title('A: Embedding recall during warmup')

            elif has_tp:
                for key, color, marker, lbl in [
                        ('tp_at_5',  '#DD4444', 'o', 'tp@5'),
                        ('tp_at_10', '#4C72B0', 's', 'tp@10'),
                        ('tp_at_20', '#44AA44', '^', 'tp@20')]:
                    vals = np.array([e.get(key, np.nan)
                                    for e in recall_warmup])
                    ax_A.plot(rnds_A, vals, lw=2.0, color=color,
                            marker=marker, ms=5, label=lbl)
                ax_A.set_ylabel('Precision')
                ax_A.set_title('A: Embedding precision during warmup')

            else:
                ax_A.text(0.5, 0.5, 'No recall/precision keys in log',
                        ha='center', va='center',
                        transform=ax_A.transAxes, fontsize=9, color='grey')
                ax_A.set_title('A: Embedding recall during warmup')

            # Phase markers
            if confirm_start is not None and confirm_start < warmup_rounds:
                ax_A.axvline(x=confirm_start, color='orange', lw=1.2,
                            ls='--', alpha=0.7,
                            label=f'confirm start (r={confirm_start})')
            ax_A.axvline(x=warmup_rounds, color='black', lw=1.2,
                        ls='--', alpha=0.6,
                        label=f'warmup end (r={warmup_rounds})')
            if plateau_rnd > 0:
                ax_A.axvline(x=plateau_rnd + 1, color='purple', lw=1.5,
                            ls=':', alpha=0.8,
                            label=f'plateau (r={plateau_rnd + 1})')

            ax_A.set_ylim(-0.02, 1.08)
            ax_A.set_xlabel('Round')
            ax_A.legend(fontsize=8, loc='lower right')
            ax_A.grid(alpha=0.25)

        else:
            ax_A.text(0.5, 0.5,
                    'No recall log found\n(v1_emb_stats missing)\n'
                    'Check monitor_embeddings() saves recall_at_m keys',
                    ha='center', va='center',
                    transform=ax_A.transAxes, fontsize=10, color='grey')
            ax_A.set_title('A: Embedding recall during warmup')

        # ── Panel B: Distance separation (dense log) ──────────────────
        if dist_warmup:
            rnds_B = np.array([e['round'] + 1 for e in dist_warmup])
            intra  = np.array([e.get('intra_mean', np.nan) for e in dist_warmup])
            inter  = np.array([e.get('inter_mean', np.nan) for e in dist_warmup])
            sep    = np.array([e.get('sep_ratio',  np.nan) for e in dist_warmup])

            ax_B.plot(rnds_B, intra, lw=2.0, color='#4C72B0',
                    marker='o', ms=3, label='intra-cluster')
            ax_B.plot(rnds_B, inter, lw=2.0, color='#DD4444',
                    marker='o', ms=3, label='inter-cluster')
            ax_B.fill_between(rnds_B, intra, inter,
                            where=(inter >= intra),
                            alpha=0.12, color='#44AA44',
                            label='separation gap')

            # Warmup end marker
            ax_B.axvline(x=warmup_rounds, color='black', lw=1.2,
                        ls='--', alpha=0.6,
                        label=f'warmup end (r={warmup_rounds})')

            ax_B.set_xlabel('Round')
            ax_B.set_ylabel('Mean embedding distance')
            ax_B.set_title('B: Intra vs inter cluster embedding distance')
            ax_B.legend(fontsize=8, loc='upper left')
            ax_B.grid(alpha=0.25)

            # Sep ratio on twin axis
            ax_B2 = ax_B.twinx()
            ax_B2.plot(rnds_B, sep, lw=1.5, color='#44AA44',
                    ls='--', alpha=0.7, label='sep ratio')
            ax_B2.set_ylabel('sep ratio (inter/intra)', color='#44AA44')
            ax_B2.tick_params(axis='y', labelcolor='#44AA44')
            ax_B2.legend(fontsize=8, loc='lower right')
        else:
            ax_B.text(0.5, 0.5,
                    'No distance log found\n(v1_emb_dist_log)\n'
                    'Add metrics[\'v1_emb_dist_log\'] = '
                    'pagan_v1_protocol.emb_dist_log\nto main.py save block',
                    ha='center', va='center', transform=ax_B.transAxes,
                    fontsize=9, color='grey')
            ax_B.set_title('B: Intra vs inter cluster embedding distance')

        # ── Panel C: Recall saturation curve ──────────────────────────
        recall_sat = self.metrics.get('v1_recall_saturation', [])
        if recall_sat:
            m_vals = np.arange(1, len(recall_sat) + 1)
            ax_C.plot(m_vals, recall_sat, lw=2.0, color='#4C72B0')
            ax_C.axhline(y=1.0, color='black', lw=0.8, ls='--', alpha=0.4)

            # Mark m* where recall first reaches 0.999
            m_star = next((i + 1 for i, v in enumerate(recall_sat)
                        if v >= 0.999), None)
            if m_star:
                ax_C.axvline(x=m_star, color='#DD4444', lw=1.5,
                            ls='--', alpha=0.8,
                            label=f'm* = {m_star} (recall ≈ 1.0)')
                ax_C.scatter([m_star], [recall_sat[m_star - 1]],
                            color='#DD4444', s=60, zorder=5)

            ax_C.set_xlabel('m (embedding top-m)')
            ax_C.set_ylabel('Mean recall@m')
            ax_C.set_title('C: Recall saturation curve at warmup end')
            ax_C.set_ylim(-0.02, 1.08)
            ax_C.legend(fontsize=8)
            ax_C.grid(alpha=0.25)
        else:
            ax_C.text(0.5, 0.5,
                    'No recall saturation data\n'
                    '(v1_recall_saturation missing from metrics)\n'
                    'Call compute_recall_saturation() at warmup end',
                    ha='center', va='center',
                    transform=ax_C.transAxes, fontsize=9, color='grey')
            ax_C.set_title('C: Recall saturation curve at warmup end')

        # ── Panel D: Affinity gap over rounds ─────────────────────────
        W_snap_path = None
        if v1_state_path:
            W_snap_path = v1_state_path.replace(
                '_v1_state.npz', '_W_snapshots.npz')

        if W_snap_path and os.path.exists(W_snap_path) and ntc is not None:
            rnds_gap, gaps, gaps_std = TrainingPlotter._compute_affinity_gap(
                W_snap_path, ntc)

            if rnds_gap:
                rnds_arr = np.array(rnds_gap)
                gaps_arr = np.array(gaps)
                std_arr  = np.array(gaps_std)

                ax_D.plot(rnds_arr, gaps_arr, lw=2.0, color='#2166AC',
                        marker='o', ms=3)
                ax_D.fill_between(rnds_arr,
                                gaps_arr - std_arr,
                                gaps_arr + std_arr,
                                alpha=0.15, color='#2166AC')
                ax_D.axhline(y=0, color='black', lw=0.8, ls='--',
                            alpha=0.4, label='gap = 0')

                if warmup_rounds:
                    ax_D.axvline(x=warmup_rounds, color='black', lw=1.2,
                                ls='--', alpha=0.6,
                                label=f'warmup end (r={warmup_rounds})')

                ax_D.set_xlabel('Round')
                ax_D.set_ylabel('Mean affinity gap (same − diff cluster)')
                ax_D.set_title('D: Affinity gap over rounds')
                ax_D.legend(fontsize=8)
                ax_D.grid(alpha=0.25)
        else:
            ax_D.text(0.5, 0.5,
                    'W_snapshots not found\n'
                    '(auto-detected from v1_state_path)\n'
                    'Check W_snapshots are saved in main.py',
                    ha='center', va='center',
                    transform=ax_D.transAxes, fontsize=9, color='grey')
            ax_D.set_title('D: Affinity gap over rounds')

        # ── Save ──────────────────────────────────────────────────────
        fig.suptitle(f"Warmup quality — {self.exp_name}  "
                    f"(W={warmup_rounds})", fontsize=13, y=1.01)
        plt.tight_layout()
        path = f"outputs/{save_name or self.exp_name + '_warmup_quality.png'}"
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"[plot] -> {path}")
        plt.close(fig)

    @staticmethod
    def _compute_affinity_gap(W_snap_path: str,
                            ntc: np.ndarray) -> tuple:
        """
        Compute mean affinity gap per snapshot round.
        gap = mean W[i,j] for j in same cluster
            - mean W[i,j] for j in different cluster
        averaged across all nodes.
        Returns (rounds_display, gaps, gaps_std)
        """
        if not os.path.exists(W_snap_path):
            return [], [], []

        W_snaps = np.load(W_snap_path)
        rounds  = sorted(int(k.replace('rnd_', '')) + 1
                        for k in W_snaps.files
                        if k.startswith('rnd_'))
        gaps     = []
        gaps_std = []

        for r in rounds:
            W = W_snaps[f'rnd_{r - 1}']   # 0-indexed key
            N = W.shape[0]
            node_gaps = []
            for i in range(N):
                same = [W[i, j] for j in range(N)
                        if j != i and ntc[j] == ntc[i]]
                diff = [W[i, j] for j in range(N)
                        if ntc[j] != ntc[i]]
                if same and diff:
                    node_gaps.append(
                        float(np.mean(same)) - float(np.mean(diff)))
            if node_gaps:
                gaps.append(float(np.mean(node_gaps)))
                gaps_std.append(float(np.std(node_gaps)))
            else:
                gaps.append(0.0)
                gaps_std.append(0.0)

        return rounds, gaps, gaps_std