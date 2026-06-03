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
        """
        4-panel dashboard summarising what was learned during warmup.

        Panel A — Embedding precision (tp@5/10/15) over warmup rounds.
                Plateau marker shows where further warmup adds no value.
        Panel B — Intra vs inter cluster embedding distances + sep_ratio.
                Rising sep_ratio confirms embeddings are discriminating.
        Panel C — Warmup confidence distribution histogram.
                Shows coverage quality: how many pairs were well-observed.
        Panel D — d_tent distribution at warmup end, intra vs inter cluster.
                Shows the distance signal is already discriminative before
                any mixing begins.

        Parameters
        ----------
        v1_state_path : path to _v1_state.npz (needed for panels C and D).
                        Auto-detected from exp_name if None.
        """
        # ── Load warmup embedding log ──────────────────────────────────
        # Dense distance log (every round during warmup + buffer) → panels B
        dist_log = self.metrics.get('v1_emb_dist_log', [])

        # Sparse recall log (every 5 rounds) → panel A
        recall_log = (self.metrics.get('v1_emb_stats', None)
                    or self.metrics.get('v2_emb_stats', []))


        # Split into warmup-time entries and post-warmup entries
        warmup_rounds = (self.metrics.get('v2_warmup_rounds', None)
                        or self.metrics.get('v1_warmup_rounds', 25))
        confirm_start = self.metrics.get('v1_confirm_start', None)
        summary       = self.metrics.get('v1_summary', {})
        plateau_rnd   = summary.get('plateau_round', -1)

        warmup_emb = [e for e in emb_log if e['round'] < warmup_rounds]

        # ── Load v1 state (for C and D) ────────────────────────────────
        state = None
        if v1_state_path is None:
            candidate = f"outputs/{self.exp_name}_v1_state.npz"
            if os.path.exists(candidate):
                v1_state_path = candidate
        if v1_state_path and os.path.exists(v1_state_path):
            state = np.load(v1_state_path)

        ntc = self.node_to_cluster

        # ── Figure ─────────────────────────────────────────────────────
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        ax_A, ax_B = axes[0, 0], axes[0, 1]
        ax_C, ax_D = axes[1, 0], axes[1, 1]

        # ── Panel A: Embedding precision over warmup rounds ────────────
        if warmup_emb:
            rnds = np.array([e['round'] + 1 for e in warmup_emb])  # 1-indexed
            tp5  = np.array([e['tp_at_5']  for e in warmup_emb])
            tp10 = np.array([e['tp_at_10'] for e in warmup_emb])
            tp15 = np.array([e.get('tp_at_15', np.nan) for e in warmup_emb])

            ax_A.plot(rnds, tp5,  lw=2.0, color='#DD4444',
                    marker='o', ms=5, label='tp@5')
            ax_A.plot(rnds, tp10, lw=2.0, color='#4C72B0',
                    marker='s', ms=5, label='tp@10')
            ax_A.plot(rnds, tp15, lw=2.0, color='#44AA44',
                    marker='^', ms=5, label='tp@15')

            # Warmup phase markers
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
            ax_A.set_ylabel('Top-k precision')
            ax_A.set_title('A: Embedding precision during warmup')
            ax_A.legend(fontsize=8, loc='lower right')
            ax_A.grid(alpha=0.25)
        else:
            ax_A.text(0.5, 0.5, 'No warmup embedding log found',
                    ha='center', va='center', transform=ax_A.transAxes)
            ax_A.set_title('A: Embedding precision during warmup')

        # ── Panel B: Distance separation over warmup rounds ───────────
        if warmup_emb:
            intra = np.array([e.get('intra_mean', np.nan) for e in warmup_emb])
            inter = np.array([e.get('inter_mean', np.nan) for e in warmup_emb])
            sep   = np.array([e.get('sep_ratio',  np.nan) for e in warmup_emb])

            ax_B.plot(rnds, intra, lw=2.0, color='#4C72B0',
                    marker='o', ms=4, label='intra-cluster')
            ax_B.plot(rnds, inter, lw=2.0, color='#DD4444',
                    marker='o', ms=4, label='inter-cluster')
            ax_B.fill_between(rnds, intra, inter,
                            where=(inter >= intra),
                            alpha=0.12, color='#44AA44',
                            label='separation gap')
            ax_B.set_xlabel('Round')
            ax_B.set_ylabel('Mean embedding distance')
            ax_B.set_title('B: Intra vs inter cluster embedding distance')
            ax_B.legend(fontsize=8, loc='upper left')
            ax_B.grid(alpha=0.25)

            # Sep ratio on twin axis
            ax_B2 = ax_B.twinx()
            ax_B2.plot(rnds, sep, lw=1.5, color='#44AA44',
                    ls='--', alpha=0.7, label='sep ratio')
            ax_B2.set_ylabel('sep ratio (inter/intra)', color='#44AA44')
            ax_B2.tick_params(axis='y', labelcolor='#44AA44')
            ax_B2.legend(fontsize=8, loc='lower right')
        else:
            ax_B.text(0.5, 0.5, 'No warmup embedding log found',
                    ha='center', va='center', transform=ax_B.transAxes)
            ax_B.set_title('B: Intra vs inter cluster embedding distance')

        # ── Panel C: Confidence distribution histogram ─────────────────
        if state is not None:
            N    = state['confidence'].shape[0]
            conf = state['confidence']
            off  = ~np.eye(N, dtype=bool)
            conf_vals = conf[off]

            # Bin edges match natural confidence levels
            # with count_threshold=3: 0, 0.33, 0.67, 1.0
            bins = np.linspace(0, 1, 21)
            counts, edges = np.histogram(conf_vals, bins=bins)
            centers = (edges[:-1] + edges[1:]) / 2

            # Color bars by confidence level
            bar_colors = []
            for c in centers:
                if c < 0.01:
                    bar_colors.append('#DD4444')    # never seen — red
                elif c < 0.35:
                    bar_colors.append('#FF8844')    # low — orange
                elif c < 0.70:
                    bar_colors.append('#4C72B0')    # medium — blue
                else:
                    bar_colors.append('#44AA44')    # high — green

            ax_C.bar(centers, counts, width=(edges[1]-edges[0])*0.9,
                    color=bar_colors, edgecolor='white', lw=0.4)

            mean_conf = float(conf_vals.mean())
            zero_frac = float((conf_vals < 0.01).mean())
            full_frac = float((conf_vals >= 1.0).mean())
            ax_C.axvline(x=mean_conf, color='black', lw=1.5, ls='--',
                        label=f'mean={mean_conf:.3f}')

            # Annotations
            ax_C.text(0.97, 0.95,
                    f"never seen: {zero_frac:.1%}\nfull conf:  {full_frac:.1%}",
                    transform=ax_C.transAxes, ha='right', va='top',
                    fontsize=9, family='monospace',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

            ax_C.set_xlabel('Confidence')
            ax_C.set_ylabel('Directed pair count')
            ax_C.set_title('C: Warmup confidence distribution')
            ax_C.legend(fontsize=8)
            ax_C.grid(alpha=0.20, axis='y')
        else:
            ax_C.text(0.5, 0.5, 'v1_state.npz not found\n(needed for panels C and D)',
                    ha='center', va='center', transform=ax_C.transAxes,
                    fontsize=10, color='grey')
            ax_C.set_title('C: Warmup confidence distribution')

        # ── Panel D: d_tent distribution at warmup end ────────────────
        if state is not None and ntc is not None:
            d_tent = state['d_tent']   # [N, N]
            N      = d_tent.shape[0]
            off    = ~np.eye(N, dtype=bool)

            intra_d = []
            inter_d = []
            for i in range(N):
                for j in range(N):
                    if i == j:
                        continue
                    d = float(d_tent[i, j])
                    if not np.isfinite(d):
                        continue
                    if ntc[i] == ntc[j]:
                        intra_d.append(d)
                    else:
                        inter_d.append(d)

            # Shared bin range
            all_d   = intra_d + inter_d
            d_max   = float(np.percentile(all_d, 99)) if all_d else 5.0
            bins_d  = np.linspace(0, d_max, 40)

            ax_D.hist(intra_d, bins=bins_d, alpha=0.55, color='#4C72B0',
                    label=f'intra-cluster (n={len(intra_d):,})',
                    density=True)
            ax_D.hist(inter_d, bins=bins_d, alpha=0.55, color='#DD4444',
                    label=f'inter-cluster (n={len(inter_d):,})',
                    density=True)

            if intra_d and inter_d:
                overlap = float(np.mean(np.array(intra_d) <
                                        np.median(inter_d)))
                ax_D.text(0.97, 0.95,
                        f"intra median: {np.median(intra_d):.3f}\n"
                        f"inter median: {np.median(inter_d):.3f}\n"
                        f"intra < inter_median: {overlap:.1%}",
                        transform=ax_D.transAxes, ha='right', va='top',
                        fontsize=9, family='monospace',
                        bbox=dict(boxstyle='round', facecolor='white',
                                    alpha=0.7))

            ax_D.set_xlabel('d_tent (tent-weighted model distance)')
            ax_D.set_ylabel('Density')
            ax_D.set_title('D: Model distance at warmup end — intra vs inter cluster')
            ax_D.legend(fontsize=8)
            ax_D.grid(alpha=0.20, axis='y')
        else:
            if ntc is None:
                msg = 'node_to_cluster not found in metrics'
            else:
                msg = 'v1_state.npz not found'
            ax_D.text(0.5, 0.5, msg, ha='center', va='center',
                    transform=ax_D.transAxes, fontsize=10, color='grey')
            ax_D.set_title('D: Model distance at warmup end — intra vs inter cluster')

        # ── Save ───────────────────────────────────────────────────────
        fig.suptitle(f"Warmup quality — {self.exp_name}  "
                    f"(W={warmup_rounds})", fontsize=13, y=1.01)
        plt.tight_layout()
        path = f"outputs/{save_name or self.exp_name + '_warmup_quality.png'}"
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"[plot] -> {path}")
        plt.close(fig)