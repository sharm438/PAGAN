"""
compare_protocols.py
====================
Standalone script for comparing multiple P2PL protocols across
heterogeneity setups.

Metrics are extracted from raw per-node accuracy lists
(metrics['local_acc'] or metrics['global_acc']) and summary
stats are computed on the fly — consistent with utils.print_acc_values().

Usage
-----
1. Define RUNS using make_runs() for clean template-based file naming.
2. Call print_summary_table() to verify numbers before plotting.
3. Call final_accuracy_candles() and accuracy_over_rounds() to plot.
"""

import json
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Patch

# ── Colour palette ─────────────────────────────────────────────────────
PROTOCOL_COLORS = {
    'PAGANv1':      '#2166AC',
    'PAGANv2':      '#4DAC26',
    'FedSPD':       '#D6604D',
    'Federico':     '#8073AC',
    'L2C':          '#F4A582',
    'Isolated':     '#AAAAAA',
    'k-Regular':    '#BABABA',
}

SETUP_ORDER  = ['dir0.1', 'patho1', 'patho2', 'patho3', 'dir100']
SETUP_LABELS = {
    'dir0.1': 'Dir(0.1)',
    'patho1': 'Patho-1',
    'patho2': 'Patho-2',
    'patho3': 'Patho-3',
    'dir100': 'Dir(100)',
}


# ──────────────────────────────────────────────────────────────────────
# File name template helper
# ──────────────────────────────────────────────────────────────────────

def make_runs(suffix: str,
              prefix: str = '',
              setups: list = None,
              out_dir: str = 'outputs',
              ext: str = '_metrics.json') -> dict:
    """
    Generate a {setup: path} dict from a naming template.

    Pattern: {out_dir}/{prefix}{setup}{suffix}{ext}

    Examples
    --------
    make_runs('_federico_eps0.5')
    # → {'dir0.1': 'outputs/dir0.1_federico_eps0.5_metrics.json', ...}

    make_runs('_v1_inv_sq_smart')
    # → {'dir0.1': 'outputs/dir0.1_v1_inv_sq_smart_metrics.json', ...}

    make_runs('_spd_s20_ckc20')
    # → {'dir0.1': 'outputs/dir0.1_spd_s20_ckc20_metrics.json', ...}
    """
    setups = setups or SETUP_ORDER
    return {
        setup: os.path.join(out_dir, f"{prefix}{setup}{suffix}{ext}")
        for setup in setups
    }


# ──────────────────────────────────────────────────────────────────────
# Stats computation (mirrors utils.print_acc_values)
# ──────────────────────────────────────────────────────────────────────

def compute_stats(accs: list) -> dict:
    """
    Compute Min / Low20% / Mid60% / Top20% / Max / Avg
    from a list of per-node accuracy values.
    Matches utils.print_acc_values() exactly.
    """
    valid = [a for a in accs if a is not None and not math.isnan(a)]
    if not valid:
        return {k: None for k in
                ['min', 'low20', 'mid60', 'top20', 'max', 'avg']}

    s    = sorted(valid)
    n    = len(s)
    c_lo = int(n * 0.2)
    c_hi = int(n * 0.8)

    bot  = s[:c_lo]
    mid  = s[c_lo:c_hi]
    top  = s[c_hi:]

    return {
        'min':   s[0],
        'low20': sum(bot) / len(bot) if bot else s[0],
        'mid60': sum(mid) / len(mid) if mid else sum(s) / n,
        'top20': sum(top) / len(top) if top else s[-1],
        'max':   s[-1],
        'avg':   sum(valid) / len(valid),
    }


# ──────────────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────────────

def load_metrics(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Not found: {path}")
    with open(path) as f:
        return json.load(f)


def _get_acc_key(metrics: dict) -> str:
    """Return the accuracy key to use: prefer local_acc, fall back to global_acc."""
    if 'local_acc' in metrics and metrics['local_acc']:
        return 'local_acc'
    if 'global_acc' in metrics and metrics['global_acc']:
        return 'global_acc'
    raise KeyError("Neither 'local_acc' nor 'global_acc' found in metrics.")


def extract_final_round_stats(metrics: dict,
                               target_display_rnd: int = 500) -> dict:
    """
    Extract accuracy stats from the last saved eval round.

    target_display_rnd : 1-indexed expected final round (default 500).
    Warns if the last saved round doesn't match.

    Returns dict with: round, min, low20, mid60, top20, max, avg
    """
    acc_key = _get_acc_key(metrics)
    rounds  = metrics.get('round', [])
    accs    = metrics[acc_key]

    if not accs:
        raise ValueError(f"Empty {acc_key} in metrics.")

    last_accs = accs[-1]
    last_rnd  = rounds[-1] if rounds else None

    if last_rnd is not None and last_rnd != target_display_rnd:
        print(f"  WARNING: expected round {target_display_rnd}, "
              f"last saved round is {last_rnd}")

    stats = compute_stats(last_accs)
    stats['round'] = last_rnd
    return stats


def extract_all_rounds(metrics: dict) -> dict:
    """
    Extract per-round summary stats for time-series plots.
    Returns: {rounds, min, avg, top20}  (lists, one value per eval round)
    """
    acc_key = _get_acc_key(metrics)
    rounds  = metrics.get('round', [])
    accs    = metrics[acc_key]

    mins   = []
    avgs   = []
    top20s = []

    for acc_list in accs:
        s = compute_stats(acc_list)
        mins.append(s['min'])
        avgs.append(s['avg'])
        top20s.append(s['top20'])

    return dict(rounds=list(rounds), min=mins, avg=avgs, top20=top20s)


def load_all_runs(runs: dict,
                  target_display_rnd: int = 500) -> dict:
    """
    Load final-round stats for all protocol/setup combinations.
    Returns {protocol: {setup: stats_dict}}
    """
    results = {}
    for proto, setup_paths in runs.items():
        results[proto] = {}
        for setup, path in setup_paths.items():
            try:
                m     = load_metrics(path)
                stats = extract_final_round_stats(m, target_display_rnd)
                results[proto][setup] = stats
                print(f"  [{proto:20s}][{setup:8s}] "
                      f"rnd={stats['round']}  "
                      f"min={stats['min']:.2f}  "
                      f"avg={stats['avg']:.2f}")
            except Exception as e:
                print(f"  [{proto:20s}][{setup:8s}] ERROR: {e}")
                results[proto][setup] = None
    return results


def load_all_runs_timeseries(runs: dict) -> dict:
    """Load full time series for all protocol/setup combinations."""
    results = {}
    for proto, setup_paths in runs.items():
        results[proto] = {}
        for setup, path in setup_paths.items():
            try:
                m = load_metrics(path)
                results[proto][setup] = extract_all_rounds(m)
            except Exception as e:
                print(f"  [{proto}][{setup}] timeseries ERROR: {e}")
                results[proto][setup] = None
    return results


# ──────────────────────────────────────────────────────────────────────
# Print summary table
# ──────────────────────────────────────────────────────────────────────

def print_summary_table(runs: dict,
                         setups: list = None,
                         target_display_rnd: int = 500,
                         metrics: list = None):
    """
    Print a formatted table of final-round accuracy stats.
    metrics : stat keys to show — default ['min', 'avg']
    """
    setups    = setups or SETUP_ORDER
    metrics   = metrics or ['min', 'avg']
    protocols = list(runs.keys())
    results   = load_all_runs(runs, target_display_rnd)

    col_w = 9
    m_w   = col_w * len(metrics)

    print("\n" + "─" * (22 + m_w * len(setups)))
    header = f"{'Protocol':<22}" + "".join(
        f"{SETUP_LABELS.get(s, s):^{m_w}}" for s in setups)
    print(header)
    sub = f"{'':22}" + "".join(
        f"{'  '.join(m.upper() for m in metrics):^{m_w}}"
        for _ in setups)
    print(sub)
    print("─" * (22 + m_w * len(setups)))

    for proto in protocols:
        row = f"{proto:<22}"
        for setup in setups:
            stats = results.get(proto, {}).get(setup, None)
            if stats is None:
                row += " " * m_w
            else:
                for m in metrics:
                    val = stats.get(m, None)
                    row += (f"{val:>{col_w}.2f}" if val is not None
                            else f"{'N/A':>{col_w}}")
        print(row)

    print("─" * (22 + m_w * len(setups)))


# ──────────────────────────────────────────────────────────────────────
# Plot 1 — Final accuracy candle comparison
# ──────────────────────────────────────────────────────────────────────

def final_accuracy_candles(runs: dict,
                            setups: list = None,
                            target_display_rnd: int = 500,
                            metric: str = 'full_candle',
                            save_path: str = 'outputs/compare_final_candle.png',
                            title: str = None):
    """
    One panel per setup. Within each panel, one candlestick per protocol.

    metric='full_candle'  : min→max range, low20→top20 inner box, avg line
    metric='min_and_avg'  : simpler bar (0→avg) + min marker
    """
    setups    = setups or SETUP_ORDER
    protocols = list(runs.keys())
    n_setups  = len(setups)
    n_proto   = len(protocols)

    print(f"\nLoading final-round stats (round={target_display_rnd})...")
    results = load_all_runs(runs, target_display_rnd)

    fig, axes = plt.subplots(1, n_setups,
                              figsize=(3.2 * n_setups, 5.5),
                              sharey=False)
    if n_setups == 1:
        axes = [axes]

    bar_w = 0.70 / max(n_proto, 1)

    for ax_idx, (ax, setup) in enumerate(zip(axes, setups)):
        for p_idx, proto in enumerate(protocols):
            stats = results.get(proto, {}).get(setup, None)
            if stats is None:
                continue

            color = PROTOCOL_COLORS.get(proto, f'C{p_idx}')
            x     = p_idx

            if metric == 'full_candle':
                v = {k: stats[k] for k in
                     ['min', 'low20', 'mid60', 'top20', 'max']
                     if stats.get(k) is not None}
                if len(v) < 3:
                    continue
                # Outer range
                ax.bar(x, v['max'] - v['min'], bottom=v['min'],
                       width=bar_w, color=color, alpha=0.20,
                       edgecolor=color, linewidth=0.8)
                # Inner box
                ax.bar(x, v['top20'] - v['low20'], bottom=v['low20'],
                       width=bar_w, color=color, alpha=0.55,
                       edgecolor=color, linewidth=0.8)
                # Avg line
                if stats.get('avg') is not None:
                    ax.hlines(stats['avg'],
                              x - bar_w/2, x + bar_w/2,
                              colors=color, linewidths=2.2, zorder=5)
                # Min marker
                ax.scatter([x], [v['min']], marker='v',
                           color=color, s=45, zorder=6)
            else:
                if stats.get('avg') is None:
                    continue
                ax.bar(x, stats['avg'], width=bar_w,
                       color=color, alpha=0.65,
                       edgecolor=color, linewidth=0.8)
                if stats.get('min') is not None:
                    ax.scatter([x], [stats['min']], marker='v',
                               color=color, s=50, zorder=5)
                    ax.hlines(stats['min'],
                              x - bar_w/2, x + bar_w/2,
                              colors=color, linewidths=1.5,
                              linestyles='--', alpha=0.7)

        ax.set_title(SETUP_LABELS.get(setup, setup), fontsize=11)
        ax.set_xticks(range(n_proto))
        ax.set_xticklabels(protocols, rotation=30, ha='right', fontsize=8)
        ax.set_ylim(0, 105)
        ax.yaxis.set_major_locator(plt.MultipleLocator(10))
        ax.grid(axis='y', alpha=0.22)
        if ax_idx == 0:
            ax.set_ylabel('Local test accuracy (%)')

    handles = [Patch(facecolor=PROTOCOL_COLORS.get(p, f'C{i}'),
                     alpha=0.65, label=p)
               for i, p in enumerate(protocols)]
    handles.append(plt.Line2D([0], [0], marker='v', color='black',
                               linestyle='None', markersize=6,
                               label='▼ = min accuracy'))
    if metric == 'full_candle':
        handles.append(plt.Line2D([0], [0], color='black',
                                   lw=2.2, label='— = mean'))
    fig.legend(handles=handles, loc='lower center',
               ncol=min(len(handles), 6),
               bbox_to_anchor=(0.5, -0.02), fontsize=9)

    fig.suptitle(title or f'Final accuracy — round {target_display_rnd}',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[compare] -> {save_path}")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# Plot 2 — Accuracy over rounds (downsampled)
# ──────────────────────────────────────────────────────────────────────
def accuracy_candles_over_rounds(runs: dict,
                                  setups: list = None,
                                  checkpoints: list = None,
                                  target_display_rnd: int = 500,
                                  save_path: str = 'outputs/compare_candles_over_rounds.png',
                                  title: str = None):
    """
    Grouped candlestick bars at selected round checkpoints.
    One panel per setup. Within each panel, x-axis = checkpoints,
    grouped bars = protocols. Shows full distribution evolution over time.

    checkpoints : 1-indexed round numbers to show.
                  Default: [50, 100, 200, 300, 500]
    """
    setups      = setups or SETUP_ORDER
    protocols   = list(runs.keys())
    checkpoints = checkpoints or [50, 100, 200, 300, 500]
    n_setups    = len(setups)
    n_proto     = len(protocols)
    n_checks    = len(checkpoints)

    # Load all timeseries
    print("\nLoading timeseries for candle plot...")
    ts = load_all_runs_timeseries(runs)

    # For each protocol/setup/checkpoint, extract stats
    # ts gives us per-round raw acc lists — need to find the closest
    # saved round to each checkpoint
    def get_stats_at_checkpoint(proto, setup, chk):
        data = ts.get(proto, {}).get(setup, None)
        if not data or not data.get('rounds'):
            return None
        rounds = data['rounds']
        # Find closest saved round to checkpoint
        idx = min(range(len(rounds)),
                  key=lambda i: abs(rounds[i] - chk))
        # We need per-node accs at this round — reload from metrics
        return idx  # return index, use raw accs below

    # Re-load raw per-node accs for stats computation
    raw = {}
    for proto, setup_paths in runs.items():
        raw[proto] = {}
        for setup, path in setup_paths.items():
            try:
                m       = load_metrics(path)
                acc_key = _get_acc_key(m)
                rounds  = m.get('round', [])
                accs    = m[acc_key]
                raw[proto][setup] = {'rounds': rounds, 'accs': accs}
            except Exception as e:
                print(f"  [{proto}][{setup}] ERROR: {e}")
                raw[proto][setup] = None

    def get_stats_at(proto, setup, chk):
        d = raw.get(proto, {}).get(setup, None)
        if d is None:
            return None
        rounds = d['rounds']
        accs   = d['accs']
        if not rounds:
            return None
        idx = min(range(len(rounds)),
                  key=lambda i: abs(rounds[i] - chk))
        return compute_stats(accs[idx])

    # ── Plot ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, n_setups,
                              figsize=(max(3 * n_checks, 8) * n_setups / 3,
                                       5.5),
                              sharey=False)
    if n_setups == 1:
        axes = [axes]

    bar_w     = 0.4 / n_proto
    spacing = 0.5 / max(n_proto, 1)   
    x_centers = np.arange(n_checks)

    for ax_idx, (ax, setup) in enumerate(zip(axes, setups)):
        for p_idx, proto in enumerate(protocols):
            color  = PROTOCOL_COLORS.get(proto, f'C{p_idx}')
            offset = (p_idx - (n_proto - 1) / 2) * spacing

            for c_idx, chk in enumerate(checkpoints):
                stats = get_stats_at(proto, setup, chk)
                if stats is None:
                    continue

                x = x_centers[c_idx] + offset

                # Outer range min→max
                if stats['min'] is not None and stats['max'] is not None:
                    ax.bar(x, stats['max'] - stats['min'],
                           bottom=stats['min'],
                           width=bar_w * 0.9, color=color, alpha=0.18,
                           edgecolor=color, linewidth=0.6)
                # Inner box low20→top20
                if stats['low20'] is not None and stats['top20'] is not None:
                    ax.bar(x, stats['top20'] - stats['low20'],
                           bottom=stats['low20'],
                           width=bar_w * 0.9, color=color, alpha=0.55,
                           edgecolor=color, linewidth=0.6)
                # Avg line
                if stats['avg'] is not None:
                    ax.hlines(stats['avg'],
                              x - bar_w*0.45, x + bar_w*0.45,
                              colors=color, linewidths=2.0, zorder=5)
                # Min marker
                if stats['min'] is not None:
                    ax.scatter([x], [stats['min']], marker='v',
                               color=color, s=30, zorder=6)

        ax.set_title(SETUP_LABELS.get(setup, setup), fontsize=11)
        ax.set_xticks(x_centers)
        ax.set_xticklabels([str(c) for c in checkpoints], fontsize=9)
        ax.set_xlabel('Round')
        ax.set_ylim(0, 105)
        ax.yaxis.set_major_locator(plt.MultipleLocator(10))
        ax.grid(axis='y', alpha=0.22)
        if ax_idx == 0:
            ax.set_ylabel('Local test accuracy (%)')

    handles = [Patch(facecolor=PROTOCOL_COLORS.get(p, f'C{i}'),
                     alpha=0.65, label=p)
               for i, p in enumerate(protocols)]
    handles.append(plt.Line2D([0], [0], marker='v', color='black',
                               linestyle='None', markersize=5,
                               label='▼ = min'))
    handles.append(plt.Line2D([0], [0], color='black', lw=2.0,
                               label='— = mean'))
    fig.legend(handles=handles, loc='lower center',
               ncol=min(len(handles), 7),
               bbox_to_anchor=(0.5, -0.02), fontsize=9)

    fig.suptitle(title or 'Accuracy over rounds — all protocols',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[compare] -> {save_path}")
    plt.close(fig)

# ──────────────────────────────────────────────────────────────────────
# Example usage
# ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    # # ── Federico parameter comparison ─────────────────────────────────
    # RUNS_FEDERICO = {
    #     'Federico-eps0.1': make_runs('_federico_eps0.1'),
    #     'Federico-eps0.3': make_runs('_federico'),
    #     'Federico-eps0.5': make_runs('_federico_eps0.5'),
    # }

    # print("=== Federico parameter sweep ===")
    # print_summary_table(RUNS_FEDERICO, metrics=['min', 'avg'])
    # final_accuracy_candles(
    #     RUNS_FEDERICO,
    #     metric='full_candle',
    #     save_path='outputs/compare_federico_params.png',
    #     title='Federico — parameter comparison')
    # accuracy_candles_over_rounds(
    #     RUNS_FEDERICO,
    #     checkpoints=[50, 100, 200, 300, 400, 500],
    #     save_path='outputs/compare_federico_rounds.png')

    # # ── FedSPD parameter comparison ────────────────────────────────────
    # RUNS_FEDSPD = {
    #     'FedSPD-s10': make_runs('_spd_s10_ckc20'),
    #     'FedSPD-s20': make_runs('_spd_s20_ckc20'),
    # }

    # print("\n=== FedSPD parameter sweep ===")
    # print_summary_table(RUNS_FEDSPD, metrics=['min', 'avg'])
    # final_accuracy_candles(
    #     RUNS_FEDSPD,
    #     metric='full_candle',
    #     save_path='outputs/compare_fedspd_params.png',
    #     title='FedSPD — parameter comparison')

    # # ── Main protocol comparison (fill in best configs after sweep) ────
    RUNS_MAIN = {
        'PAGANv1':  make_runs('_v1_0602'),
        #'FedSPD':   make_runs('_spd_s20_ckc20'),       # update after sweep
        'Federico': make_runs('_federico'),      # update after sweep
        #'Isolated': make_runs('_isolated'),
    }

    print("\n=== Main protocol comparison ===")
    print_summary_table(RUNS_MAIN, metrics=['min', 'avg'])
    # final_accuracy_candles(
    #     RUNS_MAIN,
    #     metric='full_candle',
    #     save_path='outputs/compare_main.png',
    #     title='Protocol comparison — round 500')
    accuracy_candles_over_rounds(
        RUNS_MAIN,
        save_path='outputs/compare_main_rounds.png')
