#!/usr/bin/env python3
"""
Visualization for the noise-robustness experiment.

Reads ``noise_experiment_results.json`` and produces:
  * Robustness curves (metric vs FP rate, one line per FN level)
  * Heatmaps (FP x FN → metric, one per strategy)
  * Comparison CSV / LaTeX table at representative noise levels

Usage:
    python experiments/plot_noise_results.py [--input-file path] [--output-dir path]
"""

import os
import sys
import json
import csv
import argparse
from collections import defaultdict
from typing import Dict, List, Any, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Key metrics to plot
METRICS = [
    ("protected_value_mean", "Protected Value ($)"),
    ("value_preserved_mean", "Value Preserved ($)"),
    ("compromised_assets_count_mean", "Compromised Assets"),
    ("attack_success_rate_mean", "Attack Success Rate (%)"),
    ("roi_mean", "ROI (%)"),
    ("lost_value_mean", "Lost Value ($)"),
]

# Representative noise levels for the comparison table
TABLE_LEVELS = [
    {"label": "Low",    "fp": 0.4, "fn": 0.1},
    {"label": "Medium", "fp": 0.6, "fn": 0.2},
    {"label": "High",   "fp": 0.9, "fn": 0.3},
]


def load_results(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


# ===================================================================
# 1. Robustness curves
# ===================================================================

def plot_robustness_curves(data: Dict, output_dir: str, corr: float = 0.0):
    """
    For each metric, plot strategy lines vs FP rate.
    Separate curve sets for each FN rate level.
    Filters to a single sensor_correlation value for clarity.
    """
    strategy_names = data["strategy_names"]
    configs = [c for c in data["configs"] if abs(c["sensor_correlation"] - corr) < 0.01]

    fn_rates = sorted({c["fn_rate"] for c in configs})
    fp_rates = sorted({c["fp_rate"] for c in configs})

    for metric_key, metric_label in METRICS:
        fig, axes = plt.subplots(
            1, len(fn_rates), figsize=(5 * len(fn_rates), 4.5),
            sharey=True, squeeze=False,
        )
        axes = axes[0]

        for ax_idx, fn in enumerate(fn_rates):
            ax = axes[ax_idx]
            subset = [c for c in configs if abs(c["fn_rate"] - fn) < 0.001]
            subset.sort(key=lambda c: c["fp_rate"])

            for sname in strategy_names:
                xs, ys = [], []
                for c in subset:
                    val = c["results"].get(sname, {}).get(metric_key)
                    if val is not None:
                        xs.append(c["fp_rate"])
                        ys.append(val)
                if xs:
                    ax.plot(xs, ys, marker="o", markersize=4, label=sname)

            ax.set_title(f"FN = {fn:.0%}", fontsize=10)
            ax.set_xlabel("False Positive Rate")
            ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
            if ax_idx == 0:
                ax.set_ylabel(metric_label)
            ax.grid(True, alpha=0.3)

        axes[-1].legend(fontsize=7, loc="best")
        fig.suptitle(f"{metric_label} vs Noise (corr={corr})", fontsize=12, y=1.02)
        fig.tight_layout()
        fname = f"robustness_{metric_key.replace('_mean', '')}_corr{corr}.png"
        fig.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches="tight")
        plt.close(fig)


# ===================================================================
# 2. Heatmaps
# ===================================================================

def plot_heatmaps(data: Dict, output_dir: str, corr: float = 0.0):
    """
    For each strategy × metric, draw a heatmap of FP (x) × FN (y).
    """
    strategy_names = data["strategy_names"]
    configs = [c for c in data["configs"] if abs(c["sensor_correlation"] - corr) < 0.01]

    fp_rates = sorted({c["fp_rate"] for c in configs})
    fn_rates = sorted({c["fn_rate"] for c in configs})

    for metric_key, metric_label in METRICS:
        fig, axes = plt.subplots(
            1, len(strategy_names),
            figsize=(3.5 * len(strategy_names), 3.5),
            squeeze=False,
        )
        axes = axes[0]

        for s_idx, sname in enumerate(strategy_names):
            grid = np.full((len(fn_rates), len(fp_rates)), np.nan)
            for c in configs:
                fi = fn_rates.index(c["fn_rate"])
                fj = fp_rates.index(c["fp_rate"])
                val = c["results"].get(sname, {}).get(metric_key)
                if val is not None:
                    grid[fi, fj] = val

            ax = axes[s_idx]
            im = ax.imshow(grid, aspect="auto", origin="lower", cmap="RdYlGn")
            ax.set_xticks(range(len(fp_rates)))
            ax.set_xticklabels([f"{r:.0%}" for r in fp_rates], fontsize=7, rotation=45)
            ax.set_yticks(range(len(fn_rates)))
            ax.set_yticklabels([f"{r:.0%}" for r in fn_rates], fontsize=7)
            ax.set_xlabel("FP Rate", fontsize=8)
            if s_idx == 0:
                ax.set_ylabel("FN Rate", fontsize=8)
            ax.set_title(sname, fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.suptitle(f"{metric_label} Heatmap (corr={corr})", fontsize=11, y=1.02)
        fig.tight_layout()
        fname = f"heatmap_{metric_key.replace('_mean', '')}_corr{corr}.png"
        fig.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches="tight")
        plt.close(fig)


# ===================================================================
# 3. Correlation impact plot
# ===================================================================

def plot_correlation_impact(data: Dict, output_dir: str):
    """
    Show how sensor correlation affects a key metric (value_preserved)
    at a fixed (FP=0.6, FN=0.2) noise level.
    """
    strategy_names = data["strategy_names"]
    target_fp, target_fn = 0.6, 0.2
    metric_key = "value_preserved_mean"

    corrs = sorted({c["sensor_correlation"] for c in data["configs"]})
    configs = [
        c for c in data["configs"]
        if abs(c["fp_rate"] - target_fp) < 0.01 and abs(c["fn_rate"] - target_fn) < 0.01
    ]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for sname in strategy_names:
        xs, ys = [], []
        for c in sorted(configs, key=lambda c: c["sensor_correlation"]):
            val = c["results"].get(sname, {}).get(metric_key)
            if val is not None:
                xs.append(c["sensor_correlation"])
                ys.append(val)
        if xs:
            ax.plot(xs, ys, marker="o", label=sname)

    ax.set_xlabel("Sensor Correlation")
    ax.set_ylabel("Value Preserved ($)")
    ax.set_title(f"Impact of Sensor Correlation (FP={target_fp:.0%}, FN={target_fn:.0%})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "noise_correlation_impact.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ===================================================================
# 4. Comparison table
# ===================================================================

def write_comparison_table(data: Dict, output_dir: str, corr: float = 0.0):
    """Write CSV + LaTeX table at representative noise levels."""
    strategy_names = data["strategy_names"]
    configs = [c for c in data["configs"] if abs(c["sensor_correlation"] - corr) < 0.01]

    rows = []
    for level in TABLE_LEVELS:
        fp, fn = level["fp"], level["fn"]
        matching = [
            c for c in configs
            if abs(c["fp_rate"] - fp) < 0.01 and abs(c["fn_rate"] - fn) < 0.01
        ]
        if not matching:
            continue
        cfg = matching[0]
        for sname in strategy_names:
            sres = cfg["results"].get(sname, {})
            rows.append({
                "Noise Level": level["label"],
                "FP Rate": f"{fp:.0%}",
                "FN Rate": f"{fn:.0%}",
                "Strategy": sname,
                "Protected Value ($)": f"{sres.get('protected_value_mean', 0):,.0f}",
                "Protected Value Std": f"{sres.get('protected_value_std', 0):,.0f}",
                "Value Preserved ($)": f"{sres.get('value_preserved_mean', 0):,.0f}",
                "Lost Value ($)": f"{sres.get('lost_value_mean', 0):,.0f}",
                "Compromised Assets": f"{sres.get('compromised_assets_count_mean', 0):.1f}",
                "Attack Success (%)": f"{sres.get('attack_success_rate_mean', 0):.1f}",
                "ROI (%)": f"{sres.get('roi_mean', 0):.1f}",
            })

    # CSV
    csv_path = os.path.join(output_dir, "noise_comparison_table.csv")
    if rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        print(f"Comparison CSV saved to {csv_path}")

    # LaTeX
    latex_path = os.path.join(output_dir, "noise_comparison_table.tex")
    with open(latex_path, "w") as f:
        cols = "l l l l r r r r r r r"
        f.write("\\begin{table}[htbp]\n\\centering\n\\caption{CyGATE performance under observation noise}\n")
        f.write(f"\\begin{{tabular}}{{{cols}}}\n\\hline\n")
        headers = ["Level", "FP", "FN", "Strategy", "Protected\\$", "Prot.Std",
                    "Preserved\\$", "Lost\\$", "Comp.", "Att.Succ.", "ROI"]
        f.write(" & ".join(headers) + " \\\\\n\\hline\n")
        for r in rows:
            vals = list(r.values())
            f.write(" & ".join(str(v) for v in vals) + " \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")
    print(f"LaTeX table saved to {latex_path}")


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="Plot noise experiment results")
    parser.add_argument(
        "--input-file",
        default=os.path.join(SCRIPT_DIR, "noise_results", "noise_experiment_results.json"),
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(SCRIPT_DIR, "noise_results"),
    )
    args = parser.parse_args()

    data = load_results(args.input_file)
    os.makedirs(args.output_dir, exist_ok=True)

    correlations = sorted({c["sensor_correlation"] for c in data["configs"]})

    for corr in correlations:
        print(f"Plotting for sensor_correlation = {corr} …")
        plot_robustness_curves(data, args.output_dir, corr=corr)
        plot_heatmaps(data, args.output_dir, corr=corr)

    if len(correlations) > 1:
        plot_correlation_impact(data, args.output_dir)

    write_comparison_table(data, args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
