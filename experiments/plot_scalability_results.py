#!/usr/bin/env python3
"""
Visualization for the scalability benchmark.

Reads ``scalability_results.json`` and produces:
  * Runtime vs assets (log-scale y)
  * Peak memory vs assets
  * Planner calls vs assets
  * Capacity table CSV (also written by the benchmark runner itself)

Usage:
    python experiments/plot_scalability_results.py [--input-file path] [--output-dir path]
"""

import os
import sys
import json
import csv
import argparse
from typing import Dict, List, Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

STRATEGY_STYLES = {
    "CVSS-Only":            {"marker": "o", "linestyle": "-",  "color": "#1f77b4"},
    "CVSS+Exploit":         {"marker": "s", "linestyle": "--", "color": "#ff7f0e"},
    "Business Value":       {"marker": "D", "linestyle": "-.", "color": "#2ca02c"},
    "Cost-Benefit":         {"marker": "^", "linestyle": ":",  "color": "#d62728"},
    "Threat Intelligence":  {"marker": "v", "linestyle": "-",  "color": "#9467bd"},
    "RL Defender":          {"marker": "P", "linestyle": "--", "color": "#8c564b"},
    "Hybrid Defender":      {"marker": "X", "linestyle": "-.", "color": "#e377c2"},
}

def _style(sname: str) -> dict:
    return STRATEGY_STYLES.get(sname, {"marker": "o", "linestyle": "-"})


def load_results(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


# ===================================================================
# 1. Runtime vs assets
# ===================================================================

def plot_runtime(data: Dict, output_dir: str):
    strategy_names = data["strategy_names"]
    sizes = [s["num_assets"] for s in data["sizes"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    for sname in strategy_names:
        st = _style(sname)
        ys, errs = [], []
        for sd in data["sizes"]:
            sdata = sd["strategies"].get(sname, {})
            val = sdata.get("avg_step_latency_s_mean", None)
            std = sdata.get("avg_step_latency_s_std", 0)
            ys.append(val * 1000 if val is not None else np.nan)
            errs.append(std * 1000)
        ax.errorbar(sizes, ys, yerr=errs, marker=st["marker"], linestyle=st["linestyle"],
                     color=st.get("color"), markersize=5, capsize=3, label=sname, linewidth=1.5)

    ax.set_xlabel("Number of Assets")
    ax.set_ylabel("Avg Step Latency (ms)")
    ax.set_title("Step Latency vs Network Size")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "runtime_vs_assets.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ===================================================================
# 2. Memory vs assets
# ===================================================================

def plot_memory(data: Dict, output_dir: str):
    strategy_names = data["strategy_names"]
    sizes = [s["num_assets"] for s in data["sizes"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    for sname in strategy_names:
        st = _style(sname)
        ys, errs = [], []
        for sd in data["sizes"]:
            sdata = sd["strategies"].get(sname, {})
            val = sdata.get("peak_memory_bytes_mean", None)
            std = sdata.get("peak_memory_bytes_std", 0)
            ys.append(val / 1e6 if val is not None else np.nan)
            errs.append(std / 1e6)
        ax.errorbar(sizes, ys, yerr=errs, marker=st["marker"], linestyle=st["linestyle"],
                     color=st.get("color"), markersize=5, capsize=3, label=sname, linewidth=1.5)

    ax.set_xlabel("Number of Assets")
    ax.set_ylabel("Peak Memory (MB)")
    ax.set_title("Peak Memory vs Network Size")
    ax.set_xscale("log")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "memory_vs_assets.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ===================================================================
# 3. Planner calls vs assets
# ===================================================================

def plot_planner_calls(data: Dict, output_dir: str):
    strategy_names = data["strategy_names"]
    sizes = [s["num_assets"] for s in data["sizes"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    for sname in strategy_names:
        st = _style(sname)
        ys, errs = [], []
        for sd in data["sizes"]:
            sdata = sd["strategies"].get(sname, {})
            val = sdata.get("planner_calls_mean", None)
            std = sdata.get("planner_calls_std", 0)
            ys.append(val if val is not None else np.nan)
            errs.append(std)
        ax.errorbar(sizes, ys, yerr=errs, marker=st["marker"], linestyle=st["linestyle"],
                     color=st.get("color"), markersize=5, capsize=3, label=sname, linewidth=1.5)

    ax.set_xlabel("Number of Assets")
    ax.set_ylabel("Planner Calls / Trial")
    ax.set_title("Attack-Graph Rebuilds vs Network Size")
    ax.set_xscale("log")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "planner_calls_vs_assets.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ===================================================================
# 4. Combined summary figure
# ===================================================================

def plot_combined(data: Dict, output_dir: str):
    """Three subplots side-by-side for a compact paper figure."""
    strategy_names = data["strategy_names"]
    sizes = [s["num_assets"] for s in data["sizes"]]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.5))

    for sname in strategy_names:
        st = _style(sname)
        lats, mems, pcs = [], [], []
        for sd in data["sizes"]:
            sdata = sd["strategies"].get(sname, {})
            lats.append((sdata.get("avg_step_latency_s_mean", 0) or 0) * 1000)
            mems.append((sdata.get("peak_memory_bytes_mean", 0) or 0) / 1e6)
            pcs.append(sdata.get("planner_calls_mean", 0) or 0)
        ax1.plot(sizes, lats, marker=st["marker"], linestyle=st["linestyle"],
                 color=st.get("color"), markersize=4, label=sname, linewidth=1.3)
        ax2.plot(sizes, mems, marker=st["marker"], linestyle=st["linestyle"],
                 color=st.get("color"), markersize=4, label=sname, linewidth=1.3)
        ax3.plot(sizes, pcs, marker=st["marker"], linestyle=st["linestyle"],
                 color=st.get("color"), markersize=4, label=sname, linewidth=1.3)

    ax1.set_xlabel("Assets"); ax1.set_ylabel("Latency (ms/step)")
    ax1.set_title("(a) Step Latency"); ax1.set_yscale("log"); ax1.set_xscale("log")
    ax1.grid(True, alpha=0.3, which="both")

    ax2.set_xlabel("Assets"); ax2.set_ylabel("Memory (MB)")
    ax2.set_title("(b) Peak Memory"); ax2.set_xscale("log")
    ax2.grid(True, alpha=0.3)

    ax3.set_xlabel("Assets"); ax3.set_ylabel("Planner Calls")
    ax3.set_title("(c) Attack-Graph Rebuilds"); ax3.set_xscale("log")
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=6, loc="upper left")

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "scalability_combined.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ===================================================================
# 5. Capacity table (re-generate from results JSON)
# ===================================================================

def _feasibility_label(latency_ms: float) -> str:
    if latency_ms < 100:
        return "Real-time"
    if latency_ms < 500:
        return "Near-RT"
    if latency_ms < 5000:
        return "Batch"
    return "Slow"


def write_capacity_table(data: Dict, output_dir: str):
    size_labels = {20: "S", 50: "S", 100: "M", 200: "M", 500: "L", 1000: "XL"}
    rows = []

    for sd in data["sizes"]:
        n = sd["num_assets"]
        label = size_labels.get(n, "?")
        lats, mems, pcs = [], [], []
        for sdata in sd["strategies"].values():
            if sdata.get("avg_step_latency_s_mean") is not None:
                lats.append(sdata["avg_step_latency_s_mean"] * 1000)
            if sdata.get("peak_memory_bytes_mean") is not None:
                mems.append(sdata["peak_memory_bytes_mean"] / 1e6)
            if sdata.get("planner_calls_mean") is not None:
                pcs.append(sdata["planner_calls_mean"])

        rows.append({
            "Size": label,
            "Num Assets": n,
            "Num Vulns": sd.get("num_vulnerabilities", "?"),
            "Num Connections": sd.get("num_connections", "?"),
            "Avg Step Latency (ms)": f"{np.mean(lats):.1f}" if lats else "N/A",
            "Peak RAM (MB)": f"{np.mean(mems):.1f}" if mems else "N/A",
            "Planner Calls/Trial": f"{np.mean(pcs):.1f}" if pcs else "N/A",
            "Feasibility": _feasibility_label(np.mean(lats)) if lats else "N/A",
        })

    csv_path = os.path.join(output_dir, "capacity_table.csv")
    if rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        print(f"Capacity table saved to {csv_path}")

    # LaTeX version
    tex_path = os.path.join(output_dir, "capacity_table.tex")
    with open(tex_path, "w") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n")
        f.write("\\caption{CyGATE Scalability -- Deployment Capacity}\n")
        f.write("\\begin{tabular}{l r r r r r r l}\n\\hline\n")
        f.write("Size & Assets & Vulns & Conns & Latency(ms) & RAM(MB) & Planner & Feasibility \\\\\n\\hline\n")
        for r in rows:
            vals = list(r.values())
            f.write(" & ".join(str(v) for v in vals) + " \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")
    print(f"LaTeX capacity table saved to {tex_path}")


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="Plot scalability benchmark results")
    parser.add_argument(
        "--input-file",
        default=os.path.join(SCRIPT_DIR, "scalability_results", "scalability_results.json"),
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(SCRIPT_DIR, "scalability_results"),
    )
    args = parser.parse_args()

    data = load_results(args.input_file)
    os.makedirs(args.output_dir, exist_ok=True)

    plot_runtime(data, args.output_dir)
    plot_memory(data, args.output_dir)
    plot_planner_calls(data, args.output_dir)
    plot_combined(data, args.output_dir)
    write_capacity_table(data, args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
