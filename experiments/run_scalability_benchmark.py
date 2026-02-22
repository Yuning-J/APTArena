#!/usr/bin/env python3
"""
Scalability benchmark for CyGATE.

Generates synthetic enterprise / ICS topologies at increasing sizes
(20 → 1000 assets), runs the full simulation for all seven strategies,
and profiles wall-clock latency, peak memory, and planner (attack-graph
rebuild) calls.

Usage:
    python experiments/run_scalability_benchmark.py                 # full sweep
    python experiments/run_scalability_benchmark.py --quick-test    # smallest size, 1 strategy, 2 trials
"""

import sys
import os
import copy
import json
import time
import tracemalloc
import argparse
import logging
import tempfile
from datetime import datetime
from typing import Dict, List, Any, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, PROJECT_ROOT)

from classes.topology_generator import EnterpriseTopologyGenerator
from apt3_simulation_run import APT3SimulationRunner
from classes.state import KillChainStage, State, create_vuln_key
from classes.patching_strategies import (
    CVSSOnlyStrategy,
    CVSSExploitAwareStrategy,
    BusinessValueStrategy,
    CostBenefitStrategy,
    ThreatIntelligenceStrategy,
)
from classes.hybrid_strategy import HybridStrategy
from src.RL_defender_strategy import RLAdaptiveThreatIntelligenceStrategy

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_TOPOLOGY_SIZES = [20, 50, 100, 200, 500, 1000]


# ===================================================================
# Profiled simulation runner
# ===================================================================

class ProfiledSimulationRunner(APT3SimulationRunner):
    """Thin wrapper that counts attack-graph rebuilds (planner calls)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.planner_calls = 0
        self._orig_build = self.attacker.attack_graph.build_attack_graph

        def _counted_build(*a, **kw):
            self.planner_calls += 1
            return self._orig_build(*a, **kw)

        self.attacker.attack_graph.build_attack_graph = _counted_build


def _make_strategies() -> Dict[str, Any]:
    return {
        "CVSS-Only": CVSSOnlyStrategy(),
        "CVSS+Exploit": CVSSExploitAwareStrategy(),
        "Business Value": BusinessValueStrategy(),
        "Cost-Benefit": CostBenefitStrategy(),
        "Threat Intelligence": ThreatIntelligenceStrategy(),
        "RL Defender": RLAdaptiveThreatIntelligenceStrategy(),
        "Hybrid Defender": HybridStrategy(),
    }


# ===================================================================
# Run one topology size
# ===================================================================

def run_single_size(
    num_assets: int,
    strategies: Dict[str, Any],
    num_steps: int,
    num_trials: int,
    defender_budget: int,
    attacker_budget: int,
    topo_seed: int = 42,
) -> Dict[str, Any]:
    """
    Generate a topology of *num_assets*, run all strategies, and return
    profiling data.
    """
    gen = EnterpriseTopologyGenerator(seed=topo_seed)
    topo_json = gen.generate(num_assets)

    # Write to temp file for data_loader
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, prefix=f"topo_{num_assets}_",
    )
    json.dump(topo_json, tmp, indent=2)
    tmp.close()
    topo_path = tmp.name

    total_bv = sum(a.get("business_value", 0) for a in topo_json["Assets"])
    # Scale budgets proportionally to business value
    base_bv = 194000  # approximate total BV of original 9-asset scenario
    scale = max(1.0, total_bv / base_bv)
    scaled_def_budget = int(defender_budget * scale)
    scaled_att_budget = int(attacker_budget * scale)

    size_results: Dict[str, Any] = {
        "num_assets": num_assets,
        "total_business_value": total_bv,
        "num_connections": len(topo_json["Connections"]),
        "num_vulnerabilities": sum(
            len(v)
            for a in topo_json["Assets"]
            for c in a.get("components", [])
            for v in [c.get("vulnerabilities", [])]
        ),
        "scaled_defender_budget": scaled_def_budget,
        "scaled_attacker_budget": scaled_att_budget,
        "strategies": {},
    }

    for strat_name, strat_obj in strategies.items():
        trial_profiles: List[Dict] = []

        for trial in range(num_trials):
            try:
                tracemalloc.start()
                trial_start = time.perf_counter()

                sim = ProfiledSimulationRunner(
                    data_file=topo_path,
                    num_steps=num_steps,
                    defender_budget=scaled_def_budget,
                    attacker_budget=scaled_att_budget,
                )
                sim.initialize_cost_cache()

                strat_instance = copy.deepcopy(strat_obj)
                strat_instance.initialize(sim.state, sim._cost_cache)
                strat_instance.state = sim.state
                sim.reset_exploit_status()

                step_latencies: List[float] = []
                total_patch_cost = 0.0
                total_patches = 0

                for step in range(num_steps):
                    t0 = time.perf_counter()
                    sim.run_step(strat_instance, step)
                    sim.update_attack_graph(sim.state)
                    step_latencies.append(time.perf_counter() - t0)

                    step_cost = sum(sim.results["dollar_costs"]["patch_costs"][-1:])
                    total_patch_cost += step_cost
                    total_patches += len(sim.results["patched_vulns"][-1:])

                    should_stop, _ = sim.should_terminate_simulation(
                        step, num_steps, total_patch_cost,
                    )
                    if should_stop:
                        break

                trial_wall = time.perf_counter() - trial_start
                _, peak_mem = tracemalloc.get_traced_memory()
                tracemalloc.stop()

                metrics = sim.calculate_metrics(sim.state, total_patch_cost, total_patches)

                trial_profiles.append({
                    "wall_clock_s": trial_wall,
                    "peak_memory_bytes": peak_mem,
                    "planner_calls": sim.planner_calls,
                    "steps_completed": len(step_latencies),
                    "avg_step_latency_s": float(np.mean(step_latencies)) if step_latencies else 0,
                    "max_step_latency_s": float(np.max(step_latencies)) if step_latencies else 0,
                    "protected_value": metrics["protected_value"],
                    "compromised_assets": metrics["compromised_assets_count"],
                    "value_preserved": metrics["value_preserved"],
                })

            except Exception as exc:
                if tracemalloc.is_tracing():
                    tracemalloc.stop()
                logger.error(
                    "Trial %d failed for size=%d / %s: %s",
                    trial, num_assets, strat_name, exc,
                )

        # Aggregate
        agg: Dict[str, Any] = {"num_trials_completed": len(trial_profiles)}
        if trial_profiles:
            for k in trial_profiles[0]:
                vals = [tp[k] for tp in trial_profiles if isinstance(tp.get(k), (int, float))]
                if vals:
                    agg[f"{k}_mean"] = float(np.mean(vals))
                    agg[f"{k}_std"] = float(np.std(vals))
        size_results["strategies"][strat_name] = agg

    # Clean up temp file
    try:
        os.unlink(topo_path)
    except OSError:
        pass

    return size_results


# ===================================================================
# Full experiment
# ===================================================================

def run_experiment(
    topology_sizes: List[int],
    num_steps: int = 100,
    num_trials: int = 100,
    defender_budget: int = 7500,
    attacker_budget: int = 15000,
    output_dir: str = None,
    strategies: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:

    if output_dir is None:
        output_dir = os.path.join(SCRIPT_DIR, "scalability_results")
    os.makedirs(output_dir, exist_ok=True)

    if strategies is None:
        strategies = _make_strategies()

    all_results: Dict[str, Any] = {
        "topology_sizes": topology_sizes,
        "strategy_names": list(strategies.keys()),
        "num_steps": num_steps,
        "num_trials": num_trials,
        "sizes": [],
    }
    experiment_start = time.time()

    print(f"\n{'='*70}")
    print(f"SCALABILITY BENCHMARK")
    print(f"  Topology sizes : {topology_sizes}")
    print(f"  Strategies     : {len(strategies)}")
    print(f"  Trials/size    : {num_trials}")
    print(f"  Steps/trial    : {num_steps}")
    print(f"{'='*70}\n")

    for size in topology_sizes:
        print(f"--- Size = {size} assets ---", flush=True)
        t0 = time.time()

        size_res = run_single_size(
            num_assets=size,
            strategies=strategies,
            num_steps=num_steps,
            num_trials=num_trials,
            defender_budget=defender_budget,
            attacker_budget=attacker_budget,
        )
        all_results["sizes"].append(size_res)

        elapsed = time.time() - t0
        print(f"    Completed in {elapsed:.1f}s")

        for sname, sdata in size_res["strategies"].items():
            lat = sdata.get("avg_step_latency_s_mean", 0) * 1000
            mem = sdata.get("peak_memory_bytes_mean", 0) / 1e6
            pc = sdata.get("planner_calls_mean", 0)
            print(
                f"      {sname:20s}  latency={lat:8.1f}ms/step  "
                f"mem={mem:8.1f}MB  planner={pc:6.1f}"
            )

    total_elapsed = time.time() - experiment_start

    results_path = os.path.join(output_dir, "scalability_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")
    print(f"Total elapsed: {total_elapsed:.1f}s")

    # Generate capacity table
    _write_capacity_table(all_results, output_dir)

    return all_results


def _feasibility_label(latency_ms: float) -> str:
    if latency_ms < 100:
        return "Real-time"
    if latency_ms < 500:
        return "Near-RT"
    if latency_ms < 5000:
        return "Batch"
    return "Slow"


def _write_capacity_table(results: Dict, output_dir: str):
    """Write a CSV capacity table summarizing S/M/L/XL deployments."""
    rows = []
    size_labels = {20: "S", 50: "S", 100: "M", 200: "M", 500: "L", 1000: "XL"}

    for size_data in results["sizes"]:
        n = size_data["num_assets"]
        label = size_labels.get(n, "?")

        # Average across all strategies
        lats, mems, pcs = [], [], []
        for sdata in size_data["strategies"].values():
            if sdata.get("avg_step_latency_s_mean") is not None:
                lats.append(sdata["avg_step_latency_s_mean"] * 1000)
            if sdata.get("peak_memory_bytes_mean") is not None:
                mems.append(sdata["peak_memory_bytes_mean"] / 1e6)
            if sdata.get("planner_calls_mean") is not None:
                pcs.append(sdata["planner_calls_mean"])

        rows.append({
            "Size": label,
            "Num Assets": n,
            "Num Vulns": size_data["num_vulnerabilities"],
            "Num Connections": size_data["num_connections"],
            "Avg Step Latency (ms)": f"{np.mean(lats):.1f}" if lats else "N/A",
            "Peak RAM (MB)": f"{np.mean(mems):.1f}" if mems else "N/A",
            "Planner Calls/Trial": f"{np.mean(pcs):.1f}" if pcs else "N/A",
            "Feasibility": _feasibility_label(np.mean(lats) if lats else float("inf")),
        })

    path = os.path.join(output_dir, "capacity_table.csv")
    import csv
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Capacity table saved to {path}")


# ===================================================================
# CLI
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="CyGATE scalability benchmark")
    parser.add_argument("--num-steps", type=int, default=25)
    parser.add_argument("--num-trials", type=int, default=25)
    parser.add_argument("--defender-budget", type=int, default=7500)
    parser.add_argument("--attacker-budget", type=int, default=15000)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--quick-test", action="store_true",
        help="Minimal run (size=20, 1 strategy, 2 trials)",
    )
    args = parser.parse_args()

    if args.quick_test:
        strategies = {"CVSS-Only": CVSSOnlyStrategy()}
        run_experiment(
            topology_sizes=[20],
            num_steps=20,
            num_trials=2,
            defender_budget=args.defender_budget,
            attacker_budget=args.attacker_budget,
            output_dir=args.output_dir,
            strategies=strategies,
        )
    else:
        run_experiment(
            topology_sizes=DEFAULT_TOPOLOGY_SIZES,
            num_steps=args.num_steps,
            num_trials=args.num_trials,
            defender_budget=args.defender_budget,
            attacker_budget=args.attacker_budget,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
