#!/usr/bin/env python3
"""
Noise robustness experiment for CyGATE.

Sweeps false-positive / false-negative rates and sensor correlation levels,
running the full APT3 simulation across all seven defender strategies and
collecting key metrics under each noise configuration.

Usage:
    python experiments/run_noise_experiment.py                # full grid (slow)
    python experiments/run_noise_experiment.py --quick-test   # 1 strategy, 2 trials
"""

import sys
import os
import copy
import json
import time
import argparse
import logging
import itertools
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, PROJECT_ROOT)

from classes.noise_model import NoiseConfig, NoisyObservationModel
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


# ===================================================================
# NoisySimulationRunner – injects noise without touching core classes
# ===================================================================

class NoisySimulationRunner(APT3SimulationRunner):
    """
    Extends the simulation runner with configurable observation noise.

    Overrides the observation creation path so that:
    * Real observations are dropped with probability ``fn_rate`` (false negatives).
    * Phantom observations are injected with probability ``fp_rate`` per asset
      per step (false positives).
    * Detection-confidence values are perturbed by correlated noise.
    * Defender belief-update accuracy is degraded proportionally.
    """

    def __init__(self, noise_config: NoiseConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_config = noise_config
        self.noise_model = NoisyObservationModel(noise_config, self.system)
        # Degrade defender detection confidence
        self.defender_policy.belief_state.detection_confidence = (
            self.noise_model.degrade_defender_detection_confidence(
                self.defender_policy.belief_state.detection_confidence
            )
        )

    # Override the core observation creator
    def _create_attack_observation(self, action, action_result, step):
        obs = super()._create_attack_observation(action, action_result, step)
        obs = self.noise_model.apply_fn(obs)
        if obs is not None:
            obs = self.noise_model.corrupt_detection_confidence(obs)
        return obs

    def run_step(self, strategy, step, verbose=False):
        """
        Run a single step, injecting false-positive observations into the
        defender's observation stream.
        """
        self.current_step = step

        self.threat_processor.update_threat_levels(self.state)

        # --- Attacker phase (unchanged) ---
        attacker_action = self.get_next_attack_action(self.state)
        attacker_actions = (
            [attacker_action]
            if attacker_action and attacker_action.get("action_type") != "pause"
            else []
        )

        total_exploit_cost = 0.0
        exploit_events = []
        exploit_attempts = []
        action_results = []

        if not hasattr(self.state.system, "action_history"):
            self.state.system.action_history = []

        recon_count = sum(
            1
            for a in self.state.system.action_history
            if a.get("action_type") == "reconnaissance" and a.get("action_result", False)
        )
        if recon_count >= 5 and attacker_action.get("action_type") == "reconnaissance":
            self.state.suggest_attacker_stage(KillChainStage.DELIVERY.value)
            attacker_action = self.get_next_attack_action(self.state)
            attacker_actions = (
                [attacker_action]
                if attacker_action and attacker_action.get("action_type") != "pause"
                else []
            )

        attack_observations: List[Dict] = []

        if not attacker_actions:
            exploit_attempts.append({
                "step": step, "tactic": "None", "action_type": "none",
                "success": False, "asset_id": None, "target_asset": "None",
                "vulnerability": None, "cvss": "N/A", "epss": "N/A",
                "impact": 0, "details": "No attack action taken",
            })

        from apt3_simulation_core import ExploitEvent

        for action in attacker_actions:
            if action.get("action_type") in [
                "initial_access", "exploitation", "lateral_movement",
                "privilege_escalation", "persistence", "command_and_control",
                "exfiltration",
            ]:
                vuln_id = action.get("target_vuln")
                asset_id = action.get("target_asset")
                comp_id = action.get("target_component", "0")
                if all([vuln_id, asset_id]):
                    if self._should_skip_due_to_failures(vuln_id, asset_id, comp_id):
                        continue

            action_result = self.execute_attack_action(self.state, action)
            if action_result is None:
                action_result = {
                    "action_type": action.get("action_type", "unknown"),
                    "action_result": False, "reason": "null_result",
                    "tactic": action.get("tactic", "Unknown"),
                    "is_recon": action.get("action_type") == "reconnaissance",
                }

            action_results.append((action, action_result))
            self.state.system.action_history.append(action_result)
            self.attacker.observe_result(action_result)

            attack_observation = self._create_attack_observation(action, action_result, step)
            if attack_observation:
                attack_observations.append(attack_observation)

            is_recon = action_result.get("is_recon", action.get("action_type") == "reconnaissance")
            exploit_attempt = self._build_exploit_attempt(action, action_result, step, is_recon)
            exploit_attempts.append(exploit_attempt)

            if is_recon:
                total_exploit_cost += action_result.get("cost", 0)
            else:
                total_exploit_cost += (
                    action_result.get("cost", 0)
                    if action_result.get("action_result", False)
                    else action_result.get("attempt_cost", 0)
                )
                if action_result.get("action_result", False) and action.get("action_type") in [
                    "initial_access", "exploitation", "lateral_movement", "privilege_escalation",
                ]:
                    vuln_key = action_result.get("vuln_key")
                    if vuln_key:
                        exploit_events.append(ExploitEvent(
                            vuln_id=action_result.get("target_vuln"),
                            step=step,
                            asset_id=action_result.get("target_asset"),
                            success=True,
                            technique=exploit_attempt.get("technique") or "Unknown",
                        ).to_dict())
                    if action.get("tactic"):
                        self.mitre_techniques_used.add(action["tactic"])

        # ---- Inject false-positive observations ----
        fp_observations = self.noise_model.generate_fp_observations(step, self.state.system)
        attack_observations.extend(fp_observations)

        # --- Defender phase ---
        for obs in attack_observations:
            if hasattr(strategy, "observe_attack_behavior"):
                strategy.observe_attack_behavior(obs)

        defender_actions = strategy.select_patches(
            self.state, self._remaining_defender_budget, step, self.num_steps,
        )
        step_cost = 0.0
        applied_patches = []

        for vuln, cost in defender_actions:
            vuln_asset = vuln_component = None
            for asset in self.state.system.assets:
                for comp in asset.components:
                    for v in comp.vulnerabilities:
                        if v.cve_id == vuln.cve_id:
                            vuln_asset, vuln_component, vuln = asset, comp, v
                            break
                    if vuln_asset:
                        break
                if vuln_asset:
                    break
            if vuln_asset and vuln_component and not vuln.is_patched:
                if cost <= self._remaining_defender_budget:
                    vuln.apply_patch()
                    applied_patches.append(vuln)
                    step_cost += cost
                    self._patched_cves.add(vuln.cve_id)
                    self._remaining_defender_budget -= cost

        # --- Transitions ---
        attacker_vuln_objects = []
        for action, result in action_results:
            if result.get("action_result", False) and action.get("action_type") in [
                "initial_access", "exploitation", "lateral_movement", "privilege_escalation",
            ]:
                tid = action.get("target_vuln")
                taid = str(action.get("target_asset", ""))
                tcid = str(action.get("target_component", ""))
                for asset in self.state.system.assets:
                    if str(asset.asset_id) == taid:
                        for comp in asset.components:
                            if str(comp.id) == tcid:
                                for v in comp.vulnerabilities:
                                    if v.cve_id == tid:
                                        attacker_vuln_objects.append(v)
                                        break

        self.transition.apply_actions(self.state, applied_patches, attacker_vuln_objects)
        if any(
            r.get("action_result", False)
            for _, r in action_results
            if r.get("action_type") in [
                "initial_access", "exploitation", "lateral_movement", "privilege_escalation",
            ]
        ):
            self.state.update_kill_chain_stage()
        self.state.process_attacker_stage_suggestion()

        # --- Record results ---
        self.results["patched_vulns"].append(applied_patches)
        self.results["dollar_costs"]["patch_costs"].append(step_cost)
        self.results["dollar_costs"]["exploit_costs"].append(total_exploit_cost)
        self.results["exploit_events"].extend(exploit_events)
        self.results["exploit_attempts"].extend(exploit_attempts)
        self.results["kill_chain_stages"].append(self.state.k)
        self.results["compromised_assets"].append(
            sum(1 for a in self.state.system.assets if a.is_compromised)
        )
        self.results["mitre_techniques_used"].append(list(self.mitre_techniques_used))
        self.results["mitre_techniques_detected"].append(list(self.mitre_techniques_detected))
        self.results["time_to_detection"].append(self.time_to_detection.copy())
        self.results["attack_disruption_rate"].append(1.0 if self.attack_disrupted else 0.0)

        if hasattr(self.attacker, "get_decision_history"):
            dh = self.attacker.get_decision_history()
            if dh:
                self.results["strategic_decisions"].append(dh[-1])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _build_exploit_attempt(action, action_result, step, is_recon):
        if is_recon:
            return {
                "step": step, "tactic": action.get("tactic", "Reconnaissance"),
                "action_type": "reconnaissance",
                "success": action_result.get("action_result", False),
                "asset_id": action_result.get("target_asset"),
                "target_asset": "Unknown", "vulnerability": None,
                "cvss": "N/A", "epss": "N/A", "impact": 0,
                "details": f"{'Successful' if action_result.get('action_result') else 'Failed'} reconnaissance",
                "technique": None, "is_recon": True,
            }
        return {
            "step": step, "tactic": action.get("tactic", "Unknown"),
            "action_type": action_result.get("action_type", "none"),
            "success": action_result.get("action_result", False),
            "asset_id": action_result.get("target_asset"),
            "target_asset": "Unknown",
            "vulnerability": action_result.get("target_vuln"),
            "cvss": action.get("cvss", "Unknown"),
            "epss": action.get("epss", "Unknown"),
            "details": (
                f"{'Successfully' if action_result.get('action_result') else 'Failed'} "
                f"{action_result.get('action_type', 'action')}"
            ),
            "technique": None, "is_recon": False,
        }


# ===================================================================
# Experiment grid
# ===================================================================

DEFAULT_FP_RATES = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
DEFAULT_FN_RATES = [0.1, 0.15, 0.2, 0.25, 0.3]
DEFAULT_CORRELATIONS = [0.0, 0.3, 0.6]


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


def run_single_config(
    noise_config: NoiseConfig,
    data_file: str,
    strategies: Dict[str, Any],
    num_steps: int,
    num_trials: int,
    defender_budget: int,
    attacker_budget: int,
) -> Dict[str, Any]:
    """
    Run all strategies under one noise configuration for *num_trials* trials
    and return aggregated metrics.
    """
    config_results: Dict[str, Dict] = {}

    for strat_name, strat_obj in strategies.items():
        trial_metrics: List[Dict] = []

        for trial in range(num_trials):
            try:
                sim = NoisySimulationRunner(
                    noise_config=noise_config,
                    data_file=data_file,
                    num_steps=num_steps,
                    defender_budget=defender_budget,
                    attacker_budget=attacker_budget,
                )
                sim.noise_model.reset()

                original_system = copy.deepcopy(sim.system)
                sim.initialize_cost_cache()
                strat_instance = copy.deepcopy(strat_obj)
                strat_instance.initialize(sim.state, sim._cost_cache)
                strat_instance.state = sim.state
                sim.reset_exploit_status()

                total_patch_cost = 0.0
                total_patches = 0

                for step in range(num_steps):
                    sim.run_step(strat_instance, step)
                    step_cost = sum(sim.results["dollar_costs"]["patch_costs"][-1:])
                    total_patch_cost += step_cost
                    total_patches += len(sim.results["patched_vulns"][-1:])
                    should_stop, _ = sim.should_terminate_simulation(
                        step, num_steps, total_patch_cost,
                    )
                    if should_stop:
                        break

                metrics = sim.calculate_metrics(sim.state, total_patch_cost, total_patches)
                metrics["total_patch_cost"] = total_patch_cost
                metrics["total_patches"] = total_patches
                trial_metrics.append(metrics)

            except Exception as exc:
                logger.error(
                    "Trial %d failed for %s / %s: %s",
                    trial, noise_config.label(), strat_name, exc,
                )

        # Aggregate across trials
        agg: Dict[str, Any] = {}
        if trial_metrics:
            keys = trial_metrics[0].keys()
            for k in keys:
                vals = [m[k] for m in trial_metrics if isinstance(m.get(k), (int, float))]
                if vals:
                    agg[f"{k}_mean"] = float(np.mean(vals))
                    agg[f"{k}_std"] = float(np.std(vals))
            agg["num_trials_completed"] = len(trial_metrics)
        config_results[strat_name] = agg

    return config_results


def run_experiment(
    data_file: str,
    fp_rates: List[float],
    fn_rates: List[float],
    correlations: List[float],
    num_steps: int = 100,
    num_trials: int = 100,
    defender_budget: int = 7500,
    attacker_budget: int = 15000,
    output_dir: str = None,
    strategies: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:

    if output_dir is None:
        output_dir = os.path.join(SCRIPT_DIR, "noise_results")
    os.makedirs(output_dir, exist_ok=True)

    if strategies is None:
        strategies = _make_strategies()

    grid = list(itertools.product(fp_rates, fn_rates, correlations))
    total = len(grid)
    all_results: Dict[str, Any] = {"configs": [], "strategy_names": list(strategies.keys())}
    experiment_start = time.time()

    print(f"\n{'='*70}")
    print(f"NOISE ROBUSTNESS EXPERIMENT")
    print(f"  Grid size     : {total} configurations")
    print(f"  Strategies    : {len(strategies)}")
    print(f"  Trials/config : {num_trials}")
    print(f"  Total runs    : {total * len(strategies) * num_trials}")
    print(f"{'='*70}\n")

    for idx, (fp, fn, corr) in enumerate(grid, 1):
        nc = NoiseConfig(
            fp_rate=fp, fn_rate=fn,
            sensor_correlation=corr,
            temporal_correlation=0.3,
            noise_seed=42,
        )
        print(f"[{idx}/{total}] {nc.label()} …", end=" ", flush=True)
        t0 = time.time()

        config_res = run_single_config(
            noise_config=nc,
            data_file=data_file,
            strategies=strategies,
            num_steps=num_steps,
            num_trials=num_trials,
            defender_budget=defender_budget,
            attacker_budget=attacker_budget,
        )

        entry = {
            "fp_rate": fp, "fn_rate": fn, "sensor_correlation": corr,
            "results": config_res,
        }
        all_results["configs"].append(entry)
        elapsed = time.time() - t0
        print(f"done ({elapsed:.1f}s)")

    total_elapsed = time.time() - experiment_start

    # Persist
    results_path = os.path.join(output_dir, "noise_experiment_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")
    print(f"Total elapsed: {total_elapsed:.1f}s")
    return all_results


# ===================================================================
# CLI
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="CyGATE noise robustness experiment")
    parser.add_argument(
        "--data-file", default=os.path.join(PROJECT_ROOT, "data", "systemData", "apt3_scenario_enriched.json"),
    )
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--num-trials", type=int, default=100)
    parser.add_argument("--defender-budget", type=int, default=7500)
    parser.add_argument("--attacker-budget", type=int, default=15000)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--quick-test", action="store_true",
        help="Run a minimal config for smoke-testing (1 strategy, 2 trials)",
    )
    args = parser.parse_args()

    if args.quick_test:
        strategies = {"CVSS-Only": CVSSOnlyStrategy()}
        run_experiment(
            data_file=args.data_file,
            fp_rates=[0.5],
            fn_rates=[0.2],
            correlations=[0.0],
            num_steps=20,
            num_trials=2,
            defender_budget=args.defender_budget,
            attacker_budget=args.attacker_budget,
            output_dir=args.output_dir,
            strategies=strategies,
        )
    else:
        run_experiment(
            data_file=args.data_file,
            fp_rates=DEFAULT_FP_RATES,
            fn_rates=DEFAULT_FN_RATES,
            correlations=DEFAULT_CORRELATIONS,
            num_steps=args.num_steps,
            num_trials=args.num_trials,
            defender_budget=args.defender_budget,
            attacker_budget=args.attacker_budget,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
