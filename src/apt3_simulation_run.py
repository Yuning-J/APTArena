#!/usr/bin/env python3
"""
APT3 RTU Simulation Runner - Main execution logic
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import random
import copy
import json
import logging
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import pickle
from typing import Dict, List, Any, Optional

from apt3_simulation_core import APT3RTUSimulation, ExploitEvent
from classes.state import KillChainStage
from classes.patching_strategies import (
    CVSSOnlyStrategy,
    CVSSExploitAwareStrategy,
    BusinessValueStrategy,
    CostBenefitStrategy,
    ThreatIntelligenceStrategy
)
from classes.hybrid_strategy import HybridStrategy
from src.RL_defender_strategy import RLAdaptiveThreatIntelligenceStrategy

logger = logging.getLogger(__name__)

class APT3SimulationRunner(APT3RTUSimulation):
    """Extended simulation class with run_step and strategy comparison methods"""
    
    def run_step(self, strategy, step, verbose=False):
        """Run a single step of the simulation with POMDP-based attacker and threat intelligence integration."""
        self.current_step = step
        if verbose:
            print(f"\n==== Step {step + 1} ====")
            print(f"Current kill chain stage: {KillChainStage(self.state.k).name}")
            print(f"Remaining defender budget: ${self._remaining_defender_budget:.2f}")
            print(f"Remaining attacker budget: ${self._remaining_attacker_budget:.2f}")

        self.threat_processor.update_threat_levels(self.state)

        # Get attacker action based on belief state
        attacker_action = self.get_next_attack_action(self.state)
        attacker_actions = [attacker_action] if attacker_action and attacker_action.get(
            'action_type') != 'pause' else []

        total_exploit_cost = 0.0
        exploit_events = []
        exploit_attempts = []
        action_results = []

        if not hasattr(self.state.system, 'action_history'):
            self.state.system.action_history = []

        # Handle reconnaissance limit
        recon_count = sum(1 for a in self.state.system.action_history
                          if a.get('action_type') == 'reconnaissance' and a.get('action_result', False))
        if recon_count >= 5 and attacker_action.get('action_type') == 'reconnaissance':
            logger.info(f"Step {step + 1}: Excessive reconnaissance ({recon_count}), suggesting kill chain progression")
            self.state.suggest_attacker_stage(KillChainStage.DELIVERY.value)
            attacker_action = self.get_next_attack_action(self.state)
            attacker_actions = [attacker_action] if attacker_action and attacker_action.get(
                'action_type') != 'pause' else []

        # Execute attacker actions and collect observations
        attack_observations = []

        if not attacker_actions:
            logger.info(f"Step {step + 1}: No attack action taken")
            if verbose:
                print("Attacker: No action taken")
            exploit_attempts.append({
                'step': step, 'tactic': 'None', 'action_type': 'none', 'success': False,
                'asset_id': None, 'target_asset': 'None', 'vulnerability': None,
                'cvss': 'N/A', 'epss': 'N/A', 'impact': 0, 'details': 'No attack action taken'
            })

        for action in attacker_actions:
            # Skip actions based on excessive failures as per POMDP logic
            if action.get('action_type') in ['initial_access', 'exploitation', 'lateral_movement',
                                             'privilege_escalation', 'persistence', 'command_and_control',
                                             'exfiltration']:
                vuln_id = action.get('target_vuln')
                asset_id = action.get('target_asset')
                comp_id = action.get('target_component', '0')
                if all([vuln_id, asset_id]):
                    if self._should_skip_due_to_failures(vuln_id, asset_id, comp_id):
                        logger.info(
                            f"Skipping {self.create_vuln_key(vuln_id, str(asset_id), str(comp_id))} due to excessive failures")
                        continue

            # Execute action and determine outcome based on actual state
            action_result = self.execute_attack_action(self.state, action)
            if action_result is None:
                logger.error(f"execute_attack_action returned None for action: {action}")
                action_result = {
                    'action_type': action.get('action_type', 'unknown'),
                    'action_result': False,
                    'reason': 'null_result',
                    'tactic': action.get('tactic', 'Unknown'),
                    'is_recon': action.get('action_type') == 'reconnaissance'
                }

            action_results.append((action, action_result))
            self.state.system.action_history.append(action_result)

            # Let the attacker observe the result to update its belief state
            self.attacker.observe_result(action_result)

            # Create attack observation for Threat Intelligence defender
            attack_observation = self._create_attack_observation(action, action_result, step)
            if attack_observation:
                attack_observations.append(attack_observation)

            # Process exploit attempts for metrics
            is_recon = action_result.get('is_recon', action.get('action_type') == 'reconnaissance')
            if is_recon:
                exploit_attempt = {
                    'step': step,
                    'tactic': action.get('tactic', 'Reconnaissance'),
                    'action_type': 'reconnaissance',
                    'success': action_result.get('action_result', False),
                    'asset_id': action_result.get('target_asset', None),
                    'target_asset': 'Unknown',
                    'vulnerability': None,
                    'cvss': 'N/A',
                    'epss': 'N/A',
                    'impact': 0,
                    'details': f"{'Successful' if action_result.get('action_result') else 'Failed'} reconnaissance on {action_result.get('target_asset', 'unknown')} (Reason: {action_result.get('reason', 'none')})",
                    'technique': None,
                    'is_recon': True
                }
                exploit_attempts.append(exploit_attempt)
                total_exploit_cost += action_result.get('cost', 0)
                if verbose:
                    print(f"Attacker: {exploit_attempt['details']}")
            else:
                exploit_attempt = {
                    'step': step,
                    'tactic': action.get('tactic', 'Unknown'),
                    'action_type': action_result.get('action_type', 'none'),
                    'success': action_result.get('action_result', False),
                    'asset_id': action_result.get('target_asset'),
                    'target_asset': next((a.name for a in self.state.system.assets if
                                          str(a.asset_id) == str(action_result.get('target_asset'))), 'Unknown'),
                    'vulnerability': action_result.get('target_vuln'),
                    'cvss': action.get('cvss', 'Unknown'),
                    'epss': action.get('epss', 'Unknown'),
                    'details': f"{'Successfully' if action_result.get('action_result') else 'Failed'} {action_result.get('action_type', 'action')} on {action_result.get('target_asset', 'unknown')} (Reason: {action_result.get('reason', 'unknown')})",
                    'technique': next((t for t in
                                       getattr(self.vuln_lookup.get(action_result.get('vuln_key', ''), [None])[0],
                                               'mitre_techniques', [])), None),
                    'is_recon': False
                }
                exploit_attempts.append(exploit_attempt)
                total_exploit_cost += action_result.get('cost', 0) if action_result.get('action_result',
                                                                                        False) else action_result.get(
                    'attempt_cost', 0)
                if verbose:
                    print(f"Attacker: {exploit_attempt['details']}")

                # Track successful exploits
                if action_result.get('action_result', False) and action.get('action_type') in ['initial_access',
                                                                                               'exploitation',
                                                                                               'lateral_movement',
                                                                                               'privilege_escalation']:
                    vuln_key = action_result.get('vuln_key')
                    if vuln_key:
                        exploit_event = ExploitEvent(
                            vuln_id=action_result.get('target_vuln'),
                            step=step,
                            asset_id=action_result.get('target_asset'),
                            success=True,
                            technique=exploit_attempt['technique'] or 'Unknown'
                        )
                        exploit_events.append(exploit_event.to_dict())
                        exploit_attempt['impact'] = self._cost_cache['exploit_losses'].get(vuln_key, 0)
                    if action.get('tactic'):
                        self.mitre_techniques_used.add(action['tactic'])

        # DEFENDER PHASE: Apply strategy-specific learning and patching
        for observation in attack_observations:
            if hasattr(strategy, 'observe_attack_behavior'):
                strategy.observe_attack_behavior(observation)
            if verbose:
                print(
                    f"  Threat Intel: Observed attack - {observation.get('action_type')} on asset {observation.get('target_asset')} ({'successful' if observation.get('success') else 'failed'})")

        # Execute defender strategy
        defender_actions = strategy.select_patches(self.state, self._remaining_defender_budget, step, self.num_steps)
        step_cost = 0.0
        applied_patches = []

        for vuln, cost in defender_actions:
            vuln_asset = None
            vuln_component = None
            for asset in self.state.system.assets:
                for comp in asset.components:
                    for v in comp.vulnerabilities:
                        if v.cve_id == vuln.cve_id:
                            vuln_asset = asset
                            vuln_component = comp
                            vuln = v
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
                    if verbose:
                        print(f"Defender: Patched {vuln.cve_id} on asset {vuln_asset.asset_id} (Cost: ${cost:.2f})")
        if not defender_actions and verbose:
            print(f"Step {step}: No patches applied, Budget: ${self._remaining_defender_budget:.2f}")

        # Apply state transitions
        attacker_vuln_objects = []
        for action, result in action_results:
            if result.get('action_result', False) and action.get('action_type') in ['initial_access', 'exploitation',
                                                                                    'lateral_movement',
                                                                                    'privilege_escalation']:
                target_vuln_id = action.get('target_vuln')
                target_asset_id = str(action.get('target_asset', ''))
                target_comp_id = str(action.get('target_component', ''))
                for asset in self.state.system.assets:
                    if str(asset.asset_id) == target_asset_id:
                        for comp in asset.components:
                            if str(comp.id) == target_comp_id:
                                for vuln in comp.vulnerabilities:
                                    if vuln.cve_id == target_vuln_id:
                                        attacker_vuln_objects.append(vuln)
                                        logger.debug(
                                            f"Added {vuln.cve_id} on asset {asset.asset_id} to attacker_vuln_objects")
                                        break
                            if attacker_vuln_objects and attacker_vuln_objects[-1].cve_id == target_vuln_id:
                                break
                    if attacker_vuln_objects and attacker_vuln_objects[-1].cve_id == target_vuln_id:
                        break

        self.transition.apply_actions(self.state, applied_patches, attacker_vuln_objects)
        if any(result.get('action_result', False) for _, result in action_results
               if result.get('action_type') in ['initial_access', 'exploitation', 'lateral_movement',
                                                'privilege_escalation']):
            self.state.update_kill_chain_stage()
        self.state.process_attacker_stage_suggestion()

        # Update results tracking
        self.results["patched_vulns"].append(applied_patches)
        self.results["dollar_costs"]["patch_costs"].append(step_cost)
        self.results["dollar_costs"]["exploit_costs"].append(total_exploit_cost)
        self.results["exploit_events"].extend(exploit_events)
        self.results["exploit_attempts"].extend(exploit_attempts)
        self.results["kill_chain_stages"].append(self.state.k)
        self.results["compromised_assets"].append(sum(1 for asset in self.state.system.assets if asset.is_compromised))
        self.results["mitre_techniques_used"].append(list(self.mitre_techniques_used))
        self.results["mitre_techniques_detected"].append(list(self.mitre_techniques_detected))
        self.results["time_to_detection"].append(self.time_to_detection.copy())
        self.results["attack_disruption_rate"].append(1.0 if self.attack_disrupted else 0.0)

        if hasattr(self.attacker, 'get_decision_history'):
            decision_history = self.attacker.get_decision_history()
            if decision_history:
                self.results["strategic_decisions"].append(decision_history[-1])

        compromised_assets = [str(asset.asset_id) for asset in self.state.system.assets if asset.is_compromised]
        logger.info(f"After step {step + 1}: {len(compromised_assets)} assets compromised: {compromised_assets}")
        if compromised_assets:
            for asset_id in compromised_assets:
                asset = next((a for a in self.state.system.assets if str(a.asset_id) == asset_id), None)
                if asset:
                    logger.debug(
                        f"Asset {asset_id} details: is_compromised={asset.is_compromised}, compromise_time={asset._compromise_time}")

    def compare_strategies(self, defender_budget: int, num_steps: int, num_trials: int = 1, verbose: bool = False):
        print(
            f"\n{'=' * 80}\nCOMPARING STRATEGIES - Budget: ${defender_budget:,.2f}, Steps: {num_steps}, Trials: {num_trials}\n{'=' * 80}")
        self.defender_budget = defender_budget
        self.num_steps = num_steps
        strategies = {
            'CVSS-Only': CVSSOnlyStrategy(),
            'CVSS+Exploit': CVSSExploitAwareStrategy(),
            'Business Value': BusinessValueStrategy(),
            'Cost-Benefit': CostBenefitStrategy(),
            'Threat Intelligence': ThreatIntelligenceStrategy(),
            'RL Defender': RLAdaptiveThreatIntelligenceStrategy(),
            'Hybrid Defender': HybridStrategy(),
        }
        self.initialize_cost_cache()
        total_business_value = sum(asset.business_value for asset in self.system.assets)
        original_system = copy.deepcopy(self.system)
        output_dir = getattr(self, 'output_dir', 'apt3_simulation_results')
        os.makedirs(output_dir, exist_ok=True)

        # Create strategy-specific directories
        strategy_dirs = {name: os.path.join(output_dir, name.replace(' ', '_')) for name in strategies}
        for strategy_dir in strategy_dirs.values():
            os.makedirs(strategy_dir, exist_ok=True)

        aggregated_results = {
            strategy_name: {
                'trials': [],
                'mean_metrics': {},
                'std_metrics': {},
                'run_statistics': {
                    'protected_value_runs': [],
                    'lost_value_runs': [],
                    'value_preserved_runs': [],
                    'unpatched_critical_runs': [],
                    'total_patch_cost_runs': [],
                    'total_patches_runs': [],
                    'roi_runs': [],
                    'compromised_assets_runs': [],
                    'attack_success_rate_runs': [],
                    'successful_exploits_runs': [],
                    'total_attempts_runs': [],
                    'detection_coverage_runs': [],
                    'avg_time_to_detection_runs': [],
                    'attack_disruption_rate_runs': [],
                    'time_to_rtu_compromise_runs': [],
                    'spearphishing_success_rate_runs': [],
                    'credential_harvesting_count_runs': [],
                    'observations_collected_runs': [],
                    'learning_adaptations_runs': [],
                    'threat_level_changes_runs': [],
                    'predictions_made_runs': [],
                    'compromise_sequence_learned_runs': [],
                    'exploit_attempts_tracked_runs': [],
                    'techniques_learned_runs': [],
                    'hybrid_adaptations_runs': [],
                    'final_ti_weight_runs': [],
                    'final_rl_weight_runs': [],
                    'average_confidence_runs': [],
                    'decision_history_length_runs': [],
                    'ti_performance_runs': [],
                    'rl_performance_runs': [],
                    'hybrid_performance_runs': []
                }
            } for strategy_name in strategies
        }

        for trial in range(num_trials):
            print(f"\n{'=' * 40}\nRUNNING TRIAL {trial + 1}/{num_trials}\n{'=' * 40}")
            for strategy_name, strategy_obj in strategies.items():
                print(f"\n{'-' * 40}\nRUNNING STRATEGY: {strategy_name}\n{'-' * 40}")

                # Configure trial-specific logging
                trial_log_file = os.path.join(strategy_dirs[strategy_name], f"trial_{trial + 1}_log.txt")
                for handler in logger.handlers[:]:
                    logger.removeHandler(handler)
                file_handler = logging.FileHandler(trial_log_file)
                file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
                logger.addHandler(file_handler)
                logger.info(f"Starting simulation for defender strategy: {strategy_name} (Trial {trial + 1})")

                # Log whether this strategy has threat intelligence capabilities
                print(f"  Strategy uses basic vulnerability management approach")

                # Reinitialize attacker for each trial
                from classes.attacker_hybrid_apt3 import HybridGraphPOSGAttackerAPT3
                self.attacker = HybridGraphPOSGAttackerAPT3(
                    system=copy.deepcopy(original_system),
                    mitre_mapper=self.mitre_mapper,
                    cwe_canfollow_path="../data/CTI/raw/canfollow.json",
                    cost_aware=self.attacker.cost_aware,
                    detection_averse=self.attacker.detection_averse,
                    enhanced_exploit_priority=True,
                    sophistication_level=self.attacker_sophistication,
                    cost_calculator=self.cost_calculator,
                    cost_cache=self._cost_cache
                )

                # Initialize results tracking for this trial
                results = {
                    'protected_value': [], 'lost_value': [], 'value_preserved': [], 'unpatched_critical': [],
                    'total_patch_cost': 0.0, 'total_patches': 0, 'total_patch_time': 0.0, 'roi': 0.0,
                    'compromised_assets': [], 'final_compromised_assets': 0, 'exploit_attempts': [],
                    'exploit_events': [], 'rtu_compromised_step': None, 'attack_paths_used': [],
                    'attacker_metrics': {
                        'total_knowledge': len(self.vuln_lookup),
                        'total_exploits': 0,
                        'attack_success_rate': 0.0,
                        'total_cost': 0.0,
                        'total_gain': 0.0
                    },
                    'step_metrics': {
                        'protected_value_over_time': [],
                        'lost_value_over_time': [],
                        'value_preserved_over_time': [],
                        'compromised_assets_over_time': [],
                        'business_value_at_risk_over_time': []
                    },
                    'detection_metrics': {
                        'detection_coverage': 0.0,
                        'avg_time_to_detection': 0.0,
                        'attack_disruption_rate': 0.0,
                        'time_to_rtu_compromise': None
                    },
                    'apt3_metrics': {
                        'spearphishing_success_rate': 0.0,
                        'credential_harvesting_count': 0
                    },
                    'threat_intelligence_metrics': {
                        'observations_collected': 0,
                        'learning_adaptations': 0,
                        'threat_level_changes': 0,
                        'predictions_made': 0,
                        'compromise_sequence_learned': 0,
                        'exploit_attempts_tracked': 0,
                        'techniques_learned': 0
                    }
                }

                # Reset simulation state
                self.system = copy.deepcopy(original_system)
                from classes.state import State
                self.state = State(k=KillChainStage.RECONNAISSANCE.value, system=self.system)
                self.vuln_lookup = {}
                for asset in self.system.assets:
                    for comp in asset.components:
                        for vuln in comp.vulnerabilities:
                            from classes.state import create_vuln_key
                            vuln_key = create_vuln_key(vuln.cve_id, str(asset.asset_id), str(comp.id))
                            self.vuln_lookup[vuln_key] = (vuln, asset, comp)
                logger.info(
                    f"Reinitialized vuln_lookup with {len(self.vuln_lookup)} entries for strategy {strategy_name}, trial {trial + 1}")
                self._remaining_defender_budget = defender_budget
                self._remaining_attacker_budget = self.attacker_budget
                self._patched_cves = set()
                self._exploited_cves = set()
                self._exploit_failures = {}
                self._lateral_movement_attempts = {}
                self.mitre_techniques_used = set()
                self.mitre_techniques_detected = set()
                self.time_to_detection = {}
                self.attack_disrupted = False
                self.results = {k: [] if isinstance(v, list) or v is None else v for k, v in self.results.items()}
                self.results['attack_paths_used'] = []
                self.attacker.attack_graph.build_attack_graph()
                
                # Initialize strategy
                strategy_obj.initialize(self.state, self._cost_cache)
                strategy_obj.state = self.state
                
                # Special handling for RL Defender strategy
                if strategy_name == 'RL Defender':
                    # Load the most recent trained Q-table
                    rl_results_dir = "src/rl_defender_training_results"
                    if os.path.exists(rl_results_dir):
                        # Find the most recent training directory
                        training_dirs = [d for d in os.listdir(rl_results_dir) if d.startswith('rl_defender_training_')]
                        if training_dirs:
                            latest_dir = max(training_dirs)
                            q_table_path = os.path.join(rl_results_dir, latest_dir, "q_table.pkl")
                            if os.path.exists(q_table_path):
                                try:
                                    with open(q_table_path, 'rb') as f:
                                        strategy_obj.q_table = pickle.load(f)
                                    print(f"  Loaded trained Q-table from {q_table_path}")
                                    print(f"  Q-table entries: {len(strategy_obj.q_table)}")
                                except Exception as e:
                                    print(f"  Warning: Could not load Q-table: {e}")
                            else:
                                print(f"  Warning: Q-table file not found at {q_table_path}")
                        else:
                            print("  Warning: No RL training directories found")
                    else:
                        print("  Warning: RL training results directory not found")
                
                # Special handling for Hybrid Defender strategy
                if strategy_name == 'Hybrid Defender':
                    # Load the most recent trained Q-table for the RL component
                    rl_results_dir = "src/rl_defender_training_results"
                    if os.path.exists(rl_results_dir):
                        # Find the most recent training directory
                        training_dirs = [d for d in os.listdir(rl_results_dir) if d.startswith('rl_defender_training_')]
                        if training_dirs:
                            latest_dir = max(training_dirs)
                            q_table_path = os.path.join(rl_results_dir, latest_dir, "q_table.pkl")
                            if os.path.exists(q_table_path):
                                try:
                                    with open(q_table_path, 'rb') as f:
                                        strategy_obj.rl_defender.q_table = pickle.load(f)
                                    print(f"  Loaded trained Q-table for Hybrid RL component from {q_table_path}")
                                    print(f"  Q-table entries: {len(strategy_obj.rl_defender.q_table)}")
                                except Exception as e:
                                    print(f"  Warning: Could not load Q-table for Hybrid RL component: {e}")
                            else:
                                print(f"  Warning: Q-table file not found at {q_table_path}")
                        else:
                            print("  Warning: No RL training directories found for Hybrid RL component")
                    else:
                        print("  Warning: RL training results directory not found for Hybrid RL component")
                    
                    print(f"  Hybrid Strategy initialized with TI weight: {strategy_obj.threat_intelligence_weight:.3f}, RL weight: {strategy_obj.rl_weight:.3f}")
                
                self.reset_exploit_status()

                # Track initial threat intelligence state
                initial_observations = len(strategy_obj.attack_observations) if hasattr(strategy_obj, 'attack_observations') else 0
                initial_threat_levels = dict(strategy_obj.asset_threat_levels) if hasattr(strategy_obj, 'asset_threat_levels') else {}

                # Run simulation steps
                total_patch_cost = 0.0
                total_patches = 0
                total_patch_time = 0.0
                termination_reason = "campaign_horizon_elapsed"
                
                for step in range(num_steps):
                    # Instead of passing select_patches, pass the full strategy object
                    self.run_step(strategy_obj, step, verbose)
                    step_metrics = self.calculate_metrics(self.state, total_patch_cost, total_patches, verbose)
                    total_patch_cost += sum(self.results["dollar_costs"]["patch_costs"][-1:])
                    total_patches += len(self.results["patched_vulns"][-1:])
                    total_patch_time += len(self.results["patched_vulns"][-1:])
                    results['step_metrics']['protected_value_over_time'].append(step_metrics['protected_value'])
                    results['step_metrics']['lost_value_over_time'].append(step_metrics['lost_value'])
                    results['step_metrics']['value_preserved_over_time'].append(step_metrics['value_preserved'])
                    results['step_metrics']['compromised_assets_over_time'].append(
                        step_metrics['compromised_assets_count'])
                    results['step_metrics']['business_value_at_risk_over_time'].append(
                        sum(asset.business_value for asset in self.state.system.assets if not asset.is_compromised))
                    
                    # Check for early termination
                    should_terminate, reason = self.should_terminate_simulation(step, num_steps, total_patch_cost)
                    if should_terminate:
                        termination_reason = reason
                        break

                # Calculate final metrics
                final_metrics = self.calculate_metrics(self.state, total_patch_cost, total_patches, verbose)
                # Override total_patches with the actual count
                final_metrics['total_patches'] = total_patches
                
                if final_metrics['lost_value'] > total_business_value:
                    final_metrics['lost_value'] = total_business_value
                    final_metrics['protected_value'] = 0.0
                    final_metrics['value_preserved'] = max(0.0,
                                                           total_business_value - final_metrics['total_patch_cost'])
                detection_metrics = self.calculate_detection_metrics(self.results)
                time_to_rtu = self.calculate_time_to_rtu_compromise(self.results)
                successful_exploits = len([e for e in self.results["exploit_attempts"] if e.get('success', False)])
                total_attempts = len(self.results["exploit_attempts"])

                # Get strategic metrics
                strategic_metrics = self.attacker.get_strategic_metrics() if hasattr(self.attacker,
                                                                                     'get_strategic_metrics') else {}

                # Calculate threat intelligence specific metrics
                threat_intel_metrics = {
                    'observations_collected': 0,
                    'learning_adaptations': 0,
                    'threat_level_changes': 0,
                    'predictions_made': 0,
                    'compromise_sequence_learned': 0,
                    'exploit_attempts_tracked': 0,
                    'techniques_learned': 0
                }
                if True:
                    final_observations = len(strategy_obj.attack_observations) if hasattr(strategy_obj, 'attack_observations') else 0
                    final_threat_levels = dict(strategy_obj.asset_threat_levels) if hasattr(strategy_obj, 'asset_threat_levels') else {}
                    threat_intel_metrics.update({
                        'observations_collected': final_observations,
                        'learning_adaptations': final_observations - initial_observations,
                        'threat_level_changes': sum(1 for asset_id in final_threat_levels
                                                    if abs(
                            final_threat_levels[asset_id] - initial_threat_levels.get(asset_id, 0.3)) > 0.1),
                        'predictions_made': len(strategy_obj.predict_next_targets()) if hasattr(strategy_obj,
                                                                                                'predict_next_targets') else 0,
                        'compromise_sequence_learned': len(strategy_obj.compromise_sequence) if hasattr(strategy_obj, 'compromise_sequence') else 0,
                        'exploit_attempts_tracked': len(strategy_obj.exploit_attempt_history) if hasattr(strategy_obj, 'exploit_attempt_history') else 0,
                        'techniques_learned': len(strategy_obj.technique_frequency) if hasattr(strategy_obj, 'technique_frequency') else 0
                    })

                # Calculate hybrid strategy specific metrics
                hybrid_metrics = {
                    'hybrid_adaptations': 0,
                    'final_ti_weight': 0.4,
                    'final_rl_weight': 0.6,
                    'average_confidence': 0.0,
                    'decision_history_length': 0,
                    'ti_performance': 0.0,
                    'rl_performance': 0.0,
                    'hybrid_performance': 0.0
                }
                if strategy_name == 'Hybrid Defender':
                    hybrid_metrics.update({
                        'hybrid_adaptations': strategy_obj.adaptations,
                        'final_ti_weight': strategy_obj.threat_intelligence_weight,
                        'final_rl_weight': strategy_obj.rl_weight,
                        'average_confidence': np.mean([d['hybrid_decision']['confidence'] for d in strategy_obj.decision_history]) if strategy_obj.decision_history else 0.0,
                        'decision_history_length': len(strategy_obj.decision_history),
                        'ti_performance': np.mean(strategy_obj.threat_intelligence_performance) if strategy_obj.threat_intelligence_performance else 0.0,
                        'rl_performance': np.mean(strategy_obj.rl_performance) if strategy_obj.rl_performance else 0.0,
                        'hybrid_performance': np.mean(strategy_obj.hybrid_performance) if strategy_obj.hybrid_performance else 0.0
                    })

                # Update results
                results.update({
                    'protected_value': [final_metrics['protected_value']],
                    'lost_value': [final_metrics['lost_value']],
                    'value_preserved': [final_metrics['value_preserved']],
                    'unpatched_critical': [final_metrics['unpatched_critical']],
                    'total_patch_cost': final_metrics['total_patch_cost'],
                    'total_patches': final_metrics['total_patches'],
                    'total_patch_time': total_patch_time,
                    'roi': final_metrics['roi'],
                    'compromised_assets': self.results['compromised_assets'],
                    'final_compromised_assets': final_metrics['compromised_assets_count'],
                    'exploit_attempts': self.results['exploit_attempts'],
                    'exploit_events': self.results['exploit_events'],
                    'rtu_compromised_step': self.results['rtu_compromised_step'],
                    'attack_paths_used': self.results['attack_paths_used'],
                    'attacker_metrics': {
                        'total_knowledge': len(self.vuln_lookup),
                        'total_exploits': successful_exploits,
                        'attack_success_rate': (successful_exploits / max(1, total_attempts)) * 100,
                        'total_cost': sum(self.results["dollar_costs"]["exploit_costs"]),
                        'total_gain': final_metrics['lost_value']
                    },
                    'detection_metrics': {
                        'detection_coverage': detection_metrics['detection_coverage'],
                        'avg_time_to_detection': detection_metrics['avg_time_to_detection'],
                        'attack_disruption_rate': detection_metrics['attack_disruption_rate'],
                        'time_to_rtu_compromise': time_to_rtu if time_to_rtu is not None else num_steps + 1
                    },
                    'apt3_metrics': {
                        'spearphishing_success_rate': final_metrics['spearphishing_success_rate'] * 100,
                        'credential_harvesting_count': final_metrics['credential_harvesting_count']
                    },
                    'threat_intelligence_metrics': threat_intel_metrics,
                    'hybrid_metrics': hybrid_metrics
                })

                # Store trial results
                aggregated_results[strategy_name]['trials'].append(results)
                aggregated_results[strategy_name]['run_statistics']['protected_value_runs'].append(
                    final_metrics['protected_value'])
                aggregated_results[strategy_name]['run_statistics']['lost_value_runs'].append(
                    final_metrics['lost_value'])
                aggregated_results[strategy_name]['run_statistics']['value_preserved_runs'].append(
                    final_metrics['value_preserved'])
                aggregated_results[strategy_name]['run_statistics']['unpatched_critical_runs'].append(
                    final_metrics['unpatched_critical'])
                aggregated_results[strategy_name]['run_statistics']['total_patch_cost_runs'].append(
                    final_metrics['total_patch_cost'])
                aggregated_results[strategy_name]['run_statistics']['total_patches_runs'].append(
                    final_metrics['total_patches'])
                aggregated_results[strategy_name]['run_statistics']['roi_runs'].append(final_metrics['roi'])
                aggregated_results[strategy_name]['run_statistics']['compromised_assets_runs'].append(
                    final_metrics['compromised_assets_count'])
                aggregated_results[strategy_name]['run_statistics']['attack_success_rate_runs'].append(
                    (successful_exploits / max(1, total_attempts)) * 100)
                aggregated_results[strategy_name]['run_statistics']['successful_exploits_runs'].append(
                    successful_exploits)
                aggregated_results[strategy_name]['run_statistics']['total_attempts_runs'].append(total_attempts)
                aggregated_results[strategy_name]['run_statistics']['detection_coverage_runs'].append(
                    detection_metrics['detection_coverage'])
                aggregated_results[strategy_name]['run_statistics']['avg_time_to_detection_runs'].append(
                    detection_metrics['avg_time_to_detection'])
                aggregated_results[strategy_name]['run_statistics']['attack_disruption_rate_runs'].append(
                    detection_metrics['attack_disruption_rate'])
                aggregated_results[strategy_name]['run_statistics']['time_to_rtu_compromise_runs'].append(
                    time_to_rtu if time_to_rtu is not None else num_steps + 1)
                aggregated_results[strategy_name]['run_statistics']['spearphishing_success_rate_runs'].append(
                    final_metrics['spearphishing_success_rate'] * 100)
                aggregated_results[strategy_name]['run_statistics']['credential_harvesting_count_runs'].append(
                    final_metrics['credential_harvesting_count'])
                aggregated_results[strategy_name]['run_statistics']['observations_collected_runs'].append(
                    threat_intel_metrics['observations_collected'])
                aggregated_results[strategy_name]['run_statistics']['learning_adaptations_runs'].append(
                    threat_intel_metrics['learning_adaptations'])
                aggregated_results[strategy_name]['run_statistics']['threat_level_changes_runs'].append(
                    threat_intel_metrics['threat_level_changes'])
                aggregated_results[strategy_name]['run_statistics']['predictions_made_runs'].append(
                    threat_intel_metrics['predictions_made'])
                aggregated_results[strategy_name]['run_statistics']['compromise_sequence_learned_runs'].append(
                    threat_intel_metrics['compromise_sequence_learned'])
                aggregated_results[strategy_name]['run_statistics']['exploit_attempts_tracked_runs'].append(
                    threat_intel_metrics['exploit_attempts_tracked'])
                aggregated_results[strategy_name]['run_statistics']['techniques_learned_runs'].append(
                    threat_intel_metrics['techniques_learned'])
                
                # Add hybrid metrics to run statistics
                aggregated_results[strategy_name]['run_statistics']['hybrid_adaptations_runs'].append(
                    hybrid_metrics['hybrid_adaptations'])
                aggregated_results[strategy_name]['run_statistics']['final_ti_weight_runs'].append(
                    hybrid_metrics['final_ti_weight'])
                aggregated_results[strategy_name]['run_statistics']['final_rl_weight_runs'].append(
                    hybrid_metrics['final_rl_weight'])
                aggregated_results[strategy_name]['run_statistics']['average_confidence_runs'].append(
                    hybrid_metrics['average_confidence'])
                aggregated_results[strategy_name]['run_statistics']['decision_history_length_runs'].append(
                    hybrid_metrics['decision_history_length'])
                aggregated_results[strategy_name]['run_statistics']['ti_performance_runs'].append(
                    hybrid_metrics['ti_performance'])
                aggregated_results[strategy_name]['run_statistics']['rl_performance_runs'].append(
                    hybrid_metrics['rl_performance'])
                aggregated_results[strategy_name]['run_statistics']['hybrid_performance_runs'].append(
                    hybrid_metrics['hybrid_performance'])

                # Save trial results
                trial_result_file = os.path.join(strategy_dirs[strategy_name], f"trial_{trial + 1}_results.json")
                with open(trial_result_file, 'w') as f:
                    json.dump(results, f, indent=4, default=str)
                print(f"Saved trial {trial + 1} results for {strategy_name} to {trial_result_file}")

                # Print trial results
                print(f"\nStrategy {strategy_name} Trial {trial + 1} Complete:")
                print(f"  Total Patches: {results['total_patches']}")
                print(f"  Total Cost: ${results['total_patch_cost']:,.2f}")
                print(f"  Protected Value: ${final_metrics['protected_value']:,.2f}")
                print(f"  Lost Value: ${final_metrics['lost_value']:,.2f}")
                print(f"  Value Preserved: ${final_metrics['value_preserved']:,.2f}")
                print(f"  ROI: {final_metrics['roi']:.1f}%")
                print(f"  Compromised Assets: {results['final_compromised_assets']}")
                print(f"  Spearphishing Success Rate: {final_metrics['spearphishing_success_rate'] * 100:.1f}%")
                print(f"  Credential Harvesting Successes: {final_metrics['credential_harvesting_count']}")
                print(f"  Attack Paths Used: {[p['path_id'] for p in self.results['attack_paths_used']]}")
                
                if strategy_name == 'Hybrid Defender':
                    print(f"  === Hybrid Strategy Results ===")
                    print(f"  Hybrid Adaptations: {hybrid_metrics['hybrid_adaptations']}")
                    print(f"  Final TI Weight: {hybrid_metrics['final_ti_weight']:.3f}")
                    print(f"  Final RL Weight: {hybrid_metrics['final_rl_weight']:.3f}")
                    print(f"  Average Confidence: {hybrid_metrics['average_confidence']:.3f}")
                    print(f"  Decision History Length: {hybrid_metrics['decision_history_length']}")
                    print(f"  TI Performance: {hybrid_metrics['ti_performance']:.3f}")
                    print(f"  RL Performance: {hybrid_metrics['rl_performance']:.3f}")
                    print(f"  Hybrid Performance: {hybrid_metrics['hybrid_performance']:.3f}")

                # Close trial log handler
                for handler in logger.handlers[:]:
                    handler.close()
                    logger.removeHandler(handler)

        # Aggregate and summarize results
        viz_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        for strategy_name in strategies:
            metrics = [
                'protected_value', 'lost_value', 'value_preserved', 'unpatched_critical',
                'total_patch_cost', 'total_patches', 'roi', 'compromised_assets',
                'attack_success_rate', 'successful_exploits', 'total_attempts',
                'detection_coverage', 'avg_time_to_detection', 'attack_disruption_rate',
                'time_to_rtu_compromise', 'spearphishing_success_rate', 'credential_harvesting_count',
                'observations_collected', 'learning_adaptations', 'threat_level_changes',
                'predictions_made', 'compromise_sequence_learned', 'exploit_attempts_tracked',
                'techniques_learned', 'hybrid_adaptations', 'final_ti_weight', 'final_rl_weight',
                'average_confidence', 'decision_history_length', 'ti_performance', 'rl_performance',
                'hybrid_performance'
            ]
            for metric in metrics:
                key = metric + '_runs'
                values = aggregated_results[strategy_name]['run_statistics'][key]
                valid_values = [v for v in values if v is not None]
                if metric == 'time_to_rtu_compromise':
                    valid_values = [v for v in valid_values if v != num_steps + 1]
                    aggregated_results[strategy_name]['mean_metrics'][metric] = np.mean(
                        valid_values) if valid_values else None
                    aggregated_results[strategy_name]['std_metrics'][metric] = np.std(
                        valid_values) if valid_values else 0.0
                else:
                    aggregated_results[strategy_name]['mean_metrics'][metric] = np.mean(
                        valid_values) if valid_values else 0.0
                    aggregated_results[strategy_name]['std_metrics'][metric] = np.std(
                        valid_values) if valid_values else 0.0

        # Display comparison summary
        print("\n" + "=" * 180)
        print(f"STRATEGY COMPARISON SUMMARY - Total Business Value: ${total_business_value:,.2f}, {num_trials} Trials")
        print("=" * 180)
        headers = [
            "Strategy", "Avg Protected Value", "Avg Lost Value", "Avg Value Preserved", "Avg Patches",
            "Avg Patch Cost", "Avg ROI", "Avg Compromised", "Attack Success", "Detection Coverage",
            "Avg Time to Detection", "Attack Disruption", "RTU Compromise Time", "Spearphishing SR",
            "Cred Harvesting", "Observations"
        ]
        print(
            "{:<15} {:<18} {:<18} {:<18} {:<10} {:<13} {:<8} {:<13} {:<13} {:<15} {:<18} {:<15} {:<18} {:<15} {:<15} {:<12}".format(
                *headers))
        print("-" * 180)
        for strategy_name in strategies:
            rtu_time = aggregated_results[strategy_name]['mean_metrics']['time_to_rtu_compromise']
            rtu_display = f"{rtu_time:.1f}" if rtu_time is not None and not np.isnan(rtu_time) else "None"
            print(
                "{:<15} ${:<17,.2f} ${:<17,.2f} ${:<17,.2f} {:<10.1f} ${:<12,.2f} {:<7.1f}% {:<12.1f} {:<12.1f}% {:<14.1f}% {:<17.2f} {:<14.1f}% {:<17} {:<14.1f}% {:<14.1f} {:<12.1f}".format(
                    strategy_name,
                    aggregated_results[strategy_name]['mean_metrics']['protected_value'],
                    aggregated_results[strategy_name]['mean_metrics']['lost_value'],
                    aggregated_results[strategy_name]['mean_metrics']['value_preserved'],
                    aggregated_results[strategy_name]['mean_metrics']['total_patches'],
                    aggregated_results[strategy_name]['mean_metrics']['total_patch_cost'],
                    aggregated_results[strategy_name]['mean_metrics']['roi'],
                    aggregated_results[strategy_name]['mean_metrics']['compromised_assets'],
                    aggregated_results[strategy_name]['mean_metrics']['attack_success_rate'],
                    aggregated_results[strategy_name]['mean_metrics']['detection_coverage'],
                    aggregated_results[strategy_name]['mean_metrics']['avg_time_to_detection'],
                    aggregated_results[strategy_name]['mean_metrics']['attack_disruption_rate'],
                    rtu_display,
                    aggregated_results[strategy_name]['mean_metrics']['spearphishing_success_rate'],
                    aggregated_results[strategy_name]['mean_metrics']['credential_harvesting_count'],
                    aggregated_results[strategy_name]['mean_metrics']['observations_collected']
                ))

        # Highlight best strategy
        best_strategy_name = max(strategies.keys(),
                                 key=lambda s: aggregated_results[s]['mean_metrics']['value_preserved'])
        print(f"\nBest performing strategy (by mean value preserved): {best_strategy_name}")

        return aggregated_results
