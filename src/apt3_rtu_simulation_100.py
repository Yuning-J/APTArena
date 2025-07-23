import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import random
import copy
import time
import argparse
import json
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import uuid
import pickle

from classes.state import Asset, State, System, Vulnerability, KillChainStage, create_vuln_key
from classes.mitre import MitreMapper
from classes.payoff import PayoffFunctions
from classes.transition import TransitionFunction
from classes.threatIntelligence import ThreatIntelligenceProcessor
from classes.cost import CostCalculator
from classes.attacker_hybrid_apt3 import HybridGraphPOSGAttackerAPT3
from classes.defender_posg import DefenderPOMDPPolicy
from classes.patching_strategies import (
    CVSSOnlyStrategy,
    CVSSExploitAwareStrategy,
    BusinessValueStrategy,
    CostBenefitStrategy,
    ThreatIntelligenceStrategy
)
from data_loader import load_data
from classes.hybrid_strategy import HybridStrategy
from src.RL_defender_strategy import RLAdaptiveThreatIntelligenceStrategy

# Create visualization directory
viz_dir = "visualization"
if not os.path.exists(viz_dir):
    os.makedirs(viz_dir)

logging.basicConfig(level=logging.DEBUG, filename="apt3_rtu_simulation_log.txt",
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ExploitEvent:
    vuln_id: str
    step: int
    asset_id: str
    success: bool
    technique: str

    def to_dict(self):
        return {
            'vuln_id': self.vuln_id,
            'step': self.step,
            'asset_id': self.asset_id,
            'success': self.success,
            'technique': self.technique
        }

class APT3RTUSimulation:
    _mitre_mapper_cache = None

    def __init__(self, data_file: str, num_steps: int, defender_budget: int, attacker_budget: int,
                 psi: float = 1.0, cost_aware_attacker: bool = True, cost_aware_defender: bool = True,
                 detection_averse: bool = True, gamma: float = 0.9, business_values_file: str = None,
                 use_hmm: bool = False, attacker_sophistication: float = 0.9, cost_cache_file: str = None):
        logger.info("Initializing APT3RTUSimulation")
        self._exploit_failures = {}
        self._cost_cache = {}
        self.cost_cache_file = cost_cache_file
        self.cost_calculator = CostCalculator()

        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if defender_budget < 0 or attacker_budget < 0:
            raise ValueError("Budgets must be non-negative")
        self.num_steps = num_steps
        self.defender_budget = defender_budget
        self._remaining_defender_budget = defender_budget
        self.attacker_budget = attacker_budget
        self._remaining_attacker_budget = attacker_budget
        self._patched_cves = set()
        self._exploited_cves = set()
        self.attacker_sophistication = max(0.0, min(1.0, attacker_sophistication))

        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file {data_file} not found")
        self.system = load_data(data_file)
        if not self.system.assets:
            raise ValueError("System must have at least one asset")

        self.vuln_lookup = {}
        for asset in self.system.assets:
            for comp in asset.components:
                for vuln in comp.vulnerabilities:
                    vuln_key = create_vuln_key(vuln.cve_id, str(asset.asset_id), str(comp.id))
                    self.vuln_lookup[vuln_key] = (vuln, asset, comp)
        logger.info(f"Initialized vulnerability lookup with {len(self.vuln_lookup)} entries")

        if business_values_file and os.path.exists(business_values_file):
            with open(business_values_file, 'r') as f:
                business_values = json.load(f)
            for asset in self.system.assets:
                if str(asset.asset_id) in business_values:
                    asset.business_value = business_values[str(asset.asset_id)]
        else:
            for asset in self.system.assets:
                if not hasattr(asset, 'business_value') or asset.business_value is None:
                    criticality = getattr(asset, 'criticality_level', 1)
                    asset.business_value = [10000, 50000, 100000, 500000, 1000000][min(criticality - 1, 4)]

        self.state = State(k=KillChainStage.RECONNAISSANCE.value, system=self.system)
        if APT3RTUSimulation._mitre_mapper_cache is None:
            APT3RTUSimulation._mitre_mapper_cache = MitreMapper()
        self.mitre_mapper = APT3RTUSimulation._mitre_mapper_cache

        if self.cost_cache_file and os.path.exists(self.cost_cache_file):
            self._load_cost_cache()
        else:
            self.initialize_cost_cache()

        self.attacker = HybridGraphPOSGAttackerAPT3(
            system=self.system,
            mitre_mapper=self.mitre_mapper,
            cwe_canfollow_path="../data/CTI/raw/canfollow.json",
            cost_aware=cost_aware_attacker,
            detection_averse=detection_averse,
            enhanced_exploit_priority=True,
            sophistication_level=attacker_sophistication,
            cost_calculator=self.cost_calculator,
            cost_cache=self._cost_cache
        )
        self.rtu_id = "8"

        self.defender_policy = DefenderPOMDPPolicy(
            budget=defender_budget,
            threat_weight=1.5,
            cost_aware=cost_aware_defender,
            recent_attack_weight=2.0,
            use_hmm=use_hmm
        )
        self.transition = TransitionFunction(exploit_success_base_rate=0.5)
        self.payoffs = PayoffFunctions(attack_cost_per_action=1.0, base_patch_cost=1.0, psi=psi)
        self.threat_processor = ThreatIntelligenceProcessor(update_frequency=0.1)
        self.gamma = gamma

        self.results = {
            "attacker_payoff": [], "defender_payoff": [], "operational_costs": [], "risk_costs": [],
            "detection_probabilities": [], "compromised_assets": [], "patched_vulns": [],
            "kill_chain_stages": [], "exploit_actions": [], "discovery_actions": [], "persistence_actions": [],
            "dollar_costs": {"patch_costs": [], "exploit_costs": []}, "business_value_at_risk": [],
            "belief_states": {"attacker": [], "defender": []}, "total_value_preserved": [], "residual_risk": [],
            "time_to_patch_critical": {}, "policy_adaptability": [], "noise_perturbed_metrics": {},
            "attacker_actions": [], "exploit_events": [], "exploit_attempts": [], "mitre_techniques_used": [],
            "mitre_techniques_detected": [], "time_to_detection": [], "attack_disruption_rate": [],
            "rtu_compromised_step": None, "attack_paths_used": [], "strategic_decisions": [],
            "spearphishing_attempts": [], "credential_harvesting_successes": []
        }
        self.mitre_techniques_used = set()
        self.mitre_techniques_detected = set()
        self.time_to_detection = {}
        self.attack_disrupted = False
        self.current_step = 0
        self._lateral_movement_attempts = {}

        if self.cost_cache_file:
            self._save_cost_cache()

        logger.info(f"APT3RTUSimulation initialized with {len(self.system.assets)} assets")

    def _load_cost_cache(self):
        try:
            with open(self.cost_cache_file, 'r') as f:
                cached_data = json.load(f)
            required_keys = {'patch_costs', 'exploit_costs', 'exploit_losses', 'vulnerability_info'}
            if not all(key in cached_data for key in required_keys):
                logger.warning(f"Invalid cost cache file {self.cost_cache_file}. Reinitializing cache.")
                self.initialize_cost_cache()
                return
            cached_vuln_keys = set(cached_data['vulnerability_info'].keys())
            current_vuln_keys = set(self.vuln_lookup.keys())
            if cached_vuln_keys != current_vuln_keys:
                logger.warning(f"Cost cache mismatch. Reinitializing cache.")
                self.initialize_cost_cache()
                return
            self._cost_cache = cached_data
            logger.info(f"Loaded cost cache from {self.cost_cache_file}")
        except Exception as e:
            logger.error(f"Error loading cost cache: {e}. Reinitializing cache.")
            self.initialize_cost_cache()

    def _save_cost_cache(self):
        try:
            serializable_cache = {
                'patch_costs': self._cost_cache['patch_costs'],
                'exploit_costs': self._cost_cache['exploit_costs'],
                'exploit_losses': self._cost_cache['exploit_losses'],
                'vulnerability_info': {
                    key: {k: v for k, v in info.items() if k not in ['asset', 'component']}
                    for key, info in self._cost_cache['vulnerability_info'].items()
                }
            }
            with open(self.cost_cache_file, 'w') as f:
                json.dump(serializable_cache, f, indent=4)
            logger.info(f"Saved cost cache to {self.cost_cache_file}")
        except Exception as e:
            logger.error(f"Error saving cost cache: {e}")

    def initialize_cost_cache(self):
        cost_cache = {
            'patch_costs': {}, 'exploit_costs': {}, 'exploit_losses': {}, 'vulnerability_info': {}
        }
        for asset in self.system.assets:
            business_value = getattr(asset, 'business_value', getattr(asset, 'criticality_level', 3) * 5000)
            for comp in asset.components:
                for vuln in comp.vulnerabilities:
                    cve_id = getattr(vuln, 'cve_id', '')
                    if not cve_id or cve_id.lower() == 'unknown':
                        continue
                    vuln_key = create_vuln_key(cve_id, str(asset.asset_id), str(comp.id))
                    cost_cache['vulnerability_info'][vuln_key] = {
                        'cvss': getattr(vuln, 'cvss', 5.0), 'epss': getattr(vuln, 'epss', 0.1),
                        'exploit': getattr(vuln, 'exploit', False), 'ransomWare': getattr(vuln, 'ransomWare', False),
                        'component_id': str(comp.id), 'asset_id': str(asset.asset_id), 'asset_name': asset.name,
                        'business_value': business_value, 'asset': asset, 'component': comp
                    }
                    patch_cost = self.cost_calculator.calculate_patch_cost(vuln, self.state, asset, component_id=str(comp.id))
                    exploit_cost = self.cost_calculator.calculate_exploit_cost(vuln, self.state, asset, component_id=str(comp.id))
                    exploit_loss = self.cost_calculator.calculate_exploit_loss(vuln, self.state, asset, component_id=str(comp.id))
                    cost_cache['patch_costs'][vuln_key] = patch_cost
                    cost_cache['exploit_costs'][vuln_key] = exploit_cost
                    cost_cache['exploit_losses'][vuln_key] = exploit_loss
        self._cost_cache = cost_cache
        logger.info(f"Cost cache initialized with {len(cost_cache['patch_costs'])} entries")
        return cost_cache

    def update_attack_graph(self, state):
        try:
            self.attacker.state = state
            changes_detected = False
            for asset in state.system.assets:
                if asset.is_compromised:
                    changes_detected = True
                    break
                for comp in asset.components:
                    for vuln in comp.vulnerabilities:
                        if vuln.is_patched or getattr(vuln, 'is_exploited', False):
                            changes_detected = True
                            break
                    if changes_detected:
                        break
                if changes_detected:
                    break
            if changes_detected:
                logger.info("System state changed, rebuilding attack graph")
                self.attacker.attack_graph.build_attack_graph()
                self.attacker.path_cache = []
                self.attacker.cache_last_updated = 0
                logger.debug("Attack graph rebuilt successfully")
            else:
                logger.debug("No state changes detected, skipping attack graph rebuild")
        except Exception as e:
            logger.error(f"Error updating attack graph: {e}")

    def _should_skip_due_to_failures(self, vuln_id, asset_id, component_id):
        vuln_key = create_vuln_key(vuln_id, str(asset_id), str(component_id))
        failure_count = self.attacker.strategic_manager.exploit_failures.get(vuln_key, 0)
        STANDARD_MAX_FAILURES = 3
        PRIORITY_MAX_FAILURES = 5
        max_failures = PRIORITY_MAX_FAILURES if vuln_id in {'CVE-2018-13379', 'ZERO-DAY-001', 'CVE-2015-3113'} else STANDARD_MAX_FAILURES
        if failure_count >= max_failures:
            logger.info(f"Skipping {vuln_key} due to {failure_count} previous failures (threshold: {max_failures})")
            return True
        return False

    def get_next_attack_action(self, state):
        if state is None or not isinstance(state, State):
            logger.error("Invalid state provided: state is None or not a State instance")
            return {'action_type': 'pause', 'reason': 'invalid_state'}
        if not hasattr(state, 'system') or state.system is None or not state.system.assets:
            logger.error("State has no valid system or assets")
            return {'action_type': 'pause', 'reason': 'missing_system_or_assets'}

        try:
            self.attacker.state = state
            recon_count = sum(1 for action in self.results['exploit_attempts']
                              if action.get('action_type') == 'reconnaissance' and action.get('success'))
            if recon_count > 5 and state.k == KillChainStage.RECONNAISSANCE.value:
                logger.info("Reconnaissance overused, forcing initial_access action")
                action = self.attacker.select_action(state, force_action_type='initial_access')
            else:
                action = self.attacker.select_action(state)

            if not action or 'action_type' not in action:
                logger.warning("Invalid action returned, attempting fallback initial_access")
                action = self.attacker.select_action(state, force_action_type='initial_access')
                if not action or 'action_type' not in action:
                    logger.error("Failed to generate valid action")
                    self.attacker.path_failures += 1
                    return {'action_type': 'pause', 'reason': 'invalid_action'}

            if action.get('action_type') in ['exploit', 'exploitation', 'initial_access', 'lateral_movement']:
                vuln_key = create_vuln_key(action.get('target_vuln', ''),
                                           str(action.get('target_asset', '')),
                                           str(action.get('target_component', '0')))
                cost = self._cost_cache['exploit_costs'].get(vuln_key, 50.0)
                if cost > self._remaining_attacker_budget:
                    logger.warning(
                        f"Insufficient budget for {action['action_type']}: cost={cost}, budget={self._remaining_attacker_budget}")
                    return {'action_type': 'pause', 'reason': 'insufficient_budget'}

            if hasattr(state, 'attacker_suggested_stage') and state.attacker_suggested_stage:
                state.process_attacker_stage_suggestion()

            logger.debug(f"Generated action: {action}, state.k: {KillChainStage(state.k).name}")
            return action
        except Exception as e:
            logger.error(f"Error getting next attack action: {e}", exc_info=True)
            self.attacker.path_failures += 1
            return {'action_type': 'pause', 'reason': f'action_error_{str(e)}'}

    def execute_attack_action(self, state, action):
        if not action or not isinstance(action, dict):
            logger.error(f"Invalid action: {action}")
            return {'action_result': False, 'reason': 'invalid_action', 'action_type': 'none'}

        if 'action_type' not in action:
            logger.error(f"Missing action_type in action: {action}")
            return {'action_result': False, 'reason': 'missing_action_type', 'action_type': 'none'}

        try:
            if action['action_type'] == 'pause':
                logger.info(f"Attacker paused: {action.get('reason', 'unknown')}")
                return {'action_result': False, 'reason': action.get('reason', 'pause'), 'action_type': 'pause'}
            elif action['action_type'] in ['exploit', 'exploitation']:
                return self._execute_exploit_action(state, action)
            elif action['action_type'] == 'initial_access':
                return self._execute_initial_access_action(state, action)
            elif action['action_type'] in ['lateral_movement']:
                return self._execute_movement_action(state, action)
            elif action['action_type'] in ['reconnaissance', 'persistence']:
                return self._execute_other_action(state, action)

            return {'action_result': False, 'reason': 'unsupported_action', 'action_type': action['action_type']}
        except Exception as e:
            logger.error(f"Error executing attack action: {e}", exc_info=True)
            return {'action_result': False, 'reason': f'execution_error_{str(e)}',
                    'action_type': action.get('action_type', 'none')}

    def _execute_initial_access_action(self, state, action):
        try:
            logger.info(f"Executing initial access to asset {action.get('target_asset')}")
            required_fields = ['target_asset', 'target_vuln', 'target_component', 'probability']
            missing_fields = [field for field in required_fields if not action.get(field)]
            if missing_fields:
                logger.error(f"Missing fields: {missing_fields}")
                return {
                    'action_type': 'initial_access', 'action_result': False,
                    'reason': f'missing_fields_{missing_fields}', 'target_asset': action.get('target_asset'),
                    'target_vuln': action.get('target_vuln')
                }

            vuln_key = create_vuln_key(action.get('target_vuln'), str(action.get('target_asset')),
                                       str(action.get('target_component')))
            if vuln_key not in self.vuln_lookup:
                logger.error(f"Invalid vulnerability: {vuln_key}")
                return {
                    'action_type': 'initial_access', 'action_result': False, 'reason': 'invalid_target',
                    'target_asset': action.get('target_asset'), 'target_vuln': action.get('target_vuln'),
                    'vuln_key': vuln_key
                }

            vuln, asset, comp = self.vuln_lookup[vuln_key]
            logger.debug(f"Initial access vuln: {vuln.cve_id}, asset: {asset.asset_id}, comp: {comp.id}")

            if vuln.is_patched:
                logger.debug(f"Vulnerability {vuln.cve_id} is patched")
                return {
                    'action_type': 'initial_access', 'target_vuln': vuln.cve_id,
                    'target_asset': str(asset.asset_id), 'target_component': str(comp.id),
                    'action_result': False, 'reason': 'patched'
                }

            if vuln.is_exploited:
                logger.debug(f"Vulnerability {vuln.cve_id} already exploited")
                return {
                    'action_type': 'initial_access', 'target_vuln': vuln.cve_id,
                    'target_asset': str(asset.asset_id), 'target_component': str(comp.id),
                    'action_result': False, 'reason': 'already_exploited'
                }

            if self._should_skip_due_to_failures(action.get('target_vuln'), action.get('target_asset'),
                                                 action.get('target_component')):
                logger.debug(f"Skipping {vuln_key} due to excessive failures")
                return {
                    'action_type': 'initial_access', 'target_vuln': vuln.cve_id,
                    'target_asset': str(asset.asset_id), 'target_component': str(comp.id),
                    'action_result': False, 'reason': 'excessive_failures'
                }

            exploit_cost = self._cost_cache['exploit_costs'].get(vuln_key, 50 + getattr(vuln, 'cvss', 5.0) * 20)
            if exploit_cost > self._remaining_attacker_budget * 1.5:
                logger.debug(
                    f"Insufficient budget: cost={exploit_cost}, budget={self._remaining_attacker_budget * 1.5}")
                return {
                    'action_type': 'initial_access', 'target_vuln': vuln.cve_id,
                    'target_asset': str(asset.asset_id), 'target_component': str(comp.id),
                    'action_result': False, 'reason': 'insufficient_budget',
                    'required_cost': exploit_cost, 'available_budget': self._remaining_attacker_budget
                }

            final_probability = max(0.5, min(0.95, action.get('probability')))
            is_successful = random.random() < final_probability

            action_details = {
                'action_type': 'initial_access', 'target_vuln': vuln.cve_id,
                'target_asset': str(asset.asset_id), 'target_component': str(comp.id),
                'cost': exploit_cost, 'probability': final_probability,
                'action_result': is_successful, 'vuln_key': vuln_key,
                'tactic': action.get('tactic', 'Initial Access')
            }

            if is_successful:
                logger.info(f"Attempting initial access via {vuln.cve_id} on {asset.name}")
                if vuln.mark_as_exploited():  # Checks is_patched internally
                    logger.info(f"Successfully gained initial access via {vuln.cve_id} on {asset.name}")
                    asset.mark_as_compromised(True)
                    asset.record_compromise(self.current_step)
                    self._remaining_attacker_budget -= exploit_cost
                    self.attacker.current_compromised_node = str(asset.asset_id)
                    self.attacker.compromised_nodes.add(str(asset.asset_id))
                    self.attacker.strategic_manager.compromised_assets.add(str(asset.asset_id))
                    self.attacker.strategic_manager.exploited_vulnerabilities.add(vuln_key)
                    self._exploited_cves.add(vuln.cve_id)
                    # Update vuln_lookup to ensure consistency
                    self.vuln_lookup[vuln_key] = (vuln, asset, comp)
                    if str(asset.asset_id) == '8' and not self.results['rtu_compromised_step']:
                        self.results['rtu_compromised_step'] = self.current_step
                    mitre_techniques = getattr(vuln, 'mitre_techniques', [])
                    if any(t in ['T1566.001', 'T1566.002'] for t in mitre_techniques):
                        self.results['spearphishing_attempts'].append({
                            'step': self.current_step, 'vuln_id': vuln.cve_id, 'asset_id': str(asset.asset_id),
                            'success': True
                        })
                    if 'T1003' in mitre_techniques:
                        self.results['credential_harvesting_successes'].append({
                            'step': self.current_step, 'vuln_id': vuln.cve_id, 'asset_id': str(asset.asset_id)
                        })
                else:
                    logger.warning(f"Failed initial access via {vuln.cve_id}: vulnerability is patched")
                    action_details['action_result'] = False
                    action_details['reason'] = 'patched'
            else:
                logger.info(f"Failed initial access attempt on {vuln.cve_id} on {asset.name}")
                attempt_cost = exploit_cost * 0.3
                self._remaining_attacker_budget -= attempt_cost
                action_details['attempt_cost'] = attempt_cost
                if any(t in ['T1566.001', 'T1566.002'] for t in getattr(vuln, 'mitre_techniques', [])):
                    self.results['spearphishing_attempts'].append({
                        'step': self.current_step, 'vuln_id': vuln.cve_id, 'asset_id': str(asset.asset_id),
                        'success': False
                    })

            logger.debug(f"Returning initial access action details: {action_details}")
            return action_details

        except Exception as e:
            logger.error(f"Error in _execute_initial_access_action: {e}", exc_info=True)
            return {
                'action_type': 'initial_access',
                'target_vuln': action.get('target_vuln', 'unknown'),
                'target_asset': action.get('target_asset', 'unknown'),
                'target_component': str(action.get('target_component', '0')),
                'action_result': False,
                'reason': f'error_{str(e)}'
            }

    def _execute_exploit_action(self, state, action):
        try:
            logger.debug(f"Executing exploit action: {action}")
            vuln_key = create_vuln_key(action.get('target_vuln', ''), str(action.get('target_asset', '')),
                                       str(action.get('target_component', '0')))
            if self._should_skip_due_to_failures(
                    action.get('target_vuln', ''),
                    action.get('target_asset', ''),
                    action.get('target_component', '0')
            ):
                logger.debug(f"Skipping exploit due to excessive failures for {vuln_key}")
                return {
                    'action_type': 'exploitation',
                    'target_vuln': action.get('target_vuln', ''),
                    'target_asset': str(action.get('target_asset', '')),
                    'target_component': str(action.get('target_component', '0')),
                    'action_result': False,
                    'reason': 'excessive_failures'
                }

            if vuln_key not in self.vuln_lookup:
                logger.warning(f"Invalid vulnerability: {vuln_key}")
                return {'action_result': False, 'reason': 'invalid_target', 'action_type': 'exploitation'}

            vuln, asset, comp = self.vuln_lookup[vuln_key]
            logger.debug(f"Processing vulnerability {vuln.cve_id} on asset {asset.asset_id}, component {comp.id}")

            if vuln.is_patched:
                logger.debug(f"Vulnerability {vuln.cve_id} is patched")
                return {
                    'action_type': 'exploitation', 'target_vuln': vuln.cve_id,
                    'target_asset': str(asset.asset_id), 'target_component': str(comp.id),
                    'action_result': False, 'reason': 'patched'
                }

            if vuln.is_exploited:
                logger.debug(f"Vulnerability {vuln.cve_id} already exploited")
                return {
                    'action_type': 'exploitation', 'target_vuln': vuln.cve_id,
                    'target_asset': str(asset.asset_id), 'target_component': str(comp.id),
                    'action_result': False, 'reason': 'already_exploited'
                }

            exploit_cost = self._cost_cache['exploit_costs'].get(vuln_key, 50 + getattr(vuln, 'cvss', 5.0) * 20)
            if exploit_cost > self._remaining_attacker_budget:
                logger.debug(
                    f"Insufficient budget for exploit: cost={exploit_cost}, budget={self._remaining_attacker_budget}")
                return {
                    'action_type': 'exploitation', 'target_vuln': vuln.cve_id,
                    'target_asset': str(asset.asset_id), 'target_component': str(comp.id),
                    'action_result': False, 'reason': 'insufficient_budget'
                }

            success_probability = min(0.9, action.get('probability', 0.5) * 1.5)
            is_successful = random.random() < success_probability

            action_details = {
                'action_type': 'exploitation', 'target_vuln': vuln.cve_id,
                'target_asset': str(asset.asset_id), 'target_component': str(comp.id),
                'cost': exploit_cost, 'probability': success_probability,
                'action_result': is_successful, 'vuln_key': vuln_key,
                'tactic': action.get('tactic', 'Exploitation')
            }

            if is_successful:
                logger.info(f"Attempting to exploit {vuln_key}")
                if vuln.mark_as_exploited():  # Checks is_patched internally
                    logger.info(f"Successfully exploited {vuln_key}")
                    asset.mark_as_compromised(True)
                    asset.record_compromise(self.current_step)
                    self._remaining_attacker_budget -= exploit_cost
                    self.attacker.current_compromised_node = str(asset.asset_id)
                    self.attacker.compromised_nodes.add(str(asset.asset_id))
                    self.attacker.strategic_manager.compromised_assets.add(str(asset.asset_id))
                    self.attacker.strategic_manager.exploited_vulnerabilities.add(vuln_key)
                    self._exploited_cves.add(vuln.cve_id)
                    # Update vuln_lookup to ensure consistency
                    self.vuln_lookup[vuln_key] = (vuln, asset, comp)
                    if str(asset.asset_id) == '8' and not self.results['rtu_compromised_step']:
                        self.results['rtu_compromised_step'] = self.current_step
                    mitre_techniques = getattr(vuln, 'mitre_techniques', [])
                    if 'T1003' in mitre_techniques:
                        self.results['credential_harvesting_successes'].append({
                            'step': self.current_step, 'vuln_id': vuln.cve_id, 'asset_id': str(asset.asset_id)
                        })
                    if any(t in ['T1566.001', 'T1566.002'] for t in mitre_techniques):
                        self.results['spearphishing_attempts'].append({
                            'step': self.current_step, 'vuln_id': vuln.cve_id, 'asset_id': str(asset.asset_id),
                            'success': True
                        })
                else:
                    logger.warning(f"Failed to exploit {vuln_key}: vulnerability is patched")
                    action_details['action_result'] = False
                    action_details['reason'] = 'patched'
            else:
                logger.info(f"Failed to exploit {vuln_key}")
                attempt_cost = exploit_cost * 0.3
                self._remaining_attacker_budget -= attempt_cost
                action_details['attempt_cost'] = attempt_cost
                if any(t in ['T1566.001', 'T1566.002'] for t in getattr(vuln, 'mitre_techniques', [])):
                    self.results['spearphishing_attempts'].append({
                        'step': self.current_step, 'vuln_id': vuln.cve_id, 'asset_id': str(asset.asset_id),
                        'success': False
                    })

            logger.debug(f"Returning exploit action details: {action_details}")
            return action_details

        except Exception as e:
            logger.error(f"Error in _execute_exploit_action for {vuln_key}: {e}", exc_info=True)
            return {
                'action_type': 'exploitation',
                'target_vuln': action.get('target_vuln', 'unknown'),
                'target_asset': str(action.get('target_asset', 'unknown')),
                'target_component': str(action.get('target_component', '0')),
                'action_result': False,
                'reason': f'error_{str(e)}'
            }

    def _execute_movement_action(self, state, action):
        try:
            asset_id = action.get('target_asset')
            action_type = action.get('action_type')

            logger.info(f"Executing movement action: {action_type} to asset {asset_id}")

            if not asset_id:
                logger.error(f"Missing target_asset in movement action")
                return {'action_result': False, 'reason': 'invalid_asset', 'action_type': action_type}

            asset = next((a for a in state.system.assets if str(a.asset_id) == str(asset_id)), None)
            if not asset:
                logger.error(f"Target asset {asset_id} not found")
                return {'action_result': False, 'reason': 'asset_not_found', 'action_type': action_type}

            if action_type == 'initial_access':
                logger.info("initial_access action routed to movement handler - redirecting to initial access handler")
                return self._execute_initial_access_action(state, action)

            if asset.is_compromised:
                logger.warning(f"Asset {asset_id} is already compromised")
                return {
                    'action_type': action_type,
                    'target_asset': str(asset_id),
                    'action_result': False,
                    'reason': 'already_compromised'
                }

            return self._execute_lateral_movement(state, action, asset)

        except Exception as e:
            logger.error(f"Error in _execute_movement_action: {e}", exc_info=True)
            return {
                'action_type': action.get('action_type', 'lateral_movement'),
                'target_asset': str(action.get('target_asset', 'unknown')),
                'action_result': False,
                'reason': f'error_{str(e)}'
            }

    def _execute_lateral_movement(self, state, action, asset):
        try:
            current_position = self._determine_current_position(state)
            target_asset_id = str(asset.asset_id)
            logger.debug(f"Attempting lateral movement from {current_position} to {target_asset_id}")

            if current_position == 'internet':
                logger.info(f"Failed lateral movement to {target_asset_id}: no initial foothold")
                return {
                    'action_type': 'lateral_movement', 'target_asset': target_asset_id,
                    'action_result': False, 'reason': 'no_initial_foothold'
                }

            if not self._are_assets_connected(current_position, target_asset_id):
                logger.info(f"Failed lateral movement to {target_asset_id}: not network connected")
                return {
                    'action_type': 'lateral_movement', 'target_asset': target_asset_id,
                    'action_result': False, 'reason': 'not_network_connected'
                }

            action_vuln_id = action.get('target_vuln')
            action_comp_id = action.get('target_component', '0')
            vuln_info = None

            # Prefer the vulnerability specified in the action if it exists
            if action_vuln_id:
                for comp in asset.components:
                    if str(comp.id) == str(action_comp_id):
                        for vuln in comp.vulnerabilities:
                            if vuln.cve_id == action_vuln_id:
                                vuln_info = {'cve_id': vuln.cve_id, 'component_id': comp.id}
                                break
                    if vuln_info:
                        break

            # Fallback to finding an exploitable vulnerability if action's vuln is invalid
            if not vuln_info:
                vuln_info = self._find_exploitable_vulnerability(asset)
                if not vuln_info:
                    logger.info(f"Failed lateral movement to {target_asset_id}: no exploitable vulnerabilities")
                    return {
                        'action_type': 'lateral_movement', 'target_asset': target_asset_id,
                        'action_result': False, 'reason': 'no_vulnerabilities'
                    }

            if not isinstance(vuln_info, dict):
                logger.error(f"Invalid vuln_info type: {type(vuln_info)}, expected dict, received: {vuln_info}")
                return {
                    'action_type': 'lateral_movement', 'target_asset': target_asset_id,
                    'action_result': False, 'reason': f'invalid_vuln_info_type_{type(vuln_info)}'
                }

            cve_id = vuln_info.get('cve_id')
            comp_id = vuln_info.get('component_id')
            if not cve_id or not comp_id:
                logger.error(f"Missing cve_id or component_id in vuln_info: {vuln_info}")
                return {
                    'action_type': 'lateral_movement', 'target_asset': target_asset_id,
                    'action_result': False, 'reason': 'missing_vuln_details'
                }

            vuln_key = create_vuln_key(cve_id, target_asset_id, str(comp_id))
            if vuln_key not in self.vuln_lookup:
                logger.error(f"Invalid vulnerability key: {vuln_key}")
                return {
                    'action_type': 'lateral_movement', 'target_asset': target_asset_id,
                    'action_result': False, 'reason': 'invalid_vuln_key'
                }

            vuln, asset, comp = self.vuln_lookup[vuln_key]
            logger.info(f"Selected vulnerability {cve_id} on asset {target_asset_id} for lateral movement")

            if action_vuln_id and action_vuln_id != cve_id:
                logger.warning(f"Vulnerability mismatch corrected: action specified {action_vuln_id}, using {cve_id}")

            if self._should_skip_due_to_failures(cve_id, target_asset_id, comp_id):
                logger.info(f"Skipping lateral movement to {target_asset_id} via {vuln_key} due to excessive failures")
                return {
                    'action_type': 'lateral_movement', 'target_asset': target_asset_id,
                    'action_result': False, 'reason': 'excessive_failures'
                }

            if vuln.is_patched:
                logger.debug(f"Vulnerability {vuln.cve_id} is patched")
                return {
                    'action_type': 'lateral_movement', 'target_vuln': vuln.cve_id,
                    'target_asset': str(asset.asset_id), 'target_component': str(comp.id),
                    'action_result': False, 'reason': 'patched'
                }

            if vuln.is_exploited:
                logger.debug(f"Vulnerability {vuln.cve_id} already exploited")
                return {
                    'action_type': 'lateral_movement', 'target_vuln': vuln.cve_id,
                    'target_asset': str(asset.asset_id), 'target_component': str(comp.id),
                    'action_result': False, 'reason': 'already_exploited'
                }

            base_probability = min(0.9, vuln.epss * 1.5 if hasattr(vuln, 'epss') else 0.6)
            if vuln.exploit:
                base_probability *= 1.2
            if 'AV:N' in getattr(vuln, 'cvssV3Vector', ''):
                base_probability *= 1.1
            base_probability *= self.attacker_sophistication
            previous_attempts = self._lateral_movement_attempts.get(vuln_key, 0)
            if previous_attempts > 0:
                base_probability *= (0.7 ** previous_attempts)
            final_probability = max(0.1, min(0.9, base_probability))

            cost = self._cost_cache['exploit_costs'].get(vuln_key, 50 + getattr(vuln, 'cvss', 5.0) * 20)
            if cost > self._remaining_attacker_budget:
                logger.info(
                    f"Failed lateral movement to {target_asset_id}: insufficient budget (cost={cost}, budget={self._remaining_attacker_budget})")
                return {
                    'action_type': 'lateral_movement', 'target_asset': target_asset_id,
                    'action_result': False, 'reason': 'insufficient_budget'
                }

            self._lateral_movement_attempts[vuln_key] = previous_attempts + 1
            is_successful = random.random() < final_probability

            action_details = {
                'action_type': 'lateral_movement', 'target_asset': target_asset_id,
                'from_asset': current_position, 'cost': cost, 'probability': final_probability,
                'action_result': is_successful, 'attempt_number': self._lateral_movement_attempts[vuln_key],
                'tactic': action.get('tactic', 'Lateral Movement'), 'target_vuln': cve_id,
                'target_component': str(comp_id), 'vuln_key': vuln_key
            }

            if is_successful:
                logger.info(f"Successfully moved laterally from {current_position} to {target_asset_id} via {cve_id}")
                vuln.is_exploited = True
                self._exploited_cves.add(cve_id)
                asset.mark_as_compromised(True)
                asset.record_compromise(self.current_step)
                self._remaining_attacker_budget -= cost
                self.attacker.current_compromised_node = target_asset_id
                self.attacker.compromised_nodes.add(target_asset_id)
                self.attacker.strategic_manager.compromised_assets.add(target_asset_id)
                self.attacker.strategic_manager.exploited_vulnerabilities.add(vuln_key)
                if not hasattr(self.state, 'lateral_movement_targets'):
                    self.state.lateral_movement_targets = []
                if target_asset_id not in self.state.lateral_movement_targets:
                    self.state.lateral_movement_targets.append(target_asset_id)
                if not hasattr(self.state, 'lateral_movement_chain'):
                    self.state.lateral_movement_chain = []
                self.state.lateral_movement_chain.append((target_asset_id, current_position))
                for sys_asset in self.state.system.assets:
                    if str(sys_asset.asset_id) == str(asset.asset_id):
                        sys_asset.mark_as_compromised(True)
                        sys_asset.record_compromise(self.current_step)
                        logger.debug(f"System asset {sys_asset.asset_id} marked as compromised")
                # Update vuln_lookup to ensure consistency
                self.vuln_lookup[vuln_key] = (vuln, asset, comp)
            else:
                logger.info(f"Failed lateral movement from {current_position} to {target_asset_id} via {cve_id}")
                self._remaining_attacker_budget -= cost * 0.3
                action_details['attempt_cost'] = cost * 0.3

            logger.debug(f"Returning lateral movement action details: {action_details}")
            return action_details

        except Exception as e:
            logger.error(f"Error in _execute_lateral_movement to {target_asset_id}: {e}", exc_info=True)
            return {
                'action_type': 'lateral_movement',
                'target_asset': str(asset.asset_id),
                'action_result': False,
                'reason': f'error_{str(e)}'
            }

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
                            f"Skipping {create_vuln_key(vuln_id, str(asset_id), str(comp_id))} due to excessive failures")
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
    def _create_attack_observation(self, action: Dict, action_result: Dict, step: int) -> Optional[Dict]:
        """
        Enhanced attack observation creation with better validation.
        """
        if not action or not action_result:
            return None

        # Only certain action types are observable by threat intelligence systems
        observable_actions = {
            'initial_access', 'exploitation', 'lateral_movement', 'privilege_escalation',
            'persistence', 'command_and_control', 'exfiltration', 'reconnaissance'
        }

        action_type = action_result.get('action_type', action.get('action_type'))
        if action_type not in observable_actions:
            return None

        # Create the observation
        observation = {
            'step': step,
            'action_type': action_type,
            'target_asset': str(action_result.get('target_asset', action.get('target_asset', ''))),
            'success': action_result.get('action_result', False),
            'timestamp': step,
            'reason': action_result.get('reason', 'unknown')  # Include failure reason
        }

        # Add vulnerability information if available
        target_vuln = action_result.get('target_vuln', action.get('target_vuln'))
        if target_vuln:
            observation['target_vuln'] = target_vuln

        # Add MITRE techniques if available (simulating advanced threat detection)
        vuln_key = action_result.get('vuln_key')
        if vuln_key and vuln_key in self.vuln_lookup:
            vuln, _, _ = self.vuln_lookup[vuln_key]
            techniques = getattr(vuln, 'mitre_techniques', [])
            if techniques:
                observation['techniques'] = techniques

        # Add default APT3 techniques based on action type (simulating threat intelligence correlation)
        if 'techniques' not in observation:
            apt3_technique_mapping = {
                'initial_access': ['T1190', 'T1566'],
                'exploitation': ['T1068', 'T1203'],
                'lateral_movement': ['T1021', 'T1563'],
                'privilege_escalation': ['T1068', 'T1548'],
                'persistence': ['T1053', 'T1543'],
                'command_and_control': ['T1071', 'T1095'],
                'exfiltration': ['T1041', 'T1048'],
                'reconnaissance': ['T1595', 'T1590']
            }
            observation['techniques'] = apt3_technique_mapping.get(action_type, ['T1001'])

        # Add detection confidence (simulating IDS confidence levels)
        if action_result.get('action_result', False):
            # Successful attacks are easier to detect with high confidence
            observation['detection_confidence'] = 0.9
        else:
            # Failed attacks might be detected with lower confidence
            # Higher confidence for "patched" failures (defender is doing well)
            if action_result.get('reason') == 'patched':
                observation['detection_confidence'] = 0.95
            else:
                observation['detection_confidence'] = 0.6

        # Add contextual information for better threat intelligence
        observation['tactic'] = action.get('tactic', action_result.get('tactic', 'Unknown'))

        return observation
    def _execute_other_action(self, state, action):
        try:
            action_type = action.get('action_type')
            if action_type == 'reconnaissance':
                cost = 1.0
                if cost > self._remaining_attacker_budget:
                    logger.debug(
                        f"Insufficient budget for {action_type}: cost={cost}, budget={self._remaining_attacker_budget}")
                    return {
                        'action_type': action_type,
                        'action_result': False,
                        'reason': 'insufficient_budget',
                        'tactic': action.get('tactic', 'Reconnaissance'),
                        'is_recon': True
                    }

                success_probability = action.get('probability', 0.7)
                is_successful = random.random() < success_probability
                self._remaining_attacker_budget -= cost

                action_details = {
                    'action_type': action_type,
                    'action_result': is_successful,
                    'cost': cost,
                    'probability': success_probability,
                    'tactic': action.get('tactic', 'Reconnaissance'),
                    'is_recon': True
                }

                recon_count = sum(1 for a in getattr(self.state.system, 'action_history', [])
                                  if a.get('action_type') == 'reconnaissance' and a.get('action_result', False))
                if recon_count >= 5 and is_successful:
                    logger.info("Limiting reconnaissance, suggesting kill chain progression")
                    self.state.suggest_attacker_stage(KillChainStage.DELIVERY.value)

                logger.debug(f"Returning reconnaissance action details: {action_details}")
                return action_details

            cost = 10.0
            if cost > self._remaining_attacker_budget:
                logger.debug(
                    f"Insufficient budget for {action_type}: cost={cost}, budget={self._remaining_attacker_budget}")
                return {
                    'action_type': action_type,
                    'action_result': False,
                    'reason': 'insufficient_budget',
                    'tactic': action.get('tactic', 'Unknown'),
                    'is_recon': False
                }

            success_probability = action.get('probability', 0.7)
            is_successful = random.random() < success_probability
            self._remaining_attacker_budget -= cost

            action_details = {
                'action_type': action_type,
                'action_result': is_successful,
                'cost': cost,
                'probability': success_probability,
                'tactic': action.get('tactic', 'Unknown'),
                'is_recon': False
            }

            logger.debug(f"Returning other action details: {action_details}")
            return action_details

        except Exception as e:
            logger.error(f"Error in _execute_other_action: {e}", exc_info=True)
            return {
                'action_type': action.get('action_type', 'unknown'),
                'action_result': False,
                'reason': f'error_{str(e)}',
                'tactic': action.get('tactic', 'Unknown'),
                'is_recon': action.get('action_type') == 'reconnaissance'
            }

    def _find_exploitable_vulnerability(self, asset):
        vuln_candidates = []
        for comp in asset.components:
            for vuln in comp.vulnerabilities:
                if vuln.is_patched or getattr(vuln, 'is_exploited', False):
                    continue
                epss = getattr(vuln, 'epss', 0.0)
                has_exploit = getattr(vuln, 'exploit', False)
                cvss = getattr(vuln, 'cvss', 0.0)
                priority_score = epss
                if has_exploit:
                    priority_score *= 1.5
                if cvss >= 7.0:
                    priority_score *= 1.2
                cvss_vector = getattr(vuln, 'cvssV3Vector', '')
                is_network_accessible = 'AV:N' in cvss_vector
                priority_cves = {'CVE-2018-13379', 'ZERO-DAY-001', 'CVE-2015-3113'}
                is_priority_cve = vuln.cve_id in priority_cves
                if is_priority_cve:
                    priority_score *= 2.0
                vuln_candidate = {
                    'cve_id': vuln.cve_id,
                    'component_id': str(comp.id),
                    'priority_score': priority_score,
                    'is_network_accessible': is_network_accessible,
                    'is_priority_cve': is_priority_cve
                }
                vuln_candidates.append(vuln_candidate)
                logger.debug(f"Added vuln candidate: {vuln_candidate}")
        if not vuln_candidates:
            logger.debug(f"No valid vulnerabilities found for asset {asset.asset_id}")
            return None
        vuln_candidates.sort(key=lambda x: (
            x['is_network_accessible'],
            x['is_priority_cve'],
            x['priority_score']
        ), reverse=True)
        best_vuln = vuln_candidates[0]
        logger.info(f"Selected vulnerability {best_vuln['cve_id']} on asset {asset.asset_id}")
        return best_vuln

    def _determine_current_position(self, state):
        if hasattr(self.attacker, 'current_compromised_node') and self.attacker.current_compromised_node and \
           self.attacker.current_compromised_node != 'internet':
            if ':' in str(self.attacker.current_compromised_node):
                parts = str(self.attacker.current_compromised_node).split(':')
                if len(parts) >= 2:
                    return parts[1]
            return str(self.attacker.current_compromised_node)
        compromised_assets = [str(asset.asset_id) for asset in state.system.assets if asset.is_compromised]
        if compromised_assets:
            asset_values = [(asset_id, self._get_asset_by_id(asset_id).business_value) for asset_id in compromised_assets]
            return max(asset_values, key=lambda x: x[1])[0]
        return 'internet'

    def _get_asset_by_id(self, asset_id):
        asset_id_str = str(asset_id)
        for asset in self.system.assets:
            if str(asset.asset_id) == asset_id_str:
                return asset
        logger.warning(f"No asset found with ID: {asset_id_str}")
        return None

    def _are_assets_connected(self, asset1_id, asset2_id):
        asset1_str = str(asset1_id)
        asset2_str = str(asset2_id)
        for conn in self.system.connections:
            from_id = str(conn.from_asset.asset_id) if hasattr(conn, 'from_asset') else None
            to_id = str(conn.to_asset.asset_id) if hasattr(conn, 'to_asset') else None
            if from_id and to_id:
                if (from_id == asset1_str and to_id == asset2_str) or (from_id == asset2_str and to_id == asset1_str):
                    return True
        return False

    def calculate_metrics(self, state, total_patch_cost, total_patches, verbose=False):
        protected_value = 0.0
        lost_value = 0.0
        total_business_value = sum(asset.business_value for asset in state.system.assets)
        unpatched_critical = 0
        patched_vulns = set()
        compromised_assets = 0
        processed_assets = set()
        spearphishing_success_rate = 0.0
        credential_harvesting_count = len(self.results['credential_harvesting_successes'])

        if self.results['spearphishing_attempts']:
            spearphishing_success_rate = sum(1 for a in self.results['spearphishing_attempts'] if a['success']) / len(
                self.results['spearphishing_attempts'])

        for asset in state.system.assets:
            asset_id = str(asset.asset_id)
            if asset_id in processed_assets:
                continue
            processed_assets.add(asset_id)
            business_value = asset.business_value
            if asset.is_compromised:
                compromised_assets += 1
                lost_value += business_value
                if verbose:
                    print(f"Asset {asset_id} ({asset.name}) compromised - Lost value: ${business_value:,.2f}")
            else:
                protected_value += business_value
                if verbose:
                    print(f"Asset {asset_id} ({asset.name}) protected - Protected value: ${business_value:,.2f}")
            for comp in asset.components:
                for vuln in comp.vulnerabilities:
                    vuln_key = create_vuln_key(vuln.cve_id, str(asset.asset_id), str(comp.id))
                    if not vuln.is_patched and getattr(vuln, 'cvss', 5.0) >= 7.0:
                        unpatched_critical += 1
                    elif vuln.is_patched:
                        patched_vulns.add(vuln_key)

        if lost_value > total_business_value:
            lost_value = total_business_value
            protected_value = 0.0

        value_preserved = max(0.0, total_business_value - total_patch_cost - lost_value)
        roi = value_preserved / total_patch_cost * 100.0 if total_patch_cost > 0 else 0.0
        detection_coverage = len(self.mitre_techniques_detected) / len(
            self.mitre_techniques_used) if self.mitre_techniques_used else 0.0
        avg_time_to_detection = sum(self.time_to_detection.values()) / len(
            self.time_to_detection) if self.time_to_detection else 0.0
        attack_disruption_rate = 1.0 if self.attack_disrupted else 0.0

        action_history = getattr(state.system, 'action_history', self.results['exploit_attempts'])
        reconnaissance_attempts = len([e for e in action_history
                                       if e.get('action_type') == 'reconnaissance'])
        successful_reconnaissance = len([e for e in action_history
                                         if e.get('action_type') == 'reconnaissance' and e.get('action_result', False)])
        reconnaissance_success_rate = successful_reconnaissance / reconnaissance_attempts if reconnaissance_attempts > 0 else 0.0

        successful_exploits = len([e for e in action_history
                                   if e.get('action_result', False) and not e.get('is_recon', False)])
        total_exploit_attempts = len([e for e in action_history
                                      if not e.get('is_recon', False) and e.get('action_type') != 'none'])
        attack_success_rate = successful_exploits / total_exploit_attempts if total_exploit_attempts > 0 else 0.0

        if verbose:
            print(f"\n=== Metrics Summary ===")
            print(f"Total Business Value: ${total_business_value:,.2f}")
            print(f"Protected Value: ${protected_value:,.2f}")
            print(f"Lost Value: ${lost_value:,.2f}")
            print(f"Value Preserved: ${value_preserved:,.2f}")
            print(f"Unpatched Critical Vulnerabilities: {unpatched_critical}")
            print(f"Total Patched Vulnerabilities: {len(patched_vulns)}")
            print(f"Compromised Assets: {compromised_assets}")
            print(f"ROI: {roi:.1f}%")
            print(f"Detection Coverage: {detection_coverage * 100:.1f}%")
            print(f"Reconnaissance Attempts: {reconnaissance_attempts}")
            print(f"Successful Reconnaissance: {successful_reconnaissance}")
            print(f"Reconnaissance Success Rate: {reconnaissance_success_rate * 100:.1f}%")
            print(f"Exploit Attempts: {total_exploit_attempts}")
            print(f"Successful Exploits: {successful_exploits}")
            print(f"Attack Success Rate: {attack_success_rate * 100:.1f}%")
            print(f"Spearphishing Success Rate: {spearphishing_success_rate * 100:.1f}%")
            print(f"Credential Harvesting Successes: {credential_harvesting_count}")

        return {
            'protected_value': protected_value,
            'lost_value': lost_value,
            'value_preserved': value_preserved,
            'unpatched_critical': unpatched_critical,
            'total_patch_cost': total_patch_cost,
            'total_patches': total_patches,
            'roi': roi,
            'compromised_assets_count': compromised_assets,
            'detection_coverage': detection_coverage,
            'avg_time_to_detection': avg_time_to_detection,
            'attack_disruption_rate': attack_disruption_rate,
            'spearphishing_success_rate': spearphishing_success_rate,
            'credential_harvesting_count': credential_harvesting_count,
            'reconnaissance_attempts': reconnaissance_attempts,
            'successful_reconnaissance': successful_reconnaissance,
            'reconnaissance_success_rate': reconnaissance_success_rate,
            'attack_success_rate': attack_success_rate
        }

    def calculate_detection_metrics(self, results):
        detection_coverage = len(self.mitre_techniques_detected) / len(self.mitre_techniques_used) if self.mitre_techniques_used else 0.0
        avg_time_to_detection = sum(self.time_to_detection.values()) / len(self.time_to_detection) if self.time_to_detection else 0.0
        attack_disruption_rate = 1.0 if self.attack_disrupted else 0.0
        return {
            'detection_coverage': detection_coverage * 100,
            'avg_time_to_detection': avg_time_to_detection,
            'attack_disruption_rate': attack_disruption_rate * 100
        }

    def calculate_time_to_rtu_compromise(self, results):
        return results.get('rtu_compromised_step')

    def should_terminate_simulation(self, step: int, max_steps: int, total_patch_cost: float) -> tuple[bool, str]:
        """
        Determine if simulation should terminate based on realistic criteria.
        
        Returns:
            tuple: (should_terminate, reason)
        """
        # (i) Defender's budget is depleted
        if self._remaining_defender_budget <= 0:
            return True, "defender_budget_depleted"
        
        # (ii) All vulnerabilities are patched
        unpatched_vulns = 0
        for asset in self.state.system.assets:
            for comp in asset.components:
                for vuln in comp.vulnerabilities:
                    if not vuln.is_patched:
                        unpatched_vulns += 1
        if unpatched_vulns == 0:
            return True, "all_vulnerabilities_patched"
        
        # (iii) Campaign horizon elapses
        if step >= max_steps - 1:
            return True, "campaign_horizon_elapsed"
        
        # (iv) Economic break-even: cumulative expenditure exceeds residual business value
        total_business_value = sum(
            getattr(asset, 'business_value', getattr(asset, 'criticality_level', 3) * 5000)
            for asset in self.state.system.assets if not asset.is_compromised
        )
        if total_patch_cost >= total_business_value:
            return True, "economic_break_even"
        
        return False, "continue"

    def reset_exploit_status(self):
        logger.info("Resetting exploit and compromise status")
        for asset in self.state.system.assets:
            asset.mark_as_compromised(False)
            asset._compromise_time = None
            asset._last_attack_time = None
            for comp in asset.components:
                for vuln in comp.vulnerabilities:
                    vuln.is_exploited = False
                    vuln.is_patched = False
        self._patched_cves = set()
        self._exploited_cves = set()
        self._lateral_movement_attempts = {}
        self._exploit_failures = {}  # Explicitly reset simulation failure counts

        # Reset attacker state completely
        self.attacker.strategic_manager.compromised_assets.clear()
        self.attacker.strategic_manager.exploited_vulnerabilities.clear()
        self.attacker.strategic_manager.exploit_failures.clear()  # Reset strategic manager failure counts
        self.attacker.current_compromised_node = 'internet'
        self.attacker.compromised_nodes.clear()
        self.attacker.exploited_vulns.clear()

        # Reset any failure tracking in the attacker itself
        if hasattr(self.attacker, 'exploit_failures'):
            self.attacker.exploit_failures.clear()

        if hasattr(self.attacker, 'reset_state'):
            self.attacker.reset_state()

        logger.debug("Exploit and compromise status reset complete")

    def compare_strategies(self, defender_budget: int, num_steps: int, num_trials: int = 100, verbose: bool = False):
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
                self.state = State(k=KillChainStage.RECONNAISSANCE.value, system=self.system)
                self.vuln_lookup = {}
                for asset in self.system.assets:
                    for comp in asset.components:
                        for vuln in comp.vulnerabilities:
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

        # Track attack path usage
        path_usage = {strategy_name: {} for strategy_name in strategies}
        rtu_compromise_counts = {strategy_name: {} for strategy_name in strategies}
        for strategy_name in strategies:
            for trial in aggregated_results[strategy_name]['trials']:
                for path in trial['attack_paths_used']:
                    path_key = tuple(
                        (step['target_vuln'] or 'None', step['target_asset'], step['tactic'], step['success']) for step
                        in path['path_steps'])
                    path_usage[strategy_name][path_key] = path_usage[strategy_name].get(path_key, 0) + 1
                    if path['rtu_compromised']:
                        rtu_compromise_counts[strategy_name][path_key] = rtu_compromise_counts[strategy_name].get(
                            path_key, 0) + 1

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

        # Threat Intelligence specific analysis
        threat_intel_results = aggregated_results.get('Threat Intelligence', {})
        if threat_intel_results['mean_metrics']['observations_collected'] > 0:
            print(f"\n=== Threat Intelligence Learning Analysis ===")
            ti_metrics = threat_intel_results['mean_metrics']
            print(f"Successfully observed {ti_metrics['observations_collected']:.1f} attack behaviors on average")
            print(f"Made {ti_metrics['learning_adaptations']:.1f} learning adaptations during simulation")
            print(f"Detected threat level changes on {ti_metrics['threat_level_changes']:.1f} assets")
            print(f"Learning enabled more effective defense prioritization")

        # Visualization
        metrics_to_plot = {
            "Protected Value": ('protected_value_runs', 'Protected Value ($)', 'protected_value'),
            "Lost Value": ('lost_value_runs', 'Lost Value ($)', 'lost_value'),
            "Value Preserved": ('value_preserved_runs', 'Value Preserved ($)', 'value_preserved'),
            "ROI": ('roi_runs', 'Return on Investment (%)', 'roi'),
            "Compromised Assets": ('compromised_assets_runs', 'Compromised Assets', 'compromised_assets'),
            "Detection Coverage": ('detection_coverage_runs', 'Detection Coverage (%)', 'detection_coverage'),
            "Avg Time to Detection": (
                'avg_time_to_detection_runs', 'Average Time to Detection (steps)', 'avg_time_to_detection'),
            "Attack Disruption Rate": (
                'attack_disruption_rate_runs', 'Attack Disruption Rate (%)', 'attack_disruption_rate'),
            "Spearphishing Success Rate": (
                'spearphishing_success_rate_runs', 'Spearphishing Success Rate (%)', 'spearphishing_success_rate'),
            "Credential Harvesting Count": (
                'credential_harvesting_count_runs', 'Credential Harvesting Count', 'credential_harvesting_count'),
            "Observations Collected": (
                'observations_collected_runs', 'Observations Collected', 'observations_collected')
        }
        for metric_name, (metric_key, metric_label, file_prefix) in metrics_to_plot.items():
            plt.figure(figsize=(12, 6))
            values = [aggregated_results[strategy_name]['run_statistics'][metric_key] for strategy_name in strategies]
            # Filter out empty or all-None lists for bar plot
            means = [np.mean([v for v in vals if v is not None]) if vals and any(v is not None for v in vals) else 0.0 for vals in values]
            stds = [np.std([v for v in vals if v is not None]) if vals and any(v is not None for v in vals) else 0.0 for vals in values]
            bars = plt.bar(strategies.keys(), means, yerr=stds, color='skyblue', alpha=0.7, edgecolor='navy', capsize=5)
            for bar, value in zip(bars, means):
                height = bar.get_height()
                label = f'${value:,.0f}' if 'Value' in metric_name else f'{value:.1f}%' if 'Rate' in metric_name or 'Coverage' in metric_name else f'{value:.1f}'
                plt.text(bar.get_x() + bar.get_width() / 2., height + max(means) * 0.01 if max(means) > 0 else 0.1, label, ha='center', va='bottom', fontsize=10)
            plt.xlabel("Strategy", fontsize=12)
            plt.ylabel(metric_name, fontsize=12)
            plt.title(f"Mean {metric_name} by Strategy (APT3 Enhanced, {num_trials} Trials)", fontsize=14, fontweight='bold')
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            save_path = os.path.join(viz_dir, f"apt3_enhanced_{file_prefix}_bar.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved {metric_name} bar plot to {save_path}")
            # Violin plot with data validation
            plt.figure(figsize=(14, 8))
            num_strategies = len(strategies)
            positions = np.arange(num_strategies)
            labels = list(strategies.keys())
            data = [aggregated_results[strategy_name]['run_statistics'][metric_key] for strategy_name in strategies]
            valid_data = [[v for v in d if v is not None] for d in data]
            # Only keep non-empty lists
            nonempty_indices = [i for i, d in enumerate(valid_data) if len(d) > 0]
            if nonempty_indices:
                plot_data = [valid_data[i] for i in nonempty_indices]
                plot_positions = [i for i in nonempty_indices]
                plot_labels = [labels[i] for i in nonempty_indices]
                # Debug: print data types and lengths
                print(f"[DEBUG] {metric_name} plot_data types: {[type(d) for d in plot_data]}")
                print(f"[DEBUG] {metric_name} plot_data lengths: {[len(d) for d in plot_data]}")
                # Ensure all entries are 1D lists of numbers
                all_1d = all(isinstance(d, list) and all(isinstance(x, (int, float, np.integer, np.floating)) for x in d) for d in plot_data)
                if not all_1d:
                    print(f"Skipping {metric_name} violin plot: plot_data contains non-1D or non-numeric entries.")
                else:
                    try:
                        violin_parts = plt.violinplot(plot_data, positions=plot_positions, widths=0.15, showmeans=True)
                        for pc in violin_parts['bodies']:
                            pc.set_facecolor('lightcoral')
                            pc.set_edgecolor('darkred')
                            pc.set_alpha(0.6)
                        plt.boxplot(plot_data, positions=plot_positions, widths=0.15, showfliers=True)
                        plt.xlabel('Strategy', fontsize=12)
                        plt.ylabel(metric_label, fontsize=12)
                        plt.title(f'Distribution of {metric_label} Across {num_trials} Trials per Strategy', fontsize=14)
                        plt.xticks(plot_positions, plot_labels, rotation=45, ha='right')
                        plt.grid(True, linestyle='--', alpha=0.7)
                        plt.tight_layout()
                        save_path = os.path.join(viz_dir, f"apt3_enhanced_{file_prefix}_distribution.png")
                        plt.savefig(save_path, dpi=300, bbox_inches='tight')
                        plt.close()
                        print(f"Saved {metric_name} distribution plot to {save_path}")
                    except Exception as e:
                        print(f"Skipping {metric_name} violin plot due to error: {e}")
            else:
                print(f"Skipping {metric_name} violin plot: no valid data to plot.")
        plt.figure(figsize=(12, 6))
        for strategy_name in strategies:
            unique_paths = list(path_usage[strategy_name].keys())
            counts = list(path_usage[strategy_name].values())
            plt.bar([f"{strategy_name}_{i}" for i in range(len(unique_paths))], counts, width=0.2, label=strategy_name)
        plt.xlabel("Unique Attack Paths", fontsize=12)
        plt.ylabel("Usage Count", fontsize=12)
        plt.title("Attack Path Usage by Strategy (APT3 Enhanced, 100 Trials)", fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        save_path = os.path.join(viz_dir, "apt3_enhanced_path_usage.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved path usage plot to {save_path}")
        plt.figure(figsize=(12, 6))
        for strategy_name in strategies:
            unique_paths = list(rtu_compromise_counts[strategy_name].keys())
            counts = list(rtu_compromise_counts[strategy_name].values())
            plt.bar([f"{strategy_name}_{i}" for i in range(len(unique_paths))], counts, width=0.2, label=strategy_name)
        plt.xlabel("Unique Attack Paths", fontsize=12)
        plt.ylabel("RTU Compromise Count", fontsize=12)
        plt.title("RTU Compromise Rate by Attack Path and Strategy (APT3 Enhanced, 100 Trials)", fontsize=14,
                  fontweight='bold')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        save_path = os.path.join(viz_dir, "apt3_enhanced_rtu_compromise_by_path.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved RTU compromise plot to {save_path}")
        plt.figure(figsize=(14, 8))
        for strategy_name in strategies:
            rtu_times = [t for t in aggregated_results[strategy_name]['run_statistics']['time_to_rtu_compromise_runs']
                         if t is not None and t != num_steps + 1]
            # Debug: print type and shape
            print(f"[DEBUG] RTU times for {strategy_name}: type={type(rtu_times)}, length={len(rtu_times)}")
            if rtu_times and all(isinstance(x, (int, float, np.integer, np.floating)) for x in rtu_times):
                plt.violinplot([rtu_times], positions=[list(strategies.keys()).index(strategy_name)], widths=0.15, showmeans=True)
            else:
                print(f"Skipping RTU compromise violin plot for {strategy_name}: data is not a flat list of numbers.")
        plt.xlabel('Strategy', fontsize=12)
        plt.ylabel('Time to RTU Compromise (steps)', fontsize=12)
        plt.title(f'Distribution of Time to RTU Compromise Across {num_trials} Trials per Strategy', fontsize=14)
        plt.xticks(range(len(strategies)), strategies.keys(), rotation=45, ha='right')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        save_path = os.path.join(viz_dir, "apt3_enhanced_time_to_rtu_compromise_distribution.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved RTU compromise time distribution to {save_path}")
        summary_file = os.path.join(output_dir, "simulation_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(aggregated_results, f, indent=4, default=str)
        print(f"Saved summary to {summary_file}")
        for strategy_name in strategies:
            strategy_summary_file = os.path.join(strategy_dirs[strategy_name],
                                                 f"{strategy_name.replace(' ', '_')}_trials.json")
            with open(strategy_summary_file, 'w') as f:
                json.dump(aggregated_results[strategy_name]['trials'], f, indent=4, default=str)
            print(f"Saved {strategy_name} trials to {strategy_summary_file}")
        return aggregated_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run APT3 RTU-targeted simulation with enhanced strategic attacker")
    parser.add_argument("--data-file", type=str, default="../data/systemData/apt3_scenario_enriched.json",
                        help="Path to system data JSON file")
    parser.add_argument("--num-steps", type=int, default=100, help="Number of simulation steps")
    parser.add_argument("--defender-budget", type=int, default=7500, help="Defender's patching budget in dollars")
    parser.add_argument("--attacker-budget", type=int, default=15000, help="Attacker's budget in dollars")
    parser.add_argument("--psi", type=float, default=1.0, help="Weight balancing risk and operational cost")
    parser.add_argument("--cost-aware-attacker", action="store_true", default=True)
    parser.add_argument("--cost-aware-defender", action="store_true", default=True)
    parser.add_argument("--detection-averse", action="store_true", default=True)
    parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor for payoffs")
    parser.add_argument("--business-values-file", type=str, help="Path to business values JSON file")
    parser.add_argument("--use-hmm", action="store_true", default=False)
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--output-dir", type=str, default="apt3_simulation_results")
    parser.add_argument("--attacker-sophistication", type=float, default=0.9)
    parser.add_argument("--cost-cache-file", type=str, default=None)
    parser.add_argument("--num-trials", type=int, default=100, help="Number of simulation trials")
    args = parser.parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"simulation_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    try:
        simulation = APT3RTUSimulation(
            data_file=args.data_file,
            num_steps=args.num_steps,
            defender_budget=args.defender_budget,
            attacker_budget=args.attacker_budget,
            psi=args.psi,
            cost_aware_attacker=args.cost_aware_attacker,
            cost_aware_defender=args.cost_aware_defender,
            detection_averse=args.detection_averse,
            gamma=args.gamma,
            business_values_file=args.business_values_file,
            use_hmm=args.use_hmm,
            attacker_sophistication=args.attacker_sophistication,
            cost_cache_file=args.cost_cache_file
        )
        simulation.output_dir = output_dir
        results = simulation.compare_strategies(
            defender_budget=args.defender_budget,
            num_steps=args.num_steps,
            num_trials=args.num_trials,
            verbose=args.verbose
        )
        summary_file = os.path.join(output_dir, "simulation_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=4, default=str)
        print(f"Saved summary to {summary_file}")
        strategic_summary = {name: [trial.get('attacker_metrics', {}) for trial in result['trials']] for name, result in results.items()}
        strategic_file = os.path.join(output_dir, "strategic_metrics.json")
        with open(strategic_file, 'w') as f:
            json.dump(strategic_summary, f, indent=4)
        print(f"Saved strategic metrics to {strategic_file}")
        logger.info(f"Enhanced APT3 simulation completed successfully with {args.num_trials} trials")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        logger.error(f"FileNotFoundError: {e}")
    except ValueError as e:
        print(f"Validation error: {e}")
        logger.error(f"Validation error: {e}")
    except TypeError as e:
        print(f"Serialization error: {e}")
        logger.error(f"Serialization error: {e}", exc_info=True)
    except Exception as e:
        print(f"Unexpected error: {e}")
        logger.error(f"Unexpected error: {e}", exc_info=True)