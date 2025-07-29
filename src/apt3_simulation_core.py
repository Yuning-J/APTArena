#!/usr/bin/env python3
"""
APT3 RTU Simulation Core - Base simulation logic
"""

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
from classes.threat_intelligence import ThreatIntelligenceProcessor
from classes.cost import CostCalculator
from classes.attacker_hybrid_apt3 import HybridGraphPOSGAttackerAPT3
from classes.defender_posg import DefenderPOMDPPolicy
from data_loader import load_data

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
            