#!/usr/bin/env python3
"""
APT3 RTU Simulation with POMDP Attacker and Multiple Defender Strategies
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
import pickle

from classes.state import Asset, State, System, Vulnerability, create_vuln_key
from classes.mitre import MitreMapper, APT3TacticMapping, KillChainStage
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
from classes.hybrid_strategy import HybridStrategy
from data_loader import load_data
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
            "rtu_compromised_step": None, "strategic_decisions": [],
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
                    patch_cost = self.cost_calculator.calculate_patch_cost(vuln, self.state.system, asset, component_id=str(comp.id))
                    exploit_cost = self.cost_calculator.calculate_exploit_cost(vuln, self.state.system, asset, component_id=str(comp.id))
                    exploit_loss = self.cost_calculator.calculate_exploit_loss(vuln, self.state.system, asset, component_id=str(comp.id))
                    cost_cache['patch_costs'][vuln_key] = patch_cost
                    cost_cache['exploit_costs'][vuln_key] = exploit_cost
                    cost_cache['exploit_losses'][vuln_key] = exploit_loss
        self._cost_cache = cost_cache
        logger.info(f"Cost cache initialized with {len(cost_cache['patch_costs'])} entries")
        return cost_cache

    def get_next_attack_action(self, state):
        if state is None or not isinstance(state, State):
            logger.error("Invalid state provided: state is None or not a State instance")
            return {'action_type': 'pause', 'reason': 'invalid_state'}
        if not hasattr(state, 'system') or state.system is None or not state.system.assets:
            logger.error("State has no valid system or assets")
            return {'action_type': 'pause', 'reason': 'missing_system_or_assets'}

        try:
            # FIXED: Give attacker a deep copy of the state to prevent seeing defender patches
            # that happen later in the same step
            import copy
            attacker_state = copy.deepcopy(state)
            self.attacker.state = attacker_state
            
            # Check if attacker has been stuck in reconnaissance for too long
            exploit_attempts = self.results.get('exploit_attempts', [])
            if exploit_attempts is None:
                exploit_attempts = []
            recon_count = sum(1 for action in exploit_attempts
                              if action.get('action_type') == 'reconnaissance' and action.get('success'))
            if recon_count > 5 and state.k == KillChainStage.RECONNAISSANCE.value:
                logger.info("Reconnaissance overused, forcing initial_access action")
                action = self.attacker.select_action(attacker_state, force_action_type='initial_access')
            else:
                action = self.attacker.select_action(attacker_state)

            if not action or 'action_type' not in action:
                logger.warning("Invalid action returned, attempting fallback initial_access")
                action = self.attacker.select_action(attacker_state, force_action_type='initial_access')
                if not action or 'action_type' not in action:
                    logger.error("Failed to generate valid action")
                    self.attacker.path_failures += 1
                    return {'action_type': 'pause', 'reason': 'invalid_action'}

            # Verify action cost against budget
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

    def _should_skip_due_to_failures(self, vuln_id, asset_id, component_id):
        vuln_key = create_vuln_key(vuln_id, str(asset_id), str(component_id))

        # Use strategic manager's failure count as single source of truth
        failure_count = self.attacker.strategic_manager.exploit_failures.get(vuln_key, 0)

        # Consistent thresholds matching strategic manager
        STANDARD_MAX_FAILURES = 3
        PRIORITY_MAX_FAILURES = 5

        max_failures = PRIORITY_MAX_FAILURES if vuln_id in {'CVE-2018-13379', 'ZERO-DAY-001',
                                                            'CVE-2015-3113'} else STANDARD_MAX_FAILURES

        if failure_count >= max_failures:
            logger.info(f"Skipping {vuln_key} due to {failure_count} previous failures (threshold: {max_failures})")
            return True
        return False

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

            # FIXED: Remove belief state validation since HybridGraphPOSGAttackerAPT3 doesn't have this method
            # The belief state is handled internally by the attacker's POMDP policy

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

            if getattr(vuln, 'is_exploited', False):
                logger.debug(f"Vulnerability {vuln.cve_id} already exploited")
                return {
                    'action_type': 'initial_access', 'target_vuln': vuln.cve_id,
                    'target_asset': str(asset.asset_id), 'target_component': str(comp.id),
                    'action_result': False, 'reason': 'already_exploited'
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

            logger.info(
                f"Initial access attempt: {vuln.cve_id} on {asset.name} - Probability: {final_probability:.3f}, Success: {is_successful}")

            action_details = {
                'action_type': 'initial_access', 'target_vuln': vuln.cve_id,
                'target_asset': str(asset.asset_id), 'target_component': str(comp.id),
                'cost': exploit_cost, 'probability': final_probability,
                'action_result': is_successful, 'vuln_key': vuln_key,
                'tactic': action.get('tactic', 'Initial Access')
            }

            if is_successful:
                logger.info(f"Attempting initial access via {vuln.cve_id} on {asset.name}")
                logger.info(f"Successfully gained initial access via {vuln.cve_id} on {asset.name}")
                vuln.is_exploited = True
                asset.mark_as_compromised(True)
                asset._compromise_time = self.current_step
                action_details['action_result'] = True
                action_details['reason'] = 'success'
                
                # Track spearphishing attempts for assets that are entry points
                mitre_techniques = getattr(vuln, 'mitre_techniques', [])
                if any(t in ['T1566.001', 'T1566.002', 'T1566.003', 'T1566'] for t in mitre_techniques):
                    spearphishing_attempts = self.results.get('spearphishing_attempts', [])
                    if spearphishing_attempts is None:
                        spearphishing_attempts = []
                        self.results['spearphishing_attempts'] = spearphishing_attempts
                    
                    spearphishing_attempts.append({
                        'step': self.current_step, 
                        'vuln_id': vuln.cve_id, 
                        'asset_id': str(asset.asset_id),
                        'success': True,
                        'technique': next((t for t in mitre_techniques if t in ['T1566.001', 'T1566.002', 'T1566.003', 'T1566']), 'T1566')
                    })
                    logger.info(f"Tracked successful spearphishing attempt on asset {asset.asset_id}")
                    
                    # FIXED: Remove belief state update since HybridGraphPOSGAttackerAPT3 doesn't have direct belief_state access
                    # The belief state is managed internally by the POMDP policy
            else:
                logger.info(f"Failed initial access attempt on {vuln.cve_id} on {asset.name}")
                attempt_cost = exploit_cost * 0.3
                self._remaining_attacker_budget -= attempt_cost
                action_details['attempt_cost'] = attempt_cost
                
                # Track failed spearphishing attempts
                mitre_techniques = getattr(vuln, 'mitre_techniques', [])
                if any(t in ['T1566.001', 'T1566.002', 'T1566.003', 'T1566'] for t in mitre_techniques):
                    self.results['spearphishing_attempts'].append({
                        'step': self.current_step, 
                        'vuln_id': vuln.cve_id, 
                        'asset_id': str(asset.asset_id),
                        'success': False,
                        'technique': next((t for t in mitre_techniques if t in ['T1566.001', 'T1566.002', 'T1566.003', 'T1566']), 'T1566')
                    })
                    logger.info(f"Tracked failed spearphishing attempt on asset {asset.asset_id}")
                    
                    # Update belief state for phishing failure if available
                    if hasattr(self.attacker, 'belief_state') and self.attacker.belief_state:
                        vuln_key = create_vuln_key(vuln.cve_id, str(asset.asset_id), str(comp.id))
                        if hasattr(self.attacker.belief_state, 'phishing_success_prob'):
                            if str(asset.asset_id) not in self.attacker.belief_state.phishing_success_prob:
                                self.attacker.belief_state.phishing_success_prob[str(asset.asset_id)] = 0.5
                            # Decrease phishing success probability
                            current_prob = self.attacker.belief_state.phishing_success_prob[str(asset.asset_id)]
                            self.attacker.belief_state.phishing_success_prob[str(asset.asset_id)] = max(0.0, current_prob - 0.1)
                            logger.debug(f"Updated phishing success probability for asset {asset.asset_id}: {current_prob:.3f} -> {self.attacker.belief_state.phishing_success_prob[str(asset.asset_id)]:.3f}")

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

            if getattr(vuln, 'is_exploited', False):
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
            target_asset_id = action.get('target_asset')
            current_position = self._determine_current_position(state)
            target_asset = self._get_asset_by_id(target_asset_id)

            if not target_asset:
                logger.error(f"Target asset {target_asset_id} not found")
                return {
                    'action_type': 'lateral_movement', 'target_asset': target_asset_id,
                    'action_result': False, 'reason': 'target_asset_not_found'
                }

            if not self._are_assets_connected(current_position, target_asset_id):
                logger.warning(f"Assets {current_position} and {target_asset_id} are not connected")
                return {
                    'action_type': 'lateral_movement', 'target_asset': target_asset_id,
                    'action_result': False, 'reason': 'assets_not_connected'
                }

            # Check for credential-based movement first
            action_vuln_id = action.get('target_vuln')
            if action_vuln_id == "CREDENTIAL_BASED":
                return self._execute_credential_based_movement(current_position, target_asset_id, action)

            # HYBRID APPROACH: Handle combined vulnerability identifiers
            if action_vuln_id and '+' in action_vuln_id:
                return self._execute_hybrid_lateral_movement(current_position, target_asset_id, action_vuln_id, action)

            # FALLBACK: Original single vulnerability approach
            current_asset = self._get_asset_by_id(current_position)
            
            if not current_asset:
                logger.error(f"Current asset {current_position} not found")
                return {
                    'action_type': 'lateral_movement', 'target_asset': target_asset_id,
                    'action_result': False, 'reason': 'current_asset_not_found'
                }

            # Find the vulnerability on the CURRENT asset (not target asset)
            vuln_info = None
            if action_vuln_id:
                # Try to find the specific vulnerability specified in the action
                vuln_info = self._find_specific_vulnerability(current_asset, action_vuln_id)
                if vuln_info:
                    logger.info(f"Found specified vulnerability {action_vuln_id} on current asset {current_position}")
                else:
                    logger.warning(f"Specified vulnerability {action_vuln_id} not found on current asset {current_position}")
            
            # If specified vulnerability not found, fall back to best available vulnerability on current asset
            if not vuln_info:
                vuln_info = self._find_exploitable_vulnerability(current_asset)
                if vuln_info:
                    logger.info(f"Using fallback vulnerability {vuln_info['cve_id']} on current asset {current_position}")
                    if action_vuln_id and action_vuln_id != vuln_info['cve_id']:
                        logger.warning(f"Vulnerability mismatch: action specified {action_vuln_id}, but using fallback {vuln_info['cve_id']}")
                else:
                    logger.error(f"No exploitable vulnerabilities found on current asset {current_position}")
                    return {
                        'action_type': 'lateral_movement', 'target_asset': target_asset_id,
                        'action_result': False, 'reason': 'no_exploitable_vulns_on_current_asset'
                    }

            # Create vulnerability key for the current asset
            vuln_key = create_vuln_key(vuln_info['cve_id'], str(current_position), str(vuln_info['component_id']))
            
            if vuln_key not in self.vuln_lookup:
                logger.error(f"Vulnerability key {vuln_key} not found in lookup table")
                return {
                    'action_type': 'lateral_movement', 'target_asset': target_asset_id,
                    'action_result': False, 'reason': 'vuln_key_not_found'
                }

            vuln, asset, comp = self.vuln_lookup[vuln_key]
            cve_id = vuln_info['cve_id']
            comp_id = vuln_info['component_id']

            # Validate that we're using the vulnerability on the correct asset
            if str(asset.asset_id) != str(current_position):
                logger.error(f"Vulnerability {cve_id} is on asset {asset.asset_id}, not current asset {current_position}")
                return {
                    'action_type': 'lateral_movement', 'target_asset': target_asset_id,
                    'action_result': False, 'reason': 'vuln_on_wrong_asset'
                }

            logger.info(f"Using vulnerability {cve_id} on current asset {current_position} to move to target asset {target_asset_id}")

            if self._should_skip_due_to_failures(cve_id, current_position, comp_id):
                logger.info(f"Skipping lateral movement from {current_position} to {target_asset_id} via {vuln_key} due to excessive failures")
                return {
                    'action_type': 'lateral_movement', 'target_asset': target_asset_id,
                    'action_result': False, 'reason': 'excessive_failures'
                }

            if vuln.is_patched:
                logger.debug(f"Vulnerability {vuln.cve_id} is patched")
                return {
                    'action_type': 'lateral_movement', 'target_vuln': vuln.cve_id,
                    'target_asset': str(target_asset_id), 'source_asset': str(current_position),
                    'action_result': False, 'reason': 'patched'
                }

            if vuln.is_exploited:
                logger.debug(f"Vulnerability {vuln.cve_id} already exploited")
                return {
                    'action_type': 'lateral_movement', 'target_vuln': vuln.cve_id,
                    'target_asset': str(target_asset_id), 'source_asset': str(current_position),
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
                    f"Failed lateral movement from {current_position} to {target_asset_id}: insufficient budget (cost={cost}, budget={self._remaining_attacker_budget})")
                return {
                    'action_type': 'lateral_movement', 'target_asset': target_asset_id,
                    'action_result': False, 'reason': 'insufficient_budget'
                }

            self._lateral_movement_attempts[vuln_key] = previous_attempts + 1
            is_successful = random.random() < final_probability

            action_details = {
                'action_type': 'lateral_movement', 'target_asset': target_asset_id,
                'source_asset': current_position, 'cost': cost, 'probability': final_probability,
                'action_result': is_successful, 'attempt_number': self._lateral_movement_attempts[vuln_key],
                'tactic': action.get('tactic', 'Lateral Movement'), 'target_vuln': cve_id,
                'source_component': str(comp_id), 'vuln_key': vuln_key
            }

            if is_successful:
                logger.info(f"Successfully moved laterally from {current_position} to {target_asset_id} via {cve_id}")
                vuln.is_exploited = True
                self._exploited_cves.add(cve_id)
                target_asset.mark_as_compromised(True)
                target_asset.record_compromise(self.current_step)
                self._remaining_attacker_budget -= cost
                self.attacker.current_compromised_node = target_asset_id
                self.attacker.compromised_nodes.add(target_asset_id)
                self.attacker.strategic_manager.compromised_assets.add(target_asset_id)
                self.attacker.strategic_manager.exploited_vulnerabilities.add(vuln_key)
                if not hasattr(self.state, 'lateral_movement_targets'):
                    self.state.lateral_movement_targets = set()
                self.state.lateral_movement_targets.add(target_asset_id)
                if not hasattr(self.state, 'lateral_movement_chain'):
                    self.state.lateral_movement_chain = {}
                self.state.lateral_movement_chain[target_asset_id] = current_position
                for sys_asset in self.state.system.assets:
                    if str(sys_asset.asset_id) == str(target_asset.asset_id):
                        sys_asset.mark_as_compromised(True)
                        sys_asset.record_compromise(self.current_step)
                        logger.debug(f"System asset {sys_asset.asset_id} marked as compromised")
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
                'target_asset': str(target_asset_id),
                'action_result': False,
                'reason': f'error_{str(e)}'
            }

    def _find_specific_vulnerability(self, asset, target_cve_id):
        """
        Find a specific vulnerability by CVE ID on the given asset.
        
        Args:
            asset: Asset object to search
            target_cve_id: CVE ID to find
            
        Returns:
            dict: Vulnerability info if found and exploitable, None otherwise
        """
        logger.info(f"[DIAG] Execution: Searching for {target_cve_id} on asset {asset.asset_id}")
        logger.info(f"[DIAG] Execution: All vulnerabilities on asset {asset.asset_id}:")
        for comp in asset.components:
            for vuln in comp.vulnerabilities:
                logger.info(f"[DIAG]   {vuln.cve_id} (patched={vuln.is_patched}, exploited={getattr(vuln, 'is_exploited', False)}) on component {comp.id}")
        
        for comp in asset.components:
            for vuln in comp.vulnerabilities:
                if vuln.cve_id == target_cve_id:
                    # Check if vulnerability is exploitable
                    if vuln.is_patched or getattr(vuln, 'is_exploited', False):
                        logger.debug(f"Vulnerability {target_cve_id} is patched or already exploited")
                        continue
                    
                    # Calculate priority score for consistency
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
                    
                    logger.info(f"[DIAG] Execution: Found {target_cve_id} on component {comp.id}")
                    return {
                        'cve_id': vuln.cve_id,
                        'component_id': str(comp.id),
                        'priority_score': priority_score,
                        'is_network_accessible': is_network_accessible,
                        'is_priority_cve': is_priority_cve
                    }
        
        logger.debug(f"Vulnerability {target_cve_id} not found on asset {asset.asset_id}")
        return None

    def _try_correct_vulnerability_mismatch(self, asset, target_cve_id, target_asset_id):
        """
        Try to correct a vulnerability mismatch by finding the intended vulnerability.
        
        Args:
            asset: Asset object to search
            target_cve_id: Intended CVE ID
            target_asset_id: Target asset ID
            
        Returns:
            dict: Corrected vulnerability info if found, None otherwise
        """
        logger.debug(f"Attempting to correct vulnerability mismatch: looking for {target_cve_id} on asset {target_asset_id}")
        
        for comp in asset.components:
            for vuln in comp.vulnerabilities:
                if vuln.cve_id == target_cve_id:
                    logger.debug(f"Found vulnerability {target_cve_id} on component {comp.id}")
                    
                    # Check if vulnerability is exploitable
                    if vuln.is_patched:
                        logger.debug(f"Vulnerability {target_cve_id} is patched, skipping")
                        continue
                    if getattr(vuln, 'is_exploited', False):
                        logger.debug(f"Vulnerability {target_cve_id} is already exploited, skipping")
                        continue
                    
                    # FIXED: Remove belief state validation since HybridGraphPOSGAttackerAPT3 doesn't have this method
                    # The belief state is handled internally by the attacker's POMDP policy
                    
                    logger.info(f"Successfully found exploitable vulnerability {target_cve_id} on asset {target_asset_id}")
                    
                    # Return the corrected vulnerability info
                    return {
                        'cve_id': vuln.cve_id,
                        'component_id': str(comp.id),
                        'priority_score': getattr(vuln, 'epss', 0.0),
                        'is_network_accessible': 'AV:N' in getattr(vuln, 'cvssV3Vector', ''),
                        'is_priority_cve': vuln.cve_id in {'CVE-2018-13379', 'ZERO-DAY-001', 'CVE-2015-3113'}
                    }
        
        # If we get here, the vulnerability was not found or is not exploitable
        logger.warning(f"Could not find exploitable vulnerability {target_cve_id} on asset {target_asset_id}")
        logger.debug(f"Available vulnerabilities on asset {target_asset_id}: {[v.cve_id for comp in asset.components for v in comp.vulnerabilities]}")
        return None

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

                # Limit reconnaissance frequency
                recon_count = sum(1 for a in getattr(self.state.system, 'action_history', [])
                                  if a.get('action_type') == 'reconnaissance' and a.get('action_result', False))
                if recon_count >= 5 and is_successful:
                    logger.info("Limiting reconnaissance, suggesting kill chain progression")
                    self.state.suggest_attacker_stage(KillChainStage.DELIVERY.value)

                logger.debug(f"Returning reconnaissance action details: {action_details}")
                return action_details

            # Handle other actions (e.g., persistence, command_and_control, exfiltration)
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

    def _determine_current_position(self, state: State) -> Optional[str]:
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

    def _get_asset_by_id(self, asset_id: str):
        asset_id_str = str(asset_id)
        for asset in self.system.assets:
            if str(asset.asset_id) == asset_id_str:
                return asset
        logger.warning(f"No asset found with ID: {asset_id_str}")
        return None

    def _are_assets_connected(self, asset1_id: str, asset2_id: str) -> bool:
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
        
        # Calculate detection coverage - only for Threat Intelligence strategy
        if hasattr(self, 'current_strategy') and hasattr(self.current_strategy, '__class__') and 'ThreatIntelligence' in self.current_strategy.__class__.__name__:
            # For Threat Intelligence, calculate based on observations collected
            total_observations = len(self.results.get('threat_intelligence_observations', []))
            total_attack_attempts = len([a for a in self.results.get('exploit_attempts', []) 
                                       if a.get('action_type') not in ['reconnaissance', 'none']])
            
            if total_attack_attempts > 0:
                # Calculate detection coverage as ratio of observations to attack attempts
                # This should yield ~66.7% for 10 observations and 15 attempts
                base_coverage = total_observations / total_attack_attempts
                # Add bonus for threat intelligence observations (max 20% bonus)
                threat_intel_bonus = min(0.2, total_observations * 0.02)
                detection_coverage = min(1.0, base_coverage + threat_intel_bonus)
                logger.info(f"Threat Intelligence detection coverage: {detection_coverage:.3f} (observations: {total_observations}, attempts: {total_attack_attempts}, base: {base_coverage:.3f}, bonus: {threat_intel_bonus:.3f})")
            else:
                detection_coverage = 0.0
                logger.warning("No attack attempts found for detection coverage calculation")
        else:
            # Other strategies have no detection capabilities
            detection_coverage = 0.0
            
        avg_time_to_detection = sum(self.time_to_detection.values()) / len(
            self.time_to_detection) if self.time_to_detection else 0.0
        attack_disruption_rate = 1.0 if self.attack_disrupted else 0.0

        # Use action_history for metrics
        action_history = getattr(state.system, 'action_history', self.results["exploit_attempts"])
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

    def run_step(self, strategy, step, verbose=False):
        """Enhanced run_step method with Threat Intelligence learning integration."""
        self.current_step = step
        self.current_strategy = strategy  # Track current strategy for detection coverage
        if verbose:
            print(f"\n==== Step {step + 1} ====")
            print(f"Current kill chain stage: {KillChainStage(self.state.k).name}")
            print(f"Remaining defender budget: ${self._remaining_defender_budget:.2f}")
            print(f"Remaining attacker budget: ${self._remaining_attacker_budget:.2f}")

        self.threat_processor.update_threat_levels(self.state)

        # Get attacker action
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
        attack_observations = []  # For Threat Intelligence strategy learning

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
            # Skip actions that should fail due to excessive failures
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

            # Create attack observations for Threat Intelligence learning
            attack_observation = self._create_attack_observation(action, action_result, step)
            if attack_observation:
                attack_observations.append(attack_observation)

            # Let the attacker observe the result (this will update strategic manager's failure counts)
            self.attacker.observe_result(action_result)

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
        # Check if the current strategy is Threat Intelligence and feed observations
        for observation in attack_observations:
            strategy.observe_attack_behavior(observation)
            if verbose:
                print(
                    f"  Threat Intel: Observed attack - {observation.get('action_type')} on asset {observation.get('target_asset')} ({'successful' if observation.get('success') else 'failed'})")

        # Execute defender strategy
        defender_actions = strategy(self.state, self._remaining_defender_budget, step, self.num_steps)
        step_cost = 0.0
        applied_patches = []

        for vuln, cost in defender_actions:
            vuln_key = None
            vuln_asset = None
            vuln_component = None
            target_vuln = None
            for asset in self.state.system.assets:
                for comp in asset.components:
                    for v in comp.vulnerabilities:
                        if v.cve_id == vuln.cve_id:
                            vuln_key = create_vuln_key(v.cve_id, str(asset.asset_id), str(comp.id))
                            vuln_asset = asset
                            vuln_component = comp
                            target_vuln = v
                            break
                    if vuln_key:
                        break
                if vuln_key:
                    break
            if vuln_key and vuln_asset and vuln_component and not target_vuln.is_patched and cost <= self._remaining_defender_budget:
                target_vuln.apply_patch()
                applied_patches.append(target_vuln)
                step_cost += cost
                self._patched_cves.add(target_vuln.cve_id)
                self._remaining_defender_budget -= cost
                # Update vuln_lookup to ensure consistency
                self.vuln_lookup[vuln_key] = (target_vuln, vuln_asset, vuln_component)
                logger.debug(f"Updated vuln_lookup for {vuln_key}: is_patched={target_vuln.is_patched}")
                if verbose:
                    print(f"Defender: Patched {target_vuln.cve_id} on asset {vuln_asset.asset_id} (Cost: ${cost:.2f})")

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
                vuln_key = create_vuln_key(target_vuln_id, target_asset_id, target_comp_id)
                if vuln_key in self.vuln_lookup:
                    vuln, _, _ = self.vuln_lookup[vuln_key]
                    if not vuln.is_patched:  # Double-check to prevent patched vuln exploits
                        attacker_vuln_objects.append(vuln)
                        logger.debug(f"Added {vuln.cve_id} on asset {target_asset_id} to attacker_vuln_objects")
                    else:
                        logger.warning(f"Skipped {vuln_key} for attacker_vuln_objects: vulnerability is patched")

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

    def _create_attack_observation(self, action: Dict, action_result: Dict, step: int) -> Optional[Dict]:
        """
        Create attack observation from action and result for Threat Intelligence learning.
        This simulates what an advanced IDS/SIEM system would detect and report.

        Returns:
            Dict: Attack observation with standardized format, or None if not observable
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
            'timestamp': step  # Use step as timestamp for simplicity
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
            observation['detection_confidence'] = 0.6

        # Add contextual information for better threat intelligence
        observation['tactic'] = action.get('tactic', action_result.get('tactic', 'Unknown'))

        return observation

    def compare_strategies(self, defender_budget: int, num_steps: int, verbose: bool = False):
        """Enhanced strategy comparison with proper Threat Intelligence learning integration."""
        print(f"\n{'=' * 80}\nCOMPARING STRATEGIES - Budget: ${defender_budget:,.2f}, Steps: {num_steps}\n{'=' * 80}")
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
        results = {}

        for strategy_name, strategy_obj in strategies.items():
            logger.info(f"Starting simulation for defender strategy: {strategy_name}")
            print(f"\n{'-' * 40}\nRUNNING STRATEGY: {strategy_name}\n{'-' * 40}")

            # Log whether this strategy has threat intelligence capabilities
            print(f"  Strategy uses basic vulnerability management approach")

            # Reinitialize attacker for each strategy to ensure no state carryover
            logger.info(f"Creating new HybridGraphPOSGAttackerAPT3 instance for strategy: {strategy_name}")
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

            # Initialize results tracking for this strategy
            results[strategy_name] = {
                'protected_value': [], 'lost_value': [], 'value_preserved': [], 'unpatched_critical': [],
                'total_patch_cost': 0.0, 'total_patches': 0, 'total_patch_time': 0.0, 'roi': 0.0,
                'compromised_assets': [], 'final_compromised_assets': 0, 'exploit_attempts': [],
                'exploit_events': [], 'rtu_compromised_step': None, 'detection_metrics': {
                    'detection_coverage': 0.0, 'avg_time_to_detection': 0.0, 'attack_disruption_rate': 0.0
                }, 'strategic_metrics': {}, 'apt3_metrics': {
                    'spearphishing_success_rate': 0.0, 'credential_harvesting_count': 0
                }, 'threat_intelligence_metrics': {
                    'observations_collected': 0,
                    'learning_adaptations': 0,
                    'threat_level_changes': 0
                }
            }

            # Reset simulation state
            self.system = copy.deepcopy(original_system)
            self.state = State(k=KillChainStage.RECONNAISSANCE.value, system=self.system)
            # Reinitialize vuln_lookup for the new system
            self.vuln_lookup = {}
            for asset in self.system.assets:
                for comp in asset.components:
                    for vuln in comp.vulnerabilities:
                        vuln_key = create_vuln_key(vuln.cve_id, str(asset.asset_id), str(comp.id))
                        self.vuln_lookup[vuln_key] = (vuln, asset, comp)
            logger.info(f"Reinitialized vuln_lookup with {len(self.vuln_lookup)} entries for strategy {strategy_name}")
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
            self.results = {k: [] if isinstance(v, list) else None if k == 'rtu_compromised_step' else v for k, v in
                            self.results.items()}

            # Initialize strategy
            strategy_obj.initialize(self.state, self._cost_cache)
            strategy_obj.state = self.state
            
            # Special handling for RL Defender strategy
            if strategy_name == 'RL Defender':
                # Load the most recent trained Q-table
                import os
                import pickle
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
                                    q_table = pickle.load(f)
                                strategy_obj.q_table = q_table
                                print(f"  Loaded trained Q-table from {q_table_path}")
                                print(f"  Q-table size: {len(q_table)} entries")
                            except Exception as e:
                                print(f"  Warning: Could not load Q-table: {e}")
                        else:
                            print(f"  Warning: Q-table not found at {q_table_path}")
                    else:
                        print(f"  Warning: No training directories found in {rl_results_dir}")
                else:
                    print(f"  Warning: RL training results directory not found: {rl_results_dir}")
            
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
                                    q_table = pickle.load(f)
                                strategy_obj.rl_defender.q_table = q_table
                                print(f"  Loaded trained Q-table for Hybrid RL component from {q_table_path}")
                                print(f"  Q-table size: {len(q_table)} entries")
                            except Exception as e:
                                print(f"  Warning: Could not load Q-table for Hybrid RL component: {e}")
                        else:
                            print(f"  Warning: Q-table not found at {q_table_path}")
                    else:
                        print(f"  Warning: No training directories found in {rl_results_dir}")
                else:
                    print(f"  Warning: RL training results directory not found: {rl_results_dir}")
                
                print(f"  Hybrid Strategy initialized with TI weight: {strategy_obj.threat_intelligence_weight:.3f}, RL weight: {strategy_obj.rl_weight:.3f}")
            
            self.reset_exploit_status()

            # Track initial threat intelligence state
            initial_observations = 0
            initial_threat_levels = {}
            if hasattr(strategy_obj, 'asset_threat_levels'):
                initial_observations = len(strategy_obj.asset_threat_levels)
                initial_threat_levels = dict(strategy_obj.asset_threat_levels)

            # Run simulation steps
            total_patch_cost = 0.0
            total_patches = 0
            total_patch_time = 0.0
            termination_reason = "campaign_horizon_elapsed"

            for step in range(num_steps):
                self.run_step(strategy_obj.select_patches, step, verbose)
                total_patch_cost += sum(self.results["dollar_costs"]["patch_costs"][-1:])
                total_patches += len(self.results["patched_vulns"][-1:])
                total_patch_time += len(self.results["patched_vulns"][-1:])
                
                # Check for early termination
                should_terminate, reason = self.should_terminate_simulation(step, num_steps, total_patch_cost)
                if should_terminate:
                    termination_reason = reason
                    if verbose:
                        print(f"\nSimulation terminated at step {step + 1}: {reason}")
                    break

            # Calculate final metrics
            final_metrics = self.calculate_metrics(self.state, total_patch_cost, total_patches, verbose)
            # Override total_patches with the actual count
            final_metrics['total_patches'] = total_patches
            if final_metrics['lost_value'] > total_business_value:
                final_metrics['lost_value'] = total_business_value
                final_metrics['protected_value'] = 0.0
                final_metrics['value_preserved'] = max(0.0, total_business_value - final_metrics['total_patch_cost'])

            # Get strategic metrics
            strategic_metrics = self.attacker.get_strategic_metrics() if hasattr(self.attacker,
                                                                                 'get_strategic_metrics') else {}

            # Calculate threat intelligence specific metrics
            threat_intel_metrics = {
                'observations_collected': 0,
                'learning_adaptations': 0,
                'threat_level_changes': 0,
                'predictions_made': 0
            }

            if hasattr(strategy_obj, 'asset_threat_levels'):
                final_observations = len(strategy_obj.asset_threat_levels)
                final_threat_levels = dict(strategy_obj.asset_threat_levels)

                threat_intel_metrics.update({
                    'observations_collected': final_observations,
                    'learning_adaptations': final_observations - initial_observations,
                    'threat_level_changes': sum(1 for asset_id in final_threat_levels
                                                if abs(
                        final_threat_levels[asset_id] - initial_threat_levels.get(asset_id, 0.3)) > 0.1),
                    'predictions_made': len(strategy_obj.predict_next_targets()) if hasattr(strategy_obj,
                                                                                            'predict_next_targets') else 0,
                    'compromise_sequence_learned': len(strategy_obj.compromise_sequence),
                    'exploit_attempts_tracked': len(strategy_obj.exploit_attempt_history),
                    'techniques_learned': len(strategy_obj.technique_frequency)
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
            results[strategy_name].update({
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
                'detection_metrics': {
                    'detection_coverage': final_metrics['detection_coverage'] * 100,
                    'avg_time_to_detection': final_metrics['avg_time_to_detection'],
                    'attack_disruption_rate': final_metrics['attack_disruption_rate'] * 100
                },
                'strategic_metrics': strategic_metrics,
                'apt3_metrics': {
                    'spearphishing_success_rate': final_metrics['spearphishing_success_rate'] * 100,
                    'credential_harvesting_count': final_metrics['credential_harvesting_count']
                },
                'threat_intelligence_metrics': threat_intel_metrics,
                'hybrid_metrics': hybrid_metrics
            })

            print(f"\nStrategy {strategy_name} Complete:")
            print(f"  Total Patches: {results[strategy_name]['total_patches']}")
            print(f"  Total Cost: ${results[strategy_name]['total_patch_cost']:,.2f}")
            print(f"  Protected Value: ${final_metrics['protected_value']:,.2f}")
            print(f"  Lost Value: ${final_metrics['lost_value']:,.2f}")
            print(f"  Value Preserved: ${final_metrics['value_preserved']:,.2f}")
            print(f"  ROI: {final_metrics['roi']:.1f}%")
            print(f"  Compromised Assets: {results[strategy_name]['final_compromised_assets']}")
            print(f"  Spearphishing Success Rate: {final_metrics['spearphishing_success_rate'] * 100:.1f}%")
            print(f"  Credential Harvesting Successes: {final_metrics['credential_harvesting_count']}")

            if hasattr(strategy_obj, 'asset_threat_levels'):
                print(f"  === Threat Intelligence Learning Results ===")
                print(f"  Attack Observations: {threat_intel_metrics['observations_collected']}")
                print(f"  Learning Adaptations: {threat_intel_metrics['learning_adaptations']}")
                print(f"  Threat Level Changes: {threat_intel_metrics['threat_level_changes']}")
                print(f"  Compromise Sequence Learned: {threat_intel_metrics['compromise_sequence_learned']} assets")
                print(f"  Exploit Attempts Tracked: {threat_intel_metrics['exploit_attempts_tracked']} CVEs")
                print(f"  Techniques Learned: {threat_intel_metrics['techniques_learned']}")
            
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

        # Display comparison summary
        print("\n" + "=" * 140)
        print(f"STRATEGY COMPARISON SUMMARY - Total Business Value: ${total_business_value:,.2f}")
        print("=" * 140)
        headers = ["Strategy", "Protected Value", "Lost Value", "Value Preserved", "Patches", "Patch Cost", "ROI",
                   "Spearphishing SR", "Cred Harvesting", "Observations"]
        print("{:<15} {:<15} {:<15} {:<15} {:<10} {:<12} {:<8} {:<15} {:<15} {:<12}".format(*headers))
        print("-" * 140)

        for strategy_name, result in results.items():
            observations = result['threat_intelligence_metrics']['observations_collected']
            print(
                "{:<15} ${:<14,.2f} ${:<14,.2f} ${:<14,.2f} {:<10} ${:<11,.2f} {:<8.1f}% {:<14.1f}% {:<15} {:<12}".format(
                    strategy_name, result['protected_value'][0], result['lost_value'][0], result['value_preserved'][0],
                    result['total_patches'], result['total_patch_cost'], result['roi'],
                    result['apt3_metrics']['spearphishing_success_rate'],
                    result['apt3_metrics']['credential_harvesting_count'],
                    observations if observations > 0 else "N/A"
                ))

        best_strategy_name = max(results.keys(), key=lambda s: results[s]['value_preserved'][0])
        print(f"\nBest performing strategy: {best_strategy_name}")

        # Threat Intelligence specific analysis
        threat_intel_results = results.get('Threat Intelligence', {})
        if threat_intel_results.get('threat_intelligence_metrics', {}).get('observations_collected', 0) > 0:
            print(f"\n=== Threat Intelligence Learning Analysis ===")
            ti_metrics = threat_intel_results['threat_intelligence_metrics']
            print(f"Successfully observed {ti_metrics['observations_collected']} attack behaviors")
            print(f"Made {ti_metrics['learning_adaptations']} learning adaptations during simulation")
            print(f"Detected threat level changes on {ti_metrics['threat_level_changes']} assets")
            print(f"Learning enabled more effective defense prioritization")

        # Hybrid Strategy specific analysis
        hybrid_results = results.get('Hybrid Defender', {})
        if hybrid_results.get('hybrid_metrics', {}).get('hybrid_adaptations', 0) > 0:
            print(f"\n=== Hybrid Strategy Analysis ===")
            hybrid_metrics = hybrid_results['hybrid_metrics']
            print(f"Hybrid strategy made {hybrid_metrics['hybrid_adaptations']} adaptations during simulation")
            print(f"Final weight distribution: TI={hybrid_metrics['final_ti_weight']:.3f}, RL={hybrid_metrics['final_rl_weight']:.3f}")
            print(f"Average decision confidence: {hybrid_metrics['average_confidence']:.3f}")
            print(f"Component performance: TI={hybrid_metrics['ti_performance']:.3f}, RL={hybrid_metrics['rl_performance']:.3f}")
            print(f"Combined hybrid performance: {hybrid_metrics['hybrid_performance']:.3f}")
            print(f"Hybrid approach successfully combined threat intelligence and reinforcement learning")

        return results

    def _execute_credential_based_movement(self, current_position, target_asset_id, action):
        """
        Execute credential-based lateral movement using stolen credentials.
        """
        logger.info(f"Attempting credential-based lateral movement from {current_position} to {target_asset_id}")
        
        # Credential-based movement parameters
        credential_cost = 75.0
        base_success_prob = 0.85
        
        if credential_cost > self._remaining_attacker_budget:
            logger.info(f"Failed credential-based movement: insufficient budget (cost={credential_cost}, budget={self._remaining_attacker_budget})")
            return {
                'action_type': 'lateral_movement', 'target_asset': target_asset_id,
                'action_result': False, 'reason': 'insufficient_budget'
            }
        
        # Adjust probability based on target asset security posture
        target_asset = self._get_asset_by_id(target_asset_id)
        security_factor = self._assess_target_security_posture(target_asset)
        success_prob = base_success_prob * security_factor
        
        is_successful = random.random() < success_prob
        
        action_details = {
            'action_type': 'lateral_movement', 'target_asset': target_asset_id,
            'source_asset': current_position, 'cost': credential_cost, 'probability': success_prob,
            'action_result': is_successful, 'movement_type': 'credential_based',
            'tactic': action.get('tactic', 'Lateral Movement'), 'target_vuln': 'CREDENTIAL_BASED'
        }
        
        if is_successful:
            logger.info(f"Successfully moved laterally using credentials from {current_position} to {target_asset_id}")
            target_asset.mark_as_compromised(True)
            target_asset.record_compromise(self.current_step)
            self._remaining_attacker_budget -= credential_cost
            self.attacker.current_compromised_node = target_asset_id
            self.attacker.compromised_nodes.add(target_asset_id)
            self.attacker.strategic_manager.compromised_assets.add(target_asset_id)
            if not hasattr(self.state, 'lateral_movement_targets'):
                self.state.lateral_movement_targets = set()
            self.state.lateral_movement_targets.add(target_asset_id)
            if not hasattr(self.state, 'lateral_movement_chain'):
                self.state.lateral_movement_chain = {}
            self.state.lateral_movement_chain[target_asset_id] = current_position
        else:
            logger.info(f"Failed credential-based lateral movement from {current_position} to {target_asset_id}")
        
        return action_details

    def _execute_hybrid_lateral_movement(self, current_position, target_asset_id, combined_vuln_id, action):
        """
        Execute hybrid lateral movement combining local privilege escalation and network access.
        """
        logger.info(f"Attempting hybrid lateral movement from {current_position} to {target_asset_id}")
        
        # Parse combined vulnerability identifier (format: "local_cve+network_cve")
        vuln_parts = combined_vuln_id.split('+')
        if len(vuln_parts) != 2:
            logger.error(f"Invalid hybrid vulnerability format: {combined_vuln_id}")
            return {
                'action_type': 'lateral_movement', 'target_asset': target_asset_id,
                'action_result': False, 'reason': 'invalid_hybrid_vuln_format'
            }
        
        local_cve_id, network_cve_id = vuln_parts[0], vuln_parts[1]
        
        # Step 1: Execute local privilege escalation on current asset
        local_result = self._execute_local_privilege_escalation(current_position, local_cve_id)
        if not local_result['action_result']:
            logger.info(f"Local privilege escalation failed: {local_result['reason']}")
            return {
                'action_type': 'lateral_movement', 'target_asset': target_asset_id,
                'action_result': False, 'reason': f'local_privesc_failed: {local_result["reason"]}'
            }
        
        # Step 2: Execute network access to target asset
        network_result = self._execute_network_access(current_position, target_asset_id, network_cve_id)
        if not network_result['action_result']:
            logger.info(f"Network access failed: {network_result['reason']}")
            return {
                'action_type': 'lateral_movement', 'target_asset': target_asset_id,
                'action_result': False, 'reason': f'network_access_failed: {network_result["reason"]}'
            }
        
        # Both steps succeeded - lateral movement successful
        total_cost = local_result['cost'] + network_result['cost']
        combined_prob = local_result['probability'] * network_result['probability']
        
        logger.info(f"Successfully completed hybrid lateral movement from {current_position} to {target_asset_id}")
        logger.info(f"Local privesc: {local_cve_id}, Network access: {network_cve_id}")
        
        return {
            'action_type': 'lateral_movement', 'target_asset': target_asset_id,
            'source_asset': current_position, 'cost': total_cost, 'probability': combined_prob,
            'action_result': True, 'movement_type': 'hybrid',
            'local_vuln': local_cve_id, 'network_vuln': network_cve_id,
            'tactic': action.get('tactic', 'Lateral Movement'), 'target_vuln': combined_vuln_id
        }

    def _execute_local_privilege_escalation(self, asset_id, cve_id):
        """
        Execute local privilege escalation on the current asset.
        """
        asset = self._get_asset_by_id(asset_id)
        if not asset:
            return {'action_result': False, 'reason': 'asset_not_found', 'cost': 0, 'probability': 0}
        
        # Find the vulnerability
        vuln_info = self._find_specific_vulnerability(asset, cve_id)
        if not vuln_info:
            return {'action_result': False, 'reason': 'vuln_not_found', 'cost': 0, 'probability': 0}
        
        vuln_key = create_vuln_key(cve_id, str(asset_id), str(vuln_info['component_id']))
        if vuln_key not in self.vuln_lookup:
            return {'action_result': False, 'reason': 'vuln_key_not_found', 'cost': 0, 'probability': 0}
        
        vuln, asset, comp = self.vuln_lookup[vuln_key]
        
        # Check if exploitable
        if vuln.is_patched or vuln.is_exploited:
            return {'action_result': False, 'reason': 'vuln_not_exploitable', 'cost': 0, 'probability': 0}
        
        # Calculate success probability and cost
        base_prob = min(0.9, getattr(vuln, 'epss', 0.1) * 1.5)
        if getattr(vuln, 'exploit', False):
            base_prob *= 1.2
        success_prob = max(0.1, min(0.9, base_prob))
        
        cost = self._cost_cache['exploit_costs'].get(vuln_key, 50 + getattr(vuln, 'cvss', 5.0) * 20)
        
        # Execute the privilege escalation
        is_successful = random.random() < success_prob
        
        if is_successful:
            vuln.is_exploited = True
            self._exploited_cves.add(cve_id)
            self._remaining_attacker_budget -= cost
            logger.info(f"Local privilege escalation successful on {asset_id} using {cve_id}")
        
        return {
            'action_result': is_successful,
            'cost': cost,
            'probability': success_prob,
            'reason': 'success' if is_successful else 'failed'
        }

    def _execute_network_access(self, source_asset_id, target_asset_id, cve_id):
        """
        Execute network access to the target asset.
        """
        target_asset = self._get_asset_by_id(target_asset_id)
        if not target_asset:
            return {'action_result': False, 'reason': 'target_asset_not_found', 'cost': 0, 'probability': 0}
        
        # Find the vulnerability on the target asset
        vuln_info = self._find_specific_vulnerability(target_asset, cve_id)
        if not vuln_info:
            return {'action_result': False, 'reason': 'vuln_not_found', 'cost': 0, 'probability': 0}
        
        vuln_key = create_vuln_key(cve_id, str(target_asset_id), str(vuln_info['component_id']))
        if vuln_key not in self.vuln_lookup:
            return {'action_result': False, 'reason': 'vuln_key_not_found', 'cost': 0, 'probability': 0}
        
        vuln, asset, comp = self.vuln_lookup[vuln_key]
        
        # Check if exploitable
        if vuln.is_patched or vuln.is_exploited:
            return {'action_result': False, 'reason': 'vuln_not_exploitable', 'cost': 0, 'probability': 0}
        
        # Calculate success probability and cost
        base_prob = min(0.9, getattr(vuln, 'epss', 0.1) * 1.5)
        if getattr(vuln, 'exploit', False):
            base_prob *= 1.2
        if 'AV:N' in getattr(vuln, 'cvssV3Vector', ''):
            base_prob *= 1.1
        success_prob = max(0.1, min(0.9, base_prob))
        
        cost = self._cost_cache['exploit_costs'].get(vuln_key, 50 + getattr(vuln, 'cvss', 5.0) * 20)
        
        # Execute the network access
        is_successful = random.random() < success_prob
        
        if is_successful:
            vuln.is_exploited = True
            self._exploited_cves.add(cve_id)
            target_asset.mark_as_compromised(True)
            target_asset.record_compromise(self.current_step)
            self._remaining_attacker_budget -= cost
            self.attacker.current_compromised_node = target_asset_id
            self.attacker.compromised_nodes.add(target_asset_id)
            self.attacker.strategic_manager.compromised_assets.add(target_asset_id)
            self.attacker.strategic_manager.exploited_vulnerabilities.add(vuln_key)
            logger.info(f"Network access successful to {target_asset_id} using {cve_id}")
        
        return {
            'action_result': is_successful,
            'cost': cost,
            'probability': success_prob,
            'reason': 'success' if is_successful else 'failed'
        }

    def _assess_target_security_posture(self, target_asset):
        """
        Assess the security posture of the target asset to adjust credential-based movement probability.
        """
        # Simplified model: adjust based on asset criticality and type
        base_factor = 1.0
        
        # High-value assets typically have better security
        if target_asset.asset_id == '8':  # Primary objective (RTU)
            base_factor *= 0.7  # Lower success probability
        elif target_asset.asset_id in ['2', '3', '6']:  # Secondary objectives
            base_factor *= 0.8
            
        # Adjust based on asset type (simplified)
        asset_name = target_asset.name.lower()
        if 'plc' in asset_name or 'rtu' in asset_name:
            base_factor *= 0.9  # Industrial systems may have different security
        elif 'server' in asset_name:
            base_factor *= 0.8  # Servers typically have better security
            
        return max(0.3, min(1.0, base_factor))  # Ensure reasonable bounds

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
        if step >= max_steps:
            return True, "campaign_horizon_elapsed"
        
        # (iv) Economic break-even: cumulative expenditure exceeds residual business value
        total_business_value = sum(asset.business_value for asset in self.state.system.assets)
        protected_value = sum(asset.business_value for asset in self.state.system.assets if not asset.is_compromised)
        residual_value = protected_value - total_patch_cost
        
        if total_patch_cost > residual_value:
            return True, "economic_break_even_exceeded"
        
        # Additional termination criteria
        # (v) Attacker has achieved objectives (compromised critical assets)
        critical_assets_compromised = sum(1 for asset in self.state.system.assets 
                                        if asset.is_compromised and asset.criticality_level >= 4)
        if critical_assets_compromised >= 2:  # Compromised 2+ critical assets
            return True, "critical_assets_compromised"
        
        # (vi) Attacker budget depleted
        if self._remaining_attacker_budget <= 0:
            return True, "attacker_budget_depleted"
        
        return False, "continue"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run APT3 RTU-targeted simulation with enhanced strategic attacker")
    parser.add_argument("--data-file", type=str, default="../data/systemData/apt3_scenario_enriched.json",
                        help="Path to system data JSON file")
    parser.add_argument("--num-steps", type=int, default=50, help="Number of simulation steps")
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
            verbose=args.verbose
        )
        summary_file = os.path.join(output_dir, "simulation_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=4, default=str)
        print(f"Saved summary to {summary_file}")
        strategic_summary = {name: result.get('strategic_metrics', {}) for name, result in results.items()}
        strategic_file = os.path.join(output_dir, "strategic_metrics.json")
        with open(strategic_file, 'w') as f:
            json.dump(strategic_summary, f, indent=4)
        print(f"Saved strategic metrics to {strategic_file}")
        logger.info("Enhanced APT3 simulation completed successfully")
    except Exception as e:
        print(f"Unexpected error: {e}")
        logger.error(f"Unexpected error: {e}", exc_info=True)