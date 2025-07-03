"""
This module implements an RL-optimized AdaptiveThreatIntelligenceStrategy for the cybersecurity
simulation. It uses Q-learning to dynamically adjust weights for vulnerability prioritization,
including risk-to-cost ratio for cost efficiency, aligned with CostBenefitStrategy.
"""

import random
import numpy as np
from typing import List, Dict, Tuple, Any, Set
import pickle
import os
from collections import Counter
import logging

from classes.state import State, Vulnerability
from classes.cost import CostCalculator
from classes.patching_strategies import PatchingStrategy

logger = logging.getLogger(__name__)

class RLAdaptiveThreatIntelligenceStrategy(PatchingStrategy):
    """
    RL-optimized adaptive threat intelligence-based vulnerability patching strategy.
    Uses Q-learning to adjust weights, including risk-to-cost ratio for cost efficiency.
    """

    def __init__(self, q_table_file: str = "q_table.pkl"):
        super().__init__("RL Adaptive Threat Intelligence")
        self.patching_history = []
        self._recently_scored_vulns = {}
        self._attacker_actions = []
        self.defender_budget = 0
        self.defender_budget_history = []
        self._last_budget_step = -1
        self._last_budget_value = None
        self._processed_steps = set()
        self._unpatched_vulns_cache = None
        self._current_weights = None
        self.total_patch_cost = 0.0
        self.total_patch_count = 0

        # RL parameters
        self.q_table_file = q_table_file
        self.q_table = self._load_q_table()
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 0.2
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995  # Faster decay for convergence
        self.episode_count = 0
        self.state_bins = {
            'compromise_rate': [0.0, 0.1, 0.3, 1.0],
            'unpatched_critical': [0, 10, 20, 50],
            'budget_remaining': [0.0, 0.25, 0.5, 1.0]
        }
        self.weight_configs = [
            {'cvss_weight': 0.20, 'epss_weight': 0.20, 'exploit_weight': 0.10, 'ransomware_weight': 0.10, 'business_value_weight': 0.00, 'risk_to_cost_weight': 0.40},
            {'cvss_weight': 0.15, 'epss_weight': 0.15, 'exploit_weight': 0.15, 'ransomware_weight': 0.05, 'business_value_weight': 0.10, 'risk_to_cost_weight': 0.40},
            {'cvss_weight': 0.10, 'epss_weight': 0.10, 'exploit_weight': 0.20, 'ransomware_weight': 0.10, 'business_value_weight': 0.10, 'risk_to_cost_weight': 0.40},
            {'cvss_weight': 0.10, 'epss_weight': 0.10, 'exploit_weight': 0.10, 'ransomware_weight': 0.05, 'business_value_weight': 0.25, 'risk_to_cost_weight': 0.40},
            {'cvss_weight': 0.25, 'epss_weight': 0.25, 'exploit_weight': 0.05, 'ransomware_weight': 0.05, 'business_value_weight': 0.00, 'risk_to_cost_weight': 0.40}
        ]
        self.last_state = None
        self.last_action = None
        self.last_reward = 0.0

    def _load_q_table(self) -> Dict:
        if os.path.exists(self.q_table_file):
            try:
                with open(self.q_table_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading Q-table: {e}")
        return {}

    def _save_q_table(self):
        try:
            with open(self.q_table_file, 'wb') as f:
                pickle.dump(self.q_table, f)
        except Exception as e:
            print(f"Error saving Q-table: {e}")

    def initialize(self, state: State, cost_cache: Dict, defender_budget: float = None):
        super().initialize(state, cost_cache)
        if defender_budget is not None:
            self.defender_budget = defender_budget
        self.defender_budget_history = []
        self._last_budget_step = -1
        self._last_budget_value = None
        self._processed_steps = set()
        self._unpatched_vulns_cache = None
        self._current_weights = None
        self._attacker_actions = []
        self.state = state
        self.last_state = None
        self.last_action = None
        self.last_reward = 0.0
        self.total_patch_cost = 0.0
        self.total_patch_count = 0
        self.episode_count += 1

    def update_attacker_actions(self, actions: List[Dict]) -> None:
        self._attacker_actions = actions
        self._unpatched_vulns_cache = None

    def _get_unpatched_vulnerabilities(self, state: State, verbose: bool = False, force_refresh: bool = False) -> List[Tuple[Vulnerability, Any, Any]]:
        if not force_refresh and self._unpatched_vulns_cache is not None:
            return self._unpatched_vulns_cache.copy()

        if not verbose:
            import sys, io
            original_stdout = sys.stdout
            sys.stdout = io.StringIO()

        try:
            unpatched = []
            for asset in state.system.assets:
                for component in asset.components:
                    for vuln in component.vulnerabilities:
                        vuln_key = f"{vuln.cve_id}:{asset.asset_id}:{component.id}"
                        if not vuln.is_patched and vuln_key not in self._patched_vulns:
                            unpatched.append((vuln, asset, component))
        finally:
            if not verbose:
                sys.stdout = original_stdout

        if verbose:
            print(f"Found {len(unpatched)} unpatched vulnerabilities" if unpatched else "Warning: No unpatched vulnerabilities found")

        self._unpatched_vulns_cache = unpatched.copy()
        return unpatched

    def _calculate_adaptive_step_budget(self, state: State, remaining_budget: float, current_step: int, total_steps: int) -> float:
        if remaining_budget <= 0:
            return 0
        if total_steps <= 1:
            return remaining_budget

        if self._last_budget_step == current_step and self._last_budget_value is not None:
            return self._last_budget_value

        config = {
            'min_budget': 500.0,
            'max_budget': 5000.0,
            'risk_weight': 0.60,
            'attack_weight': 0.25,
            'compromise_weight': 0.10,
            'progress_weight': 0.05,
            'exploit_boost': 2.0,
            'compromise_threshold': 0.15,
            'criticality_threshold': 4
        }

        steps_remaining = total_steps - current_step
        unpatched_vulns = self._get_unpatched_vulnerabilities(state, verbose=False)

        risk_ratios = []
        for vuln, asset, comp in unpatched_vulns:
            try:
                vuln_key = f"{vuln.cve_id}:{asset.asset_id}:{comp.id}"
                risk_ratio = self._cost_cache.get('risk_to_cost_ratios', {}).get(
                    vuln_key, self.cost_calculator.calculate_risk_to_cost_ratio(vuln, state, asset, comp.id))
                risk_ratios.append(risk_ratio)
            except Exception as e:
                logger.error(f"Error calculating risk ratio for {vuln_key}: {e}")
                risk_ratios.append(0.1)

        total_risk = sum(risk_ratios) if risk_ratios else 0.0
        max_risk = max(risk_ratios + [1.0]) if risk_ratios else 1.0
        risk_score = total_risk / max_risk if max_risk > 0 else 0.1
        risk_budget = remaining_budget * config['risk_weight'] * risk_score

        attack_score = 0.0
        if self._attacker_actions:
            window_size = min(10, len(self._attacker_actions))
            recent_actions = self._attacker_actions[-window_size:] if window_size > 0 else []
            exploit_attempts = sum(1 for action in recent_actions if action.get('exploit_success', False))
            high_value_targeting = any(
                action.get('asset_id') and any(
                    asset.asset_id == action.get('asset_id') and
                    asset.criticality_level >= config['criticality_threshold']
                    for asset in state.system.assets
                ) for action in recent_actions
            )
            attack_score = (exploit_attempts / window_size if window_size > 0 else 0.0) * config['exploit_boost']
            if high_value_targeting:
                attack_score += 0.4
            attack_score = min(attack_score, 1.0)
        attack_budget = remaining_budget * config['attack_weight'] * attack_score

        compromised_assets = [asset for asset in state.system.assets if asset.is_compromised]
        compromise_rate = len(compromised_assets) / len(state.system.assets) if state.system.assets else 0.0
        compromise_score = 1.0 if compromise_rate > config['compromise_threshold'] else compromise_rate
        high_value_compromised = any(
            asset.criticality_level >= config['criticality_threshold'] for asset in compromised_assets
        )
        if high_value_compromised:
            compromise_score *= 1.8
        compromise_budget = remaining_budget * config['compromise_weight'] * compromise_score

        progress = current_step / total_steps if total_steps > 0 else 0.0
        progress_score = progress if progress < 0.7 else 1.0
        progress_budget = remaining_budget * config['progress_weight'] * progress_score

        step_budget = risk_budget + attack_budget + compromise_budget + progress_budget
        step_budget = max(config['min_budget'], min(step_budget, config['max_budget'], remaining_budget))

        logger.debug(f"Step {current_step}: Adaptive Budget Allocation: "
                    f"Risk=${risk_budget:.2f}, Attack=${attack_budget:.2f}, Compromise=${compromise_budget:.2f}, "
                    f"Progress=${progress_budget:.2f}, Total=${step_budget:.2f}")

        self._last_budget_step = current_step
        self._last_budget_value = step_budget
        self.defender_budget_history.append(step_budget)
        return step_budget

    def _get_state(self, state: State, remaining_budget: float, current_step: int) -> Tuple:
        compromised_assets = [asset for asset in state.system.assets if asset.is_compromised]
        compromise_rate = len(compromised_assets) / len(state.system.assets) if state.system.assets else 0.0
        compromise_bin = np.digitize(compromise_rate, self.state_bins['compromise_rate'], right=True) - 1

        unpatched_vulns = self._get_unpatched_vulnerabilities(state, verbose=False)
        critical_vulns = sum(1 for vuln, _, _ in unpatched_vulns if vuln.cvss >= 7.0)
        unpatched_bin = np.digitize(critical_vulns, self.state_bins['unpatched_critical'], right=True) - 1

        budget_fraction = remaining_budget / self.defender_budget if self.defender_budget > 0 else 0.0
        budget_bin = np.digitize(budget_fraction, self.state_bins['budget_remaining'], right=True) - 1

        return (compromise_bin, unpatched_bin, budget_bin)

    def _calculate_reward(self, state: State, patch_cost: float, patch_count: int) -> float:
        # Calculate total business value
        total_business_value = sum(
            getattr(asset, 'business_value', getattr(asset, 'criticality_level', 3) * 5000)
            for asset in state.system.assets
        )
        # Calculate lost value from exploited vulnerabilities
        lost_value = 0.0
        processed_vulns = set()
        for asset in state.system.assets:
            for component in asset.components:
                for vuln in component.vulnerabilities:
                    if vuln.is_exploited and vuln.cve_id not in processed_vulns:
                        lost_value += getattr(asset, 'business_value', 10000)
                        processed_vulns.add(vuln.cve_id)
        value_preserved = total_business_value - lost_value
        value_reward = value_preserved / total_business_value if total_business_value > 0 else 0.0
        roi = (value_preserved - patch_cost) / patch_cost if patch_cost > 0 else 0.0
        roi_reward = roi / 10.0  # normalize
        patch_reward = 1.0 - (patch_count / 50.0)  # prefer fewer patches
        # Count unpatched critical vulns
        unpatched_critical = sum(
            1 for asset in state.system.assets for comp in asset.components for vuln in comp.vulnerabilities
            if not vuln.is_patched and getattr(vuln, 'cvss', 0) >= 9.0
        )
        critical_penalty = unpatched_critical / 50.0  # normalize
        patch_cost_penalty = patch_cost / 10000.0
        value_loss_penalty = (total_business_value - value_preserved) / total_business_value if total_business_value > 0 else 0.0
        compromised_assets = sum(1 for asset in state.system.assets if asset.is_compromised)
        total_assets = len(state.system.assets)
        compromised_assets_penalty = compromised_assets / total_assets if total_assets > 0 else 0.0
        reward = (
            0.20 * value_reward
            + 0.20 * roi_reward
            + 0.10 * patch_reward
            - 0.50 * critical_penalty
            - 0.10 * patch_cost_penalty
            - 0.30 * value_loss_penalty
            - 0.20 * compromised_assets_penalty
        )
        print(f"[RL Reward Debug] value_reward={value_reward:.3f}, roi_reward={roi_reward:.3f}, patch_reward={patch_reward:.3f}, critical_penalty={critical_penalty:.3f}, patch_cost_penalty={patch_cost_penalty:.3f}, value_loss_penalty={value_loss_penalty:.3f}, compromised_assets_penalty={compromised_assets_penalty:.3f}, reward={reward:.3f}")
        return reward

    def _calculate_adaptive_weights(self, state: State, current_step: int, total_steps: int, silent_mode: bool) -> Dict[str, float]:
        current_state = self._get_state(state, sum(self.defender_budget_history) if self.defender_budget_history else self.defender_budget, current_step)

        state_key = str(current_state)
        if state_key not in self.q_table:
            self.q_table[state_key] = [0.0 for _ in range(len(self.weight_configs))]  # Zero initialization

        if random.random() < self.epsilon:
            action_idx = 0 if random.random() < 0.5 else random.randint(0, len(self.weight_configs) - 1)
        else:
            action_idx = np.argmax(self.q_table[state_key])

        weights = self.weight_configs[action_idx]

        if self.last_state is not None and self.last_action is not None:
            last_state_key = str(self.last_state)
            reward = self.last_reward

            next_max_q = max(self.q_table[state_key]) if state_key in self.q_table else 0.0

            self.q_table[last_state_key][self.last_action] += self.learning_rate * (
                reward + self.discount_factor * next_max_q - self.q_table[last_state_key][self.last_action]
            )

        self.last_state = current_state
        self.last_action = action_idx

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        if not silent_mode:
            print(f"RL Weights (Step {current_step}, Episode {self.episode_count}): {', '.join([f'{k}={v:.2f}' for k, v in weights.items()])}")
            print(f"State: {current_state}, Action: {action_idx}, Epsilon: {self.epsilon:.2f}")

        if current_step % 5 == 0:
            self._save_q_table()

        return weights

    def _adaptive_scoring_function(self, vuln_data, weights=None, config=None, silent_mode=False):
        w = weights or getattr(self, '_current_weights', {}) or {
            'cvss_weight': 0.20,
            'epss_weight': 0.20,
            'exploit_weight': 0.10,
            'ransomware_weight': 0.10,
            'business_value_weight': 0.00,
            'risk_to_cost_weight': 0.40
        }

        c = config or {
            'max_business_value': 10000,
            'exploit_boost_factor': 2.8,
            'action_window_size': 10,
        }

        vuln = vuln_data['vulnerability']
        asset = vuln_data['asset']
        component = vuln_data['component']
        vuln_key = vuln_data['vuln_key']

        likelihood_weights = {
            'cvss_weight': w['cvss_weight'],
            'epss_weight': w['epss_weight'],
            'exploit_weight': w['exploit_weight'],
            'ransomware_weight': w.get('ransomware_weight', 0.10)
        }

        likelihood = self.calculate_threat_intelligence_likelihood(vuln, likelihood_weights)
        normalized_cvss = vuln_data['cvss'] / 10.0
        impact = vuln_data['business_value'] * normalized_cvss

        risk_to_cost = self.cost_calculator.calculate_risk_to_cost_ratio(vuln, self.state, asset, component.id, self._cost_cache)

        score = (likelihood * impact / vuln_data['patch_cost'] if vuln_data['patch_cost'] > 0 else 0.0) * (
            1.0 - w['risk_to_cost_weight']) + risk_to_cost * w['risk_to_cost_weight']

        if self._attacker_actions:
            window_size = min(c['action_window_size'], len(self._attacker_actions))
            recent_actions = self._attacker_actions[-window_size:] if window_size > 0 else []
            for action in recent_actions:
                if action.get('vuln_id') == vuln_key and action.get('exploit_success', False):
                    if not silent_mode:
                        logger.debug(f"Vulnerability {vuln_key} recently exploited - boosting priority by {c['exploit_boost_factor']}x")
                    score *= c['exploit_boost_factor']

        vuln_data['_scoring_info'] = {
            'score': score,
            'top_factor': 'risk_to_cost_weight' if w['risk_to_cost_weight'] * risk_to_cost > max(
                likelihood_weights['cvss_weight'] * normalized_cvss,
                likelihood_weights['epss_weight'] * vuln_data['epss'],
                likelihood_weights['exploit_weight'] * (1.0 if vuln_data['has_exploit'] else 0.0),
                likelihood_weights['ransomware_weight'] * (1.0 if vuln_data['is_ransomware'] else 0.0)
            ) else 'cvss_weight',
            'normalized_cvss': normalized_cvss,
            'epss': vuln_data.get('epss', 0.0),
            'has_exploit': vuln_data['has_exploit'],
            'is_ransomware': vuln_data['is_ransomware'],
            'business_value': vuln_data['business_value'],
            'patch_cost': vuln_data['patch_cost'],
            'impact': impact,
            'risk': likelihood * impact,
            'risk_to_cost': risk_to_cost
        }

        if not silent_mode:
            logger.debug(f"Vuln {vuln_key}: Score={score:.2f}, Risk-to-Cost={risk_to_cost:.2f}, Top Factor={vuln_data['_scoring_info']['top_factor']}")
        return score

    def _prepare_vulnerability_data(self, state: State, step_budget: float) -> List[Dict]:
        unpatched_vulns = self._get_unpatched_vulnerabilities(state, verbose=False)

        vuln_data_list = []
        for vuln, asset, component in unpatched_vulns:
            vuln_key = f"{vuln.cve_id}:{asset.asset_id}:{component.id}"
            if vuln.is_patched or vuln_key in self._patched_vulns:
                continue

            vuln_info = self._cost_cache['vulnerability_info'].get(vuln_key, {})
            patch_cost = self._cost_cache['patch_costs'].get(vuln_key, 200.0)

            if patch_cost > step_budget:
                continue

            exploit_cost = self._cost_cache.get('exploit_costs', {}).get(vuln_key, 0.0)
            exploit_loss = self._cost_cache.get('exploit_losses', {}).get(vuln_key, 0.0)
            business_value = vuln_info.get('business_value',
                getattr(asset, 'business_value', getattr(asset, 'criticality_level', 3) * 5000))
            roi = self._cost_cache.get('roi', {}).get(vuln_key, 0.0)

            vuln_data = {
                'vulnerability': vuln,
                'asset': asset,
                'component': component,
                'cvss': vuln_info.get('cvss', getattr(vuln, 'cvss', 5.0)),
                'epss': vuln_info.get('epss', getattr(vuln, 'epss', 0.1)),
                'has_exploit': vuln_info.get('exploit', getattr(vuln, 'exploit', False)),
                'is_ransomware': vuln_info.get('ransomWare', getattr(vuln, 'ransomWare', False)),
                'business_value': business_value,
                'patch_cost': patch_cost,
                'exploit_cost': exploit_cost,
                'exploit_loss': exploit_loss,
                'roi': roi,
                'vuln_key': vuln_key,
                'asset_name': asset.name,
                'asset_id': asset.asset_id,
                'component_id': component.id
            }

            vuln_data_list.append(vuln_data)

        return vuln_data_list

    def select_patches(self, state: State, remaining_budget: float, current_step: int, total_steps: int) -> List[Tuple[Vulnerability, float]]:
        silent_mode = current_step in self._processed_steps

        step_budget = self._calculate_adaptive_step_budget(
            state=state,
            remaining_budget=remaining_budget,
            current_step=current_step,
            total_steps=total_steps
        )

        self._current_weights = self._calculate_adaptive_weights(state, current_step, total_steps, silent_mode)
        vuln_data_list = self._prepare_vulnerability_data(state, step_budget)

        if not vuln_data_list:
            print(f"No vulnerabilities to patch for step {current_step}")
            return []

        for vuln_data in vuln_data_list:
            score = self._adaptive_scoring_function(vuln_data, self._current_weights,
                                                   config={
                                                       'max_business_value': max(1000, sum(
                                                           getattr(asset, 'business_value', 0) for asset in state.system.assets)),
                                                       'exploit_boost_factor': 2.8,
                                                       'action_window_size': min(10, current_step),
                                                   },
                                                   silent_mode=silent_mode)
            vuln_data['score'] = score

        self._recently_scored_vulns = {vd['vuln_key']: vd for vd in vuln_data_list}
        vuln_data_list.sort(key=lambda x: x['score'], reverse=True)

        patch_list = []
        total_cost = 0
        step_patched_vulns = set()
        per_step_limit = 10  # Reduced but enforced

        if not silent_mode:
            print(f"\n{self.name} - Patching decisions for Step {current_step}:")
            print(f"{'Vuln Key':<30} {'Asset':<20} {'CVSS':<6} {'Score':<10} {'Cost':<10} {'Decision':<15}")
            print("-" * 80)

        for vuln_info in vuln_data_list:
            if vuln_info['vuln_key'] in step_patched_vulns or vuln_info['vuln_key'] in self._patched_vulns:
                continue

            patch_cost = vuln_info['patch_cost']
            decision = "SKIP (budget)"

            if total_cost + patch_cost <= step_budget and len(patch_list) < per_step_limit:
                patch_list.append((vuln_info['vulnerability'], patch_cost))
                total_cost += patch_cost
                step_patched_vulns.add(vuln_info['vuln_key'])
                decision = "PATCH"

            if not silent_mode:
                print(f"{vuln_info['vuln_key']:<30} {vuln_info['asset_name'][:20]:<20} {vuln_info['cvss']:<6} "
                      f"{vuln_info['score']:<10.2f} ${patch_cost:<9.2f} {decision:<15}")

        # Ensure at least one patch if budget allows
        if not patch_list and vuln_data_list and step_budget > min(vd['patch_cost'] for vd in vuln_data_list):
            top_vuln = vuln_data_list[0]
            patch_cost = top_vuln['patch_cost']
            if patch_cost <= step_budget:
                patch_list.append((top_vuln['vulnerability'], patch_cost))
                total_cost += patch_cost
                step_patched_vulns.add(top_vuln['vuln_key'])
                if not silent_mode:
                    print(f"Forced patch: {top_vuln['vuln_key']} (Cost: ${patch_cost:.2f})")

        self.total_patch_cost += total_cost
        self.total_patch_count += len(patch_list)
        self.last_reward = self._calculate_reward(state, total_cost, len(patch_list))

        print(f"\n{self.name} - Step {current_step} Summary:")
        print(f"  Vulnerabilities patched: {len(patch_list)}")
        print(f"  Total patch cost: ${total_cost:.2f}")
        print(f"  Remaining step budget: ${step_budget - total_cost:.2f}")
        print(f"  Remaining total budget: ${remaining_budget - total_cost:.2f}")
        print(f"  Reward: {self.last_reward:.2f}")
        print(f"  Strategy: {self.name}")
        print(f"  Total Patches (Episode): {self.total_patch_count}")
        print(f"  Total Patch Cost (Episode): ${self.total_patch_cost:.2f}")
        print("-" * 80)

        try:
            for vuln, cost in patch_list:
                matching_vulns = [vd for vd in vuln_data_list if vd['vulnerability'] == vuln]
                if matching_vulns:
                    vuln_key = matching_vulns[0]['vuln_key']
                    top_factor = self._recently_scored_vulns.get(vuln_key, {}).get('_scoring_info', {}).get(
                        'top_factor', 'cvss_weight')
                else:
                    vuln_key = f"{vuln.cve_id if hasattr(vuln, 'cve_id') else 'unknown'}:unknown:unknown"
                    top_factor = 'cvss_weight'
                    print(f"Warning: Could not find vulnerability data for patched vulnerability")

                self.patching_history.append({
                    'vuln_key': vuln_key,
                    'cost': cost,
                    'success': True,
                    'downtime': 1.0,
                    'top_factor': top_factor
                })

                self._patched_vulns.add(vuln_key)
        except Exception as e:
            print(f"Error updating patching history: {e}")

        self._processed_steps.add(current_step)
        self._unpatched_vulns_cache = None

        return patch_list
