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
        
        # IMPROVED RL PARAMETERS for stability
        self.learning_rate = 0.05      # Reduced for stability
        self.discount_factor = 0.90    # Slightly reduced for near-term focus
        self.epsilon = 0.3             # Higher initial exploration
        self.epsilon_min = 0.1         # Higher minimum for continued exploration
        self.epsilon_decay = 0.998     # Slower decay for better exploration
        
        # Add performance tracking
        self.recent_rewards = []
        self.recent_patch_counts = []
        self.performance_window = 20
        self.episode_count = 0
        
        # ENHANCED: Threat Intelligence Integration
        self.threat_intel_features = {
            'asset_threat_levels': {},
            'exploit_attempt_history': {},
            'technique_frequency': {},
            'compromise_sequence': [],
            'attack_observations': []
        }
        
        # ENHANCED: Improved state bins with threat intelligence features
        self.state_bins = {
            'compromise_rate': [0.0, 0.05, 0.15, 0.30, 1.0],
            'unpatched_critical': [0, 3, 8, 15, 50],  # More granular
            'budget_remaining': [0.0, 0.15, 0.35, 0.60, 1.0],
            'threat_level': [0.0, 0.3, 0.6, 0.8, 1.0],  # NEW: Threat intelligence level
            'attack_frequency': [0, 1, 3, 6, 10]  # NEW: Recent attack frequency
        }
        self.weight_configs = [
            # Configuration 0: Balanced defensive approach
            {
                'cvss_weight': 0.25, 
                'epss_weight': 0.25, 
                'exploit_weight': 0.15, 
                'ransomware_weight': 0.10, 
                'business_value_weight': 0.15, 
                'risk_to_cost_weight': 0.10
            },
            
            # Configuration 1: Cost-conscious approach
            {
                'cvss_weight': 0.20, 
                'epss_weight': 0.20, 
                'exploit_weight': 0.10, 
                'ransomware_weight': 0.05, 
                'business_value_weight': 0.10, 
                'risk_to_cost_weight': 0.35
            },
            
            # Configuration 2: High-value asset protection
            {
                'cvss_weight': 0.15, 
                'epss_weight': 0.15, 
                'exploit_weight': 0.15, 
                'ransomware_weight': 0.10, 
                'business_value_weight': 0.35, 
                'risk_to_cost_weight': 0.10
            },
            
            # Configuration 3: Threat-reactive approach
            {
                'cvss_weight': 0.30, 
                'epss_weight': 0.30, 
                'exploit_weight': 0.20, 
                'ransomware_weight': 0.15, 
                'business_value_weight': 0.05, 
                'risk_to_cost_weight': 0.00
            },
            
            # Configuration 4: Aggressive patching
            {
                'cvss_weight': 0.35, 
                'epss_weight': 0.25, 
                'exploit_weight': 0.25, 
                'ransomware_weight': 0.15, 
                'business_value_weight': 0.00, 
                'risk_to_cost_weight': 0.00
            }
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

    def initialize(self, state: State, cost_cache: Dict, defender_budget: float = 0.0):
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
        """
        Enhanced budget allocation with minimum spending guarantees
        """
        base_config = {
            'min_budget_ratio': 0.08,      # Minimum 8% per step
            'max_budget_ratio': 0.25,      # Maximum 25% per step
            'emergency_ratio': 0.40,       # Emergency 40% spending
            'compromise_threshold': 0.10,   # 10% compromise triggers emergency
            'critical_threshold': 3,        # Critical asset threshold
        }
        
        # Calculate base step budget (ensure consistent spending)
        steps_remaining = max(total_steps - current_step, 1)
        base_step_budget = remaining_budget / steps_remaining
        
        # Minimum budget guarantee (prevent under-spending)
        min_budget = remaining_budget * base_config['min_budget_ratio']
        max_budget = remaining_budget * base_config['max_budget_ratio']
        
        step_budget = max(min_budget, min(base_step_budget, max_budget))
        
        # EMERGENCY BUDGET TRIGGERS
        compromised_assets = [asset for asset in state.system.assets if asset.is_compromised]
        compromise_rate = len(compromised_assets) / len(state.system.assets) if state.system.assets else 0.0
        
        # Critical vulnerability emergency
        unpatched_critical = sum(
            1 for asset in state.system.assets 
            for comp in asset.components 
            for vuln in comp.vulnerabilities
            if not vuln.is_patched and getattr(vuln, 'cvss', 0) >= 9.0
        )
        
        # High-value asset compromise emergency
        high_value_compromised = any(
            getattr(asset, 'business_value', 0) > 8000 and asset.is_compromised 
            for asset in state.system.assets
        )
        
        # TRIGGER EMERGENCY SPENDING
        emergency_triggers = [
            compromise_rate > base_config['compromise_threshold'],
            unpatched_critical > 5,
            high_value_compromised,
            current_step > total_steps * 0.8 and remaining_budget > self.defender_budget * 0.3  # End-game spending
        ]
        
        if any(emergency_triggers):
            emergency_budget = remaining_budget * base_config['emergency_ratio']
            step_budget = min(emergency_budget, remaining_budget)
            print(f"EMERGENCY BUDGET ACTIVATED: ${step_budget:.2f} (Triggers: {emergency_triggers})")
        
        # PROACTIVE BUDGET BOOST (early prevention)
        elif compromise_rate == 0.0 and current_step < total_steps * 0.3:
            # Boost early spending for prevention
            step_budget *= 1.3
            step_budget = min(step_budget, max_budget, remaining_budget)
            print(f"PROACTIVE BUDGET BOOST: ${step_budget:.2f}")
        
        # Budget validation
        step_budget = max(min_budget, min(step_budget, remaining_budget))
        
        print(f"Step {current_step}: Budget=${step_budget:.2f}, Remaining=${remaining_budget:.2f}, "
              f"Compromise={compromise_rate:.1%}, Critical={unpatched_critical}")
        
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

        # ENHANCED: Threat Intelligence Features
        threat_level_bin = self._calculate_threat_level_bin(state)
        attack_frequency_bin = self._calculate_attack_frequency_bin()

        return (compromise_bin, unpatched_bin, budget_bin, threat_level_bin, attack_frequency_bin)

    def _calculate_threat_level_bin(self, state: State) -> int:
        """Calculate threat level bin based on asset threat levels and attack patterns."""
        if not self.threat_intel_features['asset_threat_levels']:
            return 0  # No threat intelligence data available
        
        # Calculate average threat level across all assets
        threat_levels = list(self.threat_intel_features['asset_threat_levels'].values())
        avg_threat_level = np.mean(threat_levels) if threat_levels else 0.0
        
        # Bin the threat level
        threat_bin = np.digitize(avg_threat_level, self.state_bins['threat_level'], right=True) - 1
        return int(max(0, min(len(self.state_bins['threat_level']) - 1, threat_bin)))

    def _calculate_attack_frequency_bin(self) -> int:
        """Calculate attack frequency bin based on recent attack observations."""
        recent_attacks = len(self.threat_intel_features['attack_observations'])
        
        # Bin the attack frequency
        frequency_bin = np.digitize(recent_attacks, self.state_bins['attack_frequency'], right=True) - 1
        return int(max(0, min(len(self.state_bins['attack_frequency']) - 1, frequency_bin)))

    def update_threat_intelligence_features(self, threat_intel_strategy):
        """Update threat intelligence features from a ThreatIntelligenceStrategy instance."""
        if hasattr(threat_intel_strategy, 'asset_threat_levels'):
            self.threat_intel_features['asset_threat_levels'] = threat_intel_strategy.asset_threat_levels.copy()
        
        if hasattr(threat_intel_strategy, 'exploit_attempt_history'):
            self.threat_intel_features['exploit_attempt_history'] = threat_intel_strategy.exploit_attempt_history.copy()
        
        if hasattr(threat_intel_strategy, 'technique_frequency'):
            self.threat_intel_features['technique_frequency'] = threat_intel_strategy.technique_frequency.copy()
        
        if hasattr(threat_intel_strategy, 'compromise_sequence'):
            self.threat_intel_features['compromise_sequence'] = threat_intel_strategy.compromise_sequence.copy()
        
        if hasattr(threat_intel_strategy, 'attack_observations'):
            self.threat_intel_features['attack_observations'] = threat_intel_strategy.attack_observations.copy()

    def _calculate_threat_intel_priority(self, asset_id: str, vuln_key: str) -> float:
        """Calculate threat intelligence priority score for a vulnerability."""
        priority = 0.0
        
        # Asset threat level contribution
        asset_threat = self.threat_intel_features['asset_threat_levels'].get(asset_id, 0.3)
        priority += asset_threat * 0.4  # 40% weight for asset threat level
        
        # Exploit attempt history contribution
        exploit_data = self.threat_intel_features['exploit_attempt_history'].get(vuln_key, {'attempts': 0, 'successes': 0})
        if exploit_data['attempts'] > 0:
            success_rate = exploit_data['successes'] / exploit_data['attempts']
            priority += success_rate * 0.3  # 30% weight for exploit success rate
            priority += min(exploit_data['attempts'] / 5.0, 1.0) * 0.2  # 20% weight for attempt frequency
        
        # Compromise sequence contribution
        if asset_id in self.threat_intel_features['compromise_sequence']:
            priority += 0.1  # 10% bonus for assets in compromise sequence
        
        return min(priority, 1.0)  # Cap at 1.0

    def _calculate_reward(self, state: State, patch_cost: float, patch_count: int) -> float:
        """
        Stabilized reward function with reduced volatility and better balance
        """
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
        
        # STABILIZED REWARD COMPONENTS with capping and normalization
        
        # 1. Value preservation reward (normalized and capped)
        value_reward = min(value_preserved / total_business_value, 1.0) if total_business_value > 0 else 0.0
        
        # 2. ROI reward (capped to prevent extreme values)
        roi = (value_preserved - patch_cost) / max(patch_cost, 1.0) if patch_cost > 0 else 0.0
        roi_reward = max(-1.0, min(1.0, roi / 5.0))  # Cap ROI between -1 and 1
        
        # 3. Patch efficiency reward (encourage moderate patching)
        optimal_patches = 3  # Target 3 patches per episode
        patch_efficiency = 1.0 - abs(patch_count - optimal_patches) / optimal_patches
        patch_reward = max(0.0, patch_efficiency)
        
        # 4. Critical vulnerability penalty (normalized)
        unpatched_critical = sum(
            1 for asset in state.system.assets 
            for comp in asset.components 
            for vuln in comp.vulnerabilities
            if not vuln.is_patched and getattr(vuln, 'cvss', 0) >= 9.0
        )
        total_critical = sum(
            1 for asset in state.system.assets 
            for comp in asset.components 
            for vuln in comp.vulnerabilities
            if getattr(vuln, 'cvss', 0) >= 9.0
        )
        critical_penalty = (unpatched_critical / max(total_critical, 1)) if total_critical > 0 else 0.0
        
        # 5. Cost efficiency penalty (normalized)
        max_reasonable_cost = total_business_value * 0.1  # 10% of business value
        cost_penalty = min(patch_cost / max_reasonable_cost, 1.0) if max_reasonable_cost > 0 else 0.0
        
        # 6. Compromise penalty (normalized)
        compromised_assets = sum(1 for asset in state.system.assets if asset.is_compromised)
        total_assets = len(state.system.assets)
        compromise_penalty = compromised_assets / total_assets if total_assets > 0 else 0.0
        
        # 7. Proactive defense bonus (reward early patching)
        proactive_bonus = 0.1 if patch_count > 0 and compromise_penalty < 0.2 else 0.0
        
        # 8. ENHANCED: Threat Intelligence Awareness Bonus
        threat_intel_bonus = 0.0
        if self.threat_intel_features['asset_threat_levels']:
            # Reward for patching high-threat assets
            high_threat_assets = sum(1 for level in self.threat_intel_features['asset_threat_levels'].values() if level > 0.7)
            if high_threat_assets > 0:
                threat_intel_bonus = 0.05  # Small bonus for threat intelligence awareness
        
        # BALANCED REWARD CALCULATION (all components normalized to [-1, 1])
        reward = (
            0.30 * value_reward          # Increased: Main objective
            + 0.20 * roi_reward          # Balanced: Cost effectiveness  
            + 0.15 * patch_reward        # Increased: Encourage activity
            + 0.10 * proactive_bonus     # New: Reward proactive defense
            + 0.05 * threat_intel_bonus  # ENHANCED: Threat intelligence awareness
            - 0.25 * critical_penalty    # Reduced: Less harsh
            - 0.15 * cost_penalty        # Increased: Cost awareness
            - 0.20 * compromise_penalty  # Reduced: Less harsh
        )
        
        # Final reward capping to prevent extreme values
        reward = max(-2.0, min(2.0, reward))
        
        # Enhanced debug logging
        print(f"[Stabilized Reward] value={value_reward:.3f}, roi={roi_reward:.3f}, "
              f"patch_eff={patch_reward:.3f}, proactive={proactive_bonus:.3f}, "
              f"critical_pen={critical_penalty:.3f}, cost_pen={cost_penalty:.3f}, "
              f"compromise_pen={compromise_penalty:.3f}, final={reward:.3f}")
        
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

        # Update learning parameters
        if hasattr(self, 'last_reward'):
            patch_count = len(getattr(self, '_last_patch_list', []))
            self._update_learning_parameters(self.last_reward, patch_count)

        if not silent_mode:
            print(f"RL Weights (Step {current_step}, Episode {getattr(self, 'episode_count', 0)}): {', '.join([f'{k}={v:.2f}' for k, v in weights.items()])}")
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
            'zero_day_multiplier': 5.0  # NEW: Massive boost for zero-days
        }

        vuln = vuln_data['vulnerability']
        asset = vuln_data['asset']
        component = vuln_data['component']
        vuln_key = vuln_data['vuln_key']
        is_zero_day = vuln_data.get('is_zero_day', False)

        likelihood_weights = {
            'cvss_weight': w['cvss_weight'],
            'epss_weight': w['epss_weight'],
            'exploit_weight': w['exploit_weight'],
            'ransomware_weight': w.get('ransomware_weight', 0.10)
        }

        likelihood = self.calculate_threat_intelligence_likelihood(vuln, likelihood_weights)
        normalized_cvss = vuln_data['cvss'] / 10.0
        impact = vuln_data['business_value'] * normalized_cvss

        current_state = self.state
        risk_to_cost = self.cost_calculator.calculate_risk_to_cost_ratio(vuln, current_state.system, asset, component.id)

        # ENHANCED: Threat Intelligence Integration in Scoring
        threat_intel_boost = 1.0
        
        # Apply threat intelligence priority boost
        threat_intel_priority = vuln_data.get('threat_intel_priority', 0.0)
        if threat_intel_priority > 0.5:  # High threat intelligence priority
            threat_intel_boost = 1.0 + threat_intel_priority * 0.5  # Up to 1.5x boost
            if not silent_mode:
                print(f"🎯 THREAT INTEL BOOST: {vuln_key} priority={threat_intel_priority:.2f}, boost={threat_intel_boost:.2f}x")
        
        # Apply asset threat level boost
        asset_threat_level = vuln_data.get('asset_threat_level', 0.3)
        if asset_threat_level > 0.7:  # High threat asset
            asset_boost = 1.0 + asset_threat_level * 0.3  # Up to 1.3x boost
            threat_intel_boost *= asset_boost
            if not silent_mode:
                print(f"🔥 HIGH THREAT ASSET: {vuln_key} threat_level={asset_threat_level:.2f}, boost={asset_boost:.2f}x")
        
        # Apply exploit attempt boost
        exploit_attempts = vuln_data.get('exploit_attempts', 0)
        if exploit_attempts > 0:
            exploit_boost = 1.0 + min(exploit_attempts / 3.0, 1.0) * 0.4  # Up to 1.4x boost
            threat_intel_boost *= exploit_boost
            if not silent_mode:
                print(f"⚡ EXPLOIT ATTEMPT BOOST: {vuln_key} attempts={exploit_attempts}, boost={exploit_boost:.2f}x")
        
        # Base score calculation
        base_score = (likelihood * impact / vuln_data['patch_cost'] if vuln_data['patch_cost'] > 0 else 0.0) * (
            1.0 - w['risk_to_cost_weight']) + risk_to_cost * w['risk_to_cost_weight']

        # Apply threat intelligence boost
        base_score *= threat_intel_boost

        # CRITICAL: Apply massive boost for zero-day vulnerabilities
        if is_zero_day:
            base_score *= c['zero_day_multiplier']
            if not silent_mode:
                print(f"🚨 ZERO-DAY BOOST: {vuln_key} score boosted by {c['zero_day_multiplier']}x")

        # Apply attacker action boost
        if self._attacker_actions:
            window_size = min(c['action_window_size'], len(self._attacker_actions))
            recent_actions = self._attacker_actions[-window_size:] if window_size > 0 else []
            for action in recent_actions:
                if action.get('vuln_id') == vuln_key and action.get('exploit_success', False):
                    if not silent_mode:
                        logger.debug(f"Vulnerability {vuln_key} recently exploited - boosting priority by {c['exploit_boost_factor']}x")
                    base_score *= c['exploit_boost_factor']

        vuln_data['_scoring_info'] = {
            'score': base_score,
            'is_zero_day': is_zero_day,
            'zero_day_boost_applied': is_zero_day,
            'top_factor': 'zero_day' if is_zero_day else ('risk_to_cost_weight' if w['risk_to_cost_weight'] * risk_to_cost > max(
                likelihood_weights['cvss_weight'] * normalized_cvss,
                likelihood_weights['epss_weight'] * vuln_data['epss'],
                likelihood_weights['exploit_weight'] * (1.0 if vuln_data['has_exploit'] else 0.0),
                likelihood_weights['ransomware_weight'] * (1.0 if vuln_data['is_ransomware'] else 0.0)
            ) else 'cvss_weight'),
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
            zero_day_info = " [ZERO-DAY]" if is_zero_day else ""
            logger.debug(f"Vuln {vuln_key}{zero_day_info}: Score={base_score:.2f}, Risk-to-Cost={risk_to_cost:.2f}")
        
        return base_score

    def _prepare_vulnerability_data(self, state: State, step_budget: float) -> List[Dict]:
        unpatched_vulns = self._get_unpatched_vulnerabilities(state, verbose=False)
        zero_days_found = 0  # Track zero-day discoveries

        vuln_data_list = []
        for vuln, asset, component in unpatched_vulns:
            vuln_key = f"{vuln.cve_id}:{asset.asset_id}:{component.id}"
            if vuln.is_patched or vuln_key in self._patched_vulns:
                continue

            # CRITICAL: Check for zero-day vulnerabilities and prioritize them
            is_zero_day = self.is_zero_day_vulnerability(vuln)
            if is_zero_day:
                zero_days_found += 1
                print(f"🚨 ZERO-DAY DETECTED: {vuln_key} - HIGH PRIORITY!")

            vuln_info = self._cost_cache['vulnerability_info'].get(vuln_key, {})
            patch_cost = self._cost_cache['patch_costs'].get(vuln_key, 200.0)

            # MODIFIED: Allow zero-days even if they exceed step budget (emergency)
            if patch_cost > step_budget and not is_zero_day:
                continue

            exploit_cost = self._cost_cache.get('exploit_costs', {}).get(vuln_key, 0.0)
            exploit_loss = self._cost_cache.get('exploit_losses', {}).get(vuln_key, 0.0)
            business_value = vuln_info.get('business_value',
                getattr(asset, 'business_value', getattr(asset, 'criticality_level', 3) * 5000))
            roi = self._cost_cache.get('roi', {}).get(vuln_key, 0.0)

            # ENHANCED: Add threat intelligence features
            asset_id = str(asset.asset_id)
            asset_threat_level = self.threat_intel_features['asset_threat_levels'].get(asset_id, 0.3)
            exploit_attempts = self.threat_intel_features['exploit_attempt_history'].get(vuln_key, {'attempts': 0, 'successes': 0})
            is_in_compromise_sequence = asset_id in self.threat_intel_features['compromise_sequence']
            
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
                'component_id': component.id,
                'is_zero_day': is_zero_day,  # Add zero-day flag
                # ENHANCED: Threat Intelligence Features
                'asset_threat_level': asset_threat_level,
                'exploit_attempts': exploit_attempts['attempts'],
                'exploit_successes': exploit_attempts['successes'],
                'is_in_compromise_sequence': is_in_compromise_sequence,
                'threat_intel_priority': self._calculate_threat_intel_priority(asset_id, vuln_key)
            }

            vuln_data_list.append(vuln_data)

        if zero_days_found > 0:
            print(f"📊 Found {zero_days_found} zero-day vulnerabilities that RL can now see!")

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

        # Score all vulnerabilities
        for vuln_data in vuln_data_list:
            score = self._adaptive_scoring_function(vuln_data, self._current_weights,
                                                   config={
                                                       'max_business_value': max(1000, sum(
                                                           getattr(asset, 'business_value', 0) for asset in state.system.assets)),
                                                       'exploit_boost_factor': 2.8,
                                                       'action_window_size': min(10, current_step),
                                                       'zero_day_multiplier': 5.0  # Massive boost for zero-days
                                                   },
                                                   silent_mode=silent_mode)
            vuln_data['score'] = score

        self._recently_scored_vulns = {vd['vuln_key']: vd for vd in vuln_data_list}
        
        # CRITICAL: Separate zero-days for priority handling
        zero_day_vulns = [vd for vd in vuln_data_list if vd.get('is_zero_day', False)]
        regular_vulns = [vd for vd in vuln_data_list if not vd.get('is_zero_day', False)]
        
        # Sort both lists by score
        zero_day_vulns.sort(key=lambda x: x['score'], reverse=True)
        regular_vulns.sort(key=lambda x: x['score'], reverse=True)
        
        # Prioritize zero-days first, then regular vulns
        prioritized_vulns = zero_day_vulns + regular_vulns

        patch_list = []
        total_cost = 0
        step_patched_vulns = set()
        per_step_limit = 10

        if not silent_mode:
            print(f"\n{self.name} - Patching decisions for Step {current_step}:")
            print(f"Zero-day vulnerabilities found: {len(zero_day_vulns)}")
            print(f"{'Vuln Key':<30} {'Asset':<20} {'CVSS':<6} {'Score':<10} {'Cost':<10} {'Type':<10} {'Decision':<15}")
            print("-" * 95)

        # MANDATORY: Patch zero-days first (allow budget overflow for emergencies)
        emergency_budget_used = 0.0
        for vuln_info in zero_day_vulns:
            if vuln_info['vuln_key'] in step_patched_vulns or vuln_info['vuln_key'] in self._patched_vulns:
                continue

            patch_cost = vuln_info['patch_cost']
            # Allow emergency spending for zero-days up to 2x step budget
            if total_cost + patch_cost <= step_budget * 2.0:
                patch_list.append((vuln_info['vulnerability'], patch_cost))
                total_cost += patch_cost
                if total_cost > step_budget:
                    emergency_budget_used += (total_cost - step_budget)
                step_patched_vulns.add(vuln_info['vuln_key'])
                decision = "EMERGENCY PATCH"
                vuln_type = "ZERO-DAY"
                
                if not silent_mode:
                    print(f"{vuln_info['vuln_key']:<30} {vuln_info['asset_name'][:20]:<20} {vuln_info['cvss']:<6} "
                          f"{vuln_info['score']:<10.2f} ${patch_cost:<9.2f} {vuln_type:<10} {decision:<15}")

        # Regular vulnerability patching
        remaining_budget_for_regular = max(0, step_budget - total_cost + emergency_budget_used)
        regular_patches_added = 0
        
        for vuln_info in regular_vulns:
            if vuln_info['vuln_key'] in step_patched_vulns or vuln_info['vuln_key'] in self._patched_vulns:
                continue
            
            if regular_patches_added >= per_step_limit:
                break

            patch_cost = vuln_info['patch_cost']
            decision = "SKIP (budget)"
            vuln_type = "REGULAR"

            if total_cost + patch_cost <= step_budget:
                patch_list.append((vuln_info['vulnerability'], patch_cost))
                total_cost += patch_cost
                step_patched_vulns.add(vuln_info['vuln_key'])
                regular_patches_added += 1
                decision = "PATCH"

            if not silent_mode:
                print(f"{vuln_info['vuln_key']:<30} {vuln_info['asset_name'][:20]:<20} {vuln_info['cvss']:<6} "
                      f"{vuln_info['score']:<10.2f} ${patch_cost:<9.2f} {vuln_type:<10} {decision:<15}")

        # Minimum activity enforcement (only if no patches at all)
        if not patch_list and prioritized_vulns and step_budget > min(vd['patch_cost'] for vd in prioritized_vulns):
            top_vuln = prioritized_vulns[0]
            patch_cost = top_vuln['patch_cost']
            if patch_cost <= step_budget:
                patch_list.append((top_vuln['vulnerability'], patch_cost))
                total_cost += patch_cost
                step_patched_vulns.add(top_vuln['vuln_key'])
                if not silent_mode:
                    print(f"MINIMUM ACTIVITY: {top_vuln['vuln_key']} (Cost: ${patch_cost:.2f})")

        self.total_patch_cost += total_cost
        self.total_patch_count += len(patch_list)
        self.last_reward = self._calculate_reward(state, total_cost, len(patch_list))
        self._last_patch_list = patch_list

        print(f"\n{self.name} - Step {current_step} Summary:")
        print(f"  Vulnerabilities patched: {len(patch_list)} (Zero-days: {len(zero_day_vulns)} found)")
        print(f"  Total patch cost: ${total_cost:.2f}")
        if emergency_budget_used > 0:
            print(f"  Emergency budget used: ${emergency_budget_used:.2f}")
        print(f"  Remaining step budget: ${step_budget - total_cost:.2f}")
        print(f"  Remaining total budget: ${remaining_budget - total_cost:.2f}")
        print(f"  Reward: {self.last_reward:.2f}")
        print(f"  Strategy: {self.name}")
        print(f"  Total Patches (Episode): {self.total_patch_count}")
        print(f"  Total Patch Cost (Episode): ${self.total_patch_cost:.2f}")
        print("-" * 80)

        # Update patching history
        try:
            for vuln, cost in patch_list:
                matching_vulns = [vd for vd in vuln_data_list if vd['vulnerability'] == vuln]
                if matching_vulns:
                    vuln_key = matching_vulns[0]['vuln_key']
                    is_zero_day = matching_vulns[0].get('is_zero_day', False)
                    top_factor = 'zero_day' if is_zero_day else self._recently_scored_vulns.get(vuln_key, {}).get('_scoring_info', {}).get('top_factor', 'cvss_weight')
                else:
                    vuln_key = f"{vuln.cve_id if hasattr(vuln, 'cve_id') else 'unknown'}:unknown:unknown"
                    top_factor = 'cvss_weight'
                    print(f"Warning: Could not find vulnerability data for patched vulnerability")

                self.patching_history.append({
                    'vuln_key': vuln_key,
                    'cost': cost,
                    'success': True,
                    'downtime': 1.0,
                    'top_factor': top_factor,
                    'is_zero_day': is_zero_day if 'is_zero_day' in locals() else False
                })

                self._patched_vulns.add(vuln_key)
        except Exception as e:
            print(f"Error updating patching history: {e}")

        self._processed_steps.add(current_step)
        self._unpatched_vulns_cache = None

        return patch_list

    def _enhance_scoring_with_attack_prediction(self, vuln_data, score):
        asset = vuln_data['asset']
        is_adjacent_to_compromise = self._is_adjacent_to_compromised_asset(asset)
        is_high_value = getattr(asset, 'business_value', 0) > 7000
        enables_lateral_movement = vuln_data.get('has_exploit', False) and vuln_data.get('cvss', 0) >= 7.0
        boost_factor = 1.0
        if is_adjacent_to_compromise:
            boost_factor *= 1.5
        if is_high_value:
            boost_factor *= 1.3
        if enables_lateral_movement:
            boost_factor *= 1.4
        return score * boost_factor

    def _adjust_learning_parameters(self):
        if self.recent_rewards is not None and len(self.recent_rewards) >= 10:
            avg_recent_reward = sum(self.recent_rewards[-10:]) / 10
            if avg_recent_reward < -0.5:
                self.learning_rate = min(0.15, self.learning_rate * 1.1)
                self.epsilon = min(0.3, self.epsilon * 1.1)
            elif avg_recent_reward > 0.2:
                self.learning_rate = max(0.05, self.learning_rate * 0.95)
                self.epsilon = max(self.epsilon_min, self.epsilon * 0.98)

    def _is_adjacent_to_compromised_asset(self, asset):
        # Dummy implementation: always return False unless you have attack graph adjacency logic
        # Replace with real logic if available
        return False

    def _update_learning_parameters(self, reward, patch_count):
        """
        Adaptively adjust learning parameters based on performance
        """
        self.recent_rewards.append(reward)
        self.recent_patch_counts.append(patch_count)
        
        # Keep only recent performance
        if len(self.recent_rewards) > self.performance_window:
            self.recent_rewards.pop(0)
            self.recent_patch_counts.pop(0)
        
        if len(self.recent_rewards) >= 10:
            avg_reward = sum(self.recent_rewards[-10:]) / 10
            avg_patches = sum(self.recent_patch_counts[-10:]) / 10
            reward_volatility = np.std(self.recent_rewards[-10:])
            
            # Adjust learning rate based on stability
            if reward_volatility > 1.0:  # High volatility
                self.learning_rate = max(0.02, self.learning_rate * 0.95)
                print(f"High volatility detected, reducing learning rate to {self.learning_rate:.3f}")
            
            # Adjust exploration based on performance
            if avg_reward < 0 and avg_patches < 1:  # Poor performance
                self.epsilon = min(0.4, self.epsilon * 1.05)
                print(f"Poor performance, increasing exploration to {self.epsilon:.3f}")
            elif avg_reward > 0.5 and reward_volatility < 0.5:  # Good stable performance
                self.epsilon = max(self.epsilon_min, self.epsilon * 0.99)

    def can_recognize_zero_day(self) -> bool:
        """RL Adaptive Threat Intelligence strategy can recognize zero-day vulnerabilities."""
        return True
