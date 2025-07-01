"""
This module contains different vulnerability patching strategies used in the cybersecurity
simulation. Each strategy implements a different approach to prioritizing vulnerabilities
for patching based on various factors such as CVSS score, exploit presence, business value,
cost efficiency, and threat intelligence.

Modified to include zero-day vulnerability recognition capability for Threat Intelligence strategy only.
"""

import random
from collections import Counter
import numpy as np
from typing import List, Dict, Tuple, Any, Set, Optional
import copy
import sys
import io

from classes.state import State, Vulnerability
from classes.cost import CostCalculator


class PatchingStrategy:
    """
    Base class for vulnerability patching strategies.
    Provides common functionality for all patching strategies.
    """

    def __init__(self, name: str):
        """
        Initialize the patching strategy.

        Args:
            name: Name of the strategy
        """
        self.name = name
        self.cost_calculator = CostCalculator()
        self.patching_history = []
        self._patched_vulns = set()  # Changed to store composite keys (CVE-ID:asset_id:component_id)
        self._cost_cache = {}
        self._recently_scored_vulns = {}

    def initialize(self, state: State, cost_cache: Dict):
        """
        Initialize the strategy with the current state and cost cache.

        Args:
            state: Current simulation state
            cost_cache: Pre-computed cost cache with composite keys
        """
        self._cost_cache = cost_cache
        self.state = state

    def is_zero_day_vulnerability(self, vuln) -> bool:
        """
        Check if a vulnerability is a zero-day vulnerability.
        By default, only CVE IDs starting with "ZERO-DAY" are considered zero-day.

        Args:
            vuln: Vulnerability object

        Returns:
            bool: True if zero-day, False otherwise
        """
        return getattr(vuln, 'cve_id', '').startswith('ZERO-DAY')

    def can_recognize_zero_day(self) -> bool:
        """
        Determine if this strategy can recognize zero-day vulnerabilities.
        By default, only Threat Intelligence strategy can recognize zero-days.

        Returns:
            bool: True if strategy can recognize zero-days
        """
        return False

    def calculate_patch_priority_list(self, state, max_budget=None, per_step_limit=2):
        """
        Calculate an optimized vulnerability patching priority list based on
        risk-to-cost ratio and budget constraints. Includes a per-step limit
        to spread patching across multiple time steps.

        Args:
            state: Current system state
            max_budget: Maximum patching budget in dollars (defaults to defender's budget)
            per_step_limit: Maximum number of vulnerabilities to patch in one step (default: 2)

        Returns:
            list: Prioritized list of vulnerabilities to patch in this step
        """
        if max_budget is None:
            max_budget = 500.0  # Default fallback budget per step

        per_step_budget = max_budget
        unpatched_vulns = self._get_unpatched_vulnerabilities(state, verbose=False)

        if not unpatched_vulns:
            return []

        vuln_metrics = []
        for vuln, asset, component in unpatched_vulns:
            try:
                vuln_key = f"{vuln.cve_id}:{asset.asset_id}:{component.id}"
                if vuln_key in self._patched_vulns:
                    continue

                # Skip zero-day vulnerabilities if strategy cannot recognize them
                if self.is_zero_day_vulnerability(vuln) and not self.can_recognize_zero_day():
                    continue

                patch_cost = self.cost_calculator.calculate_patch_cost(vuln, state, asset, component_id=component.id)
                ratio = self.cost_calculator.calculate_risk_to_cost_ratio(vuln, state, asset, component_id=component.id)

                vuln_metrics.append({
                    'vulnerability': vuln,
                    'asset': asset,
                    'component': component,
                    'patch_cost': patch_cost,
                    'ratio': ratio,
                    'vuln_key': vuln_key,
                    'cvss': getattr(vuln, 'cvss', 5.0)
                })
            except Exception as e:
                print(
                    f"Error calculating metrics for {vuln.cve_id} (Asset {asset.asset_id}, Component {component.id}): {e}")
                # Skip zero-day vulnerabilities in error handling if strategy cannot recognize them
                if self.is_zero_day_vulnerability(vuln) and not self.can_recognize_zero_day():
                    continue
                vuln_metrics.append({
                    'vulnerability': vuln,
                    'asset': asset,
                    'component': component,
                    'patch_cost': 220.0,
                    'ratio': 0.1,
                    'vuln_key': f"{vuln.cve_id}:{asset.asset_id}:{component.id}",
                    'cvss': getattr(vuln, 'cvss', 5.0)
                })

        sorted_vulns = sorted(vuln_metrics, key=lambda x: x['ratio'], reverse=True)
        patch_list = []
        total_cost = 0.0
        vulns_selected = 0

        for vuln_info in sorted_vulns:
            if vulns_selected >= per_step_limit:
                break
            if total_cost + vuln_info['patch_cost'] > per_step_budget:
                continue

            patch_list.append(vuln_info['vulnerability'])
            total_cost += vuln_info['patch_cost']
            vulns_selected += 1
            self._patched_vulns.add(vuln_info['vuln_key'])

        print(f"Selected {len(patch_list)} vulnerabilities to patch with total cost ${total_cost:.2f}")
        return patch_list

    def _calculate_decreasing_step_budget(self, total_budget: float, current_step: int,
                                          total_steps: int, remaining_budget: float) -> float:
        """
        Calculate a decreasing budget allocation with improved dynamics to ensure
        budget is available throughout all steps.

        Args:
            total_budget: Total budget for the entire patching campaign
            current_step: Current simulation step (0-indexed)
            total_steps: Total number of simulation steps
            remaining_budget: Current remaining budget

        Returns:
            float: Budget allocated for this step
        """
        if remaining_budget <= 0:
            return 0
        if total_steps <= 1:
            return remaining_budget

        steps_remaining = total_steps - current_step
        if steps_remaining == 1:
            fraction = 1.0
        else:
            fraction = max(0.1, 1.5 / max(1, steps_remaining))

        base_budget = remaining_budget * fraction
        unpatched_vulns = self._get_unpatched_vulnerabilities(self.state, verbose=False)
        patch_costs = [
            self._cost_cache['patch_costs'].get(f"{vuln.cve_id}:{asset.asset_id}:{comp.id}", 200.0)
            for vuln, asset, comp in unpatched_vulns
            if not (self.is_zero_day_vulnerability(vuln) and not self.can_recognize_zero_day())
        ]

        min_budget = min(patch_costs) if patch_costs else 100.0
        median_cost = sorted(patch_costs)[len(patch_costs) // 2] if patch_costs else 200.0
        max_budget = median_cost * 3.5 if patch_costs else 1200.0

        step_budget = max(min_budget, min(base_budget, max_budget, remaining_budget))
        print(f"Step {current_step}: Steps remaining={steps_remaining}, "
              f"Fraction={fraction:.2f}, Step Budget=${step_budget:.2f}, "
              f"Min Budget=${min_budget:.2f}, Max Budget=${max_budget:.2f}, "
              f"Remaining Budget=${remaining_budget:.2f}")
        return step_budget

    def calculate_cvss_exploit_likelihood(self, vulnerability):
        """
        Calculate exploit likelihood based on EPSS score.
        Args:
            vulnerability: Vulnerability object with epss attribute
        Returns:
            float: Exploit likelihood in [0, 1]
        """
        epss = float(getattr(vulnerability, 'epss', 0.01))
        return min(max(epss, 0.0), 1.0)

    def calculate_threat_intelligence_likelihood(self, vuln, weights: Optional[Dict[str, float]] = None):
        """
        Calculate exploit likelihood based on CVSS, EPSS, exploit presence, and ransomware association.
        Uses calculate_cvss_exploit_likelihood() for EPSS contribution.
        Args:
            vuln: Vulnerability object with cvss, epss, exploit, and ransomWare attributes
            weights: Optional dictionary with keys 'cvss_weight', 'epss_weight', 'exploit_weight', 'ransomware_weight'
        Returns:
            float: Likelihood in [0, 1]
        """
        default_weights = {
            'cvss_weight': 0.4,
            'epss_weight': 0.4,
            'exploit_weight': 0.1,
            'ransomware_weight': 0.1
        }
        weights = weights or default_weights

        normalized_cvss = getattr(vuln, 'cvss', 5.0) / 10.0
        cvss_contribution = weights['cvss_weight'] * normalized_cvss
        epss_contribution = weights['epss_weight'] * self.calculate_cvss_exploit_likelihood(vuln)
        has_exploit = getattr(vuln, 'exploit', False)
        is_ransomware = getattr(vuln, 'ransomWare', False)
        exploit_contribution = weights['exploit_weight'] * (1.0 if has_exploit else 0.0)
        ransomware_contribution = weights['ransomware_weight'] * (1.0 if is_ransomware else 0.0)

        likelihood = cvss_contribution + epss_contribution + exploit_contribution + ransomware_contribution
        return max(0.0, min(1.0, likelihood))

    def _get_unpatched_vulnerabilities(self, state: State, verbose: bool = False) -> List[
        Tuple[Vulnerability, Any, Any]]:
        """
        Get list of unpatched vulnerabilities from the state.
        Returns (vulnerability, asset, component) tuples to include component_id.
        Filters out zero-day vulnerabilities if the strategy cannot recognize them.

        Args:
            state: Current simulation state
            verbose: Whether to print detailed cost info

        Returns:
            list: List of (vulnerability, asset, component) tuples
        """
        if not verbose:
            original_stdout = sys.stdout
            sys.stdout = io.StringIO()

        try:
            unpatched = []
            for asset in state.system.assets:
                for component in asset.components:
                    for vuln in component.vulnerabilities:
                        vuln_key = f"{vuln.cve_id}:{asset.asset_id}:{component.id}"

                        # Skip zero-day vulnerabilities if strategy cannot recognize them
                        if self.is_zero_day_vulnerability(vuln) and not self.can_recognize_zero_day():
                            continue

                        if not vuln.is_patched and vuln_key not in self._patched_vulns:
                            unpatched.append((vuln, asset, component))
        finally:
            if not verbose:
                sys.stdout = original_stdout

        if verbose:
            if unpatched:
                print(f"Found {len(unpatched)} unpatched vulnerabilities")
            else:
                print("Warning: No unpatched vulnerabilities found")

        return unpatched

    def _store_scoring_information(self, vuln_data_list: List[Dict]) -> None:
        """
        Store scoring information for adaptive strategy learning.
        Uses composite keys for vulnerability instances.
        """
        self._recently_scored_vulns = {}
        for vuln_info in vuln_data_list:
            if 'vuln_key' in vuln_info:
                self._recently_scored_vulns[vuln_info['vuln_key']] = vuln_info

    def _generic_strategy(self, state: State, remaining_budget: float, current_step: int,
                          total_steps: int, scoring_function: Any) -> List[Tuple[Vulnerability, float]]:
        """
        Generic vulnerability patching strategy that can be used for all strategies.
        Optimized to avoid redundant cost calculations and uses composite keys.
        Filters out zero-day vulnerabilities if the strategy cannot recognize them.

        Args:
            state: Current simulation state
            remaining_budget: Remaining patching budget
            current_step: Current simulation step
            total_steps: Total number of steps
            scoring_function: Function to score vulnerabilities

        Returns:
            list: List of (vulnerability, cost) tuples to patch
        """
        unpatched_vulns = self._get_unpatched_vulnerabilities(state, verbose=False)
        step_budget = self._calculate_decreasing_step_budget(
            remaining_budget, current_step, total_steps, remaining_budget)

        print(f"Step {current_step}: {self.name} has ${step_budget:.2f} budget remaining")

        if not unpatched_vulns:
            print(f"Step {current_step}: No unpatched vulnerabilities found for {self.name} strategy")
            return []

        local_cache = {}
        scored_vulns = []
        zero_days_filtered = 0

        for vuln, asset, component in unpatched_vulns:
            vuln_key = f"{vuln.cve_id}:{asset.asset_id}:{component.id}"

            # Skip zero-day vulnerabilities if strategy cannot recognize them
            if self.is_zero_day_vulnerability(vuln) and not self.can_recognize_zero_day():
                zero_days_filtered += 1
                continue

            if vuln.is_patched or vuln_key in self._patched_vulns:
                continue

            if vuln_key in local_cache:
                vuln_data = local_cache[vuln_key].copy()
                vuln_data['vulnerability'] = vuln
                vuln_data['asset'] = asset
                vuln_data['component'] = component
                score = scoring_function(vuln_data)
                vuln_data['score'] = score
                scored_vulns.append(vuln_data)
                continue

            vuln_info = self._cost_cache['vulnerability_info'].get(vuln_key, {})
            patch_cost = self._cost_cache['patch_costs'].get(vuln_key, 200.0)

            if patch_cost > step_budget:
                continue

            vuln_data = {
                'vulnerability': vuln,
                'asset': asset,
                'component': component,
                'cvss': vuln_info.get('cvss', getattr(vuln, 'cvss', 5.0)),
                'epss': vuln_info.get('epss', getattr(vuln, 'epss', 0.1)),
                'has_exploit': vuln_info.get('exploit', getattr(vuln, 'exploit', False)),
                'is_ransomware': vuln_info.get('ransomWare', getattr(vuln, 'ransomWare', False)),
                'business_value': vuln_info.get('business_value',
                                                getattr(asset, 'business_value',
                                                        getattr(asset, 'criticality_level', 3) * 5000)),
                'patch_cost': patch_cost,
                'vuln_key': vuln_key,
                'asset_name': asset.name,
                'asset_id': asset.asset_id,
                'component_id': component.id
            }

            local_cache[vuln_key] = vuln_data.copy()
            score = scoring_function(vuln_data)
            vuln_data['score'] = score
            scored_vulns.append(vuln_data)

        if zero_days_filtered > 0 and not self.can_recognize_zero_day():
            print(f"  {self.name}: Filtered out {zero_days_filtered} zero-day vulnerabilities (not recognized)")

        if self.name == "Adaptive Threat Intelligence":
            self._store_scoring_information(scored_vulns)

        scored_vulns.sort(key=lambda x: x['score'], reverse=True)

        patch_list = []
        total_cost = 0
        step_patched_vulns = set()

        for vuln_info in scored_vulns:
            if vuln_info['vuln_key'] in step_patched_vulns or vuln_info['vuln_key'] in self._patched_vulns:
                continue

            if total_cost + vuln_info['patch_cost'] <= step_budget:
                patch_list.append((vuln_info['vulnerability'], vuln_info['patch_cost']))
                total_cost += vuln_info['patch_cost']
                step_patched_vulns.add(vuln_info['vuln_key'])
                decision = "PATCH"
            else:
                decision = "SKIP (budget)"

        print(f"\n{self.name} - Step {current_step} Summary:")
        print(f"  Vulnerabilities patched: {len(patch_list)}")
        print(f"  Total patch cost: ${total_cost:.2f}")
        print(f"  Remaining step budget: ${step_budget - total_cost:.2f}")
        print(f"  Remaining total budget: ${remaining_budget - total_cost:.2f}")
        print(f"  Strategy: {self.name}")
        if zero_days_filtered > 0 and not self.can_recognize_zero_day():
            print(f"  Zero-day vulnerabilities filtered: {zero_days_filtered}")
        print("-" * 80)

        for vuln, _ in patch_list:
            vuln_key = f"{vuln.cve_id}:{asset.asset_id}:{component.id}"
            self._patched_vulns.add(vuln_key)

        return patch_list

    def select_patches(self, state: State, remaining_budget: float, current_step: int,
                       total_steps: int) -> List[Tuple[Vulnerability, float]]:
        """
        Abstract method to be implemented by subclasses.

        Args:
            state: Current simulation state
            remaining_budget: Remaining patching budget
            current_step: Current simulation step
            total_steps: Total number of steps

        Returns:
            list: List of (vulnerability, cost) tuples to patch
        """
        raise NotImplementedError("Subclasses must implement this method")


class CVSSOnlyStrategy(PatchingStrategy):
    """
    CVSS-only vulnerability patching strategy.
    Prioritizes vulnerabilities based on their CVSS scores.
    Cannot recognize zero-day vulnerabilities.
    """

    def __init__(self):
        super().__init__("CVSS-Only")

    def can_recognize_zero_day(self) -> bool:
        """CVSS-Only strategy cannot recognize zero-day vulnerabilities."""
        return False

    def select_patches(self, state: State, remaining_budget: float, current_step: int,
                       total_steps: int) -> List[Tuple[Vulnerability, float]]:
        def cvss_scoring(vuln_data):
            return vuln_data['cvss']

        return self._generic_strategy(state, remaining_budget, current_step, total_steps, cvss_scoring)


class CVSSExploitAwareStrategy(PatchingStrategy):
    """
    CVSS+Exploit vulnerability patching strategy.
    Prioritizes vulnerabilities based on their CVSS scores and EPSS scores.
    Cannot recognize zero-day vulnerabilities.
    """

    def __init__(self):
        super().__init__("CVSS+Exploit")

    def can_recognize_zero_day(self) -> bool:
        """CVSS+Exploit strategy cannot recognize zero-day vulnerabilities."""
        return False

    def select_patches(self, state: State, remaining_budget: float, current_step: int,
                       total_steps: int) -> List[Tuple[Vulnerability, float]]:
        def cvss_exploit_scoring(vuln_data):
            exploit_likelihood = self.calculate_cvss_exploit_likelihood(vuln_data['vulnerability'])
            return vuln_data['cvss'] * exploit_likelihood

        return self._generic_strategy(state, remaining_budget, current_step, total_steps, cvss_exploit_scoring)


class CostBenefitStrategy(PatchingStrategy):
    """
    Cost-benefit vulnerability patching strategy.
    Prioritizes vulnerabilities based on their risk-to-cost ratio.
    Cannot recognize zero-day vulnerabilities.
    """

    def __init__(self):
        super().__init__("Cost-Benefit")

    def can_recognize_zero_day(self) -> bool:
        """Cost-Benefit strategy cannot recognize zero-day vulnerabilities."""
        return False

    def select_patches(self, state: State, remaining_budget: float, current_step: int,
                       total_steps: int) -> List[Tuple[Vulnerability, float]]:
        def cost_benefit_scoring(vuln_data):
            vuln = vuln_data['vulnerability']
            asset = vuln_data['asset']
            component = vuln_data['component']
            if not asset:
                print(f"Warning: No asset found for {vuln_data['vuln_key']}, using default score")
                return 0.1
            score = self.cost_calculator.calculate_risk_to_cost_ratio(
                vuln, state.system, asset, component_id=component.id, cost_cache=self._cost_cache)
            return score

        return self._generic_strategy(state, remaining_budget, current_step, total_steps, cost_benefit_scoring)


class BusinessValueStrategy(PatchingStrategy):
    """
    Business value-based vulnerability patching strategy.
    Prioritizes vulnerabilities based on the business value of the affected assets.
    Cannot recognize zero-day vulnerabilities.
    """

    def __init__(self):
        super().__init__("Business Value")

    def can_recognize_zero_day(self) -> bool:
        """Business Value strategy cannot recognize zero-day vulnerabilities."""
        return False

    def select_patches(self, state: State, remaining_budget: float, current_step: int,
                       total_steps: int) -> List[Tuple[Vulnerability, float]]:
        def business_value_scoring(vuln_data):
            return vuln_data['business_value']

        return self._generic_strategy(state, remaining_budget, current_step, total_steps, business_value_scoring)


class ThreatIntelligenceStrategy(PatchingStrategy):
    """
    Enhanced Threat intelligence-based vulnerability patching strategy.
    Prioritizes vulnerabilities based on a combination of CVSS, EPSS,
    exploit/ransomware existence, and learned APT3 behavior patterns.
    Can recognize and prioritize zero-day vulnerabilities.

    New features:
    - Learns from observed attacker behavior via simplified POMDP
    - Focuses resources on APT3-relevant assets and techniques
    - Adapts patching priorities based on attack patterns
    """

    def __init__(self):
        super().__init__("Threat Intelligence")

        # APT3-specific configuration
        self.apt3_config = {
            'primary_objective': '8',  # RTU asset
            'secondary_objectives': ['2', '3', '6'],
            'entry_points': ['1', '3', '4', '5'],  # DC, Workstation, VPN, Web server
            'priority_cves': {
                'CVE-2018-13379': 2.0,  # VPN exploit
                'ZERO-DAY-001': 2.5,
                'CVE-2015-3113': 1.5
            },
            'preferred_techniques': {
                'T1203', 'T1068', 'T1055', 'T1548', 'T1190',
                'T1040', 'T1557', 'T1134', 'T1027', 'T1562.003'
            }
        }

        # Simplified POMDP learning components
        self.attack_observations = []  # Track observed attack patterns
        self.asset_threat_levels = {}  # Dynamic threat assessment per asset
        self.technique_frequency = {}  # Track frequency of observed techniques
        self.exploit_attempt_history = {}  # Track exploit attempts by CVE
        self.compromise_sequence = []  # Track sequence of compromised assets

        # Learning parameters
        self.learning_rate = 0.3
        self.observation_window = 10  # Number of recent observations to consider
        self.threat_decay_factor = 0.9  # How quickly threat levels decay

        # Resource allocation weights (sum to 1.0)
        self.allocation_weights = {
            'apt3_primary': 0.35,  # Primary objective (RTU)
            'apt3_secondary': 0.25,  # Secondary objectives
            'apt3_entry_points': 0.20,  # Known entry points
            'high_threat_assets': 0.15,  # Assets with observed attacks
            'general_defense': 0.05  # Everything else
        }

    def can_recognize_zero_day(self) -> bool:
        """Threat Intelligence strategy can recognize zero-day vulnerabilities."""
        return True

    def initialize(self, state: State, cost_cache: Dict):
        """Enhanced initialization with threat level assessment and attack detection setup."""
        super().initialize(state, cost_cache)
        self._initialize_asset_threat_levels(state)

        # Initialize tracking for automatic learning
        self._last_compromised_assets = set()
        self._last_exploited_vulns = set()
        self._last_step = 0

        # Track which vulnerabilities are being repeatedly targeted
        self._step_attack_patterns = []

    def _initialize_asset_threat_levels(self, state: State):
        """Initialize threat levels for all assets based on APT3 targeting."""
        for asset in state.system.assets:
            asset_id = str(asset.asset_id)

            # Base threat level
            if asset_id == self.apt3_config['primary_objective']:
                threat_level = 1.0  # Maximum threat
            elif asset_id in self.apt3_config['secondary_objectives']:
                threat_level = 0.8
            elif asset_id in self.apt3_config['entry_points']:
                threat_level = 0.7
            else:
                threat_level = 0.3  # Base level for all assets

            self.asset_threat_levels[asset_id] = threat_level

    def observe_attack_behavior(self, attack_observation: Dict):
        """
        Learn from observed attack behavior using simplified POMDP approach.

        Args:
            attack_observation: Dict containing attack details
                - action_type: Type of attack action
                - target_asset: Asset being targeted
                - target_vuln: Vulnerability being exploited (if any)
                - success: Whether the attack succeeded
                - techniques: MITRE techniques used (if known)
        """
        if not attack_observation:
            return

        # Add to observation history
        self.attack_observations.append(attack_observation)

        # Keep only recent observations
        if len(self.attack_observations) > self.observation_window:
            self.attack_observations.pop(0)

        target_asset = str(attack_observation.get('target_asset', ''))
        target_vuln = attack_observation.get('target_vuln', '')
        success = attack_observation.get('success', False)
        techniques = attack_observation.get('techniques', [])

        # Update asset threat levels based on targeting
        if target_asset:
            current_threat = self.asset_threat_levels.get(target_asset, 0.3)

            if success:
                # Successful attack significantly increases threat
                new_threat = min(1.0, current_threat + self.learning_rate * 0.5)

                # Track compromise sequence for lateral movement prediction
                if target_asset not in self.compromise_sequence:
                    self.compromise_sequence.append(target_asset)

            else:
                # Failed attack still indicates targeting
                new_threat = min(1.0, current_threat + self.learning_rate * 0.2)

            self.asset_threat_levels[target_asset] = new_threat

            # Increase threat for connected assets (lateral movement prediction)
            self._update_connected_asset_threats(target_asset)

        # Track exploit attempts
        if target_vuln:
            if target_vuln not in self.exploit_attempt_history:
                self.exploit_attempt_history[target_vuln] = {'attempts': 0, 'successes': 0}

            self.exploit_attempt_history[target_vuln]['attempts'] += 1
            if success:
                self.exploit_attempt_history[target_vuln]['successes'] += 1

        # Track technique frequency
        for technique in techniques:
            self.technique_frequency[technique] = self.technique_frequency.get(technique, 0) + 1

        print(f"  Threat Intel: Learned from attack on {target_asset} "
              f"({'successful' if success else 'failed'}), "
              f"new threat level: {self.asset_threat_levels.get(target_asset, 0.0):.3f}")

    def _update_connected_asset_threats(self, compromised_asset: str):
        """Update threat levels for assets connected to a compromised one."""
        if not hasattr(self, 'state') or not self.state:
            return

        # Find connected assets
        connected_assets = []
        for conn in self.state.system.connections:
            if (hasattr(conn, 'from_asset') and conn.from_asset and
                    str(conn.from_asset.asset_id) == compromised_asset):
                connected_assets.append(str(conn.to_asset.asset_id))
            elif (hasattr(conn, 'to_asset') and conn.to_asset and
                  str(conn.to_asset.asset_id) == compromised_asset):
                connected_assets.append(str(conn.from_asset.asset_id))

        # Increase threat for connected assets (lateral movement risk)
        for asset_id in connected_assets:
            if asset_id in self.asset_threat_levels:
                current_threat = self.asset_threat_levels[asset_id]
                boost = self.learning_rate * 0.3  # Smaller boost for indirect targeting
                self.asset_threat_levels[asset_id] = min(1.0, current_threat + boost)

    def _decay_threat_levels(self):
        """Gradually decay threat levels over time if no attacks observed."""
        for asset_id in self.asset_threat_levels:
            current_threat = self.asset_threat_levels[asset_id]

            # Don't decay below base levels for APT3 targets
            if asset_id == self.apt3_config['primary_objective']:
                min_threat = 1.0
            elif asset_id in self.apt3_config['secondary_objectives']:
                min_threat = 0.8
            elif asset_id in self.apt3_config['entry_points']:
                min_threat = 0.7
            else:
                min_threat = 0.3

            decayed_threat = max(min_threat, current_threat * self.threat_decay_factor)
            self.asset_threat_levels[asset_id] = decayed_threat

    def calculate_adaptive_budget_allocation(self, total_budget: float) -> Dict[str, float]:
        """
        Calculate budget allocation based on learned threat patterns and APT3 focus.

        Returns:
            Dict mapping asset categories to budget amounts
        """
        allocation = {}

        # Allocate based on predefined weights
        allocation['apt3_primary'] = total_budget * self.allocation_weights['apt3_primary']
        allocation['apt3_secondary'] = total_budget * self.allocation_weights['apt3_secondary']
        allocation['apt3_entry_points'] = total_budget * self.allocation_weights['apt3_entry_points']
        allocation['high_threat_assets'] = total_budget * self.allocation_weights['high_threat_assets']
        allocation['general_defense'] = total_budget * self.allocation_weights['general_defense']

        # Adjust based on recent attack patterns
        if len(self.attack_observations) >= 3:
            recent_targets = [obs.get('target_asset', '') for obs in self.attack_observations[-3:]]

            # If primary objective recently targeted, increase its allocation
            if self.apt3_config['primary_objective'] in recent_targets:
                boost = total_budget * 0.1
                allocation['apt3_primary'] += boost
                allocation['general_defense'] = max(0, allocation['general_defense'] - boost)

            # If entry points heavily targeted, increase their allocation
            entry_point_attacks = sum(1 for target in recent_targets
                                      if target in self.apt3_config['entry_points'])
            if entry_point_attacks >= 2:
                boost = total_budget * 0.05
                allocation['apt3_entry_points'] += boost
                allocation['general_defense'] = max(0, allocation['general_defense'] - boost)

        return allocation

    def calculate_apt3_asset_priority_multiplier(self, asset_id: str, vuln: Any) -> float:
        """
        Calculate priority multiplier based on APT3 relevance and learned behavior.

        Args:
            asset_id: Asset identifier
            vuln: Vulnerability object

        Returns:
            Priority multiplier (higher = more priority)
        """
        multiplier = 1.0

        # APT3 target priority
        if asset_id == self.apt3_config['primary_objective']:
            multiplier *= 3.0
        elif asset_id in self.apt3_config['secondary_objectives']:
            multiplier *= 2.0
        elif asset_id in self.apt3_config['entry_points']:
            multiplier *= 1.5

        # CVE priority based on APT3 preferences
        cve_id = getattr(vuln, 'cve_id', '')
        if cve_id in self.apt3_config['priority_cves']:
            multiplier *= self.apt3_config['priority_cves'][cve_id]

        # Dynamic threat level from observations
        current_threat = self.asset_threat_levels.get(asset_id, 0.3)
        threat_multiplier = 1.0 + (current_threat - 0.3) * 2.0  # Scale 0.3-1.0 to 1.0-2.4
        multiplier *= threat_multiplier

        # Exploit attempt history
        if cve_id in self.exploit_attempt_history:
            attempts = self.exploit_attempt_history[cve_id]['attempts']
            if attempts > 0:
                # CVEs that have been attempted get higher priority
                multiplier *= (1.0 + min(attempts * 0.2, 1.0))

        # Zero-day recognition bonus
        if self.is_zero_day_vulnerability(vuln):
            multiplier *= 2.5  # High priority for zero-days

        return multiplier

    def _learn_from_current_state(self, state: State, current_step: int):
        """
        Learn from the current state to detect attacks and compromises.
        This simulates the POMDP learning from observed system state.
        """
        # Detect newly compromised assets
        current_compromised = {str(asset.asset_id) for asset in state.system.assets if asset.is_compromised}

        # Find assets that were compromised since last observation
        if not hasattr(self, '_last_compromised_assets'):
            self._last_compromised_assets = set()

        newly_compromised = current_compromised - self._last_compromised_assets

        # Learn from newly compromised assets
        for asset_id in newly_compromised:
            self.observe_attack_behavior({
                'action_type': 'compromise',
                'target_asset': asset_id,
                'success': True,
                'step': current_step,
                'techniques': ['T1190', 'T1068']  # Common APT3 techniques
            })
            print(f"  Threat Intel: Learned from compromise of asset {asset_id}")

        # Detect repeated failed attacks (from logs pattern)
        if current_step > 0 and len(newly_compromised) == 0:
            # Infer failed attacks from repeated patterns
            for asset_id in ['3', '5', '6']:  # Common targets from logs
                if asset_id not in current_compromised:
                    # Simulate learning from failed attack attempts
                    self.observe_attack_behavior({
                        'action_type': 'failed_attack',
                        'target_asset': asset_id,
                        'success': False,
                        'step': current_step
                    })

        # Update our knowledge
        self._last_compromised_assets = current_compromised

        # Detect exploited vulnerabilities
        current_exploited = set()
        for asset in state.system.assets:
            for comp in asset.components:
                for vuln in comp.vulnerabilities:
                    if getattr(vuln, 'is_exploited', False):
                        vuln_key = f"{vuln.cve_id}:{asset.asset_id}:{comp.id}"
                        current_exploited.add(vuln_key)

        # Learn from newly exploited vulnerabilities
        if not hasattr(self, '_last_exploited_vulns'):
            self._last_exploited_vulns = set()

        newly_exploited = current_exploited - self._last_exploited_vulns
        for vuln_key in newly_exploited:
            parts = vuln_key.split(':')
            if len(parts) >= 2:
                cve_id, asset_id = parts[0], parts[1]
                self.observe_attack_behavior({
                    'action_type': 'exploitation',
                    'target_asset': asset_id,
                    'target_vuln': cve_id,
                    'success': True,
                    'step': current_step
                })

        self._last_exploited_vulns = current_exploited

    def predict_next_targets(self) -> List[str]:
        """
        Enhanced prediction based on observed attack patterns and APT3 behavior.
        """
        if len(self.attack_observations) < 2:
            # Default APT3 progression with better cost consideration
            return ['1', '4', '5']  # Prioritize lower-cost entry points first

        predictions = []

        # Analyze recent attack patterns
        recent_targets = [obs.get('target_asset', '') for obs in self.attack_observations[-5:]]
        successful_attacks = [obs for obs in self.attack_observations if obs.get('success', False)]

        # If entry points are being attacked, predict lateral movement
        entry_point_attacks = sum(1 for target in recent_targets
                                  if target in self.apt3_config['entry_points'])

        if entry_point_attacks >= 2:
            # Heavy focus on entry points, predict they'll move to secondary objectives
            predictions.extend(['2', '6'])  # Secondary objectives

        # If any secondary objectives compromised, predict primary objective attack
        if any(obs.get('target_asset') in self.apt3_config['secondary_objectives']
               for obs in successful_attacks):
            predictions.append(self.apt3_config['primary_objective'])  # RTU

        # Add highly threatened assets from our learned threat levels
        high_threat_assets = [asset_id for asset_id, threat in self.asset_threat_levels.items()
                              if threat > 0.6 and asset_id not in self.compromise_sequence]
        predictions.extend(high_threat_assets)

        # Fallback to APT3 typical targets with cost consideration
        if not predictions:
            # Prioritize based on both APT3 preferences and patch costs
            cost_effective_targets = []
            for asset_id in ['1', '4', '5', '3']:  # Reordered by typical cost-effectiveness
                if asset_id not in self.compromise_sequence:
                    cost_effective_targets.append(asset_id)
            predictions.extend(cost_effective_targets[:3])

        # Remove duplicates and limit to reasonable number
        unique_predictions = list(dict.fromkeys(predictions))[:4]

        return unique_predictions

    def select_patches(self, state: State, remaining_budget: float, current_step: int,
                       total_steps: int) -> List[Tuple[Vulnerability, float]]:
        """Simplified patch selection with balanced budget allocation and basic threat intelligence scoring."""

        # FIXED: Use balanced budget allocation for Threat Intelligence
        step_budget = self._calculate_balanced_step_budget(
            remaining_budget, current_step, total_steps, state)

        # FIXED: Use simplified scoring similar to original Threat Intelligence
        def simplified_threat_intel_scoring(vuln_data):
            vuln = vuln_data['vulnerability']
            
            # Base threat intelligence score (same as original)
            likelihood = self.calculate_threat_intelligence_likelihood(vuln)
            impact = vuln_data['business_value'] * (vuln_data['cvss'] / 10.0)
            risk = likelihood * impact
            
            # Simple cost-effectiveness calculation
            score = risk / vuln_data['patch_cost'] if vuln_data['patch_cost'] > 0 else 0.0
            
            # FIXED: Add minimal APT3 awareness without complex learning
            asset_id = vuln_data['asset_id']
            if asset_id == '8':  # RTU - primary objective
                score *= 1.2
            elif asset_id in ['2', '3', '6']:  # Secondary objectives
                score *= 1.1
            elif asset_id in ['1', '4', '5']:  # Entry points
                score *= 1.05
                
            # FIXED: Zero-day recognition (simple version)
            if self.is_zero_day_vulnerability(vuln):
                score *= 1.5
                print(f"  Threat Intel: Recognized zero-day {vuln.cve_id} on {asset_id}")

            return score

        # Use the generic strategy with simplified scoring and balanced budget
        patches = self._generic_strategy(state, step_budget, current_step, total_steps,
                                         simplified_threat_intel_scoring)

        return patches

    def _calculate_balanced_step_budget(self, remaining_budget: float, current_step: int,
                                       total_steps: int, state: State) -> float:
        """
        Calculate a balanced budget allocation for Threat Intelligence strategy.
        Fixes conservative spending without being overly aggressive.
        """
        if remaining_budget <= 0:
            return 0
        if total_steps <= 1:
            return remaining_budget

        steps_remaining = total_steps - current_step
        
        # FIXED: Use balanced fraction calculation - more aggressive than original but not too aggressive
        if current_step < 5:  # Early steps get more budget
            fraction = max(0.12, 1.8 / max(1, steps_remaining))  # Balanced approach
        elif current_step < 10:  # Middle steps
            fraction = max(0.10, 1.6 / max(1, steps_remaining))
        else:  # Later steps
            fraction = max(0.08, 1.4 / max(1, steps_remaining))

        base_budget = remaining_budget * fraction
        
        # Get unpatched vulnerabilities to calculate cost-based constraints
        unpatched_vulns = self._get_unpatched_vulnerabilities(state, verbose=False)
        patch_costs = [
            self._cost_cache['patch_costs'].get(f"{vuln.cve_id}:{asset.asset_id}:{comp.id}", 200.0)
            for vuln, asset, comp in unpatched_vulns
            if not (self.is_zero_day_vulnerability(vuln) and not self.can_recognize_zero_day())
        ]

        if not patch_costs:
            return min(remaining_budget, 800.0)  # Reasonable default budget

        min_budget = min(patch_costs)
        median_cost = sorted(patch_costs)[len(patch_costs) // 2]
        
        # FIXED: Balanced max budget - more than original but not excessive
        if current_step < 5:
            max_budget = median_cost * 4.0  # Allow reasonable number of patches
        else:
            max_budget = median_cost * 3.5  # Standard approach for later steps

        # FIXED: Ensure minimum budget allows at least 1-2 patches
        min_required_budget = min_budget * 2 if current_step < 5 else min_budget * 1.5
        
        # Calculate final step budget
        step_budget = max(min_required_budget, min(base_budget, max_budget, remaining_budget))
        
        # FIXED: Small urgency bonus for early steps with high-value assets at risk
        if current_step < 5:
            high_value_assets = [asset for asset in state.system.assets 
                               if getattr(asset, 'business_value', 0) > 5000]
            if high_value_assets:
                urgency_bonus = remaining_budget * 0.03  # 3% bonus for urgency (reduced from 5%)
                step_budget = min(step_budget + urgency_bonus, remaining_budget)

        print(f"Step {current_step}: Threat Intel - Steps remaining={steps_remaining}, "
              f"Fraction={fraction:.2f}, Step Budget=${step_budget:.2f}, "
              f"Min Budget=${min_budget:.2f}, Max Budget=${max_budget:.2f}, "
              f"Remaining Budget=${remaining_budget:.2f}")
        
        return step_budget