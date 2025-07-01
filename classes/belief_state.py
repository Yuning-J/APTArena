# belief_state.py
"""
BeliefState class for tracking attacker beliefs in a POMDP framework.
This represents the attacker's probability distribution over possible system states.
"""
import logging
import numpy as np
from collections import defaultdict
from classes.state import KillChainStage
from classes.belief_update import update_detection_belief

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BeliefState:
    """
    Represents the attacker's belief about the current system state.
    Maintains probability distributions over vulnerability states, asset states, and detection risks.
    """

    def __init__(self, system=None):
        """
        Initialize belief state with uniform distributions.

        Args:
            system: Optional initial system state to initialize beliefs
        """
        # Belief about which vulnerabilities are patched (vuln_key -> probability)
        self.vuln_patched_prob = {}  # Format: cve_id:asset_id:component_id

        # Belief about which vulnerabilities are exploited (vuln_key -> probability)
        self.vuln_exploited_prob = {}  # Format: cve_id:asset_id:component_id

        # Belief about which assets are compromised (asset_id -> probability)
        self.asset_compromised_prob = {}

        # Belief about global detection probability
        self.detection_prob = 0.1  # Initial belief that attacker is detected

        # Belief about per-asset detection probabilities (asset_id -> probability)
        self.asset_detection_probs = {}

        # Belief about phishing success probabilities (asset_id -> probability)
        self.phishing_success_prob = {}

        # Map vulnerability keys to objects (for reference)
        self._vuln_map = {}  # vuln_key -> Vulnerability object

        # Map asset IDs to objects (for reference)
        self._asset_map = {}

        # Initialize from system state if provided
        if system:
            self._initialize_from_system(system)

    def _initialize_from_system(self, system):
        """
        Initialize belief state from system information.
        Initially assumes uniform uncertainty about patch and exploited status.

        Args:
            system: System object with assets and vulnerabilities
        """
        for asset in system.assets:
            asset_id = str(asset.asset_id)
            # Initial belief that asset is not compromised
            self.asset_compromised_prob[asset_id] = 0.0 if not asset.is_compromised else 1.0
            self._asset_map[asset_id] = asset
            # Initial per-asset detection probability based on security_controls
            base_prob = 0.1 * (1 + 0.1 * getattr(asset, 'security_controls', 0))
            self.asset_detection_probs[asset_id] = min(base_prob, 1.0)

            for comp in asset.components:
                for vuln in comp.vulnerabilities:
                    vuln_key = f"{vuln.cve_id}:{asset_id}:{comp.id}"
                    # Initial belief: 50% uncertainty for patch status
                    self.vuln_patched_prob[vuln_key] = 0.5
                    # Initial belief: not exploited unless known
                    self.vuln_exploited_prob[vuln_key] = 0.0
                    self._vuln_map[vuln_key] = vuln

        logger.debug(
            f"Initialized beliefs for {len(self.asset_compromised_prob)} assets and {len(self.vuln_patched_prob)} vulnerabilities")

    def get_detection_prob(self, asset_id: str, kill_chain_stage: KillChainStage) -> float:
        """
        Return detection probability for a specific asset and kill chain stage.

        Args:
            asset_id: ID of the asset
            kill_chain_stage: Current kill chain stage (KillChainStage enum)

        Returns:
            float: Detection probability adjusted by asset properties and stage
        """
        # Get base detection probability for the asset
        base_prob = self.asset_detection_probs.get(asset_id, self.detection_prob)
        # Increase detection probability in later kill chain stages
        stage_modifier = 1.0 + 0.1 * (kill_chain_stage.value - 1)
        # Adjust based on global detection probability
        adjusted_prob = base_prob * stage_modifier * (1 + self.detection_prob)
        return min(1.0, adjusted_prob)

    def update_from_observation(self, observation):
        """
        Update belief state based on an observation using Bayesian inference.

        Args:
            observation: Dictionary containing observation details
                - action_result: Success/failure of last action
                - detected_patches: List of vulnerabilities observed to be patched
                - action_type: Type of last action (e.g., exploit, scan)
                - target_asset: Asset targeted by last action
                - target_vuln: Vulnerability targeted by last action
                - target_component: Component ID of the targeted vulnerability
                - detected: Boolean indicating defender detection
        """
        # Update based on exploit attempt results
        if observation.get('action_type') in ['exploit', 'initial_access', 'exploitation', 'lateral_movement',
                                              'privilege_escalation'] and 'target_vuln' in observation:
            vuln_key = f"{observation['target_vuln']}:{observation.get('target_asset')}:{observation.get('target_component', '0')}"
            self._update_exploit_result(
                observation['action_result'],
                vuln_key,
                observation.get('target_asset')
            )

        # Update based on observed patches from scanning
        if 'detected_patches' in observation:
            for vuln_key in observation['detected_patches']:
                if vuln_key in self.vuln_patched_prob:
                    # If patch is detected, high confidence it's patched
                    self.vuln_patched_prob[vuln_key] = 0.95
                    logger.debug(f"Updated patch belief for {vuln_key} to 0.95 due to detected patch")

        # Update based on lateral movement result
        if observation.get('action_type') == 'lateral_movement' and 'target_asset' in observation:
            vuln_key = f"{observation.get('target_vuln')}:{observation.get('target_asset')}:{observation.get('target_component', '0')}" if observation.get(
                'target_vuln') else None
            self._update_lateral_movement_result(
                observation['action_result'],
                observation['target_asset'],
                vuln_key
            )

        # Update detection probabilities based on defender actions
        if 'detected' in observation:
            # Update global detection probability
            detection_likelihood = 0.8 if observation['detected'] else 0.2
            prior = self.detection_prob
            self.detection_prob = update_detection_belief(prior, observation, defender_accuracy=0.8)
            logger.debug(f"Updated global detection probability: {prior:.3f} -> {self.detection_prob:.3f}")

            # Update per-asset detection probabilities
            for asset_id in self.asset_detection_probs:
                self.asset_detection_probs[asset_id] = update_detection_belief(
                    self.asset_detection_probs[asset_id],
                    observation,
                    defender_accuracy=0.8
                )
                logger.debug(
                    f"Updated detection probability for asset {asset_id}: {self.asset_detection_probs[asset_id]:.3f}")

        # Update based on impact operations (which likely increase detection)
        if observation.get('action_type') == 'impact' and observation.get('action_result', False):
            self.detection_prob = min(self.detection_prob + 0.3, 1.0)
            for asset_id in self.asset_detection_probs:
                self.asset_detection_probs[asset_id] = min(
                    self.asset_detection_probs[asset_id] + 0.3, 1.0
                )
            logger.debug(f"Increased detection probabilities due to impact action: global={self.detection_prob:.3f}")

    def _update_exploit_result(self, result, vuln_key, target_asset_id=None):
        """
        Update beliefs based on the result of an exploit attempt.
        Uses Bayes' rule to update probability distributions with enhanced failure tracking.

        Args:
            result: Boolean indicating success/failure of action
            vuln_key: Vulnerability key (cve_id:asset_id:component_id)
            target_asset_id: Optional ID of asset targeted
        """
        # Track failure count for this vulnerability
        if not hasattr(self, '_failure_counts'):
            self._failure_counts = {}
        
        if vuln_key not in self._failure_counts:
            self._failure_counts[vuln_key] = 0
            
        # Update belief about vulnerability patch status
        if vuln_key in self.vuln_patched_prob:
            if result is False:
                # If exploit failed, increase belief vulnerability is patched
                self._failure_counts[vuln_key] += 1
                failure_count = self._failure_counts[vuln_key]
                
                # Enhanced failure response: more failures = higher patch belief
                if failure_count == 1:
                    # First failure: moderate increase in patch belief
                    p_patched = self.vuln_patched_prob[vuln_key]
                    p_fail_given_patched = 0.95
                    p_fail_given_unpatched = 0.4
                    p_failed = (p_fail_given_patched * p_patched) + (p_fail_given_unpatched * (1 - p_patched))
                    if p_failed > 0:
                        p_patched_given_fail = (p_fail_given_patched * p_patched) / p_failed
                        self.vuln_patched_prob[vuln_key] = p_patched_given_fail
                        logger.debug(f"Updated patch belief for {vuln_key}: {p_patched:.3f} -> {p_patched_given_fail:.3f} (failure #{failure_count})")
                elif failure_count >= 2:
                    # Multiple failures: significant increase in patch belief
                    current_patch_belief = self.vuln_patched_prob[vuln_key]
                    # Increase patch belief by 0.2 per additional failure, capped at 0.95
                    new_patch_belief = min(0.95, current_patch_belief + 0.2)
                    self.vuln_patched_prob[vuln_key] = new_patch_belief
                    logger.debug(f"Enhanced patch belief for {vuln_key}: {current_patch_belief:.3f} -> {new_patch_belief:.3f} (failure #{failure_count})")
                    
                    # Also reduce exploited belief on repeated failures
                    if vuln_key in self.vuln_exploited_prob:
                        current_exploited_belief = self.vuln_exploited_prob[vuln_key]
                        new_exploited_belief = max(0.0, current_exploited_belief - 0.3)
                        self.vuln_exploited_prob[vuln_key] = new_exploited_belief
                        logger.debug(f"Reduced exploited belief for {vuln_key}: {current_exploited_belief:.3f} -> {new_exploited_belief:.3f}")
            else:
                # If exploit succeeded, vulnerability is definitely not patched
                self.vuln_patched_prob[vuln_key] = 0.0
                # Update exploited belief
                self.vuln_exploited_prob[vuln_key] = 1.0
                # Reset failure count on success
                self._failure_counts[vuln_key] = 0
                logger.debug(f"Updated beliefs for {vuln_key}: patch=0.0, exploited=1.0 (success)")

        # Update belief about asset compromise status
        if target_asset_id and target_asset_id in self.asset_compromised_prob:
            if result is True:
                # If exploit succeeded, asset is definitely compromised
                self.asset_compromised_prob[target_asset_id] = 1.0
                logger.debug(f"Updated compromise belief for asset {target_asset_id}: 1.0")
            else:
                # Slightly decrease compromise belief on failure
                current_belief = self.asset_compromised_prob[target_asset_id]
                self.asset_compromised_prob[target_asset_id] = max(0.0, current_belief - 0.1)
                logger.debug(
                    f"Updated compromise belief for asset {target_asset_id}: {current_belief:.3f} -> {self.asset_compromised_prob[target_asset_id]:.3f}")

    def _update_lateral_movement_result(self, result, target_asset_id, vuln_key=None):
        """
        Update beliefs based on lateral movement result.

        Args:
            result: Boolean indicating success/failure of lateral movement
            target_asset_id: ID of target asset
            vuln_key: Optional vulnerability key (cve_id:asset_id:component_id)
        """
        # Update asset compromise belief
        if target_asset_id in self.asset_compromised_prob:
            if result is True:
                # If movement succeeded, target asset is definitely compromised
                self.asset_compromised_prob[target_asset_id] = 1.0
                logger.debug(f"Updated compromise belief for asset {target_asset_id}: 1.0")
            else:
                # If movement failed, slightly lower belief that target is compromised
                current_belief = self.asset_compromised_prob[target_asset_id]
                self.asset_compromised_prob[target_asset_id] = max(0, current_belief - 0.2)
                logger.debug(
                    f"Updated compromise belief for asset {target_asset_id}: {current_belief:.3f} -> {self.asset_compromised_prob[target_asset_id]:.3f}")

        # Update vulnerability patch and exploited beliefs if specified
        if vuln_key and vuln_key in self.vuln_patched_prob:
            if result is False:
                # If movement failed, increase belief vulnerability is patched
                current_belief = self.vuln_patched_prob[vuln_key]
                self.vuln_patched_prob[vuln_key] = min(1.0, current_belief + 0.3)
                logger.debug(
                    f"Updated patch belief for {vuln_key}: {current_belief:.3f} -> {self.vuln_patched_prob[vuln_key]:.3f}")
            else:
                # If movement succeeded using this vulnerability, it's not patched and is exploited
                self.vuln_patched_prob[vuln_key] = 0.0
                self.vuln_exploited_prob[vuln_key] = 1.0
                logger.debug(f"Updated beliefs for {vuln_key}: patch=0.0, exploited=1.0")

    def update_from_scanning(self, scan_results):
        """
        Update beliefs based on network scanning results.

        Args:
            scan_results: Dictionary mapping vulnerability keys to observed patch status
        """
        for vuln_key, is_patched in scan_results.items():
            if vuln_key in self.vuln_patched_prob:
                # Scanning has high but not perfect accuracy
                self.vuln_patched_prob[vuln_key] = 0.9 if is_patched else 0.1
                logger.debug(f"Updated patch belief for {vuln_key}: {'0.9' if is_patched else '0.1'}")

    def get_vulnerability_by_id(self, vuln_key):
        """Get vulnerability object by key (cve_id:asset_id:component_id)."""
        return self._vuln_map.get(vuln_key)

    def get_asset_by_id(self, asset_id):
        """Get asset object by ID."""
        return self._asset_map.get(asset_id)

    def is_likely_patched(self, vuln_key, threshold=0.7):
        """Check if vulnerability is likely patched based on belief."""
        return self.vuln_patched_prob.get(vuln_key, 0.5) > threshold

    def is_likely_compromised(self, asset_id, threshold=0.7):
        """Check if asset is likely compromised based on belief."""
        return self.asset_compromised_prob.get(asset_id, 0.0) > threshold

    def get_patch_belief(self, vuln_key: str) -> float:
        """
        Get the belief probability that a vulnerability is patched.

        Args:
            vuln_key: Vulnerability key in format 'cve_id:asset_id:component_id'

        Returns:
            float: Probability that the vulnerability is patched
        """
        return self.vuln_patched_prob.get(vuln_key, 0.5)

    def get_exploited_belief(self, vuln_key: str) -> float:
        """
        Get the belief probability that a vulnerability is exploited.

        Args:
            vuln_key: Vulnerability key in format 'cve_id:asset_id:component_id'

        Returns:
            float: Probability that the vulnerability is exploited
        """
        return self.vuln_exploited_prob.get(vuln_key, 0.5)