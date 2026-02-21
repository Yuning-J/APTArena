# attacker_posg.py
"""
POMDP-based attacker policy that makes decisions under uncertainty.
Uses a belief state to track probability distributions over the system state.
"""
import random
import logging
import numpy as np
from collections import defaultdict
from functools import lru_cache

# Import directly from the modules
from .state import State
from .mitre import MitreTTP, APT3TacticMapping, KillChainStage, mitre_ttps, ckc_to_mitre, technique_to_tactic
from .cost import CostCalculator
from .belief_state import BeliefState
from .transition import create_vuln_key

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AttackerPOMDPPolicy:
    """
    Enhanced attacker policy that implements a POMDP approach.
    Maintains and updates belief states to handle partial observability.
    Makes decisions based on belief distributions rather than full observation.
    """

    def __init__(self, cost_aware=True, detection_averse=True, enhanced_exploit_priority=False, system=None):
        self.observation_history = []
        self.cost_aware = cost_aware
        self.detection_averse = detection_averse
        self.enhanced_exploit_priority = enhanced_exploit_priority
        self.exploit_outcomes = {}
        self.tactic_frustration = {}
        self.tactic_cooldown = {}
        self.action_asset_failures = defaultdict(lambda: defaultdict(int))
        self.technique_to_tactic = technique_to_tactic
        self._mitre_ttps_by_stage = self._group_ttps_by_stage()
        self._mitre_ttps_by_name = {ttp.name: ttp for ttp in mitre_ttps}
        self.operation_count = 0
        self.last_target_asset = None
        self.target_history = []
        self.recent_failures = 0
        self.last_actions = []
        self.high_value_priority = 1.0
        self.exploited_vulnerabilities = set()
        self.kill_chain_stage = KillChainStage.RECONNAISSANCE
        self.belief_state = BeliefState(system=system) if system else None
        logger.info(
            f"APT3 POMDP Attacker Policy initialized (cost_aware={cost_aware}, detection_averse={detection_averse})")

    def reset(self):
        """Reset policy for new simulation."""
        self.observation_history = []
        self.exploit_outcomes = {}
        self.tactic_frustration = {}
        self.tactic_cooldown = {}
        self.action_asset_failures = defaultdict(lambda: defaultdict(int))
        self.operation_count = 0
        self.last_target_asset = None
        self.target_history = []
        self.recent_failures = 0
        self.last_actions = []
        self.exploited_vulnerabilities = set()
        self.kill_chain_stage = KillChainStage.RECONNAISSANCE
        self.belief_state = None  # Clear belief state
        logger.info("APT3 POMDP Attacker Policy reset to initial state.")

    def update_belief_state(self, action_result: dict):
        """
        Update the belief state based on the action result.

        Args:
            action_result: Dictionary containing action result details
        """
        if not self.belief_state:
            logger.warning("Belief state not initialized, skipping update")
            return
        if not isinstance(action_result, dict):
            logger.error(f"Invalid action_result type: {type(action_result)}, expected dict")
            return
        try:
            self.belief_state.update_from_observation(action_result)
            logger.debug("Belief state updated successfully")
        except Exception as e:
            logger.error(f"Error updating belief state: {e}", exc_info=True)

    def initialize_belief_state(self, system):
        """
        Initialize or update belief state.
        """
        if system is None:
            logger.warning("System is None, initializing empty belief state")
            self.belief_state = BeliefState(system=None)
        else:
            logger.info("Initializing belief state with system")
            try:
                self.belief_state = BeliefState(system=system)
            except Exception as e:
                logger.error(f"Failed to initialize belief state: {e}")
                self.belief_state = BeliefState(system=None)
        return self.belief_state

    def _group_ttps_by_stage(self):
        stage_mapping = defaultdict(list)
        for ttp in mitre_ttps:
            if ttp.kill_chain_stage is not None:
                stage_mapping[ttp.kill_chain_stage].append(ttp)
        return stage_mapping

    def get_exploitability_under_uncertainty(self, vuln, tactic_name, state=None, current_position=None):
        """
        Calculate exploitability score under uncertainty for a vulnerability.
        Modified to use belief-based probabilities for patch and exploited status.

        Args:
            vuln: Vulnerability object
            tactic_name: Name of the tactic
            state: Current system state (optional)
            current_position: Current attacker position (optional)

        Returns:
            float: Exploitability score
        """
        if not hasattr(vuln, 'cve_id') or vuln.cve_id is None or vuln.cve_id.lower() == 'unknown':
            logger.debug(f"Skipping vulnerability with invalid CVE ID: {getattr(vuln, 'cve_id', 'None')}")
            return 0.01

        # Find asset and component for vuln to construct vuln_key
        vuln_key = None
        if state and state.system:
            for asset in state.system.assets:
                for comp in asset.components:
                    if vuln in comp.vulnerabilities:
                        vuln_key = f"{vuln.cve_id}:{asset.asset_id}:{comp.id}"
                        break
                if vuln_key:
                    break

        # Use belief-based patch probability
        if vuln_key and self.belief_state:
            patch_prob = self.belief_state.get_patch_belief(vuln_key)
            if patch_prob > 0.7:  # Align with is_likely_patched threshold
                logger.debug(f"Vulnerability {vuln.cve_id} likely patched (belief: {patch_prob:.3f})")
                return 0.05

            # Check exploited status
            exploited_prob = self.belief_state.get_exploited_belief(vuln_key)
            if exploited_prob > 0.7:
                logger.debug(f"Vulnerability {vuln.cve_id} likely exploited (belief: {exploited_prob:.3f})")
                return 0.01

        base_likelihood = self._get_base_exploit_probability(vuln)
        vuln_techniques = set(getattr(vuln, 'mitre_techniques', []))
        apt3_techniques = set(APT3TacticMapping.get_preferred_techniques(self.kill_chain_stage))
        if vuln_techniques & apt3_techniques:
            base_likelihood *= 1.5
            logger.debug(f"APT3 technique boost applied: {vuln_techniques & apt3_techniques}")
        if vuln.cve_id in APT3TacticMapping.APT3_CVE_TO_TECHNIQUE:
            base_likelihood *= 2.0
            logger.debug(f"APT3 CVE boost applied: {vuln.cve_id}")
        if self._is_client_side_vulnerability(vuln):
            base_likelihood *= 1.3
            logger.debug(f"Client-side boost applied: {vuln.cve_id}")

        # Boost for compromised or connected assets
        if state and state.system:
            for asset in state.system.assets:
                for comp in asset.components:
                    if vuln in comp.vulnerabilities:
                        if self.belief_state and self.belief_state.is_likely_compromised(asset.asset_id):
                            base_likelihood *= 1.5
                            logger.debug(f"Compromised asset boost applied: {vuln.cve_id} on {asset.asset_id}")
                        elif current_position and any(
                                conn.from_asset.asset_id == current_position and conn.to_asset.asset_id == asset.asset_id
                                for conn in state.system.connections
                        ):
                            base_likelihood *= 1.3
                            logger.debug(f"Connected asset boost applied: {vuln.cve_id} on {asset.asset_id}")
                        break

        base_likelihood = self._apply_detection_aversion(base_likelihood)
        base_likelihood = self._apply_prior_attempt_adjustment(vuln, base_likelihood)
        base_likelihood = self._apply_tactic_alignment(vuln, tactic_name, base_likelihood)
        if self.enhanced_exploit_priority:
            has_exploit = getattr(vuln, 'exploit', False)
            base_likelihood *= 2.5 if has_exploit else 0.7
            epss = getattr(vuln, 'epss', 0.1)
            if epss > 0.3:
                base_likelihood *= 1.8
            if getattr(vuln, 'ransomWare', False):
                base_likelihood *= 1.5

        # Adjust for patch belief
        if vuln_key and self.belief_state:
            base_likelihood *= (1.0 - self.belief_state.get_patch_belief(vuln_key))

        return min(max(base_likelihood, 0.0), 1.0)

    def _is_client_side_vulnerability(self, vuln):
        client_side_keywords = ['browser', 'office', 'pdf', 'flash', 'java']
        description = getattr(vuln, 'description', '').lower()
        return any(keyword in description for keyword in client_side_keywords)

    def _update_after_failed_exploit(self, vuln_key, asset_id=None):
        """
        Update beliefs more aggressively after failed exploit attempts.

        Args:
            vuln_key: Vulnerability key in format 'cve_id:asset_id:component_id'
            asset_id: Optional ID of the targeted asset
        """
        if self.belief_state and vuln_key in self.belief_state.vuln_patched_prob:
            # Get current beliefs
            current_patch_belief = self.belief_state.vuln_patched_prob[vuln_key]
            current_exploited_belief = self.belief_state.vuln_exploited_prob[vuln_key]

            # Update patch belief (increase likelihood of being patched)
            new_patch_belief = min(current_patch_belief + 0.3, 0.9)

            # Update exploited belief (decrease likelihood of being exploited)
            new_exploited_belief = max(current_exploited_belief - 0.1, 0.0)

            self.belief_state.vuln_patched_prob[vuln_key] = new_patch_belief
            self.belief_state.vuln_exploited_prob[vuln_key] = new_exploited_belief
            logger.info(
                f"Updated beliefs for {vuln_key}: patch={new_patch_belief:.3f}, exploited={new_exploited_belief:.3f}")

    def update_belief(self, observation):
        """
        Update belief state based on new observation.

        Args:
            observation: Dictionary containing observation details
        """
        self.observation_history.append(observation)

        # Update belief state with observation
        if self.belief_state:
            self.belief_state.update_from_observation(observation)

        # Update exploit outcome history and failure counts
        action_type = observation.get('action_type')
        target_vuln = observation.get('target_vuln')
        target_asset = observation.get('target_asset')
        target_component = observation.get('target_component', '0')
        success = observation.get('action_result', False)

        if action_type in ['exploit', 'initial_access', 'exploitation', 'lateral_movement',
                           'privilege_escalation'] and target_vuln:
            vuln_key = f"{target_vuln}:{target_asset}:{target_component}"
            self.exploit_outcomes[vuln_key] = success
            if success:
                self.exploited_vulnerabilities.add(vuln_key)
                # Reset failure count on success
                if vuln_key in self.action_asset_failures[action_type]:
                    self.action_asset_failures[action_type][str(target_asset)] = 0
                    logger.debug(f"Reset failure count for {action_type} on asset {target_asset}")
            else:
                # Increment failure count only here
                self._update_after_failed_exploit(vuln_key, target_asset)
                self.action_asset_failures[action_type][str(target_asset)] += 1
                logger.debug(f"Incremented failure count for {action_type} on asset {target_asset}: "
                             f"{self.action_asset_failures[action_type][str(target_asset)]}")
        elif action_type and target_asset:
            if success:
                # Reset failure count on success
                self.action_asset_failures[action_type][str(target_asset)] = 0
                logger.debug(f"Reset failure count for {action_type} on asset {target_asset}")
            else:
                # Increment failure count for non-vulnerability actions
                self.action_asset_failures[action_type][str(target_asset)] += 1
                logger.debug(f"Incremented failure count for {action_type} on asset {target_asset}: "
                             f"{self.action_asset_failures[action_type][str(target_asset)]}")

    def _adjust_for_patch_belief(self, vuln, probability):
        """
        Adjust probability based on belief about vulnerability patch status.

        Args:
            vuln: Vulnerability object
            probability: Base probability to adjust

        Returns:
            float: Adjusted probability
        """
        vuln_key = None
        # Find asset and component to construct vuln_key
        for obs in self.observation_history:
            if hasattr(obs, 'system') and obs.system:
                for asset in obs.system.assets:
                    for comp in asset.components:
                        if vuln in comp.vulnerabilities:
                            vuln_key = f"{vuln.cve_id}:{asset.asset_id}:{comp.id}"
                            break
                    if vuln_key:
                        break
        if vuln_key and self.belief_state:
            patch_prob = self.belief_state.get_patch_belief(vuln_key)
            probability *= (1.0 - patch_prob)
            logger.debug(f"Adjusted probability for {vuln_key} by patch belief: {patch_prob:.3f}")
        return probability

    def _apply_detection_aversion(self, probability):
        if self.detection_averse and self.belief_state:
            detection_prob = self.belief_state.detection_prob
            if detection_prob > 0.7:
                if probability < 0.7:
                    probability *= 0.5
            elif detection_prob > 0.4:
                if probability < 0.5:
                    probability *= 0.8
        return probability

    def _apply_prior_attempt_adjustment(self, vuln, probability):
        if vuln.cve_id in self.exploit_outcomes:
            if self.exploit_outcomes[vuln.cve_id]:
                probability *= 1.2
            else:
                probability *= 0.5
        return probability

    def _apply_tactic_alignment(self, vuln, tactic_name, probability):
        if tactic_name and hasattr(vuln, 'mitre_techniques') and vuln.mitre_techniques:
            mitre_techniques_tuple = tuple(vuln.mitre_techniques)
            tactics = self._get_tactics_from_techniques(mitre_techniques_tuple)
            if tactics:
                if tactic_name in tactics:
                    probability = min(probability * 1.5, 1.0)
                else:
                    probability *= 0.2
            else:
                probability *= 0.3
        return probability

    def _apply_complexity_adjustment(self, vuln, probability):
        """Helper method to adjust probability based on vulnerability complexity."""
        # Consider vulnerability complexity factors in exploitability
        if hasattr(vuln, 'complexity'):
            if vuln.complexity == 'high':
                probability *= 0.7  # 30% reduction for high complexity
            elif vuln.complexity == 'low':
                probability *= 1.2  # 20% increase for low complexity

        # Consider if vulnerability requires privileges
        if hasattr(vuln, 'cvssV3Vector') and isinstance(vuln.cvssV3Vector, str):
            if "PR:H" in vuln.cvssV3Vector:  # High privileges required
                probability *= 0.6  # 40% reduction if high privileges required
            elif "PR:L" in vuln.cvssV3Vector:  # Low privileges required
                probability *= 0.8  # 20% reduction if low privileges required

        return probability

    def _find_asset_for_vuln(self, vuln):
        assets = []
        for state in self.observation_history:
            if hasattr(state, 'system') and state.system:
                for asset in state.system.assets:
                    for comp in asset.components:
                        if vuln in comp.vulnerabilities:
                            assets.append(asset)
        return assets

    def _get_tactic_by_name(self, tactic_name):
        """Helper method to get tactic object by name."""
        if hasattr(self, '_mitre_ttps_by_name'):
            return self._mitre_ttps_by_name.get(tactic_name)
        return None

    def _legacy_get_exploitability(self, vuln, tactic_name):
        """Legacy method for calculating exploitability when asset info is unavailable."""
        base_likelihood = self._get_base_exploit_probability(vuln)

        # Apply basic adjustments
        base_likelihood = self._adjust_for_patch_belief(vuln, base_likelihood)
        base_likelihood = self._apply_detection_aversion(base_likelihood)
        base_likelihood = self._apply_prior_attempt_adjustment(vuln, base_likelihood)
        base_likelihood = self._apply_tactic_alignment(vuln, tactic_name, base_likelihood)

        return min(max(base_likelihood, 0.0), 1.0)
    
    @lru_cache(maxsize=128)
    def _get_tactics_from_techniques(self, techniques_tuple):
        techniques = list(techniques_tuple)
        tactics = set()
        for technique in techniques:
            base_technique = technique.split('.')[0] if '.' in technique else technique
            tactic = self.technique_to_tactic.get(base_technique)
            if tactic:
                if isinstance(tactic, list):
                    tactics.update(tactic)
                else:
                    tactics.add(tactic)
            full_tactic = self.technique_to_tactic.get(technique)
            if full_tactic:
                if isinstance(full_tactic, list):
                    tactics.update(full_tactic)
                else:
                    tactics.add(full_tactic)
        return tactics

    def select_tactic(self, state: State, current_position: str = None) -> MitreTTP:
        """
        Select a tactic based on the current belief state and system state.
        """
        # Validate state
        if state is None or not hasattr(state, 'system') or state.system is None:
            logger.error("Invalid state or missing system in select_tactic")
            return self.get_fallback_tactic(KillChainStage.RECONNAISSANCE)

        # Initialize belief state only if not set
        if self.belief_state is None:
            logger.debug("Belief state not initialized, initializing with current system")
            self.initialize_belief_state(state.system)

        # Existing tactic selection logic
        previous_action_result = self.observation_history[-1].get('action_result',
                                                                  False) if self.observation_history else None
        current_stage = KillChainStage(state.k) if isinstance(state.k, int) else KillChainStage.RECONNAISSANCE
        recommended_stage = self.evaluate_apt3_kill_chain_progression(state, previous_action_result)
        if recommended_stage.value > current_stage.value:
            logger.info(f"Kill chain progression: {current_stage.name} -> {recommended_stage.name}")
            state.attacker_suggested_stage = recommended_stage
            self.kill_chain_stage = recommended_stage
            state.k = recommended_stage.value
            current_stage = recommended_stage

        relevant_tactics = ckc_to_mitre.get(current_stage, ["Initial Access"])
        if self.detection_averse:
            max_detection_prob = max(
                self.belief_state.get_detection_prob(asset.asset_id, current_stage)
                for asset in state.system.assets
            )
            if max_detection_prob > 0.9:
                logger.info(f"Pausing tactic selection: Detection probability {max_detection_prob:.2f} exceeds 0.9")
                return None
            elif max_detection_prob > 0.7:
                relevant_tactics = [t for t in relevant_tactics if self._get_tactic_detection_risk(t) < 0.5]
                if not relevant_tactics:
                    logger.info("No low-risk tactics available; pausing")
                    return None

        for tactic in list(self.tactic_cooldown.keys()):
            self.tactic_cooldown[tactic] -= 1
            if self.tactic_cooldown[tactic] <= 0:
                del self.tactic_cooldown[tactic]

        tactic_scores = defaultdict(float)
        for asset in state.system.assets:
            asset_detection_prob = self.belief_state.get_detection_prob(asset.asset_id, current_stage)
            for comp in asset.components:
                for vuln in comp.vulnerabilities:
                    if self.belief_state.is_likely_patched(vuln.cve_id):
                        continue
                    techniques = set(getattr(vuln, 'mitre_techniques', []))
                    apt3_techniques = set(APT3TacticMapping.get_preferred_techniques(current_stage))
                    if not techniques & apt3_techniques:
                        continue
                    for tactic in relevant_tactics:
                        score = self.get_exploitability_under_uncertainty(vuln, tactic, state,
                                                                          current_position)  # Pass current_position
                        cvss_weight = getattr(vuln, 'cvss', 5.0) / 10.0
                        has_exploit_factor = 1.5 if getattr(vuln, 'exploit', False) else 1.0
                        detection_penalty = 1.0 - asset_detection_prob if self.detection_averse else 1.0
                        tactic_scores[tactic] += score * cvss_weight * has_exploit_factor * detection_penalty

        for tactic in tactic_scores:
            if tactic in self.tactic_cooldown:
                tactic_scores[tactic] *= 0.1
                continue
            frustration = self.tactic_frustration.get(tactic, 0)
            if frustration >= 3:
                tactic_scores[tactic] *= 0.2
                self.tactic_cooldown[tactic] = 3
            elif frustration == 2:
                tactic_scores[tactic] *= 0.5
            elif frustration == 1:
                tactic_scores[tactic] *= 0.8

        valid_tactics = [(ttp, score) for tactic_name, score in tactic_scores.items()
                         if (ttp := self._mitre_ttps_by_name.get(tactic_name))]
        if not valid_tactics:
            possible_tactics = self._mitre_ttps_by_stage.get(current_stage, [])
            return random.choice(possible_tactics) if possible_tactics else self.get_fallback_tactic(current_stage)

        total_score = sum(score for _, score in valid_tactics)
        if total_score > 0:
            probabilities = [score / total_score for _, score in valid_tactics]
            chosen_idx = random.choices(range(len(valid_tactics)), weights=probabilities, k=1)[0]
            selected_tactic = valid_tactics[chosen_idx][0]
            logger.info(f"Selected tactic: {selected_tactic.name} (score: {valid_tactics[chosen_idx][1]:.2f})")
            return selected_tactic

        return self.get_fallback_tactic(current_stage)

    def _get_tactic_detection_risk(self, tactic: str) -> float:
        """
        Estimate detection risk for a tactic (placeholder; adjust based on MITRE data).
        """
        high_risk_tactics = ["Exfiltration", "Impact"]
        low_risk_tactics = ["Reconnaissance", "Discovery"]
        if tactic in high_risk_tactics:
            return 0.8
        elif tactic in low_risk_tactics:
            return 0.2
        return 0.5

    def get_fallback_tactic(self, kill_chain_stage):
        stage_to_fallback = {
            KillChainStage.RECONNAISSANCE: "Reconnaissance",
            KillChainStage.WEAPONIZATION: "Phishing",
            KillChainStage.DELIVERY: "Phishing",
            KillChainStage.EXPLOITATION: "Execution",
            KillChainStage.INSTALLATION: "Persistence",
            KillChainStage.COMMAND_AND_CONTROL: "Command and Control",
            KillChainStage.ACTIONS_ON_OBJECTIVES: "Collection"
        }
        fallback_name = stage_to_fallback.get(kill_chain_stage, "Phishing")
        for ttp in mitre_ttps:
            if ttp.name == fallback_name:
                logger.info(f"Using APT3 fallback tactic: {fallback_name}")
                return ttp
        return mitre_ttps[0]


    def update_tactic_frustration(self, tactic_name, success):
        """
        Update frustration levels for tactics based on success or failure.

        Args:
            tactic_name: Name of the tactic used
            success: Whether the tactic was successful
        """
        if success:
            # Reset frustration on success
            self.tactic_frustration[tactic_name] = 0
            logger.info(f"Tactic {tactic_name} succeeded - resetting frustration")
        else:
            # Increment frustration on failure
            current = self.tactic_frustration.get(tactic_name, 0)
            self.tactic_frustration[tactic_name] = current + 1
            logger.info(f"Tactic {tactic_name} failed - frustration level now {current + 1}")

    def select_target_under_uncertainty(self, state: State, tactic: MitreTTP):
        """
        Select target assets and vulnerabilities based on belief state.
        Modified to use vuln_key-based beliefs and exclude likely exploited vulnerabilities.
        For lateral movement, only considers vulnerabilities on compromised assets.

        Args:
            state: Current system state
            tactic: Selected MITRE tactic

        Returns:
            tuple: (target_asset, target_vuln, expected_value)
        """
        potential_targets = []

        # For lateral movement, only look at compromised assets (current position)
        if tactic.name in ["Lateral Movement", "Lateral Movement Techniques"]:
            target_assets = [asset for asset in state.system.assets if 
                           asset.is_compromised or (self.belief_state and self.belief_state.is_likely_compromised(asset.asset_id))]
            if not target_assets:
                logger.debug("No compromised assets for lateral movement target selection")
                return None, None, 0.0
        else:
            # For other tactics, look at all assets
            target_assets = state.system.assets

        for asset in target_assets:
            for comp in asset.components:
                for vuln in comp.vulnerabilities:
                    vuln_key = f"{vuln.cve_id}:{asset.asset_id}:{comp.id}"
                    # Skip if likely patched or exploited based on belief
                    if self.belief_state:
                        if self.belief_state.get_patch_belief(vuln_key) > 0.7:
                            continue
                        if self.belief_state.get_exploited_belief(vuln_key) > 0.7:
                            continue
                    exploit_prob = self.get_exploitability_under_uncertainty(vuln, tactic.name, state)
                    asset_value = getattr(asset, 'business_value', asset.criticality_level * 5000)
                    expected_value = exploit_prob * asset_value
                    potential_targets.append((asset, vuln, expected_value))

        if potential_targets:
            potential_targets.sort(key=lambda x: x[2], reverse=True)
            logger.debug(
                f"Selected target: asset {potential_targets[0][0].asset_id}, vuln {potential_targets[0][1].cve_id}, score {potential_targets[0][2]:.3f}")
            return potential_targets[0]

        logger.debug("No viable targets found for tactic")
        return None, None, 0.0

    def _get_base_exploit_probability(self, vuln):
        base_likelihood = getattr(vuln, 'exploitability', None)
        if base_likelihood is None and hasattr(vuln, 'exploit_likelihood'):
            base_likelihood = vuln.exploit_likelihood
        if base_likelihood is None and hasattr(vuln, 'epss'):
            base_likelihood = vuln.epss
        if base_likelihood is None:
            base_likelihood = getattr(vuln, 'cvss', 0) / 10.0
        return base_likelihood

    def evaluate_apt3_kill_chain_progression(self, state: State, previous_action_result):
        current_stage = KillChainStage(state.k) if isinstance(state.k, int) else state.k
        stage_techniques = APT3TacticMapping.get_preferred_techniques(current_stage)
        successful_techniques = [
            obs.get('target_vuln') for obs in self.observation_history[-10:]
            if obs.get('action_result', False) and obs.get('tactic') in
            [tactic for tactic, _ in APT3TacticMapping.APT3_PREFERRED_TTPS.get(current_stage, [])]
        ]
        if current_stage == KillChainStage.RECONNAISSANCE:
            recon_actions = [a for a in self.observation_history[-10:] if a.get('action_type') == 'reconnaissance']
            if len(recon_actions) >= 3 or len(successful_techniques) >= 2:
                logger.info(f"Progressing from RECONNAISSANCE to WEAPONIZATION: {len(recon_actions)} recon actions")
                return KillChainStage.WEAPONIZATION
        elif current_stage == KillChainStage.WEAPONIZATION:
            if any(tech in successful_techniques for tech in ['T1566.001', 'T1566.002']):
                logger.info("Progressing from WEAPONIZATION to DELIVERY: Spearphishing success")
                return KillChainStage.DELIVERY
        elif current_stage == KillChainStage.DELIVERY:
            delivery_actions = [a for a in self.observation_history[-10:] if a.get('action_type') in ['delivery', 'initial_access']]
            if len(delivery_actions) >= 2 or any(asset.is_compromised for asset in state.system.assets):
                logger.info(f"Progressing from DELIVERY to EXPLOITATION: {len(delivery_actions)} delivery actions")
                return KillChainStage.EXPLOITATION
        elif current_stage == KillChainStage.EXPLOITATION:
            if any(tech in successful_techniques for tech in ['T1003', 'T1068']):
                logger.info("Progressing from EXPLOITATION to INSTALLATION: Credential/privilege success")
                return KillChainStage.INSTALLATION
        elif current_stage == KillChainStage.INSTALLATION:
            install_actions = [a for a in self.observation_history[-10:] if a.get('action_type') in ['installation', 'persistence']]
            if len(install_actions) >= 2 or len([a for a in state.system.assets if a.is_compromised]) >= 2:
                logger.info(f"Progressing from INSTALLATION to COMMAND_AND_CONTROL: {len(install_actions)} install actions")
                return KillChainStage.COMMAND_AND_CONTROL
        elif current_stage == KillChainStage.COMMAND_AND_CONTROL:
            c2_actions = [a for a in self.observation_history[-10:] if a.get('action_type') == 'command_and_control']
            if len(c2_actions) >= 2 or len([a for a in state.system.assets if a.is_compromised]) >= 3:
                logger.info(f"Progressing from COMMAND_AND_CONTROL to ACTIONS_ON_OBJECTIVES: {len(c2_actions)} C2 actions")
                return KillChainStage.ACTIONS_ON_OBJECTIVES
        return current_stage

    def adjust_high_value_priority(self, action_result):
        """
        Dynamically adjusts the priority given to high-value assets based on past results.

        Args:
            action_result: Result of the latest action (success or failure)
        """
        if action_result:
            # If successful, slightly increase high-value asset priority
            self.high_value_priority = min(self.high_value_priority * 1.05, 2.0)
            logger.info(f"Successful action: Increasing high-value asset priority to {self.high_value_priority:.2f}")
        else:
            # If failed, slightly decrease high-value asset priority
            self.high_value_priority = max(self.high_value_priority * 0.95, 1.0)
            logger.info(f"Failed action: Decreasing high-value asset priority to {self.high_value_priority:.2f}")

    def perform_action(self, state: State, tactic: MitreTTP):
        """
        Execute an attack action based on the selected tactic and belief state.
        Modified to ensure all actions are based on specific vulnerabilities.

        Args:
            state: Current system state
            tactic: Selected MITRE tactic

        Returns:
            dict: Action result with observation
        """
        # Track this operation
        self.operation_count += 1

        if tactic is None:
            logger.info("No tactic selected, skipping attack action")
            return {'action_type': 'none', 'action_result': False}

        logger.info(f"Performing {tactic.name} attack action under uncertainty")

        # Get current kill chain stage
        if isinstance(state.k, int):
            try:
                current_stage = KillChainStage(state.k)
            except ValueError:
                current_stage = KillChainStage.RECONNAISSANCE
        else:
            current_stage = state.k

        # For lateral movement, use the specialized method that understands current position
        if tactic.name in ["Lateral Movement", "Lateral Movement Techniques"]:
            logger.info(f"Using specialized lateral movement method for {tactic.name}")
            action_result = self._perform_lateral_movement(state, tactic)
            if action_result:
                # Update belief state with this observation
                self.update_belief(action_result)
                # Update tactic frustration based on result
                self.update_tactic_frustration(tactic.name, action_result.get('action_result', False))
                # Adjust high-value asset priority based on result
                self.adjust_high_value_priority(action_result.get('action_result', False))
                # Store this action in history
                self.last_actions.append({
                    'tactic': tactic.name,
                    'result': action_result.get('action_result', False)
                })
                if len(self.last_actions) > 5:
                    self.last_actions = self.last_actions[-5:]
                return action_result

        # For all other stages, first find a suitable vulnerability to target
        target_asset, target_vuln, expected_value = self.select_target_under_uncertainty(state, tactic)

        # If we found a viable target with a specific vulnerability, check it's not already exploited
        if target_asset is not None and target_vuln is not None:
            if getattr(target_vuln, 'is_exploited', False) or target_vuln.cve_id in self.exploited_vulnerabilities:
                logger.info(
                    f"Selected vulnerability {target_vuln.cve_id} is already exploited, looking for another.")
                # Try to find another vulnerability
                for asset in state.system.assets:
                    for comp in asset.components:
                        for vuln in comp.vulnerabilities:
                            if not getattr(vuln, 'is_exploited',
                                           False) and vuln.cve_id not in self.exploited_vulnerabilities:
                                if (not self.belief_state or not self.belief_state.is_likely_patched(vuln.cve_id)):
                                    logger.info(f"Found alternative vulnerability {vuln.cve_id} on {asset.name}")
                                    return self._execute_general_tactic(state, tactic, asset, vuln,
                                                                        0.5)  # Use default expected value

                # If no alternative found, fall back to specialized methods
                logger.info(f"No unexploited vulnerabilities found for {tactic.name}")
            else:
                logger.info(f"Using vulnerability {target_vuln.cve_id} on {target_asset.name} for {tactic.name}")
                return self._execute_general_tactic(state, tactic, target_asset, target_vuln, expected_value)

        # If no viable vulnerability target found, use the specialized methods as fallback
        # but ensure they also try to find specific vulnerabilities
        action_result = None

        # RECONNAISSANCE stage - use specialized method that focuses on vulnerabilities
        if current_stage == KillChainStage.RECONNAISSANCE:
            action_result = self._perform_reconnaissance(state, tactic)

        # WEAPONIZATION stage
        elif current_stage == KillChainStage.WEAPONIZATION:
            action_result = self._perform_weaponization(state, tactic)

        # DELIVERY stage
        elif current_stage == KillChainStage.DELIVERY:
            action_result = self._perform_initial_access(state, tactic)

        # EXPLOITATION stage
        elif current_stage == KillChainStage.EXPLOITATION:
            action_result = self._perform_exploitation(state, tactic)

        # INSTALLATION stage
        elif current_stage == KillChainStage.INSTALLATION:
            action_result = self._perform_persistence(state, tactic)

        # COMMAND_AND_CONTROL stage
        elif current_stage == KillChainStage.COMMAND_AND_CONTROL:
            action_result = self._perform_command_and_control(state, tactic)

        # ACTIONS_ON_OBJECTIVES stage
        elif current_stage == KillChainStage.ACTIONS_ON_OBJECTIVES:
            if tactic.name in ["Impact", "Resource Hijacking", "Data Destruction"]:
                action_result = self._perform_impact(state, tactic)
            elif tactic.name in ["Exfiltration", "Data Exfiltration", "Transfer Data to Cloud Account"]:
                action_result = self._perform_exfiltration(state, tactic)
            else:
                # For other objectives, use collection
                action_result = self._perform_collection(state, tactic)

        # If no action result yet, try reconnaissance as a fallback
        if action_result is None:
            logger.info(f"No specific action for {tactic.name} in {current_stage.name}, falling back to reconnaissance")
            action_result = self._perform_reconnaissance(state, tactic)

        # Get action result
        is_successful = action_result.get('action_result', False)

        # Update belief state with this observation
        self.update_belief(action_result)

        # Update tactic frustration based on result
        self.update_tactic_frustration(tactic.name, is_successful)

        # Adjust high-value asset priority based on result
        self.adjust_high_value_priority(is_successful)

        # Store this action in history (keep only last 5)
        self.last_actions.append({
            'tactic': tactic.name,
            'result': is_successful
        })
        if len(self.last_actions) > 5:
            self.last_actions = self.last_actions[-5:]

        return action_result

    def _execute_general_tactic(self, state, tactic, target_asset, target_vuln, expected_value):
        """
        Execute a general tactic against a target.
        Modified to use vuln_key-based beliefs and include component_id.

        Args:
            state: Current system state
            tactic: Selected MITRE tactic
            target_asset: Target asset
            target_vuln: Target vulnerability
            expected_value: Expected value of the action

        Returns:
            dict: Action result with observation
        """
        # Find component_id for vuln_key
        component_id = None
        for comp in target_asset.components:
            if target_vuln in comp.vulnerabilities:
                component_id = comp.id
                break
        if not component_id:
            component_id = '0'
            logger.warning(
                f"No component found for {target_vuln.cve_id} on asset {target_asset.asset_id}, using default '0'")

        vuln_key = f"{target_vuln.cve_id}:{target_asset.asset_id}:{component_id}"

        # Check if likely exploited based on belief
        if self.belief_state and self.belief_state.get_exploited_belief(vuln_key) > 0.7:
            logger.info(f"Skipping likely exploited vulnerability {vuln_key}")
            return {
                'action_type': tactic.name.lower(),
                'target_asset': target_asset.asset_id,
                'target_vuln': target_vuln.cve_id,
                'target_component': component_id,
                'action_result': False,
                'reason': 'likely_exploited'
            }

        # Determine success probability
        success_prob = self.get_exploitability_under_uncertainty(target_vuln, tactic.name, state)

        # Determine if action is successful
        is_successful = random.random() < success_prob

        # Execute action
        if is_successful:
            logger.info(f"Successfully executed {tactic.name} on {target_asset.name} using {target_vuln.cve_id}")
            # Update asset and vuln state
            if tactic.name in ["Exploitation for Client Execution", "Exploit Public-Facing Application",
                               "Initial Access", "Lateral Movement"]:
                target_asset.mark_as_compromised(True)
                logger.info(f"Asset {target_asset.name} is now compromised")
        else:
            logger.info(f"Failed to execute {tactic.name} on {target_asset.name} using {target_vuln.cve_id}")

        # Create observation
        observation = {
            'action_type': tactic.name.lower(),
            'target_asset': target_asset.asset_id,
            'target_vuln': target_vuln.cve_id,
            'target_component': component_id,
            'action_result': is_successful,
            'expected_value': expected_value,
            'cvss': getattr(target_vuln, 'cvss', 'Unknown'),
            'epss': getattr(target_vuln, 'epss', 'Unknown'),
            'exploit': getattr(target_vuln, 'exploit', False),
            'ransomWare': getattr(target_vuln, 'ransomWare', False)
        }

        return observation

    def _perform_weaponization(self, state: State, tactic: MitreTTP):
        """
        Perform weaponization based on actual vulnerabilities in the system.
        Modified to ensure weaponization is tied to specific vulnerabilities.

        Args:
            state: Current system state
            tactic: Selected MITRE tactic

        Returns:
            dict: Action result and observation
        """
        # First, find a suitable vulnerability to weaponize against
        target_asset, target_vuln, expected_value = self.select_target_under_uncertainty(state, tactic)

        # If no suitable vulnerability found, return failure
        if target_asset is None or target_vuln is None:
            logger.info("No suitable vulnerabilities found for weaponization")
            return {
                'action_type': 'weaponization',
                'action_result': False,
                'reason': 'no_suitable_vulnerabilities'
            }

        logger.info(f"Performing weaponization for {target_vuln.cve_id} on {target_asset.name}")

        # Calculate success chance based on vulnerability and detection risk
        success_chance = self.get_exploitability_under_uncertainty(target_vuln, tactic.name)

        # Slightly lower success chance if detection probability is high
        if self.detection_averse and self.belief_state and self.belief_state.detection_prob > 0.7:
            success_chance *= 0.9

        # Determine success
        is_successful = random.random() < success_chance

        if is_successful:
            logger.info(f"Successfully weaponized exploit for {target_vuln.cve_id} on {target_asset.name}")
        else:
            logger.info(f"Failed to weaponize exploit for {target_vuln.cve_id} on {target_asset.name}")

        # Create observation with specific vulnerability information
        observation = {
            'action_type': 'weaponization',
            'tactic': tactic.name,
            'target_asset': target_asset.asset_id,
            'target_vuln': target_vuln.cve_id,
            'action_result': is_successful,
            'expected_value': expected_value,
            'detection_probability': self.belief_state.detection_prob if self.belief_state else None
        }

        return observation

    def _perform_exploitation(self, state: State, tactic: MitreTTP):
        """
        Perform exploitation using specific vulnerabilities.
        Modified to use vuln_key-based beliefs.

        Args:
            state: Current system state
            tactic: Selected MITRE tactic

        Returns:
            dict: Action result and observation
        """
        logger.info("Performing exploitation under uncertainty")

        # Find suitable targets
        best_asset = None
        best_vuln = None
        best_score = 0
        best_component_id = '0'

        for asset in state.system.assets:
            for comp in asset.components:
                for vuln in comp.vulnerabilities:
                    vuln_key = f"{vuln.cve_id}:{asset.asset_id}:{comp.id}"
                    if self.belief_state:
                        if self.belief_state.get_patch_belief(vuln_key) > 0.7:
                            continue
                        if self.belief_state.get_exploited_belief(vuln_key) > 0.7:
                            continue
                    exploit_score = self.get_exploitability_under_uncertainty(vuln, tactic.name, state)
                    asset_value = getattr(asset, 'business_value', asset.criticality_level * 5000)
                    total_score = exploit_score * asset_value
                    if total_score > best_score:
                        best_asset = asset
                        best_vuln = vuln
                        best_score = total_score
                        best_component_id = comp.id

        if best_asset is None or best_vuln is None:
            logger.info("No suitable vulnerabilities found for exploitation")
            return {
                'action_type': 'exploitation',
                'action_result': False,
                'reason': 'no_suitable_vulnerabilities'
            }

        success_chance = self.get_exploitability_under_uncertainty(best_vuln, tactic.name, state)
        if self.detection_averse and self.belief_state and self.belief_state.detection_prob > 0.6:
            success_chance *= 0.8

        is_successful = random.random() < success_chance

        if is_successful:
            logger.info(f"Successfully exploited {best_vuln.cve_id} on {best_asset.name}")
            best_asset.mark_as_compromised(True)
            if self.belief_state:
                vuln_key = f"{best_vuln.cve_id}:{best_asset.asset_id}:{best_component_id}"
                self.belief_state.vuln_patched_prob[vuln_key] = 0.0
                self.belief_state.vuln_exploited_prob[vuln_key] = 1.0
                self.belief_state.asset_compromised_prob[best_asset.asset_id] = 1.0
                detection_chance = 0.3
                detected = random.random() < detection_chance
                if detected:
                    detection_before = self.belief_state.detection_prob
                    self.belief_state.detection_prob = min(self.belief_state.detection_prob + 0.2, 1.0)
                    logger.info(
                        f"Exploitation detected, detection belief increased from {detection_before:.3f} to {self.belief_state.detection_prob:.3f}")
        else:
            logger.info(f"Failed to exploit {best_vuln.cve_id} on {best_asset.name}")

        observation = {
            'action_type': 'exploitation',
            'tactic': tactic.name,
            'target_asset': best_asset.asset_id,
            'target_vuln': best_vuln.cve_id,
            'target_component': best_component_id,
            'action_result': is_successful,
            'expected_value': best_score,
            'detection_probability': self.belief_state.detection_prob if self.belief_state else None,
            'cvss': getattr(best_vuln, 'cvss', 'Unknown'),
            'epss': getattr(best_vuln, 'epss', 'Unknown')
        }

        return observation

    def _perform_reconnaissance(self, state: State, tactic: MitreTTP):
        """
        Perform reconnaissance to gather information about vulnerabilities.
        Modified to use vuln_key-based beliefs.

        Args:
            state: Current system state
            tactic: Selected MITRE tactic

        Returns:
            dict: Action result and observation
        """
        logger.info("Performing reconnaissance under uncertainty")

        target_asset, target_vuln, expected_value = self.select_target_under_uncertainty(state, tactic)

        if target_asset is None or target_vuln is None:
            logger.info("No vulnerabilities available for reconnaissance")
            return {
                'action_type': 'reconnaissance',
                'action_result': False,
                'reason': 'no_vulnerabilities_available'
            }

        component_id = '0'
        for comp in target_asset.components:
            if target_vuln in comp.vulnerabilities:
                component_id = comp.id
                break

        vuln_key = f"{target_vuln.cve_id}:{target_asset.asset_id}:{component_id}"
        logger.info(f"Performing reconnaissance on {target_asset.name} focusing on {target_vuln.cve_id}")

        observation = {
            'action_type': 'reconnaissance',
            'target_asset': target_asset.asset_id,
            'target_vuln': target_vuln.cve_id,
            'target_component': component_id,
            'action_result': True,
            'observed_assets': [{
                'asset_id': target_asset.asset_id,
                'name': target_asset.name,
                'observed_vulnerabilities': [{
                    'cve_id': target_vuln.cve_id,
                    'cvss': getattr(target_vuln, 'cvss', 'Unknown'),
                    'is_patched': self.belief_state.get_patch_belief(vuln_key) > 0.7 if self.belief_state else False,
                    'is_exploited': self.belief_state.get_exploited_belief(
                        vuln_key) > 0.7 if self.belief_state else False
                }]
            }],
            'detection_probability': self.belief_state.detection_prob if self.belief_state else None
        }

        if self.belief_state:
            detection_chance = 0.2
            detected = random.random() < detection_chance
            if detected:
                detection_before = self.belief_state.detection_prob
                self.belief_state.detection_prob = min(self.belief_state.detection_prob + 0.1, 1.0)
                logger.info(
                    f"Reconnaissance detected, detection belief increased from {detection_before:.3f} to {self.belief_state.detection_prob:.3f}")

        return observation

    def _perform_initial_access(self, state: State, tactic: MitreTTP):
        """
        Attempt to gain initial access to the system under uncertainty.
        Modified to use vuln_key-based beliefs.

        Args:
            state: Current system state
            tactic: Selected MITRE tactic

        Returns:
            dict: Action result and observation
        """
        logger.info("Attempting initial access under uncertainty")

        # Find external-facing assets
        external_facing = set()
        for conn in state.system.connections:
            if conn.from_asset is None or (hasattr(conn.from_asset, 'type') and conn.from_asset.type == "External"):
                external_facing.add(conn.to_asset)

        entry_assets = [asset for asset in state.system.assets if
                        getattr(asset, 'type', '').lower() in ['entry', 'vpn server', 'web server']]
        target_assets = list(external_facing.union(entry_assets))

        if not target_assets:
            logger.info("No external-facing assets found for Initial Access")
            return {
                'action_type': 'initial_access',
                'action_result': False,
                'reason': 'no_targets'
            }

        if self.detection_averse and self.belief_state and self.belief_state.detection_prob > 0.7:
            target_assets = [a for a in target_assets if a.criticality_level >= 3]
            if not target_assets:
                logger.info("Avoiding attack due to high detection probability and low-value targets")
                return {
                    'action_type': 'initial_access',
                    'action_result': False,
                    'reason': 'high_detection_probability'
                }

        best_asset = None
        best_vuln = None
        best_score = 0
        best_component_id = '0'

        for asset in target_assets:
            for comp in asset.components:
                for vuln in comp.vulnerabilities:
                    vuln_key = f"{vuln.cve_id}:{asset.asset_id}:{comp.id}"
                    if self.belief_state:
                        if self.belief_state.get_patch_belief(vuln_key) > 0.7:
                            continue
                        if self.belief_state.get_exploited_belief(vuln_key) > 0.7:
                            continue
                    exploit_score = self.get_exploitability_under_uncertainty(vuln, tactic.name, state)
                    asset_value = getattr(asset, 'business_value', asset.criticality_level * 5000)
                    total_score = exploit_score * asset_value
                    if total_score > best_score:
                        best_asset = asset
                        best_vuln = vuln
                        best_score = total_score
                        best_component_id = comp.id

        if best_asset is None or best_vuln is None:
            logger.info("No suitable vulnerabilities found for Initial Access")
            return {
                'action_type': 'initial_access',
                'action_result': False,
                'reason': 'no_suitable_vulnerabilities'
            }

        success_chance = self.get_exploitability_under_uncertainty(best_vuln, tactic.name, state)
        success = random.random() < success_chance

        if success:
            logger.info(f"Successfully exploited {best_vuln.cve_id} on {best_asset.name} for Initial Access")
            if self.belief_state:
                vuln_key = f"{best_vuln.cve_id}:{best_asset.asset_id}:{best_component_id}"
                self.belief_state.vuln_patched_prob[vuln_key] = 0.0
                self.belief_state.vuln_exploited_prob[vuln_key] = 1.0
                self.belief_state.asset_compromised_prob[best_asset.asset_id] = 1.0
                detection_chance = 0.3
                detected = random.random() < detection_chance
                if detected:
                    detection_before = self.belief_state.detection_prob
                    self.belief_state.detection_prob = min(self.belief_state.detection_prob + 0.2, 1.0)
                    logger.info(
                        f"Initial access was detected, detection belief increased from {detection_before:.3f} to {self.belief_state.detection_prob:.3f}")
            return {
                'action_type': 'initial_access',
                'target_asset': best_asset.asset_id,
                'target_vuln': best_vuln.cve_id,
                'target_component': best_component_id,
                'action_result': True,
                'expected_score': best_score,
                'detection_probability': self.belief_state.detection_prob if self.belief_state else None
            }
        else:
            logger.info(f"Failed to exploit {best_vuln.cve_id} on {best_asset.name} for Initial Access")
            return {
                'action_type': 'initial_access',
                'target_asset': best_asset.asset_id,
                'target_vuln': best_vuln.cve_id,
                'target_component': best_component_id,
                'action_result': False,
                'expected_score': best_score
            }

    def _perform_lateral_movement(self, state: State, tactic: MitreTTP):
        """
        Perform lateral movement to connected assets.
        Modified to use vuln_key-based beliefs.

        Args:
            state: Current system state
            tactic: Selected MITRE tactic

        Returns:
            dict: Action result and observation
        """
        logger.info("Attempting lateral movement under uncertainty")

        # Find compromised assets (current position)
        compromised_assets = [asset for asset in state.system.assets if
                              asset.is_compromised or (self.belief_state and self.belief_state.is_likely_compromised(
                                  asset.asset_id))]
        if not compromised_assets:
            logger.info("No compromised assets for lateral movement")
            return {
                'action_type': 'lateral_movement',
                'action_result': False,
                'reason': 'no_compromised_assets'
            }

        best_source_asset = None
        best_target_asset = None
        best_vuln = None
        best_score = 0
        best_component_id = '0'

        for source_asset in compromised_assets:
            # Find connected assets (targets for movement)
            adjacent_assets = [conn.to_asset for conn in state.system.connections
                               if conn.from_asset == source_asset and conn.to_asset is not None]
            
            # For lateral movement, we need vulnerabilities on the CURRENT asset (source_asset)
            for comp in source_asset.components:
                for vuln in comp.vulnerabilities:
                    vuln_key = f"{vuln.cve_id}:{source_asset.asset_id}:{comp.id}"
                    if self.belief_state:
                        if self.belief_state.get_patch_belief(vuln_key) > 0.7:
                            continue
                        if self.belief_state.get_exploited_belief(vuln_key) > 0.7:
                            continue
                    
                    # Check if this vulnerability can be used to move to any adjacent asset
                    for target_asset in adjacent_assets:
                        exploit_score = self.get_exploitability_under_uncertainty(vuln, tactic.name, state)
                        asset_value = getattr(target_asset, 'business_value', target_asset.criticality_level * 5000)
                        total_score = exploit_score * asset_value
                        if total_score > best_score:
                            best_source_asset = source_asset
                            best_target_asset = target_asset
                            best_vuln = vuln
                            best_score = total_score
                            best_component_id = comp.id

        if best_source_asset is None or best_target_asset is None or best_vuln is None:
            logger.info("No suitable vulnerabilities found for lateral movement")
            return {
                'action_type': 'lateral_movement',
                'action_result': False,
                'reason': 'no_suitable_vulnerabilities'
            }

        success_chance = self.get_exploitability_under_uncertainty(best_vuln, tactic.name, state)
        is_successful = random.random() < success_chance

        if is_successful:
            logger.info(f"Successfully moved laterally from {best_source_asset.name} to {best_target_asset.name} using {best_vuln.cve_id}")
            best_target_asset.mark_as_compromised(True)
            if self.belief_state:
                vuln_key = f"{best_vuln.cve_id}:{best_source_asset.asset_id}:{best_component_id}"
                self.belief_state.vuln_patched_prob[vuln_key] = 0.0
                self.belief_state.vuln_exploited_prob[vuln_key] = 1.0
                self.belief_state.asset_compromised_prob[best_target_asset.asset_id] = 1.0
                detection_chance = 0.3
                detected = random.random() < detection_chance
                if detected:
                    detection_before = self.belief_state.detection_prob
                    self.belief_state.detection_prob = min(self.belief_state.detection_prob + 0.2, 1.0)
                    logger.info(
                        f"Lateral movement detected, detection belief increased from {detection_before:.3f} to {self.belief_state.detection_prob:.3f}")
        else:
            logger.info(f"Failed to move laterally from {best_source_asset.name} to {best_target_asset.name} using {best_vuln.cve_id}")

        if best_source_asset and best_target_asset and best_vuln:
            logger.info(f"[DIAG] POMDP selection: Vulnerabilities on asset {best_source_asset.asset_id}:")
            for comp in best_source_asset.components:
                for vuln in comp.vulnerabilities:
                    logger.info(f"[DIAG]   {vuln.cve_id} (patched={vuln.is_patched}, exploited={vuln.is_exploited})")
            logger.info(f"[DIAG] POMDP selected vuln: {best_vuln.cve_id} (for movement to {best_target_asset.asset_id})")

        return {
            'action_type': 'lateral_movement',
            'target_asset': best_target_asset.asset_id,
            'target_vuln': best_vuln.cve_id,
            'target_component': best_component_id,
            'action_result': is_successful,
            'expected_value': best_score,
            'detection_probability': self.belief_state.detection_prob if self.belief_state else None,
            'cvss': getattr(best_vuln, 'cvss', 'Unknown'),
            'epss': getattr(best_vuln, 'epss', 'Unknown')
        }

    def _perform_privilege_escalation(self, state: State, tactic: MitreTTP):
        """
        Attempt to escalate privileges on compromised assets under uncertainty.
        Modified to use vuln_key-based beliefs.

        Args:
            state: Current system state
            tactic: Selected MITRE tactic

        Returns:
            dict: Action result and observation
        """
        logger.info("Attempting privilege escalation under uncertainty")

        likely_compromised = [asset for asset in state.system.assets if
                              asset.is_compromised or (self.belief_state and self.belief_state.is_likely_compromised(
                                  asset.asset_id))]
        if not likely_compromised:
            logger.info("No compromised assets for privilege escalation")
            return {
                'action_type': 'privilege_escalation',
                'action_result': False,
                'reason': 'no_compromised_assets'
            }

        best_asset = None
        best_vuln = None
        best_score = 0
        best_component_id = '0'

        for asset in likely_compromised:
            for comp in asset.components:
                for vuln in comp.vulnerabilities:
                    vuln_key = f"{vuln.cve_id}:{asset.asset_id}:{comp.id}"
                    if self.belief_state:
                        if self.belief_state.get_patch_belief(vuln_key) > 0.7:
                            continue
                        if self.belief_state.get_exploited_belief(vuln_key) > 0.7:
                            continue
                    exploit_score = self.get_exploitability_under_uncertainty(vuln, tactic.name, state)
                    asset_value = getattr(asset, 'business_value', asset.criticality_level * 5000)
                    total_score = exploit_score * asset_value
                    if total_score > best_score:
                        best_asset = asset
                        best_vuln = vuln
                        best_score = total_score
                        best_component_id = comp.id

        if best_asset is None or best_vuln is None:
            logger.info("No suitable vulnerabilities found for privilege escalation")
            return {
                'action_type': 'privilege_escalation',
                'action_result': False,
                'reason': 'no_suitable_vulnerabilities'
            }

        if self.cost_aware:
            high_value_targets = [(a, v, s, c) for a, v, s, c in
                                  [(a, v, self.get_exploitability_under_uncertainty(v, tactic.name, state) * getattr(a,
                                                                                                                     'business_value',
                                                                                                                     a.criticality_level * 5000),
                                    c_id)
                                   for a in likely_compromised for c in a.components for v in c.vulnerabilities
                                   for c_id in [c.id]
                                   if not (self.belief_state and (self.belief_state.get_patch_belief(
                                      f"{v.cve_id}:{a.asset_id}:{c.id}") > 0.7 or
                                                                  self.belief_state.get_exploited_belief(
                                                                      f"{v.cve_id}:{a.asset_id}:{c.id}") > 0.7))]
                                  if a.criticality_level >= 4 and s >= 0.7 * best_score]
            if high_value_targets:
                best_asset, best_vuln, best_score, best_component_id = high_value_targets[0]
                logger.info(f"Prioritizing high-value asset {best_asset.name} for privilege escalation")

        success_chance = self.get_exploitability_under_uncertainty(best_vuln, tactic.name, state)
        is_successful = random.random() < success_chance

        if is_successful:
            logger.info(f"Successfully escalated privileges on {best_asset.name} using {best_vuln.cve_id}")
            if self.belief_state:
                vuln_key = f"{best_vuln.cve_id}:{best_asset.asset_id}:{best_component_id}"
                self.belief_state.vuln_patched_prob[vuln_key] = 0.0
                self.belief_state.vuln_exploited_prob[vuln_key] = 1.0
                detection_chance = 0.25
                detected = random.random() < detection_chance
                if detected:
                    detection_before = self.belief_state.detection_prob
                    self.belief_state.detection_prob = min(self.belief_state.detection_prob + 0.15, 1.0)
                    logger.info(
                        f"Privilege escalation detected, detection belief increased from {detection_before:.3f} to {self.belief_state.detection_prob:.3f}")
        else:
            logger.info(f"Failed to escalate privileges on {best_asset.name} using {best_vuln.cve_id}")

        return {
            'action_type': 'privilege_escalation',
            'target_asset': best_asset.asset_id,
            'target_vuln': best_vuln.cve_id,
            'target_component': best_component_id,
            'action_result': is_successful,
            'expected_score': best_score,
            'detection_probability': self.belief_state.detection_prob if self.belief_state else None,
            'cvss': getattr(best_vuln, 'cvss', 'Unknown'),
            'epss': getattr(best_vuln, 'epss', 'Unknown')
        }

    def _perform_defense_evasion(self, state: State, tactic: MitreTTP):
        """
        Attempt to evade defensive measures.
        Adjusts tactics based on belief about detection probability.

        Args:
            state: Current system state
            tactic: Selected MITRE tactic

        Returns:
            dict: Action result and observation
        """
        # Step 1: Log the action for debugging
        logger.info("Performing defense evasion actions under uncertainty")

        # Step 2: Choose appropriate evasion approach based on detection probability
        if self.belief_state and self.belief_state.detection_prob > 0.5:
            # High detection probability requires more advanced evasion
            logger.info("Taking advanced evasion measures due to high detection probability")
            evasion_method = "advanced"

            # Step 3: Calculate success probability based on detection probability
            # Advanced evasion has higher cost but better chance of reducing detection
            # Success probability decreases as detection probability increases
            # (harder to evade as defender becomes more aware)
            success_chance = max(0.3, 0.8 - self.belief_state.detection_prob)
        else:
            # Low detection probability allows for basic evasion
            logger.info("Taking basic evasion measures due to low detection probability")
            evasion_method = "basic"

            # Step 4: Basic evasion has high success rate but smaller effect
            success_chance = 0.9

        # Step 5: Attempt defense evasion
        success = random.random() < success_chance

        # Step 6: Handle successful evasion
        if success:
            # Step 7: Update detection belief if successful
            if self.belief_state:
                # Store original for reporting
                detection_before = self.belief_state.detection_prob

                # Step 8: Reduce detection probability based on evasion method
                if evasion_method == "advanced":
                    self.belief_state.detection_prob *= 0.5  # Reduce by half
                else:
                    self.belief_state.detection_prob *= 0.8  # Reduce by 20%

                logger.info(
                    f"Successfully evaded detection, belief reduced from {detection_before:.2f} to {self.belief_state.detection_prob:.2f}")

            # Step 9: Return successful action result
            return {
                'action_type': 'defense_evasion',
                'evasion_method': evasion_method,
                'action_result': True,
                'detection_before': detection_before if 'detection_before' in locals() else None,
                'detection_after': self.belief_state.detection_prob if self.belief_state else None
            }
        # Step 10: Handle failed evasion
        else:
            # Step 11: Update detection belief if evasion failed
            if self.belief_state:
                # Store original for reporting
                detection_before = self.belief_state.detection_prob

                # Step 12: Increase detection probability (evasion attempt was noticed)
                self.belief_state.detection_prob = min(self.belief_state.detection_prob * 1.2, 1.0)

                logger.info(
                    f"Failed to evade detection, belief increased from {detection_before:.2f} to {self.belief_state.detection_prob:.2f}")

            # Step 13: Return failed action result
            return {
                'action_type': 'defense_evasion',
                'evasion_method': evasion_method,
                'action_result': False,
                'detection_before': detection_before if 'detection_before' in locals() else None,
                'detection_after': self.belief_state.detection_prob if self.belief_state else None
            }

    def _perform_credential_access(self, state: State, tactic: MitreTTP):
        """
        Attempt to steal credentials from compromised assets.
        Modified to use vuln_key-based beliefs.

        Args:
            state: Current system state
            tactic: Selected MITRE tactic

        Returns:
            dict: Action result and observation
        """
        logger.info("Attempting credential access under uncertainty")

        likely_compromised = [asset for asset in state.system.assets if
                              asset.is_compromised or (self.belief_state and self.belief_state.is_likely_compromised(
                                  asset.asset_id))]
        if not likely_compromised:
            logger.info("No compromised assets for credential access")
            return {
                'action_type': 'credential_access',
                'action_result': False,
                'reason': 'no_compromised_assets'
            }

        best_asset = None
        best_vuln = None
        best_score = 0
        best_component_id = '0'

        for asset in likely_compromised:
            for comp in asset.components:
                for vuln in comp.vulnerabilities:
                    vuln_key = f"{vuln.cve_id}:{asset.asset_id}:{comp.id}"
                    if self.belief_state:
                        if self.belief_state.get_patch_belief(vuln_key) > 0.7:
                            continue
                        if self.belief_state.get_exploited_belief(vuln_key) > 0.7:
                            continue
                    exploit_score = self.get_exploitability_under_uncertainty(vuln, tactic.name, state)
                    asset_value = getattr(asset, 'business_value', asset.criticality_level * 5000)
                    total_score = exploit_score * asset_value
                    if total_score > best_score:
                        best_asset = asset
                        best_vuln = vuln
                        best_score = total_score
                        best_component_id = comp.id

        if best_asset and best_vuln:
            success_chance = self.get_exploitability_under_uncertainty(best_vuln, tactic.name, state)
            if self.detection_averse and self.belief_state and self.belief_state.detection_prob > 0.7:
                success_chance *= 0.7
            is_successful = random.random() < success_chance
            if is_successful:
                logger.info(f"Successfully accessed credentials on {best_asset.name} using {best_vuln.cve_id}")
                if self.belief_state:
                    vuln_key = f"{best_vuln.cve_id}:{best_asset.asset_id}:{best_component_id}"
                    self.belief_state.vuln_patched_prob[vuln_key] = 0.0
                    self.belief_state.vuln_exploited_prob[vuln_key] = 1.0
                    detection_chance = 0.3
                    detected = random.random() < detection_chance
                    if detected:
                        detection_before = self.belief_state.detection_prob
                        self.belief_state.detection_prob = min(self.belief_state.detection_prob + 0.2, 1.0)
                        logger.info(
                            f"Credential access detected, detection belief increased from {detection_before:.3f} to {self.belief_state.detection_prob:.3f}")
                return {
                    'action_type': 'credential_access',
                    'target_asset': best_asset.asset_id,
                    'target_vuln': best_vuln.cve_id,
                    'target_component': best_component_id,
                    'action_result': True,
                    'expected_score': best_score,
                    'detection_probability': self.belief_state.detection_prob if self.belief_state else None,
                    'cvss': getattr(best_vuln, 'cvss', 'Unknown'),
                    'epss': getattr(best_vuln, 'epss', 'Unknown')
                }
            else:
                logger.info(f"Failed to access credentials on {best_asset.name} using {best_vuln.cve_id}")
                return {
                    'action_type': 'credential_access',
                    'target_asset': best_asset.asset_id,
                    'target_vuln': best_vuln.cve_id,
                    'target_component': best_component_id,
                    'action_result': False,
                    'expected_score': best_score
                }

        # Fallback to general access if no suitable vulnerability
        for asset in likely_compromised:
            success_chance = 0.5 + (asset.criticality_level / 20.0)
            if self.detection_averse and self.belief_state and self.belief_state.detection_prob > 0.7:
                success_chance *= 0.7
            is_successful = random.random() < success_chance
            if is_successful:
                logger.info(f"Successfully accessed credentials on {asset.name} (general access)")
                if self.belief_state:
                    detection_chance = 0.3
                    detected = random.random() < detection_chance
                    if detected:
                        detection_before = self.belief_state.detection_prob
                        self.belief_state.detection_prob = min(self.belief_state.detection_prob + 0.2, 1.0)
                        logger.info(
                            f"Credential access detected, detection belief increased from {detection_before:.3f} to {self.belief_state.detection_prob:.3f}")
                return {
                    'action_type': 'credential_access',
                    'target_asset': asset.asset_id,
                    'method': 'general_access',
                    'action_result': True,
                    'detection_probability': self.belief_state.detection_prob if self.belief_state else None
                }

        logger.info("Failed to access credentials on any compromised asset")
        return {
            'action_type': 'credential_access',
            'action_result': False,
            'reason': 'access_failed'
        }

    def _perform_execution(self, state: State, tactic: MitreTTP):
        """
        Attempt to execute malicious code on compromised assets.
        Modified to use vuln_key-based beliefs.

        Args:
            state: Current system state
            tactic: Selected MITRE tactic

        Returns:
            dict: Action result and observation
        """
        logger.info("Attempting code execution under uncertainty")

        likely_compromised = [asset for asset in state.system.assets if
                              asset.is_compromised or (self.belief_state and self.belief_state.is_likely_compromised(
                                  asset.asset_id))]
        if not likely_compromised:
            logger.info("No compromised assets for code execution")
            return {
                'action_type': 'execution',
                'action_result': False,
                'reason': 'no_compromised_assets'
            }

        likely_compromised.sort(key=lambda a: a.criticality_level, reverse=True)
        high_value_targets = [a for a in likely_compromised if a.criticality_level >= 3]
        target_assets = high_value_targets if high_value_targets else likely_compromised

        if self.detection_averse and self.belief_state and self.belief_state.detection_prob > 0.7:
            target_assets = [a for a in likely_compromised if a.criticality_level >= 4] or high_value_targets

        for asset in target_assets:
            best_vuln = None
            best_score = 0
            best_component_id = '0'

            for comp in asset.components:
                for vuln in comp.vulnerabilities:
                    vuln_key = f"{vuln.cve_id}:{asset.asset_id}:{comp.id}"
                    if self.belief_state:
                        if self.belief_state.get_patch_belief(vuln_key) > 0.7:
                            continue
                        if self.belief_state.get_exploited_belief(vuln_key) > 0.7:
                            continue
                    score = self.get_exploitability_under_uncertainty(vuln, tactic.name, state)
                    if score > best_score:
                        best_vuln = vuln
                        best_score = score
                        best_component_id = comp.id

            if best_vuln:
                success_chance = best_score
                if self.detection_averse and self.belief_state and self.belief_state.detection_prob > 0.7:
                    success_chance *= 0.7
                is_successful = random.random() < success_chance
                if is_successful:
                    logger.info(f"Successfully executed code on {asset.name} using {best_vuln.cve_id}")
                    if self.belief_state:
                        vuln_key = f"{best_vuln.cve_id}:{asset.asset_id}:{best_component_id}"
                        self.belief_state.vuln_patched_prob[vuln_key] = 0.0
                        self.belief_state.vuln_exploited_prob[vuln_key] = 1.0
                        detection_chance = 0.3
                        detected = random.random() < detection_chance
                        if detected:
                            detection_before = self.belief_state.detection_prob
                            self.belief_state.detection_prob = min(self.belief_state.detection_prob + 0.2, 1.0)
                            logger.info(
                                f"Code execution detected, detection belief increased from {detection_before:.3f} to {self.belief_state.detection_prob:.3f}")
                    return {
                        'action_type': 'execution',
                        'target_asset': asset.asset_id,
                        'target_vuln': best_vuln.cve_id,
                        'target_component': best_component_id,
                        'action_result': True,
                        'expected_score': best_score,
                        'detection_probability': self.belief_state.detection_prob if self.belief_state else None,
                        'cvss': getattr(best_vuln, 'cvss', 'Unknown'),
                        'epss': getattr(best_vuln, 'epss', 'Unknown')
                    }
                else:
                    logger.info(f"Failed to execute code on {asset.name} using {best_vuln.cve_id}")
                    return {
                        'action_type': 'execution',
                        'target_asset': asset.asset_id,
                        'target_vuln': best_vuln.cve_id,
                        'target_component': best_component_id,
                        'action_result': False,
                        'expected_score': best_score
                    }

            if asset.is_compromised:
                success_chance = 0.7
                if self.detection_averse and self.belief_state and self.belief_state.detection_prob > 0.7:
                    success_chance *= 0.7
                is_successful = random.random() < success_chance
                if is_successful:
                    logger.info(f"Successfully executed code on {asset.name} (direct execution)")
                    if self.belief_state:
                        detection_chance = 0.25
                        detected = random.random() < detection_chance
                        if detected:
                            detection_before = self.belief_state.detection_prob
                            self.belief_state.detection_prob = min(self.belief_state.detection_prob + 0.15, 1.0)
                            logger.info(
                                f"Direct code execution detected, detection belief increased from {detection_before:.3f} to {self.belief_state.detection_prob:.3f}")
                    return {
                        'action_type': 'execution',
                        'target_asset': asset.asset_id,
                        'method': 'direct_execution',
                        'action_result': True,
                        'detection_probability': self.belief_state.detection_prob if self.belief_state else None
                    }

        logger.info("Failed to execute code on any target asset")
        return {
            'action_type': 'execution',
            'action_result': False,
            'reason': 'execution_failed'
        }

    def _perform_impact(self, state: State, tactic: MitreTTP):
        """
        Attempt to cause negative impact on compromised assets.
        Modified to use vuln_key-based beliefs.

        Args:
            state: Current system state
            tactic: Selected MITRE tactic

        Returns:
            dict: Action result and observation
        """
        logger.info("Attempting impact operations under uncertainty")

        best_asset = None
        best_vuln = None
        best_score = 0
        best_component_id = '0'

        for asset in state.system.assets:
            if not (asset.is_compromised or (
                    self.belief_state and self.belief_state.is_likely_compromised(asset.asset_id))):
                continue
            for comp in asset.components:
                for vuln in comp.vulnerabilities:
                    vuln_key = f"{vuln.cve_id}:{asset.asset_id}:{comp.id}"
                    if self.belief_state:
                        if self.belief_state.get_patch_belief(vuln_key) > 0.7:
                            continue
                        if self.belief_state.get_exploited_belief(vuln_key) > 0.7:
                            continue
                    exploit_score = self.get_exploitability_under_uncertainty(vuln, tactic.name, state)
                    asset_value = getattr(asset, 'business_value', asset.criticality_level * 5000)
                    total_score = exploit_score * asset_value
                    if total_score > best_score:
                        best_asset = asset
                        best_vuln = vuln
                        best_score = total_score
                        best_component_id = comp.id

        if best_asset is None or best_vuln is None:
            logger.info("No suitable vulnerabilities found for impact operations")
            return {
                'action_type': 'impact',
                'action_result': False,
                'reason': 'no_suitable_vulnerabilities'
            }

        success_chance = self.get_exploitability_under_uncertainty(best_vuln, tactic.name, state)
        if self.detection_averse and self.belief_state and self.belief_state.detection_prob > 0.7:
            success_chance *= 0.6

        is_successful = random.random() < success_chance

        if is_successful:
            has_ransomware = getattr(best_vuln, 'ransomWare', False)
            if has_ransomware:
                impact_type = "ransomware"
                severity = random.uniform(0.7, 1.0)
            else:
                impact_types = ["data_destruction", "service_disruption", "system_damage"]
                impact_type = random.choice(impact_types)
                severity = random.uniform(0.3, 0.8)
            impact_value = best_asset.criticality_level * severity * 5000
            logger.info(f"Successfully caused {impact_type} impact on {best_asset.name} using {best_vuln.cve_id}")
            if self.belief_state:
                vuln_key = f"{best_vuln.cve_id}:{best_asset.asset_id}:{best_component_id}"
                self.belief_state.vuln_patched_prob[vuln_key] = 0.0
                self.belief_state.vuln_exploited_prob[vuln_key] = 1.0
                detection_before = self.belief_state.detection_prob
                self.belief_state.detection_prob = min(self.belief_state.detection_prob + 0.5, 1.0)
                logger.info(
                    f"Impact operations detected, detection belief increased from {detection_before:.3f} to {self.belief_state.detection_prob:.3f}")
        else:
            logger.info(f"Failed to cause impact using {best_vuln.cve_id} on {best_asset.name}")
            impact_type = "none"
            severity = 0.0
            impact_value = 0.0

        return {
            'action_type': 'impact',
            'target_asset': best_asset.asset_id,
            'target_vuln': best_vuln.cve_id,
            'target_component': best_component_id,
            'impact_type': impact_type,
            'severity': severity,
            'impact_value': impact_value,
            'action_result': is_successful,
            'expected_value': best_score,
            'detection_probability': self.belief_state.detection_prob if self.belief_state else None,
            'cvss': getattr(best_vuln, 'cvss', 'Unknown'),
            'epss': getattr(best_vuln, 'epss', 'Unknown')
        }

    def _perform_persistence(self, state: State, tactic: MitreTTP):
        """
        Attempt to establish persistence on compromised assets under uncertainty.
        Modified to use vuln_key-based beliefs.

        Args:
            state: Current system state
            tactic: Selected MITRE tactic

        Returns:
            dict: Action result and observation
        """
        logger.info("Attempting persistence under uncertainty")

        best_asset = None
        best_vuln = None
        best_score = 0
        best_component_id = '0'

        for asset in state.system.assets:
            if not (asset.is_compromised or (
                    self.belief_state and self.belief_state.is_likely_compromised(asset.asset_id))):
                continue
            for comp in asset.components:
                for vuln in comp.vulnerabilities:
                    vuln_key = f"{vuln.cve_id}:{asset.asset_id}:{comp.id}"
                    if self.belief_state:
                        if self.belief_state.get_patch_belief(vuln_key) > 0.7:
                            continue
                        if self.belief_state.get_exploited_belief(vuln_key) > 0.7:
                            continue
                    exploit_score = self.get_exploitability_under_uncertainty(vuln, tactic.name, state)
                    asset_value = getattr(asset, 'business_value', asset.criticality_level * 5000)
                    total_score = exploit_score * asset_value
                    if total_score > best_score:
                        best_asset = asset
                        best_vuln = vuln
                        best_score = total_score
                        best_component_id = comp.id

        if best_asset is None or best_vuln is None:
            logger.info("No suitable vulnerabilities found for persistence")
            return {
                'action_type': 'persistence',
                'action_result': False,
                'reason': 'no_suitable_vulnerabilities'
            }

        success_chance = self.get_exploitability_under_uncertainty(best_vuln, tactic.name, state)
        if self.detection_averse and self.belief_state and self.belief_state.detection_prob > 0.7:
            success_chance *= 0.8

        is_successful = random.random() < success_chance

        if is_successful:
            logger.info(f"Successfully established persistence using {best_vuln.cve_id} on {best_asset.name}")
            method = "exploit-based_persistence" if getattr(best_vuln, 'exploit',
                                                            False) else "vulnerability_persistence"
            if self.belief_state:
                vuln_key = f"{best_vuln.cve_id}:{best_asset.asset_id}:{best_component_id}"
                self.belief_state.vuln_patched_prob[vuln_key] = 0.0
                self.belief_state.vuln_exploited_prob[vuln_key] = 1.0
                detection_chance = 0.3
                detected = random.random() < detection_chance
                if detected:
                    detection_before = self.belief_state.detection_prob
                    self.belief_state.detection_prob = min(self.belief_state.detection_prob + 0.2, 1.0)
                    logger.info(
                        f"Persistence detected, detection belief increased from {detection_before:.3f} to {self.belief_state.detection_prob:.3f}")
        else:
            logger.info(f"Failed to establish persistence using {best_vuln.cve_id} on {best_asset.name}")
            method = "none"

        return {
            'action_type': 'persistence',
            'target_asset': best_asset.asset_id,
            'target_vuln': best_vuln.cve_id,
            'target_component': best_component_id,
            'method': method,
            'action_result': is_successful,
            'expected_value': best_score,
            'detection_probability': self.belief_state.detection_prob if self.belief_state else None,
            'cvss': getattr(best_vuln, 'cvss', 'Unknown'),
            'epss': getattr(best_vuln, 'epss', 'Unknown')
        }

    def _perform_discovery(self, state: State, tactic: MitreTTP):
        """
        Discover information about the network and systems.
        Updates belief state with new information.

        Args:
            state: Current system state
            tactic: Selected MITRE tactic

        Returns:
            dict: Action result and observation
        """
        # Step 1: Log the action for debugging
        logger.info("Performing discovery actions under uncertainty")

        # Step 2: Find assets that are likely compromised
        likely_compromised = []
        for asset in state.system.assets:
            if asset.is_compromised or (self.belief_state and self.belief_state.is_likely_compromised(asset.asset_id)):
                likely_compromised.append(asset)

        # Step 3: Check if there are any compromised assets
        if not likely_compromised:
            logger.info("No compromised assets for Discovery")
            return {
                'action_type': 'discovery',
                'action_result': False,
                'reason': 'no_compromised_assets'
            }

        # Step 4: Determine discovery method based on detection probability
        if self.detection_averse and self.belief_state and self.belief_state.detection_prob > 0.7:
            # Use stealthier discovery when detection probability is high
            discovery_method = "passive"
            detection_chance = 0.15  # Lower detection chance
            observation_accuracy = 0.8  # Lower accuracy
            logger.info("Using passive discovery methods due to high detection probability")
        else:
            # Use more active discovery methods when detection probability is low
            discovery_method = "active"
            detection_chance = 0.25  # Higher detection chance
            observation_accuracy = 0.9  # Higher accuracy
            logger.info("Using active discovery methods")

        # Step 5: Initialize container for discovered information
        discovered_info = []

        # Step 6: For each compromised asset, discover connected assets
        for source_asset in likely_compromised:
            # Step 7: Find connected assets
            adjacent_assets = [conn.to_asset for conn in state.system.connections
                               if conn.from_asset == source_asset and conn.to_asset is not None]

            # Step 8: Discover information about adjacent assets
            for target_asset in adjacent_assets:
                # Step 9: Create asset information dictionary
                asset_info = {
                    'asset_id': target_asset.asset_id,
                    'name': target_asset.name,
                    'criticality': target_asset.criticality_level,
                    'discovery_method': discovery_method,
                    'vulns': []
                }

                # Step 10: Calculate observation probability based on discovery method
                if discovery_method == "active":
                    observe_prob = 0.7  # 70% chance to observe each vulnerability
                else:
                    observe_prob = 0.5  # 50% chance to observe each vulnerability

                # Step 11: Discover vulnerabilities on the asset
                for comp in target_asset.components:
                    for vuln in comp.vulnerabilities:
                        # Step 12: Try to observe each vulnerability with calculated probability
                        if random.random() < observe_prob:
                            # Step 13: Observation has a chance of error based on accuracy
                            observed_is_patched = vuln.is_patched
                            error_chance = 1.0 - observation_accuracy
                            if random.random() < error_chance:
                                observed_is_patched = not observed_is_patched

                                # At Step 14: Add to observed vulnerabilities list
                                asset_info['vulns'].append({
                                    'cve_id': vuln.cve_id,
                                    'is_patched': observed_is_patched,
                                    'is_exploited': getattr(vuln, 'is_exploited', False),
                                    # Add this line to track exploited status
                                    'cvss': vuln.cvss
                                })

                            # Step 15: Update belief state based on observation
                            if self.belief_state:
                                # Set confidence in patch status based on observation accuracy
                                patch_prob = observation_accuracy if observed_is_patched else (
                                            1.0 - observation_accuracy)
                                self.belief_state.vuln_patched_prob[vuln.cve_id] = patch_prob

                                # Store reference to vulnerability object if not already stored
                                if vuln.cve_id not in self.belief_state._vuln_map:
                                    self.belief_state._vuln_map[vuln.cve_id] = vuln

                # Step 16: Add asset info to discovered information
                discovered_info.append(asset_info)

        # Step 17: Check if discovery was detected
        if self.detection_averse and self.belief_state:
            detected = random.random() < detection_chance

            # Step 18: Update detection belief if detected
            if detected:
                detection_before = self.belief_state.detection_prob
                detection_increase = 0.15 if discovery_method == "passive" else 0.25
                self.belief_state.detection_prob = min(self.belief_state.detection_prob + detection_increase, 1.0)
                logger.info(
                    f"Discovery actions were detected, detection belief increased from {detection_before:.2f} to {self.belief_state.detection_prob:.2f}")

        # Step 19: Return discovery results
        if discovered_info:
            logger.info(f"Discovered information about {len(discovered_info)} assets")
            return {
                'action_type': 'discovery',
                'action_result': True,
                'discovery_method': discovery_method,
                'discovered_info': discovered_info,
                'detection_probability': self.belief_state.detection_prob if self.belief_state else None
            }
        else:
            logger.info("No valuable information discovered")
            return {
                'action_type': 'discovery',
                'action_result': False,
                'reason': 'no_information_discovered'
            }

    def _perform_collection(self, state: State, tactic: MitreTTP):
        """
        Collect valuable data from compromised assets.
        Uses belief state to guide targeting.

        Args:
            state: Current system state
            tactic: Selected MITRE tactic

        Returns:
            dict: Action result and observation
        """
        # Step 1: Log the action for debugging
        logger.info("Attempting data collection under uncertainty")

        # Step 2: Find assets that are likely compromised
        likely_compromised = []
        for asset in state.system.assets:
            if asset.is_compromised or (self.belief_state and self.belief_state.is_likely_compromised(asset.asset_id)):
                likely_compromised.append(asset)

        # Step 3: Check if there are any compromised assets
        if not likely_compromised:
            logger.info("No compromised assets for Collection")
            return {
                'action_type': 'collection',
                'action_result': False,
                'reason': 'no_compromised_assets'
            }

        # Step 4: Prioritize high-value assets for collection
        likely_compromised.sort(key=lambda a: a.criticality_level, reverse=True)
        high_value_targets = [a for a in likely_compromised if a.criticality_level >= 3]
        target_assets = high_value_targets if high_value_targets else likely_compromised

        logger.info(f"Collecting data from {len(target_assets)} compromised assets")

        # Step 5: Choose collection method based on detection probability
        if self.detection_averse and self.belief_state and self.belief_state.detection_prob > 0.7:
            # Use stealthier collection when detection probability is high
            logger.info("Using stealthy collection methods due to high detection probability")
            method = "stealthy"
            success_chance = 0.7  # Slightly lower success rate
            detection_chance = 0.2  # Lower detection chance
            collection_efficiency = 0.5  # Collect less data to avoid detection
        else:
            # Use standard collection methods
            logger.info("Using standard collection methods")
            method = "standard"
            success_chance = 0.9  # High success rate for collection
            detection_chance = 0.3  # Moderate detection chance
            collection_efficiency = 1.0  # Full collection efficiency

        # Step 6: Initialize container for collected data
        collected_data = []

        # Step 7: Attempt to collect data from each target asset
        for asset in target_assets:
            # Step 8: Skip low-value assets if cost-aware and in stealthy mode
            if self.cost_aware and method == "stealthy" and asset.criticality_level < 3:
                # Skip with 80% probability
                if random.random() < 0.8:
                    logger.info(f"Skipping low-value asset {asset.name} for stealthy collection (cost optimization)")
                    continue

            # Step 9: Attempt collection
            success = random.random() < success_chance

            # Step 10: Handle successful collection
            if success:
                # Step 11: Calculate data volume based on asset value and efficiency
                base_volume = random.randint(1, 3) * asset.criticality_level
                actual_volume = int(base_volume * collection_efficiency)

                logger.info(
                    f"Successfully collected {actual_volume} units of data from {asset.name} using {method} collection")

                # Step 12: Add to collected data list
                collected_data.append({
                    'asset_id': asset.asset_id,
                    'name': asset.name,
                    'method': method,
                    'volume': actual_volume,
                    'criticality': asset.criticality_level
                })

                # Step 13: Check if collection was detected
                detected = random.random() < detection_chance

                # Step 14: Update detection belief if detected
                if detected and self.belief_state:
                    detection_before = self.belief_state.detection_prob
                    self.belief_state.detection_prob = min(self.belief_state.detection_prob + 0.2, 1.0)
                    logger.info(
                        f"Collection from {asset.name} was detected, detection belief increased from {detection_before:.2f} to {self.belief_state.detection_prob:.2f}")
            # Step 15: Handle failed collection
            else:
                logger.info(f"Failed to collect data from {asset.name}")

        # Step 16: Return collection results
        if collected_data:
            return {
                'action_type': 'collection',
                'action_result': True,
                'collected_data': collected_data,
                'method': method,
                'total_volume': sum(data['volume'] for data in collected_data),
                'detection_probability': self.belief_state.detection_prob if self.belief_state else None
            }
        else:
            return {
                'action_type': 'collection',
                'action_result': False,
                'reason': 'collection_failed',
                'method': method
            }

    def _perform_command_and_control(self, state: State, tactic: MitreTTP):
        """
        Establish command and control channels with compromised assets.
        Modified to use vuln_key-based beliefs.

        Args:
            state: Current system state
            tactic: Selected MITRE tactic

        Returns:
            dict: Action result and observation
        """
        logger.info("Attempting command and control under uncertainty")

        best_asset = None
        best_vuln = None
        best_score = 0
        best_component_id = '0'

        for asset in state.system.assets:
            if not (asset.is_compromised or (
                    self.belief_state and self.belief_state.is_likely_compromised(asset.asset_id))):
                continue
            for comp in asset.components:
                for vuln in comp.vulnerabilities:
                    vuln_key = f"{vuln.cve_id}:{asset.asset_id}:{comp.id}"
                    if self.belief_state:
                        if self.belief_state.get_patch_belief(vuln_key) > 0.7:
                            continue
                        if self.belief_state.get_exploited_belief(vuln_key) > 0.7:
                            continue
                    exploit_score = self.get_exploitability_under_uncertainty(vuln, tactic.name, state)
                    asset_value = getattr(asset, 'business_value', asset.criticality_level * 5000)
                    total_score = exploit_score * asset_value
                    if total_score > best_score:
                        best_asset = asset
                        best_vuln = vuln
                        best_score = total_score
                        best_component_id = comp.id

        if best_asset is None or best_vuln is None:
            logger.info("No suitable vulnerabilities found for command and control")
            return {
                'action_type': 'command_and_control',
                'action_result': False,
                'reason': 'no_suitable_vulnerabilities'
            }

        success_chance = self.get_exploitability_under_uncertainty(best_vuln, tactic.name, state)
        if self.detection_averse and self.belief_state and self.belief_state.detection_prob > 0.7:
            success_chance *= 0.7
            architecture = "decentralized"
        else:
            architecture = "centralized"

        is_successful = random.random() < success_chance

        if is_successful:
            logger.info(f"Successfully established C2 channel using {best_vuln.cve_id} on {best_asset.name}")
            if self.belief_state:
                vuln_key = f"{best_vuln.cve_id}:{best_asset.asset_id}:{best_component_id}"
                self.belief_state.vuln_patched_prob[vuln_key] = 0.0
                self.belief_state.vuln_exploited_prob[vuln_key] = 1.0
                detection_chance = 0.35
                detected = random.random() < detection_chance
                if detected:
                    detection_before = self.belief_state.detection_prob
                    self.belief_state.detection_prob = min(self.belief_state.detection_prob + 0.3, 1.0)
                    logger.info(
                        f"C2 channel detected, detection belief increased from {detection_before:.3f} to {self.belief_state.detection_prob:.3f}")
        else:
            logger.info(f"Failed to establish C2 channel using {best_vuln.cve_id} on {best_asset.name}")

        return {
            'action_type': 'command_and_control',
            'target_asset': best_asset.asset_id,
            'target_vuln': best_vuln.cve_id,
            'target_component': best_component_id,
            'architecture': architecture,
            'action_result': is_successful,
            'expected_value': best_score,
            'detection_probability': self.belief_state.detection_prob if self.belief_state else None,
            'cvss': getattr(best_vuln, 'cvss', 'Unknown'),
            'epss': getattr(best_vuln, 'epss', 'Unknown')
        }

    def _perform_exfiltration(self, state: State, tactic: MitreTTP):
        """
        Exfiltrate collected data from compromised assets.
        Modified to use vuln_key-based beliefs.

        Args:
            state: Current system state
            tactic: Selected MITRE tactic

        Returns:
            dict: Action result and observation
        """
        logger.info("Attempting exfiltration under uncertainty")

        best_asset = None
        best_vuln = None
        best_score = 0
        best_component_id = '0'

        for asset in state.system.assets:
            if not (asset.is_compromised or (
                    self.belief_state and self.belief_state.is_likely_compromised(asset.asset_id))):
                continue
            for comp in asset.components:
                for vuln in comp.vulnerabilities:
                    vuln_key = f"{vuln.cve_id}:{asset.asset_id}:{comp.id}"
                    if self.belief_state:
                        if self.belief_state.get_patch_belief(vuln_key) > 0.7:
                            continue
                        if self.belief_state.get_exploited_belief(vuln_key) > 0.7:
                            continue
                    exploit_score = self.get_exploitability_under_uncertainty(vuln, tactic.name, state)
                    asset_value = getattr(asset, 'business_value', asset.criticality_level * 5000)
                    total_score = exploit_score * asset_value
                    if total_score > best_score:
                        best_asset = asset
                        best_vuln = vuln
                        best_score = total_score
                        best_component_id = comp.id

        if best_asset is None or best_vuln is None:
            logger.info("No suitable vulnerabilities found for exfiltration")
            return {
                'action_type': 'exfiltration',
                'action_result': False,
                'reason': 'no_suitable_vulnerabilities'
            }

        success_chance = self.get_exploitability_under_uncertainty(best_vuln, tactic.name, state)
        if self.detection_averse and self.belief_state and self.belief_state.detection_prob > 0.7:
            success_chance *= 0.6
            exfil_method = "stealthy"
        else:
            exfil_method = "standard"

        is_successful = random.random() < success_chance

        if is_successful:
            data_volume = random.randint(1, 5) * best_asset.criticality_level
            logger.info(
                f"Successfully exfiltrated {data_volume} units of data using {best_vuln.cve_id} on {best_asset.name}")
            if self.belief_state:
                vuln_key = f"{best_vuln.cve_id}:{best_asset.asset_id}:{best_component_id}"
                self.belief_state.vuln_patched_prob[vuln_key] = 0.0
                self.belief_state.vuln_exploited_prob[vuln_key] = 1.0
                detection_chance = 0.4 if exfil_method == "standard" else 0.2
                detected = random.random() < detection_chance
                if detected:
                    detection_before = self.belief_state.detection_prob
                    self.belief_state.detection_prob = min(self.belief_state.detection_prob + 0.3, 1.0)
                    logger.info(
                        f"Exfiltration detected, detection belief increased from {detection_before:.3f} to {self.belief_state.detection_prob:.3f}")
        else:
            logger.info(f"Failed to exfiltrate data using {best_vuln.cve_id} on {best_asset.name}")
            data_volume = 0

        return {
            'action_type': 'exfiltration',
            'target_asset': best_asset.asset_id,
            'target_vuln': best_vuln.cve_id,
            'target_component': best_component_id,
            'method': exfil_method,
            'volume': data_volume,
            'action_result': is_successful,
            'expected_value': best_score,
            'detection_probability': self.belief_state.detection_prob if self.belief_state else None,
            'cvss': getattr(best_vuln, 'cvss', 'Unknown'),
            'epss': getattr(best_vuln, 'epss', 'Unknown')
        }

    def select_action(self, state, available_actions):
        """Select action based on POMDP belief state, not actual patch status."""
        if not available_actions:
            return None

        # Filter actions based on belief state validation
        valid_actions = []
        for action in available_actions:
            action_type = action.get('action_type')
            
            # For impact actions, validate against belief state
            if action_type in ['data_destruction', 'ransomware', 'exfiltration']:
                target_vuln = action.get('target_vuln')
                target_asset = action.get('target_asset')
                
                if target_vuln and target_asset:
                    # Check belief state for this vulnerability
                    vuln_key = create_vuln_key(target_vuln, str(target_asset), str(action.get('target_component', '0')))
                    
                    # Add null check for belief_state
                    if self.belief_state is None:
                        logger.warning("Belief state is None, allowing impact action without validation")
                        valid_actions.append(action)
                        continue
                        
                    patch_belief = self.belief_state.get_patch_belief(vuln_key)
                    exploited_belief = self.belief_state.get_exploited_belief(vuln_key)
                    
                    # Only allow impact actions on vulnerabilities with high exploited belief
                    if exploited_belief > 0.5:
                        valid_actions.append(action)
                        logger.debug(f"Impact action validated: {action_type} on {target_vuln} (exploited belief: {exploited_belief:.3f})")
                    else:
                        logger.warning(f"Impact action rejected: {action_type} on {target_vuln} (exploited belief: {exploited_belief:.3f} < 0.5)")
                else:
                    # If no specific vulnerability, allow the action (fallback)
                    valid_actions.append(action)
            else:
                # For non-impact actions, use normal validation
                valid_actions.append(action)

        if not valid_actions:
            logger.warning("No valid actions found after belief state validation")
            return None

        # Continue with normal action selection from valid actions
        # Use the original action selection logic from the parent class
        return self._select_action_from_valid_actions(valid_actions, state)
    
    def _select_action_from_valid_actions(self, valid_actions, state):
        """Helper method to select action from validated actions."""
        if not valid_actions:
            return None
        
        # Use the existing action selection logic
        # This is a simplified version - in practice, you'd want to use the full POMDP logic
        return valid_actions[0] if valid_actions else None

    def validate_exploit_against_belief(self, target_vuln: str, target_asset: str, target_component: str = '0') -> bool:
        """
        Validate exploit attempt against belief state before execution.
        Returns True if exploit should be allowed based on belief state.
        """
        if not self.belief_state:
            logger.warning("Belief state not initialized, allowing exploit without validation")
            return True
            
        vuln_key = create_vuln_key(target_vuln, str(target_asset), str(target_component))
        patch_belief = self.belief_state.get_patch_belief(vuln_key)
        exploited_belief = self.belief_state.get_exploited_belief(vuln_key)
        
        # If belief indicates vulnerability is likely patched, reject exploit
        if patch_belief > 0.7:
            logger.warning(f"Exploit rejected: {target_vuln} on {target_asset} (patch belief: {patch_belief:.3f} > 0.7)")
            return False
            
        # If belief indicates vulnerability is already exploited, reject exploit
        if exploited_belief > 0.7:
            logger.warning(f"Exploit rejected: {target_vuln} on {target_asset} (exploited belief: {exploited_belief:.3f} > 0.7)")
            return False
            
        logger.debug(f"Exploit validated: {target_vuln} on {target_asset} (patch: {patch_belief:.3f}, exploited: {exploited_belief:.3f})")
        return True