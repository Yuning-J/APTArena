import logging
import random
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from enum import Enum
from dataclasses import dataclass
from collections import defaultdict
from classes.state import create_vuln_key, State
from classes.mitre import KillChainStage, APT3TacticMapping
import os

logger = logging.getLogger(__name__)
logger.debug(f"Loading strategic_decision_manager from: {os.path.abspath(__file__)}")

class ActionType(Enum):
    RECONNAISSANCE = "reconnaissance"
    DELIVERY = "initial_access"
    EXPLOITATION = "exploitation"
    LATERAL_MOVEMENT = "lateral_movement"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    PERSISTENCE = "persistence"
    COMMAND_AND_CONTROL = "command_and_control"
    EXFILTRATION = "exfiltration"

class MissionPhase(Enum):
    RECONNAISSANCE = KillChainStage.RECONNAISSANCE
    WEAPONIZATION = KillChainStage.WEAPONIZATION
    DELIVERY = KillChainStage.DELIVERY
    EXPLOITATION = KillChainStage.EXPLOITATION
    INSTALLATION = KillChainStage.INSTALLATION
    COMMAND_AND_CONTROL = KillChainStage.COMMAND_AND_CONTROL
    ACTIONS_ON_OBJECTIVES = KillChainStage.ACTIONS_ON_OBJECTIVES

@dataclass
class ActionOpportunity:
    action_type: ActionType
    target_asset_id: str
    target_vuln_id: Optional[str] = None
    success_probability: float = 0.0
    resource_cost: float = 0.0
    expected_value: float = 0.0
    detection_risk: float = 0.0
    mission_alignment: float = 0.0
    strategic_score: float = 0.0
    target_component_id: Optional[str] = None

    def calculate_score(self, weights: Dict[str, float]) -> float:
        self.strategic_score = (
            weights.get('success', 0.2) * self.success_probability +
            weights.get('value', 0.3) * self.expected_value +
            weights.get('cost', 0.1) * (1.0 - min(self.resource_cost / 1000, 1.0)) +
            weights.get('stealth', 0.2) * (1.0 - self.detection_risk) +
            weights.get('mission', 0.2) * self.mission_alignment
        )
        return self.strategic_score

class APT3StrategicDecisionManager:
    def __init__(self, attack_graph, system, cost_calculator=None, cost_cache: Optional[Dict] = None, attacker_policy=None):
        self.attack_graph = attack_graph
        self.system = system
        self.cost_calculator = cost_calculator
        self.cost_cache = cost_cache or {}
        self.attacker_policy = attacker_policy
        self.apt3_config = {
            'primary_objective': '8',
            'secondary_objectives': ['2', '3', '6'],
            'preferred_techniques': {
                'T1203', 'T1068', 'T1055', 'T1548', 'T1190',
                'T1040', 'T1557', 'T1134', 'T1027', 'T1562.003'
            },
            'priority_cves': {
                'CVE-2018-13379': 1.5,
                'ZERO-DAY-001': 2.0,
                'CVE-2015-3113': 1.3
            }
        }
        self.current_mission_phase = MissionPhase.RECONNAISSANCE
        self.compromised_assets: Set[str] = set()
        self.exploited_vulnerabilities: Set[str] = set()
        self.intelligence_gathered: Dict[str, any] = {}

        # Initialize exploit_failures as empty - CRITICAL
        self.exploit_failures: Dict[str, int] = {}

        self.base_weights = {
            'success': 0.25,
            'value': 0.30,
            'cost': 0.10,
            'stealth': 0.15,
            'mission': 0.20
        }
        logger.info("APT3 Strategic Decision Manager initialized with clean exploit_failures dictionary")
    def _update_mission_phase(self, state):
        compromised_count = len(self.compromised_assets)
        rtu_compromised = self.apt3_config['primary_objective'] in self.compromised_assets
        recon_count = sum(1 for a in getattr(state.system, 'action_history', [])
                          if a.get('action_type') == 'reconnaissance' and a.get('action_result', False))
        if compromised_count == 0:
            self.current_mission_phase = MissionPhase.RECONNAISSANCE
            if recon_count >= 5:
                self.current_mission_phase = MissionPhase.DELIVERY
        elif compromised_count < 2:
            self.current_mission_phase = MissionPhase.EXPLOITATION
        elif not rtu_compromised and compromised_count < 4:
            self.current_mission_phase = MissionPhase.EXPLOITATION
        elif not rtu_compromised:
            self.current_mission_phase = MissionPhase.COMMAND_AND_CONTROL
        else:
            self.current_mission_phase = MissionPhase.ACTIONS_ON_OBJECTIVES
        logger.debug(f"Mission phase updated to: {self.current_mission_phase.value.name}")

    def make_strategic_decision(self, current_asset_id: str, state, attacker_budget: float,
                                detection_probability: float = 0.0, attacker_policy=None,
                                force_action_type: Optional[str] = None) -> ActionOpportunity:
        try:
            logger.debug(f"Making strategic decision from position: {current_asset_id}")
            logger.debug(f"Available budget: ${attacker_budget:.2f}")
            logger.debug(f"Detection probability: {detection_probability:.3f}")
            logger.debug(f"Force action type: {force_action_type}")

            self.attacker_policy = attacker_policy  # Update attacker_policy for this decision
            self._update_mission_phase(state)

            if current_asset_id == 'internet':
                logger.info("Attacker at internet position - assessing initial access opportunities")
                initial_opportunities = self._assess_initial_access_opportunities(
                    state, attacker_budget, detection_probability
                )
                if not initial_opportunities:
                    logger.warning("No initial access opportunities found from internet")
                    return ActionOpportunity(
                        action_type=ActionType.RECONNAISSANCE,
                        target_asset_id='internet',
                        success_probability=0.9,
                        expected_value=100.0,
                        resource_cost=1.0
                    )
                dynamic_weights = self._calculate_dynamic_weights(current_asset_id, state, detection_probability)
                for opp in initial_opportunities:
                    opp.calculate_score(dynamic_weights)
                initial_opportunities.sort(key=lambda x: x.strategic_score, reverse=True)
                if force_action_type == ActionType.DELIVERY.value:
                    initial_opportunities = [opp for opp in initial_opportunities
                                             if opp.action_type == ActionType.DELIVERY]
                selected = self._apply_apt3_selection_logic(initial_opportunities, current_asset_id, attacker_policy,
                                                            attacker_budget)
                if selected is None or (force_action_type and selected.action_type.value != force_action_type):
                    logger.warning(
                        "APT3 selection logic returned None or mismatched forced action, using highest scoring initial access")
                    selected = next((opp for opp in initial_opportunities if opp.action_type == ActionType.DELIVERY),
                                    initial_opportunities[0] if initial_opportunities else None)
                    if selected is None:
                        logger.error("No valid initial access opportunities after filtering")
                        return ActionOpportunity(
                            action_type=ActionType.RECONNAISSANCE,
                            target_asset_id='internet',
                            success_probability=0.9,
                            expected_value=100.0,
                            resource_cost=1.0
                        )
                logger.info(f"Selected initial access: {selected.action_type.value} on {selected.target_asset_id} "
                            f"using {getattr(selected, 'target_vuln_id', 'N/A')} (score: {selected.strategic_score:.3f})")
                return selected
            opportunities = self._assess_all_opportunities(
                current_asset_id, state, attacker_budget, detection_probability
            )
            if not opportunities:
                logger.warning(f"No viable opportunities found from asset {current_asset_id}")
                return ActionOpportunity(
                    action_type=ActionType.RECONNAISSANCE,
                    target_asset_id=current_asset_id,
                    success_probability=0.9,
                    expected_value=100.0,
                    resource_cost=1.0
                )
            dynamic_weights = self._calculate_dynamic_weights(
                current_asset_id, state, detection_probability
            )
            for opp in opportunities:
                opp.calculate_score(dynamic_weights)
            if force_action_type:
                action_type_map = {
                    ActionType.DELIVERY.value: ActionType.DELIVERY,
                    ActionType.EXPLOITATION.value: ActionType.EXPLOITATION,
                    ActionType.LATERAL_MOVEMENT.value: ActionType.LATERAL_MOVEMENT,
                    ActionType.PRIVILEGE_ESCALATION.value: ActionType.PRIVILEGE_ESCALATION,
                    ActionType.PERSISTENCE.value: ActionType.PERSISTENCE,
                    ActionType.COMMAND_AND_CONTROL.value: ActionType.COMMAND_AND_CONTROL,
                    ActionType.EXFILTRATION.value: ActionType.EXFILTRATION,
                    ActionType.RECONNAISSANCE.value: ActionType.RECONNAISSANCE
                }
                target_action_type = action_type_map.get(force_action_type)
                if target_action_type:
                    opportunities = [opp for opp in opportunities if opp.action_type == target_action_type]
                    if not opportunities:
                        logger.warning(f"No opportunities match forced action type {force_action_type}")
                        return ActionOpportunity(
                            action_type=ActionType.RECONNAISSANCE,
                            target_asset_id=current_asset_id,
                            success_probability=0.9,
                            expected_value=100.0,
                            resource_cost=1.0
                        )
            opportunities.sort(key=lambda x: x.strategic_score, reverse=True)
            selected = self._apply_apt3_selection_logic(opportunities, current_asset_id, attacker_policy,
                                                        attacker_budget)
            if selected is None or (force_action_type and selected.action_type.value != force_action_type):
                logger.warning(
                    "APT3 selection logic returned None or mismatched forced action, using highest scoring opportunity")
                selected = opportunities[0]
            logger.info(f"Selected action: {selected.action_type.value} on {selected.target_asset_id} "
                        f"(score: {selected.strategic_score:.3f})")
            return selected
        except Exception as e:
            logger.error(f"Error in make_strategic_decision: {e}")
            return ActionOpportunity(
                action_type=ActionType.RECONNAISSANCE,
                target_asset_id=current_asset_id if current_asset_id else 'unknown',
                success_probability=0.9,
                expected_value=100.0,
                resource_cost=1.0
            )

    def _action_type_to_tactic_name(self, action_type: ActionType) -> str:
        mapping = {
            ActionType.RECONNAISSANCE: "Reconnaissance",
            ActionType.DELIVERY: "Initial Access",
            ActionType.EXPLOITATION: "Exploitation",
            ActionType.LATERAL_MOVEMENT: "Lateral Movement",
            ActionType.PRIVILEGE_ESCALATION: "Privilege Escalation",
            ActionType.PERSISTENCE: "Persistence",
            ActionType.COMMAND_AND_CONTROL: "Command and Control",
            ActionType.EXFILTRATION: "Exfiltration"
        }
        return mapping.get(action_type, "Unknown")

    def _assess_all_opportunities(self, current_asset_id: str, state, attacker_budget: float,
                                 detection_probability: float) -> List[ActionOpportunity]:
        opportunities = []

        if current_asset_id == 'internet':
            initial_access_opps = self._assess_initial_access_opportunities(
                state, attacker_budget, detection_probability
            )
            opportunities.extend(initial_access_opps)
        else:
            exploitation_opps = self._assess_exploitation(
                current_asset_id, state, attacker_budget, detection_probability
            )
            opportunities.extend(exploitation_opps)

            lateral_opps = self._assess_lateral_movement(
                current_asset_id, state, attacker_budget, detection_probability
            )
            opportunities.extend(lateral_opps)

            privesc_opps = self._assess_privilege_escalation(
                current_asset_id, state, attacker_budget, detection_probability
            )
            opportunities.extend(privesc_opps)

            persistence_opps = self._assess_persistence_opportunities(
                current_asset_id, state, attacker_budget, detection_probability
            )
            opportunities.extend(persistence_opps)

            c2_opps = self._assess_command_and_control_opportunities(
                current_asset_id, state, attacker_budget, detection_probability
            )
            opportunities.extend(c2_opps)

            exfil_opps = self._assess_exfiltration_opportunities(
                current_asset_id, state, attacker_budget, detection_probability
            )
            opportunities.extend(exfil_opps)

        opportunities.append(ActionOpportunity(
            action_type=ActionType.RECONNAISSANCE,
            target_asset_id=current_asset_id,
            success_probability=0.9,
            expected_value=100.0,
            resource_cost=10.0,
            strategic_score=0.4
        ))

        return opportunities

    def _get_patch_belief(self, vuln_key: str, state) -> float:
        """
        Retrieve the belief probability that a vulnerability is patched from AttackerPOMDPPolicy.
        """
        if hasattr(self, 'attacker_policy') and self.attacker_policy and hasattr(self.attacker_policy, 'belief_state'):
            belief = self.attacker_policy.belief_state.get_patch_belief(vuln_key)
            logger.debug(f"Patch belief for {vuln_key}: {belief:.3f}")
            return belief
        logger.warning(f"No attacker policy or belief state available for {vuln_key}, defaulting to 0.5")
        return 0.5  # Default belief if policy is unavailable

    def _get_exploited_belief(self, vuln_key: str, state) -> float:
        """
        Retrieve the belief probability that a vulnerability is exploited from AttackerPOMDPPolicy.
        """
        if hasattr(self, 'attacker_policy') and self.attacker_policy and hasattr(self.attacker_policy, 'belief_state'):
            belief = self.attacker_policy.belief_state.get_exploited_belief(vuln_key)
            logger.debug(f"Exploited belief for {vuln_key}: {belief:.3f}")
            return belief
        logger.warning(f"No attacker policy or belief state available for {vuln_key}, defaulting to 0.5")
        return 0.5  # Default belief if policy is unavailable
    def _assess_initial_access_opportunities(self, state, attacker_budget: float,
                                             detection_probability: float) -> List[ActionOpportunity]:
        opportunities = []
        entry_point_assets = set()
        internet_facing_keywords = ['web server', 'vpn', 'firewall', 'gateway', 'proxy', 'dmz']
        apt3_entry_points = {
            '1': 'Domain Controller misconfiguration (exposed Active Directory services)',
            '3': 'User Workstation phishing (spearphishing email exploit)',
            '4': 'VPN exploitation (CVE-2018-13379 - known APT3 vector)',
            '5': 'Web server compromise (internet-facing DMZ asset)'
        }

        self.compromised_assets.clear()
        for asset in state.system.assets:
            if asset.is_compromised:
                self.compromised_assets.add(str(asset.asset_id))
                logger.debug(f"Synchronized compromised asset {asset.asset_id}")

        for asset in state.system.assets:
            asset_id = str(asset.asset_id)
            if asset.is_compromised:
                logger.debug(f"Skipping compromised asset {asset_id}")
                continue

            asset_type = getattr(asset, 'type', '').lower()
            asset_name = getattr(asset, 'name', '').lower()

            has_internet_connection = any(
                conn.from_asset and str(conn.from_asset.asset_id) == 'internet'
                for conn in state.system.connections
                if conn.to_asset and str(conn.to_asset.asset_id) == asset_id
            )

            is_internet_facing = any(
                keyword in asset_type or keyword in asset_name for keyword in internet_facing_keywords)

            is_apt3_entry = asset_id in apt3_entry_points and (
                    has_internet_connection or apt3_entry_points[asset_id].lower().startswith(
                'user workstation phishing'))

            if not (has_internet_connection or is_apt3_entry or is_internet_facing):
                continue

            network_vuln_count = 0
            for comp in asset.components:
                for vuln in comp.vulnerabilities:
                    vuln_key = create_vuln_key(vuln.cve_id, asset_id, comp.id)
                    # Remove direct is_patched/is_exploited checks
                    if 'AV:N' in getattr(vuln, 'cvssV3Vector', '') or vuln.cve_id in self.apt3_config.get(
                            'priority_cves', []):
                        network_vuln_count += 1

            if network_vuln_count > 0 or is_internet_facing or is_apt3_entry:
                entry_point_assets.add(asset_id)
                logger.info(
                    f"Added dynamic entry point {asset_id} ({asset.name}) with {network_vuln_count} network vulns")

        total_vulns_checked = 0
        total_opportunities_created = 0

        for asset_id in entry_point_assets:
            if asset_id == 'internet':
                continue

            asset = self._get_asset_by_id(asset_id)
            if not asset:
                logger.warning(f"Asset {asset_id} not found")
                continue

            for comp in asset.components:
                for vuln in comp.vulnerabilities:
                    total_vulns_checked += 1
                    vuln_key = create_vuln_key(vuln.cve_id, asset_id, str(comp.id))
                    logger.debug(f"Checking vuln {vuln.cve_id} on asset {asset_id}, comp {comp.id}")

                    # Get belief about patch status from AttackerPOMDPPolicy
                    patch_belief = self._get_patch_belief(vuln_key, state)
                    if patch_belief > 0.8:  # High confidence vulnerability is patched
                        logger.debug(
                            f"Skipping {vuln.cve_id} on {asset.name} due to high patch belief ({patch_belief:.3f})")
                        continue

                    # Get belief about exploited status
                    exploited_belief = self._get_exploited_belief(vuln_key, state)
                    if exploited_belief > 0.8:  # High confidence vulnerability is exploited
                        logger.debug(
                            f"Skipping {vuln.cve_id} on {asset.name} due to high exploited belief ({exploited_belief:.3f})")
                        continue

                    exploit_cost = self.cost_cache.get('exploit_costs', {}).get(vuln_key, None)
                    if exploit_cost is None:
                        logger.debug(f"Cache miss for {vuln_key}. Calculating exploit cost.")
                        exploit_cost = self._calculate_exploit_cost(vuln, asset, state, str(comp.id))

                    if exploit_cost > attacker_budget * 1.5:
                        logger.debug(
                            f"Skipping {vuln.cve_id} due to cost {exploit_cost} > budget {attacker_budget * 1.5}")
                        continue

                    success_prob = self._calculate_initial_access_probability(vuln, asset)
                    # Adjust success probability based on patch belief
                    success_prob *= (1.0 - patch_belief)
                    if vuln.cve_id in self.apt3_config['priority_cves']:
                        success_prob = min(0.95, success_prob * 1.2)
                    expected_value = self._calculate_exploit_value(vuln, asset) * self._get_apt3_technique_multiplier(
                        vuln)

                    if asset_id in ['1', '3']:
                        expected_value *= 1.3
                    elif asset_id in ['4', '5']:
                        expected_value *= 1.2

                    detection_risk = detection_probability * 0.6
                    mission_alignment = self._calculate_initial_access_mission_alignment(asset_id, vuln.cve_id)

                    opportunity = ActionOpportunity(
                        action_type=ActionType.DELIVERY,
                        target_asset_id=asset_id,
                        target_vuln_id=vuln.cve_id,
                        success_probability=success_prob,
                        resource_cost=exploit_cost,
                        expected_value=expected_value,
                        detection_risk=detection_risk,
                        mission_alignment=mission_alignment,
                        target_component_id=str(comp.id)
                    )

                    opportunities.append(opportunity)
                    total_opportunities_created += 1
                    logger.debug(
                        f"Created opportunity: {vuln.cve_id} on {asset.name} (prob: {success_prob:.3f}, value: {expected_value:.0f}, cost: ${exploit_cost:.2f})")

        logger.info(
            f"Initial access assessment: {len(entry_point_assets)} assets, {total_vulns_checked} vulns, {total_opportunities_created} opportunities")

        if opportunities:
            sorted_opps = sorted(opportunities,
                                 key=lambda x: x.success_probability * x.expected_value * x.mission_alignment,
                                 reverse=True)
            for i, opp in enumerate(sorted_opps[:5]):
                composite = opp.success_probability * opp.expected_value * opp.mission_alignment
                logger.info(
                    f"  {i + 1}. {opp.target_vuln_id} on {opp.target_asset_id}: prob={opp.success_probability:.3f}, "
                    f"value={opp.expected_value:.0f}, cost=${opp.resource_cost:.2f}, composite={composite:.0f}")

        return opportunities

    def _apply_apt3_selection_logic(self, opportunities: List[ActionOpportunity], current_asset_id: str,
                                    attacker_policy=None, attacker_budget: float = 15000) -> ActionOpportunity:
        if not opportunities:
            logger.warning("No opportunities provided")
            return ActionOpportunity(
                action_type=ActionType.RECONNAISSANCE,
                target_asset_id=current_asset_id,
                success_probability=0.9,
                expected_value=100.0,
                resource_cost=1.0
            )

        exploit_opportunities = []
        non_exploit_opportunities = []
        system = self.system
        logger.debug(f"Current exploit failures tracked: {dict(self.exploit_failures)}")
        logger.debug(f"Compromised assets: {self.compromised_assets}")
        logger.debug(f"Exploited vulnerabilities: {self.exploited_vulnerabilities}")
        logger.debug(
            f"Processing {len(opportunities)} initial opportunities with attacker_budget=${attacker_budget:.2f}")

        # Define consistent failure thresholds
        STANDARD_MAX_FAILURES = 3
        PRIORITY_MAX_FAILURES = 5

        for opp in opportunities:
            logger.debug(
                f"Evaluating opportunity: {opp.action_type.value} on {opp.target_asset_id}, vuln={opp.target_vuln_id}")
            requires_vuln = opp.action_type in [
                ActionType.EXPLOITATION,
                ActionType.PRIVILEGE_ESCALATION,
                ActionType.DELIVERY,
                ActionType.LATERAL_MOVEMENT
            ]

            if requires_vuln and opp.target_vuln_id is None:
                logger.debug(f"Skipping {opp.action_type.value} with no vuln_id on {opp.target_asset_id}")
                continue

            if not requires_vuln:
                non_exploit_opportunities.append(opp)
                logger.debug(f"Added non-exploit opportunity: {opp.action_type.value} on {opp.target_asset_id}")
                continue

            target_asset = None
            target_vuln = None
            target_comp = str(opp.target_component_id or "0")

            for asset in system.assets:
                if str(asset.asset_id) == str(opp.target_asset_id):
                    target_asset = asset
                    for comp in asset.components:
                        if str(comp.id) == target_comp:
                            for vuln in comp.vulnerabilities:
                                if vuln.cve_id == opp.target_vuln_id:
                                    target_vuln = vuln
                                    break
                            if target_vuln:
                                break
                    break

            if not target_asset:
                logger.debug(f"Skipping {opp.target_vuln_id} on {opp.target_asset_id}: asset not found")
                continue
            if not target_vuln:
                logger.debug(f"Skipping {opp.target_vuln_id} on {opp.target_asset_id}: vuln not found")
                continue

            vuln_key = create_vuln_key(opp.target_vuln_id, opp.target_asset_id, target_comp)
            failure_count = self.exploit_failures.get(vuln_key, 0)

            # Use consistent thresholds
            MAX_FAILURES = PRIORITY_MAX_FAILURES if opp.target_vuln_id in self.apt3_config[
                'priority_cves'] else STANDARD_MAX_FAILURES

            # Get belief about patch and exploited status
            patch_belief = self._get_patch_belief(vuln_key, None)
            exploited_belief = self._get_exploited_belief(vuln_key, None)

            logger.debug(f"Filtering {opp.target_vuln_id} on {opp.target_asset_id}: "
                         f"failures={failure_count}/{MAX_FAILURES}, "
                         f"patch_belief={patch_belief:.3f}, exploited_belief={exploited_belief:.3f}, "
                         f"budget_ok={opp.resource_cost <= attacker_budget}")

            # Skip if asset is already compromised for DELIVERY actions
            if target_asset.is_compromised and opp.action_type == ActionType.DELIVERY:
                logger.debug(
                    f"FILTERING: Skipping {opp.target_vuln_id} on {opp.target_asset_id} (asset already compromised)")
                continue

            # Skip if vulnerability is already exploited
            if vuln_key in self.exploited_vulnerabilities or exploited_belief > 0.8:
                logger.debug(
                    f"FILTERING: Skipping {opp.target_vuln_id} on {opp.target_asset_id} (in exploited_vulnerabilities or high exploited belief)")
                continue

            # Skip if failure count exceeds MAX_FAILURES
            if failure_count >= MAX_FAILURES:
                logger.debug(
                    f"FILTERING: Skipping {opp.target_vuln_id} on {opp.target_asset_id} (failures: {failure_count}/{MAX_FAILURES})")
                continue

            # Skip if cost exceeds budget
            if opp.resource_cost > attacker_budget:
                logger.debug(
                    f"FILTERING: Skipping {opp.target_vuln_id} on {opp.target_asset_id} (cost {opp.resource_cost} exceeds budget {attacker_budget})")
                continue

            # Skip if patch belief is too high
            if patch_belief > 0.8:
                logger.debug(
                    f"FILTERING: Skipping {opp.target_vuln_id} on {opp.target_asset_id} due to high patch belief ({patch_belief:.3f})")
                continue

            # Apply APT3 technique multiplier
            technique_multiplier = 1.0
            if opp.target_vuln_id in APT3TacticMapping.APT3_CVE_TO_TECHNIQUE:
                techniques = APT3TacticMapping.APT3_CVE_TO_TECHNIQUE[opp.target_vuln_id]
                preferred_techniques = APT3TacticMapping.get_preferred_techniques(self.current_mission_phase.value)
                if any(tech in preferred_techniques for tech in techniques):
                    technique_multiplier = 1.3
                    logger.debug(f"Boosting score for {opp.target_vuln_id} due to APT3-preferred technique")

            # Adjust success probability and score
            opp.success_probability = min(opp.success_probability * (1.0 - patch_belief), 0.95)
            opp.strategic_score *= technique_multiplier
            exploit_opportunities.append(opp)
            logger.debug(
                f"Added exploit opportunity: {opp.target_vuln_id} on {opp.target_asset_id} (failures: {failure_count}, score: {opp.strategic_score:.3f})")

        all_valid_opportunities = exploit_opportunities + non_exploit_opportunities

        logger.info(
            f"FILTERING RESULTS: {len(exploit_opportunities)} exploit opportunities, {len(non_exploit_opportunities)} non-exploit opportunities out of {len(opportunities)} total")

        if not all_valid_opportunities:
            logger.warning("No valid opportunities after filtering, returning reconnaissance action")
            return ActionOpportunity(
                action_type=ActionType.RECONNAISSANCE,
                target_asset_id=current_asset_id,
                success_probability=0.9,
                expected_value=100.0,
                resource_cost=1.0,
                strategic_score=0.5
            )

        all_valid_opportunities.sort(key=lambda x: x.strategic_score, reverse=True)

        # Prioritize secondary objectives if they are close in score
        if exploit_opportunities:
            secondary_opps = []
            for opp in exploit_opportunities:
                if opp.target_asset_id in self.apt3_config['secondary_objectives']:
                    vuln_key = create_vuln_key(opp.target_vuln_id, opp.target_asset_id, opp.target_component_id or "0")
                    failure_count = self.exploit_failures.get(vuln_key, 0)
                    # Use the same consistent thresholds
                    MAX_FAILURES = PRIORITY_MAX_FAILURES if opp.target_vuln_id in self.apt3_config[
                        'priority_cves'] else STANDARD_MAX_FAILURES
                    if failure_count < MAX_FAILURES:
                        secondary_opps.append(opp)

            if secondary_opps and secondary_opps[0].strategic_score >= all_valid_opportunities[0].strategic_score * 0.8:
                selected = secondary_opps[0]
                logger.info(
                    f"APT3 Rule: Selected secondary objective {selected.target_vuln_id} on {selected.target_asset_id}")
                return selected

            selected = exploit_opportunities[0]
            logger.info(f"Selected exploit: {selected.action_type.value} on {selected.target_asset_id} "
                        f"using {selected.target_vuln_id} (score: {selected.strategic_score:.3f})")
            return selected

        selected = all_valid_opportunities[0]
        logger.info(f"Selected action: {selected.action_type.value} on {selected.target_asset_id} "
                    f"using {getattr(selected, 'target_vuln_id', 'None')} (score: {selected.strategic_score:.3f})")
        return selected

    def _calculate_initial_access_mission_alignment(self, asset_id: str, vuln_id: str) -> float:
        base_alignment = 1.0
        if self.current_mission_phase == MissionPhase.DELIVERY:
            base_alignment *= 1.5
        elif self.current_mission_phase != MissionPhase.RECONNAISSANCE:
            base_alignment *= 0.8

        if vuln_id in self.apt3_config['priority_cves']:
            base_alignment *= 1.3
        if asset_id in self.apt3_config['secondary_objectives']:
            base_alignment *= 1.2
        if asset_id == self.apt3_config['primary_objective']:
            base_alignment *= 1.5

        logger.debug(f"Mission alignment for asset {asset_id}, vuln {vuln_id}: {base_alignment}")
        return base_alignment

    def _assess_exploitation(self, current_asset_id: str, state, attacker_budget: float,
                            detection_probability: float) -> List[ActionOpportunity]:
        opportunities = []
        current_asset = self._get_asset_by_id(current_asset_id)
        if not current_asset:
            return opportunities
        for comp in current_asset.components:
            for vuln in comp.vulnerabilities:
                vuln_key = create_vuln_key(vuln.cve_id, current_asset_id, str(comp.id))
                if vuln.is_patched or vuln.is_exploited:
                    logger.debug(f"Skipping {vuln.cve_id} on {current_asset.name} (patched/exploited)")
                    continue
                exploit_cost = self.cost_cache.get('exploit_costs', {}).get(vuln_key, None)
                if exploit_cost is None:
                    logger.debug(f"Cache miss for {vuln_key}. Calculating exploit cost.")
                    exploit_cost = self._calculate_exploit_cost(vuln, current_asset, state, str(comp.id))
                if exploit_cost > attacker_budget:
                    continue
                success_prob = self._calculate_exploit_probability(vuln, current_asset)
                base_value = self._calculate_exploit_value(vuln, current_asset)
                apt3_multiplier = self._get_apt3_technique_multiplier(vuln)
                expected_value = base_value * apt3_multiplier
                detection_risk = detection_probability * 0.3
                mission_alignment = self._calculate_mission_alignment(
                    ActionType.EXPLOITATION, current_asset_id, vuln.cve_id
                )
                opportunity = ActionOpportunity(
                    action_type=ActionType.EXPLOITATION,
                    target_asset_id=current_asset_id,
                    target_vuln_id=vuln.cve_id,
                    success_probability=success_prob,
                    resource_cost=exploit_cost,
                    expected_value=expected_value,
                    detection_risk=detection_risk,
                    mission_alignment=mission_alignment,
                    target_component_id=str(comp.id)
                )
                opportunities.append(opportunity)
        return opportunities

    def _assess_lateral_movement(self, current_asset_id: str, state, attacker_budget: float,
                                detection_probability: float) -> List[ActionOpportunity]:
        """
        Refined lateral movement assessment using hybrid approach:
        1. Current asset vulnerabilities for local privilege escalation
        2. Target asset vulnerabilities for network access
        3. Credential-based movement as alternative
        4. Fallback to traditional single-vulnerability approach
        """
        opportunities = []
        connected_assets = self._find_connected_assets(current_asset_id)
        current_asset = self._get_asset_by_id(current_asset_id)
        
        if not current_asset:
            return opportunities
            
        for target_asset_id in connected_assets:
            if target_asset_id in self.compromised_assets:
                continue
            target_asset = self._get_asset_by_id(target_asset_id)
            if not target_asset:
                continue
                
            # HYBRID APPROACH: Combine current and target asset vulnerabilities
            hybrid_opportunities = self._assess_hybrid_lateral_movement(
                current_asset, target_asset, attacker_budget, detection_probability
            )
            opportunities.extend(hybrid_opportunities)
            
            # CREDENTIAL-BASED MOVEMENT: Alternative to vulnerability exploitation
            credential_opportunity = self._assess_credential_based_movement(
                current_asset, target_asset, attacker_budget, detection_probability
            )
            if credential_opportunity:
                opportunities.append(credential_opportunity)
            
            # FALLBACK: Traditional single-vulnerability approach on current asset
            # This ensures we always have some lateral movement options
            fallback_opportunities = self._assess_fallback_lateral_movement(
                current_asset, target_asset, attacker_budget, detection_probability
            )
            opportunities.extend(fallback_opportunities)
                
        return opportunities

    def _assess_hybrid_lateral_movement(self, current_asset, target_asset, attacker_budget: float,
                                      detection_probability: float) -> List[ActionOpportunity]:
        """
        Hybrid lateral movement combining:
        - Current asset vulnerabilities for local privilege escalation
        - Target asset vulnerabilities for network access
        """
        opportunities = []
        
        # Step 1: Find local privilege escalation vulnerabilities on current asset
        local_privesc_vulns = self._find_local_privilege_escalation_vulns(current_asset)
        
        # Step 2: Find network access vulnerabilities on target asset
        network_access_vulns = self._find_network_access_vulns(target_asset)
        
        # Step 3: Create hybrid opportunities combining both
        for local_vuln in local_privesc_vulns:
            for network_vuln in network_access_vulns:
                # Calculate combined cost and probability
                total_cost = local_vuln['cost'] + network_vuln['cost']
                if total_cost > attacker_budget:
                    continue
                    
                # Combined success probability (both must succeed)
                combined_prob = local_vuln['success_prob'] * network_vuln['success_prob']
                
                # Enhanced value for successful lateral movement
                target_value = self._calculate_asset_strategic_value(target_asset) * 1.5
                if target_asset.asset_id == self.apt3_config['primary_objective']:
                    target_value *= 3.0
                elif target_asset.asset_id in self.apt3_config['secondary_objectives']:
                    target_value *= 1.8
                    
                detection_risk = detection_probability * 0.7  # Slightly lower than pure network exploitation
                mission_alignment = self._calculate_mission_alignment(
                    ActionType.LATERAL_MOVEMENT, target_asset.asset_id, network_vuln['cve_id']
                )
                
                opportunity = ActionOpportunity(
                    action_type=ActionType.LATERAL_MOVEMENT,
                    target_asset_id=target_asset.asset_id,
                    target_vuln_id=f"{local_vuln['cve_id']}+{network_vuln['cve_id']}",  # Combined identifier
                    success_probability=combined_prob,
                    resource_cost=total_cost,
                    expected_value=target_value,
                    detection_risk=detection_risk,
                    mission_alignment=mission_alignment,
                    target_component_id=f"{local_vuln['component_id']}+{network_vuln['component_id']}"
                )
                opportunities.append(opportunity)
                logger.debug(f"Hybrid lateral movement: {local_vuln['cve_id']} (local) + {network_vuln['cve_id']} (network), prob={combined_prob:.3f}")
                
        return opportunities

    def _assess_credential_based_movement(self, current_asset, target_asset, attacker_budget: float,
                                        detection_probability: float) -> Optional[ActionOpportunity]:
        """
        Assess credential-based lateral movement as alternative to vulnerability exploitation.
        This models real-world attacks using stolen credentials via tools like Mimikatz.
        """
        # Check if we have credentials available (simplified model)
        has_credentials = self._check_credential_availability(current_asset, target_asset)
        if not has_credentials:
            return None
            
        # Credential-based movement parameters
        credential_cost = 75.0  # Cost of using credential dumping tools
        base_success_prob = 0.85  # High success rate for valid credentials
        
        if credential_cost > attacker_budget:
            return None
            
        # Adjust probability based on target asset security posture
        security_factor = self._assess_target_security_posture(target_asset)
        success_prob = base_success_prob * security_factor
        
        target_value = self._calculate_asset_strategic_value(target_asset)
        if target_asset.asset_id == self.apt3_config['primary_objective']:
            target_value *= 3.0
        elif target_asset.asset_id in self.apt3_config['secondary_objectives']:
            target_value *= 1.8
            
        detection_risk = detection_probability * 0.6  # Lower detection risk than vulnerability exploitation
        mission_alignment = self._calculate_mission_alignment(
            ActionType.LATERAL_MOVEMENT, target_asset.asset_id
        )
        
        return ActionOpportunity(
            action_type=ActionType.LATERAL_MOVEMENT,
            target_asset_id=target_asset.asset_id,
            target_vuln_id="CREDENTIAL_BASED",  # Special identifier for credential-based movement
            success_probability=success_prob,
            resource_cost=credential_cost,
            expected_value=target_value,
            detection_risk=detection_risk,
            mission_alignment=mission_alignment,
            target_component_id="credentials"
        )

    def _find_local_privilege_escalation_vulns(self, asset) -> List[Dict]:
        """
        Find vulnerabilities suitable for local privilege escalation on the current asset.
        These are typically local vulnerabilities that can be exploited to gain higher privileges.
        """
        local_vulns = []
        
        for comp in asset.components:
            for vuln in comp.vulnerabilities:
                if vuln.is_patched or vuln.is_exploited:
                    continue
                    
                # Check if this is a local privilege escalation vulnerability
                if not self._is_local_privilege_escalation_vuln(vuln):
                    continue
                    
                vuln_key = create_vuln_key(vuln.cve_id, asset.asset_id, str(comp.id))
                cost = self.cost_cache.get('exploit_costs', {}).get(vuln_key, 50 + getattr(vuln, 'cvss', 5.0) * 20)
                
                # Calculate success probability for local exploitation
                base_prob = min(0.9, getattr(vuln, 'epss', 0.1) * 1.5)
                if getattr(vuln, 'exploit', False):
                    base_prob *= 1.2
                success_prob = max(0.1, min(0.9, base_prob))
                
                local_vulns.append({
                    'cve_id': vuln.cve_id,
                    'component_id': str(comp.id),
                    'cost': cost,
                    'success_prob': success_prob,
                    'vuln_key': vuln_key
                })
                
        return local_vulns

    def _find_network_access_vulns(self, asset) -> List[Dict]:
        """
        Find vulnerabilities suitable for network-based lateral movement to the target asset.
        These are typically remote vulnerabilities that can be exploited over the network.
        """
        network_vulns = []
        
        for comp in asset.components:
            for vuln in comp.vulnerabilities:
                if vuln.is_patched or vuln.is_exploited:
                    continue
                    
                # Check if this is a network-accessible vulnerability
                if not self._is_network_access_vuln(vuln):
                    continue
                    
                vuln_key = create_vuln_key(vuln.cve_id, asset.asset_id, str(comp.id))
                cost = self.cost_cache.get('exploit_costs', {}).get(vuln_key, 50 + getattr(vuln, 'cvss', 5.0) * 20)
                
                # Calculate success probability for network exploitation
                base_prob = min(0.9, getattr(vuln, 'epss', 0.1) * 1.5)
                if getattr(vuln, 'exploit', False):
                    base_prob *= 1.2
                if 'AV:N' in getattr(vuln, 'cvssV3Vector', ''):
                    base_prob *= 1.1
                success_prob = max(0.1, min(0.9, base_prob))
                
                network_vulns.append({
                    'cve_id': vuln.cve_id,
                    'component_id': str(comp.id),
                    'cost': cost,
                    'success_prob': success_prob,
                    'vuln_key': vuln_key
                })
                
        return network_vulns

    def _is_local_privilege_escalation_vuln(self, vuln) -> bool:
        """
        Determine if a vulnerability is suitable for local privilege escalation.
        """
        # Check CVSS vector for local access
        cvss_vector = getattr(vuln, 'cvssV3Vector', '')
        if 'AV:L' in cvss_vector:  # Local access vector
            return True
            
        # Check for privilege escalation keywords in description
        description = getattr(vuln, 'description', '').lower()
        privesc_keywords = ['privilege escalation', 'elevation', 'local', 'kernel', 'driver']
        if any(keyword in description for keyword in privesc_keywords):
            return True
            
        # Check for specific CVE patterns known for privilege escalation
        privesc_cves = {'CVE-2021-1636', 'CVE-2021-34527', 'CVE-2021-36934'}  # Example CVEs
        if vuln.cve_id in privesc_cves:
            return True
            
        return False

    def _is_network_access_vuln(self, vuln) -> bool:
        """
        Determine if a vulnerability is suitable for network-based lateral movement.
        """
        # Check CVSS vector for network access
        cvss_vector = getattr(vuln, 'cvssV3Vector', '')
        if 'AV:N' in cvss_vector:  # Network access vector
            return True
            
        # Check for network service keywords in description
        description = getattr(vuln, 'description', '').lower()
        network_keywords = ['smb', 'rdp', 'ssh', 'telnet', 'ftp', 'http', 'https', 'network', 'remote']
        if any(keyword in description for keyword in network_keywords):
            return True
            
        # Check for specific CVE patterns known for network exploitation
        network_cves = {'CVE-2020-0796', 'CVE-2017-0144', 'CVE-2019-0708'}  # Example CVEs
        if vuln.cve_id in network_cves:
            return True
            
        return False

    def _check_credential_availability(self, current_asset, target_asset) -> bool:
        """
        Check if credentials are available for lateral movement.
        Simplified model: assume credentials are available if current asset is compromised.
        """
        # In a more sophisticated model, this would check for:
        # - Stolen credentials from current asset
        # - Domain credentials if in domain environment
        # - Service account credentials
        # - Password reuse patterns
        
        return current_asset.is_compromised

    def _assess_target_security_posture(self, target_asset) -> float:
        """
        Assess the security posture of the target asset to adjust credential-based movement probability.
        """
        # Simplified model: adjust based on asset criticality and type
        base_factor = 1.0
        
        # High-value assets typically have better security
        if target_asset.asset_id == self.apt3_config['primary_objective']:
            base_factor *= 0.7  # Lower success probability
        elif target_asset.asset_id in self.apt3_config['secondary_objectives']:
            base_factor *= 0.8
            
        # Adjust based on asset type (simplified)
        asset_name = target_asset.name.lower()
        if 'plc' in asset_name or 'rtu' in asset_name:
            base_factor *= 0.9  # Industrial systems may have different security
        elif 'server' in asset_name:
            base_factor *= 0.8  # Servers typically have better security
            
        return max(0.3, min(1.0, base_factor))  # Ensure reasonable bounds

    def _assess_privilege_escalation(self, current_asset_id: str, state, attacker_budget: float,
                                    detection_probability: float) -> List[ActionOpportunity]:
        opportunities = []
        current_asset = self._get_asset_by_id(current_asset_id)
        if not current_asset:
            return opportunities
        for comp in current_asset.components:
            for vuln in comp.vulnerabilities:
                if not self._is_privilege_escalation_vuln(vuln):
                    continue
                vuln_key = create_vuln_key(vuln.cve_id, current_asset_id, str(comp.id))
                if vuln.is_patched or vuln.is_exploited:
                    logger.debug(f"Skipping {vuln.cve_id} on {current_asset.name} (patched/exploited)")
                    continue
                exploit_cost = self.cost_cache.get('exploit_costs', {}).get(vuln_key, None)
                if exploit_cost is None:
                    logger.debug(f"Cache miss for {vuln_key}. Calculating exploit cost.")
                    exploit_cost = self._calculate_exploit_cost(vuln, current_asset, state, str(comp.id))
                exploit_cost *= 1.2
                if exploit_cost > attacker_budget:
                    continue
                success_prob = self._calculate_exploit_probability(vuln, current_asset) * 0.8
                privesc_value = self._calculate_privilege_escalation_value(current_asset)
                detection_risk = detection_probability * 0.4
                mission_alignment = self._calculate_mission_alignment(ActionType.PRIVILEGE_ESCALATION, current_asset_id, vuln.cve_id)
                opportunity = ActionOpportunity(
                    action_type=ActionType.PRIVILEGE_ESCALATION,
                    target_asset_id=current_asset_id,
                    target_vuln_id=vuln.cve_id,
                    success_probability=success_prob,
                    resource_cost=exploit_cost,
                    expected_value=privesc_value,
                    detection_risk=detection_risk,
                    mission_alignment=mission_alignment,
                    target_component_id=str(comp.id)
                )
                opportunities.append(opportunity)
        return opportunities

    def _assess_persistence_opportunities(self, current_asset_id: str, state, attacker_budget: float,
                                         detection_probability: float) -> List[ActionOpportunity]:
        opportunities = []
        if self._has_persistence(current_asset_id):
            return opportunities
        current_asset = self._get_asset_by_id(current_asset_id)
        if not current_asset:
            return opportunities
        persistence_cost = 100.0
        success_prob = 0.85
        if persistence_cost > attacker_budget:
            return opportunities
        persistence_value = self._calculate_persistence_value(current_asset)
        detection_risk = detection_probability * 0.5
        mission_alignment = self._calculate_mission_alignment(ActionType.PERSISTENCE, current_asset_id)
        opportunity = ActionOpportunity(
            action_type=ActionType.PERSISTENCE,
            target_asset_id=current_asset_id,
            success_probability=success_prob,
            resource_cost=persistence_cost,
            expected_value=persistence_value,
            detection_risk=detection_risk,
            mission_alignment=mission_alignment
        )
        opportunities.append(opportunity)
        return opportunities

    def _assess_command_and_control_opportunities(self, current_asset_id: str, state, attacker_budget: float,
                                                 detection_probability: float) -> List[ActionOpportunity]:
        opportunities = []
        current_asset = self._get_asset_by_id(current_asset_id)
        if not current_asset:
            return opportunities
        c2_cost = 50.0
        success_prob = 0.9
        if c2_cost > attacker_budget:
            return opportunities
        c2_value = self._calculate_asset_strategic_value(current_asset) * 0.5
        detection_risk = detection_probability * 0.3
        mission_alignment = self._calculate_mission_alignment(ActionType.COMMAND_AND_CONTROL, current_asset_id)
        opportunity = ActionOpportunity(
            action_type=ActionType.COMMAND_AND_CONTROL,
            target_asset_id=current_asset_id,
            success_probability=success_prob,
            resource_cost=c2_cost,
            expected_value=c2_value,
            detection_risk=detection_risk,
            mission_alignment=mission_alignment
        )
        opportunities.append(opportunity)
        return opportunities

    def _assess_exfiltration_opportunities(self, current_asset_id: str, state, attacker_budget: float,
                                          detection_probability: float) -> List[ActionOpportunity]:
        opportunities = []
        current_asset = self._get_asset_by_id(current_asset_id)
        if not current_asset:
            return opportunities
        exfil_cost = 75.0
        success_prob = 0.8
        if exfil_cost > attacker_budget:
            return opportunities
        exfil_value = self._calculate_asset_strategic_value(current_asset) * 0.7
        detection_risk = detection_probability * 0.4
        mission_alignment = self._calculate_mission_alignment(ActionType.EXFILTRATION, current_asset_id)
        opportunity = ActionOpportunity(
            action_type=ActionType.EXFILTRATION,
            target_asset_id=current_asset_id,
            success_probability=success_prob,
            resource_cost=exfil_cost,
            expected_value=exfil_value,
            detection_risk=detection_risk,
            mission_alignment=mission_alignment
        )
        opportunities.append(opportunity)
        return opportunities

    def _calculate_dynamic_weights(self, current_asset_id: str, state, detection_probability: float) -> Dict[str, float]:
        weights = self.base_weights.copy()
        if self.current_mission_phase == MissionPhase.DELIVERY:
            weights['success'] += 0.1
            weights['stealth'] += 0.1
        elif self.current_mission_phase == MissionPhase.COMMAND_AND_CONTROL:
            weights['mission'] += 0.15
            weights['value'] += 0.1
        elif self.current_mission_phase == MissionPhase.ACTIONS_ON_OBJECTIVES:
            weights['mission'] += 0.2
        if detection_probability > 0.7:
            weights['stealth'] += 0.2
            weights['success'] -= 0.1
        elif detection_probability < 0.3:
            weights['value'] += 0.1
        if current_asset_id == self.apt3_config['primary_objective']:
            weights['mission'] += 0.3
        elif self._is_adjacent_to_rtu(current_asset_id):
            weights['mission'] += 0.2
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}

    def _calculate_mission_alignment(self, action_type: ActionType, target_asset_id: str, vuln_id: Optional[str] = None) -> float:
        alignment = 0.5
        if target_asset_id == self.apt3_config['primary_objective']:
            alignment = 1.0
        elif target_asset_id in self.apt3_config['secondary_objectives']:
            alignment = 0.8
        if vuln_id and self._uses_apt3_techniques(vuln_id):
            alignment += 0.2
        if self.current_mission_phase == MissionPhase.COMMAND_AND_CONTROL:
            if action_type == ActionType.LATERAL_MOVEMENT and target_asset_id == self.apt3_config['primary_objective']:
                alignment += 0.3
        elif self.current_mission_phase == MissionPhase.ACTIONS_ON_OBJECTIVES:
            if action_type == ActionType.EXPLOITATION and target_asset_id == self.apt3_config['primary_objective']:
                alignment += 0.3
        return min(alignment, 1.0)

    def _get_asset_by_id(self, asset_id: str):
        for asset in self.system.assets:
            if str(asset.asset_id) == str(asset_id):
                return asset
        return None

    def _find_connected_assets(self, asset_id: str) -> List[str]:
        connected = []
        for conn in self.system.connections:
            if str(conn.from_asset.asset_id) == str(asset_id):
                connected.append(str(conn.to_asset.asset_id))
            elif str(conn.to_asset.asset_id) == str(asset_id):
                connected.append(str(conn.from_asset.asset_id))
        return connected

    def _is_adjacent_to_rtu(self, asset_id: str) -> bool:
        connected = self._find_connected_assets(asset_id)
        return self.apt3_config['primary_objective'] in connected

    def _moves_closer_to_rtu(self, current_asset_id: str, target_asset_id: str) -> bool:
        if not self.attack_graph or not hasattr(self.attack_graph, 'graph'):
            return target_asset_id == self.apt3_config['primary_objective']
        try:
            import networkx as nx
            current_distance = nx.shortest_path_length(
                self.attack_graph.graph, current_asset_id, self.apt3_config['primary_objective']
            )
            target_distance = nx.shortest_path_length(
                self.attack_graph.graph, target_asset_id, self.apt3_config['primary_objective']
            )
            return target_distance < current_distance
        except:
            return target_asset_id == self.apt3_config['primary_objective']

    def _calculate_exploit_probability(self, vuln, asset) -> float:
        base_prob = getattr(vuln, 'epss', 0.1)
        if getattr(vuln, 'exploit', False):
            base_prob *= 1.5
        return min(base_prob, 0.95)

    def _calculate_exploit_cost(self, vuln, asset, state=None, component_id=None) -> float:
        if self.cost_cache and 'exploit_costs' in self.cost_cache:
            vuln_key = f"{getattr(vuln, 'cve_id', 'unknown')}:{getattr(asset, 'asset_id', 'unknown')}:{component_id}"
            cost = self.cost_cache['exploit_costs'].get(vuln_key, None)
            if cost is not None:
                logger.debug(f"Using cached exploit cost for {vuln_key}: ${cost:.2f}")
                return cost
        if self.cost_calculator:
            try:
                return self.cost_calculator.calculate_exploit_cost(vuln, state, asset, component_id=component_id)
            except Exception as e:
                logger.warning(f"Error calculating exploit cost for {getattr(vuln, 'cve_id', 'unknown')}: {e}")
        return 50 + getattr(vuln, 'cvss', 5.0) * 20

    def _calculate_exploit_value(self, vuln, asset) -> float:
        business_value = getattr(asset, 'business_value', asset.criticality_level * 5000)
        vuln_impact = getattr(vuln, 'cvss', 5.0) / 10.0
        return business_value * vuln_impact

    def _get_apt3_technique_multiplier(self, vuln) -> float:
        multiplier = 1.0
        cve_id = getattr(vuln, 'cve_id', '')
        if cve_id in self.apt3_config['priority_cves']:
            multiplier *= self.apt3_config['priority_cves'][cve_id]
        techniques = getattr(vuln, 'mitre_techniques', [])
        if any(tech in self.apt3_config['preferred_techniques'] for tech in techniques):
            multiplier *= 1.3
        return multiplier

    def _calculate_asset_strategic_value(self, asset) -> float:
        return getattr(asset, 'business_value', asset.criticality_level * 5000)

    def _is_privilege_escalation_vuln(self, vuln) -> bool:
        techniques = getattr(vuln, 'mitre_techniques', [])
        return 'T1068' in techniques or getattr(vuln, 'cvss', 0) >= 7.0

    def _calculate_privilege_escalation_value(self, asset) -> float:
        return asset.criticality_level * 1000

    def _has_persistence(self, asset_id: str) -> bool:
        return f"persistence_{asset_id}" in self.intelligence_gathered

    def _calculate_persistence_value(self, asset) -> float:
        return asset.criticality_level * 500

    def _uses_apt3_techniques(self, vuln_id: str) -> bool:
        # Simplified check for APT3 techniques
        apt3_cves = {'CVE-2018-13379', 'ZERO-DAY-001', 'CVE-2015-3113'}
        return vuln_id in apt3_cves

    def _calculate_initial_access_probability(self, vuln, asset) -> float:
        """Calculate success probability for initial access actions."""
        base_prob = getattr(vuln, 'epss', 0.1)
        if getattr(vuln, 'exploit', False):
            base_prob *= 1.5
        if 'AV:N' in getattr(vuln, 'cvssV3Vector', ''):
            base_prob *= 1.2  # Boost for network-accessible vulnerabilities
        return min(base_prob, 0.95)

    def update_state(self, action_result: Dict):
        """Update strategic state based on action result."""
        logger.debug(f"Updating strategic state with action result: {action_result}")

        if action_result.get('action_result', False):
            target_asset = action_result.get('target_asset')
            if target_asset:
                self.compromised_assets.add(str(target_asset))
                logger.debug(f"Added asset {target_asset} to compromised assets")

            target_vuln = action_result.get('target_vuln')
            target_comp = action_result.get('target_component', '0')

            if target_vuln and target_asset:
                vuln_key = create_vuln_key(target_vuln, target_asset, target_comp)
                self.exploited_vulnerabilities.add(vuln_key)
                logger.info(f"Added {vuln_key} to exploited vulnerabilities")

                if vuln_key in self.exploit_failures:
                    logger.info(f"Resetting failure count for {vuln_key} due to successful exploit")
                    del self.exploit_failures[vuln_key]
        else:
            if action_result.get('action_type') in ['initial_access', 'exploitation', 'lateral_movement',
                                                    'privilege_escalation']:
                target_vuln = action_result.get('target_vuln')
                target_asset = action_result.get('target_asset')
                target_comp = action_result.get('target_component', '0')

                if target_vuln and target_asset:
                    vuln_key = create_vuln_key(target_vuln, target_asset, target_comp)
                    self.exploit_failures[vuln_key] = self.exploit_failures.get(vuln_key, 0) + 1
                    failure_count = self.exploit_failures[vuln_key]
                    logger.info(f"Recorded failure for {vuln_key}, total failures: {failure_count}")

                    if failure_count >= 2:
                        logger.warning(
                            f"Vulnerability {vuln_key} has failed {failure_count} times - will be deprioritized")

        logger.debug(f"Strategic state: {len(self.compromised_assets)} compromised assets, "
                     f"{len(self.exploited_vulnerabilities)} exploited vulnerabilities, "
                     f"{len(self.exploit_failures)} tracked failure counts")

    def _assess_fallback_lateral_movement(self, current_asset, target_asset, attacker_budget: float,
                                        detection_probability: float) -> List[ActionOpportunity]:
        """
        Fallback lateral movement using vulnerabilities on the current asset.
        This ensures the simulation can still function when hybrid opportunities are not available.
        """
        opportunities = []
        
        # Find any exploitable vulnerabilities on the current asset
        for comp in current_asset.components:
            for vuln in comp.vulnerabilities:
                if vuln.is_patched or vuln.is_exploited:
                    continue
                    
                vuln_key = create_vuln_key(vuln.cve_id, current_asset.asset_id, str(comp.id))
                cost = self.cost_cache.get('exploit_costs', {}).get(vuln_key, 50 + getattr(vuln, 'cvss', 5.0) * 20)
                
                if cost > attacker_budget:
                    continue
                    
                # Calculate success probability
                base_prob = min(0.9, getattr(vuln, 'epss', 0.1) * 1.5)
                if getattr(vuln, 'exploit', False):
                    base_prob *= 1.2
                if 'AV:N' in getattr(vuln, 'cvssV3Vector', ''):
                    base_prob *= 1.1
                success_prob = max(0.1, min(0.9, base_prob))
                
                target_value = self._calculate_asset_strategic_value(target_asset)
                if target_asset.asset_id == self.apt3_config['primary_objective']:
                    target_value *= 3.0
                elif target_asset.asset_id in self.apt3_config['secondary_objectives']:
                    target_value *= 1.8
                    
                detection_risk = detection_probability * 0.8
                mission_alignment = self._calculate_mission_alignment(
                    ActionType.LATERAL_MOVEMENT, target_asset.asset_id, vuln.cve_id
                )
                
                opportunity = ActionOpportunity(
                    action_type=ActionType.LATERAL_MOVEMENT,
                    target_asset_id=target_asset.asset_id,
                    target_vuln_id=vuln.cve_id,  # Single vulnerability ID
                    success_probability=success_prob,
                    resource_cost=cost,
                    expected_value=target_value,
                    detection_risk=detection_risk,
                    mission_alignment=mission_alignment,
                    target_component_id=str(comp.id)
                )
                opportunities.append(opportunity)
                logger.debug(f"Fallback lateral movement: {vuln.cve_id} on current asset, prob={success_prob:.3f}")
                
        return opportunities