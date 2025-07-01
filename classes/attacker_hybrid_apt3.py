import logging
from typing import Dict, Optional, List, Tuple, Set
from classes.state import System, State, Asset, Vulnerability, KillChainStage, create_vuln_key
from classes.mitre import MitreTTP, mitre_to_ckc_mapping, mitre_tactics_by_name
from classes.attack_graph_apt3 import AttackGraph
from classes.attacker_posg_apt3 import AttackerPOMDPPolicy
from classes.strategic_decision_manager import APT3StrategicDecisionManager, ActionType, MissionPhase
import networkx as nx
import time
import random
import os

logger = logging.getLogger(__name__)
logger.debug(f"Loading attacker_hybrid_apt3 from: {os.path.abspath(__file__)}")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HybridGraphPOSGAttackerAPT3:
    def __init__(self, system: System, mitre_mapper=None, cwe_canfollow_path: str = None,
                 cost_aware: bool = True, detection_averse: bool = True, enhanced_exploit_priority: bool = True,
  
                 sophistication_level: float = 0.5, cost_calculator=None, cost_cache: Dict = None):
        self.system = system
        self.state = State(system=system, k=KillChainStage.RECONNAISSANCE)
        self.cost_calculator = cost_calculator
        self.cost_cache = cost_cache or {}
        # Store parameters as instance variables
        self.cost_aware = cost_aware
        self.detection_averse = detection_averse
        self.sophistication_level = max(0.0, min(1.0, sophistication_level))

        # Initialize failure tracking - MUST be empty on init
        self.exploit_failures = {}

        try:
            self._build_system_graph(system)
            logger.info("System graph built successfully")
        except Exception as e:
            logger.error(f"Failed to build system graph: {e}")
            raise
        try:
            self.attack_graph = AttackGraph(
                system_graph=self.graph,
                mitre_mapper=mitre_mapper,
                cwe_canfollow_path=cwe_canfollow_path
            )
            self.attack_graph.build_attack_graph()
            logger.info("Attack graph built successfully")
        except Exception as e:
            logger.error(f"Failed to build attack graph: {e}")
            raise
        try:
            self.posg_policy = AttackerPOMDPPolicy(
                cost_aware=cost_aware,
                detection_averse=detection_averse,
                enhanced_exploit_priority=enhanced_exploit_priority,
                system=self.system
            )
            logger.info("POMDP Attacker Policy initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize POMDP Attacker Policy: {e}")
            self.posg_policy = None
            raise

        # Initialize strategic manager with clean state
        self.strategic_manager = APT3StrategicDecisionManager(
            attack_graph=self.attack_graph,
            system=system,
            cost_calculator=cost_calculator,
            cost_cache=self.cost_cache,
            attacker_policy=self.posg_policy
        )
        logger.info("Strategic Decision Manager initialized")

        self.current_path = None
        self.current_path_idx = 0
        self.last_observation = None
        self.path_failures = 0
        self.last_path_failure_time = 0
        self.backoff_base_delay = 1.0
        self.max_backoff_delay = 60.0
        self.max_path_failures = 3
        self.path_cache: List[Tuple[List[str], KillChainStage, float]] = []
        self.cache_size = 20
        self.cache_last_updated = 0
        self.cache_max_age = 300
        self.cache_hits = 0
        self.cache_misses = 0
        self.current_stage = KillChainStage.RECONNAISSANCE
        self.current_compromised_node = None
        self.compromised_nodes: Set[str] = set()
        self.exploited_vulns: Set[str] = set()
        self.decision_history = []
        self.strategic_mode_enabled = True

        logger.info(
            f"Enhanced APT3 Hybrid Attacker initialized with {len(self.system.assets)} assets, "
            f"cost_aware={self.cost_aware}, detection_averse={self.detection_averse}, "
            f"sophistication_level={self.sophistication_level}"
        )

    def select_action(self, state: State, force_action_type: Optional[str] = None) -> Dict:
        if state is None or not isinstance(state, State):
            logger.error("Invalid state provided")
            return {'action_type': 'pause', 'reason': 'invalid_state'}
        if not hasattr(state, 'system') or state.system is None:
            logger.error("State has no system attribute")
            return {'action_type': 'pause', 'reason': 'missing_system'}

        try:
            self.state = state
            current_stage = KillChainStage(state.k)
            self.current_stage = current_stage
            self.strategic_manager.compromised_assets = self.compromised_nodes.copy()
            self.strategic_manager.exploited_vulnerabilities = self.exploited_vulns.copy()
            current_position = self._determine_current_position(state)

            if not current_position:
                logger.warning("No current position, falling back")
                return self._fallback_select_action(state)

            logger.debug(f"Current position: {current_position}, Stage: {current_stage.name}")

            recon_count = sum(1 for d in self.decision_history
                              if d.get('action', {}).get('action_type') == 'reconnaissance')
            if recon_count >= 5 and not force_action_type:
                logger.info("Excessive reconnaissance detected, selecting forced action type")
                mission_phase = self.strategic_manager.current_mission_phase.value if hasattr(self.strategic_manager,
                                                                                              'current_mission_phase') else MissionPhase.RECONNAISSANCE
                if current_position == 'internet':
                    force_action_type = ActionType.DELIVERY.value
                elif mission_phase == MissionPhase.EXPLOITATION:
                    force_action_type = random.choice([ActionType.EXPLOITATION.value, ActionType.LATERAL_MOVEMENT.value,
                                                       ActionType.PRIVILEGE_ESCALATION.value])
                elif mission_phase == MissionPhase.INSTALLATION:
                    force_action_type = ActionType.PERSISTENCE.value
                elif mission_phase == MissionPhase.COMMAND_AND_CONTROL:
                    force_action_type = ActionType.COMMAND_AND_CONTROL.value
                elif mission_phase == MissionPhase.ACTIONS_ON_OBJECTIVES:
                    force_action_type = ActionType.EXFILTRATION.value
                else:
                    force_action_type = ActionType.EXPLOITATION.value

            detection_prob = self.posg_policy.belief_state.detection_prob if self.posg_policy and self.posg_policy.belief_state else 0.0
            remaining_budget = getattr(self, '_remaining_attacker_budget', 15000)

            if self.strategic_mode_enabled and self.path_failures < self.max_path_failures:
                strategic_opportunity = self.strategic_manager.make_strategic_decision(
                    current_asset_id=current_position,
                    state=state,
                    attacker_budget=remaining_budget,
                    detection_probability=detection_prob,
                    attacker_policy=self.posg_policy,
                    force_action_type=force_action_type
                )

                if strategic_opportunity is None or strategic_opportunity.action_type is None:
                    logger.warning(f"Strategic manager returned invalid opportunity: {strategic_opportunity}")
                    return self._fallback_select_action(state, ActionType.EXPLOITATION.value)

                if force_action_type == ActionType.DELIVERY.value and strategic_opportunity.action_type == ActionType.RECONNAISSANCE:
                    logger.warning("Forced DELIVERY but got reconnaissance, retrying with fallback")
                    return self._fallback_select_action(state, ActionType.DELIVERY.value)

                vuln_key = f"{strategic_opportunity.target_vuln_id}:{strategic_opportunity.target_asset_id}" if strategic_opportunity.target_vuln_id else None
                if vuln_key:
                    failure_count = self.strategic_manager.exploit_failures.get(vuln_key, 0)
                    logger.debug(f"Checking {vuln_key}: {failure_count} failures")
                    if failure_count >= 3:
                        logger.warning(f"Opportunity {vuln_key} has {failure_count} failures, switching to fallback")
                        return self._fallback_select_action(state, ActionType.EXPLOITATION.value)

                action = self._convert_opportunity_to_action(strategic_opportunity, current_stage)
                if not action or not isinstance(action, dict) or 'action_type' not in action:
                    logger.error(
                        f"Invalid action converted from opportunity: {strategic_opportunity}, action: {action}")
                    return self._fallback_select_action(state, ActionType.EXPLOITATION.value)

                if action.get('action_type') in ['initial_access', 'exploitation', 'lateral_movement',
                                                 'privilege_escalation', 'persistence', 'command_and_control',
                                                 'exfiltration']:
                    vuln_key = create_vuln_key(action.get('target_vuln', ''),
                                               str(action.get('target_asset', '')),
                                               str(action.get('target_component', '0')))
                    cost = self.cost_cache.get('exploit_costs', {}).get(vuln_key, 50.0)
                    if cost > remaining_budget:
                        logger.warning(f"Action {action['action_type']} cost {cost} exceeds budget {remaining_budget}")
                        return self._fallback_select_action(state, ActionType.EXPLOITATION.value)

                self.decision_history.append({
                    'step': len(self.decision_history),
                    'position': current_position,
                    'opportunity': strategic_opportunity,
                    'action': action
                })
                logger.info(f"Strategic action: {action.get('action_type')} on asset {action.get('target_asset')}")
                self.last_observation = action
                return action
            else:
                logger.info(
                    f"Path failures ({self.path_failures}) reached or strategic mode disabled, switching to fallback")
                return self._fallback_select_action(state, ActionType.EXPLOITATION.value)
        except Exception as e:
            logger.error(f"Error in action selection: {e}", exc_info=True)
            return self._fallback_select_action(state, ActionType.EXPLOITATION.value)

    def _fallback_select_action(self, state: State, force_action_type: Optional[str] = None) -> Dict:
        if state is None or not hasattr(state, 'system') or state.system is None:
            logger.error("Invalid state or missing system in fallback")
            return {'action_type': 'pause', 'reason': 'invalid_state'}

        current_position = self._determine_current_position(state)
        compromised_assets = [str(asset.asset_id) for asset in state.system.assets if asset.is_compromised]
        remaining_budget = getattr(self, '_remaining_attacker_budget', 15000)

        logger.info(f"Fallback action selection from position: {current_position}")
        logger.debug(f"Compromised assets: {compromised_assets}")
        logger.debug(f"Current exploit failures: {dict(self.strategic_manager.exploit_failures)}")

        viable_actions = []

        # Helper function to validate vulnerability existence
        def validate_vulnerability(asset_id, vuln_id, comp_id):
            asset = self._get_asset_by_id(asset_id)
            if not asset:
                return False
            for comp in asset.components:
                if str(comp.id) == str(comp_id):
                    for vuln in comp.vulnerabilities:
                        if vuln.cve_id == vuln_id:
                            # Check if vulnerability is available for exploitation
                            if vuln.is_patched or vuln.is_exploited:
                                return False
                            return True
            return False

        # 1. Handle forced action type
        if force_action_type:
            if force_action_type == ActionType.DELIVERY.value and current_position == 'internet':
                entry_points = getattr(self.attack_graph, 'entry_nodes', [])
                for asset_id in entry_points:
                    asset = self._get_asset_by_id(asset_id)
                    if not asset or asset.is_compromised:
                        continue
                    for comp in asset.components:
                        for vuln in comp.vulnerabilities:
                            vuln_key = f"{vuln.cve_id}:{asset.asset_id}:{comp.id}"
                            if vuln.is_patched or vuln.is_exploited:
                                continue
                            failure_count = self.strategic_manager.exploit_failures.get(vuln_key, 0)
                            if failure_count >= 3:
                                continue
                            cost = self.cost_cache.get('exploit_costs', {}).get(vuln_key, 50.0)
                            if cost > remaining_budget:
                                continue
                            probability = max(0.4, min(0.85, getattr(vuln, 'epss', 0.1) * 1.8))
                            if failure_count > 0:
                                probability *= (0.6 ** failure_count)
                            viable_actions.append({
                                'action_type': 'initial_access',
                                'target_vuln': vuln.cve_id,
                                'target_asset': asset.asset_id,
                                'target_component': comp.id,
                                'probability': probability,
                                'cvss': getattr(vuln, 'cvss', 5.0),
                                'epss': getattr(vuln, 'epss', 0.1),
                                'exploit': getattr(vuln, 'exploit', False),
                                'failure_count': failure_count,
                                'priority': 'high'
                            })
            elif force_action_type == ActionType.LATERAL_MOVEMENT.value and current_position != 'internet':
                connected_assets = [conn.to_asset.asset_id for conn in state.system.connections
                                    if (hasattr(conn, 'from_asset') and conn.from_asset and
                                        str(conn.from_asset.asset_id) == current_position and
                                        not conn.to_asset.is_compromised)]
                for target_asset_id in connected_assets:
                    # For lateral movement, we need vulnerabilities on the CURRENT asset (not target)
                    current_asset = self._get_asset_by_id(current_position)
                    if not current_asset:
                        continue
                    
                    # Look for vulnerabilities on the current asset that can be used for lateral movement
                    for comp in current_asset.components:
                        for vuln in comp.vulnerabilities:
                            vuln_key = f"{vuln.cve_id}:{current_asset.asset_id}:{comp.id}"
                            if vuln.is_patched or vuln.is_exploited:
                                continue
                            failure_count = self.strategic_manager.exploit_failures.get(vuln_key, 0)
                            if failure_count >= 3:
                                continue
                            cost = self.cost_cache.get('exploit_costs', {}).get(vuln_key, 50.0)
                            if cost > remaining_budget:
                                continue
                            probability = max(0.3, min(0.85, getattr(vuln, 'epss', 0.1) * 1.5))
                            if failure_count > 0:
                                probability *= (0.6 ** failure_count)
                            if validate_vulnerability(current_position, vuln.cve_id, comp.id):
                                viable_actions.append({
                                    'action_type': 'lateral_movement',
                                    'target_vuln': vuln.cve_id,
                                    'target_asset': target_asset_id,  # Target asset for movement
                                    'target_component': comp.id,      # Component on current asset
                                    'probability': probability,
                                    'cvss': getattr(vuln, 'cvss', 5.0),
                                    'epss': getattr(vuln, 'epss', 0.1),
                                    'exploit': getattr(vuln, 'exploit', False),
                                    'failure_count': failure_count,
                                    'priority': 'high'
                                })
            # [Other action types remain unchanged]

        # 2. Non-forced actions
        if not force_action_type or force_action_type == ActionType.RECONNAISSANCE.value:
            if current_position != 'internet':
                viable_actions.append({
                    'action_type': 'reconnaissance',
                    'target_asset': current_position,
                    'probability': 0.95,
                    'cvss': 0.0,
                    'epss': 0.0,
                    'exploit': False,
                    'priority': 'medium'
                })
                for conn in state.system.connections:
                    if (hasattr(conn, 'from_asset') and conn.from_asset and
                            str(conn.from_asset.asset_id) == current_position):
                        viable_actions.append({
                            'action_type': 'reconnaissance',
                            'target_asset': str(conn.to_asset.asset_id),
                            'probability': 0.85,
                            'cvss': 0.0,
                            'epss': 0.0,
                            'exploit': False,
                            'priority': 'low'
                        })

        # 3. Exploit-based actions
        if not force_action_type or force_action_type in [ActionType.DELIVERY.value, ActionType.EXPLOITATION.value,
                                                          ActionType.LATERAL_MOVEMENT.value,
                                                          ActionType.PRIVILEGE_ESCALATION.value]:
            if compromised_assets and current_position != 'internet':
                connected_assets = [conn.to_asset.asset_id for conn in state.system.connections
                                    if (hasattr(conn, 'from_asset') and conn.from_asset and
                                        str(conn.from_asset.asset_id) == current_position and
                                        not conn.to_asset.is_compromised)]
                for target_asset_id in connected_assets:
                    # For lateral movement, we need vulnerabilities on the CURRENT asset (not target)
                    current_asset = self._get_asset_by_id(current_position)
                    if not current_asset:
                        continue
                    
                    # Look for vulnerabilities on the current asset that can be used for lateral movement
                    for comp in current_asset.components:
                        for vuln in comp.vulnerabilities:
                            vuln_key = f"{vuln.cve_id}:{current_asset.asset_id}:{comp.id}"
                            if vuln.is_patched or vuln.is_exploited:
                                continue
                            failure_count = self.strategic_manager.exploit_failures.get(vuln_key, 0)
                            if failure_count >= 3:
                                continue
                            cost = self.cost_cache.get('exploit_costs', {}).get(vuln_key, 50.0)
                            if cost > remaining_budget:
                                continue
                            probability = max(0.3, min(0.85, getattr(vuln, 'epss', 0.1) * 1.5))
                            if failure_count > 0:
                                probability *= (0.6 ** failure_count)
                            if validate_vulnerability(current_position, vuln.cve_id, comp.id):
                                viable_actions.append({
                                    'action_type': 'lateral_movement',
                                    'target_vuln': vuln.cve_id,
                                    'target_asset': target_asset_id,  # Target asset for movement
                                    'target_component': comp.id,      # Component on current asset
                                    'probability': probability,
                                    'cvss': getattr(vuln, 'cvss', 5.0),
                                    'epss': getattr(vuln, 'epss', 0.1),
                                    'exploit': getattr(vuln, 'exploit', False),
                                    'failure_count': failure_count,
                                    'priority': 'medium' if failure_count == 0 else 'low'
                                })
            # [Other exploit-based actions remain unchanged]

        # Select the best action from viable options
        if viable_actions:
            priority_order = {'high': 3, 'medium': 2, 'low': 1}
            viable_actions.sort(key=lambda x: (
                priority_order.get(x.get('priority', 'low'), 1),
                x['probability'],
                -x.get('failure_count', 0)
            ), reverse=True)
            action = viable_actions[0]
            tactic = self._get_tactic_for_stage(self.current_stage)
            action['tactic'] = tactic if tactic else 'Unknown'
            if action['action_type'] == 'lateral_movement':
                current_asset = self._get_asset_by_id(current_position)
                logger.info(f"[DIAG] Fallback selection: Vulnerabilities on asset {current_position}:")
                for comp in current_asset.components:
                    for vuln in comp.vulnerabilities:
                        logger.info(f"[DIAG]   {vuln.cve_id} (patched={vuln.is_patched}, exploited={vuln.is_exploited}) on component {comp.id}")
                logger.info(f"[DIAG] Fallback selected vuln: {action['target_vuln']} (for movement to {action['target_asset']}) on component {action.get('target_component', 'unknown')}")
                logger.info(f"[DIAG] Current position: {current_position}, Current asset ID: {current_asset.asset_id if current_asset else 'None'}")
            logger.info(f"Fallback: Selected {action['action_type']} on {action['target_asset']} "
                        f"with probability {action['probability']:.3f} "
                        f"(failures: {action.get('failure_count', 0)})")
            self.last_observation = action
            return action

        # Last resort: POMDP or minimal reconnaissance
        logger.warning("No viable actions found, using last resort actions")
        if self.posg_policy:
            try:
                tactic = self.posg_policy.select_tactic(self.state, current_position)
                if tactic:
                    action = self.posg_policy.perform_action(self.state, tactic)
                    action.setdefault('probability', max(0.5, action.get('expected_score', 0.5)))
                    action.setdefault('tactic', tactic.name)
                    logger.info(f"POMDP fallback action: {action['action_type']}")
                    self.last_observation = action
                    return action
            except Exception as e:
                logger.error(f"POMDP fallback failed: {e}")

        # Absolute last resort: reconnaissance with reduced probability
        fallback_action = {
            'action_type': 'reconnaissance',
            'target_asset': current_position if current_position != 'internet' else 'unknown',
            'probability': 0.7,
            'tactic': 'Reconnaissance',
            'reason': 'absolute_fallback'
        }
        logger.info("Using absolute fallback: reconnaissance")
        self.last_observation = fallback_action
        return fallback_action

    def _determine_current_position(self, state: State) -> Optional[str]:
        if (hasattr(self, 'current_compromised_node') and
                self.current_compromised_node and
                self.current_compromised_node in self.attack_graph.graph.nodes):
            node_data = self.attack_graph.graph.nodes[self.current_compromised_node]
            if node_data.get('type') == 'asset':
                return self.current_compromised_node
            elif node_data.get('type') == 'vulnerability':
                return node_data.get('asset_id')
        compromised_assets = [str(asset.asset_id) for asset in state.system.assets if asset.is_compromised]
        if compromised_assets:
            asset_values = []
            for asset_id in compromised_assets:
                asset = self._get_asset_by_id(asset_id)
                if asset:
                    value = getattr(asset, 'business_value', asset.criticality_level * 5000)
                    asset_values.append((asset_id, value))
            if asset_values:
                asset_values.sort(key=lambda x: x[1], reverse=True)
                return asset_values[0][0]
        return 'internet'

    def _convert_opportunity_to_action(self, opportunity, current_stage: KillChainStage) -> Dict:
        if opportunity is None:
            logger.error("Cannot convert None opportunity to action")
            self.path_failures += 1
            return {'action_type': 'pause', 'reason': 'none_opportunity'}
        if not hasattr(opportunity, 'action_type'):
            logger.error(f"Opportunity missing action_type attribute: {opportunity}")
            self.path_failures += 1
            return {'action_type': 'pause', 'reason': 'missing_action_type'}
        if opportunity.action_type is None:
            logger.error("Opportunity has None action_type")
            self.path_failures += 1
            return {'action_type': 'pause', 'reason': 'none_action_type'}
        if not hasattr(opportunity, 'target_asset_id') or opportunity.target_asset_id is None:
            logger.error(f"Opportunity missing target_asset_id: {opportunity}")
            self.path_failures += 1
            return {'action_type': 'pause', 'reason': 'missing_target_asset'}
        if not hasattr(self, 'state') or self.state is None:
            logger.error("State is None or not initialized")
            self.path_failures += 1
            return {'action_type': 'pause', 'reason': 'missing_state'}
        if not hasattr(self.state, 'system') or self.state.system is None:
            logger.error("State has no system attribute")
            self.path_failures += 1
            return {'action_type': 'pause', 'reason': 'missing_system'}

        current_position = self._determine_current_position(self.state)
        vuln_key = f"{opportunity.target_vuln_id}:{opportunity.target_asset_id}:{opportunity.target_component_id or '0'}" if opportunity.target_vuln_id else None
        if vuln_key:
            failure_count = self.strategic_manager.exploit_failures.get(vuln_key, 0)
            logger.debug(f"Converting {vuln_key}: {failure_count} failures")
            if failure_count >= 3:
                logger.warning(f"Opportunity {vuln_key} has {failure_count} failures, rejecting")
                self.path_failures += 1
                return {'action_type': 'pause', 'reason': 'excessive_failures'}

        target_asset = self._get_asset_by_id(opportunity.target_asset_id)
        if not target_asset:
            logger.error(f"Invalid target asset ID: {opportunity.target_asset_id}")
            self.path_failures += 1
            return {'action_type': 'pause', 'reason': 'invalid_target_asset'}

        if opportunity.action_type in [ActionType.EXPLOITATION, ActionType.PRIVILEGE_ESCALATION, ActionType.DELIVERY,
                                       ActionType.LATERAL_MOVEMENT]:
            if not opportunity.target_vuln_id:
                logger.error(f"Exploit action missing target_vuln_id: {opportunity}")
                self.path_failures += 1
                return {'action_type': 'pause', 'reason': 'missing_vuln_id'}

            # Handle special cases for lateral movement
            if opportunity.action_type == ActionType.LATERAL_MOVEMENT:
                # Handle credential-based movement
                if opportunity.target_vuln_id == "CREDENTIAL_BASED":
                    logger.info(f"Converting to credential-based lateral movement for {opportunity.target_asset_id}")
                    base_action = {
                        'action_type': 'lateral_movement',
                        'target_vuln': 'CREDENTIAL_BASED',
                        'target_component': 'credentials',
                        'movement_type': 'credential_based'
                    }
                    return base_action
                
                # Handle hybrid lateral movement (combined vulnerability IDs)
                if '+' in opportunity.target_vuln_id:
                    logger.info(f"Converting to hybrid lateral movement for {opportunity.target_asset_id}")
                    base_action = {
                        'action_type': 'lateral_movement',
                        'target_vuln': opportunity.target_vuln_id,
                        'target_component': opportunity.target_component_id,
                        'movement_type': 'hybrid'
                    }
                    return base_action

            # For regular vulnerability-based actions, verify vulnerability exists
            vuln_found = False
            target_vuln = None
            target_component = None
            for comp in target_asset.components:
                for vuln in comp.vulnerabilities:
                    if vuln.cve_id == opportunity.target_vuln_id:
                        # Check actual system state, not belief state
                        if vuln.is_patched:
                            logger.error(
                                f"Attempted to exploit patched vulnerability {vuln.cve_id} on asset {target_asset.name}")
                            self.path_failures += 1
                            return {'action_type': 'pause', 'reason': 'vuln_patched'}
                        if vuln.is_exploited:
                            logger.warning(
                                f"Vulnerability {vuln.cve_id} is already exploited on asset {target_asset.name}")
                            self.path_failures += 1
                            return {'action_type': 'pause', 'reason': 'vuln_exploited'}
                        vuln_found = True
                        target_vuln = vuln
                        target_component = comp.id
                        break
                if vuln_found:
                    break
            if not vuln_found:
                logger.error(
                    f"Vulnerability {opportunity.target_vuln_id} not found on asset {opportunity.target_asset_id}")
                self.path_failures += 1
                return {'action_type': 'pause', 'reason': 'vuln_not_found'}

        logger.debug(f"Converting opportunity: {opportunity.action_type.value} on {opportunity.target_asset_id}")
        logger.debug(f"Current position: {current_position}")
        try:
            base_action = {
                'action_type': opportunity.action_type.value,
                'target_asset': opportunity.target_asset_id,
                'probability': getattr(opportunity, 'success_probability', 0.5),
                'expected_value': getattr(opportunity, 'expected_value', 0.0),
                'cost': getattr(opportunity, 'resource_cost', 0.0),
                'tactic': self._get_tactic_for_stage(current_stage),
                'strategic_score': getattr(opportunity, 'strategic_score', 0.0)
            }
            if opportunity.action_type == ActionType.EXPLOITATION:
                logger.info(f"Converting to exploitation action for {opportunity.target_asset_id}")
                base_action.update({
                    'action_type': 'exploitation',
                    'target_vuln': opportunity.target_vuln_id,
                    'target_component': target_component
                })
            elif opportunity.action_type == ActionType.DELIVERY:
                logger.info(f"Converting to initial_access action for {opportunity.target_asset_id}")
                base_action.update({
                    'action_type': 'initial_access',
                    'target_vuln': opportunity.target_vuln_id,
                    'target_component': target_component
                })
            elif opportunity.action_type == ActionType.LATERAL_MOVEMENT:
                logger.info(f"Converting to lateral_movement action for {opportunity.target_asset_id}")
                base_action.update({
                    'action_type': 'lateral_movement',
                    'target_vuln': opportunity.target_vuln_id,
                    'target_component': target_component
                })
            elif opportunity.action_type == ActionType.PRIVILEGE_ESCALATION:
                logger.info(f"Converting to privilege_escalation action for {opportunity.target_asset_id}")
                base_action.update({
                    'action_type': 'privilege_escalation',
                    'target_vuln': opportunity.target_vuln_id,
                    'target_component': target_component,
                    'escalation_type': 'privilege'
                })
            elif opportunity.action_type in [ActionType.PERSISTENCE, ActionType.COMMAND_AND_CONTROL,
                                             ActionType.EXFILTRATION, ActionType.RECONNAISSANCE]:
                base_action.update({
                    'action_type': opportunity.action_type.value
                })
            else:
                logger.warning(f"Unknown action type: {opportunity.action_type}, defaulting to pause")
                self.path_failures += 1
                return {'action_type': 'pause', 'reason': 'unknown_opportunity_type'}

            if opportunity.target_vuln_id:
                vuln_details = self._get_vulnerability_details(opportunity.target_vuln_id, opportunity.target_asset_id)
                base_action.update(vuln_details)
                if not base_action.get('target_vuln'):
                    logger.warning(f"Missing target_vuln in action: {base_action}")
                    base_action['target_vuln'] = opportunity.target_vuln_id
                if not base_action.get('target_component'):
                    logger.error(
                        f"Missing target_component for vuln {opportunity.target_vuln_id} on asset {opportunity.target_asset_id}")
                    self.path_failures += 1
                    return {'action_type': 'pause', 'reason': 'missing_component'}
            logger.debug(f"Converted action: {base_action}")
            return base_action
        except Exception as e:
            logger.error(f"Error converting opportunity to action: {e}")
            self.path_failures += 1
            return {'action_type': 'pause', 'reason': f'conversion_error_{str(e)}'}

    def _get_component_for_vuln(self, vuln_id: str, asset_id: str) -> Optional[str]:
        asset = self._get_asset_by_id(asset_id)
        if not asset:
            return None
        for comp in asset.components:
            for vuln in comp.vulnerabilities:
                if vuln.cve_id == vuln_id:
                    return comp.id
        return None

    def _get_vulnerability_details(self, vuln_id: str, asset_id: str) -> Dict:
        asset = self._get_asset_by_id(asset_id)
        if not asset:
            return {}
        for comp in asset.components:
            for vuln in comp.vulnerabilities:
                if vuln.cve_id == vuln_id:
                    return {
                        'cvss': getattr(vuln, 'cvss', 5.0),
                        'epss': getattr(vuln, 'epss', 0.1),
                        'exploit': getattr(vuln, 'exploit', False),
                        'ransomWare': getattr(vuln, 'ransomWare', False)
                    }
        return {}

    def _get_asset_by_id(self, asset_id: str):
        asset_id_str = str(asset_id)
        for asset in self.system.assets:
            if str(asset.asset_id) == asset_id_str:
                return asset
        return None

    def observe_result(self, action_result: Dict):
        """
        Update attacker state based on action results and synchronize failure counts.
        """
        if not action_result or not isinstance(action_result, dict):
            logger.warning("Invalid action result received")
            return

        # Store the observation in history
        if not hasattr(self, 'observation_history'):
            self.observation_history = []
        self.observation_history.append(action_result)

        # Update belief state
        if hasattr(self, 'posg_policy') and self.posg_policy and hasattr(self.posg_policy,
                                                                         'belief_state') and self.posg_policy.belief_state:
            try:
                self.posg_policy.update_belief(action_result)
            except Exception as e:
                logger.error(f"Failed to update belief state: {e}", exc_info=True)
        else:
            logger.warning("POMDP policy or belief state not properly initialized, skipping belief update")

        # Get relevant information from the action result
        action_type = action_result.get('action_type')
        target_vuln = action_result.get('target_vuln')
        target_asset = action_result.get('target_asset')
        target_component = action_result.get('target_component', '0')
        success = action_result.get('action_result', False)

        # Create a standardized vulnerability key
        vuln_key = None
        if target_vuln and target_asset:
            vuln_key = create_vuln_key(target_vuln, target_asset, target_component)

        # Update the strategic manager with the action result
        if hasattr(self, 'strategic_manager'):
            self.strategic_manager.update_state(action_result)

        # Synchronize exploit failures with strategic manager
        if vuln_key and action_type in ['exploit', 'exploitation', 'initial_access', 'lateral_movement',
                                        'privilege_escalation']:
            failure_count = self.strategic_manager.exploit_failures.get(vuln_key, 0)
            self.exploit_failures[vuln_key] = failure_count
            logger.info(f"Synchronized failure count for {vuln_key}: {failure_count} with strategic manager")
            if success:
                # On success, reset failure counts and update exploited vulnerabilities
                if vuln_key in self.exploit_failures:
                    del self.exploit_failures[vuln_key]
                    logger.info(f"Reset local failure count for {vuln_key} on success")
                if vuln_key in self.strategic_manager.exploit_failures:
                    del self.strategic_manager.exploit_failures[vuln_key]
                    logger.info(f"Reset strategic manager failure count for {vuln_key} on success")
                self.exploited_vulns.add(vuln_key)
                logger.info(f"Added {vuln_key} to exploited vulnerabilities")
                vuln, asset, comp = self._find_vulnerability_by_key(vuln_key)
                if vuln:
                    # Ensure system state is respected
                    if not vuln.is_patched:
                        vuln.is_exploited = True
                        logger.debug(f"Marked vulnerability {target_vuln} as exploited in system state")
                    else:
                        logger.error(f"Cannot mark {target_vuln} as exploited: it is patched in system state")
                        return
                    if hasattr(self, 'attack_graph') and hasattr(self.attack_graph, 'graph'):
                        for node_id in self.attack_graph.graph.nodes:
                            node_data = self.attack_graph.graph.nodes[node_id]
                            if (node_data.get('type') == 'vulnerability' and
                                    node_data.get('cve_id') == target_vuln and
                                    node_data.get('asset_id') == target_asset):
                                node_data['is_exploited'] = True
                                logger.debug(f"Marked attack graph node {node_id} as exploited")
                                break
                if asset:
                    asset.mark_as_compromised(True)
                    asset_node_id = str(target_asset)
                    self.compromised_nodes.add(asset_node_id)
                    self.current_compromised_node = asset_node_id
                    logger.info(f"Updated current compromised node to asset: {asset_node_id}")
                    if hasattr(self, 'attack_graph') and hasattr(self.attack_graph, 'graph'):
                        if asset_node_id in self.attack_graph.graph.nodes:
                            self.attack_graph.graph.nodes[asset_node_id]['is_compromised'] = True
                    self.strategic_manager.compromised_assets.add(asset_node_id)
                    self.strategic_manager.exploited_vulnerabilities.add(vuln_key)
                # Decrease path failures on success
                self.path_failures = max(0, self.path_failures - 1)
                logger.debug(f"Decremented path_failures to {self.path_failures} due to successful action")
            else:
                # On failure, increment path failures
                self.path_failures += 1
                logger.debug(f"Incremented path_failures to {self.path_failures} due to failed action")
        elif not success:
            # For non-vulnerability actions, increment path failures
            self.path_failures += 1
            logger.debug(f"Incremented path_failures to {self.path_failures} for non-vulnerability action failure")
        else:
            # Decrease path failures on success for non-vulnerability actions
            self.path_failures = max(0, self.path_failures - 1)
            logger.debug(
                f"Decremented path_failures to {self.path_failures} due to successful non-vulnerability action")

        logger.debug(f"Observed action result: {action_result}")


    def _find_vulnerability_by_key(self, vuln_key: str) -> tuple:
        try:
            parts = vuln_key.split(':')
            if len(parts) != 3:
                logger.error(f"Invalid vulnerability key format: {vuln_key}")
                return None, None, None
            cve_id, asset_id, comp_id = parts
            for asset in self.system.assets:
                if str(asset.asset_id) == asset_id:
                    for comp in asset.components:
                        if str(comp.id) == comp_id:
                            for vuln in comp.vulnerabilities:
                                if vuln.cve_id == cve_id:
                                    return vuln, asset, comp
            logger.warning(f"Vulnerability not found for key: {vuln_key}")
            return None, None, None
        except ValueError:
            logger.error(f"Invalid vulnerability key format: {vuln_key}")
            return None, None, None

    def _legacy_path_following_action(self, state: State) -> Dict:
        current_stage = KillChainStage(state.k)
        if (not hasattr(self, 'path_cache') or
                not self.path_cache or
                time.time() - self.cache_last_updated > self.cache_max_age or
                self.current_stage != current_stage):
            self.current_stage = current_stage
            self._populate_path_cache(current_stage)
        need_new_path = (
            not hasattr(self, 'current_path') or
            self.current_path is None or
            self.current_path_idx >= len(self.current_path) or
            not self._is_path_valid(self.current_path)
        )
        if need_new_path:
            path, stage, path_score = self._select_path()
            if not path:
                logger.warning("No valid path selected")
                return self._fallback_select_action(state)
            self.current_path = path
            self.current_stage = stage
            self.current_path_idx = 0
            logger.info(f"Selected new attack path with score {path_score:.4f}")
        if self.current_path_idx < len(self.current_path):
            path_node = self.current_path[self.current_path_idx]
            self.current_path_idx += 1
            node_type = self.attack_graph.graph.nodes[path_node].get('type') if path_node in self.attack_graph.graph.nodes else None
            if node_type == 'vulnerability':
                return self._create_exploit_action(path_node, current_stage)
            elif node_type == 'asset':
                return self._create_movement_action(path_node, current_stage)
        return self._fallback_select_action(state)

    def _create_exploit_action(self, vuln_node: str, current_stage: KillChainStage) -> Dict:
        node_data = self.attack_graph.graph.nodes.get(vuln_node, {})
        vuln_id = node_data.get('cve_id')
        asset_id = node_data.get('asset_id')
        comp_id = node_data.get('component_id')
        if not all([vuln_id, asset_id, comp_id]):
            logger.warning(f"Invalid vulnerability node: {vuln_node}")
            return self._fallback_select_action(self.state)
        probability = self._calculate_exploit_probability(vuln_id, asset_id)
        tactic = self._get_tactic_for_stage(current_stage)
        action = {
            'action_type': 'exploit',
            'target_vuln': vuln_id,
            'target_asset': asset_id,
            'target_component': comp_id,
            'probability': probability,
            'tactic': tactic,
            'path_node_type': 'vulnerability',
            'cvss': node_data.get('cvss', 5.0),
            'epss': node_data.get('epss', 0.1),
            'exploit': node_data.get('has_exploit', False)
        }
        logger.debug(f"Created exploit action: {action}")
        return action

    def _create_movement_action(self, asset_node: str, current_stage: KillChainStage) -> Dict:
        compromised_assets = [a.asset_id for a in self.state.system.assets if a.is_compromised]
        action_type = 'initial_access' if not compromised_assets else 'lateral_movement'
        tactic = self._get_tactic_for_stage(current_stage)
        action = {
            'action_type': action_type,
            'target_asset': asset_node,
            'probability': 0.8 if action_type == 'lateral_movement' else 0.6,
            'tactic': tactic,
            'path_node_type': 'asset'
        }
        logger.debug(f"Created movement action: {action}")
        return action

    def _calculate_exploit_probability(self, vuln_id: str, asset_id: str) -> float:
        for asset in self.system.assets:
            if str(asset.asset_id) == str(asset_id):
                for comp in asset.components:
                    for vuln in comp.vulnerabilities:
                        if vuln.cve_id == vuln_id:
                            base_prob = min(0.9, getattr(vuln, 'epss', 0.1) * 2.0)
                            base_prob += 0.2 if getattr(vuln, 'exploit', False) else 0.0
                            base_prob *= self.sophistication_level
                            return max(0.1, min(0.9, base_prob))
        return 0.1

    def _get_tactic_for_stage(self, stage: KillChainStage) -> str:
        if not isinstance(stage, KillChainStage):
            logger.error(f"Invalid stage type: {type(stage)}, defaulting to Reconnaissance")
            return "Reconnaissance"
        if self.posg_policy:
            try:
                current_position = self._determine_current_position(self.state)
                tactic = self.posg_policy.select_tactic(self.state, current_position)
                if tactic and hasattr(tactic, 'name'):
                    logger.debug(f"Selected tactic via POMDP policy: {tactic.name}")
                    return tactic.name
                else:
                    logger.warning(f"POMDP policy returned invalid tactic for stage {stage.name}")
            except Exception as e:
                logger.warning(f"Failed to select tactic via POMDP policy: {e}")
        stage_to_tactic = {
            KillChainStage.RECONNAISSANCE: "Reconnaissance",
            KillChainStage.WEAPONIZATION: "Resource Development",
            KillChainStage.DELIVERY: "Initial Access",
            KillChainStage.EXPLOITATION: "Execution",
            KillChainStage.INSTALLATION: "Persistence",
            KillChainStage.COMMAND_AND_CONTROL: "Command and Control",
            KillChainStage.ACTIONS_ON_OBJECTIVES: "Impact"
        }
        tactic = stage_to_tactic.get(stage, "Reconnaissance")
        logger.debug(f"Selected default tactic: {tactic} for stage {stage.name}")
        return tactic

    def _populate_path_cache(self, current_stage: KillChainStage):
        self.path_cache = []
        self.cache_last_updated = time.time()
        logger.debug("Path cache populated (legacy mode)")

    def _select_path(self) -> tuple:
        if hasattr(self.attack_graph, 'entry_nodes') and self.attack_graph.entry_nodes:
            entry = self.attack_graph.entry_nodes[0]
            if hasattr(self.attack_graph, 'high_value_targets') and self.attack_graph.high_value_targets:
                target = self.attack_graph.high_value_targets[0]
                return [entry, target], self.current_stage, 0.5
        return [], self.current_stage, 0.0

    def _is_path_valid(self, path: List[str]) -> bool:
        if not path:
            return False
        return all(node in self.attack_graph.graph.nodes for node in path)

    def _build_system_graph(self, system):
        G = nx.DiGraph()
        for asset in system.assets:
            G.add_node(str(asset.asset_id), type='asset', asset_name=asset.name, is_compromised=False)
        for conn in system.connections:
            if conn.from_asset and conn.to_asset:
                G.add_edge(
                    str(conn.from_asset.asset_id),
                    str(conn.to_asset.asset_id),
                    internet_facing=(conn.from_asset.asset_id == 'internet')
                )
        apt3_entry_points = {
            '1': 'Domain Controller misconfiguration (exposed Active Directory services)',
            '3': 'User Workstation phishing (spearphishing email exploit)',
            '4': 'VPN exploitation (CVE-2018-13379 - known APT3 vector)',
            '5': 'Web server compromise (internet-facing DMZ asset)'
        }
        entry_points = []
        for entry_id, justification in apt3_entry_points.items():
            if any(str(asset.asset_id) == entry_id for asset in system.assets):
                has_internet_connection = any(
                    conn.from_asset and conn.from_asset.asset_id == 'internet'
                    for conn in system.connections if conn.to_asset and conn.to_asset.asset_id == entry_id
                )
                is_phishing = 'phishing' in justification.lower()
                if has_internet_connection or is_phishing:
                    entry_points.append(entry_id)
                    if entry_id in G.nodes:
                        G.nodes[entry_id]['is_internet_facing'] = True
                        G.nodes[entry_id]['is_entry_point'] = True
                        G.nodes[entry_id]['apt3_entry_justification'] = justification
                        logger.info(f"APT3 entry point: asset {entry_id} - {justification}")
        self.graph = G
        self.entry_nodes = entry_points

    def get_decision_history(self) -> List[Dict]:
        return self.decision_history.copy()

    def get_strategic_metrics(self) -> Dict:
        if not self.decision_history:
            return {
                'total_decisions': 0,
                'successful_decisions': 0,
                'success_rate': 0.0,
                'action_type_distribution': {},
                'average_strategic_score': 0.0
            }
        successful_decisions = 0
        action_types = {}
        strategic_scores = []
        for decision in self.decision_history:
            opportunity = decision.get('opportunity')
            if opportunity:
                action_type = opportunity.action_type.value
                action_types[action_type] = action_types.get(action_type, 0) + 1
                strategic_scores.append(opportunity.strategic_score)
        return {
            'total_decisions': len(self.decision_history),
            'action_type_distribution': action_types,
            'average_strategic_score': sum(strategic_scores) / len(strategic_scores) if strategic_scores else 0.0,
            'strategic_mode_enabled': self.strategic_mode_enabled
        }

    def set_strategic_mode(self, enabled: bool):
        self.strategic_mode_enabled = enabled
        logger.info(f"Strategic decision-making mode {'enabled' if enabled else 'disabled'}")

    def get_current_position_info(self) -> Dict:
        current_pos = self._determine_current_position(self.state) if hasattr(self, 'state') else None
        return {
            'current_compromised_node': self.current_compromised_node,
            'determined_position': current_pos,
            'compromised_nodes': list(self.compromised_nodes),
            'exploited_vulnerabilities': list(self.exploited_vulns),
            'mission_phase': self.strategic_manager.current_mission_phase.value if hasattr(self.strategic_manager, 'current_mission_phase') else 'unknown'
        }

    def reset_state(self):
        """Reset attacker state for new simulation run."""
        self.current_compromised_node = None
        self.compromised_nodes.clear()
        self.exploited_vulns.clear()
        self.decision_history.clear()
        self.current_path = None
        self.current_path_idx = 0
        self.path_failures = 0

        # Clear any local exploit failure tracking
        if hasattr(self, 'exploit_failures'):
            self.exploit_failures.clear()

        # Reinitialize strategic manager to ensure fresh state
        self.strategic_manager = APT3StrategicDecisionManager(
            attack_graph=self.attack_graph,
            system=self.system,
            cost_calculator=self.cost_calculator,
            cost_cache=self.cost_cache,
            attacker_policy=self.posg_policy
        )

        if hasattr(self, 'posg_policy') and self.posg_policy:
            self.posg_policy.reset()
            self.posg_policy.initialize_belief_state(self.system)

        logger.info("Attacker state reset for new simulation, including new StrategicDecisionManager instance")