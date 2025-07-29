# transition.py
import random
import math
from typing import List
from classes.state import State, Vulnerability, KillChainStage, create_vuln_key
import logging
from collections import defaultdict
logger = logging.getLogger(__name__)

class TransitionFunction:
    def __init__(self, exploit_success_base_rate: float = 0.3, stage_progression_threshold: int = 2):
        self.exploit_success_base_rate = exploit_success_base_rate
        self.stage_progression_threshold = stage_progression_threshold

    def apply_actions(self, state: State, defender_actions: List[Vulnerability],
                     attacker_actions: List[Vulnerability], current_step: int = None) -> None:
        """
        Apply defender and attacker actions to the system state.
        
        Args:
            state: Current system state
            defender_actions: List of vulnerabilities to patch
            attacker_actions: List of vulnerabilities to exploit
            current_step: Current simulation step (for logging)
        """
        # Apply defender actions first
        self._apply_defender_actions(state, defender_actions)
        
        # Apply attacker actions
        successful_exploits = self._apply_attacker_actions(state, attacker_actions)
        
        # Update kill chain stage
        self._update_kill_chain_stage(state, successful_exploits)
        
        # Update system metrics
        state.phi = state.compute_phi()
        state.system.increment_time()
        state.record_defense(defender_actions)
        state.record_attack(attacker_actions, len(successful_exploits))

        # Log current state with detailed asset information
        compromised_assets = [asset for asset in state.system.assets if asset.is_compromised]
        step_display = current_step if current_step is not None else state.system.time_step
        logger.info(f"Step {step_display}: {len(compromised_assets)} assets compromised: "
                    f"{[(asset.asset_id, asset.name) for asset in compromised_assets]}")

    def apply(self, state: State, defender_actions: List[Vulnerability], attacker_actions: List[Vulnerability]) -> None:
        self.apply_actions(state, defender_actions, attacker_actions)

    def _apply_defender_actions(self, state: State, defender_actions: List[Vulnerability]) -> None:
        """
        Apply defender patching actions with REALISTIC recovery logic.
        Recovery should only happen when the ACTUAL attack vectors are addressed.
        """
        # Track patched vulnerabilities by asset
        patched_vulns_by_asset = defaultdict(list)

        for vuln in defender_actions:
            # Apply patch
            vuln.apply_patch()

            # Track which assets had vulnerabilities patched
            for asset in state.system.assets:
                for comp in asset.components:
                    if vuln in comp.vulnerabilities:
                        patched_vulns_by_asset[asset.asset_id].append(vuln)
                        logger.info(f"Vulnerability {vuln.cve_id} patched in asset {asset.asset_id}")

        # Recovery logic
        for asset in state.system.assets:
            if not asset.is_compromised:
                continue

            asset_id = asset.asset_id
            recovery_chance = 0.0
            recovery_reasons = []

            # Find exploited vulnerabilities
            exploited_vulns = []
            for comp in asset.components:
                for vuln in comp.vulnerabilities:
                    if vuln.is_exploited and not vuln.is_patched:
                        exploited_vulns.append(vuln.cve_id)

            # Log all exploited vulnerabilities for debugging
            logger.debug(f"Asset {asset_id} exploited vulnerabilities: {[v for v in exploited_vulns]}")

            # Check lateral movement
            was_lateral_movement_target = False
            if hasattr(state, 'lateral_movement_targets'):
                was_lateral_movement_target = asset_id in state.lateral_movement_targets

            # RECOVERY RULE 1: All exploited vulnerabilities are patched
            if not exploited_vulns:
                # All exploited vulnerabilities are patched - force recovery
                asset.mark_as_compromised(False)
                asset._compromise_time = None
                recovery_reasons.append("all exploited vulnerabilities patched")
                logger.info(f"Asset {asset_id} automatically recovered: {', '.join(recovery_reasons)}")
                continue

            # RECOVERY RULE 2: Lateral movement targets
            elif was_lateral_movement_target:
                source_assets_clean = True
                time_since_compromise = 0
                if hasattr(asset, '_compromise_time') and asset._compromise_time is not None:
                    time_since_compromise = state.system.time_step - asset._compromise_time

                if time_since_compromise >= 5:
                    recovery_chance = 0.15
                    recovery_reasons.append(f"lateral movement cleanup after {time_since_compromise} steps")
                elif time_since_compromise >= 3:
                    recovery_chance = 0.05
                    recovery_reasons.append(f"partial lateral movement cleanup after {time_since_compromise} steps")
                else:
                    logger.info(
                        f"Asset {asset_id} remains compromised: lateral movement target, insufficient time for cleanup ({time_since_compromise} steps)")
                    continue

            # RECOVERY RULE 3: Unknown compromise vector - IMPROVED LOGIC WITH BELIEF STATE
            else:
                time_since_compromise = 0
                if hasattr(asset, '_compromise_time') and asset._compromise_time is not None:
                    time_since_compromise = state.system.time_step - asset._compromise_time

                # Check if there are any unpatched vulnerabilities that could maintain compromise
                unpatched_vulns = []
                for comp in asset.components:
                    for vuln in comp.vulnerabilities:
                        if not vuln.is_patched and not vuln.is_exploited:
                            unpatched_vulns.append(vuln)

                # If unpatched vulnerabilities exist, asset stays compromised
                if unpatched_vulns:
                    recovery_chance = 0.0
                    recovery_reasons.append(f"Asset has {len(unpatched_vulns)} unpatched vulnerabilities")
                    logger.debug(f"Asset {asset_id} stays compromised due to {len(unpatched_vulns)} unpatched vulnerabilities")
                else:
                    # No unpatched vulnerabilities - check belief state for recovery probability
                    recovery_chance = self._calculate_belief_based_recovery_chance(asset_id, time_since_compromise, state)
                    recovery_reasons.append("No unpatched vulnerabilities remain - belief-based recovery")
                    logger.info(f"Asset {asset_id} recovery chance: {recovery_chance:.3f} based on belief state")

            # Apply recovery chance
            if recovery_chance > 0 and random.random() < recovery_chance:
                asset.mark_as_compromised(False)
                asset._compromise_time = None
                logger.info(
                    f"Asset {asset_id} recovered (chance: {recovery_chance:.2f}): {', '.join(recovery_reasons)}")
            else:
                if recovery_chance > 0:
                    logger.info(
                        f"Asset {asset_id} recovery failed despite {recovery_chance:.2f} chance: {', '.join(recovery_reasons)}")

    def _apply_attacker_actions(self, state: State, attacker_actions: List[Vulnerability]) -> List[Vulnerability]:
        """Apply attacker actions with synchronized vulnerability state"""
        successful_exploits = []
        max_exploits_per_step = 3
        exploit_count = 0

        for vuln in attacker_actions:
            if exploit_count >= max_exploits_per_step:
                logger.info(f"Reached maximum of {max_exploits_per_step} exploits for this step, stopping")
                break

            if vuln.is_patched:
                logger.info(f"Skipping patched vulnerability {vuln.cve_id}")
                continue

            if vuln.is_exploited:
                logger.info(f"Vulnerability {vuln.cve_id} already exploited")
                continue

            success_prob = self._calculate_exploit_success_probability(vuln, state)
            if random.random() < success_prob:
                exploit_success = vuln.mark_as_exploited()  # Checks is_patched internally
                if exploit_success:
                    successful_exploits.append(vuln)
                    exploit_count += 1
                    # Find asset and update state
                    for asset in state.system.assets:
                        for comp in asset.components:
                            if vuln in comp.vulnerabilities:
                                vuln_key = create_vuln_key(vuln.cve_id, str(asset.asset_id), str(comp.id))
                                asset.mark_as_compromised(True)
                                asset.record_compromise(state.system.time_step)
                                # Update attacker's tracking
                                if hasattr(state, '_attacker_ref'):
                                    state._attacker_ref.compromised_nodes.add(str(asset.asset_id))
                                    state._attacker_ref.current_compromised_node = str(asset.asset_id)
                                logger.info(f"Asset {asset.asset_id} ({asset.name}) compromised via {vuln.cve_id}")
                                break
                        if exploit_success:
                            break

        return successful_exploits
    def _calculate_exploit_success_probability(self, vuln: Vulnerability, state: State) -> float:
        """
        Calculate exploit success probability using a threat intelligence approach
        with a maximum success rate of 0.5 (50%).
        """
        # Start with a much lower base rate
        base_rate = self.exploit_success_base_rate * 0.15  # Reduce from default

        # Get EPSS score if available, otherwise use a default
        epss = getattr(vuln, 'epss', 0.1)

        # CVSS contribution (normalized to 0-1 range, but scaled down)
        cvss_factor = getattr(vuln, 'cvss', 5.0) / 10.0
        cvss_contribution = 0.15 * cvss_factor  # Reduced from 0.25

        # EPSS contribution (scaled down)
        epss_contribution = 0.15 * epss  # Reduced from 0.25

        # Exploit availability contribution (scaled down)
        exploit_available = getattr(vuln, 'exploit', False)
        exploit_contribution = 0.1 if exploit_available else 0.0  # Reduced from 0.15

        # Ransomware factor (scaled down)
        is_ransomware = getattr(vuln, 'ransomWare', False)
        ransomware_contribution = 0.05 if is_ransomware else 0.0  # Reduced from 0.1

        # Consider kill chain stage - exploitation gets harder in later stages
        if hasattr(state, 'k'):
            current_stage = state.k
            current_stage_value = getattr(current_stage, 'value', current_stage)
            if isinstance(current_stage_value, str) and current_stage_value.isdigit():
                current_stage_value = int(current_stage_value)
            # Apply stronger penalty for advanced stages
            stage_penalty = 1.0 - (current_stage_value * 0.12)
            stage_penalty = max(0.25, stage_penalty)  # At least 25% of original probability
        else:
            stage_penalty = 1.0

        # Calculate complexity penalty with stronger penalties
        complexity_penalty = 1.0
        if hasattr(vuln, 'complexity'):
            if vuln.complexity == 'high':
                complexity_penalty = 0.3  # 60% reduction for high complexity
            elif vuln.complexity == 'medium':
                complexity_penalty = 0.6  # 40% reduction for medium complexity

        # Calculate final probability
        success_prob = base_rate + cvss_contribution + epss_contribution + exploit_contribution + ransomware_contribution

        # Apply penalties
        success_prob *= stage_penalty * complexity_penalty

        # Additional random factor to create more variability
        #randomness = 0.8 + (random.random() * 0.4)  # Random factor between 0.8 and 1.2
        #success_prob *= randomness

        # Hard cap at 0.5 (50%) as requested
        return min(max(success_prob, 0.05), 0.5)  # Between 5% and 50%

    # Fix 3: Kill Chain Progression - Enhanced progression logic

    def _update_kill_chain_stage(self, state, successful_exploits):
        """
        Enhanced kill chain progression with better logic and validation.
        """
        # Ensure k is an integer
        if hasattr(state.k, 'value'):
            current_stage_value = state.k.value
        else:
            current_stage_value = int(state.k) if isinstance(state.k, (int, str)) else 1

        max_stage_value = 7  # Actions on Objectives

        # Initialize time tracking
        if not hasattr(state, 'time_in_stage'):
            state.time_in_stage = 0
        else:
            state.time_in_stage += 1
            
        # Emergency cap on time_in_stage to prevent runaway counters
        if state.time_in_stage > 1000:
            logger.error(f"EMERGENCY: time_in_stage exceeded 1000 (current: {state.time_in_stage}), resetting")
            state.time_in_stage = 0

        # Initialize stage-specific tracking
        if not hasattr(state, 'stage_achievements'):
            state.stage_achievements = {
                1: 0,  # Reconnaissance actions
                2: 0,  # Weaponization/preparation
                3: 0,  # Delivery attempts
                4: 0,  # Exploitation successes
                5: 0,  # Installation/persistence
                6: 0,  # C2 establishment
                7: 0  # Actions on objectives
            }

        # Get current system state metrics
        compromised_assets = [a for a in state.system.assets if a.is_compromised]
        num_compromised = len(compromised_assets)
        high_value_compromised = sum(1 for a in compromised_assets
                                     if hasattr(a, 'business_value') and a.business_value >= 50000)

        # Track achievements for current action
        if successful_exploits:
            if current_stage_value <= 3:  # Early stages
                state.stage_achievements[4] += len(successful_exploits)  # Exploitation
            elif current_stage_value == 4:
                state.stage_achievements[5] += len(successful_exploits)  # Installation
            elif current_stage_value == 5:
                state.stage_achievements[6] += 1  # C2
            elif current_stage_value >= 6:
                state.stage_achievements[7] += 1  # Actions on objectives

        # Stage-specific progression logic
        should_advance = False
        advancement_reason = ""

        if current_stage_value == 1:  # Reconnaissance
            # Advance after sufficient reconnaissance or first successful exploit
            recon_threshold = 2
            if (state.time_in_stage >= recon_threshold and
                    (successful_exploits or state.stage_achievements[1] >= 3)):
                should_advance = True
                advancement_reason = "reconnaissance complete"

        elif current_stage_value == 2:  # Weaponization
            # Advance when tools are prepared or exploitation begins
            if state.time_in_stage >= 2 and (successful_exploits or num_compromised > 0):
                should_advance = True
                advancement_reason = "weaponization complete"

        elif current_stage_value == 3:  # Delivery
            # Advance when initial access is achieved
            if num_compromised > 0 or state.stage_achievements[4] > 0:
                should_advance = True
                advancement_reason = "initial access achieved"

        elif current_stage_value == 4:  # Exploitation
            # Advance when multiple assets compromised or persistence established
            exploitation_threshold = 2
            if (num_compromised >= 2 or
                    state.stage_achievements[4] >= exploitation_threshold or
                    state.time_in_stage >= 4):
                should_advance = True
                advancement_reason = f"exploitation objectives met ({num_compromised} assets compromised)"

        elif current_stage_value == 5:  # Installation
            # Advance when persistence is established on multiple systems
            installation_threshold = 1
            if (num_compromised >= 2 or
                    state.stage_achievements[5] >= installation_threshold or
                    state.time_in_stage >= 3):
                should_advance = True
                advancement_reason = "installation complete"

        elif current_stage_value == 6:  # Command and Control
            # Advance when ready for final objectives
            c2_threshold = 2
            if (num_compromised >= 3 or
                    high_value_compromised >= 1 or
                    state.time_in_stage >= 3):
                should_advance = True
                advancement_reason = "C2 established, ready for objectives"

        # Apply advancement with proper state management
        if should_advance and current_stage_value < max_stage_value:
            new_stage_value = current_stage_value + 1
            state.time_in_stage = 0  # Reset timer for new stage

            # Update state.k properly
            try:
                # Import the KillChainStage enum from the correct module
                from classes.mitre import KillChainStage
                state.k = new_stage_value  # Store as integer as per state.py design
            except ImportError:
                state.k = new_stage_value

            logger.info(
                f"Kill chain progression: Stage {current_stage_value} -> {new_stage_value} ({advancement_reason})")
            logger.info(f"Progression metrics: {num_compromised} assets compromised, "
                        f"{high_value_compromised} high-value targets, "
                        f"stage achievements: {dict(state.stage_achievements)}")
        else:
            logger.info(f"Time in kill chain stage {current_stage_value}: {state.time_in_stage}")
            if current_stage_value < max_stage_value:
                logger.debug(f"Advancement conditions not met: compromised={num_compromised}, "
                             f"achievements={state.stage_achievements[current_stage_value + 1] if current_stage_value + 1 <= max_stage_value else 'N/A'}")

    # Enhanced stage validation for State class
    def validate_and_update_kill_chain_stage(self):
        """
        Enhanced validation and auto-correction of kill chain stage based on system state.
        Add this method to the State class in state.py
        """
        if not hasattr(self, 'k'):
            self.k = 1  # Default to Reconnaissance
            return

        # Ensure k is an integer
        if hasattr(self.k, 'value'):
            current_stage = self.k.value
        else:
            current_stage = int(self.k) if isinstance(self.k, (int, str)) else 1

        # Get system metrics
        compromised_assets = [a for a in self.system.assets if a.is_compromised]
        num_compromised = len(compromised_assets)

        exploited_vulns = []
        for asset in self.system.assets:
            for comp in asset.components:
                for vuln in comp.vulnerabilities:
                    if vuln.is_exploited:
                        exploited_vulns.append(vuln)

        # Determine minimum required stage based on system state
        min_required_stage = 1  # Reconnaissance

        if exploited_vulns:
            min_required_stage = max(min_required_stage, 4)  # Exploitation

        if num_compromised > 0:
            min_required_stage = max(min_required_stage, 5)  # Installation

        if num_compromised >= 2:
            min_required_stage = max(min_required_stage, 6)  # Command and Control

        if num_compromised >= 3:
            min_required_stage = max(min_required_stage, 7)  # Actions on Objectives

        # Auto-advance if current stage is too low
        if current_stage < min_required_stage:
            self.k = min_required_stage
            logger.info(f"Auto-advanced kill chain stage from {current_stage} to {min_required_stage} "
                        f"based on system state ({num_compromised} compromised, {len(exploited_vulns)} exploited)")

    # Add this method to APT3RTUSimulation class to track progression better
    def log_kill_chain_progression(self, step: int):
        """
        Enhanced logging for kill chain progression tracking.
        """
        try:
            from classes.mitre import KillChainStage
            current_stage = KillChainStage(self.state.k)
            stage_name = current_stage.name
        except (ImportError, ValueError):
            stage_name = f"Stage_{self.state.k}"

        compromised_assets = [a for a in self.state.system.assets if a.is_compromised]
        compromised_count = len(compromised_assets)

        exploited_vulns = sum(1 for asset in self.state.system.assets
                              for comp in asset.components
                              for vuln in comp.vulnerabilities
                              if vuln.is_exploited)

        logger.info(f"Step {step + 1} Kill Chain Status:")
        logger.info(f"  Current Stage: {self.state.k} ({stage_name})")
        logger.info(f"  Compromised Assets: {compromised_count}")
        logger.info(f"  Exploited Vulnerabilities: {exploited_vulns}")
        logger.info(f"  Time in Stage: {getattr(self.state, 'time_in_stage', 0)}")

        if compromised_assets:
            asset_names = [f"{a.asset_id}({a.name})" for a in compromised_assets]
            logger.info(f"  Compromised: {', '.join(asset_names)}")

    # Enhanced run_step method modification for better progression tracking
    def run_step_with_enhanced_progression(self, strategy, step, verbose=False):
        """
        Enhanced run_step with better kill chain progression tracking.
        Replace or modify the existing run_step method in APT3RTUSimulation.
        """
        self.current_step = step

        # Validate and potentially correct kill chain stage
        self.state.validate_and_update_kill_chain_stage()

        if verbose:
            print(f"\n==== Step {step + 1} ====")
            self.log_kill_chain_progression(step)
            print(f"Remaining defender budget: ${self._remaining_defender_budget:.2f}")
            print(f"Remaining attacker budget: ${self._remaining_attacker_budget:.2f}")

        # Store pre-action state for comparison
        pre_action_compromised = {str(a.asset_id): a.is_compromised for a in self.state.system.assets}

        # Continue with existing run_step logic...
        self.threat_processor.update_threat_levels(self.state)
        attacker_action = self.get_next_attack_action(self.state)

        # ... (rest of existing run_step logic)

        # After applying actions, check for progression
        post_action_compromised = {str(a.asset_id): a.is_compromised for a in self.state.system.assets}

        # Detect new compromises
        new_compromises = [asset_id for asset_id, is_comp in post_action_compromised.items()
                           if is_comp and not pre_action_compromised[asset_id]]

        if new_compromises:
            logger.info(f"Step {step + 1}: New asset compromises detected: {new_compromises}")

        # Log final progression state
        if verbose:
            self.log_kill_chain_progression(step)

    def _calculate_belief_based_recovery_chance(self, asset_id: str, time_since_compromise: int, state: State) -> float:
        """
        Calculate belief-based recovery chance based on time since compromise and system state.
        Enhanced to provide more realistic recovery probabilities.
        """
        # Base recovery probability increases with time
        if time_since_compromise < 3:
            return 0.0  # No recovery in first 3 steps
        elif time_since_compromise < 5:
            return 0.25  # 25% chance after 3-4 steps
        elif time_since_compromise < 8:
            return 0.40  # 40% chance after 5-7 steps
        elif time_since_compromise < 12:
            return 0.60  # 60% chance after 8-11 steps
        else:
            return 0.80  # 80% chance after 12+ steps
        
        # Note: This replaces the previous low 0.15 probability that was causing
        # assets to remain compromised indefinitely