# defender_pomdp.py
import math
import random
from enum import Enum
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Any

from .state import State, Asset, Vulnerability, KillChainStage, System


class AssetCategory(Enum):
    CRITICAL = "critical"
    EXTERNAL = "external"
    INTERNAL = "internal"


class DefenderBeliefState:
    """
    Represents the defender's belief about the attacker's state,
    particularly the current kill chain stage.
    """

    def __init__(self):
        # Initialize kill chain stage belief (probability distribution)
        self.kill_chain_beliefs = {
            KillChainStage.RECONNAISSANCE: 0.9,
            KillChainStage.WEAPONIZATION: 0.1,
            KillChainStage.DELIVERY: 0.0,
            KillChainStage.EXPLOITATION: 0.0,
            KillChainStage.INSTALLATION: 0.0,
            KillChainStage.COMMAND_AND_CONTROL: 0.0,
            KillChainStage.ACTIONS_ON_OBJECTIVES: 0.0
        }
        # Track observations
        self.observation_history = []
        # Confidence in detection capability
        self.detection_confidence = 0.7
        # Initialize the stage transition matrix
        self._init_transition_matrix()

    def _init_transition_matrix(self):
        """Initialize the stage transition probability matrix."""
        # Create matrix where T[i,j] = P(next_stage=j | current_stage=i)
        # Higher probabilities for forward progression, small chance of skipping stages
        self.transition_matrix = {
            KillChainStage.RECONNAISSANCE: {
                KillChainStage.RECONNAISSANCE: 0.6,
                KillChainStage.WEAPONIZATION: 0.3,
                KillChainStage.DELIVERY: 0.1,
                KillChainStage.EXPLOITATION: 0.0,
                KillChainStage.INSTALLATION: 0.0,
                KillChainStage.COMMAND_AND_CONTROL: 0.0,
                KillChainStage.ACTIONS_ON_OBJECTIVES: 0.0
            },
            KillChainStage.WEAPONIZATION: {
                KillChainStage.RECONNAISSANCE: 0.1,
                KillChainStage.WEAPONIZATION: 0.5,
                KillChainStage.DELIVERY: 0.4,
                KillChainStage.EXPLOITATION: 0.0,
                KillChainStage.INSTALLATION: 0.0,
                KillChainStage.COMMAND_AND_CONTROL: 0.0,
                KillChainStage.ACTIONS_ON_OBJECTIVES: 0.0
            },
            KillChainStage.DELIVERY: {
                KillChainStage.RECONNAISSANCE: 0.05,
                KillChainStage.WEAPONIZATION: 0.1,
                KillChainStage.DELIVERY: 0.5,
                KillChainStage.EXPLOITATION: 0.3,
                KillChainStage.INSTALLATION: 0.05,
                KillChainStage.COMMAND_AND_CONTROL: 0.0,
                KillChainStage.ACTIONS_ON_OBJECTIVES: 0.0
            },
            KillChainStage.EXPLOITATION: {
                KillChainStage.RECONNAISSANCE: 0.0,
                KillChainStage.WEAPONIZATION: 0.05,
                KillChainStage.DELIVERY: 0.1,
                KillChainStage.EXPLOITATION: 0.5,
                KillChainStage.INSTALLATION: 0.3,
                KillChainStage.COMMAND_AND_CONTROL: 0.05,
                KillChainStage.ACTIONS_ON_OBJECTIVES: 0.0
            },
            KillChainStage.INSTALLATION: {
                KillChainStage.RECONNAISSANCE: 0.0,
                KillChainStage.WEAPONIZATION: 0.0,
                KillChainStage.DELIVERY: 0.05,
                KillChainStage.EXPLOITATION: 0.1,
                KillChainStage.INSTALLATION: 0.5,
                KillChainStage.COMMAND_AND_CONTROL: 0.3,
                KillChainStage.ACTIONS_ON_OBJECTIVES: 0.05
            },
            KillChainStage.COMMAND_AND_CONTROL: {
                KillChainStage.RECONNAISSANCE: 0.0,
                KillChainStage.WEAPONIZATION: 0.0,
                KillChainStage.DELIVERY: 0.0,
                KillChainStage.EXPLOITATION: 0.05,
                KillChainStage.INSTALLATION: 0.1,
                KillChainStage.COMMAND_AND_CONTROL: 0.5,
                KillChainStage.ACTIONS_ON_OBJECTIVES: 0.35
            },
            KillChainStage.ACTIONS_ON_OBJECTIVES: {
                KillChainStage.RECONNAISSANCE: 0.05,  # New reconnaissance for next target
                KillChainStage.WEAPONIZATION: 0.0,
                KillChainStage.DELIVERY: 0.0,
                KillChainStage.EXPLOITATION: 0.0,
                KillChainStage.INSTALLATION: 0.0,
                KillChainStage.COMMAND_AND_CONTROL: 0.15,
                KillChainStage.ACTIONS_ON_OBJECTIVES: 0.8
            }
        }

    def get_most_likely_stage(self) -> KillChainStage:
        """Return the most likely kill chain stage based on current beliefs."""
        return max(self.kill_chain_beliefs.items(), key=lambda x: x[1])[0]

    def get_estimated_stage_value(self) -> float:
        """Calculate the expected value of the kill chain stage."""
        return sum(stage.value * prob for stage, prob in self.kill_chain_beliefs.items())

    def update_belief_from_observations(self, state: State) -> None:
        """Update belief about attacker's kill chain stage using both observations and transition model."""
        # Step 1: Apply transition model (prediction step)
        predicted_belief = self._apply_transition_model()

        # Step 2: Apply observation model (correction step)
        # Gather observable indicators
        indicators = self._gather_observable_indicators(state)

        # Calculate observation likelihoods P(indicators | stage)
        observation_likelihoods = self._calculate_observation_likelihoods(indicators)

        # Apply Bayes' rule
        updated_belief = {}
        for stage in KillChainStage:
            # P(stage | indicators) ∝ P(indicators | stage) * P_predicted(stage)
            updated_belief[stage] = observation_likelihoods[stage] * predicted_belief[stage]

        # Normalize
        total = sum(updated_belief.values())
        if total > 0:
            for stage in updated_belief:
                updated_belief[stage] /= total

        self.kill_chain_beliefs = updated_belief

        # Add observation to history
        self.observation_history.append({
            'time_step': state.system.time_step,
            'indicators': indicators,
            'updated_belief': self.kill_chain_beliefs.copy()
        })

    def _apply_transition_model(self) -> Dict[KillChainStage, float]:
        """Apply the stage transition model to predict next belief state."""
        predicted_belief = {stage: 0.0 for stage in KillChainStage}

        # Calculate P(next_stage) = Σ P(next_stage | current_stage) * P(current_stage)
        for current_stage, current_prob in self.kill_chain_beliefs.items():
            for next_stage, transition_prob in self.transition_matrix[current_stage].items():
                predicted_belief[next_stage] += transition_prob * current_prob

        return predicted_belief

    def _calculate_observation_likelihoods(self, indicators: Dict[str, Any]) -> Dict[KillChainStage, float]:
        """Calculate P(indicators | stage) for each stage using a detailed observation model."""
        # Define the observation function mapping indicators to likelihoods
        observation_model = {
            'compromised_assets': {
                0: {  # No compromised assets
                    KillChainStage.RECONNAISSANCE: 0.9,
                    KillChainStage.WEAPONIZATION: 0.8,
                    KillChainStage.DELIVERY: 0.7,
                    KillChainStage.EXPLOITATION: 0.3,
                    KillChainStage.INSTALLATION: 0.2,
                    KillChainStage.COMMAND_AND_CONTROL: 0.1,
                    KillChainStage.ACTIONS_ON_OBJECTIVES: 0.1
                },
                1: {  # One compromised asset
                    KillChainStage.RECONNAISSANCE: 0.1,
                    KillChainStage.WEAPONIZATION: 0.2,
                    KillChainStage.DELIVERY: 0.3,
                    KillChainStage.EXPLOITATION: 0.7,
                    KillChainStage.INSTALLATION: 0.6,
                    KillChainStage.COMMAND_AND_CONTROL: 0.4,
                    KillChainStage.ACTIONS_ON_OBJECTIVES: 0.3
                },
                2: {  # Multiple compromised assets
                    KillChainStage.RECONNAISSANCE: 0.05,
                    KillChainStage.WEAPONIZATION: 0.1,
                    KillChainStage.DELIVERY: 0.1,
                    KillChainStage.EXPLOITATION: 0.4,
                    KillChainStage.INSTALLATION: 0.5,
                    KillChainStage.COMMAND_AND_CONTROL: 0.7,
                    KillChainStage.ACTIONS_ON_OBJECTIVES: 0.8
                }
            },
            'lateral_movement_indicators': {
                True: {  # Indicators present
                    KillChainStage.RECONNAISSANCE: 0.05,
                    KillChainStage.WEAPONIZATION: 0.05,
                    KillChainStage.DELIVERY: 0.1,
                    KillChainStage.EXPLOITATION: 0.3,
                    KillChainStage.INSTALLATION: 0.5,
                    KillChainStage.COMMAND_AND_CONTROL: 0.8,
                    KillChainStage.ACTIONS_ON_OBJECTIVES: 0.7
                },
                False: {  # No indicators
                    KillChainStage.RECONNAISSANCE: 0.95,
                    KillChainStage.WEAPONIZATION: 0.95,
                    KillChainStage.DELIVERY: 0.9,
                    KillChainStage.EXPLOITATION: 0.7,
                    KillChainStage.INSTALLATION: 0.5,
                    KillChainStage.COMMAND_AND_CONTROL: 0.2,
                    KillChainStage.ACTIONS_ON_OBJECTIVES: 0.3
                }
            },
            'data_exfiltration_indicators': {
                True: {  # Indicators present
                    KillChainStage.RECONNAISSANCE: 0.01,
                    KillChainStage.WEAPONIZATION: 0.01,
                    KillChainStage.DELIVERY: 0.05,
                    KillChainStage.EXPLOITATION: 0.1,
                    KillChainStage.INSTALLATION: 0.2,
                    KillChainStage.COMMAND_AND_CONTROL: 0.3,
                    KillChainStage.ACTIONS_ON_OBJECTIVES: 0.9
                },
                False: {  # No indicators
                    KillChainStage.RECONNAISSANCE: 0.99,
                    KillChainStage.WEAPONIZATION: 0.99,
                    KillChainStage.DELIVERY: 0.95,
                    KillChainStage.EXPLOITATION: 0.9,
                    KillChainStage.INSTALLATION: 0.8,
                    KillChainStage.COMMAND_AND_CONTROL: 0.7,
                    KillChainStage.ACTIONS_ON_OBJECTIVES: 0.1
                }
            },
            'recent_attack_success_rate': {
                'low': {  # Low success rate (0-0.3)
                    KillChainStage.RECONNAISSANCE: 0.7,
                    KillChainStage.WEAPONIZATION: 0.6,
                    KillChainStage.DELIVERY: 0.5,
                    KillChainStage.EXPLOITATION: 0.4,
                    KillChainStage.INSTALLATION: 0.3,
                    KillChainStage.COMMAND_AND_CONTROL: 0.2,
                    KillChainStage.ACTIONS_ON_OBJECTIVES: 0.1
                },
                'medium': {  # Medium success rate (0.3-0.7)
                    KillChainStage.RECONNAISSANCE: 0.2,
                    KillChainStage.WEAPONIZATION: 0.3,
                    KillChainStage.DELIVERY: 0.4,
                    KillChainStage.EXPLOITATION: 0.5,
                    KillChainStage.INSTALLATION: 0.4,
                    KillChainStage.COMMAND_AND_CONTROL: 0.3,
                    KillChainStage.ACTIONS_ON_OBJECTIVES: 0.3
                },
                'high': {  # High success rate (0.7-1.0)
                    KillChainStage.RECONNAISSANCE: 0.1,
                    KillChainStage.WEAPONIZATION: 0.1,
                    KillChainStage.DELIVERY: 0.2,
                    KillChainStage.EXPLOITATION: 0.3,
                    KillChainStage.INSTALLATION: 0.5,
                    KillChainStage.COMMAND_AND_CONTROL: 0.6,
                    KillChainStage.ACTIONS_ON_OBJECTIVES: 0.7
                }
            }
        }

        # Process indicators to match observation model categories
        processed_indicators = {
            'compromised_assets': min(2, indicators['compromised_assets']),
            'lateral_movement_indicators': indicators['lateral_movement_indicators'] > 0.5,
            'data_exfiltration_indicators': indicators['data_exfiltration_indicators'] > 0.5,
            'recent_attack_success_rate': self._categorize_success_rate(indicators['attack_success_rate'])
        }

        # Calculate likelihood for each stage by multiplying individual indicator likelihoods
        likelihoods = {stage: 1.0 for stage in KillChainStage}

        for indicator_name, indicator_value in processed_indicators.items():
            if indicator_name in observation_model:
                for stage in KillChainStage:
                    likelihoods[stage] *= observation_model[indicator_name][indicator_value][stage]

        return likelihoods

    def _gather_observable_indicators(self, state: State) -> Dict[str, Any]:
        """Extract observable indicators from the current state."""
        # Basic indicators
        compromised_assets = [a for a in state.system.assets if a.is_compromised]

        # Calculate attack success rate
        attack_success_rate = self._calculate_attack_success_rate(state.attack_history)

        # Calculate network-based indicators with uncertainty
        lateral_movement_score = self._detect_lateral_movement(state, compromised_assets)
        data_exfiltration_score = self._detect_exfiltration(state, compromised_assets)

        # Add observation noise based on detection confidence
        # Lower confidence = more observation noise
        noise_factor = 1.0 - self.detection_confidence
        lateral_movement_score = self._add_observation_noise(lateral_movement_score, noise_factor)
        data_exfiltration_score = self._add_observation_noise(data_exfiltration_score, noise_factor)

        return {
            'compromised_assets': len(compromised_assets),
            'attack_success_rate': attack_success_rate,
            'lateral_movement_indicators': lateral_movement_score,
            'data_exfiltration_indicators': data_exfiltration_score,
            'exploited_vuln_count': sum(1 for a in state.system.assets
                                        for c in a.components
                                        for v in c.vulnerabilities if v.is_exploited),
            'critically_compromised': sum(1 for a in compromised_assets
                                          if a.criticality_level >= 4)
        }

    def _calculate_attack_success_rate(self, attack_history: List[Dict], window: int = 5) -> float:
        """Calculate success rate of recent attack attempts."""
        if not attack_history or len(attack_history) == 0:
            return 0.0

        recent = attack_history[-window:] if len(attack_history) > window else attack_history
        successes = sum(record['success_count'] for record in recent)
        attempts = len(recent)

        return successes / attempts if attempts > 0 else 0.0

    def _detect_lateral_movement(self, state: State, compromised_assets: List[Asset]) -> float:
        """Detect indicators of lateral movement with a probabilistic model."""
        # Base score starts at 0
        score = 0.0

        # Multiple compromised assets is a prerequisite for lateral movement
        if len(compromised_assets) < 2:
            return score

        # Check for connected compromised assets (stronger indicator)
        for i, asset1 in enumerate(compromised_assets[:-1]):
            for asset2 in compromised_assets[i + 1:]:
                # Check if direct connection exists
                direct_connection = any(
                    c.from_asset == asset1 and c.to_asset == asset2 for c in state.system.connections) or \
                                    any(c.from_asset == asset2 and c.to_asset == asset1 for c in
                                        state.system.connections)

                if direct_connection:
                    score += 0.4  # Strong indicator

                # Check temporal pattern (compromise within short time window)
                if hasattr(asset1, '_compromise_time') and hasattr(asset2, '_compromise_time'):
                    time_gap = abs((asset1._compromise_time or 0) - (asset2._compromise_time or 0))
                    if time_gap <= 3:  # Short time window
                        score += 0.3  # Moderate indicator

        # Check for privilege escalation indicators (often accompanies lateral movement)
        if any(a._compromise_time is not None and a.is_compromised for a in state.system.assets
               if a.criticality_level >= 4):
            score += 0.2  # Additional indicator

        # Normalize score to [0,1] range
        return min(1.0, score)

    def _detect_exfiltration(self, state: State, compromised_assets: List[Asset]) -> float:
        """Detect indicators of data exfiltration with a probabilistic model."""
        # Base score starts at 0
        score = 0.0

        # Check for compromise of sensitive data assets
        for asset in compromised_assets:
            if getattr(asset, 'contains_sensitive_data', False):
                score += 0.6  # Strong indicator

            # Critical assets are also valuable targets
            if asset.criticality_level >= 4:
                score += 0.3  # Moderate indicator

        # Check for patterns in attack history
        late_stage_actions = 0
        if state.attack_history:
            # Look at recent attack history (last 5 actions)
            recent_history = state.attack_history[-5:]
            for record in recent_history:
                # Check for actions consistent with exfiltration
                # (This would be more detailed in a real implementation)
                if 'exfiltration' in str(record.get('actions', [])).lower():
                    score += 0.4
                    late_stage_actions += 1

        # Normalize score to [0,1] range
        return min(1.0, score)

    def _add_observation_noise(self, value: float, noise_factor: float) -> float:
        """Add realistic observation noise based on detection confidence."""
        # Add Gaussian noise with standard deviation based on noise factor
        noise = random.gauss(0, noise_factor * 0.2)
        # Ensure value stays in valid range [0,1]
        return max(0.0, min(1.0, value + noise))

    def _categorize_success_rate(self, rate: float) -> str:
        """Categorize attack success rate into low/medium/high."""
        if rate < 0.3:
            return 'low'
        elif rate < 0.7:
            return 'medium'
        else:
            return 'high'

    def initialize_hmm(self) -> None:
        """Initialize a Hidden Markov Model for kill chain stage inference."""
        # Define states (kill chain stages)
        self.states = list(KillChainStage)

        # Define observations (simplified for implementation)
        self.observations = ['no_compromise', 'single_compromise', 'multiple_compromise',
                             'lateral_movement', 'data_exfiltration']

        # Initial state probabilities (start at reconnaissance)
        self.initial_probabilities = {
            KillChainStage.RECONNAISSANCE: 0.8,
            KillChainStage.WEAPONIZATION: 0.2,
            KillChainStage.DELIVERY: 0.0,
            KillChainStage.EXPLOITATION: 0.0,
            KillChainStage.INSTALLATION: 0.0,
            KillChainStage.COMMAND_AND_CONTROL: 0.0,
            KillChainStage.ACTIONS_ON_OBJECTIVES: 0.0
        }

        # Emission matrix B[i,k] = P(observation=k | state=i)
        self.emission_matrix = {
            KillChainStage.RECONNAISSANCE: {
                'no_compromise': 0.8,
                'single_compromise': 0.1,
                'multiple_compromise': 0.05,
                'lateral_movement': 0.03,
                'data_exfiltration': 0.02
            },
            KillChainStage.WEAPONIZATION: {
                'no_compromise': 0.7,
                'single_compromise': 0.2,
                'multiple_compromise': 0.05,
                'lateral_movement': 0.03,
                'data_exfiltration': 0.02
            },
            KillChainStage.DELIVERY: {
                'no_compromise': 0.5,
                'single_compromise': 0.3,
                'multiple_compromise': 0.1,
                'lateral_movement': 0.05,
                'data_exfiltration': 0.05
            },
            KillChainStage.EXPLOITATION: {
                'no_compromise': 0.3,
                'single_compromise': 0.5,
                'multiple_compromise': 0.1,
                'lateral_movement': 0.05,
                'data_exfiltration': 0.05
            },
            KillChainStage.INSTALLATION: {
                'no_compromise': 0.1,
                'single_compromise': 0.3,
                'multiple_compromise': 0.3,
                'lateral_movement': 0.2,
                'data_exfiltration': 0.1
            },
            KillChainStage.COMMAND_AND_CONTROL: {
                'no_compromise': 0.05,
                'single_compromise': 0.15,
                'multiple_compromise': 0.3,
                'lateral_movement': 0.4,
                'data_exfiltration': 0.1
            },
            KillChainStage.ACTIONS_ON_OBJECTIVES: {
                'no_compromise': 0.05,
                'single_compromise': 0.1,
                'multiple_compromise': 0.15,
                'lateral_movement': 0.2,
                'data_exfiltration': 0.5
            }
        }

    def update_belief_with_hmm(self, state: State) -> None:
        """Update belief using HMM and the Forward algorithm."""
        # Determine current observation
        observation = self._determine_observation_category(state)

        # If this is the first observation, use initial probabilities
        if not self.observation_history:
            forward = {}
            for s in self.states:
                forward[s] = self.initial_probabilities.get(s, 0) * self.emission_matrix[s][observation]
        else:
            # Get previous forward probabilities
            prev_forward = {s: self.kill_chain_beliefs[s] for s in self.states}

            # Calculate new forward probabilities
            forward = {}
            for s_j in self.states:
                forward[s_j] = 0
                for s_i in self.states:
                    forward[s_j] += prev_forward[s_i] * self.transition_matrix[s_i][s_j]
                forward[s_j] *= self.emission_matrix[s_j][observation]

        # Normalize
        total = sum(forward.values())
        if total > 0:
            forward = {s: p / total for s, p in forward.items()}

        # Update beliefs
        self.kill_chain_beliefs = forward

        # Add to observation history
        self.observation_history.append({
            'observation': observation,
            'beliefs': self.kill_chain_beliefs.copy()
        })

    def _determine_observation_category(self, state: State) -> str:
        """Determine which observation category best describes the current state."""
        indicators = self._gather_observable_indicators(state)

        if indicators['data_exfiltration_indicators'] > 0.5:
            return 'data_exfiltration'
        elif indicators['lateral_movement_indicators'] > 0.5:
            return 'lateral_movement'
        elif indicators['compromised_assets'] > 1:
            return 'multiple_compromise'
        elif indicators['compromised_assets'] == 1:
            return 'single_compromise'
        else:
            return 'no_compromise'


class DefenderPOMDPPolicy:
    """
    Defender policy that operates under partial observability.
    Maintains and updates a belief state about the attacker's kill chain stage.
    """

    def __init__(self, budget: int, threat_weight: float = 1.0, cost_aware: bool = True,
                 recent_attack_weight: float = 2.0, use_hmm: bool = False):
        self.budget = budget  # Max patches per step
        self.temp_budget = budget  # Temporary budget for dynamic adjustment
        self.threat_weight = threat_weight  # Weight for external threat level in prioritization
        self.cost_aware = cost_aware  # Whether to consider operational costs in decision-making
        self.recent_attack_weight = recent_attack_weight  # Weight for recent attack indicators
        self.attack_history = {}  # Track attack attempts by asset {asset_id: last_attack_step}
        self.current_step = 0  # Current simulation step
        self.use_hmm = use_hmm  # Whether to use HMM for belief updates
        self.belief_state = DefenderBeliefState()

        # Initialize HMM if requested
        if use_hmm:
            self.belief_state.initialize_hmm()

    def update_step(self) -> None:
        """Increment the simulation step counter."""
        self.current_step += 1

    def track_attack_attempts(self, state: State, attacker_actions: List[Vulnerability]) -> None:
        """Track attack attempts on assets to inform future prioritization."""
        for vuln in attacker_actions:
            # Find the asset this vulnerability belongs to
            for asset in state.system.assets:
                for comp in asset.components:
                    if vuln in comp.vulnerabilities:
                        # Record this attack attempt with current step
                        self.attack_history[asset.id] = self.current_step
                        break

    def high_level_policy(self, state: State) -> List[Asset]:
        """Prioritize assets using belief about attacker's stage."""
        # Update belief state
        if self.use_hmm:
            self.belief_state.update_belief_with_hmm(state)
        else:
            self.belief_state.update_belief_from_observations(state)

        # Get expected kill chain stage and uncertainty
        expected_stage_value = self.belief_state.get_estimated_stage_value()
        stage_entropy = self._calculate_belief_entropy()

        priority_assets = []
        for asset in state.system.assets:
            # Get the highest threat vulnerability on this asset
            vuln_threat = max([v.epss for comp in asset.components
                               for v in comp.vulnerabilities if not v.is_patched],
                              default=0.0)

            # Base score incorporates asset properties and vulnerability threats
            score = asset.criticality_level + self.threat_weight * vuln_threat

            # Incorporate business value
            if hasattr(asset, 'business_value'):
                score *= (1.0 + asset.business_value / 10)

            # Factor in attack recency
            attack_recency = 0
            if asset.id in self.attack_history:
                steps_since_attack = self.current_step - self.attack_history[asset.id]
                if steps_since_attack <= 2:
                    attack_recency = self.recent_attack_weight
                elif steps_since_attack <= 5:
                    attack_recency = self.recent_attack_weight * 0.5

            score += attack_recency

            # Respond to compromise status
            if asset.is_compromised:
                score *= 1.5

            # Consider estimated kill chain stage in prioritization
            # Higher stages = more advanced attack = focus on preventing escalation
            stage_multiplier = self._get_stage_multiplier(expected_stage_value, asset)
            score *= stage_multiplier

            # Add uncertainty factor - higher uncertainty = more conservative approach
            if stage_entropy > 1.0:  # High uncertainty
                # With high uncertainty, prioritize critical assets more
                if asset.criticality_level >= 4 or getattr(asset, 'contains_sensitive_data', False):
                    score *= (1.0 + stage_entropy * 0.2)  # Up to 20% boost based on uncertainty

            # If cost aware, consider asset centrality in prioritization
            if self.cost_aware and hasattr(asset, 'centrality'):
                if asset.type == AssetCategory.CRITICAL.value or vuln_threat > 0.8:
                    # Still prioritize critical assets, but slightly less so if high centrality
                    centrality_factor = 1.0 - (asset.centrality * 0.2)  # 0.8-1.0 range
                    score *= centrality_factor * 1.5
                else:
                    # For regular assets, consider centrality more strongly
                    centrality_factor = 1.0 - (asset.centrality * 0.4)  # 0.6-1.0 range
                    score *= centrality_factor

            priority_assets.append((asset, score))

        # Sort by score in descending order and return just the assets
        return [asset for asset, _ in sorted(priority_assets, key=lambda x: x[1], reverse=True)]

    def _get_stage_multiplier(self, stage_value: float, asset: Asset) -> float:
        """Get stage-specific multiplier for asset prioritization."""
        if stage_value >= 7:  # Actions on Objectives
            if getattr(asset, 'contains_sensitive_data', False):
                return 3.0  # Extreme priority
            elif asset.criticality_level >= 4:
                return 2.5
            else:
                return 2.0
        elif stage_value >= 6:  # Command & Control
            if hasattr(asset, 'centrality') and asset.centrality > 0.6:
                return 2.3  # High priority for central assets
            elif asset.criticality_level >= 4:
                return 2.0
            else:
                return 1.5
        elif stage_value >= 5:  # Installation
            if asset.criticality_level >= 4:
                return 2.0
            else:
                return 1.5
        elif stage_value >= 4:  # Exploitation
            if asset.criticality_level >= 4:
                return 1.8
            else:
                return 1.3
        elif stage_value >= 3:  # Delivery
            if asset.criticality_level >= 4:
                return 1.5
            else:
                return 1.2
        else:  # Reconnaissance or Weaponization
            if asset.criticality_level >= 4:
                return 1.3
            elif asset.type == "EXTERNAL":
                return 1.2
            else:
                return 1.0

    def _calculate_belief_entropy(self) -> float:
        """Calculate entropy of the kill chain belief distribution as uncertainty measure."""
        entropy = 0
        for stage, prob in self.belief_state.kill_chain_beliefs.items():
            if prob > 0:
                entropy -= prob * math.log2(prob)
        return entropy

    def low_level_policy(self, state: State, priority_assets: List[Asset]) -> List[Vulnerability]:
        """Select specific vulnerabilities to patch from prioritized assets."""
        from .cost import CostCalculator  # Import here to avoid circular imports

        unpatched_vulns = []
        for asset in priority_assets:
            for comp in asset.components:
                for vuln in comp.vulnerabilities:
                    if not vuln.is_patched:
                        # Track if this vulnerability is on a recently attacked asset
                        recent_attack = asset.id in self.attack_history and \
                                        (self.current_step - self.attack_history[asset.id]) <= 3

                        # Add the vulnerability with metadata
                        unpatched_vulns.append((vuln, asset, recent_attack))

        if self.cost_aware:
            # Cost-aware approach: Balance risk and operational cost
            scored_vulns = []
            for vuln, asset, recent_attack in unpatched_vulns:
                # Calculate risk-to-cost ratio
                ratio = CostCalculator.calculate_risk_to_cost_ratio(vuln, state)

                # Boost ratio for vulnerabilities on recently attacked assets
                if recent_attack:
                    ratio *= 1.5

                # Consider business value in vulnerability prioritization
                if hasattr(asset, 'business_value'):
                    ratio *= (1.0 + asset.business_value / 20)  # Scale factor

                # Boost ratio for vulnerabilities with exploit code available
                if hasattr(vuln, 'exploit') and vuln.exploit:
                    ratio *= 1.3

                # Adjustments based on vulnerability characteristics
                if hasattr(vuln, 'complexity') and vuln.complexity == 'low':
                    # Low complexity = easier to exploit = higher priority
                    ratio *= 1.2

                scored_vulns.append((vuln, ratio))

            # Sort by risk-to-cost ratio (higher is better)
            sorted_vulns = sorted(scored_vulns, key=lambda x: x[1], reverse=True)
            return [v for v, _ in sorted_vulns[:self.temp_budget]]
        else:
            # Original approach enhanced with dynamic adaptation
            scored_vulns = []
            for vuln, asset, recent_attack in unpatched_vulns:
                # Base score from original approach
                base_score = vuln.cvss * getattr(vuln, 'exploit_likelihood', 0.5)

                # Apply boost for vulnerabilities on recently attacked assets
                if recent_attack:
                    base_score *= 1.5

                scored_vulns.append((vuln, base_score))

            # Sort by adjusted score
            sorted_vulns = sorted(scored_vulns, key=lambda x: x[1], reverse=True)
            return [v for v, _ in sorted_vulns[:self.temp_budget]]

    def _adapt_budget_to_belief(self) -> None:
        """Dynamically adapt budget based on belief state."""
        # Get most likely stage and stage probabilities
        likely_stage = self.belief_state.get_most_likely_stage()
        stage_probs = self.belief_state.kill_chain_beliefs

        # Calculate belief certainty for adaptation
        certainty = max(stage_probs.values())

        # Base budget adjustment on stage and certainty
        if likely_stage.value >= KillChainStage.EXPLOITATION.value:
            # Increase budget for later stages
            stage_factor = 1.0 + ((likely_stage.value - 3) * 0.15)  # 15% increase per stage after Exploitation
            certainty_factor = 0.5 + (certainty * 0.5)  # Scales from 0.5 to 1.0 based on certainty

            # Apply adjusted budget (with caps)
            adjustment = stage_factor * certainty_factor
            self.temp_budget = min(int(self.budget * adjustment), self.budget * 2)
        else:
            # Reset to normal budget for early stages
            self.temp_budget = self.budget

    def select_actions(self, state: State, attacker_actions: List[Vulnerability] = None) -> List[Vulnerability]:
        """Combine high-level and low-level policies with belief-based adaptation."""
        # Update tracking and step counter
        if attacker_actions:
            self.track_attack_attempts(state, attacker_actions)
        self.update_step()

        # Dynamically adapt budget based on belief state
        self._adapt_budget_to_belief()

        # Apply policy with belief-informed decisions
        priority_assets = self.high_level_policy(state)
        return self.low_level_policy(state, priority_assets)