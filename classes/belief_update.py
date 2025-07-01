# belief_updates.py
"""
Implementation of Bayesian belief update methods for the POMDP attacker policy.
These methods are used to update the attacker's beliefs based on observations.
"""
import numpy as np
import logging

logger = logging.getLogger(__name__)

def update_exploited_belief(current_belief: float, observation_result: bool, observation_accuracy: float = 0.9) -> float:
    """
    Update belief about whether a vulnerability is exploited using Bayes' rule.

    Args:
        current_belief: Current probability that vulnerability is exploited (0-1)
        observation_result: Boolean indicating whether vulnerability was observed as exploited
        observation_accuracy: Accuracy of the observation (default: 0.9)

    Returns:
        float: Updated belief probability
    """
    # Handle edge cases
    if current_belief <= 0 and not observation_result:
        return 0.0
    if current_belief >= 1 and observation_result:
        return 1.0

    # Set conditional probabilities
    p_observe_exploited_given_exploited = observation_accuracy
    p_observe_exploited_given_not_exploited = 1 - observation_accuracy
    p_observe_not_exploited_given_exploited = 1 - observation_accuracy
    p_observe_not_exploited_given_not_exploited = observation_accuracy

    # Bayes' rule: P(is_exploited | observation) = P(observation | is_exploited) * P(is_exploited) / P(observation)
    if observation_result:  # Observed as exploited
        numerator = p_observe_exploited_given_exploited * current_belief
        denominator = (p_observe_exploited_given_exploited * current_belief) + \
                      (p_observe_exploited_given_not_exploited * (1 - current_belief))
    else:  # Observed as not exploited
        numerator = p_observe_not_exploited_given_exploited * current_belief
        denominator = (p_observe_not_exploited_given_exploited * current_belief) + \
                      (p_observe_not_exploited_given_not_exploited * (1 - current_belief))

    # Handle potential numerical issues
    if denominator == 0:
        return current_belief

    updated_belief = numerator / denominator

    # Ensure result is within valid probability range
    return max(0.0, min(1.0, updated_belief))

def update_patch_belief(current_belief, observation_result, observation_accuracy=0.9):
    """
    Update belief about whether a vulnerability is patched using Bayes' rule.
    
    Args:
        current_belief: Current probability that vulnerability is patched (0-1)
        observation_result: Boolean indicating whether vulnerability was observed as patched
        observation_accuracy: Accuracy of the observation (default: 0.9)
    
    Returns:
        float: Updated belief probability
    """
    # Handle edge cases
    if current_belief <= 0:
        return 0.0 if not observation_result else 0.1
    if current_belief >= 1:
        return 1.0 if observation_result else 0.9
    
    # Set conditional probabilities
    # P(observe_patched | is_patched)
    p_observe_patched_given_patched = observation_accuracy
    
    # P(observe_patched | not_patched)
    p_observe_patched_given_not_patched = 1 - observation_accuracy
    
    # P(observe_not_patched | is_patched)
    p_observe_not_patched_given_patched = 1 - observation_accuracy
    
    # P(observe_not_patched | not_patched)
    p_observe_not_patched_given_not_patched = observation_accuracy
    
    # Bayes' rule: P(is_patched | observation) = P(observation | is_patched) * P(is_patched) / P(observation)
    if observation_result:  # Observed as patched
        numerator = p_observe_patched_given_patched * current_belief
        denominator = (p_observe_patched_given_patched * current_belief) + \
                      (p_observe_patched_given_not_patched * (1 - current_belief))
    else:  # Observed as not patched
        numerator = p_observe_not_patched_given_patched * current_belief
        denominator = (p_observe_not_patched_given_patched * current_belief) + \
                      (p_observe_not_patched_given_not_patched * (1 - current_belief))
    
    # Handle potential numerical issues
    if denominator == 0:
        return current_belief  # No update if denominator is zero
        
    updated_belief = numerator / denominator
    
    # Ensure result is within valid probability range
    return max(0.0, min(1.0, updated_belief))

def update_compromise_belief(current_belief, exploit_success, exploit_likelihood=0.7):
    """
    Update belief about whether an asset is compromised based on exploit results.
    
    Args:
        current_belief: Current probability that asset is compromised (0-1)
        exploit_success: Boolean indicating whether the exploit was successful
        exploit_likelihood: Probability of exploit success against unpatched vulnerability (default: 0.7)
    
    Returns:
        float: Updated belief probability
    """
    if exploit_success:
        # If exploit succeeded, asset is definitely compromised
        return 1.0
    else:
        # If exploit failed, it could still be compromised through other means
        # P(compromised | exploit_failed) using Bayes' rule
        
        # P(exploit_failed | compromised)
        # Even if compromised, exploit could fail for various reasons
        p_fail_given_compromised = 0.3
        
        # P(exploit_failed | not_compromised)
        # If not compromised, exploit likely fails
        p_fail_given_not_compromised = 0.8
        
        # Bayes' rule
        numerator = p_fail_given_compromised * current_belief
        denominator = (p_fail_given_compromised * current_belief) + \
                      (p_fail_given_not_compromised * (1 - current_belief))
        
        # Handle potential numerical issues
        if denominator == 0:
            return current_belief
            
        updated_belief = numerator / denominator
        
        # Ensure result is within valid probability range
        return max(0.0, min(1.0, updated_belief))

def update_detection_belief(current_belief, observation, defender_accuracy=0.8):
    """
    Update belief about whether the attacker has been detected by the defender.
    
    Args:
        current_belief: Current probability that attacker is detected (0-1)
        observation: Dictionary containing observation about detection
            - 'detected': Boolean indicating whether detection was observed
            - 'confidence': Optional confidence in the detection observation (0-1)
        defender_accuracy: Accuracy of defender detection capabilities (default: 0.8)
    
    Returns:
        float: Updated belief probability
    """
    # Extract detection observation
    detected = observation.get('detected', False)
    confidence = observation.get('confidence', defender_accuracy)
    
    # Set conditional probabilities
    if detected:
        # P(observe_detected | actually_detected)
        p_observe_given_true = confidence
        
        # P(observe_detected | not_detected)
        p_observe_given_false = 1 - confidence
    else:
        # P(observe_not_detected | actually_detected)
        p_observe_given_true = 1 - confidence
        
        # P(observe_not_detected | not_detected)
        p_observe_given_false = confidence
    
    # Bayes' rule
    numerator = p_observe_given_true * current_belief
    denominator = (p_observe_given_true * current_belief) + \
                  (p_observe_given_false * (1 - current_belief))
    
    # Handle potential numerical issues
    if denominator == 0:
        return current_belief
        
    updated_belief = numerator / denominator
    
    # Ensure result is within valid probability range
    return max(0.0, min(1.0, updated_belief))

def update_action_outcome_belief(action_history, current_stage, tactic, target_vuln=None):
    """
    Calculate belief about success probability for a specific action based on history.
    
    Args:
        action_history: List of past action results
        current_stage: Current kill chain stage
        tactic: Tactic being considered
        target_vuln: Optional vulnerability being targeted
    
    Returns:
        float: Probability of success for the action
    """
    # Filter history to relevant actions
    relevant_history = [a for a in action_history if a.get('action_type') == tactic.name.lower()]
    
    # If no relevant history, use prior based on kill chain stage
    if not relevant_history:
        # Actions in early kill chain stages are typically easier
        if current_stage.value <= 3:  # Recon, Weaponization, Delivery
            return 0.7
        elif current_stage.value <= 5:  # Exploitation, Installation
            return 0.5
        else:  # Command & Control, Actions on Objectives
            return 0.3
    
    # Calculate success rate from history
    success_count = sum(1 for a in relevant_history if a.get('action_result', False))
    total_count = len(relevant_history)
    
    # If targeting specific vulnerability, weight by its history
    if target_vuln:
        vuln_history = [a for a in action_history if a.get('target_vuln') == target_vuln.cve_id]
        if vuln_history:
            vuln_success_count = sum(1 for a in vuln_history if a.get('action_result', False))
            vuln_total_count = len(vuln_history)
            
            # Combine general tactic history with specific vulnerability history
            # Weight vulnerability history more heavily
            combined_success_rate = (0.3 * (success_count / total_count)) + \
                                   (0.7 * (vuln_success_count / vuln_total_count))
            
            return combined_success_rate
    
    # Return basic success rate
    return success_count / total_count