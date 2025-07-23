#!/usr/bin/env python3
"""
Training script for the enhanced RL defender with threat intelligence features.
This script trains the RL agent while incorporating threat intelligence learning.
"""

import sys
import os
import json
import pickle
import logging
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from RL_defender_strategy import RLAdaptiveThreatIntelligenceStrategy
from classes.patching_strategies import ThreatIntelligenceStrategy
from apt3_rtu_simulation_100 import APT3RTUSimulation
from classes.state import State

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedRLTrainer:
    """Trainer for the enhanced RL defender with threat intelligence integration."""
    
    def __init__(self, data_file: str = "../data/systemData/apt3_scenario.json", 
                 num_episodes: int = 100, steps_per_episode: int = 20,
                 defender_budget: int = 5000, attacker_budget: int = 3000):
        self.data_file = data_file
        self.num_episodes = num_episodes
        self.steps_per_episode = steps_per_episode
        self.defender_budget = defender_budget
        self.attacker_budget = attacker_budget
        
        # Training results
        self.training_results = {
            'episodes': [],
            'final_q_table_size': 0,
            'final_epsilon': 0.0,
            'average_reward': 0.0,
            'average_roi': 0.0,
            'threat_intel_learning': []
        }
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"rl_defender_training_results/rl_defender_training_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"Enhanced RL Trainer initialized. Output directory: {self.output_dir}")

    def train_episode(self, episode: int) -> Dict:
        """Train a single episode with threat intelligence integration."""
        logger.info(f"Starting episode {episode + 1}/{self.num_episodes}")
        
        # Initialize simulation
        simulation = APT3RTUSimulation(
            data_file=self.data_file,
            num_steps=self.steps_per_episode,
            defender_budget=self.defender_budget,
            attacker_budget=self.attacker_budget,
            cost_aware_defender=True,
            cost_aware_attacker=True
        )
        
        # Initialize threat intelligence strategy
        threat_intel_strategy = ThreatIntelligenceStrategy()
        threat_intel_strategy.initialize(simulation.state, simulation._cost_cache)
        
        # Initialize enhanced RL defender
        rl_defender = RLAdaptiveThreatIntelligenceStrategy()
        rl_defender.initialize(simulation.state, simulation._cost_cache, self.defender_budget)
        
        # Episode tracking
        episode_rewards = []
        episode_patch_costs = []
        episode_patch_counts = []
        threat_intel_updates = []
        
        # Run episode steps
        for step in range(self.steps_per_episode):
            logger.debug(f"Episode {episode + 1}, Step {step + 1}")
            
            # Update threat intelligence with current state
            threat_intel_strategy._learn_from_current_state(simulation.state, step)
            
            # Update RL defender with threat intelligence features
            rl_defender.update_threat_intelligence_features(threat_intel_strategy)
            
            # Record threat intelligence state
            threat_intel_state = {
                'step': step,
                'asset_threat_levels': threat_intel_strategy.asset_threat_levels.copy(),
                'exploit_attempts': len(threat_intel_strategy.exploit_attempt_history),
                'compromise_sequence_length': len(threat_intel_strategy.compromise_sequence)
            }
            threat_intel_updates.append(threat_intel_state)
            
            # Get attacker action
            attacker_action = simulation.get_next_attack_action(simulation.state)
            
            # Execute attacker action
            attack_result = simulation.execute_attack_action(simulation.state, attacker_action)
            
            # Update threat intelligence with attack observation
            if attack_result.get('action_result', False):
                attack_observation = {
                    'action_type': attacker_action.get('action_type', 'unknown'),
                    'target_asset': str(attack_result.get('target_asset', '')),
                    'target_vuln': attack_result.get('target_vuln', ''),
                    'success': attack_result.get('action_result', False),
                    'techniques': attack_result.get('techniques', [])
                }
                threat_intel_strategy.observe_attack_behavior(attack_observation)
            
            # Get defender patches
            remaining_budget = self.defender_budget - sum(episode_patch_costs)
            patches = rl_defender.select_patches(
                simulation.state, 
                remaining_budget, 
                step, 
                self.steps_per_episode
            )
            
            # Apply patches
            patch_cost = 0.0
            for vuln, cost in patches:
                vuln.is_patched = True
                patch_cost += cost
            
            episode_patch_costs.append(patch_cost)
            episode_patch_counts.append(len(patches))
            
            # Calculate reward
            reward = rl_defender.last_reward
            episode_rewards.append(reward)
            
            # Update simulation state
            simulation.state.system.time_step += 1
            
            # Check for simulation termination
            should_terminate, reason = simulation.should_terminate_simulation(
                step, self.steps_per_episode, sum(episode_patch_costs)
            )
            if should_terminate:
                logger.info(f"Episode {episode + 1} terminated early: {reason}")
                break
        
        # Calculate episode metrics
        total_patch_cost = sum(episode_patch_costs)
        total_patches = sum(episode_patch_counts)
        average_reward = np.mean(episode_rewards) if episode_rewards else 0.0
        
        # Calculate ROI
        total_business_value = sum(
            getattr(asset, 'business_value', getattr(asset, 'criticality_level', 3) * 5000)
            for asset in simulation.state.system.assets
        )
        preserved_value = total_business_value
        for asset in simulation.state.system.assets:
            if asset.is_compromised:
                preserved_value -= getattr(asset, 'business_value', 10000)
        
        roi = ((preserved_value - total_patch_cost) / max(total_patch_cost, 1.0)) * 100 if total_patch_cost > 0 else 0.0
        
        episode_result = {
            'episode': episode + 1,
            'total_reward': sum(episode_rewards),
            'average_reward': average_reward,
            'total_patch_cost': total_patch_cost,
            'total_patches': total_patches,
            'roi': roi,
            'final_epsilon': rl_defender.epsilon,
            'q_table_size': len(rl_defender.q_table),
            'threat_intel_updates': threat_intel_updates,
            'compromised_assets': sum(1 for asset in simulation.state.system.assets if asset.is_compromised),
            'total_assets': len(simulation.state.system.assets)
        }
        
        logger.info(f"Episode {episode + 1} completed: "
                   f"Reward={average_reward:.3f}, ROI={roi:.1f}%, "
                   f"Patches={total_patches}, Cost=${total_patch_cost:.2f}")
        
        return episode_result

    def train(self):
        """Main training loop."""
        logger.info(f"Starting enhanced RL training with {self.num_episodes} episodes")
        logger.info(f"Steps per episode: {self.steps_per_episode}")
        logger.info(f"Defender budget: ${self.defender_budget}")
        logger.info(f"Attacker budget: ${self.attacker_budget}")
        
        # Training loop
        for episode in range(self.num_episodes):
            try:
                episode_result = self.train_episode(episode)
                self.training_results['episodes'].append(episode_result)
                
                # Save intermediate results every 10 episodes
                if (episode + 1) % 10 == 0:
                    self.save_intermediate_results(episode + 1)
                    
            except Exception as e:
                logger.error(f"Error in episode {episode + 1}: {e}")
                continue
        
        # Finalize training
        self.finalize_training()
        
        logger.info("Enhanced RL training completed successfully!")

    def save_intermediate_results(self, episode_num: int):
        """Save intermediate training results."""
        # Save Q-table
        q_table_file = os.path.join(self.output_dir, f"q_table_episode_{episode_num}.pkl")
        with open(q_table_file, 'wb') as f:
            pickle.dump(self.training_results['episodes'][-1].get('q_table', {}), f)
        
        # Save episode results
        results_file = os.path.join(self.output_dir, f"training_results_episode_{episode_num}.json")
        with open(results_file, 'w') as f:
            json.dump(self.training_results, f, indent=2, default=str)
        
        logger.info(f"Intermediate results saved for episode {episode_num}")

    def finalize_training(self):
        """Finalize training and save results."""
        # Calculate final metrics
        episodes = self.training_results['episodes']
        if not episodes:
            logger.error("No episodes completed successfully")
            return
        
        # Calculate averages
        avg_reward = np.mean([ep['average_reward'] for ep in episodes])
        avg_roi = np.mean([ep['roi'] for ep in episodes])
        avg_patches = np.mean([ep['total_patches'] for ep in episodes])
        avg_cost = np.mean([ep['total_patch_cost'] for ep in episodes])
        
        # Get final Q-table size and epsilon
        final_episode = episodes[-1]
        final_q_table_size = final_episode['q_table_size']
        final_epsilon = final_episode['final_epsilon']
        
        # Update training results
        self.training_results.update({
            'final_q_table_size': final_q_table_size,
            'final_epsilon': final_epsilon,
            'average_reward': avg_reward,
            'average_roi': avg_roi,
            'average_patches': avg_patches,
            'average_cost': avg_cost,
            'total_episodes': len(episodes)
        })
        
        # Save final Q-table
        q_table_file = os.path.join(self.output_dir, "q_table.pkl")
        with open(q_table_file, 'wb') as f:
            pickle.dump(final_episode.get('q_table', {}), f)
        
        # Save training summary
        summary_file = os.path.join(self.output_dir, "training_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(self.training_results, f, indent=2, default=str)
        
        # Print training summary
        logger.info("=" * 60)
        logger.info("ENHANCED RL TRAINING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total Episodes: {len(episodes)}")
        logger.info(f"Average Reward: {avg_reward:.3f}")
        logger.info(f"Average ROI: {avg_roi:.1f}%")
        logger.info(f"Average Patches per Episode: {avg_patches:.1f}")
        logger.info(f"Average Cost per Episode: ${avg_cost:.2f}")
        logger.info(f"Final Q-table Size: {final_q_table_size}")
        logger.info(f"Final Epsilon: {final_epsilon:.3f}")
        logger.info(f"Results saved to: {self.output_dir}")
        logger.info("=" * 60)

def main():
    """Main training function."""
    # Training parameters
    trainer = EnhancedRLTrainer(
        data_file="../data/systemData/apt3_scenario.json",
        num_episodes=50,  # Reduced for faster training
        steps_per_episode=15,
        defender_budget=5000,
        attacker_budget=3000
    )
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main() 