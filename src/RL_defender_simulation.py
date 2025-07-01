"""
RL_defender_simulation.py: Training script for RLAdaptiveThreatIntelligenceStrategy using APT3RTUSimulation.
This script runs multiple simulation episodes to train the Q-learning table, saving it to q_table.pkl.
"""

import os
import sys
import argparse
import json
import numpy as np
from datetime import datetime
import copy
import random
import logging

# Add parent directory to path to import simulation modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from apt3_rtu_simulation import APT3RTUSimulation
from classes.state import State, KillChainStage
from classes.attacker_hybrid_apt3 import HybridGraphPOSGAttackerAPT3
from RL_defender_strategy import RLAdaptiveThreatIntelligenceStrategy

# Configure logging
logging.basicConfig(level=logging.INFO, filename="rl_defender_training_log.txt",
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_rl_defender(data_file: str, num_episodes: int, num_steps: int, defender_budget: float, attacker_budget: float,
                      output_dir: str, verbose: bool = False) -> dict:
    """
    Train the RLAdaptiveThreatIntelligenceStrategy using APT3RTUSimulation over multiple episodes.

    Args:
        data_file: Path to the JSON data file for system configuration.
        num_episodes: Number of training episodes to run.
        num_steps: Number of steps per simulation episode.
        defender_budget: Defender's patching budget in dollars.
        attacker_budget: Attacker's budget in dollars.
        output_dir: Directory to save training results and Q-table.
        verbose: Whether to print detailed logs during training.

    Returns:
        dict: Training summary with metrics (e.g., average ROI, value preserved).
    """
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    training_output_dir = os.path.join(output_dir, f"rl_defender_training_{timestamp}")
    os.makedirs(training_output_dir, exist_ok=True)

    # Initialize simulation
    try:
        simulation = APT3RTUSimulation(
            data_file=data_file,
            num_steps=num_steps,
            defender_budget=int(defender_budget),
            attacker_budget=int(attacker_budget),
            psi=1.0,
            cost_aware_attacker=True,
            cost_aware_defender=True,
            detection_averse=True,
            gamma=0.9,
            business_values_file="",
            use_hmm=False,
            attacker_sophistication=0.8
        )
        original_system = copy.deepcopy(simulation.system)
        logger.info(f"Initial simulation setup with {len(simulation.vuln_lookup)} vulnerabilities")
    except Exception as e:
        print(f"Error initializing simulation: {e}")
        logger.error(f"Error initializing simulation: {e}", exc_info=True)
        return {}

    # Initialize cost cache
    simulation.initialize_cost_cache()

    # Initialize RL strategy
    q_table_file = os.path.join(training_output_dir, "q_table.pkl")
    rl_strategy = RLAdaptiveThreatIntelligenceStrategy(q_table_file=q_table_file)
    rl_strategy.initialize(simulation.state, simulation._cost_cache)

    # Training metrics
    training_metrics = {
        "episodes": num_episodes,
        "average_roi": [],
        "average_value_preserved": [],
        "average_protected_value": [],
        "average_lost_value": [],
        "average_total_patches": [],
        "average_unpatched_critical": [],
        "average_compromised_assets": [],
        "average_reward": []
    }

    print(f"\n{'=' * 80}")
    print(f"TRAINING RLAdaptiveThreatIntelligenceStrategy - {num_episodes} Episodes")
    print(f"Defender Budget: ${defender_budget:,.2f}, Steps per Episode: {num_steps}")
    print(f"Attacker Budget: ${attacker_budget:,.2f}")
    print(f"Output Directory: {training_output_dir}")
    print(f"{'=' * 80}")

    # Training loop
    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        logger.info(f"Episode {episode + 1}: Starting training")

        # Reset simulation state
        simulation.system = copy.deepcopy(original_system)
        simulation.reset_exploit_status()
        simulation.state = State(k=KillChainStage.RECONNAISSANCE.value, system=simulation.system)
        simulation._remaining_defender_budget = int(defender_budget)
        simulation._remaining_attacker_budget = int(attacker_budget)
        simulation._patched_cves = set()
        simulation._exploited_cves = set()

        # Reinitialize vuln_lookup
        simulation.vuln_lookup = {}
        for asset in simulation.system.assets:
            for comp in asset.components:
                for vuln in comp.vulnerabilities:
                    vuln_key = f"{vuln.cve_id}:{asset.asset_id}:{comp.id}"
                    simulation.vuln_lookup[vuln_key] = (vuln, asset, comp)

        # Reinitialize attacker
        simulation.attacker = HybridGraphPOSGAttackerAPT3(
            system=simulation.system,
            mitre_mapper=simulation.mitre_mapper,
            cwe_canfollow_path="../data/CTI/raw/canfollow.json",
            cost_aware=True,
            detection_averse=True,
            enhanced_exploit_priority=True,
            sophistication_level=0.8,
            cost_calculator=simulation.cost_calculator,
            cost_cache=simulation._cost_cache
        )

        # Reinitialize strategy
        rl_strategy.initialize(simulation.state, simulation._cost_cache)

        # Episode metrics
        episode_attacker_successes = 0
        episode_attacker_actions = 0
        episode_defender_patches = 0
        episode_defender_cost = 0.0
        episode_defender_reward = 0.0

        # Run simulation episode
        try:
            for step in range(num_steps):
                if verbose:
                    print(f"  Step {step + 1}/{num_steps}")

                # Get attacker action
                attacker_action = simulation.get_next_attack_action(simulation.state)
                
                if attacker_action and attacker_action.get('action_type') != 'pause':
                    episode_attacker_actions += 1
                    
                    # Execute attacker action
                    action_result = simulation.execute_attack_action(simulation.state, attacker_action)
                    
                    # Track attacker metrics
                    if action_result.get('action_result', False):
                        episode_attacker_successes += 1

                # Defender phase - use RL strategy
                defender_actions = rl_strategy.select_patches(simulation.state, simulation._remaining_defender_budget, step, num_steps)
                
                # Apply defender patches
                for vuln, cost in defender_actions:
                    if cost <= simulation._remaining_defender_budget:
                        vuln.apply_patch()
                        simulation._remaining_defender_budget -= cost
                        simulation._patched_cves.add(vuln.cve_id)
                        episode_defender_patches += 1
                        episode_defender_cost += cost

                # Update state
                simulation.current_step = step + 1

        except Exception as e:
            print(f"Error running simulation in episode {episode + 1}: {e}")
            logger.error(f"Error running simulation in episode {episode + 1}: {e}", exc_info=True)
            continue

        # Calculate episode metrics
        attacker_success_rate = episode_attacker_successes / max(episode_attacker_actions, 1)
        compromised_assets = sum(1 for asset in simulation.system.assets if asset.is_compromised)
        
        # Calculate defender metrics
        total_business_value = sum(asset.business_value for asset in simulation.system.assets)
        preserved_value = sum(asset.business_value for asset in simulation.system.assets if not asset.is_compromised)
        lost_value = total_business_value - preserved_value
        
        # Calculate ROI
        roi = ((preserved_value - lost_value) / max(episode_defender_cost, 1)) * 100 if episode_defender_cost > 0 else 0

        # Store metrics
        training_metrics["average_roi"].append(roi)
        training_metrics["average_value_preserved"].append(preserved_value)
        training_metrics["average_protected_value"].append(preserved_value)
        training_metrics["average_lost_value"].append(lost_value)
        training_metrics["average_total_patches"].append(episode_defender_patches)
        training_metrics["average_unpatched_critical"].append(0)  # Simplified
        training_metrics["average_compromised_assets"].append(compromised_assets)
        training_metrics["average_reward"].append(rl_strategy.last_reward if hasattr(rl_strategy, 'last_reward') else 0.0)

        if verbose:
            print(f"Episode {episode + 1} Summary:")
            print(f"  ROI: {roi:.1f}%")
            print(f"  Value Preserved: ${preserved_value:,.2f}")
            print(f"  Protected Value: ${preserved_value:,.2f}")
            print(f"  Lost Value: ${lost_value:,.2f}")
            print(f"  Total Patches: {episode_defender_patches}")
            print(f"  Compromised Assets: {compromised_assets}")
            print(f"  Attacker Success Rate: {attacker_success_rate:.2f}")

        # Save Q-table periodically
        if (episode + 1) % 50 == 0:
            rl_strategy._save_q_table()
            print(f"Saved Q-table to {q_table_file}")

    # Save final Q-table
    rl_strategy._save_q_table()
    print(f"Final Q-table saved to {q_table_file}")

    # Compute average metrics
    summary = {
        "num_episodes": num_episodes,
        "average_roi": np.mean(training_metrics["average_roi"]),
        "std_roi": np.std(training_metrics["average_roi"]),
        "average_value_preserved": np.mean(training_metrics["average_value_preserved"]),
        "std_value_preserved": np.std(training_metrics["average_value_preserved"]),
        "average_protected_value": np.mean(training_metrics["average_protected_value"]),
        "std_protected_value": np.std(training_metrics["average_protected_value"]),
        "average_lost_value": np.mean(training_metrics["average_lost_value"]),
        "std_lost_value": np.std(training_metrics["average_lost_value"]),
        "average_total_patches": np.mean(training_metrics["average_total_patches"]),
        "std_total_patches": np.std(training_metrics["average_total_patches"]),
        "average_unpatched_critical": np.mean(training_metrics["average_unpatched_critical"]),
        "std_unpatched_critical": np.std(training_metrics["average_unpatched_critical"]),
        "average_compromised_assets": np.mean(training_metrics["average_compromised_assets"]),
        "std_compromised_assets": np.std(training_metrics["average_compromised_assets"]),
        "average_reward": np.mean(training_metrics["average_reward"]),
        "std_reward": np.std(training_metrics["average_reward"]),
        "final_q_table_size": len(rl_strategy.q_table),
        "final_epsilon": rl_strategy.epsilon
    }

    # Save training summary
    summary_file = os.path.join(training_output_dir, "training_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nTraining completed!")
    print(f"Final Q-table size: {len(rl_strategy.q_table)}")
    print(f"Final epsilon: {rl_strategy.epsilon:.3f}")
    print(f"Average ROI: {summary['average_roi']:.1f}%")
    print(f"Average value preserved: ${summary['average_value_preserved']:,.2f}")
    print(f"Training summary saved to: {summary_file}")

    return summary

def main():
    parser = argparse.ArgumentParser(description='Train RL Defender Strategy')
    parser.add_argument('--data_file', type=str, default='data/systemData/apt3_scenario_enriched.json',
                       help='Path to system data file')
    parser.add_argument('--num_episodes', type=int, default=100,
                       help='Number of training episodes')
    parser.add_argument('--num_steps', type=int, default=20,
                       help='Number of steps per episode')
    parser.add_argument('--defender_budget', type=float, default=100000,
                       help='Defender budget')
    parser.add_argument('--attacker_budget', type=float, default=15000,
                       help='Attacker budget')
    parser.add_argument('--output_dir', type=str, default='src/rl_defender_training_results',
                       help='Output directory for results')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Run training
    summary = train_rl_defender(
        data_file=args.data_file,
        num_episodes=args.num_episodes,
        num_steps=args.num_steps,
        defender_budget=args.defender_budget,
        attacker_budget=args.attacker_budget,
        output_dir=args.output_dir,
        verbose=args.verbose
    )

    print(f"\nTraining completed with summary: {summary}")

if __name__ == "__main__":
    main()
