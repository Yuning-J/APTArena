"""
use_trained_rl_defender.py: Example script showing how to use a trained RL defender
in the main APT3RTUSimulation.
"""

import os
import sys
import copy
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from apt3_rtu_simulation import APT3RTUSimulation
from classes.state import State, KillChainStage
from RL_defender_strategy import RLAdaptiveThreatIntelligenceStrategy

def run_simulation_with_rl_defender(data_file: str, q_table_file: str, num_steps: int = 20,
                                   defender_budget: int = 100000, attacker_budget: int = 15000,
                                   verbose: bool = True):
    """
    Run simulation using a trained RL defender.
    
    Args:
        data_file: Path to system data file
        q_table_file: Path to trained Q-table file
        num_steps: Number of simulation steps
        defender_budget: Defender budget
        attacker_budget: Attacker budget
        verbose: Whether to print detailed output
    """
    
    print(f"Loading trained RL defender from: {q_table_file}")
    
    # Initialize simulation
    simulation = APT3RTUSimulation(
        data_file=data_file,
        num_steps=num_steps,
        defender_budget=defender_budget,
        attacker_budget=attacker_budget,
        psi=1.0,
        cost_aware_attacker=True,
        cost_aware_defender=True,
        detection_averse=True,
        gamma=0.9,
        business_values_file="",
        use_hmm=False,
        attacker_sophistication=0.8
    )
    
    # Initialize cost cache
    simulation.initialize_cost_cache()
    
    # Load trained RL defender
    rl_defender = RLAdaptiveThreatIntelligenceStrategy(q_table_file=q_table_file)
    rl_defender.initialize(simulation.state, simulation._cost_cache)
    
    print(f"\n{'=' * 80}")
    print(f"RUNNING SIMULATION WITH TRAINED RL DEFENDER")
    print(f"Steps: {num_steps}")
    print(f"Defender Budget: ${defender_budget:,}")
    print(f"Attacker Budget: ${attacker_budget:,}")
    print(f"Q-table Size: {len(rl_defender.q_table)}")
    print(f"{'=' * 80}")
    
    # Track metrics
    attacker_successes = 0
    attacker_actions = 0
    defender_patches = 0
    defender_cost = 0.0
    compromised_assets = []
    
    # Run simulation
    for step in range(num_steps):
        if verbose:
            print(f"\n--- Step {step + 1}/{num_steps} ---")
        
        # Get attacker action
        attacker_action = simulation.get_next_attack_action(simulation.state)
        
        if verbose:
            print(f"Attacker Action: {attacker_action.get('action_type', 'unknown')}")
        
        if attacker_action and attacker_action.get('action_type') != 'pause':
            attacker_actions += 1
            
            # Execute attacker action
            action_result = simulation.execute_attack_action(simulation.state, attacker_action)
            
            # Track results
            if action_result.get('action_result', False):
                attacker_successes += 1
                if verbose:
                    print(f"✓ Attack successful: {action_result.get('action_type', 'unknown')}")
            else:
                if verbose:
                    print(f"✗ Attack failed: {action_result.get('reason', 'unknown')}")
        
        # Defender phase - use trained RL defender
        defender_actions = rl_defender.select_patches(simulation.state, simulation._remaining_defender_budget, step, num_steps)
        
        # Apply defender patches
        patches_applied = 0
        for vuln, cost in defender_actions:
            if cost <= simulation._remaining_defender_budget:
                vuln.apply_patch()
                simulation._remaining_defender_budget -= cost
                patches_applied += 1
                defender_patches += 1
                defender_cost += cost
        
        if verbose:
            print(f"RL Defender applied {patches_applied} patches")
            print(f"Remaining defender budget: ${simulation._remaining_defender_budget:,.2f}")
        
        # Update state
        simulation.current_step = step + 1
        
        # Track compromised assets
        current_compromised = [asset.asset_id for asset in simulation.system.assets if asset.is_compromised]
        if current_compromised:
            compromised_assets = current_compromised
    
    # Final results
    print(f"\n{'=' * 80}")
    print(f"SIMULATION COMPLETED")
    print(f"Attacker Success Rate: {attacker_successes}/{attacker_actions} ({attacker_successes/max(attacker_actions,1)*100:.1f}%)")
    print(f"Defender Patches Applied: {defender_patches}")
    print(f"Defender Total Cost: ${defender_cost:,.2f}")
    print(f"Assets Compromised: {len(compromised_assets)}")
    if compromised_assets:
        print(f"Compromised Asset IDs: {compromised_assets}")
    
    # Calculate business value impact
    total_business_value = sum(asset.business_value for asset in simulation.system.assets)
    preserved_value = sum(asset.business_value for asset in simulation.system.assets if not asset.is_compromised)
    lost_value = total_business_value - preserved_value
    
    # Calculate ROI
    roi = ((preserved_value - lost_value) / max(defender_cost, 1)) * 100 if defender_cost > 0 else 0
    
    print(f"Total Business Value: ${total_business_value:,.2f}")
    print(f"Value Preserved: ${preserved_value:,.2f}")
    print(f"Value Lost: ${lost_value:,.2f}")
    print(f"Value Preservation Rate: {preserved_value/total_business_value*100:.1f}%")
    print(f"ROI: {roi:.1f}%")
    print(f"{'=' * 80}")
    
    return {
        'attacker_success_rate': attacker_successes / max(attacker_actions, 1),
        'assets_compromised': len(compromised_assets),
        'compromised_asset_ids': compromised_assets,
        'defender_patches': defender_patches,
        'defender_cost': defender_cost,
        'value_preserved': preserved_value,
        'value_lost': lost_value,
        'value_preservation_rate': preserved_value / total_business_value,
        'roi': roi
    }

def main():
    """Example usage of trained RL defender."""
    
    # Example paths - adjust these to your actual file paths
    data_file = "data/systemData/apt3_scenario_enriched.json"
    q_table_file = "src/rl_defender_training_results/rl_training_20250616_142837/q_table.pkl"
    
    # Check if files exist
    if not os.path.exists(data_file):
        print(f"Error: Data file not found: {data_file}")
        print("Please update the data_file path in the script.")
        return
    
    if not os.path.exists(q_table_file):
        print(f"Error: Q-table file not found: {q_table_file}")
        print("Please train the RL defender first using RL_defender_simulation.py")
        print("Or update the q_table_file path to point to your trained model.")
        return
    
    # Run simulation with trained RL defender
    results = run_simulation_with_rl_defender(
        data_file=data_file,
        q_table_file=q_table_file,
        num_steps=20,
        defender_budget=100000,
        attacker_budget=15000,
        verbose=True
    )
    
    print(f"\nSimulation completed successfully!")
    print(f"Results: {results}")

if __name__ == "__main__":
    main() 