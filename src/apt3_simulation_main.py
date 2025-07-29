#!/usr/bin/env python3
"""
APT3 RTU Simulation - Main entry point
This script coordinates the core simulation, running, and visualization components
"""

import argparse
import os
import json
from datetime import datetime

from apt3_simulation_run import APT3SimulationRunner
from apt3_simulation_viz import APT3SimulationVisualizer

def main():
    parser = argparse.ArgumentParser(description="Run APT3 RTU-targeted simulation with enhanced strategic attacker")
    parser.add_argument("--data-file", type=str, default="./data/systemData/apt3_scenario_enriched.json",
                        help="Path to system data JSON file")
    parser.add_argument("--num-steps", type=int, default=100, help="Number of simulation steps")
    parser.add_argument("--defender-budget", type=int, default=7500, help="Defender's patching budget in dollars")
    parser.add_argument("--attacker-budget", type=int, default=15000, help="Attacker's budget in dollars")
    parser.add_argument("--psi", type=float, default=1.0, help="Weight balancing risk and operational cost")
    parser.add_argument("--cost-aware-attacker", action="store_true", default=True)
    parser.add_argument("--cost-aware-defender", action="store_true", default=True)
    parser.add_argument("--detection-averse", action="store_true", default=True)
    parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor for payoffs")
    parser.add_argument("--business-values-file", type=str, help="Path to business values JSON file")
    parser.add_argument("--use-hmm", action="store_true", default=False)
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--output-dir", type=str, default="apt3_simulation_results")
    parser.add_argument("--attacker-sophistication", type=float, default=0.9)
    parser.add_argument("--cost-cache-file", type=str, default=None)
    parser.add_argument("--num-trials", type=int, default=100, help="Number of simulation trials")
    
    args = parser.parse_args()
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"simulation_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    try:
        # Initialize and run simulation
        print(f"Initializing APT3 RTU Simulation...")
        simulation = APT3SimulationRunner(
            data_file=args.data_file,
            num_steps=args.num_steps,
            defender_budget=args.defender_budget,
            attacker_budget=args.attacker_budget,
            psi=args.psi,
            cost_aware_attacker=args.cost_aware_attacker,
            cost_aware_defender=args.cost_aware_defender,
            detection_averse=args.detection_averse,
            gamma=args.gamma,
            business_values_file=args.business_values_file,
            use_hmm=args.use_hmm,
            attacker_sophistication=args.attacker_sophistication,
            cost_cache_file=args.cost_cache_file
        )
        simulation.output_dir = output_dir
        
        print(f"Running strategy comparison with {args.num_trials} trials...")
        results = simulation.compare_strategies(
            defender_budget=args.defender_budget,
            num_steps=args.num_steps,
            num_trials=args.num_trials,
            verbose=args.verbose
        )
        
        # Save results
        summary_file = os.path.join(output_dir, "simulation_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=4, default=str)
        print(f"Saved summary to {summary_file}")
        
        # Save strategic metrics
        strategic_summary = {name: [trial.get('attacker_metrics', {}) for trial in result['trials']] 
                            for name, result in results.items()}
        strategic_file = os.path.join(output_dir, "strategic_metrics.json")
        with open(strategic_file, 'w') as f:
            json.dump(strategic_summary, f, indent=4)
        print(f"Saved strategic metrics to {strategic_file}")
        
        # Create visualizations
        print(f"\nGenerating visualizations...")
        visualizer = APT3SimulationVisualizer(output_dir)
        visualizer.visualize_results(results, args.num_trials)
        visualizer.save_summary_report(results)
        
        print(f"\n{'='*80}")
        print(f"SIMULATION COMPLETE")
        print(f"{'='*80}")
        print(f"Results saved to: {output_dir}")
        print(f"Visualizations saved to: {viz_dir}")
        print(f"Summary report: {os.path.join(output_dir, 'simulation_summary_report.txt')}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Validation error: {e}")
    except TypeError as e:
        print(f"Serialization error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()