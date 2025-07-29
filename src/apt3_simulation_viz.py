#!/usr/bin/env python3
"""
APT3 RTU Simulation Visualization - Result analysis and plotting
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
from typing import Dict, List, Any

class APT3SimulationVisualizer:
    """Visualization and analysis for APT3 simulation results"""
    
    def __init__(self, output_dir: str = "apt3_simulation_results"):
        self.output_dir = output_dir
        self.viz_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(self.viz_dir, exist_ok=True)
    
    def visualize_results(self, aggregated_results: Dict, num_trials: int = 100):
        """Create all visualizations for the simulation results"""
        
        # Extract strategy names
        strategies = list(aggregated_results.keys())
        
        # Visualization metrics
        metrics_to_plot = {
            "Protected Value": ('protected_value_runs', 'Protected Value ($)', 'protected_value'),
            "Lost Value": ('lost_value_runs', 'Lost Value ($)', 'lost_value'),
            "Value Preserved": ('value_preserved_runs', 'Value Preserved ($)', 'value_preserved'),
            "ROI": ('roi_runs', 'Return on Investment (%)', 'roi'),
            "Compromised Assets": ('compromised_assets_runs', 'Compromised Assets', 'compromised_assets'),
            "Detection Coverage": ('detection_coverage_runs', 'Detection Coverage (%)', 'detection_coverage'),
            "Avg Time to Detection": (
                'avg_time_to_detection_runs', 'Average Time to Detection (steps)', 'avg_time_to_detection'),
            "Attack Disruption Rate": (
                'attack_disruption_rate_runs', 'Attack Disruption Rate (%)', 'attack_disruption_rate'),
            "Spearphishing Success Rate": (
                'spearphishing_success_rate_runs', 'Spearphishing Success Rate (%)', 'spearphishing_success_rate'),
            "Credential Harvesting Count": (
                'credential_harvesting_count_runs', 'Credential Harvesting Count', 'credential_harvesting_count'),
            "Observations Collected": (
                'observations_collected_runs', 'Observations Collected', 'observations_collected')
        }
        
        # Create bar plots
        for metric_name, (metric_key, metric_label, file_prefix) in metrics_to_plot.items():
            self._create_bar_plot(aggregated_results, strategies, metric_name, metric_key, 
                                metric_label, file_prefix, num_trials)
            
            # Create violin plots for distribution analysis
            self._create_violin_plot(aggregated_results, strategies, metric_name, metric_key,
                                   metric_label, file_prefix, num_trials)
        
        # Create attack path usage plot
        self._create_attack_path_plot(aggregated_results, strategies, num_trials)
        
        # Create RTU compromise analysis plots
        self._create_rtu_compromise_plots(aggregated_results, strategies, num_trials)
        
        # Create hybrid strategy specific plots if applicable
        if 'Hybrid Defender' in strategies:
            self._create_hybrid_strategy_plots(aggregated_results, num_trials)
        
        print(f"\nAll visualizations saved to {self.viz_dir}")
    
    def _create_bar_plot(self, aggregated_results, strategies, metric_name, metric_key, 
                        metric_label, file_prefix, num_trials):
        """Create bar plot for a specific metric"""
        plt.figure(figsize=(12, 6))
        values = [aggregated_results[strategy_name]['run_statistics'][metric_key] for strategy_name in strategies]
        
        # Filter out empty or all-None lists for bar plot
        means = [np.mean([v for v in vals if v is not None]) if vals and any(v is not None for v in vals) else 0.0 
                for vals in values]
        stds = [np.std([v for v in vals if v is not None]) if vals and any(v is not None for v in vals) else 0.0 
               for vals in values]
        
        bars = plt.bar(strategies, means, yerr=stds, color='skyblue', alpha=0.7, edgecolor='navy', capsize=5)
        
        # Add value labels on bars
        for bar, value in zip(bars, means):
            height = bar.get_height()
            label = f'${value:,.0f}' if 'Value' in metric_name else f'{value:.1f}%' if 'Rate' in metric_name or 'Coverage' in metric_name else f'{value:.1f}'
            plt.text(bar.get_x() + bar.get_width() / 2., height + max(means) * 0.01 if max(means) > 0 else 0.1, 
                    label, ha='center', va='bottom', fontsize=10)
        
        plt.xlabel("Strategy", fontsize=12)
        plt.ylabel(metric_name, fontsize=12)
        plt.title(f"Mean {metric_name} by Strategy (APT3 Enhanced, {num_trials} Trials)", 
                 fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        save_path = os.path.join(self.viz_dir, f"apt3_enhanced_{file_prefix}_bar.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {metric_name} bar plot to {save_path}")
    
    def _create_violin_plot(self, aggregated_results, strategies, metric_name, metric_key,
                           metric_label, file_prefix, num_trials):
        """Create violin plot for distribution analysis"""
        plt.figure(figsize=(14, 8))
        
        # Prepare data
        data = [aggregated_results[strategy_name]['run_statistics'][metric_key] for strategy_name in strategies]
        valid_data = [[v for v in d if v is not None] for d in data]
        
        # Only keep non-empty lists
        nonempty_indices = [i for i, d in enumerate(valid_data) if len(d) > 0]
        if not nonempty_indices:
            print(f"Skipping {metric_name} violin plot: no valid data to plot.")
            plt.close()
            return
        
        plot_data = [valid_data[i] for i in nonempty_indices]
        plot_positions = [i for i in nonempty_indices]
        plot_labels = [strategies[i] for i in nonempty_indices]
        
        # Ensure all entries are valid
        all_1d = all(isinstance(d, list) and all(isinstance(x, (int, float, np.integer, np.floating)) for x in d) 
                    for d in plot_data)
        if not all_1d:
            print(f"Skipping {metric_name} violin plot: plot_data contains non-1D or non-numeric entries.")
            plt.close()
            return
        
        try:
            violin_parts = plt.violinplot(plot_data, positions=plot_positions, widths=0.15, showmeans=True)
            for pc in violin_parts['bodies']:
                pc.set_facecolor('lightcoral')
                pc.set_edgecolor('darkred')
                pc.set_alpha(0.6)
            
            plt.boxplot(plot_data, positions=plot_positions, widths=0.15, showfliers=True)
            plt.xlabel('Strategy', fontsize=12)
            plt.ylabel(metric_label, fontsize=12)
            plt.title(f'Distribution of {metric_label} Across {num_trials} Trials per Strategy', fontsize=14)
            plt.xticks(plot_positions, plot_labels, rotation=45, ha='right')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            save_path = os.path.join(self.viz_dir, f"apt3_enhanced_{file_prefix}_distribution.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved {metric_name} distribution plot to {save_path}")
        except Exception as e:
            print(f"Skipping {metric_name} violin plot due to error: {e}")
            plt.close()
    
    def _create_attack_path_plot(self, aggregated_results, strategies, num_trials):
        """Create attack path usage visualization"""
        # Track attack path usage
        path_usage = {strategy_name: {} for strategy_name in strategies}
        rtu_compromise_counts = {strategy_name: {} for strategy_name in strategies}
        
        for strategy_name in strategies:
            for trial in aggregated_results[strategy_name]['trials']:
                for path in trial.get('attack_paths_used', []):
                    path_key = tuple(
                        (step['target_vuln'] or 'None', step['target_asset'], step['tactic'], step['success']) 
                        for step in path.get('path_steps', [])
                    )
                    path_usage[strategy_name][path_key] = path_usage[strategy_name].get(path_key, 0) + 1
                    if path.get('rtu_compromised', False):
                        rtu_compromise_counts[strategy_name][path_key] = rtu_compromise_counts[strategy_name].get(
                            path_key, 0) + 1
        
        # Plot attack path usage
        plt.figure(figsize=(12, 6))
        for strategy_name in strategies:
            unique_paths = list(path_usage[strategy_name].keys())
            counts = list(path_usage[strategy_name].values())
            plt.bar([f"{strategy_name}_{i}" for i in range(len(unique_paths))], counts, width=0.2, label=strategy_name)
        
        plt.xlabel("Unique Attack Paths", fontsize=12)
        plt.ylabel("Usage Count", fontsize=12)
        plt.title(f"Attack Path Usage by Strategy (APT3 Enhanced, {num_trials} Trials)", fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        save_path = os.path.join(self.viz_dir, "apt3_enhanced_path_usage.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved path usage plot to {save_path}")
        
        # Plot RTU compromise by path
        plt.figure(figsize=(12, 6))
        for strategy_name in strategies:
            unique_paths = list(rtu_compromise_counts[strategy_name].keys())
            counts = list(rtu_compromise_counts[strategy_name].values())
            plt.bar([f"{strategy_name}_{i}" for i in range(len(unique_paths))], counts, width=0.2, label=strategy_name)
        
        plt.xlabel("Unique Attack Paths", fontsize=12)
        plt.ylabel("RTU Compromise Count", fontsize=12)
        plt.title(f"RTU Compromise Rate by Attack Path and Strategy (APT3 Enhanced, {num_trials} Trials)", 
                 fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        save_path = os.path.join(self.viz_dir, "apt3_enhanced_rtu_compromise_by_path.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved RTU compromise plot to {save_path}")
    
    def _create_rtu_compromise_plots(self, aggregated_results, strategies, num_trials):
        """Create RTU compromise time distribution plot"""
        plt.figure(figsize=(14, 8))
        
        for i, strategy_name in enumerate(strategies):
            rtu_times = [t for t in aggregated_results[strategy_name]['run_statistics']['time_to_rtu_compromise_runs']
                         if t is not None and t != num_trials + 1]
            
            if rtu_times and all(isinstance(x, (int, float, np.integer, np.floating)) for x in rtu_times):
                plt.violinplot([rtu_times], positions=[i], widths=0.15, showmeans=True)
        
        plt.xlabel('Strategy', fontsize=12)
        plt.ylabel('Time to RTU Compromise (steps)', fontsize=12)
        plt.title(f'Distribution of Time to RTU Compromise Across {num_trials} Trials per Strategy', fontsize=14)
        plt.xticks(range(len(strategies)), strategies, rotation=45, ha='right')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        save_path = os.path.join(self.viz_dir, "apt3_enhanced_time_to_rtu_compromise_distribution.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved RTU compromise time distribution to {save_path}")
    
    def _create_hybrid_strategy_plots(self, aggregated_results, num_trials):
        """Create specific plots for Hybrid Defender strategy analysis"""
        hybrid_data = aggregated_results.get('Hybrid Defender', {})
        if not hybrid_data or not hybrid_data.get('run_statistics'):
            return
        
        # Plot weight evolution
        plt.figure(figsize=(12, 6))
        
        ti_weights = hybrid_data['run_statistics']['final_ti_weight_runs']
        rl_weights = hybrid_data['run_statistics']['final_rl_weight_runs']
        
        if ti_weights and rl_weights:
            trials = list(range(1, len(ti_weights) + 1))
            plt.plot(trials, ti_weights, 'b-', label='TI Weight', linewidth=2)
            plt.plot(trials, rl_weights, 'r-', label='RL Weight', linewidth=2)
            
            plt.xlabel('Trial', fontsize=12)
            plt.ylabel('Weight', fontsize=12)
            plt.title(f'Hybrid Strategy Weight Evolution Across {num_trials} Trials', fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            save_path = os.path.join(self.viz_dir, "hybrid_weight_evolution.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved hybrid weight evolution plot to {save_path}")
        
        # Plot component performance comparison
        plt.figure(figsize=(10, 6))
        
        ti_perf = hybrid_data['run_statistics']['ti_performance_runs']
        rl_perf = hybrid_data['run_statistics']['rl_performance_runs']
        hybrid_perf = hybrid_data['run_statistics']['hybrid_performance_runs']
        
        if ti_perf and rl_perf and hybrid_perf:
            performance_data = [ti_perf, rl_perf, hybrid_perf]
            labels = ['TI Component', 'RL Component', 'Hybrid Combined']
            
            valid_data = [[v for v in d if v is not None] for d in performance_data]
            means = [np.mean(d) if d else 0 for d in valid_data]
            stds = [np.std(d) if d else 0 for d in valid_data]
            
            bars = plt.bar(labels, means, yerr=stds, color=['blue', 'red', 'purple'], alpha=0.7, capsize=5)
            
            for bar, value in zip(bars, means):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01, 
                        f'{value:.3f}', ha='center', va='bottom', fontsize=10)
            
            plt.ylabel('Performance Score', fontsize=12)
            plt.title(f'Hybrid Strategy Component Performance Comparison ({num_trials} Trials)', fontsize=14)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            save_path = os.path.join(self.viz_dir, "hybrid_component_performance.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved hybrid component performance plot to {save_path}")
    
    def save_summary_report(self, aggregated_results, output_file: str = None):
        """Save a comprehensive summary report"""
        if output_file is None:
            output_file = os.path.join(self.output_dir, "simulation_summary_report.txt")
        
        with open(output_file, 'w') as f:
            f.write("APT3 RTU SIMULATION SUMMARY REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary statistics for each strategy
            for strategy_name, data in aggregated_results.items():
                f.write(f"\nSTRATEGY: {strategy_name}\n")
                f.write("-" * 40 + "\n")
                
                mean_metrics = data.get('mean_metrics', {})
                std_metrics = data.get('std_metrics', {})
                
                f.write(f"Protected Value: ${mean_metrics.get('protected_value', 0):,.2f} (±${std_metrics.get('protected_value', 0):,.2f})\n")
                f.write(f"Lost Value: ${mean_metrics.get('lost_value', 0):,.2f} (±${std_metrics.get('lost_value', 0):,.2f})\n")
                f.write(f"Value Preserved: ${mean_metrics.get('value_preserved', 0):,.2f} (±${std_metrics.get('value_preserved', 0):,.2f})\n")
                f.write(f"ROI: {mean_metrics.get('roi', 0):.1f}% (±{std_metrics.get('roi', 0):.1f}%)\n")
                f.write(f"Total Patches: {mean_metrics.get('total_patches', 0):.1f} (±{std_metrics.get('total_patches', 0):.1f})\n")
                f.write(f"Total Cost: ${mean_metrics.get('total_patch_cost', 0):,.2f} (±${std_metrics.get('total_patch_cost', 0):,.2f})\n")
                f.write(f"Compromised Assets: {mean_metrics.get('compromised_assets', 0):.1f} (±{std_metrics.get('compromised_assets', 0):.1f})\n")
                f.write(f"Attack Success Rate: {mean_metrics.get('attack_success_rate', 0):.1f}% (±{std_metrics.get('attack_success_rate', 0):.1f}%)\n")
                
                # Threat Intelligence specific metrics
                if strategy_name in ['Threat Intelligence', 'RL Defender', 'Hybrid Defender']:
                    f.write(f"\nThreat Intelligence Metrics:\n")
                    f.write(f"  Observations Collected: {mean_metrics.get('observations_collected', 0):.1f}\n")
                    f.write(f"  Learning Adaptations: {mean_metrics.get('learning_adaptations', 0):.1f}\n")
                    f.write(f"  Threat Level Changes: {mean_metrics.get('threat_level_changes', 0):.1f}\n")
                    f.write(f"  Detection Coverage: {mean_metrics.get('detection_coverage', 0):.1f}%\n")
                
                # Hybrid specific metrics
                if strategy_name == 'Hybrid Defender':
                    f.write(f"\nHybrid Strategy Metrics:\n")
                    f.write(f"  Adaptations: {mean_metrics.get('hybrid_adaptations', 0):.1f}\n")
                    f.write(f"  Final TI Weight: {mean_metrics.get('final_ti_weight', 0):.3f}\n")
                    f.write(f"  Final RL Weight: {mean_metrics.get('final_rl_weight', 0):.3f}\n")
                    f.write(f"  Average Confidence: {mean_metrics.get('average_confidence', 0):.3f}\n")
                    f.write(f"  TI Performance: {mean_metrics.get('ti_performance', 0):.3f}\n")
                    f.write(f"  RL Performance: {mean_metrics.get('rl_performance', 0):.3f}\n")
                    f.write(f"  Hybrid Performance: {mean_metrics.get('hybrid_performance', 0):.3f}\n")
            
            # Best strategy identification
            f.write("\n" + "=" * 80 + "\n")
            f.write("BEST PERFORMING STRATEGIES\n")
            f.write("-" * 40 + "\n")
            
            strategies = list(aggregated_results.keys())
            
            # By value preserved
            best_value = max(strategies, key=lambda s: aggregated_results[s]['mean_metrics'].get('value_preserved', 0))
            f.write(f"By Value Preserved: {best_value} (${aggregated_results[best_value]['mean_metrics']['value_preserved']:,.2f})\n")
            
            # By ROI
            best_roi = max(strategies, key=lambda s: aggregated_results[s]['mean_metrics'].get('roi', 0))
            f.write(f"By ROI: {best_roi} ({aggregated_results[best_roi]['mean_metrics']['roi']:.1f}%)\n")
            
            # By fewest compromised assets
            least_compromised = min(strategies, key=lambda s: aggregated_results[s]['mean_metrics'].get('compromised_assets', float('inf')))
            f.write(f"By Security (Fewest Compromised): {least_compromised} ({aggregated_results[least_compromised]['mean_metrics']['compromised_assets']:.1f} assets)\n")
        
        print(f"Saved summary report to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Visualize APT3 RTU simulation results")
    parser.add_argument("--input-file", type=str, required=True, help="Path to aggregated results JSON file")
    parser.add_argument("--output-dir", type=str, default="apt3_simulation_results", help="Output directory for visualizations")
    parser.add_argument("--num-trials", type=int, default=100, help="Number of trials run in the simulation")
    
    args = parser.parse_args()
    
    # Load results
    with open(args.input_file, 'r') as f:
        aggregated_results = json.load(f)
    
    # Create visualizer and generate plots
    visualizer = APT3SimulationVisualizer(args.output_dir)
    visualizer.visualize_results(aggregated_results, args.num_trials)
    visualizer.save_summary_report(aggregated_results)
    
    print("\nVisualization complete!")


if __name__ == "__main__":
    main()