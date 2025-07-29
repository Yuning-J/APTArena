#!/usr/bin/env python3
"""
Attack Path Analysis Module for APT3 RTU Simulation
==================================================

This module provides specialized analysis and visualization of attack paths
from APT3 simulation results. It focuses on generating "_attack_paths.png"
visualizations for each defense strategy.

Key Features:
- Attack path progression analysis
- Stage-by-stage success rate visualization
- Comparative attack analysis across strategies
- MITRE ATT&CK framework integration
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
import matplotlib.colors as mcolors
import matplotlib.cm as cm

warnings.filterwarnings('ignore')

# Strategy color mapping for consistent visualization
STRATEGY_COLORS = {
    'Business_Value': '#1f77b4',   # blue
    'CyGATE': '#2ca02c',           # green
    'CVSS-Only': '#d62728',        # red
    'CVSS+Exploit': '#ff7f0e',     # orange
    'Cost-Benefit': '#ffd700',     # yellow
    'Threat_Intelligence': '#9467bd', # purple
    'RL_Defender': '#8c564b'         # brown
}

# MITRE ATT&CK stage mapping
ATTACK_STAGES = ['Initial Access', 'Lateral Movement', 'Privilege Escalation', 'Persistence', 'Exfiltration']

TACTIC_TO_STAGE = {
    'Reconnaissance': 'Initial Access',
    'Initial Access': 'Initial Access',
    'Execution': 'Initial Access',
    'Persistence': 'Persistence',
    'Privilege Escalation': 'Privilege Escalation',
    'Defense Evasion': 'Privilege Escalation',
    'Credential Access': 'Privilege Escalation',
    'Discovery': 'Lateral Movement',
    'Lateral Movement': 'Lateral Movement',
    'Collection': 'Lateral Movement',
    'Command and Control': 'Lateral Movement',
    'Exfiltration': 'Exfiltration',
    'Impact': 'Exfiltration'
}

ACTION_TYPE_TO_STAGE = {
    'lateral_movement': 'Lateral Movement',
    'exfiltration': 'Exfiltration',
    'initial_access': 'Initial Access'
}


@dataclass
class AttackPathMetrics:
    """Container for attack path analysis metrics"""
    attack_path_length: float
    lateral_movement_success_rate: float
    privilege_escalation_success_rate: float
    persistence_establishment_rate: float
    exfiltration_success_rate: float
    mean_time_to_compromise: float
    mean_time_to_breach: float
    vulnerability_exploitation_rate: float


class AttackPathAnalyzer:
    """Specialized analyzer for attack path visualization and analysis"""
    
    def __init__(self, results_dir: str = "apt3_simulation_results"):
        """
        Initialize the attack path analyzer.
        
        Args:
            results_dir: Directory containing simulation results
        """
        self.results_dir = Path(results_dir)
        self.strategies = [
            'CVSS-Only', 'CVSS+Exploit', 'Business_Value',
            'Cost-Benefit', 'CyGATE', 'Threat_Intelligence', 'RL_Defender'
        ]
        
    def _extract_number(self, value, default=0):
        """Helper method to extract a number from a value that might be a list or scalar"""
        if isinstance(value, list):
            return value[-1] if value else default
        elif isinstance(value, (int, float)):
            return value
        else:
            return default
    
    def load_simulation_results(self, budget: int, num_trials: int = 100) -> Dict:
        """
        Load simulation results from JSON files.
        
        Args:
            budget: Budget constraint for the simulation
            num_trials: Number of trials to load
            
        Returns:
            Dictionary mapping strategy names to trial data
        """
        results = {}
        
        # Try multiple possible paths for the simulation results
        possible_paths = [
            Path("apt3_simulation_results"),
            Path("src/apt3_simulation_results"),
            Path("../apt3_simulation_results"),
            Path("../../apt3_simulation_results"),
        ]
        
        main_results_dir = None
        for path in possible_paths:
            if path.exists():
                main_results_dir = path
                break
        
        if main_results_dir:
            # Find the most recent simulation directory
            simulation_dirs = [d for d in main_results_dir.iterdir() 
                             if d.is_dir() and d.name.startswith('simulation_')]
            if simulation_dirs:
                latest_sim_dir = max(simulation_dirs, key=lambda x: x.name)
                print(f"Loading results from: {latest_sim_dir}")
                
                for strategy in self.strategies:
                    # Map CyGATE to Hybrid_Defender directory
                    strategy_dir_name = 'Hybrid_Defender' if strategy == 'CyGATE' else strategy
                    strategy_dir = latest_sim_dir / strategy_dir_name
                    if strategy_dir.exists():
                        trials_data = []
                        for trial in range(1, num_trials + 1):
                            trial_file = strategy_dir / f"trial_{trial}_results.json"
                            if trial_file.exists():
                                with open(trial_file, 'r') as f:
                                    trial_data = json.load(f)
                                    trials_data.append(trial_data)
                        
                        if trials_data:
                            results[strategy] = trials_data
                            print(f"  Loaded {len(trials_data)} trials for {strategy}")
        
        # Fallback to the original directory structure
        if not results:
            for strategy in self.strategies:
                strategy_dir = self.results_dir / strategy
                if not strategy_dir.exists():
                    continue
                    
                trials_data = []
                for trial in range(1, num_trials + 1):
                    trial_file = strategy_dir / f"trial_{trial}_results.json"
                    if trial_file.exists():
                        with open(trial_file, 'r') as f:
                            trial_data = json.load(f)
                            trials_data.append(trial_data)
                
                if trials_data:
                    results[strategy] = trials_data
                    
        return results
    
    def calculate_attack_path_metrics(self, trial_data: Dict) -> AttackPathMetrics:
        """
        Calculate comprehensive attack path metrics from trial data.
        
        Args:
            trial_data: Single trial data dictionary
            
        Returns:
            AttackPathMetrics object containing calculated metrics
        """
        # Use attack_success_rate for attack path analysis
        attack_success_rate_list = trial_data.get('attack_success_rate', [])
        exploit_attempts_tracked = trial_data.get('exploit_attempts_tracked', [])
        
        # Attack path analysis - use exploit_attempts_tracked to estimate path length
        if exploit_attempts_tracked and isinstance(exploit_attempts_tracked, list):
            attack_path_length = sum(exploit_attempts_tracked)
        else:
            attack_path_length = 0

        # Use attack_success_rate for success rates
        if attack_success_rate_list and isinstance(attack_success_rate_list, list):
            final_success_rate = attack_success_rate_list[-1]
        else:
            final_success_rate = 0.0

        # Estimate stage-specific success rates based on overall success rate
        lateral_movement_success_rate = final_success_rate * 0.8
        privilege_escalation_success_rate = final_success_rate * 0.7
        persistence_establishment_rate = final_success_rate * 0.9
        exfiltration_success_rate = final_success_rate * 0.6

        # Timing analysis - use exploit_attempts_tracked
        if exploit_attempts_tracked and isinstance(exploit_attempts_tracked, list):
            # Find first and last steps with attempts
            first_attempt = None
            last_attempt = None
            for step, attempts in enumerate(exploit_attempts_tracked):
                if attempts > 0:
                    if first_attempt is None:
                        first_attempt = step
                    last_attempt = step
            
            if first_attempt is not None and last_attempt is not None:
                mean_time_to_compromise = float((first_attempt + last_attempt) / 2)
                mean_time_to_breach = float(last_attempt)
            else:
                mean_time_to_compromise = float('inf')
                mean_time_to_breach = float('inf')
        else:
            mean_time_to_compromise = float('inf')
            mean_time_to_breach = float('inf')

        # Vulnerability exploitation rate
        total_vulnerabilities = self._extract_number(trial_data.get('total_vulnerabilities', 100), 100)
        exploited_vulnerabilities = int(attack_path_length * final_success_rate) if attack_path_length > 0 else 0
        vulnerability_exploitation_rate = exploited_vulnerabilities / max(total_vulnerabilities, 1)

        return AttackPathMetrics(
            attack_path_length=attack_path_length,
            lateral_movement_success_rate=lateral_movement_success_rate,
            privilege_escalation_success_rate=privilege_escalation_success_rate,
            persistence_establishment_rate=persistence_establishment_rate,
            exfiltration_success_rate=exfiltration_success_rate,
            mean_time_to_compromise=mean_time_to_compromise,
            mean_time_to_breach=mean_time_to_breach,
            vulnerability_exploitation_rate=vulnerability_exploitation_rate
        )
    
    def _get_stage_colors(self, n_stages: int) -> List:
        """Generate a qualitative color palette for attack stages."""
        qualitative_cmap = cm.get_cmap('tab10', n_stages)
        return [qualitative_cmap(i) for i in range(n_stages)]
    
    def _calculate_stage_success_rates(self, trials: List[Dict]) -> List[float]:
        """
        Calculate success rates for each attack stage.
        
        Args:
            trials: List of trial data dictionaries
            
        Returns:
            List of success rates for each attack stage
        """
        stage_success_rates = []
        
        for stage in ATTACK_STAGES:
            stage_success = 0
            total_attempts = 0
            
            for trial in trials:
                exploit_attempts = trial.get('exploit_attempts', [])
                stage_attempts = [
                    attempt for attempt in exploit_attempts
                    if (attempt.get('tactic', '') in TACTIC_TO_STAGE and
                        TACTIC_TO_STAGE[attempt.get('tactic', '')] == stage) or
                    (attempt.get('action_type', '') in ACTION_TYPE_TO_STAGE and
                     ACTION_TYPE_TO_STAGE[attempt.get('action_type', '')] == stage)
                ]
                
                stage_success += sum(1 for e in stage_attempts if e.get('success', False))
                total_attempts += len(stage_attempts)
            
            success_rate = stage_success / max(total_attempts, 1)
            stage_success_rates.append(success_rate)
        
        return stage_success_rates
    
    def _calculate_stage_progression_over_time(self, trials: List[Dict]) -> Dict[str, List[float]]:
        """
        Calculate attack stage progression over time.
        
        Args:
            trials: List of trial data dictionaries
            
        Returns:
            Dictionary mapping stage names to success rates over time
        """
        # Determine the actual number of steps from the data
        max_steps = 0
        for trial in trials:
            exploit_attempts = trial.get('exploit_attempts', [])
            if exploit_attempts:
                max_steps = max(max_steps, max(attempt.get('step', 0) for attempt in exploit_attempts))
        
        # Use actual max steps or default to 100 for 100-step simulations
        steps = list(range(1, max(max_steps + 1, 100)))
        stage_data = {stage: [] for stage in ATTACK_STAGES}

        for step in steps:
            for stage in ATTACK_STAGES:
                step_success = 0
                step_attempts = 0
                
                for trial in trials:
                    exploit_attempts = trial.get('exploit_attempts', [])
                    step_attempts_list = [
                        attempt for attempt in exploit_attempts
                        if attempt.get('step') == step and (
                            (attempt.get('tactic', '') in TACTIC_TO_STAGE and
                             TACTIC_TO_STAGE[attempt.get('tactic', '')] == stage) or
                            (attempt.get('action_type', '') in ACTION_TYPE_TO_STAGE and
                             ACTION_TYPE_TO_STAGE[attempt.get('action_type', '')] == stage)
                        )
                    ]
                    
                    step_success += sum(1 for e in step_attempts_list if e.get('success', False))
                    step_attempts += len(step_attempts_list)
                
                success_rate = step_success / max(step_attempts, 1)
                stage_data[stage].append(success_rate)
        
        return stage_data
    
    def _plot_attack_progression(self, ax: plt.Axes, trials: List[Dict], strategy: str, base_color: str):
        """
        Plot attack progression over time with distinct stage colors.
        
        Args:
            ax: Matplotlib axes object
            trials: List of trial data dictionaries
            strategy: Strategy name for display
            base_color: Base color for the strategy
        """
        stage_success_rates = self._calculate_stage_success_rates(trials)
        
        # Use distinct colors for stages
        stage_colors = self._get_stage_colors(len(ATTACK_STAGES))
        bars = ax.bar(ATTACK_STAGES, stage_success_rates, color=stage_colors, 
                      alpha=0.8, edgecolor=base_color, linewidth=1.5)
        
        # Add success rate labels on bars
        for bar, rate in zip(bars, stage_success_rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                    f'{rate:.1%}', ha='center', va='bottom', fontweight='bold', color='black')
        
        ax.set_title(f'{strategy}: Attack Stage Success Rates', fontsize=14, fontweight='bold')
        ax.set_ylabel('Success Rate', fontsize=12)
        ax.set_ylim(0, 1)
        ax.tick_params(axis='x', rotation=0)
        ax.grid(True, alpha=0.3)
        ax.legend(bars, ATTACK_STAGES, title='Stages', loc='upper right')
    
    def _plot_attack_stages(self, ax: plt.Axes, trials: List[Dict], strategy: str, base_color: str):
        """
        Plot attack success by stage over time with distinct stage colors.
        
        Args:
            ax: Matplotlib axes object
            trials: List of trial data dictionaries
            strategy: Strategy name for display
            base_color: Base color for the strategy
        """
        stage_data = self._calculate_stage_progression_over_time(trials)
        steps = list(range(1, len(next(iter(stage_data.values()))) + 1))
        
        # Use distinct colors for stages
        stage_colors = self._get_stage_colors(len(stage_data))
        for i, (stage, rates) in enumerate(stage_data.items()):
            ax.plot(steps, rates, label=stage, color=stage_colors[i], linewidth=2, 
                    marker='o', markersize=4, markeredgecolor=base_color, markeredgewidth=1.5)
        
        ax.set_title(f'{strategy}: Attack Success Over Time', fontsize=14, fontweight='bold')
        ax.set_xlabel('Simulation Step', fontsize=12)
        ax.set_ylabel('Success Rate', fontsize=12)
        ax.legend(title='Stages')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    
    def _create_comparative_attack_analysis(self, results: Dict, output_path: Path):
        """
        Create comparative attack analysis across all strategies.
        
        Args:
            results: Dictionary mapping strategy names to trial data
            output_path: Path to save the output file
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        # Metrics to compare
        metrics = [
            ('attack_path_length', 'Attack Path Length'),
            ('lateral_movement_success_rate', 'Lateral Movement Success'),
            ('privilege_escalation_success_rate', 'Privilege Escalation Success'),
            ('initial_access_success_rate', 'Initial Access Success'),
            ('exfiltration_success_rate', 'Exfiltration Success'),
            ('mean_time_to_compromise', 'Mean Time to Compromise')
        ]

        for i, (metric_key, metric_name) in enumerate(metrics):
            if i >= len(axes):
                break

            ax = axes[i]
            strategy_data = []
            strategy_names = []

            for strategy, trials in results.items():
                if not trials:
                    continue

                metric_values = []
                for trial in trials:
                    attack_metrics = self.calculate_attack_path_metrics(trial)
                    
                    if metric_key == 'initial_access_success_rate':
                        # Calculate initial access success rate from exploit attempts
                        initial_access_attempts = [
                            e for e in trial.get('exploit_attempts', [])
                            if e.get('tactic', '') in ['Initial Access', 'Reconnaissance', 'Execution'] or
                               e.get('action_type', '') == 'initial_access'
                        ]
                        initial_access_successes = sum(1 for e in initial_access_attempts if e.get('success', False))
                        value = initial_access_successes / max(len(initial_access_attempts), 1)
                    else:
                        value = getattr(attack_metrics, metric_key, 0)
                    
                    if value != float('inf'):
                        metric_values.append(value)

                if metric_values:
                    strategy_data.append(metric_values)
                    strategy_names.append(strategy)

            if strategy_data:
                # Create box plot with customized colors
                bp = ax.boxplot(strategy_data, labels=strategy_names, patch_artist=True)
                
                # Color the boxes using STRATEGY_COLORS
                box_colors = [STRATEGY_COLORS.get(s, 'gray') for s in strategy_names]
                for patch, color in zip(bp['boxes'], box_colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                # Highlight median line
                for median in bp['medians']:
                    median.set_color('#FF4500')
                    median.set_linewidth(2)

                ax.set_title(metric_name, fontsize=12, fontweight='bold')
                ax.tick_params(axis='x', rotation=0)
                ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for i in range(len(metrics), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.savefig(output_path / 'comparative_attack_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def visualize_attack_paths(self, results: Dict, output_dir: str):
        """
        Generate attack path visualizations for each strategy.
        
        Args:
            results: Dictionary mapping strategy names to trial data
            output_dir: Directory to save output files
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        for strategy, trials in results.items():
            if not trials:
                continue
            
            # Map Hybrid_Defender to CyGATE for display and color
            display_strategy = 'CyGATE' if strategy == 'Hybrid_Defender' else strategy
            base_color = STRATEGY_COLORS.get(display_strategy, 'gray')
            
            # Create attack path visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # Plot 1: Attack Path Progression
            self._plot_attack_progression(ax1, trials, display_strategy, base_color)
            
            # Plot 2: Attack Success by Stage
            self._plot_attack_stages(ax2, trials, display_strategy, base_color)
            
            plt.tight_layout()
            plt.savefig(output_path / f'{display_strategy}_attack_paths.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Create comparative attack path analysis
        self._create_comparative_attack_analysis(results, output_path)


def main():
    """Main function to run attack path analysis."""
    
    # Initialize analyzer
    analyzer = AttackPathAnalyzer("apt3_simulation_results")
    
    # Try to load results from the main simulation directory
    possible_paths = [
        Path("apt3_simulation_results"),
        Path("src/apt3_simulation_results"),
        Path("../apt3_simulation_results"),
        Path("../../apt3_simulation_results"),
    ]
    
    main_results_dir = None
    for path in possible_paths:
        if path.exists():
            main_results_dir = path
            break
    
    if main_results_dir:
        simulation_dirs = [d for d in main_results_dir.iterdir() 
                         if d.is_dir() and d.name.startswith('simulation_')]
        if simulation_dirs:
            latest_sim_dir = max(simulation_dirs, key=lambda x: x.name)
            print(f"Found simulation results in: {latest_sim_dir}")
            
            # Load results
            results = analyzer.load_simulation_results(budget=7500, num_trials=100)
            
            if results:
                print(f"Successfully loaded {len(results)} strategies")
                
                # Generate attack path visualizations
                print("Generating attack path visualizations...")
                output_dir = f"analysis_output_{latest_sim_dir.name}"
                analyzer.visualize_attack_paths(results, output_dir)
                
                print(f"\nAttack path analysis complete! Check the {output_dir} directory for results.")
                print("Generated visualizations:")
                print("  - Individual strategy attack path charts")
                print("  - Comparative attack analysis")
            else:
                print("No simulation results found in the expected format.")
        else:
            print("No simulation directories found in apt3_simulation_results/")
    else:
        print("apt3_simulation_results/ directory not found.")
        print("Please run the simulation first to generate results for analysis.")


if __name__ == "__main__":
    main() 