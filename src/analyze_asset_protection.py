#!/usr/bin/env python3
"""
Comprehensive Asset Protection Analysis
======================================

This script analyzes asset protection metrics across core defense strategies
including Business_Value, CyGATE (Hybrid_Defender), CVSS-Only, CVSS+Exploit, and Cost-Benefit.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import os
import matplotlib.colors as mcolors
import matplotlib.cm as cm

ALLOWED_STRATEGIES = [
    'CVSS-Only', 'CVSS+Exploit', 'Business_Value', 'Cost-Benefit', 'Hybrid_Defender'
]

# Consistent color scheme with simulation_analysis_systematic.py
STRATEGY_COLORS = {
    'CVSS-Only': '#d62728',        # red
    'CVSS+Exploit': '#ff7f0e',     # orange
    'Business_Value': '#1f77b4',   # blue
    'Cost-Benefit': '#ffd700',      # yellow
    'Hybrid_Defender': '#2ca02c',  # green (CyGATE)
    'CyGATE': '#2ca02c'           # green (alias)
}

def _get_shades(base_color, n):
    base_rgba = mcolors.to_rgba(base_color)
    cmap = cm.get_cmap('Greens') if base_color == '#2ca02c' else cm.get_cmap('Blues') if base_color == '#1f77b4' else cm.get_cmap('Oranges') if base_color == '#ff7f0e' else cm.get_cmap('Reds') if base_color == '#d62728' else cm.get_cmap('YlOrBr')
    return [cmap(0.4 + 0.5 * i / max(n-1,1)) for i in range(n)]

def load_all_trial_results(simulation_dir: str) -> Dict[str, List[Dict]]:
    """Load all trial results for all strategies"""
    results = {}
    strategies = [
        'CVSS-Only', 'CVSS+Exploit', 'Business_Value', 'Cost-Benefit', 'Hybrid_Defender'
    ]
    
    for strategy in strategies:
        strategy_dir = Path(simulation_dir) / strategy
        if not strategy_dir.exists():
            print(f"Strategy directory not found: {strategy_dir}")
            continue
            
        trials_data = []
        trial_files = list(strategy_dir.glob("trial_*_results.json"))
        
        for trial_file in trial_files:
            try:
                with open(trial_file, 'r') as f:
                    trial_data = json.load(f)
                    trials_data.append(trial_data)
            except Exception as e:
                print(f"Error loading {trial_file}: {e}")
        
        if trials_data:
            results[strategy] = trials_data
            print(f"Loaded {len(trials_data)} trials for {strategy}")
        else:
            print(f"No trials found for {strategy}")
    
    return results

def extract_asset_protection_metrics(trial_data: Dict) -> Dict:
    """Extract asset protection metrics from a single trial"""
    return {
        'protected_value': trial_data.get('protected_value', [0])[0] if isinstance(trial_data.get('protected_value'), list) else trial_data.get('protected_value', 0),
        'lost_value': trial_data.get('lost_value', [0])[0] if isinstance(trial_data.get('lost_value'), list) else trial_data.get('lost_value', 0),
        'final_compromised_assets': trial_data.get('final_compromised_assets', 0),
        'total_patch_cost': trial_data.get('total_patch_cost', 0),
        'total_patches': trial_data.get('total_patches', 0),
        'roi': trial_data.get('roi', 0),
        'compromised_assets_over_time': trial_data.get('compromised_assets', []),
        'protected_value_over_time': trial_data.get('step_metrics', {}).get('protected_value_over_time', []),
        'lost_value_over_time': trial_data.get('step_metrics', {}).get('lost_value_over_time', [])
    }

def calculate_asset_protection_statistics(all_results: Dict) -> pd.DataFrame:
    """Calculate comprehensive asset protection statistics"""
    stats_data = []
    
    for strategy, trials in all_results.items():
        strategy_stats = {
            'Strategy': strategy,
            'Total_Trials': len(trials),
            'Avg_Protected_Value': np.mean([extract_asset_protection_metrics(trial)['protected_value'] for trial in trials]),
            'Std_Protected_Value': np.std([extract_asset_protection_metrics(trial)['protected_value'] for trial in trials]),
            'Avg_Lost_Value': np.mean([extract_asset_protection_metrics(trial)['lost_value'] for trial in trials]),
            'Std_Lost_Value': np.std([extract_asset_protection_metrics(trial)['lost_value'] for trial in trials]),
            'Avg_Final_Compromised_Assets': np.mean([extract_asset_protection_metrics(trial)['final_compromised_assets'] for trial in trials]),
            'Std_Final_Compromised_Assets': np.std([extract_asset_protection_metrics(trial)['final_compromised_assets'] for trial in trials]),
            'Avg_Total_Patch_Cost': np.mean([extract_asset_protection_metrics(trial)['total_patch_cost'] for trial in trials]),
            'Avg_Total_Patches': np.mean([extract_asset_protection_metrics(trial)['total_patches'] for trial in trials]),
            'Avg_ROI': np.mean([extract_asset_protection_metrics(trial)['roi'] for trial in trials]),
            'Min_Protected_Value': np.min([extract_asset_protection_metrics(trial)['protected_value'] for trial in trials]),
            'Max_Protected_Value': np.max([extract_asset_protection_metrics(trial)['protected_value'] for trial in trials]),
            'Min_Compromised_Assets': np.min([extract_asset_protection_metrics(trial)['final_compromised_assets'] for trial in trials]),
            'Max_Compromised_Assets': np.max([extract_asset_protection_metrics(trial)['final_compromised_assets'] for trial in trials])
        }
        
        # Calculate asset protection rate (percentage of trials with 0 compromised assets)
        zero_compromise_trials = sum(1 for trial in trials if extract_asset_protection_metrics(trial)['final_compromised_assets'] == 0)
        strategy_stats['Asset_Protection_Rate'] = (zero_compromise_trials / len(trials)) * 100
        
        # Calculate average time to first compromise
        first_compromise_times = []
        for trial in trials:
            compromised_over_time = extract_asset_protection_metrics(trial)['compromised_assets_over_time']
            if compromised_over_time:
                for i, count in enumerate(compromised_over_time):
                    if count > 0:
                        first_compromise_times.append(i)
                        break
                else:
                    first_compromise_times.append(len(compromised_over_time))  # No compromise
        
        strategy_stats['Avg_Time_To_First_Compromise'] = np.mean(first_compromise_times) if first_compromise_times else 0
        
        stats_data.append(strategy_stats)
    
    return pd.DataFrame(stats_data)

def analyze_attack_progression(all_results: Dict) -> Dict[str, Dict]:
    """Analyze attack progression over time for each strategy"""
    progression_data = {}
    for strategy, trials in all_results.items():
        # Collect compromised assets over time for all trials
        all_compromised_over_time = []
        all_protected_over_time = []
        for trial in trials:
            metrics = extract_asset_protection_metrics(trial)
            if metrics['compromised_assets_over_time']:
                all_compromised_over_time.append(metrics['compromised_assets_over_time'])
            if metrics['protected_value_over_time']:
                all_protected_over_time.append(metrics['protected_value_over_time'])
        if all_compromised_over_time:
            # Calculate average progression
            max_steps = max(len(prog) for prog in all_compromised_over_time)
            avg_compromised_progression = []
            for step in range(max_steps):
                step_values = []
                for prog in all_compromised_over_time:
                    if step < len(prog):
                        step_values.append(prog[step])
                avg_compromised_progression.append(np.mean(step_values))
            display_name = 'CyGATE' if strategy == 'Hybrid_Defender' else strategy
            progression_data[display_name] = {
                'avg_compromised_progression': avg_compromised_progression,
                'max_compromised_assets': np.max([np.max(prog) for prog in all_compromised_over_time]),
                'final_compromised_assets': np.mean([prog[-1] if prog else 0 for prog in all_compromised_over_time])
            }
    return progression_data

def create_comprehensive_visualizations(stats_df: pd.DataFrame, progression_data: Dict, output_dir: str):
    """Create comprehensive visualizations of asset protection performance"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Comprehensive Asset Protection Analysis', fontsize=16, fontweight='bold')
    
    # Prepare strategy names
    strategies = [s if s != 'Hybrid_Defender' else 'CyGATE' for s in stats_df['Strategy']]
    display_strategies = [s.replace('_', '\n') for s in strategies]
    
    # Load all trial results for boxplots
    # Find the most recent simulation directory
    possible_dirs = [
        "apt3_simulation_results",
        "../apt3_simulation_results", 
        "src/apt3_simulation_results"
    ]
    simulation_dir = None
    for dir_path in possible_dirs:
        if os.path.exists(dir_path):
            sim_dirs = [d for d in os.listdir(dir_path) if d.startswith('simulation_')]
            if sim_dirs:
                latest_sim = max(sim_dirs)
                simulation_dir = os.path.join(dir_path, latest_sim)
                break
    all_results = load_all_trial_results(simulation_dir)
    
    # Helper to get per-trial values for a metric
    def get_metric_per_trial(metric):
        data = []
        for s in strategies:
            key = 'Hybrid_Defender' if s == 'CyGATE' else s
            if key in all_results:
                vals = [extract_asset_protection_metrics(trial)[metric] for trial in all_results[key]]
                data.append(vals)
            else:
                data.append([])
        return data
    
    # 1. Protected Value Boxplot
    protected_value_data = get_metric_per_trial('protected_value')
    box1 = axes[0, 0].boxplot(protected_value_data, patch_artist=True, tick_labels=display_strategies)
    for patch, s in zip(box1['boxes'], strategies):
        patch.set_facecolor(STRATEGY_COLORS.get(s, 'gray'))
    axes[0, 0].set_title('Protected Value by Strategy')
    axes[0, 0].set_ylabel('Protected Value ($)')
    axes[0, 0].tick_params(axis='x', rotation=0)
    
    # 2. Asset Protection Rate Bar Plot
    percent_zero_data = []
    for s in strategies:
        key = 'Hybrid_Defender' if s == 'CyGATE' else s
        if key in all_results:
            zero_compromise_trials = sum(1 for trial in all_results[key] if extract_asset_protection_metrics(trial)['final_compromised_assets'] == 0)
            percent_zero = (zero_compromise_trials / len(all_results[key])) * 100 if all_results[key] else 0
            percent_zero_data.append(percent_zero)
        else:
            percent_zero_data.append(0)
    bar_colors3 = [STRATEGY_COLORS.get(s, 'gray') for s in strategies]
    axes[0, 1].bar(display_strategies, percent_zero_data, alpha=0.7, color=bar_colors3)
    axes[0, 1].set_title('Asset Protection Rate (% Zero Compromise)')
    axes[0, 1].set_ylabel('Protection Rate (%)')
    axes[0, 1].tick_params(axis='x', rotation=0)
    
    # 3. Normalized ROI Boxplot
    roi_data = get_metric_per_trial('roi')
    max_roi = max([max(vals) if vals else 0 for vals in roi_data])
    norm_roi_data = [[(v / max_roi) * 99 if max_roi > 0 else 0 for v in vals] for vals in roi_data]
    box4 = axes[1, 0].boxplot(norm_roi_data, patch_artist=True, tick_labels=display_strategies)
    for patch, s in zip(box4['boxes'], strategies):
        patch.set_facecolor(STRATEGY_COLORS.get(s, 'gray'))
    axes[1, 0].set_title('Normalized ROI by Strategy')
    axes[1, 0].set_ylabel('Normalized ROI (%)')
    axes[1, 0].tick_params(axis='x', rotation=0)
    
    # 4. Average Attack Progression Over Time
    ax6 = axes[1, 1]
    for s, orig_s in zip(display_strategies, strategies):
        plot_name = 'CyGATE' if orig_s == 'Hybrid_Defender' else orig_s
        if plot_name in progression_data:
            progression = progression_data[plot_name]['avg_compromised_progression']
            steps = range(len(progression))
            base_color = STRATEGY_COLORS.get(plot_name, 'gray')
            ax6.plot(steps, progression, marker='o', label=s, linewidth=2, markersize=4, color=base_color)
    ax6.set_title('Average Attack Progression Over Time')
    ax6.set_xlabel('Simulation Steps')
    ax6.set_ylabel('Average Compromised Assets')
    ax6.legend(loc='upper right', fontsize=10)
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comprehensive_asset_protection_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def generate_detailed_report(stats_df: pd.DataFrame, progression_data: Dict, output_dir: str):
    """Generate a detailed text report of the analysis"""
    report_path = os.path.join(output_dir, 'asset_protection_analysis_report.md')
    
    with open(report_path, 'w') as f:
        f.write("# Comprehensive Asset Protection Analysis Report\n\n")
        
        f.write("## Executive Summary\n\n")
        
        # Find best performers
        best_protected = stats_df.loc[stats_df['Avg_Protected_Value'].idxmax()]
        best_asset_protection = stats_df.loc[stats_df['Asset_Protection_Rate'].idxmax()]
        best_roi = stats_df.loc[stats_df['Avg_ROI'].idxmax()]
        lowest_compromised = stats_df.loc[stats_df['Avg_Final_Compromised_Assets'].idxmin()]
        
        f.write(f"**Best Protected Value**: {best_protected['Strategy']} (${best_protected['Avg_Protected_Value']:,.0f})\n")
        f.write(f"**Best Asset Protection Rate**: {best_asset_protection['Strategy']} ({best_asset_protection['Asset_Protection_Rate']:.1f}%)\n")
        f.write(f"**Best ROI**: {best_roi['Strategy']} ({best_roi['Avg_ROI']:.1f}%)\n")
        f.write(f"**Lowest Compromised Assets**: {lowest_compromised['Strategy']} ({lowest_compromised['Avg_Final_Compromised_Assets']:.1f})\n\n")
        
        f.write("## Detailed Strategy Comparison\n\n")
        
        # Sort by protected value for ranking
        sorted_df = stats_df.sort_values(by='Avg_Protected_Value', ascending=False)
        
        f.write("### Strategy Rankings by Protected Value\n\n")
        f.write("| Rank | Strategy | Avg Protected Value | Avg Compromised Assets | Protection Rate | ROI |\n")
        f.write("|------|----------|-------------------|----------------------|----------------|-----|\n")
        
        for i, (_, row) in enumerate(sorted_df.iterrows(), 1):
            f.write(f"| {i} | {row['Strategy']} | ${row['Avg_Protected_Value']:,.0f} | {row['Avg_Final_Compromised_Assets']:.1f} | {row['Asset_Protection_Rate']:.1f}% | {row['Avg_ROI']:.1f}% |\n")
        
        f.write("\n## Key Insights\n\n")
        
        # Analyze CyGATE vs Business_Value
        cygate_row = stats_df[stats_df['Strategy'] == 'Hybrid_Defender']
        business_value_row = stats_df[stats_df['Strategy'] == 'Business_Value']
        
        if not cygate_row.empty and not business_value_row.empty:
            f.write("### CyGATE vs Business_Value Analysis\n\n")
            
            cygate = cygate_row.iloc[0]
            business_value = business_value_row.iloc[0]
            
            f.write(f"**Protected Value**: Business_Value (${business_value['Avg_Protected_Value']:,.0f}) vs CyGATE (${cygate['Avg_Protected_Value']:,.0f})\n")
            f.write(f"**Compromised Assets**: Business_Value ({business_value['Avg_Final_Compromised_Assets']:.1f}) vs CyGATE ({cygate['Avg_Final_Compromised_Assets']:.1f})\n")
            f.write(f"**Protection Rate**: Business_Value ({business_value['Asset_Protection_Rate']:.1f}%) vs CyGATE ({cygate['Asset_Protection_Rate']:.1f}%)\n")
            f.write(f"**ROI**: Business_Value ({business_value['Avg_ROI']:.1f}%) vs CyGATE ({cygate['Avg_ROI']:.1f}%)\n\n")
            
            if business_value['Avg_Protected_Value'] > cygate['Avg_Protected_Value']:
                f.write("**Conclusion**: Business_Value strategy provides better asset protection in terms of protected value.\n\n")
            else:
                f.write("**Conclusion**: CyGATE strategy provides better asset protection in terms of protected value.\n\n")
        
        f.write("## Recommendations\n\n")
        
        # Generate recommendations based on different criteria
        f.write("### For Maximum Asset Protection\n")
        best_protection = stats_df.loc[stats_df['Asset_Protection_Rate'].idxmax()]
        f.write(f"- **Primary**: {best_protection['Strategy']} ({best_protection['Asset_Protection_Rate']:.1f}% protection rate)\n")
        
        f.write("\n### For Maximum Protected Value\n")
        best_value = stats_df.loc[stats_df['Avg_Protected_Value'].idxmax()]
        f.write(f"- **Primary**: {best_value['Strategy']} (${best_value['Avg_Protected_Value']:,.0f} average protected value)\n")
        
        f.write("\n### For Best ROI\n")
        best_roi_strategy = stats_df.loc[stats_df['Avg_ROI'].idxmax()]
        f.write(f"- **Primary**: {best_roi_strategy['Strategy']} ({best_roi_strategy['Avg_ROI']:.1f}% ROI)\n")
        
        f.write("\n### For Balanced Performance\n")
        # Calculate a composite score
        stats_df['Composite_Score'] = (
            stats_df['Avg_Protected_Value'] / stats_df['Avg_Protected_Value'].max() * 0.4 +
            (1 - stats_df['Avg_Final_Compromised_Assets'] / stats_df['Avg_Final_Compromised_Assets'].max()) * 0.3 +
            stats_df['Asset_Protection_Rate'] / stats_df['Asset_Protection_Rate'].max() * 0.2 +
            stats_df['Avg_ROI'] / stats_df['Avg_ROI'].max() * 0.1
        )
        
        best_balanced = stats_df.loc[stats_df['Composite_Score'].idxmax()]
        f.write(f"- **Primary**: {best_balanced['Strategy']} (Composite Score: {best_balanced['Composite_Score']:.3f})\n")
    
    print(f"Detailed report saved to: {report_path}")

def filter_allowed_strategies(df: pd.DataFrame) -> pd.DataFrame:
    return df[df['Strategy'].isin(ALLOWED_STRATEGIES)].reset_index(drop=True)

def main():
    """Main analysis function"""
    print("Starting Comprehensive Asset Protection Analysis...")
    
    # Find the most recent simulation directory
    possible_dirs = [
        "apt3_simulation_results",
        "../apt3_simulation_results", 
        "src/apt3_simulation_results"
    ]
    
    simulation_dir = None
    for dir_path in possible_dirs:
        if os.path.exists(dir_path):
            sim_dirs = [d for d in os.listdir(dir_path) if d.startswith('simulation_')]
            if sim_dirs:
                latest_sim = max(sim_dirs)
                simulation_dir = os.path.join(dir_path, latest_sim)
                break
    
    if not simulation_dir:
        print("No simulation directory found!")
        return
    
    print(f"Analyzing results from: {simulation_dir}")
    
    # Load all trial results
    all_results = load_all_trial_results(simulation_dir)
    
    if not all_results:
        print("No results found!")
        return
    
    # Calculate statistics
    print("Calculating asset protection statistics...")
    stats_df = calculate_asset_protection_statistics(all_results)
    stats_df = filter_allowed_strategies(stats_df)
    assert isinstance(stats_df, pd.DataFrame)
    
    # Analyze attack progression
    print("Analyzing attack progression...")
    progression_data = analyze_attack_progression(all_results)
    
    # Create output directory
    output_dir = "asset_protection_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save statistics to CSV
    stats_df.to_csv(os.path.join(output_dir, 'asset_protection_statistics.csv'), index=False)
    print(f"Statistics saved to: {os.path.join(output_dir, 'asset_protection_statistics.csv')}")
    
    # Display summary statistics
    print("\n" + "="*80)
    print("ASSET PROTECTION ANALYSIS SUMMARY")
    print("="*80)
    
    print("\nStrategy Rankings by Protected Value:")
    sorted_df = stats_df.sort_values(by='Avg_Protected_Value', ascending=False)
    for i, (_, row) in enumerate(sorted_df.iterrows(), 1):
        print(f"{i}. {row['Strategy']}: ${row['Avg_Protected_Value']:,.0f} protected, {row['Avg_Final_Compromised_Assets']:.1f} compromised, {row['Asset_Protection_Rate']:.1f}% protection rate")
    
    print("\nStrategy Rankings by Asset Protection Rate:")
    sorted_by_protection = stats_df.sort_values(by='Asset_Protection_Rate', ascending=False)
    for i, (_, row) in enumerate(sorted_by_protection.iterrows(), 1):
        print(f"{i}. {row['Strategy']}: {row['Asset_Protection_Rate']:.1f}% protection rate, ${row['Avg_Protected_Value']:,.0f} protected")
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_comprehensive_visualizations(stats_df, progression_data, output_dir)
    
    # Generate detailed report
    print("Generating detailed report...")
    generate_detailed_report(stats_df, progression_data, output_dir)
    
    print(f"\nAnalysis complete! Results saved to: {output_dir}")

if __name__ == "__main__":
    main() 