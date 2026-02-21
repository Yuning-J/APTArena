#!/usr/bin/env python3
"""
Comprehensive analysis of APT3 simulation results comparing different defense strategies.
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

class SimulationAnalyzer:
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.strategies = [
            'CVSS-Only', 'CVSS+Exploit', 'Business_Value', 'Cost-Benefit',
            'Threat_Intelligence', 'RL_Defender', 'Hybrid_Defender'
        ]
        self.results = {}
        self.summary_stats = {}
        self.strategic_metrics = {}

    @staticmethod
    def _normalize_trial(trial: Dict) -> Dict:
        """Convert trial to flat scalars for mean/std (handles list and nested fields)."""
        out = {}
        # Value-like fields: often stored as single-element lists
        for key in ('value_preserved', 'protected_value', 'lost_value', 'unpatched_critical'):
            v = trial.get(key)
            if v is not None:
                out[key] = v[0] if isinstance(v, (list, tuple)) and v else (v if not isinstance(v, (list, tuple)) else np.nan)
            else:
                out[key] = np.nan
        # Scalars
        for key in ('roi', 'total_patch_cost', 'total_patches', 'total_patch_time'):
            v = trial.get(key)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                out[key] = float(v) if not isinstance(v, (list, tuple)) else (float(v[0]) if v else np.nan)
            else:
                out[key] = np.nan
        # compromised_assets: use final count (final_compromised_assets or last of list)
        ca = trial.get('compromised_assets')
        fc = trial.get('final_compromised_assets')
        if fc is not None and not (isinstance(fc, float) and np.isnan(fc)):
            out['compromised_assets'] = float(fc)
        elif ca is not None and isinstance(ca, (list, tuple)) and len(ca) > 0:
            out['compromised_assets'] = float(ca[-1])
        else:
            out['compromised_assets'] = np.nan
        # Nested metrics (summary.json often omits these; derive from exploit_attempts when possible)
        am = trial.get('attacker_metrics') or {}
        dm = trial.get('detection_metrics') or {}
        apt = trial.get('apt3_metrics') or {}
        attempts = trial.get('exploit_attempts') or []
        n_success = sum(1 for a in attempts if a.get('success') is True)
        n_attempts = len(attempts)
        out['attack_success_rate'] = (100.0 * n_success / n_attempts) if n_attempts else am.get('attack_success_rate', np.nan)
        if np.isnan(out['attack_success_rate']) and am:
            out['attack_success_rate'] = am.get('attack_success_rate', np.nan)
        out['detection_coverage'] = dm.get('detection_coverage', np.nan)
        out['avg_time_to_detection'] = dm.get('avg_time_to_detection', np.nan)
        out['attack_disruption_rate'] = dm.get('attack_disruption_rate', np.nan)
        out['time_to_rtu_compromise'] = np.nan  # aggregate elsewhere if needed
        out['spearphishing_success_rate'] = apt.get('spearphishing_success_rate', np.nan)
        out['credential_harvesting_count'] = apt.get('credential_harvesting_count', np.nan)
        # Canonical KPIs (same definitions/scales as legacy table; from incident timestamps)
        canonical = trial.get('canonical_run_summary') or {}
        out['incidents_total'] = canonical.get('incidents_total', np.nan)
        out['canonical_mttd_steps'] = canonical.get('mttd_steps', np.nan)
        out['canonical_mttr_steps'] = canonical.get('mttr_steps', np.nan)
        out['protected_asset_ratio_pct'] = canonical.get('protected_asset_ratio_pct', np.nan)
        return out

    def load_results(self):
        """Load simulation results: prefer simulation_summary.json, else per-strategy trial files."""
        print("Loading simulation results...")

        strategic_metrics_file = self.results_dir / "strategic_metrics.json"
        if strategic_metrics_file.exists():
            try:
                with open(strategic_metrics_file, 'r') as f:
                    self.strategic_metrics = json.load(f)
            except Exception as e:
                print(f"Note: could not load strategic_metrics.json: {e}")

        summary_file = self.results_dir / "simulation_summary.json"
        if summary_file.exists():
            try:
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                for strategy_name, strategy_data in summary.items():
                    if not isinstance(strategy_data, dict):
                        continue
                    trials_raw = strategy_data.get("trials", [])
                    if not trials_raw:
                        continue
                    trials_normalized = [self._normalize_trial(t) for t in trials_raw]
                    self.results[strategy_name] = {'trials': trials_normalized, 'summary': strategy_data.get('summary', {})}
                print(f"Loaded results from simulation_summary.json for {len(self.results)} strategies")
                return
            except Exception as e:
                print(f"Could not load simulation_summary.json: {e}, falling back to strategy directories")

        for strategy in self.strategies:
            strategy_dir = self.results_dir / strategy
            if strategy_dir.exists():
                data = self._load_strategy_results(strategy_dir)
                if data['trials']:
                    data['trials'] = [self._normalize_trial(t) for t in data['trials']]
                self.results[strategy] = data

        print(f"Loaded results for {len(self.results)} strategies")
    
    def _load_strategy_results(self, strategy_dir: Path) -> Dict:
        """Load results for a specific strategy."""
        results = {
            'trials': [],
            'summary': {}
        }
        
        # Load trials
        for trial_file in strategy_dir.glob("trial_*_results.json"):
            try:
                with open(trial_file, 'r') as f:
                    trial_data = json.load(f)
                    results['trials'].append(trial_data)
            except Exception as e:
                print(f"Error loading {trial_file}: {e}")
        
        # Load summary if available
        summary_file = strategy_dir / f"{strategy_dir.name}_trials.json"
        if summary_file.exists():
            try:
                with open(summary_file, 'r') as f:
                    results['summary'] = json.load(f)
            except Exception as e:
                print(f"Error loading summary {summary_file}: {e}")
        
        return results
    
    def calculate_performance_metrics(self):
        """Calculate comprehensive performance metrics for each strategy."""
        print("Calculating performance metrics...")
        
        for strategy_name, strategy_data in self.results.items():
            if not strategy_data['trials']:
                continue
                
            metrics = self._calculate_strategy_metrics(strategy_name, strategy_data['trials'])
            self.summary_stats[strategy_name] = metrics
    
    def _calculate_strategy_metrics(self, strategy_name: str, trials: List[Dict]) -> Dict:
        """Calculate metrics for a specific strategy."""
        metrics = {
            'roi': [],
            'value_preserved': [],
            'protected_value': [],
            'lost_value': [],
            'total_patches': [],
            'total_patch_cost': [],
            'compromised_assets': [],
            'attack_success_rate': [],
            'unpatched_critical': [],
            'detection_coverage': [],
            'avg_time_to_detection': [],
            'attack_disruption_rate': [],
            'time_to_rtu_compromise': [],
            'spearphishing_success_rate': [],
            'credential_harvesting_count': [],
            'incidents_total': [],
            'canonical_mttd_steps': [],
            'canonical_mttr_steps': [],
            'protected_asset_ratio_pct': []
        }
        
        for trial in trials:
            # Extract metrics from trial data (trials may be normalized with flat scalars)
            for metric in metrics.keys():
                value = trial.get(metric)
                if value is not None and not (isinstance(value, float) and np.isnan(value)):
                    metrics[metric].append(value)
        
        # Calculate summary statistics
        summary = {}
        for metric, values in metrics.items():
            if values:
                summary[f'{metric}_mean'] = np.mean(values)
                summary[f'{metric}_std'] = np.std(values)
                summary[f'{metric}_min'] = np.min(values)
                summary[f'{metric}_max'] = np.max(values)
                summary[f'{metric}_median'] = np.median(values)
            else:
                summary[f'{metric}_mean'] = 0
                summary[f'{metric}_std'] = 0
                summary[f'{metric}_min'] = 0
                summary[f'{metric}_max'] = 0
                summary[f'{metric}_median'] = 0
        
        return summary
    
    def create_performance_comparison_table(self) -> pd.DataFrame:
        """Create a comprehensive comparison table. Include MTTR/MTTD/Incidents/Protected % when kpi_aggregates.json exists."""
        print("Creating performance comparison table...")

        kpi_file = self.results_dir / "kpi_aggregates.json"
        kpi_operational = {}
        if kpi_file.exists():
            try:
                with open(kpi_file, "r") as f:
                    kpi_data = json.load(f)
                kpi_operational = kpi_data.get("operational") or {}
            except Exception:
                pass

        comparison_data = []
        for strategy, metrics in self.summary_stats.items():
            row = {
                'Strategy': strategy,
                'ROI (%)': f"{metrics.get('roi_mean', 0):.1f} ± {metrics.get('roi_std', 0):.1f}",
                'Value Preserved ($)': f"{metrics.get('value_preserved_mean', 0):,.0f} ± {metrics.get('value_preserved_std', 0):,.0f}",
                'Lost Value ($)': f"{metrics.get('lost_value_mean', 0):,.0f} ± {metrics.get('lost_value_std', 0):,.0f}",
                'Total Patches': f"{metrics.get('total_patches_mean', 0):.1f} ± {metrics.get('total_patches_std', 0):.1f}",
                'Patch Cost ($)': f"{metrics.get('total_patch_cost_mean', 0):,.0f} ± {metrics.get('total_patch_cost_std', 0):,.0f}",
                'Compromised Assets': f"{metrics.get('compromised_assets_mean', 0):.1f} ± {metrics.get('compromised_assets_std', 0):.1f}",
                'Attack Success Rate (%)': f"{metrics.get('attack_success_rate_mean', 0):.1f} ± {metrics.get('attack_success_rate_std', 0):.1f}",
                'Detection Coverage (%)': f"{metrics.get('detection_coverage_mean', 0):.1f} ± {metrics.get('detection_coverage_std', 0):.1f}",
                'Avg Time to Detection (steps)': f"{metrics.get('avg_time_to_detection_mean', 0):.2f} ± {metrics.get('avg_time_to_detection_std', 0):.2f}",
                'Attack Disruption Rate (%)': f"{metrics.get('attack_disruption_rate_mean', 0):.1f} ± {metrics.get('attack_disruption_rate_std', 0):.1f}"
            }
            # Canonical operational KPIs (from incident timestamps; same scale as legacy table)
            inc_mean, inc_std = metrics.get('incidents_total_mean', 0), metrics.get('incidents_total_std', 0)
            mttd_mean, mttd_std = metrics.get('canonical_mttd_steps_mean'), metrics.get('canonical_mttd_steps_std')
            mttr_mean, mttr_std = metrics.get('canonical_mttr_steps_mean'), metrics.get('canonical_mttr_steps_std')
            par_mean, par_std = metrics.get('protected_asset_ratio_pct_mean', 0), metrics.get('protected_asset_ratio_pct_std', 0)
            row['Incidents'] = f"{inc_mean:.1f} ± {inc_std:.1f}" if inc_mean is not None and not (isinstance(inc_mean, float) and np.isnan(inc_mean)) else "—"
            row['MTTD (steps)'] = f"{mttd_mean:.2f} ± {mttd_std:.2f}" if mttd_mean is not None and not (isinstance(mttd_mean, float) and np.isnan(mttd_mean)) else "—"
            row['MTTR (steps)'] = f"{mttr_mean:.2f} ± {mttr_std:.2f}" if mttr_mean is not None and not (isinstance(mttr_mean, float) and np.isnan(mttr_mean)) else "—"
            row['Protected Asset Ratio (%)'] = f"{par_mean:.1f} ± {par_std:.1f}" if par_mean is not None and not (isinstance(par_mean, float) and np.isnan(par_mean)) else "—"
            op = kpi_operational.get(strategy, {})
            if op:
                mttr = op.get("mttr", {})
                mttd = op.get("mttd", {})
                inc = op.get("incidents", {})
                prot = op.get("protected_ratio_pct", {})
                if inc:
                    row['Incidents (kpi file)'] = f"{inc.get('mean', 0):.1f} ± {inc.get('sd', 0):.1f}"
                if mttd and mttd.get('n'):
                    row['MTTD (steps, kpi)'] = f"{mttd.get('mean', 0):.1f} ± {mttd.get('sd', 0):.1f} (n={mttd.get('n', 0)})"
                if mttr and mttr.get('n'):
                    row['MTTR (steps, kpi)'] = f"{mttr.get('mean', 0):.1f} ± {mttr.get('sd', 0):.1f} (n={mttr.get('n', 0)})"
                if prot:
                    row['Protected ratio (kpi, %)'] = f"{prot.get('mean', 0):.1f} ± {prot.get('sd', 0):.1f}"
            comparison_data.append(row)

        return pd.DataFrame(comparison_data)
    
    def create_visualizations(self):
        """Create comprehensive visualizations."""
        print("Creating visualizations...")
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('APT3 Defense Strategy Performance Comparison', fontsize=16, fontweight='bold')
        
        # 1. ROI Comparison
        self._plot_roi_comparison(axes[0, 0])
        
        # 2. Value Preservation
        self._plot_value_preservation(axes[0, 1])
        
        # 3. Attack Success Rate
        self._plot_attack_success_rate(axes[0, 2])
        
        # 4. Patch Efficiency
        self._plot_patch_efficiency(axes[1, 0])
        
        # 5. Detection Performance
        self._plot_detection_performance(axes[1, 1])
        
        # 6. Cost-Benefit Analysis
        self._plot_cost_benefit(axes[1, 2])
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create additional detailed plots
        self._create_detailed_plots()
    
    def _plot_roi_comparison(self, ax):
        """Plot ROI comparison."""
        strategies = list(self.summary_stats.keys())
        roi_means = [self.summary_stats[s].get('roi_mean', 0) for s in strategies]
        roi_stds = [self.summary_stats[s].get('roi_std', 0) for s in strategies]
        
        bars = ax.bar(strategies, roi_means, yerr=roi_stds, capsize=5, alpha=0.7)
        ax.set_title('ROI Comparison', fontweight='bold')
        ax.set_ylabel('ROI (%)')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, mean in zip(bars, roi_means):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(roi_stds),
                   f'{mean:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    def _plot_value_preservation(self, ax):
        """Plot value preservation comparison."""
        strategies = list(self.summary_stats.keys())
        preserved_means = [self.summary_stats[s].get('value_preserved_mean', 0) for s in strategies]
        lost_means = [self.summary_stats[s].get('lost_value_mean', 0) for s in strategies]
        
        x = np.arange(len(strategies))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, preserved_means, width, label='Value Preserved', alpha=0.7)
        bars2 = ax.bar(x + width/2, lost_means, width, label='Value Lost', alpha=0.7)
        
        ax.set_title('Value Preservation vs Loss', fontweight='bold')
        ax.set_ylabel('Value ($)')
        ax.set_xticks(x)
        ax.set_xticklabels(strategies, rotation=45)
        ax.legend()
    
    def _plot_attack_success_rate(self, ax):
        """Plot attack success rate comparison."""
        strategies = list(self.summary_stats.keys())
        success_rates = [self.summary_stats[s].get('attack_success_rate_mean', 0) for s in strategies]
        
        bars = ax.bar(strategies, success_rates, alpha=0.7, color='red')
        ax.set_title('Attack Success Rate', fontweight='bold')
        ax.set_ylabel('Success Rate (%)')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, rate in zip(bars, success_rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    def _plot_patch_efficiency(self, ax):
        """Plot patch efficiency comparison."""
        strategies = list(self.summary_stats.keys())
        patches = [self.summary_stats[s].get('total_patches_mean', 0) for s in strategies]
        costs = [self.summary_stats[s].get('total_patch_cost_mean', 0) for s in strategies]
        
        scatter = ax.scatter(patches, costs, s=100, alpha=0.7)
        ax.set_title('Patch Efficiency', fontweight='bold')
        ax.set_xlabel('Total Patches')
        ax.set_ylabel('Patch Cost ($)')
        
        # Add strategy labels
        for i, strategy in enumerate(strategies):
            ax.annotate(strategy, (patches[i], costs[i]), xytext=(5, 5), 
                       textcoords='offset points', fontsize=8)
    
    def _plot_detection_performance(self, ax):
        """Plot detection performance comparison."""
        strategies = list(self.summary_stats.keys())
        detection_coverage = [self.summary_stats[s].get('detection_coverage_mean', 0) for s in strategies]
        disruption_rate = [self.summary_stats[s].get('attack_disruption_rate_mean', 0) for s in strategies]
        
        x = np.arange(len(strategies))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, detection_coverage, width, label='Detection Coverage', alpha=0.7)
        bars2 = ax.bar(x + width/2, disruption_rate, width, label='Attack Disruption', alpha=0.7)
        
        ax.set_title('Detection Performance', fontweight='bold')
        ax.set_ylabel('Rate (%)')
        ax.set_xticks(x)
        ax.set_xticklabels(strategies, rotation=45)
        ax.legend()
    
    def _plot_cost_benefit(self, ax):
        """Plot cost-benefit analysis."""
        strategies = list(self.summary_stats.keys())
        roi_values = [self.summary_stats[s].get('roi_mean', 0) for s in strategies]
        patch_costs = [self.summary_stats[s].get('total_patch_cost_mean', 0) for s in strategies]
        
        scatter = ax.scatter(patch_costs, roi_values, s=100, alpha=0.7)
        ax.set_title('Cost-Benefit Analysis', fontweight='bold')
        ax.set_xlabel('Patch Cost ($)')
        ax.set_ylabel('ROI (%)')
        
        # Add strategy labels
        for i, strategy in enumerate(strategies):
            ax.annotate(strategy, (patch_costs[i], roi_values[i]), xytext=(5, 5), 
                       textcoords='offset points', fontsize=8)
    
    def _create_detailed_plots(self):
        """Create additional detailed plots."""
        # Create a separate figure for detailed analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Detailed Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Compromised Assets vs Patches
        self._plot_compromised_vs_patches(axes[0, 0])
        
        # 2. Time to Detection
        self._plot_time_to_detection(axes[0, 1])
        
        # 3. Strategic Metrics
        self._plot_strategic_metrics(axes[1, 0])
        
        # 4. Performance Distribution
        self._plot_performance_distribution(axes[1, 1])
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'detailed_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_compromised_vs_patches(self, ax):
        """Plot compromised assets vs patches."""
        strategies = list(self.summary_stats.keys())
        compromised = [self.summary_stats[s].get('compromised_assets_mean', 0) for s in strategies]
        patches = [self.summary_stats[s].get('total_patches_mean', 0) for s in strategies]
        
        scatter = ax.scatter(patches, compromised, s=100, alpha=0.7)
        ax.set_title('Patches vs Compromised Assets', fontweight='bold')
        ax.set_xlabel('Total Patches')
        ax.set_ylabel('Compromised Assets')
        
        # Add trend line
        z = np.polyfit(patches, compromised, 1)
        p = np.poly1d(z)
        ax.plot(patches, p(patches), "r--", alpha=0.8)
        
        # Add strategy labels
        for i, strategy in enumerate(strategies):
            ax.annotate(strategy, (patches[i], compromised[i]), xytext=(5, 5), 
                       textcoords='offset points', fontsize=8)
    
    def _plot_time_to_detection(self, ax):
        """Plot time to detection comparison."""
        strategies = list(self.summary_stats.keys())
        detection_times = [self.summary_stats[s].get('avg_time_to_detection_mean', 0) for s in strategies]
        
        bars = ax.bar(strategies, detection_times, alpha=0.7, color='orange')
        ax.set_title('Average Time to Detection', fontweight='bold')
        ax.set_ylabel('Time (steps)')
        ax.tick_params(axis='x', rotation=45)
    
    def _plot_strategic_metrics(self, ax):
        """Plot strategic metrics from strategic_metrics.json."""
        if hasattr(self, 'strategic_metrics'):
            strategies = list(self.strategic_metrics.keys())
            attack_success_rates = []
            
            for strategy in strategies:
                if strategy in self.strategic_metrics and self.strategic_metrics[strategy]:
                    # Calculate average attack success rate
                    rates = [trial.get('attack_success_rate', 0) for trial in self.strategic_metrics[strategy]]
                    attack_success_rates.append(np.mean(rates) if rates else 0)
                else:
                    attack_success_rates.append(0)
            
            bars = ax.bar(strategies, attack_success_rates, alpha=0.7, color='purple')
            ax.set_title('Strategic Attack Success Rates', fontweight='bold')
            ax.set_ylabel('Success Rate (%)')
            ax.tick_params(axis='x', rotation=45)
    
    def _plot_performance_distribution(self, ax):
        """Plot performance distribution."""
        strategies = list(self.summary_stats.keys())
        roi_values = [self.summary_stats[s].get('roi_mean', 0) for s in strategies]
        
        # Create a horizontal bar chart
        y_pos = np.arange(len(strategies))
        bars = ax.barh(y_pos, roi_values, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(strategies)
        ax.set_xlabel('ROI (%)')
        ax.set_title('Performance Ranking', fontweight='bold')
        
        # Add value labels
        for i, (bar, roi) in enumerate(zip(bars, roi_values)):
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height()/2,
                   f'{roi:.1f}%', ha='left', va='center', fontweight='bold')
    
    def generate_analysis_report(self):
        """Generate a comprehensive analysis report."""
        print("Generating analysis report...")
        
        # Create comparison table
        comparison_table = self.create_performance_comparison_table()

        # Ensure results directory exists before writing
        self.results_dir.mkdir(parents=True, exist_ok=True)
        # Save comparison table
        comparison_table.to_csv(self.results_dir / 'performance_comparison.csv', index=False)
        
        # Generate text report
        report = self._generate_text_report(comparison_table)
        
        # Save report
        with open(self.results_dir / 'analysis_report.md', 'w') as f:
            f.write(report)
        
        print(f"Analysis complete! Results saved to {self.results_dir}")
        return comparison_table, report
    
    def _generate_text_report(self, comparison_table: pd.DataFrame) -> str:
        """Generate a comprehensive text report."""
        report = f"""# APT3 Defense Strategy Performance Analysis Report

## Executive Summary

This report analyzes the performance of {len(self.summary_stats)} different defense strategies against APT3 attacks across multiple simulation trials.

## Performance Rankings

### Top Performers by ROI:
"""
        
        # Sort strategies by ROI
        roi_ranking = []
        for strategy, metrics in self.summary_stats.items():
            roi = metrics.get('roi_mean', 0)
            roi_ranking.append((strategy, roi))
        
        roi_ranking.sort(key=lambda x: x[1], reverse=True)
        
        for i, (strategy, roi) in enumerate(roi_ranking, 1):
            report += f"{i}. **{strategy}**: {roi:.1f}% ROI\n"
        
        report += f"""
### Key Findings:

1. **Best Value Preservation**: {roi_ranking[0][0]} achieved the highest ROI at {roi_ranking[0][1]:.1f}%
2. **Most Cost-Effective**: Analysis of patch costs vs ROI shows optimal efficiency
3. **Attack Resistance**: Comparison of attack success rates across strategies
4. **Detection Capabilities**: Assessment of detection coverage and disruption rates

## Detailed Performance Metrics

{comparison_table.to_markdown(index=False)}
"""
        # Include operational and economics KPIs if generated by calculate_operational_kpis.py
        kpi_aggregates_file = self.results_dir / "kpi_aggregates.json"
        op_kpis_file = self.results_dir / "operational_kpis.md"
        econ_kpis_file = self.results_dir / "economics_kpis.md"
        if op_kpis_file.exists() or econ_kpis_file.exists() or kpi_aggregates_file.exists():
            report += """
## Operational and Economics KPIs

The following tables are produced by `calculate_operational_kpis.py` (per-trial KPIs aggregated with mean ± sd, n). Use them alongside the performance metrics above for a full picture (e.g. incidents, MTTD/MTTR, protected-asset ratio, lateral-movement events; value preserved, patch cost, net value, budget utilization).

"""
            if op_kpis_file.exists():
                report += "### Operational KPIs\n\n"
                report += op_kpis_file.read_text(encoding="utf-8", errors="replace") + "\n\n"
            if econ_kpis_file.exists():
                report += "### Economics KPIs\n\n"
                report += econ_kpis_file.read_text(encoding="utf-8", errors="replace") + "\n\n"
            if kpi_aggregates_file.exists():
                report += "*(Full aggregates: `kpi_aggregates.json`)*\n\n"

        primary = roi_ranking[0][0] if roi_ranking else "N/A"
        secondary = roi_ranking[1][0] if len(roi_ranking) > 1 else "N/A"
        report += f"""
## Strategic Insights

### Zero-Day Capabilities
- RL Defender and Hybrid Defender strategies include zero-day vulnerability awareness
- Emergency budget activation for critical threats
- Enhanced scoring with 5x multiplier for zero-day vulnerabilities

### Cost-Benefit Analysis
- Patch efficiency varies significantly between strategies
- ROI optimization through strategic vulnerability prioritization
- Balance between security investment and value protection

### Attack Resistance
- Different strategies show varying levels of attack success resistance
- Detection and disruption capabilities impact overall performance
- Time to detection affects damage mitigation

## Recommendations

1. **Primary Strategy**: {primary} for optimal ROI and value preservation
2. **Secondary Strategy**: {secondary} for balanced performance
3. **Specialized Use**: Consider strategy-specific strengths for different threat scenarios

## Methodology

- Simulation trials: Multiple runs per strategy for statistical significance
- Metrics: ROI, value preservation, attack resistance, detection capabilities
- Analysis: Comprehensive comparison with confidence intervals
- Visualization: Multi-dimensional performance assessment

---
*Report generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        return report

def main():
    """Main analysis function."""
    import argparse
    parser = argparse.ArgumentParser(description="Analyze APT3 simulation results.")
    parser.add_argument("results_dir", nargs="?", default="apt3_simulation_results/simulation_20250703_111341",
                        help="Path to simulation results (default: simulation_20250703_111341)")
    args = parser.parse_args()
    results_dir = args.results_dir
    analyzer = SimulationAnalyzer(results_dir)
    
    # Load and analyze results
    analyzer.load_results()
    if not analyzer.results:
        import sys
        print(f"Error: No strategies loaded. Check that the path exists and contains simulation_summary.json")
        print(f"  Path used: {results_dir}")
        sys.exit(1)
    analyzer.calculate_performance_metrics()
    
    # Generate comprehensive analysis
    comparison_table, report = analyzer.generate_analysis_report()
    
    # Create visualizations
    analyzer.create_visualizations()
    
    # Print summary
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Results saved to: {results_dir}")
    print(f"Files generated:")
    print(f"  - performance_comparison.csv")
    print(f"  - analysis_report.md")
    print(f"  - performance_comparison.png")
    print(f"  - detailed_analysis.png")
    print("\nTop 3 Strategies by ROI:")
    
    roi_ranking = []
    for strategy, metrics in analyzer.summary_stats.items():
        roi = metrics.get('roi_mean', 0)
        roi_ranking.append((strategy, roi))
    
    roi_ranking.sort(key=lambda x: x[1], reverse=True)
    for i, (strategy, roi) in enumerate(roi_ranking[:3], 1):
        print(f"  {i}. {strategy}: {roi:.1f}% ROI")

if __name__ == "__main__":
    main() 