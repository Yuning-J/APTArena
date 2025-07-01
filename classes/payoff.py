# payoff.py
from .state import State, Vulnerability
from .cost import CostCalculator

class PayoffFunctions:
    def __init__(self, attack_cost_per_action: float = 1.0, base_patch_cost: float = 1.0, psi: float = 1.0):
        self.attack_cost_per_action = attack_cost_per_action  # Base cost per attack action
        self.base_patch_cost = base_patch_cost                # Base cost per patch
        self.psi = psi                                        # Weight balancing risk and operational cost
        # Store patch costs for reporting
        self.patch_costs_history = []

    # In payoff.py
    def attacker_payoff(self, state: State, attacker_actions: list) -> float:
        """
        Compute attacker payoff with exploit costs and business value consideration.
        U_A = sum(compromised_asset_value * business_value) - sum(exploit_costs)
        """
        # Calculate reward from compromised assets
        total_reward = 0.0
        for asset in state.system.assets:
            if asset.is_compromised:
                # Base reward from asset criticality
                asset_reward = asset.criticality_level

                # Incorporate business value with clearly defined scaling
                if hasattr(asset, 'business_value'):
                    # Apply a quadratic scaling to emphasize high-value targets
                    # Formula: asset_reward * (1 + (business_value/10)^2)
                    business_factor = 1.0 + ((asset.business_value / 10.0) ** 2)
                    asset_reward *= business_factor

                # Apply kill chain stage multiplier - more reward in later stages
                if hasattr(state, 'k'):
                    # Escalating reward based on kill chain progression
                    if state.k >= 7:  # Actions on Objectives / Exfiltration
                        asset_reward *= 3.0
                    elif state.k >= 5:  # Command & Control / Lateral Movement
                        asset_reward *= 2.0
                    elif state.k >= 3:  # Exploitation / Installation
                        asset_reward *= 1.5

                # Apply data sensitivity multiplier if available
                if hasattr(asset, 'contains_sensitive_data') and asset.contains_sensitive_data:
                    asset_reward *= 1.5  # +50% reward for sensitive data

                total_reward += asset_reward

        # Calculate exploit costs for all attempted actions
        total_exploit_cost = 0.0
        for vuln in attacker_actions:
            # Find the asset for this vulnerability
            asset = None
            for a in state.system.assets:
                for comp in a.components:
                    if vuln in comp.vulnerabilities:
                        asset = a
                        break
                if asset:
                    break

            total_exploit_cost += CostCalculator.calculate_exploit_cost(
                vuln,
                state=state,  # Pass state as named parameter
                asset=asset,  # Pass asset as named parameter
                base_cost=self.attack_cost_per_action  # Pass cost as named parameter
            )

        return total_reward - total_exploit_cost

    def defender_payoff(self, state: State, defender_actions: list) -> dict:
        """
        Compute defender payoff with detailed breakdown and business value consideration.
        U_D = -sum(compromised_asset_value * business_value) - psi * sum(patch_costs)
        
        Args:
            state: Current system state
            defender_actions: List of vulnerabilities being patched
            
        Returns:
            dict: Contains total payoff and breakdown of costs
        """
        # Calculate risk component from compromised assets
        total_risk = 0.0
        assets_at_risk = []
        
        for asset in state.system.assets:
            if asset.is_compromised:
                # Base risk from asset criticality
                asset_risk = asset.criticality_level
                
                # Incorporate business value with identical scaling to attacker payoff
                if hasattr(asset, 'business_value'):
                    # Apply the same quadratic scaling as in attacker payoff
                    # Formula: asset_risk * (1 + (business_value/10)^2)
                    business_factor = 1.0 + ((asset.business_value / 10.0) ** 2)
                    asset_risk *= business_factor
                
                # Apply kill chain stage multiplier - more risk in later stages
                if hasattr(state, 'k'):
                    # Use identical scaling to attacker reward calculation
                    if state.k >= 7:  # Actions on Objectives / Exfiltration
                        asset_risk *= 3.0
                    elif state.k >= 5:  # Command & Control / Lateral Movement
                        asset_risk *= 2.0
                    elif state.k >= 3:  # Exploitation / Installation
                        asset_risk *= 1.5
                
                # Apply data sensitivity multiplier if available
                if hasattr(asset, 'contains_sensitive_data') and asset.contains_sensitive_data:
                    asset_risk *= 1.5  # +50% risk for sensitive data
                
                total_risk += asset_risk
                
                # Track compromised assets for reporting
                assets_at_risk.append({
                    'name': asset.name,
                    'criticality': asset.criticality_level,
                    'business_value': getattr(asset, 'business_value', 0),
                    'risk_contribution': asset_risk
                })
        
        # Calculate operational cost with variable patch costs
        patch_costs = []
        for vuln in defender_actions:
            # Use enhanced patch cost calculation
            patch_cost = CostCalculator.calculate_patch_cost(vuln, state, self.base_patch_cost)
            
            # Find the asset for this vulnerability
            asset = None
            for a in state.system.assets:
                for comp in a.components:
                    if vuln in comp.vulnerabilities:
                        asset = a
                        break
                if asset:
                    break
            
            # Store detailed information for reporting
            centrality = getattr(asset, 'centrality', 0.5) if asset else 0.5
            asset_name = asset.name if asset else "Unknown"
            criticality = asset.criticality_level if asset else 0
            business_value = getattr(asset, 'business_value', 0) if asset else 0
            
            patch_costs.append({
                'cve_id': vuln.cve_id,
                'cvss': vuln.cvss,
                'asset': asset_name,
                'asset_criticality': criticality,
                'business_value': business_value,
                'centrality': centrality,
                'cost': patch_cost
            })
        
        total_operational_cost = sum(item['cost'] for item in patch_costs)
        
        # Save costs for historical tracking
        self.patch_costs_history.append(patch_costs)
        
        # Calculate total payoff with weighting factor
        total_payoff = -total_risk - (self.psi * total_operational_cost)
        
        # Return detailed breakdown
        return {
            'total_payoff': total_payoff,
            'risk_component': -total_risk,
            'operational_component': -(self.psi * total_operational_cost),
            'patch_costs': patch_costs,
            'assets_at_risk': assets_at_risk,
            'total_operational_cost': total_operational_cost
        }
    
    def get_cumulative_costs(self) -> dict:
        """
        Calculate cumulative operational costs across all simulation steps.
        
        Returns:
            dict: Summary of operational costs
        """
        # Flatten the list of patch costs
        all_patches = [patch for step_patches in self.patch_costs_history for patch in step_patches]
        
        # Calculate total cost
        total_cost = sum(patch['cost'] for patch in all_patches)
        
        # Get top 5 most expensive patches
        sorted_patches = sorted(all_patches, key=lambda x: x['cost'], reverse=True)
        most_expensive = sorted_patches[:5] if len(sorted_patches) >= 5 else sorted_patches
        
        # Calculate costs by asset centrality
        centrality_groups = {
            "high": [],    # Centrality > 0.7
            "medium": [],  # Centrality 0.3-0.7
            "low": []      # Centrality < 0.3
        }
        
        for patch in all_patches:
            centrality = patch.get('centrality', 0.5)
            if centrality > 0.7:
                centrality_groups["high"].append(patch)
            elif centrality > 0.3:
                centrality_groups["medium"].append(patch)
            else:
                centrality_groups["low"].append(patch)
        
        # Calculate costs by asset criticality
        criticality_groups = {
            "critical": [],   # Criticality >= 4
            "important": [],  # Criticality 2-3
            "low": []         # Criticality 0-1
        }
        
        for patch in all_patches:
            criticality = patch.get('asset_criticality', 0)
            if criticality >= 4:
                criticality_groups["critical"].append(patch)
            elif criticality >= 2:
                criticality_groups["important"].append(patch)
            else:
                criticality_groups["low"].append(patch)
        
        # NEW: Calculate costs by business value
        business_value_groups = {
            "high": [],    # Business Value > 7
            "medium": [],  # Business Value 3-7
            "low": []      # Business Value < 3
        }
        
        for patch in all_patches:
            business_value = patch.get('business_value', 0)
            if business_value > 7:
                business_value_groups["high"].append(patch)
            elif business_value >= 3:
                business_value_groups["medium"].append(patch)
            else:
                business_value_groups["low"].append(patch)
        
        # Calculate summary statistics
        centrality_costs = {
            "high": sum(p['cost'] for p in centrality_groups["high"]),
            "medium": sum(p['cost'] for p in centrality_groups["medium"]),
            "low": sum(p['cost'] for p in centrality_groups["low"]),
            "high_count": len(centrality_groups["high"]),
            "medium_count": len(centrality_groups["medium"]),
            "low_count": len(centrality_groups["low"])
        }
        
        criticality_costs = {
            "critical": sum(p['cost'] for p in criticality_groups["critical"]),
            "important": sum(p['cost'] for p in criticality_groups["important"]),
            "low": sum(p['cost'] for p in criticality_groups["low"]),
            "critical_count": len(criticality_groups["critical"]),
            "important_count": len(criticality_groups["important"]),
            "low_count": len(criticality_groups["low"])
        }
        
        business_value_costs = {
            "high": sum(p['cost'] for p in business_value_groups["high"]),
            "medium": sum(p['cost'] for p in business_value_groups["medium"]),
            "low": sum(p['cost'] for p in business_value_groups["low"]),
            "high_count": len(business_value_groups["high"]),
            "medium_count": len(business_value_groups["medium"]),
            "low_count": len(business_value_groups["low"])
        }
        
        return {
            'total_operational_cost': total_cost,
            'patches_applied': len(all_patches),
            'average_patch_cost': round(total_cost / len(all_patches), 2) if all_patches else 0,
            'most_expensive_patches': most_expensive,
            'costs_by_centrality': centrality_costs,
            'costs_by_criticality': criticality_costs,
            'costs_by_business_value': business_value_costs
        }
