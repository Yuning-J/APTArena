# threat_intelligence.py
import random
from .state import State, System  # Import State and System from state.py


class ThreatIntelligenceProcessor:
    def __init__(self, update_frequency: float = 0.1, threat_increase: float = 0.1):
        """
        Initialize the threat intelligence processor.

        Args:
            update_frequency (float): Probability of a threat update per vulnerability (default: 0.1).
            threat_increase (float): Amount to increase epss per update (default: 0.1).
        """
        self.update_frequency = update_frequency
        self.threat_increase = threat_increase

    def update_threat_levels(self, state: State):
        """
        Update threat-related attributes based on simulated LLM-augmented rag_pipeline pipeline.
        Updates epss, asset criticality, and phi(s) to reflect C_LLM.

        Args:
            state (State): Current system state with k, G (system), and phi.
        """
        # Simulate threat intelligence updates for vulnerabilities
        for asset in state.system.assets:
            for component in asset.components:
                for vuln in component.vulnerabilities:
                    if random.random() < self.update_frequency:  # Randomly simulate new intelligence
                        # Update vulnerability's exploit probability (epss)
                        vuln.epss = min(vuln.epss + self.threat_increase, 1.0)
                        # Update asset criticality based on threat intelligence
                        asset.criticality_level = min(asset.criticality_level + vuln.epss * 0.1, 10.0)  # Cap at 10

        # Recompute phi(s) to reflect updated threat levels
        state.phi = self._compute_updated_phi(state)

    def _compute_updated_phi(self, state: State) -> dict:
        """
        Recompute the feature vector phi(s) with updated ExternalThreatLevel.

        Args:
            state (State): Current system state.

        Returns:
            dict: Updated phi with all components, including ExternalThreatLevel.
        """
        # Existing phi components
        total_risk = sum(v.cvss for asset in state.system.assets
                         for component in asset.components
                         for v in component.vulnerabilities if not v.is_patched)
        num_compromised = sum(1 for asset in state.system.assets if asset.is_compromised)

        # Add ExternalThreatLevel based on average epss
        total_epss = sum(vuln.epss for asset in state.system.assets
                         for comp in asset.components
                         for vuln in comp.vulnerabilities)
        vuln_count = sum(1 for asset in state.system.assets
                         for comp in asset.components
                         for vuln in comp.vulnerabilities)
        external_threat_level = total_epss / vuln_count if vuln_count > 0 else 0.0

        # Additional components from the design
        detection_rate = 0.0  # Placeholder: could be based on monitoring logic
        recent_patches = sum(1 for asset in state.system.assets
                             for comp in asset.components
                             for v in comp.vulnerabilities if v.is_patched)
        recent_threat_hunts = 0  # Placeholder: not implemented yet
        observed_exploits = sum(1 for asset in state.system.assets
                                for comp in asset.components
                                for v in comp.vulnerabilities if v.is_exploited)

        return {
            "RiskScore": total_risk,
            "DetectionRate": detection_rate,
            "RecentPatches": recent_patches,
            "RecentThreatHunts": recent_threat_hunts,
            "ObservedExploits": observed_exploits,
            "ExternalThreatLevel": external_threat_level
        }

"""

from rag_pipeline.regression_infer import RegressionInfer

self.vuln_regressor = RegressionInfer("models/regressor/vuln")
self.asset_regressor = RegressionInfer("models/regressor/asset")

def update_threat_levels(self, state):
    for asset in state.system.assets:
        text = build_asset_text(asset)  # (build text using asset_id, type, value, etc.)
        asset.TR = self.asset_regressor.infer(text)

        for vuln in asset.vulnerabilities:
            text = build_vuln_text(vuln)  # (build text using CVE id, CVSS, vector, snippets)
            vuln.exploit_likelihood = self.vuln_regressor.infer(text)

"""