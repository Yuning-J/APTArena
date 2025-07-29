"""
State validation utilities for the APT Arena simulation.

This module provides centralized state validation to ensure data consistency
and reduce validation code duplication across the simulation.
"""

from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass

from .state import State, Asset, Vulnerability, Component, System
from .constants import RTU_ASSET_ID


@dataclass
class ValidationResult:
    """Result of a validation check."""
    is_valid: bool
    reason: str
    details: Optional[Dict[str, Any]] = None


class StateValidator:
    """Validates simulation state and data consistency."""
    
    def __init__(self):
        """Initialize state validator."""
        pass
    
    def validate_state(self, state: State) -> ValidationResult:
        """Validate the overall simulation state."""
        if state is None:
            return ValidationResult(False, "State is None")
        
        if not isinstance(state, State):
            return ValidationResult(False, "State is not a State instance")
        
        if not hasattr(state, 'system') or state.system is None:
            return ValidationResult(False, "State has no valid system")
        
        if not state.system.assets:
            return ValidationResult(False, "System has no assets")
        
        return ValidationResult(True, "State is valid")
    
    def validate_asset(self, asset: Asset) -> ValidationResult:
        """Validate an asset."""
        if asset is None:
            return ValidationResult(False, "Asset is None")
        
        if not isinstance(asset, Asset):
            return ValidationResult(False, "Asset is not an Asset instance")
        
        if not hasattr(asset, 'asset_id'):
            return ValidationResult(False, "Asset has no asset_id")
        
        if not asset.components:
            return ValidationResult(False, f"Asset {asset.asset_id} has no components")
        
        return ValidationResult(True, "Asset is valid")
    
    def validate_vulnerability(self, vuln: Vulnerability, asset_id: str, 
                             component_id: str) -> ValidationResult:
        """Validate a vulnerability."""
        if vuln is None:
            return ValidationResult(False, "Vulnerability is None")
        
        if not isinstance(vuln, Vulnerability):
            return ValidationResult(False, "Vulnerability is not a Vulnerability instance")
        
        if not hasattr(vuln, 'cve_id') or not vuln.cve_id:
            return ValidationResult(False, "Vulnerability has no CVE ID")
        
        # Check for invalid CVE IDs
        if vuln.cve_id.lower() == 'unknown':
            return ValidationResult(False, "Vulnerability has unknown CVE ID")
        
        return ValidationResult(True, "Vulnerability is valid")
    
    def validate_action(self, action: Dict[str, Any]) -> ValidationResult:
        """Validate an action dictionary."""
        if not action:
            return ValidationResult(False, "Action is empty or None")
        
        if not isinstance(action, dict):
            return ValidationResult(False, "Action is not a dictionary")
        
        if 'action_type' not in action:
            return ValidationResult(False, "Action missing action_type")
        
        action_type = action['action_type']
        required_fields = self._get_required_fields_for_action(action_type)
        
        missing_fields = [field for field in required_fields if not action.get(field)]
        if missing_fields:
            return ValidationResult(
                False, 
                f"Action missing required fields: {missing_fields}",
                {'missing_fields': missing_fields}
            )
        
        return ValidationResult(True, "Action is valid")
    
    def validate_budget_constraints(self, cost: float, available_budget: float, 
                                  entity_type: str) -> ValidationResult:
        """Validate budget constraints."""
        if cost < 0:
            return ValidationResult(False, f"{entity_type} cost cannot be negative")
        
        if available_budget < 0:
            return ValidationResult(False, f"{entity_type} budget cannot be negative")
        
        if cost > available_budget:
            return ValidationResult(
                False, 
                f"{entity_type} cost exceeds available budget",
                {
                    'cost': cost,
                    'available_budget': available_budget,
                    'shortfall': cost - available_budget
                }
            )
        
        return ValidationResult(True, "Budget constraints satisfied")
    
    def validate_vulnerability_lookup(self, vuln_lookup: Dict[str, Tuple], 
                                    system: System) -> ValidationResult:
        """Validate vulnerability lookup consistency."""
        if not vuln_lookup:
            return ValidationResult(False, "Vulnerability lookup is empty")
        
        # Check that all vulnerabilities in lookup exist in system
        for vuln_key, (vuln, asset, comp) in vuln_lookup.items():
            if not self._vulnerability_exists_in_system(vuln, asset, comp, system):
                return ValidationResult(
                    False, 
                    f"Vulnerability {vuln_key} not found in system"
                )
        
        return ValidationResult(True, "Vulnerability lookup is consistent")
    
    def validate_rtu_compromise(self, asset_id: str, step: int) -> ValidationResult:
        """Validate RTU compromise recording."""
        if str(asset_id) == RTU_ASSET_ID:
            return ValidationResult(
                True, 
                "RTU compromise recorded",
                {'rtu_compromised_step': step}
            )
        
        return ValidationResult(True, "Non-RTU asset compromise")
    
    def validate_network_connectivity(self, asset1_id: str, asset2_id: str, 
                                    connections: List) -> ValidationResult:
        """Validate network connectivity between assets."""
        asset1_str = str(asset1_id)
        asset2_str = str(asset2_id)
        
        for conn in connections:
            from_id = str(conn.from_asset.asset_id) if hasattr(conn, 'from_asset') else None
            to_id = str(conn.to_asset.asset_id) if hasattr(conn, 'to_asset') else None
            
            if from_id and to_id:
                if (from_id == asset1_str and to_id == asset2_str) or \
                   (from_id == asset2_str and to_id == asset1_str):
                    return ValidationResult(True, "Assets are network connected")
        
        return ValidationResult(False, "Assets are not network connected")
    
    def _get_required_fields_for_action(self, action_type: str) -> List[str]:
        """Get required fields for a specific action type."""
        field_mappings = {
            'initial_access': ['target_asset', 'target_vuln', 'target_component', 'probability'],
            'exploitation': ['target_asset', 'target_vuln', 'target_component', 'probability'],
            'lateral_movement': ['target_asset', 'probability'],
            'reconnaissance': ['probability'],
            'persistence': ['target_asset', 'probability']
        }
        
        return field_mappings.get(action_type, [])
    
    def _vulnerability_exists_in_system(self, vuln: Vulnerability, asset: Asset, 
                                      comp: Component, system: System) -> bool:
        """Check if a vulnerability exists in the system."""
        for sys_asset in system.assets:
            if str(sys_asset.asset_id) == str(asset.asset_id):
                for sys_comp in sys_asset.components:
                    if str(sys_comp.id) == str(comp.id):
                        for sys_vuln in sys_comp.vulnerabilities:
                            if sys_vuln.cve_id == vuln.cve_id:
                                return True
        return False 