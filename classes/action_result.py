"""
ActionResult class for standardizing action results across the simulation.

This module provides a standardized way to create and handle action results,
eliminating code duplication and improving consistency.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class ActionResult:
    """Standardized action result for simulation actions."""
    
    action_type: str
    action_result: bool
    reason: str
    target_vuln: Optional[str] = None
    target_asset: Optional[str] = None
    target_component: Optional[str] = None
    cost: Optional[float] = None
    probability: Optional[float] = None
    vuln_key: Optional[str] = None
    tactic: Optional[str] = None
    is_recon: bool = False
    attempt_cost: Optional[float] = None
    from_asset: Optional[str] = None
    attempt_number: Optional[int] = None
    required_cost: Optional[float] = None
    available_budget: Optional[float] = None
    
    @classmethod
    def success(cls, action_type: str, **kwargs) -> 'ActionResult':
        """Create a successful action result."""
        return cls(action_type=action_type, action_result=True, reason='success', **kwargs)
    
    @classmethod
    def failure(cls, action_type: str, reason: str, **kwargs) -> 'ActionResult':
        """Create a failed action result."""
        return cls(action_type=action_type, action_result=False, reason=reason, **kwargs)
    
    @classmethod
    def pause(cls, reason: str) -> 'ActionResult':
        """Create a pause action result."""
        return cls(action_type='pause', action_result=False, reason=reason)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the action result to a dictionary."""
        result = {
            'action_type': self.action_type,
            'action_result': self.action_result,
            'reason': self.reason,
            'is_recon': self.is_recon
        }
        
        # Add optional fields if they have values
        optional_fields = [
            'target_vuln', 'target_asset', 'target_component', 'cost', 
            'probability', 'vuln_key', 'tactic', 'attempt_cost', 'from_asset',
            'attempt_number', 'required_cost', 'available_budget'
        ]
        
        for field in optional_fields:
            value = getattr(self, field)
            if value is not None:
                result[field] = value
                
        return result