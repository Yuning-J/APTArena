#!/usr/bin/env python3
"""
SimulationConfig class for managing APT3 simulation configuration.
This module centralizes all configuration parameters and validation logic.
"""

import os
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class SimulationConfig:
    """
    Configuration class for APT3 simulation parameters.
    
    This class centralizes all configuration parameters and provides validation
    to ensure simulation parameters are within acceptable ranges.
    """
    
    # Core simulation parameters
    data_file: str
    num_steps: int
    defender_budget: int
    attacker_budget: int
    
    # Attacker behavior parameters
    psi: float = 1.0
    cost_aware_attacker: bool = True
    cost_aware_defender: bool = True
    detection_averse: bool = True
    attacker_sophistication: float = 0.9
    
    # POMDP parameters
    gamma: float = 0.9
    use_hmm: bool = False
    
    # Optional parameters
    business_values_file: Optional[str] = None
    cost_cache_file: Optional[str] = None
    
    # Validation ranges
    _min_steps: int = 1
    _max_steps: int = 10000
    _min_budget: int = 0
    _max_budget: int = 10000000
    _min_psi: float = 0.0
    _max_psi: float = 2.0
    _min_gamma: float = 0.0
    _max_gamma: float = 1.0
    _min_sophistication: float = 0.0
    _max_sophistication: float = 1.0
    
    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        self._validate_parameters()
        self._validate_files()
    
    def _validate_parameters(self):
        """Validate all configuration parameters."""
        # Validate num_steps
        if not isinstance(self.num_steps, int) or self.num_steps < self._min_steps:
            raise ValueError(f"num_steps must be an integer >= {self._min_steps}, got {self.num_steps}")
        if self.num_steps > self._max_steps:
            raise ValueError(f"num_steps must be <= {self._max_steps}, got {self.num_steps}")
        
        # Validate budgets
        if not isinstance(self.defender_budget, (int, float)) or self.defender_budget < self._min_budget:
            raise ValueError(f"defender_budget must be >= {self._min_budget}, got {self.defender_budget}")
        if self.defender_budget > self._max_budget:
            raise ValueError(f"defender_budget must be <= {self._max_budget}, got {self.defender_budget}")
        
        if not isinstance(self.attacker_budget, (int, float)) or self.attacker_budget < self._min_budget:
            raise ValueError(f"attacker_budget must be >= {self._min_budget}, got {self.attacker_budget}")
        if self.attacker_budget > self._max_budget:
            raise ValueError(f"attacker_budget must be <= {self._max_budget}, got {self.attacker_budget}")
        
        # Validate psi
        if not isinstance(self.psi, (int, float)) or self.psi < self._min_psi:
            raise ValueError(f"psi must be >= {self._min_psi}, got {self.psi}")
        if self.psi > self._max_psi:
            raise ValueError(f"psi must be <= {self._max_psi}, got {self.psi}")
        
        # Validate gamma
        if not isinstance(self.gamma, (int, float)) or self.gamma < self._min_gamma:
            raise ValueError(f"gamma must be >= {self._min_gamma}, got {self.gamma}")
        if self.gamma > self._max_gamma:
            raise ValueError(f"gamma must be <= {self._max_gamma}, got {self.gamma}")
        
        # Validate attacker_sophistication
        if not isinstance(self.attacker_sophistication, (int, float)) or self.attacker_sophistication < self._min_sophistication:
            raise ValueError(f"attacker_sophistication must be >= {self._min_sophistication}, got {self.attacker_sophistication}")
        if self.attacker_sophistication > self._max_sophistication:
            raise ValueError(f"attacker_sophistication must be <= {self._max_sophistication}, got {self.attacker_sophistication}")
        
        # Validate boolean parameters
        if not isinstance(self.cost_aware_attacker, bool):
            raise ValueError(f"cost_aware_attacker must be a boolean, got {type(self.cost_aware_attacker)}")
        if not isinstance(self.cost_aware_defender, bool):
            raise ValueError(f"cost_aware_defender must be a boolean, got {type(self.cost_aware_defender)}")
        if not isinstance(self.detection_averse, bool):
            raise ValueError(f"detection_averse must be a boolean, got {type(self.detection_averse)}")
        if not isinstance(self.use_hmm, bool):
            raise ValueError(f"use_hmm must be a boolean, got {type(self.use_hmm)}")
    
    def _validate_files(self):
        """Validate that required files exist."""
        # Validate data_file
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(f"Data file not found: {self.data_file}")
        
        # Validate business_values_file if provided
        if self.business_values_file and not os.path.exists(self.business_values_file):
            raise FileNotFoundError(f"Business values file not found: {self.business_values_file}")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SimulationConfig':
        """Create SimulationConfig from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_json_file(cls, json_file: str) -> 'SimulationConfig':
        """Create SimulationConfig from JSON file."""
        if not os.path.exists(json_file):
            raise FileNotFoundError(f"Configuration file not found: {json_file}")
        
        with open(json_file, 'r') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_cli_args(cls, args) -> 'SimulationConfig':
        """Create SimulationConfig from command line arguments."""
        config_dict = {
            'data_file': args.data_file,
            'num_steps': args.num_steps,
            'defender_budget': args.defender_budget,
            'attacker_budget': args.attacker_budget,
            'psi': getattr(args, 'psi', 1.0),
            'cost_aware_attacker': getattr(args, 'cost_aware_attacker', True),
            'cost_aware_defender': getattr(args, 'cost_aware_defender', True),
            'detection_averse': getattr(args, 'detection_averse', True),
            'gamma': getattr(args, 'gamma', 0.9),
            'business_values_file': getattr(args, 'business_values_file', None),
            'use_hmm': getattr(args, 'use_hmm', False),
            'attacker_sophistication': getattr(args, 'attacker_sophistication', 0.9),
            'cost_cache_file': getattr(args, 'cost_cache_file', None),
        }
        
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'data_file': self.data_file,
            'num_steps': self.num_steps,
            'defender_budget': self.defender_budget,
            'attacker_budget': self.attacker_budget,
            'psi': self.psi,
            'cost_aware_attacker': self.cost_aware_attacker,
            'cost_aware_defender': self.cost_aware_defender,
            'detection_averse': self.detection_averse,
            'gamma': self.gamma,
            'business_values_file': self.business_values_file,
            'use_hmm': self.use_hmm,
            'attacker_sophistication': self.attacker_sophistication,
            'cost_cache_file': self.cost_cache_file,
        }
    
    def to_json_file(self, json_file: str):
        """Save configuration to JSON file."""
        with open(json_file, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get a summary of configuration validation."""
        return {
            'valid': True,
            'data_file_exists': os.path.exists(self.data_file),
            'business_values_file_exists': self.business_values_file is None or os.path.exists(self.business_values_file),
            'cost_cache_file_exists': self.cost_cache_file is None or os.path.exists(self.cost_cache_file),
            'parameter_ranges': {
                'num_steps': f"{self._min_steps} <= {self.num_steps} <= {self._max_steps}",
                'defender_budget': f"{self._min_budget} <= {self.defender_budget} <= {self._max_budget}",
                'attacker_budget': f"{self._min_budget} <= {self.attacker_budget} <= {self._max_budget}",
                'psi': f"{self._min_psi} <= {self.psi} <= {self._max_psi}",
                'gamma': f"{self._min_gamma} <= {self.gamma} <= {self._max_gamma}",
                'attacker_sophistication': f"{self._min_sophistication} <= {self.attacker_sophistication} <= {self._max_sophistication}",
            }
        }
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return f"SimulationConfig(data_file='{self.data_file}', num_steps={self.num_steps}, " \
               f"defender_budget={self.defender_budget}, attacker_budget={self.attacker_budget})"
    
    def __repr__(self) -> str:
        """Detailed string representation of configuration."""
        return f"SimulationConfig(\n" \
               f"  data_file='{self.data_file}',\n" \
               f"  num_steps={self.num_steps},\n" \
               f"  defender_budget={self.defender_budget},\n" \
               f"  attacker_budget={self.attacker_budget},\n" \
               f"  psi={self.psi},\n" \
               f"  cost_aware_attacker={self.cost_aware_attacker},\n" \
               f"  cost_aware_defender={self.cost_aware_defender},\n" \
               f"  detection_averse={self.detection_averse},\n" \
               f"  gamma={self.gamma},\n" \
               f"  business_values_file={self.business_values_file},\n" \
               f"  use_hmm={self.use_hmm},\n" \
               f"  attacker_sophistication={self.attacker_sophistication},\n" \
               f"  cost_cache_file={self.cost_cache_file}\n" \
               f")" 