#!/usr/bin/env python3
"""
Hybrid Strategy
==============

A hybrid cybersecurity defense strategy that combines Threat Intelligence and RL Defender approaches.
This strategy can be easily integrated into the existing APT3 RTU simulation.
"""

import random
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from .patching_strategies import ThreatIntelligenceStrategy
from src.RL_defender_strategy import RLAdaptiveThreatIntelligenceStrategy

@dataclass
class HybridDecision:
    """Container for hybrid decision results"""
    action: Optional[List[Tuple]]
    confidence: float
    strategy_used: str
    threat_intelligence_score: float
    rl_score: float
    reasoning: str

class HybridStrategy:
    """
    Hybrid Strategy combining Threat Intelligence and RL Defender
    
    This strategy:
    1. Uses both Threat Intelligence and RL Defender strategies
    2. Combines their recommendations with weighted fusion
    3. Provides explainable decisions
    4. Adapts weights based on performance
    """
    
    def __init__(self, threat_intelligence_weight: float = 0.4, rl_weight: float = 0.6, 
                 confidence_threshold: float = 0.7, adaptation_rate: float = 0.1):
        """
        Initialize hybrid strategy
        
        Args:
            threat_intelligence_weight: Initial weight for threat intelligence (0-1)
            rl_weight: Initial weight for RL defender (0-1)
            confidence_threshold: Threshold for RL decisions
            adaptation_rate: Rate at which weights adapt
        """
        # Validate weights
        if abs(threat_intelligence_weight + rl_weight - 1.0) > 0.01:
            raise ValueError("Weights must sum to 1.0")
        
        self.threat_intelligence_weight = threat_intelligence_weight
        self.rl_weight = rl_weight
        self.confidence_threshold = confidence_threshold
        self.adaptation_rate = adaptation_rate
        
        # Initialize component strategies
        self.threat_intelligence = ThreatIntelligenceStrategy()
        self.rl_defender = RLAdaptiveThreatIntelligenceStrategy()
        
        # Performance tracking
        self.threat_intelligence_performance = []
        self.rl_performance = []
        self.hybrid_performance = []
        
        # Decision history
        self.decision_history = []
        
        # Adaptation tracking
        self.adaptations = 0
        
        # Cost cache
        self._cost_cache = {}
        
        logging.info(f"Hybrid Strategy initialized with TI weight: {threat_intelligence_weight}, RL weight: {rl_weight}")
    
    def initialize(self, state, cost_cache):
        """Initialize both component strategies"""
        self._cost_cache = cost_cache
        self.threat_intelligence.initialize(state, cost_cache)
        self.rl_defender.initialize(state, cost_cache)
        
        # Set state for both strategies
        self.threat_intelligence.state = state
        self.rl_defender.state = state
        
        logging.info("Hybrid Strategy initialized with system state")
    
    def select_patches(self, state, remaining_budget: float, current_step: int, total_steps: int) -> List[Tuple]:
        """
        Select patches using hybrid approach - matches simulation interface
        
        Args:
            state: Current system state
            remaining_budget: Available budget
            current_step: Current simulation step
            total_steps: Total simulation steps
            
        Returns:
            List of tuples (vulnerability, cost)
        """
        try:
            # Get recommendations from both strategies
            ti_patches = self.threat_intelligence.select_patches(state, remaining_budget, current_step, total_steps)
            rl_patches = self.rl_defender.select_patches(state, remaining_budget, current_step, total_steps)
            
            # Calculate scores for each strategy
            ti_score = self._calculate_simple_score(ti_patches, state)
            rl_score = self._calculate_simple_score(rl_patches, state)
            
            # Combine recommendations based on weights and confidence
            combined_score = (ti_score * self.threat_intelligence_weight) + (rl_score * self.rl_weight)
            
            # Determine final action
            if combined_score > self.confidence_threshold:
                # High confidence: prefer RL recommendation
                final_patches = rl_patches
                strategy_used = "RL_Dominant"
                reasoning = f"High confidence ({combined_score:.3f}): Using RL recommendation"
            else:
                # Low confidence: prefer threat intelligence recommendation
                final_patches = ti_patches
                strategy_used = "TI_Dominant"
                reasoning = f"Low confidence ({combined_score:.3f}): Using Threat Intelligence recommendation"
            
            # If one strategy has no recommendation, use the other
            if not ti_patches and rl_patches:
                final_patches = rl_patches
                strategy_used = "RL_Only"
                reasoning = "No Threat Intelligence recommendation available"
            elif not rl_patches and ti_patches:
                final_patches = ti_patches
                strategy_used = "TI_Only"
                reasoning = "No RL recommendation available"
            elif not ti_patches and not rl_patches:
                final_patches = []
                strategy_used = "No_Recommendation"
                reasoning = "No recommendations from either strategy"
            
            # Record decision
            hybrid_decision = HybridDecision(
                action=final_patches,
                confidence=combined_score,
                strategy_used=strategy_used,
                threat_intelligence_score=ti_score,
                rl_score=rl_score,
                reasoning=reasoning
            )
            
            self._record_decision(ti_patches, rl_patches, hybrid_decision)
            
            return final_patches
            
        except Exception as e:
            logging.error(f"Error in hybrid strategy select_patches: {e}")
            # Fallback to threat intelligence strategy
            return self.threat_intelligence.select_patches(state, remaining_budget, current_step, total_steps)
    
    def _calculate_simple_score(self, patches: List[Tuple], state) -> float:
        """Calculate a simple score for patches"""
        if not patches:
            return 0.0
        
        total_score = 0.0
        for vuln, cost in patches:
            # Simple scoring based on CVSS and cost
            cvss = getattr(vuln, 'cvss', 5.0)
            cost_factor = 1.0 / max(cost, 1.0)  # Lower cost is better
            score = cvss * cost_factor
            total_score += score
        
        return total_score / len(patches) if patches else 0.0
    
    def _record_decision(self, ti_patches: List[Tuple], rl_patches: List[Tuple], 
                        hybrid_decision: HybridDecision) -> None:
        """Record the hybrid decision for analysis"""
        decision_record = {
            'step': len(self.decision_history),
            'ti_patches_count': len(ti_patches),
            'rl_patches_count': len(rl_patches),
            'hybrid_decision': {
                'action_count': len(hybrid_decision.action) if hybrid_decision.action else 0,
                'confidence': hybrid_decision.confidence,
                'strategy_used': hybrid_decision.strategy_used,
                'reasoning': hybrid_decision.reasoning
            },
            'scores': {
                'ti_score': hybrid_decision.threat_intelligence_score,
                'rl_score': hybrid_decision.rl_score
            }
        }
        
        self.decision_history.append(decision_record)
        
        # Update performance metrics
        self.threat_intelligence_performance.append(hybrid_decision.threat_intelligence_score)
        self.rl_performance.append(hybrid_decision.rl_score)
        self.hybrid_performance.append(hybrid_decision.confidence)
        
        # Adapt weights periodically
        if len(self.decision_history) % 5 == 0:
            self._adapt_weights()
    
    def _adapt_weights(self) -> None:
        """Adapt weights based on recent performance"""
        if len(self.threat_intelligence_performance) < 3 or len(self.rl_performance) < 3:
            return
        
        # Calculate recent performance averages
        recent_ti_perf = np.mean(self.threat_intelligence_performance[-3:])
        recent_rl_perf = np.mean(self.rl_performance[-3:])
        
        # Adjust weights based on performance
        if recent_ti_perf > recent_rl_perf:
            # Threat intelligence performing better
            self.threat_intelligence_weight = min(0.8, self.threat_intelligence_weight + self.adaptation_rate)
            self.rl_weight = max(0.2, self.rl_weight - self.adaptation_rate)
        else:
            # RL performing better
            self.rl_weight = min(0.8, self.rl_weight + self.adaptation_rate)
            self.threat_intelligence_weight = max(0.2, self.threat_intelligence_weight - self.adaptation_rate)
        
        self.adaptations += 1
        logging.info(f"Adapted weights: TI={self.threat_intelligence_weight:.3f}, RL={self.rl_weight:.3f}")
    
    def get_strategy_metrics(self) -> Dict:
        """Get metrics about the hybrid strategy performance"""
        return {
            'hybrid_adaptations': self.adaptations,
            'final_ti_weight': self.threat_intelligence_weight,
            'final_rl_weight': self.rl_weight,
            'average_confidence': np.mean([d['hybrid_decision']['confidence'] for d in self.decision_history]) if self.decision_history else 0.0,
            'decision_history_length': len(self.decision_history),
            'ti_performance': np.mean(self.threat_intelligence_performance) if self.threat_intelligence_performance else 0.0,
            'rl_performance': np.mean(self.rl_performance) if self.rl_performance else 0.0,
            'hybrid_performance': np.mean(self.hybrid_performance) if self.hybrid_performance else 0.0
        }
    
    def get_explanation(self, action: List[Tuple]) -> str:
        """Get explanation for the hybrid strategy's decision"""
        if not self.decision_history:
            return "No decisions recorded yet"
        
        latest_decision = self.decision_history[-1]
        return f"Hybrid Strategy Decision: {latest_decision['hybrid_decision']['reasoning']}" 