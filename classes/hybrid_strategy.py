#!/usr/bin/env python3
"""
Enhanced Three-Tier Hybrid Strategy
===================================

A sophisticated hybrid cybersecurity defense strategy that implements a three-tier approach:
- Tier 1: TI-Driven Emergency Response (Zero-days, TI critical alerts)
- Tier 2: RL-Optimized Normal Operations (Cost-efficient day-to-day patching)
- Tier 3: Hybrid Coordination (Balanced decision making)

This strategy provides clear escalation protocols and leverages the strengths of both TI and RL.
"""

import random
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from .patching_strategies import ThreatIntelligenceStrategy
from src.RL_defender_strategy import RLAdaptiveThreatIntelligenceStrategy

class TierLevel(Enum):
    """Enumeration for the three tiers of operation"""
    EMERGENCY = 1    # Tier 1: TI-Driven Emergency Response
    NORMAL = 2       # Tier 2: RL-Optimized Normal Operations  
    HYBRID = 3       # Tier 3: Hybrid Coordination

@dataclass
class HybridDecision:
    """Container for hybrid decision results"""
    action: Optional[List[Tuple]]
    confidence: float
    strategy_used: str
    tier_level: TierLevel
    threat_intelligence_score: float
    rl_score: float
    reasoning: str
    emergency_triggers: List[str]
    budget_multiplier: float

class EnhancedHybridStrategy:
    """
    Enhanced Three-Tier Hybrid Strategy
    
    This strategy implements a sophisticated three-tier approach:
    
    Tier 1: TI-Driven Emergency Response
    - Always patch: Zero-days, TI critical alerts
    - Budget: Allow 2x normal spending for emergencies
    - Trigger: Any TI critical alert or zero-day detection
    
    Tier 2: RL-Optimized Normal Operations
    - Use RL: For cost-efficient day-to-day patching
    - Enhanced with TI features: Threat levels as state inputs
    - Target: Maintain low cost while maximizing protection
    
    Tier 3: Hybrid Coordination
    - TI dominance: When >2 critical alerts detected
    - RL dominance: During normal operations
    - Balanced: For moderate threat levels
    """
    
    def __init__(self, threat_intelligence_weight: float = 0.4, rl_weight: float = 0.6, 
                 confidence_threshold: float = 0.7, adaptation_rate: float = 0.1,
                 emergency_budget_multiplier: float = 2.0, critical_alert_threshold: int = 2):
        """
        Initialize enhanced hybrid strategy
        
        Args:
            threat_intelligence_weight: Initial weight for threat intelligence (0-1)
            rl_weight: Initial weight for RL defender (0-1)
            confidence_threshold: Threshold for RL decisions
            adaptation_rate: Rate at which weights adapt
            emergency_budget_multiplier: Budget multiplier for emergency responses
            critical_alert_threshold: Number of critical alerts to trigger hybrid mode
        """
        # Validate weights
        if abs(threat_intelligence_weight + rl_weight - 1.0) > 0.01:
            raise ValueError("Weights must sum to 1.0")
        
        self.threat_intelligence_weight = threat_intelligence_weight
        self.rl_weight = rl_weight
        self.confidence_threshold = confidence_threshold
        self.adaptation_rate = adaptation_rate
        self.emergency_budget_multiplier = emergency_budget_multiplier
        self.critical_alert_threshold = critical_alert_threshold
        
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
        
        # Tier tracking
        self.current_tier = TierLevel.NORMAL
        self.tier_history = []
        
        # Emergency tracking
        self.emergency_triggers = []
        self.critical_alerts_count = 0
        
        logging.info(f"Enhanced Hybrid Strategy initialized with TI weight: {threat_intelligence_weight}, RL weight: {rl_weight}")
        logging.info(f"Emergency budget multiplier: {emergency_budget_multiplier}x, Critical alert threshold: {critical_alert_threshold}")
    
    def initialize(self, state, cost_cache):
        """Initialize both component strategies"""
        self._cost_cache = cost_cache
        self.threat_intelligence.initialize(state, cost_cache)
        self.rl_defender.initialize(state, cost_cache)
        
        # Set state for both strategies
        self.threat_intelligence.state = state
        self.rl_defender.state = state
        
        # Reset tier tracking
        self.current_tier = TierLevel.NORMAL
        self.emergency_triggers = []
        self.critical_alerts_count = 0
        
        logging.info("Enhanced Hybrid Strategy initialized with system state")
    
    def _determine_tier_level(self, state, remaining_budget: float, current_step: int) -> TierLevel:
        """
        Determine which tier to operate in based on current conditions
        
        Returns:
            TierLevel: The appropriate tier for current conditions
        """
        # Check for emergency conditions (Tier 1)
        emergency_conditions = self._check_emergency_conditions(state)
        if emergency_conditions:
            self.emergency_triggers = emergency_conditions
            logging.info(f"🚨 EMERGENCY TIER ACTIVATED: {emergency_conditions}")
            return TierLevel.EMERGENCY
        
        # Check for hybrid coordination conditions (Tier 3)
        if self.critical_alerts_count >= self.critical_alert_threshold:
            logging.info(f"🔄 HYBRID TIER ACTIVATED: {self.critical_alerts_count} critical alerts")
            return TierLevel.HYBRID
        
        # Default to normal operations (Tier 2)
        return TierLevel.NORMAL
    
    def _check_emergency_conditions(self, state) -> List[str]:
        """
        Check for emergency conditions that require immediate TI-driven response
        
        Returns:
            List of emergency trigger descriptions
        """
        emergency_triggers = []
        
        # Check for zero-day vulnerabilities
        zero_day_count = 0
        for asset in state.system.assets:
            for component in asset.components:
                for vuln in component.vulnerabilities:
                    if not vuln.is_patched and self.threat_intelligence.is_zero_day_vulnerability(vuln):
                        zero_day_count += 1
                        emergency_triggers.append(f"Zero-day detected: {vuln.cve_id}")
        
        # Check for critical threat intelligence alerts
        critical_threat_assets = []
        for asset_id, threat_level in self.threat_intelligence.asset_threat_levels.items():
            if threat_level >= 0.9:  # Very high threat level
                critical_threat_assets.append(asset_id)
                emergency_triggers.append(f"Critical TI alert: Asset {asset_id} (threat: {threat_level:.2f})")
        
        # Check for recent successful attacks
        recent_successful_attacks = [obs for obs in self.threat_intelligence.attack_observations[-3:] 
                                   if obs.get('success', False)]
        if len(recent_successful_attacks) >= 2:
            emergency_triggers.append(f"Multiple recent successful attacks: {len(recent_successful_attacks)}")
        
        # Update critical alerts count
        self.critical_alerts_count = len(emergency_triggers)
        
        return emergency_triggers
    
    def _tier_1_emergency_response(self, state, remaining_budget: float, current_step: int, total_steps: int) -> List[Tuple]:
        """
        Tier 1: TI-Driven Emergency Response
        
        - Always patch: Zero-days, TI critical alerts
        - Budget: Allow 2x normal spending for emergencies
        - Trigger: Any TI critical alert or zero-day detection
        """
        logging.info("🆘 TIER 1: Executing TI-Driven Emergency Response")
        
        # Use emergency budget multiplier
        emergency_budget = remaining_budget * self.emergency_budget_multiplier
        
        # Get TI recommendations with emergency budget
        ti_patches = self.threat_intelligence.select_patches(state, emergency_budget, current_step, total_steps)
        
        # Prioritize zero-days and critical alerts
        prioritized_patches = self._prioritize_emergency_patches(ti_patches, state)
        
        logging.info(f"Emergency response: {len(prioritized_patches)} patches selected with {emergency_budget:.2f} budget")
        
        return prioritized_patches
    
    def _tier_2_normal_operations(self, state, remaining_budget: float, current_step: int, total_steps: int) -> List[Tuple]:
        """
        Tier 2: RL-Optimized Normal Operations
        
        - Use RL: For cost-efficient day-to-day patching
        - Enhanced with TI features: Threat levels as state inputs
        - Target: Maintain low cost while maximizing protection
        """
        logging.info("⚙️ TIER 2: Executing RL-Optimized Normal Operations")
        
        # Update RL defender with current threat intelligence features
        self.rl_defender.update_threat_intelligence_features(self.threat_intelligence)
        
        # Get RL recommendations
        rl_patches = self.rl_defender.select_patches(state, remaining_budget, current_step, total_steps)
        
        logging.info(f"Normal operations: {len(rl_patches)} patches selected with {remaining_budget:.2f} budget")
        
        return rl_patches
    
    def _tier_3_hybrid_coordination(self, state, remaining_budget: float, current_step: int, total_steps: int) -> List[Tuple]:
        """
        Tier 3: Hybrid Coordination
        
        - TI dominance: When >2 critical alerts detected
        - RL dominance: During normal operations
        - Balanced: For moderate threat levels
        """
        logging.info("🔄 TIER 3: Executing Hybrid Coordination")
        
        # Get recommendations from both strategies
        ti_patches = self.threat_intelligence.select_patches(state, remaining_budget, current_step, total_steps)
        rl_patches = self.rl_defender.select_patches(state, remaining_budget, current_step, total_steps)
        
        # Calculate scores for each strategy
        ti_score = self._calculate_simple_score(ti_patches, state)
        rl_score = self._calculate_simple_score(rl_patches, state)
        
        # Determine coordination strategy based on critical alerts
        if self.critical_alerts_count > 2:
            # TI dominance for high critical alerts
            final_patches = ti_patches
            strategy_used = "TI_Dominant"
            reasoning = f"High critical alerts ({self.critical_alerts_count}): TI dominance"
        elif self.critical_alerts_count == 2:
            # Balanced approach for moderate alerts
            combined_score = (ti_score * self.threat_intelligence_weight) + (rl_score * self.rl_weight)
            if combined_score > self.confidence_threshold:
                final_patches = rl_patches
                strategy_used = "RL_Balanced"
            else:
                final_patches = ti_patches
                strategy_used = "TI_Balanced"
            reasoning = f"Moderate alerts ({self.critical_alerts_count}): Balanced approach"
        else:
            # RL dominance for low alerts
            final_patches = rl_patches
            strategy_used = "RL_Dominant"
            reasoning = f"Low alerts ({self.critical_alerts_count}): RL dominance"
        
        logging.info(f"Hybrid coordination: {strategy_used} - {len(final_patches)} patches selected")
        
        return final_patches
    
    def _prioritize_emergency_patches(self, patches: List[Tuple], state) -> List[Tuple]:
        """
        Prioritize patches for emergency response, focusing on zero-days and critical alerts
        """
        if not patches:
            return patches
        
        # Separate zero-days and regular patches
        zero_day_patches = []
        critical_patches = []
        regular_patches = []
        
        for vuln, cost in patches:
            if self.threat_intelligence.is_zero_day_vulnerability(vuln):
                zero_day_patches.append((vuln, cost))
            elif self._is_critical_alert(vuln, state):
                critical_patches.append((vuln, cost))
            else:
                regular_patches.append((vuln, cost))
        
        # Return prioritized list: zero-days first, then critical alerts, then regular
        prioritized = zero_day_patches + critical_patches + regular_patches
        
        logging.info(f"Emergency prioritization: {len(zero_day_patches)} zero-days, {len(critical_patches)} critical, {len(regular_patches)} regular")
        
        return prioritized
    
    def _is_critical_alert(self, vuln, state) -> bool:
        """Check if a vulnerability represents a critical alert"""
        # Check if vulnerability is on a high-threat asset
        for asset in state.system.assets:
            for component in asset.components:
                if vuln in component.vulnerabilities:
                    asset_id = str(asset.asset_id)
                    threat_level = self.threat_intelligence.asset_threat_levels.get(asset_id, 0.3)
                    if threat_level >= 0.8:  # High threat threshold
                        return True
        
        # Check if vulnerability has high CVSS score
        if getattr(vuln, 'cvss', 0) >= 9.0:
            return True
        
        return False
    
    def select_patches(self, state, remaining_budget: float, current_step: int, total_steps: int) -> List[Tuple]:
        """
        Select patches using enhanced three-tier hybrid approach
        
        Args:
            state: Current system state
            remaining_budget: Available budget
            current_step: Current simulation step
            total_steps: Total simulation steps
            
        Returns:
            List of tuples (vulnerability, cost)
        """
        try:
            # Determine current tier level
            tier_level = self._determine_tier_level(state, remaining_budget, current_step)
            self.current_tier = tier_level
            
            # Execute appropriate tier strategy
            if tier_level == TierLevel.EMERGENCY:
                final_patches = self._tier_1_emergency_response(state, remaining_budget, current_step, total_steps)
                strategy_used = "TI_Emergency"
                budget_multiplier = self.emergency_budget_multiplier
            elif tier_level == TierLevel.NORMAL:
                final_patches = self._tier_2_normal_operations(state, remaining_budget, current_step, total_steps)
                strategy_used = "RL_Normal"
                budget_multiplier = 1.0
            else:  # TierLevel.HYBRID
                final_patches = self._tier_3_hybrid_coordination(state, remaining_budget, current_step, total_steps)
                strategy_used = "Hybrid_Coordination"
                budget_multiplier = 1.0
            
            # Calculate scores for tracking
            ti_score = self._calculate_simple_score(
                self.threat_intelligence.select_patches(state, remaining_budget, current_step, total_steps), 
                state
            )
            rl_score = self._calculate_simple_score(
                self.rl_defender.select_patches(state, remaining_budget, current_step, total_steps), 
                state
            )
            
            # Record decision
            hybrid_decision = HybridDecision(
                action=final_patches,
                confidence=max(ti_score, rl_score),
                strategy_used=strategy_used,
                tier_level=tier_level,
                threat_intelligence_score=ti_score,
                rl_score=rl_score,
                reasoning=f"Tier {tier_level.value}: {strategy_used}",
                emergency_triggers=self.emergency_triggers.copy(),
                budget_multiplier=budget_multiplier
            )
            
            self._record_decision(final_patches, hybrid_decision)
            
            # Log tier information
            logging.info(f"Tier {tier_level.value} ({tier_level.name}): {len(final_patches)} patches, budget multiplier: {budget_multiplier}x")
            
            return final_patches
            
        except Exception as e:
            logging.error(f"Error in enhanced hybrid strategy select_patches: {e}")
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
    
    def _record_decision(self, patches: List[Tuple], hybrid_decision: HybridDecision) -> None:
        """Record the hybrid decision for analysis"""
        decision_record = {
            'step': len(self.decision_history),
            'tier_level': hybrid_decision.tier_level.value,
            'tier_name': hybrid_decision.tier_level.name,
            'patches_count': len(patches) if patches else 0,
            'hybrid_decision': {
                'confidence': hybrid_decision.confidence,
                'strategy_used': hybrid_decision.strategy_used,
                'reasoning': hybrid_decision.reasoning,
                'budget_multiplier': hybrid_decision.budget_multiplier
            },
            'scores': {
                'ti_score': hybrid_decision.threat_intelligence_score,
                'rl_score': hybrid_decision.rl_score
            },
            'emergency_triggers': hybrid_decision.emergency_triggers
        }
        
        self.decision_history.append(decision_record)
        self.tier_history.append(hybrid_decision.tier_level)
        
        # Update performance metrics
        self.threat_intelligence_performance.append(hybrid_decision.threat_intelligence_score)
        self.rl_performance.append(hybrid_decision.rl_score)
        self.hybrid_performance.append(hybrid_decision.confidence)
    
    def _adapt_weights(self) -> None:
        """Adapt weights based on performance"""
        if len(self.threat_intelligence_performance) < 5 or len(self.rl_performance) < 5:
            return
        
        # Calculate recent performance averages
        recent_ti_perf = np.mean(self.threat_intelligence_performance[-5:])
        recent_rl_perf = np.mean(self.rl_performance[-5:])
        
        # Calculate performance ratios
        total_perf = recent_ti_perf + recent_rl_perf
        if total_perf > 0:
            ti_ratio = recent_ti_perf / total_perf
            rl_ratio = recent_rl_perf / total_perf
            
            # Adapt weights
            self.threat_intelligence_weight = (self.threat_intelligence_weight * (1 - self.adaptation_rate) + 
                                             ti_ratio * self.adaptation_rate)
            self.rl_weight = (self.rl_weight * (1 - self.adaptation_rate) + 
                            rl_ratio * self.adaptation_rate)
            
            # Normalize weights
            total_weight = self.threat_intelligence_weight + self.rl_weight
            self.threat_intelligence_weight /= total_weight
            self.rl_weight /= total_weight
            
            self.adaptations += 1
            
            logging.info(f"Weights adapted: TI={self.threat_intelligence_weight:.3f}, RL={self.rl_weight:.3f}")
    
    def get_strategy_metrics(self) -> Dict:
        """Get comprehensive strategy metrics"""
        return {
            'current_tier': self.current_tier.value,
            'tier_distribution': {
                'emergency': sum(1 for tier in self.tier_history if tier == TierLevel.EMERGENCY),
                'normal': sum(1 for tier in self.tier_history if tier == TierLevel.NORMAL),
                'hybrid': sum(1 for tier in self.tier_history if tier == TierLevel.HYBRID)
            },
            'emergency_triggers': self.emergency_triggers,
            'critical_alerts_count': self.critical_alerts_count,
            'weights': {
                'threat_intelligence': self.threat_intelligence_weight,
                'rl': self.rl_weight
            },
            'performance': {
                'threat_intelligence': np.mean(self.threat_intelligence_performance) if self.threat_intelligence_performance else 0,
                'rl': np.mean(self.rl_performance) if self.rl_performance else 0,
                'hybrid': np.mean(self.hybrid_performance) if self.hybrid_performance else 0
            },
            'adaptations': self.adaptations,
            'decision_history_length': len(self.decision_history)
        }
    
    def get_explanation(self, action: List[Tuple]) -> str:
        """Get explanation for the current action"""
        if not self.decision_history:
            return "No decisions recorded yet"
        
        latest_decision = self.decision_history[-1]
        tier_name = latest_decision['tier_name']
        strategy_used = latest_decision['hybrid_decision']['strategy_used']
        reasoning = latest_decision['hybrid_decision']['reasoning']
        
        explanation = f"Tier {latest_decision['tier_level']} ({tier_name}): {strategy_used}\n"
        explanation += f"Reasoning: {reasoning}\n"
        explanation += f"Patches selected: {latest_decision['patches_count']}\n"
        
        if latest_decision['emergency_triggers']:
            explanation += f"Emergency triggers: {', '.join(latest_decision['emergency_triggers'])}\n"
        
        return explanation

# Backward compatibility alias
HybridStrategy = EnhancedHybridStrategy 