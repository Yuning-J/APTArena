"""
Noise injection model for robustness experiments.

Provides configurable false-positive / false-negative rates and spatially /
temporally correlated sensor noise that wraps the simulation's existing
observation pipeline.
"""

import random
import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class NoiseConfig:
    """Configuration for observation noise injection."""

    fp_rate: float = 0.0
    fn_rate: float = 0.0
    sensor_correlation: float = 0.0
    temporal_correlation: float = 0.3
    noise_seed: Optional[int] = None

    def __post_init__(self):
        for name, val, lo, hi in [
            ("fp_rate", self.fp_rate, 0.0, 1.0),
            ("fn_rate", self.fn_rate, 0.0, 1.0),
            ("sensor_correlation", self.sensor_correlation, 0.0, 1.0),
            ("temporal_correlation", self.temporal_correlation, 0.0, 1.0),
        ]:
            if not (lo <= val <= hi):
                raise ValueError(f"{name} must be in [{lo}, {hi}], got {val}")

    def label(self) -> str:
        return f"FP{self.fp_rate:.0%}_FN{self.fn_rate:.0%}_corr{self.sensor_correlation}"


class NoisyObservationModel:
    """
    Wraps the simulation observation pipeline and injects configurable noise.

    * **False negatives** – real attack observations are dropped with
      probability ``fn_rate``.
    * **False positives** – phantom observations are generated with
      probability ``fp_rate`` per asset per step.
    * **Correlated noise** – detection-confidence values are perturbed by
      noise drawn from a multivariate normal whose covariance encodes
      spatial (network-adjacency) and temporal (AR(1)) correlation.
    """

    def __init__(self, config: NoiseConfig, system: Any):
        self.config = config
        self.rng = np.random.default_rng(config.noise_seed)
        self._py_rng = random.Random(config.noise_seed)

        self._asset_ids: List[str] = [
            str(a.asset_id) for a in system.assets if str(a.asset_id) != "internet"
        ]
        self._n = len(self._asset_ids)
        self._asset_idx: Dict[str, int] = {
            aid: i for i, aid in enumerate(self._asset_ids)
        }

        self._cov = self._build_covariance(system)
        self._prev_noise: np.ndarray = np.zeros(self._n)

    # ------------------------------------------------------------------
    # Covariance construction
    # ------------------------------------------------------------------
    def _build_covariance(self, system: Any) -> np.ndarray:
        """Build spatial covariance matrix from network adjacency."""
        rho = self.config.sensor_correlation
        cov = np.eye(self._n)
        if rho <= 0.0 or self._n < 2:
            return cov

        adj: Set[Tuple[str, str]] = set()
        for conn in system.connections:
            a = str(conn.from_asset.asset_id)
            b = str(conn.to_asset.asset_id)
            adj.add((a, b))
            adj.add((b, a))

        for (a, b) in adj:
            i = self._asset_idx.get(a)
            j = self._asset_idx.get(b)
            if i is not None and j is not None and i != j:
                cov[i, j] = rho
                cov[j, i] = rho

        # Ensure positive-semi-definite
        eigvals = np.linalg.eigvalsh(cov)
        if eigvals.min() < 0:
            cov += (abs(eigvals.min()) + 1e-6) * np.eye(self._n)
        return cov

    def _sample_correlated_noise(self) -> np.ndarray:
        """Sample spatially + temporally correlated noise vector."""
        spatial = self.rng.multivariate_normal(np.zeros(self._n), self._cov)
        alpha = self.config.temporal_correlation
        combined = alpha * self._prev_noise + math.sqrt(1 - alpha ** 2) * spatial
        self._prev_noise = combined
        return combined

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def apply_fn(self, observation: Optional[Dict]) -> Optional[Dict]:
        """Apply false-negative: drop a real observation with probability fn_rate."""
        if observation is None:
            return None
        if self._py_rng.random() < self.config.fn_rate:
            logger.debug("Noise: dropped observation (FN)")
            return None
        return observation

    def generate_fp_observations(
        self, step: int, system: Any
    ) -> List[Dict]:
        """Generate false-positive phantom observations for this step."""
        fps: List[Dict] = []
        if self.config.fp_rate <= 0.0:
            return fps

        fp_techniques = ["T1190", "T1566", "T1068", "T1203", "T1021", "T1563"]
        fp_actions = [
            "initial_access", "exploitation", "lateral_movement", "reconnaissance",
        ]

        for asset in system.assets:
            aid = str(asset.asset_id)
            if aid == "internet":
                continue
            if self._py_rng.random() < self.config.fp_rate:
                obs = {
                    "step": step,
                    "action_type": self._py_rng.choice(fp_actions),
                    "target_asset": aid,
                    "success": self._py_rng.random() < 0.3,
                    "timestamp": step,
                    "reason": "false_positive",
                    "detection_confidence": max(
                        0.1, min(1.0, 0.5 + self._py_rng.gauss(0, 0.15))
                    ),
                    "techniques": [self._py_rng.choice(fp_techniques)],
                    "tactic": "Unknown",
                    "is_phantom": True,
                }
                fps.append(obs)
        return fps

    def corrupt_detection_confidence(self, observation: Dict) -> Dict:
        """Add correlated noise to the detection_confidence field."""
        if observation is None:
            return observation

        aid = observation.get("target_asset", "")
        idx = self._asset_idx.get(aid)
        if idx is None:
            return observation

        noise_vec = self._sample_correlated_noise()
        noise_val = noise_vec[idx] * 0.2

        orig = observation.get("detection_confidence", 0.7)
        observation["detection_confidence"] = max(0.05, min(1.0, orig + noise_val))
        return observation

    def degrade_defender_detection_confidence(self, base_confidence: float) -> float:
        """Scale the defender's base detection_confidence by noise parameters."""
        degraded = base_confidence * (1.0 - self.config.fn_rate * 0.5)
        return max(0.1, degraded)

    def degrade_observation_accuracy(self, base_accuracy: float) -> float:
        """Return degraded observation accuracy for belief updates."""
        degraded = base_accuracy * (1.0 - self.config.fp_rate * 0.4)
        return max(0.5, degraded)

    def reset(self):
        """Reset temporal state between trials."""
        self._prev_noise = np.zeros(self._n)
        self.rng = np.random.default_rng(self.config.noise_seed)
        self._py_rng = random.Random(self.config.noise_seed)
