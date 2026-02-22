# CyGATE Experiment Results Update

## 1. Noise Robustness Experiment (Action Point 3)

### Implementation

A `NoisySimulationRunner` subclass of `APT3SimulationRunner` was created (`experiments/run_noise_experiment.py`). It injects noise at three points in the simulation pipeline without modifying any core classes:

- **False negatives:** Overrides `_create_attack_observation()`. After the parent creates a real observation, `NoisyObservationModel.apply_fn()` drops it with probability `fn_rate` (10–30%).
- **False positives:** At the end of each `run_step()`, `NoisyObservationModel.generate_fp_observations()` creates phantom observations — one per asset with probability `fp_rate` (40–90%) — with randomized attack types, techniques, and detection confidence. These are fed into the defender's observation stream alongside real observations.
- **Correlated noise:** `NoisyObservationModel` builds a spatial covariance matrix from the network adjacency graph and samples temporally correlated noise via an AR(1) process. This is applied to `detection_confidence` fields on all observations. The defender's base `detection_confidence` is also degraded proportionally.

A `NoiseConfig` dataclass (`classes/noise_model.py`) parametrizes each configuration. The experiment sweeps a grid of 6 FP rates × 5 FN rates × 3 correlation levels = 90 configurations, running all 7 strategies with 100 trials each, using the existing `apt3_scenario_enriched.json` topology.

### Results

All strategies are highly robust to observation noise. The worst-case degradation in value preserved from low noise (FP=40%, FN=10%) to extreme noise (FP=90%, FN=30%) is only **1.3%** (Business Value). Most strategies degrade <1%, and some slightly improve (Threat Intelligence: −0.5%, Cost-Benefit: −0.4%). Sensor correlation (0.0 to 0.6) has negligible impact (~1.4% max shift, no consistent trend).

This robustness stems from CyGATE's design: patch prioritization is driven by structural properties (CVSS, kill chain stage, business value) rather than real-time detection signals. The noise primarily perturbs the `detection_confidence` belief parameter, which plays a secondary role in the core patching decision loop.

---

## 2. Scalability Benchmark (Action Point 4)

### Implementation

An `EnterpriseTopologyGenerator` (`classes/topology_generator.py`) creates synthetic network topologies following the Purdue model architecture: Corporate IT (40%), DMZ (10%), Control Network (30%), Field Devices (20%). It assigns realistic asset types (domain controllers, HMIs, PLCs, etc.), business values, inter-zone connections, and vulnerabilities drawn from a pool of 17 real CVEs with accurate CVSS/EPSS scores. Each topology is written to a temp file and loaded by the simulation's standard data loader.

`ProfiledSimulationRunner` (`experiments/run_scalability_benchmark.py`) subclasses `APT3SimulationRunner` and monkey-patches `attacker.attack_graph.build_attack_graph` with a counting wrapper. Each trial runs the step-by-step loop with `run_step()` followed by `update_attack_graph()`, measuring per-step wall-clock latency via `time.perf_counter()` and peak memory via `tracemalloc`. Budgets are scaled proportionally to total business value.

The benchmark sweeps 6 topology sizes (20, 50, 100, 200, 500, 1000 assets) across all 7 strategies with 25 trials and 25 steps per trial.

### Results

| Size | Assets | Vulns | Conns  | Avg Latency | Peak RAM | Feasibility |
|------|--------|-------|--------|-------------|----------|-------------|
| S    | 20     | 76    | 67     | 18 ms       | 0.7 MB   | Real-time   |
| S    | 50     | 213   | 169    | 52 ms       | 1.7 MB   | Real-time   |
| M    | 100    | 420   | 338    | 122 ms      | 3.6 MB   | Near-RT     |
| M    | 200    | 816   | 675    | 308 ms      | 6.9 MB   | Near-RT     |
| L    | 500    | 2,090 | 1,687  | 1,393 ms    | 20.8 MB  | Batch       |
| XL   | 1,000  | 4,193 | 3,356  | 3,922 ms    | 33.2 MB  | Batch       |

**Memory scales linearly** (~O(n)): ~48× growth for a 50× increase in assets. Peak RAM at 1000 assets is only 30–40 MB. **Latency scales super-linearly** but remains tractable — the log-log runtime plot shows roughly O(n^~1.5). Cost-Benefit and Business Value are the most efficient at scale (~1.8–2.4s/step at 1000 assets); CVSS-Only and Hybrid Defender are the most expensive (~5.8–6.9s/step). RL Defender and Hybrid Defender use ~30–35% more memory than the static strategies due to their additional state tracking (Q-tables, extended belief vectors).

Planner calls (attack-graph rebuilds) are constant at 25/trial (1 per step) regardless of topology size, meaning the rebuild is triggered every step. The cost of each rebuild scales with network size (captured in the latency figures), but the frequency doesn't increase — this is expected since `update_attack_graph()` checks for any state change and the attacker acts every step.
