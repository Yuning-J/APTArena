# CyGATE: Cybersecurity Game-based Attack and Threat Evaluation

## Overview
CyGATE is a research-oriented cybersecurity simulation platform for evaluating and comparing defense strategies against advanced persistent threats (APTs) in realistic enterprise environments. It supports hybrid, RL-based, and traditional patching strategies, and provides detailed analysis and visualization tools for reproducible experiments.

## Features
- Simulation of APT3-style attack scenarios on enterprise networks
- Multiple defender strategies: Hybrid (CyGATE), CVSS, Business Value, Cost-Benefit, and more
- Reinforcement Learning (RL) component for adaptive defense
- Threat intelligence integration
- Comprehensive result analysis and visualizations
- Modular codebase for research and extension

## Folder Structure
```
CyGATE/
  classes/                # Core simulation, attacker, defender, and strategy classes
  src/                    # Main simulation, RL training, and analysis scripts
  data/                   # Scenario, vulnerability, and threat intelligence data
  data_collection/        # Scripts for data gathering and preprocessing
  results/                # Output and visualization files
```

## Installation
1. **Clone the repository:**
   ```sh
   git clone https://github.com/YOUR_USERNAME/CyGATE.git
   cd CyGATE
   ```
2. **Set up a Python environment:**
   ```sh
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt  # (create this file if needed)
   ```
3. **Install additional dependencies:**
   - See the `requirements.txt` or `environment.yml` for details.

## Usage
### 1. **Run a Simulation**
To run a simulation with the Hybrid (CyGATE) strategy and baselines:
```sh
python src/apt3_rtu_simulation.py --data_file data/systemData/apt3_scenario_enriched.json --num_steps 50 --defender_budget 100000
```
Or for 100-trial experiments:
```sh
python src/apt3_rtu_simulation_100.py --data_file data/systemData/apt3_scenario_enriched.json --num_steps 50 --num_trials 100 --defender_budget 100000
```

### 2. **Train the RL Component**
To train the RL component for the Hybrid strategy:
```sh
python src/RL_defender_simulation.py --data_file data/systemData/apt3_scenario_enriched.json --num_episodes 500 --num_steps 50 --defender_budget 100000
```
The trained Q-table will be saved in `src/rl_defender_training_results/`.

### 3. **Analyze Results**
Use the analysis scripts to generate plots and tables:
```sh
python src/analyze_asset_protection.py
python src/simulation_analysis_systematic.py
```

## Customization
- **Scenario files:** Edit or add JSON files in `data/systemData/`.
- **Strategies:** Implement new strategies in `classes/patching_strategies.py` or `classes/hybrid_strategy.py`.
- **Visualization:** Modify or extend plotting in the analysis scripts in `src/`.

## Citing CyGATE
If you use CyGATE in your research, please cite:
```
@article{YOUR_CITATION,
  title={CyGATE: Cybersecurity Game-based Attack and Threat Evaluation},
  author={Your Name et al.},
  journal={...},
  year={2024}
}
```

## License
Specify your license here (e.g., MIT, Apache 2.0).

## Contact
For questions or contributions, please open an issue or contact the maintainer at your.email@domain.com. 