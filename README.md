
<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/Yuni0217/APTArena">
  </a>
  <br />

  <!-- Badges -->
  <img src="https://img.shields.io/github/repo-size/Yuning-J/APTArena?style=for-the-badge" alt="GitHub repo size" height="25">
  <img src="https://img.shields.io/github/last-commit/Yuning-J/APTArena?style=for-the-badge" alt="GitHub last commit" height="25">
  <img src="https://img.shields.io/github/license/Yuning-J/APTArena?style=for-the-badge" alt="License" height="25">
  <br />
  
  <h3 align="center">APTArena</h3>
  <p align="center">
    Game Theory based APT Attack and Defense Evaluation.
 
  </p>
</p>


## Overview
Cybersecurity simulation arena for testing and benchmarking defense strategies against APTs using RL-based, hybrid, and traditional approaches.

## Features
- Simulation of APT3-style attack scenarios on enterprise networks
- Multiple defender strategies: Hybrid (CyGATE), CVSS, Business Value, Cost-Benefit, and more
- Reinforcement Learning (RL) component for adaptive defense
- Threat intelligence integration using our project [CVE-KGRAG](https://github.com/Yuning-J/CVE-KGRAG) 
- Comprehensive result analysis and visualizations
- Modular codebase for research and extension


## Installation
1. **Clone the repository:**
   ```sh
   git clone https://github.com/Yuning-J/APTArena.git
   cd APTArena
   ```
2. **Set up a Python environment:**
   ```sh
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt  # 
   ```

## Usage
### 1. **Run a Simulation**
To run simulation with the Hybrid (CyGATE) strategy and baselines for 100-trial experiments:
```sh
python src/apt3_simulation_main.py --data_file data/systemData/apt3_scenario_enriched.json ----num-trials 100  
```


### 2. **Train the RL Component**
To train the RL component for the Hybrid strategy:
```sh
python src/RL_defender_simulation.py --data_file data/systemData/apt3_scenario_enriched.json --num_episodes 500 --num_steps 50 --defender_budget 7500
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
If you use APTArena in your research, please cite:
```
@article{YOUR_CITATION,
  title={CyGATE: Cybersecurity Game-based Attack and Threat Evaluation},
  author={Your Name et al.},
  journal={...},
  year={2024}
}
```
