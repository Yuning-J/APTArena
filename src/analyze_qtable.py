#!/usr/bin/env python3
"""
Analyze Q-table and training summary for RL Defender
"""

import pickle
import json
import os

def analyze_qtable():
    q_table_path = 'src/rl_defender_training_results/rl_defender_training_20250629_094929/q_table.pkl'
    summary_path = 'src/rl_defender_training_results/rl_defender_training_20250629_094929/training_summary.json'
    
    print('=== Q-TABLE ANALYSIS (IMPROVED PARAMETERS) ===')
    print(f'Q-table size: {os.path.getsize(q_table_path)} bytes')
    
    # Load Q-table
    with open(q_table_path, 'rb') as f:
        q_table = pickle.load(f)
    
    print(f'Q-table entries: {len(q_table)}')
    print('\nQ-table contents:')
    for state, actions in q_table.items():
        print(f'  State {state}: {actions}')
    
    # Load training summary
    print('\n=== TRAINING SUMMARY (IMPROVED PARAMETERS) ===')
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    for key, value in summary.items():
        print(f'{key}: {value}')

if __name__ == "__main__":
    analyze_qtable() 