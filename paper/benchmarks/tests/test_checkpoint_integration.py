#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script to verify checkpoint tracking integration with main benchmark system.
This runs a minimal version of the benchmark to test the new functionality.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'runner'))

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from feature_selectors import SelectorWrapper
from evaluator import evaluate_downstream

# Create a small test dataset
print("Creating test dataset...")
X, y = make_classification(
    n_samples=200,
    n_features=30,
    n_informative=15,
    n_redundant=5,
    random_state=42
)
print(f"Dataset shape: {X.shape}\n")

# Test configuration
TEST_MAIN_RESULTS = "test_main_results.csv"
TEST_CHECKPOINT_RESULTS = "test_checkpoint_results.csv"

# Clean up previous test files if they exist
for f in [TEST_MAIN_RESULTS, TEST_CHECKPOINT_RESULTS]:
    if os.path.exists(f):
        os.remove(f)
        print(f"Removed existing {f}")

print("\n" + "=" * 80)
print("Testing Vanilla Boruta with Checkpoint Tracking")
print("=" * 80)

# Test Vanilla with history tracking
method = "Vanilla"
alpha = 0.05
run_i = 0
task_type = "class"

base_est = RandomForestClassifier(max_depth=5, random_state=42)
wrapper = SelectorWrapper(method, clone(base_est), alpha, random_state=42)

# Enable history tracking
print("\nFitting Vanilla Boruta with track_history=True...")
wrapper.fit(X, y, track_history=True)

print(f"Fitting complete!")
print(f"  Time: {wrapper.running_time:.2f}s")
print(f"  Iterations: {wrapper.n_iterations}")

# Get results
mask_confirmed = wrapper.get_confirmed_mask()
mask_tentative = wrapper.get_tentative_mask()
n_confirmed = np.sum(mask_confirmed)
n_tentative = np.sum(mask_tentative)

print(f"  Confirmed features: {n_confirmed}")
print(f"  Tentative features: {n_tentative}")

# Evaluate downstream
score_mean_conf, score_std_conf = evaluate_downstream(
    X, y, mask_confirmed,
    task_type=task_type,
    random_state=run_i
)
print(f"  Score (confirmed): {score_mean_conf:.4f} ± {score_std_conf:.4f}")

# Get history
history = wrapper.get_history()
print(f"\nHistory tracking:")
print(f"  History available: {history is not None}")
if history:
    print(f"  Number of checkpoints: {len(history)}")

    # Show checkpoint details
    print("\n  Checkpoints:")
    for i, cp in enumerate(history):
        print(f"    [{i}] Iter={cp['iteration']}, "
              f"Confirmed={len(cp['confirmed'])}, "
              f"Tentative={len(cp['tentative'])}, "
              f"Rejected={len(cp['rejected'])}, "
              f"Time={cp['elapsed_sec']:.3f}s")

# Save main result
main_row = {
    "Dataset": "test",
    "Task": task_type,
    "n_samples": X.shape[0],
    "n_features_orig": X.shape[1],
    "Estimator": "rf",
    "Method": method,
    "Alpha": alpha,
    "Run_ID": run_i,
    "Time_Sec": wrapper.running_time,
    "Iterations": wrapper.n_iterations,
    "N_Confirmed": n_confirmed,
    "N_Tentative": n_tentative,
    "Score_Confirmed_Mean": score_mean_conf,
    "Score_Confirmed_Std": score_std_conf
}

df_main = pd.DataFrame([main_row])
df_main.to_csv(TEST_MAIN_RESULTS, index=False)
print(f"\nSaved main result to: {TEST_MAIN_RESULTS}")

# Process and save checkpoints
if history:
    checkpoint_rows = []

    for checkpoint_idx, checkpoint in enumerate(history):
        iteration = checkpoint['iteration']
        confirmed = checkpoint['confirmed']
        tentative = checkpoint['tentative']
        rejected = checkpoint['rejected']
        elapsed_sec = checkpoint['elapsed_sec']

        n_confirmed = len(confirmed)
        n_tentative = len(tentative)
        n_rejected = len(rejected)

        # Evaluate only if there are confirmed features
        if n_confirmed > 0:
            confirmed_list = sorted(list(confirmed))
            mask_confirmed = np.zeros(X.shape[1], dtype=bool)
            mask_confirmed[confirmed_list] = True

            score_mean, score_std = evaluate_downstream(
                X, y, mask_confirmed,
                task_type=task_type,
                random_state=run_i
            )
        else:
            score_mean = 0.0
            score_std = 0.0

        checkpoint_row = {
            "Dataset": "test",
            "Task": task_type,
            "n_samples": X.shape[0],
            "n_features_orig": X.shape[1],
            "Estimator": "rf",
            "Method": method,
            "Alpha": alpha,
            "Run_ID": run_i,
            'Checkpoint_ID': checkpoint_idx,
            'Iteration': iteration,
            'Elapsed_Sec': elapsed_sec,
            'N_Confirmed': n_confirmed,
            'N_Tentative': n_tentative,
            'N_Rejected': n_rejected,
            'Score_Mean': score_mean,
            'Score_Std': score_std,
            'Is_Final': checkpoint_idx == len(history) - 1
        }

        checkpoint_rows.append(checkpoint_row)

    df_checkpoints = pd.DataFrame(checkpoint_rows)
    df_checkpoints.to_csv(TEST_CHECKPOINT_RESULTS, index=False)
    print(f"Saved {len(checkpoint_rows)} checkpoint rows to: {TEST_CHECKPOINT_RESULTS}")

    # Display checkpoint results
    print("\n" + "=" * 80)
    print("Checkpoint Results Summary")
    print("=" * 80)
    print(df_checkpoints[['Checkpoint_ID', 'Iteration', 'N_Confirmed',
                          'Score_Mean', 'Elapsed_Sec', 'Is_Final']].to_string(index=False))

print("\n" + "=" * 80)
print("Testing Greedy Boruta (should NOT generate checkpoints)")
print("=" * 80)

# Test that Greedy doesn't create history
method = "Greedy"
wrapper_greedy = SelectorWrapper(method, clone(base_est), alpha, random_state=42)
wrapper_greedy.fit(X, y, track_history=False)  # Greedy ignores this anyway

history_greedy = wrapper_greedy.get_history()
print(f"\nGreedy history available: {history_greedy is not None}")
if history_greedy:
    print(f"WARNING: Greedy should not have history! Found {len(history_greedy)} checkpoints")
else:
    print("✓ Correct: Greedy does not track history")

print("\n" + "=" * 80)
print("Test Complete!")
print("=" * 80)
print("\nFiles created:")
print(f"  - {TEST_MAIN_RESULTS}")
print(f"  - {TEST_CHECKPOINT_RESULTS}")
print("\nYou can inspect these CSV files to verify the integration works correctly.")
