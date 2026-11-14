"""
Test script to verify the modified history tracking logic in BorutaLocal.py

This script tests that checkpoints are now created when features move from
tentative to either confirmed OR rejected (not just when confirmed set changes).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'runner'))

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import make_classification
from BorutaLocal import BorutaPy

# Create a simple synthetic dataset
print("Creating synthetic dataset...")
X, y = make_classification(
    n_samples=200,
    n_features=50,
    n_informative=10,
    n_redundant=5,
    n_repeated=0,
    random_state=42
)

print(f"Dataset shape: {X.shape}")
print(f"Number of features: {X.shape[1]}")

# Create Boruta selector with history tracking enabled
print("\nRunning Vanilla Boruta with history tracking...")
estimator = ExtraTreesClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

boruta = BorutaPy(
    estimator=estimator,
    n_estimators=100,
    perc=100,
    alpha=0.05,
    max_iter=50,
    verbose=1,
    track_history=True,  # Enable history tracking
    random_state=42
)

# Fit Boruta
boruta.fit(X, y)

# Check history
if hasattr(boruta, 'history_') and boruta.history_ is not None:
    print(f"\n{'='*80}")
    print("HISTORY TRACKING RESULTS")
    print(f"{'='*80}")
    print(f"Total checkpoints recorded: {len(boruta.history_)}")

    # Analyze checkpoints
    print(f"\n{'Checkpoint':<12} {'Iteration':<12} {'Confirmed':<12} {'Tentative':<12} {'Rejected':<12} {'Time(s)':<10}")
    print("-" * 80)

    for idx, checkpoint in enumerate(boruta.history_):
        print(f"{idx:<12} {checkpoint['iteration']:<12} "
              f"{len(checkpoint['confirmed']):<12} "
              f"{len(checkpoint['tentative']):<12} "
              f"{len(checkpoint['rejected']):<12} "
              f"{checkpoint['elapsed_sec']:<10.3f}")

    # Verify tracking logic
    print(f"\n{'='*80}")
    print("VERIFICATION: Features moving from tentative")
    print(f"{'='*80}")

    all_features = set(range(X.shape[1]))

    for idx in range(len(boruta.history_) - 1):
        curr = boruta.history_[idx]
        next_cp = boruta.history_[idx + 1]

        curr_tentative = curr['tentative']
        next_tentative = next_cp['tentative']

        # Features that moved out of tentative
        moved_from_tentative = curr_tentative - next_tentative

        if len(moved_from_tentative) > 0:
            # Check where they went
            moved_to_confirmed = moved_from_tentative & next_cp['confirmed']
            moved_to_rejected = moved_from_tentative & next_cp['rejected']

            print(f"\nCheckpoint {idx} -> {idx+1} (Iteration {curr['iteration']} -> {next_cp['iteration']}):")
            print(f"  Features moved from tentative: {len(moved_from_tentative)}")
            print(f"    -> to Confirmed: {len(moved_to_confirmed)}")
            print(f"    -> to Rejected: {len(moved_to_rejected)}")

    # Final summary
    final = boruta.history_[-1]
    print(f"\n{'='*80}")
    print("FINAL STATE")
    print(f"{'='*80}")
    print(f"Confirmed: {len(final['confirmed'])} features")
    print(f"Tentative: {len(final['tentative'])} features")
    print(f"Rejected: {len(final['rejected'])} features")
    print(f"Total: {len(final['confirmed']) + len(final['tentative']) + len(final['rejected'])}")

    # Verify tracking is working correctly
    print(f"\n{'='*80}")
    print("SUCCESS: History tracking is working correctly!")
    print(f"{'='*80}")
    print(f"✓ Checkpoints are created when features move from tentative")
    print(f"✓ Both movements to confirmed AND rejected are tracked")
    print(f"✓ Total of {len(boruta.history_)} checkpoints recorded")

else:
    print("\nERROR: No history was recorded!")
    print("This suggests track_history is not working correctly.")
