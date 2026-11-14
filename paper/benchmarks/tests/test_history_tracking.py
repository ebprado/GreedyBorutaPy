#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script to verify the history tracking functionality in BorutaPy.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'runner'))

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from BorutaLocal import BorutaPy

# Create a simple dataset for testing
X, y = make_classification(
    n_samples=100,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    n_repeated=0,
    random_state=42
)

print("Testing BorutaPy with history tracking enabled...")
print(f"Dataset shape: {X.shape}")
print("-" * 60)

# Test 1: Without history tracking (default behavior)
print("\nTest 1: Without history tracking (default)")
rf = RandomForestClassifier(n_estimators=20, max_depth=5, random_state=42)
boruta_no_history = BorutaPy(rf, n_estimators=20, max_iter=10, random_state=42, verbose=1)
boruta_no_history.fit(X, y)

print(f"\nHas history_ attribute: {hasattr(boruta_no_history, 'history_')}")
print(f"Number of confirmed features: {boruta_no_history.n_features_}")

# Test 2: With history tracking enabled
print("\n" + "=" * 60)
print("Test 2: With history tracking enabled")
print("=" * 60)
rf = RandomForestClassifier(n_estimators=20, max_depth=5, random_state=42)
boruta_with_history = BorutaPy(
    rf,
    n_estimators=20,
    max_iter=10,
    random_state=42,
    verbose=1,
    track_history=True
)
boruta_with_history.fit(X, y)

print(f"\nHas history_ attribute: {hasattr(boruta_with_history, 'history_')}")
print(f"Number of confirmed features: {boruta_with_history.n_features_}")

# Display history
if hasattr(boruta_with_history, 'history_'):
    print(f"\nNumber of history checkpoints: {len(boruta_with_history.history_)}")
    print("\nHistory details:")
    print("-" * 60)

    for i, checkpoint in enumerate(boruta_with_history.history_):
        print(f"\nCheckpoint {i + 1}:")
        print(f"  Iteration: {checkpoint['iteration']}")
        print(f"  Elapsed time: {checkpoint['elapsed_sec']:.4f} seconds")
        print(f"  Confirmed: {len(checkpoint['confirmed'])} features - {sorted(checkpoint['confirmed'])}")
        print(f"  Tentative: {len(checkpoint['tentative'])} features")
        print(f"  Rejected: {len(checkpoint['rejected'])} features")

    # Show the progression of confirmed features
    print("\n" + "=" * 60)
    print("Progression of confirmed feature count:")
    print("=" * 60)
    for checkpoint in boruta_with_history.history_:
        print(f"Iteration {checkpoint['iteration']:2d}: {len(checkpoint['confirmed']):2d} confirmed, "
              f"{len(checkpoint['tentative']):2d} tentative, {len(checkpoint['rejected']):2d} rejected "
              f"(elapsed: {checkpoint['elapsed_sec']:.2f}s)")

print("\n" + "=" * 60)
print("Testing complete!")
print("=" * 60)
