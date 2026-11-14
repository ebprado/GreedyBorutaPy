#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for GreedyBorutaFlexible with different importance metrics.

This demonstrates the usage of standard feature importances vs SHAP values.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'runner'))

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from GreedyBorutaFlexible import GreedyBorutaPyFlexible
import time

print("=" * 80)
print("Testing GreedyBorutaPyFlexible with Different Importance Metrics")
print("=" * 80)

# Create a test dataset
print("\nCreating test dataset...")
X, y = make_classification(
    n_samples=200,
    n_features=30,
    n_informative=15,
    n_redundant=5,
    n_repeated=0,
    random_state=42
)
print(f"Dataset shape: {X.shape}")
print(f"Number of informative features (ground truth): 15")

# Base estimator
rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)

# ============================================================================
# Test 1: Standard Feature Importances (Default)
# ============================================================================
print("\n" + "=" * 80)
print("Test 1: Standard Feature Importances")
print("=" * 80)

selector_standard = GreedyBorutaPyFlexible(
    estimator=rf,
    n_estimators='auto',
    alpha=0.05,
    importance_metric='feature_importances',  # Default
    n_jobs=-1,
    verbose=1,
    random_state=42
)

start_time = time.time()
selector_standard.fit(X, y)
elapsed_standard = time.time() - start_time

print(f"\nResults (Standard Feature Importances):")
print(f"  Time elapsed: {elapsed_standard:.2f}s")
print(f"  Confirmed features: {selector_standard.n_features_}")
print(f"  Confirmed indices: {np.where(selector_standard.support_)[0].tolist()}")
print(f"  Iterations performed: {selector_standard.max_iter}")

# ============================================================================
# Test 2: SHAP-based Importances
# ============================================================================
print("\n" + "=" * 80)
print("Test 2: SHAP-based Importances")
print("=" * 80)

try:
    selector_shap = GreedyBorutaPyFlexible(
        estimator=rf,
        n_estimators='auto',
        alpha=0.05,
        importance_metric='shap',
        shap_kwargs={'check_additivity': False},  # Faster, less strict
        n_jobs=-1,
        verbose=1,
        random_state=42
    )

    start_time = time.time()
    selector_shap.fit(X, y)
    elapsed_shap = time.time() - start_time

    print(f"\nResults (SHAP Importances):")
    print(f"  Time elapsed: {elapsed_shap:.2f}s")
    print(f"  Confirmed features: {selector_shap.n_features_}")
    print(f"  Confirmed indices: {np.where(selector_shap.support_)[0].tolist()}")
    print(f"  Iterations performed: {selector_shap.max_iter}")

    # ========================================================================
    # Comparison
    # ========================================================================
    print("\n" + "=" * 80)
    print("Comparison")
    print("=" * 80)

    standard_features = set(np.where(selector_standard.support_)[0])
    shap_features = set(np.where(selector_shap.support_)[0])

    overlap = standard_features & shap_features
    only_standard = standard_features - shap_features
    only_shap = shap_features - standard_features

    print(f"\nFeature Selection Agreement:")
    print(f"  Features in both: {len(overlap)} - {sorted(overlap)}")
    print(f"  Only in Standard: {len(only_standard)} - {sorted(only_standard)}")
    print(f"  Only in SHAP: {len(only_shap)} - {sorted(only_shap)}")
    print(f"  Jaccard similarity: {len(overlap) / len(standard_features | shap_features):.3f}")

    print(f"\nTime Comparison:")
    print(f"  Standard: {elapsed_standard:.2f}s")
    print(f"  SHAP: {elapsed_shap:.2f}s")
    print(f"  SHAP overhead: {(elapsed_shap / elapsed_standard - 1) * 100:.1f}%")

except ImportError as e:
    print(f"\nSkipping SHAP test: {e}")
    print("\nTo use SHAP importance, install with: pip install shap")

# ============================================================================
# Test 3: Invalid Metric
# ============================================================================
print("\n" + "=" * 80)
print("Test 3: Invalid Importance Metric (should raise error)")
print("=" * 80)

try:
    selector_invalid = GreedyBorutaPyFlexible(
        estimator=rf,
        importance_metric='invalid_metric',
        random_state=42
    )
    print("ERROR: Should have raised ValueError!")
except ValueError as e:
    print(f"✓ Correctly raised ValueError: {e}")

# ============================================================================
# Test 4: Transform Method
# ============================================================================
print("\n" + "=" * 80)
print("Test 4: Transform Method")
print("=" * 80)

X_transformed = selector_standard.transform(X)
print(f"Original shape: {X.shape}")
print(f"Transformed shape: {X_transformed.shape}")
print(f"✓ Transform method works correctly")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("\nGreedyBorutaPyFlexible supports two importance metrics:")
print("  1. 'feature_importances' (default) - Uses estimator.feature_importances_")
print("  2. 'shap' - Uses SHAP values for model-agnostic importance")
print("\nBoth methods:")
print("  ✓ Support parallel execution (n_jobs parameter)")
print("  ✓ Are deterministic (with random_state)")
print("  ✓ Have same API as GreedyBorutaPyParallel")
print("\nSHAP-based importance:")
print("  + Model-agnostic (works with any estimator)")
print("  + Theoretically more robust (game theory foundation)")
print("  - Slower (especially for large datasets)")
print("  - Requires 'shap' package installation")
print("\n" + "=" * 80)
