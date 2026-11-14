#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example: Steelman Analysis using History Tracking

This script demonstrates how to use the new history tracking feature to perform
a "steelman" analysis - evaluating model performance at each checkpoint where
the confirmed feature set changes during Boruta execution.

This allows you to:
1. Track how model performance evolves as features are confirmed
2. Compare convergence speed between vanilla Boruta and Greedy Boruta
3. Identify the optimal stopping point (if early stopping is beneficial)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'runner'))

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from BorutaLocal import BorutaPy
import pandas as pd

# Create a more complex dataset for testing
X, y = make_classification(
    n_samples=500,
    n_features=50,
    n_informative=20,
    n_redundant=10,
    n_repeated=5,
    random_state=42
)

print("=" * 80)
print("STEELMAN ANALYSIS: History-Based Performance Tracking")
print("=" * 80)
print(f"Dataset shape: {X.shape}")
print(f"Number of samples: {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")
print()

# Configure Boruta with history tracking
rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
boruta = BorutaPy(
    rf,
    n_estimators=50,
    max_iter=128,
    random_state=42,
    verbose=2,
    track_history=True  # Enable history tracking
)

# Fit Boruta
print("\nRunning Boruta with history tracking...")
print("-" * 80)
boruta.fit(X, y)

# Analyze history
print("\n" + "=" * 80)
print("CHECKPOINT ANALYSIS")
print("=" * 80)

results = []

for i, checkpoint in enumerate(boruta.history_):
    iteration = checkpoint['iteration']
    confirmed = checkpoint['confirmed']
    tentative = checkpoint['tentative']
    rejected = checkpoint['rejected']
    elapsed = checkpoint['elapsed_sec']

    print(f"\nCheckpoint {i + 1} (Iteration {iteration}):")
    print(f"  Elapsed time: {elapsed:.4f}s")
    print(f"  Confirmed: {len(confirmed)}, Tentative: {len(tentative)}, Rejected: {len(rejected)}")

    # Evaluate model performance at this checkpoint
    if len(confirmed) > 0:
        # Use only confirmed features
        confirmed_list = sorted(list(confirmed))
        X_subset = X[:, confirmed_list]

        # Cross-validate with the confirmed features
        rf_eval = RandomForestClassifier(n_estimators=50, max_depth=7, random_state=42)
        scores = cross_val_score(rf_eval, X_subset, y, cv=5, scoring='accuracy')

        mean_score = scores.mean()
        std_score = scores.std()

        print(f"  Model performance (5-fold CV): {mean_score:.4f} ± {std_score:.4f}")

        results.append({
            'checkpoint': i + 1,
            'iteration': iteration,
            'n_confirmed': len(confirmed),
            'n_tentative': len(tentative),
            'n_rejected': len(rejected),
            'elapsed_sec': elapsed,
            'cv_accuracy_mean': mean_score,
            'cv_accuracy_std': std_score,
            'confirmed_features': confirmed_list
        })
    else:
        print(f"  Model performance: N/A (no confirmed features yet)")
        results.append({
            'checkpoint': i + 1,
            'iteration': iteration,
            'n_confirmed': 0,
            'n_tentative': len(tentative),
            'n_rejected': len(rejected),
            'elapsed_sec': elapsed,
            'cv_accuracy_mean': np.nan,
            'cv_accuracy_std': np.nan,
            'confirmed_features': []
        })

# Create summary DataFrame
df_results = pd.DataFrame(results)

print("\n" + "=" * 80)
print("SUMMARY TABLE")
print("=" * 80)
print(df_results[['checkpoint', 'iteration', 'n_confirmed', 'cv_accuracy_mean',
                   'cv_accuracy_std', 'elapsed_sec']].to_string(index=False))

# Find best performing checkpoint
if len(df_results[df_results['cv_accuracy_mean'].notna()]) > 0:
    best_checkpoint = df_results.loc[df_results['cv_accuracy_mean'].idxmax()]
    print("\n" + "=" * 80)
    print("BEST PERFORMING CHECKPOINT")
    print("=" * 80)
    print(f"Checkpoint: {int(best_checkpoint['checkpoint'])}")
    print(f"Iteration: {int(best_checkpoint['iteration'])}")
    print(f"Number of confirmed features: {int(best_checkpoint['n_confirmed'])}")
    print(f"CV Accuracy: {best_checkpoint['cv_accuracy_mean']:.4f} ± {best_checkpoint['cv_accuracy_std']:.4f}")
    print(f"Time to reach: {best_checkpoint['elapsed_sec']:.4f}s")
    print(f"Features: {best_checkpoint['confirmed_features']}")

# Final comparison
print("\n" + "=" * 80)
print("FINAL RESULT COMPARISON")
print("=" * 80)

# Evaluate with all confirmed + tentative (final Boruta result)
final_features = np.where(boruta.support_ | boruta.support_weak_)[0]
if len(final_features) > 0:
    X_final = X[:, final_features]
    rf_final = RandomForestClassifier(n_estimators=50, max_depth=7, random_state=42)
    final_scores = cross_val_score(rf_final, X_final, y, cv=5, scoring='accuracy')

    print(f"Final Boruta selection (confirmed + tentative):")
    print(f"  Number of features: {len(final_features)}")
    print(f"  CV Accuracy: {final_scores.mean():.4f} ± {final_scores.std():.4f}")

# Evaluate with only confirmed features (final)
final_confirmed = np.where(boruta.support_)[0]
if len(final_confirmed) > 0:
    X_confirmed = X[:, final_confirmed]
    rf_confirmed = RandomForestClassifier(n_estimators=50, max_depth=7, random_state=42)
    confirmed_scores = cross_val_score(rf_confirmed, X_confirmed, y, cv=5, scoring='accuracy')

    print(f"\nFinal Boruta selection (confirmed only):")
    print(f"  Number of features: {len(final_confirmed)}")
    print(f"  CV Accuracy: {confirmed_scores.mean():.4f} ± {confirmed_scores.std():.4f}")

# Baseline with all features
rf_all = RandomForestClassifier(n_estimators=50, max_depth=7, random_state=42)
all_scores = cross_val_score(rf_all, X, y, cv=5, scoring='accuracy')
print(f"\nBaseline (all features):")
print(f"  Number of features: {X.shape[1]}")
print(f"  CV Accuracy: {all_scores.mean():.4f} ± {all_scores.std():.4f}")

print("\n" + "=" * 80)
print("How to use this for Greedy Boruta comparison:")
print("=" * 80)
print("1. Run the same analysis on your GreedyBoruta implementation")
print("2. Compare the convergence speed (time to reach similar accuracy)")
print("3. Compare the number of features selected at each checkpoint")
print("4. Analyze the stability of feature selection across checkpoints")
print("=" * 80)
