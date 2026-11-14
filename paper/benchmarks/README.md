# Benchmarks Directory Structure

This directory contains benchmarking code, tests, and results for comparing Boruta variants.

## Directory Structure

```
benchmarks/
├── data/                           # All CSV data files
│   ├── benchmark_results_v*.csv   # Main benchmark results
│   ├── checkpoint_*.csv           # Checkpoint tracking data
│   └── *.csv                      # Other analysis results
│
├── tests/                         # Test files
│   ├── test_checkpoint_integration.py
│   ├── test_flexible_importance.py
│   ├── test_history_tracking.py
│   └── test_tracking_logic.py
│
├── examples/                      # Example scripts
│   └── steelman_analysis_example.py
│
├── BorutaLocal.py                # Local Boruta implementation with tracking
├── GreedyBoruta.py               # Greedy Boruta implementation
├── GreedyBorutaParallel.py       # Parallel Greedy implementation
├── GreedyBorutaFlexible.py       # Flexible importance metric support
│
├── config.py                     # Configuration (datasets, alphas, etc.)
├── main.py                       # Main benchmark runner
├── data_loader.py                # Dataset loading utilities
├── evaluator.py                  # Model evaluation utilities
├── feature_selectors.py          # Feature selector wrappers
│
├── analyze_results.ipynb         # Results analysis notebook
├── analyze_checkpoint_results.ipynb  # Checkpoint analysis notebook
│
└── README.md                     # This file
```

## Key Files

### Core Implementation
- **BorutaLocal.py**: Modified Boruta with history tracking support
- **GreedyBorutaParallel.py**: Parallelized greedy variant
- **GreedyBorutaFlexible.py**: Supports both standard and SHAP-based importance

### Configuration & Execution
- **config.py**: Central configuration for datasets, alphas, estimators, and output paths
- **main.py**: Main benchmarking script with resume functionality

### Analysis
- **analyze_checkpoint_results.ipynb**: Analyzes convergence behavior and statistical equivalence
- **analyze_results.ipynb**: General results visualization and comparison

## Data Files

All CSV files are stored in the `data/` folder:
- **benchmark_results_v*.csv**: Main results with performance metrics
- **checkpoint_history_vanilla_v*.csv**: Iteration-level checkpoint data for Vanilla Boruta
- **checkpoint_equivalence_summary.csv**: Statistical equivalence analysis
- **checkpoint_statistical_tests.csv**: Detailed test results

## Running Benchmarks

```bash
# Run main benchmarks
python main.py

# Run specific tests
python tests/test_tracking_logic.py
```

## Key Features

### History Tracking
The modified Boruta implementation tracks checkpoints when features move from tentative to either confirmed or rejected status. This allows analysis of:
- Convergence speed
- Performance at intermediate states
- Statistical equivalence with Greedy variants

### Resume Functionality
The main benchmark script supports resuming interrupted runs. It checks existing results and only runs missing configurations.

## Configuration

Edit `config.py` to modify:
- Datasets to benchmark
- Alpha values to test
- Number of runs per configuration
- Estimator types (extra_trees, rf, lgbm, xgb)
- Output file paths (automatically use `data/` folder)
