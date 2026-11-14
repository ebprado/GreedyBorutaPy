# Fresh publication experiments (v6)

This is the current entry point. It runs the implemented experiments only:
no budget-matched Vanilla variant, no ARFS comparator, and no metric rename.
It never imports results from earlier experiment directories.

## Run on this Windows machine

The simplest start command creates a new timestamped output directory and begins
the five-synthetic/six-real experiment immediately. It needs no prepared files:

```powershell
.\benchmarks\publication\run.ps1 --cpu-budget -1 --outer-jobs 7
```

Save the `NEW RUN:` path printed at startup for later `--resume`. `--run` is only
for a directory previously created with `--prepare`; a missing `manifest.json`
means there is no prepared run at that path. Use the one-step command above to
create a fresh experiment instead of supplying a nonexistent directory to `--run`.

Open PowerShell in the repository root. The launcher uses the existing Anaconda
scientific stack. It temporarily adds the compatible, already cached
threadpoolctl 3.6.0 to the child process's Python path when necessary; it does
not change the Python installation or the calling shell's environment.

```powershell
# Inspect the entire design without fitting anything:
.\benchmarks\publication\run.ps1 --plan --cpu-budget -1 --outer-jobs 7

# Create a brand-new timestamped run and print its path, without starting it:
.\benchmarks\publication\run.ps1 --prepare --cpu-budget -1 --outer-jobs 7

# Start the prepared experiment (replace RUN_DIRECTORY with the printed path):
.\benchmarks\publication\run.ps1 --run RUN_DIRECTORY

# After Ctrl+C, a crash or a reboot, continue only that same run:
.\benchmarks\publication\run.ps1 --resume RUN_DIRECTORY

# Regenerate analysis tables while the run is stopped:
.\benchmarks\publication\run.ps1 --export RUN_DIRECTORY
```

You can also omit `--prepare` to immediately create and start a fresh run, or use
`python benchmarks/publication/main.py ...` with the requirements from
`benchmarks/runner/requirements-publication.txt` installed.

On hosts that block unsigned PowerShell scripts, invoke the launcher for this
process with `powershell -ExecutionPolicy Bypass -File benchmarks/publication/run.ps1 ...`.
Keep the machine awake during a long run. A restart requires explicit `--resume`.

Prepared runs freeze the scientific configuration, code fingerprints and package
versions. Set scientific options when preparing, not when running/resuming.
The run refuses incompatible code, package versions or dataset contents rather
than silently mixing protocols. `--run` refuses an already-started directory;
`--resume` must be explicit. New output directories must not already exist.

## Exact full workload

The full default is `--suite both`: five synthetic cases and six real datasets,
one synthetic dataset seed (0),
three repeated CV runs per realization, five folds per repeat, and alpha values
0.1, 0.001, 0.00001. Methods have a common maximum depth of 5 and automatic tree
counts. Vanilla runs for at most 512 fitted iterations (with its existing
early-stopping patience); Greedy runs for its computed K.

The machine has 20 physical cores and 28 logical CPUs. `--cpu-budget -1` is now
the default: it resolves all available logical CPUs when preparing the run and
records that positive number in the manifest. On this machine, serial selection
gets 28 estimator threads; with `--outer-jobs 7`, seven outer Greedy workers get four each. Downstream
LightGBM uses 28 threads. No CPU capacity is intentionally reserved for other apps.
An explicit positive budget is still available. It must divide evenly among the
requested outer workers; on other machines choose a compatible `--outer-jobs` value.

Sequential selectors now use joblib's threading backend for ExtraTrees, whose
tree-fitting code releases the GIL. The earlier v6 runner incorrectly forced
the loky process backend on those tree fits, overriding ExtraTrees' preference
for threads and adding serialization/communication overhead. Parallel Greedy
still uses loky for independent Boruta iterations, with nested threads for trees.
The backend is logged in each result row as `Selection_Backend`.

Nested native BLAS/OpenMP pools use one thread during selection. This does not
restrict ExtraTrees' joblib thread pool; it prevents hidden nested native pools
from multiplying the configured parallelism. See the
[joblib parallelism documentation](https://joblib.readthedocs.io/en/stable/generated/joblib.parallel_config.html).
This is a worker/thread budget, not CPU affinity, throttling, or a guarantee of
100% utilization. Preprocessing, shadow shuffling, short shallow-tree fits, memory
traffic, and synchronization can leave CPUs idle even when all CPUs are available.

Prepared/running runs keep their old settings. Stop the old process with Ctrl+C
and use a fresh directory to adopt this backend/CPU change; do not combine timing
results across these execution configurations. Existing checkpoints/results remain
on disk, but their code fingerprints refer to the earlier implementation.

Six methods are compared: NoSelection; Vanilla; parallel Greedy (the requested
outer jobs) with shadow fractions 1.0, 0.25 and 0.1; and serial Greedy (outer
jobs 1) with shadow fraction 1.0 only. Serial and parallel Greedy draw identical
seeds and select identical masks, so serial runs only provide the parallel-speedup
reference; `--serial-shadow-fractions` (a subset of `--shadow-fractions`) adds
more. Vanilla produces two output policies (strict and strict plus rough fix)
from one selector fit. Thus there are seven result rows per split/alpha.

Three repeats keep five folds (80% training data per fit). The corrected
resampled-t variance factor 1/J + n_test/n_train is 1/15 + 1/4 = 0.32 for 3x5,
0.29 for 5x5 and 0.27 for 10x5, so repeats beyond three add little power.
Two-fold CV would instead halve training sets (35 rows for Riboflavin) and raise
the factor above 1. Use `--repeats 5` or `--repeats 10` for tighter intervals.

| Work | Synthetic | Standard benchmarks | Combined |
|---|---:|---:|---:|
| Dataset names/scenarios | 5 | 6 | 11 |
| Dataset realizations | 5 | 6 | 11 |
| Distinct CV training/validation partitions | 75 | 90 | 165 |
| Vanilla selector fits | 225 | 270 | 495 |
| Greedy selector fits | 900 | 1,080 | 1,980 |
| All selector fits | 1,125 | 1,350 | **2,475** |
| Baseline tasks, evaluated once per partition | 75 | 90 | 165 |
| Progress jobs (selectors plus baseline tasks) | 1,200 | 1,440 | **2,640** |
| Logged fold/policy/alpha result rows | 1,575 | 1,890 | **3,465** |
| Maximum downstream fits with baseline deduplication | 1,425 | 1,710 | **3,135** |

There are 594 nominal dataset-realization/alpha/repeat/method CV combinations
before deduplicating the alpha-independent baseline. A selector fit means an
entire Boruta run, containing multiple forest fits. The totals above are NOT
the number of individual trees or forest iterations.

## Synthetic scenarios

Unless changed below, n=1,000, p=500, five directly informative features plus
20 correlated relevant proxies, proxy correlation rho=0.8, and logistic signal
multiplier=2. Features outside that relevant set are independent noise. Direct
predictor effects are scaled by sqrt(number informative).

| Scenario | Difference from reference |
|---|---|
| reference | Reference described above |
| n_features_2000 | p=2,000: many more noise features |
| signal_0.5 | Weaker signal multiplier 0.5 |
| kind_xor | Two binary XOR predictors, no proxies; remaining features are noise |
| kind_null | Target independent of every feature; no relevant predictors |

These five retain a reference with correlated relevant features, dimensionality
stress, weak-signal recovery, interaction recovery, and all-null false positives.
The nine other cases are excluded from the default sweep: the two sample-size
variants, 100-feature variant, strong signal, two sparsity variants, two correlation
variants, and mixed feature types. This removes the corresponding parameter sweeps;
it does not support claims of robustness to every omitted setting.

`DEFAULT_SYNTHETIC_SCENARIOS` in `benchmarks/runner/experiment_run.py` controls
the default list. All generators remain available through explicit `--datasets`
arguments for later experiments. Existing prepared/running manifests retain their
original lists; use a newly prepared directory for this reduced design.

Each scenario has one generated realization (seed 0). Repeated CV and estimator
seeds still vary, but the resulting uncertainty does not capture variation across
independently generated datasets. The six standard
benchmarks below were loaded successfully for validation; these are the actual
dimensions of the configured OpenML versions (after removing missing targets).

| Dataset | Description | Samples | Features | OpenML ID |
|---|---|---:|---:|---:|
| Madelon | Artificial binary classification | 2,600 | 500 | 1485 |
| Arcene | Binary classification from mass spectrometry | 200 | 10,000 | 1458 |
| Gisette | Binary handwritten digit recognition | 7,000 | 5,000 | 41026 |
| Dexter | Binary text classification | 600 | 20,000 | 4136 |
| Isolet | Spoken letter classification, 26 classes | 7,797 | 617 | 300 |
| Riboflavin | Gene expression regression | 71 | 4,088 | 46983 |

Actual n and p are also logged upon loading; dataset versions/content are
fingerprinted. OpenML may reuse downloaded raw data, but no old experiment
selections, scores or timing results are read.

## Computation saved without changing feature-selection inputs

1. Preprocess once per CV partition, shared across alpha and methods. Training
   data alone fit imputation/encoding; all original columns are preserved.
2. NoSelection is independent of alpha: evaluate once and attach the same score
   to all three alpha configurations. This saves 1,100 downstream fits.
3. Within a partition, if two methods/policies/alphas select exactly the same
   ordered feature mask, reuse the downstream score from the first evaluation.
   Training rows, validation rows, model parameters, seed and CPU allocation are
   identical. This commonly avoids repeated serial/parallel downstream fits and
   identical strict/rough-fix fits. Cache hits are explicitly marked.
4. All selector fits are still executed and timed independently; selection
   speedup comparisons are not inferred from a cached fit. Job order is rotated
   reproducibly to reduce consistent order/warm-up effects.

`Downstream_Time_Sec` and downstream memory columns refer to the source measured
fit. `Downstream_Source` identifies it. `Downstream_Fit_Performed` and
`Downstream_Actual_Time_Sec` distinguish actual computation from reused scores.
`Selection_CPU_Sec`, `Downstream_CPU_Sec` and `Downstream_Actual_CPU_Sec` are
user + system CPU seconds of the runner and its joblib workers; divide by the
matching wall time for the mean number of busy cores. Selection RSS is sampled
every 250 ms, so short peaks can be missed.
Do not sum copied timings to estimate total run duration. For independent
downstream timing repetitions, prepare with `--no-score-cache`; NoSelection is
still intrinsically one baseline per partition.

No feature prefilter, sample reduction, fewer CV folds, or smaller tree budget
was silently introduced. Those would change the experiment and cannot be
promised to preserve selection quality. Reduced shadows are a measured
experimental variant, not a claim of zero quality cost.

## Progress, interruptions and time estimates

Terminal output and `run.log` identify dataset, seed, repeat, fold, method, alpha,
selection duration, selected count and score. Long stages emit a heartbeat every
30 seconds. `progress.json` records the current stage, completion counts and a
rough elapsed-rate ETA. The ETA is provisional: datasets/methods have very
different costs, and only completed work in the current session informs it.

After each complete method job, an atomic partition checkpoint is written before
the corresponding JSONL rows are appended. On resume, JSONL is reconstructed
from checkpoints, avoiding duplicate or truncated records. At worst, a method
job interrupted before checkpoint completion is repeated. No old runs are
searched. Only the explicitly supplied v6 directory is eligible for resume.

The first real-data comparison now has five separate selector fits per CV repeat,
where the old leakage-prone design fitted selection once. Expanded methods and
synthetic cases add more work. A two-day previous sweep does not imply a similar
duration here. Exact selector counts are available; a trustworthy wall-time
estimate requires fresh timing on representative cases.

For scale, the earlier 10-repeat run (four outer jobs) measured per-partition
wall time, summed over alphas and the methods kept here: about 860 s for all
five synthetic cases together, 200 s for Madelon, 940 s for Arcene and 3,160 s
for Gisette. Vanilla was 54% of that run's wall time, and it reached its
512-iteration cap on Arcene and Gisette. With 15 partitions per dataset, those
eight datasets need roughly 21 hours; Dexter, Riboflavin and Isolet were not
yet measured and add to that. Early rate ETAs are provisional.

You can start a prepared run with `--max-partitions 1` to measure one complete
training/validation partition from the five-fold CV (all alpha/method combinations) and
stop cleanly; `--resume` then continues its remaining partitions. This does not
reduce the ultimate experiment. The first synthetic reference partition is not
representative of every real dataset, so do not extrapolate it uncritically.

For a smaller separate pilot, keep all datasets/alphas/methods but prepare with
`--repeats 2`: 1,650 selector fits and 2,310 rows. This reduces the
statistical evidence and is NOT an equivalent substitute for the full run.

## Results for analysis and plots

- `manifest.json`: exact parameters, dataset definitions, versions and code hashes.
- `dataset_fingerprints.json`: checks against changed raw data on resume.
- `fold_results.jsonl`: full per-fold records, including selected feature IDs;
  appended during execution, so it can be inspected before completion.
- `checkpoints/`: authoritative atomic method results, grouped by CV partition.
- `fold_metrics.csv`: scalar per-fold data for plotting.
- `summary.csv`: grouped means, standard deviations and counts (descriptive,
  not independent-observation confidence intervals).
- `stability.csv`: mean pairwise selection Jaccard within each realization.
- `selection_frequency.csv`: per-feature selection frequencies.
- `run.log` and `progress.json`: progress and diagnostic history.
- `noninferiority.csv`: generated when the run is complete, using the prespecified
  F1/R² margins stored in its manifest (both default to 0.01).

CSVs are exported on successful completion or a requested partition-limit stop.
After an unexpected interruption, use `--export RUN_DIRECTORY` while stopped.
The full JSONL includes scores, selection size, CPU configuration, iteration and
shadow counts, selection/downstream timings, approximate process-tree RSS, and
known-ground-truth metrics where applicable. Read the memory and NI caveats in
`../PUBLICATION_BENCHMARK.md`.

## Prespecified non-inferiority margins

The new runner defaults to an absolute loss margin of **0.01 weighted F1** for
classification and **0.01 R²** for regression, compared with Vanilla Boruta's
strict-plus-rough-fix selection on exactly the same folds. These are benchmark
design choices: a small explicit score-loss budget in exchange for faster
selection, not a universal definition of practically identical models and not
thresholds estimated from the observed results.

For weighted F1, this is one percentage point on its 0–1 scale: if Vanilla's
mean score is 0.85, the boundary is 0.84. For R², a baseline of 0.80 gives a
boundary of 0.79. On a shared test fold, an R² decrease of 0.01 corresponds to
additional squared error equal to 1% of the target's total sum of squares. It
does not mean a 1% relative increase in prediction error (at baseline R²=0.80,
that increase would be 5%). This interpretation follows the
[R² definition](https://scikit-learn.org/stable/modules/model_evaluation.html#r2-score).
Neither margin promises preservation of minority-class or application-specific
utility. Practical margins depend on the metric and use case; see the discussion
in [JMLR 24, 22-0907](https://jmlr.org/papers/volume24/22-0907/22-0907.pdf).

The existing analysis uses paired differences, the approximate repeated-CV
variance correction, and Holm adjustment across reported comparisons. The
one-sided test level remains 0.01. A mean loss below the margin alone is not
enough: uncertainty must also exclude the non-inferiority boundary, and the
adjusted test must pass. Failure means non-inferiority was not established;
it does not establish inferiority.

Change `DEFAULT_DATA_SEEDS`, `DEFAULT_F1_MARGIN`, and `DEFAULT_R2_MARGIN` near the
top of `benchmarks/runner/experiment_run.py` to change future defaults. Prefer
CLI overrides when preparing a run, so the code itself remains unchanged:

```powershell
.\benchmarks\publication\run.ps1 --prepare --cpu-budget -1 --outer-jobs 7 --f1-margin 0.005 --r2-margin 0.02
```

The manifest's `config.f1_margin`, `config.r2_margin`, and `config.data_seeds`
record the values actually used. An already prepared run uses its manifest,
not subsequently changed defaults or flags passed to `--run`/`--resume`.
Choose the margins before inspecting results; changing them afterwards makes
the corresponding analysis exploratory. Preparing creates a new empty run.

Example analysis:

```python
import pandas as pd
from pathlib import Path

run = Path(r"RUN_DIRECTORY")
scores = pd.read_csv(run / "fold_metrics.csv")
full_records = pd.read_json(run / "fold_results.jsonl", lines=True)
print(scores.groupby(["Dataset", "Method", "Policy"])["Score"].mean())
```

## Smoke check

```powershell
.\benchmarks\publication\run.ps1 --smoke --cpu-budget 2 --outer-jobs 2 --shadow-fractions 1 .25
```

This creates a separate tiny execution check (80 samples, 12 features, fewer
trees/iterations, two-fold CV). Its manifest marks it as smoke data; it is never
imported into the full experiment.

## Augmenting a finished run: coupled Greedy and Vanilla histories

`augment.ps1` (`runner/augment_coupled.py`) adds two analyses to a **finished** run
without touching it. It copies the run to `<run>_aug` once, writes only into the copy,
and resumes when the same command is run again:

```powershell
.\benchmarks\publication\augment.ps1 --source benchmarks\data\publication_runs\RUN_DIRECTORY --plan
.\benchmarks\publication\augment.ps1 --source benchmarks\data\publication_runs\RUN_DIRECTORY
```

- **coupled**: for each CV partition, Vanilla is refitted with the logged arguments and
  seed for K_max = max K(alpha) iterations. Nothing can be confirmed or rejected before
  iteration K(alpha), so these iterations equal the first K(alpha) iterations of every
  logged Vanilla fit. Greedy on that shared importance sequence is added as method
  `GreedyCoupled` (hit counts and strong set included) and scored downstream. The stage
  fails loudly unless Vanilla's strict and rough-fix sets are subsets of the coupled
  Greedy set (Theorem 1), the strong set is a subset of Vanilla's strict set, and K
  matches the logged Greedy runs. Its selection time is Vanilla's own elapsed time at
  iteration K (the "same work stopped at K" speedup baseline).
- **history**: full Vanilla refits with per-iteration confirmed/tentative/rejected
  counts and timing for a subset (default: every dataset, alpha 0.001, repeat 1,
  folds 1-3; see `--history-*`). Each refit must reproduce the logged masks, iteration
  count and coupled prefix exactly.

Selector arguments are captured from the runner's own `fit_selection`, so the script
also works for future fresh runs. Bookkeeping consumes no random numbers. Run it in the
same Python environment as the source run (it refuses otherwise). `GreedyCoupled` is
descriptive: it never enters the prespecified non-inferiority family.
