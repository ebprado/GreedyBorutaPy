# Expanded publication benchmark

**Current entry point:** `publication/main.py` (or `publication/run.ps1` on Windows).
See [the v6 run guide](publication/README.md) for fresh timestamped runs, safe resume,
progress logs, exact workload counts and computation reuse. The v5 runner below
is retained as the original implementation/reference.

Use `runner/publication_benchmark.py` for the expanded experiment, rather than the
older `runner/main.py`. It leaves previous results and their analysis notebooks
intact. No publication-sized sweep has been run as part of implementation.

From the repository root, with an appropriate Python environment:

```powershell
python -m pip install -r benchmarks/runner/requirements-publication.txt
python benchmarks/runner/publication_benchmark.py --smoke --cpu-budget 4 --outer-jobs 4
python benchmarks/runner/publication_benchmark.py --suite synthetic --cpu-budget 4 --outer-jobs 4 --output benchmarks/data/synthetic_publication
python benchmarks/runner/publication_benchmark.py --suite real --cpu-budget 4 --outer-jobs 4 --output benchmarks/data/real_publication
```

The current local Anaconda environment has threadpoolctl 2.2.0, which fails while
inspecting OpenBLAS. Validation used the locally cached threadpoolctl 3.6.0 via
PYTHONPATH; the environment itself was not modified. Install the requirements
before normal use. Smoke output is only an execution check, not research evidence.

## Experimental design

- Synthetic suite: reference plus 13 one-factor variations covering sample size,
  dimensionality, signal strength, number of informative features, correlation,
  mixed continuous/binary/ordered-discrete features, all-null data, and pure XOR.
  Scenarios have known feature IDs and three independent dataset seeds by default.
  Within a dataset realization, ten shuffled five-fold CV repeats share exactly
  the same splits across methods. `--datasets` selects scenario names; `--data-seeds`,
  `--repeats`, and `--alphas` limit or expand the sweep. Real datasets use one fixed
  realization and repeated splits. No blind pooling of different dataset seeds.
- Every training fold fits its own imputation/encoding and selector; validation
  labels never enter fitting. Original feature order is restored after encoding.
- NoSelection: LightGBM using all original features with the same splits,
  hyperparameters and downstream CPU allocation as selected-feature models.
- Vanilla: strict confirmations and strict-plus-rough-fix confirmations are both
  recorded explicitly. Maximum 512 fitted iterations; the local implementation
  uses an exclusive max_iter bound, so the constructor receives 513.
- Greedy: the same GreedyBorutaPyParallel class with outer n_jobs=1 and the requested
  parallel count, for shadow fractions 1, 0.25 and 0.1 by default. A CPU budget B
  gives the serial fit B estimator threads and the parallel fit B/outer threads
  per worker. B must be divisible by outer. Nested native thread pools are restricted to one thread during selection, while estimator joblib threads use the allocated budget.
  Methods execute sequentially in a reproducibly rotated order; CV folds also
  execute sequentially. Estimator='auto' tree counts stay based on original p
  across shadow variants so the shadow ablation does not change the tree budget.
- Empty selections use a training-only constant predictor, rather than an
  invented score of zero. This particularly matters for all-null experiments.
- The new runner does not implement a runtime/iteration-matched Vanilla variant
  or an external ARFS alternative; these are proposals awaiting a design choice.

## Recorded outputs

`manifest.json` stores arguments, methods, scenario definitions, protocol and
versions. `fold_results.jsonl` records every fold's score, split hash, train/test
sizes, selected feature IDs/counts, selection time, downstream time, iteration
count, actual shadow count for Greedy, CPU allocation and memory measurements.
It additionally records TP/FP/FN/TN, recall, precision, F1, FPR, realized FDP,
and FP/original-feature-count when ground truth is available. Undefined precision
or recall is null, not silently converted to a perfect score. FDP is zero for an
empty selection. Strict and rough-fix Vanilla rows share the same selection cost.

`stability.csv` reports mean pairwise Jaccard across training-fold selections
within each dataset realization/method/configuration. `selection_frequency.csv`
reports the empirical selection frequency of every feature. Identical empty
sets have Jaccard 1. This is raw set stability, not a chance-corrected measure or
an independent-observation significance test. All-feature selection trivially
has stability 1; that alone does not make it a useful selector.

Memory is sampled process-tree RSS (parent plus descendants) every 10 ms. Absolute
baseline, peak and peak increase are reported for selection and downstream fits.
It includes reusable joblib workers, can double-count shared pages and can miss
short peaks. It is not exact allocated memory or a clean process-isolated peak.

Each run requires a fresh output directory to avoid mixing protocols. Individual
folds are flushed as they finish, but automatic resume is not implemented. Failed
fits stop the run instead of being converted to successful zero scores. Retain
the manifest with the results. Large default sweeps can be expensive; start with
`--smoke` or a subset.

## Non-inferiority

Choose absolute loss margins BEFORE inspecting the new scores. The current v6
runner defaults to 0.01 weighted F1 and 0.01 R²; see the rationale and override
instructions in [the v6 run guide](publication/README.md#prespecified-non-inferiority-margins).
The original v5 runner and standalone analysis require explicit `--f1-margin`
and `--r2-margin` values. If a required margin is absent in v5, it still saves
scores but does not run NI.
For existing fold results, use:

```text
python benchmarks/runner/noninferiority.py PATH_TO_FOLD_RESULTS --f1-margin PREDEFINED_F1_MARGIN --r2-margin PREDEFINED_R2_MARGIN --output PATH_TO_NI_CSV
```

The comparison uses D = candidate score - Vanilla rough-fix score and tests
H0: mean(D) <= -margin against H1: mean(D) > -margin. Pairing includes repeat,
fold, split hash, training size and test size. Missing/duplicate pairs are errors.
Each dataset realization, task and alpha is analyzed separately.

The standard error uses the approximate Nadeau–Bengio correction:
sqrt((1/N + mean(n_test/n_train)) * sample_variance(D)), with N-1 t degrees of
freedom. This accounts approximately for dependence from overlapping repeated-CV
training sets; it does not offer an exact finite-sample guarantee. See
https://www.iro.umontreal.ca/~lisa/bib/pub_subject/language/pointeurs/nadeau_MLJ1597.pdf

Output includes the mean paired difference, margin, corrected SE, one-sided lower
confidence bound, raw p-value, and Holm-adjusted p-value over all tests in that
output table. The reported CI is pointwise, not Holm-adjusted. The final decision
uses the Holm-adjusted test at level 0.01 by default. Failure to establish
non-inferiority means inconclusive, not inferior or equivalent. Zero-variance
comparisons are flagged explicitly. Do not report smoke-test inference.

## Reduced shadows API

```python
selector = GreedyBorutaPyParallel(
    estimator, n_jobs=4, random_state=42, n_shadows=0.25
)
```

`None` uses p shadows; an integer uses exactly that count; a float in (0, 1]
uses ceil(fraction*p). Thus `1` means one shadow and `1.0` means all shadows.
Source columns are sampled without replacement on each iteration, then shuffled
independently. Exactly r shadows are used, even below five: the old duplication
of shadows for p<5 has been removed. The K iteration budget still uses p. Fitted
attributes include `n_shadows_`, `n_iter_`, and `hit_counts_`. Fixed integer seeds
reproduce results on refitting and across outer worker counts under the same
estimator behavior. The theoretical K/(r+1) bound requires exchangeability and
the maximum-shadow threshold (`perc=100`); it is not claimed for lower percentiles.

Weak/strong confirmation semantics have not been changed.
