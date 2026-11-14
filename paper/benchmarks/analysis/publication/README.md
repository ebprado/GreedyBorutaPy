# Publication analysis

Open **publication_analysis.ipynb** and run all cells. The included notebook
includes executed tables and plots for a checkpoint snapshot of
`run_20260923_200722_492094`. Its first configuration cell controls the source run,
focus alpha, and output directory. Launch Jupyter from this directory or anywhere
inside the repository; paths are resolved from the repository root.

Requirements: Python 3.10+, numpy, pandas, scipy, matplotlib, IPython, nbformat,
nbclient, ipykernel and Jupyter Notebook/Lab. The existing Anaconda environment
used by the benchmark has the analysis dependencies. No selector, dataset download,
LightGBM fit, or benchmark export command is invoked.

```powershell
& "$env:USERPROFILE/anaconda3/python.exe" -m jupyter notebook benchmarks/analysis/publication/publication_analysis.ipynb
& "$env:USERPROFILE/anaconda3/python.exe" -m pytest benchmarks/analysis/publication/test_analysis_helpers.py -q
```

For batch regeneration without a Jupyter server, run
`python benchmarks/analysis/publication/execute_notebook.py` (or use the Anaconda
Python path above). This executes every code cell with IPython and captures tables
and inline figures. It was used to produce the included outputs because the local
Anaconda ZeroMQ transport failed to start a regular kernel. It requires
`matplotlib-inline`; the regression checks additionally require `pytest`.

**`paper_results.ipynb`** builds the figures, LaTeX tables and `key_numbers.json` for
the paper's results section into `outputs/<run>/paper/`
(`python benchmarks/analysis/publication/execute_notebook.py paper_results.ipynb`).
Both notebooks prefer the augmented copy `<run>_aug` when it exists (see
`benchmarks/publication/README.md`); the environment variable `PUBLICATION_RUN_DIR`
overrides the source. Non-inferiority adds a labelled sensitivity line at one-sided
level 0.025 next to the prespecified 0.01 decision.

## Inputs and outputs

The authoritative inputs are the manifest and atomically saved checkpoint jobs
under `benchmarks/data/publication_runs/<run>/checkpoints`. Aggregate JSONL/CSV
exports can be stale during a running experiment and are not used. Source files
are read only. A live read spans a recorded interval; it is not a simultaneous
snapshot of all checkpoint files. Duplicate observations, mismatched splits,
invalid masks, inconsistent ground truth, or execution configurations fail loudly.

Running the notebook writes only `outputs/<run>/` under this analysis directory:

- PNG (300 dpi) and vector PDF figures, with the same figures displayed inline;
- fold summaries with valid counts, coverage, paired comparisons, speedups,
  shadow/parallel ablations, stability and synthetic feature frequencies;
- non-inferiority diagnostics and decisions, with decisions withheld for partial runs;
- `analyzed_rows.jsonl.gz`, the exact analyzed row values serialized to 15 decimal
  places, the source manifest, and provenance with per-checkpoint SHA-256 hashes.

Generated outputs are ignored by Git; the executed notebook is the reviewable
artifact. Re-running refreshes these outputs and notebook results. Copy the output
directory before rerunning if you want to retain multiple dated snapshots.
To inspect a retained snapshot directly:

```python
rows = pandas.read_json("outputs/RUN/analyzed_rows.jsonl.gz", lines=True)
```

## Analysis choices

Coverage includes every planned dataset, including missing ones. Comparative
plots use only dataset-realization/alpha blocks with every planned method, policy,
repeat and fold present. Full observed summaries and resource accounting also
include partial blocks, explicitly separated from comparisons.

Score is weighted F1 for classification and R² for regression. Feature-recovery
F1 is a different metric. No dataset seeds or task metrics are pooled for inference.
Runtime speedups divide exactly paired selector times before summarizing; the
all-feature baseline has no selector speedup. Vanilla policies share one fit;
NoSelection alpha rows share one job; actual downstream time excludes cache hits.
Sampled process-tree RSS includes reusable workers and can double-count shared
pages or miss short peaks. Summed fit times exclude preprocessing and overhead.

Non-inferiority uses candidate minus Vanilla rough-fix score, margins and test
level from the frozen manifest, and the runner's approximate corrected resampled-t
formula. Holm correction uses the full planned family (231 tests for this run),
including NoSelection and reserving missing tests as p=1. Interim estimates do not
support final or sequential claims. Final decisions are enabled only once the full
design is complete and is not a smoke run. Bounds are pointwise one-sided bounds,
not simultaneous or two-sided intervals. Failure to establish NI is inconclusive,
not evidence of inferiority or equivalence.

Synthetic truth follows `synthetic_suite.py`: direct predictors plus correlated
proxies, two direct XOR predictors, or no relevant predictors under the null.
Undefined precision/recall remain missing, with valid counts in summary tables.
Raw pairwise Jaccard is descriptive; all-feature and identical empty sets have
Jaccard 1. Fold dispersion is not an independent-observation confidence interval.

The v6 checkpoints hold final method jobs, not per-iteration histories. Therefore
the older convergence, undecidedness, potential-miss-rate and earliest-equivalence
plots cannot be reconstructed. Their former `p > 0.05` equivalence rule is not used.
