"""Read-only analysis of publication-v6 checkpoints; no fitting or runner imports."""
from __future__ import annotations

import gzip
import hashlib
import itertools
import json
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t

RUN_NAME = "run_20260926_102546_793908"
BLOCK = ["Dataset", "Data_Seed", "Task", "Estimator", "Alpha"]
GROUP = BLOCK + ["Method", "Policy"]
PAIR = BLOCK + ["Repeat", "Fold", "Split_ID", "Partition_ID", "N_Train", "N_Test"]
LABELS = {
    ("NoSelection", "selected"): "All features",
    ("Vanilla", "strict"): "Vanilla strict",
    ("Vanilla", "rough_fix"): "Vanilla rough fix",
}
COLORS = dict(zip(LABELS.values(), ["#777777", "#222222", "#0072B2"]))
# Any --outer-jobs value the runner accepts; serial (j=1) and parallel colors by fraction.
for _jobs, (_fraction, _serial, _parallel) in itertools.product(
        range(1, 129), [(1, "#009E73", "#D55E00"), (0.25, "#56B4E9", "#E69F00"),
                        (0.1, "#CC79A7", "#882255")]):
    _label = f"Greedy j={_jobs}, s={_fraction:g}"
    LABELS[(f"Greedy_jobs{_jobs}_shadows{_fraction}", "selected")] = _label
    COLORS[_label] = _serial if _jobs == 1 else _parallel
# Augmentation: Greedy evaluated on the first K iterations of the logged Vanilla fit.
COUPLED = "GreedyCoupled"
LABELS[(COUPLED, "selected")] = "Greedy coupled"
COLORS["Greedy coupled"] = "#AA4499"
ORDER = list(LABELS.values())
# Paper figures: validated categorical slots (blue, orange, aqua), recessive ink.
PAPER = dict(vanilla="#2a78d6", greedy="#eb6834", third="#1baf7a", ink="#0b0b0b",
             muted="#52514e", faint="#9a9893", grid="#e6e5e0")
# Ordered layers of a Greedy selection, most to least evidence: one hue, dark to light.
# FATE_COLORS = ["#8f3510", "#c4511f", "#eb6834", "#f6b896"]
FATE_COLORS = ["#2a78d6", "#6ca2e3", "#aecbef", "#eb6834"]



@dataclass
class RunData:
    directory: Path
    manifest: dict
    progress: dict
    frame: pd.DataFrame
    coverage: pd.DataFrame
    provenance: dict

    @property
    def complete(self):
        """The prespecified design; optional augmentations never gate final decisions."""
        return bool(self.coverage.loc[~self.coverage.Augmented, "Complete"].all())


def repository_root(start=None):
    """Works from the repository root, analysis directory, or notebook directory."""
    start = Path(start or Path.cwd()).resolve()
    for path in [start, *start.parents]:
        if (path / "benchmarks" / "runner" / "experiment_run.py").is_file():
            return path
    raise FileNotFoundError("Start Jupyter within the GreedyBorutaPy repository.")


def augmented_methods(manifest):
    """Methods added after the run by runner/augment_coupled.py (not prespecified)."""
    return [m for record in manifest.get("augmentations", []) for m in record.get("methods", [])]


def relevant_features(spec):
    """Synthetic ground truth from synthetic_suite.generate_synthetic."""
    if spec["kind"] == "null":
        return set()
    q = spec.get("n_correlated", 0) if spec.get("correlation", 0) > 0 else 0
    return set(range(spec["n_informative"] + q))


def expected_groups(manifest):
    config = manifest["config"]
    methods = ([dict(m, Augmented=False) for m in config["methods"]]
               + [dict(m, Augmented=True) for m in augmented_methods(manifest)])
    for dataset, (kind, _) in config["datasets"].items():
        for seed in (config["data_seeds"] if kind == "synthetic" else [0]):
            for alpha, method in itertools.product(config["alphas"], methods):
                policies = ["strict", "rough_fix"] if method["name"] == "Vanilla" else ["selected"]
                for policy in policies:
                    yield dict(Dataset=dataset, Data_Seed=seed, Alpha=alpha, Method=method["name"],
                               Policy=policy, Suite=kind, Augmented=method["Augmented"])


def validate_frame(frame, manifest):
    """Reject corrupt observations instead of silently dropping or averaging them."""
    required = set(PAIR + ["Method", "Policy", "Job_ID", "Score", "N_Features",
                          "N_Selected", "Selected_Features", "Mask_ID", "Iterations",
                          "Selection_Time_Sec", "Downstream_Actual_Time_Sec",
                          "Downstream_Fit_Performed", "Downstream_Source", "CPU_Budget",
                          "Outer_Jobs", "Estimator_Jobs", "Selection_Backend"])
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required fields: {sorted(missing)}")
    key = BLOCK + ["Repeat", "Fold", "Method", "Policy"]
    if frame.duplicated(key).any():
        raise ValueError("Duplicate fold/policy observations")
    if frame[list(required - {"Selected_Features"})].isna().any().any():
        raise ValueError("Null identifiers, scores, or required resource fields")
    config = manifest["config"]
    expected = pd.DataFrame(expected_groups(manifest))
    design = ["Dataset", "Data_Seed", "Alpha", "Method", "Policy"]
    if not frame[design].merge(expected[design], how="left", indicator=True)["_merge"].eq("both").all():
        raise ValueError("Rows outside the manifest design")
    for col, n in [("Repeat", config["repeats"]), ("Fold", config["folds"])]:
        if not frame[col].isin(range(n)).all():
            raise ValueError(f"Invalid {col}")
    identity = ["Dataset", "Data_Seed", "Repeat", "Fold"]
    if frame.groupby(identity)[["Split_ID", "Partition_ID", "N_Train", "N_Test"]].nunique().gt(1).any().any():
        raise ValueError("Mismatched split identities or sizes across methods")
    if frame.groupby(["Dataset", "Data_Seed"])[["Task", "Estimator", "N_Features"]].nunique().gt(1).any().any():
        raise ValueError("Inconsistent dataset metadata")
    numeric = ["Score", "Selection_Time_Sec", "Downstream_Actual_Time_Sec", "N_Train", "N_Test"]
    if not np.isfinite(frame[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite scores or resource measurements")
    if frame[["N_Train", "N_Test"]].le(0).any().any() or frame[numeric[1:3]].lt(0).any().any():
        raise ValueError("Invalid sizes or negative runtimes")
    if not frame.CPU_Budget.eq(config["cpu_budget"]).all():
        raise ValueError("CPU budget differs from the manifest")
    for method in config["methods"] + augmented_methods(manifest):
        g = frame[frame.Method.eq(method["name"])]
        backend = "threading" if method["outer"] == 1 else "loky_outer_threading_inner"
        if not (g.Outer_Jobs.eq(method["outer"]) & g.Estimator_Jobs.eq(method["inner"])
                & g.Selection_Backend.eq(backend)).all():
            raise ValueError(f"Execution configuration mismatch: {method['name']}")
    for row in frame.itertuples(index=False):
        selected = row.Selected_Features
        if (len(selected) != row.N_Selected or len(set(selected)) != len(selected)
                or any(not isinstance(i, int) or i < 0 or i >= row.N_Features for i in selected)):
            raise ValueError("Invalid feature IDs or selection counts")
        mask = np.zeros(row.N_Features, dtype=np.uint8)
        mask[selected] = 1
        if hashlib.sha256(mask.tobytes()).hexdigest() != row.Mask_ID:
            raise ValueError("Feature mask hash mismatch")
        kind, spec = config["datasets"][row.Dataset]
        expected_task = "class" if kind == "synthetic" else spec[1]
        if row.Task != expected_task:
            raise ValueError("Task differs from the manifest")
        if kind == "synthetic":
            relevant = relevant_features(spec)
            tp, fp = len(set(selected) & relevant), len(set(selected) - relevant)
            fn, tn = len(relevant) - tp, row.N_Features - len(relevant) - fp
            expected_metrics = dict(TP=tp, FP=fp, FN=fn, TN=tn,
                                    Recall=tp / len(relevant) if relevant else np.nan,
                                    Precision=tp / len(selected) if selected else np.nan,
                                    FDP=fp / len(selected) if selected else 0.,
                                    FPR=fp / (fp + tn) if fp + tn else np.nan,
                                    F1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan)
            for metric, expected_value in expected_metrics.items():
                actual = getattr(row, metric, np.nan)
                if not np.isclose(actual, expected_value, equal_nan=True):
                    raise ValueError(f"Inconsistent ground-truth metric: {metric}")
    # One selector fit can have multiple output policies or baseline alpha copies.
    jobs = frame.groupby(["Partition_ID", "Job_ID"])
    if jobs.Selection_Time_Sec.nunique().gt(1).any():
        raise ValueError("Inconsistent shared selector timing")
    # Cached masks must give identical scores within a partition.
    if frame.groupby(["Partition_ID", "Mask_ID"]).Score.nunique().gt(1).any():
        raise ValueError("Inconsistent scores for an identical cached mask")
    if not frame.loc[~frame.Downstream_Fit_Performed, "Downstream_Actual_Time_Sec"].eq(0).all():
        raise ValueError("Cached score charged as a new downstream fit")
    if frame.loc[frame.Downstream_Fit_Performed, "Downstream_Source"].duplicated().any():
        raise ValueError("Downstream fit counted more than once")


def coverage_table(frame, manifest):
    keys = ["Dataset", "Data_Seed", "Alpha", "Method", "Policy"]
    count = frame.groupby(keys).size().rename("Observed").reset_index()
    result = pd.DataFrame(expected_groups(manifest)).merge(count, on=keys, how="left")
    result["Observed"] = result.Observed.fillna(0).astype(int)
    result["Expected"] = manifest["config"]["repeats"] * manifest["config"]["folds"]
    result["Complete"] = result.Observed.eq(result.Expected)
    result["Fraction"] = result.Observed / result.Expected
    result["Label"] = [LABELS.get((m, p), f"{m} / {p}") for m, p in zip(result.Method, result.Policy)]
    return result


def load_run(directory):
    """Read each atomically saved checkpoint once, without touching the live run.

    A live read spans an interval, not a transaction. Exact bytes of every input
    are hashed, and the notebook exports the actual analyzed rows as a snapshot.
    Aggregate JSONL/CSV exports are intentionally never combined with checkpoints.
    """
    directory = Path(directory).resolve()
    started = datetime.now(timezone.utc).isoformat()
    manifest_bytes = (directory / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("protocol") != "publication-v6-fresh-cached-scoring":
        raise ValueError("This analysis requires publication-v6 checkpoints")
    progress_path = directory / "progress.json"
    progress = json.loads(progress_path.read_bytes()) if progress_path.exists() else {}
    rows, inputs = [], []
    paths = sorted((directory / "checkpoints").glob("*.json"))
    for path in paths:
        raw = path.read_bytes()
        inputs.append(dict(file=path.name, sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw)))
        checkpoint = json.loads(raw)
        for job_id, job in checkpoint["jobs"].items():
            for row in job["rows"]:
                if row["Job_ID"] != job_id or row["Partition_ID"] != path.stem:
                    raise ValueError(f"Checkpoint identity mismatch: {path.name}")
                rows.append(row)
    if not rows:
        raise ValueError("No completed checkpoint jobs are available yet")
    frame = pd.DataFrame(rows)
    validate_frame(frame, manifest)
    frame["Suite"] = frame.Dataset.map({k: v[0] for k, v in manifest["config"]["datasets"].items()})
    frame["Label"] = [LABELS.get((m, p), f"{m} / {p}") for m, p in zip(frame.Method, frame.Policy)]
    frame["Selected_Fraction"] = frame.N_Selected / frame.N_Features
    frame["Any_Selected"] = frame.N_Selected.gt(0).astype(float)
    dataset_order = {name: i for i, name in enumerate(manifest["config"]["datasets"])}
    frame["_order"] = frame.Dataset.map(dataset_order)
    frame = frame.sort_values(["_order", "Data_Seed", "Alpha", "Repeat", "Fold", "Method", "Policy"]).drop(columns="_order").reset_index(drop=True)
    provenance = dict(run=directory.name, source="atomic checkpoint jobs", read_started_utc=started,
                      read_finished_utc=datetime.now(timezone.utc).isoformat(),
                      manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(), inputs=inputs,
                      python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__,
                      matplotlib=mpl.__version__, progress_at_read=progress)
    return RunData(directory, manifest, progress, frame,
                   coverage_table(frame, manifest), provenance)


def complete_blocks(run):
    """All planned methods/policies and repeat/fold cells must exist in a block.

    Augmented methods are kept only in blocks where they are complete too; a
    missing augmentation never removes prespecified comparisons.
    """
    keys = ["Dataset", "Data_Seed", "Alpha"]
    coverage = run.coverage
    ready = coverage[~coverage.Augmented].groupby(keys).Complete.all().rename("Ready").reset_index()
    frame = run.frame.merge(ready[ready.Ready][keys], on=keys, validate="many_to_one")
    extra = coverage[coverage.Augmented]
    if extra.empty:
        return frame
    done = extra.groupby(keys + ["Method"]).Complete.all().rename("Done").reset_index()
    frame = frame.merge(done, on=keys + ["Method"], how="left")
    return frame[frame.Done.isna() | frame.Done.eq(True)].drop(columns="Done").reset_index(drop=True)


def summary_table(frame):
    metrics = ["Score", "Selection_Time_Sec", "Selection_CPU_Sec", "N_Selected", "Selected_Fraction", "Iterations",
               "RSS_Peak_MB", "RSS_Peak_Increase_MB", "Downstream_Time_Sec",
               "TP", "FP", "FN", "TN", "Recall", "Precision", "F1", "FPR", "FDP",
               "FP_Original_Fraction", "Any_Selected"]
    metrics = [m for m in metrics if m in frame]
    result = frame.groupby(GROUP + ["Suite", "Label"])[metrics].agg(["mean", "std", "median", "count"])
    result.columns = ["_".join(c) for c in result.columns]
    return result.reset_index()


def paired_comparisons(frame, baseline="Vanilla", policy="rough_fix"):
    reference = frame[frame.Method.eq(baseline) & frame.Policy.eq(policy)]
    candidates = frame[~frame.Method.eq(baseline)]
    chunks = []
    for _, group in candidates.groupby(["Method", "Policy"]):
        # Only blocks this candidate covers; a missing fold within a block still fails.
        covered = reference.merge(group[BLOCK].drop_duplicates(), on=BLOCK)
        pair = group.merge(covered[PAIR + ["Score", "Selection_Time_Sec"]], on=PAIR,
                           how="outer", suffixes=("", "_Reference"), indicator=True,
                           validate="one_to_one")
        if not pair["_merge"].eq("both").all():
            raise ValueError("Unmatched comparison splits; use complete blocks first")
        pair["Score_Difference"] = pair.Score - pair.Score_Reference
        # NoSelection is not a selector and its near-zero bookkeeping time is not a speedup.
        pair["Speedup"] = np.where(pair.Method.str.startswith("Greedy") & pair.Selection_Time_Sec.gt(0),
                                   pair.Selection_Time_Sec_Reference / pair.Selection_Time_Sec, np.nan)
        chunks.append(pair.drop(columns="_merge"))
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def holm_adjust(pvalues, family_size=None):
    """Holm with unobserved planned tests conservatively assigned p=1."""
    p = np.asarray(pvalues, float)
    size = len(p) if family_size is None else family_size
    if size < len(p):
        raise ValueError("Family size cannot be smaller than observed tests")
    order = np.argsort(p)
    adjusted = np.empty(len(p))
    adjusted[order] = np.minimum(1, np.maximum.accumulate((size - np.arange(len(p))) * p[order]))
    return adjusted


def noninferiority(paired, run, sensitivity_levels=(0.025,)):
    """Same corrected resampled-t calculation and reference as the v6 runner.

    Formal decisions are withheld until the entire manifest design is complete.
    Interim numerical estimates are diagnostic and not sequentially valid tests.
    Only prespecified methods enter the family; augmented methods are descriptive.
    Sensitivity levels add a labelled bound and Holm decision at that level; the
    confirmatory decision always uses the manifest's test level.
    """
    config = run.manifest["config"]
    level = config["test_level"]
    margins = {"class": config["f1_margin"], "reg": config["r2_margin"]}
    candidates = sum(m["name"] != "Vanilla" for m in config["methods"])
    blocks = run.coverage[["Dataset", "Data_Seed", "Alpha"]].drop_duplicates().shape[0]
    family = blocks * candidates
    rows = []
    if not paired.empty:
        paired = paired[paired.Method.isin({m["name"] for m in config["methods"]})]
    if paired.empty:
        return pd.DataFrame()
    for key, group in paired.groupby(GROUP + ["Label"]):
        meta = dict(zip(GROUP + ["Label"], key))
        margin = margins[meta["Task"]]
        if margin is None or not np.isfinite(margin) or margin <= 0:
            raise ValueError("A positive prespecified NI margin is required in the manifest")
        d = group.Score_Difference.to_numpy()
        n = len(d)
        if n < 2:
            raise ValueError("NI requires at least two paired observations")
        se = np.sqrt((1 / n + (group.N_Test / group.N_Train).mean()) * d.var(ddof=1))
        mean = d.mean()
        p = float(t.sf((mean + margin) / se, n - 1)) if se else float(mean <= -margin)
        rows.append(dict(meta, N_Pairs=n, Mean_Difference=mean, Margin=margin,
                         Corrected_SE=se, Lower_One_Sided_CI=mean - t.ppf(1 - level, n - 1) * se,
                         CI_Level=1 - level, P_NI=p, Zero_Variance=bool(se == 0),
                         **{f"Lower_Bound_{100 * (1 - s):g}": mean - t.ppf(1 - s, n - 1) * se
                            for s in sensitivity_levels}))
    result = pd.DataFrame(rows)
    result["Planned_Family_Size"] = family
    result["P_NI_Holm"] = holm_adjust(result.P_NI, family)
    final = run.complete and not config.get("smoke", False)
    def decide(threshold):
        return np.where(result.P_NI_Holm < threshold, "Non-inferiority established",
                        "Non-inferiority not established") if final else "Withheld: incomplete or smoke run"
    result["Decision"] = decide(level)
    for s in sensitivity_levels:
        result[f"Decision_Level_{s:g}"] = decide(s)
    result["Analysis_Status"] = "Final" if final else "Interim descriptive only"
    return result


def speedup_summary(paired):
    greedy = paired[paired.Method.str.startswith("Greedy")]
    return greedy.groupby(GROUP + ["Label"]).Speedup.agg(
        Median="median", Q25=lambda s: s.quantile(.25), Q75=lambda s: s.quantile(.75),
        Geometric_Mean=lambda s: np.exp(np.log(s).mean()), N_Pairs="count").reset_index()


def parallel_ablation(frame):
    """Pair j=1 and j>1 at the same shadow count, not ratios of marginal means.

    Runs may fit serial Greedy for only some shadow fractions; only those are paired.
    """
    g = frame[frame.Method.str.startswith("Greedy_jobs")].copy()
    keys = PAIR + ["N_Shadows"]
    fraction = g.Method.str.split("_shadows").str[1]
    serial = g[g.Outer_Jobs.eq(1)]
    parallel = g[g.Outer_Jobs.gt(1) & fraction.isin(set(fraction[g.Outer_Jobs.eq(1)]))]
    joined = parallel.merge(serial[keys + ["Selection_Time_Sec", "Score", "Mask_ID"]],
                            on=keys, suffixes=("", "_Serial"), how="outer",
                            validate="one_to_one", indicator=True)
    if not joined["_merge"].eq("both").all():
        raise ValueError("Unmatched serial/parallel configurations")
    joined["Parallel_Speedup"] = joined.Selection_Time_Sec_Serial / joined.Selection_Time_Sec
    joined["Parallel_Score_Difference"] = joined.Score - joined.Score_Serial
    joined["Identical_Mask"] = joined.Mask_ID.eq(joined.Mask_ID_Serial)
    return joined.drop(columns="_merge")


def stability_tables(frame, manifest):
    """Raw Jaccard; feature groups use this run's generator, not old notebook IDs."""
    stability, frequencies = [], []
    for key, group in frame.groupby(GROUP + ["Label", "Suite"]):
        meta = dict(zip(GROUP + ["Label", "Suite"], key))
        p = int(group.N_Features.iloc[0])
        sets = [set(s) for s in group.Selected_Features]
        scores = [len(a & b) / len(a | b) if a | b else 1. for a, b in itertools.combinations(sets, 2)]
        stability.append(dict(meta, N_Fits=len(sets), N_Set_Pairs=len(scores),
                              Mean_Pairwise_Jaccard=np.mean(scores) if scores else np.nan,
                              Mean_Selected=group.N_Selected.mean()))
        # Feature frequencies for synthetic data are compact and interpretable.
        if meta["Suite"] == "synthetic":
            _, spec = manifest["config"]["datasets"][meta["Dataset"]]
            k, q = spec["n_informative"], spec.get("n_correlated", 0)
            count = np.zeros(p)
            for selected in sets:
                count[list(selected)] += 1
            for i, frequency in enumerate(count / len(sets)):
                role = "Direct" if i < k else ("Proxy" if i < k + q and spec.get("correlation", 0) > 0 else "Noise")
                frequencies.append(dict(meta, Feature=i, Role=role, Selection_Frequency=frequency))
    return pd.DataFrame(stability), pd.DataFrame(frequencies)


def resource_accounting(frame):
    """Actual jobs, not replicated output policies or copied cached scoring costs.

    Covers the prespecified run only; augmentation fits are accounted separately.
    """
    frame = frame[~frame.Method.eq(COUPLED)]
    jobs = frame.drop_duplicates(["Partition_ID", "Job_ID"])
    fits = jobs[~jobs.Method.eq("NoSelection")]
    return pd.DataFrame([
        ("Logged result rows", len(frame), "rows"),
        ("Completed method jobs (includes baseline)", len(jobs), "jobs"),
        ("Selector fits (Vanilla policies counted once)", len(fits), "fits"),
        ("Actually performed downstream fits", int(frame.Downstream_Fit_Performed.sum()), "fits"),
        ("Selector fit time", fits.Selection_Time_Sec.sum() / 3600, "hours"),
        ("Actual downstream fit time", frame.Downstream_Actual_Time_Sec.sum() / 3600, "hours"),
    ], columns=["Quantity", "Value", "Unit"])


def apply_theme():
    mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 11, "axes.labelsize": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 110, "savefig.dpi": 300,
                         "pdf.fonttype": 42, "ps.fonttype": 42})


def coverage_plot(run):
    keys = ["Dataset", "Alpha"]
    coverage = run.coverage.groupby(keys)[["Observed", "Expected"]].sum().reset_index()
    names = list(run.manifest["config"]["datasets"])
    alphas = run.manifest["config"]["alphas"]
    fig, ax = plt.subplots(figsize=(11, 6), layout="constrained")
    height = .75 / len(alphas)
    for i, alpha in enumerate(alphas):
        g = coverage[coverage.Alpha.eq(alpha)].set_index("Dataset").reindex(names)
        y = np.arange(len(names)) + (i - (len(alphas) - 1) / 2) * height
        ax.barh(y, g.Observed / g.Expected, height=height, label=f"alpha = {alpha:g}")
    ax.set_yticks(range(len(names)), names)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Fraction of planned fold/policy rows saved")
    ax.set_title(f"Checkpoint coverage: {len(run.frame):,} / {run.manifest['workload']['result_rows']:,} rows")
    ax.legend(loc="lower right")
    return fig


def save_figure(fig, output, name):
    output = Path(output) / "figures"
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ["pdf", "png"]:
        fig.savefig(output / f"{name}.{suffix}", bbox_inches="tight", facecolor="white")
    return fig


def heatmap(frame, metric, title, *, alphas=None, labels=None, center=False,
            bounds=None, fmt=".2f", agg="mean"):
    """Explicit missing cells; facet alpha and keep dataset realizations separate."""
    if frame.empty:
        fig, ax = plt.subplots(figsize=(9, 2))
        ax.text(.5, .5, "No complete observations available", ha="center", va="center")
        ax.set_axis_off()
        return fig
    data = frame.copy()
    data["Realization"] = data.Dataset + " / seed " + data.Data_Seed.astype(str)
    alphas = list(alphas if alphas is not None else sorted(data.Alpha.unique(), reverse=True))
    labels = labels or [label for label in ORDER if label in set(data.Label)]
    datasets = list(dict.fromkeys(data.Realization))
    matrices = [data[data.Alpha.eq(a)].pivot_table(index="Label", columns="Realization",
                values=metric, aggfunc=agg).reindex(index=labels, columns=datasets) for a in alphas]
    finite = np.concatenate([m.to_numpy().ravel() for m in matrices])
    finite = finite[np.isfinite(finite)]
    if bounds:
        vmin, vmax = bounds
    elif center:
        vmax = max(abs(finite).max(), .001) if len(finite) else 1
        vmin = -vmax
    else:
        vmin, vmax = (finite.min(), finite.max()) if len(finite) else (0, 1)
        if vmin == vmax:
            vmax = vmin + 1
    cmap = mpl.colormaps["RdBu" if center else "viridis"].copy()
    cmap.set_bad("#eeeeee")
    fig, axes = plt.subplots(1, len(alphas), figsize=(max(11, len(alphas) * len(datasets) * .63 + 3),
                                                      max(4, len(labels) * .44 + 2.5)),
                             squeeze=False, layout="constrained")
    for ax, alpha, matrix in zip(axes.flat, alphas, matrices):
        values = matrix.to_numpy(dtype=float)
        im = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(datasets)), datasets, rotation=55, ha="right", fontsize=8)
        ax.set_yticks(range(len(labels)), labels, fontsize=9)
        ax.set_title(f"alpha = {alpha:g}")
        for (i, j), value in np.ndenumerate(values):
            scaled = (value - vmin) / (vmax - vmin)
            color = "white" if (center and (scaled < .22 or scaled > .78)) or (not center and scaled < .45) else "black"
            ax.text(j, i, format(value, fmt) if np.isfinite(value) else "NA", ha="center", va="center",
                    fontsize=8, color=color if np.isfinite(value) else "#777777")
    fig.colorbar(im, ax=list(axes.flat), shrink=.7, pad=.02)
    fig.suptitle(title)
    return fig


def score_runtime_plot(frame, alpha):
    data = frame[frame.Alpha.eq(alpha)]
    groups = list(data.groupby(["Dataset", "Data_Seed", "Task"], sort=False))
    fig, axes = plt.subplots(max(1, int(np.ceil(len(groups) / 3))), 3,
                             figsize=(15, max(4, int(np.ceil(len(groups) / 3)) * 3.6)),
                             squeeze=False, layout="constrained")
    for ax in axes.flat:
        ax.set_visible(False)
    for ax, ((dataset, seed, task), group) in zip(axes.flat, groups):
        ax.set_visible(True)
        for label in ORDER:
            g = group[group.Label.eq(label)]
            if g.empty:
                continue
            if label == "All features":
                ax.axhline(g.Score.mean(), color=COLORS[label], ls=":", label=label)
                continue
            x, y = g.Selection_Time_Sec.median(), g.Score.mean()
            ax.scatter(x, y, color=COLORS[label], marker="s" if "Vanilla" in label else "o",
                       s=45, label=label, edgecolors="white", linewidths=.5)
        ax.set_xscale("log")
        ax.set_title(f"{dataset} / seed {seed}")
        ax.set_xlabel("Median selector time (seconds, log scale)")
        ax.set_ylabel("Mean weighted F1" if task == "class" else "Mean R²")
        ax.grid(alpha=.2)
    if groups:
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=9)
    fig.suptitle(f"Predictive performance and selection cost | alpha = {alpha:g}")
    return fig


def ni_plot(table, alpha):
    data = table[table.Alpha.eq(alpha)] if not table.empty else table
    groups = list(data.groupby(["Dataset", "Data_Seed", "Task"], sort=False)) if not data.empty else []
    fig, axes = plt.subplots(max(1, int(np.ceil(len(groups) / 2))), 2,
                             figsize=(14, max(3.5, int(np.ceil(len(groups) / 2)) * 3.5)),
                             squeeze=False, layout="constrained")
    for ax in axes.flat:
        ax.set_visible(False)
    for ax, ((dataset, seed, task), group) in zip(axes.flat, groups):
        ax.set_visible(True)
        for i, row in enumerate(group.itertuples()):
            ax.plot([row.Lower_One_Sided_CI, row.Mean_Difference], [i, i], color=COLORS[row.Label], lw=2)
            ax.scatter(row.Mean_Difference, i, color=COLORS[row.Label], s=30)
            ax.scatter(row.Lower_One_Sided_CI, i, color=COLORS[row.Label], marker="|", s=60)
        ax.axvline(0, color="gray", lw=1)
        ax.axvline(-group.Margin.iloc[0], color="#D55E00", ls="--", lw=1)
        ax.set_yticks(range(len(group)), group.Label, fontsize=9)
        ax.invert_yaxis()
        ax.set_title(f"{dataset} / seed {seed}")
        ax.set_xlabel("Candidate − Vanilla rough fix (weighted F1)" if task == "class" else "Candidate − Vanilla rough fix (R²)")
        ax.grid(axis="x", alpha=.2)
    status = table.Analysis_Status.iloc[0] if not table.empty else "No complete blocks"
    level = f"{100 * table.CI_Level.iloc[0]:g}%" if not table.empty else ""
    fig.suptitle(f"{status} | mean and {level} one-sided lower bound | alpha = {alpha:g}\n"
                 "Dashed: prespecified loss margin. Segment ends at mean; not a two-sided CI.")
    return fig


def export_tables(run, tables, output):
    """Only writes beneath the analysis output, never beneath experiment inputs."""
    output = Path(output).resolve()
    if output == run.directory or run.directory in output.parents:
        raise ValueError("Analysis outputs must be outside the source run")
    output.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False)
    snapshot = output / "analyzed_rows.jsonl.gz"
    with gzip.open(snapshot, "wt", encoding="utf-8") as stream:
        stream.write(run.frame.to_json(orient="records", lines=True, double_precision=15))
    (output / "source_manifest.json").write_text(json.dumps(run.manifest, indent=2), encoding="utf-8")
    provenance = dict(run.provenance, result_rows=len(run.frame), complete=run.complete,
                      snapshot_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                      helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output / "analysis_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")


# =============================================================================
# Paper results: coupled augmentation, Vanilla histories, figures and tables
# =============================================================================

PAPER_GREEDY = "Greedy j=1, s=1"  # j=1 and j>1 select identical masks (checked by parallel_ablation)
FATE_LAYERS = ["Strong (Vanilla confirms at K)", "Vanilla confirms after K",
               "Vanilla rough fix", "Greedy only"]


def paper_style():
    """Print-oriented theme for LNCS figures (light surface, recessive axes)."""
    apply_theme()
    mpl.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                         "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
                         "axes.edgecolor": PAPER["faint"], "axes.labelcolor": PAPER["ink"],
                         "xtick.color": PAPER["muted"], "ytick.color": PAPER["muted"],
                         "axes.titlecolor": PAPER["ink"], "lines.linewidth": 1.5,
                         "axes.grid": True, "grid.color": PAPER["grid"], "grid.linewidth": .6,
                         "legend.frameon": False, "figure.dpi": 130})


def greedy_parallel_label(frame):
    """Label of the full-shadow Greedy run with the largest outer worker count."""
    names = frame.loc[frame.Method.str.match(r"Greedy_jobs\d+_shadows1(\.0)?$"), "Method"].unique()
    best = max(names, key=lambda n: int(n.split("_jobs")[1].split("_")[0]))
    return LABELS[(best, "selected")]


def parallel_quantity(frame):
    return f"Greedy parallel ({greedy_parallel_label(frame).split(', ')[0].split(' ')[1]})"


def vanilla_convergence(frame, manifest):
    """Iteration cap hits and rough-fix promotions per dataset/alpha (all folds)."""
    cap = manifest["config"]["vanilla_iterations"]
    v = frame[frame.Method.eq("Vanilla")].pivot_table(
        index=PAIR, columns="Policy", values=["N_Selected", "Iterations"], aggfunc="first")
    v.columns = [f"{a}_{b}" for a, b in v.columns]
    v = v.reset_index()
    g = frame[frame.Label.eq(PAPER_GREEDY)][PAIR + ["Iterations"]].rename(columns={"Iterations": "K"})
    v = v.merge(g, on=PAIR, how="left")
    v["At_Cap"] = v.Iterations_strict.ge(cap)
    v["Rough_Fix_Promoted"] = v.N_Selected_rough_fix - v.N_Selected_strict
    v["Iteration_Ratio"] = v.Iterations_strict / v.K
    return v.groupby(BLOCK).agg(K=("K", "first"), Median_Iterations=("Iterations_strict", "median"),
                                Min_Iterations=("Iterations_strict", "min"),
                                Fraction_At_Cap=("At_Cap", "mean"),
                                Fraction_Rough_Fix_Used=("Rough_Fix_Promoted", lambda s: s.gt(0).mean()),
                                Mean_Rough_Fix_Promoted=("Rough_Fix_Promoted", "mean"),
                                Max_Rough_Fix_Promoted=("Rough_Fix_Promoted", "max"),
                                Median_Iteration_Ratio=("Iteration_Ratio", "median"),
                                N_Folds=("At_Cap", "size")).reset_index()


def history_records(directory):
    """Verified full Vanilla refits written by augment_coupled.py."""
    paths = sorted((Path(directory) / "augmentations" / "vanilla_history").glob("*.json"))
    for path in paths:
        if path.name.endswith(".failed.json"):
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        if not all(record["Verification"].values()):
            raise ValueError(f"History failed verification: {path.name}")
        yield record


def load_histories(directory):
    """(per-refit summary, long per-iteration trajectories); empty frames if absent."""
    summaries, trajectories = [], []
    for record in history_records(directory):
        trajectory = pd.DataFrame({k: v for k, v in record["Trajectory"].items() if k != "n_features"})
        meta = {k: record[k] for k in ["Dataset", "Data_Seed", "Repeat", "Fold", "Partition_ID",
                                       "Alpha", "N_Features", "K"]}
        trajectories.append(trajectory.assign(**meta))
        summaries.append(dict({k: v for k, v in record.items() if not isinstance(v, (list, dict))},
                              Same_Run_Speedup=record["Selection_Time_Sec"] / record["Time_At_K_Sec"],
                              Session_Time_Ratio=record["Selection_Time_Sec"] / record["Logged_Selection_Time_Sec"],
                              Rough_Fix_Promoted_Count=len(record["Rough_Fix_Promoted"])))
    if not summaries:
        return pd.DataFrame(), pd.DataFrame()
    trajectories = pd.concat(trajectories, ignore_index=True)
    trajectories["Tentative_Fraction"] = trajectories.tentative / trajectories.N_Features
    return pd.DataFrame(summaries), trajectories


def coupled_sets(frame, manifest):
    """Nested decomposition of each coupled Greedy selection by Vanilla's outcome.

    Strong <= Vanilla strict <= Vanilla rough fix <= Greedy holds exactly for the
    coupled method (verified while augmenting); each layer is a set difference.
    """
    coupled = frame[frame.Method.eq(COUPLED)]
    if coupled.empty:
        return pd.DataFrame()
    v = frame[frame.Method.eq("Vanilla")]
    strict = v[v.Policy.eq("strict")].set_index(PAIR).Selected_Features
    rough = v[v.Policy.eq("rough_fix")].set_index(PAIR).Selected_Features
    datasets = manifest["config"]["datasets"]
    rows = []
    for row in coupled.itertuples(index=False):
        key = tuple(getattr(row, k) for k in PAIR)
        g, s, r = set(row.Selected_Features), set(strict.loc[key]), set(rough.loc[key])
        strong = set(row.Strong_Features)
        if not (strong <= s <= r <= g):
            raise ValueError("Coupled nesting violated; augmentation verification should have failed")
        kind, spec = datasets[row.Dataset]
        truth = relevant_features(spec) if kind == "synthetic" else None
        out = dict(zip(PAIR, key), N_Features=row.N_Features, K=row.Iterations, N_Greedy=len(g),
                   N_Vanilla_Rough_Fix=len(r), N_Strong=len(strong), N_Weak=len(g) - len(strong))
        for name, layer in zip(FATE_LAYERS, [strong, s - strong, r - s, g - r]):
            out[name] = len(layer)
            if truth is not None:
                out[name + " FP"] = len(layer - truth)
        if truth is not None:
            out.update(FP_Greedy=len(g - truth), FP_Strong=len(strong - truth),
                       FP_Weak=len((g - strong) - truth), M0=row.N_Features - len(truth))
        rows.append(out)
    return pd.DataFrame(rows)


def history_fates(directory, frame):
    """For refit partitions: what Vanilla finally did with coupled Greedy's weak features."""
    coupled = frame[frame.Method.eq(COUPLED)].set_index(["Partition_ID", "Alpha"])
    rows = []
    for record in history_records(directory):
        key = (record["Partition_ID"], record["Alpha"])
        if key not in coupled.index:
            continue
        strong = set(coupled.loc[key, "Strong_Features"])
        weak = np.asarray([j for j in coupled.loc[key, "Selected_Features"] if j not in strong], dtype=int)
        final = np.asarray(record["Final_Decision"])[weak]
        promoted = set(record["Rough_Fix_Promoted"])
        decided = np.asarray(record["Decision_Iteration"])[weak]
        rows.append(dict(Dataset=record["Dataset"], Alpha=record["Alpha"], Repeat=record["Repeat"],
                         Fold=record["Fold"], N_Weak=len(weak),
                         Weak_Confirmed=int((final == 1).sum()),
                         Weak_Tentative_Promoted=int(sum(j in promoted for j in weak[final == 0])),
                         Weak_Tentative_Dropped=int(sum(j not in promoted for j in weak[final == 0])),
                         Weak_Rejected=int((final == -1).sum()),
                         Weak_Median_Decision_Iteration=float(np.median(decided[decided > 0]))
                         if (decided > 0).any() else np.nan))
    return pd.DataFrame(rows)


def speedup_decomposition(paired, frame, histories=None):
    """Median paired speedups next to the 'same work, stopped at K' baseline.

    Iteration ratio: Vanilla iterations / K (naive expectation).
    Same work stopped at K: logged Vanilla time / elapsed time of the coupled prefix
    at iteration K (identical computation, measured in a separate session).
    Same run stopped at K: a history refit's total time / its own elapsed time at K.
    Observed: logged Vanilla time / logged Greedy time.
    Per-iteration cost ratio: mean seconds per iteration of the shared first K
    iterations / mean seconds per Vanilla iteration over the whole fit.
    """
    iters = frame[frame.Method.eq("Vanilla") & frame.Policy.eq("rough_fix")][PAIR + ["Iterations"]]
    p = paired.merge(iters.rename(columns={"Iterations": "Vanilla_Iterations"}), on=PAIR)
    parallel, parallel_name = greedy_parallel_label(frame), parallel_quantity(frame)
    rows = []

    def add(block, quantity, values):
        values = pd.Series(values, dtype=float).dropna()
        if len(values):
            rows.append(dict(zip(BLOCK, block), Quantity=quantity, Median=values.median(),
                             Q25=values.quantile(.25), Q75=values.quantile(.75), N=len(values)))
    for block, g in p.groupby(BLOCK):
        serial, coupled = g[g.Label.eq(PAPER_GREEDY)], g[g.Method.eq(COUPLED)]
        add(block, "Iteration ratio", serial.Vanilla_Iterations / serial.Iterations)
        add(block, "Same work stopped at K", coupled.Speedup)
        add(block, "Greedy serial (j=1)", serial.Speedup)
        add(block, parallel_name, g.loc[g.Label.eq(parallel), "Speedup"])
        add(block, "Per-iteration cost ratio",
            (coupled.Selection_Time_Sec / coupled.Iterations)
            / (coupled.Selection_Time_Sec_Reference / coupled.Vanilla_Iterations))
    if histories is not None and not histories.empty:
        h = histories.merge(frame[BLOCK].drop_duplicates(), on=["Dataset", "Data_Seed", "Alpha"])
        for block, g in h.groupby(BLOCK):
            add(block, "Same run stopped at K (history)", g.Same_Run_Speedup)
    return pd.DataFrame(rows)


def _row_axes(n, ncols, width=7.0, height=1.9):
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height * nrows), squeeze=False,
                             layout="constrained")
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    return fig, list(axes.flat)






def fp_bound_plot(frame, manifest, scenarios=None, ncols=4):
    """Mean false positives per fold vs alpha (IQR whiskers), with the m0 K/(p+1) bound."""
    datasets = manifest["config"]["datasets"]
    names = scenarios or [n for n, (kind, spec) in datasets.items()
                          if kind == "synthetic" and spec["kind"] != "xor"]
    series = [("Vanilla strict", dict(marker="s", mfc="white", mec=PAPER["vanilla"], color=PAPER["vanilla"])),
              ("Vanilla rough fix", dict(marker="s", mfc=PAPER["vanilla"], mec=PAPER["vanilla"], color=PAPER["vanilla"])),
              (PAPER_GREEDY, dict(marker="o", mfc=PAPER["greedy"], mec=PAPER["greedy"], color=PAPER["greedy"]))]
    fig, axes = _row_axes(len(names), ncols, height=2.1)
    for ax, name in zip(axes, names):
        g = frame[frame.Dataset.eq(name)]
        spec = datasets[name][1]
        p, m0 = spec["n_features"], spec["n_features"] - len(relevant_features(spec))
        alphas = sorted(g.Alpha.unique(), reverse=True)
        ks = [int(np.ceil(np.log2(p / a))) for a in alphas]
        ax.plot(range(len(alphas)), [m0 * k / (p + 1) for k in ks], color=PAPER["muted"], ls="--",
                lw=1, label="Bound $m_0K/(p+1)$")
        for i, (label, style) in enumerate(series):
            s = g[g.Label.eq(label)].groupby("Alpha").FP
            mean = s.mean().reindex(alphas)
            lo, hi = s.quantile(.25).reindex(alphas), s.quantile(.75).reindex(alphas)
            x = np.arange(len(alphas)) + (i - 1) * .09
            ax.errorbar(x, mean, yerr=[mean - lo, hi - mean], fmt="none", ecolor=style["color"],
                        elinewidth=.9, alpha=.6)
            ax.plot(x, mean, ms=5, lw=1, label="Greedy" if label == PAPER_GREEDY else label, **style)
        ax.set_xticks(range(len(alphas)), [f"{a:g}\nK={k}" for a, k in zip(alphas, ks)])
        ax.set_ylim(bottom=0)

        title_aux = {
            "reference": "Reference",
            "n_features_2000": "High-dimensional",
            "signal_0.5": "Weak signal",
            "kind_null": "Null",
        }.get(name)

        ax.set_title(f"{title_aux}\np={p:,}, $m_0$={m0:,}")
        ax.set_xlabel("α")
    axes[0].set_ylabel("False positives per fold")
    handles, labels = axes[0].get_legend_handles_labels()
    # fig.legend(handles, labels, loc="outside upper center", ncol=4)
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=4)

    return fig










def shadow_fp_plot(frame, manifest, scenarios):
    """Reduced shadows: false positives vs 1/(r+1) (log-log) with the K/(r+1) bound."""
    datasets = manifest["config"]["datasets"]
    g = frame[frame.Method.str.match(r"Greedy_jobs\d+_shadows") & frame.Dataset.isin(scenarios)]
    g = g[g.Outer_Jobs.eq(g.Outer_Jobs.max())]
    fig, axes = _row_axes(len(scenarios), len(scenarios), height=2.3)
    for ax, name in zip(axes, scenarios):
        spec = datasets[name][1]
        p, m0 = spec["n_features"], spec["n_features"] - len(relevant_features(spec))
        d = g[g.Dataset.eq(name)]
        for alpha, marker in zip(sorted(d.Alpha.unique(), reverse=True), ["o", "s", "^"]):
            s = d[d.Alpha.eq(alpha)].groupby("N_Shadows").FP.mean()
            ax.plot(1 / (s.index + 1), s, marker=marker, color=PAPER["greedy"], ms=4.5, lw=1,
                    label=f"observed, α={alpha:g}")
            k = int(np.ceil(np.log2(p / alpha)))
            ax.plot(1 / (s.index + 1), m0 * k / (s.index + 1), ls="--", color=PAPER["muted"], lw=.9,
                    marker=marker, mfc="white", mec=PAPER["muted"], ms=3.5, label=f"bound, α={alpha:g}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("1/(r+1)")
        ax.set_title(f"{name} (p={p:,})")
    axes[0].set_ylabel("False positives per fold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3)
    return fig


def fmt_score(mean, sd, digits=3):
    return f"${mean:.{digits}f} \\pm {sd:.{digits}f}$"


def latex_table(frame, caption, label, column_format=None, notes=None, group_rules=True):
    """Minimal booktabs table from preformatted strings; blank first cells continue a group."""
    cols = list(frame.columns)
    lines = [r"\begin{table}", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\small", rf"\begin{{tabular}}{{{column_format or 'l' + 'r' * (len(cols) - 1)}}}",
             r"\toprule", " & ".join(cols) + r" \\", r"\midrule"]
    for i, row in enumerate(frame.itertuples(index=False)):
        values = [str(v) for v in row]
        if group_rules and i and values[0]:
            lines.append(r"\midrule")
        lines.append(" & ".join(values) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if notes:
        lines.append(rf"\par\smallskip\footnotesize {notes}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def real_results_table(summary, ni, decomposition, alphas, datasets, parallel_name):
    """Rows: dataset x alpha. Scores, paired difference, NI bound/decision, sizes, speedups."""
    rows = []
    for name in datasets:
        for alpha in alphas:
            s = summary[summary.Dataset.eq(name) & summary.Alpha.eq(alpha)].set_index("Label")
            n = ni[ni.Dataset.eq(name) & ni.Alpha.eq(alpha)].set_index("Label")
            sp = decomposition[decomposition.Dataset.eq(name) & decomposition.Alpha.eq(alpha)].set_index("Quantity").Median
            if PAPER_GREEDY not in s.index or PAPER_GREEDY not in n.index:
                continue
            g, v, test = s.loc[PAPER_GREEDY], s.loc["Vanilla rough fix"], n.loc[PAPER_GREEDY]
            rows.append({"Dataset": name if alpha == alphas[0] else "", r"$\alpha$": f"{alpha:g}",
                         "Vanilla": fmt_score(v.Score_mean, v.Score_std),
                         "Greedy": fmt_score(g.Score_mean, g.Score_std),
                         r"$\bar D$": f"${test.Mean_Difference:+.3f}$",
                         r"LB$_{99}$": f"${test.Lower_One_Sided_CI:+.3f}$",
                         r"LB$_{97.5}$": f"${test['Lower_Bound_97.5']:+.3f}$",
                         "NI": r"\checkmark" if test.Decision.startswith("Non-inferiority established") else "--",
                         r"\#sel.\ V/G": f"{v.N_Selected_mean:.0f}/{g.N_Selected_mean:.0f}",
                         "Speedup": f"{sp.get('Greedy serial (j=1)', np.nan):.1f}/{sp.get(parallel_name, np.nan):.1f}"})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- figures (v2)

ALPHA_STYLE = {0.1: dict(marker="o", color="#f6b896"), 0.001: dict(marker="s", color="#eb6834"),
               1e-05: dict(marker="^", color="#8f3510")}
FATE_NAMES = {"Strong (Vanilla confirms at K)": "Strong: Vanilla confirms at K",
              "Vanilla confirms after K": "Weak: Vanilla confirms later",
              "Vanilla rough fix": "Weak: kept by Vanilla's rough fix",
              "Greedy only": "Weak: Greedy only"}


def verify_coupled(frame):
    """Re-check the coupled identities on every logged row (independent of the augmentation log)."""
    coupled = frame[frame.Method.eq(COUPLED)]
    vanilla = frame[frame.Method.eq("Vanilla")]
    strict = vanilla[vanilla.Policy.eq("strict")].set_index(PAIR).Selected_Features
    rough = vanilla[vanilla.Policy.eq("rough_fix")].set_index(PAIR).Selected_Features
    greedy_k = frame[frame.Label.eq(PAPER_GREEDY)].set_index(PAIR).Iterations
    rows = []
    for row in coupled.itertuples(index=False):
        key = tuple(getattr(row, k) for k in PAIR)
        chosen = set(row.Selected_Features)
        rows.append(dict(zip(PAIR, key), Theorem1_Strict=set(strict.loc[key]) <= chosen,
                         Theorem1_Rough_Fix=set(rough.loc[key]) <= chosen,
                         Strong_In_Strict=set(row.Strong_Features) <= set(strict.loc[key]),
                         K_Matches_Greedy=bool(greedy_k.loc[key] == row.Iterations)))
    return pd.DataFrame(rows)


def hit_threshold(k, alpha, p):
    """Smallest hit count that passes Vanilla's Bonferroni confirmation test at iteration K."""
    from scipy.stats import binom
    h = np.arange(1, k + 1)
    passing = h[binom.sf(h - 1, k, .5) <= alpha / p]
    return int(passing[0]) if len(passing) else None


def runtime_anatomy_plot(trajectories, alpha, datasets, ncols=3, height=1.75):
    """Per refit: share of features still tentative and cumulative share of Vanilla's runtime."""
    t = trajectories[trajectories.Alpha.eq(alpha)].copy()
    t["Runtime_Share"] = t.elapsed / t.groupby("Partition_ID").elapsed.transform("max")
    names = [n for n in datasets if n in set(t.Dataset)]
    fig, axes = _row_axes(len(names), ncols, height=height)
    for ax, name in zip(axes, names):
        g = t[t.Dataset.eq(name)]
        for _, fold in g.groupby("Partition_ID"):
            ax.plot(fold.iteration, fold.Runtime_Share, color=PAPER["muted"], lw=1.1, alpha=.8)
            ax.plot(fold.iteration, fold.Tentative_Fraction, color=PAPER["vanilla"], lw=1.3, alpha=.85)
        k = int(g.K.iloc[0])
        at_k = g[g.iteration.eq(k)]
        ends = g.sort_values("iteration").groupby("Partition_ID").tail(1)
        ax.axvline(k, color=PAPER["greedy"], lw=1.3, ls="--")
        ax.set_xlim(0, max(t.iteration.max(), 1) * 1.02)
        ax.set_ylim(0, 1.02)
        ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1, decimals=0))
        ax.set_title(f"{name} (p={g.N_Features.iloc[0]:,})\n"
                     f"K={k}: {100 * at_k.Runtime_Share.median():.0f}% of runtime, "
                     f"{100 * at_k.Tentative_Fraction.median():.1f}% tentative", fontsize=7.5)
        ax.text(.98, .06, f"{100 * ends.Tentative_Fraction.median():.1f}% tentative at stop",
                ha="right", va="bottom", fontsize=6.5, color=PAPER["vanilla"], transform=ax.transAxes)
    for ax in axes[::ncols]:
        ax.set_ylabel("Share of runtime")
    for ax in axes[-ncols:]:
        ax.set_xlabel("Vanilla Boruta iteration")
    handles = [mpl.lines.Line2D([], [], color=PAPER["vanilla"], lw=1.5, label="Features still tentative"),
               mpl.lines.Line2D([], [], color=PAPER["muted"], lw=1.5, label="Cumulative share of Vanilla's runtime"),
               mpl.lines.Line2D([], [], color=PAPER["greedy"], lw=1.3, ls="--", label="Greedy's fixed budget K")]
    # fig.legend(handles=handles, loc="outside upper center", ncol=3)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=3)

    return fig


def speedup_plot(decomposition, alpha, datasets, histories=None, parallel_name=None):
    """Dot plot (log x): naive expectation, Vanilla stopped at K (same run), observed Greedy."""
    d = decomposition[decomposition.Alpha.eq(alpha) & decomposition.Dataset.isin(datasets)]
    names = [n for n in datasets if n in set(d.Dataset)]
    parallel_name = parallel_name or next(q for q in d.Quantity.unique() if q.startswith("Greedy parallel"))
    workers = parallel_name.split("j=")[1].rstrip(")")
    styles = [("Iteration ratio", "Vanilla iterations / K",
               dict(marker="D", mfc="white", mec=PAPER["muted"]), PAPER["muted"]),
              ("Same run stopped at K (history)", "Vanilla stopped at iteration K (same run)",
               dict(marker="s", mfc="white", mec=PAPER["third"]), PAPER["third"]),
              ("Greedy serial (j=1)", "Greedy, 1 worker",
               dict(marker="o", mfc="white", mec=PAPER["greedy"]), PAPER["greedy"]),
              (parallel_name, f"Greedy, {workers} workers",
               dict(marker="o", mfc=PAPER["greedy"], mec=PAPER["greedy"]), PAPER["greedy"])]
    styles = [s for s in styles if s[0] in set(d.Quantity)]
    fig, ax = plt.subplots(figsize=(7.0, .36 * len(names) + 1.0), layout="constrained")
    for offset, (quantity, label, style, color) in zip(np.linspace(-.22, .22, len(styles)), styles):
        g = d[d.Quantity.eq(quantity)].set_index("Dataset").reindex(names)
        y = np.arange(len(names)) + offset
        ax.errorbar(g.Median, y, xerr=[g.Median - g.Q25, g.Q75 - g.Median], fmt="none",
                    ecolor=color, elinewidth=1, alpha=.55)
        ax.plot(g.Median, y, ls="none", ms=5.5, mew=1.2, label=label, **style)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}×"))
    ax.axvline(1, color=PAPER["faint"], lw=.8)
    labels = names
    if histories is not None and not histories.empty:
        share = (histories[histories.Alpha.eq(alpha)].assign(S=lambda h: h.Time_At_K_Sec / h.Selection_Time_Sec)
                 .groupby("Dataset").S.median())
        labels = [f"{n}\n{100 * share.get(n, np.nan):.0f}% runtime before K" for n in names]
    ax.set_yticks(range(len(names)), labels, fontsize=7)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"Speedup over Vanilla Boruta (median and IQR over folds, log scale), α = {alpha:g}")
    fig.legend(loc="outside upper center", ncol=2)
    return fig


def hits_plot(frame, manifest, alpha, scenarios, ncols=4):
    """Hit counts of Greedy-selected features (coupled run): relevant vs noise, strong threshold."""
    datasets = manifest["config"]["datasets"]
    coupled = frame[frame.Method.eq(COUPLED) & frame.Alpha.eq(alpha)]
    fig, axes = _row_axes(len(scenarios), ncols, height=1.9)
    for ax, name in zip(axes, scenarios):
        spec = datasets[name][1]
        truth, p = relevant_features(spec), spec["n_features"]
        g = coupled[coupled.Dataset.eq(name)]
        k = int(g.Iterations.iloc[0])
        counts = {True: np.zeros(k + 1), False: np.zeros(k + 1)}
        for row in g.itertuples(index=False):
            for j, h in zip(row.Selected_Features, row.Selected_Hits):
                counts[j in truth][h] += 1
        h = np.arange(1, k + 1)
        threshold = hit_threshold(k, alpha, p)
        if threshold:
            ax.axvspan(threshold - .5, k + .5, color=PAPER["grid"], alpha=.6, lw=0)
            ax.text(threshold - .7, .97, "strong →", transform=ax.get_xaxis_transform(), fontsize=6.6,
                    va="top", ha="right", color=PAPER["muted"])
        for is_relevant, color, label, shift in [(False, PAPER["muted"], "Irrelevant (false positive)", -.2),
                                                 (True, PAPER["third"], "Relevant (true positive)", .2)]:
            per_fold = counts[is_relevant][1:] / len(g)
            if per_fold.sum():
                ax.bar(h + shift, per_fold, width=.4, color=color, label=label, linewidth=0)
        ax.set_xlim(.4, k + .6)

        title_aux = {
            "reference": "Reference",
            "n_features_2000": "High-dimensional",
            "signal_0.5": "Weak signal",
            "kind_null": "Null",
        }.get(name)

        ax.set_title(f"{title_aux} (K={k})")
        ax.set_xlabel("Hits in K iterations")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Features per fold")
    handles = [mpl.patches.Patch(color=PAPER["muted"], label="Irrelevant (false positive)"),
               mpl.patches.Patch(color=PAPER["third"], label="Relevant (true positive)"),
               mpl.patches.Patch(color=PAPER["grid"], label="Strong confirmation region")]
    # fig.legend(handles=handles, loc="outside upper center", ncol=3)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=3)

    return fig


def fates_plot(sets, alpha, groups):
    """Stacked shares of each coupled Greedy selection by Vanilla's outcome, grouped by suite."""
    d = sets[sets.Alpha.eq(alpha)]
    rows, ticks, labels = [], [], []
    position = 0
    for group, names in groups.items():
        names = [n for n in names if n in set(d.Dataset)]
        if not names:
            continue
        rows.append((position, None, group))
        position += .75
        for name in names:
            rows.append((position, name, None))
            position += 1
        position += .35
    means = d.groupby("Dataset")[FATE_LAYERS + ["N_Greedy", "N_Vanilla_Rough_Fix"]].mean()
    fig, ax = plt.subplots(figsize=(7.0, .27 * position + 1.0), layout="constrained")
    for y, name, group in rows:
        if group:
            ax.text(-.01, y, group, fontsize=7.5, fontweight="bold", color=PAPER["ink"], ha="right",
                    va="center", transform=ax.get_yaxis_transform())
            continue
        m = means.loc[name]
        left = 0
        for layer, color in zip(FATE_LAYERS, FATE_COLORS):
            share = m[layer] / m.N_Greedy if m.N_Greedy else 0
            ax.barh(y, share, left=left, color=color, height=.7, edgecolor="white", linewidth=1,
                    label=FATE_NAMES[layer] if name == rows[1][1] else None)
            left += share
        ax.text(1.01, y, f"{m.N_Greedy:,.0f} vs {m.N_Vanilla_Rough_Fix:,.0f}", va="center", fontsize=6.8,
                color=PAPER["muted"], transform=ax.get_yaxis_transform())
        ticks.append(y)

        title_aux = {
            "Madelon": "Madelon",
            "Isolet": "Isolet",
            "Riboflavin": "Riboflavin",
            "Gisette": "Gisette",
            "Arcene": "Arcene",
            "Dexter": "Dexter",
            "reference": "Reference",
            "n_features_2000": "High-dimensional",
            "signal_0.5": "Weak signal",
            "kind_null": "Null",
        }.get(name)

        labels.append(title_aux)
    ax.text(1.01, -.6, "Greedy vs Vanilla\nselected", fontsize=6.8, color=PAPER["muted"], va="bottom",
            transform=ax.get_yaxis_transform())
    ax.set_yticks(ticks, labels)
    ax.set_ylim(position - .3, -.9)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1, decimals=0))
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"Share of Greedy's selected features (coupled runs, mean over 15 folds), α = {alpha:g}")
    # fig.legend(loc="outside upper center", ncol=2)
    fig.legend(loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=2)

    return fig


def alpha_strip_plot(paired, frame, manifest, datasets):
    """Rows: datasets; one marker per alpha. Overlapping markers mean alpha barely matters."""
    parallel = greedy_parallel_label(frame)
    alphas = sorted(frame.Alpha.unique(), reverse=True)
    panels = [(paired[paired.Label.eq(parallel)], "Speedup", "median", "Speedup (Greedy, parallel)", "log"),
              (frame[frame.Label.eq(PAPER_GREEDY)], "N_Selected", "mean", "Features selected by Greedy", "log"),
              (paired[paired.Label.eq(PAPER_GREEDY)], "Score_Difference", "mean",
               "Score difference vs Vanilla", "linear")]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, .3 * len(datasets) + 1.1), sharey=True, layout="constrained")
    y = np.arange(len(datasets))
    for ax, (data, metric, agg, title, scale) in zip(axes, panels):
        values = data[data.Dataset.isin(datasets)].groupby(["Dataset", "Alpha"])[metric].agg(agg).unstack()
        values = values.reindex(index=datasets, columns=alphas)
        if metric == "Score_Difference":
            margin = manifest["config"]["f1_margin"]
            ax.axvspan(-margin, margin, color=PAPER["grid"], alpha=.7, lw=0)
            ax.axvline(0, color=PAPER["faint"], lw=.8)
        for i in y:
            row = values.iloc[i].dropna()
            ax.plot([row.min(), row.max()], [i, i], color=PAPER["faint"], lw=1, zorder=1)
        for alpha in alphas:
            style = ALPHA_STYLE.get(alpha, dict(marker="o", color=PAPER["greedy"]))
            ax.scatter(values[alpha], y, s=26, marker=style["marker"], color=style["color"],
                       edgecolors=PAPER["ink"], linewidths=.4, zorder=3, label=f"α = {alpha:g}")
        ax.set_xscale(scale)
        if scale == "log":
            ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(
                lambda v, _: (f"{v:g}×" if metric == "Speedup" else f"{v:,.0f}")))
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, datasets)
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=len(alphas))
    return fig


def ni_forest_plot(ni, alpha, datasets, candidates=(PAPER_GREEDY, "All features"),
                   sensitivity="Lower_Bound_97.5", ncols=2):
    """Mean paired difference, prespecified one-sided bound (bar) and sensitivity bound (tick)."""
    d = ni[ni.Alpha.eq(alpha) & ni.Dataset.isin(datasets) & ni.Label.isin(candidates)]
    colors = {PAPER_GREEDY: PAPER["greedy"], "All features": PAPER["muted"]}
    fig, axes = _row_axes(len(datasets), ncols, height=1.25)
    for ax, name in zip(axes, datasets):
        g = d[d.Dataset.eq(name)].set_index("Label").reindex(list(candidates))
        margin = g.Margin.dropna().iloc[0]
        for i, (label, row) in enumerate(g.iterrows()):
            if pd.isna(row.Mean_Difference):
                continue
            color = colors.get(label, PAPER["muted"])
            ax.plot([row.Lower_One_Sided_CI, row.Mean_Difference], [i, i], color=color, lw=2.2,
                    solid_capstyle="butt")
            if sensitivity in row and pd.notna(row[sensitivity]):
                ax.plot(row[sensitivity], i, marker="|", ms=10, mew=1.6, color=PAPER["ink"])
            ax.plot(row.Mean_Difference, i, marker="o", ms=5, color=color, mec="white", mew=.6)
            established = str(row.Decision).startswith("Non-inferiority established")
            ax.text(1.0, i, " NI" if established else " inconclusive", transform=ax.get_yaxis_transform(),
                    va="center", fontsize=6.8, color=PAPER["ink"] if established else PAPER["muted"],
                    fontweight="bold" if established else "normal")
        ax.axvline(0, color=PAPER["faint"], lw=.8)
        ax.axvline(-margin, color=PAPER["ink"], lw=.8, ls="--")
        low = min(g.Lower_One_Sided_CI.min(), -margin)
        high = max(g.Mean_Difference.max(), 0)
        pad = .08 * (high - low)
        ax.set_xlim(low - pad, high + pad)
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
        ax.set_yticks(range(len(candidates)), ["Greedy" if c == PAPER_GREEDY else c for c in candidates])
        ax.set_ylim(len(candidates) - .5, -.5)
        ax.grid(axis="y", visible=False)
        ax.set_title(name)
    handles = [mpl.lines.Line2D([], [], color=PAPER["greedy"], marker="o", lw=2.2,
                                label="Mean difference to 99% one-sided lower bound (prespecified)"),
               mpl.lines.Line2D([], [], color=PAPER["ink"], marker="|", ls="none", ms=10, mew=1.6,
                                label="97.5% bound (sensitivity)"),
               mpl.lines.Line2D([], [], color=PAPER["ink"], ls="--", lw=.8, label="−margin")]
    fig.legend(handles=handles, loc="outside upper center", ncol=3)
    fig.supxlabel("Paired score difference: candidate − Vanilla rough fix", fontsize=8)
    return fig
