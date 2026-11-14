"""Generate benchmarks/analysis/publication/paper_results.ipynb (cells only, no outputs)."""
from pathlib import Path
import nbformat as nbf

TARGET = Path(__file__).with_name("paper_results.ipynb")
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Paper results: Experimental Evaluation

Figures, tables and key numbers for Section 4 and the appendices:

| Section | Question | Theory | Main figures |
|---|---|---|---|
| 4.2 Convergence and cost | Does Vanilla finish? Where does its runtime go, and what speedup does stopping at K give? | Theorem 2, parallel execution | runtime anatomy, speedup |
| 4.3 Ground truth | What does the single-hit rule cost in recall and false positives? | Theorems 1 and 3, weak/strong | FP vs bound, hit counts |
| 4.4 Real data | Which features does Greedy add, and does a downstream model notice? | Theorem 1, weak/strong | fates, NI, α strip |
| Appendix B | Reduced shadows | K/(r+1) bound | FP vs 1/(r+1) |

The source is the augmented copy `<run>_aug` (`runner/augment_coupled.py`) when it
exists, otherwise the original run (coupled and history sections are skipped).
Diagnostics stay in `publication_analysis.ipynb`. Nothing here fits models.

**Greedy** means the prespecified full-shadow Greedy run (`j=1`); its masks are identical
to the parallel run (checked below), which matters only for timing. **Greedy coupled** is
Greedy evaluated on the first K iterations of the very Vanilla fit it is compared with.
""")

code("""
from pathlib import Path
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents]
             if (p / "benchmarks" / "runner" / "experiment_run.py").is_file()), None)
if ROOT is None:
    raise FileNotFoundError("Launch Jupyter from within this repository")
ANALYSIS_DIR = ROOT / "benchmarks" / "analysis" / "publication"
sys.path.insert(0, str(ANALYSIS_DIR))
import analysis_helpers as ah

RUNS = ROOT / "benchmarks" / "data" / "publication_runs"
RUN_DIR = Path(os.environ.get("PUBLICATION_RUN_DIR") or
               next(p for p in [RUNS / f"{ah.RUN_NAME}_aug", RUNS / ah.RUN_NAME] if p.exists()))
FOCUS_ALPHA = 0.001
OUTPUT_DIR = ANALYSIS_DIR / "outputs" / RUN_DIR.name / "paper"
(OUTPUT_DIR / "tables").mkdir(parents=True, exist_ok=True)
ah.paper_style()
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 200)
pd.set_option("display.precision", 4)
FACTS = {}

def show(fig, name):
    ah.save_figure(fig, OUTPUT_DIR, name)
    display(fig)
    plt.close(fig)

def save_latex(table, name, caption, label, **kwargs):
    text = ah.latex_table(table, caption, label, **kwargs)
    (OUTPUT_DIR / "tables" / f"{name}.tex").write_text(text, encoding="utf-8")
    return text
""")

code("""
run = ah.load_run(RUN_DIR)
config = run.manifest["config"]
complete = ah.complete_blocks(run)
summary = ah.summary_table(complete)
paired = ah.paired_comparisons(complete)
ni = ah.noninferiority(paired, run)
histories, trajectories = ah.load_histories(RUN_DIR)
HAS_COUPLED = complete.Method.eq(ah.COUPLED).any()
HAS_HISTORY = not histories.empty
sets = ah.coupled_sets(complete, run.manifest) if HAS_COUPLED else pd.DataFrame()
datasets = config["datasets"]
p_of = complete.groupby("Dataset").N_Features.first()
# Real datasets ordered by dimensionality throughout; XOR only in the appendix.
REAL = sorted([n for n, (kind, _) in datasets.items() if kind == "real"], key=lambda n: p_of[n])
SYNTHETIC = [n for n, (kind, spec) in datasets.items() if kind == "synthetic" and spec["kind"] != "xor"]
PARALLEL = ah.greedy_parallel_label(complete)
PARALLEL_Q = ah.parallel_quantity(complete)
display(Markdown(
    f"**Run:** `{RUN_DIR.name}` | prespecified design complete: **{run.complete}** | "
    f"coupled Greedy: **{HAS_COUPLED}** | Vanilla histories: **{len(histories)}** refits  \\n"
    f"Real datasets (by p): {REAL}; synthetic: {SYNTHETIC}."))
""")

md("""
**Coupled identities, re-checked on every logged row** (independently of the augmentation
log): Vanilla's strict and rough-fix sets inside coupled Greedy's set (Theorem 1), strong
confirmations inside Vanilla's strict set, and K equal to the logged Greedy budget.
""")

code("""
if HAS_COUPLED:
    checks = ah.verify_coupled(complete)
    display(checks.filter(regex="Theorem|Strong|K_").agg(["sum", "count"]))
    FACTS["coupled_identity_checks"] = {c: f"{int(checks[c].sum())}/{len(checks)}"
                                        for c in checks.filter(regex="Theorem|Strong|K_").columns}
""")

md("""
## 4.1 Setup table
""")

code("""
sizes = complete.groupby("Dataset").agg(n=("N_Train", "first"), n_test=("N_Test", "first"),
                                        p=("N_Features", "first"), Task=("Task", "first"))
rows = []
for name in REAL + [n for n, (kind, _) in datasets.items() if kind == "synthetic"]:
    kind, spec = datasets[name]
    s = sizes.loc[name]
    row = {"Dataset": name, "Suite": kind, "Task": s.Task, "n": int(s.n + s.n_test), "p": int(s.p),
           "Relevant": len(ah.relevant_features(spec)) if kind == "synthetic" else None}
    row.update({f"K (α={a:g})": int(np.ceil(np.log2(s.p / a))) for a in sorted(config["alphas"], reverse=True)})
    rows.append(row)
setup = pd.DataFrame(rows)
display(setup)
latex = setup.assign(Relevant=setup.Relevant.map(lambda v: "--" if pd.isna(v) else f"{int(v)}"),
                     n=setup.n.map("{:,}".format), p=setup.p.map("{:,}".format)).drop(columns="Suite")
save_latex(latex, "setup_datasets", "Datasets, relevant features and Greedy iteration budget $K$.", "tab:datasets", group_rules=False);
""")

md("""
## 4.2 Convergence and computational cost

**Does Vanilla finish?** Every fold: whether it reached the iteration cap, and how many
tentative features the rough fix had to promote.
""")

code("""
convergence = ah.vanilla_convergence(complete, run.manifest)
display(convergence[["Dataset", "Alpha", "K", "Median_Iterations", "Fraction_At_Cap",
                     "Fraction_Rough_Fix_Used", "Mean_Rough_Fix_Promoted", "Max_Rough_Fix_Promoted",
                     "Median_Iteration_Ratio"]])
real_conv = convergence[convergence.Dataset.isin(REAL)]
FACTS["vanilla_cap_fraction_by_dataset_alpha"] = real_conv.set_index(["Dataset", "Alpha"]).Fraction_At_Cap.round(3).to_dict()
FACTS["datasets_always_at_cap"] = sorted(real_conv.groupby("Dataset").Fraction_At_Cap.min().loc[lambda s: s.eq(1)].index)
FACTS["rough_fix_promoted_max"] = int(convergence.Max_Rough_Fix_Promoted.max())
FACTS["rough_fix_used_fraction_real"] = real_conv.set_index(["Dataset", "Alpha"]).Fraction_Rough_Fix_Used.round(2).to_dict()
""")

md("""
**Where Vanilla's runtime goes** (full Vanilla refits with history, focus α, three folds
per dataset). Blue: features still tentative; gray: cumulative share of the run's own
wall-clock time; dashed: Greedy's budget K. At K, Vanilla has resolved most features but
spent only part of its runtime; the remaining tentatives absorb the rest.
""")

code("""
if HAS_HISTORY:
    show(ah.runtime_anatomy_plot(trajectories, FOCUS_ALPHA, REAL), "fig_runtime_anatomy")
    h = histories[histories.Alpha.eq(FOCUS_ALPHA)].copy()
    at_k = trajectories[trajectories.iteration.eq(trajectories.K)][["Partition_ID", "Alpha", "Tentative_Fraction"]]
    h = h.merge(at_k, on=["Partition_ID", "Alpha"])
    h["Runtime_Share_At_K"] = h.Time_At_K_Sec / h.Selection_Time_Sec
    anatomy = h.groupby("Dataset").agg(Refits=("Fold", "size"), Iterations=("Iterations", "median"),
        Stop=("Stop_Reason", lambda s: ", ".join(f"{k}:{v}" for k, v in s.value_counts().items())),
        Runtime_Share_At_K=("Runtime_Share_At_K", "median"), Tentative_At_K=("Tentative_Fraction", "median"),
        Final_Tentative=("Final_Tentative", "median"), Same_Run_Speedup=("Same_Run_Speedup", "median"),
        Session_Time_Ratio=("Session_Time_Ratio", "median"))
    display(anatomy.round(3))
    FACTS["runtime_share_at_K"] = anatomy.Runtime_Share_At_K.round(3).to_dict()
    FACTS["tentative_share_at_K"] = anatomy.Tentative_At_K.round(4).to_dict()
    FACTS["history_stop_reasons"] = anatomy.Stop.to_dict()
    FACTS["history_session_time_ratio_range"] = [round(histories.Session_Time_Ratio.min(), 3),
                                                 round(histories.Session_Time_Ratio.max(), 3)]
else:
    display(Markdown("_No Vanilla history refits yet: run `publication/augment.ps1`._"))
""")

md("""
**Speedup.** Per dataset at the focus α: the naive expectation (Vanilla iterations / K),
the speedup Vanilla itself would get if stopped at iteration K (same refit, so no
cross-session timing), and the observed speedups of Greedy. The cross-session variant
(logged Vanilla time / coupled prefix time) is kept in the table only: the augmentation
session ran up to ~2x faster than the original on some datasets, which biases it.
""")

code("""
decomposition = ah.speedup_decomposition(paired, complete, histories)
focus = decomposition[decomposition.Alpha.eq(FOCUS_ALPHA)].pivot_table(index="Dataset", columns="Quantity", values="Median")
display(focus.round(2))
show(ah.speedup_plot(decomposition, FOCUS_ALPHA, REAL, histories, PARALLEL_Q), "fig_speedup")
real_dec = decomposition[decomposition.Dataset.isin(REAL)]
for quantity in ["Greedy serial (j=1)", PARALLEL_Q, "Same run stopped at K (history)", "Iteration ratio"]:
    values = real_dec[real_dec.Quantity.eq(quantity)].Median
    if len(values):
        FACTS[f"real_median_speedup_range::{quantity}"] = [round(values.min(), 1), round(values.max(), 1)]
if HAS_HISTORY:
    compare = focus.loc[REAL, ["Same run stopped at K (history)", "Greedy serial (j=1)"]]
    FACTS["serial_vs_stopped_at_K_ratio"] = (compare.iloc[:, 1] / compare.iloc[:, 0]).round(2).to_dict()
""")

md("""
**Compute, not only wall clock.** CPU-seconds and sampled peak RSS increase at the focus α.
""")

code("""
cost = complete[complete.Alpha.eq(FOCUS_ALPHA) & complete.Label.isin(["Vanilla rough fix", ah.PAPER_GREEDY, PARALLEL])]
cost = cost.groupby(["Dataset", "Label"]).agg(Time=("Selection_Time_Sec", "median"), CPU=("Selection_CPU_Sec", "median"),
                                              RSS=("RSS_Peak_Increase_MB", "median")).unstack("Label")
cpu_ratio = (cost["CPU"]["Vanilla rough fix"] / cost["CPU"][[ah.PAPER_GREEDY, PARALLEL]].T).T
display(cost.loc[REAL + SYNTHETIC].round(1))
display(cpu_ratio.loc[REAL].rename(columns=lambda c: f"Vanilla CPU / {c}").round(2))
parallel = ah.parallel_ablation(complete)
FACTS["parallel_identical_mask_fraction"] = float(parallel.Identical_Mask.mean())
FACTS["parallel_pairs"] = int(len(parallel))
FACTS["cpu_ratio_parallel_real"] = cpu_ratio.loc[REAL, PARALLEL].round(2).to_dict()
FACTS["cpu_ratio_serial_real"] = cpu_ratio.loc[REAL, ah.PAPER_GREEDY].round(2).to_dict()
FACTS["rss_increase_mb_real"] = cost["RSS"].loc[REAL].round(0).to_dict()
""")

md("""
## 4.3 Selection quality against ground truth

Mean false positives per fold versus α with the Theorem 3 bound E[FP] <= m0 K/(p+1)
(whiskers: IQR over folds), then the hit counts of Greedy's selections (coupled run).
""")

code("""
show(ah.fp_bound_plot(complete, run.manifest, SYNTHETIC), "fig_fp_vs_bound")
gt = summary[summary.Dataset.isin(SYNTHETIC) & summary.Label.isin(["Vanilla strict", "Vanilla rough fix", ah.PAPER_GREEDY])]
gt = gt[["Dataset", "Alpha", "Label", "Recall_mean", "Precision_mean", "FP_mean", "FP_median", "FPR_mean"]].copy()
gt["K"] = [int(np.ceil(np.log2(datasets[d][1]["n_features"] / a))) for d, a in zip(gt.Dataset, gt.Alpha)]
gt["Bound_E_FP"] = [(datasets[d][1]["n_features"] - len(ah.relevant_features(datasets[d][1]))) * k
                    / (datasets[d][1]["n_features"] + 1) for d, k in zip(gt.Dataset, gt.K)]
gt["FP_over_bound"] = gt.FP_mean / gt.Bound_E_FP
display(gt)
greedy_gt = gt[gt.Label.eq(ah.PAPER_GREEDY)]
FACTS["greedy_fp_over_bound_range"] = [round(greedy_gt.FP_over_bound.min(), 2), round(greedy_gt.FP_over_bound.max(), 2)]
FACTS["null_fp_mean"] = gt[gt.Dataset.eq("kind_null")].pivot_table(index="Alpha", columns="Label", values="FP_mean").round(2).to_dict()
FACTS["fp_mean_focus"] = gt[gt.Alpha.eq(FOCUS_ALPHA)].pivot_table(index="Dataset", columns="Label", values="FP_mean").round(2).to_dict()
FACTS["recall_focus"] = gt[gt.Alpha.eq(FOCUS_ALPHA)].pivot_table(index="Dataset", columns="Label", values="Recall_mean").round(3).to_dict()
FACTS["fpr_by_p_greedy"] = greedy_gt[greedy_gt.Alpha.eq(FOCUS_ALPHA)].set_index("Dataset").FPR_mean.round(4).to_dict()
""")

code("""
if HAS_COUPLED:
    show(ah.hits_plot(complete, run.manifest, FOCUS_ALPHA, SYNTHETIC), "fig_hits")
    synth_sets = sets[sets.Dataset.isin(SYNTHETIC)]
    weak_strong = synth_sets.groupby(["Dataset", "Alpha"]).agg(
        Greedy=("N_Greedy", "mean"), Strong=("N_Strong", "mean"), Weak=("N_Weak", "mean"),
        FP=("FP_Greedy", "mean"), FP_Strong=("FP_Strong", "mean"), FP_Weak=("FP_Weak", "mean")).reset_index()
    weak_strong["Share_of_FP_in_weak"] = weak_strong.FP_Weak / weak_strong.FP
    display(weak_strong)
    hits = []
    for row in complete[complete.Method.eq(ah.COUPLED) & complete.Dataset.isin(SYNTHETIC) & complete.Alpha.eq(FOCUS_ALPHA)].itertuples():
        truth = ah.relevant_features(datasets[row.Dataset][1])
        hits += [dict(Dataset=row.Dataset, Relevant=j in truth, Hits=h, K=row.Iterations)
                 for j, h in zip(row.Selected_Features, row.Selected_Hits)]
    hits = pd.DataFrame(hits)
    display(hits.groupby(["Dataset", "Relevant"]).Hits.describe().round(1))
    FACTS["share_fp_weak_range"] = [round(weak_strong.Share_of_FP_in_weak.min(), 3), round(weak_strong.Share_of_FP_in_weak.max(), 3)]
    FACTS["fp_hits_share_1_or_2"] = round(float(hits[~hits.Relevant].Hits.le(2).mean()), 3)
    FACTS["fp_hits_median"] = float(hits[~hits.Relevant].Hits.median())
    FACTS["tp_hits_over_K_median_by_dataset"] = (hits[hits.Relevant].assign(r=lambda d: d.Hits / d.K)
                                                  .groupby("Dataset").r.median().round(2).to_dict())
else:
    display(Markdown("_Coupled Greedy not available yet._"))
""")

md("""
## 4.4 Real-world benchmarks

**Which features does Greedy add?** Each coupled Greedy selection splits into nested
layers (Strong ⊆ Vanilla strict ⊆ Vanilla rough fix ⊆ Greedy, verified exactly).
""")

code("""
if HAS_COUPLED:
    show(ah.fates_plot(sets, FOCUS_ALPHA, {"Real-world": REAL, "Synthetic": SYNTHETIC}), "fig_greedy_fates")
    fates = sets.groupby(["Dataset", "Alpha"])[ah.FATE_LAYERS + ["N_Greedy", "N_Vanilla_Rough_Fix", "N_Strong", "N_Weak"]].mean()
    display(fates.round(1))
    focus_fates = fates.xs(FOCUS_ALPHA, level="Alpha")
    FACTS["greedy_only_share_focus"] = (focus_fates["Greedy only"] / focus_fates.N_Greedy).round(3).to_dict()
    FACTS["vanilla_share_of_greedy_focus"] = (focus_fates.N_Vanilla_Rough_Fix / focus_fates.N_Greedy).round(3).to_dict()
    FACTS["strong_share_focus"] = (focus_fates.N_Strong / focus_fates.N_Greedy).round(3).to_dict()
    if HAS_HISTORY:
        weak = ah.history_fates(RUN_DIR, complete)
        weak = weak[weak.Alpha.eq(FOCUS_ALPHA)].groupby("Dataset").sum(numeric_only=True)
        shares = weak[["Weak_Confirmed", "Weak_Tentative_Promoted", "Weak_Tentative_Dropped", "Weak_Rejected"]].div(weak.N_Weak, axis=0)
        display(Markdown("Final Vanilla decision for Greedy's **weak** features (history refits, shares):"))
        display(shares.round(3))
        FACTS["weak_final_decision_shares"] = shares.round(3).to_dict()
        median_iter = ah.history_fates(RUN_DIR, complete).groupby("Dataset").Weak_Median_Decision_Iteration.median()
        FACTS["weak_median_decision_iteration"] = median_iter.round(0).to_dict()
else:
    display(Markdown("_Coupled Greedy not available yet._"))
""")

md("""
**Independent versus coupled.** Same algorithm, different random streams: sizes and scores
should agree in distribution, and sharing the stream should shrink the spread of the paired
differences (descriptive only; the confirmatory test stays prespecified).
""")

code("""
if HAS_COUPLED:
    agree = summary[summary.Label.isin([ah.PAPER_GREEDY, "Greedy coupled"]) & summary.Alpha.eq(FOCUS_ALPHA)].pivot_table(
        index="Dataset", columns="Label", values=["N_Selected_mean", "Score_mean"])
    display(agree.loc[REAL + SYNTHETIC].round(3))
    spread = paired[paired.Label.isin([ah.PAPER_GREEDY, "Greedy coupled"]) & paired.Alpha.eq(FOCUS_ALPHA)].groupby(
        ["Dataset", "Label"]).Score_Difference.std().unstack()
    spread["Reduction"] = 1 - spread["Greedy coupled"] / spread[ah.PAPER_GREEDY]
    display(spread.loc[REAL].round(4))
    FACTS["coupled_sd_reduction_real"] = spread.loc[REAL, "Reduction"].round(2).to_dict()
""")

md("""
**Prespecified non-inferiority** (margin 0.01, one-sided level 0.01, Holm over the full
planned family) with the **0.025 sensitivity bound** (black tick). *All features* (no
selection) calibrates the test: where it also passes, the test cannot discriminate selectors.
""")

code("""
show(ah.ni_forest_plot(ni, FOCUS_ALPHA, REAL), "fig_noninferiority")
cols = ["Dataset", "Alpha", "Label", "Mean_Difference", "Corrected_SE", "Lower_One_Sided_CI",
        "Lower_Bound_97.5", "P_NI_Holm", "Decision", "Decision_Level_0.025"]
display(ni[ni.Dataset.isin(REAL) & ni.Label.isin([ah.PAPER_GREEDY, "All features"])][cols])
greedy_ni = ni[ni.Dataset.isin(REAL) & ni.Label.eq(ah.PAPER_GREEDY)]
established = lambda col: sorted(set(greedy_ni.loc[greedy_ni[col].str.startswith("Non-inferiority established"), "Dataset"]))
FACTS["ni_established_greedy_real"] = {"level_0.01": established("Decision"), "level_0.025": established("Decision_Level_0.025"),
                                       "count_0.01": int(greedy_ni.Decision.str.startswith("Non-inferiority established").sum()),
                                       "of": int(len(greedy_ni))}
all_ni = ni[ni.Dataset.isin(REAL) & ni.Label.eq("All features")]
FACTS["ni_established_all_features_real"] = sorted(set(all_ni.loc[all_ni.Decision.str.startswith("Non-inferiority established"), "Dataset"]))
FACTS["greedy_mean_difference_within_margin"] = int(greedy_ni.Mean_Difference.ge(-greedy_ni.Margin).sum())
FACTS["greedy_mean_difference_focus"] = greedy_ni[greedy_ni.Alpha.eq(FOCUS_ALPHA)].set_index("Dataset").Mean_Difference.round(4).to_dict()
""")

md("""
**Main table** (focus α in the main text; all α for the appendix).
""")

code("""
alphas = sorted(config["alphas"], reverse=True)
main_table = ah.real_results_table(summary, ni, decomposition, [FOCUS_ALPHA], REAL, PARALLEL_Q)
full_table = ah.real_results_table(summary, ni, decomposition, alphas, REAL, PARALLEL_Q)
display(main_table)
notes = (r"$\\bar D$: mean paired difference Greedy $-$ Vanilla rough fix; LB: one-sided lower bounds "
         r"(prespecified 99\\%, sensitivity 97.5\\%); NI: Holm-adjusted non-inferiority at margin 0.01, level 0.01.")
save_latex(main_table.drop(columns=r"$\\alpha$"), "real_results_focus",
           f"Real-world benchmarks at $\\\\alpha={FOCUS_ALPHA:g}$.", "tab:realworld_results", notes=notes, group_rules=False)
save_latex(full_table, "real_results_all_alpha", "Real-world benchmarks at every $\\\\alpha$.",
           "tab:realworld_results_all", notes=notes);
""")

md("""
**Sensitivity to α.** One marker per α on each dataset row: overlapping markers mean α
changes little beyond the iteration budget. The shaded band is the NI margin.
""")

code("""
show(ah.alpha_strip_plot(paired, complete, run.manifest, REAL), "fig_alpha_sensitivity")
""")

md("""
## Appendix B: reduced shadows
""")

code("""
shadow_scenarios = [n for n in ["kind_null", "reference"] if n in datasets]
show(ah.shadow_fp_plot(complete, run.manifest, shadow_scenarios), "fig_shadow_fp")
reduced = complete[complete.Method.str.match(r"Greedy_jobs\\d+_shadows") & complete.Outer_Jobs.eq(complete.Outer_Jobs.max())]
shadow = reduced.groupby(["Dataset", "Alpha", "Label"]).agg(
    Shadows=("N_Shadows", "first"), Time=("Selection_Time_Sec", "median"),
    RSS=("RSS_Peak_Increase_MB", "median"), Selected=("N_Selected", "mean"), Score=("Score", "mean"),
    FP=("FP", "mean"), Recall=("Recall", "mean")).reset_index()
display(shadow[shadow.Alpha.eq(FOCUS_ALPHA)])
shadow_diff = paired[paired.Method.isin(reduced.Method.unique())].groupby(
    ["Dataset", "Alpha", "Label"]).Score_Difference.mean().unstack("Label")
display(shadow_diff.xs(FOCUS_ALPHA, level="Alpha").round(4))
""")

md("""
## Appendix C: synthetic runtime anatomy and speedup, XOR, stability
""")

code("""
if HAS_HISTORY:
    show(ah.runtime_anatomy_plot(trajectories, FOCUS_ALPHA, SYNTHETIC, ncols=4), "fig_runtime_anatomy_synthetic")
show(ah.speedup_plot(decomposition, FOCUS_ALPHA, SYNTHETIC, histories, PARALLEL_Q), "fig_speedup_synthetic")
xor = [n for n, (kind, spec) in datasets.items() if kind == "synthetic" and spec["kind"] == "xor"]
display(summary[summary.Dataset.isin(xor)][["Dataset", "Alpha", "Label", "Recall_mean", "FP_mean", "Score_mean"]])
stability, _ = ah.stability_tables(complete, run.manifest)
display(stability[stability.Alpha.eq(FOCUS_ALPHA) & stability.Label.isin(
    ["Vanilla rough fix", ah.PAPER_GREEDY, "Greedy coupled"])].pivot_table(
    index="Dataset", columns="Label", values="Mean_Pairwise_Jaccard").round(3))
""")

md("## Key numbers and exports")

code("""
def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value.item() if hasattr(value, "item") else value

(OUTPUT_DIR / "key_numbers.json").write_text(json.dumps(plain(FACTS), indent=2), encoding="utf-8")
for name, table in {"convergence": convergence, "speedup_decomposition": decomposition,
                    "ground_truth": gt, "noninferiority": ni, "coupled_sets": sets,
                    "vanilla_histories": histories}.items():
    if isinstance(table, pd.DataFrame) and not table.empty:
        table.to_csv(OUTPUT_DIR / "tables" / f"{name}.csv", index=False)
print(json.dumps(plain(FACTS), indent=1)[:8000])
print(f"Exported to {OUTPUT_DIR.relative_to(ROOT)}")
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3",
                                                               "language": "python"}})
nbf.write(nb, TARGET)
print("wrote", TARGET)
