"""
analyze_results_functions.py
============================
Reusable analysis and plotting functions for the benchmark results analysis,
covering both final-state comparisons and checkpoint convergence analysis
of Vanilla Boruta vs Greedy/GreedyParallel Boruta.

All functions operate on DataFrames and configuration values passed in
explicitly, making them easy to test in isolation and straightforward to reuse
across notebooks or scripts.

Sections
--------
1.  Data loading & preparation
2.  Data reshaping (explode Vanilla strict/soft variants)
3.  Score comparison boxplots
4.  Runtime distribution plots
5.  Pareto efficiency scatter
6.  Speedup bar chart
7.  Statistical tests (score)
8.  Checkpoint convergence vs time
9.  Checkpoint convergence vs iteration
10. Undecidedness factor
11. Potential miss rate
12. Checkpoint statistical equivalence tests
13. Minimum iterations for equivalence
14. Summary statistics
15. Equivalence visualization
16. Key findings printout
17. Export
"""

from __future__ import annotations

import warnings
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import mannwhitneyu, ttest_ind

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _save_fig(fig: plt.Figure, name: str, figures_dir: "str | None") -> None:
    """Save *fig* as a high-DPI PNG to *figures_dir* if it is not None."""
    if figures_dir is None:
        return
    import os
    os.makedirs(figures_dir, exist_ok=True)
    path = os.path.join(figures_dir, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")


# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------
# Central colour tokens — all plot functions reference these two
_GREEDY_COLOR  = "#e47e1e"
_VANILLA_COLOR = "#6C6D6D"

PAL_METHOD_VARIANT: dict[str, str] = {
    "Greedy":             _GREEDY_COLOR,
    "GreedyParallel":     _GREEDY_COLOR,
    "Vanilla (Strict)":   _VANILLA_COLOR,
    "Vanilla (Soft/All)": "#f4a261",   # lighter sibling, kept for completeness
}
PAL_METHOD: dict[str, str] = {
    "Greedy":         _GREEDY_COLOR,
    "GreedyParallel": _GREEDY_COLOR,
    "Vanilla":        _VANILLA_COLOR,
}
# Global figure scale — multiply all base dimensions by this value.
# Increase (e.g. 1.5 or 2.0) if figures look too small in the exported PDF.
FIG_SCALE: float = 0.8

# ['#e2514a', '#fca55d', '#fee999', '#edf8a3', '#a2d9a4', '#47a0b3']
# Dataset palette — 6 visually distinct, non-seaborn colours (same across all plots)
PAL_DATASET: dict[str, str] = {
    "Arcene":    "#e2514a",
    "Dexter":    "#fca55d",
    "Gisette":   "#fee999",
    "Isolet":    "#edf8a3",
    "Madelon":   "#a2d9a4",
    "riboflavin":"#47a0b3",
}


# ---------------------------------------------------------------------------
# 0. Plot theme
# ---------------------------------------------------------------------------

def apply_plot_theme(font_scale: float = 1.2, dpi: int = 110) -> None:
    """Apply the standard seaborn whitegrid paper theme."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=font_scale)
    plt.rcParams["figure.dpi"] = dpi


# ---------------------------------------------------------------------------
# 1. Data loading & preparation
# ---------------------------------------------------------------------------

def load_data(
    main_file: str,
    checkpoint_file: Optional[str] = None,
    estimator: str = "extra_trees",
    max_time_sec: float = 1000.0,
) -> tuple:
    """Load and filter the main benchmark results and optional checkpoint CSV.

    Parameters
    ----------
    main_file:
        Path to the benchmark_results CSV.
    checkpoint_file:
        Path to the checkpoint_history_vanilla CSV. Pass None to skip.
    estimator:
        Estimator name to filter on (e.g. ``'extra_trees'``).
    max_time_sec:
        Drop rows where Time_Sec exceeds this threshold.

    Returns
    -------
    df_main:
        Filtered main results DataFrame.
    df_checkpoints:
        Filtered checkpoint DataFrame, or None if *checkpoint_file* is None.
    """
    df_main = pd.read_csv(main_file)
    df_main = df_main[df_main["Time_Sec"] < max_time_sec]
    df_main = df_main[df_main["Estimator"] == estimator].copy()

    print(f"Main results ({estimator}): {len(df_main)} rows")
    print(f"Datasets : {sorted(df_main['Dataset'].unique())}")
    print(f"Methods  : {sorted(df_main['Method'].unique())}")
    print(f"Alphas   : {sorted(df_main['Alpha'].unique())}")

    df_checkpoints = None
    if checkpoint_file is not None:
        df_checkpoints = pd.read_csv(checkpoint_file)
        df_checkpoints = df_checkpoints[df_checkpoints["Estimator"] == estimator].copy()
        print(f"\nCheckpoints ({estimator}): {len(df_checkpoints)} rows")

    return df_main, df_checkpoints


def prepare_greedy(df_main: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Extract the Greedy (preferring GreedyParallel) subset from *df_main*.

    Returns
    -------
    df_greedy:
        Rows for the chosen Greedy variant.
    greedy_label:
        ``'GreedyParallel'`` or ``'Greedy'``.
    """
    df_greedy = df_main[df_main["Method"].str.contains("Greedy", case=False)].copy()
    if "GreedyParallel" in df_greedy["Method"].values:
        df_greedy = df_greedy[df_greedy["Method"] == "GreedyParallel"].copy()
        greedy_label = "GreedyParallel"
    else:
        df_greedy = df_greedy[df_greedy["Method"] == "Greedy"].copy()
        greedy_label = "Greedy"
    print(f"Using {greedy_label}: {len(df_greedy)} rows")
    return df_greedy, greedy_label


# ---------------------------------------------------------------------------
# 2. Data reshaping
# ---------------------------------------------------------------------------

def explode_variants(df_main: pd.DataFrame) -> pd.DataFrame:
    """Reshape *df_main* so each row represents one (Method_Variant, run).

    Vanilla rows are split into two variants:
    - ``'Vanilla (Strict)'``  — Confirmed features only
    - ``'Vanilla (Soft/All)'`` — Confirmed + Tentative features

    Greedy rows keep their original method name as Method_Variant.

    Parameters
    ----------
    df_main:
        Filtered main results DataFrame from :func:`load_data`.

    Returns
    -------
    df_viz:
        Reshaped DataFrame with columns:
        Dataset, Task, Alpha, Method_Variant, Score, Time_Sec, Selected_Feats.
    """
    rows = []
    for _, row in df_main.iterrows():
        if "Greedy" in row["Method"]:
            rows.append({
                "Dataset": row["Dataset"],
                "Task": row["Task"],
                "Alpha": str(row["Alpha"]),
                "Method_Variant": row["Method"],
                "Score": row["Score_Confirmed_Mean"],
                "Time_Sec": row["Time_Sec"],
                "Selected_Feats": row["N_Confirmed"],
            })
        elif row["Method"] == "Vanilla":
            rows.append({
                "Dataset": row["Dataset"],
                "Task": row["Task"],
                "Alpha": str(row["Alpha"]),
                "Method_Variant": "Vanilla (Strict)",
                "Score": row["Score_Confirmed_Mean"],
                "Time_Sec": row["Time_Sec"],
                "Selected_Feats": row["N_Confirmed"],
            })
            rows.append({
                "Dataset": row["Dataset"],
                "Task": row["Task"],
                "Alpha": str(row["Alpha"]),
                "Method_Variant": "Vanilla (Soft/All)",
                "Score": row["Score_All_Mean"],
                "Time_Sec": row["Time_Sec"],
                "Selected_Feats": row["N_Confirmed"] + row["N_Tentative"],
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Score comparison boxplots
# ---------------------------------------------------------------------------

def plot_score_comparison(
    df_viz: pd.DataFrame,
    estimator: str,
    palette: Optional[dict] = None,
    figures_dir: Optional[str] = None,
) -> None:
    """Faceted strip + point plot of Score by Alpha and Dataset.

    Parameters
    ----------
    df_viz:
        Reshaped DataFrame from :func:`explode_variants`.
    estimator:
        Estimator name used in the plot title.
    palette:
        Color mapping for Method_Variant. Defaults to :data:`PAL_METHOD_VARIANT`.
    figures_dir:
        If not None, save figure here as a PNG.
    """
    if palette is None:
        palette = PAL_METHOD_VARIANT

    df_plot = df_viz[df_viz["Method_Variant"] != "Vanilla (Soft/All)"].copy()
    df_plot["Alpha"] = df_plot["Alpha"].astype(str)

    datasets = sorted(df_plot["Dataset"].unique())
    alpha_order = sorted(df_plot["Alpha"].unique(), key=float)

    available_methods = df_plot["Method_Variant"].unique()
    method_order = [m for m in ["GreedyParallel", "Greedy", "Vanilla (Strict)"] if m in available_methods]
    pal_dict = {m: palette[m] for m in method_order if m in palette}

    n_cols = 2
    n_rows = (len(datasets) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(7 * n_cols * FIG_SCALE, 4 * n_rows * FIG_SCALE),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for i, dataset in enumerate(datasets):
        ax = axes_flat[i]
        df_ds = df_plot[df_plot["Dataset"] == dataset]

        sns.stripplot(
            data=df_ds, x="Alpha", y="Score", hue="Method_Variant",
            hue_order=method_order,
            palette=pal_dict,
            order=alpha_order, ax=ax,
            dodge=True, jitter=0.15,
            size=6, alpha=0.45, linewidth=0.4, edgecolor="white",
            zorder=3, legend=False,
        )
        sns.pointplot(
            data=df_ds, x="Alpha", y="Score", hue="Method_Variant",
            hue_order=method_order,
            palette=pal_dict,
            order=alpha_order, ax=ax,
            dodge=0.4, markers=["o", "P"],
            capsize=0.1, err_kws={"linewidth": 1.5},
            linestyles="none", markersize=7, zorder=5,
        )

        ax.set_title(dataset, fontsize=12, fontweight="bold")
        # ax.set_xlabel("Alpha (Significance Level)", fontsize=10)
        ax.set_xlabel(r"$\alpha$", fontsize=10)
        ax.set_ylabel("Score (F1 Weighted or R2)", fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")
        # ax.legend(title="Method", fontsize=8, title_fontsize=8)
        ax.legend().set_visible(False)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    # fig.suptitle(f"Score Comparison: Greedy vs Vanilla", y=1.02)
    fig.tight_layout()
    _save_fig(fig, "score_comparison", figures_dir)
    plt.show()


# ---------------------------------------------------------------------------
# 4. Runtime distribution plots
# ---------------------------------------------------------------------------

def plot_runtime_distribution(
    df_main: pd.DataFrame,
    estimator: str,
    palette: Optional[dict] = None,
    figures_dir: Optional[str] = None,
) -> None:
    """Faceted point + strip plot of Time_Sec (log scale) by Alpha and Dataset.

    Parameters
    ----------
    df_main:
        Filtered main results DataFrame (not exploded).
    estimator:
        Estimator name used in the plot title.
    palette:
        Color mapping for Method. Defaults to :data:`PAL_METHOD`.
    figures_dir:
        If not None, save figure here as a PNG.
    """
    if palette is None:
        palette = PAL_METHOD

    df_plot = df_main.copy()
    df_plot["Alpha"] = df_plot["Alpha"].astype(str)

    g = sns.FacetGrid(df_plot, col="Dataset", col_wrap=2, height=4 * FIG_SCALE, aspect=1.5, sharey=False)
    g.map_dataframe(
        sns.pointplot,
        x="Alpha", y="Time_Sec", hue="Method",
        palette=palette,
        dodge=0.2, markers=["o", "s", "^"], capsize=0.1,
        err_kws={"linewidth": 1.5},
    )
    g.map_dataframe(
        sns.stripplot,
        x="Alpha", y="Time_Sec", hue="Method",
        palette=palette,
        dodge=0.1, alpha=0.6, jitter=0.15, size=5, legend=False,
    )
    for ax in g.axes.flat:
        ax.set_yscale("log")
        ax.set_ylabel("Time (s) - Log Scale")
        ax.grid(True, which="both", ls="--", linewidth=0.5)

    g.add_legend(title="Method")
    g.fig.suptitle(f"Runtime Distribution ({estimator})", y=1.02)
    _save_fig(g.fig, "runtime_distribution", figures_dir)
    plt.show()


# ---------------------------------------------------------------------------
# 5. Pareto efficiency scatter
# ---------------------------------------------------------------------------

def plot_pareto_efficiency(
    df_main: pd.DataFrame,
    estimator: str,
    figures_dir: Optional[str] = None,
) -> None:
    """Scatter of mean runtime vs mean score — Pareto frontier view.

    Parameters
    ----------
    df_main:
        Filtered main results DataFrame (not exploded).
    estimator:
        Estimator name used in the plot title.
    figures_dir:
        If not None, save figure here as a PNG.
    """
    df_avg = df_main.groupby(["Dataset", "Method", "Alpha"]).agg(
        Time_Sec=("Time_Sec", "mean"),
        Score_All_Mean=("Score_All_Mean", "mean"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(10 * FIG_SCALE, 6 * FIG_SCALE))
    sns.scatterplot(
        data=df_avg,
        x="Time_Sec",
        y="Score_All_Mean",
        hue="Dataset",
        style="Method",
        s=150,
        alpha=0.8,
        palette=PAL_DATASET,
        ax=ax,
    )
    ax.set_xscale("log")
    ax.set_title(f"Pareto Efficiency: Performance vs. Cost")
    ax.set_xlabel("Time (Seconds) - Log Scale")
    ax.set_ylabel("Score (F1 or R2)")
    ax.grid(True, which="both", ls="--", linewidth=0.5)
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    fig.tight_layout()
    _save_fig(fig, "pareto_efficiency", figures_dir)
    plt.show()


# ---------------------------------------------------------------------------
# 6. Speedup bar chart
# ---------------------------------------------------------------------------

def plot_speedup(
    df_main: pd.DataFrame,
    estimator: str,
    figures_dir: Optional[str] = None,
) -> None:
    """Bar chart of computational speedup (Vanilla / Greedy) by Alpha and Dataset.

    Parameters
    ----------
    df_main:
        Filtered main results DataFrame (not exploded).
    estimator:
        Estimator name used in the plot title.
    figures_dir:
        If not None, save figure here as a PNG.
    """
    df_pivot = df_main.groupby(["Dataset", "Alpha", "Method"])["Time_Sec"].mean().unstack()

    if "Vanilla" not in df_pivot.columns:
        print("Cannot plot Speedup: 'Vanilla' method missing from results.")
        return

    if "GreedyParallel" in df_pivot.columns:
        df_pivot["Speedup_Factor"] = df_pivot["Vanilla"] / df_pivot["GreedyParallel"]
        compare_label = "Vanilla vs GreedyParallel"
    else:
        df_pivot["Speedup_Factor"] = df_pivot["Vanilla"] / df_pivot["Greedy"]
        compare_label = "Vanilla vs Greedy"

    df_pivot = df_pivot.reset_index()
    df_pivot["Alpha"] = df_pivot["Alpha"].astype(str)

    fig, ax = plt.subplots(figsize=(10 * FIG_SCALE, 5 * FIG_SCALE))
    sns.barplot(
        data=df_pivot,
        x="Alpha",
        y="Speedup_Factor",
        hue="Dataset",
        palette=PAL_DATASET,
        ax=ax,
    )
    ax.set_title(f"Computational Advantage ({estimator}): {compare_label}")
    ax.set_ylabel("Speedup Factor (Higher is Better)")
    ax.axhline(1, color="r", linestyle="--", label="Parity (1x)")
    ax.legend()
    fig.tight_layout()
    _save_fig(fig, "speedup", figures_dir)
    plt.show()


# ---------------------------------------------------------------------------
# 7. Statistical tests (score)
# ---------------------------------------------------------------------------

def run_score_statistical_tests(
    df_viz: pd.DataFrame,
    significance_level: float = 0.1,
) -> pd.DataFrame:
    """Welch t-tests comparing Greedy vs Vanilla (Strict) scores.

    Parameters
    ----------
    df_viz:
        Reshaped DataFrame from :func:`explode_variants`.
    significance_level:
        Alpha threshold for declaring a winner.

    Returns
    -------
    DataFrame with columns: Dataset, Alpha, Greedy_Mean, Greedy_SE,
    Vanilla_Mean, Vanilla_SE, Diff (G-V), Greedy_Time, Vanilla_Time,
    Speedup, Winner_Stat, P_Any_Diff, P_Greedy_Better, P_Vanilla_Better.
    """
    results = []
    for dataset in df_viz["Dataset"].unique():
        for alpha in df_viz["Alpha"].unique():
            subset = df_viz[(df_viz["Dataset"] == dataset) & (df_viz["Alpha"] == alpha)]

            greedy_scores = subset[subset["Method_Variant"] == "GreedyParallel"]["Score"]
            if len(greedy_scores) == 0:
                greedy_scores = subset[subset["Method_Variant"] == "Greedy"]["Score"]
            vanilla_strict_scores = subset[subset["Method_Variant"] == "Vanilla (Strict)"]["Score"]

            if len(greedy_scores) < 2 or len(vanilla_strict_scores) < 2:
                continue

            _, p_diff = stats.ttest_ind(greedy_scores, vanilla_strict_scores,
                                        equal_var=False, alternative="two-sided")
            _, p_greedy_wins = stats.ttest_ind(greedy_scores, vanilla_strict_scores,
                                               equal_var=False, alternative="greater")
            _, p_vanilla_wins = stats.ttest_ind(greedy_scores, vanilla_strict_scores,
                                                equal_var=False, alternative="less")

            winner = "No difference detected"
            if p_greedy_wins < significance_level:
                winner = "Greedy"
            elif p_vanilla_wins < significance_level:
                winner = "Vanilla"

            greedy_time = subset[subset["Method_Variant"].str.contains("Greedy")]["Time_Sec"].mean()
            vanilla_time = subset[subset["Method_Variant"].str.contains("Vanilla")]["Time_Sec"].mean()
            speedup = vanilla_time / greedy_time if greedy_time > 0 else 0

            greedy_se = greedy_scores.std() / np.sqrt(len(greedy_scores))
            vanilla_se = vanilla_strict_scores.std() / np.sqrt(len(vanilla_strict_scores))

            results.append({
                "Dataset": dataset,
                "Alpha": alpha,
                "Greedy_Mean": greedy_scores.mean(),
                "Greedy_SE": greedy_se,
                "Vanilla_Mean": vanilla_strict_scores.mean(),
                "Vanilla_SE": vanilla_se,
                "Diff (G-V)": greedy_scores.mean() - vanilla_strict_scores.mean(),
                "Greedy_Time": greedy_time,
                "Vanilla_Time": vanilla_time,
                "Speedup": speedup,
                "Winner_Stat": winner,
                "P_Any_Diff": p_diff,
                "P_Greedy_Better": p_greedy_wins,
                "P_Vanilla_Better": p_vanilla_wins,
            })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# 8. Checkpoint convergence vs time
# ---------------------------------------------------------------------------

def plot_convergence_vs_time(
    dataset: str,
    estimator: str,
    df_checkpoints: pd.DataFrame,
    df_greedy: pd.DataFrame,
    alphas: list,
    greedy_label: str = "GreedyParallel",
    figures_dir: Optional[str] = None,
) -> None:
    """Plot Vanilla checkpoint score trajectories vs elapsed time for each alpha.

    Parameters
    ----------
    dataset:
        Dataset name to plot.
    estimator:
        Estimator name (used in title only).
    df_checkpoints:
        Checkpoint history DataFrame (filtered to target estimator).
    df_greedy:
        Greedy results DataFrame from :func:`prepare_greedy`.
    alphas:
        Ordered list of alpha values to plot (one subplot each).
    greedy_label:
        Label for the Greedy variant (used in legend).
    figures_dir:
        If not None, save figure here as a PNG.
    """
    checkpoints = df_checkpoints[df_checkpoints["Dataset"] == dataset].copy()
    greedy_data = df_greedy[df_greedy["Dataset"] == dataset].copy()

    if len(checkpoints) == 0 or len(greedy_data) == 0:
        print(f"No data for {dataset}")
        return

    task_type = checkpoints["Task"].iloc[0]
    score_label = "F1 Score" if task_type == "class" else "R\u00b2 Score"

    n_alphas = len(alphas)
    fig, axes = plt.subplots(1, n_alphas, figsize=(6 * n_alphas * FIG_SCALE, 5 * FIG_SCALE), squeeze=False)
    axes = axes.flatten()

    for idx, alpha in enumerate(alphas):
        ax = axes[idx]
        cp_alpha = checkpoints[checkpoints["Alpha"] == alpha]
        greedy_alpha = greedy_data[greedy_data["Alpha"] == alpha]

        if len(cp_alpha) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(f"Alpha = {alpha}")
            continue

        for run_id in cp_alpha["Run_ID"].unique():
            run_data = cp_alpha[cp_alpha["Run_ID"] == run_id].sort_values("Checkpoint_ID")
            ax.plot(
                run_data["Elapsed_Sec"], run_data["Score_Mean"],
                marker="o", alpha=0.6, linewidth=2, markersize=6,
                color=_VANILLA_COLOR,
                label=f"Vanilla Run {run_id}" if idx == 0 and run_id == 0 else "",
            )

        if len(greedy_alpha) > 0:
            greedy_mean = greedy_alpha["Score_Confirmed_Mean"].mean()
            greedy_time = greedy_alpha["Time_Sec"].mean()
            ax.axhline(y=greedy_mean, color=_GREEDY_COLOR, linestyle="--", linewidth=2.5,
                       label=f"{greedy_label} (mean)" if idx == 0 else "")
            ax.axvline(x=greedy_time, color=_GREEDY_COLOR, linestyle=":", alpha=0.5, linewidth=1.5)

        ax.set_xlabel("Time (seconds)", fontsize=11)
        ax.set_ylabel(score_label, fontsize=11)
        ax.set_title(f"Alpha = {alpha}", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(loc="best", fontsize=9)

    fig.suptitle(
        f"Convergence: Performance vs Time\n{dataset} ({estimator})",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    _save_fig(fig, f"convergence_time_{dataset}", figures_dir)
    plt.show()


# ---------------------------------------------------------------------------
# 9. Checkpoint convergence vs iteration
# ---------------------------------------------------------------------------

def plot_convergence_vs_iteration(
    dataset: str,
    estimator: str,
    df_checkpoints: pd.DataFrame,
    df_greedy: pd.DataFrame,
    alphas: list,
    greedy_label: str = "GreedyParallel",
    figures_dir: Optional[str] = None,
) -> None:
    """Plot Vanilla checkpoint score trajectories vs iteration number for each alpha.

    Parameters
    ----------
    dataset:
        Dataset name to plot.
    estimator:
        Estimator name (used in title only).
    df_checkpoints:
        Checkpoint history DataFrame (filtered to target estimator).
    df_greedy:
        Greedy results DataFrame from :func:`prepare_greedy`.
    alphas:
        Ordered list of alpha values to plot (one subplot each).
    greedy_label:
        Label for the Greedy variant (used in legend).
    figures_dir:
        If not None, save figure here as a PNG.
    """
    checkpoints = df_checkpoints[df_checkpoints["Dataset"] == dataset].copy()
    greedy_data = df_greedy[df_greedy["Dataset"] == dataset].copy()

    if len(checkpoints) == 0 or len(greedy_data) == 0:
        print(f"No data for {dataset}")
        return

    task_type = checkpoints["Task"].iloc[0]
    score_label = "F1 Score" if task_type == "class" else "R\u00b2 Score"

    n_alphas = len(alphas)
    fig, axes = plt.subplots(1, n_alphas, figsize=(6 * n_alphas * FIG_SCALE, 5 * FIG_SCALE), squeeze=False)
    axes = axes.flatten()

    for idx, alpha in enumerate(alphas):
        ax = axes[idx]
        cp_alpha = checkpoints[checkpoints["Alpha"] == alpha]
        greedy_alpha = greedy_data[greedy_data["Alpha"] == alpha]

        if len(cp_alpha) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(f"Alpha = {alpha}")
            continue

        for run_id in cp_alpha["Run_ID"].unique():
            run_data = cp_alpha[cp_alpha["Run_ID"] == run_id].sort_values("Checkpoint_ID")
            ax.plot(
                run_data["Iteration"], run_data["Score_Mean"],
                marker="o", alpha=0.6, linewidth=2, markersize=6,
                color=_VANILLA_COLOR,
                label=f"Vanilla Run {run_id}" if idx == 0 and run_id == 0 else "",
            )

        if len(greedy_alpha) > 0:
            greedy_mean = greedy_alpha["Score_Confirmed_Mean"].mean()
            ax.axhline(y=greedy_mean, color=_GREEDY_COLOR, linestyle="--", linewidth=2.5,
                       label=f"{greedy_label} (mean)" if idx == 0 else "")

        ax.set_xlabel("Iteration", fontsize=11)
        ax.set_ylabel(score_label, fontsize=11)
        ax.set_title(f"Alpha = {alpha}", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(loc="best", fontsize=9)

    fig.suptitle(
        f"Convergence: Performance vs Iteration\n{dataset} ({estimator})",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    _save_fig(fig, f"convergence_iteration_{dataset}", figures_dir)
    plt.show()


# ---------------------------------------------------------------------------
# 10. Undecidedness factor
# ---------------------------------------------------------------------------

def plot_undecidedness_factor(
    dataset: str,
    estimator: str,
    df_checkpoints: pd.DataFrame,
    alphas: list,
    x_axis: str = "iteration",
    figures_dir: Optional[str] = None,
) -> None:
    """Plot Tentative / all_features ratio over time or iterations.

    Parameters
    ----------
    dataset:
        Dataset name to plot.
    estimator:
        Estimator name (used in title only).
    df_checkpoints:
        Checkpoint history DataFrame (filtered to target estimator).
    alphas:
        Ordered list of alpha values to plot (one subplot each).
    x_axis:
        ``'time'`` for elapsed seconds, ``'iteration'`` for iteration count.
    figures_dir:
        If not None, save figure here as a PNG.
    """
    checkpoints = df_checkpoints[df_checkpoints["Dataset"] == dataset].copy()
    if len(checkpoints) == 0:
        print(f"No data for {dataset}")
        return

    all_features = (
        checkpoints["N_Tentative"] + checkpoints["N_Rejected"] + checkpoints["N_Confirmed"]
    )
    checkpoints["Undecidedness_Factor"] = np.where(
        all_features > 0,
        checkpoints["N_Tentative"] / all_features,
        np.nan,
    )

    x_col = "Elapsed_Sec" if x_axis == "time" else "Iteration"
    x_label = "Time (seconds)" if x_axis == "time" else "Iteration"
    x_type = "Time" if x_axis == "time" else "Iteration"

    n_alphas = len(alphas)
    fig, axes = plt.subplots(1, n_alphas, figsize=(6 * n_alphas * FIG_SCALE, 5 * FIG_SCALE), squeeze=False)
    axes = axes.flatten()

    for idx, alpha in enumerate(alphas):
        ax = axes[idx]
        cp_alpha = checkpoints[checkpoints["Alpha"] == alpha]
        if len(cp_alpha) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(f"Alpha = {alpha}")
            continue

        for run_id in cp_alpha["Run_ID"].unique():
            run_data = (
                cp_alpha[(cp_alpha["Run_ID"] == run_id) & (cp_alpha["Checkpoint_ID"] > 0)]
                .sort_values("Checkpoint_ID")
            )
            ax.plot(
                run_data[x_col], run_data["Undecidedness_Factor"],
                marker="o", alpha=0.6, linewidth=2, markersize=6,
                color=_VANILLA_COLOR,
                label=f"Run {run_id}" if idx == 0 and run_id < 3 else "",
            )

        ax.axhline(y=0, color=_GREEDY_COLOR, linestyle="--", linewidth=1.5, alpha=0.5,
                   label="All Decided" if idx == 0 else "")
        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel("Undecidedness Factor\n(Tentative / All Features)", fontsize=11)
        ax.set_title(f"Alpha = {alpha}", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=-0.01)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        if idx == 0:
            ax.legend(loc="best", fontsize=9)

    fig.suptitle(
        f"Undecidedness Factor vs {x_type}\n{dataset} ({estimator})",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    _save_fig(fig, f"undecidedness_{x_axis}_{dataset}", figures_dir)
    plt.show()


# ---------------------------------------------------------------------------
# 11. Potential miss rate
# ---------------------------------------------------------------------------

def plot_potential_miss_rate(
    dataset: str,
    estimator: str,
    df_checkpoints: pd.DataFrame,
    alphas: list,
    x_axis: str = "time",
    figures_dir: Optional[str] = None,
) -> None:
    """Plot Tentative / (Tentative + Confirmed) ratio over time or iterations.

    Parameters
    ----------
    dataset:
        Dataset name to plot.
    estimator:
        Estimator name (used in title only).
    df_checkpoints:
        Checkpoint history DataFrame (filtered to target estimator).
    alphas:
        Ordered list of alpha values to plot (one subplot each).
    x_axis:
        ``'time'`` for elapsed seconds, ``'iteration'`` for iteration count.
    figures_dir:
        If not None, save figure here as a PNG.
    """
    checkpoints = df_checkpoints[df_checkpoints["Dataset"] == dataset].copy()
    if len(checkpoints) == 0:
        print(f"No data for {dataset}")
        return

    denominator = checkpoints["N_Tentative"] + checkpoints["N_Confirmed"]
    checkpoints["Potential_Miss_Rate"] = np.where(
        denominator > 0,
        checkpoints["N_Tentative"] / denominator,
        np.nan,
    )

    x_col = "Elapsed_Sec" if x_axis == "time" else "Iteration"
    x_label = "Time (seconds)" if x_axis == "time" else "Iteration"
    x_type = "Time" if x_axis == "time" else "Iteration"

    n_alphas = len(alphas)
    fig, axes = plt.subplots(1, n_alphas, figsize=(6 * n_alphas * FIG_SCALE, 5 * FIG_SCALE), squeeze=False)
    axes = axes.flatten()

    for idx, alpha in enumerate(alphas):
        ax = axes[idx]
        cp_alpha = checkpoints[checkpoints["Alpha"] == alpha]
        if len(cp_alpha) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(f"Alpha = {alpha}")
            continue

        for run_id in cp_alpha["Run_ID"].unique():
            run_data = (
                cp_alpha[(cp_alpha["Run_ID"] == run_id) & (cp_alpha["Checkpoint_ID"] > 0)]
                .sort_values("Checkpoint_ID")
            )
            valid = run_data[~run_data["Potential_Miss_Rate"].isna()]
            if len(valid) > 0:
                ax.plot(
                    valid[x_col], valid["Potential_Miss_Rate"],
                    marker="o", alpha=0.6, linewidth=2, markersize=6,
                    color=_VANILLA_COLOR,
                    label=f"Run {run_id}" if idx == 0 and run_id < 3 else "",
                )

        ax.axhline(y=0, color=_GREEDY_COLOR, linestyle="--", linewidth=1.5, alpha=0.5,
                   label="All Confirmed" if idx == 0 else "")
        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel("Potential Miss Rate\n(Tentative / (Tentative + Confirmed))", fontsize=11)
        ax.set_title(f"Alpha = {alpha}", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=-0.01)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        if idx == 0:
            ax.legend(loc="best", fontsize=9)

    fig.suptitle(
        f"Potential Miss Rate vs {x_type}\n{dataset} ({estimator})",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    _save_fig(fig, f"potential_miss_rate_{x_axis}_{dataset}", figures_dir)
    plt.show()


# ---------------------------------------------------------------------------
# 11b. Combined undecidedness + potential-miss-rate for a single alpha
# ---------------------------------------------------------------------------

def plot_undecidedness_single_alpha(
    dataset: str,
    estimator: str,
    df_checkpoints: pd.DataFrame,
    alpha: float,
    x_axis: str = "iteration",
    figures_dir: Optional[str] = None,
) -> None:
    """Side-by-side undecidedness factor and potential miss rate for one alpha.

    Parameters
    ----------
    dataset:
        Dataset name to plot.
    estimator:
        Estimator name (used in the figure title).
    df_checkpoints:
        Checkpoint history DataFrame (filtered to target estimator).
    alpha:
        The alpha value to focus on.
    x_axis:
        ``'iteration'`` (default) or ``'time'`` for elapsed seconds.
    figures_dir:
        If not None, save figure here as a PNG.
    """
    checkpoints = df_checkpoints[
        (df_checkpoints["Dataset"] == dataset) & (df_checkpoints["Alpha"] == alpha)
    ].copy()
    if len(checkpoints) == 0:
        print(f"No checkpoint data for {dataset}, alpha={alpha}")
        return

    all_features = checkpoints["N_Tentative"] + checkpoints["N_Rejected"] + checkpoints["N_Confirmed"]
    checkpoints["Undecidedness_Factor"] = np.where(
        all_features > 0, checkpoints["N_Tentative"] / all_features, np.nan
    )
    denom = checkpoints["N_Tentative"] + checkpoints["N_Confirmed"]
    checkpoints["Potential_Miss_Rate"] = np.where(
        denom > 0, checkpoints["N_Tentative"] / denom, np.nan
    )

    x_col = "Elapsed_Sec" if x_axis == "time" else "Iteration"
    x_label = "Time (seconds)" if x_axis == "time" else "Iteration"

    run_ids = sorted(checkpoints["Run_ID"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(14 * FIG_SCALE, 5 * FIG_SCALE))

    panels = [
        (
            axes[0],
            "Undecidedness_Factor",
            "Undecidedness Factor\n(Tentative / All Features)",
            "Undecidedness Factor",
            "All Decided",
        ),
        (
            axes[1],
            "Potential_Miss_Rate",
            "Potential Miss Rate\n(Tentative / (Tentative + Confirmed))",
            "Potential Miss Rate",
            "All Confirmed",
        ),
    ]

    for ax, col_name, ylabel, title, resolved_label in panels:
        for run_id in run_ids:
            run = checkpoints[
                (checkpoints["Run_ID"] == run_id) & (checkpoints["Checkpoint_ID"] > 0)
            ].sort_values("Checkpoint_ID")
            run_valid = run[~run[col_name].isna()]
            if len(run_valid) == 0:
                continue
            ax.plot(
                run_valid[x_col], run_valid[col_name],
                marker="o", markersize=5, linewidth=2, alpha=0.6,
                color=_VANILLA_COLOR,
                label="Vanilla run" if run_id == run_ids[0] else "",
            )

        ax.axhline(0, color=_GREEDY_COLOR, linestyle="--", linewidth=1.5, alpha=0.7,
                   label=resolved_label)
        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylim(bottom=-0.01)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle(
        f"Undecidedness & Potential Miss Rate \u2014 {dataset}, Alpha = {alpha}",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    _save_fig(fig, f"undecidedness_single_alpha_{dataset}_{alpha}", figures_dir)
    plt.show()


# ---------------------------------------------------------------------------
# 12. Checkpoint statistical equivalence tests
# ---------------------------------------------------------------------------

def test_checkpoint_equivalence(
    dataset: str,
    alpha: float,
    checkpoint_id: int,
    df_checkpoints: pd.DataFrame,
    df_greedy: pd.DataFrame,
    significance_level: float = 0.05,
) -> Optional[dict]:
    """Mann-Whitney U test: checkpoint vs Greedy performance.

    Parameters
    ----------
    dataset:
        Dataset name.
    alpha:
        Alpha value for the run.
    checkpoint_id:
        Checkpoint ID to test.
    df_checkpoints:
        Checkpoint history DataFrame (filtered to target estimator).
    df_greedy:
        Greedy results DataFrame from :func:`prepare_greedy`.
    significance_level:
        p-value threshold for equivalence (default 0.05).

    Returns
    -------
    dict with test results, or None if insufficient data.
    """
    cp_data = df_checkpoints[
        (df_checkpoints["Dataset"] == dataset)
        & (df_checkpoints["Alpha"] == alpha)
        & (df_checkpoints["Checkpoint_ID"] == checkpoint_id)
    ]
    greedy_data = df_greedy[
        (df_greedy["Dataset"] == dataset) & (df_greedy["Alpha"] == alpha)
    ]

    if len(cp_data) == 0 or len(greedy_data) == 0:
        return None

    checkpoint_scores = cp_data["Score_Mean"].values
    greedy_scores = greedy_data["Score_Confirmed_Mean"].values

    if len(checkpoint_scores) < 2 or len(greedy_scores) < 2:
        return None

    statistic, p_value = mannwhitneyu(checkpoint_scores, greedy_scores, alternative="two-sided")
    t_stat, p_value_ttest = ttest_ind(checkpoint_scores, greedy_scores, equal_var=False)

    return {
        "Dataset": dataset,
        "Alpha": alpha,
        "Checkpoint_ID": checkpoint_id,
        "Iteration": cp_data["Iteration"].iloc[0],
        "Elapsed_Sec": cp_data["Elapsed_Sec"].mean(),
        "N_Confirmed": cp_data["N_Confirmed"].mean(),
        "Checkpoint_Mean": checkpoint_scores.mean(),
        "Greedy_Mean": greedy_scores.mean(),
        "Diff": checkpoint_scores.mean() - greedy_scores.mean(),
        "P_Value_MW": p_value,
        "P_Value_TTest": p_value_ttest,
        "Is_Equivalent": p_value > significance_level,
        "N_Samples_CP": len(checkpoint_scores),
        "N_Samples_Greedy": len(greedy_scores),
    }


def run_checkpoint_equivalence_tests(
    df_checkpoints: pd.DataFrame,
    df_greedy: pd.DataFrame,
    datasets: list,
    alphas: list,
    significance_level: float = 0.05,
) -> pd.DataFrame:
    """Run :func:`test_checkpoint_equivalence` for all dataset/alpha/checkpoint combinations.

    Parameters
    ----------
    df_checkpoints:
        Checkpoint history DataFrame (filtered to target estimator).
    df_greedy:
        Greedy results DataFrame from :func:`prepare_greedy`.
    datasets:
        List of dataset names.
    alphas:
        List of alpha values.
    significance_level:
        p-value threshold for equivalence.

    Returns
    -------
    DataFrame of all test results.
    """
    results = []
    for dataset in datasets:
        for alpha in alphas:
            cp_ids = df_checkpoints[
                (df_checkpoints["Dataset"] == dataset)
                & (df_checkpoints["Alpha"] == alpha)
            ]["Checkpoint_ID"].unique()

            for cp_id in sorted(cp_ids):
                result = test_checkpoint_equivalence(
                    dataset, alpha, cp_id,
                    df_checkpoints, df_greedy,
                    significance_level=significance_level,
                )
                if result is not None:
                    results.append(result)

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# 13. Minimum iterations for equivalence
# ---------------------------------------------------------------------------

def find_minimum_equivalence(
    df_tests: pd.DataFrame,
    df_greedy: pd.DataFrame,
    datasets: list,
    alphas: list,
) -> pd.DataFrame:
    """Find the earliest checkpoint achieving statistical equivalence with Greedy.

    Parameters
    ----------
    df_tests:
        Output of :func:`run_checkpoint_equivalence_tests`.
    df_greedy:
        Greedy results DataFrame from :func:`prepare_greedy`.
    datasets:
        List of dataset names.
    alphas:
        List of alpha values.

    Returns
    -------
    DataFrame with one row per (dataset, alpha), summarising the minimum
    iterations / time needed for equivalence and the Greedy baseline.
    """
    results = []
    for dataset in datasets:
        for alpha in alphas:
            subset = df_tests[
                (df_tests["Dataset"] == dataset) & (df_tests["Alpha"] == alpha)
            ].sort_values("Checkpoint_ID")

            greedy_slice = df_greedy[
                (df_greedy["Dataset"] == dataset) & (df_greedy["Alpha"] == alpha)
            ]
            greedy_time = greedy_slice["Time_Sec"].mean()
            greedy_score = greedy_slice["Score_Confirmed_Mean"].mean()

            equiv = subset[subset["Is_Equivalent"] == True]
            if len(equiv) > 0:
                first = equiv.iloc[0]
                results.append({
                    "Dataset": dataset,
                    "Alpha": alpha,
                    "Min_Iter_Equiv": first["Iteration"],
                    "Time_To_Equiv": first["Elapsed_Sec"],
                    "Greedy_Time": greedy_time,
                    "Time_Ratio": first["Elapsed_Sec"] / greedy_time if greedy_time > 0 else np.nan,
                    "N_Confirmed_At_Equiv": first["N_Confirmed"],
                    "Score_At_Equiv": first["Checkpoint_Mean"],
                    "Greedy_Score": greedy_score,
                    "P_Value": first["P_Value_MW"],
                })
            else:
                results.append({
                    "Dataset": dataset,
                    "Alpha": alpha,
                    "Min_Iter_Equiv": np.nan,
                    "Time_To_Equiv": np.nan,
                    "Greedy_Time": greedy_time,
                    "Time_Ratio": np.nan,
                    "N_Confirmed_At_Equiv": np.nan,
                    "Score_At_Equiv": np.nan,
                    "Greedy_Score": greedy_score,
                    "P_Value": np.nan,
                })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# 14. Summary statistics
# ---------------------------------------------------------------------------

def print_equivalence_summary(df_min_equiv: pd.DataFrame) -> None:
    """Print a human-readable summary of checkpoint equivalence results.

    Parameters
    ----------
    df_min_equiv:
        Output of :func:`find_minimum_equivalence`.
    """
    equiv_achieved = df_min_equiv[~df_min_equiv["Min_Iter_Equiv"].isna()]

    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    if len(equiv_achieved) == 0:
        print("\nNo configurations achieved statistical equivalence.")
        return

    print(f"\nConfigurations achieving equivalence: {len(equiv_achieved)} / {len(df_min_equiv)}")
    print(f"Average iterations to equivalence : {equiv_achieved['Min_Iter_Equiv'].mean():.1f}")
    print(f"Median  iterations to equivalence : {equiv_achieved['Min_Iter_Equiv'].median():.1f}")
    print(f"Average time to equivalence       : {equiv_achieved['Time_To_Equiv'].mean():.2f}s")
    print(f"Average Greedy time               : {equiv_achieved['Greedy_Time'].mean():.2f}s")
    print(f"Average time ratio (Vanilla/Greedy): {equiv_achieved['Time_Ratio'].mean():.2f}x")
    print(f"Median  time ratio                : {equiv_achieved['Time_Ratio'].median():.2f}x")

    print("\n" + "-" * 80)
    print("By Dataset:")
    print("-" * 80)
    summary = equiv_achieved.groupby("Dataset").agg(
        Min_Iter_mean=("Min_Iter_Equiv", "mean"),
        Min_Iter_std=("Min_Iter_Equiv", "std"),
        Time_To_Equiv_mean=("Time_To_Equiv", "mean"),
        Time_To_Equiv_std=("Time_To_Equiv", "std"),
        Time_Ratio_mean=("Time_Ratio", "mean"),
        Time_Ratio_std=("Time_Ratio", "std"),
    ).round(2)
    try:
        from IPython.display import display
        display(summary)
    except Exception:
        print(summary.to_string())


# ---------------------------------------------------------------------------
# 15. Equivalence visualization
# ---------------------------------------------------------------------------

def plot_equivalence(
    df_min_equiv: pd.DataFrame,
    figures_dir: Optional[str] = None,
) -> None:
    """Bar charts of iterations and time ratio needed to reach Greedy equivalence.

    Parameters
    ----------
    df_min_equiv:
        Output of :func:`find_minimum_equivalence`.
    figures_dir:
        If not None, save figure here as a PNG.
    """
    equiv_achieved = df_min_equiv[~df_min_equiv["Min_Iter_Equiv"].isna()].copy()
    if len(equiv_achieved) == 0:
        print("No equivalence data to plot.")
        return

    equiv_achieved["Alpha"] = equiv_achieved["Alpha"].astype(str)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14 * FIG_SCALE, 5 * FIG_SCALE))

    sns.barplot(data=equiv_achieved, x="Alpha", y="Min_Iter_Equiv",
                hue="Dataset", ax=ax1, palette=PAL_DATASET)
    ax1.set_ylabel("Minimum Iterations", fontsize=12)
    ax1.set_xlabel("Alpha", fontsize=12)
    ax1.set_title("Iterations Required for Statistical Equivalence",
                  fontsize=13, fontweight="bold")
    ax1.legend(title="Dataset", fontsize=9)
    ax1.grid(True, alpha=0.3, axis="y")

    sns.barplot(data=equiv_achieved, x="Alpha", y="Time_Ratio",
                hue="Dataset", ax=ax2, palette=PAL_DATASET)
    ax2.set_ylabel("Time Ratio (Vanilla / Greedy)", fontsize=12)
    ax2.set_xlabel("Alpha", fontsize=12)
    ax2.set_title("Time Overhead to Reach Equivalence",
                  fontsize=13, fontweight="bold")
    ax2.axhline(y=1, color="red", linestyle="--", linewidth=2, label="Parity")
    ax2.legend(title="Dataset", fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    _save_fig(fig, "equivalence_summary", figures_dir)
    plt.show()


# ---------------------------------------------------------------------------
# 16. Key findings printout
# ---------------------------------------------------------------------------

def print_key_findings(
    df_viz: pd.DataFrame,
    df_min_equiv: pd.DataFrame,
    significance_level: float = 0.1,
) -> None:
    """Print a structured summary of score findings and convergence results.

    Parameters
    ----------
    df_viz:
        Reshaped DataFrame from :func:`explode_variants`.
    df_min_equiv:
        Output of :func:`find_minimum_equivalence`.
    significance_level:
        Threshold used when calling :func:`run_score_statistical_tests`.
    """
    stats_df = run_score_statistical_tests(df_viz, significance_level=significance_level)
    equiv_achieved = df_min_equiv[~df_min_equiv["Min_Iter_Equiv"].isna()]

    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    # Score winner tally
    if len(stats_df) > 0:
        print("\n1. SCORE COMPARISON (Greedy vs Vanilla Strict)")
        tally = stats_df["Winner_Stat"].value_counts()
        for winner, count in tally.items():
            print(f"   {winner}: {count} configuration(s)")

    # Speedup
    if len(stats_df) > 0:
        mean_speedup = stats_df["Speedup"].mean()
        print(f"\n2. AVERAGE SPEEDUP: {mean_speedup:.1f}x  (Vanilla time / Greedy time)")

    # Convergence
    if len(equiv_achieved) > 0:
        median_ratio = equiv_achieved["Time_Ratio"].median()
        print(f"\n3. CONVERGENCE SPEED")
        print(f"   Vanilla reaches Greedy equivalence in "
              f"{equiv_achieved['Min_Iter_Equiv'].mean():.1f} iterations on average")
        print(f"   Taking {median_ratio:.1f}x Greedy time (median)")

        fastest = equiv_achieved.loc[equiv_achieved["Min_Iter_Equiv"].idxmin()]
        slowest = equiv_achieved.loc[equiv_achieved["Min_Iter_Equiv"].idxmax()]
        print(f"\n4. FASTEST CONVERGENCE")
        print(f"   Dataset={fastest['Dataset']}, Alpha={fastest['Alpha']}, "
              f"Iterations={fastest['Min_Iter_Equiv']:.0f}, "
              f"Time={fastest['Time_To_Equiv']:.2f}s")
        print(f"\n5. SLOWEST CONVERGENCE")
        print(f"   Dataset={slowest['Dataset']}, Alpha={slowest['Alpha']}, "
              f"Iterations={slowest['Min_Iter_Equiv']:.0f}, "
              f"Time={slowest['Time_To_Equiv']:.2f}s")
    else:
        print("\n3. CONVERGENCE: No configurations achieved statistical equivalence.")

    print("\n" + "=" * 80)


# ---------------------------------------------------------------------------
# 17. Export
# ---------------------------------------------------------------------------

def export_results(
    df_tests: pd.DataFrame,
    df_min_equiv: pd.DataFrame,
    tests_path: str = "../data/checkpoint_statistical_tests.csv",
    equiv_path: str = "../data/checkpoint_equivalence_summary.csv",
) -> None:
    """Write statistical test results and equivalence summary to CSV files.

    Parameters
    ----------
    df_tests:
        Output of :func:`run_checkpoint_equivalence_tests`.
    df_min_equiv:
        Output of :func:`find_minimum_equivalence`.
    tests_path:
        Destination path for the detailed test results CSV.
    equiv_path:
        Destination path for the equivalence summary CSV.
    """
    df_tests.to_csv(tests_path, index=False)
    df_min_equiv.to_csv(equiv_path, index=False)
    print(f"Exported:\n  {tests_path}\n  {equiv_path}")
