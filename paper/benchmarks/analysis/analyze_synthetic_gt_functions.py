"""
analyze_synthetic_gt_functions.py
==================================
Reusable analysis and plotting functions for the synthetic dataset ground truth
analysis of Vanilla Boruta vs GreedyParallel Boruta.

All functions operate on DataFrames and configuration values that are passed in
explicitly, making them easy to test in isolation and straightforward to reuse
across notebooks or scripts.

Sections
--------
1. Data loading & preparation
2. Summary tables
3. Final-state bar / point plots
4. Confusion breakdown
5. Convergence trajectory plots (per-alpha and single-alpha focus)
6. Recall–Precision phase-space paths
7. Feature-count trajectories
8. Undecidedness & potential-miss-rate diagnostics
9. F1 heatmap
10. Speed vs quality scatter
11. Alpha-effect boxplots
12. Speedup analysis
13. Statistical tests
14. Mean trajectory with confidence band
15. Key findings printout
16. Export
"""

from __future__ import annotations

import warnings
try:
    from IPython.display import display as _ipy_display
except ImportError:
    _ipy_display = None
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Helper: save figure to disk
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
# Constants / defaults (callers may override by passing their own values)
# ---------------------------------------------------------------------------
# Central colour tokens — kept in sync with analyze_results_functions.py
_GREEDY_COLOR  = "#e47e1e"
_VANILLA_COLOR = "#6C6D6D"

PAL_METHOD_DEFAULT: dict[str, str] = {
    "Vanilla":        _VANILLA_COLOR,
    "GreedyParallel": _GREEDY_COLOR,
    "Greedy":         _GREEDY_COLOR,
}
METRICS_DEFAULT = ["Recall", "Precision", "Specificity", "F1"]
GREEDY_MARKER = "P"
VANILLA_MARKER = "o"

# Global figure scale — multiply all base dimensions by this value.
# Increase (e.g. 1.5 or 2.0) if figures look too small in the exported PDF.
FIG_SCALE: float = 0.8


# ---------------------------------------------------------------------------
# 0. Plot theme
# ---------------------------------------------------------------------------

def apply_plot_theme(font_scale: float = 1.2, dpi: int = 110) -> None:
    """Apply the standard seaborn whitegrid paper theme and matplotlib DPI.

    Parameters
    ----------
    font_scale:
        Seaborn font scale factor (default 1.2).
    dpi:
        Figure DPI for inline display (default 110).
    """
    sns.set_theme(style="whitegrid", context="paper", font_scale=font_scale)
    plt.rcParams["figure.dpi"] = dpi


# ---------------------------------------------------------------------------
# 1. Data loading & preparation
# ---------------------------------------------------------------------------

def load_data(
    gt_file: str,
    main_file: str,
    dataset: str,
    estimator: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and filter the GT and main benchmark result CSVs.

    Parameters
    ----------
    gt_file:
        Path to the benchmark_gt_results CSV (contains checkpoint-level rows).
    main_file:
        Path to the benchmark_results CSV (contains per-run summary rows).
    dataset:
        Dataset name to filter on (e.g. ``'correlated_noise'``).
    estimator:
        Estimator name to filter on (e.g. ``'extra_trees'``).

    Returns
    -------
    df_gt:
        Filtered GT log DataFrame.
    df_main:
        Filtered main results DataFrame.
    """
    df_gt = pd.read_csv(gt_file)
    df_main = pd.read_csv(main_file)

    df_gt = df_gt[
        (df_gt["Dataset"] == dataset) & (df_gt["Estimator"] == estimator)
    ].copy()
    df_main = df_main[
        (df_main["Dataset"] == dataset) & (df_main["Estimator"] == estimator)
    ].copy()

    print(f"GT log rows  : {len(df_gt)}")
    print(f"Main log rows: {len(df_main)}")
    print(f"Methods      : {df_gt['Method'].unique()}")
    print(f"Alpha values : {sorted(df_gt['Alpha'].unique())}")
    if len(df_gt):
        print(f"GT Broad size: {df_gt['GT_Broad_Size'].iloc[0]}")

    return df_gt, df_main


def prepare_data(
    df_gt: pd.DataFrame,
    gt_label: str = "Broad",
) -> dict:
    """Split the GT log into Vanilla / Greedy sub-frames and extract metadata.

    Parameters
    ----------
    df_gt:
        Filtered GT log DataFrame from :func:`load_data`.
    gt_label:
        Which ground-truth column suffix to use (default ``'Broad'``).

    Returns
    -------
    A dict with keys:
        ``df_vanilla``, ``df_greedy``, ``df_vanilla_final``, ``df_greedy_final``,
        ``greedy_label``, ``alphas``, ``n_features``, ``gt_broad``, ``GT``.
    """
    df_vanilla = df_gt[df_gt["Method"] == "Vanilla"].copy()
    df_greedy = df_gt[df_gt["Method"].str.contains("Greedy")].copy()

    if "GreedyParallel" in df_greedy["Method"].values:
        df_greedy = df_greedy[df_greedy["Method"] == "GreedyParallel"].copy()
        greedy_label = "GreedyParallel"
    else:
        greedy_label = "Greedy"

    df_vanilla_final = df_vanilla[df_vanilla["Is_Final"] == True].copy()
    df_greedy_final = df_greedy.copy()

    alphas = sorted(df_gt["Alpha"].unique())
    n_features = int(df_gt["n_features_orig"].iloc[0])
    gt_broad = int(df_gt["GT_Broad_Size"].iloc[0])

    print(f"Vanilla checkpoints      : {len(df_vanilla)}")
    print(f"Vanilla final-state rows : {len(df_vanilla_final)}")
    print(f"{greedy_label} snapshots : {len(df_greedy)}")
    print(f"Alpha values: {alphas}")
    print(f"n_features={n_features}, GT broad={gt_broad}")

    return dict(
        df_vanilla=df_vanilla,
        df_greedy=df_greedy,
        df_vanilla_final=df_vanilla_final,
        df_greedy_final=df_greedy_final,
        greedy_label=greedy_label,
        alphas=alphas,
        n_features=n_features,
        gt_broad=gt_broad,
        GT=gt_label,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _gt_col(metric: str, GT: str) -> str:
    """Return the DataFrame column name for a metric under the given GT label."""
    return f"{metric}_{GT}"


def _greedy_hline(
    ax: plt.Axes,
    df_greedy: pd.DataFrame,
    metric: str,
    alpha: float,
    GT: str,
    greedy_label: str,
    pal: dict,
    idx: int = 0,
) -> Optional[float]:
    """Draw a dashed horizontal reference line at the Greedy mean for *metric*.

    Returns the mean value, or None if no data is available.
    """
    g = df_greedy[df_greedy["Alpha"] == alpha][_gt_col(metric, GT)]
    if len(g) == 0:
        return None
    gm = g.mean()
    ax.axhline(
        gm,
        color=pal.get(greedy_label, "#00b894"),
        linestyle="--",
        linewidth=2,
        label=f"{greedy_label} mean" if idx == 0 else "",
    )
    return gm


def _greedy_iterations(alpha: float, n_features: int, bonferroni: bool = True) -> int:
    """Compute the theoretical GreedyBoruta convergence iteration count.

    Replicates ``GreedyBorutaParallel._compute_iterations_to_convergence``.

    Parameters
    ----------
    alpha:
        Significance level.
    n_features:
        Total number of features (used for Bonferroni correction).
    bonferroni:
        If True (default), divide alpha by n_features before computing.
    """
    threshold = alpha / float(n_features) if bonferroni else alpha
    return int(np.ceil(np.log2(1.0 / threshold)))


# ---------------------------------------------------------------------------
# 2. Final-state summary table
# ---------------------------------------------------------------------------

def build_final_summary_table(
    df_vanilla_final: pd.DataFrame,
    df_greedy_final: pd.DataFrame,
    alphas: list,
    GT: str,
    greedy_label: str,
    metrics: list = METRICS_DEFAULT,
) -> pd.DataFrame:
    """Build a tidy DataFrame of mean ± std for every metric × method × alpha.

    Parameters
    ----------
    df_vanilla_final, df_greedy_final:
        Final-state rows for each method.
    alphas:
        Sorted list of alpha values.
    GT:
        Ground-truth column suffix (e.g. ``'Broad'``).
    greedy_label:
        Label for the Greedy method (e.g. ``'GreedyParallel'``).
    metrics:
        List of metric names (default :data:`METRICS_DEFAULT`).

    Returns
    -------
    df_final_summary:
        DataFrame with columns Method, Alpha, Metric, Mean, Std.
    """
    rows = []
    for metric in metrics:
        col = _gt_col(metric, GT)
        for alpha in alphas:
            for method_df, method_name in [
                (df_vanilla_final, "Vanilla"),
                (df_greedy_final, greedy_label),
            ]:
                subset = method_df[method_df["Alpha"] == alpha][col]
                if len(subset) == 0:
                    continue
                rows.append(
                    {
                        "Method": method_name,
                        "Alpha": str(alpha),
                        "Metric": metric,
                        "Mean": subset.mean(),
                        "Std": subset.std(),
                    }
                )

    df = pd.DataFrame(rows)
    print(f"Final-state metrics — {GT} GT (mean ± std across runs):")
    pivot = df.pivot_table(
        index=["Method", "Alpha"], columns="Metric", values="Mean"
    )[metrics].round(3)
    try:
        from IPython.display import display
        display(pivot)
    except Exception:
        print(pivot.to_string())
    return df


# ---------------------------------------------------------------------------
# 3. Final-state barplot (all metrics)
# ---------------------------------------------------------------------------

def compute_barplot_significance(
    df_vanilla_final: pd.DataFrame,
    df_greedy_final: pd.DataFrame,
    alphas: list,
    GT: str,
    greedy_label: str,
    metrics: list = METRICS_DEFAULT,
    significance: float = 0.05,
) -> dict:
    """Compute Welch t-test results for every (metric, alpha) pair.

    Two tests are run per pair:
    - Two-sided: detects any difference between methods.
    - One-sided (Greedy > Vanilla): tests whether Greedy is strictly higher.

    For metrics where Vanilla is expected to be higher (e.g. Specificity,
    Precision), a significant two-sided result with Vanilla_mean > Greedy_mean
    is reported as ``direction='vanilla'``.

    Parameters
    ----------
    df_vanilla_final, df_greedy_final:
        Final-state rows (one row per run).
    alphas:
        Sorted list of alpha values.
    GT:
        GT column suffix (e.g. ``'Broad'``).
    greedy_label:
        Display label for the Greedy method.
    metrics:
        Metrics to test (default: Recall, Precision, Specificity, F1).
    significance:
        p-value threshold (default 0.05).

    Returns
    -------
    dict
        Keys are ``(metric, alpha_str)`` tuples.  Each value is a dict with:
        ``p_two``, ``p_greedy_better``, ``direction`` (``'greedy'``,
        ``'vanilla'``, or ``'tie'``), ``stars`` (e.g. ``'***'``),
        ``vanilla_mean``, ``greedy_mean``.
    """
    def _stars(p: float) -> str:
        if p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        return "ns"

    result = {}
    for metric in metrics:
        col = _gt_col(metric, GT)
        for alpha in alphas:
            v = df_vanilla_final[df_vanilla_final["Alpha"] == alpha][col].values
            g = df_greedy_final[df_greedy_final["Alpha"] == alpha][col].values
            if len(v) < 2 or len(g) < 2:
                continue

            _, p_two = stats.ttest_ind(v, g, equal_var=False)
            _, p_greedy_better = stats.ttest_ind(
                g, v, equal_var=False, alternative="greater"
            )

            if p_greedy_better < significance:
                direction = "greedy"
                stars = _stars(p_greedy_better)
            elif p_two < significance and v.mean() > g.mean():
                direction = "vanilla"
                stars = _stars(p_two)
            else:
                direction = "tie"
                stars = _stars(p_two)

            result[(metric, str(alpha))] = {
                "p_two": p_two,
                "p_greedy_better": p_greedy_better,
                "direction": direction,
                "stars": stars,
                "vanilla_mean": v.mean(),
                "greedy_mean": g.mean(),
            }
    return result


def _annotate_significance(
    ax,
    x_pos: int,
    y_vanilla: float,
    y_greedy: float,
    stars: str,
    direction: str,
    dodge: float = 0.4,
    y_top: float = 1.02,
) -> None:
    """Draw a significance bracket between the two method point estimates.

    Parameters
    ----------
    ax:
        Matplotlib Axes to annotate.
    x_pos:
        Integer x-position of the alpha group on the axis.
    y_vanilla, y_greedy:
        Mean values of Vanilla and Greedy at this alpha (used to anchor the
        bracket slightly above the higher of the two points).
    stars:
        Significance label (e.g. ``'*'``, ``'**'``, ``'ns'``).
    direction:
        ``'greedy'``, ``'vanilla'``, or ``'tie'`` — controls the arrow/label.
    dodge:
        Horizontal dodge used in the pointplot (default 0.4).
    y_top:
        Upper y-axis limit; bracket is placed relative to this.
    """
    if stars == "ns":
        return  # skip non-significant pairs to keep the plot clean

    x_v = x_pos - dodge / 2  # Vanilla point x (seaborn dodges left)
    x_g = x_pos + dodge / 2  # Greedy point x (seaborn dodges right)

    bracket_y = max(y_vanilla, y_greedy) + 0.012
    bracket_y = min(bracket_y, y_top - 0.04)
    tick_h = 0.005

    # Direction arrow prepended to star label
    if direction == "greedy":
        label = f"G>{stars}"
    elif direction == "vanilla":
        label = f"V>{stars}"
    else:
        label = stars

    # Horizontal bar
    ax.plot([x_v, x_g], [bracket_y, bracket_y], color="0.3", lw=1.0, clip_on=False)
    # Tick marks
    ax.plot([x_v, x_v], [bracket_y - tick_h, bracket_y], color="0.3", lw=1.0, clip_on=False)
    ax.plot([x_g, x_g], [bracket_y - tick_h, bracket_y], color="0.3", lw=1.0, clip_on=False)
    # Label
    ax.text(
        (x_v + x_g) / 2, bracket_y + 0.002,
        label,
        ha="center", va="bottom", fontsize=7.5, color="0.2",
    )


def plot_final_state_barplot(
    df_vanilla_final: pd.DataFrame,
    df_greedy_final: pd.DataFrame,
    alphas: list,
    GT: str,
    greedy_label: str,
    dataset: str,
    estimator: str,
    metrics: list = METRICS_DEFAULT,
    pal: dict = PAL_METHOD_DEFAULT,
    figures_dir: "str | None" = "../figures/",
) -> None:
    """Four-panel strip + point plot of final-state GT metrics vs alpha.

    Parameters
    ----------
    df_vanilla_final, df_greedy_final:
        Final-state rows.
    alphas:
        Sorted alpha values.
    GT:
        GT column suffix.
    greedy_label:
        Label for the Greedy method.
    dataset, estimator:
        Used in the figure title.
    metrics:
        Metrics to show (one panel each, max 4).
    pal:
        Method colour palette dict.

    Returns
    -------
    None
    """
    method_order = ["Vanilla", greedy_label]
    method_colors = [pal["Vanilla"], pal.get(greedy_label, "#00b894")]
    alpha_order = [str(a) for a in sorted(alphas)]

    fig, axes = plt.subplots(2, 2, figsize=(14 * FIG_SCALE, 10 * FIG_SCALE))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        col = _gt_col(metric, GT)

        plot_data = pd.concat(
            [
                df_vanilla_final[["Method", "Alpha", col]],
                df_greedy_final[["Method", "Alpha", col]],
            ],
            ignore_index=True,
        )
        plot_data["Alpha"] = plot_data["Alpha"].astype(str)

        sns.stripplot(
            data=plot_data, x="Alpha", y=col, hue="Method",
            hue_order=method_order,
            palette=dict(zip(method_order, method_colors)),
            order=alpha_order, ax=ax,
            dodge=True, jitter=0.15,
            size=6, alpha=0.45, linewidth=0.4, edgecolor="white",
            zorder=3, legend=False,
        )
        sns.pointplot(
            data=plot_data, x="Alpha", y=col, hue="Method",
            hue_order=method_order,
            palette=dict(zip(method_order, method_colors)),
            order=alpha_order, ax=ax,
            dodge=0.4, markers=["o", "P"],
            capsize=0.1, err_kws={"linewidth": 1.5},
            linestyles="none", markersize=7, zorder=5,
        )

        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.set_xlabel("Alpha", fontsize=11)
        ax.set_ylabel(metric, fontsize=11)
        col_min = plot_data[col].min()
        ax.set_ylim(max(0, col_min - 0.05), 1.02)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.legend(title="Method", fontsize=9, title_fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"Final-State Ground Truth Metrics ({GT} GT)",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    if _ipy_display is not None:
        _ipy_display(fig)
    _save_fig(fig, "final_state_barplot", figures_dir)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Confusion breakdown
# ---------------------------------------------------------------------------

def plot_confusion_breakdown(
    df_vanilla_final: pd.DataFrame,
    df_greedy_final: pd.DataFrame,
    alphas: list,
    GT: str,
    greedy_label: str,
    dataset: str,
    estimator: str,
    figures_dir: "str | None" = "../figures/",
) -> None:
    """Grouped bar chart of mean TP/FP/FN/TN counts per method and alpha.

    Parameters
    ----------
    df_vanilla_final, df_greedy_final:
        Final-state rows.
    alphas:
        Sorted alpha values (one subplot per alpha).
    GT:
        GT column suffix.
    greedy_label:
        Label for the Greedy method.
    dataset, estimator:
        Used in the figure title.

    Returns
    -------
    matplotlib Figure.
    """
    count_rows = []
    for alpha in alphas:
        for method_df, method_name in [
            (df_vanilla_final, "Vanilla"),
            (df_greedy_final, greedy_label),
        ]:
            sub = method_df[method_df["Alpha"] == alpha]
            if len(sub) == 0:
                continue
            for cnt in ["TP", "FP", "FN", "TN"]:
                count_rows.append(
                    {
                        "Method": method_name,
                        "Alpha": str(alpha),
                        "Count_Type": cnt,
                        "Count": sub[f"{cnt}_{GT}"].mean(),
                    }
                )

    df_counts = pd.DataFrame(count_rows)
    cnt_pal = {"TP": "#2ecc71", "FP": "#e74c3c", "FN": "#e67e22", "TN": "#95a5a6"}

    fig, axes = plt.subplots(1, len(alphas), figsize=(5 * len(alphas) * FIG_SCALE, 5 * FIG_SCALE), sharey=True)
    if len(alphas) == 1:
        axes = [axes]

    for idx, alpha in enumerate(alphas):
        ax = axes[idx]
        sub = df_counts[df_counts["Alpha"] == str(alpha)]
        sns.barplot(
            data=sub, x="Method", y="Count", hue="Count_Type",
            hue_order=["TP", "FP", "FN", "TN"],
            palette=cnt_pal, ax=ax,
        )
        ax.set_title(f"Alpha = {alpha}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Method", fontsize=11)
        ax.set_ylabel("Mean Count (across runs)" if idx == 0 else "", fontsize=11)
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(title="Type", fontsize=9)

    fig.suptitle(
        f"Confusion Breakdown — {GT} GT",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    # plt.show()
    if _ipy_display is not None:
        _ipy_display(fig)
    _save_fig(fig, "confusion_breakdown", figures_dir)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Convergence trajectory plots
# ---------------------------------------------------------------------------

def plot_metric_trajectories(
    metric: str,
    df_vanilla: pd.DataFrame,
    df_greedy: pd.DataFrame,
    alphas: list,
    n_features: int,
    GT: str,
    greedy_label: str,
    dataset: str,
    estimator: str,
    x_col: str = "Iteration",
    x_label: str = "Iteration",
    pal: dict = PAL_METHOD_DEFAULT,
    figures_dir: "str | None" = "../figures/",
) -> None:
    """Line plot of a single GT metric across all Vanilla runs, one panel per alpha.

    A dashed horizontal line shows the GreedyParallel mean and a vertical dotted
    line marks the GreedyParallel reference point on the x-axis.

    Parameters
    ----------
    metric:
        Metric name (e.g. ``'Recall'``).
    df_vanilla:
        All Vanilla checkpoint rows.
    df_greedy:
        All Greedy rows.
    alphas:
        Sorted alpha values.
    n_features:
        Total feature count (for theoretical iteration calculation).
    GT:
        GT column suffix.
    greedy_label:
        Label for the Greedy method.
    dataset, estimator:
        Used in the figure title.
    x_col:
        Column to use for the x-axis (``'Iteration'`` or ``'Elapsed_Sec'``).
    x_label:
        Axis label for the x-axis.
    pal:
        Method colour palette dict.

    Returns
    -------
    matplotlib Figure.
    """
    col = _gt_col(metric, GT)
    n_alphas = len(alphas)
    fig, axes = plt.subplots(1, n_alphas, figsize=(6 * n_alphas * FIG_SCALE, 5 * FIG_SCALE), squeeze=False)
    axes = axes.flatten()

    for idx, alpha in enumerate(alphas):
        ax = axes[idx]
        cp = df_vanilla[df_vanilla["Alpha"] == alpha]
        run_ids = sorted(cp["Run_ID"].unique())

        for run_id in run_ids:
            run = cp[cp["Run_ID"] == run_id].sort_values("Checkpoint_ID")
            ax.plot(
                run[x_col], run[col],
                marker=VANILLA_MARKER, markersize=5, linewidth=2, alpha=0.65,
                color=pal["Vanilla"],
                label="Vanilla runs" if (idx == 0 and run_id == run_ids[0]) else "",
            )

        g_vals = df_greedy[df_greedy["Alpha"] == alpha][col]
        if len(g_vals):
            ax.axhline(
                g_vals.mean(),
                color=pal.get(greedy_label, "#00b894"),
                linestyle="--", linewidth=2.5,
                label=f"{greedy_label} mean" if idx == 0 else "",
            )

        if x_col == "Iteration":
            greedy_x = _greedy_iterations(alpha, n_features)
            vline_label = f"{greedy_label} iters ({greedy_x})" if idx == 0 else ""
        else:
            g_alpha = df_greedy[df_greedy["Alpha"] == alpha]
            greedy_x = g_alpha["Elapsed_Sec"].mean() if len(g_alpha) else None
            vline_label = f"{greedy_label} mean time" if idx == 0 else ""

        if greedy_x is not None:
            ax.axvline(
                greedy_x,
                color=pal.get(greedy_label, "#00b894"),
                linestyle=":", linewidth=1.8, alpha=0.7,
                label=vline_label,
            )

        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel(f"{metric} ({GT})", fontsize=11)
        ax.set_title(f"Alpha = {alpha}", fontsize=12, fontweight="bold")
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=9)

    fig.suptitle(
        f"{metric} ({GT} GT) vs {x_label}",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    # plt.show()
    if _ipy_display is not None:
        _ipy_display(fig)
    _save_fig(fig, f"metric_trajectories_{metric}", figures_dir)
    plt.close(fig)


def plot_convergence_single_alpha_iteration(
    focus_alpha: float,
    df_vanilla: pd.DataFrame,
    df_greedy: pd.DataFrame,
    n_features: int,
    GT: str,
    greedy_label: str,
    dataset: str,
    estimator: str,
    metrics: list = METRICS_DEFAULT,
    pal: dict = PAL_METHOD_DEFAULT,
    figures_dir: "str | None" = "../figures/",
) -> None:
    """2x2 convergence trajectory plot for a single alpha value vs iteration.

    All four metrics in one figure. Each line is one Vanilla run; dashed line
    is the GreedyParallel mean.

    Parameters
    ----------
    focus_alpha:
        The alpha value to plot.
    df_vanilla:
        All Vanilla checkpoint rows.
    df_greedy:
        All Greedy rows.
    n_features:
        Total feature count.
    GT:
        GT column suffix.
    greedy_label:
        Label for the Greedy method.
    dataset, estimator:
        Used in the figure title.
    metrics:
        Metrics to show (max 4, displayed in 2x2 grid).
    pal:
        Method colour palette dict.

    Returns
    -------
    matplotlib Figure.
    """
    cp_focus = df_vanilla[df_vanilla["Alpha"] == focus_alpha]
    greedy_x = _greedy_iterations(focus_alpha, n_features)
    run_ids = sorted(cp_focus["Run_ID"].unique())

    fig, axes = plt.subplots(2, 2, figsize=(14 * FIG_SCALE, 10 * FIG_SCALE))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        col = _gt_col(metric, GT)

        for run_id in run_ids:
            run = cp_focus[cp_focus["Run_ID"] == run_id].sort_values("Checkpoint_ID")
            ax.plot(
                run["Iteration"], run[col],
                marker=VANILLA_MARKER, markersize=4, linewidth=1.8, alpha=0.6,
                color=pal["Vanilla"],
                label="Vanilla runs" if run_id == run_ids[0] else "",
            )

        g_vals = df_greedy[df_greedy["Alpha"] == focus_alpha][col]
        if len(g_vals):
            ax.axhline(
                g_vals.mean(),
                color=pal.get(greedy_label, "#00b894"),
                linestyle="--", linewidth=2.5,
                label=f"{greedy_label} mean",
            )

        ax.axvline(
            greedy_x,
            color=pal.get(greedy_label, "#00b894"),
            linestyle=":", linewidth=1.8, alpha=0.7,
            label=f"{greedy_label} iters ({greedy_x})",
        )

        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.set_xlabel("Iteration", fontsize=11)
        ax.set_ylabel(f"{metric} ({GT})", fontsize=11)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle(
        f"Convergence Trajectories vs Iteration — Alpha = {focus_alpha}",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    # plt.show()
    if _ipy_display is not None:
        _ipy_display(fig)
    _save_fig(fig, f"convergence_single_alpha_iteration_{focus_alpha}", figures_dir)
    plt.close(fig)


def plot_convergence_single_alpha_time(
    focus_alpha: float,
    df_vanilla: pd.DataFrame,
    df_greedy: pd.DataFrame,
    GT: str,
    greedy_label: str,
    dataset: str,
    estimator: str,
    metrics: list = METRICS_DEFAULT,
    pal: dict = PAL_METHOD_DEFAULT,
    figures_dir: "str | None" = "../figures/",
) -> None:
    """2x2 convergence trajectory plot for a single alpha value vs elapsed time.

    Parameters
    ----------
    focus_alpha:
        The alpha value to plot.
    df_vanilla:
        All Vanilla checkpoint rows.
    df_greedy:
        All Greedy rows.
    GT:
        GT column suffix.
    greedy_label:
        Label for the Greedy method.
    dataset, estimator:
        Used in the figure title.
    metrics:
        Metrics to show (max 4, displayed in 2x2 grid).
    pal:
        Method colour palette dict.

    Returns
    -------
    matplotlib Figure.
    """
    cp_focus = df_vanilla[df_vanilla["Alpha"] == focus_alpha]
    g_focus = df_greedy[df_greedy["Alpha"] == focus_alpha]
    greedy_time = g_focus["Elapsed_Sec"].mean() if len(g_focus) else None
    run_ids = sorted(cp_focus["Run_ID"].unique())

    fig, axes = plt.subplots(2, 2, figsize=(14 * FIG_SCALE, 10 * FIG_SCALE))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        col = _gt_col(metric, GT)

        for run_id in run_ids:
            run = cp_focus[cp_focus["Run_ID"] == run_id].sort_values("Checkpoint_ID")
            ax.plot(
                run["Elapsed_Sec"], run[col],
                marker=VANILLA_MARKER, markersize=4, linewidth=1.8, alpha=0.6,
                color=pal["Vanilla"],
                label="Vanilla runs" if run_id == run_ids[0] else "",
            )

        g_vals = g_focus[col]
        if len(g_vals):
            ax.axhline(
                g_vals.mean(),
                color=pal.get(greedy_label, "#00b894"),
                linestyle="--", linewidth=2.5,
                label=f"{greedy_label} mean",
            )

        if greedy_time is not None:
            ax.axvline(
                greedy_time,
                color=pal.get(greedy_label, "#00b894"),
                linestyle=":", linewidth=1.8, alpha=0.7,
                label=f"{greedy_label} mean time",
            )

        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.set_xlabel("Time (seconds)", fontsize=11)
        ax.set_ylabel(f"{metric} ({GT})", fontsize=11)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle(
        f"Convergence Trajectories vs Time — Alpha = {focus_alpha}",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    # plt.show()
    if _ipy_display is not None:
        _ipy_display(fig)
    _save_fig(fig, f"convergence_single_alpha_time_{focus_alpha}", figures_dir)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. Recall–Precision phase-space paths
# ---------------------------------------------------------------------------

def plot_recall_precision_paths(
    df_vanilla: pd.DataFrame,
    df_greedy: pd.DataFrame,
    alphas: list,
    GT: str,
    greedy_label: str,
    dataset: str,
    estimator: str,
    pal: dict = PAL_METHOD_DEFAULT,
    figures_dir: "str | None" = "../figures/",
) -> None:
    """Recall–Precision phase-space trajectories for all alpha values.

    Each Vanilla run is drawn as a line with a square start-marker and a star
    end-marker. The GreedyParallel mean final position is shown as a large
    filled pentagon.

    Parameters
    ----------
    df_vanilla:
        All Vanilla checkpoint rows.
    df_greedy:
        All Greedy rows.
    alphas:
        Sorted alpha values (one subplot per alpha).
    GT, greedy_label, dataset, estimator:
        See other plot functions.
    pal:
        Method colour palette dict.

    Returns
    -------
    matplotlib Figure.
    """
    n_alphas = len(alphas)
    fig, axes = plt.subplots(1, n_alphas, figsize=(6 * n_alphas * FIG_SCALE, 5 * FIG_SCALE), squeeze=False)
    axes = axes.flatten()

    for idx, alpha in enumerate(alphas):
        ax = axes[idx]
        cp = df_vanilla[df_vanilla["Alpha"] == alpha]
        run_ids = sorted(cp["Run_ID"].unique())

        for run_id in run_ids:
            run = cp[cp["Run_ID"] == run_id].sort_values("Checkpoint_ID")
            ax.plot(
                run[_gt_col("Recall", GT)], run[_gt_col("Precision", GT)],
                marker=VANILLA_MARKER, markersize=4, linewidth=1.8, alpha=0.6,
                color=pal["Vanilla"],
            )
            ax.scatter(
                run[_gt_col("Recall", GT)].iloc[0],
                run[_gt_col("Precision", GT)].iloc[0],
                color="#2c3e50", zorder=6, s=55, marker="s",
                label="Start" if run_id == run_ids[0] else "",
            )
            ax.scatter(
                run[_gt_col("Recall", GT)].iloc[-1],
                run[_gt_col("Precision", GT)].iloc[-1],
                color="#2c3e50", zorder=6, s=80, marker="*",
                label="End" if run_id == run_ids[0] else "",
            )

        g = df_greedy[df_greedy["Alpha"] == alpha]
        if len(g):
            ax.scatter(
                g[_gt_col("Recall", GT)].mean(),
                g[_gt_col("Precision", GT)].mean(),
                color=pal.get(greedy_label, "#00b894"),
                s=250, marker=GREEDY_MARKER, zorder=7,
                edgecolors="black", linewidths=0.8,
                label=greedy_label,
            )

        ax.set_xlabel("Recall (Broad)", fontsize=11)
        ax.set_ylabel("Precision (Broad)", fontsize=11)
        ax.set_title(f"Alpha = {alpha}", fontsize=12, fontweight="bold")
        ax.set_xlim(-0.02, 1.05)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, alpha=0.3)
        ax.annotate(
            "Ideal\n(1, 1)", xy=(1, 1), xytext=(0.78, 0.86),
            fontsize=9, color="grey",
            arrowprops=dict(arrowstyle="->", color="grey", lw=1.2),
        )
        ax.legend(fontsize=9, loc="lower left")

    fig.suptitle(
        f"Recall\u2013Precision Path ({GT} GT) \u2014 Vanilla Boruta",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    # plt.show()
    if _ipy_display is not None:
        _ipy_display(fig)
    _save_fig(fig, "recall_precision_paths", figures_dir)
    plt.close(fig)


def plot_recall_precision_single_alpha(
    focus_alpha: float,
    df_vanilla: pd.DataFrame,
    df_greedy: pd.DataFrame,
    GT: str,
    greedy_label: str,
    dataset: str,
    estimator: str,
    pal: dict = PAL_METHOD_DEFAULT,
    figures_dir: "str | None" = "../figures/",
) -> None:
    """Single-panel Recall–Precision phase-space plot for one alpha value.

    Parameters
    ----------
    focus_alpha:
        Alpha value to focus on.
    df_vanilla:
        All Vanilla checkpoint rows.
    df_greedy:
        All Greedy rows.
    GT, greedy_label, dataset, estimator:
        See other plot functions.
    pal:
        Method colour palette dict.

    Returns
    -------
    matplotlib Figure.
    """
    cp_focus = df_vanilla[df_vanilla["Alpha"] == focus_alpha]
    run_ids = sorted(cp_focus["Run_ID"].unique())

    fig, ax = plt.subplots(figsize=(7 * FIG_SCALE, 6 * FIG_SCALE))

    for run_id in run_ids:
        run = cp_focus[cp_focus["Run_ID"] == run_id].sort_values("Checkpoint_ID")
        ax.plot(
            run[_gt_col("Recall", GT)], run[_gt_col("Precision", GT)],
            marker=VANILLA_MARKER, markersize=4, linewidth=1.8, alpha=0.6,
            color=pal["Vanilla"],
        )
        ax.scatter(
            run[_gt_col("Recall", GT)].iloc[0],
            run[_gt_col("Precision", GT)].iloc[0],
            color="#2c3e50", zorder=6, s=55, marker="s",
            label="Start" if run_id == run_ids[0] else "",
        )
        ax.scatter(
            run[_gt_col("Recall", GT)].iloc[-1],
            run[_gt_col("Precision", GT)].iloc[-1],
            color="#2c3e50", zorder=6, s=80, marker="*",
            label="End" if run_id == run_ids[0] else "",
        )

    ax.plot([], [], color=pal["Vanilla"], linewidth=2, label="Vanilla runs")

    g = df_greedy[df_greedy["Alpha"] == focus_alpha]
    if len(g):
        ax.scatter(
            g[_gt_col("Recall", GT)].mean(),
            g[_gt_col("Precision", GT)].mean(),
            color=pal.get(greedy_label, "#00b894"),
            s=250, marker=GREEDY_MARKER, zorder=7,
            edgecolors="black", linewidths=0.8,
            label=greedy_label,
        )

    ax.set_xlabel("Recall (Broad)", fontsize=12)
    ax.set_ylabel("Precision (Broad)", fontsize=12)
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.3)
    ax.annotate(
        "Ideal\n(1, 1)", xy=(1, 1), xytext=(0.78, 0.86),
        fontsize=9, color="grey",
        arrowprops=dict(arrowstyle="->", color="grey", lw=1.2),
    )
    ax.legend(fontsize=10, loc="lower left")
    ax.set_title(
        f"Recall\u2013Precision Path ({GT} GT) \u2014 Alpha = {focus_alpha}",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    # plt.show()
    if _ipy_display is not None:
        _ipy_display(fig)
    _save_fig(fig, f"recall_precision_single_alpha_{focus_alpha}", figures_dir)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 7. Feature-count trajectories
# ---------------------------------------------------------------------------

def plot_feature_count_trajectories(
    df_vanilla: pd.DataFrame,
    alphas: list,
    gt_broad: int,
    dataset: str,
    estimator: str,
    pal: dict = PAL_METHOD_DEFAULT,
    figures_dir: "str | None" = "../figures/",
) -> None:
    """Mean Confirmed / Tentative / Rejected counts over iterations for each alpha.

    Parameters
    ----------
    df_vanilla:
        All Vanilla checkpoint rows.
    alphas:
        Sorted alpha values (one subplot per alpha).
    gt_broad:
        Known ground-truth feature count (shown as a horizontal reference line).
    dataset, estimator:
        Used in the figure title.
    pal:
        Method colour palette dict (Vanilla colour used for the marker).

    Returns
    -------
    matplotlib Figure.
    """
    count_pal = {
        "N_Confirmed": "#2ecc71",
        "N_Tentative": "#3498db",
        "N_Rejected": "#e74c3c",
    }

    n_alphas = len(alphas)
    fig, axes = plt.subplots(1, n_alphas, figsize=(6 * n_alphas * FIG_SCALE, 5 * FIG_SCALE), squeeze=False)
    axes = axes.flatten()

    for idx, alpha in enumerate(alphas):
        ax = axes[idx]
        cp = df_vanilla[df_vanilla["Alpha"] == alpha]
        agg = cp.groupby("Iteration")[list(count_pal)].mean()

        for col, color in count_pal.items():
            ax.plot(
                agg.index, agg[col],
                marker=VANILLA_MARKER, markersize=4, linewidth=2,
                color=color, label=col.replace("N_", ""),
            )

        ax.axhline(
            gt_broad, color="grey", linestyle=":", linewidth=1.5,
            label=f"GT size ({gt_broad})",
        )

        ax.set_xlabel("Iteration", fontsize=11)
        ax.set_ylabel("Mean Feature Count" if idx == 0 else "", fontsize=11)
        ax.set_title(f"Alpha = {alpha}", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle(
        f"Feature State Counts over Iterations \u2014 Vanilla",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    # plt.show()
    if _ipy_display is not None:
        _ipy_display(fig)
    _save_fig(fig, "feature_count_trajectories", figures_dir)
    plt.close(fig)


def plot_feature_count_single_alpha(
    focus_alpha: float,
    df_vanilla: pd.DataFrame,
    gt_broad: int,
    dataset: str,
    estimator: str,
    figures_dir: "str | None" = "../figures/",
) -> None:
    """Single-panel feature count trajectory plot for one alpha value.

    Parameters
    ----------
    focus_alpha:
        Alpha value to focus on.
    df_vanilla:
        All Vanilla checkpoint rows.
    gt_broad:
        Known ground-truth feature count.
    dataset, estimator:
        Used in the figure title.

    Returns
    -------
    matplotlib Figure.
    """
    count_pal = {
        "N_Confirmed": "#2ecc71",
        "N_Tentative": "#3498db",
        "N_Rejected": "#e74c3c",
    }

    cp_focus = df_vanilla[df_vanilla["Alpha"] == focus_alpha]
    agg = cp_focus.groupby("Iteration")[list(count_pal)].mean()

    fig, ax = plt.subplots(figsize=(9 * FIG_SCALE, 5 * FIG_SCALE))
    for col, color in count_pal.items():
        ax.plot(
            agg.index, agg[col],
            marker=VANILLA_MARKER, markersize=4, linewidth=2,
            color=color, label=col.replace("N_", ""),
        )

    ax.axhline(
        gt_broad, color="grey", linestyle=":", linewidth=1.5,
        label=f"GT size ({gt_broad})",
    )
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Mean Feature Count", fontsize=12)
    ax.set_title(
        f"Feature State Counts over Iterations \u2014 Alpha = {focus_alpha}",
        fontsize=13, fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    # plt.show()
    if _ipy_display is not None:
        _ipy_display(fig)
    _save_fig(fig, f"feature_count_single_alpha_{focus_alpha}", figures_dir)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 8. Undecidedness & potential-miss-rate diagnostics
# ---------------------------------------------------------------------------

def plot_undecidedness_factor(
    df_vanilla: pd.DataFrame,
    alphas: list,
    dataset: str,
    estimator: str,
    x_axis: str = "iteration",
    pal: dict = PAL_METHOD_DEFAULT,
    figures_dir: "str | None" = "../figures/",
) -> None:
    """Plot Tentative / all_features ratio over iterations or time.

    Parameters
    ----------
    df_vanilla:
        All Vanilla checkpoint rows.
    alphas:
        Sorted alpha values (one subplot per alpha).
    dataset, estimator:
        Used in the figure title.
    x_axis:
        ``'iteration'`` (default) or ``'time'``.
    pal:
        Method colour palette dict.

    Returns
    -------
    matplotlib Figure.
    """
    x_col = "Elapsed_Sec" if x_axis == "time" else "Iteration"
    x_label = "Time (seconds)" if x_axis == "time" else "Iteration"

    n_alphas = len(alphas)
    fig, axes = plt.subplots(1, n_alphas, figsize=(6 * n_alphas * FIG_SCALE, 5 * FIG_SCALE), squeeze=False)
    axes = axes.flatten()

    for idx, alpha in enumerate(alphas):
        ax = axes[idx]
        cp = df_vanilla[df_vanilla["Alpha"] == alpha].copy()

        all_features = cp["N_Tentative"] + cp["N_Rejected"] + cp["N_Confirmed"]
        cp["Undecidedness_Factor"] = np.where(all_features > 0, cp["N_Tentative"] / all_features, np.nan)

        run_ids = sorted(cp["Run_ID"].unique())
        for run_id in run_ids:
            run = cp[(cp["Run_ID"] == run_id) & (cp["Checkpoint_ID"] > 0)].sort_values("Checkpoint_ID")
            ax.plot(
                run[x_col], run["Undecidedness_Factor"],
                marker=VANILLA_MARKER, markersize=5, linewidth=2, alpha=0.6,
                color=pal["Vanilla"],
                label="Vanilla run" if (idx == 0 and run_id == run_ids[0]) else "",
            )

        ax.axhline(0, color="green", linestyle="--", linewidth=1.5, alpha=0.6,
                   label="All Decided" if idx == 0 else "")
        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel("Undecidedness Factor\n(Tentative / All Features)", fontsize=11)
        ax.set_title(f"Alpha = {alpha}", fontsize=12, fontweight="bold")
        ax.set_ylim(bottom=-0.01)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=9)

    x_type = "Time" if x_axis == "time" else "Iteration"
    fig.suptitle(
        f"Undecidedness Factor vs {x_type} \u2014 Vanilla",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    # plt.show()
    if _ipy_display is not None:
        _ipy_display(fig)
    _save_fig(fig, "undecidedness_factor", figures_dir)
    plt.close(fig)


def plot_potential_miss_rate(
    df_vanilla: pd.DataFrame,
    alphas: list,
    dataset: str,
    estimator: str,
    x_axis: str = "iteration",
    pal: dict = PAL_METHOD_DEFAULT,
    figures_dir: "str | None" = "../figures/",
) -> None:
    """Plot Tentative / (Tentative + Confirmed) ratio over iterations or time.

    Parameters
    ----------
    df_vanilla:
        All Vanilla checkpoint rows.
    alphas:
        Sorted alpha values (one subplot per alpha).
    dataset, estimator:
        Used in the figure title.
    x_axis:
        ``'iteration'`` (default) or ``'time'``.
    pal:
        Method colour palette dict.

    Returns
    -------
    matplotlib Figure.
    """
    x_col = "Elapsed_Sec" if x_axis == "time" else "Iteration"
    x_label = "Time (seconds)" if x_axis == "time" else "Iteration"

    n_alphas = len(alphas)
    fig, axes = plt.subplots(1, n_alphas, figsize=(6 * n_alphas * FIG_SCALE, 5 * FIG_SCALE), squeeze=False)
    axes = axes.flatten()

    for idx, alpha in enumerate(alphas):
        ax = axes[idx]
        cp = df_vanilla[df_vanilla["Alpha"] == alpha].copy()

        denom = cp["N_Tentative"] + cp["N_Confirmed"]
        cp["Potential_Miss_Rate"] = np.where(denom > 0, cp["N_Tentative"] / denom, np.nan)

        run_ids = sorted(cp["Run_ID"].unique())
        for run_id in run_ids:
            run = cp[(cp["Run_ID"] == run_id) & (cp["Checkpoint_ID"] > 0)].sort_values("Checkpoint_ID")
            run_valid = run[~run["Potential_Miss_Rate"].isna()]
            if len(run_valid) == 0:
                continue
            ax.plot(
                run_valid[x_col], run_valid["Potential_Miss_Rate"],
                marker=VANILLA_MARKER, markersize=5, linewidth=2, alpha=0.6,
                color=pal["Vanilla"],
                label="Vanilla run" if (idx == 0 and run_id == run_ids[0]) else "",
            )

        ax.axhline(0, color="green", linestyle="--", linewidth=1.5, alpha=0.6,
                   label="All Confirmed" if idx == 0 else "")
        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel("Potential Miss Rate\n(Tentative / (Tentative + Confirmed))", fontsize=11)
        ax.set_title(f"Alpha = {alpha}", fontsize=12, fontweight="bold")
        ax.set_ylim(bottom=-0.01)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=9)

    x_type = "Time" if x_axis == "time" else "Iteration"
    fig.suptitle(
        f"Potential Miss Rate vs {x_type} \u2014 Vanilla",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    # plt.show()
    if _ipy_display is not None:
        _ipy_display(fig)
    _save_fig(fig, "potential_miss_rate", figures_dir)
    plt.close(fig)


def plot_undecidedness_single_alpha(
    focus_alpha: float,
    df_vanilla: pd.DataFrame,
    dataset: str,
    estimator: str,
    pal: dict = PAL_METHOD_DEFAULT,
    figures_dir: "str | None" = "../figures/",
) -> None:
    """Side-by-side undecidedness factor and potential miss rate for one alpha.

    Parameters
    ----------
    focus_alpha:
        Alpha value to focus on.
    df_vanilla:
        All Vanilla checkpoint rows.
    dataset, estimator:
        Used in the figure title.
    pal:
        Method colour palette dict.

    Returns
    -------
    matplotlib Figure.
    """
    cp_focus = df_vanilla[df_vanilla["Alpha"] == focus_alpha].copy()

    all_features = cp_focus["N_Tentative"] + cp_focus["N_Rejected"] + cp_focus["N_Confirmed"]
    cp_focus["Undecidedness_Factor"] = np.where(all_features > 0, cp_focus["N_Tentative"] / all_features, np.nan)

    denom = cp_focus["N_Tentative"] + cp_focus["N_Confirmed"]
    cp_focus["Potential_Miss_Rate"] = np.where(denom > 0, cp_focus["N_Tentative"] / denom, np.nan)

    run_ids = sorted(cp_focus["Run_ID"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(14 * FIG_SCALE, 5 * FIG_SCALE))

    for col_name, ax, ylabel, title in [
        (
            "Undecidedness_Factor", axes[0],
            "Undecidedness Factor\n(Tentative / All Features)", "Undecidedness Factor",
        ),
        (
            "Potential_Miss_Rate", axes[1],
            "Potential Miss Rate\n(Tentative / (Tentative + Confirmed))", "Potential Miss Rate",
        ),
    ]:
        for run_id in run_ids:
            run = cp_focus[
                (cp_focus["Run_ID"] == run_id) & (cp_focus["Checkpoint_ID"] > 0)
            ].sort_values("Checkpoint_ID")
            run_valid = run[~run[col_name].isna()]
            if len(run_valid) == 0:
                continue
            ax.plot(
                run_valid["Iteration"], run_valid[col_name],
                marker=VANILLA_MARKER, markersize=5, linewidth=2, alpha=0.3,
                color=pal["Vanilla"],
                label="Vanilla run" if run_id == run_ids[0] else "",
            )

        ax.axhline(0, color=_GREEDY_COLOR, linestyle="--", linewidth=1.5, alpha=0.7, label="Fully resolved")
        ax.set_xlabel("Iteration", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylim(bottom=-0.01)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle(
        f"Undecidedness & Potential Miss Rate \u2014 Alpha = {focus_alpha}",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    # plt.show()
    if _ipy_display is not None:
        _ipy_display(fig)
    _save_fig(fig, f"undecidedness_single_alpha_{focus_alpha}", figures_dir)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 9. F1 heatmap
# ---------------------------------------------------------------------------

def plot_f1_heatmap(
    df_vanilla_final: pd.DataFrame,
    df_greedy_final: pd.DataFrame,
    GT: str,
    greedy_label: str,
    dataset: str,
    estimator: str,
    figures_dir: "str | None" = "../figures/",
) -> None:
    """Heatmap of mean final F1 score by Method x Alpha.

    Parameters
    ----------
    df_vanilla_final, df_greedy_final:
        Final-state rows.
    GT:
        GT column suffix.
    greedy_label:
        Label for the Greedy method.
    dataset, estimator:
        Used in the figure title.

    Returns
    -------
    matplotlib Figure.
    """
    col = _gt_col("F1", GT)

    both = pd.concat(
        [
            df_vanilla_final[["Method", "Alpha", col]],
            df_greedy_final[["Method", "Alpha", col]],
        ],
        ignore_index=True,
    )
    pivot = both.groupby(["Method", "Alpha"])[col].mean().unstack("Alpha")

    fig, ax = plt.subplots(figsize=(7 * FIG_SCALE, 3 * FIG_SCALE))
    sns.heatmap(
        pivot, annot=True, fmt=".3f", cmap="YlGn",
        vmin=0.9, vmax=1.0, linewidths=0.5, ax=ax,
        cbar_kws={"label": f"F1 ({GT})"},
    )
    ax.set_title(
        f"Mean Final F1 ({GT} GT) \u2014 Method \u00d7 Alpha",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("Alpha", fontsize=11)
    ax.set_ylabel("Method", fontsize=11)
    plt.tight_layout()
    # plt.show()
    if _ipy_display is not None:
        _ipy_display(fig)
    _save_fig(fig, "f1_heatmap", figures_dir)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 10. Speed vs quality scatter
# ---------------------------------------------------------------------------

def plot_speed_vs_quality(
    df_vanilla_final: pd.DataFrame,
    df_greedy_final: pd.DataFrame,
    df_main: pd.DataFrame,
    GT: str,
    greedy_label: str,
    dataset: str,
    estimator: str,
    pal: dict = PAL_METHOD_DEFAULT,
    figures_dir: "str | None" = "../figures/",
) -> None:
    """Scatter: mean runtime vs mean final F1, point size proportional to N_Confirmed.

    Parameters
    ----------
    df_vanilla_final, df_greedy_final:
        Final-state rows.
    df_main:
        Main results DataFrame containing ``'Time_Sec'`` column.
    GT:
        GT column suffix.
    greedy_label:
        Label for the Greedy method.
    dataset, estimator:
        Used in the figure title.
    pal:
        Method colour palette dict.

    Returns
    -------
    matplotlib Figure.
    """
    col = _gt_col("F1", GT)
    runtime_cols = ["Dataset", "Method", "Alpha", "Run_ID", "Time_Sec"]
    df_rt = df_main[runtime_cols].copy()
    df_rt["Alpha"] = df_rt["Alpha"].astype(str)

    fig, ax = plt.subplots(figsize=(9 * FIG_SCALE, 6 * FIG_SCALE))

    for method_df, method_name, marker, msize in [
        (df_vanilla_final.copy(), "Vanilla", VANILLA_MARKER, 5),
        (df_greedy_final.copy(), greedy_label, GREEDY_MARKER, 5),
    ]:
        method_df = method_df.copy()
        method_df["Alpha"] = method_df["Alpha"].astype(str)
        merged = method_df.merge(
            df_rt[df_rt["Method"] == method_name],
            on=["Dataset", "Method", "Alpha", "Run_ID"],
            how="left",
        )
        agg = merged.groupby("Alpha").agg(
            F1_mean=(col, "mean"),
            Time_mean=("Time_Sec", "mean"),
            N_Conf_mean=("N_Confirmed", "mean"),
        ).reset_index()

        ax.scatter(
            agg["Time_mean"], agg["F1_mean"],
            s=agg["N_Conf_mean"] * 3 + msize,
            color=pal.get(method_name, "#95a5a6"),
            marker=marker, alpha=0.88,
            edgecolors="black", linewidths=0.7,
            label=method_name, zorder=5,
        )
        for _, row in agg.iterrows():
            ax.annotate(
                f'a={row["Alpha"]}',
                (row["Time_mean"], row["F1_mean"]),
                textcoords="offset points", xytext=(7, 4), fontsize=9,
            )

    ax.set_xlabel("Mean Runtime (seconds)", fontsize=12)
    ax.set_ylabel(f"Mean Final F1 ({GT} GT)", fontsize=12)
    ax.set_title(
        f"Speed vs GT Quality Trade-off ({GT} GT)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    # plt.show()
    if _ipy_display is not None:
        _ipy_display(fig)
    _save_fig(fig, "speed_vs_quality", figures_dir)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 11. Alpha-effect boxplots
# ---------------------------------------------------------------------------

def plot_alpha_effect_boxplots(
    df_vanilla_final: pd.DataFrame,
    df_greedy_final: pd.DataFrame,
    alphas: list,
    GT: str,
    greedy_label: str,
    dataset: str,
    estimator: str,
    metrics: list = METRICS_DEFAULT,
    pal: dict = PAL_METHOD_DEFAULT,
    df_main: "pd.DataFrame | None" = None,
    figures_dir: "str | None" = "../figures/",
) -> None:
    """Strip + point plots showing each metric's distribution across alpha values.

    When *df_main* is provided a third row is appended containing a speedup
    panel.  Speedup is computed per seed as ``time_vanilla / time_greedy``
    for matched (Alpha, Run_ID) pairs and displayed as a single series (no
    method split).

    Parameters
    ----------
    df_vanilla_final, df_greedy_final:
        Final-state rows.
    alphas:
        Sorted alpha values.
    GT, greedy_label, dataset, estimator:
        See other plot functions.
    metrics:
        Metrics to show (max 4).
    pal:
        Method colour palette dict.
    df_main:
        Optional main results DataFrame with a ``'Time_Sec'`` column.  When
        supplied the figure gains a bottom speedup row.

    Returns
    -------
    matplotlib Figure.
    """
    both_final = pd.concat(
        [
            df_vanilla_final[["Method", "Alpha"] + [_gt_col(m, GT) for m in metrics] + ["N_Confirmed"]],
            df_greedy_final[["Method", "Alpha"] + [_gt_col(m, GT) for m in metrics] + ["N_Confirmed"]],
        ],
        ignore_index=True,
    )
    both_final["Alpha"] = both_final["Alpha"].astype(str)
    alpha_order = [str(a) for a in sorted(alphas)]
    method_order = ["Vanilla", greedy_label]
    method_colors = [pal["Vanilla"], pal.get(greedy_label, "#00b894")]

    include_speedup = df_main is not None
    nrows = 3 if include_speedup else 2
    fig = plt.figure(figsize=(14 * FIG_SCALE, 5 * nrows * FIG_SCALE))

    # -- top 2×2 metric panels -----------------------------------------------
    metric_axes = []
    for i in range(4):
        ax = fig.add_subplot(nrows, 2, i + 1)
        metric_axes.append(ax)

    for i, metric in enumerate(metrics):
        ax = metric_axes[i]
        col = _gt_col(metric, GT)

        sns.stripplot(
            data=both_final, x="Alpha", y=col, hue="Method",
            hue_order=method_order,
            palette=dict(zip(method_order, method_colors)),
            order=alpha_order, ax=ax,
            dodge=True, jitter=0.15,
            size=6, alpha=0.45, linewidth=0.4, edgecolor="white",
            zorder=3, legend=False,
        )
        sns.pointplot(
            data=both_final, x="Alpha", y=col, hue="Method",
            hue_order=method_order,
            palette=dict(zip(method_order, method_colors)),
            order=alpha_order, ax=ax,
            dodge=0.4, markers=["o", "P"],
            capsize=0.1, err_kws={"linewidth": 1.5},
            linestyles="none", markersize=7, zorder=5,
        )

        ax.set_title(f"{metric}", fontsize=12, fontweight="bold")
        # ax.set_xlabel("Alpha", fontsize=10)
        ax.set_xlabel(r"$\alpha$", fontsize=10)
        ax.set_ylabel(f"", fontsize=10)
        ax.set_ylim(max(0, both_final[col].min() - 0.05), 1.05)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(title="Method", fontsize=8, title_fontsize=8)

    # -- bottom speedup row (spans both columns) ------------------------------
    if include_speedup:
        ax_su = fig.add_subplot(nrows, 1, nrows)

        rt = df_main[["Method", "Alpha", "Run_ID", "Time_Sec"]].copy()
        rt["Alpha"] = rt["Alpha"].astype(str)

        vanilla_rt = rt[rt["Method"] == "Vanilla"][["Alpha", "Run_ID", "Time_Sec"]].rename(
            columns={"Time_Sec": "Time_Vanilla"}
        )
        greedy_rt = rt[rt["Method"] == greedy_label][["Alpha", "Run_ID", "Time_Sec"]].rename(
            columns={"Time_Sec": "Time_Greedy"}
        )
        speedup_df = vanilla_rt.merge(greedy_rt, on=["Alpha", "Run_ID"], how="inner")
        speedup_df["Speedup"] = speedup_df["Time_Vanilla"] / speedup_df["Time_Greedy"]

        speedup_color = "#505252"

        sns.stripplot(
            data=speedup_df, x="Alpha", y="Speedup",
            order=alpha_order, ax=ax_su,
            jitter=0.15, size=6, alpha=0.45,
            linewidth=0.4, edgecolor="white",
            color=speedup_color, zorder=3,
        )
        sns.pointplot(
            data=speedup_df, x="Alpha", y="Speedup",
            order=alpha_order, ax=ax_su,
            color=speedup_color,
            markers="D", capsize=0.1,
            err_kws={"linewidth": 1.5},
            linestyles="none", markersize=7, zorder=5,
        )

        ax_su.axhline(1.0, color="grey", linestyle="--", linewidth=1.5, alpha=0.7,
                      label=f"{greedy_label} baseline (×1)")
        # ax_su.set_xlabel("Alpha", fontsize=11)
        ax_su.set_xlabel(r"$\alpha$", fontsize=11)
        ax_su.set_ylabel("Speedup\n(Vanilla / Greedy runtime)", fontsize=11)
        ax_su.set_title(
            f"Runtime Speedup — Vanilla vs {greedy_label} (per seed)",
            fontsize=12, fontweight="bold",
        )
        ax_su.legend(fontsize=9)
        ax_su.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"Ground-Truth Metrics by Alpha",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    # plt.show()
    if _ipy_display is not None:
        _ipy_display(fig)
    _save_fig(fig, "alpha_effect_boxplots", figures_dir)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 12. Speedup analysis
# ---------------------------------------------------------------------------

def plot_speedup_by_alpha(
    df_main: pd.DataFrame,
    alphas: list,
    greedy_label: str,
    dataset: str,
    estimator: str,
    pal: dict = PAL_METHOD_DEFAULT,
    figures_dir: "str | None" = "../figures/",
) -> None:
    """Strip + point plot of runtime speedup (Vanilla / GreedyParallel mean) by alpha.

    Parameters
    ----------
    df_main:
        Main results DataFrame with ``'Time_Sec'`` column.
    alphas:
        Sorted alpha values.
    greedy_label:
        Label for the Greedy method.
    dataset, estimator:
        Used in the figure title.
    pal:
        Method colour palette dict.

    Returns
    -------
    matplotlib Figure.
    """
    runtime_cols = ["Method", "Alpha", "Run_ID", "Time_Sec"]
    df_rt = df_main[runtime_cols].copy()
    df_rt["Alpha"] = df_rt["Alpha"].astype(str)
    alpha_order = [str(a) for a in sorted(alphas)]

    greedy_mean_time = (
        df_rt[df_rt["Method"] == greedy_label]
        .groupby("Alpha")["Time_Sec"]
        .mean()
        .rename("Greedy_Mean_Time")
    )

    vanilla_rt = df_rt[df_rt["Method"] == "Vanilla"].copy()
    vanilla_rt = vanilla_rt.join(greedy_mean_time, on="Alpha")
    vanilla_rt["Speedup"] = vanilla_rt["Time_Sec"] / vanilla_rt["Greedy_Mean_Time"]

    greedy_rt = df_rt[df_rt["Method"] == greedy_label].copy()
    greedy_rt = greedy_rt.join(greedy_mean_time, on="Alpha")
    greedy_rt["Speedup"] = greedy_rt["Time_Sec"] / greedy_rt["Greedy_Mean_Time"]

    speedup_df = pd.concat(
        [vanilla_rt[["Method", "Alpha", "Speedup"]], greedy_rt[["Method", "Alpha", "Speedup"]]],
        ignore_index=True,
    )

    method_order = ["Vanilla", greedy_label]
    method_colors = [pal["Vanilla"], pal.get(greedy_label, "#00b894")]

    fig, ax = plt.subplots(figsize=(10 * FIG_SCALE, 5 * FIG_SCALE))

    sns.stripplot(
        data=speedup_df, x="Alpha", y="Speedup", hue="Method",
        hue_order=method_order,
        palette=dict(zip(method_order, method_colors)),
        order=alpha_order, ax=ax,
        dodge=True, jitter=0.15,
        size=6, alpha=0.45, linewidth=0.4, edgecolor="white",
        zorder=3, legend=False,
    )
    sns.pointplot(
        data=speedup_df, x="Alpha", y="Speedup", hue="Method",
        hue_order=method_order,
        palette=dict(zip(method_order, method_colors)),
        order=alpha_order, ax=ax,
        dodge=0.4, markers=["o", "P"],
        capsize=0.1, err_kws={"linewidth": 1.5},
        linestyles="none", markersize=7, zorder=5,
    )

    ax.axhline(1.0, color="grey", linestyle="--", linewidth=1.5, alpha=0.7,
               label=f"{greedy_label} baseline (x1)")
    ax.set_xlabel("Alpha", fontsize=12)
    ax.set_ylabel(f"Speedup (x {greedy_label} mean runtime)", fontsize=12)
    ax.set_title(
        f"Runtime Speedup by Alpha \u2014 Vanilla vs {greedy_label}",
        fontsize=13, fontweight="bold",
    )
    ax.legend(title="Method", fontsize=10, title_fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    # plt.show()
    if _ipy_display is not None:
        _ipy_display(fig)
    _save_fig(fig, "speedup_by_alpha", figures_dir)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 13. Statistical tests
# ---------------------------------------------------------------------------

def run_statistical_tests(
    df_vanilla_final: pd.DataFrame,
    df_greedy_final: pd.DataFrame,
    alphas: list,
    GT: str,
    greedy_label: str,
    metrics: list = METRICS_DEFAULT,
    significance: float = 0.05,
) -> pd.DataFrame:
    """Welch t-tests comparing Vanilla vs Greedy final-state metrics.

    Tests are run two-sided (any difference) and one-sided (Greedy better).

    Parameters
    ----------
    df_vanilla_final, df_greedy_final:
        Final-state rows.
    alphas:
        Sorted alpha values.
    GT:
        GT column suffix.
    greedy_label:
        Label for the Greedy method.
    metrics:
        Metrics to test.
    significance:
        p-value threshold for declaring a winner (default 0.05).

    Returns
    -------
    df_tests : pd.DataFrame
        One row per (metric, alpha) with columns Metric, Alpha, Vanilla_Mean,
        Greedy_Mean, Diff (V-G), P_TwoSided, P_Greedy_gt_Vanilla,
        P_Vanilla_gt_Greedy, Winner.
    """
    def _sig(p: float) -> str:
        if p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        return "ns"

    rows = []
    for metric in metrics:
        col = _gt_col(metric, GT)
        for alpha in alphas:
            v_scores = df_vanilla_final[df_vanilla_final["Alpha"] == alpha][col].values
            g_scores = df_greedy_final[df_greedy_final["Alpha"] == alpha][col].values

            if len(v_scores) < 2 or len(g_scores) < 2:
                continue

            _, p_two = stats.ttest_ind(v_scores, g_scores, equal_var=False)
            _, p_greedy_gt = stats.ttest_ind(
                g_scores, v_scores, equal_var=False, alternative="greater"
            )
            _, p_vanilla_gt = stats.ttest_ind(
                v_scores, g_scores, equal_var=False, alternative="greater"
            )

            winner = "Tie"
            if p_greedy_gt < significance:
                winner = greedy_label
            elif p_vanilla_gt < significance:
                winner = "Vanilla"

            rows.append(
                {
                    "Metric": metric,
                    "Alpha": str(alpha),
                    "Vanilla_Mean": round(v_scores.mean(), 4),
                    "Greedy_Mean": round(g_scores.mean(), 4),
                    "Diff (V-G)": round(v_scores.mean() - g_scores.mean(), 4),
                    "P_TwoSided": round(p_two, 4),
                    "Sig_Two": _sig(p_two),
                    "P_Greedy>Vanilla": round(p_greedy_gt, 4),
                    "Sig_G>V": _sig(p_greedy_gt),
                    "P_Vanilla>Greedy": round(p_vanilla_gt, 4),
                    "Sig_V>G": _sig(p_vanilla_gt),
                    "Winner": winner,
                }
            )

    df_tests = pd.DataFrame(rows)
    print(f"Statistical comparison \u2014 {GT} GT (Welch t-test, significance = {significance})")
    print(f"Columns: P_TwoSided (any diff) | P_Greedy>Vanilla (one-sided) | P_Vanilla>Greedy (one-sided)")
    print("-" * 110)
    try:
        from IPython.display import display
        display(df_tests)
    except Exception:
        print(df_tests.to_string(index=False))
    return df_tests


# ---------------------------------------------------------------------------
# 14. Mean trajectory with confidence band
# ---------------------------------------------------------------------------

def plot_mean_trajectory_band(
    metric: str,
    df_vanilla: pd.DataFrame,
    df_greedy: pd.DataFrame,
    alphas: list,
    n_features: int,
    GT: str,
    greedy_label: str,
    dataset: str,
    estimator: str,
    x_col: str = "Iteration",
    x_label: str = "Iteration",
    pal: dict = PAL_METHOD_DEFAULT,
    figures_dir: "str | None" = "../figures/",
) -> None:
    """Mean +/- 1 std band for a metric across all Vanilla runs, one panel per alpha.

    GreedyParallel is shown as a dashed mean line with a shaded std band, and a
    vertical dotted line marks the GreedyParallel reference x-position.

    Parameters
    ----------
    metric:
        Metric name.
    df_vanilla:
        All Vanilla checkpoint rows.
    df_greedy:
        All Greedy rows.
    alphas:
        Sorted alpha values.
    n_features:
        Total feature count (for theoretical iteration count).
    GT:
        GT column suffix.
    greedy_label:
        Label for the Greedy method.
    dataset, estimator:
        Used in the figure title.
    x_col:
        ``'Iteration'`` or ``'Elapsed_Sec'``.
    x_label:
        Axis label for the x-axis.
    pal:
        Method colour palette dict.

    Returns
    -------
    matplotlib Figure.
    """
    col = _gt_col(metric, GT)
    n_alphas = len(alphas)
    fig, axes = plt.subplots(1, n_alphas, figsize=(6 * n_alphas * FIG_SCALE, 5 * FIG_SCALE), squeeze=False)
    axes = axes.flatten()

    for idx, alpha in enumerate(alphas):
        ax = axes[idx]
        cp = df_vanilla[df_vanilla["Alpha"] == alpha]

        agg = cp.groupby(x_col)[col].agg(["mean", "std"]).reset_index()
        agg["std"] = agg["std"].fillna(0)

        ax.plot(agg[x_col], agg["mean"],
                color=pal["Vanilla"], linewidth=2.5, label="Vanilla mean")
        ax.fill_between(
            agg[x_col],
            (agg["mean"] - agg["std"]).clip(lower=0),
            (agg["mean"] + agg["std"]).clip(upper=1),
            alpha=0.2, color=pal["Vanilla"],
        )

        g_vals = df_greedy[df_greedy["Alpha"] == alpha][col]
        if len(g_vals):
            gm = g_vals.mean()
            gstd = g_vals.std()
            ax.axhline(
                gm,
                color=pal.get(greedy_label, "#00b894"),
                linestyle="--", linewidth=2.2,
                label=f"{greedy_label} mean",
            )
            ax.axhspan(
                max(0, gm - gstd), min(1, gm + gstd),
                alpha=0.12, color=pal.get(greedy_label, "#00b894"),
            )

        if x_col == "Iteration":
            greedy_x = _greedy_iterations(alpha, n_features)
            vline_label = f"{greedy_label} iters ({greedy_x})"
        else:
            g_alpha = df_greedy[df_greedy["Alpha"] == alpha]
            greedy_x = g_alpha["Elapsed_Sec"].mean() if len(g_alpha) else None
            vline_label = f"{greedy_label} mean time"

        if greedy_x is not None:
            ax.axvline(
                greedy_x,
                color=pal.get(greedy_label, "#00b894"),
                linestyle=":", linewidth=1.8, alpha=0.7,
                label=vline_label,
            )

        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel(f"{metric} ({GT})", fontsize=11)
        ax.set_title(f"Alpha = {alpha}", fontsize=12, fontweight="bold")
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=9)

    fig.suptitle(
        f"Mean {metric} \u00b1 1 Std ({GT} GT) vs {x_label}",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    # plt.show()
    if _ipy_display is not None:
        _ipy_display(fig)
    _save_fig(fig, f"mean_trajectory_band_{metric}", figures_dir)
    plt.close(fig)


def plot_mean_trajectory_band_single_alpha(
    focus_alpha: float,
    df_vanilla: pd.DataFrame,
    df_greedy: pd.DataFrame,
    n_features: int,
    GT: str,
    greedy_label: str,
    dataset: str,
    estimator: str,
    metrics: list = METRICS_DEFAULT,
    pal: dict = PAL_METHOD_DEFAULT,
    figures_dir: "str | None" = "../figures/",
) -> None:
    """2x2 mean +/- band figures for a single alpha, both vs Iteration and vs Time.

    Produces two figures (one per x-axis choice).

    Parameters
    ----------
    focus_alpha:
        Alpha value to focus on.
    df_vanilla:
        All Vanilla checkpoint rows.
    df_greedy:
        All Greedy rows.
    n_features:
        Total feature count.
    GT:
        GT column suffix.
    greedy_label:
        Label for the Greedy method.
    dataset, estimator:
        Used in the figure titles.
    metrics:
        Metrics to show (max 4).
    pal:
        Method colour palette dict.
    """
    col_fn = _gt_col

    def _band_axes(ax, x_col, metric, cp_focus, g_focus, greedy_ref_x, x_label):
        col = col_fn(metric, GT)
        agg = cp_focus.groupby(x_col)[col].agg(["mean", "std"]).reset_index()
        agg["std"] = agg["std"].fillna(0)

        ax.plot(agg[x_col], agg["mean"],
                color=pal["Vanilla"], linewidth=2.5, label="Vanilla mean")
        ax.fill_between(
            agg[x_col],
            (agg["mean"] - agg["std"]).clip(lower=0),
            (agg["mean"] + agg["std"]).clip(upper=1),
            alpha=0.2, color=pal["Vanilla"],
        )

        g_vals = g_focus[col]
        if len(g_vals):
            gm, gstd = g_vals.mean(), g_vals.std()
            ax.axhline(gm, color=pal.get(greedy_label, "#00b894"),
                       linestyle="--", linewidth=2.2, label=f"{greedy_label} mean")
            ax.axhspan(max(0, gm - gstd), min(1, gm + gstd),
                       alpha=0.12, color=pal.get(greedy_label, "#00b894"))

        if greedy_ref_x is not None:
            ax.axvline(greedy_ref_x,
                       color=pal.get(greedy_label, "#00b894"),
                       linestyle=":", linewidth=1.8, alpha=0.7,
                       label=f"{greedy_label} ref")

        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel(f"{metric} ({GT})", fontsize=11)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    cp_focus = df_vanilla[df_vanilla["Alpha"] == focus_alpha]
    g_focus = df_greedy[df_greedy["Alpha"] == focus_alpha]

    for x_col, x_label, greedy_ref in [
        ("Iteration", "Iteration", _greedy_iterations(focus_alpha, n_features)),
        ("Elapsed_Sec", "Time (seconds)", g_focus["Elapsed_Sec"].mean() if len(g_focus) else None),
    ]:
        fig, axes = plt.subplots(2, 2, figsize=(14 * FIG_SCALE, 10 * FIG_SCALE))
        axes = axes.flatten()

        for i, metric in enumerate(metrics):
            _band_axes(axes[i], x_col, metric, cp_focus, g_focus, greedy_ref, x_label)

        fig.suptitle(
            f"Mean \u00b1 1 Std ({GT} GT) vs {x_label} \u2014 Alpha = {focus_alpha}",
            fontsize=14, fontweight="bold", y=1.01,
        )
        plt.tight_layout()
        if _ipy_display is not None:
            _ipy_display(fig)
        _save_fig(fig, f"mean_trajectory_band_single_alpha_{focus_alpha}_{x_col}", figures_dir)
        plt.close(fig)


# ---------------------------------------------------------------------------
# 15. Key findings printout
# ---------------------------------------------------------------------------

def print_key_findings(
    df_vanilla_final: pd.DataFrame,
    df_greedy_final: pd.DataFrame,
    df_vanilla: pd.DataFrame,
    alphas: list,
    GT: str,
    gt_broad: int,
    n_features: int,
    greedy_label: str,
    metrics: list = METRICS_DEFAULT,
) -> None:
    """Print a structured summary of final-state metrics and convergence thresholds.

    Parameters
    ----------
    df_vanilla_final, df_greedy_final:
        Final-state rows.
    df_vanilla:
        All Vanilla checkpoint rows (used for convergence analysis).
    alphas:
        Sorted alpha values.
    GT:
        GT column suffix.
    gt_broad:
        Known ground-truth feature count.
    n_features:
        Total feature count.
    greedy_label:
        Label for the Greedy method.
    metrics:
        Metrics to report.
    """
    print("=" * 80)
    print(f"KEY FINDINGS \u2014 SYNTHETIC GROUND TRUTH ANALYSIS ({GT} GT)")
    print(f"GT definition: informative + redundant features (n={gt_broad} out of {n_features})")
    print("=" * 80)

    for metric in metrics:
        col = _gt_col(metric, GT)
        print(f"\n  {metric}:")
        for alpha in alphas:
            v = df_vanilla_final[df_vanilla_final["Alpha"] == alpha][col]
            g = df_greedy_final[df_greedy_final["Alpha"] == alpha][col]
            print(
                f"    alpha={alpha:6g} | Vanilla: {v.mean():.3f} \u00b1 {v.std():.3f}  |  "
                f"{greedy_label}: {g.mean():.3f} \u00b1 {g.std():.3f}"
            )

    print(
        f"\n--- CONVERGENCE (Vanilla \u2014 iteration where mean Recall/F1 first exceeds threshold) ---"
    )
    for threshold, metric in [(0.80, "Recall"), (0.90, "Recall"), (0.90, "F1")]:
        print(f"\n  {metric} >= {threshold}:")
        for alpha in alphas:
            cp = df_vanilla[df_vanilla["Alpha"] == alpha]
            agg = cp.groupby("Iteration")[_gt_col(metric, GT)].mean()
            above = agg[agg >= threshold]
            if len(above):
                print(f"    alpha={alpha:6g} | First at iteration {above.index[0]}")
            else:
                print(f"    alpha={alpha:6g} | Never reached {threshold}")

    print("\n" + "=" * 80)


# ---------------------------------------------------------------------------
# 16. Export
# ---------------------------------------------------------------------------

def export_results(
    df_tests: pd.DataFrame,
    df_final_summary: pd.DataFrame,
    tests_path: str = "data/synthetic_gt_statistical_tests.csv",
    summary_path: str = "data/synthetic_gt_final_summary.csv",
) -> None:
    """Write statistical tests and final summary DataFrames to CSV files.

    Parameters
    ----------
    df_tests:
        Output of :func:`run_statistical_tests`.
    df_final_summary:
        Output of :func:`build_final_summary_table`.
    tests_path:
        Destination path for the tests CSV.
    summary_path:
        Destination path for the summary CSV.
    """
    df_tests.to_csv(tests_path, index=False)
    df_final_summary.to_csv(summary_path, index=False)
    print("Exported:")
    print(f"  {tests_path}")
    print(f"  {summary_path}")
