"""Regression checks for pairing, incomplete runs, NI, and reused computation."""
import copy
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import analysis_helpers as ah


@pytest.fixture
def run():
    methods = [dict(name="NoSelection", outer=1, inner=4),
               dict(name="Vanilla", outer=1, inner=4),
               dict(name="Greedy_jobs1_shadows1", outer=1, inner=4),
               dict(name="Greedy_jobs4_shadows1", outer=4, inner=1)]
    manifest = dict(config=dict(datasets={"toy": ["real", [1, "class"]]},
                    methods=methods, data_seeds=[0], repeats=2, folds=2,
                    alphas=[.1, .01], cpu_budget=4, test_level=.01,
                    f1_margin=.01, r2_margin=.01, smoke=False))
    rows = []
    for group in ah.expected_groups(manifest):
        for repeat in range(2):
            for fold in range(2):
                method = next(m for m in methods if m["name"] == group["Method"])
                vanilla = group["Method"] == "Vanilla"
                baseline = group["Method"] == "NoSelection"
                selected = [0, 1, 2] if baseline else ([0] if vanilla else [0, 1])
                mask = np.zeros(3, dtype=np.uint8)
                mask[selected] = 1
                partition = f"p{repeat}{fold}"
                job = f"{group['Method']}:{None if baseline else group['Alpha']}"
                performed = (baseline and group["Alpha"] == .1) or (
                    not baseline and (not vanilla or group["Policy"] == "strict"))
                rows.append(dict(group, Task="class", Estimator="extra_trees", Repeat=repeat,
                    Fold=fold, Split_ID=f"s{repeat}{fold}", Partition_ID=partition,
                    N_Train=80, N_Test=20, Job_ID=job, Score=.8 + (.001 * (repeat + fold) if not vanilla and not baseline else 0),
                    N_Features=3, N_Selected=len(selected), Selected_Features=selected,
                    Mask_ID=hashlib.sha256(mask.tobytes()).hexdigest(), Iterations=20,
                    Selection_Time_Sec=.001 if baseline else (10 if vanilla else 2),
                    Downstream_Actual_Time_Sec=1. if performed else 0.,
                    Downstream_Fit_Performed=performed, Downstream_Source=f"{partition}/{job}",
                    CPU_Budget=4, Outer_Jobs=method["outer"], Estimator_Jobs=method["inner"],
                    Selection_Backend="threading" if method["outer"] == 1 else "loky_outer_threading_inner",
                    Label=ah.LABELS[(group["Method"], group["Policy"])], N_Shadows=3))
    frame = pd.DataFrame(rows)
    return ah.RunData(Path("toy"), manifest, {}, frame, ah.coverage_table(frame, manifest), {})


def test_validation_and_duplicate_rejection(run):
    ah.validate_frame(run.frame, run.manifest)
    with pytest.raises(ValueError, match="Duplicate"):
        ah.validate_frame(pd.concat([run.frame, run.frame.iloc[[0]]]), run.manifest)


def test_split_mismatch_rejected(run):
    bad = run.frame.copy()
    bad.loc[0, "Split_ID"] = "wrong"
    with pytest.raises(ValueError, match="split identities"):
        ah.validate_frame(bad, run.manifest)


def test_invalid_mask_rejected(run):
    bad = run.frame.copy()
    bad.loc[0, "Mask_ID"] = "bad"
    with pytest.raises(ValueError, match="mask hash"):
        ah.validate_frame(bad, run.manifest)


def test_partial_block_excluded_and_decisions_withheld(run):
    run.frame = run.frame.drop(index=0)
    run.coverage = ah.coverage_table(run.frame, run.manifest)
    complete = ah.complete_blocks(run)
    assert set(complete.Alpha) == {.01}
    ni = ah.noninferiority(ah.paired_comparisons(complete), run)
    assert ni.Decision.str.startswith("Withheld").all()
    assert ni.Planned_Family_Size.eq(6).all()
    with pytest.raises(ValueError, match="Unmatched"):
        ah.paired_comparisons(run.frame)


def test_paired_speedup_and_no_selection_exclusion(run):
    pair = ah.paired_comparisons(run.frame)
    assert pair.loc[pair.Method.str.startswith("Greedy"), "Speedup"].eq(5).all()
    assert pair.loc[pair.Method.eq("NoSelection"), "Speedup"].isna().all()
    assert ah.parallel_ablation(run.frame).Identical_Mask.all()


def test_parallel_ablation_pairs_only_serial_fractions(run):
    reduced = run.frame[run.frame.Method.eq("Greedy_jobs4_shadows1")].assign(
        Method="Greedy_jobs4_shadows0.25", N_Shadows=1,
        Label=ah.LABELS[("Greedy_jobs4_shadows0.25", "selected")])
    frame = pd.concat([run.frame, reduced], ignore_index=True)
    assert set(ah.parallel_ablation(frame).Method) == {"Greedy_jobs4_shadows1"}
    with pytest.raises(ValueError, match="Unmatched"):
        ah.parallel_ablation(frame.drop(index=frame.index[frame.Method.eq("Greedy_jobs1_shadows1")][0]))


def test_ni_agrees_with_runner_and_full_family(run):
    path = Path(__file__).resolve().parents[2] / "runner" / "noninferiority.py"
    spec = importlib.util.spec_from_file_location("runner_ni", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = module.noninferiority_table(run.frame, {"class": .01}, level=.01)
    actual = ah.noninferiority(ah.paired_comparisons(run.frame), run)
    joined = expected.merge(actual, on=ah.GROUP, suffixes=("_expected", "_actual"))
    for col in ["Mean_Difference", "Corrected_SE", "Lower_One_Sided_CI", "P_NI", "P_NI_Holm"]:
        np.testing.assert_allclose(joined[col + "_actual"], joined[col + "_expected"])
    assert actual.Analysis_Status.eq("Final").all()
    run.manifest["config"]["smoke"] = True
    assert ah.noninferiority(ah.paired_comparisons(run.frame), run).Decision.str.startswith("Withheld").all()


def test_holm_reserves_missing_tests():
    np.testing.assert_allclose(ah.holm_adjust([.01, .03], 4), [.04, .09])


def test_unique_cost_accounting(run):
    table = ah.resource_accounting(run.frame).set_index("Quantity").Value
    assert table["Completed method jobs (includes baseline)"] == 28
    assert table["Selector fits (Vanilla policies counted once)"] == 24
    assert table["Selector fit time"] == pytest.approx(112 / 3600)
    assert table["Actual downstream fit time"] == pytest.approx(28 / 3600)


def test_missing_dataset_coverage(run):
    manifest = copy.deepcopy(run.manifest)
    manifest["config"]["datasets"]["absent"] = ["real", [2, "class"]]
    coverage = ah.coverage_table(run.frame, manifest)
    assert coverage.loc[coverage.Dataset.eq("absent"), "Observed"].eq(0).all()


def test_empty_set_stability_and_missing_metric_counts(run):
    frame = run.frame.iloc[:4].copy()
    frame["Selected_Features"] = [[] for _ in range(len(frame))]
    frame["N_Selected"] = 0
    frame["Recall"] = np.nan
    frame["Precision"] = np.nan
    stability, _ = ah.stability_tables(frame, run.manifest)
    assert stability.Mean_Pairwise_Jaccard.eq(1).all()
    summary = ah.summary_table(frame)
    assert summary.Recall_count.eq(0).all()
    assert summary.Recall_mean.isna().all()


def with_coupled(run, folds=None):
    """Add an augmented GreedyCoupled method whose masks nest over Vanilla's."""
    manifest = copy.deepcopy(run.manifest)
    manifest["augmentations"] = [dict(version="test", methods=[dict(name=ah.COUPLED, outer=1, inner=4,
                                                                     shadows=1.0, policy="selected")])]
    base = run.frame[run.frame.Method.eq("Greedy_jobs1_shadows1")].copy()
    if folds is not None:
        base = base[base.Fold.isin(folds)]
    base = base.assign(Method=ah.COUPLED, Label="Greedy coupled", Job_ID=lambda f: ah.COUPLED + ":" + f.Alpha.astype(str),
                       Strong_Features=[[0]] * len(base), Downstream_Fit_Performed=False,
                       Downstream_Actual_Time_Sec=0.)
    frame = pd.concat([run.frame, base], ignore_index=True)
    return ah.RunData(Path("toy"), manifest, {}, frame, ah.coverage_table(frame, manifest), {})


def test_augmented_method_validates_and_stays_out_of_ni_family(run):
    augmented = with_coupled(run)
    ah.validate_frame(augmented.frame, augmented.manifest)
    ni = ah.noninferiority(ah.paired_comparisons(ah.complete_blocks(augmented)), augmented)
    assert ah.COUPLED not in set(ni.Method)
    assert ni.Planned_Family_Size.eq(6).all()
    assert {"Lower_Bound_97.5", "Decision_Level_0.025"} <= set(ni.columns)
    assert (ni["Lower_Bound_97.5"] >= ni.Lower_One_Sided_CI).all()


def test_incomplete_augmentation_never_drops_prespecified_blocks(run):
    augmented = with_coupled(run, folds=[0])
    assert augmented.complete
    complete = ah.complete_blocks(augmented)
    assert ah.COUPLED not in set(complete.Method)
    assert len(complete) == len(run.frame)


def test_coupled_sets_nesting(run):
    augmented = with_coupled(run)
    sets = ah.coupled_sets(augmented.frame, augmented.manifest)
    assert sets[ah.FATE_LAYERS].sum(axis=1).eq(sets.N_Greedy).all()
    # Strong {0} = Vanilla strict = rough fix {0}; Greedy {0, 1} adds one Greedy-only feature.
    assert sets[ah.FATE_LAYERS].drop_duplicates().values.tolist() == [[1, 0, 0, 1]]
    bad = augmented.frame.copy()
    bad.loc[bad.Method.eq(ah.COUPLED), "Strong_Features"] = pd.Series([[1]] * len(bad), index=bad.index)
    with pytest.raises(ValueError, match="nesting"):
        ah.coupled_sets(bad, augmented.manifest)
