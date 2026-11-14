"""Paired, corrected resampled-t non-inferiority analysis (approximate).

Nadeau & Bengio, Inference for the Generalization Error (2003):
https://www.iro.umontreal.ca/~lisa/bib/pub_subject/language/pointeurs/nadeau_MLJ1597.pdf
The test/train correction allows for overlapping repeated-CV training sets;
it is an approximation, not an exact independence or coverage guarantee.
"""
import argparse
import numpy as np
import pandas as pd
from scipy.stats import t


def corrected_noninferiority(differences, test_train_ratios, margin, level=0.01):
    """H0: mean(candidate - baseline) <= -margin; H1: greater than -margin."""
    d = np.asarray(differences, dtype=float)
    ratios = np.asarray(test_train_ratios, dtype=float)
    if not np.isfinite(margin) or margin <= 0 or not 0 < level < 1:
        raise ValueError("Supply a positive, prespecified margin and a valid test level")
    if d.ndim != 1 or len(d) < 2 or not np.isfinite(d).all():
        raise ValueError("At least two finite paired fold differences are required")
    if ratios.shape != d.shape or not np.isfinite(ratios).all() or np.any(ratios <= 0):
        raise ValueError("One positive test/train ratio is required per paired fold")
    mean = d.mean()
    se = np.sqrt((1 / len(d) + ratios.mean()) * d.var(ddof=1))
    lower = mean - t.ppf(1 - level, len(d) - 1) * se
    pvalue = t.sf((mean + margin) / se, len(d) - 1) if se else (0.0 if mean > -margin else 1.0)
    return dict(Mean_Difference=mean, Margin=margin, Corrected_SE=se,
                Lower_One_Sided_CI=lower, CI_Level=1-level, P_NI=float(pvalue),
                Noninferior_Unadjusted=bool(lower > -margin), N_Pairs=len(d),
                Zero_Variance=bool(se == 0))


def noninferiority_table(frame, margins, level=0.01, baseline="Vanilla", baseline_policy="rough_fix"):
    """Pair exact split IDs within each dataset realization; Holm across returned tests.

    Margins map task names ('class', 'reg') to absolute F1/R² losses. No default
    is inferred from scores. Per-realization tests are not population-level tests
    over newly generated datasets. Missing/mismatched pairs raise, not disappear.
    """
    groups = ['Dataset', 'Data_Seed', 'Task', 'Alpha', 'Estimator']
    pairs = ['Repeat', 'Fold', 'Split_ID', 'N_Train', 'N_Test']
    rows = []
    for values, group in frame.groupby(groups, dropna=False):
        meta = dict(zip(groups, values))
        task = meta['Task']
        if task not in margins or margins[task] is None:
            raise ValueError(f"Specify a non-inferiority margin for task {task}")
        reference = group[(group.Method == baseline) & (group.Policy == baseline_policy)]
        if reference.empty:
            raise ValueError(f"Missing baseline for {meta}")
        for (method, policy), candidate in group.groupby(['Method', 'Policy']):
            if method == baseline:
                continue
            joined = candidate.merge(reference, on=pairs, suffixes=('_candidate', '_baseline'),
                                     how='outer', indicator=True, validate='one_to_one')
            if not (joined['_merge'] == 'both').all():
                raise ValueError(f"Unmatched evaluation splits for {method}, {meta}")
            result = corrected_noninferiority(
                joined.Score_candidate - joined.Score_baseline,
                joined.N_Test / joined.N_Train, margins[task], level)
            rows.append(dict(meta, Method=method, Policy=policy, Baseline=baseline,
                             Baseline_Policy=baseline_policy, **result))
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    order = np.argsort(result.P_NI.to_numpy())
    adjusted = np.maximum.accumulate((len(order) - np.arange(len(order))) * result.P_NI.to_numpy()[order])
    result['P_NI_Holm'] = np.nan
    result.loc[order, 'P_NI_Holm'] = np.minimum(1, adjusted)
    result['Noninferior_Holm'] = result.P_NI_Holm < level
    result['Decision'] = np.where(result.Noninferior_Holm, 'Non-inferiority established',
                                  'Non-inferiority not established')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('fold_results')
    parser.add_argument('--f1-margin', type=float)
    parser.add_argument('--r2-margin', type=float)
    parser.add_argument('--level', type=float, default=0.01)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = noninferiority_table(pd.read_json(args.fold_results, lines=True),
                                 {'class': args.f1_margin, 'reg': args.r2_margin}, args.level)
    result.to_csv(args.output, index=False)
