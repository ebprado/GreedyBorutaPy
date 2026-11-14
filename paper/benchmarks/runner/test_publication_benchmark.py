import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import ExtraTreesClassifier
from scipy.stats import t

from GreedyBorutaParallel import GreedyBorutaPyParallel
from BorutaLocal import BorutaPy
from synthetic_suite import synthetic_scenarios, generate_synthetic
from experiment_metrics import recovery_metrics, selection_stability, PeakRSS, CPUTime
from noninferiority import corrected_noninferiority, noninferiority_table
from publication_benchmark import preprocess_fold, method_specs, run_repeat


def selector(**kwargs):
    return GreedyBorutaPyParallel(ExtraTreesClassifier(max_depth=3, n_jobs=1),
                                 n_estimators=8, alpha=.1, random_state=42, **kwargs)


@pytest.mark.parametrize('value, expected', [(None, 12), (1, 1), (4, 4), (.25, 3), (.01, 1), (1., 12)])
def test_shadow_counts(value, expected):
    assert clone(selector(n_shadows=value))._resolve_n_shadows(12) == expected


@pytest.mark.parametrize('value', [0, -1, 13, 1.5, np.nan, True, 'auto'])
def test_invalid_shadow_counts(value):
    with pytest.raises(ValueError):
        selector(n_shadows=value)._resolve_n_shadows(12)


def test_serial_parallel_and_repeated_fit_match():
    rng = np.random.RandomState(5)
    X = rng.normal(size=(60, 12))
    y = (X[:, 0] > 0).astype(int)
    serial, parallel = selector(n_shadows=.25), selector(n_shadows=.25, n_jobs=2)
    serial.estimator.set_params(n_jobs=2)  # Equal total budget: 1x2 versus 2x1.
    serial.fit(X, y)
    parallel.fit(X, y)
    np.testing.assert_array_equal(serial.hit_counts_, parallel.hit_counts_)
    previous = serial.hit_counts_.copy()
    serial.fit(X, y)
    np.testing.assert_array_equal(serial.hit_counts_, previous)
    assert serial.n_shadows_ == 3
    assert serial.n_iter_ == selector().fit(X, y).n_iter_
    np.testing.assert_array_equal(selector().fit(X, y).hit_counts_, selector(n_shadows=12).fit(X, y).hit_counts_)


class InspectEstimator(BaseEstimator):
    def __init__(self, random_state=None):
        self.random_state = random_state

    def fit(self, X, y):
        self.seen = X.copy()
        self.feature_importances_ = np.ones(X.shape[1])
        return self


def test_exact_shadow_width_and_permutations():
    X = np.arange(100).reshape(20, 5)
    original = X.copy()
    estimator = InspectEstimator()
    hits = GreedyBorutaPyParallel._parallel_fit_iteration(estimator, X, np.zeros(20), 5, 100, False, 2)
    assert estimator.seen.shape == (20, 7)
    np.testing.assert_array_equal(estimator.seen[:, :5], X)
    np.testing.assert_array_equal(X, original)
    sources = []
    for shadow in estimator.seen[:, 5:].T:
        sources.append(next(j for j in range(5) if np.array_equal(np.sort(shadow), X[:, j])))
    assert len(set(sources)) == 2
    assert not hits.any()  # Equal importances are not strict hits.


def test_boundary_rejection_matches_ceiling():
    greedy = selector()
    assert greedy._compute_iterations_to_convergence(.0625, 16) == 8
    vanilla = BorutaPy(ExtraTreesClassifier(), alpha=.0625, two_step=False)
    decisions = vanilla._do_tests(np.zeros(16, dtype=int), np.zeros(16, dtype=int), 8)
    assert np.all(decisions == -1)


def test_synthetic_suite_truth_and_reproducibility():
    for name, spec in synthetic_scenarios().items():
        spec = dict(spec, n_samples=60)
        X, y, truth = generate_synthetic(spec, 5)
        X2, y2, truth2 = generate_synthetic(spec, 5)
        np.testing.assert_array_equal(X, X2)
        np.testing.assert_array_equal(y, y2)
        assert truth == truth2 and max(truth, default=0) < X.shape[1]
        if spec['kind'] == 'null':
            assert not truth
        if spec['kind'] == 'xor':
            np.testing.assert_array_equal(y, np.logical_xor(X[:, 0], X[:, 1]))
        assert not np.array_equal(X, generate_synthetic(spec, 6)[0])


def test_fold_preprocessing_does_not_learn_validation_values():
    X = pd.DataFrame({'cat': ['a', 'b', 'unseen'], 'num': [1., 3., np.nan],
                      'empty_train': [np.nan, np.nan, 99.]})
    train, test = preprocess_fold(X, np.array([0, 1]), np.array([2]))
    assert train.shape == (2, 3) and test.shape == (1, 3)
    assert test[0, 0] == -1 and test[0, 1] == 2 and test[0, 2] == 99


def test_metrics_and_cpu_allocation():
    metrics = recovery_metrics({0, 1}, {1, 2}, 5)
    assert metrics['Recall'] == metrics['Precision'] == .5
    assert metrics['FP_Original_Fraction'] == .2
    assert np.isnan(recovery_metrics(set(), set(), 5)['Recall'])
    jaccard, frequency = selection_stability([{0, 1}, {1, 2}], 4)
    assert jaccard == 1/3
    np.testing.assert_array_equal(frequency, [.5, 1, .5, 0])
    for method in method_specs(2, 2, [1., .25]):
        assert method['outer'] * method['inner'] == 2
    with pytest.raises(ValueError):
        method_specs(3, 2, [1.])
    with PeakRSS() as memory:
        values = np.ones(1000)
    assert memory.as_dict()['RSS_Peak_MB'] > 0
    with CPUTime() as cpu:
        sum(i * i for i in range(10**6))
    assert cpu.seconds > 0


def test_corrected_ni_formula_direction_and_margin():
    differences = np.array([-.002, .001, -.001, 0])
    result = corrected_noninferiority(differences, np.repeat(.25, 4), .01)
    expected_se = np.sqrt((1/4 + .25) * differences.var(ddof=1))
    assert result['Corrected_SE'] == pytest.approx(expected_se)
    assert result['P_NI'] == pytest.approx(t.sf((differences.mean() + .01) / expected_se, 3))
    assert result['Noninferior_Unadjusted']
    assert not corrected_noninferiority(differences - .1, np.repeat(.25, 4), .01)['Noninferior_Unadjusted']
    with pytest.raises(ValueError):
        corrected_noninferiority(differences, np.repeat(.25, 4), 0)


def test_ni_pairs_and_rejects_mismatches():
    rows = []
    for i in range(4):
        for method, policy, score in [('Vanilla', 'rough_fix', .8), ('Greedy', 'selected', .8 + i*.001)]:
            rows.append(dict(Dataset='test', Data_Seed=0, Task='class', Alpha=.1, Estimator='extra_trees',
                             Repeat=i//2, Fold=i%2, Split_ID=str(i), N_Train=40, N_Test=10,
                             Method=method, Policy=policy, Score=score))
    frame = pd.DataFrame(rows)
    assert len(noninferiority_table(frame, {'class': .01})) == 1
    with pytest.raises(ValueError, match='Unmatched'):
        noninferiority_table(frame.iloc[:-1], {'class': .01})
    with pytest.raises(ValueError, match='margin'):
        noninferiority_table(frame, {'class': None})


def test_regression_end_to_end():
    rng = np.random.RandomState(7)
    X = rng.normal(size=(40, 6))
    y = X[:, 0] + rng.normal(scale=.1, size=40)
    meta = dict(Dataset='regression_smoke', Data_Seed=0, Task='reg', Alpha=.1, Estimator='extra_trees')
    rows = list(run_repeat(X, y, {0}, meta, method_specs(1, 1, [1., .25]),
                           0, folds=2, cpu_budget=1, trees=5, vanilla_iterations=8))
    assert len(rows) == 10  # Two folds; Vanilla has two selection policies.
    assert all(np.isfinite(row['Score']) for row in rows)
    assert all(row['N_Selected'] == 6 for row in rows if row['Method'] == 'NoSelection')
