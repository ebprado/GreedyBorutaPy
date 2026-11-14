"""Expanded, fold-local evaluation. See PUBLICATION_BENCHMARK.md for usage."""
import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path
from importlib.metadata import version

import numpy as np
import pandas as pd
import sklearn
from joblib import parallel_config
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, r2_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder
from threadpoolctl import threadpool_limits
import threadpoolctl

from BorutaLocal import BorutaPy
from GreedyBorutaParallel import GreedyBorutaPyParallel
from experiment_metrics import PeakRSS, recovery_metrics, selection_stability
from synthetic_suite import generate_synthetic, synthetic_scenarios
from noninferiority import noninferiority_table


REAL_DATASETS = {
    'Madelon': (1485, 'class'), 'Arcene': (1458, 'class'),
    'Gisette': (41026, 'class'), 'Dexter': (4136, 'class'),
    'Riboflavin': (46983, 'reg'), 'Isolet': (300, 'class'),
}


def method_specs(cpu_budget, outer_jobs, shadow_fractions, serial_fractions=None):
    """serial_fractions limits the outer=1 Greedy runs (None: every shadow fraction).

    Serial and parallel Greedy select identical masks from identical seeds, so the
    serial runs only measure parallel speedup and need not cover every fraction.
    """
    if cpu_budget < 1 or outer_jobs < 1 or cpu_budget % outer_jobs:
        raise ValueError('CPU budget must be positive and divisible by outer_jobs')
    if cpu_budget > (os.cpu_count() or 1):
        raise ValueError('CPU budget exceeds available logical CPUs')
    if serial_fractions is not None and not set(serial_fractions) <= set(shadow_fractions):
        raise ValueError('Serial shadow fractions must be a subset of the shadow fractions')
    methods = [dict(name='NoSelection', outer=1, inner=cpu_budget, shadows=None),
               dict(name='Vanilla', outer=1, inner=cpu_budget, shadows=None)]
    for outer in sorted({1, outer_jobs}):
        for fraction in shadow_fractions:
            if outer == 1 < outer_jobs and serial_fractions is not None and fraction not in serial_fractions:
                continue
            if not 0 < fraction <= 1:
                raise ValueError('Shadow fractions must be in (0, 1]')
            methods.append(dict(name=f'Greedy_jobs{outer}_shadows{fraction:g}',
                                outer=outer, inner=cpu_budget // outer, shadows=float(fraction)))
    return methods


def preprocess_fold(X, train, test):
    """Learn all preprocessing on train; preserve one output per input column."""
    if isinstance(X, pd.DataFrame):
        transformers = []
        numeric = [j for j, dtype in enumerate(X.dtypes) if pd.api.types.is_numeric_dtype(dtype)]
        numeric_set = set(numeric)
        categorical = [j for j in range(X.shape[1]) if j not in numeric_set]
        if numeric:
            transformers.append(('numeric', SimpleImputer(strategy='median', keep_empty_features=True), numeric))
        if categorical:
            transform = make_pipeline(
                SimpleImputer(strategy='most_frequent', keep_empty_features=True),
                OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
            transformers.append(('categorical', transform, categorical))
        prep = ColumnTransformer(transformers)
        restore_order = np.argsort(numeric + categorical)
        return (prep.fit_transform(X.iloc[train])[:, restore_order],
                prep.transform(X.iloc[test])[:, restore_order])
    prep = SimpleImputer(strategy='median', keep_empty_features=True)
    return prep.fit_transform(X[train]), prep.transform(X[test])


def fit_selection(X, y, method, alpha, seed, trees, vanilla_iterations):
    if method['name'] == 'NoSelection':
        return {'selected': np.ones(X.shape[1], dtype=bool)}, 0, 0
    estimator_cls = ExtraTreesClassifier if method['task'] == 'class' else ExtraTreesRegressor
    estimator = estimator_cls(max_depth=5, n_jobs=method['inner'], random_state=seed)
    if method['name'] == 'Vanilla':
        selector = BorutaPy(estimator, n_estimators=trees, alpha=alpha,
                            two_step=False, max_iter=vanilla_iterations + 1,
                            early_stopping=True, n_iter_no_change=128,
                            random_state=seed, track_history=False)
    else:
        selector = GreedyBorutaPyParallel(estimator, n_estimators=trees, alpha=alpha,
                                         n_jobs=method['outer'], n_shadows=method['shadows'],
                                         random_state=seed)
    selector.fit(X, y)
    if method['name'] == 'Vanilla':
        masks = {'strict': selector.support_,
                 'rough_fix': selector.support_ | selector.support_weak_}
        # Boruta's loop counter is incremented after fitting; history rows are exact.
        iterations = selector.importance_history_.shape[0] - 1
        return masks, iterations, np.nan  # Shadow count changes as features are dropped.
    return {'selected': selector.support_}, selector.n_iter_, selector.n_shadows_


def score_mask(X_train, y_train, X_test, y_test, mask, task, seed, cpu_budget):
    if not mask.any():
        # A no-feature method still makes a valid baseline prediction, including on null data.
        model = DummyClassifier(strategy='prior') if task == 'class' else DummyRegressor()
        train, test = np.zeros((len(y_train), 1)), np.zeros((len(y_test), 1))
    else:
        cls = LGBMClassifier if task == 'class' else LGBMRegressor
        model = cls(n_estimators=256, max_depth=5, learning_rate=0.1,
                    n_jobs=cpu_budget, random_state=seed, verbose=-1)
        train, test = X_train[:, mask], X_test[:, mask]
    model.fit(train, y_train)
    prediction = model.predict(test)
    return (f1_score(y_test, prediction, average='weighted', zero_division=0)
            if task == 'class' else r2_score(y_test, prediction))


def run_repeat(X, y, relevant, metadata, methods, repeat, folds=5,
               cpu_budget=4, trees='auto', vanilla_iterations=512):
    cv_cls = StratifiedKFold if metadata['Task'] == 'class' else KFold
    cv = cv_cls(n_splits=folds, shuffle=True, random_state=repeat)
    for fold, (train, test) in enumerate(cv.split(X, y)):
        X_train, X_test = preprocess_fold(X, train, test)
        split_id = hashlib.sha256(train.tobytes() + test.tobytes()).hexdigest()
        # Rotate execution order reproducibly to reduce systematic warm-up/order bias.
        order = np.random.RandomState(repeat * folds + fold).permutation(len(methods))
        for position in order:
            method = dict(methods[position], task=metadata['Task'])
            seed = repeat * folds + fold
            # ExtraTrees uses inner joblib threads; keep nested native pools single-threaded.
            with threadpool_limits(limits=1), parallel_config(
                    backend='loky', inner_max_num_threads=1):
                with PeakRSS() as memory:
                    start = time.perf_counter()
                    masks, iterations, shadows = fit_selection(
                        X_train, y[train], method, metadata['Alpha'], seed, trees, vanilla_iterations)
                    elapsed = time.perf_counter() - start
            for policy, mask in masks.items():
                selected = np.flatnonzero(mask).tolist()
                with threadpool_limits(limits=cpu_budget), PeakRSS() as downstream_memory:
                    downstream_start = time.perf_counter()
                    score = score_mask(X_train, y[train], X_test, y[test], mask,
                                       metadata['Task'], seed, cpu_budget)
                    downstream_time = time.perf_counter() - downstream_start
                row = dict(metadata, Repeat=repeat, Fold=fold, Split_ID=split_id,
                           Method=method['name'], Policy=policy, Score=score,
                           N_Train=len(train), N_Test=len(test), N_Features=X.shape[1],
                           N_Selected=len(selected), Selected_Features=selected,
                           Selection_Time_Sec=elapsed, Downstream_Time_Sec=downstream_time,
                           Iterations=iterations, N_Shadows=shadows,
                           CPU_Budget=cpu_budget, Outer_Jobs=method['outer'],
                           Estimator_Jobs=method['inner'], **memory.as_dict())
                row.update({f'Downstream_{k}': v for k, v in downstream_memory.as_dict().items()})
                if relevant is not None:
                    row.update(recovery_metrics(selected, relevant, X.shape[1]))
                yield row


def summarize_stability(frame):
    rows, frequencies = [], []
    keys = ['Dataset', 'Data_Seed', 'Task', 'Alpha', 'Estimator', 'Method', 'Policy']
    for values, group in frame.groupby(keys, dropna=False):
        metadata = dict(zip(keys, values))
        score, frequency = selection_stability(group.Selected_Features, int(group.N_Features.iloc[0]))
        rows.append(dict(metadata, Mean_Pairwise_Jaccard=score, N_Fits=len(group),
                         Mean_Selected=group.N_Selected.mean()))
        frequencies.extend(dict(metadata, Feature=j, Selection_Frequency=float(value))
                           for j, value in enumerate(frequency))
    return pd.DataFrame(rows), pd.DataFrame(frequencies)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=['synthetic', 'real', 'both'], default='synthetic')
    parser.add_argument('--datasets', nargs='+', help='Optional scenario or real dataset names')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'data' / 'publication_v5')
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--data-seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--alphas', type=float, nargs='+', default=[0.1, 0.001, 0.00001])
    parser.add_argument('--cpu-budget', type=int, default=4)
    parser.add_argument('--outer-jobs', type=int, default=4)
    parser.add_argument('--shadow-fractions', type=float, nargs='+', default=[1.0, 0.25, 0.1])
    parser.add_argument('--f1-margin', type=float)
    parser.add_argument('--r2-margin', type=float)
    parser.add_argument('--test-level', type=float, default=0.01)
    parser.add_argument('--smoke', action='store_true', help='Tiny synthetic execution, not publication evidence')
    args = parser.parse_args()
    if tuple(int(part) for part in threadpoolctl.__version__.split('.')[:2]) < (3, 6):
        parser.error('threadpoolctl >= 3.6 is required; install runner/requirements-publication.txt')
    if args.repeats < 1 or any(not 0 < a < 1 for a in args.alphas):
        parser.error('Require positive repeats and alpha in (0, 1)')
    methods = method_specs(args.cpu_budget, args.outer_jobs, args.shadow_fractions)
    scenarios = synthetic_scenarios()
    if args.smoke:
        args.suite, args.repeats, args.data_seeds, args.alphas = 'synthetic', 2, [0], [0.1]
        scenarios = {key: dict(scenarios[key], n_samples=80, n_features=12,
                              n_correlated=0) for key in ['reference', 'kind_null', 'kind_xor']}
    available = {}
    if args.suite in ('synthetic', 'both'):
        available.update({key: ('synthetic', value) for key, value in scenarios.items()})
    if args.suite in ('real', 'both'):
        available.update({key: ('real', value) for key, value in REAL_DATASETS.items()})
    if args.datasets:
        unknown = set(args.datasets) - set(available)
        if unknown:
            parser.error(f'Unknown dataset names: {sorted(unknown)}')
        available = {key: available[key] for key in args.datasets}
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / 'fold_results.jsonl'
    if target.exists():
        parser.error(f'{target} exists; choose a new output directory to avoid mixing protocols')
    manifest = dict(vars(args), output=str(args.output), methods=methods, datasets=available,
                    python=platform.python_version(), numpy=np.__version__, sklearn=sklearn.__version__,
                    packages={name: version(name) for name in ['scipy', 'pandas', 'lightgbm', 'joblib', 'psutil']},
                    threadpoolctl=threadpoolctl.__version__,
                    protocol='fold-local preprocessing and selection; paired repeated CV; v5',
                    ni_method='approximate corrected resampled t; Holm over all reported tests')
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    with target.open('w', encoding='utf-8') as output:
        for name, (kind, spec) in available.items():
            seeds = args.data_seeds if kind == 'synthetic' else [0]
            for data_seed in seeds:
                if kind == 'synthetic':
                    X, y, relevant = generate_synthetic(spec, data_seed)
                    task = 'class'
                else:
                    from data_loader import load_openml_dataset
                    X, y = load_openml_dataset(spec[0], spec[1], preprocess=False)
                    relevant, task = None, spec[1]
                for alpha in args.alphas:
                    metadata = dict(Dataset=name, Data_Seed=data_seed, Task=task,
                                    Alpha=alpha, Estimator='extra_trees')
                    for repeat in range(args.repeats):
                        print(f'{name} data_seed={data_seed} alpha={alpha} repeat={repeat}', flush=True)
                        for row in run_repeat(X, y, relevant, metadata, methods, repeat,
                                              folds=2 if args.smoke else 5, cpu_budget=args.cpu_budget,
                                              trees=8 if args.smoke else 'auto',
                                              vanilla_iterations=12 if args.smoke else 512):
                            clean = {k: (None if isinstance(v, (float, np.floating)) and not np.isfinite(v) else v)
                                     for k, v in row.items()}
                            output.write(json.dumps(clean) + '\n')
                            output.flush()
    frame = pd.read_json(target, lines=True)
    stability, frequency = summarize_stability(frame)
    stability.to_csv(args.output / 'stability.csv', index=False)
    frequency.to_csv(args.output / 'selection_frequency.csv', index=False)
    margins = {'class': args.f1_margin, 'reg': args.r2_margin}
    if all(margins[task] is not None for task in frame.Task.unique()):
        noninferiority_table(frame, margins, args.test_level).to_csv(args.output / 'noninferiority.csv', index=False)
    else:
        print('Non-inferiority not calculated: supply prespecified margins for each task type.')
    print(f'Results: {args.output}')


if __name__ == '__main__':
    main()
