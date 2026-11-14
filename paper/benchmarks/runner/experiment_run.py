"""Fresh, resumable v6 experiment runner; never reads earlier experiment results."""
import argparse
import hashlib
import json
import os
import platform
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import threadpoolctl
from joblib import parallel_config, cpu_count
from sklearn.model_selection import StratifiedKFold, KFold
from threadpoolctl import threadpool_limits

from publication_benchmark import (REAL_DATASETS, method_specs, preprocess_fold,
                                   fit_selection, score_mask)
from experiment_metrics import PeakRSS, CPUTime, recovery_metrics, selection_stability
from synthetic_suite import synthetic_scenarios, generate_synthetic
from noninferiority import noninferiority_table
from run_management import (atomic_json, fresh_run, open_run, workload, partition_key,
                            checkpoint_paths, checkpoint_rows, rebuild_results, clean_json,
                            Progress, duration)


# Defaults for newly prepared runs; the manifest freezes the values actually used.
DEFAULT_DATA_SEEDS = (0,)
# Three shuffled 5-fold repeats: the corrected resampled-t variance factor is
# 1/15 + 1/4 = 0.32 versus 0.27 for ten repeats. Keeping five folds preserves
# 80% training size; 2-fold CV would raise the factor to 1/J + 1 instead.
DEFAULT_REPEATS = 3
# Serial Greedy repeats the parallel masks exactly; it is only a speedup reference.
DEFAULT_SERIAL_SHADOW_FRACTIONS = (1.,)
DEFAULT_CPU_BUDGET = -1  # Resolve all available logical CPUs when preparing a run.
# Compact default suite: correlated reference, dimensionality, weak signal,
# pure interaction, and false-positive behavior when every feature is noise.
DEFAULT_SYNTHETIC_SCENARIOS = (
    'reference', 'n_features_2000', 'signal_0.5', 'kind_xor', 'kind_null',
)
# Prespecified absolute score-loss budgets versus Vanilla's rough-fix policy.
# These are benchmark design choices, not universal F1/R2 equivalence thresholds.
DEFAULT_F1_MARGIN = 0.01
DEFAULT_R2_MARGIN = 0.01


def environment():
    return dict(python=platform.python_version(), threadpoolctl=threadpoolctl.__version__,
                packages={name: version(name) for name in
                          ['numpy', 'scipy', 'pandas', 'scikit-learn', 'lightgbm', 'joblib', 'psutil']})


def scheduled_jobs(config, repeat, fold):
    """One baseline job across alpha, plus every genuine selector timing measurement."""
    jobs = []
    for method in config['methods']:
        alphas = [None] if method['name'] == 'NoSelection' else config['alphas']
        for alpha in alphas:
            jobs.append((f"{method['name']}:{alpha}", method, alpha))
    rng = np.random.RandomState(repeat * config['folds'] + fold)
    return [jobs[index] for index in rng.permutation(len(jobs))]


def cached_scores(checkpoint):
    cache = {}
    for job in checkpoint['jobs'].values():
        for row in job['rows']:
            cache[row['Mask_ID']] = {key: value for key, value in row.items()
                                    if key.startswith('Downstream_') or key == 'Score'}
    return cache


def selection_parallelism(method):
    """Threads for ExtraTrees; processes only for concurrent Boruta iterations.

    Forcing loky around a sequential selector overrides ExtraTrees' preference
    for threads and serializes tree tasks between processes on every iteration.
    Native BLAS/OpenMP limits are separate from ExtraTrees' joblib thread count.
    """
    if method['outer'] == 1:
        return parallel_config(backend='threading')
    return parallel_config(backend='loky', inner_max_num_threads=1)


def run_partition(X_train, X_test, y_train, y_test, relevant, metadata, config,
                  checkpoint, progress=None):
    """Yield complete method jobs. Cache scores only for EXACT matching feature masks.

    The cache is partition-local and never loads data from any other run. Selection
    remains independently fitted for every alpha/method so its timing is measured.
    """
    cache = cached_scores(checkpoint) if config['cache_scores'] else {}
    p = X_train.shape[1]
    repeat, fold = metadata['Repeat'], metadata['Fold']
    seed = repeat * config['folds'] + fold
    for job_id, method_spec, alpha in scheduled_jobs(config, repeat, fold):
        if job_id in checkpoint['jobs']:
            continue
        method = dict(method_spec, task=metadata['Task'])
        label = f"{metadata['Dataset']} seed={metadata['Data_Seed']} repeat={repeat+1}/{config['repeats']} fold={fold+1}/{config['folds']} {method['name']} alpha={alpha}"
        if progress:
            progress.update('SELECT ' + label)
        wall_start = time.perf_counter()
        # ExtraTrees uses method['inner'] joblib threads, unaffected by this native limit.
        with threadpool_limits(limits=1), selection_parallelism(method):
            with PeakRSS() as memory, CPUTime() as cpu:
                start = time.perf_counter()
                masks, iterations, shadows = fit_selection(
                    X_train, y_train, method, alpha, seed, config['trees'], config['vanilla_iterations'])
                selection_time = time.perf_counter() - start
        rows = []
        for policy, mask in masks.items():
            selected = np.flatnonzero(mask).tolist()
            mask_id = hashlib.sha256(np.asarray(mask, dtype=np.uint8).tobytes()).hexdigest()
            measured = mask_id not in cache
            if measured:
                if progress:
                    progress.update(f'SCORE {label} policy={policy} selected={len(selected)}/{p} selection={selection_time:.1f}s')
                with threadpool_limits(limits=config['cpu_budget']), PeakRSS() as downstream_memory, \
                        CPUTime() as downstream_cpu:
                    start = time.perf_counter()
                    score = score_mask(X_train, y_train, X_test, y_test, mask,
                                       metadata['Task'], seed, config['cpu_budget'])
                    downstream_time = time.perf_counter() - start
                scoring = dict(Score=score, Downstream_Time_Sec=downstream_time,
                               Downstream_CPU_Sec=downstream_cpu.seconds,
                               Downstream_Source=f'{metadata["Partition_ID"]}/{job_id}/{policy}')
                scoring.update({f'Downstream_{k}': v for k, v in downstream_memory.as_dict().items()})
                if config['cache_scores']:
                    cache[mask_id] = scoring
            else:
                scoring = cache[mask_id]
            # NoSelection is alpha-independent. Duplicate observations, not model fits.
            output_alphas = config['alphas'] if alpha is None else [alpha]
            for alpha_index, output_alpha in enumerate(output_alphas):
                actual_fit = measured and alpha_index == 0
                row = dict(metadata, Alpha=output_alpha, Method=method['name'], Policy=policy,
                           Job_ID=job_id, Mask_ID=mask_id, N_Features=p, N_Selected=len(selected),
                           Selected_Features=selected, Iterations=iterations, N_Shadows=shadows,
                           Selection_Time_Sec=selection_time, Selection_CPU_Sec=cpu.seconds,
                           CPU_Budget=config['cpu_budget'],
                           Outer_Jobs=method['outer'], Estimator_Jobs=method['inner'],
                           Selection_Backend='threading' if method['outer']==1 else 'loky_outer_threading_inner',
                           **memory.as_dict(), **scoring)
                row['Downstream_Fit_Performed'] = actual_fit
                row['Downstream_Actual_Time_Sec'] = scoring['Downstream_Time_Sec'] if actual_fit else 0.0
                row['Downstream_Actual_CPU_Sec'] = scoring['Downstream_CPU_Sec'] if actual_fit else 0.0
                if relevant is not None:
                    row.update(recovery_metrics(selected, relevant, p))
                rows.append(clean_json(row))
        yield job_id, dict(rows=rows, wall_seconds=time.perf_counter()-wall_start)


def export_run(directory, manifest):
    """Export only this run, while it is stopped or after fitting has finished."""
    path = rebuild_results(directory)
    if not path.stat().st_size:
        return
    frame = pd.read_json(path, lines=True)
    # Convenient scalar CSV plus masks separately, retaining full information in JSONL.
    frame.drop(columns=['Selected_Features']).to_csv(directory / 'fold_metrics.csv', index=False)
    keys = ['Dataset', 'Data_Seed', 'Task', 'Alpha', 'Estimator', 'Method', 'Policy']
    scalar = [key for key in ['Score', 'N_Selected', 'Selection_Time_Sec', 'Selection_CPU_Sec',
                             'Downstream_Time_Sec', 'RSS_Peak_MB', 'Recall', 'Precision', 'FPR',
                             'FDP', 'FP_Original_Fraction']
              if key in frame]
    summary = frame.groupby(keys, dropna=False)[scalar].agg(['mean', 'std', 'count'])
    summary.columns = ['_'.join(parts) for parts in summary.columns]
    summary.reset_index().to_csv(directory / 'summary.csv', index=False)
    stability_rows = []
    frequency_file = directory / 'selection_frequency.csv'
    with frequency_file.open('w', encoding='utf-8', newline='') as output:
        first = True
        for values, group in frame.groupby(keys, dropna=False):
            meta = dict(zip(keys, values))
            jaccard, frequency = selection_stability(group.Selected_Features, int(group.N_Features.iloc[0]))
            stability_rows.append(dict(meta, Mean_Pairwise_Jaccard=jaccard, N_Fits=len(group),
                                       Mean_Selected=group.N_Selected.mean()))
            frequencies = pd.DataFrame({'Feature': np.arange(len(frequency)),
                                        'Selection_Frequency': frequency}).assign(**meta)
            frequencies.to_csv(output, index=False, header=first)
            first = False
    pd.DataFrame(stability_rows).to_csv(directory / 'stability.csv', index=False)
    if len(frame) == manifest['workload']['result_rows']:
        config = manifest['config']
        margins = {'class': config['f1_margin'], 'reg': config['r2_margin']}
        if all(margins[task] is not None for task in frame.Task.unique()):
            noninferiority_table(frame, margins, config['test_level']).to_csv(
                directory / 'noninferiority.csv', index=False)


def verify_data(directory, dataset, seed, X, y):
    """Prevent resume across changed raw data, while permitting cached OpenML downloads."""
    digest = hashlib.sha256(pd.util.hash_pandas_object(pd.DataFrame(X), index=True).values.tobytes())
    digest.update(np.asarray(y).tobytes())
    digest.update(str((X.shape, np.asarray(y).dtype)).encode())
    signature = digest.hexdigest()
    target = directory / 'dataset_fingerprints.json'
    existing = json.loads(target.read_text()) if target.exists() else {}
    key = f'{dataset}:{seed}'
    if key in existing and existing[key] != signature:
        raise ValueError(f'Raw data changed for {key}; start a fresh experiment.')
    existing[key] = signature
    atomic_json(target, existing)


class RunLock:
    """OS-held lock; a crash releases it without stale PID bookkeeping."""
    def __init__(self, directory):
        self.path = Path(directory) / '.run.lock'

    def __enter__(self):
        self.file = self.path.open('a+b')
        self.file.seek(0, 2)
        if self.file.tell() == 0:
            self.file.write(b'0')
            self.file.flush()
        self.file.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.file.close()
            raise RuntimeError('This experiment is already running or being exported.')
        return self

    def __exit__(self, *args):
        self.file.close()


def execute(directory, manifest, max_partitions=None):
    config = manifest['config']
    if config['environment'] != environment():
        raise ValueError('Python/package versions differ from the prepared run; prepare a fresh run with this environment.')
    with RunLock(directory):
        counts = [len(json.loads(path.read_text())['jobs']) for path in checkpoint_paths(directory)]
        completed = sum(counts)
        rebuild_results(directory)
        completed_partitions = 0
        with Progress(directory, manifest['workload']['method_jobs'], completed) as progress:
            allocations = sorted({(method['outer'], method['inner']) for method in config['methods']})
            progress.update(f"CPU budget={config['cpu_budget']}; available logical CPUs={cpu_count()}; "
                            f"outer workers x estimator threads={allocations}; ExtraTrees uses threads")
            with (directory / 'fold_results.jsonl').open('a', encoding='utf-8') as output:
                for name, (kind, spec) in config['datasets'].items():
                    seeds = config['data_seeds'] if kind == 'synthetic' else [0]
                    for data_seed in seeds:
                        X = y = relevant = None
                        for repeat in range(config['repeats']):
                            all_paths = [directory / 'checkpoints' / f'{partition_key(name,data_seed,repeat,f)}.json'
                                         for f in range(config['folds'])]
                            expected_jobs = len(scheduled_jobs(config, repeat, 0))
                            if all(path.exists() and len(json.loads(path.read_text())['jobs']) == expected_jobs
                                   for path in all_paths):
                                continue
                            if X is None:
                                progress.update(f'LOAD {name} data_seed={data_seed}')
                                if kind == 'synthetic':
                                    X, y, relevant = generate_synthetic(spec, data_seed)
                                    task = 'class'
                                else:
                                    from data_loader import load_openml_dataset
                                    X, y = load_openml_dataset(spec[0], spec[1], preprocess=False)
                                    relevant, task = None, spec[1]
                                verify_data(directory, name, data_seed, X, y)
                                progress.update(f'LOADED {name}: n={X.shape[0]}, p={X.shape[1]}')
                            cv_cls = StratifiedKFold if task == 'class' else KFold
                            cv = cv_cls(n_splits=config['folds'], shuffle=True, random_state=repeat)
                            for fold, (train, test) in enumerate(cv.split(X, y)):
                                checkpoint_path = all_paths[fold]
                                checkpoint = json.loads(checkpoint_path.read_text()) if checkpoint_path.exists() else {'jobs': {}}
                                if len(checkpoint['jobs']) == expected_jobs:
                                    continue
                                progress.update(f'PREPROCESS {name} seed={data_seed} repeat={repeat+1} fold={fold+1}')
                                X_train, X_test = preprocess_fold(X, train, test)
                                split_id = hashlib.sha256(train.tobytes()+test.tobytes()).hexdigest()
                                metadata = dict(Dataset=name, Data_Seed=data_seed, Task=task, Estimator='extra_trees',
                                                Repeat=repeat, Fold=fold, Split_ID=split_id,
                                                Partition_ID=partition_key(name,data_seed,repeat,fold),
                                                N_Train=len(train), N_Test=len(test))
                                for job_id, job in run_partition(X_train,X_test,y[train],y[test],relevant,
                                                                metadata,config,checkpoint,progress):
                                    checkpoint['jobs'][job_id] = job
                                    atomic_json(checkpoint_path, checkpoint)
                                    for row in job['rows']:
                                        output.write(json.dumps(row, allow_nan=False)+'\n')
                                    output.flush()
                                    completed += 1
                                    rows = job['rows']
                                    progress.update(f'DONE {job_id}: {job["wall_seconds"]:.1f}s; '
                                                    f'selected={rows[0]["N_Selected"]}; score={rows[0]["Score"]:.4f}',
                                                    completed=completed)
                                completed_partitions += 1
                                if max_partitions and completed_partitions >= max_partitions:
                                    progress.update('Requested partition limit reached; exporting partial results')
                                    output.close()  # Windows cannot replace an open JSONL file.
                                    export_run(directory, manifest)
                                    finished = completed == manifest['workload']['method_jobs']
                                    progress.update('Complete' if finished else 'Stopped at requested partition limit; use --resume to continue',
                                                    status='complete' if finished else 'paused')
                                    return
            progress.update('Exporting analysis tables')
            export_run(directory, manifest)
            progress.update('Complete', status='complete')


def build_config(args):
    resolved_cpu_budget = cpu_count() if args.cpu_budget == -1 else args.cpu_budget
    scenarios = synthetic_scenarios()
    if args.smoke:
        args.suite, args.repeats, args.data_seeds, args.alphas = 'synthetic', 2, [0], [.1]
        scenarios = {key: dict(scenarios[key], n_samples=80, n_features=12, n_correlated=0)
                     for key in ['reference', 'kind_null', 'kind_xor']}
    elif not args.datasets:
        scenarios = {key: scenarios[key] for key in DEFAULT_SYNTHETIC_SCENARIOS}
    datasets = {}
    if args.suite in ('synthetic', 'both'):
        datasets.update({key: ['synthetic', spec] for key,spec in scenarios.items()})
    if args.suite in ('real', 'both'):
        datasets.update({key: ['real', spec] for key,spec in REAL_DATASETS.items()})
    if args.datasets:
        if set(args.datasets) - datasets.keys():
            raise ValueError('Unknown dataset/scenario name')
        datasets = {name: datasets[name] for name in args.datasets}
    if args.repeats < 1 or any(not 0<a<1 for a in args.alphas):
        raise ValueError('Require positive repeats and alpha in (0,1)')
    if len(set(args.alphas)) != len(args.alphas) or len(set(args.data_seeds)) != len(args.data_seeds):
        raise ValueError('Duplicate alphas/data seeds would duplicate observations')
    if len(set(args.shadow_fractions)) != len(args.shadow_fractions):
        raise ValueError('Duplicate shadow fractions would duplicate methods')
    if not 0<args.test_level<1 or any(m is not None and (not np.isfinite(m) or m<=0)
                                   for m in [args.f1_margin,args.r2_margin]):
        raise ValueError('Margins must be positive and test level in (0,1)')
    return dict(datasets=datasets, data_seeds=args.data_seeds, repeats=args.repeats,
                folds=2 if args.smoke else 5, alphas=args.alphas, cpu_budget=resolved_cpu_budget,
                cpu_budget_requested=args.cpu_budget,
                methods=method_specs(resolved_cpu_budget,args.outer_jobs,args.shadow_fractions,
                                     args.serial_shadow_fractions),
                cache_scores=not args.no_score_cache, trees=8 if args.smoke else 'auto',
                vanilla_iterations=12 if args.smoke else 512, smoke=args.smoke,
                f1_margin=args.f1_margin,r2_margin=args.r2_margin,test_level=args.test_level,
                environment=environment())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    action=parser.add_mutually_exclusive_group()
    action.add_argument('--prepare',action='store_true',help='Create a new run without launching fits')
    action.add_argument('--plan',action='store_true',help='Print workload without creating files or fitting')
    action.add_argument('--run',type=Path,help='Start a previously prepared fresh run')
    action.add_argument('--resume',type=Path,help='Continue only this new-protocol run; never imports old results')
    action.add_argument('--export',type=Path,help='Regenerate analysis CSVs from this run checkpoints')
    parser.add_argument('--suite',choices=['synthetic','real','both'],default='both')
    parser.add_argument('--datasets',nargs='+',
                        help='Explicit scenario/dataset names; defaults to five synthetic cases and/or six real datasets')
    parser.add_argument('--output',type=Path,help='New directory; defaults to a unique timestamped directory')
    parser.add_argument('--repeats',type=int,default=DEFAULT_REPEATS,
                        help='Shuffled 5-fold CV repeats per dataset realization (default: 3)')
    parser.add_argument('--data-seeds',type=int,nargs='+',default=list(DEFAULT_DATA_SEEDS),
                        help='Synthetic generation seeds (default: one realization, seed 0)')
    parser.add_argument('--alphas',type=float,nargs='+',default=[.1,.001,.00001])
    parser.add_argument('--cpu-budget',type=int,default=DEFAULT_CPU_BUDGET,
                        help='Total estimator threads; -1 (default) uses all available logical CPUs')
    parser.add_argument('--outer-jobs',type=int,default=4)
    parser.add_argument('--shadow-fractions',type=float,nargs='+',default=[1.,.25,.1])
    parser.add_argument('--serial-shadow-fractions',type=float,nargs='+',
                        default=list(DEFAULT_SERIAL_SHADOW_FRACTIONS),
                        help='Shadow fractions also run with outer jobs=1 as the parallel-speedup '
                             'reference; a subset of --shadow-fractions (default: 1)')
    parser.add_argument('--no-score-cache',action='store_true')
    parser.add_argument('--f1-margin',type=float,default=DEFAULT_F1_MARGIN,
                        help='Prespecified absolute weighted-F1 loss margin (default: 0.01)')
    parser.add_argument('--r2-margin',type=float,default=DEFAULT_R2_MARGIN,
                        help='Prespecified absolute R2 loss margin (default: 0.01)')
    parser.add_argument('--test-level',type=float,default=.01)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--max-partitions',type=int,help='Stop after this many newly completed CV partitions')
    args=parser.parse_args()
    if args.max_partitions is not None and args.max_partitions<1:
        parser.error('--max-partitions must be positive')
    if tuple(map(int,threadpoolctl.__version__.split('.')[:2]))<(3,6):
        parser.error('threadpoolctl >=3.6 required; install requirements-publication.txt or use publication/run.ps1')
    existing=args.run or args.resume or args.export
    try:
        if existing:
            directory,manifest=open_run(existing)
            if args.export:
                with RunLock(directory):
                    export_run(directory,manifest)
                print(f'Exported {directory}')
                return
            if args.run and checkpoint_paths(directory):
                raise ValueError('This run has checkpoints; use --resume explicitly to continue it.')
        else:
            config=build_config(args)
            print(json.dumps(workload(config),indent=2))
            if args.plan:
                for name,(kind,spec) in config['datasets'].items():
                    print(name,kind,spec)
                return
            directory=fresh_run(config,args.output)
            directory,manifest=open_run(directory)
            print(f'NEW RUN: {directory}',flush=True)
            if args.prepare:
                print('Prepared only. Start with --run pointing to this directory.')
                return
        execute(directory,manifest,args.max_partitions)
        print(f'Results: {directory}')
    except (ValueError,FileExistsError,RuntimeError) as exc:
        parser.error(str(exc))


if __name__=='__main__':
    main()
