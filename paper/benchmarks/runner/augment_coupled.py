"""Augment a finished v6 run with coupled Greedy selections and Vanilla histories.

Everything is written into a NEW copy of the source run; the source is only read.
Running the same command again resumes the copy and skips finished work.

Stage ``coupled`` (one fit per CV partition)
    Refit Vanilla Boruta with the logged configuration and seed, but stop after
    K_max = max over alphas of K(alpha) = ceil(log2(p / alpha)) iterations. No
    feature can be confirmed or rejected before iteration K(alpha), so these
    iterations are exactly the first K(alpha) iterations of every logged Vanilla
    fit. Greedy Boruta evaluated on that shared importance sequence is added as
    method ``GreedyCoupled``: the coupled realization of Theorem 1. The stage
    verifies, and fails loudly otherwise, that C_Vanilla is a subset of
    C_GreedyCoupled, that strongly confirmed features are a subset of Vanilla's
    strict confirmations, and that K matches the logged Greedy iteration count.
    Its selection time is Vanilla's elapsed time at iteration K: the cost of the
    same computation stopped at Greedy's budget.

Stage ``history`` (full Vanilla refits for a configurable subset)
    Refit Vanilla to termination with per-iteration bookkeeping: confirmed,
    tentative and rejected counts, the iteration at which each feature was
    decided, elapsed and CPU time. It verifies exact reproduction of the logged
    strict and rough-fix masks and iteration count and, when available, of the
    coupled prefix importances.

Bookkeeping hooks consume no random numbers, so recorded fits are identical to
unrecorded ones. Selector arguments are captured from the runner's own
``fit_selection`` rather than copied, so the script stays valid for future runs.
"""
import argparse
import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psutil
from scipy.stats import binom
from sklearn.model_selection import StratifiedKFold, KFold
from threadpoolctl import threadpool_limits
import threadpoolctl

import publication_benchmark
from BorutaLocal import BorutaPy
from GreedyBorutaParallel import GreedyBorutaPyParallel
from experiment_metrics import PeakRSS, CPUTime, recovery_metrics
from experiment_run import (RunLock, cached_scores, environment, export_run,
                            selection_parallelism, verify_data)
from publication_benchmark import fit_selection, preprocess_fold, score_mask
from run_management import atomic_json, clean_json, open_run, partition_key, Progress
from synthetic_suite import generate_synthetic


AUGMENTATION = 'coupled-prefix-v1'
COUPLED = 'GreedyCoupled'
AUG_DIR = 'augmentations'
SCRIPT_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class CouplingError(RuntimeError):
    """A theoretical identity failed: the refit is not the logged trajectory."""


class RecordingBoruta(BorutaPy):
    """BorutaPy plus bookkeeping at hooks that never touch the random state."""
    snapshot_iterations = ()

    def _fit(self, X, y):
        self._process = psutil.Process()
        self.trace_ = dict(sha_max=[], elapsed=[], cpu=[])
        self.status_ = {}
        self.decision_iteration_ = np.full(X.shape[1], -1)
        self.final_decisions_ = np.zeros(X.shape[1], dtype=int)
        self.hit_snapshots_ = {}
        self._iteration = 0
        self._start_cpu = self._cpu_seconds()
        self._start = time.perf_counter()
        return super()._fit(X, y)

    def _cpu_seconds(self):
        used = self._process.cpu_times()
        return used.user + used.system

    def _assign_hits(self, hit_reg, cur_imp, imp_sha_max):
        hit_reg = super()._assign_hits(hit_reg, cur_imp, imp_sha_max)
        self._iteration += 1
        self.trace_['sha_max'].append(float(imp_sha_max))
        self.trace_['elapsed'].append(time.perf_counter() - self._start)
        self.trace_['cpu'].append(self._cpu_seconds() - self._start_cpu)
        if self._iteration in self.snapshot_iterations:
            self.hit_snapshots_[self._iteration] = hit_reg.copy()
        return hit_reg

    def _do_tests(self, dec_reg, hit_reg, _iter):
        if _iter != self._iteration:
            raise CouplingError('Iteration bookkeeping is out of step with BorutaPy')
        before = dec_reg.copy()
        dec_reg = super()._do_tests(dec_reg, hit_reg, _iter)
        self.decision_iteration_[(before == 0) & (dec_reg != 0)] = _iter
        self.status_[_iter] = tuple(int((dec_reg == code).sum()) for code in (1, 0, -1))
        self.final_decisions_ = dec_reg.copy()
        return dec_reg

    def trajectory(self):
        """Per-iteration counts after that iteration's tests (all tentative before tests)."""
        p, counts, last = len(self.final_decisions_), [], (0, len(self.final_decisions_), 0)
        for iteration in range(1, self._iteration + 1):
            last = self.status_.get(iteration, last)
            counts.append(last)
        confirmed, tentative, rejected = (list(c) for c in zip(*counts)) if counts else ([], [], [])
        return dict(iteration=list(range(1, self._iteration + 1)), elapsed=self.trace_['elapsed'],
                    cpu=self.trace_['cpu'], sha_max=self.trace_['sha_max'],
                    confirmed=confirmed, tentative=tentative, rejected=rejected, n_features=p)


class _Captured(Exception):
    pass


def runner_vanilla_arguments(config, task, seed, alpha):
    """The exact estimator and BorutaPy arguments the runner's fit_selection uses."""
    captured = {}

    class Capture:
        def __init__(self, estimator, **kwargs):
            captured.update(kwargs, estimator=estimator)

        def fit(self, X, y):
            raise _Captured

    method = dict(vanilla_method(config), task=task)
    original = publication_benchmark.BorutaPy
    publication_benchmark.BorutaPy = Capture
    try:
        fit_selection(np.zeros((2, 1)), np.zeros(2), method, alpha, seed,
                      config['trees'], config['vanilla_iterations'])
    except _Captured:
        pass
    finally:
        publication_benchmark.BorutaPy = original
    if 'estimator' not in captured:
        raise RuntimeError('Could not capture the runner Vanilla configuration')
    return captured


def recording_vanilla(config, task, seed, alpha, max_iter=None):
    arguments = runner_vanilla_arguments(config, task, seed, alpha)
    if max_iter is not None:
        arguments['max_iter'] = max_iter
    return RecordingBoruta(arguments.pop('estimator'), **arguments)


def vanilla_method(config):
    return next(m for m in config['methods'] if m['name'] == 'Vanilla')


def greedy_reference_method(config):
    """A logged full-shadow Greedy method, used to cross-check K."""
    full = [m for m in config['methods'] if m['name'].startswith('Greedy') and m['shadows'] == 1.0]
    return min(full, key=lambda m: m['outer'])['name'] if full else None


def k_iterations(alpha, p):
    return GreedyBorutaPyParallel(None, alpha=alpha)._compute_iterations_to_convergence(alpha, p)


def strong_features(hits, k, alpha, p):
    """Vanilla's Bonferroni confirmation test at iteration K (BorutaLocal._do_tests)."""
    return np.flatnonzero((hits > 0) & (binom.sf(hits - 1, k, .5) <= alpha / float(p)))


def job_id(name, alpha):
    return f'{name}:{alpha}'


def alpha_key(alpha):
    return f'{alpha:g}'


def utc_now():
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------- run directories

def source_identity(source):
    return dict(run=source.name, path=str(source),
                manifest_sha256=hashlib.sha256((source / 'manifest.json').read_bytes()).hexdigest())


def prepare_output(source, output):
    """Clone the source once; afterwards only verify that the clone matches it."""
    if output.exists():
        manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
        if manifest.get('cloned_from', {}).get('manifest_sha256') != source_identity(source)['manifest_sha256']:
            raise ValueError(f'{output} exists but is not an augmentation copy of {source.name}')
        return manifest, False
    shutil.copytree(source, output, ignore=shutil.ignore_patterns('.run.lock', '*.tmp'))
    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    manifest['cloned_from'] = dict(source_identity(source), cloned_utc=utc_now())
    manifest['augmentations'] = []
    atomic_json(output / 'manifest.json', manifest)
    return manifest, True


def register_augmentation(output, manifest, args, history_items):
    """One record per augmentation version; every invocation is appended to its log."""
    record = next((a for a in manifest['augmentations'] if a['version'] == AUGMENTATION), None)
    if record is None:
        record = dict(version=AUGMENTATION, name='coupled Greedy prefix and Vanilla histories',
                      created_utc=utc_now(), script_sha256=[SCRIPT_SHA256],
                      methods=[dict(name=COUPLED, outer=1, inner=vanilla_method(manifest['config'])['inner'],
                                    shadows=1.0, policy='selected', derived_from='Vanilla',
                                    description='Greedy Boruta on the first K iterations of the '
                                                'logged Vanilla fit (shared importance sequence)')],
                      timing_note='GreedyCoupled selection time is the elapsed time of the shared '
                                  'Vanilla computation at iteration K, measured in this session.',
                      invocations=[], status='running')
        manifest['augmentations'].append(record)
    elif SCRIPT_SHA256 not in record['script_sha256']:
        if not args.allow_code_change:
            raise ValueError('augment_coupled.py changed since this copy was started. Delete the copy '
                             'to restart, or pass --allow-code-change to record mixed code versions.')
        record['script_sha256'].append(SCRIPT_SHA256)
    record['invocations'].append(dict(started_utc=utc_now(), stages=args.stages, datasets=args.datasets,
                                      history=[list(item) for item in history_items]))
    record['status'] = 'running'
    atomic_json(output / 'manifest.json', manifest)
    return record


# ---------------------------------------------------------------- partitions

def partitions(config, datasets=None):
    """(name, kind, spec, data_seed, repeat, fold) in the runner's order."""
    for name, (kind, spec) in config['datasets'].items():
        if datasets and name not in datasets:
            continue
        for data_seed in (config['data_seeds'] if kind == 'synthetic' else [0]):
            for repeat in range(config['repeats']):
                for fold in range(config['folds']):
                    yield name, kind, spec, data_seed, repeat, fold


class DataCache:
    """Load each dataset realization once and reproduce the runner's CV splits."""
    def __init__(self, directory, progress):
        self.directory, self.progress, self.key, self.value = directory, progress, None, None

    def partition(self, config, name, kind, spec, data_seed, repeat, fold):
        if self.key != (name, data_seed):
            self.progress.update(f'LOAD {name} data_seed={data_seed}')
            if kind == 'synthetic':
                X, y, relevant = generate_synthetic(spec, data_seed)
                task = 'class'
            else:
                from data_loader import load_openml_dataset
                X, y = load_openml_dataset(spec[0], spec[1], preprocess=False)
                relevant, task = None, spec[1]
            verify_data(self.directory, name, data_seed, X, y)
            self.key, self.value = (name, data_seed), (X, y, relevant, task)
        X, y, relevant, task = self.value
        cv_cls = StratifiedKFold if task == 'class' else KFold
        train, test = list(cv_cls(n_splits=config['folds'], shuffle=True, random_state=repeat).split(X, y))[fold]
        X_train, X_test = preprocess_fold(X, train, test)
        split_id = hashlib.sha256(train.tobytes() + test.tobytes()).hexdigest()
        return X_train, X_test, y[train], y[test], relevant, task, split_id


def checkpoint_path(directory, name, data_seed, repeat, fold):
    return directory / 'checkpoints' / f'{partition_key(name, data_seed, repeat, fold)}.json'


def logged_rows(checkpoint, method, alpha):
    key = job_id(method, alpha)
    if key not in checkpoint['jobs']:
        raise ValueError(f'Source run is missing job {key}; augment only completed partitions')
    return {row['Policy']: row for row in checkpoint['jobs'][key]['rows']}


# ---------------------------------------------------------------- coupled stage

def prefix_path(directory, partition_id):
    return directory / AUG_DIR / 'coupled_prefix' / f'{partition_id}.npz'


def save_npz(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def run_prefix(config, task, seed, X_train, y_train, p):
    """Vanilla for K_max iterations at the smallest alpha (no decisions before K_max)."""
    alpha = min(config['alphas'])
    ks = {a: k_iterations(a, p) for a in config['alphas']}
    k_max = max(ks.values())
    selector = recording_vanilla(config, task, seed, alpha, max_iter=k_max + 1)
    selector.snapshot_iterations = tuple(sorted(set(ks.values())))
    with threadpool_limits(limits=1), selection_parallelism(vanilla_method(config)):
        with PeakRSS() as memory, CPUTime() as cpu:
            start = time.perf_counter()
            selector.fit(X_train, y_train)
            wall = time.perf_counter() - start
    importances = selector.importance_history_[1:]
    if importances.shape[0] != k_max or np.isnan(importances).any():
        raise CouplingError('Prefix fit did not produce K_max complete importance rows')
    arrays = dict(importances=importances, sha_max=np.asarray(selector.trace_['sha_max']),
                  elapsed=np.asarray(selector.trace_['elapsed']), cpu=np.asarray(selector.trace_['cpu']))
    arrays.update({f'hits_K{k}': hits for k, hits in selector.hit_snapshots_.items()})
    meta = dict(augmentation=AUGMENTATION, script_sha256=SCRIPT_SHA256, created_utc=utc_now(),
                prefix_alpha=alpha, k_max=k_max, seed=seed, wall_seconds=wall, cpu_seconds=cpu.seconds,
                **memory.as_dict(), k_by_alpha={alpha_key(a): k for a, k in ks.items()})
    arrays['meta'] = np.array(json.dumps(meta))
    return arrays


def coupled_selection(prefix, alpha, p):
    """Greedy on the first K shared iterations, checked against BorutaPy's own hit register."""
    k = k_iterations(alpha, p)
    importances, sha_max = prefix['importances'][:k], prefix['sha_max'][:k]
    hits = (importances > sha_max[:, None]).sum(axis=0)
    snapshot = prefix.get(f'hits_K{k}')
    if snapshot is not None and not np.array_equal(hits, snapshot):
        raise CouplingError('Hit counts derived from the prefix differ from BorutaPy hit_reg')
    return k, hits


def score_fields(cache, mask, X_train, y_train, X_test, y_test, task, seed, config, source):
    mask_id = hashlib.sha256(np.asarray(mask, dtype=np.uint8).tobytes()).hexdigest()
    if mask_id in cache:
        scoring, performed = cache[mask_id], False
    else:
        with threadpool_limits(limits=config['cpu_budget']), PeakRSS() as memory, CPUTime() as cpu:
            start = time.perf_counter()
            score = score_mask(X_train, y_train, X_test, y_test, mask, task, seed, config['cpu_budget'])
            elapsed = time.perf_counter() - start
        scoring = dict(Score=score, Downstream_Time_Sec=elapsed, Downstream_CPU_Sec=cpu.seconds,
                       Downstream_Source=source)
        scoring.update({f'Downstream_{k}': v for k, v in memory.as_dict().items()})
        cache[mask_id] = scoring
        performed = True
    return mask_id, scoring, performed


def coupled_partition(directory, config, item, data, progress, verification):
    name, kind, spec, data_seed, repeat, fold = item
    path = checkpoint_path(directory, name, data_seed, repeat, fold)
    checkpoint = json.loads(path.read_text(encoding='utf-8'))
    missing = [a for a in config['alphas'] if job_id(COUPLED, a) not in checkpoint['jobs']]
    partition_id = path.stem
    npz = prefix_path(directory, partition_id)
    if not missing and npz.exists():
        return False
    label = f'{name} seed={data_seed} repeat={repeat + 1}/{config["repeats"]} fold={fold + 1}/{config["folds"]}'
    X_train, X_test, y_train, y_test, relevant, task, split_id = data.partition(config, *item)
    vanilla_rows = {a: logged_rows(checkpoint, 'Vanilla', a) for a in config['alphas']}
    reference = vanilla_rows[config['alphas'][0]]['strict']
    if reference['Split_ID'] != split_id:
        raise CouplingError(f'{label}: reproduced split differs from the logged split')
    p, seed = X_train.shape[1], repeat * config['folds'] + fold
    if npz.exists():
        with np.load(npz) as stored:
            prefix = {key: stored[key] for key in stored.files}
    else:
        progress.update(f'PREFIX {label}: Vanilla for K_max iterations')
        prefix = run_prefix(config, task, seed, X_train, y_train, p)
        save_npz(npz, prefix)
    meta = json.loads(str(prefix['meta']))
    metadata = {key: reference[key] for key in ['Dataset', 'Data_Seed', 'Task', 'Estimator', 'Repeat', 'Fold',
                                                'Split_ID', 'Partition_ID', 'N_Train', 'N_Test']}
    cache = cached_scores(checkpoint) if config['cache_scores'] else {}
    greedy_name = greedy_reference_method(config)
    method = vanilla_method(config)
    for alpha in config['alphas']:
        key = job_id(COUPLED, alpha)
        k, hits = coupled_selection(prefix, alpha, p)
        selected = np.flatnonzero(hits > 0)
        strong = strong_features(hits, k, alpha, p)
        logged = vanilla_rows[alpha]
        chosen = set(selected.tolist())
        checks = dict(Dataset=name, Data_Seed=data_seed, Repeat=repeat, Fold=fold, Alpha=alpha, K=k,
                      Theorem1_Strict=set(logged['strict']['Selected_Features']) <= chosen,
                      Theorem1_Rough_Fix=set(logged['rough_fix']['Selected_Features']) <= chosen,
                      Strong_In_Strict=set(strong.tolist()) <= set(logged['strict']['Selected_Features']),
                      K_Matches_Greedy=True)
        if greedy_name:
            logged_k = logged_rows(checkpoint, greedy_name, alpha)['selected']['Iterations']
            checks['K_Matches_Greedy'] = logged_k == k
        if not all(v for c, v in checks.items() if c.startswith(('Theorem', 'Strong', 'K_'))):
            raise CouplingError(f'{label} alpha={alpha}: coupling identity failed: {checks}')
        verification.append(checks)
        # Written immediately so an interrupted session keeps its record.
        with (directory / AUG_DIR / 'coupled_verification.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(clean_json(checks)) + '\n')
        if key in checkpoint['jobs']:
            continue
        mask = hits > 0
        progress.update(f'SCORE {label} {COUPLED} alpha={alpha} selected={len(selected)}/{p} strong={len(strong)}')
        source = f'{partition_id}/{key}/selected'
        mask_id, scoring, performed = score_fields(cache, mask, X_train, y_train, X_test, y_test,
                                                   task, seed, config, source)
        row = dict(metadata, Alpha=alpha, Method=COUPLED, Policy='selected', Job_ID=key, Mask_ID=mask_id,
                   N_Features=p, N_Selected=len(selected), Selected_Features=selected.tolist(),
                   Iterations=k, N_Shadows=p,
                   Selection_Time_Sec=float(prefix['elapsed'][k - 1]),
                   Selection_CPU_Sec=float(prefix['cpu'][k - 1]),
                   CPU_Budget=config['cpu_budget'], Outer_Jobs=1, Estimator_Jobs=method['inner'],
                   Selection_Backend='threading',
                   RSS_Baseline_MB=meta['RSS_Baseline_MB'], RSS_Peak_MB=meta['RSS_Peak_MB'],
                   RSS_Peak_Increase_MB=meta['RSS_Peak_Increase_MB'], **scoring)
        row['Downstream_Fit_Performed'] = performed
        row['Downstream_Actual_Time_Sec'] = scoring['Downstream_Time_Sec'] if performed else 0.0
        row['Downstream_Actual_CPU_Sec'] = scoring['Downstream_CPU_Sec'] if performed else 0.0
        if relevant is not None:
            row.update(recovery_metrics(selected.tolist(), relevant, p))
        row.update(Selected_Hits=hits[selected].tolist(), Strong_Features=strong.tolist(),
                   N_Strong=len(strong), Coupled_Prefix_Iterations=meta['k_max'],
                   Coupled_Prefix_Time_Sec=meta['wall_seconds'],
                   Augmentation=AUGMENTATION, Augmentation_Code_SHA256=SCRIPT_SHA256)
        checkpoint['jobs'][key] = dict(rows=[clean_json(row)],
                                       wall_seconds=float(prefix['elapsed'][k - 1]) + row['Downstream_Actual_Time_Sec'])
        atomic_json(path, checkpoint)
    return True


# ---------------------------------------------------------------- history stage

def history_path(directory, partition_id, alpha):
    return directory / AUG_DIR / 'vanilla_history' / f'{partition_id}_alpha{alpha_key(alpha)}.json'


def history_item(directory, config, item, alpha, data, progress):
    name, kind, spec, data_seed, repeat, fold = item
    path = checkpoint_path(directory, name, data_seed, repeat, fold)
    target = history_path(directory, path.stem, alpha)
    if target.exists():
        return False
    label = f'{name} seed={data_seed} repeat={repeat + 1} fold={fold + 1} alpha={alpha}'
    checkpoint = json.loads(path.read_text(encoding='utf-8'))
    logged = logged_rows(checkpoint, 'Vanilla', alpha)
    X_train, _, y_train, _, _, task, split_id = data.partition(config, *item)
    if logged['strict']['Split_ID'] != split_id:
        raise CouplingError(f'{label}: reproduced split differs from the logged split')
    p, seed = X_train.shape[1], repeat * config['folds'] + fold
    k = k_iterations(alpha, p)
    selector = recording_vanilla(config, task, seed, alpha)
    selector.snapshot_iterations = (k,)
    progress.update(f'HISTORY {label}: full Vanilla refit (logged {logged["strict"]["Selection_Time_Sec"]:.0f}s)')
    with threadpool_limits(limits=1), selection_parallelism(vanilla_method(config)):
        with PeakRSS() as memory, CPUTime() as cpu:
            start = time.perf_counter()
            selector.fit(X_train, y_train)
            wall = time.perf_counter() - start
    iterations = selector.importance_history_.shape[0] - 1
    strict = np.flatnonzero(selector.support_).tolist()
    rough = np.flatnonzero(selector.support_ | selector.support_weak_).tolist()
    trajectory = selector.trajectory()
    verification = dict(Strict_Mask_Equal=strict == logged['strict']['Selected_Features'],
                        Rough_Fix_Mask_Equal=rough == logged['rough_fix']['Selected_Features'],
                        Iterations_Equal=iterations == logged['strict']['Iterations'])
    npz = prefix_path(directory, path.stem)
    if npz.exists():
        with np.load(npz) as prefix:
            verification['Prefix_Importances_Equal'] = bool(np.array_equal(
                selector.importance_history_[1:k + 1], prefix['importances'][:k]))
            verification['Prefix_Shadow_Max_Equal'] = bool(np.array_equal(
                np.asarray(trajectory['sha_max'][:k]), prefix['sha_max'][:k]))
    tentative = int((selector.final_decisions_ == 0).sum())
    stop = ('resolved' if tentative == 0 else
            'iteration_cap' if iterations >= config['vanilla_iterations'] else 'early_stopping')
    hits_at_k = selector.hit_snapshots_.get(k)
    record = dict(Dataset=name, Data_Seed=data_seed, Repeat=repeat, Fold=fold, Partition_ID=path.stem,
                  Alpha=alpha, N_Features=p, K=k, Iterations=iterations,
                  Max_Iterations=config['vanilla_iterations'], Stop_Reason=stop,
                  Final_Confirmed=int((selector.final_decisions_ == 1).sum()), Final_Tentative=tentative,
                  Final_Rejected=int((selector.final_decisions_ == -1).sum()),
                  Rough_Fix_Promoted=np.flatnonzero(selector.support_weak_).tolist(),
                  Selection_Time_Sec=wall, Selection_CPU_Sec=cpu.seconds, **memory.as_dict(),
                  Time_At_K_Sec=trajectory['elapsed'][k - 1], CPU_At_K_Sec=trajectory['cpu'][k - 1],
                  Logged_Selection_Time_Sec=logged['strict']['Selection_Time_Sec'],
                  Logged_Selection_CPU_Sec=logged['strict'].get('Selection_CPU_Sec'),
                  Trajectory=trajectory,
                  Decision_Iteration=selector.decision_iteration_.tolist(),
                  Final_Decision=selector.final_decisions_.tolist(),
                  Hit_Features_At_K=np.flatnonzero(hits_at_k).tolist() if hits_at_k is not None else None,
                  Hits_At_K=hits_at_k[hits_at_k > 0].tolist() if hits_at_k is not None else None,
                  Verification=verification, Augmentation=AUGMENTATION,
                  Augmentation_Code_SHA256=SCRIPT_SHA256, Created_UTC=utc_now())
    target.parent.mkdir(parents=True, exist_ok=True)
    if not all(verification.values()):
        atomic_json(target.with_name(target.stem + '.failed.json'), record)
        raise CouplingError(f'{label}: refit does not reproduce the logged Vanilla fit: {verification}')
    atomic_json(target, record)
    return True


# ---------------------------------------------------------------- driver

def default_history_alphas(config):
    alphas = sorted(config['alphas'])
    return [0.001] if 0.001 in alphas else [alphas[len(alphas) // 2]]


def history_items(config, args):
    datasets = args.history_datasets or args.datasets or list(config['datasets'])
    alphas = args.history_alphas or default_history_alphas(config)
    unknown = set(datasets) - set(config['datasets'])
    if unknown or set(alphas) - set(config['alphas']):
        raise ValueError(f'History selection outside the run design: {sorted(unknown)} {alphas}')
    folds = [f for f in args.history_folds if f < config['folds']]
    repeats = [r for r in args.history_repeats if r < config['repeats']]
    return [(item, alpha) for item in partitions(config, datasets)
            if item[4] in repeats and item[5] in folds for alpha in alphas]


def logged_seconds(directory, item, method, alpha):
    path = checkpoint_path(directory, *[item[i] for i in (0, 3, 4, 5)])
    checkpoint = json.loads(path.read_text(encoding='utf-8'))
    key = job_id(method, alpha)
    return checkpoint['jobs'][key]['rows'][0]['Selection_Time_Sec'] if key in checkpoint['jobs'] else 0.0


def print_plan(directory, config, coupled_items, history):
    greedy = greedy_reference_method(config)
    alpha = min(config['alphas'])
    coupled = sum(logged_seconds(directory, item, greedy, alpha) for item in coupled_items) if greedy else 0
    full = sum(logged_seconds(directory, item, 'Vanilla', a) for item, a in history)
    print(f'Coupled prefix fits: {len(coupled_items)} partitions, roughly {coupled / 3600:.1f} h '
          f'(serial Greedy time at alpha={alpha:g} as a proxy) plus downstream LightGBM fits')
    print(f'Full Vanilla history refits: {len(history)}, roughly {full / 3600:.1f} h (logged Vanilla time)')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', type=Path, required=True, help='Finished v6 run directory (read only)')
    parser.add_argument('--output', type=Path,
                        help='Augmented copy; default <source>_aug. An existing copy is resumed.')
    parser.add_argument('--stages', nargs='+', choices=['coupled', 'history'], default=['coupled', 'history'])
    parser.add_argument('--datasets', nargs='+', help='Restrict both stages to these datasets (testing)')
    parser.add_argument('--history-datasets', nargs='+', help='Default: every dataset in the run')
    parser.add_argument('--history-alphas', type=float, nargs='+', help='Default: 0.001 (or the middle alpha)')
    parser.add_argument('--history-repeats', type=int, nargs='+', default=[0])
    parser.add_argument('--history-folds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--limit', type=int, help='Stop each stage after this many new items (testing)')
    parser.add_argument('--plan', action='store_true', help='Print the workload and exit without writing')
    parser.add_argument('--allow-code-change', action='store_true',
                        help='Resume a copy started with a different version of this script')
    args = parser.parse_args()
    if tuple(map(int, threadpoolctl.__version__.split('.')[:2])) < (3, 6):
        parser.error('threadpoolctl >=3.6 required; use benchmarks/publication/augment.ps1')
    try:
        source, source_manifest = open_run(args.source)
        config = source_manifest['config']
        if config['environment'] != environment():
            raise ValueError('Python/package versions differ from the source run; timings and fits '
                             'would not be comparable.')
        progress_file = json.loads((source / 'progress.json').read_text(encoding='utf-8'))
        if progress_file.get('status') != 'complete':
            raise ValueError('The source run is not complete; finish it before augmenting.')
        if args.datasets and set(args.datasets) - set(config['datasets']):
            raise ValueError('Unknown dataset name')
        output = (args.output or source.parent / f'{source.name}_aug').resolve()
        coupled_items = list(partitions(config, args.datasets)) if 'coupled' in args.stages else []
        history = history_items(config, args) if 'history' in args.stages else []
        if not coupled_items and not history:
            raise ValueError('Nothing selected: check --stages, --datasets and the history options')
        if args.plan:
            print_plan(source, config, coupled_items, history)
            return
        manifest, created = prepare_output(source, output)
        print(('NEW AUGMENTED COPY: ' if created else 'RESUMING: ') + str(output), flush=True)
        print_plan(output, config, coupled_items, history)
        verification = []
        with RunLock(output):
            record = register_augmentation(output, manifest, args, [(i[0], i[3], i[4], i[5], a) for i, a in history])
            (output / AUG_DIR).mkdir(exist_ok=True)
            done = 0
            with Progress(output / AUG_DIR, len(coupled_items) + len(history)) as progress:
                data = DataCache(output, progress)
                new = 0
                for item in coupled_items:
                    if args.limit and new >= args.limit:
                        break
                    new += coupled_partition(output, config, item, data, progress, verification)
                    done += 1
                    progress.update(f'COUPLED done {item[0]} repeat={item[4] + 1} fold={item[5] + 1}', completed=done)
                new = 0
                for item, alpha in history:
                    if args.limit and new >= args.limit:
                        break
                    new += history_item(output, config, item, alpha, data, progress)
                    done += 1
                    progress.update(f'HISTORY done {item[0]} repeat={item[4] + 1} fold={item[5] + 1} alpha={alpha}',
                                    completed=done)
                progress.update('Rebuilding fold_results.jsonl and exported tables')
                export_run(output, manifest)
                finished = done == len(coupled_items) + len(history)
                manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
                record = next(a for a in manifest['augmentations'] if a['version'] == AUGMENTATION)
                record['invocations'][-1].update(finished_utc=utc_now(), new_coupled_checks=len(verification))
                record['status'] = 'complete' if finished and not args.datasets else 'partial'
                atomic_json(output / 'manifest.json', manifest)
                progress.update('Complete' if finished else 'Stopped at --limit', status=record['status'])
        print(f'Augmented run: {output}')
    except (ValueError, FileExistsError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
