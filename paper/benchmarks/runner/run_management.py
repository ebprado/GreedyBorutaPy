"""Fresh experiment directories, atomic checkpoints, and durable progress logs."""
import hashlib
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


PROTOCOL = 'publication-v6-fresh-cached-scoring'


def clean_json(value):
    import numpy as np
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w', encoding='utf-8') as output:
        json.dump(clean_json(value), output, indent=2, allow_nan=False)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temp, path)


def code_fingerprints():
    directory = Path(__file__).resolve().parent
    names = ['experiment_run.py', 'run_management.py', 'publication_benchmark.py',
             'synthetic_suite.py', 'GreedyBorutaParallel.py', 'BorutaLocal.py',
             'data_loader.py', 'experiment_metrics.py', 'noninferiority.py']
    return {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in names}


def workload(config):
    realizations = sum(len(config['data_seeds']) if kind == 'synthetic' else 1
                       for kind, spec in config['datasets'].values())
    partitions = realizations * config['repeats'] * config['folds']
    selectors = sum(m['name'] != 'NoSelection' for m in config['methods'])
    alphas = len(config['alphas'])
    policies = len(config['methods']) + 1  # Vanilla contributes strict and rough-fix.
    return dict(dataset_names=len(config['datasets']), dataset_realizations=realizations,
                cv_partitions=partitions, selector_fits=partitions * alphas * selectors,
                method_jobs=partitions * (1 + alphas * selectors),
                result_rows=partitions * alphas * policies,
                downstream_fits_upper_bound=partitions * (1 + alphas * (policies - 1)),
                no_selection_fits=partitions,
                baseline_fits_saved=partitions * (alphas - 1))


def fresh_run(config, output=None):
    if output is None:
        root = Path(__file__).resolve().parents[1] / 'data' / 'publication_runs'
        output = root / datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'checkpoints').mkdir()
    manifest = dict(protocol=PROTOCOL, created_utc=datetime.now(timezone.utc).isoformat(),
                    config=config, workload=workload(config), code=code_fingerprints())
    atomic_json(output / 'manifest.json', manifest)
    atomic_json(output / 'progress.json', dict(status='prepared', completed_jobs=0,
                                             total_jobs=manifest['workload']['method_jobs']))
    return output


def open_run(directory):
    directory = Path(directory).resolve()
    manifest_path = directory / 'manifest.json'
    try:
        manifest_text = manifest_path.read_text(encoding='utf-8')
    except FileNotFoundError:
        raise ValueError(
            f'No prepared run found: {manifest_path}. '
            '--run/--resume/--export require an existing manifest. '
            'To create and start a fresh run, use: '
            '.\\benchmarks\\publication\\run.ps1 --cpu-budget -1 --outer-jobs 4 '
            '(without --run). To prepare without starting, add --prepare.'
        ) from None
    manifest = json.loads(manifest_text)
    if manifest.get('protocol') != PROTOCOL:
        raise ValueError('Only this v6 experiment protocol can be run/resumed; old results are never imported.')
    if manifest['code'] != code_fingerprints():
        raise ValueError('Experiment code changed since preparation. Create a fresh run instead of mixing implementations.')
    return directory, manifest


def partition_key(dataset, data_seed, repeat, fold):
    text = json.dumps([dataset, int(data_seed), int(repeat), int(fold)])
    return hashlib.sha256(text.encode()).hexdigest()[:24]


def checkpoint_paths(directory):
    return sorted((Path(directory) / 'checkpoints').glob('*.json'))


def checkpoint_rows(directory):
    for path in checkpoint_paths(directory):
        checkpoint = json.loads(path.read_text(encoding='utf-8'))
        for job in checkpoint['jobs'].values():
            yield from job['rows']


def rebuild_results(directory):
    """Checkpoints are authoritative; reconstruct JSONL after interruption/export."""
    target = Path(directory) / 'fold_results.jsonl'
    temp = target.with_suffix('.jsonl.tmp')
    with temp.open('w', encoding='utf-8') as output:
        for row in checkpoint_rows(directory):
            output.write(json.dumps(row, allow_nan=False) + '\n')
    os.replace(temp, target)
    return target


def duration(seconds):
    seconds = max(0, int(seconds))
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{days}d {hours:02}:{minutes:02}:{seconds:02}' if days else f'{hours:02}:{minutes:02}:{seconds:02}'


class Progress:
    def __init__(self, directory, total, completed=0, heartbeat=30):
        self.directory = Path(directory)
        self.total, self.completed = total, completed
        self.initial_completed = completed
        self.started = time.perf_counter()
        self.stage_started = self.started
        self.stage = 'starting'
        self.heartbeat = heartbeat
        self.lock = threading.RLock()
        self.stop = threading.Event()

    def __enter__(self):
        self.log = (self.directory / 'run.log').open('a', encoding='utf-8', buffering=1)
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)
        self.thread.start()
        self.update('Starting session')
        return self

    def update(self, message, completed=None, status='running'):
        with self.lock:
            if completed is not None:
                self.completed = completed
            if status != 'heartbeat':
                self.stage, self.stage_started = message, time.perf_counter()
            elapsed = time.perf_counter() - self.started
            finished = self.completed - self.initial_completed
            eta = elapsed / finished * (self.total - self.completed) if finished else None
            prefix = f'[{datetime.now():%Y-%m-%d %H:%M:%S}] {self.completed}/{self.total} jobs ({100*self.completed/self.total:.1f}%)'
            line = f'{prefix} | elapsed {duration(elapsed)} | {message}'
            if eta is not None:
                line += f' | rough rate ETA {duration(eta)}'
            print(line, flush=True)
            self.log.write(line + '\n')
            atomic_json(self.directory / 'progress.json', dict(
                status='running' if status == 'heartbeat' else status,
                completed_jobs=self.completed, total_jobs=self.total,
                session_elapsed_seconds=elapsed, rough_rate_eta_seconds=eta,
                eta_caveat='Extrapolated from this session; later datasets/methods can cost much more.',
                current_stage=self.stage))

    def _heartbeat(self):
        while not self.stop.wait(self.heartbeat):
            with self.lock:
                self.update(f'Still running: {self.stage}; stage elapsed {duration(time.perf_counter()-self.stage_started)}',
                            status='heartbeat')

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop.set()
        self.thread.join()
        if exc_type:
            self.update(f'Interrupted/failed: {exc_value}. Resume this same run to continue.', status='interrupted')
        self.log.close()
