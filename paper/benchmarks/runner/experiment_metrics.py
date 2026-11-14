"""Feature recovery, stability, and sampled process-tree resident memory."""
import threading
import time
from itertools import combinations
import numpy as np
import psutil


def recovery_metrics(selected, relevant, p):
    selected, relevant = set(selected), set(relevant)
    tp, fp = len(selected & relevant), len(selected - relevant)
    fn, tn = len(relevant - selected), p - len(selected | relevant)
    return dict(TP=tp, FP=fp, FN=fn, TN=tn,
                Recall=tp / len(relevant) if relevant else np.nan,
                Precision=tp / len(selected) if selected else np.nan,
                FPR=fp / (p - len(relevant)) if p > len(relevant) else np.nan,
                FDP=fp / len(selected) if selected else 0.0,
                FP_Original_Fraction=fp / p,
                F1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan)


def selection_stability(selections, p):
    """Raw pairwise Jaccard and feature frequencies; no significance claim."""
    sets = [set(s) for s in selections]
    pairs = [len(a & b) / len(a | b) if a | b else 1.0 for a, b in combinations(sets, 2)]
    frequency = np.zeros(p)
    for selected in sets:
        frequency[list(selected)] += 1
    return (float(np.mean(pairs)) if pairs else np.nan,
            frequency / len(sets) if sets else frequency)


class PeakRSS:
    """Sample parent + descendant RSS every 250 ms; shared pages may be counted twice.

    This is a process-tree RSS estimate, not allocated bytes or an exact peak.
    Includes idle/reused joblib workers. Report baseline, absolute peak and delta.
    Walking descendants costs ~8 ms of GIL time on Windows (it enumerates every
    system process), so the tree is re-listed only every `refresh` seconds.
    """
    def __init__(self, interval=0.25, refresh=1.0):
        self.interval = interval
        self.refresh = refresh
        self.process = psutil.Process()
        self.stop = threading.Event()

    def _tree(self):
        self.listed = time.perf_counter()
        self.processes = [self.process] + self.process.children(recursive=True)

    def _rss(self):
        if time.perf_counter() - self.listed >= self.refresh:
            self._tree()
        total = 0
        for process in self.processes:
            try:
                total += process.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return total

    def _sample(self):
        while not self.stop.wait(self.interval):
            self.peak = max(self.peak, self._rss())

    def __enter__(self):
        self._tree()
        self.baseline = self.peak = self._rss()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        self._tree()
        self.peak = max(self.peak, self._rss())

    def as_dict(self):
        return dict(RSS_Baseline_MB=self.baseline / 2**20,
                    RSS_Peak_MB=self.peak / 2**20,
                    RSS_Peak_Increase_MB=max(0, self.peak - self.baseline) / 2**20)


class CPUTime:
    """User + system CPU seconds used by this process and its descendants in the block.

    Descendants are listed at entry and exit only; reused joblib workers are counted,
    but a worker that both starts and exits inside the block would be missed.
    CPU seconds / wall seconds gives the mean number of busy cores.
    """
    def __init__(self):
        self.process = psutil.Process()

    def _snapshot(self):
        times = {}
        for process in [self.process] + self.process.children(recursive=True):
            try:
                used = process.cpu_times()
                # (pid, create_time) guards against a recycled pid.
                times[process.pid, process.create_time()] = used.user + used.system
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return times

    def __enter__(self):
        self.start = self._snapshot()
        return self

    def __exit__(self, *args):
        end = self._snapshot()
        self.seconds = sum(value - self.start.get(key, 0.0) for key, value in end.items())
