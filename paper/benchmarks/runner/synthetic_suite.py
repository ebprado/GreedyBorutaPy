"""Controlled generators with explicit population relevance and stable column IDs."""
import numpy as np


def synthetic_scenarios():
    base = dict(n_samples=1000, n_features=500, n_informative=5,
                n_correlated=20, correlation=0.8, signal=2.0, kind="linear")
    scenarios = {"reference": base.copy()}
    for key, values in {
        "n_samples": [200, 2000], "n_features": [100, 2000],
        "signal": [0.5, 5.0], "n_informative": [2, 30],
        "correlation": [0.2, 0.95], "kind": ["mixed", "xor", "null"],
    }.items():
        for value in values:
            case = dict(base, **{key: value})
            if value in ("xor", "null"):
                case["n_correlated"] = 0
                case["n_informative"] = 2 if value == "xor" else 0
            scenarios[f"{key}_{value}"] = case
    return scenarios


def generate_synthetic(spec, seed):
    """Relevant = direct predictors plus noisy correlated proxies; noise is independent.

    Mixed cases alternate continuous, binary and four-level ordered predictors.
    XOR uses two balanced binary predictors, each marginally uninformative.
    The all-null target is independent of every predictor.
    """
    rng = np.random.RandomState(seed)
    n, p = spec["n_samples"], spec["n_features"]
    k, q = spec["n_informative"], spec.get("n_correlated", 0)
    rho, signal, kind = spec.get("correlation", 0.8), spec.get("signal", 2), spec.get("kind", "linear")
    if n < 10 or p < 1 or k < 0 or q < 0 or k + q > p or not 0 <= rho < 1:
        raise ValueError("Invalid synthetic dimensions or correlation")
    if kind not in {"null", "xor", "mixed", "linear"}:
        raise ValueError("Unknown synthetic kind")
    X = rng.normal(size=(n, p))
    if kind == "null":
        return X, rng.randint(0, 2, n), set()
    if k < 1 or signal <= 0:
        raise ValueError("Signal scenarios require informative features and positive signal")
    if kind == "xor":
        if k != 2 or q != 0:
            raise ValueError("XOR requires exactly two informative features and no proxies")
        X[:, :2] = rng.randint(0, 2, (n, 2))
        y = np.logical_xor(X[:, 0], X[:, 1]).astype(int)
        return X, y, {0, 1}
    if kind == "mixed":
        X[:, 1::3] = (X[:, 1::3] > 0).astype(float)
        X[:, 2::3] = np.digitize(X[:, 2::3], [-0.67449, 0, 0.67449])
    direct = X[:, :k]
    # Population centering/scaling constants, not sample-fitted transformations.
    if kind == "mixed":
        direct = direct.copy()
        direct[:, 1::3] = (direct[:, 1::3] - 0.5) / 0.5
        direct[:, 2::3] = (direct[:, 2::3] - 1.5) / np.sqrt(1.25)
    for j in range(q):
        X[:, k + j] = rho * direct[:, j % k] + np.sqrt(1 - rho ** 2) * rng.normal(size=n)
    logits = np.clip(signal * direct.sum(axis=1) / np.sqrt(k), -30, 30)
    y = rng.binomial(1, 1 / (1 + np.exp(-logits)))
    relevant = set(range(k)) | (set(range(k, k + q)) if rho > 0 else set())
    return X, y, relevant
