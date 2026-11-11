from __future__ import annotations

import numpy as np
from sklearn.covariance import oas


def oas_covariance_trials(
    X: np.ndarray,
    *,
    epsilon: float,
    trace_normalization: bool = True,
) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 3 or min(X.shape) < 1 or X.shape[2] < 2:
        raise ValueError("X must have shape [N,C,T], with N,C >= 1 and T >= 2")
    if not np.isfinite(X).all() or not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("X must be finite and epsilon must be positive and finite")
    covs = []
    eye = np.eye(X.shape[1], dtype=np.float64)
    for trial in X:
        cov, _ = oas(trial.T, assume_centered=False)
        cov = 0.5 * (cov + cov.T)
        if trace_normalization:
            tr = float(np.trace(cov))
            if not np.isfinite(tr) or tr <= 0:
                raise FloatingPointError("Invalid covariance trace")
            cov = cov / tr
        cov = cov + epsilon * eye
        evals = np.linalg.eigvalsh(cov)
        if not np.isfinite(cov).all() or float(evals[0]) <= 0:
            raise FloatingPointError("Covariance is not finite SPD")
        covs.append(cov.astype(np.float64, copy=False))
    return np.stack(covs)
