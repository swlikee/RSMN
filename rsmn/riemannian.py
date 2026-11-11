from __future__ import annotations

import numpy as np


def ensure_spd(A: np.ndarray, eps: float) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] == 0:
        raise ValueError("Expected a nonempty square matrix")
    if not np.isfinite(A).all() or not np.isfinite(eps) or eps <= 0:
        raise ValueError("Matrix must be finite and eps must be positive and finite")
    A = 0.5 * (A + A.T)
    vals, vecs = np.linalg.eigh(A)
    vals = np.clip(vals, eps, None)
    out = (vecs * vals) @ vecs.T
    if not np.isfinite(out).all():
        raise FloatingPointError("SPD projection produced non-finite values")
    cond = float(vals.max() / vals.min())
    if not np.isfinite(cond):
        raise FloatingPointError("SPD matrix has invalid condition number")
    return 0.5 * (out + out.T)


def matrix_power(A: np.ndarray, power: float, eps: float) -> np.ndarray:
    A = ensure_spd(A, eps)
    vals, vecs = np.linalg.eigh(A)
    vals = np.clip(vals, eps, None)
    out = (vecs * (vals**power)) @ vecs.T
    return 0.5 * (out + out.T)


def matrix_sqrt(A: np.ndarray, eps: float) -> np.ndarray:
    return matrix_power(A, 0.5, eps)


def matrix_inv_sqrt(A: np.ndarray, eps: float) -> np.ndarray:
    return matrix_power(A, -0.5, eps)


def matrix_log(A: np.ndarray, eps: float) -> np.ndarray:
    A = ensure_spd(A, eps)
    vals, vecs = np.linalg.eigh(A)
    vals = np.clip(vals, eps, None)
    out = (vecs * np.log(vals)) @ vecs.T
    return 0.5 * (out + out.T)


def matrix_exp(S: np.ndarray) -> np.ndarray:
    S = 0.5 * (S + S.T)
    vals, vecs = np.linalg.eigh(S)
    out = (vecs * np.exp(vals)) @ vecs.T
    return 0.5 * (out + out.T)


def airm_distance(A: np.ndarray, B: np.ndarray, eps: float) -> float:
    inv = matrix_inv_sqrt(A, eps)
    M = ensure_spd(inv @ B @ inv, eps)
    return float(np.linalg.norm(matrix_log(M, eps), ord="fro"))


def riemannian_mean(
    covs: np.ndarray, eps: float, max_iter: int = 25, tol: float = 1e-7
) -> np.ndarray:
    if len(covs) == 0:
        raise ValueError("Cannot compute Riemannian mean of empty set")
    if max_iter < 1 or not np.isfinite(tol) or tol <= 0:
        raise ValueError("max_iter and tol must be positive")
    G = ensure_spd(np.mean(covs, axis=0), eps)
    for _ in range(max_iter):
        sqrt_G = matrix_sqrt(G, eps)
        inv_sqrt_G = matrix_inv_sqrt(G, eps)
        tangent = np.mean([matrix_log(inv_sqrt_G @ C @ inv_sqrt_G, eps) for C in covs], axis=0)
        norm = float(np.linalg.norm(tangent, ord="fro"))
        G = ensure_spd(sqrt_G @ matrix_exp(tangent) @ sqrt_G, eps)
        if norm < tol:
            break
    return G


def geodesic_interpolation(A: np.ndarray, B: np.ndarray, alpha: float, eps: float) -> np.ndarray:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    sqrt_A = matrix_sqrt(A, eps)
    inv_sqrt_A = matrix_inv_sqrt(A, eps)
    return ensure_spd(sqrt_A @ matrix_power(inv_sqrt_A @ B @ inv_sqrt_A, alpha, eps) @ sqrt_A, eps)


def tangent_vector_at_identity(A: np.ndarray, B: np.ndarray, eps: float) -> np.ndarray:
    inv = matrix_inv_sqrt(A, eps)
    return matrix_log(inv @ B @ inv, eps)


def flatten_symmetric(A: np.ndarray) -> np.ndarray:
    idx = np.triu_indices_from(A)
    return np.asarray(A[idx], dtype=np.float64)
