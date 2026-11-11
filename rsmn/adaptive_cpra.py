from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import assert_channel_consistency, normalize_channel_names
from .riemannian import airm_distance, geodesic_interpolation, matrix_inv_sqrt, matrix_sqrt


@dataclass(frozen=True)
class AlignmentResult:
    aligned_sources: dict[str, np.ndarray]
    rows: list[dict[str, object]]
    transforms: dict[tuple[str, int], np.ndarray]


def compute_tau(patient_means: dict[int, np.ndarray], profiles: list, config) -> dict[int, float]:
    if not profiles:
        raise ValueError("At least one source profile is required to estimate tau")
    tau = {}
    for cls in (0, 1):
        distances = [
            airm_distance(profile.class_means[cls], patient_means[cls], config.SPD_EPSILON)
            for profile in profiles
        ]
        tau[cls] = max(float(np.median(distances)), config.TAU_EPSILON)
    return tau


def align_selected_sources(
    selected_ids: list[str],
    source_by_id: dict[str, object],
    profiles_by_id: dict[str, object],
    patient_means: dict[int, np.ndarray],
    tau: dict[int, float],
    config,
    *,
    patient_channel_names: tuple[str, ...],
) -> AlignmentResult:
    """Align labeled source trials; patient trials are left in their own coordinates."""
    if not selected_ids or len(set(selected_ids)) != len(selected_ids):
        raise ValueError("selected_ids must be nonempty and unique")
    channels = normalize_channel_names(patient_channel_names)
    n_channels = len(channels)
    if any(np.shape(patient_means[cls]) != (n_channels, n_channels) for cls in (0, 1)):
        raise ValueError("Patient class means must have shape [C, C] matching patient channels")
    if any(not np.isfinite(tau[cls]) or tau[cls] <= 0 for cls in (0, 1)):
        raise ValueError("Each class tau must be positive and finite")
    aligned: dict[str, np.ndarray] = {}
    rows: list[dict[str, object]] = []
    transforms: dict[tuple[str, int], np.ndarray] = {}
    for source_id in selected_ids:
        source = source_by_id[source_id]
        profile = profiles_by_id[source_id]
        assert_channel_consistency(channels, source.channel_names, context=f"Source {source_id}")
        assert_channel_consistency(
            channels, profile.channel_names, context=f"Source profile {source_id}"
        )
        if source.X.ndim != 3 or source.X.shape[1] != n_channels:
            raise ValueError(
                f"Source {source_id}: X must have shape [N, C, T] matching patient channels"
            )
        if any(np.shape(profile.class_means[cls]) != (n_channels, n_channels) for cls in (0, 1)):
            raise ValueError(f"Source profile {source_id}: class means must have shape [C, C]")
        if source.y.shape != (len(source.X),) or not np.isin(source.y, [0, 1]).all():
            raise ValueError("CPRA requires binary source labels matching X")
        X_new = np.empty_like(source.X, dtype=np.float32)
        for cls in (0, 1):
            Gs = profile.class_means[cls]
            Gp = patient_means[cls]
            d_pre = airm_distance(Gs, Gp, config.SPD_EPSILON)
            alpha = config.ALPHA_MIN + (config.ALPHA_MAX - config.ALPHA_MIN) * d_pre / (
                d_pre + tau[cls]
            )
            alpha = float(np.clip(alpha, config.ALPHA_MIN, config.ALPHA_MAX))
            if not config.USE_ADAPTIVE_CPRA:
                alpha = 0.0
            if not config.ALPHA_MIN <= alpha <= config.ALPHA_MAX and config.USE_ADAPTIVE_CPRA:
                raise AssertionError("alpha outside configured bounds")
            B = geodesic_interpolation(Gs, Gp, alpha, config.SPD_EPSILON)
            T = matrix_sqrt(B, config.SPD_EPSILON) @ matrix_inv_sqrt(Gs, config.SPD_EPSILON)
            cond = float(np.linalg.cond(T))
            if not np.isfinite(T).all() or not np.isfinite(cond):
                raise FloatingPointError("Invalid CPRA transform")
            transforms[(source_id, cls)] = T
            mask = source.y == cls
            X_new[mask] = np.einsum("ij,njt->nit", T.astype(np.float32), source.X[mask])
            d_post = airm_distance(B, Gp, config.SPD_EPSILON)
            rows.append(
                {
                    "source_id": source_id,
                    "class_id": cls,
                    "distance_pre": float(d_pre),
                    "tau": float(tau[cls]),
                    "alpha": float(alpha),
                    "distance_post": float(d_post),
                    "transform_condition_number": cond,
                }
            )
        aligned[source_id] = np.ascontiguousarray(X_new)
    return AlignmentResult(aligned, rows, transforms)
