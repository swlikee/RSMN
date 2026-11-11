"""Class centers and dispersion used by source matching, CPRA and RAME."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .covariance import oas_covariance_trials
from .rame import MirrorMapping, mirror_distance_from_stats
from .riemannian import airm_distance, riemannian_mean


@dataclass(frozen=True)
class SubjectRiemannianStats:
    class_centers: dict[int, np.ndarray]
    class_dispersion: dict[int, float]
    covariances: np.ndarray
    channel_names: tuple[str, ...]
    mirror_distance: float | None = None


def compute_subject_riemannian_stats(
    subject, config, mapping: MirrorMapping | None = None
) -> SubjectRiemannianStats:
    covs = oas_covariance_trials(
        subject.X, epsilon=config.SPD_EPSILON, trace_normalization=config.TRACE_NORMALIZATION
    )
    centers, dispersion = {}, {}
    for cls in (0, 1):
        class_covs = covs[subject.y == cls]
        center = riemannian_mean(
            class_covs,
            config.SPD_EPSILON,
            config.RIEMANNIAN_MEAN_MAX_ITER,
            config.RIEMANNIAN_MEAN_TOL,
        )
        centers[cls] = center
        dispersion[cls] = float(
            np.mean([airm_distance(cov, center, config.SPD_EPSILON) for cov in class_covs])
        )
    stats = SubjectRiemannianStats(centers, dispersion, covs, tuple(subject.channel_names))
    if mapping is None:
        return stats
    distance = mirror_distance_from_stats(stats, mapping, config.SPD_EPSILON)
    return SubjectRiemannianStats(centers, dispersion, covs, stats.channel_names, distance)
