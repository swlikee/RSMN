from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np

from .covariance import oas_covariance_trials
from .data import assert_channel_consistency, normalize_channel_names
from .riemannian import (
    airm_distance,
    flatten_symmetric,
    riemannian_mean,
    tangent_vector_at_identity,
)
from .source_reliability import SourceProfile, robust_norm


@dataclass(frozen=True)
class SourceSelection:
    rows: list[dict[str, object]]
    ranked_source_ids: list[str]
    patient_class_means: dict[int, np.ndarray]
    patient_covariances_train: np.ndarray


def select_patient_sources(
    patient_id: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    profiles: list[SourceProfile],
    config,
    patient_stats=None,
    *,
    patient_channel_names: tuple[str, ...],
) -> SourceSelection:
    """Rank sources after verifying their electrode names and order against the patient."""
    if not profiles:
        raise ValueError("At least one source profile is required")
    channels = normalize_channel_names(patient_channel_names)
    n_channels = len(channels)
    X_train = np.asarray(X_train)
    if X_train.ndim != 3 or X_train.shape[1] != n_channels:
        raise ValueError("X_train must have shape [N, C, T] matching patient_channel_names")
    for profile in profiles:
        assert_channel_consistency(
            channels, profile.channel_names, context=f"Source profile {profile.source_id}"
        )
        if any(np.shape(profile.class_means[cls]) != (n_channels, n_channels) for cls in (0, 1)):
            raise ValueError(
                f"Source profile {profile.source_id}: class means must have shape [C, C]"
            )
    y_train = np.asarray(y_train)
    if y_train.shape != (len(X_train),) or set(y_train.tolist()) != {0, 1}:
        raise ValueError("y_train must match X_train and contain both binary classes")
    if patient_stats is None:
        target_covs = oas_covariance_trials(
            X_train, epsilon=config.SPD_EPSILON, trace_normalization=config.TRACE_NORMALIZATION
        )
        patient_means = {
            cls: riemannian_mean(
                target_covs[y_train == cls],
                config.SPD_EPSILON,
                config.RIEMANNIAN_MEAN_MAX_ITER,
                config.RIEMANNIAN_MEAN_TOL,
            )
            for cls in (0, 1)
        }
    else:
        assert_channel_consistency(
            channels, patient_stats.channel_names, context=f"Patient statistics {patient_id}"
        )
        target_covs = patient_stats.covariances
        patient_means = patient_stats.class_centers
        if np.shape(target_covs) != (len(X_train), n_channels, n_channels):
            raise ValueError("Patient covariances must have shape [N, C, C] matching X_train")
        if any(np.shape(patient_means[cls]) != (n_channels, n_channels) for cls in (0, 1)):
            raise ValueError(
                "Patient class means must have shape [C, C] matching patient channels"
            )
    distances, similarities, reliabilities = [], [], []
    for profile in profiles:
        d = 0.5 * sum(
            airm_distance(profile.class_means[cls], patient_means[cls], config.SPD_EPSILON)
            for cls in (0, 1)
        )
        base = profile.class_means[0]
        vh = flatten_symmetric(
            tangent_vector_at_identity(base, profile.class_means[1], config.SPD_EPSILON)
        )
        vp = flatten_symmetric(
            tangent_vector_at_identity(base, patient_means[1], config.SPD_EPSILON)
            - tangent_vector_at_identity(base, patient_means[0], config.SPD_EPSILON)
        )
        denom = float(np.linalg.norm(vh) * np.linalg.norm(vp))
        sim = 0.0 if denom <= 1e-12 else float(np.clip(np.dot(vh, vp) / denom, -1.0, 1.0))
        distances.append(d)
        similarities.append(sim)
        reliabilities.append(profile.reliability)
    nd, ns, nr = (
        robust_norm(np.asarray(distances)),
        robust_norm(np.asarray(similarities)),
        robust_norm(np.asarray(reliabilities)),
    )
    rows = []
    for i, profile in enumerate(profiles):
        score = (
            -config.SOURCE_DISTANCE_WEIGHT * nd[i]
            + config.SOURCE_DIRECTION_WEIGHT * ns[i]
            + config.SOURCE_RELIABILITY_WEIGHT * nr[i]
        )
        rows.append(
            {
                "patient_id": patient_id,
                "source_id": profile.source_id,
                "class_distance": float(distances[i]),
                "direction_similarity": float(similarities[i]),
                "source_reliability": float(profile.reliability),
                "composite_score": float(score),
            }
        )
    rows.sort(key=lambda row: (-float(row["composite_score"]), str(row["source_id"])))
    ranked = [str(row["source_id"]) for row in rows]
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    return SourceSelection(rows, ranked, patient_means, target_covs)


def sources_for_mode(ranked: list[str], mode: str) -> list[str]:
    """Select topK or all sources, requiring 1 <= K <= the ranked pool size."""
    if not ranked:
        raise ValueError("At least one source must be available")
    if mode == "all":
        count = len(ranked)
    elif isinstance(mode, str) and re.fullmatch(r"top[1-9][0-9]*", mode):
        count = int(mode[3:])
    else:
        raise ValueError("mode must be 'all' or 'topK', where K is a positive integer")
    if count > len(ranked):
        raise ValueError(f"Requested {count} sources, but only {len(ranked)} are available")
    return ranked[:count]
