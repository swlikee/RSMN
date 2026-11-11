"""Source reliability from class geometry and optional cross-session LDA."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score

from .data import assert_channel_consistency
from .rame import build_hemisphere_mirror_permutation
from .riemannian import airm_distance, flatten_symmetric, matrix_log, riemannian_mean
from .riemannian_statistics import compute_subject_riemannian_stats


@dataclass(frozen=True)
class SourceProfile:
    source_id: str
    class_means: dict[int, np.ndarray]
    reliability_raw: float
    reliability: float
    between: float
    within: float
    session_drift: float
    q_session: float | None
    covariances: np.ndarray
    channel_names: tuple[str, ...]
    mirror_distance: float | None = None


def _safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _cross_session_q(
    covs: np.ndarray, y: np.ndarray, sessions: np.ndarray, eps: float
) -> float | None:
    unique = sorted(set(sessions.tolist()))
    if len(unique) < 2:
        return None
    features = np.stack([flatten_symmetric(matrix_log(c, eps)) for c in covs])
    scores = []
    for session in unique:
        test = sessions == session
        train = ~test
        if len(set(y[train].tolist())) < 2 or len(set(y[test].tolist())) < 2:
            continue
        if int(train.sum()) <= 2:
            continue  # LDA requires more training samples than classes.
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        clf.fit(features[train], y[train])
        scores.append(float(balanced_accuracy_score(y[test], clf.predict(features[test]))))
    return _safe_mean(scores) if scores else None


def _session_drift(
    covs: np.ndarray, y: np.ndarray, sessions: np.ndarray, means: dict[int, np.ndarray], eps: float
) -> float:
    distances: list[float] = []
    for cls in (0, 1):
        for session in sorted(set(sessions.tolist())):
            mask = (y == cls) & (sessions == session)
            if int(mask.sum()) >= 2:
                distances.append(airm_distance(riemannian_mean(covs[mask], eps), means[cls], eps))
    return _safe_mean(distances)


def robust_norm(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("robust_norm requires a nonempty finite vector")
    med = float(np.median(values))
    q75, q25 = np.percentile(values, [75, 25])
    iqr = float(q75 - q25)
    if iqr <= 1e-12:
        return np.zeros_like(values)
    return (values - med) / iqr


def compute_source_profiles(sources, config) -> tuple[list[SourceProfile], str]:
    """Compute reliability without filesystem caches or experiment dependencies.

    If any source lacks a valid cross-session classification score, the entire
    pool uses normalized geometric reliability. This excludes the separate
    classification-score term, while valid sources retain their session-drift
    penalties. Drift is zero only for sources without a valid session score.
    """
    if not sources:
        raise ValueError("At least one source is required")
    if len({source.subject_id for source in sources}) != len(sources):
        raise ValueError("Source IDs must be unique")
    channels = sources[0].channel_names
    profiles = []
    for subject in sources:
        assert_channel_consistency(
            channels, subject.channel_names, context=f"Source {subject.subject_id}"
        )
        mapping = build_hemisphere_mirror_permutation(subject.channel_names)
        stats = compute_subject_riemannian_stats(subject, config, mapping)
        between = airm_distance(stats.class_centers[0], stats.class_centers[1], config.SPD_EPSILON)
        within = 0.5 * sum(stats.class_dispersion.values())
        q = _cross_session_q(stats.covariances, subject.y, subject.session_ids, config.SPD_EPSILON)
        drift = (
            _session_drift(
                stats.covariances,
                subject.y,
                subject.session_ids,
                stats.class_centers,
                config.SPD_EPSILON,
            )
            if q is not None
            else 0.0
        )
        raw = between / (within + config.LAMBDA_SESSION * drift + 1e-12)
        profiles.append(
            SourceProfile(
                subject.subject_id,
                stats.class_centers,
                raw,
                0.0,
                between,
                within,
                drift,
                q,
                stats.covariances,
                stats.channel_names,
                stats.mirror_distance,
            )
        )
    geometry = robust_norm(np.asarray([p.reliability_raw for p in profiles]))
    if all(p.q_session is not None for p in profiles):
        session = robust_norm(np.asarray([p.q_session for p in profiles]))
        reliability = 0.60 * session + 0.40 * geometry
        mode = "cross_session_plus_geometry"
    else:
        reliability, mode = geometry, "geometry_only"
    return [replace(p, reliability=float(reliability[i])) for i, p in enumerate(profiles)], mode
