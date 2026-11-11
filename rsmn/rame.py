from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from .riemannian import airm_distance, ensure_spd
from .tcpa import frozen_bn_running_stats


MIRROR_PAIRS = {
    "fp1": "fp2",
    "af3": "af4",
    "f7": "f8",
    "f5": "f6",
    "f3": "f4",
    "f1": "f2",
    "ft7": "ft8",
    "fc5": "fc6",
    "fc3": "fc4",
    "fc1": "fc2",
    "t7": "t8",
    "c5": "c6",
    "c3": "c4",
    "c1": "c2",
    "tp7": "tp8",
    "cp5": "cp6",
    "cp3": "cp4",
    "cp1": "cp2",
    "p7": "p8",
    "p5": "p6",
    "p3": "p4",
    "p1": "p2",
    "po7": "po8",
    "po3": "po4",
    "o1": "o2",
}
MIRROR_PAIRS.update({right: left for left, right in list(MIRROR_PAIRS.items())})
MIDLINE = {"fpz", "afz", "fz", "fcz", "cz", "cpz", "pz", "poz", "oz"}


@dataclass(frozen=True)
class MirrorMapping:
    mirror_index: np.ndarray
    paired_channels: tuple[tuple[str, str], ...]
    midline_channels: tuple[str, ...]
    unpaired_channels: tuple[str, ...]
    P: np.ndarray


@dataclass(frozen=True)
class RameStats:
    mirror_distance_patient: float
    healthy_mirror_median: float
    healthy_mirror_mad: float
    healthy_mirror_scale: float
    mirror_z: float
    mirror_reliability: float
    test_eta: float


def build_hemisphere_mirror_permutation(ch_names: list[str] | tuple[str, ...]) -> MirrorMapping:
    lower_to_index = {name.strip().lower(): i for i, name in enumerate(ch_names)}
    if not ch_names or "" in lower_to_index or len(lower_to_index) != len(ch_names):
        raise ValueError("Channel names must be nonempty and unique, ignoring case")
    mirror_index = np.arange(len(ch_names), dtype=np.int64)
    paired: list[tuple[str, str]] = []
    midline: list[str] = []
    unpaired: list[str] = []
    seen: set[str] = set()
    for i, name in enumerate(ch_names):
        key = name.strip().lower()
        if key in seen:
            continue
        if key in MIDLINE:
            midline.append(name)
            seen.add(key)
            continue
        partner = MIRROR_PAIRS.get(key)
        if partner is not None and partner in lower_to_index:
            j = lower_to_index[partner]
            mirror_index[i] = j
            mirror_index[j] = i
            paired.append((name, ch_names[j]))
            seen.update({key, partner})
        else:
            unpaired.append(name)
            seen.add(key)
    if sorted(mirror_index.tolist()) != list(range(len(ch_names))):
        raise AssertionError("Mirror index is not a legal permutation")
    P = np.eye(len(ch_names), dtype=np.float64)[mirror_index]
    return MirrorMapping(mirror_index, tuple(paired), tuple(midline), tuple(unpaired), P)


def mirror_eeg(
    x: np.ndarray | torch.Tensor, mirror_index: np.ndarray | torch.Tensor
) -> np.ndarray | torch.Tensor:
    if x.ndim not in (3, 4) or (x.ndim == 4 and x.shape[1] != 1):
        raise ValueError("mirror_eeg expects [N,C,T] or [B,1,C,T]")
    n_channels = x.shape[-2]
    if isinstance(x, torch.Tensor):
        index = torch.as_tensor(mirror_index, dtype=torch.long, device=x.device)
        if index.shape != (n_channels,) or not torch.equal(
            index.sort().values, torch.arange(n_channels, device=x.device)
        ):
            raise ValueError("mirror_index must be a channel permutation")
        return x.index_select(x.ndim - 2, index)
    index_np = np.asarray(mirror_index, dtype=np.int64)
    if index_np.shape != (n_channels,) or not np.array_equal(
        np.sort(index_np), np.arange(n_channels)
    ):
        raise ValueError("mirror_index must be a channel permutation")
    return np.take(x, index_np, axis=x.ndim - 2)


def mirror_spd(G: np.ndarray, P: np.ndarray, eps: float) -> np.ndarray:
    return ensure_spd(P @ G @ P.T, eps)


def mirror_distance_from_stats(stats: Any, mapping: MirrorMapping, eps: float) -> float:
    return airm_distance(
        stats.class_centers[0], mirror_spd(stats.class_centers[1], mapping.P, eps), eps
    )


def compute_rame_stats(
    patient_stats: Any, source_stats: list[Any], mapping: MirrorMapping, config
) -> RameStats:
    distances = np.asarray(
        [float(s.mirror_distance) for s in source_stats if s.mirror_distance is not None],
        dtype=np.float64,
    )
    if distances.size == 0:
        raise ValueError("RAME requires at least one healthy source mirror distance")
    delta_p = mirror_distance_from_stats(patient_stats, mapping, config.SPD_EPSILON)
    if not np.isfinite(distances).all() or not np.isfinite(delta_p):
        raise ValueError("Mirror distances must be finite")
    median = float(np.median(distances))
    mad = float(np.median(np.abs(distances - median)))
    scale = max(float(1.4826 * mad + 1e-12), float(getattr(config, "RAME_MIN_SCALE", 1e-6)))
    z = float((delta_p - median) / scale)
    r = float(np.exp(-np.logaddexp(0.0, z)))
    r = float(np.clip(r, config.RAME_R_MIN, config.RAME_R_MAX))
    eta = float(config.RAME_TEST_ETA_MAX * r) if config.RAME_USE_TEST_ENSEMBLE else 0.0
    return RameStats(delta_p, median, mad, scale, z, r, eta)


def js_divergence_from_probs(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    p = p.float()
    q = q.float()
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    p = p / p.sum(dim=1, keepdim=True)
    q = q / q.sum(dim=1, keepdim=True)
    m = 0.5 * (p + q)
    return 0.5 * (
        F.kl_div(m.log(), p, reduction="batchmean") + F.kl_div(m.log(), q, reduction="batchmean")
    )


def rame_losses(
    model,
    xb: torch.Tensor,
    yb: torch.Tensor,
    logits_orig: torch.Tensor,
    mirror_index: np.ndarray,
    reliability: float,
    lambda_mce: float,
    lambda_me: float,
    *,
    include_mce: bool,
    freeze_bn_context=frozen_bn_running_stats,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reliability-weighted mirror CE and label-swapped JS consistency."""
    if logits_orig.ndim != 2 or logits_orig.shape != (len(xb), 2):
        raise ValueError("RAME requires binary logits of shape [B,2]")
    if yb.shape != (len(xb),) or not bool(((yb == 0) | (yb == 1)).all()):
        raise ValueError("RAME requires binary labels of shape [B]")
    if (
        not 0 <= reliability <= 1
        or not np.isfinite([lambda_mce, lambda_me]).all()
        or min(lambda_mce, lambda_me) < 0
    ):
        raise ValueError(
            "Reliability must be in [0,1] and loss weights must be finite and nonnegative"
        )
    x_mirror = mirror_eeg(xb, mirror_index)
    y_mirror = 1 - yb
    with freeze_bn_context(model):
        logits_mirror = model(x_mirror)
    mce = (
        F.cross_entropy(logits_mirror.float(), y_mirror)
        if include_mce
        else logits_orig.float().new_tensor(0.0)
    )
    p_orig = torch.softmax(logits_orig.float(), dim=1)
    p_mirror_back = torch.softmax(logits_mirror.float(), dim=1)[:, [1, 0]]
    me = js_divergence_from_probs(p_orig, p_mirror_back)
    loss = float(reliability) * (float(lambda_mce) * mce + float(lambda_me) * me)
    return loss, mce, me


def fuse_mirror_probabilities(
    logits: torch.Tensor, mirrored_logits: torch.Tensor, eta: float
) -> torch.Tensor:
    """Fuse original and label-swapped mirrored probabilities (Method inference equation)."""
    if logits.ndim != 2 or logits.shape[1] != 2 or mirrored_logits.shape != logits.shape:
        raise ValueError("Expected original and mirrored binary logits of shape [B,2]")
    if not 0 <= eta <= 1:
        raise ValueError("eta must be in [0,1]")
    original = logits.float().softmax(dim=1)
    mirrored = mirrored_logits.float().softmax(dim=1)[:, [1, 0]]
    return (1 - eta) * original + eta * mirrored
