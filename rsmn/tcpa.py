"""Target-Class Prototype Alignment: patient-anchored BN and prototype matching."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


def class_prototypes(features: torch.Tensor, labels: torch.Tensor) -> dict[int, torch.Tensor]:
    if features.ndim != 2 or labels.shape != (len(features),):
        raise ValueError("Expected features [B,D] and labels [B]")
    if not bool(((labels == 0) | (labels == 1)).all()):
        raise ValueError("TCPA requires binary labels 0 and 1")
    return {cls: features[labels == cls].mean(0) for cls in (0, 1) if bool((labels == cls).any())}


def normalized_prototype_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Squared unit-prototype distance, divided by feature dimension exactly once."""
    d = a.numel()
    return F.mse_loss(F.normalize(a, dim=0), F.normalize(b, dim=0), reduction="sum") / d


def tcpa_loss(
    target_features: torch.Tensor, target_labels: torch.Tensor, source_batches
) -> torch.Tensor:
    """Average dimension-normalized distance over available (source, class) pairs.

    source_batches contains (features, labels) pairs. Both domains retain their
    gradients. The result includes 1/d and is weighted by CoreConfig.TCPA_WEIGHT
    when forming the joint objective.
    """
    target = class_prototypes(target_features, target_labels)
    losses = []
    for features, labels in source_batches:
        if features.ndim != 2 or features.shape[1] != target_features.shape[1]:
            raise ValueError("Source and target feature dimensions must match")
        source = class_prototypes(features, labels)
        for cls in target.keys() & source.keys():
            losses.append(normalized_prototype_distance(target[cls], source[cls]))
    return torch.stack(losses).mean() if losses else target_features.sum() * 0.0


@contextmanager
def frozen_bn_running_stats(model: nn.Module):
    """Use each training batch's moments without updating BN running buffers."""
    states = []
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            counter = module.num_batches_tracked
            states.append((module, module.momentum, None if counter is None else counter.clone()))
            module.momentum = 0.0
    try:
        yield
    finally:
        for module, momentum, counter in states:
            module.momentum = momentum
            if counter is not None:
                module.num_batches_tracked.copy_(counter)


@dataclass
class TCPAOutput:
    target_logits: torch.Tensor
    source_logits: list[torch.Tensor]
    loss: torch.Tensor
    augmented_target_logits: torch.Tensor | None = None


def tcpa_forward(
    model: nn.Module,
    target_x: torch.Tensor,
    target_labels: torch.Tensor,
    source_batches,
    *,
    augmented_target_x: torch.Tensor | None = None,
) -> TCPAOutput:
    """Run TCPA on standardized patient inputs and CPRA-aligned source inputs.

    source_batches contains (input, label) pairs. Only the original and optional
    augmented patient pass update running BN statistics. Prototypes use the
    original patient features and current source features; neither is detached.
    The returned loss includes 1/d. Weight it externally with config.TCPA_WEIGHT.
    This is one differentiable forward operation, without an optimizer or loop.
    """
    if not model.training:
        raise ValueError("TCPA requires model.train(); ordinary inference uses the backbone")
    source_batches = list(source_batches)
    if not source_batches:
        raise ValueError("TCPA requires at least one source batch")
    target_logits, target_features = model(target_x, return_features=True)
    augmented_logits = None
    if augmented_target_x is not None:
        augmented_logits = model(augmented_target_x)
    source_logits, source_features = [], []
    with frozen_bn_running_stats(model):
        for source_x, source_labels in source_batches:
            logits, features = model(source_x, return_features=True)
            source_logits.append(logits)
            source_features.append((features, source_labels))
    loss = tcpa_loss(target_features, target_labels, source_features)
    return TCPAOutput(target_logits, source_logits, loss, augmented_logits)
