"""RSMN core algorithms and PyTorch backbones."""

from .adaptive_cpra import align_selected_sources, compute_tau
from .backbones import DeepConvNet, EEGConformer, EEGNet, ShallowConvNet, build_backbone
from .config import CoreConfig
from .data import SubjectData
from .rame import (
    build_hemisphere_mirror_permutation,
    compute_rame_stats,
    fuse_mirror_probabilities,
    mirror_eeg,
    rame_losses,
)
from .riemannian_statistics import compute_subject_riemannian_stats
from .source_reliability import compute_source_profiles
from .source_selection import select_patient_sources, sources_for_mode
from .standardizer import fit_channel_standardizer
from .tcpa import TCPAOutput, frozen_bn_running_stats, tcpa_forward, tcpa_loss

__all__ = [
    "CoreConfig",
    "SubjectData",
    "EEGNet",
    "ShallowConvNet",
    "DeepConvNet",
    "EEGConformer",
    "build_backbone",
    "compute_subject_riemannian_stats",
    "compute_source_profiles",
    "select_patient_sources",
    "sources_for_mode",
    "compute_tau",
    "align_selected_sources",
    "fit_channel_standardizer",
    "tcpa_loss",
    "TCPAOutput",
    "tcpa_forward",
    "build_hemisphere_mirror_permutation",
    "compute_rame_stats",
    "mirror_eeg",
    "rame_losses",
    "fuse_mirror_probabilities",
    "frozen_bn_running_stats",
]
