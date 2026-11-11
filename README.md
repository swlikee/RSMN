# RSMN

**Patient-Specific Riemannian Source Matching Network**

Core algorithms and PyTorch backbones extracted from the research project. The only dependencies are NumPy, scikit-learn, and PyTorch.

## Code Scope

| File | Functionality |
| --- | --- |
| `rsmn/backbones.py` | EEGNet, ShallowConvNet, DeepConvNet, EEG Conformer |
| `rsmn/source_reliability.py` | Inter-class separability, intra-class dispersion, session drift, and source reliability |
| `rsmn/source_selection.py` | Patient-specific source matching and Top-K selection |
| `rsmn/adaptive_cpra.py` | Adaptive class-conditional partial Riemannian alignment (CPRA) |
| `rsmn/tcpa.py` | TCPA: patient-anchored BN, class prototypes, dimension-normalized loss, and a unified forward interface |
| `rsmn/rame.py` | Hemisphere mirroring, reliability estimation, mirror cross-entropy, consistency regularization, and probability fusion (RAME) |
| `rsmn/riemannian.py` | SPD matrix operations, AIRM distance, Riemannian means, and geodesic interpolation |
| `rsmn/covariance.py` | Per-trial OAS covariance estimation |
| `rsmn/riemannian_statistics.py` | Class centers, intra-class dispersion, and mirror distance |
| `rsmn/standardizer.py` | Channel-wise Z-score normalization fitted on patient training data |
| `rsmn/config.py` / `data.py` | Core parameters and in-memory data structures |


## Installation and Backbones

Run from this directory:

```bash
python -m pip install -r requirements.txt
```

```python
import torch
from rsmn import build_backbone

model = build_backbone("eegnet", n_channels=29, n_times=1024)
model.eval()
with torch.no_grad():
    logits, features = model(torch.randn(2, 1, 29, 1024), return_features=True)
# logits: [2, 2]; features: [2, feature_dim]
```

Available names: `eegnet`, `shallowconvnet`, `deepconvnet`, and `eegconformer`. Additional architecture parameters are passed as keyword arguments, such as `dropout=0.5`. Inputs must have shape `[B, 1, C, T]`, with C and T matching the constructor arguments; with its default settings, DeepConvNet requires T to be at least 441. `return_features=True` returns the features before the complete classification head. The `separable` attribute denotes the backend module in all backbones.

## Using the Core Modules

The `X_source` and `X_patient_train` variables below are preprocessed arrays of shape `[N, C, T]`, and `y_source` and `y_patient_train` are the corresponding label arrays of shape `[N]`; both source and patient data must contain classes 0 and 1. All data must share the same channel order, sampling rate, time window, and label semantics. `channel_names` uses standard electrode names, such as `("C3", "C4", "Cz")`.

```python
from rsmn import (
    CoreConfig, SubjectData, compute_source_profiles,
    compute_subject_riemannian_stats, select_patient_sources,
    sources_for_mode, compute_tau, align_selected_sources,
    build_hemisphere_mirror_permutation, compute_rame_stats,
    fit_channel_standardizer,
)

config = CoreConfig()
sources = [SubjectData("source_01", X_source, y_source, channel_names)]
patient = SubjectData("patient_01", X_patient_train, y_patient_train, channel_names)
profiles, reliability_mode = compute_source_profiles(sources, config)
mapping = build_hemisphere_mirror_permutation(channel_names)
patient_stats = compute_subject_riemannian_stats(patient, config, mapping)
selection = select_patient_sources(
    patient.subject_id, patient.X, patient.y, profiles, config, patient_stats,
    patient_channel_names=patient.channel_names,
)
selected_ids = sources_for_mode(selection.ranked_source_ids, "top1")
tau = compute_tau(selection.patient_class_means, profiles, config)
alignment = align_selected_sources(
    selected_ids,
    {s.subject_id: s for s in sources},
    {p.source_id: p for p in profiles},
    selection.patient_class_means, tau, config,
    patient_channel_names=patient.channel_names,
)
scaler = fit_channel_standardizer(X_patient_train)
aligned_source = scaler.transform(alignment.aligned_sources[selected_ids[0]])
rame_stats = compute_rame_stats(patient_stats, profiles, mapping, config)
```

Both `select_patient_sources` and `align_selected_sources` require the keyword argument `patient_channel_names`. Source profiles and subject statistics retain the electrode names associated with their covariance matrices. These entry points compare electrode names and order after ignoring case and surrounding whitespace; inconsistent names, order, or channel counts raise `ValueError`. Channels are never reordered automatically. Supply the actual electrode names in the order of each data array's channel axis.

When session information is available for a source, pass `session_ids` to `SubjectData`. If every source has a valid cross-session classification score, reliability is `0.60 × normalized session score + 0.40 × normalized geometry score`. If any source lacks a valid score, the entire source pool uses normalized geometric reliability `N(J)`. Here, `J = between / (within + LAMBDA_SESSION × session_drift + 1e-12)`: sources with valid cross-session scores retain their session-drift penalties, while drift is set to zero only for sources without a valid score. Thus, `geometry_only` excludes the separate classification-score term without discarding available drift information, consistent with the paper's current definition.

`sources_for_mode(ranked, "topK")` supports any positive integer K, such as `top1`, `top5`, or `top17`; K must not exceed `len(ranked)`. `all` returns all sources; an empty source pool, zero or negative counts, non-integer counts, and requests exceeding the total number of sources raise `ValueError`.


## TCPA and RAME

In the paper, **TCPA (Target-Class Prototype Alignment)** combines patient-anchored BN with prototype alignment. `tcpa_forward` first processes the original patient batch and the optional augmented patient batch, followed by each source batch; source passes use their own batch means and variances and the shared affine parameters, but do not update BN running statistics. Prototypes are computed from the original patient features and the corresponding source features in the current batch, and gradients are retained for both domains.

```python
from rsmn import CoreConfig, tcpa_forward

config = CoreConfig(TCPA_WEIGHT=0.10)
model = build_backbone("eegnet", n_channels=29, n_times=1024)
model.train()
target_x = torch.randn(4, 1, 29, 1024)
source_x = torch.randn(4, 1, 29, 1024)
labels = torch.tensor([0, 1, 0, 1])
output = tcpa_forward(model, target_x, labels, [(source_x, labels)])
weighted_tcpa_loss = config.TCPA_WEIGHT * output.loss
# output.target_logits: patient logits
# output.source_logits: source logits in input order
# output.loss: the TCPA prototype loss defined in the paper
```

This is a single differentiable forward operation without a training loop or optimizer. In actual use, pass standardized patient data and source data that have undergone CPRA and standardization using patient statistics. The optional `augmented_target_x` parameter accepts a previously constructed augmented patient batch, whose logits are available in `output.augmented_target_logits`; the augmented pass updates BN but does not replace the original patient prototypes.

`CoreConfig.TCPA_WEIGHT` corresponds to `lambda_TCPA` in the paper's joint objective and defaults to `0.10`; set it to the coefficient used in the corresponding experiment. This coefficient only weights the total loss and does not affect the update policy for patient BN statistics.

`tcpa_loss(target_features, target_labels, source_batches)` is the prototype loss interface within the same TCPA module. For a single source, it computes exactly `sum(||normalized_source - normalized_target||²) / (d × number of valid classes)` and returns zero when there are no shared classes. For multiple sources, it averages the distances equally over valid (source, class) pairs, extending the single-source formula; the paper's main experiment still uses Top-1.

The prototype distance explicitly uses `reduction="sum"` inside the module and is then divided by the actual feature dimension `d`, which is equivalent to the original default mean reduction in MSE. `output.loss` already includes `1/d` once; during training, multiply it directly by `config.TCPA_WEIGHT` without dividing by `d` again outside the module. Coefficient values do not need conversion. This document uses the original paper's loss definition with `1/(d × number of valid classes)`.

RAME uses `rame_losses(...)`, which returns `(weighted_loss, mirror_ce, mirror_js)`; binary mirrored labels are `1 - y`. Mirroring is performed after standardization using patient statistics, without further standardization, and mirrored passes also leave BN running statistics unchanged. `mirrored_logits` should come from a forward pass on mirrored inputs with `model.eval()` enabled. The BN protection utility `frozen_bn_running_stats` is shared by TCPA and RAME.

