"""Four PyTorch backbones adapted to RSMN input shapes.

EEGNet: Lawhern et al. (2018), doi:10.1088/1741-2552/aace8c.
Shallow/DeepConvNet: Schirrmeister et al. (2017), doi:10.1002/hbm.23730.
EEG Conformer: Song et al., https://github.com/eeyhsong/EEG-Conformer.
The ``separable`` attribute names the backend for all four architectures.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class _Backbone(nn.Module):
    def __init__(self, n_channels: int, n_times: int):
        super().__init__()
        if n_channels < 1 or n_times < 1:
            raise ValueError("n_channels and n_times must be positive")
        self.n_channels, self.n_times = n_channels, n_times

    def _features(self, x):
        return self.separable(self.spatial(self.temporal(x)))

    def _feature_dim(self):
        # Shape inference must not update BN statistics or sample dropout masks.
        training = self.training
        self.eval()
        try:
            with torch.no_grad():
                dummy = torch.zeros(1, 1, self.n_channels, self.n_times)
                return self._features(dummy).flatten(1).shape[1]
        finally:
            self.train(training)

    def forward(self, x, return_features=False):
        if x.ndim != 4 or tuple(x.shape[1:]) != (1, self.n_channels, self.n_times):
            raise ValueError(
                f"Expected [B,1,{self.n_channels},{self.n_times}], got {tuple(x.shape)}"
            )
        features = self._features(x).flatten(1)
        logits = self.classifier(features)
        return (logits, features) if return_features else logits


class _Square(nn.Module):
    def forward(self, x):
        return x.float().square()  # Power/log features stay finite under float16 AMP.


class _SafeLog(nn.Module):
    def forward(self, x):
        return x.clamp_min(1e-6).log()


def _init_convnet(module):
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class ShallowConvNet(_Backbone):
    def __init__(
        self,
        n_channels,
        n_times,
        num_classes=2,
        dropout=0.5,
        n_filters=40,
        kernel_length=25,
        pool_length=75,
        pool_stride=15,
    ):
        super().__init__(n_channels, n_times)
        self.temporal = nn.Conv2d(1, n_filters, (1, kernel_length))
        self.spatial = nn.Sequential(
            nn.Conv2d(n_filters, n_filters, (n_channels, 1), bias=False),
            nn.BatchNorm2d(n_filters),
        )
        # Squared amplitudes and log pooling form a parameter-free backend.
        self.separable = nn.Sequential(
            _Square(),
            nn.AvgPool2d((1, pool_length), (1, pool_stride)),
            _SafeLog(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(self._feature_dim(), num_classes)
        self.apply(_init_convnet)


class DeepConvNet(_Backbone):
    def __init__(
        self,
        n_channels,
        n_times,
        num_classes=2,
        dropout=0.5,
        filters=(25, 50, 100, 200),
        kernel_length=10,
        pool_length=3,
        pool_stride=3,
    ):
        super().__init__(n_channels, n_times)
        if len(filters) != 4 or any(v < 1 for v in filters):
            raise ValueError("DeepConvNet requires four positive filter widths")
        self.temporal = nn.Conv2d(1, filters[0], (1, kernel_length))
        self.spatial = nn.Sequential(
            nn.Conv2d(filters[0], filters[0], (n_channels, 1), bias=False),
            nn.BatchNorm2d(filters[0]),
            nn.ELU(),
            nn.MaxPool2d((1, pool_length), (1, pool_stride)),
        )
        blocks = []
        for n_in, n_out in zip(filters[:-1], filters[1:]):
            blocks.extend(
                [
                    nn.Dropout(dropout),
                    nn.Conv2d(n_in, n_out, (1, kernel_length), bias=False),
                    nn.BatchNorm2d(n_out),
                    nn.ELU(),
                    nn.MaxPool2d((1, pool_length), (1, pool_stride)),
                ]
            )
        self.separable = nn.Sequential(*blocks)
        self.classifier = nn.Linear(self._feature_dim(), num_classes)
        self.apply(_init_convnet)


class _Tokens(nn.Module):
    def forward(self, x):
        return x.squeeze(2).transpose(1, 2)


class _Attention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.dim, self.heads, self.dropout = dim, heads, dropout
        self.qkv = nn.Linear(dim, 3 * dim)
        self.output = nn.Linear(dim, dim)

    def forward(self, x):
        batch, length, _ = x.shape
        q, k, v = (
            self.qkv(x)
            .reshape(batch, length, 3, self.heads, self.dim // self.heads)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        # Original EEG Conformer uses sqrt(embedding_dim), not sqrt(head_dim).
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            scale=self.dim**-0.5,
        )
        return self.output(attended.transpose(1, 2).reshape(batch, length, self.dim))


class _ConformerBlock(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.attention = nn.Sequential(
            nn.LayerNorm(dim), _Attention(dim, heads, dropout), nn.Dropout(dropout)
        )
        self.feedforward = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attention(x)
        return x + self.feedforward(x)


class EEGConformer(_Backbone):
    def __init__(
        self,
        n_channels,
        n_times,
        num_classes=2,
        dropout=0.5,
        embedding_dim=40,
        depth=6,
        num_heads=10,
        kernel_length=25,
        pool_length=75,
        pool_stride=15,
        attention_dropout=0.5,
        head_hidden=(256, 32),
    ):
        super().__init__(n_channels, n_times)
        if num_heads < 1 or embedding_dim < 1 or embedding_dim % num_heads or depth < 1:
            raise ValueError(
                "Conformer needs positive depth and embedding_dim divisible by num_heads"
            )
        if len(head_hidden) != 2 or any(v < 1 for v in head_hidden):
            raise ValueError("Conformer head_hidden must contain two positive widths")
        self.temporal = nn.Conv2d(1, embedding_dim, (1, kernel_length))
        self.spatial = nn.Sequential(
            nn.Conv2d(embedding_dim, embedding_dim, (n_channels, 1)),
            nn.BatchNorm2d(embedding_dim),
            nn.ELU(),
            nn.AvgPool2d((1, pool_length), (1, pool_stride)),
            nn.Dropout(dropout),
        )
        self.separable = nn.Sequential(
            nn.Conv2d(embedding_dim, embedding_dim, (1, 1)),
            _Tokens(),
            *[_ConformerBlock(embedding_dim, num_heads, attention_dropout) for _ in range(depth)],
        )
        self.classifier = nn.Sequential(
            nn.Linear(self._feature_dim(), head_hidden[0]),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden[0], head_hidden[1]),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(head_hidden[1], num_classes),
        )


class EEGNet(_Backbone):
    def __init__(
        self,
        n_channels: int,
        n_times: int,
        num_classes: int = 2,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        dropout: float = 0.5,
        kernel_length: int = 64,
    ) -> None:
        super().__init__(n_channels, n_times)
        left = (kernel_length - 1) // 2
        right = kernel_length - 1 - left
        self.temporal = nn.Sequential(
            nn.ZeroPad2d((left, right, 0, 0)),
            nn.Conv2d(1, F1, (1, kernel_length), bias=False),
            nn.BatchNorm2d(F1),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        self.separable = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(F1 * D, F1 * D, (1, 16), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool2d((1, 8)),
        )
        self.classifier = nn.Linear(self._feature_dim(), num_classes)


def build_backbone(
    name: str, n_channels: int, n_times: int, num_classes: int = 2, **kwargs
) -> nn.Module:
    """Build a backbone directly, without an experiment configuration."""
    models = {
        "eegnet": EEGNet,
        "shallowconvnet": ShallowConvNet,
        "deepconvnet": DeepConvNet,
        "eegconformer": EEGConformer,
    }
    if name not in models:
        raise ValueError(f"Unknown backbone {name!r}; choose from {tuple(models)}")
    return models[name](n_channels=n_channels, n_times=n_times, num_classes=num_classes, **kwargs)
