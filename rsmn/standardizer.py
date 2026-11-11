from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ChannelStandardizer:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim != 3 or X.shape[1] != len(self.mean) or not np.isfinite(X).all():
            raise ValueError("X must be finite [N,C,T] data with the fitted channel count")
        return ((X - self.mean[None, :, None]) / self.std[None, :, None]).astype(np.float32)


def fit_channel_standardizer(X_train: np.ndarray) -> ChannelStandardizer:
    X_train = np.asarray(X_train, dtype=np.float64)
    if X_train.ndim != 3 or min(X_train.shape) < 1 or not np.isfinite(X_train).all():
        raise ValueError("X_train must be nonempty finite [N,C,T] data")
    mean = X_train.mean(axis=(0, 2)).astype(np.float32)
    std = X_train.std(axis=(0, 2)).astype(np.float32)
    std[std < 1e-8] = 1.0
    return ChannelStandardizer(mean, std)
