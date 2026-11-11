from dataclasses import dataclass

import numpy as np


def normalize_channel_names(channel_names: tuple[str, ...]) -> tuple[str, ...]:
    """Validate electrode names and normalize case and surrounding whitespace."""
    if isinstance(channel_names, (str, bytes)):
        raise ValueError("channel_names must be a sequence of electrode names")
    if any(not isinstance(name, str) for name in channel_names):
        raise ValueError("Channel names must be strings")
    names = tuple(name.strip().casefold() for name in channel_names)
    if not names or any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("Channel names must be nonempty and unique, ignoring case")
    return names


def assert_channel_consistency(
    expected: tuple[str, ...], actual: tuple[str, ...], *, context: str
) -> None:
    """Require the same ordered electrodes without reordering either input."""
    expected = normalize_channel_names(expected)
    actual = normalize_channel_names(actual)
    if len(expected) != len(actual):
        raise ValueError(f"{context}: expected {len(expected)} channels, got {len(actual)}")
    for position, (left, right) in enumerate(zip(expected, actual), 1):
        if left != right:
            raise ValueError(
                f"{context}: channel {position} must be {left!r}, got {right!r}; "
                "electrode names and order must match"
            )


@dataclass
class SubjectData:
    subject_id: str
    X: np.ndarray
    y: np.ndarray
    channel_names: tuple[str, ...]
    session_ids: np.ndarray | None = None

    def __post_init__(self):
        self.X = np.asarray(self.X, dtype=np.float32)
        self.y = np.asarray(self.y)
        normalize_channel_names(self.channel_names)
        self.channel_names = tuple(self.channel_names)
        if self.X.ndim != 3 or min(self.X.shape) < 1 or self.X.shape[2] < 2:
            raise ValueError("X must have shape [N, C, T] with N,C >= 1 and T >= 2")
        if not np.isfinite(self.X).all():
            raise ValueError("X must contain finite values")
        if self.y.shape != (len(self.X),) or set(self.y.tolist()) != {0, 1}:
            raise ValueError("y must have shape [N] and contain both binary classes 0 and 1")
        self.y = self.y.astype(np.int64)
        if len(self.channel_names) != self.X.shape[1]:
            raise ValueError("channel_names must match the channel axis of X")
        if self.session_ids is None:
            self.session_ids = np.zeros(len(self.X), dtype=np.int64)
        self.session_ids = np.asarray(self.session_ids).astype(str)
        if self.session_ids.shape != self.y.shape:
            raise ValueError("session_ids must have shape [N]")
