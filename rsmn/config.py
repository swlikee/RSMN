
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class CoreConfig:
    SPD_EPSILON: float = 1e-6
    TRACE_NORMALIZATION: bool = True
    RIEMANNIAN_MEAN_MAX_ITER: int = 25
    RIEMANNIAN_MEAN_TOL: float = 1e-7
    LAMBDA_SESSION: float = 1.0
    SOURCE_DISTANCE_WEIGHT: float = 0.55
    SOURCE_DIRECTION_WEIGHT: float = 0.25
    SOURCE_RELIABILITY_WEIGHT: float = 0.20
    ALPHA_MIN: float = 0.25
    ALPHA_MAX: float = 0.65
    TAU_EPSILON: float = 1e-8
    USE_ADAPTIVE_CPRA: bool = True
    TCPA_WEIGHT: float = 0.10
    RAME_R_MIN: float = 0.10
    RAME_R_MAX: float = 0.90
    RAME_MIN_SCALE: float = 1e-6
    RAME_USE_TEST_ENSEMBLE: bool = True
    RAME_TEST_ETA_MAX: float = 0.50

    def __post_init__(self):
        if not all(isfinite(value) for value in vars(self).values()):
            raise ValueError("Core parameters must be finite")
        if (
            min(self.SPD_EPSILON, self.TAU_EPSILON, self.RAME_MIN_SCALE, self.RIEMANNIAN_MEAN_TOL)
            <= 0
        ):
            raise ValueError("Numerical tolerances must be positive")
        if self.RIEMANNIAN_MEAN_MAX_ITER < 1:
            raise ValueError("RIEMANNIAN_MEAN_MAX_ITER must be positive")
        if self.TCPA_WEIGHT < 0:
            raise ValueError("TCPA_WEIGHT must be nonnegative")
        if not 0 <= self.ALPHA_MIN <= self.ALPHA_MAX < 1:
            raise ValueError("Require 0 <= ALPHA_MIN <= ALPHA_MAX < 1 for partial alignment")
        if not 0 <= self.RAME_R_MIN <= self.RAME_R_MAX <= 1:
            raise ValueError("Require 0 <= RAME_R_MIN <= RAME_R_MAX <= 1")
        if not 0 <= self.RAME_TEST_ETA_MAX <= 1:
            raise ValueError("RAME_TEST_ETA_MAX must be in [0, 1]")
        if (
            min(
                self.LAMBDA_SESSION,
                self.SOURCE_DISTANCE_WEIGHT,
                self.SOURCE_DIRECTION_WEIGHT,
                self.SOURCE_RELIABILITY_WEIGHT,
            )
            < 0
        ):
            raise ValueError("Session and source matching weights must be nonnegative")
