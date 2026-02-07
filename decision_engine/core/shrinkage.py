from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def _is_bad(x) -> bool:
    return x is None or (isinstance(x, float) and np.isnan(x))


def shrink_value(
    observed: Optional[float],
    n_obs: Optional[float],
    prior: Optional[float],
    n_prior_equiv: float = 50.0,
) -> Optional[float]:
    """Shrink an observed metric toward a prior with a prior-equivalent sample size.

    Notes:
      - This is a simple convex blend, not a full conjugate Bayesian model.
      - We treat `n_obs` as the effective sample size for the metric.
    """
    if _is_bad(observed) or _is_bad(prior) or n_obs is None:
        return observed
    try:
        n = float(n_obs)
    except Exception:
        return observed
    if n_prior_equiv <= 0:
        return float(observed)
    n = max(n, 0.0)
    w = n / (n + float(n_prior_equiv))
    return float(w * float(observed) + (1.0 - w) * float(prior))


def confidence_tier(n: Optional[int], low: int = 50, high: int = 100) -> str:
    if n is None:
        return "Low"
    try:
        n_i = int(n)
    except Exception:
        return "Low"
    if n_i >= high:
        return "High"
    if n_i >= low:
        return "Medium"
    return "Low"


@dataclass(frozen=True)
class ShrinkageConfig:
    n_prior_equiv_whiff: float = 50.0
    n_prior_equiv_csw: float = 80.0
    n_prior_equiv_chase: float = 80.0
    n_prior_equiv_ev: float = 40.0
    n_prior_equiv_barrel: float = 80.0

