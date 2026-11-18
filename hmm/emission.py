"""
emission.py – Member 3: Emission Probability Calculation
=========================================================

Responsibility
--------------
Calculate emission probabilities P(O_t | s_t = c_i) for each candidate road arc
given the GPS observation O_t.

Supported Emission Modes:
  1. 'ml': Uses P(y=1 | x) from Member 2's best-performing trained ML model.
  2. 'geometric': Gaussian observation error likelihood:
        P_geom(O_t | c_i) = (1 / (sqrt(2*pi) * sigma_z)) * exp(-d(O_t, c_i)^2 / (2 * sigma_z^2))
  3. 'hybrid' (default): Log-convex combination of ML-informed score and geometric distance:
        log P_emit(c_i) = (1 - alpha) * log P_ml(c_i) + alpha * log P_geom(c_i)
     where alpha in [0, 1] controls geometric weighting (default alpha = 0.2).

Numerical Safeguards:
  - All probabilities are clipped to [EPSILON, 1.0 - EPSILON] to prevent log(0) underflow.
  - Emission scores per observation are optionally normalized across candidate states.
"""

from __future__ import annotations

import logging
import math
from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Numerical safeguard to avoid log(0) and extreme infinities
EPSILON = 1e-12

# Standard GPS noise standard deviation (metres)
DEFAULT_SIGMA_Z = 20.0


def compute_geometric_emission(
    distance_m: Union[float, np.ndarray, pd.Series],
    sigma_z: float = DEFAULT_SIGMA_Z,
) -> np.ndarray:
    """
    Gaussian likelihood for GPS observation distance to candidate road arc.

    Formula:
        P(distance | candidate) = (1 / (sqrt(2*pi) * sigma_z)) * exp(-distance^2 / (2 * sigma_z^2))

    Parameters
    ----------
    distance_m : float or array-like
        Perpendicular distance in metres from GPS point to candidate arc.
    sigma_z : float, default 20.0
        Standard deviation of GPS measurement error (metres).

    Returns
    -------
    np.ndarray of likelihoods, clipped to [EPSILON, 1.0].
    """
    dist = np.asarray(distance_m, dtype=np.float64)
    # Gaussian density (unnormalized or normalized)
    norm_const = 1.0 / (math.sqrt(2.0 * math.pi) * sigma_z)
    exponent = -0.5 * (dist / sigma_z) ** 2
    prob = norm_const * np.exp(exponent)
    return np.clip(prob, EPSILON, 1.0)


def extract_positive_class_probability(
    model: Any,
    X: np.ndarray,
) -> np.ndarray:
    """
    Safely extract P(y=1 | X) from an estimator with predict_proba.

    Handles estimators where classes_ may be [0, 1], [1, 0], or single-class.
    """
    if not hasattr(model, "predict_proba"):
        raise ValueError(f"Model {type(model)} does not support predict_proba.")

    proba = model.predict_proba(X)

    if hasattr(model, "classes_"):
        classes = list(model.classes_)
        if 1 in classes:
            pos_idx = classes.index(1)
            p1 = proba[:, pos_idx]
        else:
            # Fallback to last column if binary
            p1 = proba[:, -1]
    else:
        p1 = proba[:, -1]

    return np.clip(np.asarray(p1, dtype=np.float64), EPSILON, 1.0 - EPSILON)


def get_emission_probabilities(
    candidate_df: pd.DataFrame,
    model: Optional[Any] = None,
    features: Optional[List[str]] = None,
    mode: str = "hybrid",
    alpha: float = 0.2,
    sigma_z: float = DEFAULT_SIGMA_Z,
    prob_col: Optional[str] = None,
    normalize_per_obs: bool = True,
) -> pd.DataFrame:
    """
    Compute emission probabilities for every row in a candidate DataFrame.

    Parameters
    ----------
    candidate_df : pd.DataFrame
        Must contain columns: 'trip_id', 'gps_point_id', 'candidate_arc_id'.
        If mode in ('geometric', 'hybrid'), 'distance_m' must be present.
    model : trained ML model pipeline (sklearn-compatible)
        Required if prob_col is not provided and mode in ('ml', 'hybrid').
    features : list of str, optional
        Features used by the ML model. If None, tries to read from candidate_df.
    mode : {'ml', 'geometric', 'hybrid'}, default 'hybrid'
        How emission score is computed.
    alpha : float, default 0.2
        Geometric weight in hybrid mode: (1 - alpha)*log(P_ml) + alpha*log(P_geom).
    sigma_z : float, default 20.0
        GPS error standard deviation in metres.
    prob_col : str, optional
        If candidate_df already contains precomputed probabilities (e.g. from
        Member 2's student_test_predictions.csv, column 'predicted_probability'),
        pass the column name here to bypass re-fitting/predicting.
    normalize_per_obs : bool, default True
        Whether to normalize candidate emission probabilities to sum to 1.0
        for each GPS observation.

    Returns
    -------
    pd.DataFrame
        Copy of candidate_df with added columns:
          - 'emission_prob': final combined emission probability in (0, 1]
          - 'log_emission': log(emission_prob)
    """
    df = candidate_df.copy()

    # 1. Determine ML probability P_ml
    if prob_col is not None and prob_col in df.columns:
        p_ml = np.clip(df[prob_col].to_numpy(dtype=np.float64), EPSILON, 1.0 - EPSILON)
    elif model is not None and features is not None:
        X = df[features].to_numpy(dtype=np.float32)
        p_ml = extract_positive_class_probability(model, X)
    elif "predicted_probability" in df.columns:
        p_ml = np.clip(df["predicted_probability"].to_numpy(dtype=np.float64), EPSILON, 1.0 - EPSILON)
    else:
        if mode in ("ml", "hybrid"):
            logger.warning(
                "Neither model nor precomputed probabilities provided. Falling back to geometric mode."
            )
            mode = "geometric"
        p_ml = np.full(len(df), 0.5, dtype=np.float64)

    # 2. Determine Geometric probability P_geom
    if "distance_m" in df.columns:
        p_geom = compute_geometric_emission(df["distance_m"].fillna(50.0).to_numpy(), sigma_z=sigma_z)
    else:
        p_geom = np.full(len(df), 0.5, dtype=np.float64)

    # 3. Combine according to mode
    if mode == "ml":
        p_final = p_ml
    elif mode == "geometric":
        p_final = p_geom
    elif mode == "hybrid":
        # Log-convex combination: log P = (1 - alpha) * log P_ml + alpha * log P_geom
        # Equivalent to: P = (P_ml ^ (1 - alpha)) * (P_geom ^ alpha)
        log_p = (1.0 - alpha) * np.log(p_ml) + alpha * np.log(p_geom)
        p_final = np.exp(log_p)
    else:
        raise ValueError(f"Unknown emission mode: {mode}. Choose 'ml', 'geometric', or 'hybrid'.")

    df["raw_emission_prob"] = np.clip(p_final, EPSILON, 1.0)

    # 4. Optional normalization per GPS observation
    if normalize_per_obs and "gps_point_id" in df.columns:
        obs_sums = df.groupby(["trip_id", "gps_point_id"])["raw_emission_prob"].transform("sum")
        obs_sums = np.where(obs_sums > 0, obs_sums, 1.0)
        df["emission_prob"] = np.clip(df["raw_emission_prob"] / obs_sums, EPSILON, 1.0)
    else:
        df["emission_prob"] = df["raw_emission_prob"]

    df["log_emission"] = np.log(df["emission_prob"])
    return df


def compute_hybrid_emission(
    ml_prob: float,
    distance_m: float,
    alpha: float = 0.2,
    sigma_z: float = DEFAULT_SIGMA_Z,
) -> float:
    """
    Single-observation helper for hybrid emission.
    """
    p_ml = max(EPSILON, min(1.0 - EPSILON, float(ml_prob)))
    p_geom = float(compute_geometric_emission(distance_m, sigma_z=sigma_z))
    log_p = (1.0 - alpha) * math.log(p_ml) + alpha * math.log(p_geom)
    return max(EPSILON, math.exp(log_p))
