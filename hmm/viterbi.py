"""
viterbi.py – Member 3: Viterbi Algorithm for Sequence Decoding
==============================================================

Responsibility
--------------
Implement the Viterbi dynamic programming algorithm in log-space to find the
globally most probable sequence of road segments across a complete GPS trajectory.

Viterbi Recurrence
------------------
  Initialization (t = 0):
    delta_0(j) = log pi(j) + log emission_0(j)

  Recursion (t = 1 ... T-1):
    delta_t(j) = log emission_t(j) + max_i [ delta_{t-1}(i) + log transition_t(i, j) ]
    psi_t(j)   = argmax_i [ delta_{t-1}(i) + log transition_t(i, j) ]

  Termination:
    best_state_{T-1} = argmax_j [ delta_{T-1}(j) ]
    best_score       = max_j [ delta_{T-1}(j) ]

  Backtracking:
    best_state_t = psi_{t+1}(best_state_{t+1})  for t = T-2, T-3, ..., 0

Guarantees:
  - Exact log-space dynamic programming (no underflow).
  - Handles single-observation trips, variable candidate list sizes, and large trajectories.
  - Returns complete step-by-step decoding metadata.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .transition import TransitionModel, compute_transition_matrix

logger = logging.getLogger(__name__)


@dataclass
class DecodedStep:
    """Represents the decoded state at a single GPS observation."""
    step_idx: int
    gps_point_id: int
    selected_arc_id: int
    emission_prob: float
    log_emission: float
    viterbi_score: float


@dataclass
class ViterbiResult:
    """Complete result of Viterbi decoding for a trip."""
    trip_id: str
    total_log_likelihood: float
    steps: List[DecodedStep]
    arc_sequence: List[int]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert decoded steps into a standard DataFrame."""
        rows = [
            {
                "trip_id": self.trip_id,
                "sequence_position": s.step_idx,
                "gps_point_id": s.gps_point_id,
                "selected_arc_id": s.selected_arc_id,
                "emission_probability": s.emission_prob,
                "log_emission": s.log_emission,
                "viterbi_score": s.viterbi_score,
            }
            for s in self.steps
        ]
        return pd.DataFrame(rows)


def viterbi_decode(
    trip_id: str,
    observations: List[int],
    candidate_dict: Dict[int, List[int]],
    emission_dict: Dict[int, Dict[int, float]],
    transition_model: TransitionModel,
    track_df: Optional[pd.DataFrame] = None,
) -> ViterbiResult:
    """
    Run Viterbi decoding on candidate states of a single GPS trajectory.

    Parameters
    ----------
    trip_id : str
        Identifier of the trip.
    observations : list of int
        Chronological list of gps_point_id observations [0, 1, ..., T-1].
    candidate_dict : dict
        Mapping gps_point_id -> list of candidate arc_ids for that observation.
    emission_dict : dict
        Mapping gps_point_id -> dict {candidate_arc_id: emission_prob in (0, 1]}.
    transition_model : TransitionModel
        Transition model configured with the road network.
    track_df : pd.DataFrame, optional
        Preprocessed GPS track to extract distances and time gaps between fixes.

    Returns
    -------
    ViterbiResult
        Object containing decoded arc sequence, metadata, and Viterbi scores.
    """
    T = len(observations)
    if T == 0:
        return ViterbiResult(trip_id=trip_id, total_log_likelihood=0.0, steps=[], arc_sequence=[])

    # Pre-extract observed distances and time gaps from track_df if available
    obs_dist = [30.0] * T
    time_gaps = [1.0] * T
    if track_df is not None and not track_df.empty:
        track_map = track_df.set_index("gps_point_id").to_dict(orient="index")
        for idx in range(1, T):
            gid_curr = observations[idx]
            if gid_curr in track_map:
                row = track_map[gid_curr]
                d = row.get("distance_m")
                dt = row.get("time_gap_s")
                obs_dist[idx] = float(d) if pd.notna(d) and d >= 0 else 30.0
                time_gaps[idx] = float(dt) if pd.notna(dt) and dt > 0 else 1.0

    # ------------------------------------------------------------------ #
    # 1. Initialization (t = 0)
    # ------------------------------------------------------------------ #
    gid_0 = observations[0]
    cands_0 = candidate_dict.get(gid_0, [])
    if not cands_0:
        raise ValueError(f"Trip {trip_id}: observation 0 (gps_id={gid_0}) has no candidate arcs.")

    n_cands_0 = len(cands_0)
    uniform_prior = -math.log(n_cands_0)

    # delta[t] is an array of shape (len(cands_t),) storing max log-prob to state j
    delta_0 = np.zeros(n_cands_0, dtype=np.float64)
    em_0 = emission_dict[gid_0]

    for j, arc in enumerate(cands_0):
        p_emit = max(1e-12, em_0.get(arc, 1e-12))
        delta_0[j] = uniform_prior + math.log(p_emit)

    deltas = [delta_0]
    # psi stores backpointer index for each state at t >= 1
    psi = []

    # ------------------------------------------------------------------ #
    # 2. Recursion (t = 1 ... T-1)
    # ------------------------------------------------------------------ #
    for t in range(1, T):
        gid_t = observations[t]
        cands_t = candidate_dict.get(gid_t, [])
        if not cands_t:
            raise ValueError(f"Trip {trip_id}: observation {t} (gps_id={gid_t}) has no candidate arcs.")

        cands_prev = candidate_dict[observations[t - 1]]
        n_curr = len(cands_t)
        n_prev = len(cands_prev)

        # Log transition matrix: shape (n_prev, n_curr)
        log_T = compute_transition_matrix(
            cands_prev=cands_prev,
            cands_curr=cands_t,
            d_obs=obs_dist[t],
            transition_model=transition_model,
            time_gap_s=time_gaps[t],
        )

        em_t = emission_dict[gid_t]
        log_emit_t = np.array([math.log(max(1e-12, em_t.get(arc, 1e-12))) for arc in cands_t])

        # Vectorized Viterbi step:
        # scores[i, j] = delta_{t-1}(i) + log_T[i, j]
        prev_scores = deltas[t - 1][:, np.newaxis] + log_T

        # Max over previous states i (axis 0)
        best_prev_idx = np.argmax(prev_scores, axis=0)
        best_prev_val = np.max(prev_scores, axis=0)

        # delta_t(j) = log_emit(j) + max_i [ prev_scores ]
        delta_t = log_emit_t + best_prev_val

        deltas.append(delta_t)
        psi.append(best_prev_idx)

    # ------------------------------------------------------------------ #
    # 3. Termination
    # ------------------------------------------------------------------ #
    best_last_idx = int(np.argmax(deltas[-1]))
    best_score = float(deltas[-1][best_last_idx])

    # ------------------------------------------------------------------ #
    # 4. Backtracking
    # ------------------------------------------------------------------ #
    best_path_indices = [0] * T
    best_path_indices[-1] = best_last_idx

    for t in range(T - 2, -1, -1):
        best_path_indices[t] = int(psi[t][best_path_indices[t + 1]])

    # ------------------------------------------------------------------ #
    # 5. Build Result Steps
    # ------------------------------------------------------------------ #
    decoded_steps = []
    arc_sequence = []

    for t in range(T):
        gid = observations[t]
        cands = candidate_dict[gid]
        c_idx = best_path_indices[t]
        selected_arc = cands[c_idx]
        arc_sequence.append(selected_arc)

        em_val = emission_dict[gid].get(selected_arc, 1e-12)
        decoded_steps.append(
            DecodedStep(
                step_idx=t,
                gps_point_id=gid,
                selected_arc_id=selected_arc,
                emission_prob=em_val,
                log_emission=math.log(max(1e-12, em_val)),
                viterbi_score=float(deltas[t][c_idx]),
            )
        )

    logger.debug(
        "Trip %s Viterbi decode complete: %d steps, log-likelihood=%.3f",
        trip_id,
        T,
        best_score,
    )

    return ViterbiResult(
        trip_id=trip_id,
        total_log_likelihood=best_score,
        steps=decoded_steps,
        arc_sequence=arc_sequence,
    )
