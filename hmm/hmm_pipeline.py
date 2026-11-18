"""
hmm_pipeline.py – Member 3: Complete HMM Map Matching Pipeline
==============================================================

Responsibility
--------------
Orchestrates emission probability extraction, transition probability modeling,
Viterbi decoding, and route reconstruction across single or multiple trips.

Features:
  - Integrates dynamically with the best-performing ML model (Random Forest by default).
  - Supports inference using precomputed probabilities or fitted scikit-learn models.
  - Strict isolation: reference .route files are never passed to the decoder (zero data leakage).
  - Evaluates decoded routes against ground truth routes when available.
  - Exports standard evaluation outputs to outputs/.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml

from .emission import get_emission_probabilities
from .route_reconstruction import RouteReconstructionResult, reconstruct_route
from .transition import DEFAULT_BETA, TransitionModel
from .viterbi import ViterbiResult, viterbi_decode

logger = logging.getLogger(__name__)


class HMMMapMatcher:
    """
    High-level HMM Map Matching engine for GPS trajectories on road networks.
    """

    def __init__(
        self,
        road_network: dict,
        model: Optional[Any] = None,
        features: Optional[List[str]] = None,
        emission_mode: str = "hybrid",
        alpha: float = 0.2,
        sigma_z: float = 20.0,
        beta: float = DEFAULT_BETA,
        model_name: Optional[str] = "random_forest",
    ):
        """
        Parameters
        ----------
        road_network : dict
            Road network dictionary (output of Member 1's build_road_network).
        model : Any, optional
            Fitted ML classifier pipeline.
        features : list of str, optional
            Features used by the ML model.
        emission_mode : {'ml', 'geometric', 'hybrid'}, default 'hybrid'
        alpha : float, default 0.2
            Weight for geometric likelihood in hybrid mode.
        sigma_z : float, default 20.0
            GPS measurement standard deviation in metres.
        beta : float, default 20.0
            Transition scale parameter in metres.
        model_name : str, optional
            Name of the model (for reporting/metadata).
        """
        self.road_network = road_network
        self.model = model
        self.features = features
        self.emission_mode = emission_mode
        self.alpha = alpha
        self.sigma_z = sigma_z
        self.beta = beta
        self.model_name = model_name

        # Initialize transition model with the road network graph
        self.transition_model = TransitionModel(road_network=road_network, beta=beta)

    def match_trip(
        self,
        trip_id: str,
        candidates_df: pd.DataFrame,
        track_df: Optional[pd.DataFrame] = None,
        prob_col: Optional[str] = None,
        infer_intermediate: bool = True,
    ) -> RouteReconstructionResult:
        """
        Perform map matching on a single trip.

        Parameters
        ----------
        trip_id : str
        candidates_df : pd.DataFrame
            Candidate feature table for this trip.
        track_df : pd.DataFrame, optional
            GPS track points for this trip.
        prob_col : str, optional
            Name of column containing precomputed probabilities.
        infer_intermediate : bool, default True
            Whether to infer connecting road arcs for sparse GPS fixes.

        Returns
        -------
        RouteReconstructionResult
        """
        cands = candidates_df[candidates_df["trip_id"] == trip_id].copy()
        if cands.empty:
            cands = candidates_df.copy()

        if cands.empty:
            raise ValueError(f"Trip {trip_id}: candidate DataFrame is empty.")

        # 1. Compute emission probabilities
        em_df = get_emission_probabilities(
            candidate_df=cands,
            model=self.model,
            features=self.features,
            mode=self.emission_mode,
            alpha=self.alpha,
            sigma_z=self.sigma_z,
            prob_col=prob_col,
            normalize_per_obs=True,
        )

        # 2. Prepare structures for Viterbi
        observations = sorted(em_df["gps_point_id"].unique())
        candidate_dict: Dict[int, List[int]] = {}
        emission_dict: Dict[int, Dict[int, float]] = {}

        for row in em_df.itertuples(index=False):
            gid = int(row.gps_point_id)
            aid = int(row.candidate_arc_id)
            prob = float(row.emission_prob)

            if gid not in candidate_dict:
                candidate_dict[gid] = []
                emission_dict[gid] = {}

            candidate_dict[gid].append(aid)
            emission_dict[gid][aid] = prob

        # Filter out observations that have no candidates
        valid_obs = [gid for gid in observations if gid in candidate_dict and candidate_dict[gid]]

        # 3. Viterbi sequence decoding
        viterbi_res = viterbi_decode(
            trip_id=trip_id,
            observations=valid_obs,
            candidate_dict=candidate_dict,
            emission_dict=emission_dict,
            transition_model=self.transition_model,
            track_df=track_df,
        )

        # 4. Reconstruct physical route sequence
        route_res = reconstruct_route(
            viterbi_res=viterbi_res,
            road_network=self.road_network,
            infer_intermediate=infer_intermediate,
        )

        return route_res


def decode_trip(
    trip_id: str,
    candidates_df: pd.DataFrame,
    road_network: dict,
    track_df: Optional[pd.DataFrame] = None,
    model: Optional[Any] = None,
    features: Optional[List[str]] = None,
    emission_mode: str = "hybrid",
    prob_col: Optional[str] = None,
    beta: float = DEFAULT_BETA,
) -> RouteReconstructionResult:
    """
    Convenience functional interface to decode a single trip.
    """
    matcher = HMMMapMatcher(
        road_network=road_network,
        model=model,
        features=features,
        emission_mode=emission_mode,
        beta=beta,
    )
    return matcher.match_trip(
        trip_id=trip_id,
        candidates_df=candidates_df,
        track_df=track_df,
        prob_col=prob_col,
    )


def run_hmm_pipeline(
    candidates_df: pd.DataFrame,
    road_network: dict,
    trips_track: Optional[Dict[str, pd.DataFrame]] = None,
    reference_routes: Optional[Dict[str, pd.DataFrame]] = None,
    model: Optional[Any] = None,
    features: Optional[List[str]] = None,
    model_name: str = "random_forest",
    prob_col: Optional[str] = None,
    output_dir: Union[str, Path] = "outputs",
    emission_mode: str = "hybrid",
    alpha: float = 0.2,
    sigma_z: float = 20.0,
    beta: float = DEFAULT_BETA,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Execute HMM map matching over all trips in candidates_df and export results.

    Parameters
    ----------
    candidates_df : pd.DataFrame
    road_network : dict
    trips_track : dict, optional
        Mapping trip_id -> track_df.
    reference_routes : dict, optional
        Mapping trip_id -> route_df (used strictly for evaluation scoring, never for decoding).
    model : trained ML model, optional
    features : list of str, optional
    model_name : str, default 'random_forest'
    prob_col : str, optional
    output_dir : str or Path, default 'outputs'
    emission_mode : str, default 'hybrid'
    beta : float, default 20.0

    Returns
    -------
    (predicted_routes_df, summary_df, metrics_df)
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    matcher = HMMMapMatcher(
        road_network=road_network,
        model=model,
        features=features,
        model_name=model_name,
        emission_mode=emission_mode,
        alpha=alpha,
        sigma_z=sigma_z,
        beta=beta,
    )

    trip_ids = sorted(candidates_df["trip_id"].unique())
    logger.info("Running HMM decoding across %d trip(s)...", len(trip_ids))

    all_point_matches = []
    all_reconstructed_routes = []
    summary_rows = []
    metric_rows = []

    for tid in trip_ids:
        tid_str = str(tid)
        cands_sub = candidates_df[candidates_df["trip_id"] == tid]
        track_sub = trips_track.get(tid_str) if trips_track else None

        route_res = matcher.match_trip(
            trip_id=tid_str,
            candidates_df=cands_sub,
            track_df=track_sub,
            prob_col=prob_col,
        )

        pts_df = route_res.point_matches
        all_point_matches.append(pts_df)
        all_reconstructed_routes.append(route_res.reconstructed_route)

        # Basic summary
        n_obs = len(pts_df)
        continuity = route_res.continuity_score
        n_inferred = route_res.n_inferred_arcs

        # Evaluation against reference route if provided
        top1_acc = np.nan
        route_f1 = np.nan
        route_overlap = np.nan

        if reference_routes and tid_str in reference_routes:
            ref_route = reference_routes[tid_str]
            ref_arcs = set(ref_route["arc_id"].unique())
            pred_arcs = set(route_res.route_arc_sequence)

            # Match accuracy on observation points
            matched_correct = pts_df["selected_arc_id"].isin(ref_arcs).sum()
            top1_acc = matched_correct / max(n_obs, 1)

            # Jaccard route overlap
            intersect = len(ref_arcs & pred_arcs)
            union = len(ref_arcs | pred_arcs)
            route_overlap = intersect / max(union, 1)

            # Route F1
            prec = intersect / max(len(pred_arcs), 1)
            rec = intersect / max(len(ref_arcs), 1)
            route_f1 = (2 * prec * rec) / max((prec + rec), 1e-12)

        summary_rows.append({
            "trip_id": tid_str,
            "n_observations": n_obs,
            "n_route_arcs": len(route_res.route_arc_sequence),
            "n_collapsed_arcs": len(route_res.collapsed_arc_sequence),
            "n_inferred_arcs": n_inferred,
            "route_continuity": continuity,
            "is_fully_connected": route_res.is_fully_connected,
            "top1_accuracy": top1_acc,
            "route_f1": route_f1,
            "route_overlap_jaccard": route_overlap,
        })

    # Concatenate tables
    routes_df = pd.concat(all_point_matches, ignore_index=True) if all_point_matches else pd.DataFrame()
    full_routes_df = pd.concat(all_reconstructed_routes, ignore_index=True) if all_reconstructed_routes else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows)

    # Global aggregate metrics
    metrics_df = pd.DataFrame([{
        "model_name": model_name,
        "n_trips": len(trip_ids),
        "mean_route_continuity": summary_df["route_continuity"].mean() if not summary_df.empty else np.nan,
        "mean_top1_accuracy": summary_df["top1_accuracy"].mean() if not summary_df.empty else np.nan,
        "mean_route_f1": summary_df["route_f1"].mean() if not summary_df.empty else np.nan,
        "mean_inferred_arcs_per_trip": summary_df["n_inferred_arcs"].mean() if not summary_df.empty else np.nan,
    }])

    # Export CSVs to outputs/
    routes_df.to_csv(out_path / "hmm_predicted_routes.csv", index=False)
    summary_df.to_csv(out_path / "hmm_route_summary.csv", index=False)
    metrics_df.to_csv(out_path / "hmm_metrics.csv", index=False)
    if not full_routes_df.empty:
        full_routes_df.to_csv(out_path / "hmm_full_reconstructed_routes.csv", index=False)

    logger.info(
        "HMM outputs successfully written to %s (hmm_predicted_routes.csv, hmm_route_summary.csv, hmm_metrics.csv)",
        out_path,
    )

    return routes_df, summary_df, metrics_df
