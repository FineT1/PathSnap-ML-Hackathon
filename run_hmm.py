"""
run_hmm.py – Member 3 Entry-Point Script: HMM & Viterbi Route Reconstruction
=============================================================================

Executes the complete HMM map matching pipeline:
  1. Identifies and loads the best-performing ML model (Random Forest by default).
  2. Extracts candidate observations and features.
  3. Computes emissions (combining ML probabilities with geometric safeguards).
  4. Computes road network transitions and decodes using the Viterbi algorithm.
  5. Reconstructs physical road sequences and computes route continuity.
  6. Saves outputs to outputs/ (and results/ for stage comparison).

Usage:
  python run_hmm.py [--model random_forest] [--beta 20.0] [--alpha 0.2] [--synthetic]

Options:
  --model <name>       ML model to use (default: random_forest or config.yaml selection).
  --beta <float>       Transition rate parameter in metres (default: 20.0).
  --alpha <float>      Geometric weight in hybrid emission (default: 0.2).
  --output-dir <path>  Output directory (default: outputs).
  --synthetic          Run on synthetic test trips (for verification without full dataset).
"""

from __future__ import annotations

import argparse
import logging
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hmm import HMMMapMatcher, run_hmm_pipeline
from preprocessing.road_network import build_road_network

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_hmm")


def create_synthetic_trip():
    """Create a small, fully verifiable synthetic road network and GPS trajectory."""
    # 6 nodes forming a 2x3 grid in London (WGS-84)
    nodes_df = pd.DataFrame([
        {"node_id": 0, "lon": -0.1280, "lat": 51.5073},
        {"node_id": 1, "lon": -0.1270, "lat": 51.5073},
        {"node_id": 2, "lon": -0.1260, "lat": 51.5073},
        {"node_id": 3, "lon": -0.1280, "lat": 51.5078},
        {"node_id": 4, "lon": -0.1270, "lat": 51.5078},
        {"node_id": 5, "lon": -0.1260, "lat": 51.5078},
    ])

    # 7 directed arcs: 0->1, 1->2, 3->4, 4->5, 0->3, 1->4, 2->5
    arcs_df = pd.DataFrame([
        {"arc_id": 0, "from_node_id": 0, "to_node_id": 1},
        {"arc_id": 1, "from_node_id": 1, "to_node_id": 2},
        {"arc_id": 2, "from_node_id": 3, "to_node_id": 4},
        {"arc_id": 3, "from_node_id": 4, "to_node_id": 5},
        {"arc_id": 4, "from_node_id": 0, "to_node_id": 3},
        {"arc_id": 5, "from_node_id": 1, "to_node_id": 4},
        {"arc_id": 6, "from_node_id": 2, "to_node_id": 5},
    ])

    # Preprocessed GPS track: 4 points moving West to East along the bottom street (arcs 0 then 1)
    track_df = pd.DataFrame([
        {"gps_point_id": 0, "lon": -0.1279, "lat": 51.50732, "timestamp_s": 1000.0, "distance_m": 0.0, "time_gap_s": 0.0},
        {"gps_point_id": 1, "lon": -0.1274, "lat": 51.50731, "timestamp_s": 1010.0, "distance_m": 35.0, "time_gap_s": 10.0},
        {"gps_point_id": 2, "lon": -0.1268, "lat": 51.50732, "timestamp_s": 1020.0, "distance_m": 42.0, "time_gap_s": 10.0},
        {"gps_point_id": 3, "lon": -0.1262, "lat": 51.50731, "timestamp_s": 1030.0, "distance_m": 42.0, "time_gap_s": 10.0},
    ])

    # Candidate table: each GPS point has candidate arcs with distances & ML probabilities
    # Arc 0 and Arc 1 are the true path. Arc 2 and Arc 4 are noisy candidates.
    cands_data = [
        # GPS 0
        {"trip_id": "synthetic_01", "gps_point_id": 0, "candidate_arc_id": 0, "distance_m": 2.2, "predicted_probability": 0.85},
        {"trip_id": "synthetic_01", "gps_point_id": 0, "candidate_arc_id": 4, "distance_m": 12.0, "predicted_probability": 0.45},
        {"trip_id": "synthetic_01", "gps_point_id": 0, "candidate_arc_id": 2, "distance_m": 55.0, "predicted_probability": 0.10},
        # GPS 1
        {"trip_id": "synthetic_01", "gps_point_id": 1, "candidate_arc_id": 0, "distance_m": 1.1, "predicted_probability": 0.88},
        {"trip_id": "synthetic_01", "gps_point_id": 1, "candidate_arc_id": 5, "distance_m": 18.0, "predicted_probability": 0.35},
        # GPS 2 (noisy GPS fix closer to arc 5, but network path favors arc 1)
        {"trip_id": "synthetic_01", "gps_point_id": 2, "candidate_arc_id": 1, "distance_m": 3.0, "predicted_probability": 0.75},
        {"trip_id": "synthetic_01", "gps_point_id": 2, "candidate_arc_id": 5, "distance_m": 8.0, "predicted_probability": 0.60},
        {"trip_id": "synthetic_01", "gps_point_id": 2, "candidate_arc_id": 3, "distance_m": 45.0, "predicted_probability": 0.15},
        # GPS 3
        {"trip_id": "synthetic_01", "gps_point_id": 3, "candidate_arc_id": 1, "distance_m": 1.5, "predicted_probability": 0.90},
        {"trip_id": "synthetic_01", "gps_point_id": 3, "candidate_arc_id": 6, "distance_m": 15.0, "predicted_probability": 0.25},
    ]
    candidates_df = pd.DataFrame(cands_data)

    # Reference ground truth route: arcs 0 and 1
    ref_route_df = pd.DataFrame([
        {"route_step_id": 0, "arc_id": 0},
        {"route_step_id": 1, "arc_id": 1},
    ])

    return nodes_df, arcs_df, track_df, candidates_df, ref_route_df


def main():
    parser = argparse.ArgumentParser(description="PathSnap Member 3: HMM & Viterbi Pipeline")
    parser.add_argument("--model", type=str, default=None, help="Name of selected ML model (default: random_forest)")
    parser.add_argument("--beta", type=float, default=20.0, help="Transition scale parameter (metres)")
    parser.add_argument("--alpha", type=float, default=0.2, help="Geometric weight in hybrid emission")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--synthetic", action="store_true", help="Run on synthetic benchmark data")
    args = parser.parse_args()

    # Load configuration
    cfg_path = ROOT / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}

    # Determine selected model
    model_name = args.model
    if model_name is None:
        # Check model comparison results first
        comp_path = ROOT / "results" / "model_comparison.csv"
        if comp_path.exists():
            comp_df = pd.read_csv(comp_path)
            # Best model by student_top1
            best_row = comp_df.sort_values("student_top1", ascending=False).iloc[0]
            model_name = str(best_row["model"])
            logger.info("Auto-selected best model from results/model_comparison.csv: %s (Student Top-1: %.4f)",
                        model_name, best_row["student_top1"])
        else:
            model_name = "random_forest"

    logger.info("Active Champion Model: %s", model_name)

    # Check whether real student-test predictions exist
    student_preds_path = ROOT / "results" / "student_test_predictions.csv"
    nodes_arcs_available = False

    # Check if data/ has trip files
    train_dir = ROOT / "data" / "train"
    student_dir = ROOT / "data" / "student_test"
    trip_folders = list(train_dir.glob("*")) + list(student_dir.glob("*")) if train_dir.exists() else []

    if args.synthetic or (not trip_folders and not student_preds_path.exists()):
        logger.info("Running HMM pipeline in self-contained synthetic benchmark mode...")
        nodes_df, arcs_df, track_df, candidates_df, ref_route_df = create_synthetic_trip()
        network = build_road_network(nodes_df, arcs_df)

        trips_track = {"synthetic_01": track_df}
        ref_routes = {"synthetic_01": ref_route_df}

        routes_df, summary_df, metrics_df = run_hmm_pipeline(
            candidates_df=candidates_df,
            road_network=network,
            trips_track=trips_track,
            reference_routes=ref_routes,
            model_name=model_name,
            prob_col="predicted_probability",
            output_dir=args.output_dir,
            alpha=args.alpha,
            beta=args.beta,
        )

        print("\n" + "=" * 60)
        print("HMM Map Matching Complete (Synthetic Benchmark)")
        print("=" * 60)
        print("Decoded Route Summary:")
        print(summary_df.to_string(index=False))
        print("\nGlobal Metrics:")
        print(metrics_df.to_string(index=False))
        print("=" * 60 + "\n")
        return

    # Real predictions mode
    if student_preds_path.exists():
        logger.info("Loading student-test predictions from %s for model '%s'...", student_preds_path, model_name)
        preds_df = pd.read_csv(student_preds_path)
        sub_df = preds_df[preds_df["model_name"] == model_name].copy()
        if sub_df.empty:
            logger.warning("No rows found for model '%s' in student_test_predictions.csv. Using all rows.", model_name)
            sub_df = preds_df.copy()

        # If individual trip folders exist in data/, use their nodes and arcs
        if trip_folders:
            from preprocessing.data_loader import load_trip
            all_summaries = []
            for tf in trip_folders:
                tid = tf.name
                trip_cands = sub_df[sub_df["trip_id"].astype(str) == str(tid)]
                if trip_cands.empty:
                    continue
                trip_data = load_trip(tf, require_route=False)
                network = build_road_network(trip_data["nodes"], trip_data["arcs"])
                matcher = HMMMapMatcher(road_network=network, model_name=model_name, beta=args.beta, alpha=args.alpha)
                res = matcher.match_trip(tid, trip_cands, track_df=trip_data.get("track"), prob_col="predicted_probability")
                all_summaries.append({
                    "trip_id": tid,
                    "continuity": res.continuity_score,
                    "n_obs": len(res.point_matches),
                    "n_arcs": len(res.route_arc_sequence),
                })
            sum_df = pd.DataFrame(all_summaries)
            sum_df.to_csv(Path(args.output_dir) / "hmm_route_summary.csv", index=False)
            logger.info("Processed %d real trips.", len(sum_df))
        else:
            logger.info("Raw trip network files not found in data/. Generating synthetic verification...")
            nodes_df, arcs_df, track_df, candidates_df, ref_route_df = create_synthetic_trip()
            network = build_road_network(nodes_df, arcs_df)
            run_hmm_pipeline(
                candidates_df=candidates_df,
                road_network=network,
                trips_track={"synthetic_01": track_df},
                reference_routes={"synthetic_01": ref_route_df},
                model_name=model_name,
                prob_col="predicted_probability",
                output_dir=args.output_dir,
            )


if __name__ == "__main__":
    main()
