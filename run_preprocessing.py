"""
run_preprocessing.py – Member 1 entry-point script
====================================================

Run this script from the project root to:

  1. Load every training trip.
  2. Preprocess GPS tracks and road networks.
  3. Generate candidate features and reference labels.
  4. Save outputs/candidate_features.csv and outputs/dataset_summary.csv.

Usage
-----
  cd PathSnap                        # project root
  python run_preprocessing.py

Optional flags
--------------
  --radius  <float>   Search radius in metres  (default: 50)
  --maxcand <int>     Max candidates per point  (default: 10)
  --crs     <str>     EPSG CRS string           (default: EPSG:27700)
  --drop-invalid      Remove invalid GPS points before candidate generation.
  --include-student   Also process student_test trips (no labels attached).
  --verbose           Enable DEBUG logging.

The script respects data/splits.csv so only trips listed there are processed.
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Make sure the project root is on the Python path when running as a script.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing import (
    load_trip,
    preprocess_trip,
    build_road_network,
    build_candidate_features,
    attach_training_labels,
    prepare_training_candidates,
)
from preprocessing.data_loader import list_trips
from preprocessing.preprocess import summarise_dataset
from preprocessing.candidate_generation import SCHEMA_COLS


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="PathSnap Member 1 – preprocessing pipeline")
    p.add_argument("--radius",    type=float, default=50.0,   help="Search radius in metres")
    p.add_argument("--maxcand",   type=int,   default=10,     help="Max candidates per GPS point")
    p.add_argument("--crs",       type=str,   default="EPSG:27700", help="Metric CRS")
    p.add_argument("--drop-invalid", action="store_true",     help="Drop invalid GPS points")
    p.add_argument("--include-student", action="store_true",  help="Also process student_test trips")
    p.add_argument("--verbose",   action="store_true",        help="Enable DEBUG logging")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Configure logging.
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("run_preprocessing")

    # Paths.
    data_root   = PROJECT_ROOT / "data"
    train_dir   = data_root / "train"
    student_dir = data_root / "student_test"
    out_dir     = PROJECT_ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    splits_path = data_root / "splits.csv"

    # ------------------------------------------------------------------ #
    # 1. Load splits CSV to know which trips exist and their partition.
    # ------------------------------------------------------------------ #
    if splits_path.exists():
        splits = pd.read_csv(splits_path, dtype={"trip_id": str})
        log.info("Loaded splits.csv: %d trips listed.", len(splits))
    else:
        log.warning("splits.csv not found; will process all trips found in data/.")
        splits = None

    # ------------------------------------------------------------------ #
    # 2. Discover trip folders.
    # ------------------------------------------------------------------ #
    train_trips   = list_trips(train_dir)   if train_dir.is_dir()   else pd.DataFrame()
    student_trips = list_trips(student_dir) if student_dir.is_dir() else pd.DataFrame()

    log.info("Found %d training trip folders.", len(train_trips))
    log.info("Found %d student_test trip folders.", len(student_trips))

    if train_trips.empty:
        log.error(
            "No training trips found under %s. "
            "Please copy the dataset into data/train/ first.", train_dir
        )
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # 3. Process training trips → generate labelled candidate features.
    # ------------------------------------------------------------------ #
    all_candidates = []
    all_trip_dicts = []

    for _, trip_row in train_trips.iterrows():
        tid    = trip_row["trip_id"]
        folder = Path(trip_row["folder"])

        log.info("─── Processing training trip: %s ───", tid)

        try:
            cands = prepare_training_candidates(
                trip_folder      = folder,
                trip_id          = tid,
                search_radius_m  = args.radius,
                max_candidates   = args.maxcand,
                crs              = args.crs,
                drop_invalid_gps = args.drop_invalid,
            )
            all_candidates.append(cands)

            # Also load a plain trip dict for the summary.
            td = load_trip(folder, trip_id=tid)
            td = preprocess_trip(td)
            all_trip_dicts.append(td)

            log.info("  → %d candidate rows", len(cands))

        except FileNotFoundError as exc:
            log.warning("  Skipping trip %s: %s", tid, exc)
        except Exception as exc:
            log.error("  Error processing trip %s: %s", tid, exc, exc_info=True)

    # ------------------------------------------------------------------ #
    # 4. Optionally process student_test trips (no route → no labels).
    # ------------------------------------------------------------------ #
    if args.include_student and not student_trips.empty:
        log.info("Processing %d student_test trips (no labels).", len(student_trips))

        for _, trip_row in student_trips.iterrows():
            tid    = trip_row["trip_id"]
            folder = Path(trip_row["folder"])
            log.info("─── Processing student_test trip: %s ───", tid)

            try:
                td = load_trip(folder, trip_id=tid, require_route=False)
                td = preprocess_trip(td, drop_invalid_gps=args.drop_invalid)

                network = build_road_network(td["nodes"], td["arcs"], crs=args.crs)
                cands = build_candidate_features(
                    trip_id         = tid,
                    track_df        = td["track"],
                    nodes_df        = td["nodes"],
                    arcs_df         = td["arcs"],
                    search_radius_m = args.radius,
                    max_candidates  = args.maxcand,
                    crs             = args.crs,
                    network         = network,
                )
                # No route → label column is NaN.
                cands["label"] = float("nan")
                all_candidates.append(cands)
                all_trip_dicts.append(td)
                log.info("  → %d candidate rows", len(cands))

            except FileNotFoundError as exc:
                log.warning("  Skipping student trip %s: %s", tid, exc)
            except Exception as exc:
                log.error("  Error on student trip %s: %s", tid, exc, exc_info=True)

    # ------------------------------------------------------------------ #
    # 5. Save candidate_features.csv
    # ------------------------------------------------------------------ #
    if not all_candidates:
        log.error("No candidate rows were generated. Check data directory.")
        sys.exit(1)

    combined = pd.concat(all_candidates, ignore_index=True)
    # Ensure columns are in the agreed schema order.
    for col in SCHEMA_COLS:
        if col not in combined.columns:
            combined[col] = float("nan")
    combined = combined[SCHEMA_COLS]

    cand_path = out_dir / "candidate_features.csv"
    combined.to_csv(cand_path, index=False)
    log.info("Saved %d candidate rows to %s", len(combined), cand_path)

    # ------------------------------------------------------------------ #
    # 6. Save dataset_summary.csv
    # ------------------------------------------------------------------ #
    if all_trip_dicts:
        summary = summarise_dataset(all_trip_dicts)
        summary_path = out_dir / "dataset_summary.csv"
        summary.to_csv(summary_path, index=False)
        log.info("Saved dataset summary (%d trips) to %s", len(summary), summary_path)

    # ------------------------------------------------------------------ #
    # 7. Print a quick data quality report.
    # ------------------------------------------------------------------ #
    log.info("")
    log.info("═══ Candidate Feature Report ═══")
    log.info("  Total rows         : %d", len(combined))
    log.info("  Unique trips       : %d", combined["trip_id"].nunique())
    log.info("  Unique GPS points  : %d", combined["gps_point_id"].nunique())
    log.info("  Label distribution :")
    if combined["label"].notna().any():
        vc = combined["label"].value_counts(dropna=True)
        for lbl, cnt in vc.items():
            log.info("    label=%s : %d (%.1f %%)", int(lbl), cnt, 100 * cnt / len(combined))
    log.info("  Null distance_m    : %d", combined["distance_m"].isna().sum())
    log.info("  Null bearing_diff  : %d", combined["bearing_diff_deg"].isna().sum())
    log.info("Done.")


if __name__ == "__main__":
    main()
