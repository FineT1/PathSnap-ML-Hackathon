## Member 1: Data Preprocessing and Candidate Generation

### Summary
Complete implementation of the **Member 1** module for the PathSnap GPS map-matching project. This PR adds a clean, modular, and fully-tested preprocessing and candidate generation pipeline.

---

### What's included

| File | Purpose |
|---|---|
| `preprocessing/__init__.py` | Public API — single import point for the whole module |
| `preprocessing/data_loader.py` | Load `.track`, `.nodes`, `.arcs`, `.route` files with stable 0-based IDs |
| `preprocessing/preprocess.py` | GPS track cleaning: null checks, coord validation, timestamps, speed, heading |
| `preprocessing/road_network.py` | Metric projection (EPSG:27700), arc lengths, bearings, NetworkX graph |
| `preprocessing/candidate_generation.py` | Candidate arcs per GPS point, feature engineering, training labels |
| `run_preprocessing.py` | End-to-end pipeline runner (outputs `candidate_features.csv`) |
| `tests/test_preprocessing.py` | 24 unit tests — **all passing** |
| `pathsnap/data.py` | Updated to re-export Member 1 API |
| `pathsnap/features.py` | Updated to re-export candidate generation API |
| `requirements.txt` | Added `networkx` |

---

### Candidate Feature Schema (outputs/candidate_features.csv)

| Column | Role | Description |
|---|---|---|
| `trip_id` | identifier | Trip name |
| `gps_point_id` | identifier | GPS observation index (0-based) |
| `candidate_arc_id` | identifier | Arc index in road network |
| `distance_m` | ML feature | Perpendicular distance from GPS to arc |
| `bearing_diff_deg` | ML feature | GPS heading vs arc bearing difference |
| `speed_mps` | ML feature | Estimated GPS speed |
| `time_gap_s` | ML feature | Seconds since previous fix |
| `segment_length_m` | ML feature | Length of candidate arc |
| `gps_lon / gps_lat` | HMM / viz | GPS coordinates |
| `arc_bearing_deg` | HMM | Arc direction |
| `gps_heading_deg` | HMM | Estimated travel heading |
| `n_candidates` | diagnostic | Candidates per GPS point |
| `label` | target | 1 = reference arc, 0 = not, NaN = no route |

---

### Test Results

Results: 24 passed, 0 failed

---

### Integration Notes for Members 2 and 3

**Member 2 (ML):**

    import pandas as pd
    df = pd.read_csv('outputs/candidate_features.csv')
    # Drop identifiers before training: trip_id, gps_point_id, candidate_arc_id
    # Target: label (0/1). Use class_weight='balanced' — data is imbalanced.

**Member 3 (HMM):**

    from preprocessing import load_trip, preprocess_trip, build_road_network, build_candidate_features
    trip = load_trip('data/student_test/00000007', require_route=False)
    candidates = build_candidate_features(trip_id=..., track_df=..., nodes_df=..., arcs_df=...)
    # Use distance_m for emission probabilities, network['graph'] for transitions.

---

### How to run

    pip install -r requirements.txt
    python run_preprocessing.py          # generates outputs/candidate_features.csv
    python tests/test_preprocessing.py   # run all 24 tests
