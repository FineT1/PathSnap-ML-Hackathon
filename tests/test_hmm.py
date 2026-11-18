"""
test_hmm.py – Unit and Integration Tests for Member 3: HMM and Viterbi Map Matching
====================================================================================

Tests cover:
  1.  Emission probability calculations (Gaussian & ML).
  2.  Probability clipping (prevents underflow).
  3.  Positive-class probability extraction from model.
  4.  Transition probability calculation (exponential penalty).
  5.  Directed graph connectivity (one-way restriction).
  6.  Shortest-path network distance.
  7.  Unreachable transitions (disconnected components).
  8.  Viterbi initialization (t=0 prior).
  9.  Viterbi recursion (t >= 1 dynamic programming step).
  10. Backpointer reconstruction and backtracking.
  11. Correct sequence selection against known ground-truth path.
  12. Empty candidate list handling.
  13. Single-observation trip (T=1).
  14. Multiple GPS observations (T=5).
  15. Multiple trips processing.
  16. Large GPS time gaps (scale relaxation).
  17. Repeated GPS coordinates (zero displacement).
  18. Numerical stability (extreme values, near-zero probabilities).
  19. Route continuity metric calculation.
  20. Integration with selected ML model / mock classifier.

Run with:
  python tests/test_hmm.py
or:
  pytest tests/test_hmm.py -v
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

# Project root setup
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hmm.emission import (
    EPSILON,
    compute_geometric_emission,
    compute_hybrid_emission,
    extract_positive_class_probability,
    get_emission_probabilities,
)
from hmm.transition import (
    UNREACHABLE_LOG_PROB,
    TransitionModel,
    compute_network_distance,
    compute_transition_matrix,
    compute_transition_probability,
)
from hmm.viterbi import (
    ViterbiResult,
    viterbi_decode,
)
from hmm.route_reconstruction import (
    compute_route_continuity,
    reconstruct_route,
)
from hmm.hmm_pipeline import (
    HMMMapMatcher,
    run_hmm_pipeline,
)
from run_hmm import create_synthetic_trip
from preprocessing.road_network import build_road_network


class MockEstimator:
    """Mock classifier conforming to scikit-learn interface for unit tests."""
    def __init__(self, classes=None):
        self.classes_ = np.array([0, 1]) if classes is None else np.array(classes)

    def predict_proba(self, X):
        n = len(X)
        # Mock probabilities: 0.8 for class 1, 0.2 for class 0
        p1 = np.full(n, 0.82)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])


# ---------------------------------------------------------------------------
# Test Functions
# ---------------------------------------------------------------------------

def test_1_emission_calculation():
    """Verify Gaussian emission likelihood formula."""
    d0 = compute_geometric_emission(0.0, sigma_z=20.0)
    d20 = compute_geometric_emission(20.0, sigma_z=20.0)
    assert d0 > d20, "Distance 0m must have higher likelihood than distance 20m"
    expected_d0 = 1.0 / (math.sqrt(2 * math.pi) * 20.0)
    assert math.isclose(float(d0), expected_d0, rel_tol=1e-3)


def test_2_probability_clipping():
    """Verify probability clipping bounds."""
    extreme_p = compute_geometric_emission(10000.0, sigma_z=20.0)
    assert extreme_p >= EPSILON, f"Probability must be bounded below by EPSILON, got {extreme_p}"
    assert math.log(extreme_p) > -100.0, "Log probability must be finite"


def test_3_positive_class_probability_extraction():
    """Verify extraction of P(y=1) regardless of class column ordering."""
    mock = MockEstimator(classes=[0, 1])
    X = np.zeros((3, 2))
    p = extract_positive_class_probability(mock, X)
    assert len(p) == 3
    assert math.isclose(p[0], 0.82, rel_tol=1e-3)

    # Inverted classes [1, 0]
    class InvertedMock:
        classes_ = np.array([1, 0])
        def predict_proba(self, X):
            return np.array([[0.75, 0.25]] * len(X))
    p_inv = extract_positive_class_probability(InvertedMock(), X)
    assert math.isclose(p_inv[0], 0.75, rel_tol=1e-3)


def test_4_transition_probability_exponential():
    """Verify exponential penalty for difference between network and GPS distance."""
    # Match: d_net == d_obs
    prob_exact, log_exact = compute_transition_probability(d_net=50.0, d_obs=50.0, beta=20.0)
    # Detour: d_net = 150m, d_obs = 50m
    prob_detour, log_detour = compute_transition_probability(d_net=150.0, d_obs=50.0, beta=20.0)
    assert prob_exact > prob_detour, "Exact distance match must have higher transition probability than detour"
    assert math.isclose(log_exact, -math.log(20.0), rel_tol=1e-3)
    assert math.isclose(log_detour, -math.log(20.0) - (100.0 / 20.0), rel_tol=1e-3)


def test_5_directed_graph_connectivity():
    """Verify directed road connectivity: A->B is valid, but B->A is invalid on one-way road."""
    G = nx.DiGraph()
    G.add_edge(0, 1, arc_id=10, length_m=100.0)
    G.add_edge(1, 2, arc_id=11, length_m=100.0)

    arc_lookup = {
        10: (0, 1, 100.0),
        11: (1, 2, 100.0),
    }
    # Forward transition: arc 10 to 11
    d_fwd = compute_network_distance(G, arc_from=10, arc_to=11, arc_lookup=arc_lookup)
    assert math.isfinite(d_fwd), "Forward transition on directed graph must be reachable"

    # Reverse transition: arc 11 to 10 (one-way restriction violation)
    d_rev = compute_network_distance(G, arc_from=11, arc_to=10, arc_lookup=arc_lookup)
    assert math.isinf(d_rev), "Reverse transition against directed edge must be unreachable (inf)"


def test_6_shortest_path_distance():
    """Verify multi-hop shortest path calculation."""
    G = nx.DiGraph()
    G.add_edge(0, 1, arc_id=1, length_m=50.0)
    G.add_edge(1, 2, arc_id=2, length_m=50.0)
    G.add_edge(2, 3, arc_id=3, length_m=50.0)

    arc_lookup = {
        1: (0, 1, 50.0),
        2: (1, 2, 50.0),
        3: (2, 3, 50.0),
    }
    # From arc 1 to arc 3 (traversing through node 1 -> 2 -> 3)
    d = compute_network_distance(G, arc_from=1, arc_to=3, arc_lookup=arc_lookup)
    # v_from=1, u_to=2 -> shortest path is node 1->2 (length 50) + 0.5*(50 + 50) = 100m
    assert math.isclose(d, 100.0, rel_tol=1e-3)


def test_7_unreachable_transitions():
    """Verify that disconnected components receive bounded penalty without crashing."""
    G = nx.DiGraph()
    G.add_edge(0, 1, arc_id=1, length_m=50.0)
    G.add_edge(10, 11, arc_id=2, length_m=50.0)  # completely disconnected component

    arc_lookup = {
        1: (0, 1, 50.0),
        2: (10, 11, 50.0),
    }
    d = compute_network_distance(G, arc_from=1, arc_to=2, arc_lookup=arc_lookup)
    assert math.isinf(d)
    prob, log_p = compute_transition_probability(d_net=d, d_obs=20.0)
    assert log_p == UNREACHABLE_LOG_PROB
    assert prob > 0.0


def test_8_viterbi_initialization():
    """Verify t=0 prior and emission scoring in Viterbi."""
    nodes_df, arcs_df, track_df, candidates_df, _ = create_synthetic_trip()
    network = build_road_network(nodes_df, arcs_df)
    matcher = HMMMapMatcher(road_network=network, emission_mode="geometric")

    res = matcher.match_trip("synthetic_01", candidates_df.iloc[:3], track_df=track_df.iloc[:1])
    assert len(res.point_matches) == 1
    # Candidate 0 has lowest distance (2.2m) -> should be selected
    assert res.point_matches.iloc[0]["selected_arc_id"] == 0


def test_9_viterbi_recursion():
    """Verify dynamic programming step aggregates emissions and transitions."""
    nodes_df, arcs_df, track_df, candidates_df, _ = create_synthetic_trip()
    network = build_road_network(nodes_df, arcs_df)
    matcher = HMMMapMatcher(road_network=network, beta=20.0)

    res = matcher.match_trip("synthetic_01", candidates_df, track_df=track_df, prob_col="predicted_probability")
    # Scores must exist and be finite
    assert (res.point_matches["viterbi_score"] < 0).all()
    assert not res.point_matches["viterbi_score"].isna().any()


def test_10_backpointer_reconstruction():
    """Verify backtracking reconstructs the chronological sequence."""
    nodes_df, arcs_df, track_df, candidates_df, _ = create_synthetic_trip()
    network = build_road_network(nodes_df, arcs_df)
    matcher = HMMMapMatcher(road_network=network)

    res = matcher.match_trip("synthetic_01", candidates_df, track_df=track_df, prob_col="predicted_probability")
    steps = res.point_matches["sequence_position"].tolist()
    assert steps == [0, 1, 2, 3], "Sequence positions must be chronologically contiguous"


def test_11_correct_sequence_selection():
    """
    Key Map-Matching Test:
    At GPS fix 2, candidate arc 5 is closer in straight distance than arc 1,
    BUT arc 5 would require an impossible/long detour. Viterbi must select arc 1!
    """
    nodes_df, arcs_df, track_df, candidates_df, _ = create_synthetic_trip()
    network = build_road_network(nodes_df, arcs_df)
    matcher = HMMMapMatcher(road_network=network, beta=20.0)

    res = matcher.match_trip("synthetic_01", candidates_df, track_df=track_df, prob_col="predicted_probability")
    decoded_arcs = res.point_matches["selected_arc_id"].tolist()
    # True vehicle route is arc 0 -> arc 0 -> arc 1 -> arc 1
    assert decoded_arcs == [0, 0, 1, 1], f"Viterbi should reconstruct [0, 0, 1, 1], got {decoded_arcs}"


def test_12_empty_candidates_handling():
    """Verify graceful handling when candidates list is empty."""
    nodes_df, arcs_df, track_df, _, _ = create_synthetic_trip()
    network = build_road_network(nodes_df, arcs_df)
    matcher = HMMMapMatcher(road_network=network)

    empty_df = pd.DataFrame(columns=["trip_id", "gps_point_id", "candidate_arc_id", "distance_m"])
    try:
        matcher.match_trip("empty_trip", empty_df)
        assert False, "Should raise ValueError for empty candidate DataFrame"
    except ValueError:
        pass


def test_13_single_observation_trip():
    """Verify Viterbi operates correctly on single GPS fix trips (T=1)."""
    nodes_df, arcs_df, track_df, candidates_df, _ = create_synthetic_trip()
    network = build_road_network(nodes_df, arcs_df)
    matcher = HMMMapMatcher(road_network=network)

    single_cand = candidates_df[candidates_df["gps_point_id"] == 0]
    res = matcher.match_trip("synthetic_single", single_cand)
    assert len(res.point_matches) == 1
    assert res.point_matches.iloc[0]["selected_arc_id"] == 0
    assert res.continuity_score == 1.0


def test_14_multiple_observations_trajectory():
    """Verify trajectory with multiple GPS fixes decodes smoothly."""
    nodes_df, arcs_df, track_df, candidates_df, _ = create_synthetic_trip()
    network = build_road_network(nodes_df, arcs_df)
    matcher = HMMMapMatcher(road_network=network)

    res = matcher.match_trip("synthetic_01", candidates_df, track_df=track_df)
    assert len(res.point_matches) == 4
    assert len(res.route_arc_sequence) >= 4


def test_15_multiple_trips_pipeline():
    """Verify pipeline processes multiple distinct trips and aggregates summaries."""
    nodes_df, arcs_df, track_df, candidates_df, ref_route_df = create_synthetic_trip()
    network = build_road_network(nodes_df, arcs_df)

    cands_trip1 = candidates_df.copy()
    cands_trip1["trip_id"] = "trip_alpha"
    cands_trip2 = candidates_df.copy()
    cands_trip2["trip_id"] = "trip_beta"
    both_cands = pd.concat([cands_trip1, cands_trip2], ignore_index=True)

    routes_df, summary_df, metrics_df = run_hmm_pipeline(
        candidates_df=both_cands,
        road_network=network,
        prob_col="predicted_probability",
    )
    assert len(summary_df) == 2
    assert set(summary_df["trip_id"]) == {"trip_alpha", "trip_beta"}
    assert metrics_df["n_trips"].iloc[0] == 2


def test_16_large_gps_time_gaps():
    """Verify that large time gaps relax beta without failing."""
    prob_normal, _ = compute_transition_probability(d_net=100.0, d_obs=20.0, beta=20.0, time_gap_s=10.0)
    prob_large_gap, _ = compute_transition_probability(d_net=100.0, d_obs=20.0, beta=20.0, time_gap_s=600.0)
    assert prob_large_gap > prob_normal, "Larger time gap should relax scale parameter"


def test_17_repeated_gps_coordinates():
    """Verify zero displacement (stationary vehicle)."""
    prob, log_p = compute_transition_probability(d_net=0.0, d_obs=0.0, beta=20.0)
    assert math.isclose(prob, 1.0 / 20.0, rel_tol=1e-3)


def test_18_numerical_stability():
    """Verify extreme probability clipping and zero underflow."""
    hybrid_p = compute_hybrid_emission(ml_prob=0.0, distance_m=1000.0, alpha=0.5, sigma_z=20.0)
    assert hybrid_p >= EPSILON
    assert math.isfinite(math.log(hybrid_p))


def test_19_route_continuity_metric():
    """Verify route continuity calculation."""
    arc_lookup = {
        0: (0, 1, 50.0),
        1: (1, 2, 50.0),
        2: (5, 6, 50.0),  # disconnected arc
    }
    # Continuous: 0 -> 1
    assert compute_route_continuity([0, 1], arc_lookup) == 1.0
    # Discontinuous: 0 -> 2 (node 1 != node 5)
    assert compute_route_continuity([0, 2], arc_lookup) == 0.0


def test_20_integration_with_ml_model():
    """Verify end-to-end integration with mock scikit-learn model."""
    nodes_df, arcs_df, track_df, candidates_df, _ = create_synthetic_trip()
    network = build_road_network(nodes_df, arcs_df)

    mock_model = MockEstimator()
    candidates_with_feats = candidates_df.copy()
    candidates_with_feats["feat1"] = np.random.randn(len(candidates_with_feats))
    candidates_with_feats["feat2"] = np.random.randn(len(candidates_with_feats))

    matcher = HMMMapMatcher(
        road_network=network,
        model=mock_model,
        features=["feat1", "feat2"],
        emission_mode="hybrid",
    )
    res = matcher.match_trip("synthetic_ml", candidates_with_feats, track_df=track_df)
    assert len(res.point_matches) == 4
    assert res.continuity_score > 0.0


# ---------------------------------------------------------------------------
# Test Runner
# ---------------------------------------------------------------------------

def run_all_tests():
    tests = [
        ("1.  Emission probability calculations",          test_1_emission_calculation),
        ("2.  Probability clipping & underflow prevention", test_2_probability_clipping),
        ("3.  Positive-class probability extraction",       test_3_positive_class_probability_extraction),
        ("4.  Transition probability (exponential rate)",   test_4_transition_probability_exponential),
        ("5.  Directed graph connectivity (one-way check)", test_5_directed_graph_connectivity),
        ("6.  Shortest-path network distance",              test_6_shortest_path_distance),
        ("7.  Unreachable transitions (penalty)",           test_7_unreachable_transitions),
        ("8.  Viterbi initialization (t=0 prior)",          test_8_viterbi_initialization),
        ("9.  Viterbi recursion (t >= 1)",                  test_9_viterbi_recursion),
        ("10. Backpointer reconstruction",                  test_10_backpointer_reconstruction),
        ("11. Correct sequence selection (known truth)",    test_11_correct_sequence_selection),
        ("12. Empty candidate lists handling",              test_12_empty_candidates_handling),
        ("13. Single-observation trip (T=1)",               test_13_single_observation_trip),
        ("14. Multiple GPS observations trajectory",        test_14_multiple_observations_trajectory),
        ("15. Multiple trips pipeline processing",          test_15_multiple_trips_pipeline),
        ("16. Large GPS time gaps relaxation",              test_16_large_gps_time_gaps),
        ("17. Repeated GPS coordinates (stopped vehicle)",  test_17_repeated_gps_coordinates),
        ("18. Numerical stability safeguards",              test_18_numerical_stability),
        ("19. Route continuity metric calculation",         test_19_route_continuity_metric),
        ("20. ML model integration interface",              test_20_integration_with_ml_model),
    ]

    print("\n" + "=" * 65)
    print("PathSnap – Member 3 (HMM & Viterbi) Test Suite")
    print("=" * 65)

    passed, failed = [], []
    for name, fn in tests:
        try:
            fn()
            passed.append(name)
            print(f"  ✓ {name}")
        except Exception as exc:
            failed.append((name, exc))
            print(f"  ✗ {name} -> {type(exc).__name__}: {exc}")

    print("\n" + "=" * 65)
    print(f"Results: {len(passed)} passed, {len(failed)} failed")
    print("=" * 65 + "\n")

    return len(failed) == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
