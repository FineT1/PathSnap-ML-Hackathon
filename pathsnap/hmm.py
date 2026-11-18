"""
pathsnap/hmm.py
===============
HMM over candidate segments: emissions from distance & ML probabilities,
transitions from road network connectivity, Viterbi decoding.

This module re-exports the complete Member 3 HMM & Viterbi API:

    from pathsnap.hmm import (
        HMMMapMatcher,
        decode_trip,
        run_hmm_pipeline,
        viterbi_decode,
        get_emission_probabilities,
        reconstruct_route,
        compute_route_continuity,
    )
"""

from hmm import (
    HMMMapMatcher,
    RouteReconstructionResult,
    TransitionModel,
    ViterbiResult,
    compute_geometric_emission,
    compute_hybrid_emission,
    compute_network_distance,
    compute_route_continuity,
    compute_transition_matrix,
    compute_transition_probability,
    decode_trip,
    get_emission_probabilities,
    reconstruct_route,
    run_hmm_pipeline,
    viterbi_decode,
)

__all__ = [
    "HMMMapMatcher",
    "RouteReconstructionResult",
    "TransitionModel",
    "ViterbiResult",
    "compute_geometric_emission",
    "compute_hybrid_emission",
    "compute_network_distance",
    "compute_route_continuity",
    "compute_transition_matrix",
    "compute_transition_probability",
    "decode_trip",
    "get_emission_probabilities",
    "reconstruct_route",
    "run_hmm_pipeline",
    "viterbi_decode",
]
