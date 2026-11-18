"""
PathSnap HMM Package – Member 3: HMM and Viterbi Route Reconstruction
=====================================================================

Public API:
  - get_emission_probabilities
  - compute_transition_matrix
  - viterbi_decode
  - reconstruct_route
  - HMMMapMatcher / decode_trip / run_hmm_pipeline
"""

from .emission import (
    get_emission_probabilities,
    compute_geometric_emission,
    compute_hybrid_emission,
)
from .transition import (
    compute_network_distance,
    compute_transition_probability,
    compute_transition_matrix,
    TransitionModel,
)
from .viterbi import (
    viterbi_decode,
    ViterbiResult,
)
from .route_reconstruction import (
    reconstruct_route,
    compute_route_continuity,
    RouteReconstructionResult,
)
from .hmm_pipeline import (
    HMMMapMatcher,
    decode_trip,
    run_hmm_pipeline,
)

__all__ = [
    "get_emission_probabilities",
    "compute_geometric_emission",
    "compute_hybrid_emission",
    "compute_network_distance",
    "compute_transition_probability",
    "compute_transition_matrix",
    "TransitionModel",
    "viterbi_decode",
    "ViterbiResult",
    "reconstruct_route",
    "compute_route_continuity",
    "RouteReconstructionResult",
    "HMMMapMatcher",
    "decode_trip",
    "run_hmm_pipeline",
]
