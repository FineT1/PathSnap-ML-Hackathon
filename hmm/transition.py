"""
transition.py – Member 3: Transition Probability Calculation
=============================================================

Responsibility
--------------
Calculate transition probabilities P(s_t = c_j | s_{t-1} = c_i) between
candidate road segments belonging to consecutive GPS observations.

Mathematical Formulation
------------------------
Following Newson & Krumm (2009) "Hidden Markov Map Matching Through Noise
and Sparseness":

  Let:
    d_obs = distance between consecutive GPS observations O_{t-1} and O_t.
    d_net = shortest directed road-network distance from candidate arc c_i to c_j.

  Transition probability:
    P(c_j | c_i) = (1 / beta) * exp(-|d_net - d_obs| / beta)

  Log transition probability:
    log P(c_j | c_i) = -log(beta) - (|d_net - d_obs| / beta)

Physical Reasoning:
  If a vehicle drove between two GPS points, the road distance travelled should
  closely match the observed GPS displacement. If d_net >> d_obs, it implies a
  significant detour or wrong turn. If d_net is infinite (no directed path in graph),
  the transition is physically impossible under standard one-way restrictions.

Handling Disconnections & Unreachable Candidates:
  - If no directed path exists in the graph, d_net = inf.
  - An unreachable transition is penalized with UNREACHABLE_LOG_PROB (-100.0)
    to prevent Viterbi trellis collapse while strongly discouraging illegal jumps.
  - Large time gaps (dt > 60s) relax beta proportionally to allow greater routing leeway.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default beta parameter in metres (rate parameter of exponential distribution)
DEFAULT_BETA = 20.0

# Log probability assigned to disconnected / impossible transitions
UNREACHABLE_LOG_PROB = -100.0
UNREACHABLE_PROB = math.exp(UNREACHABLE_LOG_PROB)

# Maximum plausible driving speed in m/s (e.g. 50 m/s = 180 km/h)
MAX_PLAUSIBLE_SPEED_MPS = 50.0


def compute_network_distance(
    graph: Optional[nx.DiGraph],
    arc_from: int,
    arc_to: int,
    arc_lookup: Dict[int, Tuple[int, int, float]],
    cache: Optional[Dict[Tuple[int, int], float]] = None,
    max_search_dist: float = 3000.0,
) -> float:
    """
    Calculate the directed road network distance from arc_from to arc_to.

    Parameters
    ----------
    graph : nx.DiGraph or None
        Directed road graph with edge attribute 'length_m'.
    arc_from : int
        Source arc ID.
    arc_to : int
        Target arc ID.
    arc_lookup : dict
        Mapping arc_id -> (from_node_id, to_node_id, length_m).
    cache : dict, optional
        Memoization dictionary for shortest paths: (from_node, to_node) -> length.
    max_search_dist : float, default 3000m
        Cutoff to bound shortest-path searches.

    Returns
    -------
    float – network distance in metres (math.inf if no directed path exists).
    """
    # Case 1: Same arc
    if arc_from == arc_to:
        return 0.0

    if arc_from not in arc_lookup or arc_to not in arc_lookup:
        return math.inf

    u_from, v_from, len_from = arc_lookup[arc_from]
    u_to, v_to, len_to = arc_lookup[arc_to]

    # Case 2: Directly connected arcs (v_from == u_to)
    if v_from == u_to:
        return 0.5 * (len_from + len_to)

    if graph is None:
        # Fallback if no graph provided (straight-line topological penalty)
        return math.inf

    # Check cache for node pair (v_from -> u_to)
    node_pair = (v_from, u_to)
    if cache is not None and node_pair in cache:
        node_dist = cache[node_pair]
    else:
        try:
            # Dijkstra shortest path on directed graph weighted by road length
            node_dist = nx.shortest_path_length(
                graph,
                source=v_from,
                target=u_to,
                weight="length_m",
            )
            if cache is not None:
                cache[node_pair] = node_dist
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            node_dist = math.inf
            if cache is not None:
                cache[node_pair] = math.inf

    if math.isinf(node_dist):
        return math.inf

    # Total network distance includes fractions of the endpoint arcs
    return node_dist + 0.5 * (len_from + len_to)


def compute_transition_probability(
    d_net: float,
    d_obs: float,
    beta: float = DEFAULT_BETA,
    time_gap_s: Optional[float] = None,
) -> Tuple[float, float]:
    """
    Compute transition probability and log-probability between two candidates.

    Returns
    -------
    (prob, log_prob)
    """
    if math.isinf(d_net):
        return UNREACHABLE_PROB, UNREACHABLE_LOG_PROB

    # Adjust beta for large time gaps (relax scale parameter)
    eff_beta = beta
    if time_gap_s is not None and time_gap_s > 60.0:
        eff_beta = beta * (1.0 + math.log10(time_gap_s / 60.0))

    # Speed plausibility check
    if time_gap_s is not None and time_gap_s > 0:
        implied_speed = d_net / time_gap_s
        if implied_speed > MAX_PLAUSIBLE_SPEED_MPS:
            # Exponential speed penalty
            excess = implied_speed - MAX_PLAUSIBLE_SPEED_MPS
            log_p = -math.log(eff_beta) - (abs(d_net - d_obs) / eff_beta) - (excess / 10.0)
            log_p = max(UNREACHABLE_LOG_PROB, log_p)
            return math.exp(log_p), log_p

    diff = abs(d_net - d_obs)
    log_p = -math.log(eff_beta) - (diff / eff_beta)
    log_p = max(UNREACHABLE_LOG_PROB, log_p)
    return math.exp(log_p), log_p


class TransitionModel:
    """
    Stateful transition model with caching and fast lookup.
    """

    def __init__(
        self,
        road_network: dict,
        beta: float = DEFAULT_BETA,
    ):
        """
        Parameters
        ----------
        road_network : dict
            Output of build_road_network, containing 'graph' and 'arcs'.
        beta : float, default 20.0
            Scale parameter for transition probability.
        """
        self.graph = road_network.get("graph")
        self.beta = beta
        self.cache: Dict[Tuple[int, int], float] = {}

        # Build arc lookup: arc_id -> (from_node_id, to_node_id, length_m)
        arcs_df = road_network["arcs"]
        self.arc_lookup: Dict[int, Tuple[int, int, float]] = {}
        for row in arcs_df.itertuples(index=False):
            self.arc_lookup[int(row.arc_id)] = (
                int(row.from_node_id),
                int(row.to_node_id),
                float(row.length_m) if pd.notna(row.length_m) else 10.0,
            )

    def transition_log_prob(
        self,
        arc_prev: int,
        arc_curr: int,
        d_obs: float,
        time_gap_s: Optional[float] = None,
    ) -> float:
        """
        Return the log transition probability between two candidate arcs.
        """
        d_net = compute_network_distance(
            graph=self.graph,
            arc_from=arc_prev,
            arc_to=arc_curr,
            arc_lookup=self.arc_lookup,
            cache=self.cache,
        )
        _, log_p = compute_transition_probability(
            d_net=d_net,
            d_obs=d_obs,
            beta=self.beta,
            time_gap_s=time_gap_s,
        )
        return log_p

    def transition_prob(
        self,
        arc_prev: int,
        arc_curr: int,
        d_obs: float,
        time_gap_s: Optional[float] = None,
    ) -> float:
        """
        Return the transition probability between two candidate arcs.
        """
        return math.exp(self.transition_log_prob(arc_prev, arc_curr, d_obs, time_gap_s))


def compute_transition_matrix(
    cands_prev: List[int],
    cands_curr: List[int],
    d_obs: float,
    transition_model: TransitionModel,
    time_gap_s: Optional[float] = None,
) -> np.ndarray:
    """
    Compute the |cands_prev| x |cands_curr| log-transition matrix.

    Returns
    -------
    log_T : 2D np.ndarray of shape (len(cands_prev), len(cands_curr))
    """
    n_prev = len(cands_prev)
    n_curr = len(cands_curr)
    log_T = np.zeros((n_prev, n_curr), dtype=np.float64)

    for i, a_prev in enumerate(cands_prev):
        for j, a_curr in enumerate(cands_curr):
            log_T[i, j] = transition_model.transition_log_prob(
                arc_prev=a_prev,
                arc_curr=a_curr,
                d_obs=d_obs,
                time_gap_s=time_gap_s,
            )

    return log_T
