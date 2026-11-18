"""
route_reconstruction.py – Member 3: Road Sequence Reconstruction
=================================================================

Responsibility
--------------
Reconstruct the complete, physically continuous road route from the Viterbi-decoded
arc sequence:
  1. Validates directed graph connectivity between consecutive selected arcs.
  2. Identifies and infers any unobserved intermediate road arcs between sparse GPS fixes
     using shortest-path traversal on Member 1's directed road network graph.
  3. Clearly distinguishes directly matched arcs (from GPS fixes) from inferred intermediate arcs.
  4. Collapses redundant consecutive duplicate arcs for route sequence summarization.
  5. Computes quantitative Route Continuity metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import networkx as nx
import pandas as pd

from .viterbi import ViterbiResult

logger = logging.getLogger(__name__)


@dataclass
class RouteSegment:
    """A single arc in the reconstructed route."""
    sequence_index: int
    arc_id: int
    from_node_id: int
    to_node_id: int
    length_m: float
    source: str           # 'direct_gps_match' or 'inferred_path'
    gps_point_id: Optional[int]


@dataclass
class RouteReconstructionResult:
    """Complete route reconstruction outcome."""
    trip_id: str
    point_matches: pd.DataFrame
    reconstructed_route: pd.DataFrame
    route_arc_sequence: List[int]
    collapsed_arc_sequence: List[int]
    continuity_score: float
    n_inferred_arcs: int
    is_fully_connected: bool


def compute_route_continuity(
    arc_sequence: List[int],
    arc_lookup: Dict[int, Tuple[int, int, float]],
    graph: Optional[nx.DiGraph] = None,
) -> float:
    """
    Calculate the proportion of consecutive arc transitions that are physically
    connected in the directed road network.

    Parameters
    ----------
    arc_sequence : list of int
        Chronological list of arc IDs.
    arc_lookup : dict
        Mapping arc_id -> (from_node, to_node, length_m).
    graph : nx.DiGraph, optional
        Directed road network.

    Returns
    -------
    float – continuity score between 0.0 and 1.0 (1.0 = perfectly continuous).
    """
    if len(arc_sequence) <= 1:
        return 1.0

    valid_transitions = 0
    total_transitions = len(arc_sequence) - 1

    for i in range(total_transitions):
        a1, a2 = arc_sequence[i], arc_sequence[i + 1]

        # Staying on the same arc is continuous
        if a1 == a2:
            valid_transitions += 1
            continue

        if a1 not in arc_lookup or a2 not in arc_lookup:
            continue

        u1, v1, _ = arc_lookup[a1]
        u2, v2, _ = arc_lookup[a2]

        # Direct head-to-tail connection
        if v1 == u2:
            valid_transitions += 1
        elif graph is not None and nx.has_path(graph, v1, u2):
            valid_transitions += 1

    return float(valid_transitions / total_transitions)


def reconstruct_route(
    viterbi_res: ViterbiResult,
    road_network: dict,
    infer_intermediate: bool = True,
) -> RouteReconstructionResult:
    """
    Reconstruct the full route from Viterbi decoding results.

    Parameters
    ----------
    viterbi_res : ViterbiResult
        Output of viterbi_decode.
    road_network : dict
        Output of build_road_network (contains 'arcs' DataFrame and 'graph' DiGraph).
    infer_intermediate : bool, default True
        Whether to route through intermediate arcs when GPS fixes are sparse.

    Returns
    -------
    RouteReconstructionResult
    """
    trip_id = viterbi_res.trip_id
    point_df = viterbi_res.to_dataframe()

    graph = road_network.get("graph")
    arcs_df = road_network["arcs"]

    # Build node-edge mapping: (u, v) -> arc_id
    arc_lookup: Dict[int, Tuple[int, int, float]] = {}
    edge_to_arc: Dict[Tuple[int, int], int] = {}

    for row in arcs_df.itertuples(index=False):
        aid = int(row.arc_id)
        u, v = int(row.from_node_id), int(row.to_node_id)
        length = float(row.length_m) if pd.notna(row.length_m) else 10.0
        arc_lookup[aid] = (u, v, length)
        # Store directed edge
        if (u, v) not in edge_to_arc:
            edge_to_arc[(u, v)] = aid

    decoded_arcs = viterbi_res.arc_sequence
    T = len(decoded_arcs)

    if T == 0:
        return RouteReconstructionResult(
            trip_id=trip_id,
            point_matches=point_df,
            reconstructed_route=pd.DataFrame(),
            route_arc_sequence=[],
            collapsed_arc_sequence=[],
            continuity_score=1.0,
            n_inferred_arcs=0,
            is_fully_connected=True,
        )

    # ------------------------------------------------------------------ #
    # 1. Traverse and infer connecting arcs
    # ------------------------------------------------------------------ #
    full_route_segments: List[RouteSegment] = []
    n_inferred = 0
    seq_idx = 0
    fully_connected = True

    # First point
    first_arc = decoded_arcs[0]
    u0, v0, len0 = arc_lookup.get(first_arc, (-1, -1, 0.0))
    full_route_segments.append(
        RouteSegment(
            sequence_index=seq_idx,
            arc_id=first_arc,
            from_node_id=u0,
            to_node_id=v0,
            length_m=len0,
            source="direct_gps_match",
            gps_point_id=viterbi_res.steps[0].gps_point_id,
        )
    )
    seq_idx += 1

    for t in range(1, T):
        prev_arc = decoded_arcs[t - 1]
        curr_arc = decoded_arcs[t]
        curr_gps_id = viterbi_res.steps[t].gps_point_id

        if prev_arc == curr_arc:
            # Same arc repeated
            full_route_segments.append(
                RouteSegment(
                    sequence_index=seq_idx,
                    arc_id=curr_arc,
                    from_node_id=u0,
                    to_node_id=v0,
                    length_m=len0,
                    source="direct_gps_match",
                    gps_point_id=curr_gps_id,
                )
            )
            seq_idx += 1
            continue

        u_prev, v_prev, len_prev = arc_lookup.get(prev_arc, (-1, -1, 0.0))
        u_curr, v_curr, len_curr = arc_lookup.get(curr_arc, (-1, -1, 0.0))

        if v_prev == u_curr:
            # Direct connection: no intermediate arcs required
            pass
        elif infer_intermediate and graph is not None and v_prev != -1 and u_curr != -1:
            try:
                # Find intermediate node path
                node_path = nx.shortest_path(graph, source=v_prev, target=u_curr, weight="length_m")
                # Add intermediate edges
                for k in range(len(node_path) - 1):
                    nu, nv = node_path[k], node_path[k + 1]
                    intermediate_arc = edge_to_arc.get((nu, nv))
                    if intermediate_arc is not None:
                        iu, iv, ilen = arc_lookup.get(intermediate_arc, (nu, nv, 10.0))
                        full_route_segments.append(
                            RouteSegment(
                                sequence_index=seq_idx,
                                arc_id=intermediate_arc,
                                from_node_id=iu,
                                to_node_id=iv,
                                length_m=ilen,
                                source="inferred_path",
                                gps_point_id=None,
                            )
                        )
                        seq_idx += 1
                        n_inferred += 1
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                fully_connected = False
        else:
            fully_connected = False

        # Add the current directly-matched arc
        full_route_segments.append(
            RouteSegment(
                sequence_index=seq_idx,
                arc_id=curr_arc,
                from_node_id=u_curr,
                to_node_id=v_curr,
                length_m=len_curr,
                source="direct_gps_match",
                gps_point_id=curr_gps_id,
            )
        )
        seq_idx += 1

    # ------------------------------------------------------------------ #
    # 2. Extract sequences & collapsed sequences
    # ------------------------------------------------------------------ #
    route_arc_seq = [s.arc_id for s in full_route_segments]

    # Collapse consecutive duplicate arcs for route overview
    collapsed_seq: List[int] = []
    for arc in route_arc_seq:
        if not collapsed_seq or collapsed_seq[-1] != arc:
            collapsed_seq.append(arc)

    # Calculate continuity on the decoded sequence
    continuity = compute_route_continuity(decoded_arcs, arc_lookup, graph)

    # Build DataFrame
    route_df = pd.DataFrame([
        {
            "trip_id": trip_id,
            "sequence_index": s.sequence_index,
            "arc_id": s.arc_id,
            "from_node_id": s.from_node_id,
            "to_node_id": s.to_node_id,
            "length_m": s.length_m,
            "source": s.source,
            "gps_point_id": s.gps_point_id,
        }
        for s in full_route_segments
    ])

    return RouteReconstructionResult(
        trip_id=trip_id,
        point_matches=point_df,
        reconstructed_route=route_df,
        route_arc_sequence=route_arc_seq,
        collapsed_arc_sequence=collapsed_seq,
        continuity_score=continuity,
        n_inferred_arcs=n_inferred,
        is_fully_connected=fully_connected,
    )
