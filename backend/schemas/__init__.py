from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

# Dashboard schemas
class DashboardSummaryResponse(BaseModel):
    total_trips: int
    total_observations: int
    total_candidates: int
    reconstructed_trips: int
    champion_model: str
    champion_top1_accuracy: float
    champion_log_loss: float
    mean_route_continuity: float
    total_inferred_arcs: int
    status: str

# Model schemas
class ModelMetricItem(BaseModel):
    model: str
    family: str
    cv_macro_f1_mean: float
    cv_top1_mean: float
    cv_log_loss_mean: float
    student_macro_f1: float
    student_top1: float
    student_log_loss: float
    student_infer_us_per_row: float
    is_champion: bool

class ModelComparisonResponse(BaseModel):
    champion_model: str
    models: List[ModelMetricItem]

# Trip & Route schemas
class TripListItem(BaseModel):
    trip_id: str
    partition: str
    n_gps_points: Optional[int] = None
    duration_s: Optional[float] = None
    route_status: str

class DecodedPoint(BaseModel):
    sequence_position: int
    gps_point_id: int
    selected_arc_id: int
    emission_probability: float
    viterbi_score: float
    lat: Optional[float] = None
    lon: Optional[float] = None
    timestamp_s: Optional[float] = None

class ReconstructedSegment(BaseModel):
    sequence_index: int
    arc_id: int
    from_node_id: int
    to_node_id: int
    length_m: float
    source: str  # 'direct_gps_match' or 'inferred_path'
    is_inferred: bool = False
    coords: List[List[float]] = []  # [[lat, lon], [lat, lon], ...]
    geojson: Optional[Dict[str, Any]] = None

class TripRouteDetailResponse(BaseModel):
    trip_id: str
    model_used: str
    continuity_score: float
    is_fully_connected: bool
    n_observations: int
    n_route_arcs: int
    n_inferred_arcs: int
    points: List[DecodedPoint]
    segments: List[ReconstructedSegment]
    route_geojson: Optional[Dict[str, Any]] = None

# Evaluation schemas
class DecodingComparisonItem(BaseModel):
    method: str
    model: str
    top1_accuracy: float
    route_continuity: float
    guaranteed_connectivity: bool

class EvaluationSummaryResponse(BaseModel):
    champion_model: str
    student_test_top1_acc: float
    student_test_log_loss: float
    student_test_macro_f1: float
    mean_route_continuity: float
    decoding_comparison: List[DecodingComparisonItem]
    per_trip_metrics: List[Dict[str, Any]]

# Job / Processing schemas
class ProcessJobRequest(BaseModel):
    trip_id: Optional[str] = "all"
    model_name: Optional[str] = "random_forest"
    beta: Optional[float] = 20.0
    alpha: Optional[float] = 0.2

class JobStatusResponse(BaseModel):
    job_id: str
    status: str  # 'pending', 'running', 'completed', 'failed'
    progress_pct: int
    current_stage: str
    stages_completed: List[str]
    message: str
    result: Optional[Dict[str, Any]] = None
