from fastapi import APIRouter, HTTPException
from backend.schemas import TripRouteDetailResponse
from backend.services.data_service import DataService

router = APIRouter(prefix="/api/routes", tags=["routes"])
data_service = DataService()

@router.get("/{trip_id}", response_model=TripRouteDetailResponse)
def get_trip_route(trip_id: str):
    """
    Returns point matches and continuous road segment coordinates
    for Leaflet interactive map rendering.
    """
    try:
        route_detail = data_service.get_route_details(trip_id)
        return route_detail
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Route for trip {trip_id} not found: {exc}")
