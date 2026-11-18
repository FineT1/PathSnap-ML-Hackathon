from fastapi import APIRouter
from backend.schemas import DashboardSummaryResponse
from backend.services.data_service import DataService

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])
data_service = DataService()

@router.get("/summary", response_model=DashboardSummaryResponse)
def get_dashboard_summary():
    """Returns actual aggregated overview statistics for the PathSnap dashboard."""
    return data_service.get_dashboard_summary()
