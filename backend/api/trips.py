from fastapi import APIRouter
from typing import List
from backend.schemas import TripListItem
from backend.services.data_service import DataService

router = APIRouter(prefix="/api/trips", tags=["trips"])
data_service = DataService()

@router.get("", response_model=List[TripListItem])
def list_trips():
    """List available trips in the dataset."""
    return data_service.get_trips_list()
