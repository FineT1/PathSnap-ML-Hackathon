from fastapi import APIRouter
from backend.schemas import ModelComparisonResponse
from backend.services.data_service import DataService

router = APIRouter(prefix="/api/models", tags=["models"])
data_service = DataService()

@router.get("/comparison", response_model=ModelComparisonResponse)
def get_model_comparison():
    """Returns actual 5-model comparison metrics from results/model_comparison.csv."""
    return data_service.get_models_comparison()

@router.get("/champion")
def get_champion_model():
    """Returns the verified champion model details."""
    comp = data_service.get_models_comparison()
    champion_name = comp["champion_model"]
    for m in comp["models"]:
        if m["is_champion"]:
            return m
    return {"model": champion_name}
