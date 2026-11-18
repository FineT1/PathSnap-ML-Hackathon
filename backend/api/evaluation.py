from fastapi import APIRouter
from backend.schemas import EvaluationSummaryResponse
from backend.services.data_service import DataService

router = APIRouter(prefix="/api/evaluation", tags=["evaluation"])
data_service = DataService()

@router.get("/summary", response_model=EvaluationSummaryResponse)
def get_evaluation_summary():
    """
    Returns actual evaluation summary, including decoding comparison
    (Classifier Only vs Classifier + HMM Viterbi).
    """
    return data_service.get_evaluation_summary()
