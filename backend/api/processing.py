from fastapi import APIRouter, HTTPException
from backend.schemas import ProcessJobRequest, JobStatusResponse
from backend.services.processing_service import JobManager

router = APIRouter(prefix="/api/process", tags=["processing"])
job_manager = JobManager()

@router.post("", response_model=JobStatusResponse)
def start_processing_job(req: ProcessJobRequest):
    """Start an asynchronous 10-stage map matching pipeline job."""
    job_id = job_manager.create_job(req.model_dump())
    job = job_manager.get_job(job_id)
    return job

@router.get("/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str):
    """Check status of a running or completed map-matching job."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
    return job
