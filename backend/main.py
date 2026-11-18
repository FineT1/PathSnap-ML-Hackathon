"""
backend/main.py – PathSnap Web Application Backend Entrypoint
=============================================================

FastAPI REST server integrating:
  - Preprocessing & Road Network (Member 1)
  - Machine Learning & Model Evaluation (Member 2)
  - HMM & Viterbi Route Reconstruction (Member 3)
  - Interactive Web Dashboard & Geospatial Map (Member 4)
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from backend.api.dashboard import router as dashboard_router
from backend.api.evaluation import router as evaluation_router
from backend.api.models import router as models_router
from backend.api.processing import router as processing_router
from backend.api.routes import router as routes_router
from backend.api.trips import router as trips_router
from backend.api.upload import router as upload_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pathsnap.backend")

app = FastAPI(
    title="PathSnap API",
    description="Enterprise GPS Map Matching, Machine Learning & HMM Route Reconstruction Platform",
    version="1.0.0",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows frontend on localhost:5173, localhost:3000, etc.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Routers
app.include_router(dashboard_router)
app.include_router(models_router)
app.include_router(trips_router)
app.include_router(routes_router)
app.include_router(evaluation_router)
app.include_router(upload_router)
app.include_router(processing_router)

STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/api/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "PathSnap Map-Matching API",
        "champion_model": "RandomForestClassifier",
        "version": "1.0.0",
    }


@app.get("/", response_class=HTMLResponse)
def serve_index():
    """Serve the PathSnap single-page application frontend."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>PathSnap API Running</h1><p>Visit /docs for Swagger API documentation.</p>")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
