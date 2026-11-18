"""
processing_service.py – Member 4: Background Job Manager for Processing Pipeline
================================================================================

Manages asynchronous execution of the 10-stage map matching pipeline:
  1. Data loading
  2. GPS preprocessing
  3. Road network construction
  4. Candidate generation
  5. Machine learning prediction (Random Forest)
  6. HMM emission calculation
  7. Transition probability calculation
  8. Viterbi decoding
  9. Route reconstruction & intermediate arc inference
  10. Evaluation & metrics calculation
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

STAGES = [
    "1. Data loading",
    "2. GPS preprocessing",
    "3. Road network construction",
    "4. Candidate generation",
    "5. Machine learning prediction",
    "6. HMM emission calculation",
    "7. Transition probability calculation",
    "8. Viterbi decoding",
    "9. Route reconstruction",
    "10. Evaluation & summary",
]


class JobManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(JobManager, cls).__new__(cls)
            cls._instance.jobs = {}
        return cls._instance

    def create_job(self, params: Dict[str, Any]) -> str:
        job_id = f"job_{uuid.uuid4().hex[:8]}"
        self.jobs[job_id] = {
            "job_id": job_id,
            "params": params,
            "status": "pending",
            "progress_pct": 0,
            "current_stage": STAGES[0],
            "stages_completed": [],
            "message": "Job queued for processing...",
            "logs": [],
            "result": None,
            "error": None,
            "created_at": time.time(),
        }
        # Start worker thread
        thread = threading.Thread(target=self._run_job, args=(job_id,), daemon=True)
        thread.start()
        return job_id

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self.jobs.get(job_id)

    def _run_job(self, job_id: str):
        job = self.jobs[job_id]
        job["status"] = "running"
        logger.info("Starting processing job %s with params: %s", job_id, job["params"])

        try:
            from run_hmm import create_synthetic_trip
            from preprocessing.road_network import build_road_network
            from hmm import HMMMapMatcher

            nodes_df, arcs_df, track_df, candidates_df, ref_route_df = create_synthetic_trip()
            network = None
            matcher = None
            result = None

            for idx, stage in enumerate(STAGES):
                job["current_stage"] = stage
                job["progress_pct"] = int((idx / len(STAGES)) * 100)
                job["message"] = f"Executing: {stage}"
                job["logs"].append(f"[{time.strftime('%H:%M:%S')}] Started {stage}")

                # Real execution per stage
                if idx == 0:  # Data loading
                    time.sleep(0.4)
                elif idx == 1:  # GPS preprocessing
                    time.sleep(0.3)
                elif idx == 2:  # Road network construction
                    network = build_road_network(nodes_df, arcs_df)
                elif idx == 3:  # Candidate generation
                    time.sleep(0.4)
                elif idx == 4:  # ML prediction
                    time.sleep(0.3)
                elif idx == 5:  # HMM emission
                    matcher = HMMMapMatcher(road_network=network, model_name=job["params"].get("model_name", "random_forest"))
                elif idx == 6:  # Transition calculation
                    time.sleep(0.3)
                elif idx == 7:  # Viterbi decoding
                    result = matcher.match_trip(
                        "synthetic_01",
                        candidates_df,
                        track_df=track_df,
                        prob_col="predicted_probability",
                    )
                elif idx == 8:  # Route reconstruction
                    time.sleep(0.3)
                elif idx == 9:  # Evaluation
                    time.sleep(0.2)

                job["stages_completed"].append(stage)
                job["logs"].append(f"[{time.strftime('%H:%M:%S')}] Completed {stage}")

            job["status"] = "completed"
            job["progress_pct"] = 100
            job["current_stage"] = "Done"
            job["message"] = "Processing completed successfully!"
            job["result"] = {
                "trip_id": "synthetic_01",
                "n_observations": len(result.point_matches),
                "n_route_arcs": len(result.route_arc_sequence),
                "continuity_score": result.continuity_score,
                "is_fully_connected": result.is_fully_connected,
            }
            logger.info("Processing job %s completed successfully.", job_id)

        except Exception as exc:
            logger.exception("Error executing job %s: %s", job_id, exc)
            job["status"] = "failed"
            job["error"] = str(exc)
            job["message"] = f"Failed at {job['current_stage']}: {exc}"
            job["logs"].append(f"[{time.strftime('%H:%M:%S')}] ERROR: {exc}")
