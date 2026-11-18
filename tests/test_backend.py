"""
test_backend.py – FastAPI REST API Unit and Integration Tests for Member 4
===========================================================================

Tests:
  1. Health check endpoint (/api/health)
  2. Dashboard summary statistics (/api/dashboard/summary)
  3. Model comparison metrics verification (/api/models/comparison)
  4. Champion model retrieval (/api/models/champion)
  5. Trips list retrieval (/api/trips)
  6. Route details & coordinates endpoint (/api/routes/synthetic_01)
  7. Evaluation summary endpoint (/api/evaluation/summary)
  8. Map matching background job launch (/api/process)
  9. Job status polling (/api/process/{job_id})
  10. File upload endpoint validation (/api/upload)
  11. Frontend root serving (/)

Run with:
  python tests/test_backend.py
"""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_1_health():
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "RandomForestClassifier" in data["champion_model"]


def test_2_dashboard_summary():
    res = client.get("/api/dashboard/summary")
    assert res.status_code == 200
    data = res.json()
    assert data["total_trips"] >= 80
    assert data["champion_model"] == "random_forest"
    assert data["champion_top1_accuracy"] == 0.7535
    assert data["champion_log_loss"] == 0.6167
    assert data["mean_route_continuity"] == 1.0


def test_3_model_comparison():
    res = client.get("/api/models/comparison")
    assert res.status_code == 200
    data = res.json()
    assert data["champion_model"] == "random_forest"
    models = {m["model"]: m for m in data["models"]}
    assert "random_forest" in models
    assert "hist_gradient_boosting" in models
    assert "naive_bayes" in models
    assert "logreg" in models
    assert "decision_tree" in models
    # Verify verified figures
    rf = models["random_forest"]
    assert rf["student_top1"] == 0.7535
    assert rf["student_log_loss"] == 0.6167
    assert rf["is_champion"] == True


def test_4_champion_model():
    res = client.get("/api/models/champion")
    assert res.status_code == 200
    data = res.json()
    assert data["model"] == "random_forest"


def test_5_trips_list():
    res = client.get("/api/trips")
    assert res.status_code == 200
    trips = res.json()
    assert len(trips) >= 80
    assert any(t["trip_id"] == "00000000" for t in trips)


def test_6_route_details():
    res = client.get("/api/routes/synthetic_01")
    assert res.status_code == 200
    data = res.json()
    assert data["trip_id"] == "synthetic_01"
    assert len(data["points"]) == 4
    assert len(data["segments"]) >= 2
    assert data["continuity_score"] == 1.0
    assert data["is_fully_connected"] == True


def test_7_evaluation_summary():
    res = client.get("/api/evaluation/summary")
    assert res.status_code == 200
    data = res.json()
    assert data["champion_model"] == "random_forest"
    assert data["student_test_top1_acc"] == 0.7535
    assert data["mean_route_continuity"] == 1.0
    assert len(data["decoding_comparison"]) >= 2


def test_8_and_9_processing_job():
    # 8. Start job
    res = client.post("/api/process", json={"trip_id": "synthetic_01", "model_name": "random_forest"})
    assert res.status_code == 200
    job = res.json()
    job_id = job["job_id"]
    assert job_id.startswith("job_")

    # 9. Poll status
    time.sleep(1.0)
    res_poll = client.get(f"/api/process/{job_id}")
    assert res_poll.status_code == 200
    poll_data = res_poll.json()
    assert poll_data["job_id"] == job_id
    assert poll_data["status"] in ("pending", "running", "completed")


def test_10_upload_endpoint():
    dummy_track = b"-0.1278 51.5074 1000.0\n-0.1275 51.5075 1010.0\n"
    files = [("files", ("test.track", io.BytesIO(dummy_track), "text/plain"))]
    res = client.post("/api/upload", data={"trip_id": "test_trip_upload"}, files=files)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert "test.track" in data["saved_files"]


def test_11_serve_html_frontend():
    res = client.get("/")
    assert res.status_code == 200
    assert "PathSnap" in res.text
    assert "leaflet" in res.text.lower()


def run_all():
    tests = [
        ("1.  Health check (/api/health)",                    test_1_health),
        ("2.  Dashboard summary (/api/dashboard/summary)",     test_2_dashboard_summary),
        ("3.  Model comparison (/api/models/comparison)",      test_3_model_comparison),
        ("4.  Champion model (/api/models/champion)",          test_4_champion_model),
        ("5.  Trips list (/api/trips)",                        test_5_trips_list),
        ("6.  Route coordinates (/api/routes/{trip_id})",      test_6_route_details),
        ("7.  Evaluation summary (/api/evaluation/summary)",   test_7_evaluation_summary),
        ("8/9. Job management (/api/process)",                 test_8_and_9_processing_job),
        ("10. File upload validation (/api/upload)",          test_10_upload_endpoint),
        ("11. Web Frontend root serving (/)",                 test_11_serve_html_frontend),
    ]

    print("\n" + "=" * 65)
    print("PathSnap – Member 4 (FastAPI Backend & Frontend) Test Suite")
    print("=" * 65)

    passed, failed = [], []
    for name, fn in tests:
        try:
            fn()
            passed.append(name)
            print(f"  ✓ {name}")
        except Exception as exc:
            failed.append((name, exc))
            print(f"  ✗ {name} -> {type(exc).__name__}: {exc}")

    print("\n" + "=" * 65)
    print(f"Results: {len(passed)} passed, {len(failed)} failed")
    print("=" * 65 + "\n")

    return len(failed) == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
