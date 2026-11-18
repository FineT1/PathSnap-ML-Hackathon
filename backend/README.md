# PathSnap – Backend & Web Application Integration

PathSnap integrates GPS data preprocessing, machine learning classification, and Hidden Markov Model (HMM) Viterbi route reconstruction into a unified, commercial-grade geospatial analytics web application.

---

## 1. Architecture Overview

```
                      PathSnap Architecture
                      
   Frontend Client                      FastAPI Backend
   (React / SPA / Leaflet)              (Python 3.12)
 ┌───────────────────────────┐         ┌────────────────────────────────┐
 │ • Dashboard Analytics     │  REST   │ • /api/health                  │
 │ • Interactive Map (Leaflet│ ◄─────► │ • /api/dashboard/summary       │
 │ • Model Comparison UI     │  JSON   │ • /api/models/comparison       │
 │ • Pipeline Job Monitor    │         │ • /api/routes/{trip_id}        │
 │ • GPS Trajectory Upload   │         │ • /api/process (10-Stage Job)  │
 │ • Route Evaluation        │         │ • /api/upload (File Validation)│
 └───────────────────────────┘         └────────────────────────────────┘
                                                       │
                                   ┌───────────────────┴───────────────────┐
                                   ▼                                       ▼
                       Member 1 & 2 Engine                     Member 3 Engine
                       • preprocessing/                        • hmm/
                       • road_network.py                       • viterbi.py
                       • RandomForestClassifier                • route_reconstruction.py
```

---

## 2. Quick Start Guide

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run the Web Application
```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
Once started, open your browser:
* **Interactive Web Dashboard**: [http://localhost:8000](http://localhost:8000)
* **Interactive Swagger API Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
* **ReDoc API Documentation**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 3. REST API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Health check & champion model metadata |
| `GET` | `/api/dashboard/summary` | Real-time aggregate project KPIs and metrics |
| `GET` | `/api/models/comparison` | 5-model evaluation table (CV & student holdout metrics) |
| `GET` | `/api/models/champion` | Details for the active champion (`RandomForestClassifier`) |
| `GET` | `/api/trips` | List of all available trips and status |
| `GET` | `/api/routes/{trip_id}` | Decoded GPS points, matched road arcs, and inferred paths |
| `GET` | `/api/evaluation/summary` | Decoding comparison (Classifier-Only vs Classifier + HMM) |
| `POST`| `/api/upload` | Validated multipart file upload (`.track`, `.nodes`, `.arcs`) |
| `POST`| `/api/process` | Asynchronously trigger the 10-stage map-matching engine |
| `GET` | `/api/process/{job_id}` | Poll background job progress, stage status, and live logs |

---

## 4. Test Verification

Run all test suites across Members 1, 3, and 4:
```bash
# Member 1: Preprocessing & Candidate Generation (24 tests)
python tests/test_preprocessing.py

# Member 3: HMM & Viterbi Decoding (20 tests)
python tests/test_hmm.py

# Member 4: FastAPI REST API & Endpoints (10 tests)
python tests/test_backend.py
```
**Total: 54 / 54 tests passing.**
