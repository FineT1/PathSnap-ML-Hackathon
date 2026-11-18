# PathSnap: Map-Matching & Route Reconstruction

Team: `<team name>` · UE24CS352A Machine Learning

## Layout

```
pathsnap/
├── README.md
├── requirements.txt
├── config.yaml              # all settings in one place
├── data/
│   ├── train/<trip_id>/     # 80 trips, as provided (git-ignored)
│   ├── student_test/<trip_id>/  # 10 trips, as provided (git-ignored)
│   ├── processed/           # generated candidates and features (git-ignored)
│   └── splits.csv           # trip_id -> partition (committed)
├── pathsnap/                # library: all reusable logic
│   ├── data.py              # parse .track/.nodes/.arcs/.route, validate, project
│   ├── features.py          # candidate generation, features, labels
│   ├── models.py            # model registry, group CV, metrics
│   ├── tune.py              # Bayesian optimisation (Optuna)
│   ├── hmm.py               # HMM states, emissions, transitions, Viterbi
│   └── viz.py               # plots and folium maps
├── scripts/                 # run in order
│   ├── 0_make_splits.py
│   ├── 1_eda.py
│   ├── 2_compare_models.py  # >= 5 models
│   ├── 3_tune.py
│   ├── 4_export_final.py    # refit, write artifacts/
│   └── 5_decode_hmm.py      # classifier-only vs classifier + HMM
├── app/                     # deployment
│   ├── api.py               # FastAPI backend (Laptop 1)
│   ├── evaluator.py         # ONLY module that reads ground truth
│   └── ui.html              # evaluator interface (Laptop 2)
├── artifacts/               # final model, preprocessing, HMM config (committed; LFS if large)
├── results/                 # metrics, tuning trials, plots, folium maps (committed)
└── report/                  # final PDF + source
```

## Data

- Provided: 90 trips = 80 training + 10 student-testing. Each trip folder holds `<id>.track`, `.nodes`, `.arcs`, `.route`.
- Road networks are per trip, not shared. Keep them with their trip.
- Files are headerless. `.track`: lon, lat, time (s). `.nodes`: lon, lat (row = node id). `.arcs`: from, to (row = arc id). `.route`: one arc id per line.
- Evaluator trips (10) are not in this folder. They go in a folder outside the repo, set by `private_dir` in `config.yaml`.

## Rules

- Split by trip only. `data/splits.csv` is the single source.
- Fit preprocessing and tuning on `train` only. `student_test` is for comparison. Evaluator trips are never used before the final evaluation.
- Evaluator ground truth stays in `private_dir`. Never import it in `pathsnap/`, `scripts/`, or the UI.
- Commit code, configs, results and the final model. Do not commit `data/` or evaluator files.

## Run

```bash
pip install -r requirements.txt
python scripts/0_make_splits.py
python scripts/1_eda.py
# 2 to 5 in order
```

Laptop 1: `uvicorn app.api:app --host 0.0.0.0 --port 8000`
Laptop 2: open `app/ui.html` and point it at Laptop 1's address.

## Team

| Member | Owns | Can explain |
|---|---|---|
| | | |
