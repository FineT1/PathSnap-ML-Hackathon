# PathSnap

Team: `<team name>` · UE24CS352A Machine Learning

PathSnap takes noisy GPS points from a vehicle and works out which road it actually drove on.

## Folders

- `data/`: the trips we were given. Not in git, except `splits.csv`.
- `pathsnap/`: the main code (loading data, features, models, tuning, HMM).
- `scripts/`: the steps to run, in order (0 to 5).
- `app/`: the API (Laptop 1) and the evaluator screen (Laptop 2).
- `artifacts/`: the saved final model and its settings.
- `results/`: scores, tables, plots and maps.
- `report/`: the final PDF report.
- `config.yaml`: all the settings in one place.

## Data

- 90 trips: 80 for training, 10 for student testing.
- Each trip has four files with the same name: `.track` (GPS points), `.nodes` (road points), `.arcs` (road segments) and `.route` (the correct route).
- The files have no header row.
- The 10 evaluator trips are not in this repo. They are kept in a separate folder outside it.

## Rules

- Each trip belongs to one group only. `data/splits.csv` says which group.
- Build and tune the models on the training trips only.
- Student-test trips are for comparing models and tuning the HMM settings.
- Evaluator trips are used only at the final evaluation. Their answers must never appear in the code or on the screen.
- Don't commit `data/` or the evaluator folder.

## Setup

1. Install the packages: `pip install -r requirements.txt`
2. Copy the dataset into `data/` (the steps are in the setup notes).

The scripts and the API are placeholders for now. They will be filled in as the team works, so there's nothing to run yet.

## Team

| Member | Part |
|---|---|
| | |