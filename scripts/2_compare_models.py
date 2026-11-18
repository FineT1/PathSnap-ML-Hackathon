"""Stage 2: compare the five models.

1. Group CV by trip on the training trips (fold-level results).
2. Fit on all training trips, score the student-test trips once.

Writes to results/:
    cv_fold_results.csv, model_comparison.csv, student_test_predictions.csv
    figures/cv_macro_f1.png, figures/student_comparison.png, figures/log_loss.png

Run from the repo root:  python scripts/2_compare_models.py [--models logreg naive_bayes] [--folds 3]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pathsnap.models import (  # noqa: E402
    MODEL_FAMILY, check_columns, cross_validate, fit_and_score_holdout,
)
from pathsnap.viz import plot_cv_macro_f1, plot_log_loss, plot_student_comparison  # noqa: E402


def load_candidates(path: Path, feats: list[str]) -> pd.DataFrame:
    usecols = ["trip_id", "gps_point_id", "candidate_arc_id", "label"] + feats
    if path.suffix == ".parquet":
        df = pd.read_parquet(path, columns=usecols)
    else:
        dtype = {"trip_id": str, "gps_point_id": "int64", "candidate_arc_id": "int64"}
        dtype.update({f: "float32" for f in feats})
        df = pd.read_csv(path, usecols=usecols, dtype=dtype)
    df["trip_id"] = df["trip_id"].astype(str)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", help="subset of model names")
    parser.add_argument("--folds", type=int, help="override cv_folds")
    args = parser.parse_args()

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    mcfg = dict(cfg["modelling"], seed=cfg["seed"])
    if args.folds:
        mcfg["cv_folds"] = args.folds
    feats = cfg["features"]
    model_names = args.models or mcfg["models"]

    splits = pd.read_csv(ROOT / cfg["data"]["splits_file"], dtype={"trip_id": str})
    cand = load_candidates(ROOT / cfg["data"]["candidates_file"], feats)
    check_columns(cand.assign(label=cand["label"].fillna(0)), feats, "candidate table")

    # Guard: every trip must be listed in splits.csv. Evaluator trips must never appear here.
    unknown = set(cand["trip_id"]) - set(splits["trip_id"])
    if unknown:
        raise SystemExit(f"{len(unknown)} trip IDs are not in splits.csv, e.g. {sorted(unknown)[:5]}")

    train_ids = set(splits.loc[splits["partition"] == "train", "trip_id"])
    test_ids = set(splits.loc[splits["partition"] == "student_test", "trip_id"])
    train_df = cand[cand["trip_id"].isin(train_ids)].reset_index(drop=True)
    test_df = cand[cand["trip_id"].isin(test_ids)].reset_index(drop=True)
    check_columns(train_df, feats, "training trips")
    check_columns(test_df, feats, "student-test trips")

    print(f"Train: {train_df['trip_id'].nunique()} trips, {len(train_df):,} candidate rows, "
          f"{train_df.groupby(['trip_id','gps_point_id']).ngroups:,} GPS points")
    print(f"Student test: {test_df['trip_id'].nunique()} trips, {len(test_df):,} candidate rows")
    print(f"Features ({len(feats)}): {feats}")

    out = ROOT / "results"
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    cv_all, summary, preds_all = [], [], []

    for name in model_names:
        t0 = time.perf_counter()
        print(f"\n== {name} ==")
        folds = cross_validate(name, train_df, feats, mcfg)
        metrics, preds, fit_s, inf_us = fit_and_score_holdout(name, train_df, test_df, feats, mcfg)
        cv_all.append(folds.assign(model=name))
        preds_all.append(preds)
        summary.append({
            "model": name,
            "family": MODEL_FAMILY[name],
            "cv_macro_f1_mean": folds["macro_f1"].mean(),
            "cv_macro_f1_std": folds["macro_f1"].std(),
            "cv_top1_mean": folds["top1_acc"].mean(),
            "cv_top1_std": folds["top1_acc"].std(),
            "cv_log_loss_mean": folds["log_loss"].mean(),
            "cv_candidate_miss_rate": folds["candidate_miss_rate"].mean(),
            "cv_fit_time_s_mean": folds["fit_time_s"].mean(),
            "student_macro_f1": metrics["macro_f1"],
            "student_top1": metrics["top1_acc"],
            "student_log_loss": metrics["log_loss"],
            "student_candidate_miss_rate": metrics["candidate_miss_rate"],
            "student_fit_time_s": fit_s,
            "student_infer_us_per_row": inf_us,
            "route_continuity": "N/A here (computed in stage 5)",
        })
        print(f"CV macro F1 {folds['macro_f1'].mean():.4f} | student macro F1 {metrics['macro_f1']:.4f} | "
              f"student top-1 {metrics['top1_acc']:.4f} | {time.perf_counter() - t0:.0f}s")

    summary_df = pd.DataFrame(summary)
    cv_df = pd.concat(cv_all, ignore_index=True)
    cv_df.to_csv(out / "cv_fold_results.csv", index=False)
    summary_df.to_csv(out / "model_comparison.csv", index=False)
    pd.concat(preds_all, ignore_index=True).to_csv(out / "student_test_predictions.csv", index=False)

    plot_cv_macro_f1(cv_df, figs / "cv_macro_f1.png")
    plot_student_comparison(summary_df, figs / "student_comparison.png")
    plot_log_loss(summary_df, figs / "log_loss.png")

    print("\nSaved: results/model_comparison.csv, cv_fold_results.csv, student_test_predictions.csv, figures/*.png")


if __name__ == "__main__":
    main()
