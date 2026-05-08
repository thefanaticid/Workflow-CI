"""Training GradientBoostingRegressor untuk MLflow Project (Kriteria 3).

Dijalankan oleh `mlflow run MLProject/` di CI. Logs ke DagsHub via env var atau
fallback ke ./mlruns. Run ID di-print ke stdout supaya bisa di-capture CI untuk
build Docker image.

Penulis  : Muhammad Fadhil Rizki
Username : fadhilspooky
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, KFold

LOGGER = logging.getLogger("mlproject.modelling")
RANDOM_STATE = 42
TARGET = "MedHouseValue"


def load_dataset(data_dir: Path):
    train = pd.read_csv(data_dir / "housing_train.csv")
    test = pd.read_csv(data_dir / "housing_test.csv")
    X_train, y_train = train.drop(columns=[TARGET]), train[TARGET]
    X_test, y_test = test.drop(columns=[TARGET]), test[TARGET]
    LOGGER.info("Dataset: Train=%s | Test=%s", X_train.shape, X_test.shape)
    return X_train, X_test, y_train, y_test


def configure_tracking(experiment_name: str) -> None:
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if uri:
        mlflow.set_tracking_uri(uri)
        LOGGER.info("Tracking URI: %s", uri)
    else:
        mlflow.set_tracking_uri("file:./mlruns")
        LOGGER.info("MLFLOW_TRACKING_URI tidak diset, pakai lokal: file:./mlruns")

    mlflow.set_experiment(experiment_name)


def compute_metrics(y_true, y_pred, prefix="") -> dict:
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        f"{prefix}rmse": float(np.sqrt(mse)),
        f"{prefix}mse": mse,
        f"{prefix}mae": float(mean_absolute_error(y_true, y_pred)),
        f"{prefix}r2": float(r2_score(y_true, y_pred)),
        f"{prefix}median_ae": float(median_absolute_error(y_true, y_pred)),
        f"{prefix}explained_variance": float(explained_variance_score(y_true, y_pred)),
        f"{prefix}mape": float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-6)))),
    }


def create_artifacts(model, feature_names, X_test, y_test, y_pred, cv_results, tmp: Path):
    # Feature importance
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values()
    fig, ax = plt.subplots(figsize=(8, 6))
    importances.plot(kind="barh", color="steelblue", ax=ax)
    ax.set_title("Feature Importance — GBR")
    fig.tight_layout(); fig.savefig(tmp / "feature_importance.png", dpi=120); plt.close(fig)

    # Residuals
    residuals = y_test - y_pred
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_pred, residuals, alpha=0.3, s=10, color="darkorange")
    ax.axhline(0, color="black", linestyle="--")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Residual"); ax.set_title("Residual Plot")
    fig.tight_layout(); fig.savefig(tmp / "residuals_plot.png", dpi=120); plt.close(fig)

    # Pred vs Actual
    lo, hi = float(min(y_test.min(), y_pred.min())), float(max(y_test.max(), y_pred.max()))
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_test, y_pred, alpha=0.3, s=10, color="seagreen")
    ax.plot([lo, hi], [lo, hi], "r--")
    ax.set_xlabel("Actual"); ax.set_ylabel("Predicted"); ax.set_title("Predicted vs Actual")
    fig.tight_layout(); fig.savefig(tmp / "predictions_vs_actual.png", dpi=120); plt.close(fig)

    # CV results
    pd.DataFrame(cv_results).to_csv(tmp / "cv_results.csv", index=False)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("housing_preprocessing"))
    p.add_argument("--experiment-name", type=str, default="california-housing-ci")
    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument("--learning-rate", type=float, default=0.1)
    p.add_argument("--max-depth", type=int, default=5)
    p.add_argument("--cv-folds", type=int, default=3)
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    configure_tracking(args.experiment_name)

    X_train, X_test, y_train, y_test = load_dataset(args.data_dir)

    param_grid = {
        "n_estimators": [args.n_estimators],
        "learning_rate": [args.learning_rate],
        "max_depth": [args.max_depth],
    }
    cv = KFold(n_splits=args.cv_folds, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(
        GradientBoostingRegressor(random_state=RANDOM_STATE),
        param_grid, scoring="neg_root_mean_squared_error",
        cv=cv, n_jobs=-1, return_train_score=True,
    )

    with mlflow.start_run(run_name="gbr_ci_run") as run:
        run_id = run.info.run_id
        LOGGER.info("Run ID: %s", run_id)

        mlflow.log_param("model", "GradientBoostingRegressor")
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("learning_rate", args.learning_rate)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("cv_folds", args.cv_folds)
        mlflow.log_dict(param_grid, "param_grid.json")

        grid.fit(X_train, y_train)
        best = grid.best_estimator_
        mlflow.log_metric("cv_rmse", float(-grid.best_score_))

        train_metrics = compute_metrics(y_train.values, best.predict(X_train), "train_")
        test_pred = best.predict(X_test)
        test_metrics = compute_metrics(y_test.values, test_pred, "test_")
        for k, v in {**train_metrics, **test_metrics}.items():
            mlflow.log_metric(k, v)

        signature = infer_signature(X_train, best.predict(X_train))
        mlflow.sklearn.log_model(best, "model", signature=signature, input_example=X_train.head(3))

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            create_artifacts(best, X_train.columns, X_test, y_test.values, test_pred,
                             grid.cv_results_, tmp_path)
            for f in tmp_path.glob("*"):
                mlflow.log_artifact(str(f), artifact_path="extras")

        LOGGER.info("Test: %s", json.dumps({k: round(v, 4) for k, v in test_metrics.items()}))

    # Print run_id ke stdout untuk di-capture CI
    print(f"MLFLOW_RUN_ID={run_id}")


if __name__ == "__main__":
    main()
