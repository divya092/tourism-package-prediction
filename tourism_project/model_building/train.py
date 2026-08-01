"""
Model training with experiment tracking.

Loads the train/test splits downloaded from the workflow artifact, tunes an XGBoost
classifier with GridSearchCV, logs every parameter set to MLflow, evaluates the best
model, and saves it into the deployment folder so the workflow can commit it to the
repository.
"""

import json
import os

import numpy as np
import pandas as pd
import joblib
import mlflow
import requests

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
)
import xgboost as xgb

TRACKING_URI = "http://localhost:5000"
FALLBACK_URI = "sqlite:///mlflow.db"     # used only if the server is not reachable
EXPERIMENT_NAME = "tourism-package-prediction"


def configure_mlflow():
    """Point MLflow at the tracking server started by the workflow.

    If that server is unreachable -- for example it has not finished starting --
    fall back to a local SQLite backend. Tracking still happens either way, and the
    training job never fails just because the UI was slow to come up.
    """
    try:
        requests.get(TRACKING_URI, timeout=5)
        mlflow.set_tracking_uri(TRACKING_URI)
        print(f"MLflow tracking : {TRACKING_URI}")
    except Exception:
        mlflow.set_tracking_uri(FALLBACK_URI)
        print(f"MLflow tracking : {FALLBACK_URI} (server unreachable, using local store)")

    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"MLflow experiment: {EXPERIMENT_NAME}")


configure_mlflow()

# The model must land here: this is the folder the workflow commits and the
# folder the Streamlit app loads from
DEPLOYMENT_DIR = "tourism_project/deployment"
MODEL_PATH = os.path.join(DEPLOYMENT_DIR, "best_tourism_model_v1.joblib")
METRICS_PATH = os.path.join(DEPLOYMENT_DIR, "metrics.json")

NUMERIC_FEATURES = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips",
    "Passport", "PitchSatisfactionScore", "OwnCar",
    "NumberOfChildrenVisiting", "MonthlyIncome",
]

CATEGORICAL_FEATURES = [
    "TypeofContact", "Occupation", "Gender",
    "ProductPitched", "MaritalStatus", "Designation",
]


def build_pipeline():
    """Preprocessing and estimator bundled into one object.

    Keeping them together means the transformations are fitted on training folds
    only (no leakage) and the deployed app can pass a raw dataframe straight in.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), NUMERIC_FEATURES),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), CATEGORICAL_FEATURES),
        ]
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", xgb.XGBClassifier(random_state=42, n_jobs=1, eval_metric="logloss")),
    ])


def main():
    print("=" * 70)
    print("MODEL TRAINING")
    print("=" * 70)

    # ---- Load the splits from the workflow artifact --------------------------------
    Xtrain = pd.read_csv("Xtrain.csv")
    Xtest = pd.read_csv("Xtest.csv")
    ytrain = pd.read_csv("ytrain.csv").squeeze()
    ytest = pd.read_csv("ytest.csv").squeeze()

    print(f"Xtrain: {Xtrain.shape}   Xtest: {Xtest.shape}")
    print(f"Train positive rate: {ytrain.mean():.4f}")
    print()

    # ---- Define the model and the search space -------------------------------------
    model_pipeline = build_pipeline()

    # Correction factor for the ~19% positive rate
    pos_weight = round((ytrain == 0).sum() / (ytrain == 1).sum(), 2)

    param_grid = {
        "model__n_estimators":     [100, 200],
        "model__max_depth":        [3, 5, 7],
        "model__learning_rate":    [0.05, 0.1],
        "model__subsample":        [0.8, 1.0],
        "model__colsample_bytree": [0.8, 1.0],
        "model__scale_pos_weight": [1, pos_weight],
    }

    n_combinations = int(np.prod([len(v) for v in param_grid.values()]))
    print(f"Tuning {n_combinations} parameter combinations with 5-fold CV")
    print()

    with mlflow.start_run(run_name="xgboost-gridsearch-production"):

        # ---- Tune ------------------------------------------------------------------
        # F1 is used instead of accuracy because the target is imbalanced
        grid_search = GridSearchCV(
            model_pipeline, param_grid, cv=5, n_jobs=-1, scoring="f1", verbose=1
        )
        grid_search.fit(Xtrain, ytrain)

        # ---- Log every tuned parameter set as a nested run --------------------------
        results = grid_search.cv_results_
        for i in range(len(results["params"])):
            with mlflow.start_run(nested=True):
                mlflow.log_params(results["params"][i])
                mlflow.log_metric("mean_cv_f1", results["mean_test_score"][i])
                mlflow.log_metric("std_cv_f1", results["std_test_score"][i])

        # ---- Log the winning configuration -----------------------------------------
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("best_cv_f1", grid_search.best_score_)

        best_model = grid_search.best_estimator_

        print()
        print("Best parameters:")
        for key, value in grid_search.best_params_.items():
            print(f"  {key:<32} {value}")
        print(f"Best cross-validated F1: {grid_search.best_score_:.4f}")
        print()

        # ---- Evaluate ---------------------------------------------------------------
        metrics = {}
        for split_name, X_split, y_split in [("train", Xtrain, ytrain),
                                             ("test", Xtest, ytest)]:
            pred = best_model.predict(X_split)
            proba = best_model.predict_proba(X_split)[:, 1]
            metrics.update({
                f"{split_name}_accuracy":  float(accuracy_score(y_split, pred)),
                f"{split_name}_precision": float(precision_score(y_split, pred)),
                f"{split_name}_recall":    float(recall_score(y_split, pred)),
                f"{split_name}_f1":        float(f1_score(y_split, pred)),
                f"{split_name}_roc_auc":   float(roc_auc_score(y_split, proba)),
            })
        mlflow.log_metrics(metrics)

        print("-" * 70)
        print("PERFORMANCE")
        print("-" * 70)
        for name, value in metrics.items():
            print(f"  {name:<20} {value:.4f}")
        print()

        print("Classification report (test set):")
        print(classification_report(ytest, best_model.predict(Xtest), digits=3,
                                    target_names=["No purchase", "Purchase"]))
        print("Confusion matrix (test set):")
        print(confusion_matrix(ytest, best_model.predict(Xtest)))
        print()

        # ---- Save so the pipeline can commit the model ------------------------------
        os.makedirs(DEPLOYMENT_DIR, exist_ok=True)
        joblib.dump(best_model, MODEL_PATH)
        print(f"Model saved to: {MODEL_PATH}")

        # A record of how the committed model actually performed
        with open(METRICS_PATH, "w") as f:
            json.dump({
                "best_params": {k: str(v) for k, v in grid_search.best_params_.items()},
                "best_cv_f1": float(grid_search.best_score_),
                "metrics": metrics,
            }, f, indent=2)
        print(f"Metrics saved to: {METRICS_PATH}")

        # Also register the model as an MLflow artifact
        mlflow.log_artifact(MODEL_PATH, artifact_path="model")
        mlflow.log_artifact(METRICS_PATH, artifact_path="model")

    print("=" * 70)
    print("MODEL TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
