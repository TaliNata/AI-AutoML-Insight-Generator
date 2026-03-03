"""
Main pipeline for automated machine learning workflow.

Handles data loading, preprocessing, automatic task detection, model selection,
evaluation, and report generation.
"""

import joblib
import json
import logging
import os
from typing import Dict, Any
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from src.llm_report import generate_llm_report
from src.model_selection import select_best_model
from src import config

logger = logging.getLogger(__name__)


def run_pipeline(file_path: str, target: str) -> None:
    """
    Execute the complete ML pipeline: data loading, preprocessing, model selection,
    evaluation, and report generation.

    Args:
        file_path: Path to the CSV file containing the dataset.
        target: Name of the target column.
    """
    logger.info("Loading data...")
    df = pd.read_csv(file_path)

    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Target column: {target}")

    # -----------------------------
    # Task type detection
    # -----------------------------
    target_series = df[target]

    if target_series.dtype in ["int64", "float64"]:
        unique_values = target_series.nunique()
        if unique_values > config.CLASSIFICATION_THRESHOLD:
            task_type = "regression"
        else:
            task_type = "classification"
    else:
        task_type = "classification"

    logger.info(f"Detected task type: {task_type}")

    # -----------------------------
    # Feature / target split
    # -----------------------------
    X = df.drop(columns=[target])
    y = df[target]

    logger.info(f"Features before encoding: {X.shape}")

    # Encoding categorical features
    X = pd.get_dummies(X, drop_first=True)

    logger.info(f"Features after encoding: {X.shape}")

    # Fill missing values
    X = X.fillna(X.mean())

    # -----------------------------
    # Train / Test split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )

    logger.info(f"Train shape: {X_train.shape}")
    logger.info(f"Test shape: {X_test.shape}")

    # -----------------------------
    # Auto Model Selection (CV)
    # -----------------------------
    best_model_name, model, cv_results = select_best_model(X_train, y_train, task_type)

    logger.info("Cross-validation results:")
    for name, score in cv_results.items():
        logger.info(f"{name}: {score:.4f}")

    logger.info(f"Selected best model: {best_model_name}")

    # Train best model
    model.fit(X_train, y_train)
    logger.info("Best model trained successfully")

    # -----------------------------
    # Evaluation
    # -----------------------------
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    logger.debug(f"MAE: {mae}")
    logger.debug(f"R2: {r2}")

    # -----------------------------
    # Save metrics
    # -----------------------------
    os.makedirs(config.REPORTS_DIR, exist_ok=True)

    metrics: Dict[str, Any] = {
        "task_type": task_type,
        "best_model": best_model_name,
        "mae": float(mae),
        "r2": float(r2),
        "cv_results": {k: float(v) for k, v in cv_results.items()},
    }

    with open(os.path.join(config.REPORTS_DIR, config.METRICS_FILE), "w") as f:
        json.dump(metrics, f, indent=4)

    logger.info(f"Metrics saved to {config.REPORTS_DIR}/{config.METRICS_FILE}")

    # -----------------------------
    # Save model
    # -----------------------------
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(config.MODELS_DIR, "model.pkl"))

    logger.info(f"Model saved to {config.MODELS_DIR}/model.pkl")

    # -----------------------------
    # Feature Importance
    # -----------------------------
    if hasattr(model, "coef_"):
        importance_values = model.coef_
    elif hasattr(model, "feature_importances_"):
        importance_values = model.feature_importances_
    else:
        importance_values = None

    if importance_values is not None:
        feature_importance = pd.DataFrame(
            {"feature": X.columns, "importance": importance_values}
        )

        feature_importance["abs_importance"] = feature_importance["importance"].abs()
        feature_importance = feature_importance.sort_values(
            by="abs_importance", ascending=False
        )

        logger.debug("Top 5 most important features:")
        logger.debug(feature_importance.head(5).to_string())

        feature_importance.to_csv(
            os.path.join(config.REPORTS_DIR, config.FEATURE_IMPORTANCE_FILE),
            index=False,
        )

        logger.info(
            f"Feature importance saved to {config.REPORTS_DIR}/{config.FEATURE_IMPORTANCE_FILE}"
        )

        top_5 = feature_importance.head(5).to_string(index=False)

    else:
        top_5 = "Feature importance not available for this model."

    # -----------------------------
    # LLM Insight Report
    # -----------------------------
    report = generate_llm_report(metrics, top_5)

    with open(
        os.path.join(config.REPORTS_DIR, config.REPORT_FILE),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(report)

    logger.info(f"LLM report saved to {config.REPORTS_DIR}/{config.REPORT_FILE}")
