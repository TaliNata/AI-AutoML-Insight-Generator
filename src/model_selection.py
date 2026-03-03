"""
Model selection module for regression and classification tasks.

Automatically selects the best model based on cross-validation scores
using models and hyperparameters defined in the configuration.
"""

import logging
from typing import Dict, Tuple, Any
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.model_selection import cross_val_score
import numpy as np

from src import config

logger = logging.getLogger(__name__)


def select_best_model(
    X: Any, y: Any, task_type: str
) -> Tuple[str, Any, Dict[str, float]]:
    """
    Select the best model for the given task using cross-validation.

    Args:
        X: Feature matrix.
        y: Target variable.
        task_type: Either "regression" or "classification".

    Returns:
        Tuple of (best_model_name, best_model, cv_results_dict).
    """
    if task_type == "regression":
        return _select_regression_model(X, y)
    else:
        return _select_classification_model(X, y)


def _select_regression_model(X: Any, y: Any) -> Tuple[str, Any, Dict[str, float]]:
    """Select the best regression model based on cross-validation."""
    models: Dict[str, Any] = {}

    if config.ENABLE_LINEAR_REGRESSION:
        models["LinearRegression"] = LinearRegression(**config.LINEAR_REGRESSION_PARAMS)

    if config.ENABLE_RANDOM_FOREST:
        models["RandomForest"] = RandomForestRegressor(
            **config.RANDOM_FOREST_REGRESSION_PARAMS
        )

    if config.ENABLE_GRADIENT_BOOSTING:
        models["GradientBoosting"] = GradientBoostingRegressor(
            **config.GRADIENT_BOOSTING_REGRESSION_PARAMS
        )

    results: Dict[str, float] = {}
    cv_n_jobs = config.CV_N_JOBS if config.ENABLE_PARALLEL_CV else 1

    for name, model in models.items():
        logger.debug(f"Training {name}...")
        scores = cross_val_score(
            model,
            X,
            y,
            cv=config.CV_FOLDS,
            scoring=config.SCORING_REGRESSION,
            n_jobs=cv_n_jobs,
        )
        results[name] = float(np.mean(scores))
        logger.debug(f"{name}: {results[name]:.4f}")

    best_model_name = sorted(results.keys(), key=lambda x: results[x])[-1]
    best_model = models[best_model_name]

    return best_model_name, best_model, results


def _select_classification_model(X: Any, y: Any) -> Tuple[str, Any, Dict[str, float]]:
    """Select the best classification model based on cross-validation."""
    models: Dict[str, Any] = {}

    if config.ENABLE_LINEAR_CLASSIFICATION:
        models["LogisticRegression"] = LogisticRegression(
            **config.LOGISTIC_REGRESSION_PARAMS
        )

    if config.ENABLE_RANDOM_FOREST:
        models["RandomForest"] = RandomForestClassifier(
            **config.RANDOM_FOREST_CLASSIFICATION_PARAMS
        )

    if config.ENABLE_GRADIENT_BOOSTING:
        models["GradientBoosting"] = GradientBoostingClassifier(
            **config.GRADIENT_BOOSTING_CLASSIFICATION_PARAMS
        )

    results: Dict[str, float] = {}
    cv_n_jobs = config.CV_N_JOBS if config.ENABLE_PARALLEL_CV else 1

    for name, model in models.items():
        logger.debug(f"Training {name}...")
        scores = cross_val_score(
            model,
            X,
            y,
            cv=config.CV_FOLDS,
            scoring=config.SCORING_CLASSIFICATION,
            n_jobs=cv_n_jobs,
        )
        results[name] = float(np.mean(scores))
        logger.debug(f"{name}: {results[name]:.4f}")

    best_model_name = sorted(results.keys(), key=lambda x: results[x])[-1]
    best_model = models[best_model_name]

    return best_model_name, best_model, results
