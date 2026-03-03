"""
Centralized configuration for the AI AutoML project.

This module contains all hardcoded parameters and hyperparameters used throughout
the project, including model selection flags, preprocessing settings, and LLM settings.
"""

# ============================================================================
# Data Preprocessing & Train-Test Split
# ============================================================================

TEST_SIZE: float = 0.2
"""Fraction of dataset to use for testing."""

RANDOM_STATE: int = 42
"""Random seed for reproducibility in train-test split and model initialization."""


# ============================================================================
# Cross-Validation & Scoring
# ============================================================================

CV_FOLDS: int = 5
"""Number of folds for cross-validation."""

SCORING_REGRESSION: str = "r2"
"""Scoring metric for regression tasks."""

SCORING_CLASSIFICATION: str = "f1_weighted"
"""Scoring metric for classification tasks."""


# ============================================================================
# Model Selection & Hyperparameters
# ============================================================================

# Enable/Disable specific models
ENABLE_LINEAR_REGRESSION: bool = True
"""Enable LinearRegression model for regression tasks."""

ENABLE_LINEAR_CLASSIFICATION: bool = True
"""Enable LogisticRegression model for classification tasks."""

ENABLE_RANDOM_FOREST: bool = True
"""Enable RandomForestRegressor/RandomForestClassifier models."""

ENABLE_GRADIENT_BOOSTING: bool = True
"""Enable GradientBoostingRegressor/GradientBoostingClassifier models."""

# Regression hyperparameters
LINEAR_REGRESSION_PARAMS: dict = {}
"""Hyperparameters for LinearRegression model."""

RANDOM_FOREST_REGRESSION_PARAMS: dict = {
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "n_estimators": 50,
}
"""Hyperparameters for RandomForestRegressor model."""

GRADIENT_BOOSTING_REGRESSION_PARAMS: dict = {
    "random_state": RANDOM_STATE,
    "n_estimators": 50,
}
"""Hyperparameters for GradientBoostingRegressor model."""

# Classification hyperparameters
LOGISTIC_REGRESSION_PARAMS: dict = {
    "max_iter": 1000,
    "random_state": RANDOM_STATE,
}
"""Hyperparameters for LogisticRegression model."""

RANDOM_FOREST_CLASSIFICATION_PARAMS: dict = {
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}
"""Hyperparameters for RandomForestClassifier model."""

GRADIENT_BOOSTING_CLASSIFICATION_PARAMS: dict = {
    "random_state": RANDOM_STATE,
}
"""Hyperparameters for GradientBoostingClassifier model."""

ENABLE_PARALLEL_CV: bool = True
"""Enable parallel processing in cross-validation."""

CV_N_JOBS: int = -1
"""Number of jobs for parallel processing in cross-validation (-1 uses all cores)."""


# ============================================================================
# LLM Configuration
# ============================================================================

LLM_MODEL: str = "openai/gpt-4o-mini"
"""LLM model to use for report generation."""

LLM_TEMPERATURE: float = 0.3
"""Temperature parameter for LLM (controls randomness/creativity)."""

LLM_TIMEOUT: int = 30
"""Timeout in seconds for LLM API calls."""


# ============================================================================
# File Paths & Output
# ============================================================================

MODELS_DIR: str = "models"
"""Directory to save trained models."""

REPORTS_DIR: str = "reports"
"""Directory to save generated reports and metrics."""

METRICS_FILE: str = "metrics.json"
"""Filename for metrics output."""

FEATURE_IMPORTANCE_FILE: str = "feature_importance.csv"
"""Filename for feature importance output."""

REPORT_FILE: str = "report.txt"
"""Filename for LLM-generated report."""


# ============================================================================
# Task Type Detection
# ============================================================================

CLASSIFICATION_THRESHOLD: int = 20
"""Number of unique values threshold for classification/regression detection."""
