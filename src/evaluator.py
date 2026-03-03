"""
Model evaluation module for the AI AutoML project.

Computes regression metrics (MAE, R2) for model performance assessment.
Provides a simple, focused interface for evaluating regression models.
"""

from typing import Dict
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score


def evaluate_regression(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    """
    Evaluate regression model performance using MAE and R2 metrics.

    Computes standard regression evaluation metrics:
    - MAE (Mean Absolute Error): Average absolute difference between true and predicted values.
    - R2 (Coefficient of Determination): Proportion of variance explained by the model.

    Args:
        y_true: True target values (Series with shape (n_samples,)).
        y_pred: Predicted target values (Series with shape (n_samples,)).

    Returns:
        Dictionary containing:
        - "mae" (float): Mean Absolute Error metric
        - "r2" (float): R2 score (ranges from -inf to 1.0, where 1.0 is perfect)

    Raises:
        TypeError: If y_true is not a pandas Series.
        TypeError: If y_pred is not a pandas Series.
        ValueError: If y_true is empty (0 samples).
        ValueError: If y_pred is empty (0 samples).
        ValueError: If y_true and y_pred have mismatched lengths.

    Examples:
        >>> import pandas as pd
        >>> y_true = pd.Series([3, -0.5, 2, 7])
        >>> y_pred = pd.Series([2.5, 0.0, 2, 8])
        >>> metrics = evaluate_regression(y_true, y_pred)
        >>> print(metrics)
        {'mae': 0.375, 'r2': 0.960...}
    """
    # =========================================================================
    # Input Type Validation
    # =========================================================================
    if not isinstance(y_true, pd.Series):
        raise TypeError(
            f"Expected y_true to be a pandas Series, got {type(y_true).__name__}. "
            f"Please provide a valid Series."
        )

    if not isinstance(y_pred, pd.Series):
        raise TypeError(
            f"Expected y_pred to be a pandas Series, got {type(y_pred).__name__}. "
            f"Please provide a valid Series."
        )

    # =========================================================================
    # Data Validation
    # =========================================================================
    if y_true.empty:
        raise ValueError(
            "y_true is empty (0 samples). "
            "Cannot evaluate model on empty true values."
        )

    if y_pred.empty:
        raise ValueError(
            "y_pred is empty (0 samples). "
            "Cannot evaluate model on empty predictions."
        )

    # =========================================================================
    # Length Mismatch Validation
    # =========================================================================
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred have mismatched lengths: "
            f"y_true has {len(y_true)} samples, "
            f"y_pred has {len(y_pred)} samples. "
            f"Lengths must match."
        )

    # =========================================================================
    # Compute Metrics
    # =========================================================================
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    metrics: Dict[str, float] = {
        "mae": float(mae),
        "r2": float(r2),
    }

    return metrics
