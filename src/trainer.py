"""
Model training module for the AI AutoML project.

Handles model training with comprehensive input validation.
Provides a simple, focused interface for fitting models to training data.
"""

from typing import Any
import pandas as pd


def train_model(model: Any, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
    """
    Train a machine learning model on the provided training data.

    Validates inputs and fit the provided model to the training data.
    Returns the trained model ready for prediction. 

    Args:
        model: Machine learning model with a fit() method (e.g., sklearn estimator).
        X_train: Training feature matrix (DataFrame with shape (n_samples, n_features)).
        y_train: Training target variable (Series with shape (n_samples,)).

    Returns:
        The fitted model object (same type as input model but with learned parameters).

    Raises:
        TypeError: If model does not have a fit() method.
        TypeError: If X_train is not a pandas DataFrame.
        TypeError: If y_train is not a pandas Series.
        ValueError: If X_train is empty (0 rows).
        ValueError: If y_train is empty (0 samples).
        ValueError: If X_train and y_train have mismatched lengths.

    Examples:
        >>> from sklearn.linear_model import LinearRegression
        >>> import pandas as pd
        >>> X = pd.DataFrame({'feature': [1, 2, 3, 4]})
        >>> y = pd.Series([2, 4, 6, 8])
        >>> model = LinearRegression()
        >>> trained_model = train_model(model, X, y)
        >>> trained_model.coef_
        array([2.])
    """
    # =========================================================================
    # Model Validation
    # =========================================================================
    if not hasattr(model, "fit"):
        raise TypeError(
            f"Model must have a fit() method. "
            f"Got object of type {type(model).__name__} which does not have fit()."
        )

    # =========================================================================
    # Input Type Validation
    # =========================================================================
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError(
            f"Expected X_train to be a pandas DataFrame, got {type(X_train).__name__}. "
            f"Please provide a valid DataFrame."
        )

    if not isinstance(y_train, pd.Series):
        raise TypeError(
            f"Expected y_train to be a pandas Series, got {type(y_train).__name__}. "
            f"Please provide a valid Series."
        )

    # =========================================================================
    # Data Validation
    # =========================================================================
    if X_train.empty:
        raise ValueError(
            "X_train is empty (0 rows). " "Cannot train model on empty feature matrix."
        )

    if y_train.empty:
        raise ValueError(
            "y_train is empty (0 samples). "
            "Cannot train model on empty target variable."
        )

    # =========================================================================
    # Length Mismatch Validation
    # =========================================================================
    if len(X_train) != len(y_train):
        raise ValueError(
            f"X_train and y_train have mismatched lengths: "
            f"X_train has {len(X_train)} samples, "
            f"y_train has {len(y_train)} samples. "
            f"Lengths must match."
        )

    # =========================================================================
    # Train Model
    # =========================================================================
    model.fit(X_train, y_train)

    return model
