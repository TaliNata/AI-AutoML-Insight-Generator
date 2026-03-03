"""
Feature preprocessing module for the AI AutoML project.

Handles categorical encoding and missing value imputation for numeric features.
Ensures all data is numeric and ready for model training.
"""

from typing import Optional
import pandas as pd


def preprocess_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess features by encoding categorical variables and imputing missing values.

    Performs the following transformations in order:
    1. One-hot encode categorical columns (dropping first category to avoid multicollinearity)
    2. Fill missing numeric values with column-wise mean
    3. Return a fully numeric DataFrame

    Args:
        X: Input feature DataFrame. Can contain both numeric and categorical columns.

    Returns:
        pd.DataFrame: Processed feature matrix with all numeric columns.

    Raises:
        TypeError: If X is not a pandas DataFrame.
        ValueError: If X is an empty DataFrame (0 rows).

    Examples:
        >>> df = pd.DataFrame({'age': [25, 30, None], 'city': ['NYC', 'LA', 'NYC']})
        >>> X_processed = preprocess_features(df)
        >>> print(X_processed.dtypes)
        age          float64
        city_LA      uint8
        city_NYC     uint8
        dtype: object
    """
    # =========================================================================
    # Input Validation
    # =========================================================================
    if not isinstance(X, pd.DataFrame):
        raise TypeError(
            f"Expected X to be a pandas DataFrame, got {type(X).__name__}. "
            f"Please provide a valid DataFrame."
        )

    if X.empty:
        raise ValueError(
            "Input DataFrame is empty (0 rows). " "Cannot preprocess an empty dataset."
        )

    # =========================================================================
    # Create a Copy to Avoid Modifying Original Data
    # =========================================================================
    X = X.copy()

    # =========================================================================
    # Categorical Encoding
    # =========================================================================
    X = pd.get_dummies(X, drop_first=True)

    # =========================================================================
    # Fill Missing Values with Column Mean
    # =========================================================================
    numeric_columns = X.select_dtypes(
        include=["int64", "int32", "float64", "float32"]
    ).columns

    for col in numeric_columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].mean())

    return X
