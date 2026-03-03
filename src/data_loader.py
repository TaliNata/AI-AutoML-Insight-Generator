"""
Data loading module for the AI AutoML project.

Handles file validation, CSV loading, and target column validation.
Provides a production-ready interface for loading training datasets.
"""

import os
from typing import Tuple
import pandas as pd


def load_data(file_path: str, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load data from a CSV file with validation.

    Performs the following validations:
    - Checks that the file exists
    - Validates that the target column exists in the dataset
    - Returns feature matrix and target series

    Args:
        file_path: Path to the CSV file to load.
        target: Name of the target column in the dataset.

    Returns:
        Tuple of (X, y) where:
        - X (pd.DataFrame): Feature matrix with target column removed
        - y (pd.Series): Target column as a pandas Series

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the target column is not found in the dataset.

    Examples:
        >>> X, y = load_data('data/housing.csv', 'price')
        >>> print(X.shape, y.shape)
        ((506, 13), (506,))
    """
    # =========================================================================
    # File Existence Validation
    # =========================================================================
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Data file not found: '{file_path}'. "
            f"Please check the file path and try again."
        )

    # =========================================================================
    # Load CSV Data
    # =========================================================================
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f"Failed to load CSV file '{file_path}': {str(e)}") from e

    # =========================================================================
    # Target Column Validation
    # =========================================================================
    if target not in df.columns:
        available_columns = list(df.columns)
        raise ValueError(
            f"Target column '{target}' not found in dataset. "
            f"Available columns: {available_columns}"
        )

    # =========================================================================
    # Extract Features and Target
    # =========================================================================
    X = df.drop(columns=[target])
    y = df[target]

    return X, y
