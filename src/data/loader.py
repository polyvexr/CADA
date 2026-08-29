"""
Data loading and schema validation utilities for CADA motion telemetry.
"""

from pathlib import Path
from typing import Union, List, Optional
import pandas as pd

from src.config import RAW_SENSOR_COLS, TIMESTAMP_COL, TARGET_COL


class DataValidationError(ValueError):
    """Raised when motion data fails schema or integrity validation."""
    pass


def validate_motion_dataframe(
    df: pd.DataFrame,
    require_target: bool = False,
    sensor_cols: Optional[List[str]] = None,
    timestamp_col: Optional[str] = None,
    target_col: Optional[str] = None
) -> bool:
    """
    Validates that a DataFrame contains expected sensor columns, data types, and non-empty rows.

    Parameters
    ----------
    df : pd.DataFrame
        Input motion telemetry DataFrame.
    require_target : bool, default=False
        Whether the target 'Class' column is required.
    sensor_cols : list of str, optional
        List of required sensor column names.
    timestamp_col : str, optional
        Timestamp column name.
    target_col : str, optional
        Target classification column name.

    Returns
    -------
    bool
        True if validation passes.

    Raises
    ------
    DataValidationError
        If validation checks fail.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise DataValidationError("Input must be a valid non-empty pandas DataFrame.")

    if df.empty:
        raise DataValidationError("DataFrame contains 0 rows.")

    sensor_cols = sensor_cols or RAW_SENSOR_COLS
    timestamp_col = timestamp_col or TIMESTAMP_COL
    target_col = target_col or TARGET_COL

    # Check required sensor columns
    missing_sensors = [col for col in sensor_cols if col not in df.columns]
    if missing_sensors:
        raise DataValidationError(f"Missing required sensor columns: {missing_sensors}")

    # Check timestamp column if expected
    if timestamp_col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[timestamp_col]):
            raise DataValidationError(f"Timestamp column '{timestamp_col}' must be numeric.")

    # Check target column if required
    if require_target and target_col not in df.columns:
        raise DataValidationError(f"Required target column '{target_col}' not found in DataFrame.")

    # Check numeric types of sensor columns
    for col in sensor_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise DataValidationError(f"Sensor column '{col}' must contain numeric values.")

    return True


def load_motion_data(
    file_path: Union[str, Path],
    require_target: bool = False
) -> pd.DataFrame:
    """
    Loads motion telemetry data from a CSV file with automatic schema validation.

    Parameters
    ----------
    file_path : str or Path
        Path to the CSV file.
    require_target : bool, default=False
        Whether the target column 'Class' is required.

    Returns
    -------
    pd.DataFrame
        Loaded and validated DataFrame.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Motion data file not found at: {path}")

    df = pd.read_csv(path)
    validate_motion_dataframe(df, require_target=require_target)
    return df

