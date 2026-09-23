"""Dataset loading and explicit quality checks."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

INCOME_COLUMN = "Annual Income (k$)"
SPENDING_COLUMN = "Spending Score (1-100)"
FEATURES = (INCOME_COLUMN, SPENDING_COLUMN)
REQUIRED_COLUMNS = ("CustomerID", "Gender", "Age", *FEATURES)


def validate_customer_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate schema and business ranges, returning a defensive copy."""

    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"missing required columns: {', '.join(missing_columns)}")
    if frame.empty:
        raise ValueError("customer dataset must not be empty")
    if frame[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("required columns must not contain missing values")
    if frame["CustomerID"].duplicated().any():
        raise ValueError("CustomerID values must be unique")
    if not frame["Age"].between(0, 120).all():
        raise ValueError("Age must be between 0 and 120")
    if not frame[INCOME_COLUMN].ge(0).all():
        raise ValueError("annual income must be non-negative")
    if not frame[SPENDING_COLUMN].between(1, 100).all():
        raise ValueError("spending score must be between 1 and 100")
    return frame.copy()


def load_customer_data(path: str | Path) -> pd.DataFrame:
    return validate_customer_data(pd.read_csv(path))

