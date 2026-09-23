from pathlib import Path

import pandas as pd
import pytest

from customer_segmentation.validation import load_customer_data, validate_customer_data

DATA_PATH = Path(__file__).parents[1] / "data" / "Mall_Customers.csv"


def test_repository_dataset_is_valid_and_complete():
    frame = load_customer_data(DATA_PATH)
    assert frame.shape == (200, 5)
    assert frame["CustomerID"].is_unique


def test_validation_rejects_missing_column():
    frame = load_customer_data(DATA_PATH).drop(columns=["Age"])
    with pytest.raises(ValueError, match="missing required columns"):
        validate_customer_data(frame)


def test_validation_rejects_invalid_spending_score():
    frame = load_customer_data(DATA_PATH)
    frame.loc[0, "Spending Score (1-100)"] = 101
    with pytest.raises(ValueError, match="between 1 and 100"):
        validate_customer_data(frame)

