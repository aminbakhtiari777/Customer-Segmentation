from pathlib import Path

import pytest

from customer_segmentation.model import segment_summary, train_segmenter
from customer_segmentation.validation import load_customer_data

DATA_PATH = Path(__file__).parents[1] / "data" / "Mall_Customers.csv"


@pytest.fixture(scope="module")
def trained():
    frame = load_customer_data(DATA_PATH)
    model, metrics = train_segmenter(frame)
    return frame, model, metrics


def test_training_produces_five_stable_business_segments(trained):
    _, model, metrics = trained
    assert set(model.segment_names.values()) == {
        "mainstream",
        "high-value",
        "high-engagement",
        "growth-opportunity",
        "budget-conscious",
    }
    assert metrics.samples == 200
    assert metrics.clusters == 5
    assert metrics.silhouette > 0.5
    assert metrics.davies_bouldin < 1.0


def test_prediction_matches_known_growth_opportunity_profile(trained):
    _, model, _ = trained
    prediction = model.predict(88, 17)
    assert prediction.segment == "growth-opportunity"


def test_segment_summary_accounts_for_every_customer(trained):
    frame, model, _ = trained
    summary = segment_summary(model, frame)
    assert summary["customers"].sum() == 200
    assert summary["customer_share"].sum() == pytest.approx(1.0)


def test_prediction_rejects_out_of_range_score(trained):
    _, model, _ = trained
    with pytest.raises(ValueError, match="between 1 and 100"):
        model.predict(50, 0)

