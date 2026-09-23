from pathlib import Path

from customer_segmentation.io import load_model, save_bundle
from customer_segmentation.model import model_bundle, train_segmenter
from customer_segmentation.validation import load_customer_data

DATA_PATH = Path(__file__).parents[1] / "data" / "Mall_Customers.csv"


def test_saved_bundle_round_trip(tmp_path):
    model, metrics = train_segmenter(load_customer_data(DATA_PATH))
    target = tmp_path / "segmenter.joblib"
    save_bundle(model_bundle(model, metrics), target)
    restored = load_model(target)
    assert restored.predict(88, 17).segment == "growth-opportunity"

