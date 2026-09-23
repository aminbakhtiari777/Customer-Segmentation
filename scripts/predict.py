"""Predict one customer segment from the saved bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from customer_segmentation.io import load_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("artifacts/customer_segmenter.joblib"))
    parser.add_argument("--income", type=float, required=True, help="Annual income in thousands")
    parser.add_argument("--spending", type=float, required=True, help="Spending score from 1 to 100")
    args = parser.parse_args()
    prediction = load_model(args.model).predict(args.income, args.spending)
    print(json.dumps(prediction.as_dict(), indent=2))


if __name__ == "__main__":
    main()

