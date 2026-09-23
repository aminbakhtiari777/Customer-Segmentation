"""Train the customer segmenter and write reproducible reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from customer_segmentation.io import save_bundle
from customer_segmentation.model import model_bundle, segment_summary, train_segmenter
from customer_segmentation.validation import load_customer_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/Mall_Customers.csv"))
    parser.add_argument("--clusters", type=int, default=5)
    parser.add_argument("--model", type=Path, default=Path("artifacts/customer_segmenter.joblib"))
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = load_customer_data(args.data)
    model, metrics = train_segmenter(frame, clusters=args.clusters)
    summary = segment_summary(model, frame)

    args.reports.mkdir(parents=True, exist_ok=True)
    (args.reports / "metrics.json").write_text(
        json.dumps(metrics.as_dict(), indent=2) + "\n", encoding="utf-8"
    )
    summary.to_csv(args.reports / "segment_summary.csv", index=False)
    save_bundle(model_bundle(model, metrics), args.model)
    print(json.dumps(metrics.as_dict(), indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

