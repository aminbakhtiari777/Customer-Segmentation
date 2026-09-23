"""Training, evaluation, semantic labeling, and inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .validation import FEATURES, INCOME_COLUMN, SPENDING_COLUMN, validate_customer_data


@dataclass(frozen=True)
class SegmentationMetrics:
    samples: int
    clusters: int
    inertia: float
    silhouette: float
    davies_bouldin: float

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class SegmentPrediction:
    segment: str
    cluster_id: int
    annual_income_k: float
    spending_score: float

    def as_dict(self) -> dict[str, str | int | float]:
        return asdict(self)


def _semantic_segment(income_z: float, spending_z: float) -> str:
    if abs(income_z) < 0.5 and abs(spending_z) < 0.5:
        return "mainstream"
    if income_z >= 0 and spending_z >= 0:
        return "high-value"
    if income_z < 0 <= spending_z:
        return "high-engagement"
    if spending_z < 0 <= income_z:
        return "growth-opportunity"
    return "budget-conscious"


class CustomerSegmenter:
    """A fitted K-Means pipeline with stable human-readable segment names."""

    def __init__(self, pipeline: Pipeline, segment_names: dict[int, str]):
        self.pipeline = pipeline
        self.segment_names = segment_names

    def predict(self, annual_income_k: float, spending_score: float) -> SegmentPrediction:
        if annual_income_k < 0:
            raise ValueError("annual_income_k must be non-negative")
        if not 1 <= spending_score <= 100:
            raise ValueError("spending_score must be between 1 and 100")
        sample = pd.DataFrame(
            [{INCOME_COLUMN: annual_income_k, SPENDING_COLUMN: spending_score}]
        )
        cluster_id = int(self.pipeline.predict(sample)[0])
        return SegmentPrediction(
            segment=self.segment_names[cluster_id],
            cluster_id=cluster_id,
            annual_income_k=float(annual_income_k),
            spending_score=float(spending_score),
        )

    def assign(self, frame: pd.DataFrame) -> pd.DataFrame:
        checked = validate_customer_data(frame)
        output = checked.copy()
        output["cluster_id"] = self.pipeline.predict(output[list(FEATURES)])
        output["segment"] = output["cluster_id"].map(self.segment_names)
        return output


def train_segmenter(
    frame: pd.DataFrame, *, clusters: int = 5, random_state: int = 42
) -> tuple[CustomerSegmenter, SegmentationMetrics]:
    checked = validate_customer_data(frame)
    if not 2 <= clusters < len(checked):
        raise ValueError("clusters must be at least 2 and smaller than sample count")

    features = checked[list(FEATURES)]
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "kmeans",
                KMeans(n_clusters=clusters, random_state=random_state, n_init=20),
            ),
        ]
    )
    labels = pipeline.fit_predict(features)
    kmeans: KMeans = pipeline.named_steps["kmeans"]
    names = {
        index: _semantic_segment(float(income_z), float(spending_z))
        for index, (income_z, spending_z) in enumerate(kmeans.cluster_centers_)
    }
    if len(set(names.values())) != clusters:
        raise ValueError("cluster geometry did not produce unique semantic segment names")

    metrics = SegmentationMetrics(
        samples=len(checked),
        clusters=clusters,
        inertia=float(kmeans.inertia_),
        silhouette=float(silhouette_score(pipeline.named_steps["scaler"].transform(features), labels)),
        davies_bouldin=float(
            davies_bouldin_score(pipeline.named_steps["scaler"].transform(features), labels)
        ),
    )
    return CustomerSegmenter(pipeline, names), metrics


def segment_summary(model: CustomerSegmenter, frame: pd.DataFrame) -> pd.DataFrame:
    assigned = model.assign(frame)
    summary = (
        assigned.groupby("segment", as_index=False)
        .agg(
            customers=("CustomerID", "count"),
            average_age=("Age", "mean"),
            average_income_k=(INCOME_COLUMN, "mean"),
            average_spending_score=(SPENDING_COLUMN, "mean"),
        )
        .sort_values("average_income_k", ascending=False)
        .reset_index(drop=True)
    )
    summary["customer_share"] = summary["customers"] / len(assigned)
    return summary


def model_bundle(model: CustomerSegmenter, metrics: SegmentationMetrics) -> dict[str, Any]:
    return {"model": model, "metrics": metrics.as_dict(), "version": "1.0"}

