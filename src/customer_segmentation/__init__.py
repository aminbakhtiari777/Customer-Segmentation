"""Customer segmentation training and inference package."""

from .model import (
    CustomerSegmenter,
    SegmentationMetrics,
    SegmentPrediction,
    train_segmenter,
)
from .validation import FEATURES, load_customer_data, validate_customer_data

__all__ = [
    "CustomerSegmenter",
    "FEATURES",
    "SegmentationMetrics",
    "SegmentPrediction",
    "load_customer_data",
    "train_segmenter",
    "validate_customer_data",
]

