"""Artifact persistence helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

from .model import CustomerSegmenter


def save_bundle(bundle: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, target)


def load_model(path: str | Path) -> CustomerSegmenter:
    bundle = joblib.load(path)
    if bundle.get("version") != "1.0" or not isinstance(bundle.get("model"), CustomerSegmenter):
        raise ValueError("unsupported or invalid model bundle")
    return bundle["model"]

