from __future__ import annotations

from pathlib import Path
from typing import Mapping
import json

from .config import GRADE_UPPER_BOUNDS


def pd_to_grade(pd_score: float, thresholds: Mapping[str, float] | None = None) -> str:
    bounds = thresholds or GRADE_UPPER_BOUNDS
    score = min(max(float(pd_score), 0.0), 1.0)

    ordered = sorted(bounds.items(), key=lambda item: item[1])
    for grade, upper in ordered:
        if score < upper:
            return grade
    return ordered[-1][0]


def load_grade_thresholds(path: Path) -> dict[str, float]:
    if not path.exists():
        return dict(GRADE_UPPER_BOUNDS)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "thresholds" in payload and isinstance(payload["thresholds"], dict):
            loaded = {str(k): float(v) for k, v in payload["thresholds"].items()}
        elif isinstance(payload, dict):
            loaded = {str(k): float(v) for k, v in payload.items()}
        else:
            return dict(GRADE_UPPER_BOUNDS)

        # Validate required keys and monotonicity by sorting on threshold values.
        required = {"A", "B", "C", "D", "E"}
        if not required.issubset(set(loaded.keys())):
            return dict(GRADE_UPPER_BOUNDS)
        return loaded
    except Exception:
        return dict(GRADE_UPPER_BOUNDS)
