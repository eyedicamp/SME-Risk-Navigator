from __future__ import annotations

from pathlib import Path
from typing import Dict

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
DEMO_CASES_DIR = REPORTS_DIR / "demo_cases"
GRADE_THRESHOLDS_PATH = MODELS_DIR / "grade_thresholds.json"

DEFAULT_LABEL_COLUMN = "target"
DEFAULT_RANDOM_SEED = 42
DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"
DEFAULT_TOP_K_DRIVERS = 10

# Upper-bound thresholds, evaluated with strict `<` semantics.
GRADE_UPPER_BOUNDS: Dict[str, float] = {
    "A": 0.05,
    "B": 0.10,
    "C": 0.20,
    "D": 0.35,
    "E": 1.01,
}
