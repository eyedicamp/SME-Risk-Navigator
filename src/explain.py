from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from .config import (
    DEFAULT_LABEL_COLUMN,
    DEFAULT_TOP_K_DRIVERS,
    GRADE_THRESHOLDS_PATH,
    MODELS_DIR,
    REPORTS_DIR,
)
from .copilot import build_facts, generate_copilot_memo, get_fallback_memo
from .grade import load_grade_thresholds, pd_to_grade
from .utils import save_json


def _extract_shap_array(shap_values: Any) -> np.ndarray:
    # SHAP output varies by model/version:
    # - list[class] of (n_samples, n_features)
    # - ndarray (n_samples, n_features)
    # - ndarray (n_samples, n_features, n_classes)
    # - Explanation object with `.values`
    if hasattr(shap_values, "values"):
        arr = np.asarray(shap_values.values)
    elif isinstance(shap_values, list):
        if len(shap_values) == 0:
            return np.array([])
        arr = np.asarray(shap_values[-1])
    else:
        arr = np.asarray(shap_values)

    if arr.ndim == 3:
        # Prefer positive class for binary classification.
        class_idx = 1 if arr.shape[-1] > 1 else 0
        arr = arr[:, :, class_idx]
    elif arr.ndim > 3:
        arr = arr.reshape(arr.shape[0], -1)

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def save_global_shap_plot(
    model: Any,
    X: pd.DataFrame,
    out_path: Path,
    max_samples: int = 2000,
    top_k: int = 20,
) -> None:
    if X.shape[0] > max_samples:
        sampled = X.sample(n=max_samples, random_state=42)
    else:
        sampled = X

    explainer = shap.TreeExplainer(model)
    shap_values = _extract_shap_array(explainer.shap_values(sampled))
    if shap_values.size == 0:
        raise ValueError("SHAP values are empty. Cannot create global SHAP plot.")

    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][: min(top_k, len(mean_abs))]

    plt.figure(figsize=(8, 6))
    plt.barh(
        [sampled.columns[i] for i in order][::-1],
        mean_abs[order][::-1],
        color="#2C7FB8",
    )
    plt.xlabel("mean(|SHAP value|)")
    plt.title("Global Feature Importance (SHAP)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def get_local_drivers(
    model: Any,
    X_row: pd.DataFrame,
    feature_names: list[str],
    top_k: int = DEFAULT_TOP_K_DRIVERS,
) -> list[dict[str, Any]]:
    row_df = X_row.copy()
    if row_df.shape[0] != 1:
        raise ValueError("X_row must contain exactly one row.")

    explainer = shap.TreeExplainer(model)
    shap_values = _extract_shap_array(explainer.shap_values(row_df))
    if shap_values.ndim == 2:
        row_shap = shap_values[0]
    else:
        row_shap = shap_values.reshape(-1)

    abs_order = np.argsort(np.abs(row_shap))[::-1][: min(top_k, len(row_shap))]

    drivers: list[dict[str, Any]] = []
    for rank, idx in enumerate(abs_order, start=1):
        shap_val = float(row_shap[idx])
        drivers.append(
            {
                "feature": feature_names[idx],
                "value": float(row_df.iloc[0, idx]),
                "shap_value": shap_val,
                "direction": "risk_up" if shap_val > 0 else "risk_down",
                "abs_rank": rank,
            }
        )
    return drivers


def _pick_quantile_index(scores: np.ndarray, quantile: float) -> int:
    target = float(np.quantile(scores, quantile))
    return int(np.argmin(np.abs(scores - target)))


def run_explain(
    test_csv: Path,
    label_col: str = DEFAULT_LABEL_COLUMN,
    top_k: int = DEFAULT_TOP_K_DRIVERS,
    call_llm: bool = False,
    model_dir: Path = MODELS_DIR,
    report_dir: Path = REPORTS_DIR,
) -> dict[str, Any]:
    xgb_model = joblib.load(model_dir / "xgb_model.joblib")
    calibrated_model = joblib.load(model_dir / "calibrated_model.joblib")
    grade_thresholds = load_grade_thresholds(GRADE_THRESHOLDS_PATH)

    test_df = pd.read_csv(test_csv)
    if label_col not in test_df.columns:
        raise ValueError(f"Label column '{label_col}' not found in {test_csv}")

    X_test = test_df.drop(columns=[label_col])
    probs = calibrated_model.predict_proba(X_test.to_numpy(dtype=np.float32))[:, 1]

    cases = {
        "good": _pick_quantile_index(probs, 0.10),
        "borderline": _pick_quantile_index(probs, 0.50),
        "bad": _pick_quantile_index(probs, 0.90),
    }

    outputs: dict[str, Any] = {}
    for name, idx in cases.items():
        X_row = X_test.iloc[[idx]]
        pd_score = float(probs[idx])
        grade = pd_to_grade(pd_score, grade_thresholds)
        drivers = get_local_drivers(xgb_model, X_row, list(X_test.columns), top_k=top_k)
        key_inputs = {col: float(X_row.iloc[0][col]) for col in X_test.columns[: min(10, X_test.shape[1])]}
        facts = build_facts(pd_score, grade, drivers, key_inputs)

        if call_llm:
            memo = generate_copilot_memo(facts)
        else:
            memo = get_fallback_memo(
                pd_score,
                grade,
                drivers,
                "LLM call skipped (use --call_llm with OPENAI_API_KEY to generate real memo).",
            )
        memo["facts"] = facts

        out_path = report_dir / "demo_cases" / f"memo_case_{name}.json"
        save_json(memo, out_path)
        outputs[name] = memo

        if name == "borderline":
            save_json(
                {
                    "case": name,
                    "risk_score_pd": pd_score,
                    "risk_grade": grade,
                    "drivers": drivers,
                },
                report_dir / "shap_local_example.json",
            )

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SHAP local explanations and demo copilot cases")
    parser.add_argument("--test_data", type=Path, default=Path("data/processed/test.csv"))
    parser.add_argument("--label", type=str, default=DEFAULT_LABEL_COLUMN)
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K_DRIVERS)
    parser.add_argument("--call_llm", action="store_true")
    args = parser.parse_args()

    results = run_explain(
        test_csv=args.test_data,
        label_col=args.label,
        top_k=args.top_k,
        call_llm=args.call_llm,
    )
    print({"generated_cases": list(results.keys())})


if __name__ == "__main__":
    main()
