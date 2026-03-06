from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.copilot import build_facts, generate_copilot_memo, memo_to_markdown
from src.explain import get_local_drivers
from src.grade import load_grade_thresholds, pd_to_grade

MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"
PROCESSED_TEST_PATH = BASE_DIR / "data" / "processed" / "test.csv"
GRADE_THRESHOLDS_PATH = MODEL_DIR / "grade_thresholds.json"


@st.cache_resource
def load_models() -> dict[str, Any] | None:
    xgb_path = MODEL_DIR / "xgb_model.joblib"
    cal_path = MODEL_DIR / "calibrated_model.joblib"
    schema_path = MODEL_DIR / "feature_schema.json"

    if not xgb_path.exists() or not cal_path.exists() or not schema_path.exists():
        return None

    try:
        xgb = joblib.load(xgb_path)
        calibrated = joblib.load(cal_path)
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        grade_thresholds = load_grade_thresholds(GRADE_THRESHOLDS_PATH)
        return {
            "xgb": xgb,
            "calibrated": calibrated,
            "schema": schema,
            "grade_thresholds": grade_thresholds,
            "reference_case": _load_reference_case(
                calibrated,
                schema,
                grade_thresholds,
            ),
        }
    except Exception as exc:
        return {"load_error": str(exc)}


def _load_reference_case(
    calibrated_model: Any,
    schema: dict[str, Any],
    grade_thresholds: dict[str, float],
) -> dict[str, Any] | None:
    if not PROCESSED_TEST_PATH.exists():
        return None

    try:
        df = pd.read_csv(PROCESSED_TEST_PATH)
        feature_order = schema["feature_order"]
        if not all(col in df.columns for col in feature_order):
            return None

        X = df[feature_order].to_numpy(dtype=np.float32)
        probs = calibrated_model.predict_proba(X)[:, 1]
        grades = pd.Series([pd_to_grade(float(p), grade_thresholds) for p in probs])

        # Prefer borderline defaults for manual testing.
        for preferred_grade in ["C", "B", "D", "E", "A"]:
            idx_list = grades[grades == preferred_grade].index.tolist()
            if idx_list:
                idx = idx_list[len(idx_list) // 2]
                row = df.loc[idx, feature_order]
                return {
                    "idx": int(idx),
                    "pd": float(probs[idx]),
                    "grade": preferred_grade,
                    "inputs": {k: float(row[k]) for k in feature_order},
                }
    except Exception:
        return None
    return None


def _build_default_input(schema: dict[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for item in schema["features"]:
        values[item["feature_name"]] = float(item.get("default_value", 0.0))
    return values


def _init_manual_state(schema: dict[str, Any], defaults: dict[str, float]) -> None:
    feature_order = schema["feature_order"]
    signature = tuple(feature_order)
    if st.session_state.get("manual_signature") != signature:
        st.session_state["manual_signature"] = signature
        for name in feature_order:
            st.session_state[f"manual_{name}"] = float(defaults.get(name, 0.0))


def _manual_input_form(schema: dict[str, Any], defaults: dict[str, float]) -> dict[str, float]:
    values = {}
    st.subheader("Manual Input")
    for item in schema["features"]:
        name = item["feature_name"]
        default_val = float(defaults.get(name, item.get("default_value", 0.0)))
        min_val = float(item.get("min", default_val - 1.0))
        max_val = float(item.get("max", default_val + 1.0))
        if not np.isfinite(default_val):
            default_val = 0.0
        if not np.isfinite(min_val):
            min_val = default_val - 1.0
        if not np.isfinite(max_val):
            max_val = default_val + 1.0
        if min_val > max_val:
            min_val, max_val = max_val, min_val

        span = max_val - min_val
        step = max(abs(span) / 100, abs(default_val) * 0.02, 0.001)

        # Do not hard-limit user input with min/max. Show quantile range as guidance only.
        values[name] = st.number_input(
            label=name,
            value=default_val,
            step=step,
            format="%.6f",
            help=f"Suggested range from training quantiles: [{min_val:.6g}, {max_val:.6g}]",
            key=f"manual_{name}",
        )
    return values


def _csv_row_input(schema: dict[str, Any]) -> dict[str, float]:
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    defaults = _build_default_input(schema)

    if uploaded is None:
        st.info("Upload a CSV file in sidebar to select one row.")
        return defaults

    df = pd.read_csv(uploaded)
    if df.empty:
        st.warning("Uploaded CSV is empty. Using default values.")
        return defaults

    numeric_df = df.select_dtypes(include=[np.number]).copy()
    if numeric_df.empty:
        st.warning("No numeric columns in uploaded CSV. Using default values.")
        return defaults

    selected_idx = st.sidebar.number_input(
        "Row index",
        min_value=0,
        max_value=max(0, len(numeric_df) - 1),
        value=0,
        step=1,
    )

    row = numeric_df.iloc[int(selected_idx)]
    values = defaults.copy()
    for feature in schema["feature_order"]:
        if feature in row.index and pd.notna(row[feature]):
            values[feature] = float(row[feature])
    st.write("Selected CSV row (mapped numeric fields):")
    st.json(values)
    return values


def main() -> None:
    st.set_page_config(page_title="SME Risk Navigator", layout="wide")
    st.title("SME Risk Navigator")
    st.caption("XGBoost Risk Scoring + SHAP Explainability + LLM Copilot Memo")

    artifacts = load_models()

    st.sidebar.header("Data Source")
    source = st.sidebar.radio("Choose input mode", ["Manual Input", "CSV Upload"])

    if artifacts is None:
        st.sidebar.error("Model status: NOT LOADED")
        st.warning("Model artifacts are missing. Run: `python -m src.train --data data/raw/dataset.csv --label target`")
        return
    if isinstance(artifacts, dict) and artifacts.get("load_error"):
        st.sidebar.error("Model status: LOAD FAILED")
        st.error(f"Model loading failed: {artifacts['load_error']}")
        st.info(
            "Likely cause: model serialization version mismatch. "
            "Align scikit-learn/xgboost versions between training and deployment."
        )
        return

    st.sidebar.success("Model status: LOADED")
    schema = artifacts["schema"]
    reference_case = artifacts.get("reference_case")
    grade_thresholds = artifacts.get("grade_thresholds", {})
    if grade_thresholds:
        st.sidebar.caption(
            "Grade thresholds: "
            f"A<{grade_thresholds.get('A', 0.05):.6f}, "
            f"B<{grade_thresholds.get('B', 0.10):.6f}, "
            f"C<{grade_thresholds.get('C', 0.20):.6f}, "
            f"D<{grade_thresholds.get('D', 0.35):.6f}, "
            f"E>={grade_thresholds.get('D', 0.35):.6f}"
        )

    if reference_case:
        st.sidebar.info(
            f"Suggested default profile: Grade {reference_case['grade']} / PD {reference_case['pd']:.4f}"
        )

    if source == "Manual Input":
        default_values = _build_default_input(schema)
        if reference_case and "inputs" in reference_case:
            default_values.update(reference_case["inputs"])

        _init_manual_state(schema, default_values)

        if st.sidebar.button("Reset Manual Inputs to Suggested Case"):
            for fname, fval in default_values.items():
                st.session_state[f"manual_{fname}"] = float(fval)

        input_values = _manual_input_form(schema, default_values)
    else:
        input_values = _csv_row_input(schema)

    if st.button("Predict"):
        input_df = pd.DataFrame([input_values], columns=schema["feature_order"])
        X_np = input_df.to_numpy(dtype=np.float32)
        pd_score = float(
            artifacts["calibrated"].predict_proba(X_np)[:, 1][0]
        )
        raw_score = float(artifacts["xgb"].predict_proba(X_np)[:, 1][0])
        grade = pd_to_grade(pd_score, grade_thresholds)
        drivers = get_local_drivers(
            artifacts["xgb"],
            input_df,
            schema["feature_order"],
            top_k=10,
        )
        st.session_state["prediction"] = {
            "pd_score": pd_score,
            "raw_score": raw_score,
            "grade": grade,
            "drivers": drivers,
            "input_values": input_values,
        }

    prediction = st.session_state.get("prediction")
    if prediction:
        st.subheader("Risk Output")
        st.metric("Calibrated Risk Score (PD)", f"{prediction['pd_score']:.6f}")
        st.metric("Raw Model Score", f"{prediction['raw_score']:.6f}")
        st.metric("Risk Grade", prediction["grade"])

        st.subheader("Top Drivers")
        st.dataframe(pd.DataFrame(prediction["drivers"]))

        col_local, col_global = st.columns(2)

        with col_local:
            st.subheader("Local SHAP Plot")
            drivers_df = pd.DataFrame(prediction["drivers"]).sort_values("abs_rank", ascending=False)
            fig, ax = plt.subplots(figsize=(6, 4))
            colors = np.where(drivers_df["shap_value"] >= 0, "#D55E00", "#0072B2")
            ax.barh(drivers_df["feature"], drivers_df["shap_value"], color=colors)
            ax.axvline(0.0, color="black", linewidth=1)
            ax.set_xlabel("SHAP value")
            ax.set_ylabel("Feature")
            ax.set_title("Local Risk Drivers")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)

        with col_global:
            st.subheader("Global SHAP Importance")
            global_shap_path = REPORT_DIR / "shap_global.png"
            if global_shap_path.exists() and global_shap_path.stat().st_size > 0:
                st.image(
                    str(global_shap_path),
                    caption="Global SHAP importance (from training)",
                    use_column_width=True,
                )
            else:
                st.info("Global SHAP image not found. Run training without --skip_shap.")

        if st.button("Generate Copilot Memo"):
            key_inputs = {
                k: prediction["input_values"][k]
                for k in list(prediction["input_values"].keys())[: min(10, len(prediction["input_values"]))]
            }
            facts = build_facts(
                prediction["pd_score"],
                prediction["grade"],
                prediction["drivers"],
                key_inputs,
            )
            memo = generate_copilot_memo(facts)
            st.session_state["memo"] = memo

        memo = st.session_state.get("memo")
        if memo:
            st.subheader("Copilot JSON")
            st.json(memo)

            st.subheader("Copilot Rendered Memo")
            st.markdown(memo_to_markdown(memo))

            st.download_button(
                label="Download memo.json",
                data=json.dumps(memo, ensure_ascii=True, indent=2),
                file_name="memo.json",
                mime="application/json",
            )


if __name__ == "__main__":
    main()
