from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from .config import DEFAULT_LABEL_COLUMN, MODELS_DIR, REPORTS_DIR
from .utils import ks_statistic, save_json


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ks": float(ks_statistic(y_true, y_prob)),
    }


def save_calibration_plot(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
    plt.figure(figsize=(6, 5))
    plt.plot(prob_pred, prob_true, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def run_evaluation(
    test_csv: Path,
    label_col: str = DEFAULT_LABEL_COLUMN,
    model_dir: Path = MODELS_DIR,
    report_dir: Path = REPORTS_DIR,
) -> dict[str, Any]:
    xgb_path = model_dir / "xgb_model.joblib"
    cal_path = model_dir / "calibrated_model.joblib"
    if not xgb_path.exists() or not cal_path.exists():
        raise FileNotFoundError(
            "Model artifacts are missing. Run training first: "
            "python -m src.train --data data/raw/dataset.csv --label target"
        )

    test_df = pd.read_csv(test_csv)
    if label_col not in test_df.columns:
        raise ValueError(f"Label column '{label_col}' not found in {test_csv}")

    y_test = test_df[label_col].to_numpy()
    X_test = test_df.drop(columns=[label_col]).to_numpy(dtype=np.float32)

    xgb_model = joblib.load(xgb_path)
    calibrated_model = joblib.load(cal_path)

    xgb_prob = xgb_model.predict_proba(X_test)[:, 1]
    calibrated_prob = calibrated_model.predict_proba(X_test)[:, 1]

    results = {
        "xgb_raw": evaluate_predictions(y_test, xgb_prob),
        "xgb_calibrated": evaluate_predictions(y_test, calibrated_prob),
    }
    save_json(results, report_dir / "metrics_eval.json")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-evaluate saved models on test split")
    parser.add_argument("--test_data", type=Path, default=Path("data/processed/test.csv"))
    parser.add_argument("--label", type=str, default=DEFAULT_LABEL_COLUMN)
    args = parser.parse_args()

    results = run_evaluation(args.test_data, args.label)
    print(results)


if __name__ == "__main__":
    main()
