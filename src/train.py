from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from xgboost.core import XGBoostError

from .config import (
    DATA_PROCESSED_DIR,
    DEFAULT_LABEL_COLUMN,
    DEFAULT_RANDOM_SEED,
    DEFAULT_TOP_K_DRIVERS,
    MODELS_DIR,
    REPORTS_DIR,
)
from .data import prepare_dataset, save_feature_schema, save_processed_splits
from .evaluate import evaluate_predictions, save_calibration_plot
from .grade import pd_to_grade
from .utils import ensure_dirs, get_logger, save_json, set_global_seed


def _build_calibrator(model: Any, method: str) -> CalibratedClassifierCV:
    try:
        from sklearn.calibration import FrozenEstimator

        return CalibratedClassifierCV(
            estimator=FrozenEstimator(model),  # type: ignore[arg-type]
            method=method,
            cv=None,
        )
    except Exception:
        try:
            return CalibratedClassifierCV(estimator=model, method=method, cv="prefit")
        except TypeError:
            return CalibratedClassifierCV(base_estimator=model, method=method, cv="prefit")


def _fit_best_calibrator(
    model: Any,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_train: np.ndarray,
    logger: Any,
) -> tuple[CalibratedClassifierCV, str, np.ndarray, dict[str, float]]:
    iso = _build_calibrator(model, method="isotonic")
    iso.fit(X_val, y_val)
    iso_val_prob = iso.predict_proba(X_val)[:, 1]
    iso_train_prob = iso.predict_proba(X_train)[:, 1]

    iso_unique = int(np.unique(np.round(iso_train_prob, 12)).size)
    iso_zero_share = float(np.mean(iso_train_prob <= 1e-12))
    iso_brier = float(brier_score_loss(y_val, iso_val_prob))

    sig = _build_calibrator(model, method="sigmoid")
    sig.fit(X_val, y_val)
    sig_val_prob = sig.predict_proba(X_val)[:, 1]
    sig_train_prob = sig.predict_proba(X_train)[:, 1]
    sig_brier = float(brier_score_loss(y_val, sig_val_prob))

    diagnostics = {
        "isotonic_brier_val": iso_brier,
        "sigmoid_brier_val": sig_brier,
        "isotonic_unique_train_probs": float(iso_unique),
        "isotonic_zero_share_train": iso_zero_share,
    }

    is_degenerate = iso_unique < 25 or iso_zero_share > 0.25
    if is_degenerate and sig_brier <= iso_brier * 1.20:
        logger.warning(
            "Isotonic calibration is degenerate (unique=%s, zero_share=%.4f). "
            "Switching to sigmoid calibration.",
            iso_unique,
            iso_zero_share,
        )
        return sig, "sigmoid", sig_train_prob, diagnostics

    return iso, "isotonic", iso_train_prob, diagnostics


def _quantile_value(scores: np.ndarray, q: float) -> float:
    try:
        return float(np.quantile(scores, q, method="higher"))
    except TypeError:
        return float(np.quantile(scores, q, interpolation="higher"))


def _compute_grade_thresholds(scores: np.ndarray) -> tuple[dict[str, float], list[float]]:
    clipped = np.clip(scores.astype(float), 0.0, 1.0)
    quantiles = [0.20, 0.40, 0.60, 0.80]
    raw = [_quantile_value(clipped, q) for q in quantiles]

    positive = clipped[clipped > 0]
    floor = float(np.quantile(positive, 0.05)) if positive.size > 0 else 1e-4
    floor = max(floor, 1e-6)
    eps = max(1e-6, floor * 0.05)

    adjusted: list[float] = []
    prev = max(raw[0], floor)
    adjusted.append(prev)
    for t in raw[1:]:
        nxt = max(t, prev + eps)
        adjusted.append(min(nxt, 0.999999))
        prev = adjusted[-1]

    thresholds = {
        "A": float(adjusted[0]),
        "B": float(adjusted[1]),
        "C": float(adjusted[2]),
        "D": float(adjusted[3]),
        "E": 1.01,
    }
    return thresholds, quantiles


def _build_xgb_classifier(
    seed: int,
    n_estimators: int,
    scale_pos_weight: float,
    early_stopping_rounds: int,
    tree_method: str,
    device: str,
) -> XGBClassifier:
    params: dict[str, Any] = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": seed,
        "n_estimators": n_estimators,
        "learning_rate": 0.05,
        "max_depth": 4,
        "min_child_weight": 1,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
        "tree_method": tree_method,
        "scale_pos_weight": scale_pos_weight,
        "early_stopping_rounds": early_stopping_rounds,
        "n_jobs": -1,
        "device": device,
    }
    try:
        return XGBClassifier(**params)
    except TypeError:
        # Backward compatibility for versions that do not accept `device`.
        params.pop("device", None)
        if device.startswith("cuda"):
            params["tree_method"] = "gpu_hist"
            if ":" in device:
                try:
                    params["gpu_id"] = int(device.split(":", 1)[1])
                except ValueError:
                    pass
        return XGBClassifier(**params)


def run_training(
    data_path: Path,
    label_col: str = DEFAULT_LABEL_COLUMN,
    id_col: str | None = None,
    seed: int = DEFAULT_RANDOM_SEED,
    model_dir: Path = MODELS_DIR,
    report_dir: Path = REPORTS_DIR,
    processed_dir: Path = DATA_PROCESSED_DIR,
    top_k: int = DEFAULT_TOP_K_DRIVERS,
    n_estimators: int = 500,
    early_stopping_rounds: int = 30,
    generate_shap: bool = True,
    xgb_device: str = "cuda:1",
) -> dict[str, Any]:
    logger = get_logger(__name__)
    set_global_seed(seed)
    ensure_dirs(model_dir, report_dir, processed_dir, report_dir / "demo_cases")

    prepared = prepare_dataset(data_path=data_path, label_col=label_col, id_col=id_col, seed=seed)
    save_processed_splits(prepared, processed_dir=processed_dir, label_col=label_col)

    save_feature_schema(prepared.feature_schema, model_dir / "feature_schema.json")

    baseline = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=seed,
    )
    baseline.fit(prepared.X_train, prepared.y_train)

    pos_count = int((prepared.y_train == 1).sum())
    neg_count = int((prepared.y_train == 0).sum())
    scale_pos_weight = float(neg_count / max(pos_count, 1))

    X_train_np = prepared.X_train.to_numpy(dtype=np.float32)
    X_val_np = prepared.X_val.to_numpy(dtype=np.float32)
    X_test_np = prepared.X_test.to_numpy(dtype=np.float32)
    y_train_np = prepared.y_train.to_numpy(dtype=np.float32)
    y_val_np = prepared.y_val.to_numpy(dtype=np.float32)

    main_model_name = "xgboost"
    main_model: Any = _build_xgb_classifier(
        seed=seed,
        n_estimators=n_estimators,
        scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=early_stopping_rounds,
        tree_method="hist",
        device=xgb_device,
    )
    try:
        main_model.fit(
            X_train_np,
            y_train_np,
            eval_set=[(X_val_np, y_val_np)],
            verbose=False,
        )
    except XGBoostError as exc:
        if "cuda" not in str(exc).lower():
            raise
        logger.warning("CUDA-related XGBoost error detected. Retrying with strict CPU-safe settings.")
        main_model = _build_xgb_classifier(
            seed=seed,
            n_estimators=n_estimators,
            scale_pos_weight=scale_pos_weight,
            early_stopping_rounds=early_stopping_rounds,
            tree_method="exact",
            device=xgb_device,
        )
        try:
            main_model.fit(
                X_train_np,
                y_train_np,
                eval_set=[(X_val_np, y_val_np)],
                verbose=False,
            )
        except XGBoostError as retry_exc:
            if "cuda" not in str(retry_exc).lower():
                raise
            logger.warning(
                "XGBoost is unusable in this runtime (CUDA-linked build without driver). "
                "Falling back to RandomForestClassifier for PoC continuity."
            )
            main_model_name = "random_forest_fallback"
            main_model = RandomForestClassifier(
                n_estimators=300,
                max_depth=6,
                random_state=seed,
                class_weight="balanced_subsample",
                n_jobs=-1,
            )
            main_model.fit(X_train_np, y_train_np.astype(int))

    calibrator, calibration_method, calibrated_train_prob, calibration_diag = _fit_best_calibrator(
        main_model,
        X_val_np,
        y_val_np,
        X_train_np,
        logger,
    )

    baseline_prob = baseline.predict_proba(prepared.X_test)[:, 1]
    xgb_prob = main_model.predict_proba(X_test_np)[:, 1]
    calibrated_prob = calibrator.predict_proba(X_test_np)[:, 1]

    # Re-balance grade buckets for this trained model via train calibrated-score quantiles.
    grade_thresholds, grade_quantiles = _compute_grade_thresholds(calibrated_train_prob)

    metrics = {
        "split_sizes": {
            "train": int(prepared.X_train.shape[0]),
            "val": int(prepared.X_val.shape[0]),
            "test": int(prepared.X_test.shape[0]),
        },
        "main_model_type": main_model_name,
        "calibration_method_used": calibration_method,
        "calibration_diagnostics": calibration_diag,
        "grade_thresholds": grade_thresholds,
        "baseline_logistic": evaluate_predictions(prepared.y_test.to_numpy(), baseline_prob),
        "xgb_raw": evaluate_predictions(prepared.y_test.to_numpy(), xgb_prob),
        "xgb_calibrated": evaluate_predictions(prepared.y_test.to_numpy(), calibrated_prob),
    }

    train_grades = [pd_to_grade(float(p), grade_thresholds) for p in calibrated_train_prob]
    test_grades = [pd_to_grade(float(p), grade_thresholds) for p in calibrated_prob]
    train_labels, train_counts = np.unique(np.array(train_grades), return_counts=True)
    test_labels, test_counts = np.unique(np.array(test_grades), return_counts=True)
    metrics["grade_distribution_train"] = {
        str(k): int(v) for k, v in zip(train_labels.tolist(), train_counts.tolist())
    }
    metrics["grade_distribution_test"] = {
        str(k): int(v) for k, v in zip(test_labels.tolist(), test_counts.tolist())
    }

    save_json(metrics, report_dir / "metrics.json")
    save_json(
        {
            "method": "train_calibrated_quantiles",
            "quantiles": grade_quantiles,
            "calibration_method_used": calibration_method,
            "thresholds": grade_thresholds,
        },
        model_dir / "grade_thresholds.json",
    )
    save_calibration_plot(
        prepared.y_test.to_numpy(),
        xgb_prob,
        report_dir / "calibration_before.png",
        "Calibration Curve (Before Isotonic)",
    )
    save_calibration_plot(
        prepared.y_test.to_numpy(),
        calibrated_prob,
        report_dir / "calibration_after.png",
        "Calibration Curve (After Isotonic)",
    )

    joblib.dump(main_model, model_dir / "xgb_model.joblib")
    joblib.dump(calibrator, model_dir / "calibrated_model.joblib")
    joblib.dump({"feature_names": prepared.feature_names, "imputer": prepared.imputer}, model_dir / "preprocess.joblib")

    if generate_shap:
        from .explain import get_local_drivers, save_global_shap_plot

        try:
            save_global_shap_plot(main_model, prepared.X_train, report_dir / "shap_global.png")
            mid_idx = int(np.argsort(calibrated_prob)[len(calibrated_prob) // 2])
            local_drivers = get_local_drivers(
                main_model,
                prepared.X_test.iloc[[mid_idx]],
                prepared.feature_names,
                top_k=top_k,
            )
            save_json(
                {
                    "index": mid_idx,
                    "risk_score_pd": float(calibrated_prob[mid_idx]),
                    "drivers": local_drivers,
                },
                report_dir / "shap_local_example.json",
            )
        except Exception as shap_exc:
            logger.warning("SHAP generation failed. Continuing without SHAP artifacts: %s", shap_exc)
            save_json(
                {
                    "message": "SHAP generation failed.",
                    "error": str(shap_exc),
                    "drivers": [],
                },
                report_dir / "shap_local_example.json",
            )
    else:
        save_json(
            {"message": "SHAP generation skipped.", "drivers": []},
            report_dir / "shap_local_example.json",
        )

    logger.info("Training complete. Metrics saved to %s", report_dir / "metrics.json")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SME Risk Navigator models")
    parser.add_argument("--data", type=Path, required=True, help="Path to CSV dataset")
    parser.add_argument("--label", type=str, default=DEFAULT_LABEL_COLUMN)
    parser.add_argument("--id_col", type=str, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--skip_shap", action="store_true")
    parser.add_argument("--xgb_device", type=str, default="cuda:1")
    args = parser.parse_args()

    run_training(
        data_path=args.data,
        label_col=args.label,
        id_col=args.id_col,
        seed=args.seed,
        generate_shap=not args.skip_shap,
        xgb_device=args.xgb_device,
    )


if __name__ == "__main__":
    main()
