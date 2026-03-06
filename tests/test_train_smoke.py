from pathlib import Path

import pandas as pd
from sklearn.datasets import make_classification

from src.train import run_training


def test_train_smoke(tmp_path: Path) -> None:
    X, y = make_classification(
        n_samples=240,
        n_features=8,
        n_informative=5,
        n_redundant=1,
        weights=[0.8, 0.2],
        random_state=42,
    )
    columns = [f"f{i}" for i in range(8)]
    df = pd.DataFrame(X, columns=columns)
    df["target"] = y

    data_path = tmp_path / "toy.csv"
    df.to_csv(data_path, index=False)

    model_dir = tmp_path / "models"
    report_dir = tmp_path / "reports"
    processed_dir = tmp_path / "processed"

    metrics = run_training(
        data_path=data_path,
        label_col="target",
        seed=42,
        model_dir=model_dir,
        report_dir=report_dir,
        processed_dir=processed_dir,
        generate_shap=False,
        n_estimators=80,
        early_stopping_rounds=10,
    )

    assert (model_dir / "xgb_model.joblib").exists()
    assert (model_dir / "calibrated_model.joblib").exists()
    assert (report_dir / "metrics.json").exists()
    assert metrics["xgb_calibrated"]["auroc"] >= 0.0
