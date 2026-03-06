from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from .utils import ensure_dirs, get_logger


@dataclass
class PreparedData:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    imputer: SimpleImputer
    feature_names: list[str]
    feature_schema: dict[str, Any]


def _normalize_binary_label(label_series: pd.Series) -> pd.Series:
    cleaned = label_series.dropna()
    unique_values = list(pd.unique(cleaned))

    if len(unique_values) != 2:
        raise ValueError("Label column must contain exactly two classes for binary classification.")

    unique_set = set(unique_values)
    if unique_set.issubset({0, 1}):
        return label_series.astype(int)

    ordered = sorted(unique_values)
    mapping = {ordered[0]: 0, ordered[1]: 1}
    return label_series.map(mapping).astype(int)


def build_feature_schema(X_train: pd.DataFrame) -> dict[str, Any]:
    features: list[dict[str, Any]] = []
    for col in X_train.columns:
        series = X_train[col]
        q01 = float(np.nanquantile(series, 0.01))
        q99 = float(np.nanquantile(series, 0.99))
        if q01 == q99:
            q01 = float(series.min())
            q99 = float(series.max())
        features.append(
            {
                "feature_name": col,
                "default_value": float(series.median()),
                "min": q01,
                "max": q99,
                "description": f"Numeric financial feature: {col}",
            }
        )

    return {
        "feature_order": list(X_train.columns),
        "features": features,
    }


def save_feature_schema(feature_schema: dict[str, Any], schema_path: Path) -> None:
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    with schema_path.open("w", encoding="utf-8") as f:
        json.dump(feature_schema, f, ensure_ascii=True, indent=2)


def prepare_dataset(
    data_path: Path,
    label_col: str,
    id_col: str | None = None,
    seed: int = 42,
) -> PreparedData:
    logger = get_logger(__name__)
    df = pd.read_csv(data_path)

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in dataset.")

    y_raw = df[label_col]
    valid_idx = y_raw.dropna().index
    df = df.loc[valid_idx].reset_index(drop=True)
    y = _normalize_binary_label(df[label_col])

    X = df.drop(columns=[label_col])
    if id_col and id_col in X.columns:
        X = X.drop(columns=[id_col])

    numeric_X = X.select_dtypes(include=[np.number]).copy()
    dropped = sorted(set(X.columns) - set(numeric_X.columns))
    if dropped:
        logger.info("Dropping non-numeric columns: %s", dropped)

    if numeric_X.shape[1] == 0:
        raise ValueError("No numeric feature columns found after filtering.")

    X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(
        numeric_X,
        y,
        test_size=0.30,
        random_state=seed,
        stratify=y,
    )
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(
        X_temp_raw,
        y_temp,
        test_size=0.50,
        random_state=seed,
        stratify=y_temp,
    )

    imputer = SimpleImputer(strategy="median")
    X_train = pd.DataFrame(
        imputer.fit_transform(X_train_raw),
        columns=numeric_X.columns,
        index=X_train_raw.index,
    )
    X_val = pd.DataFrame(
        imputer.transform(X_val_raw),
        columns=numeric_X.columns,
        index=X_val_raw.index,
    )
    X_test = pd.DataFrame(
        imputer.transform(X_test_raw),
        columns=numeric_X.columns,
        index=X_test_raw.index,
    )

    feature_schema = build_feature_schema(X_train)

    return PreparedData(
        X_train=X_train.reset_index(drop=True),
        X_val=X_val.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_val=y_val.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        imputer=imputer,
        feature_names=list(numeric_X.columns),
        feature_schema=feature_schema,
    )


def save_processed_splits(
    prepared: PreparedData,
    processed_dir: Path,
    label_col: str,
) -> None:
    ensure_dirs(processed_dir)

    train_df = prepared.X_train.copy()
    train_df[label_col] = prepared.y_train.values
    val_df = prepared.X_val.copy()
    val_df[label_col] = prepared.y_val.values
    test_df = prepared.X_test.copy()
    test_df[label_col] = prepared.y_test.values

    train_df.to_csv(processed_dir / "train.csv", index=False)
    val_df.to_csv(processed_dir / "val.csv", index=False)
    test_df.to_csv(processed_dir / "test.csv", index=False)
