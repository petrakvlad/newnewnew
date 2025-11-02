from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit

try:  # pragma: no cover - optional dependency
    from lightgbm import LGBMClassifier, LGBMRegressor
except Exception:  # pragma: no cover - environment without LightGBM
    LGBMClassifier = None
    LGBMRegressor = None

try:  # pragma: no cover - optional dependency
    import ruptures as rpt
except Exception:  # pragma: no cover - environment without ruptures
    rpt = None

SECTORS: Tuple[str, ...] = ("virtual", "other", "tv")


@dataclass
class FoldClassificationResult:
    """Container for cross-validation fold outputs."""

    fold: int
    y_true: np.ndarray
    y_pred: np.ndarray
    y_prob: Optional[np.ndarray]
    report: Dict[str, Any]


@dataclass
class FoldRegressionResult:
    fold: int
    y_true: np.ndarray
    y_pred: np.ndarray
    mae: float
    rmse: float


def _default_classifier() -> Any:
    if LGBMClassifier is not None:  # pragma: no branch - deterministic
        return LGBMClassifier(
            objective="multiclass",
            num_class=len(SECTORS),
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            n_estimators=400,
        )
    return RandomForestClassifier(n_estimators=200, random_state=42)


def _default_regressor() -> Any:
    if LGBMRegressor is not None:  # pragma: no branch - deterministic
        return LGBMRegressor(
            objective="regression",
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            n_estimators=400,
        )
    return RandomForestRegressor(n_estimators=300, random_state=42)


def modeling_frame(df: pd.DataFrame, sectors: Sequence[str] = SECTORS) -> pd.DataFrame:
    """Prepare the modeling dataset used in steps 8–10."""

    pivot = (
        df.pivot_table(
            index=["user_id", "date"],
            columns="sector",
            values="spend",
            fill_value=0.0,
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )

    feature_cols = [
        "virtual_other_ratio",
        "virtual_tv_ratio",
        "other_tv_ratio",
        "gap_days",
        "churn_flag",
    ]
    available = [col for col in feature_cols if col in df.columns]
    features = df.drop_duplicates(["user_id", "date"])[["user_id", "date", *available]]

    merged = pivot.merge(features, on=["user_id", "date"], how="left")
    merged["next_date"] = merged.groupby("user_id")["date"].shift(-1)

    for sector in sectors:
        if sector not in merged.columns:
            merged[sector] = 0.0
        merged[f"next_{sector}"] = merged.groupby("user_id")[sector].shift(-1)

    merged["next_sector"] = (
        merged[[f"next_{s}" for s in sectors]]
        .idxmax(axis=1)
        .fillna("")
        .str.replace("next_", "", regex=False)
    )
    merged["next_spend"] = merged[[f"next_{s}" for s in sectors]].sum(axis=1)
    merged = merged.dropna(subset=["next_date"])

    # Keep churn_flag as numeric (0/1)
    if "churn_flag" in merged.columns and merged["churn_flag"].dtype == bool:
        merged["churn_flag"] = merged["churn_flag"].astype(int)

    return merged


def _split_features_targets(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    feature_cols = [
        col
        for col in df.columns
        if col
        not in {
            "user_id",
            "date",
            "next_date",
            "next_sector",
            "next_spend",
            "activate_other_tv",
        }
        and not col.startswith("next_")
    ]
    X = df[feature_cols].fillna(0)
    return X, feature_cols


def train_next_sector_classifier(
    df: pd.DataFrame,
    estimator: Optional[Any] = None,
    tscv: Optional[TimeSeriesSplit] = None,
) -> Tuple[Any, List[FoldClassificationResult], List[str]]:
    """Train the next-day sector classifier and capture fold-level outputs."""

    if estimator is None:
        estimator = _default_classifier()
    if tscv is None:
        tscv = TimeSeriesSplit(n_splits=3)

    df = df.copy()
    df = df[df["next_sector"] != ""]
    X, feature_cols = _split_features_targets(df)
    y = df["next_sector"].astype(str)

    fold_results: List[FoldClassificationResult] = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        est = clone(estimator)
        train_x, test_x = X.iloc[train_idx], X.iloc[test_idx]
        train_y, test_y = y.iloc[train_idx], y.iloc[test_idx]
        est.fit(train_x, train_y)
        preds = est.predict(test_x)
        prob = est.predict_proba(test_x) if hasattr(est, "predict_proba") else None
        report = classification_report(test_y, preds, output_dict=True, zero_division=0)
        fold_results.append(
            FoldClassificationResult(
                fold=fold,
                y_true=test_y.to_numpy(),
                y_pred=preds,
                y_prob=prob,
                report=report,
            )
        )
        estimator = est  # keep last fitted estimator

    return estimator, fold_results, feature_cols


def train_next_spend_regressor(
    df: pd.DataFrame,
    estimator: Optional[Any] = None,
    tscv: Optional[TimeSeriesSplit] = None,
) -> Tuple[Any, List[FoldRegressionResult], List[str]]:
    """Train the next-day spend regressor and capture diagnostics."""

    if estimator is None:
        estimator = _default_regressor()
    if tscv is None:
        tscv = TimeSeriesSplit(n_splits=3)

    X, feature_cols = _split_features_targets(df)
    y = df["next_spend"].fillna(0.0)

    fold_results: List[FoldRegressionResult] = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        est = clone(estimator)
        train_x, test_x = X.iloc[train_idx], X.iloc[test_idx]
        train_y, test_y = y.iloc[train_idx], y.iloc[test_idx]
        est.fit(train_x, train_y)
        preds = est.predict(test_x)
        mae = mean_absolute_error(test_y, preds)
        rmse = mean_squared_error(test_y, preds, squared=False)
        fold_results.append(
            FoldRegressionResult(
                fold=fold,
                y_true=test_y.to_numpy(),
                y_pred=preds,
                mae=mae,
                rmse=rmse,
            )
        )
        estimator = est

    return estimator, fold_results, feature_cols


def train_cross_sector_activation(
    df: pd.DataFrame,
    estimator: Optional[Any] = None,
    tscv: Optional[TimeSeriesSplit] = None,
) -> Tuple[Any, List[FoldClassificationResult], List[str]]:
    """Model the probability of activating in non-virtual sectors."""

    if estimator is None:
        estimator = _default_classifier()
    if tscv is None:
        tscv = TimeSeriesSplit(n_splits=3)

    df = df.copy()
    df["activate_other_tv"] = (df[[f"next_{s}" for s in SECTORS if s != "virtual"]].sum(axis=1) > 0).astype(int)
    X, feature_cols = _split_features_targets(df)
    y = df["activate_other_tv"].astype(int)

    fold_results: List[FoldClassificationResult] = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        est = clone(estimator)
        train_x, test_x = X.iloc[train_idx], X.iloc[test_idx]
        train_y, test_y = y.iloc[train_idx], y.iloc[test_idx]
        est.fit(train_x, train_y)
        preds = est.predict(test_x)
        prob = est.predict_proba(test_x)[:, 1] if hasattr(est, "predict_proba") else None
        report = {
            "roc_auc": roc_auc_score(test_y, prob) if prob is not None else np.nan,
            "pr_auc": average_precision_score(test_y, prob) if prob is not None else np.nan,
        }
        fold_results.append(
            FoldClassificationResult(
                fold=fold,
                y_true=test_y.to_numpy(),
                y_pred=preds,
                y_prob=None if prob is None else prob,
                report=report,
            )
        )
        estimator = est

    return estimator, fold_results, feature_cols


def prepare_confusion_matrix(result: FoldClassificationResult, labels: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Convert a fold result into a confusion matrix data frame."""

    labels = labels or sorted(np.unique(np.concatenate([result.y_true, result.y_pred])))
    matrix = confusion_matrix(result.y_true, result.y_pred, labels=labels)
    return pd.DataFrame(matrix, index=labels, columns=labels)


def prepare_feature_importance(model: Any, feature_names: Sequence[str]) -> pd.DataFrame:
    """Return feature importances if available."""

    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        values = np.mean(np.abs(coef), axis=0)
    else:
        return pd.DataFrame({"feature": feature_names, "importance": np.nan})

    return pd.DataFrame({"feature": feature_names, "importance": values}).sort_values("importance", ascending=False)


def prepare_regression_diagnostics(results: Sequence[FoldRegressionResult]) -> pd.DataFrame:
    """Flatten fold predictions to inspect residual behaviour."""

    records: List[Dict[str, Any]] = []
    for res in results:
        residuals = res.y_true - res.y_pred
        for true, pred, resid in zip(res.y_true, res.y_pred, residuals):
            records.append({
                "fold": res.fold,
                "true": true,
                "pred": pred,
                "residual": resid,
            })
    return pd.DataFrame(records)


def build_sequences(grid: pd.DataFrame, sectors: Sequence[str] = SECTORS) -> List[List[str]]:
    """Convert the user-day grid into dominant-sector sequences."""

    sequences: List[List[str]] = []
    for _, user_df in grid.groupby("user_id"):
        pivot = user_df.pivot_table(index="date", columns="sector", values="spend", fill_value=0.0)
        for sector in sectors:
            if sector not in pivot.columns:
                pivot[sector] = 0.0
        pivot = pivot[sorted(sectors)]
        sequence = pivot.apply(lambda row: "pause" if row.sum() == 0 else row.idxmax(), axis=1).tolist()
        sequences.append(sequence)
    return sequences


def sequence_transition_counts(sequences: Iterable[Sequence[str]]) -> pd.DataFrame:
    """Compute transition counts between consecutive states in the sequences."""

    counts: Dict[Tuple[str, str], int] = {}
    for seq in sequences:
        for current, nxt in zip(seq[:-1], seq[1:]):
            counts[(current, nxt)] = counts.get((current, nxt), 0) + 1
    if not counts:
        return pd.DataFrame(columns=["from", "to", "count"])
    data = [
        {"from": src, "to": dst, "count": count}
        for (src, dst), count in counts.items()
    ]
    return pd.DataFrame(data)


def detect_change_points(series: pd.Series, penalty: float = 5.0) -> List[int]:
    """Locate change points in the aggregate spend series."""

    if rpt is None or series.empty:  # pragma: no cover - optional dependency
        return []
    algo = rpt.Pelt(model="rbf").fit(series.to_numpy())
    return algo.predict(pen=penalty)


def detect_spend_anomalies(df: pd.DataFrame, contamination: float = 0.02) -> pd.DataFrame:
    """Score user-day spend levels with IsolationForest."""

    model = IsolationForest(contamination=contamination, random_state=42)
    scores = model.fit_predict(df[["spend"]])
    result = df.copy()
    result["anomaly_score"] = scores
    return result


def _transition_matrix(
    df: pd.DataFrame,
    sectors: Sequence[str],
    weighted: bool = False,
) -> pd.DataFrame:
    transitions = {
        (src, dst): 0.0
        for src in list(sectors) + ["pause"]
        for dst in list(sectors) + ["pause"]
    }
    df = df.sort_values(["user_id", "date"])
    for user_id, user_df in df.groupby("user_id"):
        prev_sector = None
        prev_spend = 0.0
        for _, row in user_df.iterrows():
            sector = row.get("sector", "pause")
            spend = row.get("spend", 0.0)
            if prev_sector is not None:
                key = (prev_sector, sector if sector else "pause")
                transitions[key] += prev_spend if weighted else 1.0
            prev_sector = sector if sector else "pause"
            prev_spend = spend
    matrix = pd.DataFrame(0.0, index=list(sectors) + ["pause"], columns=list(sectors) + ["pause"])
    for (src, dst), value in transitions.items():
        matrix.loc[src, dst] = value
    return matrix


def channel_performance_summary(feature_daily: pd.DataFrame) -> pd.DataFrame:
    """Aggregate acquisition channel metrics for step 10 visualisations."""

    group = feature_daily.groupby("acquisition_channel")
    summary = group.agg(
        users=("user_id", "nunique"),
        active_share=("spend", lambda x: float((x > 0).mean())),
        avg_total_spend=("spend", "mean"),
        avg_gap_days=("gap_days", "mean"),
        median_cooldown=("gap_days", "median"),
        virtual_focus=("virtual", "mean") if "virtual" in feature_daily.columns else ("spend", "mean"),
        virtual_other_ratio=("virtual_other_ratio", "mean"),
        virtual_tv_ratio=("virtual_tv_ratio", "mean"),
        other_tv_ratio=("other_tv_ratio", "mean"),
    ).reset_index()
    return summary


def channel_transition_matrices(
    df: pd.DataFrame,
    sectors: Sequence[str] = SECTORS,
    weighted: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Build transition matrices for every acquisition channel."""

    matrices: Dict[str, pd.DataFrame] = {}
    if "acquisition_channel" not in df.columns:
        return matrices
    for channel, channel_df in df.groupby("acquisition_channel"):
        matrix = _transition_matrix(channel_df, sectors, weighted=weighted)
        if matrix.values.sum() > 0:
            matrices[channel] = matrix
    return matrices
