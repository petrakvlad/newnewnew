import pytest

pandas_module = pytest.importorskip('pandas')
np = pytest.importorskip('numpy')
sklearn_model_selection = pytest.importorskip('sklearn.model_selection', reason='scikit-learn required for tests')

pd = pandas_module
TimeSeriesSplit = sklearn_model_selection.TimeSeriesSplit

from player_analysis import (
    modeling_frame,
    train_next_sector_classifier,
    train_next_spend_regressor,
    train_cross_sector_activation,
    prepare_confusion_matrix,
    prepare_feature_importance,
    prepare_regression_diagnostics,
    build_sequences,
    sequence_transition_counts,
    detect_spend_anomalies,
    channel_performance_summary,
    channel_transition_matrices,
)


def _synthetic_feature_daily():
    dates = pd.date_range('2024-09-01', periods=6, freq='D')
    records = []
    for user_id, channel in [(1, 'organic'), (2, 'paid_search')]:
        for idx, date in enumerate(dates):
            for sector, weight in [('virtual', 1.0), ('tv', 0.6), ('other', 0.3)]:
                records.append({
                    'user_id': user_id,
                    'date': date,
                    'sector': sector,
                    'spend': float((idx + 1) * weight + user_id),
                    'virtual_other_ratio': 1.5 + 0.1 * idx,
                    'virtual_tv_ratio': 2.0 + 0.05 * idx,
                    'other_tv_ratio': 0.8 + 0.02 * idx,
                    'gap_days': float(idx),
                    'churn_flag': int(idx > 3),
                    'acquisition_channel': channel,
                })
    return pd.DataFrame.from_records(records)


def _synthetic_grid(feature_daily: pd.DataFrame) -> pd.DataFrame:
    # Mirror the structure expected by sequence and channel helpers
    return feature_daily[['user_id', 'date', 'sector', 'spend', 'acquisition_channel']].copy()


def test_modeling_frame_outputs_expected_columns():
    feature_daily = _synthetic_feature_daily()
    model_df = modeling_frame(feature_daily)
    assert {'next_sector', 'next_spend'}.issubset(model_df.columns)
    assert not model_df.empty


def test_predictive_models_return_metrics():
    feature_daily = _synthetic_feature_daily()
    model_df = modeling_frame(feature_daily)
    tscv = TimeSeriesSplit(n_splits=2)

    clf_model, clf_folds, clf_features = train_next_sector_classifier(model_df, tscv=tscv)
    assert clf_folds, 'Classifier should return fold metrics'
    cm = prepare_confusion_matrix(clf_folds[-1])
    assert cm.shape[0] == cm.shape[1] > 0
    importance = prepare_feature_importance(clf_model, clf_features)
    assert 'feature' in importance.columns

    reg_model, reg_folds, _ = train_next_spend_regressor(model_df, tscv=tscv)
    diagnostics = prepare_regression_diagnostics(reg_folds)
    assert {'true', 'pred', 'residual'}.issubset(diagnostics.columns)

    activation_model_fit, activation_folds, _ = train_cross_sector_activation(model_df, tscv=tscv)
    assert activation_folds, 'Activation model should return fold metrics'
    for res in activation_folds:
        if not np.isnan(res.report.get('roc_auc', np.nan)):
            assert 0.0 <= res.report['roc_auc'] <= 1.0


def test_sequence_and_anomaly_helpers_surface_results():
    feature_daily = _synthetic_feature_daily()
    grid = _synthetic_grid(feature_daily)
    sequences = build_sequences(grid)
    assert len(sequences) == grid['user_id'].nunique()
    transitions = sequence_transition_counts(sequences)
    assert {'from', 'to', 'count'}.issubset(transitions.columns)

    daily_spend = grid.groupby(['user_id', 'date'], as_index=False)['spend'].sum()
    anomalies = detect_spend_anomalies(daily_spend, contamination=0.2)
    assert 'anomaly_score' in anomalies.columns


def test_channel_summary_and_transitions():
    feature_daily = _synthetic_feature_daily()
    grid = _synthetic_grid(feature_daily)
    summary = channel_performance_summary(feature_daily)
    assert {'acquisition_channel', 'avg_total_spend', 'active_share'}.issubset(summary.columns)

    matrices = channel_transition_matrices(grid)
    assert matrices, 'Should build at least one channel transition matrix'
    for matrix in matrices.values():
        assert matrix.shape[0] == matrix.shape[1]
