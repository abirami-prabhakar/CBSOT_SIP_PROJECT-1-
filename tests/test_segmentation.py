"""
Tests for Customer Segmentation and K-Means Module.
"""

import pandas as pd
import numpy as np
import pytest

from src.segmentation import (
    compute_elbow_curve,
    train_kmeans_segmentation,
    calculate_cluster_statistics,
    derive_persona_descriptions,
    predict_customer_segment
)


@pytest.fixture
def dummy_data():
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        'tenure': np.random.randint(1, 72, n),
        'MonthlyCharges': np.random.uniform(20, 120, n),
        'TotalCharges': np.random.uniform(50, 6000, n),
        'Churn': np.random.choice([0, 1], n, p=[0.7, 0.3])
    })


def test_compute_elbow_curve(dummy_data):
    res = compute_elbow_curve(dummy_data, k_min=1, k_max=5)
    assert 'k_values' in res
    assert 'inertias' in res
    assert res['k_values'] == [1, 2, 3, 4, 5]
    assert len(res['inertias']) == 5
    # Inertias should be strictly decreasing
    for i in range(len(res['inertias']) - 1):
        assert res['inertias'][i] > res['inertias'][i + 1]


def test_kmeans_training_and_stats(dummy_data):
    kmeans, scaler, labels = train_kmeans_segmentation(dummy_data, n_clusters=4)
    assert len(labels) == len(dummy_data)
    assert len(np.unique(labels)) == 4

    stats_df = calculate_cluster_statistics(dummy_data, labels, target_series=dummy_data['Churn'])
    assert len(stats_df) == 4
    assert 'Share of Base (%)' in stats_df.columns
    assert 'Avg Tenure (Mo)' in stats_df.columns
    assert 'Actual Churn Rate (%)' in stats_df.columns
    # Sum of shares should be approximately 100%
    assert round(stats_df['Share of Base (%)'].sum()) == 100

    personas = derive_persona_descriptions(stats_df)
    assert len(personas) == 4
    for c_id in range(4):
        assert 'title' in personas[c_id]
        assert 'description' in personas[c_id]
        assert 'strategy' in personas[c_id]


def test_predict_customer_segment(dummy_data):
    kmeans, scaler, _ = train_kmeans_segmentation(dummy_data, n_clusters=4)
    single = pd.DataFrame([{'tenure': 24, 'MonthlyCharges': 75.0, 'TotalCharges': 1800.0}])
    seg = predict_customer_segment(single, kmeans, scaler)
    assert seg in [0, 1, 2, 3]
