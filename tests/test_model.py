"""
Tests for Model Artifacts and Evaluation Metrics.
"""

import os
import joblib
import pytest
import pandas as pd
import numpy as np


def test_model_artifacts_exist():
    required_files = [
        "churn_rf_model.joblib",
        "preprocessor.joblib",
        "kmeans_model.joblib",
        "kmeans_scaler.joblib",
        "cluster_profiles.joblib",
        "model_metrics.joblib",
        "feature_importances.joblib"
    ]
    for fname in required_files:
        path = os.path.join("models", fname)
        assert os.path.exists(path), f"Missing artifact: {path}"
        assert os.path.getsize(path) > 100, f"Artifact file too small: {path}"


def test_evaluation_metrics_validity():
    metrics = joblib.load("models/model_metrics.joblib")
    assert 0.70 <= metrics['accuracy'] <= 1.0, f"Unexpected accuracy: {metrics['accuracy']}"
    assert 0.75 <= metrics['roc_auc'] <= 1.0, f"Unexpected ROC-AUC: {metrics['roc_auc']}"
    assert 0.0 <= metrics['precision'] <= 1.0
    assert 0.0 <= metrics['recall'] <= 1.0
    assert 0.0 <= metrics['f1_score'] <= 1.0

    cm = metrics['confusion_matrix']
    assert len(cm) == 2 and len(cm[0]) == 2
    # Ensure non-trivial predictions
    assert sum(sum(row) for row in cm) > 1000


def test_feature_importances_structure():
    df_imp = joblib.load("models/feature_importances.joblib")
    assert isinstance(df_imp, pd.DataFrame)
    assert 'clean_feature' in df_imp.columns
    assert 'importance' in df_imp.columns
    assert 'rank' in df_imp.columns
    # Gini importances should sum close to 1.0
    assert np.isclose(df_imp['importance'].sum(), 1.0, atol=1e-3)
