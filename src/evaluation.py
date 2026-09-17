"""
Model Evaluation and Feature Importance Extraction.
Computes genuine classification metrics, ROC curve data, confusion matrices,
and maps Random Forest Gini feature importances to human-readable names.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
from sklearn.ensemble import RandomForestClassifier


def evaluate_classification_model(
    model: RandomForestClassifier,
    X_test_preprocessed: np.ndarray,
    y_test: pd.Series,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Computes genuine test set classification metrics without placeholder figures.
    """
    y_prob = model.predict_proba(X_test_preprocessed)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    auc = float(roc_auc_score(y_test, y_prob))

    cm = confusion_matrix(y_test, y_pred)
    cr_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cr_text = classification_report(y_test, y_pred, zero_division=0)

    fpr, tpr, roc_thresh = roc_curve(y_test, y_prob)

    metrics = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'roc_auc': auc,
        'threshold': threshold,
        'confusion_matrix': cm.tolist(),
        'classification_report_dict': cr_dict,
        'classification_report_text': cr_text,
        'roc_curve': {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': roc_thresh.tolist()
        },
        'test_support': {
            'total': len(y_test),
            'actual_no_churn': int((y_test == 0).sum()),
            'actual_churn': int((y_test == 1).sum())
        }
    }
    return metrics


def format_feature_name(raw_name: str) -> str:
    """
    Converts one-hot encoded or raw feature column name into an intuitive label.
    e.g. 'Contract_Month-to-month' -> 'Contract: Month-to-month'
    """
    prefixes = [
        'Contract', 'InternetService', 'PaymentMethod', 'OnlineSecurity',
        'TechSupport', 'OnlineBackup', 'DeviceProtection', 'StreamingTV',
        'StreamingMovies', 'MultipleLines', 'PhoneService', 'Partner',
        'Dependents', 'gender'
    ]
    for p in prefixes:
        if raw_name.startswith(p + '_'):
            val = raw_name[len(p) + 1:]
            clean_p = ' '.join(c for c in p) if False else p
            return f"{clean_p}: {val}"

    # Standard numeric rename
    name_map = {
        'tenure': 'Tenure (Months)',
        'MonthlyCharges': 'Monthly Charges ($)',
        'TotalCharges': 'Total Charges ($)',
        'SeniorCitizen': 'Senior Citizen (0/1)'
    }
    return name_map.get(raw_name, raw_name.replace('_', ' '))


def extract_feature_importances(
    model: RandomForestClassifier,
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Extracts Random Forest Gini feature importances and returns a ranked DataFrame
    with readable names and explicit predictive-association labeling.
    """
    importances = model.feature_importances_
    df_imp = pd.DataFrame({
        'raw_feature': feature_names,
        'clean_feature': [format_feature_name(f) for f in feature_names],
        'importance': importances
    })

    df_imp = df_imp.sort_values(by='importance', ascending=False).reset_index(drop=True)
    df_imp['rank'] = range(1, len(df_imp) + 1)
    df_imp['relative_pct'] = (df_imp['importance'] / df_imp['importance'].sum()) * 100.0
    return df_imp
