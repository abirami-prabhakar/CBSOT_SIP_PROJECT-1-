"""
Supervised Churn Prediction Modeling.
Implements leak-free train/test splitting, ColumnTransformer preprocessing,
and hyperparameter-optimized Random Forest training via RandomizedSearchCV.
"""

from typing import Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer

from src.data_preprocessing import build_preprocessor


def train_churn_model(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    n_iter: int = 10,
    cv: int = 3,
    n_jobs: int = -1
) -> Tuple[RandomForestClassifier, ColumnTransformer, pd.DataFrame, pd.Series, np.ndarray, Dict[str, Any]]:
    """
    Executes the full supervised training pipeline:
    1. Stratified train/test split to prevent leakage and preserve class ratio.
    2. Fits ColumnTransformer strictly on X_train.
    3. Optimizes RandomForestClassifier using RandomizedSearchCV (3-fold CV, roc_auc scoring).
    4. Evaluates and returns the best model, preprocessor, test splits, and search results.
    """
    # 1. Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # 2. Fit Preprocessor on Training Data Only
    preprocessor = build_preprocessor()
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    # 3. Hyperparameter Distributions for Random Forest
    param_dist = {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [None, 8, 12, 16, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
        'class_weight': ['balanced', 'balanced_subsample', None]
    }

    base_rf = RandomForestClassifier(random_state=random_state)

    rf_search = RandomizedSearchCV(
        estimator=base_rf,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring='roc_auc',
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=1
    )

    # 4. Fit Search
    rf_search.fit(X_train_preprocessed, y_train)
    best_model: RandomForestClassifier = rf_search.best_estimator_

    tuning_info = {
        'best_params': rf_search.best_params_,
        'best_cv_roc_auc': float(rf_search.best_score_),
        'cv_splits': cv,
        'n_iter': n_iter
    }

    return best_model, preprocessor, X_test, y_test, X_test_preprocessed, tuning_info
