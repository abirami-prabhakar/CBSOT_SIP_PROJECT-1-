"""
Tests for Data Preprocessing and Harmonization Pipeline.
"""

import pandas as pd
import numpy as np
import pytest

from src.data_preprocessing import (
    harmonize_columns,
    clean_telco_data,
    build_preprocessor,
    ALL_FEATURE_COLS,
    NUMERICAL_COLS,
    CATEGORICAL_COLS
)


def test_harmonize_columns_space_names():
    raw_df = pd.DataFrame({
        'Tenure Months': [12, 24],
        'Monthly Charges': [70.5, 80.0],
        'Total Charges': ['846.0', '1920.0'],
        'Churn Label': ['No', 'Yes'],
        'Gender': ['Male', 'Female']
    })
    harmonized = harmonize_columns(raw_df)
    assert 'tenure' in harmonized.columns
    assert 'MonthlyCharges' in harmonized.columns
    assert 'TotalCharges' in harmonized.columns
    assert 'Churn' in harmonized.columns
    assert 'gender' in harmonized.columns


def test_clean_telco_data_missing_total_charges():
    sample_df = pd.DataFrame({
        'customerID': ['001', '002', '003'],
        'gender': ['Male', 'Female', 'Male'],
        'SeniorCitizen': [0, 1, 0],
        'Partner': ['Yes', 'No', 'No'],
        'Dependents': ['No', 'No', 'No'],
        'tenure': [0, 5, 10],
        'PhoneService': ['Yes', 'Yes', 'No'],
        'MultipleLines': ['No', 'No', 'No phone service'],
        'InternetService': ['DSL', 'Fiber optic', 'No'],
        'OnlineSecurity': ['Yes', 'No', 'No internet service'],
        'OnlineBackup': ['No', 'Yes', 'No internet service'],
        'DeviceProtection': ['No', 'No', 'No internet service'],
        'TechSupport': ['No', 'No', 'No internet service'],
        'StreamingTV': ['No', 'No', 'No internet service'],
        'StreamingMovies': ['No', 'No', 'No internet service'],
        'Contract': ['Month-to-month', 'One year', 'Two year'],
        'PaperlessBilling': ['Yes', 'No', 'Yes'],
        'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)'],
        'MonthlyCharges': [20.0, 85.0, 50.0],
        'TotalCharges': [' ', '425.0', np.nan],
        'Churn': ['No', 'Yes', 'No']
    })

    X, y = clean_telco_data(sample_df, is_training=True)

    # Verify ID is dropped
    assert 'customerID' not in X.columns
    # Verify tenure 0 TotalCharges was assigned 0.0
    assert X.loc[0, 'TotalCharges'] == 0.0
    # Verify no nulls in TotalCharges
    assert not X['TotalCharges'].isna().any()
    # Verify binary target encoding
    assert list(y) == [0, 1, 0]


def test_preprocessor_pipeline_transformation():
    sample_df = pd.DataFrame({
        'gender': ['Male', 'Female'],
        'SeniorCitizen': [0, 1],
        'Partner': ['Yes', 'No'],
        'Dependents': ['No', 'No'],
        'tenure': [12, 36],
        'PhoneService': ['Yes', 'Yes'],
        'MultipleLines': ['No', 'Yes'],
        'InternetService': ['DSL', 'Fiber optic'],
        'OnlineSecurity': ['Yes', 'No'],
        'OnlineBackup': ['No', 'Yes'],
        'DeviceProtection': ['No', 'No'],
        'TechSupport': ['No', 'No'],
        'StreamingTV': ['No', 'No'],
        'StreamingMovies': ['No', 'No'],
        'Contract': ['Month-to-month', 'Two year'],
        'PaperlessBilling': ['Yes', 'No'],
        'PaymentMethod': ['Electronic check', 'Credit card (automatic)'],
        'MonthlyCharges': [55.0, 95.0],
        'TotalCharges': [660.0, 3420.0]
    })

    preprocessor = build_preprocessor()
    X_trans = preprocessor.fit_transform(sample_df)

    assert isinstance(X_trans, np.ndarray)
    assert X_trans.shape[0] == 2
    # Ensure no NaN produced
    assert not np.isnan(X_trans).any()
