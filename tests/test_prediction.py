"""
Tests for Prediction and Inference Engine.
"""

import pytest
import pandas as pd
import numpy as np

from src.prediction import CustomerPredictor


@pytest.fixture(scope="module")
def predictor():
    pred = CustomerPredictor(models_dir="models")
    assert pred.is_loaded, "Predictor failed to load model artifacts"
    return pred


def test_predict_single_high_risk(predictor):
    # Customer with low tenure, high monthly charge, month-to-month, fiber optic
    customer = {
        'gender': 'Female',
        'SeniorCitizen': 0,
        'Partner': 'No',
        'Dependents': 'No',
        'tenure': 2,
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'Fiber optic',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'No',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'Yes',
        'StreamingMovies': 'No',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 85.0,
        'TotalCharges': 170.0
    }

    result = predictor.predict_single(customer, threshold=0.5)

    assert 'predicted_class' in result
    assert 'churn_probability' in result
    assert 'no_churn_probability' in result
    assert 'segment_id' in result
    assert 'segment_persona' in result
    assert 0.0 <= result['churn_probability'] <= 100.0
    assert 0.0 <= result['no_churn_probability'] <= 100.0
    assert result['segment_id'] in [0, 1, 2, 3]


def test_predict_single_loyal_customer(predictor):
    # Customer with long tenure, two-year contract, DSL, auto payment
    customer = {
        'gender': 'Male',
        'SeniorCitizen': 0,
        'Partner': 'Yes',
        'Dependents': 'Yes',
        'tenure': 65,
        'PhoneService': 'Yes',
        'MultipleLines': 'Yes',
        'InternetService': 'DSL',
        'OnlineSecurity': 'Yes',
        'OnlineBackup': 'Yes',
        'DeviceProtection': 'Yes',
        'TechSupport': 'Yes',
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'Contract': 'Two year',
        'PaperlessBilling': 'No',
        'PaymentMethod': 'Bank transfer (automatic)',
        'MonthlyCharges': 45.0,
        'TotalCharges': 2925.0
    }

    result = predictor.predict_single(customer, threshold=0.5)
    assert result['predicted_class'] == 0
    assert result['churn_label'] == "No Churn"
    assert result['churn_probability'] < 50.0


def test_predict_batch(predictor):
    sample_batch = pd.DataFrame([
        {
            'customerID': 'C01',
            'gender': 'Female',
            'SeniorCitizen': 0,
            'Partner': 'No',
            'Dependents': 'No',
            'tenure': 1,
            'PhoneService': 'Yes',
            'MultipleLines': 'No',
            'InternetService': 'Fiber optic',
            'OnlineSecurity': 'No',
            'OnlineBackup': 'No',
            'DeviceProtection': 'No',
            'TechSupport': 'No',
            'StreamingTV': 'No',
            'StreamingMovies': 'No',
            'Contract': 'Month-to-month',
            'PaperlessBilling': 'Yes',
            'PaymentMethod': 'Electronic check',
            'MonthlyCharges': 70.0,
            'TotalCharges': '70.0'
        },
        {
            'customerID': 'C02',
            'gender': 'Male',
            'SeniorCitizen': 0,
            'Partner': 'Yes',
            'Dependents': 'Yes',
            'tenure': 60,
            'PhoneService': 'Yes',
            'MultipleLines': 'Yes',
            'InternetService': 'Fiber optic',
            'OnlineSecurity': 'Yes',
            'OnlineBackup': 'Yes',
            'DeviceProtection': 'Yes',
            'TechSupport': 'Yes',
            'StreamingTV': 'Yes',
            'StreamingMovies': 'Yes',
            'Contract': 'Two year',
            'PaperlessBilling': 'No',
            'PaymentMethod': 'Credit card (automatic)',
            'MonthlyCharges': 105.0,
            'TotalCharges': '6300.0'
        }
    ])

    batch_res = predictor.predict_batch(sample_batch)
    assert 'Predicted_Churn' in batch_res.columns
    assert 'Predicted_Churn_Prob' in batch_res.columns
    assert 'Predicted_Segment' in batch_res.columns
    assert len(batch_res) == 2
