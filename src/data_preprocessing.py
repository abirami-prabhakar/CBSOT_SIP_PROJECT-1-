"""
Data Preprocessing Pipeline for Telco Customer Churn & Segmentation.
Provides reusable data cleaning, schema harmonization, and Scikit-Learn transformers.
"""

from typing import Tuple, List, Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Canonical feature schema
NUMERICAL_COLS: List[str] = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']

CATEGORICAL_COLS: List[str] = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod'
]

ALL_FEATURE_COLS: List[str] = NUMERICAL_COLS + CATEGORICAL_COLS
TARGET_COL: str = 'Churn'

# Column name alias mapping for robust dataset ingestion
COLUMN_ALIASES: Dict[str, str] = {
    'tenure months': 'tenure',
    'tenure': 'tenure',
    'monthly charges': 'MonthlyCharges',
    'monthlycharges': 'MonthlyCharges',
    'total charges': 'TotalCharges',
    'totalcharges': 'TotalCharges',
    'churn label': 'Churn',
    'churn value': 'Churn',
    'churn': 'Churn',
    'gender': 'gender',
    'senior citizen': 'SeniorCitizen',
    'seniorcitizen': 'SeniorCitizen',
    'partner': 'Partner',
    'dependents': 'Dependents',
    'phone service': 'PhoneService',
    'phoneservice': 'PhoneService',
    'multiple lines': 'MultipleLines',
    'multiplelines': 'MultipleLines',
    'internet service': 'InternetService',
    'internetservice': 'InternetService',
    'online security': 'OnlineSecurity',
    'onlinesecurity': 'OnlineSecurity',
    'online backup': 'OnlineBackup',
    'onlinebackup': 'OnlineBackup',
    'device protection': 'DeviceProtection',
    'deviceprotection': 'DeviceProtection',
    'tech support': 'TechSupport',
    'techsupport': 'TechSupport',
    'streaming tv': 'StreamingTV',
    'streamingtv': 'StreamingTV',
    'streaming movies': 'StreamingMovies',
    'streamingmovies': 'StreamingMovies',
    'contract': 'Contract',
    'paperless billing': 'PaperlessBilling',
    'paperlessbilling': 'PaperlessBilling',
    'payment method': 'PaymentMethod',
    'paymentmethod': 'PaymentMethod',
    'customerid': 'customerID',
    'customer id': 'customerID'
}


def harmonize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes column names across various Telco dataset releases
    (e.g., 'Tenure Months' -> 'tenure', 'Monthly Charges' -> 'MonthlyCharges').
    """
    df = df.copy()
    rename_map = {}
    for col in df.columns:
        norm_key = str(col).strip().lower()
        if norm_key in COLUMN_ALIASES:
            rename_map[col] = COLUMN_ALIASES[norm_key]
    df.rename(columns=rename_map, inplace=True)
    return df


def clean_telco_data(df: pd.DataFrame, is_training: bool = True) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Cleans raw Telco dataset:
    - Harmonizes column names
    - Strips whitespace from string fields
    - Casts TotalCharges to numeric, handling empty strings and NaN
    - Sets TotalCharges = 0.0 for customers with tenure == 0 if missing
    - Encodes Churn target (Yes/No/1/0) to binary int (1/0) if present
    - Drops customerID identifier column
    """
    df_clean = harmonize_columns(df)

    # Strip whitespace in all object columns
    for col in df_clean.select_dtypes(include=['object', 'string']).columns:
        df_clean[col] = df_clean[col].astype(str).str.strip()

    # Clean TotalCharges
    if 'TotalCharges' in df_clean.columns:
        df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'].replace('', np.nan), errors='coerce')
        if 'tenure' in df_clean.columns:
            # Customers with 0 tenure haven't accrued total charges yet
            zero_tenure_mask = (df_clean['tenure'] == 0) & (df_clean['TotalCharges'].isna())
            df_clean.loc[zero_tenure_mask, 'TotalCharges'] = 0.0
        
        # Median imputation for any residual missing charges
        median_total = df_clean['TotalCharges'].median()
        if pd.isna(median_total):
            median_total = 0.0
        df_clean['TotalCharges'] = df_clean['TotalCharges'].fillna(median_total)

    # Cast numeric columns
    for col in NUMERICAL_COLS:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

    # Drop customerID if present
    if 'customerID' in df_clean.columns:
        df_clean.drop(columns=['customerID'], inplace=True)

    # Target extraction & encoding
    y: Optional[pd.Series] = None
    if TARGET_COL in df_clean.columns:
        # Standardize target values (Yes/No, True/False, 1/0)
        target_series = df_clean[TARGET_COL].astype(str).str.strip().str.lower()
        mapping = {'yes': 1, '1': 1, 'true': 1, 'no': 0, '0': 0, 'false': 0}
        y = target_series.map(mapping).fillna(0).astype(int)
        df_clean.drop(columns=[TARGET_COL], inplace=True)
    elif is_training:
        raise ValueError(f"Target column '{TARGET_COL}' not found in training dataset.")

    # Retain only required feature columns if present
    available_features = [c for c in ALL_FEATURE_COLS if c in df_clean.columns]
    df_features = df_clean[available_features]

    return df_features, y


def build_preprocessor() -> ColumnTransformer:
    """
    Creates a leak-free Scikit-Learn ColumnTransformer pipeline:
    - Numerical: SimpleImputer (median) + StandardScaler
    - Categorical: SimpleImputer (most_frequent) + OneHotEncoder (handle_unknown='ignore')
    """
    num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, NUMERICAL_COLS),
            ('cat', cat_pipeline, CATEGORICAL_COLS)
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """
    Retrieves human-readable feature names from the fitted ColumnTransformer.
    """
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        # Fallback manual reconstruction
        names = list(NUMERICAL_COLS)
        cat_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
        if hasattr(cat_encoder, 'categories_'):
            for col, cats in zip(CATEGORICAL_COLS, cat_encoder.categories_):
                for cat in cats:
                    names.append(f"{col}_{cat}")
        return names
