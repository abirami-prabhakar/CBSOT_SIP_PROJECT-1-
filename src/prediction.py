"""
Inference Engine for Single and Batch Customer Churn & Segment Prediction.
Applies the identical fitted ColumnTransformer and models to eliminate training-serving skew.
"""

from typing import Dict, Any, Optional
import os
import joblib
import pandas as pd
import numpy as np

from src.data_preprocessing import clean_telco_data, ALL_FEATURE_COLS
from src.segmentation import predict_customer_segment, derive_persona_descriptions


class CustomerPredictor:
    """
    Loads saved model artifacts and performs live inference using the exact
    fitted Scikit-Learn pipelines.
    """

    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.rf_model = None
        self.preprocessor = None
        self.kmeans_model = None
        self.kmeans_scaler = None
        self.cluster_profiles = None
        self.model_metrics = None
        self.personas = None
        self.is_loaded = False
        self.load_artifacts()

    def load_artifacts(self) -> bool:
        """
        Loads all required serialized joblib artifacts from models_dir.
        """
        rf_path = os.path.join(self.models_dir, "churn_rf_model.joblib")
        prep_path = os.path.join(self.models_dir, "preprocessor.joblib")
        km_path = os.path.join(self.models_dir, "kmeans_model.joblib")
        scaler_path = os.path.join(self.models_dir, "kmeans_scaler.joblib")
        profiles_path = os.path.join(self.models_dir, "cluster_profiles.joblib")
        metrics_path = os.path.join(self.models_dir, "model_metrics.joblib")

        if not all(os.path.exists(p) for p in [rf_path, prep_path, km_path, scaler_path]):
            self.is_loaded = False
            return False

        self.rf_model = joblib.load(rf_path)
        self.preprocessor = joblib.load(prep_path)
        self.kmeans_model = joblib.load(km_path)
        self.kmeans_scaler = joblib.load(scaler_path)

        if os.path.exists(profiles_path):
            self.cluster_profiles = joblib.load(profiles_path)
            self.personas = derive_persona_descriptions(self.cluster_profiles)

        if os.path.exists(metrics_path):
            self.model_metrics = joblib.load(metrics_path)

        self.is_loaded = True
        return True

    def predict_single(
        self,
        customer_dict: Dict[str, Any],
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Executes live inference on a single customer dictionary.
        """
        if not self.is_loaded:
            raise RuntimeError("Model artifacts are not loaded. Run train.py first.")

        # Ensure all columns exist in single-row DataFrame
        df_raw = pd.DataFrame([customer_dict])
        df_clean, _ = clean_telco_data(df_raw, is_training=False)

        # 1. Supervised Churn Prediction
        X_preprocessed = self.preprocessor.transform(df_clean)
        proba = self.rf_model.predict_proba(X_preprocessed)[0]
        churn_prob = float(proba[1])
        no_churn_prob = float(proba[0])
        predicted_class = int(churn_prob >= threshold)
        churn_label = "Churn" if predicted_class == 1 else "No Churn"

        # 2. Unsupervised Segment Prediction
        segment_id = predict_customer_segment(
            customer_df=df_clean,
            kmeans_model=self.kmeans_model,
            scaler=self.kmeans_scaler
        )

        persona_info = self.personas.get(segment_id, {}) if self.personas else {}

        return {
            'predicted_class': predicted_class,
            'churn_label': churn_label,
            'churn_probability': round(churn_prob * 100.0, 1),
            'no_churn_probability': round(no_churn_prob * 100.0, 1),
            'threshold_used': threshold,
            'segment_id': segment_id,
            'segment_persona': persona_info,
            'customer_features': df_clean.iloc[0].to_dict()
        }

    def predict_batch(
        self,
        df_raw: pd.DataFrame,
        threshold: float = 0.5
    ) -> pd.DataFrame:
        """
        Executes inference on a batch DataFrame, appending predicted churn,
        churn probabilities, and segment IDs.
        """
        if not self.is_loaded:
            raise RuntimeError("Model artifacts are not loaded. Run train.py first.")

        df_clean, _ = clean_telco_data(df_raw, is_training=False)
        X_preprocessed = self.preprocessor.transform(df_clean)
        probabilities = self.rf_model.predict_proba(X_preprocessed)[:, 1]
        predictions = (probabilities >= threshold).astype(int)

        X_segment = self.kmeans_scaler.transform(df_clean[['tenure', 'MonthlyCharges', 'TotalCharges']])
        segments = self.kmeans_model.predict(X_segment)

        result_df = df_raw.copy()
        result_df['Predicted_Churn_Prob'] = np.round(probabilities * 100.0, 1)
        result_df['Predicted_Churn'] = ['Yes' if p == 1 else 'No' for p in predictions]
        result_df['Predicted_Segment'] = segments
        return result_df
