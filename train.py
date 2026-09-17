"""
End-to-End Training & Evaluation Pipeline for Telco Segmentation & Churn Prediction.
Executes data loading, cleaning, K-Means clustering, RandomizedSearchCV Random Forest tuning,
model evaluation on held-out test set, and artifact serialization.
"""

import os
import argparse
import joblib
import pandas as pd
import numpy as np

from src.data_preprocessing import clean_telco_data, get_feature_names
from src.segmentation import (
    compute_elbow_curve,
    train_kmeans_segmentation,
    calculate_cluster_statistics,
    derive_persona_descriptions
)
from src.churn_model import train_churn_model
from src.evaluation import evaluate_classification_model, extract_feature_importances


def run_training_pipeline(
    data_path: str = "data/Telco-Customer-Churn.csv",
    models_dir: str = "models",
    n_clusters: int = 4,
    random_state: int = 42,
    n_iter_search: int = 10,
    cv_folds: int = 3
):
    print("=" * 75)
    print("TELCO SUBSCRIBER SEGMENTATION & CHURN PREDICTION PIPELINE")
    print("=" * 75)

    os.makedirs(models_dir, exist_ok=True)

    # 1. Load Dataset
    print(f"\n[1/6] Loading dataset from: {data_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found at '{data_path}'.")

    df_raw = pd.read_csv(data_path)
    print(f"      Raw dataset shape: {df_raw.shape[0]:,} rows, {df_raw.shape[1]} columns")

    # 2. Data Cleaning & Sanitization
    print("\n[2/6] Sanitizing data and harmonizing schema...")
    X, y = clean_telco_data(df_raw, is_training=True)
    print(f"      Features available: {X.shape[1]} columns")
    churn_rate = (y.mean()) * 100.0
    print(f"      Class distribution: {int((y == 0).sum()):,} Non-Churn, {int((y == 1).sum()):,} Churned ({churn_rate:.1f}% Churn Rate)")

    # 3. Customer Segmentation (Unsupervised Branch)
    print(f"\n[3/6] Running Unsupervised Customer Segmentation (K-Means, K={n_clusters})...")
    print("      Computing Elbow curve across K=1..10...")
    elbow_data = compute_elbow_curve(X, k_min=1, k_max=10, random_state=random_state)

    print(f"      Fitting K-Means model with K={n_clusters}...")
    kmeans_model, kmeans_scaler, cluster_labels = train_kmeans_segmentation(
        X, n_clusters=n_clusters, random_state=random_state
    )

    cluster_stats = calculate_cluster_statistics(X, cluster_labels, target_series=y)
    personas = derive_persona_descriptions(cluster_stats)

    print("\n      --- Empirical Cluster Summary ---")
    for _, row in cluster_stats.iterrows():
        c_id = int(row['Cluster'])
        title = personas[c_id]['title']
        print(f"      Cluster {c_id} ({title}):")
        print(f"        • Subscribers: {int(row['Customer Count']):,} ({row['Share of Base (%)']}%)")
        print(f"        • Avg Tenure: {row['Avg Tenure (Mo)']} mos | Avg Monthly: ${row['Avg Monthly ($)']}")
        print(f"        • Actual Churn Rate: {row['Actual Churn Rate (%)']}%")

    # 4. Churn Prediction Model (Supervised Branch)
    print(f"\n[4/6] Training Supervised Random Forest Classifier with RandomizedSearchCV...")
    print(f"      Parameters: 80/20 Stratified Split, 3-Fold Cross-Validation, {n_iter_search} Iterations...")
    best_rf, preprocessor, X_test, y_test, X_test_preprocessed, tuning_info = train_churn_model(
        X, y,
        test_size=0.2,
        random_state=random_state,
        n_iter=n_iter_search,
        cv=cv_folds
    )
    print(f"      Best CV ROC-AUC: {tuning_info['best_cv_roc_auc']:.4f}")
    print(f"      Best Hyperparameters: {tuning_info['best_params']}")

    # 5. Model Evaluation
    print("\n[5/6] Evaluating Model on Held-out Test Set...")
    metrics = evaluate_classification_model(best_rf, X_test_preprocessed, y_test)
    feature_names = get_feature_names(preprocessor)
    df_importances = extract_feature_importances(best_rf, feature_names)

    print("\n" + "=" * 50)
    print("ACTUAL TEST SET EVALUATION METRICS:")
    print("=" * 50)
    print(f"  • Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  • Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"  • Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"  • F1-Score:  {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
    print(f"  • ROC-AUC:   {metrics['roc_auc']:.4f}")
    print("\nClassification Report:")
    print(metrics['classification_report_text'])

    print("\nTop 5 Churn Predictors (Statistical Association):")
    for _, row in df_importances.head(5).iterrows():
        print(f"  {int(row['rank'])}. {row['clean_feature']} (Score: {row['importance']:.4f})")

    # 6. Save Artifacts
    print(f"\n[6/6] Persisting artifacts to '{models_dir}/'...")
    joblib.dump(best_rf, os.path.join(models_dir, "churn_rf_model.joblib"))
    joblib.dump(preprocessor, os.path.join(models_dir, "preprocessor.joblib"))
    joblib.dump(kmeans_model, os.path.join(models_dir, "kmeans_model.joblib"))
    joblib.dump(kmeans_scaler, os.path.join(models_dir, "kmeans_scaler.joblib"))
    joblib.dump(cluster_stats, os.path.join(models_dir, "cluster_profiles.joblib"))
    joblib.dump(elbow_data, os.path.join(models_dir, "elbow_data.joblib"))
    joblib.dump(metrics, os.path.join(models_dir, "model_metrics.joblib"))
    joblib.dump(df_importances, os.path.join(models_dir, "feature_importances.joblib"))
    joblib.dump(tuning_info, os.path.join(models_dir, "tuning_info.joblib"))

    print("      All model artifacts successfully serialized!")
    print("=" * 75)
    print("TRAINING PIPELINE COMPLETE.")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Telco Churn & Segmentation Models")
    parser.add_argument("--data", type=str, default="data/Telco-Customer-Churn.csv", help="Path to Telco dataset CSV")
    parser.add_argument("--models", type=str, default="models", help="Output directory for serialized models")
    parser.add_argument("--clusters", type=int, default=4, help="Number of K-Means clusters")
    parser.add_argument("--iter", type=int, default=10, help="RandomizedSearchCV iterations")
    args = parser.parse_args()

    run_training_pipeline(
        data_path=args.data,
        models_dir=args.models,
        n_clusters=args.clusters,
        n_iter_search=args.iter
    )
