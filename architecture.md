# Telco Customer Intelligence: End-to-End System Architecture

This document specifies the end-to-end data and machine learning architecture of the Telco Customer Segmentation & Churn Prediction platform.

---

## 1. System Flow & Data Pipeline Architecture

```text
               ┌─────────────────────────────────────────────────────────┐
               │         Raw Telco Dataset (7,043 Subscribers)           │
               │             data/Telco-Customer-Churn.csv               │
               └────────────────────────────┬────────────────────────────┘
                                            │
                                            ▼
               ┌─────────────────────────────────────────────────────────┐
               │          Data Cleaning & Schema Harmonization           │
               │   • Normalize column naming aliases (tenure, charges)   │
               │   • Whitespace trimming on text/categorical fields      │
               │   • TotalCharges numeric coercion & median imputation   │
               │   • Tenure = 0 adjustment (TotalCharges = $0.00)        │
               │   • Drop non-predictive identifiers (customerID)        │
               └────────────────────────────┬────────────────────────────┘
                                            │
                     ┌──────────────────────┴──────────────────────┐
                     │                                             │
                     ▼                                             ▼
       ┌───────────────────────────┐                 ┌───────────────────────────┐
       │   Unsupervised Branch     │                 │     Supervised Branch     │
       │  (Customer Segmentation)  │                 │    (Churn Prediction)     │
       ├───────────────────────────┤                 ├───────────────────────────┤
       │ • Feature Extraction:     │                 │ • Stratified Split (80/20)│
       │   tenure, MonthlyCharges, │                 │   Train (80%), Test (20%) │
       │   TotalCharges            │                 │ • ColumnTransformer:      │
       │ • StandardScaler          │                 │   - SimpleImputer(median) │
       │ • Elbow Method Evaluation │                 │   - StandardScaler()      │
       │   (K=1 to K=10 WCSS drop) │                 │   - OneHotEncoder(ignore) │
       │ • K-Means (K=4 Clusters)  │                 │ • RandomForestClassifier  │
       │ • Empirical Profiling &   │                 │ • RandomizedSearchCV      │
       │   Persona Derivation      │                 │   (3-Fold Cross Validation│
       │   (Tenure, Spend, Churn%) │                 │    ROC-AUC optimization) │
       └─────────────┬─────────────┘                 └─────────────┬─────────────┘
                     │                                             │
                     └──────────────────────┬──────────────────────┘
                                            │
                                            ▼
               ┌─────────────────────────────────────────────────────────┐
               │           Model Persistence via Joblib (models/)        │
               │   • preprocessor.joblib       • churn_rf_model.joblib   │
               │   • kmeans_model.joblib       • kmeans_scaler.joblib    │
               │   • cluster_profiles.joblib   • model_metrics.joblib    │
               │   • elbow_data.joblib         • feature_importances.job │
               └────────────────────────────┬────────────────────────────┘
                                            │
                     ┌──────────────────────┴──────────────────────┐
                     │                                             │
                     ▼                                             ▼
       ┌───────────────────────────┐                 ┌───────────────────────────┐
       │    CustomerPredictor      │                 │    Evaluation & Audit     │
       │    (Real-time Inference)  │                 │    (Held-out Test Set)    │
       ├───────────────────────────┤                 ├───────────────────────────┤
       │ • Raw customer dict/df    │                 │ • Accuracy: 80.06%        │
       │ • preprocessor.transform()│                 │ • ROC-AUC:  0.8430        │
       │ • rf.predict_proba()      │                 │ • Precision: 66.55%       │
       │ • kmeans.predict()        │                 │ • Recall: 50.00%          │
       │ • Segment persona mapping │                 │ • F1-Score: 0.5710        │
       └─────────────┬─────────────┘                 └─────────────┬─────────────┘
                     │                                             │
                     └──────────────────────┬──────────────────────┘
                                            │
                                            ▼
               ┌─────────────────────────────────────────────────────────┐
               │          Streamlit Analytics Dashboard (app.py)         │
               ├─────────────────────────────────────────────────────────┤
               │  1. Executive Dashboard (KPIs, Donut Chart, Model Spec) │
               │  2. Customer Segments (Elbow Plot, 2D Scatter, Personas)│
               │  3. Real-Time Churn Predictor (Form, Probability Gauge) │
               │  4. Model Diagnostics (Confusion Matrix, ROC, Features) │
               │  5. Data Explorer (Filters, Distributions, Batch Score) │
               └─────────────────────────────────────────────────────────┘
```

---

## 2. Directory Layout & Module Responsibilities

```text
SIP/
├── app.py                      # Interactive Streamlit analytics web application
├── train.py                    # Standalone CLI training & artifact generation runner
├── requirements.txt            # Pinned environment requirements
├── architecture.md             # System architecture specification
├── README.md                   # Full portfolio documentation and empirical results
├── data/
│   └── Telco-Customer-Churn.csv # Verified IBM Telco Subscriber dataset (7,043 rows)
├── models/                     # Serialized Scikit-Learn models and evaluation data
│   ├── churn_rf_model.joblib   # Tuned Random Forest Classifier
│   ├── preprocessor.joblib     # Fitted ColumnTransformer (Scalers + Encoders)
│   ├── kmeans_model.joblib     # Fitted K-Means cluster model (K=4)
│   ├── kmeans_scaler.joblib    # StandardScaler for segmentation features
│   ├── cluster_profiles.joblib # Empirical summary statistics per cluster
│   ├── elbow_data.joblib       # WCSS inertias for K=1..10
│   ├── model_metrics.joblib    # Held-out test set accuracy, ROC-AUC, CM, etc.
│   ├── feature_importances.joblib # Ranked Gini importance table with clean labels
│   └── tuning_info.joblib      # RandomizedSearchCV hyperparameters and best CV score
├── notebooks/
│   └── CBSOT_SIP_PROJECT-1.ipynb # Preserved original exploration notebook
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py  # Data cleaning, schema harmonization, pipeline factory
│   ├── segmentation.py        # K-Means, Elbow method, empirical cluster persona engine
│   ├── churn_model.py         # Supervised train/test split, RF & RandomizedSearchCV
│   ├── evaluation.py          # Metric calculations, confusion matrix, feature ranking
│   ├── prediction.py          # Inference engine for live single and batch predictions
│   └── visualization.py       # Plotly-powered responsive charts for dashboard
└── tests/
    ├── __init__.py
    ├── test_preprocessing.py  # Unit tests for cleaning and ColumnTransformer
    ├── test_segmentation.py   # Unit tests for Elbow curve and K-Means logic
    ├── test_model.py          # Unit tests for saved model artifacts & metric validity
    └── test_prediction.py     # Unit tests for live single and batch inference
```

---

## 3. Data Flow & Transformation Stages

### Stage 1: Data Cleaning & Preprocessing
* **Column Harmonization:** Reconciles differences between various releases (`Tenure Months` vs `tenure`, `Monthly Charges` vs `MonthlyCharges`, `Churn Label` vs `Churn`).
* **Imputation:** Coerces empty string total charges to `NaN`, assigns `$0.00` to tenure=0 subscribers, and imputes any remaining missing total charges with the median ($1,397.48).
* **Leakage Prevention:** Transformers are fitted strictly on `X_train` during supervised training and serialized via Joblib. During inference, `CustomerPredictor` applies only `preprocessor.transform()`.

### Stage 2: Unsupervised Customer Segmentation
* **Feature Space:** Scaled 3D behavioral vector: `[tenure, MonthlyCharges, TotalCharges]`.
* **Clustering:** K-Means clustering ($K=4$, `init='k-means++'`, `n_init=10`, `random_state=42`).
* **Empirical Characterization:**
  - **Cluster 0:** New Budget Starters (Tenure: ~10.2 mos, Monthly: ~$31.78, Churn: 24.7%)
  - **Cluster 1:** High-Value Loyal Champions (Tenure: ~59.5 mos, Monthly: ~$93.31, Churn: 15.4%)
  - **Cluster 2:** New High-Spend / At-Risk (Tenure: ~15.4 mos, Monthly: ~$80.78, Churn: 48.2%)
  - **Cluster 3:** Established Budget Retained (Tenure: ~53.6 mos, Monthly: ~$34.91, Churn: 5.0%)

### Stage 3: Supervised Churn Modeling
* **Partitioning:** Stratified 80/20 train/test split (`random_state=42`).
* **Model:** `RandomForestClassifier` with hyperparameter optimization via `RandomizedSearchCV` (3-fold cross validation, 10 iterations, optimized on `roc_auc`).
* **Evaluation:** Evaluated on the held-out 20% test set (1,409 subscribers) yielding 80.06% Accuracy and 0.8430 ROC-AUC.

### Stage 4: Real-Time Serving Layer
* Pre-loads fitted pipelines into memory using Streamlit resource caching (`@st.cache_resource`).
* Guarantees that single customer inputs pass through the identical transformation steps as the training data, avoiding training-serving skew.
