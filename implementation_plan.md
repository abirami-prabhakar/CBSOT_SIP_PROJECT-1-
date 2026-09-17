# Implementation Plan: Portfolio-Ready Telco Customer Segmentation & Churn Prediction System

Transform this repository from a single-cell Colab scratchpad into a production-grade, modular, portfolio-ready Machine Learning system. The project will feature an authentic scikit-learn ML pipeline (K-Means for customer segmentation and hyperparameter-tuned Random Forest for churn prediction), an automated training and artifact persistence workflow, a modern multi-view Streamlit analytics dashboard, automated tests, and verified end-to-end execution.

## Proposed System Architecture

```text
               ┌──────────────────────────────────────────────┐
               │    Raw Telco Dataset (7,043 Subscribers)     │
               │        data/Telco-Customer-Churn.csv         │
               └──────────────────────┬───────────────────────┘
                                      │
                                      ▼
               ┌──────────────────────────────────────────────┐
               │   Data Cleaning & Schema Harmonization       │
               │   (TotalCharges imputation, type casting,    │
               │    whitespace stripping, ID isolation)       │
               └──────────────────────┬───────────────────────┘
                                      │
                     ┌────────────────┴────────────────┐
                     ▼                                 ▼
       ┌───────────────────────────┐     ┌───────────────────────────┐
       │   Unsupervised Branch     │     │    Supervised Branch      │
       │   (Customer Segmentation) │     │    (Churn Prediction)     │
       ├───────────────────────────┤     ├───────────────────────────┤
       │ • Feature Selection       │     │ • Stratified Train/Test   │
       │   (Tenure, Spend metrics) │     │   Split (80/20)           │
       │ • StandardScaler          │     │ • ColumnTransformer       │
       │ • Elbow Method (K=1..10)  │     │   (Num Imputer+Scaler,    │
       │ • K-Means (K=4)           │     │    Cat Imputer+OneHot)    │
       │ • Evidence-Based Personas │     │ • RandomForestClassifier  │
       │   & Cluster Statistics    │     │ • RandomizedSearchCV (CV) │
       └─────────────┬─────────────┘     └─────────────┬─────────────┘
                     │                                 │
                     └────────────────┬────────────────┘
                                      │
                                      ▼
               ┌──────────────────────────────────────────────┐
               │       Model Serialization via Joblib         │
               │   (models/*.joblib: Preprocessor, K-Means,   │
               │    Tuned RF, Test Metrics, Importances)      │
               └──────────────────────┬───────────────────────┘
                                      │
                     ┌────────────────┴────────────────┐
                     ▼                                 ▼
       ┌───────────────────────────┐     ┌───────────────────────────┐
       │    Streamlit Dashboard    │     │    Pytest Verification    │
       │ (5 Interactive Analytics  │     │ (Preprocessing, Pipeline, │
       │   Views + Live Inference) │     │  Inference, Schema tests) │
       └───────────────────────────┘     └───────────────────────────┘
```

---

## User Review Required

> [!IMPORTANT]
> **Dataset Storage & Packaging**: The repository currently only contains `CBSOT_SIP_PROJECT-1.ipynb` with a 100-row synthetic fallback snippet. We will automatically fetch and store the canonical 7,043-row IBM Telco Customer Churn dataset into `data/Telco-Customer-Churn.csv`. The data loader will support both canonical naming (`tenure`, `MonthlyCharges`, `TotalCharges`, `Churn`) and space-separated variants (`Tenure Months`, `Monthly Charges`, `Total Charges`, `Churn Label`).
>
> **Predictive Association vs. Causation**: In compliance with rigorous ML standards, all feature importance charts and prediction explanations will explicitly state that tree feature importances represent *predictive correlation / statistical association*, not *causal levers*.

---

## Proposed Changes

### 1. Project Organization & Dependencies

Refactor the flat directory into a standard, clean ML engineering repository structure:

```text
SIP/
├── app.py                     # Main Streamlit analytics application
├── train.py                   # End-to-end training and evaluation script
├── requirements.txt           # Production dependencies with pinned versions
├── architecture.md            # System architecture diagram and documentation
├── README.md                  # Comprehensive portfolio documentation
├── data/
│   └── Telco-Customer-Churn.csv # Real 7,043-row IBM Telco dataset
├── models/                    # Serialized joblib artifacts
│   ├── preprocessor.joblib
│   ├── churn_rf_model.joblib
│   ├── kmeans_model.joblib
│   ├── kmeans_features.joblib
│   ├── model_metrics.joblib
│   └── cluster_profiles.joblib
├── notebooks/
│   └── CBSOT_SIP_PROJECT-1.ipynb # Moved/preserved original reference notebook
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py # Cleaning, schema normalization & Scikit-learn pipelines
│   ├── segmentation.py       # K-Means clustering, Elbow method & cluster profiling
│   ├── churn_model.py        # Random Forest training & RandomizedSearchCV tuning
│   ├── evaluation.py         # Performance metrics, confusion matrix, ROC-AUC calculation
│   ├── prediction.py         # Real-time single & batch prediction engine
│   └── visualization.py      # Plotly & Matplotlib dashboard visualization helpers
└── tests/
    ├── __init__.py
    ├── test_preprocessing.py
    ├── test_segmentation.py
    ├── test_model.py
    └── test_prediction.py
```

#### [MODIFY] [requirements.txt](file:///e:/Desktop/Open%20source/SIP/requirements.txt)
- Ensure all required packages are present: `pandas>=2.0.0`, `numpy>=1.24.0`, `scikit-learn>=1.2.2`, `matplotlib>=3.7.0`, `seaborn>=0.12.0`, `streamlit>=1.28.0`, `plotly>=5.17.0`, `joblib>=1.3.0`, `pytest>=7.4.0`.

---

### 2. Core Machine Learning Modules (`src/`)

#### [NEW] [src/data_preprocessing.py](file:///e:/Desktop/Open%20source/SIP/src/data_preprocessing.py)
- Flexible schema normalization accommodating both column styles (`tenure` vs `Tenure Months`, `MonthlyCharges` vs `Monthly Charges`, etc.).
- Robust data cleaning: empty whitespace coercion in `TotalCharges` to `np.nan`, median imputation, stripping whitespace, and binary target encoding (`Churn`: Yes=1, No=0).
- Sklearn `ColumnTransformer` builder combining:
  - `num_pipeline`: `SimpleImputer(strategy='median')` + `StandardScaler()`
  - `cat_pipeline`: `SimpleImputer(strategy='most_frequent')` + `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`
- Guaranteed identical preprocessing during training and real-time prediction to eliminate training/serving skew.

#### [NEW] [src/segmentation.py](file:///e:/Desktop/Open%20source/SIP/src/segmentation.py)
- K-Means customer segmentation on customer behavioral and spending metrics (`tenure`, `MonthlyCharges`, `TotalCharges`).
- Dynamic Elbow Method computation across $K=1 \dots 10$ with inertia and percentage reduction tracking.
- Rigorous statistical profiling per cluster: cluster size, % distribution, mean/median tenure, mean/median monthly charges, mean/median total charges, and actual churn rate.
- Automated, evidence-backed cluster persona generator (e.g. "New High-Spend / High-Risk", "Long-Term Value / Loyal", "New Budget", "Mature Mid-Tier") strictly derived from empirical data medians.

#### [NEW] [src/churn_model.py](file:///e:/Desktop/Open%20source/SIP/src/churn_model.py)
- Stratified 80/20 train/test split with seed fixing (`random_state=42`).
- Fit `ColumnTransformer` strictly on `X_train` to prevent data leakage.
- `RandomForestClassifier` hyperparameter optimization via `RandomizedSearchCV` with 3-fold cross-validation:
  - Tuning grid: `n_estimators` [50, 100, 150, 200], `max_depth` [None, 10, 15, 20], `min_samples_split` [2, 5, 10], `min_samples_leaf` [1, 2, 4], `class_weight` ['balanced', None].
- Save best estimator and full pipeline.

#### [NEW] [src/evaluation.py](file:///e:/Desktop/Open%20source/SIP/src/evaluation.py)
- Compute genuine test metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC.
- Compute Confusion Matrix and ROC Curve (FPR, TPR, thresholds).
- Extract and map Random Forest Gini feature importances back to clean readable feature names (e.g., `Contract: Month-to-month`, `tenure`, `InternetService: Fiber optic`).

#### [NEW] [src/prediction.py](file:///e:/Desktop/Open%20source/SIP/src/prediction.py)
- `CustomerPredictor` class that loads serialized joblib models.
- Single customer prediction method: receives a raw feature dictionary/DataFrame, applies `preprocessor.transform()`, calls `rf_model.predict()` and `rf_model.predict_proba()`, and calls `kmeans_model.predict()`.
- Returns genuine churn prediction class (No Churn / Churn), churn probability score, confidence rating, assigned customer segment, and key feature values.

#### [NEW] [src/visualization.py](file:///e:/Desktop/Open%20source/SIP/src/visualization.py)
- Plotly-powered responsive, interactive visualizations:
  - Churn distribution donut chart.
  - Interactive scatter plot of Tenure vs. Monthly Charges colored by Cluster with hover details.
  - Elbow curve visualization with cluster selection marker.
  - Confusion matrix heatmap with annotated counts and normalized rates.
  - Interactive ROC Curve with AUC badge.
  - Top 10 predictive feature importance horizontal bar chart with causal disclaimer note.

---

### 3. Pipeline Runner & Streamlit Web Application

#### [NEW] [train.py](file:///e:/Desktop/Open%20source/SIP/train.py)
- End-to-end command-line training pipeline:
  1. Loads `data/Telco-Customer-Churn.csv`.
  2. Executes data sanitization and feature segregation.
  3. Prepares and fits Sklearn `ColumnTransformer`.
  4. Computes Elbow curve and trains K-Means ($K=4$).
  5. Computes cluster empirical profiles and persona descriptions.
  6. Executes `RandomizedSearchCV` for `RandomForestClassifier`.
  7. Evaluates on held-out test set and logs genuine classification metrics.
  8. Serializes all artifacts into `models/`.

#### [NEW] [app.py](file:///e:/Desktop/Open%20source/SIP/app.py)
- A multi-page interactive Streamlit dashboard featuring:
  - **1. Dashboard**: High-level KPIs (Total Customers, Overall Churn Rate, Test Accuracy, Test ROC-AUC, Active Clusters), churn ratio breakdown, model quick summary, and segment summary.
  - **2. Customer Segments**: Elbow Method curve, interactive 2D/3D cluster scatter plots, statistical profiles table (average tenure, charges, churn risk per segment), and evidence-supported persona descriptions with interactive filtering.
  - **3. Churn Prediction**: Clean customer input form (Demographics, Phone & Internet Services, Contract & Billing, Tenure & Charges) with sensible defaults; real-time execution yielding predicted churn class, probability gauge, predicted customer segment, and contributing predictive factors.
  - **4. Model Performance**: Real test set metrics cards (Accuracy, Precision, Recall, F1, ROC-AUC), interactive Confusion Matrix, interactive ROC Curve, ranked predictive Feature Importance chart, and beginner-friendly metric explanations.
  - **5. Data Explorer**: Interactive dataset preview, search/filter, schema & data types summary, missing-value audit, interactive numerical distributions (histograms & boxplots), categorical churn rate cross-tabulations, and custom CSV uploader for batch analysis.

---

### 4. Automated Testing Suite

#### [NEW] [tests/test_preprocessing.py](file:///e:/Desktop/Open%20source/SIP/tests/test_preprocessing.py)
- Test handling of whitespace in `TotalCharges`.
- Test schema normalization for both space and camelCase column headers.
- Test `ColumnTransformer` output dimensions and zero NaN propagation.

#### [NEW] [tests/test_segmentation.py](file:///e:/Desktop/Open%20source/SIP/tests/test_segmentation.py)
- Test K-Means cluster assignment integrity.
- Test that cluster statistical profiles contain valid numeric aggregates.

#### [NEW] [tests/test_model.py](file:///e:/Desktop/Open%20source/SIP/tests/test_model.py)
- Test model artifact loading from `models/`.
- Test that model evaluation metrics match real non-trivial bounds (e.g. ROC-AUC > 0.75, Accuracy > 0.70).

#### [NEW] [tests/test_prediction.py](file:///e:/Desktop/Open%20source/SIP/tests/test_prediction.py)
- Test single customer inference end-to-end.
- Test handling of edge-case inputs (e.g. 0 tenure, extreme charges, unknown categorical values).

---

### 5. Documentation & Repository Architecture

#### [MODIFY] [architecture.md](file:///e:/Desktop/Open%20source/SIP/architecture.md)
- Complete ASCII architecture and detailed workflow diagram covering the end-to-end supervised and unsupervised branches, artifact storage, and Streamlit serving layer.

#### [MODIFY] [README.md](file:///e:/Desktop/Open%20source/SIP/README.md)
- Professional portfolio README:
  - Executive summary and business problem context.
  - Architectural overview.
  - Dataset source, schema, and preprocessing strategy.
  - K-Means customer segmentation & Elbow method results.
  - Evidence-backed cluster personas with empirical stats.
  - Tuned Random Forest model with RandomizedSearchCV details.
  - Actual evaluated test set metrics (no placeholders or made-up numbers).
  - Feature importance analysis with explicit distinction between predictive association and causation.
  - Step-by-step setup, training (`python train.py`), and app launch (`streamlit run app.py`).
  - Limitations and future engineering enhancements.

---

## Verification Plan

### Automated Tests
1. **Install Dependencies**:
   ```bash
   pip install seaborn plotly
   ```
2. **Execute Full Model Training Pipeline**:
   ```bash
   python train.py
   ```
   - Verify: Dataset loads, K-Means trains, RandomizedSearchCV tunes Random Forest, genuine test metrics are printed, and `.joblib` files are written to `models/`.
3. **Execute Pytest Suite**:
   ```bash
   pytest -v tests/
   ```
   - Verify: All tests pass for preprocessing, segmentation, model artifacts, and inference.

### Manual & Interactive Verification
4. **Launch Streamlit Web App**:
   - Run `streamlit run app.py` (or verify via headless smoke test).
   - Test navigation across all 5 pages: Dashboard, Customer Segments, Churn Prediction, Model Performance, Data Explorer.
   - Enter a sample customer profile (e.g. Month-to-month, Fiber optic, Electronic check, Tenure 2 months) and verify the actual live churn prediction and segment assignment.
   - Verify responsive layout, empty states, and no hardcoded values.
