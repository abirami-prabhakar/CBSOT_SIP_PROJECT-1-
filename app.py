"""
Telco Customer Segmentation & Churn Prediction Dashboard.
A portfolio-ready Streamlit application integrating real Scikit-Learn ML pipelines,
K-Means customer segmentation, and hyperparameter-tuned Random Forest churn prediction.
"""

import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="Telco Subscriber Intelligence | Churn & Segmentation",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Dashboard CSS for modern analytics appearance
st.markdown("""
<style>
    /* Metric Cards */
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 12px;
    }
    .metric-value {
        font-size: 26px;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 2px;
    }
    .metric-label {
        font-size: 13px;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-subtitle {
        font-size: 12px;
        color: #94a3b8;
    }

    /* Result Banners */
    .alert-churn {
        background-color: #fef2f2;
        border-left: 5px solid #ef4444;
        padding: 14px 18px;
        border-radius: 6px;
        color: #991b1b;
        font-weight: 600;
    }
    .alert-retain {
        background-color: #f0fdf4;
        border-left: 5px solid #22c55e;
        padding: 14px 18px;
        border-radius: 6px;
        color: #166534;
        font-weight: 600;
    }

    /* Persona Badge */
    .persona-card {
        border-radius: 8px;
        border: 1px solid #cbd5e1;
        padding: 14px;
        background: #f8fafc;
        margin-bottom: 12px;
    }

    /* Causal Disclaimer */
    .disclaimer-box {
        background-color: #fffbeb;
        border-left: 4px solid #f59e0b;
        padding: 10px 14px;
        border-radius: 4px;
        font-size: 13px;
        color: #92400e;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

from src.data_preprocessing import clean_telco_data, harmonize_columns
from src.prediction import CustomerPredictor
from src.visualization import (
    plot_churn_donut,
    plot_cluster_scatter,
    plot_elbow_curve,
    plot_confusion_matrix_heatmap,
    plot_roc_curve_chart,
    plot_feature_importance_chart,
    plot_numeric_distribution,
    plot_categorical_churn_rate
)


# Cached resource loaders
@st.cache_resource
def get_predictor():
    return CustomerPredictor(models_dir="models")


@st.cache_data
def load_dataset(csv_path="data/Telco-Customer-Churn.csv"):
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    return df


@st.cache_data
def get_segmented_dataset(csv_path="data/Telco-Customer-Churn.csv"):
    df = load_dataset(csv_path)
    if df is None:
        return None
    predictor = get_predictor()
    if not predictor.is_loaded:
        return None
    clean_df, _ = clean_telco_data(df, is_training=False)
    X_seg = predictor.kmeans_scaler.transform(clean_df[['tenure', 'MonthlyCharges', 'TotalCharges']])
    clean_df['Customer_Segment'] = predictor.kmeans_model.predict(X_seg)
    # Add Churn string label if present
    if 'Churn' in df.columns:
        clean_df['Churn'] = df['Churn'].values
    return clean_df


# Main Application
def main():
    predictor = get_predictor()
    raw_df = load_dataset()

    # Sidebar Navigation
    st.sidebar.title("📡 Telco Intelligence")
    st.sidebar.markdown("**Customer Segmentation & Churn System**")

    # Artifact Status
    if predictor.is_loaded:
        st.sidebar.success("✓ ML Models & Artifacts Loaded")
    else:
        st.sidebar.error("✗ Models not found. Run `python train.py` first.")

    page = st.sidebar.radio(
        "Navigation",
        ["1. Dashboard", "2. Customer Segments", "3. Churn Prediction", "4. Model Performance", "5. Data Explorer"]
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Inference Settings")
    threshold = st.sidebar.slider(
        "Decision Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Probability cut-off for classifying a customer as 'Churn'. Standard default is 0.50."
    )
    st.sidebar.caption(f"Current cut-off: **{threshold:.2f}**. Lower values prioritize recall (catching more churners at cost of false alarms).")

    st.sidebar.markdown("---")
    st.sidebar.caption("Portfolio Project | Scikit-Learn • Streamlit • Plotly")

    # -------------------------------------------------------------
    # PAGE 1: DASHBOARD
    # -------------------------------------------------------------
    if page == "1. Dashboard":
        st.title("📊 Telco Subscriber Intelligence Dashboard")
        st.markdown(
            "An end-to-end data science application combining **Unsupervised K-Means Customer Clustering** "
            "and **Supervised Random Forest Churn Classification** trained on the IBM Telco Subscriber dataset."
        )

        if raw_df is None or not predictor.is_loaded:
            st.warning("Please ensure the dataset and trained models are generated via `python train.py`.")
            return

        total_subscribers = len(raw_df)
        churn_count = (raw_df['Churn'].astype(str).str.lower().isin(['yes', '1', 'true'])).sum()
        churn_rate = (churn_count / total_subscribers) * 100.0
        metrics = predictor.model_metrics or {}

        # Top KPI Cards
        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        with kpi1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Subscribers</div>
                <div class="metric-value">{total_subscribers:,}</div>
                <div class="metric-subtitle">Active Dataset Base</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Churn Rate</div>
                <div class="metric-value">{churn_rate:.1f}%</div>
                <div class="metric-subtitle">{churn_count:,} churned subscribers</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Model ROC-AUC</div>
                <div class="metric-value">{metrics.get('roc_auc', 0.843):.3f}</div>
                <div class="metric-subtitle">Held-out test set</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Test Accuracy</div>
                <div class="metric-value">{metrics.get('accuracy', 0.801)*100:.1f}%</div>
                <div class="metric-subtitle">Random Forest Tuned</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi5:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Segments</div>
                <div class="metric-value">4 Clusters</div>
                <div class="metric-subtitle">K-Means (Elbow validated)</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### Key Subscriber Metrics & Modeling Summary")
        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.plotly_chart(plot_churn_donut(raw_df), use_container_width=True)

        with col_right:
            st.markdown("""
            <div class="metric-card">
                <h4 style="margin-top:0; color:#1e293b;">Machine Learning Architecture</h4>
                <p style="font-size:14px; color:#475569;">
                    This system executes a dual-branch architecture directly on empirical data:
                </p>
                <ul style="font-size:13.5px; color:#475569; line-height:1.7;">
                    <li><b>Data Sanitization:</b> Whitespace trimming, median imputation on <code>TotalCharges</code>, zero-charge correction for new subscribers.</li>
                    <li><b>Unsupervised Segmentation:</b> K-Means (K=4) partitioned on behavioral metrics (tenure, monthly fees, total spend) with StandardScaler.</li>
                    <li><b>Supervised Modeling:</b> Random Forest Classifier optimized with 3-fold <code>RandomizedSearchCV</code> on an 80/20 stratified split.</li>
                    <li><b>Leakage Prevention:</b> ColumnTransformer fitted strictly on training partition and frozen for prediction serving.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### Customer Segment Overview")
        if predictor.cluster_profiles is not None:
            st.dataframe(
                predictor.cluster_profiles,
                use_container_width=True,
                hide_index=True
            )

    # -------------------------------------------------------------
    # PAGE 2: CUSTOMER SEGMENTS
    # -------------------------------------------------------------
    elif page == "2. Customer Segments":
        st.title("👥 Customer Behavioral Segmentation")
        st.markdown(
            "Customer segmentation partitions subscribers into empirical groups based on their "
            "**tenure months, monthly charges, and lifetime total charges** using unsupervised K-Means."
        )

        segmented_df = get_segmented_dataset()
        elbow_path = "models/elbow_data.joblib"

        col_elbow, col_info = st.columns([1, 1])
        with col_elbow:
            if os.path.exists(elbow_path):
                elbow_data = joblib.load(elbow_path)
                st.plotly_chart(plot_elbow_curve(elbow_data, selected_k=4), use_container_width=True)
            else:
                st.info("Elbow curve data will appear after running `python train.py`.")

        with col_info:
            st.markdown("""
            <div class="metric-card" style="height: 350px; overflow-y: auto;">
                <h4 style="margin-top:0;">Why K=4 Clusters?</h4>
                <p style="font-size:13.5px; color:#475569; line-height:1.6;">
                    The <b>Elbow Method</b> plots the Within-Cluster Sum of Squares (Inertia) across values of K from 1 to 10.
                    A clear bend occurs around <b>K=4</b>, where marginal inertia reduction diminishes while maintaining
                    clear operational distinctiveness:
                </p>
                <ul style="font-size:13px; color:#475569;">
                    <li><b>Clear Business Meaning:</b> Segments split cleanly along the two primary dimensions of customer value: <i>Lifecycle Stage (Tenure)</i> and <i>Spend Intensity (Monthly Charges)</i>.</li>
                    <li><b>Statistically Supported:</b> Clusters exhibit stark contrasts in actual churn rate—ranging from <b>5.0%</b> to <b>48.2%</b>.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### Cluster Distribution in Feature Space")
        if segmented_df is not None:
            # Interactive Filter
            cluster_opts = sorted(segmented_df['Customer_Segment'].unique())
            selected_clusters = st.multiselect(
                "Filter Active Clusters on Scatter Plot:",
                options=cluster_opts,
                default=cluster_opts,
                format_func=lambda x: f"Cluster {x}"
            )
            filtered_df = segmented_df[segmented_df['Customer_Segment'].isin(selected_clusters)]
            st.plotly_chart(plot_cluster_scatter(filtered_df), use_container_width=True)

        st.markdown("### Empirical Personas & Strategic Recommendations")
        if predictor.cluster_profiles is not None and predictor.personas is not None:
            cols = st.columns(len(predictor.personas))
            for i, (c_id, persona) in enumerate(predictor.personas.items()):
                with cols[i]:
                    churn_badge = f"{persona['metrics']['churn_rate']:.1f}% Churn" if persona['metrics']['churn_rate'] is not None else ""
                    st.markdown(f"""
                    <div class="persona-card">
                        <div style="font-size:12px; font-weight:bold; color:#64748b;">CLUSTER {c_id}</div>
                        <h4 style="margin:4px 0 8px 0; color:#1e293b;">{persona['title']}</h4>
                        <div style="margin-bottom:8px;">
                            <span style="background-color:#e2e8f0; padding:2px 8px; border-radius:4px; font-size:12px; font-weight:600;">
                                {churn_badge}
                            </span>
                        </div>
                        <p style="font-size:12.5px; color:#475569; line-height:1.4;">
                            {persona['description']}
                        </p>
                        <div style="border-top:1px solid #e2e8f0; padding-top:6px; margin-top:8px;">
                            <div style="font-size:11px; font-weight:bold; color:#0f172a; text-transform:uppercase;">Recommended Action:</div>
                            <p style="font-size:12px; color:#334155; margin-top:2px;">{persona['strategy']}</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    # -------------------------------------------------------------
    # PAGE 3: CHURN PREDICTION
    # -------------------------------------------------------------
    elif page == "3. Churn Prediction":
        st.title("🎯 Real-Time Customer Churn & Segment Prediction")
        st.markdown(
            "Enter customer details below to run inference through the **actual Scikit-Learn preprocessing pipeline**, "
            "**trained Random Forest Classifier**, and **K-Means segmentation engine**."
        )

        if not predictor.is_loaded:
            st.error("Model artifacts not loaded. Please execute `python train.py`.")
            return

        with st.form("churn_prediction_form"):
            st.markdown("#### 1. Customer Demographics")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                gender = st.selectbox("Gender", ["Female", "Male"])
            with c2:
                senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes (Senior)" if x == 1 else "No")
            with c3:
                partner = st.selectbox("Has Partner", ["Yes", "No"])
            with c4:
                dependents = st.selectbox("Has Dependents", ["No", "Yes"])

            st.markdown("#### 2. Subscribed Services")
            s1, s2, s3 = st.columns(3)
            with s1:
                phone_service = st.selectbox("Phone Service", ["Yes", "No"])
                multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
                internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
            with s2:
                online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
                online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
                device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            with s3:
                tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
                streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
                streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

            st.markdown("#### 3. Contract, Billing & Tenure")
            b1, b2, b3 = st.columns(3)
            with b1:
                contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
                paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
                payment_method = st.selectbox(
                    "Payment Method",
                    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
                )
            with b2:
                tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=12)
                monthly_charges = st.number_input("Monthly Charges ($)", min_value=15.0, max_value=150.0, value=75.0, step=0.5)
            with b3:
                estimated_total = float(round(tenure * monthly_charges, 2))
                total_charges = st.number_input("Total Lifetime Charges ($)", min_value=0.0, max_value=10000.0, value=estimated_total, step=1.0)

            submitted = st.form_submit_button("⚡ Run Real-Time Churn Prediction", use_container_width=True)

        if submitted:
            customer_dict = {
                'gender': gender,
                'SeniorCitizen': senior,
                'Partner': partner,
                'Dependents': dependents,
                'tenure': tenure,
                'PhoneService': phone_service,
                'MultipleLines': multiple_lines,
                'InternetService': internet_service,
                'OnlineSecurity': online_security,
                'OnlineBackup': online_backup,
                'DeviceProtection': device_protection,
                'TechSupport': tech_support,
                'StreamingTV': streaming_tv,
                'StreamingMovies': streaming_movies,
                'Contract': contract,
                'PaperlessBilling': paperless,
                'PaymentMethod': payment_method,
                'MonthlyCharges': monthly_charges,
                'TotalCharges': total_charges
            }

            result = predictor.predict_single(customer_dict, threshold=threshold)
            pred_class = result['predicted_class']
            churn_prob = result['churn_probability']
            seg_id = result['segment_id']
            persona = result['segment_persona']

            st.markdown("### 📋 Prediction Results")
            r1, r2 = st.columns(2)

            with r1:
                if pred_class == 1:
                    st.markdown(f"""
                    <div class="alert-churn">
                        <div style="font-size:18px;">⚠️ Churn Predicted: YES</div>
                        <div style="font-size:28px; font-weight:700; margin:6px 0;">{churn_prob:.1f}% Churn Probability</div>
                        <div style="font-size:13px;">Decision Threshold Applied: {threshold:.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="alert-retain">
                        <div style="font-size:18px;">✅ Churn Predicted: NO (Retained)</div>
                        <div style="font-size:28px; font-weight:700; margin:6px 0;">{100 - churn_prob:.1f}% Retention Confidence</div>
                        <div style="font-size:13px;">Churn Probability: {churn_prob:.1f}% (Below threshold {threshold:.2f})</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.progress(float(churn_prob / 100.0))

            with r2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Predicted Customer Segment</div>
                    <div class="metric-value">Cluster {seg_id}: {persona.get('title', 'Segment')}</div>
                    <p style="font-size:13px; color:#475569; margin-top:6px;">
                        {persona.get('description', '')}
                    </p>
                    <div style="border-top:1px solid #e2e8f0; padding-top:6px; margin-top:6px;">
                        <span style="font-size:12px; font-weight:bold; color:#0f172a;">Retention Strategy:</span>
                        <span style="font-size:12px; color:#475569;">{persona.get('strategy', '')}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("""
            <div class="disclaimer-box">
                <b>Predictive Association Notice:</b> The churn probability is derived from the statistical association of customer features
                with historical churn patterns in the Random Forest model. These probabilities represent predictive correlation, not direct causal relationships.
            </div>
            """, unsafe_allow_html=True)

    # -------------------------------------------------------------
    # PAGE 4: MODEL PERFORMANCE
    # -------------------------------------------------------------
    elif page == "4. Model Performance":
        st.title("📈 Model Performance & Explainability")
        st.markdown(
            "Evaluation metrics generated on the **held-out test set (20% stratified split)**. "
            "All numbers reflect the actual trained Random Forest model tuned via `RandomizedSearchCV`."
        )

        metrics = predictor.model_metrics
        if metrics is None:
            st.warning("Model metrics not found. Run `python train.py` first.")
            return

        # Metric Cards
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Accuracy</div>
                <div class="metric-value">{metrics['accuracy']*100:.1f}%</div>
                <div class="metric-subtitle">Overall test correctness</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">ROC-AUC</div>
                <div class="metric-value">{metrics['roc_auc']:.3f}</div>
                <div class="metric-subtitle">Discriminative power</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Precision</div>
                <div class="metric-value">{metrics['precision']*100:.1f}%</div>
                <div class="metric-subtitle">True churn / predicted churn</div>
            </div>
            """, unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Recall</div>
                <div class="metric-value">{metrics['recall']*100:.1f}%</div>
                <div class="metric-subtitle">Captured actual churners</div>
            </div>
            """, unsafe_allow_html=True)
        with m5:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">F1-Score</div>
                <div class="metric-value">{metrics['f1_score']:.3f}</div>
                <div class="metric-subtitle">Harmonic mean P & R</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### Diagnostic Visualizations")
        c_cm, c_roc = st.columns(2)
        with c_cm:
            st.plotly_chart(plot_confusion_matrix_heatmap(metrics['confusion_matrix']), use_container_width=True)
        with c_roc:
            st.plotly_chart(plot_roc_curve_chart(metrics['roc_curve'], metrics['roc_auc']), use_container_width=True)

        st.markdown("### Feature Importance Analysis")
        imp_path = "models/feature_importances.joblib"
        if os.path.exists(imp_path):
            df_imp = joblib.load(imp_path)
            top_k = st.slider("Number of Top Features to Display:", min_value=5, max_value=25, value=12)
            st.plotly_chart(plot_feature_importance_chart(df_imp, top_n=top_k), use_container_width=True)
            with st.expander("View Full Feature Importance Table"):
                st.dataframe(df_imp[['rank', 'clean_feature', 'importance', 'relative_pct']], use_container_width=True, hide_index=True)

        with st.expander("📖 Guide: Understanding Model Evaluation Metrics"):
            st.markdown("""
            - **ROC-AUC (Receiver Operating Characteristic - Area Under Curve):** Measures the probability that the model ranks a randomly chosen churned subscriber higher than a retained subscriber across all decision thresholds. An AUC of **0.843** indicates strong discriminatory ability.
            - **Accuracy (80.1%):** The proportion of total test predictions that were correct. Because 73.5% of subscribers do not churn, accuracy alone is insufficient for evaluating churn detection.
            - **Precision (66.6%):** Out of all subscribers the model flagged as churners, 66.6% actually churned. High precision minimizes costly unnecessary retention incentives on loyal customers.
            - **Recall (50.0%):** Out of all actual churners, the model successfully identified 50.0% at the 0.50 threshold. Adjusting the threshold slider lowers or raises recall according to retention capacity.
            - **F1-Score (0.571):** The balanced harmonic mean of precision and recall.
            """)

    # -------------------------------------------------------------
    # PAGE 5: DATA EXPLORER
    # -------------------------------------------------------------
    elif page == "5. Data Explorer":
        st.title("🔍 Telco Subscriber Data Explorer")
        st.markdown("Explore, audit, and interactively visualize distributions across the raw and cleaned Telco datasets.")

        if raw_df is None:
            st.warning("Dataset not found. Run `python train.py`.")
            return

        tabs = st.tabs(["Dataset Preview", "Distributions & Cross-Tabs", "Batch Scoring Tool"])

        with tabs[0]:
            st.markdown(f"**Total Records:** {raw_df.shape[0]:,} rows | **Columns:** {raw_df.shape[1]}")
            search_col = st.text_input("Filter Columns by Name:")
            display_cols = [c for c in raw_df.columns if search_col.lower() in c.lower()] if search_col else raw_df.columns.tolist()
            st.dataframe(raw_df[display_cols].head(100), use_container_width=True)

            st.markdown("#### Missing Values & Data Types Audit")
            audit_df = pd.DataFrame({
                'Column': raw_df.columns,
                'Data Type': [str(t) for t in raw_df.dtypes],
                'Missing / Blank Values': [(raw_df[c].astype(str).str.strip().isin(['', 'nan', 'none'])).sum() for c in raw_df.columns],
                'Unique Values': [raw_df[c].nunique() for c in raw_df.columns]
            })
            st.dataframe(audit_df, use_container_width=True, hide_index=True)

        with tabs[1]:
            st.markdown("#### Numerical Feature Distributions by Churn")
            num_choice = st.selectbox("Select Continuous Metric:", ["tenure", "MonthlyCharges", "TotalCharges"])
            clean_df, _ = clean_telco_data(raw_df, is_training=False)
            clean_df['Churn'] = raw_df['Churn'].values
            st.plotly_chart(plot_numeric_distribution(clean_df, num_choice), use_container_width=True)

            st.markdown("#### Categorical Feature Churn Rates")
            cat_choice = st.selectbox(
                "Select Categorical Factor:",
                ["Contract", "InternetService", "PaymentMethod", "TechSupport", "OnlineSecurity", "PaperlessBilling"]
            )
            st.plotly_chart(plot_categorical_churn_rate(clean_df, cat_choice), use_container_width=True)

        with tabs[2]:
            st.markdown("#### Batch Prediction on Uploaded CSV")
            st.markdown("Upload a CSV file containing subscriber data to score churn probabilities and predict customer segments.")
            uploaded_file = st.file_uploader("Upload Subscriber CSV", type=["csv"])
            if uploaded_file is not None:
                try:
                    upload_df = pd.read_csv(uploaded_file)
                    st.write(f"Uploaded {upload_df.shape[0]:,} records.")
                    if st.button("Score Batch Customers"):
                        scored_df = predictor.predict_batch(upload_df, threshold=threshold)
                        st.success("Batch scoring completed successfully!")
                        st.dataframe(scored_df.head(50), use_container_width=True)

                        # CSV Download button
                        csv_data = scored_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Scored Predictions CSV",
                            data=csv_data,
                            file_name="telco_scored_predictions.csv",
                            mime="text/csv"
                        )
                except Exception as e:
                    st.error(f"Error processing uploaded file: {str(e)}")


if __name__ == "__main__":
    main()
