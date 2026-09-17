"""
Visualization Helper Module using Plotly.
Produces clean, interactive, and responsive visualizations for the Streamlit dashboard.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# Theme palette
PALETTE = {
    'primary': '#2b5c8f',
    'secondary': '#4a90e2',
    'accent': '#e74c3c',
    'success': '#2ecc71',
    'warning': '#f39c12',
    'neutral': '#34495e',
    'background': '#f8f9fa',
    'grid': '#e9ecef',
    'clusters': ['#2b5c8f', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
}


def plot_churn_donut(df: pd.DataFrame, churn_col: str = 'Churn') -> go.Figure:
    """
    Plots an interactive donut chart of the overall churn ratio.
    """
    counts = df[churn_col].value_counts()
    labels = ['No Churn (Retained)' if str(k) in ['0', 'No', 'no', 'False'] else 'Churned' for k in counts.index]
    values = counts.values

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(colors=[PALETTE['secondary'], PALETTE['accent']]),
        textinfo='label+percent',
        hoverinfo='label+value+percent'
    )])

    fig.update_layout(
        title_text="Overall Subscriber Churn Ratio",
        title_font_size=16,
        margin=dict(t=40, b=20, l=20, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        height=320
    )
    return fig


def plot_cluster_scatter(
    df: pd.DataFrame,
    cluster_col: str = 'Customer_Segment',
    sample_n: int = 1500
) -> go.Figure:
    """
    Generates an interactive scatter plot of Tenure vs Monthly Charges colored by customer segment.
    """
    plot_df = df.copy()
    if len(plot_df) > sample_n:
        plot_df = plot_df.sample(sample_n, random_state=42)

    plot_df['Cluster_Str'] = "Cluster " + plot_df[cluster_col].astype(str)

    fig = px.scatter(
        plot_df,
        x='tenure',
        y='MonthlyCharges',
        color='Cluster_Str',
        size='TotalCharges',
        size_max=14,
        hover_data=['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract'],
        color_discrete_sequence=PALETTE['clusters'],
        labels={
            'tenure': 'Tenure (Months)',
            'MonthlyCharges': 'Monthly Charges ($)',
            'TotalCharges': 'Total Lifetime Charges ($)',
            'Cluster_Str': 'Segment'
        }
    )

    fig.update_layout(
        title="Customer Behavioral Segments (Spend vs. Tenure)",
        title_font_size=16,
        xaxis=dict(title="Tenure (Months)", gridcolor=PALETTE['grid']),
        yaxis=dict(title="Monthly Charges ($)", gridcolor=PALETTE['grid']),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=50, b=30, l=40, r=20),
        height=450
    )
    return fig


def plot_elbow_curve(elbow_data: Dict[str, Any], selected_k: int = 4) -> go.Figure:
    """
    Plots the Elbow curve with WCSS against k and marks the selected optimal k.
    """
    k_vals = elbow_data['k_values']
    inertias = elbow_data['inertias']

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=k_vals,
        y=inertias,
        mode='lines+markers',
        line=dict(color=PALETTE['primary'], width=3),
        marker=dict(size=8, color=PALETTE['secondary']),
        name='Inertia (WCSS)'
    ))

    # Mark selected k
    if selected_k in k_vals:
        idx = k_vals.index(selected_k)
        fig.add_trace(go.Scatter(
            x=[selected_k],
            y=[inertias[idx]],
            mode='markers',
            marker=dict(size=14, color=PALETTE['accent'], symbol='star'),
            name=f'Selected (K={selected_k})'
        ))

    fig.update_layout(
        title="Elbow Method for Optimal K",
        title_font_size=16,
        xaxis=dict(title="Number of Clusters (K)", tickmode='linear', dtick=1, gridcolor=PALETTE['grid']),
        yaxis=dict(title="Within-Cluster Sum of Squares (Inertia)", gridcolor=PALETTE['grid']),
        margin=dict(t=40, b=30, l=40, r=20),
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def plot_confusion_matrix_heatmap(cm: List[List[int]]) -> go.Figure:
    """
    Plots an annotated confusion matrix heatmap.
    """
    cm_arr = np.array(cm)
    total = cm_arr.sum()
    pct_arr = (cm_arr / total) * 100.0

    annotations = []
    labels = [['True Negative (TN)', 'False Positive (FP)'],
              ['False Negative (FN)', 'True Positive (TP)']]

    for i in range(2):
        for j in range(2):
            val = cm_arr[i][j]
            pct = pct_arr[i][j]
            lbl = labels[i][j]
            text = f"<b>{val:,}</b><br>({pct:.1f}%)<br><span style='font-size:10px; color:#555;'>{lbl}</span>"
            annotations.append(dict(
                x=j, y=i,
                text=text,
                showarrow=False,
                font=dict(color="white" if val > total * 0.3 else "black", size=13)
            ))

    fig = go.Figure(data=go.Heatmap(
        z=cm_arr,
        x=['Predicted No Churn', 'Predicted Churn'],
        y=['Actual No Churn', 'Actual Churn'],
        colorscale='Blues',
        showscale=False
    ))

    fig.update_layout(
        title="Confusion Matrix (Held-out Test Set)",
        title_font_size=16,
        annotations=annotations,
        xaxis=dict(side='bottom'),
        yaxis=dict(autorange='reversed'),
        margin=dict(t=40, b=30, l=40, r=20),
        height=350
    )
    return fig


def plot_roc_curve_chart(roc_curve_data: Dict[str, Any], auc_score: float) -> go.Figure:
    """
    Plots the ROC curve with the random baseline diagonal.
    """
    fpr = roc_curve_data['fpr']
    tpr = roc_curve_data['tpr']

    fig = go.Figure()

    # ROC curve
    fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode='lines',
        line=dict(color=PALETTE['accent'], width=3),
        name=f"ROC Curve (AUC = {auc_score:.3f})"
    ))

    # Diagonal baseline
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        line=dict(color=PALETTE['neutral'], width=2, dash='dash'),
        name="Random Classifier (AUC = 0.500)"
    ))

    fig.update_layout(
        title=f"Receiver Operating Characteristic (ROC-AUC: {auc_score:.3f})",
        title_font_size=16,
        xaxis=dict(title="False Positive Rate (1 - Specificity)", range=[0, 1], gridcolor=PALETTE['grid']),
        yaxis=dict(title="True Positive Rate (Recall / Sensitivity)", range=[0, 1.02], gridcolor=PALETTE['grid']),
        margin=dict(t=40, b=30, l=40, r=20),
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
    )
    return fig


def plot_feature_importance_chart(df_imp: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """
    Plots a ranked horizontal bar chart of Random Forest feature importances
    with an explicit predictive correlation vs causation disclaimer.
    """
    top_df = df_imp.head(top_n).iloc[::-1]  # reverse for top-to-bottom display

    fig = go.Figure(go.Bar(
        x=top_df['importance'],
        y=top_df['clean_feature'],
        orientation='h',
        marker=dict(
            color=top_df['importance'],
            colorscale='Tealgrn',
            showscale=False
        ),
        text=[f"{v:.3f}" for v in top_df['importance']],
        textposition='outside'
    ))

    fig.update_layout(
        title="Top Drivers of Churn (Random Forest Predictive Importance)",
        title_font_size=16,
        xaxis=dict(title="Relative Gini Importance Score", gridcolor=PALETTE['grid']),
        yaxis=dict(title=""),
        margin=dict(t=40, b=40, l=120, r=40),
        height=400,
        annotations=[
            dict(
                x=0.5, y=-0.22,
                xref='paper', yref='paper',
                text="<b>Note:</b> Feature importance reflects statistical predictive association, not causal impact.",
                showarrow=False,
                font=dict(size=11, color="#666")
            )
        ]
    )
    return fig


def plot_numeric_distribution(df: pd.DataFrame, col: str, target_col: str = 'Churn') -> go.Figure:
    """
    Plots distribution histograms comparing Churned vs Retained subscribers.
    """
    fig = px.histogram(
        df,
        x=col,
        color=target_col,
        barmode='overlay',
        marginal='box',
        color_discrete_map={'No': PALETTE['secondary'], 'Yes': PALETTE['accent'], 0: PALETTE['secondary'], 1: PALETTE['accent']},
        labels={col: col, target_col: 'Churn'}
    )

    fig.update_layout(
        title=f"Distribution of {col} by Churn Status",
        title_font_size=15,
        margin=dict(t=40, b=30, l=40, r=20),
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def plot_categorical_churn_rate(df: pd.DataFrame, col: str, target_col: str = 'Churn') -> go.Figure:
    """
    Plots churn rate percentages across categories of a discrete column.
    """
    temp = df.copy()
    temp['Churn_Num'] = temp[target_col].map({'Yes': 1, 'No': 0, 1: 1, 0: 0}).fillna(0)
    summary = temp.groupby(col)['Churn_Num'].agg(['count', 'mean']).reset_index()
    summary['Churn_Rate_%'] = summary['mean'] * 100.0

    fig = px.bar(
        summary,
        x=col,
        y='Churn_Rate_%',
        text=summary['Churn_Rate_%'].apply(lambda x: f"{x:.1f}%"),
        color='Churn_Rate_%',
        color_continuous_scale='Reds',
        labels={'Churn_Rate_%': 'Churn Rate (%)', col: col}
    )

    fig.update_layout(
        title=f"Churn Rate by {col}",
        title_font_size=15,
        yaxis=dict(title="Churn Rate (%)", range=[0, max(summary['Churn_Rate_%'].max() * 1.15, 20)]),
        margin=dict(t=40, b=30, l=40, r=20),
        height=320,
        coloraxis_showscale=False
    )
    return fig
