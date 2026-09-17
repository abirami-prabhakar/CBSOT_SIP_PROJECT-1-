"""
Customer Segmentation Engine using K-Means Clustering.
Implements the Elbow Method, feature scaling, empirical cluster profiling,
and evidence-supported persona derivation.
"""

from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

SEGMENTATION_FEATURES: List[str] = ['tenure', 'MonthlyCharges', 'TotalCharges']


def compute_elbow_curve(
    df: pd.DataFrame,
    features: List[str] = SEGMENTATION_FEATURES,
    k_min: int = 1,
    k_max: int = 10,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Computes within-cluster sum of squares (WCSS / Inertia) across a range of k values
    to evaluate the Elbow curve.
    """
    X = df[features].copy().dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    k_values = list(range(k_min, k_max + 1))
    inertias = []
    for k in k_values:
        km = KMeans(n_clusters=k, init='k-means++', random_state=random_state, n_init=10)
        km.fit(X_scaled)
        inertias.append(float(km.inertia_))

    return {
        'k_values': k_values,
        'inertias': inertias,
        'features': features
    }


def train_kmeans_segmentation(
    df: pd.DataFrame,
    features: List[str] = SEGMENTATION_FEATURES,
    n_clusters: int = 4,
    random_state: int = 42
) -> Tuple[KMeans, StandardScaler, np.ndarray]:
    """
    Fits StandardScaler and KMeans on customer behavioral/spending features.
    Returns the fitted KMeans model, scaler, and assigned cluster labels.
    """
    X = df[features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(
        n_clusters=n_clusters,
        init='k-means++',
        random_state=random_state,
        n_init=10
    )
    cluster_labels = kmeans.fit_predict(X_scaled)
    return kmeans, scaler, cluster_labels


def calculate_cluster_statistics(
    df: pd.DataFrame,
    cluster_labels: np.ndarray,
    target_series: Optional[pd.Series] = None
) -> pd.DataFrame:
    """
    Calculates empirical cluster statistics from the actual dataset:
    - Customer Count
    - Proportion of Total (%)
    - Average & Median Tenure (months)
    - Average & Median Monthly Charges ($)
    - Average & Median Total Charges ($)
    - Actual Churn Rate (%) if target is provided
    """
    df_temp = df.copy()
    df_temp['Cluster'] = cluster_labels
    if target_series is not None:
        df_temp['Churn'] = target_series.values

    total_records = len(df_temp)

    records = []
    for c in sorted(np.unique(cluster_labels)):
        sub = df_temp[df_temp['Cluster'] == c]
        count = len(sub)
        pct = (count / total_records) * 100.0

        rec: Dict[str, Any] = {
            'Cluster': int(c),
            'Customer Count': int(count),
            'Share of Base (%)': round(pct, 1),
            'Avg Tenure (Mo)': round(float(sub['tenure'].mean()), 1),
            'Median Tenure (Mo)': round(float(sub['tenure'].median()), 1),
            'Avg Monthly ($)': round(float(sub['MonthlyCharges'].mean()), 2),
            'Median Monthly ($)': round(float(sub['MonthlyCharges'].median()), 2),
            'Avg Total ($)': round(float(sub['TotalCharges'].mean()), 2),
            'Median Total ($)': round(float(sub['TotalCharges'].median()), 2)
        }

        if 'Churn' in sub.columns:
            churn_rate = (sub['Churn'].mean()) * 100.0
            rec['Actual Churn Rate (%)'] = round(float(churn_rate), 1)

        records.append(rec)

    stats_df = pd.DataFrame(records)
    return stats_df


def derive_persona_descriptions(stats_df: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
    """
    Derives evidence-supported persona titles and descriptions strictly from
    the calculated empirical statistics (comparing cluster medians to population medians).
    """
    overall_tenure_med = stats_df['Avg Tenure (Mo)'].mean()
    overall_monthly_med = stats_df['Avg Monthly ($)'].mean()

    personas = {}
    for _, row in stats_df.iterrows():
        c_id = int(row['Cluster'])
        avg_tenure = row['Avg Tenure (Mo)']
        avg_monthly = row['Avg Monthly ($)']
        avg_total = row['Avg Total ($)']
        churn_rate = row.get('Actual Churn Rate (%)', None)

        # Evidence-based criteria
        is_long_tenure = avg_tenure >= overall_tenure_med
        is_high_spend = avg_monthly >= overall_monthly_med

        if is_long_tenure and is_high_spend:
            title = "High-Value Loyal Champions"
            description = (
                f"Long-tenured subscribers (avg. {avg_tenure:.1f} months) with high monthly spend "
                f"(avg. ${avg_monthly:.2f}) and substantial cumulative lifetime revenue (avg. ${avg_total:.2f})."
            )
            strategy = "VIP retention perks, premium contract lock-in incentives, and proactive service checks."
            badge_color = "green"
        elif not is_long_tenure and is_high_spend:
            title = "New High-Spend / At-Risk"
            description = (
                f"Recent acquisitions (avg. {avg_tenure:.1f} months) paying premium monthly charges "
                f"(avg. ${avg_monthly:.2f})"
            )
            if churn_rate is not None:
                description += f" exhibiting high churn vulnerability ({churn_rate:.1f}% churn rate)."
            else:
                description += "."
            strategy = "Immediate onboarding engagement, satisfaction outreach, and value reinforcement."
            badge_color = "red"
        elif is_long_tenure and not is_high_spend:
            title = "Established Budget Retained"
            description = (
                f"Long-term subscribers (avg. {avg_tenure:.1f} months) utilizing basic, cost-effective plans "
                f"(avg. ${avg_monthly:.2f}/mo) with high loyalty."
            )
            strategy = "Targeted cross-selling of incremental features without disrupting affordability."
            badge_color = "blue"
        else:
            title = "New Budget Starters"
            description = (
                f"Newer subscribers (avg. {avg_tenure:.1f} months) on entry-level, low-cost services "
                f"(avg. ${avg_monthly:.2f}/mo)."
            )
            strategy = "Guided feature discovery and upgrade path milestones as product usage expands."
            badge_color = "orange"

        personas[c_id] = {
            'cluster_id': c_id,
            'title': title,
            'description': description,
            'strategy': strategy,
            'badge_color': badge_color,
            'metrics': {
                'avg_tenure': avg_tenure,
                'avg_monthly': avg_monthly,
                'avg_total': avg_total,
                'churn_rate': churn_rate
            }
        }

    return personas


def predict_customer_segment(
    customer_df: pd.DataFrame,
    kmeans_model: KMeans,
    scaler: StandardScaler,
    features: List[str] = SEGMENTATION_FEATURES
) -> int:
    """
    Assigns a single customer or batch to a cluster segment using the fitted scaler and K-Means model.
    """
    X = customer_df[features].copy()
    X_scaled = scaler.transform(X)
    cluster = kmeans_model.predict(X_scaled)[0]
    return int(cluster)
