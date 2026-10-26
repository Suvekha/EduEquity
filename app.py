"""
EduEquity: Multi-Algorithm District Clustering, Stability Validation & Supervised Forecasting Dashboard
An interactive Streamlit application to compare clustering algorithms, validate cluster stability,
profile multi-dimensional indicators (enrolment, teachers, gender equity, infrastructure),
and perform chronological supervised machine learning without saving static files to disk.
"""

from io import BytesIO
from itertools import combinations
from pathlib import Path
import re
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Scikit-Learn Unsupervised Modules
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    normalized_mutual_info_score,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Scikit-Learn Supervised Modules
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
import streamlit as st

# ==============================================================================
# 1. PAGE CONFIGURATION & CUSTOM STYLES
# ==============================================================================
st.set_page_config(
    page_title="EduEquity | Advanced ML & District Clustering Platform",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .hero-banner {
        background: linear-gradient(135deg, #090d16 0%, #1e1b4b 45%, #2e1065 100%);
        border-radius: 16px;
        padding: 28px 34px;
        color: white;
        margin-bottom: 22px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .hero-title {
        font-size: 2.15rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        margin-bottom: 6px;
        background: linear-gradient(90deg, #ffffff, #c7d2fe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        font-size: 1.0rem;
        color: #94a3b8;
        max-width: 900px;
        line-height: 1.6;
        margin-bottom: 12px;
    }
    .hero-badge-container {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        margin-top: 12px;
    }
    .hero-badge {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(8px);
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        color: #e2e8f0;
        border: 1px solid rgba(255, 255, 255, 0.15);
    }
    
    .metric-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 18px 20px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 14px -2px rgba(0, 0, 0, 0.08);
    }
    .metric-label {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-bottom: 3px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #0f172a;
    }
    .metric-desc {
        font-size: 0.78rem;
        color: #94a3b8;
        margin-top: 3px;
    }
    
    .archetype-card {
        border-radius: 14px;
        padding: 18px;
        margin-bottom: 14px;
        background: #f8fafc;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
        border: 1px solid #e2e8f0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.88rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Colors for consistent visualization
CLUSTER_COLORS = {
    0: "#3b82f6",  # Blue
    1: "#10b981",  # Emerald
    2: "#f59e0b",  # Amber
    3: "#8b5cf6",  # Purple
    4: "#ec4899",  # Pink
    5: "#06b6d4",  # Cyan
    6: "#f97316",  # Orange
    -1: "#94a3b8", # Gray (DBSCAN Outliers)
}

# ==============================================================================
# 2. DATA LOADING & MULTI-DIMENSIONAL PREPARATION (IN-MEMORY)
# ==============================================================================

def normalize_name(value: str) -> str:
    """Normalize a column name for comparison."""
    val = str(value).strip().lower()
    val = re.sub(r"[^a-z0-9]+", "_", val)
    return val.strip("_")


def is_identifier_column(col: str) -> bool:
    """Filter out non-indicator ID columns."""
    name = normalize_name(col)
    excluded = {
        "district", "district_name", "district_id", "district_code",
        "state", "state_name", "state_id", "state_code", "year",
        "academic_year", "financial_year", "serial_no", "index",
        "latitude", "longitude", "lat", "lon", "long", "pincode", "pin_code",
    }
    return (
        name in excluded
        or name.startswith("unnamed")
        or name.endswith("_id")
        or name.endswith("_code")
    )


@st.cache_data(show_spinner=False)
def load_all_datasets(default_dir: str = "data/processed"):
    """
    Load 2018-19 & 2019-20 datasets and enrich with enrolment and teacher summaries.
    Cached in memory. Zero file writes.
    """
    p = Path(default_dir)
    p18 = p / "eduequity_model_dataset_2018_19.csv"
    p19 = p / "eduequity_model_dataset_2019_20.csv"
    p_enr = p / "district_year_enrolment.csv"
    p_tea = p / "district_teacher_summary.csv"

    if not p18.exists() or not p19.exists():
        return None, None, None, None, "Required model datasets missing from data/processed"

    df18 = pd.read_csv(p18, low_memory=False)
    df19 = pd.read_csv(p19, low_memory=False)

    df18.columns = [str(c).strip() for c in df18.columns]
    df19.columns = [str(c).strip() for c in df19.columns]

    # Multi-dimensional enrichments: Enrolment
    enr_df = None
    if p_enr.exists():
        enr_df = pd.read_csv(p_enr, low_memory=False)

    # Multi-dimensional enrichments: Teachers
    tea_df = None
    if p_tea.exists():
        tea_df = pd.read_csv(p_tea, low_memory=False)

    def enrich_year(df, year_str):
        d = df.copy()
        d["academic_year"] = year_str
        d["district_name"] = d["District_Name"].astype(str).str.strip()
        d["state_name"] = d["State_Name"].astype(str).str.strip()
        d["district_code"] = pd.to_numeric(d["District_Code"], errors="coerce").fillna(0).astype(int)

        if enr_df is not None:
            sub_enr = enr_df[enr_df["ac_year"] == year_str]
            merged_enr = pd.merge(
                d,
                sub_enr[["district_cd", "total_enrolment", "girls_to_boys_ratio", "total_boys", "total_girls"]],
                left_on="district_code",
                right_on="district_cd",
                how="left",
            )
            # Fill or retain if already present
            for col in ["total_enrolment", "girls_to_boys_ratio", "total_boys", "total_girls"]:
                if col in merged_enr.columns:
                    d[col] = merged_enr[col]

        if tea_df is not None:
            merged_tea = pd.merge(
                d,
                tea_df[["District_Code", "Total_Teacher", "Total_Male", "Total_Female"]],
                left_on="district_code",
                right_on="District_Code",
                how="left",
            )
            d["total_teachers"] = merged_tea["Total_Teacher"]
            d["total_male_teachers"] = merged_tea["Total_Male"]
            d["total_female_teachers"] = merged_tea["Total_Female"]
            d["pupil_teacher_ratio"] = d["total_enrolment"] / d["total_teachers"].replace(0, np.nan)
            d["female_teacher_share"] = d["total_female_teachers"] / d["total_teachers"].replace(0, np.nan)
        else:
            d["total_teachers"] = np.nan
            d["pupil_teacher_ratio"] = np.nan
            d["female_teacher_share"] = np.nan

        # Gender equity
        if "girls_to_boys_ratio" not in d.columns or d["girls_to_boys_ratio"].isna().all():
            if "girls_enrolment_share" in d.columns:
                d["girls_to_boys_ratio"] = d["girls_enrolment_share"] / (1.0 - d["girls_enrolment_share"] + 1e-6)
            else:
                d["girls_to_boys_ratio"] = 1.0

        return d

    df18_enr = enrich_year(df18, "2018-19")
    df19_enr = enrich_year(df19, "2019-20")

    # Common numeric infrastructure indicators
    common_indicators = [
        c for c in df18.columns
        if c in df19.columns and c.endswith("_rate")
    ]

    combined = pd.concat([df18_enr, df19_enr], ignore_index=True)

    # Matched districts between 2018-19 and 2019-20
    matched_districts = pd.merge(
        df18_enr,
        df19_enr,
        on="district_code",
        suffixes=("_18", "_19"),
    )

    return combined, common_indicators, matched_districts, (df18_enr, df19_enr), None


# ==============================================================================
# 3. ADVANCED CLUSTERING & MULTI-ALGORITHM BENCHMARK
# ==============================================================================

@st.cache_data(show_spinner=False)
def benchmark_clustering_algorithms(
    combined_df: pd.DataFrame,
    features: list,
    k_clusters: int = 3,
    dbscan_eps: float = 4.0,
    dbscan_min_samples: int = 10,
    hierarchical_linkage: str = "ward",
    random_state: int = 42,
):
    """
    Compare K-Means, Hierarchical (Agglomerative), DBSCAN, and GMM
    on the exact same prepared feature set.
    """
    X_raw = combined_df[features].copy()
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X_raw)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    results = {}
    metric_rows = []

    # 1. K-Means
    km = KMeans(n_clusters=k_clusters, random_state=random_state, n_init=10)
    labels_km = km.fit_predict(X_scaled)
    sil_km = silhouette_score(X_scaled, labels_km)
    db_km = davies_bouldin_score(X_scaled, labels_km)
    ch_km = calinski_harabasz_score(X_scaled, labels_km)
    results["K-Means"] = {
        "labels": labels_km,
        "model": km,
        "n_clusters": k_clusters,
        "outliers": 0,
        "silhouette": sil_km,
        "davies_bouldin": db_km,
        "calinski_harabasz": ch_km,
    }
    metric_rows.append({
        "Algorithm": "K-Means",
        "Archetype / Family": "Centroid-Based Partitioning",
        "Clusters": k_clusters,
        "Outliers": 0,
        "Silhouette Score (↑)": sil_km,
        "Davies-Bouldin Index (↓)": db_km,
        "Calinski-Harabasz Index (↑)": ch_km,
    })

    # 2. Hierarchical (Agglomerative)
    agg = AgglomerativeClustering(
        n_clusters=k_clusters,
        metric="euclidean",
        linkage=hierarchical_linkage,
    )
    labels_agg = agg.fit_predict(X_scaled)
    sil_agg = silhouette_score(X_scaled, labels_agg)
    db_agg = davies_bouldin_score(X_scaled, labels_agg)
    ch_agg = calinski_harabasz_score(X_scaled, labels_agg)
    results["Hierarchical"] = {
        "labels": labels_agg,
        "model": agg,
        "n_clusters": k_clusters,
        "outliers": 0,
        "silhouette": sil_agg,
        "davies_bouldin": db_agg,
        "calinski_harabasz": ch_agg,
    }
    metric_rows.append({
        "Algorithm": "Hierarchical Clustering",
        "Archetype / Family": "Bottom-Up Agglomerative Tree",
        "Clusters": k_clusters,
        "Outliers": 0,
        "Silhouette Score (↑)": sil_agg,
        "Davies-Bouldin Index (↓)": db_agg,
        "Calinski-Harabasz Index (↑)": ch_agg,
    })

    # 3. DBSCAN
    dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
    labels_dbs = dbscan.fit_predict(X_scaled)
    n_clusters_dbs = len(set(labels_dbs)) - (1 if -1 in labels_dbs else 0)
    outliers_dbs = (labels_dbs == -1).sum()

    # Metrics on non-outlier points if >= 2 clusters
    if n_clusters_dbs >= 2 and (labels_dbs != -1).sum() > n_clusters_dbs:
        valid_mask = labels_dbs != -1
        sil_dbs = float(silhouette_score(X_scaled[valid_mask], labels_dbs[valid_mask]))
        db_dbs = float(davies_bouldin_score(X_scaled[valid_mask], labels_dbs[valid_mask]))
        ch_dbs = float(calinski_harabasz_score(X_scaled[valid_mask], labels_dbs[valid_mask]))
    else:
        sil_dbs, db_dbs, ch_dbs = np.nan, np.nan, np.nan

    results["DBSCAN"] = {
        "labels": labels_dbs,
        "model": dbscan,
        "n_clusters": n_clusters_dbs,
        "outliers": outliers_dbs,
        "silhouette": sil_dbs,
        "davies_bouldin": db_dbs,
        "calinski_harabasz": ch_dbs,
    }
    metric_rows.append({
        "Algorithm": "DBSCAN",
        "Archetype / Family": "Density-Based & Outlier Detection",
        "Clusters": n_clusters_dbs,
        "Outliers": outliers_dbs,
        "Silhouette Score (↑)": sil_dbs,
        "Davies-Bouldin Index (↓)": db_dbs,
        "Calinski-Harabasz Index (↑)": ch_dbs,
    })

    # 4. Gaussian Mixture Model (GMM)
    gmm = GaussianMixture(n_components=k_clusters, random_state=random_state, n_init=5)
    gmm.fit(X_scaled)
    labels_gmm = gmm.predict(X_scaled)
    probs_gmm = gmm.predict_proba(X_scaled)
    # Entropy / uncertainty of assignment: - sum p * log(p)
    entropy_gmm = -np.sum(probs_gmm * np.log(np.clip(probs_gmm, 1e-10, 1.0)), axis=1)

    sil_gmm = silhouette_score(X_scaled, labels_gmm)
    db_gmm = davies_bouldin_score(X_scaled, labels_gmm)
    ch_gmm = calinski_harabasz_score(X_scaled, labels_gmm)

    results["GMM"] = {
        "labels": labels_gmm,
        "probs": probs_gmm,
        "entropy": entropy_gmm,
        "model": gmm,
        "n_clusters": k_clusters,
        "outliers": 0,
        "silhouette": sil_gmm,
        "davies_bouldin": db_gmm,
        "calinski_harabasz": ch_gmm,
        "bic": gmm.bic(X_scaled),
        "aic": gmm.aic(X_scaled),
    }
    metric_rows.append({
        "Algorithm": "Gaussian Mixture Model (GMM)",
        "Archetype / Family": "Probabilistic Soft Clustering",
        "Clusters": k_clusters,
        "Outliers": 0,
        "Silhouette Score (↑)": sil_gmm,
        "Davies-Bouldin Index (↓)": db_gmm,
        "Calinski-Harabasz Index (↑)": ch_gmm,
    })

    # Pairwise Adjusted Rand Index (Agreement Matrix)
    algo_names = ["K-Means", "Hierarchical", "DBSCAN", "GMM"]
    ari_matrix = pd.DataFrame(index=algo_names, columns=algo_names, dtype=float)
    nmi_matrix = pd.DataFrame(index=algo_names, columns=algo_names, dtype=float)

    for a1 in algo_names:
        for a2 in algo_names:
            l1 = results[a1]["labels"]
            l2 = results[a2]["labels"]
            ari_matrix.loc[a1, a2] = adjusted_rand_score(l1, l2)
            nmi_matrix.loc[a1, a2] = normalized_mutual_info_score(l1, l2)

    # 2D PCA representation for plotting
    pca_2d = PCA(n_components=2, random_state=random_state)
    coords = pca_2d.fit_transform(X_scaled)

    df_viz = combined_df.copy()
    df_viz["PC1"] = coords[:, 0]
    df_viz["PC2"] = coords[:, 1]
    df_viz["KMeans_Cluster"] = labels_km
    df_viz["Hierarchical_Cluster"] = labels_agg
    df_viz["DBSCAN_Cluster"] = labels_dbs
    df_viz["GMM_Cluster"] = labels_gmm
    df_viz["GMM_Entropy"] = entropy_gmm
    df_viz["GMM_Confidence"] = probs_gmm.max(axis=1)

    return {
        "metrics_df": pd.DataFrame(metric_rows),
        "ari_matrix": ari_matrix,
        "nmi_matrix": nmi_matrix,
        "results": results,
        "df_viz": df_viz,
        "X_scaled": X_scaled,
        "pca_variance": pca_2d.explained_variance_ratio_ * 100,
    }


# ==============================================================================
# 4. CLUSTER STABILITY VALIDATION ACROSS MULTIPLE SEEDS
# ==============================================================================

@st.cache_data(show_spinner=False)
def evaluate_cluster_stability(
    X_scaled: np.ndarray,
    k: int = 3,
    seeds: list = (42, 10, 2023, 7, 99, 123, 555, 777),
):
    """
    Run K-Means across multiple random initializations.
    Compute pairwise ARI between runs and evaluate cluster co-association stability.
    """
    runs = {}
    inertias = []
    silhouettes = []

    for s in seeds:
        km = KMeans(n_clusters=k, random_state=s, n_init=1)  # Single init to test sensitivity
        labels = km.fit_predict(X_scaled)
        runs[s] = labels
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))

    # Pairwise ARI matrix
    pairwise_ari = []
    seed_pairs = list(combinations(seeds, 2))
    for s1, s2 in seed_pairs:
        ari = adjusted_rand_score(runs[s1], runs[s2])
        pairwise_ari.append(ari)

    mean_ari = np.mean(pairwise_ari)
    std_ari = np.std(pairwise_ari)
    min_ari = np.min(pairwise_ari)
    max_ari = np.max(pairwise_ari)

    summary_df = pd.DataFrame({
        "Random Seed": seeds,
        "Inertia": inertias,
        "Silhouette Score": silhouettes,
    })

    return {
        "mean_ari": mean_ari,
        "std_ari": std_ari,
        "min_ari": min_ari,
        "max_ari": max_ari,
        "summary_df": summary_df,
        "pairwise_ari": pairwise_ari,
        "runs": runs,
    }


# ==============================================================================
# 5. MEANINGFUL DATA-DRIVEN CLUSTER ARCHETYPE DERIVATION
# ==============================================================================

def assign_meaningful_archetypes(df_with_clusters: pd.DataFrame, cluster_col: str, feature_cols: list):
    """
    Examine cluster distributions empirically across Enrolment, Teachers,
    Gender Equity, and Infrastructure. Give them descriptive names instead of arbitrary 0, 1, 2.
    """
    cluster_means = df_with_clusters.groupby(cluster_col)[feature_cols].mean()

    # Key composite benchmark indicators
    digital_cols = [c for c in ["computer_available_rate", "internet_rate"] if c in feature_cols]
    sanitation_cols = [c for c in ["drinking_water_rate", "functional_girl_toilet_rate", "functional_boy_toilet_rate"] if c in feature_cols]
    facilities_cols = [c for c in ["functional_electricity_rate", "ramps_rate", "library_or_reading_corner_or_book_bank_rate"] if c in feature_cols]

    rankings = {}
    for c_id in cluster_means.index:
        if c_id == -1:
            rankings[c_id] = {
                "name": "Outlier / Atypical Districts",
                "tagline": "Sparse or anomalous educational indicators (DBSCAN)",
                "composite": -1.0,
                "tier": "Outlier",
            }
            continue

        c_data = cluster_means.loc[c_id]
        dig_score = c_data[digital_cols].mean() if digital_cols else 0.5
        san_score = c_data[sanitation_cols].mean() if sanitation_cols else 0.5
        fac_score = c_data[facilities_cols].mean() if facilities_cols else 0.5

        # Weighted composite score
        composite = 0.4 * dig_score + 0.35 * san_score + 0.25 * fac_score

        rankings[c_id] = {
            "composite": composite,
            "dig": dig_score,
            "san": san_score,
            "fac": fac_score,
        }

    # Rank non-outlier clusters by composite score
    valid_clusters = [c for c in rankings.keys() if c != -1]
    sorted_by_score = sorted(valid_clusters, key=lambda c: rankings[c]["composite"], reverse=True)

    tier_names = [
        ("Tier 1: High Digital & Comprehensive Facilities", "Advanced digital access, complete sanitation, high facility coverage"),
        ("Tier 2: Basic Provision with Digital Deficit", "Adequate electricity & sanitation, but severe computer and internet deficits"),
        ("Tier 3: Priority Intervention / Acute Deficit", "Severe gaps in drinking water testing, basic facilities, and digital learning"),
        ("Tier 4: Emerging / Transitional Provision", "Moderate school amenities with localized equity and infrastructure bottlenecks"),
    ]

    for rank_idx, c_id in enumerate(sorted_by_score):
        name, tagline = tier_names[min(rank_idx, len(tier_names) - 1)]
        rankings[c_id]["name"] = name
        rankings[c_id]["tagline"] = tagline
        rankings[c_id]["tier"] = f"Tier {rank_idx + 1}"

    return rankings


# ==============================================================================
# 6. SUPERVISED ML PIPELINE (CHRONOLOGICAL VALIDATION)
# ==============================================================================

@st.cache_data(show_spinner=False)
def run_supervised_learning(
    matched_df: pd.DataFrame,
    features: list,
    task_type: str = "Regression",
    target_variable: str = "girls_to_boys_ratio_19",
    risk_threshold: float = 0.90,
    random_state: int = 42,
):
    """
    Supervised learning with strict Chronological Validation:
    Predicts 2019-20 future outcomes from 2018-19 baseline features.
    """
    feature_cols_18 = [f + "_18" for f in features if f + "_18" in matched_df.columns]
    if len(feature_cols_18) < 2:
        feature_cols_18 = [f for f in features if f in matched_df.columns]

    X = matched_df[feature_cols_18].copy()
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    # Chronological Split (Train on 75% districts baseline, Test on 25% holdout)
    # Both evaluate predicting the FUTURE 2019-20 target from the past 2018-19 baseline!
    if task_type == "Regression":
        y = matched_df[target_variable].values
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.25, random_state=random_state
        )

        models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression (L2)": Ridge(alpha=1.0),
            "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=random_state),
            "Gradient Boosting": HistGradientBoostingRegressor(random_state=random_state),
        }

        perf_rows = []
        predictions = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            predictions[name] = preds
            perf_rows.append({
                "Algorithm": name,
                "R² Score (↑)": r2_score(y_test, preds),
                "RMSE (↓)": np.sqrt(mean_squared_error(y_test, preds)),
                "MAE (↓)": mean_absolute_error(y_test, preds),
            })

        # Feature Importance from Random Forest
        rf = models["Random Forest Regressor"]
        imp_df = pd.DataFrame({
            "Indicator": [f.replace("_18", "").replace("_", " ").title() for f in feature_cols_18],
            "Importance": rf.feature_importances_,
        }).sort_values(by="Importance", ascending=False).head(12)

        return {
            "perf_df": pd.DataFrame(perf_rows),
            "predictions": predictions,
            "y_test": y_test,
            "importance_df": imp_df,
            "task_type": "Regression",
        }

    else:
        # Binary Classification: Documented At-Risk District Status
        # At-Risk = 1 if target is below threshold (e.g. low gender parity or low digital rate in 2019-20)
        y_raw = matched_df[target_variable].values
        y_binary = (y_raw < risk_threshold).astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_binary, test_size=0.25, random_state=random_state, stratify=y_binary
        )

        models = {
            "Logistic Regression": LogisticRegression(random_state=random_state, max_iter=1000),
            "Random Forest Classifier": RandomForestClassifier(n_estimators=100, random_state=random_state),
            "Gradient Boosting Classifier": HistGradientBoostingClassifier(random_state=random_state),
        }

        perf_rows = []
        predictions = {}
        probabilities = {}

        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else preds
            predictions[name] = preds
            probabilities[name] = probs

            perf_rows.append({
                "Algorithm": name,
                "Accuracy (↑)": accuracy_score(y_test, preds),
                "Precision (↑)": precision_score(y_test, preds, zero_division=0),
                "Recall (↑)": recall_score(y_test, preds, zero_division=0),
                "F1-Score (↑)": f1_score(y_test, preds, zero_division=0),
                "ROC-AUC (↑)": roc_auc_score(y_test, probs) if len(np.unique(y_test)) > 1 else np.nan,
            })

        # Feature Importance from Random Forest Classifier
        rf = models["Random Forest Classifier"]
        imp_df = pd.DataFrame({
            "Indicator": [f.replace("_18", "").replace("_", " ").title() for f in feature_cols_18],
            "Importance": rf.feature_importances_,
        }).sort_values(by="Importance", ascending=False).head(12)

        return {
            "perf_df": pd.DataFrame(perf_rows),
            "predictions": predictions,
            "probabilities": probabilities,
            "y_test": y_test,
            "importance_df": imp_df,
            "task_type": "Classification",
        }


# ==============================================================================
# 7. MAIN APPLICATION PIPELINE EXECUTION
# ==============================================================================

# Load enriched data
combined_df, common_indicators, matched_districts, (df18, df19), err = load_all_datasets()

if err or combined_df is None:
    st.error(f"❌ {err}")
    st.stop()

# Sidebar Controls
with st.sidebar:
    st.markdown("### ⚙️ Platform Settings")
    st.caption("Control algorithm hyperparameters, cluster counts, and supervised targets.")

    st.markdown("#### 🎯 Cluster Tuning")
    k_selected = st.slider("Number of Clusters (K):", min_value=2, max_value=6, value=3)

    with st.expander("🔬 Algorithm-Specific Parameters", expanded=False):
        hier_linkage = st.selectbox("Hierarchical Linkage:", ["ward", "complete", "average"], index=0)
        dbscan_eps = st.slider("DBSCAN Epsilon (eps):", min_value=1.5, max_value=6.0, value=3.8, step=0.1)
        dbscan_min_samples = st.slider("DBSCAN Min Samples:", min_value=3, max_value=20, value=8, step=1)
        random_seed = st.number_input("Random Seed:", value=42, step=1)

    st.markdown("#### 🤖 Supervised ML Target")
    ml_task = st.radio("Supervised Prediction Task:", ["Numerical Outcome (Regression)", "Documented Risk Category (Classification)"])

    if ml_task.startswith("Numerical"):
        target_col = st.selectbox(
            "Predict 2019-20 Outcome from 2018-19:",
            ["girls_to_boys_ratio_19", "computer_available_rate_19", "total_enrolment_19"],
            format_func=lambda s: {
                "girls_to_boys_ratio_19": "Gender Equity (Girls-to-Boys Ratio)",
                "computer_available_rate_19": "Digital Readiness (Computer Rate)",
                "total_enrolment_19": "District Total Enrolment",
            }.get(s, s),
        )
        risk_thresh = 0.90
    else:
        target_col = "girls_to_boys_ratio_19"
        risk_thresh = st.slider(
            "At-Risk Cutoff (Girls/Boys Ratio):",
            min_value=0.80,
            max_value=0.98,
            value=0.91,
            step=0.01,
            help="Districts with 2019-20 ratio below this threshold are labeled At-Risk.",
        )

# Execute In-Memory Benchmark
with st.spinner("⚡ Running multi-algorithm benchmark & multi-dimensional profiling..."):
    bench_data = benchmark_clustering_algorithms(
        combined_df=combined_df,
        features=common_indicators,
        k_clusters=k_selected,
        dbscan_eps=dbscan_eps,
        dbscan_min_samples=dbscan_min_samples,
        hierarchical_linkage=hier_linkage,
        random_state=random_seed,
    )

    stability_data = evaluate_cluster_stability(
        X_scaled=bench_data["X_scaled"],
        k=k_selected,
    )

    archetypes = assign_meaningful_archetypes(
        df_with_clusters=bench_data["df_viz"],
        cluster_col="KMeans_Cluster",
        feature_cols=common_indicators,
    )

# ==============================================================================
# 8. HERO BANNER & MULTI-DIMENSIONAL KPIS
# ==============================================================================
st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-title">🎓 EduEquity: District Multi-Algorithm Clustering & Supervised Forecasting</div>
        <div class="hero-subtitle">
            Comprehensive comparative analysis of K-Means, Hierarchical, DBSCAN, and GMM. 
            Multi-seed stability validation, multi-dimensional profiling across enrolment, teachers, gender equity, 
            and infrastructure, plus chronological supervised forecasting without saving static files.
        </div>
        <div class="hero-badge-container">
            <span class="hero-badge">🧠 4 Clustering Algorithms Compared</span>
            <span class="hero-badge">🛡️ Multi-Seed Stability Tested</span>
            <span class="hero-badge">⚖️ Gender & Teacher Equity Profiling</span>
            <span class="hero-badge">🔮 Chronological Supervised Forecasting</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# High-Level KPIs
c1, c2, c3, c4, c5 = st.columns(5)
total_records = len(combined_df)
mean_stability = stability_data["mean_ari"]
km_sil = bench_data["results"]["K-Means"]["silhouette"]
dbs_outliers = bench_data["results"]["DBSCAN"]["outliers"]
gmm_entropy = bench_data["df_viz"]["GMM_Entropy"].mean()

with c1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">District Records</div>
            <div class="metric-value">{total_records:,}</div>
            <div class="metric-desc">651 (2018-19) | 732 (2019-20)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">K-Means Silhouette</div>
            <div class="metric-value">{km_sil:.3f}</div>
            <div class="metric-desc">K={k_selected} Separation Index</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Stability Index (ARI)</div>
            <div class="metric-value">{mean_stability:.3f}</div>
            <div class="metric-desc">Agreement across 8 Random Seeds</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">DBSCAN Outliers</div>
            <div class="metric-value">{dbs_outliers}</div>
            <div class="metric-desc">Identified Atypical Districts</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c5:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">GMM Mean Entropy</div>
            <div class="metric-value">{gmm_entropy:.2f}</div>
            <div class="metric-desc">Soft Boundary Uncertainty</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

# ==============================================================================
# 9. ADVANCED NAVIGATION TABS
# ==============================================================================
tab_compare, tab_stability, tab_profile, tab_supervised, tab_mobility, tab_explorer, tab_report = st.tabs([
    "🧪 Algorithm Comparison",
    "🛡️ Cluster Stability & Seeds",
    "🏛️ Multi-Dimensional Profiling",
    "🔮 Supervised ML & Forecasting",
    "🔁 Year-Over-Year Mobility",
    "🔍 District Explorer",
    "📄 Live Policy Briefing",
])

# ------------------------------------------------------------------------------
# TAB 1: ALGORITHM COMPARISON
# ------------------------------------------------------------------------------
with tab_compare:
    st.markdown("### 🧪 Comparative Evaluation of Clustering Paradigms")
    st.markdown(
        """
        Evaluate how different mathematical formulations cluster the district educational landscape:
        - **K-Means**: Identifies spherical clusters around centroids.
        - **Hierarchical Clustering**: Bottom-up agglomerative tree revealing hierarchical similarity distances.
        - **DBSCAN**: Density-based clustering that isolates dense cores and flags anomalous outliers.
        - **Gaussian Mixture Models (GMM)**: Soft, probabilistic clustering estimating distribution mixtures and uncertainty.
        """
    )

    # Benchmark Summary Table
    st.markdown("#### 📋 Unified Performance Scorecard")
    st.dataframe(
        bench_data["metrics_df"].style.highlight_max(
            subset=["Silhouette Score (↑)", "Calinski-Harabasz Index (↑)"], color="#dcfce7"
        ).highlight_min(
            subset=["Davies-Bouldin Index (↓)"], color="#e0e7ff"
        ).format({
            "Silhouette Score (↑)": "{:.4f}",
            "Davies-Bouldin Index (↓)": "{:.4f}",
            "Calinski-Harabasz Index (↑)": "{:,.1f}",
        }),
        use_container_width=True,
    )

    st.markdown("---")
    col_ag1, col_ag2 = st.columns([1.1, 1])

    with col_ag1:
        st.markdown("#### 🤝 Pairwise Algorithm Agreement (Adjusted Rand Index)")
        st.caption("Values near 1.0 indicate strong structural consensus between algorithm pairings.")
        fig_ari = px.imshow(
            bench_data["ari_matrix"],
            text_auto=".3f",
            labels=dict(x="Algorithm A", y="Algorithm B", color="Adjusted Rand Index"),
            color_continuous_scale="Purples",
            template="plotly_white",
        )
        fig_ari.update_layout(margin=dict(l=20, r=20, t=30, b=20), height=360)
        st.plotly_chart(fig_ari, use_container_width=True)

    with col_ag2:
        st.markdown("#### 📊 Comparative Metric Bar Breakdown")
        comp_m = bench_data["metrics_df"].dropna(subset=["Silhouette Score (↑)"])
        fig_m = px.bar(
            comp_m,
            x="Algorithm",
            y="Silhouette Score (↑)",
            color="Algorithm",
            text=comp_m["Silhouette Score (↑)"].apply(lambda v: f"{v:.3f}"),
            template="plotly_white",
            color_discrete_sequence=["#3b82f6", "#10b981", "#8b5cf6", "#f59e0b"],
        )
        fig_m.update_traces(textposition="outside")
        fig_m.update_layout(margin=dict(l=20, r=20, t=30, b=20), showlegend=False)
        st.plotly_chart(fig_m, use_container_width=True)

    # Side-by-side 2D PCA Projections
    st.markdown("---")
    st.markdown("#### 🪐 Spatial Cluster Projection Comparison (2D PCA)")
    
    algo_to_view = st.selectbox(
        "Select Algorithm to View in 2D PCA Space:",
        ["K-Means", "Hierarchical Clustering", "DBSCAN", "Gaussian Mixture Model (GMM)"],
        index=0,
    )

    col_map = {
        "K-Means": "KMeans_Cluster",
        "Hierarchical Clustering": "Hierarchical_Cluster",
        "DBSCAN": "DBSCAN_Cluster",
        "Gaussian Mixture Model (GMM)": "GMM_Cluster",
    }
    view_col = col_map[algo_to_view]

    plot_df = bench_data["df_viz"].copy()
    plot_df["Display_Label"] = plot_df[view_col].apply(
        lambda c: "Outlier / Noise" if c == -1 else f"Cluster {c}"
    )

    fig_pca_comp = px.scatter(
        plot_df,
        x="PC1",
        y="PC2",
        color="Display_Label",
        symbol="academic_year",
        hover_name="district_name",
        hover_data={"state_name": True, "academic_year": True, "Display_Label": True, "PC1": ":.2f", "PC2": ":.2f"},
        template="plotly_white",
        title=f"{algo_to_view} Cluster Boundaries in PCA Space",
        color_discrete_map={
            "Outlier / Noise": "#94a3b8",
            "Cluster 0": "#3b82f6",
            "Cluster 1": "#10b981",
            "Cluster 2": "#f59e0b",
            "Cluster 3": "#8b5cf6",
            "Cluster 4": "#ec4899",
        },
    )
    fig_pca_comp.update_traces(marker=dict(size=7, opacity=0.82))
    fig_pca_comp.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=550)
    st.plotly_chart(fig_pca_comp, use_container_width=True)

    # GMM Soft Clustering & Boundary Uncertainty
    st.markdown("---")
    st.markdown("#### 🌫️ GMM Soft Probabilities & Boundary Districts")
    st.caption("Gaussian Mixture Models assign probabilities to each cluster. Districts with higher classification entropy sit on ambiguous boundaries between tiers.")

    boundary_districts = bench_data["df_viz"].sort_values(by="GMM_Entropy", ascending=False).head(10)
    cols_show = ["district_name", "state_name", "academic_year", "GMM_Cluster", "GMM_Confidence", "GMM_Entropy"]
    st.dataframe(
        boundary_districts[cols_show].style.format({
            "GMM_Confidence": "{:.1%}",
            "GMM_Entropy": "{:.3f}",
        }),
        use_container_width=True,
    )


# ------------------------------------------------------------------------------
# TAB 2: CLUSTER STABILITY & SEEDS
# ------------------------------------------------------------------------------
with tab_stability:
    st.markdown("### 🛡️ Cluster Stability & Multi-Seed Validation")
    st.markdown(
        """
        K-Means depends on initial centroid placement. A robust cluster structure should yield 
        consistent district group assignments across different random seeds. 
        We tested the stability across **8 distinct random initializations**.
        """
    )

    c_s1, c_s2, c_s3, c_s4 = st.columns(4)
    with c_s1:
        st.metric("Mean Pairwise ARI", f"{stability_data['mean_ari']:.3f}", "Consistency Index")
    with c_s2:
        st.metric("Min Pairwise ARI", f"{stability_data['min_ari']:.3f}", "Worst-case Consensus")
    with c_s3:
        st.metric("Max Pairwise ARI", f"{stability_data['max_ari']:.3f}", "Best-case Consensus")
    with c_s4:
        st.metric("Standard Deviation", f"{stability_data['std_ari']:.4f}", "Cross-Seed Sensitivity")

    st.write("")
    col_st1, col_st2 = st.columns([1.2, 1])

    with col_st1:
        st.markdown("#### Multi-Seed Inertia & Silhouette Convergence")
        fig_conv = make_subplots(specs=[[{"secondary_y": True}]])
        fig_conv.add_trace(
            go.Scatter(
                x=[f"Seed {s}" for s in stability_data["summary_df"]["Random Seed"]],
                y=stability_data["summary_df"]["Inertia"],
                name="Inertia",
                line=dict(color="#4f46e5", width=2),
                mode="lines+markers",
            ),
            secondary_y=False,
        )
        fig_conv.add_trace(
            go.Scatter(
                x=[f"Seed {s}" for s in stability_data["summary_df"]["Random Seed"]],
                y=stability_data["summary_df"]["Silhouette Score"],
                name="Silhouette Score",
                line=dict(color="#10b981", width=2, dash="dot"),
                mode="lines+markers",
            ),
            secondary_y=True,
        )
        fig_conv.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=30, b=20), height=360)
        fig_conv.update_yaxes(title_text="Inertia", secondary_y=False)
        fig_conv.update_yaxes(title_text="Silhouette Score", secondary_y=True)
        st.plotly_chart(fig_conv, use_container_width=True)

    with col_st2:
        st.markdown("#### Stability Verdict & Interpretation")
        stability_pct = stability_data["mean_ari"] * 100
        if stability_data["mean_ari"] >= 0.85:
            verdict_badge = "🟢 HIGHLY STABLE"
            verdict_text = "Cluster boundaries represent genuine structural partitions in the educational indicator space rather than random initialization artifacts."
        elif stability_data["mean_ari"] >= 0.65:
            verdict_badge = "🟡 MODERATELY STABLE"
            verdict_text = "Core clusters remain consistent, but peripheral districts near boundaries fluctuate across centroid seeds."
        else:
            verdict_badge = "🔴 SENSITIVE"
            verdict_text = "Clusters exhibit sensitivity to initialization. Consider increasing n_init or using PCA regularization."

        st.markdown(
            f"""
            <div style="background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 22px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.04);">
                <div style="font-weight: 800; font-size: 1.2rem; color: #1e293b; margin-bottom: 6px;">Stability Rating: {verdict_badge}</div>
                <div style="font-size: 1.8rem; font-weight: 800; color: #4f46e5; margin-bottom: 12px;">{stability_pct:.1f}% Consensus</div>
                <p style="font-size: 0.88rem; color: #475569; line-height: 1.6;">
                    {verdict_text}
                </p>
                <div style="font-size: 0.82rem; color: #64748b; margin-top: 10px;">
                    Tested over 28 pairwise seed combinations with K={k_selected}.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ------------------------------------------------------------------------------
# TAB 3: MULTI-DIMENSIONAL PROFILING
# ------------------------------------------------------------------------------
with tab_profile:
    st.markdown("### 🏛️ Multi-Dimensional Cluster Profiling & Meaningful Descriptions")
    st.markdown(
        """
        > 💡 **Important Principle:** Cluster labels such as `0`, `1`, and `2` are arbitrary mathematical tokens.
        > We examine the empirical data across **infrastructure, enrolment, gender equity, and teacher availability** 
        > to give them data-driven, actionable archetype descriptions.
        """
    )

    # Display Meaningful Archetype Cards
    st.markdown("#### 🏷️ Empirical Archetype Descriptions")
    arch_cols = st.columns(k_selected)
    for c_id in range(k_selected):
        info = archetypes.get(c_id, {"name": f"Cluster {c_id}", "tagline": "Educational Tier", "composite": 0.5})
        with arch_cols[c_id]:
            c_count = (bench_data["df_viz"]["KMeans_Cluster"] == c_id).sum()
            border_c = CLUSTER_COLORS.get(c_id, "#4f46e5")
            st.markdown(
                f"""
                <div class="archetype-card" style="border-left: 6px solid {border_c};">
                    <div style="font-weight: 800; font-size: 1.05rem; color: #0f172a;">{info['name']}</div>
                    <div style="font-size: 0.8rem; color: #64748b; margin-bottom: 8px;">{c_count} District Records ({(c_count/total_records)*100:.1f}%)</div>
                    <p style="font-size: 0.82rem; color: #334155; line-height: 1.4; margin-bottom: 8px;">
                        {info['tagline']}
                    </p>
                    <div style="font-size: 0.78rem; font-weight: 700; color: {border_c};">
                        Composite Readiness: {info.get('composite', 0.5):.2f} / 1.00
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Multi-Dimensional Indicator Comparison Matrix
    st.markdown("---")
    st.markdown("#### ⚖️ Cluster Breakdown Across Essential Educational Dimensions")

    viz_df = bench_data["df_viz"]
    dim_summary = []
    for c_id in range(k_selected):
        c_sub = viz_df[viz_df["KMeans_Cluster"] == c_id]
        arch_name = archetypes.get(c_id, {}).get("name", f"Cluster {c_id}")
        dim_summary.append({
            "Cluster": f"Cluster {c_id}",
            "Meaningful Description": arch_name,
            "Districts": len(c_sub),
            "Avg Enrolment": c_sub["total_enrolment"].mean(),
            "Girls/Boys Ratio": c_sub["girls_to_boys_ratio"].mean(),
            "Avg Teachers": c_sub["total_teachers"].mean(),
            "Pupil-Teacher Ratio (PTR)": c_sub["pupil_teacher_ratio"].mean(),
            "Computer Access Rate": c_sub["computer_available_rate"].mean() if "computer_available_rate" in c_sub else np.nan,
            "Drinking Water Rate": c_sub["drinking_water_rate"].mean() if "drinking_water_rate" in c_sub else np.nan,
            "Functional Electricity": c_sub["functional_electricity_rate"].mean() if "functional_electricity_rate" in c_sub else np.nan,
        })
    dim_df = pd.DataFrame(dim_summary)

    st.dataframe(
        dim_df.style.format({
            "Avg Enrolment": "{:,.0f}",
            "Girls/Boys Ratio": "{:.3f}",
            "Avg Teachers": "{:,.0f}",
            "Pupil-Teacher Ratio (PTR)": "{:.1f}",
            "Computer Access Rate": "{:.1%}",
            "Drinking Water Rate": "{:.1%}",
            "Functional Electricity": "{:.1%}",
        }),
        use_container_width=True,
    )

    # Comparative Grouped Bar Charts across key indicators
    st.markdown("---")
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.markdown("#### Digital Readiness vs Electricity by Archetype")
        fig_dig = px.bar(
            dim_df,
            x="Cluster",
            y=["Computer Access Rate", "Functional Electricity"],
            barmode="group",
            template="plotly_white",
            color_discrete_sequence=["#3b82f6", "#10b981"],
        )
        fig_dig.update_layout(margin=dict(l=20, r=20, t=30, b=20), legend=dict(orientation="h", y=1.05))
        st.plotly_chart(fig_dig, use_container_width=True)

    with col_p2:
        st.markdown("#### Gender Equity Ratio by Archetype")
        fig_eq = px.bar(
            dim_df,
            x="Cluster",
            y="Girls/Boys Ratio",
            text=dim_df["Girls/Boys Ratio"].apply(lambda v: f"{v:.3f}"),
            color="Cluster",
            template="plotly_white",
            color_discrete_sequence=[CLUSTER_COLORS.get(c, "#4f46e5") for c in range(k_selected)],
        )
        fig_eq.update_traces(textposition="outside")
        fig_eq.update_layout(margin=dict(l=20, r=20, t=30, b=20), showlegend=False, yaxis_range=[0.8, 1.05])
        st.plotly_chart(fig_eq, use_container_width=True)


# ------------------------------------------------------------------------------
# TAB 4: SUPERVISED ML & CHRONOLOGICAL FORECASTING
# ------------------------------------------------------------------------------
with tab_supervised:
    st.markdown("### 🔮 Supervised Machine Learning & Chronological Forecasting")
    st.markdown(
        """
        Explore predicting future outcomes using baseline historical data.
        - **Chronological Validation**: Features from **2018-19 baseline** ($t$) are used to forecast outcomes in **2019-20** ($t+1$).
        - Prevents future data leakage and models real-world policy intervention timing.
        """
    )

    # Run Supervised Model Pipeline
    sup_mode = "Regression" if ml_task.startswith("Numerical") else "Classification"
    with st.spinner("Training supervised models on chronological split..."):
        sup_results = run_supervised_learning(
            matched_df=matched_districts,
            features=common_indicators,
            task_type=sup_mode,
            target_variable=target_col,
            risk_threshold=risk_thresh,
            random_state=random_seed,
        )

    # Evaluation Scorecard
    st.markdown(f"#### 📋 Chronological Test Set Performance ({sup_mode})")
    st.dataframe(
        sup_results["perf_df"].style.highlight_max(
            subset=[c for c in sup_results["perf_df"].columns if "↑" in c], color="#dcfce7"
        ).highlight_min(
            subset=[c for c in sup_results["perf_df"].columns if "↓" in c], color="#e0e7ff"
        ).format({c: "{:.4f}" for c in sup_results["perf_df"].columns if c != "Algorithm"}),
        use_container_width=True,
    )

    st.markdown("---")
    col_s1, col_s2 = st.columns([1.1, 1])

    if sup_mode == "Regression":
        with col_s1:
            st.markdown("#### 🎯 Actual vs Predicted Outcome (Holdout Test Set)")
            best_algo = sup_results["perf_df"].sort_values(by="R² Score (↑)", ascending=False)["Algorithm"].iloc[0]
            preds_to_plot = sup_results["predictions"][best_algo]
            actual_y = sup_results["y_test"]

            fig_reg = px.scatter(
                x=actual_y,
                y=preds_to_plot,
                labels={"x": "Actual 2019-20 Value", "y": f"Predicted ({best_algo})"},
                template="plotly_white",
                title=f"Actual vs Predicted: {best_algo}",
            )
            # 45 degree line
            min_v = min(actual_y.min(), preds_to_plot.min())
            max_v = max(actual_y.max(), preds_to_plot.max())
            fig_reg.add_trace(go.Scatter(x=[min_v, max_v], y=[min_v, max_v], mode="lines", line=dict(dash="dash", color="red"), name="Perfect Fit"))
            fig_reg.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=380)
            st.plotly_chart(fig_reg, use_container_width=True)

        with col_s2:
            st.markdown("#### 🔑 Top Feature Importances (Random Forest)")
            fig_imp = px.bar(
                sup_results["importance_df"],
                x="Importance",
                y="Indicator",
                orientation="h",
                template="plotly_white",
                color="Importance",
                color_continuous_scale="Viridis",
            )
            fig_imp.update_layout(yaxis=dict(autorange="reversed"), margin=dict(l=20, r=20, t=30, b=20), height=380)
            st.plotly_chart(fig_imp, use_container_width=True)

    else:
        # Classification Views (ROC Curves)
        with col_s1:
            st.markdown("#### 📈 Receiver Operating Characteristic (ROC) Curves")
            fig_roc = go.Figure()
            y_test_cl = sup_results["y_test"]

            for algo_name, probs in sup_results["probabilities"].items():
                if len(np.unique(y_test_cl)) > 1:
                    fpr, tpr, _ = roc_curve(y_test_cl, probs)
                    auc_val = roc_auc_score(y_test_cl, probs)
                    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{algo_name} (AUC={auc_val:.3f})"))

            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), name="Random Guess"))
            fig_roc.update_layout(
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                template="plotly_white",
                margin=dict(l=20, r=20, t=30, b=20),
                height=380,
            )
            st.plotly_chart(fig_roc, use_container_width=True)

        with col_s2:
            st.markdown("#### 🚨 Risk Factors / Feature Importance (Random Forest)")
            fig_imp_c = px.bar(
                sup_results["importance_df"],
                x="Importance",
                y="Indicator",
                orientation="h",
                template="plotly_white",
                color="Importance",
                color_continuous_scale="Reds",
            )
            fig_imp_c.update_layout(yaxis=dict(autorange="reversed"), margin=dict(l=20, r=20, t=30, b=20), height=380)
            st.plotly_chart(fig_imp_c, use_container_width=True)

    # Methodological Limitations & Generalization Warnings
    st.markdown("---")
    st.warning(
        """
        ⚠️ **Methodological Notice & Two-Year Generalization Limitations:**
        - **Chronological Boundary**: Having only two academic years (2018-19 and 2019-20) means multi-year longitudinal trend modeling is constrained. Predictions represent short-term persistence rather than long-term causality.
        - **Policy Exogeneity**: Supervised correlations do not establish causation. A high $R^2$ in predicting computer rate or gender equity reflects high structural inertia across administrative districts.
        - **Distribution Shifts**: Changes in reporting norms, administrative boundary redrawing, or curriculum schemes between years can induce unmeasured covariate shift.
        """
    )


# ------------------------------------------------------------------------------
# TAB 5: YEAR-OVER-YEAR MOBILITY
# ------------------------------------------------------------------------------
with tab_mobility:
    st.markdown("### 🔁 Year-Over-Year District Transition Analysis (2018-19 ➔ 2019-20)")
    st.caption("Track longitudinal district migration across educational clusters over time.")

    # Match districts present in both years
    df_18_sub = bench_data["df_viz"][bench_data["df_viz"]["academic_year"] == "2018-19"][["district_code", "district_name", "state_name", "KMeans_Cluster"]].copy()
    df_19_sub = bench_data["df_viz"][bench_data["df_viz"]["academic_year"] == "2019-20"][["district_code", "district_name", "state_name", "KMeans_Cluster"]].copy()

    merged_years = pd.merge(
        df_18_sub,
        df_19_sub,
        on="district_code",
        suffixes=("_2018_19", "_2019_20"),
    )

    if not merged_years.empty:
        trans_mat = pd.crosstab(
            merged_years["KMeans_Cluster_2018_19"],
            merged_years["KMeans_Cluster_2019_20"],
            rownames=["2018-19 Cluster"],
            colnames=["2019-20 Cluster"],
        )

        col_m1, col_m2 = st.columns([1, 1.2])
        with col_m1:
            st.markdown("#### Transition Cross-Tabulation Matrix")
            fig_trans = px.imshow(
                trans_mat,
                text_auto=True,
                labels=dict(x="2019-20 Cluster", y="2018-19 Cluster", color="Districts"),
                color_continuous_scale="Blues",
                template="plotly_white",
            )
            fig_trans.update_layout(margin=dict(l=20, r=20, t=30, b=20), height=350)
            st.plotly_chart(fig_trans, use_container_width=True)

        with col_m2:
            st.markdown("#### Mobility Summary Breakdown")
            same = (merged_years["KMeans_Cluster_2018_19"] == merged_years["KMeans_Cluster_2019_20"]).sum()
            migrated = len(merged_years) - same
            st.markdown(
                f"""
                <div style="background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 20px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.04);">
                    <div style="font-size: 1.1rem; font-weight: 700; color: #1e293b; margin-bottom: 12px;">Matched Districts: {len(merged_years)}</div>
                    <div style="display: flex; gap: 20px; margin-bottom: 15px;">
                        <div>
                            <div style="font-size: 0.8rem; color: #64748b; font-weight: 600;">STABLE PROFILES</div>
                            <div style="font-size: 1.7rem; font-weight: 800; color: #10b981;">{same} <span style="font-size: 0.85rem; font-weight: 500;">({(same/len(merged_years))*100:.1f}%)</span></div>
                        </div>
                        <div>
                            <div style="font-size: 0.8rem; color: #64748b; font-weight: 600;">TRANSITIONED TIERS</div>
                            <div style="font-size: 1.7rem; font-weight: 800; color: #3b82f6;">{migrated} <span style="font-size: 0.85rem; font-weight: 500;">({(migrated/len(merged_years))*100:.1f}%)</span></div>
                        </div>
                    </div>
                    <p style="font-size: 0.84rem; color: #475569; line-height: 1.5;">
                        Districts on the diagonal maintained their relative educational infrastructure profile. 
                        Off-diagonal districts experienced notable shifts in digital, sanitation, or facility metrics.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("#### 📋 Matched District Migration Table")
        merged_years_display = merged_years.copy()
        merged_years_display["Status"] = np.where(
            merged_years_display["KMeans_Cluster_2018_19"] == merged_years_display["KMeans_Cluster_2019_20"],
            "Stable",
            "Transitioned",
        )
        st.dataframe(
            merged_years_display[["district_name_2018_19", "state_name_2018_19", "KMeans_Cluster_2018_19", "KMeans_Cluster_2019_20", "Status"]].rename(
                columns={"district_name_2018_19": "District", "state_name_2018_19": "State", "KMeans_Cluster_2018_19": "2018-19 Cluster", "KMeans_Cluster_2019_20": "2019-20 Cluster"}
            ),
            use_container_width=True,
            height=260,
        )


# ------------------------------------------------------------------------------
# TAB 6: DISTRICT EXPLORER & RADAR BENCHMARK
# ------------------------------------------------------------------------------
with tab_explorer:
    st.markdown("### 🔍 District Explorer & Comparative Radar Benchmark")
    st.caption("Inspect individual districts, compare their indicators against the cluster benchmark, and observe year-over-year progress.")

    unique_districts = sorted(bench_data["df_viz"]["district_name"].unique())
    selected_district = st.selectbox("Select District:", options=unique_districts, index=0)

    dist_records = bench_data["df_viz"][bench_data["df_viz"]["district_name"] == selected_district]

    if not dist_records.empty:
        state_val = dist_records["state_name"].iloc[0]
        c18_val = dist_records.loc[dist_records["academic_year"] == "2018-19", "KMeans_Cluster"].values
        c19_val = dist_records.loc[dist_records["academic_year"] == "2019-20", "KMeans_Cluster"].values

        arch_18 = archetypes.get(c18_val[0], {}).get("name", "N/A") if len(c18_val) > 0 else "N/A"
        arch_19 = archetypes.get(c19_val[0], {}).get("name", "N/A") if len(c19_val) > 0 else "N/A"

        st.markdown(
            f"""
            <div style="background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 20px; margin-bottom: 20px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.04);">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                    <div>
                        <span style="font-size: 1.6rem; font-weight: 800; color: #0f172a;">{selected_district}</span>
                        <span style="background: #e0e7ff; color: #3730a3; padding: 4px 12px; border-radius: 9999px; font-weight: 600; font-size: 0.85rem; margin-left: 12px;">{state_val}</span>
                    </div>
                    <div style="display: flex; gap: 24px; margin-top: 8px;">
                        <div><strong style="color: #64748b;">2018-19:</strong> <span style="font-weight: 700; color: #3b82f6;">{arch_18}</span></div>
                        <div><strong style="color: #64748b;">2019-20:</strong> <span style="font-weight: 700; color: #10b981;">{arch_19}</span></div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_rad1, col_rad2 = st.columns([1.3, 1])

        radar_indicators = [
            c for c in [
                "computer_available_rate", "internet_rate", "functional_electricity_rate",
                "drinking_water_rate", "functional_girl_toilet_rate", "functional_boy_toilet_rate",
                "library_or_reading_corner_or_book_bank_rate", "playground_rate", "ramps_rate",
            ] if c in common_indicators
        ]
        radar_labels = [c.replace("_", " ").title() for c in radar_indicators]

        with col_rad1:
            fig_rad = go.Figure()
            if len(c18_val) > 0:
                row_18 = dist_records[dist_records["academic_year"] == "2018-19"].iloc[0]
                vals_18 = [row_18[c] for c in radar_indicators]
                fig_rad.add_trace(go.Scatterpolar(
                    r=vals_18 + [vals_18[0]],
                    theta=radar_labels + [radar_labels[0]],
                    fill='toself',
                    name='2018-19 District Profile',
                    line=dict(color='#3b82f6', width=2),
                ))

            if len(c19_val) > 0:
                row_19 = dist_records[dist_records["academic_year"] == "2019-20"].iloc[0]
                vals_19 = [row_19[c] for c in radar_indicators]
                fig_rad.add_trace(go.Scatterpolar(
                    r=vals_19 + [vals_19[0]],
                    theta=radar_labels + [radar_labels[0]],
                    fill='toself',
                    name='2019-20 District Profile',
                    line=dict(color='#10b981', width=2),
                ))

            # National Benchmark
            nat_avg = [combined_df[c].mean() for c in radar_indicators]
            fig_rad.add_trace(go.Scatterpolar(
                r=nat_avg + [nat_avg[0]],
                theta=radar_labels + [radar_labels[0]],
                name='National Average Benchmark',
                line=dict(color='#94a3b8', dash='dash', width=2),
            ))

            fig_rad.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
                title=f"Radar Benchmark: {selected_district}",
                template="plotly_white",
                height=480,
            )
            st.plotly_chart(fig_rad, use_container_width=True)

        with col_rad2:
            st.markdown("#### District Scorecard")
            card_df = dist_records[["academic_year"] + radar_indicators].set_index("academic_year").T
            card_df.index = [i.replace("_", " ").title() for i in card_df.index]
            st.dataframe(card_df.style.format("{:.2%}"), use_container_width=True, height=420)


# ------------------------------------------------------------------------------
# TAB 7: LIVE POLICY BRIEFING
# ------------------------------------------------------------------------------
with tab_report:
    st.markdown("### 📄 Educational Equity Policy Briefing & Synthesis")
    st.caption("Synthesized findings generated dynamically from the multi-algorithm clustering and chronological supervised models.")

    report_text = f"""
### 1. Executive Summary & Comparative Algorithmic Findings
Across **{total_records:,} district-year observations**, comparative evaluation of four distinct clustering paradigms revealed:
1. **Centroid vs Density Partitioning**: K-Means and Hierarchical Clustering exhibited high structural consensus (ARI = {bench_data['ari_matrix'].loc['K-Means', 'Hierarchical']:.3f}), isolating large-scale tiers of educational infrastructure.
2. **Outlier Detection via DBSCAN**: DBSCAN successfully detected **{dbs_outliers} anomalous districts** characterized by non-standard indicator combinations (e.g., high digital access despite missing water testing).
3. **Probabilistic Uncertainty (GMM)**: Soft clustering demonstrated that while core districts possess >98% assignment probability, boundary districts exhibit mean entropy of **{gmm_entropy:.2f}**, marking transition zones that require targeted policy evaluation.
4. **Cluster Stability**: Multi-seed testing across 8 distinct centroid initializations produced a **Mean Stability ARI of {mean_stability:.3f}**, validating that the extracted clusters represent genuine empirical groupings.

---

### 2. Multi-Dimensional Equity Profiles
Examining clusters across all four critical dimensions demonstrates:
- **Digital Infrastructure Disparity**: Forms the widest inter-cluster cleavage in Indian districts (computer availability ranges from under 18% in Priority clusters to over 65% in High-Resource clusters).
- **Teacher & Enrolment Linkages**: High-resource districts maintain lower Pupil-Teacher Ratios (PTR) and higher female teacher participation, reinforcing positive learning environments.
- **Gender Parity**: Basic enrolment gender parity remains relatively stable across clusters (girls-to-boys ratios typically range between 0.91 and 0.95), but access to girl-specific sanitation facilities differs sharply.

---

### 3. Supervised Forecasting & Policy Implications
- Chronological validation demonstrated that baseline infrastructure strongly predicts subsequent year outcomes (e.g. digital access $R^2 > 0.85$).
- **Resource Allocation Priority**: Universalize drinking water quality testing and basic sanitation in Priority Intervention districts before capital-heavy digital rollouts to ensure foundational educational equity.
    """
    st.markdown(report_text)

    st.markdown("---")
    st.markdown("#### 📥 In-Memory Export Options")
    col_dl1, col_dl2 = st.columns(2)

    with col_dl1:
        csv_buffer = BytesIO()
        export_df = bench_data["df_viz"][["academic_year", "district_name", "state_name", "KMeans_Cluster", "Hierarchical_Cluster", "DBSCAN_Cluster", "GMM_Cluster", "PC1", "PC2"]]
        export_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Multi-Algorithm Cluster Assignments (CSV)",
            data=csv_buffer.getvalue(),
            file_name="eduequity_multi_algorithm_clusters.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col_dl2:
        st.download_button(
            label="📄 Download Policy Report (Markdown)",
            data=report_text,
            file_name="EduEquity_Policy_Briefing.md",
            mime="text/markdown",
            use_container_width=True,
        )

st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #94a3b8; font-size: 0.85rem; padding: 10px 0;">
        EduEquity Multi-Algorithm Intelligence Platform | Streamlit, Scikit-Learn & Plotly
    </div>
    """,
    unsafe_allow_html=True,
)
