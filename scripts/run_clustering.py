
# ============================================================
# EduEquity: Two-Year District Clustering Analysis
# Academic years: 2018-19 and 2019-20
# File: scripts/run_clustering.py
# ============================================================

from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid")

# ------------------------------------------------------------
# 1. PATHS AND SETTINGS
# ------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

# Support either folder spelling.
if (PROJECT_DIR / "data" / "processed").exists():
    DATA_DIR = PROJECT_DIR / "data" / "processed"
elif (PROJECT_DIR / "data" / "processing").exists():
    DATA_DIR = PROJECT_DIR / "data" / "processing"
else:
    raise FileNotFoundError(
        "Could not find data\\processed or data\\processing."
    )

INPUT_FILES = {
    "2018-19": "eduequity_model_dataset_2018_19.csv",
    "2019-20": "eduequity_model_dataset_2019_20.csv",
}

# Separate output folder preserves the results from the old analysis.
OUTPUT_DIR = (
    PROJECT_DIR / "reports" / "ml_clustering_2018_19_2019_20"
)
PLOT_DIR = OUTPUT_DIR / "plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
MAX_CLUSTERS = 10


# ------------------------------------------------------------
# 2. HELPER FUNCTIONS
# ------------------------------------------------------------

def normalize_name(value):
    """Normalize a column name for comparison."""
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def find_district_column(df):
    """Find a likely district identifier column."""
    preferred = [
        "district",
        "district_name",
        "districtname",
        "district_id",
        "district_code",
    ]

    normalized = {
        normalize_name(col): col for col in df.columns
    }

    for candidate in preferred:
        if candidate in normalized:
            return normalized[candidate]

    for col in df.columns:
        if "district" in normalize_name(col):
            return col

    return None


def is_identifier_column(column):
    """Exclude identifiers and non-indicator numeric fields."""
    name = normalize_name(column)

    excluded_names = {
        "district", "district_name", "district_id",
        "district_code", "state", "state_name",
        "state_id", "state_code", "year", "academic_year",
        "financial_year", "serial_no", "index",
        "latitude", "longitude", "lat", "lon", "long",
        "pincode", "pin_code",
    }

    return (
        name in excluded_names
        or name.startswith("unnamed")
        or name.endswith("_id")
        or name.endswith("_code")
    )


def load_year_dataset(year, filename):
    """Load one model dataset and retain its year and district labels."""
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Missing input file:\n{path}\n"
            f"Check the filename and data folder."
        )

    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    if df.empty:
        raise ValueError(f"{filename} contains no data.")

    district_col = find_district_column(df)

    if district_col is not None:
        district_labels = (
            df[district_col]
            .fillna("Unknown district")
            .astype(str)
            .str.strip()
        )
    else:
        district_labels = pd.Series(
            [f"Row_{i + 1}" for i in range(len(df))],
            index=df.index,
        )
        print(
            f"WARNING: No district column detected in {filename}. "
            "Row labels will be used instead."
        )

    # Record original columns and their data types.
    column_inventory = pd.DataFrame({
        "academic_year": year,
        "source_file": filename,
        "column": df.columns,
        "dtype": [str(df[c].dtype) for c in df.columns],
        "missing_count": [
            int(df[c].isna().sum()) for c in df.columns
        ],
    })

    # Select numeric data, excluding identifiers.
    numeric = df.select_dtypes(include=[np.number]).copy()

    numeric = numeric.drop(
        columns=[
            col for col in numeric.columns
            if is_identifier_column(col)
        ],
        errors="ignore",
    )

    numeric = numeric.replace([np.inf, -np.inf], np.nan)

    # Normalize feature names so harmless spacing/case differences
    # do not prevent a match. If normalization creates duplicates,
    # stop and ask for manual resolution.
    rename_map = {
        col: normalize_name(col) for col in numeric.columns
    }
    numeric = numeric.rename(columns=rename_map)

    if numeric.columns.duplicated().any():
        duplicated = numeric.columns[
            numeric.columns.duplicated()
        ].tolist()
        raise ValueError(
            f"Duplicate normalized feature names in {filename}: "
            f"{duplicated}. Rename those columns before proceeding."
        )

    # Remove empty and constant features within this dataset.
    numeric = numeric.loc[:, numeric.notna().any(axis=0)]
    numeric = numeric.loc[
        :, numeric.nunique(dropna=True) > 1
    ]

    numeric.index = df.index
    numeric["district_label"] = district_labels
    numeric["academic_year"] = year

    print(f"\nLoaded {filename}")
    print(f"  Academic year: {year}")
    print(f"  Rows: {len(df)}")
    print(f"  District column: {district_col}")
    print(f"  Numeric indicators available: {len(numeric.columns) - 2}")

    return numeric, column_inventory, df.columns.tolist()


def save_plot():
    plt.tight_layout()
    plt.savefig(
        current_plot_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


# ------------------------------------------------------------
# 3. LOAD BOTH YEARS
# ------------------------------------------------------------

def main():
    print("=" * 65)
    print("EduEquity: 2018-19 vs 2019-20 Clustering")
    print("=" * 65)
    print(f"Input folder: {DATA_DIR}")
    print(f"Output folder: {OUTPUT_DIR}")

    datasets = {}
    inventories = {}
    original_columns = {}

    for year, filename in INPUT_FILES.items():
        data, inventory, columns = load_year_dataset(
            year, filename
        )
        datasets[year] = data
        inventories[year] = inventory
        original_columns[year] = columns

    # Save dataset-column inventory.
    pd.concat(
        inventories.values(), ignore_index=True
    ).to_csv(
        OUTPUT_DIR / "dataset_columns_by_year.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 4. ALIGN FEATURES ACROSS YEARS
    # --------------------------------------------------------

    feature_sets = {
        year: set(
            col for col in data.columns
            if col not in {"district_label", "academic_year"}
        )
        for year, data in datasets.items()
    }

    common_features = sorted(
        set.intersection(*feature_sets.values())
    )

    all_features = sorted(
        set.union(*feature_sets.values())
    )

    # Save which columns are shared and which are year-specific.
    feature_rows = []

    for feature in all_features:
        present_years = [
            year for year in INPUT_FILES
            if feature in feature_sets[year]
        ]

        feature_rows.append({
            "normalized_feature": feature,
            "present_in_2018_19": feature in feature_sets["2018-19"],
            "present_in_2019_20": feature in feature_sets["2019-20"],
            "used_for_clustering": feature in common_features,
            "reason": (
                "Shared numeric indicator"
                if feature in common_features
                else "Not present as a usable numeric indicator in both years"
            ),
        })

    pd.DataFrame(feature_rows).to_csv(
        OUTPUT_DIR / "feature_comparison_by_year.csv",
        index=False,
    )

    if len(common_features) < 2:
        raise ValueError(
            f"Only {len(common_features)} common usable numeric "
            "features were found across the two datasets. At least "
            "two are needed for this analysis.\n"
            "Review feature_comparison_by_year.csv. Check whether "
            "equivalent indicators use different names or units."
        )

    print("\nCommon features selected for comparison:")
    for feature in common_features:
        print(f"  - {feature}")

    print(f"\nNumber of shared features: {len(common_features)}")

    # Keep the common columns and align their order.
    # Rows with all common indicators missing are removed.
    year_frames = []

    for year, data in datasets.items():
        frame = data[common_features].copy()
        frame["district_label"] = data["district_label"].values
        frame["academic_year"] = year

        valid = frame[common_features].notna().any(axis=1)
        frame = frame.loc[valid].copy()

        year_frames.append(frame)

    combined = pd.concat(year_frames, ignore_index=True)

    if len(combined) < 3:
        raise ValueError(
            "Fewer than three usable district-year rows remain."
        )

    combined.to_csv(
        OUTPUT_DIR / "combined_two_year_common_features.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 5. IMPUTE MISSING VALUES AND SCALE
    # --------------------------------------------------------

    X = combined[common_features].replace(
        [np.inf, -np.inf], np.nan
    )

    # Remove features that became entirely missing after combining.
    X = X.dropna(axis=1, how="all")
    final_features = X.columns.tolist()

    # Remove features with no variation across the combined data.
    final_features = [
        col for col in final_features
        if X[col].nunique(dropna=True) > 1
    ]
    X = X[final_features]

    if len(final_features) < 2:
        raise ValueError(
            "Fewer than two usable features remain after checking "
            "missing values and variation."
        )

    imputer = SimpleImputer(strategy="median")
    X_imputed_array = imputer.fit_transform(X)

    X_imputed = pd.DataFrame(
        X_imputed_array,
        columns=final_features,
    )

    scaler = StandardScaler()
    X_scaled_array = scaler.fit_transform(X_imputed)

    X_scaled = pd.DataFrame(
        X_scaled_array,
        columns=final_features,
    )

    X_scaled.to_csv(
        OUTPUT_DIR / "scaled_common_features.csv",
        index=False,
    )

    pd.DataFrame({
        "feature": final_features,
        "missing_values_before_imputation": [
            int(X[col].isna().sum()) for col in final_features
        ],
        "median_used_for_imputation": imputer.statistics_,
        "mean_after_scaling": X_scaled_array.mean(axis=0),
        "std_after_scaling": X_scaled_array.std(axis=0),
    }).to_csv(
        OUTPUT_DIR / "feature_scaling_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 6. ELBOW AND SILHOUETTE ANALYSIS
    # --------------------------------------------------------

    max_k = min(MAX_CLUSTERS, len(X_scaled) - 1)
    if max_k < 2:
        raise ValueError("Not enough rows to evaluate K >= 2.")

    k_values = list(range(1, max_k + 1))
    inertias = []
    silhouette_results = []

    print("\nEvaluating cluster counts...")

    for k in k_values:
        model = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=10,
        )
        labels = model.fit_predict(X_scaled)
        inertias.append(float(model.inertia_))

        if k >= 2 and len(np.unique(labels)) >= 2:
            score = silhouette_score(X_scaled, labels)
            silhouette_results.append({
                "k": k,
                "silhouette_score": float(score),
            })
            print(
                f"  K={k}: inertia={model.inertia_:.3f}, "
                f"silhouette={score:.4f}"
            )
        else:
            print(f"  K={k}: inertia={model.inertia_:.3f}")

    silhouette_df = pd.DataFrame(silhouette_results)

    evaluation = pd.DataFrame({
        "k": k_values,
        "inertia": inertias,
    }).merge(
        silhouette_df, on="k", how="left"
    )

    evaluation.to_csv(
        OUTPUT_DIR / "cluster_evaluation.csv",
        index=False,
    )

    global current_plot_path

    current_plot_path = PLOT_DIR / "elbow_method.png"
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, inertias, marker="o")
    plt.title("Elbow Method: Combined 2018-19 and 2019-20 Data")
    plt.xlabel("Number of clusters (K)")
    plt.ylabel("Inertia")
    plt.xticks(k_values)
    save_plot()

    if silhouette_df.empty:
        raise ValueError(
            "Could not calculate a silhouette score. "
            "Check the number of rows and feature values."
        )

    current_plot_path = PLOT_DIR / "silhouette_analysis.png"
    plt.figure(figsize=(8, 5))
    plt.plot(
        silhouette_df["k"],
        silhouette_df["silhouette_score"],
        marker="o",
    )
    plt.title("Silhouette Analysis: Combined Two-Year Data")
    plt.xlabel("Number of clusters (K)")
    plt.ylabel("Silhouette score")
    plt.xticks(silhouette_df["k"])
    save_plot()

    best_k = int(
        silhouette_df.loc[
            silhouette_df["silhouette_score"].idxmax(), "k"
        ]
    )

    print(f"\nSelected K by silhouette score: {best_k}")
    print(
        "Review the Elbow chart too; the highest silhouette score "
        "is a guide, not an automatic guarantee of the best model."
    )

    # --------------------------------------------------------
    # 7. FIT THE FINAL COMBINED K-MEANS MODEL
    # --------------------------------------------------------

    final_model = KMeans(
        n_clusters=best_k,
        random_state=RANDOM_STATE,
        n_init=10,
    )

    labels = final_model.fit_predict(X_scaled)

    results = combined.loc[:, [
        "academic_year", "district_label"
    ]].copy()

    results["cluster"] = labels

    # Include common feature values for interpretation.
    for col in final_features:
        results[col] = X_imputed[col].values

    results.to_csv(
        OUTPUT_DIR / "district_year_cluster_assignments.csv",
        index=False,
    )

    # Cluster counts by year.
    counts = (
        results.groupby(["academic_year", "cluster"])
        .size()
        .reset_index(name="district_year_rows")
    )

    counts.to_csv(
        OUTPUT_DIR / "cluster_counts_by_year.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 8. CLUSTER PROFILES
    # --------------------------------------------------------

    mean_profiles = (
        results.groupby(["academic_year", "cluster"])[final_features]
        .mean()
        .reset_index()
    )

    median_profiles = (
        results.groupby(["academic_year", "cluster"])[final_features]
        .median()
        .reset_index()
    )

    mean_profiles.to_csv(
        OUTPUT_DIR / "cluster_profiles_by_year_means.csv",
        index=False,
    )
    median_profiles.to_csv(
        OUTPUT_DIR / "cluster_profiles_by_year_medians.csv",
        index=False,
    )

    # Overall combined standardized profile heatmap.
    profile_scaled = X_scaled.copy()
    profile_scaled["cluster"] = labels
    heatmap_data = profile_scaled.groupby("cluster").mean()

    current_plot_path = PLOT_DIR / "cluster_profile_heatmap.png"
    plt.figure(
        figsize=(
            max(10, len(final_features) * 0.7),
            max(5, best_k * 0.8),
        )
    )
    sns.heatmap(
        heatmap_data,
        cmap="vlag",
        center=0,
        annot=(len(final_features) <= 12),
        fmt=".2f",
    )
    plt.title("Combined Cluster Profiles (Standardized Features)")
    plt.xlabel("Common educational indicators")
    plt.ylabel("Cluster")
    save_plot()

    # --------------------------------------------------------
    # 9. PCA SCATTER PLOT
    # --------------------------------------------------------

    pca = PCA(n_components=2)
    coordinates = pca.fit_transform(X_scaled)

    pca_df = results[[
        "academic_year", "district_label", "cluster"
    ]].copy()
    pca_df["PC1"] = coordinates[:, 0]
    pca_df["PC2"] = coordinates[:, 1]

    pca_df.to_csv(
        OUTPUT_DIR / "pca_coordinates.csv",
        index=False,
    )

    current_plot_path = PLOT_DIR / "cluster_pca_scatter.png"
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=pca_df,
        x="PC1",
        y="PC2",
        hue="cluster",
        style="academic_year",
        palette="tab10",
        s=75,
    )

    variance = pca.explained_variance_ratio_ * 100
    plt.title("District-Year Clusters: PCA Projection")
    plt.xlabel(f"PC1 ({variance[0]:.1f}% variance explained)")
    plt.ylabel(f"PC2 ({variance[1]:.1f}% variance explained)")
    save_plot()


    # --------------------------------------------------------
    # 10. TEN ADDITIONAL VISUALIZATIONS (plots 5-14)
    # --------------------------------------------------------

    # Select a manageable set of informative features for plots.
    feature_variance = X_imputed.var().sort_values(ascending=False)
    top_features = feature_variance.head(min(10, len(feature_variance))).index.tolist()
    top_six_features = top_features[:min(6, len(top_features))]
    top_four_features = top_features[:min(4, len(top_features))]
    top_fifteen_features = feature_variance.head(min(15, len(feature_variance))).index.tolist()

    # Plot 5: Cluster counts split by academic year.
    current_plot_path = PLOT_DIR / "05_cluster_counts_by_year.png"
    count_pivot = counts.pivot(
        index="cluster", columns="academic_year",
        values="district_year_rows"
    ).fillna(0)
    ax = count_pivot.plot(kind="bar", figsize=(9, 5))
    ax.set_title("District-Year Records in Each Cluster by Academic Year")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of district-year records")
    ax.legend(title="Academic year")
    plt.xticks(rotation=0)
    save_plot()

    # Plot 6: Overall cluster sizes.
    current_plot_path = PLOT_DIR / "06_overall_cluster_sizes.png"
    cluster_sizes = results["cluster"].value_counts().sort_index()
    plt.figure(figsize=(8, 5))
    sns.barplot(x=cluster_sizes.index.astype(str), y=cluster_sizes.values)
    plt.title("Overall Cluster Sizes (Both Academic Years)")
    plt.xlabel("Cluster")
    plt.ylabel("Number of district-year records")
    save_plot()

    # Plot 7: Most variable indicators in the combined dataset.
    current_plot_path = PLOT_DIR / "07_top_feature_variance.png"
    variance_to_plot = feature_variance.head(min(12, len(feature_variance))).sort_values()
    plt.figure(figsize=(10, max(5, len(variance_to_plot) * 0.35)))
    variance_to_plot.plot(kind="barh")
    plt.title("Indicators with the Highest Variance")
    plt.xlabel("Variance (original indicator units)")
    plt.ylabel("Indicator")
    save_plot()

    # Plot 8: Mean indicator values by year for the most variable features.
    current_plot_path = PLOT_DIR / "08_feature_means_by_year.png"
    year_means = combined.groupby("academic_year")[top_features].mean().T
    plt.figure(figsize=(11, max(5, len(top_features) * 0.4)))
    year_means.plot(kind="bar", ax=plt.gca())
    plt.title("Mean Values of Selected Indicators by Academic Year")
    plt.xlabel("Indicator")
    plt.ylabel("Mean value")
    plt.xticks(rotation=60, ha="right")
    plt.legend(title="Academic year")
    save_plot()

    # Plot 9: Correlation heatmap for a manageable set of indicators.
    current_plot_path = PLOT_DIR / "09_feature_correlation_heatmap.png"
    corr = X_imputed[top_fifteen_features].corr()
    plt.figure(figsize=(max(8, len(top_fifteen_features) * 0.65),
                        max(6, len(top_fifteen_features) * 0.55)))
    sns.heatmap(
        corr, cmap="coolwarm", center=0, vmin=-1, vmax=1,
        annot=(len(top_fifteen_features) <= 10), fmt=".2f"
    )
    plt.title("Correlation Between Selected Educational Indicators")
    save_plot()

    # Plot 10: Indicator distributions by academic year.
    current_plot_path = PLOT_DIR / "10_feature_distributions_by_year.png"
    distribution_df = X_imputed[top_six_features].copy()
    distribution_df["academic_year"] = combined["academic_year"].values
    melted = distribution_df.melt(
        id_vars="academic_year", var_name="indicator", value_name="value"
    )
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=melted, x="indicator", y="value", hue="academic_year")
    plt.title("Selected Indicator Distributions by Academic Year")
    plt.xlabel("Indicator")
    plt.ylabel("Value after median imputation")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Academic year")
    save_plot()


    # Plot 12: Cluster-wise distributions for selected indicators.
    current_plot_path = PLOT_DIR / "12_indicator_distributions_by_cluster.png"
    cluster_box_df = results[["cluster"] + top_four_features].melt(
        id_vars="cluster", var_name="indicator", value_name="value"
    )
    plt.figure(figsize=(12, 6))
    sns.boxplot(
        data=cluster_box_df, x="indicator", y="value", hue="cluster"
    )
    plt.title("Selected Indicator Distributions Across Clusters")
    plt.xlabel("Indicator")
    plt.ylabel("Median-imputed indicator value")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Cluster")
    save_plot()

    # Plot 13: Standardized cluster profiles split by academic year.
    current_plot_path = PLOT_DIR / "13_cluster_profiles_by_year_heatmap.png"
    scaled_by_year = X_scaled.copy()
    scaled_by_year["academic_year"] = combined["academic_year"].values
    scaled_by_year["cluster"] = labels
    year_cluster_profiles = scaled_by_year.groupby(
        ["academic_year", "cluster"]
    )[final_features].mean()
    plt.figure(figsize=(max(10, len(final_features) * 0.55),
                        max(5, len(year_cluster_profiles) * 0.45)))
    sns.heatmap(
        year_cluster_profiles, cmap="vlag", center=0,
        annot=(len(final_features) <= 10), fmt=".2f"
    )
    plt.title("Standardized Cluster Profiles by Academic Year")
    plt.xlabel("Indicator")
    plt.ylabel("Academic year, cluster")
    save_plot()

    # Plot 14: PCA explained variance for the two displayed components.
    current_plot_path = PLOT_DIR / "14_pca_explained_variance.png"
    component_names = ["PC1", "PC2"]
    explained = pca.explained_variance_ratio_ * 100
    plt.figure(figsize=(7, 5))
    bars = plt.bar(component_names, explained)
    plt.title("Variance Explained by PCA Components")
    plt.xlabel("Principal component")
    plt.ylabel("Explained variance (%)")
    for bar, value in zip(bars, explained):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.1f}%",
            ha="center", va="bottom"
        )
    save_plot()

    # --------------------------------------------------------
    # 11. WRITE THE REPORT
    # --------------------------------------------------------

    report_path = OUTPUT_DIR / "EduEquity_Two_Year_Clustering_Report.md"

    lines = [
        "# EduEquity: Two-Year Clustering Report",
        "",
        "## 1. Objective",
        "",
        "Compare district-year records from 2018-19 and 2019-20 "
        "using shared numeric educational indicators and one combined "
        "K-Means model.",
        "",
        "## 2. Input datasets",
        "",
        f"- 2018-19: `{INPUT_FILES['2018-19']}`",
        f"- 2019-20: `{INPUT_FILES['2019-20']}`",
        f"- Input directory: `{DATA_DIR}`",
        f"- Usable district-year rows: {len(results)}",
        "",
        "## 3. Feature alignment",
        "",
        "Features were matched by normalized column names. Only numeric "
        "indicators present and usable in both datasets were used for "
        "the main two-year comparison. See `feature_comparison_by_year.csv` "
        "for shared and year-specific fields.",
        "",
        "Selected features:",
    ]

    lines.extend([f"- `{c}`" for c in final_features])

    lines.extend([
        "",
        "## 4. Preprocessing",
        "",
        "- Missing values were median-imputed across the combined data.",
        "- Common indicators were standardized using StandardScaler.",
        "- See `feature_scaling_summary.csv` for preprocessing details.",
        "",
        "## 5. Number of clusters",
        "",
        f"- Selected K by maximum silhouette score: {best_k}",
        "- Review the Elbow and Silhouette charts before finalizing K.",
        "",
        evaluation.to_markdown(index=False),
        "",
        "## 6. Cluster counts by year",
        "",
        counts.to_markdown(index=False),
        "",
        "## 7. Cluster profiles by year",
        "",
        "See `cluster_profiles_by_year_means.csv` and "
        "`cluster_profiles_by_year_medians.csv` to compare the "
        "indicators in each cluster for the two years.",
        "",
        "## 8. Interpretation cautions",
        "",
        "A cluster is a statistical grouping, not a ranking of district "
        "quality. Whether a high or low indicator is desirable depends "
        "on its definition and units. For example, a higher dropout rate "
        "may signal a challenge, while higher facility access may signal "
        "better provision. Clustering alone does not establish causation.",
        "",
        "The combined model uses the same features and scaling approach "
        "for both years, making cluster labels more comparable than "
        "independently fitted models. Nevertheless, verify district "
        "identity, indicator definitions, and data consistency.",
        "",
        "## 9. Main outputs",
        "",
        "- `district_year_cluster_assignments.csv`",
        "- `cluster_counts_by_year.csv`",
        "- `cluster_profiles_by_year_means.csv`",
        "- `cluster_profiles_by_year_medians.csv`",
        "- `cluster_evaluation.csv`",
        "- `feature_comparison_by_year.csv`",
        "- `feature_scaling_summary.csv`",
        "- `combined_two_year_common_features.csv`",
        "- `pca_coordinates.csv`",
        "- `plots/`",
        "",
    ])

    report_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("\n" + "=" * 65)
    print("TWO-YEAR CLUSTERING COMPLETED")
    print("=" * 65)
    print(f"Selected K: {best_k}")
    print(f"Report: {report_path}")
    print(f"Output folder: {OUTPUT_DIR}")
    print(f"Charts folder: {PLOT_DIR}")


if __name__ == "__main__":
    main()
