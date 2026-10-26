# EduEquity: Multi-Algorithm District Clustering, Stability Validation & Supervised Forecasting

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit 1.50](https://img.shields.io/badge/Streamlit-1.50-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.6+-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive-3F4F75?style=flat&logo=plotly&logoColor=white)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## 📌 Project Abstract (Pre-Implementation Proposal)

> **Title:** *EduEquity: A Multi-Algorithm Machine Learning Framework for Educational Equity Assessment, District Cluster Stability, and Chronological Supervised Forecasting*  
> **Course / Academic Level:** Semester 5 — Machine Learning Lab / Capstone Project  
> **Author / Investigator:** Academic ML Research Group  

### Abstract
Geographical and socio-economic disparities in basic school infrastructure, digital learning access, and teacher allocation pose persistent challenges to educational equity across India's administrative districts. While aggregate state-level metrics often mask acute localized inequalities, machine learning offers a data-driven paradigm to uncover latent district groupings, evaluate structural resilience, and forecast equity outcomes. 

This project proposes a comprehensive, multi-algorithm machine learning framework to analyze educational equity across more than 700 Indian districts using multi-year UDISE-derived datasets (Academic Years 2018–19 and 2019–20). Rather than relying on a single heuristic, the proposed methodology benchmarks four distinct clustering paradigms on an identical prepared feature space: **K-Means** (centroid partitioning), **Hierarchical Agglomerative Clustering** (bottom-up structural trees), **DBSCAN** (density-based core identification and outlier isolation), and **Gaussian Mixture Models** (probabilistic soft clustering with boundary entropy quantification). 

To ensure clustering validity, the framework incorporates a multi-seed sensitivity protocol evaluating pairwise Adjusted Rand Index (ARI) consensus across varying centroid initializations. To avoid arbitrary label designations, clusters will be empirically profiled across four multidimensional axes: (1) physical and digital infrastructure provision, (2) student enrolment scale, (3) gender parity ratios, and (4) teacher availability metrics (Pupil-Teacher Ratios and female teacher representation). Furthermore, the study introduces a chronological supervised forecasting pipeline, training regression (Linear, Ridge, Random Forest, Gradient Boosting) and classification models on historical 2018–19 baselines to predict out-of-time 2019–20 outcomes while explicitly addressing the econometric limitations of two-year temporal generalization. Finally, the entire analytical pipeline is operationalized within an interactive, zero-disk-write Streamlit web intelligence platform, enabling policy analysts to dynamically explore district transitions, radar benchmarks, and resource allocation priorities.

**Keywords:** *Educational Equity, Machine Learning, K-Means, Hierarchical Clustering, DBSCAN, Gaussian Mixture Models, Cluster Stability, Chronological Validation, UDISE, Streamlit.*

---

## 🎯 Motivation & Research Objectives

1. **Comparative Algorithmic Benchmarking**: Evaluate how centroid-based, tree-based, density-based, and mixture models partition educational indicators, measuring consensus using Silhouette, Davies-Bouldin, Calinski-Harabasz, and Adjusted Rand Indices.
2. **Cluster Stability & Boundary Validation**: Test whether district clusters are resilient structural phenomena or random initialization artifacts through multi-seed permutation tests.
3. **Data-Driven Archetype Characterization**: Refrain from treating cluster tokens (`0`, `1`, `2`) as rankings; instead, derive meaningful policy archetypes (*e.g., Digital Frontrunners, Basic Provision with Digital Deficit, Priority Intervention Districts*) based on empirical percentile distributions.
4. **Multidimensional Equity Integration**: Profile districts beyond basic facilities by synthesizing infrastructure rates with enrolment volumes, gender parity (`girls_to_boys_ratio`), and teacher availability (PTR and female teacher ratios).
5. **Chronological Supervised Forecasting**: Formulate leak-free predictive models that use baseline year ($t$) features to forecast subsequent year ($t+1$) equity indicators and risk categories.
6. **Zero-Disk-Write Interactive Visualization**: Eliminate static report dumps by serving all data visualizations, 2D/3D PCA spatial projections, and radar benchmarks via an in-memory Streamlit application.

---

## 🗂️ Data Architecture & Dimensions

The framework processes unified district-level records from India's Unified District Information System for Education (UDISE):

| Dimension | Description & Key Indicators | Records / Years |
| :--- | :--- | :--- |
| **Physical & Digital Infrastructure** | 30 normalized rate indicators ($0.0 - 1.0$), including `computer_available_rate`, `internet_rate`, `drinking_water_rate`, `functional_electricity_rate`, `functional_boy_toilet_rate`, `functional_girl_toilet_rate`, `ramps_rate`, `library_rate`, `water_tested_rate`. | 651 districts (2018–19)<br>732 districts (2019–20)<br>**Total: 1,383 District Records** |
| **Enrolment Scale** | `total_boys`, `total_girls`, `total_enrolment`, grade-wise distribution. | Matched across both academic years |
| **Gender Equity** | `girls_to_boys_ratio`, `girls_enrolment_share`. | Continuous metric ($0.80 - 1.05+$) |
| **Teacher Allocation** | `Total_Teacher`, `Total_Male`, `Total_Female`, Pupil-Teacher Ratio (PTR), Female Teacher Share. | Aggregated from 400MB+ teacher census |

---

## 🔬 Machine Learning Pipeline & Methodology

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Raw UDISE Data Sources                          │
│     (Infrastructure 18-19/19-20, Enrolment multi-year, Teacher Census) │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   In-Memory Preprocessing & Imputation                 │
│         - Median SimpleImputer for missing indicator rates             │
│         - StandardScaler (Z-Score) & MinMaxScaler normalization        │
│         - Identifier & metadata exclusion                              │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
          ┌─────────────────────────┴─────────────────────────┐
          ▼                                                   ▼
┌───────────────────────────────────┐       ┌───────────────────────────────────┐
│     Unsupervised Benchmarking     │       │   Chronological Supervised ML     │
├───────────────────────────────────┤       ├───────────────────────────────────┤
│ • K-Means (Centroid Partitioning) │       │ • Baseline Features (2018-19)     │
│ • Hierarchical (Agglomerative)    │       │ • Holdout Future Target (2019-20) │
│ • DBSCAN (Density & Outliers)     │       │ • Regression (Linear, Ridge, RF,  │
│ • GMM (Soft Probability & Entropy)│       │   Gradient Boosting)              │
│ • Multi-Seed Stability (ARI)      │       │ • Classification (Risk Status via │
│ • 4D Cluster Profiling            │       │   Logistic, RF, HistGradient)     │
└─────────────────┬─────────────────┘       └─────────────────┬─────────────────┘
                  │                                           │
                  └─────────────────────┬─────────────────────┘
                                        │
                                        ▼
┌────────────────────────────────────────────────────────────────────────┐
│                 Streamlit Dynamic Web Intelligence App                 │
│         - Zero disk writes (client-side in-memory byte streams)        │
│         - Interactive 2D/3D PCA & Radar visualizers                    │
│         - Dynamic policy briefing generation                           │
└────────────────────────────────────────────────────────────────────────┘
```

### 1. Unsupervised Clustering Algorithms
- **K-Means**: Partitions observations into $K$ Voronoi cells minimizing within-cluster inertia $\sum_{i=1}^{k} \sum_{x \in S_i} ||x - \mu_i||^2$.
- **Hierarchical Agglomerative Clustering**: Constructs bottom-up linkage dendrograms using Ward's minimum variance criterion.
- **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**: Groups core samples with minimum neighbor density (`min_samples`) within radius $\epsilon$, isolating anomalous districts as noise ($label = -1$).
- **Gaussian Mixture Models (GMM)**: Expectation-Maximization (EM) algorithm fitting a mixture of Gaussian distributions, calculating posterior assignment probabilities and classification entropy $H(x) = -\sum p_k \log p_k$.

### 2. Validation & Stability Testing
- **Internal Metrics**: Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index.
- **Cross-Algorithm Consensus**: Pairwise Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI) matrix.
- **Multi-Seed Stability**: Permutation across 8 random seeds with pairwise ARI consensus scoring.

### 3. Chronological Supervised Forecasting
- **Split Strategy**: Chronological Holdout (Train on 2018–19 baseline $\rightarrow$ Test on 2019–20 future outcomes).
- **Regression Tasks**: Predict future Gender Equity (`girls_to_boys_ratio_19`) or Digital Readiness (`computer_available_rate_19`) using Linear Regression, Ridge ($L_2$), Random Forest, and Gradient Boosting.
- **Classification Tasks**: Predict documented At-Risk / Priority Equity district status using Logistic Regression, Random Forest, and Gradient Boosting.
- **Generalization Caveats**: Explicit documentation of two-year temporal limitations and policy non-causality.

---

## 💻 Streamlit Web Application Features

The interactive dashboard in `app.py` features 7 dedicated functional modules:

1. **📊 Executive Overview & Indicators**: High-level KPIs, 2018–19 vs. 2019–20 cluster distributions, common feature inventory.
2. **🧪 Algorithm Comparison**: Unified benchmark matrix comparing K-Means, Hierarchical, DBSCAN, and GMM, plus pairwise ARI agreement heatmap and GMM boundary entropy inspection.
3. **🛡️ Cluster Stability & Seeds**: 8-seed sensitivity convergence curves and stability consensus verdict.
4. **🏛️ Multi-Dimensional Profiling**: Empirical archetype naming, composite readiness scoring, and multidimensional bar charts (Enrolment, PTR, Gender Parity, Digital Access).
5. **🔮 Supervised ML & Forecasting**: Model comparison scorecards ($R^2$, RMSE, MAE / Accuracy, F1, ROC-AUC), actual vs. predicted regression plots, ROC curves, and feature importance rankings.
6. **🔁 Year-Over-Year Mobility**: Matched district transition matrix heatmap tracking district mobility across tiers.
7. **🔍 District Explorer & Policy Briefing**: Searchable district inspection with interactive polar radar benchmark charts against national averages, plus in-memory report export.

---

## 📁 Repository Structure

```text
EduEquity/
├── app.py                           # Main Streamlit interactive web application
├── README.md                        # Project documentation and prospective abstract
├── .gitignore                       # Git ignore rules for virtualenvs and raw data
├── data/
│   └── processed/                   # Cleaned and processed model datasets
│       ├── eduequity_model_dataset_2018_19.csv
│       ├── eduequity_model_dataset_2019_20.csv
│       ├── district_year_enrolment.csv
│       └── district_teacher_summary.csv
├── reports/
│   └── ml_clustering_2018_19_2019_20/ # Archival static analysis and markdown report
│       ├── EduEquity_Two_Year_Clustering_Report.md
│       └── plots/
├── scripts/
│   ├── run_clustering.py            # Legacy command-line clustering pipeline
│   └── inspect_remaining_datasets.py
└── src/
    └── eduequity/
        ├── data/                    # Dataset processing and cleaning utilities
        └── validation/              # Data consistency and schema checkers
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.9+ (Windows, macOS, or Linux)
- Recommended: Virtual environment (`venv` or `conda`)

### 2. Installation
Clone the repository and install the dependencies:

```bash
git clone https://github.com/Suvekha/EduEquity.git
cd EduEquity
```

Ensure standard machine learning libraries are installed:
```bash
pip install streamlit scikit-learn pandas numpy plotly
```

### 3. Launching the Web Dashboard
Run the Streamlit application directly from the repository root:

```bash
python -m streamlit run app.py
```

Once launched, access the dashboard at:
👉 **`http://localhost:8501`**

---

## 🛠️ Technology Stack

- **Core Analytics & ML:** Python, NumPy, Pandas, Scikit-Learn
- **Visualization:** Plotly Express & Plotly Graph Objects (Interactive SVG/WebGL)
- **Web Interface:** Streamlit (v1.50+)
- **Dimensionality Reduction:** Principal Component Analysis (PCA)
- **Evaluation Framework:** Scikit-Learn Metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz, ARI, ROC-AUC)

---

## 📜 Academic Attribution & License

This project was developed as part of academic research in machine learning applied to educational policy and equity assessment.
Distributed under the **MIT License**. See `LICENSE` for further details.
