# EduEquity: Two-Year Clustering Report

## 1. Objective

Compare district-year records from 2018-19 and 2019-20 using shared numeric educational indicators and one combined K-Means model.

## 2. Input datasets

- 2018-19: `eduequity_model_dataset_2018_19.csv`
- 2019-20: `eduequity_model_dataset_2019_20.csv`
- Input directory: `D:\Semester 5\EduEquity\data\processed`
- Usable district-year rows: 1383

## 3. Feature alignment

Features were matched by normalized column names. Only numeric indicators present and usable in both datasets were used for the main two-year comparison. See `feature_comparison_by_year.csv` for shared and year-specific fields.

Selected features:
- `building_rate`
- `complete_medical_checkup_rate`
- `computer_available_rate`
- `drinking_water_rate`
- `functional_boy_toilet_rate`
- `functional_drinking_water_rate`
- `functional_electricity_rate`
- `functional_girl_toilet_rate`
- `functional_toilet_and_urinal_rate`
- `functional_toilet_facility_rate`
- `functional_urinal_boy_rate`
- `functional_urinal_girl_rate`
- `functional_urinal_rate`
- `furniture_rate`
- `handwash_rate`
- `incinerator_rate`
- `internet_rate`
- `kitchen_garden_rate`
- `land_available_rate`
- `librarian_rate`
- `library_or_reading_corner_or_book_bank_rate`
- `medical_checkup_rate`
- `newspaper_rate`
- `playground_rate`
- `rain_water_harvesting_rate`
- `ramps_rate`
- `separate_room_for_headmaster_rate`
- `solar_panel_rate`
- `water_purifier_rate`
- `water_tested_rate`

## 4. Preprocessing

- Missing values were median-imputed across the combined data.
- Common indicators were standardized using StandardScaler.
- See `feature_scaling_summary.csv` for preprocessing details.

## 5. Number of clusters

- Selected K by maximum silhouette score: 3
- Review the Elbow and Silhouette charts before finalizing K.

|   k |   inertia |   silhouette_score |
|----:|----------:|-------------------:|
|   1 |   41490   |         nan        |
|   2 |   32436.6 |           0.19262  |
|   3 |   27937.4 |           0.198835 |
|   4 |   25352.8 |           0.16196  |
|   5 |   23751.7 |           0.148076 |
|   6 |   22433.9 |           0.16207  |
|   7 |   21242.5 |           0.165321 |
|   8 |   20366.7 |           0.149066 |
|   9 |   19327.6 |           0.158811 |
|  10 |   18700   |           0.16312  |

## 6. Cluster counts by year

| academic_year   |   cluster |   district_year_rows |
|:----------------|----------:|---------------------:|
| 2018-19         |         0 |                   48 |
| 2018-19         |         1 |                  255 |
| 2018-19         |         2 |                  348 |
| 2019-20         |         0 |                   52 |
| 2019-20         |         1 |                  285 |
| 2019-20         |         2 |                  395 |

## 7. Cluster profiles by year

See `cluster_profiles_by_year_means.csv` and `cluster_profiles_by_year_medians.csv` to compare the indicators in each cluster for the two years.

## 8. Interpretation cautions

A cluster is a statistical grouping, not a ranking of district quality. Whether a high or low indicator is desirable depends on its definition and units. For example, a higher dropout rate may signal a challenge, while higher facility access may signal better provision. Clustering alone does not establish causation.

The combined model uses the same features and scaling approach for both years, making cluster labels more comparable than independently fitted models. Nevertheless, verify district identity, indicator definitions, and data consistency.

## 9. Main outputs

- `district_year_cluster_assignments.csv`
- `cluster_counts_by_year.csv`
- `cluster_profiles_by_year_means.csv`
- `cluster_profiles_by_year_medians.csv`
- `cluster_evaluation.csv`
- `feature_comparison_by_year.csv`
- `feature_scaling_summary.csv`
- `combined_two_year_common_features.csv`
- `pca_coordinates.csv`
- `plots/`
