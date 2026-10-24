import pandas as pd
from pathlib import Path

# Project root
ROOT_DIR = Path(__file__).resolve().parents[3]

# Processed datasets
DATA_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_DIR = DATA_DIR

# Load the three processed datasets
enrolment_path = DATA_DIR / "district_year_enrolment.csv"
infrastructure_path = DATA_DIR / "district_infrastructure_2019_20.csv"
social_path = DATA_DIR / "district_social_equity_2019_20.csv"

enrolment = pd.read_csv(enrolment_path)
infrastructure = pd.read_csv(infrastructure_path)
social = pd.read_csv(social_path)

print("Loaded datasets:")
print("Enrolment:", enrolment.shape)
print("Infrastructure:", infrastructure.shape)
print("Social equity:", social.shape)


# --------------------------------------------------
# 2. FILTER ENROLMENT TO 2019–20
# --------------------------------------------------

enrolment["ac_year"] = enrolment["ac_year"].astype(str).str.strip()

enrolment = enrolment[
    enrolment["ac_year"] == "2019-20"
].copy()

# Standardize join-key names.
enrolment = enrolment.rename(
    columns={
        "state_cd": "State_Code",
        "district_cd": "District_Code",
        "state_name": "State_Name",
        "district_name": "District_Name",
    }
)

# Normalize key types across all datasets.
for df in [enrolment, infrastructure, social]:
    df["State_Code"] = pd.to_numeric(
        df["State_Code"], errors="coerce"
    )
    df["District_Code"] = pd.to_numeric(
        df["District_Code"], errors="coerce"
    )

# Remove rows without usable keys.
enrolment = enrolment.dropna(
    subset=["State_Code", "District_Code"]
)
infrastructure = infrastructure.dropna(
    subset=["State_Code", "District_Code"]
)
social = social.dropna(
    subset=["State_Code", "District_Code"]
)

keys = ["State_Code", "District_Code"]

# --------------------------------------------------
# 3. CHECK KEY UNIQUENESS
# --------------------------------------------------

for name, df in [
    ("Enrolment", enrolment),
    ("Infrastructure", infrastructure),
    ("Social equity", social),
]:
    duplicates = df.duplicated(keys).sum()
    print(f"{name} duplicate district keys:", duplicates)

    if duplicates:
        raise ValueError(
            f"{name} contains duplicate district keys."
        )

# --------------------------------------------------
# 4. SELECT FEATURES
# --------------------------------------------------

enrolment_features = [
    col for col in [
        "State_Code",
        "District_Code",
        "State_Name",
        "District_Name",
        "total_boys",
        "total_girls",
        "total_enrolment",
        "girls_to_boys_ratio",
    ]
    if col in enrolment.columns
]

# Keep school counts and calculated facility rates.
infra_features = [
    col for col in infrastructure.columns
    if col in keys
    or col in ["total_schools", "State_Name", "District_Name"]
    or col.endswith("_rate")
]

social_features = [
    col for col in social.columns
    if col in keys
    or col in [
        "social_boys_total",
        "social_girls_total",
        "social_enrolment_total",
        "girls_enrolment_share",
    ]
    or col.startswith("total_enrolment_")
    or col.startswith("girls_enrolment_")
    or col.startswith("boys_enrolment_")
]

enrolment = enrolment[enrolment_features].copy()
infrastructure = infrastructure[infra_features].copy()
social = social[social_features].copy()

# Avoid duplicate name columns in the final dataset.
infrastructure = infrastructure.drop(
    columns=["State_Name", "District_Name"],
    errors="ignore",
)
social = social.drop(
    columns=["State_Name", "District_Name"],
    errors="ignore",
)

# --------------------------------------------------
# 5. JOIN ON STATE AND DISTRICT CODES
# --------------------------------------------------

combined = enrolment.merge(
    infrastructure,
    on=keys,
    how="inner",
    validate="one_to_one",
)

combined = combined.merge(
    social,
    on=keys,
    how="inner",
    validate="one_to_one",
)

print("\nDistrict counts:")
print("Enrolment:", len(enrolment))
print("Infrastructure:", len(infrastructure))
print("Social equity:", len(social))
print("Matched across all three:", len(combined))

print(
    "Districts excluded due to unmatched keys:",
    len(enrolment) - len(combined)
)

# --------------------------------------------------
# 6. CREATE EQUITY INDICATORS
# --------------------------------------------------

if {"total_girls", "total_boys"}.issubset(combined.columns):
    combined["girls_enrolment_share_age_class"] = (
        combined["total_girls"]
        / combined["total_enrolment"].replace(0, pd.NA)
    )

if {
    "social_enrolment_total",
    "total_enrolment",
}.issubset(combined.columns):
    combined["social_to_age_class_enrolment_ratio"] = (
        combined["social_enrolment_total"]
        / combined["total_enrolment"].replace(0, pd.NA)
    )

# --------------------------------------------------
# 7. SAVE AND REPORT
# --------------------------------------------------

output_path = OUTPUT_DIR / "eduequity_district_dataset_2019_20.csv"
combined.to_csv(output_path, index=False)

print("\nFinal dataset shape:", combined.shape)
print("Missing values:", int(combined.isna().sum().sum()))
print("Saved to:", output_path)

print("\nSample:")
print(combined.head(5).to_string(index=False))