import pandas as pd
from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).resolve().parents[3]

# Central processed-data directory
OUTPUT_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Raw infrastructure data
INFRA_DIR = (
    ROOT_DIR
    / "data"
    / "raw"
    / "infrastructure"
    / "Schools by Availability of Infrastructure and Facilities"
    / "EduEquity_Infrastructure_CSVs"
)

# Raw social-category enrolment data
SOCIAL_DIR = (
    ROOT_DIR
    / "data"
    / "raw"
    / "enrolment_by_social_category"
    / "enrollment_by_location_social_category"
    / "EduEquity_Enrolment_CSVs"
)

YEAR = "19-20"

# --------------------------------------------------
# 1. PROCESS INFRASTRUCTURE DATA
# --------------------------------------------------

print("=" * 60)
print("PROCESSING INFRASTRUCTURE DATA")
print("=" * 60)

infra_files = sorted(
    f for f in INFRA_DIR.glob("*.csv")
    if YEAR in f.name
)

print("Files found:", len(infra_files))

if not infra_files:
    raise FileNotFoundError(
        f"No infrastructure files containing {YEAR} were found."
    )

infra_frames = []

for file in infra_files:
    df = pd.read_csv(file, encoding="utf-8-sig")

    # Standardize district identifiers.
    df["State_Code"] = pd.to_numeric(
        df["State_Code"], errors="coerce"
    )
    df["District_Code"] = pd.to_numeric(
        df["District_Code"], errors="coerce"
    )

    infra_frames.append(df)

infra = pd.concat(infra_frames, ignore_index=True)

infra["Total_Number_of_Schools"] = pd.to_numeric(
    infra["Total_Number_of_Schools"], errors="coerce"
)

# Facility columns exclude identifiers and descriptive fields.
metadata_columns = [
    "Academic_Year",
    "State_Code",
    "State_Name",
    "District_Code",
    "District_Name",
    "Block_Code",
    "Udise_Block_Name",
    "Location",
    "School_Category_Id",
    "School_Category_Name",
    "School_Management_Id",
    "School_Management_Name",
    "Total_Number_of_Schools",
]

facility_columns = [
    col for col in infra.columns
    if col not in metadata_columns
]

for col in facility_columns:
    infra[col] = pd.to_numeric(infra[col], errors="coerce")

# Check whether facility counts exceed the number of schools.
invalid_counts = {}

for col in facility_columns:
    invalid = (
        infra[col].notna()
        & infra["Total_Number_of_Schools"].notna()
        & (infra[col] > infra["Total_Number_of_Schools"])
    )

    invalid_counts[col] = int(invalid.sum())

print("Infrastructure rows:", len(infra))
print(
    "Facility columns with invalid counts:",
    sum(count > 0 for count in invalid_counts.values())
)

# Stop rather than silently using questionable values.
if any(count > 0 for count in invalid_counts.values()):
    print("Columns with invalid counts:")
    print({
        col: count
        for col, count in invalid_counts.items()
        if count > 0
    })
    raise ValueError(
        "Investigate invalid facility counts before aggregation."
    )

if infra[["State_Code", "District_Code"]].isna().any().any():
    raise ValueError("Missing state or district codes in infrastructure data.")

if infra["Total_Number_of_Schools"].isna().any():
    raise ValueError("Missing school counts in infrastructure data.")

if (infra["Total_Number_of_Schools"] < 0).any():
    raise ValueError("Negative school counts detected.")

# Aggregate counts across district strata.
group_columns = ["State_Code", "District_Code"]

infra_agg = infra.groupby(
    group_columns, as_index=False
).agg(
    State_Name=("State_Name", "first"),
    District_Name=("District_Name", "first"),
    total_schools=("Total_Number_of_Schools", "sum"),
    **{
        col.lower() + "_count": (col, "sum")
        for col in facility_columns
    }
)

# Availability rate = schools with facility / total schools.
# A zero denominator becomes missing, not zero.
for col in facility_columns:
    count_col = col.lower() + "_count"
    rate_col = col.lower() + "_rate"

    infra_agg[rate_col] = (
        infra_agg[count_col]
        / infra_agg["total_schools"].replace(0, pd.NA)
    )

infra_path = OUTPUT_DIR / "district_infrastructure_2019_20.csv"
infra_agg.to_csv(infra_path, index=False)

print("Districts in infrastructure data:", len(infra_agg))
print("Saved:", infra_path)


# --------------------------------------------------
# 2. PROCESS SOCIAL-CATEGORY ENROLMENT DATA
# --------------------------------------------------

print("\n" + "=" * 60)
print("PROCESSING SOCIAL-CATEGORY ENROLMENT DATA")
print("=" * 60)

social_files = sorted(
    f for f in SOCIAL_DIR.glob("*.csv")
    if YEAR in f.name
)

print("Files found:", len(social_files))

if not social_files:
    raise FileNotFoundError(
        f"No social-category files containing {YEAR} were found."
    )

social_frames = []

for file in social_files:
    df = pd.read_csv(file, encoding="utf-8-sig")

    # Sum class-wise enrolment, including pre-primary.
    boys_columns = [
        col for col in df.columns
        if col.endswith("_boy")
    ]

    girls_columns = [
        col for col in df.columns
        if col.endswith("_girl")
    ]

    if not boys_columns or not girls_columns:
        raise ValueError(
            f"Could not identify boys/girls columns in {file.name}"
        )

    for col in boys_columns + girls_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["boys_enrolment"] = df[boys_columns].sum(
        axis=1, min_count=1
    )
    df["girls_enrolment"] = df[girls_columns].sum(
        axis=1, min_count=1
    )

    df["total_enrolment"] = (
        df["boys_enrolment"] + df["girls_enrolment"]
    )

    social_frames.append(
        df[
            [
                "st_code",
                "state_name",
                "dt_code",
                "district_name",
                "caste_name",
                "boys_enrolment",
                "girls_enrolment",
                "total_enrolment",
            ]
        ]
    )

social = pd.concat(social_frames, ignore_index=True)

social["st_code"] = pd.to_numeric(
    social["st_code"], errors="coerce"
)
social["dt_code"] = pd.to_numeric(
    social["dt_code"], errors="coerce"
)

if social[["st_code", "dt_code"]].isna().any().any():
    raise ValueError("Missing state or district codes in social data.")

# Standardize category labels for pivoting.
social["caste_name"] = (
    social["caste_name"]
    .astype("string")
    .str.strip()
    .str.upper()
)

print("\nSocial-category labels:")
print(sorted(social["caste_name"].dropna().unique().tolist()))

# Aggregate each category within each district.
category_agg = social.groupby(
    ["st_code", "dt_code", "caste_name"],
    as_index=False
).agg(
    boys_enrolment=("boys_enrolment", "sum"),
    girls_enrolment=("girls_enrolment", "sum"),
    total_enrolment=("total_enrolment", "sum"),
)

# Create one column per category and metric.
metrics = [
    "boys_enrolment",
    "girls_enrolment",
    "total_enrolment",
]

wide = category_agg.pivot(
    index=["st_code", "dt_code"],
    columns="caste_name",
    values=metrics,
)

wide.columns = [
    f"{metric}_{category.lower().replace(' ', '_')}"
    for metric, category in wide.columns
]

wide = wide.reset_index()

# Overall totals across categories.
district_totals = social.groupby(
    ["st_code", "dt_code"], as_index=False
).agg(
    State_Name=("state_name", "first"),
    District_Name=("district_name", "first"),
    social_boys_total=("boys_enrolment", "sum"),
    social_girls_total=("girls_enrolment", "sum"),
    social_enrolment_total=("total_enrolment", "sum"),
)

social_agg = district_totals.merge(
    wide,
    on=["st_code", "dt_code"],
    how="left",
    validate="one_to_one",
)

# Girls' share of total enrolment.
social_agg["girls_enrolment_share"] = (
    social_agg["social_girls_total"]
    / social_agg["social_enrolment_total"].replace(0, pd.NA)
)

# Add the standard key names used in the other datasets.
social_agg = social_agg.rename(
    columns={
        "st_code": "State_Code",
        "dt_code": "District_Code",
    }
)

social_path = OUTPUT_DIR / "district_social_equity_2019_20.csv"
social_agg.to_csv(social_path, index=False)

print("Districts in social data:", len(social_agg))
print("Saved:", social_path)


# --------------------------------------------------
# 3. VALIDATION SUMMARY
# --------------------------------------------------

print("\n" + "=" * 60)
print("VALIDATION SUMMARY")
print("=" * 60)

print("Infrastructure duplicate district keys:",
      infra_agg.duplicated(["State_Code", "District_Code"]).sum())

print("Social duplicate district keys:",
      social_agg.duplicated(["State_Code", "District_Code"]).sum())

print("Infrastructure rows:", len(infra_agg))
print("Social rows:", len(social_agg))

print("\nDone. Raw CSV files were not modified.")