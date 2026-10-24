import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]

DATA_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_DIR = DATA_DIR

social = pd.read_csv(
    OUTPUT_DIR / "district_social_equity_2019_20.csv"
)

enrolment = pd.read_csv(
    DATA_DIR / "district_year_enrolment.csv"
)

enrolment["ac_year"] = enrolment["ac_year"].astype(str).str.strip()
enrolment = enrolment[enrolment["ac_year"] == "2019-20"].copy()

# Inspect selected districts.
districts = ["KUPWARA", "BARAMULA", "SRINAGAR", "BADGAM", "PULWAMA"]

print("=" * 70)
print("SOCIAL-CATEGORY ENROLMENT BY DISTRICT")
print("=" * 70)

category_columns = [
    col for col in social.columns
    if col.startswith("total_enrolment_")
]

for district in districts:
    rows = social[
        social["District_Name"].str.upper() == district
    ]

    if rows.empty:
        print(f"\n{district}: not found")
        continue

    row = rows.iloc[0]

    print(f"\n{district}")
    print("Social-category total:", row["social_enrolment_total"])

    for col in category_columns:
        print(f"  {col}: {row[col]}")

# Compare social and age/class totals.
social["st_code"] = pd.to_numeric(social["State_Code"], errors="coerce")
social["dt_code"] = pd.to_numeric(social["District_Code"], errors="coerce")

enrolment = enrolment.rename(
    columns={
        "state_cd": "st_code",
        "district_cd": "dt_code",
    }
)

comparison = social.merge(
    enrolment[
        ["st_code", "dt_code", "total_enrolment"]
    ],
    on=["st_code", "dt_code"],
    how="inner",
    validate="one_to_one",
)

comparison["social_to_age_ratio"] = (
    comparison["social_enrolment_total"]
    / comparison["total_enrolment"].replace(0, pd.NA)
)

print("\n" + "=" * 70)
print("COMPARISON WITH AGE/CLASS ENROLMENT")
print("=" * 70)

print(comparison["social_to_age_ratio"].describe())

print("\nDistricts with the lowest ratios:")
print(
    comparison[
        [
            "District_Name",
            "social_enrolment_total",
            "total_enrolment",
            "social_to_age_ratio",
        ]
    ]
    .sort_values("social_to_age_ratio")
    .head(10)
    .to_string(index=False)
)

print("\nDistricts with the highest ratios:")
print(
    comparison[
        [
            "District_Name",
            "social_enrolment_total",
            "total_enrolment",
            "social_to_age_ratio",
        ]
    ]
    .sort_values("social_to_age_ratio", ascending=False)
    .head(10)
    .to_string(index=False)
)