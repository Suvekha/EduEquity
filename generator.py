'''
import requests
import pandas as pd
import time

API_URL = "https://www.data.gov.in/backend/dmspublic/v1/resources"

CATALOG_ID = 6887580
BATCH_SIZE = 100

all_resources = []
offset = 0
total = None

session = requests.Session()
session.headers.update({
    "User-Agent": "EduEquity-Research/1.0"
})


def get_value(record, field):
    """Extract a value from the API's list-based fields."""
    value = record.get(field, [])
    if isinstance(value, list):
        return value[0] if value else None
    return value


while total is None or offset < total:
    params = {
        "filters[catalog_reference]": CATALOG_ID,
        "offset": offset,
        "limit": BATCH_SIZE,
        "sort[changed]": "desc",
        "filters[domain_visibility]": 4
    }

    try:
        response = session.get(
            API_URL,
            params=params,
            timeout=60
        )
        response.raise_for_status()
        result = response.json()

    except (requests.RequestException, ValueError) as error:
        print(f"Error at offset {offset}: {error}")
        print("Run the script again later if the problem persists.")
        break

    if total is None:
        total = int(result.get("total", 0))
        print(f"Resources reported by API: {total}")

    rows = result.get("data", {}).get("rows", [])

    if not rows:
        print("No more records returned.")
        break

    for record in rows:
        all_resources.append({
            "nid": get_value(record, "nid"),
            "title": get_value(record, "title"),
            "datafile_url": get_value(record, "datafile_url"),
            "file_size": get_value(record, "file_size"),
            "period_from": get_value(
                record, "data_time_period_from"
            ),
            "period_to": get_value(
                record, "data_time_period_to"
            ),
            "file_format": get_value(record, "file_format")
        })

    offset += len(rows)
    print(f"Collected {len(all_resources)} / {total}")

    time.sleep(0.2)

metadata = pd.DataFrame(all_resources)
metadata.drop_duplicates(subset=["nid"], inplace=True)

metadata.to_csv(
    "eduequity_enrolment_metadata.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nMetadata collection finished.")
print(f"Unique resources collected: {len(metadata)}")
print("Saved: eduequity_enrolment_metadata.csv")
print(f"Missing file URLs: {metadata['datafile_url'].isna().sum()}")
'''




import pandas as pd

failures = pd.read_csv("failed_downloads.csv")

print("Number of failed resources:", len(failures))
print("\nError summary:")
print(failures["error"].value_counts(dropna=False))

print("\nDetails of each failure:")
print(failures[["nid", "title", "error"]].to_string(index=False))
