
import requests
import pandas as pd
import time

# Replace this with the infrastructure catalog_reference
CATALOG_ID = "6889303"

API_URL = "https://www.data.gov.in/backend/dmspublic/v1/resources"
BATCH_SIZE = 100

all_resources = []
offset = 0
total = None

session = requests.Session()
session.headers.update({
    "User-Agent": "EduEquity-Research/1.0"
})

def get_value(record, field):
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
        print("Metadata collection stopped. Check the error and rerun.")
        break

    if total is None:
        total = int(result.get("total", 0))
        print(f"Total resources reported: {total}")

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

# Save only if the API listing appears complete
metadata = pd.DataFrame(all_resources)

if total is not None and len(metadata.drop_duplicates(
    subset=["nid"]
)) == total:
    metadata.drop_duplicates(subset=["nid"], inplace=True)
    metadata.to_csv(
        "eduequity_infrastructure_metadata.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\nMetadata collection completed!")
    print(f"Unique resources: {len(metadata)}")
    print("Saved: eduequity_infrastructure_metadata.csv")
    print(
        "Resources without download URLs:",
        metadata["datafile_url"].isna().sum()
    )
else:
    print("\nCollection may be incomplete.")
    print(f"Records collected: {len(metadata)}")
    print(f"Expected total: {total}")
    metadata.to_csv(
        "eduequity_infrastructure_metadata_partial.csv",
        index=False,
        encoding="utf-8-sig"
    )
    print("Partial results saved for checking.")
