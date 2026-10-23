
import requests
import pandas as pd
import time
from pathlib import Path

# Dataset 2: Teacher data
BASE_URL = "https://www.data.gov.in/backend/dmspublic/v1/resources"
CATALOG_REFERENCE = "6891443"

OUTPUT_FILE = "eduequity_teacher_metadata.csv"
BATCH_SIZE = 100

offset = 0
all_records = []
seen_ids = set()

while True:
    params = {
        "filters[catalog_reference]": CATALOG_REFERENCE,
        "offset": offset,
        "limit": BATCH_SIZE,
        "sort[changed]": "desc",
        "filters[domain_visibility]": 4
    }

    print(f"Fetching resources from offset {offset}...")

    try:
        response = requests.get(
            BASE_URL,
            params=params,
            timeout=60
        )
        response.raise_for_status()
        result = response.json()

    except requests.RequestException as error:
        print("Request failed:", error)
        break

    except ValueError:
        print("Invalid JSON response.")
        break

    if not isinstance(result, dict):
        print("Unexpected response format.")
        break

    payload = result.get("data", {})
    records = payload.get("rows", []) if isinstance(payload, dict) else []

    if not isinstance(records, list):
        print("Unexpected rows format.")
        break

    if not records:
        print("No more resources returned.")
        break

    for record in records:
        # API fields are lists, e.g. nid: [6892040].
        # Convert each field to its first value.
        clean_record = {}

        for key, value in record.items():
            if isinstance(value, list):
                clean_record[key] = value[0] if value else None
            else:
                clean_record[key] = value

        resource_id = clean_record.get("nid")

        if resource_id is None:
            print("Skipping resource without nid.")
            continue

        resource_id = str(resource_id)

        if resource_id not in seen_ids:
            seen_ids.add(resource_id)
            all_records.append(clean_record)

    print("Unique resources collected:", len(all_records))

    offset += BATCH_SIZE
    time.sleep(0.2)

# Save metadata
if all_records:
    df = pd.DataFrame(all_records)

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nMetadata saved successfully!")
    print("File:", Path(OUTPUT_FILE).resolve())
    print("Total unique resources:", len(all_records))
    print("Total columns:", len(df.columns))

else:
    print("No metadata was saved.")
