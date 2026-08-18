from pathlib import Path
import requests
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
OUT_FILE = BASE_DIR / "Dataset" / "RMS_Crime_Incidents_2026.csv"

URL = (
    "https://services2.arcgis.com/qvkbeam7Wirps6zC/"
    "ArcGIS/rest/services/RMS_Crime_Incidents_2026/FeatureServer/0/query"
)

OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

rows = []
offset = 0
batch_size = 2000

while True:
    params = {
        "where": "1=1",
        "outFields": "*",
        "returnGeometry": "false",
        "f": "json",
        "resultOffset": offset,
        "resultRecordCount": batch_size,
        "orderByFields": "ESRI_OID",
    }

    response = requests.get(URL, params=params, timeout=60)
    response.raise_for_status()

    data = response.json()

    if "error" in data:
        raise RuntimeError(data["error"])

    features = data.get("features", [])

    if not features:
        break

    rows.extend(feature["attributes"] for feature in features)

    print(f"Downloaded {len(rows):,} records...")

    if len(features) < batch_size:
        break

    offset += batch_size

df = pd.DataFrame(rows)

date_columns = [
    "incident_occurred_at",
    "case_status_updated_at",
    "updated_in_ibr_at",
    "updated_at",
]

for column in date_columns:
    if column in df.columns:
        df[column] = pd.to_datetime(
            df[column],
            unit="ms",
            errors="coerce",
            utc=True,
        )

df.to_csv(OUT_FILE, index=False)

print(f"Saved {len(df):,} records to {OUT_FILE}")