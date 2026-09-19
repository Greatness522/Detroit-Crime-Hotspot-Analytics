# PRECINCT CONTROL CENTER FINAL BUILD 2026-08-17 — Detroit overview -> selected precinct evaluation -> scoped drill-down analysis
# VERIFIED STREET-HEATMAP + HOTSPOT HIGHLIGHT BUILD 2026-08-17
# VERIFIED STREET-HEATMAP BUILD 2026-08-17
# VERIFIED REAL-MAP BUILD 2026-08-17: storytelling + Detroit street basemap + transparent H3 + recent incident locations
from pathlib import Path
import os
import re
import json
import html

import folium
import numpy as np
import pandas as pd
import branca.colormap as cm

try:
    import h3
except ModuleNotFoundError:  # pragma: no cover - optional dependency in constrained envs
    h3 = None

from folium.plugins import HeatMap, MarkerCluster, FastMarkerCluster, Fullscreen
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.spatial import ConvexHull, QhullError


BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = BASE_DIR / "Dataset"
DATASET_GLOB = "RMS_Crime_Incidents_*.csv"
IMAGES_DIR = BASE_DIR / "Images"
DOCS_DIR = BASE_DIR / "Documentation"
RECENT_WINDOW_DAYS = 14
RECENT_WINDOW_LABEL = f"{RECENT_WINDOW_DAYS}D"

# ---------------------------------------------------------------------------
# OPERATIONAL CRIME CATEGORY DEFINITIONS
# Keep these definitions centralized so every dashboard component uses the
# same records for selectors, heatmaps, neighborhood/precinct scopes, timing,
# and hotspot analysis.
# ---------------------------------------------------------------------------
VIOLENT_CRIMES = {
    "HOMICIDE",
    "SEXUAL ASSAULT",
    "ROBBERY",
    "AGGRAVATED ASSAULT",
    "ASSAULT",
}

PROPERTY_CRIMES = {
    "BURGLARY",
    "LARCENY",
    "DAMAGE TO PROPERTY",
    "ARSON",
}

VEHICLE_LARCENY_PATTERN = re.compile(
    r"\b(?:VEHICLE|AUTO|AUTOMOBILE|CAR|TRUCK|MOTOR VEHICLE|"
    r"FROM VEHICLE|VEHICLE PART|VEHICLE PARTS|CATALYTIC CONVERTER|"
    r"LICENSE PLATE|WHEEL|WHEELS|TIRE|TIRES)\b",
    flags=re.IGNORECASE,
)

CATEGORY_FOCUS_COLUMNS = {
    "Violent Crime": "is_violent_crime",
    "Property Crime": "is_property_crime",
    "Vehicle-Related Crime": "is_vehicle_related",
}

IMAGES_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".matplotlib_cache"))

import matplotlib.pyplot as plt
import seaborn as sns


def load_data(dataset_dir: Path) -> pd.DataFrame:
    csv_files = sorted(dataset_dir.glob(DATASET_GLOB))
    if not csv_files:
        raise FileNotFoundError(
            f"No dataset files found in {dataset_dir} matching pattern {DATASET_GLOB}"
        )

    frames = []
    for file_path in csv_files:
        frame = pd.read_csv(file_path, low_memory=False)
        frame["source_file"] = file_path.name

        # Try to infer year from filename like RMS_Crime_Incidents_2026.csv.
        year_match = re.search(r"(19|20)\d{2}", file_path.stem)
        frame["source_year_hint"] = int(year_match.group(0)) if year_match else np.nan
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)
    df["incident_occurred_at"] = pd.to_datetime(
    df["incident_occurred_at"],
    format="mixed",
    errors="coerce",
    utc=True,
)

    # Ensure incident_year exists and is consistent for multi-year analysis.
    if "incident_year" not in df.columns:
        df["incident_year"] = pd.NA
    df["incident_year"] = pd.to_numeric(df["incident_year"], errors="coerce")
    df["incident_year"] = df["incident_year"].fillna(df["source_year_hint"])
    df["incident_year"] = df["incident_year"].fillna(df["incident_occurred_at"].dt.year)

    # Normalize precinct IDs (e.g., 2 -> 02, 0W -> 0W) for cleaner filtering.
    def normalize_precinct(value):
        if pd.isna(value):
            return "Unknown"
        text = str(value).strip().upper()
        if text == "" or text in {"NAN", "NONE"}:
            return "Unknown"
        if text.isdigit():
            return text.zfill(2)
        return text

    df["precinct_norm"] = df["police_precinct"].apply(normalize_precinct)

    required_cols = ["latitude", "longitude", "incident_occurred_at", "neighborhood"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    df["neighborhood"] = df["neighborhood"].fillna("").astype(str).str.strip()
    df = df.dropna(subset=required_cols).copy()
    df = df[df["neighborhood"].ne("")].copy()
    df = df[(df["latitude"].between(42.2, 42.5)) & (df["longitude"].between(-83.3, -82.9))]
    if "incident_hour_of_day" in df.columns:
        df["incident_hour_of_day"] = pd.to_numeric(df["incident_hour_of_day"], errors="coerce")
    else:
        df["incident_hour_of_day"] = df["incident_occurred_at"].dt.hour

    # Convert to timezone-naive timestamps before period conversion to avoid warning noise.
    df["week_start"] = (
        df["incident_occurred_at"].dt.tz_convert(None).dt.to_period("W-MON").dt.start_time
    )
    df["incident_date"] = df["incident_occurred_at"].dt.tz_convert(None).dt.date
    df["month_start"] = df["incident_occurred_at"].dt.tz_convert(None).dt.to_period("M").dt.start_time

    description = df.get("offense_description", pd.Series("", index=df.index)).fillna("").astype(str)
    category = (
        df.get("offense_category", pd.Series("", index=df.index))
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # Preserve the normalized category so every downstream comparison uses
    # the same spelling/casing.
    df["offense_category"] = category

    # Operational category focuses used throughout the dashboard.
    df["is_violent_crime"] = category.isin(VIOLENT_CRIMES)
    df["is_property_crime"] = category.isin(PROPERTY_CRIMES)

    # Vehicle-related = all STOLEN VEHICLE incidents plus LARCENY records
    # whose offense description indicates theft from/of vehicle components.
    vehicle_larceny = category.eq("LARCENY") & description.str.contains(
        VEHICLE_LARCENY_PATTERN,
        na=False,
    )
    df["is_vehicle_related"] = category.eq("STOLEN VEHICLE") | vehicle_larceny

    # Retain the original gun flag for any non-category legacy analysis that
    # may still reference it elsewhere in this large script.
    df["is_gun_related"] = description.str.contains(
        r"GUN|FIREARM|WEAPON|SHOT",
        case=False,
        na=False,
        regex=True,
    )

    return df


def add_neighborhood_scopes(
    df: pd.DataFrame,
    citywide: pd.DataFrame,
    build_scoped,
) -> pd.DataFrame:
    """Append records calculated independently for every cleaned neighborhood."""
    frames = [citywide.assign(neighborhood_scope="ALL")]
    for neighborhood, subset in df.groupby("neighborhood", sort=True):
        scoped = build_scoped(subset.copy())
        if not scoped.empty:
            frames.append(scoped.assign(neighborhood_scope=str(neighborhood)))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def assign_shift_window(hour: float) -> str:
    if pd.isna(hour):
        return "Unknown"
    h = int(hour)
    if 6 <= h <= 13:
        return "Day Shift (06:00-13:59)"
    if 14 <= h <= 21:
        return "Evening Shift (14:00-21:59)"
    return "Night Shift (22:00-05:59)"


# def add_map_help_box(m: folium.Map) -> None:
#     help_html = """
#     <div style="
#         position: fixed;
#         bottom: 24px;
#         left: 24px;
#         z-index: 9999;
#         background: rgba(255, 255, 255, 0.95);
#         border: 1px solid #d1d9e6;
#         border-radius: 8px;
#         padding: 10px 12px;
#         max-width: 340px;
#         font-size: 12px;
#         line-height: 1.35;
#         color: #0f172a;
#         box-shadow: 0 4px 12px rgba(15, 23, 42, 0.12);
#     ">
#       <b>How to use this map</b><br>
#             1) Start with the Top Filter panel (Core, Action, Decision, Precinct, Category, Trend).<br>
#             2) Pick one layer per filter group for the cleanest view.<br>
#             3) Use Precinct + Category together to narrow the map quickly.<br>
#             4) Crime Type/Category filters drive heatmaps; core count layers marked All Incidents stay citywide totals.<br>
#             5) Add Shift layers only when you need time-of-day context.<br>
#             6) Use spike and action marker layers for exact response points.<br>
#       Tip: Too much overlap means too many layers are active.
#     </div>
#     """
#     m.get_root().html.add_child(folium.Element(help_html))


# OPERATIONS LANDING FINAL BUILD 2026-08-17
# Citywide landing page -> precinct responsibility overview -> deep-link analysis drilldowns
# Precinct-first operational dashboard: Detroit overview -> precinct -> crime -> when/where/change.

def add_top_selector_panel(
    m: folium.Map,
    precinct_values: list[str],
    neighborhood_values: list[str],
    crime_type_values: list[str],
    precinct_improvement: pd.DataFrame | None = None,
    precinct_crime_trends: pd.DataFrame | None = None,
    precinct_crime_14d: pd.DataFrame | None = None,
    priority_concerns: pd.DataFrame | None = None,
    temporal_summary: pd.DataFrame | None = None,
    temporal_matrix: pd.DataFrame | None = None,
    hotspot_change: pd.DataFrame | None = None,
    spatial_daily: pd.DataFrame | None = None,
    precinct_bounds: dict[str, list[list[float]]] | None = None,
    neighborhood_bounds: dict[str, list[list[float]]] | None = None,
    current_year: int | None = None,
    previous_year: int | None = None,
    baseline_year: int | None = None,
) -> None:
    # Keep nonstandard/unknown precinct codes in citywide calculations, but do not
    # present them as operational precinct choices until their source meaning is verified.
    invalid_precinct_codes = {"00", "0W", "OW", "UNKNOWN", "NAN", "NONE", ""}
    precincts = [
        str(p).strip()
        for p in precinct_values
        if isinstance(p, str)
        and str(p).strip().upper() not in invalid_precinct_codes
    ]
    crime_types = [str(c) for c in crime_type_values if isinstance(c, str) and c.strip()]
    neighborhoods = sorted({str(n).strip() for n in neighborhood_values if str(n).strip()})

    precinct_options = "".join(
        [f'<option value="{html.escape(p, quote=True)}">{html.escape(p)}</option>' for p in precincts]
    )
    neighborhood_options = "".join(
        f'<option value="{html.escape(n, quote=True)}">{html.escape(n)}</option>'
        for n in neighborhoods
    )

    category_options = [
        '<option value="Category Focus | Violent Crime">Category Focus: Violent Crime</option>',
        '<option value="Category Focus | Property Crime">Category Focus: Property Crime</option>',
        '<option value="Category Focus | Vehicle-Related Crime">Category Focus: Vehicle-Related Crime</option>',
    ]
    category_options.extend(
        [
            f'<option value="Crime Type | {html.escape(c, quote=True)}">Crime Type: {html.escape(c)}</option>'
            for c in sorted(set(crime_types))
        ]
    )
    category_options_html = "".join(category_options)

    def clean_number(value):
        if pd.isna(value):
            return None
        if isinstance(value, (np.integer, int)):
            return int(value)
        if isinstance(value, (np.floating, float)):
            return float(value)
        return value

    overall_data = {}
    if precinct_improvement is not None and not precinct_improvement.empty:
        for _, row in precinct_improvement.iterrows():
            precinct = str(row.get("precinct_norm", ""))
            scope = str(row.get("neighborhood_scope", "ALL"))
            if not precinct:
                continue
            overall_data[f"{precinct}|{scope}"] = {
                "precinct": precinct,
                "neighborhood_scope": scope,
                "incidents_baseline": clean_number(row.get("incidents_baseline")),
                "incidents_previous": clean_number(row.get("incidents_previous")),
                "incidents_current": clean_number(row.get("incidents_current")),
                "pct_change_vs_previous": clean_number(row.get("pct_change_vs_previous")),
                "pct_change_vs_baseline": clean_number(row.get("pct_change_vs_baseline")),
                "trend_class": str(row.get("trend_class", row.get("improvement_status", "Unknown"))),
                "comparison_date": str(row.get("comparison_date", "")),
            }

    crime_trend_records = []
    if precinct_crime_trends is not None and not precinct_crime_trends.empty:
        wanted = [
            "neighborhood_scope",
            "precinct_norm",
            "offense_category",
            "incidents_baseline",
            "incidents_previous",
            "incidents_current",
            "pct_change_vs_previous",
            "pct_change_vs_baseline",
            "trend_class",
            "comparison_date",
        ]
        for _, row in precinct_crime_trends.iterrows():
            rec = {col: clean_number(row.get(col)) for col in wanted}
            rec["precinct_norm"] = str(row.get("precinct_norm", ""))
            rec["offense_category"] = str(row.get("offense_category", ""))
            rec["trend_class"] = str(row.get("trend_class", "Unknown"))
            rec["comparison_date"] = str(row.get("comparison_date", ""))
            crime_trend_records.append(rec)

    crime_14d_records = []
    if precinct_crime_14d is not None and not precinct_crime_14d.empty:
        wanted_14d = [
            "neighborhood_scope",
            "precinct_norm",
            "offense_category",
            "previous_14d",
            "current_14d",
            "pct_change_14d",
            "recent_movement",
            "city_pct_change_14d",
            "previous_14d_start",
            "previous_14d_end",
            "current_14d_start",
            "current_14d_end",
        ]
        for _, row in precinct_crime_14d.iterrows():
            rec = {col: clean_number(row.get(col)) for col in wanted_14d}
            # CSVs can coerce 02 -> 2, so normalize here for dashboard matching.
            pval = row.get("precinct_norm", "")
            if pd.isna(pval):
                pkey = ""
            else:
                ptxt = str(pval).strip()
                pkey = ptxt.zfill(2) if ptxt.isdigit() else ptxt
            rec["precinct_norm"] = pkey
            rec["offense_category"] = str(row.get("offense_category", ""))
            rec["recent_movement"] = str(row.get("recent_movement", "Unknown"))
            for col in ["previous_14d_start", "previous_14d_end", "current_14d_start", "current_14d_end"]:
                rec[col] = str(row.get(col, ""))
            crime_14d_records.append(rec)

    priority_records = []
    if priority_concerns is not None and not priority_concerns.empty:
        wanted_priority = [
            "neighborhood_scope",
            "precinct_norm",
            "offense_category",
            "previous_14d",
            "current_14d",
            "change_14d",
            "pct_change_14d",
            "city_pct_change_14d",
            "city_gap_14d",
            "pct_change_vs_previous",
            "priority_score",
            "priority_signal",
        ]
        for _, row in priority_concerns.iterrows():
            rec = {col: clean_number(row.get(col)) for col in wanted_priority}
            pval = row.get("precinct_norm", "")
            if pd.isna(pval):
                pkey = ""
            else:
                ptxt = str(pval).strip()
                pkey = ptxt.zfill(2) if ptxt.isdigit() else ptxt
            rec["precinct_norm"] = pkey
            rec["offense_category"] = str(row.get("offense_category", ""))
            rec["priority_signal"] = str(row.get("priority_signal", "Monitor"))
            priority_records.append(rec)

    temporal_summary_records = []
    if temporal_summary is not None and not temporal_summary.empty:
        for _, row in temporal_summary.iterrows():
            rec = {}
            for col in [
                "period", "precinct_norm", "neighborhood_scope", "selection_type", "selection_name",
                "total_incidents", "peak_day", "peak_day_count", "peak_hour",
                "peak_hour_count", "peak_shift", "peak_shift_count",
                "peak_time_block", "peak_time_block_count", "period_start", "period_end",
            ]:
                rec[col] = clean_number(row.get(col))
            rec["period"] = str(row.get("period", ""))
            rec["precinct_norm"] = str(row.get("precinct_norm", ""))
            rec["selection_type"] = str(row.get("selection_type", ""))
            rec["selection_name"] = str(row.get("selection_name", ""))
            for col in ["peak_day", "peak_shift", "peak_time_block", "period_start", "period_end"]:
                rec[col] = str(row.get(col, ""))
            temporal_summary_records.append(rec)

    temporal_matrix_records = []
    if temporal_matrix is not None and not temporal_matrix.empty:
        for _, row in temporal_matrix.iterrows():
            rec = {
                "period": str(row.get("period", "")),
                "precinct_norm": str(row.get("precinct_norm", "")),
                "neighborhood_scope": str(row.get("neighborhood_scope", "ALL")),
                "selection_type": str(row.get("selection_type", "")),
                "selection_name": str(row.get("selection_name", "")),
                "weekday": str(row.get("weekday", "")),
                "time_block": str(row.get("time_block", "")),
                "incident_count": clean_number(row.get("incident_count")),
            }
            temporal_matrix_records.append(rec)

    hotspot_change_records = []
    if hotspot_change is not None and not hotspot_change.empty:
        wanted_hotspot = [
            "precinct_norm", "neighborhood_scope", "selection_type", "selection_name", "h3_cell",
            "previous_14d", "current_14d", "change_14d", "pct_change_14d",
            "hotspot_status", "hotspot_score", "previous_hotspot_threshold",
            "current_hotspot_threshold", "neighborhood", "nearest_intersection",
            "latitude", "longitude", "previous_14d_start", "previous_14d_end",
            "current_14d_start", "current_14d_end",
        ]
        for _, row in hotspot_change.iterrows():
            rec = {col: clean_number(row.get(col)) for col in wanted_hotspot}
            for col in [
                "precinct_norm", "neighborhood_scope", "selection_type", "selection_name", "h3_cell",
                "hotspot_status", "neighborhood", "nearest_intersection",
                "previous_14d_start", "previous_14d_end", "current_14d_start", "current_14d_end",
            ]:
                rec[col] = str(row.get(col, ""))
            hotspot_change_records.append(rec)

    spatial_daily_records = []
    if spatial_daily is not None and not spatial_daily.empty:
        for _, row in spatial_daily.iterrows():
            spatial_daily_records.append({
                "date": str(row.get("date", "")),
                "weekday": str(row.get("weekday", "")),
                "hour": int(row.get("hour", -1) or -1),
                "h3": str(row.get("h3", "")),
                "precinct": str(row.get("precinct", "")),
                "neighborhood": str(row.get("neighborhood", "Unknown")),
                "crime": str(row.get("crime", "")),
                "violent": bool(row.get("violent", False)),
                "property": bool(row.get("property", False)),
                "vehicle": bool(row.get("vehicle", False)),
                "count": int(row.get("count", 0) or 0),
                "lat": clean_number(row.get("lat")),
                "lon": clean_number(row.get("lon")),
                "intersection": str(row.get("intersection", "Unknown")),
            })

    overall_json = json.dumps(overall_data, ensure_ascii=False).replace("</", "<\\/")
    crime_trend_json = json.dumps(crime_trend_records, ensure_ascii=False).replace("</", "<\\/")
    crime_14d_json = json.dumps(crime_14d_records, ensure_ascii=False).replace("</", "<\\/")
    priority_json = json.dumps(priority_records, ensure_ascii=False).replace("</", "<\\/")
    temporal_summary_json = json.dumps(temporal_summary_records, ensure_ascii=False).replace("</", "<\\/")
    temporal_matrix_json = json.dumps(temporal_matrix_records, ensure_ascii=False).replace("</", "<\\/")
    hotspot_change_json = json.dumps(hotspot_change_records, ensure_ascii=False).replace("</", "<\\/")
    spatial_daily_json = json.dumps(spatial_daily_records, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")

    year_labels = {
        "current": int(current_year) if current_year is not None else None,
        "previous": int(previous_year) if previous_year is not None else None,
        "baseline": int(baseline_year) if baseline_year is not None else None,
    }
    year_json = json.dumps(year_labels)
    precinct_bounds_json = json.dumps(precinct_bounds or {}, ensure_ascii=False)
    neighborhood_bounds_json = json.dumps(neighborhood_bounds or {}, ensure_ascii=False)

    panel_html = f"""
    <style>
        .leaflet-control-layers-base {{ display: none !important; }}
        #cpTrendCard table {{ width:100%; border-collapse:collapse; margin-top:6px; }}
        #cpTrendCard th, #cpTrendCard td {{ padding:4px 6px; border-bottom:1px solid #e2e8f0; text-align:right; }}
        #cpTrendCard th:first-child, #cpTrendCard td:first-child {{ text-align:left; }}
        #cpTrendCard .trend-up {{ color:#b91c1c; font-weight:700; }}
        #cpTrendCard .trend-down {{ color:#047857; font-weight:700; }}
        #cpTrendCard .trend-stable {{ color:#475569; font-weight:700; }}
        #cpPriorityCard table {{ width:100%; border-collapse:collapse; margin-top:6px; }}
        #cpPriorityCard th, #cpPriorityCard td {{ padding:4px 6px; border-bottom:1px solid #e2e8f0; text-align:right; }}
        #cpPriorityCard th:first-child, #cpPriorityCard td:first-child {{ text-align:left; }}
        #cpPriorityCard .priority-high {{ color:#991b1b; font-weight:800; }}
        #cpPriorityCard .priority-emerging {{ color:#c2410c; font-weight:800; }}
        #cpPriorityCard .priority-watch {{ color:#a16207; font-weight:700; }}
        #cpPriorityCard .priority-improving {{ color:#047857; font-weight:700; }}
        #cpTimingCard table {{ width:100%; border-collapse:collapse; margin-top:6px; }}
        #cpTimingCard th, #cpTimingCard td {{ padding:4px 6px; border:1px solid #e2e8f0; text-align:center; }}
        #cpTimingCard th:first-child, #cpTimingCard td:first-child {{ text-align:left; }}
        #cpTimingCard .timing-peak {{ font-weight:800; background:#e0f2fe; }}
        #cpTimingCard .timing-kpis {{ display:grid; grid-template-columns:repeat(4,minmax(120px,1fr)); gap:8px; margin-top:6px; }}
        #cpTimingCard .timing-kpi {{ border:1px solid #bae6fd; border-radius:7px; padding:6px 8px; background:#f0f9ff; }}
        #cpTimingCard .timing-kpi b {{ display:block; color:#075985; margin-bottom:2px; }}
        #cpHotspotCard table {{ width:100%; border-collapse:collapse; margin-top:6px; }}
        #cpHotspotCard th, #cpHotspotCard td {{ padding:4px 6px; border-bottom:1px solid #e2e8f0; text-align:right; }}
        #cpHotspotCard th:first-child, #cpHotspotCard td:first-child {{ text-align:left; }}
        #cpHotspotCard .hotspot-new {{ color:#991b1b; font-weight:800; }}
        #cpHotspotCard .hotspot-emerging {{ color:#c2410c; font-weight:800; }}
        #cpHotspotCard .hotspot-persistent {{ color:#7c3aed; font-weight:800; }}
        #cpHotspotCard .hotspot-declining {{ color:#047857; font-weight:800; }}
    </style>
    <div id="cpPanel" style="
        position: fixed; top:16px; left:50%; transform:translateX(-50%); z-index:9999;
        background:rgba(255,255,255,0.97); border:1px solid #d1d9e6; border-radius:10px;
        padding:10px 12px; width:min(1240px, calc(100vw - 32px)); font-size:12px; color:#0f172a;
        box-shadow:0 4px 12px rgba(15,23,42,0.12); max-height:46vh; overflow:auto;
    ">
      <div style="display:flex;flex-wrap:wrap;gap:10px;align-items:end;">
        <div style="min-width:180px;flex:1;"><div style="font-weight:800;margin-bottom:4px;">Precinct / Responsibility</div>
          <select id="cpPrecinctSelect" style="width:100%;padding:7px;border:1px solid #94a3b8;border-radius:6px;font-weight:700;">
            <option value="">Detroit Overview</option>{precinct_options}</select></div>
                <div style="min-width:220px;flex:2;"><div style="font-weight:800;margin-bottom:4px;">Neighborhood</div>
                    <select id="cpNeighborhoodSelect" style="width:100%;padding:7px;border:1px solid #94a3b8;border-radius:6px;">
                        <option value="">All Neighborhoods</option>{neighborhood_options}</select></div>
        <div style="min-width:300px;flex:3;"><div style="font-weight:800;margin-bottom:4px;">Crime / Category</div>
          <select id="cpCategorySelect" style="width:100%;padding:7px;border:1px solid #94a3b8;border-radius:6px;">
            <option value="">All Crime</option>{category_options_html}</select></div>
        <div style="min-width:190px;flex:1;"><div style="font-weight:800;margin-bottom:4px;">Analysis View</div>
          <select id="cpPeriodSelect" style="width:100%;padding:7px;border:1px solid #94a3b8;border-radius:6px;">
            <option value="Operational">Operational Summary</option>
            <option value="Recent">Recent 14-Day Emphasis</option>
            <option value="YTD">Matched YTD Emphasis</option>
          </select></div>
        <div style="display:flex;gap:6px;align-items:center;">
          <button onclick="window.backToDetroit()" style="padding:7px 10px;border:1px solid #0f172a;border-radius:6px;background:#0f172a;color:#fff;cursor:pointer;">Detroit Overview</button>
          <button onclick="window.clearTopSelectors()" style="padding:7px 10px;border:1px solid #cbd5e1;border-radius:6px;background:#fff;cursor:pointer;">Reset</button>
        </div>
      </div>
      <div style="margin-top:5px;color:#64748b;font-size:11px;">Operational precinct choices exclude unverified codes such as 00/0W; those records remain in Detroit-wide totals.</div>
      <details style="margin-top:8px;padding-top:7px;border-top:1px solid #e2e8f0;">
        <summary style="cursor:pointer;font-weight:700;color:#475569;">Advanced map layers</summary>
        <div style="margin-top:7px;display:flex;flex-wrap:wrap;gap:10px;align-items:end;">
          <div style="min-width:180px;flex:1;"><div style="font-weight:600;margin-bottom:4px;color:#64748b;">Primary map view</div>
            <select id="cpCoreSelect" style="width:100%;padding:6px;border:1px solid #cbd5e1;border-radius:6px;">
              <option value="Core | Incident Density Heatmap" selected>Street Map + Hotspot Heatmap</option>
              <option value="">Street Map Only</option>
              <option value="Core | Spike Week Markers">Spike Week Markers</option>
              <option value="Core | H3 Choropleth: Incident Count (All Incidents)">H3 Incident Count</option>
              <option value="Core | H3 Choropleth: Spike Severity (All Incidents)">H3 Spike Severity</option>
            </select></div>
          <div style="min-width:170px;flex:1;"><div style="font-weight:600;margin-bottom:4px;color:#64748b;">Location markers</div>
            <select id="cpActionSelect" style="width:100%;padding:6px;border:1px solid #cbd5e1;border-radius:6px;">
              <option value="">None</option><option value="Action | Top Intersection Markers">Top Intersections</option>
              <option value="Action | Focus Location Markers">Focus Locations</option>
            </select></div>
          <div style="min-width:180px;flex:1;"><div style="font-weight:600;margin-bottom:4px;color:#64748b;">Optional response lens</div>
            <select id="cpDecisionSelect" style="width:100%;padding:6px;border:1px solid #cbd5e1;border-radius:6px;">
              <option value="">None</option><option value="Decision | Preventive Patrol Priority">Preventive Patrol</option>
              <option value="Decision | Investigations Priority">Investigations</option>
              <option value="Decision | Community Response Priority">Community Response</option>
            </select></div>
          <div style="color:#64748b;font-size:11px;max-width:270px;">Use the Leaflet layer button at the upper-right of the map for the optional latest-14-day incident point layer and alternate basemaps.</div>
        </div>
      </details>
      <div id="cpSelectorStatus" style="margin-top:8px;font-size:11px;color:#334155;">Detroit Overview | All Crime | Operational Summary</div>
      <div id="cpKpiStrip" style="margin-top:8px;display:grid;grid-template-columns:repeat(4,minmax(135px,1fr));gap:8px;"></div>
      <div id="cpExecutiveCard" style="margin-top:8px;padding:9px 11px;border:1px solid #cbd5e1;border-radius:8px;background:#ffffff;">
        <div class="exec-title">Executive Summary</div><div style="margin-top:3px;color:#64748b;">Select a precinct and/or crime type to generate a concise what–when–where–change summary.</div>
      </div>
      <div id="cpPriorityCard" style="margin-top:8px;padding:8px 10px;border:1px solid #fed7aa;border-radius:8px;background:#fff7ed;">
        <b>Priority / Emerging Concerns</b><div style="margin-top:3px;color:#64748b;">Ranks crime signals using recent volume, absolute change, 14-day percentage change, citywide divergence, and YTD direction. Small baselines are kept visible but do not rank on percentage alone.</div>
      </div>
      <div id="cpTimingCard" style="margin-top:8px;padding:8px 10px;border:1px solid #bae6fd;border-radius:8px;background:#f0f9ff;">
        <b>When is it happening?</b><div style="margin-top:3px;color:#64748b;">Shows peak day, hour, shift, and day/time concentration for the current selection, with YTD context and the latest 14 days.</div>
      </div>
      <div id="cpHotspotCard" style="margin-top:8px;padding:8px 10px;border:1px solid #ddd6fe;border-radius:8px;background:#faf5ff;">
        <b>Where is it changing?</b><div style="margin-top:3px;color:#64748b;">Compares H3 hotspot locations across consecutive 14-day periods and identifies persistent, new, emerging, and declining concentrations for the current selection.</div>
      </div>
      <div id="cpTrendCard" style="margin-top:8px;padding:8px 10px;border:1px solid #e2e8f0;border-radius:8px;background:#f8fafc;">
        Select a precinct and/or crime type to see matched YTD trend details and recent 14-day movement. Citywide results are context; precinct results are the operational workload.
      </div>
      <div style="margin-top:4px;font-size:11px;color:#64748b;">YTD Trend uses the latest current-year incident date as the cutoff and compares the same calendar period in prior years. ±2% is treated as Stable. Recent movement compares the latest 14 days with the immediately preceding 14 days.</div>
    </div>
    <script>
    (function() {{
      var overallData = {overall_json};
      var crimeTrendData = {crime_trend_json};
      var crime14dData = {crime_14d_json};
      var priorityData = {priority_json};
      var temporalSummaryData = {temporal_summary_json};
      var temporalMatrixData = {temporal_matrix_json};
      var hotspotChangeData = {hotspot_change_json};
      var spatialDailyData = {spatial_daily_json};
      var customSpatialRange = null;
      var customSpatialLayer = null;
      var mapObjectName = "{m.get_name()}";
      var years = {year_json};
      var precinctBounds = {precinct_bounds_json};
    var neighborhoodBounds = {neighborhood_bounds_json};

      function normalizeLayerName(text) {{ var clean=(text||'').trim(); var idx=clean.lastIndexOf(' ('); return idx>0?clean.slice(0,idx).trim():clean; }}
      function eachOverlayCheckbox(callback) {{ document.querySelectorAll('.leaflet-control-layers-overlays label').forEach(function(label) {{ var cb=label.querySelector('input[type="checkbox"]'); if(cb) callback(cb,(label.innerText||'').trim(),normalizeLayerName(label.innerText||'')); }}); }}
      function setChecked(cb, yes) {{ if(cb.checked!==yes) cb.click(); }}
      function setExclusiveByPrefix(prefix, selected) {{ eachOverlayCheckbox(function(cb,_t,b) {{ if(b.startsWith(prefix)) setChecked(cb, Boolean(selected && b===selected)); }}); }}
      function fmtN(v) {{ return (v===null || v===undefined || Number.isNaN(Number(v))) ? '—' : Number(v).toLocaleString(); }}
      function fmtPct(v) {{ if(v===null || v===undefined || Number.isNaN(Number(v))) return '—'; var n=Number(v); return (n>0?'+':'')+n.toFixed(1)+'%'; }}
      function trendClassName(t) {{ if((t||'').includes('Improving')) return 'trend-down'; if((t||'').includes('Worsening')) return 'trend-up'; return 'trend-stable'; }}
      function trendMatches(actual, requested) {{ if(!requested) return true; if(requested==='Improving') return actual==='Improving'||actual==='Consistently Improving'; if(requested==='Worsening') return actual==='Worsening'||actual==='Consistently Worsening'; return actual===requested; }}
      function selectedCrime() {{ var raw=(document.getElementById('cpCategorySelect')||{{value:''}}).value; return raw.startsWith('Crime Type | ')?raw.replace('Crime Type | ',''):''; }}
    function selectedNeighborhood() {{ return (document.getElementById('cpNeighborhoodSelect')||{{value:''}}).value; }}
    function scopedRecord(r) {{ var neighborhood=selectedNeighborhood(); return r.neighborhood_scope===(neighborhood||'ALL'); }}
    function overallFor(precinct) {{ return overallData[precinct+'|'+(selectedNeighborhood()||'ALL')]; }}
    function rowsFor(precinct, crime, trend) {{ return crimeTrendData.filter(function(r) {{ return scopedRecord(r) && (!precinct||r.precinct_norm===precinct) && (!crime||r.offense_category===crime) && trendMatches(r.trend_class,trend); }}); }}
    function rows28For(precinct, crime) {{
      if(!customSpatialRange || !spatialDailyData.length) return crime14dData.filter(function(r) {{ return scopedRecord(r) && (!precinct||r.precinct_norm===precinct) && (!crime||r.offense_category===crime); }});
      var neighborhood=selectedNeighborhood(), grouped={{}}, city={{}};
      spatialDailyData.forEach(function(r) {{
        if(neighborhood && r.neighborhood!==neighborhood) return;
        if(crime && r.crime!==crime) return;
        var isPrev=r.date>=customSpatialRange.prevStart&&r.date<=customSpatialRange.prevEnd, isCurr=r.date>=customSpatialRange.currStart&&r.date<=customSpatialRange.currEnd;
        if(!isPrev&&!isCurr)return;
        var key=(precinct?r.crime:r.precinct+'|'+r.crime), g=grouped[key]||(grouped[key]={{precinct_norm:r.precinct,offense_category:r.crime,previous_14d:0,current_14d:0}});
        if(!precinct || r.precinct===precinct) {{ if(isPrev)g.previous_14d+=Number(r.count||0); if(isCurr)g.current_14d+=Number(r.count||0); }}
        var c=city[r.crime]||(city[r.crime]={{previous:0,current:0}}); if(isPrev)c.previous+=Number(r.count||0); if(isCurr)c.current+=Number(r.count||0);
      }});
      return Object.values(grouped).filter(function(g){{return !precinct||g.precinct_norm===precinct;}}).map(function(g){{
        var c=city[g.offense_category]||{{previous:0,current:0}}, ch=g.current_14d-g.previous_14d;
        g.change_14d=ch; g.pct_change_14d=g.previous_14d>0?100*ch/g.previous_14d:(g.current_14d>0?null:0);
        g.city_pct_change_14d=c.previous>0?100*(c.current-c.previous)/c.previous:(c.current>0?null:0);
        g.recent_movement=g.pct_change_14d===null?(g.current_14d>0?'Increasing':'Stable'):(g.pct_change_14d>2?'Increasing':g.pct_change_14d<-2?'Decreasing':'Stable');
        g.previous_14d_start=customSpatialRange.prevStart;g.previous_14d_end=customSpatialRange.prevEnd;g.current_14d_start=customSpatialRange.currStart;g.current_14d_end=customSpatialRange.currEnd;
        return g;
      }});
    }}
      function recentClassName(t) {{ if(t==='Increasing') return 'trend-up'; if(t==='Decreasing') return 'trend-down'; return 'trend-stable'; }}
      function recentTableHtml(rows, firstCol, firstLabel, limit) {{
        var use=rows.slice(0,limit||12); if(!use.length) return '<div style="color:#64748b;">No 14-day comparison records for this selection.</div>';
        var prevHead=customSpatialRange?'Previous Range':'Previous 14D', currHead=customSpatialRange?'Current Range':'Current 14D', chgHead=customSpatialRange?'Range %chg':'14D %chg'; var h='<table><thead><tr><th>'+firstLabel+'</th><th>'+prevHead+'</th><th>'+currHead+'</th><th>'+chgHead+'</th><th>City %chg</th><th>Recent</th></tr></thead><tbody>';
        use.forEach(function(r) {{ h+='<tr><td>'+r[firstCol]+'</td><td>'+fmtN(r.previous_14d)+'</td><td>'+fmtN(r.current_14d)+'</td><td>'+fmtPct(r.pct_change_14d)+'</td><td>'+fmtPct(r.city_pct_change_14d)+'</td><td class="'+recentClassName(r.recent_movement)+'">'+r.recent_movement+'</td></tr>'; }});
        return h+'</tbody></table>';
      }}
      function recentWindowText(rows) {{ if(!rows.length) return ''; var r=rows[0]; return ' | '+(customSpatialRange?'Selected range ':'Recent window ')+r.current_14d_start+' to '+r.current_14d_end+' vs '+r.previous_14d_start+' to '+r.previous_14d_end; }}
      function interpretationHtml(ytdRows, recentRows, precinct, crime) {{
        if(!ytdRows.length || !recentRows.length) return '';
        var y=ytdRows[0], r=recentRows[0];
        var recent=Number(r.pct_change_14d), city=Number(r.city_pct_change_14d), ytd=Number(y.pct_change_vs_previous);
        var recentValid=!Number.isNaN(recent), cityValid=!Number.isNaN(city), ytdValid=!Number.isNaN(ytd);
        var signal='MIXED', cls='trend-stable';
        if(recentValid && recent < -2) {{ signal='IMPROVING'; cls='trend-down'; }}
        else if(recentValid && recent > 2) {{ signal='WORSENING'; cls='trend-up'; }}
        else if(recentValid) {{ signal='STABLE'; cls='trend-stable'; }}
        var subject='Recent '+crime.toLowerCase()+' incidents in Precinct '+precinct;
        var text=subject+' '+(recent<0?'decreased ':'increased ')+(recent<0?Math.abs(recent).toFixed(1)+'%':fmtPct(recent))+ ' ('+fmtN(r.previous_14d)+' → '+fmtN(r.current_14d)+')';
        if(cityValid) text+=', compared with a citywide change of '+fmtPct(city);
        if(ytdValid) text+=', while the precinct is '+fmtPct(ytd)+' versus '+years.previous+' YTD';
        var relative='';
        if(recentValid && cityValid) {{
          if(recent < city) relative=' Recent movement is more favorable than the citywide trend.';
          else if(recent > city) relative=' Recent movement is less favorable than the citywide trend.';
          else relative=' Recent movement matches the citywide trend.';
        }}
        return '<div style="margin-top:10px;padding:8px 10px;border-left:4px solid #94a3b8;background:#fff;border-radius:6px;"><b>Interpretation / Signal: <span class="'+cls+'">'+signal+'</span></b><div style="margin-top:3px;color:#334155;">'+text+'.'+relative+'</div></div>';
      }}
      function tableHtml(rows, firstCol, firstLabel, limit) {{
        var use=rows.slice(0,limit||12); if(!use.length) return '<div style="color:#64748b;">No matched trend records for this selection.</div>';
        var h='<table><thead><tr><th>'+firstLabel+'</th><th>'+years.baseline+'</th><th>'+years.previous+'</th><th>'+years.current+'</th><th>vs '+years.previous+'</th><th>Trend</th></tr></thead><tbody>';
        use.forEach(function(r) {{ h+='<tr><td>'+r[firstCol]+'</td><td>'+fmtN(r.incidents_baseline)+'</td><td>'+fmtN(r.incidents_previous)+'</td><td>'+fmtN(r.incidents_current)+'</td><td>'+fmtPct(r.pct_change_vs_previous)+'</td><td class="'+trendClassName(r.trend_class)+'">'+r.trend_class+'</td></tr>'; }});
        return h+'</tbody></table>';
      }}
      function priorityClassName(t) {{ if(t==='High Priority') return 'priority-high'; if(t==='Emerging Concern') return 'priority-emerging'; if(t==='Watch') return 'priority-watch'; if(t==='Recent Improvement') return 'priority-improving'; return 'trend-stable'; }}
    function priorityRowsFor(precinct, crime) {{
      var base=priorityData.filter(function(r) {{ return scopedRecord(r) && (!precinct||r.precinct_norm===precinct) && (!crime||r.offense_category===crime); }});
      if(!customSpatialRange) return base;
      var recent=rows28For(precinct,crime), lookup={{}}; recent.forEach(function(r){{lookup[r.precinct_norm+'|'+r.offense_category]=r;}});
      return base.map(function(src){{var r=Object.assign({{}},src), d=lookup[r.precinct_norm+'|'+r.offense_category]; if(!d)return r;
        r.previous_14d=d.previous_14d;r.current_14d=d.current_14d;r.change_14d=d.change_14d;r.pct_change_14d=d.pct_change_14d;r.city_pct_change_14d=d.city_pct_change_14d;
        var volume=Number(r.current_14d||0), absChange=Number(r.change_14d||0), pct=r.pct_change_14d, cityGap=(pct===null||r.city_pct_change_14d===null)?0:pct-Number(r.city_pct_change_14d||0), ytd=Number(r.pct_change_vs_previous||0), zeroBaseline=Number(r.previous_14d||0)===0&&volume>0;
        var high=(volume>=20)&&(absChange>=10)&&((pct!==null&&pct>=25)||zeroBaseline)&&((cityGap>=10)||(ytd>2));
        var emerging=(volume>=10)&&(absChange>=5)&&((pct!==null&&pct>=10)||zeroBaseline)&&((cityGap>=5)||(ytd>2));
        var watch=(volume>=5)&&(absChange>0)&&((pct!==null&&pct>2)||zeroBaseline); var improving=(volume>=5)&&(absChange<=-5)&&(pct!==null&&pct<=-10);
        r.priority_signal=high?'High Priority':emerging?'Emerging Concern':watch?'Watch':improving?'Recent Improvement':'Monitor';
        r.priority_score=volume+Math.max(absChange,0)*1.5+Math.max(pct||0,0)*0.08+Math.max(cityGap,0)*0.08+Math.max(ytd,0)*0.04; return r;
      }});
    }}
      function priorityTableHtml(rows, limit) {{
        var use=rows.slice(0,limit||8); if(!use.length) return '<div style="color:#64748b;margin-top:4px;">No priority signals for this selection.</div>';
        var prevHead=customSpatialRange?'Prev Range':'Prev 14D', currHead=customSpatialRange?'Current Range':'Current 14D', chgHead=customSpatialRange?'Range %chg':'14D %chg'; var h='<table><thead><tr><th>Precinct / Crime</th><th>'+prevHead+'</th><th>'+currHead+'</th><th>Abs Δ</th><th>'+chgHead+'</th><th>City %chg</th><th>YTD %chg</th><th>Score</th><th>Signal</th></tr></thead><tbody>';
        use.forEach(function(r) {{ var label='P'+r.precinct_norm+' — '+r.offense_category; h+='<tr><td>'+label+'</td><td>'+fmtN(r.previous_14d)+'</td><td>'+fmtN(r.current_14d)+'</td><td>'+fmtN(r.change_14d)+'</td><td>'+fmtPct(r.pct_change_14d)+'</td><td>'+fmtPct(r.city_pct_change_14d)+'</td><td>'+fmtPct(r.pct_change_vs_previous)+'</td><td>'+Number(r.priority_score||0).toFixed(1)+'</td><td class="'+priorityClassName(r.priority_signal)+'">'+r.priority_signal+'</td></tr>'; }});
        return h+'</tbody></table>';
      }}
      function scopeDisplayLabel(precinct,scope) {{
        return (precinct?'Precinct '+precinct:'Citywide')+' — '+(scope.type==='All'?'All incidents':scope.name);
      }}
      function executiveSignalClass(signal) {{
        if(signal==='HIGH PRIORITY'||signal==='WORSENING') return 'priority-high';
        if(signal==='EMERGING CONCERN'||signal==='WATCH') return 'priority-emerging';
        if(signal==='IMPROVING') return 'priority-improving';
        return 'trend-stable';
      }}
      function renderExecutiveSummary() {{
        var precinct=(document.getElementById('cpPrecinctSelect')||{{value:''}}).value;
        var scope=temporalScope(); var crime=selectedCrime();
        var card=document.getElementById('cpExecutiveCard'); if(!card) return;
        var bullets=[]; var signal='MONITOR';

        // Change: use matched YTD + recent 14-day movement when the scope supports it.
        if(precinct && crime) {{
          var yrows=rowsFor(precinct,crime,''); var rrows=rows28For(precinct,crime);
          if(yrows.length) bullets.push('<b>Long-term:</b> '+crime+' is <span class="'+trendClassName(yrows[0].trend_class)+'">'+yrows[0].trend_class+'</span> YTD ('+fmtPct(yrows[0].pct_change_vs_previous)+' vs '+years.previous+').');
          if(rrows.length) {{
            var rr=rrows[0]; bullets.push('<b>Recent:</b> '+fmtN(rr.previous_14d)+' → '+fmtN(rr.current_14d)+' in '+(customSpatialRange?'the selected comparison periods':'consecutive 14-day periods')+' ('+fmtPct(rr.pct_change_14d)+'), versus '+fmtPct(rr.city_pct_change_14d)+' citywide.');
            if(Number(rr.pct_change_14d)<-2) signal='IMPROVING'; else if(Number(rr.pct_change_14d)>2) signal='WORSENING';
          }}
        }} else if(precinct && scope.type==='All') {{
          var ov=overallFor(precinct);
          if(ov) bullets.push('<b>Long-term:</b> Precinct '+precinct+' is <span class="'+trendClassName(ov.trend_class)+'">'+ov.trend_class+'</span> overall ('+fmtPct(ov.pct_change_vs_previous)+' vs '+years.previous+' YTD; data through '+ov.comparison_date+').');
        }} else if(scope.type==='Category Focus') {{
          bullets.push('<b>Scope:</b> '+scope.name+' is a record-level category focus. Timing and hotspot findings below use only records in that focus; offense-level YTD tables remain separate.');
        }}

        // Priority: only use priority rows when the scope is all crimes or one exact crime type.
        if(scope.type!=='Category Focus') {{
          var prows=priorityRowsFor(precinct,crime); prows.sort(function(a,b){{return Number(b.priority_score||0)-Number(a.priority_score||0);}});
          if(prows.length) {{
            var p=prows[0];
            if(p.priority_signal==='High Priority') signal='HIGH PRIORITY'; else if(p.priority_signal==='Emerging Concern' && signal==='MONITOR') signal='EMERGING CONCERN'; else if(p.priority_signal==='Watch' && signal==='MONITOR') signal='WATCH';
            bullets.push('<b>Attention:</b> '+(crime?crime:('P'+p.precinct_norm+' '+p.offense_category))+' is the leading priority signal for this view ('+p.priority_signal+', score '+Number(p.priority_score||0).toFixed(1)+').');
          }}
        }}

        // When: latest 14-day timing profile for the exact current map-analysis scope.
        var recent=timingSummary('Recent 14D',precinct,scope);
        if(recent) bullets.push('<b>When:</b> Peak day is '+recent.peak_day+', peak hour '+hourLabel(recent.peak_hour)+', with '+recent.peak_time_block+' as the busiest time block and '+recent.peak_shift+' as the dominant shift.');

        // Where: summarize hotspot state without repeating the detail table.
        var hrows=hotspotRowsFor(precinct,scope);
        if(hrows.length) {{
          var ne=hrows.filter(function(r){{return r.hotspot_status==='New Hotspot'||r.hotspot_status==='Emerging Hotspot';}}).length;
          var pe=hrows.filter(function(r){{return r.hotspot_status==='Persistent Hotspot';}}).length;
          var de=hrows.filter(function(r){{return r.hotspot_status==='Declining Hotspot';}}).length;
          bullets.push('<b>Where:</b> '+ne+' new/emerging, '+pe+' persistent, and '+de+' declining hotspot cells in the '+(customSpatialRange?'selected comparison':'latest 14-day comparison')+'.');
        }}

        if(!bullets.length) bullets.push('Choose a precinct and/or category/crime type to create a focused operational summary.');
        var title='Executive Summary — '+scopeDisplayLabel(precinct,scope);
        card.innerHTML='<div class="exec-title">'+title+' <span class="exec-signal '+executiveSignalClass(signal)+'">'+signal+'</span></div><ul><li>'+bullets.join('</li><li>')+'</li></ul>';
      }}

      function currentView() {{ var e=document.getElementById('cpPeriodSelect'); return e?e.value:'Operational'; }}
      function renderKpis() {{
        var precinct=(document.getElementById('cpPrecinctSelect')||{{value:''}}).value;
        var strip=document.getElementById('cpKpiStrip'); if(!strip) return;
        if(!precinct) {{
          strip.innerHTML='<div style="grid-column:1/-1;padding:8px 10px;border:1px solid #dbeafe;border-radius:8px;background:#eff6ff;color:#1e3a8a;"><b>Detroit Overview:</b> choose a precinct to switch into a responsibility-focused operational view. Citywide values remain comparison context, not the precinct workload.</div>';
          return;
        }}
        var ov=overallFor(precinct);
        var recent=rows28For(precinct,'');
        var prev=0,curr=0; recent.forEach(function(r){{prev+=Number(r.previous_14d||0);curr+=Number(r.current_14d||0);}});
        var rpct=prev>0?100*(curr-prev)/prev:null;
        var actionable=priorityRowsFor(precinct,'').filter(function(r){{return r.priority_signal==='High Priority'||r.priority_signal==='Emerging Concern'||r.priority_signal==='Watch';}}).length;
        function k(label,value,sub){{return '<div style="border:1px solid #dbeafe;border-radius:8px;padding:7px 9px;background:#f8fbff;"><div style="font-size:10px;text-transform:uppercase;letter-spacing:.04em;color:#64748b;font-weight:700;">'+label+'</div><div style="font-size:18px;font-weight:800;color:#0f172a;margin-top:2px;">'+value+'</div><div style="font-size:10px;color:#64748b;">'+sub+'</div></div>';}}
        strip.innerHTML=k('YTD incidents',ov?fmtN(ov.incidents_current):'—','Precinct '+precinct)+
          k('vs '+years.previous+' YTD',ov?fmtPct(ov.pct_change_vs_previous):'—',ov?ov.trend_class:'')+
          k('Recent 14D',fmtN(curr),rpct===null?'No prior baseline':fmtPct(rpct)+' vs prior 14D')+
          k('Priority concerns',fmtN(actionable),'High / emerging / watch');
      }}
      function applyViewEmphasis() {{
        var view=currentView();
        var timing=document.getElementById('cpTimingCard'), hot=document.getElementById('cpHotspotCard'), trend=document.getElementById('cpTrendCard');
        if(timing) timing.style.display=(view==='YTD'?'none':'block');
        if(hot) hot.style.display=(view==='YTD'?'none':'block');
        if(trend) trend.style.display='block';
        if(view==='Recent' && trend) trend.style.borderColor='#bae6fd';
        else if(view==='YTD' && trend) trend.style.borderColor='#c4b5fd';
        else if(trend) trend.style.borderColor='#e2e8f0';
      }}

      function renderPriorityCard() {{
        var precinct=(document.getElementById('cpPrecinctSelect')||{{value:''}}).value; var crime=selectedCrime();
        var card=document.getElementById('cpPriorityCard'); if(!card) return;
        var scope=temporalScope();
        if(scope.type==='Category Focus') {{ card.innerHTML='<b>Priority / Emerging Concerns — '+scope.name+'</b><div style="margin-top:3px;color:#64748b;">Priority ranking is offense-level and is intentionally not inferred from this broad category focus. Select a specific crime type for a matched priority score; timing and hotspot sections below remain filtered to '+scope.name+'.</div>'; return; }}
        var rows=priorityRowsFor(precinct,crime);
        rows.sort(function(a,b) {{ return Number(b.priority_score||0)-Number(a.priority_score||0); }});
        var actionable=rows.filter(function(r) {{ return r.priority_signal==='High Priority'||r.priority_signal==='Emerging Concern'||r.priority_signal==='Watch'; }});
        var improvements=rows.filter(function(r) {{ return r.priority_signal==='Recent Improvement'; }}).sort(function(a,b) {{ return Number(a.pct_change_14d||0)-Number(b.pct_change_14d||0); }});
        var title='<b>Priority / Emerging Concerns'+(precinct?' — Precinct '+precinct:'')+(crime?' — '+crime:'')+'</b>';
        if(actionable.length) {{
          card.innerHTML=title+'<div style="margin-top:3px;color:#64748b;">Ranked by recent volume + absolute increase + percentage increase + citywide divergence + YTD direction.</div>'+priorityTableHtml(actionable,8)+(improvements.length?'<div style="margin-top:8px;"><b>Recent improvements worth noting</b></div>'+priorityTableHtml(improvements,4):'');
        }} else if(improvements.length) {{
          card.innerHTML=title+'<div style="margin-top:3px;color:#047857;">No current high-priority increase signals for this selection. Recent improvements are shown below.</div>'+priorityTableHtml(improvements,6);
        }} else {{
          card.innerHTML=title+'<div style="margin-top:3px;color:#64748b;">No material priority signals for this selection based on the current thresholds.</div>';
        }}
      }}

      function temporalScope() {{
        var raw=(document.getElementById('cpCategorySelect')||{{value:''}}).value;
        if(raw.startsWith('Crime Type | ')) return {{type:'Crime Type',name:raw.replace('Crime Type | ','')}};
        if(raw.startsWith('Category Focus | ')) return {{type:'Category Focus',name:raw.replace('Category Focus | ','')}};
        return {{type:'All',name:'All'}};
      }}
      function customTimingRows(precinct,scope) {{
        if(!customSpatialRange||!spatialDailyData.length)return null; var neighborhood=selectedNeighborhood(), rows=[];
        spatialDailyData.forEach(function(r){{if(r.date<customSpatialRange.currStart||r.date>customSpatialRange.currEnd)return;if(!rowMatchesSpatialScope(r,precinct,scope,neighborhood))return;var h=Number(r.hour);if(!Number.isFinite(h)||h<0||h>23)return;rows.push(r);}}); return rows;
      }}
      function timingSummary(period,precinct,scope) {{
        if(period==='Recent 14D'&&customSpatialRange){{var rows=customTimingRows(precinct,scope)||[];if(!rows.length)return null;var day={{}},hour={{}},block={{}},shift={{}};
          rows.forEach(function(r){{var n=Number(r.count||0),h=Number(r.hour);day[r.weekday]=(day[r.weekday]||0)+n;hour[h]=(hour[h]||0)+n;var b=h<6?'00:00-05:59':h<12?'06:00-11:59':h<18?'12:00-17:59':'18:00-23:59';block[b]=(block[b]||0)+n;var sh=(h>=6&&h<=13)?'Day Shift (06:00-13:59)':(h>=14&&h<=21)?'Evening Shift (14:00-21:59)':'Night Shift (22:00-05:59)';shift[sh]=(shift[sh]||0)+n;}});
          function peak(o){{return Object.keys(o).sort(function(a,b){{return o[b]-o[a];}})[0];}} var pd=peak(day),ph=peak(hour),pb=peak(block),ps=peak(shift);return {{peak_day:pd,peak_day_count:day[pd],peak_hour:Number(ph),peak_hour_count:hour[ph],peak_time_block:pb,peak_time_block_count:block[pb],peak_shift:ps,peak_shift_count:shift[ps],period_start:customSpatialRange.currStart,period_end:customSpatialRange.currEnd}};}}
        var p=precinct||'ALL'; var rows=temporalSummaryData.filter(function(r){{return r.period===period && r.precinct_norm===p && r.selection_type===scope.type && r.selection_name===scope.name;}}); rows=rows.filter(scopedRecord); return rows.length?rows[0]:null;
      }}
      function timingMatrix(period,precinct,scope) {{
        if(period==='Recent 14D'&&customSpatialRange){{var rows=customTimingRows(precinct,scope)||[], out={{}};rows.forEach(function(r){{var h=Number(r.hour),b=h<6?'00:00-05:59':h<12?'06:00-11:59':h<18?'12:00-17:59':'18:00-23:59',k=r.weekday+'|'+b;out[k]=(out[k]||0)+Number(r.count||0);}});return Object.keys(out).map(function(k){{var a=k.split('|');return {{weekday:a[0],time_block:a[1],incident_count:out[k]}};}});}}
        var p=precinct||'ALL'; return temporalMatrixData.filter(function(r){{return scopedRecord(r) && r.period===period && r.precinct_norm===p && r.selection_type===scope.type && r.selection_name===scope.name;}});
      }}
      function hourLabel(v) {{ if(v===null||v===undefined||Number.isNaN(Number(v))) return '—'; var h=Number(v); return String(h).padStart(2,'0')+':00'; }}
      function timingKpisHtml(r) {{
        if(!r) return '<div style="color:#64748b;">No timing profile is available for this selection.</div>';
        return '<div class="timing-kpis">'+
          '<div class="timing-kpi"><b>Peak day</b>'+r.peak_day+' ('+fmtN(r.peak_day_count)+')</div>'+
          '<div class="timing-kpi"><b>Peak hour</b>'+hourLabel(r.peak_hour)+' ('+fmtN(r.peak_hour_count)+')</div>'+
          '<div class="timing-kpi"><b>Peak shift</b>'+r.peak_shift+' ('+fmtN(r.peak_shift_count)+')</div>'+
          '<div class="timing-kpi"><b>Peak time block</b>'+r.peak_time_block+' ('+fmtN(r.peak_time_block_count)+')</div>'+
          '</div>';
      }}
      function timingGridHtml(rows) {{
        if(!rows.length) return '<div style="color:#64748b;">No day/time matrix is available for this selection.</div>';
        var days=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'];
        var blocks=['00:00-05:59','06:00-11:59','12:00-17:59','18:00-23:59'];
        var lookup={{}}, maxv=0;
        rows.forEach(function(r){{var k=r.weekday+'|'+r.time_block; var v=Number(r.incident_count||0); lookup[k]=v; if(v>maxv)maxv=v;}});
        var h='<table><thead><tr><th>Day</th>'; blocks.forEach(function(b){{h+='<th>'+b+'</th>';}}); h+='</tr></thead><tbody>';
        days.forEach(function(d){{h+='<tr><td>'+d+'</td>'; blocks.forEach(function(b){{var v=lookup[d+'|'+b]||0; h+='<td class="'+(v===maxv&&maxv>0?'timing-peak':'')+'">'+fmtN(v)+'</td>';}}); h+='</tr>';}});
        return h+'</tbody></table>';
      }}
      function renderTimingCard() {{
        var precinct=(document.getElementById('cpPrecinctSelect')||{{value:''}}).value;
        var scope=temporalScope(); var card=document.getElementById('cpTimingCard'); if(!card) return;
        var ytd=timingSummary('YTD',precinct,scope), recent=timingSummary('Recent 14D',precinct,scope);
        var matrix=timingMatrix('Recent 14D',precinct,scope);
        var label=(precinct?'Precinct '+precinct:'Citywide')+' — '+(scope.type==='All'?'All incidents':scope.name);
        var recentRange=recent?(' | '+recent.period_start+' to '+recent.period_end):'';
        var story='';
        if(recent) story='<div style="margin-top:6px;color:#0f172a;"><b>Recent timing signal:</b> '+recent.peak_day+' is the highest-volume day, '+recent.peak_time_block+' is the busiest time block, and the dominant shift is '+recent.peak_shift+'.</div>';
        card.innerHTML='<b>When is it happening? — '+label+'</b>'+
          '<div style="margin-top:5px;color:#475569;"><b>'+(customSpatialRange?'Selected current range':'Latest 14 days')+'</b>'+recentRange+'</div>'+timingKpisHtml(recent)+story+
          '<div style="margin-top:8px;color:#475569;"><b>'+(customSpatialRange?'Selected-range day × time concentration':'Recent day × time concentration')+'</b> <span style="font-weight:400;">(highest cell highlighted)</span></div>'+timingGridHtml(matrix)+
          '<div style="margin-top:8px;color:#475569;"><b>YTD timing context</b>'+(ytd?(' | '+ytd.period_start+' to '+ytd.period_end):'')+'</div>'+timingKpisHtml(ytd);
      }}

      function hotspotStatusClass(t) {{
        if(t==='New Hotspot') return 'hotspot-new';
        if(t==='Emerging Hotspot') return 'hotspot-emerging';
        if(t==='Persistent Hotspot') return 'hotspot-persistent';
        if(t==='Declining Hotspot') return 'hotspot-declining';
        return 'trend-stable';
      }}
      function quantile(values,q) {{
        var a=values.filter(function(v){{return Number(v)>0;}}).map(Number).sort(function(x,y){{return x-y;}});
        if(!a.length) return 3;
        var pos=(a.length-1)*q, base=Math.floor(pos), rest=pos-base;
        var val=(a[base+1]!==undefined)?a[base]+rest*(a[base+1]-a[base]):a[base];
        return Math.max(3,Math.ceil(val));
      }}
      function rowMatchesSpatialScope(r,precinct,scope,neighborhood) {{
        if(precinct && r.precinct!==precinct) return false;
        if(neighborhood && r.neighborhood!==neighborhood) return false;
        if(scope.type==='Crime Type' && r.crime!==scope.name) return false;
        if(scope.type==='Category Focus') {{
          if(scope.name==='Violent Crime' && !r.violent) return false;
          if(scope.name==='Property Crime' && !r.property) return false;
          if(scope.name==='Vehicle-Related Crime' && !r.vehicle) return false;
        }}
        return true;
      }}
      function dynamicHotspotRows(precinct,scope) {{
        if(!customSpatialRange || !spatialDailyData.length) return null;
        var neighborhood=selectedNeighborhood();
        var byCell={{}};
        spatialDailyData.forEach(function(r) {{
          if(!rowMatchesSpatialScope(r,precinct,scope,neighborhood)) return;
          var isPrev=r.date>=customSpatialRange.prevStart && r.date<=customSpatialRange.prevEnd;
          var isCurr=r.date>=customSpatialRange.currStart && r.date<=customSpatialRange.currEnd;
          if(!isPrev && !isCurr) return;
          var c=byCell[r.h3]||(byCell[r.h3]={{h3_cell:r.h3,previous_14d:0,current_14d:0,latitude:r.lat,longitude:r.lon,neighborhood:r.neighborhood,nearest_intersection:r.intersection}});
          if(isPrev)c.previous_14d+=Number(r.count||0);
          if(isCurr)c.current_14d+=Number(r.count||0);
        }});
        var cells=Object.values(byCell);
        var pt=quantile(cells.map(function(c){{return c.previous_14d;}}),.80);
        var ct=quantile(cells.map(function(c){{return c.current_14d;}}),.80);
        var out=[];
        cells.forEach(function(c) {{
          var ph=c.previous_14d>=pt, ch=c.current_14d>=ct, status='Not Material';
          if(ph&&ch)status='Persistent Hotspot';
          else if(!ph&&ch&&c.previous_14d===0)status='New Hotspot';
          else if(!ph&&ch)status='Emerging Hotspot';
          else if(ph&&!ch)status='Declining Hotspot';
          if(status==='Not Material')return;
          c.hotspot_status=status; c.change_14d=c.current_14d-c.previous_14d;
          c.pct_change_14d=c.previous_14d>0?100*c.change_14d/c.previous_14d:null;
          c.hotspot_score=c.current_14d+1.5*Math.max(c.change_14d,0)+(status==='Persistent Hotspot'?c.current_14d*.35:0);
          c.previous_hotspot_threshold=pt;c.current_hotspot_threshold=ct;
          c.previous_14d_start=customSpatialRange.prevStart;c.previous_14d_end=customSpatialRange.prevEnd;
          c.current_14d_start=customSpatialRange.currStart;c.current_14d_end=customSpatialRange.currEnd;
          c.precinct_norm=precinct||'ALL';c.selection_type=scope.type;c.selection_name=scope.name;out.push(c);
        }});
        return out;
      }}
      function hotspotRowsFor(precinct,scope) {{
        var dynamic=dynamicHotspotRows(precinct,scope);
        if(dynamic!==null) return dynamic;
        var p=precinct||'ALL';
        return hotspotChangeData.filter(function(r) {{
          return scopedRecord(r) && r.precinct_norm===p && r.selection_type===scope.type && r.selection_name===scope.name;
        }});
      }}
      function activeSpatialRange() {{
        if(customSpatialRange) return customSpatialRange;
        if(hotspotChangeData.length) {{
          var r=hotspotChangeData[0];
          if(r.current_14d_start&&r.current_14d_end) return {{
            prevStart:r.previous_14d_start||'', prevEnd:r.previous_14d_end||'',
            currStart:r.current_14d_start, currEnd:r.current_14d_end
          }};
        }}
        return null;
      }}
      function renderCustomSpatialHeatmap() {{
        var mp=window[mapObjectName]; if(!mp) return;
        if(customSpatialLayer){{try{{mp.removeLayer(customSpatialLayer);}}catch(e){{}} customSpatialLayer=null;}}
        if(!spatialDailyData.length) return;
        var precinct=(document.getElementById('cpPrecinctSelect')||{{value:''}}).value;
        var scope=temporalScope(), neighborhood=selectedNeighborhood();
        // The unfiltered default map keeps the compact all-history core heatmap.
        // Any analytical scope is rendered on demand from the current-year daily payload,
        // avoiding thousands of duplicated pre-generated Folium heatmap layers.
        if(!customSpatialRange && !precinct && !neighborhood && scope.type==='All') return;
        var range=activeSpatialRange(); if(!range) return;
        var pts=[];
        spatialDailyData.forEach(function(r){{
          if(r.date<range.currStart||r.date>range.currEnd)return;
          if(!rowMatchesSpatialScope(r,precinct,scope,neighborhood))return;
          if(Number.isFinite(Number(r.lat))&&Number.isFinite(Number(r.lon)))pts.push([Number(r.lat),Number(r.lon),Number(r.count||1)]);
        }});
        if(pts.length && window.L && L.heatLayer){{customSpatialLayer=L.heatLayer(pts,{{radius:18,blur:15,maxZoom:15}}).addTo(mp);}}
      }}
      // Emphasize the location selected from "Where is it changing?" so the
      // analyst does not have to visually search for the target after zooming.
      var hotspotHighlightLayer=null;
      var hotspotPulseTimer=null;
      window.zoomHotspot=function(lat,lon,prev,curr,chg,pct,status,loc,prevStart,prevEnd,currStart,currEnd) {{
        var mp=window[mapObjectName];
        if(!mp || lat===null || lon===null) return;
        var y=Number(lat), x=Number(lon);
        if(!Number.isFinite(y) || !Number.isFinite(x)) return;

        // Clear the previous emphasis before highlighting the new selection.
        if(hotspotPulseTimer) {{ clearInterval(hotspotPulseTimer); hotspotPulseTimer=null; }}
        if(hotspotHighlightLayer) {{
          try {{ mp.removeLayer(hotspotHighlightLayer); }} catch(e) {{}}
          hotspotHighlightLayer=null;
        }}

        mp.flyTo([y,x],16,{{animate:true,duration:0.9}});

        hotspotHighlightLayer=L.layerGroup().addTo(mp);
        var halo=L.circle([y,x],{{
          radius:115,
          color:'#b91c1c',
          weight:3,
          opacity:0.95,
          fillColor:'#f97316',
          fillOpacity:0.12,
          interactive:false
        }}).addTo(hotspotHighlightLayer);
        var target=L.circleMarker([y,x],{{
          radius:9,
          color:'#7f1d1d',
          weight:3,
          opacity:1,
          fillColor:'#facc15',
          fillOpacity:0.95,
          interactive:true
        }}).addTo(hotspotHighlightLayer);
        var pctText=(pct===null||pct===undefined||Number.isNaN(Number(pct)))?'—':(Number(pct)>=0?'+':'')+Number(pct).toFixed(1)+'%';
        var changeText=(Number(chg)>=0?'+':'')+fmtN(chg);
        var popup='<div style="min-width:235px;line-height:1.45"><b>'+String(status||'Hotspot')+'</b><br>'+String(loc||'Selected hotspot')+'<hr style="margin:6px 0;border:0;border-top:1px solid #e2e8f0">'+
          '<b>Previous period:</b> '+fmtN(prev)+' incidents<br><span style="color:#64748b">'+String(prevStart||'')+' to '+String(prevEnd||'')+'</span><br>'+
          '<b>Current period:</b> '+fmtN(curr)+' incidents<br><span style="color:#64748b">'+String(currStart||'')+' to '+String(currEnd||'')+'</span><br>'+
          '<b>Change:</b> '+changeText+' ('+pctText+')</div>';
        target.bindPopup(popup,{{maxWidth:320}}).openPopup();
        target.bindTooltip('<b>Selected hotspot location</b><br>Click for comparison counts',{{permanent:false,direction:'top',offset:[0,-8]}});

        // Pulse the halo a few times, then leave a clear target ring in place.
        var pulse=0;
        hotspotPulseTimer=setInterval(function() {{
          pulse += 1;
          var expanded=(pulse % 2)===1;
          halo.setRadius(expanded ? 180 : 115);
          halo.setStyle({{opacity:expanded?0.35:0.95,fillOpacity:expanded?0.04:0.12}});
          target.setRadius(expanded ? 13 : 9);
          if(pulse>=8) {{
            clearInterval(hotspotPulseTimer); hotspotPulseTimer=null;
            halo.setRadius(115);
            halo.setStyle({{opacity:0.95,fillOpacity:0.12}});
            target.setRadius(9);
          }}
        }},260);
      }};
      function hotspotTableHtml(rows,limit) {{
        var use=rows.slice(0,limit||8);
        if(!use.length) return '<div style="color:#64748b;margin-top:4px;">No material hotspot change locations for this selection.</div>';
        var prevHead=customSpatialRange?'Prev Range':'Prev 14D', currHead=customSpatialRange?'Current Range':'Current 14D';
        var h='<table><thead><tr><th>Status / Location</th><th>'+prevHead+'</th><th>'+currHead+'</th><th>Abs Δ</th><th>%chg</th><th>Map</th></tr></thead><tbody>';
        use.forEach(function(r) {{
          var loc=(r.nearest_intersection && r.nearest_intersection!=='Unknown')?r.nearest_intersection:r.neighborhood;
          var label='<span class="'+hotspotStatusClass(r.hotspot_status)+'">'+r.hotspot_status+'</span><br><span style="color:#475569;">'+loc+'</span>';
          var payload=encodeURIComponent(JSON.stringify([r.latitude,r.longitude,r.previous_14d,r.current_14d,r.change_14d,r.pct_change_14d,r.hotspot_status,loc,r.previous_14d_start,r.previous_14d_end,r.current_14d_start,r.current_14d_end]));
          h+='<tr><td>'+label+'</td><td>'+fmtN(r.previous_14d)+'</td><td>'+fmtN(r.current_14d)+'</td><td>'+fmtN(r.change_14d)+'</td><td>'+fmtPct(r.pct_change_14d)+'</td><td><button type="button" class="cp-hotspot-zoom" data-hotspot="'+payload+'" style="padding:3px 6px;border:1px solid #c4b5fd;border-radius:5px;background:#fff;cursor:pointer;">Zoom</button></td></tr>';
        }});
        return h+'</tbody></table>';
      }}
      function renderHotspotCard() {{
        var precinct=(document.getElementById('cpPrecinctSelect')||{{value:''}}).value;
        var scope=temporalScope();
        var card=document.getElementById('cpHotspotCard'); if(!card) return;
        var rows=hotspotRowsFor(precinct,scope);
        var statusOrder={{'New Hotspot':0,'Emerging Hotspot':1,'Persistent Hotspot':2,'Declining Hotspot':3}};
        rows.sort(function(a,b) {{ var sa=statusOrder[a.hotspot_status]??9, sb=statusOrder[b.hotspot_status]??9; return sa!==sb?sa-sb:Number(b.hotspot_score||0)-Number(a.hotspot_score||0); }});
        var newEmerging=rows.filter(function(r){{return r.hotspot_status==='New Hotspot'||r.hotspot_status==='Emerging Hotspot';}});
        var persistent=rows.filter(function(r){{return r.hotspot_status==='Persistent Hotspot';}}).sort(function(a,b){{return Number(b.current_14d||0)-Number(a.current_14d||0);}});
        var declining=rows.filter(function(r){{return r.hotspot_status==='Declining Hotspot';}}).sort(function(a,b){{return Number(a.change_14d||0)-Number(b.change_14d||0);}});
        var label=(precinct?'Precinct '+precinct:'Citywide')+' — '+(scope.type==='All'?'All incidents':scope.name);
        var range=rows.length?(' | '+rows[0].current_14d_start+' to '+rows[0].current_14d_end+' vs '+rows[0].previous_14d_start+' to '+rows[0].previous_14d_end):'';
        var summary='<div style="margin-top:4px;color:#475569;">'+newEmerging.length+' new/emerging, '+persistent.length+' persistent, '+declining.length+' declining hotspot cells'+range+'.</div>';
        var body='';
        if(newEmerging.length) body+='<div style="margin-top:7px;"><b>New / emerging locations needing attention</b></div>'+hotspotTableHtml(newEmerging,6);
        if(persistent.length) body+='<div style="margin-top:8px;"><b>Persistent concentrations</b></div>'+hotspotTableHtml(persistent,5);
        if(declining.length) body+='<div style="margin-top:8px;"><b>Declining hotspots</b></div>'+hotspotTableHtml(declining,5);
        if(!body) body='<div style="margin-top:5px;color:#64748b;">No cells crossed the hotspot thresholds in either comparison period for this selection.</div>';
        card.innerHTML='<b>Where is it changing? — '+label+'</b>'+summary+body+'<div style="margin-top:5px;color:#64748b;font-size:11px;">Hotspots are relative to the selected precinct/crime scope: cells at or above the 80th percentile of occupied-cell counts, with a minimum of 3 incidents. Use Zoom to inspect the location on the map.</div>';
      }}

      function renderTrendCard() {{
        var precinct=(document.getElementById('cpPrecinctSelect')||{{value:''}}).value;
        var crime=selectedCrime(); var trend='';
        var card=document.getElementById('cpTrendCard'); if(!card) return;
        if(precinct && crime) {{
          var rows=rowsFor(precinct,crime,trend); var recent=rows28For(precinct,crime);
          card.innerHTML='<b>Precinct '+precinct+' — '+crime+'</b>'+
            '<div style="margin-top:5px;color:#475569;"><b>Matched YTD</b></div>'+tableHtml(rows,'offense_category','Crime Type',5)+
            '<div style="margin-top:8px;color:#475569;"><b>Recent 14-day movement</b>'+recentWindowText(recent)+'</div>'+recentTableHtml(recent,'offense_category','Crime Type',5)+
            interpretationHtml(rows,recent,precinct,crime);
          return;
        }}
        if(precinct) {{
          var overall=overallFor(precinct); var allRows=rowsFor(precinct,'',trend); var recentRows=rows28For(precinct,'');
          allRows.sort(function(a,b) {{ return Number(b.pct_change_vs_previous||0)-Number(a.pct_change_vs_previous||0); }});
          var topWorse=allRows.filter(function(r){{return (r.trend_class||'').includes('Worsening');}}).slice(0,5);
          var topBetter=allRows.filter(function(r){{return (r.trend_class||'').includes('Improving');}}).sort(function(a,b){{return Number(a.pct_change_vs_previous||0)-Number(b.pct_change_vs_previous||0);}}).slice(0,5);
          var recentUp=recentRows.filter(function(r){{return r.recent_movement==='Increasing';}}).sort(function(a,b){{return Number(b.pct_change_14d||0)-Number(a.pct_change_14d||0);}}).slice(0,5);
          var recentDown=recentRows.filter(function(r){{return r.recent_movement==='Decreasing';}}).sort(function(a,b){{return Number(a.pct_change_14d||0)-Number(b.pct_change_14d||0);}}).slice(0,5);
          var head='<b>Precinct '+precinct+'</b>';
          if(overall) head+=' — <span class="'+trendClassName(overall.trend_class)+'">Overall '+overall.trend_class+'</span> | '+years.current+' YTD '+fmtN(overall.incidents_current)+' | '+fmtPct(overall.pct_change_vs_previous)+' vs '+years.previous+' | Data through '+overall.comparison_date;
          if(trend) {{
            card.innerHTML=head+'<div style="margin-top:6px;"><b>Crime types matching '+trend+' (YTD):</b></div>'+tableHtml(allRows,'offense_category','Crime Type',15)+
              '<div style="margin-top:8px;"><b>Recent 14-day movement for this precinct</b>'+recentWindowText(recentRows)+'</div>'+recentTableHtml(recentRows,'offense_category','Crime Type',10);
          }} else {{
            card.innerHTML=head+
              '<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:6px;"><div><b>Largest YTD worsening drivers</b>'+tableHtml(topWorse,'offense_category','Crime Type',5)+'</div><div><b>Largest YTD improving drivers</b>'+tableHtml(topBetter,'offense_category','Crime Type',5)+'</div></div>'+
              '<div style="margin-top:8px;color:#475569;"><b>Recent 14-day movement</b>'+recentWindowText(recentRows)+'</div>'+
              '<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:4px;"><div><b>Largest recent increases</b>'+recentTableHtml(recentUp,'offense_category','Crime Type',5)+'</div><div><b>Largest recent decreases</b>'+recentTableHtml(recentDown,'offense_category','Crime Type',5)+'</div></div>';
          }}
          return;
        }}
        if(crime) {{
          var rows=rowsFor('',crime,trend); var recent=rows28For('',crime);
          rows.sort(function(a,b) {{ return Number(b.pct_change_vs_previous||0)-Number(a.pct_change_vs_previous||0); }});
          recent.sort(function(a,b) {{ return Number(b.pct_change_14d||0)-Number(a.pct_change_14d||0); }});
          card.innerHTML='<b>'+crime+' across precincts'+(trend?' — '+trend:'')+'</b>'+
            '<div style="margin-top:5px;color:#475569;"><b>Matched YTD</b></div>'+tableHtml(rows,'precinct_norm','Precinct',20)+
            '<div style="margin-top:8px;color:#475569;"><b>Recent 14-day movement</b>'+recentWindowText(recent)+'</div>'+recentTableHtml(recent,'precinct_norm','Precinct',20);
          return;
        }}
        if(trend) {{ var rows=rowsFor('','',trend); rows.sort(function(a,b) {{ return Math.abs(Number(b.pct_change_vs_previous||0))-Math.abs(Number(a.pct_change_vs_previous||0)); }}); card.innerHTML='<b>All crime types — '+trend+'</b><div style="color:#64748b;margin-top:3px;">Select a precinct or crime type to narrow these results.</div>'+tableHtml(rows,'offense_category','Crime Type',15); return; }}
        card.innerHTML='Select a precinct and/or crime type to see matched year-to-date trend details and recent 14-day movement.';
      }}
      function setStatus(parts) {{ var s=document.getElementById('cpSelectorStatus'); if(s) s.textContent='Active filters: '+parts.join(' | '); }}
      function zoomToPrecinct(precinct) {{
        var mp=window[mapObjectName]; if(!mp) return;
        var neighborhood=selectedNeighborhood();
        var b=neighborhoodBounds[neighborhood]||precinctBounds[precinct||'ALL'];
        if(b && b.length===2) mp.fitBounds(b, {{padding:[24,24], maxZoom:13}});
      }}
      window.applyTopSelectors=function() {{
        var core=(document.getElementById('cpCoreSelect')||{{value:''}}).value, action=(document.getElementById('cpActionSelect')||{{value:''}}).value,
            decision=(document.getElementById('cpDecisionSelect')||{{value:''}}).value, precinct=(document.getElementById('cpPrecinctSelect')||{{value:''}}).value,
            neighborhood=selectedNeighborhood(),
            category=(document.getElementById('cpCategorySelect')||{{value:''}}).value, view=currentView();
        // The operational map is a real street basemap with a scoped heatmap.
        // H3 remains available as an analytical overlay, not the default visual.
        var scoped = Boolean(precinct || neighborhood || category);
        setExclusiveByPrefix('Core | ', scoped && core==='Core | Incident Density Heatmap' ? '' : core);
        setExclusiveByPrefix('Action | ',action); setExclusiveByPrefix('Decision | ',decision);
        setExclusiveByPrefix('Precinct | ','');
        setExclusiveByPrefix('Category Focus | ','');
        setExclusiveByPrefix('Crime Type | ','');
        setExclusiveByPrefix('Scope | Precinct | ','');
                setExclusiveByPrefix('Scope | Neighborhood | ','');
                if(!customSpatialRange) {{
          if(neighborhood) {{
            setExclusiveByPrefix('Scope | Neighborhood | ','Scope | Neighborhood | '+neighborhood+' | Precinct | '+(precinct||'ALL')+' | '+(category||'All Crime'));
          }} else if(precinct && category) {{
            setExclusiveByPrefix('Scope | Precinct | ','Scope | Precinct | '+precinct+' | '+category);
          }} else if(precinct) {{
            setExclusiveByPrefix('Precinct | ','Precinct | '+precinct);
          }} else if(category.startsWith('Category Focus | ')) {{
            setExclusiveByPrefix('Category Focus | ',category);
          }} else if(category.startsWith('Crime Type | ')) {{
            setExclusiveByPrefix('Crime Type | ',category);
          }}
        }}
        setExclusiveByPrefix('Core Type | H3 Count | ', core.startsWith('Core Type | H3 Count | ')?core:'');
        zoomToPrecinct(precinct);
        var crimeLabel=category?category.replace('Category Focus | ','').replace('Crime Type | ',''):'All Crime';
        setStatus([(precinct?'Precinct '+precinct:'Detroit Overview'),(neighborhood||'All Neighborhoods'),crimeLabel,(customSpatialRange?'Custom Range '+customSpatialRange.currStart+' to '+customSpatialRange.currEnd:view==='Recent'?'Recent 14-Day Emphasis':view==='YTD'?'Matched YTD Emphasis':'Operational Summary')]);
        renderKpis();
        renderExecutiveSummary();
        renderPriorityCard();
        renderTimingCard();
        renderHotspotCard();
        renderCustomSpatialHeatmap();
        renderTrendCard();
        applyViewEmphasis();
      }};
      window.backToDetroit=function() {{ var p=document.getElementById('cpPrecinctSelect'); if(p)p.value=''; window.applyTopSelectors(); }};
    window.clearTopSelectors=function() {{ ['cpActionSelect','cpDecisionSelect','cpPrecinctSelect','cpNeighborhoodSelect','cpCategorySelect'].forEach(function(id){{var e=document.getElementById(id);if(e)e.value='';}}); var pe=document.getElementById('cpPeriodSelect'); if(pe)pe.value='Operational'; var ce=document.getElementById('cpCoreSelect'); if(ce) ce.value='Core | Incident Density Heatmap'; setExclusiveByPrefix('Core | ','');setExclusiveByPrefix('Action | ','');setExclusiveByPrefix('Decision | ','');setExclusiveByPrefix('Precinct | ','');setExclusiveByPrefix('Category Focus | ','');setExclusiveByPrefix('Crime Type | ','');setExclusiveByPrefix('Scope | Precinct | ','');setExclusiveByPrefix('Scope | Neighborhood | ','');setExclusiveByPrefix('Core Type | H3 Count | ',''); window.applyTopSelectors(); }};
    ['cpCoreSelect','cpActionSelect','cpDecisionSelect','cpPrecinctSelect','cpNeighborhoodSelect','cpCategorySelect','cpPeriodSelect'].forEach(function(id){{var e=document.getElementById(id);if(e)e.addEventListener('change',window.applyTopSelectors);}});

      window.focusAnalysisSection=function(section) {{
        var panel=document.getElementById('cpPanel');
        if(!panel) return;
        var key=(section||'').toLowerCase();
        if(key==='map') {{
          panel.style.maxHeight='32vh';
          panel.scrollTo({{top:0,behavior:'smooth'}});
          return;
        }}
        var ids={{
          summary:'cpExecutiveCard',
          priority:'cpPriorityCard',
          timing:'cpTimingCard',
          hotspot:'cpHotspotCard',
          trends:'cpTrendCard'
        }};
        var target=document.getElementById(ids[key]||'');
        if(!target) return;
        panel.style.maxHeight='56vh';
        panel.scrollTo({{top:Math.max(target.offsetTop-12,0),behavior:'smooth'}});
        var prior=target.style.boxShadow;
        target.style.boxShadow='0 0 0 3px rgba(37,99,235,0.35), 0 8px 18px rgba(15,23,42,0.12)';
        setTimeout(function(){{target.style.boxShadow=prior;}},2200);
      }};

      document.addEventListener('click',function(ev) {{
        var btn=ev.target.closest && ev.target.closest('.cp-hotspot-zoom');
        if(!btn)return;
        ev.preventDefault(); ev.stopPropagation();
        try{{var a=JSON.parse(decodeURIComponent(btn.getAttribute('data-hotspot')||''));window.zoomHotspot.apply(null,a);}}catch(err){{console.error('Hotspot zoom failed',err);}}
      }});

      function applyUrlState() {{
        try {{
          var params=new URLSearchParams(window.location.search);
          var precinct=params.get('precinct')||'';
          var neighborhood=params.get('neighborhood')||'';
          var crime=params.get('crime')||'';
          var view=params.get('view')||'';
          var section=params.get('section')||'';
          var prevStart=params.get('prevStart')||'', prevEnd=params.get('prevEnd')||'', currStart=params.get('currStart')||'', currEnd=params.get('currEnd')||'';
          if(prevStart&&prevEnd&&currStart&&currEnd&&prevStart<=prevEnd&&currStart<=currEnd) {{ var maxSpatial=spatialDailyData.length?spatialDailyData.reduce(function(m,r){{return r.date>m?r.date:m;}},''):''; if(!maxSpatial|| (prevEnd<=maxSpatial&&currEnd<=maxSpatial)) customSpatialRange={{prevStart:prevStart,prevEnd:prevEnd,currStart:currStart,currEnd:currEnd}}; else console.warn('Custom date range exceeds available dashboard data through '+maxSpatial); }}
          var psel=document.getElementById('cpPrecinctSelect');
          if(psel && precinct && Array.from(psel.options).some(function(o){{return o.value===precinct;}})) psel.value=precinct;
          var nsel=document.getElementById('cpNeighborhoodSelect');
          if(nsel && neighborhood && Array.from(nsel.options).some(function(o){{return o.value===neighborhood;}})) nsel.value=neighborhood;
          var csel=document.getElementById('cpCategorySelect');
          if(csel && crime) {{
            var candidates=['Crime Type | '+crime,'Category Focus | '+crime];
            var match=candidates.find(function(v){{return Array.from(csel.options).some(function(o){{return o.value===v;}});}});
            if(match) csel.value=match;
          }}
          var vsel=document.getElementById('cpPeriodSelect');
          if(vsel && ['Operational','Recent','YTD'].indexOf(view)>=0) vsel.value=view;
          window.applyTopSelectors();
          if(section) setTimeout(function(){{window.focusAnalysisSection(section);}},350);
        }} catch(err) {{
          window.applyTopSelectors();
        }}
      }}
      applyUrlState();
    }})();
    </script>
    """
    m.get_root().html.add_child(folium.Element(panel_html))

def assign_decision_purpose(offense_category: str) -> str:
    if not isinstance(offense_category, str):
        return "Preventive Patrol"

    value = offense_category.upper().strip()
    preventive = {
        "LARCENY",
        "BURGLARY",
        "STOLEN VEHICLE",
        "STOLEN PROPERTY",
        "ROBBERY",
        "DAMAGE TO PROPERTY",
    }
    investigations = {
        "FRAUD",
        "FORGERY",
        "EMBEZZLEMENT",
        "ARSON",
        "BRIBERY",
    }
    community_response = {
        "ASSAULT",
        "AGGRAVATED ASSAULT",
        "WEAPONS OFFENSES",
        "OBSTRUCTING THE POLICE",
        "HOMICIDE",
        "KIDNAPPING",
    }

    if value in investigations:
        return "Investigations"
    if value in community_response:
        return "Community Response"
    if value in preventive:
        return "Preventive Patrol"
    return "Preventive Patrol"


def detect_weekly_spikes(df: pd.DataFrame) -> pd.DataFrame:
    weekly = (
        df.groupby(["neighborhood", "week_start"], as_index=False)
        .size()
        .rename(columns={"size": "incident_count"})
    )

    weekly = weekly.sort_values(["neighborhood", "week_start"]).copy()
    grp = weekly.groupby("neighborhood")

    weekly["rolling_mean_4w"] = grp["incident_count"].transform(
        lambda s: s.shift(1).rolling(4, min_periods=2).mean()
    )
    weekly["rolling_std_4w"] = grp["incident_count"].transform(
        lambda s: s.shift(1).rolling(4, min_periods=2).std()
    )
    weekly["rolling_std_4w"] = weekly["rolling_std_4w"].fillna(0)

    weekly["z_score"] = (
        weekly["incident_count"] - weekly["rolling_mean_4w"]
    ) / weekly["rolling_std_4w"].replace(0, np.nan)

    weekly["z_score"] = weekly["z_score"].replace([np.inf, -np.inf], np.nan).fillna(0)

    count_threshold = max(5, int(weekly["incident_count"].quantile(0.8)))
    weekly["is_spike"] = (
        (weekly["incident_count"] >= count_threshold)
        & (weekly["incident_count"] > weekly["rolling_mean_4w"] + 1.5 * weekly["rolling_std_4w"])
    )

    return weekly


def build_h3_location_lookup(df: pd.DataFrame, resolution: int) -> pd.DataFrame:
    temp = df.copy()
    temp["h3_cell"] = temp.apply(
        lambda row: h3.latlng_to_cell(float(row["latitude"]), float(row["longitude"]), resolution),
        axis=1,
    )

    def mode_or_first(series: pd.Series):
        m = series.mode(dropna=True)
        if not m.empty:
            return m.iloc[0]
        return series.iloc[0] if len(series) else None

    lookup = (
        temp.groupby("h3_cell", as_index=False)
        .agg(
            latitude=("latitude", "median"),
            longitude=("longitude", "median"),
            neighborhood=("neighborhood", mode_or_first),
            nearest_intersection=("nearest_intersection", mode_or_first),
            police_precinct=("police_precinct", mode_or_first),
            zip_code=("zip_code", mode_or_first),
        )
    )

    # Fill sparse missing labels using nearest known records from the same dataset.
    known = temp.dropna(subset=["latitude", "longitude"]).copy()
    for col in ["neighborhood", "nearest_intersection", "police_precinct", "zip_code"]:
        missing_mask = lookup[col].isna()
        if not missing_mask.any():
            continue

        candidates = known.dropna(subset=[col])
        if candidates.empty:
            continue

        cand_lat = candidates["latitude"].to_numpy()
        cand_lon = candidates["longitude"].to_numpy()
        cand_val = candidates[col].to_numpy()

        for idx in lookup[missing_mask].index:
            lat0 = lookup.at[idx, "latitude"]
            lon0 = lookup.at[idx, "longitude"]
            d2 = (cand_lat - lat0) ** 2 + (cand_lon - lon0) ** 2
            nearest_i = int(np.argmin(d2))
            lookup.at[idx, col] = cand_val[nearest_i]

    # ZIP codes are numeric in source CSV; format as clean strings for display.
    def normalize_zip(value):
        if pd.isna(value):
            return "Unknown"
        text = str(value).strip()
        if text.endswith(".0"):
            text = text[:-2]
        if text.lower() == "nan" or text == "":
            return "Unknown"
        return text

    lookup["zip_code"] = lookup["zip_code"].apply(normalize_zip)
    for col in ["neighborhood", "nearest_intersection", "police_precinct"]:
        lookup[col] = (
            lookup[col]
            .astype("string")
            .fillna("Unknown")
            .replace({"<NA>": "Unknown", "nan": "Unknown"})
        )

    return lookup


def build_h3_incident_context_lookup(df: pd.DataFrame, resolution: int) -> pd.DataFrame:
    temp = df.copy()
    temp["h3_cell"] = temp.apply(
        lambda row: h3.latlng_to_cell(float(row["latitude"]), float(row["longitude"]), resolution),
        axis=1,
    )
    temp["shift_window"] = temp["incident_hour_of_day"].apply(assign_shift_window)

    offense_counts = (
        temp.groupby(["h3_cell", "offense_category"]).size().reset_index(name="offense_count")
    )
    idx = offense_counts.groupby("h3_cell")["offense_count"].idxmax()
    dominant = offense_counts.loc[idx].rename(
        columns={"offense_category": "dominant_offense", "offense_count": "dominant_offense_count"}
    )

    shift_counts = (
        temp.groupby(["h3_cell", "shift_window"]).size().reset_index(name="shift_count")
    )
    shift_totals = shift_counts.groupby("h3_cell", as_index=False)["shift_count"].sum().rename(
        columns={"shift_count": "total_shift_incidents"}
    )
    shift_top_idx = shift_counts.groupby("h3_cell")["shift_count"].idxmax()
    dominant_shift = shift_counts.loc[shift_top_idx].rename(
        columns={"shift_window": "dominant_shift", "shift_count": "dominant_shift_count"}
    )

    out = dominant.merge(shift_totals, on="h3_cell", how="left").merge(
        dominant_shift[["h3_cell", "dominant_shift", "dominant_shift_count"]],
        on="h3_cell",
        how="left",
    )
    out["dominant_offense_share"] = out["dominant_offense_count"] / out["total_shift_incidents"]
    out["dominant_shift_share"] = out["dominant_shift_count"] / out["total_shift_incidents"]
    return out


def build_h3_count_layer_data(df: pd.DataFrame, resolution: int) -> pd.DataFrame:
    temp = df.copy()
    temp["h3_cell"] = temp.apply(
        lambda row: h3.latlng_to_cell(float(row["latitude"]), float(row["longitude"]), resolution),
        axis=1,
    )
    counts = (
        temp.groupby("h3_cell", as_index=False)
        .size()
        .rename(columns={"size": "crime_count"})
    )
    location_lookup = build_h3_location_lookup(df, resolution)
    incident_lookup = build_h3_incident_context_lookup(df, resolution)
    return counts.merge(location_lookup, on="h3_cell", how="left").merge(
        incident_lookup, on="h3_cell", how="left"
    )


def add_h3_count_choropleth_layer(
    m: folium.Map,
    layer_data: pd.DataFrame,
    layer_name: str,
    legend_caption: str,
    colors: list[str],
    show: bool,
    add_legend: bool = True,
) -> None:
    vmin = float(layer_data["crime_count"].min())
    vmax = float(layer_data["crime_count"].max())
    colormap = cm.LinearColormap(colors=colors, vmin=vmin, vmax=vmax)
    colormap.caption = legend_caption
    if add_legend:
        colormap.add_to(m)

    features = []
    for _, row in layer_data.iterrows():
        cell = row["h3_cell"]
        count = int(row["crime_count"])
        boundary = h3.cell_to_boundary(cell)
        coordinates = [[lng, lat] for lat, lng in boundary]
        if coordinates and coordinates[0] != coordinates[-1]:
            coordinates.append(coordinates[0])

        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [coordinates]},
                "properties": {
                    "h3_cell": cell,
                    "crime_count": count,
                    "neighborhood": str(row.get("neighborhood", "Unknown")),
                    "nearest_intersection": str(row.get("nearest_intersection", "Unknown")),
                    "police_precinct": str(row.get("police_precinct", "Unknown")),
                    "zip_code": str(row.get("zip_code", "Unknown")),
                    "dominant_offense": str(row.get("dominant_offense", "Unknown")),
                    "dominant_offense_share": f"{float(row.get('dominant_offense_share', 0) or 0):.1%}",
                    "dominant_shift": str(row.get("dominant_shift", "Unknown")),
                    "dominant_shift_share": f"{float(row.get('dominant_shift_share', 0) or 0):.1%}",
                    "fill_color": colormap(count),
                },
            }
        )

    layer = folium.FeatureGroup(name=layer_name, show=show)
    folium.GeoJson(
        {"type": "FeatureCollection", "features": features},
        style_function=lambda feature: {
            "fillColor": feature["properties"]["fill_color"],
            "color": "#2b2b2b",
            "weight": 0.55,
            "opacity": 0.65,
            "fillOpacity": 0.42,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=[
                "neighborhood",
                "nearest_intersection",
                "police_precinct",
                "zip_code",
                "dominant_offense",
                "dominant_offense_share",
                "dominant_shift",
                "dominant_shift_share",
                "h3_cell",
                "crime_count",
            ],
            aliases=[
                "Neighborhood",
                "Nearest Intersection",
                "Precinct",
                "ZIP",
                "Dominant Incident Type",
                "Incident Type Share",
                "Dominant Shift",
                "Shift Share",
                "Grid ID",
                "Crime Count",
            ],
            localize=True,
        ),
    ).add_to(layer)
    layer.add_to(m)


def add_crime_type_h3_count_layers(
    m: folium.Map,
    df: pd.DataFrame,
    resolution: int,
    top_n_categories: int | None = None,
) -> None:
    offense_counts = df["offense_category"].value_counts()
    if top_n_categories is not None:
        offense_counts = offense_counts.head(top_n_categories)

    for category in offense_counts.index.tolist():
        subset = df[df["offense_category"] == category]
        if subset.empty:
            continue
        layer_data = build_h3_count_layer_data(subset, resolution)
        add_h3_count_choropleth_layer(
            m,
            layer_data,
            layer_name=f"Core Type | H3 Count | {category}",
            legend_caption=f"{category} incidents per hex cell",
            colors=["#eff6ff", "#93c5fd", "#2563eb", "#1e3a8a"],
            show=False,
            add_legend=False,
        )


def add_crime_type_and_shift_layers(
    m: folium.Map,
    df: pd.DataFrame,
    top_n_categories: int | None = None,
) -> None:
    offense_counts = df["offense_category"].value_counts()
    if top_n_categories is not None:
        offense_counts = offense_counts.head(top_n_categories)
    for category in offense_counts.index.tolist():
        subset = df[df["offense_category"] == category]
        if subset.empty:
            continue
        layer = folium.FeatureGroup(
            name=f"Crime Type | {category} ({len(subset):,})",
            show=False,
        )
        HeatMap(
            subset[["latitude", "longitude"]].values.tolist(),
            radius=10,
            blur=12,
            max_zoom=13,
        ).add_to(layer)
        layer.add_to(m)


def add_precinct_filter_layers(m: folium.Map, df: pd.DataFrame) -> None:
    precinct_counts = df["precinct_norm"].value_counts().sort_index()
    for precinct, _ in precinct_counts.items():
        subset = df[df["precinct_norm"] == precinct]
        if subset.empty:
            continue
        layer = folium.FeatureGroup(
            name=f"Precinct | {precinct} ({len(subset):,})",
            show=False,
        )
        HeatMap(
            subset[["latitude", "longitude"]].values.tolist(),
            radius=10,
            blur=12,
            max_zoom=13,
        ).add_to(layer)
        layer.add_to(m)


def add_focus_category_layers(m: folium.Map, df: pd.DataFrame) -> None:
    category_specs = [
        (f"Category Focus | {focus_name}", df[df[flag_col].fillna(False)])
        for focus_name, flag_col in CATEGORY_FOCUS_COLUMNS.items()
    ]

    for name, subset in category_specs:
        if subset.empty:
            continue
        layer = folium.FeatureGroup(name=f"{name} ({len(subset):,})", show=False)
        HeatMap(
            subset[["latitude", "longitude"]].values.tolist(),
            radius=10,
            blur=12,
            max_zoom=13,
        ).add_to(layer)
        layer.add_to(m)


def add_precinct_scope_heatmap_layers(m: folium.Map, df: pd.DataFrame) -> None:
    """Add true intersection heatmaps for Precinct × Crime/Category selections.

    This prevents the map from merely overlaying a precinct-wide heatmap with a
    citywide crime heatmap. Each combined filter now displays only incidents that
    satisfy both filters.
    """
    for precinct, p_df in df.groupby("precinct_norm"):
        p = str(precinct)
        for category, subset in p_df.groupby("offense_category"):
            if subset.empty:
                continue
            layer = folium.FeatureGroup(
                name=f"Scope | Precinct | {p} | Crime Type | {category} ({len(subset):,})",
                show=False,
            )
            HeatMap(
                subset[["latitude", "longitude"]].values.tolist(),
                radius=11, blur=13, max_zoom=15, min_opacity=0.22,
            ).add_to(layer)
            layer.add_to(m)

        focus_specs = [
            (focus_name, p_df[p_df[flag_col].fillna(False)])
            for focus_name, flag_col in CATEGORY_FOCUS_COLUMNS.items()
        ]
        for focus_name, subset in focus_specs:
            if subset.empty:
                continue
            layer = folium.FeatureGroup(
                name=f"Scope | Precinct | {p} | Category Focus | {focus_name} ({len(subset):,})",
                show=False,
            )
            HeatMap(
                subset[["latitude", "longitude"]].values.tolist(),
                radius=11, blur=13, max_zoom=15, min_opacity=0.22,
            ).add_to(layer)
            layer.add_to(m)


def build_precinct_monthly_trend(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby(["precinct_norm", "month_start"], as_index=False)
        .size()
        .rename(columns={"size": "incident_count"})
        .sort_values(["precinct_norm", "month_start"])
    )
    return out


def save_precinct_monthly_trend_heatmap(precinct_monthly: pd.DataFrame, out_path: Path) -> None:
    if precinct_monthly.empty:
        return

    temp = precinct_monthly.copy()
    temp["month_label"] = pd.to_datetime(temp["month_start"]).dt.strftime("%Y-%m")
    pivot = temp.pivot(index="precinct_norm", columns="month_label", values="incident_count").fillna(0)

    plt.figure(figsize=(13, 7))
    sns.heatmap(pivot, cmap="YlOrRd", linewidths=0.3, linecolor="#e5e7eb")
    plt.title("Monthly Incident Trend by Precinct")
    plt.xlabel("Month")
    plt.ylabel("Precinct")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def build_ytd_comparison(df: pd.DataFrame, current_year: int, previous_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    temp = df.copy()
    temp["incident_date"] = pd.to_datetime(temp["incident_date"])
    current_max = temp[temp["incident_year"] == current_year]["incident_date"].max()
    if pd.isna(current_max):
        return pd.DataFrame(), pd.DataFrame()

    cutoff_month = int(current_max.month)
    cutoff_day = int(current_max.day)

    curr = temp[
        (temp["incident_year"] == current_year)
        & ((temp["incident_date"].dt.month < cutoff_month) | ((temp["incident_date"].dt.month == cutoff_month) & (temp["incident_date"].dt.day <= cutoff_day)))
    ]
    prev = temp[
        (temp["incident_year"] == previous_year)
        & ((temp["incident_date"].dt.month < cutoff_month) | ((temp["incident_date"].dt.month == cutoff_month) & (temp["incident_date"].dt.day <= cutoff_day)))
    ]

    city = pd.DataFrame(
        {
            "year": [previous_year, current_year],
            "ytd_incidents": [len(prev), len(curr)],
        }
    )
    city["change_vs_previous"] = city["ytd_incidents"].diff()
    city["pct_change_vs_previous"] = city["ytd_incidents"].pct_change() * 100

    prev_p = prev.groupby("precinct_norm").size().rename("incidents_previous").reset_index()
    curr_p = curr.groupby("precinct_norm").size().rename("incidents_current").reset_index()
    precinct = prev_p.merge(curr_p, on="precinct_norm", how="outer").fillna(0)
    precinct["incidents_previous"] = precinct["incidents_previous"].astype(int)
    precinct["incidents_current"] = precinct["incidents_current"].astype(int)
    precinct["change"] = precinct["incidents_current"] - precinct["incidents_previous"]
    precinct["pct_change"] = np.where(
        precinct["incidents_previous"] > 0,
        100 * precinct["change"] / precinct["incidents_previous"],
        np.nan,
    )
    precinct = precinct.sort_values("incidents_current", ascending=False).reset_index(drop=True)

    return city, precinct


def build_precinct_improvement_table(
    df: pd.DataFrame,
    current_year: int,
    previous_year: int,
    baseline_year: int | None = None,
) -> pd.DataFrame:
    """Build a dashboard-friendly YTD precinct comparison for the current year."""
    temp = df.copy()
    temp["incident_date"] = pd.to_datetime(temp["incident_date"])
    current_max = temp.loc[temp["incident_year"] == current_year, "incident_date"].max()
    if pd.isna(current_max):
        return pd.DataFrame()

    cutoff_month = int(current_max.month)
    cutoff_day = int(current_max.day)
    comparison_years = [previous_year, current_year]
    if baseline_year is not None:
        comparison_years.append(int(baseline_year))

    ytd = temp[
        temp["incident_year"].isin(comparison_years)
        & (
            (temp["incident_date"].dt.month < cutoff_month)
            | (
                (temp["incident_date"].dt.month == cutoff_month)
                & (temp["incident_date"].dt.day <= cutoff_day)
            )
        )
    ].copy()

    counts = ytd.groupby(["precinct_norm", "incident_year"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=comparison_years, fill_value=0).reset_index()

    rename_map = {
        previous_year: "incidents_previous",
        current_year: "incidents_current",
    }
    if baseline_year is not None:
        rename_map[int(baseline_year)] = "incidents_baseline"
    counts = counts.rename(columns=rename_map)

    counts["change_vs_previous"] = counts["incidents_current"] - counts["incidents_previous"]
    counts["pct_change_vs_previous"] = np.where(
        counts["incidents_previous"] > 0,
        100 * counts["change_vs_previous"] / counts["incidents_previous"],
        np.nan,
    )

    counts["improvement_status"] = np.select(
        [counts["change_vs_previous"] < 0, counts["change_vs_previous"] > 0],
        ["Improved", "Worse"],
        default="No change",
    )
    counts["improvement_score"] = -counts["pct_change_vs_previous"].fillna(0)

    if baseline_year is not None:
        counts["change_vs_baseline"] = counts["incidents_current"] - counts["incidents_baseline"]
        counts["pct_change_vs_baseline"] = np.where(
            counts["incidents_baseline"] > 0,
            100 * counts["change_vs_baseline"] / counts["incidents_baseline"],
            np.nan,
        )
        counts["improvement_status"] = np.select(
            [
                (counts["change_vs_previous"] < 0) & (counts["change_vs_baseline"] < 0),
                (counts["change_vs_previous"] < 0) | (counts["change_vs_baseline"] < 0),
                (counts["change_vs_previous"] > 0) | (counts["change_vs_baseline"] > 0),
            ],
            ["Improved vs both years", "Improved vs recent baseline", "Worse vs recent baseline"],
            default="No change",
        )
        counts["improvement_score"] = -(
            0.7 * counts["pct_change_vs_previous"].fillna(0)
            + 0.3 * counts["pct_change_vs_baseline"].fillna(0)
        )

    counts["comparison_date"] = current_max.strftime("%Y-%m-%d")
    counts["improvement_score"] = counts["improvement_score"].round(2)

    return counts.sort_values(["improvement_score", "incidents_current"], ascending=[False, False]).reset_index(drop=True)

def build_precinct_crime_14d_comparison(
    df: pd.DataFrame,
    current_year: int,
) -> pd.DataFrame:
    """
    Compare the latest 14 days with the immediately preceding 14 days
    for every precinct and every offense category.

    Also calculates the same 14-day change citywide for each crime type.
    """

    temp = df.copy()
    temp["incident_date"] = pd.to_datetime(temp["incident_date"])

    # Latest incident date available in the current-year dataset.
    current_max = temp.loc[
        temp["incident_year"] == current_year,
        "incident_date",
    ].max()

    if pd.isna(current_max):
        return pd.DataFrame()

    # Current 14-day window: cutoff date plus previous 13 days.
    current_start = current_max - pd.Timedelta(days=RECENT_WINDOW_DAYS - 1)

    # Previous 14-day window immediately before the current one.
    previous_end = current_start - pd.Timedelta(days=1)
    previous_start = previous_end - pd.Timedelta(days=RECENT_WINDOW_DAYS - 1)

    current_period = temp[
        (temp["incident_date"] >= current_start)
        & (temp["incident_date"] <= current_max)
    ].copy()

    previous_period = temp[
        (temp["incident_date"] >= previous_start)
        & (temp["incident_date"] <= previous_end)
    ].copy()

    # -----------------------------
    # Precinct × Crime Type counts
    # -----------------------------
    current_counts = (
        current_period.groupby(
            ["precinct_norm", "offense_category"]
        )
        .size()
        .rename("current_14d")
        .reset_index()
    )

    previous_counts = (
        previous_period.groupby(
            ["precinct_norm", "offense_category"]
        )
        .size()
        .rename("previous_14d")
        .reset_index()
    )

    comparison = previous_counts.merge(
        current_counts,
        on=["precinct_norm", "offense_category"],
        how="outer",
    ).fillna(0)

    comparison["previous_14d"] = comparison["previous_14d"].astype(int)
    comparison["current_14d"] = comparison["current_14d"].astype(int)

    comparison["change_14d"] = (
        comparison["current_14d"] - comparison["previous_14d"]
    )

    comparison["pct_change_14d"] = np.where(
        comparison["previous_14d"] > 0,
        100
        * comparison["change_14d"]
        / comparison["previous_14d"],
        np.nan,
    )

    comparison["recent_movement"] = np.select(
        [
            comparison["change_14d"] > 0,
            comparison["change_14d"] < 0,
        ],
        [
            "Increasing",
            "Decreasing",
        ],
        default="No Change",
    )

    city_current = (
        current_period.groupby("offense_category")
        .size()
        .rename("city_current_14d")
        .reset_index()
    )

    city_previous = (
        previous_period.groupby("offense_category")
        .size()
        .rename("city_previous_14d")
        .reset_index()
    )

    city = city_previous.merge(
        city_current,
        on="offense_category",
        how="outer",
    ).fillna(0)

    city["city_previous_14d"] = city["city_previous_14d"].astype(int)
    city["city_current_14d"] = city["city_current_14d"].astype(int)

    city["city_change_14d"] = (
        city["city_current_14d"] - city["city_previous_14d"]
    )

    city["city_pct_change_14d"] = np.where(
        city["city_previous_14d"] > 0,
        100
        * city["city_change_14d"]
        / city["city_previous_14d"],
        np.nan,
    )

    comparison = comparison.merge(
        city,
        on="offense_category",
        how="left",
    )

    # Store the exact periods for dashboard/report display.
    comparison["previous_14d_start"] = previous_start.strftime("%Y-%m-%d")
    comparison["previous_14d_end"] = previous_end.strftime("%Y-%m-%d")
    comparison["current_14d_start"] = current_start.strftime("%Y-%m-%d")
    comparison["current_14d_end"] = current_max.strftime("%Y-%m-%d")

    comparison["pct_change_14d"] = comparison["pct_change_14d"].round(2)
    comparison["city_pct_change_14d"] = comparison[
        "city_pct_change_14d"
    ].round(2)

    return comparison.sort_values(
        ["precinct_norm", "current_14d"],
        ascending=[True, False],
    ).reset_index(drop=True)


def build_temporal_pattern_profiles(
    df: pd.DataFrame,
    current_year: int,
) -> dict[str, pd.DataFrame]:
    """Build YTD and recent-14-day timing profiles for dashboard selections.

    Profiles are available citywide and by precinct for: all incidents, every
    offense category, and the three operational Category Focus groups.
    """
    temp = df.copy()
    temp["incident_date"] = pd.to_datetime(temp["incident_date"])
    temp["incident_hour_of_day"] = pd.to_numeric(temp["incident_hour_of_day"], errors="coerce")
    temp = temp[(temp["incident_year"] == current_year) & temp["incident_hour_of_day"].notna()].copy()
    if temp.empty:
        return {"summary": pd.DataFrame(), "matrix": pd.DataFrame()}

    current_max = temp["incident_date"].max()
    ytd_start = pd.Timestamp(year=int(current_year), month=1, day=1)
    recent_start = current_max - pd.Timedelta(days=RECENT_WINDOW_DAYS - 1)

    temp["incident_hour_of_day"] = temp["incident_hour_of_day"].astype(int)
    temp["weekday"] = temp["incident_date"].dt.day_name()
    temp["shift_window"] = temp["incident_hour_of_day"].apply(assign_shift_window)
    temp["time_block"] = pd.cut(
        temp["incident_hour_of_day"],
        bins=[-1, 5, 11, 17, 23],
        labels=["00:00-05:59", "06:00-11:59", "12:00-17:59", "18:00-23:59"],
    ).astype(str)

    periods = [
        ("YTD", temp[(temp["incident_date"] >= ytd_start) & (temp["incident_date"] <= current_max)].copy(), ytd_start, current_max),
        ("Recent 14D", temp[(temp["incident_date"] >= recent_start) & (temp["incident_date"] <= current_max)].copy(), recent_start, current_max),
    ]

    focus_specs = [
        ("Category Focus", focus_name, flag_col)
        for focus_name, flag_col in CATEGORY_FOCUS_COLUMNS.items()
    ]
    summary_rows = []
    matrix_rows = []

    def norm_precinct(value):
        if pd.isna(value):
            return "Unknown"
        text = str(value).strip().upper()
        return text.zfill(2) if text.isdigit() else text

    def add_profile(period_name, subset, precinct_key, selection_type, selection_name, start, end):
        if subset.empty:
            return
        day_counts = subset["weekday"].value_counts()
        hour_counts = subset["incident_hour_of_day"].value_counts()
        shift_counts = subset["shift_window"].value_counts()
        block_counts = subset["time_block"].value_counts()
        summary_rows.append({
            "period": period_name,
            "precinct_norm": precinct_key,
            "selection_type": selection_type,
            "selection_name": selection_name,
            "total_incidents": int(len(subset)),
            "peak_day": str(day_counts.index[0]),
            "peak_day_count": int(day_counts.iloc[0]),
            "peak_hour": int(hour_counts.index[0]),
            "peak_hour_count": int(hour_counts.iloc[0]),
            "peak_shift": str(shift_counts.index[0]),
            "peak_shift_count": int(shift_counts.iloc[0]),
            "peak_time_block": str(block_counts.index[0]),
            "peak_time_block_count": int(block_counts.iloc[0]),
            "period_start": start.strftime("%Y-%m-%d"),
            "period_end": end.strftime("%Y-%m-%d"),
        })
        grid = subset.groupby(["weekday", "time_block"], observed=True).size().reset_index(name="incident_count")
        for _, r in grid.iterrows():
            matrix_rows.append({
                "period": period_name,
                "precinct_norm": precinct_key,
                "selection_type": selection_type,
                "selection_name": selection_name,
                "weekday": str(r["weekday"]),
                "time_block": str(r["time_block"]),
                "incident_count": int(r["incident_count"]),
            })

    for period_name, period_df, start, end in periods:
        precinct_keys = ["ALL"] + sorted(period_df["precinct_norm"].dropna().astype(str).map(norm_precinct).unique().tolist())
        period_df = period_df.copy()
        period_df["precinct_key"] = period_df["precinct_norm"].apply(norm_precinct)
        for precinct_key in precinct_keys:
            base = period_df if precinct_key == "ALL" else period_df[period_df["precinct_key"] == precinct_key]
            add_profile(period_name, base, precinct_key, "All", "All", start, end)
            for offense, crime_df in base.groupby("offense_category"):
                add_profile(period_name, crime_df, precinct_key, "Crime Type", str(offense), start, end)
            for selection_type, selection_name, flag_col in focus_specs:
                if flag_col in base.columns:
                    focus_df = base[base[flag_col].fillna(False)]
                    add_profile(period_name, focus_df, precinct_key, selection_type, selection_name, start, end)

    return {
        "summary": pd.DataFrame(summary_rows),
        "matrix": pd.DataFrame(matrix_rows),
    }


def build_priority_emerging_concerns(
    precinct_crime_trends: pd.DataFrame,
    precinct_crime_14d: pd.DataFrame,
) -> pd.DataFrame:
    """Rank precinct × crime-type signals for operational attention.

    The score is intentionally volume-aware so a tiny baseline cannot dominate
    simply because its percentage change is very large. It combines:
    - current 14-day volume,
    - absolute 14-day increase,
    - 14-day percentage change,
    - divergence from the citywide 14-day trend, and
    - matched-YTD direction versus the previous year.
    """
    if precinct_crime_14d is None or precinct_crime_14d.empty:
        return pd.DataFrame()

    recent = precinct_crime_14d.copy()
    trend = precinct_crime_trends.copy() if precinct_crime_trends is not None else pd.DataFrame()

    def norm_precinct(value):
        if pd.isna(value):
            return "Unknown"
        text = str(value).strip().upper()
        return text.zfill(2) if text.isdigit() else text

    recent["precinct_norm"] = recent["precinct_norm"].apply(norm_precinct)
    if not trend.empty:
        trend["precinct_norm"] = trend["precinct_norm"].apply(norm_precinct)
        keep = ["precinct_norm", "offense_category", "pct_change_vs_previous", "trend_class"]
        recent = recent.merge(trend[keep], on=["precinct_norm", "offense_category"], how="left")
    else:
        recent["pct_change_vs_previous"] = np.nan
        recent["trend_class"] = "Unknown"

    recent["change_14d"] = recent["current_14d"] - recent["previous_14d"]
    recent["city_gap_14d"] = recent["pct_change_14d"] - recent["city_pct_change_14d"]

    max_volume = max(float(recent["current_14d"].max()), 1.0)
    max_abs_increase = max(float(recent["change_14d"].clip(lower=0).max()), 1.0)

    volume_component = 25 * np.log1p(recent["current_14d"].clip(lower=0)) / np.log1p(max_volume)
    absolute_component = 30 * recent["change_14d"].clip(lower=0) / max_abs_increase

    # When the prior-period baseline is zero, pct_change_14d is undefined.
    # Use a capped volume-based proxy so new clusters can still surface without
    # assigning an infinite percentage increase.
    pct_signal = recent["pct_change_14d"].copy()
    zero_baseline_proxy = (recent["current_14d"].clip(lower=0) * 10).clip(upper=100)
    pct_signal = pct_signal.where(pct_signal.notna(), zero_baseline_proxy)
    percent_component = 20 * pct_signal.clip(lower=0, upper=100) / 100

    city_component = 15 * recent["city_gap_14d"].fillna(0).clip(lower=0, upper=50) / 50
    ytd_component = 10 * recent["pct_change_vs_previous"].fillna(0).clip(lower=0, upper=50) / 50

    recent["priority_score"] = (
        volume_component + absolute_component + percent_component + city_component + ytd_component
    ).round(1)

    recent_pct = recent["pct_change_14d"]
    abs_change = recent["change_14d"]
    volume = recent["current_14d"]
    city_gap = recent["city_gap_14d"].fillna(0)
    ytd_pct = recent["pct_change_vs_previous"].fillna(0)

    high = (
        (volume >= 20)
        & (abs_change >= 10)
        & ((recent_pct >= 25) | (recent["previous_14d"] == 0))
        & ((city_gap >= 10) | (ytd_pct > 2))
    )
    emerging = (
        (volume >= 10)
        & (abs_change >= 5)
        & ((recent_pct >= 10) | (recent["previous_14d"] == 0))
        & ((city_gap >= 5) | (ytd_pct > 2))
    )
    watch = (volume >= 5) & (abs_change > 0) & ((recent_pct > 2) | (recent["previous_14d"] == 0))
    improving = (volume >= 5) & (abs_change <= -5) & (recent_pct <= -10)

    recent["priority_signal"] = np.select(
        [high, emerging, watch, improving],
        ["High Priority", "Emerging Concern", "Watch", "Recent Improvement"],
        default="Monitor",
    )

    signal_order = {"High Priority": 0, "Emerging Concern": 1, "Watch": 2, "Recent Improvement": 3, "Monitor": 4}
    recent["signal_order"] = recent["priority_signal"].map(signal_order).fillna(9)
    return recent.sort_values(
        ["signal_order", "priority_score", "current_14d"],
        ascending=[True, False, False],
    ).drop(columns=["signal_order"]).reset_index(drop=True)


def build_precinct_crime_trend_table(
    df: pd.DataFrame,
    current_year: int,
    previous_year: int,
    baseline_year: int | None = None,
    stable_pct: float = 2.0,
) -> pd.DataFrame:
    """Matched-YTD trend for every Precinct x offense_category combination."""
    temp = df.copy()
    temp["incident_date"] = pd.to_datetime(temp["incident_date"])
    temp["offense_category"] = temp["offense_category"].astype("string").fillna("Unknown").astype(str)

    current_max = temp.loc[temp["incident_year"] == current_year, "incident_date"].max()
    if pd.isna(current_max):
        return pd.DataFrame()

    comparison_years = [previous_year, current_year]
    if baseline_year is not None:
        comparison_years.append(int(baseline_year))

    cutoff_month, cutoff_day = int(current_max.month), int(current_max.day)
    ytd = temp[
        temp["incident_year"].isin(comparison_years)
        & (
            (temp["incident_date"].dt.month < cutoff_month)
            | ((temp["incident_date"].dt.month == cutoff_month) & (temp["incident_date"].dt.day <= cutoff_day))
        )
    ].copy()

    counts = (
        ytd.groupby(["precinct_norm", "offense_category", "incident_year"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=comparison_years, fill_value=0)
        .reset_index()
    )
    rename_map = {previous_year: "incidents_previous", current_year: "incidents_current"}
    if baseline_year is not None:
        rename_map[int(baseline_year)] = "incidents_baseline"
    counts = counts.rename(columns=rename_map)
    if "incidents_baseline" not in counts.columns:
        counts["incidents_baseline"] = np.nan

    counts["change_vs_previous"] = counts["incidents_current"] - counts["incidents_previous"]
    counts["pct_change_vs_previous"] = np.where(
        counts["incidents_previous"] > 0,
        100 * counts["change_vs_previous"] / counts["incidents_previous"],
        np.nan,
    )
    counts["change_vs_baseline"] = counts["incidents_current"] - counts["incidents_baseline"]
    counts["pct_change_vs_baseline"] = np.where(
        counts["incidents_baseline"] > 0,
        100 * counts["change_vs_baseline"] / counts["incidents_baseline"],
        np.nan,
    )

    def classify(row) -> str:
        p = row["pct_change_vs_previous"]
        if pd.isna(p):
            return "Stable" if row["incidents_current"] == 0 else "Worsening"
        if abs(p) <= stable_pct:
            return "Stable"
        if p < -stable_pct:
            if baseline_year is not None:
                base = row["incidents_baseline"]
                prev = row["incidents_previous"]
                curr = row["incidents_current"]
                if pd.notna(base) and base > prev > curr:
                    return "Consistently Improving"
            return "Improving"
        if baseline_year is not None:
            base = row["incidents_baseline"]
            prev = row["incidents_previous"]
            curr = row["incidents_current"]
            if pd.notna(base) and base < prev < curr:
                return "Consistently Worsening"
        return "Worsening"

    counts["trend_class"] = counts.apply(classify, axis=1)
    counts["comparison_date"] = current_max.strftime("%Y-%m-%d")
    return counts.sort_values(["precinct_norm", "offense_category"]).reset_index(drop=True)

def save_ytd_precinct_comparison_chart(precinct_ytd: pd.DataFrame, out_path: Path) -> None:
    if precinct_ytd.empty:
        return

    top = precinct_ytd.head(12).copy()
    x = np.arange(len(top))
    width = 0.38

    plt.figure(figsize=(12, 6))
    plt.bar(x - width / 2, top["incidents_previous"], width=width, label="Previous Year YTD", color="#94a3b8")
    plt.bar(x + width / 2, top["incidents_current"], width=width, label="Current Year YTD", color="#2563eb")
    plt.xticks(x, top["precinct_norm"].astype(str))
    plt.xlabel("Precinct")
    plt.ylabel("Incidents")
    plt.title("YTD Incident Comparison by Precinct")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def build_target_area_ytd_comparison(
    df: pd.DataFrame,
    focus_locations: pd.DataFrame,
    current_year: int,
    previous_year: int,
    top_n_areas: int = 12,
) -> pd.DataFrame:
    target_areas = focus_locations["neighborhood"].dropna().head(top_n_areas).unique().tolist()
    if not target_areas:
        return pd.DataFrame()

    temp = df[df["neighborhood"].isin(target_areas)].copy()
    temp["incident_date"] = pd.to_datetime(temp["incident_date"])
    current_max = temp[temp["incident_year"] == current_year]["incident_date"].max()
    if pd.isna(current_max):
        return pd.DataFrame()

    cutoff_month = int(current_max.month)
    cutoff_day = int(current_max.day)
    curr = temp[
        (temp["incident_year"] == current_year)
        & ((temp["incident_date"].dt.month < cutoff_month) | ((temp["incident_date"].dt.month == cutoff_month) & (temp["incident_date"].dt.day <= cutoff_day)))
    ]
    prev = temp[
        (temp["incident_year"] == previous_year)
        & ((temp["incident_date"].dt.month < cutoff_month) | ((temp["incident_date"].dt.month == cutoff_month) & (temp["incident_date"].dt.day <= cutoff_day)))
    ]

    prev_n = prev.groupby("neighborhood").size().rename("incidents_previous").reset_index()
    curr_n = curr.groupby("neighborhood").size().rename("incidents_current").reset_index()
    out = prev_n.merge(curr_n, on="neighborhood", how="outer").fillna(0)
    out["incidents_previous"] = out["incidents_previous"].astype(int)
    out["incidents_current"] = out["incidents_current"].astype(int)
    out["change"] = out["incidents_current"] - out["incidents_previous"]
    out["pct_change"] = np.where(
        out["incidents_previous"] > 0,
        100 * out["change"] / out["incidents_previous"],
        np.nan,
    )
    return out.sort_values("incidents_current", ascending=False).reset_index(drop=True)


def build_violent_14d_summaries(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    violent_categories = {"ASSAULT", "AGGRAVATED ASSAULT", "ROBBERY", "HOMICIDE", "WEAPONS OFFENSES"}
    max_date = pd.to_datetime(df["incident_date"]).max()
    start_date = max_date - pd.Timedelta(days=RECENT_WINDOW_DAYS - 1)

    temp = df.copy()
    temp["incident_date"] = pd.to_datetime(temp["incident_date"])
    temp = temp[
        (temp["incident_date"] >= start_date)
        & (temp["incident_date"] <= max_date)
        & (temp["offense_category"].astype(str).str.upper().isin(violent_categories))
    ].copy()

    by_type = (
        temp.groupby("offense_category", as_index=False)
        .size()
        .rename(columns={"size": "incident_count"})
        .sort_values("incident_count", ascending=False)
    )

    temp["day_name"] = pd.to_datetime(temp["incident_date"]).dt.day_name()
    temp["hour"] = pd.to_numeric(temp["incident_hour_of_day"], errors="coerce").fillna(0).astype(int)
    by_day_hour = (
        temp.groupby(["day_name", "hour"], as_index=False)
        .size()
        .rename(columns={"size": "incident_count"})
    )

    ordered_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    by_day_hour["day_name"] = pd.Categorical(by_day_hour["day_name"], categories=ordered_days, ordered=True)
    by_day_hour = by_day_hour.sort_values(["day_name", "hour"])
    return by_type, by_day_hour


def save_violent_14d_charts(by_type: pd.DataFrame, by_day_hour: pd.DataFrame, by_type_out: Path, by_day_hour_out: Path) -> None:
    if not by_type.empty:
        plt.figure(figsize=(10, 5))
        sns.barplot(data=by_type, y="offense_category", x="incident_count", color="#b91c1c")
        plt.title("Violent Crime (Past 14 Days) by Type")
        plt.xlabel("Incidents")
        plt.ylabel("Offense Category")
        plt.tight_layout()
        plt.savefig(by_type_out, dpi=220)
        plt.close()

    if not by_day_hour.empty:
        pivot = by_day_hour.pivot(index="day_name", columns="hour", values="incident_count").fillna(0)
        plt.figure(figsize=(13, 5))
        sns.heatmap(pivot, cmap="Reds", linewidths=0.3, linecolor="#e5e7eb")
        plt.title("Violent Crime (Past 14 Days) by Day and Hour")
        plt.xlabel("Hour of Day")
        plt.ylabel("Day of Week")
        plt.tight_layout()
        plt.savefig(by_day_hour_out, dpi=220)
        plt.close()


def build_spatial_daily_payload(df: pd.DataFrame, current_year: int, resolution: int = 8) -> pd.DataFrame:
    """Compact current-year daily H3 records for browser-side custom spatial comparisons."""
    if h3 is None:
        return pd.DataFrame()
    cols = ["incident_date", "incident_year", "latitude", "longitude", "precinct_norm", "neighborhood",
            "offense_category", "nearest_intersection", "is_violent_crime", "is_property_crime", "is_vehicle_related", "incident_hour_of_day"]
    temp = df[[c for c in cols if c in df.columns]].copy()
    temp = temp[temp["incident_year"].astype(int).eq(int(current_year))].dropna(subset=["incident_date", "latitude", "longitude"])
    if temp.empty:
        return pd.DataFrame()
    temp["date"] = pd.to_datetime(temp["incident_date"]).dt.strftime("%Y-%m-%d")
    temp["weekday"] = pd.to_datetime(temp["incident_date"]).dt.day_name()
    temp["hour"] = pd.to_numeric(temp.get("incident_hour_of_day"), errors="coerce").fillna(-1).astype(int)
    temp["h3"] = [h3.latlng_to_cell(float(a), float(b), resolution) for a,b in zip(temp["latitude"], temp["longitude"])]
    temp["precinct"] = temp["precinct_norm"].astype(str)
    temp["crime"] = temp["offense_category"].fillna("Unknown").astype(str)
    temp["neighborhood"] = temp["neighborhood"].fillna("Unknown").astype(str)
    temp["intersection"] = temp.get("nearest_intersection", pd.Series("Unknown", index=temp.index)).fillna("Unknown").astype(str)
    temp["violent"] = temp.get("is_violent_crime", False).fillna(False).astype(bool)
    temp["property"] = temp.get("is_property_crime", False).fillna(False).astype(bool)
    temp["vehicle"] = temp.get("is_vehicle_related", False).fillna(False).astype(bool)
    keys=["date","weekday","hour","h3","precinct","neighborhood","crime","violent","property","vehicle"]
    def first_known(x):
        y=x[x.ne("Unknown")]
        return y.iloc[0] if not y.empty else "Unknown"
    return temp.groupby(keys, as_index=False).agg(count=("crime","size"),lat=("latitude","median"),lon=("longitude","median"),intersection=("intersection",first_known))

def build_hotspot_persistence_change(
    df: pd.DataFrame,
    current_year: int,
    resolution: int = 8,
    hotspot_quantile: float = 0.80,
    min_hotspot_count: int = 3,
) -> pd.DataFrame:
    """Classify H3 locations as persistent, emerging, new, or declining hotspots.

    The comparison uses the latest 14 days in the current-year data versus the
    immediately preceding 14 days. Hotspot thresholds are calculated separately
    for each precinct/selection scope using the 80th percentile of occupied H3
    cells, with a small minimum-count guardrail so one-off incidents are not
    automatically treated as hotspots.
    """
    if h3 is None:
        return pd.DataFrame()

    temp = df.copy()
    temp["incident_date"] = pd.to_datetime(temp["incident_date"])
    current_year = int(current_year)
    current_max = temp.loc[temp["incident_year"] == current_year, "incident_date"].max()
    if pd.isna(current_max):
        return pd.DataFrame()

    current_start = current_max - pd.Timedelta(days=RECENT_WINDOW_DAYS - 1)
    previous_end = current_start - pd.Timedelta(days=1)
    previous_start = previous_end - pd.Timedelta(days=RECENT_WINDOW_DAYS - 1)

    recent = temp[
        (temp["incident_year"] == current_year)
        & (temp["incident_date"] >= previous_start)
        & (temp["incident_date"] <= current_max)
    ].copy()
    if recent.empty:
        return pd.DataFrame()

    recent["h3_cell"] = recent.apply(
        lambda row: h3.latlng_to_cell(float(row["latitude"]), float(row["longitude"]), resolution),
        axis=1,
    )
    recent["hotspot_period"] = np.where(
        recent["incident_date"] >= current_start, "Current 14D", "Previous 14D"
    )

    def mode_or_first(series: pd.Series):
        vals = series.dropna()
        if vals.empty:
            return "Unknown"
        modes = vals.mode()
        return modes.iloc[0] if not modes.empty else vals.iloc[0]

    location_lookup = (
        recent.groupby("h3_cell", as_index=False)
        .agg(
            latitude=("latitude", "median"),
            longitude=("longitude", "median"),
            neighborhood=("neighborhood", mode_or_first),
            nearest_intersection=("nearest_intersection", mode_or_first),
        )
    )

    selection_specs = [
        ("All", "All", pd.Series(True, index=recent.index)),
    ]
    selection_specs.extend(
        ("Category Focus", focus_name, recent[flag_col].fillna(False))
        for focus_name, flag_col in CATEGORY_FOCUS_COLUMNS.items()
    )
    for crime_name in sorted(recent["offense_category"].dropna().astype(str).unique().tolist()):
        selection_specs.append(("Crime Type", crime_name, recent["offense_category"].astype(str).eq(crime_name)))

    outputs = []

    def threshold_from(values: pd.Series) -> int:
        positive = pd.to_numeric(values, errors="coerce").fillna(0)
        positive = positive[positive > 0]
        if positive.empty:
            return int(min_hotspot_count)
        return max(int(min_hotspot_count), int(np.ceil(positive.quantile(hotspot_quantile))))

    def classify_scope(scope_df: pd.DataFrame, precinct_key: str, selection_type: str, selection_name: str):
        if scope_df.empty:
            return
        counts = (
            scope_df.groupby(["h3_cell", "hotspot_period"])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=["Previous 14D", "Current 14D"], fill_value=0)
            .reset_index()
            .rename(columns={"Previous 14D": "previous_14d", "Current 14D": "current_14d"})
        )
        if counts.empty:
            return
        prev_threshold = threshold_from(counts["previous_14d"])
        curr_threshold = threshold_from(counts["current_14d"])
        prev_hot = counts["previous_14d"] >= prev_threshold
        curr_hot = counts["current_14d"] >= curr_threshold

        counts["hotspot_status"] = np.select(
            [
                prev_hot & curr_hot,
                (~prev_hot) & curr_hot & counts["previous_14d"].eq(0),
                (~prev_hot) & curr_hot,
                prev_hot & (~curr_hot),
            ],
            ["Persistent Hotspot", "New Hotspot", "Emerging Hotspot", "Declining Hotspot"],
            default="Not Material",
        )
        counts = counts[counts["hotspot_status"] != "Not Material"].copy()
        if counts.empty:
            return

        counts["change_14d"] = counts["current_14d"] - counts["previous_14d"]
        counts["pct_change_14d"] = np.where(
            counts["previous_14d"] > 0,
            100 * counts["change_14d"] / counts["previous_14d"],
            np.nan,
        )
        counts["hotspot_score"] = (
            counts["current_14d"]
            + 1.5 * counts["change_14d"].clip(lower=0)
            + np.where(counts["hotspot_status"].eq("Persistent Hotspot"), counts["current_14d"] * 0.35, 0)
        ).round(2)
        counts["precinct_norm"] = precinct_key
        counts["selection_type"] = selection_type
        counts["selection_name"] = selection_name
        counts["previous_hotspot_threshold"] = prev_threshold
        counts["current_hotspot_threshold"] = curr_threshold
        counts["previous_14d_start"] = previous_start.strftime("%Y-%m-%d")
        counts["previous_14d_end"] = previous_end.strftime("%Y-%m-%d")
        counts["current_14d_start"] = current_start.strftime("%Y-%m-%d")
        counts["current_14d_end"] = current_max.strftime("%Y-%m-%d")
        outputs.append(counts)

    precinct_values = sorted(recent["precinct_norm"].dropna().astype(str).unique().tolist())
    for selection_type, selection_name, selection_mask in selection_specs:
        selected = recent.loc[selection_mask].copy()
        if selected.empty:
            continue
        classify_scope(selected, "ALL", selection_type, selection_name)
        for precinct in precinct_values:
            classify_scope(
                selected[selected["precinct_norm"].astype(str).eq(precinct)],
                precinct,
                selection_type,
                selection_name,
            )

    if not outputs:
        return pd.DataFrame()

    out = pd.concat(outputs, ignore_index=True)
    out = out.merge(location_lookup, on="h3_cell", how="left")
    out["pct_change_14d"] = out["pct_change_14d"].round(2)
    status_rank = {
        "New Hotspot": 0,
        "Emerging Hotspot": 1,
        "Persistent Hotspot": 2,
        "Declining Hotspot": 3,
    }
    out["status_rank"] = out["hotspot_status"].map(status_rank).fillna(9)
    out = out.sort_values(
        ["selection_type", "selection_name", "precinct_norm", "status_rank", "hotspot_score"],
        ascending=[True, True, True, True, False],
    ).drop(columns=["status_rank"])
    return out.reset_index(drop=True)


def build_category_hotspots(df: pd.DataFrame, resolution: int = 8) -> pd.DataFrame:
    pieces = []
    category_map = {
        focus_name: df[df[flag_col].fillna(False)]
        for focus_name, flag_col in CATEGORY_FOCUS_COLUMNS.items()
    }

    for name, subset in category_map.items():
        if subset.empty:
            continue
        temp = subset.copy()
        temp["h3_cell"] = temp.apply(
            lambda row: h3.latlng_to_cell(float(row["latitude"]), float(row["longitude"]), resolution),
            axis=1,
        )
        agg = (
            temp.groupby("h3_cell", as_index=False)
            .agg(
                incident_count=("crime_id", "count"),
                neighborhood=("neighborhood", lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]),
                nearest_intersection=("nearest_intersection", lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]),
                precinct_norm=("precinct_norm", lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]),
            )
            .sort_values("incident_count", ascending=False)
        )
        agg.insert(0, "focus_category", name)
        pieces.append(agg)

    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, ignore_index=True)

    temp = df.copy()
    temp["shift_window"] = temp["incident_hour_of_day"].apply(assign_shift_window)
    for shift_name in [
        "Day Shift (06:00-13:59)",
        "Evening Shift (14:00-21:59)",
        "Night Shift (22:00-05:59)",
    ]:
        subset = temp[temp["shift_window"] == shift_name]
        if subset.empty:
            continue
        layer = folium.FeatureGroup(
            name=f"Shift View | {shift_name} ({len(subset):,})",
            show=False,
        )
        HeatMap(
            subset[["latitude", "longitude"]].values.tolist(),
            radius=10,
            blur=12,
            max_zoom=13,
        ).add_to(layer)
        layer.add_to(m)


def build_spike_points(df: pd.DataFrame, weekly: pd.DataFrame) -> pd.DataFrame:
    spikes = weekly[weekly["is_spike"]].copy()
    if spikes.empty:
        return spikes

    merged = df.merge(
        spikes[["neighborhood", "week_start", "incident_count", "z_score"]],
        on=["neighborhood", "week_start"],
        how="inner",
    )

    spike_points = (
        merged.groupby(["neighborhood", "week_start", "incident_count", "z_score"], as_index=False)
        .agg(latitude=("latitude", "median"), longitude=("longitude", "median"))
        .sort_values(["incident_count", "z_score"], ascending=False)
    )
    return spike_points


def save_hexbin_map(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(10, 8))
    hb = plt.hexbin(
        df["longitude"],
        df["latitude"],
        gridsize=70,
        cmap="inferno",
        mincnt=1,
        bins="log",
    )
    plt.colorbar(hb, label="Crime density (log scale)")
    plt.title("Detroit Crime Hotspots (Hexbin Density)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def save_cluster_map(df: pd.DataFrame, out_path: Path) -> None:
    coords = df[["latitude", "longitude"]].to_numpy()

    scaler = StandardScaler()
    scaled = scaler.fit_transform(coords)

    model = DBSCAN(eps=0.08, min_samples=20)
    labels = model.fit_predict(scaled)

    temp = df.copy()
    temp["cluster"] = labels
    clustered = temp[temp["cluster"] != -1]

    plt.figure(figsize=(10, 8))
    if clustered.empty:
        plt.scatter(temp["longitude"], temp["latitude"], s=2, alpha=0.3, color="gray")
        plt.title("No strong DBSCAN clusters detected with current settings")
    else:
        sns.scatterplot(
            data=clustered,
            x="longitude",
            y="latitude",
            hue="cluster",
            palette="tab20",
            s=10,
            linewidth=0,
            alpha=0.75,
            legend=False,
        )
        plt.title("Detroit Crime Hotspot Clusters (DBSCAN)")

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def save_folium_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    center = [df["latitude"].median(), df["longitude"].median()]
    m = folium.Map(location=center, zoom_start=11, tiles="OpenStreetMap")

    heat_data = df[["latitude", "longitude"]].values.tolist()
    HeatMap(heat_data, radius=10, blur=12, max_zoom=13).add_to(m)

    m.save(str(out_path))


def save_spike_marker_map(spike_points: pd.DataFrame, out_path: Path) -> None:
    if spike_points.empty:
        return

    center = [spike_points["latitude"].median(), spike_points["longitude"].median()]
    m = folium.Map(location=center, zoom_start=11, tiles="OpenStreetMap")

    for _, row in spike_points.iterrows():
        radius = max(6, min(20, row["incident_count"] * 0.8))
        popup = (
            f"Neighborhood: {row['neighborhood']}<br>"
            f"Week Start: {row['week_start'].date()}<br>"
            f"Incidents: {int(row['incident_count'])}<br>"
            f"Spike Z-Score: {row['z_score']:.2f}"
        )
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=radius,
            color="#b30000",
            fill=True,
            fill_opacity=0.55,
            popup=popup,
        ).add_to(m)

    m.save(str(out_path))


def save_interactive_h3_choropleth(df: pd.DataFrame, out_path: Path, resolution: int = 8) -> None:
    temp = df.copy()
    temp["h3_cell"] = temp.apply(
        lambda row: h3.latlng_to_cell(float(row["latitude"]), float(row["longitude"]), resolution),
        axis=1,
    )

    cell_counts = build_h3_count_layer_data(df, resolution)

    center = [temp["latitude"].median(), temp["longitude"].median()]
    m = folium.Map(location=center, zoom_start=11, tiles="OpenStreetMap")
    Fullscreen(position="topright", title="Expand", title_cancel="Exit", force_separate_button=True).add_to(m)

    density_layer = folium.FeatureGroup(name="Density | All Incidents Heatmap", show=False)
    HeatMap(df[["latitude", "longitude"]].values.tolist(), radius=10, blur=12, max_zoom=13).add_to(
        density_layer
    )
    density_layer.add_to(m)

    add_h3_count_choropleth_layer(
        m,
        cell_counts,
        layer_name="Core | H3 Choropleth: Incident Count (All Incidents)",
        legend_caption="Crime incidents per hex cell (all incidents)",
        colors=["#f7fbff", "#6baed6", "#2171b5", "#08306b"],
        show=True,
    )

    latest_day = pd.to_datetime(df["incident_date"]).max().date()
    day_subset = df[df["incident_date"] == latest_day].copy()
    if not day_subset.empty:
        day_layer = build_h3_count_layer_data(day_subset, resolution)
        add_h3_count_choropleth_layer(
            m,
            day_layer,
            layer_name=f"Temporal | Latest Day Count ({latest_day})",
            legend_caption=f"Latest day incidents per hex cell ({latest_day})",
            colors=["#f0fdf4", "#86efac", "#22c55e", "#14532d"],
            show=False,
        )

    latest_month = pd.to_datetime(df["month_start"]).max().strftime("%Y-%m")
    month_subset = df[df["month_start"].dt.strftime("%Y-%m") == latest_month].copy()
    if not month_subset.empty:
        month_layer = build_h3_count_layer_data(month_subset, resolution)
        add_h3_count_choropleth_layer(
            m,
            month_layer,
            layer_name=f"Temporal | Latest Month Count ({latest_month})",
            legend_caption=f"Latest month incidents per hex cell ({latest_month})",
            colors=["#f5f3ff", "#c4b5fd", "#8b5cf6", "#4c1d95"],
            show=False,
        )

    # Add optional spike-severity layer in this same map so users can toggle both views.
    weekly = detect_weekly_spikes(df)
    spike_context = weekly[["neighborhood", "week_start", "z_score"]].copy()
    temp_spike = df.merge(spike_context, on=["neighborhood", "week_start"], how="left")
    temp_spike["z_score"] = temp_spike["z_score"].clip(lower=0)
    temp_spike["h3_cell"] = temp_spike.apply(
        lambda row: h3.latlng_to_cell(float(row["latitude"]), float(row["longitude"]), resolution),
        axis=1,
    )
    severity = (
        temp_spike.groupby("h3_cell", as_index=False)
        .agg(spike_severity=("z_score", "mean"), incident_count=("z_score", "size"))
        .merge(build_h3_location_lookup(df, resolution), on="h3_cell", how="left")
        .merge(build_h3_incident_context_lookup(df, resolution), on="h3_cell", how="left")
    )

    sev_vmin = float(severity["spike_severity"].min())
    sev_vmax = float(severity["spike_severity"].max())
    sev_colormap = cm.LinearColormap(
        colors=["#fff5eb", "#fdae6b", "#e6550d", "#7f2704"],
        vmin=sev_vmin,
        vmax=sev_vmax,
    )
    sev_colormap.caption = "Average spike severity (mean positive z-score)"
    sev_colormap.add_to(m)

    sev_features = []
    for _, row in severity.iterrows():
        cell = row["h3_cell"]
        spike_severity = float(row["spike_severity"])
        incident_count = int(row["incident_count"])
        boundary = h3.cell_to_boundary(cell)
        coordinates = [[lng, lat] for lat, lng in boundary]
        if coordinates and coordinates[0] != coordinates[-1]:
            coordinates.append(coordinates[0])

        sev_features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [coordinates]},
                "properties": {
                    "h3_cell": cell,
                    "spike_severity": round(spike_severity, 3),
                    "incident_count": incident_count,
                    "neighborhood": str(row.get("neighborhood", "Unknown")),
                    "nearest_intersection": str(row.get("nearest_intersection", "Unknown")),
                    "police_precinct": str(row.get("police_precinct", "Unknown")),
                    "zip_code": str(row.get("zip_code", "Unknown")),
                    "dominant_offense": str(row.get("dominant_offense", "Unknown")),
                    "dominant_offense_share": f"{float(row.get('dominant_offense_share', 0) or 0):.1%}",
                    "dominant_shift": str(row.get("dominant_shift", "Unknown")),
                    "dominant_shift_share": f"{float(row.get('dominant_shift_share', 0) or 0):.1%}",
                    "fill_color": sev_colormap(spike_severity),
                },
            }
        )

    sev_geojson = {"type": "FeatureCollection", "features": sev_features}
    severity_layer = folium.FeatureGroup(name="Core | H3 Choropleth: Spike Severity (All Incidents)", show=False)
    folium.GeoJson(
        sev_geojson,
        style_function=lambda feature: {
            "fillColor": feature["properties"]["fill_color"],
            "color": "#2b2b2b",
            "weight": 0.55,
            "opacity": 0.65,
            "fillOpacity": 0.46,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=[
                "neighborhood",
                "nearest_intersection",
                "police_precinct",
                "zip_code",
                "dominant_offense",
                "dominant_offense_share",
                "dominant_shift",
                "dominant_shift_share",
                "h3_cell",
                "spike_severity",
                "incident_count",
            ],
            aliases=[
                "Neighborhood",
                "Nearest Intersection",
                "Precinct",
                "ZIP",
                "Dominant Incident Type",
                "Incident Type Share",
                "Dominant Shift",
                "Shift Share",
                "Grid ID",
                "Spike Severity",
                "Incident Count",
            ],
            localize=True,
        ),
    ).add_to(severity_layer)
    severity_layer.add_to(m)

    intersection_markers = build_top_intersection_markers(df, top_n=80)
    add_marker_cluster_layer(
        m,
        intersection_markers,
        layer_name="Action | Top Intersection Markers",
        color="#b91c1c",
        rank_field="location_rank",
        top_n=80,
        show=False,
    )

    add_crime_type_and_shift_layers(m, df, top_n_categories=None)
    add_crime_type_h3_count_layers(m, df, resolution=resolution, top_n_categories=None)
    add_precinct_filter_layers(m, df)
    add_focus_category_layers(m, df)
        # Build matched year-to-date precinct comparison for the dashboard selector.
    available_years = sorted(df["incident_year"].dropna().astype(int).unique())
    current_year = max(available_years)
    previous_year = current_year - 1
    baseline_year = current_year - 2

    precinct_improvement = build_precinct_improvement_table(
        df,
        current_year=current_year,
        previous_year=previous_year,
        baseline_year=baseline_year,
    )
    add_top_selector_panel(
        m,
        sorted(df["precinct_norm"].dropna().astype(str).unique().tolist()),
        sorted(df["neighborhood"].dropna().astype(str).unique().tolist()),
        sorted(df["offense_category"].dropna().astype(str).unique().tolist()),
        precinct_improvement=precinct_improvement,
        current_year=current_year,
        previous_year=previous_year,
        baseline_year=baseline_year,
    )

    #add_map_help_box(m)
    folium.LayerControl(collapsed=True, hideSingleBase=True).add_to(m)

    m.save(str(out_path))


def save_interactive_h3_spike_severity_choropleth(
    df: pd.DataFrame, weekly: pd.DataFrame, out_path: Path, resolution: int = 8
) -> None:
    spike_context = weekly[["neighborhood", "week_start", "z_score"]].copy()
    temp = df.merge(spike_context, on=["neighborhood", "week_start"], how="left")

    # Keep only positive anomaly values so severity reflects unusual surges.
    temp["z_score"] = temp["z_score"].clip(lower=0)
    temp["h3_cell"] = temp.apply(
        lambda row: h3.latlng_to_cell(float(row["latitude"]), float(row["longitude"]), resolution),
        axis=1,
    )

    severity = (
        temp.groupby("h3_cell", as_index=False)
        .agg(
            spike_severity=("z_score", "mean"),
            incident_count=("z_score", "size"),
        )
        .sort_values("spike_severity", ascending=False)
    )
    location_lookup = build_h3_location_lookup(df, resolution)
    incident_lookup = build_h3_incident_context_lookup(df, resolution)
    severity = (
        severity.merge(location_lookup, on="h3_cell", how="left")
        .merge(incident_lookup, on="h3_cell", how="left")
    )

    center = [temp["latitude"].median(), temp["longitude"].median()]
    m = folium.Map(location=center, zoom_start=11, tiles="OpenStreetMap")
    Fullscreen(position="topright", title="Expand", title_cancel="Exit", force_separate_button=True).add_to(m)

    density_layer = folium.FeatureGroup(name="Density | All Incidents Heatmap", show=False)
    HeatMap(df[["latitude", "longitude"]].values.tolist(), radius=10, blur=12, max_zoom=13).add_to(
        density_layer
    )
    density_layer.add_to(m)

    vmin = float(severity["spike_severity"].min())
    vmax = float(severity["spike_severity"].max())
    colormap = cm.LinearColormap(
        colors=["#fff5eb", "#fdae6b", "#e6550d", "#7f2704"],
        vmin=vmin,
        vmax=vmax,
    )
    colormap.caption = "Average spike severity (mean positive z-score)"
    colormap.add_to(m)

    features = []
    for _, row in severity.iterrows():
        cell = row["h3_cell"]
        spike_severity = float(row["spike_severity"])
        incident_count = int(row["incident_count"])
        boundary = h3.cell_to_boundary(cell)
        coordinates = [[lng, lat] for lat, lng in boundary]
        if coordinates and coordinates[0] != coordinates[-1]:
            coordinates.append(coordinates[0])

        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [coordinates],
                },
                "properties": {
                    "h3_cell": cell,
                    "spike_severity": round(spike_severity, 3),
                    "incident_count": incident_count,
                    "neighborhood": str(row.get("neighborhood", "Unknown")),
                    "nearest_intersection": str(row.get("nearest_intersection", "Unknown")),
                    "police_precinct": str(row.get("police_precinct", "Unknown")),
                    "zip_code": str(row.get("zip_code", "Unknown")),
                    "dominant_offense": str(row.get("dominant_offense", "Unknown")),
                    "dominant_offense_share": f"{float(row.get('dominant_offense_share', 0) or 0):.1%}",
                    "dominant_shift": str(row.get("dominant_shift", "Unknown")),
                    "dominant_shift_share": f"{float(row.get('dominant_shift_share', 0) or 0):.1%}",
                    "fill_color": colormap(spike_severity),
                },
            }
        )

    geojson = {"type": "FeatureCollection", "features": features}

    severity_layer = folium.FeatureGroup(name="Core | H3 Choropleth: Spike Severity", show=True)
    folium.GeoJson(
        geojson,
        style_function=lambda feature: {
            "fillColor": feature["properties"]["fill_color"],
            "color": "#2b2b2b",
            "weight": 0.3,
            "fillOpacity": 0.7,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=[
                "neighborhood",
                "nearest_intersection",
                "police_precinct",
                "zip_code",
                "dominant_offense",
                "dominant_offense_share",
                "dominant_shift",
                "dominant_shift_share",
                "h3_cell",
                "spike_severity",
                "incident_count",
            ],
            aliases=[
                "Neighborhood",
                "Nearest Intersection",
                "Precinct",
                "ZIP",
                "Dominant Incident Type",
                "Incident Type Share",
                "Dominant Shift",
                "Shift Share",
                "Grid ID",
                "Spike Severity",
                "Incident Count",
            ],
            localize=True,
        ),
    ).add_to(severity_layer)
    severity_layer.add_to(m)

    intersection_markers = build_top_intersection_markers(df, top_n=80)
    add_marker_cluster_layer(
        m,
        intersection_markers,
        layer_name="Action | Top Intersection Markers",
        color="#b91c1c",
        rank_field="location_rank",
        top_n=80,
        show=False,
    )

    add_crime_type_and_shift_layers(m, df, top_n_categories=5)

    #add_map_help_box(m)
    folium.LayerControl(collapsed=True, hideSingleBase=True).add_to(m)

    m.save(str(out_path))


def build_offense_type_summary(df: pd.DataFrame) -> pd.DataFrame:
    offense_summary = (
        df.groupby("offense_category", as_index=False)
        .agg(
            incident_count=("crime_id", "count"),
            latitude=("latitude", "median"),
            longitude=("longitude", "median"),
        )
        .sort_values("incident_count", ascending=False)
    )
    return offense_summary


def save_offense_type_bar_chart(offense_summary: pd.DataFrame, out_path: Path) -> None:
    top = offense_summary.head(12)
    plt.figure(figsize=(12, 7))
    sns.barplot(data=top, y="offense_category", x="incident_count", color="#006d77")
    plt.title("Top Crime Types by Incident Count")
    plt.xlabel("Incidents")
    plt.ylabel("Offense Category")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def save_interactive_offense_type_heatmaps(df: pd.DataFrame, out_path: Path, top_n: int = 6) -> None:
    offense_counts = df["offense_category"].value_counts().head(top_n)
    top_categories = offense_counts.index.tolist()

    center = [df["latitude"].median(), df["longitude"].median()]
    m = folium.Map(location=center, zoom_start=11, tiles="OpenStreetMap")

    for idx, category in enumerate(top_categories):
        subset = df[df["offense_category"] == category]
        if subset.empty:
            continue

        layer = folium.FeatureGroup(name=f"{category} ({len(subset):,})", show=(idx == 0))
        heat_data = subset[["latitude", "longitude"]].values.tolist()
        HeatMap(heat_data, radius=10, blur=12, max_zoom=13).add_to(layer)
        layer.add_to(m)

    folium.LayerControl(collapsed=True, hideSingleBase=True).add_to(m)
    m.save(str(out_path))


def save_combined_interactive_dashboard(
    df: pd.DataFrame,
    weekly: pd.DataFrame,
    spike_points: pd.DataFrame,
    focus_locations: pd.DataFrame,
    out_path: Path,
    resolution: int = 8,
    top_n_categories: int | None = None,
    precinct_improvement: pd.DataFrame | None = None,
    precinct_crime_trends: pd.DataFrame | None = None,
    precinct_crime_14d: pd.DataFrame | None = None,
    priority_concerns: pd.DataFrame | None = None,
    temporal_summary: pd.DataFrame | None = None,
    temporal_matrix: pd.DataFrame | None = None,
    hotspot_change: pd.DataFrame | None = None,
    current_year: int | None = None,
    previous_year: int | None = None,
    baseline_year: int | None = None,
) -> None:
    center = [df["latitude"].median(), df["longitude"].median()]

    # Use a keyless public basemap by default; analytical H3 layers sit transparently above it.
    m = folium.Map(
        location=center,
        zoom_start=11,
        tiles=None,
        control_scale=True,
        prefer_canvas=True,
    )
    folium.TileLayer(
        tiles="OpenStreetMap",
        name="Basemap | OpenStreetMap (Default)",
        control=True,
        show=True,
    ).add_to(m)

    Fullscreen(position="topright", title="Expand", title_cancel="Exit", force_separate_button=True).add_to(m)

    # Optional underlying event geography: latest 14 days only, so the layer remains
    # useful and responsive instead of attempting to draw ~200k individual markers.
    date_series = pd.to_datetime(df["incident_occurred_at"], errors="coerce", utc=True)
    latest_date = date_series.max()
    if pd.notna(latest_date):
        recent_start = latest_date - pd.Timedelta(days=RECENT_WINDOW_DAYS - 1)
        recent_points = df.loc[date_series.between(recent_start, latest_date)].copy()
        recent_points = recent_points.dropna(subset=["latitude", "longitude"])
        if not recent_points.empty:
            recent_layer = folium.FeatureGroup(
                name=f"Locations | Actual Incidents — Latest 14D ({len(recent_points):,})",
                show=False,
            )
            FastMarkerCluster(
                recent_points[["latitude", "longitude"]].astype(float).values.tolist(),
                name="Recent Incident Locations",
                disableClusteringAtZoom=16,
            ).add_to(recent_layer)
            recent_layer.add_to(m)

    temp_time = df.copy()
    temp_time["shift_window"] = temp_time["incident_hour_of_day"].apply(assign_shift_window)
    temp_time["decision_purpose"] = temp_time["offense_category"].apply(assign_decision_purpose)

    # Layer 1: all incidents heatmap
    all_heat_layer = folium.FeatureGroup(name="Core | Incident Density Heatmap", show=True)
    all_heat_data = df[["latitude", "longitude"]].values.tolist()
    HeatMap(all_heat_data, radius=11, blur=13, max_zoom=15, min_opacity=0.18).add_to(all_heat_layer)
    all_heat_layer.add_to(m)

    # Layer 2: spike markers
    spike_layer = folium.FeatureGroup(name="Core | Spike Week Markers", show=False)
    cluster = MarkerCluster(name="Spike Marker Clusters")
    if not spike_points.empty:
        for _, row in spike_points.iterrows():
            radius = max(6, min(20, row["incident_count"] * 0.8))
            popup = (
                f"Neighborhood: {row['neighborhood']}<br>"
                f"Week Start: {row['week_start'].date()}<br>"
                f"Incidents: {int(row['incident_count'])}<br>"
                f"Spike Z-Score: {row['z_score']:.2f}"
            )
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=radius,
                color="#b30000",
                fill=True,
                fill_opacity=0.55,
                popup=popup,
                ).add_to(cluster)
            cluster.add_to(spike_layer)
    spike_layer.add_to(m)

    # Shared H3 cell assignment for choropleth layers.
    temp = df.copy()
    temp["h3_cell"] = temp.apply(
        lambda row: h3.latlng_to_cell(float(row["latitude"]), float(row["longitude"]), resolution),
        axis=1,
    )

    # Layer 3: incident count choropleth
    cell_counts = (
        temp.groupby("h3_cell", as_index=False)
        .size()
        .rename(columns={"size": "crime_count"})
    )
    location_lookup = build_h3_location_lookup(df, resolution)
    cell_counts = cell_counts.merge(location_lookup, on="h3_cell", how="left")
    count_vmin = float(cell_counts["crime_count"].min())
    count_vmax = float(cell_counts["crime_count"].max())
    count_colormap = cm.LinearColormap(
        colors=["#f7fbff", "#6baed6", "#2171b5", "#08306b"],
        vmin=count_vmin,
        vmax=count_vmax,
    )
    count_colormap.caption = "Crime incidents per hex cell (core layer, all incidents)"
    count_colormap.add_to(m)

    count_features = []
    for _, row in cell_counts.iterrows():
        cell = row["h3_cell"]
        count = int(row["crime_count"])
        boundary = h3.cell_to_boundary(cell)
        coordinates = [[lng, lat] for lat, lng in boundary]
        if coordinates and coordinates[0] != coordinates[-1]:
            coordinates.append(coordinates[0])

        count_features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [coordinates]},
                "properties": {
                    "h3_cell": cell,
                    "crime_count": count,
                    "neighborhood": str(row.get("neighborhood", "Unknown")),
                    "nearest_intersection": str(row.get("nearest_intersection", "Unknown")),
                    "police_precinct": str(row.get("police_precinct", "Unknown")),
                    "zip_code": str(row.get("zip_code", "Unknown")),
                    "fill_color": count_colormap(count),
                },
            }
        )

    count_geojson = {"type": "FeatureCollection", "features": count_features}
    count_layer = folium.FeatureGroup(name="Core | H3 Choropleth: Incident Count (All Incidents)", show=False)
    folium.GeoJson(
        count_geojson,
        style_function=lambda feature: {
            "fillColor": feature["properties"]["fill_color"],
            "color": "#2b2b2b",
            "weight": 0.22,
            "fillOpacity": 0.34,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=[
                "neighborhood",
                "nearest_intersection",
                "police_precinct",
                "zip_code",
                "h3_cell",
                "crime_count",
            ],
            aliases=[
                "Neighborhood",
                "Nearest Intersection",
                "Precinct",
                "ZIP",
                "Grid ID",
                "Crime Count",
            ],
            localize=True,
        ),
    ).add_to(count_layer)
    count_layer.add_to(m)

    # Layer 4: spike severity choropleth
    spike_context = weekly[["neighborhood", "week_start", "z_score"]].copy()
    temp_spike = df.merge(spike_context, on=["neighborhood", "week_start"], how="left")
    temp_spike["z_score"] = temp_spike["z_score"].clip(lower=0)
    temp_spike["h3_cell"] = temp_spike.apply(
        lambda row: h3.latlng_to_cell(float(row["latitude"]), float(row["longitude"]), resolution),
        axis=1,
    )
    severity = (
        temp_spike.groupby("h3_cell", as_index=False)
        .agg(spike_severity=("z_score", "mean"), incident_count=("z_score", "size"))
    )
    severity = severity.merge(location_lookup, on="h3_cell", how="left")
    sev_vmin = float(severity["spike_severity"].min())
    sev_vmax = float(severity["spike_severity"].max())
    sev_colormap = cm.LinearColormap(
        colors=["#fff5eb", "#fdae6b", "#e6550d", "#7f2704"],
        vmin=sev_vmin,
        vmax=sev_vmax,
    )
    sev_colormap.caption = "Average spike severity (mean positive z-score)"

    sev_features = []
    for _, row in severity.iterrows():
        cell = row["h3_cell"]
        spike_severity = float(row["spike_severity"])
        incident_count = int(row["incident_count"])
        boundary = h3.cell_to_boundary(cell)
        coordinates = [[lng, lat] for lat, lng in boundary]
        if coordinates and coordinates[0] != coordinates[-1]:
            coordinates.append(coordinates[0])

        sev_features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [coordinates]},
                "properties": {
                    "h3_cell": cell,
                    "spike_severity": round(spike_severity, 3),
                    "incident_count": incident_count,
                    "neighborhood": str(row.get("neighborhood", "Unknown")),
                    "nearest_intersection": str(row.get("nearest_intersection", "Unknown")),
                    "police_precinct": str(row.get("police_precinct", "Unknown")),
                    "zip_code": str(row.get("zip_code", "Unknown")),
                    "fill_color": sev_colormap(spike_severity),
                },
            }
        )

    sev_geojson = {"type": "FeatureCollection", "features": sev_features}
    sev_layer = folium.FeatureGroup(name="Core | H3 Choropleth: Spike Severity (All Incidents)", show=False)
    folium.GeoJson(
        sev_geojson,
        style_function=lambda feature: {
            "fillColor": feature["properties"]["fill_color"],
            "color": "#2b2b2b",
            "weight": 0.3,
            "fillOpacity": 0.7,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=[
                "neighborhood",
                "nearest_intersection",
                "police_precinct",
                "zip_code",
                "h3_cell",
                "spike_severity",
                "incident_count",
            ],
            aliases=[
                "Neighborhood",
                "Nearest Intersection",
                "Precinct",
                "ZIP",
                "Grid ID",
                "Spike Severity",
                "Incident Count",
            ],
            localize=True,
        ),
    ).add_to(sev_layer)
    sev_layer.add_to(m)

    add_marker_cluster_layer(
        m,
        focus_locations,
        layer_name="Action | Focus Location Markers",
        color="#7f1d1d",
        rank_field="focus_rank",
        top_n=80,
        show=False,
    )

    # Scope-specific heatmaps are intentionally NOT pre-generated here.  The old
    # implementation embedded the same incident coordinates repeatedly for every
    # decision purpose, offense, shift, precinct, focus category, and neighborhood.
    # With current RMS volume that inflated the single HTML file beyond GitHub's
    # publishable size.  The selector panel now builds the selected precinct /
    # neighborhood / crime-focus heatmap on demand from ``spatial_daily`` instead.
    # This preserves analytical filtering while keeping one compact browser payload.

    # Bounds power precinct auto-zoom while preserving the real street basemap.
    precinct_bounds: dict[str, list[list[float]]] = {}
    neighborhood_bounds: dict[str, list[list[float]]] = {}
    valid_geo = df.dropna(subset=["latitude", "longitude"]).copy()
    if not valid_geo.empty:
        precinct_bounds["ALL"] = [
            [float(valid_geo["latitude"].min()), float(valid_geo["longitude"].min())],
            [float(valid_geo["latitude"].max()), float(valid_geo["longitude"].max())],
        ]
        for p, g in valid_geo.groupby("precinct_norm"):
            precinct_bounds[str(p)] = [
                [float(g["latitude"].min()), float(g["longitude"].min())],
                [float(g["latitude"].max()), float(g["longitude"].max())],
            ]
        for neighborhood, g in valid_geo.groupby("neighborhood"):
            neighborhood_bounds[str(neighborhood)] = [
                [float(g["latitude"].min()), float(g["longitude"].min())],
                [float(g["latitude"].max()), float(g["longitude"].max())],
            ]

    spatial_daily = build_spatial_daily_payload(df, current_year=current_year, resolution=resolution)

    add_top_selector_panel(
        m,
        sorted(df["precinct_norm"].dropna().astype(str).unique().tolist()),
        sorted(df["neighborhood"].dropna().astype(str).unique().tolist()),
        sorted(df["offense_category"].dropna().astype(str).unique().tolist()),
        precinct_improvement=precinct_improvement,
        precinct_crime_trends=precinct_crime_trends,
        precinct_crime_14d=precinct_crime_14d,
        priority_concerns=priority_concerns,
        temporal_summary=temporal_summary,
        temporal_matrix=temporal_matrix,
        hotspot_change=hotspot_change,
        spatial_daily=spatial_daily,
        precinct_bounds=precinct_bounds,
        neighborhood_bounds=neighborhood_bounds,
        current_year=current_year,
        previous_year=previous_year,
        baseline_year=baseline_year,
    )

    # Top selector panel now contains usage guidance; avoid a second overlapping help box.
    folium.LayerControl(collapsed=True, hideSingleBase=True).add_to(m)
    m.save(str(out_path))


def build_focus_locations(
    df: pd.DataFrame,
    weekly: pd.DataFrame,
    resolution: int = 9,
    min_incidents: int = 15,
) -> pd.DataFrame:
    spike_context = weekly[["neighborhood", "week_start", "z_score"]].copy()
    temp = df.merge(spike_context, on=["neighborhood", "week_start"], how="left")
    temp["z_score"] = temp["z_score"].fillna(0)
    temp["h3_cell"] = temp.apply(
        lambda row: h3.latlng_to_cell(float(row["latitude"]), float(row["longitude"]), resolution),
        axis=1,
    )

    grouped = temp.groupby("h3_cell")
    summary = grouped.agg(
        incident_count=("crime_id", "count"),
        latitude=("latitude", "median"),
        longitude=("longitude", "median"),
        mean_spike_z=("z_score", "mean"),
        neighborhood=("neighborhood", lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]),
        police_precinct=(
            "police_precinct",
            lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0],
        ),
        nearest_intersection=(
            "nearest_intersection",
            lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0],
        ),
    ).reset_index()

    dominant_offense = (
        temp.groupby(["h3_cell", "offense_category"]).size().reset_index(name="offense_count")
    )
    idx = dominant_offense.groupby("h3_cell")["offense_count"].idxmax()
    dominant = dominant_offense.loc[idx].rename(
        columns={"offense_category": "dominant_offense", "offense_count": "dominant_offense_count"}
    )

    focus = summary.merge(dominant[["h3_cell", "dominant_offense", "dominant_offense_count"]], on="h3_cell")
    focus["dominant_offense_share"] = focus["dominant_offense_count"] / focus["incident_count"]

    focus = focus[focus["incident_count"] >= min_incidents].copy()

    focus["score"] = (
        0.75 * (focus["incident_count"] / max(1, focus["incident_count"].max()))
        + 0.25 * (focus["mean_spike_z"].clip(lower=0) / max(1e-6, focus["mean_spike_z"].clip(lower=0).max()))
    )

    focus = focus.sort_values(["score", "incident_count"], ascending=False).reset_index(drop=True)
    focus.insert(0, "focus_rank", np.arange(1, len(focus) + 1))
    return focus


def build_top_intersection_markers(df: pd.DataFrame, top_n: int = 80) -> pd.DataFrame:
    temp = df.dropna(subset=["nearest_intersection", "latitude", "longitude"]).copy()
    markers = (
        temp.groupby("nearest_intersection", as_index=False)
        .agg(
            incident_count=("crime_id", "count"),
            latitude=("latitude", "median"),
            longitude=("longitude", "median"),
            neighborhood=("neighborhood", lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]),
            police_precinct=(
                "police_precinct",
                lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0],
            ),
        )
        .sort_values("incident_count", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    markers.insert(0, "location_rank", np.arange(1, len(markers) + 1))
    return markers


def add_marker_cluster_layer(
    m: folium.Map,
    markers: pd.DataFrame,
    layer_name: str,
    color: str,
    rank_field: str,
    top_n: int = 80,
    show: bool = False,
) -> None:
    layer = folium.FeatureGroup(name=layer_name, show=show)
    cluster = MarkerCluster(name=f"{layer_name} Cluster")

    for _, row in markers.head(top_n).iterrows():
        rank = int(row[rank_field]) if rank_field in row and pd.notna(row[rank_field]) else None
        rank_text = f"{rank}" if rank is not None else "-"
        neighborhood = row.get("neighborhood", "Unknown")
        precinct = row.get("police_precinct", "Unknown")
        intersection = row.get("nearest_intersection", "Unknown")
        incidents = int(row.get("incident_count", 0))

        popup = (
            f"Rank: {rank_text}<br>"
            f"Intersection: {intersection}<br>"
            f"Neighborhood: {neighborhood}<br>"
            f"Precinct: {precinct}<br>"
            f"Incidents: {incidents}"
        )

        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=max(5, min(14, int(np.sqrt(max(1, incidents)) * 1.3))),
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.65,
            popup=popup,
        ).add_to(cluster)

    cluster.add_to(layer)
    layer.add_to(m)


def save_focus_locations_map(focus: pd.DataFrame, out_path: Path, top_n: int = 40) -> None:
    top = focus.head(top_n).copy()
    center = [top["latitude"].median(), top["longitude"].median()]
    m = folium.Map(location=center, zoom_start=11, tiles="OpenStreetMap")

    colormap = cm.LinearColormap(
        colors=["#fee5d9", "#fcae91", "#fb6a4a", "#cb181d"],
        vmin=float(top["score"].min()),
        vmax=float(top["score"].max()),
    )
    colormap.caption = "Operational priority score"
    colormap.add_to(m)

    for _, row in top.iterrows():
        popup = (
            f"Focus Rank: {int(row['focus_rank'])}<br>"
            f"Precinct: {row['police_precinct']}<br>"
            f"Neighborhood: {row['neighborhood']}<br>"
            f"Nearest Intersection: {row['nearest_intersection']}<br>"
            f"Incidents: {int(row['incident_count'])}<br>"
            f"Dominant Crime: {row['dominant_offense']} ({int(row['dominant_offense_count'])})<br>"
            f"Dominant Share: {row['dominant_offense_share']:.1%}<br>"
            f"Mean Spike Z: {row['mean_spike_z']:.2f}<br>"
            f"Priority Score: {row['score']:.3f}"
        )
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=max(6, min(18, int(np.sqrt(row["incident_count"]) * 1.8))),
            color="#7f0000",
            fill=True,
            fill_color=colormap(float(row["score"])),
            fill_opacity=0.7,
            popup=popup,
        ).add_to(m)

    m.save(str(out_path))


def build_precinct_summary(df: pd.DataFrame, weekly: pd.DataFrame) -> pd.DataFrame:
    spike_weeks = weekly[weekly["is_spike"]][["neighborhood", "week_start"]].drop_duplicates()
    temp = df.merge(spike_weeks.assign(is_spike_week=True), on=["neighborhood", "week_start"], how="left")
    temp["is_spike_week"] = temp["is_spike_week"].eq(True)

    precinct_summary = (
        temp.groupby("police_precinct", as_index=False)
        .agg(
            incidents=("crime_id", "count"),
            spike_week_incidents=("is_spike_week", "sum"),
            neighborhoods_covered=("neighborhood", "nunique"),
        )
        .sort_values("incidents", ascending=False)
    )

    top_offense = (
        temp.groupby(["police_precinct", "offense_category"]).size().reset_index(name="count")
    )
    idx = top_offense.groupby("police_precinct")["count"].idxmax()
    top_offense = top_offense.loc[idx].rename(
        columns={"offense_category": "dominant_crime", "count": "dominant_crime_count"}
    )

    precinct_summary = precinct_summary.merge(top_offense, on="police_precinct", how="left")
    precinct_summary["dominant_crime_share"] = (
        precinct_summary["dominant_crime_count"] / precinct_summary["incidents"]
    )
    precinct_summary["spike_incident_share"] = (
        precinct_summary["spike_week_incidents"] / precinct_summary["incidents"]
    )
    return precinct_summary


def save_precinct_bar_chart(precinct_summary: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(11, 6))
    ordered = precinct_summary.sort_values("incidents", ascending=False)
    sns.barplot(data=ordered, x="police_precinct", y="incidents", color="#264653")
    plt.title("Total Incidents by Police Precinct")
    plt.xlabel("Police Precinct")
    plt.ylabel("Incidents")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def build_shift_summary(df: pd.DataFrame) -> pd.DataFrame:
    temp = df.copy()
    temp["shift_window"] = temp["incident_hour_of_day"].apply(assign_shift_window)
    shift_summary = (
        temp.groupby(["shift_window", "offense_category"], as_index=False)
        .size()
        .rename(columns={"size": "incident_count"})
    )

    total_by_shift = (
        shift_summary.groupby("shift_window", as_index=False)["incident_count"].sum()
        .rename(columns={"incident_count": "shift_total_incidents"})
    )
    idx = shift_summary.groupby("shift_window")["incident_count"].idxmax()
    dominant = shift_summary.loc[idx].rename(
        columns={"offense_category": "dominant_offense", "incident_count": "dominant_offense_count"}
    )

    out = total_by_shift.merge(dominant[["shift_window", "dominant_offense", "dominant_offense_count"]], on="shift_window")
    out["dominant_offense_share"] = out["dominant_offense_count"] / out["shift_total_incidents"]

    order = {
        "Day Shift (06:00-13:59)": 1,
        "Evening Shift (14:00-21:59)": 2,
        "Night Shift (22:00-05:59)": 3,
        "Unknown": 4,
    }
    out["sort_order"] = out["shift_window"].map(order).fillna(9)
    out = out.sort_values("sort_order").drop(columns=["sort_order"]).reset_index(drop=True)
    return out


def save_shift_summary_chart(shift_summary: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(9, 5))
    sns.barplot(data=shift_summary, x="shift_window", y="shift_total_incidents", color="#2a9d8f")
    plt.title("Incidents by Shift Window")
    plt.xlabel("Shift Window")
    plt.ylabel("Incidents")
    plt.xticks(rotation=12, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def build_decision_purpose_summary(df: pd.DataFrame) -> pd.DataFrame:
    temp = df.copy()
    temp["decision_purpose"] = temp["offense_category"].apply(assign_decision_purpose)

    summary = (
        temp.groupby(["decision_purpose", "offense_category"], as_index=False)
        .size()
        .rename(columns={"size": "incident_count"})
    )
    total = (
        summary.groupby("decision_purpose", as_index=False)["incident_count"].sum()
        .rename(columns={"incident_count": "purpose_total_incidents"})
    )

    idx = summary.groupby("decision_purpose")["incident_count"].idxmax()
    dominant = summary.loc[idx].rename(
        columns={"offense_category": "dominant_offense", "incident_count": "dominant_offense_count"}
    )

    out = total.merge(
        dominant[["decision_purpose", "dominant_offense", "dominant_offense_count"]],
        on="decision_purpose",
        how="left",
    )
    out["dominant_offense_share"] = out["dominant_offense_count"] / out["purpose_total_incidents"]
    out = out.sort_values("purpose_total_incidents", ascending=False).reset_index(drop=True)
    return out


def save_decision_purpose_chart(decision_summary: pd.DataFrame, out_path: Path) -> None:
    palette = {
        "Preventive Patrol": "#2563eb",
        "Investigations": "#9333ea",
        "Community Response": "#dc2626",
    }

    plt.figure(figsize=(9, 5))
    sns.barplot(
        data=decision_summary,
        x="decision_purpose",
        y="purpose_total_incidents",
        hue="decision_purpose",
        palette=palette,
        legend=False,
    )
    plt.title("Incidents by Decision Response Purpose")
    plt.xlabel("Response Purpose")
    plt.ylabel("Incidents")
    plt.xticks(rotation=12, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def build_daily_monthly_citywide_trends(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = (
        df.groupby("incident_date", as_index=False)
        .size()
        .rename(columns={"size": "incident_count"})
        .sort_values("incident_date")
    )
    monthly = (
        df.groupby("month_start", as_index=False)
        .size()
        .rename(columns={"size": "incident_count"})
        .sort_values("month_start")
    )
    return daily, monthly


def _add_change_fields(comparison_df: pd.DataFrame, previous_col: str, current_col: str) -> pd.DataFrame:
    out = comparison_df.copy()
    out["change"] = out[current_col] - out[previous_col]
    out["pct_change"] = np.where(
        out[previous_col] > 0,
        100 * out["change"] / out[previous_col],
        np.nan,
    )
    out["improvement"] = np.where(out["change"] < 0, "Improved", np.where(out["change"] > 0, "Worse", "No Change"))
    return out


def build_temporal_yoy_comparisons(
    df: pd.DataFrame,
    current_year: int,
    previous_year: int,
) -> dict[str, pd.DataFrame]:
    temp = df.copy()
    temp["incident_date"] = pd.to_datetime(temp["incident_date"])
    temp["incident_hour_of_day"] = pd.to_numeric(temp["incident_hour_of_day"], errors="coerce").fillna(0).astype(int)

    yoy = temp[temp["incident_year"].isin([previous_year, current_year])].copy()

    daily_base = (
        yoy.assign(month_day=yoy["incident_date"].dt.strftime("%m-%d"))
        .groupby(["month_day", "incident_year"], as_index=False)
        .size()
        .rename(columns={"size": "incident_count"})
    )
    daily = daily_base.pivot(index="month_day", columns="incident_year", values="incident_count").fillna(0).reset_index()
    daily = daily.rename(columns={previous_year: "incidents_previous", current_year: "incidents_current"})
    daily = _add_change_fields(daily, "incidents_previous", "incidents_current")

    hourly_base = (
        yoy.groupby(["incident_hour_of_day", "incident_year"], as_index=False)
        .size()
        .rename(columns={"size": "incident_count"})
    )
    hourly = hourly_base.pivot(index="incident_hour_of_day", columns="incident_year", values="incident_count").fillna(0).reset_index()
    hourly = hourly.rename(columns={previous_year: "incidents_previous", current_year: "incidents_current"})
    hourly = _add_change_fields(hourly, "incidents_previous", "incidents_current")

    weekly_base = (
        yoy.assign(iso_week=yoy["incident_date"].dt.isocalendar().week.astype(int))
        .groupby(["iso_week", "incident_year"], as_index=False)
        .size()
        .rename(columns={"size": "incident_count"})
    )
    weekly = weekly_base.pivot(index="iso_week", columns="incident_year", values="incident_count").fillna(0).reset_index()
    weekly = weekly.rename(columns={previous_year: "incidents_previous", current_year: "incidents_current"})
    weekly = _add_change_fields(weekly, "incidents_previous", "incidents_current")

    monthly_base = (
        yoy.assign(month=yoy["incident_date"].dt.month)
        .groupby(["month", "incident_year"], as_index=False)
        .size()
        .rename(columns={"size": "incident_count"})
    )
    monthly = monthly_base.pivot(index="month", columns="incident_year", values="incident_count").fillna(0).reset_index()
    monthly = monthly.rename(columns={previous_year: "incidents_previous", current_year: "incidents_current"})
    monthly = _add_change_fields(monthly, "incidents_previous", "incidents_current")

    yearly = (
        temp.groupby("incident_year", as_index=False)
        .size()
        .rename(columns={"size": "incident_count"})
        .sort_values("incident_year")
        .reset_index(drop=True)
    )
    yearly["change_vs_previous_year"] = yearly["incident_count"].diff()
    yearly["pct_change_vs_previous_year"] = yearly["incident_count"].pct_change() * 100

    shifts = yoy.copy()
    shifts["shift_window"] = shifts["incident_hour_of_day"].apply(assign_shift_window)
    shift_base = (
        shifts.groupby(["shift_window", "incident_year"], as_index=False)
        .size()
        .rename(columns={"size": "incident_count"})
    )
    shift = shift_base.pivot(index="shift_window", columns="incident_year", values="incident_count").fillna(0).reset_index()
    shift = shift.rename(columns={previous_year: "incidents_previous", current_year: "incidents_current"})
    shift = _add_change_fields(shift, "incidents_previous", "incidents_current")

    return {
        "daily": daily,
        "hourly": hourly,
        "weekly": weekly,
        "monthly": monthly,
        "yearly": yearly,
        "shift": shift,
    }


def build_precinct_temporal_yoy_comparisons(
    df: pd.DataFrame,
    current_year: int,
    previous_year: int,
) -> dict[str, pd.DataFrame]:
    temp = df.copy()
    temp["incident_date"] = pd.to_datetime(temp["incident_date"])
    temp["incident_hour_of_day"] = pd.to_numeric(temp["incident_hour_of_day"], errors="coerce").fillna(0).astype(int)
    yoy = temp[temp["incident_year"].isin([previous_year, current_year])].copy()

    daily_base = (
        yoy.assign(month_day=yoy["incident_date"].dt.strftime("%m-%d"))
        .groupby(["precinct_norm", "month_day", "incident_year"], as_index=False)
        .size()
        .rename(columns={"size": "incident_count"})
    )
    daily = daily_base.pivot(
        index=["precinct_norm", "month_day"],
        columns="incident_year",
        values="incident_count",
    ).fillna(0).reset_index()
    daily = daily.rename(columns={previous_year: "incidents_previous", current_year: "incidents_current"})
    daily = _add_change_fields(daily, "incidents_previous", "incidents_current")

    hourly_base = (
        yoy.groupby(["precinct_norm", "incident_hour_of_day", "incident_year"], as_index=False)
        .size()
        .rename(columns={"size": "incident_count"})
    )
    hourly = hourly_base.pivot(
        index=["precinct_norm", "incident_hour_of_day"],
        columns="incident_year",
        values="incident_count",
    ).fillna(0).reset_index()
    hourly = hourly.rename(columns={previous_year: "incidents_previous", current_year: "incidents_current"})
    hourly = _add_change_fields(hourly, "incidents_previous", "incidents_current")

    weekly_base = (
        yoy.assign(iso_week=yoy["incident_date"].dt.isocalendar().week.astype(int))
        .groupby(["precinct_norm", "iso_week", "incident_year"], as_index=False)
        .size()
        .rename(columns={"size": "incident_count"})
    )
    weekly = weekly_base.pivot(
        index=["precinct_norm", "iso_week"],
        columns="incident_year",
        values="incident_count",
    ).fillna(0).reset_index()
    weekly = weekly.rename(columns={previous_year: "incidents_previous", current_year: "incidents_current"})
    weekly = _add_change_fields(weekly, "incidents_previous", "incidents_current")

    monthly_base = (
        yoy.assign(month=yoy["incident_date"].dt.month)
        .groupby(["precinct_norm", "month", "incident_year"], as_index=False)
        .size()
        .rename(columns={"size": "incident_count"})
    )
    monthly = monthly_base.pivot(
        index=["precinct_norm", "month"],
        columns="incident_year",
        values="incident_count",
    ).fillna(0).reset_index()
    monthly = monthly.rename(columns={previous_year: "incidents_previous", current_year: "incidents_current"})
    monthly = _add_change_fields(monthly, "incidents_previous", "incidents_current")

    return {
        "daily": daily,
        "hourly": hourly,
        "weekly": weekly,
        "monthly": monthly,
    }


def save_daily_monthly_trend_charts(
    daily: pd.DataFrame,
    monthly: pd.DataFrame,
    daily_out: Path,
    monthly_out: Path,
) -> None:
    plt.figure(figsize=(12, 5))
    plt.plot(pd.to_datetime(daily["incident_date"]), daily["incident_count"], color="#1d4ed8", linewidth=1.8)
    plt.title("Daily Incident Trend")
    plt.xlabel("Date")
    plt.ylabel("Incidents")
    plt.tight_layout()
    plt.savefig(daily_out, dpi=220)
    plt.close()

    plt.figure(figsize=(10, 5))
    month_labels = pd.to_datetime(monthly["month_start"]).dt.strftime("%Y-%m")
    sns.barplot(x=month_labels, y=monthly["incident_count"], color="#7c3aed")
    plt.title("Monthly Incident Trend")
    plt.xlabel("Month")
    plt.ylabel("Incidents")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(monthly_out, dpi=220)
    plt.close()


def save_temporal_yoy_charts(
    monthly_yoy: pd.DataFrame,
    hourly_yoy: pd.DataFrame,
    monthly_out: Path,
    hourly_out: Path,
) -> None:
    if not monthly_yoy.empty:
        temp = monthly_yoy.copy()
        x = np.arange(len(temp))
        width = 0.38
        plt.figure(figsize=(11, 5))
        plt.bar(x - width / 2, temp["incidents_previous"], width=width, label="Previous Year", color="#94a3b8")
        plt.bar(x + width / 2, temp["incidents_current"], width=width, label="Current Year", color="#1d4ed8")
        plt.xticks(x, temp["month"].astype(int).astype(str))
        plt.title("Monthly YoY Comparison")
        plt.xlabel("Month")
        plt.ylabel("Incidents")
        plt.legend()
        plt.tight_layout()
        plt.savefig(monthly_out, dpi=220)
        plt.close()

    if not hourly_yoy.empty:
        temp = hourly_yoy.copy()
        x = np.arange(len(temp))
        width = 0.38
        plt.figure(figsize=(13, 5))
        plt.bar(x - width / 2, temp["incidents_previous"], width=width, label="Previous Year", color="#a7f3d0")
        plt.bar(x + width / 2, temp["incidents_current"], width=width, label="Current Year", color="#059669")
        plt.xticks(x, temp["incident_hour_of_day"].astype(int).astype(str), rotation=0)
        plt.title("Hour-of-Day YoY Comparison")
        plt.xlabel("Hour")
        plt.ylabel("Incidents")
        plt.legend()
        plt.tight_layout()
        plt.savefig(hourly_out, dpi=220)
        plt.close()


def save_comparison_guide(out_path: Path, previous_year: int, current_year: int) -> None:
    guide = f"""Detroit Crime YoY Comparison Guide ({previous_year} vs {current_year})

This project now writes comparison tables for day, hour, week, month, shift, and year.

Key rules:
- incidents_previous = count in {previous_year}
- incidents_current = count in {current_year}
- change = incidents_current - incidents_previous
- pct_change = 100 * change / incidents_previous
- improvement = "Improved" when change < 0 (fewer incidents), "Worse" when change > 0

Files to use:
- city_daily_yoy_{previous_year}_vs_{current_year}.csv (month-day comparison)
- city_hourly_yoy_{previous_year}_vs_{current_year}.csv (hour-of-day comparison)
- city_weekly_yoy_{previous_year}_vs_{current_year}.csv (ISO week comparison)
- city_monthly_yoy_{previous_year}_vs_{current_year}.csv (month comparison)
- city_shift_yoy_{previous_year}_vs_{current_year}.csv (shift window comparison)
- city_yearly_totals_*.csv (long-term year trend)
- precinct_ytd_{previous_year}_vs_{current_year}.csv (precinct YTD comparison)
- precinct_daily_yoy_{previous_year}_vs_{current_year}.csv (precinct by day)
- precinct_hourly_yoy_{previous_year}_vs_{current_year}.csv (precinct by hour)
- precinct_weekly_yoy_{previous_year}_vs_{current_year}.csv (precinct by week)
- precinct_monthly_yoy_{previous_year}_vs_{current_year}.csv (precinct by month)
- target_area_ytd_{previous_year}_vs_{current_year}.csv (focus neighborhood YTD comparison)

Charts:
- detroit_monthly_yoy_comparison_{previous_year}_vs_{current_year}.png
- detroit_hourly_yoy_comparison_{previous_year}_vs_{current_year}.png

Map behavior:
- Crime Type selector now also switches matching H3 count layers: Core Type | H3 Count | <Crime Type>
- Core layers marked "All Incidents" remain citywide totals.
"""
    out_path.write_text(guide, encoding="utf-8")




def _norm_operational_precinct(value) -> str:
    """Normalize precinct IDs and keep special/non-operational codes distinct."""
    if pd.isna(value):
        return ""
    value = str(value).strip().upper()
    return value.zfill(2) if value.isdigit() else value


def _operational_precinct_values(df: pd.DataFrame) -> list[str]:
    invalid = {"", "00", "0W", "OW", "HP", "UNKNOWN", "NAN", "NONE"}
    values = {
        _norm_operational_precinct(v)
        for v in df["precinct_norm"].dropna().tolist()
    }
    return sorted(
        [p for p in values if p not in invalid],
        key=lambda x: (0, int(x)) if x.isdigit() else (1, x),
    )


def build_daily_precinct_category_counts(df: pd.DataFrame, precincts: list[str]) -> dict:
    """Compact daily precinct x crime-category incident counts.

    Shipped to the dashboard so users can manually pick arbitrary "previous"
    and "current" date ranges instead of only the automated trailing 14 days.
    Encoded as index-referenced rows (precinct index, category index, day
    offset from base_date, count) to keep the JSON payload small.
    """
    temp = df.copy()
    temp["incident_date"] = pd.to_datetime(temp["incident_date"])
    temp["precinct_key"] = temp["precinct_norm"].apply(_norm_operational_precinct)
    temp["offense_category"] = temp["offense_category"].astype(str).str.upper()
    temp = temp[temp["precinct_key"].isin(precincts)]

    if temp.empty:
        return {"precincts": [], "categories": [], "base_date": None, "rows": []}

    counts = (
        temp.groupby(["precinct_key", "offense_category", "incident_date"])
        .size()
        .reset_index(name="count")
    )

    precinct_list = sorted(counts["precinct_key"].unique().tolist())
    category_list = sorted(counts["offense_category"].unique().tolist())
    precinct_idx = {p: i for i, p in enumerate(precinct_list)}
    category_idx = {c: i for i, c in enumerate(category_list)}
    base_date = counts["incident_date"].min()

    rows = [
        [
            precinct_idx[r.precinct_key],
            category_idx[r.offense_category],
            int((r.incident_date - base_date).days),
            int(r.count),
        ]
        for r in counts.itertuples(index=False)
    ]

    return {
        "precincts": precinct_list,
        "categories": category_list,
        "base_date": base_date.strftime("%Y-%m-%d"),
        "rows": rows,
    }


def build_hourly_precinct_category_counts(df: pd.DataFrame, precincts: list[str]) -> dict:
    """Compact daily precinct x crime-category x hour incident counts.

    Shipped to the dashboard so custom date ranges can recompute shift-level
    demand and timing patterns without sending individual incident records
    to the browser.

    Rows are encoded as:
    [precinct index, category index, day offset, hour, count]
    """
    temp = df.copy()
    temp["incident_date"] = pd.to_datetime(temp["incident_date"])
    temp["precinct_key"] = temp["precinct_norm"].apply(_norm_operational_precinct)
    temp["offense_category"] = temp["offense_category"].astype(str).str.upper()
    temp["incident_hour_of_day"] = pd.to_numeric(
        temp["incident_hour_of_day"], errors="coerce"
    )

    temp = temp[
        temp["precinct_key"].isin(precincts)
        & temp["incident_hour_of_day"].notna()
    ].copy()

    if temp.empty:
        return {
            "precincts": [],
            "categories": [],
            "base_date": None,
            "rows": [],
        }

    temp["incident_hour_of_day"] = temp["incident_hour_of_day"].astype(int)

    counts = (
        temp.groupby(
            [
                "precinct_key",
                "offense_category",
                "incident_date",
                "incident_hour_of_day",
            ]
        )
        .size()
        .reset_index(name="count")
    )

    precinct_list = sorted(counts["precinct_key"].unique().tolist())
    category_list = sorted(counts["offense_category"].unique().tolist())
    precinct_idx = {p: i for i, p in enumerate(precinct_list)}
    category_idx = {c: i for i, c in enumerate(category_list)}
    base_date = counts["incident_date"].min()

    rows = [
        [
            precinct_idx[r.precinct_key],
            category_idx[r.offense_category],
            int((r.incident_date - base_date).days),
            int(r.incident_hour_of_day),
            int(r.count),
        ]
        for r in counts.itertuples(index=False)
    ]

    return {
        "precincts": precinct_list,
        "categories": category_list,
        "base_date": base_date.strftime("%Y-%m-%d"),
        "rows": rows,
    }


def build_neighborhood_boundaries(df: pd.DataFrame, min_incidents: int = 10) -> dict:
    """Approximate neighborhood outlines derived from incident point clusters.

    There is no authoritative neighborhood boundary shapefile in this dataset,
    so each outline is a convex hull (or, for very small/degenerate point
    sets, a small buffered box around the centroid) built from that
    neighborhood's own incident coordinates. These are approximations for
    visual orientation only, not official city boundaries.
    """
    boundaries = {}
    for neighborhood, subset in df.groupby("neighborhood"):
        pts = subset[["latitude", "longitude"]].dropna().drop_duplicates().to_numpy()
        if len(pts) < 1:
            continue
        if len(subset) < min_incidents:
            continue
        if len(pts) >= 3:
            try:
                hull = ConvexHull(pts)
                ring = pts[hull.vertices].tolist()
                ring.append(ring[0])
                boundaries[str(neighborhood)] = [[round(lat, 6), round(lng, 6)] for lat, lng in ring]
                continue
            except QhullError:
                pass
        # Degenerate case (collinear or too few points): buffer a small box.
        lat_c, lng_c = float(pts[:, 0].mean()), float(pts[:, 1].mean())
        pad = 0.0025
        boundaries[str(neighborhood)] = [
            [round(lat_c - pad, 6), round(lng_c - pad, 6)],
            [round(lat_c - pad, 6), round(lng_c + pad, 6)],
            [round(lat_c + pad, 6), round(lng_c + pad, 6)],
            [round(lat_c + pad, 6), round(lng_c - pad, 6)],
            [round(lat_c - pad, 6), round(lng_c - pad, 6)],
        ]
    return boundaries


def build_area_crime_points(df: pd.DataFrame) -> dict:
    """Point-level export for the precinct-scoped additive map builder.

    The browser can display the full precinct first, optionally narrow to a
    neighborhood, then filter by dates, crime type, and crime description.
    Each exported incident also carries its best available location label and
    occurred time so map popups can show operational detail.
    """
    required = [
        "latitude", "longitude", "neighborhood", "offense_category",
        "offense_description", "incident_date", "incident_occurred_at",
    ]
    temp = df.dropna(subset=required).copy()
    temp["precinct_key"] = temp["precinct_norm"].apply(_norm_operational_precinct)
    valid_precincts = set(_operational_precinct_values(temp))
    temp = temp[temp["precinct_key"].isin(valid_precincts)].copy()
    temp["neighborhood"] = temp["neighborhood"].astype(str).str.strip()
    temp["offense_category"] = temp["offense_category"].astype(str).str.upper().str.strip()
    temp["offense_description"] = temp["offense_description"].astype(str).str.strip()
    temp["incident_date"] = pd.to_datetime(temp["incident_date"])

    # Use a true address field when one exists. Otherwise fall back to the
    # RMS nearest-intersection field. If neither is available, coordinates
    # remain the final transparent fallback.
    address_candidates = [
        "incident_address", "street_address", "address",
        "location_address", "block_address",
    ]
    address_col = next((c for c in address_candidates if c in temp.columns), None)
    if address_col:
        address_text = temp[address_col].fillna("").astype(str).str.strip()
    else:
        address_text = pd.Series("", index=temp.index, dtype="object")

    if "nearest_intersection" in temp.columns:
        intersection_text = temp["nearest_intersection"].fillna("").astype(str).str.strip()
    else:
        intersection_text = pd.Series("", index=temp.index, dtype="object")

    bad = {"", "UNKNOWN", "NAN", "NONE", "<NA>"}
    location_labels = []
    for idx, row in temp.iterrows():
        a = str(address_text.loc[idx]).strip()
        x = str(intersection_text.loc[idx]).strip()
        if a.upper() not in bad:
            location_labels.append(a)
        elif x.upper() not in bad:
            location_labels.append(x)
        else:
            location_labels.append(f"{float(row['latitude']):.5f}, {float(row['longitude']):.5f}")
    temp["map_location"] = location_labels

    # Keep the dashboard's existing date-filter semantics, but expose a
    # readable incident date/time in the popup.
    occurred = pd.to_datetime(temp["incident_occurred_at"], errors="coerce", utc=True)
    temp["map_incident_date"] = occurred.dt.strftime("%Y-%m-%d").fillna("")
    temp["map_incident_time"] = occurred.dt.strftime("%I:%M %p").fillna("Unknown time")

    precinct_list = sorted(
        temp["precinct_key"].unique().tolist(),
        key=lambda x: (0, int(x)) if str(x).isdigit() else (1, str(x)),
    )
    neighborhood_list = sorted(temp["neighborhood"].unique().tolist())
    category_list = sorted(temp["offense_category"].unique().tolist())
    description_list = sorted(temp["offense_description"].unique().tolist())
    location_list = sorted(temp["map_location"].unique().tolist())
    incident_date_list = sorted(temp["map_incident_date"].unique().tolist())
    incident_time_list = sorted(temp["map_incident_time"].unique().tolist())

    precinct_idx = {p: i for i, p in enumerate(precinct_list)}
    neighborhood_idx = {n: i for i, n in enumerate(neighborhood_list)}
    category_idx = {c: i for i, c in enumerate(category_list)}
    description_idx = {d: i for i, d in enumerate(description_list)}
    location_idx = {v: i for i, v in enumerate(location_list)}
    incident_date_idx = {v: i for i, v in enumerate(incident_date_list)}
    incident_time_idx = {v: i for i, v in enumerate(incident_time_list)}
    base_date = temp["incident_date"].min()

    # [lat, lng, precinct, neighborhood, category, description,
    #  day_offset, location, incident_date, incident_time]
    points = [
        [
            round(float(r.latitude), 6),
            round(float(r.longitude), 6),
            precinct_idx[r.precinct_key],
            neighborhood_idx[r.neighborhood],
            category_idx[r.offense_category],
            description_idx[r.offense_description],
            int((r.incident_date - base_date).days),
            location_idx[r.map_location],
            incident_date_idx[r.map_incident_date],
            incident_time_idx[r.map_incident_time],
        ]
        for r in temp.itertuples(index=False)
    ]

    neighborhoods_by_precinct = {
        str(p): sorted(g["neighborhood"].dropna().astype(str).unique().tolist())
        for p, g in temp.groupby("precinct_key", sort=True)
    }
    descriptions_by_category = {
        str(c): sorted(g["offense_description"].dropna().astype(str).unique().tolist())
        for c, g in temp.groupby("offense_category", sort=True)
    }

    neighborhood_boundaries = build_neighborhood_boundaries(temp)
    precinct_boundaries = {}
    for precinct, grp in temp.groupby("precinct_key", sort=True):
        coords = grp[["latitude", "longitude"]].dropna()
        if len(coords) < 3:
            continue
        lat_min, lat_max = float(coords["latitude"].min()), float(coords["latitude"].max())
        lng_min, lng_max = float(coords["longitude"].min()), float(coords["longitude"].max())
        lat_pad = max((lat_max - lat_min) * 0.03, 0.001)
        lng_pad = max((lng_max - lng_min) * 0.03, 0.001)
        precinct_boundaries[str(precinct)] = [
            [round(lat_min - lat_pad, 6), round(lng_min - lng_pad, 6)],
            [round(lat_min - lat_pad, 6), round(lng_max + lng_pad, 6)],
            [round(lat_max + lat_pad, 6), round(lng_max + lng_pad, 6)],
            [round(lat_max + lat_pad, 6), round(lng_min - lng_pad, 6)],
            [round(lat_min - lat_pad, 6), round(lng_min - lng_pad, 6)],
        ]

    return {
        "precincts": precinct_list,
        "neighborhoods": neighborhood_list,
        "neighborhoods_by_precinct": neighborhoods_by_precinct,
        "categories": category_list,
        "descriptions": description_list,
        "descriptions_by_category": descriptions_by_category,
        "locations": location_list,
        "incident_dates": incident_date_list,
        "incident_times": incident_time_list,
        "neighborhood_boundaries": neighborhood_boundaries,
        "precinct_boundaries": precinct_boundaries,
        "base_date": base_date.strftime("%Y-%m-%d") if pd.notna(base_date) else None,
        "min_date": temp["incident_date"].min().strftime("%Y-%m-%d") if not temp.empty else None,
        "max_date": temp["incident_date"].max().strftime("%Y-%m-%d") if not temp.empty else None,
        "points": points,
    }

CRIME_ICON_SPECS = {
    "ASSAULT": {"symbol": "✹", "color": "#b91c1c", "shape": "circle"},
    "AGGRAVATED ASSAULT": {"symbol": "⚠", "color": "#7f1d1d", "shape": "diamond"},
    "HOMICIDE": {"symbol": "◎", "color": "#450a0a", "shape": "square"},
    "ROBBERY": {"symbol": "◆", "color": "#9a3412", "shape": "hex"},
    "SEXUAL ASSAULT": {"symbol": "!", "color": "#831843", "shape": "triangle"},
    "SEX OFFENSES": {"symbol": "!", "color": "#9d174d", "shape": "triangle"},
    "LARCENY": {"symbol": "▣", "color": "#1d4ed8", "shape": "circle"},
    "BURGLARY": {"symbol": "⌂", "color": "#0369a1", "shape": "square"},
    "STOLEN VEHICLE": {"symbol": "🚗", "color": "#0e7490", "shape": "diamond"},
    "STOLEN PROPERTY": {"symbol": "▤", "color": "#0f766e", "shape": "hex"},
    "DAMAGE TO PROPERTY": {"symbol": "✕", "color": "#a16207", "shape": "square"},
    "ARSON": {"symbol": "♨", "color": "#c2410c", "shape": "triangle"},
    "WEAPONS OFFENSES": {"symbol": "⚠", "color": "#78350f", "shape": "diamond"},
    "KIDNAPPING": {"symbol": "!", "color": "#581c87", "shape": "hex"},
    "FRAUD": {"symbol": "$", "color": "#4338ca", "shape": "circle"},
    "FORGERY": {"symbol": "✎", "color": "#4338ca", "shape": "square"},
    "EMBEZZLEMENT": {"symbol": "$", "color": "#4338ca", "shape": "diamond"},
    "BRIBERY": {"symbol": "$", "color": "#4338ca", "shape": "hex"},
    "DANGEROUS DRUGS": {"symbol": "✚", "color": "#166534", "shape": "circle"},
    "OUIL": {"symbol": "🚗", "color": "#854d0e", "shape": "diamond"},
    "OBSTRUCTING THE POLICE": {"symbol": "✦", "color": "#334155", "shape": "square"},
    "OBSTRUCTING JUDICIARY": {"symbol": "§", "color": "#334155", "shape": "hex"},
    "FAMILY OFFENSE": {"symbol": "!", "color": "#7c2d12", "shape": "circle"},
    "INVASION OF PRIVACY -OTHER": {"symbol": "◉", "color": "#6d28d9", "shape": "diamond"},
    "DISORDERLY CONDUCT": {"symbol": "!", "color": "#57534e", "shape": "square"},
    "RUNAWAY": {"symbol": "➜", "color": "#57534e", "shape": "circle"},
    "EXTORTION": {"symbol": "$", "color": "#4338ca", "shape": "hex"},
    "HEALTH AND SAFETY": {"symbol": "✚", "color": "#065f46", "shape": "triangle"},
    "LIQUOR": {"symbol": "◇", "color": "#854d0e", "shape": "circle"},
}
CRIME_ICON_DEFAULT = {"symbol": "•", "color": "#334155", "shape": "circle"}


def save_area_crime_map_builder_html(area_crime_data: dict, out_path: Path) -> None:
    """Dedicated precinct-scoped custom GIS workspace."""
    icon_specs = {
        cat: CRIME_ICON_SPECS.get(cat, CRIME_ICON_DEFAULT)
        for cat in area_crime_data["categories"]
    }
    payload = {**area_crime_data, "icon_specs": icon_specs}
    payload_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")

    html_doc = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Detroit Crime Analysis — Custom Map Builder</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.css" />
<link rel="stylesheet" href="https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.css" />
<link rel="stylesheet" href="https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.Default.css" />
<script src="https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.js"></script>
<script src="https://unpkg.com/leaflet.markercluster@1.5.3/dist/leaflet.markercluster.js"></script>
<script src="https://unpkg.com/leaflet.heat@0.2.0/dist/leaflet-heat.js"></script>
<style>
:root{--navy:#0b2d50;--blue:#0b5cab;--blue2:#1477c9;--ink:#0f172a;--muted:#64748b;--line:#dbe3ef;--soft:#f4f7fb;--card:#fff}
*{box-sizing:border-box} body{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif;background:var(--soft);color:var(--ink)}
.topbar{height:62px;background:linear-gradient(90deg,#082b4f,#0b3c6d);color:#fff;display:flex;align-items:center;justify-content:space-between;padding:0 22px;box-shadow:0 2px 8px rgba(15,23,42,.2)}
.brand{display:flex;align-items:center;gap:11px;font-weight:900}.brand-badge{width:35px;height:35px;border-radius:9px;background:rgba(255,255,255,.14);display:flex;align-items:center;justify-content:center;font-size:19px}
.top-actions{display:flex;gap:9px;align-items:center}.top-actions a{color:#fff;text-decoration:none;font-weight:760;font-size:.86rem;padding:8px 10px;border-radius:7px}.top-actions a:hover{background:rgba(255,255,255,.1)}
.workspace{display:grid;grid-template-columns:360px 1fr;min-height:calc(100vh - 62px)}
.sidebar{background:#fff;border-right:1px solid var(--line);padding:18px 16px;overflow:auto;max-height:calc(100vh - 62px)}
.main{padding:16px;min-width:0}
.section-title{font-size:.78rem;text-transform:uppercase;letter-spacing:.04em;color:var(--blue);font-weight:900;margin-bottom:8px}
.scope{display:inline-flex;align-items:center;gap:7px;padding:7px 10px;background:#e8f2ff;color:#16477a;border:1px solid #bfdbfe;border-radius:999px;font-weight:900;font-size:.82rem;margin-bottom:14px}
.field{margin-bottom:11px}.field label{display:block;font-size:.77rem;font-weight:820;color:#334155;margin-bottom:5px}
select,input,button{font:inherit} select,input[type=date]{width:100%;padding:9px 10px;border:1px solid #aebdce;border-radius:8px;background:#fff;font-weight:700;color:#172033}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.mode-row{display:grid;grid-template-columns:repeat(3,1fr);gap:7px}.mode-btn{border:1px solid #aebdce;border-radius:8px;background:#fff;padding:9px 5px;font-weight:800;font-size:.76rem;cursor:pointer;color:#334155}.mode-btn.active{background:#e8f2ff;color:#0b5cab;border-color:#7fb0df}
.primary{width:100%;background:var(--blue);color:#fff;border:0;border-radius:8px;padding:11px 12px;font-weight:900;cursor:pointer;margin-top:3px}.secondary{width:100%;background:#fff;color:#334155;border:1px solid #aebdce;border-radius:8px;padding:10px 12px;font-weight:850;cursor:pointer;margin-top:7px}
.panel{border:1px solid var(--line);border-radius:11px;background:#fff;padding:12px;margin-top:14px}.panel h3{font-size:.9rem;margin:0 0 8px}.status{font-size:.8rem;color:var(--muted);line-height:1.45}
.layer-list{display:flex;flex-direction:column;gap:7px}.layer-card{border:1px solid #e2e8f0;border-radius:9px;padding:9px;background:#fafcff}.layer-top{display:flex;align-items:center;gap:8px}.layer-name{flex:1;font-size:.78rem;font-weight:850}.layer-actions{display:flex;gap:5px;margin-top:7px}.layer-actions button{border:1px solid #cbd5e1;background:#fff;border-radius:6px;padding:5px 7px;font-size:.7rem;font-weight:800;cursor:pointer}.layer-actions .remove{color:#991b1b;background:#fff5f5;border-color:#fecaca}
.icon-preview{width:28px;height:28px;border-radius:8px;display:flex;align-items:center;justify-content:center;color:#fff;font-weight:900;box-shadow:0 1px 4px rgba(0,0,0,.25)}
.map-shell{position:relative} #map{height:calc(100vh - 118px);min-height:650px;border:1px solid var(--line);border-radius:13px;box-shadow:0 3px 12px rgba(15,23,42,.08)}
.map-tools{position:absolute;z-index:600;right:12px;top:12px;background:#fff;border:1px solid var(--line);border-radius:9px;padding:8px;box-shadow:0 2px 8px rgba(15,23,42,.15);display:flex;gap:6px}.map-tools button{background:#fff;border:1px solid #d3dce7;border-radius:6px;padding:6px 8px;font-size:.72rem;font-weight:800;cursor:pointer}
.legend{position:absolute;z-index:600;right:12px;bottom:12px;max-width:280px;background:#fff;border:1px solid var(--line);border-radius:10px;padding:10px;box-shadow:0 2px 8px rgba(15,23,42,.15);font-size:.77rem}.legend-items{display:flex;flex-direction:column;gap:5px;margin-top:7px}.legend-item{display:flex;align-items:center;gap:7px}
.marker-wrap{background:transparent!important;border:0!important}.marker-symbol{width:32px;height:32px;border-radius:50%;display:flex;align-items:center;justify-content:center;color:#fff;font-size:16px;border:2px solid #fff;box-shadow:0 1px 5px rgba(0,0,0,.55);position:relative}.marker-count{position:absolute;right:-8px;top:-9px;min-width:18px;height:18px;border-radius:999px;background:#fff;color:#0f172a;border:2px solid currentColor;display:flex;align-items:center;justify-content:center;font-size:9px;font-weight:900;padding:0 3px}
.modal-backdrop{display:none;position:fixed;z-index:3000;inset:0;background:rgba(15,23,42,.44);align-items:center;justify-content:center;padding:18px}.modal-backdrop.open{display:flex}.modal{background:#fff;width:min(920px,96vw);max-height:88vh;overflow:auto;border-radius:15px;border:1px solid #d7e0ea;box-shadow:0 24px 70px rgba(15,23,42,.3);padding:18px}
.modal-head{display:flex;justify-content:space-between;gap:12px;align-items:flex-start}.modal-head h2{font-size:1.18rem;margin:0}.close-btn{border:0;background:#f1f5f9;border-radius:7px;padding:6px 9px;cursor:pointer;font-weight:900}
.custom-grid{display:grid;grid-template-columns:1fr 250px;gap:16px;margin-top:14px}.icon-library{display:grid;grid-template-columns:repeat(7,1fr);gap:8px}.icon-choice{height:48px;border:1px solid #d8e0ea;border-radius:9px;background:#fff;display:flex;align-items:center;justify-content:center;font-size:22px;cursor:pointer}.icon-choice.selected{border:2px solid var(--blue);background:#eef6ff}
.color-grid{display:grid;grid-template-columns:repeat(6,34px);gap:8px;margin-top:8px}.color-choice{width:34px;height:34px;border-radius:50%;border:3px solid #fff;box-shadow:0 0 0 1px #cbd5e1;cursor:pointer}.color-choice.selected{box-shadow:0 0 0 3px #0b5cab}
.custom-preview{border:1px solid #e2e8f0;border-radius:12px;padding:15px;background:#fafcff;text-align:center}.big-preview{width:64px;height:64px;border-radius:50%;margin:8px auto 15px;display:flex;align-items:center;justify-content:center;color:#fff;font-size:30px;border:3px solid #fff;box-shadow:0 2px 10px rgba(15,23,42,.25)}
.apply-style{width:100%;background:var(--blue);color:#fff;border:0;border-radius:8px;padding:10px;font-weight:900;cursor:pointer;margin-top:12px}
@media(max-width:980px){.workspace{grid-template-columns:1fr}.sidebar{max-height:none;border-right:0;border-bottom:1px solid var(--line)}#map{height:650px}.custom-grid{grid-template-columns:1fr}.icon-library{grid-template-columns:repeat(6,1fr)}} 
@media(max-width:560px){.top-actions{display:none}.workspace{display:block}.main{padding:8px}.grid2{grid-template-columns:1fr}.icon-library{grid-template-columns:repeat(5,1fr)}#map{height:560px;min-height:560px}}
</style>
</head>
<body>
<div class="topbar">
 <div class="brand"><div class="brand-badge">🛡</div><div>Detroit Crime Analysis <span style="font-weight:500;opacity:.78">/ Custom Map Builder</span></div></div>
 <div class="top-actions"><a href="../index.html">← Dashboard</a><a href="#" id="fitTop">Fit Precinct</a></div>
</div>

<div class="workspace">
 <aside class="sidebar">
  <div class="section-title">Build Your Map</div>
  <div class="scope" id="precinctScope">Precinct not selected</div>

  <div class="field"><label>Neighborhood (optional)</label><select id="areaSelect"><option value="">All neighborhoods in selected precinct</option></select></div>
  <div class="grid2">
   <div class="field"><label>Start date</label><input type="date" id="startDate"></div>
   <div class="field"><label>End date</label><input type="date" id="endDate"></div>
  </div>
  <div class="field"><label>Crime type</label><select id="crimeSelect"><option value="">Select a crime type…</option></select></div>
  <div class="field"><label>Crime description (optional)</label><select id="descriptionSelect" disabled><option value="">Select a crime type first…</option></select></div>

  <div class="field"><label>Visualization style</label>
   <div class="mode-row">
    <button type="button" class="mode-btn active" data-mode="markers">Markers</button>
    <button type="button" class="mode-btn" data-mode="clusters">Clusters</button>
    <button type="button" class="mode-btn" data-mode="heatmap">Heat map</button>
   </div>
  </div>

  <button class="primary" id="addBtn">＋ Add layer to map</button>
  <button class="secondary" id="resetBtn">Reset all layers</button>

  <div class="panel"><h3>Status</h3><div class="status" id="status">Choose a precinct from the dashboard to begin.</div></div>
  <div class="panel"><h3>Visible Layers</h3><div class="layer-list" id="layerList"><div class="status">No layers added yet.</div></div></div>
 </aside>

 <main class="main">
  <div class="map-shell">
   <div class="map-tools"><button id="fitBtn">Fit scope</button><button id="clearOutlineBtn">Hide outline</button></div>
   <div id="map"></div>
   <div class="legend"><b>Legend</b><div class="legend-items" id="legendItems"><span class="status">Add a crime layer to populate the legend.</span></div></div>
  </div>
 </main>
</div>

<div class="modal-backdrop" id="styleModal">
 <div class="modal">
  <div class="modal-head"><div><div class="section-title">Layer Styling</div><h2 id="styleTitle">Customize crime layer</h2><div class="status">Choose an icon and color for this crime layer.</div></div><button class="close-btn" id="closeStyle">✕</button></div>
  <div class="custom-grid">
   <div>
    <div style="font-weight:850;margin-bottom:8px;">Icon library</div>
    <div class="icon-library" id="iconLibrary"></div>
   </div>
   <div class="custom-preview">
    <div style="font-weight:850">Preview</div>
    <div class="big-preview" id="bigPreview">●</div>
    <div style="font-weight:850;margin-top:4px;">Color</div>
    <div class="color-grid" id="colorGrid"></div>
    <button class="apply-style" id="applyStyleBtn">Apply style</button>
   </div>
  </div>
 </div>
</div>

<script>
const DATA=__PAYLOAD__;
const precincts=DATA.precincts||[], neighborhoods=DATA.neighborhoods||[], neighborhoodsByPrecinct=DATA.neighborhoods_by_precinct||{};
const categories=DATA.categories||[], descriptions=DATA.descriptions||[], descriptionsByCategory=DATA.descriptions_by_category||{};
const locations=DATA.locations||[], incidentDates=DATA.incident_dates||[], incidentTimes=DATA.incident_times||[];
const neighborhoodBoundaries=DATA.neighborhood_boundaries||{}, precinctBoundaries=DATA.precinct_boundaries||{}, points=DATA.points||[];
const iconSpecs=DATA.icon_specs||{}, baseDate=DATA.base_date?new Date(DATA.base_date+'T00:00:00Z'):null;

const ICON_LIBRARY=['🚗','🏠','💰','🎒','📦','⚠️','🛡️','🚨','🎯','✹','◆','●','▲','■','★','✚','✕','🔔','👁️','📍','🔑','🚪','🔒','💥','🔥','🧰','🧱','🧾','💳','🔎','⚡','⬢','⬟','◉','◎'];
const COLORS=['#b91c1c','#dc2626','#ea580c','#f59e0b','#65a30d','#15803d','#0f766e','#0891b2','#2563eb','#1d4ed8','#4f46e5','#7c3aed','#9333ea','#c026d3','#db2777','#475569','#111827','#6b7280'];

const areaSelect=document.getElementById('areaSelect'),crimeSelect=document.getElementById('crimeSelect'),descriptionSelect=document.getElementById('descriptionSelect');
const startDate=document.getElementById('startDate'),endDate=document.getElementById('endDate'),statusEl=document.getElementById('status'),layerList=document.getElementById('layerList'),legendItems=document.getElementById('legendItems');
const precinctScope=document.getElementById('precinctScope');
let selectedMode='markers', precinctOutline=null, areaOutline=null, showOutline=true, editingKey=null, draftIcon='●', draftColor='#0b5cab';
const crimeLayers={};

startDate.min=DATA.min_date||'';startDate.max=DATA.max_date||'';endDate.min=DATA.min_date||'';endDate.max=DATA.max_date||'';endDate.value=DATA.max_date||'';
if(DATA.max_date){const d=new Date(DATA.max_date+'T00:00:00Z');d.setUTCDate(d.getUTCDate()-13);startDate.value=d.toISOString().slice(0,10);}
categories.forEach(c=>{const o=document.createElement('option');o.value=c;o.textContent=c;crimeSelect.appendChild(o);});

function populateDescriptions(category){
 descriptionSelect.innerHTML='';
 if(!category){descriptionSelect.disabled=true;descriptionSelect.innerHTML='<option value="">Select a crime type first…</option>';return;}
 descriptionSelect.disabled=false;const all=document.createElement('option');all.value='';all.textContent='All '+category+' descriptions';descriptionSelect.appendChild(all);
 (descriptionsByCategory[category]||[]).forEach(d=>{const o=document.createElement('option');o.value=d;o.textContent=d;descriptionSelect.appendChild(o);});
}
crimeSelect.addEventListener('change',()=>populateDescriptions(crimeSelect.value));

const qs=new URLSearchParams(window.location.search),selectedPrecinct=(qs.get('precinct')||'').trim().toUpperCase(),precinctIndex=precincts.indexOf(selectedPrecinct);
const map=L.map('map').setView([42.3468,-83.0700],11);
L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:19,attribution:'&copy; OpenStreetMap contributors'}).addTo(map);

function dateOffset(s){if(!baseDate||!s)return null;const d=new Date(s+'T00:00:00Z');return Math.round((d-baseDate)/86400000);}
function populateNeighborhoods(){areaSelect.innerHTML='<option value="">All neighborhoods in Precinct '+selectedPrecinct+'</option>';(neighborhoodsByPrecinct[selectedPrecinct]||[]).forEach(n=>{const o=document.createElement('option');o.value=n;o.textContent=n;areaSelect.appendChild(o);});}
function clearAreaOutline(){if(areaOutline){map.removeLayer(areaOutline);areaOutline=null;}}
function clearPrecinctOutline(){if(precinctOutline){map.removeLayer(precinctOutline);precinctOutline=null;}}
function drawPrecinct(){clearPrecinctOutline();const ring=precinctBoundaries[selectedPrecinct];if(!ring)return;precinctOutline=L.polygon(ring,{color:'#1e3a8a',weight:3,dashArray:'8 5',fillColor:'#3b82f6',fillOpacity:.035}).addTo(map);map.fitBounds(precinctOutline.getBounds(),{padding:[24,24]});}
function drawArea(name){clearAreaOutline();const ring=neighborhoodBoundaries[name];if(!ring)return;areaOutline=L.polygon(ring,{color:'#0b5cab',weight:2,fillColor:'#60a5fa',fillOpacity:.08}).addTo(map);map.fitBounds(areaOutline.getBounds(),{padding:[24,24]});}
areaSelect.addEventListener('change',()=>{if(areaSelect.value)drawArea(areaSelect.value);else{clearAreaOutline();if(precinctOutline)map.fitBounds(precinctOutline.getBounds(),{padding:[24,24]});}});

document.querySelectorAll('.mode-btn').forEach(b=>b.addEventListener('click',()=>{{
 document.querySelectorAll('.mode-btn').forEach(x=>x.classList.remove('active'));
 b.classList.add('active');
 selectedMode=b.dataset.mode;
 const label=selectedMode==='heatmap'?'Heat map':selectedMode==='clusters'?'Clusters':'Markers';
 statusEl.textContent=label+' selected for the next layer. You can also change the display style of any existing layer below.';
}}));

function styleFor(category){return iconSpecs[category]||{symbol:'●',color:'#334155'};}
function markerIcon(symbol,color,count){const badge=count>1?'<span class="marker-count" style="color:'+color+'">'+count+'</span>':'';return L.divIcon({className:'marker-wrap',html:'<div class="marker-symbol" style="background:'+color+'"><span>'+symbol+'</span>'+badge+'</div>',iconSize:[32,32],iconAnchor:[16,16]});}

function matchingPoints(area,category,description,startStr,endStr){
 const s=dateOffset(startStr),e=dateOffset(endStr),areaIdx=area?neighborhoods.indexOf(area):-1,catIdx=categories.indexOf(category),descIdx=description?descriptions.indexOf(description):-1;
 return points.filter(p=>p[2]===precinctIndex&&p[4]===catIdx&&(areaIdx===-1||p[3]===areaIdx)&&(descIdx===-1||p[5]===descIdx)&&p[6]>=s&&p[6]<=e);
}

function popupHtml(rows,category,scopeLabel){
 const first=rows[0],location=locations[first[7]]||'Location unavailable';
 const items=rows.slice(0,12).map(p=>'<div style="margin:6px 0;padding-top:5px;border-top:1px solid #e5e7eb"><b>'+(incidentDates[p[8]]||'Date unavailable')+' · '+(incidentTimes[p[9]]||'Time unavailable')+'</b><br>'+(descriptions[p[5]]||'Description unavailable')+'</div>').join('');
 const more=rows.length>12?'<div style="margin-top:6px;color:#64748b">+'+(rows.length-12)+' more matching incidents</div>':'';
 return '<div style="min-width:240px"><b style="font-size:1.05rem">'+category+'</b><div style="margin-top:6px"><b>Location:</b> '+location+'</div><div><b>Neighborhood:</b> '+neighborhoods[first[3]]+'</div><div><b>Matching incidents here:</b> '+rows.length+'</div><div><b>Layer scope:</b> '+scopeLabel+'</div><div style="margin-top:8px;font-weight:850">Incident date / time</div>'+items+more+'</div>';
}

function buildLayer(entry){
 const selected=matchingPoints(entry.area,entry.category,entry.description,entry.startStr,entry.endStr);
 if(!selected.length)return null;

 let group;

 if(entry.mode==='heatmap'){
  if(typeof L.heatLayer==='function'){
   group=L.heatLayer(selected.map(p=>[p[0],p[1],1]),{radius:26,blur:20,maxZoom:17,minOpacity:.32});
   group.addTo(map);
  }else{
   // Safe fallback if the heat plugin fails to load: show translucent density circles.
   group=L.layerGroup();
   selected.forEach(p=>L.circleMarker([p[0],p[1]],{
    radius:11,stroke:false,fillColor:entry.color,fillOpacity:.18
   }).addTo(group));
   group.addTo(map);
   statusEl.textContent='Heat-map plugin was unavailable, so a density-circle fallback is being shown.';
  }
 }else{
  const canCluster=entry.mode==='clusters'&&typeof L.markerClusterGroup==='function';
  group=canCluster
    ?L.markerClusterGroup({showCoverageOnHover:false,spiderfyOnMaxZoom:true,maxClusterRadius:48})
    :L.layerGroup();

  const byLoc={};
  selected.forEach(p=>{
   const loc=locations[p[7]]||'';
   const k=loc+'||'+p[0].toFixed(5)+'||'+p[1].toFixed(5);
   (byLoc[k]||(byLoc[k]=[])).push(p);
  });

  Object.values(byLoc).forEach(rows=>{
   rows.sort((a,b)=>String(incidentDates[a[8]]||'').localeCompare(String(incidentDates[b[8]]||'')));
   const f=rows[0];
   L.marker([f[0],f[1]],{icon:markerIcon(entry.icon,entry.color,rows.length)})
    .bindPopup(popupHtml(rows,entry.category,entry.scopeLabel),{maxWidth:370})
    .addTo(group);
  });
  group.addTo(map);

  if(entry.mode==='clusters'&&!canCluster){
   statusEl.textContent='Cluster plugin was unavailable, so standard location markers are being shown.';
  }
 }

 entry.group=group;
 entry.count=selected.length;
 return entry;
}

function addLayer(){
 const area=areaSelect.value,category=crimeSelect.value,description=descriptionSelect.value,startStr=startDate.value,endStr=endDate.value;
 if(precinctIndex<0){statusEl.textContent='Return to the dashboard and choose a precinct first.';return;}
 if(!category||!startStr||!endStr){statusEl.textContent='Choose dates and a crime type.';return;} if(startStr>endStr){statusEl.textContent='Start date must be on or before end date.';return;}
 const pts=matchingPoints(area,category,description,startStr,endStr); if(!pts.length){statusEl.textContent='No matching '+category+' incidents were found. No empty layer was added.';return;}
 const spec=styleFor(category),scopeLabel=area||('Precinct '+selectedPrecinct),key=[selectedPrecinct,area||'ALL',startStr,endStr,category,description||'ALL',selectedMode,Date.now()].join('||');
 const entry={key,area,category,description,descLabel:description||('All '+category+' descriptions'),startStr,endStr,mode:selectedMode,modeLabel:selectedMode==='heatmap'?'Heat map':selectedMode==='clusters'?'Clusters':'Markers',scopeLabel,icon:spec.symbol||'●',color:spec.color||'#0b5cab',group:null,count:0};
 const built=buildLayer(entry);
 if(!built){statusEl.textContent='The layer could not be built for this selection.';return;}
 crimeLayers[key]=built;
 statusEl.textContent='Added '+entry.category+' — '+entry.scopeLabel+' — '+entry.modeLabel+'. You can switch this layer between Markers, Clusters, and Heat below.';
 renderLayers();renderLegend();
}
document.getElementById('addBtn').addEventListener('click',addLayer);

function removeLayer(key){const e=crimeLayers[key];if(!e)return;if(e.group)map.removeLayer(e.group);delete crimeLayers[key];renderLayers();renderLegend();}
function toggleLayer(key){const e=crimeLayers[key];if(!e)return;if(map.hasLayer(e.group))map.removeLayer(e.group);else e.group.addTo(map);renderLayers();}
function rebuildLayer(key){
 const e=crimeLayers[key];if(!e)return;
 if(e.group&&map.hasLayer(e.group))map.removeLayer(e.group);
 buildLayer(e);renderLayers();renderLegend();
}
function setLayerMode(key,mode){
 const e=crimeLayers[key];if(!e)return;
 e.mode=mode;
 e.modeLabel=mode==='heatmap'?'Heat map':mode==='clusters'?'Clusters':'Markers';
 rebuildLayer(key);
 statusEl.textContent=e.category+' is now displayed as '+e.modeLabel+'.';
}

function renderLayers(){
 const keys=Object.keys(crimeLayers);if(!keys.length){layerList.innerHTML='<div class="status">No layers added yet.</div>';return;}
 layerList.innerHTML='';keys.forEach(key=>{const e=crimeLayers[key],card=document.createElement('div');card.className='layer-card';
  card.innerHTML='<div class="layer-top"><div class="icon-preview" style="background:'+e.color+'">'+e.icon+'</div><div class="layer-name">'+e.category+'<div class="status">'+e.scopeLabel+' · '+e.modeLabel+' · '+e.count+'</div></div></div>';
  const actions=document.createElement('div');actions.className='layer-actions';
  const toggle=document.createElement('button');toggle.type='button';toggle.textContent=map.hasLayer(e.group)?'Hide':'Show';toggle.onclick=()=>toggleLayer(key);
  const markers=document.createElement('button');markers.type='button';markers.textContent='Markers';markers.onclick=()=>setLayerMode(key,'markers');
  const clusters=document.createElement('button');clusters.type='button';clusters.textContent='Clusters';clusters.onclick=()=>setLayerMode(key,'clusters');
  const heat=document.createElement('button');heat.type='button';heat.textContent='Heat';heat.onclick=()=>setLayerMode(key,'heatmap');
  const style=document.createElement('button');style.type='button';style.textContent='Customize';style.onclick=()=>openStyle(key);
  const rm=document.createElement('button');rm.type='button';rm.className='remove';rm.textContent='Remove';rm.onclick=()=>removeLayer(key);
  actions.append(toggle,markers,clusters,heat,style,rm);card.appendChild(actions);layerList.appendChild(card);
 });
}
function renderLegend(){const vals=Object.values(crimeLayers);if(!vals.length){legendItems.innerHTML='<span class="status">Add a crime layer to populate the legend.</span>';return;}legendItems.innerHTML=vals.map(e=>'<div class="legend-item"><span class="icon-preview" style="width:23px;height:23px;background:'+e.color+'">'+e.icon+'</span><span>'+e.category+' · '+e.scopeLabel+'</span></div>').join('');}

const modal=document.getElementById('styleModal'),iconLibrary=document.getElementById('iconLibrary'),colorGrid=document.getElementById('colorGrid'),bigPreview=document.getElementById('bigPreview');
function renderStyleChoices(){
 iconLibrary.innerHTML=ICON_LIBRARY.map(i=>'<button class="icon-choice'+(i===draftIcon?' selected':'')+'" data-icon="'+i+'">'+i+'</button>').join('');
 colorGrid.innerHTML=COLORS.map(c=>'<button class="color-choice'+(c===draftColor?' selected':'')+'" data-color="'+c+'" style="background:'+c+'"></button>').join('');
 bigPreview.textContent=draftIcon;bigPreview.style.background=draftColor;
 iconLibrary.querySelectorAll('[data-icon]').forEach(b=>b.onclick=()=>{draftIcon=b.dataset.icon;renderStyleChoices();});
 colorGrid.querySelectorAll('[data-color]').forEach(b=>b.onclick=()=>{draftColor=b.dataset.color;renderStyleChoices();});
}
function openStyle(key){const e=crimeLayers[key];if(!e)return;editingKey=key;draftIcon=e.icon;draftColor=e.color;document.getElementById('styleTitle').textContent='Customize '+e.category;renderStyleChoices();modal.classList.add('open');}
function closeStyle(){modal.classList.remove('open');editingKey=null;}
document.getElementById('closeStyle').onclick=closeStyle;modal.addEventListener('click',e=>{if(e.target===modal)closeStyle();});
document.getElementById('applyStyleBtn').onclick=()=>{if(!editingKey)return;crimeLayers[editingKey].icon=draftIcon;crimeLayers[editingKey].color=draftColor;rebuildLayer(editingKey);closeStyle();};

document.getElementById('resetBtn').onclick=()=>{Object.keys(crimeLayers).forEach(removeLayer);crimeSelect.value='';populateDescriptions('');areaSelect.value='';selectedMode='markers';document.querySelectorAll('.mode-btn').forEach(b=>b.classList.toggle('active',b.dataset.mode==='markers'));statusEl.textContent='All layers reset. Precinct '+selectedPrecinct+' remains selected.';if(precinctOutline)map.fitBounds(precinctOutline.getBounds(),{padding:[24,24]});};
document.getElementById('fitBtn').onclick=()=>{if(areaOutline)map.fitBounds(areaOutline.getBounds(),{padding:[24,24]});else if(precinctOutline)map.fitBounds(precinctOutline.getBounds(),{padding:[24,24]});};
document.getElementById('fitTop').onclick=e=>{e.preventDefault();document.getElementById('fitBtn').click();};
document.getElementById('clearOutlineBtn').onclick=()=>{showOutline=!showOutline;if(!showOutline){clearAreaOutline();clearPrecinctOutline();document.getElementById('clearOutlineBtn').textContent='Show outline';}else{drawPrecinct();if(areaSelect.value)drawArea(areaSelect.value);document.getElementById('clearOutlineBtn').textContent='Hide outline';}};

if(precinctIndex<0){precinctScope.textContent='No valid precinct selected';document.getElementById('addBtn').disabled=true;areaSelect.disabled=true;statusEl.textContent='Return to the main dashboard, select a precinct, then open Build Custom Crime Map.';}
else{precinctScope.textContent='Precinct '+selectedPrecinct+' — fixed parent scope';populateNeighborhoods();drawPrecinct();statusEl.textContent='Precinct '+selectedPrecinct+' loaded. Start with the whole precinct or narrow to a neighborhood, then build as many crime layers as needed.';}
</script>
</body>
</html>"""
    html_doc = html_doc.replace("__PAYLOAD__", payload_json)
    out_path.write_text(html_doc, encoding="utf-8")

def generate_precinct_drilldown_assets(
    df: pd.DataFrame,
    focus_locations: pd.DataFrame,
    current_year: int,
    period_tag: str,
) -> dict[str, dict[str, str]]:
    """Generate the familiar drill-down outputs separately for every precinct.

    The landing page can therefore preserve the old "blue links" concept while
    guaranteeing that each linked map/chart contains only the selected precinct.
    """
    manifest: dict[str, dict[str, str]] = {}
    precincts = _operational_precinct_values(df)

    for precinct in precincts:
        pdir = IMAGES_DIR / f"precinct_{precinct}"
        pdir.mkdir(parents=True, exist_ok=True)

        p_df = df[df["precinct_norm"].apply(_norm_operational_precinct).eq(precinct)].copy()
        if p_df.empty:
            continue

        # Recalculate weekly spikes inside the precinct so spike severity is not
        # inherited from the citywide workload.
        p_weekly = detect_weekly_spikes(p_df)

        # Re-rank existing operational focus cells within the precinct.
        p_focus = focus_locations[
            focus_locations["police_precinct"].apply(_norm_operational_precinct).eq(precinct)
        ].copy()
        if not p_focus.empty:
            p_focus = p_focus.sort_values(["score", "incident_count"], ascending=False).reset_index(drop=True)
            p_focus["focus_rank"] = np.arange(1, len(p_focus) + 1)

        focus_map = pdir / f"precinct_{precinct}_focus_locations_{period_tag}.html"
        spike_map = pdir / f"precinct_{precinct}_spike_severity_{period_tag}.html"
        shift_img = pdir / f"precinct_{precinct}_shift_summary_{period_tag}.png"
        decision_img = pdir / f"precinct_{precinct}_decision_purpose_{period_tag}.png"
        monthly_img = pdir / f"precinct_{precinct}_monthly_trend_heatmap_{period_tag}.png"
        violent_type_img = pdir / f"precinct_{precinct}_violent_crime_14d_by_type_{period_tag}.png"
        violent_day_hour_img = pdir / f"precinct_{precinct}_violent_crime_14d_by_day_hour_{period_tag}.png"

        if not p_focus.empty:
            save_focus_locations_map(p_focus, focus_map, top_n=min(40, len(p_focus)))

        if h3 is not None and not p_weekly.empty:
            save_interactive_h3_spike_severity_choropleth(p_df, p_weekly, spike_map)

        p_shift = build_shift_summary(p_df)
        if not p_shift.empty:
            save_shift_summary_chart(p_shift, shift_img)

        p_decision = build_decision_purpose_summary(p_df)
        if not p_decision.empty:
            save_decision_purpose_chart(p_decision, decision_img)

        p_monthly = build_precinct_monthly_trend(p_df)
        if not p_monthly.empty:
            save_precinct_monthly_trend_heatmap(p_monthly, monthly_img)

        p_violent_type, p_violent_day_hour = build_violent_14d_summaries(p_df)
        save_violent_14d_charts(
            p_violent_type,
            p_violent_day_hour,
            violent_type_img,
            violent_day_hour_img,
        )

        def rel(path: Path) -> str:
            # Operations overview lives in Documentation/, so all Images paths
            # are one directory up.
            return "../Images/" + path.relative_to(IMAGES_DIR).as_posix()

        manifest[precinct] = {
            "focus_map": rel(focus_map) if focus_map.exists() else "",
            "spike_map": rel(spike_map) if spike_map.exists() else "",
            "shift_chart": rel(shift_img) if shift_img.exists() else "",
            "decision_chart": rel(decision_img) if decision_img.exists() else "",
            "monthly_heatmap": rel(monthly_img) if monthly_img.exists() else "",
            "violent_type_chart": rel(violent_type_img) if violent_type_img.exists() else "",
            "violent_day_hour": rel(violent_day_hour_img) if violent_day_hour_img.exists() else "",
        }

    return manifest

def save_operations_landing_html(
    df: pd.DataFrame,
    weekly: pd.DataFrame,
    focus_locations: pd.DataFrame,
    city_ytd: pd.DataFrame,
    precinct_improvement: pd.DataFrame,
    precinct_crime_14d: pd.DataFrame,
    precinct_crime_trends: pd.DataFrame,
    priority_concerns: pd.DataFrame,
    temporal_summary: pd.DataFrame,
    hotspot_change: pd.DataFrame,
    baseline_year: int,
    precinct_assets: dict[str, dict[str, str]],
    current_year: int,
    previous_year: int,
    period_tag: str,
    map_filename: str,
    area_map_filename: str,
    out_path: Path,
) -> None:
    """Citywide landing page that becomes a full precinct evaluation after selection."""
    invalid_precinct_codes = {"00", "0W", "OW", "HP", "UNKNOWN", "NAN", "NONE", ""}

    def norm_precinct(value) -> str:
        return _norm_operational_precinct(value)

    def clean_number(value):
        if value is None or pd.isna(value):
            return None
        if isinstance(value, (np.integer, int)):
            return int(value)
        if isinstance(value, (np.floating, float)):
            return float(value)
        return value

    def records_clean(frame: pd.DataFrame, cols: list[str]) -> list[dict]:
        if frame is None or frame.empty:
            return []
        out = []
        for _, row in frame.iterrows():
            rec = {}
            for col in cols:
                value = row.get(col)
                rec[col] = clean_number(value)
            out.append(rec)
        return out

    temp = df.copy()
    temp["incident_date"] = pd.to_datetime(temp["incident_date"])
    temp["precinct_key"] = temp["precinct_norm"].apply(norm_precinct)
    years_available = sorted(temp["incident_year"].dropna().astype(int).unique().tolist())
    min_year = min(years_available) if years_available else int(current_year)
    max_year = max(years_available) if years_available else int(current_year)
    current_max = temp.loc[temp["incident_year"] == int(current_year), "incident_date"].max()
    data_through = current_max.strftime("%Y-%m-%d") if pd.notna(current_max) else "Unknown"

    # The operations landing page is a precinct-wide view. Upstream tables also
    # contain neighborhood-specific records so the interactive drill-down can
    # recalculate correctly when a neighborhood is selected. If those scoped
    # rows are mixed into this page, one crime can appear multiple times for the
    # same precinct (for example, several LARCENY rows from different
    # neighborhoods). Keep only the independently calculated ALL-neighborhood
    # records here; neighborhood-specific rows remain available to the detailed
    # interactive dashboard.
    def precinct_wide_only(frame: pd.DataFrame | None) -> pd.DataFrame:
        if frame is None or frame.empty:
            return pd.DataFrame()

        out = frame.copy()
        if "neighborhood_scope" in out.columns:
            out = out[out["neighborhood_scope"].astype(str).eq("ALL")].copy()

        if not out.empty and "precinct_norm" in out.columns:
            out["precinct_key"] = out["precinct_norm"].apply(norm_precinct)
        return out

    improvement = precinct_wide_only(precinct_improvement)
    recent = precinct_wide_only(precinct_crime_14d)
    trends = precinct_wide_only(precinct_crime_trends)
    priorities = precinct_wide_only(priority_concerns)
    timing = precinct_wide_only(temporal_summary)
    hotspots = precinct_wide_only(hotspot_change)

    focus = focus_locations.copy() if focus_locations is not None else pd.DataFrame()
    if not focus.empty:
        focus["precinct_key"] = focus["police_precinct"].apply(norm_precinct)

    precincts = _operational_precinct_values(temp)

    # Citywide headline context.
    current_ytd = None
    previous_ytd = None
    if city_ytd is not None and not city_ytd.empty:
        current_row = city_ytd[city_ytd["year"].astype(int).eq(int(current_year))]
        previous_row = city_ytd[city_ytd["year"].astype(int).eq(int(previous_year))]
        if not current_row.empty:
            current_ytd = int(current_row.iloc[0]["ytd_incidents"])
        if not previous_row.empty:
            previous_ytd = int(previous_row.iloc[0]["ytd_incidents"])
    city_pct = (
        100 * (current_ytd - previous_ytd) / previous_ytd
        if current_ytd is not None and previous_ytd not in (None, 0)
        else None
    )

    # Highest current matched-YTD precinct.
    highest_volume_precinct = None
    highest_volume = None
    if not improvement.empty:
        valid_imp = improvement[
            ~improvement["precinct_key"].str.upper().isin(invalid_precinct_codes)
        ].copy()
        if not valid_imp.empty:
            r = valid_imp.sort_values("incidents_current", ascending=False).iloc[0]
            highest_volume_precinct = str(r["precinct_key"])
            highest_volume = int(r["incidents_current"])

    city_data = {
        "total_incidents": int(len(temp)),
        "current_ytd": current_ytd,
        "previous_ytd": previous_ytd,
        "pct_change_ytd": city_pct,
        "current_year": int(current_year),
        "previous_year": int(previous_year),
        "min_year": int(min_year),
        "max_year": int(max_year),
        "data_through": data_through,
        "precinct_count": int(len(precincts)),
        "highest_volume_precinct": highest_volume_precinct,
        "highest_volume": highest_volume,
        "range_min_date": temp["incident_date"].min().strftime("%Y-%m-%d") if not temp.empty else None,
        "range_max_date": temp["incident_date"].max().strftime("%Y-%m-%d") if not temp.empty else None,
    }

    # Global spike-week marker used to calculate each precinct's deployment summary.
    spike_weeks = weekly[weekly["is_spike"]][["neighborhood", "week_start"]].drop_duplicates()
    spike_temp = temp.merge(
        spike_weeks.assign(is_spike_week=True),
        on=["neighborhood", "week_start"],
        how="left",
    )
    spike_temp["is_spike_week"] = spike_temp["is_spike_week"].eq(True)

    precinct_data = {}
    for precinct in precincts:
        p_df = temp[temp["precinct_key"].eq(precinct)].copy()

        # Overall matched YTD.
        overall = None
        if not improvement.empty:
            pp = improvement[improvement["precinct_key"].eq(precinct)]
            if not pp.empty:
                r = pp.iloc[0]
                overall = {
                    "incidents_baseline": clean_number(r.get("incidents_baseline")),
                    "incidents_previous": clean_number(r.get("incidents_previous")),
                    "incidents_current": clean_number(r.get("incidents_current")),
                    "change_vs_previous": clean_number(r.get("change_vs_previous")),
                    "pct_change_vs_previous": clean_number(r.get("pct_change_vs_previous")),
                    "pct_change_vs_baseline": clean_number(r.get("pct_change_vs_baseline")),
                    "improvement_status": str(r.get("improvement_status", "")),
                    "improvement_score": clean_number(r.get("improvement_score")),
                    "comparison_date": str(r.get("comparison_date", data_through)),
                }

        # Aggregate recent 14-day all-crime movement by summing crime rows.
        prev28 = curr28 = 0
        if not recent.empty:
            rr = recent[recent["precinct_key"].eq(precinct)]
            prev28 = int(pd.to_numeric(rr["previous_14d"], errors="coerce").fillna(0).sum())
            curr28 = int(pd.to_numeric(rr["current_14d"], errors="coerce").fillna(0).sum())
        recent_pct = 100 * (curr28 - prev28) / prev28 if prev28 else None

        # Priority concerns and improvements.
        concern_records, improvement_records = [], []
        concern_count = 0
        if not priorities.empty:
            pp = priorities[priorities["precinct_key"].eq(precinct)].copy()
            if not pp.empty:
                pp = pp.sort_values(
                    ["priority_score", "current_14d"],
                    ascending=[False, False],
                )
                concern_df = pp[
                    pp["priority_signal"].isin(["High Priority", "Emerging Concern", "Watch"])
                ]
                improve_df = pp[
                    pp["priority_signal"].eq("Recent Improvement")
                ].sort_values("priority_score", ascending=False)
                concern_count = int(len(concern_df))
                for _, r in concern_df.head(8).iterrows():
                    concern_records.append({
                        "crime": str(r.get("offense_category", "Unknown")),
                        "previous_14d": clean_number(r.get("previous_14d")),
                        "current_14d": clean_number(r.get("current_14d")),
                        "pct_change_14d": clean_number(r.get("pct_change_14d")),
                        "city_pct_change_14d": clean_number(r.get("city_pct_change_14d")),
                        "ytd_pct_change": clean_number(r.get("pct_change_vs_previous")),
                        "signal": str(r.get("priority_signal", "Monitor")),
                        "score": clean_number(r.get("priority_score")),
                    })
                for _, r in improve_df.head(5).iterrows():
                    improvement_records.append({
                        "crime": str(r.get("offense_category", "Unknown")),
                        "pct_change_14d": clean_number(r.get("pct_change_14d")),
                        "signal": str(r.get("priority_signal", "Recent Improvement")),
                    })

        # Top operational focus locations within this precinct.
        focus_records = []
        if not focus.empty:
            ff = focus[focus["precinct_key"].eq(precinct)].copy()
            if not ff.empty:
                ff = ff.sort_values(["score", "incident_count"], ascending=False).reset_index(drop=True)
                for i, (_, r) in enumerate(ff.head(10).iterrows(), start=1):
                    focus_records.append({
                        "rank": i,
                        "neighborhood": str(r.get("neighborhood", "Unknown")),
                        "intersection": str(r.get("nearest_intersection", "Unknown")),
                        "incident_count": clean_number(r.get("incident_count")),
                        "dominant_offense": str(r.get("dominant_offense", "Unknown")),
                        "dominant_offense_share": clean_number(r.get("dominant_offense_share")),
                        "mean_spike_z": clean_number(r.get("mean_spike_z")),
                        "score": clean_number(r.get("score")),
                    })

        # Deployment summary for only this precinct.
        ps = spike_temp[spike_temp["precinct_key"].eq(precinct)].copy()
        incidents = int(len(ps))
        spike_incidents = int(ps["is_spike_week"].sum())
        neighborhoods_covered = int(ps["neighborhood"].nunique())
        dominant_crime = "Unknown"
        dominant_count = 0
        if not ps.empty:
            vc = ps["offense_category"].astype(str).value_counts()
            if not vc.empty:
                dominant_crime = str(vc.index[0])
                dominant_count = int(vc.iloc[0])
        deployment = {
            "incidents": incidents,
            "spike_week_incidents": spike_incidents,
            "spike_incident_share": (spike_incidents / incidents if incidents else None),
            "neighborhoods_covered": neighborhoods_covered,
            "dominant_crime": dominant_crime,
            "dominant_crime_count": dominant_count,
            "dominant_crime_share": (dominant_count / incidents if incidents else None),
        }

        # Dominant crime types in the selected precinct (full data coverage).
        crime_records = []
        if not p_df.empty:
            vc = p_df["offense_category"].astype(str).value_counts()
            total_p = int(vc.sum())
            trend_lookup = {}
            if not trends.empty:
                pt = trends[trends["precinct_key"].eq(precinct)]
                trend_lookup = {
                    str(r["offense_category"]): {
                        "trend": str(r.get("trend_class", "")),
                        "ytd_pct": clean_number(r.get("pct_change_vs_previous")),
                        "current_ytd": clean_number(r.get("incidents_current")),
                    }
                    for _, r in pt.iterrows()
                }
            recent_lookup = {}
            if not recent.empty:
                pr = recent[recent["precinct_key"].eq(precinct)]
                recent_lookup = {
                    str(r["offense_category"]): {
                        "recent_pct": clean_number(r.get("pct_change_14d")),
                        "current_14d": clean_number(r.get("current_14d")),
                    }
                    for _, r in pr.iterrows()
                }
            for crime, count in vc.head(12).items():
                tr = trend_lookup.get(str(crime), {})
                rc = recent_lookup.get(str(crime), {})
                crime_records.append({
                    "crime": str(crime),
                    "incident_count": int(count),
                    "share": (int(count) / total_p if total_p else None),
                    "current_ytd": tr.get("current_ytd"),
                    "ytd_pct_change": tr.get("ytd_pct"),
                    "trend": tr.get("trend", ""),
                    "current_14d": rc.get("current_14d"),
                    "recent_pct_change": rc.get("recent_pct"),
                })

        # Crime-specific YTD drivers: strongest worsening and strongest improving.
        worsening_drivers, improving_drivers = [], []
        if not trends.empty:
            pt = trends[trends["precinct_key"].eq(precinct)].copy()
            if not pt.empty:
                pt["pct_numeric"] = pd.to_numeric(pt["pct_change_vs_previous"], errors="coerce")
                worsen = pt[pt["pct_numeric"].notna()].sort_values("pct_numeric", ascending=False)
                improve = pt[pt["pct_numeric"].notna()].sort_values("pct_numeric", ascending=True)
                for _, r in worsen.head(5).iterrows():
                    if float(r["pct_numeric"]) > 2:
                        worsening_drivers.append({
                            "crime": str(r["offense_category"]),
                            "previous": clean_number(r.get("incidents_previous")),
                            "current": clean_number(r.get("incidents_current")),
                            "pct": clean_number(r.get("pct_change_vs_previous")),
                            "trend": str(r.get("trend_class", "")),
                        })
                for _, r in improve.head(5).iterrows():
                    if float(r["pct_numeric"]) < -2:
                        improving_drivers.append({
                            "crime": str(r["offense_category"]),
                            "previous": clean_number(r.get("incidents_previous")),
                            "current": clean_number(r.get("incidents_current")),
                            "pct": clean_number(r.get("pct_change_vs_previous")),
                            "trend": str(r.get("trend_class", "")),
                        })

        # Shift summary and decision-purpose workload within this precinct.
        p_shift = build_shift_summary(p_df)
        shift_records = []
        if not p_shift.empty:
            for _, r in p_shift.iterrows():
                shift_records.append({
                    "shift": str(r.get("shift_window", "Unknown")),
                    "incidents": clean_number(r.get("shift_total_incidents")),
                    "dominant_offense": str(r.get("dominant_offense", "Unknown")),
                    "dominant_offense_count": clean_number(r.get("dominant_offense_count")),
                    "dominant_offense_share": clean_number(r.get("dominant_offense_share")),
                })

        p_decision = build_decision_purpose_summary(p_df)
        decision_records = []
        if not p_decision.empty:
            for _, r in p_decision.iterrows():
                decision_records.append({
                    "purpose": str(r.get("decision_purpose", "Unknown")),
                    "incidents": clean_number(r.get("purpose_total_incidents")),
                    "dominant_offense": str(r.get("dominant_offense", "Unknown")),
                    "dominant_offense_count": clean_number(r.get("dominant_offense_count")),
                    "dominant_offense_share": clean_number(r.get("dominant_offense_share")),
                })

        # Recent all-crime timing profile.
        timing_record = None
        if not timing.empty:
            tt = timing[
                (timing["precinct_key"].eq(precinct))
                & timing["period"].astype(str).eq("Recent 14D")
                & timing["selection_type"].astype(str).eq("All")
                & timing["selection_name"].astype(str).eq("All")
            ]
            if not tt.empty:
                r = tt.iloc[0]
                timing_record = {
                    "peak_day": str(r.get("peak_day", "Unknown")),
                    "peak_hour": clean_number(r.get("peak_hour")),
                    "peak_shift": str(r.get("peak_shift", "Unknown")),
                    "peak_time_block": str(r.get("peak_time_block", "Unknown")),
                    "period_start": str(r.get("period_start", "")),
                    "period_end": str(r.get("period_end", "")),
                }

        hotspot_counts = {"new": 0, "emerging": 0, "persistent": 0, "declining": 0}
        if not hotspots.empty:
            hh = hotspots[
                (hotspots["precinct_key"].eq(precinct))
                & hotspots["selection_type"].astype(str).eq("All")
                & hotspots["selection_name"].astype(str).eq("All")
            ]
            if not hh.empty:
                vc = hh["hotspot_status"].value_counts()
                hotspot_counts = {
                    "new": int(vc.get("New Hotspot", 0)),
                    "emerging": int(vc.get("Emerging Hotspot", 0)),
                    "persistent": int(vc.get("Persistent Hotspot", 0)),
                    "declining": int(vc.get("Declining Hotspot", 0)),
                }

        precinct_data[precinct] = {
            "overall": overall,
            "recent": {
                "previous_14d": int(prev28),
                "current_14d": int(curr28),
                "pct_change_14d": recent_pct,
            },
            "concern_count": concern_count,
            "concerns": concern_records,
            "recent_improvements": improvement_records,
            "focus_locations": focus_records,
            "deployment": deployment,
            "dominant_crimes": crime_records,
            "worsening_drivers": worsening_drivers,
            "improving_drivers": improving_drivers,
            "shift_summary": shift_records,
            "decision_summary": decision_records,
            "timing": timing_record,
            "hotspots": hotspot_counts,
            "assets": precinct_assets.get(precinct, {}),
        }

    daily_counts = build_daily_precinct_category_counts(temp, precincts)
    hourly_counts = build_hourly_precinct_category_counts(temp, precincts)
    # Compact current-year H3/day records let the overview recompute focus locations
    # and hotspot summaries for the exact custom comparison selected by the user.
    overview_spatial_daily = build_spatial_daily_payload(temp, current_year=max_year, resolution=8)
    overview_spatial_records = []
    if overview_spatial_daily is not None and not overview_spatial_daily.empty:
        for _, row in overview_spatial_daily.iterrows():
            overview_spatial_records.append({
                "date": str(row.get("date", "")), "h3": str(row.get("h3", "")),
                "precinct": str(row.get("precinct", "")), "neighborhood": str(row.get("neighborhood", "Unknown")),
                "crime": str(row.get("crime", "Unknown")), "count": int(row.get("count", 0) or 0),
                "lat": clean_number(row.get("lat")), "lon": clean_number(row.get("lon")),
                "intersection": str(row.get("intersection", "Unknown")),
            })

    payload = {
        "city": city_data,
        "precincts": precinct_data,
        "map_filename": map_filename,
        "area_map_filename": area_map_filename,
        "daily_counts": daily_counts,
        "hourly_counts": hourly_counts,
        "spatial_daily": overview_spatial_records,
    }
    payload_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    precinct_options = "".join(
        f'<option value="{html.escape(p, quote=True)}">Precinct {html.escape(p)}</option>'
        for p in precincts
    )

    html_text = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Detroit Crime Operations — {min_year} through {max_year}</title>
<style>
:root {{ --ink:#0f172a;--muted:#64748b;--line:#dbe3ef;--blue:#0b5cab;--soft:#f5f7fb;--card:#fff;--green:#047857;--red:#991b1b;--orange:#c2410c; }}
*{{box-sizing:border-box}} body{{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif;background:var(--soft);color:var(--ink)}}
.wrap{{max-width:1320px;margin:0 auto;padding:28px 24px 52px}} .eyebrow{{color:var(--blue);font-weight:800;letter-spacing:.04em;text-transform:uppercase;font-size:.78rem}}
h1{{margin:6px 0 4px;font-size:2.1rem}} h2{{margin:0 0 8px;font-size:1.4rem}} h3{{margin:0 0 8px;font-size:1.02rem}} .sub{{color:var(--muted);margin:0}}
.note{{color:var(--muted);font-size:.86rem;line-height:1.4}} .cards{{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:11px;margin:18px 0}}
.card,.section,.brief{{background:#fff;border:1px solid var(--line);border-radius:12px;box-shadow:0 1px 2px rgba(15,23,42,.03)}}
.card{{padding:14px 15px}} .card .label{{font-size:.78rem;color:var(--muted);font-weight:750}} .card .value{{font-size:1.45rem;font-weight:850;margin-top:5px}}
.selector{{background:#eef5ff;border:1px solid #bfdbfe;border-radius:12px;padding:16px;margin:18px 0 22px}} .selector-row{{display:flex;gap:14px;align-items:end;flex-wrap:wrap}}
.selector select{{min-width:260px;padding:10px 12px;border:1px solid #93a4b8;border-radius:8px;background:#fff;font-weight:750;font-size:.95rem}}
.brief{{padding:18px;margin-bottom:14px}} .section{{padding:17px;margin-top:14px}} .grid2{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}
table{{width:100%;border-collapse:collapse}} th,td{{padding:7px 7px;border-bottom:1px solid #e7edf5;text-align:right;font-size:.84rem;vertical-align:top}}
th{{color:#475569;font-size:.76rem}} th:first-child,td:first-child{{text-align:left}} tr:last-child td{{border-bottom:0}}
.links{{display:flex;gap:8px 16px;flex-wrap:wrap;margin:10px 0 2px}} .links a{{color:var(--blue);font-weight:760;text-decoration:none}} .links a:hover{{text-decoration:underline}}
.hero-row{{display:flex;justify-content:space-between;gap:16px;align-items:center;flex-wrap:wrap;margin:4px 0 14px}}
.hero-actions{{display:flex;gap:9px;align-items:center;flex-wrap:wrap}}
.primary-action{{display:inline-flex;align-items:center;justify-content:center;background:var(--blue);color:#fff!important;text-decoration:none!important;font-weight:850;padding:11px 16px;border-radius:9px;box-shadow:0 1px 2px rgba(15,23,42,.12)}}
.primary-action:hover{{filter:brightness(.96)}}
.secondary-action{{display:inline-flex;align-items:center;justify-content:center;border:1px solid #bfd0e5;background:#fff;color:var(--blue)!important;text-decoration:none!important;font-weight:780;padding:9px 13px;border-radius:8px}}
.overview-grid{{display:grid;grid-template-columns:1.35fr .65fr;gap:14px;margin:14px 0}}
.insight-card{{background:#fff;border:1px solid var(--line);border-radius:12px;padding:18px;box-shadow:0 1px 2px rgba(15,23,42,.03)}}
.insight-pair{{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:10px}}
.insight-stat{{background:#f8fafc;border:1px solid #e5ebf3;border-radius:9px;padding:11px}}
.insight-stat .k{{font-size:.76rem;color:var(--muted);font-weight:760;text-transform:uppercase;letter-spacing:.03em}}
.insight-stat .v{{font-size:1.02rem;font-weight:850;margin-top:4px}}
.comparison-panel{{background:#f8fbff;border:1px solid #bfdbfe;border-radius:12px;padding:18px;margin:14px 0}}
.comparison-title{{display:flex;justify-content:space-between;gap:10px;align-items:center;flex-wrap:wrap}}
.comparison-grid{{display:grid;grid-template-columns:repeat(4,minmax(145px,1fr));gap:10px;margin-top:12px}}
.comparison-grid label{{display:block;font-size:.76rem;color:#475569;font-weight:780;margin-bottom:5px}}
.comparison-grid input{{width:100%;padding:9px 10px;border:1px solid #a8b8ca;border-radius:8px;background:#fff}}
.comparison-actions{{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}}
.comparison-actions button{{border:0;border-radius:8px;padding:10px 13px;font-weight:800;cursor:pointer;background:var(--blue);color:#fff}}
.comparison-actions button.secondary-btn{{background:#fff;color:var(--blue);border:1px solid #9eb3ca}}
.detail-nav{{background:#fff;border:1px solid var(--line);border-radius:12px;padding:14px 16px;margin:14px 0}}
.detail-nav h3{{margin-bottom:4px}}
.city-hero{{background:linear-gradient(115deg,#082f5f,#0b5cab 62%,#1477c9);color:#fff;border-radius:16px;padding:28px 30px;margin:18px 0 14px;box-shadow:0 8px 22px rgba(15,23,42,.12);position:relative;overflow:hidden}}
.city-hero:after{{content:"";position:absolute;width:260px;height:260px;border:46px solid rgba(255,255,255,.06);border-radius:50%;right:-70px;top:-105px}}
.city-hero .eyebrow{{color:#bfdbfe}} .city-hero h2{{font-size:2rem;margin:5px 0 6px}} .city-hero p{{max-width:760px;color:#e7f1ff;line-height:1.55;margin:0}}
.city-kpis{{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:11px;margin:14px 0}}
.city-grid{{display:grid;grid-template-columns:1.05fr .95fr;gap:14px;margin-top:14px}}
.city-panel{{background:#fff;border:1px solid var(--line);border-radius:12px;padding:17px;box-shadow:0 1px 2px rgba(15,23,42,.03)}}
.city-panel-head{{display:flex;justify-content:space-between;align-items:flex-start;gap:10px;margin-bottom:10px}}
.precinct-glance{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}}
.precinct-tile{{border:1px solid #dbe3ef;border-radius:11px;padding:13px;background:#fff;cursor:pointer;transition:.15s ease;min-height:142px}}
.precinct-tile:hover{{transform:translateY(-2px);border-color:#8fb9e5;box-shadow:0 6px 15px rgba(15,23,42,.08)}}
.precinct-tile .phead{{display:flex;justify-content:space-between;gap:8px;align-items:center}} .precinct-tile .pnum{{font-size:1.2rem;font-weight:900}}
.precinct-tile .change{{font-weight:900}} .precinct-tile .mini{{display:grid;grid-template-columns:1fr 1fr;gap:7px;margin-top:11px}}
.precinct-tile .mini div{{background:#f8fafc;border-radius:7px;padding:7px}} .precinct-tile .mini span{{display:block;color:var(--muted);font-size:.68rem;font-weight:760}}
.precinct-tile .concern{{margin-top:9px;font-size:.78rem;color:#475569}} .precinct-tile .concern b{{color:var(--ink)}}
.attention-row{{display:grid;grid-template-columns:34px 80px 1fr 90px;gap:8px;align-items:center;padding:9px 4px;border-bottom:1px solid #edf1f6;font-size:.82rem}}
.attention-row:last-child{{border-bottom:0}} .attention-rank{{width:26px;height:26px;border-radius:50%;background:#fee2e2;color:#991b1b;display:flex;align-items:center;justify-content:center;font-weight:900}}
.attention-change{{font-weight:900;text-align:right}} .city-crime-row{{display:grid;grid-template-columns:145px 1fr 62px;gap:9px;align-items:center;margin:9px 0;font-size:.8rem}}
.city-bar{{height:8px;background:#e8eef6;border-radius:999px;overflow:hidden}} .city-bar i{{display:block;height:100%;background:#2f7ec5;border-radius:999px}}
.city-actions{{display:flex;gap:9px;flex-wrap:wrap;margin-top:13px}}
.city-map-callout{{background:#eef6ff;border:1px solid #bfdbfe;border-radius:11px;padding:14px;margin-top:12px}}
#cityPrompt{{border:0;background:transparent;box-shadow:none;padding:0;margin:0}}
.signal{{font-weight:800}} .high{{color:var(--red)}} .emerging{{color:var(--orange)}} .watch{{color:#a16207}} .improving{{color:var(--green)}}
.badge{{display:inline-block;border-radius:999px;padding:4px 9px;font-size:.76rem;font-weight:800;background:#e2e8f0;color:#334155}}
.metric-list{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:9px}} .metric{{padding:10px;border:1px solid #e5e7eb;border-radius:8px;background:#fafafa}}
.metric b{{display:block;font-size:1.05rem;margin-top:3px}} .driver-grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px}} .empty{{color:var(--muted);padding:8px 0}}
#precinctOverview{{display:none}} .anchor-target{{scroll-margin-top:12px}}
@media(max-width:980px){{.cards,.city-kpis{{grid-template-columns:repeat(2,minmax(0,1fr))}}.grid2,.driver-grid,.overview-grid,.city-grid{{grid-template-columns:1fr}}.comparison-grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}.precinct-glance{{grid-template-columns:repeat(2,minmax(0,1fr))}}}}
@media(max-width:560px){{.wrap{{padding:18px 12px 34px}}.cards,.city-kpis,.precinct-glance{{grid-template-columns:1fr}}.metric-list,.comparison-grid,.insight-pair{{grid-template-columns:1fr}}.selector select{{width:100%;min-width:0}}.primary-action,.secondary-action{{width:100%}}.city-hero{{padding:22px 18px}}.attention-row{{grid-template-columns:30px 70px 1fr 72px}}}}
</style>
</head>
<body>
<div class="wrap">
<div class="eyebrow">Operational Control Center</div>
<h1>Detroit Crime Operations Dashboard</h1>
<p class="sub">Data coverage: {min_year} through {max_year}</p>
<p class="note">Matched current-year comparisons use incidents through <b>{data_through}</b>. Select a precinct to evaluate only the workload and geography that precinct controls.</p>

<div class="cards" id="cityCards"></div>

<div class="selector">
  <div class="selector-row">
    <div><div style="font-weight:800;margin-bottom:6px;">Select precinct</div>
      <select id="precinctSelect"><option value="">Detroit Overview</option>{precinct_options}</select>
    </div>
    <div style="max-width:650px;color:#475569;line-height:1.45;">The selected precinct becomes the master context. Every overview section and every blue drill-down link below is scoped to that precinct.</div>
  </div>
</div>

<div id="cityPrompt" class="brief">
  <div class="city-hero">
    <div class="eyebrow">Detroit Citywide Overview</div>
    <h2>Crime operations at a glance</h2>
    <p>Start citywide, compare recent activity across operational precincts, and identify where attention may be needed. Select any precinct card—or use the precinct selector above—to open the existing full Precinct Operations Evaluation.</p>
  </div>

  <div class="city-kpis" id="cityOverviewKpis"></div>

  <div class="city-grid">
    <div class="city-panel">
      <div class="city-panel-head">
        <div><div class="eyebrow">Precinct Comparison</div><h3 style="font-size:1.12rem;margin-top:4px;">Precincts requiring attention</h3></div>
        <div class="note">Ranked by recent 14-day increase</div>
      </div>
      <div id="cityAttention"></div>
    </div>

    <div class="city-panel">
      <div class="city-panel-head">
        <div><div class="eyebrow">Citywide Crime Snapshot</div><h3 style="font-size:1.12rem;margin-top:4px;">Current 14 days by crime type</h3></div>
        <div class="note">Largest current volumes</div>
      </div>
      <div id="cityCrimeSnapshot"></div>
      <div class="city-map-callout">
        <b>Need the citywide spatial view?</b>
        <div class="note" style="margin-top:4px;">Open the existing Detroit interactive map. Custom layer building remains available from inside a selected precinct.</div>
        <div class="city-actions"><a class="secondary-action" href="../Images/{area_map_filename}" target="_blank">Open Detroit Interactive Map</a></div>
      </div>
    </div>
  </div>

  <div class="city-panel" style="margin-top:14px;">
    <div class="city-panel-head">
      <div><div class="eyebrow">Detroit Precincts at a Glance</div><h3 style="font-size:1.12rem;margin-top:4px;">Choose a precinct to investigate</h3></div>
      <div class="note">Click a card to open its Operations Evaluation</div>
    </div>
    <div class="precinct-glance" id="cityPrecinctGlance"></div>
  </div>
</div>

<div id="precinctOverview">
  <div class="hero-row">
    <div>
      <div class="eyebrow">Precinct Evaluation</div>
      <h2 id="precinctTitle" style="font-size:1.8rem;margin:4px 0 3px;"></h2>
      <div id="precinctDate" class="sub"></div>
    </div>
    <div class="hero-actions">
      <span class="badge" id="trendBadge"></span>
      <a id="buildMapBtn" class="primary-action" href="#" target="_blank">Build Custom Crime Map</a>
    </div>
  </div>

  <div class="cards" id="precinctCards"></div>

  <div class="overview-grid">
    <div class="insight-card">
      <div class="eyebrow">Overall Evaluation</div>
      <h3 style="font-size:1.12rem;margin-top:5px;">What changed?</h3>
      <p id="precinctBrief" style="line-height:1.62;margin:7px 0 0;"></p>
    </div>
    <div class="insight-card">
      <div class="eyebrow">Operational Snapshot</div>
      <div class="insight-pair">
        <div class="insight-stat"><div class="k">Leading concern</div><div class="v" id="topConcernQuick">—</div></div>
        <div class="insight-stat"><div class="k">Peak timing</div><div class="v" id="timingQuick">—</div></div>
      </div>
      <div class="note" style="margin-top:10px;">Use the detailed sections below to investigate crime mix, workload, timing, focus locations, and hotspot change.</div>
    </div>
  </div>

  <div class="comparison-panel" id="rangeFilterBox">
    <div class="comparison-title">
      <div>
        <div class="eyebrow">Comparison Period</div>
        <h3 style="font-size:1.12rem;margin-top:4px;">Automated 14-day comparison or custom dates</h3>
      </div>
      <div class="note" id="rangeStatus"></div>
    </div>
    <div class="note" id="comparisonScopeNote">By default, operational sections compare the latest 14 days with the immediately preceding 14 days. When a custom comparison is applied, period-based sections use those selected dates consistently; matched-YTD sections remain YTD.</div>
    <div class="comparison-grid">
      <div><label for="rangePrevStart">Previous start</label><input type="date" id="rangePrevStart"></div>
      <div><label for="rangePrevEnd">Previous end</label><input type="date" id="rangePrevEnd"></div>
      <div><label for="rangeCurrStart">Current start</label><input type="date" id="rangeCurrStart"></div>
      <div><label for="rangeCurrEnd">Current end</label><input type="date" id="rangeCurrEnd"></div>
    </div>
    <div class="comparison-actions">
      <button id="rangeApplyBtn" type="button">Apply comparison</button>
      <button id="rangeResetBtn" type="button" class="secondary-btn">Reset to automated 14-day</button>
    </div>
  </div>

  <div class="detail-nav">
    <h3>Detailed analysis</h3>
    <div class="note">Open a focused drill-down only when you need a separate analytical view.</div>
    <div class="links" id="modernLinks"></div>
    <div class="links" id="classicLinks" style="padding-top:5px;border-top:1px solid #eef2f7;"></div>
  </div>

  <div class="section anchor-target" id="focusSection">
    <h2>Top 10 Operational Focus Locations</h2>
    <div class="note" id="focusNote">Highest-activity spatial cells inside the selected precinct for the current analysis period. With a custom comparison, these locations are recomputed from the selected Current range.</div>
    <div id="focusLocations"></div>
  </div>

  <div class="section anchor-target" id="deploymentSection">
    <h2>Precinct Summary for Deployment</h2>
    <div class="note">Spike incident share indicates pressure from spike-week workload inside this precinct.</div>
    <div id="deploymentSummary"></div>
  </div>

  <div class="section anchor-target" id="improvementSection">
    <h2>Precinct Improvement — Matched Year-to-Date</h2>
    <div id="improvementSummary"></div>
    <div class="driver-grid" style="margin-top:12px;">
      <div><h3>Largest worsening drivers</h3><div id="worseningDrivers"></div></div>
      <div><h3>Largest improving drivers</h3><div id="improvingDrivers"></div></div>
    </div>
  </div>

  <div class="section anchor-target" id="crimeSection">
    <h2>Dominant Crime Types — Selected Precinct</h2>
    <div class="note" id="crimeNote">Long-run and matched-YTD context are shown alongside the current operational period. When custom dates are applied, the recent columns use the selected Previous and Current ranges.</div>
    <div id="dominantCrimes"></div>
  </div>

  <div class="section anchor-target" id="prioritySection">
    <h2>Priority / Emerging Concerns</h2>
    <div class="note" id="priorityNote">Compares offense-level activity in the current operational period with the preceding period, while retaining city and matched-YTD context.</div>
    <div id="concernsTable"></div>
  </div>

  <div class="grid2">
    <div class="section anchor-target" id="shiftSection"><h2>Shift-Level Demand and Dominant Crime</h2><div class="note" id="shiftNote">Shows when incidents occurred during the current operational period and the dominant crime within each shift.</div><div id="shiftSummary"></div></div>
    <div class="section anchor-target" id="decisionSection"><h2>Decision-Purpose Workload and Dominant Crime</h2><div class="note" id="decisionNote">Groups incidents in the current operational period into broad response purposes using the dashboard's offense-to-purpose rules; it is an analytical grouping, not an official disposition.</div><div id="decisionSummary"></div></div>
  </div>

  <div class="grid2">
    <div class="section anchor-target" id="timingSection"><h2>Timing Snapshot</h2><div class="note" id="timingNote">Summarizes the peak day, hour, broad time block, and shift for the current operational period.</div><div id="timingSummary"></div></div>
    <div class="section anchor-target" id="hotspotSection"><h2>Hotspot Change Snapshot</h2><div class="note" id="hotspotNote">Compares spatial concentration cells between the Previous and Current operational periods. Custom dates recompute these counts using the selected ranges.</div><div id="hotspotSummary"></div></div>
  </div>
</div>
</div>

<script>
const DATA={payload_json};
const city=DATA.city, precincts=DATA.precincts;
const select=document.getElementById('precinctSelect');
function fmtN(v){{if(v===null||v===undefined||Number.isNaN(Number(v)))return '—';return Number(v).toLocaleString();}}
function fmtPct(v){{if(v===null||v===undefined||Number.isNaN(Number(v)))return '—';let n=Number(v);return (n>0?'+':'')+n.toFixed(1)+'%';}}
function fmtShare(v){{if(v===null||v===undefined||Number.isNaN(Number(v)))return '—';return (100*Number(v)).toFixed(1)+'%';}}
function pctClass(v){{if(v===null||v===undefined||Number.isNaN(Number(v)))return '';return Number(v)>2?'high':Number(v)<-2?'improving':'';}}
function signalClass(s){{return s==='High Priority'?'high':s==='Emerging Concern'?'emerging':s==='Watch'?'watch':s==='Recent Improvement'?'improving':'';}}
function hourLabel(h){{if(h===null||h===undefined||Number.isNaN(Number(h)))return '—';let n=Number(h),s=n>=12?'PM':'AM',h12=n%12||12;return h12+' '+s;}}
function analysisUrl(p,section){{
 let u='../Images/'+DATA.map_filename+'?precinct='+encodeURIComponent(p)+'&section='+encodeURIComponent(section);
 if(customRange){{u+='&prevStart='+encodeURIComponent(customRange.prevStart)+'&prevEnd='+encodeURIComponent(customRange.prevEnd)+'&currStart='+encodeURIComponent(customRange.currStart)+'&currEnd='+encodeURIComponent(customRange.currEnd);}}
 return u;
}}
function assetLink(path,label){{return path?`<a href="${{path}}" target="_blank">${{label}}</a>`:'';}}

const DECISION_PURPOSE_MAP={{'LARCENY':'Preventive Patrol','BURGLARY':'Preventive Patrol','STOLEN VEHICLE':'Preventive Patrol','STOLEN PROPERTY':'Preventive Patrol','ROBBERY':'Preventive Patrol','DAMAGE TO PROPERTY':'Preventive Patrol','FRAUD':'Investigations','FORGERY':'Investigations','EMBEZZLEMENT':'Investigations','ARSON':'Investigations','BRIBERY':'Investigations','ASSAULT':'Community Response','AGGRAVATED ASSAULT':'Community Response','WEAPONS OFFENSES':'Community Response','OBSTRUCTING THE POLICE':'Community Response','HOMICIDE':'Community Response','KIDNAPPING':'Community Response'}};
function decisionPurposeFor(crime){{return DECISION_PURPOSE_MAP[String(crime||'').toUpperCase()]||'Preventive Patrol';}}

const dailyCounts=DATA.daily_counts||{{precincts:[],categories:[],base_date:null,rows:[]}};
const overviewSpatial=DATA.spatial_daily||[];
function customSpatialCells(precinct,range){{
 const cells={{}};
 (overviewSpatial||[]).forEach(r=>{{
  if(String(r.precinct)!==String(precinct))return;
  const isPrev=r.date>=range.prevStart&&r.date<=range.prevEnd;
  const isCurr=r.date>=range.currStart&&r.date<=range.currEnd;
  if(!isPrev&&!isCurr)return;
  const key=r.h3;if(!cells[key])cells[key]={{h3:key,prev:0,curr:0,lat:r.lat,lon:r.lon,neighborhood:r.neighborhood||'Unknown',intersection:r.intersection||'Unknown',crimes:{{}}}};
  const c=cells[key];
  if(isPrev)c.prev+=Number(r.count||0);
  if(isCurr){{c.curr+=Number(r.count||0);c.crimes[r.crime]=(c.crimes[r.crime]||0)+Number(r.count||0);}}
 }});
 const arr=Object.values(cells);
 const prevPositive=arr.map(c=>c.prev).filter(v=>v>0).sort((a,b)=>a-b);
 const currPositive=arr.map(c=>c.curr).filter(v=>v>0).sort((a,b)=>a-b);
 function q80(a){{if(!a.length)return 3;const i=Math.min(a.length-1,Math.floor(.8*(a.length-1)));return Math.max(3,a[i]);}}
 const pt=q80(prevPositive),ct=q80(currPositive);
 arr.forEach(c=>{{
  const ph=c.prev>=pt,ch=c.curr>=ct;
  c.status=(!ph&&ch)?(c.prev===0?'New Hotspot':'Emerging Hotspot'):(ph&&ch)?'Persistent Hotspot':(ph&&!ch)?'Declining Hotspot':'Not Hotspot';
  let dom='—',mx=0;Object.keys(c.crimes).forEach(k=>{{if(c.crimes[k]>mx){{mx=c.crimes[k];dom=k;}}}});c.dominant=dom;c.dominant_count=mx;
 }});
 return arr;
}}
function customFocusRows(precinct,range){{return customSpatialCells(precinct,range).filter(c=>c.curr>0).sort((a,b)=>b.curr-a.curr).slice(0,10);}}
function customHotspotCounts(precinct,range){{const out={{new:0,emerging:0,persistent:0,declining:0}};customSpatialCells(precinct,range).forEach(c=>{{if(c.status==='New Hotspot')out.new++;else if(c.status==='Emerging Hotspot')out.emerging++;else if(c.status==='Persistent Hotspot')out.persistent++;else if(c.status==='Declining Hotspot')out.declining++;}});return out;}}
const dcPrecinctIdx={{}};(dailyCounts.precincts||[]).forEach((p,i)=>{{dcPrecinctIdx[p]=i;}});
const dcBaseDate=dailyCounts.base_date?new Date(dailyCounts.base_date+'T00:00:00Z'):null;
function dayOffsetFor(dateStr){{if(!dcBaseDate||!dateStr)return null;const d=new Date(dateStr+'T00:00:00Z');return Math.round((d-dcBaseDate)/86400000);}}
function categoryCountsInRange(precinct,startOffset,endOffset){{
 const out={{}};let total=0;
 if(startOffset===null||endOffset===null||startOffset>endOffset)return{{byCategory:out,total:0}};
 const pIdx=(precinct!==null&&precinct!==undefined)?dcPrecinctIdx[precinct]:null;
 const cats=dailyCounts.categories||[];
 (dailyCounts.rows||[]).forEach(r=>{{
  const pi=r[0],ci=r[1],day=r[2],cnt=r[3];
  if(pIdx!==null&&pi!==pIdx)return;
  if(day<startOffset||day>endOffset)return;
  const cat=cats[ci];
  out[cat]=(out[cat]||0)+cnt;total+=cnt;
 }});
 return{{byCategory:out,total:total}};
}}

const hourlyCounts=DATA.hourly_counts||{{precincts:[],categories:[],base_date:null,rows:[]}};
const hcPrecinctIdx={{}};(hourlyCounts.precincts||[]).forEach((p,i)=>{{hcPrecinctIdx[p]=i;}});
const hcBaseDate=hourlyCounts.base_date?new Date(hourlyCounts.base_date+'T00:00:00Z'):null;

function hourlyDayOffsetFor(dateStr){{
 if(!hcBaseDate||!dateStr)return null;
 const d=new Date(dateStr+'T00:00:00Z');
 return Math.round((d-hcBaseDate)/86400000);
}}
function shiftForHour(hour){{
 const h=Number(hour);
 if(h>=6&&h<=13)return 'Day Shift (06:00-13:59)';
 if(h>=14&&h<=21)return 'Evening Shift (14:00-21:59)';
 return 'Night Shift (22:00-05:59)';
}}
function blockForHour(hour){{
 const h=Number(hour);
 if(h<=5)return '00:00-05:59';
 if(h<=11)return '06:00-11:59';
 if(h<=17)return '12:00-17:59';
 return '18:00-23:59';
}}
function weekdayForOffset(dayOffset){{
 if(!hcBaseDate)return 'Unknown';
 const d=new Date(hcBaseDate.getTime()+Number(dayOffset)*86400000);
 return ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'][d.getUTCDay()];
}}
function hourlyRowsForRange(precinct,startDate,endDate){{
 const startOffset=hourlyDayOffsetFor(startDate),endOffset=hourlyDayOffsetFor(endDate);
 if(startOffset===null||endOffset===null||startOffset>endOffset)return[];
 const pIdx=hcPrecinctIdx[precinct];
 if(pIdx===undefined)return[];
 return (hourlyCounts.rows||[]).filter(r=>r[0]===pIdx&&r[2]>=startOffset&&r[2]<=endOffset);
}}
function customShiftSummary(precinct,startDate,endDate){{
 const rows=hourlyRowsForRange(precinct,startDate,endDate);
 const cats=hourlyCounts.categories||[];
 const byShift={{}};
 rows.forEach(r=>{{
  const cat=cats[r[1]]||'Unknown',hour=r[3],cnt=Number(r[4]||0),shift=shiftForHour(hour);
  if(!byShift[shift])byShift[shift]={{incidents:0,byCrime:{{}}}};
  byShift[shift].incidents+=cnt;
  byShift[shift].byCrime[cat]=(byShift[shift].byCrime[cat]||0)+cnt;
 }});
 const order=['Day Shift (06:00-13:59)','Evening Shift (14:00-21:59)','Night Shift (22:00-05:59)'];
 return order.filter(s=>byShift[s]).map(shift=>{{
  const info=byShift[shift];
  let dom='—',domCount=0;
  Object.keys(info.byCrime).forEach(cat=>{{if(info.byCrime[cat]>domCount){{dom=cat;domCount=info.byCrime[cat];}}}});
  return{{shift:shift,incidents:info.incidents,dominant_offense:dom,dominant_offense_count:domCount,dominant_offense_share:info.incidents?domCount/info.incidents:null}};
 }});
}}
function customTimingSummary(precinct,startDate,endDate){{
 const rows=hourlyRowsForRange(precinct,startDate,endDate);
 if(!rows.length)return null;
 const byDay={{}},byHour={{}},byShift={{}},byBlock={{}};
 let total=0;
 rows.forEach(r=>{{
  const day=weekdayForOffset(r[2]),hour=Number(r[3]),cnt=Number(r[4]||0);
  const shift=shiftForHour(hour),block=blockForHour(hour);
  byDay[day]=(byDay[day]||0)+cnt;
  byHour[hour]=(byHour[hour]||0)+cnt;
  byShift[shift]=(byShift[shift]||0)+cnt;
  byBlock[block]=(byBlock[block]||0)+cnt;
  total+=cnt;
 }});
 function peak(obj){{
  let key=null,count=-1;
  Object.keys(obj).forEach(k=>{{if(obj[k]>count){{key=k;count=obj[k];}}}});
  return [key,count];
 }}
 const pd=peak(byDay),ph=peak(byHour),ps=peak(byShift),pb=peak(byBlock);
 return{{
  total_incidents:total,
  peak_day:pd[0]||'—',peak_day_count:pd[1]||0,
  peak_hour:ph[0]===null?null:Number(ph[0]),peak_hour_count:ph[1]||0,
  peak_shift:ps[0]||'—',peak_shift_count:ps[1]||0,
  peak_time_block:pb[0]||'—',peak_time_block_count:pb[1]||0,
  period_start:startDate,period_end:endDate
 }};
}}

function computeConcernRows(prevByCat,currByCat,cityPrevByCat,cityCurrByCat,ytdLookup){{
 const cats=Object.keys(Object.assign({{}},prevByCat,currByCat));
 const rows=cats.map(cat=>{{
  const prev=prevByCat[cat]||0,curr=currByCat[cat]||0;
  const change=curr-prev;
  const pct=prev>0?(100*change/prev):null;
  const cityPrev=cityPrevByCat[cat]||0,cityCurr=cityCurrByCat[cat]||0;
  const cityChange=cityCurr-cityPrev;
  const cityPct=cityPrev>0?(100*cityChange/cityPrev):null;
  const ytd=ytdLookup[cat];
  const ytdPct=(ytd===undefined||ytd===null||Number.isNaN(Number(ytd)))?null:Number(ytd);
  return{{crime:cat,previous_28d:prev,current_28d:curr,change:change,pct_change_28d:pct,city_pct_change_28d:cityPct,ytd_pct_change:ytdPct}};
 }});
 const maxVolume=Math.max(1,...rows.map(r=>r.current_28d));
 const maxAbsIncrease=Math.max(1,...rows.map(r=>Math.max(r.change,0)));
 rows.forEach(r=>{{
  const volumeComponent=25*Math.log1p(Math.max(r.current_28d,0))/Math.log1p(maxVolume);
  const absoluteComponent=30*Math.max(r.change,0)/maxAbsIncrease;
  let pctSignal=r.pct_change_28d;
  if(pctSignal===null||pctSignal===undefined||Number.isNaN(pctSignal))pctSignal=Math.min(Math.max(r.current_28d,0)*10,100);
  const percentComponent=20*Math.min(Math.max(pctSignal,0),100)/100;
  const cityGap=(r.pct_change_28d===null||r.pct_change_28d===undefined||Number.isNaN(r.pct_change_28d)||r.city_pct_change_28d===null||r.city_pct_change_28d===undefined||Number.isNaN(r.city_pct_change_28d))?0:(r.pct_change_28d-r.city_pct_change_28d);
  const cityComponent=15*Math.min(Math.max(cityGap,0),50)/50;
  const ytdVal=(r.ytd_pct_change===null||r.ytd_pct_change===undefined)?0:r.ytd_pct_change;
  const ytdComponent=10*Math.min(Math.max(ytdVal,0),50)/50;
  r.score=Math.round((volumeComponent+absoluteComponent+percentComponent+cityComponent+ytdComponent)*10)/10;
  const volume=r.current_28d,absChange=r.change,recentPct=r.pct_change_28d,zeroBaseline=r.previous_28d===0;
  const high=(volume>=20)&&(absChange>=10)&&((recentPct!==null&&recentPct>=25)||zeroBaseline)&&((cityGap>=10)||(ytdVal>2));
  const emerging=(volume>=10)&&(absChange>=5)&&((recentPct!==null&&recentPct>=10)||zeroBaseline)&&((cityGap>=5)||(ytdVal>2));
  const watch=(volume>=5)&&(absChange>0)&&((recentPct!==null&&recentPct>2)||zeroBaseline);
  const improving=(volume>=5)&&(absChange<=-5)&&(recentPct!==null&&recentPct<=-10);
  r.signal=high?'High Priority':emerging?'Emerging Concern':watch?'Watch':improving?'Recent Improvement':'Monitor';
 }});
 const order={{'High Priority':0,'Emerging Concern':1,'Watch':2,'Recent Improvement':3,'Monitor':4}};
 rows.sort((a,b)=>{{const oa=order[a.signal],ob=order[b.signal];if(oa!==ob)return oa-ob;if(b.score!==a.score)return b.score-a.score;return b.current_28d-a.current_28d;}});
 return rows;
}}
let customRange=null;
(function initRangeInputs(){{
 const minD=city.range_min_date,maxD=city.range_max_date;
 ['rangePrevStart','rangePrevEnd','rangeCurrStart','rangeCurrEnd'].forEach(id=>{{
  const el=document.getElementById(id);if(!el)return;
  if(minD)el.min=minD;if(maxD)el.max=maxD;
 }});
}})();

function renderCity(){{
 // Existing top context cards remain useful above the citywide landing experience.
 document.getElementById('cityCards').innerHTML=`
 <div class="card"><div class="label">Total incidents ${{city.min_year}}–${{city.max_year}}</div><div class="value">${{fmtN(city.total_incidents)}}</div></div>
 <div class="card"><div class="label">${{city.current_year}} YTD incidents</div><div class="value">${{fmtN(city.current_ytd)}}</div></div>
 <div class="card"><div class="label">${{city.current_year}} vs ${{city.previous_year}} YTD</div><div class="value ${{pctClass(city.pct_change_ytd)}}">${{fmtPct(city.pct_change_ytd)}}</div></div>
 <div class="card"><div class="label">Operational precincts</div><div class="value">${{fmtN(city.precinct_count)}}</div></div>
 <div class="card"><div class="label">Highest-volume precinct</div><div class="value">${{city.highest_volume_precinct?'P'+city.highest_volume_precinct:'—'}}</div><div class="note">${{city.highest_volume?fmtN(city.highest_volume)+' current YTD':''}}</div></div>`;

 const rows=Object.keys(precincts).map(p=>{{
  const d=precincts[p]||{{}},r=d.recent||{{}},ov=d.overall||{{}},concerns=d.concerns||[];
  return {{
   precinct:p,
   current:Number(r.current_14d||0),
   previous:Number(r.previous_14d||0),
   change:r.pct_change_14d,
   ytdChange:ov.pct_change_vs_previous,
   ytd:Number(ov.incidents_current||0),
   concern:concerns.length?(concerns[0].crime||concerns[0].offense_category||'Attention signal'):'No elevated signal',
   concernSignal:concerns.length?(concerns[0].signal||concerns[0].priority_signal||''):''
  }};
 }});

 const currentTotal=rows.reduce((a,r)=>a+r.current,0);
 const previousTotal=rows.reduce((a,r)=>a+r.previous,0);
 const recentPct=previousTotal?100*(currentTotal-previousTotal)/previousTotal:null;
 const increasing=rows.filter(r=>r.change!==null&&r.change!==undefined&&Number(r.change)>0).length;
 const improving=rows.filter(r=>r.change!==null&&r.change!==undefined&&Number(r.change)<0).length;

 document.getElementById('cityOverviewKpis').innerHTML=`
 <div class="card"><div class="label">Current 14 days</div><div class="value">${{fmtN(currentTotal)}}</div></div>
 <div class="card"><div class="label">Previous 14 days</div><div class="value">${{fmtN(previousTotal)}}</div></div>
 <div class="card"><div class="label">Citywide recent change</div><div class="value ${{pctClass(recentPct)}}">${{fmtPct(recentPct)}}</div></div>
 <div class="card"><div class="label">Precincts increasing</div><div class="value high">${{increasing}}</div></div>
 <div class="card"><div class="label">Precincts decreasing</div><div class="value improving">${{improving}}</div></div>`;

 const attention=rows.filter(r=>r.change!==null&&r.change!==undefined&&Number(r.change)>0)
   .sort((a,b)=>Number(b.change)-Number(a.change)).slice(0,5);
 document.getElementById('cityAttention').innerHTML=attention.length?attention.map((r,i)=>`
  <div class="attention-row" data-precinct="${{r.precinct}}" style="cursor:pointer">
   <div class="attention-rank">${{i+1}}</div>
   <div><b>P${{r.precinct}}</b></div>
   <div><b>${{r.concern}}</b><div class="note">${{fmtN(r.current)}} current vs ${{fmtN(r.previous)}} previous</div></div>
   <div class="attention-change high">${{fmtPct(r.change)}}</div>
  </div>`).join(''):'<div class="empty">No precincts currently show a recent increase.</div>';

 // Aggregate the already-calculated precinct crime rows into a citywide current-14D snapshot.
 const crimeTotals={{}};
 Object.values(precincts).forEach(d=>{{
  (d.dominant_crimes||[]).forEach(r=>{{
   const name=r.crime||r.offense_category||r.crime_type||'Unknown';
   const val=Number(r.current_14d||r.current_28d||r.incidents_current||r.current_incidents||0);
   crimeTotals[name]=(crimeTotals[name]||0)+val;
  }});
 }});
 const crimeRows=Object.entries(crimeTotals).sort((a,b)=>b[1]-a[1]).slice(0,7);
 const maxCrime=crimeRows.length?crimeRows[0][1]:1;
 document.getElementById('cityCrimeSnapshot').innerHTML=crimeRows.length?crimeRows.map(([name,val])=>`
  <div class="city-crime-row"><div><b>${{name}}</b></div><div class="city-bar"><i style="width:${{Math.max(4,100*val/maxCrime)}}%"></i></div><div style="text-align:right;font-weight:850">${{fmtN(val)}}</div></div>`).join('')
  :'<div class="empty">Crime-type summary unavailable.</div>';

 const sorted=rows.slice().sort((a,b)=>Number(b.change||0)-Number(a.change||0));
 document.getElementById('cityPrecinctGlance').innerHTML=sorted.map(r=>`
  <div class="precinct-tile" data-precinct="${{r.precinct}}" tabindex="0" role="button" aria-label="Open Precinct ${{r.precinct}} Operations Evaluation">
   <div class="phead"><div class="pnum">Precinct ${{r.precinct}}</div><div class="change ${{pctClass(r.change)}}">${{fmtPct(r.change)}}</div></div>
   <div class="mini">
    <div><span>Current 14D</span><b>${{fmtN(r.current)}}</b></div>
    <div><span>Previous 14D</span><b>${{fmtN(r.previous)}}</b></div>
   </div>
   <div class="concern">Top concern<br><b>${{r.concern}}</b></div>
   <div class="note" style="margin-top:8px;">Open precinct evaluation →</div>
  </div>`).join('');

 function openPrecinctFromCity(p){{
  select.value=p;
  renderPrecinct(p);
  window.scrollTo({{top:0,behavior:'smooth'}});
 }}
 document.querySelectorAll('#cityPrecinctGlance [data-precinct],#cityAttention [data-precinct]').forEach(el=>{{
  el.addEventListener('click',()=>openPrecinctFromCity(el.dataset.precinct));
  el.addEventListener('keydown',e=>{{if(e.key==='Enter'||e.key===' '){{e.preventDefault();openPrecinctFromCity(el.dataset.precinct);}}}});
 }});
}}

function simpleTable(headers,rows){{
 if(!rows||!rows.length)return '<div class="empty">No records available.</div>';
 let h='<table><thead><tr>'+headers.map(x=>'<th>'+x+'</th>').join('')+'</tr></thead><tbody>';
 rows.forEach(row=>{{h+='<tr>'+row.map(x=>'<td>'+x+'</td>').join('')+'</tr>';}});
 return h+'</tbody></table>';
}}

function renderPrecinct(p){{
 const d=precincts[p]; if(!d)return;
 document.getElementById('cityPrompt').style.display='none';
 document.getElementById('precinctOverview').style.display='block';
 const ov=d.overall||{{}}, a=d.assets||{{}};
 document.getElementById('precinctTitle').textContent='Precinct '+p+' Operations Evaluation';
 document.getElementById('precinctDate').textContent='RMS data available through '+city.data_through+' · matched YTD through '+(ov.comparison_date||city.data_through)+' · coverage '+city.min_year+'–'+city.max_year;
 document.getElementById('trendBadge').textContent=ov.improvement_status||'Trend unavailable';

 let rec, crimeRecordsOverride=null, concernsOverride=null, decisionOverride=null, shiftOverride=null, timingOverride=null, focusOverride=null, hotspotOverride=null;
 const statusEl=document.getElementById('rangeStatus');
 if(customRange){{
  const prevStartOff=dayOffsetFor(customRange.prevStart),prevEndOff=dayOffsetFor(customRange.prevEnd);
  const currStartOff=dayOffsetFor(customRange.currStart),currEndOff=dayOffsetFor(customRange.currEnd);
  const prevAgg=categoryCountsInRange(p,prevStartOff,prevEndOff);
  const currAgg=categoryCountsInRange(p,currStartOff,currEndOff);
  const cityPrevAgg=categoryCountsInRange(null,prevStartOff,prevEndOff);
  const cityCurrAgg=categoryCountsInRange(null,currStartOff,currEndOff);
  rec={{previous_28d:prevAgg.total,current_28d:currAgg.total,pct_change_28d:prevAgg.total?100*(currAgg.total-prevAgg.total)/prevAgg.total:null}};

  const baseCrimeLookup={{}};(d.dominant_crimes||[]).forEach(r=>{{baseCrimeLookup[r.crime]=r;}});
  const allCats=Object.keys(Object.assign({{}},prevAgg.byCategory,currAgg.byCategory));
  crimeRecordsOverride=allCats.map(cat=>{{
   const base=baseCrimeLookup[cat]||{{}};
   const curr=currAgg.byCategory[cat]||0,prev=prevAgg.byCategory[cat]||0;
   return{{
    crime:cat,
    incident_count:base.incident_count!==undefined?base.incident_count:null,
    share:base.share!==undefined?base.share:null,
    current_ytd:base.current_ytd!==undefined?base.current_ytd:null,
    ytd_pct_change:base.ytd_pct_change!==undefined?base.ytd_pct_change:null,
    trend:base.trend||'—',
    current_28d:curr,
    recent_pct_change:prev?100*(curr-prev)/prev:null
   }};
  }}).sort((a,b)=>(b.current_28d||0)-(a.current_28d||0));

  const ytdLookup={{}};(d.dominant_crimes||[]).forEach(r=>{{ytdLookup[r.crime]=r.ytd_pct_change;}});
  concernsOverride=computeConcernRows(prevAgg.byCategory,currAgg.byCategory,cityPrevAgg.byCategory,cityCurrAgg.byCategory,ytdLookup)
   .filter(r=>r.signal!=='Monitor').slice(0,8);

  const decisionTotals={{}};
  allCats.forEach(cat=>{{
   const purpose=decisionPurposeFor(cat);
   const curr=currAgg.byCategory[cat]||0;
   if(!decisionTotals[purpose])decisionTotals[purpose]={{incidents:0,offenses:{{}}}};
   decisionTotals[purpose].incidents+=curr;
   decisionTotals[purpose].offenses[cat]=(decisionTotals[purpose].offenses[cat]||0)+curr;
  }});
  decisionOverride=Object.keys(decisionTotals).map(purpose=>{{
   const info=decisionTotals[purpose];
   let domCat='—',domCount=0;
   Object.keys(info.offenses).forEach(cat=>{{if(info.offenses[cat]>domCount){{domCount=info.offenses[cat];domCat=cat;}}}});
   return{{purpose:purpose,incidents:info.incidents,dominant_offense:domCat,dominant_offense_count:domCount,dominant_offense_share:info.incidents?domCount/info.incidents:null}};
  }}).sort((a,b)=>b.incidents-a.incidents);

  shiftOverride=customShiftSummary(p,customRange.currStart,customRange.currEnd);
  timingOverride=customTimingSummary(p,customRange.currStart,customRange.currEnd);
  focusOverride=customFocusRows(p,customRange);
  hotspotOverride=customHotspotCounts(p,customRange);

  if(statusEl)statusEl.textContent='Custom range applied: '+customRange.currStart+' to '+customRange.currEnd+' vs '+customRange.prevStart+' to '+customRange.prevEnd+'.';
 }} else {{
  rec=d.recent||{{}};
  if(statusEl)statusEl.textContent='';
 }}
 const periodLabel=customRange?(customRange.currStart+' to '+customRange.currEnd):'the latest automated 14-day period';
 const comparisonLabel=customRange?(customRange.currStart+' to '+customRange.currEnd+' versus '+customRange.prevStart+' to '+customRange.prevEnd):'the latest 14 days versus the immediately preceding 14 days';
 const noteText={{
  focusNote:'Ranks the highest-activity spatial cells in Precinct '+p+' during '+periodLabel+'.',
  crimeNote:'Shows long-run and matched-YTD context alongside '+(customRange?'the selected operational comparison ('+comparisonLabel+')':'the automated recent 14-day direction')+'.',
  priorityNote:'Compares offense-level activity for '+comparisonLabel+', with city-period and matched-YTD context shown separately.',
  shiftNote:'Shows incident demand by shift during '+periodLabel+' and the dominant crime within each shift.',
  decisionNote:'Groups incidents during '+periodLabel+' into broad response purposes using the dashboard offense-to-purpose rules; this is an analytical grouping, not an official disposition.',
  timingNote:'Summarizes peak day, hour, time block, and shift during '+periodLabel+'.',
  hotspotNote:'Compares spatial concentration cells for '+comparisonLabel+'.'
 }};Object.keys(noteText).forEach(id=>{{const el=document.getElementById(id);if(el)el.textContent=noteText[id];}});

 document.getElementById('precinctCards').innerHTML=`
 <div class="card"><div class="label">${{customRange?'Current custom range':'Current 14 days'}}</div><div class="value">${{fmtN(customRange?rec.current_28d:rec.current_14d)}}</div></div>
 <div class="card"><div class="label">${{customRange?'Previous custom range':'Previous 14 days'}}</div><div class="value">${{fmtN(customRange?rec.previous_28d:rec.previous_14d)}}</div></div>
 <div class="card"><div class="label">Recent change</div><div class="value ${{pctClass(customRange?rec.pct_change_28d:rec.pct_change_14d)}}">${{fmtPct(customRange?rec.pct_change_28d:rec.pct_change_14d)}}</div></div>
 <div class="card"><div class="label">${{city.current_year}} vs ${{city.previous_year}} YTD</div><div class="value ${{pctClass(ov.pct_change_vs_previous)}}">${{fmtPct(ov.pct_change_vs_previous)}}</div><div class="note">${{fmtN(ov.incidents_current)}} current YTD</div></div>
 <div class="card"><div class="label">Priority concerns</div><div class="value">${{fmtN((concernsOverride||d.concerns||[]).length)}}</div><div class="note">${{fmtN((d.focus_locations||[]).length)}} focus locations</div></div>`;

 let s=[];
 if(ov.pct_change_vs_previous!==null&&ov.pct_change_vs_previous!==undefined)s.push(`Overall matched-YTD incidents are ${{Math.abs(Number(ov.pct_change_vs_previous)).toFixed(1)}}% ${{Number(ov.pct_change_vs_previous)<0?'below':'above'}} ${{city.previous_year}}.`);
 if(customRange && rec.pct_change_28d!==null && rec.pct_change_28d!==undefined)s.push(`Recent activity moved from ${{fmtN(rec.previous_28d)}} to ${{fmtN(rec.current_28d)}} incidents (${{fmtPct(rec.pct_change_28d)}}).`);
 else if(rec.pct_change_14d!==null&&rec.pct_change_14d!==undefined)s.push(`Recent activity moved from ${{fmtN(rec.previous_14d)}} to ${{fmtN(rec.current_14d)}} incidents (${{fmtPct(rec.pct_change_14d)}}).`);
 const concernsForBrief=concernsOverride||d.concerns;
 if(concernsForBrief&&concernsForBrief.length)s.push(`${{concernsForBrief[0].crime}} is the leading current attention signal (${{concernsForBrief[0].signal}}).`);
 const timingForBrief=timingOverride||d.timing;
 if(timingForBrief)s.push(`${{customRange?'Selected-period':'Recent'}} activity peaks on ${{timingForBrief.peak_day}} around ${{hourLabel(timingForBrief.peak_hour)}}, with ${{timingForBrief.peak_time_block}} as the busiest broad time block.`);
 if(d.focus_locations&&d.focus_locations.length)s.push(`The highest-ranked focus location is ${{d.focus_locations[0].neighborhood}} near ${{d.focus_locations[0].intersection}}.`);
 document.getElementById('precinctBrief').textContent=s.join(' ');
 const quickConcern=(concernsForBrief&&concernsForBrief.length)?concernsForBrief[0]:null;
 document.getElementById('topConcernQuick').textContent=quickConcern?(quickConcern.crime+' · '+quickConcern.signal):'No elevated signal';
 document.getElementById('timingQuick').textContent=timingForBrief?(timingForBrief.peak_day+' · '+hourLabel(timingForBrief.peak_hour)):'Timing unavailable';

 document.getElementById('buildMapBtn').href=analysisUrl(p,'map');
 document.getElementById('classicLinks').innerHTML=[
   assetLink(analysisUrl(p,'map'),'Open Interactive Precinct Map'),
   assetLink(a.shift_chart,'Shift Summary Chart'),
   assetLink(a.decision_chart,'Decision Purpose Chart'),
   assetLink(a.monthly_heatmap,'Monthly Trend Heatmap'),
   assetLink(a.violent_type_chart,'Violent Crime 14-Day Breakdown'),
   assetLink(a.violent_day_hour,'Violent Crime Day / Hour')
 ].filter(Boolean).join('');
 document.getElementById('modernLinks').innerHTML=[
   assetLink(analysisUrl(p,'trends'),'Crime Trend Analysis'),
   assetLink(analysisUrl(p,'priority'),'Priority / Emerging Concerns'),
   assetLink(analysisUrl(p,'timing'),'Temporal Analysis'),
   assetLink(analysisUrl(p,'hotspot'),'Hotspot Changes')
 ].filter(Boolean).join('');

 const fl=customRange?(focusOverride||[]).map((r,i)=>[
   fmtN(i+1),r.neighborhood,r.intersection,fmtN(r.curr),r.dominant,fmtShare(r.curr?r.dominant_count/r.curr:null)
 ]):(d.focus_locations||[]).map(r=>[
   fmtN(r.rank),r.neighborhood,r.intersection,fmtN(r.incident_count),r.dominant_offense,fmtShare(r.dominant_offense_share)
 ]);
 document.getElementById('focusLocations').innerHTML=simpleTable(['Rank','Neighborhood','Nearest intersection','Incidents','Dominant crime','Share'],fl);

 const dep=d.deployment||{{}};
 document.getElementById('deploymentSummary').innerHTML=`<div class="metric-list">
 <div class="metric"><span class="note">Total incidents</span><b>${{fmtN(dep.incidents)}}</b></div>
 <div class="metric"><span class="note">Spike-week incidents</span><b>${{fmtN(dep.spike_week_incidents)}}</b></div>
 <div class="metric"><span class="note">Spike incident share</span><b>${{fmtShare(dep.spike_incident_share)}}</b></div>
 <div class="metric"><span class="note">Neighborhoods covered</span><b>${{fmtN(dep.neighborhoods_covered)}}</b></div>
 <div class="metric"><span class="note">Dominant crime</span><b>${{dep.dominant_crime||'—'}}</b></div>
 <div class="metric"><span class="note">Dominant crime share</span><b>${{fmtShare(dep.dominant_crime_share)}}</b></div>
 </div>`;

 document.getElementById('improvementSummary').innerHTML=simpleTable(
 ['2024','2025','2026','vs 2025','vs 2024','Status','Score'],
 [[fmtN(ov.incidents_baseline),fmtN(ov.incidents_previous),fmtN(ov.incidents_current),fmtPct(ov.pct_change_vs_previous),fmtPct(ov.pct_change_vs_baseline),ov.improvement_status||'—',ov.improvement_score===null?'—':Number(ov.improvement_score).toFixed(2)]]
 );
 const wd=(d.worsening_drivers||[]).map(r=>[r.crime,fmtN(r.previous),fmtN(r.current),fmtPct(r.pct),r.trend]);
 const id=(d.improving_drivers||[]).map(r=>[r.crime,fmtN(r.previous),fmtN(r.current),fmtPct(r.pct),r.trend]);
 document.getElementById('worseningDrivers').innerHTML=simpleTable(['Crime',city.previous_year,city.current_year,'% change','Trend'],wd);
 document.getElementById('improvingDrivers').innerHTML=simpleTable(['Crime',city.previous_year,city.current_year,'% change','Trend'],id);

 const dc=(crimeRecordsOverride||d.dominant_crimes||[]).map(r=>[
   r.crime,
   fmtN(r.incident_count),
   fmtShare(r.share),
   fmtN(r.current_ytd),
   fmtPct(r.ytd_pct_change),
   r.trend||'—',
   fmtN(customRange?r.current_28d:r.current_14d),
   fmtPct(r.recent_pct_change)
 ]);
 document.getElementById('dominantCrimes').innerHTML=simpleTable(
   ['Crime','All-period incidents','Share',city.current_year+' YTD','YTD %chg','YTD trend',customRange?'Custom Range':'Current 14D',customRange?'Range %chg':'14D %chg'],
   dc
 );

 const cr=(concernsOverride||d.concerns||[]).map(r=>[
   r.crime,
   fmtN(customRange?r.previous_28d:r.previous_14d),
   fmtN(customRange?r.current_28d:r.current_14d),
   fmtPct(customRange?r.pct_change_28d:r.pct_change_14d),
   fmtPct(customRange?r.city_pct_change_28d:r.city_pct_change_14d),
   fmtPct(r.ytd_pct_change),
   `<span class="signal ${{signalClass(r.signal)}}">${{r.signal}}</span>`
 ]);
 document.getElementById('concernsTable').innerHTML=simpleTable(
   ['Crime',customRange?'Prev Range':'Prev 14D',customRange?'Current Range':'Current 14D',customRange?'Range %chg':'14D','City','YTD','Signal'],
   cr
 );

 const sh=(shiftOverride||d.shift_summary||[]).map(r=>[r.shift,fmtN(r.incidents),r.dominant_offense,fmtN(r.dominant_offense_count),fmtShare(r.dominant_offense_share)]);
 document.getElementById('shiftSummary').innerHTML=simpleTable(['Shift','Incidents','Dominant crime','Dominant count','Share'],sh);

 const ds=(decisionOverride||d.decision_summary||[]).map(r=>[r.purpose,fmtN(r.incidents),r.dominant_offense,fmtN(r.dominant_offense_count),fmtShare(r.dominant_offense_share)]);
 document.getElementById('decisionSummary').innerHTML=simpleTable(['Decision purpose','Incidents','Dominant crime','Dominant count','Share'],ds);

 const timingView=timingOverride||d.timing;
 if(timingView)document.getElementById('timingSummary').innerHTML=`<div class="metric-list"><div class="metric"><span class="note">Peak day</span><b>${{timingView.peak_day}}</b></div><div class="metric"><span class="note">Peak hour</span><b>${{hourLabel(timingView.peak_hour)}}</b></div><div class="metric"><span class="note">Busiest block</span><b>${{timingView.peak_time_block}}</b></div><div class="metric"><span class="note">Dominant shift</span><b>${{timingView.peak_shift}}</b></div></div>`;
 else document.getElementById('timingSummary').innerHTML='<div class="empty">Timing profile unavailable.</div>';

 const h=hotspotOverride||d.hotspots||{{new:0,emerging:0,persistent:0,declining:0}};
 document.getElementById('hotspotSummary').innerHTML=`<div class="metric-list"><div class="metric"><span class="note">New</span><b>${{h.new}}</b></div><div class="metric"><span class="note">Emerging</span><b>${{h.emerging}}</b></div><div class="metric"><span class="note">Persistent</span><b>${{h.persistent}}</b></div><div class="metric"><span class="note">Declining</span><b>${{h.declining}}</b></div></div>`;
}}

function resetToCity(){{document.getElementById('cityPrompt').style.display='block';document.getElementById('precinctOverview').style.display='none';}}
function clearCustomRange(){{customRange=null;const statusEl=document.getElementById('rangeStatus');if(statusEl)statusEl.textContent='';}}
select.addEventListener('change',()=>{{let p=select.value;p?renderPrecinct(p):resetToCity();}});
document.getElementById('rangeApplyBtn').addEventListener('click',()=>{{
 const p=select.value;if(!p)return;
 const ps=document.getElementById('rangePrevStart').value,pe=document.getElementById('rangePrevEnd').value;
 const cs=document.getElementById('rangeCurrStart').value,ce=document.getElementById('rangeCurrEnd').value;
 const statusEl=document.getElementById('rangeStatus');
 if(!ps||!pe||!cs||!ce){{if(statusEl)statusEl.textContent='Choose all four dates before applying.';return;}}
 if(ps>pe||cs>ce){{if(statusEl)statusEl.textContent='Each range needs a start on or before its end.';return;}}
 const availableThrough=String(city.data_through||'').slice(0,10);
 if(availableThrough && (pe>availableThrough||ce>availableThrough)){{if(statusEl)statusEl.textContent='Selected dates extend beyond the available RMS data. Data are available through '+availableThrough+'.';return;}}
 customRange={{prevStart:ps,prevEnd:pe,currStart:cs,currEnd:ce}};
 renderPrecinct(p);
}});
document.getElementById('rangeResetBtn').addEventListener('click',()=>{{
 clearCustomRange();
 const p=select.value;if(p)renderPrecinct(p);
}});
renderCity();resetToCity();
</script>
</body></html>"""
    out_path.write_text(html_text, encoding="utf-8")

def save_decision_dashboard_html(
    focus: pd.DataFrame,
    precinct_summary: pd.DataFrame,
    precinct_improvement: pd.DataFrame,
    offense_summary: pd.DataFrame,
    shift_summary: pd.DataFrame,
    decision_summary: pd.DataFrame,
    period_tag: str,
    out_path: Path,
) -> None:
    top_focus = focus.head(20).copy()
    top_focus["dominant_offense_share"] = (top_focus["dominant_offense_share"] * 100).round(1)
    top_focus["mean_spike_z"] = top_focus["mean_spike_z"].round(2)
    top_focus["score"] = top_focus["score"].round(3)

    precinct_table = precinct_summary.copy()
    precinct_table["dominant_crime_share"] = (precinct_table["dominant_crime_share"] * 100).round(1)
    precinct_table["spike_incident_share"] = (precinct_table["spike_incident_share"] * 100).round(1)

    offense_table = offense_summary.head(12).copy()
    shift_table = shift_summary.copy()
    shift_table["dominant_offense_share"] = (shift_table["dominant_offense_share"] * 100).round(1)
    decision_table = decision_summary.copy()
    decision_table["dominant_offense_share"] = (decision_table["dominant_offense_share"] * 100).round(1)

    improvement_table = precinct_improvement.copy()
    if not improvement_table.empty:
        improvement_table["pct_change_vs_previous"] = improvement_table["pct_change_vs_previous"].round(1)
        if "pct_change_vs_baseline" in improvement_table.columns:
            improvement_table["pct_change_vs_baseline"] = improvement_table["pct_change_vs_baseline"].round(1)

    focus_html = top_focus[
        [
            "focus_rank",
            "police_precinct",
            "neighborhood",
            "nearest_intersection",
            "incident_count",
            "dominant_offense",
            "dominant_offense_share",
            "mean_spike_z",
            "score",
        ]
    ].to_html(index=False, classes="table table-sm table-striped", border=0)

    precinct_html = precinct_table[
        [
            "police_precinct",
            "incidents",
            "spike_week_incidents",
            "spike_incident_share",
            "dominant_crime",
            "dominant_crime_share",
            "neighborhoods_covered",
        ]
    ].to_html(index=False, classes="table table-sm table-striped", border=0)

    offense_html = offense_table[["offense_category", "incident_count"]].to_html(
        index=False, classes="table table-sm table-striped", border=0
    )
    shift_html = shift_table[
        ["shift_window", "shift_total_incidents", "dominant_offense", "dominant_offense_count", "dominant_offense_share"]
    ].to_html(index=False, classes="table table-sm table-striped", border=0)
    decision_html = decision_table[
        ["decision_purpose", "purpose_total_incidents", "dominant_offense", "dominant_offense_count", "dominant_offense_share"]
    ].to_html(index=False, classes="table table-sm table-striped", border=0)

    improvement_columns = [
        "precinct_norm",
        "incidents_current",
        "incidents_previous",
        "change_vs_previous",
        "pct_change_vs_previous",
        "improvement_status",
        "improvement_score",
    ]
    if "incidents_baseline" in improvement_table.columns:
        improvement_columns.extend(["incidents_baseline", "pct_change_vs_baseline"])
    improvement_html = improvement_table[improvement_columns].to_html(
        index=False, classes="table table-sm table-striped", border=0
    )

    top_precinct = precinct_summary.iloc[0] if not precinct_summary.empty else None
    top_focus_row = top_focus.iloc[0] if not top_focus.empty else None
    top_improving_precinct = improvement_table.iloc[0] if not improvement_table.empty else None

    html = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
    <title>Detroit Crime Operations Dashboard {period_tag}</title>
  <style>
    body {{ font-family: 'Helvetica Neue', Arial, sans-serif; margin: 24px; background: #f6f8fb; color: #0f172a; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin-bottom: 20px; }}
    .card {{ background: #ffffff; border: 1px solid #dbe3ef; border-radius: 10px; padding: 14px; }}
    .card h3 {{ margin: 0; font-size: 0.9rem; color: #475569; }}
    .card p {{ margin: 6px 0 0 0; font-size: 1.35rem; font-weight: 700; }}
    h1 {{ margin-bottom: 6px; }}
    h2 {{ margin-top: 28px; margin-bottom: 10px; }}
    .links a {{ margin-right: 16px; color: #0b5cab; text-decoration: none; font-weight: 600; }}
    .links a:hover {{ text-decoration: underline; }}
    .table {{ width: 100%; border-collapse: collapse; background: #fff; }}
    .table th, .table td {{ border: 1px solid #e2e8f0; padding: 6px 8px; text-align: left; font-size: 0.86rem; }}
    .table th {{ background: #e2e8f0; }}
  </style>
</head>
<body>
    <h1>Detroit Crime Operations Dashboard ({period_tag})</h1>
  <p>Decision support for precinct deployment: focus locations, dominant crimes, and spike pressure.</p>

  <div class=\"cards\">
    <div class=\"card\"><h3>Total Incidents</h3><p>{int(offense_summary['incident_count'].sum()):,}</p></div>
    <div class=\"card\"><h3>Total Focus Cells</h3><p>{len(focus):,}</p></div>
    <div class=\"card\"><h3>Top Focus Precinct</h3><p>{top_focus_row['police_precinct'] if top_focus_row is not None else 'N/A'}</p></div>
    <div class=\"card\"><h3>Highest Volume Precinct</h3><p>{top_precinct['police_precinct'] if top_precinct is not None else 'N/A'}</p></div>
  </div>

  <div class=\"links\">
        <a href=\"../Images/detroit_crime_interactive_dashboard_{period_tag}.html\" target=\"_blank\">Open Interactive Map Dashboard</a>
                <a href="../Images/detroit_shift_incidents_{period_tag}.png" target="_blank">Open Shift Summary Chart</a>
                <a href="../Images/detroit_decision_purpose_incidents_{period_tag}.png" target="_blank">Open Decision Purpose Chart</a>
            <a href="../Images/detroit_precinct_monthly_trend_heatmap_{period_tag}.png" target="_blank">Open Precinct Monthly Trend Heatmap</a>
    <a href="../Images/detroit_violent_crime_14d_by_day_hour_{period_tag}.png" target="_blank">Open Violent Crime 14-Day Day/Hour</a>
  </div>

    <h2>Layer Color Key (Decision Purpose)</h2>
    <table class="table table-sm table-striped" style="max-width: 760px;">
        <thead><tr><th>Purpose</th><th>Color Theme</th><th>Typical Use</th></tr></thead>
        <tbody>
            <tr><td>Preventive Patrol</td><td>Blue</td><td>Deterrence and patrol saturation in theft and property-crime zones</td></tr>
            <tr><td>Investigations</td><td>Purple</td><td>Case-building, fraud follow-up, and detective-led targeting</td></tr>
            <tr><td>Community Response</td><td>Red</td><td>Violence interruption, victim support, and high-risk intervention</td></tr>
        </tbody>
    </table>

  <h2>Top 20 Operational Focus Locations</h2>
  <p>Dominant offense share is shown in percent.</p>
  {focus_html}

  <h2>Precinct Summary for Deployment</h2>
  <p>Spike incident share indicates pressure from spike-week workload.</p>
  {precinct_html}

  <h2>Precinct Improvement (Matched Year-to-Date)</h2>
  <p>Positive improvement scores mean fewer incidents than the comparison years through the same calendar date.</p>
  {improvement_html}

  <h2>Dominant Crime Types Citywide</h2>
  {offense_html}

    <h2>Shift-Level Demand and Dominant Crime</h2>
    <p>Dominant offense share is shown in percent.</p>
    {shift_html}

    <h2>Decision Purpose Workload and Dominant Crime</h2>
    <p>Dominant offense share is shown in percent.</p>
    {decision_html}
</body>
</html>
"""

    out_path.write_text(html, encoding="utf-8")


def save_spike_summary_plot(weekly: pd.DataFrame, out_path: Path) -> None:
    spikes = weekly[weekly["is_spike"]]

    if spikes.empty:
        plt.figure(figsize=(8, 4))
        plt.text(0.5, 0.5, "No spike weeks detected with current rules", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_path, dpi=220)
        plt.close()
        return

    top = (
        spikes.groupby("neighborhood", as_index=False)
        .agg(spike_weeks=("is_spike", "sum"), max_weekly_incidents=("incident_count", "max"))
        .sort_values(["spike_weeks", "max_weekly_incidents"], ascending=False)
        .head(12)
    )

    plt.figure(figsize=(11, 6))
    sns.barplot(data=top, y="neighborhood", x="spike_weeks", color="#1f78b4")
    plt.title("Neighborhoods With Most Crime Spike Weeks")
    plt.xlabel("Number of Spike Weeks")
    plt.ylabel("Neighborhood")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def main() -> None:
    # OPERATIONS LANDING FINAL BUILD:
    # Generate only the outputs that support the dashboard narrative and auditability.
    # Redundant standalone charts/maps remain available as helper functions above,
    # but are intentionally not generated in the final portfolio workflow.
    df = load_data(DATASET_DIR)
    years_available = sorted(pd.Series(df["incident_year"].dropna().astype(int).unique()).tolist())
    current_year = max(years_available) if years_available else 2026
    previous_year = current_year - 1
    baseline_year = current_year - 2 if years_available and (current_year - 2) in years_available else None
    period_tag = f"{min(years_available)}_{max(years_available)}" if years_available else str(current_year)

    # Core analytical pipeline that directly feeds the interactive story.
    weekly = detect_weekly_spikes(df)
    spike_points = build_spike_points(df, weekly)
    focus_locations = build_focus_locations(df, weekly)
    city_ytd, precinct_ytd = build_ytd_comparison(
        df,
        current_year=current_year,
        previous_year=previous_year,
    )
    precinct_improvement = build_precinct_improvement_table(
        df,
        current_year=current_year,
        previous_year=previous_year,
        baseline_year=baseline_year,
    )
    precinct_crime_14d = build_precinct_crime_14d_comparison(
        df,
        current_year=current_year,
    )
    precinct_crime_trends = build_precinct_crime_trend_table(
        df,
        current_year=current_year,
        previous_year=previous_year,
        baseline_year=baseline_year,
    )
    priority_concerns = build_priority_emerging_concerns(
        precinct_crime_trends,
        precinct_crime_14d,
    )
    temporal_patterns = build_temporal_pattern_profiles(df, current_year=current_year)
    temporal_summary = temporal_patterns["summary"]
    temporal_matrix = temporal_patterns["matrix"]
    hotspot_change = build_hotspot_persistence_change(df, current_year=current_year)

    # The combined dashboard needs records calculated within, rather than merely
    # labeled with, the selected neighborhood. Keep the citywide records as ALL.
    precinct_improvement = add_neighborhood_scopes(
        df,
        precinct_improvement,
        lambda scoped: build_precinct_improvement_table(
            scoped, current_year, previous_year, baseline_year
        ),
    )
    precinct_crime_14d = add_neighborhood_scopes(
        df,
        precinct_crime_14d,
        lambda scoped: build_precinct_crime_14d_comparison(scoped, current_year),
    )
    precinct_crime_trends = add_neighborhood_scopes(
        df,
        precinct_crime_trends,
        lambda scoped: build_precinct_crime_trend_table(
            scoped, current_year, previous_year, baseline_year
        ),
    )
    priority_frames = [priority_concerns.assign(neighborhood_scope="ALL")]
    temporal_summary_frames = [temporal_summary.assign(neighborhood_scope="ALL")]
    temporal_matrix_frames = [temporal_matrix.assign(neighborhood_scope="ALL")]
    hotspot_frames = [hotspot_change.assign(neighborhood_scope="ALL")]
    for neighborhood, scoped in df.groupby("neighborhood", sort=True):
        scoped_recent = build_precinct_crime_14d_comparison(scoped, current_year)
        scoped_trends = build_precinct_crime_trend_table(
            scoped, current_year, previous_year, baseline_year
        )
        priority_frames.append(
            build_priority_emerging_concerns(scoped_trends, scoped_recent).assign(
                neighborhood_scope=str(neighborhood)
            )
        )
        scoped_temporal = build_temporal_pattern_profiles(scoped, current_year)
        temporal_summary_frames.append(
            scoped_temporal["summary"].assign(neighborhood_scope=str(neighborhood))
        )
        temporal_matrix_frames.append(
            scoped_temporal["matrix"].assign(neighborhood_scope=str(neighborhood))
        )
        hotspot_frames.append(
            build_hotspot_persistence_change(scoped, current_year).assign(
                neighborhood_scope=str(neighborhood)
            )
        )
    priority_concerns = pd.concat(priority_frames, ignore_index=True)
    temporal_summary = pd.concat(temporal_summary_frames, ignore_index=True)
    temporal_matrix = pd.concat(temporal_matrix_frames, ignore_index=True)
    hotspot_change = pd.concat(hotspot_frames, ignore_index=True)

    # Primary entry point + interactive drill-down.
    operations_overview_html = DOCS_DIR / f"detroit_crime_operations_overview_{period_tag}.html"
    combined_dashboard_html = IMAGES_DIR / f"detroit_crime_interactive_dashboard_{period_tag}.html"
    area_map_builder_html = IMAGES_DIR / f"area_crime_map_builder_{period_tag}.html"

    # Preserve the useful legacy drill-down ideas, but regenerate each asset
    # separately for every operational precinct.
    precinct_assets = generate_precinct_drilldown_assets(
        df=df,
        focus_locations=focus_locations,
        current_year=current_year,
        period_tag=period_tag,
    )

    # Standalone area + crime-type map builder (additive, removable layers).
    area_crime_data = build_area_crime_points(df)
    save_area_crime_map_builder_html(area_crime_data, area_map_builder_html)

    # Lean audit outputs: each one directly supports a dashboard statement.
    weekly_csv = DOCS_DIR / f"weekly_neighborhood_counts_and_spikes_{period_tag}.csv"
    focus_csv = DOCS_DIR / f"focus_locations_for_precinct_action_{period_tag}.csv"
    city_ytd_csv = DOCS_DIR / f"city_ytd_{previous_year}_vs_{current_year}.csv"
    precinct_ytd_csv = DOCS_DIR / f"precinct_ytd_{previous_year}_vs_{current_year}.csv"
    improvement_year_tag = (
        f"{baseline_year}_{previous_year}_vs_{current_year}"
        if baseline_year is not None
        else f"{previous_year}_vs_{current_year}"
    )
    precinct_improvement_csv = DOCS_DIR / f"precinct_improvement_ytd_{improvement_year_tag}.csv"
    precinct_crime_14d_csv = DOCS_DIR / f"precinct_crime_14d_comparison_{current_year}.csv"
    precinct_crime_trends_csv = DOCS_DIR / f"precinct_crime_type_trends_ytd_{improvement_year_tag}.csv"
    priority_concerns_csv = DOCS_DIR / f"priority_emerging_concerns_{current_year}.csv"
    temporal_summary_csv = DOCS_DIR / f"temporal_pattern_summary_{current_year}.csv"
    temporal_matrix_csv = DOCS_DIR / f"temporal_day_time_matrix_{current_year}.csv"
    hotspot_change_csv = DOCS_DIR / f"hotspot_persistence_change_{current_year}.csv"

    save_operations_landing_html(
        df=df,
        weekly=weekly,
        focus_locations=focus_locations,
        city_ytd=city_ytd,
        precinct_improvement=precinct_improvement,
        precinct_crime_14d=precinct_crime_14d,
        precinct_crime_trends=precinct_crime_trends,
        priority_concerns=priority_concerns,
        temporal_summary=temporal_summary,
        hotspot_change=hotspot_change,
        precinct_assets=precinct_assets,
        baseline_year=baseline_year,
        current_year=current_year,
        previous_year=previous_year,
        period_tag=period_tag,
        map_filename=combined_dashboard_html.name,
        area_map_filename=area_map_builder_html.name,
        out_path=operations_overview_html,
    )
    # Keep GitHub Pages homepage synchronized with the latest operations overview.
    index_html = BASE_DIR / "index.html"
    homepage_html = operations_overview_html.read_text(encoding="utf-8")
    homepage_html = homepage_html.replace("../Images/", "Images/")
    homepage_html = homepage_html.replace("./Images/", "Images/")
    homepage_html = homepage_html.replace("/Images/", "Images/")
    index_html.write_text(
        homepage_html,
        encoding="utf-8",
    )
    save_combined_interactive_dashboard(
        df,
        weekly,
        spike_points,
        focus_locations,
        combined_dashboard_html,
        precinct_improvement=precinct_improvement,
        precinct_crime_trends=precinct_crime_trends,
        precinct_crime_14d=precinct_crime_14d,
        priority_concerns=priority_concerns,
        temporal_summary=temporal_summary,
        temporal_matrix=temporal_matrix,
        hotspot_change=hotspot_change,
        current_year=current_year,
        previous_year=previous_year,
        baseline_year=baseline_year,
    )

    weekly.to_csv(weekly_csv, index=False)
    focus_locations.to_csv(focus_csv, index=False)
    city_ytd.to_csv(city_ytd_csv, index=False)
    precinct_ytd.to_csv(precinct_ytd_csv, index=False)
    precinct_improvement.to_csv(precinct_improvement_csv, index=False)
    precinct_crime_14d.to_csv(precinct_crime_14d_csv, index=False)
    precinct_crime_trends.to_csv(precinct_crime_trends_csv, index=False)
    priority_concerns.to_csv(priority_concerns_csv, index=False)
    temporal_summary.to_csv(temporal_summary_csv, index=False)
    temporal_matrix.to_csv(temporal_matrix_csv, index=False)
    hotspot_change.to_csv(hotspot_change_csv, index=False)

    source_files = sorted(df["source_file"].dropna().unique().tolist())
    total_spikes = int(weekly["is_spike"].sum())
    print("Analysis complete — PRECINCT CONTROL CENTER FINAL BUILD.")
    print(f"Source files loaded: {len(source_files)}")
    print(f"Years covered: {', '.join(map(str, years_available)) if years_available else 'Unknown'}")
    print(f"Incidents analyzed: {len(df):,}")
    print(f"Weekly neighborhood spikes detected: {total_spikes}")
    print("\nPrimary entry point:")
    print(f"  - {operations_overview_html}")
    print("\nInteractive drill-down:")
    print(f"  - {combined_dashboard_html}")
    print("\nPrecinct-specific drill-down folders:")
    for precinct in sorted(precinct_assets):
        print(f"  - {IMAGES_DIR / ('precinct_' + precinct)}")
    print("\nAudit outputs:")
    for path in [
        weekly_csv,
        focus_csv,
        city_ytd_csv,
        precinct_ytd_csv,
        precinct_improvement_csv,
        precinct_crime_14d_csv,
        precinct_crime_trends_csv,
        priority_concerns_csv,
        temporal_summary_csv,
        temporal_matrix_csv,
        hotspot_change_csv,
    ]:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
