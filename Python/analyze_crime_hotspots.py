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


BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = BASE_DIR / "Dataset"
DATASET_GLOB = "RMS_Crime_Incidents_*.csv"
IMAGES_DIR = BASE_DIR / "Images"
DOCS_DIR = BASE_DIR / "Documentation"

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

    df = df.dropna(subset=required_cols).copy()
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

    description = df.get("offense_description", pd.Series("", index=df.index)).astype(str)
    category = df.get("offense_category", pd.Series("", index=df.index)).astype(str).str.upper()
    df["is_gun_related"] = description.str.contains("GUN|FIREARM|WEAPON|SHOT", case=False, na=False)
    df["is_property_related"] = category.isin(
        ["LARCENY", "BURGLARY", "STOLEN VEHICLE", "STOLEN PROPERTY", "DAMAGE TO PROPERTY"]
    )
    df["is_larceny_related"] = category.eq("LARCENY")

    return df


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
    crime_type_values: list[str],
    precinct_improvement: pd.DataFrame | None = None,
    precinct_crime_trends: pd.DataFrame | None = None,
    precinct_crime_28d: pd.DataFrame | None = None,
    priority_concerns: pd.DataFrame | None = None,
    temporal_summary: pd.DataFrame | None = None,
    temporal_matrix: pd.DataFrame | None = None,
    hotspot_change: pd.DataFrame | None = None,
    precinct_bounds: dict[str, list[list[float]]] | None = None,
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

    precinct_options = "".join(
        [f'<option value="{html.escape(p, quote=True)}">{html.escape(p)}</option>' for p in precincts]
    )

    category_options = [
        '<option value="Category Focus | Gun-Related">Category Focus: Gun-Related</option>',
        '<option value="Category Focus | Property Crime">Category Focus: Property Crime</option>',
        '<option value="Category Focus | Larceny">Category Focus: Larceny</option>',
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
            key = str(row.get("precinct_norm", ""))
            if not key:
                continue
            overall_data[key] = {
                "precinct": key,
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

    crime_28d_records = []
    if precinct_crime_28d is not None and not precinct_crime_28d.empty:
        wanted_28d = [
            "precinct_norm",
            "offense_category",
            "previous_28d",
            "current_28d",
            "pct_change_28d",
            "recent_movement",
            "city_pct_change_28d",
            "previous_28d_start",
            "previous_28d_end",
            "current_28d_start",
            "current_28d_end",
        ]
        for _, row in precinct_crime_28d.iterrows():
            rec = {col: clean_number(row.get(col)) for col in wanted_28d}
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
            for col in ["previous_28d_start", "previous_28d_end", "current_28d_start", "current_28d_end"]:
                rec[col] = str(row.get(col, ""))
            crime_28d_records.append(rec)

    priority_records = []
    if priority_concerns is not None and not priority_concerns.empty:
        wanted_priority = [
            "precinct_norm",
            "offense_category",
            "previous_28d",
            "current_28d",
            "change_28d",
            "pct_change_28d",
            "city_pct_change_28d",
            "city_gap_28d",
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
                "period", "precinct_norm", "selection_type", "selection_name",
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
            "precinct_norm", "selection_type", "selection_name", "h3_cell",
            "previous_28d", "current_28d", "change_28d", "pct_change_28d",
            "hotspot_status", "hotspot_score", "previous_hotspot_threshold",
            "current_hotspot_threshold", "neighborhood", "nearest_intersection",
            "latitude", "longitude", "previous_28d_start", "previous_28d_end",
            "current_28d_start", "current_28d_end",
        ]
        for _, row in hotspot_change.iterrows():
            rec = {col: clean_number(row.get(col)) for col in wanted_hotspot}
            for col in [
                "precinct_norm", "selection_type", "selection_name", "h3_cell",
                "hotspot_status", "neighborhood", "nearest_intersection",
                "previous_28d_start", "previous_28d_end", "current_28d_start", "current_28d_end",
            ]:
                rec[col] = str(row.get(col, ""))
            hotspot_change_records.append(rec)

    overall_json = json.dumps(overall_data, ensure_ascii=False).replace("</", "<\\/")
    crime_trend_json = json.dumps(crime_trend_records, ensure_ascii=False).replace("</", "<\\/")
    crime_28d_json = json.dumps(crime_28d_records, ensure_ascii=False).replace("</", "<\\/")
    priority_json = json.dumps(priority_records, ensure_ascii=False).replace("</", "<\\/")
    temporal_summary_json = json.dumps(temporal_summary_records, ensure_ascii=False).replace("</", "<\\/")
    temporal_matrix_json = json.dumps(temporal_matrix_records, ensure_ascii=False).replace("</", "<\\/")
    hotspot_change_json = json.dumps(hotspot_change_records, ensure_ascii=False).replace("</", "<\\/")

    year_labels = {
        "current": int(current_year) if current_year is not None else None,
        "previous": int(previous_year) if previous_year is not None else None,
        "baseline": int(baseline_year) if baseline_year is not None else None,
    }
    year_json = json.dumps(year_labels)
    precinct_bounds_json = json.dumps(precinct_bounds or {}, ensure_ascii=False)

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
        <div style="min-width:300px;flex:3;"><div style="font-weight:800;margin-bottom:4px;">Crime / Category</div>
          <select id="cpCategorySelect" style="width:100%;padding:7px;border:1px solid #94a3b8;border-radius:6px;">
            <option value="">All Crime</option>{category_options_html}</select></div>
        <div style="min-width:190px;flex:1;"><div style="font-weight:800;margin-bottom:4px;">Analysis View</div>
          <select id="cpPeriodSelect" style="width:100%;padding:7px;border:1px solid #94a3b8;border-radius:6px;">
            <option value="Operational">Operational Summary</option>
            <option value="Recent">Recent 28-Day Emphasis</option>
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
          <div style="color:#64748b;font-size:11px;max-width:270px;">Use the Leaflet layer button at the upper-right of the map for the optional latest-28-day incident point layer and alternate basemaps.</div>
        </div>
      </details>
      <div id="cpSelectorStatus" style="margin-top:8px;font-size:11px;color:#334155;">Detroit Overview | All Crime | Operational Summary</div>
      <div id="cpKpiStrip" style="margin-top:8px;display:grid;grid-template-columns:repeat(4,minmax(135px,1fr));gap:8px;"></div>
      <div id="cpExecutiveCard" style="margin-top:8px;padding:9px 11px;border:1px solid #cbd5e1;border-radius:8px;background:#ffffff;">
        <div class="exec-title">Executive Summary</div><div style="margin-top:3px;color:#64748b;">Select a precinct and/or crime type to generate a concise what–when–where–change summary.</div>
      </div>
      <div id="cpPriorityCard" style="margin-top:8px;padding:8px 10px;border:1px solid #fed7aa;border-radius:8px;background:#fff7ed;">
        <b>Priority / Emerging Concerns</b><div style="margin-top:3px;color:#64748b;">Ranks crime signals using recent volume, absolute change, 28-day percentage change, citywide divergence, and YTD direction. Small baselines are kept visible but do not rank on percentage alone.</div>
      </div>
      <div id="cpTimingCard" style="margin-top:8px;padding:8px 10px;border:1px solid #bae6fd;border-radius:8px;background:#f0f9ff;">
        <b>When is it happening?</b><div style="margin-top:3px;color:#64748b;">Shows peak day, hour, shift, and day/time concentration for the current selection, with YTD context and the latest 28 days.</div>
      </div>
      <div id="cpHotspotCard" style="margin-top:8px;padding:8px 10px;border:1px solid #ddd6fe;border-radius:8px;background:#faf5ff;">
        <b>Where is it changing?</b><div style="margin-top:3px;color:#64748b;">Compares H3 hotspot locations across consecutive 28-day periods and identifies persistent, new, emerging, and declining concentrations for the current selection.</div>
      </div>
      <div id="cpTrendCard" style="margin-top:8px;padding:8px 10px;border:1px solid #e2e8f0;border-radius:8px;background:#f8fafc;">
        Select a precinct and/or crime type to see matched YTD trend details and recent 28-day movement. Citywide results are context; precinct results are the operational workload.
      </div>
      <div style="margin-top:4px;font-size:11px;color:#64748b;">YTD Trend uses the latest current-year incident date as the cutoff and compares the same calendar period in prior years. ±2% is treated as Stable. Recent movement compares the latest 28 days with the immediately preceding 28 days.</div>
    </div>
    <script>
    (function() {{
      var overallData = {overall_json};
      var crimeTrendData = {crime_trend_json};
      var crime28dData = {crime_28d_json};
      var priorityData = {priority_json};
      var temporalSummaryData = {temporal_summary_json};
      var temporalMatrixData = {temporal_matrix_json};
      var hotspotChangeData = {hotspot_change_json};
      var mapObjectName = "{m.get_name()}";
      var years = {year_json};
      var precinctBounds = {precinct_bounds_json};

      function normalizeLayerName(text) {{ var clean=(text||'').trim(); var idx=clean.lastIndexOf(' ('); return idx>0?clean.slice(0,idx).trim():clean; }}
      function eachOverlayCheckbox(callback) {{ document.querySelectorAll('.leaflet-control-layers-overlays label').forEach(function(label) {{ var cb=label.querySelector('input[type="checkbox"]'); if(cb) callback(cb,(label.innerText||'').trim(),normalizeLayerName(label.innerText||'')); }}); }}
      function setChecked(cb, yes) {{ if(cb.checked!==yes) cb.click(); }}
      function setExclusiveByPrefix(prefix, selected) {{ eachOverlayCheckbox(function(cb,_t,b) {{ if(b.startsWith(prefix)) setChecked(cb, Boolean(selected && b===selected)); }}); }}
      function fmtN(v) {{ return (v===null || v===undefined || Number.isNaN(Number(v))) ? '—' : Number(v).toLocaleString(); }}
      function fmtPct(v) {{ if(v===null || v===undefined || Number.isNaN(Number(v))) return '—'; var n=Number(v); return (n>0?'+':'')+n.toFixed(1)+'%'; }}
      function trendClassName(t) {{ if((t||'').includes('Improving')) return 'trend-down'; if((t||'').includes('Worsening')) return 'trend-up'; return 'trend-stable'; }}
      function trendMatches(actual, requested) {{ if(!requested) return true; if(requested==='Improving') return actual==='Improving'||actual==='Consistently Improving'; if(requested==='Worsening') return actual==='Worsening'||actual==='Consistently Worsening'; return actual===requested; }}
      function selectedCrime() {{ var raw=(document.getElementById('cpCategorySelect')||{{value:''}}).value; return raw.startsWith('Crime Type | ')?raw.replace('Crime Type | ',''):''; }}
      function rowsFor(precinct, crime, trend) {{ return crimeTrendData.filter(function(r) {{ return (!precinct||r.precinct_norm===precinct) && (!crime||r.offense_category===crime) && trendMatches(r.trend_class,trend); }}); }}
      function rows28For(precinct, crime) {{ return crime28dData.filter(function(r) {{ return (!precinct||r.precinct_norm===precinct) && (!crime||r.offense_category===crime); }}); }}
      function recentClassName(t) {{ if(t==='Increasing') return 'trend-up'; if(t==='Decreasing') return 'trend-down'; return 'trend-stable'; }}
      function recentTableHtml(rows, firstCol, firstLabel, limit) {{
        var use=rows.slice(0,limit||12); if(!use.length) return '<div style="color:#64748b;">No 28-day comparison records for this selection.</div>';
        var h='<table><thead><tr><th>'+firstLabel+'</th><th>Previous 28D</th><th>Current 28D</th><th>28D %chg</th><th>City %chg</th><th>Recent</th></tr></thead><tbody>';
        use.forEach(function(r) {{ h+='<tr><td>'+r[firstCol]+'</td><td>'+fmtN(r.previous_28d)+'</td><td>'+fmtN(r.current_28d)+'</td><td>'+fmtPct(r.pct_change_28d)+'</td><td>'+fmtPct(r.city_pct_change_28d)+'</td><td class="'+recentClassName(r.recent_movement)+'">'+r.recent_movement+'</td></tr>'; }});
        return h+'</tbody></table>';
      }}
      function recentWindowText(rows) {{ if(!rows.length) return ''; var r=rows[0]; return ' | Recent window '+r.current_28d_start+' to '+r.current_28d_end+' vs '+r.previous_28d_start+' to '+r.previous_28d_end; }}
      function interpretationHtml(ytdRows, recentRows, precinct, crime) {{
        if(!ytdRows.length || !recentRows.length) return '';
        var y=ytdRows[0], r=recentRows[0];
        var recent=Number(r.pct_change_28d), city=Number(r.city_pct_change_28d), ytd=Number(y.pct_change_vs_previous);
        var recentValid=!Number.isNaN(recent), cityValid=!Number.isNaN(city), ytdValid=!Number.isNaN(ytd);
        var signal='MIXED', cls='trend-stable';
        if(recentValid && recent < -2) {{ signal='IMPROVING'; cls='trend-down'; }}
        else if(recentValid && recent > 2) {{ signal='WORSENING'; cls='trend-up'; }}
        else if(recentValid) {{ signal='STABLE'; cls='trend-stable'; }}
        var subject='Recent '+crime.toLowerCase()+' incidents in Precinct '+precinct;
        var text=subject+' '+(recent<0?'decreased ':'increased ')+(recent<0?Math.abs(recent).toFixed(1)+'%':fmtPct(recent))+ ' ('+fmtN(r.previous_28d)+' → '+fmtN(r.current_28d)+')';
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
      function priorityRowsFor(precinct, crime) {{ return priorityData.filter(function(r) {{ return (!precinct||r.precinct_norm===precinct) && (!crime||r.offense_category===crime); }}); }}
      function priorityTableHtml(rows, limit) {{
        var use=rows.slice(0,limit||8); if(!use.length) return '<div style="color:#64748b;margin-top:4px;">No priority signals for this selection.</div>';
        var h='<table><thead><tr><th>Precinct / Crime</th><th>Prev 28D</th><th>Current 28D</th><th>Abs Δ</th><th>28D %chg</th><th>City %chg</th><th>YTD %chg</th><th>Score</th><th>Signal</th></tr></thead><tbody>';
        use.forEach(function(r) {{ var label='P'+r.precinct_norm+' — '+r.offense_category; h+='<tr><td>'+label+'</td><td>'+fmtN(r.previous_28d)+'</td><td>'+fmtN(r.current_28d)+'</td><td>'+fmtN(r.change_28d)+'</td><td>'+fmtPct(r.pct_change_28d)+'</td><td>'+fmtPct(r.city_pct_change_28d)+'</td><td>'+fmtPct(r.pct_change_vs_previous)+'</td><td>'+Number(r.priority_score||0).toFixed(1)+'</td><td class="'+priorityClassName(r.priority_signal)+'">'+r.priority_signal+'</td></tr>'; }});
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

        // Change: use matched YTD + recent 28-day movement when the scope supports it.
        if(precinct && crime) {{
          var yrows=rowsFor(precinct,crime,''); var rrows=rows28For(precinct,crime);
          if(yrows.length) bullets.push('<b>Long-term:</b> '+crime+' is <span class="'+trendClassName(yrows[0].trend_class)+'">'+yrows[0].trend_class+'</span> YTD ('+fmtPct(yrows[0].pct_change_vs_previous)+' vs '+years.previous+').');
          if(rrows.length) {{
            var rr=rrows[0]; bullets.push('<b>Recent:</b> '+fmtN(rr.previous_28d)+' → '+fmtN(rr.current_28d)+' in consecutive 28-day periods ('+fmtPct(rr.pct_change_28d)+'), versus '+fmtPct(rr.city_pct_change_28d)+' citywide.');
            if(Number(rr.pct_change_28d)<-2) signal='IMPROVING'; else if(Number(rr.pct_change_28d)>2) signal='WORSENING';
          }}
        }} else if(precinct && scope.type==='All') {{
          var ov=overallData[precinct];
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

        // When: latest 28-day timing profile for the exact current map-analysis scope.
        var recent=timingSummary('Recent 28D',precinct,scope);
        if(recent) bullets.push('<b>When:</b> Peak day is '+recent.peak_day+', peak hour '+hourLabel(recent.peak_hour)+', with '+recent.peak_time_block+' as the busiest time block and '+recent.peak_shift+' as the dominant shift.');

        // Where: summarize hotspot state without repeating the detail table.
        var hrows=hotspotRowsFor(precinct,scope);
        if(hrows.length) {{
          var ne=hrows.filter(function(r){{return r.hotspot_status==='New Hotspot'||r.hotspot_status==='Emerging Hotspot';}}).length;
          var pe=hrows.filter(function(r){{return r.hotspot_status==='Persistent Hotspot';}}).length;
          var de=hrows.filter(function(r){{return r.hotspot_status==='Declining Hotspot';}}).length;
          bullets.push('<b>Where:</b> '+ne+' new/emerging, '+pe+' persistent, and '+de+' declining hotspot cells in the latest 28-day comparison.');
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
        var ov=overallData[precinct];
        var recent=rows28For(precinct,'');
        var prev=0,curr=0; recent.forEach(function(r){{prev+=Number(r.previous_28d||0);curr+=Number(r.current_28d||0);}});
        var rpct=prev>0?100*(curr-prev)/prev:null;
        var actionable=priorityRowsFor(precinct,'').filter(function(r){{return r.priority_signal==='High Priority'||r.priority_signal==='Emerging Concern'||r.priority_signal==='Watch';}}).length;
        function k(label,value,sub){{return '<div style="border:1px solid #dbeafe;border-radius:8px;padding:7px 9px;background:#f8fbff;"><div style="font-size:10px;text-transform:uppercase;letter-spacing:.04em;color:#64748b;font-weight:700;">'+label+'</div><div style="font-size:18px;font-weight:800;color:#0f172a;margin-top:2px;">'+value+'</div><div style="font-size:10px;color:#64748b;">'+sub+'</div></div>';}}
        strip.innerHTML=k('YTD incidents',ov?fmtN(ov.incidents_current):'—','Precinct '+precinct)+
          k('vs '+years.previous+' YTD',ov?fmtPct(ov.pct_change_vs_previous):'—',ov?ov.trend_class:'')+
          k('Recent 28D',fmtN(curr),rpct===null?'No prior baseline':fmtPct(rpct)+' vs prior 28D')+
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
        var improvements=rows.filter(function(r) {{ return r.priority_signal==='Recent Improvement'; }}).sort(function(a,b) {{ return Number(a.pct_change_28d||0)-Number(b.pct_change_28d||0); }});
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
      function timingSummary(period,precinct,scope) {{
        var p=precinct||'ALL';
        var rows=temporalSummaryData.filter(function(r){{return r.period===period && r.precinct_norm===p && r.selection_type===scope.type && r.selection_name===scope.name;}});
        return rows.length?rows[0]:null;
      }}
      function timingMatrix(period,precinct,scope) {{
        var p=precinct||'ALL';
        return temporalMatrixData.filter(function(r){{return r.period===period && r.precinct_norm===p && r.selection_type===scope.type && r.selection_name===scope.name;}});
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
        var ytd=timingSummary('YTD',precinct,scope), recent=timingSummary('Recent 28D',precinct,scope);
        var matrix=timingMatrix('Recent 28D',precinct,scope);
        var label=(precinct?'Precinct '+precinct:'Citywide')+' — '+(scope.type==='All'?'All incidents':scope.name);
        var recentRange=recent?(' | '+recent.period_start+' to '+recent.period_end):'';
        var story='';
        if(recent) story='<div style="margin-top:6px;color:#0f172a;"><b>Recent timing signal:</b> '+recent.peak_day+' is the highest-volume day, '+recent.peak_time_block+' is the busiest time block, and the dominant shift is '+recent.peak_shift+'.</div>';
        card.innerHTML='<b>When is it happening? — '+label+'</b>'+
          '<div style="margin-top:5px;color:#475569;"><b>Latest 28 days</b>'+recentRange+'</div>'+timingKpisHtml(recent)+story+
          '<div style="margin-top:8px;color:#475569;"><b>Recent day × time concentration</b> <span style="font-weight:400;">(highest cell highlighted)</span></div>'+timingGridHtml(matrix)+
          '<div style="margin-top:8px;color:#475569;"><b>YTD timing context</b>'+(ytd?(' | '+ytd.period_start+' to '+ytd.period_end):'')+'</div>'+timingKpisHtml(ytd);
      }}

      function hotspotStatusClass(t) {{
        if(t==='New Hotspot') return 'hotspot-new';
        if(t==='Emerging Hotspot') return 'hotspot-emerging';
        if(t==='Persistent Hotspot') return 'hotspot-persistent';
        if(t==='Declining Hotspot') return 'hotspot-declining';
        return 'trend-stable';
      }}
      function hotspotRowsFor(precinct,scope) {{
        var p=precinct||'ALL';
        return hotspotChangeData.filter(function(r) {{
          return r.precinct_norm===p && r.selection_type===scope.type && r.selection_name===scope.name;
        }});
      }}
      // Emphasize the location selected from "Where is it changing?" so the
      // analyst does not have to visually search for the target after zooming.
      var hotspotHighlightLayer=null;
      var hotspotPulseTimer=null;
      window.zoomHotspot=function(lat,lon) {{
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
        target.bindTooltip('<b>Selected hotspot location</b><br>Highlighted from Where is it changing?',{{permanent:false,direction:'top',offset:[0,-8]}}).openTooltip();

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
        var h='<table><thead><tr><th>Status / Location</th><th>Prev 28D</th><th>Current 28D</th><th>Abs Δ</th><th>%chg</th><th>Map</th></tr></thead><tbody>';
        use.forEach(function(r) {{
          var loc=(r.nearest_intersection && r.nearest_intersection!=='Unknown')?r.nearest_intersection:r.neighborhood;
          var label='<span class="'+hotspotStatusClass(r.hotspot_status)+'">'+r.hotspot_status+'</span><br><span style="color:#475569;">'+loc+'</span>';
          h+='<tr><td>'+label+'</td><td>'+fmtN(r.previous_28d)+'</td><td>'+fmtN(r.current_28d)+'</td><td>'+fmtN(r.change_28d)+'</td><td>'+fmtPct(r.pct_change_28d)+'</td><td><button onclick="window.zoomHotspot('+r.latitude+','+r.longitude+')" style="padding:3px 6px;border:1px solid #c4b5fd;border-radius:5px;background:#fff;cursor:pointer;">Zoom</button></td></tr>';
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
        var persistent=rows.filter(function(r){{return r.hotspot_status==='Persistent Hotspot';}}).sort(function(a,b){{return Number(b.current_28d||0)-Number(a.current_28d||0);}});
        var declining=rows.filter(function(r){{return r.hotspot_status==='Declining Hotspot';}}).sort(function(a,b){{return Number(a.change_28d||0)-Number(b.change_28d||0);}});
        var label=(precinct?'Precinct '+precinct:'Citywide')+' — '+(scope.type==='All'?'All incidents':scope.name);
        var range=rows.length?(' | '+rows[0].current_28d_start+' to '+rows[0].current_28d_end+' vs '+rows[0].previous_28d_start+' to '+rows[0].previous_28d_end):'';
        var summary='<div style="margin-top:4px;color:#475569;">'+newEmerging.length+' new/emerging, '+persistent.length+' persistent, '+declining.length+' declining hotspot cells'+range+'.</div>';
        var body='';
        if(newEmerging.length) body+='<div style="margin-top:7px;"><b>New / emerging locations needing attention</b></div>'+hotspotTableHtml(newEmerging,6);
        if(persistent.length) body+='<div style="margin-top:8px;"><b>Persistent concentrations</b></div>'+hotspotTableHtml(persistent,5);
        if(declining.length) body+='<div style="margin-top:8px;"><b>Declining hotspots</b></div>'+hotspotTableHtml(declining,5);
        if(!body) body='<div style="margin-top:5px;color:#64748b;">No cells crossed the hotspot thresholds in either 28-day period for this selection.</div>';
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
            '<div style="margin-top:8px;color:#475569;"><b>Recent 28-day movement</b>'+recentWindowText(recent)+'</div>'+recentTableHtml(recent,'offense_category','Crime Type',5)+
            interpretationHtml(rows,recent,precinct,crime);
          return;
        }}
        if(precinct) {{
          var overall=overallData[precinct]; var allRows=rowsFor(precinct,'',trend); var recentRows=rows28For(precinct,'');
          allRows.sort(function(a,b) {{ return Number(b.pct_change_vs_previous||0)-Number(a.pct_change_vs_previous||0); }});
          var topWorse=allRows.filter(function(r){{return (r.trend_class||'').includes('Worsening');}}).slice(0,5);
          var topBetter=allRows.filter(function(r){{return (r.trend_class||'').includes('Improving');}}).sort(function(a,b){{return Number(a.pct_change_vs_previous||0)-Number(b.pct_change_vs_previous||0);}}).slice(0,5);
          var recentUp=recentRows.filter(function(r){{return r.recent_movement==='Increasing';}}).sort(function(a,b){{return Number(b.pct_change_28d||0)-Number(a.pct_change_28d||0);}}).slice(0,5);
          var recentDown=recentRows.filter(function(r){{return r.recent_movement==='Decreasing';}}).sort(function(a,b){{return Number(a.pct_change_28d||0)-Number(b.pct_change_28d||0);}}).slice(0,5);
          var head='<b>Precinct '+precinct+'</b>';
          if(overall) head+=' — <span class="'+trendClassName(overall.trend_class)+'">Overall '+overall.trend_class+'</span> | '+years.current+' YTD '+fmtN(overall.incidents_current)+' | '+fmtPct(overall.pct_change_vs_previous)+' vs '+years.previous+' | Data through '+overall.comparison_date;
          if(trend) {{
            card.innerHTML=head+'<div style="margin-top:6px;"><b>Crime types matching '+trend+' (YTD):</b></div>'+tableHtml(allRows,'offense_category','Crime Type',15)+
              '<div style="margin-top:8px;"><b>Recent 28-day movement for this precinct</b>'+recentWindowText(recentRows)+'</div>'+recentTableHtml(recentRows,'offense_category','Crime Type',10);
          }} else {{
            card.innerHTML=head+
              '<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:6px;"><div><b>Largest YTD worsening drivers</b>'+tableHtml(topWorse,'offense_category','Crime Type',5)+'</div><div><b>Largest YTD improving drivers</b>'+tableHtml(topBetter,'offense_category','Crime Type',5)+'</div></div>'+
              '<div style="margin-top:8px;color:#475569;"><b>Recent 28-day movement</b>'+recentWindowText(recentRows)+'</div>'+
              '<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:4px;"><div><b>Largest recent increases</b>'+recentTableHtml(recentUp,'offense_category','Crime Type',5)+'</div><div><b>Largest recent decreases</b>'+recentTableHtml(recentDown,'offense_category','Crime Type',5)+'</div></div>';
          }}
          return;
        }}
        if(crime) {{
          var rows=rowsFor('',crime,trend); var recent=rows28For('',crime);
          rows.sort(function(a,b) {{ return Number(b.pct_change_vs_previous||0)-Number(a.pct_change_vs_previous||0); }});
          recent.sort(function(a,b) {{ return Number(b.pct_change_28d||0)-Number(a.pct_change_28d||0); }});
          card.innerHTML='<b>'+crime+' across precincts'+(trend?' — '+trend:'')+'</b>'+
            '<div style="margin-top:5px;color:#475569;"><b>Matched YTD</b></div>'+tableHtml(rows,'precinct_norm','Precinct',20)+
            '<div style="margin-top:8px;color:#475569;"><b>Recent 28-day movement</b>'+recentWindowText(recent)+'</div>'+recentTableHtml(recent,'precinct_norm','Precinct',20);
          return;
        }}
        if(trend) {{ var rows=rowsFor('','',trend); rows.sort(function(a,b) {{ return Math.abs(Number(b.pct_change_vs_previous||0))-Math.abs(Number(a.pct_change_vs_previous||0)); }}); card.innerHTML='<b>All crime types — '+trend+'</b><div style="color:#64748b;margin-top:3px;">Select a precinct or crime type to narrow these results.</div>'+tableHtml(rows,'offense_category','Crime Type',15); return; }}
        card.innerHTML='Select a precinct and/or crime type to see matched year-to-date trend details and recent 28-day movement.';
      }}
      function setStatus(parts) {{ var s=document.getElementById('cpSelectorStatus'); if(s) s.textContent='Active filters: '+parts.join(' | '); }}
      function zoomToPrecinct(precinct) {{
        var mp=window[mapObjectName]; if(!mp) return;
        var b=precinctBounds[precinct||'ALL'];
        if(b && b.length===2) mp.fitBounds(b, {{padding:[24,24], maxZoom:13}});
      }}
      window.applyTopSelectors=function() {{
        var core=(document.getElementById('cpCoreSelect')||{{value:''}}).value, action=(document.getElementById('cpActionSelect')||{{value:''}}).value,
            decision=(document.getElementById('cpDecisionSelect')||{{value:''}}).value, precinct=(document.getElementById('cpPrecinctSelect')||{{value:''}}).value,
            category=(document.getElementById('cpCategorySelect')||{{value:''}}).value, view=currentView();
        // The operational map is a real street basemap with a scoped heatmap.
        // H3 remains available as an analytical overlay, not the default visual.
        var scoped = Boolean(precinct || category);
        setExclusiveByPrefix('Core | ', scoped && core==='Core | Incident Density Heatmap' ? '' : core);
        setExclusiveByPrefix('Action | ',action); setExclusiveByPrefix('Decision | ',decision);
        setExclusiveByPrefix('Precinct | ','');
        setExclusiveByPrefix('Category Focus | ','');
        setExclusiveByPrefix('Crime Type | ','');
        setExclusiveByPrefix('Scope | Precinct | ','');
        if(precinct && category) {{
          setExclusiveByPrefix('Scope | Precinct | ','Scope | Precinct | '+precinct+' | '+category);
        }} else if(precinct) {{
          setExclusiveByPrefix('Precinct | ','Precinct | '+precinct);
        }} else if(category.startsWith('Category Focus | ')) {{
          setExclusiveByPrefix('Category Focus | ',category);
        }} else if(category.startsWith('Crime Type | ')) {{
          setExclusiveByPrefix('Crime Type | ',category);
        }}
        setExclusiveByPrefix('Core Type | H3 Count | ', core.startsWith('Core Type | H3 Count | ')?core:'');
        zoomToPrecinct(precinct);
        var crimeLabel=category?category.replace('Category Focus | ','').replace('Crime Type | ',''):'All Crime';
        setStatus([(precinct?'Precinct '+precinct:'Detroit Overview'),crimeLabel,(view==='Recent'?'Recent 28-Day Emphasis':view==='YTD'?'Matched YTD Emphasis':'Operational Summary')]);
        renderKpis();
        renderExecutiveSummary();
        renderPriorityCard();
        renderTimingCard();
        renderHotspotCard();
        renderTrendCard();
        applyViewEmphasis();
      }};
      window.backToDetroit=function() {{ var p=document.getElementById('cpPrecinctSelect'); if(p)p.value=''; window.applyTopSelectors(); }};
      window.clearTopSelectors=function() {{ ['cpActionSelect','cpDecisionSelect','cpPrecinctSelect','cpCategorySelect'].forEach(function(id){{var e=document.getElementById(id);if(e)e.value='';}}); var pe=document.getElementById('cpPeriodSelect'); if(pe)pe.value='Operational'; var ce=document.getElementById('cpCoreSelect'); if(ce) ce.value='Core | Incident Density Heatmap'; setExclusiveByPrefix('Core | ','');setExclusiveByPrefix('Action | ','');setExclusiveByPrefix('Decision | ','');setExclusiveByPrefix('Precinct | ','');setExclusiveByPrefix('Category Focus | ','');setExclusiveByPrefix('Crime Type | ','');setExclusiveByPrefix('Scope | Precinct | ','');setExclusiveByPrefix('Core Type | H3 Count | ',''); window.applyTopSelectors(); }};
      ['cpCoreSelect','cpActionSelect','cpDecisionSelect','cpPrecinctSelect','cpCategorySelect','cpPeriodSelect'].forEach(function(id){{var e=document.getElementById(id);if(e)e.addEventListener('change',window.applyTopSelectors);}});

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

      function applyUrlState() {{
        try {{
          var params=new URLSearchParams(window.location.search);
          var precinct=params.get('precinct')||'';
          var crime=params.get('crime')||'';
          var view=params.get('view')||'';
          var section=params.get('section')||'';
          var psel=document.getElementById('cpPrecinctSelect');
          if(psel && precinct && Array.from(psel.options).some(function(o){{return o.value===precinct;}})) psel.value=precinct;
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
        ("Category Focus | Gun-Related", df[df["is_gun_related"]]),
        ("Category Focus | Property Crime", df[df["is_property_related"]]),
        ("Category Focus | Larceny", df[df["is_larceny_related"]]),
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
            ("Gun-Related", p_df[p_df["is_gun_related"]]),
            ("Property Crime", p_df[p_df["is_property_related"]]),
            ("Larceny", p_df[p_df["is_larceny_related"]]),
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

def build_precinct_crime_28d_comparison(
    df: pd.DataFrame,
    current_year: int,
) -> pd.DataFrame:
    """
    Compare the latest 28 days with the immediately preceding 28 days
    for every precinct and every offense category.

    Also calculates the same 28-day change citywide for each crime type.
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

    # Current 28-day window: cutoff date plus previous 27 days.
    current_start = current_max - pd.Timedelta(days=27)

    # Previous 28-day window immediately before the current one.
    previous_end = current_start - pd.Timedelta(days=1)
    previous_start = previous_end - pd.Timedelta(days=27)

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
        .rename("current_28d")
        .reset_index()
    )

    previous_counts = (
        previous_period.groupby(
            ["precinct_norm", "offense_category"]
        )
        .size()
        .rename("previous_28d")
        .reset_index()
    )

    comparison = previous_counts.merge(
        current_counts,
        on=["precinct_norm", "offense_category"],
        how="outer",
    ).fillna(0)

    comparison["previous_28d"] = comparison["previous_28d"].astype(int)
    comparison["current_28d"] = comparison["current_28d"].astype(int)

    comparison["change_28d"] = (
        comparison["current_28d"] - comparison["previous_28d"]
    )

    comparison["pct_change_28d"] = np.where(
        comparison["previous_28d"] > 0,
        100
        * comparison["change_28d"]
        / comparison["previous_28d"],
        np.nan,
    )

    comparison["recent_movement"] = np.select(
        [
            comparison["change_28d"] > 0,
            comparison["change_28d"] < 0,
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
        .rename("city_current_28d")
        .reset_index()
    )

    city_previous = (
        previous_period.groupby("offense_category")
        .size()
        .rename("city_previous_28d")
        .reset_index()
    )

    city = city_previous.merge(
        city_current,
        on="offense_category",
        how="outer",
    ).fillna(0)

    city["city_previous_28d"] = city["city_previous_28d"].astype(int)
    city["city_current_28d"] = city["city_current_28d"].astype(int)

    city["city_change_28d"] = (
        city["city_current_28d"] - city["city_previous_28d"]
    )

    city["city_pct_change_28d"] = np.where(
        city["city_previous_28d"] > 0,
        100
        * city["city_change_28d"]
        / city["city_previous_28d"],
        np.nan,
    )

    comparison = comparison.merge(
        city,
        on="offense_category",
        how="left",
    )

    # Store the exact periods for dashboard/report display.
    comparison["previous_28d_start"] = previous_start.strftime("%Y-%m-%d")
    comparison["previous_28d_end"] = previous_end.strftime("%Y-%m-%d")
    comparison["current_28d_start"] = current_start.strftime("%Y-%m-%d")
    comparison["current_28d_end"] = current_max.strftime("%Y-%m-%d")

    comparison["pct_change_28d"] = comparison["pct_change_28d"].round(2)
    comparison["city_pct_change_28d"] = comparison[
        "city_pct_change_28d"
    ].round(2)

    return comparison.sort_values(
        ["precinct_norm", "current_28d"],
        ascending=[True, False],
    ).reset_index(drop=True)


def build_temporal_pattern_profiles(
    df: pd.DataFrame,
    current_year: int,
) -> dict[str, pd.DataFrame]:
    """Build YTD and recent-28-day timing profiles for dashboard selections.

    Profiles are available citywide and by precinct for: all incidents, every
    offense category, and the three existing Category Focus groups.
    """
    temp = df.copy()
    temp["incident_date"] = pd.to_datetime(temp["incident_date"])
    temp["incident_hour_of_day"] = pd.to_numeric(temp["incident_hour_of_day"], errors="coerce")
    temp = temp[(temp["incident_year"] == current_year) & temp["incident_hour_of_day"].notna()].copy()
    if temp.empty:
        return {"summary": pd.DataFrame(), "matrix": pd.DataFrame()}

    current_max = temp["incident_date"].max()
    ytd_start = pd.Timestamp(year=int(current_year), month=1, day=1)
    recent_start = current_max - pd.Timedelta(days=27)

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
        ("Recent 28D", temp[(temp["incident_date"] >= recent_start) & (temp["incident_date"] <= current_max)].copy(), recent_start, current_max),
    ]

    focus_specs = [
        ("Category Focus", "Gun-Related", "is_gun_related"),
        ("Category Focus", "Property Crime", "is_property_related"),
        ("Category Focus", "Larceny", "is_larceny_related"),
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
    precinct_crime_28d: pd.DataFrame,
) -> pd.DataFrame:
    """Rank precinct × crime-type signals for operational attention.

    The score is intentionally volume-aware so a tiny baseline cannot dominate
    simply because its percentage change is very large. It combines:
    - current 28-day volume,
    - absolute 28-day increase,
    - 28-day percentage change,
    - divergence from the citywide 28-day trend, and
    - matched-YTD direction versus the previous year.
    """
    if precinct_crime_28d is None or precinct_crime_28d.empty:
        return pd.DataFrame()

    recent = precinct_crime_28d.copy()
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

    recent["change_28d"] = recent["current_28d"] - recent["previous_28d"]
    recent["city_gap_28d"] = recent["pct_change_28d"] - recent["city_pct_change_28d"]

    max_volume = max(float(recent["current_28d"].max()), 1.0)
    max_abs_increase = max(float(recent["change_28d"].clip(lower=0).max()), 1.0)

    volume_component = 25 * np.log1p(recent["current_28d"].clip(lower=0)) / np.log1p(max_volume)
    absolute_component = 30 * recent["change_28d"].clip(lower=0) / max_abs_increase

    # When the prior-period baseline is zero, pct_change_28d is undefined.
    # Use a capped volume-based proxy so new clusters can still surface without
    # assigning an infinite percentage increase.
    pct_signal = recent["pct_change_28d"].copy()
    zero_baseline_proxy = (recent["current_28d"].clip(lower=0) * 10).clip(upper=100)
    pct_signal = pct_signal.where(pct_signal.notna(), zero_baseline_proxy)
    percent_component = 20 * pct_signal.clip(lower=0, upper=100) / 100

    city_component = 15 * recent["city_gap_28d"].fillna(0).clip(lower=0, upper=50) / 50
    ytd_component = 10 * recent["pct_change_vs_previous"].fillna(0).clip(lower=0, upper=50) / 50

    recent["priority_score"] = (
        volume_component + absolute_component + percent_component + city_component + ytd_component
    ).round(1)

    recent_pct = recent["pct_change_28d"]
    abs_change = recent["change_28d"]
    volume = recent["current_28d"]
    city_gap = recent["city_gap_28d"].fillna(0)
    ytd_pct = recent["pct_change_vs_previous"].fillna(0)

    high = (
        (volume >= 20)
        & (abs_change >= 10)
        & ((recent_pct >= 25) | (recent["previous_28d"] == 0))
        & ((city_gap >= 10) | (ytd_pct > 2))
    )
    emerging = (
        (volume >= 10)
        & (abs_change >= 5)
        & ((recent_pct >= 10) | (recent["previous_28d"] == 0))
        & ((city_gap >= 5) | (ytd_pct > 2))
    )
    watch = (volume >= 5) & (abs_change > 0) & ((recent_pct > 2) | (recent["previous_28d"] == 0))
    improving = (volume >= 5) & (abs_change <= -5) & (recent_pct <= -10)

    recent["priority_signal"] = np.select(
        [high, emerging, watch, improving],
        ["High Priority", "Emerging Concern", "Watch", "Recent Improvement"],
        default="Monitor",
    )

    signal_order = {"High Priority": 0, "Emerging Concern": 1, "Watch": 2, "Recent Improvement": 3, "Monitor": 4}
    recent["signal_order"] = recent["priority_signal"].map(signal_order).fillna(9)
    return recent.sort_values(
        ["signal_order", "priority_score", "current_28d"],
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


def build_violent_28d_summaries(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    violent_categories = {"ASSAULT", "AGGRAVATED ASSAULT", "ROBBERY", "HOMICIDE", "WEAPONS OFFENSES"}
    max_date = pd.to_datetime(df["incident_date"]).max()
    start_date = max_date - pd.Timedelta(days=27)

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


def save_violent_28d_charts(by_type: pd.DataFrame, by_day_hour: pd.DataFrame, by_type_out: Path, by_day_hour_out: Path) -> None:
    if not by_type.empty:
        plt.figure(figsize=(10, 5))
        sns.barplot(data=by_type, y="offense_category", x="incident_count", color="#b91c1c")
        plt.title("Violent Crime (Past 28 Days) by Type")
        plt.xlabel("Incidents")
        plt.ylabel("Offense Category")
        plt.tight_layout()
        plt.savefig(by_type_out, dpi=220)
        plt.close()

    if not by_day_hour.empty:
        pivot = by_day_hour.pivot(index="day_name", columns="hour", values="incident_count").fillna(0)
        plt.figure(figsize=(13, 5))
        sns.heatmap(pivot, cmap="Reds", linewidths=0.3, linecolor="#e5e7eb")
        plt.title("Violent Crime (Past 28 Days) by Day and Hour")
        plt.xlabel("Hour of Day")
        plt.ylabel("Day of Week")
        plt.tight_layout()
        plt.savefig(by_day_hour_out, dpi=220)
        plt.close()


def build_hotspot_persistence_change(
    df: pd.DataFrame,
    current_year: int,
    resolution: int = 8,
    hotspot_quantile: float = 0.80,
    min_hotspot_count: int = 3,
) -> pd.DataFrame:
    """Classify H3 locations as persistent, emerging, new, or declining hotspots.

    The comparison uses the latest 28 days in the current-year data versus the
    immediately preceding 28 days. Hotspot thresholds are calculated separately
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

    current_start = current_max - pd.Timedelta(days=27)
    previous_end = current_start - pd.Timedelta(days=1)
    previous_start = previous_end - pd.Timedelta(days=27)

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
        recent["incident_date"] >= current_start, "Current 28D", "Previous 28D"
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
        ("Category Focus", "Gun-Related", recent["is_gun_related"].fillna(False)),
        ("Category Focus", "Property Crime", recent["is_property_related"].fillna(False)),
        ("Category Focus", "Larceny", recent["is_larceny_related"].fillna(False)),
    ]
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
            .reindex(columns=["Previous 28D", "Current 28D"], fill_value=0)
            .reset_index()
            .rename(columns={"Previous 28D": "previous_28d", "Current 28D": "current_28d"})
        )
        if counts.empty:
            return
        prev_threshold = threshold_from(counts["previous_28d"])
        curr_threshold = threshold_from(counts["current_28d"])
        prev_hot = counts["previous_28d"] >= prev_threshold
        curr_hot = counts["current_28d"] >= curr_threshold

        counts["hotspot_status"] = np.select(
            [
                prev_hot & curr_hot,
                (~prev_hot) & curr_hot & counts["previous_28d"].eq(0),
                (~prev_hot) & curr_hot,
                prev_hot & (~curr_hot),
            ],
            ["Persistent Hotspot", "New Hotspot", "Emerging Hotspot", "Declining Hotspot"],
            default="Not Material",
        )
        counts = counts[counts["hotspot_status"] != "Not Material"].copy()
        if counts.empty:
            return

        counts["change_28d"] = counts["current_28d"] - counts["previous_28d"]
        counts["pct_change_28d"] = np.where(
            counts["previous_28d"] > 0,
            100 * counts["change_28d"] / counts["previous_28d"],
            np.nan,
        )
        counts["hotspot_score"] = (
            counts["current_28d"]
            + 1.5 * counts["change_28d"].clip(lower=0)
            + np.where(counts["hotspot_status"].eq("Persistent Hotspot"), counts["current_28d"] * 0.35, 0)
        ).round(2)
        counts["precinct_norm"] = precinct_key
        counts["selection_type"] = selection_type
        counts["selection_name"] = selection_name
        counts["previous_hotspot_threshold"] = prev_threshold
        counts["current_hotspot_threshold"] = curr_threshold
        counts["previous_28d_start"] = previous_start.strftime("%Y-%m-%d")
        counts["previous_28d_end"] = previous_end.strftime("%Y-%m-%d")
        counts["current_28d_start"] = current_start.strftime("%Y-%m-%d")
        counts["current_28d_end"] = current_max.strftime("%Y-%m-%d")
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
    out["pct_change_28d"] = out["pct_change_28d"].round(2)
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
        "Gun-Related": df[df["is_gun_related"]],
        "Property Crime": df[df["is_property_related"]],
        "Larceny": df[df["is_larceny_related"]],
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
    m = folium.Map(location=center, zoom_start=11, tiles="CartoDB positron")

    heat_data = df[["latitude", "longitude"]].values.tolist()
    HeatMap(heat_data, radius=10, blur=12, max_zoom=13).add_to(m)

    m.save(str(out_path))


def save_spike_marker_map(spike_points: pd.DataFrame, out_path: Path) -> None:
    if spike_points.empty:
        return

    center = [spike_points["latitude"].median(), spike_points["longitude"].median()]
    m = folium.Map(location=center, zoom_start=11, tiles="CartoDB positron")

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
    m = folium.Map(location=center, zoom_start=11, tiles="CartoDB positron")
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
    m = folium.Map(location=center, zoom_start=11, tiles="CartoDB positron")
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
    m = folium.Map(location=center, zoom_start=11, tiles="CartoDB positron")

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
    precinct_crime_28d: pd.DataFrame | None = None,
    priority_concerns: pd.DataFrame | None = None,
    temporal_summary: pd.DataFrame | None = None,
    temporal_matrix: pd.DataFrame | None = None,
    hotspot_change: pd.DataFrame | None = None,
    current_year: int | None = None,
    previous_year: int | None = None,
    baseline_year: int | None = None,
) -> None:
    center = [df["latitude"].median(), df["longitude"].median()]

    # Real-world Detroit basemap first; analytical H3 layers sit transparently above it.
    # Voyager is the default because it keeps street names, major roads, neighborhoods,
    # parks, and landmarks readable beneath the hotspot polygons.
    m = folium.Map(
        location=center,
        zoom_start=11,
        tiles=None,
        control_scale=True,
        prefer_canvas=True,
    )
    folium.TileLayer(
        tiles="CartoDB Voyager",
        name="Basemap | Detroit Streets (Recommended)",
        control=True,
        show=True,
    ).add_to(m)
    folium.TileLayer(
        tiles="OpenStreetMap",
        name="Basemap | OpenStreetMap",
        control=True,
        show=False,
    ).add_to(m)
    folium.TileLayer(
        tiles="CartoDB positron",
        name="Basemap | Light Analytical",
        control=True,
        show=False,
    ).add_to(m)
    Fullscreen(position="topright", title="Expand", title_cancel="Exit", force_separate_button=True).add_to(m)

    # Optional underlying event geography: latest 28 days only, so the layer remains
    # useful and responsive instead of attempting to draw ~200k individual markers.
    date_series = pd.to_datetime(df["incident_occurred_at"], errors="coerce", utc=True)
    latest_date = date_series.max()
    if pd.notna(latest_date):
        recent_start = latest_date - pd.Timedelta(days=27)
        recent_points = df.loc[date_series.between(recent_start, latest_date)].copy()
        recent_points = recent_points.dropna(subset=["latitude", "longitude"])
        if not recent_points.empty:
            recent_layer = folium.FeatureGroup(
                name=f"Locations | Actual Incidents — Latest 28D ({len(recent_points):,})",
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

    # Layer 5+: decision-purpose layers (color-coded by operational response).
    purpose_styles = {
        "Preventive Patrol": {
            "gradient": {0.25: "#dbeafe", 0.5: "#60a5fa", 0.75: "#2563eb", 1.0: "#1e3a8a"},
            "label": "Decision | Preventive Patrol Priority",
        },
        "Investigations": {
            "gradient": {0.25: "#f3e8ff", 0.5: "#c084fc", 0.75: "#9333ea", 1.0: "#581c87"},
            "label": "Decision | Investigations Priority",
        },
        "Community Response": {
            "gradient": {0.25: "#fee2e2", 0.5: "#f87171", 0.75: "#dc2626", 1.0: "#7f1d1d"},
            "label": "Decision | Community Response Priority",
        },
    }

    for purpose, style in purpose_styles.items():
        subset = temp_time[temp_time["decision_purpose"] == purpose]
        if subset.empty:
            continue
        layer = folium.FeatureGroup(
            name=f"{style['label']} ({len(subset):,})",
            show=False,
        )
        heat_data = subset[["latitude", "longitude"]].values.tolist()
        HeatMap(
            heat_data,
            radius=11,
            blur=14,
            max_zoom=13,
            gradient=style["gradient"],
        ).add_to(layer)
        layer.add_to(m)

    # Layer 8+: top offense-category heatmaps.
    offense_counts = df["offense_category"].value_counts()
    if top_n_categories is not None:
        offense_counts = offense_counts.head(top_n_categories)
    for idx, category in enumerate(offense_counts.index.tolist()):
        subset = df[df["offense_category"] == category]
        if subset.empty:
            continue
        layer = folium.FeatureGroup(
            name=f"Crime Type | {category} ({len(subset):,})",
            show=False,
        )
        heat_data = subset[["latitude", "longitude"]].values.tolist()
        HeatMap(heat_data, radius=10, blur=12, max_zoom=13).add_to(layer)
        layer.add_to(m)

    # Layer 12+: time-of-day heatmaps for deployment by shift.
    for shift_name in [
        "Day Shift (06:00-13:59)",
        "Evening Shift (14:00-21:59)",
        "Night Shift (22:00-05:59)",
    ]:
        subset = temp_time[temp_time["shift_window"] == shift_name]
        if subset.empty:
            continue
        layer = folium.FeatureGroup(
            name=f"Shift View | {shift_name} ({len(subset):,})",
            show=False,
        )
        heat_data = subset[["latitude", "longitude"]].values.tolist()
        HeatMap(heat_data, radius=10, blur=12, max_zoom=13).add_to(layer)
        layer.add_to(m)

    add_precinct_filter_layers(m, df)
    add_focus_category_layers(m, df)
    add_precinct_scope_heatmap_layers(m, df)
    add_crime_type_h3_count_layers(m, df, resolution=resolution, top_n_categories=None)
    # Bounds power precinct auto-zoom while preserving the real street basemap.
    precinct_bounds: dict[str, list[list[float]]] = {}
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

    add_top_selector_panel(
        m,
        sorted(df["precinct_norm"].dropna().astype(str).unique().tolist()),
        sorted(df["offense_category"].dropna().astype(str).unique().tolist()),
        precinct_improvement=precinct_improvement,
        precinct_crime_trends=precinct_crime_trends,
        precinct_crime_28d=precinct_crime_28d,
        priority_concerns=priority_concerns,
        temporal_summary=temporal_summary,
        temporal_matrix=temporal_matrix,
        hotspot_change=hotspot_change,
        precinct_bounds=precinct_bounds,
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
    m = folium.Map(location=center, zoom_start=11, tiles="CartoDB positron")

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
        violent_type_img = pdir / f"precinct_{precinct}_violent_crime_28d_by_type_{period_tag}.png"
        violent_day_hour_img = pdir / f"precinct_{precinct}_violent_crime_28d_by_day_hour_{period_tag}.png"

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

        p_violent_type, p_violent_day_hour = build_violent_28d_summaries(p_df)
        save_violent_28d_charts(
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
    precinct_crime_28d: pd.DataFrame,
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

    improvement = precinct_improvement.copy() if precinct_improvement is not None else pd.DataFrame()
    if not improvement.empty:
        improvement["precinct_key"] = improvement["precinct_norm"].apply(norm_precinct)

    recent = precinct_crime_28d.copy() if precinct_crime_28d is not None else pd.DataFrame()
    if not recent.empty:
        recent["precinct_key"] = recent["precinct_norm"].apply(norm_precinct)

    trends = precinct_crime_trends.copy() if precinct_crime_trends is not None else pd.DataFrame()
    if not trends.empty:
        trends["precinct_key"] = trends["precinct_norm"].apply(norm_precinct)

    priorities = priority_concerns.copy() if priority_concerns is not None else pd.DataFrame()
    if not priorities.empty:
        priorities["precinct_key"] = priorities["precinct_norm"].apply(norm_precinct)

    timing = temporal_summary.copy() if temporal_summary is not None else pd.DataFrame()
    if not timing.empty:
        timing["precinct_key"] = timing["precinct_norm"].apply(norm_precinct)

    hotspots = hotspot_change.copy() if hotspot_change is not None else pd.DataFrame()
    if not hotspots.empty:
        hotspots["precinct_key"] = hotspots["precinct_norm"].apply(norm_precinct)

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

        # Aggregate recent 28-day all-crime movement by summing crime rows.
        prev28 = curr28 = 0
        if not recent.empty:
            rr = recent[recent["precinct_key"].eq(precinct)]
            prev28 = int(pd.to_numeric(rr["previous_28d"], errors="coerce").fillna(0).sum())
            curr28 = int(pd.to_numeric(rr["current_28d"], errors="coerce").fillna(0).sum())
        recent_pct = 100 * (curr28 - prev28) / prev28 if prev28 else None

        # Priority concerns and improvements.
        concern_records, improvement_records = [], []
        concern_count = 0
        if not priorities.empty:
            pp = priorities[priorities["precinct_key"].eq(precinct)].copy()
            if not pp.empty:
                pp = pp.sort_values(
                    ["priority_score", "current_28d"],
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
                        "previous_28d": clean_number(r.get("previous_28d")),
                        "current_28d": clean_number(r.get("current_28d")),
                        "pct_change_28d": clean_number(r.get("pct_change_28d")),
                        "city_pct_change_28d": clean_number(r.get("city_pct_change_28d")),
                        "ytd_pct_change": clean_number(r.get("pct_change_vs_previous")),
                        "signal": str(r.get("priority_signal", "Monitor")),
                        "score": clean_number(r.get("priority_score")),
                    })
                for _, r in improve_df.head(5).iterrows():
                    improvement_records.append({
                        "crime": str(r.get("offense_category", "Unknown")),
                        "pct_change_28d": clean_number(r.get("pct_change_28d")),
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
                        "recent_pct": clean_number(r.get("pct_change_28d")),
                        "current_28d": clean_number(r.get("current_28d")),
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
                    "current_28d": rc.get("current_28d"),
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
                & timing["period"].astype(str).eq("Recent 28D")
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
                "previous_28d": int(prev28),
                "current_28d": int(curr28),
                "pct_change_28d": recent_pct,
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

    payload = {
        "city": city_data,
        "precincts": precinct_data,
        "map_filename": map_filename,
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
.links{{display:flex;gap:8px 18px;flex-wrap:wrap;margin:10px 0 2px}} .links a{{color:var(--blue);font-weight:760;text-decoration:none}} .links a:hover{{text-decoration:underline}}
.signal{{font-weight:800}} .high{{color:var(--red)}} .emerging{{color:var(--orange)}} .watch{{color:#a16207}} .improving{{color:var(--green)}}
.badge{{display:inline-block;border-radius:999px;padding:4px 9px;font-size:.76rem;font-weight:800;background:#e2e8f0;color:#334155}}
.metric-list{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:9px}} .metric{{padding:10px;border:1px solid #e5e7eb;border-radius:8px;background:#fafafa}}
.metric b{{display:block;font-size:1.05rem;margin-top:3px}} .driver-grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px}} .empty{{color:var(--muted);padding:8px 0}}
#precinctOverview{{display:none}} .anchor-target{{scroll-margin-top:12px}}
@media(max-width:980px){{.cards{{grid-template-columns:repeat(2,minmax(0,1fr))}}.grid2,.driver-grid{{grid-template-columns:1fr}}}}
@media(max-width:560px){{.wrap{{padding:18px 12px 34px}}.cards{{grid-template-columns:1fr}}.metric-list{{grid-template-columns:1fr}}.selector select{{width:100%;min-width:0}}}}
</style>
</head>
<body>
<div class="wrap">
<div class="eyebrow">Operational Control Center</div>
<h1>Detroit Crime Operations</h1>
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
<h2>Detroit Overview</h2>
<p style="line-height:1.55;margin:0;">Citywide totals provide context only. Choose a precinct above to open its full operational evaluation: focus locations, deployment summary, matched-YTD improvement, dominant crime types, shift demand, decision-purpose workload, recent concerns, and precinct-only maps/charts.</p>
<div class="links"><a href="../Images/{map_filename}?section=map" target="_blank">Open Detroit Interactive Map</a></div>
</div>

<div id="precinctOverview">
  <div style="display:flex;justify-content:space-between;gap:12px;align-items:end;flex-wrap:wrap;margin-bottom:8px;">
    <div><div class="eyebrow">Precinct Evaluation</div><h2 id="precinctTitle" style="font-size:1.7rem;margin-top:4px;"></h2><div id="precinctDate" class="sub"></div></div>
    <span class="badge" id="trendBadge"></span>
  </div>

  <div class="brief">
    <h3>Open precinct analysis</h3>
    <div class="links" id="classicLinks"></div>
    <div class="links" id="modernLinks" style="padding-top:4px;border-top:1px solid #eef2f7;"></div>
  </div>

  <div class="cards" id="precinctCards"></div>

  <div class="brief"><h3>Overall Evaluation</h3><p id="precinctBrief" style="line-height:1.58;margin:0;"></p></div>

  <div class="section anchor-target" id="focusSection">
    <h2>Top 10 Operational Focus Locations</h2>
    <div class="note">Highest-ranked focus cells inside the selected precinct only.</div>
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
    <div class="note">Full-period workload with current matched-YTD and recent 28-day direction beside it.</div>
    <div id="dominantCrimes"></div>
  </div>

  <div class="section anchor-target" id="prioritySection">
    <h2>Priority / Emerging Concerns</h2>
    <div id="concernsTable"></div>
  </div>

  <div class="grid2">
    <div class="section anchor-target" id="shiftSection"><h2>Shift-Level Demand and Dominant Crime</h2><div id="shiftSummary"></div></div>
    <div class="section anchor-target" id="decisionSection"><h2>Decision-Purpose Workload and Dominant Crime</h2><div id="decisionSummary"></div></div>
  </div>

  <div class="grid2">
    <div class="section anchor-target" id="timingSection"><h2>Recent Timing Snapshot</h2><div id="timingSummary"></div></div>
    <div class="section anchor-target" id="hotspotSection"><h2>Hotspot Change Snapshot</h2><div id="hotspotSummary"></div></div>
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
function analysisUrl(p,section){{return '../Images/'+DATA.map_filename+'?precinct='+encodeURIComponent(p)+'&section='+encodeURIComponent(section);}}
function assetLink(path,label){{return path?`<a href="${{path}}" target="_blank">${{label}}</a>`:'';}}

function renderCity(){{
 document.getElementById('cityCards').innerHTML=`
 <div class="card"><div class="label">Total incidents ${{city.min_year}}–${{city.max_year}}</div><div class="value">${{fmtN(city.total_incidents)}}</div></div>
 <div class="card"><div class="label">${{city.current_year}} YTD incidents</div><div class="value">${{fmtN(city.current_ytd)}}</div></div>
 <div class="card"><div class="label">${{city.current_year}} vs ${{city.previous_year}} YTD</div><div class="value ${{pctClass(city.pct_change_ytd)}}">${{fmtPct(city.pct_change_ytd)}}</div></div>
 <div class="card"><div class="label">Operational precincts</div><div class="value">${{fmtN(city.precinct_count)}}</div></div>
 <div class="card"><div class="label">Highest-volume precinct</div><div class="value">${{city.highest_volume_precinct?'P'+city.highest_volume_precinct:'—'}}</div><div class="note">${{city.highest_volume?fmtN(city.highest_volume)+' current YTD':''}}</div></div>`;
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
 const ov=d.overall||{{}}, rec=d.recent||{{}}, a=d.assets||{{}};
 document.getElementById('precinctTitle').textContent='Precinct '+p+' Operations Evaluation';
 document.getElementById('precinctDate').textContent='Matched YTD through '+(ov.comparison_date||city.data_through)+' · full data coverage '+city.min_year+'–'+city.max_year;
 document.getElementById('trendBadge').textContent=ov.improvement_status||'Trend unavailable';

 document.getElementById('precinctCards').innerHTML=`
 <div class="card"><div class="label">${{city.current_year}} YTD incidents</div><div class="value">${{fmtN(ov.incidents_current)}}</div></div>
 <div class="card"><div class="label">vs ${{city.previous_year}} YTD</div><div class="value ${{pctClass(ov.pct_change_vs_previous)}}">${{fmtPct(ov.pct_change_vs_previous)}}</div></div>
 <div class="card"><div class="label">Current 28 days</div><div class="value">${{fmtN(rec.current_28d)}}</div><div class="note ${{pctClass(rec.pct_change_28d)}}">${{fmtPct(rec.pct_change_28d)}} vs prior 28D</div></div>
 <div class="card"><div class="label">Priority concerns</div><div class="value">${{fmtN(d.concern_count)}}</div></div>
 <div class="card"><div class="label">Focus locations</div><div class="value">${{fmtN((d.focus_locations||[]).length)}}</div></div>`;

 let s=[];
 if(ov.pct_change_vs_previous!==null&&ov.pct_change_vs_previous!==undefined)s.push(`Overall matched-YTD incidents are ${{Math.abs(Number(ov.pct_change_vs_previous)).toFixed(1)}}% ${{Number(ov.pct_change_vs_previous)<0?'below':'above'}} ${{city.previous_year}}.`);
 if(rec.pct_change_28d!==null&&rec.pct_change_28d!==undefined)s.push(`Recent activity moved from ${{fmtN(rec.previous_28d)}} to ${{fmtN(rec.current_28d)}} incidents (${{fmtPct(rec.pct_change_28d)}}).`);
 if(d.concerns&&d.concerns.length)s.push(`${{d.concerns[0].crime}} is the leading current attention signal (${{d.concerns[0].signal}}).`);
 if(d.timing)s.push(`Recent activity peaks on ${{d.timing.peak_day}} around ${{hourLabel(d.timing.peak_hour)}}, with ${{d.timing.peak_time_block}} as the busiest broad time block.`);
 if(d.focus_locations&&d.focus_locations.length)s.push(`The highest-ranked focus location is ${{d.focus_locations[0].neighborhood}} near ${{d.focus_locations[0].intersection}}.`);
 document.getElementById('precinctBrief').textContent=s.join(' ');

 document.getElementById('classicLinks').innerHTML=[
   assetLink(analysisUrl(p,'map'),'Open Interactive Precinct Map'),
   assetLink(a.focus_map,'Open Focus Locations Map'),
   assetLink(a.spike_map,'Open Spike Severity Choropleth'),
   assetLink(a.shift_chart,'Open Shift Summary Chart'),
   assetLink(a.decision_chart,'Open Decision Purpose Chart'),
   assetLink(a.monthly_heatmap,'Open Precinct Monthly Trend Heatmap'),
   assetLink(a.violent_day_hour,'Open Violent Crime 28-Day Day/Hour')
 ].filter(Boolean).join('');
 document.getElementById('modernLinks').innerHTML=[
   assetLink(analysisUrl(p,'trends'),'Open Crime Trend Analysis'),
   assetLink(analysisUrl(p,'priority'),'Open Priority / Emerging Concerns'),
   assetLink(analysisUrl(p,'timing'),'Open Temporal Analysis'),
   assetLink(analysisUrl(p,'hotspot'),'Open Hotspot Changes')
 ].filter(Boolean).join('');

 const fl=(d.focus_locations||[]).map(r=>[
   fmtN(r.rank),r.neighborhood,r.intersection,fmtN(r.incident_count),r.dominant_offense,fmtShare(r.dominant_offense_share),Number(r.score||0).toFixed(3)
 ]);
 document.getElementById('focusLocations').innerHTML=simpleTable(['Rank','Neighborhood','Nearest intersection','Incidents','Dominant crime','Share','Score'],fl);

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
 ['${baseline_year if baseline_year is not None else "Baseline"}','${previous_year}','${current_year}','vs ${previous_year}','vs ${baseline_year if baseline_year is not None else "baseline"}','Status','Score'],
 [[fmtN(ov.incidents_baseline),fmtN(ov.incidents_previous),fmtN(ov.incidents_current),fmtPct(ov.pct_change_vs_previous),fmtPct(ov.pct_change_vs_baseline),ov.improvement_status||'—',ov.improvement_score===null?'—':Number(ov.improvement_score).toFixed(2)]]
 );
 const wd=(d.worsening_drivers||[]).map(r=>[r.crime,fmtN(r.previous),fmtN(r.current),fmtPct(r.pct),r.trend]);
 const id=(d.improving_drivers||[]).map(r=>[r.crime,fmtN(r.previous),fmtN(r.current),fmtPct(r.pct),r.trend]);
 document.getElementById('worseningDrivers').innerHTML=simpleTable(['Crime',city.previous_year,city.current_year,'% change','Trend'],wd);
 document.getElementById('improvingDrivers').innerHTML=simpleTable(['Crime',city.previous_year,city.current_year,'% change','Trend'],id);

 const dc=(d.dominant_crimes||[]).map(r=>[r.crime,fmtN(r.incident_count),fmtShare(r.share),fmtN(r.current_ytd),fmtPct(r.ytd_pct_change),r.trend||'—',fmtN(r.current_28d),fmtPct(r.recent_pct_change)]);
 document.getElementById('dominantCrimes').innerHTML=simpleTable(['Crime','All-period incidents','Share',city.current_year+' YTD','YTD %chg','YTD trend','Current 28D','28D %chg'],dc);

 const cr=(d.concerns||[]).map(r=>[r.crime,fmtN(r.previous_28d),fmtN(r.current_28d),fmtPct(r.pct_change_28d),fmtPct(r.city_pct_change_28d),fmtPct(r.ytd_pct_change),`<span class="signal ${{signalClass(r.signal)}}">${{r.signal}}</span>`]);
 document.getElementById('concernsTable').innerHTML=simpleTable(['Crime','Prev 28D','Current 28D','28D','City','YTD','Signal'],cr);

 const sh=(d.shift_summary||[]).map(r=>[r.shift,fmtN(r.incidents),r.dominant_offense,fmtN(r.dominant_offense_count),fmtShare(r.dominant_offense_share)]);
 document.getElementById('shiftSummary').innerHTML=simpleTable(['Shift','Incidents','Dominant crime','Dominant count','Share'],sh);

 const ds=(d.decision_summary||[]).map(r=>[r.purpose,fmtN(r.incidents),r.dominant_offense,fmtN(r.dominant_offense_count),fmtShare(r.dominant_offense_share)]);
 document.getElementById('decisionSummary').innerHTML=simpleTable(['Decision purpose','Incidents','Dominant crime','Dominant count','Share'],ds);

 if(d.timing)document.getElementById('timingSummary').innerHTML=`<div class="metric-list"><div class="metric"><span class="note">Peak day</span><b>${{d.timing.peak_day}}</b></div><div class="metric"><span class="note">Peak hour</span><b>${{hourLabel(d.timing.peak_hour)}}</b></div><div class="metric"><span class="note">Busiest block</span><b>${{d.timing.peak_time_block}}</b></div><div class="metric"><span class="note">Dominant shift</span><b>${{d.timing.peak_shift}}</b></div></div>`;
 else document.getElementById('timingSummary').innerHTML='<div class="empty">Recent timing profile unavailable.</div>';

 const h=d.hotspots||{{new:0,emerging:0,persistent:0,declining:0}};
 document.getElementById('hotspotSummary').innerHTML=`<div class="metric-list"><div class="metric"><span class="note">New</span><b>${{h.new}}</b></div><div class="metric"><span class="note">Emerging</span><b>${{h.emerging}}</b></div><div class="metric"><span class="note">Persistent</span><b>${{h.persistent}}</b></div><div class="metric"><span class="note">Declining</span><b>${{h.declining}}</b></div></div>`;
}}

function resetToCity(){{document.getElementById('cityPrompt').style.display='block';document.getElementById('precinctOverview').style.display='none';}}
select.addEventListener('change',()=>{{let p=select.value;p?renderPrecinct(p):resetToCity();}});
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
        <a href=\"../Images/detroit_precinct_focus_locations_{period_tag}.html\" target=\"_blank\">Open Focus Locations Map</a>
        <a href=\"../Images/detroit_crime_h3_spike_severity_choropleth_{period_tag}.html\" target=\"_blank\">Open Spike Severity Choropleth</a>
                <a href="../Images/detroit_shift_incidents_{period_tag}.png" target="_blank">Open Shift Summary Chart</a>
                <a href="../Images/detroit_decision_purpose_incidents_{period_tag}.png" target="_blank">Open Decision Purpose Chart</a>
            <a href="../Images/detroit_precinct_monthly_trend_heatmap_{period_tag}.png" target="_blank">Open Precinct Monthly Trend Heatmap</a>
    <a href="../Images/detroit_violent_crime_28d_by_day_hour_{period_tag}.png" target="_blank">Open Violent Crime 28-Day Day/Hour</a>
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
    precinct_crime_28d = build_precinct_crime_28d_comparison(
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
        precinct_crime_28d,
    )
    temporal_patterns = build_temporal_pattern_profiles(df, current_year=current_year)
    temporal_summary = temporal_patterns["summary"]
    temporal_matrix = temporal_patterns["matrix"]
    hotspot_change = build_hotspot_persistence_change(df, current_year=current_year)

    # Primary entry point + interactive drill-down.
    operations_overview_html = DOCS_DIR / f"detroit_crime_operations_overview_{period_tag}.html"
    combined_dashboard_html = IMAGES_DIR / f"detroit_crime_interactive_dashboard_{period_tag}.html"

    # Preserve the useful legacy drill-down ideas, but regenerate each asset
    # separately for every operational precinct.
    precinct_assets = generate_precinct_drilldown_assets(
        df=df,
        focus_locations=focus_locations,
        current_year=current_year,
        period_tag=period_tag,
    )

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
    precinct_crime_28d_csv = DOCS_DIR / f"precinct_crime_28d_comparison_{current_year}.csv"
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
        precinct_crime_28d=precinct_crime_28d,
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
        precinct_crime_28d=precinct_crime_28d,
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
    precinct_crime_28d.to_csv(precinct_crime_28d_csv, index=False)
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
        precinct_crime_28d_csv,
        precinct_crime_trends_csv,
        priority_concerns_csv,
        temporal_summary_csv,
        temporal_matrix_csv,
        hotspot_change_csv,
    ]:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
