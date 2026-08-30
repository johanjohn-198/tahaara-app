"""
TAHAARA live app — Field Operations UI

Run with: streamlit run tahaara_app.py

Requires tahaara_core.py in the same folder, plus the drift model files
(tahaara_drift_model.joblib, tahaara_drift_features.joblib). The detection
model files are optional at first — if missing, the app runs in "drift only"
mode with a manual override, so the drift half is fully demoable before the
detector is trained.
"""

import tempfile
from pathlib import Path

import folium
import streamlit as st
import streamlit.components.v1 as components

import tahaara_core as core

st.set_page_config(page_title="TAHAARA", page_icon="⚓", layout="wide")

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
    --ink: #E7E2D3;
    --masthead: #0A121A;
    --paper: #121F2B;
    --paper-panel: #182838;
    --hairline: #2E4051;
    --rust: #D98E56;
    --teal: #5AAB98;
    --muted: #8B9AA8;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: var(--paper); color: var(--ink); }

[data-testid="stHeader"] { background: var(--masthead) !important; }
[data-testid="stToolbar"] svg, [data-testid="stHeader"] svg { fill: var(--ink) !important; }
[data-testid="stAppViewContainer"] { background: var(--paper) !important; }
[data-testid="stMainBlockContainer"] { padding-top: 3.75rem !important; }

h1, h2, h3 { font-family: 'Fraunces', serif !important; font-weight: 600 !important; color: var(--ink) !important; }
p, label, span, div { color: var(--ink); }

.tahaara-masthead {
    display: flex; align-items: center; justify-content: space-between;
    background: var(--masthead); color: #EDE7D8;
    padding: 18px 28px; margin: 0 -1rem 1.5rem -1rem;
    border-bottom: 3px solid var(--rust);
}
.tahaara-masthead-left { display: flex; align-items: center; gap: 14px; }
.tahaara-wordmark { font-family: 'Fraunces', serif; font-weight: 700; font-size: 24px; letter-spacing: 0.03em; }
.tahaara-tagline { font-family: 'IBM Plex Mono', monospace; font-size: 10.5px; letter-spacing: 0.14em; text-transform: uppercase; color: #8FA3B3; margin-top: 2px; }
.tahaara-status { font-family: 'IBM Plex Mono', monospace; font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase; color: #EDE7D8; border: 1px solid #34495E; padding: 5px 12px; border-radius: 2px; }
.tahaara-status .dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; background: #7FB07A; margin-right: 6px; }

.section-label {
    font-family: 'IBM Plex Mono', monospace; font-size: 12px; letter-spacing: 0.1em;
    text-transform: uppercase; color: var(--rust); border-bottom: 1px solid var(--hairline);
    padding-bottom: 6px; margin-bottom: 14px;
}

.mode-row {
    font-family: 'IBM Plex Mono', monospace; font-size: 11px; letter-spacing: 0.06em;
    text-transform: uppercase; color: var(--muted); margin-bottom: 10px;
}

[data-testid="stFileUploaderDropzone"] {
    background: var(--paper-panel) !important; border: 1px dashed var(--hairline) !important; border-radius: 2px !important;
}
[data-testid="stFileUploaderDropzone"] * { color: var(--muted) !important; }
[data-testid="stFileUploaderDropzone"] button {
    background: var(--paper) !important; color: var(--ink) !important; border: 1px solid var(--hairline) !important;
}
.stNumberInput input, .stSelectbox > div, .stTextInput input, .stSelectbox div[data-baseweb="select"] {
    background: var(--paper-panel) !important; border: none !important; border-bottom: 1px solid var(--hairline) !important;
    border-radius: 0 !important; color: var(--ink) !important;
}
div[data-baseweb="select"] > div { background: var(--paper-panel) !important; color: var(--ink) !important; border-color: var(--hairline) !important; }
div[data-baseweb="select"] * { color: var(--ink) !important; }
div[data-baseweb="popover"] { background: var(--paper-panel) !important; }
ul[data-testid="stSelectboxVirtualDropdown"], ul[role="listbox"] { background: var(--paper-panel) !important; }
ul[role="listbox"] li { color: var(--ink) !important; }
ul[role="listbox"] li:hover { background: var(--hairline) !important; }
.stSelectbox svg, .stNumberInput svg { fill: var(--muted) !important; }
.stNumberInput button { background: var(--paper-panel) !important; border-color: var(--hairline) !important; }
input[type="checkbox"], input[type="radio"] { accent-color: var(--rust); }
[data-testid="stCheckbox"] [data-baseweb="checkbox"] div:first-child { border-color: var(--muted) !important; background: transparent !important; }
[data-testid="stCheckbox"] [aria-checked="true"] div:first-child { background: var(--rust) !important; border-color: var(--rust) !important; }
[data-testid="stWidgetLabel"] p { color: var(--muted) !important; font-size: 13px !important; }
[data-testid="stRadio"] label p { color: var(--ink) !important; font-size: 13px !important; }

.stButton > button {
    background: var(--rust) !important; color: #12202C !important; font-family: 'IBM Plex Mono', monospace !important;
    letter-spacing: 0.08em; text-transform: uppercase; font-size: 13px !important; font-weight: 600 !important; border: none !important;
    border-radius: 2px !important; padding: 10px 20px !important;
}
.stButton > button:hover { background: #E8A26C !important; }
.stButton > button p { color: #12202C !important; }

.tahaara-stamp {
    display: inline-block; font-family: 'IBM Plex Mono', monospace; font-size: 11px; letter-spacing: 0.1em;
    text-transform: uppercase; border: 1.5px solid var(--rust); color: var(--rust); padding: 6px 12px;
    transform: rotate(-1.5deg); border-radius: 2px;
}

.ledger { width: 100%; border-collapse: collapse; font-family: 'IBM Plex Mono', monospace; margin: 6px 0 18px 0; }
.ledger td { padding: 10px 4px; border-bottom: 1px solid var(--hairline); vertical-align: baseline; }
.ledger td.label { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; width: 40%; }
.ledger td.value { color: var(--ink); font-size: 19px; font-weight: 600; text-align: right; }
.ledger td.value .unit { font-size: 12px; color: var(--muted); font-weight: 400; margin-left: 4px; }

[data-testid="stVerticalBlockBorderWrapper"] { border-color: var(--hairline) !important; border-radius: 2px !important; background: var(--paper-panel) !important; }
hr { border-color: var(--hairline) !important; }
[data-testid="stCaptionContainer"] { font-family: 'IBM Plex Mono', monospace !important; letter-spacing: 0.03em; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="tahaara-masthead">
        <div class="tahaara-masthead-left">
            <svg width="34" height="34" viewBox="0 0 34 34" fill="none">
                <circle cx="17" cy="17" r="15.5" stroke="#EDE7D8" stroke-width="1.2"/>
                <path d="M17 3 L19.6 15 L31 17 L19.6 19 L17 31 L14.4 19 L3 17 L14.4 15 Z" fill="#A8562E"/>
                <circle cx="17" cy="17" r="2.2" fill="#EDE7D8"/>
            </svg>
            <div>
                <div class="tahaara-wordmark">TAHAARA</div>
                <div class="tahaara-tagline">Spill Detection &amp; Drift Response — Persian Gulf</div>
            </div>
        </div>
        <div class="tahaara-status"><span class="dot"></span>Station Active</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="mode-row">Analysis Mode</div>', unsafe_allow_html=True)
pipeline_mode = st.radio(
    "Analysis mode", ["Full Pipeline", "Detection Only"],
    horizontal=True, label_visibility="collapsed",
)
run_full_pipeline = pipeline_mode == "Full Pipeline"

left_col, right_col = st.columns([1, 1.4])

with left_col:
    st.markdown('<div class="section-label">01 — Detection Input</div>', unsafe_allow_html=True)

    if not core.DETECTION_AVAILABLE:
        st.markdown('<span class="tahaara-stamp">Detector Offline — Manual Mode</span>', unsafe_allow_html=True)
        st.write("")

    uploaded_image = st.file_uploader("Drone image", type=["jpg", "jpeg", "png"])

    manual_override = st.checkbox(
        "Assume oil detected (manual override)",
        value=not core.DETECTION_AVAILABLE,
        help="Use while the detection model is still being trained.",
    )

    latitude = longitude = oil_type = initial_area_km2 = fleet_size = None

    if run_full_pipeline:
        st.markdown('<div class="section-label" style="margin-top:28px;">02 — Spill Location</div>', unsafe_allow_html=True)
        latitude = st.number_input("Latitude", value=26.0000, format="%.4f")
        longitude = st.number_input("Longitude", value=53.0000, format="%.4f")
        oil_type = st.selectbox("Oil type", list(core.OIL_TYPE_PROPERTIES.keys()))

        if core.is_on_land(latitude, longitude):
            st.error("This coordinate is on land. Choose an offshore point before running the analysis.")

        st.markdown('<div class="section-label" style="margin-top:28px;">03 — Spill Area</div>', unsafe_allow_html=True)
        area_choice = st.radio(
            "Estimated size",
            list(core.AREA_PRESETS_KM2.keys()) + ["Exact value"],
            horizontal=True,
        )
        if area_choice == "Exact value":
            initial_area_km2 = st.number_input("Area (km²)", min_value=0.01, value=1.0, step=0.1)
        else:
            initial_area_km2 = core.AREA_PRESETS_KM2[area_choice]
            st.caption(f"Using {initial_area_km2} km² for '{area_choice}'.")

        with st.expander("Fleet constraint (optional)"):
            use_fleet_limit = st.checkbox("Limit to a specific number of available pods")
            if use_fleet_limit:
                fleet_size = st.number_input("Pods available", min_value=1, value=6, step=1)

    run_button = st.button("Run Analysis")

with right_col:
    st.markdown('<div class="section-label">Situation Report</div>', unsafe_allow_html=True)

    if run_button:
        if run_full_pipeline and core.is_on_land(latitude, longitude):
            st.error("Coordinates are on land — choose an open-water location and try again.")
            st.stop()

        detection_result = None

        if uploaded_image is not None and core.DETECTION_AVAILABLE:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                tmp_file.write(uploaded_image.getbuffer())
                tmp_path = Path(tmp_file.name)

            detection_result = core.classify_drone_image(tmp_path)
            is_polluted = detection_result["is_polluted"]

            stamp_color = "#A8562E" if is_polluted else "#2C6459"
            stamp_text = "Pollution Detected" if is_polluted else "Clean — No Pollution"
            st.markdown(
                f'<span class="tahaara-stamp" style="border-color:{stamp_color};color:{stamp_color};">{stamp_text}</span>',
                unsafe_allow_html=True,
            )
            st.caption(f"Model confidence: {detection_result['confidence_pct']}% ({detection_result['confidence_level']})")

        elif manual_override:
            is_polluted = True
            st.markdown('<span class="tahaara-stamp">Manual Override — Treated as Pollution</span>', unsafe_allow_html=True)

        else:
            is_polluted = False
            st.info("Upload an image or enable manual override to proceed.")

        if not run_full_pipeline:
            st.info("Detection-only mode — drift and pod deployment skipped.")
        elif is_polluted:
            with st.spinner("Fetching live conditions and predicting drift..."):
                drift_result = core.predict_spill_direction(latitude, longitude, oil_type=oil_type)

            if not drift_result.get("valid", True):
                st.error(drift_result["reason"])
                st.stop()

            conditions = drift_result["conditions"]

            st.markdown(
                f"""
                <table class="ledger">
                    <tr><td class="label">Direction</td><td class="value">{drift_result['direction_label']}</td></tr>
                    <tr><td class="label">Confidence</td><td class="value">{drift_result['confidence_pct']}<span class="unit">%</span></td></tr>
                    <tr><td class="label">Drift speed</td><td class="value">{drift_result['drift_speed_kmh']}<span class="unit">km/h</span></td></tr>
                    <tr><td class="label">24h drift distance</td><td class="value">{drift_result['drift_24h_km']}<span class="unit">km</span></td></tr>
                </table>
                """,
                unsafe_allow_html=True,
            )
            st.caption(
                f"Wind {conditions['wind_speed_ms']:.1f} m/s from {conditions['wind_direction_deg']:.0f}° · "
                f"Current {conditions['current_speed_ms']:.2f} m/s from {conditions['current_direction_deg']:.0f}° · "
                f"Starting area {initial_area_km2} km²"
            )

            if drift_result["landfall_predicted"]:
                st.markdown(
                    '<span class="tahaara-stamp" style="border-color:#A8562E;color:#A8562E;">'
                    f'Landfall Predicted — ~{drift_result["landfall_hour"]}h</span>',
                    unsafe_allow_html=True,
                )
                st.caption(
                    f"Near ({drift_result['landfall_point'][0]:.4f}, {drift_result['landfall_point'][1]:.4f}). "
                    "Drift path capped at the coastline."
                )
                st.write("")

            spill_map = folium.Map(location=[latitude, longitude], zoom_start=10, tiles=None)
            folium.TileLayer(
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                attr="Esri, Maxar, Earthstar Geographics", name="Satellite", overlay=False,
            ).add_to(spill_map)
            folium.TileLayer(
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
                attr="Esri", name="Labels", overlay=True, control=True,
            ).add_to(spill_map)

            folium.CircleMarker(
                [latitude, longitude], radius=8, color="#EDE7D8", weight=2,
                fill=True, fill_color="#A8562E", fill_opacity=0.95, tooltip="Spill location",
            ).add_to(spill_map)

            path_color = "#A8562E" if drift_result["landfall_predicted"] else "#2C6459"
            folium.PolyLine(
                [[latitude, longitude], [drift_result["predicted_latitude"], drift_result["predicted_longitude"]]],
                color=path_color, weight=4, dash_array="8,10",
                tooltip=f"Predicted drift: {drift_result['direction_label']}",
            ).add_to(spill_map)

            end_tooltip = "Predicted landfall" if drift_result["landfall_predicted"] else "Predicted 24h position"
            folium.CircleMarker(
                [drift_result["predicted_latitude"], drift_result["predicted_longitude"]],
                radius=7, color="#EDE7D8", weight=2,
                fill=True, fill_color=path_color, fill_opacity=0.95, tooltip=end_tooltip,
            ).add_to(spill_map)

            deployment_plan = core.recommend_pod_deployment(
                drift_result, initial_area_km2=initial_area_km2, fleet_size=fleet_size
            )

            if deployment_plan.get("valid", True):
                for line in deployment_plan["lines"]:
                    color = "#A8562E" if line["priority"].startswith("Primary") else "#2C6459"
                    for pod in line["pods"]:
                        folium.CircleMarker(
                            [pod["latitude"], pod["longitude"]], radius=6, color="#0F2436", weight=1.5,
                            fill=True, fill_color=color, fill_opacity=0.95,
                            tooltip=f"{line['priority']} — pod {pod['pod_id']} ({line['intercept_hour']}h intercept)",
                        ).add_to(spill_map)

            folium.LayerControl(position="topright", collapsed=True).add_to(spill_map)

            with st.container(border=True):
                components.html(spill_map._repr_html_(), height=430)
                st.caption("Chart plate — projected drift and recommended pod lines · all times local")

            if deployment_plan.get("valid", True):
                st.markdown('<div class="section-label" style="margin-top:24px;">Pod Deployment Plan</div>', unsafe_allow_html=True)
                st.caption(deployment_plan["rationale"])
                for line in deployment_plan["lines"]:
                    st.markdown(
                        f"**{line['priority']} line — {line['intercept_hour']}h intercept** · "
                        f"{line['pod_count']} pods · est. slick radius {line['estimated_slick_radius_km']} km"
                    )
                st.markdown(
                    f'<table class="ledger"><tr><td class="label">Total pods recommended</td>'
                    f'<td class="value">{deployment_plan["total_pods_recommended"]}</td></tr></table>',
                    unsafe_allow_html=True,
                )
                if deployment_plan["uncovered_pods"] > 0:
                    st.markdown(
                        '<span class="tahaara-stamp" style="border-color:#A8562E;color:#A8562E;">'
                        f'Fleet Short — {deployment_plan["uncovered_pods"]} Pod(s) Uncovered</span>',
                        unsafe_allow_html=True,
                    )
    else:
        st.info("Fill in the inputs on the left and run the analysis.")
