"""
TAHAARA core inference module.

Loads the two trained models (detection, drift) once and exposes clean
functions any frontend — Streamlit, Flask, a notebook UI — can call without
knowing anything about how the models were trained.

Requires the saved model files to be present alongside this module:
    pollution_detector_v1.keras   (MobileNetV2 transfer + fine-tune, trained in tahaaradrone.py)
    tahaara_drift_model.joblib
    tahaara_drift_features.joblib
"""

import math
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import requests
import tensorflow as tf
from global_land_mask import globe

IMAGE_SIZE = (224, 224)

COMPASS_DIRECTIONS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
COMPASS_ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]

OIL_TYPE_PROPERTIES = {
    "crude_oil": {"spread_factor": 1.0},
    "light_crude": {"spread_factor": 1.3},
    "heavy_crude": {"spread_factor": 0.7},
    "diesel": {"spread_factor": 1.5},
}

# Preset starting-area categories for the manual area input. Practical
# usability buckets, not an official spill-classification standard — lets
# a responder describe size quickly before a precise number is available.
AREA_PRESETS_KM2 = {
    "Small": 0.3,
    "Medium": 2.0,
    "Large": 8.0,
}

# Threshold from the fine-tuned MobileNetV2 retrain (ROC/PR curve analysis,
# favouring recall since a missed spill is worse than a false alarm).
DETECTION_THRESHOLD = 0.316

DETECTION_AVAILABLE = False
try:
    _detection_model = tf.keras.models.load_model("pollution_detector_v1.keras")
    DETECTION_AVAILABLE = True
except (OSError, IOError, ValueError):
    _detection_model = None

_drift_model = joblib.load("tahaara_drift_model.joblib")
_drift_feature_columns = joblib.load("tahaara_drift_features.joblib")


def compass_direction_to_angle(direction):
    return COMPASS_ANGLES[COMPASS_DIRECTIONS.index(direction)]


def is_on_land(latitude, longitude):
    """
    True if this point is on land, not water. Uses a ~10 arcmin (about
    18.5 km) resolution global land/ocean mask — fine for open-water
    decisions, but not precise enough to resolve small islands or narrow
    channels. Good enough to stop a drift path from crossing a coastline;
    not a substitute for real bathymetry/coastline data in production.
    """
    return bool(globe.is_land(latitude, longitude))


def _project_point(latitude, longitude, bearing_deg, distance_km):
    """Move a point a given distance/bearing, correcting for longitude
    compression at higher latitudes (flat-earth approximation, fine at
    the sub-100km scale used here)."""
    angle_rad = math.radians(bearing_deg)
    delta_lat = (distance_km / 111.0) * math.cos(angle_rad)
    delta_lon = (distance_km / (111.0 * math.cos(math.radians(latitude)))) * math.sin(angle_rad)
    return latitude + delta_lat, longitude + delta_lon


# ---------------------------------------------------------------------------
# Model 1: detection
# preprocess_input is baked into the model graph (see tahaaradrone.py) —
# do NOT divide by 255 here, that double-scales the input and silently
# breaks predictions.
# ---------------------------------------------------------------------------
def classify_drone_image(image_path, threshold=DETECTION_THRESHOLD):
    if not DETECTION_AVAILABLE:
        raise RuntimeError(
            "Detection model not found. Make sure pollution_detector_v1.keras "
            "is uploaded alongside this module."
        )
    image = tf.keras.utils.load_img(image_path, target_size=IMAGE_SIZE)
    image_array = tf.keras.utils.img_to_array(image)
    image_batch = tf.expand_dims(image_array, axis=0)

    probability = float(_detection_model.predict(image_batch, verbose=0)[0][0])
    is_polluted = probability > threshold

    margin = abs(probability - threshold)
    if margin > 0.3:
        confidence_level = "High"
    elif margin > 0.15:
        confidence_level = "Medium"
    else:
        confidence_level = "Low"

    display_confidence_pct = round(
        (probability if is_polluted else 1 - probability) * 100, 1
    )

    return {
        "image_path": str(image_path),
        "is_polluted": bool(is_polluted),
        "classification": "POLLUTED" if is_polluted else "CLEAN",
        "pollution_probability": round(probability, 4),
        "confidence_pct": display_confidence_pct,
        "confidence_level": confidence_level,
        "threshold_used": threshold,
        "classified_at": datetime.now().isoformat(timespec="seconds"),
    }


# ---------------------------------------------------------------------------
# Live conditions: Open-Meteo (no API key required)
# ---------------------------------------------------------------------------
def fetch_live_conditions(latitude, longitude, timeout=10):
    conditions = {
        "latitude": latitude,
        "longitude": longitude,
        "fetched_at": datetime.now().isoformat(timespec="seconds"),
    }

    try:
        wind_response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": "wind_speed_10m,wind_direction_10m",
                "wind_speed_unit": "ms",
                "timezone": "auto",
            },
            timeout=timeout,
        )
        wind_data = wind_response.json().get("current", {})
        conditions["wind_speed_ms"] = wind_data.get("wind_speed_10m")
        conditions["wind_direction_deg"] = wind_data.get("wind_direction_10m")
    except (requests.RequestException, ValueError):
        conditions["wind_speed_ms"] = None
        conditions["wind_direction_deg"] = None

    try:
        marine_response = requests.get(
            "https://api.open-meteo.com/v1/marine",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": "ocean_current_velocity,ocean_current_direction",
                "timezone": "auto",
            },
            timeout=timeout,
        )
        marine_data = marine_response.json().get("current", {})
        conditions["current_speed_ms"] = marine_data.get("ocean_current_velocity")
        conditions["current_direction_deg"] = marine_data.get("ocean_current_direction")
    except (requests.RequestException, ValueError):
        conditions["current_speed_ms"] = None
        conditions["current_direction_deg"] = None

    return conditions


def _fill_missing_conditions(conditions):
    defaults = {
        "wind_speed_ms": 4.0,
        "wind_direction_deg": 315.0,
        "current_speed_ms": 0.2,
        "current_direction_deg": 135.0,
    }
    filled = dict(conditions)
    for key, default_value in defaults.items():
        if filled.get(key) is None:
            filled[key] = default_value
    return filled


# ---------------------------------------------------------------------------
# Model 2: drift
# Direction is driven by wind/current only, not spill area -- a bigger
# slick doesn't travel a different direction at leading order. Area
# matters for containment planning (below), not this step.
# ---------------------------------------------------------------------------
def predict_spill_direction(latitude, longitude, oil_type="crude_oil", month=None):
    if oil_type not in OIL_TYPE_PROPERTIES:
        raise ValueError(f"Unknown oil type '{oil_type}'. Choose from {list(OIL_TYPE_PROPERTIES)}.")

    if is_on_land(latitude, longitude):
        return {
            "valid": False,
            "reason": (
                f"Coordinates ({latitude:.4f}, {longitude:.4f}) are on land, not water. "
                "This location is not applicable for a spill drift prediction — "
                "pick a point in open water."
            ),
            "latitude": latitude,
            "longitude": longitude,
        }

    raw_conditions = fetch_live_conditions(latitude, longitude)
    conditions = _fill_missing_conditions(raw_conditions)
    month = month or datetime.now().month

    bearing_radians = math.radians(conditions["current_direction_deg"])
    feature_row = pd.DataFrame([{
        "current_speed_ms": conditions["current_speed_ms"],
        "current_bearing_sin": math.sin(bearing_radians),
        "current_bearing_cos": math.cos(bearing_radians),
        "latitude": latitude,
        "longitude": longitude,
        "month_sin": math.sin(2 * math.pi * month / 12),
        "month_cos": math.cos(2 * math.pi * month / 12),
    }])[_drift_feature_columns]

    predicted_label = _drift_model.predict(feature_row)[0]
    class_probabilities = _drift_model.predict_proba(feature_row)[0]
    confidence = float(np.max(class_probabilities)) * 100

    wind_drift_ms = conditions["wind_speed_ms"] * 0.03
    drift_speed_ms = conditions["current_speed_ms"] + wind_drift_ms
    spread_factor = OIL_TYPE_PROPERTIES[oil_type]["spread_factor"]
    drift_speed_kmh = drift_speed_ms * spread_factor * 3.6
    drift_24h_km = drift_speed_kmh * 24

    direction_deg = compass_direction_to_angle(predicted_label)

    step_km = 0.5
    traveled_km = 0.0
    current_lat, current_lon = latitude, longitude
    landfall_hour = None
    landfall_point = None

    while traveled_km < drift_24h_km:
        next_km = min(step_km, drift_24h_km - traveled_km)
        next_lat, next_lon = _project_point(current_lat, current_lon, direction_deg, next_km)

        if is_on_land(next_lat, next_lon):
            landfall_point = (round(current_lat, 5), round(current_lon, 5))
            landfall_hour = round(traveled_km / drift_speed_kmh, 1) if drift_speed_kmh > 0 else 0.0
            break

        current_lat, current_lon = next_lat, next_lon
        traveled_km += next_km

    predicted_latitude, predicted_longitude = current_lat, current_lon

    result = {
        "valid": True,
        "latitude": latitude,
        "longitude": longitude,
        "oil_type": oil_type,
        "direction_label": predicted_label,
        "direction_deg": direction_deg,
        "confidence_pct": round(confidence, 1),
        "drift_speed_kmh": round(drift_speed_kmh, 2),
        "drift_24h_km": round(drift_24h_km, 1),
        "predicted_latitude": round(predicted_latitude, 5),
        "predicted_longitude": round(predicted_longitude, 5),
        "landfall_predicted": landfall_point is not None,
        "landfall_hour": landfall_hour,
        "landfall_point": landfall_point,
        "conditions": conditions,
    }
    return result


# ---------------------------------------------------------------------------
# Pod deployment recommendation (geometry/physics algorithm, not a trained
# model -- there is no labelled dataset of "optimal pod placement").
#
# Area growth: Fay's (1971) gravity-viscous spreading law -- slick area
# grows approximately as t^0.5 in the hours-to-days regime relevant here.
# Simplified application of Fay's scaling exponent (not the full volume/
# density/viscosity equations), applied to a user-supplied starting area
# so it doesn't require guessing oil property constants we have no real
# data for. Growth rate is lightly adjusted per oil type (thinner oils
# spread faster) as our own extension on top of the cited base law.
# ---------------------------------------------------------------------------
FAY_GRAVITY_VISCOUS_EXPONENT = 0.5


def _estimate_slick_radius_km(hours_elapsed, initial_area_km2, oil_type="crude_oil"):
    spread_factor = OIL_TYPE_PROPERTIES.get(oil_type, {}).get("spread_factor", 1.0)
    exponent = FAY_GRAVITY_VISCOUS_EXPONENT * spread_factor
    area_km2 = initial_area_km2 * (1 + hours_elapsed) ** exponent
    return math.sqrt(area_km2 / math.pi)


def recommend_pod_deployment(
    drift_result,
    initial_area_km2,
    pod_reach_km=0.3,
    primary_hour=3,
    secondary_hour=6,
    fleet_size=None,
):
    """
    initial_area_km2 is required (no silent fallback) -- this is the fix
    for the earlier bug where a hardcoded default made pod counts nearly
    identical for every spill regardless of real size.

    fleet_size, if given, caps total pods; demand beyond that is reported
    as an explicit uncovered gap, with secondary-line coverage trimmed
    first, rather than silently dropped.
    """
    if drift_result.get("valid") is False:
        return {"valid": False, "reason": drift_result["reason"]}

    latitude = drift_result["latitude"]
    longitude = drift_result["longitude"]
    direction_deg = drift_result["direction_deg"]
    drift_speed_kmh = drift_result["drift_speed_kmh"]
    oil_type = drift_result.get("oil_type", "crude_oil")
    landfall_hour = drift_result.get("landfall_hour")
    landfall_point = drift_result.get("landfall_point")
    perpendicular_deg = (direction_deg + 90) % 360

    lines = []
    for hour, priority in [(primary_hour, "Primary"), (secondary_hour, "Secondary")]:
        is_past_landfall = landfall_hour is not None and hour >= landfall_hour

        if is_past_landfall:
            center_lat, center_lon = landfall_point
            priority_label = f"{priority} (coastal — landfall expected)"
            radius_km = _estimate_slick_radius_km(landfall_hour, initial_area_km2, oil_type)
        else:
            distance_km = drift_speed_kmh * hour
            center_lat, center_lon = _project_point(latitude, longitude, direction_deg, distance_km)
            priority_label = priority
            radius_km = _estimate_slick_radius_km(hour, initial_area_km2, oil_type)

        pods_needed = max(1, math.ceil((2 * radius_km) / (2 * pod_reach_km)))
        offsets_km = np.linspace(-radius_km, radius_km, pods_needed)

        pods = []
        for i, offset_km in enumerate(offsets_km):
            pod_lat, pod_lon = _project_point(center_lat, center_lon, perpendicular_deg, offset_km)
            if is_on_land(pod_lat, pod_lon):
                continue
            pods.append({
                "pod_id": f"{priority[0]}{hour}-{i + 1}",
                "latitude": round(pod_lat, 5),
                "longitude": round(pod_lon, 5),
            })

        lines.append({
            "priority": priority_label,
            "intercept_hour": hour,
            "intercept_center": {"latitude": round(center_lat, 5), "longitude": round(center_lon, 5)},
            "estimated_slick_radius_km": round(radius_km, 2),
            "pods": pods,
            "pod_count": len(pods),
        })

    total_pods_needed = sum(line["pod_count"] for line in lines)
    uncovered = 0
    if fleet_size is not None and total_pods_needed > fleet_size:
        uncovered = total_pods_needed - fleet_size
        remaining_budget = fleet_size
        for line in lines:
            keep = min(len(line["pods"]), remaining_budget)
            line["pods"] = line["pods"][:keep]
            line["pod_count"] = keep
            remaining_budget -= keep

    total_pods = sum(line["pod_count"] for line in lines)

    rationale = (
        f"Primary containment line at {primary_hour}h intercept ahead of the "
        f"predicted {drift_result['direction_label']} drift path, with a "
        f"secondary backup line at {secondary_hour}h. Slick growth uses "
        f"Fay's (1971) gravity-viscous scaling (area ~ t^0.5) from a "
        f"starting area of {initial_area_km2} km²."
    )
    if landfall_hour is not None:
        rationale += (
            f" Landfall predicted at {landfall_hour}h — lines due after that "
            f"point are repositioned to the coastline for shoreline protection."
        )
    if uncovered > 0:
        rationale += (
            f" WARNING: {uncovered} more pod(s) would be needed for full "
            f"coverage than the available fleet of {fleet_size} — "
            f"secondary-line coverage has been reduced to fit."
        )

    return {
        "valid": True,
        "spill_latitude": latitude,
        "spill_longitude": longitude,
        "direction_label": drift_result["direction_label"],
        "lines": lines,
        "total_pods_recommended": total_pods,
        "total_pods_needed": total_pods_needed,
        "uncovered_pods": uncovered,
        "rationale": rationale,
    }
