"""Prediction engine for single login attempt scoring."""

from pathlib import Path
import json
import pickle
from typing import Any
import pandas as pd
import yaml


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file missing at '{path}'.")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _encode_value(label, encoder) -> int:
    text = str(label)
    if text in encoder.classes_:
        return int(encoder.transform([text])[0])
    return -1


def _prepare_features(record: dict[str, Any], encoders: dict, profiles: dict, feature_order: list[str]) -> pd.DataFrame:
    timestamp = pd.to_datetime(record["login_timestamp"], errors="coerce")
    if pd.isna(timestamp):
        raise ValueError("Invalid login_timestamp in record.")

    ip = str(record["ip_address"])
    country = str(record["country"])
    os_value = str(record["os"])
    browser = str(record["browser"])

    login_hour = int(timestamp.hour)
    is_night = int(login_hour >= 23 or login_hour <= 5)
    ip_velocity_1h = int(profiles.get("ip_velocity_baseline", {}).get(ip, 0))

    last_country_encoded = profiles.get("ip_country_last", {}).get(ip)
    encoded_country = _encode_value(country, encoders["country"])
    country_changed = int(last_country_encoded is not None and int(last_country_encoded) != encoded_country)

    historical_device = profiles.get("ip_device_mode", {}).get(ip, "")
    device_signature = f"{os_value}|{browser}"
    device_changed = int(historical_device != "" and device_signature != historical_device)

    rtt = float(record["round_trip_ms"])
    ip_mean = float(profiles.get("ip_rtt_mean", {}).get(ip, profiles.get("global_rtt_mean", 0.0)))
    ip_std = float(profiles.get("ip_rtt_std", {}).get(ip, profiles.get("global_rtt_std", 1.0)))
    if ip_std == 0:
        ip_std = float(profiles.get("global_rtt_std", 1.0)) or 1.0
    rtt_zscore = (rtt - ip_mean) / ip_std

    feature_row = {
        "country": encoded_country,
        "region": _encode_value(record["region"], encoders["region"]),
        "os": _encode_value(os_value, encoders["os"]),
        "browser": _encode_value(browser, encoders["browser"]),
        "device_type": _encode_value(record["device_type"], encoders["device_type"]),
        "round_trip_ms": rtt,
        "login_successful": int(record["login_successful"]),
        "is_attack_ip": int(record["is_attack_ip"]),
        "login_hour": login_hour,
        "is_night": is_night,
        "ip_velocity_1h": ip_velocity_1h,
        "country_changed": country_changed,
        "device_changed": device_changed,
        "rtt_zscore": rtt_zscore,
    }

    ordered = {col: feature_row.get(col, 0) for col in feature_order}
    return pd.DataFrame([ordered])


def predict_login(record: dict[str, Any]) -> dict[str, Any]:
    """Predict account takeover risk for a single login attempt."""
    model_dir = Path("models")
    reports_path = Path("reports/model_metrics.json")
    config = _load_yaml(Path("config.yaml"))

    required = [
        "login_timestamp",
        "ip_address",
        "country",
        "region",
        "os",
        "browser",
        "device_type",
        "round_trip_ms",
        "login_successful",
        "is_attack_ip",
    ]
    missing = [field for field in required if field not in record]
    if missing:
        raise ValueError(f"Record missing required keys: {missing}")

    if not reports_path.exists():
        raise FileNotFoundError("Missing reports/model_metrics.json. Train models first.")

    with reports_path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    best_model_name = metrics["best_model"]
    model_file = metrics["models"][best_model_name]["model_file"]

    with (model_dir / model_file).open("rb") as handle:
        model = pickle.load(handle)
    with (model_dir / "encoders.pkl").open("rb") as handle:
        encoders = pickle.load(handle)
    with (model_dir / "feature_profiles.json").open("r", encoding="utf-8") as handle:
        profiles = json.load(handle)
    with (model_dir / "feature_columns.json").open("r", encoding="utf-8") as handle:
        feature_order = json.load(handle)["columns"]

    x_input = _prepare_features(record, encoders, profiles, feature_order)
    probability = float(model.predict_proba(x_input)[0][1])
    prediction = int(probability >= 0.5)

    medium = float(config["risk_thresholds"]["medium"])
    high = float(config["risk_thresholds"]["high"])
    actions = config["actions"]
    if probability >= high:
        risk_level = "High"
        action = actions["high"]
    elif probability >= medium:
        risk_level = "Medium"
        action = actions["medium"]
    else:
        risk_level = "Low"
        action = actions["low"]

    return {
        "prediction": prediction,
        "probability": probability,
        "risk_level": risk_level,
        "recommended_action": action,
    }
