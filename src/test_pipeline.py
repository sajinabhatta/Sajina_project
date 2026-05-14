"""Pytest suite for end-to-end pipeline validation."""

from pathlib import Path
import json
import pandas as pd
from src.predict import predict_login


def test_feature_engineering_has_no_nulls():
    features_file = Path("data/processed/features.csv")
    assert features_file.exists(), "Run pipeline first to generate features.csv"
    features = pd.read_csv(features_file)
    assert int(features.isnull().sum().sum()) == 0


def test_model_files_exist():
    expected = [
        Path("models/logistic_regression.pkl"),
        Path("models/random_forest.pkl"),
        Path("models/xgboost.pkl"),
    ]
    for model_file in expected:
        assert model_file.exists(), f"Missing model file: {model_file}"


def test_predict_login_required_output_keys():
    sample_record = {
        "login_timestamp": "2021-11-01 23:45:00",
        "ip_address": "10.1.1.1",
        "country": "Norway",
        "region": "Oslo",
        "os": "Windows",
        "browser": "Chrome",
        "device_type": "Desktop",
        "round_trip_ms": 180.0,
        "login_successful": 1,
        "is_attack_ip": 0,
    }
    prediction = predict_login(sample_record)
    for key in ["prediction", "probability", "risk_level", "recommended_action"]:
        assert key in prediction


def test_smote_class_balance_recorded():
    metrics_file = Path("reports/model_metrics.json")
    assert metrics_file.exists(), "Missing model_metrics.json"
    with metrics_file.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    after = metrics["data_distribution"]["after_smote"]
    assert int(after["0"]) == int(after["1"])


def test_attack_pattern_returns_high_risk():
    attack_like = {
        "login_timestamp": "2021-12-05 02:15:00",
        "ip_address": "185.220.101.50",
        "country": "Russia",
        "region": "Moscow",
        "os": "Linux",
        "browser": "Tor Browser",
        "device_type": "Desktop",
        "round_trip_ms": 1500.0,
        "login_successful": 1,
        "is_attack_ip": 1,
    }
    result = predict_login(attack_like)
    assert result["risk_level"] == "High"
