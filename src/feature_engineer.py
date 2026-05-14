"""Feature engineering pipeline for login takeover detection."""

from pathlib import Path
import argparse
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


CATEGORICAL_COLUMNS = ["country", "region", "os", "browser", "device_type"]


def _safe_datetime(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.isna().any():
        raise ValueError("Found invalid values in 'login_timestamp'.")
    return parsed


def _encode_categorical(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    encoders = {}
    for col in CATEGORICAL_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Required categorical column '{col}' missing.")
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col].astype(str))
        encoders[col] = encoder
    return df, encoders


def engineer_features(input_path: str, output_path: str, encoder_path: str, profile_path: str) -> Path:
    """Create engineered features and persist artifacts required for inference."""
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Sample file not found at '{input_file}'.")

    print(f"Loading sample data from {input_file} ...")
    df = pd.read_csv(input_file)
    required = {
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
        "is_account_takeover",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input data is missing required columns: {sorted(missing)}")

    df["login_timestamp"] = _safe_datetime(df["login_timestamp"])
    df = df.sort_values(["ip_address", "login_timestamp"]).reset_index(drop=True)

    # (a) Hour of login from timestamp.
    df["login_hour"] = df["login_timestamp"].dt.hour
    # (b) Night logins are higher risk (23:00-05:59).
    df["is_night"] = ((df["login_hour"] >= 23) | (df["login_hour"] <= 5)).astype(int)

    # (c) IP velocity in previous one hour window.
    rolling_counts = (
        df.groupby("ip_address")
        .rolling("1h", on="login_timestamp")["is_account_takeover"]
        .count()
        .reset_index(level=0, drop=True)
        .reset_index(drop=True)
    )
    # Assign by position to avoid duplicate timestamp-index reindexing issues in pandas.
    df["ip_velocity_1h"] = (rolling_counts - 1).clip(lower=0).fillna(0).astype(int).to_numpy()

    # (d) Country change relative to most recent previous login for same IP.
    df["prev_country"] = df.groupby("ip_address")["country"].shift(1)
    df["country_changed"] = ((df["country"] != df["prev_country"]) & df["prev_country"].notna()).astype(int)

    # (e) Device change relative to historical mode signature (os + browser).
    df["device_signature"] = df["os"].astype(str) + "|" + df["browser"].astype(str)
    historical_mode = (
        df.groupby("ip_address")["device_signature"]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])
        .to_dict()
    )
    df["device_changed"] = df.apply(
        lambda row: int(row["device_signature"] != historical_mode.get(row["ip_address"], row["device_signature"])),
        axis=1,
    )

    # (f) RTT z-score relative to per-IP mean/std proxying unusual network path.
    ip_rtt_mean = df.groupby("ip_address")["round_trip_ms"].transform("mean")
    ip_rtt_std = df.groupby("ip_address")["round_trip_ms"].transform("std").replace(0, np.nan)
    df["rtt_zscore"] = ((df["round_trip_ms"] - ip_rtt_mean) / ip_rtt_std).fillna(0.0)

    for bool_col in ["login_successful", "is_attack_ip", "is_account_takeover"]:
        df[bool_col] = df[bool_col].astype(int)

    df, encoders = _encode_categorical(df)

    # Persist inference profiles to reproduce feature logic for single-record predictions.
    profiles = {
        "ip_country_last": (
            df.sort_values("login_timestamp")
            .groupby("ip_address")
            .tail(1)
            .set_index("ip_address")["country"]
            .to_dict()
        ),
        "ip_device_mode": {
            ip: f"{sig.split('|')[0]}|{sig.split('|')[1]}" for ip, sig in historical_mode.items()
        },
        "ip_rtt_mean": df.groupby("ip_address")["round_trip_ms"].mean().to_dict(),
        "ip_rtt_std": df.groupby("ip_address")["round_trip_ms"].std().fillna(0.0).to_dict(),
        "ip_velocity_baseline": df.groupby("ip_address")["ip_velocity_1h"].max().to_dict(),
        "global_rtt_mean": float(df["round_trip_ms"].mean()),
        "global_rtt_std": float(df["round_trip_ms"].std(ddof=0) or 1.0),
    }

    drop_columns = ["login_timestamp", "ip_address", "prev_country", "device_signature"]
    features_df = df.drop(columns=drop_columns).fillna(0)

    output_file = Path(output_path)
    encoder_file = Path(encoder_path)
    profile_file = Path(profile_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    encoder_file.parent.mkdir(parents=True, exist_ok=True)
    profile_file.parent.mkdir(parents=True, exist_ok=True)

    features_df.to_csv(output_file, index=False)
    with encoder_file.open("wb") as handle:
        pickle.dump(encoders, handle)
    with profile_file.open("w", encoding="utf-8") as handle:
        json.dump(profiles, handle, indent=2)

    print(f"Saved engineered features to {output_file}")
    print(f"Saved label encoders to {encoder_file}")
    print(f"Saved inference profiles to {profile_file}")
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Engineer model features from sampled login data.")
    parser.add_argument("--input", default="data/processed/sample.csv")
    parser.add_argument("--output", default="data/processed/features.csv")
    parser.add_argument("--encoders", default="models/encoders.pkl")
    parser.add_argument("--profiles", default="models/feature_profiles.json")
    args = parser.parse_args()
    engineer_features(args.input, args.output, args.encoders, args.profiles)
