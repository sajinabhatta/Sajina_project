import pandas as pd
import numpy as np
import random
import os
from datetime import datetime, timedelta

def generate_rba_dummy_data(output_path, num_records=500000):
    """
    Generates a synthetic dataset mimicking the Norwegian SSO RBA dataset.
    This ensures the pipeline runs correctly without requiring the 33M row download.
    """
    print(f"Generating {num_records} synthetic RBA login records...")
    
    np.random.seed(42)
    random.seed(42)
    
    # Simulate Date Ranges (2020 - 2021)
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2021, 12, 31)
    date_diff = (end_date - start_date).total_seconds()
    
    timestamps = [start_date + timedelta(seconds=random.randint(0, int(date_diff))) for _ in range(num_records)]
    
    # IPs - let's make a pool of 50,000 distinct IP addresses
    print("Generating IP pool...")
    ip_pool = [f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}" for _ in range(50000)]
    
    countries = ["NO", "US", "IN", "DE", "GB", "SE", "DK", "FI", "RU", "CN"]
    regions = ["Oslo", "Viken", "Vestland", "Rogaland", "Other"]
    oses = ["Windows", "MacOS", "Linux", "iOS", "Android"]
    browsers = ["Chrome", "Firefox", "Safari", "Edge", "Opera"]
    device_types = ["desktop", "mobile", "tablet"]
    
    # We want a class imbalance. Say 2% of attempts are account takeovers
    is_takeover = np.random.choice([0, 1], size=num_records, p=[0.98, 0.02])
    
    print("Assigning attributes...")
    df = pd.DataFrame({
        "login_timestamp": timestamps,
        "is_account_takeover": is_takeover
    })
    
    # Introduce patterns based on the target class to make the model training meaningful
    
    # 1. Takeovers often come from foreign countries or specific risk countries (e.g. RU, CN)
    df["country"] = np.where(df["is_account_takeover"] == 1, 
                             np.random.choice(["RU", "CN", "US", "DE"], size=num_records, p=[0.4, 0.4, 0.1, 0.1]),
                             np.random.choice(countries, size=num_records, p=[0.7, 0.05, 0.05, 0.05, 0.05, 0.04, 0.03, 0.01, 0.01, 0.01]))
    
    # 2. Add round trip ms. Takeovers might have higher ping due to VPN proxy
    df["round_trip_ms"] = np.where(df["is_account_takeover"] == 1,
                                   np.random.normal(500, 200, size=num_records),  # Mean 500ms
                                   np.random.normal(50, 20, size=num_records))    # Mean 50ms
    df["round_trip_ms"] = df["round_trip_ms"].clip(lower=5).astype(int)
    
    # 3. is_attack_ip
    # We say 80% of takeovers come from known attack IPs.
    df["is_attack_ip"] = np.where(df["is_account_takeover"] == 1,
                                  np.random.choice([0, 1], size=num_records, p=[0.2, 0.8]),
                                  np.random.choice([0, 1], size=num_records, p=[0.99, 0.01]))
    
    # Random categorical features
    df["region"] = np.random.choice(regions, size=num_records)
    df["os"] = np.random.choice(oses, size=num_records, p=[0.5, 0.2, 0.05, 0.15, 0.1])
    df["browser"] = np.random.choice(browsers, size=num_records, p=[0.6, 0.15, 0.15, 0.08, 0.02])
    df["device_type"] = np.random.choice(device_types, size=num_records, p=[0.6, 0.35, 0.05])
    
    # Assign IPs.
    df["ip_address"] = np.random.choice(ip_pool, size=num_records)
    
    # Login successful (usually takeovers have higher fail rate initially, but we are looking at total attempts)
    df["login_successful"] = np.where(df["is_account_takeover"] == 1,
                                      np.random.choice([False, True], size=num_records, p=[0.7, 0.3]),
                                      np.random.choice([False, True], size=num_records, p=[0.1, 0.9]))
    
    # Order columns
    df = df[["login_timestamp", "ip_address", "country", "region", "os", "browser", "device_type", "round_trip_ms", "login_successful", "is_attack_ip", "is_account_takeover"]]
    
    # Sort by timestamp
    df = df.sort_values(by="login_timestamp").reset_index(drop=True)
    
    print(f"Saving to {output_path}...")
    df.to_csv(output_path, index=False)
    print("Done generating synthetic data.")

if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    generate_rba_dummy_data("data/raw/rba-dataset.csv")
