"""Run the complete ML workflow from raw data to diagrams."""

from pathlib import Path
from data_loader import load_and_sample_data
from feature_engineer import engineer_features
from train import train_models
from generate_diagrams import main as generate_diagrams


def run() -> None:
    raw = Path("data/raw/rba-dataset.csv")
    sample = Path("data/processed/sample.csv")
    features = Path("data/processed/features.csv")
    encoders = Path("models/encoders.pkl")
    profiles = Path("models/feature_profiles.json")

    load_and_sample_data(str(raw), str(sample), sample_size=500_000)
    engineer_features(str(sample), str(features), str(encoders), str(profiles))
    train_models(str(features), "models", "reports")
    generate_diagrams()
    print("Pipeline completed successfully.")


if __name__ == "__main__":
    run()
