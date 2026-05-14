"""Data loading and stratified sampling for the RBA dataset."""

from pathlib import Path
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
DEFAULT_SAMPLE_SIZE = 500_000


def _print_distribution(series: pd.Series, label: str) -> None:
    counts = series.value_counts().sort_index()
    ratios = series.value_counts(normalize=True).sort_index()
    print(f"\n{label}")
    print("Counts:")
    print(counts.to_string())
    print("Ratios:")
    print(ratios.to_string())


def load_and_sample_data(input_path: str, output_path: str, sample_size: int = DEFAULT_SAMPLE_SIZE) -> Path:
    """Load CSV and create stratified sample preserving class ratio."""
    input_file = Path(input_path)
    output_file = Path(output_path)

    if not input_file.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at '{input_file}'. Place the Kaggle RBA CSV there and rerun."
        )

    print(f"Loading raw dataset from {input_file} ...")
    df = pd.read_csv(input_file)
    if "is_account_takeover" not in df.columns:
        raise ValueError("Column 'is_account_takeover' not found in source dataset.")

    _print_distribution(df["is_account_takeover"], "Original class distribution")

    if len(df) <= sample_size:
        print(f"Dataset has {len(df):,} rows (<= {sample_size:,}); saving full data as sample.")
        sampled_df = df.copy()
    else:
        test_size = sample_size / len(df)
        # train_test_split with stratify preserves class proportions exactly enough for large datasets.
        _, sampled_df = train_test_split(
            df,
            test_size=test_size,
            stratify=df["is_account_takeover"],
            random_state=RANDOM_STATE,
        )
        sampled_df = sampled_df.reset_index(drop=True)

    _print_distribution(sampled_df["is_account_takeover"], "Sampled class distribution")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    sampled_df.to_csv(output_file, index=False)
    print(f"Saved sampled data to {output_file}")
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load RBA CSV and produce a stratified sample.")
    parser.add_argument("--input", default="data/raw/rba-dataset.csv", help="Path to raw Kaggle CSV.")
    parser.add_argument("--output", default="data/processed/sample.csv", help="Path for sampled CSV.")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE, help="Number of sampled rows.")
    args = parser.parse_args()
    load_and_sample_data(args.input, args.output, args.sample_size)
