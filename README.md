# Detection of Attempting Login Using Machine Learning

Final year project for **Sajina Bhatta (Student No: 230238118)**.

## Project Overview

This project builds an end-to-end machine learning system to detect suspicious login attempts and likely account takeover events. The target variable is `is_account_takeover`, predicted from network, device, location, and temporal login behavior features.

## Research Questions

1. Can account takeover attempts be reliably detected from login telemetry?
2. Which model family performs best on this dataset: linear, tree-based ensemble, or boosting?
3. Do engineered behavioral features (velocity, country/device change, RTT anomaly) improve detection performance?

## Hypothesis

If a login attempt deviates from normal user patterns (time, location, device signature, and network latency), then a supervised ML model can identify elevated takeover risk and recommend an actionable response.

## Dataset

- **Source:** Kaggle RBA Dataset (synthetic records from a Norwegian SSO service, 2020-2021)
- **Raw volume:** 33M+ records
- **Sample used in pipeline:** 500,000 stratified rows

### Column Definitions

- `login_timestamp`: Login event timestamp
- `ip_address`: Source IP address
- `country`: Login country
- `region`: Login region/state
- `os`: Operating system
- `browser`: Browser family
- `device_type`: Device category
- `round_trip_ms`: Network latency in milliseconds
- `login_successful`: Login success flag
- `is_attack_ip`: Whether IP is known suspicious
- `is_account_takeover`: **Target label** (0 = normal, 1 = takeover)

## Project Structure

```text
data/
  raw/                    # Original Kaggle dataset CSV
  processed/              # sample.csv and features.csv
notebooks/                # Optional exploratory notebooks
src/
  data_loader.py          # Stratified sampling
  feature_engineer.py     # Feature engineering + encoding
  train.py                # Model training + evaluation + metrics
  generate_diagrams.py    # Report-ready plots/figures
  predict.py              # Single-record prediction engine
  run_pipeline.py         # End-to-end orchestrator
  test_pipeline.py        # Pytest validation suite
models/                   # Saved model artifacts and metadata
diagrams/                 # Generated PNG diagrams at 300 DPI
dashboard/                # Flask web app + templates + CSS
reports/                  # model_metrics.json
config.yaml               # Risk threshold/action configuration
requirements.txt          # Pinned dependencies
```

## Installation (Python 3.10)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run End-to-End Pipeline

1. Place Kaggle CSV at `data/raw/rba-dataset.csv`
2. Run:

```bash
python src/run_pipeline.py
```

This executes, in order:
- data loading + stratified sampling
- feature engineering + label encoding
- model training with SMOTE (train split only)
- evaluation + metrics logging
- diagram generation

## Launch Dashboard

```bash
python dashboard/app.py
```

Open [http://localhost:5000](http://localhost:5000).

## Diagrams Generated

All diagrams are saved to `diagrams/` as high-resolution PNG files:

1. `a_system_architecture.png`: Full pipeline flowchart (input -> preprocessing -> model -> action output)
2. `b_class_imbalance.png`: Class distribution before and after SMOTE
3. `c_correlation_heatmap.png`: Numeric feature correlation matrix
4. `d_feature_importance.png`: Top 15 Random Forest feature importances
5. `e_roc_curves.png`: ROC comparison for all models
6. `f_confusion_matrices.png`: Confusion matrices in 1x3 layout
7. `g_login_hours.png`: Login hour distribution by class
8. `h_country_attack_map.png`: Country-level attack choropleth

## Results Summary Table

Pipeline outputs metrics to `reports/model_metrics.json`. Populate the final report table with values from that file:

| Model | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | from JSON | from JSON | from JSON | from JSON |
| Random Forest | from JSON | from JSON | from JSON | from JSON |
| XGBoost | from JSON | from JSON | from JSON | from JSON |

## Testing

Run unit tests with:

```bash
pytest src/test_pipeline.py
```

Tests cover:
- feature null checks
- model artifact existence
- prediction contract keys
- SMOTE class balancing validation
- high-risk attack-pattern end-to-end behavior

## Notes for University Report

- Risk thresholds and recommended actions are controlled by `config.yaml`.
- All project paths are relative for portability.
- Missing file scenarios raise clear error messages for reproducibility.
