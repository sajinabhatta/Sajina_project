"""Programmatic report diagram generator."""

from pathlib import Path
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns

plt.style.use("seaborn-v0_8-whitegrid")
TEAL = "#2A9D8F"
CORAL = "#E76F51"
GRAY = "#6C757D"
LIGHT_GRAY = "#DEE2E6"

def create_diagram_a_architecture(output_dir: Path) -> None:
    """(a) System architecture diagram using Matplotlib."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.axis('off')
    
    # Define boxes: x, y, width, height, text, color
    boxes = [
        (0.05, 0.5, 0.15, 0.2, "Raw Data\n(33M Records)", LIGHT_GRAY),
        (0.25, 0.5, 0.15, 0.2, "Data Loader\n& Sampler", TEAL),
        (0.45, 0.5, 0.15, 0.2, "Feature\nEngineering", TEAL),
        (0.65, 0.5, 0.15, 0.2, "Model Training\n& SMOTE", TEAL),
        (0.85, 0.7, 0.1, 0.15, "Allow", "#81B29A"),
        (0.85, 0.525, 0.1, 0.15, "Flag", "#F4A261"),
        (0.85, 0.35, 0.1, 0.15, "Block", CORAL),
        (0.65, 0.2, 0.15, 0.15, "Prediction\nEngine", GRAY)
    ]
    
    for x, y, w, h, text, color in boxes:
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", 
                                      linewidth=1, edgecolor=GRAY, facecolor=color)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', 
                fontsize=10, fontweight='bold', color='black' if color == LIGHT_GRAY else 'white')
        
    # Draw arrows
    arrows = [
        (0.2, 0.6, 0.25, 0.6),
        (0.4, 0.6, 0.45, 0.6),
        (0.6, 0.6, 0.65, 0.6),
        (0.8, 0.6, 0.85, 0.775), # To Allow
        (0.8, 0.6, 0.85, 0.6),   # To Flag
        (0.8, 0.6, 0.85, 0.425), # To Block
        (0.725, 0.35, 0.725, 0.5)  # Engine to Training (Model Load)
    ]
    
    for ax_x1, ax_y1, ax_x2, ax_y2 in arrows:
        ax.annotate("", xy=(ax_x2, ax_y2), xytext=(ax_x1, ax_y1),
                    arrowprops=dict(arrowstyle="->", color=GRAY, lw=2))
        
    ax.text(0.725, 0.425, "Loads Model", ha='center', va='center', rotation=90, fontsize=8, color=GRAY)
    
    plt.title("System Architecture: Threat Detection Pipeline", fontsize=14, fontweight='bold', color=GRAY)
    plt.tight_layout()
    plt.savefig(output_dir / "a_system_architecture.png", dpi=300, bbox_inches="tight")
    plt.close()

def create_diagram_b_imbalance(metrics_data: dict, output_dir: Path) -> None:
    """(b) Class imbalance bar chart."""
    dist = metrics_data.get('data_distribution', {})
    if not dist:
        return
    
    before = dist['before_smote']
    after = dist['after_smote']
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=300)
    
    labels = ['Normal (0)', 'Attack (1)']
    
    # Before SMOTE
    axes[0].bar(labels, [before.get('0', 0), before.get('1', 0)], color=[TEAL, CORAL])
    axes[0].set_title('Before SMOTE (Training Set)', fontweight='bold')
    axes[0].set_ylabel('Number of Login Attempts')
    
    # After SMOTE
    axes[1].bar(labels, [after.get('0', 0), after.get('1', 0)], color=[TEAL, CORAL])
    axes[1].set_title('After SMOTE (Training Set)', fontweight='bold')
    
    plt.suptitle("Class Imbalance Resolution using SMOTE", fontsize=14, fontweight='bold', color=GRAY)
    plt.tight_layout()
    plt.savefig(output_dir / "b_class_imbalance.png", dpi=300, bbox_inches="tight")
    plt.close()

def create_diagram_c_correlation(features_df: pd.DataFrame, output_dir: Path) -> None:
    """(c) Correlation heatmap."""
    numeric_df = features_df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    
    plt.figure(figsize=(10, 8), dpi=300)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap="RdBu_r", center=0, annot=True, fmt=".2f", 
                square=True, linewidths=.5, cbar_kws={"shrink": .8})
    plt.title("Correlation Heatmap of Features", fontsize=14, fontweight='bold', color=GRAY)
    plt.tight_layout()
    plt.savefig(output_dir / "c_correlation_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()

def create_diagram_d_feature_importance(metrics_data: dict, output_dir: Path) -> None:
    """(d) Feature importance chart."""
    rf_data = metrics_data.get('models', {}).get('Random Forest')
    if not rf_data or "feature_importances" not in rf_data:
        return
    
    importances = rf_data['feature_importances']
    # Sort and get top 15
    sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:15]
    features = [x[0] for x in sorted_imp]
    scores = [x[1] for x in sorted_imp]
    
    plt.figure(figsize=(10, 6), dpi=300)
    sns.barplot(x=scores, y=features, color=TEAL)
    plt.title("Top 15 Feature Importances (Random Forest)", fontsize=14, fontweight='bold', color=GRAY)
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(output_dir / "d_feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close()

def create_diagram_e_roc(metrics_data: dict, output_dir: Path) -> None:
    """(e) ROC curve comparison."""
    plt.figure(figsize=(8, 6), dpi=300)
    colors = [TEAL, CORAL, '#E9C46A']
    
    for i, (name, data) in enumerate(metrics_data.get('models', {}).items()):
        roc = data.get('roc_curve', {})
        auc = data.get('roc_auc', 0)
        if roc:
            plt.plot(roc['fpr'], roc['tpr'], color=colors[i % len(colors)], lw=2, label=f"{name} (AUC = {auc:.3f})")
            
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison', fontsize=14, fontweight='bold', color=GRAY)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "e_roc_curves.png", dpi=300, bbox_inches="tight")
    plt.close()

def create_diagram_f_confusion_matrices(metrics_data: dict, output_dir: Path) -> None:
    """(f) Confusion matrices."""
    models = list(metrics_data.get('models', {}).keys())
    if not models:
        return
    
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4), dpi=300)
    if len(models) == 1: axes = [axes]
    
    for ax, name in zip(axes, models):
        cm = np.array(metrics_data['models'][name]['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt="d", cmap=sns.light_palette(TEAL, as_cmap=True), cbar=False, ax=ax)
        ax.set_title(f"{name}", fontweight='bold')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_xticklabels(['Normal', 'Attack'])
        ax.set_yticklabels(['Normal', 'Attack'])
        
    plt.suptitle("Confusion Matrices on Test Set", fontsize=14, fontweight='bold', color=GRAY, y=1.05)
    plt.tight_layout()
    plt.savefig(output_dir / "f_confusion_matrices.png", dpi=300, bbox_inches="tight")
    plt.close()

def create_diagram_g_login_hours(features_df: pd.DataFrame, output_dir: Path) -> None:
    """(g) Login hour distribution."""
    if "login_hour" not in features_df.columns or "is_account_takeover" not in features_df.columns:
        return
        
    plt.figure(figsize=(10, 6), dpi=300)
    sns.histplot(data=features_df, x="login_hour", hue="is_account_takeover",
                 multiple="stack", bins=24, palette=[TEAL, CORAL])
    plt.title("Login Attempts by Hour of Day", fontsize=14, fontweight='bold', color=GRAY)
    plt.xlabel("Hour of Day (0-23)")
    plt.ylabel("Count")
    # Replace legend labels
    plt.legend(title="Class", labels=["Normal", "Attack"])
    plt.tight_layout()
    plt.savefig(output_dir / "g_login_hours.png", dpi=300, bbox_inches="tight")
    plt.close()

def create_diagram_h_attack_map(sample_df: pd.DataFrame, output_dir: Path) -> None:
    """(h) Country-level attack map using Plotly."""
    if "country" not in sample_df.columns or "is_account_takeover" not in sample_df.columns:
        return
        
    attack_df = sample_df[sample_df["is_account_takeover"] == 1]
    country_counts = attack_df.groupby("country").size().reset_index(name="Attack Count")
    
    fig = px.choropleth(
        country_counts,
        locations="country",
        locationmode="country names",
        color="Attack Count",
        hover_name="country",
        color_continuous_scale=[(0, LIGHT_GRAY), (0.5, TEAL), (1, CORAL)],
        title="Country-Level Account Takeover Attempts",
    )
    fig.update_layout(title_x=0.5, margin={"r": 0, "t": 40, "l": 0, "b": 0})
    fig.write_image(output_dir / "h_country_attack_map.png", width=1600, height=900, scale=2)

def main() -> None:
    metrics_file = Path("reports/model_metrics.json")
    features_file = Path("data/processed/features.csv")
    sample_file = Path("data/processed/sample.csv")
    output_dir = Path("diagrams")

    if not metrics_file.exists():
        raise FileNotFoundError("Missing reports/model_metrics.json. Run training first.")
    if not features_file.exists():
        raise FileNotFoundError("Missing data/processed/features.csv. Run feature engineering first.")
    if not sample_file.exists():
        raise FileNotFoundError("Missing data/processed/sample.csv. Run data loading first.")

    with metrics_file.open("r", encoding="utf-8") as handle:
        metrics_data = json.load(handle)
    features_df = pd.read_csv(features_file)
    sample_df = pd.read_csv(sample_file)
    output_dir.mkdir(parents=True, exist_ok=True)

    create_diagram_a_architecture(output_dir)
    create_diagram_b_imbalance(metrics_data, output_dir)
    create_diagram_c_correlation(features_df, output_dir)
    create_diagram_d_feature_importance(metrics_data, output_dir)
    create_diagram_e_roc(metrics_data, output_dir)
    create_diagram_f_confusion_matrices(metrics_data, output_dir)
    create_diagram_g_login_hours(features_df, output_dir)
    create_diagram_h_attack_map(sample_df, output_dir)
    print(f"Saved diagrams to {output_dir.resolve()}")

if __name__ == "__main__":
    main()
