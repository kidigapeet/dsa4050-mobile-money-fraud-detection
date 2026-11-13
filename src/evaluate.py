"""Consolidate ML vs. DL benchmarks, demonstrate the Accuracy Paradox, and render
publication figures for Part C and the <=2,000-word report.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODELS_DIR = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figures"


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ml_data = json.loads((MODELS_DIR / "ml_metrics.json").read_text(encoding="utf-8"))
    dl_data = json.loads((MODELS_DIR / "dl_metrics.json").read_text(encoding="utf-8"))
    ae_dist = json.loads((MODELS_DIR / "ae_error_distribution.json").read_text(encoding="utf-8"))

    all_results = ml_data["ml_results"] + dl_data["dl_results"]
    pr_curves = {**ml_data["pr_curves"], **dl_data["pr_curves"]}
    ablations = [
        ml_data["ablation_no_balance_error"],
        dl_data["ablation_no_balance_error"],
    ]

    summary = {
        "benchmark_table": all_results,
        "ablation_table": ablations,
        "pr_curves": pr_curves,
        "smote_before": ml_data["smote_before"],
        "smote_after": ml_data["smote_after"],
    }
    (MODELS_DIR / "benchmark_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    df_res = pd.DataFrame(all_results)
    df_res.to_csv(MODELS_DIR / "benchmark_table.csv", index=False)

    plt.rcParams.update({
        "figure.dpi": 140,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    })

    # =========================================================================
    # Figure 4: Precision-Recall Curves & The Accuracy Paradox (Part C)
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

    palette = {
        "Logistic Regression": "#64748b",
        "Isolation Forest": "#f59e0b",
        "XGBoost + SMOTE": "#10b981",
        "Deep Autoencoder (Unsupervised)": "#8b5cf6",
        "Entity-Embedding MLP (Deep Learning)": "#ef4444",
    }
    for name, curve in pr_curves.items():
        row = df_res[df_res["model"] == name].iloc[0]
        axes[0].plot(
            curve["recall"],
            curve["precision"],
            label=f"{name} (PR-AUC={row['pr_auc']:.3f})",
            linewidth=2.2,
            color=palette.get(name, "#1e293b"),
        )
    axes[0].set_title("A. Precision–Recall Curves on Held-Out Test Stream", fontweight="bold")
    axes[0].set_xlabel("Recall (Fraction of Fraud Caught)")
    axes[0].set_ylabel("Precision (Fraction of Alerts that are True Fraud)")
    axes[0].set_xlim(0, 1.02)
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(loc="lower left", fontsize=8.2)
    axes[0].grid(alpha=0.25)

    # Panel B: Raw Accuracy vs PR-AUC & F1 (The Accuracy Paradox visual)
    plot_models = [
        "Naive Majority Baseline (All Legit)",
        "Static Rule Engine (isFlaggedFraud)",
        "Isolation Forest",
        "Logistic Regression",
        "Deep Autoencoder (Unsupervised)",
        "XGBoost + SMOTE",
        "Entity-Embedding MLP (Deep Learning)",
    ]
    short_labels = [
        "Naive All-Legit",
        "Static Rule",
        "Isolation Forest",
        "Logistic Reg.",
        "Deep Autoencoder",
        "XGBoost+SMOTE",
        "Embedding MLP",
    ]
    sub = df_res.set_index("model").loc[plot_models].reset_index()
    x = np.arange(len(short_labels))
    w = 0.26
    axes[1].bar(x - w, sub["accuracy"] * 100, width=w, label="Raw Accuracy (%)", color="#94a3b8")
    axes[1].bar(x, sub["pr_auc"] * 100, width=w, label="PR-AUC (×100)", color="#2563eb")
    axes[1].bar(x + w, sub["f1"] * 100, width=w, label="F1-Score (×100)", color="#dc2626")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(short_labels, rotation=24, ha="right", fontsize=8.5)
    axes[1].set_ylabel("Score (%)")
    axes[1].set_ylim(0, 112)
    axes[1].set_title("B. The Accuracy Paradox: 99% Accuracy with 0% Fraud Caught", fontweight="bold")
    axes[1].legend(loc="lower right", fontsize=8.5)
    axes[1].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_pr_curves_and_accuracy_paradox.png", bbox_inches="tight")
    plt.close(fig)

    # =========================================================================
    # Figure 5: Confusion Matrices Grid (5 trained models)
    # =========================================================================
    cm_models = [
        "Logistic Regression",
        "Isolation Forest",
        "XGBoost + SMOTE",
        "Deep Autoencoder (Unsupervised)",
        "Entity-Embedding MLP (Deep Learning)",
    ]
    fig, axes = plt.subplots(1, 5, figsize=(15.5, 3.3))
    for ax, mname in zip(axes, cm_models):
        r = df_res[df_res["model"] == mname].iloc[0]
        cm = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["Pred Legit", "Pred Fraud"],
            yticklabels=["True Legit", "True Fraud"],
        )
        short_title = mname.replace(" (Unsupervised)", "").replace(" (Deep Learning)", "")
        ax.set_title(f"{short_title}\nF1={r['f1']:.3f} | PR-AUC={r['pr_auc']:.3f}", fontsize=9, fontweight="bold")
    fig.suptitle("Held-Out Test Confusion Matrices Across Classical ML & Deep Learning Models", fontsize=11.5, fontweight="bold", y=1.05)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_confusion_matrices.png", bbox_inches="tight")
    plt.close(fig)

    # =========================================================================
    # Figure 6: Autoencoder Reconstruction Error Distribution (Part A & D)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8.8, 4.1))
    sns.kdeplot(
        ae_dist["legit_log10_mse"],
        fill=True,
        color="#2563eb",
        alpha=0.35,
        label=f"Legitimate Transactions (median MSE={ae_dist['median_legit_mse']:.3f})",
        ax=ax,
    )
    sns.kdeplot(
        ae_dist["fraud_log10_mse"],
        fill=True,
        color="#dc2626",
        alpha=0.35,
        label=f"Fraudulent Transactions (median MSE={ae_dist['median_fraud_mse']:.2f})",
        ax=ax,
    )
    ax.axvline(
        ae_dist["thr_log10_mse"],
        color="#111827",
        linestyle="--",
        linewidth=1.8,
        label=f"Validation F1 Anomaly Threshold (MSE={ae_dist['thr_mse']:.2f})",
    )
    ax.set_title("Unsupervised Deep Autoencoder: Reconstruction Error Separation (log10 MSE)", fontweight="bold")
    ax.set_xlabel("log10(Reconstruction Mean Squared Error)")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right", fontsize=8.8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig6_autoencoder_reconstruction_error.png", bbox_inches="tight")
    plt.close(fig)

    print("[evaluate] Saved benchmark_summary.json, benchmark_table.csv, and figures 4–6.")


if __name__ == "__main__":
    main()

