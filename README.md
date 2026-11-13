# Real-Time Mobile Money & Credit Card Fraud Detection

**DSA 4050 — 20-Mark Packaged Write-Up, Notebooks, Models & Interactive Streamlit Simulator**

This repository contains the complete, tested solution for **Real-Time Mobile Money & Credit Card Fraud Detection**:
- **Part (A) — Why Deep Learning:** Unsupervised **Deep Autoencoder** (`15 → 32 → 16 → 6 → 16 → 32 → 15`) learning a compressed 6-D latent manifold of normal user behavior + Supervised **Entity-Embedding MLP** (`nn.Embedding` for hour of day and transaction channel fused with continuous financial ratios and Autoencoder latent features) in **PyTorch**.
- **Part (B) — Dataset Mining & Class Presentation:** PaySim Mobile Money log analysis, proving that 100% of fraud occurs in `TRANSFER` and `CASH_OUT`, uncovering nocturnal attack spikes (`00:00–05:00`), and demonstrating why legacy static rule engines (`isFlaggedFraud`) fail.
- **Part (C) — ML vs. DL Benchmarking & The Accuracy Paradox:** Head-to-head comparison of **Logistic Regression**, **Isolation Forest**, and **XGBoost + SMOTE** against the **Deep Autoencoder** and **Entity-Embedding MLP** using **PR-AUC**, **F1-Score**, **Confusion Matrices**, and an explicit demonstration of the **Accuracy Paradox** (where a naive all-legitimate classifier scores **98.90% accuracy** with **0% fraud caught**).
- **Part (D) — Interactive Streamlit Dashboard (`app/dashboard.py`):** Live **Transaction Simulator** with 1-click attack/legitimate presets, a **Plotly Fraud Probability Gauge**, an exact **SHAP Feature-Importance Waterfall Chart**, an **Autoencoder Reconstruction Error Distribution Plot**, a **Model Leaderboard**, and a **Live Stream Replay** tab.
- **Write-Up (`writeup/Fraud_Detection_Report.docx` & `.md`):** Complete academic report (~1,260 words — strictly under the 2,000-word maximum) with embedded publication figures and benchmark tables.

---

## Quick Start (3 Commands)

> **Note on Windows Python Launcher:** On this machine, Python 3.12 is invoked via `py -3.12`.

### 1. Launch the Interactive Streamlit Dashboard (Part D)
All models, scalers, SHAP background samples, notebooks, and figures are **already trained and saved** in `models/` and `reports/figures/`. You can launch the dashboard immediately:
```powershell
py -3.12 -m streamlit run app/dashboard.py
```

### 2. Open the Executed Presentation Notebooks (Parts A, B, C)
```powershell
py -3.12 -m jupyter notebook notebooks/
```
- `notebooks/01_EDA_PaySim_Mining.ipynb` — Part (B) Data Mining & Class Presentation Visuals
- `notebooks/02_Classical_ML_Baselines.ipynb` — Part (C) Classical ML Baselines, SMOTE & Accuracy Paradox
- `notebooks/03_Deep_Learning_AE_MLP.ipynb` — Parts (A & C) Deep Autoencoder, Entity-Embedding MLP & Full Benchmark

### 3. Re-Run the Entire Pipeline from Scratch (Optional)
If you download the full 470 MB Kaggle PaySim CSV from **https://www.kaggle.com/datasets/ealaxi/paysim1** and place `PS_20174392719_1491204439457_log.csv` inside `data/raw/` (or use the built-in calibrated PaySim dataset in `data/raw/paysim_calibrated_sample.parquet`), run:
```powershell
py -3.12 src/data_prep.py
py -3.12 src/train_ml.py
py -3.12 src/train_dl.py
py -3.12 src/evaluate.py
py -3.12 src/build_notebooks.py
py -3.12 src/generate_report.py
```

---

## Held-Out Chronological Test Set Benchmark Summary

| Model | Family | Accuracy (%) | PR-AUC | F1-Score | Precision | Recall | TP | FP | FN | TN | Latency (ms/tx) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Naive Majority Baseline (All Legit) | Baseline | 98.90% | 0.0110 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 179 | 16,160 | 0.0010 |
| Static Rule Engine (`isFlaggedFraud`) | Rule-Based | 98.90% | 0.0110 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 179 | 16,160 | 0.0050 |
| Isolation Forest | Unsupervised ML | 98.98% | 0.6878 | 0.5917 | 0.5261 | 0.6760 | 121 | 109 | 58 | 16,051 | 0.0252 |
| Deep Autoencoder (Unsupervised) | Deep Learning (Unsupervised) | 99.68% | 0.7830 | 0.8301 | 1.0000 | 0.7095 | 127 | 0 | 52 | 16,160 | 0.0029 |
| Logistic Regression (`class_weight='balanced'`) | Classical ML | 99.77% | 0.9289 | 0.8812 | 1.0000 | 0.7877 | 141 | 0 | 38 | 16,160 | 0.0002 |
| XGBoost + SMOTE | Ensemble ML | 99.78% | 0.9398 | 0.8882 | 1.0000 | 0.7989 | 143 | 0 | 36 | 16,160 | 0.0018 |
| **Entity-Embedding MLP (Deep Learning)** | **Deep Learning (Supervised)** | **99.78%** | **0.9437** | **0.8953** | **0.9333** | **0.8603** | **154** | **11** | **25** | **16,149** | **0.0044** |

