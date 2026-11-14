"""Generate and execute a single, unified master Jupyter Notebook in `notebooks/`:
- `Fraud_Detection_Complete_Project.ipynb`

Covers the entire DSA 4050 rubric in one seamless presentation & submission file:
- Part (A): Why Deep Learning for Real-Time Financial Fraud Streams (Deep Autoencoder + Entity-Embedding MLP)
- Part (B): PaySim Dataset Mining & Exploratory Data Analysis (Class Imbalance, Channel Confinement, Nocturnal Velocity, Account Draining)
- Feature Engineering & Zero-Leakage Chronological Validation
- Part (C): Classical ML (LR, Isolation Forest, XGBoost + SMOTE) vs. Deep Learning Benchmarking, Precision-Recall Curves, Confusion Matrices, Ablation Study & The Accuracy Paradox
- Part (D): Live Transaction Simulator & Local SHAP / Autoencoder Demo Inside the Notebook
"""
from __future__ import annotations

from pathlib import Path
import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"


def build_master_notebook() -> nbformat.NotebookNode:
    nb = new_notebook()
    nb.cells = [
        new_markdown_cell(
            "# Real-Time Mobile Money & Credit Card Fraud Detection\n"
            "### **DSA 4050 — Complete Packaged Master Notebook (Parts A, B, C & D)**\n\n"
            "---\n"
            "**Rubric Roadmap Covered in This Single Notebook:**\n"
            "1. **Part (A) — Why Deep Learning?** Architectural justification and PyTorch implementation of an **Unsupervised Deep Autoencoder** (`15 → 32 → 16 → 6 → 16 → 32 → 15`) and a **Supervised Entity-Embedding Multi-Layer Perceptron (MLP)**.\n"
            "2. **Part (B) — Dataset Mining & Class Presentation:** Mining the **PaySim Mobile Money** dataset (`250,000` transactions across `743` hourly steps), proving that **100% of fraud occurs in `TRANSFER` and `CASH_OUT`**, exposing the failure of legacy static rules (`isFlaggedFraud`), and visualizing **overnight attack spikes (`00:00–05:00`)** and **100% origin-account emptying**.\n"
            "3. **Feature Engineering & Leak-Free Chronological Split:** 15 domain-engineered accounting and behavioral ratios (`errorBalanceOrig`, `amt_to_bal_ratio`, `orig_emptied`, cyclical `hour_sin`/`hour_cos`) evaluated on a strict time-ordered `70% Train / 15% Validation / 15% Test` split with train-only `SMOTE`.\n"
            "4. **Part (C) — ML vs. DL Benchmarking & The Accuracy Paradox:** Comparing **Logistic Regression**, **Isolation Forest**, and **XGBoost + SMOTE** against the **Deep Autoencoder** and **Entity-Embedding MLP** using **PR-AUC**, **F1-Score**, **Confusion Matrices**, an **Ablation Study**, and an explicit proof of why **Raw Accuracy is misleading on imbalanced data**.\n"
            "5. **Part (D) — Interactive Transaction Simulator Demo:** Scoring live legitimate and fraudulent transactions with millisecond latency, multi-model consensus, Autoencoder reconstruction MSE, and SHAP feature attributions (plus instructions to launch the full Streamlit dashboard in `app/dashboard.py`)."
        ),
        new_markdown_cell(
            "## Part (A): Why Deep Learning for Financial Fraud Streams?\n\n"
            "Financial transaction streams suffer from **extreme class imbalance (< 0.5% fraud)** and **rapidly evolving adversarial tactics** where static threshold rules (`TRANSFER > $200,000`) either miss structured attacks or generate overwhelming false positives.\n\n"
            "We combine two complementary Deep Learning paradigms in PyTorch (`src/train_dl.py`):\n"
            "1. **Unsupervised Deep Autoencoder (`DeepAutoencoder`)**: Trained *strictly on legitimate transactions (`isFraud == 0`)* to compress normal 15-dimensional user behavior and balance-conservation laws into a **6-dimensional latent bottleneck (`15 → 32 → 16 → 6 → 16 → 32 → 15`)**. Because fraudulent or zero-day attacks violate the learned normal manifold, they fail to reconstruct accurately and produce a massive **Reconstruction Mean Squared Error (MSE)**.\n"
            "2. **Supervised Entity-Embedding MLP (`EntityEmbeddingMLP`)**: Maps cyclical and categorical context (`hour` of day `0..23 → 6-D` and `transaction type → 3-D`) into dense learnable metric spaces via `nn.Embedding`, concatenated with the 15 continuous financial features **plus** the Autoencoder's 6-D latent vector and reconstruction error (`128 → 64 → 32 → 1`)."
        ),
        new_code_cell(
            "import json\n"
            "from pathlib import Path\n"
            "import sys\n\n"
            "import torch\n"
            "import joblib\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "from IPython.display import Image, display\n\n"
            "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
            "if str(ROOT) not in sys.path:\n"
            "    sys.path.insert(0, str(ROOT))\n\n"
            "from src.data_prep import load_raw_or_sample\n"
            "from src.features import NUM_FEATURES, LEAKY_FEATURES, engineer, single_transaction, to_model_matrix\n"
            "from src.train_dl import DeepAutoencoder, EntityEmbeddingMLP\n\n"
            "ae_ckpt = torch.load(ROOT / 'models' / 'autoencoder.pt', map_location='cpu', weights_only=False)\n"
            "mlp_ckpt = torch.load(ROOT / 'models' / 'entity_mlp.pt', map_location='cpu', weights_only=False)\n\n"
            "ae_model = DeepAutoencoder(in_dim=ae_ckpt['in_dim'], latent_dim=ae_ckpt['latent_dim'])\n"
            "mlp_model = EntityEmbeddingMLP(num_cont=mlp_ckpt['num_cont'], hour_emb_dim=mlp_ckpt['hour_emb_dim'], type_emb_dim=mlp_ckpt['type_emb_dim'])\n"
            "print('=== 1. Unsupervised Deep Autoencoder Architecture ===')\n"
            "print(ae_model)\n"
            "print('\\n=== 2. Supervised Entity-Embedding MLP Architecture ===')\n"
            "print(mlp_model)"
        ),
        new_markdown_cell(
            "---\n"
            "## Part (B): Mining the PaySim Mobile Money Dataset & Exploratory Analysis\n\n"
            "We mine the **PaySim Mobile Money** dataset (`step`, `type`, `amount`, `nameOrig`, `oldbalanceOrg`, `newbalanceOrig`, `nameDest`, `oldbalanceDest`, `newbalanceDest`, `isFraud`, `isFlaggedFraud`), which models 30 days (`743` hourly steps) of African mobile money logs."
        ),
        new_code_cell(
            "df, source_desc = load_raw_or_sample()\n"
            "meta = json.loads((ROOT / 'data' / 'processed' / 'metadata.json').read_text())\n"
            "print('Active Dataset:', source_desc)\n"
            "print(f\"Total Transactions: {len(df):,} | Fraud Cases: {int(df['isFraud'].sum()):,} ({100*df['isFraud'].mean():.3f}%)\")\n"
            "df.head(8)"
        ),
        new_markdown_cell(
            "### B.1 Channel Confinement: 100% of Fraud Occurs in `TRANSFER` & `CASH_OUT`\n"
            "Inspecting fraud incidence across all five transaction types reveals a critical domain property: **fraud occurs exclusively in `TRANSFER` and `CASH_OUT`** (where attackers transfer stolen funds to a mule account and immediately cash out). Zero fraud occurs in `PAYMENT`, `CASH_IN`, or `DEBIT`."
        ),
        new_code_cell(
            "type_df = pd.DataFrame(meta['type_summary']).sort_values('total_tx', ascending=False)\n"
            "display(type_df.style.format({\n"
            "    'total_tx': '{:,}', 'fraud_tx': '{:,}',\n"
            "    'mean_amount': '${:,.2f}', 'median_amount': '${:,.2f}', 'fraud_rate_pct': '{:.3f}%'\n"
            "}))\n"
            "display(Image(filename=str(ROOT / 'reports' / 'figures' / 'fig1_fraud_by_type.png')))"
        ),
        new_markdown_cell(
            "### B.2 Failure of Legacy Static Rule Engine (`isFlaggedFraud`)\n"
            "PaySim's built-in static threshold rule (`isFlaggedFraud`, which triggers only on `TRANSFER > $200,000` under static zero-balance conditions) catches **only 1.23% of actual fraud**, proving why simple rule engines fail:"
        ),
        new_code_cell(
            "ct = pd.crosstab(df['isFraud'], df['isFlaggedFraud'], rownames=['Actual isFraud'], colnames=['Legacy Rule (isFlaggedFraud)'])\n"
            "recall_rule = 100.0 * meta['flagged_tp'] / max(meta['total_fraud'], 1)\n"
            "print(f\"Legacy Rule Engine Recall: {recall_rule:.2f}% ({meta['flagged_tp']} caught out of {meta['total_fraud']:,} actual frauds)\")\n"
            "display(ct)"
        ),
        new_markdown_cell(
            "### B.3 Behavioral Attack Signatures: Nocturnal Velocity (`00:00–05:00`) & Account Draining\n"
            "1. **Nocturnal Attack Spike:** Legitimate customer transactions drop by over 80% overnight (`hours 00:00–05:00`), whereas automated account-takeover scripts run 24/7—causing the hourly fraud percentage to spike sharply overnight.\n"
            "2. **100% Origin Account Emptying:** Fraudulent transfers cluster tightly around `Amount / OldBalanceOrig = 1.0`, draining the victim's entire balance in a single transaction."
        ),
        new_code_cell(
            "display(Image(filename=str(ROOT / 'reports' / 'figures' / 'fig2_hourly_fraud_pattern.png')))\n"
            "display(Image(filename=str(ROOT / 'reports' / 'figures' / 'fig3_amount_and_drain.png')))"
        ),
        new_markdown_cell(
            "---\n"
            "## Feature Engineering & Leak-Free Chronological Split\n\n"
            "To prevent data leakage and test forward temporal generalization:\n"
            "- We filter the modeling pipeline to the high-risk `TRANSFER` and `CASH_OUT` channels (`108,922` transactions, `100%` of all frauds).\n"
            "- We engineer **15 domain features** in `src/features.py` (ledger conservation errors `errorBalanceOrig` and `errorBalanceDest`, balance deltas, `amt_to_bal_ratio`, `orig_emptied`, `dest_zero_before`, and cyclical `hour_sin`/`hour_cos`).\n"
            "- We perform a **chronological split ordered by `step`**: **70% Train (`76,245` rows)**, **15% Validation (`16,338` rows)** for F1-optimal threshold tuning, and **15% Held-Out Test (`16,339` rows)** retaining its natural class imbalance.\n"
            "- `StandardScaler` and **SMOTE (`sampling_strategy=0.15`)** are fit **exclusively on the training split**."
        ),
        new_code_cell(
            "ml_metrics = json.loads((ROOT / 'models' / 'ml_metrics.json').read_text())\n"
            "print(f\"Chronological Split Sizes -> Train: {meta['train_rows']:,} (Fraud={meta['train_fraud']:,}) | \"\n"
            "      f\"Val: {meta['val_rows']:,} (Fraud={meta['val_fraud']:,}) | \"\n"
            "      f\"Test: {meta['test_rows']:,} (Fraud={meta['test_fraud']:,})\")\n"
            "print('SMOTE on Training Split Only -> Before:', ml_metrics['smote_before'], '-> After:', ml_metrics['smote_after'])\n"
            "print('15 Engineered Features:', NUM_FEATURES)"
        ),
        new_markdown_cell(
            "---\n"
            "## Part (C): Classical ML vs. Deep Learning Benchmarking & The Accuracy Paradox\n\n"
            "### C.1 Why Raw Accuracy is Misleading on Imbalanced Streams (Bonus Credit Analysis)\n"
            "In our held-out test stream, `98.90%` of high-risk transactions are legitimate (`>99.5%` across all five channels). Consequently:\n"
            "- The **Naive Majority Baseline** (predicting *Legitimate* for every single transaction) achieves **98.90% Raw Accuracy** while catching **0 out of 179 frauds (`TP = 0`, `Recall = 0.00%`, `F1 = 0.0000`, `PR-AUC = 0.0110`)**.\n"
            "- **Precision–Recall AUC (PR-AUC)** and **F1-Score** ignore the `16,160` true negatives and evaluate performance strictly on the minority fraud class."
        ),
        new_code_cell(
            "bench = json.loads((ROOT / 'models' / 'benchmark_summary.json').read_text())\n"
            "df_all = pd.DataFrame(bench['benchmark_table'])[\n"
            "    ['model', 'family', 'accuracy', 'pr_auc', 'f1', 'precision', 'recall', 'tp', 'fp', 'fn', 'tn', 'latency_ms']\n"
            "]\n"
            "display(df_all.style.format({\n"
            "    'accuracy': '{:.2%}', 'pr_auc': '{:.4f}', 'f1': '{:.4f}',\n"
            "    'precision': '{:.4f}', 'recall': '{:.4f}', 'latency_ms': '{:.4f} ms'\n"
            "}))"
        ),
        new_markdown_cell(
            "### C.2 Precision–Recall Curves, Confusion Matrices & Autoencoder Latent Separation\n"
            "- **Supervised Champion:** **Entity-Embedding MLP** achieves the highest overall **PR-AUC (`0.9437`)**, **F1-Score (`0.8953`)**, and **Fraud Recall (`86.03%`, catching `154 / 179` frauds)** in **0.0044 ms/tx**, closely followed by **XGBoost + SMOTE** (`PR-AUC = 0.9398`, `F1 = 0.8882`, `0` false positives).\n"
            "- **Unsupervised Champion:** Without seeing a single fraud label during training, the **Deep Autoencoder** achieves **F1 = `0.8301`** (`PR-AUC = 0.7830`, `0` false positives), dramatically outperforming classical **Isolation Forest** (`F1 = 0.5917`, `109` false positives)."
        ),
        new_code_cell(
            "display(Image(filename=str(ROOT / 'reports' / 'figures' / 'fig4_pr_curves_and_accuracy_paradox.png')))\n"
            "display(Image(filename=str(ROOT / 'reports' / 'figures' / 'fig5_confusion_matrices.png')))\n"
            "display(Image(filename=str(ROOT / 'reports' / 'figures' / 'fig6_autoencoder_reconstruction_error.png')))"
        ),
        new_markdown_cell(
            "### C.3 Ablation Study: What If Attackers Camouflage Ledger Errors?\n"
            "Even when the two accounting discrepancy features (`errorBalanceOrig` and `errorBalanceDest`) are completely removed, both **XGBoost + SMOTE** and the **Entity-Embedding MLP** retain **>88% F1** and **~0.92 PR-AUC** by leveraging nocturnal timing embeddings and balance-depletion ratios:"
        ),
        new_code_cell(
            "df_ablation = pd.DataFrame(bench['ablation_table'])[['model', 'accuracy', 'pr_auc', 'f1', 'precision', 'recall']]\n"
            "display(df_ablation.style.format({'accuracy': '{:.2%}', 'pr_auc': '{:.4f}', 'f1': '{:.4f}', 'precision': '{:.4f}', 'recall': '{:.4f}'}))"
        ),
        new_markdown_cell(
            "---\n"
            "## Part (D): Live Transaction Simulator & Explainability Demo\n\n"
            "Our full interactive Streamlit dashboard is located in `app/dashboard.py` (launch with `py -3.12 -m streamlit run app/dashboard.py`). Below, we invoke the exact same scoring and SHAP explainability engine directly inside this notebook on three preset scenarios:\n"
            "1. `🚨 Classic Account-Draining Transfer (02:00 AM)`\n"
            "2. `⚠️ Structuring / Near-Drain Transfer (94% of Balance)`\n"
            "3. `✅ Normal Daytime Cash-Out (Healthy Remaining Balance)`"
        ),
        new_code_cell(
            "from app.dashboard import load_artifacts, score_single_transaction, PRESETS\n\n"
            "art = load_artifacts()\n"
            "demo_presets = [\n"
            "    '🚨 Classic Account-Draining Transfer (02:00 AM)',\n"
            "    '⚠️ Structuring / Near-Drain Transfer (94% of Balance)',\n"
            "    '✅ Normal Daytime Cash-Out (Healthy Remaining Balance)',\n"
            "]\n\n"
            "sim_rows = []\n"
            "for name in demo_presets:\n"
            "    p = PRESETS[name]\n"
            "    raw_tx = single_transaction(**p)\n"
            "    sc = score_single_transaction(art, raw_tx, 'Hybrid Deep Learning + XGBoost Ensemble (Recommended)')\n"
            "    top_shap_idx = int(np.argmax(np.abs(sc['shap_vals'])))\n"
            "    sim_rows.append({\n"
            "        'Scenario': name,\n"
            "        'Type': p['tx_type'],\n"
            "        'Hour': f\"{p['hour']:02d}:00\",\n"
            "        'Amount ($)': f\"${p['amount']:,.2f}\",\n"
            "        'Hybrid Fraud Prob': f\"{sc['prob']*100:.2f}%\",\n"
            "        'MLP Prob': f\"{sc['all_probs']['Entity-Embedding MLP (Deep Learning)']*100:.2f}%\",\n"
            "        'XGBoost Prob': f\"{sc['all_probs']['XGBoost + SMOTE']*100:.2f}%\",\n"
            "        'Autoencoder MSE': f\"{sc['recon_mse']:.3f} (Thr={sc['ae_thr_mse']:.2f})\",\n"
            "        'Top SHAP Driver': f\"{NUM_FEATURES[top_shap_idx]} ({sc['shap_vals'][top_shap_idx]:+.2f})\",\n"
            "        'Decision': '🚨 BLOCK' if sc['prob'] >= 0.65 else ('⚠️ REVIEW' if sc['prob'] >= 0.30 else '✅ APPROVE'),\n"
            "    })\n\n"
            "display(pd.DataFrame(sim_rows))"
        ),
    ]
    return nb


def main() -> None:
    NB_DIR.mkdir(parents=True, exist_ok=True)
    # Remove old split notebooks so there is a single clean notebook for submission
    for old in [
        "01_EDA_PaySim_Mining.ipynb",
        "02_Classical_ML_Baselines.ipynb",
        "03_Deep_Learning_AE_MLP.ipynb",
    ]:
        p = NB_DIR / old
        if p.exists():
            p.unlink()
            print(f"[build_notebooks] Removed split notebook: {old}")

    nb = build_master_notebook()
    out_path = NB_DIR / "Fraud_Detection_Complete_Project.ipynb"
    nbformat.write(nb, out_path)
    print(f"[build_notebooks] Executing unified master notebook: {out_path.name} ...")
    client = NotebookClient(nb, timeout=180, kernel_name="python3", resources={"metadata": {"path": str(NB_DIR)}})
    client.execute()
    nbformat.write(nb, out_path)
    print(f"[build_notebooks] Saved executed master notebook: {out_path}")


if __name__ == "__main__":
    main()
