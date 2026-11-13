"""Generate and execute the 3 presentation-ready Jupyter Notebooks in `notebooks/`:
- `01_EDA_PaySim_Mining.ipynb` (Part B: Data Mining & Class Presentation)
- `02_Classical_ML_Baselines.ipynb` (Part C: Logistic Regression, Isolation Forest, XGBoost + SMOTE, Accuracy Paradox)
- `03_Deep_Learning_AE_MLP.ipynb` (Part A & C: Deep Autoencoder, Entity-Embedding MLP, and Full Benchmark Comparison)
"""
from __future__ import annotations

from pathlib import Path
import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"


def build_nb01() -> nbformat.NotebookNode:
    nb = new_notebook()
    nb.cells = [
        new_markdown_cell(
            "# 01 — PaySim Mobile Money Data Mining & Exploratory Analysis (Part B)\n\n"
            "**Course:** DSA 4050 · **Topic:** Real-Time Mobile Money & Credit Card Fraud Detection\n\n"
            "### Presentation Objectives\n"
            "1. Inspect the PaySim transaction log structure (`step`, `type`, `amount`, origin/destination balances, `isFraud`, `isFlaggedFraud`).\n"
            "2. Quantify the **extreme class imbalance** (< 0.5% overall fraud) and demonstrate why naive rule engines (`isFlaggedFraud`) fail.\n"
            "3. Prove domain-level structural properties: **100% of fraud occurs in `TRANSFER` and `CASH_OUT`**.\n"
            "4. Uncover behavioral attack signatures: **overnight fraud spikes (00:00–05:00)** and **100% origin-account emptying**."
        ),
        new_code_cell(
            "import json\n"
            "from pathlib import Path\n"
            "import sys\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "from IPython.display import Image, display\n\n"
            "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
            "sys.path.insert(0, str(ROOT))\n"
            "from src.data_prep import load_raw_or_sample\n\n"
            "df, source_desc = load_raw_or_sample()\n"
            "meta = json.loads((ROOT / 'data' / 'processed' / 'metadata.json').read_text())\n"
            "print('Dataset loaded:', source_desc)\n"
            "print(f\"Total rows: {len(df):,} | Fraud cases: {int(df['isFraud'].sum()):,} ({100*df['isFraud'].mean():.3f}%)\")\n"
            "df.head(8)"
        ),
        new_markdown_cell(
            "## 1. Fraud Distribution Across Transaction Types\n"
            "Every mobile money transaction falls into one of five channels (`CASH_OUT`, `PAYMENT`, `CASH_IN`, `TRANSFER`, `DEBIT`). "
            "Mining the table below reveals that **fraud is strictly confined to `TRANSFER` and `CASH_OUT`**—where attackers first transfer stolen funds to a mule account and immediately cash out."
        ),
        new_code_cell(
            "type_df = pd.DataFrame(meta['type_summary']).sort_values('total_tx', ascending=False)\n"
            "display(type_df.style.format({'total_tx': '{:,}', 'fraud_tx': '{:,}', 'mean_amount': '${:,.2f}', 'median_amount': '${:,.2f}', 'fraud_rate_pct': '{:.3f}%'}))\n"
            "display(Image(filename=str(ROOT / 'reports' / 'figures' / 'fig1_fraud_by_type.png')))"
        ),
        new_markdown_cell(
            "## 2. Failure of the Legacy Rule-Based Engine (`isFlaggedFraud`)\n"
            "PaySim includes a built-in static rule column `isFlaggedFraud` that triggers when a single `TRANSFER` exceeds $200,000 under specific zero-balance conditions. Let's inspect how many actual frauds it catches:"
        ),
        new_code_cell(
            "ct = pd.crosstab(df['isFraud'], df['isFlaggedFraud'], rownames=['Actual isFraud'], colnames=['Rule isFlaggedFraud'])\n"
            "recall_rule = 100.0 * meta['flagged_tp'] / max(meta['total_fraud'], 1)\n"
            "print(f'Rule Engine Recall: {recall_rule:.2f}% ({meta[\"flagged_tp\"]} caught out of {meta[\"total_fraud\"]} frauds)')\n"
            "ct"
        ),
        new_markdown_cell(
            "## 3. Nocturnal Fraud Velocity & Account-Draining Signatures\n"
            "While legitimate mobile money traffic drops sharply between **00:00 and 05:00**, automated account-takeover scripts continue unabated—causing the hourly fraud rate to spike overnight. Furthermore, fraudulent transactions disproportionately empty **100% of the victim's origin balance (`amount / oldbalanceOrg ≈ 1.0`)**."
        ),
        new_code_cell(
            "display(Image(filename=str(ROOT / 'reports' / 'figures' / 'fig2_hourly_fraud_pattern.png')))\n"
            "display(Image(filename=str(ROOT / 'reports' / 'figures' / 'fig3_amount_and_drain.png')))"
        ),
    ]
    return nb


def build_nb02() -> nbformat.NotebookNode:
    nb = new_notebook()
    nb.cells = [
        new_markdown_cell(
            "# 02 — Classical Machine Learning Baselines & The Accuracy Paradox (Part C)\n\n"
            "### Models Evaluated\n"
            "1. **Naive Majority Baseline** (predicts `0` / Legitimate for every transaction — demonstrates the **Accuracy Paradox**)\n"
            "2. **Static Rule Engine** (`isFlaggedFraud`)\n"
            "3. **Logistic Regression** (`class_weight='balanced'`)\n"
            "4. **Isolation Forest** (unsupervised tree-based anomaly detector fit strictly on legitimate training rows)\n"
            "5. **XGBoost + SMOTE** (Synthetic Minority Over-sampling applied **strictly to the training split** to avoid data leakage)"
        ),
        new_code_cell(
            "import json\n"
            "from pathlib import Path\n"
            "import pandas as pd\n"
            "import joblib\n\n"
            "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
            "ml_metrics = json.loads((ROOT / 'models' / 'ml_metrics.json').read_text())\n"
            "print('SMOTE Training Split Before:', ml_metrics['smote_before'])\n"
            "print('SMOTE Training Split After :', ml_metrics['smote_after'])\n\n"
            "df_ml = pd.DataFrame(ml_metrics['ml_results'])[\n"
            "    ['model', 'family', 'accuracy', 'pr_auc', 'f1', 'precision', 'recall', 'tp', 'fp', 'fn', 'tn', 'latency_ms']\n"
            "]\n"
            "df_ml"
        ),
        new_markdown_cell(
            "## Why Raw Accuracy is Misleading on Imbalanced Streams\n"
            "- Notice that the **Naive Majority Baseline** achieves **98.90% accuracy** on the high-risk test stream (and **>99.5% accuracy** across all transaction types) while catching **0 frauds (`TP = 0`, `F1 = 0.0000`, `PR-AUC = 0.0110`)**.\n"
            "- By contrast, **XGBoost + SMOTE** achieves **PR-AUC = 0.9398** and **F1 = 0.8882**, catching nearly 80% of all fraudulent transactions with zero false alarms at its validation-optimal F1 threshold."
        ),
        new_code_cell(
            "ablation = pd.DataFrame([ml_metrics['ablation_no_balance_error']])[\n"
            "    ['model', 'accuracy', 'pr_auc', 'f1', 'precision', 'recall']\n"
            "]\n"
            "print('Ablation Study (Removing errorBalanceOrig & errorBalanceDest):')\n"
            "ablation"
        ),
    ]
    return nb


def build_nb03() -> nbformat.NotebookNode:
    nb = new_notebook()
    nb.cells = [
        new_markdown_cell(
            "# 03 — Deep Learning Models (Autoencoder + Entity-Embedding MLP) & Full Benchmark (Parts A & C)\n\n"
            "### Why Deep Learning for Real-Time Financial Fraud Streams?\n"
            "1. **Unsupervised Deep Autoencoder (`15 → 32 → 16 → 6 → 16 → 32 → 15`)**: Trained *exclusively on legitimate transactions* to compress normal user behavior into a 6-dimensional latent manifold. Novel zero-day fraud patterns violate learned balance-flow conservation and produce high reconstruction MSE.\n"
            "2. **Supervised Entity-Embedding MLP**: Maps categorical context (`hour` of day and `transaction type`) into dense learnable metric spaces (`nn.Embedding`) concatenated with continuous features and the Autoencoder's latent representation—flagging subtle multi-dimensional anomalies in **sub-millisecond per-transaction latency**."
        ),
        new_code_cell(
            "import json\n"
            "from pathlib import Path\n"
            "import pandas as pd\n"
            "from IPython.display import Image, display\n\n"
            "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
            "bench = json.loads((ROOT / 'models' / 'benchmark_summary.json').read_text())\n"
            "df_all = pd.DataFrame(bench['benchmark_table'])[\n"
            "    ['model', 'family', 'accuracy', 'pr_auc', 'f1', 'precision', 'recall', 'tp', 'fp', 'fn', 'tn', 'latency_ms']\n"
            "]\n"
            "df_all.style.format({\n"
            "    'accuracy': '{:.4f}', 'pr_auc': '{:.4f}', 'f1': '{:.4f}',\n"
            "    'precision': '{:.4f}', 'recall': '{:.4f}', 'latency_ms': '{:.4f} ms'\n"
            "})"
        ),
        new_markdown_cell(
            "## Precision–Recall Curves, Confusion Matrices & Autoencoder Latent Anomaly Separation"
        ),
        new_code_cell(
            "display(Image(filename=str(ROOT / 'reports' / 'figures' / 'fig4_pr_curves_and_accuracy_paradox.png')))\n"
            "display(Image(filename=str(ROOT / 'reports' / 'figures' / 'fig5_confusion_matrices.png')))\n"
            "display(Image(filename=str(ROOT / 'reports' / 'figures' / 'fig6_autoencoder_reconstruction_error.png')))"
        ),
        new_markdown_cell(
            "## Ablation Comparison: Classical ML vs. Deep Learning Without Ledger-Error Features\n"
            "When the explicit accounting identity columns (`errorBalanceOrig`, `errorBalanceDest`) are withheld, the models must rely on subtle multi-dimensional ratios, nocturnal timing embeddings, and latent interactions:"
        ),
        new_code_cell(
            "pd.DataFrame(bench['ablation_table'])[['model', 'pr_auc', 'f1', 'precision', 'recall']]"
        ),
    ]
    return nb


def main() -> None:
    NB_DIR.mkdir(parents=True, exist_ok=True)
    notebooks = {
        "01_EDA_PaySim_Mining.ipynb": build_nb01(),
        "02_Classical_ML_Baselines.ipynb": build_nb02(),
        "03_Deep_Learning_AE_MLP.ipynb": build_nb03(),
    }
    for name, nb in notebooks.items():
        path = NB_DIR / name
        nbformat.write(nb, path)
        print(f"[build_notebooks] Executing {name} ...")
        client = NotebookClient(nb, timeout=180, kernel_name="python3", resources={"metadata": {"path": str(NB_DIR)}})
        client.execute()
        nbformat.write(nb, path)
        print(f"[build_notebooks] Saved executed notebook: {path}")


if __name__ == "__main__":
    main()

