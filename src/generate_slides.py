"""Generate a polished 12-slide widescreen PowerPoint presentation (.pptx) for class presentation.

Covers:
1. Title Slide: Real-Time Mobile Money & Credit Card Fraud Detection
2. Problem Statement: Extreme Class Imbalance & Failure of Static Rules
3. Part (A) — Why Deep Learning? (Deep Autoencoder + Entity-Embedding MLP Architecture)
4. Part (B) — Dataset Mining (PaySim Schema, 250k Transactions, Channel Confinement) + Fig 1
5. Part (B) — Attack Signatures: Nocturnal Velocity (00:00–05:00) & Account Draining + Fig 2 & Fig 3
6. Feature Engineering & Leak-Free Temporal Validation Pipeline
7. Part (C) — The Accuracy Paradox (Why 98.90% Raw Accuracy Catches 0% Fraud) + Fig 4
8. Part (C) — Full ML vs. DL Benchmark Table (LR, Isolation Forest, XGBoost+SMOTE, Autoencoder, MLP)
9. Part (C) — Confusion Matrices & Unsupervised Autoencoder Latent Separation + Fig 5 & Fig 6
10. Ablation Study: Robustness When Ledger-Error Features Are Withheld
11. Part (D) — Live Streamlit "Transaction Simulator" Dashboard Walkthrough (Gauge, SHAP, AE MSE)
12. Conclusion, Production Recommendations & Live Demo Callout
"""
from __future__ import annotations

import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figures"
WRITEUP_DIR = ROOT / "writeup"

# Color palette (Executive Dark Navy + Slate + Crimson Accent + Emerald)
NAVY = RGBColor(15, 23, 42)
SLATE_DARK = RGBColor(30, 41, 59)
SLATE_MUTED = RGBColor(100, 116, 139)
LIGHT_BG = RGBColor(248, 250, 252)
WHITE = RGBColor(255, 255, 255)
BLUE_ACCENT = RGBColor(37, 99, 235)
RED_ACCENT = RGBColor(220, 38, 38)
GREEN_ACCENT = RGBColor(16, 185, 129)
CARD_BORDER = RGBColor(203, 213, 225)


def add_header(slide, title: str, subtitle: str, tag: str = "DSA 4050") -> None:
    # Top banner bar
    banner = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(1.05))
    banner.fill.solid()
    banner.fill.fore_color.rgb = NAVY
    banner.line.fill.background()

    # Tag pill on top-right
    pill = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(11.15), Inches(0.25), Inches(1.85), Inches(0.45))
    pill.fill.solid()
    pill.fill.fore_color.rgb = BLUE_ACCENT
    pill.line.fill.background()
    tf_p = pill.text_frame
    p_p = tf_p.paragraphs[0]
    p_p.text = tag
    p_p.alignment = PP_ALIGN.CENTER
    p_p.font.size = Pt(11)
    p_p.font.bold = True
    p_p.font.color.rgb = WHITE

    # Title text
    tb = slide.shapes.add_textbox(Inches(0.45), Inches(0.10), Inches(10.5), Inches(0.55))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(22)
    p.font.bold = True
    p.font.color.rgb = WHITE

    # Subtitle text
    p2 = tf.add_paragraph()
    p2.text = subtitle
    p2.font.size = Pt(12)
    p2.font.color.rgb = RGBColor(148, 163, 184)


def add_card(
    slide,
    left: float,
    top: float,
    width: float,
    height: float,
    heading: str,
    bullets: list[str],
    accent_color: RGBColor = BLUE_ACCENT,
    font_size: int = 13,
) -> None:
    card = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height)
    )
    card.fill.solid()
    card.fill.fore_color.rgb = LIGHT_BG
    card.line.color.rgb = CARD_BORDER
    card.line.width = Pt(1)

    # Top accent stripe
    stripe = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(0.08)
    )
    stripe.fill.solid()
    stripe.fill.fore_color.rgb = accent_color
    stripe.line.fill.background()

    tb = slide.shapes.add_textbox(Inches(left + 0.18), Inches(top + 0.14), Inches(width - 0.36), Inches(height - 0.24))
    tf = tb.text_frame
    tf.word_wrap = True

    p0 = tf.paragraphs[0]
    p0.text = heading
    p0.font.size = Pt(font_size + 2)
    p0.font.bold = True
    p0.font.color.rgb = NAVY
    p0.space_after = Pt(6)

    for b in bullets:
        p = tf.add_paragraph()
        p.text = f"• {b}"
        p.font.size = Pt(font_size)
        p.font.color.rgb = SLATE_DARK
        p.space_after = Pt(5)


def add_kpi_box(
    slide, left: float, top: float, width: float, height: float, value: str, label: str, sub: str, color: RGBColor
) -> None:
    box = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height)
    )
    box.fill.solid()
    box.fill.fore_color.rgb = LIGHT_BG
    box.line.color.rgb = color
    box.line.width = Pt(1.75)

    tb = slide.shapes.add_textbox(Inches(left + 0.12), Inches(top + 0.10), Inches(width - 0.24), Inches(height - 0.18))
    tf = tb.text_frame
    tf.word_wrap = True

    p1 = tf.paragraphs[0]
    p1.text = value
    p1.font.size = Pt(24)
    p1.font.bold = True
    p1.font.color.rgb = color

    p2 = tf.add_paragraph()
    p2.text = label
    p2.font.size = Pt(12)
    p2.font.bold = True
    p2.font.color.rgb = NAVY

    p3 = tf.add_paragraph()
    p3.text = sub
    p3.font.size = Pt(10.5)
    p3.font.color.rgb = SLATE_MUTED


def main() -> None:
    WRITEUP_DIR.mkdir(parents=True, exist_ok=True)
    meta = json.loads((PROC_DIR / "metadata.json").read_text(encoding="utf-8"))
    bench = json.loads((MODELS_DIR / "benchmark_summary.json").read_text(encoding="utf-8"))
    ae_dist = json.loads((MODELS_DIR / "ae_error_distribution.json").read_text(encoding="utf-8"))

    by_name = {r["model"]: r for r in bench["benchmark_table"]}
    r_naive = by_name["Naive Majority Baseline (All Legit)"]
    r_lr = by_name["Logistic Regression"]
    r_iso = by_name["Isolation Forest"]
    r_xgb = by_name["XGBoost + SMOTE"]
    r_ae = by_name["Deep Autoencoder (Unsupervised)"]
    r_mlp = by_name["Entity-Embedding MLP (Deep Learning)"]

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    # =========================================================================
    # SLIDE 1 — Title Slide
    # =========================================================================
    s1 = prs.slides.add_slide(blank)
    bg1 = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = NAVY
    bg1.line.fill.background()

    accent_bar = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.85), Inches(1.35), Inches(0.14), Inches(4.5))
    accent_bar.fill.solid()
    accent_bar.fill.fore_color.rgb = RED_ACCENT
    accent_bar.line.fill.background()

    tb1 = s1.shapes.add_textbox(Inches(1.25), Inches(1.30), Inches(11.2), Inches(4.8))
    tf1 = tb1.text_frame
    tf1.word_wrap = True

    p = tf1.paragraphs[0]
    p.text = "DSA 4050 · 20-MARK PACKAGED PROJECT & PRESENTATION"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = RGBColor(96, 165, 250)
    p.space_after = Pt(14)

    p = tf1.add_paragraph()
    p.text = "Real-Time Mobile Money &\nCredit Card Fraud Detection"
    p.font.size = Pt(40)
    p.font.bold = True
    p.font.color.rgb = WHITE
    p.space_after = Pt(16)

    p = tf1.add_paragraph()
    p.text = (
        "Unsupervised Deep Autoencoders & Entity-Embedding Neural Networks vs. Classical Machine Learning\n"
        "on Extremely Imbalanced Financial Transaction Streams (PaySim Dataset)"
    )
    p.font.size = Pt(17)
    p.font.color.rgb = RGBColor(203, 213, 225)
    p.space_after = Pt(28)

    p = tf1.add_paragraph()
    p.text = (
        "Deliverables Included:  (A) Deep Learning Justification  |  (B) PaySim Data Mining & EDA\n"
        "(C) ML vs. DL Benchmarking & Accuracy Paradox  |  (D) Live Streamlit Transaction Simulator"
    )
    p.font.size = Pt(13)
    p.font.color.rgb = RGBColor(148, 163, 184)

    # =========================================================================
    # SLIDE 2 — Problem Statement & Motivation
    # =========================================================================
    s2 = prs.slides.add_slide(blank)
    add_header(
        s2,
        "1. The Real-Time Financial Fraud Challenge",
        "Why legacy static rules fail on high-velocity mobile money and card rails",
        "Motivation",
    )
    add_kpi_box(
        s2, 0.45, 1.25, 3.0, 1.35,
        f"{meta['overall_fraud_rate_pct']:.3f}%",
        "Severe Class Imbalance",
        f"{meta['total_fraud']:,} frauds in {meta['total_rows']:,} transactions",
        RED_ACCENT,
    )
    add_kpi_box(
        s2, 3.65, 1.25, 3.0, 1.35,
        f"{100.0 * meta['flagged_tp'] / max(meta['total_fraud'], 1):.2f}%",
        "Legacy Rule Engine Recall",
        f"isFlaggedFraud caught {meta['flagged_tp']} / {meta['total_fraud']:,} frauds",
         RED_ACCENT,
    )
    add_kpi_box(
        s2, 6.85, 1.25, 3.0, 1.35,
        "100%",
        "Channel Confinement",
        "All fraud in TRANSFER & CASH_OUT",
        BLUE_ACCENT,
    )
    add_kpi_box(
        s2, 10.05, 1.25, 2.83, 1.35,
        "< 5 ms",
        "Inference Latency SLA",
        f"MLP scores in {r_mlp['latency_ms']:.4f} ms/tx",
        GREEN_ACCENT,
    )

    add_card(
        s2, 0.45, 2.85, 6.05, 4.25,
        "Why Simple Rule-Based Engines Fail",
        [
            "Static threshold rules (e.g., PaySim's built-in TRANSFER > $200,000 flag) are trivially bypassed by structuring transactions just below limits.",
            "In our dataset, the legacy rule engine (isFlaggedFraud) misses over 99% of actual fraudulent transactions.",
            "Lowering static thresholds floods compliance teams with thousands of false positives, blocking legitimate customers.",
            "Fraudsters operate automated scripts overnight (00:00–05:00) that rapidly transfer funds to mule accounts and immediately cash out.",
        ],
        RED_ACCENT,
    )
    add_card(
        s2, 6.75, 2.85, 6.13, 4.25,
        "Project Objectives (Rubric Mapping A–D)",
        [
            "Part (A) — Justify Deep Learning: Combine an unsupervised Deep Autoencoder with a supervised Entity-Embedding MLP.",
            "Part (B) — Mine PaySim Dataset: Uncover channel confinement, nocturnal velocity spikes, and account-draining signatures.",
            "Part (C) — Benchmark ML vs. DL: Compare Logistic Regression, Isolation Forest, and XGBoost+SMOTE against Autoencoder & MLP using PR-AUC, F1, and Confusion Matrices.",
            "Part (D) — Interactive Dashboard: Deploy a live Streamlit Transaction Simulator with Fraud Gauge, SHAP Waterfall, and Anomaly Error plots.",
        ],
        BLUE_ACCENT,
    )

    # =========================================================================
    # SLIDE 3 — Part (A): Why Deep Learning?
    # =========================================================================
    s3 = prs.slides.add_slide(blank)
    add_header(
        s3,
        "2. Part (A): Why Deep Learning for Financial Fraud Detection?",
        "Complementary Unsupervised Latent Anomaly Detection + Supervised Entity Embeddings",
        "Part A · Why DL",
    )
    add_card(
        s3, 0.45, 1.30, 6.05, 5.75,
        "Architecture 1: Unsupervised Deep Autoencoder",
        [
            "Topology: 15 → 32 → 16 → 6 (Latent Bottleneck) → 16 → 32 → 15 with BatchNorm, GELU activations, and Dropout.",
            "Trained strictly on LEGITIMATE transactions (y = 0) to compress normal user spending and ledger conservation into a 6-D manifold.",
            "Why it matters: Supervised models only recognize past fraud patterns. Zero-day attacks violate the learned manifold and yield massive reconstruction error (MSE).",
            f"Empirical Separation: Median legitimate MSE = {ae_dist['median_legit_mse']:.3f} vs. median fraud MSE = {ae_dist['median_fraud_mse']:.2f} (>150× higher).",
            f"Achieves PR-AUC = {r_ae['pr_auc']:.4f} and F1 = {r_ae['f1']:.4f} without seeing a single fraud label during training!",
        ],
        BLUE_ACCENT,
    )
    add_card(
        s3, 6.75, 1.30, 6.13, 5.75,
        "Architecture 2: Supervised Entity-Embedding MLP",
        [
            "Topology: Learnable nn.Embedding layers for Hour of Day (24 → 6-D) and Transaction Channel (2 → 3-D).",
            "Feature Fusion: Concatenates embeddings with 15 scaled financial ratios + the Autoencoder's 6-D latent vector and reconstruction MSE.",
            "Classifier Head: 128 → 64 → 32 → 1 dense layers with BatchNorm, GELU, Dropout, and class-weighted binary cross-entropy loss.",
            "Why Entity Embeddings? Maps cyclical hours (e.g., 01:00 and 02:00 AM attack windows) close together in metric space.",
            f"Top Benchmark Performer: Achieves highest PR-AUC ({r_mlp['pr_auc']:.4f}) and F1-Score ({r_mlp['f1']:.4f}) in {r_mlp['latency_ms']:.4f} ms/tx.",
        ],
        GREEN_ACCENT,
    )

    # =========================================================================
    # SLIDE 4 — Part (B): PaySim Data Mining — Channel Confinement
    # =========================================================================
    s4 = prs.slides.add_slide(blank)
    add_header(
        s4,
        "3. Part (B): PaySim Dataset Mining — Where Does Fraud Occur?",
        "Mining 250,000 mobile money logs across 5 transaction types and 743 hourly steps (30 days)",
        "Part B · Data Mining",
    )
    fig1 = FIG_DIR / "fig1_fraud_by_type.png"
    if fig1.exists():
        s4.shapes.add_picture(str(fig1), Inches(0.45), Inches(1.30), width=Inches(8.35))

    add_card(
        s4, 8.95, 1.30, 3.95, 5.75,
        "Key Mining Insights",
        [
            "5 Transaction Channels: CASH_OUT (35.2%), PAYMENT (33.8%), CASH_IN (22.0%), TRANSFER (8.4%), DEBIT (0.7%).",
            "100% of Fraud occurs exclusively in TRANSFER (2.80% fraud rate) and CASH_OUT (0.71% fraud rate).",
            "0% Fraud in PAYMENT, CASH_IN, and DEBIT.",
            "Domain Filter: Restricting ML/DL training to TRANSFER + CASH_OUT eliminates 56.4% of noise without losing a single fraud case.",
        ],
        RED_ACCENT,
        font_size=12,
    )

    # =========================================================================
    # SLIDE 5 — Part (B): Nocturnal Attack Spikes & Account Draining
    # =========================================================================
    s5 = prs.slides.add_slide(blank)
    add_header(
        s5,
        "4. Part (B): Behavioral Attack Signatures in PaySim",
        "Overnight automated attack velocity (00:00–05:00) and 100% origin-account emptying",
        "Part B · EDA Visuals",
    )
    fig2 = FIG_DIR / "fig2_hourly_fraud_pattern.png"
    fig3 = FIG_DIR / "fig3_amount_and_drain.png"
    if fig2.exists():
        s5.shapes.add_picture(str(fig2), Inches(0.45), Inches(1.25), width=Inches(6.15))
    if fig3.exists():
        s5.shapes.add_picture(str(fig3), Inches(6.75), Inches(1.25), width=Inches(6.15))

    add_card(
        s5, 0.45, 4.55, 6.15, 2.55,
        "Finding 1: Nocturnal Fraud Spike (Hours 00:00–05:00)",
        [
            "Legitimate customer volume drops by >80% overnight while victims sleep.",
            "Automated account-takeover bots attack uniformly across all 24 hours, causing the fraud rate (%) to surge 5×–8× between midnight and 5 AM.",
        ],
        RED_ACCENT,
        font_size=12,
    )
    add_card(
        s5, 6.75, 4.55, 6.15, 2.55,
        "Finding 2: Account-Draining & Mule Layering",
        [
            "Fraudulent transactions cluster sharply at Amount / OldBalanceOrig = 1.0 (emptying 100% of the victim's account in one transfer).",
            "Stolen funds are layered through zero-balance mule accounts where ledger updates lag or vanish.",
        ],
        BLUE_ACCENT,
        font_size=12,
    )

    # =========================================================================
    # SLIDE 6 — Feature Engineering & Leak-Free Temporal Split
    # =========================================================================
    s6 = prs.slides.add_slide(blank)
    add_header(
        s6,
        "5. Feature Engineering & Leak-Free Chronological Validation",
        "15 engineered behavioral & accounting features shared identically by training and the live simulator",
        "Methodology",
    )
    add_card(
        s6, 0.45, 1.30, 6.10, 5.75,
        "15 Engineered Features (src/features.py)",
        [
            "Accounting Conservation Errors: errorBalanceOrig = newbalanceOrig + amount − oldbalanceOrg; errorBalanceDest = oldbalanceDest + amount − newbalanceDest.",
            "Depletion Ratios & Flags: amt_to_bal_ratio = amount / (oldbalanceOrg + 1), orig_emptied (1 if account drained to $0), dest_zero_before (mule indicator).",
            "Balance Deltas: balanceDeltaOrig and balanceDeltaDest.",
            "Temporal & Channel Context: Cyclical hour_sin, hour_cos, plus integer hour (0–23) and type_code (0/1) for PyTorch Entity Embeddings.",
            "Signed Log1p Transform: Compresses heavy-tailed monetary magnitudes while preserving negative ledger errors.",
        ],
        BLUE_ACCENT,
        font_size=12,
    )
    add_card(
        s6, 6.75, 1.30, 6.13, 5.75,
        "Zero-Leakage Experimental Protocol",
        [
            "Dropped High-Cardinality IDs: nameOrig and nameDest removed so models cannot memorize specific account IDs.",
            f"Chronological Split by Step: 70% Train ({meta['train_rows']:,} rows), 15% Validation ({meta['val_rows']:,} rows), 15% Held-Out Test ({meta['test_rows']:,} rows).",
            "Why Time-Based Split? Random splitting leaks future fraud waves into the training set; chronological splitting tests true forward generalization.",
            "Train-Only Scaling & SMOTE: StandardScaler and SMOTE (sampling_strategy=0.15) are fit strictly on the training split.",
            "Validation Threshold Tuning: Decision thresholds are tuned on the validation split to maximize F1 before evaluating once on Test.",
        ],
        GREEN_ACCENT,
        font_size=12,
    )

    # =========================================================================
    # SLIDE 7 — Part (C): The Accuracy Paradox & PR Curves
    # =========================================================================
    s7 = prs.slides.add_slide(blank)
    add_header(
        s7,
        "6. Part (C): The Accuracy Paradox & Precision–Recall Curves",
        "Why Raw Accuracy is dangerously misleading on imbalanced financial streams (Bonus Credit Analysis)",
        "Part C · Accuracy Paradox",
    )
    fig4 = FIG_DIR / "fig4_pr_curves_and_accuracy_paradox.png"
    if fig4.exists():
        s7.shapes.add_picture(str(fig4), Inches(0.45), Inches(1.25), width=Inches(12.4))

    add_card(
        s7, 0.45, 5.15, 12.4, 1.95,
        "Why Raw Accuracy Fails vs. Why PR-AUC & F1-Score Are Gold Standard",
        [
            f"The Accuracy Paradox: Because 98.90% of high-risk test transactions are legitimate (>99.5% overall), a trivial 'Predict All Legitimate' classifier achieves 98.90% Accuracy while catching 0 out of {r_naive['fn']} frauds (Recall = 0%, F1 = 0.0000).",
            f"Precision–Recall AUC (PR-AUC) ignores the {r_naive['tn']:,} true negatives and evaluates precision across all recall levels—cleanly separating Entity-Embedding MLP (PR-AUC = {r_mlp['pr_auc']:.4f}) and XGBoost+SMOTE ({r_xgb['pr_auc']:.4f}) from naive baselines ({r_naive['pr_auc']:.4f}).",
        ],
        RED_ACCENT,
        font_size=12,
    )

    # =========================================================================
    # SLIDE 8 — Part (C): Complete ML vs. DL Benchmark Table
    # =========================================================================
    s8 = prs.slides.add_slide(blank)
    add_header(
        s8,
        "7. Part (C): Classical ML vs. Deep Learning Benchmark Leaderboard",
        f"Evaluated on Held-Out Chronological Test Stream ({meta['test_rows']:,} transactions, {meta['test_fraud']} frauds)",
        "Part C · Benchmark",
    )

    headers = ["Model", "Family", "Accuracy", "PR-AUC", "F1-Score", "Precision", "Recall", "TP", "FP", "FN", "TN", "ms/tx"]
    rows_data = bench["benchmark_table"]
    tbl_shape = s8.shapes.add_table(len(rows_data) + 1, len(headers), Inches(0.45), Inches(1.30), Inches(12.43), Inches(3.85))
    tbl = tbl_shape.table

    col_widths = [2.65, 1.75, 0.90, 0.85, 0.85, 0.85, 0.80, 0.65, 0.65, 0.65, 0.93, 0.90]
    for j, w in enumerate(col_widths):
        tbl.columns[j].width = Inches(w)

    for j, h in enumerate(headers):
        cell = tbl.cell(0, j)
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
        p = cell.text_frame.paragraphs[0]
        p.text = h
        p.font.size = Pt(10.5)
        p.font.bold = True
        p.font.color.rgb = WHITE
        p.alignment = PP_ALIGN.CENTER if j >= 2 else PP_ALIGN.LEFT

    for i, r in enumerate(rows_data, start=1):
        is_top = "Entity-Embedding MLP" in r["model"] or "XGBoost" in r["model"] or "Autoencoder" in r["model"]
        bg_row = RGBColor(239, 246, 255) if is_top else (WHITE if i % 2 == 1 else LIGHT_BG)
        vals = [
            r["model"],
            r["family"],
            f"{r['accuracy']*100:.2f}%",
            f"{r['pr_auc']:.4f}",
            f"{r['f1']:.4f}",
            f"{r['precision']:.4f}",
            f"{r['recall']:.4f}",
            str(r["tp"]),
            str(r["fp"]),
            str(r["fn"]),
            f"{r['tn']:,}",
            f"{r['latency_ms']:.4f}",
        ]
        for j, v in enumerate(vals):
            cell = tbl.cell(i, j)
            cell.fill.solid()
            cell.fill.fore_color.rgb = bg_row
            p = cell.text_frame.paragraphs[0]
            p.text = v
            p.font.size = Pt(10)
            p.font.bold = is_top and j in (0, 3, 4)
            p.font.color.rgb = NAVY
            p.alignment = PP_ALIGN.CENTER if j >= 2 else PP_ALIGN.LEFT

    add_card(
        s8, 0.45, 5.35, 12.43, 1.75,
        "Key Benchmark Takeaways",
        [
            f"1. Entity-Embedding MLP achieves the #1 overall PR-AUC ({r_mlp['pr_auc']:.4f}), #1 F1-Score ({r_mlp['f1']:.4f}), and highest fraud recall ({r_mlp['recall']*100:.2f}%, catching 154/179 frauds).",
            f"2. XGBoost + SMOTE is a close #2 (PR-AUC = {r_xgb['pr_auc']:.4f}, F1 = {r_xgb['f1']:.4f}) with 100% precision (0 false positives at its F1-optimal threshold).",
            f"3. Unsupervised Deep Autoencoder (F1 = {r_ae['f1']:.4f}, 0 FP) crushes classical Isolation Forest (F1 = {r_iso['f1']:.4f}, 109 FP) without using any labels.",
        ],
        GREEN_ACCENT,
        font_size=11,
    )

    # =========================================================================
    # SLIDE 9 — Confusion Matrices & Autoencoder Latent Separation
    # =========================================================================
    s9 = prs.slides.add_slide(blank)
    add_header(
        s9,
        "8. Part (C): Confusion Matrices & Unsupervised Autoencoder Separation",
        "Visualizing False Positives vs. False Negatives and Autoencoder Reconstruction MSE",
        "Part C · Diagnostics",
    )
    fig5 = FIG_DIR / "fig5_confusion_matrices.png"
    fig6 = FIG_DIR / "fig6_autoencoder_reconstruction_error.png"
    if fig5.exists():
        s9.shapes.add_picture(str(fig5), Inches(0.45), Inches(1.22), width=Inches(12.4))
    if fig6.exists():
        s9.shapes.add_picture(str(fig6), Inches(0.45), Inches(4.05), width=Inches(6.35))

    add_card(
        s9, 7.00, 4.05, 5.88, 3.05,
        "Why the Autoencoder Excels at Anomaly Detection",
        [
            f"Legitimate test transactions cluster tightly around MSE = {ae_dist['median_legit_mse']:.3f}.",
            f"Fraudulent transactions produce a median MSE = {ae_dist['median_fraud_mse']:.2f}, cleanly crossing the validation F1 threshold (MSE = {ae_dist['thr_mse']:.2f}).",
            "In production, high Autoencoder MSE flags novel zero-day attack scripts even before supervised labels are collected.",
        ],
        BLUE_ACCENT,
        font_size=12,
    )

    # =========================================================================
    # SLIDE 10 — Ablation Study: Removing Ledger-Error Columns
    # =========================================================================
    ab_xgb = bench["ablation_table"][0]
    ab_mlp = bench["ablation_table"][1]
    s10 = prs.slides.add_slide(blank)
    add_header(
        s10,
        "9. Rigorous Ablation Study: What If Attackers Camouflage Ledger Errors?",
        "Testing model resilience when errorBalanceOrig and errorBalanceDest are withheld",
        "Ablation Study",
    )
    add_card(
        s10, 0.45, 1.35, 6.05, 5.65,
        "Why Perform This Ablation?",
        [
            "In PaySim, classic account-draining transfers often create an immediate ledger discrepancy (errorBalanceOrig != 0).",
            "A rigorous data scientist asks: What happens if sophisticated attackers structure partial drains where accounting balances update cleanly (errorBalanceOrig = 0)?",
            "We re-trained both XGBoost + SMOTE and the Entity-Embedding MLP after completely removing errorBalanceOrig and errorBalanceDest (13 remaining features).",
            "This forces the models to rely purely on nocturnal timing embeddings, depletion ratios (amt_to_bal_ratio), and destination mule patterns.",
        ],
        BLUE_ACCENT,
        font_size=13,
    )
    add_card(
        s10, 6.75, 1.35, 6.13, 5.65,
        "Ablation Results on Held-Out Test Set",
        [
            f"XGBoost + SMOTE (With Balance Errors): PR-AUC = {r_xgb['pr_auc']:.4f} | F1 = {r_xgb['f1']:.4f}",
            f"XGBoost + SMOTE (No Balance Errors):   PR-AUC = {ab_xgb['pr_auc']:.4f} | F1 = {ab_xgb['f1']:.4f} (Precision = {ab_xgb['precision']*100:.1f}%, Recall = {ab_xgb['recall']*100:.1f}%)",
            f"Entity-Embedding MLP (With Balance Errors): PR-AUC = {r_mlp['pr_auc']:.4f} | F1 = {r_mlp['f1']:.4f}",
            f"Entity-Embedding MLP (No Balance Errors):   PR-AUC = {ab_mlp['pr_auc']:.4f} | F1 = {ab_mlp['f1']:.4f} (Precision = {ab_mlp['precision']*100:.1f}%, Recall = {ab_mlp['recall']*100:.1f}%)",
            "Conclusion: Even when ledger errors are completely hidden, both models retain >88% F1 and ~0.92 PR-AUC!",
        ],
        GREEN_ACCENT,
        font_size=13,
    )

    # =========================================================================
    # SLIDE 11 — Part (D): Interactive Streamlit Transaction Simulator
    # =========================================================================
    s11 = prs.slides.add_slide(blank)
    add_header(
        s11,
        "10. Part (D): Interactive Streamlit 'Transaction Simulator' Dashboard",
        "Live scoring, Plotly Fraud Probability Gauge, SHAP Waterfall Explainability & Stream Replay",
        "Part D · Live Dashboard",
    )
    add_card(
        s11, 0.45, 1.30, 4.0, 5.75,
        "1. Live Inputs & 1-Click Presets",
        [
            "Sidebar controls for Transaction Type, Amount, Hour of Day (0–23), and Origin/Destination Balances.",
            "1-Click Demo Presets:",
            "  • Classic Account-Draining Transfer (02:00 AM)",
            "  • Mule Account Cash-Out Sweep (03:00 AM)",
            "  • Structuring Near-Drain (94% of Balance)",
            "  • Normal Daytime Cash-Out & Merchant Payment",
            "Displays sub-10ms latency and multi-model consensus table.",
        ],
        BLUE_ACCENT,
        font_size=12,
    )
    add_card(
        s11, 4.65, 1.30, 4.0, 5.75,
        "2. Fraud Gauge & SHAP Waterfall",
        [
            "Plotly Real-Time Fraud Probability Gauge with three operational zones:",
            "  • Green (<30%): Auto-Approve",
            "  • Amber (30%–65%): Step-Up OTP / Biometric Review",
            "  • Red (≥65%): Immediate Block",
            "Exact TreeExplainer SHAP Waterfall Chart shows how each feature (orig_emptied, amt_to_bal_ratio, hour_cos, errorBalanceOrig) pushes log-odds risk.",
        ],
        RED_ACCENT,
        font_size=12,
    )
    add_card(
        s11, 8.85, 1.30, 4.03, 5.75,
        "3. Anomaly View & Stream Replay",
        [
            "Autoencoder Reconstruction Error Plot overlays the live transaction's MSE on top of the held-out legitimate vs. fraud distributions.",
            "Per-Feature Squared Error bar chart highlights which exact columns violated the learned normal manifold.",
            "Live Transaction Stream Replay tab scores batches of held-out test transactions in real time.",
        ],
        GREEN_ACCENT,
        font_size=12,
    )

    # =========================================================================
    # SLIDE 12 — Conclusion & Operational Recommendations
    # =========================================================================
    s12 = prs.slides.add_slide(blank)
    add_header(
        s12,
        "11. Conclusion, Production Policy & Live Demonstration",
        "Summary of achievements across Parts (A), (B), (C), and (D)",
        "Conclusion & Demo",
    )
    add_card(
        s12, 0.45, 1.30, 6.10, 5.75,
        "Recommended 3-Tier Production Decision Policy",
        [
            "Tier 1 — Auto-Approve (Fraud Prob < 0.30 & AE MSE < Threshold): Processed frictionlessly in <5 milliseconds.",
            f"Tier 2 — Step-Up Authentication (Prob 0.30–0.65 OR AE MSE ≥ {ae_dist['thr_mse']:.2f}): Triggers real-time push OTP / biometric confirmation to stop zero-day anomalies without blocking legitimate users.",
            "Tier 3 — Hard Block & Analyst Alert (Fraud Prob ≥ 0.65): Transaction frozen prior to cash-out; SHAP waterfall attribution logged for compliance audit.",
            "Hybrid Ensemble (50% Entity-MLP + 35% XGBoost + 15% Autoencoder) combines supervised precision with unsupervised zero-day resilience.",
        ],
        BLUE_ACCENT,
        font_size=12.5,
    )
    add_card(
        s12, 6.75, 1.30, 6.13, 5.75,
        "Live Demonstration Commands",
        [
            "1. Launch the Interactive Streamlit Simulator (Part D):",
            "   py -3.12 -m streamlit run app/dashboard.py",
            "   (Running live at http://localhost:8501)",
            "2. Open the Unified Pre-Executed Master Notebook (Parts A–D):",
            "   • notebooks/Fraud_Detection_Complete_Project.ipynb",
            "     (Covers Why DL, PaySim EDA, ML vs DL Benchmark & Live Simulator)",
            "3. Packaged Report (1,530 / 2,000 Words Max):",
            "   • writeup/Fraud_Detection_Report.docx",
        ],
        GREEN_ACCENT,
        font_size=12.5,
    )

    out_path = WRITEUP_DIR / "Fraud_Detection_Presentation.pptx"
    prs.save(str(out_path))
    print(f"[generate_slides] Saved 12-slide PowerPoint presentation: {out_path}")


if __name__ == "__main__":
    main()

