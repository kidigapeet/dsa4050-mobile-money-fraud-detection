"""Streamlit Interactive Dashboard (Part D):
Real-Time Mobile Money & Credit Card Fraud Detection — Live Transaction Simulator

Features:
1. Sidebar Transaction Simulator with 1-click preset attack & legitimate scenarios
2. Instant Fraud Probability Gauge (Plotly) + millisecond latency badge + risk verdict
3. Exact SHAP Feature-Importance Waterfall Chart explaining every prediction
4. Deep Autoencoder Anomaly Reconstruction Error Plot (test distribution + live tx marker)
5. ML vs. DL Benchmarking Leaderboard, Precision-Recall Curves, Confusion Matrices,
   and interactive "Accuracy Paradox" explainer
6. Live Transaction Stream Simulator (replays test stream transactions with live alerts)
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import torch

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import ALL_TYPES, FRAUD_TYPES, NUM_FEATURES, engineer, single_transaction, to_model_matrix
from src.train_dl import DeepAutoencoder, EntityEmbeddingMLP

MODELS_DIR = ROOT / "models"
PROC_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "reports" / "figures"

st.set_page_config(
    page_title="Real-Time Mobile Money Fraud Detection Simulator",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_artifacts() -> dict:
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    lr_pack = joblib.load(MODELS_DIR / "logistic_regression.joblib")
    iso_pack = joblib.load(MODELS_DIR / "isolation_forest.joblib")
    xgb_pack = joblib.load(MODELS_DIR / "xgboost_smote.joblib")
    shap_bg = joblib.load(MODELS_DIR / "shap_background.joblib")

    ae_ckpt = torch.load(MODELS_DIR / "autoencoder.pt", map_location="cpu", weights_only=False)
    ae = DeepAutoencoder(in_dim=ae_ckpt["in_dim"], latent_dim=ae_ckpt["latent_dim"])
    ae.load_state_dict(ae_ckpt["state_dict"])
    ae.eval()

    mlp_ckpt = torch.load(MODELS_DIR / "entity_mlp.pt", map_location="cpu", weights_only=False)
    mlp = EntityEmbeddingMLP(
        num_cont=mlp_ckpt["num_cont"],
        hour_emb_dim=mlp_ckpt["hour_emb_dim"],
        type_emb_dim=mlp_ckpt["type_emb_dim"],
    )
    mlp.load_state_dict(mlp_ckpt["state_dict"])
    mlp.eval()

    explainer = shap.TreeExplainer(xgb_pack["model"], data=shap_bg)

    bench = json.loads((MODELS_DIR / "benchmark_summary.json").read_text(encoding="utf-8"))
    ae_dist = json.loads((MODELS_DIR / "ae_error_distribution.json").read_text(encoding="utf-8"))
    meta = json.loads((PROC_DIR / "metadata.json").read_text(encoding="utf-8"))
    test_sample = pd.read_parquet(PROC_DIR / "test.parquet").sample(n=400, random_state=42)

    return {
        "scaler": scaler,
        "lr": lr_pack,
        "iso": iso_pack,
        "xgb": xgb_pack,
        "ae": ae,
        "ae_ckpt": ae_ckpt,
        "mlp": mlp,
        "mlp_ckpt": mlp_ckpt,
        "explainer": explainer,
        "bench": bench,
        "ae_dist": ae_dist,
        "meta": meta,
        "test_sample": test_sample,
    }


PRESETS = {
    "Custom Input": None,
    "🚨 Classic Account-Draining Transfer (02:00 AM)": {
        "tx_type": "TRANSFER",
        "amount": 485_000.0,
        "hour": 2,
        "oldbalanceOrg": 485_000.0,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
    },
    "🚨 Mule Account Cash-Out Sweep (03:00 AM)": {
        "tx_type": "CASH_OUT",
        "amount": 215_000.0,
        "hour": 3,
        "oldbalanceOrg": 215_000.0,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 12_000.0,
        "newbalanceDest": 12_000.0,
    },
    "⚠️ Structuring / Near-Drain Transfer (94% of Balance)": {
        "tx_type": "TRANSFER",
        "amount": 169_200.0,
        "hour": 1,
        "oldbalanceOrg": 180_000.0,
        "newbalanceOrig": 10_800.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 40_000.0,
    },
    "✅ Normal Daytime Cash-Out (Healthy Remaining Balance)": {
        "tx_type": "CASH_OUT",
        "amount": 18_500.0,
        "hour": 13,
        "oldbalanceOrg": 142_000.0,
        "newbalanceOrig": 123_500.0,
        "oldbalanceDest": 85_000.0,
        "newbalanceDest": 103_500.0,
    },
    "✅ Standard Merchant Payment (Zero Risk Domain Rule)": {
        "tx_type": "PAYMENT",
        "amount": 4_250.0,
        "hour": 15,
        "oldbalanceOrg": 65_000.0,
        "newbalanceOrig": 60_750.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
    },
}


def score_single_transaction(art: dict, raw_df: pd.DataFrame, active_model: str) -> dict:
    t0 = time.perf_counter()
    tx_type = str(raw_df["type"].iloc[0])
    feat_df = engineer(raw_df)
    X_mat = to_model_matrix(feat_df, NUM_FEATURES)
    X_scaled = art["scaler"].transform(X_mat).astype(np.float32)
    X_scaled_df = pd.DataFrame(X_scaled, columns=NUM_FEATURES)

    # 1. Deep Autoencoder reconstruction error + latent vector
    with torch.no_grad():
        tx_tensor = torch.from_numpy(X_scaled)
        x_hat, z = art["ae"](tx_tensor)
        recon_mse = float(torch.mean((tx_tensor - x_hat) ** 2).item())
        per_feat_sq_err = ((tx_tensor - x_hat) ** 2).squeeze(0).numpy()

    ae_ckpt = art["ae_ckpt"]
    log_mse = np.log1p(recon_mse)
    ae_prob = float(np.clip((log_mse - ae_ckpt["q01"]) / max(ae_ckpt["q99"] - ae_ckpt["q01"], 1e-6), 0.0, 1.0))

    # 2. Entity-Embedding MLP probability
    mlp_ckpt = art["mlp_ckpt"]
    e_col = np.array([[(log_mse - mlp_ckpt["err_mean"]) / mlp_ckpt["err_std"]]], dtype=np.float32)
    X_mlp = np.hstack([X_scaled, e_col, z.numpy().astype(np.float32)])
    h_idx = torch.tensor([int(feat_df["hour"].iloc[0])], dtype=torch.long)
    t_idx = torch.tensor([int(feat_df["type_code"].iloc[0])], dtype=torch.long)
    with torch.no_grad():
        mlp_logit = art["mlp"](torch.from_numpy(X_mlp), h_idx, t_idx)
        mlp_prob = float(torch.sigmoid(mlp_logit).item())

    # 3. Classical ML probabilities
    xgb_prob = float(art["xgb"]["model"].predict_proba(X_scaled_df)[0, 1])
    lr_prob = float(art["lr"]["model"].predict_proba(X_scaled_df)[0, 1])
    iso_raw = float(-art["iso"]["model"].decision_function(X_scaled_df)[0])
    iso_prob = float(
        np.clip((iso_raw - art["iso"]["s_min"]) / max(art["iso"]["s_max"] - art["iso"]["s_min"], 1e-8), 0.0, 1.0)
    )

    # Domain architectural filter: in PaySim, PAYMENT / CASH_IN / DEBIT have 0% historical fraud
    domain_zero_risk = tx_type not in FRAUD_TYPES
    if domain_zero_risk:
        mlp_prob *= 0.03
        xgb_prob *= 0.03
        lr_prob *= 0.03

    model_probs = {
        "Hybrid Deep Learning + XGBoost Ensemble (Recommended)": 0.50 * mlp_prob + 0.35 * xgb_prob + 0.15 * ae_prob,
        "Entity-Embedding MLP (Deep Learning)": mlp_prob,
        "XGBoost + SMOTE": xgb_prob,
        "Deep Autoencoder (Unsupervised)": ae_prob,
        "Logistic Regression": lr_prob,
        "Isolation Forest": iso_prob,
    }
    prob = float(model_probs[active_model])
    latency_ms = (time.perf_counter() - t0) * 1000.0

    # Compute exact SHAP values for the transaction
    shap_exp = art["explainer"](X_scaled_df)
    shap_vals = shap_exp.values[0]
    if shap_vals.ndim > 1:
        shap_vals = shap_vals[:, 1]
    base_val = float(np.atleast_1d(shap_exp.base_values)[0])

    return {
        "prob": prob,
        "all_probs": model_probs,
        "latency_ms": latency_ms,
        "recon_mse": recon_mse,
        "recon_log10_mse": float(np.log10(max(recon_mse, 1e-5))),
        "ae_thr_mse": float(ae_ckpt["thr_mse"]),
        "per_feat_sq_err": per_feat_sq_err,
        "feat_df": feat_df,
        "shap_vals": shap_vals,
        "shap_base": base_val,
        "domain_zero_risk": domain_zero_risk,
    }


def make_gauge_chart(prob: float) -> go.Figure:
    pct = prob * 100.0
    bar_color = "#10b981" if pct < 30 else ("#f59e0b" if pct < 65 else "#ef4444")
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pct,
            number={"suffix": "%", "font": {"size": 42}},
            title={"text": "Real-Time Fraud Probability", "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": bar_color, "thickness": 0.32},
                "steps": [
                    {"range": [0, 30], "color": "rgba(16, 185, 129, 0.18)"},
                    {"range": [30, 65], "color": "rgba(245, 158, 11, 0.20)"},
                    {"range": [65, 100], "color": "rgba(239, 68, 68, 0.22)"},
                ],
                "threshold": {
                    "line": {"color": "#111827", "width": 4},
                    "thickness": 0.8,
                    "value": 50,
                },
            },
        )
    )
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=45, b=15))
    return fig


def make_shap_waterfall(feat_df: pd.DataFrame, shap_vals: np.ndarray, base_val: float) -> go.Figure:
    raw_vals = feat_df[NUM_FEATURES].iloc[0]
    order = np.argsort(np.abs(shap_vals))[::-1][:10]
    top_idx = order[::-1]  # bottom to top

    labels = [f"{NUM_FEATURES[i]} = {raw_vals.iloc[i]:,.2f}" for i in top_idx]
    vals = [float(shap_vals[i]) for i in top_idx]
    colors = ["#ef4444" if v > 0 else "#2563eb" for v in vals]

    fig = go.Figure(
        go.Waterfall(
            name="SHAP",
            orientation="h",
            measure=["relative"] * len(vals),
            y=labels,
            x=vals,
            text=[f"{v:+.2f}" for v in vals],
            textposition="outside",
            decreasing={"marker": {"color": "#2563eb"}},
            increasing={"marker": {"color": "#ef4444"}},
            connector={"line": {"color": "#94a3b8", "width": 1}},
        )
    )
    fig.update_layout(
        title=f"SHAP Feature Attribution Waterfall (Log-Odds Shift from Base={base_val:.2f})",
        xaxis_title="SHAP Contribution (+ pushes toward FRAUD, − pushes toward LEGITIMATE)",
        yaxis_title="Engineered Feature & Current Value",
        height=390,
        margin=dict(l=10, r=40, t=45, b=25),
    )
    return fig


def make_ae_error_plot(ae_dist: dict, tx_log10_mse: float, tx_mse: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=ae_dist["legit_log10_mse"],
            name="Legitimate Test Stream",
            histnorm="probability density",
            marker_color="#2563eb",
            opacity=0.55,
            nbinsx=45,
        )
    )
    fig.add_trace(
        go.Histogram(
            x=ae_dist["fraud_log10_mse"],
            name="Fraudulent Test Stream",
            histnorm="probability density",
            marker_color="#ef4444",
            opacity=0.55,
            nbinsx=45,
        )
    )
    fig.add_vline(
        x=ae_dist["thr_log10_mse"],
        line_dash="dash",
        line_color="#111827",
        line_width=2,
        annotation_text=f"Anomaly Threshold (MSE={ae_dist['thr_mse']:.2f})",
        annotation_position="top left",
    )
    fig.add_vline(
        x=tx_log10_mse,
        line_color="#dc2626" if tx_mse >= ae_dist["thr_mse"] else "#10b981",
        line_width=3.5,
        annotation_text=f"Current Tx (MSE={tx_mse:.3f})",
        annotation_position="top right",
    )
    fig.update_layout(
        barmode="overlay",
        title="Deep Autoencoder Latent Reconstruction Error Distribution (log10 MSE)",
        xaxis_title="log10(Reconstruction Mean Squared Error)",
        yaxis_title="Density",
        height=370,
        margin=dict(l=15, r=15, t=45, b=25),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def main() -> None:
    art = load_artifacts()
    meta = art["meta"]

    st.title("🛡️ Real-Time Mobile Money & Credit Card Fraud Detection")
    st.caption(
        f"**DSA 4050 Packaged Project** · Dataset: *{meta.get('source_description', 'PaySim')}* · "
        f"Unsupervised **Deep Autoencoder** + **Entity-Embedding MLP** vs. Classical ML (**LR, Isolation Forest, XGBoost + SMOTE**)"
    )

    # =========================================================================
    # Sidebar — Live Transaction Simulator Controls
    # =========================================================================
    st.sidebar.header("⚡ Live Transaction Simulator")
    preset_name = st.sidebar.selectbox("1-Click Demo Preset", list(PRESETS.keys()), index=1)
    preset = PRESETS[preset_name]

    default_type = preset["tx_type"] if preset else "TRANSFER"
    default_amt = preset["amount"] if preset else 120_000.0
    default_hr = preset["hour"] if preset else 2
    default_old_orig = preset["oldbalanceOrg"] if preset else 120_000.0
    default_new_orig = preset["newbalanceOrig"] if preset else 0.0
    default_old_dest = preset["oldbalanceDest"] if preset else 0.0
    default_new_dest = preset["newbalanceDest"] if preset else 0.0

    tx_type = st.sidebar.selectbox("Transaction Type", ALL_TYPES, index=ALL_TYPES.index(default_type))
    amount = st.sidebar.number_input("Transaction Amount ($)", min_value=1.0, max_value=20_000_000.0, value=float(default_amt), step=1000.0)
    hour = st.sidebar.slider("Hour of Day (00:00 – 23:00)", min_value=0, max_value=23, value=int(default_hr))

    st.sidebar.subheader("Origin Account Balances")
    oldbalanceOrg = st.sidebar.number_input("Origin Old Balance ($)", min_value=0.0, max_value=50_000_000.0, value=float(default_old_orig), step=1000.0)
    auto_orig = st.sidebar.checkbox("Auto-compute Origin New Balance (Old − Amount)", value=False)
    if auto_orig:
        newbalanceOrig = max(0.0, oldbalanceOrg - amount) if tx_type != "CASH_IN" else oldbalanceOrg + amount
        st.sidebar.info(f"Computed Origin New Balance: **${newbalanceOrig:,.2f}**")
    else:
        newbalanceOrig = st.sidebar.number_input("Origin New Balance ($)", min_value=0.0, max_value=50_000_000.0, value=float(default_new_orig), step=1000.0)

    st.sidebar.subheader("Destination Account Balances")
    oldbalanceDest = st.sidebar.number_input("Destination Old Balance ($)", min_value=0.0, max_value=50_000_000.0, value=float(default_old_dest), step=1000.0)
    newbalanceDest = st.sidebar.number_input("Destination New Balance ($)", min_value=0.0, max_value=50_000_000.0, value=float(default_new_dest), step=1000.0)

    active_model = st.sidebar.selectbox(
        "Primary Scoring Engine",
        [
            "Hybrid Deep Learning + XGBoost Ensemble (Recommended)",
            "Entity-Embedding MLP (Deep Learning)",
            "XGBoost + SMOTE",
            "Deep Autoencoder (Unsupervised)",
            "Logistic Regression",
            "Isolation Forest",
        ],
        index=0,
    )

    raw_tx = single_transaction(
        tx_type=tx_type,
        amount=amount,
        hour=hour,
        oldbalanceOrg=oldbalanceOrg,
        newbalanceOrig=newbalanceOrig,
        oldbalanceDest=oldbalanceDest,
        newbalanceDest=newbalanceDest,
    )
    res = score_single_transaction(art, raw_tx, active_model)
    feat_row = res["feat_df"].iloc[0]

    # Top KPI Strip
    verdict = (
        "🚨 BLOCK — HIGH FRAUD RISK"
        if res["prob"] >= 0.65
        else ("⚠️ STEP-UP AUTH / OTP REVIEW" if res["prob"] >= 0.30 else "✅ APPROVE — NORMAL BEHAVIOR")
    )
    k1, k2, k3, k4 = st.columns([1.6, 1.1, 1.1, 1.1])
    k1.metric("Real-Time Decision", verdict, f"Fraud Prob: {res['prob']*100:.2f}%")
    k2.metric(
        "Origin Balance Delta",
        f"${feat_row['balanceDeltaOrig']:,.2f}",
        f"Book Error: ${feat_row['errorBalanceOrig']:,.2f}",
    )
    k3.metric(
        "Autoencoder Recon. MSE",
        f"{res['recon_mse']:.3f}",
        "ANOMALY FLAGGED" if res["recon_mse"] >= res["ae_thr_mse"] else "Within Normal Manifold",
        delta_color="inverse" if res["recon_mse"] >= res["ae_thr_mse"] else "normal",
    )
    k4.metric("Inference Latency", f"{res['latency_ms']:.2f} ms", "Sub-10ms SLA Ready")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🎛️ Live Simulator & Explainability (Part D)",
        "📊 ML vs. DL Benchmark & Accuracy Paradox (Part C)",
        "🔍 Data Mining & EDA Insights (Part B)",
        "📡 Live Transaction Stream Replay",
    ])

    with tab1:
        c_left, c_right = st.columns([1.05, 1.35])
        with c_left:
            st.plotly_chart(make_gauge_chart(res["prob"]), use_container_width=True)
            st.markdown("##### All-Model Consensus for This Transaction")
            comp_df = pd.DataFrame([
                {"Model": k, "Fraud Probability (%)": round(v * 100, 2)}
                for k, v in res["all_probs"].items()
            ])
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

        with c_right:
            st.plotly_chart(
                make_shap_waterfall(res["feat_df"], res["shap_vals"], res["shap_base"]),
                use_container_width=True,
            )

        st.divider()
        b_left, b_right = st.columns([1.35, 1.05])
        with b_left:
            st.plotly_chart(
                make_ae_error_plot(art["ae_dist"], res["recon_log10_mse"], res["recon_mse"]),
                use_container_width=True,
            )
        with b_right:
            err_df = pd.DataFrame({
                "Feature": NUM_FEATURES,
                "Squared Reconstruction Error": res["per_feat_sq_err"],
            }).sort_values("Squared Reconstruction Error", ascending=True).tail(8)
            fig_fe = px.bar(
                err_df,
                x="Squared Reconstruction Error",
                y="Feature",
                orientation="h",
                title="Top Features Driving Autoencoder Reconstruction Error",
                color="Squared Reconstruction Error",
                color_continuous_scale="Reds",
            )
            fig_fe.update_layout(height=370, margin=dict(l=10, r=15, t=45, b=25), coloraxis_showscale=False)
            st.plotly_chart(fig_fe, use_container_width=True)

    with tab2:
        st.subheader("Classical ML vs. Deep Learning Benchmarking (Held-Out Chronological Test Set)")
        bench_df = pd.DataFrame(art["bench"]["benchmark_table"])
        display_df = bench_df[[
            "model", "family", "accuracy", "pr_auc", "f1", "precision", "recall", "tp", "fp", "fn", "tn", "latency_ms"
        ]].copy()
        for col in ["accuracy", "pr_auc", "f1", "precision", "recall"]:
            display_df[col] = (display_df[col] * 100).round(2)
        display_df["latency_ms"] = display_df["latency_ms"].round(4)
        display_df.columns = [
            "Model", "Family", "Accuracy (%)", "PR-AUC (%)", "F1-Score (%)",
            "Precision (%)", "Recall (%)", "TP", "FP", "FN", "TN", "Latency (ms/tx)"
        ]
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        st.info(
            "**Why Raw Accuracy is Misleading on Imbalanced Streams (The Accuracy Paradox):** "
            "Because fraudulent transactions represent less than 1% of traffic, the **Naive Majority Baseline** "
            "(predicting *Legitimate* for every single transaction) achieves **>99% raw accuracy** while catching **0% of fraud** "
            "(F1 = 0.00). Likewise, PaySim's built-in static rule engine (`isFlaggedFraud`, which only flags transfers > $200k) "
            "misses over 98% of actual fraud. **Precision–Recall AUC (PR-AUC)** and **F1-Score** evaluate performance strictly on "
            "the minority fraud class and are the gold-standard metrics for financial anomaly detection."
        )

        if (FIG_DIR / "fig4_pr_curves_and_accuracy_paradox.png").exists():
            st.image(str(FIG_DIR / "fig4_pr_curves_and_accuracy_paradox.png"), use_container_width=True)
        if (FIG_DIR / "fig5_confusion_matrices.png").exists():
            st.image(str(FIG_DIR / "fig5_confusion_matrices.png"), use_container_width=True)

        st.markdown("##### Ablation Study: Removing Balance-Error Features (`errorBalanceOrig`, `errorBalanceDest`)")
        ab_df = pd.DataFrame(art["bench"]["ablation_table"])[
            ["model", "pr_auc", "f1", "precision", "recall"]
        ].copy()
        for col in ["pr_auc", "f1", "precision", "recall"]:
            ab_df[col] = (ab_df[col] * 100).round(2)
        st.dataframe(ab_df, use_container_width=True, hide_index=True)

    with tab3:
        st.subheader("PaySim Data Mining Findings (Part B — Class Presentation Visuals)")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Transactions Mined", f"{meta['total_rows']:,}", f"Fraud: {meta['total_fraud']:,} ({meta['overall_fraud_rate_pct']:.3f}%)")
        c2.metric("High-Risk Subset (TRANSFER + CASH_OUT)", f"{meta['risk_rows']:,}", f"100% of all fraud occurs here")
        c3.metric("Static Rule Engine (`isFlaggedFraud`) Recall", f"{100.0 * meta['flagged_tp'] / max(meta['total_fraud'], 1):.2f}%", f"Caught {meta['flagged_tp']} / {meta['total_fraud']} frauds")

        if (FIG_DIR / "fig1_fraud_by_type.png").exists():
            st.image(str(FIG_DIR / "fig1_fraud_by_type.png"), use_container_width=True)
        if (FIG_DIR / "fig2_hourly_fraud_pattern.png").exists():
            st.image(str(FIG_DIR / "fig2_hourly_fraud_pattern.png"), use_container_width=True)
        if (FIG_DIR / "fig3_amount_and_drain.png").exists():
            st.image(str(FIG_DIR / "fig3_amount_and_drain.png"), use_container_width=True)

    with tab4:
        st.subheader("📡 Live Transaction Stream Replay")
        st.write(
            "Sample a mixed batch of legitimate and fraudulent transactions from the held-out test stream "
            "and score them in real time with the Hybrid Deep Learning + XGBoost engine."
        )
        n_stream = st.slider("Number of transactions to replay", 10, 50, 20)
        sample_df = art["test_sample"].copy()
        frauds = sample_df[sample_df["isFraud"] == 1].head(max(3, n_stream // 4))
        legits = sample_df[sample_df["isFraud"] == 0].head(n_stream - len(frauds))
        stream_batch = pd.concat([frauds, legits]).sample(frac=1.0, random_state=42).reset_index(drop=True)

        rows = []
        for idx, r in stream_batch.iterrows():
            raw_one = single_transaction(
                tx_type=str(r["type"]),
                amount=float(r["amount"]),
                hour=int(r["hour"]),
                oldbalanceOrg=float(r["oldbalanceOrg"]),
                newbalanceOrig=float(r["newbalanceOrig"]),
                oldbalanceDest=float(r["oldbalanceDest"]),
                newbalanceDest=float(r["newbalanceDest"]),
            )
            sc = score_single_transaction(art, raw_one, active_model)
            rows.append({
                "Tx #": idx + 1,
                "Type": r["type"],
                "Hour": f"{int(r['hour']):02d}:00",
                "Amount ($)": round(float(r["amount"]), 2),
                "Origin Old Bal ($)": round(float(r["oldbalanceOrg"]), 2),
                "Origin New Bal ($)": round(float(r["newbalanceOrig"]), 2),
                "AE Recon MSE": round(sc["recon_mse"], 3),
                "Fraud Prob (%)": round(sc["prob"] * 100, 2),
                "Engine Action": "🚨 BLOCK" if sc["prob"] >= 0.65 else ("⚠️ REVIEW" if sc["prob"] >= 0.30 else "✅ APPROVE"),
                "Ground Truth": "FRAUD" if int(r["isFraud"]) == 1 else "LEGIT",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
