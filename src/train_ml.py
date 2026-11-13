"""Train classical Machine Learning baselines (Part C):
1. Logistic Regression (class_weight='balanced')
2. Isolation Forest (unsupervised anomaly detector trained on legitimate rows)
3. XGBoost + SMOTE (oversampling strictly on training data)

Also trains a clean-feature XGBoost ablation (excluding errorBalanceOrig/Dest)
and saves a SHAP TreeExplainer background sample for instant dashboard waterfall plots.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

from imblearn.over_sampling import SMOTE
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import LEAKY_FEATURES, NUM_FEATURES, to_model_matrix

PROC_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Find the decision threshold on validation scores that maximizes F1."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, scores)
    f1s = 2 * precisions[:-1] * recalls[:-1] / np.maximum(precisions[:-1] + recalls[:-1], 1e-12)
    best_idx = int(np.argmax(f1s))
    return float(thresholds[best_idx]), float(f1s[best_idx])


def compute_metrics(
    name: str,
    family: str,
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    latency_ms_per_tx: float,
) -> dict:
    preds = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    return {
        "model": name,
        "family": family,
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "latency_ms": float(latency_ms_per_tx),
    }


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    train_df = pd.read_parquet(PROC_DIR / "train.parquet")
    val_df = pd.read_parquet(PROC_DIR / "val.parquet")
    test_df = pd.read_parquet(PROC_DIR / "test.parquet")
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")

    X_train = pd.DataFrame(
        scaler.transform(to_model_matrix(train_df, NUM_FEATURES)),
        columns=NUM_FEATURES,
    )
    y_train = train_df["isFraud"].to_numpy()

    X_val = pd.DataFrame(
        scaler.transform(to_model_matrix(val_df, NUM_FEATURES)),
        columns=NUM_FEATURES,
    )
    y_val = val_df["isFraud"].to_numpy()

    X_test = pd.DataFrame(
        scaler.transform(to_model_matrix(test_df, NUM_FEATURES)),
        columns=NUM_FEATURES,
    )
    y_test = test_df["isFraud"].to_numpy()

    results: list[dict] = []
    pr_curves: dict[str, dict] = {}

    # 0. Naive Majority-Class Baseline ("Predict All Legitimate" — Accuracy Paradox demo)
    zero_scores = np.zeros_like(y_test, dtype=float)
    results.append(
        compute_metrics(
            name="Naive Majority Baseline (All Legit)",
            family="Baseline",
            y_true=y_test,
            scores=zero_scores,
            threshold=0.5,
            latency_ms_per_tx=0.001,
        )
    )

    # 0b. Built-in Rule Engine (`isFlaggedFraud`: TRANSFER > 200,000 rule)
    rule_scores = test_df["isFlaggedFraud"].to_numpy(dtype=float)
    results.append(
        compute_metrics(
            name="Static Rule Engine (isFlaggedFraud)",
            family="Rule-Based",
            y_true=y_test,
            scores=rule_scores,
            threshold=0.5,
            latency_ms_per_tx=0.005,
        )
    )

    # 1. Logistic Regression (class_weight='balanced')
    print("[train_ml] Training Logistic Regression (class_weight='balanced')...")
    lr = LogisticRegression(class_weight="balanced", max_iter=500, random_state=42)
    lr.fit(X_train, y_train)
    val_scores_lr = lr.predict_proba(X_val)[:, 1]
    thr_lr, _ = best_f1_threshold(y_val, val_scores_lr)

    t0 = time.perf_counter()
    test_scores_lr = lr.predict_proba(X_test)[:, 1]
    lat_lr = (time.perf_counter() - t0) * 1000.0 / len(X_test)

    results.append(
        compute_metrics("Logistic Regression", "Classical ML", y_test, test_scores_lr, thr_lr, lat_lr)
    )
    prec, rec, _ = precision_recall_curve(y_test, test_scores_lr)
    pr_curves["Logistic Regression"] = {"precision": prec[::5].tolist(), "recall": rec[::5].tolist()}
    joblib.dump({"model": lr, "threshold": thr_lr}, MODELS_DIR / "logistic_regression.joblib")

    # 2. Isolation Forest (unsupervised, trained on normal transactions only)
    print("[train_ml] Training Isolation Forest on legitimate transactions...")
    X_train_legit = X_train[y_train == 0]
    iso = IsolationForest(
        n_estimators=200,
        max_samples=min(20_000, len(X_train_legit)),
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(X_train_legit)
    # Negate decision_function so higher score = more anomalous
    val_scores_iso = -iso.decision_function(X_val)
    s_min, s_max = float(val_scores_iso.min()), float(val_scores_iso.max())
    val_scores_iso_norm = (val_scores_iso - s_min) / max(s_max - s_min, 1e-8)
    thr_iso, _ = best_f1_threshold(y_val, val_scores_iso_norm)

    t0 = time.perf_counter()
    test_scores_iso = -iso.decision_function(X_test)
    test_scores_iso_norm = np.clip((test_scores_iso - s_min) / max(s_max - s_min, 1e-8), 0.0, 1.0)
    lat_iso = (time.perf_counter() - t0) * 1000.0 / len(X_test)

    results.append(
        compute_metrics("Isolation Forest", "Unsupervised ML", y_test, test_scores_iso_norm, thr_iso, lat_iso)
    )
    prec, rec, _ = precision_recall_curve(y_test, test_scores_iso_norm)
    pr_curves["Isolation Forest"] = {"precision": prec[::5].tolist(), "recall": rec[::5].tolist()}
    joblib.dump(
        {"model": iso, "threshold": thr_iso, "s_min": s_min, "s_max": s_max},
        MODELS_DIR / "isolation_forest.joblib",
        compress=3,
    )

    # 3. XGBoost + SMOTE (oversampling ONLY on training split)
    print("[train_ml] Applying SMOTE to training split and training XGBoost...")
    smote = SMOTE(sampling_strategy=0.15, random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    print(f"[train_ml] Post-SMOTE train size: {len(X_train_sm):,} (fraud={int(y_train_sm.sum()):,})")

    xgb = XGBClassifier(
        n_estimators=260,
        max_depth=6,
        learning_rate=0.07,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
    )
    xgb.fit(X_train_sm, y_train_sm)
    val_scores_xgb = xgb.predict_proba(X_val)[:, 1]
    thr_xgb, _ = best_f1_threshold(y_val, val_scores_xgb)

    t0 = time.perf_counter()
    test_scores_xgb = xgb.predict_proba(X_test)[:, 1]
    lat_xgb = (time.perf_counter() - t0) * 1000.0 / len(X_test)

    results.append(
        compute_metrics("XGBoost + SMOTE", "Ensemble ML", y_test, test_scores_xgb, thr_xgb, lat_xgb)
    )
    prec, rec, _ = precision_recall_curve(y_test, test_scores_xgb)
    pr_curves["XGBoost + SMOTE"] = {"precision": prec[::5].tolist(), "recall": rec[::5].tolist()}
    joblib.dump({"model": xgb, "threshold": thr_xgb}, MODELS_DIR / "xgboost_smote.joblib")

    # Save a compact background sample for SHAP TreeExplainer in the dashboard
    bg_sample = X_train.sample(n=min(300, len(X_train)), random_state=42)
    joblib.dump(bg_sample, MODELS_DIR / "shap_background.joblib")

    # 4. Ablation: XGBoost without the two balance-error columns
    clean_features = [c for c in NUM_FEATURES if c not in LEAKY_FEATURES]
    scaler_clean = joblib.load(MODELS_DIR / "scaler_clean.joblib")
    X_tr_c = pd.DataFrame(scaler_clean.transform(to_model_matrix(train_df, clean_features)), columns=clean_features)
    X_va_c = pd.DataFrame(scaler_clean.transform(to_model_matrix(val_df, clean_features)), columns=clean_features)
    X_te_c = pd.DataFrame(scaler_clean.transform(to_model_matrix(test_df, clean_features)), columns=clean_features)

    X_tr_c_sm, y_tr_c_sm = SMOTE(sampling_strategy=0.15, random_state=42).fit_resample(X_tr_c, y_train)
    xgb_clean = XGBClassifier(
        n_estimators=220, max_depth=6, learning_rate=0.07, subsample=0.85,
        colsample_bytree=0.85, eval_metric="aucpr", random_state=42, n_jobs=-1,
    )
    xgb_clean.fit(X_tr_c_sm, y_tr_c_sm)
    thr_xc, _ = best_f1_threshold(y_val, xgb_clean.predict_proba(X_va_c)[:, 1])
    scores_xc = xgb_clean.predict_proba(X_te_c)[:, 1]
    ablation_metrics = compute_metrics(
        "XGBoost + SMOTE (No Balance-Error Features)", "Ablation", y_test, scores_xc, thr_xc, lat_xgb
    )

    out = {
        "ml_results": results,
        "ablation_no_balance_error": ablation_metrics,
        "pr_curves": pr_curves,
        "smote_before": {"total": int(len(y_train)), "fraud": int(y_train.sum())},
        "smote_after": {"total": int(len(y_train_sm)), "fraud": int(y_train_sm.sum())},
    }
    (MODELS_DIR / "ml_metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    for r in results:
        print(
            f"  -> {r['model']:<36} | Acc={r['accuracy']*100:6.2f}% | "
            f"PR-AUC={r['pr_auc']:.4f} | F1={r['f1']:.4f} | Prec={r['precision']:.4f} | Rec={r['recall']:.4f}"
        )


if __name__ == "__main__":
    main()

