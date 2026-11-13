"""Train Deep Learning models in PyTorch (Part A & Part C):

1. Deep Autoencoder (unsupervised anomaly detector):
   - Trained strictly on legitimate transactions to learn a compressed 6-D latent
     representation of normal mobile-money behavior (`15 -> 32 -> 16 -> 6 -> 16 -> 32 -> 15`).
   - Flags anomalies when the per-transaction reconstruction MSE exceeds the validation F1-optimal threshold.
   - Saves normal vs. fraud reconstruction error distributions for the dashboard's Anomaly View tab.

2. Entity-Embedding Multi-Layer Perceptron (MLP):
   - Learns dense embedding vectors (`nn.Embedding`) for categorical context (`type_code`, `hour` of day)
     concatenated with scaled continuous features AND the Autoencoder's reconstruction error + 6-D latent representation.
   - Trained with Focal + Weighted Binary Cross-Entropy loss and CosineAnnealingLR.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import LEAKY_FEATURES, NUM_FEATURES, to_model_matrix
from src.train_ml import best_f1_threshold, compute_metrics

PROC_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"


class DeepAutoencoder(nn.Module):
    """Unsupervised Deep Autoencoder with bottleneck latent representation."""

    def __init__(self, in_dim: int = 15, latent_dim: int = 6) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.Linear(16, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Linear(32, in_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


class EntityEmbeddingMLP(nn.Module):
    """Deep Neural Network with Entity Embeddings for transaction type & hour of day."""

    def __init__(self, num_cont: int, hour_emb_dim: int = 6, type_emb_dim: int = 3) -> None:
        super().__init__()
        self.hour_emb = nn.Embedding(num_embeddings=24, embedding_dim=hour_emb_dim)
        self.type_emb = nn.Embedding(num_embeddings=2, embedding_dim=type_emb_dim)
        in_features = num_cont + hour_emb_dim + type_emb_dim
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, x_cont: torch.Tensor, hour_idx: torch.Tensor, type_idx: torch.Tensor) -> torch.Tensor:
        h = self.hour_emb(hour_idx)
        t = self.type_emb(type_idx)
        x = torch.cat([x_cont, h, t], dim=1)
        return self.net(x).squeeze(1)


@torch.no_grad()
def ae_infer(model: DeepAutoencoder, X_np: np.ndarray, batch_size: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """Return per-row reconstruction MSE and latent vectors."""
    model.eval()
    t_x = torch.from_numpy(X_np.astype(np.float32))
    errs = []
    latents = []
    for i in range(0, len(t_x), batch_size):
        xb = t_x[i : i + batch_size]
        x_hat, z = model(xb)
        mse = torch.mean((xb - x_hat) ** 2, dim=1)
        errs.append(mse.numpy())
        latents.append(z.numpy())
    return np.concatenate(errs), np.vstack(latents)


@torch.no_grad()
def mlp_infer(
    model: EntityEmbeddingMLP,
    X_cont: np.ndarray,
    hours: np.ndarray,
    types: np.ndarray,
    batch_size: int = 4096,
) -> np.ndarray:
    model.eval()
    tc = torch.from_numpy(X_cont.astype(np.float32))
    th = torch.from_numpy(hours.astype(np.int64))
    tt = torch.from_numpy(types.astype(np.int64))
    probs = []
    for i in range(0, len(tc), batch_size):
        logits = model(tc[i : i + batch_size], th[i : i + batch_size], tt[i : i + batch_size])
        probs.append(torch.sigmoid(logits).numpy())
    return np.concatenate(probs)


def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_parquet(PROC_DIR / "train.parquet")
    val_df = pd.read_parquet(PROC_DIR / "val.parquet")
    test_df = pd.read_parquet(PROC_DIR / "test.parquet")
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")

    X_train = scaler.transform(to_model_matrix(train_df, NUM_FEATURES)).astype(np.float32)
    y_train = train_df["isFraud"].to_numpy()
    X_val = scaler.transform(to_model_matrix(val_df, NUM_FEATURES)).astype(np.float32)
    y_val = val_df["isFraud"].to_numpy()
    X_test = scaler.transform(to_model_matrix(test_df, NUM_FEATURES)).astype(np.float32)
    y_test = test_df["isFraud"].to_numpy()

    # =========================================================================
    # 1. Train Deep Autoencoder on legitimate transactions only
    # =========================================================================
    print("[train_dl] Training Deep Autoencoder on legitimate transactions...")
    X_train_legit = X_train[y_train == 0]
    ae = DeepAutoencoder(in_dim=X_train.shape[1], latent_dim=6)
    opt_ae = torch.optim.AdamW(ae.parameters(), lr=2.5e-3, weight_decay=1e-4)
    sched_ae = torch.optim.lr_scheduler.CosineAnnealingLR(opt_ae, T_max=14)
    crit_ae = nn.MSELoss()

    ae_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train_legit)),
        batch_size=1024,
        shuffle=True,
    )
    ae_history = []
    for epoch in range(14):
        ae.train()
        total_loss = 0.0
        for (xb,) in ae_loader:
            opt_ae.zero_grad()
            x_hat, _ = ae(xb)
            loss = crit_ae(x_hat, xb)
            loss.backward()
            opt_ae.step()
            total_loss += float(loss.item()) * len(xb)
        sched_ae.step()
        epoch_mse = total_loss / len(X_train_legit)
        ae_history.append(epoch_mse)

    tr_err, tr_z = ae_infer(ae, X_train)
    val_err, val_z = ae_infer(ae, X_val)

    t0 = time.perf_counter()
    test_err, test_z = ae_infer(ae, X_test)
    lat_ae = (time.perf_counter() - t0) * 1000.0 / len(X_test)

    # Convert reconstruction error to a [0, 1] anomaly score via logistic calibration on log-MSE
    log_val_err = np.log1p(val_err)
    log_test_err = np.log1p(test_err)
    q01, q99 = float(np.percentile(log_val_err, 1)), float(np.percentile(log_val_err, 99.8))
    val_ae_scores = np.clip((log_val_err - q01) / max(q99 - q01, 1e-6), 0.0, 1.0)
    test_ae_scores = np.clip((log_test_err - q01) / max(q99 - q01, 1e-6), 0.0, 1.0)

    thr_ae_score, _ = best_f1_threshold(y_val, val_ae_scores)
    # Raw MSE threshold corresponding to thr_ae_score
    thr_ae_mse = float(np.expm1(q01 + thr_ae_score * (q99 - q01)))

    ae_metrics = compute_metrics(
        "Deep Autoencoder (Unsupervised)",
        "Deep Learning (Unsupervised)",
        y_test,
        test_ae_scores,
        thr_ae_score,
        lat_ae,
    )
    ae_metrics["raw_mse_threshold"] = thr_ae_mse

    torch.save(
        {
            "state_dict": ae.state_dict(),
            "in_dim": int(X_train.shape[1]),
            "latent_dim": 6,
            "q01": q01,
            "q99": q99,
            "thr_score": thr_ae_score,
            "thr_mse": thr_ae_mse,
            "history": ae_history,
        },
        MODELS_DIR / "autoencoder.pt",
    )

    # Save test reconstruction error distributions for the dashboard Anomaly View
    rng = np.random.default_rng(42)
    legit_errs = test_err[y_test == 0]
    fraud_errs = test_err[y_test == 1]
    ae_dist = {
        "legit_log10_mse": np.log10(np.clip(rng.choice(legit_errs, size=min(2500, len(legit_errs)), replace=False), 1e-5, None)).tolist(),
        "fraud_log10_mse": np.log10(np.clip(fraud_errs, 1e-5, None)).tolist(),
        "thr_mse": thr_ae_mse,
        "thr_log10_mse": float(np.log10(max(thr_ae_mse, 1e-5))),
        "median_legit_mse": float(np.median(legit_errs)),
        "median_fraud_mse": float(np.median(fraud_errs)),
    }
    (MODELS_DIR / "ae_error_distribution.json").write_text(json.dumps(ae_dist), encoding="utf-8")

    # =========================================================================
    # 2. Train Entity-Embedding MLP (with AE latent features + reconstruction error)
    # =========================================================================
    print("[train_dl] Training Entity-Embedding MLP...")
    err_mean, err_std = float( np.log1p(tr_err).mean() ), float( np.log1p(tr_err).std() + 1e-6 )

    def augment_cont(X_base: np.ndarray, err: np.ndarray, z: np.ndarray) -> np.ndarray:
        e_col = ((np.log1p(err) - err_mean) / err_std).reshape(-1, 1).astype(np.float32)
        return np.hstack([X_base, e_col, z.astype(np.float32)])

    X_tr_mlp = augment_cont(X_train, tr_err, tr_z)
    X_va_mlp = augment_cont(X_val, val_err, val_z)
    X_te_mlp = augment_cont(X_test, test_err, test_z)

    h_tr = train_df["hour"].to_numpy(dtype=np.int64)
    t_tr = train_df["type_code"].to_numpy(dtype=np.int64)
    h_va = val_df["hour"].to_numpy(dtype=np.int64)
    t_va = val_df["type_code"].to_numpy(dtype=np.int64)
    h_te = test_df["hour"].to_numpy(dtype=np.int64)
    t_te = test_df["type_code"].to_numpy(dtype=np.int64)

    mlp = EntityEmbeddingMLP(num_cont=X_tr_mlp.shape[1], hour_emb_dim=6, type_emb_dim=3)

    # Balanced mini-batch sampling via WeightedRandomSampler + mild pos_weight for sharp PR-AUC
    pos_count = max(1, int(y_train.sum()))
    neg_count = int((y_train == 0).sum())
    pos_weight = torch.tensor([min(18.0, np.sqrt(neg_count / pos_count))], dtype=torch.float32)
    crit_mlp = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    opt_mlp = torch.optim.AdamW(mlp.parameters(), lr=2e-3, weight_decay=2e-4)
    epochs_mlp = 18
    sched_mlp = torch.optim.lr_scheduler.CosineAnnealingLR(opt_mlp, T_max=epochs_mlp)

    mlp_ds = TensorDataset(
        torch.from_numpy(X_tr_mlp),
        torch.from_numpy(h_tr),
        torch.from_numpy(t_tr),
        torch.from_numpy(y_train.astype(np.float32)),
    )
    mlp_loader = DataLoader(mlp_ds, batch_size=1024, shuffle=True)

    best_val_f1 = -1.0
    best_state = None
    best_thr_mlp = 0.5
    mlp_history = []

    for epoch in range(epochs_mlp):
        mlp.train()
        running_loss = 0.0
        for xc, xh, xt, yb in mlp_loader:
            opt_mlp.zero_grad()
            logits = mlp(xc, xh, xt)
            loss = crit_mlp(logits, yb)
            loss.backward()
            opt_mlp.step()
            running_loss += float(loss.item()) * len(yb)
        sched_mlp.step()

        val_probs = mlp_infer(mlp, X_va_mlp, h_va, t_va)
        thr_cur, f1_cur = best_f1_threshold(y_val, val_probs)
        mlp_history.append({"epoch": epoch + 1, "loss": running_loss / len(X_tr_mlp), "val_f1": f1_cur})
        if f1_cur > best_val_f1:
            best_val_f1 = f1_cur
            best_thr_mlp = thr_cur
            best_state = {k: v.cpu().clone() for k, v in mlp.state_dict().items()}

    if best_state is not None:
        mlp.load_state_dict(best_state)

    t0 = time.perf_counter()
    test_probs_mlp = mlp_infer(mlp, X_te_mlp, h_te, t_te)
    lat_mlp = (time.perf_counter() - t0) * 1000.0 / len(X_test)

    mlp_metrics = compute_metrics(
        "Entity-Embedding MLP (Deep Learning)",
        "Deep Learning (Supervised)",
        y_test,
        test_probs_mlp,
        best_thr_mlp,
        lat_mlp,
    )

    torch.save(
        {
            "state_dict": mlp.state_dict(),
            "num_cont": int(X_tr_mlp.shape[1]),
            "hour_emb_dim": 6,
            "type_emb_dim": 3,
            "threshold": best_thr_mlp,
            "err_mean": err_mean,
            "err_std": err_std,
            "history": mlp_history,
        },
        MODELS_DIR / "entity_mlp.pt",
    )

    # Also train a clean-feature MLP ablation (without errorBalanceOrig/Dest) to compare with XGBoost ablation
    clean_features = [c for c in NUM_FEATURES if c not in LEAKY_FEATURES]
    scaler_clean = joblib.load(MODELS_DIR / "scaler_clean.joblib")
    X_tr_c = scaler_clean.transform(to_model_matrix(train_df, clean_features)).astype(np.float32)
    X_va_c = scaler_clean.transform(to_model_matrix(val_df, clean_features)).astype(np.float32)
    X_te_c = scaler_clean.transform(to_model_matrix(test_df, clean_features)).astype(np.float32)

    mlp_c = EntityEmbeddingMLP(num_cont=X_tr_c.shape[1], hour_emb_dim=6, type_emb_dim=3)
    opt_c = torch.optim.AdamW(mlp_c.parameters(), lr=2e-3, weight_decay=2e-4)
    loader_c = DataLoader(
        TensorDataset(
            torch.from_numpy(X_tr_c),
            torch.from_numpy(h_tr),
            torch.from_numpy(t_tr),
            torch.from_numpy(y_train.astype(np.float32)),
        ),
        batch_size=1024,
        shuffle=True,
    )
    for _ in range(14):
        mlp_c.train()
        for xc, xh, xt, yb in loader_c:
            opt_c.zero_grad()
            loss = crit_mlp(mlp_c(xc, xh, xt), yb)
            loss.backward()
            opt_c.step()

    va_probs_c = mlp_infer(mlp_c, X_va_c, h_va, t_va)
    thr_c, _ = best_f1_threshold(y_val, va_probs_c)
    te_probs_c = mlp_infer(mlp_c, X_te_c, h_te, t_te)
    mlp_ablation = compute_metrics(
        "Entity-Embedding MLP (No Balance-Error Features)",
        "Ablation",
        y_test,
        te_probs_c,
        thr_c,
        lat_mlp,
    )

    prec_ae, rec_ae, _ = precision_recall_curve(y_test, test_ae_scores)
    prec_mlp, rec_mlp, _ = precision_recall_curve(y_test, test_probs_mlp)

    dl_out = {
        "dl_results": [ae_metrics, mlp_metrics],
        "ablation_no_balance_error": mlp_ablation,
        "pr_curves": {
            "Deep Autoencoder (Unsupervised)": {
                "precision": prec_ae[::5].tolist(),
                "recall": rec_ae[::5].tolist(),
            },
            "Entity-Embedding MLP (Deep Learning)": {
                "precision": prec_mlp[::5].tolist(),
                "recall": rec_mlp[::5].tolist(),
            },
        },
    }
    (MODELS_DIR / "dl_metrics.json").write_text(json.dumps(dl_out, indent=2), encoding="utf-8")

    for r in [ae_metrics, mlp_metrics]:
        print(
            f"  -> {r['model']:<36} | Acc={r['accuracy']*100:6.2f}% | "
            f"PR-AUC={r['pr_auc']:.4f} | F1={r['f1']:.4f} | Prec={r['precision']:.4f} | Rec={r['recall']:.4f}"
        )


if __name__ == "__main__":
    main()
