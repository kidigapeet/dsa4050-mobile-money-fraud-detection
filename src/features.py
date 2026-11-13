"""Shared feature pipeline for the PaySim fraud project.

Training notebooks and the Streamlit dashboard both import this module, so a
transaction is always transformed in exactly the same way.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Transaction types where fraud actually happens in PaySim
FRAUD_TYPES = ["TRANSFER", "CASH_OUT"]
ALL_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

MONEY_COLS = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]

# Engineered numeric features fed to every model
NUM_FEATURES = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "errorBalanceOrig",
    "errorBalanceDest",
    "balanceDeltaOrig",
    "balanceDeltaDest",
    "amt_to_bal_ratio",
    "orig_emptied",
    "dest_zero_before",
    "hour_sin",
    "hour_cos",
    "is_transfer",
]

# The two balance-error features are known to be almost "too good" on PaySim.
# Notebook 02 also benchmarks the models without them.
LEAKY_FEATURES = ["errorBalanceOrig", "errorBalanceDest"]

# Categorical features that get entity embeddings in the MLP
CAT_FEATURES = {"type_code": 2, "hour": 24}  # name -> cardinality

# Columns that get a signed log transform before scaling (heavy-tailed money values)
LOG_COLS = [
    "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
    "errorBalanceOrig", "errorBalanceDest", "balanceDeltaOrig", "balanceDeltaDest",
    "amt_to_bal_ratio",
]


def _slog(x: pd.Series) -> pd.Series:
    """Signed log1p: keeps the sign but compresses magnitudes."""
    return np.sign(x) * np.log1p(np.abs(x))


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add the engineered features to a raw PaySim-style dataframe.

    Required columns: step, type, amount, oldbalanceOrg, newbalanceOrig,
    oldbalanceDest, newbalanceDest.
    """
    out = pd.DataFrame(index=df.index)
    for c in MONEY_COLS:
        out[c] = df[c].astype("float64")

    out["hour"] = (df["step"].astype(int) % 24).astype("int64")
    out["day"] = (df["step"].astype(int) // 24).astype("int64")

    # Money that "disappears" from the books: legitimate transactions balance out
    out["errorBalanceOrig"] = out["newbalanceOrig"] + out["amount"] - out["oldbalanceOrg"]
    out["errorBalanceDest"] = out["oldbalanceDest"] + out["amount"] - out["newbalanceDest"]

    out["balanceDeltaOrig"] = out["oldbalanceOrg"] - out["newbalanceOrig"]
    out["balanceDeltaDest"] = out["newbalanceDest"] - out["oldbalanceDest"]
    out["amt_to_bal_ratio"] = out["amount"] / (out["oldbalanceOrg"] + 1.0)
    out["orig_emptied"] = ((out["oldbalanceOrg"] > 0) & (out["newbalanceOrig"] == 0)).astype("float64")
    out["dest_zero_before"] = ((out["oldbalanceDest"] == 0) & (out["newbalanceDest"] == 0)).astype("float64")

    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)

    out["is_transfer"] = (df["type"] == "TRANSFER").astype("float64")
    out["type_code"] = out["is_transfer"].astype("int64")  # 0 = CASH_OUT, 1 = TRANSFER
    return out


def to_model_matrix(feat: pd.DataFrame, features: list[str] | None = None) -> pd.DataFrame:
    """Apply the signed-log transform and return the numeric model matrix (unscaled)."""
    features = features or NUM_FEATURES
    X = feat[features].copy()
    for c in features:
        if c in LOG_COLS:
            X[c] = _slog(X[c])
    return X


def single_transaction(
    tx_type: str,
    amount: float,
    hour: int,
    oldbalanceOrg: float,
    newbalanceOrig: float,
    oldbalanceDest: float,
    newbalanceDest: float,
) -> pd.DataFrame:
    """Build a 1-row raw dataframe from dashboard inputs."""
    return pd.DataFrame(
        [{
            "step": int(hour),
            "type": tx_type,
            "amount": amount,
            "oldbalanceOrg": oldbalanceOrg,
            "newbalanceOrig": newbalanceOrig,
            "oldbalanceDest": oldbalanceDest,
            "newbalanceDest": newbalanceDest,
        }]
    )

