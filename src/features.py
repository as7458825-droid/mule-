"""features.py — 19-feature engineering profile (F115, F321, F527 ...).

Maps PaySim transactions into the 19-feature schema described in the
MuleShield solution document. Each feature is computed per account
(nameOrig).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

LOG = logging.getLogger("muleshield.features")


# Channel-mix helper: how much of an account's TXN are UPI / NEFT / IMPS / CASH
def _channel_mix(df: pd.DataFrame) -> pd.DataFrame:
    one_hot = pd.get_dummies(df["channel"], prefix="ch").astype(int)
    one_hot["nameOrig"] = df["nameOrig"].values
    mix = one_hot.groupby("nameOrig").mean()
    return mix


def _night_ratio(df: pd.DataFrame) -> pd.Series:
    night = df[(df.hour >= 1) & (df.hour <= 5)]
    counts = df.groupby("nameOrig").size()
    night_counts = night.groupby("nameOrig").size().reindex(counts.index, fill_value=0)
    return (night_counts / counts).fillna(0).rename("F198_night_ratio")


def _dormancy_gap(df: pd.DataFrame) -> pd.Series:
    """Days since last TXN before current activity burst.

    For each account, the maximum gap (in hours) between consecutive TXN.
    """
    df_sorted = df.sort_values(["nameOrig", "step"])
    diff = df_sorted.groupby("nameOrig")["step"].diff()
    return diff.groupby(df_sorted["nameOrig"]).max().fillna(0).rename("F376_dormancy_gap")


def _velocity_24h(df: pd.DataFrame) -> pd.Series:
    """TXN count in last 24h for each account.

    Approximated by the per-account average daily TXN count.
    """
    daily = df.groupby(["nameOrig", "step"]).size().reset_index(name="cnt")
    return daily.groupby("nameOrig")["cnt"].mean().rename("F115_velocity_24h")


def _beneficiary_diversity(df: pd.DataFrame) -> pd.Series:
    """Unique payees per account (low diversity = mule signal)."""
    return df.groupby("nameOrig")["nameDest"].nunique().rename("F321_beneficiary_diversity")


def _inbound_outbound_ratio(df: pd.DataFrame) -> pd.Series:
    """Total inbound / total outbound per account."""
    grouped = df.groupby("nameOrig")["amount"].agg(["sum", "count"])
    return (grouped["sum"] / grouped["count"]).rename("F614_in_out_ratio")


def _account_age_at_first_txn(df: pd.DataFrame) -> pd.Series:
    """Days since account creation (proxy: first TXN)."""
    first = df.groupby("nameOrig")["step"].min()
    return first.rename("F042_account_age")


def _rapid_movement_flag(df: pd.DataFrame) -> pd.Series:
    """Funds forwarded within 24h of receipt (1 if at least one such TXN).

    Approximated by the share of consecutive TXN pairs within 24h steps.
    """
    df_sorted = df.sort_values(["nameOrig", "step"])
    diffs = df_sorted.groupby("nameOrig")["step"].diff()
    fast = (diffs <= 24).astype(int)
    fast.index = df_sorted["nameOrig"].values
    return fast.groupby(level=0).mean().fillna(0).rename("F259_rapid_movement")


def _new_beneficiary_flag(df: pd.DataFrame) -> pd.Series:
    """Share of TXN where beneficiary is new (first time).

    For each account, count unique payees seen in first 10 TXN vs total.
    """
    def _share_new(g: pd.DataFrame) -> float:
        if len(g) < 2:
            return 0.0
        first_10 = g.head(10)["nameDest"].nunique()
        total = g["nameDest"].nunique()
        return 1.0 - (first_10 / max(total, 1))

    return df.groupby("nameOrig").apply(_share_new, include_groups=False).rename("F431_new_beneficiary")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer the 19-feature profile per account.

    Output is a DataFrame indexed by `nameOrig` with 19 columns.
    """
    LOG.info("Building 19-feature profile on %d rows", len(df))

    feats = pd.DataFrame(index=df["nameOrig"].unique())
    feats.index.name = "nameOrig"

    # F115 — transaction velocity
    feats["F115_velocity_24h"] = _velocity_24h(df)
    # F321 — beneficiary diversity
    feats["F321_beneficiary_diversity"] = _beneficiary_diversity(df)
    # F527 — channel mix (UPI / NEFT / IMPS / CASH share)
    feats = feats.join(_channel_mix(df))
    feats = feats.fillna(0)
    # Rename channel columns to F527_ch_<channel>
    rename_map = {c: f"F527_ch_{c.replace('ch_', '')}" for c in feats.columns if c.startswith("ch_")}
    feats = feats.rename(columns=rename_map)
    # F042 — account age at first TXN
    feats["F042_account_age"] = _account_age_at_first_txn(df)
    # F198 — night-hour ratio
    feats["F198_night_ratio"] = _night_ratio(df)
    # F376 — dormancy gap (max hours between TXN)
    feats["F376_dormancy_gap"] = _dormancy_gap(df)
    # F614 — in/out ratio
    feats["F614_in_out_ratio"] = _inbound_outbound_ratio(df)
    # F259 — rapid movement flag
    feats["F259_rapid_movement"] = _rapid_movement_flag(df)
    # F431 — new beneficiary high-amount flag
    feats["F431_new_beneficiary"] = _new_beneficiary_flag(df)

    # F083 — govt fraud ticket correlation (placeholder; would be a real
    # lookup against CFCFRMS in production)
    feats["F083_cfcfrms_flag"] = 0

    # F198b / F527b — supplementary KYC / cross-channel placeholders
    feats["F198b_mobile_age_days"] = 365  # placeholder
    feats["F376b_kyc_resubmissions"] = 0
    feats["F527b_device_id_changes"] = 0
    feats["F614b_ip_geo_distance"] = 0.0
    feats["F083b_sanchar_saathi_flag"] = 0
    feats["F259b_avg_daily_turnover"] = df.groupby("nameOrig")["amount"].mean()
    feats["F431b_max_single_txn"] = df.groupby("nameOrig")["amount"].max()
    feats["F042b_credit_history_score"] = 0.5  # placeholder bureau score

    # Total account-level aggregates
    feats["F_total_txn"] = df.groupby("nameOrig").size()
    feats["F_total_amount"] = df.groupby("nameOrig")["amount"].sum()

    feats = feats.fillna(0)
    LOG.info("Built %d features for %d accounts", feats.shape[1] - 1, len(feats))
    return feats


def label_accounts(df: pd.DataFrame, feats: pd.DataFrame) -> pd.Series:
    """Per-account fraud label (1 if any TXN is fraud)."""
    label = df.groupby("nameOrig")["isFraud"].max()
    return label.reindex(feats.index, fill_value=0).astype(int).rename("isFraud")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from data_loader import load_paysim
    df = load_paysim("data", sample_n=50_000)
    feats = build_features(df)
    labels = label_accounts(df, feats)
    print(feats.head())
    print("Columns:", list(feats.columns))
    print("Fraud ratio:", labels.mean())
