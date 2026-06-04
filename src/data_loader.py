"""data_loader.py — Load and clean the PaySim dataset.

Adapted to read either the full PaySim CSV or a smaller subset.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

LOG = logging.getLogger("muleshield.data_loader")

# Standard PaySim columns (after cleanup)
PAYSIM_COLS = [
    "step", "type", "amount", "nameOrig", "oldbalanceOrg",
    "newbalanceOrig", "nameDest", "oldbalanceDest",
    "newbalanceDest", "isFraud", "isFlaggedFraud",
]

# Mapping from PaySim to our internal channel names (India context)
CHANNEL_MAP = {
    "CASH_IN": "CASH",
    "CASH_OUT": "CASH",
    "DEBIT": "NEFT",
    "PAYMENT": "UPI",
    "TRANSFER": "IMPS",
}


def find_dataset(data_dir: Path) -> Path:
    """Locate the PaySim CSV in data/raw/.

    Accepts either the full file or a *_subset*.csv file.
    """
    raw = Path(data_dir) / "raw"
    if not raw.exists():
        raise FileNotFoundError(
            f"Data directory not found: {raw}. "
            "Create it and place the PaySim CSV inside (see data/README.md)."
        )

    for name in ("PS_20174392719_1491204439457_log.csv",):
        candidate = raw / name
        if candidate.exists():
            return candidate

    # Fall back to any subset
    subs = list(raw.glob("*subset*.csv")) + list(raw.glob("*.csv"))
    if subs:
        LOG.warning("Full PaySim not found; using subset %s", subs[0].name)
        return subs[0]

    raise FileNotFoundError(
        f"No PaySim CSV found in {raw}. See data/README.md for download instructions."
    )


def load_paysim(data_dir: Path | str, sample_n: int | None = None) -> pd.DataFrame:
    """Load PaySim, normalize columns, and (optionally) sample.

    If the real PaySim CSV is not present, automatically generates a
    small synthetic dataset so the pipeline can be run end-to-end for
    testing without downloading 470 MB.
    """
    data_dir = Path(data_dir)
    try:
        csv_path = find_dataset(data_dir)
    except FileNotFoundError:
        LOG.warning("PaySim CSV not found; generating synthetic data ...")
        return _synthetic_paysim(n=sample_n or 20_000)

    LOG.info("Loading %s ...", csv_path.name)

    df = pd.read_csv(csv_path)

    # PaySim uses 1 step = 1 hour; 744 steps = ~31 days
    df["hour"] = df["step"] % 24

    # Indian-channel mapping
    df["channel"] = df["type"].map(CHANNEL_MAP).fillna("OTHER")

    # Sort chronologically
    df = df.sort_values("step").reset_index(drop=True)

    if sample_n is not None and sample_n < len(df):
        fraud = df[df.isFraud == 1]
        legit = df[df.isFraud == 0]
        n_fraud = len(fraud)
        n_legit = min(sample_n - n_fraud, len(legit))
        if n_legit < sample_n - n_fraud:
            LOG.warning("Not enough legitimate rows; using %d total", n_fraud + n_legit)
        subset = pd.concat(
            [fraud, legit.sample(n=n_legit, random_state=42)]
        ).sample(frac=1, random_state=42).reset_index(drop=True)
        df = subset
        LOG.info("Sampled to %d rows (fraud ratio: %.3f%%)",
                 len(df), 100 * df.isFraud.mean())

    LOG.info("Loaded %d rows, %d fraud", len(df), int(df.isFraud.sum()))
    return df


def _synthetic_paysim(n: int = 20_000, fraud_ratio: float = 0.005, seed: int = 42) -> pd.DataFrame:
    """Generate a small synthetic PaySim-like dataset for testing.

    Includes:
    - 5% of accounts are 'mule' accounts that exhibit suspicious patterns
      (rapid pass-through, odd-hour activity, dormancy, etc.)
    - The rest are normal.
    """
    rng = np.random.default_rng(seed)
    n_accounts = max(int(n * 0.7), 500)
    accounts = [f"C{i:08d}" for i in range(n_accounts)]
    n_mules = int(n_accounts * 0.05)
    mule_set = set(rng.choice(accounts, size=n_mules, replace=False))

    rows = []
    for step in range(min(n // 50, 200)):
        for _ in range(max(n // 200, 1)):
            src = accounts[rng.integers(0, n_accounts)]
            dst = accounts[rng.integers(0, n_accounts)]
            if src == dst:
                continue

            is_mule = src in mule_set
            # Mules: larger amounts, off-hour, more often
            amount = float(rng.lognormal(8, 1.5)) if is_mule else float(rng.lognormal(6, 1.2))
            hour = int(rng.choice([2, 3, 4, 5])) if is_mule and rng.random() < 0.4 else int(rng.integers(8, 22))
            tx_type = str(rng.choice(["CASH_OUT", "TRANSFER", "PAYMENT", "DEBIT", "CASH_IN"]))
            is_fraud = 1 if is_mule and rng.random() < 0.6 else 0

            old_org = float(rng.uniform(0, 50_000))
            new_org = max(old_org - amount, 0)
            old_dest = float(rng.uniform(0, 50_000))
            new_dest = old_dest + amount

            rows.append({
                "step": int(step),
                "type": tx_type,
                "amount": amount,
                "nameOrig": src,
                "oldbalanceOrg": old_org,
                "newbalanceOrig": new_org,
                "nameDest": dst,
                "oldbalanceDest": old_dest,
                "newbalanceDest": new_dest,
                "isFraud": is_fraud,
                "isFlaggedFraud": 0,
                "hour": hour,
            })

    df = pd.DataFrame(rows)
    if len(df) > n:
        df = df.sample(n=n, random_state=seed).reset_index(drop=True)
    df["channel"] = df["type"].map(CHANNEL_MAP).fillna("OTHER")
    LOG.info("Synthetic dataset: %d rows, %d fraud (%.2f%%)",
             len(df), int(df.isFraud.sum()), 100 * df.isFraud.mean())
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = load_paysim("data", sample_n=100_000)
    print(df.head())
    print(df.isFraud.value_counts())
