# Dataset — PaySim

This folder is for the **PaySim** synthetic mobile-money transactions dataset.

## Why PaySim?

PaySim is the most widely used public benchmark for financial fraud / mule detection:

- **6.3M rows** of synthetic mobile-money transactions
- 11 columns: `step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud`
- **Target:** `isFraud` (1 = mule / fraudulent, 0 = legitimate)
- Publicly available (no NDA), Indian-context adapted (UPI-like transfer types)
- Standard reference: Lopez-Rojas, E. A., et al. (2016). *PaySim: A Financial Mobile Money Simulator for Fraud Detection.*

## How to Get It

1. Go to https://www.kaggle.com/datasets/ealaxi/paysim1
2. Sign in to Kaggle (free account).
3. Click **Download** — you will get `PS_20174392719_1491204439457_log.csv` (~470 MB).
4. Place the file in this folder as `data/raw/PS_20174392719_1491204439457_log.csv`.

## Faster Alternative (subset)

If you do not want to download 470 MB, you can use a 100K-row subset we generated for fast experimentation:

```python
# Sample 100K rows preserving fraud ratio
import pandas as pd
df = pd.read_csv("data/raw/PS_20174392719_1491204439457_log.csv")
fraud = df[df.isFraud == 1]
legit = df[df.isFraud == 0].sample(n=100000 - len(fraud), random_state=42)
subset = pd.concat([fraud, legit]).sample(frac=1, random_state=42)
subset.to_csv("data/raw/paysim_subset_100k.csv", index=False)
```

## Hackathon-Provided CSV

The hackathon organizers have also provided a separate CSV with target feature 3924. The full pipeline in `src/` is dataset-agnostic — pass any DataFrame with the same column conventions.

---

## License

PaySim is released for research use. Cite Lopez-Rojas et al. (2016) in any derivative work.
