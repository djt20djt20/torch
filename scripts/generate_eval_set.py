"""
Generates evals/eval_set.json — 20 labeled records for evaluating the assessment agent.

Run once: uv run python scripts/generate_eval_set.py
"""

import json
from pathlib import Path

import pandas as pd

df = pd.read_csv(Path(__file__).parent.parent / "data" / "records.csv")

df_clean = df[
    df["is_loss_making"].notna()
    & df["industry"].notna()
    & df["prior_claims"].notna()
    & df["years_trading"].notna()
    & (df["loss_ratio"] < 4.0)
    & (df["limit"] > 1000)
    & (df["premium"] > 0)
].copy()

loss = df_clean[df_clean["is_loss_making"] == True].sample(10, random_state=42)
not_loss = df_clean[df_clean["is_loss_making"] == False].sample(10, random_state=42)
eval_df = pd.concat([loss, not_loss]).sample(frac=1, random_state=42).reset_index(drop=True)

records = []
for _, row in eval_df.iterrows():
    records.append(
        {
            "record_id": row["record_id"],
            "risk_type": str(row["risk_type"]).strip().lower(),
            "territory": str(row["territory"]).strip(),
            "industry": str(row["industry"]).strip(),
            "limit": int(row["limit"]),
            "premium": int(row["premium"]),
            "broker": str(row["broker"]).strip(),
            "prior_claims": int(row["prior_claims"]),
            "years_trading": int(row["years_trading"]),
            "ground_truth": {
                "is_loss_making": bool(row["is_loss_making"]),
                "loss_ratio": float(row["loss_ratio"]),
            },
        }
    )

out = Path(__file__).parent.parent / "evals" / "eval_set.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(records, indent=2))
print(f"Wrote {len(records)} eval records to {out}")
