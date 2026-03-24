"""
Data generation script for the Torch take-home technical test.
Generates 500 synthetic insurance records with realistic loss correlations,
intentional data quality issues for candidates to discover and handle,
plus one document per record and a new_record.json for the agent to assess.

Run once: uv run python generate_data.py
"""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)
random.seed(SEED)

N = 500

# ── Categorical pools ────────────────────────────────────────────────────────

RISK_TYPES = ["property", "liability", "marine", "cyber", "aviation"]
TERRITORIES = ["UK", "US", "EU", "APAC", "LATAM"]
INDUSTRIES = [
    "manufacturing",
    "technology",
    "retail",
    "energy",
    "financial_services",
    "healthcare",
    "transport",
]
BROKERS = [
    "Hartwell & Sons",
    "Meridian Re",
    "Caldwell Specialty",
    "Nexus Broking",
    "Pinnacle MGA",
    "Alston Risk Partners",
    "Brockton & Gray",
]

# ── Base loss-ratio means by risk type ───────────────────────────────────────

RISK_TYPE_MEAN = {
    "property": 0.62,
    "liability": 0.78,
    "marine": 0.70,
    "cyber": 0.85,
    "aviation": 0.72,
}

INDUSTRY_MEAN = {
    "manufacturing": 0.00,
    "technology": 0.05,
    "retail": 0.02,
    "energy": 0.08,
    "financial_services": 0.03,
    "healthcare": 0.06,
    "transport": 0.04,
}


# ── Clean record generation ──────────────────────────────────────────────────

def generate_records(n: int) -> pd.DataFrame:
    record_ids = [f"REC_{i:04d}" for i in range(1, n + 1)]
    risk_types = rng.choice(RISK_TYPES, size=n)
    territories = rng.choice(TERRITORIES, size=n)
    industries = rng.choice(INDUSTRIES, size=n)
    brokers = rng.choice(BROKERS, size=n)

    # Limit: log-uniform between 500k and 50M
    limits = np.exp(rng.uniform(np.log(500_000), np.log(50_000_000), size=n))
    limits = np.round(limits / 1000) * 1000

    # Premium: 0.5%–5% of limit, skewed toward lower end
    premium_rates = rng.beta(1.5, 6, size=n) * 0.045 + 0.005
    premiums = np.round(limits * premium_rates / 100) * 100

    # Prior claims: Poisson-ish, capped at 10
    prior_claims = np.clip(rng.poisson(1.2, size=n), 0, 10)

    # Years trading: 1–50
    years_trading = rng.integers(1, 51, size=n)

    # Loss ratio: base mean driven by risk type + industry + claims history + age
    base_means = np.array([RISK_TYPE_MEAN[rt] for rt in risk_types])
    industry_offsets = np.array([INDUSTRY_MEAN[ind] for ind in industries])
    claims_effect = prior_claims * 0.06
    age_effect = np.clip((10 - years_trading) * 0.008, -0.05, 0.12)

    mu = base_means + industry_offsets + claims_effect + age_effect
    noise = rng.normal(0, 0.18, size=n)
    loss_ratios = np.clip(mu + noise, 0.01, 4.0)
    loss_ratios = np.round(loss_ratios, 4)

    is_loss_making = loss_ratios > 1.0

    return pd.DataFrame(
        {
            "record_id": record_ids,
            "risk_type": risk_types,
            "territory": territories,
            "industry": industries,
            "limit": limits.astype(int),
            "premium": premiums.astype(int),
            "broker": brokers,
            "prior_claims": prior_claims,
            "years_trading": years_trading,
            "loss_ratio": loss_ratios,
            "is_loss_making": is_loss_making,
        }
    )


# ── Data quality issues ──────────────────────────────────────────────────────

def introduce_errors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Introduce realistic data quality issues that a candidate must discover
    and handle before modelling. Issues are seeded for reproducibility.

    Issues introduced:
      1. Missing values  — NaN in industry, years_trading, prior_claims (~4% each)
      2. Extreme outliers — a handful of implausible loss_ratio and limit values
      3. Label inconsistency — is_loss_making flag disagrees with loss_ratio on ~8 rows
      4. Near-duplicate rows — 4 records duplicated with minor field variation
      5. Categorical noise — mixed casing and whitespace in risk_type and territory
      6. Impossible values — negative years_trading on 2 rows; premium > limit on 3 rows
    """
    df = df.copy()
    rng_err = np.random.default_rng(99)  # separate seed so errors are stable

    n = len(df)

    # 1. Missing values (~4% per column)
    for col in ["industry", "years_trading", "prior_claims"]:
        missing_idx = rng_err.choice(n, size=int(n * 0.04), replace=False)
        df.loc[missing_idx, col] = np.nan

    # 2. Extreme outliers — loss_ratio
    outlier_lr_idx = rng_err.choice(n, size=6, replace=False)
    extreme_loss_ratios = [8.2, 11.7, 15.0, 9.4, 6.8, 12.3]
    for idx, val in zip(outlier_lr_idx, extreme_loss_ratios):
        df.loc[idx, "loss_ratio"] = val
        df.loc[idx, "is_loss_making"] = True

    # Implausible limit values (orders of magnitude off)
    outlier_limit_idx = rng_err.choice(n, size=3, replace=False)
    df.loc[outlier_limit_idx[0], "limit"] = 999_999_999   # $1bn — clearly wrong
    df.loc[outlier_limit_idx[1], "limit"] = 50             # $50 — clearly wrong
    df.loc[outlier_limit_idx[2], "premium"] = 0            # zero premium

    # 3. Label inconsistency — is_loss_making doesn't match loss_ratio
    # Pick rows where loss_ratio is just above or below 1.0 and flip the flag
    inconsistent_idx = rng_err.choice(n, size=8, replace=False)
    df.loc[inconsistent_idx, "is_loss_making"] = ~df.loc[inconsistent_idx, "is_loss_making"]

    # 4. Near-duplicate rows — copy 4 records, tweak one field slightly
    dup_source_idx = rng_err.choice(n, size=4, replace=False)
    dups = df.loc[dup_source_idx].copy()
    dups["record_id"] = [f"REC_{n + i + 1:04d}" for i in range(len(dups))]
    # Slightly vary the premium on each duplicate (e.g. rounding difference)
    dups["premium"] = (dups["premium"] * rng_err.uniform(0.99, 1.01, size=len(dups))).astype(int)
    df = pd.concat([df, dups], ignore_index=True)

    # 5. Categorical noise — mixed casing and whitespace
    # risk_type: a few rows with title-case or uppercase
    casing_rt_idx = rng_err.choice(n, size=10, replace=False)
    df.loc[casing_rt_idx, "risk_type"] = df.loc[casing_rt_idx, "risk_type"].str.title()

    # territory: a few rows with trailing/leading whitespace
    ws_idx = rng_err.choice(n, size=8, replace=False)
    df.loc[ws_idx, "territory"] = " " + df.loc[ws_idx, "territory"] + " "

    # broker: a few rows with extra whitespace
    broker_ws_idx = rng_err.choice(n, size=5, replace=False)
    df.loc[broker_ws_idx, "broker"] = df.loc[broker_ws_idx, "broker"] + " "

    # 6. Impossible values
    neg_idx = rng_err.choice(n, size=2, replace=False)
    df.loc[neg_idx, "years_trading"] = [-3, -1]

    prem_gt_limit_idx = rng_err.choice(n, size=3, replace=False)
    df.loc[prem_gt_limit_idx, "premium"] = (df.loc[prem_gt_limit_idx, "limit"] * rng_err.uniform(1.1, 2.0, size=3)).astype(int)

    return df


# ── Document generation ──────────────────────────────────────────────────────

PRIOR_CLAIMS_PHRASES = {
    0: "no prior claims history",
    1: "one prior claim on record",
    2: "two prior claims on record",
    3: "three prior claims on record",
}

TERRITORY_CONTEXT = {
    "UK": "based in the UK",
    "US": "operating in the US market",
    "EU": "domiciled in the EU",
    "APAC": "headquartered in the Asia-Pacific region",
    "LATAM": "operating across Latin America",
}

RISK_CONTEXT = {
    "property": "property damage and business interruption",
    "liability": "third-party liability and legal defence costs",
    "marine": "marine cargo and hull exposures",
    "cyber": "cyber incident response and data breach liability",
    "aviation": "aviation hull and liability",
}


def claims_phrase(n: int) -> str:
    if n in PRIOR_CLAIMS_PHRASES:
        return PRIOR_CLAIMS_PHRASES[n]
    return f"{n} prior claims on record"


def document_for_row(row: pd.Series) -> str:
    limit_m = row["limit"] / 1_000_000
    limit_str = f"${limit_m:.1f}m" if limit_m >= 1 else f"${row['limit']:,.0f}"
    premium_k = row["premium"] / 1_000
    premium_str = f"${premium_k:.0f}k"

    # Strip whitespace from categoricals for document generation (use clean values)
    risk_type = str(row["risk_type"]).strip().lower()
    territory = str(row["territory"]).strip()
    industry = str(row["industry"]).strip() if pd.notna(row["industry"]) else "an undisclosed industry"
    broker = str(row["broker"]).strip()
    years = row["years_trading"] if pd.notna(row["years_trading"]) else None
    prior = int(row["prior_claims"]) if pd.notna(row["prior_claims"]) else None

    territory_desc = TERRITORY_CONTEXT.get(territory, territory)
    risk_desc = RISK_CONTEXT.get(risk_type, risk_type)
    claims_desc = claims_phrase(prior) if prior is not None else "unknown prior claims history"

    trading_note = (
        "a recently established entity"
        if years is not None and years < 5
        else f"a business with {int(years)} years of trading history"
        if years is not None
        else "an entity with unknown trading history"
    )

    sentences = [
        f"This is a {risk_type} record covering {risk_desc}, "
        f"submitted by {broker} for a {industry} account {territory_desc}.",
        f"The insured is {trading_note}, with {claims_desc}.",
        f"The policy limit is {limit_str} and the quoted premium is {premium_str}, "
        f"placed with Torch Underwriting.",
    ]
    return " ".join(sentences)


def write_documents(df: pd.DataFrame, docs_dir: Path) -> None:
    docs_dir.mkdir(parents=True, exist_ok=True)
    for _, row in df.iterrows():
        text = document_for_row(row)
        (docs_dir / f"{row['record_id']}.txt").write_text(text)


def write_new_record(df: pd.DataFrame, path: Path) -> None:
    # Use a clean slice of the original 500 — pick a borderline record
    clean = df[df["record_id"].str.startswith("REC_")].copy()
    clean["loss_ratio_num"] = pd.to_numeric(clean["loss_ratio"], errors="coerce")
    borderline = clean[
        (clean["loss_ratio_num"] >= 0.85) & (clean["loss_ratio_num"] <= 1.15)
    ]
    row = borderline.sample(1, random_state=SEED).iloc[0]
    record = {
        "record_id": "NEW_0001",
        "risk_type": str(row["risk_type"]).strip().lower(),
        "territory": str(row["territory"]).strip(),
        "industry": str(row["industry"]).strip() if pd.notna(row["industry"]) else None,
        "limit": int(row["limit"]),
        "premium": int(row["premium"]),
        "broker": str(row["broker"]).strip(),
        "prior_claims": int(row["prior_claims"]) if pd.notna(row["prior_claims"]) else None,
        "years_trading": int(row["years_trading"]) if pd.notna(row["years_trading"]) else None,
    }
    path.write_text(json.dumps(record, indent=2))


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    base = Path(__file__).parent
    data_dir = base / "data"
    data_dir.mkdir(exist_ok=True)

    print("Generating clean records...")
    df_clean = generate_records(N)
    loss_making = df_clean["is_loss_making"].sum()
    print(f"  {N} records, {loss_making} loss-making ({loss_making / N:.1%})")

    print("Introducing data quality issues...")
    df = introduce_errors(df_clean)
    print(f"  Total rows after duplicates injected: {len(df)}")
    print(f"  Missing values per column:")
    print(df.isnull().sum()[df.isnull().sum() > 0].to_string(header=False))

    csv_path = data_dir / "records.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Wrote → {csv_path}")

    print("Writing documents...")
    write_documents(df, data_dir / "documents")
    print(f"  Wrote {len(df)} documents → {data_dir / 'documents'}/")

    print("Writing new_record.json...")
    write_new_record(df, data_dir / "new_record.json")
    print(f"  Wrote → {data_dir / 'new_record.json'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
