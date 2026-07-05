"""Generate a synthetic personal auto comprehensive loss dataset.

The data is fully fictional and designed for public portfolio use. It creates
controlled performance patterns that make the dashboard analytically useful
without relying on any proprietary source data or business rules.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_SEED = 42
ROW_COUNT = 60_000

STATES = ["CA", "TX", "FL", "NY", "IL", "AZ", "NV", "GA", "WA", "CO"]
STATE_REGION = {
    "CA": "West",
    "TX": "South",
    "FL": "South",
    "NY": "Northeast",
    "IL": "Midwest",
    "AZ": "West",
    "NV": "West",
    "GA": "South",
    "WA": "West",
    "CO": "West",
}
STATE_RISK = {
    "CA": 1.20,
    "TX": 1.15,
    "FL": 1.30,
    "NY": 1.05,
    "IL": 0.95,
    "AZ": 1.08,
    "NV": 1.18,
    "GA": 1.10,
    "WA": 0.88,
    "CO": 1.12,
}
STATE_PREMIUM = {
    "CA": 1.18,
    "TX": 1.08,
    "FL": 1.20,
    "NY": 1.12,
    "IL": 0.96,
    "AZ": 1.02,
    "NV": 1.10,
    "GA": 1.00,
    "WA": 0.94,
    "CO": 1.03,
}

SALES_CHANNELS = ["Independent Agent", "Direct", "Partner"]
CHANNEL_PREMIUM = {"Independent Agent": 1.00, "Direct": 0.95, "Partner": 1.04}
CHANNEL_RISK = {"Independent Agent": 1.00, "Direct": 0.94, "Partner": 1.16}

BUSINESS_TYPES = ["New Business", "Renewal"]
BUSINESS_PREMIUM = {"New Business": 1.05, "Renewal": 0.98}
BUSINESS_RISK = {"New Business": 1.18, "Renewal": 0.95}

VEHICLE_MODELS = {
    "Apex": ["Apex City", "Apex Cross", "Apex LX"],
    "Falcon": ["Falcon S", "Falcon Trail", "Falcon Prime", "Falcon EV"],
    "Orion": ["Orion Base", "Orion GT", "Orion Touring"],
    "Summit": ["Summit Ridge", "Summit Peak", "Summit Max"],
    "Metro": ["Metro One", "Metro Plus", "Metro Wagon"],
    "Titan": ["Titan Work", "Titan Crew", "Titan Sport", "Titan Max"],
    "Nova": ["Nova Spark", "Nova Glide", "Nova X"],
    "Cascade": ["Cascade Rain", "Cascade Snow", "Cascade Trail"],
}
MAKE_PREMIUM = {
    "Apex": 1.02,
    "Falcon": 1.08,
    "Orion": 1.00,
    "Summit": 1.12,
    "Metro": 0.88,
    "Titan": 1.18,
    "Nova": 0.94,
    "Cascade": 1.06,
}
MAKE_RISK = {
    "Apex": 1.10,
    "Falcon": 1.20,
    "Orion": 0.98,
    "Summit": 1.02,
    "Metro": 0.86,
    "Titan": 1.15,
    "Nova": 0.92,
    "Cascade": 1.05,
}
THEFT_HOT_MODELS = {"Falcon S", "Falcon Prime", "Titan Crew", "Apex Cross"}

LOSS_CATEGORIES = ["Theft", "Glass", "Weather", "Fire", "Other"]
LOSS_CATEGORY_PROB = [0.22, 0.30, 0.25, 0.06, 0.17]
LOSS_SEVERITY_MEAN = {
    "Theft": 6_800,
    "Glass": 950,
    "Weather": 3_700,
    "Fire": 9_500,
    "Other": 2_200,
}


def _weighted_choice(rng: np.random.Generator, values: list[str], probs: list[float], size: int) -> np.ndarray:
    return rng.choice(values, size=size, p=np.array(probs) / np.sum(probs))


def build_synthetic_dataset(row_count: int = ROW_COUNT, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Return a deterministic synthetic monthly policy-vehicle dataset."""
    rng = np.random.default_rng(seed)
    months = pd.date_range("2023-01-01", "2025-12-01", freq="MS").strftime("%Y-%m")

    state_code = _weighted_choice(
        rng,
        STATES,
        [0.18, 0.14, 0.12, 0.10, 0.09, 0.08, 0.07, 0.08, 0.07, 0.07],
        row_count,
    )
    sales_channel = _weighted_choice(rng, SALES_CHANNELS, [0.48, 0.34, 0.18], row_count)
    business_type = _weighted_choice(rng, BUSINESS_TYPES, [0.28, 0.72], row_count)
    vehicle_make = _weighted_choice(
        rng,
        list(VEHICLE_MODELS),
        [0.13, 0.14, 0.12, 0.12, 0.14, 0.13, 0.11, 0.11],
        row_count,
    )
    vehicle_model = np.array([rng.choice(VEHICLE_MODELS[make]) for make in vehicle_make])
    period_month = rng.choice(months, size=row_count)

    region = np.array([STATE_REGION[state] for state in state_code])
    exposure_units = np.round(rng.uniform(0.70, 1.00, row_count), 3)

    base_premium = rng.normal(115, 24, row_count).clip(45, 260)
    earned_premium = base_premium.copy()
    for idx in range(row_count):
        earned_premium[idx] *= STATE_PREMIUM[state_code[idx]]
        earned_premium[idx] *= CHANNEL_PREMIUM[sales_channel[idx]]
        earned_premium[idx] *= BUSINESS_PREMIUM[business_type[idx]]
        earned_premium[idx] *= MAKE_PREMIUM[vehicle_make[idx]]
    earned_premium = np.round(earned_premium * exposure_units, 2)

    period_year = pd.Series(period_month).str[:4].astype(int).to_numpy()
    on_level_factor = np.select(
        [period_year == 2023, period_year == 2024, period_year == 2025],
        [1.13, 1.06, 1.00],
        default=1.00,
    )
    on_level_premium = np.round(earned_premium * on_level_factor * rng.normal(1.0, 0.015, row_count), 2)

    risk_score = np.full(row_count, 0.030)
    for idx in range(row_count):
        risk_score[idx] *= STATE_RISK[state_code[idx]]
        risk_score[idx] *= CHANNEL_RISK[sales_channel[idx]]
        risk_score[idx] *= BUSINESS_RISK[business_type[idx]]
        risk_score[idx] *= MAKE_RISK[vehicle_make[idx]]
        if vehicle_model[idx] in THEFT_HOT_MODELS:
            risk_score[idx] *= 1.35

    catastrophe_months = {"2024-08", "2025-03"}
    catastrophe_state = np.isin(state_code, ["FL", "TX", "CO"])
    catastrophe_period = np.isin(period_month, list(catastrophe_months))
    catastrophe_flag = np.where(catastrophe_state & catastrophe_period & (rng.random(row_count) < 0.48), "Yes", "No")

    risk_score *= np.where(catastrophe_flag == "Yes", 2.6, 1.0)
    claim_count = rng.poisson(risk_score).clip(0, 2)

    loss_category = np.full(row_count, "Other", dtype=object)
    claim_rows = claim_count > 0
    loss_category[claim_rows] = _weighted_choice(
        rng,
        LOSS_CATEGORIES,
        LOSS_CATEGORY_PROB,
        int(claim_rows.sum()),
    )

    theft_boost = claim_rows & np.isin(vehicle_model, list(THEFT_HOT_MODELS))
    theft_random = rng.random(row_count) < 0.55
    loss_category[theft_boost & theft_random] = "Theft"
    loss_category[(catastrophe_flag == "Yes") & claim_rows] = np.where(
        rng.random(((catastrophe_flag == "Yes") & claim_rows).sum()) < 0.80,
        "Weather",
        loss_category[(catastrophe_flag == "Yes") & claim_rows],
    )

    incurred_loss = np.zeros(row_count)
    for category, mean in LOSS_SEVERITY_MEAN.items():
        mask = claim_rows & (loss_category == category)
        sigma = 0.65 if category in {"Theft", "Fire"} else 0.50
        severity = rng.lognormal(mean=np.log(mean), sigma=sigma, size=mask.sum())
        incurred_loss[mask] = severity * claim_count[mask]

    incurred_loss *= np.where(catastrophe_flag == "Yes", rng.normal(1.55, 0.12, row_count), 1.0)
    incurred_loss *= np.where(business_type == "New Business", 1.08, 0.98)
    incurred_loss = np.round(incurred_loss.clip(0), 2)

    df = pd.DataFrame(
        {
            "policy_id": [f"POL{1000000 + i}" for i in range(row_count)],
            "vehicle_id": [f"VEH{2000000 + i}" for i in range(row_count)],
            "period_month": period_month,
            "state_code": state_code,
            "region": region,
            "sales_channel": sales_channel,
            "business_type": business_type,
            "vehicle_make": vehicle_make,
            "vehicle_model": vehicle_model,
            "coverage_type": "Comprehensive",
            "loss_category": loss_category,
            "catastrophe_flag": catastrophe_flag,
            "exposure_units": exposure_units,
            "earned_premium": earned_premium,
            "on_level_premium": on_level_premium,
            "claim_count": claim_count,
            "incurred_loss": incurred_loss,
        }
    )
    return df.sort_values(["period_month", "state_code", "policy_id"]).reset_index(drop=True)


def main() -> None:
    output_path = Path(__file__).resolve().parents[1] / "data" / "synthetic_auto_claims.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = build_synthetic_dataset()
    df.to_csv(output_path, index=False)
    print(f"Wrote {len(df):,} synthetic rows to {output_path}")


if __name__ == "__main__":
    main()
