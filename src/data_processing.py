"""Reusable data processing functions for the loss monitoring dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from metrics import add_core_metrics


AGG_COLUMNS = {
    "exposure_units": "sum",
    "earned_premium": "sum",
    "on_level_premium": "sum",
    "claim_count": "sum",
    "incurred_loss": "sum",
}


def load_data(path: str | Path) -> pd.DataFrame:
    """Load synthetic auto claims data and normalize date fields."""
    df = pd.read_csv(path)
    df["period_month"] = df["period_month"].astype(str)
    return df


def _filter_optional(df: pd.DataFrame, column: str, selected_values: Iterable[str] | None) -> pd.DataFrame:
    if selected_values is None:
        return df
    selected_values = list(selected_values)
    if not selected_values:
        return df
    return df[df[column].isin(selected_values)]


def filter_data(
    df: pd.DataFrame,
    selected_states=None,
    selected_months=None,
    selected_makes=None,
    selected_channels=None,
    selected_business_types=None,
    include_catastrophe: bool = True,
) -> pd.DataFrame:
    """Apply dashboard filters to the synthetic dataset."""
    filtered = df.copy()
    filtered = _filter_optional(filtered, "state_code", selected_states)
    filtered = _filter_optional(filtered, "period_month", selected_months)
    filtered = _filter_optional(filtered, "vehicle_make", selected_makes)
    filtered = _filter_optional(filtered, "sales_channel", selected_channels)
    filtered = _filter_optional(filtered, "business_type", selected_business_types)
    if not include_catastrophe:
        filtered = filtered[filtered["catastrophe_flag"] == "No"]
    return filtered


def prepare_monthly_summary(df: pd.DataFrame, premium_basis: str = "earned_premium") -> pd.DataFrame:
    """Aggregate data to monthly grain and add core performance metrics."""
    monthly = df.groupby("period_month", as_index=False).agg(AGG_COLUMNS)
    monthly = monthly.sort_values("period_month")
    return add_core_metrics(monthly, premium_basis=premium_basis)


def prepare_segment_summary(
    df: pd.DataFrame,
    segment_cols: list[str],
    premium_basis: str = "earned_premium",
) -> pd.DataFrame:
    """Aggregate data by selected segment columns and add core metrics."""
    segment = df.groupby(segment_cols, as_index=False).agg(AGG_COLUMNS)
    return add_core_metrics(segment, premium_basis=premium_basis)


def apply_credibility_filter(df: pd.DataFrame, min_claim_count: int = 10) -> pd.DataFrame:
    """Keep only segments with enough claim volume for directional comparison."""
    return df[df["claim_count"] >= min_claim_count].copy()
