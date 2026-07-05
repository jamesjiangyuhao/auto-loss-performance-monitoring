"""Metric helpers for insurance loss performance analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_loss_ratio(incurred_loss, premium):
    """Calculate incurred loss divided by premium with zero-safe handling."""
    return np.divide(incurred_loss, premium, out=np.zeros_like(incurred_loss, dtype=float), where=np.asarray(premium) != 0)


def calculate_claim_frequency(claim_count, exposure_units):
    """Calculate claims per exposure unit with zero-safe handling."""
    return np.divide(
        claim_count,
        exposure_units,
        out=np.zeros_like(claim_count, dtype=float),
        where=np.asarray(exposure_units) != 0,
    )


def calculate_severity(incurred_loss, claim_count):
    """Calculate average incurred loss per claim with zero-safe handling."""
    return np.divide(
        incurred_loss,
        claim_count,
        out=np.zeros_like(incurred_loss, dtype=float),
        where=np.asarray(claim_count) != 0,
    )


def add_core_metrics(df: pd.DataFrame, premium_basis: str = "earned_premium") -> pd.DataFrame:
    """Add loss ratio, claim frequency, and severity columns to a dataframe."""
    output = df.copy()
    output["loss_ratio"] = calculate_loss_ratio(output["incurred_loss"].to_numpy(), output[premium_basis].to_numpy())
    output["claim_frequency"] = calculate_claim_frequency(
        output["claim_count"].to_numpy(),
        output["exposure_units"].to_numpy(),
    )
    output["severity"] = calculate_severity(output["incurred_loss"].to_numpy(), output["claim_count"].to_numpy())
    return output
