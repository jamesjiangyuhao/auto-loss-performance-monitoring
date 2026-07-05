"""Streamlit dashboard for synthetic auto comprehensive loss monitoring."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.append(str(SRC))

from charts import claim_mix_chart, loss_ratio_trend_chart, state_performance_map, top_segments_chart
from data_processing import (
    apply_credibility_filter,
    filter_data,
    load_data,
    prepare_monthly_summary,
    prepare_segment_summary,
)
from generate_synthetic_data import main as generate_data
from metrics import add_core_metrics


DATA_PATH = ROOT / "data" / "synthetic_auto_claims.csv"


st.set_page_config(page_title="Interactive Loss Performance Monitoring", layout="wide")


@st.cache_data(show_spinner=False)
def get_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        generate_data()
    return load_data(DATA_PATH)


def format_currency(value: float) -> str:
    return f"${value:,.0f}"


def format_percent(value: float) -> str:
    return f"{value:.1%}"


def format_table(df: pd.DataFrame, currency_cols=None, percent_cols=None, numeric_cols=None) -> pd.DataFrame:
    """Return a display-safe formatted dataframe without using pandas Styler."""
    output = df.copy()
    for col in currency_cols or []:
        if col in output:
            output[col] = output[col].map(lambda value: f"${value:,.0f}")
    for col in percent_cols or []:
        if col in output:
            output[col] = output[col].map(lambda value: f"{value:.1%}")
    for col in numeric_cols or []:
        if col in output:
            output[col] = output[col].map(lambda value: f"{value:,.3f}" if abs(value) < 1 else f"{value:,.0f}")
    return output


def build_insights(filtered_df: pd.DataFrame, premium_basis: str, min_claim_count: int) -> list[str]:
    if filtered_df.empty:
        return ["No records match the current filter selection."]

    state_summary = apply_credibility_filter(
        prepare_segment_summary(filtered_df, ["state_code"], premium_basis),
        min_claim_count,
    )
    vehicle_summary = apply_credibility_filter(
        prepare_segment_summary(filtered_df, ["vehicle_make", "vehicle_model"], premium_basis),
        min_claim_count,
    )
    business_summary = prepare_segment_summary(filtered_df, ["business_type"], premium_basis)
    channel_summary = prepare_segment_summary(filtered_df, ["sales_channel"], premium_basis)

    insights = []
    if not state_summary.empty:
        row = state_summary.sort_values("loss_ratio", ascending=False).iloc[0]
        insights.append(f"{row['state_code']} has the highest credible loss ratio at {format_percent(row['loss_ratio'])}.")

    if not vehicle_summary.empty:
        row = vehicle_summary.sort_values("loss_ratio", ascending=False).iloc[0]
        insights.append(
            f"{row['vehicle_make']} {row['vehicle_model']} is the highest credible vehicle segment at "
            f"{format_percent(row['loss_ratio'])}."
        )

    if set(filtered_df["catastrophe_flag"].unique()) == {"Yes", "No"}:
        cat_summary = prepare_segment_summary(filtered_df, ["catastrophe_flag"], premium_basis)
        cat_lr = dict(zip(cat_summary["catastrophe_flag"], cat_summary["loss_ratio"]))
        lift = cat_lr.get("Yes", 0) - cat_lr.get("No", 0)
        insights.append(f"Catastrophe records change loss ratio by approximately {lift:.1%} points versus non-catastrophe records.")

    if set(business_summary["business_type"]) >= {"New Business", "Renewal"}:
        new_lr = business_summary.loc[business_summary["business_type"] == "New Business", "loss_ratio"].iloc[0]
        renewal_lr = business_summary.loc[business_summary["business_type"] == "Renewal", "loss_ratio"].iloc[0]
        direction = "worse" if new_lr > renewal_lr else "better"
        insights.append(f"New business performs {direction} than renewal on loss ratio in the current view.")

    if not channel_summary.empty:
        row = channel_summary.sort_values("claim_frequency", ascending=False).iloc[0]
        insights.append(f"{row['sales_channel']} has the highest claim frequency at {row['claim_frequency']:.3f} claims per exposure.")

    return insights[:5]


df = get_data()

st.title("Interactive Loss Performance Monitoring Dashboard")
st.caption(
    "Synthetic public portfolio project for monitoring comprehensive coverage performance across premium, losses, "
    "claims, geography, vehicle segments, channels, business type, and catastrophe indicator."
)

with st.sidebar:
    st.header("Filters")
    selected_states = st.multiselect("State", sorted(df["state_code"].unique()), default=sorted(df["state_code"].unique()))
    selected_months = st.multiselect("Period Month", sorted(df["period_month"].unique()), default=sorted(df["period_month"].unique()))
    selected_makes = st.multiselect("Vehicle Make", sorted(df["vehicle_make"].unique()), default=sorted(df["vehicle_make"].unique()))
    selected_channels = st.multiselect("Sales Channel", sorted(df["sales_channel"].unique()), default=sorted(df["sales_channel"].unique()))
    selected_business_types = st.multiselect(
        "Business Type",
        sorted(df["business_type"].unique()),
        default=sorted(df["business_type"].unique()),
    )
    include_catastrophe = st.toggle("Include catastrophe losses", value=True)
    premium_basis = st.radio("Premium Basis", ["earned_premium", "on_level_premium"], horizontal=False)
    min_claim_count = st.slider("Minimum claim count threshold", 1, 100, 10)
    top_n = st.slider("Top N segments", 5, 25, 10)

filtered = filter_data(
    df,
    selected_states=selected_states,
    selected_months=selected_months,
    selected_makes=selected_makes,
    selected_channels=selected_channels,
    selected_business_types=selected_business_types,
    include_catastrophe=include_catastrophe,
)

if filtered.empty:
    st.warning("No data matches the current filter selection.")
    st.stop()

overall = add_core_metrics(
    filtered.agg(
        {
            "exposure_units": "sum",
            "earned_premium": "sum",
            "on_level_premium": "sum",
            "claim_count": "sum",
            "incurred_loss": "sum",
        }
    ).to_frame()
    .T,
    premium_basis=premium_basis,
).iloc[0]

st.subheader("Executive KPIs")
kpi_cols = st.columns(7)
kpi_cols[0].metric("Earned Premium", format_currency(overall["earned_premium"]))
kpi_cols[1].metric("On-Level Premium", format_currency(overall["on_level_premium"]))
kpi_cols[2].metric("Incurred Loss", format_currency(overall["incurred_loss"]))
kpi_cols[3].metric("Claim Count", f"{overall['claim_count']:,.0f}")
kpi_cols[4].metric("Loss Ratio", format_percent(overall["loss_ratio"]))
kpi_cols[5].metric("Claim Frequency", f"{overall['claim_frequency']:.3f}")
kpi_cols[6].metric("Avg Severity", format_currency(overall["severity"]))

st.subheader("Monthly Trend")
monthly_df = prepare_monthly_summary(filtered, premium_basis=premium_basis)
st.plotly_chart(loss_ratio_trend_chart(monthly_df), width="stretch")

st.subheader("Claim Mix")
st.plotly_chart(claim_mix_chart(filtered), width="stretch")

st.subheader("Segment Diagnostics")
segment_df = prepare_segment_summary(filtered, ["vehicle_make", "vehicle_model"], premium_basis=premium_basis)
segment_df["vehicle_segment"] = segment_df["vehicle_make"] + " " + segment_df["vehicle_model"]
credible_segments = apply_credibility_filter(segment_df, min_claim_count=min_claim_count)
top_segments = credible_segments.sort_values("loss_ratio", ascending=False).head(top_n)
if top_segments.empty:
    st.info("No vehicle segments meet the current credibility threshold.")
else:
    st.plotly_chart(top_segments_chart(top_segments, "vehicle_segment"), width="stretch")
    st.dataframe(
        format_table(
            top_segments[
                [
                    "vehicle_make",
                    "vehicle_model",
                    "earned_premium",
                    "incurred_loss",
                    "claim_count",
                    "loss_ratio",
                    "claim_frequency",
                    "severity",
                ]
            ],
            currency_cols=["earned_premium", "incurred_loss", "severity"],
            percent_cols=["loss_ratio"],
            numeric_cols=["claim_count", "claim_frequency"],
        ),
        width="stretch",
    )

st.subheader("Geographic View")
state_df = prepare_segment_summary(filtered, ["state_code"], premium_basis=premium_basis)
st.plotly_chart(state_performance_map(state_df), width="stretch")
st.dataframe(
    format_table(
        state_df[["state_code", "exposure_units", "incurred_loss", "claim_count", "loss_ratio"]],
        currency_cols=["incurred_loss"],
        percent_cols=["loss_ratio"],
        numeric_cols=["exposure_units", "claim_count"],
    ),
    width="stretch",
)

st.subheader("Business Interpretation")
for insight in build_insights(filtered, premium_basis, min_claim_count):
    st.write(f"- {insight}")

st.subheader("Data Preview")
st.dataframe(filtered.head(500), width="stretch")
st.download_button(
    "Download filtered CSV",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="filtered_synthetic_auto_claims.csv",
    mime="text/csv",
)
