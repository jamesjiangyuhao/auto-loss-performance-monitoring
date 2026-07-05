"""Plotly chart builders for the Streamlit dashboard and notebook."""

from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go


PLOT_TEMPLATE = "plotly_white"


def loss_ratio_trend_chart(monthly_df):
    fig = px.line(
        monthly_df,
        x="period_month",
        y="loss_ratio",
        markers=True,
        title="Monthly Loss Ratio Trend",
        template=PLOT_TEMPLATE,
    )
    fig.update_layout(yaxis_tickformat=".0%", xaxis_title="Month", yaxis_title="Loss Ratio")
    return fig


def claim_mix_chart(df):
    mix = df.groupby("loss_category", as_index=False).agg(claim_count=("claim_count", "sum"), incurred_loss=("incurred_loss", "sum"))
    fig = px.bar(
        mix.sort_values("incurred_loss", ascending=False),
        x="loss_category",
        y="incurred_loss",
        text_auto=".2s",
        title="Claim Mix by Incurred Loss",
        template=PLOT_TEMPLATE,
    )
    fig.update_layout(xaxis_title="Loss Category", yaxis_title="Incurred Loss")
    return fig


def top_segments_chart(segment_df, segment_name):
    chart_df = segment_df.copy()
    fig = px.bar(
        chart_df.sort_values("loss_ratio", ascending=True),
        x="loss_ratio",
        y=segment_name,
        orientation="h",
        color="claim_count",
        color_continuous_scale="Teal",
        title=f"Top {len(chart_df)} Credible Segments by Loss Ratio",
        template=PLOT_TEMPLATE,
    )
    fig.update_layout(xaxis_tickformat=".0%", xaxis_title="Loss Ratio", yaxis_title="")
    return fig


def state_performance_map(state_df):
    fig = px.choropleth(
        state_df,
        locations="state_code",
        locationmode="USA-states",
        color="loss_ratio",
        scope="usa",
        hover_data={
            "state_code": True,
            "exposure_units": ":,.0f",
            "incurred_loss": ":,.0f",
            "loss_ratio": ":.1%",
        },
        color_continuous_scale="YlOrRd",
        title="State-Level Loss Ratio",
        template=PLOT_TEMPLATE,
    )
    fig.update_layout(coloraxis_colorbar_tickformat=".0%")
    return fig


def comparison_bar_chart(comparison_df):
    fig = go.Figure()
    for metric in ["loss_ratio", "claim_frequency", "severity"]:
        if metric in comparison_df.columns:
            fig.add_trace(go.Bar(name=metric.replace("_", " ").title(), x=comparison_df["segment"], y=comparison_df[metric]))
    fig.update_layout(
        barmode="group",
        title="Performance Comparison",
        template=PLOT_TEMPLATE,
        xaxis_title="Segment",
        yaxis_title="Metric Value",
    )
    return fig
