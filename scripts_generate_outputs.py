"""Generate static portfolio output charts for the auto loss monitor."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter

from src.generate_synthetic_data import build_synthetic_dataset


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
FIGSIZE = (10.5, 6.2)
BLUE = "#4f7ecb"
TEAL = "#0f766e"
RED = "#bd1f2d"
GRID = "#d1d5db"


def save(fig, name):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def add_metrics(df):
    out = df.copy()
    out["loss_ratio"] = out["incurred_loss"] / out["earned_premium"]
    return out


def create_loss_ratio_trend(df):
    monthly = df[df["period_month"].str.startswith("2025")].groupby("period_month", as_index=False).agg(
        earned_premium=("earned_premium", "sum"),
        incurred_loss=("incurred_loss", "sum"),
    )
    monthly["loss_ratio"] = monthly["incurred_loss"] / monthly["earned_premium"]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(monthly["period_month"], monthly["loss_ratio"], color=BLUE, marker="o", linewidth=2.6)
    ax.fill_between(monthly["period_month"], monthly["loss_ratio"], color=BLUE, alpha=0.14)
    ax.axhspan(0.30, 0.50, color=TEAL, alpha=0.08, label="Target monitoring band")
    ax.set_title("Monthly Comprehensive Loss Ratio Trend - 2025", fontsize=15, weight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Loss Ratio")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_ylim(0, max(0.62, monthly["loss_ratio"].max() + 0.05))
    ax.grid(axis="y", color=GRID, alpha=0.55)
    ax.legend(frameon=False, loc="upper left")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    return save(fig, "loss_ratio_trend.png")


def create_claim_mix(df):
    order = ["Glass", "Theft", "Vandalism", "Animal", "Hail", "Other"]
    mix = df.groupby("loss_category", as_index=False).agg(incurred_loss=("incurred_loss", "sum"))
    mix["share"] = mix["incurred_loss"] / mix["incurred_loss"].sum()
    mix = mix.set_index("loss_category").reindex(order).reset_index()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.barh(mix["loss_category"], mix["share"], color=BLUE)
    ax.invert_yaxis()
    for idx, value in enumerate(mix["share"]):
        ax.text(value + 0.006, idx, f"{value:.1%}", va="center", fontsize=10, color="#374151")
    ax.set_title("Comprehensive Loss Mix by Category", fontsize=15, weight="bold")
    ax.set_xlabel("Share of Incurred Loss")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlim(0, max(0.65, mix["share"].max() + 0.08))
    ax.grid(axis="x", color=GRID, alpha=0.65)
    return save(fig, "claim_mix_by_category.png")


def create_top_segments(df):
    segment = df.groupby(["vehicle_make", "vehicle_model"], as_index=False).agg(
        earned_premium=("earned_premium", "sum"),
        incurred_loss=("incurred_loss", "sum"),
        claim_count=("claim_count", "sum"),
    )
    segment = add_metrics(segment)
    segment["vehicle_segment"] = segment["vehicle_make"] + " " + segment["vehicle_model"]
    top = segment[segment["claim_count"] >= 10].sort_values("loss_ratio", ascending=False).head(10)
    colors = [RED if value >= 1.0 else TEAL for value in top["loss_ratio"]]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.barh(top["vehicle_segment"][::-1], top["loss_ratio"][::-1], color=colors[::-1])
    ax.axvline(1.0, color=RED, linestyle="--", linewidth=1.2, label="100% loss ratio")
    for idx, value in enumerate(top["loss_ratio"][::-1]):
        ax.text(value + 0.015, idx, f"{value:.0%}", va="center", fontsize=9.5, color="#374151")
    ax.set_title("Vehicle Segments with Elevated Comprehensive Loss Ratio", fontsize=15, weight="bold")
    ax.set_xlabel("Loss Ratio")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlim(0, max(1.18, top["loss_ratio"].max() + 0.12))
    ax.grid(axis="x", color=GRID, alpha=0.65)
    ax.legend(frameon=False, loc="lower right")
    return save(fig, "top_segments.png")


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = build_synthetic_dataset()
    df.to_csv(DATA_DIR / "synthetic_auto_claims.csv", index=False)
    create_loss_ratio_trend(df)
    create_claim_mix(df)
    create_top_segments(df)
    print(f"Wrote static outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
