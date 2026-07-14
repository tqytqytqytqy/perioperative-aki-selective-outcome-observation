#!/usr/bin/env python3
"""Build the v3.2 main and supplementary figures from frozen aggregate outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter, NullLocator


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
DATA = ROOT / "data" / "processed"
FIGURES = ROOT / "figures"
TEAL = "#087E8B"
GOLD = "#D69E2E"
RED = "#B6423C"
BLUE = "#3267A8"
GRAY = "#60646C"


def save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def style_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#D7D9DC", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8)


def set_odds_multiplier_axis(ax: plt.Axes) -> None:
    """Use only the prespecified MNAR multipliers on a log-scaled axis."""
    ticks = [0.5, 2 / 3, 1, 1.5, 2, 3]
    labels = ["0.5", "0.67", "1", "1.5", "2", "3"]
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter(labels))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(0.46, 3.25)


def figure1() -> None:
    flow = pd.read_csv(TABLES / "01_cohort_flow_and_roles_v32.csv")
    flow = flow.loc[flow["cohort"].isin(["INSPIRE", "MOVER 2021", "MOVER 2022"])].copy()
    observed = flow["observed_outcome_n"].to_numpy(dtype=float)
    unobserved = flow["unobserved_outcome_n"].to_numpy(dtype=float)
    total = observed + unobserved
    x = np.arange(len(flow))
    fig, ax = plt.subplots(figsize=(7.2, 4.7), constrained_layout=True)
    ax.bar(x, observed / total * 100, color=TEAL, label="Outcome observed")
    ax.bar(x, unobserved / total * 100, bottom=observed / total * 100, color=GOLD, label="Outcome unobserved")
    for i, (obs, miss, n) in enumerate(zip(observed, unobserved, total)):
        ax.text(i, obs / n * 50, f"{int(obs):,}\n({obs / n:.1%})", ha="center", va="center", color="white", fontsize=9, weight="bold")
        ax.text(i, obs / n * 100 + miss / n * 50, f"{int(miss):,}\n({miss / n:.1%})", ha="center", va="center", color="#3D2D05", fontsize=9, weight="bold")
        ax.text(i, 103, f"Eligible n={int(n):,}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, flow["cohort"].tolist())
    ax.set_ylabel("Percentage of eligible operations")
    ax.set_ylim(0, 111)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    style_axis(ax)
    save(fig, "Figure_1_outcome_observation_v32")


def figure2() -> None:
    chain = pd.read_csv(TABLES / "36_main_table2_propagation_v32.csv")
    labels = ["Source model\nunupdated", "Source model +\nlocal recalibration"]
    metrics = [
        ("update_alpha", "Update alpha", None),
        ("update_beta", "Update beta", 1.0),
        ("oe_ratio", "Target O/E ratio", 1.0),
        ("calibration_slope", "Target calibration slope", 1.0),
    ]
    colors = [GRAY, TEAL]
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 6.0), constrained_layout=True)
    for ax, (column, label, ideal) in zip(axes.flat, metrics):
        values = chain[column].to_numpy(dtype=float)
        bars = ax.bar(np.arange(2), values, color=colors, width=0.62)
        if ideal is not None:
            ax.axhline(ideal, color=RED, linestyle="--", linewidth=1)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(np.arange(2), labels)
        ax.set_ylabel(label)
        lower = min(0.0, float(values.min()) - 0.15)
        upper = max(float(values.max()) + 0.18, 1.18 if ideal is not None else 0.25)
        ax.set_ylim(lower, upper)
        style_axis(ax)
    save(fig, "Figure_2_selection_propagation_v32")


def figure_s1() -> None:
    distribution = pd.read_parquet(DATA / "canonical_bootstrap_distribution_v32.parquet")
    canonical = pd.read_csv(TABLES / "31_canonical_primary_bootstrap_v32.csv").set_index("metric")
    metrics = [
        ("oe_ratio", "O/E ratio", TEAL),
        ("calibration_slope", "Calibration slope", GOLD),
        ("auroc", "AUROC", BLUE),
        ("brier", "Brier score", RED),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.0), constrained_layout=True)
    for ax, (metric, label, color) in zip(axes.flat, metrics):
        values = distribution[metric].to_numpy(dtype=float)
        point = float(canonical.loc[metric, "estimate"])
        lower = float(canonical.loc[metric, "ci_lower"])
        upper = float(canonical.loc[metric, "ci_upper"])
        ax.hist(values, bins=28, color=color, alpha=0.82, edgecolor="white", linewidth=0.3)
        ax.axvline(point, color="black", linewidth=1.3)
        ax.axvline(lower, color=GRAY, linestyle="--", linewidth=1)
        ax.axvline(upper, color=GRAY, linestyle="--", linewidth=1)
        ax.set_xlabel(label)
        ax.set_ylabel("Bootstrap replicates")
        style_axis(ax)
    save(fig, "Figure_S1_canonical_bootstrap_v32")


def figure_s2() -> None:
    points = pd.read_csv(TABLES / "38_update_mnar_propagation_v32.csv")
    sensitivity = points.loc[~points["is_canonical"].astype(bool)].sort_values("odds_multiplier")
    canonical = points.loc[points["is_canonical"].astype(bool)].iloc[0]
    intervals = pd.read_csv(TABLES / "47_mnar_chain_bootstrap_intervals_v32.csv")
    intervals = intervals.loc[intervals["stage_varied"].eq("update")]
    metrics = [
        ("target_oe_ratio", "Target O/E ratio", 1.0),
        ("target_calibration_slope", "Target calibration slope", 1.0),
        ("target_auroc", "Target AUROC", None),
        ("threshold_10_alerts", "Alerts at 10%", None),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.0), constrained_layout=True)
    for ax, (metric, label, ideal) in zip(axes.flat, metrics):
        interval = intervals.loc[intervals["metric"].eq(metric)].sort_values("odds_multiplier")
        x = sensitivity["odds_multiplier"].to_numpy(dtype=float)
        y = sensitivity[metric].to_numpy(dtype=float)
        lower = interval["ci_lower"].to_numpy(dtype=float)
        upper = interval["ci_upper"].to_numpy(dtype=float)
        ax.plot(x, y, color=TEAL, marker="o", linewidth=1.6)
        ax.fill_between(x, lower, upper, color=TEAL, alpha=0.15)
        ax.axhline(float(canonical[metric]), color=RED, linestyle="--", linewidth=1, label="Canonical MAR")
        ax.axvline(1.0, color=GRAY, linestyle=":", linewidth=1)
        if ideal is not None:
            ax.axhline(ideal, color="#999999", linestyle=":", linewidth=0.8)
        set_odds_multiplier_axis(ax)
        ax.set_xlabel("Update-stage odds multiplier")
        ax.set_ylabel(label)
        style_axis(ax)
    axes[0, 0].legend(frameon=False, loc="lower left", fontsize=8)
    save(fig, "Figure_S2_update_mnar_v32")


def figure_s3() -> None:
    points = pd.read_csv(TABLES / "43_source_mnar_propagation_v32.csv")
    sensitivity = points.loc[~points["is_canonical"].astype(bool)].sort_values("odds_multiplier")
    intervals = pd.read_csv(TABLES / "47_mnar_chain_bootstrap_intervals_v32.csv")
    intervals = intervals.loc[intervals["stage_varied"].eq("source")]
    metrics = [
        ("target_oe_ratio", "Target O/E ratio", 1.0),
        ("target_calibration_slope", "Target calibration slope", 1.0),
        ("target_auroc", "Target AUROC", None),
        ("threshold_10_alerts", "Alerts at 10% threshold", None),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.0), constrained_layout=True)
    for ax, (metric, label, ideal) in zip(axes.flat, metrics):
        interval = intervals.loc[intervals["metric"].eq(metric)].sort_values("odds_multiplier")
        x = sensitivity["odds_multiplier"].to_numpy(dtype=float)
        y = sensitivity[metric].to_numpy(dtype=float)
        ax.plot(x, y, color=TEAL, marker="o", linewidth=1.6)
        ax.fill_between(
            x,
            interval["ci_lower"].to_numpy(dtype=float),
            interval["ci_upper"].to_numpy(dtype=float),
            color=TEAL,
            alpha=0.15,
        )
        ax.axvline(1.0, color=GRAY, linestyle="--", linewidth=1)
        if ideal is not None:
            ax.axhline(ideal, color="#999999", linestyle=":", linewidth=0.8)
        set_odds_multiplier_axis(ax)
        ax.set_xlabel("Source unobserved-outcome odds multiplier")
        ax.set_ylabel(label)
        style_axis(ax)
    save(fig, "Figure_S3_source_mnar_propagation_v32")


def figure_s4() -> None:
    tables = {
        "Source": pd.read_csv(TABLES / "43_source_mnar_propagation_v32.csv"),
        "Update": pd.read_csv(TABLES / "38_update_mnar_propagation_v32.csv"),
        "Target": pd.read_csv(TABLES / "40_target_mnar_sensitivity_v32.csv"),
    }
    metrics = [
        ("target_oe_ratio", "O/E"),
        ("target_calibration_slope", "Slope"),
        ("target_auroc", "AUROC"),
        ("threshold_10_alerts", "10% alerts"),
    ]
    fig, axes = plt.subplots(3, 4, figsize=(12.0, 8.2), constrained_layout=True)
    for row_index, (stage, table) in enumerate(tables.items()):
        sensitivity = table.loc[~table["is_canonical"].astype(bool)].sort_values("odds_multiplier")
        canonical = table.loc[table["is_canonical"].astype(bool)].iloc[0]
        for col_index, (metric, label) in enumerate(metrics):
            ax = axes[row_index, col_index]
            ax.plot(
                sensitivity["odds_multiplier"],
                sensitivity[metric],
                marker="o",
                color="#C8553D",
                linewidth=1.3,
            )
            ax.axhline(float(canonical[metric]), color=TEAL, linestyle="--", linewidth=1.1)
            ax.axvline(1.0, color=GRAY, linestyle=":", linewidth=1)
            set_odds_multiplier_axis(ax)
            ax.grid(axis="y", color="#E2E2E2", linewidth=0.5)
            if row_index == 0:
                ax.set_title(label)
            if col_index == 0:
                ax.set_ylabel(f"{stage} varied")
            if row_index == 2:
                ax.set_xlabel("Odds multiplier")
    save(fig, "Figure_S4_mnar_chain_overview_v32")


def main() -> int:
    figure1()
    figure2()
    figure_s1()
    figure_s2()
    figure_s3()
    figure_s4()
    print("Built 6 v3.2 figure pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
