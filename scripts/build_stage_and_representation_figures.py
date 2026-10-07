"""Build stage-specific and fixed-prediction figures from retained aggregate outputs."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, NullLocator


PROJECT = Path(__file__).resolve().parents[1]
TABLES = PROJECT / "tables"
STAGES = {
    "source": ("43_source_mnar_propagation_v32.csv", "Source development | INSPIRE", "#0072B2", "o"),
    "update": ("38_update_mnar_propagation_v32.csv", "Local recalibration | MOVER 2021", "#B44E28", "s"),
    "target": ("40_target_mnar_sensitivity_v32.csv", "Target evaluation | MOVER 2022", "#008571", "D"),
}
INTERVALS = "47_mnar_chain_bootstrap_intervals_v32.csv"
PAIRED = "R1_S21_paired_reclassification.csv"
METRICS = "R1_S22_fixed_prediction_metrics.csv"
GRID = np.array([0.5, 0.6666666667, 1.0, 1.5, 2.0, 3.0])
INK = "#20252A"
GREY = "#657078"


def read_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def select_one(rows, **criteria):
    selected = [row for row in rows if all(row[k] == v for k, v in criteria.items())]
    require(len(selected) == 1, f"Expected one row: {criteria}; found {len(selected)}")
    return selected[0]


def load_inputs(tables):
    interval_rows = read_rows(tables / INTERVALS)
    stage_data, plot_rows = {}, []
    for stage, (filename, _, _, _) in STAGES.items():
        rows = read_rows(tables / filename)
        canonical = select_one(rows, is_canonical="True")
        scenarios = sorted([r for r in rows if r["is_canonical"] == "False"], key=lambda r: float(r["odds_multiplier"]))
        require(np.allclose([float(r["odds_multiplier"]) for r in scenarios], GRID, atol=1e-10), f"Multiplier grid: {stage}")
        require(all(r["stage_varied"] == stage and r["target_eligible_n"] == "2587" for r in rows), f"Stage or denominator mismatch: {stage}")
        require(scenarios[2]["reference_type"] == "OR-completion reference, k=1", f"Incorrect k=1 reference: {stage}")
        for row in scenarios:
            for metric in ["target_oe_ratio", "threshold_10_alerts_per_1000"]:
                ci = select_one(interval_rows, stage_varied=stage, odds_multiplier=row["odds_multiplier"], metric=metric)
                require(ci["estimator_family"] == row["estimator_family"] and ci["reference_type"] == row["reference_type"], "Mismatched estimator/CI")
                require(ci["is_canonical"] == "False" and ci["bootstrap_replicates"] == "1000", "Wrong interval family")
                require(ci["canonical_source"] == "data/processed/mnar_chain_bootstrap_v32.parquet", "Not full-chain MNAR intervals")
                require(np.isclose(float(ci["estimate"]), float(row[metric]), atol=1e-12, rtol=0), "Point estimate/CI mismatch")
                require(float(ci["ci_lower"]) <= float(ci["estimate"]) <= float(ci["ci_upper"]), "Inverted CI")
                plot_rows.append({"stage": stage, "k": float(row["odds_multiplier"]), "metric": metric, "estimate": float(row[metric]), "ci_lower": float(ci["ci_lower"]), "ci_upper": float(ci["ci_upper"]), "canonical_mar": float(canonical[metric]), "target_n": int(row["target_eligible_n"]), "point_source": filename, "interval_source": INTERVALS})
        stage_data[stage] = scenarios
    counts = select_one(read_rows(tables / PAIRED), cohort="MOVER 2022")
    paired_metrics = read_rows(tables / METRICS)
    require(all(r["observed_n"] == "2033" and r["replicates"] == "1000" for r in paired_metrics), "Wrong paired cohort or bootstrap count")
    matrix = np.array([[int(counts[f"original_{a}_coarsened_{b}"]) for b in range(2)] for a in range(2)])
    require(matrix.sum() == int(counts["observed_n"]) == 2033, "Matrix denominator")
    require(matrix[0, 1] == 120 and matrix[1, 0] == 43, "Wrong transition direction/count")
    require(matrix[0, 1] + matrix[1, 0] == int(counts["discordant_n"]), "Discordance count")
    for metric in ["oe_ratio", "auroc"]:
        original = float(select_one(paired_metrics, metric=metric, contrast="original")["estimate"])
        coarsened = float(select_one(paired_metrics, metric=metric, contrast="coarsened")["estimate"])
        difference = float(select_one(paired_metrics, metric=metric, contrast="difference")["estimate"])
        require(np.isclose(coarsened - original, difference, atol=1e-12), "Wrong paired difference direction")
    return plot_rows, counts, matrix, paired_metrics


def configure():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "text.color": INK,
        "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.edgecolor": "#B6BCC0", "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.unicode_minus": False,
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })


def save_figure(fig, stem, out):
    fig.savefig(out / f"{stem}.pdf", metadata={"Title": stem, "Creator": "Matplotlib; aggregate-table visualization", "CreationDate": None, "ModDate": None})
    fig.savefig(out / f"{stem}.png", dpi=300)
    plt.close(fig)


def figure_2(rows, out):
    fig = plt.figure(figsize=(7.1, 6.35))
    fig.text(0.055, 0.976, "Stage-specific MNAR sensitivity", fontsize=12, weight="bold", va="top")
    fig.text(0.055, 0.935, "Target: MOVER 2022 eligible operations (n = 2,587)", fontsize=9.3, va="top", color=GREY)
    fig.text(0.105, 0.885, "A   Target O/E", fontsize=10.5, weight="bold")
    fig.text(0.60, 0.885, "B   Alerts per 1,000 at 10%", fontsize=10.5, weight="bold")
    for i, (stage, (_, label, color, marker)) in enumerate(STAGES.items()):
        bottom = 0.655 - 0.218 * i
        fig.text(0.105, bottom + 0.166, label, fontsize=9.5, color=color, weight="bold")
        for col, metric in enumerate(["target_oe_ratio", "threshold_10_alerts_per_1000"]):
            ax = fig.add_axes([0.105 if col == 0 else 0.60, bottom, 0.36, 0.143])
            selected = [r for r in rows if r["stage"] == stage and r["metric"] == metric]
            x = np.array([r["k"] for r in selected])
            y = np.array([r["estimate"] for r in selected])
            lo = np.array([r["ci_lower"] for r in selected])
            hi = np.array([r["ci_upper"] for r in selected])
            ax.set_xscale("log")
            ax.set_xlim(0.45, 3.33)
            ax.xaxis.set_major_locator(FixedLocator(GRID))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.set_xticks(GRID, ["0.5", "0.67", "1", "1.5", "2", "3"] if i == 2 else [""] * 6)
            ax.axvspan(0.96, 1.04, facecolor="#EAECEE", edgecolor="none", zorder=0)
            ax.axvline(1, color="#D1D5D7", lw=0.7, zorder=0)
            ax.axhline(selected[0]["canonical_mar"], color=GREY, lw=0.9, ls=(0, (4, 3)), zorder=1)
            ax.errorbar(x, y, yerr=np.vstack([y - lo, hi - y]), color=color, marker=marker, markersize=4.1, markeredgewidth=0.7, markeredgecolor="white", lw=1.25, elinewidth=0.85, capsize=2.3, capthick=0.85, zorder=3)
            ax.set_ylim((0.65, 1.45) if col == 0 else (0, 900))
            ax.set_yticks([0.75, 1.00, 1.25] if col == 0 else [0, 400, 800])
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", color="#EAECEF", linewidth=0.55, zorder=-1)
            ax.tick_params(axis="x", length=3, pad=4)
            if i == 2:
                ax.set_xlabel("MNAR odds multiplier, k", labelpad=6, fontsize=9)
    ci_handle = ax.errorbar([], [], yerr=[], fmt="o", color=INK, markersize=3, lw=0.8, capsize=2)
    legend = [Line2D([0], [0], color=GREY, ls=(0, (4, 3)), lw=1), ci_handle]
    fig.legend(legend, ["Canonical MAR (separate estimator)", "95% full-chain bootstrap CI"], loc="lower left", bbox_to_anchor=(0.055, 0.102), ncol=2, frameon=False, fontsize=8.1, handlelength=2.4, columnspacing=1.6, borderaxespad=0)
    fig.text(0.055, 0.078, "k = 1: OR-completion reference; not the canonical MAR estimate.", fontsize=8.4)
    fig.text(0.055, 0.048, "1,000 bootstrap replicates. Sensitivity scenarios, not established clinical risk bounds.", fontsize=8.1)
    save_figure(fig, "Figure_3", out)


def figure_3(counts, matrix, metrics, out):
    fig = plt.figure(figsize=(7.1, 4.85))
    fig.text(0.055, 0.976, "Creatinine coarsening with fixed predictions", fontsize=12, weight="bold", va="top")
    fig.text(0.055, 0.924, "MOVER 2022 observed subset (n = 2,033)", fontsize=9.5, color=GREY, va="top")
    fig.text(0.055, 0.878, "Same patients, original predictors and fitted predictions; only outcome labels change.", fontsize=8.7, va="top")
    fig.text(0.055, 0.797, "A   Paired label changes", fontsize=10.2, weight="bold")
    fig.text(0.56, 0.797, "B   Paired metric differences", fontsize=10.2, weight="bold")
    ax = fig.add_axes([0.15, 0.255, 0.29, 0.425])
    ax.set_xlim(0, 2)
    ax.set_ylim(2, 0)
    ax.set_axis_off()
    fills = {(0, 0): "#EDF0F2", (0, 1): "#F8E4DB", (1, 0): "#DCEFEA", (1, 1): "#EDF0F2"}
    for a in range(2):
        for b in range(2):
            ax.add_patch(Rectangle((b, a), 1, 1, facecolor=fills[a, b], edgecolor="white", linewidth=2))
            ax.text(b + 0.5, a + 0.44, f"{matrix[a, b]:,}", fontsize=16, weight="bold", ha="center", va="center")
            ax.text(b + 0.5, a + 0.70, "Changed" if a != b else "Unchanged", fontsize=8, ha="center", va="center", color=GREY)
    for j, label in enumerate(["Non-AKI", "AKI"]):
        ax.text(j + 0.5, -0.10, label, ha="center", va="bottom", fontsize=9)
        ax.text(-0.10, j + 0.5, label, ha="right", va="center", fontsize=9)
    fig.text(0.295, 0.735, "Coarsened label", ha="center", fontsize=9)
    fig.text(0.040, 0.467, "Original label", rotation=90, va="center", ha="center", fontsize=9)
    fig.text(0.295, 0.198, f"{int(counts['discordant_n']):,} / {int(counts['observed_n']):,} labels changed ({float(counts['discordant_percent']):.1f}%)", ha="center", fontsize=9.2, weight="bold")

    for metric, label, bottom, limits, ticks, color in [
        ("oe_ratio", "O/E", 0.525, (-0.06, 0.48), [0, 0.2, 0.4], "#0072B2"),
        ("auroc", "AUROC", 0.265, (-0.078, 0.018), [-0.06, -0.03, 0], "#B44E28"),
    ]:
        r = select_one(metrics, metric=metric, contrast="difference")
        original = float(select_one(metrics, metric=metric, contrast="original")["estimate"])
        coarsened = float(select_one(metrics, metric=metric, contrast="coarsened")["estimate"])
        estimate, lo, hi = [float(r[k]) for k in ["estimate", "ci_lower", "ci_upper"]]
        fig.text(0.56, bottom + 0.195, label, fontsize=10, weight="bold", color=color)
        fig.text(0.56, bottom + 0.154, f"Original {original:.3f}   |   Coarsened {coarsened:.3f}", fontsize=8.7, color=GREY)
        fig.text(0.56, bottom + 0.113, f"{estimate:+.3f}  (95% CI {lo:.3f} to {hi:.3f})", fontsize=9.1)
        ax = fig.add_axes([0.56, bottom, 0.375, 0.080])
        ax.set_xlim(*limits)
        ax.set_ylim(-0.6, 0.6)
        ax.axvline(0, color=GREY, ls=(0, (3, 3)), lw=0.8)
        ax.errorbar([estimate], [0], xerr=[[estimate - lo], [hi - estimate]], fmt="o", color=color, markersize=5, lw=1.5, capsize=3)
        ax.set_yticks([])
        ax.set_xticks(ticks)
        ax.spines[["top", "left", "right"]].set_visible(False)
        ax.tick_params(axis="x", length=3, pad=3)
    fig.text(0.7475, 0.194, "Difference (coarsened - original)", fontsize=8.7, ha="center")
    fig.text(0.055, 0.105, "Observed-subset estimates, not full-population AIPW/IPW estimates.", fontsize=8.4)
    fig.text(0.055, 0.066, "95% CIs: 1,000 paired patient bootstraps, no refitting. Label changes do not establish clinical error.", fontsize=8.1)
    save_figure(fig, "Figure_4", out)


def captions(counts, metrics):
    return (
        "Figure 3. Stage-specific MNAR sensitivity.\n"
        "Target observed-to-expected event ratio (O/E; A) and alerts per 1,000 eligible target operations at a 10% predicted-risk threshold (B) are shown for source-development, local-recalibration and target-evaluation sensitivity analyses. Each row varies the missing-not-at-random (MNAR) odds multiplier at one stage (k = 0.5, 0.67, 1, 1.5, 2 or 3; 0.67 denotes 2/3). Points are stage-specific outcome-regression (OR) completion estimates; error bars are 95% percentile intervals from the corresponding 1,000 full-chain patient/operation bootstrap replicates. The shaded vertical reference marks k = 1, the OR-completion reference, not the canonical missing-at-random (MAR) estimator. Dashed horizontal lines show the separate canonical MAR estimates. All panels refer to 2,587 eligible MOVER 2022 operations. Multipliers are sensitivity parameters, not established clinical risk bounds; intervals quantify sampling uncertainty conditional on each scenario, not uncertainty about the true missingness mechanism.\n\n"
        "Figure 4. Creatinine coarsening with fixed predictions.\n"
        "A deterministic creatinine-coarsening stress test in the MOVER 2022 outcome-observed subset (n = 2,033), holding patients, timestamps, follow-up, observation status, original predictor values and fitted predictions fixed. (A) Paired original and coarsened outcome labels: 120 non-AKI labels changed to AKI and 43 AKI labels changed to non-AKI (163/2,033; 8.0%). (B) Coarsened-minus-original differences in observed-to-expected event ratio (O/E), 0.319 (95% CI 0.218 to 0.421), and area under the receiver operating characteristic curve (AUROC), -0.034 (95% CI -0.062 to -0.006). Intervals use 1,000 paired patient bootstrap replicates without refitting. These observed-subset estimates are distinct from full-population augmented inverse-probability-weighted/inverse-probability-weighted (AIPW/IPW) estimates. Label discordance is not an estimate of clinical diagnostic error or patient harm.\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", type=Path, default=TABLES)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    source_paths = [args.tables / item[0] for item in STAGES.values()] + [args.tables / f for f in [INTERVALS, PAIRED, METRICS]]
    before = {str(p): sha256(p) for p in source_paths}
    rows, counts, matrix, metrics = load_inputs(args.tables)
    configure()
    figure_2(rows, args.out)
    figure_3(counts, matrix, metrics, args.out)
    with (args.out / "Figure_3_plot_data.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.out / "figure_captions.txt").write_text(captions(counts, metrics), encoding="utf-8")
    after = {str(p): sha256(p) for p in source_paths}
    require(before == after, "Source file changed during generation")
    manifest = {"source_sha256_before": before, "source_sha256_after": after, "source_files_unchanged": before == after, "figure_2_plotted_metric_points": len(rows), "figure_2_intervals": "stage/k/metric/estimator-matched full-chain bootstrap; 1000 replicates", "figure_3_counts": counts, "figure_3_metrics": metrics, "figure_3_intervals": "separate paired observed-subset bootstrap; no refitting; 1000 replicates", "matplotlib_version": matplotlib.__version__, "figures": {"Figure_3": {"inches": [7.1, 6.35], "png_dpi": 300}, "Figure_4": {"inches": [7.1, 4.85], "png_dpi": 300}}}
    (args.out / "source_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "generated", "out": str(args.out), "source_files_unchanged": before == after, "mnAR_metric_points": len(rows)}))


if __name__ == "__main__":
    main()
