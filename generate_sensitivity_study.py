"""
generate_sensitivity_study.py

Systematic sensitivity analysis for the ASPJ paper.
Varies one parameter at a time while holding others constant,
measuring how coverage metrics respond to each change.

Studies:
  Study 1 — Swarm size sensitivity (5, 10, 20, 30, 50 drones)
  Study 2 — Duration sensitivity (30, 60, 90, 120 seconds)
  Study 3 — Drone type sensitivity (micro, small, medium)

All studies use perimeter_sweep as the baseline pattern since it
was the best performer in the main results.

Outputs:
  results/sensitivity/sensitivity_results.json
  results/sensitivity/sensitivity_results.csv
  paper/figures/fig_sensitivity_swarm_size.png
  paper/figures/fig_sensitivity_duration.png
  paper/figures/fig_sensitivity_drone_type.png

Usage:
    python generate_sensitivity_study.py
    python generate_sensitivity_study.py --study swarm_size
    python generate_sensitivity_study.py --no-sensor
"""

import os
import sys
import json
import csv
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from run_gis_simulation import run_simulation

# ── Baseline parameters ────────────────────────────────────────
BASELINE = {
    "pattern":    "perimeter_sweep",
    "n_drones":   20,
    "duration":   60.0,
    "drone_type": "small",
    "timestep":   0.5,
}

# ── Parameter ranges ───────────────────────────────────────────
SWARM_SIZES  = [5, 10, 20, 30, 50]
DURATIONS    = [30, 60, 90, 120]
DRONE_TYPES  = ["micro", "small", "medium"]

RESULTS_DIR = "results/sensitivity"
FIGURES_DIR = "paper/figures"
DPI         = 300

# Colors
C_VALLEY    = "#1565C0"
C_MIDSLOPE  = "#E65100"
C_ALPINE    = "#6A1B9A"
C_OVERALL   = "#2E7D32"


def run_study_case(label: str, overrides: dict, use_sensor: bool) -> dict:
    """Run one simulation case with baseline + overrides."""
    params = {**BASELINE, **overrides}
    print(f"\n  Running: {label}")

    outputs = run_simulation(
        n_drones      = params["n_drones"],
        pattern       = params["pattern"],
        duration      = params["duration"],
        timestep      = params["timestep"],
        drone_type    = params["drone_type"],
        output_prefix = f"sens_{label.replace(' ', '_')}",
        use_sensor    = use_sensor,
    )

    with open(outputs["report"]) as f:
        report = json.load(f)

    oc = report["overall_coverage"]
    eb = report["elevation_band_coverage"]

    return {
        "label":                  label,
        "n_drones":               params["n_drones"],
        "duration_s":             params["duration"],
        "drone_type":             params["drone_type"],
        "coverage_pct":           oc["coverage_pct"],
        "mean_confidence":        oc["mean_confidence"],
        "gap_zones":              report["coverage_gaps"]["n_gap_zones"],
        "total_gap_km2":          report["coverage_gaps"]["total_gap_km2"],
        "valley_coverage_pct":    eb["valley"]["coverage_pct"],
        "midslope_coverage_pct":  eb["mid_slope"]["coverage_pct"],
        "alpine_coverage_pct":    eb["alpine"]["coverage_pct"],
        "valley_confidence":      eb["valley"]["mean_confidence"],
        "midslope_confidence":    eb["mid_slope"]["mean_confidence"],
        "alpine_confidence":      eb["alpine"]["mean_confidence"],
    }


def study_swarm_size(use_sensor: bool) -> list[dict]:
    """Study 1 — vary swarm size, hold everything else constant."""
    print("\n" + "=" * 55)
    print("Study 1: Swarm Size Sensitivity")
    print(f"Sizes: {SWARM_SIZES} drones")
    print("=" * 55)

    results = []
    for n in SWARM_SIZES:
        r = run_study_case(
            label      = f"{n}drones",
            overrides  = {"n_drones": n},
            use_sensor = use_sensor,
        )
        results.append(r)
        print(f"    Coverage: {r['coverage_pct']}% | "
              f"Confidence: {r['mean_confidence']}")

    return results


def study_duration(use_sensor: bool) -> list[dict]:
    """Study 2 — vary duration, hold everything else constant."""
    print("\n" + "=" * 55)
    print("Study 2: Duration Sensitivity")
    print(f"Durations: {DURATIONS} seconds")
    print("=" * 55)

    results = []
    for d in DURATIONS:
        r = run_study_case(
            label      = f"{d}s",
            overrides  = {"duration": float(d)},
            use_sensor = use_sensor,
        )
        results.append(r)
        print(f"    Coverage: {r['coverage_pct']}% | "
              f"Confidence: {r['mean_confidence']}")

    return results


def study_drone_type(use_sensor: bool) -> list[dict]:
    """Study 3 — vary drone type, hold everything else constant."""
    print("\n" + "=" * 55)
    print("Study 3: Drone Type Sensitivity")
    print(f"Types: {DRONE_TYPES}")
    print("=" * 55)

    results = []
    for dt in DRONE_TYPES:
        r = run_study_case(
            label      = dt,
            overrides  = {"drone_type": dt},
            use_sensor = use_sensor,
        )
        results.append(r)
        print(f"    Coverage: {r['coverage_pct']}% | "
              f"Confidence: {r['mean_confidence']}")

    return results


def plot_swarm_size(results: list[dict]) -> str:
    """Figure: coverage vs swarm size."""
    sizes    = [r["n_drones"] for r in results]
    overall  = [r["coverage_pct"] for r in results]
    valley   = [r["valley_coverage_pct"] for r in results]
    midslope = [r["midslope_coverage_pct"] for r in results]
    alpine   = [r["alpine_coverage_pct"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: overall coverage vs swarm size
    ax1.plot(sizes, overall, "o-", color=C_OVERALL, linewidth=2,
             markersize=8, label="Overall", zorder=5)
    ax1.fill_between(sizes, overall, alpha=0.1, color=C_OVERALL)
    ax1.set_xlabel("Number of drones", fontsize=12)
    ax1.set_ylabel("Coverage (%)", fontsize=12)
    ax1.set_title("Overall Coverage vs Swarm Size", fontsize=12,
                  fontweight="bold")
    ax1.grid(alpha=0.3)
    ax1.set_xticks(sizes)

    # Annotate each point
    for x, y in zip(sizes, overall):
        ax1.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=9)

    # Right: elevation band breakdown
    ax2.plot(sizes, valley,   "o-", color=C_VALLEY,   linewidth=2,
             markersize=7, label="Valley (1375–1800m)")
    ax2.plot(sizes, midslope, "s-", color=C_MIDSLOPE,  linewidth=2,
             markersize=7, label="Mid-slope (1800–2600m)")
    ax2.plot(sizes, alpine,   "^-", color=C_ALPINE,   linewidth=2,
             markersize=7, label="Alpine (2600–3500m)")
    ax2.set_xlabel("Number of drones", fontsize=12)
    ax2.set_ylabel("Coverage (%)", fontsize=12)
    ax2.set_title("Elevation Band Coverage vs Swarm Size",
                  fontsize=12, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    ax2.set_xticks(sizes)

    plt.suptitle(
        "Sensitivity Study 1: Swarm Size\n"
        f"Pattern: perimeter_sweep | Duration: {BASELINE['duration']}s | "
        f"Drone: {BASELINE['drone_type']}",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig_sensitivity_swarm_size.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")
    return path


def plot_duration(results: list[dict]) -> str:
    """Figure: coverage vs simulation duration."""
    durations = [r["duration_s"] for r in results]
    overall   = [r["coverage_pct"] for r in results]
    valley    = [r["valley_coverage_pct"] for r in results]
    midslope  = [r["midslope_coverage_pct"] for r in results]
    alpine    = [r["alpine_coverage_pct"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(durations, overall, "o-", color=C_OVERALL, linewidth=2,
             markersize=8)
    ax1.fill_between(durations, overall, alpha=0.1, color=C_OVERALL)
    ax1.set_xlabel("Simulation duration (seconds)", fontsize=12)
    ax1.set_ylabel("Coverage (%)", fontsize=12)
    ax1.set_title("Overall Coverage vs Duration", fontsize=12,
                  fontweight="bold")
    ax1.grid(alpha=0.3)
    ax1.set_xticks(durations)
    for x, y in zip(durations, overall):
        ax1.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=9)

    ax2.plot(durations, valley,   "o-", color=C_VALLEY,   linewidth=2,
             markersize=7, label="Valley (1375–1800m)")
    ax2.plot(durations, midslope, "s-", color=C_MIDSLOPE,  linewidth=2,
             markersize=7, label="Mid-slope (1800–2600m)")
    ax2.plot(durations, alpine,   "^-", color=C_ALPINE,   linewidth=2,
             markersize=7, label="Alpine (2600–3500m)")
    ax2.set_xlabel("Simulation duration (seconds)", fontsize=12)
    ax2.set_ylabel("Coverage (%)", fontsize=12)
    ax2.set_title("Elevation Band Coverage vs Duration",
                  fontsize=12, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    ax2.set_xticks(durations)

    plt.suptitle(
        "Sensitivity Study 2: Simulation Duration\n"
        f"Pattern: perimeter_sweep | Drones: {BASELINE['n_drones']} | "
        f"Drone: {BASELINE['drone_type']}",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig_sensitivity_duration.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")
    return path


def plot_drone_type(results: list[dict]) -> str:
    """Figure: coverage vs drone type — bar chart."""
    types    = [r["drone_type"] for r in results]
    bands    = ["valley_coverage_pct", "midslope_coverage_pct",
                "alpine_coverage_pct"]
    labels   = ["Valley\n(1375–1800m)", "Mid-slope\n(1800–2600m)",
                "Alpine\n(2600–3500m)"]
    colors   = [C_VALLEY, C_MIDSLOPE, C_ALPINE]

    x     = np.arange(len(types))
    width = 0.25

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: grouped bar chart by elevation band
    for i, (band, label, color) in enumerate(zip(bands, labels, colors)):
        vals = [r[band] for r in results]
        bars = ax1.bar(x + i * width, vals, width, label=label,
                       color=color, alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, h + 0.1,
                     f"{h:.1f}%", ha="center", va="bottom", fontsize=8)

    ax1.set_xlabel("Drone type", fontsize=12)
    ax1.set_ylabel("Coverage (%)", fontsize=12)
    ax1.set_title("Coverage by Elevation Band and Drone Type",
                  fontsize=12, fontweight="bold")
    ax1.set_xticks(x + width)
    ax1.set_xticklabels([t.title() for t in types], fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(axis="y", alpha=0.3)

    # Right: overall coverage + confidence
    overall    = [r["coverage_pct"] for r in results]
    confidence = [r["mean_confidence"] for r in results]

    ax2b = ax2.twinx()
    bars = ax2.bar(x, overall, 0.4, color=C_OVERALL, alpha=0.75,
                   label="Overall coverage (%)")
    ax2b.plot(x, confidence, "D--", color="#C62828", linewidth=2,
              markersize=9, label="Mean confidence")

    ax2.set_xlabel("Drone type", fontsize=12)
    ax2.set_ylabel("Overall coverage (%)", fontsize=12, color=C_OVERALL)
    ax2b.set_ylabel("Mean confidence", fontsize=12, color="#C62828")
    ax2.set_title("Overall Coverage and Confidence by Drone Type",
                  fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([t.title() for t in types], fontsize=11)
    ax2.legend(loc="upper left", fontsize=10)
    ax2b.legend(loc="upper right", fontsize=10)
    ax2.grid(axis="y", alpha=0.3)

    plt.suptitle(
        "Sensitivity Study 3: Drone Type\n"
        f"Pattern: perimeter_sweep | Drones: {BASELINE['n_drones']} | "
        f"Duration: {BASELINE['duration']}s",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig_sensitivity_drone_type.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")
    return path


def save_results(all_results: dict, timestamp: str):
    """Save all sensitivity results as JSON and CSV."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    json_path = os.path.join(RESULTS_DIR,
                             f"sensitivity_results_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved JSON → {json_path}")

    csv_path = os.path.join(RESULTS_DIR,
                            f"sensitivity_results_{timestamp}.csv")
    all_rows = []
    for study_name, rows in all_results.items():
        for row in rows:
            all_rows.append({"study": study_name, **row})

    if all_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_rows)
    print(f"Saved CSV  → {csv_path}")

    return json_path, csv_path


def print_sensitivity_summary(all_results: dict):
    """Print key findings to console."""
    print("\n" + "=" * 60)
    print("SENSITIVITY STUDY SUMMARY")
    print("=" * 60)

    if "swarm_size" in all_results:
        print("\nStudy 1 — Swarm Size:")
        print(f"  {'Drones':<10} {'Coverage':>10} {'Valley':>8} "
              f"{'Mid':>8} {'Alpine':>8}")
        for r in all_results["swarm_size"]:
            print(f"  {r['n_drones']:<10} {r['coverage_pct']:>9.2f}% "
                  f"{r['valley_coverage_pct']:>7.2f}% "
                  f"{r['midslope_coverage_pct']:>7.2f}% "
                  f"{r['alpine_coverage_pct']:>7.2f}%")

    if "duration" in all_results:
        print("\nStudy 2 — Duration:")
        print(f"  {'Duration':>10} {'Coverage':>10} {'Valley':>8} "
              f"{'Mid':>8} {'Alpine':>8}")
        for r in all_results["duration"]:
            print(f"  {r['duration_s']:>9.0f}s {r['coverage_pct']:>9.2f}% "
                  f"{r['valley_coverage_pct']:>7.2f}% "
                  f"{r['midslope_coverage_pct']:>7.2f}% "
                  f"{r['alpine_coverage_pct']:>7.2f}%")

    if "drone_type" in all_results:
        print("\nStudy 3 — Drone Type:")
        print(f"  {'Type':<10} {'Coverage':>10} {'Confidence':>12} "
              f"{'Valley':>8} {'Alpine':>8}")
        for r in all_results["drone_type"]:
            print(f"  {r['drone_type']:<10} {r['coverage_pct']:>9.2f}% "
                  f"{r['mean_confidence']:>11.4f} "
                  f"{r['valley_coverage_pct']:>7.2f}% "
                  f"{r['alpine_coverage_pct']:>7.2f}%")

    print("=" * 60)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sensitivity analysis for drone swarm GIS simulation"
    )
    parser.add_argument(
        "--study", type=str, default="all",
        choices=["all", "swarm_size", "duration", "drone_type"],
        help="Which study to run (default: all)"
    )
    parser.add_argument(
        "--no-sensor", action="store_true",
        help="Skip sensor confidence calculation (faster)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args       = parse_args()
    use_sensor = not args.no_sensor
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("=" * 55)
    print("Drone Swarm GIS — Sensitivity Study")
    print(f"Sensor confidence: {'enabled' if use_sensor else 'disabled'}")
    print("=" * 55)

    all_results = {}

    if args.study in ("all", "swarm_size"):
        all_results["swarm_size"] = study_swarm_size(use_sensor)
        plot_swarm_size(all_results["swarm_size"])

    if args.study in ("all", "duration"):
        all_results["duration"] = study_duration(use_sensor)
        plot_duration(all_results["duration"])

    if args.study in ("all", "drone_type"):
        all_results["drone_type"] = study_drone_type(use_sensor)
        plot_drone_type(all_results["drone_type"])

    save_results(all_results, timestamp)
    print_sensitivity_summary(all_results)

    print("\nDone. Figures saved to paper/figures/")
    print("Run generate_paper_figures.py to regenerate main paper figures.")
