"""
generate_paper_results.py

One-click script that runs all flight patterns and generates
a complete results dataset for the ASPJ paper.

Runs all 5 patterns with consistent parameters, saves all outputs,
and produces a summary comparison table as JSON and CSV.

Usage:
    python generate_paper_results.py
"""

import os
import sys
import json
import csv
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from run_gis_simulation import run_simulation

# Consistent parameters across all runs for fair comparison
N_DRONES   = 20
DURATION   = 60.0
TIMESTEP   = 0.5
DRONE_TYPE = "small"

PATTERNS = [
    "perimeter_sweep",
    "formation_flying",
    "random_dispersal",
    "evasive_maneuvers",
    "coordinated_attack",
]

RESULTS_DIR = "results/paper"


def run_all_patterns() -> list[dict]:
    """Run simulation for all patterns and collect results."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = []

    print("=" * 60)
    print("Generating all paper results")
    print(f"Drones: {N_DRONES} | Duration: {DURATION}s | Type: {DRONE_TYPE}")
    print("=" * 60)

    for i, pattern in enumerate(PATTERNS):
        print(f"\n[{i+1}/{len(PATTERNS)}] Running {pattern}...")
        print("-" * 40)

        outputs = run_simulation(
            n_drones      = N_DRONES,
            pattern       = pattern,
            duration      = DURATION,
            timestep      = TIMESTEP,
            drone_type    = DRONE_TYPE,
            output_prefix = f"paper_{pattern}",
        )

        with open(outputs["report"]) as f:
            report = json.load(f)

        oc = report["overall_coverage"]
        eb = report["elevation_band_coverage"]

        all_results.append({
            "pattern":             pattern,
            "n_drones":            N_DRONES,
            "duration_s":          DURATION,
            "n_positions":         outputs["n_positions"],
            "coverage_pct":        oc["coverage_pct"],
            "mean_confidence":     oc["mean_confidence"],
            "covered_cells":       oc["covered_cells"],
            "gap_cells":           oc["gap_cells"],
            "gap_pct":             oc["gap_pct"],
            "gap_zones":           report["coverage_gaps"]["n_gap_zones"],
            "total_gap_km2":       report["coverage_gaps"]["total_gap_km2"],
            "valley_coverage_pct":    eb["valley"]["coverage_pct"],
            "midslope_coverage_pct":  eb["mid_slope"]["coverage_pct"],
            "alpine_coverage_pct":    eb["alpine"]["coverage_pct"],
            "valley_confidence":      eb["valley"]["mean_confidence"],
            "midslope_confidence":    eb["mid_slope"]["mean_confidence"],
            "alpine_confidence":      eb["alpine"]["mean_confidence"],
            "map_path":            outputs["map"],
            "report_path":         outputs["report"],
        })

    return all_results, timestamp


def save_comparison_table(results: list[dict], timestamp: str):
    """Save results as JSON and CSV comparison tables."""

    # JSON
    json_path = os.path.join(RESULTS_DIR, f"comparison_table_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved JSON table → {json_path}")

    # CSV
    csv_path = os.path.join(RESULTS_DIR, f"comparison_table_{timestamp}.csv")
    if results:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames = [k for k in results[0].keys()
                              if k not in ("map_path", "report_path")]
            )
            writer.writeheader()
            for row in results:
                clean = {k: v for k, v in row.items()
                         if k not in ("map_path", "report_path")}
                writer.writerow(clean)
    print(f"Saved CSV table  → {csv_path}")

    return json_path, csv_path


def print_summary_table(results: list[dict]):
    """Print a formatted summary table to console."""
    print("\n" + "=" * 75)
    print("PAPER RESULTS SUMMARY — All Flight Patterns")
    print(f"{'Pattern':<22} {'Coverage':>10} {'Valley':>8} "
          f"{'Mid-slope':>10} {'Alpine':>8} {'Gaps':>6}")
    print("-" * 75)
    for r in results:
        print(
            f"{r['pattern']:<22} "
            f"{r['coverage_pct']:>9.2f}% "
            f"{r['valley_coverage_pct']:>7.2f}% "
            f"{r['midslope_coverage_pct']:>9.2f}% "
            f"{r['alpine_coverage_pct']:>7.2f}% "
            f"{r['gap_zones']:>6}"
        )
    print("=" * 75)


if __name__ == "__main__":
    results, timestamp = run_all_patterns()
    save_comparison_table(results, timestamp)
    print_summary_table(results)

    print("\nAll paper results generated.")
    print(f"Results saved to: {RESULTS_DIR}/")
    print("Run generate_paper_figures.py next to create figures.")
