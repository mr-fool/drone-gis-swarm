"""
generate_replicate_study.py

Replicated simulations with statistics, for the pattern comparison and the
sensitivity study.

Design
------
* Every condition is run N times (default 32). Run r uses seed
  base_seed + r and mission orientation r % 8, so each of the 8 orientations
  (start edge / heading, see src/mission_orientation.py) appears equally
  often. Results therefore describe a mission flown from a random edge in a
  random direction, not one fixed geometry.
* Every condition uses the SAME list of (seed, orientation) pairs (common
  random numbers), so conditions can be compared run-by-run with paired
  tests.
* Each run is one independent sample. Grid cells within a run are never
  treated as samples.

Reported
--------
* Per condition and metric: mean, SD, n, 95% t confidence interval.
* Pattern comparison: every pair of patterns, paired mean difference with
  95% CI, paired t-test and Wilcoxon signed-rank test, Holm-adjusted
  p-values (per metric).
* Elevation bands: within each pattern, paired differences between bands
  (alpine - valley, mid - valley, alpine - mid), Holm-adjusted across all
  band tests.
* Per-orientation means, to show how much results depend on the start edge.
* Sensitivity (perimeter sweep): the same summaries for swarm size,
  duration and drone type, plus paired differences between adjacent levels.

Coverage is the camera-footprint area metric (gis/footprint_coverage.py),
terrain-following ("agl") mode, 60-degree FOV. Mapping quality is the
median ground sample distance (GSD, cm/pixel) of the imaged ground.

Usage
-----
    python generate_replicate_study.py                    # everything, 32 runs/condition
    python generate_replicate_study.py --study patterns
    python generate_replicate_study.py --study sensitivity --runs 64
    python generate_replicate_study.py --workers 4
"""

import argparse
import io
import contextlib
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.mission_orientation import N_ORIENTATIONS, ORIENTATION_NAMES

RESULTS_DIR = "results/replicates"
FIGURES_DIR = "paper/figures"
DPI = 300

PATTERNS = ["perimeter_sweep", "formation_flying", "random_dispersal",
            "evasive_maneuvers", "coordinated_attack"]
PATTERN_NAMES = {
    "perimeter_sweep":    "Perimeter Sweep",
    "formation_flying":   "Formation Traverse",
    "random_dispersal":   "Stochastic Dispersal",
    "evasive_maneuvers":  "Sinusoidal Dispersal",
    "coordinated_attack": "Centripetal Convergence",
}
BASELINE = {"pattern": "perimeter_sweep", "n_drones": 20, "duration": 1800.0,
            "drone_type": "small", "timestep": 0.5}
SWARM_SIZES = [5, 10, 20, 30, 50]
DURATIONS   = [600, 1200, 1800, 2400]
DRONE_TYPES = ["micro", "small", "medium"]

METRICS = ["coverage_pct", "valley_pct", "mid_slope_pct", "alpine_pct",
           "gap_zones", "imaged_km2", "median_gsd_cm", "quality_coverage_pct"]
BANDS = ["valley_pct", "mid_slope_pct", "alpine_pct"]

C_VALLEY, C_MIDSLOPE, C_ALPINE, C_OVERALL = "#1565C0", "#E65100", "#6A1B9A", "#2E7D32"


# ── worker ──────────────────────────────────────────────────────
_CT = None


def _init_worker():
    global _CT
    from gis.coordinate_transformer import CoordinateTransformer
    with contextlib.redirect_stdout(io.StringIO()):
        _CT = CoordinateTransformer()


def run_case(task: dict) -> dict:
    """One simulation: trajectories -> footprint coverage -> metrics."""
    from run_gis_simulation import generate_trajectories
    from gis.footprint_coverage import FootprintCoverage
    from gis.sensor_heatmap import SensorHeatmap
    from gis.coverage_analyzer import CoverageAnalyzer, ELEVATION_BANDS

    np.random.seed(task["seed"])
    swarm = generate_trajectories(task["n_drones"], task["pattern"], task["duration"],
                                  task["timestep"], task["drone_type"], task["orientation"])
    with contextlib.redirect_stdout(io.StringIO()):
        fp = FootprintCoverage(_CT)
        fp.add_trajectories(swarm["trajectories"])
        hm = SensorHeatmap(_CT, output_dir=os.path.join(RESULTS_DIR, "_tmp"))
        an = CoverageAnalyzer(_CT, hm, output_dir=os.path.join(RESULTS_DIR, "_tmp"),
                              footprint=fp)
        bands = fp.band_coverage(ELEVATION_BANDS)
        gaps = an.count_gap_zones()
    return {
        **task,
        "coverage_pct":  fp.coverage_pct(),
        "imaged_km2":    fp.imaged_area_km2(),
        "valley_pct":    bands["valley"]["coverage_pct"],
        "mid_slope_pct": bands["mid_slope"]["coverage_pct"],
        "alpine_pct":    bands["alpine"]["coverage_pct"],
        "gap_zones":     gaps,
        **{k: fp.gsd_stats()[k] for k in ("median_gsd_cm", "quality_coverage_pct")},
    }


# ── design ──────────────────────────────────────────────────────
def build_tasks(study: str, n_runs: int, base_seed: int) -> list[dict]:
    design = [(base_seed + r, r % N_ORIENTATIONS) for r in range(n_runs)]
    conds = []
    if study in ("all", "patterns"):
        for p in PATTERNS:
            conds.append(("patterns", PATTERN_NAMES[p], {**BASELINE, "pattern": p}))
    if study in ("all", "sensitivity"):
        for n in SWARM_SIZES:
            conds.append(("swarm_size", f"{n} drones", {**BASELINE, "n_drones": n}))
        for d in DURATIONS:
            conds.append(("duration", f"{d} s", {**BASELINE, "duration": float(d)}))
        for t in DRONE_TYPES:
            conds.append(("drone_type", t, {**BASELINE, "drone_type": t}))
    tasks = []
    for study_name, label, params in conds:
        for r, (seed, orient) in enumerate(design):
            tasks.append({"study": study_name, "condition": label, "run": r,
                          "seed": seed, "orientation": orient, **params})
    return tasks


# ── statistics ──────────────────────────────────────────────────
def ci95(x: np.ndarray):
    x = np.asarray(x, dtype=float)
    n = len(x); m = x.mean(); sd = x.std(ddof=1) if n > 1 else 0.0
    half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n) if n > 1 else np.nan
    return m, sd, n, m - half, m + half


def holm(pvals) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p); m = len(p)
    adj = np.empty(m); running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def paired_test(a: np.ndarray, b: np.ndarray) -> dict:
    d = np.asarray(a, float) - np.asarray(b, float)
    m, sd, n, lo, hi = ci95(d)
    t_p = stats.ttest_rel(a, b).pvalue if sd > 0 else (0.0 if m != 0 else 1.0)
    try:
        w_p = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    except ValueError:
        w_p = np.nan
    return {"mean_diff": m, "ci_low": lo, "ci_high": hi, "n": n,
            "t_p": t_p, "wilcoxon_p": w_p,
            "a_greater_in_runs": int(np.sum(d > 0)), "b_greater_in_runs": int(np.sum(d < 0))}


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (study, cond), g in runs.groupby(["study", "condition"], sort=False):
        for met in METRICS:
            m, sd, n, lo, hi = ci95(g[met].values)
            rows.append({"study": study, "condition": cond, "metric": met,
                         "mean": m, "sd": sd, "n": n, "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def pattern_comparisons(runs: pd.DataFrame) -> pd.DataFrame:
    pr = runs[runs.study == "patterns"]
    if pr.empty:
        return pd.DataFrame()
    rows = []
    for met in METRICS:
        wide = pr.pivot(index="run", columns="condition", values=met)
        block = []
        for a, b in combinations(wide.columns, 2):
            block.append({"metric": met, "a": a, "b": b, **paired_test(wide[a].values, wide[b].values)})
        adj_t = holm([r["t_p"] for r in block]); adj_w = holm([r["wilcoxon_p"] for r in block])
        for r, at, aw in zip(block, adj_t, adj_w):
            r["t_p_holm"] = at; r["wilcoxon_p_holm"] = aw
        rows += block
    return pd.DataFrame(rows)


def band_tests(runs: pd.DataFrame) -> pd.DataFrame:
    pr = runs[runs.study == "patterns"]
    rows = []
    for cond, g in pr.groupby("condition", sort=False):
        for a, b in [("alpine_pct", "valley_pct"), ("mid_slope_pct", "valley_pct"),
                     ("alpine_pct", "mid_slope_pct")]:
            rows.append({"pattern": cond, "a": a, "b": b, **paired_test(g[a].values, g[b].values)})
    df = pd.DataFrame(rows)
    if not df.empty:
        df["t_p_holm"] = holm(df["t_p"]); df["wilcoxon_p_holm"] = holm(df["wilcoxon_p"])
    return df


def sensitivity_steps(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for study in ["swarm_size", "duration", "drone_type"]:
        sr = runs[runs.study == study]
        if sr.empty:
            continue
        levels = list(dict.fromkeys(sr.condition))
        wide = sr.pivot(index="run", columns="condition", values="coverage_pct")
        block = [{"study": study, "from": a, "to": b,
                  **paired_test(wide[b].values, wide[a].values)}
                 for a, b in zip(levels[:-1], levels[1:])]
        for r, at in zip(block, holm([r["t_p"] for r in block])):
            r["t_p_holm"] = at
        rows += block
    return pd.DataFrame(rows)


def orientation_table(runs: pd.DataFrame) -> pd.DataFrame:
    pr = runs[runs.study == "patterns"].copy()
    if pr.empty:
        return pd.DataFrame()
    pr["orientation_name"] = pr.orientation.map(ORIENTATION_NAMES)
    return (pr.groupby(["condition", "orientation", "orientation_name"], sort=False)
              [["coverage_pct"] + BANDS].mean().round(2).reset_index())


# ── figures ─────────────────────────────────────────────────────
def _err(summary, study, metric, conds):
    s = summary[(summary.study == study) & (summary.metric == metric)].set_index("condition").loc[conds]
    return s["mean"].values, np.vstack([s["mean"] - s["ci_low"], s["ci_high"] - s["mean"]])


def plot_patterns(summary: pd.DataFrame, n_runs: int) -> str:
    conds = [PATTERN_NAMES[p] for p in PATTERNS]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    m, e = _err(summary, "patterns", "coverage_pct", conds)
    ax1.bar(range(len(conds)), m, yerr=e, capsize=5, color=C_OVERALL, alpha=0.85)
    ax1.set_xticks(range(len(conds))); ax1.set_xticklabels(conds, rotation=20, ha="right")
    ax1.set_ylabel("Area imaged (%)"); ax1.set_title("Overall coverage", fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)
    w = 0.26
    for i, (band, lab, col) in enumerate(zip(BANDS, ["Valley", "Mid-slope", "Alpine"],
                                             [C_VALLEY, C_MIDSLOPE, C_ALPINE])):
        m, e = _err(summary, "patterns", band, conds)
        ax2.bar(np.arange(len(conds)) + (i - 1) * w, m, w, yerr=e, capsize=3,
                color=col, alpha=0.85, label=lab)
    ax2.set_xticks(range(len(conds))); ax2.set_xticklabels(conds, rotation=20, ha="right")
    ax2.set_ylabel("Area imaged (%)"); ax2.set_title("Coverage by elevation band", fontweight="bold")
    ax2.legend(); ax2.grid(axis="y", alpha=0.3)
    plt.suptitle(f"Flight pattern comparison — mean and 95% CI over {n_runs} runs "
                 f"(random seed and start orientation)\n20 drones, 30 min, small drones",
                 fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig_replicates_patterns.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight"); plt.close()
    return path


def plot_sensitivity(summary: pd.DataFrame, n_runs: int) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    specs = [("swarm_size", [f"{n} drones" for n in SWARM_SIZES], SWARM_SIZES, "Number of drones"),
             ("duration", [f"{d} s" for d in DURATIONS], [d / 60 for d in DURATIONS], "Mission duration (min)"),
             ("drone_type", DRONE_TYPES, [0, 1, 2], "Drone type")]
    for ax, (study, conds, xs, xlabel) in zip(axes, specs):
        if summary[summary.study == study].empty:
            ax.axis("off"); continue
        for met, lab, col, mk in [("coverage_pct", "Overall", C_OVERALL, "o"),
                                  ("valley_pct", "Valley", C_VALLEY, "s"),
                                  ("mid_slope_pct", "Mid-slope", C_MIDSLOPE, "^"),
                                  ("alpine_pct", "Alpine", C_ALPINE, "D")]:
            m, e = _err(summary, study, met, conds)
            ax.errorbar(xs, m, yerr=e, marker=mk, capsize=4, color=col, label=lab,
                        linewidth=2.2 if met == "coverage_pct" else 1.4)
        ax.set_xlabel(xlabel); ax.set_ylabel("Area imaged (%)"); ax.grid(alpha=0.3)
        if study == "drone_type":
            ax.set_xticks(xs); ax.set_xticklabels([c.title() for c in conds])
        else:
            ax.set_xticks(xs)
    axes[0].legend()
    plt.suptitle(f"Sensitivity (perimeter sweep) — mean and 95% CI over {n_runs} runs "
                 f"(random seed and start orientation)", fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig_replicates_sensitivity.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight"); plt.close()
    return path


# ── main ────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--study", choices=["all", "patterns", "sensitivity"], default="all")
    ap.add_argument("--runs", type=int, default=32,
                    help="runs per condition (use a multiple of 8 to balance orientations)")
    ap.add_argument("--base-seed", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True); os.makedirs(FIGURES_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tasks = build_tasks(args.study, args.runs, args.base_seed)
    print(f"Replicate study: {len(tasks)} simulations "
          f"({args.runs} runs per condition, {args.workers} workers)")

    results = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as ex:
        for k, res in enumerate(ex.map(run_case, tasks, chunksize=1), 1):
            results.append(res)
            if k % 20 == 0 or k == len(tasks):
                print(f"  {k}/{len(tasks)} done", flush=True)

    runs = pd.DataFrame(results)
    summary = summarize(runs)
    out = {
        "runs": runs, "summary": summary,
        "pattern_comparisons": pattern_comparisons(runs),
        "band_tests": band_tests(runs),
        "sensitivity_steps": sensitivity_steps(runs),
        "orientation_means": orientation_table(runs),
    }
    for name, df in out.items():
        if not df.empty:
            path = os.path.join(RESULTS_DIR, f"{name}_{ts}.csv")
            df.to_csv(path, index=False, float_format="%.6g")
            print(f"Saved {path}")

    if not summary[summary.study == "patterns"].empty:
        print(f"Saved {plot_patterns(summary, args.runs)}")
    if not summary[summary.study != "patterns"].empty:
        print(f"Saved {plot_sensitivity(summary, args.runs)}")

    view = summary[summary.metric.isin(["coverage_pct"] + BANDS)].copy()
    view["value"] = view.apply(lambda r: f"{r['mean']:.2f} [{r.ci_low:.2f}, {r.ci_high:.2f}]", axis=1)
    print("\nMean [95% CI] by condition:")
    print(view.pivot_table(index=["study", "condition"], columns="metric", values="value",
                           aggfunc="first", sort=False)
              [["coverage_pct"] + BANDS].to_string())


if __name__ == "__main__":
    main()
