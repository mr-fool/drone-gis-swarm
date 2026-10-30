# Results

All files in this folder come from one canonical run of the current code
(seed 42), generated on 2026-10-29. Each file name carries the time it was
written; when a folder holds more than one file of the same kind, the scripts
use the newest.

## How to reproduce

From the repository root, with the DEM in `data/raw/` (`git lfs pull`):

```
python generate_paper_results.py --seed 42
python generate_sensitivity_study.py --seed 42
python generate_paper_figures.py
python generate_replicate_study.py
```

Same seed, same numbers: the outputs are identical on every run and machine
tested (Linux and Windows). Check with perimeter sweep: 39.26% coverage in
`paper/`, and 38.60% [38.18, 39.03] in the replicate summary.

## What is here

| Folder | Produced by | Contents |
|---|---|---|
| `paper/` | `generate_paper_results.py --seed 42` | One run per flight pattern (20 drones, 30 min, small drones, west-to-east). Table used for the single-run figures. |
| `sensitivity/` | `generate_sensitivity_study.py --seed 42` | One run per swarm size, mission duration and drone type (perimeter sweep). |
| `replicates/` | `generate_replicate_study.py` | 32 runs per pattern and per sensitivity condition (seeds 1000-1031, start orientation cycling through all 8). Means with 95% confidence intervals, paired pattern comparisons, elevation-band tests. **Use these for the paper's quantitative claims.** |
| `simulation_outputs/` | the two single-run scripts | One report JSON per run, with the full run configuration under `run_config` (seed, orientation, camera, altitude mode). Rasters, GeoJSON and maps are written to subfolders that are not committed (see `.gitignore`); they regenerate with the commands above. |

## Metrics

* `coverage_pct`: share of the area (DEM pixels) imaged by any drone camera.
* `median_gsd_cm`: median ground sample distance of the imaged ground, in cm
  per image pixel (lower is finer).
* `quality_coverage_pct`: share of the area imaged at 15 cm/pixel or finer.
* `valley_pct`, `mid_slope_pct`, `alpine_pct`: coverage within each
  elevation band (1375-1800, 1800-2600, 2600-3500 m).
* `gap_zones`: connected uncovered areas on the 0.001° grid. The count
  depends on the minimum gap size, so it is descriptive only.

The single-run tables describe one seed and one start orientation; the
replicate study averages over seeds and orientations and is the basis for
statements about which pattern performs better.
