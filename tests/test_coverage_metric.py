"""
Regression tests for the coverage metric and real-scale simulation.

These encode properties the original metric violated:
  * coverage must not depend on the sampling timestep,
  * coverage must grow with altitude (wider camera footprint),
  * drones must not pile up against the boundary,
  * a fixed seed must reproduce the same trajectories.

Requires the DEM (git lfs pull); tests are skipped if it is missing.
Run: pytest tests/
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from gis.coordinate_transformer import DEFAULT_DEM_PATH, SIM_WIDTH_M

DEM_OK = os.path.exists(DEFAULT_DEM_PATH) and os.path.getsize(DEFAULT_DEM_PATH) > 1_000_000
pytestmark = pytest.mark.skipif(not DEM_OK, reason="DEM not present (run git lfs pull)")


@pytest.fixture(scope="module")
def ct():
    from gis.coordinate_transformer import CoordinateTransformer
    return CoordinateTransformer()


def _traj(pattern="perimeter_sweep", n=10, duration=900.0, timestep=0.5, seed=1):
    from run_gis_simulation import BCC_SIM_BOUNDS, PATTERN_MAP
    from src.drone_trajectory_generator import DroneSwarmGenerator
    np.random.seed(seed)
    gen = DroneSwarmGenerator(BCC_SIM_BOUNDS)
    return gen.generate_swarm_trajectories(n, PATTERN_MAP[pattern], duration, timestep, "small")["trajectories"]


def _coverage(ct, traj, **kw):
    from gis.footprint_coverage import FootprintCoverage
    fp = FootprintCoverage(ct, **kw)
    fp.add_trajectories(traj)
    return fp


def test_straight_line_matches_analytic_swath(ct):
    t = np.zeros((1, 2, 3))
    t[0, :, 1] = 6000.0
    t[0, 0, 0], t[0, 1, 0] = 4000.0, 14000.0
    t[0, :, 2] = 200.0
    r = 200.0 * np.tan(np.radians(30))
    expected = (10000.0 * 2 * r + np.pi * r * r) / 1e6
    assert _coverage(ct, t).imaged_area_km2() == pytest.approx(expected, rel=0.05)


def test_coverage_independent_of_timestep(ct):
    a = _coverage(ct, _traj(timestep=1.0)).coverage_pct()
    b = _coverage(ct, _traj(timestep=0.25)).coverage_pct()
    assert abs(a - b) / b < 0.03


def test_coverage_increases_with_altitude(ct):
    base = _traj()
    covs = []
    for z in (100.0, 200.0, 400.0):
        t = base.copy(); t[..., 2] = z
        covs.append(_coverage(ct, t).coverage_pct())
    assert covs[0] < covs[1] < covs[2]


def test_msl_mode_depends_on_terrain(ct):
    t = _traj(); t[..., 2] = 300.0
    agl = _coverage(ct, t, altitude_mode="agl").coverage_pct()
    msl = _coverage(ct, t, altitude_mode="msl", max_mapping_agl=1e9).coverage_pct()
    assert msl > agl  # flying above the highest peak => larger footprints elsewhere


def test_no_boundary_pileup():
    x = _traj(duration=3600.0)[..., 0]
    at_edge = np.mean((x < 15) | (x > SIM_WIDTH_M - 15))
    assert at_edge < 0.01


def test_seed_reproducible():
    for p in ("perimeter_sweep", "evasive_maneuvers", "random_dispersal"):
        assert np.array_equal(_traj(p, seed=3), _traj(p, seed=3))


def test_no_altitude_creep():
    # Wind is applied horizontally only; altitude-holding patterns stay level
    z = _traj(duration=1800.0)[..., 2]
    assert abs(z[:, -1].mean() - z[:, 0].mean()) < 10.0


def test_empty_map_is_one_gap_zone(ct, tmp_path):
    from gis.sensor_heatmap import SensorHeatmap
    from gis.coverage_analyzer import CoverageAnalyzer
    hm = SensorHeatmap(ct, output_dir=str(tmp_path))
    gaps = CoverageAnalyzer(ct, hm, output_dir=str(tmp_path)).get_coverage_gaps()
    assert len(gaps) == 1
    assert gaps.n_cells.iloc[0] == hm.n_rows * hm.n_cols


def test_gap_zones_match_connected_components(ct, tmp_path):
    from scipy import ndimage
    from gis.sensor_heatmap import SensorHeatmap
    from gis.coverage_analyzer import CoverageAnalyzer
    from gis.footprint_coverage import FootprintCoverage
    fp = FootprintCoverage(ct)
    fp.add_trajectories(_traj())
    an = CoverageAnalyzer(ct, SensorHeatmap(ct, output_dir=str(tmp_path)),
                          output_dir=str(tmp_path), footprint=fp)
    labels, n = ndimage.label(an._coverage_raster() == 0)
    sizes = np.bincount(labels.ravel())[1:]
    assert len(an.get_coverage_gaps()) == int((sizes >= 4).sum())
