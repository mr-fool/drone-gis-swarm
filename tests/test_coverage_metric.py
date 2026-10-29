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


@pytest.mark.parametrize("orientation", range(8))
def test_orientations_stay_in_bounds(orientation):
    from run_gis_simulation import generate_trajectories, BCC_SIM_BOUNDS as B
    for p in ("perimeter_sweep", "formation_flying", "coordinated_attack"):
        np.random.seed(0)
        t = generate_trajectories(10, p, 600.0, 0.5, "small", orientation)["trajectories"]
        assert (t[..., 0] >= B.x_min - 1).all() and (t[..., 0] <= B.x_max + 1).all()
        assert (t[..., 1] >= B.y_min - 1).all() and (t[..., 1] <= B.y_max + 1).all()


def test_orientation_zero_is_original_frame():
    from run_gis_simulation import generate_trajectories
    np.random.seed(5)
    a = generate_trajectories(10, "perimeter_sweep", 300.0, 0.5, "small", 0)["trajectories"]
    assert np.array_equal(a, _traj(n=10, duration=300.0, seed=5))


def test_orientation_sets_start_edge():
    from run_gis_simulation import generate_trajectories, BCC_SIM_BOUNDS as B
    starts = {}
    for o in (0, 1, 4, 5):
        np.random.seed(0)
        starts[o] = generate_trajectories(10, "perimeter_sweep", 10.0, 0.5, "small", o)["trajectories"][:, 0, :2].mean(0)
    assert starts[0][0] < 100 and starts[1][0] > B.x_max - 100      # west / east edge
    assert starts[4][1] < 100 and starts[5][1] > B.y_max - 100      # south / north edge


def test_wind_fixed_in_world_frame():
    from src.mission_orientation import world_to_canonical_vector, canonical_to_world
    from run_gis_simulation import BCC_SIM_BOUNDS as B
    w = np.array([2.0, 1.0, 0.0])
    for o in range(8):
        c = world_to_canonical_vector(w, o)
        p0 = np.array([[5000.0, 4000.0, 0.0]])
        d = canonical_to_world(p0 + c, B, o) - canonical_to_world(p0, B, o)
        assert np.allclose(d[0], w)


def test_wind_does_not_compound():
    # Wind contributes (0.2, 0.1) m/s; random dispersal's momentum term must
    # not amplify it into a systematic drift
    disp = []
    for s in range(6):
        t = _traj("random_dispersal", n=20, duration=600.0, seed=100 + s)
        disp.append((t[:, -1, :2] - t[:, 0, :2]).mean(0))
    drift = np.mean(disp, axis=0) / 600.0
    assert abs(drift[0]) < 1.0 and abs(drift[1]) < 1.0


def test_gsd_matches_camera_geometry(ct):
    from gis.footprint_coverage import FootprintCoverage
    t = np.zeros((1, 2, 3))
    t[0, :, 1] = 6000.0
    t[0, 0, 0], t[0, 1, 0] = 4000.0, 14000.0
    t[0, :, 2] = 200.0
    fp = _coverage(ct, t)
    nadir_cm = 2 * 200.0 * np.tan(np.radians(30)) / 1920 * 100      # 12.03 cm/px
    g = fp.best_gsd[fp.imaged] * 100
    assert g.min() == pytest.approx(nadir_cm, rel=0.02)
    assert g.max() <= nadir_cm * (1 + np.tan(np.radians(30)) ** 2) * 1.02   # edge of image
    # Pixels imaged across the swath, uniformly: mean of 1 + u^2 tan^2(30), u in [0, 1]
    assert fp.gsd_stats()["mean_gsd_cm"] == pytest.approx(nadir_cm * (1 + np.tan(np.radians(30)) ** 2 / 3), rel=0.05)


def test_gsd_scales_with_altitude(ct):
    base = _traj()
    med = []
    for z in (100.0, 200.0):
        t = base.copy(); t[..., 2] = z
        med.append(_coverage(ct, t).gsd_stats()["median_gsd_cm"])
    assert med[1] == pytest.approx(2 * med[0], rel=0.05)


def test_quality_coverage_threshold(ct):
    t = _traj(); t[..., 2] = 200.0          # nadir 12 cm, edge 16 cm
    loose  = _coverage(ct, t, gsd_threshold_cm=100.0)
    strict = _coverage(ct, t, gsd_threshold_cm=12.5)
    assert loose.gsd_stats()["quality_coverage_pct"] == pytest.approx(loose.coverage_pct(), abs=0.01)
    assert strict.gsd_stats()["quality_coverage_pct"] < strict.coverage_pct()
