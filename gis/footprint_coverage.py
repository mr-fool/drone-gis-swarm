"""
footprint_coverage.py

Area-based coverage: marks every DEM pixel that falls inside a drone camera's
ground footprint along its continuous flight path.

This replaces the original "a 0.001-degree cell is covered if any position
sample lands in it" metric, which counted samples rather than area: it was
capped at (number of samples / number of cells), roughly doubled when the
timestep was halved, and ignored altitude entirely.

Model
-----
* Nadir-pointing camera with horizontal field of view ``fov_deg``. The ground
  footprint is approximated as a disc of radius AGL * tan(fov/2), i.e. its
  cross-track width equals the camera swath.
* The path between successive samples is densified so consecutive footprint
  discs are at most ``densify_frac`` * radius apart; the union of discs then
  approximates the swept swath (edge scalloping < 1% of radius at 0.25).
* Coverage is computed on the clipped DEM grid itself (1/3 arc-second,
  ~7.8 m E-W x ~10.3 m N-S), so elevation bands are assigned per DEM pixel
  rather than by sampling one pixel per coarse cell.
* Altitude modes:
    "agl" - trajectory z is height above ground (ideal terrain following).
            The footprint depends on z but not on terrain.
    "msl" - drones hold a fixed altitude: z_msl = (highest terrain in the
            area) + z. AGL, and therefore the footprint, varies with the
            terrain below. Pixels are only counted while
            AGL <= max_mapping_agl (beyond that, ground resolution is
            treated as too coarse to count as mapped).
  Terrain occlusion (viewshed) is not modeled in either mode.
"""

import numpy as np
from rasterio.transform import rowcol

from gis.coordinate_transformer import CoordinateTransformer, BCC_BOUNDS


DEFAULT_FOV_DEG = 60.0          # matches the visible-spectrum camera spec in src/
DEFAULT_MAX_MAPPING_AGL = 500.0  # meters; only used in "msl" mode


class FootprintCoverage:

    def __init__(
        self,
        transformer: CoordinateTransformer,
        fov_deg: float = DEFAULT_FOV_DEG,
        altitude_mode: str = "agl",
        max_mapping_agl: float = DEFAULT_MAX_MAPPING_AGL,
        densify_frac: float = 0.25,
    ):
        if altitude_mode not in ("agl", "msl"):
            raise ValueError("altitude_mode must be 'agl' or 'msl'")
        self.transformer     = transformer
        self.fov_deg         = fov_deg
        self.tan_half_fov    = np.tan(np.radians(fov_deg) / 2)
        self.altitude_mode   = altitude_mode
        self.max_mapping_agl = max_mapping_agl
        self.densify_frac    = densify_frac

        self.dem = transformer.get_dem_array().astype(np.float64)
        self.dem_transform = transformer.get_dem_transform()
        self.h, self.w = self.dem.shape
        self.valid = self.dem > -9000
        self.reference_msl = float(self.dem[self.valid].max())

        # Pixel size in meters (same flat-earth factors as coordinate_transformer)
        self.px_w_m = abs(self.dem_transform.a) * 84000.0
        self.px_h_m = abs(self.dem_transform.e) * 111000.0

        self.imaged = np.zeros((self.h, self.w), dtype=bool)
        self._stencils = {}
        self.n_samples = 0
        self.n_ground_contacts = 0
        self.n_above_mapping_agl = 0

    # ── geometry helpers ──────────────────────────────────────
    def _sim_to_pixel(self, x_m: np.ndarray, y_m: np.ndarray):
        """Simulation meters -> fractional DEM (row, col)."""
        b = BCC_BOUNDS
        lon = b["west"]  + x_m / self.transformer.sim_width_m  * (b["east"]  - b["west"])
        lat = b["south"] + y_m / self.transformer.sim_height_m * (b["north"] - b["south"])
        inv = ~self.dem_transform
        lon = np.asarray(lon, dtype=np.float64); lat = np.asarray(lat, dtype=np.float64)
        col = inv.a * lon + inv.b * lat + inv.c
        row = inv.d * lon + inv.e * lat + inv.f
        return row, col

    def _ground(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        r = np.clip(rows.astype(int), 0, self.h - 1)
        c = np.clip(cols.astype(int), 0, self.w - 1)
        return self.dem[r, c]

    def _stencil(self, radius_m: float):
        """Pixel offsets inside an ellipse of radius_m (pixels are not square)."""
        key = int(round(radius_m))          # 1 m quantization
        st = self._stencils.get(key)
        if st is None:
            rr = max(key, 1)
            ny = int(np.ceil(rr / self.px_h_m)); nx = int(np.ceil(rr / self.px_w_m))
            dy, dx = np.mgrid[-ny:ny + 1, -nx:nx + 1]
            inside = (dy * self.px_h_m) ** 2 + (dx * self.px_w_m) ** 2 <= rr ** 2
            st = (dy[inside], dx[inside])
            self._stencils[key] = st
        return st

    # ── main entry point ──────────────────────────────────────
    def add_trajectories(self, trajectories: np.ndarray):
        """
        trajectories : [n_drones, n_steps, 3] in simulation meters
                       (x east, y north from the SW corner; z per altitude_mode)
        """
        for traj in trajectories:
            self._add_path(np.asarray(traj, dtype=np.float64))

    def _add_path(self, traj: np.ndarray):
        # Densify so that footprint discs overlap: spacing <= densify_frac * radius
        pts = [traj[0]]
        for a, b in zip(traj[:-1], traj[1:]):
            seg = np.linalg.norm(b[:2] - a[:2])
            r_min = max(min(a[2], b[2]), 1.0) * self.tan_half_fov
            n = int(np.ceil(seg / max(self.densify_frac * r_min, 1.0)))
            if n > 1:
                f = np.linspace(0, 1, n + 1)[1:, None]
                pts.extend(a + f * (b - a))
            else:
                pts.append(b)
        pts = np.asarray(pts)

        rows, cols = self._sim_to_pixel(pts[:, 0], pts[:, 1])
        if self.altitude_mode == "agl":
            agl = pts[:, 2]
            usable = agl > 0
        else:
            agl = self.reference_msl + pts[:, 2] - self._ground(rows, cols)
            self.n_ground_contacts   += int(np.sum(agl <= 0))
            self.n_above_mapping_agl += int(np.sum(agl > self.max_mapping_agl))
            usable = (agl > 0) & (agl <= self.max_mapping_agl)
        self.n_samples += len(pts)

        radii = agl * self.tan_half_fov
        r_int = np.rint(rows).astype(int); c_int = np.rint(cols).astype(int)
        for k in np.flatnonzero(usable):
            dy, dx = self._stencil(radii[k])
            rr = r_int[k] + dy; cc = c_int[k] + dx
            ok = (rr >= 0) & (rr < self.h) & (cc >= 0) & (cc < self.w)
            self.imaged[rr[ok], cc[ok]] = True

    # ── statistics ────────────────────────────────────────────
    def coverage_pct(self, mask: np.ndarray = None) -> float:
        m = self.valid if mask is None else (self.valid & mask)
        n = int(m.sum())
        return round(100.0 * float(self.imaged[m].sum()) / n, 2) if n else 0.0

    def band_coverage(self, bands: dict) -> dict:
        out = {}
        for name, (lo, hi) in bands.items():
            m = self.valid & (self.dem >= lo) & (self.dem < hi)
            out[name] = {
                "elevation_range_m": f"{lo}–{hi}",
                "total_pixels":      int(m.sum()),
                "imaged_pixels":     int((self.imaged & m).sum()),
                "coverage_pct":      self.coverage_pct(m),
            }
        return out

    def imaged_area_km2(self) -> float:
        return round(float(self.imaged[self.valid].sum()) * self.px_w_m * self.px_h_m / 1e6, 4)

    def total_area_km2(self) -> float:
        return round(float(self.valid.sum()) * self.px_w_m * self.px_h_m / 1e6, 4)

    def coarse_fraction(self, n_rows: int, n_cols: int, bounds: dict, resolution: float) -> np.ndarray:
        """Fraction of each coarse (heatmap-grid) cell that was imaged."""
        rows_idx = np.arange(self.h); cols_idx = np.arange(self.w)
        # DEM pixel centers -> lat/lon -> coarse cell index
        lon = self.dem_transform.c + (cols_idx + 0.5) * self.dem_transform.a
        lat = self.dem_transform.f + (rows_idx + 0.5) * self.dem_transform.e
        cc = np.floor((lon - bounds["west"]) / resolution).astype(int)
        cr = np.floor((bounds["north"] - lat) / resolution).astype(int)
        cc = np.clip(cc, 0, n_cols - 1); cr = np.clip(cr, 0, n_rows - 1)
        flat = (cr[:, None] * n_cols + cc[None, :]).ravel()
        v = self.valid.ravel()
        tot = np.bincount(flat[v], minlength=n_rows * n_cols)
        hit = np.bincount(flat[v], weights=self.imaged.ravel()[v].astype(float),
                          minlength=n_rows * n_cols)
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = np.where(tot > 0, hit / tot, 0.0)
        return frac.reshape(n_rows, n_cols).astype(np.float32)

    def summary(self) -> dict:
        return {
            "metric":               "camera footprint area (DEM-pixel resolution)",
            "fov_deg":              self.fov_deg,
            "altitude_mode":        self.altitude_mode,
            "max_mapping_agl_m":    self.max_mapping_agl if self.altitude_mode == "msl" else None,
            "imaged_km2":           self.imaged_area_km2(),
            "total_km2":            self.total_area_km2(),
            "coverage_pct":         self.coverage_pct(),
            "footprint_samples":    self.n_samples,
            "ground_contact_samples": self.n_ground_contacts,
            "above_mapping_agl_samples": self.n_above_mapping_agl,
        }
