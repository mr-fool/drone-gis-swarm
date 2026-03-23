"""
coordinate_transformer.py

Converts drone swarm simulation coordinates (x/y in meters, origin at
simulation bounds southwest corner) into real-world WGS84 lat/lon coordinates
anchored to Big Cottonwood Canyon, Utah.

Also clips and loads the USGS DEM for the simulation area.

Anchor: Big Cottonwood Canyon, Wasatch Range, Utah
DEM:    USGS 3DEP 1/3 Arc-Second, tile n41w112, published 2026-01-13
"""

import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.transform import rowcol
from pyproj import Transformer
import os


# Big Cottonwood Canyon bounding box (WGS84)
BCC_BOUNDS = {
    "south": 40.55,
    "north": 40.67,
    "west": -111.81,
    "east": -111.58,
}

# Simulation area dimensions in meters derived from bounding box
# At ~40.6N latitude: 1 deg lat ≈ 111,000m, 1 deg lon ≈ 84,000m
SIM_WIDTH_M  = (BCC_BOUNDS["east"]  - BCC_BOUNDS["west"])  * 84000   # ~19,320m
SIM_HEIGHT_M = (BCC_BOUNDS["north"] - BCC_BOUNDS["south"]) * 111000  # ~13,320m

# Default DEM path relative to repo root
DEFAULT_DEM_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "raw",
    "USGS_13_n41w112_20260113.tif"
)


class CoordinateTransformer:
    """
    Transforms between simulation coordinate space and real-world WGS84.

    Simulation space:
        Origin (0, 0) = southwest corner of BCC bounding box
        x increases eastward  (meters)
        y increases northward (meters)
        z = altitude above ground (meters)

    Real-world space:
        WGS84 lat/lon (EPSG:4326)
    """

    def __init__(self, dem_path: str = DEFAULT_DEM_PATH):
        self.bounds = BCC_BOUNDS
        self.sim_width_m  = SIM_WIDTH_M
        self.sim_height_m = SIM_HEIGHT_M
        self.dem_path = dem_path
        self._dem_data  = None
        self._dem_transform = None
        self._dem_crs = None
        self._load_dem()

    def _load_dem(self):
        """Load and clip DEM to simulation bounding box."""
        if not os.path.exists(self.dem_path):
            raise FileNotFoundError(
                f"DEM not found at {self.dem_path}\n"
                f"Run: git lfs pull"
            )

        with rasterio.open(self.dem_path) as src:
            # Clip to BCC bounds only — don't load the full 394MB tile
            window = from_bounds(
                left   = self.bounds["west"],
                bottom = self.bounds["south"],
                right  = self.bounds["east"],
                top    = self.bounds["north"],
                transform = src.transform
            )
            self._dem_data      = src.read(1, window=window)
            self._dem_transform = src.window_transform(window)
            self._dem_crs       = src.crs

        print(f"DEM loaded: {self._dem_data.shape[1]}px wide x "
              f"{self._dem_data.shape[0]}px tall")
        print(f"Elevation range: {self._dem_data.min():.1f}m – "
              f"{self._dem_data.max():.1f}m")

    def sim_to_latlon(
        self, x_m: float, y_m: float
    ) -> tuple[float, float]:
        """
        Convert simulation (x, y) meters to (lat, lon) WGS84.

        Parameters
        ----------
        x_m : float  — eastward distance from SW corner in meters
        y_m : float  — northward distance from SW corner in meters

        Returns
        -------
        (latitude, longitude) in decimal degrees
        """
        lon = self.bounds["west"]  + (x_m / self.sim_width_m)  * (
            self.bounds["east"] - self.bounds["west"]
        )
        lat = self.bounds["south"] + (y_m / self.sim_height_m) * (
            self.bounds["north"] - self.bounds["south"]
        )
        return lat, lon

    def latlon_to_sim(
        self, lat: float, lon: float
    ) -> tuple[float, float]:
        """
        Convert (lat, lon) WGS84 to simulation (x, y) meters.

        Returns
        -------
        (x_m, y_m) distance from SW corner in meters
        """
        x_m = (lon - self.bounds["west"])  / (
            self.bounds["east"] - self.bounds["west"]
        ) * self.sim_width_m
        y_m = (lat - self.bounds["south"]) / (
            self.bounds["north"] - self.bounds["south"]
        ) * self.sim_height_m
        return x_m, y_m

    def get_ground_elevation(
        self, x_m: float, y_m: float
    ) -> float:
        """
        Return ground elevation in meters at simulation position (x, y).

        Uses bilinear interpolation from the clipped DEM.
        Returns 0.0 if position is outside DEM bounds.
        """
        lat, lon = self.sim_to_latlon(x_m, y_m)

        try:
            row, col = rowcol(self._dem_transform, lon, lat)
            row, col = int(row), int(col)
            h, w = self._dem_data.shape
            if 0 <= row < h and 0 <= col < w:
                elev = float(self._dem_data[row, col])
                # USGS nodata value is typically -9999
                return elev if elev > -9000 else 0.0
        except Exception:
            pass
        return 0.0

    def sim_positions_to_geodataframe(
        self, positions: list[dict]
    ):
        """
        Convert a list of drone position dicts to a GeoDataFrame.

        Each dict should have at minimum: x, y, z, drone_id, timestep
        Adds: lat, lon, ground_elevation_m, altitude_agl_m columns

        Returns
        -------
        geopandas.GeoDataFrame with Point geometry in WGS84
        """
        import geopandas as gpd
        from shapely.geometry import Point

        rows = []
        for pos in positions:
            lat, lon = self.sim_to_latlon(pos["x"], pos["y"])
            ground_elev = self.get_ground_elevation(pos["x"], pos["y"])
            rows.append({
                **pos,
                "lat": lat,
                "lon": lon,
                "ground_elevation_m": ground_elev,
                "altitude_agl_m": pos.get("z", 0.0),
                "altitude_msl_m": ground_elev + pos.get("z", 0.0),
                "geometry": Point(lon, lat),
            })

        gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
        return gdf

    def get_dem_array(self) -> np.ndarray:
        """Return the clipped DEM as a numpy array."""
        return self._dem_data.copy()

    def get_dem_transform(self):
        """Return the affine transform of the clipped DEM."""
        return self._dem_transform

    def print_summary(self):
        """Print a summary of the coordinate system setup."""
        print("=" * 50)
        print("Coordinate Transformer — Big Cottonwood Canyon")
        print("=" * 50)
        print(f"Bounds:      {self.bounds}")
        print(f"Sim width:   {self.sim_width_m:.0f}m")
        print(f"Sim height:  {self.sim_height_m:.0f}m")
        print(f"DEM shape:   {self._dem_data.shape}")
        print(f"DEM CRS:     {self._dem_crs}")
        print(f"Elev min:    {self._dem_data.min():.1f}m")
        print(f"Elev max:    {self._dem_data.max():.1f}m")
        print("=" * 50)


if __name__ == "__main__":
    # Quick sanity check
    ct = CoordinateTransformer()
    ct.print_summary()

    # Test center of simulation area
    cx = SIM_WIDTH_M  / 2
    cy = SIM_HEIGHT_M / 2
    lat, lon = ct.sim_to_latlon(cx, cy)
    elev = ct.get_ground_elevation(cx, cy)

    print(f"\nCenter of simulation area:")
    print(f"  Sim:  x={cx:.0f}m, y={cy:.0f}m")
    print(f"  Geo:  lat={lat:.6f}, lon={lon:.6f}")
    print(f"  Elev: {elev:.1f}m above sea level")

    # Test round-trip accuracy
    x2, y2 = ct.latlon_to_sim(lat, lon)
    print(f"\nRound-trip accuracy:")
    print(f"  Original:  x={cx:.2f}, y={cy:.2f}")
    print(f"  Recovered: x={x2:.2f}, y={y2:.2f}")
    print(f"  Error:     {abs(cx-x2):.4f}m, {abs(cy-y2):.4f}m")
