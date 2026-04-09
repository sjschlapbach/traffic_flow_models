"""urban_demand_model.py

Generates a full 24-hour SUMO demand profile from a realistic hourly
distribution, feeding the METANET/CTM calibration pipeline with genuine
peak / off-peak traffic regimes.

Usage
-----
    from urban_demand_model import UrbanDemandModel

    model = UrbanDemandModel(
        net_file="my_network.net.xml",
        rou_file="my_routes.rou.xml",
        fringe_factor=10,   # 10 = highway/arterial bias; 1 = urban grid
        seed=42,            # optional, for reproducibility
    )
    model.generate_demand(vehicle_count=5000)
"""

from __future__ import annotations  # Python 3.8/3.9 compat for list[str] hints

import os
import re
import sys
import subprocess
import xml.etree.ElementTree as ET
from typing import List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Regex compiled once at import time for safe depart= extraction
# ---------------------------------------------------------------------------


class UrbanDemandModel:
    """
    Urban demand for OSM-derived SUMO networks feeding a
    METANET/CTM calibration pipeline.

    Runs randomTrips.py in time slices with a realistic hourly
    profile so highway aggregation sees genuine peak/off-peak
    regimes — the only thing CTM calibration actually needs.

    Parameters
    ----------
    net_file : str
        Path to the SUMO .net.xml file.
    rou_file : str
        Path where the merged .rou.xml will be written.
    fringe_factor : int
        Weight given to fringe (boundary) edges when randomTrips.py
        selects trip endpoints.
        - 1  → equal weight for fringe and internal edges (urban grids).
        - 10 → strongly prefers fringe edges (highway / arterial networks).
        Default: 1.
    seed : int | None
        Random seed passed to randomTrips.py for reproducible outputs.
        Pass None to let SUMO pick its own seed. Default: None.
    subprocess_timeout : int
        Maximum seconds to wait for each randomTrips.py call before
        raising TimeoutExpired. Default: 300.
    """

    # Fraction of daily demand generated each hour (24 values).
    # Intentionally NOT normalised here; normalisation happens at runtime
    # so the array is easy to read and edit.
    HOURLY_PROFILE = np.array([
        0.003, 0.002, 0.002, 0.002, 0.005, 0.012,
        0.040, 0.085, 0.095, 0.060, 0.038, 0.040,   # AM peak (hours 6-11)
        0.045, 0.042, 0.040, 0.050, 0.065, 0.095,   # PM peak (hours 12-17)
        0.088, 0.060, 0.040, 0.025, 0.013, 0.006,
    ])

    def __init__(
        self,
        net_file: str,
        rou_file: str,
        fringe_factor: int = 1,
        seed: Optional[int] = None,
        subprocess_timeout: int = 300,
    ) -> None:
        self.net_file = net_file
        self.rou_file = rou_file
        self.fringe_factor = fringe_factor
        self.seed = seed
        self.subprocess_timeout = subprocess_timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_demand(self, vehicle_count: int) -> None:
        """Generate a full-day demand profile and write it to ``rou_file``.

        Parameters
        ----------
        vehicle_count : int
            Total number of vehicles to distribute across the 24-hour period.

        Raises
        ------
        EnvironmentError
            If the ``SUMO_HOME`` environment variable is not set.
        FileNotFoundError
            If ``randomTrips.py`` cannot be found under ``$SUMO_HOME/tools``.
        """
        random_trips = self._locate_random_trips()
        vehicles_per_hour = self._allocate_vehicles(vehicle_count)

        temp_route_files: List[str] = []

        for hour, n_veh in enumerate(vehicles_per_hour):
            if n_veh == 0:
                continue

            period = 3600.0 / n_veh          # average inter-vehicle headway (s)
            begin  = hour * 3600
            end    = begin + 3600
            temp_trips_file = f"temp_trips_h{hour:02d}.xml"
            temp_route_file = f"temp_routes_h{hour:02d}.xml"

            cmd = [
                sys.executable, random_trips,
                "-n",             self.net_file,
                "-o",             temp_trips_file,
                "--route-file",   temp_route_file,
                "--begin",        str(begin),
                "--end",          str(end),
                "--period",       f"{period:.4f}",
                "--fringe-factor", str(self.fringe_factor),
                "--validate",
                "--remove-loops",
                "--trip-attributes",
                'departLane="best" departSpeed="max"',
            ]

            if self.seed is not None:
                cmd += ["--seed", str(self.seed + hour)]  # unique seed per hour

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.subprocess_timeout,
                )
            except subprocess.TimeoutExpired:
                print(f"  Warning hour {hour:02d}: randomTrips.py timed out after "
                      f"{self.subprocess_timeout}s — skipping.")
                continue

            if result.returncode != 0:
                print(f"  Warning hour {hour:02d}: randomTrips.py exited with error:\n"
                      f"    {result.stderr[:300]}")
                continue

            temp_route_files.append((hour, temp_route_file))
            print(f"  Hour {hour:02d}: {n_veh:4d} vehicles  (period={period:.1f} s)")

        self._merge_route_files(temp_route_files)
        self._cleanup(vehicles_per_hour)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _locate_random_trips(self) -> str:
        """Return the absolute path to randomTrips.py, raising clearly if missing."""
        if "SUMO_HOME" not in os.environ:
            raise EnvironmentError(
                "Environment variable SUMO_HOME is not set. "
                "Point it to your SUMO installation directory."
            )
        path = os.path.join(os.environ["SUMO_HOME"], "tools", "randomTrips.py")
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"randomTrips.py not found at '{path}'. "
                "Check that SUMO_HOME is set correctly."
            )
        return path

    def _allocate_vehicles(self, vehicle_count: int) -> np.ndarray:
        """Distribute ``vehicle_count`` across 24 hours using the largest-remainder
        method so the total is always exactly ``vehicle_count``.

        Plain rounding (``np.round``) can under- or over-count by several
        vehicles — the largest-remainder method guarantees exact conservation.
        """
        profile = self.HOURLY_PROFILE / self.HOURLY_PROFILE.sum()
        raw     = profile * vehicle_count
        floored = np.floor(raw).astype(int)

        deficit = vehicle_count - floored.sum()          # always >= 0
        top_idx = np.argsort(raw - floored)[::-1][:deficit]
        floored[top_idx] += 1

        assert floored.sum() == vehicle_count, (
            f"Allocation bug: got {floored.sum()}, expected {vehicle_count}"
        )
        return floored

    def _merge_route_files(self, files: List[str]) -> None:
        vehicles = []

        for hour_idx, path in files:
            if not os.path.exists(path):
                print(f"  Warning: expected temp file '{path}' not found — skipping.")
                continue
            try:
                tree = ET.parse(path)
            except ET.ParseError as e:
                print(f"  Warning: could not parse '{path}': {e} — skipping.")
                continue

            for elem in tree.getroot():
                if elem.tag != "vehicle":
                    continue
                # Prefix ID to avoid cross-hour collisions
                elem.set("id", f"h{hour_idx:02d}_{elem.get('id')}")
                vehicles.append(elem)

        # Sort by depart time
        vehicles.sort(key=lambda e: float(e.get("depart", 0)))

        root = ET.Element("routes")
        vtype = ET.SubElement(root, "vType")
        vtype.set("id", "car")
        vtype.set("vClass", "passenger")
        vtype.set("accel", "2.6")
        vtype.set("decel", "4.5")
        vtype.set("sigma", "0.5")
        vtype.set("length", "5.0")
        vtype.set("minGap", "2.5")
        vtype.set("maxSpeed", "33.0")
        vtype.set("speedFactor", "normc(1,0.1,0.6,1.4)")

        for v in vehicles:
            root.append(v)

        ET.ElementTree(root).write(self.rou_file, encoding="unicode", xml_declaration=True)
        print(f"\nMerged {len(vehicles)} vehicles → {self.rou_file}")

    def _cleanup(self, vehicles_per_hour: np.ndarray) -> None:
        """Remove all per-hour temp trip and route files."""
        for hour in range(len(vehicles_per_hour)):
            for path in [
                f"temp_trips_h{hour:02d}.xml",
                f"temp_routes_h{hour:02d}.xml",
            ]:
                if os.path.exists(path):
                    os.remove(path)