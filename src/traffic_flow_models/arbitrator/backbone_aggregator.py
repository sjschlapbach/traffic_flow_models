import csv
import json
import warnings
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence, TypedDict

from traffic_flow_models.arbitrator.aggregation_helpers import (
    make_rolling_window_aggregator,
)


class DetectorMetadata(TypedDict):
    edge_id: str
    cell_key: str
    cell_index: int
    type: str
    position: float | None


# (begin_sec, count, speed_kmh, occupancy_percent, sampled_seconds)
DetectorInterval = tuple[float, int, float, float, float]

# (begin_sec, count, mean_speed_kmh, mean_occupancy_percent, n_lanes, exposure_seconds)
EdgeInterval = tuple[float, float, float, float, float, float]


class TrafficState(TypedDict):
    flow: float
    speed: float
    density_derived: float
    density_occupancy: float
    n_lanes: float


@dataclass
class CellAggregate:
    """Aggregated counters for a single cell/time bucket used during spatial aggregation.

    Holds summed vehicle counts, exposure-weighted speed sums, the number of lane
    observations (used for occupancy averaging), summed occupancy percentages,
    and total exposure seconds (sampledSeconds) used as weights when available.
    """

    count: float = 0.0
    weighted_speed_sum: float = 0.0
    lane_count: int = 0
    occupancy_sum: float = 0.0
    exposure_sum: float = 0.0


class BackboneStateAggregator:
    """Aggregate backbone loop detector data into macroscopic traffic state parameters.

    Processes SUMO induction loop outputs placed along backbone edges and computes
    time-varying flow (veh/h), speed (km/h), and density (veh/km) using a rolling
    window approach. Each edge produces a callable triple that can be queried at
    any simulation time.

    The fundamental relation  k = q / v  is used to derive density from the
    aggregated flow and space-mean speed. When speed data are absent or zero for
    a window, density falls back to zero rather than raising a division error.

    Attributes:
        detector_output_path: Path to SUMO detector output XML file.
        detector_spec_path: Path to detector specification CSV file.
        window_size_sec: Rolling window size in seconds.
        detector_intervals: Raw (begin, count, speed, occupancy, sampledSeconds) tuples indexed by detector ID.
        detector_mapping: Maps detector IDs to edge IDs, positions, and lane indices.
        edge_intervals: Per-edge lists of (begin, count, speed, occupancy, lanes, exposure_sec) after spatial aggregation.
            ``speed`` here remains the detector interval meanSpeed (spot/time-mean),
            and ``exposure_sec`` is sampledSeconds used for time-weighted smoothing.
        max_time: Maximum simulation time observed in detector data (seconds).
    """

    def __init__(
        self,
        detector_output_path: str,
        detector_spec_path: str,
        window_size_minutes: float = 2.0,
    ):
        """Initialise the backbone state aggregator.

        Args:
            detector_output_path: Path to the SUMO detector output XML file.
            detector_spec_path: Path to the detector specification CSV file.
            window_size_minutes: Rolling window half-width in minutes (default 2.0).
                At query time t the window covers [t - window/2, t + window/2].
        """
        self.detector_output_path: str = detector_output_path
        self.detector_spec_path: str = detector_spec_path
        self.window_size_sec: float = window_size_minutes * 60

        # raw readings: {det_id: [(begin_sec, count, speed_kmh, occupancy_pct), ...]}
        self.detector_intervals: defaultdict[str, list[DetectorInterval]] = defaultdict[
            str, list[DetectorInterval]
        ](list)
        self.detector_mapping: dict[str, DetectorMetadata] = {}

        # spatially aggregated per edge: {edge_id: [(begin_sec, count, speed_kmh, occupancy_pct, n_lanes, exposure_sec), ...]}
        self.edge_intervals: defaultdict[str, list[EdgeInterval]] = defaultdict[
            str, list[EdgeInterval]
        ](list)
        self.max_time: float = 0.0

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    def parse_detector_output(self) -> None:
        """Parse SUMO detector output XML and extract per-interval counts and speeds.

        The XML may contain a mix of E2 lane-area detectors (backbone segments)
        and E1 induction loops (origin / inflow / turning-rate). Each schema is
        parsed against its own required attribute set:

        - E2 intervals expose ``sampledSeconds``, ``meanSpeed``, ``meanOccupancy``.
        - E1 intervals expose ``speed`` and ``occupancy`` (no ``sampledSeconds``).

        """
        tree = ET.parse(self.detector_output_path)
        root = tree.getroot()

        for interval in root.findall("interval"):
            det_id = interval.get("id")
            begin_str = interval.get("begin")
            end_str = interval.get("end")

            if det_id is None or begin_str is None:
                warnings.warn(
                    f"Skipping malformed interval: {ET.tostring(interval, encoding='unicode')}",
                    stacklevel=2,
                )
                continue

            begin = float(begin_str)
            end = float(end_str) if end_str is not None else begin

            if end < begin:
                raise ValueError(
                    f"Detector {det_id}: end time {end} < begin time {begin}."
                )

            raw_count = interval.get("nVehEntered")
            if raw_count is None:
                raise ValueError(f"Detector {det_id}: missing nVehEntered.")
            count = int(raw_count)

            # Branch on schema using the E2-only attribute as the discriminator.
            if interval.get("sampledSeconds") is not None:
                # E2 lane-area detector (backbone segment)
                sampled_seconds = float(interval.get("sampledSeconds"))

                raw_speed_ms = interval.get("meanSpeed")
                if raw_speed_ms is None:
                    raise ValueError(
                        f"Detector {det_id}: E2 interval missing meanSpeed."
                    )
                speed_ms = float(raw_speed_ms)

                raw_occupancy = interval.get("meanOccupancy")
                if raw_occupancy is None:
                    raise ValueError(
                        f"Detector {det_id}: E2 interval missing meanOccupancy."
                    )
                occupancy = float(raw_occupancy)
            else:
                # E1 induction loop (origin / inflow / turning rate)
                sampled_seconds = 0.0  # not defined for point detectors

                raw_speed_ms = interval.get("speed")
                if raw_speed_ms is None:
                    raise ValueError(f"Detector {det_id}: E1 interval missing speed.")
                speed_ms = float(raw_speed_ms)

                raw_occupancy = interval.get("occupancy")
                if raw_occupancy is None:
                    raise ValueError(
                        f"Detector {det_id}: E1 interval missing occupancy."
                    )
                occupancy = float(raw_occupancy)

            # SUMO uses -1 to signal "no vehicles"; map to 0 km/h.
            speed_kmh = speed_ms * 3.6 if speed_ms >= 0 else 0.0

            self.detector_intervals[det_id].append(
                (begin, count, speed_kmh, occupancy, sampled_seconds)
            )
            self.max_time = max(self.max_time, end)

    def reset_state(self) -> None:
        self.detector_intervals = defaultdict[str, list[DetectorInterval]](list)
        self.detector_mapping = {}
        self.edge_intervals = defaultdict[str, list[EdgeInterval]](list)
        self.max_time = 0.0

    def classify_and_map(self) -> None:
        """Map only backbone cell detectors from the specification CSV.

        This class is intentionally backbone-only:
        - backbone_segment detectors are used for state aggregation
        - mainline_origin_interface / inflow / outflow / turning_rate are ignored here
        """
        with open(self.detector_spec_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)

            for row in reader:
                det_id = row["detector_id"].strip().strip('"').strip("'")
                det_type = row.get("type", "").strip().lower()

                if det_type != "backbone_segment":
                    continue

                edge_id = row.get("edge_id", "").strip().strip('"').strip("'")
                if not edge_id:
                    continue

                position_str = row.get("position", "").strip()
                position = float(position_str) if position_str else None

                cell_key = row.get("cell_key", "").strip()
                if not cell_key:
                    cell_index_str = row.get("cell_index", "").strip()
                    if cell_index_str == "":
                        warnings.warn(
                            f"Backbone detector '{det_id}' missing both cell_key and cell_index. Skipping.",
                            stacklevel=2,
                        )
                        continue
                    cell_index = int(cell_index_str)
                    cell_key = f"{edge_id}_cell{cell_index}"
                else:
                    cell_index_str = row.get("cell_index", "").strip()
                    cell_index = int(cell_index_str) if cell_index_str != "" else None

                if cell_index is None:
                    raise ValueError(
                        f"Backbone detector '{det_id}' has cell_key '{cell_key}' but missing or invalid cell_index."
                    )

                for variant in [
                    det_id,
                    det_id.replace("detector_", ""),
                    f"detector_{det_id}",
                ]:
                    self.detector_mapping[variant] = DetectorMetadata(
                        {
                            "edge_id": edge_id,
                            "cell_key": cell_key,
                            "cell_index": cell_index,
                            "type": det_type,
                            "position": position,
                        }
                    )

    def aggregate_spatially(self) -> None:
        cell_time_data: defaultdict[str, defaultdict[float, CellAggregate]] = (
            defaultdict(lambda: defaultdict(CellAggregate))
        )

        for det_id, intervals in self.detector_intervals.items():
            if det_id not in self.detector_mapping:
                continue

            mapping = self.detector_mapping[det_id]
            cell_key = mapping["cell_key"]

            for begin, count, speed_kmh, occupancy, sampled_seconds in intervals:
                bucket = cell_time_data[cell_key][begin]

                bucket.count += count
                # Time-weighted (vehicle-seconds) speed sum; empty intervals contribute 0.
                bucket.weighted_speed_sum += speed_kmh * sampled_seconds
                bucket.exposure_sum += sampled_seconds
                bucket.lane_count += 1  # Interval for occupancy averaging
                bucket.occupancy_sum += occupancy

        # Compute stats per cell_key and store in edge_intervals (holds cell level data)
        for cell_key, time_data in cell_time_data.items():
            for begin in sorted(time_data.keys()):
                aggregate = time_data[begin]

                mean_speed = (
                    aggregate.weighted_speed_sum / aggregate.exposure_sum
                    if aggregate.exposure_sum > 0
                    else 0.0
                )
                n_lanes = float(aggregate.lane_count)
                mean_occupancy = (
                    aggregate.occupancy_sum / n_lanes if n_lanes > 0 else 0.0
                )

                # self.edge_intervals[cell_key].append((begin, float(count), mean_speed))
                self.edge_intervals[cell_key].append(
                    (
                        begin,
                        float(aggregate.count),
                        mean_speed,
                        mean_occupancy,
                        n_lanes,
                        float(aggregate.exposure_sum),
                    )
                )

    def compute_origin_demand_functions(
        self,
        origin_ids: list[str],
        sumo_network_path: str,
    ) -> dict[str, Callable[[float], float]]:
        """Derive demand functions for mainline origins directly connected to backbone.

        This keeps direct motorway origins as backbone demand, but excludes on-ramp
        inflows by requiring a verified motorway-mainline connection and using only
        detectors of type 'mainline_origin_interface'.
        """

        net_tree = ET.parse(sumo_network_path)
        net_root = net_tree.getroot()

        node_to_out_edges: dict[str, list[ET.Element]] = {}
        for edge in net_root.findall("edge"):
            if edge.get("function") == "internal":
                continue
            f_node = edge.get("from")
            if f_node:
                node_to_out_edges.setdefault(f_node, []).append(edge)

        mainline_interface_by_edge: dict[str, list[str]] = defaultdict(list)

        with open(self.detector_spec_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)

            for row in reader:
                det_id = row["detector_id"].strip().strip('"').strip("'")
                det_type = row.get("type", "").strip().lower()
                edge_id = row.get("edge_id", "").strip().strip('"').strip("'")

                if det_type != "mainline_origin_interface":
                    continue
                if not edge_id:
                    continue

                for variant in [
                    det_id,
                    det_id.replace("detector_", ""),
                    f"detector_{det_id}",
                ]:
                    mainline_interface_by_edge[edge_id].append(variant)

        origin_demands: dict[str, Callable[[float], float]] = {}

        for origin_id in origin_ids:
            if not origin_id.startswith("origin_"):
                continue

            raw_node = origin_id.replace("origin_", "")
            connector_edges = node_to_out_edges.get(raw_node, [])

            # Collect ALL outgoing pure-mainline motorway edges. A mainline origin
            # that splits into multiple motorway carriageways at the boundary has
            # its demand distributed across them; we must aggregate all to get the
            # true total inflow.
            target_edge_ids: list[str] = []
            for c_edge in connector_edges:
                etype = c_edge.get("type", "").lower()
                eid = c_edge.get("id", "")
                if "motorway" in etype and "link" not in etype:
                    target_edge_ids.append(eid)

            if not target_edge_ids:
                continue

            # Aggregate detectors from every target edge by summing counts per
            # timestamp across all lanes of all edges.
            det_ids_all: list[str] = []
            for edge_id in target_edge_ids:
                det_ids_all.extend(mainline_interface_by_edge.get(edge_id, []))

            if not det_ids_all:
                print(
                    f"[X] Skipping {origin_id}: no mainline_origin_interface detectors "
                    f"found for edges {target_edge_ids}."
                )
                continue

            time_count_agg: defaultdict[float, float] = defaultdict(float)
            found_any_data = False

            for det_id in det_ids_all:
                det_intervals = self.detector_intervals.get(det_id)
                if det_intervals is None:
                    continue
                found_any_data = True
                for begin, count, _, _, _ in det_intervals:
                    time_count_agg[begin] += float(count)

            if not found_any_data or not time_count_agg:
                print(
                    f"[!] Warning: No detector data for verified highway origin "
                    f"{origin_id} on edges {target_edge_ids}"
                )
                continue

            flow_intervals: list[tuple[float, float]] = sorted(time_count_agg.items())
            # print(f"  origin {origin_id}: total interface count = {sum(time_count_agg.values())}")

            demand_fn = make_rolling_window_aggregator(
                intervals={"flow": flow_intervals},
                window_size_sec=self.window_size_sec,
                max_time=self.max_time,
                aggregation_type="demand",
            )

            if demand_fn:
                origin_demands[origin_id] = lambda t, fn=demand_fn: fn(t).get(
                    "flow", 0.0
                )
                print(
                    f"[OK] Highway mainline origin secured: {origin_id} "
                    f"(Edges: {target_edge_ids}, interface detectors: {len(det_ids_all)})"
                )
        return origin_demands

    def compute_traffic_state(
        self, free_flow_speed: float, jam_density: float
    ) -> dict[str, Callable[[float], TrafficState]]:
        """Compute time-varying flow, density, and speed functions for every edge.

        For each backbone edge a callable is returned.  At query time ``t`` (hours)
        the function aggregates all observations in the rolling window and returns:

        - ``flow``              – vehicles per hour across all lanes (veh/h)
        - ``speed``             – detector spot/time-mean speed (km/h), smoothed by exposure time (sampledSeconds)
        - ``density_derived``   – vehicles per kilometre per lane (veh/km/lane), derived as flow / speed
        - ``density_occupancy`` – vehicles per kilometre per lane (veh/km/lane), derived from occupancy
        - ``n_lanes``           – average lane count observed in the window

        Edges with no recorded vehicle observations are excluded with a warning.

        Returns:
            Mapping from edge IDs to state functions ``f(t_hours) → {"flow", "speed", "density"}``.
        """
        state_functions: dict[str, Callable[[float], TrafficState]] = {}

        for edge_id, intervals in self.edge_intervals.items():
            total_vehicles = sum(count for _, count, _, _, _, _ in intervals)

            if total_vehicles == 0:
                warnings.warn(
                    f"No vehicles detected on backbone edge '{edge_id}'. "
                    "Skipping this edge in state estimation.",
                    stacklevel=2,
                )
                continue

            state_fn = self._make_state_function(
                intervals=intervals,
                free_flow_speed=free_flow_speed,
                jam_density=jam_density,
            )
            if state_fn is not None:
                state_functions[edge_id] = state_fn

        return state_functions

    def _make_state_function(
        self, intervals: list[EdgeInterval], free_flow_speed: float, jam_density: float
    ) -> Callable[[float], TrafficState] | None:
        """Build a rolling-window state function for a single edge.

        Separates the joint ``(begin, count, speed, occupancy, exposure)`` tuples
        into two independent rolling-window aggregators — one for flow (count-based)
        and one for speed (exposure-weighted time-mean smoothing) — then combines
        their outputs via the fundamental relation k = q / v. Exposure is
        ``sampledSeconds`` from SUMO; if missing, vehicle count is used instead.

        Args:
            intervals: Sorted list of ``(begin_sec, count, speed_kmh)`` triples.

        Returns:
            Callable ``f(t_hours) → {"flow", "speed", "density"}``, or ``None``
            if the rolling-window helper returns ``None`` for the count stream.
        """

        # split into two parallel interval streams for the existing helper
        count_intervals: dict[str, list[tuple[float, float]]] = {
            "flow": [(begin, count) for begin, count, _, _, _, _ in intervals]
        }
        # represent speed as a pseudo-count stream weighted by vehicle count
        # aggregation_type="rate" will normalise by window duration → flow-like unit
        speed_weight_intervals: dict[str, list[tuple[float, float]]] = {
            "speed_x_count": [
                (begin, float(speed * exposure))  # weighted speed sum per interval
                for begin, _, speed, _, _, exposure in intervals
            ],
            "count_weight": [
                (begin, float(exposure)) for begin, _, _, _, _, exposure in intervals
            ],
        }

        raw_occupancy: list[tuple[float, float]] = [
            (begin, occupancy) for begin, _, _, occupancy, _, _ in intervals
        ]

        raw_n_lanes: list[tuple[float, float]] = [
            (begin, n_lanes) for begin, _, _, _, n_lanes, _ in intervals
        ]

        flow_fn = make_rolling_window_aggregator(
            intervals=count_intervals,
            window_size_sec=self.window_size_sec,
            max_time=self.max_time,
            aggregation_type="demand",  # → veh/h
        )

        if flow_fn is None:
            return None

        speed_numerator_fn = make_rolling_window_aggregator(
            intervals={"speed_x_count": speed_weight_intervals["speed_x_count"]},
            window_size_sec=self.window_size_sec,
            max_time=self.max_time,
            aggregation_type="demand",  # raw sum of (speed * count)
        )

        count_denom_fn = make_rolling_window_aggregator(
            intervals={"count_weight": speed_weight_intervals["count_weight"]},
            window_size_sec=self.window_size_sec,
            max_time=self.max_time,
            aggregation_type="demand",  # raw vehicle count in window
        )

        max_time = self.max_time
        window_size_sec = self.window_size_sec

        def mean_in_window(
            t_hours: float, data_stream: Sequence[tuple[float, float]]
        ) -> float:
            """Compute a simple average of values in a centred rolling window.

            The window is centred at the query time ``t_hours`` and clamped to
            the available data range [0, max_time]. For queries near the start
            or end of the recording the window is shifted so it remains within
            the observed interval.
            """
            query_sec = t_hours * 3600.0

            w_start = query_sec - window_size_sec / 2
            w_end = query_sec + window_size_sec / 2

            if w_start < 0:
                w_start, w_end = 0.0, min(window_size_sec, max_time)
            elif w_end > max_time:
                w_start = max(0.0, max_time - window_size_sec)
                w_end = max_time

            values = [val for begin, val in data_stream if w_start <= begin < w_end]
            return sum(values) / len(values) if values else 0.0

        def state_fn(t_hours: float) -> TrafficState:
            # TODO: We need to pre-process the user-specified input data to make sure all vehicles in SUMO simulation are of type car!
            # Effective vehicle length (km) used to convert occupancy fraction
            # to vehicle density. Assume that SUMO simulation was run with car
            # type vehicles only -> 5m vehicle length (default SUMO value)
            L_EFF_KM = 0.005

            flow_dict = flow_fn(t_hours)
            flow_total = flow_dict.get("flow")
            if flow_total is None:
                raise ValueError(
                    f"Flow function returned None for time {t_hours} hours."
                )

            # 1. Per-lane flow for density, but keep reported flow as total
            n_lanes = mean_in_window(t_hours, raw_n_lanes)
            flow_per_lane = flow_total / n_lanes if n_lanes > 0 else 0.0

            # 2. Calculate weighted mean speed
            if speed_numerator_fn is not None and count_denom_fn is not None:
                num = speed_numerator_fn(t_hours).get("speed_x_count")
                den = count_denom_fn(t_hours).get("count_weight")
                if num is None or den is None:
                    raise ValueError(
                        f"Speed numerator or denominator function returned None for time {t_hours} hours."
                    )
                speed = num / den if den > 0 else free_flow_speed
            else:
                speed = free_flow_speed
            # 3. Derive per-lane density from Flow/Speed: k = q / v
            density_derived = flow_per_lane / speed if speed > 0 else 0.0

            # 4. Convert Occupancy % to Density (veh/km/lane)
            # Formula: k = Occupancy_fraction / L_eff
            occ_percent = mean_in_window(t_hours, raw_occupancy)
            occ_fraction = occ_percent / 100.0
            density_occupancy = occ_fraction / L_EFF_KM

            # ! CAUTION: Speed is defined as a derived quantity here -> if calibration is
            # ! applied to this data, fix one degree of freedom to avoid correlation issues.
            # compute the vehicle flow based on the occupancy-derived
            # density, mean speed and the number of lanes
            flow_total = density_occupancy * speed * n_lanes

            return {
                "flow": flow_total,
                "speed": speed,
                "density_derived": density_derived,
                "density_occupancy": density_occupancy,
                "n_lanes": n_lanes,
            }

        return state_fn

    # ------------------------------------------------------------------
    # JSON output
    # ------------------------------------------------------------------

    def write_state_json(
        self,
        state_functions: Mapping[str, Callable[[float], TrafficState]],
        output_path: str,
        origin_demands: dict[str, Callable[[float], float]] | None = None,
        query_times_hours: list[float] | None = None,
        time_step_minutes: float = 1.0,
        dt: float | None = None,
        duration: float | None = None,
        preferred_cell_size: float | None = None,
        free_flow_speed: float | None = None,
        jam_density: float | None = None,
    ) -> str:
        """Evaluate state functions on a time grid and write results to JSON.

        The output is written in the same high-level format expected by
        :meth:`traffic_flow_models.network.simulation.Simulation.save_results`.
        This function fills per-motorway-link `flows`, `densities`, and
        `speeds` time series and provides a minimal `metadata` block with
        placeholders where backbone data cannot provide the value (for
        example, `model_type` is set to "MICRO").

        Args:
            state_functions: Mapping from cell keys (or edge keys) to state
                callables as returned by :meth:`compute_traffic_state`.
            output_path: Filesystem path for the output ``.json`` file.
            query_times_hours: Explicit list of query times in hours. If
                ``None`` a uniform grid from ``0`` to ``max_time`` with step
                ``time_step_minutes`` is used.
            time_step_minutes: Grid resolution in minutes when
                ``query_times_hours`` is not supplied (default 1.0).
            dt: Optional timestep in hours. If ``None`` it is derived from
                ``time_step_minutes`` (dt = time_step_minutes / 60.0).
            duration: Optional total simulation duration in hours. If
                ``None`` it defaults to ``self.max_time / 3600.0``.
            preferred_cell_size: Optional preferred cell size (units depend on
                the network configuration); used to populate metadata when
                known.
            free_flow_speed: Optional free-flow speed in km/h used when
                deriving densities.
            jam_density: Optional jam density in veh/km/lane used when
                converting occupancy to density.

        Returns:
            The resolved ``output_path`` string.
        """
        if query_times_hours is None:
            step_sec = time_step_minutes * 60.0
            query_times_hours = [
                t / 3600.0
                for t in [
                    i * step_sec for i in range(int(self.max_time / step_sec) + 1)
                ]
            ]

        # Build topology-driven edge -> {cell_index: state_fn} mapping.
        # Prefer explicit topology from `self.detector_mapping` (which is
        # populated from the detector spec). Also include any indices seen in
        # the `state_functions` keys as a fallback. For cells without a
        # corresponding state function, store ``None`` and emit zero-values
        # during export so that the exported arrays remain topology-stable
        # (not traffic-dependent).
        edge_cell_indices: dict[str, set[int]] = {}

        # Collect declared cells from detector mapping (topology source)
        for mapping in self.detector_mapping.values():
            edge_id = mapping.get("edge_id")
            if edge_id is None:
                continue
            cell_index = int(mapping.get("cell_index", 0))
            edge_cell_indices.setdefault(edge_id, set()).add(cell_index)

        # Also include any indices present in state_functions keys
        for key in state_functions.keys():
            if "_cell" in key:
                parts = key.rsplit("_cell", 1)
                e = parts[0]
                try:
                    ci = int(parts[1])
                except Exception:
                    ci = 0
            else:
                e = key
                ci = 0
            edge_cell_indices.setdefault(e, set()).add(ci)

        # Build contiguous per-edge cell index maps (0..max_index)
        edge_cells: dict[str, dict[int, Callable[[float], TrafficState] | None]] = {}
        for edge_id, indices in edge_cell_indices.items():
            if indices:
                max_idx = max(indices)
                cells_map: dict[int, Callable[[float], TrafficState] | None] = {}
                for idx in range(max_idx + 1):
                    # prefer explicit keyed functions like "edge_cell{idx}"
                    key_cell = f"{edge_id}_cell{idx}"
                    fn = None
                    if key_cell in state_functions:
                        fn = state_functions[key_cell]
                    elif idx == 0 and edge_id in state_functions:
                        # fallback: unindexed key may represent single-cell edges
                        fn = state_functions[edge_id]
                    else:
                        fn = None
                    cells_map[idx] = fn
                edge_cells[edge_id] = cells_map
            else:
                edge_cells[edge_id] = {}

        origin_demands_series: dict[str, list[float]] = {}
        if origin_demands:
            for oid in origin_demands.keys():
                origin_demands_series[oid] = []

        # Prepare output containers matching Simulation.save_results layout
        flows_time: dict[str, list] = {}
        densities_time: dict[str, list] = {}
        speeds_time: dict[str, list] = {}

        # pre-fill per-edge containers
        for edge_id, cells in edge_cells.items():
            flows_time[edge_id] = []
            densities_time[edge_id] = []
            speeds_time[edge_id] = []

        # Evaluate state functions for every timestep and assemble per-edge arrays
        for t_h in query_times_hours:
            for edge_id, cells in edge_cells.items():
                # determine number of cells (sparse indices supported)
                n_cells = max(cells.keys()) + 1 if cells else 0
                flows_row = [0.0] * n_cells
                densities_row = [0.0] * n_cells
                speeds_row = [0.0] * n_cells

                for idx in range(n_cells):
                    fn = cells.get(idx)
                    if fn is None:
                        # leave zeros for missing cells
                        continue

                    state = fn(t_h)
                    flows_row[idx] = float(round(state["flow"], 2))
                    speeds_row[idx] = float(round(state["speed"], 2))
                    densities_row[idx] = float(round(state["density_occupancy"], 4))

                flows_time[edge_id].append(flows_row)
                densities_time[edge_id].append(densities_row)
                speeds_time[edge_id].append(speeds_row)

            if origin_demands:
                for oid, fn in origin_demands.items():
                    val = fn(t_h)
                    origin_demands_series[oid].append(float(round(val, 2)))

        # Construct minimal link_properties using available information
        link_properties: dict[str, dict] = {}
        for edge_id, cells in edge_cells.items():
            n_cells = max(cells.keys()) + 1 if cells else 0
            # attempt to obtain lane counts from the first available state function
            n_lanes_vals: list[int] = []
            if n_cells > 0:
                for fn in cells.values():
                    if fn is None:
                        n_lanes_vals.append(0)
                        continue
                    try:
                        val = fn(query_times_hours[0])["n_lanes"]
                        n_lanes_vals.append(int(val))
                    except Exception:
                        n_lanes_vals.append(0)

            # avg_lanes = (
            #     float(sum(n_lanes_vals) / len(n_lanes_vals)) if n_lanes_vals else 0.0
            # )
            valid_lane_vals = [v for v in n_lanes_vals if v > 0]
            avg_lanes = (
                float(sum(valid_lane_vals) / len(valid_lane_vals))
                if valid_lane_vals
                else 0.0
            )

            link_properties[edge_id] = {
                "num_cells": n_cells,
                "n_lanes": avg_lanes,
            }

        # Build metadata matching Simulation.save_results exact fields.
        # Use provided values where available, otherwise sensible placeholders.
        # model_type is fixed for aggregator outputs
        _model_type = "MICRO"
        _dt = dt
        _duration = duration
        _pref_cell = preferred_cell_size

        # if dt/duration not provided, derive from time_step_minutes and max_time
        if _dt is None:
            _dt = time_step_minutes / 60.0 if time_step_minutes is not None else None
        if _duration is None:
            _duration = self.max_time / 3600.0 if self.max_time is not None else None

        metadata = {
            "model_type": _model_type,
            "simulation_parameters": {
                "dt": _dt,
                "duration": _duration,
                "preferred_cell_size": _pref_cell,
            },
            "link_properties": {},
            "critical_densities": {},
        }

        # Populate link_properties for motorway links (best-effort)
        for edge_id, info in link_properties.items():
            n_cells = int(info["num_cells"])
            cell_lengths = [None] * n_cells if n_cells > 0 else []
            metadata["link_properties"][edge_id] = {
                "length": None,
                "lanes": info["n_lanes"],
                "lane_capacity": None,
                "free_flow_speed": free_flow_speed,
                "jam_density": jam_density,
                "num_cells": n_cells,
                "cell_lengths": cell_lengths,
            }

        # set critical_densities placeholders per edge
        for edge_id in edge_cells.keys():
            metadata["critical_densities"][edge_id] = None

        out = {
            "metadata": metadata,
            "time_array": [round(t, 6) for t in query_times_hours],
            "state_time_series": {
                "flows": flows_time,
                "densities": densities_time,
                "speeds": speeds_time,
                "origin_queues": {},
                "onramp_queues": {},
                "offramp_queues": {},
            },
            # no disturbance inputs available from backbone data
            "disturbance_time_series": {
                "origin_demands": origin_demands_series,
                "turning_rates": {},
                "flow_boundary_conditions": {},
                "density_boundary_conditions": {},
            },
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)

        print(f"Backbone traffic state written → {output_path}")
        print(f"  Edges: {len(edge_cells)}")
        print(f"  Time steps per edge: {len(query_times_hours)}")

        return output_path

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run(
        self,
        output_path: str,
        urban_demands: dict[str, Callable[[float], float]],
        free_flow_speed: float,
        jam_density: float,
        sumo_network_path: str,
        origin_ids: list[str],
        query_times_hours: list[float] | None = None,
        time_step_minutes: float = 1.0,
        preferred_cell_size: float | None = None,
    ) -> tuple[str, dict[str, Callable[[float], float]]]:
        """Execute the full backbone state estimation pipeline.

        Parses detector output, maps detectors to backbone edges, aggregates
        lane-level readings spatially, constructs rolling-window state functions,
        and writes the results to a JSON file.

        Args:
            output_path: Path for the output JSON file.
            free_flow_speed: Free flow speed in km/h for density derivation.
            jam_density: Jam density in veh/km/lane for occupancy-based density.
            query_times_hours: Optional explicit time grid (hours).
            time_step_minutes: Grid resolution when ``query_times_hours`` is
                not provided.
            preferred_cell_size: Optional preferred cell size to hint cell
                discretization in metadata (units as per network configuration).

        Returns:
            Path to the written JSON file and a dictionary of origin demand functions.
        """
        self.reset_state()
        self.parse_detector_output()
        self.classify_and_map()
        self.aggregate_spatially()

        state_functions = self.compute_traffic_state(
            free_flow_speed=free_flow_speed, jam_density=jam_density
        )

        total_vehicles = sum(
            sum(count for _, count, _, _, _, _ in ivs)
            for ivs in self.edge_intervals.values()
        )

        highway_demands = self.compute_origin_demand_functions(
            origin_ids=origin_ids, sumo_network_path=sumo_network_path
        )

        print("BACKBONE STATE AGGREGATION SUMMARY:")
        print(f"  Backbone edges instrumented: {len(self.edge_intervals)}")
        print(f"  Total vehicles observed:     {total_vehicles}")
        print(f"  Edges with valid state fns:  {len(state_functions)}")
        print(
            f"  Edges skipped (no traffic):  "
            f"{len(self.edge_intervals) - len(state_functions)}"
        )

        path = self.write_state_json(
            state_functions,
            output_path,
            origin_demands={**urban_demands, **highway_demands},
            query_times_hours=query_times_hours,
            time_step_minutes=time_step_minutes,
            dt=(time_step_minutes / 60.0) if time_step_minutes is not None else None,
            duration=(self.max_time / 3600.0) if self.max_time is not None else None,
            preferred_cell_size=preferred_cell_size,
            free_flow_speed=free_flow_speed,
            jam_density=jam_density,
        )

        return path, highway_demands
