import csv
import json
import warnings
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Callable, Tuple

from traffic_flow_models.arbitrator.aggregation_helpers import (
    make_rolling_window_aggregator,
)


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
        detector_intervals: Raw (begin, count, speed) triples indexed by detector ID.
        detector_mapping: Maps detector IDs to edge IDs, positions, and lane indices.
        edge_intervals: Per-edge lists of (begin, count, speed) after spatial aggregation.
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

        # raw readings: {det_id: [(begin_sec, count, speed_kmh), ...]}
        self.detector_intervals: defaultdict[str, list[Tuple[float, int, float]]] = (
            defaultdict(list)
        )
        self.detector_mapping: dict[str, dict[str, str]] = {}

        # spatially aggregated per edge: {edge_id: [(begin_sec, count, speed_kmh), ...]}
        self.edge_intervals: defaultdict[str, list[Tuple[float, int, float]]] = (
            defaultdict(list)
        )
        self.max_time: float = 0.0

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    def parse_detector_output(self) -> None:
        """Parse SUMO detector output XML and extract per-interval counts and speeds.

        SUMO induction loop intervals expose ``nVehEntered`` (or ``nVehContrib``)
        for vehicle counts and ``speed`` for the time-mean speed of passing vehicles
        in m/s.  Both are extracted and the speed is converted to km/h.
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

            count = int(interval.get("nVehEntered", interval.get("nVehContrib", 0)))

            # SUMO reports speed in m/s; -1 indicates no vehicles in interval
            raw_speed_ms = float(interval.get("speed", -1.0))
            speed_kmh = raw_speed_ms * 3.6 if raw_speed_ms >= 0 else 0.0

            self.detector_intervals[det_id].append((begin, count, speed_kmh))
            self.max_time = max(self.max_time, end)

    def reset_state(self) -> None:
        self.detector_intervals = defaultdict(list)
        self.detector_mapping = {}
        self.edge_intervals = defaultdict(list)
        self.max_time = 0.0

    def classify_and_map(self) -> None:
        """Map backbone detector IDs to their edge IDs from the specification CSV.

        Only rows whose ``type`` field contains ``backbone_segment`` are processed;
        all other detector types (inflow, outflow, turning_rate …) are ignored so
        that only the regularly-spaced backbone detectors contribute to state
        estimation.
        """
        with open(self.detector_spec_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)

            for row in reader:
                det_id = row["detector_id"].strip().strip('"').strip("'")
                det_type = row.get("type", "").strip().lower()

                if "backbone_segment" not in det_type:
                    continue

                edge_id = row["edge_id"].strip().strip('"').strip("'")
                if not edge_id:
                    continue

                position_str = row.get("position", "").strip()
                position = float(position_str) if position_str else None

                # support the same ID variants used in other aggregators
                for variant in [
                    det_id,
                    det_id.replace("detector_", ""),
                    f"detector_{det_id}",
                ]:
                    self.detector_mapping[variant] = {
                        "edge_id": edge_id,
                        "type": det_type,
                        "position": position,
                    }

    def aggregate_spatially(self) -> None:
        # accumulate per (edge, position) cross-section per timestamp
        # structure: {edge_id: {position: {begin: [count_sum, speed_weighted_sum, weight]}}}
        edge_position_time: defaultdict[
            str, defaultdict[float | None, defaultdict[float, list]]
        ] = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0, 0.0, 0])))

        for det_id, intervals in self.detector_intervals.items():
            if det_id not in self.detector_mapping:
                continue

            mapping = self.detector_mapping[det_id]
            edge_id = mapping["edge_id"]
            position = mapping.get("position")

            for begin, count, speed_kmh in intervals:
                bucket = edge_position_time[edge_id][position][begin]
                bucket[0] += count
                bucket[1] += speed_kmh * count
                bucket[2] += count

        # for each edge, average across positions then store as edge_intervals
        for edge_id, position_data in edge_position_time.items():
            num_positions = len(position_data)
            if num_positions == 0:
                continue

            # collect all timestamps across all positions
            all_timestamps: set[float] = set()
            for time_data in position_data.values():
                all_timestamps.update(time_data.keys())

            for begin in sorted(all_timestamps):
                total_count = 0
                weighted_speed_sum = 0.0
                total_weight = 0

                for time_data in position_data.values():
                    if begin in time_data:
                        count, wspeed, weight = time_data[begin]
                        total_count += count
                        weighted_speed_sum += wspeed
                        total_weight += weight

                # normalize count by number of positions to avoid length inflation
                normalized_count = total_count / num_positions
                mean_speed = (
                    weighted_speed_sum / total_weight if total_weight > 0 else 0.0
                )
                self.edge_intervals[edge_id].append(
                    (begin, normalized_count, mean_speed)
                )

    def compute_traffic_state(
        self,
    ) -> dict[
        str,
        Callable[[float], dict[str, float]],
    ]:
        """Compute time-varying flow, density, and speed functions for every edge.

        For each backbone edge a callable is returned.  At query time ``t`` (hours)
        the function aggregates all observations in the rolling window and returns:

        - ``flow``    – vehicles per hour (veh/h)
        - ``speed``   – space-mean speed (km/h)
        - ``density`` – vehicles per kilometre (veh/km), derived as flow / speed

        Edges with no recorded vehicle observations are excluded with a warning.

        Returns:
            Mapping from edge IDs to state functions ``f(t_hours) → {"flow", "speed", "density"}``.
        """
        state_functions: dict[str, Callable[[float], dict[str, float]]] = {}

        for edge_id, intervals in self.edge_intervals.items():
            total_vehicles = sum(count for _, count, _ in intervals)

            if total_vehicles == 0:
                warnings.warn(
                    f"No vehicles detected on backbone edge '{edge_id}'. "
                    "Skipping this edge in state estimation.",
                    stacklevel=2,
                )
                continue

            state_fn = self._make_state_function(intervals)
            if state_fn is not None:
                state_functions[edge_id] = state_fn

        return state_functions

    def _make_state_function(
        self,
        intervals: list[Tuple[float, int, float]],
    ) -> Callable[[float], dict[str, float]] | None:
        """Build a rolling-window state function for a single edge.

        Separates the joint ``(begin, count, speed)`` triples into two independent
        rolling-window aggregators — one for flow (count-based) and one for
        speed (rate-weighted average) — then combines their outputs via the
        fundamental relation k = q / v.

        Args:
            intervals: Sorted list of ``(begin_sec, count, speed_kmh)`` triples.

        Returns:
            Callable ``f(t_hours) → {"flow", "speed", "density"}``, or ``None``
            if the rolling-window helper returns ``None`` for the count stream.
        """

        SCALE = 1000

        # split into two parallel interval streams for the existing helper
        count_intervals: dict[str, list[Tuple[float, int]]] = {
            "flow": [(begin, count * SCALE) for begin, count, _ in intervals]
        }
        # represent speed as a pseudo-count stream weighted by vehicle count
        # aggregation_type="rate" will normalise by window duration → flow-like unit
        speed_weight_intervals: dict[str, list[Tuple[float, int]]] = {
            "speed_x_count": [
                (begin, int(speed * count * SCALE))  # weighted speed sum per interval
                for begin, count, speed in intervals
            ],
            "count_weight": [(begin, count * SCALE) for begin, count, _ in intervals],
        }

        flow_fn = make_rolling_window_aggregator(
            intervals=count_intervals,
            window_size_sec=self.window_size_sec,
            max_time=self.max_time,
            aggregation_type="rate",  # → veh/h
        )

        if flow_fn is None:
            return None

        speed_numerator_fn = make_rolling_window_aggregator(
            intervals={"speed_x_count": speed_weight_intervals["speed_x_count"]},
            window_size_sec=self.window_size_sec,
            max_time=self.max_time,
            aggregation_type="rate",  # raw sum of (speed * count)
        )

        count_denom_fn = make_rolling_window_aggregator(
            intervals={"count_weight": speed_weight_intervals["count_weight"]},
            window_size_sec=self.window_size_sec,
            max_time=self.max_time,
            aggregation_type="rate",  # raw vehicle count in window
        )

        def state_fn(t_hours: float) -> dict[str, float]:
            flow_dict = flow_fn(t_hours)
            flow = flow_dict.get("flow", 0.0)

            # weighted-mean speed: sum(v_i * n_i) / sum(n_i)
            if speed_numerator_fn is not None and count_denom_fn is not None:
                num = speed_numerator_fn(t_hours).get("speed_x_count", 0.0)
                den = count_denom_fn(t_hours).get("count_weight", 0.0)
                speed = num / den if den > 0 else 0.0
            else:
                speed = 0.0

            # fundamental relation: k = q / v  (veh/km)
            density = flow / speed if speed > 0 else 0.0

            return {"flow": flow, "speed": speed, "density": density}

        return state_fn

    # ------------------------------------------------------------------
    # JSON output
    # ------------------------------------------------------------------

    def write_state_json(
        self,
        state_functions: dict[str, Callable[[float], dict[str, float]]],
        output_path: str,
        query_times_hours: list[float] | None = None,
        time_step_minutes: float = 1.0,
    ) -> str:
        """Evaluate state functions on a time grid and write results to JSON.

        The output format is::

            {
                "metadata": {
                    "window_size_minutes": ...,
                    "time_step_minutes": ...,
                    "num_edges": ...,
                    "max_simulation_time_hours": ...
                },
                "edges": {
                    "<edge_id>": [
                        {"time_hours": 0.0, "flow": ..., "speed": ..., "density": ...},
                        ...
                    ],
                    ...
                }
            }

        Args:
            state_functions: Mapping from edge IDs to state callables as returned
                by :meth:`compute_traffic_state`.
            output_path: Filesystem path for the output ``.json`` file.
            query_times_hours: Explicit list of query times in hours.  If ``None``
                a uniform grid from ``0`` to ``max_time`` with step
                ``time_step_minutes`` is used.
            time_step_minutes: Grid resolution in minutes when
                ``query_times_hours`` is not supplied (default 1.0).

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

        output: dict = {
            "metadata": {
                "window_size_minutes": self.window_size_sec / 60.0,
                "time_step_minutes": time_step_minutes,
                "num_edges": len(state_functions),
                "max_simulation_time_hours": round(self.max_time / 3600.0, 4),
            },
            "edges": {},
        }

        for edge_id, state_fn in state_functions.items():
            time_series = []
            for t_h in query_times_hours:
                state = state_fn(t_h)
                time_series.append(
                    {
                        "time_hours": round(t_h, 6),
                        "flow": round(state["flow"], 2),
                        "speed": round(state["speed"], 2),
                        "density": round(state["density"], 4),
                    }
                )
            output["edges"][edge_id] = time_series

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)

        print(f"Backbone traffic state written → {output_path}")
        print(f"  Edges: {len(state_functions)}")
        print(f"  Time steps per edge: {len(query_times_hours)}")

        return output_path

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run(
        self,
        output_path: str,
        query_times_hours: list[float] | None = None,
        time_step_minutes: float = 1.0,
    ) -> str:
        """Execute the full backbone state estimation pipeline.

        Parses detector output, maps detectors to backbone edges, aggregates
        lane-level readings spatially, constructs rolling-window state functions,
        and writes the results to a JSON file.

        Args:
            output_path: Path for the output JSON file.
            query_times_hours: Optional explicit time grid (hours).
            time_step_minutes: Grid resolution when ``query_times_hours`` is
                not provided.

        Returns:
            Path to the written JSON file.
        """
        self.reset_state()
        self.parse_detector_output()
        self.classify_and_map()
        self.aggregate_spatially()

        state_functions = self.compute_traffic_state()

        total_vehicles = sum(
            sum(count for _, count, _ in ivs) for ivs in self.edge_intervals.values()
        )

        print("BACKBONE STATE AGGREGATION SUMMARY:")
        print(f"  Backbone edges instrumented: {len(self.edge_intervals)}")
        print(f"  Total vehicles observed:     {total_vehicles}")
        print(f"  Edges with valid state fns:  {len(state_functions)}")
        print(
            f"  Edges skipped (no traffic):  "
            f"{len(self.edge_intervals) - len(state_functions)}"
        )

        return self.write_state_json(
            state_functions,
            output_path,
            query_times_hours=query_times_hours,
            time_step_minutes=time_step_minutes,
        )
