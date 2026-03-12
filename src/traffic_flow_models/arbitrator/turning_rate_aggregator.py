import csv
import warnings
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Callable, Mapping, Tuple

from traffic_flow_models.arbitrator.aggregation_helpers import (
    make_rolling_window_aggregator,
)


class TurningRateAggregator:
    """Aggregate microscopic SUMO detector data into macroscopic turning rate functions.

    This class processes loop detector outputs from SUMO simulations placed at diverge
    nodes and aggregates them spatially and temporally to produce time-varying turning
    rate functions. These functions describe the fraction of vehicles using each
    outgoing link at a diverge node as a function of time.

    Uses a rolling window approach for temporal aggregation to smooth turning rates
    while preserving time-varying behavior.

    Attributes:
        detector_output_path: Path to SUMO detector output XML file.
        detector_spec_path: Path to detector specification CSV file.
        window_size_sec: Rolling window size in seconds for temporal aggregation.
        detector_intervals: Raw detector readings indexed by detector ID.
        detector_mapping: Maps detector IDs to diverge node IDs, edge IDs, and types.
        diverge_link_intervals: Raw interval data per diverge node per edge.
        max_time: Maximum simulation time observed in detector data.
    """

    def __init__(
        self,
        detector_output_path: str,
        detector_spec_path: str,
        window_size_minutes: float = 2.0,
    ):
        """Initialize the turning rate aggregator.

        Args:
            detector_output_path: Path to the SUMO detector output XML file.
            detector_spec_path: Path to the detector specification CSV file.
            window_size_minutes: Rolling window size in minutes (default: 2.0).
                At query time t, vehicle counts from [t - window/2, t + window/2] are aggregated.
        """
        self.detector_output_path: str = detector_output_path
        self.detector_spec_path: str = detector_spec_path
        self.window_size_sec: float = window_size_minutes * 60

        self.detector_intervals: defaultdict[str, list[Tuple[float, int]]] = (
            defaultdict(list)
        )
        self.detector_mapping: dict[str, dict[str, str]] = {}
        # structure: {diverge_node_id: {edge_id: [(begin_time, count), ...]}}
        self.diverge_link_intervals: defaultdict[
            str, defaultdict[str, list[Tuple[float, int]]]
        ] = defaultdict(lambda: defaultdict(list))
        self.max_time: float = 0.0

    def parse_detector_output(self) -> None:
        """Parse SUMO detector output XML file and extract interval data.

        Reads the detector output XML file produced by SUMO and extracts vehicle
        count data for each detector over time. Stores raw interval data indexed
        by detector ID and updates the maximum observed simulation time.
        """
        tree = ET.parse(self.detector_output_path)
        root = tree.getroot()

        for interval in root.findall("interval"):
            det_id = interval.get("id")
            begin_str = interval.get("begin")
            end_str = interval.get("end")

            # skip malformed entries
            if det_id is None or begin_str is None or end_str is None:
                warnings.warn(
                    f"Skipping malformed interval entry in detector output: {ET.tostring(interval, encoding='unicode')}",
                    stacklevel=2,
                )
                continue

            interval_start = float(begin_str)
            interval_end = float(end_str) if end_str is not None else interval_start

            if interval_end < interval_start:
                raise ValueError(
                    f"Invalid measurement data for detector {det_id}: end time {interval_end} is less than begin time {interval_start}."
                )

            count = int(interval.get("nVehEntered", interval.get("nVehContrib", 0)))
            self.detector_intervals[det_id].append((interval_start, count))
            self.max_time = max(self.max_time, interval_end)

    def classify_and_map(self) -> None:
        """Map detector IDs to diverge nodes and edges from CSV specification.

        Reads the detector specification CSV file and creates a mapping between
        detector IDs and their corresponding diverge nodes and edge IDs. Only
        processes detectors of type "turning_rate".

        The method handles various detector ID formats by creating multiple
        variants to ensure robust matching with the detector output data.
        """
        with open(self.detector_spec_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)

            for row in reader:
                det_id = row["detector_id"].strip().strip('"').strip("'")
                det_type = row["type"].strip().lower()

                # only process turning rate detectors
                if "turning_rate" not in det_type:
                    continue

                # create ID variants for robust matching
                det_id_variants = [
                    det_id,
                    det_id.replace("detector_", ""),
                    f"detector_{det_id}",
                ]

                diverge_node_id = (
                    row.get("diverge_node_id", "").strip().strip('"').strip("'")
                )
                edge_id = row["edge_id"].strip().strip('"').strip("'")

                if diverge_node_id and edge_id:
                    # store all variants
                    for variant in det_id_variants:
                        self.detector_mapping[variant] = {
                            "diverge_node_id": diverge_node_id,
                            "edge_id": edge_id,
                            "type": det_type,
                        }

    def aggregate_spatially(self) -> None:
        """Aggregate lane-level detector counts into edge-level counts per diverge node.

        Sums vehicle counts from all lane detectors on the same edge at the same
        diverge node and organizes them into time intervals. This spatial aggregation
        consolidates multi-lane detector data into single edge-level measurements
        suitable for computing turning rates.

        The aggregation preserves raw time intervals from SUMO detector output for
        use in rolling window temporal aggregation.
        """
        for det_id, intervals in self.detector_intervals.items():
            if det_id not in self.detector_mapping:
                continue

            diverge_node_id = self.detector_mapping[det_id]["diverge_node_id"]
            edge_id = self.detector_mapping[det_id]["edge_id"]

            # store raw intervals for rolling window aggregation
            for begin, count in intervals:
                self.diverge_link_intervals[diverge_node_id][edge_id].append(
                    (begin, count)
                )

    def compute_turning_rates(
        self,
    ) -> dict[str, Callable[[float], dict[str, float]]]:
        """Compute time-varying turning rate functions for all diverge nodes.

        Creates callable functions for each diverge node that return the fraction
        of vehicles using each outgoing edge as a function of time using rolling
        window aggregation. Nodes with no vehicles detected across the entire
        simulation are excluded (returning None internally), allowing fallback to
        lane-based splits.

        Returns:
            Dictionary mapping diverge node IDs to turning rate functions.
            Each function takes time in hours and returns a dictionary mapping
            edge IDs to their split ratios (fractions between 0 and 1).
            Nodes with no vehicle observations are excluded from the dictionary.
        """
        turning_rates: dict[str, Callable[[float], dict[str, float]]] = {}

        for diverge_node_id, link_intervals in self.diverge_link_intervals.items():
            # check if there's any vehicle data for this node
            total_vehicles = sum(
                sum(count for _, count in intervals)
                for intervals in link_intervals.values()
            )

            if total_vehicles == 0:
                # skip this node - will fall back to lane-based splits
                warnings.warn(
                    f"No vehicles detected at diverge node {diverge_node_id}. "
                    "Falling back to lane-based splits for this node.",
                    stacklevel=2,
                )
                continue

            # create the turning rate function for this diverge node
            turning_rate_fn = self._make_turning_rate_function(
                link_intervals=link_intervals
            )

            if turning_rate_fn is not None:
                turning_rates[diverge_node_id] = turning_rate_fn

        return turning_rates

    def _make_turning_rate_function(
        self, link_intervals: Mapping[str, list[Tuple[float, int]]]
    ) -> Callable[[float], dict[str, float]] | None:
        """Create a time-varying turning rate function using rolling window aggregation.

        Constructs a callable function that computes turning rates from detector
        observations using a rolling time window. At query time t, the function
        aggregates all vehicle counts within [t - window/2, t + window/2].
        If the window extends beyond the simulation horizon, it is shifted to
        fit within [0, max_time].

        Args:
            link_intervals: Dictionary mapping edge IDs to lists of (begin_time, count) tuples.

        Returns:
            A callable function that takes time in hours and returns a dictionary
            mapping edge IDs to their split ratios (fractions between 0 and 1).
            Returns None if no vehicles were detected on any edge.
        """
        return make_rolling_window_aggregator(
            intervals=link_intervals,
            window_size_sec=self.window_size_sec,
            max_time=self.max_time,
            aggregation_type="rate",
        )

    def run(
        self,
    ) -> dict[str, Callable[[float], dict[str, float]]]:
        """Execute the complete turning rate aggregation pipeline.

        Orchestrates the full workflow: parsing detector outputs, mapping
        detectors to diverge nodes and edges, performing spatial aggregation,
        and computing time-varying turning rate functions.

        Returns:
            Dictionary mapping diverge node IDs to turning rate functions.
            Each function takes time in hours and returns a dictionary mapping
            edge IDs to their split ratios.
        """
        self.parse_detector_output()
        self.classify_and_map()
        self.aggregate_spatially()

        turning_rates = self.compute_turning_rates()

        # print summary statistics
        all_detector_vehicles = sum(
            sum(count for _, count in intervals)
            for link_intervals in self.diverge_link_intervals.values()
            for intervals in link_intervals.values()
        )

        print("TURNING RATE AGGREGATION SUMMARY:")
        print(f"  Total diverge nodes: {len(self.diverge_link_intervals)}")
        print(f"  Total detector vehicles: {all_detector_vehicles}")
        print(f"  Diverge nodes with turning rates: {len(turning_rates)}")
        print(
            f"  Nodes falling back to lane-based: {len(self.diverge_link_intervals) - len(turning_rates)}"
        )

        return turning_rates
