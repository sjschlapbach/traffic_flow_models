import xml.etree.ElementTree as ET
import csv
import math
from typing import Tuple


class LoopDetectorGenerator:
    """Generate loop detectors at macroscopic-microscopic network interface points.

    This class identifies interface edges between the consolidated macroscopic network
    and the detailed microscopic SUMO network, then generates loop detector
    configurations to measure traffic flow at these boundaries. Detectors are
    placed strategically to capture inflow and outflow at network interfaces.

    Attributes:
        sumo_network_path: Path to the SUMO network XML file.
        origin_ids: List of origin node IDs in the network.
        onramp_ids: List of onramp node IDs in the network.
        destination_ids: List of destination node IDs in the network.
        output_dir: Directory for output files.
        detection_freq: Detector measurement frequency in seconds.
        detector_filename: Name of the detector configuration XML file.
        spec_filename: Name of the detector specification CSV file.
        output_xml_filename: Name of the detector output XML file.
        backbone_nodes: Set of nodes belonging to the macroscopic backbone.
        interface_edges: List of edges at the network interface.
        edge_detectors: List of detector specifications.
    """

    def __init__(
        self,
        sumo_network_path: str,
        origin_ids: list[str],
        onramp_ids: list[str],
        offramp_ids: list[str],
        destination_ids: list[str],
        output_dir: str,
        diverge_node_info: dict[str, list[str]],
        backbone_node_ids: set[str],
        target_cell_length_km: float = 0.3,
        detection_freq: int = 15,
        detector_filename: str = "detector.xml",
        spec_filename: str = "_detectors_spec.csv",
        output_xml_filename: str = "detectors_output.xml",
    ):
        """Initialize the loop detector generator.

        Args:
            sumo_network_path: Path to the SUMO network XML file.
            origin_ids: List of origin node IDs in the network.
            onramp_ids: List of onramp node IDs in the network.
            destination_ids: List of destination node IDs in the network.
            output_dir: Directory where output files will be written.
            diverge_node_info: Dictionary mapping diverge node IDs to lists of SUMO edge IDs.
            detection_freq: Measurement frequency in seconds (default: 15).
            detector_filename: Output detector XML filename (default: "detector.xml").
            spec_filename: Output specification CSV filename (default: "_detectors_spec.csv").
            output_xml_filename: Detector output XML filename (default: "detectors_output.xml").
        """

        self.sumo_network_path: str = sumo_network_path
        self.origin_ids: list[str] = origin_ids
        self.onramp_ids: list[str] = onramp_ids
        self.destination_ids: list[str] = destination_ids
        self.output_dir: str = output_dir
        self.diverge_node_info: dict[str, list[str]] = diverge_node_info or {}
        self.detection_freq: int = detection_freq
        self.detector_filename: str = detector_filename
        self.spec_filename: str = spec_filename
        self.output_xml_filename: str = output_xml_filename

        self.backbone_nodes = backbone_node_ids
        self.target_cell_length_km = target_cell_length_km
        self.interface_edges: list = []
        self.edge_detectors: list[dict] = []

    def find_interface_edges(self) -> Tuple[int, int]:
        """Find interface points between macroscopic and microscopic networks.

        Identifies edges where the macroscopic backbone network interfaces with
        the detailed microscopic SUMO network. Classifies interface edges as
        inflow (entering backbone), outflow (leaving backbone), or ramp
        connections. Places detectors on each lane of interface edges.

        Returns:
            A tuple containing:
                - inflow_count: Number of inflow detectors created.
                - outflow_count: Number of outflow detectors created.
        """

        tree = ET.parse(self.sumo_network_path)
        root = tree.getroot()

        inflow_count = 0
        outflow_count = 0

        inflow_boundary_nodes = self.backbone_nodes | set(self.onramp_ids)
        onramp_nodes = set(self.onramp_ids)

        for edge in root.findall("edge"):
            if edge.get("function") == "internal":
                continue

            edge_id = edge.get("id")
            edge_type = edge.get("type", "").lower()
            from_node = edge.get("from")
            to_node = edge.get("to")

            to_is_backbone = to_node in inflow_boundary_nodes
            from_is_backbone = from_node in inflow_boundary_nodes

            detector_type = None
            detector_node = None

            if edge_id is None:
                raise ValueError("Edge is missing 'id' attribute")

            is_motorway_mainline = edge_type == "motorway"
            is_motorway_link = "motorway_link" in edge_type
            is_backbone_edge = is_motorway_mainline or is_motorway_link

            # urban road entering backbone (mainline origin interface)
            if to_is_backbone and not is_backbone_edge:
                detector_type = "inflow"
                detector_node = to_node
                inflow_count += 1

            # motorway_link entering an onramp node — treat as inflow too
            elif to_node in onramp_nodes and is_motorway_link:
                detector_type = "inflow"
                detector_node = to_node
                inflow_count += 1

            # backbone exiting to urban road (mainline destination interface)
            elif from_is_backbone and not is_backbone_edge:
                detector_type = "outflow"
                detector_node = from_node
                outflow_count += 1

            # motorway_link ramp connections
            elif is_motorway_link:
                if to_is_backbone and not from_is_backbone:
                    detector_type = "ramp_inflow"  # onramp merging onto mainline
                    detector_node = to_node
                    inflow_count += 1
                elif from_is_backbone and not to_is_backbone:
                    detector_type = "ramp_outflow"  # offramp leaving mainline
                    detector_node = from_node
                    outflow_count += 1

            if detector_type:
                lanes = edge.findall("lane")
                for lane_idx, lane in enumerate(lanes):
                    lane_id = lane.get("id")
                    length_str = lane.get("length")
                    if length_str is None:
                        continue
                    lane_length = float(length_str)

                    detector_pos = min(lane_length * 0.9, max(1, lane_length - 5))

                    self.edge_detectors.append(
                        {
                            "edge_id": edge_id,
                            "lane_id": lane_id,
                            "lane_index": lane_idx,
                            "position": detector_pos,
                            "node_id": detector_node,
                            "type": detector_type,
                            "from_node": from_node,
                            "to_node": to_node,
                        }
                    )
        return inflow_count, outflow_count

    """
    def add_detectors_backbone_network(self) -> int:
        segment_detector_count = 0
        ramp_edges = set()

        tree = ET.parse(self.sumo_network_path)
        root = tree.getroot()

        for edge in root.findall("edge"):
            if edge.get("function") == "internal":
                continue
            from_node = edge.get("from")
            to_node = edge.get("to")
            edge_id = edge.get("id")

            if (
                edge_id in ramp_edges
                or from_node not in self.backbone_nodes
                or to_node not in self.backbone_nodes
            ):
                continue

            lanes = edge.findall("lane")
            for lane_idx, lane in enumerate(lanes):
                lane_id = lane.get("id")
                length_str = lane.get("length")
                if length_str is None:
                    continue
                lane_length = float(length_str)

                position = lane_length / 2.0

                self.edge_detectors.append(
                    {
                        "edge_id": edge_id,
                        "lane_id": lane_id,
                        "lane_index": lane_idx,
                        "position": position,
                        "segment_index": 0,
                        "type": "backbone_segment",
                        "from_node": from_node,
                        "to_node": to_node,
                        "node_id": None,
                    }
                )
                segment_detector_count += 1

        return segment_detector_count
    """

    def add_detectors_backbone_network(self) -> int:
        segment_detector_count = 0

        tree = ET.parse(self.sumo_network_path)
        root = tree.getroot()

        for edge in root.findall("edge"):
            if edge.get("function") == "internal":
                continue

            edge_id = edge.get("id")
            from_node = edge.get("from")
            to_node = edge.get("to")

            # only instrument edges that directly connect two macro backbone nodes
            if (
                from_node not in self.backbone_nodes
                or to_node not in self.backbone_nodes
            ):
                continue

            lanes = edge.findall("lane")
            for lane_idx, lane in enumerate(lanes):
                lane_id = lane.get("id")
                length_str = lane.get("length")
                if length_str is None:
                    continue

                lane_length_m = float(length_str)
                lane_length_km = lane_length_m / 1000.0

                # replicate partition_link logic from MotorwayLink
                num_cells = max(
                    1, math.ceil(lane_length_km / self.target_cell_length_km)
                )
                cell_length_m = lane_length_m / num_cells

                for cell_idx in range(num_cells):
                    cell_start_m = cell_idx * cell_length_m
                    # cell_mid_m = cell_start_m + cell_length_m / 2.0
                    actual_length = min(cell_length_m, lane_length_m - cell_start_m)

                    self.edge_detectors.append(
                        {
                            "edge_id": edge_id,
                            "lane_id": lane_id,
                            "lane_index": lane_idx,
                            "position": cell_start_m,  # E2 start position
                            "detector_length": actual_length,  # E2 spans full cell
                            "cell_index": cell_idx,
                            "num_cells": num_cells,
                            "type": "backbone_segment",
                            "from_node": from_node,
                            "to_node": to_node,
                            "node_id": None,
                        }
                    )
                    segment_detector_count += 1

        return segment_detector_count

    def find_turning_rate_edges(self) -> int:
        """Find and place detectors at diverge nodes for turning rate measurement.

        Identifies edges at diverge nodes (nodes with multiple outgoing edges) and
        places detectors at the start of each outgoing edge to measure the number
        of vehicles choosing each path. This data is used to compute time-varying
        turning rates that reflect actual traffic distribution.

        Returns:
            Number of turning rate detectors created.
        """
        if not self.diverge_node_info:
            return 0

        tree = ET.parse(self.sumo_network_path)
        root = tree.getroot()
        turning_rate_count = 0

        # iterate through each diverge node and its outgoing edges
        for diverge_node_id, edge_ids in self.diverge_node_info.items():
            for edge_id in edge_ids:
                # find the edge in the SUMO network
                edge = None
                for e in root.findall("edge"):
                    if e.get("id") == edge_id and e.get("function") != "internal":
                        edge = e
                        break

                if edge is None:
                    continue

                from_node = edge.get("from")
                to_node = edge.get("to")

                # place detector on each lane of this edge
                lanes = edge.findall("lane")
                for lane_idx, lane in enumerate(lanes):
                    lane_id = lane.get("id")
                    length_str = lane.get("length")
                    if length_str is None:
                        continue
                    lane_length = float(length_str)

                    # place detector near start of edge (5m from start)
                    detector_pos = min(5.0, lane_length * 0.1)

                    self.edge_detectors.append(
                        {
                            "edge_id": edge_id,
                            "lane_id": lane_id,
                            "lane_index": lane_idx,
                            "position": detector_pos,
                            "node_id": diverge_node_id,
                            "type": "turning_rate",
                            "from_node": from_node,
                            "to_node": to_node,
                            "diverge_node_id": diverge_node_id,
                        }
                    )
                    turning_rate_count += 1

        return turning_rate_count

    def build_det_id(self, det: dict) -> str:
        """Build a unique detector ID, appending segment_index for backbone detectors."""
        det_type = det.get("type", "interface").replace("_", "")
        base = f"detector_{det_type}_{det['edge_id']}_{det['lane_index']}"
        if "cell_index" in det:
            return f"{base}_cell{det['cell_index']}"
        if "segment_index" in det:
            return f"{base}_{det['segment_index']}"
        return base

    def write_detector_xml(self) -> str:
        """Write SUMO loop detector configuration XML file.

        Generates the SUMO additional file containing induction loop elements
        for all identified interface detectors. Each detector is configured
        with its lane position, measurement frequency, and output file.

        Returns:
            Path to the generated detector XML file.
        """
        output_file = f"{self.output_dir}/{self.detector_filename}"

        root = ET.Element("additional")

        for det in self.edge_detectors:
            det_id = self.build_det_id(det)

            if det["type"] == "backbone_segment":
                # E2 lane area detector
                detector = ET.SubElement(root, "laneAreaDetector")
                detector.set("id", det_id)
                detector.set("lane", det["lane_id"])
                detector.set("pos", f"{det['position']:.2f}")
                detector.set("length", f"{det['detector_length']:.2f}")
                detector.set("freq", str(self.detection_freq))
                detector.set("file", self.output_xml_filename)
            else:
                # keep point detectors for inflow/outflow/turning_rate
                detector = ET.SubElement(root, "inductionLoop")
                detector.set("id", det_id)
                detector.set("lane", det["lane_id"])
                detector.set("pos", f"{det['position']:.2f}")
                detector.set("freq", str(self.detection_freq))
                detector.set("file", self.output_xml_filename)

        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        tree.write(output_file, encoding="utf-8", xml_declaration=True)

        return output_file

    def write_detector_spec_csv(self) -> str:
        """Write detector specification CSV file.

        Creates a CSV file documenting each detector's metadata including
        detector ID, type (inflow/outflow/ramp), edge topology (from/to nodes),
        and associated backbone node. This specification is used by the
        demand aggregator to map detector readings to network nodes.

        Returns:
            Path to the generated detector specification CSV file.
        """
        output_file = f"{self.output_dir}/{self.spec_filename}"

        with open(output_file, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "detector_id",
                    "type",
                    "from",
                    "to",
                    "edge_id",
                    "backbone_node",
                    "diverge_node_id",
                    "position",
                    "cell_index",
                    "detector_length",
                ],
            )
            writer.writeheader()

            for det in self.edge_detectors:
                # include type in detector ID to avoid conflicts between interface and turning rate detectors
                det_type = det.get("type", "interface").replace("_", "")
                # det_id = f"detector_{det_type}_{det['edge_id']}_{det['lane_index']}"
                det_id = self.build_det_id(det)

                writer.writerow(
                    {
                        "detector_id": det_id,
                        "type": det["type"],
                        "from": det["from_node"],
                        "to": det["to_node"],
                        "edge_id": det["edge_id"],
                        "backbone_node": det["node_id"],
                        "diverge_node_id": det.get("diverge_node_id", ""),
                        "position": det.get("position", ""),
                        "cell_index": det.get("cell_index", ""),
                        "detector_length": det.get("detector_length", ""),
                    }
                )

        return output_file

    def generate(self) -> Tuple[str, str, str]:
        """Execute the complete detector generation pipeline.

        Orchestrates the full workflow: finding interface edges, finding turning
        rate edges at diverge nodes, generating detector configurations, and writing
        both the SUMO XML file and the specification CSV file.

        Returns:
            A tuple containing:
                - detector_xml: Path to the generated detector definition XML file.
                - output_xml: Path to the detector output XML file (where SUMO writes results).
                - detector_csv: Path to the generated specification CSV file.
        """
        inflow_count, outflow_count = self.find_interface_edges()
        turning_rate_count = self.find_turning_rate_edges()
        backbone_detector_count = self.add_detectors_backbone_network()

        print(f"Detector placement summary:")
        print(f"  Inflow detectors: {inflow_count}")
        print(f"  Outflow detectors: {outflow_count}")
        print(f"  Turning rate detectors: {turning_rate_count}")
        print(f"  Backbone detectors: {backbone_detector_count}")
        print(f"  Total detectors: {len(self.edge_detectors)}")

        detector_xml = self.write_detector_xml()
        detector_csv = self.write_detector_spec_csv()
        output_xml = f"{self.output_dir}/{self.output_xml_filename}"

        return detector_xml, output_xml, detector_csv
