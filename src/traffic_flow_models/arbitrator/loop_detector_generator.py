import xml.etree.ElementTree as ET
import csv
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
        destination_ids: list[str],
        output_dir: str,
        detection_freq: int = 900,
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
            detection_freq: Measurement frequency in seconds (default: 900).
            detector_filename: Output detector XML filename (default: "detector.xml").
            spec_filename: Output specification CSV filename (default: "_detectors_spec.csv").
            output_xml_filename: Detector output XML filename (default: "detectors_output.xml").
        """

        self.sumo_network_path: str = sumo_network_path
        self.origin_ids: list[str] = origin_ids
        self.onramp_ids: list[str] = onramp_ids
        self.destination_ids: list[str] = destination_ids
        self.output_dir: str = output_dir
        self.detection_freq: int = detection_freq
        self.detector_filename: str = detector_filename
        self.spec_filename: str = spec_filename
        self.output_xml_filename: str = output_xml_filename

        self.backbone_nodes: set[str] = self._extract_backbone_nodes(
            origin_ids, onramp_ids, destination_ids
        )
        self.interface_edges: list = []
        self.edge_detectors: list[dict] = []

    def _extract_backbone_nodes(
        self, origin_ids: list[str], onramp_ids: list[str], destination_ids: list[str]
    ) -> set[str]:
        """Extract all nodes from the consolidated network.

        Identifies nodes that are part of the macroscopic backbone network by
        processing origin, onramp, and destination node IDs.
        These nodes represent the macroscopic network structure.

        Args:
            origin_ids: List of origin node IDs in the network.
            onramp_ids: List of onramp node IDs in the network.
            destination_ids: List of destination node IDs in the network.

        Returns:
            Set of node IDs belonging to the macroscopic backbone network.
        """
        backbone = set()

        for oid in origin_ids:
            backbone.add(oid.replace("origin_", ""))

        # add onramp nodes
        for oid in onramp_ids:
            backbone.add(oid.replace("onramp_", ""))

        # add destination nodes
        for did in destination_ids:
            backbone.add(did.replace("dest_", ""))

        return backbone

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

        for edge in root.findall("edge"):
            if edge.get("function") == "internal":
                continue

            edge_id = edge.get("id")
            edge_type = edge.get("type", "").lower()
            from_node = edge.get("from")
            to_node = edge.get("to")

            is_motorway = "motorway" in edge_type
            to_is_backbone = to_node in self.backbone_nodes
            from_is_backbone = from_node in self.backbone_nodes

            detector_type = None
            detector_node = None

            if edge_id is None:
                raise ValueError("Edge is missing 'id' attribute")

            # urban → motorway (macroscopic network inflow)
            if to_is_backbone and not is_motorway:
                detector_type = "inflow"
                detector_node = to_node
                inflow_count += 1

            # motorway → urban (macroscopic network outflow)
            elif from_is_backbone and not is_motorway:
                detector_type = "outflow"
                detector_node = from_node
                outflow_count += 1

            # direct backbone interface (ramps connecting to backbone)
            elif edge_id.endswith("_link") or "link" in edge_type:
                if to_is_backbone and not from_is_backbone:
                    detector_type = "ramp_inflow"
                    detector_node = to_node
                    inflow_count += 1
                elif from_is_backbone and not to_is_backbone:
                    detector_type = "ramp_outflow"
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

                    # place detector near end of edge
                    if lane_length < 10:  # less than 10 meters
                        print(
                            f"  Skipping short lane {lane_id} (length={lane_length}m)"
                        )
                        continue

                    detector_pos = min(lane_length * 0.9, lane_length - 5)

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
            det_id = f"detector_{det['edge_id']}_{det['lane_index']}"

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
                ],
            )
            writer.writeheader()

            for det in self.edge_detectors:
                det_id = f"detector_{det['edge_id']}_{det['lane_index']}"

                writer.writerow(
                    {
                        "detector_id": det_id,
                        "type": det["type"],
                        "from": det["from_node"],
                        "to": det["to_node"],
                        "edge_id": det["edge_id"],
                        "backbone_node": det["node_id"],
                    }
                )

        return output_file

    def generate(self) -> Tuple[str, str]:
        """Execute the complete detector generation pipeline.

        Orchestrates the full workflow: finding interface edges, generating
        detector configurations, and writing both the SUMO XML file and
        the specification CSV file.

        Returns:
            A tuple containing:
                - detector_xml: Path to the generated detector XML file.
                - detector_csv: Path to the generated specification CSV file.
        """
        self.find_interface_edges()
        detector_xml = self.write_detector_xml()
        detector_csv = self.write_detector_spec_csv()

        return detector_xml, detector_csv
