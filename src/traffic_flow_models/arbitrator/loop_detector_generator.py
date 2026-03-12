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
        diverge_node_info: dict[str, list[str]],
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

    def add_detectors_backbone_network(self) -> int:
        segment_detector_count = 0
        detector_interval = 10.0  # meters

        # Get backbone nodes (strip prefixes from origin/onramp/destination IDs)
        backbone_nodes = self._extract_backbone_nodes(
        self.origin_ids, 
        self.onramp_ids, 
        self.destination_ids)

        # Identify ramp edges to exclude
        ramp_edges = set()
        for onramp_id in self.onramp_ids:
            # Get the actual node ID (with prefix removed)
            node_id = onramp_id.replace("onramp_", "")
            node = self.net.getNode(node_id)
            if node:
                # Incoming edges to onramp are ramp edges
                for edge in node.getIncoming():
                    ramp_edges.add(edge.getID())
                # Outgoing edges from onramp are ramp edges  
                for edge in node.getOutgoing():
                    ramp_edges.add(edge.getID())
        
        # Iterate through all edges in the network
        for edge in self.net.getEdges():
            edge_id = edge.getID()
            from_node_id = edge.getFromNode().getID()
            to_node_id = edge.getToNode().getID()
            
            # Skip internal edges, ramps, and non-backbone edges
            if (edge.isSpecial() or 
                edge_id in ramp_edges or
                from_node_id not in backbone_nodes or 
                to_node_id not in backbone_nodes):
                continue

        # Get edge length
        edge_length = edge.getLength()
        
        # Calculate detector positions (0m, 10m, 20m, ...)
        positions = []
        current_pos = 0.0
        while current_pos <= edge_length:
            positions.append(current_pos)
            current_pos += detector_interval

        # Place detector on each lane at each position
        num_lanes = edge.getLaneNumber()
        for lane_index in range(num_lanes):
            lane_id = f"{edge_id}_{lane_index}"
            
            for position in positions:
                # Store detector information
                self.edge_detectors.append({
                    'edge_id': edge_id,
                    'lane_id': lane_id,
                    'lane_index': lane_index,
                    'position': position,
                    'type': 'backbone_segment',
                    'from_node': from_node_id,
                    'to_node': to_node_id,
                    'node_id': None,  # Not associated with single backbone node
                })

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

                    # skip very short lanes
                    if lane_length < 10:
                        print(
                            f"  Skipping short lane {lane_id} (length={lane_length}m)"
                        )
                        continue

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
            det_type = det.get("type", "interface").replace("_", "")
            det_id = f"detector_{det_type}_{det['edge_id']}_{det['lane_index']}"

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
                ],
            )
            writer.writeheader()

            for det in self.edge_detectors:
                # include type in detector ID to avoid conflicts between interface and turning rate detectors
                det_type = det.get("type", "interface").replace("_", "")
                det_id = f"detector_{det_type}_{det['edge_id']}_{det['lane_index']}"

                writer.writerow(
                    {
                        "detector_id": det_id,
                        "type": det["type"],
                        "from": det["from_node"],
                        "to": det["to_node"],
                        "edge_id": det["edge_id"],
                        "backbone_node": det["node_id"],
                        "diverge_node_id": det.get("diverge_node_id", ""),
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
