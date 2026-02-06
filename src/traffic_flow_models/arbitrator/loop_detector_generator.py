import xml.etree.ElementTree as ET
import csv


class LoopDetectorGenerator:

    def __init__(
        self,
        sumo_network_path,
        metadata,
        output_dir="results/zurich",
        detection_freq=900,
        detector_filename="detector.xml",
        spec_filename="_detectors_spec.csv",
        output_xml_filename="detectors_output.xml",
    ):

        self.sumo_network_path = sumo_network_path
        self.metadata = metadata
        self.output_dir = output_dir
        self.detection_freq = detection_freq
        self.detector_filename = detector_filename
        self.spec_filename = spec_filename
        self.output_xml_filename = output_xml_filename

        self.backbone_nodes = self._extract_backbone_nodes(metadata)

        self.interface_edges = []
        self.edge_detectors = []

    # Extracts all the nodes from the consolidated network metadata
    def _extract_backbone_nodes(self, metadata):
        backbone = set()

        for oid in metadata.get("origin_ids", []):
            backbone.add(oid.replace("Origin_", ""))

        # Add onramp nodes
        for oid in metadata.get("onramp_ids", []):
            backbone.add(oid.replace("onramp_", ""))

        # Add destination nodes
        for did in metadata.get("destination_ids", []):
            backbone.add(did.replace("Dest_", ""))

        return backbone

    # Finds the points where the juxtaposed macroscopic network meets the microscopic network
    def find_interface_edges(self):

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

            # Urban → Motorway (INFLOW)
            if to_is_backbone and not is_motorway:
                detector_type = "inflow"
                detector_node = to_node
                inflow_count += 1

            # Motorway → Urban (OUTFLOW)
            elif from_is_backbone and not is_motorway:
                detector_type = "outflow"
                detector_node = from_node
                outflow_count += 1

            # Direct backbone interface (ramps connecting to backbone)
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

                    # Place detector near end of edge
                    if lane_length < 10:  # Less than 10 meters
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

    def write_detector_xml(self):
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

    def write_detector_spec_csv(self):
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

    def generate(self):
        self.find_interface_edges()

        detector_xml = self.write_detector_xml()
        detector_csv = self.write_detector_spec_csv()

        return detector_xml, detector_csv
