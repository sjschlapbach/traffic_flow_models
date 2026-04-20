import xml.etree.ElementTree as ET
import csv
import math
from typing import Tuple


class LoopDetectorGenerator:
    # """Generate SUMO detectors for the macro–micro interface and backbone links.

    # This class inspects a SUMO network (.net.xml) and produces two kinds of
    # detectors:
    # - point induction loops (`inductionLoop`) used for interface/turning-rate
    #     measurements, and
    # - lane-area detectors (`laneAreaDetector`) placed along backbone motorway
    #     links that represent macroscopic cells for state aggregation.

    # The generator tracks processed ``(lane_id, role)`` pairs so a single lane
    # can receive both an interface detector and one or more backbone cell
    # detectors without duplication.

    # Methods of interest:
    # - find_interface_edges(): detect and classify interface detectors
    # - add_detectors_backbone_network(): create backbone cell detectors
    # - find_turning_rate_edges(): add detectors to diverge outgoing edges
    # - write_detector_xml(): write SUMO additional XML with detector defs
    # - write_detector_spec_csv(): write a CSV mapping detectors to edges/nodes
    # """

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
        target_cell_length_km: float,
        motorway_links: list,
        detection_freq: int = 15,
        detector_filename: str = "detector.xml",
        spec_filename: str = "_detectors_spec.csv",
        output_xml_filename: str = "detectors_output.xml",
    ):
        self.sumo_network_path: str = sumo_network_path
        self.origin_ids: list[str] = origin_ids
        self.onramp_ids: list[str] = onramp_ids
        self.offramp_ids: list[str] = offramp_ids
        self.destination_ids: list[str] = destination_ids
        self.output_dir: str = output_dir
        self.diverge_node_info: dict[str, list[str]] = diverge_node_info or {}
        self.detection_freq: int = detection_freq
        self.detector_filename: str = detector_filename
        self.spec_filename: str = spec_filename
        self.output_xml_filename: str = output_xml_filename

        self.backbone_nodes = backbone_node_ids
        self.target_cell_length_km = target_cell_length_km
        self.motorway_links = motorway_links

        self.interface_edges: list = []
        self.edge_detectors: list[dict] = []

        # Track processed lanes per role, not globally.
        # This allows one lane to have both:
        # - an interface detector
        # - one or more backbone cell detectors
        self.processed_lane_roles: set[tuple[str, str]] = set()

    def _mark_lane_role(self, lane_id: str, role: str) -> bool:
        """Return True if this (lane, role) is new and should be processed."""
        key = (lane_id, role)
        if key in self.processed_lane_roles:
            return False
        self.processed_lane_roles.add(key)
        return True

    def _offramp_downstream_nodes(self, root) -> set[str]:
        """Nodes reachable from offramp exits via non-motorway edges (post-exit routes)."""
        import networkx as nx

        urban_graph = nx.DiGraph()
        for edge in root.findall("edge"):
            if edge.get("function") == "internal":
                continue
            if "motorway" in edge.get("type", "").lower():
                continue
            fn, tn = edge.get("from"), edge.get("to")
            if fn and tn:
                urban_graph.add_edge(fn, tn)

        downstream: set[str] = set()
        for offramp_node in self.offramp_ids:
            if urban_graph.has_node(offramp_node):
                reachable = nx.single_source_shortest_path_length(
                    urban_graph, offramp_node, cutoff=8
                )
                downstream.update(reachable.keys())
        return downstream

    def find_interface_edges(self) -> Tuple[int, int]:
        """Find interface detectors and classify them without colliding with backbone cells.

        Detector types:
        - mainline_origin_interface: motorway mainline entering the modeled backbone
        - inflow: urban or ramp inflow entering the modeled backbone
        - outflow: traffic leaving the modeled backbone toward non-backbone roads
        """
        tree = ET.parse(self.sumo_network_path)
        root = tree.getroot()

        inflow_count = 0
        outflow_count = 0

        # offramp nodes are now included so outflow detectors are placed at
        # offramp exits even when they are not backbone nodes.
        boundary_nodes = (
            self.backbone_nodes | set(self.onramp_ids) | set(self.offramp_ids)
        )
        offramp_downstream = self._offramp_downstream_nodes(root)

        for edge in root.findall("edge"):
            if edge.get("function") == "internal":
                continue

            edge_id = edge.get("id")
            edge_type = edge.get("type", "").lower()
            from_node = edge.get("from")
            to_node = edge.get("to")

            to_is_boundary = to_node in boundary_nodes
            from_is_boundary = from_node in boundary_nodes

            is_motorway_mainline = "motorway" in edge_type and "link" not in edge_type
            is_motorway_link = "motorway_link" in edge_type
            is_backbone_road = is_motorway_mainline or is_motorway_link
            onramp_node_set = set(self.onramp_ids)

            detector_type = None
            detector_node = None

            if to_is_boundary and is_motorway_mainline:
                detector_type = "mainline_origin_interface"
                detector_node = to_node
                inflow_count += 1

            elif is_motorway_link and from_node in onramp_node_set:
                detector_type = "inflow"
                detector_node = from_node
                inflow_count += 1

            # elif to_is_boundary and (not is_backbone_road):
            elif (
                to_is_boundary
                and not is_backbone_road
                and (from_node not in boundary_nodes)
            ):
                # elif (to_is_boundary and not is_backbone_road and from_node not in boundary_nodes and from_node not in offramp_downstream):
                detector_type = "inflow"
                detector_node = to_node
                inflow_count += 1

            # elif from_is_boundary and not is_backbone_road:
            elif (
                from_is_boundary
                and not is_backbone_road
                and (to_node not in boundary_nodes)
            ):
                detector_type = "outflow"
                detector_node = from_node
                outflow_count += 1

            if detector_type is None:
                continue

            for lane_idx, lane in enumerate(edge.findall("lane")):
                lane_id = lane.get("id")
                if lane_id is None:
                    continue

                if not self._mark_lane_role(lane_id, detector_type):
                    continue

                length_str = lane.get("length")
                if length_str is None:
                    continue

                lane_length = float(length_str)
                # Place detector near the downstream end of the lane but ensure
                # it is at least 1 m from the start and at most 5 m from the end
                # (or 90% of lane length for very short lanes).
                # detector_pos = min(lane_length * 0.9, max(1.0, lane_length - 5.0))
                if detector_type == "outflow":
                    detector_pos = min(lane_length * 0.9, max(1.0, lane_length - 5.0))
                else:  # inflow, mainline_origin_interface
                    detector_pos = min(5.0, lane_length * 0.1)

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
                        "detector_length": 5.0,
                    }
                )

        return inflow_count, outflow_count

    def add_detectors_backbone_network(self) -> int:
        """Place E2 area detectors along each consolidated motorway link as true cells."""
        tree = ET.parse(self.sumo_network_path)
        root = tree.getroot()

        sumo_edges = {
            e.get("id"): e
            for e in root.findall("edge")
            if e.get("function") != "internal" and e.get("id") is not None
        }

        segment_detector_count = 0

        for link in self.motorway_links:
            edge = sumo_edges.get(link.id)
            if edge is None:
                raise ValueError(
                    f"[add_detectors_backbone_network] MotorwayLink '{link.id}' "
                    f"not found in SUMO network XML. "
                    f"Check that the consolidated network was built from this SUMO file."
                )

            from_node = edge.get("from")
            to_node = edge.get("to")

            for lane_idx, lane in enumerate(edge.findall("lane")):
                lane_id = lane.get("id")
                if lane_id is None:
                    continue

                if not self._mark_lane_role(lane_id, "backbone_segment"):
                    continue

                length_str = lane.get("length")
                if length_str is None:
                    continue

                lane_length_m = float(length_str)
                lane_length_km = lane_length_m / 1000.0
                # Determine number of macroscopic cells by dividing the lane
                # length by the target cell length (in km). Always keep at
                # least one cell per lane.
                num_cells = max(
                    1, math.floor(lane_length_km / self.target_cell_length_km)
                )
                cell_length_m = lane_length_m / num_cells

                for cell_idx in range(num_cells):
                    cell_start_m = cell_idx * cell_length_m
                    actual_length = min(cell_length_m, lane_length_m - cell_start_m)
                    cell_key = f"{link.id}_cell{cell_idx}"

                    self.edge_detectors.append(
                        {
                            "edge_id": link.id,
                            "lane_id": lane_id,
                            "lane_index": lane_idx,
                            "position": cell_start_m,
                            "detector_length": actual_length,
                            "cell_index": cell_idx,
                            "cell_key": cell_key,
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
        """Place detectors at diverge-node outgoing edges for turning-rate measurement."""
        if not self.diverge_node_info:
            return 0

        tree = ET.parse(self.sumo_network_path)
        root = tree.getroot()
        turning_rate_count = 0

        for diverge_node_id, edge_ids in self.diverge_node_info.items():
            for edge_id in edge_ids:
                edge = None
                for e in root.findall("edge"):
                    if e.get("id") == edge_id and e.get("function") != "internal":
                        edge = e
                        break

                if edge is None:
                    continue

                from_node = edge.get("from")
                to_node = edge.get("to")

                for lane_idx, lane in enumerate(edge.findall("lane")):
                    lane_id = lane.get("id")
                    if lane_id is None:
                        continue

                    if not self._mark_lane_role(lane_id, "turning_rate"):
                        continue

                    length_str = lane.get("length")
                    if length_str is None:
                        continue

                    lane_length = float(length_str)
                    # Place a small point detector near the diverge edge. Use
                    # the lesser of 5 m and 10% of lane length to avoid placing
                    # detectors outside short lanes.
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
        """Build a unique detector ID.

        Rules:
        - backbone_segment detectors include cell index
        - interface and turning-rate detectors never use cell indexing
        """
        det_type = det.get("type", "interface").replace("_", "")
        base = f"detector_{det_type}_{det['edge_id']}_{det['lane_index']}"

        if det.get("type") == "backbone_segment":
            return f"{base}_cell{det['cell_index']}"

        return base

    # def write_detector_xml(self) -> str:
    #     output_file = f"{self.output_dir}/{self.detector_filename}"
    #     root = ET.Element("additional")

    #     for det in self.edge_detectors:
    #         det_id = self.build_det_id(det)

    #         if det["type"] == "backbone_segment":
    #             detector = ET.SubElement(root, "laneAreaDetector")
    #             detector.set("id", det_id)
    #             detector.set("lane", det["lane_id"])
    #             detector.set("pos", f"{det['position']:.2f}")
    #             detector.set("length", f"{det['detector_length']:.2f}")
    #             detector.set("freq", str(self.detection_freq))
    #             detector.set("file", self.output_xml_filename)
    #         else:
    #             detector = ET.SubElement(root, "inductionLoop")
    #             detector.set("id", det_id)
    #             detector.set("lane", det["lane_id"])
    #             detector.set("pos", f"{det['position']:.2f}")
    #             detector.set("freq", str(self.detection_freq))
    #             detector.set("file", self.output_xml_filename)
    #             if det["type"] in {"inflow", "outflow"}:
    #                 detector.set("vTypes", "urban")

    #     tree = ET.ElementTree(root)
    #     ET.indent(tree, space="  ")
    #     tree.write(output_file, encoding="utf-8", xml_declaration=True)

    #     return output_file

    def write_detector_xml(self) -> str:
        """Write SUMO additional XML with detector definitions.

        Detector vTypes logic
        ─────────────────────
        backbone_segment      → laneAreaDetector, no vTypes filter (counts all)
        mainline_origin_interface → inductionLoop, no vTypes filter (counts all)
        turning_rate          → inductionLoop, no vTypes filter (counts all)
        inflow / outflow      → inductionLoop, vTypes="urban" (counts ONLY urban-typed vehicles)

        For this to work correctly the route file MUST:
        1. define  <vType id="urban" vClass="passenger" ... />
        2. assign  type="urban" to every vehicle whose origin is an onramp node.
        See write_vtype_additional() below for how to emit the vType definition as
        a standalone additional file (load it before detector.xml in sumocfg).
        """
        output_file = f"{self.output_dir}/{self.detector_filename}"
        root = ET.Element("additional")

        # Types that should only observe urban (ramp-entering) vehicles.
        URBAN_FILTERED_TYPES = {"inflow", "outflow"}

        # Types that use laneAreaDetectors instead of inductionLoops.
        AREA_DETECTOR_TYPES = {"backbone_segment"}

        for det in self.edge_detectors:
            det_id = self.build_det_id(det)
            det_type = det["type"]

            if det_type in AREA_DETECTOR_TYPES:
                detector = ET.SubElement(root, "laneAreaDetector")
                detector.set("id", det_id)
                detector.set("lane", det["lane_id"])
                detector.set("pos", f"{det['position']:.2f}")
                detector.set("length", f"{det['detector_length']:.2f}")
                detector.set("freq", str(self.detection_freq))
                detector.set("file", self.output_xml_filename)
                # No vTypes — backbone detectors must see every vehicle.

            else:
                detector = ET.SubElement(root, "inductionLoop")
                detector.set("id", det_id)
                detector.set("lane", det["lane_id"])
                detector.set("pos", f"{det['position']:.2f}")
                detector.set("freq", str(self.detection_freq))
                detector.set("file", self.output_xml_filename)

                if det_type in URBAN_FILTERED_TYPES:
                    # Only count vehicles explicitly typed "urban" in the route file.
                    # mainline_origin_interface and turning_rate detectors intentionally
                    # have NO filter here — they must observe all vehicle types.
                    detector.set("vTypes", "urban")

        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        tree.write(output_file, encoding="utf-8", xml_declaration=True)
        return output_file

    def write_vtype_additional(self, filename: str = "vtypes.xml") -> str:
        """Emit a SUMO additional file that defines the 'urban' vehicle type.

        This must be listed before detector.xml in the sumocfg additional-files
        attribute so SUMO recognises the type before evaluating vTypes filters.

        The parameters below mirror the SUMO passenger car defaults — adjust
        maxSpeed / accel / decel / length to match your calibration.
        """
        output_file = f"{self.output_dir}/{filename}"
        root = ET.Element("additional")

        vtype = ET.SubElement(root, "vType")
        vtype.set("id", "urban")
        vtype.set("vClass", "passenger")
        vtype.set("maxSpeed", "50")  # m/s — ~180 km/h cap, typical for urban
        vtype.set("accel", "2.6")
        vtype.set("decel", "4.5")
        vtype.set("length", "5.0")
        vtype.set("minGap", "2.5")
        vtype.set("sigma", "0.5")  # Krauss driver imperfection

        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        tree.write(output_file, encoding="utf-8", xml_declaration=True)
        return output_file

    def write_detector_spec_csv(self) -> str:
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
                    "cell_key",
                    "detector_length",
                ],
            )
            writer.writeheader()

            for det in self.edge_detectors:
                det_id = self.build_det_id(det)

                writer.writerow(
                    {
                        "detector_id": det_id,
                        "type": det["type"],
                        "from": det["from_node"],
                        "to": det["to_node"],
                        "edge_id": det["edge_id"],
                        "backbone_node": det.get("node_id", ""),
                        "diverge_node_id": det.get("diverge_node_id", ""),
                        "position": det.get("position", ""),
                        "cell_index": det.get("cell_index", ""),
                        "cell_key": det.get("cell_key", ""),
                        "detector_length": det.get("detector_length", ""),
                    }
                )

        return output_file

    def generate(self) -> Tuple[str, str, str]:
        inflow_count, outflow_count = self.find_interface_edges()
        turning_rate_count = self.find_turning_rate_edges()
        backbone_detector_count = self.add_detectors_backbone_network()

        print("Detector placement summary:")
        print(f"  Inflow detectors: {inflow_count}")
        print(f"  Outflow detectors: {outflow_count}")
        print(f"  Turning rate detectors: {turning_rate_count}")
        print(f"  Backbone detectors: {backbone_detector_count}")
        print(f"  Total detectors: {len(self.edge_detectors)}")

        detector_xml = self.write_detector_xml()
        detector_csv = self.write_detector_spec_csv()
        output_xml = f"{self.output_dir}/{self.output_xml_filename}"

        return detector_xml, output_xml, detector_csv
