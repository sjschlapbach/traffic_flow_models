import xml.etree.ElementTree as ET
import csv
import math
from typing import Tuple


class LoopDetectorGenerator:
    """Generate SUMO detectors for the macro-micro interface and backbone links.

    This class inspects a SUMO network (.net.xml) and produces two kinds of
    detectors:
    - Point induction loops (`inductionLoop`) used for interface and
      turning-rate measurements.
    - Lane-area detectors (`laneAreaDetector`) placed along backbone motorway
      links that represent macroscopic cells for state aggregation.

    The generator tracks processed ``(lane_id, role)`` pairs so a single lane
    can receive both an interface detector and one or more backbone cell
    detectors without duplication.

    Methods of interest:
        find_interface_edges(): detect and classify interface detectors.
        add_detectors_backbone_network(): create backbone cell detectors.
        find_turning_rate_edges(): add detectors to diverge outgoing edges.
        write_detector_xml(): write SUMO additional XML with detector definitions.
        write_detector_spec_csv(): write a CSV mapping detectors to edges/nodes.
        generate(): run the full pipeline and write all output files.
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
        target_cell_length_km: float,
        motorway_links: list,
        sumo_edge_id_map: "dict[str, list[str]] | None" = None,
        detection_freq: int = 15,
        detector_filename: str = "detector.xml",
        spec_filename: str = "_detectors_spec.csv",
        output_xml_filename: str = "detectors_output.xml",
    ):
        """Initialize the loop detector generator.

        Args:
            sumo_network_path: Path to the SUMO network XML file (.net.xml).
            origin_ids: List of mainline origin node IDs from the macroscopic network.
            onramp_ids: List of onramp source node IDs from the macroscopic network.
            offramp_ids: List of offramp sink node IDs from the macroscopic network.
            destination_ids: List of destination node IDs from the macroscopic network.
            output_dir: Directory where output files will be written.
            diverge_node_info: Mapping from diverge node IDs to lists of outgoing macroscopic
                link IDs (resolved to real SUMO edges via sumo_edge_id_map), used to place
                turning-rate detectors.
            backbone_node_ids: Set of node IDs belonging to the mainline motorway backbone.
            target_cell_length_km: Target macroscopic cell length in kilometres, used to
                divide backbone links into equally-sized detector segments.
            motorway_links: List of MotorwayLink objects representing consolidated backbone
                links. Each link's .id must be resolvable to one or more real SUMO edges
                via sumo_edge_id_map.
            sumo_edge_id_map: Optional mapping from macroscopic link ID to the ordered list
                of original SUMO edge IDs that compose it. Required whenever the network
                arbitrator applied serial-edge merging (merge_serial_edges), because the
                resulting "merged_A_B" IDs do not exist in the .net.xml. If None, each link
                ID is assumed to be a real SUMO edge ID (backward-compatible behaviour).
            detection_freq: Detector aggregation interval in seconds (default: 15).
            detector_filename: Output filename for the SUMO additional XML (default: "detector.xml").
            spec_filename: Output filename for the detector specification CSV
                (default: "_detectors_spec.csv").
            output_xml_filename: Filename for the SUMO detector output XML referenced
                inside the additional file (default: "detectors_output.xml").
        """
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
        self.sumo_edge_id_map: dict[str, list[str]] = dict(sumo_edge_id_map or {})

        self.interface_edges: list = []
        self.edge_detectors: list[dict] = []

        # Track processed lanes per role, not globally.
        # This allows one lane to have both:
        # - an interface detector
        # - one or more backbone cell detectors
        self.processed_lane_roles: set[tuple[str, str]] = set()

        # Mainline origin nodes are macro-graph nodes that serve as network
        # entry points on the motorway backbone (an Origin feeds them and they
        # are NOT onramp source nodes). find_interface_edges() needs this set
        # because merge_serial_edges can absorb the downstream node of a
        # mainline-origin macro link: the original "to_is_boundary" test then
        # never fires on the first constituent SUMO edge, and the aggregator
        # drops the origin for missing detectors. Recognising the origin by
        # its FROM node fixes that.
        self._mainline_origin_nodes: set[str] = {
            oid[len("origin_") :] if oid.startswith("origin_") else oid
            for oid in self.origin_ids
        } - set(self.onramp_ids)

    def _mark_lane_role(self, lane_id: str, role: str) -> bool:
        """Return True if this (lane, role) is new and should be processed."""
        key = (lane_id, role)
        if key in self.processed_lane_roles:
            return False
        self.processed_lane_roles.add(key)
        return True

    def find_interface_edges(self) -> Tuple[int, int]:
        """Find interface detectors and classify them without colliding with backbone cells.

        Detector types assigned per edge:
        - mainline_origin_interface: motorway mainline edge whose downstream node is a
          backbone boundary (mainline entry point).
        - inflow: non-backbone edge (or onramp motorway_link) whose downstream node is a
          backbone boundary (urban or ramp demand entering the backbone).
        - outflow: non-backbone edge whose upstream node is a backbone boundary
          (traffic leaving the backbone toward non-backbone roads).

        Results are appended to self.edge_detectors.

        Returns:
            A tuple (inflow_count, outflow_count) with the number of inflow and
            outflow detector entries added.
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

        # edges between two boundary nodes are intentionally excluded; ramp demand is captured via the motorway_link branch

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

            if is_motorway_mainline and from_node in self._mainline_origin_nodes:
                detector_type = "mainline_origin_interface"
                detector_node = from_node
                inflow_count += 1

            elif is_motorway_link and from_node in onramp_node_set:
                detector_type = "inflow"
                detector_node = from_node
                inflow_count += 1

            elif (
                to_is_boundary
                and to_node not in onramp_node_set
                and not is_backbone_road
                and (from_node not in boundary_nodes)
            ):
                detector_type = "inflow"
                detector_node = to_node
                inflow_count += 1

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
        """Place E2 area detectors along each consolidated motorway link as true cells.

        For each MotorwayLink in self.motorway_links, divides the combined lane length
        across all constituent SUMO edges into evenly-spaced cells based on
        target_cell_length_km, then places one laneAreaDetector per cell per lane. The
        detector for cell i is anchored at the SUMO lane that contains the cell's
        upstream boundary, and its length is truncated to remain within that SUMO lane
        so SUMO accepts the placement.

        This method resolves a macroscopic link ID to its real SUMO edges via
        self.sumo_edge_id_map. Unmerged links resolve to a single SUMO edge and behave
        identically to the original implementation; serial-merged links resolve to an
        ordered list of SUMO edges that together form the macro link.

        Returns:
            Total number of backbone segment detector entries added.

        Raises:
            ValueError: If a constituent SUMO edge of any MotorwayLink is missing from
                the SUMO network XML.
        """
        tree = ET.parse(self.sumo_network_path)
        root = tree.getroot()

        sumo_edges = {
            e.get("id"): e
            for e in root.findall("edge")
            if e.get("function") != "internal" and e.get("id") is not None
        }

        segment_detector_count = 0

        for link in self.motorway_links:
            # Resolve the macro link ID to the ordered list of real SUMO edge
            # IDs that compose it. For unmerged links this is [link.id]; for
            # serial-merged links it's the original upstream-to-downstream
            # sequence. Fall back to [link.id] so callers that skip the map
            # (backward compatibility) still work on unmerged networks.
            sumo_ids = self.sumo_edge_id_map.get(link.id, [link.id])

            constituent_edges = []
            for sid in sumo_ids:
                e = sumo_edges.get(sid)
                if e is None:
                    raise ValueError(
                        f"[add_detectors_backbone_network] Constituent SUMO edge '{sid}' "
                        f"of MotorwayLink '{link.id}' not found in SUMO network XML. "
                        f"Check that sumo_edge_id_map was built from this SUMO file."
                    )
                constituent_edges.append(e)

            if not constituent_edges:
                continue

            # merge_serial_edges only merges edges with equal lane counts, so
            # every constituent edge has the same number of lanes. Use the
            # first edge as the reference.
            num_lanes = len(constituent_edges[0].findall("lane"))

            for lane_idx in range(num_lanes):
                # Build ordered list of lane segments (one entry per
                # constituent SUMO edge) for this lane index.
                lane_segments: list[tuple[str, float, str, str, str]] = []
                for e in constituent_edges:
                    lanes = e.findall("lane")
                    if lane_idx >= len(lanes):
                        continue
                    lane = lanes[lane_idx]
                    lane_id = lane.get("id")
                    length_str = lane.get("length")
                    if lane_id is None or length_str is None:
                        continue
                    lane_segments.append(
                        (
                            lane_id,
                            float(length_str),
                            e.get("id"),
                            e.get("from"),
                            e.get("to"),
                        )
                    )

                if not lane_segments:
                    continue

                total_lane_len_m = sum(seg[1] for seg in lane_segments)
                total_lane_len_km = total_lane_len_m / 1000.0

                # Keep this cell count consistent with the macroscopic cell
                # division performed by NetworkArbitrator.instantiate_network
                # so detector cell indices line up with model cell indices.
                num_cells = max(
                    1,
                    math.floor(total_lane_len_km / self.target_cell_length_km),
                )
                cell_len_m = total_lane_len_m / num_cells

                for cell_idx in range(num_cells):
                    cell_start_global = cell_idx * cell_len_m
                    cell_end_global = cell_start_global + cell_len_m

                    # Walk the lane segments to find which SUMO segment
                    # contains the cell's upstream boundary. Place the
                    # laneAreaDetector there, truncating its length so the
                    # detector stays within that SUMO lane.
                    cum = 0.0
                    for (
                        seg_lane_id,
                        seg_len,
                        seg_edge_id,
                        seg_from,
                        seg_to,
                    ) in lane_segments:
                        seg_start = cum
                        seg_end = cum + seg_len
                        cum = seg_end

                        if cell_start_global >= seg_end - 1e-9:
                            continue

                        local_pos = max(0.0, cell_start_global - seg_start)
                        available = seg_len - local_pos
                        actual_length = min(
                            cell_len_m,
                            cell_end_global - cell_start_global,
                            available,
                        )
                        if actual_length <= 0.0:
                            break

                        # Role is unique per cell so we can place multiple
                        # backbone cell detectors on the same SUMO lane (large
                        # SUMO edges commonly hold several cells).
                        role = f"backbone_segment_cell{cell_idx}"
                        if not self._mark_lane_role(seg_lane_id, role):
                            break

                        cell_key = f"{link.id}_cell{cell_idx}"
                        self.edge_detectors.append(
                            {
                                # Use the macroscopic link ID so the backbone
                                # aggregator can group detectors by macro link.
                                "edge_id": link.id,
                                "lane_id": seg_lane_id,
                                "lane_index": lane_idx,
                                "position": local_pos,
                                "detector_length": actual_length,
                                "cell_index": cell_idx,
                                "cell_key": cell_key,
                                "num_cells": num_cells,
                                "type": "backbone_segment",
                                "from_node": seg_from,
                                "to_node": seg_to,
                                "node_id": None,
                            }
                        )
                        segment_detector_count += 1
                        break

        return segment_detector_count

    def find_turning_rate_edges(self) -> int:
        """Place detectors at diverge-node outgoing edges for turning-rate measurement.

        Iterates over self.diverge_node_info and appends one inductionLoop entry per
        lane per outgoing edge to self.edge_detectors. Each detector is positioned
        near the start of the lane (at most 5 m or 10% of lane length).

        Returns:
            Total number of turning-rate detector entries added.
        """
        if not self.diverge_node_info:
            return 0

        tree = ET.parse(self.sumo_network_path)
        root = tree.getroot()
        turning_rate_count = 0

        for diverge_node_id, edge_ids in self.diverge_node_info.items():
            for edge_id in edge_ids:
                # diverge_node_info values are macroscopic link IDs. For merged
                # links these look like "merged_A_B" and no such edge exists in
                # the SUMO .net.xml, so we must resolve to the real SUMO edge
                # that actually leaves the diverge node — which is the FIRST
                # constituent SUMO edge (upstream-most) of the macro link.
                resolved_ids = self.sumo_edge_id_map.get(edge_id, [edge_id])
                if not resolved_ids:
                    continue
                sumo_edge_id = resolved_ids[0]

                edge = None
                for e in root.findall("edge"):
                    if e.get("id") == sumo_edge_id and e.get("function") != "internal":
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
                            # Keep the macroscopic edge_id so the turning-rate
                            # aggregator can group detectors by macro link.
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
        """Build a unique detector ID from a detector entry dictionary.

        ID format:
        - backbone_segment: ``detector_backbonesegment_<edge_id>_<lane_index>_cell<cell_index>``
        - all other types:  ``detector_<type>_<edge_id>_<lane_index>``

        Args:
            det: Detector entry dictionary as stored in self.edge_detectors.

        Returns:
            Unique string identifier for the detector.
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

        Detector element mapping:
        - backbone_segment             → laneAreaDetector, no vTypes filter (counts all vehicles)
        - mainline_origin_interface    → inductionLoop,    no vTypes filter (counts all vehicles)
        - turning_rate                 → inductionLoop,    no vTypes filter (counts all vehicles)
        - inflow / outflow             → inductionLoop,    no vTypes filter (counts all vehicles)

        Returns:
            Absolute path to the written detector XML file.
        """
        output_file = f"{self.output_dir}/{self.detector_filename}"
        root = ET.Element("additional")

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

                # if det_type in URBAN_FILTERED_TYPES:
                #     # Only count vehicles explicitly typed "urban" in the route file.
                #     # mainline_origin_interface and turning_rate detectors intentionally
                #     # have NO filter here — they must observe all vehicle types.
                #     detector.set("vTypes", "urban")

        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        tree.write(output_file, encoding="utf-8", xml_declaration=True)
        return output_file

    def write_detector_spec_csv(self) -> str:
        """Write a CSV file mapping each detector to its edge, node, and cell metadata.

        Columns: detector_id, type, from, to, edge_id, backbone_node,
        diverge_node_id, position, cell_index, cell_key, detector_length.

        Returns:
            Absolute path to the written CSV file.
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
        """Run the full detector generation pipeline and write all output files.

        Calls find_interface_edges(), find_turning_rate_edges(),
        add_detectors_backbone_network(), write_detector_xml(), and
        write_detector_spec_csv() in sequence, then prints a placement summary.

        Returns:
            A tuple (detector_xml_path, detector_output_xml_path, detector_csv_path)
            with the paths to the written SUMO additional XML, the detector output XML
            reference path, and the detector specification CSV respectively.
        """
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
