import xml.etree.ElementTree as ET
from typing import Set, Dict, List
import logging
import os
from collections import defaultdict
from traffic_flow_models.arbitrator.network_arbitrator import NetworkArbitrator


class LoopDetectorGenerator:
    def __init__(self, consolidated_network, urban_network_path :str):
        self.consolidated_network = consolidated_network
        self.urban_network_path = urban_network_path

        self.consolidated_nodes: Set[str] = set()

        self.urban_tree = None
        self.urban_root = None
        self.edges = {}

        self.onramp_edges: Dict[str, Dict] = {}
        self.offramp_edges: Dict[str, Dict] = {}


    def extract_consolidated_nodes(self):
        
        if hasattr(self.consolidated_network, "list_nodes"):
            nodes_iter = self.consolidated_network.list_nodes()
        
        elif hasattr(self.consolidated_network, "_nodes"):
            nodes_iter = self.consolidated_network._nodes
        
        elif hasattr(self.consolidated_network, "nodes"):
            nodes_iter = self.consolidated_network.nodes
        
        else:
            raise AttributeError("The consollidated_network has no accessible nodes.")


        for node in nodes_iter:
            self.consolidated_nodes.add(str(node.id))

    
    def load_urban_network(self):

        self.urban_tree = ET.parse(self.urban_network_path)
        self.urban_root = self.urban_tree.getroot()

# Identifies the locations to place loop detectors on the urban network
    def identify_interface_edges(self):
        
        for edge in self.urban_root.findall('edge'):
            if edge.get('function') == 'internal':
                continue

            edge_id = edge.get('id')
            from_j = edge.get('from')
            to_j = edge.get('to')

            from_in = from_j in self.consolidated_nodes
            to_in = to_j in self.consolidated_nodes

            #onramp
            if not from_in and to_in:
                self.onramp_edges[edge_id] = {'from': from_j, 'to': to_j}

            #offramp
            elif from_in and not to_in:
                self.offramp_edges[edge_id] = {'from': from_j, 'to': to_j}


    def create_detector(self, edge_id, lane_index):
        
        detector_id = f"detector_{edge_id}_{lane_index}"
        lane_id = f"{edge_id}_{lane_index}" 

        detector = ET.Element('laneAreaDetector')
        detector.set('id', detector_id)
        detector.set('lane', lane_id)
        detector.set('pos', '0.1')    
        detector.set('length', '1.0')
        detector.set('freq', '60')
        detector.set('file', 'detectors_output.xml')
        
        return detector
    

    #creates an addtional file with all the detector positions
    def create_additional_file(self, output_path):
        additional_root = ET.Element('additional')
        total_detector_count = 0

        interface_edges = list(self.onramp_edges.keys()) + list(self.offramp_edges.keys())

        for edge_id in interface_edges:
            edge_element = self.urban_root.find(f".//edge[@id='{edge_id}']")
            
            if edge_element is not None:
                lanes = edge_element.findall('lane')
    
                for i in range(len(lanes)):
                    detector = self.create_detector(edge_id, i)
                    additional_root.append(detector)
                    total_detector_count += 1

        tree = ET.ElementTree(additional_root)
        tree.write(output_path, encoding='utf-8', xml_declaration=True)

        return output_path
    

    #creates a csv file with all the detector positions
    def write_detector_spec(self):
        
        base = self.urban_network_path.replace('.net.xml', '')
        spec_path = f"{base}_detectors_spec.csv"
        
        with open(spec_path, 'w') as f:
            f.write("detector_id,edge_id,type,from,to,measurement\n")
            
            for edge_id, info in self.onramp_edges.items():
                det_id = f"detector_{edge_id}_0"
                f.write(f'"{det_id}","{edge_id}","onramp","{info["from"]}","{info["to"]}","origin demand"\n')
            
            for edge_id, info in self.offramp_edges.items():
                det_id = f"detector_{edge_id}_0"
                f.write(f'"{det_id}","{edge_id}","offramp","{info["from"]}","{info["to"]}","destination demand"\n')
        
        return spec_path
    

    def generate(self):
       
        self.extract_consolidated_nodes()
        self.load_urban_network()
        self.identify_interface_edges()
        
        base = self.urban_network_path.replace('.net.xml', '')
        additional_path = f"{base}_detectors.xml"
        additional_path = self.create_additional_file(additional_path)
        
        spec_path = self.write_detector_spec()
        
        return additional_path, spec_path
    

"""
if __name__ == "__main__":

    arbitrator = NetworkArbitrator("Zurich.net.xml")
    consolidated_network = arbitrator.run()
    
    gen = LoopDetectorGenerator(consolidated_network, "Zurich.net.xml")
    additional_path, spec_path = gen.generate()
"""    

