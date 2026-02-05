import xml.etree.ElementTree as ET
import csv
import networkx as nx
from typing import Dict, Callable, Tuple, Set
from collections import defaultdict

class DemandAggregator:
    
    def __init__(
        self, 
        detector_output_path: str, 
        detector_spec_path: str, 
        time_period_minutes: float = 15):
        self.detector_output_path = detector_output_path
        self.detector_spec_path = detector_spec_path
        self.time_period_sec = time_period_minutes * 60
        
        self.detector_intervals = defaultdict(list)  
        self.detector_mapping = {}                  
        self.node_counts = defaultdict(lambda: defaultdict(int))  
        self.max_time = 0.0
        


    def parse_detector_output(self):
        
        tree = ET.parse(self.detector_output_path)
        root = tree.getroot()

        for interval in root.findall('interval'):
            det_id = interval.get('id')
            begin = float(interval.get('begin'))
            
            count = int(interval.get('nVehEntered', interval.get('nVehContrib', 0)))
            
            self.detector_intervals[det_id].append((begin, count))
            self.max_time = max(self.max_time, begin)
        

    #Map detector IDs to node IDs from CSV specification.
    def classify_and_map(self):
 
        with open(self.detector_spec_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                det_id = row['detector_id'].strip().strip('"').strip("'")
                
                det_id_variants = [
                    det_id,
                    det_id.replace("detector_", ""),  
                    f"detector_{det_id}",              
                ]
                
                det_type = row['type'].strip().lower()
                from_node = row['from'].strip().strip('"').strip("'")
                to_node = row['to'].strip().strip('"').strip("'")
                
                if 'onramp' in det_type or 'origin' in det_type:
                    node_id = to_node
                elif 'offramp' in det_type or 'destination' in det_type:
                    node_id = from_node
                else:
                    node_id = to_node
                
                if node_id:
                    # Store all variants
                    for variant in det_id_variants:
                        self.detector_mapping[variant] = {
                            'node_id': node_id, 
                            'type': det_type
                        }


    #Aggregate lane-level detector counts into node-level counts.
    def aggregate_spatially(self):
        
        # Aggregate: sum all lane detectors at same node
        for det_id, intervals in self.detector_intervals.items():
            if det_id not in self.detector_mapping:
                continue
            
            node_id = self.detector_mapping[det_id]['node_id']
            
            for begin, count in intervals:
                time_bin = int(begin / self.time_period_sec)
                self.node_counts[node_id][time_bin] += count


    #Aggregate ALL detector data from roads feeding into METANET entry points.
    def aggregate_upstream_to_metanet(self, metadata, sumo_network_path):
        
        if not metadata:
            raise ValueError("metadata parameter is required")
        
        G = self._build_network_graph(sumo_network_path)
        
        metanet_origins = metadata.get("origin_ids", [])
        metanet_onramps = metadata.get("onramp_ids", [])
        
        origin_node_ids = [oid.replace("Origin_", "") for oid in metanet_origins]
        onramp_node_ids = [oid.replace("onramp_", "") for oid in metanet_onramps]
        
        origin_demands = {}
        
        for origin_node in origin_node_ids:
            upstream_nodes = self._find_upstream_nodes(G, origin_node)
            aggregated_bins = self._aggregate_demand(upstream_nodes)
            
            metanet_id = f"Origin_{origin_node}"
            origin_demands[metanet_id] = self._make_demand_function(aggregated_bins)
            
            total_vehicles = sum(aggregated_bins.values())
        

        onramp_demands = {}
        
        for onramp_node in onramp_node_ids:
            upstream_nodes = self._find_upstream_nodes(G, onramp_node)
            aggregated_bins = self._aggregate_demand(upstream_nodes)
            
            metanet_id = f"onramp_{onramp_node}"
            onramp_demands[metanet_id] = self._make_demand_function(aggregated_bins)
            
            total_vehicles = sum(aggregated_bins.values())
        
        all_detector_vehicles = sum(sum(bins.values()) for bins in self.node_counts.values())
        all_metanet_vehicles = sum(
            sum(aggregated_bins.values()) 
            for demands in [origin_demands, onramp_demands]
            for func in demands.values())
        
    
        print(f"AGGREGATION SUMMARY:")
        print(f"  Total detector nodes: {len(self.node_counts)}")
        print(f"  Total detector vehicles: {all_detector_vehicles}")
        print(f"  METANET entry points: {len(origin_demands) + len(onramp_demands)}")
        print(f"  ✅ ALL DETECTOR DATA AGGREGATED INTO METANET DEMANDS")
     
        return origin_demands, onramp_demands


    #Build directed graph from SUMO network XML.
    def _build_network_graph(self, sumo_network_path):
        G = nx.DiGraph()
        tree = ET.parse(sumo_network_path)
        root = tree.getroot()
        
        for edge in root.findall('edge'):
            if edge.get('function') != 'internal':
                from_node = edge.get('from')
                to_node = edge.get('to')
                if from_node and to_node:
                    G.add_edge(from_node, to_node)
        
        return G


    #Find all nodes that have a path leading to the target node.
    def _find_upstream_nodes(self, G, target_node):
        upstream_nodes = {target_node} 
        
        for node in self.node_counts.keys():
            if node == target_node:
                continue
            
            try:
                if G.has_node(node) and G.has_node(target_node):
                    if nx.has_path(G, node, target_node):
                        upstream_nodes.add(node)
            except nx.NetworkXError:
                continue
        
        return upstream_nodes

    #Aggregate vehicle counts from multiple nodes.
    def _aggregate_demand(self, node_set):
        aggregated_bins = defaultdict(int)
        
        for node in node_set:
            if node in self.node_counts:
                for time_bin, count in self.node_counts[node].items():
                    aggregated_bins[time_bin] += count
        
        return dict(aggregated_bins)

    #demand function that returns veh/h for given time in hours.
    def _make_demand_function(self, aggregated_bins):
        time_period_sec = self.time_period_sec
        
        def demand_at_time(time_hours):
            t_sec = time_hours * 3600
            time_bin = int(t_sec / time_period_sec)
            count = aggregated_bins.get(time_bin, 0)

            return count * (3600.0 / time_period_sec)
        
        return demand_at_time


    def run(self, metadata, sumo_network_path):
        if not metadata:
            raise ValueError("metadata is required")

        if not sumo_network_path:
            raise ValueError("sumo_network_path is required when aggregate_upstream=True")

        self.parse_detector_output()
        self.classify_and_map()
        self.aggregate_spatially()
        
        return self.aggregate_upstream_to_metanet(metadata, sumo_network_path)

