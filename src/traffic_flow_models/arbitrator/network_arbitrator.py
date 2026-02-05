import xml.etree.ElementTree as ET
import networkx as nx
import math
import logging
import sys
from traffic_flow_models.network.node import Node
from traffic_flow_models.network.motorway_link import MotorwayLink
from traffic_flow_models.network.cell import Cell
from traffic_flow_models.network.origin import Origin
from traffic_flow_models.network.destination import Destination
from traffic_flow_models.network.network import Network


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')


class NetworkArbitrator:
    
    ROAD_PARAMS = {
        "motorway": {
            "cap": 2000.0,    # Capacity per lane
            "jam": 150.0,     # Jam density
            "alpha": 1.868,   # Fundamental diagram exponent
            "speed": 120.0,   # Free-flow speed
            "tau": 18/3600,   # Relaxation time (18s converted to hours)
            "eta": 60.0,      # Anticipation factor (km²)
            "kappa": 40.0     # Lane-changing sensitivity
        },
        "trunk": {
            "cap": 1800.0, "jam": 160.0, "alpha": 1.6, "speed": 100.0,
            "tau": 20/3600, "eta": 50.0, "kappa": 35.0
        },
        "primary": {
            "cap": 1600.0, "jam": 180.0, "alpha": 1.4, "speed": 80.0,
            "tau": 22/3600, "eta": 40.0, "kappa": 30.0
        },
        "secondary": {
            "cap": 1200.0, "jam": 200.0, "alpha": 1.2, "speed": 50.0,
            "tau": 25/3600, "eta": 30.0, "kappa": 25.0
        },
        "tertiary": {
            "cap": 1000.0, "jam": 210.0, "alpha": 1.1, "speed": 30.0,
            "tau": 30/3600, "eta": 20.0, "kappa": 20.0
        },
        "default": {
            "cap": 1500.0, "jam": 160.0, "alpha": 1.5, "speed": 60.0,
            "tau": 20/3600, "eta": 40.0, "kappa": 30.0
        }
    }
    
    def __init__(self, net_xml_path, hwy_filter=None):
       
        self.path = net_xml_path
        self.target_cell_length = 0.5  #500m
        self.G = nx.MultiDiGraph()
        self.roundabouts = []
        self.found_types = set()
        self.node_coordinates = {}
        self.link_metanet_params = {}  
        self.selected_types = []

        if hwy_filter is not None:
            self.hwy_filter = hwy_filter
        else:
            self.hwy_filter = [["motorway", "motorway_link"], ["trunk", "trunk_link"], ["primary", "primary_link"], ["secondary", "secondary_link"], ["tertiary", "tertiary_link"] ]        
        
        logging.info(f"Initialized NetworkArbitrator for {net_xml_path}")
        logging.info(f"Target cell length: {self.target_cell_length} km")

    def run(self):
        """Execute the complete transformation pipeline."""
        logging.info("Starting SUMO to METANET transformation...")
        
        self.parse_sumo_xml()
        
        if self.G.number_of_edges() == 0:
            logging.error("No edges found in network after parsing. Check highway filter.")
            sys.exit(1)
        
        logging.info(f"Parsed network: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")
        
        self.eliminate_roundabouts()
        self.filter()
        self.merge_serial_edges()
        
        logging.info(f"After processing: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")
        
        metanet_network = self.instantiate_network()
        
        self._log_network_statistics(metanet_network)
        
        return metanet_network

    def parse_sumo_xml(self):
        """Parse SUMO .net.xml file and extract network topology."""
        tree = ET.parse(self.path)
        root = tree.getroot()

        available_types = set()
        for edge in root.findall('edge'):
            if edge.get('function') == 'internal':
                continue
            edge_type = edge.get('type', '')
            if edge_type:
                available_types.add(edge_type)
        
        #print(f"Available road types in network: {sorted(available_types)}")
    
        # Step 2: Select highest priority level available
        self.selected_types = []
        for priority_level in self.hwy_filter:
            # Check if ANY type from this priority exists
            matching = [t for t in priority_level 
                    if any(t in avail for avail in available_types)]
            if matching:
                self.selected_types = priority_level
                print(f"Selected road types: {self.selected_types}")
                break
        
        if not self.selected_types:
            print("ERROR: No matching road types found!")
            return

        # Extract junction coordinates
        raw_coordinates = {}
        for junction in root.findall('junction'):
            junction_id = junction.get('id')
            if junction.get('x') and junction.get('y'):
                raw_coordinates[junction_id] = (
                    float(junction.get('x')),
                    float(junction.get('y'))
                )

        # Normalize coordinates (shift to origin)
        if raw_coordinates:
            min_x = min(c[0] for c in raw_coordinates.values())
            min_y = min(c[1] for c in raw_coordinates.values())
            self.node_coordinates = {
                junction_id: (c[0] - min_x, c[1] - min_y)
                for junction_id, c in raw_coordinates.items()
            }

        # Parse roundabouts
        for roundabout in root.findall('roundabout'):
            self.roundabouts.append(roundabout.get('nodes', '').split())

        # Parse edges
        for edge in root.findall('edge'):
            if edge.get('function') == 'internal':
                continue
            
            edge_type = edge.get('type', '')
            self.found_types.add(edge_type)

            if not any(selected in edge_type for selected in self.selected_types):
                continue
                
            lanes = edge.findall('lane')
            if not lanes:
                continue

            # Convert SUMO units to METANET units
            length_km = float(lanes[0].get('length')) / 1000.0  # m -> km
            speed_kmh = float(lanes[0].get('speed')) * 3.6      # m/s -> km/h
        
            self.G.add_edge(
                edge.get('from'),
                edge.get('to'),
                id=edge.get('id'),
                length=length_km,
                speed=speed_kmh,
                lanes=len(lanes),
                type=edge_type
            )

    def eliminate_roundabouts(self):
        """
        Collapse roundabouts into single nodes.
        """
        for nodes in self.roundabouts:
            valid_nodes = [n for n in nodes if self.G.has_node(n)]
            if len(valid_nodes) <= 1:
                continue

            pivot = valid_nodes[0]

            # Calculate internal roundabout length
            internal_length = 0
            for node in valid_nodes:
                for _, v, data in self.G.out_edges(node, data=True):
                    if v in valid_nodes:
                        internal_length += data.get('length', 0)

            # Calculate centroid for merged node position
            coordinates_to_merge = [
                self.node_coordinates.get(n, (0, 0))
                for n in valid_nodes
            ]
            centroid_x = sum(c[0] for c in coordinates_to_merge) / len(coordinates_to_merge)
            centroid_y = sum(c[1] for c in coordinates_to_merge) / len(coordinates_to_merge)
            self.node_coordinates[pivot] = (centroid_x, centroid_y)

            # Distribute internal length to incident edges
            extra_length = (internal_length / max(1, len(valid_nodes))) / 2.0

            # Contract nodes into pivot
            for other in valid_nodes[1:]:
                if self.G.has_node(other):
                    self.G = nx.contracted_nodes(self.G, pivot, other, self_loops=False)

            # Add extra length to incident edges
            for u, v, key, data in self.G.edges(keys=True, data=True):
                if u == pivot or v == pivot:
                    data['length'] += extra_length

        logging.info(f"Eliminated {len(self.roundabouts)} roundabouts")

    def filter(self):
        """Remove isolated nodes and keep only the largest connected component."""
        self.G.remove_nodes_from(list(nx.isolates(self.G)))

        if self.G.number_of_nodes() > 0 and not nx.is_weakly_connected(self.G):
            largest = max(nx.weakly_connected_components(self.G), key=len)
            self.G = self.G.subgraph(largest).copy()
            logging.info("Filtered to largest connected component")

    def merge_serial_edges(self):
        """
        Merge serial edges with similar characteristics. 
        """
        merge_count = 0
        while True:
            candidates = [
                n for n in self.G.nodes()
                if self.G.in_degree(n) == 1 and self.G.out_degree(n) == 1
            ]
            merged = False

            for n in candidates:
                in_edges = list(self.G.in_edges(n, data=True))
                out_edges = list(self.G.out_edges(n, data=True))
                u, _, d_in = in_edges[0]
                _, v, d_out = out_edges[0]

                # Check if edges can be merged (similar characteristics)
                same_lanes = d_in['lanes'] == d_out['lanes']
                same_speed = abs(d_in['speed'] - d_out['speed']) < 5.0

                if same_lanes and same_speed:
                    new_attr = {
                        'id': f"merged_{d_in['id']}",
                        'length': d_in['length'] + d_out['length'],
                        'speed': min(d_in['speed'], d_out['speed']),
                        'lanes': d_in['lanes'],
                        'type': d_in.get('type', 'default')
                    }
                    self.G.add_edge(u, v, **new_attr)
                    self.G.remove_node(n)
                    
                    if n in self.node_coordinates:
                        del self.node_coordinates[n]
                    
                    merged = True
                    merge_count += 1
                    break
            
            if not merged:
                break
        
        logging.info(f"Merged {merge_count} serial edges")


    def instantiate_network(self):
        """
        Create METANET network objects with proper parameters. 
        """
        metanet_nodes = {}
        total_cells = 0

        # Create Node objects
        for nid in self.G.nodes():
            n_obj = Node(id=str(nid))
            n_obj.x, n_obj.y = self.node_coordinates.get(nid, (0, 0))
            metanet_nodes[nid] = n_obj

        # Create MotorwayLink objects with cells
        for u, v, data in self.G.edges(data=True):
            edge_type = data.get('type', 'default').lower()
            
            # Match road type parameters
            params = next(
                (val for key, val in self.ROAD_PARAMS.items() if key in edge_type),
                self.ROAD_PARAMS["default"]
            )

            link_id = str(data['id'])
            num_lanes = data['lanes']
            
            # Calculate critical density: rho_crit = C / (v_free * lambda)
            # where C is capacity per lane, v_free is free-flow speed, lambda is number of lanes
            critical_density = params['cap'] / params['speed']
            
            # Store all parameters for this link
            self.link_metanet_params[link_id] = {
                'alpha': params['alpha'],
                'tau': params['tau'],
                'eta': params['eta'],
                'kappa': params['kappa'],
                'critical_density': critical_density
            }

            # Create MotorwayLink with METANET parameters
            self.link_metanet_params[link_id] = {
                'alpha': params['alpha'],
                'tau': params['tau'],
                'eta': params['eta'],
                'kappa': params['kappa'],
                'critical_density': critical_density
            }

            link = MotorwayLink(
                id=link_id,
                length=data['length'],
                lanes=num_lanes,
                lane_capacity=params['cap'],
                free_flow_speed=params['speed'],
                jam_density=params['jam'],
                origin_node_id=str(u),
                destination_node_id=str(v)
            )
            
            # Discretize link into cells 
            num_cells = max(1, math.ceil(data['length'] / self.target_cell_length))
            cell_len = data['length'] / num_cells
            
            for _ in range(num_cells):
                link.add_cell(length=cell_len)
            
            total_cells += num_cells
            
            # Add to node connections
            metanet_nodes[u].outgoing.append(link)
            metanet_nodes[v].incoming.append(link)

        # Set up origins and destinations
        for nid, node_obj in metanet_nodes.items():
            if not node_obj.incoming:
                orig = Origin(id=f"Origin_{nid}", destination_node_id=str(nid))
                node_obj.set_incoming([orig])
            else:
                node_obj.set_incoming(list(node_obj.incoming))

            if not node_obj.outgoing:
                dest = Destination(id=f"Dest_{nid}", origin_node_id=str(nid))
                node_obj.set_outgoing([dest])
            else:
                node_obj.set_outgoing(list(node_obj.outgoing))

        logging.info(f"Created {total_cells} cells across {self.G.number_of_edges()} links")
        logging.info(f"METANET parameters stored for {len(self.link_metanet_params)} links")
        
        return Network(nodes=list(metanet_nodes.values()))

    def get_link_params(self, link_id):
        """
        Retrieve METANET parameters for a specific link.
        
        Returns:
            dict with keys: alpha, tau, eta, kappa, critical_density
        """
        return self.link_metanet_params.get(link_id, {})

    def _log_network_statistics(self, network):
        """Log comprehensive network statistics."""
        num_nodes = len(network._nodes)
        num_links = sum(len(node.outgoing) for node in network._nodes 
                       if not isinstance(node.outgoing[0], Destination))
        num_origins = sum(1 for node in network._nodes 
                         if node.incoming and isinstance(node.incoming[0], Origin))
        num_destinations = sum(1 for node in network._nodes 
                              if node.outgoing and isinstance(node.outgoing[0], Destination))
        
        total_length = sum(
            link.length for node in network._nodes
            for link in node.outgoing
            if isinstance(link, MotorwayLink)
        )
        
        logging.info("=" * 60)
        logging.info("METANET Network Statistics:")
        logging.info(f"  Nodes: {num_nodes}")
        logging.info(f"  Links: {num_links}")
        logging.info(f"  Origins: {num_origins}")
        logging.info(f"  Destinations: {num_destinations}")
        logging.info(f"  Total network length: {total_length:.2f} km")
        logging.info("=" * 60)


"""
if __name__ == "__main__":
    # Example usage with enhanced configuration
    arbitrator = NetworkArbitrator(
        "Zurich.net.xml",
        simulation_timestep=1  # timestep
    )
    metanet_net = arbitrator.run()
    
    logging.info("METANET network transformation completed successfully!")
"""