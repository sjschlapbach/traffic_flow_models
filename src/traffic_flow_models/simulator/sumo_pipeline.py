import os
import subprocess
import sys
import shutil
import osmnx as ox
from functools import wraps
import matplotlib.pyplot as plt
from traffic_flow_models.arbitrator.loop_detector_generator import LoopDetectorGenerator
from traffic_flow_models.arbitrator.network_arbitrator import NetworkArbitrator


def skip_if_exists(attr_name):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if os.path.exists(getattr(self, attr_name)):
                print(f"[SKIP] {getattr(self, attr_name)} already exists.")
                return
            return func(self, *args, **kwargs)

        return wrapper

    return decorator


class SUMOPipeline:
    """A pipeline for fetching OSM data and preparing SUMO simulation files.

    Attributes:
        name: Name identifier for the simulation.
        location: Geographic location to fetch OSM data from.
        output_dir: Directory where output files are stored.
        osm_file: Path to the downloaded OSM file.
        net_file: Path to the SUMO network file.
        detector_file: Path to the SUMO loop detectors file.
        rou_file: Path to the SUMO route file.
    """

    def __init__(self, name, location):
        """Initialize the SUMO pipeline.

        Args:
            name: Name identifier for the simulation.
            location: Geographic location to fetch OSM data from.
        """
        self.name = name
        self.location = location

        # set up output directory
        self.output_dir = os.path.join("results", name)
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        self.osm_file = os.path.join(self.output_dir, f"{name}.osm")
        self.net_file = os.path.join(self.output_dir, f"{name}.net.xml")
        self.detector_file = os.path.join(self.output_dir, f"{name}_detectors.xml")
        self.rou_file = os.path.join(self.output_dir, f"{name}.rou.xml")

        self.detector_spec_path = os.path.join(self.output_dir, f"{name}_detectors_spec.csv")
        self.consolidated_network = None
        self.arbitrator = None

    @skip_if_exists("osm_file")
    def fetch_OSM(self):
        """Download OSM data for the specified location.

        Fetches road network data from OpenStreetMap, plots the network,
        and saves it as an OSM XML file.
        """
        ox.settings.all_oneway = True
        graph = ox.graph_from_place(self.location, network_type="drive", simplify=False)
        fig, ax = ox.plot_graph(
            graph,
            bgcolor="white",
            node_color="blue",
            node_size=2,
            edge_color="gray",
            edge_linewidth=0.3,
            show=False,
            close=False,
        )
        ax.set_title(f"{self.location} Road Network", fontsize=16)
        fig.savefig(os.path.join(self.output_dir, "network_plot.png"), dpi=500)
        plt.close(fig)
        ox.save_graph_xml(graph, filepath=self.osm_file)
        print(f"OSM data downloaded for {self.location}")

    @skip_if_exists("net_file")
    def covert_to_sumo(self):
        """Convert OSM file to SUMO network format.

        Converts the downloaded OSM data to a SUMO .net.xml file using netconvert
        with various processing options for geometry simplification, junction handling,
        and traffic signal detection.
        """
        cmd = [
            "netconvert",
            "--osm-files",
            self.osm_file,
            "--output-file",
            self.net_file,
            "--geometry.remove",
            "true",
            "--junctions.join",
            "true",
            "--roundabouts.guess",
            "true",
            "--tls.discard-simple",
            "true",
            "--verbose",
            "true",
            "--ramps.guess",
            "true",
            "--remove-edges.isolated",
            "true",
            "--tls.guess-signals",
            "true",
            "--tls.join",
            "true",
            "--tls.ignore-internal-junction-jam",
            "true",
        ]

        subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"{self.net_file} file generated.")

    # @skip_if_exists('rou_file')
    def generate_demand(self, vehicle_count):
        """Generate traffic demand and create route file.

        Creates random trips using SUMO's randomTrips.py tool and generates
        a route file with the specified vehicle count.

        Args:
            vehicle_count: Number of vehicles to generate in the simulation.
        """
        if "SUMO_HOME" not in os.environ:
            print("Error: Please set the 'SUMO_HOME' environment variable.")
            return

        random_trips = os.path.join(os.environ["SUMO_HOME"], "tools", "randomTrips.py")

        cmd = [
            sys.executable,
            random_trips,
            "-n",
            self.net_file,
            "-o",
            "temp_trips.xml",
            "--route-file",
            self.rou_file,
            "--period",
            str(3600 / vehicle_count),
            "--fringe-factor",
            "10",
            "--validate",
            "--remove-loops",
        ]

        try:
            subprocess.run(cmd, check=True)
            print(f"{self.rou_file} file generated.")

            if os.path.exists("temp_trips.xml"):
                os.remove("temp_trips.xml")
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while generating demand: {e}")


    def generate_detectors(self):
        self.arbitrator = NetworkArbitrator(os.path.normpath(self.net_file))
        self.consolidated_network = self.arbitrator.run()

        generator = LoopDetectorGenerator(self.consolidated_network, self.net_file)
        self.detector_file, self.detector_spec_path = generator.generate()

        return self.detector_file
    

    def get_consolidated_network(self):
        
        if self.consolidated_network is None:
            raise ValueError("Must call generate_detectors() first to create consolidated network")
        
        return self.consolidated_network