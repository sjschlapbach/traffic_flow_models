import osmnx as ox
import matplotlib.pyplot as plt
import os
import subprocess
import sys
from functools import wraps
import xml.etree.ElementTree as ET
import argparse


# Use this command to run the code
# pipeline.py --name run_name --loc location --veh number_of_vehicles(demand)
# Ex: pipeline.py --name run1 --loc Blacksburg,Virgin,USA --veh 3000

def skip_if_exists(attr_name):  # Decorator to check if a file exists
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            file_path = getattr(self, attr_name)
            if os.path.exists(file_path):
                print(f"[SKIP] {file_path} already exists.")
                return
            return func(self, *args, **kwargs)
        return wrapper
    return decorator



class SUMOpipeline: 
    def __init__(self, name, location):
        self.name = name
        self.location = location
        
        self.output_dir = name
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.osm_file = os.path.join(self.output_dir, f"{name}.osm")
        self.net_file = os.path.join(self.output_dir, f"{name}.net.xml")
        self.rou_file = os.path.join(self.output_dir, f"{name}.rou.xml")

    
    @skip_if_exists('osm_file')
    def fetch_OSM(self):    # Function to download osm file for the specified location
        ox.settings.all_oneway = True
        graph = ox.graph_from_place(self.location, network_type='drive',simplify=False)
        fig, ax = ox.plot_graph(graph, bgcolor ='white', node_color='blue', node_size=2, edge_color='gray', edge_linewidth=0.3,show=False,close=False)
        ax.set_title(f"{self.location} Road Network", fontsize=16)
        fig.savefig(os.path.join(self.output_dir,"network_plot.png"),dpi=500)
        plt.close(fig)
        ox.save_graph_xml(graph, filepath=self.osm_file)
        print(f"OSM data downloaded for {self.location}")


    @skip_if_exists('net_file')
    def covert_to_sumo(self):   # Function to convert osm file to .net.xml (sumo network file)
        cmd = [
        "netconvert",
        "--osm-files", self.osm_file,
        "--output-file", self.net_file,
        "--geometry.remove", "true",        
        "--junctions.join", "true",       
        "--roundabouts.guess", "true",
        "--tls.discard-simple", "true",
        "--verbose", "true",
        "--ramps.guess", "true",
        "--remove-edges.isolated", "true",           
        "--tls.guess-signals", "true",
        "--tls.join", "true",
        "--tls.ignore-internal-junction-jam", "true"]

        subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"{self.net_file} file generated.")


    #@skip_if_exists('rou_file')
    def generate_demand(self, vehicle_count):   # Function to generate demand and write it to the .rou.xml file
        if 'SUMO_HOME' not in os.environ:
            print("Error: Please set the 'SUMO_HOME' environment variable.")
            return

        random_trips = os.path.join(os.environ['SUMO_HOME'], 'tools', 'randomTrips.py')
    
        cmd = [
            sys.executable, random_trips,
            "-n", self.net_file,
            "-o", "temp_trips.xml",
            "--route-file", self.rou_file,
            "--period", str(3600 / vehicle_count), 
            "--fringe-factor", "10",              
            "--validate",
            "--remove-loops"
            ]

        try:
            subprocess.run(cmd, check=True)
            print(f"{self.rou_file} file generated.")
            
            if os.path.exists("temp_trips.xml"):
                os.remove("temp_trips.xml")
                
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while generating demand: {e}")



class SUMOsimulation:
    def __init__(self, name, net_file, rou_file, output_dir):
        self.name = name
        self.net_file = net_file
        self.rou_file = rou_file
        self.output_dir = output_dir
        self.cfg_file = os.path.join(self.output_dir, f"{name}.sumocfg")
        self.stats_file = os.path.join(self.output_dir, f"{self.name}_stats.xml")
   
    

    def write_config(self): # Function to generate the sumo config file
        config_content = f"""
        <configuration>
            <input>
                <net-file value="{os.path.basename(self.net_file)}"/>
                <route-files value="{os.path.basename(self.rou_file)}"/>
            </input>
            <output>
            <statistic-output value="{os.path.basename(self.stats_file)}"/>
            </output>
             <report>
                <duration-log.statistics value="true"/>
            </report>
        </configuration>"""
        
        with open(self.cfg_file, "w") as f:
            f.write(config_content.strip())
        print(f"{self.cfg_file} file generated.")

    
    def run_simulation(self):   # Function to run the sumo simulation without gui
        subprocess.run(["sumo", "-c", self.cfg_file, "--no-step-log", "true"], check=True)
        
        if os.path.exists(self.stats_file):
            self.print_summary()


    def print_summary(self):    # Function to print summary of results from the summary file
        tree = ET.parse(self.stats_file)
        root = tree.getroot()
        stats = root.find('vehicleTripStatistics')
        if stats is not None:
            results = {
                "mean_speed": float(stats.get('speed')),  # Average speed of all trips
                "total_vehicles": int(stats.get('count')),
                "mean_duration": float(stats.get('duration'))
            }
            print(f"\n[RESULTS] {self.name}: {results['mean_speed']:.2f} m/s average speed over {results['total_vehicles']} vehicles.")
            return results
        return None
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--loc", required=True)
    parser.add_argument("--veh", type=int, default=2000)
    args = parser.parse_args()

    network = SUMOpipeline(args.name, args.loc)
    network.fetch_OSM()
    network.covert_to_sumo()
    network.generate_demand(args.veh)

    sim = SUMOsimulation(args.name, network.net_file, network.rou_file, network.output_dir)
    sim.write_config()
    sim.run_simulation()


if __name__=="__main__":
    main()