import os
import subprocess
import xml.etree.ElementTree as ET


class SUMOSimulation:
    """A class for configuring and running SUMO traffic simulations.

    Attributes:
        name: Name identifier for the simulation.
        net_file: Path to the SUMO network file.
        detector_file: Path to the SUMO loop detectors file.
        rou_file: Path to the SUMO route file.
        output_dir: Directory where simulation outputs are stored.
        cfg_file: Path to the SUMO configuration file.
        stats_file: Path to the simulation statistics output file.
    """

    def __init__(self, name, net_file, rou_file, output_dir, detector_file=None):
        """Initialize the SUMO simulation.

        Args:
            name: Name identifier for the simulation.
            net_file: Path to the SUMO network file.
            rou_file: Path to the SUMO route file.
            output_dir: Directory where simulation outputs are stored.
        """
        self.name = name
        self.net_file = net_file
        self.detector_file = detector_file
        self.rou_file = rou_file
        self.output_dir = output_dir
        self.cfg_file = os.path.join(self.output_dir, f"{name}.sumocfg")
        self.stats_file = os.path.join(self.output_dir, f"{self.name}_stats.xml")

    def write_config(self):
        """Generate and write the SUMO configuration file.

        Creates a SUMO configuration XML file that specifies input network and route files,
        and configures output statistics reporting.
        """
        config_content = f"""
        <configuration>
            <input>
                <net-file value="{os.path.basename(self.net_file)}"/>
                {f'<additional-files value="{os.path.basename(self.detector_file)}"/>' if self.detector_file is not None else ''}
                <route-files value="{os.path.basename(self.rou_file)}"/>
            </input>
            <time>
                <end value="86400"/>
            </time>
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

    def run_simulation(self):
        """Run the SUMO simulation without GUI.

        Executes the SUMO simulation using the configuration file and prints
        a summary of results if statistics are available.
        """
        # TODO: remove "result ="
        result = subprocess.run(
            ["sumo", "-c", self.cfg_file, "--no-step-log=true"],
            capture_output=True,
            text=True,
        )
        # Diagnostic
        if result.returncode != 0:
            print(f"SUMO error:\n{result.stderr[:1000]}")
            return

        if os.path.exists(self.stats_file):
            self.print_summary()

    def print_summary(self):
        """Print a summary of simulation results.

        Parses the statistics XML file and prints key metrics including
        mean speed, total vehicles, and mean duration.

        Returns:
            A dictionary containing simulation results with keys 'mean_speed',
            'total_vehicles', and 'mean_duration', or None if statistics are unavailable.
        """
        tree = ET.parse(self.stats_file)
        root = tree.getroot()
        stats = root.find("vehicleTripStatistics")
        if stats is not None:
            results = {
                "mean_speed": float(stats.get("speed", 0)),
                "total_vehicles": int(stats.get("count", 0)),
                "mean_duration": float(stats.get("duration", 0)),
            }
            print(
                f"\n[RESULTS] {self.name}: {results['mean_speed']:.2f} m/s average speed over {results['total_vehicles']} vehicles.\n"
            )
            return results
        return None
