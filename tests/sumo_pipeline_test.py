import os
import json
import pytest

from traffic_flow_models import SUMOPipeline


@pytest.fixture
def road_params_config(tmp_path):
    """Create a temporary road parameters config file for testing."""
    config = {
        "motorway": {
            "lane_capacity": 2000.0,
            "jam_density": 180.0,
            "free_flow_speed": 120.0,
        },
        "trunk": {
            "lane_capacity": 1800.0,
            "jam_density": 180.0,
            "free_flow_speed": 100.0,
        },
        "primary": {
            "lane_capacity": 1600.0,
            "jam_density": 180.0,
            "free_flow_speed": 80.0,
        },
        "secondary": {
            "lane_capacity": 1200.0,
            "jam_density": 200.0,
            "free_flow_speed": 50.0,
        },
        "tertiary": {
            "lane_capacity": 1000.0,
            "jam_density": 210.0,
            "free_flow_speed": 30.0,
        },
        "default": {
            "lane_capacity": 2000.0,
            "jam_density": 180.0,
            "free_flow_speed": 100.0,
        },
    }
    config_path = tmp_path / "test_road_params.json"
    with open(config_path, "w") as f:
        json.dump(config, f)
    return str(config_path)


class TestSUMOPipeline:
    def test_init_creates_output_dir(self, tmp_path, monkeypatch, road_params_config):
        # run inside temporary working dir so results/ is created there
        monkeypatch.chdir(tmp_path)
        p = SUMOPipeline(
            "myloc",
            "Somewhere",
            road_params_config,
            os.path.join("results", "myloc"),
        )
        assert os.path.isdir(os.path.join("results", "myloc"))

        # verify that class parameters have been set correctly
        assert p.name == "myloc"
        assert p.location == "Somewhere"
        assert p.output_dir == os.path.join("results", "myloc")

        # paths should point inside the results folder and be stored correctly in the class
        assert p.osm_file.endswith(os.path.join("results", "myloc", "myloc.osm"))
        assert p.net_file.endswith(os.path.join("results", "myloc", "myloc.net.xml"))
        assert p.rou_file.endswith(os.path.join("results", "myloc", "myloc.rou.xml"))

    def test_skip_if_exists_decorator_skips(
        self, monkeypatch, tmp_path, capsys, road_params_config
    ):
        monkeypatch.chdir(tmp_path)
        p = SUMOPipeline(
            "skiptest",
            "Nowhere",
            road_params_config,
            os.path.join("results", "skiptest"),
        )

        # create the osm file so fetch_OSM should be skipped
        os.makedirs(os.path.dirname(p.osm_file), exist_ok=True)
        open(p.osm_file, "w").write("exists")

        # call should be skipped and not raise
        p.fetch_OSM()
        captured = capsys.readouterr()
        assert ".osm already exists" in captured.out

    def test_convert_to_sumo_runs_netconvert(
        self, monkeypatch, tmp_path, road_params_config
    ):
        monkeypatch.chdir(tmp_path)
        p = SUMOPipeline(
            "convtest",
            "Loc",
            road_params_config,
            os.path.join("results", "convtest"),
        )

        # create a dummy osm file path
        os.makedirs(os.path.dirname(p.osm_file), exist_ok=True)
        open(p.osm_file, "w").write("dummy")

        called = {}
        monkeypatch.setattr(
            "subprocess.run",
            lambda *a, **k: called.__setitem__("cmd", a[0] if a else k.get("cmd")),
        )

        p.convert_to_sumo()
        assert "cmd" in called
        assert called["cmd"][0] == "netconvert"

        # ensure osm input and output-file flags present with the correct files
        assert "--osm-files" in called["cmd"]
        assert "--output-file" in called["cmd"]
        osm_index = called["cmd"].index("--osm-files") + 1
        out_index = called["cmd"].index("--output-file") + 1
        assert called["cmd"][osm_index] == p.osm_file
        assert called["cmd"][out_index] == p.net_file

    def test_generate_demand_handles_missing_sumo_home(
        self, monkeypatch, tmp_path, capsys, road_params_config
    ):
        monkeypatch.chdir(tmp_path)
        p = SUMOPipeline(
            "dtest",
            "Loc",
            road_params_config,
            os.path.join("results", "dtest"),
        )

        # ensure SUMO_HOME not set
        monkeypatch.delenv("SUMO_HOME", raising=False)

        # The pipeline raises an EnvironmentError when SUMO_HOME is missing,
        # and the message should be explicit and stable for callers.
        with pytest.raises(EnvironmentError) as excinfo:
            p.generate_demand(10, 3600.0)
        assert str(excinfo.value) == "Please set the 'SUMO_HOME' environment variable."

    def test_generate_demand_invokes_randomTrips(
        self, monkeypatch, tmp_path, road_params_config
    ):
        monkeypatch.chdir(tmp_path)
        p = SUMOPipeline(
            "dtest2",
            "Loc",
            road_params_config,
            os.path.join("results", "dtest2"),
        )

        # 1. Setup fake SUMO environment
        fake_sumo = tmp_path / "sumo"
        tools_dir = fake_sumo / "tools"
        tools_dir.mkdir(parents=True)
        random_trips = tools_dir / "randomTrips.py"
        random_trips.write_text("#dummy")
        monkeypatch.setenv("SUMO_HOME", str(fake_sumo))

        called = {}
        # Accept arbitrary positional and keyword args so tests are robust
        # to subprocess.run being called with kwargs like capture_output/text.
        monkeypatch.setattr(
            "subprocess.run",
            lambda *a, **k: called.__setitem__("cmd", a[0] if a else k.get("cmd")),
        )

        # 2. Create a dummy VALID XML net file (ET.parse will fail on plain text "net")
        os.makedirs(os.path.dirname(p.net_file), exist_ok=True)
        with open(p.net_file, "w") as f:
            f.write('<net><edge id="e1" from="n1" to="n2"/></net>')

        # 3. Create a dummy VALID XML route file
        # This is what was missing! ET.parse needs this file to exist.
        os.makedirs(os.path.dirname(p.rou_file), exist_ok=True)
        with open(p.rou_file, "w") as f:
            # Create enough trip elements to satisfy your vehicle_count (20)
            trips = "".join([f'<trip id="{i}" depart="0"/>' for i in range(20)])
            f.write(f"<routes>{trips}</routes>")

        # 4. Now run the demand generation
        # subprocess.run is monkeypatched and won't create the temporary
        # urban route file. Create it from the final route file so ET.parse
        # inside generate_demand can proceed.
        temp_urban_rou = os.path.join(p.output_dir, "_temp_urban.rou.xml")
        with open(p.rou_file, "r") as src, open(temp_urban_rou, "w") as dst:
            dst.write(src.read())

        p.generate_demand(20, 3600.0)

        # Assertions to ensure subprocess was called
        assert "cmd" in called

    def test_convert_to_sumo_skips_when_net_exists(
        self, monkeypatch, tmp_path, capsys, road_params_config
    ):
        monkeypatch.chdir(tmp_path)
        p = SUMOPipeline(
            "convskip",
            "Place",
            road_params_config,
            os.path.join("results", "convskip"),
        )

        # create the net file so convert_to_sumo should be skipped
        os.makedirs(os.path.dirname(p.net_file), exist_ok=True)
        open(p.net_file, "w").write("exists")

        # monkeypatch subprocess.run to raise if called
        monkeypatch.setattr(
            "subprocess.run",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("should_not_be_called")),
        )

        p.convert_to_sumo()
        captured = capsys.readouterr()
        assert ".net.xml already exists." in captured.out
