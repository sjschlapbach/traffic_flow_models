import os

from traffic_flow_models import SUMOPipeline


class TestSUMOPipeline:
    def test_init_creates_output_dir(self, tmp_path, monkeypatch):
        # run inside temporary working dir so results/ is created there
        monkeypatch.chdir(tmp_path)
        p = SUMOPipeline("myloc", "Somewhere")
        assert os.path.isdir(os.path.join("results", "myloc"))

        # verify that class parameters have been set correctly
        assert p.name == "myloc"
        assert p.location == "Somewhere"
        assert p.output_dir == os.path.join("results", "myloc")

        # paths should point inside the results folder and be stored correctly in the class
        assert p.osm_file.endswith(os.path.join("results", "myloc", "myloc.osm"))
        assert p.net_file.endswith(os.path.join("results", "myloc", "myloc.net.xml"))
        assert p.rou_file.endswith(os.path.join("results", "myloc", "myloc.rou.xml"))

    def test_skip_if_exists_decorator_skips(self, monkeypatch, tmp_path, capsys):
        monkeypatch.chdir(tmp_path)
        p = SUMOPipeline("skiptest", "Nowhere")

        # create the osm file so fetch_OSM should be skipped
        os.makedirs(os.path.dirname(p.osm_file), exist_ok=True)
        open(p.osm_file, "w").write("exists")

        # call should be skipped and not raise
        p.fetch_OSM()
        captured = capsys.readouterr()
        assert ".osm already exists" in captured.out

    def test_covert_to_sumo_runs_netconvert(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        p = SUMOPipeline("convtest", "Loc")

        # create a dummy osm file path
        os.makedirs(os.path.dirname(p.osm_file), exist_ok=True)
        open(p.osm_file, "w").write("dummy")

        called = {}
        monkeypatch.setattr(
            "subprocess.run",
            lambda cmd, capture_output, text, check: called.__setitem__("cmd", cmd),
        )

        p.covert_to_sumo()
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
        self, monkeypatch, tmp_path, capsys
    ):
        monkeypatch.chdir(tmp_path)
        p = SUMOPipeline("dtest", "Loc")

        # ensure SUMO_HOME not set
        monkeypatch.delenv("SUMO_HOME", raising=False)

        p.generate_demand(10)
        captured = capsys.readouterr()
        assert "Please set the 'SUMO_HOME' environment variable" in captured.out

    def test_generate_demand_invokes_randomTrips(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        p = SUMOPipeline("dtest2", "Loc")

        # create fake SUMO_HOME and a dummy randomTrips.py
        fake_sumo = tmp_path / "sumo"
        tools_dir = fake_sumo / "tools"
        tools_dir.mkdir(parents=True)
        random_trips = tools_dir / "randomTrips.py"
        random_trips.write_text("#dummy")
        monkeypatch.setenv("SUMO_HOME", str(fake_sumo))

        called = {}
        monkeypatch.setattr(
            "subprocess.run", lambda cmd, check: called.__setitem__("cmd", cmd)
        )

        # create a dummy net file to reference
        os.makedirs(os.path.dirname(p.net_file), exist_ok=True)
        open(p.net_file, "w").write("net")

        p.generate_demand(20)
        assert "cmd" in called

        # command should contain path to our fake randomTrips.py
        assert any(str(random_trips) in str(c) for c in called["cmd"])

    def test_covert_to_sumo_skips_when_net_exists(self, monkeypatch, tmp_path, capsys):
        monkeypatch.chdir(tmp_path)
        p = SUMOPipeline("convskip", "Place")

        # create the net file so covert_to_sumo should be skipped
        os.makedirs(os.path.dirname(p.net_file), exist_ok=True)
        open(p.net_file, "w").write("exists")

        # monkeypatch subprocess.run to raise if called
        monkeypatch.setattr(
            "subprocess.run",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("should_not_be_called")),
        )

        p.covert_to_sumo()
        captured = capsys.readouterr()
        assert ".net.xml already exists." in captured.out
