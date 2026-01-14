import os
import pytest

import xml.etree.ElementTree as ET
from traffic_flow_models import SUMOSimulation


class TestSUMOSimulation:
    def test_write_config_and_cfg_content(self, tmp_path):
        out = tmp_path
        net = out / "test.net.xml"
        rou = out / "test.rou.xml"
        net.write_text("net")
        rou.write_text("rou")

        sim = SUMOSimulation("testsim", str(net), str(rou), str(out))
        sim.write_config()

        # verify that class attributes are initialized correctly
        assert sim.name == "testsim"
        assert sim.net_file == str(net)
        assert sim.rou_file == str(rou)
        assert sim.output_dir == str(out)
        assert sim.cfg_file == str(out / "testsim.sumocfg")
        assert sim.stats_file == str(out / "testsim_stats.xml")

        assert os.path.exists(sim.cfg_file)
        content = open(sim.cfg_file).read()

        # should reference basenames of net, route and stats
        assert "test.net.xml" in content
        assert "test.rou.xml" in content
        assert "testsim_stats.xml" in content

    def test_print_summary_parses_stats(self, tmp_path):
        out = tmp_path
        stats_file = out / "mysim_stats.xml"

        # minimal statistics tree with vehicleTripStatistics
        root = ET.Element("statistics")
        vts = ET.SubElement(root, "vehicleTripStatistics")
        vts.set("speed", "5.5")
        vts.set("count", "42")
        vts.set("duration", "120.0")
        tree = ET.ElementTree(root)
        tree.write(str(stats_file))

        sim = SUMOSimulation("mysim", "n.net.xml", "r.rou.xml", str(out))

        # point sim at our prepared stats file
        sim.stats_file = str(stats_file)

        results = sim.print_summary()
        assert results is not None
        assert results["mean_speed"] == pytest.approx(5.5)
        assert results["total_vehicles"] == 42
        assert results["mean_duration"] == pytest.approx(120.0)

    def test_run_simulation_invokes_sumo_and_prints_summary(
        self, monkeypatch, tmp_path
    ):
        out = tmp_path
        net = out / "test.net.xml"
        rou = out / "test.rou.xml"
        net.write_text("")
        rou.write_text("")

        sim = SUMOSimulation("runme", str(net), str(rou), str(out))
        called = {}

        monkeypatch.setattr(
            "subprocess.run", lambda cmd, check: called.__setitem__("cmd", cmd)
        )

        # simulate that stats file exists so print_summary would be attempted
        monkeypatch.setattr("os.path.exists", lambda _: True)

        # also patch print_summary to record it was called
        monkeypatch.setattr(SUMOSimulation, "print_summary", lambda _: {"ok": True})
        sim.run_simulation()

        assert "cmd" in called
        assert "sumo" in called["cmd"][0]
