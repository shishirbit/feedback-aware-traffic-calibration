import tempfile
import unittest
import json
from pathlib import Path
from ps1.simulation.network import build_network
from ps1.simulation.demand import generate_routes
from ps1.simulation.runner import write_config, run_offline, parse_e1
from ps1.simulation.availability import sumo_available


SUMO_OK,SUMO_REASON=sumo_available()
@unittest.skipUnless(SUMO_OK,SUMO_REASON)
class SumoIntegrationTests(unittest.TestCase):
    def test_network_build_is_byte_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            first=build_network(Path(directory)/"a")
            second=build_network(Path(directory)/"b")
            self.assertEqual(first["net_sha256"],second["net_sha256"])
            self.assertEqual(first["detectors_sha256"],second["detectors_sha256"])

    def test_short_network_run_produces_all_stations(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); network=root/"network"; episode=root/"episode"
            build_network(network)
            generate_routes(network,episode,600,11,"P0")
            config=write_config(network,episode,600,11)
            run_offline(config,episode)
            intervals,speeds,counts=parse_e1(episode/"detectors.xml")
            self.assertEqual(speeds.shape,(2,20))
            self.assertEqual(counts.shape,(2,20))
            self.assertTrue((counts.sum(axis=0)>0).all())

    def test_offline_run_applies_and_records_speed_restriction(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); network=root/"network"; episode=root/"episode"
            build_network(network)
            generate_routes(network,episode,600,22,"P0")
            config=write_config(network,episode,600,22)
            event={"edge":"m15","start_s":60,"end_s":120,"speed_mps":8.33,"restore_mps":27.78}
            result=run_offline(config,episode,[event])
            log=json.loads((episode/"physical_event_log.json").read_text(encoding="utf-8"))
            self.assertEqual(result["physical_event_actions"],2)
            self.assertEqual([row["action"] for row in log["applied"]],["restrict","restore"])
            self.assertEqual([row["time_s"] for row in log["applied"]],[60,120])


if __name__=="__main__": unittest.main()
