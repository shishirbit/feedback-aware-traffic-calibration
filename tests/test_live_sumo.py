import tempfile
import unittest
from pathlib import Path
from ps1.simulation.network import build_network
from ps1.simulation.demand import generate_routes
from ps1.simulation.runner import write_config
from ps1.simulation.live import run_live
from ps1.simulation.availability import sumo_available


SUMO_OK,SUMO_REASON=sumo_available()
@unittest.skipUnless(SUMO_OK,SUMO_REASON)
class LiveSumoTests(unittest.TestCase):
    def test_traci_matches_offline_e1_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); network=root/"network"; episode=root/"episode"
            build_network(network); generate_routes(network,episode,600,44,"P0")
            config=write_config(network,episode,600,44)
            result=run_live(config,episode)
            self.assertTrue(result["equivalent"],result)


if __name__=="__main__": unittest.main()
