import tempfile
import unittest
from pathlib import Path
import numpy as np
import yaml
from ps1.config import resolve, config_hash
from ps1.simulation.runner import parse_e1, MPH_PER_MPS


ROOT=Path(__file__).resolve().parents[1]


class ConfigAndSumoTests(unittest.TestCase):
    def test_resolved_smoke_is_deterministic(self):
        config=resolve(ROOT/"configs/base.yaml",ROOT/"configs/smoke.yaml")
        self.assertEqual(config["sumo"]["episode_seconds"],10800)
        self.assertEqual(config_hash(config),config_hash(resolve(ROOT/"configs/base.yaml",ROOT/"configs/smoke.yaml")))

    def test_unknown_override_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            bad=Path(directory)/"bad.yaml"
            bad.write_text("training:\n  epohs: 2\n",encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"training.epohs"):
                resolve(ROOT/"configs/base.yaml",bad)

    def test_e1_vehicle_weighting_and_empty_invalid(self):
        xml='''<detector><interval begin="0" end="300" id="e1_0_0" nVehContrib="1" speed="10"/><interval begin="0" end="300" id="e1_0_1" nVehContrib="3" speed="20"/><interval begin="0" end="300" id="e1_1_0" nVehContrib="0" speed="-1"/><interval begin="0" end="300" id="e1_1_1" nVehContrib="0" speed="-1"/></detector>'''
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"e1.xml"; path.write_text(xml,encoding="utf-8")
            intervals,speeds,counts=parse_e1(path,stations=2,lanes=2)
        self.assertAlmostEqual(speeds[0,0],17.5*MPH_PER_MPS)
        self.assertTrue(np.isnan(speeds[0,1]))
        self.assertEqual(counts[0,0],4)


if __name__=="__main__": unittest.main()
