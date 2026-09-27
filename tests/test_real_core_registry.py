import json
import unittest
from pathlib import Path


class RealCoreRegistryTests(unittest.TestCase):
    def test_locked_core_scope_matches_protocol(self):
        root = Path(__file__).resolve().parents[1]
        registry = json.loads((root / "configs/real_core_c1_c2_c5.json").read_text(encoding="utf-8"))
        self.assertEqual(registry["status"], "locked_before_execution")
        self.assertEqual(registry["backbone"], "far_gw")
        self.assertEqual(registry["datasets"], ["metr_la", "pems_bay"])
        self.assertEqual(registry["scenarios"], ["C1", "C2", "C5"])
        self.assertEqual(registry["model_seeds"], [11, 22, 33])
        self.assertEqual(registry["fault_seeds"], [101, 202, 303])
        self.assertIn("not severity-matched", registry["interpretation"]["C5"])


if __name__ == "__main__":
    unittest.main()
