import unittest
from collections import Counter

from ps1.simulation.paper import FAMILY_COUNTS, SPLIT_COUNTS, paper_episode_plan


class SumoPaperPlanTests(unittest.TestCase):
    def test_registered_plan_is_unique_balanced_and_complete(self):
        plan=paper_episode_plan()
        self.assertEqual(len(plan),120)
        self.assertEqual(len({row["seed"] for row in plan}),120)
        self.assertEqual(len({row["episode_id"] for row in plan}),120)
        self.assertEqual(Counter(row["split"] for row in plan),Counter(SPLIT_COUNTS))
        self.assertEqual(Counter(row["family"] for row in plan),Counter({"P0":30,"P1":30,"P2":30,"P3":30}))
        for split,counts in FAMILY_COUNTS.items():
            self.assertEqual(Counter(row["family"] for row in plan if row["split"]==split),Counter(counts))


if __name__=="__main__": unittest.main()
