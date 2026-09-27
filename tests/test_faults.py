import unittest
import numpy as np
from ps1.online.faults import generate_fault_schedule


class FaultScheduleTests(unittest.TestCase):
    def setUp(self):
        self.adjacency=np.zeros((20,20))
        for i in range(19): self.adjacency[i,i+1]=self.adjacency[i+1,i]=1

    def test_c3_is_coupled_seeded_and_overtakable(self):
        first=generate_fault_schedule("C3",288,self.adjacency,101)
        second=generate_fault_schedule("C3",288,self.adjacency,101)
        np.testing.assert_array_equal(first.input_arrival,first.feedback_arrival)
        np.testing.assert_array_equal(first.input_arrival,second.input_arrival)
        event=first.events[0]; sensors=event["affected_sensors"]
        self.assertTrue(event["graph_connected"])
        self.assertTrue((first.input_arrival[event["start_bin"]:event["end_bin"],sensors]>=event["end_bin"]).all())
        self.assertTrue((first.input_arrival[event["end_bin"],sensors]==event["end_bin"]).all())

    def test_channel_isolation(self):
        c1=generate_fault_schedule("C1",288,self.adjacency,101)
        c2=generate_fault_schedule("C2",288,self.adjacency,101)
        baseline=np.broadcast_to(np.arange(288)[:,None],(288,20))
        np.testing.assert_array_equal(c1.input_arrival,baseline)
        np.testing.assert_array_equal(c2.feedback_arrival,baseline)
        self.assertTrue((c1.feedback_arrival!=baseline).any())
        self.assertTrue((c2.input_arrival!=baseline).any())

    def test_loss_uses_negative_infinity_sentinel_only_in_private_schedule(self):
        c4=generate_fault_schedule("C4",288,self.adjacency,101,loss_within=1.)
        self.assertTrue((c4.input_arrival<0).any()); self.assertTrue((c4.feedback_arrival<0).any())
        self.assertEqual(c4.arrival(0,0,"input"),0)
        event=c4.events[0]; self.assertEqual(c4.arrival(int(event["affected_sensors"][0]),event["start_bin"],"input"),float("inf"))

    def test_c5_realized_rates_are_reported(self):
        c5=generate_fault_schedule("C5",1000,self.adjacency,303)
        rate=c5.manifest()["realized"]["input_unreleased_fraction"]
        self.assertGreater(rate,.08); self.assertLess(rate,.12)


if __name__=="__main__": unittest.main()
