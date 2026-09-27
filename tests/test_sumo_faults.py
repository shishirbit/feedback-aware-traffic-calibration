import tempfile
import unittest
from pathlib import Path

import numpy as np

from ps1.simulation.sumo_faults import generate_sumo_faults,derive_immediate_feedback_oracle
from ps1.utils.artifacts import write_json


class SumoFaultTests(unittest.TestCase):
    def test_faults_reset_by_episode_and_match_registered_rate(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); prepared=root/"sumo.npz"; source=root/"sumo.manifest.json"
            episodes=np.repeat(np.arange(30),96); partition=np.full(len(episodes),3,dtype=np.uint8)
            adjacency=np.eye(20,k=1,dtype=np.float32)+np.eye(20,k=-1,dtype=np.float32)
            np.savez(prepared,episode_index=episodes,partition_code=partition,adjacency=adjacency)
            write_json(source,{"fixture":True})
            config={"faults":{"connected_fraction":.1,"duration_bins":6,"backlog_release_bins":3,"events_per_day":2}}
            result=generate_sumo_faults(prepared,source,"C4",101,root/"faults",config)
            self.assertEqual(result["event_count"],20)
            self.assertEqual(len(result["affected_episode_indices"]),20)
            with np.load(root/"faults/release_schedule.npz") as schedule:
                for episode in range(30):
                    rows=slice(episode*96,(episode+1)*96)
                    arrivals=schedule["input_arrival"][rows]
                    self.assertFalse(((arrivals>=0)&((arrivals<episode*96)|(arrivals>=(episode+1)*96))).any())

    def test_immediate_feedback_oracle_preserves_c4_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); source=root/"source"; source.mkdir()
            inputs=np.array([[0,-1],[2,3]],dtype=np.int32); feedback=np.array([[0,-1],[2,-1]],dtype=np.int32)
            np.savez_compressed(source/"release_schedule.npz",input_arrival=inputs,feedback_arrival=feedback)
            write_json(source/"fault_manifest.json",{"scenario_id":"C4","seed":101,"schema_version":1,"events":[],"realized":{}})
            result=derive_immediate_feedback_oracle(source,root/"oracle")
            with np.load(root/"oracle/release_schedule.npz") as oracle:
                self.assertTrue(np.array_equal(oracle["input_arrival"],inputs))
                self.assertTrue(np.array_equal(oracle["feedback_arrival"],np.array([[0,0],[1,1]])))
            self.assertEqual(result["scenario_id"],"C4I")
            self.assertEqual(result["realized"]["feedback_unreleased_fraction"],0.)

    def test_tuning_faults_use_only_tuning_episodes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); prepared=root/"sumo.npz"; source=root/"sumo.manifest.json"
            episodes=np.repeat(np.arange(4),96); partition=np.repeat(np.asarray([0,1,1,3],dtype=np.uint8),96)
            adjacency=np.eye(20,k=1,dtype=np.float32)+np.eye(20,k=-1,dtype=np.float32)
            np.savez(prepared,episode_index=episodes,partition_code=partition,adjacency=adjacency); write_json(source,{"fixture":True})
            config={"faults":{"connected_fraction":.1,"duration_bins":6,"backlog_release_bins":3,"events_per_day":2}}
            result=generate_sumo_faults(prepared,source,"C3",101,root/"faults",config,"tune")
            self.assertEqual(result["partition"],"tune"); self.assertEqual(result["episode_indices"],[1,2])
            self.assertEqual(result["partition_start_row"],96); self.assertEqual(result["partition_length"],192)


if __name__=="__main__": unittest.main()
