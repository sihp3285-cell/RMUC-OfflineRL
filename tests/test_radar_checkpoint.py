import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from rm_rl.waypoint.radar import SCHEMA, FEATURE_DIM, contract, ANCHORS
from rm_rl.waypoint.model import WaypointClassifier
from rm_rl.waypoint.radar_sim import Predictor


class ContractTests(unittest.TestCase):
    def test_legacy_four_point_checkpoint_and_wrong_contract(self):
        model=WaypointClassifier(FEATURE_DIM,4,16,2)
        state=dict(observation_schema=SCHEMA,observation_contract=contract(),obs_dim=FEATURE_DIM,
                   n_classes=4,hidden=16,depth=2,state_dict=model.state_dict(),
                   mean=np.zeros(FEATURE_DIM,dtype=np.float32),std=np.ones(FEATURE_DIM,dtype=np.float32),anchors=ANCHORS)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'test.pt';torch.save(state,p)
            predictor=Predictor(p,SCHEMA)
            np.testing.assert_array_equal(predictor.supply_anchor,ANCHORS[3])
            goal=predictor.predict(np.zeros(FEATURE_DIM,dtype=np.float32))
            self.assertTrue(any(np.array_equal(goal,a) for a in ANCHORS))
            with self.assertRaises(ValueError):Predictor(p,'unsupported-experimental-schema')
            state['observation_contract']={'schema':SCHEMA}
            torch.save(state,p)
            with self.assertRaisesRegex(ValueError,'contract mismatch'):Predictor(p,SCHEMA)

if __name__=='__main__':unittest.main()
