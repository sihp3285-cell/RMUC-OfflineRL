import unittest
from rm_rl.waypoint.radar_matrix import aggregate

class MatrixTests(unittest.TestCase):
    def test_pairing_preserves_side_and_clusters_seeds(self):
        rows=[]
        for seed in (1,2):
            for side,offset in (('red',0),('blue',100)):
                for mode,delta in (('baseline',0),('rule',1),('radar',2),('radar_hysteresis',5)):
                    rows.append(dict(seed=seed,side=side,mode=mode,winner=side,metrics=dict(damage=offset+seed+delta,optional=None),prediction_errors=0,candidate_policy_errors=0))
        s=aggregate(rows)
        p=s['hysteresis_minus_reference']['radar']['both']
        self.assertEqual(p['complete_seed_pairs'],2)
        self.assertEqual(p['damage']['mean_delta'],3)
        self.assertEqual(p['damage']['seed_bootstrap_95_ci'],[3,3])
        self.assertEqual(s['means']['both']['radar']['n'],4)
        self.assertIsNone(p['optional']['mean_delta'])
        rows=[r for r in rows if (r['seed'],r['side'],r['mode'])!=(2,'blue','radar')]
        s=aggregate(rows)
        self.assertEqual(s['hysteresis_minus_reference']['radar']['both']['complete_seed_pairs'],1)
        self.assertEqual(s['hysteresis_minus_reference']['radar']['red']['complete_seed_pairs'],2)

if __name__=='__main__':unittest.main()
