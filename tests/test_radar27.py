import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from rm_rl.waypoint.radar import (RadarAdapter, RadarConfig, FEATURE_DIM, FEATURE_NAMES,
                                  FIELD_INDEX, SCHEMA, value, canonical_xy)
from rm_rl.waypoint.radar_sim import metrics, Candidate


def frame(t=0):
    robots = []
    for side in ('red', 'blue'):
        for slot, role in ((1, 'heavy'), (3, 'infantry'), (4, 'infantry'), (7, 'sentry')):
            x, y = canonical_xy(3.123, 5.456, side)
            robots.append(dict(id=f'{side}-{slot}', team=side, slot=slot, role=role,
                               x=x, y=y, hp=300, max_hp=400, heat=20, max_heat=260,
                               ammo=100, alive=True, posture='move', heading=0 if side=='red' else np.pi))
        robots.append(dict(id=f'{side}-6', team=side, slot=6, role='aerial', hp=0, x=14, y=7))
    return dict(season='2027', arena={'name': 'rmuc2027'}, time=t, remaining=420-t,
                robots=robots, teams={side:dict(resources=100, attack_damage=0) for side in ('red','blue')},
                bases={side:dict(hp=10000) for side in ('red','blue')},
                outposts={side:dict(hp=1000) for side in ('red','blue')},
                zones={'red_central':dict(occupants=['blue-7'])})


def instant(**kwargs):
    return RadarAdapter(RadarConfig(latency_s=0, loss=0, **kwargs))


class RadarTests(unittest.TestCase):
    def test_shape_and_quantization(self):
        x = instant().observe(frame())
        self.assertEqual(x.shape, (FEATURE_DIM,))
        self.assertTrue(np.isfinite(x).all())
        self.assertAlmostEqual(value(x, 'ally.7.x'), 3.12, places=5)
        self.assertAlmostEqual(value(x, 'ally.7.y'), 5.46, places=5)

    def test_no_engineer_or_aircraft_slots(self):
        self.assertFalse(any('.2.' in n or '.6.' in n for n in FEATURE_NAMES))
        f = frame(); g = copy.deepcopy(f)
        for r in g['robots']:
            if r['slot']==6: r.update(hp=9999, ammo=9999, x=-100)
        g['robots'].append(dict(team='red', slot=2, role='engineer', x=999))
        np.testing.assert_array_equal(instant().observe(f), instant().observe(g))

    def test_disallowed_truth_does_not_leak(self):
        f = frame(); g = copy.deepcopy(f)
        for r in g['robots']:
            r.update(vx=999, vy=999, z=999, damage=999, kills=999, armor_hit_times=[0])
            if r['team']=='blue': r.update(heat=999, max_heat=999, max_hp=999)
        for t in g['teams'].values(): t.update(control=999, kills=999, tech_income=999)
        g['positions']={'red_central':dict(progress=1, controller='blue')}
        np.testing.assert_array_equal(instant().observe(f), instant().observe(g))

    def test_missing_not_zero(self):
        f = frame()
        r = next(r for r in f['robots'] if r['id']=='blue-7'); r.pop('hp')
        a = instant().observe(f); r['hp']=0; b=instant().observe(f)
        self.assertIsNone(value(a,'enemy.7.hp')); self.assertEqual(value(b,'enemy.7.hp'),0)

    def test_air_system_cannot_set_ground_occupancy(self):
        f = frame()
        f['zones']['red_central']['occupants'] = ['blue-6', 'blue-drone-1']
        x = instant().observe(f)
        self.assertEqual(value(x, 'ally.central.occupied_enemy'), 0)

    def test_nonfinite_masked(self):
        f=frame(); f['robots'][0]['hp']=float('nan')
        x=instant().observe(f)
        self.assertIsNone(value(x,'ally.1.hp')); self.assertTrue(np.isfinite(x).all())

    def test_latency_and_no_future(self):
        a=RadarAdapter(RadarConfig(latency_s=.13,loss=0))
        self.assertIsNone(value(a.observe(frame()),'enemy.7.x'))
        self.assertIsNone(value(a.observe(frame(.1)),'enemy.7.x'))
        x=a.observe(frame(.2))
        self.assertIsNotNone(value(x,'enemy.7.x'))
        self.assertAlmostEqual(float(x[3*FIELD_INDEX['enemy.7.x']+2])*5,.2,places=5)
        with self.assertRaises(ValueError): a.observe(frame(.1))

    def test_disabled_relay(self):
        x=instant(sentry_relay=False).observe(frame())
        self.assertIsNone(value(x,'ally.7.x')); self.assertIsNone(value(x,'ally.1.x'))
        self.assertEqual(round(value(x,'ally.7.hp')),300)

    def test_disabled_enemy(self):
        x=instant(enemy_feed=False).observe(frame())
        self.assertIsNone(value(x,'enemy.7.hp'))
        self.assertEqual(round(value(x,'enemy.base_hp')),10000)

    def test_stale_and_total_loss(self):
        a=instant(); a.observe(frame())
        a.config=RadarConfig(latency_s=0,loss=1)
        x=a.observe(frame(3))
        self.assertIsNone(value(x,'ally.7.x'))
        b=RadarAdapter(RadarConfig(loss=1))
        self.assertEqual(float(b.observe(frame())[1::3].sum()),0)

    def test_blue_rotation(self):
        f=frame(); a=instant().observe(f)
        b=RadarAdapter(RadarConfig(latency_s=0,loss=0),team='blue').observe(f)
        self.assertAlmostEqual(value(a,'ally.7.x'),value(b,'ally.7.x'),places=5)
        self.assertAlmostEqual(value(a,'ally.7.heading_cos'),value(b,'ally.7.heading_cos'),places=5)

    def test_reject_wrong_season_and_role(self):
        f=frame();f['season']='2026'
        with self.assertRaises(ValueError): instant().observe(f)
        f=frame();f['robots'][0]['role']='hero'
        with self.assertRaises(ValueError): instant().observe(f)

    def test_seed_reproducible(self):
        a=RadarAdapter(seed=17);b=RadarAdapter(seed=17)
        for t in np.arange(0,2,.1): np.testing.assert_array_equal(a.observe(frame(t)),b.observe(frame(t)))

    def test_rate_is_not_per_decision(self):
        a=instant()
        for t in (0,.1,.2,.3,.4): a.observe(frame(t))
        stamp,_=a.received['broadcast_hp']; self.assertAlmostEqual(stamp,.4)
        stamp,_=a.received['clock']; self.assertEqual(stamp,0)

    def test_lethal_damage_and_elapsed_time(self):
        a=dict(x=1,y=1,hp=30,alive=True,heat=0,max_heat=100,damage=5,vx=0,vy=0)
        b=dict(a,hp=0,alive=False,damage=25)
        result=metrics([dict(before=a,after=b,dt=.2,t=0,goal=[1,1])])
        self.assertEqual(result['damage_taken_hp_proxy'],30)
        self.assertEqual(result['deaths'],1)
        self.assertEqual(result['alive_time_s'],.2)
        self.assertEqual(result['arrival_rate'],0)
        self.assertEqual(result['goal_switches'],0)

    def test_hysteresis_candidate_fallback_and_urgent(self):
        module = SimpleNamespace(drive=lambda *a: None, act=lambda obs, state: ({}, state))
        args = SimpleNamespace(baseline='unused', side='blue', ckpt='unused', confirm_s=1., min_hold_s=2.)
        with patch('rm_rl.waypoint.radar_sim.StrategyRunner', return_value=SimpleNamespace(module=module)), patch('rm_rl.waypoint.radar_sim.Predictor') as predictor:
            predictor.return_value.predict.return_value = (10.8, 7.5)
            candidate = Candidate(args, 'unused', 'radar_hysteresis')
            candidate.features = instant().observe(frame())
            candidate.act({'time': 0}, None)
            np.testing.assert_allclose(candidate.selected, (17.2, 7.5))
            predictor.return_value.predict.return_value = (1.4, 13.)
            f = frame(.1)
            next(r for r in f['robots'] if r['id'] == 'red-7')['ammo'] = 10
            candidate.features = instant().observe(f)
            candidate.act({'time': .1}, None)
            self.assertEqual(candidate.filter_reason, 'urgent')
            np.testing.assert_allclose(candidate.selected, (26.6, 2.))
            candidate.features = np.zeros(FEATURE_DIM, dtype=np.float32)
            candidate.act({'time': .2}, None)
            self.assertIsNone(candidate.selected)
            self.assertIsNone(candidate.filter.current)
            self.assertEqual(candidate.fallbacks, 1)
            candidate.features = instant().observe(frame(.3))
            candidate.act({'time': .3}, None)
            self.assertEqual(candidate.filter_reason, 'initial')
            predictor.return_value.predict.side_effect = ValueError('bad input')
            candidate.act({'time': .4}, None)
            self.assertIsNone(candidate.filter.current)
            self.assertEqual(candidate.prediction_errors, 1)

    def test_low_level_errors_are_observed_before_internal_catch(self):
        def broken(*args):
            raise RuntimeError('test controller failure')
        module = SimpleNamespace(drive=lambda *a: None, decide=broken)
        with patch('rm_rl.waypoint.radar_sim.StrategyRunner', return_value=SimpleNamespace(module=module)):
            candidate = Candidate(SimpleNamespace(baseline='unused', side='red'), 'unused', 'baseline')
        with self.assertRaises(RuntimeError):
            module.decide(None, None)
        self.assertEqual(candidate.policy_errors, 1)
        self.assertEqual(candidate.policy_error_messages, ['test controller failure'])

if __name__=='__main__': unittest.main()
