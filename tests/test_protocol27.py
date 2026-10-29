import copy
import json
import struct
import unittest
import numpy as np
from rm_rl.waypoint import protocol27 as p
from rm_rl.waypoint.protocol27_sim import ProtocolAdapter
from rm_rl.waypoint.radar import RadarConfig

class ProtocolTests(unittest.TestCase):
    def setUp(self):self.x=p.ProtocolInput('red',lambda x,y:(x,y))
    def test_signed_damage_physical_outpost_and_scores(self):
        b=bytearray(24);struct.pack_into('<h',b,8,-1234)
        for o,v in ((12,333),(16,777),(20,123),(22,456)):struct.pack_into('<H',b,o,v)
        red=self.x.decode(3,b);blue=p.ProtocolInput('blue',lambda x,y:(x,y)).decode(3,b)
        self.assertEqual(red['damage_difference'],-1234)
        self.assertEqual(red['ally.outpost_hp'],333);self.assertEqual(blue['ally.outpost_hp'],777)
        self.assertEqual(red['enemy.control_degree'],456)
    def test_coordinate_units_reserved_slots_and_calibration(self):
        b=bytearray(24);struct.pack_into('<HH',b,20,1234,567)
        x=p.ProtocolInput('blue',lambda x,y:(x+1,y+2)).decode(0xa01,b)
        self.assertAlmostEqual(x['enemy.7.x'],14.66);self.assertAlmostEqual(x['enemy.7.y'],7.33)
        self.assertFalse(any('.2.' in k or '.6.' in k for k in x))
        with self.assertRaises(ValueError):p.ProtocolInput('red',None)
    def test_control_none_does_not_become_enemy_full(self):
        d=self.x.decode(0x101,struct.pack('<II',2<<28,0))
        self.assertEqual(d['ally.outpost.ally_control_none'],1)
        self.assertNotIn('ally.outpost.enemy_control_full',d)
        d=self.x.decode(0xa04,struct.pack('<III',0,0,1<<22))
        self.assertEqual(d['ally.outpost.enemy_control_temporary'],1)
        d=self.x.decode(0x101,struct.pack('<II',3<<28,0))
        self.assertNotIn('ally.outpost.ally_control_full',d)
    def test_buff_percent_and_posture(self):
        b=bytearray(47);b[31]=50;b[32]=25;struct.pack_into('<H',b,33,50);b[35]=6;b[40]=3
        d=self.x.decode(0xa05,b)
        self.assertEqual(d['enemy.7.attack_multiplier'],1.5)
        self.assertEqual(d['enemy.7.defense_buff'],.5)
        self.assertEqual(d['enemy.7.posture_move'],1)
        self.assertEqual(d['enemy.7.weak'],1)
        self.assertFalse(any('armor' in k for k in d))
    def test_route_size_identity_rejected(self):
        with self.assertRaises(ValueError):self.x.ingest(0x203,bytes(12),0,.1,'direct')
        with self.assertRaises(ValueError):self.x.decode(0xa01,bytes(23))
        with self.assertRaises(ValueError):self.x.decode(0x201,bytes(17))
    def test_freshness_uses_upstream_not_forward_time(self):
        b=struct.pack('<fff',3,4,0)
        self.x.ingest(0x203,b,0,.26,'relay')
        self.assertIsNone(p.value(self.x.vector(.2),'ally.7.x'))
        self.assertAlmostEqual(p.value(self.x.vector(.3),'ally.7.x'),3,places=5)
        self.x.ingest(0x203,b,0,1.9,'relay')
        self.assertIsNone(p.value(self.x.vector(2.01),'ally.7.x'))
        with self.assertRaises(ValueError):self.x.vector(1)
    def test_no_unproven_features(self):
        for f in p.FIELDS:
            self.assertNotIn('occupied',f.name);self.assertNotIn('heading',f.name)
            self.assertNotIn('progress',f.name)
        self.assertNotIn('ally.7.weak',p.FIELD_INDEX)
        self.assertNotIn('ally.7.invulnerable',p.FIELD_INDEX)
    def test_own_allowance_and_reserve_separate(self):
        d=self.x.decode(0x208,struct.pack('<HHHH',23,0,100,150))
        self.assertEqual(d['ally.7.allowance'],23)
        self.assertEqual(d['ally.7.fortress_reserve'],150)
    def test_clock_only_battle(self):
        b=bytearray(11);b[0]=0x31
        self.assertEqual(self.x.decode(1,b),{})

class SimulatorBoundaryTests(unittest.TestCase):
    def frame(self,t=0):
        from test_radar27 import frame
        f=frame(t)
        for r in f['robots']:
            r.update(level=1,cooling_rate=30,cooling_bonus=0,defense_buff=.5,
                     vulnerability=.25,attack_multiplier=1,buffer_energy_j=60,
                     invulnerable=False,weak=False)
        for side in ('red','blue'):f['teams'][side].update(control=12,fortress_ammo=90)
        f['positions']={side+'_'+kind:dict(controller=side,control='full',progress=.7)
                        for side in ('red','blue') for kind in ('outpost','base')}
        return f
    def adapter(self,**kw):return ProtocolAdapter(RadarConfig(latency_s=0,loss=0,**kw))
    def test_truth_is_filtered_and_ammo_semantics(self):
        f=self.frame();g=copy.deepcopy(f)
        for r in g['robots']:
            r.update(heading=100,vx=100,damage=999,target='secret')
            if r['team']=='red':r.update(weak=True,invulnerable=True)
        for pos in g['positions'].values():pos['progress']=999
        g['zones']={}
        a=self.adapter().observe(f);b=self.adapter().observe(g)
        np.testing.assert_array_equal(a,b)
        self.assertEqual(round(p.value(a,'enemy.7.allowance')),190)
        self.assertEqual(round(p.value(a,'ally.7.allowance')),100)
        self.assertEqual(round(p.value(a,'ally.7.fortress_reserve')),90)
        self.assertTrue(np.isfinite(a).all())
    def test_two_hops_and_one_hz_position(self):
        a=ProtocolAdapter(RadarConfig(latency_s=.13,loss=0))
        a.observe(self.frame())
        v=a.observe(self.frame(.2))
        self.assertIsNotNone(p.value(v,'enemy.7.x'));self.assertIsNone(p.value(v,'ally.7.x'))
        v=a.observe(self.frame(.3));self.assertIsNotNone(p.value(v,'ally.7.x'))
        f=self.frame(.9);next(r for r in f['robots'] if r['id']=='red-7')['x']=10
        v=a.observe(f);self.assertAlmostEqual(p.value(v,'ally.7.x'),3.123,places=4)
        self.assertAlmostEqual(v[p.FIELD_INDEX['ally.7.x']*3+2]*5,.9,places=5)
    def test_no_relay_no_self_telemetry(self):
        v=self.adapter(sentry_relay=False).observe(self.frame())
        self.assertIsNone(p.value(v,'ally.7.x'));self.assertIsNone(p.value(v,'ally.7.max_hp'))
        self.assertEqual(round(p.value(v,'ally.7.hp')),300)
    def test_blue_canonical(self):
        f=self.frame()
        a=self.adapter().observe(f)
        b=ProtocolAdapter(RadarConfig(latency_s=0,loss=0),team='blue').observe(f)
        self.assertAlmostEqual(p.value(a,'ally.7.x'),p.value(b,'ally.7.x'),places=5)

if __name__=='__main__':unittest.main()
