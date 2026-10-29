import unittest
from pathlib import Path
from types import SimpleNamespace
from rm_rl.waypoint.executor import install_goal_resolver
from rm_rl.waypoint.teachers import StrategyRunner
from rm_rl.waypoint.radar_sim import APP, BASELINE


class ExecutorTests(unittest.TestCase):
    def module(self):
        return StrategyRunner(BASELINE, str(Path(APP)/'rmuc_arena')).module

    def context(self, m):
        robot=dict(slot=7,role='sentry',x=10.8,y=7.5,hp=400,max_hp=400,weak=False,alive=True,
                   posture='defense',posture_ready_at=0,posture_enhanced=False)
        ctx=SimpleNamespace(obs={'robots':[robot],'time':0},team='red',enemy='blue',state={},memory={},time=0,
                            zone=lambda team,zone: (1.4,13) if zone=='supply' else (10.8,7.5),
                            controls=lambda *a,**kw:False)
        m.weapon=lambda *a,**kw:{}
        m.ammo_fields=lambda *a,**kw:{}
        m.drive=lambda r,g,*a,**kw: {'actual_goal':g}
        return robot,ctx

    def test_far_goal_used_by_navigation_and_posture(self):
        m=self.module();r,c=self.context(m)
        install_goal_resolver(m,lambda robot,ctx,goal:(1.4,13))
        result=m.decide(r,c)
        self.assertEqual(result['actual_goal'],(1.4,13))
        self.assertEqual(result['posture'],'move')

    def test_fallback_preserves_original_behavior(self):
        a=self.module();r,c=self.context(a);before=a.decide(r,c)
        b=self.module();r,c=self.context(b)
        install_goal_resolver(b,lambda robot,ctx,goal:goal)
        self.assertEqual(before,b.decide(r,c))

    def test_near_override_does_not_keep_original_far_move(self):
        m=self.module();r,c=self.context(m);r['weak']=True
        install_goal_resolver(m,lambda robot,ctx,goal:(10.8,7.5))
        result=m.decide(r,c)
        self.assertEqual(result['actual_goal'],(10.8,7.5))
        self.assertNotEqual(result.get('posture'),'move')

    def test_reject_double_install(self):
        m=self.module();install_goal_resolver(m,lambda r,c,g:g)
        with self.assertRaises(ValueError):install_goal_resolver(m,lambda r,c,g:g)

if __name__=='__main__':unittest.main()
