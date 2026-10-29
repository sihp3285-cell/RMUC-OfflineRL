"""Arena truth -> rate-limited synthetic protocol payloads -> shared decoder.

This is the only new-schema module permitted to access full simulation state.
No zone occupancy is translated to activation. No capture progress is exposed.
"""
import heapq
import math
import random
import struct
from .radar import RadarConfig, GROUND_SLOTS
from .protocol27 import ProtocolInput, MESSAGES

class ProtocolAdapter:
    def __init__(self,config=None,team='red',seed=0):
        self.config=config or RadarConfig();self.team=team;self.rng=random.Random(seed)
        self.input=ProtocolInput(team,lambda x,y:(x,y),self.config)
        self.next_sample={c:0. for c in MESSAGES};self.pending=[];self.seq=0
        self.sent=self.dropped=0;self.last_time=-1.

    def payload(self,frame,cmd):
        if str(frame.get('season'))!='2027' or frame.get('arena',{}).get('name')!='rmuc2027':raise ValueError('requires rmuc2027')
        other='blue' if self.team=='red' else 'red'
        robots={(r['team'],int(r['slot'])):r for r in frame['robots']}
        b=bytearray(MESSAGES[cmd][0])
        def put(fmt,o,v):
            if fmt in ('B','H','h','I'):
                lo,hi={'B':(0,255),'H':(0,65535),'h':(-32768,32767),'I':(0,2**32-1)}[fmt]
                v=max(lo,min(hi,int(round(v))))
            struct.pack_into('<'+fmt,b,o,v)
        def xy(r,o,cm=False):
            x=r['x']+self.rng.gauss(0,self.config.position_noise_m)
            y=r['y']+self.rng.gauss(0,self.config.position_noise_m)
            if cm:put('H',o,x*100);put('H',o+2,y*100)
            else:put('f',o,x);put('f',o+4,y)
        def buffs(r,o):
            put('H',o+1,r['cooling_bonus']);put('B',o+3,100*r['defense_buff'])
            put('B',o+4,100*r['vulnerability']);put('H',o+5,100*(r['attack_multiplier']-1))
        def control(physical,kind,view):
            p=frame['positions'][physical+'_'+kind]
            if p['controller']!=view:return 2
            if p['control']=='full':return 0
            if p['control']=='temporary':return 1
            # Unknown simulator control semantics must fail, not become "full".
            raise ValueError(f'unrecognized control state: {p}')
        own=robots[self.team,7]
        if cmd==0x0001:put('B',0,0x41);put('H',1,math.floor(frame['remaining']))
        elif cmd==0x0003:
            for slot,o in ((1,0),(3,4),(4,6),(7,10)):put('H',o,robots[self.team,slot]['hp'])
            put('h',8,frame['teams'][self.team]['attack_damage']-frame['teams'][other]['attack_damage'])
            for team,o in (('red',12),('blue',16)):put('H',o,frame['outposts'][team]['hp'])
            for team,o in ((self.team,14),(other,18)):put('H',o,frame['bases'][team]['hp'])
            for team,o in ((self.team,20),(other,22)):put('H',o,frame['teams'][team]['control'])
        elif cmd in (0x0101,0x0A04):
            if cmd==0x0101:
                # Unused activation bits are marked hidden; supply omitted by schema.
                a=sum(4<<s for s in (5,8,11,14,22,25))+(3<<18)+(3<<20)
                a|=control('red','outpost',self.team)<<28|control('blue','outpost',self.team)<<30
                z=control('red','base',self.team)|control('blue','base',self.team)<<2
                put('I',0,a);put('I',4,z)
            else:
                put('I',4,sum(4<<s for s in (1,4,7,10,18,21))+(3<<14)+(3<<16))
                put('I',8,sum(control(p,k,other)<<s for p,k,s in [('red','outpost',22),('blue','outpost',24),('red','base',26),('blue','base',28)]))
        elif cmd==0x0A01:
            for slot,o in ((1,0),(3,8),(4,12),(7,20)):xy(robots[other,slot],o,True)
        elif cmd in (0x0A02,0x0A03):
            offsets=((1,0),(3,4),(4,6),(7,10)) if cmd==0x0A02 else ((1,0),(3,2),(4,4),(7,8))
            for slot,o in offsets:
                r=robots[other,slot]
                v=r['hp'] if cmd==0x0A02 else r['ammo']+(frame['teams'][other]['fortress_ammo'] if slot!=1 else 0)
                put('H',o,v)
        elif cmd==0x0A05:
            for slot,o,s in ((1,0,36),(3,14,38),(4,21,39),(7,28,40)):
                r=robots[other,slot];buffs(r,o)
                put('B',s,1 if not r['alive'] else (3 if r['weak'] else 2) if r['invulnerable'] else 0)
            put('B',35,('attack','defense','move').index(robots[other,7]['posture'])+1)
        elif cmd==0x0201:
            put('B',0,7 if self.team=='red' else 107);put('B',1,own['level'])
            for k,o in [('hp',2),('max_hp',4),('cooling_rate',6),('max_heat',8)]:put('H',o,own[k])
        elif cmd==0x0202:put('H',8,own['buffer_energy_j']);put('H',10,own['heat'])
        elif cmd==0x0203:xy(own,0)
        elif cmd==0x0204:buffs(own,0)
        elif cmd==0x0208:
            put('H',0,own['ammo']);put('H',6,frame['teams'][self.team]['fortress_ammo'])
        elif cmd==0x020B:
            for slot,o in ((1,0),(3,16),(4,24)):xy(robots[self.team,slot],o)
        elif cmd==0x020D:put('H',4,(('attack','defense','move').index(own['posture'])+1)<<12)
        return bytes(b)

    def observe(self,frame):
        now=float(frame['time'])
        if not math.isfinite(now) or now<self.last_time:raise ValueError('time moved backwards')
        self.last_time=now;cfg=self.config
        for cmd,(_,hz,route) in MESSAGES.items():
            if route=='relay' and (not cfg.sentry_relay or cmd==0x020B and not cfg.ally_relay):continue
            if route=='wireless' and not cfg.enemy_feed:continue
            if now+1e-8<self.next_sample[cmd]:continue
            self.next_sample[cmd]=(math.floor(now*hz+1e-8)+1)/hz
            hops=2 if route=='relay' else 1
            self.sent+=1
            if any(self.rng.random()<cfg.loss for _ in range(hops)):
                self.dropped+=1;continue
            payload=self.payload(frame,cmd)
            heapq.heappush(self.pending,(now+hops*cfg.latency_s,self.seq,cmd,now,payload,route));self.seq+=1
        while self.pending and self.pending[0][0]<=now+1e-8:
            arrival,_,cmd,stamp,payload,route=heapq.heappop(self.pending)
            self.input.ingest(cmd,payload,stamp,arrival,route)
        return self.input.vector(now)
