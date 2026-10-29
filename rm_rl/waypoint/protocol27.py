"""2027 V1.0.0 payload boundary (subset), independent of Arena.

ROS/serial integration must validate frames/CRC and provide source timestamps.
Coordinates must be transformed by a calibrated field_to_arena function.
Little-endian payload layout is an explicit bridge convention to verify with
hardware captures. This module does not implement serial transport or 0x0301.
"""
from dataclasses import asdict
import math
import struct
import numpy as np
from .radar import Field, GROUND_SLOTS, RadarConfig, canonical_xy

SCHEMA = 'radar27-protocol-20261029-v1'
# (payload length, source rate Hz, route). Relay rates are upstream rates.
MESSAGES = {0x0001:(11,1,'direct'), 0x0003:(24,3,'direct'),
            0x0101:(8,1,'direct'), 0x0A01:(24,10,'wireless'),
            0x0A02:(12,10,'wireless'), 0x0A03:(10,10,'wireless'),
            0x0A04:(12,10,'wireless'), 0x0A05:(47,10,'wireless'),
            0x0201:(17,10,'relay'), 0x0202:(14,10,'relay'),
            0x0203:(12,1,'relay'), 0x0204:(7,3,'relay'),
            0x0208:(8,10,'relay'), 0x020B:(40,1,'relay'),
            0x020D:(14,1,'relay')}

def _fields():
    f = [Field('remaining_s','0001',420)]
    for side in ('ally','enemy'):
        for slot in GROUND_SLOTS:
            p=f'{side}.{slot}'
            ch='0a01' if side=='enemy' else ('0203' if slot==7 else '020b')
            f.extend([Field(p+'.x',ch,28),Field(p+'.y',ch,15)])
            f.append(Field(p+'.hp','0003' if side=='ally' else '0a02',1000))
            if side=='enemy' or slot==7:
                f.append(Field(p+'.allowance','0a03' if side=='enemy' else '0208',700))
                f.append(Field(p+'.alive','0a05' if side=='enemy' else '0201'))
                if side=='enemy':
                    f.extend(Field(p+'.'+k,'0a05') for k in ('invulnerable','weak'))
                for k,scale in [('defense_buff',1),('vulnerability',1),('attack_multiplier',3),('cooling_bonus',100)]:
                    f.append(Field(p+'.'+k,'0a05' if side=='enemy' else '0204',scale))
                if slot==7:
                    f.extend(Field(p+'.posture_'+k,'0a05' if side=='enemy' else '020d') for k in ('attack','defense','move'))
        f.extend([Field(side+'.base_hp','0003',10000),Field(side+'.outpost_hp','0003',1000),
                  Field(side+'.control_degree','0003',1000)])
        # Separate evidence from each source: "not controlled by own" does not
        # imply enemy full control. Never merge contradictory/stale reports.
        for site in ('base','outpost'):
            for view,ch in (('ally','0101'),('enemy','0a04')):
                f.extend(Field(f'{side}.{site}.{view}_control_{k}',ch) for k in ('full','temporary','none'))
    f.append(Field('damage_difference','0003',10000))
    for k,ch,scale in [('max_hp','0201',1000),('max_heat','0201',500),('cooling_rate','0201',100),
                        ('heat','0202',500),('buffer_energy_j','0202',100),('fortress_reserve','0208',700)]:
        f.append(Field('ally.7.'+k,ch,scale))
    return tuple(f)

FIELDS=_fields()
FIELD_INDEX={f.name:i for i,f in enumerate(FIELDS)}
FEATURE_NAMES=tuple(f'{f.name}.{p}' for f in FIELDS for p in ('value','valid','age_s_div5'))
FEATURE_DIM=len(FEATURE_NAMES)

def contract():
    return dict(schema=SCHEMA, feature_names=FEATURE_NAMES,fields=[asdict(f) for f in FIELDS],
        messages={f'{k:04x}':{'length':v[0],'source_rate_hz':v[1],'route':v[2]} for k,v in MESSAGES.items()},
        protocol_reference='RM2027 V1.0.0 20261029, printed pp.7-24,39-44',
        ground_slots=GROUND_SLOTS, default_channel_config=asdict(RadarConfig()),
        assumptions=[
            'Source-time age survives relay; relay adds a second independent latency/loss hop.',
            'Own 0201/2/3/4/8/B/D must be forwarded by sentry, not read from radar-local robot messages.',
            'Own allowance excludes fortress reserve (separate field); enemy allowance includes reserve for 3/4/7.',
            'Arena team fortress_ammo represents shared reserve, added to each enemy 3/4/7 allowance.',
            'Only base posture is used; enhanced posture collapses to its corresponding base posture.',
            'Zone activation omitted: Arena occupants are not proven equivalent to protocol activation.',
            'Own invulnerable/weak, heading, recovery buff, gold and unverified terrain flags omitted.',
            'No 2/6 ground slots, progress, velocity, target locks, future state or raw capture progress.',
            'Arena uses identity field transform; physical field-to-Arena calibration required before live use.',
            'Little-endian payload bridge convention requires validation against real decoded captures.',
            'Not a deployed ROS node or measured wireless/0301 transport; relay packing/budget validation pending.'])

def value(features,name):
    if name=='ally.7.ammo_proxy': name='ally.7.allowance'  # executor compatibility only
    if name not in FIELD_INDEX: return None
    i=FIELD_INDEX[name];return float(features[3*i]*FIELDS[i].scale) if features[3*i+1]>.5 else None

class ProtocolInput:
    def __init__(self,team,field_to_arena,config=None):
        if team not in ('red','blue'): raise ValueError('unknown team')
        if not callable(field_to_arena): raise ValueError('coordinate calibration required')
        self.team=team;self.transform=field_to_arena;self.config=config or RadarConfig();self.received={}
        self.last_time=-math.inf

    def decode(self,cmd,payload):
        if cmd not in MESSAGES or len(payload)!=MESSAGES[cmd][0]: raise ValueError('unsupported command or payload length')
        d={}
        def get(fmt,offset):return struct.unpack_from('<'+fmt,payload,offset)[0]
        def xy(prefix,x,y):
            x,y=self.transform(x,y)
            if math.isfinite(x) and math.isfinite(y) and 0<=x<=28 and 0<=y<=15:
                d[prefix+'.x'],d[prefix+'.y']=canonical_xy(x,y,self.team)
        def posture(prefix,n):
            if 1<=n<=6:
                for i,k in enumerate(('attack','defense','move')):d[prefix+'.posture_'+k]=float((n-1)%3==i)
        def buffs(prefix,o):
            for k,v in [('cooling_bonus',get('H',o+1)),('defense_buff',get('B',o+3)/100),
                        ('vulnerability',get('B',o+4)/100),('attack_multiplier',1+get('H',o+5)/100)]:d[prefix+'.'+k]=v
        def site(physical,kind):return ('ally' if physical==self.team else 'enemy')+'.'+kind
        if cmd==0x0001:
            if get('B',0)>>4==4:d['remaining_s']=get('H',1)
        elif cmd==0x0003:
            for slot,o in ((1,0),(3,4),(4,6),(7,10)):d[f'ally.{slot}.hp']=get('H',o)
            for physical,o in (('red',12),('blue',16)):d[site(physical,'outpost')+'_hp']=get('H',o)
            for k,o in [('ally.base_hp',14),('enemy.base_hp',18),('ally.control_degree',20),('enemy.control_degree',22)]:d[k]=get('H',o)
            d['damage_difference']=get('h',8)
        elif cmd in (0x0101,0x0A04):
            view='ally' if cmd==0x0101 else 'enemy'
            if cmd==0x0101:
                a,b=get('I',0),get('I',4)
                items=[('red','outpost',(a>>28)&3),('blue','outpost',(a>>30)&3),('red','base',b&3),('blue','base',(b>>2)&3)]
            else:
                b=get('I',8)
                items=[(p,k,(b>>s)&3) for p,k,s in [('red','outpost',22),('blue','outpost',24),('red','base',26),('blue','base',28)]]
            for p,k,n in items:
                if n<3:
                    for i,label in enumerate(('full','temporary','none')):d[f'{site(p,k)}.{view}_control_{label}']=float(n==i)
        elif cmd==0x0A01:
            for slot,o in ((1,0),(3,8),(4,12),(7,20)):xy(f'enemy.{slot}',get('H',o)/100,get('H',o+2)/100)
        elif cmd in (0x0A02,0x0A03):
            offsets=((1,0),(3,4),(4,6),(7,10)) if cmd==0x0A02 else ((1,0),(3,2),(4,4),(7,8))
            for slot,o in offsets:d[f'enemy.{slot}.'+('hp' if cmd==0x0A02 else 'allowance')]=get('H',o)
        elif cmd==0x0A05:
            for slot,o,s in ((1,0,36),(3,14,38),(4,21,39),(7,28,40)):
                p=f'enemy.{slot}';buffs(p,o);state=get('B',s)
                if state<=3:d.update({p+'.alive':float(state!=1),p+'.invulnerable':float(state>=2),p+'.weak':float(state==3)})
            posture('enemy.7',get('B',35))
        elif cmd==0x0201:
            expected=7 if self.team=='red' else 107
            if get('B',0)!=expected:raise ValueError('relay 0201 is not own sentry')
            d['ally.7.alive']=float(get('H',2)>0)
            for k,o in [('max_hp',4),('cooling_rate',6),('max_heat',8)]:d['ally.7.'+k]=get('H',o)
        elif cmd==0x0202:d.update({'ally.7.heat':get('H',10),'ally.7.buffer_energy_j':get('H',8)})
        elif cmd==0x0203:xy('ally.7',get('f',0),get('f',4))
        elif cmd==0x0204:buffs('ally.7',0)
        elif cmd==0x0208:d.update({'ally.7.allowance':get('H',0),'ally.7.fortress_reserve':get('H',6)})
        elif cmd==0x020B:
            for slot,o in ((1,0),(3,16),(4,24)):xy(f'ally.{slot}',get('f',o),get('f',o+4))
        elif cmd==0x020D:posture('ally.7',(get('H',4)>>12)&3)
        return d

    def ingest(self,cmd,payload,source_time,received_time,route):
        if cmd not in MESSAGES or route!=MESSAGES[cmd][2]:raise ValueError('command is unavailable on this route')
        if not all(math.isfinite(t) for t in (source_time,received_time)) or source_time>received_time:raise ValueError('invalid timestamp')
        decoded=self.decode(cmd,payload)
        old=self.received.get(cmd)
        if old is None or source_time>old[0]:self.received[cmd]=(source_time,received_time,decoded)

    def vector(self,now):
        if not math.isfinite(now) or now<self.last_time:raise ValueError('time moved backwards')
        self.last_time=now;out=[]
        for f in FIELDS:
            cmd=int(f.channel,16);stamp,arrival,packet=self.received.get(cmd,(-math.inf,math.inf,{}))
            age=now-stamp;ttl=self.config.fast_ttl_s if MESSAGES[cmd][1]>=3 else self.config.slow_ttl_s
            valid=f.name in packet and arrival<=now and 0<=age<=ttl
            out.extend((packet[f.name]/f.scale if valid else 0.,float(valid),max(0,min(5,age))/5))
        return np.asarray(out,dtype=np.float32)
