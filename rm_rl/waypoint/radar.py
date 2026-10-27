"""Provisional 2027 ground-decision observations, NOT a 2027 wire protocol.

2026 message semantics are references only. Arena slot 6 is a hangar/air
system, not the former gun aircraft; it deliberately has no ground feature
slot. Unknown fields remain masked. Snapshot access is confined to this
simulated sensor boundary; consumers receive only the fixed feature vector.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import heapq
import math
import random

import numpy as np

SCHEMA = 'radar27-ground-v1'
GROUND_SLOTS = (1, 3, 4, 7)
ROLES = {1: 'heavy', 3: 'infantry', 4: 'infantry', 7: 'sentry'}
ANCHORS = np.asarray([(10.8, 7.5), (11.13, 10.64), (11.71, 12.08), (1.4, 13.0)], dtype=np.float32)
# Rates are provisional. Relay rates are design choices, not official limits.
RATES = {'clock': 1, 'broadcast_hp': 3, 'own_zones': 1,
         'enemy_xy': 10, 'enemy_hp': 10, 'enemy_ammo': 10,
         'enemy_status': 10, 'enemy_macro': 10, 'sentry_relay': 10,
         'ally_relay': 1}

@dataclass(frozen=True)
class Field:
    name: str
    channel: str
    scale: float = 1.0


def _fields():
    fields = [Field('remaining_s', 'clock', 420)]
    for side in ('ally', 'enemy'):
        for slot in GROUND_SLOTS:
            p = f'{side}.{slot}'
            xy_channel = 'enemy_xy' if side == 'enemy' else ('sentry_relay' if slot == 7 else 'ally_relay')
            fields.extend([Field(p + '.x', xy_channel, 28), Field(p + '.y', xy_channel, 15)])
            fields.append(Field(p + '.hp', 'enemy_hp' if side == 'enemy' else 'broadcast_hp', 1000))
            if side == 'enemy' or slot == 7:
                ch = 'enemy_ammo' if side == 'enemy' else 'sentry_relay'
                fields.append(Field(p + '.ammo_proxy', ch, 700))
                ch = 'enemy_status' if side == 'enemy' else 'sentry_relay'
                for key, scale in [('alive', 1), ('invulnerable', 1), ('weak', 1),
                                   ('defense_buff', 1), ('vulnerability', 1),
                                   ('attack_multiplier', 3), ('cooling_bonus', 100)]:
                    fields.append(Field(p + '.' + key, ch, scale))
                if slot == 7:
                    for posture in ('attack', 'defense', 'move'):
                        fields.append(Field(p + '.posture_' + posture, ch))
        fields.append(Field(side + '.base_hp', 'broadcast_hp', 10000))
        fields.append(Field(side + '.outpost_hp', 'broadcast_hp', 1000))
        fields.append(Field(side + '.resources_proxy', 'enemy_macro' if side == 'enemy' else 'sentry_relay', 4000))
        # Occupancy only; no 2027 capture progress or total control-score leakage.
        for zone in ('central', 'outpost', 'base', 'fortress', 'supply'):
            for who in ('ally', 'enemy'):
                fields.append(Field(f'{side}.{zone}.occupied_{who}', 'own_zones' if side == 'ally' else 'enemy_macro'))
    fields.append(Field('damage_difference', 'broadcast_hp', 10000))
    for key, scale in [('max_hp', 1000), ('heat', 500), ('max_heat', 500),
                       ('cooling_rate', 100), ('buffer_energy_j', 100), ('heading_sin', 1), ('heading_cos', 1)]:
        fields.append(Field('ally.7.' + key, 'sentry_relay', scale))
    return tuple(fields)

FIELDS = _fields()
FIELD_INDEX = {f.name: i for i, f in enumerate(FIELDS)}
FEATURE_NAMES = tuple(f'{f.name}.{part}' for f in FIELDS for part in ('value', 'valid', 'age_s_div5'))
FEATURE_DIM = len(FEATURE_NAMES)

@dataclass(frozen=True)
class RadarConfig:
    latency_s: float = 0.13
    loss: float = 0.01
    position_noise_m: float = 0.0
    sentry_relay: bool = True
    ally_relay: bool = True
    enemy_feed: bool = True
    fast_ttl_s: float = 0.6
    slow_ttl_s: float = 2.0

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.latency_s, self.loss, self.position_noise_m, self.fast_ttl_s, self.slow_ttl_s)):
            raise ValueError('non-finite channel configuration')
        if self.latency_s < 0 or self.position_noise_m < 0 or not 0 <= self.loss <= 1:
            raise ValueError('invalid latency/noise/loss')
        if min(self.fast_ttl_s, self.slow_ttl_s) <= 0:
            raise ValueError('TTL must be positive')


def contract():
    return {'schema': SCHEMA, 'feature_names': FEATURE_NAMES, 'fields': [asdict(f) for f in FIELDS],
            'rates_hz': RATES, 'ground_slots': GROUND_SLOTS,
            'rules_reference': 'RMUC 2027 V1.0.0, printed pp.20,27-28',
            'protocol_reference': 'RM2026 V2.0.0; semantics only, no invented 2027 command IDs',
            'assumptions': [
                '1 is heavy, not 2026 hero; 2 absent; 6 air system excluded from ground observations, retained in simulation',
                'All ground enemy fields assumed interceptable when enemy_feed enabled; not camera visibility',
                'Ally XY and own sentry telemetry arrive only through a simulated sentry relay',
                'Arena ammo is an allowance proxy; fortress reserve not merged; resources is a gold proxy',
                '2027 zone occupant teams approximate discrete 2026 event flags; no occupancy progress exposed',
                'Arena posture supplies only base posture; strengthened posture and expiry unsupported',
                'Unsupported recovery buffs, radar marks, air-system telemetry are omitted',
                'Global canonical coordinates: blue rotates 180 degrees on the 28x15 Arena field',
                'No raw wire parser, real radio, camera or navigation fidelity validation'],
            'default_channel_config': asdict(RadarConfig())}


def canonical_xy(x, y, team):
    return (x, y) if team == 'red' else (28.0 - x, 15.0 - y)


def value(features, name):
    i = 3 * FIELD_INDEX[name]
    return float(features[i] * FIELDS[FIELD_INDEX[name]].scale) if features[i + 1] > 0.5 else None


class RadarAdapter:
    def __init__(self, config=None, team='red', seed=0):
        if team not in ('red', 'blue'):
            raise ValueError('unknown team')
        self.config = config or RadarConfig()
        self.team = team
        self.rng = random.Random(seed)
        self.next_sample = {k: 0.0 for k in RATES}
        self.pending = []
        self.received = {}
        self.seq = 0
        self.last_time = -1.0
        self.sent = self.dropped = 0

    def _source(self, frame):
        if str(frame.get('season')) != '2027' or frame.get('arena', {}).get('name') != 'rmuc2027':
            raise ValueError('radar27 adapter requires Arena rmuc2027')
        enemy = 'blue' if self.team == 'red' else 'red'
        data = {}
        robots = {(r['team'], int(r['slot'])): r for r in frame.get('robots', []) if int(r.get('slot', -1)) in GROUND_SLOTS}
        for (team, slot), r in robots.items():
            if r.get('role') != ROLES[slot]:
                raise ValueError(f'2027 role mismatch: {team}/{slot}/{r.get("role")}')
            side = 'ally' if team == self.team else 'enemy'
            p = f'{side}.{slot}'
            if 'x' in r and 'y' in r:
                x, y = canonical_xy(float(r['x']), float(r['y']), self.team)
                data[p + '.x'], data[p + '.y'] = x, y
            for key in ('hp', 'alive', 'invulnerable', 'weak', 'defense_buff', 'vulnerability',
                        'attack_multiplier', 'cooling_bonus', 'max_hp', 'heat', 'max_heat', 'cooling_rate', 'buffer_energy_j'):
                if key in r:
                    data[p + '.' + key] = r[key]
            if 'ammo' in r:
                data[p + '.ammo_proxy'] = r['ammo']
            if r.get('posture') in ('attack', 'defense', 'move'):
                for posture in ('attack', 'defense', 'move'):
                    data[p + '.posture_' + posture] = float(r['posture'] == posture)
            if 'heading' in r:
                heading = float(r['heading']) + (math.pi if self.team == 'blue' else 0)
                data[p + '.heading_sin'], data[p + '.heading_cos'] = math.sin(heading), math.cos(heading)
        if 'remaining' in frame:
            data['remaining_s'] = frame['remaining']
        teams = frame.get('teams', {})
        for side, team in (('ally', self.team), ('enemy', enemy)):
            for source, name in (('bases', 'base_hp'), ('outposts', 'outpost_hp')):
                item = frame.get(source, {}).get(team, {})
                if 'hp' in item:
                    data[side + '.' + name] = item['hp']
            if 'resources' in teams.get(team, {}):
                data[side + '.resources_proxy'] = teams[team]['resources']
            for zone in ('central', 'outpost', 'base', 'fortress', 'supply'):
                z = frame.get('zones', {}).get(team + '_' + zone, {})
                if 'occupants' in z:
                    for who, actual in (('ally', self.team), ('enemy', enemy)):
                        ground_ids = {f'{actual}-{slot}' for slot in GROUND_SLOTS}
                        data[f'{side}.{zone}.occupied_{who}'] = float(any(str(rid) in ground_ids for rid in z['occupants']))
        if all('attack_damage' in teams.get(t, {}) for t in (self.team, enemy)):
            data['damage_difference'] = teams[self.team]['attack_damage'] - teams[enemy]['attack_damage']
        return data

    def observe(self, frame):
        now = float(frame['time'])
        if not math.isfinite(now) or now < self.last_time:
            raise ValueError('time moved backwards; instantiate a new adapter per match')
        self.last_time = now
        source = self._source(frame)
        cfg = self.config
        for channel, hz in RATES.items():
            enabled = not ((channel == 'sentry_relay' and not cfg.sentry_relay)
                           or (channel == 'ally_relay' and not (cfg.ally_relay and cfg.sentry_relay))
                           or (channel.startswith('enemy_') and not cfg.enemy_feed))
            if not enabled or now + 1e-8 < self.next_sample[channel]:
                continue
            # Sample only the current frame; never label present state as a past sample.
            self.next_sample[channel] = (math.floor(now * hz + 1e-8) + 1) / hz
            self.sent += 1
            if self.rng.random() < cfg.loss:
                self.dropped += 1
                continue
            packet = {}
            for f in FIELDS:
                if f.channel != channel or f.name not in source:
                    continue
                v = float(source[f.name])
                if not math.isfinite(v):
                    continue
                if f.name.endswith(('.x', '.y')):
                    v += self.rng.gauss(0, cfg.position_noise_m)
                    v = round(v * 100) / 100
                    limit = 28 if f.name.endswith('.x') else 15
                    if not 0 <= v <= limit:
                        continue
                packet[f.name] = v
            heapq.heappush(self.pending, (now + cfg.latency_s, self.seq, channel, now, packet))
            self.seq += 1
        while self.pending and self.pending[0][0] <= now + 1e-8:
            _, _, ch, stamp, packet = heapq.heappop(self.pending)
            self.received[ch] = (stamp, packet)
        out = []
        for f in FIELDS:
            stamp, packet = self.received.get(f.channel, (-math.inf, {}))
            age = now - stamp
            ttl = cfg.fast_ttl_s if RATES[f.channel] >= 3 else cfg.slow_ttl_s
            valid = f.name in packet and 0 <= age <= ttl
            out.extend((packet[f.name] / f.scale if valid else 0.0, float(valid), min(5.0, age) / 5.0))
        return np.asarray(out, dtype=np.float32)
