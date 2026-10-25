"""Single source of truth for database and live-radio feature semantics."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np

VERSION = "rmuc-navigation-v1"
ROLES = (1, 2, 3, 4, 6, 7, 10, 11)
IDS = ROLES + tuple(r + 100 for r in ROLES)
HOLD = 420
ACTION_COUNT = 421
ENTITY_FEATURES = (
    "x/28", "y/15", "position_known", "position_age/3",
    "hp/maxhp", "hp_ratio_known", "maxhp/5000", "maxhp_known", "hp_age/3",
    "alive", "hp_known", "vx/8", "vy/8", "velocity_known",
    "hp_loss_3s/5000", "loss_known", "death_3s", "revival_3s", "history_known",
    "distance_to_ego/32", "distance_known", "near_enemy/8", "near_ally/8",
    "proximity_known", "ally", "ego", "hero", "engineer", "infantry",
    "aerial", "sentry", "base", "outpost",
)
GLOBAL_FEATURES = (
    "time/420", "own_coin_left/10000", "own_coin_total/10000",
    "own_coin_left_known", "own_coin_total_known", "enemy_coin_left/10000",
    "enemy_coin_total/10000", "enemy_coin_left_known", "enemy_coin_total_known",
)
SPEC = dict(
    version=VERSION, entities=16, entity_features=ENTITY_FEATURES,
    global_features=GLOBAL_FEATURES, position_ttl_s=3., hp_ttl_s=3.,
    economy_ttl_s=3., history_s=3., proximity_radius_m=5., velocity_limit_mps=8.,
    frame="28x15 metres; blue rotated (28-x,15-y); ego first; others own then enemy ID",
    action="a=28*floor(y)+floor(x), cell centres; 420=hold at decision position",
    ego_role="all role bits zero; peers 3/4 share infantry bit",
    missing="zero value AND known=0; no backward fill; no future interpolation",
    source_policy="radio then vision for position; per-field timestamps; unknown ID excluded",
    buff_history=False,
)


def fingerprint():
    return hashlib.sha256(json.dumps(SPEC, sort_keys=True).encode()).hexdigest()


def finite(x):
    return isinstance(x, (int, float, np.number)) and not isinstance(x, bool) and math.isfinite(x)


def side(rid):
    return "blue" if rid >= 100 else "red"


def canonical(xy, camp):
    p = np.asarray(xy, dtype=np.float32)
    return np.array([28., 15.], np.float32) - p if camp == "blue" else p.copy()


def action_points(ego_xy):
    """Canonical coordinates; HOLD is dynamic, all other points are fixed."""
    pts = [[x + .5, y + .5] for y in range(15) for x in range(28)]
    return np.asarray(pts + [list(ego_xy)], np.float32)


def cell(xy):
    x, y = xy
    if not (0 <= x <= 28 and 0 <= y <= 15):
        raise ValueError("goal outside field")
    return min(14, int(y)) * 28 + min(27, int(x))


@dataclass
class Observation:
    entities: np.ndarray
    global_: np.ndarray
    ids: list[int]
    ego_xy: np.ndarray | None
    ego_alive: bool


class FeatureEncoder:
    """Stateful causal encoder. One instance per match/ego, called at 1 Hz.

    Input is a full snapshot of *last known* values, never a delta packet.
    Missing keys mean unavailable; timestamps must describe observation time,
    not the time a cached value was republished. No raw radio protocol assumptions.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.key = None
        self.last_t = None
        self.history = []

    def encode(self, packet):
        if packet.get("schema_version") != VERSION:
            raise ValueError("schema_version mismatch")
        t, ego = packet["time_s"], packet["ego_id"]
        camp = packet["own_camp"]
        if not finite(t) or t < 0 or camp not in ("red", "blue"):
            raise ValueError("invalid time/camp")
        if ego not in IDS or side(ego) != camp or ego % 100 not in (3, 4, 7):
            raise ValueError("ego must be own infantry or sentry with referee ID")
        key = (str(packet["match_id"]), ego, camp)
        if key != self.key or (self.last_t is not None and t < self.last_t):
            self.reset()
        if self.last_t is not None and t == self.last_t:
            raise ValueError("duplicate snapshot timestamp; run policy once per tick")
        self.key, self.last_t = key, float(t)
        ids = [ego] + sorted((i for i in IDS if i != ego), key=lambda i: (side(i) != camp, i % 100))
        robots = {}
        for r in packet.get("robots", []):
            rid = r.get("robot_id")
            if rid not in IDS or rid in robots:
                raise ValueError("unknown/duplicate referee robot_id; do not pass track_id")
            robots[rid] = r
        state = {}
        for rid in IDS:
            r = robots.get(rid, {})
            s = {}
            for field in ("position", "hp", "maxhp"):
                at = r.get(field + "_time_s")
                value = r.get(field)
                if finite(at) and at > t:
                    raise ValueError("future field timestamp")
                valid = finite(at) and 0 <= t - at <= 3
                if field == "position":
                    valid = valid and isinstance(value, (list, tuple)) and len(value) == 2
                    valid = valid and all(finite(v) for v in value)
                    valid = valid and 0 <= value[0] <= 28 and 0 <= value[1] <= 15
                    if valid:
                        value = canonical(value, camp)
                else:
                    valid = valid and finite(value) and value >= (1 if field == "maxhp" else 0)
                if valid:
                    s[field], s[field + "_at"] = value, float(at)
            state[rid] = s
        # Earlier real observations only; no fabricated velocity across a gap.
        self.history = [(at, s) for at, s in self.history if t - at <= 3.01]
        old = self.history[0][1] if self.history and 2 <= t - self.history[0][0] <= 3.01 else {}
        a = np.zeros((16, len(ENTITY_FEATURES)), np.float32)
        e = state[ego]
        ep = e.get("position")
        for i, rid in enumerate(ids):
            s, prev = state[rid], old.get(rid, {})
            p, hp, mh = s.get("position"), s.get("hp"), s.get("maxhp")
            row = a[i]
            if p is not None:
                row[:4] = [p[0] / 28, p[1] / 15, 1, (t - s["position_at"]) / 3]
                if ep is not None:
                    row[19:21] = [np.linalg.norm(p - ep) / 32, 1]
                opponents = allies = known = 0
                for other, os in state.items():
                    if other == rid or other % 100 in (10, 11):
                        continue
                    if "position" in os and "hp" in os:
                        known += 1
                        if os["hp"] > 0 and np.linalg.norm(os["position"] - p) <= 5:
                            if side(other) == side(rid):
                                allies += 1
                            else:
                                opponents += 1
                row[21:24] = [opponents / 8, allies / 8, known / 12]
            if hp is not None:
                row[8:11] = [(t - s["hp_at"]) / 3, float(hp > 0), 1]
                if mh is not None:
                    row[4:6] = [min(hp / mh, 1.5), 1]
            if mh is not None:
                row[6:8] = [mh / 5000, 1]
            if p is not None and "position" in prev:
                dt = s["position_at"] - prev["position_at"]
                if 0 < dt <= 3.01:
                    v = (p - prev["position"]) / dt
                    if np.linalg.norm(v) <= 8:
                        row[11:14] = [v[0] / 8, v[1] / 8, 1]
            if hp is not None and "hp" in prev and s["hp_at"] > prev["hp_at"]:
                values = [ps[rid]["hp"] for _, ps in self.history if "hp" in ps[rid]] + [hp]
                row[14:19] = [sum(max(0, x-y) for x, y in zip(values, values[1:])) / 5000,
                              1, any(x > 0 and y == 0 for x,y in zip(values, values[1:])),
                              any(x == 0 and y > 0 for x,y in zip(values, values[1:])), 1]
            row[24:26] = [side(rid) == camp, rid == ego]
            if rid != ego:
                role_index = {1: 26, 2: 27, 3: 28, 4: 28, 6: 29, 7: 30, 10: 31, 11: 32}
                row[role_index[rid % 100]] = 1
        g = np.zeros(len(GLOBAL_FEATURES), np.float32)
        g[0] = t / 420
        for start, cs in ((1, camp), (5, "blue" if camp == "red" else "red")):
            econ = packet.get("economy", {}).get(cs, {})
            for j, name in enumerate(("remaining", "total")):
                v, at = econ.get(name), econ.get(name + "_time_s")
                if finite(at) and at > t:
                    raise ValueError("future economy timestamp")
                if finite(v) and v >= 0 and finite(at) and 0 <= t-at <= 3:
                    g[start+j], g[start+2+j] = v / 10000, 1
        self.history.append((float(t), state))
        return Observation(a, g, ids, ep, e.get("hp", 0) > 0)
