"""Legacy 307-D Arena observation contract (335-D after augmentation).

Arena observation legality does not establish real radar availability. This
contract includes simulator-specific state and the engine's slot-6 air-system
object. Use radar.py for the provisional 2027 ground radar input contract;
existing checkpoints retain this schema unchanged for comparison.
"""
from __future__ import annotations

import math

import numpy as np

RED_SLOTS = (1, 3, 4, 6, 7)
BLUE_SLOTS = (1, 3, 4, 6, 7)
ROLES = ("heavy", "infantry", "aerial", "sentry")

# 10 robots * 24 + 2 teams * 12 + structures 34 + context 9
FEATURE_DIM = 10 * 24 + 2 * 12 + 34 + 9


def _num(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _ratio(value, denom, default=0.0):
    value = _num(value, default)
    return _num(value / denom, default) if denom else default


def _clip(value, lo, hi):
    return max(lo, min(hi, _num(value)))


def _robot_features(robot, time_s, is_red, slot):
    role = str(robot.get("role", ""))
    hp = _num(robot.get("hp"))
    max_hp = _num(robot.get("max_hp"), 1.0)
    heat = _num(robot.get("heat"))
    max_heat = _num(robot.get("max_heat"))
    hits = robot.get("armor_hit_times") or []
    last_hit = max([_num(v, -1000.0) for v in hits] + [_num(robot.get("last_damage_time"), -1000.0)])
    since_hit = _clip(time_s - last_hit, 0.0, 10.0) / 10.0
    return [
        float(is_red),
        slot / 7.0,
        *[1.0 if role == name else 0.0 for name in ROLES],
        _clip(_num(robot.get("x")) / 28.0, -0.5, 1.5),
        _clip(_num(robot.get("y")) / 15.0, -0.5, 1.5),
        _clip(_num(robot.get("z")) / 2.0, -0.5, 2.0),
        _clip(_num(robot.get("vx")) / 5.0, -2.0, 2.0),
        _clip(_num(robot.get("vy")) / 5.0, -2.0, 2.0),
        _clip(_ratio(hp, max_hp), 0.0, 1.5),
        1.0 if robot.get("alive") else 0.0,
        _clip(_num(robot.get("ammo")) / 700.0, 0.0, 2.0),
        _clip(_ratio(heat, max_heat), 0.0, 2.0),
        _clip(_num(robot.get("level")) / 10.0, 0.0, 2.0),
        _clip(_num(robot.get("defense_buff")), 0.0, 2.0),
        _clip(_num(robot.get("vulnerability")), 0.0, 2.0),
        _clip(_num(robot.get("attack_multiplier"), 1.0), 0.0, 3.0),
        1.0 if robot.get("grounded") else 0.0,
        1.0 if robot.get("tipped") else 0.0,
        since_hit,
        _clip(_num(robot.get("respawn_progress")), 0.0, 1.0),
        1.0 if robot.get("support_active") else 0.0,
    ]


def _team_features(team):
    return [
        _clip(_num(team.get("resources")) / 4000.0, 0.0, 3.0),
        _clip(_num(team.get("attack_damage")) / 10000.0, 0.0, 5.0),
        _clip(_num(team.get("control")) / 400.0, 0.0, 3.0),
        _clip(_num(team.get("kills")) / 10.0, 0.0, 3.0),
        _clip(_num(team.get("base_lowest_hp"), 10000.0) / 10000.0, 0.0, 2.0),
        _clip(_num(team.get("tech_income")) / 1000.0, 0.0, 2.0),
        _clip(_num(team.get("level_cap")) / 10.0, 0.0, 2.0),
        _clip(_num(team.get("purchased_17")) / 1000.0, 0.0, 3.0),
        _clip(_num(team.get("purchased_42")) / 1000.0, 0.0, 3.0),
        _clip(_num(team.get("energy_units_left")) / 6.0, 0.0, 2.0),
        _clip(len(team.get("completed_difficulties") or []) / 4.0, 0.0, 2.0),
        _clip(_num(team.get("tech_defense")), 0.0, 2.0),
    ]


def build_features(frame):
    """frame: engine snapshot at the decision tick. Returns float32 [FEATURE_DIM]."""
    robots = frame.get("robots") or []
    by_key = {}
    for robot in robots:
        team = str(robot.get("team", ""))
        try:
            slot = int(robot.get("slot", 0))
        except (TypeError, ValueError):
            continue
        by_key[(team, slot)] = robot

    time_s = _num(frame.get("time"))
    feats = []
    for team, slots in (("red", RED_SLOTS), ("blue", BLUE_SLOTS)):
        for slot in slots:
            feats.extend(_robot_features(by_key.get((team, slot), {}), time_s, team == "red", slot))

    red = frame.get("teams", {}).get("red", {}) or {}
    blue = frame.get("teams", {}).get("blue", {}) or {}
    feats.extend(_team_features(red))
    feats.extend(_team_features(blue))

    bases = frame.get("bases", {}) or {}
    for side in ("red", "blue"):
        base = bases.get(side, {}) or {}
        hp = _num(base.get("hp"))
        max_hp = _num(base.get("max_hp"), 10000.0)
        feats.extend([
            _clip(_ratio(hp, max_hp), 0.0, 2.0),
            1.0 if hp > 0 else 0.0,
            1.0 if base.get("armor_open") else 0.0,
            _clip(_num(base.get("armor_open_progress")), 0.0, 1.0),
        ])

    outposts = frame.get("outposts", {}) or {}
    for side in ("red", "blue"):
        outpost = outposts.get(side, {}) or {}
        hp = _num(outpost.get("hp"))
        max_hp = _num(outpost.get("max_hp"), 1.0)
        owner = str(outpost.get("owner", side))
        feats.extend([
            _clip(_ratio(hp, max_hp), 0.0, 2.0),
            1.0 if owner == "red" else 0.0,
            1.0 if owner == "blue" else 0.0,
            1.0 if outpost.get("armor_open") else 0.0,
            _clip(_num(outpost.get("armor_open_progress")), 0.0, 1.0),
        ])

    positions = frame.get("positions", {}) or {}
    for key in ("red_central", "blue_central", "red_outpost", "blue_outpost"):
        point = positions.get(key, {}) or {}
        controller = str(point.get("controller", ""))
        feats.extend([
            1.0 if controller == "red" else 0.0,
            1.0 if controller == "blue" else 0.0,
            _clip(_num(point.get("progress")), 0.0, 1.0),
            1.0 if point.get("control") == "full" else 0.0,
        ])

    red_alive = sum(1 for robot in robots if robot.get("team") == "red" and robot.get("alive"))
    blue_alive = sum(1 for robot in robots if robot.get("team") == "blue" and robot.get("alive"))
    feats.extend([
        _clip(time_s / 420.0, 0.0, 1.2),
        _clip(_num(frame.get("remaining")) / 420.0, 0.0, 1.2),
        _clip(_num(frame.get("tick")) / 8400.0, 0.0, 1.2),
        _clip((_num(red.get("control")) - _num(blue.get("control"))) / 400.0, -3.0, 3.0),
        _clip((_num(red.get("attack_damage")) - _num(blue.get("attack_damage"))) / 10000.0, -3.0, 3.0),
        _clip((_num(red.get("resources")) - _num(blue.get("resources"))) / 4000.0, -3.0, 3.0),
        _clip((_num(red.get("base_lowest_hp"), 10000.0) - _num(blue.get("base_lowest_hp"), 10000.0)) / 10000.0, -2.0, 2.0),
        red_alive / 5.0,
        blue_alive / 5.0,
    ])

    value = np.asarray(feats, dtype=np.float32)
    if value.size != FEATURE_DIM:
        raise RuntimeError(f"feature size {value.size} != {FEATURE_DIM}")
    return value


def ego_xy(frame):
    for robot in frame.get("robots") or []:
        if robot.get("team") == "red" and int(robot.get("slot", 0)) == 7:
            return float(robot.get("x", 0.0)), float(robot.get("y", 0.0))
    return 0.0, 0.0


# Ego (red-7) position inside the fixed 10-robot block.
EGO_X_INDEX = 4 * 24 + 6
EGO_Y_INDEX = 4 * 24 + 7


def anchor_features(features, anchors):
    """Append ego->anchor relative geometry for candidate scoring."""
    values = np.asarray(features, dtype=np.float32)
    single = values.ndim == 1
    if single:
        values = values[None, :]
    ex = values[:, EGO_X_INDEX] * 28.0
    ey = values[:, EGO_Y_INDEX] * 15.0
    extra = []
    for ax, ay in np.asarray(anchors, dtype=np.float32):
        dx = (float(ax) - ex) / 28.0
        dy = (float(ay) - ey) / 15.0
        dist = np.hypot(float(ax) - ex, float(ay) - ey) / 28.0
        extra.extend([dx, dy, dist])
    stacked = np.stack(extra, axis=1).astype(np.float32)
    out = np.concatenate([values, stacked], axis=1)
    return out[0] if single else out


# Derived rule/context features from the same base vector.  These encode the
# thresholds the 2772 sentry teacher actually uses (ammo/resupply, low HP,
# outpost control, enemy central card) without copying its private memory.
def rule_features(features):
    values = np.asarray(features, dtype=np.float32)
    single = values.ndim == 1
    if single:
        values = values[None, :]
    ammo = values[:, 96 + 13] * 700.0
    hp = values[:, 96 + 11]
    red_outpost_full = values[:, 290 + 3]
    blue_outpost_full = values[:, 294 + 3]
    red_central_enemy = values[:, 282 + 1]
    blue_central_enemy = values[:, 286 + 1]

    ally_outpost = np.zeros(len(values), dtype=np.float32)
    for start in (0, 24, 48, 72):
        ax = values[:, start + 6] * 28.0
        ay = values[:, start + 7] * 15.0
        alive = values[:, start + 12] > 0.5
        healthy = values[:, start + 11] >= 0.4
        near = np.hypot(ax - 10.98, ay - 11.40) < 2.2
        ally_outpost = np.maximum(ally_outpost, (alive & healthy & near).astype(np.float32))

    extra = np.stack([
        (ammo <= 0.5).astype(np.float32),
        (ammo < 30.0).astype(np.float32),
        (ammo >= 100.0).astype(np.float32),
        ((ammo >= 30.0) & (ammo < 100.0)).astype(np.float32),
        (hp < 0.24).astype(np.float32),
        (hp < 0.85).astype(np.float32),
        red_outpost_full,
        blue_outpost_full,
        red_central_enemy,
        ally_outpost,
    ], axis=1).astype(np.float32)
    out = np.concatenate([values, extra], axis=1)
    return out[0] if single else out


def sequence_context(features, window_short=50, window_long=100):
    """Rolling latch proxies: ammo/HP minima and time since low thresholds.

    The teacher's resupply/recovery memories latch on ammo<30 / HP<24% and clear
    later; a single frame cannot tell whether a mid-range ammo/HP value is before
    or after the latch.  These six features make that history observable.
    """
    values = np.asarray(features, dtype=np.float32)
    if values.ndim == 1:
        values = values[None, :]
    n = len(values)
    ammo = values[:, 96 + 13] * 700.0
    hp = values[:, 96 + 11]
    out = np.zeros((n, 6), dtype=np.float32)
    last_ammo_low = -10_000
    last_hp_low = -10_000
    for i in range(n):
        lo_s = max(0, i - window_short + 1)
        lo_l = max(0, i - window_long + 1)
        out[i, 0] = ammo[lo_s:i + 1].min() / 700.0
        out[i, 1] = ammo[lo_l:i + 1].min() / 700.0
        out[i, 2] = hp[lo_s:i + 1].min()
        out[i, 3] = hp[lo_l:i + 1].min()
        if ammo[i] < 30.0:
            last_ammo_low = i
        if hp[i] < 0.24:
            last_hp_low = i
        out[i, 4] = min(100, i - last_ammo_low) / 100.0 if last_ammo_low >= 0 else 1.0
        out[i, 5] = min(100, i - last_hp_low) / 100.0 if last_hp_low >= 0 else 1.0
    return np.concatenate([values, out], axis=1)


def sequence_context_one(current, history):
    """Single-frame version; history is a list/array of base 307-D frames."""
    if history is None or len(history) == 0:
        return sequence_context(current[None, :])[0]
    hist = np.asarray(history[-99:], dtype=np.float32)
    return sequence_context(np.concatenate([hist, current[None, :]], axis=0))[-1]


def obs_to_frame(obs):
    """Build the fixed-contract frame from an Arena legal observation."""
    team = str(obs.get("team", "red"))
    enemy_team = "blue" if team == "red" else "red"
    robots = list(obs.get("robots") or [])
    for enemy in obs.get("enemies") or []:
        enemy = dict(enemy)
        enemy.setdefault("team", enemy_team)
        robots.append(enemy)
    return {
        "robots": robots,
        "teams": obs.get("teams") or {},
        "bases": obs.get("bases") or {},
        "outposts": obs.get("outposts") or {},
        "positions": obs.get("positions") or {},
        "time": obs.get("time", 0.0),
        "remaining": obs.get("remaining", 0.0),
        "tick": obs.get("tick", 0),
    }
