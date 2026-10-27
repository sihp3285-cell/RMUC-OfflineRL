# ---- distilled waypoint runtime (pure stdlib; injected after the hold strategy) ----
import base64 as _WP_b64
import json as _WP_json
import math as _WP_math
import struct as _WP_struct
import zlib as _WP_zlib

_WP_DATA = _WP_json.loads(_WP_zlib.decompress(_WP_b64.b85decode("@@PAYLOAD@@")))
_WP_MEAN = _WP_DATA["mean"]
_WP_STD = _WP_DATA["std"]
_WP_ANCHORS = _WP_DATA["anchors"]
_WP_TENSORS = {}
for _WP_name, (_WP_shape, _WP_blob) in _WP_DATA["tensors"].items():
    _WP_raw = _WP_b64.b85decode(_WP_blob)
    _WP_vals = _WP_struct.unpack("<" + "e" * (len(_WP_raw) // 2), _WP_raw)
    if len(_WP_shape) == 2:
        _WP_rows, _WP_cols = int(_WP_shape[0]), int(_WP_shape[1])
        _WP_TENSORS[_WP_name] = [_WP_vals[i:i + _WP_cols] for i in range(0, len(_WP_vals), _WP_cols)]
    else:
        _WP_TENSORS[_WP_name] = list(_WP_vals)
del _WP_raw, _WP_vals, _WP_blob, _WP_shape, _WP_name

_WP_SAFE = True
_WP_CENTRAL = (10.8, 7.5)
_WP_SUPPLY = (1.4, 13.0)
_WP_RED = (1, 3, 4, 6, 7)
_WP_BLUE = (1, 3, 4, 6, 7)
_WP_ROLES = ("heavy", "infantry", "aerial", "sentry")


def _wp_num(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if _WP_math.isfinite(value) else default


def _wp_ratio(value, denom, default=0.0):
    value = _wp_num(value, default)
    return _wp_num(value / denom, default) if denom else default


def _wp_clip(value, lo, hi):
    return max(lo, min(hi, _wp_num(value)))


def _wp_robot(robot, now, is_red, slot):
    role = str(robot.get("role", ""))
    hp = _wp_num(robot.get("hp"))
    max_hp = _wp_num(robot.get("max_hp"), 1.0)
    heat = _wp_num(robot.get("heat"))
    max_heat = _wp_num(robot.get("max_heat"))
    hits = robot.get("armor_hit_times") or []
    last_hit = max([_wp_num(v, -1000.0) for v in hits] + [_wp_num(robot.get("last_damage_time"), -1000.0)])
    since = _wp_clip(now - last_hit, 0.0, 10.0) / 10.0
    return [
        float(is_red), slot / 7.0,
        *[1.0 if role == r else 0.0 for r in _WP_ROLES],
        _wp_clip(_wp_num(robot.get("x")) / 28.0, -0.5, 1.5),
        _wp_clip(_wp_num(robot.get("y")) / 15.0, -0.5, 1.5),
        _wp_clip(_wp_num(robot.get("z")) / 2.0, -0.5, 2.0),
        _wp_clip(_wp_num(robot.get("vx")) / 5.0, -2.0, 2.0),
        _wp_clip(_wp_num(robot.get("vy")) / 5.0, -2.0, 2.0),
        _wp_clip(_wp_ratio(hp, max_hp), 0.0, 1.5),
        1.0 if robot.get("alive") else 0.0,
        _wp_clip(_wp_num(robot.get("ammo")) / 700.0, 0.0, 2.0),
        _wp_clip(_wp_ratio(heat, max_heat), 0.0, 2.0),
        _wp_clip(_wp_num(robot.get("level")) / 10.0, 0.0, 2.0),
        _wp_clip(_wp_num(robot.get("defense_buff")), 0.0, 2.0),
        _wp_clip(_wp_num(robot.get("vulnerability")), 0.0, 2.0),
        _wp_clip(_wp_num(robot.get("attack_multiplier"), 1.0), 0.0, 3.0),
        1.0 if robot.get("grounded") else 0.0,
        1.0 if robot.get("tipped") else 0.0,
        since,
        _wp_clip(_wp_num(robot.get("respawn_progress")), 0.0, 1.0),
        1.0 if robot.get("support_active") else 0.0,
    ]


def _wp_team(team):
    return [
        _wp_clip(_wp_num(team.get("resources")) / 4000.0, 0.0, 3.0),
        _wp_clip(_wp_num(team.get("attack_damage")) / 10000.0, 0.0, 5.0),
        _wp_clip(_wp_num(team.get("control")) / 400.0, 0.0, 3.0),
        _wp_clip(_wp_num(team.get("kills")) / 10.0, 0.0, 3.0),
        _wp_clip(_wp_num(team.get("base_lowest_hp"), 10000.0) / 10000.0, 0.0, 2.0),
        _wp_clip(_wp_num(team.get("tech_income")) / 1000.0, 0.0, 2.0),
        _wp_clip(_wp_num(team.get("level_cap")) / 10.0, 0.0, 2.0),
        _wp_clip(_wp_num(team.get("purchased_17")) / 1000.0, 0.0, 3.0),
        _wp_clip(_wp_num(team.get("purchased_42")) / 1000.0, 0.0, 3.0),
        _wp_clip(_wp_num(team.get("energy_units_left")) / 6.0, 0.0, 2.0),
        _wp_clip(len(team.get("completed_difficulties") or []) / 4.0, 0.0, 2.0),
        _wp_clip(_wp_num(team.get("tech_defense")), 0.0, 2.0),
    ]


def _wp_features(frame):
    robots = frame.get("robots") or []
    by = {}
    for robot in robots:
        try:
            by[(str(robot.get("team", "")), int(robot.get("slot", 0)))] = robot
        except (TypeError, ValueError):
            pass
    now = _wp_num(frame.get("time"))
    out = []
    for team, slots in (("red", _WP_RED), ("blue", _WP_BLUE)):
        for slot in slots:
            out.extend(_wp_robot(by.get((team, slot), {}), now, team == "red", slot))
    red = (frame.get("teams") or {}).get("red") or {}
    blue = (frame.get("teams") or {}).get("blue") or {}
    out.extend(_wp_team(red))
    out.extend(_wp_team(blue))
    bases = frame.get("bases") or {}
    for side in ("red", "blue"):
        base = bases.get(side) or {}
        hp = _wp_num(base.get("hp"))
        out.extend([
            _wp_clip(_wp_ratio(hp, _wp_num(base.get("max_hp"), 10000.0)), 0.0, 2.0),
            1.0 if hp > 0 else 0.0,
            1.0 if base.get("armor_open") else 0.0,
            _wp_clip(_wp_num(base.get("armor_open_progress")), 0.0, 1.0),
        ])
    outposts = frame.get("outposts") or {}
    for side in ("red", "blue"):
        op = outposts.get(side) or {}
        hp = _wp_num(op.get("hp"))
        owner = str(op.get("owner", side))
        out.extend([
            _wp_clip(_wp_ratio(hp, _wp_num(op.get("max_hp"), 1.0)), 0.0, 2.0),
            1.0 if owner == "red" else 0.0,
            1.0 if owner == "blue" else 0.0,
            1.0 if op.get("armor_open") else 0.0,
            _wp_clip(_wp_num(op.get("armor_open_progress")), 0.0, 1.0),
        ])
    positions = frame.get("positions") or {}
    for key in ("red_central", "blue_central", "red_outpost", "blue_outpost"):
        point = positions.get(key) or {}
        ctrl = str(point.get("controller", ""))
        out.extend([
            1.0 if ctrl == "red" else 0.0,
            1.0 if ctrl == "blue" else 0.0,
            _wp_clip(_wp_num(point.get("progress")), 0.0, 1.0),
            1.0 if point.get("control") == "full" else 0.0,
        ])
    red_alive = sum(1 for r in robots if r.get("team") == "red" and r.get("alive"))
    blue_alive = sum(1 for r in robots if r.get("team") == "blue" and r.get("alive"))
    out.extend([
        _wp_clip(now / 420.0, 0.0, 1.2),
        _wp_clip(_wp_num(frame.get("remaining")) / 420.0, 0.0, 1.2),
        _wp_clip(_wp_num(frame.get("tick")) / 8400.0, 0.0, 1.2),
        _wp_clip((_wp_num(red.get("control")) - _wp_num(blue.get("control"))) / 400.0, -3.0, 3.0),
        _wp_clip((_wp_num(red.get("attack_damage")) - _wp_num(blue.get("attack_damage"))) / 10000.0, -3.0, 3.0),
        _wp_clip((_wp_num(red.get("resources")) - _wp_num(blue.get("resources"))) / 4000.0, -3.0, 3.0),
        _wp_clip((_wp_num(red.get("base_lowest_hp"), 10000.0) - _wp_num(blue.get("base_lowest_hp"), 10000.0)) / 10000.0, -2.0, 2.0),
        red_alive / 5.0,
        blue_alive / 5.0,
    ])
    if len(out) != 307:
        raise ValueError("feature size")
    return out


def _wp_rule(base):
    ammo = base[109] * 700.0
    hp = base[107]
    red_full = base[293]
    blue_full = base[297]
    red_central_enemy = base[283]
    ally = 0.0
    for start in (0, 24, 48, 72):
        ax, ay = base[start + 6] * 28.0, base[start + 7] * 15.0
        if base[start + 12] > 0.5 and base[start + 11] >= 0.4 and _WP_math.hypot(ax - 10.98, ay - 11.40) < 2.2:
            ally = 1.0
    return [
        1.0 if ammo <= 0.5 else 0.0,
        1.0 if ammo < 30.0 else 0.0,
        1.0 if ammo >= 100.0 else 0.0,
        1.0 if 30.0 <= ammo < 100.0 else 0.0,
        1.0 if hp < 0.24 else 0.0,
        1.0 if hp < 0.85 else 0.0,
        red_full, blue_full, red_central_enemy, ally,
    ]


def _wp_anchor(base):
    ex, ey = base[102] * 28.0, base[103] * 15.0
    out = []
    for ax, ay in _WP_ANCHORS:
        out.extend([(ax - ex) / 28.0, (ay - ey) / 15.0, _WP_math.hypot(ax - ex, ay - ey) / 28.0])
    return out


def _wp_seq(base, state):
    hist = state.setdefault("_wp_hist", [])
    hist.append([base[109] * 700.0, base[107]])
    if len(hist) > 100:
        del hist[:-100]
    ammo = [h[0] for h in hist]
    hp = [h[1] for h in hist]
    a5 = min(ammo[-50:]) / 700.0
    a10 = min(ammo[-100:]) / 700.0
    h5 = min(hp[-50:])
    h10 = min(hp[-100:])
    last_a = 1.0
    last_h = 1.0
    for i in range(len(ammo) - 1, -1, -1):
        if ammo[i] < 30.0:
            last_a = min(100, len(ammo) - 1 - i) / 100.0
            break
    for i in range(len(hp) - 1, -1, -1):
        if hp[i] < 0.24:
            last_h = min(100, len(hp) - 1 - i) / 100.0
            break
    return [a5, a10, h5, h10, last_a, last_h]


def _wp_linear(x, prefix):
    weights = _WP_TENSORS[prefix + ".weight"]
    bias = _WP_TENSORS[prefix + ".bias"]
    return [sum(a * b for a, b in zip(row, x)) + c for row, c in zip(weights, bias)]


def _wp_norm_relu(x, prefix):
    mu = sum(x) / len(x)
    inv = 1.0 / _WP_math.sqrt(sum((v - mu) ** 2 for v in x) / len(x) + 1e-5)
    return [max(0.0, (v - mu) * inv * w + b)
            for v, w, b in zip(x, _WP_TENSORS[prefix + ".weight"], _WP_TENSORS[prefix + ".bias"])]


def _wp_predict(obs, state):
    if obs.get("team") != "red":
        return None
    robots = list(obs.get("robots") or [])
    for enemy in obs.get("enemies") or []:
        enemy = dict(enemy)
        enemy.setdefault("team", "blue")
        robots.append(enemy)
    frame = {
        "robots": robots,
        "teams": obs.get("teams") or {},
        "bases": obs.get("bases") or {},
        "outposts": obs.get("outposts") or {},
        "positions": obs.get("positions") or {},
        "time": obs.get("time", 0.0),
        "remaining": obs.get("remaining", 0.0),
        "tick": obs.get("tick", 0),
    }
    base = _wp_features(frame)
    feat = base + _wp_seq(base, state) + _wp_rule(base) + _wp_anchor(base)
    if len(feat) != len(_WP_MEAN):
        raise ValueError("normalization size")
    x = [(v - m) / s for v, m, s in zip(feat, _WP_MEAN, _WP_STD)]
    h = _wp_norm_relu(_wp_linear(x, "net.0"), "net.1")
    h = _wp_norm_relu(_wp_linear(h, "net.3"), "net.4")
    logits = _wp_linear(h, "net.6")
    index = max(range(len(logits)), key=lambda i: logits[i])
    goal = _WP_ANCHORS[index]

    # Conservative gate: do not abandon a fully controlled central card for a
    # distant outpost, and do not run to supply with full ammo/HP.  The model
    # still chooses among anchors; this only prevents the catastrophic cases.
    if _WP_SAFE:
        sentry = next((r for r in obs.get("robots") or [] if str(r.get("slot")) == "7"), {})
        ammo = _wp_num(sentry.get("ammo"), 999.0)
        hp = _wp_ratio(sentry.get("hp"), _wp_num(sentry.get("max_hp"), 1.0), 1.0)
        central_full = ((obs.get("positions") or {}).get("red_central") or {}).get("control") == "full"
        supply_d = ((goal[0] - _WP_SUPPLY[0]) ** 2 + (goal[1] - _WP_SUPPLY[1]) ** 2) ** 0.5
        central_d = ((goal[0] - _WP_CENTRAL[0]) ** 2 + (goal[1] - _WP_CENTRAL[1]) ** 2) ** 0.5
        if supply_d < 1.0 and ammo > 30.0 and hp > 0.5:
            goal = _WP_CENTRAL
        elif central_d > 2.0 and central_full:
            goal = _WP_CENTRAL
    return goal


_WP_ORIG_DRIVE = drive
_WP_ORIG_ACT = act
_WP_CURRENT_GOAL = None


def drive(robot, goal, obs, memory, tolerance=.35):
    if robot.get("role") == "sentry" and str(robot.get("slot")) == "7" and _WP_CURRENT_GOAL is not None:
        goal = _WP_CURRENT_GOAL
    return _WP_ORIG_DRIVE(robot, goal, obs, memory, tolerance)


def act(obs, state):
    global _WP_CURRENT_GOAL
    state = dict(state or {})
    try:
        _WP_CURRENT_GOAL = _wp_predict(obs, state)
    except Exception:
        _WP_CURRENT_GOAL = None
    return _WP_ORIG_ACT(obs, state)
