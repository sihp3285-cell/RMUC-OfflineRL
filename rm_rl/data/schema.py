"""Constants describing the RMUC 2026 referee-system SQLite schema.

The database ships with Chinese column names.  We keep the exact original
strings here (so SQL stays correct) and expose ASCII aliases that the rest of
the pipeline uses.  Everything the code needs to know about the *meaning* of the
data is centralised in this file — if a future dataset renames a column, you
only edit here.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Table + column names (exactly as stored in the .sqlite file)
# ---------------------------------------------------------------------------
T_MATCHES = "matches"
T_TIMESERIES = "timeseries"
T_EVENTS = "events"

# timeseries columns -> ascii alias used everywhere downstream
TS_COLS = {
    "game_id": "game_id",
    "时刻秒": "t",
    "robot_id": "robot_id",
    "机器人类型": "rtype",
    "阵营": "camp",
    "学校名": "school",
    "当前血量": "hp",
    "最大血量": "maxhp",
    "x": "x",
    "y": "y",
    "z": "z",
    "枪口朝向": "yaw",          # gun/turret compass heading, degrees
    "底盘功率": "power",        # chassis power (W)
    "小热量": "heat17",         # 17mm barrel heat
    "小热量上限": "heat17_max",
    "大热量": "heat42",
    "大热量上限": "heat42_max",
    "累计17mm发弹": "ammo17",   # CUMULATIVE rounds fired (not remaining)
    "累计42mm发弹": "ammo42",
    "队伍总金币": "coin_total",
    "队伍剩余金币": "coin_left",
    "是否易伤": "vuln",         # 1 = highlighted / marked vulnerable by enemy radar
}

MATCH_COLS = {
    "game_id": "game_id",
    "赛区": "region",
    "胜方": "winner",           # '红' or '蓝'
    "红方学校": "red_school",
    "蓝方学校": "blue_school",
    "时长秒": "duration",
}

# events columns
EV_COLS = {
    "game_id": "game_id",
    "时刻秒": "t",
    "事件类型": "etype",
    "robot_id": "robot_id",
    "机器人类型": "rtype",
    "阵营": "camp",
    "目标robot_id": "target_id",
    "目标类型": "target_type",
    "类别": "category",
    "数值": "value",
}

# ---------------------------------------------------------------------------
# Camps
# ---------------------------------------------------------------------------
CAMP_RED = "红"
CAMP_BLUE = "蓝"
CAMPS = (CAMP_RED, CAMP_BLUE)

def enemy_camp(camp: str) -> str:
    return CAMP_BLUE if camp == CAMP_RED else CAMP_RED

# ---------------------------------------------------------------------------
# Robot types (机器人类型) and the canonical robot_id layout.
# Red ids are the base ids; blue ids are base + 100.
# ---------------------------------------------------------------------------
TYPE_HERO = "英雄"
TYPE_ENGINEER = "工程"
TYPE_INFANTRY3 = "步兵3"
TYPE_INFANTRY4 = "步兵4"
TYPE_AERIAL = "空中"
TYPE_SENTRY = "哨兵"
TYPE_BASE = "基地"
TYPE_OUTPOST = "前哨站"

# base robot_id for the RED side (blue = +100)
RED_ID = {
    TYPE_HERO: 1,
    TYPE_ENGINEER: 2,
    TYPE_INFANTRY3: 3,
    TYPE_INFANTRY4: 4,
    TYPE_AERIAL: 6,
    TYPE_SENTRY: 7,
    TYPE_BASE: 10,
    TYPE_OUTPOST: 11,
}
BLUE_OFFSET = 100

# mobile (drivable) units, in a *fixed* order used to build fixed-length obs
MOBILE_TYPES = [
    TYPE_HERO,
    TYPE_ENGINEER,
    TYPE_INFANTRY3,
    TYPE_INFANTRY4,
    TYPE_AERIAL,
    TYPE_SENTRY,
]
BUILDING_TYPES = [TYPE_BASE, TYPE_OUTPOST]
ALL_TYPES = MOBILE_TYPES + BUILDING_TYPES

# max HP by type (upper tier reached via in-game level upgrades); used only for
# normalisation fall-backs when a row's 最大血量 is missing.
MAX_HP = {
    TYPE_HERO: 450.0,
    TYPE_ENGINEER: 250.0,
    TYPE_INFANTRY3: 400.0,
    TYPE_INFANTRY4: 400.0,
    TYPE_AERIAL: 100.0,
    TYPE_SENTRY: 400.0,
    TYPE_BASE: 5000.0,
    TYPE_OUTPOST: 1500.0,
}

# ASCII aliases for the robot types, so Windows users never have to type
# Chinese on the command line / in a .ps1 (PowerShell 5.1 mangles non-BOM CJK).
AGENT_ALIASES = {
    "sentry": TYPE_SENTRY,
    "hero": TYPE_HERO,
    "engineer": TYPE_ENGINEER,
    "infantry3": TYPE_INFANTRY3,
    "infantry4": TYPE_INFANTRY4,
    "aerial": TYPE_AERIAL,
}


# Groups that expand to several robot types, pooled into one dataset.
# `infantry` is the important one: both infantry slots are human-driven and
# structurally identical to the sentry (both cap at 400 HP / 260 heat / 17 mm),
# so pooling them doubles the human-demonstration data for a sentry policy.
AGENT_GROUPS = {
    "infantry": [TYPE_INFANTRY3, TYPE_INFANTRY4],
    "步兵": [TYPE_INFANTRY3, TYPE_INFANTRY4],
}


def resolve_agent(name: str) -> str:
    """Map an ASCII alias (or a Chinese type name) to the canonical 机器人类型."""
    return AGENT_ALIASES.get(name.strip().lower(), name)


def resolve_agents(name: str) -> list:
    """Resolve a comma-separated list / group name into robot types.

    'infantry' -> ['步兵3', '步兵4'];  'sentry' -> ['哨兵'];
    'infantry3,sentry' -> ['步兵3', '哨兵'].
    """
    out = []
    for part in str(name).split(","):
        part = part.strip()
        if not part:
            continue
        key = part.lower()
        if key in AGENT_GROUPS:
            out.extend(AGENT_GROUPS[key])
        elif part in AGENT_GROUPS:
            out.extend(AGENT_GROUPS[part])
        else:
            out.append(resolve_agent(part))
    seen, uniq = set(), []
    for t in out:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


def robot_id(rtype: str, camp: str) -> int:
    rid = RED_ID[rtype]
    return rid if camp == CAMP_RED else rid + BLUE_OFFSET


def all_robot_ids():
    ids = []
    for camp in CAMPS:
        for t in ALL_TYPES:
            ids.append(robot_id(t, camp))
    return ids


# ---------------------------------------------------------------------------
# Field geometry (metres).  The real RMUC arena is ~28 x 15 m; positions in the
# data outside this box are radar-tracking noise and get clipped.
# Blue robots are canonicalised into the red frame by a 180-degree rotation
# about the field centre, giving a side-invariant policy and doubling the data.
# ---------------------------------------------------------------------------
FIELD_X = 28.0
FIELD_Y = 15.0
FIELD_Z = 6.0

# ---------------------------------------------------------------------------
# Turret heading (枪口朝向)
# ---------------------------------------------------------------------------
# Measured over the whole dataset the column spans [-140, 220] degrees and is a
# plain math-convention bearing: atan2(dy, dx) in degrees, zero offset.  This was
# confirmed empirically — on seconds where a robot fired, the median angular
# error to the nearest live enemy is 6.7 deg (vs 42.1 deg when not firing).
#
# 工程 / 基地 / 前哨站 have no tracked turret and hold the constant sentinel
# below; those rows must never be used for aiming or engagement inference.
YAW_SENTINEL = -140.0

# max heat limits used to normalise the upgrade-tier capability features
HEAT17_MAX_REF = 260.0
HEAT42_MAX_REF = 240.0

# ---------------------------------------------------------------------------
# Event type strings (表 3)
# ---------------------------------------------------------------------------
EV_SHOOT = "发弹"
EV_HIT = "受击"
EV_ASSEMBLE = "装配成功"
EV_RUNE = "能量机关"
EV_DART_GATE = "飞镖闸门开"
EV_DART_HIT = "飞镖命中"
EV_RADAR_COUNTER = "雷达反制UAV"
EV_BUFF = "增益"
