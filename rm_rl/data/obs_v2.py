"""Observation v2: the 161-D tactical observation plus 22 deployable slots.

The 161-D contract in :mod:`rm_rl.data.features` is kept byte-for-byte; this
module only *appends* 22 columns so that old checkpoints, old shards and old
parity tests remain valid.  The semantic-slot table is in

What is actually computed from the referee SQLite today
-------------------------------------------------------
* ``ego.respawn_remaining`` / ``ego.is_dead``  (E10)
* ``ego.paid_revivals_used`` / ``ego.can_buyback`` (E11, coin-drop heuristic)
* ``ego.buff[0..3]`` (E13, start event + rule duration)
* ``fortress.*`` (F01, geofence + outpost-destroyed rule)

What is a *placeholder* until the real information wave is recorded
-------------------------------------------------------------------
* ``ego.last_hit_*`` (E12, armor-hit direction)
* ``build.*.base_armor`` (B02)
* ``buff_points[*]`` (G01)

Placeholders are zeros and are listed in ``meta["extra_placeholder"]`` so a
training report can say so instead of silently claiming 183 real features.
They are deliberately kept in the array: the adapter and the model can be
implemented/tested now, and swapping in the real source later does not change
the observation width or the checkpoint's first layer.
"""
from __future__ import annotations

import math

import numpy as np

from . import schema as S
from .features import build_obs, obs_dim, obs_feature_names

# Order is part of the contract.  Do not reorder without bumping OBS_VERSION.
EXTRA_NAMES = [
    # E10
    "ego.respawn_remaining", "ego.is_dead",
    # E11
    "ego.paid_revivals_used", "ego.can_buyback",
    # E12 (placeholder)
    "ego.last_hit_dir_sin", "ego.last_hit_dir_cos",
    "ego.last_hit_age", "ego.last_hit_damage",
    # E13
    "ego.buff[0]", "ego.buff[1]", "ego.buff[2]", "ego.buff[3]",
    # B02 (placeholder)
    "build.own.base_armor", "build.enemy.base_armor",
    # F01
    "fortress.ego_in_own", "fortress.ego_in_enemy",
    "fortress.own_active", "fortress.enemy_vulnerable",
    # G01 (placeholder)
    "buff_points[0]", "buff_points[1]", "buff_points[2]", "buff_points[3]",
]

assert len(EXTRA_NAMES) == 22

# Placeholder columns (source not in the referee SQLite).  Kept as zeros so the
# width is stable; listed in meta.json so training reports are honest.
EXTRA_PLACEHOLDER = [
    "ego.last_hit_dir_sin", "ego.last_hit_dir_cos",
    "ego.last_hit_age", "ego.last_hit_damage",
    "build.own.base_armor", "build.enemy.base_armor",
    "buff_points[0]", "buff_points[1]", "buff_points[2]", "buff_points[3]",
]

# Rule-manual buff durations used by rm-battlescope (V2.1.0).  Regional data
# predates V2.1.0; these are approximations, not referee-confirmed ends.
BUFF_DURATIONS = {
    "飞坡": 30.0,
    "过中央高地": 30.0,
    "台阶跨越": 30.0,
    "小能量机关增益": 45.0,
    "大能量机关增益": 45.0,  # arm count not parsed here; BattleScope uses 30-60
}
TERRAIN_BUFFS = {"飞坡", "过中央高地", "台阶跨越"}

# Same approximate centres as rm-battlescope/rmuc_trajectory/field.py.
FORTRESS_POSITIONS = {"红": (6.22, 7.45), "蓝": (21.78, 7.45)}
FORTRESS_RADIUS_M = 1.3

RESPAWN_NORM = 30.0
PAID_NORM = 3.0
BUYBACK_COIN = 50.0  # placeholder cost; only used for the binary can_buyback


def obs_dim_v2(agent_type: str = S.TYPE_SENTRY) -> int:
    return obs_dim(agent_type) + len(EXTRA_NAMES)


def obs_feature_names_v2(agent_type: str = S.TYPE_SENTRY) -> list[str]:
    return obs_feature_names(agent_type) + list(EXTRA_NAMES)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _buff_slot(category: str) -> int:
    if category in TERRAIN_BUFFS:
        return 0
    if category == "小能量机关增益":
        return 1
    if category == "大能量机关增益":
        return 2
    return 3


def _paid_revivals(health: np.ndarray, max_health: np.ndarray,
                   coins: np.ndarray) -> np.ndarray:
    """Causal paid-revival detector.

    Full-health revive after a death plus a team-coin drop is the same signal
    rm-battlescope uses, but this version only looks at ``[i-1, i]`` so that a
    feature at time ``i`` never reads ``i+1``.  Returns a cumulative count per
    second (float32).
    """
    T = len(health)
    paid = np.zeros(T, np.float32)
    count = 0
    for i in range(1, T):
        if not np.isfinite(health[i]) or not np.isfinite(health[i - 1]):
            continue
        if max_health[i] <= 0 or max_health[i - 1] <= 0:
            continue
        if health[i - 1] > 0 or health[i] <= 0:
            continue
        if health[i] / max_health[i] < 0.75:
            continue
        drop = 0.0
        for j in (i - 1, i):
            if j > 0 and np.isfinite(coins[j - 1]) and np.isfinite(coins[j]):
                drop = max(drop, float(coins[j - 1] - coins[j]))
        if drop >= 50.0:
            count += 1
        paid[i] = count
    # forward-fill: a revival observed at i is known from i onward
    for i in range(1, T):
        if paid[i] == 0:
            paid[i] = paid[i - 1]
    return paid


def _respawn_and_buyback(ego, coins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """E10/E11 arrays: remaining/30, is_dead, paid/3, can_buyback."""
    T = len(ego.hp)
    remaining = np.zeros(T, np.float32)
    is_dead = np.zeros(T, np.float32)
    paid_cum = _paid_revivals(ego.hp, ego.maxhp, coins)
    can_buy = np.zeros(T, np.float32)

    valid = np.isfinite(ego.hp) & np.isfinite(ego.maxhp) & (ego.maxhp > 0)
    dead = valid & (ego.hp <= 0)

    death_time = None
    paid_before = 0
    last_paid = 0
    for i in range(T):
        if not valid[i]:
            death_time = None
            continue
        if dead[i]:
            if death_time is None:
                death_time = float(i + 1)  # arrays are 1..T seconds
                required = round(10.0 + death_time / 10.0 + 20.0 * paid_before)
            elapsed = (i + 1) - death_time
            remaining[i] = min(1.0, max(0.0, required - elapsed) / RESPAWN_NORM)
            is_dead[i] = 1.0
            can_buy[i] = 1.0 if (np.isfinite(coins[i]) and coins[i] >= BUYBACK_COIN) else 0.0
        else:
            death_time = None
        # count paid revivals that became known at or before i
        if paid_cum[i] > last_paid:
            paid_before = int(paid_cum[i])
            last_paid = paid_before
    return remaining, is_dead, np.minimum(paid_cum / PAID_NORM, 1.0), can_buy


def _buff_arrays(events, camp: str, robot_id: int, T: int
                 ) -> tuple[np.ndarray, np.ndarray]:
    """E13: remaining/30 for [terrain, small rune, big rune, other]."""
    out = np.zeros((T, 4), np.float32)
    if not events:
        return out, np.zeros(4, np.float32)
    for event in events:
        if event.get("etype") != "增益":
            continue
        if event.get("camp") != camp or event.get("robot_id") != robot_id:
            continue
        category = event.get("category")
        if not category:
            continue
        duration = BUFF_DURATIONS.get(str(category))
        if duration is None:
            continue
        try:
            start = int(round(float(event.get("t"))))
        except (TypeError, ValueError):
            continue
        slot = _buff_slot(str(category))
        for tt in range(max(0, start), min(T, start + int(math.ceil(duration)))):
            out[tt, slot] = max(out[tt, slot], (start + duration - (tt + 1)) / 30.0)
    return np.clip(out, 0.0, 1.0), np.ones(4, np.float32)


def _fortress_arrays(game, camp: str, ego, T: int) -> np.ndarray:
    """F01: in-own, in-enemy, own-active, enemy-vulnerable."""
    out = np.zeros((T, 4), np.float32)
    own = np.asarray(FORTRESS_POSITIONS[camp], np.float32)
    enemy_camp = S.enemy_camp(camp)
    enemy = np.asarray(FORTRESS_POSITIONS[enemy_camp], np.float32)
    xy = np.stack([ego.x, ego.y], axis=1)
    finite = np.isfinite(xy).all(axis=1)
    alive = ego.hp > 0
    d_own = np.linalg.norm(xy - own, axis=1)
    d_enemy = np.linalg.norm(xy - enemy, axis=1)
    out[:, 0] = (finite & alive & (d_own <= FORTRESS_RADIUS_M)).astype(np.float32)
    out[:, 1] = (finite & alive & (d_enemy <= FORTRESS_RADIUS_M)).astype(np.float32)

    own_out = game.get(S.TYPE_OUTPOST, camp)
    enemy_out = game.get(S.TYPE_OUTPOST, enemy_camp)
    out[:, 2] = ((own_out.hp <= 0) & np.isfinite(own_out.hp)).astype(np.float32)
    seconds = np.arange(1, T + 1, dtype=np.float32)
    out[:, 3] = (((enemy_out.hp <= 0) & np.isfinite(enemy_out.hp))
                 & (seconds >= 180.0)).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def compute_extras(game, events, agent_type: str, camp: str
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Return (extra[T,22] float32, known[22] float32).

    ``known`` is per-slot, not per-second: a zero column with ``known=0`` means
    "source unavailable", while a zero column with ``known=1`` means "source
    available and currently zero".  The adapter may use this to decide whether
    to apply dropout or a neutral value.
    """
    T = game.T
    ego = game.get(agent_type, camp)
    base = game.get(S.TYPE_BASE, camp)
    coins = base.coin_left
    extra = np.zeros((T, len(EXTRA_NAMES)), np.float32)
    known = np.ones(len(EXTRA_NAMES), np.float32)

    remaining, is_dead, paid, can_buy = _respawn_and_buyback(ego, coins)
    extra[:, 0] = remaining
    extra[:, 1] = is_dead
    extra[:, 2] = paid
    extra[:, 3] = can_buy

    buffs, buff_known = _buff_arrays(
        events, camp, S.robot_id(agent_type, camp), T)
    extra[:, 8:12] = buffs

    extra[:, 14:18] = _fortress_arrays(game, camp, ego, T)

    # Placeholders: source not in the referee SQLite.
    for idx, name in enumerate(EXTRA_NAMES):
        if name in EXTRA_PLACEHOLDER:
            known[idx] = 0.0
    # If no event table was provided, the buff source is unavailable, not zero.
    if not events:
        known[8:12] = 0.0

    return np.nan_to_num(extra, nan=0.0, posinf=0.0, neginf=0.0), known


def build_obs_v2(game, camp: str, agent_type: str, events=None, **kwargs) -> np.ndarray:
    """161-D :func:`build_obs` + 22 appended v2 slots.

    All keyword arguments (``vis_map``, ``team_feat``, ``vis_radius``, ...) are
    forwarded unchanged, so a v2 dataset is exactly a v1 dataset with 22 extra
    columns at the end.
    """
    base = build_obs(game, camp, agent_type, **kwargs)
    extra, _ = compute_extras(game, events, agent_type, camp)
    if extra.shape[0] != base.shape[0]:
        raise ValueError(f"extra length {extra.shape[0]} != obs length {base.shape[0]}")
    return np.concatenate([base, extra], axis=1).astype(np.float32)
