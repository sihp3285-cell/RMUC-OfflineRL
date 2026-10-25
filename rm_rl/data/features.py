"""Observation / action feature engineering for the RMUC offline dataset.

This module turns the per-second referee state of one game into fixed-length,
side-invariant observation and action vectors for a chosen *ego* robot.

Design choices (see README for the full rationale):
  * The policy acts at the referee sample rate of **1 Hz** — it is a *tactical*
    policy (where to go / whom to engage / whether weapons are free), not a
    motor-level controller.  Onboard classical controllers and auto-aim execute
    the command.
  * Blue-side games are mirrored (180-degree rotation about the field centre)
    into the red frame, so a single policy is side-invariant and the data set
    doubles.
  * The observation carries every entity's **capability tier** (HP / heat-limit
    upgrades and live chassis power), not just its position and health, so the
    policy can react to *how dangerous* a specific enemy currently is rather
    than treating all heroes alike.
  * The full referee state is visible offline, so the observation is
    *centralised* (ego + all allies + all enemies + buildings + economy).
    Decentralised execution: at deploy time you feed whatever the robot can
    actually perceive/estimate into the same slots (unknown enemy position ->
    pos_known = 0).

Action layouts
--------------
``velocity`` / ``goal``  (legacy, 4-D): vx, vy, yaw_rate, fire-rate.
``tactical``             (10-D): nav sub-goal + weapons-free gate + soft target
                         distribution.  See ``algos/action_spec.py`` for why the
                         turret rate and fire rate were wrong quantities to
                         predict at 1 Hz.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from . import schema as S
from ..algos.action_spec import (AERIAL_TARGET_SLOT, N_TARGET_CLASSES,
                                 NO_TARGET)

# ---------------------------------------------------------------------------
# Normalisation constants (physical scaling -> roughly unit range).
# A learned StandardScaler is fit on top of these at dataset-build time.
# ---------------------------------------------------------------------------
POWER_NORM = 120.0          # chassis power reference (W)
AMMO_NORM = 400.0           # cumulative 17mm rounds reference
COIN_NORM = 4000.0          # team coin reference
FIELD_DIAG = float(np.hypot(S.FIELD_X, S.FIELD_Y))

TARGET_CONE_DEG = 15.0      # muzzle cone that counts as "aiming at"
TARGET_TAU_DEG = 8.0        # softmax temperature over angular error (degrees)


@dataclass
class Entity:
    """Time-aligned arrays for one robot over a full game (shape [T])."""
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    hp: np.ndarray
    maxhp: np.ndarray
    yaw: np.ndarray
    power: np.ndarray
    heat17: np.ndarray
    heat17_max: np.ndarray
    heat42_max: np.ndarray
    ammo17: np.ndarray
    coin_left: np.ndarray
    coin_total: np.ndarray
    vuln: np.ndarray
    alive: np.ndarray

    @classmethod
    def empty(cls, T: int, rtype: str) -> "Entity":
        z = np.zeros(T, dtype=np.float32)
        return cls(x=z, y=z, z=z, hp=z, maxhp=z + S.MAX_HP.get(rtype, 1.0),
                   yaw=z, power=z, heat17=z, heat17_max=z, heat42_max=z,
                   ammo17=z, coin_left=z, coin_total=z, vuln=z, alive=z)


class GameArrays:
    """Holds every entity of one game as time-aligned numpy arrays."""

    def __init__(self, T: int, ent: Dict[int, Entity]):
        self.T = T
        self.ent = ent

    def get(self, rtype: str, camp: str) -> Entity:
        e = self.ent.get(S.robot_id(rtype, camp))
        return e if e is not None else Entity.empty(self.T, rtype)


# ---------------------------------------------------------------------------
# Canonicalisation (mirror blue -> red frame)
# ---------------------------------------------------------------------------
def _mirror_xy(x, y, mirror: bool):
    if not mirror:
        return x, y
    return S.FIELD_X - x, S.FIELD_Y - y


def _mirror_yaw(yaw, mirror: bool):
    if not mirror:
        return yaw
    return _wrap_deg(yaw + 180.0)


def _wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


def _pos_known(x, y):
    """Buildings and radar-lost robots sit at (0,0); treat as unknown."""
    return ((np.abs(x) > 1e-6) | (np.abs(y) > 1e-6)).astype(np.float32)


def yaw_valid(yaw) -> np.ndarray:
    """False where the turret heading is the 'no turret data' sentinel."""
    return (~np.isclose(np.asarray(yaw, np.float64), S.YAW_SENTINEL)).astype(np.float32)


def _tier(e: Entity, rtype: str) -> np.ndarray:
    """HP-upgrade tier in ~[0,1]: the level-scaled max HP of this entity."""
    return _safe_frac(e.maxhp, np.full_like(e.maxhp, S.MAX_HP.get(rtype, 400.0)))


def _capability(e: Entity, rtype: str):
    """4 features describing *how strong this unit currently is*.

    最大血量 / 小热量上限 / 大热量上限 all move with in-game level upgrades
    (infantry HP takes 11 distinct values 150->400, hero 15 values 150->450,
    hero 42mm heat limit 16 values 100->240), so together they encode the
    opponent's upgrade level — exactly what a targeted decision needs.
    底盘功率 is live power draw, a proxy for how hard the unit is moving.
    """
    return [
        _tier(e, rtype),
        e.heat17_max / S.HEAT17_MAX_REF,
        e.heat42_max / S.HEAT42_MAX_REF,
        e.power / POWER_NORM,
    ]


# ---------------------------------------------------------------------------
# Observation layout
# ---------------------------------------------------------------------------
def _ally_types(agent_type: str):
    """Team slots, in a FIXED order that does not depend on who the ego is.

    The ego occupies its own slot (relative position 0, still alive) rather than
    being removed from the list.  That keeps every observation column meaning the
    same thing for every role, which is what makes a policy **transferable across
    robot types** — train on the human-driven infantry, run the result on the
    sentry.  Dropping the ego from the list (the original behaviour) silently
    shifted the slot semantics per role and made such transfer impossible.

    Deliberately NOT flagged with a one-hot `is_ego`: such a flag is 1 only on
    the slot(s) actually trained on, so deploying on another chassis puts it on a
    column that was always zero, and the policy collapses to a constant command
    (measured — the sentry got an identical goal every timestep).  "Distance 0
    while alive" already identifies the ego unambiguously (dead allies zero out
    too, but carry alive=0), and that cue is role-agnostic, so it transfers.
    """
    return list(S.MOBILE_TYPES)


# feature counts, kept in one place so obs_dim() and build_obs() cannot drift
_EGO = 15          # 13 base + 17mm/42mm heat-limit tier
_TIME = 2
_PER_ALLY = 9      # 5 base + 4 capability (no role one-hot — see _ally_types)
_PER_ENEMY = 12    # 7 base + 4 capability + 1 engagement/visibility prior
_BUILD = 8
_ECON = 4
_TEAM = 6          # own (winrate, aggression, durability) + opponent's


def obs_dim(agent_type: str = S.TYPE_SENTRY) -> int:
    """Observation width. Identical for every role, by design (see _ally_types)."""
    n_ally = len(_ally_types(agent_type))          # always 6
    n_enemy = len(S.MOBILE_TYPES)                  # 6
    return (_EGO + _TIME + n_ally * _PER_ALLY + n_enemy * _PER_ENEMY
            + _BUILD + _ECON + _TEAM)


def action_dim(action_mode: str = "velocity") -> int:
    return 2 + 1 + N_TARGET_CLASSES if action_mode == "tactical" else 4


_CAP_NAMES = ("tier", "heat17max", "heat42max", "power")


def obs_feature_names(agent_type: str = S.TYPE_SENTRY) -> list[str]:
    """Human-readable name per observation column — keep in sync with build_obs.

    Used by `check_dataset` to tell a genuine wiring bug from a feature that is
    *legitimately* constant (a unit type with no upgrade on that axis: the
    sentry never levels its HP, infantry carry no 42mm barrel, and so on).
    """
    n = ["ego.x", "ego.y", "ego.z", "ego.hp_frac", "ego.alive", "ego.vuln",
         "ego.heat_frac", "ego.heat_headroom", "ego.power", "ego.sin_yaw",
         "ego.cos_yaw", "ego.ammo"] + [f"ego.{c}" for c in _CAP_NAMES[:3]]
    n += ["time.elapsed", "time.remaining"]
    for at in _ally_types(agent_type):
        n += [f"ally[{at}].{k}" for k in ("rx", "ry", "dist", "hp_frac", "alive")]
        n += [f"ally[{at}].{c}" for c in _CAP_NAMES]
    for et in S.MOBILE_TYPES:
        n += [f"enemy[{et}].{k}" for k in
              ("rx", "ry", "dist", "hp_frac", "alive", "vuln", "pos_known")]
        n += [f"enemy[{et}].{c}" for c in _CAP_NAMES]
        n += [f"enemy[{et}].vis_prior"]
    for side in ("own", "enemy"):
        n += [f"build.{side}.base_hp", f"build.{side}.base_alive",
              f"build.{side}.outpost_hp", f"build.{side}.outpost_alive"]
    n += ["econ.coin_left", "econ.coin_total", "econ.enemy_coin_total",
          "econ.coin_diff"]
    n += ["team.own_winrate", "team.own_aggression", "team.own_durability",
          "team.opp_winrate", "team.opp_aggression", "team.opp_durability"]
    assert len(n) == obs_dim(agent_type), (len(n), obs_dim(agent_type))
    return n


def is_capability_feature(name: str) -> bool:
    """True for upgrade-tier columns, which are constant for non-upgrading units."""
    return name.rsplit(".", 1)[-1] in _CAP_NAMES


ACTION_NAMES = ("vx", "vy", "yaw_rate", "fire")


def build_obs(game: GameArrays, camp: str, agent_type: str,
              t_max_ref: float = 450.0,
              vis_radius: float = 0.0, vis_dropout: float = 0.0,
              rng=None, vis_map=None,
              team_feat: Optional[np.ndarray] = None) -> np.ndarray:
    """Return observation array [T, obs_dim] for `agent_type` of `camp`.

    Observability (closing the train-vs-deploy gap): with full referee data every
    enemy position is known.  Set `vis_radius` (metres) to reveal an enemy's
    position only when it is within that range of the ego, and/or `vis_dropout`
    in [0,1) to randomly drop detections — mimicking what an on-board sentry can
    actually perceive.  Enemy HP/vuln stay known (the referee serial broadcasts
    them on the real robot); only *position* is gated.

    `vis_map` is an optional `vis_map.VisibilityMap`: a per-enemy empirical
    engagement prior standing in for line-of-sight, which the log does not
    record.  `team_feat` is the 6-D leak-safe opponent-strength prior.
    """
    T = game.T
    mirror = camp == S.CAMP_BLUE
    ecamp = S.enemy_camp(camp)
    if rng is None:
        rng = np.random.default_rng(0)

    ego = game.get(agent_type, camp)
    ex, ey = _mirror_xy(ego.x, ego.y, mirror)
    eyaw = np.deg2rad(_mirror_yaw(ego.yaw, mirror))

    feats = []

    # --- ego block (15) ---
    feats.append(ex / S.FIELD_X * 2 - 1)
    feats.append(ey / S.FIELD_Y * 2 - 1)
    feats.append(ego.z / S.FIELD_Z)
    feats.append(_safe_frac(ego.hp, ego.maxhp))
    feats.append(ego.alive)
    feats.append(ego.vuln)
    feats.append(_safe_frac(ego.heat17, ego.heat17_max))
    feats.append(_safe_frac(ego.heat17_max - ego.heat17, ego.heat17_max))
    feats.append(ego.power / POWER_NORM)
    feats.append(np.sin(eyaw))
    feats.append(np.cos(eyaw))
    feats.append(ego.ammo17 / AMMO_NORM)
    feats.append(_tier(ego, agent_type))                    # HP-upgrade tier
    feats.append(ego.heat17_max / S.HEAT17_MAX_REF)          # 17mm upgrade tier
    feats.append(ego.heat42_max / S.HEAT42_MAX_REF)          # 42mm upgrade tier

    # --- time block (2) ---
    tvec = np.arange(1, T + 1, dtype=np.float32)
    feats.append(tvec / t_max_ref)
    feats.append(np.clip(1.0 - tvec / t_max_ref, 0.0, 1.0))

    # --- team slots (9 each, fixed order incl. the ego's own slot) ---
    for at in _ally_types(agent_type):
        a = game.get(at, camp)
        ax, ay = _mirror_xy(a.x, a.y, mirror)
        rx, ry = (ax - ex), (ay - ey)
        rx, ry = rx * a.alive, ry * a.alive  # zero out if absent
        feats.append(rx / S.FIELD_X)
        feats.append(ry / S.FIELD_Y)
        feats.append(np.hypot(rx, ry) / FIELD_DIAG)
        feats.append(_safe_frac(a.hp, a.maxhp))
        feats.append(a.alive)
        feats.extend(_capability(a, at))

    # --- enemies (12 each) ---
    for et in S.MOBILE_TYPES:
        e = game.get(et, ecamp)
        px, py = _mirror_xy(e.x, e.y, mirror)
        dx, dy = px - ex, py - ey
        dist_m = np.hypot(dx, dy)
        known = _pos_known(e.x, e.y) * e.alive
        if vis_radius and vis_radius > 0:
            known = known * (dist_m <= vis_radius).astype(np.float32)
        if vis_dropout and vis_dropout > 0:
            known = known * (rng.random(T) > vis_dropout).astype(np.float32)
        rx, ry = dx * known, dy * known
        feats.append(rx / S.FIELD_X)
        feats.append(ry / S.FIELD_Y)
        feats.append(np.where(known > 0, np.hypot(rx, ry) / FIELD_DIAG, 1.0))
        feats.append(_safe_frac(e.hp, e.maxhp))
        feats.append(e.alive)
        feats.append(e.vuln)
        feats.append(known)
        feats.extend(_capability(e, et))
        if vis_map is not None:
            feats.append(vis_map.prior(ex, ey, px, py, known))
        else:
            feats.append(np.zeros(T, np.float32))

    # --- buildings (8) ---
    for owner in (camp, ecamp):
        base = game.get(S.TYPE_BASE, owner)
        out = game.get(S.TYPE_OUTPOST, owner)
        feats.append(_safe_frac(base.hp, base.maxhp))
        feats.append((base.hp > 0).astype(np.float32))
        feats.append(_safe_frac(out.hp, out.maxhp))
        feats.append((out.hp > 0).astype(np.float32))

    # --- economy (4) ---
    own = game.get(S.TYPE_BASE, camp)      # coins are team-wide; read off base row
    enemy = game.get(S.TYPE_BASE, ecamp)
    feats.append(own.coin_left / COIN_NORM)
    feats.append(own.coin_total / COIN_NORM)
    feats.append(enemy.coin_total / COIN_NORM)
    feats.append((own.coin_total - enemy.coin_total) / COIN_NORM)

    # --- opponent-strength prior (6), constant across the game ---
    tf = (np.zeros(_TEAM, np.float32) if team_feat is None
          else np.asarray(team_feat, np.float32))
    for k in range(_TEAM):
        feats.append(np.full(T, tf[k], np.float32))

    obs = np.stack(feats, axis=1).astype(np.float32)
    return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------
def target_soft(game: GameArrays, camp: str, agent_type: str,
                cone_deg: float = TARGET_CONE_DEG,
                tau_deg: float = TARGET_TAU_DEG) -> np.ndarray:
    """Soft target distribution [T, 7] recovered from the muzzle bearing.

    The log records *that* a robot fired but never *at whom*, so we infer it
    geometrically: the angular error between the turret heading and the bearing
    to each live enemy.  Enemies inside `cone_deg` get softmax(-err / tau)
    weight; if none are inside the cone the whole mass goes to the "no target"
    class (index 6).

    Validated against the log: on firing seconds the median error to the nearest
    live enemy is 6.7 deg vs 42.1 deg on non-firing seconds, and 77.5% of firing
    seconds have an enemy inside a 15 deg cone.  But >1 enemy sits in the cone
    55% of the time, so the assignment is genuinely ambiguous — hence a soft
    label rather than an invented hard class.
    """
    T = game.T
    mirror = camp == S.CAMP_BLUE
    ecamp = S.enemy_camp(camp)
    ego = game.get(agent_type, camp)
    ex, ey = _mirror_xy(ego.x, ego.y, mirror)
    yaw = _mirror_yaw(ego.yaw, mirror)

    err = np.full((T, N_TARGET_CLASSES - 1), np.inf, dtype=np.float64)
    for j, et in enumerate(S.MOBILE_TYPES):
        e = game.get(et, ecamp)
        px, py = _mirror_xy(e.x, e.y, mirror)
        ok = (_pos_known(e.x, e.y) * e.alive) > 0
        bearing = np.degrees(np.arctan2(py - ey, px - ex))
        d = np.abs(_wrap_deg(bearing - yaw))
        err[:, j] = np.where(ok, d, np.inf)

    # the ego needs a real turret reading for any of this to mean anything
    ok_yaw = (yaw_valid(ego.yaw) > 0) & (ego.alive > 0) & (_pos_known(ego.x, ego.y) > 0)

    out = np.zeros((T, N_TARGET_CLASSES), np.float32)
    inside = err < cone_deg
    any_in = inside.any(axis=1) & ok_yaw
    if any_in.any():
        e_in = np.where(inside[any_in], err[any_in], np.inf)
        logits = -e_in / max(tau_deg, 1e-6)
        logits -= logits.max(axis=1, keepdims=True)
        w = np.exp(logits)
        w[~np.isfinite(e_in)] = 0.0
        w /= np.maximum(w.sum(axis=1, keepdims=True), 1e-9)
        out[any_in, :N_TARGET_CLASSES - 1] = w.astype(np.float32)
    out[~any_in, NO_TARGET] = 1.0
    # The aerial unit is not a valid ground target: drop its mass and
    # renormalise.  If it was the only enemy in the cone, the shot becomes
    # "no target" rather than a class the policy must never pick.
    out[:, AERIAL_TARGET_SLOT] = 0.0
    row = out.sum(axis=1, keepdims=True)
    empty = row[:, 0] <= 1e-9
    out[~empty] /= row[~empty]
    out[empty, :] = 0.0
    out[empty, NO_TARGET] = 1.0
    return out


def fire_gate(game: GameArrays, camp: str, agent_type: str) -> np.ndarray:
    """Binary weapons-free gate [T] — did this robot put rounds downrange?

    Read straight off the cumulative 17mm counter, so this is an *exact* label
    (no inference): the referee log records every shot with the shooter's id.
    """
    ego = game.get(agent_type, camp)
    fired = np.diff(ego.ammo17, prepend=ego.ammo17[:1]) > 0.5
    return fired.astype(np.float32)


def build_action_raw(game: GameArrays, camp: str, agent_type: str,
                     action_mode: str = "velocity", goal_horizon: int = 5,
                     target_cone: float = TARGET_CONE_DEG,
                     target_tau: float = TARGET_TAU_DEG
                     ) -> np.ndarray:
    """Raw action array [T-1, D] describing transition t -> t+1.

    action_mode:
      * "velocity" : (dx, dy) per-second displacement — a raw velocity command.
      * "goal"     : (gx, gy) displacement to where the robot actually is
                     `goal_horizon` seconds later — a navigation *sub-goal*.
      * "tactical" : (gx, gy, fire_gate, target[7]) — the deployable layout.
                     The turret channel becomes *whom to engage* (auto-aim owns
                     the gimbal) and the fire channel becomes a *permission bit*
                     (auto-aim owns the trigger), instead of a 1 Hz turret rate
                     and rounds-per-second, neither of which the decision layer
                     can or should command.

    Values are in physical units; normalisation to [-1, 1] happens later — for
    `tactical` only the first two (nav) channels are scaled, the gate and the
    target distribution are already bounded.
    """
    mirror = camp == S.CAMP_BLUE
    ego = game.get(agent_type, camp)
    x, y = _mirror_xy(ego.x, ego.y, mirror)
    yaw = _mirror_yaw(ego.yaw, mirror)

    if action_mode in ("goal", "tactical"):
        T = len(x)
        ahead = np.minimum(np.arange(T) + goal_horizon, T - 1)
        vx = (x[ahead] - x)[:-1]
        vy = (y[ahead] - y)[:-1]
    else:
        vx = np.diff(x)
        vy = np.diff(y)

    live = (ego.alive[:-1] > 0) & (ego.alive[1:] > 0)

    if action_mode == "tactical":
        gate = fire_gate(game, camp, agent_type)[:-1]
        tgt = target_soft(game, camp, agent_type,
                          cone_deg=target_cone, tau_deg=target_tau)[:-1]
        act = np.concatenate(
            [np.stack([vx, vy], axis=1), gate[:, None], tgt], axis=1
        ).astype(np.float32)
        # dead steps are not decisions: zero the nav + gate, park the target
        # distribution on "no target" so the categorical loss stays well-formed
        act[~live, :3] = 0.0
        act[~live, 3:] = 0.0
        act[~live, 3 + NO_TARGET] = 1.0
        return np.nan_to_num(act, nan=0.0, posinf=0.0, neginf=0.0)

    yaw_rate = _wrap_deg(np.diff(yaw))
    fire = np.clip(np.diff(ego.ammo17), 0.0, None)
    # When the robot is dead at either end of the step, the "action" is not a
    # decision — zero it so it does not pollute the policy target.
    act = np.stack([vx, vy, yaw_rate, fire], axis=1).astype(np.float32)
    act *= live[:, None]
    return np.nan_to_num(act, nan=0.0, posinf=0.0, neginf=0.0)


def _safe_frac(num, den):
    den = np.where(np.abs(den) < 1e-6, 1.0, den)
    return np.clip(num / den, 0.0, 1.5).astype(np.float32)
