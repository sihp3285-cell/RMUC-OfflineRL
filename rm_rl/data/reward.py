"""Reward shaping for the RMUC offline dataset.

The dataset has **no reward column** — only real logged matches.  We therefore
*define* good vs. bad behaviour from observable game outcomes.  The reward is
computed from the **acting team's** perspective (teammates cooperate, the enemy
team is adversarial) because per-agent damage credit is not recoverable from the
data (the `受击` events record the victim, not the attacker).

Components (all configurable weights, see `RewardConfig`):
  attack   : enemy outpost / base HP removed, enemy team HP removed, enemy kills
  defend   : own outpost / base HP lost, own team HP lost, own deaths (negative)
  ego      : the controlled robot staying alive & keeping HP (defensive for a
             sentry), heat-overflow penalty
  economy  : growth of the team coin advantage
  terminal : +/- for winning / losing the round (from matches.胜方), plus a
             final base-HP differential shaping term

Because the reward is dense + outcome-grounded, the *return* it induces is what
separates "good" from "bad" transitions; the offline-RL value function then
learns that separation without any hand labelling.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Tuple

import numpy as np

from . import schema as S
from .features import GameArrays

# normalisation denominators (keep every term ~O(1) per second)
HP_NORM = 200.0
OUT_NORM = 1500.0
BASE_NORM = 5000.0
EGO_HP_NORM = 200.0
COIN_NORM = 1000.0


@dataclass
class RewardConfig:
    # attack
    w_outpost_atk: float = 3.0
    w_base_atk: float = 5.0
    w_dmg_deal: float = 1.0
    w_kill: float = 2.0
    # defend
    w_outpost_def: float = 3.0
    w_base_def: float = 5.0
    w_dmg_take: float = 1.0
    w_death: float = 2.0
    # ego (sentry defensive survival)
    w_ego_surv: float = 0.02
    w_ego_hp: float = 1.0
    w_heat: float = 0.5
    # economy
    w_econ: float = 0.5
    # terminal
    w_win: float = 20.0
    w_final_base: float = 5.0

    @staticmethod
    def from_dict(d: dict) -> "RewardConfig":
        base = RewardConfig()
        for k, v in (d or {}).items():
            if hasattr(base, k):
                setattr(base, k, float(v))
        return base


def _team_hp(game: GameArrays, camp: str) -> np.ndarray:
    hp = np.zeros(game.T, dtype=np.float32)
    for t in S.MOBILE_TYPES:
        hp = hp + game.get(t, camp).hp
    return hp


def _deaths(hp: np.ndarray) -> np.ndarray:
    """Per-step death indicator: hp>0 at t and ==0 at t+1. Length T-1."""
    return ((hp[:-1] > 0) & (hp[1:] <= 0)).astype(np.float32)


def _team_deaths(game: GameArrays, camp: str) -> np.ndarray:
    d = np.zeros(game.T - 1, dtype=np.float32)
    for t in S.MOBILE_TYPES:
        d = d + _deaths(game.get(t, camp).hp)
    return d


def compute_reward(game: GameArrays, camp: str, agent_type: str,
                   winner: str, cfg: RewardConfig
                   ) -> Tuple[np.ndarray, Dict[str, float]]:
    """Return (reward[T-1], component-sum dict)."""
    ecamp = S.enemy_camp(camp)
    T = game.T

    own_hp = _team_hp(game, camp)
    en_hp = _team_hp(game, ecamp)
    own_out = game.get(S.TYPE_OUTPOST, camp).hp
    en_out = game.get(S.TYPE_OUTPOST, ecamp).hp
    own_base = game.get(S.TYPE_BASE, camp).hp
    en_base = game.get(S.TYPE_BASE, ecamp).hp
    own_coin = game.get(S.TYPE_BASE, camp).coin_total
    en_coin = game.get(S.TYPE_BASE, ecamp).coin_total

    ego = game.get(agent_type, camp)

    def drop(a):  # positive when the quantity decreased
        return np.clip(a[:-1] - a[1:], 0.0, None)

    comp = {}
    r = np.zeros(T - 1, dtype=np.float32)

    comp["outpost_atk"] = cfg.w_outpost_atk * drop(en_out) / OUT_NORM
    comp["outpost_def"] = -cfg.w_outpost_def * drop(own_out) / OUT_NORM
    comp["base_atk"] = cfg.w_base_atk * drop(en_base) / BASE_NORM
    comp["base_def"] = -cfg.w_base_def * drop(own_base) / BASE_NORM
    comp["dmg_deal"] = cfg.w_dmg_deal * drop(en_hp) / HP_NORM
    comp["dmg_take"] = -cfg.w_dmg_take * drop(own_hp) / HP_NORM
    comp["kill"] = cfg.w_kill * _team_deaths(game, ecamp)
    comp["death"] = -cfg.w_death * _team_deaths(game, camp)
    comp["ego_surv"] = cfg.w_ego_surv * (ego.alive[1:] > 0).astype(np.float32)
    comp["ego_hp"] = -cfg.w_ego_hp * drop(ego.hp) / EGO_HP_NORM
    overflow = np.clip(ego.heat17[1:] - ego.heat17_max[1:], 0.0, None)
    comp["heat"] = -cfg.w_heat * overflow / 100.0
    adv = (own_coin - en_coin)
    comp["econ"] = cfg.w_econ * (adv[1:] - adv[:-1]) / COIN_NORM

    for v in comp.values():
        r = r + v

    # terminal reward on the last transition
    if winner in (S.CAMP_RED, S.CAMP_BLUE):
        r[-1] += cfg.w_win * (1.0 if winner == camp else -1.0)
    final_base_diff = (own_base[-1] - en_base[-1]) / BASE_NORM
    r[-1] += cfg.w_final_base * float(final_base_diff)

    comp_sum = {k: float(np.sum(v)) for k, v in comp.items()}
    comp_sum["terminal"] = float(cfg.w_win * (1.0 if winner == camp else -1.0)
                                 + cfg.w_final_base * final_base_diff)
    return r, comp_sum


def reward_config_defaults() -> dict:
    return asdict(RewardConfig())


# names of the shaping components, in a fixed order (for the learned reward model)
FEATURE_NAMES = [
    "outpost_atk", "outpost_def", "base_atk", "base_def",
    "dmg_deal", "dmg_take", "kill", "death",
    "ego_surv", "ego_hp", "heat", "econ", "final_base",
]


def reward_features(game: GameArrays, camp: str, agent_type: str) -> np.ndarray:
    """Per-episode, UNWEIGHTED sums of each shaping component (for the acting
    team). Used to *learn* reward weights from match outcomes instead of hand
    setting them. Order matches FEATURE_NAMES."""
    ecamp = S.enemy_camp(camp)
    own_hp = _team_hp(game, camp)
    en_hp = _team_hp(game, ecamp)
    own_out = game.get(S.TYPE_OUTPOST, camp).hp
    en_out = game.get(S.TYPE_OUTPOST, ecamp).hp
    own_base = game.get(S.TYPE_BASE, camp).hp
    en_base = game.get(S.TYPE_BASE, ecamp).hp
    own_coin = game.get(S.TYPE_BASE, camp).coin_total
    en_coin = game.get(S.TYPE_BASE, ecamp).coin_total
    ego = game.get(agent_type, camp)

    def drop(a):
        return float(np.clip(a[:-1] - a[1:], 0.0, None).sum())

    adv = own_coin - en_coin
    overflow = float(np.clip(ego.heat17[1:] - ego.heat17_max[1:], 0.0, None).sum())
    return np.array([
        drop(en_out) / OUT_NORM,
        drop(own_out) / OUT_NORM,
        drop(en_base) / BASE_NORM,
        drop(own_base) / BASE_NORM,
        drop(en_hp) / HP_NORM,
        drop(own_hp) / HP_NORM,
        float(_team_deaths(game, ecamp).sum()),
        float(_team_deaths(game, camp).sum()),
        float((ego.alive[1:] > 0).sum()) / 100.0,
        drop(ego.hp) / EGO_HP_NORM,
        overflow / 100.0,
        float((adv[-1] - adv[0]) / COIN_NORM),
        float((own_base[-1] - en_base[-1]) / BASE_NORM),
    ], dtype=np.float64)
