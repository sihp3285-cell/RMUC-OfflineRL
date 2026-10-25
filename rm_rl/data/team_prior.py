"""Leak-safe historical team-strength priors.

The dataset names both schools of every match (96 teams, median 12 matches
each), so a "how good is this opponent" prior is computable.  Two rules make it
safe to feed into a policy:

1. **Strictly causal.**  A match's prior is built only from that team's
   *earlier* matches (chronological order by 开始时间, game_id).  Using the whole
   season would leak the outcome we are trying to predict and inflate every
   offline metric.
2. **Anonymous.**  We emit three continuous strength scalars, never a team
   identity / embedding.  An identity feature would let the policy memorise
   "against school X do Y", which does not transfer to an unseen opponent and is
   useless on competition day — you rarely have the other team's history loaded.

Per team, per match, relative to the league average *at that point in the
season*:

    winrate      prior wins / prior matches           (0.5 with no history)
    aggression   rounds fired per match, / league mean (1.0 with no history)
    durability   damage taken per match,  / league mean (1.0 with no history)
                 (>1 means it historically takes more damage, i.e. squishier)

Build once, reuse for every dataset build:

    python -m rm_rl.data.team_prior --db dataset/rmuc_2026_region_dataset.sqlite \\
        --out data/team_prior.json
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from collections import defaultdict

import numpy as np

from . import schema as S

N_TEAM_FEATS = 3            # winrate, aggression, durability (per team)
DEFAULTS = (0.5, 1.0, 1.0)


class TeamPrior:
    """Loaded per-(game, camp) strength triples."""

    def __init__(self, table: dict):
        self.table = table

    @classmethod
    def load(cls, path: str | None):
        if not path or not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return cls(json.load(f))

    def feats(self, game_id: int, camp: str) -> np.ndarray:
        """6 floats: own (winrate, aggression, durability) then the opponent's."""
        own = self.table.get(f"{int(game_id)}|{camp}", list(DEFAULTS))
        opp = self.table.get(f"{int(game_id)}|{S.enemy_camp(camp)}", list(DEFAULTS))
        return np.asarray(list(own) + list(opp), dtype=np.float32)


def feats_dim() -> int:
    return 2 * N_TEAM_FEATS


def build(db_path: str, out_path: str, verbose=True) -> dict:
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    matches = cur.execute(
        f'SELECT game_id, "红方学校", "蓝方学校", "胜方", "开始时间" '
        f"FROM {S.T_MATCHES}").fetchall()
    # chronological; 开始时间 is a text timestamp, game_id breaks ties
    matches.sort(key=lambda r: (str(r[4] or ""), int(r[0])))

    # --- per-match, per-camp raw stats -------------------------------------
    shots = defaultdict(float)   # (game_id, camp) -> rounds fired
    dmg = defaultdict(float)     # (game_id, camp) -> damage taken (positive)
    for gid, camp, n in cur.execute(
            f'SELECT game_id, "阵营", COUNT(*) FROM {S.T_EVENTS} '
            f'WHERE "事件类型"=? GROUP BY game_id, "阵营"', (S.EV_SHOOT,)):
        if camp in S.CAMPS:
            shots[(int(gid), camp)] = float(n)
    for gid, camp, v in cur.execute(
            f'SELECT game_id, "阵营", SUM(ABS("数值")) FROM {S.T_EVENTS} '
            f'WHERE "事件类型"=? GROUP BY game_id, "阵营"', (S.EV_HIT,)):
        if camp in S.CAMPS:
            dmg[(int(gid), camp)] = float(v or 0.0)
    con.close()

    # --- expanding, strictly-causal accumulation ---------------------------
    hist = defaultdict(lambda: dict(n=0, w=0.0, shots=0.0, dmg=0.0))
    league = dict(n=0, shots=0.0, dmg=0.0)
    table: dict[str, list] = {}

    for gid, red, blue, winner, _t in matches:
        gid = int(gid)
        lg_shots = league["shots"] / league["n"] if league["n"] else 0.0
        lg_dmg = league["dmg"] / league["n"] if league["n"] else 0.0

        for camp, school in ((S.CAMP_RED, red), (S.CAMP_BLUE, blue)):
            h = hist[school]
            if h["n"] > 0:
                wr = h["w"] / h["n"]
                ag = (h["shots"] / h["n"] / lg_shots) if lg_shots > 0 else 1.0
                du = (h["dmg"] / h["n"] / lg_dmg) if lg_dmg > 0 else 1.0
            else:
                wr, ag, du = DEFAULTS
            table[f"{gid}|{camp}"] = [round(float(np.clip(wr, 0.0, 1.0)), 5),
                                      round(float(np.clip(ag, 0.0, 3.0)), 5),
                                      round(float(np.clip(du, 0.0, 3.0)), 5)]

        # only *after* emitting the priors do we fold this match into history
        for camp, school in ((S.CAMP_RED, red), (S.CAMP_BLUE, blue)):
            h = hist[school]
            h["n"] += 1
            h["w"] += 1.0 if winner == camp else 0.0
            h["shots"] += shots.get((gid, camp), 0.0)
            h["dmg"] += dmg.get((gid, camp), 0.0)
            league["n"] += 1
            league["shots"] += shots.get((gid, camp), 0.0)
            league["dmg"] += dmg.get((gid, camp), 0.0)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(table, f, ensure_ascii=False)
    if verbose:
        cold = sum(1 for v in table.values() if v == list(DEFAULTS))
        print(f"[team_prior] {out_path}  entries={len(table)}  teams={len(hist)}  "
              f"cold-start(no history)={cold} ({cold / max(len(table),1):.1%})")
    return table


def main():
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", default="data/team_prior.json")
    args = ap.parse_args()
    build(args.db, args.out)


if __name__ == "__main__":
    main()
