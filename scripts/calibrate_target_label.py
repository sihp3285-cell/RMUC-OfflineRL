"""Calibrate the geometric target label against BattleScope target evidence.

For every attributed hit on an infantry attacker, ask: under the current
``target_soft(cone, tau)`` labeler, what probability mass does the *actual*
victim get?  Then sweep cone/tau and report coverage / top-1 / NLL.

This is the cheapest target-label experiment: it changes no network and no
shard, and answers whether the hand-set ``cone=15, tau=8`` is leaving signal on
the table — or whether the geometric label is fundamentally unable to name the
victim (the victim is outside every cone).

Usage:
    python scripts/calibrate_target_label.py \
        --db dataset/rmuc_2026_region_dataset.sqlite \
        --evidence data/target_evidence.csv \
        --out data/target_label_calibration.csv
"""
from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rm_rl.data import schema as S
from rm_rl.data.build_dataset import load_game_arrays
from rm_rl.data.features import target_soft

CONES = (10.0, 12.5, 15.0, 17.5, 20.0, 25.0, 30.0)
TAUS = (4.0, 6.0, 8.0, 10.0, 12.0)
WEIGHTS = {"high": 1.0, "medium": 0.5, "low": 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--out", default="data/target_label_calibration.csv")
    ap.add_argument("--limit-games", type=int, default=0)
    ap.add_argument("--include-medium", action="store_true",
                    help="include medium-confidence evidence (weight 0.5)")
    args = ap.parse_args()

    df = pd.read_csv(args.evidence)
    df = df[df["attacker_type"].isin([S.TYPE_INFANTRY3, S.TYPE_INFANTRY4])]
    df = df[df["victim_type"].isin(S.MOBILE_TYPES)]
    df = df[df["victim_type"] != S.TYPE_AERIAL]
    df = df[df["confidence"] == "high"]
    if args.include_medium:
        df = pd.read_csv(args.evidence)
        df = df[df["attacker_type"].isin([S.TYPE_INFANTRY3, S.TYPE_INFANTRY4])]
        df = df[df["victim_type"].isin(S.MOBILE_TYPES)]
        df = df[df["victim_type"] != S.TYPE_AERIAL]
        df = df[df["confidence"].isin(["high", "medium"])]
    if args.limit_games > 0:
        keep = sorted(df["game_id"].unique())[: args.limit_games]
        df = df[df["game_id"].isin(keep)]
    if df.empty:
        raise SystemExit("no evidence rows after filtering")

    # one target per firing second: keep the highest-score attribution
    df = (df.sort_values(["game_id", "t", "camp", "robot_id", "score"],
                         ascending=[True, True, True, True, False])
            .drop_duplicates(["game_id", "t", "camp", "robot_id"], keep="first"))

    con = sqlite3.connect(args.db)
    # accumulators: (cone, tau) -> lists
    acc = {(c, t): dict(n=0, hit=0.0, top1=0.0, nll=0.0, seen=0.0, miss=0.0)
           for c in CONES for t in TAUS}
    for gid, g in df.groupby("game_id"):
        game = load_game_arrays(con, int(gid))
        if game.T == 0:
            continue
        for camp in S.CAMPS:
            sub = g[g["camp"] == camp]
            if sub.empty:
                continue
            for agent in (S.TYPE_INFANTRY3, S.TYPE_INFANTRY4):
                rows = sub[sub["attacker_type"] == agent]
                if rows.empty:
                    continue
                for cone in CONES:
                    for tau in TAUS:
                        tgt = target_soft(game, camp, agent, cone_deg=cone, tau_deg=tau)
                        a = acc[(cone, tau)]
                        for r in rows.itertuples(index=False):
                            idx = int(r.t) - 1
                            if idx < 0 or idx >= len(tgt) - 1:
                                continue
                            slot = S.MOBILE_TYPES.index(r.victim_type)
                            p = float(tgt[idx, slot])
                            w = WEIGHTS.get(r.confidence, 0.0)
                            a["seen"] += w
                            a["n"] += 1
                            if p > 0:
                                a["hit"] += w
                                a["nll"] += -np.log(max(p, 1e-9)) * w
                            else:
                                a["miss"] += w
                            if int(tgt[idx].argmax()) == slot:
                                a["top1"] += w
    con.close()

    rows = []
    for (cone, tau), a in acc.items():
        seen = max(a["seen"], 1e-9)
        rows.append(dict(
            cone_deg=cone, tau_deg=tau, n=int(a["n"]),
            coverage=round(a["hit"] / seen, 4),
            top1=round(a["top1"] / seen, 4),
            nll=round(a["nll"] / seen, 4),
            missed=round(a["miss"] / seen, 4),
        ))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    rows.sort(key=lambda r: (-r["top1"], r["nll"]))
    print(f"evidence rows: {int(acc[(CONES[0], TAUS[0])]['n'])}")
    print(f"{'cone':>6} {'tau':>5} {'coverage':>9} {'top1':>7} {'nll':>7} {'missed':>7}")
    for r in rows[:12]:
        print(f"{r['cone_deg']:>6} {r['tau_deg']:>5} {r['coverage']:>9.4f} "
              f"{r['top1']:>7.4f} {r['nll']:>7.4f} {r['missed']:>7.4f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
