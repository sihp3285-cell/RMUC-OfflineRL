"""Build target evidence from RM BattleScope attack attribution.

For every game, run BattleScope's ``infer_attacks`` on the cleaned tracks and
write one row per attributed hit:

    game_id, t (shot second), camp, robot_id, attacker_type,
    victim_camp, victim_robot_id, victim_type, caliber,
    confidence, ambiguous, score, damage, delay_s, angle_error_deg

This is the raw material for the target-label experiment.  It is deliberately
NOT joined to the RL shards here: the join key is ``(game_id, t, camp,
robot_id)`` and the consumer decides whether to use it as an auxiliary label,
a sample weight, or a reward-credit signal.

Run (from the RL repo root):

    python scripts/build_target_evidence.py \
        --db dataset/rmuc_2026_region_dataset.sqlite \
        --bs-root ../rm-battlescope \
        --out data/target_evidence.csv \
        --workers 8

The BattleScope checkout is used as-is (no vendoring yet); the manifest records
its git commit so the table is reproducible.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Populated in main() before the workers fork; also used by each worker.
_BS_ROOT = None
_DB = None


def _worker_init(bs_root: str, db: str):
    global _BS_ROOT, _DB
    _BS_ROOT, _DB = bs_root, db
    if bs_root not in sys.path:
        sys.path.insert(0, bs_root)


def _process_game(game_id: int) -> list[dict]:
    if _BS_ROOT not in sys.path:
        sys.path.insert(0, _BS_ROOT)
    from rmuc_trajectory.pipeline import (load_and_clean_tracks,
                                          load_match_events)
    from rmuc_trajectory.combat import infer_attacks

    db = Path(_DB)
    _, tracks = load_and_clean_tracks(db, int(game_id))
    _, objectives = load_and_clean_tracks(
        db, int(game_id), robot_types=("基地", "前哨站"),
        smooth_window=1, max_gap=3, max_speed_mps=8.0, include_static=True)
    events = load_match_events(db, int(game_id))
    attacks, _ = infer_attacks(events, tracks + objectives)
    rows = []
    for a in attacks:
        rows.append(dict(
            game_id=int(game_id),
            t=int(round(a.shot_time)),
            camp=a.attacker_camp,
            robot_id=int(a.attacker_robot_id),
            attacker_type=a.attacker_type,
            victim_camp=a.victim_camp,
            victim_robot_id=int(a.victim_robot_id),
            victim_type=a.victim_type,
            caliber=a.caliber,
            confidence=a.confidence,
            ambiguous=int(bool(a.ambiguous)),
            score=round(float(a.score), 4),
            damage=round(float(a.damage or 0.0), 3),
            delay_s=round(float(a.delay_s), 3),
            angle_error_deg=round(float(a.angle_error_deg), 3),
        ))
    return rows


def _git_commit(path: str) -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", path, "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL)
        return out.strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--bs-root", required=True,
                    help="path to the rm-battlescope checkout")
    ap.add_argument("--out", default="data/target_evidence.csv")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit-games", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    bs_root = str(Path(args.bs_root).resolve())
    db = str(Path(args.db).resolve())
    if bs_root not in sys.path:
        sys.path.insert(0, bs_root)

    import sqlite3
    con = sqlite3.connect(db)
    game_ids = [int(r[0]) for r in con.execute("SELECT game_id FROM matches")]
    con.close()
    if args.limit_games > 0:
        game_ids = game_ids[: args.limit_games]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["game_id", "t", "camp", "robot_id", "attacker_type",
              "victim_camp", "victim_robot_id", "victim_type", "caliber",
              "confidence", "ambiguous", "score", "damage", "delay_s",
              "angle_error_deg"]

    n_rows = 0
    done = 0
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        if args.workers <= 1:
            _worker_init(bs_root, db)
            for gid in game_ids:
                for row in _process_game(gid):
                    w.writerow(row)
                    n_rows += 1
                done += 1
                if done % 25 == 0 or done == len(game_ids):
                    print(f"[evidence] {done}/{len(game_ids)} games, {n_rows} rows",
                          flush=True)
        else:
            with ProcessPoolExecutor(max_workers=args.workers,
                                     initializer=_worker_init,
                                     initargs=(bs_root, db)) as ex:
                futs = {ex.submit(_process_game, gid): gid for gid in game_ids}
                for fut in as_completed(futs):
                    gid = futs[fut]
                    try:
                        rows = fut.result()
                    except Exception as exc:  # one bad game must not kill the run
                        print(f"[evidence] game {gid} failed: {exc}",
                              file=sys.stderr, flush=True)
                        rows = []
                    for row in rows:
                        w.writerow(row)
                        n_rows += 1
                    done += 1
                    if done % 25 == 0 or done == len(game_ids):
                        print(f"[evidence] {done}/{len(game_ids)} games, {n_rows} rows",
                              flush=True)

    manifest = dict(
        db=db, bs_root=bs_root, bs_commit=_git_commit(bs_root),
        n_games=len(game_ids), n_rows=n_rows, workers=args.workers,
        limit_games=args.limit_games, evidence=out.name,
        columns=fields,
        note="attacker identity is a heuristic; confidence is not calibrated",
    )
    (out.parent / "target_evidence_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
