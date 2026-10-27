"""Collect (feature, teacher-goal) pairs from the local Arena Rust engine."""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from rm_rl.waypoint.features import build_features, ego_xy, obs_to_frame
from rm_rl.waypoint.teachers import StrategyRunner, TracedSentryTeacher

_CFG = None
_MATCH = None
_TEACHER = None
_BASELINE = None
_OPPONENT = None


def _worker_init(cfg):
    global _CFG, _MATCH, _TEACHER, _BASELINE, _OPPONENT
    _CFG = cfg
    app_root = str(cfg["app_root"])
    sdk_dir = str(Path(app_root) / "rmuc_arena")
    if app_root not in sys.path:
        sys.path.insert(0, app_root)
    if sdk_dir not in sys.path:
        sys.path.insert(0, sdk_dir)
    from rmuc_arena.engine import Match
    from rmuc_arena.models import mirrored_lineups
    _MATCH = Match
    _TEACHER = TracedSentryTeacher(cfg["teacher"], sdk_dir)
    _BASELINE = StrategyRunner(cfg["baseline"], sdk_dir)
    _OPPONENT = StrategyRunner(cfg["opponent"], sdk_dir)
    lineup = {"1": 2711, "3": 2731, "4": 2732, "6": 2761, "7": 2772}
    _CFG["lineups"] = mirrored_lineups(lineup, "2027")


def _valid_goal(goal):
    if goal is None:
        return False
    x, y = goal
    return 0.0 <= x <= 28.0 and 0.0 <= y <= 15.0


def _run_one(seed):
    try:
        match = _MATCH(seed=int(seed), duration=int(_CFG["duration"]),
                       map_name="rmuc2027", lineups=_CFG["lineups"])
        teacher_state = None
        baseline_state = None
        opponent_state = None
        features, labels, goals, egos, times, metas = [], [], [], [], [], []
        while match.status != "finished":
            frame = match.snapshot()
            ex, ey = ego_xy(frame)

            red_obs = match.observation("red", [1, 3, 4, 6])
            red_actions, baseline_state = _BASELINE.act(red_obs, baseline_state)
            teacher_obs = match.observation("red", [7])
            feat = build_features(obs_to_frame(teacher_obs) if _CFG.get("legal_obs") else frame)
            teacher_actions, teacher_state = _TEACHER.act(teacher_obs, teacher_state)
            blue_obs = match.observation("blue", [1, 3, 4, 6, 7])
            blue_actions, opponent_state = _OPPONENT.act(blue_obs, opponent_state)

            red_cmd = {}
            for slot in ("1", "3", "4", "6"):
                if isinstance(red_actions, dict) and slot in red_actions:
                    red_cmd[slot] = red_actions[slot]
            if isinstance(teacher_actions, dict) and "7" in teacher_actions:
                red_cmd["7"] = teacher_actions["7"]
            blue_cmd = {}
            for slot in ("1", "3", "4", "6", "7"):
                if isinstance(blue_actions, dict) and slot in blue_actions:
                    blue_cmd[slot] = blue_actions[slot]

            goal = _TEACHER.last_goal
            if _valid_goal(goal):
                features.append(feat)
                labels.append([goal[0] / 28.0, goal[1] / 15.0])
                goals.append(goal)
                egos.append([ex, ey])
                times.append(float(frame.get("time", 0.0)))
                metas.append([int(seed), int(_CFG["opponent_id"]), 0, int(frame.get("tick", 0))])

            commands = {"red": red_cmd, "blue": blue_cmd}
            for _ in range(2):
                match.step(commands)
                if match.status == "finished":
                    break
        match.close()
        if not features:
            return None
        out = Path(_CFG["out"])
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"match-{int(seed):04d}.npz"
        np.savez_compressed(
            path,
            features=np.asarray(features, dtype=np.float32),
            labels=np.asarray(labels, dtype=np.float32),
            goals=np.asarray(goals, dtype=np.float32),
            ego=np.asarray(egos, dtype=np.float32),
            time=np.asarray(times, dtype=np.float32),
            meta=np.asarray(metas, dtype=np.int32),
        )
        return str(path)
    except Exception:
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--app-root", default="/home/delphine/rm/arena-runtime/squashfs-root/resources/runtime/app")
    parser.add_argument("--teacher", default="/home/delphine/.config/Arena-1/.arena_data/strategy-library/versions/17978d64f27265c1a3acc7c074f0802bb4b3ee349598428c87136c0e5255d72f/strategies/gpt61_2027_adaptive_2772/strategy.py")
    parser.add_argument("--baseline", default="/home/delphine/rm/arena-runtime/squashfs-root/resources/runtime/app/examples/claude_opus55_xhigh_2027_hold/strategy.py")
    parser.add_argument("--opponent", choices=("hold", "adaptive_2731"), default="hold")
    parser.add_argument("--matches", type=int, default=8)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--duration", type=int, default=420)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", default="data/waypoint_mvp")
    parser.add_argument("--legal-obs", action="store_true")
    args = parser.parse_args()

    app = Path(args.app_root)
    opponents = {
        "hold": app / "examples/claude_opus55_xhigh_2027_hold/strategy.py",
        "adaptive_2731": app / "examples/claude_opus55_xhigh_2027_adaptive_2731/strategy.py",
    }
    cfg = {
        "app_root": str(app),
        "teacher": args.teacher,
        "baseline": args.baseline,
        "opponent": str(opponents[args.opponent]),
        "opponent_id": 0 if args.opponent == "hold" else 1,
        "duration": args.duration,
        "out": args.out,
        "legal_obs": bool(args.legal_obs),
    }
    for key in ("teacher", "baseline", "opponent"):
        if not Path(cfg[key]).is_file():
            raise SystemExit(f"missing {key}: {cfg[key]}")
    seeds = list(range(args.seed_start, args.seed_start + args.matches))
    print(json.dumps({"seeds": seeds, "workers": args.workers, "out": args.out}, ensure_ascii=False))
    with Pool(processes=args.workers, initializer=_worker_init, initargs=(cfg,), maxtasksperchild=2) as pool:
        for path in pool.imap_unordered(_run_one, seeds, chunksize=1):
            print("done", path, flush=True)


if __name__ == "__main__":
    main()
