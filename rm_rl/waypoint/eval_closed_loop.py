"""Closed-loop A/B: teacher goal vs distilled student goal in the local engine."""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

from rm_rl.waypoint.features import build_features, rule_features, anchor_features, sequence_context_one
from rm_rl.waypoint.model import WaypointMLP
from rm_rl.waypoint.teachers import StrategyRunner, TracedSentryTeacher

_CFG = None
_MATCH = None
_TEACHER = None
_BASELINE = None
_OPPONENT = None


class StudentTeacher:
    """Teacher low-level skills with the student's goal substituted."""

    def __init__(self, ckpt, sdk_dir):
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        if state.get("observation_schema", "arena307") != "arena307":
            raise ValueError("Use radar_sim for radar checkpoints")
        if "anchors" in state:
            from rm_rl.waypoint.model import WaypointClassifier
            self.model = WaypointClassifier(state["obs_dim"], state["n_classes"], state["hidden"], state["depth"])
        else:
            self.model = WaypointMLP(state["obs_dim"], state["hidden"], state["depth"])
        self.model.load_state_dict(state["state_dict"])
        self.model.eval()
        self.mean = np.asarray(state["mean"], dtype=np.float32)
        self.std = np.asarray(state["std"], dtype=np.float32)
        self.anchors = np.asarray(state["anchors"], dtype=np.float32) if "anchors" in state else None
        self.history = []
        self.module = None
        self.original_drive = None
        self.predicted = (10.8, 7.5)
        self.last_goal = None
        self.frame = None
        self._load(sdk_dir)

    def _load(self, sdk_dir):
        runner = StrategyRunner(_CFG["teacher"], sdk_dir)
        self.module = runner.module
        self.original_drive = self.module.drive
        self.module.drive = self._drive

    def _drive(self, robot, goal, obs, memory, tolerance=0.35):
        if robot.get("role") == "sentry" and str(robot.get("slot")) == "7":
            goal = (float(self.predicted[0]), float(self.predicted[1]))
        return self.original_drive(robot, goal, obs, memory, tolerance)

    def act(self, observation, state):
        frame = self.frame
        if frame is None:
            return self.module.act(observation, state)
        feat = build_features(frame)
        feat = sequence_context_one(feat, self.history)
        self.history.append(build_features(frame))
        self.history = self.history[-100:]
        feat = rule_features(feat)
        if getattr(self, "anchors", None) is not None:
            feat = anchor_features(feat, self.anchors)
        feat = (feat - self.mean) / self.std
        with torch.no_grad():
            pred = self.model(torch.from_numpy(feat).unsqueeze(0)).numpy()[0]
        if self.anchors is not None:
            x, y = self.anchors[int(np.argmax(pred))]
        else:
            x, y = float(pred[0]) * 28.0, float(pred[1]) * 15.0
        if not (np.isfinite(x) and np.isfinite(y)):
            x, y = 10.8, 7.5
        self.predicted = (float(min(28.0, max(0.0, x))), float(min(15.0, max(0.0, y))))
        self.last_goal = self.predicted
        return self.module.act(observation, state)


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
    _BASELINE = StrategyRunner(cfg["baseline"], sdk_dir)
    _OPPONENT = StrategyRunner(cfg["opponent"], sdk_dir)
    if cfg["mode"] == "teacher":
        _TEACHER = TracedSentryTeacher(cfg["teacher"], sdk_dir)
    else:
        _TEACHER = StudentTeacher(cfg["ckpt"], sdk_dir)
    _CFG["lineups"] = mirrored_lineups({"1": 2711, "3": 2731, "4": 2732, "6": 2761, "7": 2772}, "2027")


def _run_one(seed):
    try:
        match = _MATCH(seed=int(seed), duration=int(_CFG["duration"]), map_name="rmuc2027",
                       lineups=_CFG["lineups"])
        teacher_state = None
        baseline_state = None
        opponent_state = None
        goal_errors = []
        while match.status != "finished":
            frame = match.snapshot()
            if isinstance(_TEACHER, StudentTeacher):
                _TEACHER.frame = frame
            red_obs = match.observation("red", [1, 3, 4, 6])
            red_actions, baseline_state = _BASELINE.act(red_obs, baseline_state)
            teacher_obs = match.observation("red", [7])
            teacher_actions, teacher_state = _TEACHER.act(teacher_obs, teacher_state)
            blue_obs = match.observation("blue", [1, 3, 4, 6, 7])
            blue_actions, opponent_state = _OPPONENT.act(blue_obs, opponent_state)
            red_cmd = {slot: red_actions[slot] for slot in ("1", "3", "4", "6")
                       if isinstance(red_actions, dict) and slot in red_actions}
            if isinstance(teacher_actions, dict) and "7" in teacher_actions:
                red_cmd["7"] = teacher_actions["7"]
            blue_cmd = {slot: blue_actions[slot] for slot in ("1", "3", "4", "6", "7")
                        if isinstance(blue_actions, dict) and slot in blue_actions}
            for _ in range(2):
                match.step({"red": red_cmd, "blue": blue_cmd})
                if match.status == "finished":
                    break
        frame = match.snapshot()
        result = {
            "seed": int(seed), "mode": _CFG["mode"], "winner": match.winner,
            "reason": match.reason, "time": float(match.time),
            "teams": frame.get("teams", {}),
            "bases": frame.get("bases", {}),
        }
        match.close()
        return result
    except Exception:
        traceback.print_exc()
        return {"seed": int(seed), "mode": _CFG["mode"], "error": "worker failed"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--app-root", default="/home/delphine/rm/arena-runtime/squashfs-root/resources/runtime/app")
    parser.add_argument("--teacher", default="/home/delphine/.config/Arena-1/.arena_data/strategy-library/versions/17978d64f27265c1a3acc7c074f0802bb4b3ee349598428c87136c0e5255d72f/strategies/gpt61_2027_adaptive_2772/strategy.py")
    parser.add_argument("--baseline", default="/home/delphine/rm/arena-runtime/squashfs-root/resources/runtime/app/examples/claude_opus55_xhigh_2027_hold/strategy.py")
    parser.add_argument("--opponent", default="/home/delphine/rm/arena-runtime/squashfs-root/resources/runtime/app/examples/claude_opus55_xhigh_2027_hold/strategy.py")
    parser.add_argument("--ckpt", default="rm_runs/waypoint_bc/best.pt")
    parser.add_argument("--mode", choices=("teacher", "student"), required=True)
    parser.add_argument("--matches", type=int, default=4)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--duration", type=int, default=420)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", default="rm_runs/waypoint_eval.json")
    args = parser.parse_args()
    cfg = {"app_root": args.app_root, "teacher": args.teacher, "baseline": args.baseline,
           "opponent": args.opponent, "ckpt": args.ckpt, "mode": args.mode,
           "duration": args.duration}
    seeds = list(range(args.seed_start, args.seed_start + args.matches))
    with Pool(processes=args.workers, initializer=_worker_init, initargs=(cfg,), maxtasksperchild=1) as pool:
        results = pool.map(_run_one, seeds, chunksize=1)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    for row in results:
        if "error" in row:
            print(row)
            continue
        red = row["teams"].get("red", {})
        blue = row["teams"].get("blue", {})
        print(json.dumps({"seed": row["seed"], "mode": row["mode"], "winner": row["winner"],
                          "control_red": red.get("control"), "control_blue": blue.get("control"),
                          "damage_red": red.get("attack_damage"), "damage_blue": blue.get("attack_damage")},
                         ensure_ascii=False))


if __name__ == "__main__":
    main()
