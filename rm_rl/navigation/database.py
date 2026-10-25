"""Read-only SQLite -> causal radio-shaped snapshots and separate weak labels."""
from __future__ import annotations

import math
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np

from .contract import IDS, VERSION, finite, side


def connect(path):
    con = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def matches(con):
    return [dict(r) for r in con.execute('SELECT * FROM matches ORDER BY "赛区","场次号","局号"')]


def load_game(con, match):
    """No rounding future rows backwards, no interpolation, no cross-match cache.

    `teacher` contains yaw and shoot events; it never enters FeatureEncoder.
    Packet tick t includes source records whose timestamps are <= t.
    """
    rows = con.execute('SELECT * FROM timeseries WHERE game_id=? ORDER BY "时刻秒",robot_id',
                       (match["game_id"],)).fetchall()
    events = con.execute('SELECT * FROM events WHERE game_id=? ORDER BY "时刻秒"',
                         (match["game_id"],)).fetchall()
    if not rows:
        return [], []
    stop = math.ceil(max(r["时刻秒"] for r in rows))
    if stop > 1800 or stop < 0:
        raise ValueError("unexpected match time range")
    by_tick, event_ticks = defaultdict(list), defaultdict(list)
    for r in rows:
        if r["robot_id"] in IDS and r["时刻秒"] >= 0:
            by_tick[math.ceil(r["时刻秒"])].append(r)
    for ev in events:
        if finite(ev["时刻秒"]) and ev["时刻秒"] >= 0:
            event_ticks[math.ceil(ev["时刻秒"])].append(dict(ev))
    last, economy, telemetry = {}, {}, {}
    packets, teachers = [], []
    for t in range(stop + 1):
        for r in by_tick[t]:
            rid, at = int(r["robot_id"]), float(r["时刻秒"])
            robot = last.setdefault(rid, {"robot_id": rid})
            for name, col in (("hp", "当前血量"), ("maxhp", "最大血量")):
                if finite(r[col]):
                    robot[name], robot[name + "_time_s"] = float(r[col]), at
            if finite(r["x"]) and finite(r["y"]):
                robot["position"], robot["position_time_s"] = [r["x"], r["y"]], at
            econ = economy.setdefault(side(rid), {})
            for name, col in (("remaining", "队伍剩余金币"), ("total", "队伍总金币")):
                if finite(r[col]):
                    econ[name], econ[name + "_time_s"] = float(r[col]), at
            telemetry[rid] = dict(x=r["x"], y=r["y"], yaw=r["枪口朝向"], at=at)
        # Fresh independent dictionaries: later records cannot rewrite history.
        packets.append(dict(schema_version=VERSION, match_id=str(match["game_id"]), time_s=float(t),
                            robots=[dict(v) for v in last.values()],
                            economy={s: dict(e) for s, e in economy.items()}))
        teachers.append(dict(poses={k: dict(v) for k,v in telemetry.items()}, events=event_ticks[t]))
    return packets, teachers


def attack_labels(teachers):
    """Heuristic, positive-only attacker labels with explicit unknowns.

    Inspired by battlescope's recent-shot/heading evidence, independently
    implemented on causal raw positions. No deployment/fortress rule guesses.
    Confidence weights are heuristic weights, NOT calibrated probabilities.
    Teacher-only: radio does not have to supply enemy yaw/shoot events.
    """
    output, shots = [], []
    for t, frame in enumerate(teachers):
        shots = [s for s in shots if t - s[0] <= 4]
        for ev in frame["events"]:
            rid, at = ev["robot_id"], ev["时刻秒"]
            caliber = ev["类别"]
            if ev["事件类型"] != "发弹" or caliber not in ("17mm", "42mm") or rid not in IDS:
                continue
            # floor selects a pose available at the shot time, never its future.
            pose = teachers[min(int(at), t)]["poses"].get(rid)
            if pose and all(finite(pose[k]) for k in ("x", "y", "yaw")):
                if 0 <= at - pose["at"] <= 1 and abs(pose["yaw"] + 140) > 1e-6:
                    shots.append((at, rid, caliber, pose))
        labels = {}
        for ev in frame["events"]:
            victim, at, caliber = ev["robot_id"], ev["时刻秒"], ev["类别"]
            if ev["事件类型"] != "受击" or caliber not in ("17mm", "42mm") or victim not in IDS:
                continue
            pose = teachers[min(int(at), t)]["poses"].get(victim)
            if not pose or not all(finite(pose[k]) for k in ("x", "y")) or not 0 <= at-pose["at"] <= 1:
                continue
            scores = {}
            for st, rid, cal, sp in shots:
                delay = at - st
                if cal != caliber or side(rid) == side(victim) or not 0 <= delay <= (2 if cal == "17mm" else 3):
                    continue
                dx, dy = pose["x"]-sp["x"], pose["y"]-sp["y"]
                distance = math.hypot(dx, dy)
                if distance < .1 or distance > (25 if cal == "17mm" else 16.5)*(delay+1)+3:
                    continue
                angle = abs((math.degrees(math.atan2(dy, dx))-sp["yaw"]+180)%360-180)
                if angle > 25:
                    continue
                score = math.exp(-(angle/20)**2 - delay/4)
                scores[rid] = max(score, scores.get(rid, 0))
            ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
            if not ranked:
                continue
            rid, score = ranked[0]
            margin = score - (ranked[1][1] if len(ranked)>1 else 0)
            if score >= .55 and margin >= .20:
                # If two hits disagree this tick, mark victim ambiguous.
                if victim in labels and labels[victim][0] != rid:
                    labels[victim] = (-1, 0.)
                else:
                    labels[victim] = (rid, min(score, margin * 2))
        output.append(labels)
    return output
