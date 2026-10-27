"""Run a real Arena strategy.py against the local engine with a full team."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

_CFG = None
_MATCH = None
_CAND = None
_BASE = None
_OPP = None


def _load(path, sdk_dir):
    if sdk_dir not in sys.path:
        sys.path.insert(0, sdk_dir)
    spec = importlib.util.spec_from_file_location("candidate_strategy", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _init(cfg):
    global _CFG, _MATCH, _CAND, _BASE, _OPP
    _CFG = cfg
    app = str(cfg["app"])
    sdk = str(Path(app) / "rmuc_arena")
    if app not in sys.path:
        sys.path.insert(0, app)
    if sdk not in sys.path:
        sys.path.insert(0, sdk)
    from rmuc_arena.engine import Match
    from rmuc_arena.models import mirrored_lineups
    _MATCH = Match
    _CAND = _load(cfg["strategy"], sdk)
    _BASE = _load(cfg["baseline"], sdk)
    _OPP = _load(cfg["opponent"], sdk)
    _CFG["lineups"] = mirrored_lineups({"1": 2711, "3": 2731, "4": 2732, "6": 2761, "7": 2771}, "2027")


def _sentry_metrics(hist):
    if not hist:
        return {}
    dt = 0.1
    alive = sum(dt for row in hist if row["alive"])
    deaths = sum(1 for a, b in zip(hist, hist[1:]) if a["alive"] and not b["alive"])
    taken = 0.0
    for a, b in zip(hist, hist[1:]):
        if a["alive"] and b["alive"] and b["hp"] < a["hp"]:
            taken += a["hp"] - b["hp"]
    central = [(10.8, 7.5)]
    card = [(11.13, 10.64), (11.71, 12.08)]
    supply = [(1.4, 13.0)]
    def near(points, radius=1.6):
        return sum(dt for row in hist if row["alive"] and any(
            ((row["x"] - x) ** 2 + (row["y"] - y) ** 2) ** 0.5 <= radius for x, y in points))
    travel = sum(((b["x"] - a["x"]) ** 2 + (b["y"] - a["y"]) ** 2) ** 0.5
                 for a, b in zip(hist, hist[1:]) if a["alive"] and b["alive"])
    stuck = sum(dt for row in hist if row["alive"] and (row["vx"] ** 2 + row["vy"] ** 2) ** 0.5 < 0.08)
    dist_central = [((row["x"] - 10.8) ** 2 + (row["y"] - 7.5) ** 2) ** 0.5
                    for row in hist if row["alive"]]
    goals = [row["goal"] for row in hist]
    switches = 0
    segments = []
    current = None
    for row in hist:
        goal = row["goal"]
        if goal is None:
            continue
        if current is None or ((goal[0] - current[0]) ** 2 + (goal[1] - current[1]) ** 2) ** 0.5 > 0.5:
            switches += 1
            current = (float(goal[0]), float(goal[1]))
            segments.append({"goal": current, "hit": False})
        if segments:
            gx, gy = segments[-1]["goal"]
            if ((row["x"] - gx) ** 2 + (row["y"] - gy) ** 2) ** 0.5 <= 0.8:
                segments[-1]["hit"] = True
    arrival = (sum(1 for seg in segments if seg["hit"]) / len(segments)) if segments else None
    point_errors = []
    for a, b in zip(hist, hist[1:]):
        if a["goal"] is not None and b["alive"]:
            gx, gy = a["goal"]
            point_errors.append(((b["x"] - gx) ** 2 + (b["y"] - gy) ** 2) ** 0.5)
    stuck = 0.0
    for row in hist:
        if row["alive"] and row["goal"] is not None:
            gx, gy = row["goal"]
            if ((row["x"] - gx) ** 2 + (row["y"] - gy) ** 2) ** 0.5 > 1.0 and (row["vx"] ** 2 + row["vy"] ** 2) ** 0.5 < 0.1:
                stuck += dt
    exposure = sum(dt for row in hist if row["alive"] and row["enemy_dist"] < 8.0)
    return {
        "alive_time_s": alive,
        "deaths": deaths,
        "damage_dealt": max((row["damage"] for row in hist), default=0.0),
        "damage_taken": taken,
        "kills": max((row["kills"] for row in hist), default=0),
        "time_central_s": near(central),
        "time_outpost_card_s": near(card),
        "time_supply_s": near(supply),
        "avg_dist_central_m": sum(dist_central) / len(dist_central) if dist_central else None,
        "travel_m": travel,
        "stuck_s": stuck,
        "ammo_end": hist[-1]["ammo"],
        "overheat_s": sum(dt for row in hist if row["alive"] and row["heat"] >= row["max_heat"] * 0.95),
        "posture_counts": {p: sum(1 for row in hist if row["posture"] == p) for p in set(row["posture"] for row in hist)},
        "goal_segments": len(segments),
        "arrival_rate": arrival,
        "point_error_m": (sum(point_errors) / len(point_errors)) if point_errors else None,
        "goal_switches": switches,
        "stuck_cmd_s": stuck,
        "exposure_time_s": exposure,
    }


def _run(seed):
    try:
        m = _MATCH(seed=int(seed), duration=int(_CFG["duration"]), map_name="rmuc2027", lineups=_CFG["lineups"])
        cs = bs = os_ = None
        lat, errs, goals = [], [], []
        sentry_hist = []
        while m.status != "finished":
            frame = m.snapshot()
            sentry = next((r for r in frame.get("robots", []) if r.get("id") == "red-7"), None)
            if sentry is not None:
                sentry_hist.append({
                    "t": float(frame.get("time", 0.0)),
                    "x": float(sentry.get("x", 0.0)), "y": float(sentry.get("y", 0.0)),
                    "hp": float(sentry.get("hp", 0.0)), "max_hp": float(sentry.get("max_hp", 400.0)),
                    "ammo": float(sentry.get("ammo", 0.0)), "heat": float(sentry.get("heat", 0.0)),
                    "max_heat": float(sentry.get("max_heat", 1.0)), "alive": bool(sentry.get("alive")),
                    "damage": float(sentry.get("damage", 0.0)), "kills": int(sentry.get("kills", 0)),
                    "vx": float(sentry.get("vx", 0.0)), "vy": float(sentry.get("vy", 0.0)),
                    "posture": str(sentry.get("posture", "")),
                    "goal": getattr(_CAND, "_WP_CURRENT_GOAL", None),
                    "enemy_dist": min([((float(sentry.get("x", 0.0)) - float(e.get("x", 0.0))) ** 2 +
                                        (float(sentry.get("y", 0.0)) - float(e.get("y", 0.0))) ** 2) ** 0.5
                                       for e in frame.get("robots", [])
                                       if e.get("team") == "blue" and e.get("alive")] or [99.0]),
                })
            red_team = m.observation("red", [1, 3, 4, 6])
            red_actions, bs = _BASE.act(red_team, bs)
            obs = m.observation("red", [7])
            t0 = time.perf_counter()
            try:
                acts, cs = _CAND.act(obs, cs)
                err = None
            except Exception as exc:
                acts, err = {}, str(exc)
            lat.append((time.perf_counter() - t0) * 1000.0)
            if err:
                errs.append(err)
            goal = getattr(_CAND, "_WP_CURRENT_GOAL", None)
            if goal is not None:
                goals.append([float(goal[0]), float(goal[1])])
            blue_obs = m.observation("blue", [1, 3, 4, 6, 7])
            blue_actions, os_ = _OPP.act(blue_obs, os_)
            red_cmd = {k: v for k, v in (red_actions or {}).items() if str(k) in ("1", "3", "4", "6")}
            red_cmd.update({str(k): v for k, v in (acts or {}).items() if str(k) == "7"})
            blue_cmd = {str(k): v for k, v in (blue_actions or {}).items() if str(k) in ("1", "3", "4", "6", "7")}
            for _ in range(2):
                m.step({"red": red_cmd, "blue": blue_cmd})
                if m.status == "finished":
                    break
        frame = m.snapshot()
        lat.sort()
        sentry_metrics = _sentry_metrics(sentry_hist)
        result = {
            "seed": int(seed), "winner": m.winner, "reason": m.reason, "time": float(m.time),
            "teams": frame.get("teams", {}), "bases": frame.get("bases", {}),
            "decision_p50_ms": lat[len(lat) // 2] if lat else None,
            "decision_p99_ms": lat[int(len(lat) * .99)] if lat else None,
            "decision_max_ms": max(lat) if lat else None,
            "errors": errs[:10], "error_count": len(errs),
            "goals": goals[:5], "unique_goals": len({(round(x, 2), round(y, 2)) for x, y in goals}),
            "sentry": sentry_metrics,
        }
        m.close()
        return result
    except Exception:
        traceback.print_exc()
        return {"seed": int(seed), "error": "worker failed"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--app", default="/home/delphine/rm/arena-runtime/squashfs-root/resources/runtime/app")
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--baseline", default="/home/delphine/rm/arena-runtime/squashfs-root/resources/runtime/app/examples/claude_opus55_xhigh_2027_hold/strategy.py")
    parser.add_argument("--opponent", default="/home/delphine/rm/arena-runtime/squashfs-root/resources/runtime/app/examples/claude_opus55_xhigh_2027_hold/strategy.py")
    parser.add_argument("--matches", type=int, default=2)
    parser.add_argument("--seed-start", type=int, default=2000)
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--out", default="/tmp/strategy_test.json")
    args = parser.parse_args()
    cfg = {"app": args.app, "strategy": args.strategy, "baseline": args.baseline,
           "opponent": args.opponent, "duration": args.duration}
    seeds = list(range(args.seed_start, args.seed_start + args.matches))
    with Pool(processes=args.workers, initializer=_init, initargs=(cfg,), maxtasksperchild=1) as pool:
        results = pool.map(_run, seeds, chunksize=1)
    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    for row in results:
        print(json.dumps({k: row.get(k) for k in ("seed", "winner", "reason", "decision_p50_ms",
                                                  "decision_p99_ms", "decision_max_ms", "error_count",
                                                  "unique_goals")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
