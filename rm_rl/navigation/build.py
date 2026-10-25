"""Build navigation option transitions; future data only supplies targets/reward."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from .contract import SPEC, HOLD, ACTION_COUNT, FeatureEncoder, canonical, cell, fingerprint, side
from .database import connect, matches, load_game, attack_labels


def split_for(match, seed):
    # All games in one series stay together. Camp/ego copies share this split.
    key = f'{seed}:{match["赛区"]}:{match["场次号"]}'
    bucket = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 100
    return "test" if bucket < 10 else "val" if bucket < 20 else "train"


def hp_map(packet):
    t = packet["time_s"]
    return {r["robot_id"]: r["hp"] for r in packet["robots"]
            if "hp" in r and 0 <= t-r["hp_time_s"] <= 1.01}


def rewards(packets, camp, ego, winner, cfg, completed):
    """Factual team outcomes, never reward inferred 'support' or attack labels."""
    reward = np.zeros(len(packets)-1, np.float32)
    for t in range(len(reward)):
        prev, nxt = hp_map(packets[t]), hp_map(packets[t+1])
        for rid in prev.keys() & nxt.keys():
            loss = max(0, prev[rid]-nxt[rid])
            own, role = side(rid) == camp, rid % 100
            weight = cfg["base"] if role == 10 else cfg["outpost"] if role == 11 else cfg["mobile"]
            scale = 5000 if role == 10 else 1500 if role == 11 else 400
            reward[t] += (-1 if own else 1) * weight * loss / scale
            if role not in (10, 11) and prev[rid] > 0 and nxt[rid] == 0:
                reward[t] += (-1 if own else 1) * cfg["death"]
        if ego in prev and ego in nxt and prev[ego] > 0 and nxt[ego] == 0:
            reward[t] -= cfg["ego_death"]
    if completed and winner in ("红", "蓝") and len(reward):
        reward[-1] += cfg["win"] * (1 if (winner == "红") == (camp == "red") else -1)
    return reward


def segment(obs, start, cfg):
    """First sustained stop, otherwise a bounded transit waypoint.

    A stop is an observed dwell, not a human-intent label. Death-ending and
    non-stopping movements are retained, so successes are not the only samples.
    Unknown position intervals are excluded rather than interpolated.
    """
    limit = min(start + cfg["max_seconds"], len(obs)-1)
    p = obs[start].ego_xy
    if limit <= start or p is None or not obs[start].ego_alive:
        return None
    dwell = cfg["dwell_seconds"]
    for end in range(start+1, limit+1):
        if obs[end].ego_xy is None:
            return None
        if np.linalg.norm(obs[end].ego_xy - obs[end-1].ego_xy) > 8.01:
            return None
        if not obs[end].ego_alive:
            # Location just before death; do not use a respawn/disappearance jump.
            goal = obs[end-1].ego_xy
            act = HOLD if np.linalg.norm(goal-p) < cfg["min_move_m"] else cell(goal)
            return end, act, .6, True
        if end-start >= dwell:
            window = np.stack([o.ego_xy for o in obs[end-dwell:end+1]])
            if np.max(np.linalg.norm(window-window[-1], axis=1)) <= cfg["dwell_radius_m"]:
                goal = window[-1]
                if np.linalg.norm(goal-p) < cfg["min_move_m"]:
                    # HOLD only if the entire option actually stayed nearby.
                    if max(np.linalg.norm(o.ego_xy-p) for o in obs[start:end+1]) > cfg["dwell_radius_m"]:
                        continue
                    return end, HOLD, 1., False
                return end, cell(goal), 1., False
    goal = obs[limit].ego_xy
    if np.linalg.norm(goal-p) < cfg["min_move_m"]:
        # A loop back to the start is not evidence of stationary HOLD.
        if max(np.linalg.norm(o.ego_xy-p) for o in obs[start:limit+1]) > cfg["dwell_radius_m"]:
            return None
        return limit, HOLD, .5, False
    return limit, cell(goal), .6, False


def game_samples(match, packets, teachers, cfg):
    arrays = {k: [] for k in ("entities", "global", "next_entities", "next_global", "action",
                              "reward", "discount", "weight", "attack", "attack_weight",
                              "future_loss", "future_mask", "ego", "time", "duration")}
    attacks = attack_labels(teachers)
    completed = bool(packets and packets[-1]["time_s"] >= float(match["时长秒"] or 420)-1)
    for ego in (3, 4, 103, 104):
        camp = side(ego)
        enc = FeatureEncoder()
        observations = [enc.encode(dict(p, ego_id=ego, own_camp=camp)) for p in packets]
        rs = rewards(packets, camp, ego, match["胜方"], cfg["reward"], completed)
        start = 0
        while start < len(packets)-1:
            seg = segment(observations, start, cfg["options"])
            if seg is None:
                start += 1
                continue
            end, action, weight, dead = seg
            o, no = observations[start], observations[end]
            label = np.full(16, -1, np.int64)
            label_weight = np.zeros(16, np.float32)
            for victim, (attacker, w) in attacks[start].items():
                if attacker in o.ids and victim in o.ids:
                    label[o.ids.index(victim)], label_weight[o.ids.index(victim)] = o.ids.index(attacker), w
            # Auxiliary 10-second damage target; no observed future => no label.
            future = min(start+10, len(packets)-1)
            loss, mask = np.zeros(16, np.float32), np.zeros(16, np.float32)
            hps = [hp_map(p) for p in packets[start:future+1]]
            for j, rid in enumerate(o.ids):
                if future-start == 10 and all(rid in h for h in hps):
                    scale = 5000 if rid%100 == 10 else 1500 if rid%100 == 11 else 400
                    loss[j] = sum(max(0, a[rid]-b[rid]) for a,b in zip(hps, hps[1:])) / scale
                    mask[j] = 1
            terminal = dead or end == len(packets)-1
            dt = end-start
            record = dict(entities=o.entities, global_=o.global_, next_entities=no.entities,
                          next_global=no.global_, action=action,
                          reward=sum(cfg["iql"]["gamma"]**j * float(rs[start+j]) for j in range(dt)),
                          discount=0. if terminal else cfg["iql"]["gamma"]**dt,
                          weight=weight, attack=label, attack_weight=label_weight,
                          future_loss=loss, future_mask=mask, ego=ego, time=start, duration=dt)
            record["global"] = record.pop("global_")
            for k in arrays:
                arrays[k].append(record[k])
            start = end
    if not arrays["action"]:
        return None
    result = {k: np.asarray(v, dtype=np.int64 if k in ("action", "attack", "ego", "time", "duration")
                            else np.float32) for k,v in arrays.items()}
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/navigation.yaml")
    ap.add_argument("--db", default="dataset/rmuc_2026_region_dataset.sqlite")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    out = Path(args.out or cfg["data"])
    if out.exists() and any(out.iterdir()):
        raise SystemExit("Output must be empty; use a new --out to avoid stale/mixed shards")
    out.mkdir(parents=True, exist_ok=True)
    con = connect(args.db)
    records, support, audit = [], np.zeros(ACTION_COUNT, np.int64), {}
    for m in matches(con):
        packets, teachers = load_game(con, m)
        samples = game_samples(m, packets, teachers, cfg) if packets else None
        if samples is None:
            continue
        split = split_for(m, cfg["seed"])
        name = f'{split}/{m["game_id"]}.npz'
        (out/split).mkdir(exist_ok=True)
        np.savez_compressed(out/name, **samples)
        if split == "train":
            support += np.bincount(samples["action"], minlength=ACTION_COUNT)
        n = len(samples["action"])
        audit[split] = audit.get(split, 0) + n
        records.append(dict(file=name, split=split, game_id=m["game_id"], region=m["赛区"],
                            series=m["场次号"], count=n,
                            hold=int((samples["action"] == HOLD).sum()),
                            weak_attack_labels=int((samples["attack_weight"] > 0).sum())))
        print(f'{m["game_id"]} {split}: {n} options', flush=True)
    con.close()
    if not all(audit.get(s, 0) for s in ("train", "val", "test")):
        raise SystemExit("Missing train/val/test samples; inspect source and split before training")
    manifest = dict(schema=SPEC, schema_hash=fingerprint(), config=cfg, shards=records,
                    support=support.tolist(), counts=audit,
                    source_db=str(Path(args.db).resolve()),
                    unavailable_inputs=["remaining_ammo", "current_buffs", "buff_point_state"],
                    teacher_only=["yaw", "shoot_events", "attack_weak_label", "future_hp_loss"],
                    split="SHA256(seed:region:series), 80/10/10, no series overlap",
                    quality_note="waypoints are reconstructed behavior, not verified tactical intent")
    (out/"manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(audit))


if __name__ == "__main__":
    main()
