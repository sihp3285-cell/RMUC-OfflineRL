"""Evaluate a policy's target head against BattleScope's attributed victims.

This is the metric the geometric label cannot provide: on held-out games, at
each firing second where BattleScope has a high-confidence attribution, did the
policy name the robot that was actually hit?

The dataset shard must contain ``ep_agent_type`` (built after 2026-10-24).
Run on ``--split val`` for a held-out number; ``--split train`` is in-sample.

Usage:
    python scripts/eval_target_attr.py \
        --data data/infantry_tactical_cone20 \
        --run rm_runs/infantry_iql_tactical_cone20 \
        --evidence data/target_evidence.csv --split val
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rm_rl.algos.action_spec import get_spec
from rm_rl.data import schema as S
from rm_rl.data.dataset import Normalizer
from rm_rl.deploy import load_policy

CAMP = {"红": 0, "蓝": 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--split", choices=("train", "val"), default="val")
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit-games", type=int, default=0)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    z = np.load(os.path.join(args.data, f"{args.split}.npz"))
    if "ep_agent_type" not in z.files:
        raise SystemExit(
            "this shard predates ep_agent_type; rebuild the dataset with the "
            "current rm_rl/data/build_dataset.py")
    obs = z["obs"].astype(np.float32)
    act = z["act"].astype(np.float32)
    valid = z["valid"] > 0
    ep_start = z["ep_start"]
    ep_len = z["ep_len"]
    ep_game = z["ep_game_id"]
    ep_camp = z["ep_camp"]
    ep_agent = z["ep_agent_type"].astype(str)

    # (game, camp, agent) -> (start, length)
    episode = {}
    for i in range(len(ep_start)):
        episode[(int(ep_game[i]), int(ep_camp[i]), ep_agent[i])] = (
            int(ep_start[i]), int(ep_len[i]))

    ev = pd.read_csv(args.evidence)
    ev = ev[ev["attacker_type"].isin([S.TYPE_INFANTRY3, S.TYPE_INFANTRY4])]
    ev = ev[ev["victim_type"].isin(S.MOBILE_TYPES)]
    ev = ev[ev["victim_type"] != S.TYPE_AERIAL]
    ev = ev[ev["confidence"] == "high"]
    if args.limit_games > 0:
        keep = sorted(ev["game_id"].unique())[: args.limit_games]
        ev = ev[ev["game_id"].isin(keep)]
    ev = (ev.sort_values(["game_id", "t", "camp", "robot_id", "score"],
                         ascending=[True, True, True, True, False])
            .drop_duplicates(["game_id", "t", "camp", "robot_id"], keep="first"))
    print(f"evidence rows after filter: {len(ev)}")

    model, norm, info = load_policy(args.run, device=device)
    spec = get_spec(info.get("action_mode", "tactical"))
    with torch.no_grad():
        o = norm.normalize(torch.as_tensor(obs, device=device))
        pred = model.act(o, deterministic=True).cpu().numpy()
    tgt_pred = pred[:, spec.sl_target]
    tgt_label = act[:, spec.sl_target]

    n = 0
    hit = 0
    label_hit = 0
    model_label_hit = 0
    in_cone = 0
    multi = 0
    hit_multi = 0
    n_multi = 0
    hit_single = 0
    n_single = 0
    for r in ev.itertuples(index=False):
        key = (int(r.game_id), CAMP.get(r.camp, -1), str(r.attacker_type))
        ep = episode.get(key)
        if ep is None:
            continue
        start, length = ep
        local = int(r.t) - 1
        if local < 0 or local >= length:
            continue
        gi = start + local
        if gi >= len(obs) or not valid[gi]:
            continue
        slot = S.MOBILE_TYPES.index(r.victim_type)
        n += 1
        hit += int(int(tgt_pred[gi].argmax()) == slot)
        label_hit += int(int(tgt_label[gi].argmax()) == slot)
        model_label_hit += int(int(tgt_pred[gi].argmax()) == int(tgt_label[gi].argmax()))
        cone_n = int((tgt_label[gi] > 0).sum())
        in_cone += int(float(tgt_label[gi, slot]) > 0)
        if cone_n >= 2:
            multi += 1
            n_multi += 1
            hit_multi += int(int(tgt_pred[gi].argmax()) == slot)
        else:
            n_single += 1
            hit_single += int(int(tgt_pred[gi].argmax()) == slot)
    den = max(n, 1)
    print(f"split={args.split}  rows={n}")
    print(f"policy top1 vs victim : {hit / den:.4f}")
    print(f"label  top1 vs victim : {label_hit / den:.4f}  (ceiling of a perfect imitator)")
    print(f"policy top1 vs label  : {model_label_hit / den:.4f}  (how well the head fits its own label)")
    print(f"victim in cone        : {in_cone / den:.4f}")
    print(f"multi-target seconds  : {multi / den:.4f}")
    if n_single:
        print(f"policy top1 (single)  : {hit_single / n_single:.4f}")
    if n_multi:
        print(f"policy top1 (multi)   : {hit_multi / n_multi:.4f}")


if __name__ == "__main__":
    main()
