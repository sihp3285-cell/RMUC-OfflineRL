"""Sanity-check a built dataset before committing GPU hours to it.

Catches the failure modes that are expensive to discover after a 100k-step run:
a target distribution that collapsed onto "no target", a fire gate that is
always 0, an observation column that is constant (a feature that silently did
not get populated), NaNs, or a dimension mismatch against the code.

    python -m rm_rl.data.check_dataset --data data/sentry_tactical
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from . import schema as S
from .features import obs_dim, obs_feature_names, is_capability_feature
from .obs_v2 import obs_dim_v2, obs_feature_names_v2
from ..algos.action_spec import NO_TARGET, get_spec


def check(data_dir: str) -> int:
    with open(os.path.join(data_dir, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    z = np.load(os.path.join(data_dir, "train.npz"))
    obs, act, valid = z["obs"], z["act"], z["valid"] > 0
    mode = meta.get("action_mode", "velocity")
    spec = get_spec(mode, act.shape[1])
    problems = []
    obs_version = int(meta.get("obs_version", 1))
    agent = ((meta.get("agent_types") or [meta.get("agent_type", S.TYPE_SENTRY)])[0]
             .split(",")[0])
    expected_dim = obs_dim_v2(agent) if obs_version == 2 else obs_dim(agent)
    expected_names = (obs_feature_names_v2(agent) if obs_version == 2
                      else obs_feature_names(agent))

    print(f"=== dataset {data_dir} ===")
    print(f"agent={meta.get('agent_type')}  action_mode={mode}  obs_version={obs_version}")
    print(f"transitions: train={len(obs)}  val={meta.get('n_val_transitions')}")
    print(f"obs_dim={obs.shape[1]} (code expects {expected_dim})"
          f"   act_dim={act.shape[1]} (spec {spec.dim})")
    print(f"vis_map={meta.get('vis_map')}  team_prior={meta.get('team_prior')}")

    if obs.shape[1] != expected_dim:
        problems.append(f"obs_dim {obs.shape[1]} disagrees with features ({expected_dim})")
    if act.shape[1] != spec.dim:
        problems.append("act_dim disagrees with the action spec")
    if not np.isfinite(obs).all():
        problems.append("observation contains NaN/Inf")
    if not np.isfinite(act).all():
        problems.append("action contains NaN/Inf")

    # Constant columns are expected for capability axes a unit type cannot
    # upgrade (the sentry never levels its HP, infantry carry no 42mm barrel).
    # Only a constant column *outside* that set indicates a wiring bug.
    names = (expected_names if obs.shape[1] == expected_dim
             else [f"obs[{i}]" for i in range(obs.shape[1])])
    placeholders = set(meta.get("extra_placeholder", []))
    def known_constant(nm: str) -> bool:
        if is_capability_feature(nm):
            return True
        if nm in placeholders:
            return True
        # E13 slot 3 is "other buff"; most games never populate it.
        if nm == "ego.buff[3]":
            return True
        # 空中 is a UAV and is never flagged 易伤 by the enemy radar.
        if nm == f"enemy[{S.TYPE_AERIAL}].vuln":
            return True
        # Bases are destroyed in 1 of 1226 instances across the whole season, so
        # this flag is constant in any subsample. See README "数据集".
        if nm.endswith("base_alive"):
            return True
        return False

    const = np.where(obs.std(0) < 1e-8)[0]
    benign = [i for i in const if known_constant(names[i])]
    suspect = [i for i in const if not known_constant(names[i])]
    if len(const):
        print(f"\nconstant columns: {len(const)} "
              f"({len(benign)} expected by data properties, {len(suspect)} other)")
        for i in suspect:
            print(f"    [warn] {names[i]}")
    if suspect:
        problems.append(f"{len(suspect)} non-capability observation columns are "
                        f"constant: {[names[i] for i in suspect]}")


    v = act[valid]
    print(f"\nvalid (alive) transitions: {valid.sum()} / {len(valid)} "
          f"({valid.mean():.1%})")
    if spec.is_tactical:
        nav = v[:, spec.sl_cont]
        gate = v[:, spec.sl_fire].squeeze(-1)
        tgt = v[:, spec.sl_target]
        print(f"nav      |mean|={np.abs(nav).mean(0).round(3).tolist()}  "
              f"p99={np.percentile(np.abs(nav), 99, axis=0).round(3).tolist()}")
        print(f"fire gate: on {gate.mean():.1%} of live seconds")
        share = tgt.mean(0)
        names = list(S.MOBILE_TYPES) + ["<no target>"]
        print("target distribution (mean probability mass):")
        for nm, p in zip(names, share):
            print(f"    {nm:<10} {p:.3f}")
        hard = tgt.argmax(1)
        named = (hard != NO_TARGET).mean()
        print(f"a concrete enemy is the argmax on {named:.1%} of live seconds")

        if gate.mean() < 0.005:
            problems.append("fire gate is on <0.5% of the time — check 累计17mm发弹")
        if named < 0.02:
            problems.append("target labels are almost entirely '<no target>' — "
                            "check 枪口朝向 sentinel handling / cone width")
        if float(np.abs(nav).mean()) < 1e-4:
            problems.append("nav action is ~zero everywhere")
    else:
        print(f"action |mean|={np.abs(v).mean(0).round(3).tolist()}")

    print(f"\nreward: mean={z['rew'].mean():.4f} std={z['rew'].std():.4f} "
          f"min={z['rew'].min():.2f} max={z['rew'].max():.2f}")
    print(f"episodes: {len(z['ep_len'])}  mean length {z['ep_len'].mean():.0f}s")

    if problems:
        print("\n*** PROBLEMS ***")
        for p in problems:
            print("  - " + p)
        return 1
    print("\nOK — dataset looks sane.")
    return 0


def main():
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    args = ap.parse_args()
    raise SystemExit(check(args.data))


if __name__ == "__main__":
    main()
