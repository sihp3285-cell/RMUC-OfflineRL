"""Dump one match — and the policy's opinion of it — to a JSON the viewer reads.

The referee log is already a complete replay: every robot's position, health,
muzzle heading and ammunition, once per second.  What it does not contain is
what a *policy* would have done at each of those seconds.  This script produces
both, time-aligned, so ``replay.html`` can draw the human's actual play and the
model's proposed decision on the same frame.

Two frames of reference are in play and mixing them up silently rotates half the
arrows by 180 degrees:

  * the **raw** frame, which is what the referee system logs and what we render;
  * the **canonical** frame the policy thinks in, where blue-side games are
    mirrored about the field centre so one policy covers both sides.

Everything written to the JSON is in the raw frame.  ``build_obs`` mirrors on
the way in and ``decode_action`` mirrors back on the way out, so the only place
this file has to intervene is the human ground-truth action, which comes out of
``build_action_raw`` still canonicalised.

Usage::

    # which games are worth watching?
    python -m viz.export_replay --db dataset/....sqlite --list

    # export one, with the policy's decisions for every human-driven slot
    python -m viz.export_replay --db dataset/....sqlite \\
        --game-id 1780384424933 --run rm_runs/infantry_iql_tactical \\
        --vis-map data/vis_map.npz --team-prior data/team_prior.json \\
        --out viz/replays/game.json
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys

import numpy as np
import torch

from rm_rl.data import schema as S
from rm_rl.data.build_dataset import (load_game_arrays, load_matches,
                                      load_target_evidence)
from rm_rl.data.features import build_obs, build_action_raw
from rm_rl.data.team_prior import TeamPrior
from rm_rl.data.vis_map import VisibilityMap
from rm_rl.deploy import load_policy
from rm_rl.algos.action_spec import AERIAL_TARGET_SLOT, NO_TARGET

# Slots worth asking the policy about.  The engineer has no turret and the
# aerial unit is not a ground tactical agent, so neither gets a decision track.
DEFAULT_EGOS = [S.TYPE_INFANTRY3, S.TYPE_INFANTRY4, S.TYPE_SENTRY, S.TYPE_HERO]


def _r(a, nd=2):
    """Round for transport — the JSON is ~4x smaller and nothing visible moves.

    Non-finite values become ``null``.  Python's json writes a bare ``NaN``,
    which is valid JavaScript but *not* valid JSON, so `JSON.parse` rejects the
    whole file — one stationary second is enough to blank the entire viewer.
    """
    v = np.round(np.asarray(a, np.float64), nd)
    return [None if not np.isfinite(x) else x for x in np.atleast_1d(v)]


def list_games(db, n=15):
    con = sqlite3.connect(db)
    cur = con.cursor()
    rows = cur.execute(
        'SELECT e.game_id, COUNT(*) AS shots, m."时长秒", m."红方学校", '
        'm."蓝方学校", m."胜方", m."赛区" '
        f'FROM {S.T_EVENTS} e JOIN {S.T_MATCHES} m ON m.game_id = e.game_id '
        f'WHERE e."事件类型" = ? GROUP BY e.game_id '
        'ORDER BY shots DESC LIMIT ?', (S.EV_SHOOT, int(n))).fetchall()
    con.close()
    print(f"{'game_id':>16}{'shots':>8}{'dur':>6}  winner  red vs blue")
    for gid, shots, dur, red, blue, win, region in rows:
        print(f"{gid:>16}{shots:>8}{dur:>6}  {win:^6}  {red} vs {blue}   [{region}]")
    return [r[0] for r in rows]


# ---------------------------------------------------------------------------
def entity_track(game, rtype, camp):
    """One robot's full-game track, in the RAW (unmirrored) render frame."""
    e = game.get(rtype, camp)
    rid = S.robot_id(rtype, camp)
    if rid not in game.ent:
        return None
    # A robot the referee never tracked sits at the origin for the whole game;
    # drawing it would put a permanent ghost in the corner of the field.
    known = (np.abs(e.x) > 1e-6) | (np.abs(e.y) > 1e-6)
    return dict(
        id=int(rid), type=rtype, camp=camp,
        x=_r(e.x), y=_r(e.y),
        hp=_r(e.hp, 0), maxhp=float(np.max(e.maxhp)) or 1.0,
        yaw=_r(e.yaw, 1),
        # the sentinel means "this chassis has no tracked turret", which the
        # viewer must not draw as a heading of -140 degrees
        yaw_ok=[int(v) for v in (~np.isclose(e.yaw, S.YAW_SENTINEL)).astype(int)],
        alive=[int(v) for v in e.alive.astype(int)],
        known=[int(v) for v in known.astype(int)],
        ammo=_r(e.ammo17, 0),
        heat=_r(e.heat17, 0),
        vuln=[int(v) for v in e.vuln.astype(int)],
    )


@torch.no_grad()
def ego_decisions(game, camp, rtype, policy, norm, info, vmap, tprior, gid,
                  goal_horizon, device, alt_map=None,
                  target_cone=15.0, target_tau=8.0):
    """Per-second model decision + human ground truth for one robot.

    Both are returned in the raw frame and in physical units (metres for the
    nav sub-goal), so the viewer can draw them without knowing anything about
    normalisation or canonicalisation.
    """
    tfeat = tprior.feats(gid, camp) if tprior is not None else None
    obs = build_obs(game, camp, rtype, vis_map=vmap, team_feat=tfeat)   # [T, 161]
    human = build_action_raw(game, camp, rtype, action_mode="tactical",
                             goal_horizon=goal_horizon,
                             target_cone=target_cone,
                             target_tau=target_tau)                     # [T-1, 10]

    o = torch.as_tensor(obs, dtype=torch.float32, device=device)
    o = norm.normalize(o)
    # HybridPolicy.act() hard-thresholds the gate; go through the heads directly
    # so the viewer can show *how confident* the model is, not just the bit.
    mean, fire_logit, tgt_logits = policy.policy.heads(o)
    nav = mean.clamp(-1.0, 1.0).cpu().numpy()
    p_fire = torch.sigmoid(fire_logit).cpu().numpy()
    p_tgt = torch.softmax(tgt_logits, dim=-1).cpu().numpy()
    # Safety net for old checkpoints: never let the viewer show aerial as a
    # recommended target, even if the loaded weights predate the mask.
    p_tgt[:, AERIAL_TARGET_SLOT] = 0.0
    p_tgt /= np.maximum(p_tgt.sum(axis=1, keepdims=True), 1e-9)

    scale = np.asarray(info["act_scale"], np.float32)
    gx, gy = nav[:, 0] * scale[0], nav[:, 1] * scale[1]
    hx, hy = human[:, 0], human[:, 1]
    if camp == S.CAMP_BLUE:            # undo the canonicalisation mirror
        gx, gy, hx, hy = -gx, -gy, -hx, -hy

    m_tgt = p_tgt.argmax(1)
    h_tgt = human[:, 3:].argmax(1)
    # `human` is one step shorter than `obs` (an action describes t -> t+1)
    T1 = len(human)
    # BattleScope-attributed victim, only high-confidence / non-ambiguous /
    # mobile.  -1 means "no evidence for this second".
    ev = [-1] * T1
    if alt_map:
        rid = S.robot_id(rtype, camp)
        for tt in range(1, min(int(game.T), T1 + 1)):
            vt = alt_map.get((int(gid), int(tt), camp, int(rid)))
            if vt:
                ev[tt - 1] = S.MOBILE_TYPES.index(vt)
    return dict(
        type=rtype, camp=camp, id=int(S.robot_id(rtype, camp)),
        model=dict(gx=_r(gx), gy=_r(gy),
                   p_fire=_r(p_fire, 3), fire=[int(v) for v in (p_fire > 0.5)],
                   target=[int(v) for v in m_tgt],
                   target_conf=_r(p_tgt.max(1), 3),
                   target_probs=[_r(row, 3) for row in p_tgt]),
        human=dict(gx=_r(hx), gy=_r(hy),
                   fire=[int(v) for v in human[:, 2].astype(int)],
                   target=[int(v) for v in h_tgt],
                   target_conf=_r(human[:, 3:].max(1), 3)),
        evidence=dict(victim=ev),
        agree=dict(
            fire=[int(v) for v in ((p_fire[:T1] > 0.5) == (human[:, 2] > 0.5))],
            target=[int(v) for v in (m_tgt[:T1] == h_tgt)],
            # direction agreement only: a sub-goal's magnitude is how far the
            # robot happened to travel, which is not the decision being made
            nav=_r(_cos(np.stack([gx[:T1], gy[:T1]], 1),
                        np.stack([hx, hy], 1)), 3),
        ),
    )


def _cos(a, b):
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    d = (a * b).sum(1) / np.maximum(na * nb, 1e-6)
    return np.where((na > 0.05) & (nb > 0.05), d, np.nan)


# ---------------------------------------------------------------------------
def main():
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--game-id", type=int, default=0,
                    help="0 = the highest-activity game in the database")
    ap.add_argument("--run", default="rm_runs/infantry_iql_tactical")
    ap.add_argument("--out", default="viz/replays/game.json")
    ap.add_argument("--vis-map", default="data/vis_map.npz")
    ap.add_argument("--team-prior", default="data/team_prior.json")
    ap.add_argument("--goal-horizon", type=int, default=5)
    ap.add_argument("--target-cone", type=float, default=15.0,
                    help="cone of the human target label; use 25 for cone25 runs")
    ap.add_argument("--target-tau", type=float, default=8.0)
    ap.add_argument("--evidence", default="data/target_evidence.csv",
                    help="BattleScope evidence CSV; overlay actual victim")
    ap.add_argument("--egos", default=",".join(DEFAULT_EGOS),
                    help="comma-separated robot types to compute decisions for")
    ap.add_argument("--list", action="store_true",
                    help="print the most action-packed games and exit")
    args = ap.parse_args()

    if args.list:
        list_games(args.db)
        return

    gid = args.game_id or list_games(args.db, 1)[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    con = sqlite3.connect(args.db)
    matches = load_matches(con)
    m = matches[matches.game_id == gid]
    if m.empty:
        raise SystemExit(f"game_id {gid} not in {args.db}")
    m = m.iloc[0]
    game = load_game_arrays(con, gid)
    con.close()
    if game.T == 0:
        raise SystemExit(f"game {gid} has no timeseries rows")

    vmap = VisibilityMap.load(args.vis_map)
    tprior = TeamPrior.load(args.team_prior)
    if vmap is None or tprior is None:
        # The policy was trained with both priors wired in.  Leaving them out
        # zeroes 12 of its 161 input columns, so it still runs but is being fed
        # a state it never saw in training — worth a loud warning, not a crash.
        print(f"WARNING: vis_map={'ok' if vmap else 'MISSING'} "
              f"team_prior={'ok' if tprior else 'MISSING'} — the policy will see "
              f"zeroed prior columns and its decisions will not match training.")

    alt_map = (load_target_evidence(args.evidence)
               if args.evidence and os.path.exists(args.evidence) else {})
    policy, norm, info = load_policy(args.run, device=device)
    if not info["spec"].is_tactical:
        raise SystemExit(f"{args.run} is a {info['action_mode']} policy; the "
                         f"viewer needs a tactical one")

    robots, buildings = [], []
    for camp in S.CAMPS:
        for t in S.MOBILE_TYPES:
            tr = entity_track(game, t, camp)
            if tr:
                robots.append(tr)
        for t in S.BUILDING_TYPES:
            tr = entity_track(game, t, camp)
            if tr:
                buildings.append(tr)

    egos = []
    for rtype in [r.strip() for r in args.egos.split(",") if r.strip()]:
        rtype = S.resolve_agent(rtype)
        for camp in S.CAMPS:
            e = game.get(rtype, camp)
            if e.alive.sum() < 30:            # never really fielded
                continue
            egos.append(ego_decisions(game, camp, rtype, policy, norm, info,
                                      vmap, tprior, gid, args.goal_horizon,
                                      device, alt_map=alt_map,
                                      target_cone=args.target_cone,
                                      target_tau=args.target_tau))

    out = dict(
        meta=dict(
            game_id=int(gid), region=str(m.region), winner=str(m.winner),
            red_school=str(m.red_school), blue_school=str(m.blue_school),
            duration=int(m.duration), T=int(game.T),
            field=dict(x=S.FIELD_X, y=S.FIELD_Y),
            goal_horizon=args.goal_horizon,
            target_cone=args.target_cone, target_tau=args.target_tau,
            evidence=bool(alt_map),
            policy=dict(algo=info["algo"], run=os.path.basename(
                args.run.rstrip("/\\")), obs_dim=int(info["obs_dim"])),
            target_classes=list(S.MOBILE_TYPES) + ["无目标"],
            no_target=NO_TARGET,
        ),
        robots=robots, buildings=buildings, egos=egos,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        # allow_nan=False so a future non-finite value fails here, loudly,
        # instead of producing a file the browser silently refuses to parse
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"),
                  allow_nan=False)
    kb = os.path.getsize(args.out) / 1024
    print(f"[replay] {args.out}  {kb:.0f} KB  game={gid}  T={game.T}s  "
          f"robots={len(robots)}  decision-tracks={len(egos)}")
    for e in egos:
        n = len(e["agree"]["target"])
        tgt = 100.0 * np.mean(e["agree"]["target"])
        fire = 100.0 * np.mean(e["agree"]["fire"])
        nav = np.nanmean(np.asarray(e["agree"]["nav"], dtype=np.float64))
        ev = e.get("evidence", {}).get("victim", [])
        vals = [int(e["model"]["target"][k] == v) for k, v in enumerate(ev) if v >= 0]
        ev_s = f"  evidence {100.0*np.mean(vals):5.1f}% ({len(vals)})" if vals else ""
        print(f"   {e['camp']}{e['type']:<6} agreement over {n}s: "
              f"target {tgt:5.1f}%  fire {fire:5.1f}%  nav_cos {nav:+.3f}{ev_s}")


if __name__ == "__main__":
    main()
