"""What the policy would do from *every* position on the field, as it evolves.

A replay shows one decision per second: the one the ego actually faced.  This
asks the counterfactual instead — freeze a real game state, then put the ego at
every cell of the arena in turn and query the policy at each one.  The result is
a vector field: where the policy wants to go from anywhere, given this enemy
configuration, plus where it would declare weapons free.

Beyond looking good, this answers a real objection.  ``eval/win_alignment``
showed that a *state-independent constant* policy captures +0.0260 of a +0.0265
agreement lift, so "the policy agrees with winners" is mostly a confound.  A
field that visibly deforms as the enemies move is direct evidence that the
policy is reading the state rather than emitting a habit.  If these frames were
all identical, that would be worth knowing too.

Building the observation for a hypothetical position is the fiddly part.  It is
NOT enough to overwrite the two ego-position columns: every ally offset, every
enemy bearing and distance, and the engagement prior are all computed *relative
to the ego*, so a synthetic position has to go through ``features.build_obs``
like any other.  We do that by constructing a fake game whose "time axis" is the
grid of candidate positions, then restoring the two genuine time columns.

    # one frame, for the README
    python -m viz.policy_field --db dataset/....sqlite --game-id N --t 180

    # the animation
    python -m viz.policy_field --db dataset/....sqlite --game-id N \\
        --t0 120 --t1 260 --step 2 --fmt gif
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import sqlite3
import sys

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap

from rm_rl.data import schema as S
from rm_rl.data.build_dataset import load_game_arrays, load_matches
from rm_rl.data.features import Entity, GameArrays, build_obs, obs_feature_names
from rm_rl.data.team_prior import TeamPrior
from rm_rl.data.vis_map import VisibilityMap
from rm_rl.deploy import load_policy
from . import field_canvas

# set from --field-image in main(); None disables the arena backdrop
FIELD_IMAGE: str | None = None

TGT_COL =["#ff8f4d", "#b98cff", "#4dd97e", "#4d9dff", "#ff6ec7", "#f7d154",
           "#161b22"]                      # last = "no target", stays background
TGT_CMAP = ListedColormap(TGT_COL)
BG, FG, DIM, GRID = "#0d1117", "#e6edf3", "#7d8590", "#21262d"

_FIELDS = [f.name for f in dataclasses.fields(Entity)]


def frozen_game(game: GameArrays, t: int, gx, gy, ego_id: int) -> GameArrays:
    """A fake game of length len(gx) where 'time' indexes candidate positions.

    Every entity is pinned to its state at second `t`; only the ego moves, and
    it visits every cell of the sweep grid.  ``build_obs`` then produces one row
    per candidate position in a single vectorised call.
    """
    n = len(gx)
    ent = {}
    for rid, e in game.ent.items():
        vals = {f: np.full(n, float(getattr(e, f)[t]), np.float32)
                for f in _FIELDS}
        ent[rid] = Entity(**vals)
    ego = ent.get(ego_id)
    if ego is None:
        raise SystemExit(f"robot {ego_id} not present in this game")
    ego.x = np.asarray(gx, np.float32)
    ego.y = np.asarray(gy, np.float32)
    ego.alive = np.ones(n, np.float32)      # ask the counterfactual, not "it died"
    return GameArrays(n, ent)


class FieldSweep:
    """Evaluates the policy over a grid of hypothetical ego positions."""

    def __init__(self, game, camp, rtype, policy, norm, info, vmap, tprior,
                 gid, spacing, device):
        self.__dict__.update(locals())
        del self.self
        xs = np.arange(spacing / 2, S.FIELD_X, spacing)
        ys = np.arange(spacing / 2, S.FIELD_Y, spacing)
        self.gx, self.gy = np.meshgrid(xs, ys)
        self.flat_x, self.flat_y = self.gx.ravel(), self.gy.ravel()
        self.shape = self.gx.shape
        self.ego_id = S.robot_id(rtype, camp)
        self.tfeat = tprior.feats(gid, camp) if tprior is not None else None
        self.scale = np.asarray(info["act_scale"], np.float32)

        names = obs_feature_names(rtype)
        self.i_time = [names.index("time.elapsed"), names.index("time.remaining")]
        # the genuine per-second observation, used only to restore the two time
        # columns the synthetic game necessarily gets wrong
        self.obs_real = build_obs(game, camp, rtype, vis_map=vmap,
                                  team_feat=self.tfeat)

    @torch.no_grad()
    def at(self, t: int):
        fake = frozen_game(self.game, t, self.flat_x, self.flat_y, self.ego_id)
        obs = build_obs(fake, self.camp, self.rtype, vis_map=self.vmap,
                        team_feat=self.tfeat)
        # `fake`'s time axis is the position grid, so its time block counts
        # candidate cells instead of seconds — put the real values back.
        for c in self.i_time:
            obs[:, c] = self.obs_real[t, c]

        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        mean, fire_logit, tgt_logits = self.policy.policy.heads(
            self.norm.normalize(o))
        nav = mean.clamp(-1, 1).cpu().numpy()
        return dict(
            gx=(nav[:, 0] * self.scale[0]).reshape(self.shape),
            gy=(nav[:, 1] * self.scale[1]).reshape(self.shape),
            p_fire=torch.sigmoid(fire_logit).cpu().numpy().reshape(self.shape),
            target=torch.softmax(tgt_logits, -1).argmax(-1).cpu().numpy()
                        .reshape(self.shape),
        )


# ---------------------------------------------------------------------------
def setup_axes(ax):
    ax.set_facecolor(BG)
    ax.set_xlim(0, S.FIELD_X)
    ax.set_ylim(0, S.FIELD_Y)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(GRID)
    if FIELD_IMAGE:
        field_canvas.draw(ax, FIELD_IMAGE, alpha=.42, desaturate=.55)


def draw_frame(ax, sweep, t, arrow_stride, show_fire):
    ax.clear()
    setup_axes(ax)
    r = sweep.at(t)
    game, camp = sweep.game, sweep.camp
    ecamp = S.enemy_camp(camp)

    # Background = which enemy the policy would prioritise from each cell.  This
    # is the channel worth spending colour on: it tiles the arena into tactical
    # regions and it changes as the enemies move.  (The fire gate was tried here
    # first and is a poor fit — it is near-zero across the whole field for most
    # of a match, so it renders as a flat rectangle.)
    ax.imshow(r["target"], origin="lower", cmap=TGT_CMAP, vmin=-.5, vmax=6.5,
              extent=[0, S.FIELD_X, 0, S.FIELD_Y], interpolation="nearest",
              alpha=.55, zorder=0)
    if show_fire:
        # a single contour where weapons-free flips — crisp, and draws nothing
        # at all on the many seconds where the gate is shut everywhere
        if (r["p_fire"] > .5).any() and not (r["p_fire"] > .5).all():
            ax.contour(sweep.gx, sweep.gy, r["p_fire"], levels=[.5],
                       colors="#ffffff", linewidths=1.6, alpha=.85, zorder=3)

    # Arrow *direction* is the decision; its length is how far the robot happens
    # to be committed to travel, which would swamp the plot at true scale.  Draw
    # near-uniform arrows and let the background carry the target choice.
    s = arrow_stride
    X, Y = sweep.gx[::s, ::s], sweep.gy[::s, ::s]
    U, V = r["gx"][::s, ::s], r["gy"][::s, ::s]
    n = np.hypot(U, V)
    k = np.clip(n / 4.0, .35, 1.0) * sweep.spacing * s * .92
    U, V = U / np.maximum(n, 1e-6) * k, V / np.maximum(n, 1e-6) * k
    ax.quiver(X, Y, U, V, color="#e6edf3", angles="xy", scale_units="xy",
              scale=1, width=.0019, headwidth=4.0, headlength=4.6,
              headaxislength=4.0, alpha=.9, zorder=2)

    # the real robots, at this very second
    for rt in S.MOBILE_TYPES:
        for cp in S.CAMPS:
            e = game.get(rt, cp)
            if e.alive[t] <= 0 or (abs(e.x[t]) < 1e-6 and abs(e.y[t]) < 1e-6):
                continue
            foe = cp == ecamp
            ax.plot(e.x[t], e.y[t], "o", ms=13 if foe else 10,
                    mfc="#ff5b53" if cp == S.CAMP_RED else "#4d9dff",
                    mec="#0d1117", mew=1.6, zorder=4)
            if foe:
                ax.plot(e.x[t], e.y[t], "o", ms=20, mfc="none",
                        mec=TGT_COL[S.MOBILE_TYPES.index(rt)], mew=1.6,
                        alpha=.8, zorder=3)
    # and where the ego actually was, for reference
    ego = game.get(sweep.rtype, camp)
    if ego.alive[t] > 0:
        ax.plot(ego.x[t], ego.y[t], "*", ms=22, mfc="#ffffff", mec="#0d1117",
                mew=1.4, zorder=5)
    return r


def legend_text(ax, sweep, t, r):
    cls = list(S.MOBILE_TYPES) + ["无目标"]
    counts = np.bincount(r["target"].ravel(), minlength=7)
    top = int(counts.argmax())
    ax.text(.012, .955,
            f"t = {t:>3}s     开火许可覆盖 {100 * (r['p_fire'] > .5).mean():4.1f}% 场地"
            f"     主导目标:{cls[top]}",
            transform=ax.transAxes, color=FG, fontsize=12, va="top",
            family="monospace",
            bbox=dict(fc=BG, ec=GRID, alpha=.9, pad=5))
    ax.text(.012, .045,
            "箭头 = 从该位置出发时的导航方向;底色 = 该位置会优先打谁;"
            "白线 = 开火许可的分界;★ = 该秒真实位置",
            transform=ax.transAxes, color=DIM, fontsize=10,
            bbox=dict(fc=BG, ec=GRID, alpha=.9, pad=4))
    # a colour key for the target regions, built from the classes actually present
    # boxed, because the key sits on top of the very regions it describes
    present = [c for c in range(7) if counts[c] > 0 and c != 6]
    for i, c in enumerate(present):
        ax.text(.988, .88 - i * .062, f"■ {cls[c]}", transform=ax.transAxes,
                color=TGT_COL[c], fontsize=11, ha="right", fontweight="bold",
                zorder=6, bbox=dict(fc=BG, ec=GRID, alpha=.92, pad=2.5))


def report(sweep, T, step):
    """How much does the field actually vary — over space, and over time?

    ``eval/win_alignment`` found that a state-independent constant policy
    reproduces nearly all of the win/loss agreement lift, which leaves open the
    possibility that the learned policy is also close to constant.  These two
    numbers settle that directly: a constant policy would score ~0 on both.
    """
    ts = list(range(0, T - 1, step))
    fields = [sweep.at(t) for t in ts]

    # spatial: at a fixed second, how much does the nav direction differ across
    # the arena?  Circular standard deviation of the arrow angle, in degrees.
    spatial = []
    for f in fields:
        a = np.arctan2(f["gy"], f["gx"]).ravel()
        R = np.hypot(np.cos(a).mean(), np.sin(a).mean())
        spatial.append(np.degrees(np.sqrt(-2 * np.log(max(R, 1e-9)))))

    # temporal: from one sampled second to the next, what share of cells change
    # their prioritised target, and how far does the arrow swing?
    flips, swings = [], []
    for a, b in zip(fields, fields[1:]):
        flips.append(float((a["target"] != b["target"]).mean()))
        d = np.abs(np.degrees(np.arctan2(b["gy"], b["gx"])
                              - np.arctan2(a["gy"], a["gx"])))
        swings.append(float(np.minimum(d, 360 - d).mean()))

    ncell = fields[0]["target"].size
    named = np.mean([(f["target"] != 6).sum() for f in fields])
    print(f"\n=== field variability over {len(ts)} sampled seconds "
          f"(every {step}s) ===")
    print(f"  spatial  arrow-direction circular SD across the arena : "
          f"{np.mean(spatial):6.1f} deg")
    print(f"  temporal target flips per {step}s, share of cells      : "
          f"{np.mean(flips):6.1%}")
    print(f"  temporal arrow swing per {step}s                       : "
          f"{np.mean(swings):6.1f} deg")
    print(f"  cells assigned a named target, per frame              : "
          f"{named:6.0f} / {ncell}")
    print("\nA state-independent policy would score ~0 on the first three. "
          "These are the direct\nrebuttal to the constant-policy confound "
          "win_alignment exposed.")


# ---------------------------------------------------------------------------
def main():
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--game-id", type=int, required=True)
    ap.add_argument("--run", default="rm_runs/infantry_iql_tactical")
    ap.add_argument("--out", default="viz/figures")
    ap.add_argument("--vis-map", default="data/vis_map.npz")
    ap.add_argument("--team-prior", default="data/team_prior.json")
    ap.add_argument("--agent", default="infantry3")
    ap.add_argument("--camp", default=S.CAMP_RED, choices=list(S.CAMPS))
    ap.add_argument("--spacing", type=float, default=0.7,
                    help="metres between sampled ego positions")
    ap.add_argument("--arrow-stride", type=int, default=2,
                    help="draw an arrow every Nth sampled cell")
    ap.add_argument("--t", type=int, default=-1, help="single frame, seconds")
    ap.add_argument("--t0", type=int, default=-1)
    ap.add_argument("--t1", type=int, default=-1)
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--fmt", default="gif", choices=["gif", "mp4"])
    ap.add_argument("--dpi", type=int, default=140)
    ap.add_argument("--no-fire", action="store_true")
    ap.add_argument("--report", action="store_true",
                    help="quantify how much the field varies over space and "
                         "time instead of rendering")
    ap.add_argument("--field-image", default=field_canvas.IMAGE_PATH,
                    help="arena backdrop; see viz/assets/README.md")
    ap.add_argument("--no-field-image", action="store_true")
    ap.add_argument("--font", default="Microsoft YaHei")
    args = ap.parse_args()

    global FIELD_IMAGE
    FIELD_IMAGE = (None if args.no_field_image
                   or not field_canvas.available(args.field_image)
                   else args.field_image)
    if FIELD_IMAGE is None and not args.no_field_image:
        print(f"[policy_field] no arena image at {args.field_image} — plain "
              f"background (see viz/assets/README.md)")

    for fam in ("font.sans-serif", "font.monospace", "font.serif"):
        plt.rcParams[fam] = [args.font, "SimHei", "DejaVu Sans"]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False
    os.makedirs(args.out, exist_ok=True)

    con = sqlite3.connect(args.db)
    matches = load_matches(con)
    m = matches[matches.game_id == args.game_id]
    if m.empty:
        raise SystemExit(f"game_id {args.game_id} not found")
    game = load_game_arrays(con, args.game_id)
    con.close()

    vmap = VisibilityMap.load(args.vis_map)
    tprior = TeamPrior.load(args.team_prior)
    if vmap is None or tprior is None:
        print(f"WARNING: vis_map={'ok' if vmap else 'MISSING'} "
              f"team_prior={'ok' if tprior else 'MISSING'} — zeroed prior "
              f"columns; the field will not match the trained policy.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy, norm, info = load_policy(args.run, device=device)
    if not info["spec"].is_tactical:
        raise SystemExit(f"{args.run} is not a tactical policy")

    rtype = S.resolve_agent(args.agent)
    sweep = FieldSweep(game, args.camp, rtype, policy, norm, info, vmap, tprior,
                       args.game_id, args.spacing, device)
    print(f"[policy_field] {rtype} ({args.camp}) on game {args.game_id}, "
          f"T={game.T}s, sweep grid {sweep.shape[1]}x{sweep.shape[0]} "
          f"({sweep.flat_x.size} positions/frame)")

    if args.report:
        report(sweep, game.T, args.step)
        return

    fig, ax = plt.subplots(figsize=(11, 6.2))
    fig.patch.set_facecolor(BG)
    title = (f"{m.iloc[0].red_school} vs {m.iloc[0].blue_school} — "
             f"策略在全场每个位置的决策({args.camp}方{rtype})")

    if args.t >= 0:
        r = draw_frame(ax, sweep, min(args.t, game.T - 1), args.arrow_stride,
                       not args.no_fire)
        ax.set_title(title, color=FG, fontsize=14, loc="left", pad=10,
                     fontweight="bold")
        legend_text(ax, sweep, args.t, r)
        out = os.path.join(args.out, f"policy_field_t{args.t}.png")
        fig.tight_layout()
        fig.savefig(out, dpi=args.dpi, facecolor=BG, bbox_inches="tight")
        print(f"  wrote {out}")
        return

    t0 = args.t0 if args.t0 >= 0 else int(game.T * .25)
    t1 = args.t1 if args.t1 >= 0 else int(game.T * .75)
    frames = list(range(t0, min(t1, game.T - 1), args.step))

    def render(t):
        r = draw_frame(ax, sweep, t, args.arrow_stride, not args.no_fire)
        ax.set_title(title, color=FG, fontsize=14, loc="left", pad=10,
                     fontweight="bold")
        legend_text(ax, sweep, t, r)
        return []

    anim = animation.FuncAnimation(fig, render, frames=frames, blit=False)
    out = os.path.join(args.out, f"policy_field.{args.fmt}")
    if args.fmt == "mp4":
        writer = animation.FFMpegWriter(fps=args.fps, bitrate=3200)
    else:
        writer = animation.PillowWriter(fps=args.fps)
    print(f"  rendering {len(frames)} frames -> {out}")
    # Fixed margins, NOT tight_layout: every frame calls ax.clear(), which drops
    # the title tight_layout had reserved room for, so the title of frame 2
    # onwards gets cropped off the top of the video.
    fig.subplots_adjust(left=.015, right=.985, top=.925, bottom=.02)
    anim.save(out, writer=writer, dpi=args.dpi, savefig_kwargs=dict(facecolor=BG))
    print(f"  wrote {out}  ({os.path.getsize(out) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
