"""Occupancy figures: why the sentry logs are the wrong thing to imitate.

This renders the evidence behind the project's central design decision.  Offline
RL cannot exceed its demonstrator, and the logged sentry is not a demonstrator —
it is each team's own behaviour tree.

The statistic that matters is **per match**, not pooled.  Pooling every game
together answers "where do sentries stand across the whole league", which is
necessarily spread out because different teams park in different places; it
completely hides the thing being claimed, namely that *within a single match* a
sentry barely moves.  So every number here is computed per (match, robot) and
then averaged, and the pictures are drawn per team for the same reason.

Grid resolution is 1 m — the resolution the figures in the README were measured
at.  (The engagement prior in the observation uses a coarser 2 m grid; that is a
different quantity and the two should not be compared.)

Teams are anonymised by default.  The dataset names every school, but singling
out "this team's sentry always sits here" serves no purpose the argument needs —
and it is the same reasoning that made the observation's team prior anonymous.
Pass ``--name-teams`` to label them.

    python -m viz.heatmaps --db dataset/....sqlite --out viz/figures
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

from rm_rl.data import schema as S
from . import field_canvas

# set from --field-image in main(); None disables the arena backdrop
FIELD_IMAGE: str | None = None

CELL = 1.0             # metres per cell for every published statistic
NX = int(round(S.FIELD_X / CELL))
NY = int(round(S.FIELD_Y / CELL))
NCELL = NX * NY

FINE = 0.35            # metres per pixel in the rendered images
SMOOTH = 2.0           # gaussian sigma, in fine cells
MIN_STEPS = 60         # a trajectory shorter than this is not a match

THEME = {
    "dark":  dict(bg="#0d1117", fg="#e6edf3", dim="#7d8590", grid="#21262d"),
    "light": dict(bg="#ffffff", fg="#1f2328", dim="#656d76", grid="#d0d7de"),
}
HOT = LinearSegmentedColormap.from_list(
    "hot", ["#00000000", "#1f2a6e", "#3b6fd4", "#3fc9c9", "#f7d154", "#ff5b3d"])
C_SENTRY, C_INFANTRY = "#f7643d", "#3fc9c9"


# ---------------------------------------------------------------------------
def load_trajectories(db, rtypes, limit_games=0):
    """One entry per (match, robot): positions in the canonical red frame."""
    con = sqlite3.connect(db)
    cur = con.cursor()
    gids = [r[0] for r in cur.execute(
        f"SELECT game_id FROM {S.T_MATCHES} ORDER BY game_id")]
    if limit_games > 0:
        gids = gids[:limit_games]
    gset = set(gids)
    schools = {int(g): (r, b) for g, r, b in cur.execute(
        f'SELECT game_id, "红方学校", "蓝方学校" FROM {S.T_MATCHES}')}

    want = {}
    for camp in S.CAMPS:
        for t in rtypes:
            want[S.robot_id(t, camp)] = (t, camp)
    ids = ",".join(str(i) for i in want)

    acc = {}
    for gid, rid, x, y, hp in cur.execute(
            f'SELECT game_id, robot_id, x, y, "当前血量" FROM {S.T_TIMESERIES} '
            f"WHERE robot_id IN ({ids})"):
        if gid not in gset or not hp or hp <= 0 or x is None:
            continue
        if x == 0 and y == 0:          # referee lost the target this second
            continue
        _, camp = want[rid]
        if camp == S.CAMP_BLUE:        # canonicalise, as the pipeline does
            x, y = S.FIELD_X - x, S.FIELD_Y - y
        if not (0 <= x <= S.FIELD_X and 0 <= y <= S.FIELD_Y):
            continue
        acc.setdefault((gid, rid), []).append((x, y))

    out = []
    for (gid, rid), pts in acc.items():
        if len(pts) < MIN_STEPS:
            continue
        _, camp = want[rid]
        a = np.asarray(pts, np.float32)
        out.append(dict(x=a[:, 0], y=a[:, 1],
                        school=schools[gid][0 if camp == S.CAMP_RED else 1]))
    con.close()
    return out


def hist(x, y, cell=CELL):
    nx, ny = int(round(S.FIELD_X / cell)), int(round(S.FIELD_Y / cell))
    h, _, _ = np.histogram2d(x, y, bins=[nx, ny],
                             range=[[0, nx * cell], [0, ny * cell]])
    return h


def traj_stats(t):
    h = hist(t["x"], t["y"])
    p = h / h.sum()
    nz = p[p > 0]
    return (float((h > 0).sum()),
            float(-(nz * np.log(nz)).sum()),
            float(np.sort(p.ravel())[::-1][:5].sum()))


def summarise(trajs):
    r = np.array([traj_stats(t) for t in trajs])
    return dict(n=len(trajs), cells=float(r[:, 0].mean()),
                entropy=float(r[:, 1].mean()), top5=float(r[:, 2].mean()),
                cells_all=r[:, 0])


def school_maps(trajs, min_traj=6):
    """Per-school occupancy, pooled over that school's matches."""
    by = {}
    for t in trajs:
        by.setdefault(t["school"], []).append(t)
    return {s: v for s, v in by.items() if len(v) >= min_traj}


def team_similarity(trajs, min_traj=6):
    """Mean cosine similarity between *different* schools' occupancy maps.

    High = every team stands in the same places, i.e. a universal tactic that
    transfers.  Low = every team has its own private habit, which is exactly
    what a hand-written per-team script produces.
    """
    maps = {}
    for s, v in school_maps(trajs, min_traj).items():
        g = sum(hist(t["x"], t["y"]) for t in v).ravel()
        n = np.linalg.norm(g)
        if n > 0:
            maps[s] = g / n
    keys = sorted(maps)
    if len(keys) < 2:
        return float("nan"), 0
    sims = [float(maps[a] @ maps[b])
            for i, a in enumerate(keys) for b in keys[i + 1:]]
    return float(np.mean(sims)), len(keys)


def fine_image(trajs):
    x = np.concatenate([t["x"] for t in trajs])
    y = np.concatenate([t["y"] for t in trajs])
    return gaussian_filter(hist(x, y, FINE), SMOOTH)


# ---------------------------------------------------------------------------
def _field(ax, th):
    ax.set_xlim(0, S.FIELD_X)
    ax.set_ylim(0, S.FIELD_Y)
    ax.set_aspect("equal")
    ax.set_facecolor("#070a0e" if th["bg"] == "#0d1117" else "#f2f4f7")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(th["grid"])
    if FIELD_IMAGE:
        # heavily knocked back: the arena is here to say *where* on the map the
        # density sits, not to compete with it
        field_canvas.draw(ax, FIELD_IMAGE, alpha=.30, desaturate=.65)


def _draw(ax, trajs, th):
    g = fine_image(trajs)
    ax.imshow(g.T / max(g.max(), 1e-9), origin="lower",
              extent=[0, S.FIELD_X, 0, S.FIELD_Y], cmap=HOT,
              vmin=0, vmax=1, interpolation="bilinear")


def small_multiples(sent, inf, out, th, dpi, n_teams, name_teams):
    """The whole argument in one picture: per team, sentry above infantry."""
    sm, im = school_maps(sent), school_maps(inf)
    common = sorted(set(sm) & set(im),
                    key=lambda s: -(len(sm[s]) + len(im[s])))[:n_teams]
    if not common:
        print("  skip small multiples (not enough per-school data)")
        return
    k = len(common)
    fig, axes = plt.subplots(2, k, figsize=(2.55 * k, 3.9), facecolor=th["bg"])
    axes = np.atleast_2d(axes)
    for j, s in enumerate(common):
        for i, (src, col) in enumerate(((sm[s], C_SENTRY), (im[s], C_INFANTRY))):
            ax = axes[i, j]
            _field(ax, th)
            _draw(ax, src, th)
            ax.text(.03, .09, f"{np.mean([traj_stats(t)[0] for t in src]):.0f} 格",
                    transform=ax.transAxes, color=col, fontsize=10.5,
                    fontweight="bold")
            if i == 0:
                ax.set_title(s if name_teams else f"队伍 {chr(65 + j)}",
                             color=th["fg"], fontsize=11, pad=6)
            if j == 0:
                # rotation=0: matplotlib rotates y-labels 90 deg by default,
                # which turns CJK into an unreadable vertical stack
                ax.set_ylabel("哨兵" if i == 0 else "步兵", color=col,
                              fontsize=13, fontweight="bold", labelpad=12,
                              rotation=0, ha="right", va="center")
    fig.suptitle("同一支队伍:哨兵钉在一个点,步兵铺满全场",
                 color=th["fg"], fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, facecolor=th["bg"], bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def pooled_figure(groups, out, th, dpi):
    fig, axes = plt.subplots(len(groups), 1, figsize=(9, 4.5 * len(groups)),
                             facecolor=th["bg"])
    for ax, g in zip(np.atleast_1d(axes), groups):
        _field(ax, th)
        _draw(ax, g["trajs"], th)
        st = g["stats"]
        ax.set_title(g["title"], color=th["fg"], fontsize=15, loc="left",
                     pad=10, fontweight="bold")
        # NOTE: the numbers are per-match averages while the image pools every
        # match, so the caption says so explicitly — the picture is the league,
        # the numbers are one robot in one game.
        ax.text(0.012, 0.045,
                f"单场平均:覆盖 {st['cells']:.1f}/{NCELL} 格   "
                f"位置熵 {st['entropy']:.2f} nats   前五格占 {st['top5']:.0%}",
                transform=ax.transAxes, color=th["fg"], fontsize=11.5,
                bbox=dict(fc=th["bg"], ec=th["grid"], alpha=.88, pad=5))
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, facecolor=th["bg"], bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def dispersion_figure(s_st, i_st, out, th, dpi):
    """The separation is not a mean effect — the two distributions barely touch."""
    fig, ax = plt.subplots(figsize=(8.4, 3.8), facecolor=th["bg"])
    ax.set_facecolor(th["bg"])
    hi = max(s_st["cells_all"].max(), i_st["cells_all"].max())
    bins = np.linspace(0, hi, 46)
    for vals, col, lab in ((s_st["cells_all"], C_SENTRY, "哨兵"),
                           (i_st["cells_all"], C_INFANTRY, "步兵")):
        ax.hist(vals, bins=bins, color=col, alpha=.72, label=lab, density=True)
        ax.axvline(vals.mean(), color=col, ls="--", lw=1.6)
        ax.text(vals.mean(), ax.get_ylim()[1] * .96, f" 均值 {vals.mean():.0f}",
                color=col, fontsize=10.5, va="top")
    ax.set_xlabel(f"单场覆盖的格子数(1 m 网格,全场 {NCELL} 格)",
                  color=th["dim"], fontsize=11)
    ax.set_xlim(0, hi)
    # the density value on the y-axis carries no meaning for a reader; the
    # shape and the separation are the whole point
    ax.set_yticks([])
    ax.tick_params(colors=th["dim"], labelsize=10)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(th["grid"])
    lg = ax.legend(frameon=False, fontsize=12)
    for t in lg.get_texts():
        t.set_color(th["fg"])
    ax.set_title("每一场都是如此,不是平均值的把戏", color=th["fg"],
                 fontsize=14, loc="left", pad=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, facecolor=th["bg"], bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def similarity_figure(rows, out, th, dpi):
    fig, ax = plt.subplots(figsize=(8, 3.2), facecolor=th["bg"])
    ax.set_facecolor(th["bg"])
    bars = ax.barh([r[0] for r in rows], [r[1] for r in rows],
                   color=[r[2] for r in rows], height=.5)
    for b, r in zip(bars, rows):
        ax.text(r[1] + .012, b.get_y() + b.get_height() / 2, f"{r[1]:.2f}",
                va="center", color=th["fg"], fontsize=13, fontweight="bold")
    ax.set_xlim(0, max(r[1] for r in rows) * 1.28)
    ax.set_xlabel("不同学校之间的走位相似度 (cosine)", color=th["dim"], fontsize=11)
    ax.tick_params(colors=th["fg"], labelsize=12)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("bottom", "left"):
        ax.spines[s].set_color(th["grid"])
    ax.set_title("各队哨兵各守各的角落,人类步兵却走向同样的位置",
                 color=th["fg"], fontsize=14, loc="left", pad=12,
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, facecolor=th["bg"], bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def engagement_figure(vis_map_path, out, th, dpi):
    """Where does a robot standing in the busiest cell actually trade fire?"""
    if not os.path.exists(vis_map_path):
        print(f"  skip engagement figure (no {vis_map_path})")
        return
    z = np.load(vis_map_path)
    ratio, copres = z["ratio"], z["copresent"]
    vnx, vny = int(z["nx"]), int(z["ny"])
    vcell = float(z["cell"])
    src = int(copres.sum(1).argmax())          # the map's busiest vantage point
    row = ratio[src].reshape(vny, vnx)
    sx = (src % vnx + .5) * vcell
    sy = (src // vnx + .5) * vcell

    fig, ax = plt.subplots(figsize=(9, 5.0), facecolor=th["bg"])
    _field(ax, th)
    ax.imshow(row, origin="lower", extent=[0, vnx * vcell, 0, vny * vcell],
              cmap=HOT, vmin=0, vmax=float(np.percentile(ratio, 99)),
              interpolation="nearest")
    ax.plot(sx, sy, "o", ms=13, mfc="#ffffff", mec=th["bg"], mew=2, zorder=5)
    ax.annotate("观察点", (sx, sy), (sx + 1.3, sy + 1.3), color=th["fg"],
                fontsize=12, fontweight="bold",
                arrowprops=dict(color=th["fg"], arrowstyle="-", lw=1.2))
    ax.set_title("经验交战图:从这个格子出发,历史上真正打得着的位置",
                 color=th["fg"], fontsize=14, loc="left", pad=10,
                 fontweight="bold")
    ax.text(.012, .045, "日志不含视线信息 — 用「开火且目标在枪口锥内」的频率反推",
            transform=ax.transAxes, color=th["dim"], fontsize=10.5,
            bbox=dict(fc=th["bg"], ec=th["grid"], alpha=.88, pad=4))
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, facecolor=th["bg"], bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
def main():
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", default="viz/figures")
    ap.add_argument("--vis-map", default="data/vis_map.npz")
    ap.add_argument("--limit-games", type=int, default=0)
    ap.add_argument("--style", default="dark", choices=["dark", "light"])
    ap.add_argument("--dpi", type=int, default=180)
    ap.add_argument("--teams", type=int, default=5,
                    help="how many teams in the small-multiples figure")
    ap.add_argument("--name-teams", action="store_true",
                    help="label panels with the real school names")
    ap.add_argument("--field-image", default=field_canvas.IMAGE_PATH,
                    help="arena backdrop; see viz/assets/README.md")
    ap.add_argument("--no-field-image", action="store_true")
    ap.add_argument("--font", default="Microsoft YaHei",
                    help="a font with CJK glyphs; matplotlib's default has none")
    args = ap.parse_args()

    global FIELD_IMAGE
    FIELD_IMAGE = (None if args.no_field_image
                   or not field_canvas.available(args.field_image)
                   else args.field_image)
    if FIELD_IMAGE is None and not args.no_field_image:
        print(f"[heatmaps] no arena image at {args.field_image} — plain "
              f"background (see viz/assets/README.md)")

    # Set the CJK font for *every* family: any text left on the default
    # monospace/serif stack silently loses its Chinese glyphs to tofu.
    for fam in ("font.sans-serif", "font.monospace", "font.serif"):
        plt.rcParams[fam] = [args.font, "SimHei", "DejaVu Sans"]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False
    th = THEME[args.style]
    os.makedirs(args.out, exist_ok=True)

    print("[heatmaps] reading trajectories ...")
    sent = load_trajectories(args.db, [S.TYPE_SENTRY], args.limit_games)
    inf = load_trajectories(args.db, [S.TYPE_INFANTRY3, S.TYPE_INFANTRY4],
                            args.limit_games)
    s_st, i_st = summarise(sent), summarise(inf)
    s_sim, s_n = team_similarity(sent)
    i_sim, i_n = team_similarity(inf)

    for name, st, sim, k in (("sentry", s_st, s_sim, s_n),
                             ("infantry", i_st, i_sim, i_n)):
        print(f"  {name:<9} traj={st['n']:>5}  cells={st['cells']:6.2f}/{NCELL}"
              f"  H={st['entropy']:.2f}  top5={st['top5']:.1%}"
              f"  cross-team sim={sim:.3f} ({k} schools)")
    print(f"  ratio: infantry covers {i_st['cells'] / s_st['cells']:.2f}x the map")

    # Per robot type as well: the README's comparison table has one column per
    # type, and a table whose numbers cannot be reproduced by the tool that
    # ships beside it is worse than no table.
    print(f"\n  per type (grid {CELL} m, {NCELL} cells, per-match averages):")
    per_type = {}
    for rt in (S.TYPE_SENTRY, S.TYPE_INFANTRY3, S.TYPE_INFANTRY4):
        tj = load_trajectories(args.db, [rt], args.limit_games)
        st = summarise(tj)
        sim, k = team_similarity(tj)
        per_type[rt] = (st, sim, k)
        print(f"    {rt:<6} traj={st['n']:>5}  cells={st['cells']:6.2f}"
              f"  H={st['entropy']:.3f}  top5={st['top5']:.1%}"
              f"  cross-team sim={sim:.3f} ({k} schools)")

    small_multiples(sent, inf, os.path.join(args.out, "per_team.png"),
                    th, args.dpi, args.teams, args.name_teams)
    pooled_figure(
        [dict(trajs=sent, stats=s_st, title="哨兵 — 各队自己写的行为树"),
         dict(trajs=inf, stats=i_st, title="步兵 — 人类操作手")],
        os.path.join(args.out, "occupancy.png"), th, args.dpi)
    dispersion_figure(s_st, i_st, os.path.join(args.out, "dispersion.png"),
                      th, args.dpi)
    similarity_figure([("哨兵", s_sim, C_SENTRY), ("步兵", i_sim, C_INFANTRY)],
                      os.path.join(args.out, "similarity.png"), th, args.dpi)
    engagement_figure(args.vis_map, os.path.join(args.out, "engagement.png"),
                      th, args.dpi)

    with open(os.path.join(args.out, "stats.txt"), "w", encoding="utf-8") as f:
        f.write(f"grid={CELL}m ({NX}x{NY}={NCELL} cells), per-match averages\n")
        rows = [("sentry", s_st, s_sim, s_n), ("infantry", i_st, i_sim, i_n)]
        rows += [(rt, st, sim, k) for rt, (st, sim, k) in per_type.items()]
        for name, st, sim, k in rows:
            f.write(f"{name}: trajectories={st['n']} "
                    f"cells={st['cells']:.2f}/{NCELL} "
                    f"entropy={st['entropy']:.3f} top5={st['top5']:.4f} "
                    f"cross_team_sim={sim:.4f} schools={k}\n")
    print(f"  wrote {os.path.join(args.out, 'stats.txt')}")


if __name__ == "__main__":
    main()
