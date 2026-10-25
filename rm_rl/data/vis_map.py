"""Empirical engagement map — a data-driven stand-in for line-of-sight.

The referee log contains **no visibility information**.  Positions come from the
arena's own global tracking, not from any robot's perception, so "can robot A
see robot B" is simply not recorded.  ``x = y = 0`` marks a target the referee
system momentarily lost, which is a tracking dropout, not an occlusion.

Rather than invent an obstacle map (the field geometry is not in the dataset),
we estimate visibility **behaviourally**: aggregate over every game where a
robot actually engaged from cell A a target in cell B, and divide by how often
the two were merely co-present.  The resulting ratio is high for cell pairs with
a clear firing lane and low for pairs separated by cover — an occupancy-free,
purely empirical proxy for "these two positions can see each other".

Engagement is detected the same way target labels are (``features.target_soft``):
the shooter fired 17 mm this second and the enemy sat inside its muzzle cone.

Everything is accumulated in the canonical (red-mirrored) frame, so the map is
side-invariant like the rest of the pipeline.

Build once, reuse for every dataset build:

    python -m rm_rl.data.vis_map --db dataset/rmuc_2026_region_dataset.sqlite \\
        --out data/vis_map.npz
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys

import numpy as np

from . import schema as S

CELL = 2.0                                   # metres per grid cell
NX = int(np.ceil(S.FIELD_X / CELL))          # 14
NY = int(np.ceil(S.FIELD_Y / CELL))          # 8
NCELL = NX * NY                              # 112

CONE_DEG = 15.0                              # muzzle cone that counts as "engaging"


def cell_index(x, y):
    """Vectorised (x, y) [m, canonical frame] -> flat cell id in [0, NCELL)."""
    ix = np.clip(np.floor(np.asarray(x) / CELL), 0, NX - 1).astype(np.int64)
    iy = np.clip(np.floor(np.asarray(y) / CELL), 0, NY - 1).astype(np.int64)
    return iy * NX + ix


class VisibilityMap:
    """Loaded engagement prior; ``prior()`` is what the observation consumes."""

    def __init__(self, ratio: np.ndarray):
        self.ratio = ratio.astype(np.float32)      # [NCELL, NCELL], mean ~1.0

    @classmethod
    def load(cls, path: str | None):
        if not path or not os.path.exists(path):
            return None
        z = np.load(path)
        return cls(z["ratio"])

    def prior(self, ex, ey, px, py, known=None):
        """Per-timestep engagement prior between ego (ex,ey) and target (px,py).

        Returns an array shaped like the inputs, ~1.0 for an average pair,
        >1 for cell pairs that historically trade fire, ~0 for pairs that never
        do (i.e. probably occluded).  Unknown positions get 0.
        """
        a = cell_index(ex, ey)
        b = cell_index(px, py)
        v = self.ratio[a, b]
        if known is not None:
            v = v * known
        return v.astype(np.float32)


def _wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


def build(db_path: str, out_path: str, limit_games: int = 0, cone=CONE_DEG,
          verbose=True):
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    gids = [r[0] for r in cur.execute(
        f"SELECT game_id FROM {S.T_MATCHES} ORDER BY game_id").fetchall()]
    if limit_games > 0:
        gids = gids[:limit_games]

    engage = np.zeros((NCELL, NCELL), np.float64)
    copres = np.zeros((NCELL, NCELL), np.float64)

    # types that actually carry a tracked turret (工程 / buildings hold the
    # sentinel heading -140 and must never contribute)
    shooters = [S.TYPE_HERO, S.TYPE_INFANTRY3, S.TYPE_INFANTRY4,
                S.TYPE_AERIAL, S.TYPE_SENTRY]

    for n, gid in enumerate(gids):
        rows = cur.execute(
            f'SELECT "时刻秒", robot_id, x, y, "枪口朝向", "当前血量" '
            f"FROM {S.T_TIMESERIES} WHERE game_id=?", (int(gid),)).fetchall()
        if not rows:
            continue
        st = {}
        for t, rid, x, y, yaw, hp in rows:
            st[(int(t), int(rid))] = (x, y, yaw, (hp or 0) > 0)
        fired = set()
        for t, rid in cur.execute(
                f'SELECT "时刻秒", robot_id FROM {S.T_EVENTS} '
                f'WHERE game_id=? AND "事件类型"=? AND "类别"=?',
                (int(gid), S.EV_SHOOT, "17mm")).fetchall():
            fired.add((int(t), int(rid)))

        for camp in S.CAMPS:
            mirror = camp == S.CAMP_BLUE
            ecamp = S.enemy_camp(camp)
            foes = [S.robot_id(t, ecamp) for t in S.MOBILE_TYPES]
            for stype in shooters:
                sid = S.robot_id(stype, camp)
                for (t, rid), (x, y, yaw, alive) in st.items():
                    if rid != sid or not alive or yaw is None:
                        continue
                    if x == 0 and y == 0:
                        continue
                    if float(yaw) == S.YAW_SENTINEL:      # no turret data
                        continue
                    ax, ay = (S.FIELD_X - x, S.FIELD_Y - y) if mirror else (x, y)
                    ayaw = _wrap_deg(yaw + 180.0) if mirror else yaw
                    ca = cell_index(ax, ay)
                    shot = (t, rid) in fired
                    for fid in foes:
                        e = st.get((t, fid))
                        if not e or not e[3]:
                            continue
                        ex, ey = e[0], e[1]
                        if ex == 0 and ey == 0:
                            continue
                        px, py = ((S.FIELD_X - ex, S.FIELD_Y - ey) if mirror
                                  else (ex, ey))
                        cb = cell_index(px, py)
                        copres[ca, cb] += 1.0
                        if shot:
                            err = abs(_wrap_deg(
                                np.degrees(np.arctan2(py - ay, px - ax)) - ayaw))
                            if err < cone:
                                engage[ca, cb] += 1.0
        if verbose and (n + 1) % 50 == 0:
            print(f"  vis_map: {n + 1}/{len(gids)} games", flush=True)
    con.close()

    # Laplace-smoothed engagement rate, then expressed relative to the global
    # rate so the feature is centred near 1.0 regardless of absolute fire volume.
    rate = (engage + 1.0) / (copres + 2.0)
    gmean = float(engage.sum() + 1.0) / float(copres.sum() + 2.0)
    ratio = np.clip(rate / max(gmean, 1e-9), 0.0, 4.0)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    np.savez_compressed(out_path, ratio=ratio.astype(np.float32),
                        engage=engage.astype(np.float32),
                        copresent=copres.astype(np.float32),
                        cell=CELL, nx=NX, ny=NY, cone=cone)
    cov = float((copres > 0).mean())
    print(f"[vis_map] {out_path}  cells={NCELL}  pair-coverage={cov:.1%}  "
          f"global_engage_rate={gmean:.4f}  ratio[min,max]=[{ratio.min():.2f},"
          f"{ratio.max():.2f}]")
    return ratio


def main():
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", default="data/vis_map.npz")
    ap.add_argument("--limit-games", type=int, default=0)
    ap.add_argument("--cone", type=float, default=CONE_DEG)
    args = ap.parse_args()
    build(args.db, args.out, args.limit_games, args.cone)


if __name__ == "__main__":
    main()
