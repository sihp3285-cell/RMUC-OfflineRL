"""Learn reward weights from match outcomes (instead of hand-setting them).

For every (game, side) we sum the UNWEIGHTED shaping components over the round
(reward.reward_features) and label it with win/loss.  A logistic-regression
    P(win) = sigmoid( w . features )
then tells us the *data-driven* weight of each objective, which we compare to
the hand-tuned weights.  This is the cheap, interpretable form of reward
learning recommended over classic IRL for this dataset (no simulator, mixed
quality data, ~1k trajectory-level labels).

Run:
    python -m rm_rl.data.reward_model --db dataset/rmuc_2026_region_dataset.sqlite --agent sentry
"""
from __future__ import annotations

import argparse
import sqlite3
import sys

import numpy as np
import torch

from . import schema as S
from .reward import reward_features, FEATURE_NAMES, RewardConfig
from .build_dataset import load_matches, load_game_arrays

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **k):
        return x

# signed hand weights implied by RewardConfig (defenders enter negatively)
_HAND_SIGN = {
    "outpost_atk": +1, "outpost_def": -1, "base_atk": +1, "base_def": -1,
    "dmg_deal": +1, "dmg_take": -1, "kill": +1, "death": -1,
    "ego_surv": +1, "ego_hp": -1, "heat": -1, "econ": +1, "final_base": +1,
}
_HAND_MAG = {
    "outpost_atk": "w_outpost_atk", "outpost_def": "w_outpost_def",
    "base_atk": "w_base_atk", "base_def": "w_base_def",
    "dmg_deal": "w_dmg_deal", "dmg_take": "w_dmg_take",
    "kill": "w_kill", "death": "w_death", "ego_surv": "w_ego_surv",
    "ego_hp": "w_ego_hp", "heat": "w_heat", "econ": "w_econ",
    "final_base": "w_final_base",
}


def _auc(y, s):
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    pos = y == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def main():
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--agent", default="sentry")
    ap.add_argument("--l2", type=float, default=1e-2)
    ap.add_argument("--min-len", type=int, default=60)
    ap.add_argument("--limit-games", type=int, default=0)
    args = ap.parse_args()
    agent = S.resolve_agent(args.agent)

    con = sqlite3.connect(args.db)
    matches = load_matches(con)
    m_by_id = {int(r.game_id): r for r in matches.itertuples(index=False)}
    gids = list(m_by_id.keys())
    if args.limit_games > 0:
        gids = gids[: args.limit_games]

    X, y = [], []
    for gid in tqdm(gids, desc="games"):
        m = m_by_id[gid]
        game = load_game_arrays(con, gid)
        if game.T < args.min_len:
            continue
        for camp in S.CAMPS:
            ego = game.get(agent, camp)
            if ego.alive.sum() < args.min_len * 0.3:
                continue
            X.append(reward_features(game, camp, agent))
            y.append(1.0 if m.winner == camp else 0.0)
    con.close()

    X = np.asarray(X, np.float64)
    y = np.asarray(y, np.float64)
    print(f"\nsamples: {len(y)}  win-rate: {y.mean():.3f}  features: {X.shape[1]}")

    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Xs = (X - mu) / sd

    Xt = torch.tensor(Xs)
    yt = torch.tensor(y)
    w = torch.zeros(X.shape[1], requires_grad=True, dtype=torch.float64)
    b = torch.zeros(1, requires_grad=True, dtype=torch.float64)
    opt = torch.optim.LBFGS([w, b], lr=0.5, max_iter=200)

    def closure():
        opt.zero_grad()
        logit = Xt @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, yt)
        loss = loss + args.l2 * (w * w).sum()
        loss.backward()
        return loss
    opt.step(closure)

    with torch.no_grad():
        logit = (Xt @ w + b).numpy()
        acc = float(((logit > 0).astype(float) == y).mean())
        auc = _auc(y, logit)
    w_std = w.detach().numpy()           # standardized (importance)
    w_raw = w_std / sd                   # per raw-unit weight

    hand = RewardConfig()
    print(f"\nlogistic fit — train acc {acc:.3f}  AUC {auc:.3f}\n")
    print(f"{'component':<12} {'learned(std)':>12} {'learned(raw)':>12} "
          f"{'hand(signed)':>12}  agree?")
    order = np.argsort(-np.abs(w_std))
    for i in order:
        name = FEATURE_NAMES[i]
        hw = _HAND_SIGN[name] * getattr(hand, _HAND_MAG[name])
        agree = "yes" if np.sign(w_std[i]) == np.sign(hw) or abs(w_std[i]) < 1e-3 else "NO"
        print(f"{name:<12} {w_std[i]:>12.3f} {w_raw[i]:>12.3f} {hw:>12.2f}  {agree}")
    print("\nRead: learned(std) = importance for winning (bigger |.| = matters "
          "more); sign should match hand(signed). 'NO' flags a weight whose sign "
          "the data disagrees with — worth revisiting in configs/*.yaml.")


if __name__ == "__main__":
    main()
