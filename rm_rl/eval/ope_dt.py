"""OPE for the Decision Transformer.

The DT is not a Markov policy, so we cannot call pi(s) directly.  Instead we
replay it along the real trajectories: at every step we feed the last `ctx`
(return-to-go, obs, action) tokens — obs/actions teacher-forced from the log,
return-to-go anchored to a *target* return and decremented by realised reward —
and read the action the DT would take.  Those actions are then scored with the
same Fitted-Q-Evaluation used for the MLP policies, so the numbers are directly
comparable to BC / IQL / behaviour.

Run (after training a DT into rm_runs/sentry_dt):
    python -m rm_rl.eval.ope_dt --data data/sentry --run rm_runs/sentry_dt
    python -m rm_rl.eval.ope_dt --data data/sentry --run rm_runs/sentry_dt --target-pct 90
"""
from __future__ import annotations

import argparse
import copy
import json
import os

import numpy as np
import torch

from ..algos.networks import TwinQ
from ..data.dataset import TransitionDataset
from ..deploy import load_policy
from ..train.common import ensure_utf8_stdout
from .ope_fqe import behaviour_returns


@torch.no_grad()
def dt_actions_over_data(dt, ds, ep_start, ep_len, rew_raw, target_return,
                         ctx, rtg_scale, gamma_dt, device):
    """Return a_DT[N, act_dim]: the DT action at every transition's current
    state, teacher-forced on the logged obs/action history."""
    N = ds.obs.shape[0]
    ad = ds.act_dim
    a_dt = torch.zeros((N, ad), device=device)
    obs_all = ds.obs.to(device)
    act_all = ds.act.to(device)

    for s, L in zip(ep_start.tolist(), ep_len.tolist()):
        obs = obs_all[s:s + L]                       # [L, od]
        act = act_all[s:s + L]                        # [L, ad]
        r = rew_raw[s:s + L]
        # return-to-go anchored to target, decremented by realised reward
        if gamma_dt >= 1.0:
            cum_before = np.concatenate([[0.0], np.cumsum(r)[:-1]])
            rtg = (target_return - cum_before) / rtg_scale
        else:  # discounted variant
            rtg = np.zeros(L)
            run = target_return
            for k in range(L):
                rtg[k] = run / rtg_scale
                run = (run - r[k]) / gamma_dt
        rtg_t = torch.tensor(rtg, dtype=torch.float32, device=device)
        ts_t = torch.arange(L, device=device)

        # build L left-padded windows of length ctx ending at each position
        idx = torch.arange(L, device=device)
        win = idx.unsqueeze(1) - torch.arange(ctx - 1, -1, -1, device=device)  # [L,ctx]
        valid = (win >= 0)
        win_c = win.clamp(min=0)
        oW = obs[win_c] * valid.unsqueeze(-1)                # [L,ctx,od]
        aW = act[win_c] * valid.unsqueeze(-1)                # [L,ctx,ad]
        rW = (rtg_t[win_c] * valid).unsqueeze(-1)            # [L,ctx,1]
        tW = ts_t[win_c] * valid
        mW = valid.float()
        # batch through the DT (chunk to bound memory)
        for b in range(0, L, 512):
            sl = slice(b, min(b + 512, L))
            a = dt.act(oW[sl], aW[sl], rW[sl], tW[sl], mW[sl])
            a_dt[s + b: s + min(b + 512, L)] = a
    return a_dt


def fqe_with_actions(ds, ep_start, ep_len, rew_raw, a_cur, a_next, a_s0,
                     gamma=0.99, steps=40000, lr=3e-4, batch=1024, hidden=256,
                     depth=2, polyak=0.005, device="cpu", verbose=True):
    obs = ds.obs.to(device); act = ds.act.to(device)
    rew = ds.rew.to(device); next_obs = ds.next_obs.to(device)
    done = ds.done.to(device); N = obs.shape[0]

    qf = TwinQ(ds.obs_dim, ds.act_dim, hidden, depth).to(device)
    qt = copy.deepcopy(qf)
    for p in qt.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(qf.parameters(), lr=lr)
    for step in range(1, steps + 1):
        i = torch.randint(0, N, (batch,), device=device)
        with torch.no_grad():
            y = rew[i] + gamma * (1 - done[i]) * qt.q_min(next_obs[i], a_next[i])
        q1, q2 = qf(obs[i], act[i])
        loss = ((q1 - y) ** 2 + (q2 - y) ** 2).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        for p, tp in zip(qf.parameters(), qt.parameters()):
            tp.data.mul_(1 - polyak).add_(polyak * p.data)
        if verbose and step % 10000 == 0:
            print(f"  FQE step {step}/{steps} bellman_loss {float(loss.detach()):.4f}")
    s0 = obs[ep_start]
    with torch.no_grad():
        j_pi = float(qf.q_min(s0, a_s0).mean())
        j_all = float(qf.q_min(obs, a_cur).mean())
    return j_pi, j_all


def main():
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--target-pct", type=float, default=90.0,
                    help="target return percentile the DT conditions on")
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--gamma", type=float, default=0.99)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = TransitionDataset(args.data, "train")
    z = np.load(os.path.join(args.data, "train.npz"))
    ep_start, ep_len = z["ep_start"], z["ep_len"]
    rew_raw = z["rew"].astype(np.float64)

    dt, _, info = load_policy(args.run, device=device)
    ctx = dt.ctx
    # rtg_scale / gamma_dt from the run's stored config if present
    # match load_policy(): best.pt if it exists, else final.pt
    _ck = os.path.join(args.run, "best.pt")
    if not os.path.exists(_ck):
        _ck = os.path.join(args.run, "final.pt")
    state = torch.load(open(_ck, "rb"), map_location="cpu")
    dp = state.get("config", {}).get("dt", {})
    rtg_scale = dp.get("rtg_scale", 100.0)
    gamma_dt = dp.get("gamma", 1.0)

    target = float(np.percentile(z["ep_return"], args.target_pct))
    print(f"=== DT OPE (target return p{args.target_pct:.0f}={target:.2f}, "
          f"ctx={ctx}, rtg_scale={rtg_scale}) ===")

    a_dt = dt_actions_over_data(dt, ds, ep_start, ep_len, rew_raw, target,
                                ctx, rtg_scale, gamma_dt, device)
    # a_next[t] = DT action at the next position within the same episode
    a_next = a_dt.clone()
    a_next[:-1] = a_dt[1:]
    a_s0 = a_dt[ep_start]

    j_pi, j_all = fqe_with_actions(ds, ep_start, ep_len, rew_raw, a_dt, a_next,
                                   a_s0, gamma=args.gamma, steps=args.steps,
                                   device=device)
    g0 = behaviour_returns(rew_raw, ep_start, ep_len, args.gamma)
    delta = j_pi - g0.mean()
    name = os.path.basename(args.run.rstrip("/\\")) or args.run
    print(f"[{name}] algo=dt  J(pi)={j_pi:.3f}  "
          f"J(behaviour)={g0.mean():.3f} (±{g0.std():.2f})  "
          f"Δ={delta:+.3f}  {'IMPROVES' if delta > 0 else 'no gain'}")
    print("Note: DT is return-conditioned on the p%d target; try --target-pct "
          "99 for a more aggressive target. FQE stays pessimistic on OOD "
          "actions." % int(args.target_pct))


if __name__ == "__main__":
    main()
