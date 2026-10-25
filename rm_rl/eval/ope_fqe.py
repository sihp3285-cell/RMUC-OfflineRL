"""Offline Policy Evaluation via Fitted Q-Evaluation (FQE).

This answers the question the training curves cannot: *is the learned policy
actually better than the behaviour in the data?*  We never touch a simulator.

FQE fits Q^pi for the (deterministic) policy under evaluation directly on the
logged transitions:
    Q(s,a) <- r + gamma (1-done) Q_target(s', pi(s'))
then estimates the policy's value at the true initial states,
    J(pi) = E_{s0}[ Q^pi(s0, pi(s0)) ] ,
and compares it against the Monte-Carlo discounted return the *behaviour* policy
actually achieved from those same initial states.  If J(pi) > J(behaviour) by a
margin larger than the run-to-run noise, the policy is estimated to improve.

Rewards here are the RAW (un-scaled) shaped rewards, so the numbers are directly
comparable to the behaviour return.

Usage:
    python -m rm_rl.eval.ope_fqe --data data/sentry --run rm_runs/sentry_iql
    python -m rm_rl.eval.ope_fqe --data data/sentry --run rm_runs/sentry_iql --all-ckpts
"""
from __future__ import annotations

import argparse
import copy
import glob
import os

import numpy as np
import torch

from ..algos.networks import TwinQ
from ..data.dataset import TransitionDataset, Normalizer
from ..deploy import load_policy
from ..train.common import ensure_utf8_stdout


def behaviour_returns(rew, ep_start, ep_len, gamma):
    """Monte-Carlo discounted return from each episode's initial state."""
    g0 = np.zeros(len(ep_start), dtype=np.float64)
    for i, (s, ln) in enumerate(zip(ep_start, ep_len)):
        r = rew[s:s + ln]
        disc = gamma ** np.arange(ln)
        g0[i] = float(np.sum(disc * r))
    return g0


def action_rescale(policy_scale, data_scale, spec, device):
    """Factor converting the policy's normalised actions into the eval set's.

    Continuous action channels are stored divided by a per-dataset `act_scale`
    (the 99th percentile of |action|), so "1.0" means a different number of
    metres in every dataset.  Evaluating an infantry-trained policy on sentry
    data without correcting for that feeds ~7.5 m-scaled commands into a Q
    function whose actions are ~5.1 m-scaled — a ~47% unit error that shows up
    as a (spuriously) terrible score.  Convert via physical units:
    a_eval = a_policy * scale_policy / scale_eval.
    Returns None when the scales already agree (the usual same-dataset case).
    """
    if policy_scale is None or data_scale is None:
        return None
    p = np.asarray(policy_scale, np.float32)
    d = np.asarray(data_scale, np.float32)
    if p.shape != d.shape or np.allclose(p, d, rtol=1e-4):
        return None
    r = np.ones_like(p)
    n = spec.n_cont if spec is not None else len(p)
    r[:n] = p[:n] / np.maximum(d[:n], 1e-6)      # only the physical channels
    return torch.as_tensor(r, dtype=torch.float32, device=device)


class ObsRenormalizer:
    """Re-express observations from the eval set's z-scoring into the policy's.

    The dataset is z-scored with its *own* mean/std, but a policy trained on a
    different robot type learned against a different mean/std.  Feeding it the
    eval set's normalisation is a silent input error — the same class of mistake
    as the action-scale one, and bigger: it shifts most input columns.  Undo one
    normalisation and apply the other:  x_pol = (x_eval * s_e + m_e - m_p) / s_p
    Returns None when the two agree (the usual same-dataset case).
    """

    def __init__(self, m_e, s_e, m_p, s_p, device):
        self.m_e = torch.as_tensor(m_e, dtype=torch.float32, device=device)
        self.s_e = torch.as_tensor(s_e, dtype=torch.float32, device=device)
        self.m_p = torch.as_tensor(m_p, dtype=torch.float32, device=device)
        self.s_p = torch.as_tensor(s_p, dtype=torch.float32, device=device)

    def __call__(self, obs):
        return (obs * self.s_e + self.m_e - self.m_p) / self.s_p

    @staticmethod
    def make(eval_norm, policy_norm, device):
        if eval_norm is None or policy_norm is None:
            return None
        m_e, s_e = eval_norm.mean.cpu().numpy(), eval_norm.std.cpu().numpy()
        m_p, s_p = policy_norm.mean.cpu().numpy(), policy_norm.std.cpu().numpy()
        if m_e.shape != m_p.shape:
            raise ValueError(f"obs dim mismatch: eval {m_e.shape} vs policy {m_p.shape}")
        if np.allclose(m_e, m_p, atol=1e-6) and np.allclose(s_e, s_p, rtol=1e-4):
            return None
        return ObsRenormalizer(m_e, s_e, m_p, s_p, device)


@torch.no_grad()
def _policy_actions(policy, obs, device, bs=8192, rescale=None, renorm=None):
    out = torch.empty((obs.shape[0], policy_act_dim(policy)), device=device)
    for i in range(0, obs.shape[0], bs):
        o = obs[i:i + bs].to(device)
        if renorm is not None:
            o = renorm(o)
        a = policy.act(o, deterministic=True)
        if rescale is not None:
            a = (a * rescale).clamp(-1.0, 1.0)
        out[i:i + bs] = a
    return out


def policy_act_dim(policy):
    # policy.act returns actions in the dataset's action space; ask the spec
    spec = getattr(policy, "spec", None)
    if spec is not None:
        return spec.dim
    return policy.policy.act_dim if hasattr(policy, "policy") else 4


def fqe(data_dir, run, steps=40000, gamma=0.99, lr=3e-4, batch=1024,
        hidden=256, depth=2, polyak=0.005, device=None, seed=0, verbose=True):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    ds = TransitionDataset(data_dir, "train")
    z = np.load(os.path.join(data_dir, "train.npz"))
    ep_start, ep_len = z["ep_start"], z["ep_len"]

    obs = ds.obs.to(device)
    act = ds.act.to(device)
    rew = ds.rew.to(device)
    next_obs = ds.next_obs.to(device)
    done = ds.done.to(device)
    N = obs.shape[0]

    policy, policy_norm, info = load_policy(run, device=device)

    # A policy trained on one robot type carries its own observation z-scoring
    # AND its own action scale.  Both must be translated before it can be scored
    # on another type's data, or we silently feed it wrong units on the way in
    # and read wrong units on the way out.
    from ..data.dataset import load_meta
    try:
        data_scale = load_meta(data_dir).get("act_scale")
    except Exception:
        data_scale = None
    rescale = action_rescale(info.get("act_scale"), data_scale,
                             info.get("spec"), device)
    renorm = ObsRenormalizer.make(Normalizer.load(data_dir), policy_norm, device)
    if verbose and (rescale is not None or renorm is not None):
        print("  [cross-domain] translating policy I/O to this dataset:")
        if renorm is not None:
            print("      observations: re-normalised to the policy's train stats")
        if rescale is not None:
            print(f"      actions:      x{[round(float(v),3) for v in rescale[:2]]}"
                  f" on the nav channels")

    # precompute pi(s') once (policy is fixed during FQE)
    a_next = _policy_actions(policy, next_obs, device, rescale=rescale, renorm=renorm)

    qf = TwinQ(ds.obs_dim, ds.act_dim, hidden, depth).to(device)
    qt = copy.deepcopy(qf)
    for p in qt.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(qf.parameters(), lr=lr)

    for step in range(1, steps + 1):
        idx = torch.randint(0, N, (batch,), device=device)
        s, a = obs[idx], act[idx]
        with torch.no_grad():
            y = rew[idx] + gamma * (1 - done[idx]) * qt.q_min(next_obs[idx], a_next[idx])
        q1, q2 = qf(s, a)
        loss = ((q1 - y) ** 2 + (q2 - y) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        for p, tp in zip(qf.parameters(), qt.parameters()):
            tp.data.mul_(1 - polyak).add_(polyak * p.data)
        if verbose and step % 10000 == 0:
            print(f"  FQE step {step}/{steps} bellman_loss {float(loss.detach()):.4f}")

    # J(pi) at initial states
    s0 = obs[ep_start]
    with torch.no_grad():
        a0 = _policy_actions(policy, s0, device, rescale=rescale, renorm=renorm)
        j_pi_s0 = float(qf.q_min(s0, a0).mean())
        # value over all states (secondary, denser estimate)
        a_all = _policy_actions(policy, obs, device, rescale=rescale, renorm=renorm)
        j_pi_all = float(qf.q_min(obs, a_all).mean())

    g0 = behaviour_returns(z["rew"], ep_start, ep_len, gamma)
    return dict(
        algo=info["algo"],
        j_pi_initstates=j_pi_s0,
        j_pi_allstates=j_pi_all,
        j_behaviour_mc=float(g0.mean()),
        j_behaviour_std=float(g0.std()),
        n_episodes=len(ep_start),
    )


def main():
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--run", required=True, help="run dir or .pt checkpoint")
    ap.add_argument("--all-ckpts", action="store_true",
                    help="evaluate every ckpt_*.pt + final.pt in the run dir")
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    targets = [args.run]
    if args.all_ckpts and os.path.isdir(args.run):
        cks = sorted(glob.glob(os.path.join(args.run, "ckpt_*.pt")),
                     key=lambda p: int(p.split("_")[-1].split(".")[0]))
        cks.append(os.path.join(args.run, "final.pt"))
        targets = [c for c in cks if os.path.exists(c)]

    print(f"=== FQE offline policy evaluation (gamma={args.gamma}) ===")
    base = None
    for t in targets:
        r = fqe(args.data, t, steps=args.steps, gamma=args.gamma,
                seed=args.seed, verbose=not args.all_ckpts)
        base = r["j_behaviour_mc"]
        name = os.path.basename(t.rstrip("/\\")) or t
        delta = r["j_pi_initstates"] - r["j_behaviour_mc"]
        print(f"[{name}] algo={r['algo']}  "
              f"J(pi)={r['j_pi_initstates']:.3f}  "
              f"J(behaviour)={r['j_behaviour_mc']:.3f}"
              f" (±{r['j_behaviour_std']:.2f})  "
              f"Δ={delta:+.3f}  {'IMPROVES' if delta > 0 else 'no gain'}")
    print("\nInterpretation: J(pi) is the FQE-estimated discounted return of the "
          "learned policy from the real initial states; J(behaviour) is what the "
          "logged teams actually achieved. Δ>0 (beyond the behaviour spread) = "
          "the policy is estimated to play better. FQE is itself approximate — "
          "treat it as a ranking signal, confirm in closed loop later.")


if __name__ == "__main__":
    main()
