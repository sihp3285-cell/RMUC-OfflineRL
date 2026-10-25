"""Does the policy behave more like the teams that WON?

Fitted-Q evaluation cannot answer "is this policy better than the logged one".
Its Q function is fitted on the logged state-action distribution, so any policy
that acts *differently* is scored off-support and penalised by the twin-Q min —
regardless of whether the difference is an improvement.  Deviation is exactly
what we are trying to produce, so FQE marks us down for succeeding.

This is a different, weaker, but *unbiased-in-the-right-direction* question.
Every episode carries a win/loss label.  For each logged state we ask what the
policy would have done and score how well that matches what actually happened,
then compare that agreement on **winning** episodes against **losing** ones:

    lift = agreement(won) - agreement(lost)

If the policy's proposals line up with winners more than with losers, it has
learned something that co-varies with winning.  A policy that merely copies the
average behaviour scores ~0 lift; a policy that mimics losing habits scores
negative.  Nothing here rewards or punishes deviation per se, so a policy is
free to differ from the log without being penalised for it.

Read it as evidence, not proof: this is a correlational test on observational
data, so it cannot establish that the policy *causes* wins.  A closed loop is
still the only way to settle that.

    python -m rm_rl.eval.win_alignment --data data/sentry_tactical \\
        --run rm_runs/infantry_iql_tactical
"""
from __future__ import annotations

import argparse
import math
import os

import numpy as np
import torch

from ..data.dataset import TransitionDataset, Normalizer, load_meta
from ..deploy import load_policy
from ..train.common import ensure_utf8_stdout
from .ope_fqe import ObsRenormalizer, action_rescale, _policy_actions


def _welch(a: np.ndarray, b: np.ndarray):
    """Welch t-statistic and a normal-approximation two-sided p-value."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan"), float("nan")
    va, vb = a.var(ddof=1) / na, b.var(ddof=1) / nb
    se = np.sqrt(va + vb)
    if se < 1e-12:
        return float("nan"), float("nan")
    t = (a.mean() - b.mean()) / se
    # two-sided p under a normal approximation (n is in the hundreds here)
    p = math.erfc(abs(t) / math.sqrt(2.0))
    return float(t), float(p)


def run(data_dir: str, run_dir: str, split: str = "train", device=None, verbose=True):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ds = TransitionDataset(data_dir, split)
    z = np.load(os.path.join(data_dir, f"{split}.npz"))
    ep_start, ep_len, ep_win = z["ep_start"], z["ep_len"], z["ep_win"]
    meta = load_meta(data_dir)

    policy, policy_norm, info = load_policy(run_dir, device=device)
    spec = info.get("spec")
    rescale = action_rescale(info.get("act_scale"), meta.get("act_scale"),
                             spec, device)
    renorm = ObsRenormalizer.make(Normalizer.load(data_dir), policy_norm, device)
    if verbose and (rescale is not None or renorm is not None):
        print("  [cross-domain] policy I/O translated into this dataset's units")

    obs = ds.obs.to(device)
    a_pi = _policy_actions(policy, obs, device, rescale=rescale, renorm=renorm)
    a_log = ds.act.to(device)
    valid = (ds.valid.to(device) > 0)

    # --- confound control -------------------------------------------------
    # A losing team's sentry is under pressure, damaged or retreating, so its
    # behaviour is simply *less predictable* than a winning team's.  Any
    # well-fitted policy will therefore agree more with winners, whether or not
    # it learned anything about winning.  Quantify that by scoring a trivial
    # state-independent policy (the dataset's modal action) through exactly the
    # same pipeline: whatever lift IT gets is the confound floor, and only lift
    # above that floor can be attributed to the policy.
    a_const = a_log[valid].median(dim=0).values.expand_as(a_log).contiguous()

    # --- per-transition agreement, one column per decision channel ----------
    def channels(a):
        c = {}
        if spec is not None and spec.is_tactical:
            p_nav, l_nav = a[:, spec.sl_cont], a_log[:, spec.sl_cont]
            # direction agreement, ignoring magnitude: what a nav goal means
            cos = torch.nn.functional.cosine_similarity(p_nav, l_nav, dim=-1, eps=1e-6)
            moving = l_nav.norm(dim=-1) > 0.05      # a stationary step has no direction
            c["nav_direction"] = (cos, valid & moving)
            c["fire_gate"] = (
                (a[:, spec.sl_fire].squeeze(-1).round()
                 == a_log[:, spec.sl_fire].squeeze(-1).round()).float(), valid)
            c["target_choice"] = (
                (a[:, spec.sl_target].argmax(-1)
                 == a_log[:, spec.sl_target].argmax(-1)).float(), valid)
        else:
            c["action_match"] = (-(a - a_log).pow(2).mean(-1), valid)
        return c

    cols = channels(a_pi)
    ctrl = channels(a_const)

    # --- aggregate to episodes, then split by outcome ----------------------
    print(f"\n=== win-alignment: {os.path.basename(run_dir.rstrip('/'))} "
          f"on {data_dir} [{split}] ===")
    if split == "train":
        print("  [warn] train split: this policy was fitted on these episodes; "
              "use --split val for a held-out check")
    print(f"episodes: {len(ep_start)}  "
          f"(won {int(ep_win.sum())} / lost {int((1 - ep_win).sum())})")
    def lift_of(score, mask):
        per_ep = []
        for s, ln in zip(ep_start.tolist(), ep_len.tolist()):
            m = mask[s:s + ln]
            per_ep.append(float(score[s:s + ln][m].mean()) if m.any() else np.nan)
        per_ep = np.array(per_ep)
        ok = ~np.isnan(per_ep)
        won = per_ep[ok & (ep_win[:len(per_ep)] > 0.5)]
        lost = per_ep[ok & (ep_win[:len(per_ep)] <= 0.5)]
        t, p = _welch(won, lost)
        return won.mean(), lost.mean(), won.mean() - lost.mean(), t, p

    print(f"\n{'channel':<16}{'won':>9}{'lost':>9}{'lift':>9}{'ctrl_lift':>11}"
          f"{'excess':>9}{'t':>8}{'p':>9}")
    out = {}
    for name in cols:
        w, l, lift, t, p = lift_of(*cols[name])
        _, _, c_lift, _, _ = lift_of(*ctrl[name])
        out[name] = dict(won=float(w), lost=float(l), lift=float(lift),
                         control_lift=float(c_lift),
                         excess=float(lift - c_lift), t=t, p=p)
        print(f"{name:<16}{w:>9.4f}{l:>9.4f}{lift:>+9.4f}{c_lift:>+11.4f}"
              f"{lift - c_lift:>+9.4f}{t:>8.2f}{p:>9.4f}")

    print("\nctrl_lift = the same lift for a trivial state-independent policy "
          "(the dataset's modal action).\nIt captures how much of the win/lose gap "
          "comes merely from losing episodes being less\npredictable. Only "
          "`excess` = lift - ctrl_lift can be credited to the policy; a large\n"
          "lift with a similarly large ctrl_lift means the metric is measuring the "
          "confound, not skill.\nCorrelational evidence on logged data either way — "
          "not proof of causation.")
    return out


def main():
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", choices=("train", "val"), default="train",
                    help="train (default, in-sample) or val (held-out)")
    args = ap.parse_args()
    run(args.data, args.run, split=args.split)


if __name__ == "__main__":
    main()
