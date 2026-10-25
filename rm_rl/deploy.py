"""Load a trained policy and turn its output back into a robot command.

This closes the loop between the abstract action the policy emits and what the
on-board controllers expect.  The policy always thinks in the *canonical* (red)
field frame; for a blue robot we invert the 180-degree mirror.

Physical action layout (after de-normalisation):
    vx, vy      desired planar velocity in the field frame   [m/s]
    yaw_rate    turret heading rate                          [deg/s]
    fire        17 mm rounds to fire this second             [rounds/s]

These are *set-points* for the classical navigation / gimbal / shooter loops,
sampled at 1 Hz — not motor commands.
"""
from __future__ import annotations

import json
import os
import numpy as np
import torch

from .algos.action_spec import AERIAL_TARGET_SLOT, NO_TARGET, get_spec
from .algos.iql import IQL
from .algos.bc import BC
from .algos.dt import DecisionTransformer
from .data.dataset import Normalizer


def _find(path, name):
    return path if os.path.isfile(path) else os.path.join(path, name)


def load_policy(run_dir_or_ckpt: str, device="cpu"):
    """Return (core_model, normalizer, info). Accepts a run dir or a .pt file."""
    if os.path.isdir(run_dir_or_ckpt):
        # Prefer the best *validation* checkpoint. These runs overfit within a
        # few thousand steps, so final.pt is reliably the worst model in the
        # directory — loading it by default silently evaluates the wrong thing.
        ckpt = os.path.join(run_dir_or_ckpt, "best.pt")
        if not os.path.exists(ckpt):
            ckpt = os.path.join(run_dir_or_ckpt, "final.pt")
        norm_dir = run_dir_or_ckpt
    else:
        ckpt = run_dir_or_ckpt
        norm_dir = os.path.dirname(ckpt)
    state = torch.load(ckpt, map_location=device)
    algo = state.get("algo", "iql")
    obs_dim, act_dim = state["obs_dim"], state["act_dim"]
    action_mode = state.get("action_mode", "velocity")
    cfg = state.get("config", {}) or {}
    m = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    hidden, depth = m.get("hidden", 256), m.get("depth", 2)
    spec = get_spec(action_mode, act_dim)
    if algo == "iql":
        model = IQL(obs_dim, act_dim, hidden, depth, spec=spec)
    elif algo == "bc":
        model = BC(obs_dim, act_dim, hidden, depth, spec=spec)
    elif algo == "dt":
        dp = cfg.get("dt", {}) if isinstance(cfg, dict) else {}
        model = DecisionTransformer(
            obs_dim, act_dim, ctx=state.get("ctx", 30),
            n_embd=dp.get("n_embd", 128), n_layer=dp.get("n_layer", 3),
            n_head=dp.get("n_head", 4), dropout=dp.get("dropout", 0.1),
            max_timestep=dp.get("max_timestep", 512), spec=spec)
    else:
        raise ValueError(algo)
    model.load_state_dict(state["model"])
    model.to(device).eval()
    normalizer = Normalizer.load(norm_dir).to(device)
    info = dict(algo=algo, act_scale=np.asarray(state["act_scale"], np.float32),
                obs_dim=obs_dim, act_dim=act_dim, action_mode=action_mode,
                spec=spec)
    return model, normalizer, info


def decode_action(a_norm: np.ndarray, act_scale: np.ndarray, camp: str = "红",
                  action_mode: str = "velocity"):
    """Map a policy action in [-1,1] back to a physical field-frame command.

    For ``tactical`` the command is what a real autonomy stack consumes:
    a navigation sub-goal offset, a target *priority* handed to auto-aim, and a
    weapons-free bit.  Auto-aim keeps ownership of the gimbal and the trigger.
    """
    a = np.asarray(a_norm, np.float32) * act_scale
    if action_mode == "tactical":
        spec = get_spec("tactical")
        gx, gy = float(a[..., 0]), float(a[..., 1])
        if camp == "蓝":                 # invert the canonicalisation mirror
            gx, gy = -gx, -gy
        probs = np.asarray(a[..., spec.sl_target], np.float32).copy()
        # Hard deployment guard: the aerial unit is not a valid ground target.
        # This also covers old checkpoints / exported graphs that predate the
        # training-time mask.
        probs[AERIAL_TARGET_SLOT] = 0.0
        psum = float(probs.sum())
        if psum > 1e-9:
            probs /= psum
        else:
            probs[:] = 0.0
            probs[NO_TARGET] = 1.0
        tgt = int(np.argmax(probs))
        return dict(goal_dx=gx, goal_dy=gy,
                    fire=float(a[..., 2] > 0.5),
                    target=None if tgt == NO_TARGET else tgt,
                    target_conf=float(probs[tgt]),
                    target_probs=probs.tolist())
    vx, vy, yaw_rate, fire = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    if camp == "蓝":            # invert the canonicalisation mirror
        vx, vy = -vx, -vy
        yaw_rate = yaw_rate     # a *rate* is mirror-invariant
    fire = np.clip(fire, 0.0, None)
    return dict(vx=float(vx), vy=float(vy), yaw_rate=float(yaw_rate),
                fire=float(fire))


def apply_safety(cmd: dict, *, heat=None, heat_limit=None, ammo_left=None,
                 ego_xy=None, field=(28.0, 15.0), margin=0.5,
                 max_speed=3.5, heat_per_shot=10.0, heat_safety=10.0,
                 goal_horizon=5.0) -> dict:
    """Clamp a policy command to hard safety limits before it hits the robot.

    A learned policy has no built-in respect for referee constraints, so wrap it:
      * cap planar speed at `max_speed` (m/s)
      * stop firing when barrel heat has no headroom, or ammo is out
      * cancel velocity that would push the robot out of the field
    Pass whatever live telemetry you have; unknown args are simply not enforced.
    """
    c = dict(cmd)
    import math
    tactical = "goal_dx" in c
    kx, ky = ("goal_dx", "goal_dy") if tactical else ("vx", "vy")

    # speed / step cap (for a sub-goal this bounds how far ahead we may commit)
    sp = math.hypot(c[kx], c[ky])
    cap = max_speed * (goal_horizon if tactical else 1.0)
    if sp > cap and sp > 1e-6:
        s = cap / sp
        c[kx] *= s
        c[ky] *= s
    # fire gating — for the tactical layout `fire` is a permission bit, so any
    # constraint violation simply revokes permission
    if ammo_left is not None and ammo_left <= 0:
        c["fire"] = 0.0
    if heat is not None and heat_limit is not None:
        headroom = max(heat_limit - heat - heat_safety, 0.0)
        max_shots = headroom / max(heat_per_shot, 1e-6)
        c["fire"] = 0.0 if (tactical and max_shots < 1.0) else (
            c["fire"] if tactical else float(min(c["fire"], max_shots)))
    # keep inside the field: null out motion pointing past a boundary
    if ego_xy is not None:
        x, y = ego_xy
        fx, fy = field
        if (x <= margin and c[kx] < 0) or (x >= fx - margin and c[kx] > 0):
            c[kx] = 0.0
        if (y <= margin and c[ky] < 0) or (y >= fy - margin and c[ky] > 0):
            c[ky] = 0.0
    return c


class MLPPolicyRunner:
    """Convenience wrapper for IQL / BC policies (Markov, no history)."""

    def __init__(self, run_dir, device="cpu", camp="红"):
        self.model, self.norm, self.info = load_policy(run_dir, device)
        self.device, self.camp = device, camp

    @torch.no_grad()
    def step(self, obs_vec: np.ndarray) -> dict:
        o = torch.as_tensor(obs_vec, dtype=torch.float32, device=self.device)
        o = self.norm.normalize(o).unsqueeze(0)
        a = self.model.act(o, deterministic=True)[0].cpu().numpy()
        return decode_action(a, self.info["act_scale"], self.camp,
                             self.info.get("action_mode", "velocity"))


# For Decision Transformer deployment you maintain a rolling buffer of the last
# `ctx` (obs, action, rtg) tokens, decrement the target return-to-go by each
# realised reward, and call `DecisionTransformer.act(...)`. See README section
# "Deploying the Decision Transformer".
