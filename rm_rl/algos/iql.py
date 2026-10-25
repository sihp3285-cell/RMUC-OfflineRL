"""Implicit Q-Learning (Kostrikov et al., 2021) for the RMUC offline dataset.

IQL is the primary algorithm: it never queries the Q-function on out-of-support
actions, which makes it stable on purely offline logs like this one.  The whole
agent is a single `nn.Module` whose `forward` returns the summed loss, so it
wraps cleanly in `DistributedDataParallel` for multi-GPU training.

  V:  expectile regression towards min(Q_target(s,a))
  Q:  regress towards r + gamma (1-done) V(s')
  pi: advantage-weighted regression, weight = exp(beta * (Q_target - V))
"""
from __future__ import annotations

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .action_spec import ActionSpec, get_spec
from .networks import TwinQ, ValueNet, build_policy


class IQL(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256, depth=2,
                 gamma=0.99, expectile=0.7, beta=3.0, adv_max=100.0,
                 polyak=0.005, use_quality_weight=False, reward_scale=1.0,
                 spec: ActionSpec | None = None, target_aux: float = 0.0):
        super().__init__()
        self.spec = spec or get_spec("velocity", act_dim)
        act_dim = self.spec.dim
        self.policy = build_policy(obs_dim, self.spec, hidden, depth)
        self.qf = TwinQ(obs_dim, act_dim, hidden, depth)
        self.vf = ValueNet(obs_dim, hidden, depth)
        self.qf_target = copy.deepcopy(self.qf)
        for p in self.qf_target.parameters():
            p.requires_grad_(False)

        self.gamma = gamma
        self.expectile = expectile
        self.beta = beta
        self.adv_max = adv_max
        self.polyak = polyak
        self.use_quality_weight = use_quality_weight
        # scales rewards (and thus value targets) to ~unit range; keeps Q/V from
        # slowly inflating over a long offline run
        self.reward_scale = reward_scale
        self.target_aux = target_aux

    # ------------------------------------------------------------------
    def forward(self, batch):
        obs, act = batch["obs"], batch["act"]
        rew, done = batch["rew"], batch["done"]
        next_obs, valid = batch["next_obs"], batch["valid"]

        # ---- value: expectile regression ----
        with torch.no_grad():
            q_targ = self.qf_target.q_min(obs, act)
        v = self.vf(obs)
        diff = q_targ - v
        w = torch.where(diff > 0, self.expectile, 1.0 - self.expectile)
        v_loss = (w * diff.pow(2)).mean()

        # ---- Q: TD towards r + gamma (1-done) V(s') ----
        with torch.no_grad():
            v_next = self.vf(next_obs)
            y = self.reward_scale * rew + self.gamma * (1.0 - done) * v_next
        q1, q2 = self.qf(obs, act)
        q_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        # ---- policy: advantage-weighted regression on valid steps ----
        with torch.no_grad():
            adv = q_targ - v
            weight = torch.exp(self.beta * adv).clamp(max=self.adv_max)
            if self.use_quality_weight:
                weight = weight * batch["weight"]
        logp = self.policy.log_prob(obs, act)
        m = valid
        denom = m.sum().clamp(min=1.0)
        policy_loss = -(weight * logp * m).sum() / denom

        # Auxiliary victim loss: on high-confidence attributed hits, teach the
        # target head the robot that was actually hit, not only the geometric
        # soft label.  Zero on every other transition.
        aux_loss = torch.zeros((), device=obs.device)
        if (self.target_aux > 0 and self.spec.is_tactical
                and "target_alt" in batch and hasattr(self.policy, "heads")):
            alt = batch["target_alt"]
            aw = batch["target_alt_w"]
            mask = m * (aw > 0).float()
            if float(mask.sum()) > 0:
                _, _, tgt_logits = self.policy.heads(obs)
                ce = F.cross_entropy(tgt_logits, alt.argmax(-1), reduction="none")
                aux_loss = (self.target_aux * (ce * aw * mask).sum()
                            / (aw * mask).sum().clamp(min=1.0))
                policy_loss = policy_loss + aux_loss

        loss = v_loss + q_loss + policy_loss
        metrics = dict(
            loss=loss.detach(), v_loss=v_loss.detach(), q_loss=q_loss.detach(),
            policy_loss=policy_loss.detach(), v_mean=v.mean().detach(),
            q_mean=q1.mean().detach(), adv_mean=adv.mean().detach(),
            weight_mean=weight.mean().detach(),
            aux_loss=aux_loss.detach(),
        )
        return loss, metrics

    @torch.no_grad()
    def update_target(self):
        for p, tp in zip(self.qf.parameters(), self.qf_target.parameters()):
            tp.data.mul_(1 - self.polyak).add_(self.polyak * p.data)

    @torch.no_grad()
    def act(self, obs, deterministic=True):
        return self.policy.act(obs, deterministic)
