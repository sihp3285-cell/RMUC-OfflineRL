"""Behaviour cloning baselines.

Plain BC clones every valid transition.  Two knobs turn it into stronger
"learn only from the good data" variants without any value function:
  * return weighting  -> weight = exp(beta * zscore(episode_return))   (%BC / AWR-lite)
  * quality weighting -> multiply by the acting team's strength weight
This is the cheapest way to answer "which data is good": prefer transitions
from high-return rounds and strong teams.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .action_spec import ActionSpec, get_spec
from .networks import build_policy


class BC(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256, depth=2,
                 return_beta=0.0, use_quality_weight=False,
                 weight_clip=10.0, spec: ActionSpec | None = None,
                 target_aux: float = 0.0):
        super().__init__()
        self.spec = spec or get_spec("velocity", act_dim)
        self.policy = build_policy(obs_dim, self.spec, hidden, depth)
        self.return_beta = return_beta
        self.use_quality_weight = use_quality_weight
        self.weight_clip = weight_clip
        self.target_aux = target_aux

    def forward(self, batch):
        obs, act, valid = batch["obs"], batch["act"], batch["valid"]
        logp = self.policy.log_prob(obs, act)
        w = valid.clone()
        if self.return_beta > 0.0 and "ret_z" in batch:
            w = w * torch.exp(self.return_beta * batch["ret_z"]).clamp(max=self.weight_clip)
        if self.use_quality_weight:
            w = w * batch["weight"]
        denom = valid.sum().clamp(min=1.0)
        loss = -(w * logp).sum() / denom
        aux_loss = torch.zeros((), device=obs.device)
        if (self.target_aux > 0 and self.spec.is_tactical
                and "target_alt" in batch and hasattr(self.policy, "heads")):
            alt = batch["target_alt"]
            aw = batch["target_alt_w"]
            mask = valid * (aw > 0).float()
            if float(mask.sum()) > 0:
                _, _, tgt_logits = self.policy.heads(obs)
                ce = F.cross_entropy(tgt_logits, alt.argmax(-1), reduction="none")
                aux_loss = (self.target_aux * (ce * aw * mask).sum()
                            / (aw * mask).sum().clamp(min=1.0))
                loss = loss + aux_loss
        # mse of the greedy action, for monitoring only
        with torch.no_grad():
            mse = (self.policy.act(obs, deterministic=True) - act).pow(2).mean()
        return loss, dict(loss=loss.detach(), bc_mse=mse,
                          aux_loss=aux_loss.detach())

    @torch.no_grad()
    def act(self, obs, deterministic=True):
        return self.policy.act(obs, deterministic)
