"""Shared network building blocks."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .action_spec import TARGET_MASK, ActionSpec, CONTINUOUS


def mlp(sizes, activation=nn.ReLU, out_activation=None):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.LayerNorm(sizes[i + 1]))
            layers.append(activation())
    if out_activation is not None:
        layers.append(out_activation())
    return nn.Sequential(*layers)


class GaussianPolicy(nn.Module):
    """Diagonal Gaussian policy with state-independent log-std.

    Actions in the dataset live in [-1, 1]; the mean is unbounded during
    training (clipped at eval).  We keep it un-squashed for numerical stability
    of the advantage-weighted log-likelihood used by IQL.
    """

    def __init__(self, obs_dim, act_dim, hidden=256, depth=2,
                 log_std_min=-3.0, log_std_max=2.0):
        super().__init__()
        self.net = mlp([obs_dim] + [hidden] * depth + [act_dim])
        self.log_std = nn.Parameter(torch.zeros(act_dim) * 0.0 - 0.5)
        self.log_std_min, self.log_std_max = log_std_min, log_std_max
        self.act_dim = act_dim

    def dist(self, obs):
        mean = self.net(obs)
        log_std = self.log_std.clamp(self.log_std_min, self.log_std_max)
        std = log_std.exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def log_prob(self, obs, act):
        d = self.dist(obs)
        # dataset actions are in [-1,1]; keep strictly inside for stability
        act = act.clamp(-1 + 1e-6, 1 - 1e-6)
        return d.log_prob(act).sum(-1)

    @torch.no_grad()
    def act(self, obs, deterministic=True):
        d = self.dist(obs)
        a = d.mean if deterministic else d.sample()
        return a.clamp(-1.0, 1.0)


class HybridPolicy(nn.Module):
    """Gaussian(nav) x Bernoulli(fire gate) x Categorical(target).

    The tactical action is genuinely mixed-type, so a single Gaussian is the
    wrong density.  Splitting it into the three natural heads also removes the
    mode-averaging that capped the continuous policies: the Categorical target
    head can represent "either of these two enemies" as a bimodal distribution,
    where an MSE/Gaussian head could only emit the (meaningless) average.

    `log_prob` is the joint log-likelihood.  For the target head the dataset
    label is a *distribution*, so the term is the soft cross-entropy
    sum_c p_c log q_c — the natural extension of a categorical log-likelihood to
    soft labels, and exactly what advantage weighting needs.
    """

    def __init__(self, obs_dim, spec: ActionSpec, hidden=512, depth=3,
                 log_std_min=-3.0, log_std_max=2.0):
        super().__init__()
        self.spec = spec
        self.act_dim = spec.dim
        self.trunk = nn.Sequential(
            mlp([obs_dim] + [hidden] * depth), nn.LayerNorm(hidden), nn.ReLU())
        self.head_nav = nn.Linear(hidden, spec.n_cont)
        self.log_std = nn.Parameter(torch.zeros(spec.n_cont) - 0.5)
        self.head_fire = nn.Linear(hidden, 1)
        self.head_target = nn.Linear(hidden, spec.n_target)
        self.log_std_min, self.log_std_max = log_std_min, log_std_max
        # Non-persistent: old checkpoints do not carry it, and the rule is a
        # property of the action space, not learned weights.
        self.register_buffer("target_mask",
                             torch.tensor(TARGET_MASK, dtype=torch.float32),
                             persistent=False)

    def heads(self, obs):
        h = self.trunk(obs)
        tgt = self.head_target(h)
        if tgt.shape[-1] == self.target_mask.numel():
            tgt = tgt.masked_fill(self.target_mask < 0.5, -1e9)
        return (self.head_nav(h), self.head_fire(h).squeeze(-1), tgt)

    def log_prob(self, obs, act):
        mean, fire_logit, tgt_logits = self.heads(obs)
        s = self.spec
        nav = act[..., s.sl_cont].clamp(-1 + 1e-6, 1 - 1e-6)
        std = self.log_std.clamp(self.log_std_min, self.log_std_max).exp()
        lp = torch.distributions.Normal(mean, std.expand_as(mean)).log_prob(nav).sum(-1)
        gate = act[..., s.sl_fire].squeeze(-1)
        lp = lp - F.binary_cross_entropy_with_logits(
            fire_logit, gate.clamp(0.0, 1.0), reduction="none")
        p_t = act[..., s.sl_target]
        lp = lp + (p_t * F.log_softmax(tgt_logits, dim=-1)).sum(-1)
        return lp

    @torch.no_grad()
    def act(self, obs, deterministic=True):
        mean, fire_logit, tgt_logits = self.heads(obs)
        nav = mean.clamp(-1.0, 1.0)
        p_fire = torch.sigmoid(fire_logit).unsqueeze(-1)
        p_tgt = F.softmax(tgt_logits, dim=-1)
        if deterministic:
            gate = (p_fire > 0.5).float()
        else:
            std = self.log_std.clamp(self.log_std_min, self.log_std_max).exp()
            nav = torch.distributions.Normal(
                mean, std.expand_as(mean)).sample().clamp(-1.0, 1.0)
            gate = torch.bernoulli(p_fire)
        return torch.cat([nav, gate, p_tgt], dim=-1)


def build_policy(obs_dim, spec: ActionSpec, hidden=256, depth=2):
    """Pick the density that matches the action space."""
    if spec.is_tactical:
        return HybridPolicy(obs_dim, spec, hidden, depth)
    return GaussianPolicy(obs_dim, spec.dim, hidden, depth)


class TwinQ(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256, depth=2):
        super().__init__()
        self.q1 = mlp([obs_dim + act_dim] + [hidden] * depth + [1])
        self.q2 = mlp([obs_dim + act_dim] + [hidden] * depth + [1])

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)

    def q_min(self, obs, act):
        q1, q2 = self.forward(obs, act)
        return torch.min(q1, q2)


class ValueNet(nn.Module):
    def __init__(self, obs_dim, hidden=256, depth=2):
        super().__init__()
        self.v = mlp([obs_dim] + [hidden] * depth + [1])

    def forward(self, obs):
        return self.v(obs).squeeze(-1)
