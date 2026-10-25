"""Decision Transformer (Chen et al., 2021) for the RMUC offline dataset.

An alternative to value-based offline RL that is an excellent fit here: it is
return-conditioned sequence modelling, so it sidesteps Q-learning instability,
handles the partially-observed multi-agent setting by simply conditioning on the
team return-to-go, and is pure transformer training -> trivially DDP-scalable.

At deployment you pick a *target* return-to-go (e.g. the 90th-percentile return
seen in training) and the model rolls out actions to try to achieve it.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .action_spec import ActionSpec, get_spec


class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = nn.MultiheadAttention(n_embd, n_head, dropout=dropout,
                                          batch_first=True)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.GELU(),
            nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout))

    def forward(self, x, attn_mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


class DecisionTransformer(nn.Module):
    def __init__(self, obs_dim, act_dim, ctx=30, n_embd=128, n_layer=3,
                 n_head=4, dropout=0.1, max_timestep=512,
                 spec: ActionSpec | None = None):
        super().__init__()
        self.spec = spec or get_spec("velocity", act_dim)
        act_dim = self.spec.dim
        self.obs_dim, self.act_dim, self.ctx = obs_dim, act_dim, ctx
        self.n_embd, self.n_head = n_embd, n_head
        self.embed_rtg = nn.Linear(1, n_embd)
        self.embed_obs = nn.Linear(obs_dim, n_embd)
        self.embed_act = nn.Linear(act_dim, n_embd)
        self.embed_time = nn.Embedding(max_timestep + 1, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(n_embd)
        if self.spec.is_tactical:
            # mixed-type head, mirroring networks.HybridPolicy: tanh-bounded nav
            # regression + a fire logit + target logits.  A plain Tanh+MSE head
            # would force the categorical target through a Gaussian and bring
            # back exactly the mean-collapse we are trying to remove.
            self.pred_nav = nn.Sequential(
                nn.Linear(n_embd, self.spec.n_cont), nn.Tanh())
            self.pred_fire = nn.Linear(n_embd, 1)
            self.pred_target = nn.Linear(n_embd, self.spec.n_target)
        else:
            self.pred_act = nn.Sequential(nn.Linear(n_embd, act_dim), nn.Tanh())
        self.max_timestep = max_timestep

    def _predict(self, h):
        """Return the [-1,1]/probability-space action vector for every step."""
        if not self.spec.is_tactical:
            return self.pred_act(h)
        return torch.cat([self.pred_nav(h),
                          torch.sigmoid(self.pred_fire(h)),
                          F.softmax(self.pred_target(h), dim=-1)], dim=-1)

    def _loss(self, h, act, m):
        denom = m.sum().clamp(min=1.0)
        if not self.spec.is_tactical:
            pred = self.pred_act(h)
            return ((pred - act).pow(2) * m).sum() / denom
        s = self.spec
        nav = ((self.pred_nav(h) - act[..., s.sl_cont]).pow(2) * m).sum() / denom
        fire = (F.binary_cross_entropy_with_logits(
            self.pred_fire(h).squeeze(-1),
            act[..., s.sl_fire].squeeze(-1).clamp(0.0, 1.0),
            reduction="none").unsqueeze(-1) * m).sum() / denom
        tgt = (-(act[..., s.sl_target]
                 * F.log_softmax(self.pred_target(h), dim=-1)).sum(-1, keepdim=True)
               * m).sum() / denom
        return nav + fire + tgt

    def _tokens(self, obs, act, rtg, timestep):
        B, K, _ = obs.shape
        t = timestep.clamp(max=self.max_timestep)
        time_emb = self.embed_time(t)                       # [B,K,E]
        r = self.embed_rtg(rtg) + time_emb
        s = self.embed_obs(obs) + time_emb
        a = self.embed_act(act) + time_emb
        # interleave -> (r0,s0,a0, r1,s1,a1, ...) : [B, 3K, E]
        tok = torch.stack([r, s, a], dim=2).reshape(B, 3 * K, self.n_embd)
        return self.drop(tok)

    def _attn_mask(self, mask):
        """Additive float mask [B*n_head, L, L].

        Combines a causal mask with a key-padding mask, but *always* leaves the
        diagonal open so no query row is fully masked (a fully-masked softmax
        row would produce NaNs).  Padding tokens sit at the left; masking them
        as keys stops real tokens from ever attending to padding.
        """
        B, K = mask.shape
        L = 3 * K
        dev = mask.device
        causal = torch.triu(torch.ones(L, L, device=dev, dtype=torch.bool),
                            diagonal=1)                    # [L,L] j>i
        pad_key = (mask < 0.5).repeat_interleave(3, dim=1)  # [B,L] True=pad
        block = causal.unsqueeze(0) | pad_key.unsqueeze(1)  # [B,L,L]
        eye = torch.eye(L, device=dev, dtype=torch.bool).unsqueeze(0)
        block = block & ~eye                                # keep self open
        add = torch.zeros(B, L, L, device=dev)
        add.masked_fill_(block, float("-inf"))
        return add.repeat_interleave(self.n_head, dim=0)    # [B*nh,L,L]

    def _hidden(self, obs, act, rtg, timestep, mask):
        x = self._tokens(obs, act, rtg, timestep)
        attn = self._attn_mask(mask)
        for blk in self.blocks:
            x = blk(x, attn_mask=attn)
        return self.ln(x)[:, 1::3, :]                       # obs-token outputs

    def forward(self, batch):
        obs, act, rtg = batch["obs"], batch["act"], batch["rtg"]
        timestep, mask = batch["timestep"], batch["mask"]
        obs_h = self._hidden(obs, act, rtg, timestep, mask)
        loss = self._loss(obs_h, act, mask.unsqueeze(-1))
        return loss, dict(loss=loss.detach())

    @torch.no_grad()
    def act(self, obs, act, rtg, timestep, mask):
        """Return the predicted action for the last (current) step."""
        obs_h = self._hidden(obs, act, rtg, timestep, mask)
        return self._predict(obs_h)[:, -1, :]
