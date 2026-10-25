"""Torch datasets over the pre-built shards.

`TransitionDataset` feeds the value-based / BC learners (IQL, TD3+BC, BC).
`SequenceDataset` feeds the Decision Transformer (return-conditioned sequences).
Both apply the saved observation normalisation so every algorithm sees the same
inputs.
"""
from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class Normalizer:
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = torch.as_tensor(mean, dtype=torch.float32)
        self.std = torch.as_tensor(std, dtype=torch.float32)

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    @classmethod
    def load(cls, data_dir: str) -> "Normalizer":
        z = np.load(os.path.join(data_dir, "norm_stats.npz"))
        return cls(z["obs_mean"], z["obs_std"])


def load_meta(data_dir: str) -> dict:
    with open(os.path.join(data_dir, "meta.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def _load_normalized(z, key: str, mean: np.ndarray, std: np.ndarray,
                     chunk: int = 100_000) -> torch.Tensor:
    """Read one array and z-score it **in place**, in chunks.

    The obvious `((z[key] - mean) / std).astype(np.float32)` materialises three
    full-size temporaries on top of the decompressed array.  At ~0.5 GB per
    observation matrix that is ~2.2 GB of transient per rank, and with one rank
    per GPU all loading at once it was enough to take the process down during
    start-up.  Doing it in place keeps the peak at one copy.
    """
    a = np.ascontiguousarray(z[key], dtype=np.float32)
    for i in range(0, len(a), chunk):
        blk = a[i:i + chunk]
        blk -= mean
        blk /= std
    return torch.from_numpy(a)


class TransitionDataset(Dataset):
    """Normalised (s, a, r, s', done, valid, weight) transitions."""

    def __init__(self, data_dir: str, split: str = "train",
                 normalizer: Optional[Normalizer] = None):
        z = np.load(os.path.join(data_dir, f"{split}.npz"))
        norm = normalizer or Normalizer.load(data_dir)
        mean, std = norm.mean.numpy(), norm.std.numpy()
        self.obs = _load_normalized(z, "obs", mean, std)
        self.next_obs = _load_normalized(z, "next_obs", mean, std)
        self.act = torch.from_numpy(z["act"].astype(np.float32))
        self.rew = torch.from_numpy(z["rew"].astype(np.float32))
        self.done = torch.from_numpy(z["done"].astype(np.float32))
        self.valid = torch.from_numpy(z["valid"].astype(np.float32))
        # per-transition weight = episode quality weight, broadcast
        ep_w = z["ep_weight"].astype(np.float32)
        ep_len = z["ep_len"].astype(np.int64)
        self.weight = torch.from_numpy(np.repeat(ep_w, ep_len))
        # per-transition z-scored episode return (for %BC / return weighting)
        ep_ret = z["ep_return"].astype(np.float32)
        ret_z = (ep_ret - ep_ret.mean()) / (ep_ret.std() + 1e-6)
        self.ret_z = torch.from_numpy(np.repeat(ret_z, ep_len))
        self.obs_dim = self.obs.shape[1]
        self.act_dim = self.act.shape[1]
        # Optional auxiliary target label (BattleScope victim).  Old shards do
        # not have it; zeros make the aux loss a no-op.
        if "target_alt" in z.files:
            self.target_alt = torch.from_numpy(z["target_alt"].astype(np.float32))
            self.target_alt_w = torch.from_numpy(z["target_alt_w"].astype(np.float32))
        else:
            self.target_alt = torch.zeros(len(self.obs), max(1, self.act_dim - 3))
            self.target_alt_w = torch.zeros(len(self.obs))

    def __len__(self):
        return self.obs.shape[0]

    def __getitem__(self, i):
        return dict(obs=self.obs[i], act=self.act[i], rew=self.rew[i],
                    next_obs=self.next_obs[i], done=self.done[i],
                    valid=self.valid[i], weight=self.weight[i],
                    ret_z=self.ret_z[i],
                    target_alt=self.target_alt[i],
                    target_alt_w=self.target_alt_w[i])


class SequenceDataset(Dataset):
    """Fixed-length windows for Decision Transformer.

    Each item is a length-`ctx` window ending at a random step of an episode,
    left-padded when near the episode start.  Returns-to-go are computed with
    the configured discount and scaled by `rtg_scale`.
    """

    def __init__(self, data_dir: str, split: str = "train", ctx: int = 30,
                 gamma: float = 1.0, rtg_scale: float = 100.0,
                 normalizer: Optional[Normalizer] = None):
        z = np.load(os.path.join(data_dir, f"{split}.npz"))
        norm = normalizer or Normalizer.load(data_dir)
        mean, std = norm.mean.numpy(), norm.std.numpy()
        self.obs = _load_normalized(z, "obs", mean, std).numpy()
        self.act = z["act"].astype(np.float32)
        self.rew = z["rew"].astype(np.float32)
        self.ep_start = z["ep_start"].astype(np.int64)
        self.ep_len = z["ep_len"].astype(np.int64)
        self.ctx = ctx
        self.rtg_scale = rtg_scale
        self.obs_dim = self.obs.shape[1]
        self.act_dim = self.act.shape[1]

        # discounted returns-to-go per transition (within episode)
        self.rtg = np.zeros_like(self.rew)
        for s, ln in zip(self.ep_start, self.ep_len):
            run = 0.0
            for k in range(ln - 1, -1, -1):
                run = self.rew[s + k] + gamma * run
                self.rtg[s + k] = run
        # sampling anchor per episode (one window per episode per epoch)
        self.index = list(zip(self.ep_start.tolist(), self.ep_len.tolist()))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        s, ln = self.index[i]
        # random end step in [0, ln-1]
        end = np.random.randint(0, ln)
        start = max(0, end - self.ctx + 1)
        idx = np.arange(start, end + 1) + s
        n = len(idx)
        pad = self.ctx - n

        obs = np.zeros((self.ctx, self.obs_dim), np.float32)
        act = np.zeros((self.ctx, self.act_dim), np.float32)
        rtg = np.zeros((self.ctx, 1), np.float32)
        ts = np.zeros((self.ctx,), np.int64)
        mask = np.zeros((self.ctx,), np.float32)

        obs[pad:] = self.obs[idx]
        act[pad:] = self.act[idx]
        rtg[pad:, 0] = self.rtg[idx] / self.rtg_scale
        ts[pad:] = np.arange(start, end + 1)
        mask[pad:] = 1.0
        return dict(obs=torch.from_numpy(obs), act=torch.from_numpy(act),
                    rtg=torch.from_numpy(rtg), timestep=torch.from_numpy(ts),
                    mask=torch.from_numpy(mask))
