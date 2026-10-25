"""Shared training utilities: DDP setup, config, logging, checkpointing.

Multi-GPU is standard PyTorch DDP driven by `torchrun`.  Every helper degrades
gracefully to a single process (no env vars) so the same script runs on a laptop
CPU/GPU for a smoke test and on the multi-GPU box for the real run.
"""
from __future__ import annotations

import os
import random
import sys
from typing import Iterator

import numpy as np
import torch
import torch.distributed as dist
import yaml


# ---------------------------------------------------------------------------
def ensure_utf8_stdout():
    """Avoid UnicodeEncodeError when printing Chinese on a Windows GBK console."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


def safe_num_workers(n: int) -> int:
    """Always 0: our datasets are already fully-resident in-RAM tensors.

    Workers cannot speed up indexing a tensor that is already in memory, so they
    only ever cost us.  On Windows the DataLoader uses `spawn` and they add
    startup cost and can hang.  On Linux they `fork` *after* CUDA has been
    initialised by DDP, which is undefined behaviour — with one rank per GPU and
    a ~0.5 GB observation matrix per rank this reliably crashed rank 1 with
    SIGSEGV a few seconds into start-up.  Keep the knob in the configs for
    documentation, but do not honour it.
    """
    return 0


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_ddp():
    """Return (device, rank, world_size, local_rank). Works with/without DDP."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local = int(os.environ.get("LOCAL_RANK", 0))
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, rank=rank, world_size=world)
        if torch.cuda.is_available():
            torch.cuda.set_device(local)
            device = torch.device("cuda", local)
        else:
            device = torch.device("cpu")
        return device, rank, world, local
    # single-process fallback
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    return device, 0, 1, 0


def cleanup_ddp():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank: int) -> bool:
    return rank == 0


def seed_all(seed: int, rank: int = 0):
    s = seed + rank
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def reduce_mean(value: torch.Tensor) -> torch.Tensor:
    """Average a scalar tensor across ranks (no-op if not distributed)."""
    if dist.is_available() and dist.is_initialized():
        v = value.clone()
        dist.all_reduce(v, op=dist.ReduceOp.SUM)
        v /= dist.get_world_size()
        return v
    return value


def move_batch(batch: dict, device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


class InfiniteLoader:
    """Yield batches forever, re-seeding the DistributedSampler each epoch."""

    def __init__(self, loader, sampler=None):
        self.loader = loader
        self.sampler = sampler
        self.epoch = 0

    def __iter__(self) -> Iterator:
        while True:
            if self.sampler is not None:
                self.sampler.set_epoch(self.epoch)
            n = 0
            for batch in self.loader:
                n += 1
                yield batch
            if n == 0:
                raise RuntimeError(
                    "DataLoader produced 0 batches in an epoch — batch_size is "
                    "larger than the dataset with drop_last=True. Lower "
                    "train.batch_size or add more data.")
            self.epoch += 1


class CsvLogger:
    """Append-only CSV logger. TF-free, so it works under non-ASCII paths where
    TensorBoard's reader crashes on Windows. Read with pandas.read_csv()."""

    def __init__(self, path):
        import os
        self.path = path
        self.fields = None
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def log(self, row: dict):
        import csv
        import os
        first = not os.path.exists(self.path)
        if self.fields is None:
            self.fields = list(row.keys())
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.fields)
            if first:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in self.fields})


def save_checkpoint(path, model, optimizer, step, config, extra=None):
    state = dict(
        model=(model.module if hasattr(model, "module") else model).state_dict(),
        optimizer=optimizer.state_dict(),
        step=step,
        config=config,
    )
    if extra:
        state.update(extra)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Write through a Python file handle: torch's C++ writer cannot open
    # non-ASCII paths on Windows (e.g. a project dir with Chinese characters).
    with open(path, "wb") as f:
        torch.save(state, f)
