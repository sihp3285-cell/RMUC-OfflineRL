"""DDP training entrypoint for the Decision Transformer.

    torchrun --standalone --nproc_per_node=4 \
        -m rm_rl.train.train_dt --config configs/sentry_dt.yaml
"""
from __future__ import annotations

import argparse
import math
import os
import shutil

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from ..algos.action_spec import spec_from_meta
from ..algos.dt import DecisionTransformer
from ..data.dataset import SequenceDataset, Normalizer, load_meta
from . import common as C


def lr_lambda(step, warmup, total):
    if step < warmup:
        return step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))


@torch.no_grad()
def evaluate(core, loader, device):
    """Held-out loss. Without this the DT trains blind — its training loss falls
    the whole way while the MLP learners on the same data are already overfitting
    badly, so 'loss went down' says nothing about the model you should keep."""
    core.eval()
    tot, n = 0.0, 0
    for batch in loader:
        batch = C.move_batch(batch, device)
        loss, _ = core(batch)
        tot += float(loss)
        n += 1
    core.train()
    return {"val_loss": tot / max(n, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    args = ap.parse_args()

    C.ensure_utf8_stdout()
    cfg = C.load_config(args.config)
    data_dir = args.data or cfg["data"]
    out_dir = args.out or cfg["out"]
    tr = cfg.get("train", {})
    dp = cfg.get("dt", {})
    max_steps = args.max_steps or tr.get("max_steps", 100000)

    device, rank, world, local = C.setup_ddp()
    C.seed_all(cfg.get("seed", 0), rank)
    main_proc = C.is_main(rank)

    meta = load_meta(data_dir)
    normalizer = Normalizer.load(data_dir)
    ctx = dp.get("ctx", 30)
    train_ds = SequenceDataset(data_dir, "train", ctx=ctx,
                               gamma=dp.get("gamma", 1.0),
                               rtg_scale=dp.get("rtg_scale", 100.0),
                               normalizer=normalizer)
    obs_dim, act_dim = train_ds.obs_dim, train_ds.act_dim

    nw = C.safe_num_workers(tr.get("num_workers", 4))
    sampler = DistributedSampler(train_ds, shuffle=True) if world > 1 else None
    loader = DataLoader(train_ds, batch_size=tr.get("batch_size", 128),
                        sampler=sampler, shuffle=(sampler is None),
                        num_workers=nw, pin_memory=True, drop_last=True)
    inf = iter(C.InfiniteLoader(loader, sampler))

    val_loader = None
    if main_proc:
        try:
            val_ds = SequenceDataset(data_dir, "val", ctx=ctx,
                                     gamma=dp.get("gamma", 1.0),
                                     rtg_scale=dp.get("rtg_scale", 100.0),
                                     normalizer=normalizer)
            val_loader = DataLoader(val_ds, batch_size=tr.get("batch_size", 128),
                                    shuffle=False,
                                    num_workers=C.safe_num_workers(2),
                                    pin_memory=True)
        except (FileNotFoundError, RuntimeError):
            pass

    spec = spec_from_meta({**meta, "action_dim": act_dim})
    model = DecisionTransformer(
        obs_dim, act_dim, ctx=ctx, n_embd=dp.get("n_embd", 128),
        n_layer=dp.get("n_layer", 3), n_head=dp.get("n_head", 4),
        dropout=dp.get("dropout", 0.1),
        max_timestep=dp.get("max_timestep", 512), spec=spec).to(device)
    if world > 1:
        model = DDP(model, device_ids=[local] if device.type == "cuda" else None,
                    find_unused_parameters=False)
    core = model.module if isinstance(model, DDP) else model

    opt = torch.optim.AdamW(model.parameters(), lr=tr.get("lr", 1e-4),
                            weight_decay=tr.get("weight_decay", 1e-4))
    warmup = tr.get("warmup", 1000)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_lambda(s, warmup, max_steps))
    grad_clip = tr.get("grad_clip", 0.25)

    writer = train_csv = eval_csv = None
    if main_proc:
        os.makedirs(out_dir, exist_ok=True)
        train_csv = C.CsvLogger(os.path.join(out_dir, "train_log.csv"))
        eval_csv = C.CsvLogger(os.path.join(out_dir, "eval_log.csv"))
        for f in ("norm_stats.npz", "meta.json"):
            src = os.path.join(data_dir, f)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(out_dir, f))
        # record the RTG targets a deployer should try (train return quantiles)
        z = np.load(os.path.join(data_dir, "train.npz"))
        rtg_targets = np.percentile(z["ep_return"], [50, 75, 90, 99]).tolist()
        with open(os.path.join(out_dir, "rtg_targets.txt"), "w") as fh:
            fh.write("episode-return percentiles [50,75,90,99]:\n")
            fh.write(str(rtg_targets) + "\n")
            fh.write(f"rtg_scale={dp.get('rtg_scale',100.0)}\n")
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(os.path.join(out_dir, "tb"))
        except Exception:
            writer = None
        print(f"[rank0] DT obs={obs_dim} act={act_dim} ctx={ctx} "
              f"episodes={len(train_ds)} world={world} device={device}")

    log_every = tr.get("log_every", 200)
    ckpt_every = tr.get("ckpt_every", 10000)
    eval_every = tr.get("eval_every", 2000)
    patience = int(tr.get("early_stop_patience", 0))
    best_score, best_step, since_best = float("inf"), -1, 0

    model.train()
    for step in range(1, max_steps + 1):
        batch = C.move_batch(next(inf), device)
        loss, metrics = model(batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step(); sched.step()

        if main_proc and step % log_every == 0:
            lr = sched.get_last_lr()[0]
            if train_csv:
                train_csv.log({"step": step, "lr": lr,
                               "loss": float(metrics["loss"])})
            if writer:
                writer.add_scalar("train/loss", float(metrics["loss"]), step)
                writer.add_scalar("train/lr", lr, step)
            print(f"step {step}/{max_steps} loss {float(metrics['loss']):.5f}")

        if step % eval_every == 0 and val_loader is not None and main_proc:
            ev = evaluate(core, val_loader, device)
            if eval_csv:
                eval_csv.log({"step": step, **ev})
            if writer:
                writer.add_scalar("val/loss", ev["val_loss"], step)
            print(f"[eval] step {step} val_loss {ev['val_loss']:.5f}")
            if ev["val_loss"] < best_score - 1e-6:
                best_score, best_step, since_best = ev["val_loss"], step, 0
                C.save_checkpoint(
                    os.path.join(out_dir, "best.pt"), model, opt, step, cfg,
                    extra=dict(algo="dt", obs_dim=obs_dim, act_dim=act_dim, ctx=ctx,
                               action_mode=meta.get("action_mode", "velocity"),
                               act_scale=meta["act_scale"], select_score=best_score))
                print(f"        -> new best val_loss={best_score:.5f}, saved best.pt")
            else:
                since_best += 1
                if patience and since_best >= patience:
                    print(f"[early-stop] val_loss flat for {since_best} evals "
                          f"(best {best_score:.5f} @ step {best_step}); stopping.")
                    break

        if step % ckpt_every == 0 and main_proc:
            C.save_checkpoint(os.path.join(out_dir, f"ckpt_{step}.pt"),
                              model, opt, step, cfg,
                              extra=dict(algo="dt", obs_dim=obs_dim,
                                         act_dim=act_dim, ctx=ctx,
                                         action_mode=meta.get("action_mode", "velocity"),
                                         act_scale=meta["act_scale"]))

    if main_proc:
        C.save_checkpoint(os.path.join(out_dir, "final.pt"), model, opt,
                          max_steps, cfg,
                          extra=dict(algo="dt", obs_dim=obs_dim, act_dim=act_dim,
                                     ctx=ctx,
                                     action_mode=meta.get("action_mode", "velocity"),
                                     act_scale=meta["act_scale"]))
        if writer:
            writer.close()
        print("done.")
    C.cleanup_ddp()


if __name__ == "__main__":
    main()
