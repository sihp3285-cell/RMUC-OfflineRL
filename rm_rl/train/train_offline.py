"""DDP training entrypoint for the value-based / BC learners (IQL, BC).

Single GPU / CPU:
    python -m rm_rl.train.train_offline --config configs/sentry_iql.yaml

4x GPU:
    torchrun --standalone --nproc_per_node=4 \
        -m rm_rl.train.train_offline --config configs/sentry_iql.yaml
"""
from __future__ import annotations

import argparse
import math
import os
import shutil

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from ..algos.action_spec import spec_from_meta
from ..algos.bc import BC
from ..algos.iql import IQL
from ..data.dataset import TransitionDataset, Normalizer, load_meta
from . import common as C


def build_model(algo, cfg, obs_dim, act_dim, reward_scale=1.0, spec=None):
    m = cfg.get("model", {})
    hidden, depth = m.get("hidden", 256), m.get("depth", 2)
    if algo == "iql":
        p = cfg.get("iql", {})
        return IQL(obs_dim, act_dim, hidden, depth,
                   gamma=p.get("gamma", 0.99), expectile=p.get("expectile", 0.7),
                   beta=p.get("beta", 3.0), adv_max=p.get("adv_max", 100.0),
                   polyak=p.get("polyak", 0.005),
                   use_quality_weight=p.get("use_quality_weight", False),
                   reward_scale=reward_scale, spec=spec,
                   target_aux=float(cfg.get("train", {}).get("target_aux", 0.0)))
    if algo == "bc":
        p = cfg.get("bc", {})
        return BC(obs_dim, act_dim, hidden, depth,
                  return_beta=p.get("return_beta", 0.0),
                  use_quality_weight=p.get("use_quality_weight", False),
                  spec=spec,
                  target_aux=float(cfg.get("train", {}).get("target_aux", 0.0)))
    raise ValueError(f"unknown algo {algo}")


def lr_lambda(step, warmup, total):
    if step < warmup:
        return step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))


@torch.no_grad()
def evaluate(core, loader, device):
    core.eval()
    agg, n = {}, 0
    spec = getattr(core, "spec", None)
    for batch in loader:
        batch = C.move_batch(batch, device)
        loss, metrics = core(batch)
        pred = core.act(batch["obs"], deterministic=True)
        act, valid = batch["act"], batch["valid"]
        den = valid.sum().clamp(min=1)
        mse = (((pred - act).pow(2).mean(-1)) * valid).sum() / den
        row = {"val_loss": float(loss), "action_mse": float(mse)}
        if spec is not None and spec.is_tactical:
            # A single MSE over nav metres + a gate bit + a 7-way distribution is
            # meaningless, so report each head in its own natural unit.  Both
            # discrete labels are heavily imbalanced (fire is on ~9% of seconds,
            # "no target" is ~64%), so raw accuracy flatters a degenerate model
            # that always predicts the majority class — we log the positive-class
            # F1 and the majority-class rates alongside it.
            nav = (((pred[:, spec.sl_cont] - act[:, spec.sl_cont]).pow(2).mean(-1))
                   * valid).sum() / den
            p_fire = pred[:, spec.sl_fire].squeeze(-1).round()
            y_fire = act[:, spec.sl_fire].squeeze(-1).round()
            tp = ((p_fire * y_fire) * valid).sum()
            fp = ((p_fire * (1 - y_fire)) * valid).sum()
            fn = (((1 - p_fire) * y_fire) * valid).sum()
            f1 = 2 * tp / (2 * tp + fp + fn).clamp(min=1e-6)
            p_t, y_t = pred[:, spec.sl_target].argmax(-1), act[:, spec.sl_target].argmax(-1)
            named = (y_t != spec.n_target - 1).float()          # concrete enemy
            named_hit = (((p_t == y_t).float() * named * valid).sum()
                         / (named * valid).sum().clamp(min=1))
            row.update(
                nav_mse=float(nav),
                fire_acc=float((((p_fire == y_fire).float()) * valid).sum() / den),
                fire_f1=float(f1),
                fire_pos_rate=float((y_fire * valid).sum() / den),
                target_top1=float(((p_t == y_t).float() * valid).sum() / den),
                target_top1_named=float(named_hit),
                target_named_rate=float((named * valid).sum() / den))
        # surface value magnitudes so divergence (exploding Q/V) is visible
        for k in ("q_mean", "v_mean", "adv_mean"):
            if k in metrics:
                row["val_" + k] = float(metrics[k])
        for k, v in row.items():
            agg[k] = agg.get(k, 0.0) + v
        n += 1
    core.train()
    return {k: v / max(n, 1) for k, v in agg.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--algo", default=None, choices=[None, "iql", "bc"])
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true",
                    help="allow writing into a run dir that already has "
                         "best.pt/final.pt (default: refuse, so an existing "
                         "checkpoint is never clobbered by accident)")
    args = ap.parse_args()

    C.ensure_utf8_stdout()
    cfg = C.load_config(args.config)
    data_dir = args.data or cfg["data"]
    out_dir = args.out or cfg["out"]
    algo = args.algo or cfg.get("algo", "iql")
    tr = cfg.get("train", {})
    max_steps = args.max_steps or tr.get("max_steps", 200000)

    device, rank, world, local = C.setup_ddp()
    C.seed_all(cfg.get("seed", 0), rank)
    main_proc = C.is_main(rank)
    if main_proc and not args.overwrite:
        for fname in ("best.pt", "final.pt"):
            existing = os.path.join(out_dir, fname)
            if os.path.exists(existing):
                raise SystemExit(
                    f"refusing to overwrite existing checkpoint: {existing}\n"
                    f"choose a new --out / config['out'], or pass --overwrite")

    meta = load_meta(data_dir)
    normalizer = Normalizer.load(data_dir)
    train_ds = TransitionDataset(data_dir, "train", normalizer)
    obs_dim, act_dim = train_ds.obs_dim, train_ds.act_dim

    nw = C.safe_num_workers(tr.get("num_workers", 4))
    sampler = DistributedSampler(train_ds, shuffle=True) if world > 1 else None
    train_loader = DataLoader(
        train_ds, batch_size=tr.get("batch_size", 1024),
        sampler=sampler, shuffle=(sampler is None),
        num_workers=nw, pin_memory=True, drop_last=True)
    inf = iter(C.InfiniteLoader(train_loader, sampler))

    val_loader = None
    if main_proc:
        try:
            val_ds = TransitionDataset(data_dir, "val", normalizer)
            val_loader = DataLoader(val_ds, batch_size=tr.get("batch_size", 1024),
                                    shuffle=False, num_workers=C.safe_num_workers(2),
                                    pin_memory=True)
        except FileNotFoundError:
            pass

    # auto reward normalisation for IQL: scale rewards to ~unit std so value
    # targets stay bounded over long training (prevents the slow Q/V inflation)
    reward_scale = 1.0
    if algo == "iql":
        rs_cfg = cfg.get("iql", {}).get("reward_scale", "auto")
        if rs_cfg == "auto":
            reward_scale = 1.0 / (float(train_ds.rew.std()) + 1e-6)
        else:
            reward_scale = float(rs_cfg)
        if main_proc:
            print(f"[rank0] reward_scale={reward_scale:.4f}")

    spec = spec_from_meta({**meta, "action_dim": act_dim})
    model = build_model(algo, cfg, obs_dim, act_dim, reward_scale, spec).to(device)
    if world > 1:
        model = DDP(model, device_ids=[local] if device.type == "cuda" else None,
                    find_unused_parameters=False)
    core = model.module if isinstance(model, DDP) else model

    opt = torch.optim.AdamW(model.parameters(), lr=tr.get("lr", 3e-4),
                            weight_decay=tr.get("weight_decay", 0.0))
    warmup = tr.get("warmup", 1000)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_lambda(s, warmup, max_steps))
    grad_clip = tr.get("grad_clip", 1.0)

    writer = train_csv = eval_csv = None
    if main_proc:
        os.makedirs(out_dir, exist_ok=True)
        for f in ("norm_stats.npz", "meta.json"):
            src = os.path.join(data_dir, f)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(out_dir, f))
        # TF-free CSV logs (always work, even under non-ASCII paths)
        train_csv = C.CsvLogger(os.path.join(out_dir, "train_log.csv"))
        eval_csv = C.CsvLogger(os.path.join(out_dir, "eval_log.csv"))
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(os.path.join(out_dir, "tb"))
        except Exception:
            writer = None
        print(f"[rank0] algo={algo} obs={obs_dim} act={act_dim} "
              f"train_transitions={len(train_ds)} world={world} device={device}")

    log_every = tr.get("log_every", 200)
    eval_every = tr.get("eval_every", 5000)
    ckpt_every = tr.get("ckpt_every", 10000)
    # Offline RL on this data overfits *early* — every validation metric peaked
    # at the first eval and then decayed for the rest of a 120k-step run, while
    # the training loss kept falling.  So track the best validation checkpoint
    # and stop once it stops improving; `final.pt` is not the model you want.
    select_metric = tr.get("select_metric", "action_mse")
    patience = int(tr.get("early_stop_patience", 0))       # 0 = never stop
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
        if algo == "iql":
            core.update_target()

        if main_proc and step % log_every == 0:
            lr = sched.get_last_lr()[0]
            row = {"step": step, "lr": lr,
                   **{k: float(v) for k, v in metrics.items()}}
            if train_csv:
                train_csv.log(row)
            if writer:
                for k, v in metrics.items():
                    writer.add_scalar(f"train/{k}", float(v), step)
                writer.add_scalar("train/lr", lr, step)
            print(f"step {step}/{max_steps} loss {float(metrics['loss']):.4f}")

        if step % eval_every == 0 and val_loader is not None and main_proc:
            ev = evaluate(core, val_loader, device)
            if eval_csv:
                eval_csv.log({"step": step, **ev})
            if writer:
                for k, v in ev.items():
                    writer.add_scalar(f"val/{k}", v, step)
            print("[eval] step %d " % step
                  + " ".join(f"{k} {v:.4f}" for k, v in ev.items()))

            score = ev.get(select_metric)
            if score is not None:
                if score < best_score - 1e-6:
                    best_score, best_step, since_best = score, step, 0
                    C.save_checkpoint(
                        os.path.join(out_dir, "best.pt"), model, opt, step, cfg,
                        extra=dict(algo=algo, obs_dim=obs_dim, act_dim=act_dim,
                                   action_mode=meta.get("action_mode", "velocity"),
                                   act_scale=meta["act_scale"],
                                   select_metric=select_metric, select_score=score))
                    print(f"        -> new best {select_metric}={score:.4f}, saved best.pt")
                else:
                    since_best += 1
                    if patience and since_best >= patience:
                        print(f"[early-stop] {select_metric} has not improved for "
                              f"{since_best} evals (best {best_score:.4f} @ step "
                              f"{best_step}); stopping.")
                        break

        if step % ckpt_every == 0 and main_proc:
            C.save_checkpoint(os.path.join(out_dir, f"ckpt_{step}.pt"),
                              model, opt, step, cfg,
                              extra=dict(algo=algo, obs_dim=obs_dim,
                                         act_dim=act_dim,
                                         action_mode=meta.get("action_mode", "velocity"),
                                         act_scale=meta["act_scale"]))

    if main_proc:
        C.save_checkpoint(os.path.join(out_dir, "final.pt"), model, opt,
                          max_steps, cfg,
                          extra=dict(algo=algo, obs_dim=obs_dim, act_dim=act_dim,
                                     action_mode=meta.get("action_mode", "velocity"),
                                     act_scale=meta["act_scale"]))
        if writer:
            writer.close()
        if best_step > 0:
            print(f"done. best {select_metric}={best_score:.4f} at step {best_step} "
                  f"-> best.pt   (evaluate THAT, not final.pt)")
        else:
            print("done.")
    C.cleanup_ddp()


if __name__ == "__main__":
    main()
