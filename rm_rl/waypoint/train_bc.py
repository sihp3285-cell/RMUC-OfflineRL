from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from rm_rl.waypoint.features import FEATURE_DIM
from rm_rl.waypoint.model import WaypointMLP


def load_shards(paths):
    features, labels, goals, ego, time = [], [], [], [], []
    for path in paths:
        with np.load(path) as data:
            features.append(data["features"])
            labels.append(data["labels"])
            goals.append(data["goals"])
            ego.append(data["ego"])
            time.append(data["time"])
    return (np.concatenate(features), np.concatenate(labels), np.concatenate(goals),
            np.concatenate(ego), np.concatenate(time))


def metrics(pred_m, goals_m):
    err = np.linalg.norm(pred_m - goals_m, axis=1)
    return {
        "mean_m": float(err.mean()),
        "median_m": float(np.median(err)),
        "p90_m": float(np.percentile(err, 90)),
        "p99_m": float(np.percentile(err, 99)),
        "within_1m": float((err <= 1.0).mean()),
        "within_2m": float((err <= 2.0).mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--out", default="rm_runs/waypoint_bc")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    files = []
    for entry in args.data:
        files.extend(sorted(Path(entry).glob("match-*.npz")))
    if not files:
        raise SystemExit("no match-*.npz found")
    val_files = files[::args.val_every]
    train_files = [path for path in files if path not in set(val_files)]
    if not train_files:
        train_files, val_files = files, files
    print(json.dumps({"files": len(files), "train": len(train_files), "val": len(val_files),
                      "device": device}, ensure_ascii=False))

    train = load_shards(train_files)
    val = load_shards(val_files)
    x_train, y_train, goal_train, ego_train, _ = train
    x_val, y_val, goal_val, ego_val, _ = val
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train),
                             torch.from_numpy(goal_train))
    val_x = torch.from_numpy(x_val).to(device)
    val_y = torch.from_numpy(y_val).to(device)
    val_goal = torch.from_numpy(goal_val).to(device)
    loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=False)

    model = WaypointMLP(FEATURE_DIM, args.hidden, args.depth).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "norm_stats.npz", mean=mean, std=std)

    def evaluate():
        model.eval()
        with torch.no_grad():
            pred = model(val_x)
            loss = F.mse_loss(pred, val_y).item()
            pred_m = pred.cpu().numpy() * np.array([28.0, 15.0], dtype=np.float32)
        return loss, pred_m

    best = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for xb, yb, _ in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(xb)
            count += len(xb)
        scheduler.step()
        val_loss, pred_m = evaluate()
        val_metrics = metrics(pred_m, goal_val)
        row = {"epoch": epoch, "train_mse": total / max(1, count), "val_mse": val_loss, **val_metrics}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if best is None or val_metrics["median_m"] < best["metrics"]["median_m"]:
            best = {"epoch": epoch, "metrics": val_metrics, "state": model.state_dict()}
            torch.save({"state_dict": model.state_dict(), "obs_dim": FEATURE_DIM,
                        "hidden": args.hidden, "depth": args.depth,
                        "mean": mean, "std": std, "metrics": val_metrics,
                        "epoch": epoch}, out / "best.pt")

    torch.save({"state_dict": model.state_dict(), "obs_dim": FEATURE_DIM,
                "hidden": args.hidden, "depth": args.depth,
                "mean": mean, "std": std, "metrics": best["metrics"],
                "epoch": best["epoch"]}, out / "final.pt")

    # Constant baselines on the validation split.
    central = np.tile(np.array([[10.8, 7.5]], dtype=np.float32), (len(goal_val), 1))
    center = np.tile(np.array([[14.0, 7.5]], dtype=np.float32), (len(goal_val), 1))
    report = {
        "best_epoch": best["epoch"],
        "best": best["metrics"],
        "baseline_central": metrics(central, goal_val),
        "baseline_center": metrics(center, goal_val),
        "baseline_ego": metrics(ego_val, goal_val),
        "samples": {"train": int(len(x_train)), "val": int(len(x_val))},
        "history": history,
        "config": vars(args),
    }
    (out / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("best_epoch", "best", "baseline_central",
                                             "baseline_center", "baseline_ego")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
