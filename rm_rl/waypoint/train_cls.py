from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from rm_rl.waypoint.features import FEATURE_DIM, anchor_features, rule_features, sequence_context
from rm_rl.waypoint.model import WaypointClassifier
from rm_rl.waypoint.train_bc import metrics


def build_anchors(goals, max_anchors=16, merge_m=0.8, min_count=3):
    anchors = []
    counts = []
    for goal in goals:
        best, best_d = -1, 1e9
        for i, anchor in enumerate(anchors):
            d = float(np.linalg.norm(goal - anchor))
            if d < best_d:
                best, best_d = i, d
        if best >= 0 and best_d <= merge_m:
            counts[best] += 1
            anchors[best] = (anchors[best] * (counts[best] - 1) + goal) / counts[best]
        else:
            anchors.append(goal.copy())
            counts.append(1)
    order = sorted(range(len(anchors)), key=lambda i: -counts[i])
    keep = [i for i in order if counts[i] >= min_count][:max_anchors]
    if not keep:
        keep = order[:max_anchors]
    return np.asarray([anchors[i] for i in keep], dtype=np.float32), np.asarray([counts[i] for i in keep], dtype=np.int64)


def assign(goals, anchors):
    d = np.linalg.norm(goals[:, None, :] - anchors[None, :, :], axis=2)
    idx = d.argmin(axis=1)
    return idx, d[np.arange(len(goals)), idx]


def load_sequence_shards(paths, radar=False):
    features, labels, goals, ego = [], [], [], []
    for path in paths:
        with np.load(path) as data:
            features.append(data["features"] if radar else sequence_context(data["features"]))
            labels.append(data["labels"])
            goals.append(data["goals"])
            ego.append(data["ego"])
    return (np.concatenate(features), np.concatenate(labels),
            np.concatenate(goals), np.concatenate(ego))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--out", default="rm_runs/waypoint_cls")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--max-anchors", type=int, default=16)
    parser.add_argument("--merge-m", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    files = []
    for entry in args.data:
        files.extend(sorted(Path(entry).glob("match-*.npz")))
    if not files:
        raise SystemExit("no match-*.npz")
    val_files = files[::args.val_every]
    train_files = [path for path in files if path not in set(val_files)]
    if not train_files or not val_files:
        raise SystemExit("need disjoint training and validation matches")

    schemas = set()
    for path in files:
        with np.load(path) as data:
            schemas.add(str(data["schema"].item()) if "schema" in data else "arena307")
    if len(schemas) != 1:
        raise SystemExit("cannot mix observation schemas")
    schema = schemas.pop()
    from rm_rl.waypoint.radar import SCHEMA, ANCHORS, FEATURE_DIM as RADAR_DIM, contract
    radar = schema == SCHEMA
    if schema not in ("arena307", SCHEMA):
        raise SystemExit(f"unknown observation schema: {schema}")

    x_train, _, goal_train, _ = load_sequence_shards(train_files, radar)
    x_val, _, goal_val, ego_val = load_sequence_shards(val_files, radar)

    if radar:
        if x_train.shape[1] != RADAR_DIM or x_val.shape[1] != RADAR_DIM:
            raise SystemExit("radar feature dimension mismatch")
        anchors = ANCHORS.copy()
        counts = np.bincount(assign(goal_train, anchors)[0], minlength=len(anchors))
    else:
        anchors, counts = build_anchors(goal_train, args.max_anchors, merge_m=args.merge_m)
    y_train, train_d = assign(goal_train, anchors)
    y_val, val_d = assign(goal_val, anchors)
    print(json.dumps({"files": len(files), "train": len(train_files), "val": len(val_files),
                      "anchors": anchors.round(2).tolist(), "counts": counts.tolist(),
                      "train_assign_mean_m": float(train_d.mean()),
                      "val_assign_mean_m": float(val_d.mean())}, ensure_ascii=False))

    if not radar:
        x_train = rule_features(x_train)
        x_val = rule_features(x_val)
        x_train = anchor_features(x_train, anchors)
        x_val = anchor_features(x_val, anchors)
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std

    class_weight = len(y_train) / (len(anchors) * np.maximum(counts, 1).astype(np.float32))
    class_weight = np.clip(class_weight, 0.2, 5.0).astype(np.float32)
    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train.astype(np.int64)))
    loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=False)
    val_x = torch.from_numpy(x_val).to(device)
    val_y = torch.from_numpy(y_val.astype(np.int64)).to(device)

    model = WaypointClassifier(x_train.shape[1], len(anchors), args.hidden, args.depth).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "norm_stats.npz", mean=mean, std=std)
    weight_t = torch.from_numpy(class_weight).to(device)

    def evaluate():
        model.eval()
        with torch.no_grad():
            logits = model(val_x)
            loss = F.cross_entropy(logits, val_y).item()
            pred = logits.argmax(dim=1).cpu().numpy()
        pred_m = anchors[pred]
        acc = float((pred == y_val).mean())
        return loss, pred_m, acc

    best = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb, weight=weight_t, label_smoothing=0.05)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(xb)
            count += len(xb)
        scheduler.step()
        val_loss, pred_m, acc = evaluate()
        row = {"epoch": epoch, "train_loss": total / max(1, count), "val_loss": val_loss,
               "top1": acc, **metrics(pred_m, goal_val)}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if best is None or (row["top1"], -row["median_m"]) > (best["metrics"]["top1"], -best["metrics"]["median_m"]):
            best = {"epoch": epoch, "metrics": row}
            torch.save({"state_dict": model.state_dict(), "obs_dim": x_train.shape[1],
                        "base_dim": RADAR_DIM if radar else FEATURE_DIM,
                        "observation_schema": schema,
                        "observation_contract": contract() if radar else None,
                        "n_classes": len(anchors), "hidden": args.hidden, "depth": args.depth,
                        "anchors": anchors, "mean": mean, "std": std,
                        "metrics": row, "epoch": epoch, "model_type": "classifier"},
                       out / "best.pt")

    # Baselines.
    central = np.tile(np.array([[10.8, 7.5]], dtype=np.float32), (len(goal_val), 1))
    report = {
        "best_epoch": best["epoch"],
        "best": best["metrics"],
        "anchors": anchors.tolist(),
        "counts": counts.tolist(),
        "baseline_central": metrics(central, goal_val),
        "baseline_ego": metrics(ego_val, goal_val),
        "val_assign_mean_m": float(val_d.mean()),
        "samples": {"train": int(len(x_train)), "val": int(len(x_val))},
        "history": history,
        "config": vars(args),
        "observation_schema": schema,
        "train_files": [str(p) for p in train_files],
        "val_files": [str(p) for p in val_files],
    }
    (out / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("best_epoch", "best", "baseline_central", "baseline_ego")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
