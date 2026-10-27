from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rm_rl.waypoint.model import WaypointClassifier, WaypointMLP


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="rm_runs/waypoint_bc/best.pt")
    parser.add_argument("--out", default="rm_runs/waypoint_bc/waypoint.onnx")
    args = parser.parse_args()
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    is_classifier = "anchors" in state
    if is_classifier:
        model = WaypointClassifier(state["obs_dim"], state["n_classes"], state["hidden"], state["depth"])
    else:
        model = WaypointMLP(state["obs_dim"], state["hidden"], state["depth"])
    model.load_state_dict(state["state_dict"])
    model.eval()
    dummy = torch.zeros(1, state["obs_dim"], dtype=torch.float32)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    output_name = "goal_logits" if is_classifier else "goal_norm"
    torch.onnx.export(
        model, dummy, str(out),
        input_names=["features"], output_names=[output_name],
        dynamic_axes={"features": {0: "batch"}, output_name: {0: "batch"}},
        opset_version=17, do_constant_folding=True,
    )
    meta = {
        "observation_schema": state.get("observation_schema", "arena307"),
        "observation_contract": state.get("observation_contract"),
        "model_type": "classifier" if is_classifier else "regression",
        "obs_dim": int(state["obs_dim"]),
        "base_dim": int(state.get("base_dim", state["obs_dim"])),
        "hidden": int(state["hidden"]),
        "depth": int(state["depth"]),
        "mean": np.asarray(state["mean"]).tolist(),
        "std": np.asarray(state["std"]).tolist(),
        "metrics": state.get("metrics"),
        "epoch": state.get("epoch"),
    }
    if is_classifier:
        meta["anchors"] = np.asarray(state["anchors"]).tolist()
        meta["output"] = "logits over anchors; argmax -> (x, y) metres"
    else:
        meta["output"] = "sigmoid-normalized (x/28, y/15)"
    (out.with_suffix(".json")).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
