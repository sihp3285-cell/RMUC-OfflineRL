"""Reference ONNX runtime for radar-station integration.

The C++ node must implement the same three steps: build_features(frame),
rule_features(...), anchor_features(...), then normalize with meta mean/std.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from rm_rl.waypoint.features import anchor_features, build_features, rule_features, sequence_context_one


class WaypointRuntime:
    def __init__(self, onnx_path, meta_path=None):
        onnx_path = Path(onnx_path)
        meta_path = Path(meta_path) if meta_path else onnx_path.with_suffix(".json")
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if self.meta.get("observation_schema", "arena307") != "arena307":
            raise ValueError("This runtime builds arena307 inputs; use the radar observation contract for this model")
        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.mean = np.asarray(self.meta["mean"], dtype=np.float32)
        self.std = np.asarray(self.meta["std"], dtype=np.float32)
        anchors = np.asarray(self.meta.get("anchors", []), dtype=np.float32)
        self.anchors = anchors if anchors.size else None
        self.history = []

    def predict(self, frame):
        base = build_features(frame)
        feat = sequence_context_one(base, self.history)
        self.history.append(base)
        self.history = self.history[-100:]
        feat = rule_features(feat)
        if self.anchors is not None:
            feat = anchor_features(feat, self.anchors)
        feat = (feat - self.mean) / self.std
        out = self.session.run(None, {self.input_name: feat[None, :].astype(np.float32)})[0][0]
        if self.anchors is not None:
            return tuple(float(v) for v in self.anchors[int(np.argmax(out))])
        return float(out[0]) * 28.0, float(out[1]) * 15.0
