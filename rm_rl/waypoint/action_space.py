"""Versioned coordinate vocabulary, independent of radar observation schema."""
import json
from pathlib import Path
import numpy as np
from rm_rl.waypoint.radar import ANCHORS


def validate(spec):
    a = np.asarray(spec['anchors'], dtype=np.float32)
    names = spec['names']
    if spec.get('schema') != 'radar-waypoints-v1':
        raise ValueError('unsupported action schema')
    if a.ndim != 2 or a.shape[1] != 2 or len(a) < 2 or not np.isfinite(a).all():
        raise ValueError('invalid anchors')
    if (a < 0).any() or (a > [28, 15]).any() or len(np.unique(a, axis=0)) != len(a):
        raise ValueError('out of field or duplicate anchors')
    if len(names) != len(a) or len(set(names)) != len(a):
        raise ValueError('invalid action names')
    i = spec['supply_index']
    if not isinstance(i, int) or not 0 <= i < len(a) or names[i] != 'supply':
        raise ValueError('invalid supply index')
    tolerance = float(spec['max_assignment_error_m'])
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('invalid assignment tolerance')
    return a


def load(path=None):
    spec = json.loads(Path(path).read_text()) if path else {
        'schema': 'radar-waypoints-v1', 'names': ['own_central', 'point_a', 'point_b', 'supply'],
        'anchors': ANCHORS.tolist(), 'supply_index': 3, 'max_assignment_error_m': .05}
    validate(spec)
    return spec
