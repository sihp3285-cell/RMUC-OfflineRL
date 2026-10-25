"""The arena background, and how referee coordinates land on it.

The dataset carries no field geometry at all — that is why the engagement prior
in ``data/vis_map.py`` had to be inferred behaviourally rather than read off an
obstacle map.  So the backdrop has to come from outside, and we use the RMUC
2026 rulebook's overhead diagram, with the calibration published by
`ezthor/rm-battlescope <https://github.com/ezthor/rm-battlescope>`_ (MIT, and
already cited in the README).

The subtlety is that the referee system's origin is **not** the corner of the
picture.  Tracking coordinates start at the inner corners of the valid field and
apron, so the perimeter barrier in the diagram sits outside (0, 0)–(28, 15) and
has to be cropped away before the image is stretched onto the field box.  Get
this wrong and every robot is offset by about a metre — visible, but easy to
miss.

The crop is stored as **ratios**, not pixels, so a resized or re-exported copy
of the diagram still lines up.

The image itself is NOT part of this repository: it is DJI's material, not ours
to redistribute.  See ``viz/assets/README.md`` for where to get it.  Everything
here degrades to a plain grid when the file is absent.
"""
from __future__ import annotations

import os

from rm_rl.data import schema as S

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH = os.path.join(HERE, "assets", "rmuc_2026_field_top_view.jpeg")

# Calibration, from rm-battlescope's rmuc_trajectory/field.py:
#   SOURCE_IMAGE_SIZE   = (1683, 938)
#   INNER_FIELD_RECT_PX = (100, 69, 1576, 856)
SOURCE_IMAGE_SIZE = (1683, 938)
INNER_FIELD_RECT_PX = (100, 69, 1576, 856)

CROP = (
    INNER_FIELD_RECT_PX[0] / SOURCE_IMAGE_SIZE[0],      # left   0.05942
    INNER_FIELD_RECT_PX[1] / SOURCE_IMAGE_SIZE[1],      # top    0.07356
    INNER_FIELD_RECT_PX[2] / SOURCE_IMAGE_SIZE[0],      # right  0.93643
    INNER_FIELD_RECT_PX[3] / SOURCE_IMAGE_SIZE[1],      # bottom 0.91269
)


def available(path: str | None = None) -> bool:
    return os.path.exists(path or IMAGE_PATH)


def load_cropped(path: str | None = None):
    """The diagram cropped to exactly the tracked field. None if unavailable."""
    path = path or IMAGE_PATH
    if not os.path.exists(path):
        return None
    try:
        from PIL import Image
    except ImportError:
        print("  [field] Pillow not installed — background image skipped")
        return None
    with Image.open(path) as im:
        w, h = im.size
        box = (round(CROP[0] * w), round(CROP[1] * h),
               round(CROP[2] * w), round(CROP[3] * h))
        return im.convert("RGB").crop(box).copy()


def draw(ax, path: str | None = None, alpha: float = 1.0, zorder: int = -10,
         desaturate: float = 0.0) -> bool:
    """Put the arena under a matplotlib axes already scaled in metres.

    `desaturate` blends the diagram toward grey; the rulebook render is a bright
    colour drawing and at full strength it competes with the data drawn on top.
    """
    img = load_cropped(path)
    if img is None:
        return False
    import numpy as np
    a = np.asarray(img, dtype=np.float32) / 255.0
    if desaturate > 0:
        grey = a @ np.array([.299, .587, .114], np.float32)
        a = a * (1 - desaturate) + grey[..., None] * desaturate
    ax.imshow(a, extent=[0, S.FIELD_X, 0, S.FIELD_Y], origin="upper",
              alpha=alpha, zorder=zorder, interpolation="bilinear",
              aspect="auto")
    return True
