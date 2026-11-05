"""Annotated copies of the input image: fitted outlines, centers, directions."""

from __future__ import annotations

import cv2
import numpy as np

from .detect import CLIPPED, LOW_SCORE, OK, SCALE_OUTLIER, Detection
from .package import Package
from .render import silhouette_mask

# Solarized accents in BGR.
GREEN = (0, 153, 133)
YELLOW = (0, 137, 181)
ORANGE = (22, 75, 203)
RED = (47, 50, 220)
CYAN = (152, 161, 42)
MAGENTA = (130, 54, 211)

STATUS_COLOR = {OK: GREEN, SCALE_OUTLIER: YELLOW, CLIPPED: ORANGE, LOW_SCORE: RED}


def _stroke(img: np.ndarray) -> int:
    """Line width that stays visible relative to the image size."""
    return max(1, round(min(img.shape[:2]) / 500))


def draw(img: np.ndarray, package: Package, detections: list[Detection], nozzle=None) -> np.ndarray:
    out = img.copy()
    t = _stroke(out)
    font = 0.35 * t
    for i, d in enumerate(detections):
        color = STATUS_COLOR.get(d.status, RED)
        p = d.pose
        r = int(np.ceil(package.radius * p.scale)) + 2
        x0, y0 = int(p.x) - r, int(p.y) - r
        mask = silhouette_mask(package.silhouette, p, (2 * r + 1, 2 * r + 1), origin=(x0, y0))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color, t, offset=(x0, y0))
        c = (int(round(p.x)), int(round(p.y)))
        cv2.drawMarker(out, c, color, cv2.MARKER_CROSS, 4 * t + 4, t)
        tip = p.to_image(np.array([package.extent[1] + 0.3 * package.body[3], 0.0]))
        cv2.arrowedLine(out, c, (int(round(tip[0])), int(round(tip[1]))), CYAN, t, tipLength=0.25)
        label = f"{i} {p.angle:.1f}"
        cv2.putText(out, label, (c[0] + r // 2, c[1] - r // 2), cv2.FONT_HERSHEY_SIMPLEX, font, color, t, cv2.LINE_AA)
    if nozzle is not None:
        n = (int(round(nozzle[0])), int(round(nozzle[1])))
        cv2.drawMarker(out, n, MAGENTA, cv2.MARKER_TILTED_CROSS, 8 * t + 8, t)
    return out
