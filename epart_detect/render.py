"""Rasterizing package rectangles at a pose."""

from __future__ import annotations

import cv2
import numpy as np

from .package import Pose, Rect, rect_corners

SHIFT = 4
"""Fixed-point fraction bits passed to cv2.fillPoly: vertices land on a
1/16-pixel grid instead of being rounded to whole pixels."""


def polygon(r: Rect, pose: Pose, origin: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """Image-pixel corners of one rectangle, relative to `origin`, as (4, 2)."""
    return pose.to_image(rect_corners(r)) - np.asarray(origin)


def fill_rects(
    img: np.ndarray,
    rects: tuple[Rect, ...],
    pose: Pose,
    value,
    origin: tuple[float, float] = (0.0, 0.0),
) -> None:
    """Fill each rectangle into `img` in place; `origin` is the image
    position of img[0, 0], so a crop can be drawn in full-image poses."""
    scale = 1 << SHIFT
    polys = [np.round(polygon(r, pose, origin) * scale).astype(np.int32) for r in rects]
    cv2.fillPoly(img, polys, value, lineType=cv2.LINE_8, shift=SHIFT)


def silhouette_mask(
    rects: tuple[Rect, ...], pose: Pose, shape: tuple[int, int], origin=(0.0, 0.0)
) -> np.ndarray:
    """uint8 mask (0 or 1) of the union of `rects` at `pose`."""
    m = np.zeros(shape, dtype=np.uint8)
    fill_rects(m, rects, pose, 1, origin)
    return m
