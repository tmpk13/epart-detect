"""Foreground segmentation against a plain background.

A pixel is foreground when its CIELAB color is far from the background color.
Distance in Lab, not gray level, keeps both dark bodies and bright tinned
terminals: on a mid-tone mat one is darker and the other lighter, and a gray
threshold in either direction drops one of them.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


def _lab(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)


def border_pixels(img: np.ndarray, frac: float = 0.03) -> np.ndarray:
    """Pixels of a frame `frac` of the image size wide around the edge."""
    h, w = img.shape[:2]
    b = max(1, int(round(frac * min(h, w))))
    return np.concatenate(
        [
            img[:b].reshape(-1, img.shape[2]),
            img[-b:].reshape(-1, img.shape[2]),
            img[b:-b, :b].reshape(-1, img.shape[2]),
            img[b:-b, -b:].reshape(-1, img.shape[2]),
        ]
    )


def color_distance(image: np.ndarray, background=None) -> tuple[np.ndarray, float]:
    """Per-pixel Lab distance from the background, and the background noise
    floor (median plus 6 robust sigmas of that distance on the border).

    background: BGR color of the background, or None to take the median of
    the image border (the border must be mostly background).
    """
    lab = _lab(image)
    if background is None:
        bg = np.median(border_pixels(lab), axis=0)
    else:
        bg = _lab(np.array([[background]], dtype=np.uint8))[0, 0]
    dist = np.linalg.norm(lab - bg, axis=2)
    edge = border_pixels(dist[..., None])[:, 0]
    med = float(np.median(edge))
    sigma = 1.4826 * float(np.median(np.abs(edge - med)))
    return dist, med + 6.0 * sigma


@dataclass
class Segmentation:
    mask: np.ndarray
    """uint8, 255 on parts (holes filled), 0 on the background."""
    coverage: np.ndarray
    """float32 in [0, 1]: how much of each pixel a part covers, estimated as
    the color distance over the local contrast. Meaningful only within a few
    pixels of a part; zero it elsewhere before use."""


def segment(
    image: np.ndarray, background=None, threshold: float | None = None, min_contrast: float = 12.0
) -> Segmentation:
    """Foreground mask and coverage map.

    threshold: Lab distance cutoff; by default the larger of the border noise
    floor and `min_contrast`. A pixel is also required to reach half the
    local contrast (the largest distance within 2 pixels), which puts the
    mask edge where a pixel is half covered instead of wherever the faintest
    blur tail crosses the noise floor. Holes inside a blob (a reflection on a
    body) are filled. There is no morphological opening: it would erase a
    0201 imaged 3 pixels wide; noise specks are dropped later by blob area.
    """
    dist, floor = color_distance(image, background)
    t = threshold if threshold is not None else max(floor, min_contrast)
    local = cv2.dilate(dist, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    raw = ((dist > t) & (dist > 0.5 * local)).astype(np.uint8) * 255
    contours, _ = cv2.findContours(raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mask = np.zeros_like(raw)
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
    coverage = np.clip(dist / np.maximum(local, t), 0.0, 1.0).astype(np.float32)
    coverage[(mask > 0) & (raw == 0)] = 1.0
    return Segmentation(mask, coverage)


def foreground_mask(
    image: np.ndarray, background=None, threshold: float | None = None, min_contrast: float = 12.0
) -> np.ndarray:
    """uint8 mask (0 or 255) of pixels that differ from the background."""
    return segment(image, background, threshold, min_contrast).mask
