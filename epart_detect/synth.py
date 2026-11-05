"""Synthetic scenes with known poses, the ground truth for tests and bench.

Each part is drawn as anti-aliased layers (leads, body, terminals): every
layer is rasterized at `SUPERSAMPLE` times the image resolution inside the
part's bounding box, area-averaged down to a coverage map, and blended over
the image. Lighting falloff, sensor noise and optical blur follow.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from .package import Package, Pose
from .render import fill_rects

SUPERSAMPLE = 8

# BGR colors.
BACKGROUND = (150, 100, 50)  # blue ESD mat
BODY_COLORS = {"chip": (35, 35, 38), "ic": (45, 45, 48)}
TERMINAL_COLOR = (200, 200, 205)  # tinned end caps
LEAD_COLOR = (185, 188, 192)


@dataclass
class Scene:
    image: np.ndarray
    truth: list[tuple[Package, Pose]] = field(default_factory=list)


def _blend_layer(img: np.ndarray, package: Package, rects, pose: Pose, color) -> None:
    """Blend the union of `rects` over float image `img` with exact coverage."""
    if not rects:
        return
    h, w = img.shape[:2]
    r = package.radius * pose.scale + 2.0
    x0 = max(0, int(math.floor(pose.x - r)))
    y0 = max(0, int(math.floor(pose.y - r)))
    x1 = min(w, int(math.ceil(pose.x + r)) + 1)
    y1 = min(h, int(math.ceil(pose.y + r)) + 1)
    if x1 <= x0 or y1 <= y0:
        return
    s = SUPERSAMPLE
    # A low-res pixel i spans high-res pixels s*i .. s*i + s - 1, whose
    # centers average to s*i + (s - 1) / 2.
    hi_pose = Pose(
        x=s * (pose.x - x0) + (s - 1) / 2.0,
        y=s * (pose.y - y0) + (s - 1) / 2.0,
        angle=pose.angle,
        scale=s * pose.scale,
    )
    hi = np.zeros(((y1 - y0) * s, (x1 - x0) * s), dtype=np.uint8)
    fill_rects(hi, tuple(rects), hi_pose, 255)
    cover = cv2.resize(hi, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    roi = img[y0:y1, x0:x1]
    roi += cover[..., None] * (np.asarray(color, dtype=np.float32) - roi)


def draw_part(img: np.ndarray, package: Package, pose: Pose) -> None:
    lead = package.lead_color or LEAD_COLOR
    body = package.body_color or BODY_COLORS.get(package.kind, BODY_COLORS["ic"])
    _blend_layer(img, package, package.leads, pose, lead)
    _blend_layer(img, package, (package.body,), pose, body)
    _blend_layer(img, package, package.terminals, pose, TERMINAL_COLOR)


def render(
    size: tuple[int, int],
    parts: list[tuple[Package, Pose]],
    rng: np.random.Generator,
    background=BACKGROUND,
    noise: float = 3.0,
    blur: float = 0.6,
    falloff: float = 0.15,
) -> Scene:
    """Draw `parts` on a plain background of `size` = (width, height).

    falloff: relative brightness drop from the image center to the corners.
    noise: standard deviation of Gaussian sensor noise in 8-bit levels.
    blur: Gaussian optical blur sigma in pixels (0 for none).
    """
    w, h = size
    img = np.empty((h, w, 3), dtype=np.float32)
    img[:] = np.asarray(background, dtype=np.float32)
    for package, pose in parts:
        draw_part(img, package, pose)
    if falloff:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        rr = np.hypot((xx - w / 2) / (w / 2), (yy - h / 2) / (h / 2)) / math.sqrt(2)
        img *= (1.0 - falloff * rr * rr)[..., None]
    if blur:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    if noise:
        img += rng.normal(0.0, noise, img.shape).astype(np.float32)
    return Scene(np.clip(np.round(img), 0, 255).astype(np.uint8), list(parts))


def scatter(
    package: Package,
    count: int,
    size: tuple[int, int],
    scale: float,
    rng: np.random.Generator,
    gap_mm: float = 0.5,
    margin_px: float = 4.0,
) -> list[Pose]:
    """Up to `count` poses at random positions and angles that keep every
    part fully inside the image and at least `gap_mm` from its neighbors."""
    w, h = size
    r = package.radius * scale
    edge = r + margin_px
    poses: list[Pose] = []
    tries = 0
    while len(poses) < count and tries < 200 * count:
        tries += 1
        x = rng.uniform(edge, w - 1 - edge)
        y = rng.uniform(edge, h - 1 - edge)
        if all(math.hypot(x - p.x, y - p.y) >= 2 * r + gap_mm * scale for p in poses):
            poses.append(Pose(x, y, float(rng.uniform(0.0, 360.0)), scale))
    return poses
