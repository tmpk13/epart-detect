"""Accuracy on synthetic single-part scenes, per package and image scale.

Each trial renders one part at a random sub-pixel position near the image
center and a random angle, runs the nozzle front end on it, and compares the
fitted pose with the truth. A trial fails when the status is not OK or the
angle is off by more than FLIP_DEG (the outline matched the wrong way round).
"""

from __future__ import annotations

import math

import numpy as np

from .detect import OK, Settings, detect_nozzle
from .fit import angle_diff
from .package import Package, Pose
from .synth import render

FLIP_DEG = 5.0


def trial(package: Package, scale: float, rng: np.random.Generator, settings: Settings | None = None):
    """(center error px, angle error deg, relative scale error) or None on
    failure."""
    side = int(math.ceil(4 * package.radius * scale)) + 16
    truth = Pose(
        x=(side - 1) / 2.0 + rng.uniform(-0.3, 0.3) * package.radius * scale,
        y=(side - 1) / 2.0 + rng.uniform(-0.3, 0.3) * package.radius * scale,
        angle=float(rng.uniform(0.0, 360.0)),
        scale=scale,
    )
    scene = render((side, side), [(package, truth)], rng)
    result = detect_nozzle(scene.image, package, settings=settings)
    d = result.detection
    if d is None or d.status != OK:
        return None
    da = angle_diff(d.pose.angle, truth.angle, 360.0 / package.symmetry)
    if da > FLIP_DEG:
        return None
    return math.hypot(d.pose.x - truth.x, d.pose.y - truth.y), da, d.pose.scale / scale - 1.0


def table(packages: list[Package], scales: list[float], trials: int, seed: int) -> str:
    rows = [
        "| package | px/mm | trials | failed | center p50 / p95 px | angle p50 / p95 deg | scale p50 % |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for package in packages:
        for scale in scales:
            rng = np.random.default_rng(seed)
            results = [trial(package, scale, rng) for _ in range(trials)]
            ok = np.array([r for r in results if r is not None])
            failed = trials - len(ok)
            if len(ok):
                c50, c95 = np.percentile(ok[:, 0], [50, 95])
                a50, a95 = np.percentile(ok[:, 1], [50, 95])
                s50 = 100 * np.median(ok[:, 2])
                rows.append(
                    f"| {package.name} | {scale:g} | {trials} | {failed} | {c50:.3f} / {c95:.3f} "
                    f"| {a50:.3f} / {a95:.3f} | {s50:+.2f} |"
                )
            else:
                rows.append(f"| {package.name} | {scale:g} | {trials} | {failed} | - | - | - |")
    return "\n".join(rows)
