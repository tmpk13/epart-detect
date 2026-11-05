"""Fitting one known package to one foreground blob.

Coarse step: candidate angles come from the blob's minimum-area rectangle and
from its principal axis, each tried at all four 90-degree turns. The scale
is the square root of the blob-to-silhouette area ratio, so no absolute size
is needed (the "ratio" mode), and the center is placed so the model and blob
centroids meet. Candidates are ranked by mask IoU; the best few that differ
by more than the package's symmetry go on.

Fine step: least squares over (x, y, angle, log scale) of the blob's signed
distance field sampled at points along the model outline. The field is zero
on the blob edge (halfway between the last inside pixel and the first
outside pixel), negative inside, positive outside, one unit per pixel, so
the residual of a sample is how far it lies off the edge in pixels. A soft
L1 loss keeps a merged neighbor or a lead bent out of line from dragging the
fit. The candidate with the best IoU after refinement wins, and its
outline score decides whether the fit is accepted.

Polish step: when a coverage map is available, the same least squares runs
once more on it (see fit_blob).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares

from .package import Package, Pose
from .render import silhouette_mask

REFINE_CANDIDATES = 3


OUTLINE_BAND = 1.0
"""Pixels this close to the model outline are left out of the score: pixel
quantization and edge blur alone put the blob edge up to about a pixel off."""


@dataclass(frozen=True)
class Fit:
    pose: Pose
    iou: float
    score: float
    """1 minus the area where blob and model disagree, away from the model
    outline, as a fraction of the model area. See outline_score."""
    rms: float
    """RMS distance of the model outline from the blob edge, in pixels."""


def signed_distance(blob: np.ndarray) -> np.ndarray:
    inside = blob > 0
    d_in = cv2.distanceTransform(inside.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    d_out = cv2.distanceTransform((~inside).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return np.where(inside, 0.5 - d_in, d_out - 0.5).astype(np.float64)


def iou(package: Package, pose: Pose, blob: np.ndarray) -> float:
    m = silhouette_mask(package.silhouette, pose, blob.shape) > 0
    b = blob > 0
    union = np.count_nonzero(m | b)
    return np.count_nonzero(m & b) / union if union else 0.0


def outline_score(package: Package, pose: Pose, blob: np.ndarray) -> float:
    """1 - (pixels where blob and model disagree, more than OUTLINE_BAND
    from the model outline) / (model area in pixels).

    Unlike IoU this does not punish small parts for their edge pixels: a
    correct fit of a 10 x 5 pixel chip scores near 1, while a merged
    neighbor, missing lead or wrong package leaves whole regions in
    disagreement and scores low.
    """
    model = silhouette_mask(package.silhouette, pose, blob.shape)
    near = np.abs(signed_distance(model)) < OUTLINE_BAND
    wrong = (model > 0) != (blob > 0)
    return 1.0 - np.count_nonzero(wrong & ~near) / (package.area * pose.scale**2)


def angle_diff(a: float, b: float, period: float) -> float:
    """Smallest difference between two angles taken modulo `period`."""
    d = (a - b) % period
    return min(d, period - d)


def candidates(package: Package, blob: np.ndarray, fixed_scale: float | None) -> list[Pose]:
    m = cv2.moments(blob, binaryImage=True)
    area = m["m00"]
    cx, cy = m["m10"] / area, m["m01"] / area
    scale = fixed_scale or math.sqrt(area / package.area)

    angles = []
    contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    box = cv2.boxPoints(cv2.minAreaRect(max(contours, key=cv2.contourArea)))
    e = box[1] - box[0]
    angles.append(-math.degrees(math.atan2(e[1], e[0])))
    # Image-frame principal axis, measured from +x toward +y (clockwise on
    # screen); negate for counterclockwise, then subtract the model's own
    # axis so the two major axes line up.
    theta = 0.5 * math.atan2(2.0 * m["mu11"], m["mu20"] - m["mu02"])
    angles.append(-math.degrees(theta) - package.axis_angle)

    poses = []
    for a0 in angles:
        for k in range(4):
            a = a0 + 90.0 * k
            off = Pose(0.0, 0.0, a, scale).to_image(package.centroid)
            poses.append(Pose(cx - off[0], cy - off[1], a, scale))
    return poses


def refine(package: Package, pose: Pose, field: np.ndarray, fixed_scale: float | None) -> Pose:
    """Least-squares pose that puts the model outline on the zero level of
    `field` (negative inside the blob, positive outside)."""
    pts = package.boundary(step=1.0 / pose.scale)

    def unpack(p) -> Pose:
        if fixed_scale:
            return Pose(p[0], p[1], p[2], fixed_scale)
        return Pose(p[0], p[1], p[2], math.exp(p[3]))

    def residual(p) -> np.ndarray:
        return sample(field, unpack(p).to_image(pts))

    x0 = [pose.x, pose.y, pose.angle]
    x_scale = [1.0, 1.0, 1.0]
    if not fixed_scale:
        x0.append(math.log(pose.scale))
        x_scale.append(0.01)
    res = least_squares(residual, x0, x_scale=x_scale, loss="soft_l1", f_scale=1.0, method="trf")
    return unpack(res.x)


def sample(field: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Bilinear samples of `field` at image points (N, 2)."""
    return map_coordinates(field, (xy[:, 1], xy[:, 0]), order=1, mode="nearest")


def outline_rms(package: Package, pose: Pose, sdf: np.ndarray) -> float:
    xy = pose.to_image(package.boundary(step=1.0 / pose.scale))
    return float(np.sqrt(np.mean(sample(sdf, xy) ** 2)))


SOFT_MAX_SHIFT_PX = 1.0
SOFT_MAX_TURN_DEG = 2.0
"""The coverage pass only polishes: a result farther than this from the mask
fit means the coverage map is misleading there, and the mask fit stands."""


def fit_blob(
    package: Package, blob: np.ndarray, fixed_scale: float | None = None, coverage: np.ndarray | None = None
) -> Fit:
    """Best pose of `package` for the single blob in uint8 mask `blob`.

    coverage: optional float map in [0, 1] of how much of each pixel the part
        covers, aligned with `blob` and zero away from it. When given, a
        last pass fits the outline to its 0.5 level, which recovers the
        sub-pixel edge that the binary mask rounds away: a near-axis-aligned
        edge is a one-step staircase in the mask, and a line fitted to that
        step overstates the tilt by up to half.

    The blob must not touch the mask edge; pad the crop so the model outline
    stays inside it. The returned angle is normalized to [0, 360 / symmetry).
    """
    period = 360.0 / package.symmetry
    ranked = sorted(
        ((iou(package, p, blob), p) for p in candidates(package, blob, fixed_scale)),
        key=lambda t: -t[0],
    )
    chosen: list[Pose] = []
    for _, p in ranked:
        if all(angle_diff(p.angle, q.angle, period) > 5.0 for q in chosen):
            chosen.append(p)
        if len(chosen) == REFINE_CANDIDATES:
            break

    sdf = signed_distance(blob)
    best: tuple[float, float, Pose] | None = None
    for p in chosen:
        pose = refine(package, p, sdf, fixed_scale)
        key = (iou(package, pose, blob), -outline_rms(package, pose, sdf), pose)
        if best is None or key[:2] > best[:2]:
            best = key
    assert best is not None
    pose = best[2]

    if coverage is not None:
        soft = refine(package, pose, 1.0 - 2.0 * coverage, fixed_scale)
        shift = math.hypot(soft.x - pose.x, soft.y - pose.y)
        if shift <= SOFT_MAX_SHIFT_PX and angle_diff(soft.angle, pose.angle, 360.0) <= SOFT_MAX_TURN_DEG:
            pose = soft

    final = Pose(pose.x, pose.y, pose.angle % period, pose.scale)
    return Fit(final, iou(package, final, blob), outline_score(package, final, blob), outline_rms(package, final, sdf))
