"""The two front ends: a group of loose parts, and one part on a nozzle.

Both segment the image against its plain background, cut each blob out into
its own padded crop, and fit the known package to it. They differ in which
blobs they keep and what they report.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .fit import Fit, fit_blob
from .package import Package, Pose
from .segment import Segmentation, segment

# Status of one detection, from the most to the least serious.
CLIPPED = "clipped"  # blob touches the image edge, so part of it is missing
LOW_SCORE = "low_score"  # outline does not match: merged parts, debris, wrong package
SCALE_OUTLIER = "scale_outlier"  # matches, but its size differs from the group's
OK = "ok"


@dataclass
class Detection:
    status: str
    pose: Pose
    symmetry: int
    iou: float
    score: float
    rms: float
    bbox: tuple[int, int, int, int]
    """x, y, width, height of the blob in pixels."""

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "x": round(self.pose.x, 3),
            "y": round(self.pose.y, 3),
            "angle": round(self.pose.angle, 3),
            "symmetry": self.symmetry,
            "scale": round(self.pose.scale, 4),
            "iou": round(self.iou, 4),
            "score": round(self.score, 4),
            "rms_px": round(self.rms, 4),
            "bbox": list(self.bbox),
        }


@dataclass
class Settings:
    """Tuning shared by both front ends.

    px_per_mm: known image scale; None fits the scale per part from the
        package's proportions alone.
    background: BGR background color, or None to take it from the image
        border.
    threshold: Lab distance cutoff for the foreground, or None for automatic.
    min_score: below this outline score (see fit.outline_score) a fit is
        LOW_SCORE.
    scale_tol: relative deviation from the group's median scale that makes a
        part a SCALE_OUTLIER (group mode, fitted scale only).
    min_area_px: blobs smaller than this are noise and are dropped.
    """

    px_per_mm: float | None = None
    background: tuple[int, int, int] | None = None
    threshold: float | None = None
    min_score: float = 0.9
    scale_tol: float = 0.1
    min_area_px: float = 20.0


def _touches_edge(bbox, shape) -> bool:
    x, y, w, h = bbox
    return x <= 0 or y <= 0 or x + w >= shape[1] or y + h >= shape[0]


def _crop(a: np.ndarray, x0: int, y0: int, w: int, h: int) -> np.ndarray:
    """a[y0:y0+h, x0:x0+w], zero-padded where the window leaves the image."""
    out = np.zeros((h, w), dtype=a.dtype)
    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x0 + w, a.shape[1]), min(y0 + h, a.shape[0])
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = a[sy0:sy1, sx0:sx1]
    return out


def _fit_contour(
    package: Package, contour: np.ndarray, seg: Segmentation, settings: Settings
) -> tuple[Fit, tuple[int, int, int, int]]:
    x, y, w, h = cv2.boundingRect(contour)
    pad = int(max(6, 0.5 * max(w, h)))
    x0, y0 = x - pad, y - pad
    cw, ch = w + 2 * pad, h + 2 * pad
    blob = np.zeros((ch, cw), dtype=np.uint8)
    cv2.drawContours(blob, [contour], -1, 255, thickness=cv2.FILLED, offset=(-x0, -y0))
    # Only this blob's own edge: neighbors' coverage would pull the outline.
    near = cv2.dilate(blob, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
    coverage = np.where(near, _crop(seg.coverage, x0, y0, cw, ch), 0.0)
    fit = fit_blob(package, blob, settings.px_per_mm, coverage)
    p = fit.pose
    return Fit(Pose(p.x + x0, p.y + y0, p.angle, p.scale), fit.iou, fit.score, fit.rms), (x, y, w, h)


def _contours(seg: Segmentation, settings: Settings) -> list[np.ndarray]:
    contours, _ = cv2.findContours(seg.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return [c for c in contours if cv2.contourArea(c) >= settings.min_area_px]


def detect_group(image: np.ndarray, package: Package, settings: Settings | None = None) -> list[Detection]:
    """Every part of `package` in the image, sorted top to bottom, then left
    to right."""
    settings = settings or Settings()
    seg = segment(image, settings.background, settings.threshold)
    contours = _contours(seg, settings)
    if settings.px_per_mm:
        floor = 0.25 * package.area * settings.px_per_mm**2
    else:
        # Without a scale, drop crumbs well below the typical blob.
        floor = 0.25 * float(np.median([cv2.contourArea(c) for c in contours])) if contours else 0.0
    found: list[Detection] = []
    for c in contours:
        if cv2.contourArea(c) < floor:
            continue
        fit, bbox = _fit_contour(package, c, seg, settings)
        if _touches_edge(bbox, image.shape):
            status = CLIPPED
        elif fit.score < settings.min_score:
            status = LOW_SCORE
        else:
            status = OK
        found.append(Detection(status, fit.pose, package.symmetry, fit.iou, fit.score, fit.rms, bbox))

    if not settings.px_per_mm:
        good = [d.pose.scale for d in found if d.status == OK]
        if good:
            median = float(np.median(good))
            for d in found:
                if d.status == OK and abs(d.pose.scale / median - 1.0) > settings.scale_tol:
                    d.status = SCALE_OUTLIER
    found.sort(key=lambda d: (round(d.pose.y), d.pose.x))
    return found


@dataclass
class NozzleResult:
    nozzle: tuple[float, float]
    detection: Detection | None
    offset_px: tuple[float, float] | None
    offset_mm: tuple[float, float] | None
    scale_source: str

    def to_dict(self) -> dict:
        return {
            "nozzle": [round(v, 3) for v in self.nozzle],
            "part": self.detection.to_dict() if self.detection else None,
            "offset_px": [round(v, 3) for v in self.offset_px] if self.offset_px else None,
            "offset_mm": [round(v, 4) for v in self.offset_mm] if self.offset_mm else None,
            "scale_source": self.scale_source,
        }


def detect_nozzle(
    image: np.ndarray,
    package: Package,
    nozzle: tuple[float, float] | None = None,
    settings: Settings | None = None,
) -> NozzleResult:
    """The part held on the nozzle and its offset from the nozzle center.

    nozzle: image position of the nozzle axis; defaults to the image center.
    The part is the largest blob that covers the nozzle point, else the blob
    whose centroid is nearest to it. Offsets are in image axes (+x right,
    +y down); millimeters use px_per_mm when given, otherwise the scale
    fitted from the part's nominal size.
    """
    settings = settings or Settings()
    h, w = image.shape[:2]
    nozzle = nozzle or ((w - 1) / 2.0, (h - 1) / 2.0)
    source = "fixed" if settings.px_per_mm else "fitted"
    seg = segment(image, settings.background, settings.threshold)
    contours = _contours(seg, settings)
    if not contours:
        return NozzleResult(nozzle, None, None, None, source)

    covering = [c for c in contours if cv2.pointPolygonTest(c, nozzle, False) >= 0]
    if covering:
        contour = max(covering, key=cv2.contourArea)
    else:

        def dist(c):
            m = cv2.moments(c)
            if m["m00"] == 0:
                return math.inf
            return math.hypot(m["m10"] / m["m00"] - nozzle[0], m["m01"] / m["m00"] - nozzle[1])

        contour = min(contours, key=dist)

    fit, bbox = _fit_contour(package, contour, seg, settings)
    if _touches_edge(bbox, image.shape):
        status = CLIPPED
    elif fit.score < settings.min_score:
        status = LOW_SCORE
    else:
        status = OK
    d = Detection(status, fit.pose, package.symmetry, fit.iou, fit.score, fit.rms, bbox)
    dx, dy = fit.pose.x - nozzle[0], fit.pose.y - nozzle[1]
    k = settings.px_per_mm or fit.pose.scale
    return NozzleResult(nozzle, d, (dx, dy), (dx / k, dy / k), source)

