"""Package geometry and poses.

A package is a set of axis-aligned rectangles in millimeters, in the package
frame: u along the package length, v across it, origin at the body center,
+v up on screen when the package is drawn at angle 0. The silhouette (what a
camera sees against the background) is the union of the body and the leads.
Terminals lie on the body; they only matter for drawing synthetic scenes.

All union quantities (area, centroid, symmetry) are computed exactly by
splitting the plane along every rectangle edge into cells that are either
fully inside or fully outside the union.
"""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

Rect = tuple[float, float, float, float]
"""u_center, v_center, u_size, v_size in millimeters."""

PACKAGES_FILE = Path(__file__).with_name("packages.toml")


@dataclass(frozen=True)
class Pose:
    """Where a package sits in an image.

    x, y: pixel coordinates of the package origin (body center). The image
        frame has +x right and +y down, and (0, 0) is the center of the
        top-left pixel.
    angle: degrees from the image +x axis to the package +u axis,
        counterclockwise as seen on screen.
    scale: pixels per millimeter.
    """

    x: float
    y: float
    angle: float
    scale: float

    def to_image(self, uv: np.ndarray) -> np.ndarray:
        """Map package-frame points (..., 2) in mm to image pixels (..., 2)."""
        a = math.radians(self.angle)
        c, s = math.cos(a), math.sin(a)
        u = uv[..., 0]
        v = uv[..., 1]
        x = self.x + self.scale * (u * c - v * s)
        y = self.y - self.scale * (u * s + v * c)
        return np.stack([x, y], axis=-1)


def rect_corners(r: Rect) -> np.ndarray:
    """Corners of a rectangle as a (4, 2) array, counterclockwise in u, v."""
    cu, cv, su, sv = r
    hu, hv = su / 2.0, sv / 2.0
    return np.array(
        [[cu - hu, cv - hv], [cu + hu, cv - hv], [cu + hu, cv + hv], [cu - hu, cv + hv]]
    )


def _inside(rects: tuple[Rect, ...], u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """True where (u, v) lies in the closed union of the rectangles."""
    hit = np.zeros(np.shape(u), dtype=bool)
    for cu, cv, su, sv in rects:
        hit |= (np.abs(u - cu) <= su / 2.0) & (np.abs(v - cv) <= sv / 2.0)
    return hit


def _cells(*rect_sets: tuple[Rect, ...]):
    """Grid formed by all rectangle edges: cell centers (uc, vc) and cell
    sizes (du, dv), each a 2-D array indexed [v, u]."""
    us: set[float] = set()
    vs: set[float] = set()
    for rects in rect_sets:
        for cu, cv, su, sv in rects:
            us.update((round(cu - su / 2.0, 9), round(cu + su / 2.0, 9)))
            vs.update((round(cv - sv / 2.0, 9), round(cv + sv / 2.0, 9)))
    ue = np.array(sorted(us))
    ve = np.array(sorted(vs))
    uc, vc = np.meshgrid((ue[:-1] + ue[1:]) / 2.0, (ve[:-1] + ve[1:]) / 2.0)
    du, dv = np.meshgrid(np.diff(ue), np.diff(ve))
    return uc, vc, du, dv


def _rotate90(r: Rect) -> Rect:
    cu, cv, su, sv = r
    return (-cv, cu, sv, su)


def _rotate180(r: Rect) -> Rect:
    cu, cv, su, sv = r
    return (-cu, -cv, su, sv)


def _same_union(a: tuple[Rect, ...], b: tuple[Rect, ...]) -> bool:
    uc, vc, du, dv = _cells(a, b)
    keep = du * dv > 1e-12
    return bool(np.all(_inside(a, uc, vc)[keep] == _inside(b, uc, vc)[keep]))


@dataclass(frozen=True)
class Package:
    name: str
    kind: str
    body: Rect
    leads: tuple[Rect, ...] = ()
    terminals: tuple[Rect, ...] = ()
    body_color: tuple[int, int, int] | None = None
    lead_color: tuple[int, int, int] | None = None

    @property
    def silhouette(self) -> tuple[Rect, ...]:
        return (self.body, *self.leads)

    @cached_property
    def extent(self) -> tuple[float, float, float, float]:
        """umin, umax, vmin, vmax of the silhouette."""
        c = np.concatenate([rect_corners(r) for r in self.silhouette])
        return (c[:, 0].min(), c[:, 0].max(), c[:, 1].min(), c[:, 1].max())

    @cached_property
    def radius(self) -> float:
        """Largest distance from the origin to a silhouette corner, in mm."""
        c = np.concatenate([rect_corners(r) for r in self.silhouette])
        return float(np.hypot(c[:, 0], c[:, 1]).max())

    @cached_property
    def _union_cells(self):
        """Centers and sizes of the grid cells inside the silhouette."""
        uc, vc, du, dv = _cells(self.silhouette)
        inside = _inside(self.silhouette, uc, vc)
        return uc[inside], vc[inside], du[inside], dv[inside]

    @cached_property
    def area(self) -> float:
        """Silhouette area in mm^2."""
        _, _, du, dv = self._union_cells
        return float((du * dv).sum())

    @cached_property
    def centroid(self) -> np.ndarray:
        """Silhouette centroid (u, v) in mm; not the origin for SOT-23."""
        uc, vc, du, dv = self._union_cells
        a = du * dv
        return np.array([(uc * a).sum(), (vc * a).sum()]) / a.sum()

    @cached_property
    def axis_angle(self) -> float:
        """Direction of the silhouette's major principal axis, degrees from +u.

        Second moments about the centroid: a cell of size du x dv contributes
        its point moment plus its own du^2/12 and dv^2/12 terms.
        """
        uc, vc, du, dv = self._union_cells
        a = du * dv
        pu = uc - self.centroid[0]
        pv = vc - self.centroid[1]
        m20 = (a * (pu * pu + du * du / 12.0)).sum()
        m02 = (a * (pv * pv + dv * dv / 12.0)).sum()
        m11 = (a * pu * pv).sum()
        return math.degrees(0.5 * math.atan2(2.0 * m11, m20 - m02))

    @cached_property
    def symmetry(self) -> int:
        """Rotational symmetry order of the silhouette: 1, 2 or 4.

        The angle of a fitted part is only defined modulo 360 / symmetry.
        """
        s = self.silhouette
        if _same_union(s, tuple(_rotate90(r) for r in s)):
            return 4
        if _same_union(s, tuple(_rotate180(r) for r in s)):
            return 2
        return 1

    def boundary(self, step: float) -> np.ndarray:
        """Points on the silhouette outline, about `step` mm apart, as (N, 2).

        Each rectangle edge is sampled at segment midpoints; a sample is kept
        when a point just outside it, along the edge's outward normal, is
        outside every rectangle. Edges buried inside the union (a lead's inner
        end against the body, the body edge under a lead) drop out.
        """
        eps = 1e-6
        out = []
        for r in self.silhouette:
            c = rect_corners(r)
            for i in range(4):
                p0, p1 = c[i], c[(i + 1) % 4]
                d = p1 - p0
                length = float(np.hypot(*d))
                n = max(1, math.ceil(length / step - 1e-9))
                t = (np.arange(n) + 0.5) / n
                pts = p0 + t[:, None] * d
                normal = np.array([d[1], -d[0]]) / length
                probe = pts + eps * normal
                keep = ~_inside(self.silhouette, probe[:, 0], probe[:, 1])
                out.append(pts[keep])
        return np.concatenate(out)

    def mirrored(self) -> Package:
        """The package seen from below: v negated, so the outline is flipped
        top to bottom. Use for a bottom camera that sees the part's underside
        without mirroring the image."""

        def flip(r: Rect) -> Rect:
            return (r[0], -r[1], r[2], r[3])

        return Package(
            name=self.name + " (mirrored)",
            kind=self.kind,
            body=flip(self.body),
            leads=tuple(flip(r) for r in self.leads),
            terminals=tuple(flip(r) for r in self.terminals),
            body_color=self.body_color,
            lead_color=self.lead_color,
        )


def _package_from_toml(name: str, d: dict) -> Package:
    length, width = d["body"]
    body: Rect = (0.0, 0.0, float(length), float(width))
    terminals: tuple[Rect, ...] = ()
    if "terminal" in d:
        t = float(d["terminal"])
        off = (length - t) / 2.0
        terminals = ((-off, 0.0, t, width), (off, 0.0, t, width))
    leads = tuple(tuple(float(x) for x in r) for r in d.get("leads", []))
    for r in leads:
        if len(r) != 4:
            raise ValueError(f"package {name}: lead {r} needs 4 numbers")
    return Package(
        name=name,
        kind=d.get("kind", "ic"),
        body=body,
        leads=leads,  # type: ignore[arg-type]
        terminals=terminals,
        body_color=tuple(d["body_color"]) if "body_color" in d else None,  # type: ignore[arg-type]
        lead_color=tuple(d["lead_color"]) if "lead_color" in d else None,  # type: ignore[arg-type]
    )


def load_packages(path: Path = PACKAGES_FILE) -> dict[str, Package]:
    data = tomllib.loads(path.read_text())
    return {name: _package_from_toml(name, d) for name, d in data["packages"].items()}


def get_package(name: str, path: Path = PACKAGES_FILE) -> Package:
    packages = load_packages(path)
    if name not in packages:
        raise KeyError(f"unknown package {name!r}; known: {', '.join(packages)}")
    return packages[name]
