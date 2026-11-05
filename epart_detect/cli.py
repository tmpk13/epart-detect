"""Command line: list packages, render synthetic scenes, detect, bench."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from . import bench, overlay
from .detect import Settings, detect_group, detect_nozzle
from .package import PACKAGES_FILE, Pose, load_packages
from .synth import render, scatter


def _size(text: str) -> tuple[int, int]:
    w, h = text.lower().split("x")
    return int(w), int(h)


def _floats(text: str, n: int) -> tuple[float, ...]:
    parts = tuple(float(v) for v in text.split(","))
    if len(parts) != n:
        raise argparse.ArgumentTypeError(f"expected {n} comma-separated numbers, got {text!r}")
    return parts


def _package(args):
    packages = load_packages(Path(args.packages_file))
    if args.package not in packages:
        sys.exit(f"unknown package {args.package!r}; known: {', '.join(packages)}")
    p = packages[args.package]
    return p.mirrored() if getattr(args, "mirror", False) else p


def cmd_packages(args) -> None:
    for name, p in load_packages(Path(args.packages_file)).items():
        umin, umax, vmin, vmax = p.extent
        print(
            f"{name:10s} {p.kind:5s} outline {umax - umin:.2f} x {vmax - vmin:.2f} mm, "
            f"{len(p.leads)} leads, angle modulo {360 // p.symmetry} deg"
        )


def cmd_synth(args) -> None:
    package = _package(args)
    rng = np.random.default_rng(args.seed)
    w, h = _size(args.size)
    if args.nozzle:
        reach = 0.5 * args.scale
        poses = [
            Pose(
                (w - 1) / 2.0 + rng.uniform(-reach, reach),
                (h - 1) / 2.0 + rng.uniform(-reach, reach),
                float(rng.uniform(0.0, 360.0)),
                args.scale,
            )
        ]
    else:
        poses = scatter(package, args.count, (w, h), args.scale, rng)
    scene = render((w, h), [(package, p) for p in poses], rng, noise=args.noise, blur=args.blur)
    cv2.imwrite(args.out, scene.image)
    truth = {
        "package": package.name,
        "scale": args.scale,
        "parts": [{"x": round(p.x, 4), "y": round(p.y, 4), "angle": round(p.angle, 4)} for p in poses],
    }
    if args.truth:
        Path(args.truth).write_text(json.dumps(truth, indent=2) + "\n")
    print(f"wrote {args.out}: {len(poses)} x {package.name} at {args.scale:g} px/mm", file=sys.stderr)


def cmd_detect(args) -> None:
    package = _package(args)
    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image is None:
        sys.exit(f"cannot read image {args.image}")
    settings = Settings(
        px_per_mm=args.px_per_mm,
        background=tuple(int(v) for v in args.background) if args.background else None,  # type: ignore[arg-type]
        threshold=args.threshold,
        min_score=args.min_score,
        scale_tol=args.scale_tol,
    )
    out: dict = {"image": args.image, "package": package.name, "symmetry": package.symmetry}
    nozzle = None
    if args.nozzle is not None:
        nozzle = None if args.nozzle == "center" else _floats(args.nozzle, 2)
        result = detect_nozzle(image, package, nozzle, settings)  # type: ignore[arg-type]
        nozzle = result.nozzle
        out["mode"] = "nozzle"
        out.update(result.to_dict())
        detections = [result.detection] if result.detection else []
    else:
        detections = detect_group(image, package, settings)
        out["mode"] = "group"
        scales = [d.pose.scale for d in detections if d.status == "ok"]
        out["median_scale"] = round(float(np.median(scales)), 4) if scales else None
        out["parts"] = [d.to_dict() for d in detections]
    if args.overlay:
        cv2.imwrite(args.overlay, overlay.draw(image, package, detections, nozzle))
    print(json.dumps(out, indent=2))


def cmd_bench(args) -> None:
    all_packages = load_packages(Path(args.packages_file))
    names = args.packages.split(",") if args.packages else list(all_packages)
    scales = [float(s) for s in args.scales.split(",")]
    print(bench.table([all_packages[n] for n in names], scales, args.trials, args.seed))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="epart-detect", description=__doc__)
    ap.add_argument("--packages-file", default=str(PACKAGES_FILE), help="package table (TOML)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("packages", help="list the package table").set_defaults(func=cmd_packages)

    sy = sub.add_parser("synth", help="render a synthetic scene with known poses")
    sy.add_argument("out", help="image to write (PNG keeps it lossless)")
    sy.add_argument("--package", required=True)
    sy.add_argument("--count", type=int, default=12, help="parts in a group scene")
    sy.add_argument("--scale", type=float, default=30.0, help="pixels per millimeter")
    sy.add_argument("--size", default="800x600", help="WIDTHxHEIGHT in pixels")
    sy.add_argument("--nozzle", action="store_true", help="one part near the image center")
    sy.add_argument("--mirror", action="store_true", help="draw the package seen from below")
    sy.add_argument("--noise", type=float, default=3.0, help="sensor noise sigma, 8-bit levels")
    sy.add_argument("--blur", type=float, default=0.6, help="optical blur sigma, pixels")
    sy.add_argument("--seed", type=int, default=0)
    sy.add_argument("--truth", help="write the true poses here as JSON")
    sy.set_defaults(func=cmd_synth)

    de = sub.add_parser("detect", help="fit the package to an image; JSON on stdout")
    de.add_argument("image")
    de.add_argument("--package", required=True)
    de.add_argument(
        "--nozzle",
        nargs="?",
        const="center",
        help="nozzle mode; optional X,Y of the nozzle axis in pixels (default: image center)",
    )
    de.add_argument("--px-per-mm", type=float, help="known image scale; default fits it per part")
    de.add_argument("--background", type=lambda t: _floats(t, 3), help="B,G,R background color")
    de.add_argument("--threshold", type=float, help="Lab distance cutoff for the foreground")
    de.add_argument("--min-score", type=float, default=0.9, help="fits below this outline score are low_score")
    de.add_argument("--scale-tol", type=float, default=0.1, help="relative scale spread allowed in a group")
    de.add_argument("--mirror", action="store_true", help="the camera sees the underside (bottom camera)")
    de.add_argument("--overlay", help="write an annotated copy of the image here")
    de.set_defaults(func=cmd_detect)

    be = sub.add_parser("bench", help="accuracy table on synthetic single parts (markdown)")
    be.add_argument("--packages", help="comma-separated names; default all")
    be.add_argument("--scales", default="10,20,40", help="comma-separated px/mm")
    be.add_argument("--trials", type=int, default=20)
    be.add_argument("--seed", type=int, default=0)
    be.set_defaults(func=cmd_bench)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
