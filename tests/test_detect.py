import json
import math

import numpy as np
import pytest

from epart_detect import cli
from epart_detect.detect import CLIPPED, LOW_SCORE, OK, SCALE_OUTLIER, Settings, detect_group, detect_nozzle
from epart_detect.fit import angle_diff
from epart_detect.package import Pose, load_packages
from epart_detect.synth import render, scatter

PACKAGES = load_packages()
CHIP = PACKAGES["0603"]
SIZE = (640, 480)


def nearest(detections, pose):
    return min(detections, key=lambda d: math.hypot(d.pose.x - pose.x, d.pose.y - pose.y))


def test_group_finds_every_part():
    rng = np.random.default_rng(1)
    poses = scatter(CHIP, 14, SIZE, 30.0, rng)
    scene = render(SIZE, [(CHIP, p) for p in poses], rng)
    found = detect_group(scene.image, CHIP)
    assert len(found) == len(poses)
    for p in poses:
        d = nearest(found, p)
        assert d.status == OK
        assert math.hypot(d.pose.x - p.x, d.pose.y - p.y) < 0.3
        assert angle_diff(d.pose.angle, p.angle, 180.0) < 0.5


def test_larger_package_in_the_group_is_a_scale_outlier():
    rng = np.random.default_rng(2)
    poses = scatter(CHIP, 8, SIZE, 30.0, rng)
    big = PACKAGES["1206"]
    stray = Pose(poses[0].x, poses[0].y, poses[0].angle, 30.0)
    parts = [(CHIP, p) for p in poses[1:]] + [(big, stray)]
    # Keep the stray 1206 clear of the others.
    parts = [(pk, p) for pk, p in parts if pk is big or math.hypot(p.x - stray.x, p.y - stray.y) > 120]
    scene = render(SIZE, parts, rng)
    found = detect_group(scene.image, CHIP)
    assert nearest(found, stray).status == SCALE_OUTLIER
    assert sum(d.status == OK for d in found) == len(parts) - 1

    # With a known scale the same part simply does not fit.
    fixed = detect_group(scene.image, CHIP, Settings(px_per_mm=30.0))
    assert nearest(fixed, stray).status == LOW_SCORE


def test_touching_parts_are_low_score():
    rng = np.random.default_rng(3)
    # Side by side, long edges touching: one square blob.
    a = Pose(300.0, 240.0, 0.0, 30.0)
    b = Pose(300.0, 240.0 + 0.8 * 30.0, 0.0, 30.0)
    alone = Pose(150.0, 120.0, 30.0, 30.0)
    scene = render(SIZE, [(CHIP, a), (CHIP, b), (CHIP, alone)], rng)
    found = detect_group(scene.image, CHIP)
    assert len(found) == 2
    assert nearest(found, a).status == LOW_SCORE
    assert nearest(found, alone).status == OK


def test_part_cut_by_the_image_edge_is_clipped():
    rng = np.random.default_rng(4)
    scene = render(SIZE, [(CHIP, Pose(5.0, 200.0, 0.0, 30.0)), (CHIP, Pose(300.0, 200.0, 45.0, 30.0))], rng)
    found = detect_group(scene.image, CHIP)
    assert [d.status for d in found] == [CLIPPED, OK]


def test_nozzle_offset():
    package = PACKAGES["SOT-23"]
    rng = np.random.default_rng(6)
    w, h = 320, 240
    truth = Pose((w - 1) / 2.0 + 7.25, (h - 1) / 2.0 - 4.5, 93.0, 25.0)
    scene = render((w, h), [(package, truth)], rng)
    r = detect_nozzle(scene.image, package, settings=Settings(px_per_mm=25.0))
    assert r.detection is not None and r.detection.status == OK
    assert r.offset_px == pytest.approx((7.25, -4.5), abs=0.2)
    assert r.offset_mm == pytest.approx((0.29, -0.18), abs=0.01)
    assert r.scale_source == "fixed"


def test_nozzle_without_a_part():
    rng = np.random.default_rng(8)
    scene = render((200, 200), [], rng)
    r = detect_nozzle(scene.image, CHIP)
    assert r.detection is None and r.offset_px is None


def test_cli_round_trip(tmp_path, capsys):
    image = tmp_path / "scene.png"
    truth = tmp_path / "truth.json"
    overlay = tmp_path / "overlay.png"
    cli.main(["synth", str(image), "--package", "SOT-23", "--count", "6", "--seed", "3", "--truth", str(truth)])
    capsys.readouterr()
    cli.main(["detect", str(image), "--package", "SOT-23", "--overlay", str(overlay)])
    out = json.loads(capsys.readouterr().out)
    want = json.loads(truth.read_text())["parts"]
    assert out["mode"] == "group"
    assert len(out["parts"]) == len(want)
    assert all(p["status"] == "ok" for p in out["parts"])
    assert overlay.stat().st_size > 0
