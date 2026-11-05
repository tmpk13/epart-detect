import math

import numpy as np
import pytest

from epart_detect.detect import OK, Settings, detect_nozzle
from epart_detect.fit import angle_diff
from epart_detect.package import Pose, load_packages
from epart_detect.synth import render

PACKAGES = load_packages()


def single(package, scale, rng, angle=None):
    side = int(math.ceil(4 * package.radius * scale)) + 16
    truth = Pose(
        (side - 1) / 2.0 + rng.uniform(-3, 3),
        (side - 1) / 2.0 + rng.uniform(-3, 3),
        float(rng.uniform(0, 360)) if angle is None else angle,
        scale,
    )
    return render((side, side), [(package, truth)], rng).image, truth


@pytest.mark.parametrize("name", list(PACKAGES))
def test_pose_accuracy(name):
    """With the part at least 40 px long, center within 0.3 px and angle
    within 0.5 degrees (modulo the outline's symmetry)."""
    package = PACKAGES[name]
    umin, umax, _, _ = package.extent
    scale = max(25.0, 40.0 / (umax - umin))
    rng = np.random.default_rng(7)
    for _ in range(6):
        image, truth = single(package, scale, rng)
        d = detect_nozzle(image, package).detection
        assert d is not None and d.status == OK
        assert math.hypot(d.pose.x - truth.x, d.pose.y - truth.y) < 0.3
        assert angle_diff(d.pose.angle, truth.angle, 360.0 / package.symmetry) < 0.5
        assert d.pose.scale == pytest.approx(scale, rel=0.03)
        assert 0.0 <= d.pose.angle < 360.0 / package.symmetry


def test_sot23_is_never_flipped():
    package = PACKAGES["SOT-23"]
    rng = np.random.default_rng(3)
    for angle in np.arange(0.0, 360.0, 22.5) + 4.0:
        image, truth = single(package, 15.0, rng, float(angle))
        d = detect_nozzle(image, package).detection
        assert d is not None
        assert angle_diff(d.pose.angle, truth.angle, 360.0) < 2.0


def test_mirrored_view_needs_the_mirrored_model():
    package = PACKAGES["SOT-23"]
    below = package.mirrored()
    rng = np.random.default_rng(5)
    image, truth = single(below, 20.0, rng, 30.0)
    right = detect_nozzle(image, below).detection
    wrong = detect_nozzle(image, package).detection
    assert right is not None and wrong is not None
    assert angle_diff(right.pose.angle, 30.0, 360.0) < 1.0
    # The SOT-23 outline is symmetric left to right, so seen from below it
    # matches the top-view model turned half way round.
    assert angle_diff(wrong.pose.angle, 210.0, 360.0) < 1.0


def test_fixed_scale_is_kept():
    package = PACKAGES["0805"]
    rng = np.random.default_rng(11)
    image, truth = single(package, 22.0, rng)
    d = detect_nozzle(image, package, settings=Settings(px_per_mm=22.0)).detection
    assert d is not None and d.status == OK
    assert d.pose.scale == 22.0
    assert math.hypot(d.pose.x - truth.x, d.pose.y - truth.y) < 0.3
