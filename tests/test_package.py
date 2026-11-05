import math

import numpy as np
import pytest

from epart_detect.package import Pose, load_packages

PACKAGES = load_packages()


@pytest.mark.parametrize(
    "name, order",
    [
        ("0201", 2),
        ("0603", 2),
        ("1206", 2),
        ("SOT-23", 1),
        ("SOT-23-5", 1),
        ("SOT-23-6", 2),
        ("SOD-123", 2),
        ("SOIC-8", 2),
        ("QFN-16", 4),
    ],
)
def test_symmetry(name, order):
    assert PACKAGES[name].symmetry == order


def test_sot23_area_and_centroid():
    p = PACKAGES["SOT-23"]
    lead = 0.4 * 0.55
    assert p.area == pytest.approx(2.9 * 1.3 + 3 * lead)
    # Two leads below the body, one above: the centroid sits below the origin.
    assert p.centroid[0] == pytest.approx(0.0, abs=1e-12)
    assert p.centroid[1] == pytest.approx(lead * -0.925 / p.area)


def test_boundary_traces_the_union_outline():
    p = PACKAGES["SOT-23"]
    step = 0.01
    pts = p.boundary(step)
    # Body perimeter, minus where three leads join it, plus three lead
    # outlines (two sides and the tip each).
    perimeter = 2 * (2.9 + 1.3) - 3 * 0.4 + 3 * (2 * 0.55 + 0.4)
    assert len(pts) * step == pytest.approx(perimeter, abs=0.05)
    # No sample lies on the buried edges where a lead meets the body: pins 1
    # and 2 under the bottom edge, pin 3 under the top edge.
    u, v = pts[:, 0], pts[:, 1]
    bottom = (np.abs(v + 0.65) < 1e-9) & (np.abs(np.abs(u) - 0.95) < 0.2)
    top = (np.abs(v - 0.65) < 1e-9) & (np.abs(u) < 0.2)
    assert not (bottom | top).any()
    # The top edge does keep its samples above pins 1 and 2.
    assert ((np.abs(v - 0.65) < 1e-9) & (np.abs(np.abs(u) - 0.95) < 0.2)).any()


def test_mirrored_puts_pin3_below():
    p = PACKAGES["SOT-23"].mirrored()
    assert p.leads[2][1] < 0 < p.leads[0][1]
    assert p.symmetry == 1


def test_pose_angle_is_counterclockwise_on_screen():
    # At 90 degrees the package +u axis points up the screen, which is -y.
    xy = Pose(10.0, 20.0, 90.0, 5.0).to_image(np.array([1.0, 0.0]))
    assert xy == pytest.approx([10.0, 15.0])
    xy = Pose(10.0, 20.0, 0.0, 5.0).to_image(np.array([0.0, 1.0]))
    assert xy == pytest.approx([10.0, 15.0])


def test_axis_angle_follows_the_longer_extent():
    assert PACKAGES["0603"].axis_angle == pytest.approx(0.0)
    # SOIC-8 is 6.0 mm across the leads but 4.9 mm long.
    assert abs(PACKAGES["SOIC-8"].axis_angle) == pytest.approx(90.0)
    assert math.isfinite(PACKAGES["QFN-16"].axis_angle)
