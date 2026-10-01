"""T015: the rotation target is picked on DragFlow's feature grid so the executed angle follows the request."""

import copy
import math

import pytest

from advv.config import ROTATION_ERROR_KEYS, validate_config
from advv.errors import ConfigError
from advv.sampler import (
    feature_grid,
    grid_point,
    grid_rotation_degrees,
    region_options,
    rotation_target,
    rotation_tolerance,
    sample_edit,
)
from test_region_sampler import W, H, fixture_proposals, source  # noqa: F401

# runs/v2_pilot_001, first candidate (accident_car_tree, part_000_001 "front wheel", 960x720, scale 3).
PILOT = {"start": [436, 540], "anchor": [447, 521], "grid_start": [145, 180]}
PILOT_SIZE = (960, 720)
PILOT_REQUESTED = -10.512804543427269


def region(degrees=3.0, fraction=0.3, minimum=2.0):
    return {
        "rotation_max_error_degrees": degrees,
        "rotation_max_error_fraction": fraction,
        "rotation_min_executed_degrees": minimum,
    }


def error(executed, requested):
    return abs((executed - requested + 180) % 360 - 180)


def test_pilot_plan_reproduces_the_recorded_angle_error():
    w, h = PILOT_SIZE
    assert feature_grid(w, h) == (320, 240)
    # The old target rotated the pixel start and rounded it: (440, 542) lands on grid (147, 181).
    assert grid_point([440, 542], w, h) == [147, 181] and grid_point(PILOT["anchor"], w, h) == [149, 174]
    old = grid_rotation_degrees(PILOT["grid_start"], [440, 542], PILOT["anchor"], w, h)
    assert old == pytest.approx(-17.744671625056924)  # record operation_params.grid_rotation_degrees
    assert error(old, PILOT_REQUESTED) > rotation_tolerance(region(), PILOT_REQUESTED)


def test_pilot_plan_gets_the_nearest_executed_angle():
    end, executed = rotation_target(PILOT, PILOT_REQUESTED, *PILOT_SIZE)
    assert grid_point(end, *PILOT_SIZE) == [146, 181]
    assert end == [439, 542]  # the pixel of that cell nearest to the rotated start (439.65, 541.69)
    assert executed == grid_rotation_degrees(PILOT["grid_start"], end, PILOT["anchor"], *PILOT_SIZE)
    assert error(executed, PILOT_REQUESTED) < 0.05


def test_target_is_the_best_cell_around_the_rotated_grid_start():
    w, h = PILOT_SIZE
    a, b = grid_point(PILOT["anchor"], w, h), PILOT["grid_start"]
    for requested in (-30.0, -17.3, -4.2, 2.0, 9.9, 25.0):
        end, executed = rotation_target(PILOT, requested, w, h)
        r = math.radians(requested)
        ideal = (
            a[0] + (b[0] - a[0]) * math.cos(r) - (b[1] - a[1]) * math.sin(r),
            a[1] + (b[0] - a[0]) * math.sin(r) + (b[1] - a[1]) * math.cos(r),
        )
        cells = [
            [round(ideal[0]) + dx, round(ideal[1]) + dy]
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            if [round(ideal[0]) + dx, round(ideal[1]) + dy] not in (b, a)
        ]
        assert grid_point(end, w, h) in cells
        angles = [
            math.degrees(math.atan2(c[1] - a[1], c[0] - a[0]) - math.atan2(b[1] - a[1], b[0] - a[0]))
            for c in cells
        ]
        assert error(executed, requested) == pytest.approx(min(error(x, requested) for x in angles))


def test_tolerance_boundary_is_inclusive():
    from advv.sampler import _rotation_fit

    _, executed = rotation_target(PILOT, PILOT_REQUESTED, *PILOT_SIZE)
    gap = error(executed, PILOT_REQUESTED)
    assert _rotation_fit(PILOT, PILOT_REQUESTED, region(gap, 0.0), *PILOT_SIZE) is not None
    assert _rotation_fit(PILOT, PILOT_REQUESTED, region(math.nextafter(gap, 0), 0.0), *PILOT_SIZE) is None
    # max(degrees, fraction x |requested|): the fraction alone can admit the same angle.
    assert _rotation_fit(PILOT, PILOT_REQUESTED, region(1e-9, gap / abs(PILOT_REQUESTED) * (1 + 1e-9)), *PILOT_SIZE)
    assert rotation_tolerance(region(), 5.0) == 3.0
    assert rotation_tolerance(region(), -20.0) == pytest.approx(6.0)
    assert rotation_tolerance(region(), 10.0) == 3.0  # 0.3 x 10 = 3.0 as well


def test_sampled_rotations_stay_within_tolerance_and_record_both_angles(v2cfg, source, tmp_path):  # noqa: F811
    proposals = fixture_proposals(tmp_path, source, v2cfg)
    settings = v2cfg["sampler"]["object_region"]
    rotations = [
        plan
        for plan in (
            sample_edit(source, {"summary": "s"}, v2cfg, i, tmp_path, proposals=proposals) for i in range(60)
        )
        if plan.operation == "rotation"
    ]
    assert rotations
    for plan in rotations:
        params = plan.operation_params
        requested, executed = params["requested_rotation_degrees"], params["grid_rotation_degrees"]
        assert params["rotation_tolerance_degrees"] == rotation_tolerance(settings, requested)
        assert error(executed, requested) <= params["rotation_tolerance_degrees"]
        assert executed == grid_rotation_degrees(
            params["upstream_grid_start"], plan.target_point, plan.anchor_point, W, H
        )


def test_part_without_an_accurate_angle_is_not_rotated(v2cfg, source, tmp_path):  # noqa: F811
    proposals = fixture_proposals(tmp_path, source, v2cfg)
    assert "rotation" in region_options(proposals, v2cfg, W, H)
    strict = copy.deepcopy(v2cfg)
    strict["sampler"]["object_region"].update(rotation_max_error_degrees=1e-6, rotation_max_error_fraction=0.0)
    options = region_options(proposals, strict, W, H)
    assert "rotation" not in options and "deformation" in options


def test_rotation_error_config(cfg, v2cfg):
    for key, value in (
        ("rotation_max_error_degrees", 0),
        ("rotation_max_error_degrees", 181),
        ("rotation_max_error_degrees", True),
        ("rotation_max_error_fraction", -0.1),
        ("rotation_max_error_fraction", 1.5),
        ("rotation_max_error_fraction", None),
        ("rotation_min_executed_degrees", 0),
        ("rotation_min_executed_degrees", -2.0),
        ("rotation_min_executed_degrees", 180.5),
        ("rotation_min_executed_degrees", "2"),
    ):
        bad = copy.deepcopy(v2cfg)
        bad["sampler"]["object_region"][key] = value
        with pytest.raises(ConfigError, match=key):
            validate_config(bad)
    for edge in (
        {"rotation_max_error_degrees": 180},
        {"rotation_max_error_fraction": 0},
        {"rotation_min_executed_degrees": 180},
        {"rotation_min_executed_degrees": 0.5},
    ):
        ok = copy.deepcopy(v2cfg)
        ok["sampler"]["object_region"].update(edge)
        validate_config(ok)
    missing = copy.deepcopy(v2cfg)
    del missing["sampler"]["object_region"]["rotation_max_error_degrees"]
    with pytest.raises(ConfigError, match="rotation_max_error_degrees"):
        validate_config(missing)
    missing = copy.deepcopy(v2cfg)
    del missing["sampler"]["object_region"]["rotation_min_executed_degrees"]
    with pytest.raises(ConfigError, match="rotation_min_executed_degrees"):
        validate_config(missing)
    legacy = copy.deepcopy(cfg)
    for key in ROTATION_ERROR_KEYS:
        del legacy["sampler"]["object_region"][key]
    validate_config(legacy)  # random_geometry_v1 does not use object_region
    partial = copy.deepcopy(cfg)
    del partial["sampler"]["object_region"]["rotation_min_executed_degrees"]
    with pytest.raises(ConfigError, match="rotation_min_executed_degrees"):
        validate_config(partial)  # a bundle key present in v1 checks the whole bundle


# runs/v2_pilot_001 src_cfe95f7139096ee7 part_000_002 "front bumper" (960x720, radius 11 cells), T015 QA M1.
BUMPER = {"start": [576, 556], "anchor": [575, 521], "grid_start": [192, 185]}


def test_radial_cell_is_not_a_rotation():
    from advv.sampler import _rotation_fit

    # Before T015-fix the tolerance (3.0 deg > 2.2 deg) let the radial cell win: ([577, 553], 0.0).
    assert grid_rotation_degrees(BUMPER["grid_start"], [577, 553], BUMPER["anchor"], *PILOT_SIZE) == 0.0
    end, executed = _rotation_fit(BUMPER, -2.2, region(), *PILOT_SIZE)
    assert end != [577, 553] and executed < -2.0
    assert executed == pytest.approx(-4.763641690726189)
    assert rotation_target(BUMPER, 2.2, *PILOT_SIZE, 2.0)[1] > 2.0
    # A larger minimum leaves no neighbouring cell for this small angle.
    assert _rotation_fit(BUMPER, -2.2, region(minimum=6.0), *PILOT_SIZE) is None


@pytest.mark.parametrize("option", [PILOT, BUMPER])
def test_executed_angle_keeps_the_direction_and_minimum(option):
    from advv.sampler import _rotation_fit

    for i in range(-600, 601):
        requested = i * 0.05
        if abs(requested) < 2:
            continue
        fit = _rotation_fit(option, requested, region(), *PILOT_SIZE)
        if fit is not None:
            assert fit[1] * requested > 0 and abs(fit[1]) >= 2.0


def test_grid_rounding_matches_upstream_float32():
    from advv.sampler import _cell_pixel, valid_shifts

    # 1000x750 -> grid 330x245. 350 / 1000 * 330 is 115.49999999999999 in float64 but 115.5 in float32,
    # which torch.round (half to even) sends to 116 (T015 QA L1).
    w, h = 1000, 750
    assert feature_grid(w, h) == (330, 245)
    assert grid_point([350, 100], w, h, True) == [116, 33]
    assert grid_point([350, 100], w, h) == [115, 33]  # random_geometry_v1 keeps its float64 rounding
    assert _cell_pixel((116, 33), (350, 100), w, h) == [350, 100]
    assert _cell_pixel((115, 33), (350, 100), w, h) == [349, 100]
    # Shifting 351 -> 350 stays on cell 116 upstream, so it is not a move (float64 would call it 115).
    dx, _, _ = valid_shifts([351, 100], ((-3, 3), (0, 0)), (1, 3), w, h)
    assert sorted(dx.tolist()) == [-3, -2, 3]  # 352 and 353 are cell 116 too


def test_rotation_target_rounds_every_grid_point_in_float32(monkeypatch):
    """T017 QA L6: the anchor grid point uses upstream float32 rounding like the start and target."""
    from advv import sampler

    calls = []

    def spy(point, width, height, float32=False):
        calls.append(float32)
        return grid_point(point, width, height, float32)

    monkeypatch.setattr(sampler, "grid_point", spy)
    rotation_target(PILOT, PILOT_REQUESTED, *PILOT_SIZE)
    assert calls and all(calls)
    # Where the two roundings differ (1000 px wide, x = 350), the anchor now follows upstream's float32 cell.
    assert grid_point([350, 0], 1000, 750) == [115, 0] and grid_point([350, 0], 1000, 750, True) == [116, 0]
