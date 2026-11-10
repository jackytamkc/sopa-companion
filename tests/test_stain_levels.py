"""Pyramid choices, full-image settings and physical patch units (no helper process)."""
from copy import deepcopy

import pytest

from sopa_companion.engine import stain_levels as levels
from test_stain import IMAGE


def pyramid(factors=(1, 2, 4.004, 8.009)):
    image = deepcopy(IMAGE)
    image["to_frame"][0][0] = image["to_frame"][1][1] = .2125
    image["levels"] = [{**deepcopy(IMAGE["levels"][0]), "pixel": [f, f],
                        "shape": [2, round(3000/f), round(4000/f)]} for f in factors]
    return image


def test_choices_keep_real_geometry_and_take_the_levels_sdata_names():
    image = pyramid()
    offered = levels.offered(image)
    assert [s["label"] for s in offered] == ["scale0", "scale1", "scale2", "scale3"]
    assert offered[-1] == dict(level=3, factor=pytest.approx(8.009), um_per_px=pytest.approx(.2125*8.009),
                              um_per_px_y=pytest.approx(.2125*8.009), label="scale3")
    assert levels.job_level(dict(image=image, settings={"level": 2})) == (2, image["levels"][2], pytest.approx(4.004))
    assert levels.job_level(dict(image=image))[0] == 0


def test_every_level_of_a_five_level_pyramid_is_offered_scale4_included():
    image = pyramid((1, 2, 4.004, 8.009, 16.02))           # a Xenium morphology image's five levels
    offered = levels.offered(image)
    assert [s["label"] for s in offered] == [levels.level_name(k) for k in range(5)] == [f"scale{k}" for k in range(5)]
    assert offered[4] == dict(level=4, factor=pytest.approx(16.02), um_per_px=pytest.approx(.2125*16.02),
                              um_per_px_y=pytest.approx(.2125*16.02), label="scale4")
    assert levels.job_level(dict(image=image, settings={"level": 4})) == (4, image["levels"][4], pytest.approx(16.02))
    assert levels.at_level(dict(diameter=32., min_size=256), offered[4]["factor"]) == dict(
        diameter=pytest.approx(32/16.02), min_size=1)


def test_a_pyramid_of_other_factors_lists_its_levels_by_name():
    offered = levels.offered(pyramid((1, 3, 9)))
    assert [(s["label"], s["level"]) for s in offered] == [("scale0", 0), ("scale1", 1), ("scale2", 2)]
    assert [s["factor"] for s in offered] == [1, pytest.approx(3), pytest.approx(9)]


def test_a_level_too_small_for_any_region_is_not_offered():
    image = pyramid((1, 2, 4, 8, 16, 32, 64, 128, 256))      # 3000 x 4000 pixels down to 12 x 16
    assert [s["label"] for s in levels.offered(image)] == [f"scale{k}" for k in range(7)]
    # scale6 is 47 x 62 pixels, scale7 23 x 31: no region of 32 pixels across fits it, so Start could never run it.
    assert min(image["levels"][6]["shape"][-2:]) >= levels.MIN_ACROSS > min(image["levels"][7]["shape"][-2:])
    with pytest.raises(ValueError, match="Choose a resolution"):
        levels.job_level(dict(image=image, settings={"level": 7}))


@pytest.mark.parametrize("record,name", [
    (dict(level=1, factor=2, um_per_px=.425, label="1/2"), "scale1"),      # an earlier version's label: by its level
    (dict(level=0, factor=1, label="Full"), "scale0"),
    (dict(level=2, factor=3), "scale2"),                                    # the level, not the factor, names it
    (dict(factor=4), "scale2"), (dict(factor=2.0004), "scale1"), (dict(factor=1), "scale0"),   # only a factor
    (dict(label="1/8"), "scale3"), (dict(label="Full"), "scale0"),          # only a label
    ({}, ""), (None, ""), (dict(level=-1, factor=0, label="half"), ""), (dict(level=True), "")])
def test_stored_records_are_named_by_their_level_else_their_factor(record, name):
    assert levels.resolution_name(record) == name


@pytest.mark.parametrize("difference,accepted", [(1e-4, True), (8e-4, True), (.019, True), (.022, False)])
def test_axis_rounding_up_to_two_percent_is_accepted(difference, accepted):
    image = pyramid((1, 2))
    image["levels"][1]["pixel"][1] *= 1 + difference
    assert (len(levels.offered(image)) == 2) == accepted


def test_single_scale_coarse_levels_repeated_sizes_and_unusable_levels():
    assert [s["label"] for s in levels.offered(IMAGE)] == ["scale0"]
    assert [s["label"] for s in levels.offered(pyramid((1, 4, 16, 32)))] == ["scale0", "scale1", "scale2", "scale3"]
    # A level no coarser than the one offered before it (within the 2% rounding) is left out; names stay the levels'.
    assert [s["label"] for s in levels.offered(pyramid((1, 2.2, 2, 4)))] == ["scale0", "scale1", "scale3"]
    assert [s["level"] for s in levels.offered(pyramid((1, 2, 2.03, 4)))] == [0, 1, 3]
    image = pyramid()
    image["levels"][1]["dims"] = ["c", "x", "y"]
    image["levels"][2].update(dims=["c", "z", "y", "x"], shape=[2, 3, 750, 1000])
    assert [s["level"] for s in levels.offered(image)] == [0, 3]
    with pytest.raises(ValueError, match="Choose a resolution"):
        levels.job_level(dict(image=image, settings={"level": True}))


def test_conversion_scales_pixels_and_area_without_mutating_or_validating_level_diameter():
    settings = dict(diameter=12., min_size=15, niter=400, flow_threshold=.4,
                    normalize=dict(sharpen_radius=4, smooth_radius=2, tile_norm_blocksize=33, percentile=[1, 99]))
    before = deepcopy(settings)
    result = levels.at_level(settings, 4)
    assert result == dict(diameter=3., min_size=1, niter=400, flow_threshold=.4,
                         normalize=dict(sharpen_radius=1, smooth_radius=.5, tile_norm_blocksize=8, percentile=[1, 99]))
    assert levels.at_level(dict(diameter=None, min_size=-1), 8) == dict(diameter=None, min_size=-1)
    assert settings == before
    # Nothing kept that nobody calls: the helper's tile layout works out a level's overlap itself, the page its readout.
    assert not hasattr(levels, "level_overlap") and not hasattr(levels, "cell_size")


def test_gui_conversion_uses_physical_pixels_across_samples_rounds_and_clamps():
    value = dict(diameter=12, min_size=20, niter=50,
                 normalize=dict(sharpen_radius=500, smooth_radius=2, tile_norm_blocksize=12345))
    result = levels.full_from_gui(value, 1.7, .2125)
    assert result["diameter"] == 96 and result["min_size"] == 1280 and result["niter"] == 50
    assert result["normalize"] == dict(sharpen_radius=1000, smooth_radius=16, tile_norm_blocksize=98760)
    assert levels.full_from_gui(dict(normalize={"tile_norm_blocksize": 7}), .3, .2)["normalize"]["tile_norm_blocksize"] == 10
