"""Fixes to SpatialData samples on the engine side (engine/zarr_samples.py, engine/samples.py), without
the sopa environment: what an open finds is said as notes (pixels, several transcript elements, images left
out or chosen, z planes), names the routes refuse are never offered, region_of_interest is never cells,
an element of empty geometries opens and a failed open is not remembered, float images are drawn, a sample
without transcripts opens for stain segmentation, and samples past Recent's 30 still answer. The helper's
inspection is stood in, with the fields it gains; tests/test_zarr.py runs the real
helper on synthetic samples."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import re
import sys

import numpy as np
import pytest
import zstandard
from fastapi.testclient import TestClient

from conftest import HERE

from sopa_companion.engine import samples as S, zarr_samples as Z
from sopa_companion.engine.samples import SampleError

BASE = "http://127.0.0.1:8799"
UM = [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]]      # 0.5 µm per pixel


def banned() -> list[str]:
    """The page's banned words (frontend/src/qc-copy.mjs BANNED_WORDS): no note may hold one."""
    source = (HERE.parent / "frontend" / "src" / "qc-copy.mjs").read_text()
    block = source[source.index("BANNED_WORDS"):]
    return re.findall(r"'([^']+)'", block[:block.index("]")])


def samples(tmp_path) -> Z.ZarrSamples:
    return Z.ZarrSamples({"zarr": {"python": ""}, "proseg": {"max_region_um": 1000.0},
                          "paths": {"cache_dir": str(tmp_path / "cache"), "data_dir": str(tmp_path / "var")}})


def store(tmp_path, name: str = "s.zarr") -> Path:
    """A folder the open takes for a SpatialData zarr (zarr.json and an element folder)."""
    import zarr
    path = tmp_path / name
    zarr.open_group(str(path), mode="w")
    (path / "points").mkdir()
    return path


def write_image(path: Path, key: str, planes: np.ndarray, channels: list[str], levels: int = 2, to_frame=None) -> dict:
    """An image's pyramid as zarr arrays (images/<key>/<level>, halved per level), and its entry as the helper lists it."""
    import zarr
    found = []
    for level in range(levels):
        data = np.ascontiguousarray(planes[:, ::2 ** level, ::2 ** level])
        where = f"images/{key}/{level}"
        array = zarr.open_array(store=str(path / where), mode="w", shape=data.shape, dtype=data.dtype,
                                chunks=(1, *data.shape[1:]))
        array[:] = data
        found.append({"path": where, "shape": list(data.shape), "dims": ["c", "y", "x"], "origin": [0.0, 0.0],
                      "pixel": [2.0 ** level] * 2})
    return {"key": key, "channels": channels, "dtype": str(planes.dtype), "axes": ["c", "y", "x"], "levels": found,
            "to_frame": UM if to_frame is None else to_frame}


def image_entry(key: str, channels: list[str], dtype: str = "uint16", size: float = 0.5, to_frame=True) -> dict:
    """An image as the helper lists it, with no pixels behind it (notes and choices only)."""
    return {"key": key, "channels": channels, "dtype": dtype, "axes": ["c", "y", "x"],
            "levels": [{"path": f"images/{key}/0", "shape": [len(channels), 64, 64], "dims": ["c", "y", "x"],
                        "origin": [0.0, 0.0], "pixel": [1.0, 1.0]}],
            "to_frame": [[size, 0.0, 0.0], [0.0, size, 0.0], [0.0, 0.0, 1.0]] if to_frame else None}


def shape(key: str, count: int = 40, extent=(0.0, 0.0, 200.0, 100.0), geometry="Polygon") -> dict:
    return {"key": key, "count": count, "geometry": {geometry: count}, "columns": [], "to_frame": UM,
            "extent": list(extent)}


def inspection(path: Path, **changes) -> dict:
    """The helper's inspect of a Xenium-like sample in µm (its fields: points_elements, units, z, skipped_images)."""
    info = {"schema": 1, "path": str(path), "name": "toy", "points_key": "transcripts",
            "attrs": {"cell_segmentation_image": "morphology_focus", "transcripts_dataframe": "transcripts",
                      "transcript_to_cell_assignment": ["cell_id", "UNASSIGNED"]},
            "points": [{"key": "transcripts", "columns": {"x": "float", "y": "float", "feature_name": "string"},
                        "feature_key": "feature_name", "instance_key": "cell_id", "files": 1, "count": 1000,
                        "extent": [0.0, 0.0, 200.0, 100.0], "z_range": [0.0, 10.0],
                        "prior_columns": {"cells": [{"column": "cell_id", "unassigned": "UNASSIGNED",
                                                     "assigned_fraction": 0.7}],
                                          "nuclei": [{"column": "overlaps_nucleus", "nuclear": "1",
                                                      "nuclear_fraction": 0.3}]}}],
            "images": [], "shapes": [shape("cell_boundaries"), shape("nucleus_boundaries", 42)], "labels": [],
            "tables": [], "suggested_prior": {"cells": {"kind": "shapes", "key": "cell_boundaries"},
                                              "nuclei": {"kind": "shapes", "key": "nucleus_boundaries"}},
            "mtime": Z.ZarrSamples._mtime(str(path)), "points_elements": ["transcripts"],
            "units": {"frame": "um", "why": ""}, "z": {"kind": "um"}, "skipped_images": []}
    info.update(changes)
    return info


def stand_in(zarr: Z.ZarrSamples, info: dict | None = None, made_by=None):
    """The helper answering `inspect` with `info` (zarr.inspected: what it answers now), `segmentations` with
    `made_by` and `tissue` with a square; anything else fails as the helper would."""
    zarr.inspected = info
    calls = []

    def helper(command, args, **kwargs):
        calls.append(command)
        if command == "inspect":
            if isinstance(zarr.inspected, Exception):
                raise zarr.inspected
            return copy.deepcopy(zarr.inspected)
        if command == "segmentations":
            return made_by or {"shapes": [], "tables": []}
        if command == "tissue":
            return {"polygons": [[[[0, 0], [10, 0], [10, 10], [0, 10]]]], "area_um2": 100.0}
        raise Z.HelperFailed(f"{command}: not stood in", 422)
    zarr.helper = helper
    return calls


def notes(detail) -> dict:
    return {note["code"]: note for note in detail["notes"]}


# ---- float images -----------------------------------------------------------------------------------------------------
def test_values_are_drawn_as_u16_floats_by_their_scale_others_as_stored():
    floats = np.array([0, .5, .999, 1.0, np.nan, -0.25], np.float32)
    _u16 = S._u16
    assert _u16(floats, 65535.0).tolist() == [0, 32768, 65469, 65535, 0, 0]       # the old cast: 0 below 1
    assert _u16(floats).tolist() == [0, 0, 1, 1, 0, 0]
    assert _u16(np.array([0, 7, 65535], np.uint16)).tolist() == [0, 7, 65535]
    assert _u16(np.array([0, 200, 255], np.uint8)).tolist() == [0, 200, 255]
    assert _u16(np.array([-5, 3, 32767], np.int16)).tolist() == [0, 3, 32767]
    assert _u16(np.array([1, 70000], np.uint32)).tolist() == [1, 65535]
    assert _u16(np.array([[1, 2]], np.uint16)).dtype == np.dtype("<u2")


def test_a_float_image_is_drawn_over_the_u16_range_not_black(tmp_path):
    """A float32 image in 0..1 was cast as it is: every crop and view of runs and the stain picker was 0."""
    path = store(tmp_path)
    rng = np.random.default_rng(0)
    planes = rng.random((2, 64, 128), dtype=np.float32) * np.array([1.0, 0.25], np.float32)[:, None, None]
    planes[1, :4, :4] = np.nan
    image = write_image(path, "morphology_focus", planes, ["DAPI", "Membrane"])
    zarr = samples(tmp_path)
    stand_in(zarr, inspection(path, images=[image]))
    detail = zarr.open(str(path))
    dapi, membrane = detail["channels"]
    assert dapi["scale"] == pytest.approx(65535 / float(planes[0].max()))       # on its statistics level (the first here)
    assert membrane["scale"] == pytest.approx(65535 / float(np.nanmax(planes[1])))
    assert dapi["dtype"] == "uint16" and dapi["max"] >= 65000 and dapi["p_hi"] > 60000 and dapi["p_lo"] < 2000
    assert membrane["max"] >= 65000                                # a dim channel spans the range too
    # a run's crop at full resolution: the u16 file, its PNG and its limits in the same units
    meta = zarr.crop_images(detail["id"], [0, 0, 64, 32], tmp_path / "crop")
    raw = zstandard.ZstdDecompressor().decompressobj().decompress((tmp_path / "crop" / "c0.u16.zst").read_bytes())
    crop = np.frombuffer(raw, "<u2").reshape(meta["height"], meta["width"])
    assert crop.shape == (64, 128) and crop.max() >= 65000 and (crop > 0).mean() > 0.95
    assert np.allclose(crop, np.clip(np.rint(planes[0] * dapi["scale"]), 0, 65535), atol=1)
    assert meta["channels"][0]["p_hi"] > 60000
    # the stain picker's and tiled runs' view: the same scale on any level
    view, where = zarr.read_view(detail["id"], None, 1, [0, 0, 64, 32], max_px=64)
    assert where["level"] == 1 and view.dtype == np.dtype("<u2") and view.max() >= 60000 and view[0, 0] == 0
    # statistics kept from before the scale existed are worked out again
    kept = zarr._stats_path(detail["id"], str(path), image)
    assert kept.exists() and re.fullmatch(r"stats-morphology_focus-[0-9a-f]{12}\.json", kept.name)
    kept.write_text(json.dumps([{**entry, "max": 1, "p_hi": 1.0} for entry in
                                [{k: v for k, v in c.items() if k != "scale"} for c in detail["channels"]]]))
    assert zarr._stats(detail["id"], str(path), image)[0]["max"] >= 65000


def group(path: Path, key: str, attrs: dict):
    """The image's own zarr.json (a SpatialData image keeps its transformations there)."""
    (path / "images" / key / "zarr.json").write_text(json.dumps({"zarr_format": 3, "node_type": "group",
                                                                 "attributes": attrs}))


def test_an_image_whose_transformation_alone_was_written_again_is_inspected_again(tmp_path):
    """The sample's mtime does not see an image's own metadata, so an image moved by a rewrite of
    its transformation kept its old geometry (the overview on the page in the wrong place). Each image's stamp is kept
    with its inspection; another stamp inspects the sample again, and what was derived from the image is made again
    under its new identity (the old files go)."""
    path = store(tmp_path)
    planes = (np.arange(64, dtype=np.float32).reshape(1, 8, 8) / 63)
    image = write_image(path, "morphology_focus", planes, ["DAPI"], levels=1)
    group(path, "morphology_focus", {"moved": 0})
    zarr = samples(tmp_path)
    calls = stand_in(zarr, inspection(path, images=[image], areas=1))
    detail = zarr.open(str(path))
    assert detail["overview"]["x0"] == 0 and calls.count("inspect") == 1
    first = zarr.overview_path(detail["id"])
    group(path, "morphology_focus", {"moved": 100})                # the transformation alone, written again
    moved = lambda offset: [[0.5, 0.0, offset], [0.0, 0.5, offset], [0.0, 0.0, 1.0]]
    zarr.inspected = inspection(path, images=[{**image, "to_frame": moved(100.0)}], areas=1)
    reopened = zarr.open(str(path))                               # the engine that read it first
    assert reopened["overview"]["x0"] == 100 and calls.count("inspect") == 2
    assert not first.exists() and zarr.overview_path(detail["id"]).exists()
    group(path, "morphology_focus", {"moved": 200})
    zarr.inspected = inspection(path, images=[{**image, "to_frame": moved(200.0)}], areas=1)
    restarted = samples(tmp_path)
    stand_in(restarted, zarr.inspected)
    again = restarted.detail(detail["id"])                        # a new engine, no refresh asked: the stamp says
    assert again["overview"]["x0"] == 200 and again["extent"] == detail["extent"]
    assert restarted.overview_path(detail["id"]).exists()


def test_an_inspection_kept_before_images_were_stamped_is_made_again(tmp_path):
    """An inspection kept by an earlier version carries no image stamps. Stamped as the image is now,
    an image rewritten before the upgrade (its transformation alone, which the sample's time does not see) kept its
    old geometry for good. It is inspected once more, then stamped; the next engine reads it as kept."""
    path = store(tmp_path)
    planes = (np.arange(64, dtype=np.float32).reshape(1, 8, 8) / 63)
    image = write_image(path, "morphology_focus", planes, ["DAPI"], levels=1)
    zarr = samples(tmp_path)
    stand_in(zarr, inspection(path, images=[image], areas=1))
    detail = zarr.open(str(path))
    kept = zarr.cache / detail["id"] / "inspect.json"
    earlier = json.loads(kept.read_text())
    assert all(i.get("stamp") for i in earlier["images"])
    for entry in earlier["images"]:
        entry.pop("stamp")                                         # as an earlier version kept it
    kept.write_text(json.dumps(earlier))
    moved = [[0.5, 0.0, 100.0], [0.0, 0.5, 100.0], [0.0, 0.0, 1.0]]
    now = inspection(path, images=[{**image, "to_frame": moved}], areas=1)       # moved before the upgrade
    upgraded = samples(tmp_path)
    calls = stand_in(upgraded, now)
    assert upgraded.detail(detail["id"])["overview"]["x0"] == 100 and calls.count("inspect") == 1
    assert all(i.get("stamp") for i in json.loads(kept.read_text())["images"])
    following = samples(tmp_path)
    calls = stand_in(following, now)
    assert following.detail(detail["id"])["overview"]["x0"] == 100 and calls.count("inspect") == 0


def test_an_image_written_again_with_another_shape_range_and_channels_is_read_as_it_is_now(tmp_path):
    """The open arrays, the overview and the statistics were kept by the image's name: a rewrite
    under the same key with another shape failed to read ('cannot reshape'), another range kept the old scale (98 % of
    pixels saturated), another channel count kept the old channels. All are kept by the image's identity now."""
    path = store(tmp_path)
    small = np.arange(64, dtype=np.float32).reshape(1, 8, 8) / 63
    image = write_image(path, "morphology_focus", small, ["DAPI"], levels=1)
    zarr = samples(tmp_path)
    stand_in(zarr, inspection(path, images=[image]))
    detail = zarr.open(str(path))
    assert detail["channels"][0]["scale"] == pytest.approx(65535.0)
    zarr.read_view(detail["id"], None, 0, [0, 0, 4, 4], max_px=8)
    import shutil
    shutil.rmtree(path / "images" / "morphology_focus")
    rng = np.random.default_rng(1)
    large = (rng.random((4, 16, 16)) * 100).astype(np.float32)
    names = ["DAPI", "B", "C", "D"]
    image = write_image(path, "morphology_focus", large, names, levels=1)
    zarr.inspected = inspection(path, images=[image])
    for engine in (zarr, samples(tmp_path)):
        stand_in(engine, zarr.inspected)
        now = engine.open(str(path))
        assert [c["name"] for c in now["channels"]] == names
        assert now["channels"][0]["scale"] == pytest.approx(65535 / float(large[0].max()))
        view, where = engine.read_view(now["id"], None, 3, [0, 0, 8, 8], max_px=16)
        assert view.shape == (16, 16) and where["x1"] == 8
        meta = engine.crop_images(now["id"], [0, 0, 8, 8], tmp_path / "crop")
        assert len(meta["channels"]) == 4 and meta["width"] == 16
    stats = sorted(p.name for p in (zarr.cache / detail["id"]).glob("stats-*"))
    overviews = sorted(p.name for p in (zarr.cache / detail["id"]).glob("overview-*"))
    assert len(stats) == 1 and len(overviews) == 2                # this identity's only (png and json)


def test_an_integer_image_is_drawn_as_stored(tmp_path):
    path = store(tmp_path)
    planes = np.arange(2 * 32 * 32, dtype=np.uint16).reshape(2, 32, 32)
    zarr = samples(tmp_path)
    stand_in(zarr, inspection(path, images=[write_image(path, "morphology_focus", planes, ["DAPI", "Boundary"])]))
    detail = zarr.open(str(path))
    assert "scale" not in detail["channels"][0]
    zarr.crop_images(detail["id"], [0, 0, 16, 16], tmp_path / "crop")
    raw = zstandard.ZstdDecompressor().decompressobj().decompress((tmp_path / "crop" / "c1.u16.zst").read_bytes())
    assert np.array_equal(np.frombuffer(raw, "<u2").reshape(32, 32), planes[1])


# ---- what an open found --------------------------------------------------------------------------------------------
def test_an_open_says_what_it_found_one_plain_line_each(tmp_path):
    """Transcripts in pixels, several transcript elements none of which the sample names, z planes, images left out
    (rotated, or sharing no coordinate system) and an RGB image listed first: each one line under the sample's name, the
    rest in its ⓘ; the default image is not the RGB one."""
    path = store(tmp_path)
    attrs = {}                                                 # the sample names neither its transcripts nor its image
    points = inspection(path)["points"][0]
    pixels = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    info = inspection(
        path, attrs=attrs, points_key="a_points",
        points=[{**points, "key": "a_points"}, {"key": "b_points", "columns": {}, "count": 900}],
        points_elements=["a_points", "b_points"],
        units={"frame": "pixel", "why": "every image has one pixel per unit"}, z={"kind": "index"},
        images=[image_entry("he_image", ["r", "g", "b"], "uint8", 1.0),
                write_image(path, "dapi", np.ones((2, 16, 16), np.uint16), ["DAPI", "CD45"], 1, pixels),
                {"key": "he_rotated", "error": "Image 'he_rotated' is rotated relative to the transcripts: not supported"},
                image_entry("other_cs", ["DAPI"], to_frame=False)],
        skipped_images=[{"key": "he_rotated", "reason": "rotated relative to the transcripts"}], points_choice="first",
        notes=[{"code": "unassigned_token", "text": "In the transcripts' cell_id column, -1 marks no cell",
                "detail": "The sample says 0, but -1 holds the transcripts in no cell."}])
    zarr = samples(tmp_path)
    stand_in(zarr, info)
    detail = zarr.open(str(path))
    assert detail["image_key"] == "dapi" and detail["tools"] == ["proseg", "stain"]
    found = notes(detail)
    assert list(found) == ["pixel_frame", "several_points", "z_index", "skipped_images", "default_image",
                           "unassigned_token"]                 # the helper's own notes after these
    assert found["pixel_frame"]["text"] == ("These transcripts seem to be in pixels, not micrometres: sizes shown in µm "
                                            "are pixels here.")
    assert "every image has one pixel per unit" in found["pixel_frame"]["detail"]
    assert found["several_points"]["text"] == "This sample has 2 transcript elements: this app uses a_points only."
    assert "b_points" in found["several_points"]["detail"]
    assert found["skipped_images"]["text"] == "2 images are not used here: he_rotated, other_cs."
    assert found["skipped_images"]["detail"].splitlines() == [
        "he_rotated: rotated relative to the transcripts", "other_cs: it has no coordinate system in common with the "
                                                           "transcripts"]
    assert "dapi" in found["default_image"]["text"] and "cell_segmentation_image" in found["default_image"]["detail"]
    words = banned()
    assert "prior" in words and len(words) > 20
    for note in detail["notes"]:
        assert note["text"] and note["detail"] and "\n" not in note["text"], note
        assert not [w for w in words if w.lower() in (note["text"] + note["detail"]).lower()], note
    # the overview shows the nucleus stain of the image it shows
    assert Z.overview_channel(["PolyT", "Cellbound1", "DAPI"]) == 2 and Z.overview_channel(["CD3", "hoechst"]) == 1
    assert Z.overview_channel(["r", "g", "b"]) == 0
    # What chose the transcripts, as the helper says it (here the sample's attrs): no note; the helper's own note of a
    # code said here is the one shown; its default image is the one shown.
    zarr.inspected = {**info, "points_choice": "attrs", "default_image": "he_image",
                      "notes": [{"code": "pixel_frame", "text": "In pixels.", "detail": "The helper's words."}]}
    (tmp_path / "cache" / "zarr" / detail["id"] / "inspect.json").unlink()      # inspected again
    again = zarr.open(str(path))
    assert [n["code"] for n in again["notes"]] == ["z_index", "skipped_images", "default_image", "pixel_frame",
                                                   "image_unreadable"]
    assert notes(again)["pixel_frame"]["text"] == "In pixels." and again["image_key"] == "he_image"
    # (that image has no pixels on disk: it is said, the sample opens without it)
    assert notes(again)["image_unreadable"]["text"] == "The image he_image could not be read: it is not shown."
    assert again["overview"] is None and again["channels"] == [] and "FileNotFoundError" in again["image_error"]


def test_the_overview_shows_the_nucleus_stain_not_channel_zero(tmp_path):
    """The overview always showed channel 0 (PolyT on MERSCOPE, a dim marker on CosMx); it shows the image's
    nucleus stain, as the stain picker chooses it, and says which."""
    path = store(tmp_path)
    planes = np.zeros((3, 32, 48), np.uint16)
    planes[1] = 500 + np.arange(48, dtype=np.uint16)            # DAPI: bright; PolyT and CD45: nothing
    image = write_image(path, "scan", planes, ["PolyT", "DAPI", "CD45"], levels=1)
    zarr = samples(tmp_path)
    stand_in(zarr, inspection(path, images=[image], attrs={"cell_segmentation_image": "scan"}))
    detail = zarr.open(str(path))
    assert detail["overview"]["channel"] == 1
    png = zarr.overview_path(detail["id"])
    assert re.fullmatch(r"overview-scan-c1-[0-9a-f]{12}\.png", png.name) and png.exists()
    assert png.name == f"overview-scan-c1-{detail['overview']['identity']}.png"
    from PIL import Image
    assert np.asarray(Image.open(png)).max() > 0                 # the DAPI plane, not channel 0's black


def test_numbers_json_cannot_carry_are_null_in_the_detail(tmp_path):
    """A NaN anywhere in what an open found (an element of empty geometries' extent; here the transcripts'
    z range) answered 500: JSON refuses NaN. Every number that is not finite is null in the detail."""
    assert Z._finite({"a": [1.0, float("nan")], "b": {"c": float("inf"), "d": (2, -float("inf"))}, "e": "x"}) == \
        {"a": [1.0, None], "b": {"c": None, "d": [2, None]}, "e": "x"}
    path = store(tmp_path)
    info = inspection(path)
    info["points"][0]["z_range"] = [float("nan"), float("nan")]
    zarr = samples(tmp_path)
    stand_in(zarr, info)
    detail = zarr.open(str(path))
    json.dumps(detail, allow_nan=False)                          # what the route sends
    assert detail["z_range"] in (None, [None, None])


def test_tissue_found_an_earlier_way_is_found_again(tmp_path):
    """Tissue was found with the brightest channel on H&E too (the glass: no tissue), and that answer stayed
    in the engine's cache, so Start was still refused after the fix. The way tissue is found is in the cache's stamp."""
    from sopa_companion.zarr_bridge.tissue import DETECTION_VERSION
    path = store(tmp_path)
    image = write_image(path, "he", np.full((3, 40, 60), 200, np.uint8), ["r", "g", "b"], levels=1)
    zarr = samples(tmp_path)
    calls = stand_in(zarr, inspection(path, images=[image], shapes=[], attrs={}))
    sid = zarr.open(str(path))["id"]
    assert zarr.tissue(sid)["source"] == "detected" and calls.count("tissue") == 1
    cached = next((zarr.cache / sid).glob("tissue-*.json"))
    kept = json.loads(cached.read_text())
    assert kept["stamp"][-1] == DETECTION_VERSION
    zarr.tissue(sid)
    assert calls.count("tissue") == 1                            # found this way already: from the cache
    cached.write_text(json.dumps({**kept, "polygons": [], "area_um2": 0.0, "stamp": kept["stamp"][:-1]}))
    again = zarr.tissue(sid)                                     # found the earlier way (none on H&E): found again
    assert calls.count("tissue") == 2 and again["source"] == "detected" and again["area_um2"] == 100.0


def test_an_inspection_made_before_the_notes_still_gives_them(tmp_path):
    """Without the helper's own fields (an inspection kept from before them), the engine says the same from what the
    inspection lists: images at one frame unit per pixel, every transcript element, the images it left out."""
    path = store(tmp_path)
    points = inspection(path)["points"][0]
    pixels = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    info = inspection(path, attrs={}, points_key="a_points", points=[{**points, "key": "a_points"},
                                                                     {"key": "b_points", "columns": {}, "count": 9}],
                      images=[write_image(path, "dapi", np.ones((1, 16, 16), np.uint16), ["DAPI"], 1, pixels),
                              {"key": "turned", "error": "Image 'turned' is rotated relative to the transcripts: not "
                                                         "supported"}])
    for field in ("points_elements", "units", "z", "skipped_images"):
        info.pop(field)
    zarr = samples(tmp_path)
    calls = stand_in(zarr, info)
    found = notes(zarr.open(str(path)))
    assert list(found) == ["pixel_frame", "several_points", "skipped_images"]
    assert found["skipped_images"]["detail"] == "Image 'turned' is rotated relative to the transcripts: not supported"
    assert calls.count("inspect") == 1
    zarr.open(str(path))                                       # an open inspects such a sample again (its notes)
    assert calls.count("inspect") == 2


def test_a_sample_that_names_its_parts_opens_without_notes(tmp_path):
    """The Xenium-like sample: µm, one transcripts element the attrs name, its own image named; also several transcript
    elements when the attrs name the one used, and several images when the sample names its own."""
    path = store(tmp_path)
    zarr = samples(tmp_path)
    stand_in(zarr, inspection(path, images=[image_entry("he_image", ["r", "g", "b"], "uint8"),
                                            write_image(path, "morphology_focus", np.ones((1, 16, 16), np.uint16),
                                                        ["DAPI"], levels=1)],
                              points_elements=["misc", "transcripts", "vertices"]))
    detail = zarr.open(str(path))
    assert detail["notes"] == [] and detail["image_key"] == "morphology_focus"
    assert detail["tools"] == ["proseg", "stain"] and detail["transcripts"] == 1000


# ---- what is offered and listed ----------------------------------------------------------------------------------
def test_names_the_routes_refuse_are_never_offered_nor_suggested(tmp_path, monkeypatch):
    """'-cells_leading_dash' sorted first and was the pre-selected cells, then its outlines answered 404."""
    monkeypatch.setattr(Z, "STAMP_RETRY_S", 0.0)
    path = store(tmp_path)
    long = "cell_boundaries_" + "x" * 105
    odd = ["-cells_leading_dash", long, "noyaux_é"]
    info = inspection(path, shapes=[shape(key) for key in ["-cells_leading_dash", "cell_boundaries", long, "noyaux_é",
                                                           "nucleus_boundaries"]],
                      suggested_prior={"cells": {"kind": "shapes", "key": "-cells_leading_dash"},
                                       "nuclei": {"kind": "shapes", "key": "nucleus_boundaries"}})
    zarr = samples(tmp_path)
    stand_in(zarr, info)
    detail = zarr.open(str(path))
    offered = {o.get("key") for part in ("cells", "nuclei") for o in detail["prior_options"][part]}
    assert offered == {"cell_boundaries", "nucleus_boundaries", None}          # None: the transcripts' columns
    assert detail["suggested_prior"] == {"cells": {"kind": "shapes", "key": "cell_boundaries"},
                                         "nuclei": {"kind": "shapes", "key": "nucleus_boundaries"}}
    assert [s["key"] for s in zarr.segmentations(detail["id"])] == ["cell_boundaries", "nucleus_boundaries"]
    named = notes(detail)["skipped_names"]
    assert named["text"].startswith("3 layers have names this app cannot use: -cells_leading_dash, cell_boundaries_xx")
    assert len(named["text"]) < 160 and all(key in named["detail"] for key in odd)
    with pytest.raises(SampleError):
        zarr.validate_prior(detail["id"], {"cells": {"kind": "shapes", "key": "-cells_leading_dash"}})
    for key in odd:
        with pytest.raises(SampleError) as refused:
            zarr._segmentation(detail["id"], key)
        assert refused.value.status == 404
    # Should the helper's rule not take the shapes, the part it cannot offer is left out rather than refused later.
    from sopa_companion.zarr_bridge import helper
    monkeypatch.setattr(helper, "_suggest_prior", lambda *args: (_ for _ in ()).throw(TypeError("changed")))
    assert Z.ZarrSamples._suggestion(info, info["points"][0], detail["prior_options"]) == {
        "cells": None, "nuclei": {"kind": "shapes", "key": "nucleus_boundaries"}}


def test_the_apps_own_saves_are_offered_after_the_samples_own_outlines(tmp_path):
    """A stain save of one 100 µm box (cellpose_boundaries_1) was listed, and chosen, before the sample's
    own cells (it sorts first by name). The app's own saves (the helper's app_save) come after the others in the list,
    as in the helper's suggestion."""
    path = store(tmp_path)
    saved = {**shape("cellpose_boundaries_1", 42), "app_save": True}
    info = inspection(path, shapes=[saved, shape("cells_polygons", 374), shape("nucleus_boundaries", 300)],
                      suggested_prior={"cells": {"kind": "shapes", "key": "cells_polygons"}, "nuclei": None})
    zarr = samples(tmp_path)
    stand_in(zarr, info)
    detail = zarr.open(str(path))
    cells = [o["key"] for o in detail["prior_options"]["cells"] if o["kind"] == "shapes"]
    assert cells == ["cells_polygons", "nucleus_boundaries", "cellpose_boundaries_1"]
    assert detail["suggested_prior"]["cells"]["key"] == "cells_polygons"


def test_region_of_interest_is_the_tissue_never_cells_however_many_pieces(tmp_path, monkeypatch):
    """A region_of_interest of two polygons was listed as a segmentation and offered as cells."""
    monkeypatch.setattr(Z, "STAMP_RETRY_S", 0.0)
    path = store(tmp_path)
    zarr = samples(tmp_path)
    stand_in(zarr, inspection(path, shapes=[shape("cell_boundaries"), shape("region_of_interest", 2)],
                              images=[write_image(path, "morphology_focus", np.ones((1, 16, 16), np.uint16), ["DAPI"],
                                                  levels=1)]))
    detail = zarr.open(str(path))
    assert all(o.get("key") != "region_of_interest" for part in ("cells", "nuclei")
               for o in detail["prior_options"][part])
    assert [s["key"] for s in zarr.segmentations(detail["id"])] == ["cell_boundaries"]
    assert "region_of_interest" in [s["key"] for s in detail["shapes"]]                  # listed among the shapes
    assert zarr.tissue(detail["id"])["source"] == "region_of_interest"                   # and the tissue still


def test_empty_geometries_open_in_strict_json_and_a_failed_open_is_not_remembered(config, tmp_path, monkeypatch, capsys):
    """An element whose geometries are all empty has a NaN extent: the open answered 500 ('Out of range float
    values are not JSON compliant') and the sample was in Recent all the same."""
    from sopa_companion.server import create_app
    monkeypatch.setattr(Z, "STAMP_RETRY_S", 0.0)
    path = store(tmp_path)
    info = inspection(path, shapes=[shape("cell_boundaries"), shape("empty_outlines", 5, [float("nan")] * 4)])
    headers = {"Authorization": "Bearer " + "t" * 40, "Origin": BASE}
    with TestClient(create_app(config, token="t" * 40), base_url=BASE) as client:
        zarr = client.app.state.zarr
        stand_in(zarr, info)
        opened = client.post("/api/v1/samples/open", json={"path": str(path)}, headers=headers)
        assert opened.status_code == 200, opened.text
        detail = opened.json()
        empty = next(s for s in detail["shapes"] if s["key"] == "empty_outlines")
        assert empty["extent"] is None and "empty_outlines" not in json.dumps(detail["prior_options"])
        listed = client.get(f"/api/v1/samples/{detail['id']}/segmentations", headers=headers).json()["segmentations"]
        assert [s["key"] for s in listed] == ["cell_boundaries"]
        assert [s["id"] for s in client.get("/api/v1/samples", headers=headers).json()] == [detail["id"]]
        # A failed open, the helper's or the engine's own: plain words, and the sample stays out of Recent.
        other = store(tmp_path, "other.zarr")
        zarr.inspected = Z.HelperFailed("This sample cannot be opened: shapes 'cell boundaries' has a name SpatialData "
                                        "refuses", 422)
        refused = client.post("/api/v1/samples/open", json={"path": str(other)}, headers=headers)
        assert refused.status_code == 422 and refused.json()["detail"].startswith("This sample cannot be opened")
        zarr.inspected = {k: v for k, v in inspection(other).items() if k != "name"}     # a field it cannot do without
        broken = client.post("/api/v1/samples/open", json={"path": str(other)}, headers=headers)
        assert broken.status_code == 422 and broken.json()["detail"] == ("This sample could not be opened: the engine "
                                                                         "ran into a problem (the engine log has it)")
        assert "KeyError: 'name'" in capsys.readouterr().out                            # its traceback: the log
        assert [s["id"] for s in client.get("/api/v1/samples", headers=headers).json()] == [detail["id"]]
        assert not zarr.known(Z.sample_id(str(other)))


def test_a_helper_failure_leaves_its_lines_in_the_engine_log(tmp_path, capsys):
    script = tmp_path / "python"
    script.write_text(f"#!{sys.executable}\nimport sys\n"
                      "sys.stderr.write('Traceback (most recent call last):\\n  File \"helper.py\", line 9\\n"
                      "KeyError: ome\\n')\nsys.exit(1)\n")
    script.chmod(0o755)
    zarr = Z.ZarrSamples({"zarr": {"python": str(script)}, "proseg": {"max_region_um": 1000.0},
                          "paths": {"cache_dir": str(tmp_path / "cache"), "data_dir": str(tmp_path / "var")}})
    with pytest.raises(Z.HelperFailed) as failed:
        zarr.helper("save", {})
    assert str(failed.value) == "KeyError: ome"                  # the page's one line
    log = capsys.readouterr().out
    assert "helper save failed: KeyError: ome" in log and 'File "helper.py", line 9' in log


def test_a_name_clash_the_helper_finds_answers_as_the_engines_own(tmp_path, capsys):
    """A save's names taken after the engine's own check (another save into the sample meanwhile) were
    refused by the helper with its words only (422): it now answers as the engine's check does, 409 with the names
    taken and the sample's names now, so the dialog can offer Replace."""
    words = "proseg_boundaries already exists in the sample: choose other names or allow replacing it"
    script = tmp_path / "python"
    script.write_text(f"#!{sys.executable}\nimport json\n"
                      f"print(json.dumps({{'error': {words!r}, 'clash': ['proseg_boundaries'], "
                      f"'existing': ['cells', 'proseg_boundaries']}}))\nraise SystemExit(1)\n")
    script.chmod(0o755)
    zarr = Z.ZarrSamples({"zarr": {"python": str(script)}, "proseg": {"max_region_um": 1000.0},
                          "paths": {"cache_dir": str(tmp_path / "cache"), "data_dir": str(tmp_path / "var")}})
    with pytest.raises(Z.HelperFailed) as failed:
        zarr.helper("save", {})
    assert str(failed.value) == words and failed.value.status == 409
    assert failed.value.extra == {"error": words, "clash": ["proseg_boundaries"],
                                  "existing": ["cells", "proseg_boundaries"]}
    assert "helper save failed: " + words in capsys.readouterr().out


def test_a_saves_arguments_reach_the_helper_whatever_their_size(tmp_path):
    """A save's arguments go to the helper on stdin: one command-line argument is limited to 128 KiB, and a stain
    save's provenance (every point of its removal areas) can pass that, so the save could never start ("Argument list
    too long"). The command line names only where it writes, the run's output folder, which the start's check for a
    save still at work looks for there (leftovers.named: a save helper still at work may be reading the run's folder). The stand-in is the
    real helper's main with its save stood in, which answers what it was given."""
    import threading
    import time
    from sopa_companion.engine import leftovers
    go = tmp_path / "go"
    script = tmp_path / "python"
    script.write_text(f"#!{sys.executable}\nimport json, os, sys, time\n"
                      "from sopa_companion.zarr_bridge import helper\n"
                      "def save(args):\n"
                      f"    while not os.path.exists({str(go)!r}):\n"
                      "        time.sleep(0.02)\n"
                      "    return {'argv': sys.argv[3:], 'size': len(json.dumps(args))}\n"
                      "helper.COMMANDS['save'] = save\n"
                      "sys.exit(helper.main(sys.argv[2:]))\n")
    script.chmod(0o755)
    zarr = Z.ZarrSamples({"zarr": {"python": str(script)}, "proseg": {"max_region_um": 1000.0},
                          "paths": {"cache_dir": str(tmp_path / "cache"), "data_dir": str(tmp_path / "var")}})
    run = tmp_path / "runs" / "20261023-140304-a"
    ring = [[round(0.123 + k * 0.001, 3), round(0.456 + k * 0.002, 3)] for k in range(12_000)]
    args = {"path": str(tmp_path / "s.zarr"), "output": str(run / "output"),
            "provenance": {"kind": "stain", "filters": {"areas": [{"id": 1, "points": ring}]}}}
    assert len(json.dumps(args)) > 128 * 1024
    answers, failures = [], []

    def save():
        try:
            answers.append(zarr.helper("save", args))
        except Exception as exc:                        # noqa: BLE001 - said by the assertions below
            failures.append(exc)
    thread = threading.Thread(target=save)
    thread.start()
    try:
        deadline = time.monotonic() + 30
        while thread.is_alive() and time.monotonic() < deadline and not leftovers.named([run]):
            time.sleep(0.05)
        assert leftovers.named([run, tmp_path / "runs" / "20261023-140304-b"]) == {str(run)}, failures
    finally:
        go.write_text("")
        thread.join(30)
    assert not failures and answers == [{"argv": ["save", "-", str(run / "output")], "size": len(json.dumps(args))}]


# ---- a sample without transcripts ---------------------------------------------------------------------------------------
def test_a_sample_without_transcripts_opens_for_stain_segmentation_only(tmp_path):
    """Image-only samples (and Visium) were refused at open, although stain segmentation needs an image only."""
    path = store(tmp_path)
    image = write_image(path, "scan", np.full((3, 40, 60), 7, np.uint16), ["DAPI", "CD31", "PanCK"], levels=1,
                        to_frame=[[0.25, 0.0, 10.0], [0.0, 0.25, 20.0], [0.0, 0.0, 1.0]])
    zarr = samples(tmp_path)
    stand_in(zarr, inspection(path, points_key=None, points=[], points_elements=[], images=[image], z={"kind": "none"},
                              shapes=[shape("spots", 9, geometry="Point")],
                              suggested_prior={"cells": None, "nuclei": None}))
    detail = zarr.open(str(path))
    assert detail["transcripts"] is None and detail["points_key"] is None and detail["tools"] == ["stain"]
    assert detail["extent"] == [10.0, 20.0, 25.0, 30.0] and detail["image_key"] == "scan"      # where the image lies
    assert detail["prior_options"] == {"cells": [], "nuclei": []} and detail["z_range"] is None
    assert detail["suggested_prior"] == {"cells": None, "nuclei": None}
    assert [n["code"] for n in detail["notes"]] == ["image_only"]
    assert detail["notes"][0]["text"] == "This sample has no transcripts: only stain segmentation runs here."
    assert detail["overview"]["width"] == 60 and [c["name"] for c in detail["channels"]] == ["DAPI", "CD31", "PanCK"]
    assert zarr.list()[0]["id"] == detail["id"]
    # Neither transcripts nor an image it can show: refused in plain words, not remembered.
    other = store(tmp_path, "shapes_only.zarr")
    zarr.inspected = inspection(other, points_key=None, points=[], images=[image_entry("turned", ["DAPI"],
                                                                                       to_frame=False)])
    with pytest.raises(SampleError, match="no transcripts and no image"):
        zarr.open(str(other))
    assert not zarr.known(Z.sample_id(str(other)))


# ---- Recent past 30 samples -----------------------------------------------------------------------------------------
def test_recent_lists_thirty_samples_and_still_knows_the_others(tmp_path):
    """Past 30 samples a sample's detail said 'open it again', while the layer routes opened it again, which
    put it back on top of the list. A sample opened again for one of its runs is known, and the list stays as it was."""
    zarr = samples(tmp_path)
    stand_in(zarr)
    paths = []
    for k in range(Z.RECENT_SHOWN + 2):
        paths.append(store(tmp_path, f"s{k:02d}.zarr"))
        zarr.inspected = inspection(paths[-1])
        zarr.open(str(paths[-1]))
    first, second = Z.sample_id(str(paths[0])), Z.sample_id(str(paths[5]))
    listed = [e["id"] for e in zarr.list()]
    assert len(listed) == Z.RECENT_SHOWN and first not in listed and listed[0] == Z.sample_id(str(paths[-1]))
    assert second in listed
    assert zarr.known(first) and zarr.path(first) == str(paths[0].resolve())
    zarr.details.clear()                                       # an engine started since: built from its inspection
    assert zarr.detail(first)["id"] == first
    # A run's sample forgotten from the list, opened again for the run: known again, the list as it was.
    zarr.forget(second)
    assert not zarr.known(second)
    zarr.inspected = inspection(paths[5])
    zarr.open(str(paths[5]), recent=False)
    now = [e["id"] for e in zarr.list()]
    assert zarr.known(second) and second not in now and [e for e in now if e in listed] == [e for e in listed if e != second]
    again = samples(tmp_path)                                  # the registry as the next engine reads it
    assert again.known(second) and [e["id"] for e in again.list()] == now
    # opened again: first on the list again
    zarr.open(str(paths[5]))
    assert zarr.list()[0]["id"] == second and len(zarr.list()) == Z.RECENT_SHOWN
    assert zarr.entry(second)["path"] == str(paths[5].resolve()) and zarr.entry("znobody") is None
    zarr.forget(first)
    assert not zarr.known(first) and zarr.entry(first) is None
