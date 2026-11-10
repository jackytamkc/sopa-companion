"""What other samples showed in the SpatialData helper (zarr_bridge/helper.py): what opening a sample says (the frame's
unit, z, the images left out, the gene column, the cell columns, notes, samples of images only), saving the sample's
segmentation into a format-0.1 sample, a save's journal and its recovery (its own zarr only, dead writers only, every
move journalled first), text transcript ids, circles, the counting rule's control probes and the excluded-genes
pattern. Synthetic samples only, built in the sopa environment ([zarr] python); skipped without it."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from conftest import HERE, sopa_python
from test_zarr import BASE, auth, dead_pid, make_app, needs_sopa, read_spatialdata, segmented, synthetic  # noqa: F401

# Samples, one per case, all small: a 200 x 100 µm field, 15 cells on a 40 µm
# grid, a 2-channel image of 0.5 µm pixels; each variant changes one thing.
ALPHA = r'''
import shutil, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import dask.dataframe as dd, geopandas as gpd, numpy as np, pandas as pd, shapely, spatialdata as sd, zarr
from spatialdata._io import format as F
from spatialdata.models import Image2DModel, PointsModel, ShapesModel
from spatialdata.transformations import Affine, Identity, Scale

root = Path(sys.argv[1])
UM = 0.5                                     # µm per image pixel


def transcripts(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    x, y = rng.uniform(0, 200, n), rng.uniform(0, 100, n)
    cell = (x // 40).astype(int) * 10 + (y // 40).astype(int)
    return pd.DataFrame({"x": x.astype(np.float32), "y": y.astype(np.float32), "z": rng.uniform(5, 15, n).astype(np.float32),
                         "feature_name": pd.Categorical([f"G{i % 7}" for i in range(n)]),
                         "cell_id": np.where(rng.random(n) < .7, [f"c{c}" for c in cell], "UNASSIGNED"),
                         "qv": np.full(n, 40, np.float32)})


def points(frame, transformations=None, **keys):
    coordinates = {"x": "x", "y": "y", **({"z": "z"} if "z" in frame else {})}
    return PointsModel.parse(dd.from_pandas(frame, npartitions=2 if len(frame) else 1), coordinates=coordinates,
                             transformations=transformations or {"global": Identity()}, **keys)


def image(channels=("DAPI", "Boundary"), dtype=np.uint16, transformations=None):
    data = np.random.default_rng(1).integers(0, 200, (len(channels), 200, 400)).astype(dtype)
    return Image2DModel.parse(data, dims=("c", "y", "x"), c_coords=list(channels), scale_factors=[2],
                              transformations=transformations or {"global": Scale([UM, UM], axes=("x", "y"))})


def boxes(half, prefix, transformations=None):
    centres = [(20 + 40 * i, 20 + 40 * j) for i in range(5) for j in range(3)]
    return ShapesModel.parse(gpd.GeoDataFrame(geometry=[shapely.box(cx - half, cy - half, cx + half, cy + half)
                                                        for cx, cy in centres], index=[f"{prefix}-{k}" for k in range(15)]),
                             transformations=transformations or {"global": Identity()})


ATTRS = {"transcripts_dataframe": "transcripts", "cell_segmentation_image": "morphology_focus",
         "transcript_to_cell_assignment": ["cell_id", "UNASSIGNED"]}


def write(name, points_=None, images=None, shapes=None, attrs=ATTRS, formats=None):
    if points_ is None:
        points_ = {"transcripts": points(transcripts(), feature_key="feature_name", instance_key="cell_id")}
    data = sd.SpatialData(points=points_, images={"morphology_focus": image()} if images is None else images,
                          shapes={"cell_boundaries": boxes(18, "cell"), "nucleus_boundaries": boxes(6, "nucleus")}
                          if shapes is None else shapes, attrs=attrs)
    data.write(root / f"{name}.zarr", sdata_formats=formats)


# SpatialData's format 0.1 (zarr v2), as tools of that time wrote it.
write("v01", formats=[F.SpatialDataContainerFormatV01(), F.RasterFormatV01(), F.ShapesFormatV02(), F.PointsFormatV01(),
                      F.TablesFormatV01()])
# Transcript ids as text (MERSCOPE's Ensembl ids).
frame = transcripts()
frame["transcript_id"] = [f"ENSMUST{i:011d}" for i in range(len(frame))]
write("text_ids", points_={"transcripts": points(frame, feature_key="feature_name")})
# Transcripts in image pixels (x, y in 0.5 µm units), placed in µm by Scale(0.5) as the image is; and transcripts
# whose own micrometre system says so: at scale 1 (they are µm), at 0.5 (they are not).
frame = transcripts()
frame["x"], frame["y"] = frame["x"] / UM, frame["y"] / UM
scaled = {"global": Scale([UM, UM], axes=("x", "y"))}
write("pixels", points_={"transcripts": points(frame, scaled, feature_key="feature_name")},
      images={"morphology_focus": image(transformations=scaled)})
for name, scale in (("microns_um", 1.0), ("microns_px", UM)):
    write(name, points_={"transcripts": points(transcripts(), {"global": Identity(), "microns": Scale([scale, scale], axes=("x", "y"))},
                                               feature_key="feature_name")}, images={})
# CosMx's columns: global_cell_id (sample-wide, the attrs' column, 0: no cell), cell_ID (repeats in every FOV),
# cell (each FOV's transcripts in no cell glued into one), CellComp (a compartment: the nucleus flag, never a cell).
frame = transcripts()
fov = (frame["x"] // 100).astype(int) + 1
local = np.where(frame["cell_id"] == "UNASSIGNED", 0, (frame["x"] // 40).astype(int) % 3 + 1)
frame["fov"], frame["cell_ID"], frame["global_cell_id"] = fov, local, np.where(local > 0, fov * 100 + local, 0)
frame["cell"] = [f"c_1_{f}_{c}" for f, c in zip(fov, local)]
frame["CellComp"] = np.where(local > 0, np.random.default_rng(3).choice(["Nuclear", "Cytoplasm", "Membrane"], len(frame)), "0")
frame = frame.drop(columns=["cell_id"]).rename(columns={"feature_name": "target"})
write("cosmx", points_={"transcripts": points(frame, feature_key="target")},
      attrs={**ATTRS, "transcript_to_cell_assignment": ["global_cell_id", 0]})
# An element of empty geometries only; names SpatialData allows that the engine cannot serve ("-" sorts first).
empty = ShapesModel.parse(gpd.GeoDataFrame(geometry=[shapely.Polygon()] * 5), transformations={"global": Identity()})
write("odd_shapes", shapes={"cell_boundaries": boxes(18, "cell"), "nucleus_boundaries": boxes(6, "nucleus"),
                            "empty_outlines": empty, "-cells_leading_dash": boxes(17, "dash"),
                            "-nuclei_leading_dash": boxes(5, "ndash"), "noyaux_é": boxes(5, "noyau")})
# An element SpatialData refuses (a space in its name), listed like the others.
write("invalid")
shutil.copytree(root / "invalid.zarr" / "shapes" / "cell_boundaries", root / "invalid.zarr" / "shapes" / "cell boundaries")
zarr.consolidate_metadata(str(root / "invalid.zarr"))
# Two points elements, nothing saying which holds the transcripts.
frame = transcripts()
write("two_points", points_={"a_points": points(frame[frame["x"] < 100].reset_index(drop=True), feature_key="feature_name"),
                             "b_points": points(frame[frame["x"] >= 100].reset_index(drop=True), feature_key="feature_name")},
      attrs={"cell_segmentation_image": "morphology_focus"})
# The gene column named gene, without feature_key; and no gene column at all.
write("gene_column", points_={"transcripts": points(transcripts().rename(columns={"feature_name": "gene"}))})
write("no_gene", points_={"transcripts": points(transcripts().drop(columns=["feature_name"]))})
# The app's own saves beside the sample's cells: a stain save by its default name, one only its attrs mark.
write("app_saves", shapes={"cells_polygons": boxes(18, "cell"), "cellpose_boundaries_1": boxes(10, "cp"),
                           "my_cells": boxes(12, "mine"), "proseg_boundaries": boxes(14, "sopa")})   # sopa's own proseg
# my_cells: a stain save of the previous version, its record under that version's name.
for key, name in (("cellpose_boundaries_1", "sopa_companion"), ("my_cells", "proseg_live")):
    zarr.open_group(str(root / "app_saves.zarr" / "shapes" / key), mode="r+").attrs[name] = {"kind": "stain"}
write("app_saves_only", shapes={"cellpose_boundaries_1": boxes(10, "cp")})
# Images left out (rotated; in another coordinate system) and, without attrs, an RGB picture listed first.
turn = np.array([[np.cos(.5) * UM, -np.sin(.5) * UM, 0], [np.sin(.5) * UM, np.cos(.5) * UM, 0], [0, 0, 1]])
write("images", images={"a_he": image(("r", "g", "b"), np.uint8), "b_if": image(("DAPI", "CD45", "PanCK", "Vim")),
                        "c_rotated": image(transformations={"global": Affine(turn, input_axes=("x", "y"), output_axes=("x", "y"))}),
                        "d_elsewhere": image(transformations={"aligned": Identity()})},
      attrs={"transcripts_dataframe": "transcripts"})
# Control probes of CosMx and MERSCOPE, no is_gene column.
frame = transcripts()
frame["feature_name"] = pd.Categorical(np.array(["G1", "NegPrb3", "Blank-0001", "SystemControl1", "FalseCode2",
                                                 "NegControlProbe_1", "Negative7"])[np.arange(len(frame)) % 7])
write("controls", points_={"transcripts": points(frame, feature_key="feature_name")})
# Z as plane numbers (whole numbers 0-6); transcripts without z.
frame = transcripts()
frame["z"] = (np.arange(len(frame)) % 7).astype(np.float32)
write("z_planes", points_={"transcripts": points(frame, feature_key="feature_name")})
write("flat", points_={"transcripts": points(transcripts().drop(columns=["z"]), feature_key="feature_name")})
# Transcripts without a row.
write("no_rows", points_={"transcripts": points(transcripts().iloc[:0], feature_key="feature_name")})
# The attrs say 0 marks no cell; the data use -1 (0 is one of 800 cells).
frame = transcripts()
cells = (frame["x"] // 5).astype(int) * 100 + (frame["y"] // 5).astype(int)
frame["cell_id"] = np.where(frame["cell_id"] == "UNASSIGNED", -1, cells).astype(np.int64)
write("token", points_={"transcripts": points(frame, feature_key="feature_name")},
      attrs={**ATTRS, "transcript_to_cell_assignment": ["cell_id", 0]})
# Images only. In pixels ("global" Identity) with a region_of_interest; in a micrometre system the image names; in
# one it reaches only through shapes (Visium HD: spatialdata's own path search finds several equal paths there).
roi = ShapesModel.parse(gpd.GeoDataFrame(geometry=[shapely.box(10, 10, 300, 150)]), transformations={"global": Identity()})
sd.SpatialData(images={"scan": image(transformations={"global": Identity()})},
               shapes={"region_of_interest": roi}).write(root / "image_only.zarr")
roi = ShapesModel.parse(gpd.GeoDataFrame(geometry=[shapely.box(5, 5, 150, 75)]), transformations={"microns": Identity()})
sd.SpatialData(images={"scan": image(transformations={"global": Identity(), "microns": Scale([UM, UM], axes=("x", "y"))})},
               shapes={"region_of_interest": roi}, attrs={"cell_segmentation_image": "scan"}).write(root / "image_only_um.zarr")
spots = {"px": Identity(), "microns": Scale([UM, UM], axes=("x", "y"))}
sd.SpatialData(images={"full": image(transformations={"px": Identity()})},
               shapes={"spots_a": boxes(8, "a", spots), "spots_b": boxes(4, "b", spots)}).write(root / "image_only_via_shapes.zarr")
# ... and one it reaches only through its labels (a cell labels image naming both systems), its region of interest in px.
from spatialdata.models import Labels2DModel
roi = ShapesModel.parse(gpd.GeoDataFrame(geometry=[shapely.box(10, 10, 300, 150)]), transformations={"px": Identity()})
sd.SpatialData(images={"full": image(transformations={"px": Identity()})},
               labels={"cell_labels": Labels2DModel.parse(np.zeros((200, 400), np.int32), dims=("y", "x"), transformations=spots)},
               shapes={"region_of_interest": roi}).write(root / "image_only_via_labels.zarr")
# ... and one single-scale image (read back as a DataArray, not a pyramid) that the attrs name: sopa's toy data's.
single = Image2DModel.parse(np.random.default_rng(1).integers(0, 200, (2, 200, 400)).astype(np.uint16), dims=("c", "y", "x"),
                            c_coords=["DAPI", "Boundary"], transformations={"global": Identity()})
sd.SpatialData(images={"image": single}, attrs={"cell_segmentation_image": "image"}).write(root / "image_only_single.zarr")
'''
# Every sample's inspect in one process (the helper's own function, as `python -m ... inspect` runs it); any other
# failure is kept as the engine sees it (its type, then its text), for that sample's test to show.
INSPECT = r'''
import json, sys
from sopa_companion.zarr_bridge import helper
helper.progress = lambda *a, **k: None
for path in sys.argv[1:]:
    try:
        print(json.dumps({"path": path, "info": helper.inspect({"path": path})}))
    except helper.HelperError as exc:
        print(json.dumps({"path": path, "error": str(exc)}))
    except Exception as exc:
        print(json.dumps({"path": path, "error": f"{type(exc).__name__}: {exc}"}))
'''
NO_TRANSCRIPTS = "This sample has no transcripts: only stain segmentation runs here"
DIE_AFTER_MOVE = {"SOPA_COMPANION_TEST_DIE_AFTER": "move"}       # the helper is killed right after moving a layer aside


@pytest.fixture(scope="session")
def alpha(tmp_path_factory):
    sopa = sopa_python()
    root = tmp_path_factory.mktemp("alpha")
    made = subprocess.run([sopa, "-c", ALPHA, str(root)], capture_output=True, text=True, timeout=600)
    assert made.returncode == 0, made.stderr[-3000:]
    return root


@pytest.fixture(scope="session")
def inspected(alpha, synthetic, segmented) -> dict:
    """name -> inspect's answer, or {"error": its message}, for every alpha sample and the tests' synth.zarr and
    segmented.zarr."""
    paths = sorted(alpha.glob("*.zarr")) + [synthetic / "synth.zarr", segmented]
    done = subprocess.run([sopa_python(), "-c", INSPECT, *map(str, paths)], capture_output=True, text=True, timeout=600,
                          env={**os.environ, "PYTHONPATH": str(HERE.parent.parent)})
    assert done.returncode == 0, done.stderr[-3000:]
    found = {}
    for line in done.stdout.splitlines():
        if line.startswith("{"):
            record = json.loads(line)
            found[Path(record["path"]).name.removesuffix(".zarr")] = record.get("info") or {"error": record["error"]}
    return found


def helper(command: str, args: dict, env: dict | None = None) -> tuple[int, dict]:
    """The sopa helper's answer to a command, as the engine runs it (engine/zarr_samples.py ZarrSamples.helper: the
    arguments on stdin, only where it writes on the command line): (exit status, its JSON answer)."""
    named = [str(args[key]) for key in ("output", "out") if args.get(key)]
    done = subprocess.run([sopa_python(), "-m", "sopa_companion.zarr_bridge.helper", command, "-", *named],
                          input=json.dumps(args), capture_output=True, text=True, timeout=600,
                          env={**os.environ, "PYTHONPATH": str(HERE.parent.parent), **(env or {})})
    lines = [line for line in done.stdout.splitlines() if line.strip()]
    return done.returncode, json.loads(lines[-1]) if lines else {"stderr": done.stderr[-3000:]}


def save(synthetic, sample: Path, shapes_key="proseg_boundaries", table_key="proseg_table", env=None, **options):
    """A proseg run's results (the tests' template output) saved into `sample` by the helper."""
    args = {"path": str(sample), "points_key": "transcripts", "output": str(synthetic / "template"),
            "shapes_key": shapes_key, "table_key": table_key, "set_default": False, "overwrite": False,
            "provenance": {}, **options}
    return helper("save", args, env)


def copy(source: Path, target: Path) -> Path:
    shutil.copytree(source, target)
    return target


def snapshot(path: Path) -> dict:
    return {str(p.relative_to(path)): (p.stat().st_size, p.stat().st_mtime_ns) for p in path.rglob("*") if p.is_file()}


def chosen(info: dict) -> dict:
    return next(p for p in info["points"] if p["key"] == info["points_key"])


def banned_words() -> list[str]:
    """The page's banned words (frontend/src/qc-copy.mjs BANNED_WORDS): new page text must hold none."""
    source = (HERE.parent / "frontend" / "src" / "qc-copy.mjs").read_text()
    block = re.search(r"BANNED_WORDS = Object\.freeze\(\[(.*?)\]\)", source, re.S).group(1)
    return re.findall(r"'([^']+)'", block)


def plain(text: str) -> bool:
    return not [word for word in banned_words() if word.lower() in text.lower()]


# ---- opening a sample ------------------------------------------------------------------------------------------------
@needs_sopa
def test_transcripts_in_pixels_are_told_from_micrometres(inspected):
    """The working frame's unit. Every image at 1 pixel per unit, or the transcripts' own micrometre system at
    another scale: pixels. Images of other pixel sizes, or that system at scale 1: micrometres."""
    units = {name: inspected[name]["units"] for name in ("pixels", "microns_px", "synth", "microns_um")}
    assert [units[name]["frame"] for name in units] == ["pixel", "pixel", "um", "um"]
    assert "0.5" in units["microns_px"]["why"] and "microns" in units["microns_um"]["why"]
    assert all(plain(u["why"]) and "\n" not in u["why"] for u in units.values())


@needs_sopa
def test_only_sample_wide_cell_ids_are_offered_as_cells(inspected):
    """CosMx. The attrs name global_cell_id: cell_ID (it repeats in every FOV), cell (each FOV's transcripts in no
    cell glued into one) and CellComp (a compartment) are not offered as cells; CellComp is the nucleus flag only."""
    columns = chosen(inspected["cosmx"])["prior_columns"]
    assert [(c["column"], c["unassigned"]) for c in columns["cells"]] == [("global_cell_id", "0")]
    assert 0.6 < columns["cells"][0]["assigned_fraction"] < 0.8
    assert [(c["column"], c["nuclear"]) for c in columns["nuclei"]] == [("CellComp", "Nuclear")]
    assert [c["column"] for c in chosen(inspected["synth"])["prior_columns"]["cells"]] == ["cell_id"]   # Xenium's as before


def test_a_categorical_cell_column_keeps_its_text_mark(tmp_path):
    """The rule for columns the attrs do not name: integer ids have no mark but -1 (or a background-sized 0); text,
    categorical text too ("dictionary<values=string, indices=int32>"), has UNASSIGNED. The categorical column was taken
    for integers by its indices' type: no mark, and every transcript counted as in a cell."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from sopa_companion.zarr_bridge import helper as H
    path = tmp_path / "points.parquet"
    ids = pa.array([f"c{i % 50}" if i % 3 else "UNASSIGNED" for i in range(3000)]).dictionary_encode()
    pq.write_table(pa.table({"x": np.arange(3000.0), "y": np.arange(3000.0), "cell_id": ids}), path)
    kind = str(pq.read_schema(path).field("cell_id").type)
    assert kind.startswith("dictionary<values=string")
    stats = H._column_stats([str(path)], ["x", "y"])
    [column] = H._prior_columns([str(path)], {"x": "double", "y": "double", "cell_id": kind}, {}, {}, stats)["cells"]
    assert column["unassigned"] == "UNASSIGNED" and column["assigned_fraction"] == pytest.approx(2 / 3, abs=0.01)


@needs_sopa
def test_an_element_of_empty_geometries_opens_and_is_never_offered(inspected, alpha, config, monkeypatch, synthetic):
    """An element whose geometries are all empty had a NaN extent: the open answered 500 (JSON refuses NaN).
    It now counts 0, has no extent, and is neither offered nor listed."""
    info = inspected["odd_shapes"]
    json.dumps(info, allow_nan=False)
    empty = next(s for s in info["shapes"] if s["key"] == "empty_outlines")
    assert empty["count"] == 0 and empty["geometry"] == {} and "extent" not in empty
    with TestClient(make_app(config, monkeypatch, synthetic), base_url=BASE) as client:
        opened = client.post("/api/v1/samples/open", json={"path": str(alpha / "odd_shapes.zarr")}, headers=auth())
        assert opened.status_code == 200, opened.text
        detail = opened.json()
        offered = [o.get("key") for part in ("cells", "nuclei") for o in detail["prior_options"][part]]
        assert "empty_outlines" not in offered and "cell_boundaries" in offered
        listed = client.get(f"/api/v1/samples/{detail['id']}/segmentations", headers=auth())
        assert listed.status_code == 200 and "empty_outlines" not in [e["key"] for e in listed.json()["segmentations"]]


@needs_sopa
def test_an_element_spatialdata_refuses_is_named_in_one_plain_line(alpha, config, monkeypatch, synthetic):
    """One element SpatialData refuses fails the whole open; the page showed spatialdata's whole text, advice for
    tables (sanitize_table) included. Now one line names the element and why."""
    status, answer = helper("inspect", {"path": str(alpha / "invalid.zarr")})
    assert status == 1 and answer["error"] == (
        "This sample cannot be opened: shapes 'cell boundaries': its name has characters SpatialData refuses (letters, "
        "digits, _ . - only). Fix or remove it, then open the sample again")
    with TestClient(make_app(config, monkeypatch, synthetic), base_url=BASE) as client:
        opened = client.post("/api/v1/samples/open", json={"path": str(alpha / "invalid.zarr")}, headers=auth())
        assert opened.status_code == 422 and answer["error"] in opened.json()["detail"]
        assert "sanitize_table" not in opened.text


@needs_sopa
def test_names_the_engine_cannot_serve_are_never_suggested(inspected):
    """"-cells_leading_dash" sorted first and was suggested, then its outlines answered 404 (the engine's KEY).
    The suggestion takes only names the engine serves; the others stay listed, for the engine to name them."""
    info = inspected["odd_shapes"]
    assert info["suggested_prior"] == {"cells": {"kind": "shapes", "key": "cell_boundaries"},
                                       "nuclei": {"kind": "shapes", "key": "nucleus_boundaries"}}
    assert {"-cells_leading_dash", "-nuclei_leading_dash", "noyaux_é"} <= {s["key"] for s in info["shapes"]}


@needs_sopa
def test_several_points_elements_are_listed_with_what_chose_one(inspected):
    """With several points elements and nothing to say which, the first was opened without a word. inspect lists
    them all and says what chose the one it uses."""
    info = inspected["two_points"]
    assert (info["points_key"], info["points_choice"], info["points_elements"]) == ("a_points", "first", ["a_points", "b_points"])
    assert (inspected["synth"]["points_choice"], inspected["synth"]["points_elements"]) == ("attrs", ["transcripts"])


@needs_sopa
def test_the_gene_column_is_chosen_once_and_used_by_every_command(inspected, alpha, tmp_path):
    """Without feature_key the sample opened, then every run and Compare failed ("need x, y and a gene column").
    inspect chooses the column (feature_key, feature_name, else the one of gene, target, target_name, gene_name),
    prepare and count use it, and a sample with none says so at open, naming its columns."""
    assert inspected["gene_column"]["gene_column"] == "gene" and inspected["synth"]["gene_column"] == "feature_name"
    sample = alpha / "gene_column.zarr"
    status, answer = helper("prepare", {"path": str(sample), "points_key": "transcripts", "region": {"kind": "full"},
                                        "prior": {"cells": {"kind": "shapes", "key": "cell_boundaries"}, "nuclei": None},
                                        "out": str(tmp_path / "input.parquet")})
    assert status == 0, answer
    import pyarrow.parquet as pq
    assert set(pq.read_table(tmp_path / "input.parquet")["feature_name"].to_pylist()) == {f"G{i}" for i in range(7)}
    status, answer = helper("count", {"path": str(sample), "key": "cell_boundaries", "points_key": "transcripts",
                                      "out": str(tmp_path / "counts.npz")})
    assert status == 0 and np.load(tmp_path / "counts.npz")["genes"].tolist() == [f"G{i}" for i in range(7)], answer
    error = inspected["no_gene"]["error"]
    assert error.startswith("The transcripts have no gene column (they have x, y, z, cell_id, qv)") and plain(error)


@needs_sopa
def test_the_apps_own_saves_are_suggested_only_without_other_cells(inspected):
    """A stain save of one box (cellpose_boundaries_1) sorted before cells_polygons and became the suggestion. The
    app's saves (default names, their attrs: sopa_companion, or proseg_live from the previous version) come after the
    sample's other cells."""
    info = inspected["app_saves"]
    assert info["suggested_prior"]["cells"] == {"kind": "shapes", "key": "cells_polygons"}
    assert {s["key"] for s in info["shapes"] if s.get("app_save")} == {"cellpose_boundaries_1", "my_cells"}  # not sopa's
    assert inspected["app_saves_only"]["suggested_prior"]["cells"] == {"kind": "shapes", "key": "cellpose_boundaries_1"}
    # an app's proseg save is known by its table (uns/sopa_companion, or the previous version's uns/proseg_live), its
    # stain saves by their attrs too
    flagged = {s["key"] for s in inspected["segmented"]["shapes"] if s.get("app_save")}
    assert flagged == {"proseg_boundaries", "proseg_boundaries_2", "cellpose_boundaries_1", "instanseg_boundaries_1"}


@needs_sopa
def test_images_left_out_say_why_and_an_rgb_picture_is_not_the_default(inspected):
    """Rotated images, and images in no coordinate system of the transcripts, were dropped without a word; without
    the attrs the first image listed (an H&E) was the default."""
    info = inspected["images"]
    assert info["skipped_images"] == [{"key": "c_rotated", "reason": "rotated relative to the transcripts"},
                                      {"key": "d_elsewhere", "reason": "no coordinate system in common with the transcripts"}]
    assert info["default_image"] == "b_if" and all(plain(s["reason"]) for s in info["skipped_images"])
    assert (inspected["synth"]["default_image"], inspected["synth"]["skipped_images"]) == ("morphology_focus", [])


@needs_sopa
def test_z_plane_numbers_are_told_from_micrometres(inspected):
    """Z as plane numbers (whole numbers over a few values) was taken as µm; a sample without z has none."""
    assert inspected["z_planes"]["z"] == {"kind": "index", "levels": 7}
    assert inspected["flat"]["z"] == {"kind": "none"} and inspected["synth"]["z"] == {"kind": "um"}


@needs_sopa
def test_transcripts_without_a_row_say_they_are_empty(inspected):
    """Empty transcripts were reported as "no usable x/y coordinates"."""
    assert inspected["no_rows"] == {"error": "The sample's transcripts are empty"}


@needs_sopa
def test_a_no_cell_mark_the_data_contradict_is_read_from_the_data(inspected):
    """The attrs say 0 marks no cell, but 0 is a cell like any other and -1 holds the background: the column read
    "100 % assigned". -1 is used, and a note says so."""
    info = inspected["token"]
    [column] = chosen(info)["prior_columns"]["cells"]
    assert column["unassigned"] == "-1" and 0.6 < column["assigned_fraction"] < 0.8 and "note" not in column
    [note] = info["notes"]
    assert note["code"] == "unassigned_token" and note["text"] == "In the transcripts' cell_id column, -1 marks no cell"
    assert plain(note["text"]) and plain(note["detail"]) and "\n" not in note["detail"]
    assert inspected["synth"]["notes"] == []


@needs_sopa
def test_a_sample_of_images_only_opens_for_stain_segmentation(inspected, alpha, tmp_path):
    """A sample without transcripts was refused. It opens in one of its coordinate systems (a micrometre one its
    image reaches, else "global"), its shapes and region of interest in that frame; what needs transcripts says so."""
    info = inspected["image_only"]
    assert (info["points_key"], info["points_choice"], info["points"], info["points_elements"]) == (None, None, [], [])
    assert info["frame"] == {"kind": "system", "name": "global"} and info["units"]["frame"] == "pixel"
    assert info["extent"] == [0, 0, 400, 200] and info["default_image"] == "scan" and info["z"] == {"kind": "none"}
    assert info["suggested_prior"] == {"cells": None, "nuclei": None} and info["gene_column"] is None
    microns = inspected["image_only_um"]
    assert microns["frame"] == {"kind": "system", "name": "microns"} and microns["units"]["frame"] == "um"
    image = next(i for i in microns["images"] if i["key"] == "scan")
    assert np.allclose(image["to_frame"], [[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1]]) and microns["extent"] == [0, 0, 200, 100]
    through = inspected["image_only_via_shapes"]      # reached through its shapes only
    assert through["frame"] == {"kind": "system", "name": "microns"} and through["extent"] == [0, 0, 200, 100]
    sample = alpha / "image_only_um.zarr"
    status, answer = helper("shapes", {"path": str(sample), "key": "region_of_interest", "points_key": None,
                                       "out": str(tmp_path / "roi")})
    assert status == 0 and answer["extent"] == pytest.approx([5, 5, 150, 75]), answer
    status, answer = helper("shapes", {"path": str(alpha / "image_only_via_shapes.zarr"), "key": "spots_a",
                                       "points_key": None, "out": str(tmp_path / "spots")})
    assert status == 0 and answer["extent"] == pytest.approx([6, 6, 94, 54]), answer
    # Reached through its labels only: inspect read the labels and took micrometres; the commands read them too, so
    # its outlines (and stain saves) sit in that frame, not in pixels, twice as large (shapes, tissue and save read no
    # labels, and settled on "px").
    via = inspected["image_only_via_labels"]
    assert via["frame"] == {"kind": "system", "name": "microns"} and via["extent"] == [0, 0, 200, 100]
    status, answer = helper("shapes", {"path": str(alpha / "image_only_via_labels.zarr"), "key": "region_of_interest",
                                       "points_key": None, "out": str(tmp_path / "roi_labels")})
    assert status == 0 and answer["extent"] == pytest.approx([5, 5, 150, 75]), answer
    status, answer = helper("tissue", {"path": str(sample), "image": image, "points_key": None})
    assert status == 0 and answer["area_um2"] == pytest.approx(145 * 70), answer
    for command, args in (("prepare", {"region": {"kind": "full"}, "out": str(tmp_path / "x.parquet"),
                                       "prior": {"cells": {"kind": "shapes", "key": "region_of_interest"}}}),
                          ("count", {"key": "region_of_interest", "out": str(tmp_path / "x.npz")})):
        status, answer = helper(command, {"path": str(sample), "points_key": None, **args})
        assert status == 1 and answer["error"] == NO_TRANSCRIPTS
    assert plain(NO_TRANSCRIPTS)


@needs_sopa
def test_a_single_scale_image_the_attrs_name_opens(inspected):
    """The frame of a sample without transcripts started from `named image or first image`: a
    single-scale image is a DataArray, whose truth is its pixels', so the open read the whole image and failed
    ("The truth value of an array with more than one element is ambiguous"). sopa's toy data without its transcripts
    is such a sample."""
    single = inspected["image_only_single"]
    assert "error" not in single, single
    assert single["frame"] == {"kind": "system", "name": "global"} and single["default_image"] == "image"
    assert single["extent"] == [0, 0, 400, 200] and [i["key"] for i in single["images"]] == ["image"]


@needs_sopa
def test_each_image_is_stamped_as_it_is_on_disk(inspected, synthetic, tmp_path):
    """The helper's part: the engine kept an image's overview, statistics and open arrays after the image
    was written again under its name (new size, transformation or range), and the sample's mtime does not see a
    rewrite of the image's own metadata. inspect stamps each image (image_stamp, which the engine can compute itself):
    the same while nothing changes, another once its transformation alone is written again."""
    from sopa_companion.zarr_bridge.helper import image_stamp
    [image] = inspected["synth"]["images"]
    stamp = image_stamp(synthetic / "synth.zarr", "morphology_focus")
    assert image["stamp"] == stamp and re.fullmatch(r"[0-9a-f]{12}", stamp)
    assert image_stamp(synthetic / "synth.zarr", "morphology_focus") == stamp
    assert image_stamp(synthetic / "synth.zarr", "missing") is None
    assert all(i.get("stamp") for name in ("images", "image_only", "image_only_single") for i in inspected[name]["images"])
    sample = copy(synthetic / "synth.zarr", tmp_path / "samples" / "synth.zarr")
    before = image_stamp(sample, "morphology_focus")
    read_spatialdata(sample, "from spatialdata.transformations import Scale, set_transformation\n"
                             "set_transformation(s.images['morphology_focus'], Scale([0.25, 0.25], axes=('x', 'y')), "
                             "'global', write_to_sdata=s)")
    after = image_stamp(sample, "morphology_focus")
    assert after not in (None, before)
    status, answer = helper("inspect", {"path": str(sample)})
    assert status == 0, answer
    [image] = answer["images"]
    assert image["stamp"] == after and image["to_frame"][0][0] == pytest.approx(0.25)


@needs_sopa
def test_a_stain_save_into_a_sample_of_images_only_sits_in_its_frame(synthetic, alpha, tmp_path):
    """A stain run's cells saved into a sample without transcripts: in its frame's coordinate system, and placed
    in its others as its image is. A proseg save there has nothing to save from."""
    sample = copy(alpha / "image_only_um.zarr", tmp_path / "samples" / "scan.zarr")
    run = tmp_path / "run"
    run.mkdir()
    squares = [[[10, 10], [20, 10], [20, 30], [10, 30], [10, 10]], [[30, 10], [40, 10], [40, 20], [30, 20], [30, 10]]]
    (run / "boundaries.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "id": i + 1, "properties": {}, "geometry": {"type": "Polygon", "coordinates": [ring]}}
        for i, ring in enumerate(squares)]}))
    status, answer = helper("save", {"path": str(sample), "points_key": None, "output": str(run), "kind": "stain",
                                     "shapes_key": "cellpose_boundaries_1", "table_key": None, "set_default": False,
                                     "overwrite": False, "provenance": {"kind": "stain", "method": "cellpose"}})
    assert status == 0 and answer["cells"] == 2, answer
    placed = read_spatialdata(sample, "import numpy as np\nfrom spatialdata.transformations import get_transformation\n"
                                      "g = s.shapes['cellpose_boundaries_1']\nt = get_transformation(g, get_all=True)\n"
                                      "print(sorted(t), round(float(t['global'].to_affine_matrix(('x', 'y'), ('x', 'y'))[0, 0]), 3), "
                                      "[round(float(v), 1) for v in g.total_bounds])")
    assert placed == "['global', 'microns'] 2.0 [10.0, 10.0, 40.0, 30.0]"
    status, answer = save(synthetic, sample, points_key=None)       # the engine's points_key: null
    assert status == 1 and answer["error"] == NO_TRANSCRIPTS


# ---- the commands ----------------------------------------------------------------------------------------------------
@needs_sopa
def test_text_transcript_ids_are_numbered_not_cast(alpha, tmp_path):
    """A text transcript_id (MERSCOPE's Ensembl ids) was cast to uint64 and every run failed (ArrowInvalid). It is
    numbered by row, as a missing one is."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    out = tmp_path / "input.parquet"
    status, answer = helper("prepare", {"path": str(alpha / "text_ids.zarr"), "points_key": "transcripts",
                                        "region": {"kind": "full"}, "out": str(out),
                                        "prior": {"cells": {"kind": "shapes", "key": "cell_boundaries"}, "nuclei": None}})
    assert status == 0 and answer["assigned"] > 0, answer
    ids = pq.read_table(out)["transcript_id"]
    assert ids.type == pa.uint64() and ids.to_pylist() == list(range(answer["transcripts"]))


@needs_sopa
def test_circles_become_outlines_and_seed_a_run(segmented, tmp_path):
    """Circles (points with a radius) failed as outlines and as a run's cells (geopandas' buffer given quad_segs
    twice: TypeError)."""
    status, answer = helper("shapes", {"path": str(segmented), "key": "spots", "points_key": "transcripts",
                                       "out": str(tmp_path / "spots")})
    assert status == 0 and answer["polygons"] == 2 and answer["extent"] == pytest.approx([15, 15, 65, 25], abs=0.01), answer
    status, answer = helper("prepare", {"path": str(segmented), "points_key": "transcripts", "region": {"kind": "full"},
                                        "prior": {"cells": {"kind": "shapes", "key": "spots"}, "nuclei": None},
                                        "out": str(tmp_path / "input.parquet")})
    assert status == 0 and answer["prior_cells"] == 2 and answer["assigned"] > 0, answer


@needs_sopa
def test_control_probes_are_not_counted_as_genes(alpha, tmp_path):
    """CosMx's NegPrb, SystemControl, FalseCode, Negative and MERSCOPE's Blank were counted as genes by Compare
    (and modelled by runs): one default for both, as the runs' excluded-genes pattern."""
    from sopa_companion.zarr_bridge.helper import COUNT_RULE, NOT_GENES
    for name in ("NegPrb3", "Blank-0001", "SystemControl1", "FalseCode2", "Negative7", "NegControlProbe_1",
                 "UnassignedCodeword_1", "DeprecatedCodeword_2", "Intergenic_Region_3"):
        assert NOT_GENES.search(name), name
    for name in ("G1", "ACTB", "NEGR1", "BLANKET"):
        assert not NOT_GENES.search(name), name
    assert COUNT_RULE == "qv20-v4"                   # no count made the old way is reused
    status, answer = helper("count", {"path": str(alpha / "controls.zarr"), "key": "cell_boundaries",
                                      "points_key": "transcripts", "out": str(tmp_path / "counts.npz")})
    assert status == 0 and np.load(tmp_path / "counts.npz")["genes"].tolist() == ["G1"], answer


def test_excluded_genes_patterns_that_could_hang_or_read_differently_are_refused():
    """The excluded-genes pattern is compiled by Python's re for Compare's counts. One that repeats a part
    that itself repeats can take minutes over one gene name; one with a set inside a set reads differently than in
    proseg; a long one is refused too. Each in plain words, by the helper and by the engine's counting rule."""
    from sopa_companion.engine.samples import SampleError
    from sopa_companion.engine.zarr_samples import ZarrSamples
    from sopa_companion.zarr_bridge.helper import NOT_GENES_PATTERN, HelperError, _excluded, count_rule
    for pattern, words in ((r"(\w+)+-", "repeats a part"), (r"(a*b)*", "repeats a part"), (r"((a+)?)+", "repeats a part"),
                           (r"(\w{1,30}){1,30}-", "repeats a part"), (r"(\w|\w|\w)+-", "repeats a choice"),
                           (r"(a|ab|abc)*x", "repeats a choice"),
                           ("[[:alpha:]]", "reads differently"), ("[A--B]", "reads differently"),
                           (r"\p{L}", "not a valid pattern"), ("G" * 201, "too long")):
        with pytest.raises(HelperError, match=words):
            _excluded(pattern)
        with pytest.raises(HelperError, match=words):
            count_rule(20, pattern)
        with pytest.raises(SampleError, match=words):
            ZarrSamples.count_rule({"exclude": pattern})
    for pattern in (NOT_GENES_PATTERN, "^(Deprecated|NegControl|Unassigned|Intergenic)", r"^MT-.*", r"(ab)+",
                    r"^(Neg|Blank)\d{1,4}$", "G" * 200, r"(\d{3}-){2}", r"(Neg|Blank)?\d+", r"[ab]+-"):
        assert _excluded(pattern).pattern == pattern


def test_commands_read_every_element_that_places_a_frame_without_transcripts():
    """Without transcripts the frame comes from the images and whatever links them to a coordinate system; every
    command reads what inspect reads (shapes, images, labels), or it could settle on another frame."""
    from sopa_companion.zarr_bridge import helper as H
    assert H._selection("transcripts", "shapes") == ("points", "shapes")
    assert H._selection(None, "shapes") == ("shapes", "images", "labels")
    assert H._selection(None, "shapes", "tables") == ("shapes", "tables", "images", "labels")


def test_an_excluded_genes_pattern_is_held_to_a_time_limit(monkeypatch):
    """A pattern can still backtrack over one name in a way no rule sees (open repeats in a row: twelve
    take about 10 s over one control probe's name). Compare's count works the names out under a time limit, and fails
    in plain words when it is up, instead of holding the helper for an hour; the names a pattern matches are as
    re.search finds them."""
    from sopa_companion.zarr_bridge import helper as H
    names = ["NegControlCodeword_0500", "ACTB", "NegPrb3", "Blank-0001"]
    assert H._matching("^(Neg|Blank)", names).tolist() == [True, False, True, True]
    assert H._matching("^MALAT1$", []).tolist() == []
    slow = r"\w*" * 12 + "!"
    assert H._excluded(slow).pattern == slow                      # no rule refuses it
    monkeypatch.setattr(H, "GENE_CHECK_S", 0.5)
    started = time.monotonic()
    with pytest.raises(H.HelperError, match="takes too long to check against this sample's gene names"):
        H._matching(slow, names)
    assert time.monotonic() - started < 5                          # stopped at its limit, not run to its end


# ---- saving into the sample ------------------------------------------------------------------------------------------
@needs_sopa
def test_saving_as_the_samples_segmentation_keeps_a_format_01_sample_readable(synthetic, alpha, tmp_path):
    """"Make it the sample's segmentation" stamped a format-0.1 (zarr v2) sample 0.2; spatialdata then read its
    images as 0.2 and failed (KeyError 'ome'), and so did every later open."""
    sample = copy(alpha / "v01.zarr", tmp_path / "samples" / "v01.zarr")
    status, answer = save(synthetic, sample, set_default=True)
    assert status == 0, answer
    attrs = json.loads((sample / ".zattrs").read_text())
    assert attrs["spatialdata_attrs"]["version"] == "0.1" and attrs["boundaries_shapes"] == "proseg_boundaries"
    assert read_spatialdata(sample, "print(sorted(s.images), sopa.utils.get_boundaries(s, return_key=True)[0])") == \
        "['morphology_focus'] proseg_boundaries"


@needs_sopa
def test_a_proseg_saves_table_names_its_run(synthetic, tmp_path):
    """The older runs: a run saved into its sample already is found by its id, which
    `segmentations` now gives for a proseg save's table (its provenance, uns/sopa_companion) as for a stain save's shapes."""
    sample = copy(synthetic / "synth.zarr", tmp_path / "samples" / "synth.zarr")
    assert save(synthetic, sample, provenance={"run": "20261023-101010-synth", "iteration": 3})[0] == 0
    status, answer = helper("segmentations", {"path": str(sample)})
    assert status == 0, answer
    assert [(t["key"], t["proseg"], t["run"]) for t in answer["tables"]] == [("proseg_table", True, "20261023-101010-synth")]


@needs_sopa
def test_a_name_clash_names_the_clashing_names_and_the_samples_elements(synthetic, tmp_path):
    """The helper's side: a clash answers which names clash and the sample's elements as they are now, so the
    Save dialog can offer Replace for names saved after its run ended."""
    sample = copy(synthetic / "synth.zarr", tmp_path / "samples" / "synth.zarr")
    assert save(synthetic, sample)[0] == 0
    status, answer = save(synthetic, sample)
    assert status == 1 and answer["clash"] == ["proseg_boundaries", "proseg_table"]
    assert answer["error"] == "proseg_boundaries and proseg_table already exist in the sample: choose other names or allow replacing them"
    assert answer["existing"] == ["cell_boundaries", "morphology_focus", "nucleus_boundaries", "proseg_boundaries",
                                  "proseg_table", "transcripts"]
    status, answer = save(synthetic, sample, table_key="other_table")
    assert status == 1 and answer["clash"] == ["proseg_boundaries"] and "allow replacing it" in answer["error"]


@needs_sopa
def test_a_save_killed_right_after_moving_a_layer_aside_is_rolled_back_whole(synthetic, tmp_path):
    """A layer being replaced was moved aside before the journal listed it. Killed in between, the next save's
    recovery found nothing to put back and deleted the backup folder holding the layer's only copy. Every move is now
    journalled first: the layer comes back as it was."""
    sample = copy(synthetic / "synth.zarr", tmp_path / "samples" / "synth.zarr")
    assert save(synthetic, sample)[0] == 0
    before = snapshot(sample / "shapes" / "proseg_boundaries")
    status, _ = save(synthetic, sample, overwrite=True, env=DIE_AFTER_MOVE)
    assert status == 137 and not (sample / "shapes" / "proseg_boundaries").exists()
    [folder] = (sample.parent / ".proseg-live-backups").iterdir()
    journal = json.loads((folder / "journal.json").read_text())
    assert journal["moved"] == [["shapes", "proseg_boundaries"]] and journal["path"] == str(sample.resolve())
    assert journal["writer"]["pid"] == int(folder.name.rsplit("-", 1)[1]) and journal["state"] == "writing"
    status, answer = save(synthetic, sample, shapes_key="second_boundaries", table_key="second_table")
    assert status == 0, answer
    assert snapshot(sample / "shapes" / "proseg_boundaries") == before
    assert not (sample.parent / ".proseg-live-backups").exists()
    assert read_spatialdata(sample, "print(sorted(s.shapes), sorted(s.tables))") == \
        "['cell_boundaries', 'nucleus_boundaries', 'proseg_boundaries', 'second_boundaries'] ['proseg_table', 'second_table']"


@needs_sopa
def test_recovery_only_ever_takes_its_own_samples_journal(synthetic, tmp_path):
    """Every sample of a folder shares its backups folder, and recovery took any journal whose
    folder began with the sample's name: a save into x.zarr rolled back x.zarr-copy's interrupted save against x.zarr,
    deleting x.zarr's layer and moving the copy's in. Only folders named exactly for the sample, whose journal names it,
    are its own."""
    mine = copy(synthetic / "synth.zarr", tmp_path / "samples" / "x.zarr")
    other = copy(synthetic / "synth.zarr", tmp_path / "samples" / "x.zarr-copy")
    for sample in (mine, other):
        assert save(synthetic, sample)[0] == 0
    assert save(synthetic, other, overwrite=True, env=DIE_AFTER_MOVE)[0] == 137
    # A folder of mine's name whose journal names another zarr (a sample moved away; this one took its name).
    stranger = mine.parent / ".proseg-live-backups" / f"{mine.name}-20260101-000000-{dead_pid()}"
    shutil.copytree(mine / "shapes" / "nucleus_boundaries", stranger / "shapes" / "nucleus_boundaries")
    (stranger / "journal.json").write_text(json.dumps({"state": "writing", "path": "/elsewhere/x.zarr",
                                                       "moved": [["shapes", "nucleus_boundaries"]], "written": []}))
    backups, layer = snapshot(mine.parent / ".proseg-live-backups"), snapshot(mine / "shapes" / "proseg_boundaries")
    status, answer = save(synthetic, mine, shapes_key="second_boundaries", table_key="second_table")
    assert status == 0, answer
    assert snapshot(mine.parent / ".proseg-live-backups") == backups          # neither journal was touched
    assert snapshot(mine / "shapes" / "proseg_boundaries") == layer           # mine's own layer, untouched
    # The copy's own next save puts its layer back.
    status, answer = save(synthetic, other, shapes_key="second_boundaries", table_key="second_table")
    assert status == 0 and (other / "shapes" / "proseg_boundaries").is_dir(), answer
    assert [p.name for p in (mine.parent / ".proseg-live-backups").iterdir()] == [stranger.name]


@needs_sopa
def test_a_save_still_running_is_never_rolled_back(synthetic, tmp_path):
    """The save lock is each user's own, so another user's save into this sample can be under way: its journal
    names a live process. This save then stops and touches nothing. The same journal once that process is gone (a
    later process given its pid is not it) is rolled back."""
    from sopa_companion.zarr_bridge import helper as H
    sample = copy(synthetic / "synth.zarr", tmp_path / "samples" / "synth.zarr")
    assert save(synthetic, sample)[0] == 0
    folder = sample.parent / ".proseg-live-backups" / f"{sample.name}-20260101-000000-{os.getpid()}"
    (folder / "shapes").mkdir(parents=True)
    shutil.move(str(sample / "shapes" / "proseg_boundaries"), str(folder / "shapes" / "proseg_boundaries"))
    started = H._process(os.getpid())[1]
    journal = {"state": "writing", "path": str(sample.resolve()), "moved": [["shapes", "proseg_boundaries"]], "written": [],
               "writer": {"pid": os.getpid(), "host": socket.gethostname(), "boot": H._boot(), "started": started}}
    (folder / "journal.json").write_text(json.dumps(journal))
    kept = snapshot(folder)
    status, answer = save(synthetic, sample, shapes_key="second_boundaries", table_key="second_table")
    assert status == 1 and answer["error"] == (f"Another save into this sample is under way (process {os.getpid()}): try "
                                               "again once it has finished")
    assert answer["unfinished"] == str(folder)              # the engine can tell it from a save that failed
    assert snapshot(folder) == kept and not (sample / "shapes" / "second_boundaries").exists()
    journal["writer"]["started"] = started + 1
    (folder / "journal.json").write_text(json.dumps(journal))
    status, answer = save(synthetic, sample, shapes_key="second_boundaries", table_key="second_table")
    assert status == 0 and (sample / "shapes" / "proseg_boundaries").is_dir() and not folder.exists(), answer


def test_an_earlier_versions_journal_is_not_taken_for_a_process_given_its_pid_since():
    """A journal from before journals named their writer gives only its folder's pid and stamp. Taken for
    alive whenever that pid ran again, it refused every save into the sample ("Another save into this sample is under
    way") once the pid belonged to another process, or to a thread of one. The save began before its folder was made:
    a process that started after the stamp is not it, nor is a thread; one that started before can be."""
    from sopa_companion.zarr_bridge import helper as H
    long_ago = H._stamp_time("20200101-000000")
    assert long_ago is not None and H._stamp_time("2020-01-01") is None
    later = subprocess.Popen(["sleep", "60"])                    # started after that stamp: not its writer
    try:
        assert H._alive({"pid": later.pid, "stamp": long_ago}) is False
        assert H._alive({"pid": later.pid}) is True               # nothing to go by: it may be
    finally:
        later.kill()
        later.wait()
    ready, done, ids = threading.Event(), threading.Event(), []

    def hold():
        ids.append(threading.get_native_id())
        ready.set()
        done.wait(30)
    thread = threading.Thread(target=hold, daemon=True)
    thread.start()
    assert ready.wait(30)
    try:                                                         # a thread's id: no process has that pid now
        assert H._leader(ids[0]) is False and H._alive({"pid": ids[0], "stamp": long_ago}) is False
    finally:
        done.set()
        thread.join()
    assert H._leader(os.getpid()) is True
    assert H._alive({"pid": os.getpid(), "stamp": time.time()}) is True     # started before now: it may be the writer


@needs_sopa
def test_an_earlier_versions_save_whose_pid_runs_again_is_recovered(synthetic, tmp_path):
    """The same, through a save: an earlier version's interrupted save whose folder's pid now belongs to a
    process started since is rolled back (it was refused as under way), its layer put back."""
    sample = copy(synthetic / "synth.zarr", tmp_path / "samples" / "synth.zarr")
    assert save(synthetic, sample)[0] == 0
    later = subprocess.Popen(["sleep", "300"])
    try:
        folder = sample.parent / ".proseg-live-backups" / f"{sample.name}-20200101-000000-{later.pid}"
        (folder / "shapes").mkdir(parents=True)
        shutil.move(str(sample / "shapes" / "proseg_boundaries"), str(folder / "shapes" / "proseg_boundaries"))
        (folder / "journal.json").write_text(json.dumps({"state": "writing", "moved": [["shapes", "proseg_boundaries"]],
                                                         "written": []}))
        status, answer = save(synthetic, sample, shapes_key="second_boundaries", table_key="second_table")
    finally:
        later.kill()
        later.wait()
    assert status == 0, answer
    assert (sample / "shapes" / "proseg_boundaries").is_dir() and not folder.exists()


@needs_sopa
def test_an_earlier_versions_interrupted_save_is_recovered_and_what_it_did_not_list_kept(synthetic, tmp_path):
    """A journal from before journals named their zarr and process is this sample's by its folder
    name, and is rolled back once that pid is gone. What its folder holds that its journal does not list (that version
    moved before it journalled) is never deleted: the folder stays, marked rolled back, and is left alone after.
    The save that found it says so in its answer (warnings) and goes ahead. Refusing it once protected
    nothing (the folder is marked, so the next save went ahead), and a save refused while the app closes loses its run."""
    sample = copy(synthetic / "synth.zarr", tmp_path / "samples" / "synth.zarr")
    assert save(synthetic, sample)[0] == 0
    folder = sample.parent / ".proseg-live-backups" / f"{sample.name}-20260101-000000-{dead_pid()}"
    for kind, name in (("shapes", "proseg_boundaries"), ("tables", "proseg_table")):
        (folder / kind).mkdir(parents=True)
        shutil.move(str(sample / kind / name), str(folder / kind / name))
    (folder / "journal.json").write_text(json.dumps({"state": "writing", "moved": [["shapes", "proseg_boundaries"]],
                                                     "written": []}))
    status, answer = save(synthetic, sample, shapes_key="second_boundaries", table_key="second_table")
    unlisted = sum(1 for p in (folder / "tables").rglob("*") if p.is_file())
    assert status == 0 and answer["warnings"] == [
        f"An earlier save into this sample was interrupted and could not be fully undone: {unlisted} file(s) its journal "
        f"does not list were left there. What it set aside is kept in {folder}"], answer
    assert plain(answer["warnings"][0]) and (sample / "shapes" / "second_boundaries").is_dir()
    assert (sample / "shapes" / "proseg_boundaries").is_dir() and not (sample / "tables" / "proseg_table").exists()
    assert (folder / "tables" / "proseg_table").is_dir()
    assert json.loads((folder / "journal.json").read_text())["state"] == "rolled_back"
    kept = snapshot(folder)
    status, answer = save(synthetic, sample, shapes_key="third_boundaries", table_key="third_table")
    assert status == 0 and answer["warnings"] == [] and snapshot(folder) == kept, answer
    assert read_spatialdata(sample, "print(sorted(s.shapes), sorted(s.tables))") == (
        "['cell_boundaries', 'nucleus_boundaries', 'proseg_boundaries', 'second_boundaries', 'third_boundaries'] "
        "['second_table', 'third_table']")


def test_a_rollback_cut_short_is_safe_to_run_again(tmp_path, monkeypatch):
    """A Replace save lists the names it replaces as written too. A rollback cut short after putting an
    original back (a second Ctrl+C) left its journal at "writing"; the next save's recovery then deleted every written
    name, the original put back among them, found no copy left in the backup and removed the backup folder: the
    sample's segmentation was gone, and that save said it succeeded. A replaced name is now removed only as its
    original is put back, while that original is still in the backup."""
    from sopa_companion.zarr_bridge import helper as H
    monkeypatch.setattr(H, "progress", lambda *a, **k: None)
    monkeypatch.setattr(H, "_consolidate", lambda path: None)
    sample, backup = tmp_path / "s.zarr", tmp_path / ".proseg-live-backups" / "s.zarr-20260101-000000-1"
    names = [["shapes", "proseg_boundaries"], ["tables", "proseg_table"]]
    for kind, name in names:              # a Replace save killed while writing the table: originals aside, new in place
        (backup / kind / name).mkdir(parents=True)
        (backup / kind / name / "ORIGINAL").write_text(name)
        (sample / kind / name).mkdir(parents=True)
        (sample / kind / name / "NEW").write_text(name)
    (backup / "journal.json").write_text(json.dumps({"state": "writing", "path": str(sample), "moved": names,
                                                     "written": names}))
    put_back = os.replace

    def interrupted(source, target):      # the second Ctrl+C, right after the first original is back
        put_back(source, target)
        raise KeyboardInterrupt
    monkeypatch.setattr(os, "replace", interrupted)
    with pytest.raises(KeyboardInterrupt):
        H._rollback(sample, backup / "journal.json")
    monkeypatch.setattr(os, "replace", put_back)
    assert (sample / "shapes" / "proseg_boundaries" / "ORIGINAL").is_file()
    assert json.loads((backup / "journal.json").read_text())["state"] == "writing"
    assert H._rollback(sample, backup / "journal.json") == []          # the next save's recovery: run again
    for kind, name in names:
        assert [p.name for p in (sample / kind / name).iterdir()] == ["ORIGINAL"], (kind, name)
    assert not backup.exists()


@needs_sopa
def test_a_rollback_run_again_never_deletes_the_originals_it_put_back(synthetic, tmp_path):
    """End to end: the state a rollback cut short leaves (journal "writing", every replaced name listed as
    moved and written, the originals back in the sample, the backup holding its journal only). The next save, under
    new names, keeps both originals and clears the backup."""
    from sopa_companion.zarr_bridge import helper as H
    sample = copy(synthetic / "synth.zarr", tmp_path / "samples" / "synth.zarr")
    assert save(synthetic, sample)[0] == 0
    before = {kind: snapshot(sample / kind / name) for kind, name in (("shapes", "proseg_boundaries"),
                                                                      ("tables", "proseg_table"))}
    folder = sample.parent / ".proseg-live-backups" / f"{sample.name}-20260101-000000-{dead_pid()}"
    for kind in ("shapes", "tables"):
        (folder / kind).mkdir(parents=True)
    names = [["shapes", "proseg_boundaries"], ["tables", "proseg_table"]]
    (folder / "journal.json").write_text(json.dumps({
        "state": "writing", "path": str(sample.resolve()), "moved": names, "written": names,
        "writer": {"pid": dead_pid(), "host": socket.gethostname(), "boot": H._boot(), "started": 1}}))
    status, answer = save(synthetic, sample, shapes_key="second_boundaries", table_key="second_table")
    assert status == 0, answer
    assert {kind: snapshot(sample / kind / name) for kind, name in (("shapes", "proseg_boundaries"),
                                                                    ("tables", "proseg_table"))} == before
    assert not (sample.parent / ".proseg-live-backups").exists()
    assert read_spatialdata(sample, "print(sorted(s.shapes), sorted(s.tables))") == \
        "['cell_boundaries', 'nucleus_boundaries', 'proseg_boundaries', 'second_boundaries'] ['proseg_table', 'second_table']"


@needs_sopa
def test_another_computers_unfinished_save_stops_a_save_into_the_sample(synthetic, tmp_path):
    """Another computer's unfinished save (a sample on a shared disk) was left as it was and this save went
    ahead; rolled back later from that computer, it deleted what this one had saved under its names. Only that computer
    can tell whether it still runs: the save stops, writes nothing and says what to do."""
    sample = copy(synthetic / "synth.zarr", tmp_path / "samples" / "synth.zarr")
    assert save(synthetic, sample)[0] == 0
    folder = sample.parent / ".proseg-live-backups" / f"{sample.name}-20260101-000000-4242"
    folder.mkdir(parents=True)
    (folder / "journal.json").write_text(json.dumps({
        "state": "writing", "path": str(sample.resolve()), "moved": [], "written": [["shapes", "other_boundaries"]],
        "writer": {"pid": 4242, "host": "node-b", "boot": "elsewhere", "started": 1}}))
    kept, layers = snapshot(folder), snapshot(sample)
    status, answer = save(synthetic, sample, shapes_key="other_boundaries", table_key="other_table")
    assert status == 1 and answer["error"] == (
        f"An unfinished save into this sample from node-b is in {folder}. If it is still running, try again once it has "
        "finished; if not, save into this sample once from node-b to undo it, or move that folder out of "
        ".proseg-live-backups (it holds what that save set aside) to save from here"), answer
    assert answer["unfinished"] == str(folder)
    assert plain(answer["error"]) and snapshot(folder) == kept and snapshot(sample) == layers
    shutil.move(str(folder), str(tmp_path / "moved-away"))
    status, answer = save(synthetic, sample, shapes_key="other_boundaries", table_key="other_table")
    assert status == 0 and (sample / "shapes" / "other_boundaries").is_dir(), answer


@needs_sopa
def test_a_rollback_whose_folder_cannot_be_marked_stops_every_save(synthetic, tmp_path):
    """An interrupted save kept for what it holds is marked rolled back, so that it is never rolled back
    again. When the mark could not be written, every later save rolled it back once more, deleting what the save before
    had written under the names it lists. Such a folder now stops every save into the sample until it is moved away."""
    sample = copy(synthetic / "synth.zarr", tmp_path / "samples" / "synth.zarr")
    assert save(synthetic, sample)[0] == 0
    folder = sample.parent / ".proseg-live-backups" / f"{sample.name}-20260101-000000-{dead_pid()}"
    for kind, name in (("shapes", "proseg_boundaries"), ("tables", "proseg_table")):
        (folder / kind).mkdir(parents=True)
        shutil.move(str(sample / kind / name), str(folder / kind / name))
    (folder / "journal.json").write_text(json.dumps({"state": "writing", "moved": [["shapes", "proseg_boundaries"]],
                                                     "written": [["shapes", "second_boundaries"]]}))
    (folder / "journal.json.tmp").mkdir()                    # the journal cannot be written again: no mark
    unlisted = sum(1 for p in (folder / "tables").rglob("*") if p.is_file())
    for _ in range(2):
        status, answer = save(synthetic, sample, shapes_key="second_boundaries", table_key="second_table")
        assert status == 1 and answer["error"] == (
            f"An earlier save into this sample was interrupted and could not be fully undone: {unlisted} file(s) its "
            f"journal does not list were left there; its folder could not be marked as undone. What it set aside is kept "
            f"in {folder}. Move that folder out of .proseg-live-backups to save into this sample again"), answer
        assert answer["unfinished"] == str(folder)
        assert not (sample / "shapes" / "second_boundaries").exists() and not (sample / "tables" / "second_table").exists()
        assert json.loads((folder / "journal.json").read_text())["state"] == "writing"
    assert (sample / "shapes" / "proseg_boundaries").is_dir() and (folder / "tables" / "proseg_table").is_dir()
