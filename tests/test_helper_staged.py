"""The helper's save in API mode : `path` null (no target), the
engine's own `view` zarr (the same transaction, never consolidated, its failure alone not failing the save),
`frame_from` (the zarr whose transformations place the result), `stage_to` (the built elements for the notebook,
unconsolidated) and `consolidate`. The desktop passes none of them and is answered as before. CPU only: the tests'
synthetic sample (tests/make_spatialdata.py) copied into tmp_path, the helper run in the sopa environment."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil

import pytest

from conftest import needs_sopa
from test_helper_alpha import copy, helper, plain, save, snapshot  # noqa: F401  (save: the desktop's call)
from test_zarr import dead_pid, read_spatialdata, synthetic  # noqa: F401

KINDS = ("images", "points", "shapes", "tables")


def api_save(synthetic, *, env=None, **places):
    """The proseg template's results saved the API way: the desktop's arguments with `path` and the API ones given."""
    args = {"path": None, "points_key": "transcripts", "output": str(synthetic / "template"),
            "shapes_key": "proseg_boundaries", "table_key": "proseg_table", "set_default": False, "overwrite": False,
            "provenance": {"session": "api"}}
    args.update({key: str(value) if isinstance(value, Path) else value for key, value in places.items()})
    return helper("save", args, env)


def overlay(sample: Path, view: Path) -> Path:
    """An API session's overlay view of `sample`: each kind's own group, every element a link to the sample's,
    root attrs as the sample's, never consolidated."""
    view.mkdir(parents=True)
    for folder, target in [(view, sample)] + [(view / kind, sample / kind) for kind in KINDS if (sample / kind).is_dir()]:
        folder.mkdir(exist_ok=True)
        group = json.loads((target / "zarr.json").read_text())
        group.pop("consolidated_metadata", None)
        (folder / "zarr.json").write_text(json.dumps(group))
        if folder != view:
            for element in sorted(target.iterdir()):
                if element.is_dir():
                    os.symlink(element, folder / element.name)
    return view


def consolidated(store: Path) -> set[str]:
    """The element paths a zarr's consolidated metadata lists (none: not consolidated)."""
    listed = json.loads((store / "zarr.json").read_text()).get("consolidated_metadata") or {}
    return {key for key in listed.get("metadata", {}) if key.count("/") == 1}


def links(store: Path) -> dict[str, bool]:
    return {f"{kind}/{e.name}": e.is_symlink() for kind in KINDS if (store / kind).is_dir() for e in (store / kind).iterdir()
            if e.is_dir()}


def state(store: Path) -> tuple[dict, dict]:
    """A zarr's files as snapshot has them, its root metadata by content: a rollback consolidates a consolidated zarr
    again, which rewrites that file as it was."""
    files = snapshot(store)
    files.pop("zarr.json", None)
    return files, json.loads((store / "zarr.json").read_text())


def elements(store: Path) -> str:
    return read_spatialdata(store, "print(sorted(s.shapes), sorted(s.tables), s.attrs.get('boundaries_shapes'))")


@pytest.fixture
def sample(synthetic, tmp_path):
    return copy(synthetic / "synth.zarr", tmp_path / "data" / "DU.zarr")


@needs_sopa
def test_the_desktops_save_is_answered_as_before_and_says_where_it_wrote(synthetic, sample):
    """No API argument: today's transaction into `path`, consolidated; the answer adds where it went, nothing else."""
    status, answer = save(synthetic, sample, set_default=True)
    assert status == 0, answer
    assert answer == {"shapes_key": "proseg_boundaries", "table_key": "proseg_table", "cells": 6, "genes": 7,
                      "set_default": True, "replaced": [], "staged": None, "written_to": str(sample.resolve()),
                      "view_written": None, "view_error": None, "warnings": []}
    assert {"shapes/proseg_boundaries", "tables/proseg_table"} <= consolidated(sample)
    assert elements(sample) == ("['cell_boundaries', 'nucleus_boundaries', 'proseg_boundaries'] ['proseg_table'] "
                                "proseg_boundaries")


@needs_sopa
def test_a_save_with_no_target_stages_the_result_for_the_notebook_alone(synthetic, sample, tmp_path):
    """An object without a zarr to write (unbacked, or sopa's auto-save off): the result is built in the frame of
    the zarr the app reads (frame_from) and written only to the staged folder, unconsolidated; no zarr is touched."""
    stage = tmp_path / "data" / "deliveries" / "d1" / "staged.zarr"
    before = snapshot(sample)
    status, answer = api_save(synthetic, frame_from=sample, stage_to=stage, set_default=True)
    assert status == 0, answer
    assert answer["staged"] == str(stage) and answer["written_to"] is None and answer["view_written"] is None
    assert answer["cells"] == 6 and answer["replaced"] == []
    assert snapshot(sample) == before and not (sample.parent / ".proseg-live-backups").exists()
    assert not consolidated(stage) and sorted(p.name for p in stage.iterdir()) == ["shapes", "tables", "zarr.json"]
    staged = read_spatialdata(stage, "t = s.tables['proseg_table']\n"
                                     "print(len(s.shapes['proseg_boundaries']), t.n_obs, t.n_vars, "
                                     "t.uns['spatialdata_attrs']['region'], t.uns['sopa_companion']['session'])")
    assert staged == "6 6 7 proseg_boundaries api"


@needs_sopa
def test_the_result_is_placed_by_the_zarr_the_app_reads(synthetic, sample, tmp_path):
    """frame_from: the transformations come from the view the run was made on, not from the target: here
    a view whose transcripts sit 100 µm further right."""
    view = copy(sample, tmp_path / "data" / "view.zarr")
    read_spatialdata(view, "from spatialdata.transformations import Translation, set_transformation\n"
                           "set_transformation(s.points['transcripts'], Translation([100, 0], axes=('x', 'y')), 'global', "
                           "write_to_sdata=s)")
    stage = tmp_path / "data" / "deliveries" / "d1" / "staged.zarr"
    status, answer = api_save(synthetic, path=sample, frame_from=view, stage_to=stage)
    assert status == 0, answer
    offset = ("from spatialdata.transformations import get_transformation\n"
              "m = get_transformation(s.shapes['proseg_boundaries'], 'global').to_affine_matrix(('x', 'y'), ('x', 'y'))\n"
              "print(float(m[0, 2]), float(m[1, 2]))")
    assert read_spatialdata(sample, offset) == read_spatialdata(stage, offset) == "100.0 0.0"


@needs_sopa
def test_an_overlay_view_takes_the_same_save_unconsolidated_and_nothing_goes_through_a_link(synthetic, sample, tmp_path):
    """A backed object seen through an overlay: the target zarr gets today's transaction; the view the same one,
    never consolidated. Replacing a name the view links moves the link itself aside, and clearing that backup removes
    the link, never the element it points to (rule 4)."""
    assert save(synthetic, sample)[0] == 0                   # proseg_boundaries + proseg_table there, linked below
    view = overlay(sample, tmp_path / "data" / "view.zarr")
    stage = tmp_path / "data" / "deliveries" / "d1" / "staged.zarr"
    listed = {kind: sorted(p.name for p in (sample / kind).iterdir()) for kind in KINDS}
    status, answer = api_save(synthetic, path=sample, view=view, frame_from=view, stage_to=stage, overwrite=True,
                              set_default=True)
    assert status == 0, answer
    assert (answer["written_to"], answer["view_written"], answer["view_error"]) == (str(sample.resolve()), True, None)
    assert answer["replaced"] == ["proseg_boundaries", "proseg_table"] and answer["staged"] == str(stage)
    assert {kind: sorted(p.name for p in (sample / kind).iterdir()) for kind in KINDS} == listed   # none lost via a link
    linked = links(view)
    assert linked["shapes/proseg_boundaries"] is False and linked["tables/proseg_table"] is False    # written, not linked
    assert all(linked[name] for name in ("points/transcripts", "images/morphology_focus", "shapes/cell_boundaries"))
    assert not consolidated(view) and {"shapes/proseg_boundaries", "tables/proseg_table"} <= consolidated(sample)
    for store in (sample, view):
        assert elements(store) == ("['cell_boundaries', 'nucleus_boundaries', 'proseg_boundaries'] ['proseg_table'] "
                                   "proseg_boundaries")
        assert not (store.parent / ".proseg-live-backups").exists()


@needs_sopa
def test_a_save_into_the_view_alone_replaces_a_link_without_touching_the_sample(synthetic, sample, tmp_path):
    """sopa's auto-save off (overlay): no target, so the sample is never written, not even through the view's
    links. A save killed in the view is rolled back by the next one, which still leaves the view unconsolidated."""
    assert save(synthetic, sample)[0] == 0
    view = overlay(sample, tmp_path / "data" / "view.zarr")
    before = snapshot(sample)
    status, answer = api_save(synthetic, view=view, frame_from=view, overwrite=True, env={"SOPA_COMPANION_TEST_DIE_AFTER": "view:move"})
    assert status == 137 and not (view / "shapes" / "proseg_boundaries").exists()
    [folder] = (view.parent / ".proseg-live-backups").iterdir()
    assert (folder / "shapes" / "proseg_boundaries").is_symlink()                # the link itself was moved aside
    assert json.loads((folder / "journal.json").read_text())["consolidate"] is False
    status, answer = api_save(synthetic, view=view, frame_from=view, overwrite=True,
                              stage_to=tmp_path / "data" / "deliveries" / "d2" / "staged.zarr")
    assert status == 0, answer
    assert (answer["written_to"], answer["view_written"], answer["replaced"]) == (None, True,
                                                                                  ["proseg_boundaries", "proseg_table"])
    assert snapshot(sample) == before and not consolidated(view)
    assert links(view)["shapes/proseg_boundaries"] is False and not (view.parent / ".proseg-live-backups").exists()


@needs_sopa
def test_a_link_moved_aside_whose_element_has_gone_is_still_put_back(synthetic, sample, tmp_path):
    """A Replace in the view moves the link itself aside. Killed there, and the element it points to removed from the
    sample meanwhile (the notebook deleted it), the link in the backup points nowhere: it is put back all the same, as
    the view had it, and the backup cleared. Taken for never moved (it "does not exist"), it stayed behind, said to be a
    file its journal does not list."""
    assert save(synthetic, sample)[0] == 0
    view = overlay(sample, tmp_path / "data" / "view.zarr")
    status, _ = api_save(synthetic, view=view, frame_from=view, overwrite=True,
                         env={"SOPA_COMPANION_TEST_DIE_AFTER": "view:move"})
    assert status == 137
    [folder] = (view.parent / ".proseg-live-backups").iterdir()
    moved, element = folder / "shapes" / "proseg_boundaries", sample / "shapes" / "proseg_boundaries"
    shutil.rmtree(element)
    assert moved.is_symlink() and not moved.exists()
    status, answer = api_save(synthetic, view=view, frame_from=view, shapes_key="second_boundaries",
                              table_key="second_table")
    assert status == 0 and answer["view_written"] is True and answer["warnings"] == [], answer
    assert os.readlink(view / "shapes" / "proseg_boundaries") == str(element)
    assert not (view.parent / ".proseg-live-backups").exists() and links(view)["shapes/second_boundaries"] is False


@needs_sopa
def test_a_link_its_journal_does_not_list_is_kept_and_said(synthetic, sample, tmp_path):
    """A view's backup folder holding a link its journal does not list (an element moved aside before it was
    journalled) counts as something left there, as a file does: the folder is kept, marked rolled back, and the save
    says so and goes ahead. Cleared with the folder, the link (the view's only way to that element) was lost."""
    assert save(synthetic, sample)[0] == 0
    view = overlay(sample, tmp_path / "data" / "view.zarr")
    folder = view.parent / ".proseg-live-backups" / f"{view.name}-20260101-000000-{dead_pid()}"
    (folder / "shapes").mkdir(parents=True)
    os.replace(view / "shapes" / "proseg_boundaries", folder / "shapes" / "proseg_boundaries")
    (folder / "journal.json").write_text(json.dumps({"state": "writing", "path": str(view.resolve()), "moved": [],
                                                     "written": []}))
    status, answer = api_save(synthetic, view=view, frame_from=view, shapes_key="second_boundaries",
                              table_key="second_table")
    assert status == 0 and answer["view_written"] is True, answer
    assert answer["warnings"] == [f"An earlier save into this sample was interrupted and could not be fully undone: 1 "
                                  f"file(s) its journal does not list were left there. What it set aside is kept in "
                                  f"{folder}"]
    assert os.readlink(folder / "shapes" / "proseg_boundaries") == str(sample / "shapes" / "proseg_boundaries")
    assert json.loads((folder / "journal.json").read_text())["state"] == "rolled_back"


@needs_sopa
def test_a_failure_in_the_target_fails_the_save_and_drops_the_staged_result(synthetic, sample, tmp_path):
    """A failure in the target's transaction fails the save: the target is rolled back as on the desktop, and the
    staged result goes too, so the notebook never gets what the zarr does not have."""
    stage = tmp_path / "data" / "deliveries" / "d1" / "staged.zarr"
    before = state(sample)
    status, answer = api_save(synthetic, path=sample, frame_from=sample, stage_to=stage,
                              env={"SOPA_COMPANION_TEST_FAIL_AFTER": "shapes"})
    assert status == 1 and "injected failure after writing shapes" in answer["error"], answer
    assert not stage.exists() and stage.parent.is_dir() and state(sample) == before
    assert not (sample.parent / ".proseg-live-backups").exists()


@needs_sopa
def test_a_failure_in_the_view_alone_still_saves_and_delivers(synthetic, sample, tmp_path):
    """A failure in the view's transaction alone does not fail the save: the target has the result and the staged
    copy is there for the notebook; the answer says the app's layers do not show it. The view is rolled back, without
    consolidating it (it keeps the tables group its table needed). A name the view already holds, without Replace, is
    such a failure too."""
    view = overlay(sample, tmp_path / "data" / "view.zarr")
    linked, attrs = links(view), json.loads((view / "zarr.json").read_text())
    stage = tmp_path / "data" / "deliveries" / "d1" / "staged.zarr"
    status, answer = api_save(synthetic, path=sample, view=view, frame_from=view, stage_to=stage,
                              env={"SOPA_COMPANION_TEST_FAIL_AFTER": "view:tables"})
    assert status == 0, answer
    assert (answer["written_to"], answer["view_written"]) == (str(sample.resolve()), False)
    assert answer["view_error"] == "RuntimeError: injected failure after writing tables" and plain(answer["view_error"])
    assert links(view) == linked and json.loads((view / "zarr.json").read_text()) == attrs and not consolidated(view)
    assert stage.is_dir() and (sample / "tables" / "proseg_table").is_dir()
    assert not (view.parent / ".proseg-live-backups").exists()
    before = snapshot(view)
    status, answer = api_save(synthetic, path=None, view=view, frame_from=view, table_key="proseg_table_2",
                              shapes_key="cell_boundaries", stage_to=tmp_path / "data" / "deliveries" / "d2" / "staged.zarr")
    assert status == 0 and answer["view_written"] is False, answer
    assert answer["view_error"].startswith("cell_boundaries already exists in the sample"), answer
    assert snapshot(view) == before


@needs_sopa
def test_a_save_whose_only_place_is_the_view_fails_with_it(synthetic, sample, tmp_path):
    """No target and nothing staged: the view is the one place the result goes, so its failure (its transaction, or an
    unfinished save in it) is the save's, said as a target's is (a clash names its names, for Replace), and nothing is
    written; it never answers that it saved."""
    view = overlay(sample, tmp_path / "data" / "view.zarr")
    before = snapshot(view)
    status, answer = api_save(synthetic, view=view, frame_from=view, shapes_key="cell_boundaries",
                              table_key="proseg_table_2")
    assert status == 1 and answer["error"].startswith("cell_boundaries already exists in the sample"), answer
    assert answer["clash"] == ["cell_boundaries"] and plain(answer["error"])
    assert snapshot(view) == before and not (view.parent / ".proseg-live-backups").exists()
    folder = view.parent / ".proseg-live-backups" / f"{view.name}-20260101-000000-4242"
    folder.mkdir(parents=True)
    (folder / "journal.json").write_text(json.dumps({
        "state": "writing", "path": str(view.resolve()), "moved": [], "written": [["shapes", "other_boundaries"]],
        "writer": {"pid": 4242, "host": "node-b", "boot": "elsewhere", "started": 1}}))
    status, answer = api_save(synthetic, view=view, frame_from=view)
    assert status == 1, answer
    assert answer["error"].startswith(f"An unfinished save into this sample from node-b is in {folder}"), answer
    assert answer["unfinished"] == str(folder) and snapshot(view) == before


@needs_sopa
def test_consolidating_the_target_is_the_callers_choice(synthetic, sample):
    """`consolidate` (default true) applies to `path`: a target without consolidated metadata stays so when asked.
    Null is the default, as for the other places (a caller passing on a key it was not given)."""
    group = json.loads((sample / "zarr.json").read_text())
    group.pop("consolidated_metadata", None)
    (sample / "zarr.json").write_text(json.dumps(group))
    status, answer = api_save(synthetic, path=sample, consolidate=False)
    assert status == 0 and answer["written_to"] == str(sample.resolve()) and not consolidated(sample), answer
    status, answer = api_save(synthetic, path=sample, shapes_key="second_boundaries", table_key="second_table",
                              consolidate=None)
    assert status == 0 and {"shapes/proseg_boundaries", "shapes/second_boundaries"} <= consolidated(sample), answer


@needs_sopa
def test_a_failed_save_leaves_a_consolidated_zarr_readable_though_asked_not_to_consolidate(synthetic, sample):
    """spatialdata's write_element consolidates a zarr holding consolidated metadata after every element, whatever
    `consolidate` says. A failed save asked not to consolidate then rolled back without consolidating: the metadata
    still listed the elements the rollback had removed, and the sample no longer opened (FileNotFoundError). Such a zarr
    is consolidated again; one without consolidated metadata stays without (the view, in the tests above)."""
    before = elements(sample)
    status, answer = api_save(synthetic, path=sample, consolidate=False, env={"SOPA_COMPANION_TEST_FAIL_AFTER": "tables"})
    assert status == 1 and answer["error"] == "RuntimeError: injected failure after writing tables", answer
    assert {"points/transcripts", "shapes/cell_boundaries"} <= consolidated(sample)
    assert not {"shapes/proseg_boundaries", "tables/proseg_table"} & consolidated(sample)
    assert elements(sample) == before and not (sample.parent / ".proseg-live-backups").exists()


@needs_sopa
def test_a_view_that_is_the_target_is_written_once(synthetic, sample, tmp_path):
    """In place: the engine's view is the object's zarr; one transaction, and the answer says the view has it."""
    status, answer = api_save(synthetic, path=sample, view=sample, stage_to=tmp_path / "staged.zarr")
    assert status == 0, answer
    assert (answer["written_to"], answer["view_written"], answer["view_error"]) == (str(sample.resolve()), True, None)


@needs_sopa
def test_a_format_01_zarr_keeps_its_version(synthetic, sample, tmp_path):
    """A format-0.1 (zarr v2) store seen in place (a format-0.2 view cannot link its elements), its save made the
    sample's segmentation: it stays 0.1 and opens; the notebook's staged copy is in the current format."""
    v01, stage = tmp_path / "data" / "v01.zarr", tmp_path / "data" / "deliveries" / "d1" / "staged.zarr"
    read_spatialdata(sample, "from spatialdata._io import format as F\n"
                             f"s.write({str(v01)!r}, sdata_formats=[F.SpatialDataContainerFormatV01(), F.RasterFormatV01(), "
                             "F.ShapesFormatV02(), F.PointsFormatV01(), F.TablesFormatV01()])")
    status, answer = api_save(synthetic, path=v01, view=v01, stage_to=stage, set_default=True)
    assert status == 0 and (answer["written_to"], answer["view_written"]) == (str(v01.resolve()), True), answer
    attrs = json.loads((v01 / ".zattrs").read_text())
    assert attrs["spatialdata_attrs"]["version"] == "0.1" and attrs["boundaries_shapes"] == "proseg_boundaries"
    assert not (v01 / "zarr.json").exists() and (stage / "zarr.json").is_file()
    assert elements(v01) == "['cell_boundaries', 'nucleus_boundaries', 'proseg_boundaries'] ['proseg_table'] proseg_boundaries"
    assert read_spatialdata(stage, "print(sorted(s.shapes), sorted(s.tables))") == "['proseg_boundaries'] ['proseg_table']"


@needs_sopa
@pytest.mark.parametrize("places,words", [
    ({}, "Nothing to save into: give the sample (path), the app's view or a folder for the result"),
    ({"stage_to": "STAGE"}, "Nothing places the result: give the zarr it was made on (frame_from)"),
    ({"path": "SAMPLE", "consolidate": "yes"}, "'consolidate' must be true or false"),
    ({"path": "SAMPLE", "frame_from": 3}, "'frame_from' must be a folder"),
    ({"frame_from": "SAMPLE", "stage_to": "USED"}, "The folder for the result is in use already: USED"),
])
def test_what_a_save_cannot_do_is_said_before_anything_is_written(synthetic, sample, tmp_path, places, words):
    used = tmp_path / "used"
    used.mkdir()
    (used / "something").write_text("kept")
    names = {"SAMPLE": str(sample), "STAGE": str(tmp_path / "staged.zarr"), "USED": str(used)}
    before = snapshot(sample)
    status, answer = api_save(synthetic, **{k: names.get(v, v) if isinstance(v, str) else v for k, v in places.items()})
    assert status == 1 and answer["error"] == words.replace("USED", str(used)), answer
    assert snapshot(sample) == before and (used / "something").read_text() == "kept"
    assert not (tmp_path / "staged.zarr").exists() and not (sample.parent / ".proseg-live-backups").exists()
