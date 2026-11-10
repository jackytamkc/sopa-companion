"""What a run starts from: a sample of images only takes stain runs and refuses proseg plainly; every
proseg run leaves out every platform's control probes unless the user says otherwise; the runs a start
removes are named."""
import copy
import json
from pathlib import Path
import re
import sys

import pytest
from fastapi.testclient import TestClient

from conftest import sopa_python, wait_for
from test_stain import IMAGE, SETTINGS, Images, ended
from sopa_companion.engine.proseg_options import DEFAULT_EXCLUDED_GENES
from sopa_companion.engine.runs import IMAGE_ONLY, RunError, Runs, has_transcripts
from sopa_companion.engine.samples import Samples

BASE = "http://127.0.0.1:8799"
HEADERS = {"Authorization": "Bearer " + "t" * 40, "Origin": BASE}


class ImagesOnly(Images):
    """A sample of images only, as the engine's detail says it: no transcripts, stain segmentation only, its
    extent its image's."""

    def __init__(self, path):
        super().__init__(path)
        self.image = copy.deepcopy(IMAGE)
        self.info = dict(id="image-sample", name="Images", kind="zarr", path=str(path), extent=[0, 0, 400, 300],
                         images=[self.image], transcripts=None, points_key=None, tools=["stain"],
                         prior_options={"cells": [], "nuclei": []}, suggested_prior={"cells": None, "nuclei": None})
        self.saved = []

    def validate_prior(self, sid, prior):
        raise AssertionError("a proseg run's prior was looked at on a sample without transcripts")

    def existing_elements(self, sid):
        return []

    def save(self, sid, output, shapes_key, table_key, set_default, overwrite, provenance, **kwargs):
        self.saved.append((shapes_key, table_key, provenance.get("kind")))
        return {"cells": 6}


@pytest.fixture
def runs(config, tmp_path):
    config["stain"].update(python=sys.executable, helper_module="sopa_companion.tests.fake_stain", tile_pixels=128,
                           overlap_pixels=16)
    manager = Runs(config, Samples(config), ImagesOnly(tmp_path / "images.zarr"))
    yield manager
    manager.close(save=False)


# ---- a sample of images only -----------------------------------------------------------------------------------------
def test_a_sample_of_images_only_refuses_proseg_runs_plainly(runs):
    with pytest.raises(RunError) as refused:
        runs.create("image-sample", region={"kind": "full"}, prior={"cells": {"kind": "shapes", "key": "x"}})
    assert refused.value.status == 409 and str(refused.value) == IMAGE_ONLY
    assert IMAGE_ONLY == "This sample has no transcripts: only stain segmentation runs here"
    assert not runs.runs and not list(runs.root.iterdir())                 # nothing made
    assert not has_transcripts(runs.zarr.info)
    assert has_transcripts({"transcripts": 50}) and has_transcripts({"transcripts": 50, "tools": ["proseg", "stain"]})


def test_stain_runs_work_on_a_sample_of_images_only(runs):
    """A stain run needs only an image: made, segmented and saved without transcripts (the save is the shapes alone)."""
    run = runs.get(runs.create("image-sample", kind="stain", region={"kind": "full"}, stain=SETTINGS)["id"])
    runs.control(run.meta["id"], "resume")
    wait_for(lambda: ended(run))
    assert run.meta["state"] == "done" and run.meta["summary"]["cells"] == 6
    assert run.meta["save_defaults"]["shapes_key"] == "cellpose_boundaries_1"
    runs.save_to_zarr(run.meta["id"], "cellpose_boundaries_1", None)
    wait_for(lambda: (run.meta["zarr_save"] or {}).get("state") == "saved")
    assert runs.zarr.saved == [("cellpose_boundaries_1", None, "stain")]


def test_the_api_refuses_proseg_on_images_only_and_learns_nothing_on_open(config, tmp_path, monkeypatch):
    """POST /runs of kind proseg answers 409 with the plain line; opening such a sample never asks the mixing check
    to learn it (it has no transcripts to learn from)."""
    from sopa_companion.server import create_app
    config["stain"]["python"] = sopa_python()       # a stain run needs the sopa environment (else 503)
    application = create_app(config, token="t" * 40)
    images = ImagesOnly(tmp_path / "images.zarr")
    application.state.runs.zarr = images
    learned = []
    monkeypatch.setattr(application.state.zarr, "open", lambda path, recent=True: images.info)
    monkeypatch.setattr(application.state.qc, "sample_opened", lambda detail: learned.append(detail["id"]))
    with TestClient(application, base_url=BASE, headers=HEADERS) as client:
        assert client.post("/api/v1/samples/open", json={"path": "/data/images.zarr"}).json()["tools"] == ["stain"]
        assert learned == []
        refused = client.post("/api/v1/runs", json={"sample": "image-sample", "region": {"kind": "full"},
                                                     "prior": {"cells": {"kind": "shapes", "key": "x"}}})
        assert refused.status_code == 409 and refused.json() == {"detail": IMAGE_ONLY, "status": 409}
        made = client.post("/api/v1/runs", json={"sample": "image-sample", "kind": "stain", "region": {"kind": "full"},
                                                  "stain": SETTINGS})
        assert made.status_code == 200 and made.json()["kind"] == "stain"


# ---- every platform's control probes stay out ------------------------------------------------------------------------
def test_proseg_runs_leave_out_every_platforms_control_probes(config):
    """proseg's own --xenium default leaves out Xenium's controls only: CosMx's NegPrb and SystemControl, and MERSCOPE's
    Blank, were modelled as genes. Every run now passes the app's default, unless the user set Excluded genes or let
    the controls in; proseg's log shows the command it ran. (Three runs side by side: a test engine's, which one run
    at a time does not bind.)"""
    config["proseg"]["keep_unsaved"] = True
    runs = Runs(config, Samples(config))
    runs.max_active = 3
    try:
        sample = runs.samples.list()[0]["id"]
        run = runs.get(runs.create(sample, [50, 50, 150, 150])["id"])
        command = runs._command(run, [])
        at = command.index("--excluded-genes")
        assert command[at + 1] == DEFAULT_EXCLUDED_GENES and run.meta["proseg_args"] == ["--nthreads", "2"]
        wait_for(lambda: (run.dir / "proseg.log").exists() and "$ " in (run.dir / "proseg.log").read_text())
        assert f"--excluded-genes {DEFAULT_EXCLUDED_GENES}" in (run.dir / "proseg.log").read_text()
        own = runs.get(runs.create(sample, [50, 50, 150, 150], {"excluded-genes": "^MALAT1$"})["id"])
        assert runs._command(own, []).count("--excluded-genes") == 1 and "^MALAT1$" in runs._command(own, [])
        controls = runs.get(runs.create(sample, [200, 100, 300, 200], {"include-neg-ctrls": True})["id"])
        assert "--excluded-genes" not in runs._command(controls, [])
        for run_id in (run.meta["id"], own.meta["id"], controls.meta["id"]):
            runs.control(run_id, "abort")
    finally:
        runs.close(save=False)
    pattern = re.compile(DEFAULT_EXCLUDED_GENES)
    for control in ("NegPrb3", "SystemControl1", "FalseCode12", "Negative2", "Blank-0001", "NegControlProbe_00042",
                    "NegControlCodeword_0500", "UnassignedCodeword_0499", "DeprecatedCodeword_0001", "Intergenic_Region_1"):
        assert pattern.search(control), control
    for gene in ("EPCAM", "ALB", "CD5L", "MALAT1", "NEGR1", "BLNK", "SYSTEMIC"):
        assert not pattern.search(gene), gene


# ---- what a start removes is said ------------------------------------------------------------------------------------
def write_run(config, run_id: str, **meta) -> Path:
    folder = Path(config["paths"]["data_dir"]) / "runs" / run_id
    folder.mkdir(parents=True)
    (folder / "run.json").write_text(json.dumps({"id": run_id, "kind": "proseg", "state": "running",
                                                 "sample_name": "a sample", "created": "2026-01-01T00:00:00", **meta}))
    return folder


def test_a_start_names_the_runs_it_removes(config, capsys):
    """Unsaved work does not survive the engine (a user engine keeps no run from one session to the next; runs
    from before the one-run model with nothing to save go too): each run a start removes is named in the engine
    log (they went without a word)."""
    leftover = write_run(config, "left-sampling", session_model=2)
    older = write_run(config, "older-sampling")
    runs = Runs(config, Samples(config))
    try:
        assert not leftover.exists() and not older.exists() and not runs.runs and runs.older()["runs"] == []
        out = capsys.readouterr().out
        assert "SoPA companion: removed leftover run left-sampling of a sample when the engine started" in out
        assert ("SoPA companion: removed run older-sampling of a sample from an earlier version: it was saved already or "
                "had nothing to save") in out
        assert runs.leftovers["older_removed"] == 1 and [r["id"] for r in runs.leftovers["runs"]] == ["left-sampling"]
    finally:
        runs.close(save=False)
