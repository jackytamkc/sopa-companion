"""API mode's runs (engine/runs.py) beyond test_api_runs.py: what a deliver event carries, where an in-place save may
go, and where an API session keeps InstanSeg's weights. The session is test_api_runs.py's fake, the object's view its
double: nothing is written into a sample."""
from pathlib import Path

from conftest import wait_for
from test_api_runs import FakeSession, Target, delivered
from test_runs_save import session_run
from sopa_companion.api import protocol
from sopa_companion.engine.runs import Runs, user_cache
from sopa_companion.engine.samples import Samples


def api_runs(config, tmp_path, **session):
    zarr = Target(tmp_path / "view.zarr")
    fake = FakeSession(zarr.root, **session)
    return Runs(config, Samples(config), zarr, session=fake), zarr, fake


def test_a_stain_deliver_event_carries_no_outline_points_however_much_was_drawn(config, tmp_path, monkeypatch):
    """A stain save's provenance holds its lasso, the part a stopped run covered and every removed area (up to 200 of
    5,000 points). Inlined in the deliver event it passed the 64 KiB pipe; now the event leaves the points out and stays
    a few kilobytes, while the save itself (into the zarr, and onto the staged shapes the notebook reads them from)
    keeps them whole."""
    runs, zarr, fake = api_runs(config, tmp_path, target=None, view="in_place")
    fake._target = zarr.root
    lasso = [[100.0 + i % 50 * 2.5, 50.0 + i // 50 * 1.25] for i in range(2000)]
    region = {"kind": "polygon", "points": lasso, "bbox": [100.0, 50.0, 222.5, 98.75]}
    covered = {"region": {"kind": "polygon", "points": lasso[:1500], "bbox": [100.0, 50.0, 222.5, 86.25]},
               "done": 3, "of": 6, "why": "tiles", "pieces": 1}
    run = session_run(runs, config, "drawn", kind="stain", region=region, covered=covered)
    ring = [[150.0 + i * 0.01, 60.0 + (i * 37) % 101 * 0.1] for i in range(500)]
    filters = {"version": 2, "areas_version": 1, "next": 201, "areas": [{"id": i, "points": ring} for i in range(200)],
               "size": {"min": 5, "max": None}, "brightness": {}}
    monkeypatch.setattr(runs.stain.exclusions, "load", lambda run: filters)
    monkeypatch.setattr(runs, "_stain_keep", lambda run, state: run.dir / "stain" / "keep.npy")
    runs.save_to_zarr("drawn", "cellpose_boundaries_1", None)
    event = delivered(fake)
    line = protocol.encode_line(event)
    assert len(line) < 16 * 1024 and '"points"' not in line
    assert event["provenance"]["filters"]["areas"] == [{"id": i} for i in range(200)]
    assert event["provenance"]["region"] == {"kind": "polygon", "bbox": region["bbox"]}
    assert event["provenance"]["covered"]["region"] == {"kind": "polygon", "bbox": covered["region"]["bbox"]}
    saved = zarr.saved[-1]["provenance"]                     # what the helper records on the shapes: whole
    assert saved["filters"] == filters and saved["region"] == region and saved["covered"] == covered
    fake.receive(event["id"])
    fake.apply(event["id"])
    wait_for(lambda: run.meta["zarr_save"]["state"] == "saved")
    runs.close("discard")


def test_an_in_place_save_never_goes_into_another_zarr_than_the_one_open(config, tmp_path):
    """Working in place, the object's zarr is the sample open here. Should the notebook's object point to another zarr
    (its path was relative and the kernel's working folder changed, or it was written elsewhere), a save is refused in
    plain words and nothing is written: never another sample's zarr of the same name."""
    other = tmp_path / "elsewhere" / "view.zarr"
    other.mkdir(parents=True)
    runs, zarr, fake = api_runs(config, tmp_path, target=str(other), view="in_place")
    run = session_run(runs, config, "elsewhere")
    runs.save_to_zarr("elsewhere", "proseg_boundaries", "proseg_table")
    wait_for(lambda: (run.meta["zarr_save"] or {}).get("state") == "failed")
    # The two zarrs by the ends of their paths that tell them apart, short enough for the page.
    assert run.meta["zarr_save"]["message"] == (f"Not saved: your SpatialData now belongs to elsewhere/view.zarr, not to "
                                                f"{Path(zarr.root).resolve().parent.name}/view.zarr, which is open here. "
                                                "Open the app again from your notebook to save there.")
    assert zarr.saved == [] and fake.events == [] and not any(runs.deliveries.glob("*"))
    fake._target = zarr.root                                  # the bound zarr again: saved there
    runs.save_to_zarr("elsewhere", "proseg_boundaries", "proseg_table")
    event = delivered(fake)
    assert zarr.delivers[-1]["path"] == str(Path(zarr.root).resolve())
    fake.receive(event["id"])
    fake.apply(event["id"])
    wait_for(lambda: run.meta["zarr_save"]["state"] == "saved")
    runs.close("discard")


def test_an_api_session_keeps_instanseg_weights_in_the_users_cache(config, tmp_path, monkeypatch):
    """A session's folder goes when it closes: its downloaded InstanSeg weights went with it, so every session fetched
    them again, and one without a connection never had InstanSeg. They are kept in the user's cache; the desktop keeps
    its own, in its data folder. (InstanSeg is dormant: this is where they go while its internal switch is on.)"""
    config["stain"]["instanseg_enabled"] = True
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    runs, _, _ = api_runs(config, tmp_path)
    assert user_cache() == tmp_path / "cache" / "sopa-companion"
    assert runs.stain.instanseg_models == tmp_path / "cache" / "sopa-companion" / "instanseg-models"
    env = runs.stain.helper_env(tmp_path / "run-cache")
    assert {env[key] for key in ("INSTANSEG_BIOIMAGEIO_PATH", "INSTANSEG_MODEL_PATH", "INSTANSEG_TORCHSCRIPT_PATH")} \
        == {str(tmp_path / "cache" / "sopa-companion" / "instanseg-models")}
    runs.close("discard")
    desktop = Runs(config, Samples(config), Target(tmp_path / "other.zarr"))
    assert desktop.stain.instanseg_models == Path(config["paths"]["data_dir"]) / "instanseg-models"
    desktop.close("discard")
    monkeypatch.setenv("XDG_CACHE_HOME", "relative/cache")               # not a full path: ignored, as XDG says
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert user_cache() == tmp_path / "home" / ".cache" / "sopa-companion"
