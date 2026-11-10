"""Samples, crops and the run lifecycle against a fake proseg-live."""
from __future__ import annotations

import gzip

import numpy as np
import pytest

from conftest import wait_for


def test_samples_and_crops(config, tmp_path):
    import pyarrow.parquet as pq
    from sopa_companion.engine.samples import Samples

    samples = Samples(config)
    [sample] = samples.list()
    assert sample["name"] == "TESTREGION-1"
    detail = samples.detail(sample["id"])
    assert detail["width_um"] == pytest.approx(400, abs=0.3) and detail["height_um"] == pytest.approx(300, abs=0.3)
    assert detail["channels"] == ["DAPI", "Boundary"]
    assert samples.overview(sample["id"]).exists() and samples.overview_meta(sample["id"])["um_per_px"] > 0

    count = samples.crop_transcripts(sample["id"], [100, 50, 200, 150], tmp_path / "crop.parquet")
    crop = pq.read_table(tmp_path / "crop.parquet")
    assert count == crop.num_rows > 0
    x, y = crop.column("x_location").to_numpy(), crop.column("y_location").to_numpy()
    assert x.min() >= 100 and x.max() < 200 and y.min() >= 50 and y.max() < 150
    original = pq.read_schema(samples.folder(sample["id"]) / "transcripts.parquet")
    assert crop.schema.remove_metadata() == original.remove_metadata()   # proseg's parquet reader is type-strict

    meta = samples.crop_images(sample["id"], [100, 50, 200, 150], tmp_path / "image")
    assert meta["level"] == 0 and meta["um_per_px"] == pytest.approx(0.2125)
    assert meta["x0"] <= 100 < meta["x0"] + meta["um_per_px"] and len(meta["channels"]) == 2
    assert (tmp_path / "image" / "c1.png").exists()


def make_runs(config):
    from sopa_companion.engine.runs import Runs
    from sopa_companion.engine.samples import Samples
    return Runs(config, Samples(config))


def start(runs, options=None, box=(50, 50, 150, 150)):
    """A run sampling at full speed: every run starts paused on its prior, so it is resumed at once (while preparing)."""
    sample = runs.samples.list()[0]["id"]
    run = runs.create(sample, list(box), options)
    runs.control(run["id"], "speed", "max")
    runs.control(run["id"], "resume")
    return runs.get(run["id"])


# A short schedule of proseg's (the fake writes one frame per iteration): 2 burn-in, 3 sampling of which 1 recorded,
# 2 hill-climb; the end is iteration 7.
SHORT = {"burnin-samples": 2, "samples": 3, "recorded-samples": 1, "hillclimb": 2}


def test_a_run_starts_paused_on_its_prior_then_steps_resumes_pauses_and_finishes(config):
    runs = make_runs(config)
    sample = runs.samples.list()[0]["id"]
    created = runs.create(sample, [50, 50, 150, 150])
    run = runs.get(created["id"])
    assert created["paused"] is True and "schedule" not in created and "start_paused" not in created
    # proseg's own schedule: nothing about it is passed, and proseg holds at its end instead of finishing.
    assert run.meta["proseg_args"] == ["--nthreads", "2"] and "--live-hold" in runs._command(run, [])
    wait_for(lambda: (run.public()["ack"] or {}).get("state") == "paused" and len(run.frames) == 1)
    assert run.meta["state"] == "paused" and run.frames[0]["phase"] == "init" and run.public()["phase"] == "paused"
    assert run.info["hold"] and all(part["frames"] is not None for part in run.info["plan"])
    assert [p["phase"] for p in run.info["plan"]] == ["init", "burnin", "upscale", "sampling", "recorded", "hillclimb"]
    assert "continue 0" in (run.live / "control").read_text().splitlines()
    __import__("time").sleep(0.3)
    assert len(run.frames) == 1, "it waits on the prior"

    runs.control(run.meta["id"], "speed", "max")
    runs.control(run.meta["id"], "step")
    wait_for(lambda: len(run.frames) == 2 and (run.public()["ack"] or {}).get("seq") == 1)
    runs.control(run.meta["id"], "resume")
    wait_for(lambda: len(run.frames) >= 12)
    assert run.meta["state"] == "running" and not run.meta.get("continued")

    runs.control(run.meta["id"], "pause")
    wait_for(lambda: (lambda n: (__import__("time").sleep(.2), len(run.frames) == n)[1])(len(run.frames)))
    runs.control(run.meta["id"], "finish")
    wait_for(lambda: run.meta["state"] == "done")
    names = {o["name"] for o in run.meta["outputs"]}
    assert {"proseg-output.zarr", "cell-polygons.geojson.gz"} <= names
    assert "Finished at iteration" in run.meta["message"] and "schedule_done" not in run.meta

    # Every indexed frame decodes to (2 + nz) planes.
    header = run.frames[-1]
    data = np.frombuffer(gzip.open(run.live / header["file"]).read(), "<u4")
    assert data.size == (2 + header["nz"]) * header["nx"] * header["ny"]


def test_finish_while_paused(config):
    runs = make_runs(config)
    run = start(runs)
    wait_for(lambda: len(run.frames) >= 3)
    runs.control(run.meta["id"], "pause")
    runs.control(run.meta["id"], "finish")
    wait_for(lambda: run.meta["state"] == "done")


def test_the_end_of_the_schedule_pauses_the_run_and_resume_samples_on(config):
    """proseg's schedule runs to its end and the run pauses there by itself (proseg holds); Step runs one more
    iteration, Resume samples on, open-ended ("continued"), until Finish & save."""
    import time
    if not _has_options():
        pytest.skip("option schema not installed")
    runs = make_runs(config)
    run = start(runs, SHORT)
    wait_for(lambda: run.meta.get("schedule_done") and run.meta["state"] == "paused", timeout=60)
    assert run.meta["schedule_done"] == {"iter": 7, "seq": 8} and run.meta["paused"] is True
    assert [(f["iter"], f["phase"]) for f in run.frames] == [
        (0, "init"), (1, "burnin"), (2, "burnin"), (2, "upscale"), (3, "sampling"), (4, "sampling"), (5, "recorded"),
        (6, "hillclimb"), (7, "hillclimb")]
    assert run.public()["ack"] == {"state": "paused", "iter": 7, "seq": 8} and run.public()["phase"] == "paused"
    # Not finishing: a closing engine offers it like any run still sampling (save or discard).
    assert [(r["id"], r["iter"], r["decided"]) for r in runs.pending()] == [(run.meta["id"], 7, False)]
    time.sleep(0.3)
    assert len(run.frames) == 9 and run.process.poll() is None, "it holds there, waiting for the user"
    assert "continue 0" in (run.live / "control").read_text().splitlines()

    runs.control(run.meta["id"], "step")
    wait_for(lambda: len(run.frames) == 10 and (run.public()["ack"] or {}).get("seq") == 9)
    assert run.frames[-1]["phase"] == "continued" and run.frames[-1]["iter"] == 8
    assert run.meta["continued"] is True and run.meta["state"] == "paused"
    assert "continue 1" in (run.live / "control").read_text().splitlines()
    runs.control(run.meta["id"], "resume")
    wait_for(lambda: len(run.frames) >= 20)
    assert {f["phase"] for f in run.frames[9:]} == {"continued"} and run.meta["state"] == "running"
    runs.control(run.meta["id"], "pause")
    runs.control(run.meta["id"], "finish")
    wait_for(lambda: run.meta["state"] == "done")
    assert run.meta["outputs"] and run.frames[-1]["phase"] == "continued"


def test_a_double_step_across_the_end_of_the_schedule_never_replays_after_continuing(config):
    """Paused one iteration before the end of the schedule, Step clicked twice: the first runs the last iteration, the
    second is still pending when proseg starts to hold, and is spent there. Step at the hold then runs exactly one more
    iteration, not two (the held token replayed once sampling went on)."""
    import time
    if not _has_options():
        pytest.skip("option schema not installed")
    runs = make_runs(config)
    sample = runs.samples.list()[0]["id"]
    run = runs.get(runs.create(sample, [50, 50, 150, 150], SHORT)["id"])
    runs.control(run.meta["id"], "speed", "max")
    wait_for(lambda: (run.public()["ack"] or {}).get("state") == "paused" and len(run.frames) == 1)
    for seq in (1, 3, 4, 5, 6, 7):                           # one Step at a time, to iteration 6 (iteration 2: frames 2, 3)
        runs.control(run.meta["id"], "step")
        wait_for(lambda: (run.public()["ack"] or {}).get("seq") == seq and (run.public()["ack"] or {}).get("state") == "paused")
    assert run.frames[-1]["iter"] == 6
    runs.control(run.meta["id"], "step")
    runs.control(run.meta["id"], "step")                     # a double click
    wait_for(lambda: run.meta.get("schedule_done") and run.meta["state"] == "paused", timeout=60)
    assert run.meta["schedule_done"] == {"iter": 7, "seq": 8} and not run.meta.get("continued")
    runs.control(run.meta["id"], "step")                     # one Step at the hold
    wait_for(lambda: (run.public()["ack"] or {}).get("seq") == 9 and (run.public()["ack"] or {}).get("state") == "paused")
    time.sleep(0.4)
    assert [(f["iter"], f["phase"]) for f in run.frames[8:]] == [(7, "hillclimb"), (8, "continued")], "exactly one more"
    runs.control(run.meta["id"], "abort")
    wait_for(lambda: run.meta["state"] == "stopped")


def test_finish_at_the_end_of_the_schedule_saves_proseg_s_own_result(config):
    """Finish & save while the run holds at the end of its schedule: nothing runs after it, the saved state is the
    schedule's last iteration (what proseg itself writes at that point)."""
    if not _has_options():
        pytest.skip("option schema not installed")
    runs = make_runs(config)
    run = start(runs, SHORT)
    wait_for(lambda: run.meta.get("schedule_done") and run.meta["state"] == "paused", timeout=60)
    runs.control(run.meta["id"], "finish")
    wait_for(lambda: run.meta["state"] == "done")
    assert len(run.frames) == 9 and run.frames[-1] == {**run.frames[-1], "iter": 7, "phase": "hillclimb"}
    assert run.public()["ack"]["state"] == "finishing" and run.public()["ack"]["seq"] == 8
    assert not run.meta.get("continued") and "at iteration 7" in run.meta["message"]


def test_the_end_of_the_schedule_never_undoes_a_finish_or_an_abort_on_its_way(config):
    """Finish & save (or Abort) asked for while the last iterations ran: the run reaching its end pauses nothing."""
    from sopa_companion.engine.runs import Run
    runs = make_runs(config)
    for name, meta, stop in (("finishing", {"state": "finishing", "finish_requested": True, "paused": False}, False),
                             ("aborting", {"state": "running", "paused": False}, True)):
        (runs.root / name / "live").mkdir(parents=True)
        run = Run(runs.root / name, {"id": name, "speed": "max", **meta})
        run.stop_requested = stop
        (run.live / "schedule_done").write_text('{"iter": 7, "seq": 8}')
        runs._schedule_done(run)
        assert run.meta["schedule_done"] == {"iter": 7, "seq": 8}
        assert run.meta["state"] == meta["state"] and run.meta["paused"] is False and not (run.live / "control").exists()
    held = Run(runs.root / "held", {"id": "held", "speed": "max", "state": "running", "paused": False})
    (held.live).mkdir(parents=True)
    (held.live / "schedule_done").write_text("torn")              # unreadable: the run still pauses there
    runs._schedule_done(held)
    assert held.meta["state"] == "paused" and held.meta["schedule_done"] == {"iter": None, "seq": None}
    assert "paused 1" in (held.live / "control").read_text().splitlines()


def test_abort_discards(config):
    runs = make_runs(config)
    run = start(runs)
    wait_for(lambda: len(run.frames) >= 2)
    runs.control(run.meta["id"], "abort")
    wait_for(lambda: run.meta["state"] == "stopped")
    assert "without results" in run.meta["message"]


def test_retry_without_factorization(config, monkeypatch):
    monkeypatch.setenv("FAKE_PROSEG_FAIL", "startup")
    runs = make_runs(config)
    run = start(runs)
    wait_for(lambda: len(run.frames) >= 2)
    assert run.meta.get("retried_without_factorization")
    runs.control(run.meta["id"], "finish")
    wait_for(lambda: run.meta["state"] == "done")


def test_region_and_option_validation(config):
    from sopa_companion.engine.runs import RunError
    runs = make_runs(config)
    sample = runs.samples.list()[0]["id"]
    with pytest.raises(RunError, match="at least 20"):
        runs.create(sample, [0, 0, 10, 10])
    runs.max_region = 100.0             # boxes are clamped to the section first, then size-limited
    with pytest.raises(RunError, match="limited"):
        runs.create(sample, [0, 0, 200, 200])
    runs.max_region = 1200.0
    for options in ("open", ["samples", 3], {3: 1}):    # an old page's schedule name where the options go, or junk
        with pytest.raises(RunError, match="Options must be an object"):
            runs.create(sample, [0, 0, 100, 100], options)
    if _has_options():
        with pytest.raises(RunError):
            runs.create(sample, [0, 0, 100, 100], {"no-such-flag": 1})
        with pytest.raises(RunError, match="recorded-samples"):
            runs.create(sample, [0, 0, 100, 100], {"samples": 10, "recorded-samples": 20})
    assert not runs.runs, "nothing was created"


def test_close_discards_active_and_finished_runs(config):
    runs = make_runs(config)
    run = start(runs)
    wait_for(lambda: len(run.frames) >= 2)
    runs.close("discard")
    assert not run.dir.exists() and runs.list() == []


@pytest.mark.parametrize("save", [True, False])
def test_test_engines_finish_only_into_the_run_folder(config, save):
    config["proseg"]["keep_unsaved"] = True
    runs = make_runs(config)
    run = start(runs, SHORT)
    wait_for(lambda: run.meta.get("schedule_done"), timeout=60)
    runs.close("add" if save else "discard", timeout=30)
    assert run.dir.exists()
    assert run.meta["state"] == ("done" if save else "stopped")
    assert run.meta["zarr_save"] is None


def test_unsaved_runs_do_not_survive_a_restart(config):
    runs = make_runs(config)
    run = start(runs)
    wait_for(lambda: len(run.frames) >= 2)
    runs.control(run.meta["id"], "abort")
    wait_for(lambda: run.meta["state"] == "stopped")
    again = make_runs(config)
    assert run.meta["id"] not in again.runs and not run.dir.exists()


def test_restart_marks_interrupted(config):
    config["proseg"]["keep_unsaved"] = True
    runs = make_runs(config)
    run = start(runs)
    wait_for(lambda: len(run.frames) >= 2)
    again = make_runs(config)            # a second engine over the same data directory
    assert again.get(run.meta["id"]).meta["state"] == "stopped"
    runs.control(run.meta["id"], "abort")
    wait_for(lambda: run.meta["state"] == "stopped")


def _has_options():
    try:
        from sopa_companion.engine import proseg_options  # noqa: F401
        return True
    except ImportError:
        return False


def test_active_run_limit(config):
    from sopa_companion.engine.runs import RunError
    config["proseg"]["keep_unsaved"] = True
    runs = make_runs(config)
    runs.max_active = 1
    run = start(runs)
    with pytest.raises(RunError, match="already active"):
        start(runs)
    runs.control(run.meta["id"], "abort")
    wait_for(lambda: run.meta["state"] == "stopped")


def test_pause_is_acknowledged(config):
    runs = make_runs(config)
    run = start(runs)
    wait_for(lambda: len(run.frames) >= 3)
    runs.control(run.meta["id"], "pause")
    wait_for(lambda: (run.public()["ack"] or {}).get("state") == "paused")
    ack = run.public()["ack"]
    assert ack["seq"] == run.frames[-1]["seq"] and ack["iter"] == run.frames[-1]["iter"]
    runs.control(run.meta["id"], "finish")
    wait_for(lambda: run.meta["state"] == "done")


def test_abort_while_preparing_never_starts_proseg(config, monkeypatch):
    runs = make_runs(config)
    gate = __import__("threading").Event()
    original = runs.samples.crop_images
    monkeypatch.setattr(runs.samples, "crop_images", lambda *a, **k: (gate.wait(5), original(*a, **k))[1])
    sample = runs.samples.list()[0]["id"]
    run = runs.get(runs.create(sample, [50, 50, 150, 150])["id"])
    wait_for(lambda: run.meta.get("transcripts"))
    runs.control(run.meta["id"], "abort")
    gate.set()
    wait_for(lambda: run.meta["state"] == "stopped")
    assert run.meta.get("pid") is None and not (run.live / "frames.jsonl").exists()


def test_test_engine_close_finishes_running_and_keeps_the_rest(config, monkeypatch):
    config["proseg"]["keep_unsaved"] = True
    runs = make_runs(config)
    runs.max_active = 3
    running = start(runs)
    wait_for(lambda: len(running.frames) >= 3)
    gate = __import__("threading").Event()
    original = runs.samples.crop_images
    monkeypatch.setattr(runs.samples, "crop_images", lambda *a, **k: (gate.wait(5), original(*a, **k))[1])
    sample = runs.samples.list()[0]["id"]
    preparing = runs.get(runs.create(sample, [60, 60, 160, 160])["id"])
    wait_for(lambda: preparing.meta.get("transcripts"))
    closer = __import__("threading").Thread(target=runs.close, kwargs={"save": True})
    closer.start()
    gate.set()
    closer.join(30)
    assert running.meta["state"] == "done" and running.meta["outputs"] and running.dir.exists()
    assert preparing.meta.get("pid") is None and preparing.dir.exists()     # unsaved: discarded
    from sopa_companion.engine.runs import RunError
    with pytest.raises(RunError, match="shutting down"):
        runs.create(sample, [50, 50, 150, 150])


def test_voxel_budget(config):
    from sopa_companion.engine.runs import RunError
    runs = make_runs(config)
    sample = runs.samples.list()[0]["id"]
    if not _has_options():
        pytest.skip("option schema not installed")
    with pytest.raises(RunError, match="million voxels"):
        runs.create(sample, [0, 0, 300, 300], {"voxel-size": 0.05, "burnin-voxel-size": 0.1})


def test_engine_lock_is_exclusive(config):
    import os
    from sopa_companion.server.runtime_file import acquire_engine_lock
    fd = acquire_engine_lock(config)
    try:
        import multiprocessing
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(1) as pool:
            assert pool.apply(_try_lock, (config,)) == "refused"
    finally:
        os.close(fd)


def _try_lock(config):
    from sopa_companion.server.runtime_file import acquire_engine_lock
    try:
        acquire_engine_lock(config)
        return "acquired"
    except RuntimeError:
        return "refused"


def test_new_runs_have_no_source_link(config):
    import inspect
    from sopa_companion.engine.runs import Runs
    assert "source" not in inspect.signature(Runs.create).parameters
    runs = make_runs(config)
    try:
        run = runs.create(runs.samples.list()[0]["id"], [50, 50, 150, 150])
        assert run["session_model"] == 2 and "source" not in run
    finally:
        runs.close("discard")


def test_runs_made_with_the_old_schedules_stay_viewable(config):
    """Runs saved before every run followed proseg's own schedule name theirs ("open": "Until I stop", open-ended;
    "full") and were started paused or not: a test engine (which keeps every run, the harness's fixtures among them)
    lists and serves them as they are."""
    import json
    config["proseg"]["keep_unsaved"] = True
    runs = make_runs(config)
    for name, schedule, plan in (("old-open", "open", [{"phase": "init", "frames": 1}, {"phase": "burnin", "frames": 100},
                                                      {"phase": "sampling", "frames": 60}, {"phase": "recorded", "frames": None}]),
                                 ("old-full", "full", [{"phase": "init", "frames": 1}, {"phase": "burnin", "frames": 200}])):
        folder = runs.root / name
        (folder / "live").mkdir(parents=True)
        meta = {"id": name, "sample": "s", "sample_name": "Old", "box": [0, 0, 100, 100], "state": "done", "created": name,
                "schedule": schedule, "schedule_params": {"label": "Until I stop", "indefinite": schedule == "open"},
                "paused": False, "start_paused": False, "outputs": [], "options": {}}
        (folder / "run.json").write_text(json.dumps(meta))
        (folder / "live" / "info.json").write_text(json.dumps({"indefinite": schedule == "open", "plan": plan,
                                                               "schedule": {"burnin": 100, "samples": 60, "hillclimb": 0}}))
    again = make_runs(config)
    listed = {r["id"]: r for r in again.list()}
    assert listed["old-open"]["schedule"] == "open" and listed["old-open"]["info"]["plan"][-1]["frames"] is None
    assert listed["old-full"]["phase"] == "done" and listed["old-full"]["frames"] == 0


@pytest.mark.parametrize("case", __import__("json").loads(
    (__import__("pathlib").Path(__file__).parent / "data" / "leave_cases.json").read_text()), ids=lambda c: c["name"])
def test_leave_predicate_shared_with_page(case):
    from sopa_companion.engine.runs import finish_and_save, leave_kind
    assert leave_kind(case["run"]) == case["expect"]
    assert finish_and_save(case["run"]) is case["decided"]            # a Finish & save is decided at close


@pytest.mark.parametrize("state", ["preparing", "running", "paused", "finishing", "done", "stopped", "failed"])
@pytest.mark.parametrize("kind", ["proseg", "stain", "patches"])
def test_one_run_refuses_every_kind_while_any_user_run_exists(config, tmp_path, state, kind):
    from sopa_companion.engine.runs import ONE_RUN, Run, RunError
    runs = make_runs(config)
    runs.runs["existing"] = Run(tmp_path, {"id": "existing", "state": state})
    with pytest.raises(RunError) as refused:
        runs.create("irrelevant", kind=kind)
    assert refused.value.status == 409 and str(refused.value) == ONE_RUN


def test_two_windows_cannot_create_two_runs(config, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from sopa_companion.engine.runs import RunError
    runs = make_runs(config)
    monkeypatch.setattr(runs, "_execute", lambda run: None)
    sample = runs.samples.list()[0]["id"]
    def create():
        try:
            return runs.create(sample, [50, 50, 150, 150])["id"]
        except RunError as exc:
            return exc.status
    with ThreadPoolExecutor(2) as pool:
        answers = list(pool.map(lambda _: create(), range(2)))
    assert answers.count(409) == 1 and len(runs.list()) == 1
    runs.close("discard")


def test_test_squares_do_not_block_the_one_user_run(config, tmp_path, monkeypatch):
    from sopa_companion.engine.runs import Run
    runs = make_runs(config)
    monkeypatch.setattr(runs, "_execute", lambda run: None)
    runs.runs["square"] = Run(tmp_path / "square", {"id": "square", "state": "done", "created": "",
                                                   "qc_test": {"of": "parent"}})
    sample = runs.samples.list()[0]["id"]
    made = runs.create(sample, [50, 50, 150, 150])
    assert made["session_model"] == 2
    assert runs.pending() == []
    runs.runs.pop("square")
    runs.close("discard")


def test_proseg_runs_in_a_session_of_its_own(config):
    """Proseg shared the engine's process group, so a terminal's hang-up or Ctrl+C (sent to the
    whole foreground group) killed it mid-run, a finish and save included, before the engine could close. It leads a
    session of its own; the parent-death signal still takes it down with the engine."""
    import os
    runs = make_runs(config)
    try:
        run = start(runs)
        wait_for(lambda: run.process is not None and len(run.frames) >= 1)
        pid = run.process.pid
        assert os.getsid(pid) == pid and os.getpgid(pid) != os.getpgid(0)
    finally:
        runs.close("discard")


def test_a_run_never_registers_once_the_engine_closes(config, monkeypatch):
    """The one-run check and closing were checked before a run was registered, outside the registry's lock, so a
    run could register after close() took its list and never be closed. Both are checked again where it registers."""
    import threading
    from sopa_companion.engine.runs import RunError
    runs = make_runs(config)
    monkeypatch.setattr(runs, "_execute", lambda run: None)
    checked, release = threading.Event(), threading.Event()
    original = runs._check_budget
    def budget(*args):
        checked.set()
        assert release.wait(5)
        return original(*args)
    monkeypatch.setattr(runs, "_check_budget", budget)
    answers = []
    def create():
        try:
            answers.append(runs.create(runs.samples.list()[0]["id"], [50, 50, 150, 150])["id"])
        except RunError as exc:
            answers.append(exc.status)
    maker = threading.Thread(target=create)
    maker.start()
    assert checked.wait(5)                     # past every early check, before the registry
    closer = threading.Thread(target=runs.close, args=("discard",))
    closer.start()
    wait_for(lambda: runs.closing)
    release.set()
    maker.join(5)
    closer.join(10)
    assert answers == [503] and not runs.runs and not [p for p in runs.root.iterdir() if not p.name.startswith(".")]


def test_r1_is_checked_under_the_registry_lock_with_the_test_squares_room(config, tmp_path):
    """_room: the run limits as the registration sees them (the caller holds the lock); a test square's run
    gets in beside the user's run, within its own extra places (qc_test's _WithRoom keeps its max_active)."""
    from sopa_companion.engine.runs import ONE_RUN, Run, RunError, Runs
    from sopa_companion.engine.qc_test import _WithRoom
    runs = make_runs(config)
    runs.max_active = 1
    runs.runs["user"] = Run(tmp_path, {"id": "user", "state": "paused"})
    with runs.lock:
        with pytest.raises(RunError) as refused:
            Runs._room(runs, None)
        assert refused.value.status == 409 and str(refused.value) == ONE_RUN
        with pytest.raises(RunError, match="already active"):
            Runs._room(runs, {"qc_test": {"of": "user"}})
        Runs._room(_WithRoom(runs, 2), {"qc_test": {"of": "user"}})          # the square's own places
        runs.closing = True
        with pytest.raises(RunError) as closing:
            Runs._room(runs, {"qc_test": {"of": "user"}})
        assert closing.value.status == 503
    runs.closing = False
    runs.runs.pop("user")


# ---- a test's own runs (engine/qc_test.py) -------------------------------------------------------------------------
TEST = {"of": "parent", "role": "A1", "side": "A", "test": "0123abcd"}      # a test's run's mark (meta qc_test)
# The parquet-only outputs a test's run asks proseg for, written out: a change to them is a change to every test.
PARQUET_ONLY = ["--output-spatialdata", "proseg-output.zarr", "--exclude-spatialdata-transcripts", "--overwrite",
                "--output-transcript-metadata", "transcript-metadata.parquet",
                "--output-transcript-metadata-fmt", "parquet"]


def shared_file(runs, tmp_path):
    """A test's shared input where qc_test leaves it (here: the sample's rows in a box): (path, rows)."""
    path = tmp_path / "qc-tests" / "0123abcd" / "transcripts.parquet"
    return path, runs.samples.crop_transcripts(runs.samples.list()[0]["id"], [50, 50, 150, 150], path)


def run_of_a_test(path, count) -> dict:
    """What qc_test gives Runs.create as `extra` for each run of a test: its mark, the shared input, never paused,
    at full speed."""
    return {"qc_test": dict(TEST), "input": {"path": str(path), "transcripts": count, "ids": "own", "owned": True,
                                            "prepared": {"source": "parent"}}, "paused": False, "speed": "max"}


def registered(runs, name, path, count=10, **meta):
    """A test's run as Runs.create registers it, before its worker starts (a SpatialData sample's, unless `meta` says
    otherwise), on the short schedule."""
    from sopa_companion.engine.runs import Run
    folder = runs.root / name
    folder.mkdir()
    run = Run(folder, {"id": name, "kind": "proseg", "sample": "sample-1", "sample_name": "sample-1",
                       "sample_kind": "zarr", "mode": "detail", "box": [50, 50, 150, 150], "options": SHORT,
                       "proseg_args": runs._proseg_args(runs._values(SHORT)), "state": "preparing", "message": "Queued",
                       "created": name, "live_values": {}, "outputs": [], "zarr_save": None,
                       **run_of_a_test(path, count), **meta})
    run.save()
    with runs.lock:
        runs.runs[name] = run
    return run


def executed(runs, run, timeout: float = 20.0):
    """_execute to the run's end, in a worker as Runs.create starts it. A run still going after `timeout` (held at the
    end of its schedule, say) is aborted and the test fails, instead of waiting for it for ever."""
    import threading
    run.worker = threading.Thread(target=runs._execute, args=(run,), name=f"run-{run.meta['id']}", daemon=True)
    run.worker.start()
    run.worker.join(timeout)
    if run.worker.is_alive():
        runs.control(run.meta["id"], "abort")
        run.worker.join(30)
        raise AssertionError(f"run {run.meta['id']} did not end by itself ({run.meta.get('state')}: "
                             f"{run.meta.get('message')})")


def test_a_test_run_is_marked_by_its_test_alone():
    """is_test_run: a test's run carries {of, role, side, test} (an older test's mark may be anything but a dict); the
    parent's own qc_test {state, progress}, its run detail's status card, is no such mark."""
    from sopa_companion.engine.runs import is_test_run
    assert is_test_run({"qc_test": dict(TEST)}) is True and is_test_run({"qc_test": True}) is True
    for meta in ({}, {"qc_test": None}, {"qc_test": {}}, {"qc_test": {"of": ""}},
                 {"qc_test": {"state": "running", "progress": {"iter": 212, "of": 450}}}):
        assert is_test_run(meta) is False, meta


def test_a_test_runs_command_reads_the_shared_input_and_ends_plainly(config):
    """proseg's command line, golden: a run of the user's is as it was (its own crop, every output, held at the end of
    its schedule; whole-sample ones tiled), with the parent's status card too; a test's run reads the test's one input
    where it is, asks for the per-transcript parquet alone, and ends plainly: no --live-hold, never --live-tiled. Both
    keep proseg's own count, counted when the engine asks into the run's spool folder (this proseg-live has
    --live-counts; the start block only for a user's run that starts from cells alone: tests/test_runs_counts.py)."""
    if not _has_options():
        pytest.skip("option schema not installed")
    from sopa_companion.engine.proseg_options import DEFAULT_EXCLUDED_GENES
    from sopa_companion.engine.runs import TEST_OUTPUTS, Run
    runs = make_runs(config)
    folder = runs.root / "r"
    own, out = str(folder / "input" / "transcripts.parquet"), str(folder / "output")
    rule = ["--excluded-genes", DEFAULT_EXCLUDED_GENES]
    live = ["--live-dir", str(folder / "live"), "--live-control", str(folder / "live" / "control"),
            "--live-counts", "request", "--live-counts-dir", str(runs.spool.folder / "r")]
    every = ["--output-spatialdata", "proseg-output.zarr", "--exclude-spatialdata-transcripts", "--overwrite",
             "--output-cell-polygons", "cell-polygons.geojson.gz", "--output-cell-metadata", "cell-metadata.csv.gz"]
    detail = ["--output-counts", "counts.mtx.gz", "--output-transcript-metadata", "transcript-metadata.csv.gz"]
    assert TEST_OUTPUTS == PARQUET_ONLY
    user = {"id": "r", "mode": "detail", "options": {}, "proseg_args": ["--nthreads", "2"]}
    held = [runs.binary, "--xenium", own, "--nthreads", "2", *rule, "--output-path", out, *every, *detail, *live,
            "--live-hold"]
    assert runs._command(Run(folder, user), []) == held
    assert runs._command(Run(folder, {**user, "qc_test": {"state": "running", "progress": {}}}), []) == held
    assert runs._command(Run(folder, user), ["--no-factorization"]) == [
        runs.binary, "--xenium", own, "--nthreads", "2", *rule, "--no-factorization", "--output-path", out, *every,
        *detail, *live, "--live-hold"]
    assert runs._command(Run(folder, {**user, "mode": "tiled"}), []) == [
        runs.binary, "--xenium", own, "--nthreads", "2", *rule, "--output-path", out, *every, *live, "--live-hold",
        "--live-tiled"]

    shared = "/data/qc-tests/0123abcd/transcripts.parquet"
    test = {"id": "r", "mode": "detail", "options": {"cell-compactness": 0.02, "nthreads": 8},
            "proseg_args": ["--nthreads", "8", "--cell-compactness", "0.02"],
            "qc_test": {**TEST, "role": "B1", "side": "B"},
            "input": {"path": shared, "transcripts": 1234, "ids": "own", "owned": True,
                      "prepared": {"source": "parent"}}}
    setting = ["--nthreads", "8", "--cell-compactness", "0.02", *rule]
    plain = [runs.binary, "--xenium", shared, *setting, "--output-path", out, *PARQUET_ONLY, *live]
    assert runs._command(Run(folder, test), []) == plain
    assert runs._command(Run(folder, {**test, "mode": "tiled"}), []) == plain
    assert runs._command(Run(folder, test), ["--no-factorization"]) == [
        runs.binary, "--xenium", shared, *setting, "--no-factorization", "--output-path", out, *PARQUET_ONLY, *live]


def test_a_test_run_starts_from_the_shared_input_with_no_crop_and_no_save_names(config, tmp_path, monkeypatch):
    """_execute with meta `input`: neither the helper (a SpatialData sample's) nor the crops (a Xenium folder's) run;
    the run has no image and no input bytes of its own, takes the shared input's count and preparation, ends plainly at
    the end of its schedule, and is never offered a save (save_defaults None, _can_save False)."""
    from sopa_companion.engine.runs import Run, Runs
    runs = make_runs(config)
    path, count = shared_file(runs, tmp_path)
    called = []
    for name in ("crop_transcripts", "crop_images"):
        monkeypatch.setattr(runs.samples, name, lambda *args, name=name, **kwargs: called.append(name))
    monkeypatch.setattr(runs, "_prepare_zarr", lambda run: called.append("helper"))
    monkeypatch.setattr(runs, "_save_defaults", lambda run, found=None: called.append("names") or {"shapes_key": "x"})
    for kind in ("zarr", "xenium"):
        run = registered(runs, kind, path, count, sample_kind=kind)
        executed(runs, run)
        assert run.meta["state"] == "done" and "schedule_done" not in run.meta, run.meta["message"]
        assert run.meta["image"] == {"channels": []} and run.meta["input_bytes"] == 0
        assert run.meta["transcripts"] == count > 0 and run.meta["prepared"] == {"source": "parent"}
        assert not (run.dir / "input").exists() and not (run.dir / "image").exists()
        assert run.meta["save_defaults"] is None and Runs._can_save(run) is False
        assert called == []
    done = runs.get("zarr")
    assert Runs._can_save(Run(done.dir, {**done.meta, "qc_test": None})) is True      # its mark alone keeps it unsaved


def test_a_test_run_made_by_create_reads_the_shared_input_and_ends_plainly(config, tmp_path, monkeypatch):
    """Through Runs.create, as qc_test makes them (`extra`) beside nothing else: proseg reads the shared input (its log
    says the command), asks for the per-transcript parquet, and ends at the end of its schedule without holding; the run
    is nothing to save or ask about."""
    if not _has_options():
        pytest.skip("option schema not installed")
    from sopa_companion.engine.runs import ACTIVE, leave_kind
    runs = make_runs(config)
    path, count = shared_file(runs, tmp_path)
    called = []
    for name in ("crop_transcripts", "crop_images"):
        monkeypatch.setattr(runs.samples, name, lambda *args, name=name, **kwargs: called.append(name))
    try:
        run = runs.get(runs.create(runs.samples.list()[0]["id"], [50, 50, 150, 150], SHORT,
                                   extra=run_of_a_test(path, count))["id"])
        wait_for(lambda: run.meta["state"] not in ACTIVE)
        assert run.meta["state"] == "done" and run.meta["message"] == "Finished at iteration 7" and called == []
        [line] = [line for line in (run.dir / "proseg.log").read_text().splitlines() if line.startswith("$ ")]
        assert f" --xenium {path} " in line and line.endswith(f" {' '.join(PARQUET_ONLY)} --live-dir {run.live} "
                                                                f"--live-control {run.live / 'control'} --live-counts "
                                                                f"request --live-counts-dir "
                                                                f"{runs.spool.folder / run.meta['id']}")
        assert run.meta["qc_test"] == TEST and run.meta["speed"] == "max" and run.meta["paused"] is False
        assert run.meta["transcripts"] == count and run.meta["input_bytes"] == 0 and not (run.dir / "input").exists()
        assert leave_kind(run.public())["kind"] == "none" and runs.pending() == []
    finally:
        runs.close("discard")


def test_a_test_run_whose_input_is_missing_fails_saying_so(config, tmp_path):
    """The shared input is gone before the run starts: the run fails with plain words, and proseg never starts."""
    runs = make_runs(config)
    run = registered(runs, "lost", tmp_path / "qc-tests" / "0123abcd" / "transcripts.parquet")
    executed(runs, run)
    assert run.meta["state"] == "failed" and run.meta["message"] == "The test's input is missing"
    assert run.meta.get("pid") is None and not run.live.exists()


def test_delete_later_moves_the_folder_into_the_trash_at_once(config, monkeypatch):
    """delete(later=True), a test's way: the run leaves the registry and its folder leaves the runs folder before the
    call returns (moved whole into .trash/), and is emptied in the background (leftovers.remove_later). A plain delete
    removes the folder before it returns, as before."""
    from sopa_companion.engine import leftovers
    from sopa_companion.engine.runs import Run, RunError
    runs = make_runs(config)
    runs.held = lambda folders: set()                 # nothing holds a file of these runs
    queued, remove_later = [], leftovers.remove_later
    monkeypatch.setattr(leftovers, "remove_later", lambda folder, **kwargs: queued.append((folder, kwargs)))
    for name in ("later", "now"):
        run = Run(runs.root / name, {"id": name, "kind": "proseg", "state": "done", "created": name,
                                     "qc_test": dict(TEST)})
        (run.live / "frames").mkdir(parents=True)
        (run.live / "frames" / "000000.bin.gz").write_bytes(b"x")
        run.save()
        runs.runs[name] = run
    runs.delete("later", later=True)
    [(folder, kwargs)] = queued
    assert not (runs.root / "later").exists() and "later" not in runs.runs
    assert folder.parent == runs.root / ".trash" and folder.name.startswith("later-")
    assert (folder / "run.json").exists() and (folder / "live" / "frames" / "000000.bin.gz").exists()
    remove_later(folder, **kwargs).join(5)
    assert not folder.exists()
    with pytest.raises(RunError) as unknown:
        runs.get("later")
    assert unknown.value.status == 404
    runs.delete("now")
    assert not (runs.root / "now").exists() and len(queued) == 1 and not list((runs.root / ".trash").iterdir())


class _Proseg:
    """A proseg still running, for the engine's side of it: the signals it was sent."""
    pid = 4242

    def __init__(self):
        self.signals = []

    def send_signal(self, number):
        self.signals.append(number)

    def poll(self):
        return None


def test_the_disk_guard_aborts_a_test_run_and_pauses_any_other(config, monkeypatch):
    """Under MIN_FREE_GB free: a user's run sampling pauses with the guard's words, as before; a test's run is aborted
    as Abort does (stop requested, proseg sent SIGTERM), once, with why kept for its stop (`stop_reason`). With room
    enough, neither is touched."""
    import shutil
    import signal
    import types
    from sopa_companion.engine.runs import Run
    runs = make_runs(config)
    free = {"bytes": 3.2e9}
    monkeypatch.setattr(shutil, "disk_usage", lambda path: types.SimpleNamespace(total=1e12, used=1e12 - free["bytes"],
                                                                                 free=free["bytes"]))
    made = {}
    for name, extra in (("square", {"qc_test": dict(TEST)}), ("user", {}), ("roomy", {"qc_test": dict(TEST)})):
        folder = runs.root / name
        folder.mkdir()
        run = made[name] = Run(folder, {"id": name, "kind": "proseg", "state": "running", "paused": False,
                                        "speed": "max", "created": name, "message": "", **extra})
        run.process = _Proseg()
        runs.runs[name] = run
    square, user, roomy = made["square"], made["user"], made["roomy"]
    runs._guard_disk(square)
    runs._guard_disk(user)
    assert square.stop_requested and square.process.signals == [signal.SIGTERM]
    assert square.meta["stop_reason"] == square.meta["message"] == "only 3 GB free on the data disk"
    assert square.meta["state"] == "running" and square.meta["paused"] is False      # stopped once proseg exits
    assert "paused 0" in (square.live / "control").read_text().splitlines()
    assert not user.stop_requested and user.process.signals == [] and "stop_reason" not in user.meta
    assert user.meta["state"] == "paused" and user.meta["paused"] is True
    assert user.meta["message"] == ("Paused automatically: only 3 GB free on the data disk. "
                                    "Free some space, then resume.")
    assert "paused 1" in (user.live / "control").read_text().splitlines()
    runs._guard_disk(square)                          # looked at again before proseg is gone: aborted once
    assert square.process.signals == [signal.SIGTERM]
    free["bytes"] = 40e9
    runs._guard_disk(roomy)
    assert not roomy.stop_requested and roomy.process.signals == [] and roomy.meta["message"] == ""
    for name in made:
        runs.runs.pop(name)


def test_a_test_run_the_disk_guard_stopped_says_why(config, tmp_path, monkeypatch):
    """The guard's words outlast the stop: a test's run it aborted ends "stopped" saying them, not with the snapshot
    count an Abort leaves (qc_test fails the test with them at once)."""
    import shutil
    import types
    from sopa_companion.engine.runs import ACTIVE
    runs = make_runs(config)
    path, count = shared_file(runs, tmp_path)
    monkeypatch.setattr(shutil, "disk_usage", lambda path: types.SimpleNamespace(total=1e12, used=1e12 - 2e9, free=2e9))
    try:
        sample = runs.samples.list()[0]["id"]
        run = runs.get(runs.create(sample, [50, 50, 150, 150], extra=run_of_a_test(path, count))["id"])
        wait_for(lambda: run.meta["state"] not in ACTIVE)
        assert run.meta["state"] == "stopped" and run.meta.get("exit_code") is not None
        assert run.meta["message"] == run.meta["stop_reason"] == "only 2 GB free on the data disk"
    finally:
        runs.close("discard")
