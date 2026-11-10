"""The Log button's engine side (engine/runlog.py, GET /api/v1/log and /api/v1/runs/{id}/log): the parts it writes from
proseg's snapshots, the Clusters & UMAP results (the step times the page no longer shows), the code hash a deploy
keeps, the tails of the logs, and the log of a real run through the API with the fake proseg and the fake GPU worker:
plain text, complete, never the session's secret, and read without starting or learning anything."""
from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
import sys
import time
import types

import pytest
from fastapi.testclient import TestClient

from conftest import wait_for
from sopa_companion.engine import runlog

BASE = "http://127.0.0.1:8799"
TOKEN = "t" * 40


def auth(extra=None):
    return {"Authorization": "Bearer " + TOKEN, "Origin": BASE, **(extra or {})}


def frame(seq, phase="burnin", t=None, live=None, ms=(100, 20, 5), changed=10, voxel=2.0):
    return {"seq": seq, "iter": seq, "phase": phase, "t": seq * 0.25 if t is None else t, "voxel": voxel,
            "changed": changed, "ms": {"morphology": ms[0], "repositioning": ms[1], "parameters": ms[2]},
            "live": live if live is not None else {"cell-compactness": 0.04, "temperature": 1.0}}


def test_phases_waits_and_knob_changes_come_from_the_snapshots():
    frames = [frame(0, "init", ms=(0, 0, 0)), *[frame(s) for s in range(1, 5)], frame(5, "upscale"),
              *[frame(s, "sampling", voxel=1.0, ms=(300, 40, 10)) for s in range(6, 9)]]
    frames[7]["t"] = frames[6]["t"] + 42.0                        # paused for 42 s before iteration 7
    for f in frames[8:]:
        f["t"] += 42.0
    frames[8]["live"] = {"cell-compactness": 0.019999999552965164, "temperature": 1.0}
    rows = runlog.phases(frames)
    assert [(r["phase"], r["snapshots"], r["first"], r["last"]) for r in rows] == [
        ("init", 1, 0, 0), ("burnin", 4, 1, 4), ("upscale", 1, 5, 5), ("sampling", 3, 6, 8)]
    assert rows[1]["median_ms"] == 125 and rows[3]["slowest_ms"] == 350 and rows[3]["voxel"] == 1.0
    assert rows[0]["median_ms"] is None                           # frame 0 is proseg's starting point, not an iteration
    assert runlog.gaps(frames) == [{"seconds": pytest.approx(42.0), "before": 7, "seq": 7}]
    changes = runlog.knob_changes(frames)
    assert changes == [{"seq": 8, "iter": 8, "t": frames[8]["t"], "flag": "cell-compactness", "before": 0.04,
                        "after": 0.019999999552965164}]
    assert runlog.value(changes[0]["after"]) == "0.02" and runlog.value(None) == "automatic"
    assert runlog.value(True) == "true" and runlog.knob_name("cell-compactness", {"cell-compactness": "Cell compactness"}) \
        == "Cell compactness (cell-compactness)"


def test_the_schedules_own_temperature_is_no_knob_change():
    """proseg's hill-climb cools the temperature by itself and sampling on past the standard run goes back to 1: the log
    names those once per stretch under Schedule, apart from the knobs the user changed (the timeline's rule)."""
    temps = [1.0, 0.977237, 0.954993, 0.933254]
    frames = [frame(0, "recorded", voxel=1.0, live={"cell-compactness": 0.04, "temperature": 1.0}),
              *[frame(s, "hillclimb", voxel=1.0, live={"cell-compactness": 0.04, "temperature": temps[s - 1]})
                for s in range(1, 5)],
              frame(5, "continued", voxel=1.0, live={"cell-compactness": 0.04, "temperature": 1.0}),
              frame(6, "continued", voxel=1.0, live={"cell-compactness": 0.03, "temperature": 1.0})]
    moves = []
    changes = runlog.knob_changes(frames, moves)
    assert [(c["iter"], c["flag"]) for c in changes] == [(6, "cell-compactness")]
    assert moves == [{"phase": "hillclimb", "first": 2, "last": 4, "seq": 4, "before": 1.0, "after": 0.933254},
                     {"phase": "continued", "first": 5, "last": 5, "seq": 5, "before": 0.933254, "after": 1.0}]
    assert runlog.knob_changes(frames) == changes                 # without the list: the same knob changes
    assert runlog.move_text(moves) == ("temperature 1 -> 0.933254 over iterations 2-4 (the hill-climb cools it); "
                                       "0.933254 -> 1 at iteration 5 (sampling on past the standard run)")
    out = runlog.Text()
    runlog.knobs_part(out, frames, {})
    assert out.text().splitlines()[1:] == ["Knobs you changed (as the snapshots show them)",
                                           "  iteration 6 (1.5 s): cell-compactness 0.04 -> 0.03"]


def schedule_frames(temps: dict[int, float], phase_of) -> list[dict]:
    """Snapshots at the iterations of `temps` ({iteration: temperature}), seq = iteration."""
    return [{**frame(i, phase_of(i), voxel=1.0, live={"cell-compactness": 0.04, "temperature": t}), "iter": i}
            for i, t in sorted(temps.items())]


def test_a_temperature_set_live_in_the_hill_climb_or_past_the_end_is_a_knob_change():
    """The rule follows proseg's own schedule (burn-in 200, sampling 200: the hill-climb cools 0.01^(1/200) per
    iteration from iteration 402; past the end it is 1 again), not the phase: a Temperature the user sets at the hold
    before pressing Resume (the help's own advice: "lower Temperature first") or during the hill-climb is listed as their
    change, and only the schedule's own moves are not."""
    sched = {"burnin": 200, "samples": 200}
    cool = lambda i: math.exp(math.log(0.01) / 200 * (i - 401))
    phase = lambda i: "hillclimb" if 401 <= i <= 450 else "continued" if i > 450 else "recorded"
    assert runlog.schedule_temperature({"phase": "hillclimb", "iter": 450}, sched) == pytest.approx(0.323594, rel=1e-5)
    assert runlog.schedule_temperature({"phase": "continued", "iter": 451}, sched) == 1.0
    assert runlog.schedule_temperature({"phase": "hillclimb", "iter": 450}, None) is None
    # Held at the end, Temperature set to 0.1, then Resume.
    frames = schedule_frames({448: cool(448), 449: cool(449), 450: cool(450), 451: 0.1, 452: 0.1}, phase)
    moves = []
    changes = runlog.knob_changes(frames, moves, sched)
    assert [(c["iter"], c["flag"], round(c["before"], 6), c["after"]) for c in changes] == [(451, "temperature", 0.323594, 0.1)]
    assert [(m["phase"], m["first"], m["last"]) for m in moves] == [("hillclimb", 449, 450)]
    # Nothing set: sampling on goes back to 1, the schedule's own move.
    frames = schedule_frames({449: cool(449), 450: cool(450), 451: 1.0}, phase)
    moves = []
    assert runlog.knob_changes(frames, moves, sched) == [] and moves[-1]["phase"] == "continued"
    # Set during the hill-climb: the user's change, and the hill-climb's own cooling before it stays the schedule's.
    frames = schedule_frames({419: cool(419), 420: cool(420), 421: 0.05, 422: 0.05}, phase)
    moves = []
    changes = runlog.knob_changes(frames, moves, sched)
    assert [(c["iter"], c["after"]) for c in changes] == [(421, 0.05)] and [m["last"] for m in moves] == [420]
    # A temperature set at the start (Advanced options: --temperature) holds for every iteration: no knob change.
    start = {**sched, "temperature": 0.5}
    frames = schedule_frames({0: 1.0, 1: 0.5, 2: 0.5}, lambda i: "init" if i == 0 else "burnin")
    moves = []
    assert runlog.knob_changes(frames, moves, start) == []
    assert runlog.move_text(moves, start) == "temperature 1 -> 0.5 at iteration 1 (the temperature set at the start)"
    meta = {"info": {"schedule": {"burnin": 200, "samples": 200, "recorded": 100, "hillclimb": 50}},
            "options": {"temperature": 0.5, "cell-compactness": 0.03}}
    assert runlog.temperature_schedule(meta) == {"burnin": 200, "samples": 200, "temperature": 0.5}
    assert runlog.temperature_schedule({"info": {}}) is None


def test_the_hidden_timing_line_and_the_step_times():
    ms = {"load": 6.2, "counts": 40, "pca": 754.4, "knn": 31, "leiden": 895, "markers": 23, "umap": 106, "write": 43}
    assert runlog.ms_line(ms) == ("load 6 ms · counts 40 ms · pca 754 ms · knn 31 ms · leiden 895 ms · markers 23 ms · "
                                  "umap 106 ms · write 43 ms")   # exactly the line the page showed under the clusters
    assert runlog.ms_line({}) == runlog.ms_line(None) == "no step times"
    results = [(1, {"ms": {"umap": 100, "load": 5}}), (2, {"ms": {"umap": 300, "load": 7, "extra": 1}}), (3, {})]
    assert runlog.step_stats(results) == ["load 6 (7)", "umap 200 (300)", "extra 1 (1)"]


def test_tails_split_progress_bars_and_cut_what_they_read(tmp_path):
    log = tmp_path / "proseg.log"
    log.write_bytes(b"Using 32 threads\n" + b"".join(b"\rsampling %d%%" % i for i in range(100)) + b"\nDone\n\n")
    assert runlog.tail(log, 3) == ["sampling 98%", "sampling 99%", "Done"]
    assert runlog.tail(tmp_path / "missing.log", 3) is None
    log.write_text("".join(f"line {i:04d} for run-a\n" if i % 2 else f"line {i:04d}\n" for i in range(2000)))
    assert runlog.tail(log, 2, max_bytes=40) == ["line 1998", "line 1999 for run-a"]   # a cut first line is dropped
    assert runlog.grep_tail(log, "run-a", 2) == ["line 1997 for run-a", "line 1999 for run-a"]


def test_the_code_hash_is_over_what_a_deploy_ships(tmp_path):
    for rel in ("engine/a.py", "frontend/src/b.js", "var/runs/x.bin", "harness/c.py", "frontend/dev/d.html",
                "native/bin/proseg-live", "engine/__pycache__/a.pyc", "e.membership_debug.txt", "config.local.toml",
                "sopa_companion.egg-info/PKG-INFO", "build/lib/a.py", "native/target-x/y"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(rel)
    first = runlog.code_hash(tmp_path)
    assert first["files"] == 2                                   # only engine/a.py and frontend/src/b.js ship
    (tmp_path / "var" / "runs" / "x.bin").write_text("other run data")
    (tmp_path / "harness" / "c.py").write_text("harness edit")
    assert runlog.code_hash(tmp_path) == first                   # data and the harness never change it
    (tmp_path / "engine" / "a.py").write_text("an edit")
    assert runlog.code_hash(tmp_path)["sha256"] != first["sha256"]
    gate = runlog.ROOT / "harness" / "gate.py"
    if gate.exists():                                            # the dev copy: the same rule as a deploy's
        from sopa_companion.harness import deploy, gate as G
        assert tuple(deploy.SHIP_EXCLUDES) == runlog.SHIP_EXCLUDES
        assert runlog.code_hash()["sha256"] == G.tree_hash(runlog.ROOT, deploy.SHIP_EXCLUDES)["sha256"]
    else:                                                        # production: no harness, the hash still comes out
        assert len(runlog.code_hash()["sha256"]) == 64


def test_an_old_runs_switch_is_read_as_the_engine_reads_it(tmp_path):
    """The log reads the mixing check's switch of a run not looked at in this session from its qc/settings.json without
    loading the check, with the engine's own rule (engine/qc.py stored_switch): a file from before the check was opt-in
    whose switch nobody changed follows [qc] enabled_by_default."""
    from types import SimpleNamespace
    run = SimpleNamespace(dir=tmp_path)
    src = runlog.Sources({}, None, None, qc=SimpleNamespace(s=SimpleNamespace(enabled_by_default=False)))
    assert runlog.stored_check(run, src) == "never switched on (new runs start off)"
    (tmp_path / "qc").mkdir()
    for stored, said in (({"enabled": True, "n": 0}, "off"), ({"enabled": True, "n": 2}, "on"),
                         ({"v": 2, "enabled": True, "n": 0}, "on"), ({"v": 2, "enabled": False, "n": 1}, "off")):
        (tmp_path / "qc" / "settings.json").write_text(json.dumps(stored))
        assert runlog.stored_check(run, src) == f"{said} (not looked at in this session)", stored


class Broken:
    def list(self):
        raise RuntimeError("the runs could not be listed")


@pytest.mark.parametrize("resolution,name", [
    ({"level": 0, "factor": 1, "um_per_px": .2125, "label": "Full"}, "scale0"),     # a run made before level names
    ({"level": 1, "factor": 2, "um_per_px": .425, "label": "1/2"}, "scale1"),
    ({"level": 4, "factor": 16.02, "um_per_px": 3.4, "label": "scale4"}, "scale4"),
    ({"factor": 4, "um_per_px": 2}, "scale2")])                                       # only a factor: its power of two
def test_stain_log_names_the_resolution_by_its_level_and_pixel_size(tmp_path, resolution, name):
    from types import SimpleNamespace
    from sopa_companion.engine.runs import Run
    run = Run(tmp_path, {"id": "stain", "kind": "stain", "state": "done", "progress": {"done": 2, "total": 2},
                         "stain": {"resolution": resolution}})
    src = runlog.Sources({}, SimpleNamespace(get=lambda run_id: run), None)
    out = runlog.Text()
    runlog.summary_part(out, src, "stain")
    row = next(line for line in out.lines if line.startswith("Resolution"))
    assert row.split(maxsplit=1)[1] == f"{name} · {resolution['um_per_px']} µm per pixel"


def test_a_part_that_cannot_be_read_says_so_and_the_log_goes_on(tmp_path):
    class Analysis:
        available = False
        # Why Clusters & UMAP cannot run (AnalysisManager.unavailable_why): "auto" found this Python lacking.
        why_not = "this Python environment lacks scipy, scikit-learn, igraph, umap-learn and numba"

        def log_info(self, run_id=None):
            return {"available": False, "python": "", "gpu": 1, "min_interval": 0, "why_not": self.why_not,
                    "params": {}, "events": [], "worker": {"pid": None, "since": None},
                    "runs": {"switched_off": [], "with_results": 0}}
    config = {"paths": {"state_dir": str(tmp_path / "state"), "data_dir": str(tmp_path / "var")},
              "proseg": {"binary": str(tmp_path / "missing"), "threads": 2}, "server": {"port": 8799}}
    text = runlog.build(runlog.Sources(config, Broken(), Analysis(), secrets=(TOKEN,)))
    assert text.startswith("SoPA companion log\n==================\n")
    # The engine holds one run at a time; only a test engine (keep_unsaved) keeps several, up to its limit.
    assert "for whole-sample runs; one run at a time\n" in text and "active runs" not in text
    config["proseg"].update(keep_unsaved=True, max_active_runs=3)
    assert "for whole-sample runs; at most 3 active runs\n" in runlog.build(runlog.Sources(config, Broken(), Analysis()))
    del config["proseg"]["keep_unsaved"], config["proseg"]["max_active_runs"]
    assert "(engine: this part could not be read: RuntimeError: the runs could not be listed)" in text
    assert "Runs\n----" not in text and "Page telemetry" in text
    assert "server.log (the engine's own output, the last 150 lines)" in text
    # Where Clusters & UMAP cannot run the log never also says they are on; it says why, in the engine's words.
    analysis = Analysis()
    out = runlog.Text()
    runlog.worker_part(out, runlog.Sources(config, Broken(), analysis))
    assert out.lines == ["Analysis worker not available: this Python environment lacks scipy, scikit-learn, igraph, "
                         "umap-learn and numba",
                         "Clusters & UMAP cannot run (this Python environment lacks scipy, scikit-learn, igraph, "
                         "umap-learn and numba)"]
    gone = "/opt/rapids/bin/python"
    for why in ("python under [analysis] is off", f"the python set under [analysis] ({gone}) is not there",
                "its CPU libraries could not be loaded (ImportError: Numba needs NumPy 2.5 or less. Got NumPy 2.6.)"):
        analysis.why_not = why
        out = runlog.Text()
        runlog.worker_part(out, runlog.Sources(config, Broken(), analysis))
        assert out.lines == [f"Analysis worker not available: {why}", f"Clusters & UMAP cannot run ({why})"]


def test_the_worker_and_the_mixing_check_rows_say_where_they_run_and_why_not(tmp_path):
    """The analysis worker's row: on the GPU, on the CPU (its threads and why not the GPU), not running yet, or why
    Clusters & UMAP cannot run here (the packages missing, a lasting reason, python off or not there), with the CPU's
    spacing when min_interval_s is 0. The mixing check's: offered (what its probe saw), being checked, offered though
    its probe gave no answer, or not offered and why (without Clusters & UMAP too)."""
    class Analysis:
        def __init__(self, **info):
            self.info = {"available": True, "python": "/envs/sopa/bin/python", "gpu": 1, "min_interval": 0,
                         "backend": None, "backend_why": None, "threads": None, "why_not": None, "params": {},
                         "events": [], "worker": {"pid": None, "since": None},
                         "runs": {"switched_off": [], "with_results": 0}, **info}
            self.available = self.info["available"]

        def log_info(self, run_id=None):
            return dict(self.info)

        def unavailable_why(self):
            return self.info["why_not"]

    def rows(analysis) -> list[str]:
        out = runlog.Text()
        runlog.worker_part(out, runlog.Sources({}, Broken(), analysis))
        return [" ".join(line.split()) for line in out.lines]
    since = time.time()
    running = {"pid": 4242, "since": since}
    assert rows(Analysis())[0] == "Analysis worker /envs/sopa/bin/python: not running now (it starts with the first run)"
    gpu = rows(Analysis(backend="gpu", worker=running))
    assert gpu[0] == ("Analysis worker /envs/sopa/bin/python on GPU 1: running (pid 4242, ready since "
                      f"{runlog.when(since)})")
    assert gpu[1].startswith("Clusters & UMAP on for every active run; back to back, as fast as the worker goes")
    cpu = rows(Analysis(backend="cpu", worker=running, threads=4, backend_why="cupy is not installed"))
    assert cpu[0] == ("Analysis worker /envs/sopa/bin/python on the CPU with 4 threads (cupy is not installed): running "
                      f"(pid 4242, ready since {runlog.when(since)})")
    assert "every run's cells / 15,000 s apart, 0.5-6 s (the CPU's spacing for min_interval_s = 0)" in cpu[1]
    assert "(min_interval_s = \"auto\")" in rows(Analysis(backend="cpu", min_interval=None))[1]
    for why in ("this Python environment lacks igraph and umap-learn", "its libraries could not be loaded",
                "python under [analysis] is off", "the python set under [analysis] (/gone/bin/python) is not there"):
        assert rows(Analysis(available=False, why_not=why)) == [f"Analysis worker not available: {why}",
                                                               f"Clusters & UMAP cannot run ({why})"]

    class Models:
        def __init__(self, state, why=None, error=None):
            self.cap = {"state": state, "reason": None, "why": why, "probe": {"error": error} if error else None}
            self.available = state != "unavailable"

        def capability(self):
            return dict(self.cap)

        def snapshot(self):
            return {}

    def check_row(models, analysis=None) -> str:
        out = runlog.Text()
        src = runlog.Sources({}, Broken(), analysis or Analysis(), qc=types.SimpleNamespace(prefs={"enabled": False}),
                             qc_models=models)
        runlog.qc_engine_part(out, src)
        return " ".join(" ".join(out.lines).split())          # the row, its wrapped lines joined
    tail = "; on for new runs: no (the last choice; nothing is learned until it is switched on)"
    assert check_row(Models("ready", "torch 2.14.1 sees 1 CUDA GPU(s)")) == \
        "Mixing check offered (torch 2.14.1 sees 1 CUDA GPU(s))" + tail
    assert check_row(Models("checking")) == "Mixing check being checked" + tail
    assert check_row(Models("unknown", "its Python could not be asked (no answer): offered, asked again later",
                            "ImportError: torch")) == \
        "Mixing check offered; its Python could not be asked (ImportError: torch); asked again later" + tail
    assert check_row(Models("unavailable", "this computer has no NVIDIA graphics card")) == \
        "Mixing check not offered here: this computer has no NVIDIA graphics card" + tail
    off = Analysis(available=False, why_not="python under [analysis] is off")
    assert check_row(Models("ready", "torch sees it"), off) == \
        "Mixing check not offered here: it needs Clusters & UMAP (python under [analysis] is off)" + tail


def test_the_proseg_live_row_says_how_it_was_found_or_what_makes_it(tmp_path):
    """The proseg-live row looks at the file now (a build or an install after the start shows), says how the engine
    found it (locate.py: a source folder's build, the Python environment, PATH, the settings), and, not there, what
    makes it, never an index command."""
    program = tmp_path / "proseg-live"
    config = {"proseg": {"binary": str(program)}, "locate": {"proseg": {"path": str(program), "how": "missing",
                                                                         "fix": "install"}}}
    assert runlog.program_row(config) == "missing: install it as the README's Install section says"
    config["locate"]["proseg"]["fix"] = "build"
    assert runlog.program_row(config) == "missing: native/build.sh builds it"
    config["locate"]["proseg"]["fix"] = "setting"
    assert runlog.program_row(config) == f"missing: [proseg] binary names {program}, which is not there"
    assert runlog.program_row({"proseg": {"binary": str(program)}}) == "missing: native/build.sh builds it"
    program.write_bytes(b"\x7fELF")
    # It appeared after the start, where its fix put it: said as that place's.
    for fix, where in (("build", "this source folder's build"), ("install", "this Python environment"),
                       ("setting", "set in the settings")):
        config["locate"]["proseg"]["fix"] = fix
        row = runlog.program_row(config)
        assert row.startswith(f"{program}: 4 B, built ") and " sha256 " in row and row.endswith(f" ({where})"), row
    for how, where in (("source", "this source folder's build"), ("environment", "this Python environment"),
                       ("path", "found on PATH"), ("setting", "set in the settings")):
        config["locate"]["proseg"] = {"path": str(program), "how": how, "fix": ""}
        assert runlog.program_row(config).endswith(f" ({where})"), how


def test_the_log_names_the_clusters_settings_and_what_new_runs_start_with(tmp_path):
    """The Clusters & UMAP settings runs start from and those new runs start with (the analysis worker's part), the
    run's own and whether they are the defaults (its Clusters & UMAP part), and a change among the run's events: rows that hold no
    step times (the harness reads those from the Log)."""
    from test_analysis import FakeRun, FakeRuns
    from sopa_companion.engine.analysis import AnalysisManager
    runs = FakeRuns(FakeRun("r1"))
    config = {"analysis": {"python": sys.executable, "n_pcs": 30},
              "paths": {"state_dir": str(tmp_path / "state"), "data_dir": str(tmp_path / "var")}}
    m = AnalysisManager(config, runs, start=False)
    src = runlog.Sources(config, runs, m)

    def part(write, *args):
        out = runlog.Text()
        write(out, src, *args)
        return " ".join(" ".join(line.split()) for line in out.lines)
    defaults = "resolution 1, n_neighbors 15, min_dist 0.3, min_counts 10, n_top_genes 2000"
    assert f"Settings {defaults} by default (n_pcs 30 from config.toml) New runs start with the defaults" in part(
        runlog.worker_part)
    assert f"Settings {defaults} (the defaults)" in part(runlog.analysis_part, "r1")
    m.configure("r1", params={"resolution": 1.5, "n_neighbors": 30})
    assert "New runs start with resolution 1.5, n_neighbors 30 (the settings last applied)" in part(runlog.worker_part)
    shown = part(runlog.analysis_part, "r1")
    assert ("Settings resolution 1.5, n_neighbors 30, min_dist 0.3, min_counts 10, n_top_genes 2000 (changed from the "
            "defaults: resolution 1, n_neighbors 15); the results below are of these settings") in shown
    assert "Clusters & UMAP settings changed: resolution 1 -> 1.5, n_neighbors 15 -> 30" in part(runlog.run_events_part, "r1")
    steps = re.compile(r"\b(load|counts|pca|knn|leiden|markers|umap|write)\b\s*[:=]?\s*\d+(?:\.\d+)?\s*ms\b", re.I)
    assert not steps.search(shown + part(runlog.worker_part))

def test_long_lines_are_wrapped_never_cut_and_the_command_pastes_back():
    """proseg's command line (~600 characters) was cut at 400, losing its output and --live-hold flags: lines are
    wrapped now, the command with shell continuations, so it pastes back as one command."""
    command = "$ /x/native/bin/proseg-live --xenium /x/runs/r/input/transcripts.parquet " + " ".join(
        f"--flag-{i} {i}" for i in range(60)) + " --live-hold --output-spatialdata proseg-output.zarr"
    out = runlog.Text()
    out.command(command)
    assert all(len(line) <= runlog.WRAP for line in out.lines) and len(out.lines) > 3
    assert all(line.endswith(" \\") for line in out.lines[:-1]) and not out.lines[-1].endswith("\\")
    joined = " ".join(line.strip().removesuffix(" \\").strip() for line in out.lines)
    assert joined == command and "--live-hold" in out.lines[-1] + out.lines[-2]
    out = runlog.Text()
    out.row("Options", "word " * 100)
    assert len(out.lines) == 4 and all(len(line) <= runlog.WRAP for line in out.lines)
    assert all(line.startswith(" " * 16) for line in out.lines[1:])
    assert runlog.wrap("x" * 400, width=100, first="> ", rest="  ") == [
        "> " + "x" * 98, "  " + "x" * 98, "  " + "x" * 98, "  " + "x" * 98, "  " + "x" * 8]   # one long word: broken
    huge = runlog.wrap("y" * 9000)                                   # past LINE_CHARS: cut, then wrapped
    assert len(huge) == math.ceil(runlog.LINE_CHARS / runlog.WRAP) and huge[-1].endswith("…")
    assert runlog.wrap("    File \"w.py\", line 3", first="  ") == ["      File \"w.py\", line 3"]   # indents kept


def test_the_worker_and_trainer_tails_keep_to_this_run():
    """The log of a healthy run showed two tracebacks of other, deleted runs and 40 lines of library warnings from fits
    of other samples: the GPU worker's messages naming another run are left out (and counted), the
    trainer's lines are its own "qc:" lines and errors, their UTC times in local time."""
    rows = ["analysis: warm-up import_cuml: 689 ms",
            "analysis: Traceback (most recent call last):",
            "  File \"/x/worker.py\", line 1094, in umap",
            "FileNotFoundError: [Errno 2] No such file or directory: '/v/runs/run-old/analysis/000477.cells.bin.tmp'",
            "analysis: run run-gone was deleted while frame 3 was being analysed: nothing kept",
            "analysis: Traceback (most recent call last):",
            "MemoryError: out of memory in /v/runs/run-a/analysis/000010.json",
            "analysis: Traceback (most recent call last):",
            "cudaErrorMemoryAllocation: out of memory"]
    kept, other, early = runlog.worker_lines(rows, "run-a")
    assert (other, early) == (2, 0) and kept == [rows[0], rows[5], rows[6], rows[7], rows[8]]
    assert runlog.worker_lines(rows, None) == (rows, 0, 0)
    # Every worker start logs its warm-up: only the newest one is kept (one line per step), the rest counted.
    warm = [f"analysis: warm-up step{i % 5}: {i} ms" for i in range(17)]
    kept, other, early = runlog.worker_lines(warm + rows[4:7], "run-a")
    assert kept == warm[-runlog.WARM_KEPT:] + rows[5:7] and (other, early) == (1, 17 - runlog.WARM_KEPT)
    trainer = ["You are using a CUDA device that has Tensor Cores.",
               "/env/scvi/data/_utils.py:71: UserWarning: Sparse invariant checks are implicitly disabled. Memory errors",
               "💡 Tip: For seamless cloud logging, try installing litlogger",
               "2026-10-18T20:06:18+00:00 qc: learned z465f081f2f7b (a-crop) as 418b68b3",
               "ValueError: the table has no counts"]
    kept = runlog.trainer_lines(trainer)
    assert kept == [f"{runlog.local('2026-10-18T20:06:18+00:00')} qc: learned z465f081f2f7b (a-crop) as 418b68b3",
                    "ValueError: the table has no counts"]


def test_telemetry_reads_as_a_sentence():
    report = {"t": "2026-10-18T21:30:46+00:00", "v": 2, "period_s": 10, "stretched": False,
              "run": {"mode": "tiled", "state": "running", "view": "2d", "level": "overview", "cells": 3439},
              "rates": {"stage": 6.2, "umap": 55.31, "timeline": 6.2, "batches": 7.2, "snapshots": 6.2, "frames": 55.31},
              "long_tasks": {"count": 0, "max_ms": 0}, "input": {"count": 12, "delay_p95_ms": 40, "delay_max_ms": 52},
              "heap_mb": 210.5, "fetch": {"frames": {"n": 40, "bytes": 12_300_000, "wire": 0}, "cells": {"n": 6, "bytes": 0}}}
    assert runlog.telemetry_sentence(report) == (
        "running, 2d, overview: drew the tissue 6.2/s, the UMAP 55.31/s, the timeline 6.2/s; no long tasks; input waited "
        "40 ms at most for 95 % of 12 inputs; memory 210.5 MB; downloaded 12.3 MB in 46 requests")
    late = {"run": {"state": "paused"}, "rates": {}, "long_tasks": {"count": 2, "max_ms": 120}, "stretched": True}
    assert runlog.telemetry_sentence(late) == ("paused: drew nothing; 2 long tasks (the longest 120 ms); this report came "
                                               "late (the page was busy or hidden)")


def test_where_the_run_stands_in_one_line():
    frames = [{"iter": 450}, {"iter": 455}]
    held = {"state": "paused", "schedule_done": {"iter": 450, "seq": 451}}
    assert runlog.standing(held, frames[:1]) == ("proseg's standard run completed at iteration 450; paused there "
                                                 "(Finish & save keeps this result); not saved yet")
    assert runlog.standing({**held, "continued": True}, frames) == (
        "proseg's standard run completed at iteration 450, then sampled on to iteration 455; paused; not saved yet")
    assert runlog.standing({"state": "running"}, [{"iter": 120}]) == "at iteration 120; sampling; not saved yet"
    done = {"state": "done", "outputs": [{"name": "proseg-output.zarr"}],
            "zarr_save": {"state": "saved", "shapes_key": "proseg_boundaries", "table_key": "proseg_table"}}
    assert runlog.standing(done, [{"iter": 33}]) == ("at iteration 33; finished; its results saved; saved into the sample "
                                                     "as proseg_boundaries + proseg_table")
    assert runlog.options_brief({"nthreads": 96}, {}) == "none changed (proseg's defaults)"
    assert runlog.options_brief({"cell-compactness": 0.08}, {"cell-compactness": "Cell compactness (lower = rounder)"}) \
        == "1 changed: Cell compactness 0.08"
    meta = {"info": {"schedule": {"burnin": 200, "samples": 200, "recorded": 100, "hillclimb": 50}}, "options": {}}
    assert runlog.schedule_brief(meta) == ("proseg's default: 450 iterations: burn-in 200, sampling 200 (the last 100 "
                                           "averaged), hill-climb 50")


@pytest.fixture
def app(config, monkeypatch):
    from sopa_companion.server import create_app
    tests = str(Path(__file__).resolve().parent)
    monkeypatch.setenv("PYTHONPATH", tests + os.pathsep + os.environ.get("PYTHONPATH", ""))
    config["analysis"] = {"python": sys.executable, "worker_module": "fake_analysis_worker", "gpu": 0}
    application = create_app(config, token=TOKEN)
    with TestClient(application, base_url=BASE) as client:
        yield application, client


def test_the_log_of_a_run_through_the_api(app, monkeypatch):
    application, client = app
    sample = application.state.samples.list()[0]["id"]
    run = client.post("/api/v1/runs", json={"sample": sample, "box": [50, 50, 150, 150]}, headers=auth()).json()
    runs = application.state.runs
    for action in ({"action": "speed", "speed": "max"}, {"action": "resume"}):
        client.post(f"/api/v1/runs/{run['id']}/control", json=action, headers=auth())
    wait_for(lambda: runs.get(run["id"]).info is not None and len(runs.get(run["id"]).frames) >= 4)
    # Clusters & UMAP are always on: results arrive without anyone switching them on.
    wait_for(lambda: len(client.get(f"/api/v1/runs/{run['id']}/analysis", headers=auth()).json()["seqs"]) >= 2, timeout=60)
    # The secret in the engine's own log (a sign-in URL) never reaches the text.
    log_dir = Path(application.state.analysis.log_path).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "server.log").write_text(f"GET /?token={TOKEN} 303\nSoPA companion: something about {run['id']}\n")
    (log_dir / "telemetry.jsonl").write_text(json.dumps({"t": "2026-10-18T12:00:00.000+00:00", "v": 2, "period_s": 10.0,
                                                         "rates": {"stage": 2.0}}) + "\n")
    # Reading only: the mixing check is never described for the log (that could queue a model fit).
    for owner, name in ((application.state.qc, "describe"), (application.state.qc_models, "ensure"),
                        (application.state.qc_models, "describe")):
        monkeypatch.setattr(owner, name, lambda *a, **k: pytest.fail(f"the log called {name}"))
    response = client.get(f"/api/v1/runs/{run['id']}/log", headers=auth())
    assert response.status_code == 200 and response.headers["content-type"] == "text/plain; charset=utf-8"
    assert response.headers["cache-control"] == "no-store"
    text = response.text
    assert TOKEN not in text and "token=[redacted]" in text
    # Where it stands: still sampling, or (at speed max on a slow computer) already through proseg's standard run.
    assert re.search(r"\nWhere it stands (at iteration \d+|proseg's standard run completed at iteration \d+)", text), text
    for part in ("SoPA companion log", "Summary\n-------\nSample          ",
                 "Mixing check    off", "Engine noted    nothing went wrong", "Engine\n------", "Version         SoPA companion",
                 "Code            sha256 ", f"Run {run['id']}", "Region          a rectangle of 100 x 100 µm",
                 "Phases (from proseg's snapshots", "Knobs you changed (as the snapshots show them)\n  none",
                 "\nClusters & UMAP\n", "  Results       ", "Mixing check", "Engine log lines naming this run",
                 f"SoPA companion: something about {run['id']}", "proseg's own log (proseg.log", "Page telemetry",
                 "drew the tissue 2/s; no long tasks", "analysis.log (the analysis worker's messages about this run"):
        assert part in text, part
    assert "  Switch        on" not in text and "(snapshot " not in text   # no switch to speak of; iterations, not snapshots
    assert "\nProseg's count  proseg's count, averaged over its random background draw; at an iteration counted for a " \
        "pause, the hold, a finish or the end, the\n                count proseg saves there (one random background " \
        "draw) (request mode)\n" in text
    assert "\nCounts spool    /" in text and "MB now; at most 1,024 MB a run, 2,048 MB in all" in " ".join(text.split())
    assert re.search(r"\n {16}\d+ iterations? counted \(init 1", text), text
    head, appendix = text.split(runlog.APPENDIX)                         # the appendix comes last
    assert "Summary" in head and "Page telemetry" in head and "analysis.log" in head
    lines = [line for line in appendix.splitlines() if line.strip().startswith("iteration ")]
    assert len(lines) >= 2 and all("clusters" in line and " cells · " in line for line in lines)   # the fake: no times
    session = client.get("/api/v1/log", headers=auth())
    assert session.status_code == 200 and "Runs\n----" not in session.text and run["id"] in session.text
    assert "this engine session (no run was open)" in session.text and TOKEN not in session.text
    assert client.get("/api/v1/runs/nope/log", headers=auth()).status_code == 404
    assert client.get("/api/v1/log").status_code == 401                        # the page key or token, as always
    client.post(f"/api/v1/runs/{run['id']}/control", json={"action": "abort"}, headers=auth())
    wait_for(lambda: runs.get(run["id"]).meta["state"] == "stopped")


def test_the_log_blanks_what_a_crash_report_blanks(app):
    """The Log and the crash reports share one redactor (engine/crashlog.py redact) and the same secrets: besides the
    session's token, the page's key wherever it stands, the session cookie's value, tokens and codes in JSON and
    anything shaped like the secret. The Log blanked the token alone before. A run's id keeps its own."""
    application, client = app
    key, shaped, run_id = application.state.page_key, "Ab" * 21 + "C", "20261023-140304-slide_region_1_xenium_crop2"
    log_dir = Path(application.state.analysis.log_path).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "server.log").write_text(f"SoPA companion: a line holding {key} as it is\n"
                                        "Cookie: sopa_companion_session=an-older-cookie-1; theme=dark\n"
                                        f"{json.dumps({'token': 'quoted-token-1', 'code': 'launch-code-1'})}\n"
                                        f"SoPA companion: {shaped} and {run_id}\n")
    text = client.get("/api/v1/log", headers=auth()).text
    for hidden in (key, TOKEN, "an-older-cookie-1", "quoted-token-1", "launch-code-1", shaped):
        assert hidden not in text, hidden
    assert "theme=dark" in text and f"[redacted] and {run_id}" in text


def test_the_mixing_checks_numbers_and_go_deeper_are_in_the_log_not_on_screen():
    """The numbers behind the panel's plain words (judged coverage and its parts, transcript
    retention, the raw and the like-for-like change with both thresholds, how the model was learned)
    and Go deeper's state go into the Log button's text; all of it read from what the engine holds."""
    info = {"seq": 40, "iter": 400, "coverage": {"cells": 40070, "judged": 37219, "empty": 1, "under20": 275, "under100": 2575,
                                                "judged_share": 0.9289},
            "retention": {"ratio": 0.92, "start_tx": 9_000_000, "tx": 8_280_000},
            "matched": {"paired": 32384, "used": 32300, "per100_now": 12.11, "per100_start": 13.86, "change": -1.75, "sd": 0.13, "seeds": 3},
            "change": {"raw": -3.15, "matched": -1.75, "threshold": 0.3, "model_threshold": 0.6, "meaningful": True,
                       "direction": "fewer", "enough": True},
            "prev_change": {"matched": 0.12, "threshold": 0.3, "meaningful": False, "direction": "same"}}
    rows = dict(runlog.mixing_detail(info, []))
    assert rows["Newest result"] == "iteration 400"
    assert rows["Coverage"].startswith("37,219 of 40,070 cells judged") and "92.9 %" in rows["Coverage"]
    assert "1 empty, 275 with 1-19, 2,575 with 20-99" in rows["Coverage"]
    assert rows["Retention"].startswith("the cells hold 92.0 %")
    assert "12.11 in 100 now against 13.86 at the start over 32,384 cells" in rows["Like for like"]
    assert rows["Change"].startswith("raw -3.15, like for like -1.75 points") and "fewer look mixed" in rows["Change"]
    assert "counts from 0.3 (sampler) and 0.6 (another learning" in rows["Change"]
    assert "Previous run" not in rows
    assert "Count" not in rows                                                  # a result that names no count
    # Proseg's count (engine/qc.py log_info: basis, count, constants): the like for like is deterministic, and the
    # thresholds say whether they were measured on proseg's count.
    constants = {"mode": "region", "id": "outline-2026-10", "measured": False}
    counted = dict(runlog.mixing_detail({**info, "matched": {**info["matched"], "sd": None, "seeds": 0},
                                         "basis": "proseg", "count": "averaged", "constants": constants}, []))
    assert counted["Count"] == ("proseg's count, averaged over its random background draw; thresholds outline-2026-10 "
                                "(region runs); not yet re-measured on proseg's count")
    # A frame read on the count proseg saves (counted for a pause, the hold, a finish or the end) says so.
    saved = dict(runlog.mixing_detail({**info, "basis": "proseg", "count": "sampled", "constants": constants}, []))
    assert saved["Count"] == ("the count proseg saves at this iteration (one random background draw); thresholds "
                              "outline-2026-10 (region runs); not yet re-measured on proseg's count")
    assert counted["Like for like"].endswith("cells judged in both, both scaled to the shallower of their two depths "
                                             "(no draws)")
    # A run whose start has nuclei (a Xenium bundle's own, start_kind "paired"; or nuclei only) has no start line:
    # nothing is compared with a start, and no reason is given for a verdict it cannot have (thousands judged here).
    paired = {"seq": 40, "iter": 400, "start_kind": "paired", "coverage": info["coverage"], "matched": None, "retention": None,
              "change": {"raw": None, "matched": None, "threshold": 1.0, "model_threshold": 0.6, "meaningful": False,
                         "direction": None, "enough": False}}
    for kind in ("paired", "nuclei"):
        rows = dict(runlog.mixing_detail({**paired, "start_kind": kind}, []))
        assert rows["Change"] == ("none: this run started from nuclei (alone or with their cells), so it has no start "
                                  "line to compare with"), kind
        assert "Like for like" not in rows and "Retention" not in rows
        assert not any("too few cells" in v or "since the start" in v for v in rows.values()), rows
    assert dict(runlog.mixing_detail({**info, "start_kind": "cells"}, []))["Change"].startswith("raw -3.15")
    # A run from cells alone whose change is not worked out yet (its start line not scored): nothing compared, never
    # "too few cells" with thousands judged; a change worked out on too few cells says so.
    pending = dict(runlog.mixing_detail({**paired, "start_kind": "cells"}, []))["Change"]
    assert pending.endswith("no verdict (nothing compared with the start yet)") and "too few" not in pending, pending
    few = {**paired["change"], "raw": -1.0, "matched": -1.2}
    assert dict(runlog.mixing_detail({**paired, "start_kind": "cells", "change": few}, []))["Change"].endswith(
        "no verdict (too few cells to judge)")
    assert "Like for like" not in dict(runlog.mixing_detail({"seq": 1, "iter": 5, "matched": None}, []))
    assert dict(runlog.mixing_detail({"seq": 1, "iter": 5, "matched": {"error": "the start line was counted another way"}}, []))[
        "Like for like"] == "not worked out: the start line was counted another way"
    p = {"table": "xenium_table", "other_table": "table", "priors": "pooled", "epochs_run": 429, "epochs_notebook": 50,
         "cells": 3455, "notebook_equivalent": False, "differences": ["more_epochs_small_table"]}
    # The log names none of the user's files either.
    assert runlog.provenance_text(p) == ("learned from xenium_table with table, pooled starting estimates, 429 passes over "
                                         "the cells (the reference: 50), 3,455 cells; not the reference settings "
                                         "(more_epochs_small_table)")
    assert runlog.provenance_text({**p, "notebook_equivalent": True}).endswith("; the reference settings")
    assert "notebook" not in runlog.provenance_text(p) + runlog.provenance_text({**p, "notebook_equivalent": True})
    assert runlog.provenance_text(None) == "not recorded"
    assert runlog.details_text({"deep": True}, None) == "the model holds its details (deep.npz)"
    assert runlog.details_text({"deep": False}, {"state": "learning", "progress": {"step": 3, "of": 50}}) == \
        "adding the details: learning, step 3 of 50"
    assert "could not be added (detail_mismatch)" in runlog.details_text({}, {"state": "failed", "reason": "detail_mismatch"})
    assert runlog.details_text({}, {"state": "none"}).startswith("no details kept")

    class Analysis:
        def __init__(self, deeper):
            self.deeper = deeper

        def log_info(self, run_id=None):
            return {"run": {"deeper": self.deeper}}

    class Deeper:
        def log_info(self, run_id=None):
            return {"kept": 3, "writes_queued": 0, "run": {"kept": 2, "pending": None, "on_disk": 5, "watchers": 1}}

    out = runlog.Text()
    runlog.deeper_run_part(out, runlog.Sources({}, None, Analysis({"follow": 41, "requested": ("start", 7), "failures": 2}),
                                               deeper=Deeper()), "r1")
    text = out.text()
    assert "Go deeper" in text and "cell 41 (label - 1) with every live pass" in text and "frame start, cell 7" in text
    assert "2 in a row" in text and "2 kept in memory, 5 on disk; 1 page(s) watching" in text
    quiet = runlog.Text()
    runlog.deeper_run_part(quiet, runlog.Sources({}, None, Analysis({"follow": None, "requested": None, "failures": 0})), "r1")
    assert "not used for this run in this session" in quiet.text()


def test_the_log_names_proseg_s_count_and_what_keeping_it_took():
    """The run's "Proseg's count" row: the basis every number reads (averaged over its random background draw), how
    proseg-live kept it, the iterations it counted by why, the bytes in the run's folder and through the spool (and the
    most the run's spool folder held), how often one came on proseg's clock, and proseg's time to count and write one;
    or why there is none."""
    def counted(seq, reason, file, size, extract, write, t):
        return {"seq": seq, "iter": seq * 10, "t": t,
                "counts": {"reason": reason, "file": file, "bytes": size, "ms": {"extract": extract, "write": write}}}
    frames = [counted(0, "init", "frames/000000.counts.zst", 2_000_000, 30.0, 5.0, 0.0),
              {"seq": 1, "iter": 10, "t": 0.5},
              counted(2, "request", "/dev/shm/s/r/000002.counts.zst", 1_000_000, 10.0, 2.0, 1.0),
              counted(3, "request", "/dev/shm/s/r/000003.counts.zst", 1_000_000, 20.0, 4.0, 2.0)]
    meta = {"info": {"counts": {"mode": "request", "start": True}}, "proseg_counts": {"reason": None}}
    assert runlog.counts_text(meta, frames, 2.5, 1024).splitlines() == [
        "proseg's count, averaged over its random background draw; at an iteration counted for a pause, the hold, a "
        "finish or the end, the count proseg saves there (one random background draw) (request mode, with its start "
        "block)",
        "3 iterations counted (init 1, request 2), the newest at iteration 30",
        "2.0 MB kept in the run's folder, 2.0 MB written to the spool (it held 2.5 MB at most, of 1,024 MB)",
        "one every 1.00 s (median, proseg's clock)",
        "counting took 20 ms (slowest 30) on proseg's main thread, writing 4 ms"]
    assert runlog.counts_text({"info": {"counts": {"mode": "every", "start": False}}}, frames[1:2]) == \
        "proseg's count, averaged over its random background draw; at an iteration counted for a pause, the hold, a " \
        "finish or the end, the count proseg saves there (one random background draw) (every mode)\nno iteration " \
        "counted yet"
    assert runlog.counts_text({"proseg_counts": {"reason": "no_counts"}}, frames) == \
        "not kept: this run was made before proseg's own count was kept"
    assert runlog.counts_text({"proseg_counts": {"reason": "old_proseg"}}, []).endswith("update proseg-live")
    out = runlog.Text()
    runlog.counts_part(out, runlog.Sources({}, types.SimpleNamespace(), None), "r", meta, frames)
    assert out.lines[0].startswith("Proseg's count  proseg's count, averaged") and "it held" not in out.text()
    # V2-1: a value past the file's 16 bits is kept clipped (the line's `saturated` says how many): never silent here.
    frames[2]["counts"]["saturated"], frames[3]["counts"]["saturated"] = 2, 5
    assert runlog.counts_text(meta, frames).splitlines()[2] == (
        "2 of them clipped a cell's count of a gene (5 values at most), the first at iteration 20: proseg-live's file "
        "holds 2,047.97 averaged and 65,535 transcripts at most, so those cells read low there; the saved table keeps "
        "the true count")
    frames[2]["counts"]["saturated"] = frames[3]["counts"]["saturated"] = 0
    assert "clipped" not in runlog.counts_text(meta, frames)


def test_the_prepared_line_says_how_proseg_was_seeded():
    # A Cellpose-only prior (no nuclei) reached proseg with 0 nuclear transcripts, so it started with 0
    # cells and panicked; now every transcript in a prior cell seeds it, and the log says so instead of "N nuclear"
    cells = runlog.prepared_text({"transcripts": 41627906, "assigned": 38270417, "nuclear": 38270417,
                                  "seeded_from": "cells", "prior_cells": 43724, "seconds": 23.6})
    assert "no nuclei chosen, so every one seeds its cell" in cells and "nuclear" not in cells.replace("nuclei", "")
    nuclei = runlog.prepared_text({"transcripts": 1000, "assigned": 800, "nuclear": 300, "seeded_from": "nuclei",
                                   "prior_cells": 12, "seconds": 1.0})
    assert "300 nuclear" in nuclei and "seeds" not in nuclei
    older = runlog.prepared_text({"transcripts": 1000, "assigned": 800, "nuclear": 300, "prior_cells": 12})
    assert "300 nuclear" in older                      # runs prepared before the field existed
