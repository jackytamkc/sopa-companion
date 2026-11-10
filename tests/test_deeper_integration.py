"""Go deeper end to end across the parts that ship together: the analysis scheduler and DeeperManager
(engine/analysis.py, engine/deeper.py) driving the GPU worker's own
code on the CPU (analysis/worker.py Worker(xp=numpy): op "deeper") with a model whose details the fake trainer writes
(tests/fake_qc_trainer.py write_model: encoder.npz and deep.npz with its self-test). It checks the protocol (requests,
replies, the start line on the prior's labels) and the first rule: the window's byte is the card's, the
frame's stored byte."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import textwrap

import pytest

from conftest import wait_for
from sopa_companion.engine.analysis import AnalysisManager
from sopa_companion.engine.deeper import DeeperManager, _read_codes
from test_analysis import JUDGE, NCELLS, X0, Y0, NX, NY, VOXEL, FakeRuns, Provider, disk_run, write_run

MODEL = "c0ffee1234abcdef"

WORKER = '''
import json, sys
import numpy as np
from sopa_companion.analysis.worker import Worker
worker = Worker(xp=np)
print(json.dumps({"ok": True, "ready": True}), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    if request.get("op") == "forget":
        continue
    if request.get("op") != "qc_start":                  # everything but clusters and UMAP, by the worker's own code
        request = {**request, "ops": [op for op in request.get("ops") or [] if op != "umap"]}
    print(json.dumps(worker.handle(request)), flush=True)
'''


class Check:
    """QcManager's side for one run whose check is on with a ready model that holds its details."""
    version = 0

    def __init__(self, folder: Path):
        self.folder = folder

    def describe(self, run_id):
        return {"enabled": True, "state": "ready", "sample_state": "ready", "start_source": "xenium",
                "model": {"id": MODEL, "deep": True}}

    def qc_model(self, run):
        return {"id": MODEL, "path": str(self.folder)}


class Samples:
    version = 0

    def describe(self, sample_id):
        return {}

    def snapshot(self):
        return {}


def test_a_cell_explained_by_the_workers_own_code_carries_the_cards_byte(tmp_path, monkeypatch):
    from sopa_companion.tests import fake_qc_trainer as F
    workers = tmp_path / "workers"
    workers.mkdir()
    (workers / "deeper_numpy_worker.py").write_text(textwrap.dedent(WORKER))
    monkeypatch.setenv("PYTHONPATH", str(workers) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    run = disk_run(tmp_path / "run", frames=4)
    run.meta["state"] = "done"
    # Its cells started without nuclei: the one start with a start line (engine/qc_rules.py start_of "cells").
    run.meta.update(sample="s1", sample_kind="zarr",
                    prior={"cells": {"kind": "shapes", "key": "cells"}, "nuclei": None})
    (run.live / "cells.txt").write_text("".join(f"c{i}\n" for i in range(NCELLS)))   # the learning cells' ids
    model = tmp_path / "model"
    F.write_model(model, {"zarr": "x", "table": "xenium_table", "recipe": {}, "model_id": MODEL}, 3)
    assert (model / "deep.npz").is_file()
    config = {"analysis": {"python": sys.executable, "worker_module": "deeper_numpy_worker"},
              "paths": {"state_dir": str(tmp_path / "state")}, "qc": {"fit_min_counts": 5}}
    m = AnalysisManager(config, FakeRuns(run), start=False)
    m.attach_qc(Provider(model=(MODEL, str(model))))
    deeper = DeeperManager({}, FakeRuns(run), Check(model), Samples(), m)
    m.thread.start()
    try:
        m.configure_qc("r1", enabled=True)
        wait_for(lambda: m.describe_qc("r1")["start"] == "done" and len(m.describe_qc("r1")["seqs"]) == 4, timeout=60)
        assert deeper.request("r1", {"seq": 3, "cell": 0, "follow": True})["state"] == "working"
        # Frame 3, the run's last, counted for its end with f: read on the count proseg saves.
        wait_for(lambda: ("r1", 3, 0, MODEL, 1, "sampled") in deeper.cache, timeout=60)
        assert deeper.request("r1", {"seq": "start", "cell": 1})["state"] == "working"
        wait_for(lambda: ("r1", "start", 1, MODEL, 1, "averaged") in deeper.cache, timeout=60)
    finally:
        m.close()
        m.thread.join(5)
    folder = m.qc_dir(run, MODEL)                             # the scheduler's folder of the check's files
    for seq, cell, stored in ((3, 0, "000003.qc.bin"), ("start", 1, "start.bin")):
        result = deeper.cached("r1", seq, cell)
        assert result["v"] == 1 and result["seq"] == seq and result["state"] in ("ready", "too_few", "empty")
        byte = _read_codes(folder / stored)[cell]
        if result["state"] == "ready":
            assert result["code"] == byte, "the window's number is the card's"
            assert result["transcripts"] >= JUDGE and sum(result["parts"]) == pytest.approx(1.0, abs=1e-3)
    assert m.deeper_follow == {"r1": 0} and not m.deeper_requested


def test_a_frame_proseg_never_counted_or_whose_file_went_is_answered_and_found_as_the_counted_frame_it_snaps_to(
        tmp_path, monkeypatch):
    """Go deeper asked at a frame proseg never counted (request mode counts only the frames its tokens ask for) or whose
    counts file the spool pruned: the real scheduler works it out at the counted frame it snaps to, and the reply, the
    GET and a second ask all name and find that frame's answer (until then a window paused there waited for ever: its
    reply named the frame asked, whose answer never came, and every ask was "working")."""
    from sopa_companion.engine.runs import Run
    from sopa_companion.tests import fake_qc_trainer as F
    workers = tmp_path / "workers"
    workers.mkdir()
    (workers / "deeper_numpy_worker.py").write_text(textwrap.dedent(WORKER))
    monkeypatch.setenv("PYTHONPATH", str(workers) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    folder = write_run(tmp_path / "run", frames=6)
    lines = [json.loads(line) for line in (folder / "live" / "counts.jsonl").read_text().splitlines() if line.strip()]
    (folder / "live" / "counts.jsonl").write_text("".join(json.dumps(line) + "\n" for line in lines if line["seq"] != 3))
    run = Run(folder, {"id": "r1", "state": "done", "box": [X0, Y0, X0 + NX * VOXEL, Y0 + NY * VOXEL], "mode": "detail",
                       "sample": "s1", "sample_kind": "zarr",
                       "prior": {"cells": {"kind": "shapes", "key": "cells"}, "nuclei": None}})
    run.read_new()
    (run.live / "cells.txt").write_text("".join(f"c{i}\n" for i in range(NCELLS)))
    assert [s for s, frame in enumerate(run.frames) if frame.get("counts_file")] == [0, 1, 2, 4, 5]
    model = tmp_path / "model"
    F.write_model(model, {"zarr": "x", "table": "xenium_table", "recipe": {}, "model_id": MODEL}, 3)
    config = {"analysis": {"python": sys.executable, "worker_module": "deeper_numpy_worker"},
              "paths": {"state_dir": str(tmp_path / "state")}, "qc": {"fit_min_counts": 5}}
    m = AnalysisManager(config, FakeRuns(run), start=False)
    m.attach_qc(Provider(model=(MODEL, str(model))))
    deeper = DeeperManager({}, FakeRuns(run), Check(model), Samples(), m)
    m.thread.start()
    try:
        m.configure_qc("r1", enabled=True)
        wait_for(lambda: m.describe_qc("r1")["start"] == "done" and len(m.describe_qc("r1")["seqs"]) == 5, timeout=60)
        first = deeper.request("r1", {"seq": 3, "cell": 0})
        assert (first["state"], first["seq"]) == ("working", 2), "never counted: its answer is frame 2's, and says so"
        wait_for(lambda: ("r1", 2, 0, MODEL, 1, "averaged") in deeper.cache, timeout=60)
        assert deeper.cached("r1", 3, 0) == deeper.cached("r1", 2, 0), "the GET of frame 3 finds it"
        again = deeper.request("r1", {"seq": 3, "cell": 0})
        assert (again["state"], again["seq"], again["result"]) == ("ready", 2, deeper.cached("r1", 2, 0))
        # Frame 4's file pruned from the spool: answered as frame 2, whose answer for this cell is kept.
        (run.live / run.frames[4]["counts_file"]).unlink()
        pruned = deeper.request("r1", {"seq": 4, "cell": 0})
        assert (pruned["state"], pruned["seq"]) == ("ready", 2)
        other = deeper.request("r1", {"seq": 4, "cell": 1})
        assert (other["state"], other["seq"]) == ("working", 2)
        wait_for(lambda: ("r1", 2, 1, MODEL, 1, "averaged") in deeper.cache, timeout=60)
        assert deeper.request("r1", {"seq": 4, "cell": 1})["state"] == "ready"
    finally:
        m.close()
        m.thread.join(5)
