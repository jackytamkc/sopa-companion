"""A Xenium bundle's cell and nucleus boundaries around a run (Samples.xenium_boundaries, the prior of runs made on a
bundle by an older version): a box with no boundary near it gives empty layers (it raised an
IndexError, and the run's prior answered 500 on every request), one file empty is enough, and the rest is unchanged."""
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from sopa_companion.engine import vectors
from sopa_companion.engine.samples import Samples

BASE = "http://127.0.0.1:8799"
HEADERS = {"Authorization": "Bearer " + "t" * 40, "Origin": BASE}
BUNDLE = "output-XETG00000__0000000__EDGES-1__20260101__000000"


def write_bundle(root: Path, boundaries: bool = True) -> Path:
    """The least of a bundle Samples reads here: its transcripts file (that makes it a sample) and, with `boundaries`,
    two cells and their nuclei: a 10 µm square at the origin, and one whose outline reaches (260, 260) while its
    nucleus stays at 225-235."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    folder = root / BUNDLE
    folder.mkdir(parents=True)
    pq.write_table(pa.table({"x_location": pa.array([1.0], pa.float32()), "y_location": pa.array([1.0], pa.float32())}),
                   folder / "transcripts.parquet")
    if boundaries:
        def table(rings):
            ids, xs, ys = [], [], []
            for cell, ring in rings:
                ids += [cell] * len(ring)
                xs += [float(x) for x, _ in ring]
                ys += [float(y) for _, y in ring]
            return pa.table({"cell_id": pa.array(ids), "vertex_x": pa.array(xs, pa.float32()),
                             "vertex_y": pa.array(ys, pa.float32())})
        square = lambda x0, y0, x1, y1: [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
        pq.write_table(table([("aaaa-1", square(0, 0, 10, 10)), ("bbbb-1", square(220, 220, 260, 260))]),
                       folder / "cell_boundaries.parquet")
        pq.write_table(table([("aaaa-1", square(2, 2, 8, 8)), ("bbbb-1", square(225, 225, 235, 235))]),
                       folder / "nucleus_boundaries.parquet")
    return folder


@pytest.fixture
def samples(tmp_path):
    write_bundle(tmp_path / "bundles")
    return Samples({"paths": {"sample_roots": [str(tmp_path / "bundles")], "cache_dir": str(tmp_path / "cache")}})


def counted(found: dict) -> tuple[int, int]:
    return len(found["cells"]), len(found["nuclei"])


def test_a_box_with_no_boundary_near_it_gives_empty_layers(samples):
    assert samples.xenium_boundaries(BUNDLE, [100, 100, 110, 110]) == {"cells": [], "nuclei": []}


def test_one_layer_empty_is_enough_and_the_other_still_comes(samples):
    found = samples.xenium_boundaries(BUNDLE, [262, 262, 270, 270])      # grown by 20 µm: the second cell's corner
    assert counted(found) == (1, 0) and found["cells"][0]["id"] == "bbbb-1"


def test_boundaries_in_the_box_are_unchanged(samples):
    found = samples.xenium_boundaries(BUNDLE, [0, 0, 10, 10])
    assert counted(found) == (1, 1)
    assert found["cells"][0] == {"id": "aaaa-1", "poly": [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]}
    assert found["nuclei"][0]["id"] == "aaaa-1"


def test_a_bundle_without_boundary_files_still_gives_empty_layers(tmp_path):
    write_bundle(tmp_path / "plain", boundaries=False)
    samples = Samples({"paths": {"sample_roots": [str(tmp_path / "plain")], "cache_dir": str(tmp_path / "cache")}})
    assert samples.xenium_boundaries(BUNDLE, [100, 100, 110, 110]) == {"cells": [], "nuclei": []}


def legacy_run(config, run_id: str, box) -> None:
    """A finished run made on the bundle by an older version (a test engine keeps it, as it keeps every run)."""
    folder = Path(config["paths"]["data_dir"]) / "runs" / run_id
    folder.mkdir(parents=True)
    (folder / "run.json").write_text(json.dumps({
        "id": run_id, "kind": "proseg", "sample": BUNDLE, "sample_name": "EDGES-1", "sample_kind": "xenium",
        "box": box, "region": {"kind": "box", "box": box, "bbox": box}, "mode": "detail", "state": "done",
        "created": "2026-01-01T00:00:00", "outputs": [], "zarr_save": None}))


def test_the_prior_of_a_bundle_run_far_from_every_boundary_answers_empty(config, tmp_path):
    """Through the route the page loads (GET /runs/{id}/xenium, the binary form): 200 with empty layers, every time
    (a failed build kept no file, so every request failed again), and the run near the cells has them."""
    from sopa_companion.server import create_app
    write_bundle(tmp_path / "bundles")
    config["paths"]["sample_roots"] = [str(tmp_path / "bundles")]
    config["proseg"]["keep_unsaved"] = True
    legacy_run(config, "legacy-empty", [100, 100, 130, 130])
    legacy_run(config, "legacy-cells", [0, 0, 30, 30])
    application = create_app(config, token="t" * 40)
    with TestClient(application, base_url=BASE, headers=HEADERS) as client:
        for _ in range(2):
            empty = client.get("/api/v1/runs/legacy-empty/xenium", params={"format": "bin", "v": 1})
            assert empty.status_code == 200
            assert vectors.to_json(vectors.decode(empty.content)) == {"cells": [], "nuclei": []}
        near = client.get("/api/v1/runs/legacy-cells/xenium", params={"format": "bin", "v": 1})
        assert near.status_code == 200
        layers = vectors.to_json(vectors.decode(near.content))
        assert [c["id"] for c in layers["cells"]] == ["aaaa-1"] and [n["id"] for n in layers["nuclei"]] == ["aaaa-1"]
        assert np.allclose(layers["cells"][0]["poly"], [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]])
