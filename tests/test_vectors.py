"""Prior vectors (engine/vectors.py, GET /runs/{id}/xenium?format=bin&v=1): the binary carries exactly what the JSON
carries (polygons, order, float32 coordinates, ids, source), its layout is the one docs/ARCHITECTURE.md describes, it is
built once per run (in the background for whole-sample runs) and served as the zstd file it is cached as. Outlines are
injected into ZarrSamples' cache, so none of this needs the sopa environment (test_zarr.py checks a real sample)."""
from __future__ import annotations

import json
import struct
import threading
import time

import numpy as np
import pytest
import zstandard
from fastapi.testclient import TestClient

from conftest import HERE, wait_for
from sopa_companion.engine import vectors

FIXTURES = HERE.parent / "frontend" / "test" / "fixtures"     # shared with frontend/test/vectors-decode.test.mjs
BASE = "http://127.0.0.1:8799"
SID, CELLS, NUCLEI = "zsynthetic0001", "cell_boundaries", "nucleus_boundaries"
SHAPES = {"cells": {"kind": "shapes", "key": CELLS}, "nuclei": {"kind": "shapes", "key": NUCLEI}}


def auth(extra=None):
    return {"Authorization": "Bearer " + "t" * 40, "Origin": BASE, **(extra or {})}


def outlines(seed: int, count: int) -> dict:
    """What ZarrSamples._outlines holds for a shapes element (as zarr_bridge/helper.py writes it): closed float32
    rings with full-precision decimals, every fifth shape a multipolygon of two parts sharing its owner."""
    rng = np.random.default_rng(seed)
    rings, owner = [], []
    for k in range(count):
        centre = rng.uniform(0, 1000, 2)
        for part in range(2 if k % 5 == 0 else 1):
            angles = np.sort(rng.uniform(0, 2 * np.pi, int(rng.integers(3, 30))))
            ring = centre + part * 9.0 + rng.uniform(2, 8) * np.c_[np.cos(angles), np.sin(angles)]
            rings.append(np.vstack([ring, ring[:1]]).astype(np.float32))
            owner.append(k)
    offsets = np.zeros(len(rings) + 1, np.uint32)
    offsets[1:] = np.cumsum([len(r) for r in rings])
    return {"ids": [f"cell-{k}" for k in range(count)],
            "bounds": np.array([[*r.min(0), *r.max(0)] for r in rings], np.float32),
            "offsets": offsets, "owner": np.array(owner, np.uint32), "coords": np.vstack(rings)}


@pytest.fixture
def engine(config):
    """Runs over a ZarrSamples whose sample SID is known and whose two shapes elements are already exported."""
    from sopa_companion.engine.runs import Runs
    from sopa_companion.engine.samples import Samples
    from sopa_companion.engine.zarr_samples import ZarrSamples
    zarr = ZarrSamples(config)
    inject(zarr)
    return Runs(config, Samples(config), zarr)


def inject(zarr):
    zarr.recent.insert(0, {"id": SID, "name": "synthetic", "path": "/nonexistent/synthetic.zarr", "kind": "zarr"})
    zarr.outlines[(SID, CELLS)], zarr.outlines[(SID, NUCLEI)] = outlines(1, 300), outlines(2, 280)


def add_run(runs, run_id: str, box, mode: str = "tiled", prior=None):
    from sopa_companion.engine.runs import Run
    (runs.root / run_id).mkdir()
    run = Run(runs.root / run_id, {"id": run_id, "sample": SID, "sample_name": "synthetic", "box": box, "mode": mode,
                                   "prior": SHAPES if prior is None else prior, "sample_kind": "zarr",
                                   "sample_path": "/nonexistent/synthetic.zarr", "state": "done", "created": "2026-10-17"})
    run.save()
    runs.runs[run_id] = run
    return run


def unpacked(path) -> bytes:
    return zstandard.ZstdDecompressor().decompressobj().decompress(path.read_bytes())


def test_the_binary_carries_exactly_what_the_json_carries(engine):
    cases = {"whole": ([0, 0, 1000, 1000], "tiled", None), "part": ([250, 300, 600, 520], "tiled", None),
             "detail": ([100, 100, 400, 400], "detail", None), "nowhere": ([5000, 5000, 5100, 5100], "tiled", None),
             "cells-only": ([0, 0, 1000, 1000], "tiled", {"cells": SHAPES["cells"], "nuclei": {"kind": "column", "column": "n"}})}
    for run_id, (box, mode, prior) in cases.items():
        add_run(engine, run_id, box, mode, prior)
        old = json.loads(engine.xenium_boundaries(run_id).read_text())       # the JSON endpoint's file
        new = vectors.decode(unpacked(engine.prior_vectors(run_id)))
        assert vectors.to_json(new) == old, run_id                            # polygons, order, coordinates, ids, source
        for part in vectors.LAYERS:
            layer = new[part]
            if layer is None:                                                 # no shapes prior for it: the JSON's []
                assert old[part] == [] and part not in old["source"]
                continue
            assert layer["xy"].dtype == np.float32 and np.array_equal(layer["xy"].astype(float), [v for p in old[part] for v in p["poly"]] or np.empty((0, 2)))
            boxes = [[min(x for x, _ in p["poly"]), min(y for _, y in p["poly"]), max(x for x, _ in p["poly"]), max(y for _, y in p["poly"])] for p in old[part]]
            assert np.array_equal(layer["bbox"].astype(float), boxes or np.empty((0, 4))), (run_id, part)
    assert len(json.loads(engine.xenium_boundaries("part").read_text())["cells"]) < 300 < len(vectors.decode(unpacked(engine.prior_vectors("whole")))["cells"]["ids"])
    assert vectors.decode(unpacked(engine.prior_vectors("cells-only")))["source"] == {"cells": CELLS}


def test_select_is_the_shapes_selection_with_its_limit(engine):
    data = engine.zarr._outlines(SID, CELLS)
    for box, limit in (([0, 0, 1000, 1000], 400_000), ([200, 200, 700, 700], 25), ([0, 0, 1000, 1000], 0)):
        layer = vectors.select(data, box, limit)
        expected = engine.zarr.shapes(SID, CELLS, box, limit)["polygons"]
        assert vectors.to_json({"cells": layer, "nuclei": None})["cells"] == expected


def test_layout_is_the_documented_one():
    value = json.loads((FIXTURES / "prior-vectors.json").read_text())
    data = (FIXTURES / "prior-vectors.bin").read_bytes()
    # The Python encoder makes exactly the bytes the node test decodes (frontend/test/vectors-decode.test.mjs).
    assert vectors.encode({part: vectors.from_polygons(value[part]) for part in vectors.LAYERS}, {"source": value["source"]}) == data
    magic, version, size, count = struct.unpack_from("<4sIII", data)
    records = [struct.unpack_from("<IIII", data, 16 + 16 * k) for k in range(count)]
    meta_bytes, = struct.unpack_from("<I", data, 16 + 16 * count)
    assert (magic, version, count) == (b"PLPV", 1, 2) and size % 4 == 0
    assert json.loads(data[20 + 16 * count:20 + 16 * count + meta_bytes]) == {"source": value["source"]}
    assert set(data[20 + 16 * count + meta_bytes:size]) <= {0}
    at = size
    for (flags, n, vertices, names), part, numeric in zip(records, vectors.LAYERS, (False, True)):
        polys = value[part]       # cells: string ids (and one number) in a table; nuclei: integer ids, carried as such
        assert (flags, n, vertices) == (3 if numeric else 1, len(polys), sum(len(p["poly"]) for p in polys))
        offsets = struct.unpack_from(f"<{n + 1}I", data, at)
        xy = struct.unpack_from(f"<{2 * vertices}f", data, at + 4 * (n + 1))
        bbox = struct.unpack_from(f"<{4 * n}f", data, at + 4 * (n + 1) + 8 * vertices)
        ids = struct.unpack_from(f"<{n}I", data, at + 4 * (n + 1) + 8 * vertices + 16 * n)
        assert offsets == tuple(np.cumsum([0] + [len(p["poly"]) for p in polys]))
        assert xy == tuple(np.float32(v) for p in polys for vertex in p["poly"] for v in vertex)   # float32 of the JSON's numbers
        if numeric:
            assert names == 0 and list(ids) == [p["id"] for p in polys]
        else:
            table = json.loads(data[at + 4 * (n + 1) + 8 * vertices + 20 * n:][:names])
            assert [table[i] for i in ids] == [p["id"] for p in polys] and table == ["aaaabbbb-1", "aaaacccc-1", 17]
        for k, p in enumerate(polys):
            xs, ys = [np.float32(x) for x, _ in p["poly"]], [np.float32(y) for _, y in p["poly"]]
            assert bbox[4 * k:4 * k + 4] == ((min(xs), min(ys), max(xs), max(ys)) if xs else (np.inf, np.inf, -np.inf, -np.inf))
        at += 4 * (n + 1) + 8 * vertices + 20 * n + names + -names % 4
    assert at == len(data)


def test_absent_and_empty_layers_and_malformed_files():
    empty = vectors.decode(vectors.encode({"cells": vectors.from_polygons([]), "nuclei": None}))
    assert empty["nuclei"] is None and len(empty["cells"]["ids"]) == 0 and empty["cells"]["offsets"].tolist() == [0]
    assert vectors.to_json(empty) == {"cells": [], "nuclei": []}
    data = (FIXTURES / "prior-vectors.bin").read_bytes()
    for bad in [data[:k] for k in range(len(data))] + [data + b"\0\0\0\0", b"XXXX" + data[4:],
                                                      data[:4] + struct.pack("<I", 2) + data[8:]]:
        with pytest.raises(ValueError):
            vectors.decode(bad)
    size, = struct.unpack_from("<I", data, 8)
    broken = bytearray(data)
    struct.pack_into("<I", broken, size + 4, 99)      # cells offsets[1] beyond offsets[2]: not monotonic
    with pytest.raises(ValueError, match="offsets"):
        vectors.decode(bytes(broken))


def test_one_build_however_many_ask_and_whole_sample_runs_prebuild(engine, monkeypatch):
    add_run(engine, "busy", [0, 0, 1000, 1000])
    calls, encode = [], vectors.encode

    def slow(*args, **kwargs):
        calls.append(1)
        time.sleep(.2)
        return encode(*args, **kwargs)
    monkeypatch.setattr(vectors, "encode", slow)
    paths = []
    threads = [threading.Thread(target=lambda: paths.append(engine.prior_vectors("busy"))) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(calls) == 1 and len(set(paths)) == 1 and paths[0].name == vectors.FILE
    assert [p.name for p in paths[0].parent.iterdir()] == [vectors.FILE]          # no temporary file left behind
    monkeypatch.setattr(vectors, "encode", encode)
    for run_id, mode, prior in (("detail-run", "detail", None), ("column-run", "tiled", {"cells": {"kind": "column", "column": "c"}})):
        engine._prebuild_vectors(add_run(engine, run_id, [0, 0, 1000, 1000], mode, prior))
        assert not any(t.name == f"vectors-{run_id}" for t in threading.enumerate())
        assert not (engine.root / run_id / "xenium").exists()
    run = add_run(engine, "whole-run", [0, 0, 1000, 1000])
    engine._prebuild_vectors(run)
    assert wait_for(lambda: (run.dir / "xenium" / vectors.FILE).exists())


def test_creating_a_whole_sample_run_builds_its_vectors_while_it_prepares(engine, monkeypatch):
    from sopa_companion.engine.runs import Runs
    monkeypatch.setattr(Runs, "_execute", lambda self, run: None)              # no preparation, no proseg
    options = {part: [{**spec, "label": spec["key"], "count": 1}] for part, spec in SHAPES.items()}
    engine.zarr.details[SID] = {"schema": 2, "id": SID, "kind": "zarr", "name": "synthetic", "path": "/nonexistent/synthetic.zarr",
                                "points_key": "transcripts", "transcripts": 40_000, "extent": [0, 0, 1500, 1500],
                                "width_um": 1500, "height_um": 1500, "image_key": None, "images": [], "channels": [],
                                "overview": None, "shapes": [{"key": CELLS, "count": 300}, {"key": NUCLEI, "count": 280}],
                                "prior_options": options, "suggested_prior": SHAPES, "limits": {"detail_um": 1200}}
    run = engine.create(SID, region={"kind": "full"}, prior=SHAPES)
    assert run["mode"] == "tiled"
    path = engine.get(run["id"]).dir / "xenium" / vectors.FILE
    assert wait_for(lambda: path.exists())
    assert vectors.to_json(vectors.decode(unpacked(path))) == json.loads(engine.xenium_boundaries(run["id"]).read_text())


def test_served_as_the_zstd_file_with_immutable_caching(config):
    from sopa_companion.server import create_app
    application = create_app(config, token="t" * 40)
    with TestClient(application, base_url=BASE) as client:
        runs = application.state.runs
        inject(runs.zarr)
        add_run(runs, "served", [100, 150, 900, 800])
        url = "/api/v1/runs/served/xenium"
        old = json.loads(runs.xenium_boundaries("served").read_text())    # the JSON form, kept as the reference
        # Only the binary is served (no page asks for the JSON form).
        assert client.get(url, headers=auth()).status_code == 404
        packed = client.get(url + "?format=bin&v=1", headers=auth({"Accept-Encoding": "zstd"}))   # httpx inflates it
        assert packed.status_code == 200 and packed.headers["content-encoding"] == "zstd"
        assert int(packed.headers["content-length"]) == (runs.root / "served" / "xenium" / vectors.FILE).stat().st_size
        assert packed.headers["cache-control"] == "private, max-age=31536000, immutable"
        assert packed.headers["vary"] == "Accept-Encoding" and packed.headers["content-type"] == "application/octet-stream"
        plain = client.get(url + "?format=bin&v=1", headers=auth({"Accept-Encoding": "identity"}))    # inflated by the engine
        assert plain.headers.get("content-encoding") is None and plain.content == packed.content
        assert vectors.to_json(vectors.decode(packed.content)) == old
        for query in ("?format=bin", "?format=bin&v=2", "?format=geojson&v=1"):
            assert client.get(url + query, headers=auth()).status_code == 404, query
        assert client.get(url + "?format=bin&v=one", headers=auth()).status_code == 422
        assert client.get("/api/v1/runs/missing/xenium?format=bin&v=1", headers=auth()).status_code == 404


# ---- one selection for every prior source, and builds that outlive their run --------------------------------------------
def test_every_prior_source_selects_the_same_way(engine, tmp_path):
    """One selection (vectors.hits/gather) for the picker's worker body, the run's JSON and its binary: the same
    polygons, order, rounding and ids for the same box and limit."""
    import orjson
    from sopa_companion.engine.shape_query import encode
    data = engine.zarr.outlines[(SID, CELLS)]
    for box, limit in (([100, 150, 900, 800], 60_000), ([0, 0, 1000, 1000], 37), (None, 500), ([2000, 2000, 3000, 3000], 10)):
        shapes = engine.zarr.shapes(SID, CELLS, box, limit)
        assert orjson.loads(encode(data, CELLS, box, limit)) == json.loads(json.dumps(shapes)), (box, limit)
        if box is not None:
            layer = vectors.select(data, box, limit)
            decoded = vectors.to_json(vectors.decode(vectors.encode({"cells": layer, "nuclei": None})))
            assert decoded["cells"] == json.loads(json.dumps(shapes["polygons"])), (box, limit)


def test_both_forms_use_the_one_margin_and_limit_per_mode(engine, monkeypatch):
    seen = []
    real = vectors.select
    monkeypatch.setattr(engine.zarr, "shapes", lambda sid, key, box, limit: seen.append(("json", key, box, limit)) or
                        {"polygons": []})
    monkeypatch.setattr(vectors, "select", lambda outlines, box, limit: seen.append(("bin", box, limit)) or
                        real(outlines, box, 0))
    for mode in ("tiled", "detail"):
        run = add_run(engine, f"{mode}-limits", [100, 200, 300, 400], mode)
        engine._prior_shapes(run)
        engine._prior_layers(run)
    m, limits = vectors.PRIOR_MARGIN_UM, vectors.PRIOR_LIMITS
    box = [100 - m, 200 - m, 300 + m, 400 + m]
    assert seen == [("json", CELLS, box, limits["tiled"]), ("json", NUCLEI, box, limits["tiled"]),
                    ("bin", box, limits["tiled"]), ("bin", box, limits["tiled"]),
                    ("json", CELLS, box, limits["detail"]), ("json", NUCLEI, box, limits["detail"]),
                    ("bin", box, limits["detail"]), ("bin", box, limits["detail"])]


def test_a_forgotten_sample_gives_the_same_outlines_in_both_forms(engine, monkeypatch):
    """The JSON came back empty for a sample forgotten from the recent list while the
    binary opened it again; both now open it again where the run found it (as its image stream does)."""
    add_run(engine, "forgotten", [100, 150, 900, 800])
    entry = engine.zarr.recent.pop(0)
    opened = []
    monkeypatch.setattr(engine.zarr, "open", lambda path, recent=True: opened.append((path, recent)) or
                        engine.zarr.recent.insert(0, entry))
    value = json.loads(engine.xenium_boundaries("forgotten").read_text())
    # Opened again for the run, off the open dialog's list (it put the sample back on top).
    assert opened == [("/nonexistent/synthetic.zarr", False)] and value["cells"] and value["nuclei"]
    engine.zarr.recent.pop(0)
    assert vectors.to_json(vectors.decode(unpacked(engine.prior_vectors("forgotten")))) == value
    assert len(opened) == 2


def test_a_build_never_writes_into_a_run_deleted_meanwhile(engine, monkeypatch):
    """A vectors build that outlived its run's delete wrote into the folder NTFS had
    kept (a held file), which then stayed forever."""
    from sopa_companion.engine.runs import RunError
    run = add_run(engine, "doomed", [100, 150, 900, 800])
    started, release, encode = threading.Event(), threading.Event(), vectors.encode

    def slow(*args, **kwargs):
        started.set()
        release.wait(10)
        return encode(*args, **kwargs)
    monkeypatch.setattr(vectors, "encode", slow)
    errors = []

    def build():
        try:
            engine.prior_vectors("doomed")
        except RunError as exc:
            errors.append(exc.status)
    thread = threading.Thread(target=build)
    thread.start()
    assert started.wait(10)
    held = run.dir / "live"
    held.mkdir()
    (held / ".fuse_hidden000001").write_bytes(b"x")          # what keeps a deleted run's folder on NTFS
    monkeypatch.setattr(engine, "_remove", lambda folder, delays=(): None)
    engine.delete("doomed")
    assert "doomed" not in engine.vector_guards
    release.set()
    thread.join(10)
    assert errors == [404]
    assert not (run.dir / "xenium").exists()


def test_cell_values_fixture_is_what_the_engine_encodes():
    """frontend/test/fixtures/stain-cells-{none,dapi}.bin (the page's range-chooser tests read them, main-smoke's WIRE):
    exactly what encode_cells makes of stain-cells.json, so the page decodes the engine's own bytes, never a frozen copy
    of an older layout (null in the JSON: an unknown brightness, NaN)."""
    import math
    spec = json.loads((FIXTURES / "stain-cells.json").read_text())
    nan = lambda v: math.nan if v is None else v
    assert vectors.encode_cells(spec["area"], spec["flags"], {}, spec["meta"]) == (FIXTURES / "stain-cells-none.bin").read_bytes()
    brightness = {k: [nan(v) for v in values] for k, values in spec["brightness"].items()}
    assert vectors.encode_cells(spec["area"], spec["flags"], brightness, spec["meta"]) == \
        (FIXTURES / "stain-cells-dapi.bin").read_bytes()

