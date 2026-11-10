"""The GPU worker's Compare op ("compare", analysis/worker.py Worker.compare) and its
queue in the engine's scheduler (engine/analysis.py submit_compare), all without a GPU: the worker's own code with
numpy (the same code cupy runs: its compressed rows, cuts and kinds are written once for both). The checks hold the op
as planned: each column by its own count (the run's frame by proseg's, from its counts file: read as a counts file of
the same cells reads), one area, the mixing check's own calculator, exactly 100 whole transcripts per cell for the
like-for-like number (the run's cut from proseg's sampled count f, or from Binomial(n, e / n) draws where its frame has
none), kinds made once per model and kept beside it, and never a run's live clusters, layout or colours."""
from __future__ import annotations

import json
import os
from pathlib import Path
import threading

import numpy as np
import pytest

from sopa_companion.analysis import worker as W
from sopa_companion.engine.analysis import AnalysisError
from sopa_companion.analysis import counts_file
from test_analysis import (FakeRun, JUDGE, MODEL_GENES, NCELLS, RUN_GENES, X0, Y0, cell_stats, expected_scores,
                           frame_plane, manager, mixing_model, model_rows, proseg_count, write_run)
from test_worker_deeper import EXTRA, RUN_IDS, UNLEARNED, make_deep

QC = {"mixed_at": 0.3, "judge_min": JUDGE, "fit_min": 5}
LEARNED = [c for c in RUN_IDS if c not in UNLEARNED] + EXTRA          # make_deep's learned cells (cells.parquet)


def dense_of(m: W.Csr) -> np.ndarray:
    out = np.zeros(m.shape, np.float32)
    rows = np.repeat(np.arange(m.shape[0]), np.diff(m.indptr))
    np.add.at(out, (rows, m.indices), m.data)
    return out


def csr_of(dense: np.ndarray) -> W.Csr:
    rows, cols = np.nonzero(dense)
    return W.Csr(dense[rows, cols].astype(np.float32), cols.astype(np.int32),
                 np.searchsorted(rows, np.arange(len(dense) + 1)).astype(np.int64), dense.shape)


def write_counts(path, dense, genes, x, y, ids=None, **names):
    """A counts file as zarr_bridge/helper.py count writes it (names: other accepted key names, e.g. cx="x")."""
    m = csr_of(np.asarray(dense, np.float32))
    arrays = {"data": m.data, "indices": m.indices, "indptr": m.indptr, "shape": np.array(m.shape), "genes":
              np.array(genes), "x": np.asarray(x, np.float64), "y": np.asarray(y, np.float64)}
    if ids is not None:
        arrays["ids"] = np.array(ids)
    for new, old in names.items():
        arrays[new] = arrays.pop(old)
    np.savez(path, **arrays)
    return str(path)


# ---- compressed rows ---------------------------------------------------------------------------------------------------
def test_compressed_rows_do_what_dense_arithmetic_does():
    rng = np.random.default_rng(0)
    dense = (rng.poisson(rng.gamma(0.4, 3.0, (9, 7))) * 0.75).astype(np.float32)   # proseg's averaged count
    dense[[2, 5]] = 0                                                  # empty rows
    m = csr_of(dense)
    pick = np.array([4, 2, 4, 0, 8])                                   # repeated, unordered, with an empty row
    assert np.array_equal(dense_of(W.csr_take(np, m, pick)), dense[pick])
    assert W.csr_take(np, m, np.array([2, 5])).shape == (2, 7)
    assert np.array_equal(dense_of(W.csr_stack(np, [W.csr_take(np, m, pick), m])), np.concatenate([dense[pick], dense]))
    assert np.array_equal(W.csr_row_sums(np, m), dense.sum(axis=1))
    index = np.array([3, -1, 0, 1, -1, 2, 4])                          # genes renamed, two dropped
    renamed = np.zeros((9, 5), np.float32)
    renamed[:, index[index >= 0]] = dense[:, index >= 0]
    assert np.array_equal(dense_of(W.csr_columns(np, m, index, 5)), renamed)
    weights = rng.normal(size=(7, 4)).astype(np.float32)
    scale = rng.uniform(0.5, 2.0, 9).astype(np.float32)
    for chunk in (1, 4, W.DENSE_ROWS):
        np.testing.assert_allclose(W.log1p_matmul(np, m, weights, chunk=chunk), np.log1p(dense) @ weights, rtol=1e-5)
        np.testing.assert_allclose(W.log1p_matmul(np, m, weights, scale, chunk=chunk),
                                   np.log1p(dense * scale[:, None]) @ weights, rtol=1e-5)


def test_counts_files_are_read_strictly(tmp_path):
    dense = np.array([[3, 0, 1], [0, 0, 0], [2, 5, 0]], np.float32)
    path = write_counts(tmp_path / "a.npz", dense, ["A", "B", "C"], [1.0, 2.0, np.nan], [4.0, 5.0, 6.0], ids=["p", "q", "r"])
    m, ids, x, y, genes = W.read_counts_file(path)
    assert np.array_equal(dense_of(m), dense) and ids == ["p", "q", "r"] and genes == ["A", "B", "C"]
    assert np.isnan(x[2]) and y.tolist() == [4.0, 5.0, 6.0]
    other = write_counts(tmp_path / "b.npz", dense, ["A", "B", "C"], [1, 2, 3], [4, 5, 6], ids=["p", "q", "r"],
                         cx="x", cy="y", gene_names="genes", cell_ids="ids")
    assert W.read_counts_file(other)[1] == ["p", "q", "r"] and W.read_counts_file(other)[4] == ["A", "B", "C"]
    good = dict(np.load(path))
    for change, problem in (({"indptr": np.array([0, 2, 2], np.int64)}, "malformed"),
                            ({"genes": np.array(["A", "B"])}, "malformed"),
                            ({"data": -good["data"]}, "malformed"),
                            ({"indices": good["indices"] + 3}, "malformed"),
                            ({"x": np.zeros(2)}, "malformed")):
        np.savez(tmp_path / "bad.npz", **{**good, **change})
        with pytest.raises(ValueError, match=problem):
            W.read_counts_file(tmp_path / "bad.npz")
    np.savez(tmp_path / "bad.npz", **{k: v for k, v in good.items() if k != "y"})
    with pytest.raises(ValueError, match="no y"):
        W.read_counts_file(tmp_path / "bad.npz")


# ---- like for like: exactly 100 transcripts per cell -------------------------------------------------------------------
def test_a_cut_keeps_exactly_the_depth_drawn_without_replacement():
    rng = np.random.default_rng(1)
    dense = rng.poisson(rng.gamma(0.5, 30.0, (40, 25))).astype(np.float32)
    dense[0] = 0
    dense[0, :10] = 10                                                 # exactly 100: kept whole
    dense = dense[dense.sum(axis=1) >= 100]
    m = csr_of(dense)
    seeds = [W.compare_seed(0, "a", d) for d in range(3)]
    cuts = W.cut_to(np, m, 100, seeds)
    for cut in cuts:
        got = dense_of(cut)
        assert (got.sum(axis=1) == 100).all() and (got <= dense).all() and (got >= 0).all()
        assert np.array_equal(got[0], dense[0])
        assert np.array_equal(cut.indices, m.indices) and np.array_equal(cut.indptr, m.indptr)
    assert not np.array_equal(cuts[0].data, cuts[1].data)              # three draws, not one
    again = W.cut_to(np, m, 100, seeds, chunk=7)                       # any chunking draws the same
    assert all(np.array_equal(a.data, b.data) for a, b in zip(cuts, again))
    # Without replacement: each gene's kept transcripts average depth x its share of the row (hypergeometric), and
    # spread less than binomial draws would.
    row = np.zeros((1, 4), np.float32)
    row[0] = [300, 100, 50, 50]
    draws = np.stack([dense_of(c)[0] for c in W.cut_to(np, csr_of(row), 100, list(range(400)))])
    np.testing.assert_allclose(draws.mean(axis=0), [60, 20, 10, 10], atol=1.0)
    p = row[0] / 500
    hyper = 100 * p * (1 - p) * (500 - 100) / (500 - 1)
    np.testing.assert_allclose(draws.var(axis=0), hyper, rtol=0.25)


def test_like_for_like_scores_the_cut_cells_with_the_checks_calculator(tmp_path):
    from test_analysis import make_model
    encoder = W.load_encoder(make_model(tmp_path / "encoder.npz", MODEL_GENES))
    device = W.on_device(np, encoder)
    rng = np.random.default_rng(2)
    dense = rng.poisson(rng.gamma(0.6, 40.0, (60, len(MODEL_GENES)))).astype(np.float32)
    m = csr_of(dense)
    rows = np.nonzero(dense.sum(axis=1) >= 100)[0]
    seeds = [W.compare_seed(3, "a", d) for d in range(W.LIKE_DRAWS)]
    got = W.like_for_like_at(np, m, rows, np.ascontiguousarray(encoder["W0"].T), device, QC, seeds)
    expected = []
    for cut in W.cut_to(np, W.csr_take(np, m, rows), 100, seeds):
        share = W.neighbour_share(W.mixing_weights(np, np.log1p(dense_of(cut)) @ encoder["W0"].T, encoder))
        expected.append(100.0 * (np.floor(share * 250) >= 75).sum() / len(rows))
    assert got == {"per100": round(float(np.mean(expected)), 2), "sd": round(float(np.std(expected, ddof=1)), 2),
                   "cells": len(rows), "depth": 100, "draws": 3}
    assert W.like_for_like_at(np, m, np.zeros(0, int), encoder["W0"].T, device, QC, seeds)["per100"] is None


# ---- the area, the points ----------------------------------------------------------------------------------------------
def test_the_area_is_the_tiles_holding_enough_cells_of_every_column():
    rng = np.random.default_rng(4)

    def cells(x0, x1, y0, y1, n):
        return rng.uniform(x0, x1, n), rng.uniform(y0, y1, n)
    a = cells(0, 400, 0, 300, 4000)                                    # a whole sample: 4 x 3 tiles
    b = cells(100, 300, 0, 200, 2000)                                  # a patch: 2 x 2 tiles of it
    bx, by = np.append(b[0], [350.0, 351.0]), np.append(b[1], [50.0, 51.0])   # 2 stray cells: too few for a tile
    bx[0] = np.nan                                                     # a cell without a centre
    area, inside, own = W.shared_area([a, (bx, by)])
    assert area == {"box": [100.0, 0.0, 300.0, 200.0], "size_um": [200.0, 200.0], "tiles": 4, "tile_um": 100.0}
    assert own == [12, 4]
    assert np.array_equal(inside[0], (a[0] >= 100) & (a[0] < 300) & (a[1] < 200))
    assert not inside[1][0] and not inside[1][-2:].any() and inside[1][1:-2].all()
    # A limit box (a run's region) anchors the grid at its corner; cells outside it never count.
    area, inside, _ = W.shared_area([a, (bx, by)], [150.0, 0.0, 400.0, 300.0])
    assert area["box"] == [150.0, 0.0, 350.0, 200.0] and area["tiles"] == 4
    assert not (inside[0] & (a[0] < 150)).any()
    # Columns that share no tissue share no area.
    area, inside, own = W.shared_area([cells(0, 100, 0, 100, 50), cells(500, 600, 0, 100, 50)])
    assert area == {"box": None, "size_um": [0.0, 0.0], "tiles": 0, "tile_um": 100.0}
    assert not inside[0].any() and not inside[1].any() and own == [1, 1]
    assert W.area_limit(None) is None and W.area_limit({"box": [0, 0, 1, 2]}) == [0.0, 0.0, 1.0, 2.0]
    for bad in ([0, 0, 1], [0, 0, -1, 2], {"box": ["a", 0, 1, 1]}):
        with pytest.raises((ValueError, TypeError)):
            W.area_limit(bad)


def test_the_cap_is_shared_evenly_between_the_columns_and_picks_are_seeded():
    sizes = [41775, 40602, 5037]
    quotas = W.even_quotas(sizes, 60000)
    assert sum(quotas) == 60000 and all(abs(q / s - 60000 / sum(sizes)) < 1e-3 for q, s in zip(quotas, sizes))
    assert W.even_quotas([10, 20], 60000) == [10, 20] and W.even_quotas([1, 1, 1], 2) == [1, 1, 0]
    pick = W.picked(1000, 300, 7, 1)
    assert len(set(pick.tolist())) == 300 and np.array_equal(pick, np.sort(pick))
    assert np.array_equal(pick, W.picked(1000, 300, 7, 1)) and not np.array_equal(pick, W.picked(1000, 300, 7, 2))
    assert np.array_equal(W.picked(5, 9, 0, 0), np.arange(5))


# ---- kinds ---------------------------------------------------------------------------------------------------------------
def test_kinds_are_groups_of_at_least_one_in_100_cells_most_common_first():
    rng = np.random.default_rng(5)
    truth = np.repeat(np.arange(3), [600, 300, 100])
    z = (np.array([[0, 0], [12, 0], [0, 12]])[truth] + rng.normal(0, 1, (1000, 2))).astype(np.float32)
    centres, labels, sizes = W.kmeans(np, z, 3, 0)
    assert sorted(sizes.tolist()) == [100, 300, 600]
    assert all(len(set(truth[labels == g].tolist())) == 1 for g in range(3))    # each group one true group
    outliers = np.concatenate([z, np.array([[40, 40], [-40, 40], [40, -40]], np.float32)])
    centres, labels, sizes = W.choose_kinds(np, outliers)
    assert 2 <= len(sizes) <= W.KINDS_MAX and sizes.min() >= np.ceil(0.01 * len(outliers))
    assert sizes.tolist() == sorted(sizes.tolist(), reverse=True) and np.bincount(labels).tolist() == sizes.tolist()
    assert centres.shape == (len(sizes), 2) and centres.dtype == np.float32
    for g in range(len(sizes)):                                        # every kind's centre is its cells' mean
        np.testing.assert_allclose(centres[g], outliers[labels == g].mean(axis=0), atol=1e-4)
    near = W.nearest_kind(np, np.array([[0.2, 0.1], [np.nan, 0]], np.float32), centres)
    assert near[1] == W.NO_KIND and near[0] == labels[0]
    # Repeated or empty starting groups still end with every group holding points.
    _, labels, sizes = W.kmeans(np, np.repeat(z[:5], 20, axis=0), 4, 1)
    assert sizes.sum() == 100 and (sizes > 0).sum() >= 4 - 1


def test_kind_colours_stay_clear_of_the_mixing_colours_and_of_each_other():
    import math
    mixing = ["#6b3f63", "#b2477a", "#f08cbc", "#d55181", "#5c6573"]   # stage.js MIXING_COLORS (little ... accent)
    lab = [W._lab(c) for c in W.KIND_PALETTE]
    assert len(W.KIND_PALETTE) == W.KINDS_MAX == len(set(W.KIND_PALETTE))
    assert min(math.dist(a, W._lab(m)) for a in lab for m in mixing) >= 25
    assert min(math.dist(a, b) for i, a in enumerate(lab) for b in lab[i + 1:]) >= 26

    def luminance(colour):                                             # WCAG relative luminance
        channels = [int(colour[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        r, g, b = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
        return 0.2126 * r + 0.7152 * g + 0.0722 * b
    plots = luminance("#0b0e13")                                       # Compare's UMAP plots (compare.mjs COLORS)
    assert min((luminance(c) + 0.05) / (plots + 0.05) for c in W.KIND_PALETTE) >= 3   # every kind's dots show there


def test_descriptions_are_go_deepers_and_kind_names_its_kind_words(tmp_path):
    run = write_run(tmp_path / "run")
    deep = W.load_deep(make_deep(tmp_path / "model", run) / "deep.npz")
    rng = np.random.default_rng(6)
    dense = rng.poisson(rng.gamma(0.5, 20.0, (8, len(MODEL_GENES)))).astype(np.float32)
    dense[3] = 0
    m = csr_of(dense)
    z = W.z_means(np, W.deep_device(np, deep), m, W.csr_row_sums(np, m))
    for r in range(8):
        if r == 3:
            assert np.isnan(z[r]).all()
        else:
            np.testing.assert_allclose(z[r], W.deep_z_mean(np, deep, (dense[r] / dense[r].mean())[None])[0], atol=2e-5)
    centres = np.stack([z[0], z[1], z[0]])                            # the third repeats the first
    names, genes = W.kind_names(deep, centres)
    profile = W.deep_profile(np, deep, z[0])
    words = W.deep_kind_words(np, deep, profile, 3)
    assert genes[0][:len(words)] == words and names[0] == " · ".join(deep["genes"][g] for g in genes[0])
    assert len(genes[0]) == 3 and set(genes[2]) != set(genes[0]) and genes[2][:3] == genes[0][:3]


# ---- the op --------------------------------------------------------------------------------------------------------------
@pytest.fixture
def setup(tmp_path):
    """A run (write_run, frame 2), a model with details (make_deep), the learning cells' counts file, a numpy worker
    and a compare request with the run's own column."""
    def make(tiled=False):
        run = write_run(tmp_path / ("tiled" if tiled else "run"), tiled=tiled)
        live = run / "live"
        (live / "cells.txt").write_text("".join(c + "\n" for c in RUN_IDS))
        model = make_deep(tmp_path / "model", run)
        frame = json.loads((live / "frames.jsonl").read_text().splitlines()[2])
        rng = np.random.default_rng(7)
        learning = write_counts(tmp_path / "learning.npz", rng.poisson(30, (len(LEARNED), len(MODEL_GENES))),
                                MODEL_GENES, np.zeros(len(LEARNED)), np.zeros(len(LEARNED)), ids=LEARNED)
        spec = {"run": "r1", "seq": 2, "iter": 2, "transcripts": str(live / "transcripts.bin.gz"),
                "genes": str(live / "genes.txt"), "counts_file": str(live / "frames" / "000002.counts.zst"),
                "frame": str(live / frame["file"]), "format": frame.get("format", "dense"),
                **{k: frame[k] for k in ("nx", "ny", "nz", "x0", "y0", "voxel")}, "z_origin": 0.0, "z_spacing": 0.5,
                "ncells": NCELLS, "cells_file": str(live / frame["cells_file"])}
        request = {"op": "compare", "run": "r1", "model": str(model), "model_id": "feedface",
                   "kinds": {"path": str(model / W.KINDS_FILE), "counts": learning},
                   "columns": [{"key": "run", "name": "This run · iteration 2", "frame": spec}], "area": None,
                   "max_points": W.COMPARE_MAX_POINTS, "seed": 0, **QC, "out_dir": str(tmp_path / "out")}
        n = 300                                                        # another layer's cells on the run's tissue
        other = write_counts(tmp_path / "other.npz", rng.poisson(rng.gamma(0.6, 50, (n, len(MODEL_GENES)))),
                             MODEL_GENES[::-1], rng.uniform(X0, X0 + 30, n), rng.uniform(Y0, Y0 + 20, n))
        return {"run": run, "live": live, "model": model, "request": request, "worker": W.Worker(xp=np),
                "other": other, "two": {**request, "columns": [*request["columns"],
                                                               {"key": "other", "name": "B", "counts": other}]}}
    return make


@pytest.mark.parametrize("tiled", [False, True])
def test_the_runs_column_is_proseg_count_as_a_counts_file_of_it_reads(setup, tiled):
    """The run's column reads proseg's count of its frame (its counts file: the averaged count, the same for a dense and
    a tiled run; never its outlines), as a counts file holding the same cells' counts reads: the same numbers; its
    like-for-like number cuts proseg's sampled count f (the frame has one) to exactly 100 transcripts per cell."""
    s = setup(tiled)
    e = proseg_count(s["run"], 2)                                      # write_run's frame 2: 0.75 of each assigned
    stats = cell_stats(frame_plane(2), 2)
    cx, cy = stats[5].view(np.float32), stats[6].view(np.float32)
    copy = write_counts(s["model"].parent / "copy.npz", e, RUN_GENES, cx, cy, ids=RUN_IDS)
    request = {**s["request"], "columns": [s["request"]["columns"][0], {"key": "copy", "name": "copy", "counts": copy}]}
    reply = W.answer(s["worker"], json.dumps(request))
    assert reply["ok"] and reply["seq"] is None and reply["run"] == "r1", reply
    result = reply["compare"]
    frame, copied = result["columns"]
    keys = ("cells", "cells_in_area", "judged", "mixed", "mixed_per100", "histogram", "median_tx", "under100_share",
            "umap_points", "tiles")
    assert {k: frame[k] for k in keys} == {k: copied[k] for k in keys} and frame["state"] == copied["state"] == "ready"
    # ... and as worked out here from proseg's count, the model's calculator and the check's byte codes.
    encoder = W.load_encoder(s["model"] / "encoder.npz")
    totals, share = expected_scores(encoder, e)
    inside = np.isfinite(cx)                                           # cell 12 vanished: no centre, not on the area
    judged = inside & (totals >= JUDGE) & np.isfinite(share)
    codes = np.where(judged, np.floor(share * 250), 255)
    mixed = judged & (codes >= 75)
    assert frame["cells"] == NCELLS and frame["cells_in_area"] == NCELLS - 1 and frame["judged"] == judged.sum() > 0
    assert frame["mixed"] == mixed.sum() and frame["mixed_per100"] == round(100 * mixed.sum() / judged.sum(), 2)
    assert frame["histogram"] == np.bincount(np.searchsorted([5, 13, 25, 50, 75, 125], codes[judged], side="right"),
                                             minlength=7).tolist()
    assert sum(frame["histogram"]) == frame["judged"] and sum(frame["histogram"][5:]) == frame["mixed"]
    assert frame["median_tx"] == round(float(np.median(totals[inside])), 1)
    assert frame["under100_share"] == round(float((totals[inside] < JUDGE).mean()), 4)
    # Like for like: proseg's sampled count f of the frame (write_run: floor(0.75 n)), each judged cell holding 100 of
    # them cut to exactly 100, the column's draws.
    read = counts_file.read(s["live"] / "frames" / "000002.counts.zst")
    f = model_rows(encoder, W.Worker(xp=np).matrix(read, "f"))
    rows = np.nonzero(judged & (f.sum(axis=1) >= W.LIKE_DEPTH))[0]
    want = W.like_for_like_at(np, csr_of(f), rows, np.ascontiguousarray(encoder["W0"].T), W.on_device(np, encoder), QC,
                              [W.compare_seed(0, "run", d) for d in range(W.LIKE_DRAWS)])
    assert (frame["like100_per100"], frame["like100_sd"], frame["like100_cells"]) == \
        (want["per100"], want["sd"], want["cells"]) and want["cells"] > 0
    assert result["area"] == {"box": [100.0, 0.0, 200.0, 100.0], "size_um": [100.0, 100.0], "tiles": 1, "tile_um": 100.0}
    assert result["model"] == "feedface" and result["layout_error"] == "too_few_cells"   # 2 x 8 cells: no picture
    assert {"counts", "score", "like", "kinds", "points", "layout", "write"} <= set(result["ms"])


def test_without_a_sampled_count_the_runs_column_is_cut_from_binomial_draws(setup, capsys):
    """A frame proseg kept no sampled count with (a request frame): its like-for-like cut comes from one draw of
    Binomial(n, e / n) per pair, seeded by the request and the column (thin_values, then cut_to), the same every time;
    its other numbers are proseg's averaged count as ever. No counts file: the column fails (one plain line in the
    worker's log), the others stand."""
    s = setup()
    live = s["live"]
    spec = {**s["request"]["columns"][0]["frame"], "seq": 1, "iter": 1,
            "counts_file": str(live / "frames" / "000001.counts.zst"), "cells_file": str(live / "frames/000001.cells.zst")}
    read = counts_file.read(spec["counts_file"])
    assert read.f is None
    request = {**s["request"], "seed": 4, "columns": [{"key": "run", "name": "This run", "frame": spec}]}
    column = s["worker"].handle(request)["compare"]["columns"][0]
    assert column["state"] == "ready" and column["like100_cells"] > 0
    assert s["worker"].handle(request)["compare"]["columns"][0] == column      # the same draws again
    encoder = W.load_encoder(s["model"] / "encoder.npz")
    worker = W.Worker(xp=np)
    n = worker.matrix(read, "n")
    keep = np.minimum(worker.matrix(read, "e") / np.maximum(n, 1), 1)
    entries = np.nonzero(n)
    drawn = np.zeros_like(n)
    drawn[entries] = W.thin_values(np, n[entries], keep[entries].astype(np.float32), W.compare_seed(4, "run", "whole"))
    assert (drawn <= n).all() and 0.6 < drawn.sum() / n.sum() < 0.9          # e / n = 0.75 of every pair
    whole = model_rows(encoder, drawn)
    totals, share = expected_scores(encoder, proseg_count(s["run"], 1))
    stats = cell_stats(frame_plane(1), 1)
    judged = np.isfinite(stats[5].view(np.float32)) & (totals >= JUDGE) & np.isfinite(share)
    rows = np.nonzero(judged & (whole.sum(axis=1) >= W.LIKE_DEPTH))[0]
    want = W.like_for_like_at(np, csr_of(whole), rows, np.ascontiguousarray(encoder["W0"].T), W.on_device(np, encoder),
                              QC, [W.compare_seed(4, "run", d) for d in range(W.LIKE_DRAWS)])
    assert (column["like100_per100"], column["like100_sd"], column["like100_cells"]) == \
        (want["per100"], want["sd"], want["cells"])
    capsys.readouterr()
    gone = s["worker"].handle({**request, "columns": [request["columns"][0], {"key": "other", "name": "B",
                                                                             "counts": s["other"]},
                                                      {"key": "old", "name": "old", "frame": {**spec,
                                                                                              "counts_file": None}}]})
    run, other, old = gone["compare"]["columns"]
    assert run["state"] == other["state"] == "ready" and old["state"] == "failed" and "NoCounts" in old["error"]
    err = capsys.readouterr().err
    assert "Traceback" not in err and "compare: column 'old' could not be counted: NoCounts: the request names no " \
                                      "counts file" in err, err


def test_the_runs_cut_is_proseg_sampled_count_else_one_binomial_draw_of_each_pair(setup, tmp_path):
    """compare_frame itself: the run's column is proseg's count of the frame (LIVE_ARRAY's: e) over the model's genes;
    its whole transcripts for the like-for-like cut are proseg's sampled count f where the frame has one, else one draw
    of Binomial(n, e / n) per stored pair seeded by compare_seed(seed, key, "whole"): never e itself, n itself, nor a
    draw when f is there. Here f is a draw of its own (not floor(e)) and proseg's keep differs from pair to pair; under
    a model by which some cells look mixed, the column's like-for-like number is the one of that cut."""
    s = setup()
    live = s["live"]
    read = counts_file.read(live / "frames" / "000002.counts.zst")
    rng = np.random.default_rng(21)
    q = rng.uniform(0.35, 0.95, read.nnz)                              # proseg's keep, pair by pair: e = q n
    rows = {"ncells": read.ncells, "ngenes": read.ngenes, "indptr": read.indptr, "genes": read.genes, "n": read.n,
            "e": read.n * q}
    counts_file.write(tmp_path / "f.counts.zst", iteration=2, **rows, f=rng.binomial(read.n, q))   # f: a draw
    counts_file.write(tmp_path / "n.counts.zst", iteration=2, **rows)                              # no f
    stored = counts_file.read(tmp_path / "f.counts.zst")
    assert not np.array_equal(stored.f, np.floor(stored.e))
    pair = (np.repeat(np.arange(NCELLS), np.diff(stored.indptr)), stored.genes)

    def over_run(values):                                              # one value per stored pair -> cells x genes
        dense = np.zeros((NCELLS, len(RUN_GENES)), np.float32)
        dense[pair] = values
        return dense
    encoder = W.load_encoder(s["model"] / "encoder.npz")
    spec = s["request"]["columns"][0]["frame"]
    seed = W.compare_seed(5, "run", "whole")
    worker = W.Worker(xp=np)
    sampled = worker.compare_frame({**spec, "counts_file": str(tmp_path / "f.counts.zst")}, encoder, seed)
    drawn = worker.compare_frame({**spec, "counts_file": str(tmp_path / "n.counts.zst")}, encoder, seed)
    for column in (sampled, drawn):
        assert np.array_equal(dense_of(column["csr"]), model_rows(encoder, over_run(stored.e)))
    assert np.array_equal(dense_of(sampled["whole"]), model_rows(encoder, over_run(stored.f)))
    keep = np.minimum(stored.e / np.maximum(stored.n, 1).astype(np.float32), np.float32(1))
    binomial = W.thin_values(np, stored.n.astype(np.float32), keep, seed)
    assert np.array_equal(dense_of(drawn["whole"]), model_rows(encoder, over_run(binomial)))
    assert not np.array_equal(binomial, stored.f) and (binomial <= stored.n).all() and binomial.sum() < stored.n.sum()
    # Through the op, under a model by which some cells look mixed: the like-for-like number of that cut.
    e = over_run(stored.e)
    judged_rows = model_rows(encoder, e)
    model = mixing_model(tmp_path / "mixing" / "encoder.npz", judged_rows[judged_rows.sum(axis=1) >= JUDGE])
    mixing = W.load_encoder(model)
    totals, share = expected_scores(mixing, e)
    stats = cell_stats(frame_plane(2), 2)
    judged = np.isfinite(stats[5].view(np.float32)) & (totals >= JUDGE) & np.isfinite(share)
    seeds = [W.compare_seed(5, "run", d) for d in range(W.LIKE_DRAWS)]

    def like(whole):
        cut = model_rows(mixing, whole)
        rows = np.nonzero(judged & (cut.sum(axis=1) >= W.LIKE_DEPTH))[0]
        found = W.like_for_like_at(np, csr_of(cut), rows, np.ascontiguousarray(mixing["W0"].T),
                                   W.on_device(np, mixing), QC, seeds)
        return found["per100"], found["sd"], found["cells"]
    for name, whole in (("f", over_run(stored.f)), ("n", over_run(binomial))):
        request = {**s["request"], "model": str(model.parent), "kinds": None, "seed": 5,
                   "columns": [{"key": "run", "name": "This run",
                                "frame": {**spec, "counts_file": str(tmp_path / f"{name}.counts.zst")}}]}
        column = s["worker"].handle(request)["compare"]["columns"][0]
        assert column["state"] == "ready" and 0 < column["mixed"] < column["judged"] == judged.sum(), column
        assert (column["like100_per100"], column["like100_sd"], column["like100_cells"]) == like(whole)
        assert like(whole) != like(over_run(stored.n)) and like(whole)[0] > 0      # the cut decides the number


def test_a_frame_read_on_the_count_proseg_saves_is_compared_on_f(setup, tmp_path):
    """A frame the engine reads on f (the column's frame says count "sampled": proseg counted it for a pause, the hold,
    a finish or the end) has the run's column on f, the count proseg saves, its whole transcripts f as for any frame
    holding f; "averaged", or no count, reads e."""
    s = setup()
    read = counts_file.read(s["live"] / "frames" / "000002.counts.zst")
    q = np.random.default_rng(4).uniform(0.35, 0.95, read.nnz)
    counts_file.write(tmp_path / "f.counts.zst", iteration=2, ncells=read.ncells, ngenes=read.ngenes, indptr=read.indptr,
                      genes=read.genes, n=read.n, e=read.n * q, f=np.random.default_rng(5).binomial(read.n, q))
    stored = counts_file.read(tmp_path / "f.counts.zst")
    assert not np.array_equal(stored.f, np.round(stored.e))
    pair = (np.repeat(np.arange(NCELLS), np.diff(stored.indptr)), stored.genes)

    def over_run(values):
        dense = np.zeros((NCELLS, len(RUN_GENES)), np.float32)
        dense[pair] = values
        return dense
    encoder = W.load_encoder(s["model"] / "encoder.npz")
    spec = {**s["request"]["columns"][0]["frame"], "counts_file": str(tmp_path / "f.counts.zst")}
    seed = W.compare_seed(5, "run", "whole")
    for count, live in ((None, stored.e), ("averaged", stored.e), ("sampled", stored.f)):
        column = W.Worker(xp=np).compare_frame({**spec, **({"count": count} if count else {})}, encoder, seed)
        assert np.array_equal(dense_of(column["csr"]), model_rows(encoder, over_run(live))), count
        assert np.array_equal(dense_of(column["whole"]), model_rows(encoder, over_run(stored.f))), count


def test_the_points_split_one_shared_layout_back_into_the_columns(setup, tmp_path):
    s = setup()
    request, other = s["two"], s["other"]
    result = s["worker"].handle(request)["compare"]
    run, b = result["columns"]
    assert not run["sampled"] and run["umap_points"] == run["judged"] and b["umap_points"] == b["judged"] > 30
    assert result["points"] == run["umap_points"] + b["umap_points"] and result["bounds"] is not None
    kinds = result["kinds"]
    assert 1 <= len(kinds) <= W.KINDS_MAX and [k["index"] for k in kinds] == list(range(len(kinds)))
    assert [k["color"] for k in kinds] == list(W.KIND_PALETTE[:len(kinds)]) and all(k["name"] for k in kinds)
    layouts = []
    for i, column in enumerate(result["columns"]):
        path = tmp_path / "out" / f"{i}.bin"
        assert column["points_file"] == str(path) and path.stat().st_size == 10 * column["umap_points"]
        points = np.frombuffer(path.read_bytes(), W.POINT)
        assert (points["share"] < 255).all() and (points["kind"] < len(kinds)).all()
        assert np.isfinite(points["x"]).all() and np.isfinite(points["y"]).all()
        layouts.append(np.stack([points["x"], points["y"]], axis=1))
    # One layout for both: the run's own cells lie among its copy's when they are the same cells.
    assert np.abs(np.concatenate(layouts)).max() < 10 * W.RADIUS
    # The cap shares itself evenly; the share bytes are the picked cells' own.
    capped = s["worker"].handle({**request, "max_points": 40})["compare"]
    run_c, b_c = capped["columns"]
    assert run_c["umap_points"] + b_c["umap_points"] == 40 and b_c["sampled"]
    assert b_c["umap_points"] == W.even_quotas([run["judged"], b["judged"]], 40)[1]
    m, _, _, _, genes = W.read_counts_file(other)
    totals, share = model_scores(W.load_encoder(s["model"] / "encoder.npz"), dense_of(m), genes)
    judged = np.nonzero((totals >= JUDGE) & np.isfinite(share))[0]
    assert len(judged) == b["judged"]
    rows = judged[W.picked(len(judged), b_c["umap_points"], 0, 1)]
    points = np.frombuffer((tmp_path / "out" / "1.bin").read_bytes(), W.POINT)
    assert points["share"].tolist() == np.floor(share[rows] * 250).astype(int).tolist()


def model_scores(encoder, dense, genes):
    """(transcripts of the model's genes, neighbour share: NaN under fit_min) of a counts file's cells, its genes put
    in the model's order by name, scored by the model's calculator on dense rows."""
    model = np.zeros((len(dense), encoder["ngenes"]), np.float32)
    for g, name in enumerate(genes):
        if name in MODEL_GENES:
            model[:, MODEL_GENES.index(name)] = dense[:, g]
    totals = model.sum(axis=1)
    share = np.full(len(dense), np.nan, np.float32)
    rows = totals >= QC["fit_min"]
    share[rows] = W.neighbour_share(W.mixing_weights(np, np.log1p(model[rows]) @ encoder["W0"].T, encoder))
    return totals, share


def test_kinds_are_made_once_per_models_details_and_kept_beside_them(setup, monkeypatch):
    s = setup()
    worker, request, path = s["worker"], s["request"], s["model"] / W.KINDS_FILE
    made = []
    choose = W.choose_kinds
    monkeypatch.setattr(W, "choose_kinds", lambda *a, **k: (made.append(1), choose(*a, **k))[1])
    first = worker.handle(request)["compare"]
    assert first["kinds"] and first["kinds_reason"] is None and len(made) == 1
    stored = np.load(path)
    assert str(stored["format"]) == W.KINDS_FORMAT and int(stored["version"]) == W.KINDS_VERSION
    assert str(stored["deep_sha256"]) == W.file_sha256(s["model"] / "deep.npz")
    assert stored["names"].tolist() == [k["name"] for k in first["kinds"]] and int(stored["cells"]) == len(LEARNED)
    assert worker.handle(request)["compare"]["kinds"] == first["kinds"] and len(made) == 1    # kept in memory
    assert W.Worker(xp=np).handle(request)["compare"]["kinds"] == first["kinds"] and len(made) == 1   # ... and on disk
    # New details (learned again): the kinds are made again, also by a worker that kept the old ones. A broken file too.
    make_deep(s["model"], s["run"], seed=1)
    os.utime(s["model"] / "deep.npz", ns=(10 ** 18, 10 ** 18))         # a new version whatever the clock's grain
    worker.stat_ttl_s = 0.0                                            # its file stamps looked up again at once
    worker.handle(request)
    assert len(made) == 2 and str(np.load(path)["deep_sha256"]) == W.file_sha256(s["model"] / "deep.npz")
    path.write_bytes(b"not a zip")
    W.Worker(xp=np).handle(request)
    assert len(made) == 3 and str(np.load(path)["format"]) == W.KINDS_FORMAT
    # Nothing to make them from: no kinds, every point without one; the numbers stand.
    path.unlink()
    result = W.Worker(xp=np).handle({**s["two"], "kinds": {"path": str(path)}})["compare"]
    assert result["kinds"] is None and result["kinds_reason"] == "no_source" and result["columns"][0]["judged"] > 0
    points = np.frombuffer((s["model"].parent / "out" / "1.bin").read_bytes(), W.POINT)
    assert len(points) > 30 and (points["kind"] == W.NO_KIND).all() and (points["share"] < 255).all()
    # A model without details (learned before Go deeper): "needs_detail".
    (s["model"] / "deep.npz").unlink()
    result = W.Worker(xp=np).handle(request)["compare"]
    assert result["kinds"] is None and result["kinds_reason"] == "needs_detail" and result["columns"][0]["judged"] > 0
    # The learning cells must be the model's own.
    make_deep(s["model"], s["run"])
    other = write_counts(s["model"].parent / "strangers.npz", np.ones((20, len(MODEL_GENES))), MODEL_GENES,
                         np.zeros(20), np.zeros(20), ids=[f"z{i}" for i in range(20)])
    result = W.Worker(xp=np).handle({**request, "kinds": {"path": str(path), "counts": other}})["compare"]
    assert result["kinds"] is None and result["kinds_reason"] == "failed" and "cells were found" in result["kinds_error"]


def test_the_learning_table_is_read_as_the_trainer_reads_it(setup, monkeypatch):
    s = setup()
    m, ids, _, _, genes = W.read_counts_file(s["request"]["kinds"]["counts"])
    asked = []
    monkeypatch.setattr(W, "learning_counts", lambda zarr, key, wait_s=None: (asked.append((zarr, key, wait_s)),
                                                                              (m, ids, genes))[1])
    request = {**s["request"], "kinds": {"path": str(s["model"] / W.KINDS_FILE),
                                         "table": {"zarr": "/samples/s.zarr", "key": "xenium_table"}}}
    assert s["worker"].handle(request)["compare"]["kinds"]
    assert asked == [("/samples/s.zarr", "xenium_table", W.KINDS_LOCK_WAIT_S)]      # never waits long for a save


def test_the_engines_own_area_and_request_are_taken_as_they_come(setup, tmp_path):
    """engine/compare.py works the area out itself and hands each column's cells on it: thresholds nested, the
    frame's file as "file", each column's own points file, kinds.npz's path alone (the
    learning table: the one model.json names)."""
    s = setup()
    frame = dict(s["request"]["columns"][0]["frame"])
    frame["file"] = frame.pop("frame")
    frame.update(z_origin=None, z_spacing=None)                         # flat counting needs neither
    np.save(tmp_path / "cells-0.npy", np.array([0, 1, 4, 5, 8, 9], "<i4"))   # the run's left half (its cell numbers)
    np.save(tmp_path / "cells-1.npy", np.arange(0, 300, 2, dtype="<i4"))     # every other row of the counts file
    area = {"box": [100.0, 0.0, 200.0, 100.0], "size_um": [100.0, 100.0], "tiles": 1, "tile_um": 100.0}
    m, ids, _, _, genes = W.read_counts_file(s["request"]["kinds"]["counts"])
    (s["model"] / "model.json").write_text(json.dumps({"zarr": "/samples/s.zarr", "table": {"key": "xenium_table"}}))
    asked = []
    original = W.learning_counts
    W.learning_counts = lambda zarr, key, wait_s=None: (asked.append((zarr, key)), (m, ids, genes))[1]
    (tmp_path / "p").mkdir()                                            # the comparison's folder (the engine makes it)
    try:
        request = {"op": "compare", "run": "r1", "job": "abc", "stage": 2, "model": str(s["model"]), "model_id": "m",
                   "deep": str(s["model"] / "deep.npz"), "kinds": str(s["model"] / W.KINDS_FILE),
                   "thresholds": QC, "area": area, "max_points": 60000, "seed": 3,
                   "columns": [{"index": 0, "key": "run", "name": "This run", "cells": str(tmp_path / "cells-0.npy"),
                                "out_points": str(tmp_path / "p" / "points-2-0.bin"), "frame": frame},
                               {"index": 1, "key": "xenium", "name": "Xenium cells", "counts": s["other"],
                                "cells": str(tmp_path / "cells-1.npy"),
                                "out_points": str(tmp_path / "p" / "points-2-1.bin")}]}
        reply = W.answer(s["worker"], json.dumps(request))
    finally:
        W.learning_counts = original
    assert reply["ok"], reply
    result = reply["compare"]
    run, xenium = result["columns"]
    assert result["area"] == area and run["cells_in_area"] == 6 and xenium["cells_in_area"] == 150
    assert run["tiles"] is None and xenium["judged"] <= 150 and xenium["umap_points"] == xenium["judged"]
    assert asked == [("/samples/s.zarr", "xenium_table")] and result["kinds"]
    path = tmp_path / "p" / "points-2-1.bin"
    assert xenium["points_file"] == str(path) and path.stat().st_size == 10 * xenium["umap_points"]
    # The cells of the frame are the run's own (the left half): the same numbers as the worker's own area gives there.
    own = s["worker"].handle({**s["request"], "area": [X0, Y0, X0 + 16, Y0 + 24]})["compare"]["columns"][0]
    assert {k: run[k] for k in ("judged", "mixed_per100", "histogram", "median_tx")} == \
        {k: own[k] for k in ("judged", "mixed_per100", "histogram", "median_tx")}
    np.save(tmp_path / "cells-1.npy", np.array([300], "<i4"))          # a row the counts file does not have
    reply = W.answer(s["worker"], json.dumps(request))
    assert reply["ok"] and "not among its 300 cells" in reply["compare"]["columns"][1]["error"]


class LearnedCsr:
    """What scipy's CSR gives learning_counts (this environment has no scipy)."""

    def __init__(self, m: W.Csr):
        self.data, self.indices, self.indptr, self.shape = m.data, m.indices, m.indptr, m.shape

    def tocsr(self):
        return self


def test_kinds_never_wait_long_for_a_save_into_the_sample(setup, monkeypatch, tmp_path):
    """The learning table is read only while no save into the sample writes it, and a
    Finish & save holds the sample's lock for its whole run, so making the kinds froze the GPU worker with it (the engine
    stops a worker silent for 300 s). Now the kinds wait KINDS_LOCK_WAIT_S at most: "busy", the numbers and points without
    kinds, nothing made nor kept; the next Compare makes them."""
    from sopa_companion.analysis import qc_trainer as trainer
    from sopa_companion.zarr_bridge import helper
    s = setup()
    sample = (tmp_path / "s.zarr").resolve()
    sample.mkdir()
    path = s["model"] / W.KINDS_FILE
    m, ids, _, _, genes = W.read_counts_file(s["request"]["kinds"]["counts"])
    read = []
    monkeypatch.setattr(trainer, "read_table", lambda zarr, key: (read.append(key), (
        LearnedCsr(m), {"cell_id": np.array(ids)}, np.array(genes), None, None))[1])
    monkeypatch.setattr(W, "KINDS_LOCK_WAIT_S", 0.3)
    request = {**s["two"], "kinds": {"path": str(path), "table": {"zarr": str(sample), "key": "xenium_table"}}}
    answered = []
    with helper._store_lock(sample):                                 # a save into the sample, under way
        worker = threading.Thread(target=lambda: answered.append(s["worker"].handle(request)), daemon=True)
        worker.start()
        worker.join(60)
        assert not worker.is_alive(), "the worker waited for the save"
    result = answered[0]["compare"]
    assert result["kinds"] is None and result["kinds_reason"] == "busy" and "save" in result["kinds_error"] and not read
    run, other = result["columns"]
    assert run["judged"] > 0 and other["umap_points"] > 0 and other["points_file"]   # coloured by mixing meanwhile
    assert (np.frombuffer(Path(other["points_file"]).read_bytes(), W.POINT)["kind"] == W.NO_KIND).all()
    assert not path.exists() and not s["worker"].kind_sets
    # The save done: the next Compare reads the table and makes them.
    result = s["worker"].handle(request)["compare"]
    assert result["kinds"] and result["kinds_reason"] is None and read == ["xenium_table"] and path.exists()


def test_points_never_bring_a_deleted_runs_folder_back(setup, tmp_path):
    """A run deleted while its comparison was being scored got its folder back from the
    points files. The worker writes a column's points only into a folder that exists: that column keeps its numbers and
    has no points; the others' are written."""
    s = setup()
    gone = tmp_path / "runs" / "r1" / "compare" / "abc"                 # deleted with its run
    kept = tmp_path / "kept"
    kept.mkdir()
    run_column, other_column = s["two"]["columns"]
    request = {**s["two"], "out_dir": None, "columns": [{**run_column, "out_points": str(gone / "points-1-0.bin")},
                                                        {**other_column, "out_points": str(kept / "points-1-1.bin")}]}
    reply = s["worker"].handle(request)
    assert reply["ok"], reply
    run, other = reply["compare"]["columns"]
    assert run["state"] == "ready" and run["judged"] > 0 and run["points_file"] is None and run["umap_points"] == 0
    assert other["points_file"] == str(kept / "points-1-1.bin") and other["umap_points"] > 0
    assert reply["compare"]["points"] == other["umap_points"] and not (tmp_path / "runs").exists()
    # The worker's own form: out_dir itself is made, never its parents.
    reply = s["worker"].handle({**s["two"], "out_dir": str(tmp_path / "lost" / "out")})
    assert reply["ok"] and not (tmp_path / "lost").exists()
    assert all(c["points_file"] is None and c["state"] == "ready" for c in reply["compare"]["columns"])


def test_one_column_failing_never_costs_the_others(setup):
    s = setup()
    request = {**s["request"], "columns": [*s["request"]["columns"], {"key": "gone", "name": "G", "counts": "/nope.npz"},
                                           {"key": "empty", "name": "E"}]}
    reply = s["worker"].handle(request)
    assert reply["ok"]
    run, gone, empty = reply["compare"]["columns"]
    assert run["state"] == "ready" and run["judged"] > 0
    assert gone["state"] == "failed" and "FileNotFoundError" in gone["error"] and gone["umap_points"] == 0
    assert empty["state"] == "failed" and "counts file" in empty["error"] and gone["points_file"] is None
    reply = s["worker"].handle({**request, "columns": request["columns"][1:]})
    assert not reply["ok"] and "no column could be counted" in reply["errors"]["compare"] and "compare" not in reply
    reply = W.answer(s["worker"], json.dumps({**s["request"], "columns": []}))
    assert not reply["ok"] and "no columns" in reply["error"]


def test_too_few_cells_give_numbers_without_a_picture(setup):
    s = setup()
    result = s["worker"].handle({**s["request"], "area": [X0, Y0, X0 + 16, Y0 + 24]})["compare"]
    column = result["columns"][0]
    assert result["layout"] is None and result["layout_error"] == "too_few_cells" and result["points"] == 0
    assert column["umap_points"] == 0 and column["points_file"] is None and column["judged"] > 0
    assert column["cells_in_area"] == 6                                # the left half of the run: 6 of its 11 cells
    assert result["area"]["box"] == [X0, Y0, X0 + 16, Y0 + 24]


def test_compare_never_reads_nor_changes_the_runs_live_clusters_layout_or_colours(setup):
    s = setup()
    worker = s["worker"]
    frame = s["request"]["columns"][0]["frame"]
    run = worker.state(frame)
    live = {"embedding": np.full((NCELLS, 2), 3.0, np.float32), "clusters": np.arange(NCELLS, dtype=np.int32),
            "colours": {0: 4, 1: 7}, "freed": {9: 1.5}, "updates": 41, "next_cluster": 12}
    for name, value in live.items():
        setattr(run, name, value)
    copies = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in live.items()}
    assert worker.handle(s["request"])["ok"]
    for name, value in copies.items():
        got = getattr(run, name)
        assert got is live[name] and (np.array_equal(got, value) if isinstance(value, np.ndarray) else got == value)


# ---- the engine's queue (engine/analysis.py) --------------------------------------------------------------------------
def test_compare_goes_after_explicit_requests_and_answers_its_future(tmp_path):
    run = FakeRun(cells=40070, frames=3)
    m = manager(tmp_path, run)
    first = m.submit_compare({"run": "r1", "columns": [{"key": "run"}]})
    second = m.submit_compare({"run": "r1", "columns": [{"key": "a"}]})
    m.configure("r1", seq=1)                                           # the viewer scrubbed: that goes first
    job, _ = m._next_job()
    assert job == ("r1", 1) and job.kind == "request"
    job, _ = m._next_job()
    assert job.ops == ("compare",) and job.kind == "compare" and job.compare[0]["columns"] == [{"key": "run"}]
    assert first.running()
    calls = []
    reply = {"ok": True, "run": "r1", "seq": None, "compare": {"columns": []}}
    m._call = lambda request: (calls.append(request), reply)[1]
    assert m._compare(job) is None and calls[-1]["op"] == "compare" and first.result(0) is reply
    assert second.cancel()                                             # the view closed before it started: dropped
    job, _ = m._next_job()
    assert job == ("r1", 2) and job.kind == "live"


def test_a_compare_without_a_reply_raises_in_its_future_and_close_cancels_the_rest(tmp_path):
    m = manager(tmp_path, FakeRun(cells=10, frames=1, state="done"))
    future = m.submit_compare({"run": "r1", "columns": []})
    job, _ = m._next_job()

    def stopped(request):
        raise AnalysisError("the GPU worker stopped")
    m._call = stopped
    assert m._compare(job) is None
    with pytest.raises(AnalysisError, match="stopped"):
        future.result(0)
    assert any("Compare failed: the GPU worker stopped" in text for _, _, text in m.log_info()["events"])
    m._call = lambda request: {"ok": False, "run": "r1", "seq": None, "errors": {"compare": "ValueError: x"},
                               "error": "ValueError: x"}
    failed = m.submit_compare({"run": "r1", "columns": []})
    m._compare(m._next_job()[0])
    assert failed.result(0)["errors"] == {"compare": "ValueError: x"}
    queued = m.submit_compare({"run": "r1", "columns": []})
    m.close()
    assert queued.cancelled()
    with pytest.raises(AnalysisError):
        m.submit_compare({"run": "r1"})
    with pytest.raises(AnalysisError) as refused:
        manager(tmp_path, FakeRun(), python="").submit_compare({"run": "r1"})
    assert refused.value.status == 503


def test_a_run_that_is_not_live_leaves_the_worker_after_its_comparison(tmp_path):
    """Compare's op loads the run's transcripts on GPU 1, and an ended run kept them for
    good (only Go deeper set an ended run's release again). Now a run that is not live (ended, failed or deleted) leaves
    the worker COMPARE_HOLD_S after its comparison's last op, the way Go deeper's runs go; a live run never does."""
    from sopa_companion.engine.analysis import COMPARE_HOLD_S
    from test_analysis import Provider, finish

    def ended(name):
        run = FakeRun(name, cells=3000, frames=3, state="done", directory=tmp_path / name)
        run.info.update(prior={"file": "prior.bin.zst"})
        run.live.mkdir(parents=True)
        for file in ("prior.bin.zst", "transcripts.bin.gz"):
            (run.live / file).write_bytes(b"")
        return run

    def compare(run_id, fail=False, run=True):
        def call(request):
            if fail:
                raise AnalysisError("the GPU worker stopped")
            return {"ok": True, "run": run_id, "seq": None, "compare": {"columns": []}}
        m._call = call
        m.submit_compare({"run": run_id, "columns": [{"key": "run", "frame": {"run": run_id}},
                                                     {"key": "xenium", "counts": "xenium.npz"}]})
        if run:
            job, _ = m._next_job()
            assert job.ops == ("compare",)
            m._compare(job)

    now = [100.0]
    m = manager(tmp_path, ended("old"), ended("checked"), FakeRun("live", cells=3000, frames=3, state="running"))
    m.clock = lambda: now[0]
    m.configure("live", enabled=False)                                  # no passes of its own here
    m.attach_qc(Provider())
    m.configure_qc("checked", enabled=True)                             # scored, then released: the scheduler's rule
    while (job := m._next_job()[0]) is not None:
        finish(m, job)
    assert m.to_forget == {"checked"} and m.qc_runs["checked"].released
    m.to_forget.clear()
    # An ended run without the check: held for the comparison's next op, then let go.
    compare("old")
    assert m.compared == {"old": 100.0 + COMPARE_HOLD_S} and m._next_job() == (None, COMPARE_HOLD_S) and not m.to_forget
    now[0] += 30
    compare("old", fail=True)                                           # its next op (one that failed loads it too)
    now[0] += COMPARE_HOLD_S - 1
    assert m._next_job() == (None, 1.0) and not m.to_forget
    now[0] += 1
    assert m._next_job() == (None, None) and m.to_forget == {"old"} and not m.compared
    m.to_forget.clear()
    # The scored run: loaded again by Compare, released again by the scheduler's own rule once its hold is over.
    compare("checked")
    assert m._next_job() == (None, COMPARE_HOLD_S) and not m.to_forget and m.qc_runs["checked"].released
    now[0] += COMPARE_HOLD_S
    assert m._next_job() == (None, None) and m.to_forget == {"checked"} and m.qc_runs["checked"].released
    m.to_forget.clear()
    # Its next op still queued when the hold ends: kept for it, and held again after it.
    compare("old")
    compare("old", run=False)
    now[0] += COMPARE_HOLD_S
    m._release_compared()
    assert not m.to_forget and not m.compared
    m._compare(m._next_job()[0])
    assert m.compared == {"old": now[0] + COMPARE_HOLD_S}
    # Something asked of it meanwhile (the viewer stepped to a frame): it stays.
    m.configure("old", seq=0)
    now[0] += COMPARE_HOLD_S
    m._release_compared()
    assert not m.to_forget
    # A live run is never let go; a deleted one goes with its settings.
    m.requested.clear()
    compare("live")
    compare("gone")
    assert set(m.compared) == {"gone"}
    now[0] += COMPARE_HOLD_S
    assert m._next_job() == (None, None) and m.to_forget == {"gone"}


def test_the_runs_column_is_the_frame_a_pass_would_send(tmp_path):
    from test_analysis import disk_run, stub_worker
    run = disk_run(tmp_path / "run")
    m = manager(tmp_path, run)
    calls = []
    stub_worker(m, calls)
    m._analyse("r1", 1)
    sent = calls[-1]
    column = m.compare_frame("r1", 1)
    assert column["seq"] == 1 and column["iter"] == 1 and column["run"] == "r1"
    assert {k: column[k] for k in column if k not in ("seq", "iter", "run", "cells_file")} == \
        {k: sent[k] for k in column if k not in ("seq", "iter", "run", "cells_file")}
    assert column["cells_file"] == str(run.live / "frames/000001.cells.zst")
    assert m.compare_frame("r1")["seq"] == len(run.frames) - 1          # the newest by default
    with pytest.raises(AnalysisError) as unknown:
        m.compare_frame("r1", 99)
    assert unknown.value.status == 404
    # ... and the worker reads it: the frame's proseg count (the engine names its counts file).
    column.setdefault("counts_file", str(run.live / "frames" / "000001.counts.zst"))
    (run.live / "cells.txt").write_text("".join(c + "\n" for c in RUN_IDS))
    model = make_deep(tmp_path / "model", run.dir)
    result = W.Worker(xp=np).handle({"op": "compare", "run": "r1", "model": str(model), "model_id": "m", **QC,
                                     "columns": [{"key": "run", "name": "This run", "frame": column}]})
    assert result["ok"] and result["compare"]["columns"][0]["cells"] == NCELLS, result


def test_the_scheduler_thread_runs_compare_ops(tmp_path):
    m = manager(tmp_path, FakeRun(cells=10, frames=1, state="done"))
    seen = threading.Event()

    def call(request):
        seen.set()
        return {"ok": True, "run": request["run"], "seq": None, "compare": {"columns": len(request["columns"])}}
    m._call = call
    m.thread.start()
    try:
        future = m.submit_compare({"run": "r1", "columns": [{"key": "run"}, {"key": "a"}]})
        assert future.result(timeout=5)["compare"] == {"columns": 2} and seen.is_set()
    finally:
        m.close()
        m.thread.join(5)
    assert not m.thread.is_alive()


def test_the_compare_managers_blocking_call_reads_the_reply_at_its_top(tmp_path):
    """engine/compare.py calls `analysis.compare(request)` and reads ok, error, columns and kinds at the reply's top."""
    m = manager(tmp_path, FakeRun(cells=10, frames=1, state="done"))
    block, entered = threading.Event(), threading.Event()

    def call(request):
        entered.set()
        block.wait(5)
        return {"ok": True, "run": request["run"], "seq": None,
                "compare": {"columns": [{"key": "run", "judged": 7, "error": None}], "kinds": None}}
    m._call = call
    m.thread.start()
    try:
        block.set()
        reply = m.compare({"op": "compare", "run": "r1", "columns": [{"key": "run"}]}, timeout=5)
        assert reply["ok"] and reply["columns"] == [{"key": "run", "judged": 7, "error": None}] and reply["kinds"] is None
        assert reply["compare"]["columns"] == reply["columns"] and reply["seq"] is None
        block.clear()                                                  # the worker busy: the call gives up in time
        entered.clear()
        first = threading.Thread(target=lambda: m.compare({"run": "r1", "columns": []}, timeout=5))
        first.start()
        assert entered.wait(5)                                         # the worker has taken the first one
        with pytest.raises(AnalysisError, match="in time"):
            m.compare({"run": "r1", "columns": []}, timeout=0.2)
        assert not m.compare_queue                                     # the waiting one was dropped
        block.set()
        first.join(5)
    finally:
        m.close()
        m.thread.join(5)
    with pytest.raises(AnalysisError) as closed:
        m.compare({"run": "r1"})
    assert closed.value.status == 503


def test_each_its_own_lines_a_turned_and_flipped_copy_back_up_without_reshaping_it():
    rng = np.random.default_rng(3)
    ids = np.arange(300) % 6
    centres = np.array([[0, 0], [6, 1], [-4, 5], [2, -7], [-8, -3], [9, 8]], np.float32)
    ref = (centres[ids] + rng.normal(scale=0.01, size=(300, 2))).astype(np.float32)   # six tight groups, spread out
    turn = np.array([[np.cos(1.1), -np.sin(1.1)], [np.sin(1.1), np.cos(1.1)]]) @ np.diag([1.0, -1.0])   # a turn and a flip
    moved = (ref @ turn.T + [4.0, -7.0]).astype(np.float32)
    src, dst, w = W.landmarks(moved, ids, ref, ids)
    assert len(w) == 6 and (w == 50).all()
    back = W.align_onto(moved, src, dst, w)
    assert np.abs(back - ref).max() < 0.05, "turned, flipped and moved back (the groups' medians)"
    # never scaled or reshaped: every distance kept, also when the groups sit in another arrangement than the
    # yardstick's (a fitted scale would shrink the map towards a dot)
    for target in (dst, dst[::-1]):
        lined = W.align_onto(moved, src, target, w)
        for a, b in ((0, 1), (2, 3), (10, 200)):
            assert abs(np.linalg.norm(moved[a] - moved[b]) - np.linalg.norm(lined[a] - lined[b])) < 1e-3
    # too few points in a group: not a landmark
    assert len(W.landmarks(moved[:30], ids[:30], ref[:30], ids[:30])[2]) == 0


def test_each_its_own_matches_a_columns_clusters_to_the_runs_by_their_profiles():
    ref = np.array([[5, 0, 0, 1], [0, 5, 0, 1], [0, 0, 5, 1]], np.float64)
    own = np.array([[0, 0, 4.5, 1.2], [4.8, 0.2, 0, 1], [1, 1, 1, 1]], np.float64)   # the last like none of them
    assert W.match_profiles(ref, own) == {0: 2, 1: 0}
    assert W.match_profiles(ref, own, minimum=0.9999) == {}
    ids, prof = W.cluster_profiles(np.array([[1.0, 0], [3.0, 0], [0, 2.0]]), np.array([4, 4, 7]))
    assert ids.tolist() == [4, 7] and prof.tolist() == [[2.0, 0.0], [0.0, 2.0]]


def test_each_its_own_writes_every_columns_own_map_and_clusters_beside_its_shared_one(setup, tmp_path):
    s = setup()
    result = s["worker"].handle(s["two"])["compare"]
    assert result["own_layout"] == "pca" and result["own_error"] is None and result["own_bounds"] is not None
    mapped = []
    for i, column in enumerate(result["columns"]):
        shared, own, clusters = (tmp_path / "out" / f"{i}.bin", tmp_path / "out" / f"{i}.own.bin", tmp_path / "out" / f"{i}.clusters.bin")
        if column["umap_points"] < 30:          # too few for a map of its own: none, and nothing left from before
            assert column["own_points_file"] is None and not own.exists() and not clusters.exists()
            continue
        mapped.append(column)
        assert column["own_points_file"] == str(own) and column["clusters_file"] == str(clusters)
        a, b = np.frombuffer(shared.read_bytes(), W.POINT), np.frombuffer(own.read_bytes(), W.POINT)
        assert len(a) == len(b) == column["umap_points"]
        assert (a["kind"] == b["kind"]).all() and (a["share"] == b["share"]).all(), "the same points, in the same order"
        ids = np.frombuffer(clusters.read_bytes(), W.CLUSTER)
        assert len(ids) == len(a) and (ids < W.NO_CLUSTER).all() and column["clusters"] == len(np.unique(ids))
    assert mapped and mapped[0]["clusters_matched"] == mapped[0]["clusters"], "the reference's own clusters (the run's, else the first with a map)"
