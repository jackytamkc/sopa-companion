"""Fixes on the engine side of Compare, without the sopa
environment: the counting rule in the counts' cache, a table's own counts kept by the table and its version, the
listing's whose count each column reads, the helper's failures never saying its progress lines
and an element whose folder does not answer never taken for a version. ZarrSamples' sample and
helper are stood in for; tests/test_zarr.py runs the real helper on synthetic samples."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import sys

import numpy as np
import pytest

from sopa_companion.engine import zarr_samples as Z
from sopa_companion.engine.samples import SampleError
from sopa_companion.engine.vectors import VERSION as VECTORS_VERSION
from sopa_companion.zarr_bridge.helper import COUNT_RULE, COUNT_VERSION, PROSEG_TABLE_KEYS, count_rule

STAMP, OLD = "0123456789ab", "ba9876543210"
V = f"v{COUNT_VERSION}"                 # counts made the current way (the helper's COUNT_VERSION)


def samples(tmp_path, python: str = "") -> Z.ZarrSamples:
    return Z.ZarrSamples({"zarr": {"python": python}, "proseg": {"max_region_um": 1000.0},
                          "paths": {"cache_dir": str(tmp_path / "cache"), "data_dir": str(tmp_path / "var")}})


def reads(*stamps):
    """An element_stamp that answers these in turn (None: the folder did not answer), then the last one for good."""
    order = list(stamps)
    return lambda sid, key: order.pop(0) if len(order) > 1 else order[0]


def stood_in(tmp_path, monkeypatch):
    """ZarrSamples with one sample "s" holding the segmentation "cells" (its folder's stamp STAMP), and a helper whose
    `count` writes the counts file it is asked for: (samples, the helper's count arguments, the counts folder)."""
    monkeypatch.setattr(Z, "STAMP_RETRY_S", 0.0)
    zarr = samples(tmp_path)
    shape = {"key": "cells", "count": 4, "geometry": {"Polygon": 4}, "extent": [0, 0, 40, 40]}
    zarr.detail = lambda sid: {"points_key": "transcripts", "shapes": [shape]}
    zarr.path = lambda sid: str(tmp_path / "s.zarr")
    zarr._made_by = lambda sid: None
    zarr.element_stamp = reads(STAMP)
    calls = []

    def helper(command, args, **kwargs):
        assert command == "count"
        calls.append(args)
        Path(args["out"]).parent.mkdir(parents=True, exist_ok=True)
        rule = f"table:{args['table']}" if "table" in args else count_rule(args["min_qv"], args["exclude"])
        np.savez(args["out"], rule=np.array(rule), count=np.array(args.get("by", "outlines")))
        return {"rule": rule}
    zarr.helper = helper
    return zarr, calls, tmp_path / "cache" / "zarr" / "s" / "counts"


# ---- each counting rule its own counts -------------------------------------------------------------------------------
def test_counts_are_kept_per_counting_rule(tmp_path, monkeypatch):
    zarr, calls, folder = stood_in(tmp_path, monkeypatch)
    assert COUNT_RULE == count_rule() == count_rule(20.0, None) == f"qv20-{V}"   # no count made before is reused
    whole = zarr.counts("s", "cells")
    assert whole.name == f"cells-qv20-{V}-{STAMP}.npz" and (calls[-1]["min_qv"], calls[-1]["exclude"]) == (20.0, None)
    strict = zarr.counts("s", "cells", rule={"min_qv": 30})
    assert strict.name == f"cells-qv30-{V}-{STAMP}.npz" and calls[-1]["min_qv"] == 30.0 and len(calls) == 2
    pattern = "^(Deprecated|NegControl|Unassigned|Intergenic|MALAT1)"
    named = zarr.counts("s", "cells", rule={"min_qv": 20, "exclude": pattern})
    assert named.name == f"cells-{count_rule(20, pattern)}-{STAMP}.npz" and calls[-1]["exclude"] == pattern
    assert count_rule(20, pattern).startswith("qv20-x") and count_rule(20, pattern).endswith(f"-{V}")
    assert len({whole, strict, named}) == 3 and whole.exists() and strict.exists()      # one never prunes another
    assert zarr.counts("s", "cells", rule={"min_qv": 30.0}) == strict and zarr.counts("s", "cells", rule={}) == whole
    assert len(calls) == 3
    # Boxes: a rule's own cached boxes or whole only, never another rule's.
    box = zarr.counts("s", "cells", [0, 0, 10, 10], rule={"min_qv": 25})
    assert box.name == f"cells-qv25-{V}-{STAMP}-0_0_10_10.npz" and calls[-1]["box"] == [0.0, 0.0, 10.0, 10.0]
    assert zarr.counts("s", "cells", [2, 2, 8, 8], rule={"min_qv": 25}, cached_only=True) == box
    assert zarr.counts("s", "cells", [2, 2, 8, 8], rule={"min_qv": 26}, cached_only=True) is None
    assert zarr.counts("s", "cells", [2, 2, 8, 8], rule={"min_qv": 30}, cached_only=True) == strict   # its whole
    # What can never serve again goes: another stamp's (also an unreadable "0" an earlier version kept) and every count
    # made the old way (v1: a box count then joined only its own outlines); another element's stays.
    stale = [folder / f"cells-qv20-v1-{STAMP}.npz", folder / f"cells-qv20-v1-{STAMP}-0_0_5_5.npz",
             folder / f"cells-qv20-v2-{OLD}.npz", folder / "cells-qv20-v2-0.npz",
             folder / f"cells-qv30-x0123abcd-v2-{OLD}-1_1_2_2.npz"]
    for path in stale + [folder / f"nuclei-qv20-v1-{STAMP}.npz"]:
        path.write_bytes(b"x")
    zarr.counts("s", "cells", rule={"min_qv": 35})
    assert not any(path.exists() for path in stale) and (folder / f"nuclei-qv20-v1-{STAMP}.npz").exists()
    assert all(path.exists() for path in (whole, strict, named, box))


def test_a_rule_that_is_not_one_is_refused_in_plain_words(tmp_path, monkeypatch):
    zarr, calls, _ = stood_in(tmp_path, monkeypatch)
    for bad, words in (({"min_qv": -1}, "0 or more"), ({"min_qv": "high"}, "0 or more"), ({"min_qv": True}, "0 or more"),
                       ({"min_qv": float("nan")}, "0 or more"), ({"exclude": "(MALAT1"}, "not a valid pattern"),
                       ({"exclude": 3}, "text pattern"), ("qv20", "counting rule")):
        with pytest.raises(SampleError, match=words) as refused:
            zarr.counts("s", "cells", rule=bad)
        assert refused.value.status == 422
    assert not calls
    assert Z.ZarrSamples.count_rule({"min_qv": 30, "exclude": None}) == (30.0, None, f"qv30-{V}")
    assert Z.ZarrSamples.count_rule(None) == (20.0, None, f"qv20-{V}")


# ---- a table's own count: kept by the table and its version ----------------------------------------------------------
def test_a_tables_own_counts_are_kept_by_the_table_and_its_version(tmp_path, monkeypatch):
    """An element whose table holds its own count (a saved proseg result's X, Xenium's cells') is read from that table:
    the helper is asked for the table and whose count it is, never the run's input rule (no transcript is counted), and
    the file is named by the count, the table and its version (t<digest>-v<n>, never a rule's name), so another rule
    reads the same file, a table written again is read again (its old counts go), and the element's outline counts
    stay. A count that is not one is refused before any helper runs; a table that does not answer is never a version;
    one written again while read is never kept."""
    zarr, calls, folder = stood_in(tmp_path, monkeypatch)
    zarr.table_stamp = reads("aaaaaaaaaaaa")
    own = zarr.counts("s", "cells", count_by="xenium", table="xenium_table")
    assert re.fullmatch(rf"cells-t[0-9a-f]{{8}}-{V}-{STAMP}\.npz", own.name) and Z._count_files("cells").fullmatch(own.name)
    assert {k: v for k, v in calls[-1].items() if k not in ("path", "out")} == {
        "key": "cells", "points_key": "transcripts", "box": None, "table": "xenium_table", "by": "xenium"}
    assert str(np.load(own)["count"]) == "xenium"
    assert zarr.counts("s", "cells", rule={"min_qv": 30}, count_by="xenium", table="xenium_table") == own
    assert zarr.counts("s", "cells", [0, 0, 10, 10], cached_only=True, count_by="xenium", table="xenium_table") == own
    assert len(calls) == 1
    other = zarr.counts("s", "cells", count_by="proseg", table="proseg_table")      # another table: its own file
    assert other != own and calls[-1]["table"] == "proseg_table" and not own.exists()   # an element reads one table
    outlines = zarr.counts("s", "cells")
    assert outlines.name == f"cells-qv20-{V}-{STAMP}.npz" and other.exists() and "table" not in calls[-1]
    box = zarr.counts("s", "cells", [0, 0, 10, 10], count_by="proseg", table="proseg_table")
    assert box == other and len(calls) == 3                                          # the whole serves any box
    # Written again: read again, the table's old counts go, the outline counts stay.
    zarr.table_stamp = reads("bbbbbbbbbbbb")
    newer = zarr.counts("s", "cells", count_by="proseg", table="proseg_table")
    assert newer != other and not other.exists() and outlines.exists() and len(calls) == 4
    # Not a count: refused in plain words (422), no helper asked.
    for count_by, table in (("xenium", None), ("nope", "xenium_table"), ("outlines", "xenium_table"),
                            ("proseg", "../tables"), ("proseg", 3)):
        with pytest.raises(SampleError, match="its own table") as refused:
            zarr.counts("s", "cells", count_by=count_by, table=table)
        assert refused.value.status == 422
    assert len(calls) == 4
    # A table whose folder does not answer, twice: refused (503), nothing kept under no version.
    zarr.table_stamp = reads(None)
    with pytest.raises(SampleError, match="could not be read just now") as refused:
        zarr.counts("s", "cells", count_by="xenium", table="xenium_table")
    assert refused.value.status == 503 and len(calls) == 4
    # Written again while it was read: never kept under the version read before.
    zarr.table_stamp = reads("cccccccccccc", "dddddddddddd")
    with pytest.raises(SampleError, match="'xenium_table' was written again") as refused:
        zarr.counts("s", "cells", count_by="xenium", table="xenium_table")
    assert refused.value.status == 409 and len(calls) == 5
    assert sorted(p.name for p in folder.iterdir()) == sorted([outlines.name, newer.name])


def test_a_tables_version_follows_its_counts(tmp_path):
    """table_stamp: a table written again (a new folder), or its counts alone written again (an array of X written
    anew under an unchanged X folder: seen one level inside X only; X's arrays replaced), is another version; the same
    table read twice is the same one; a table not there, or not readable now, has none."""
    zarr = samples(tmp_path)
    zarr.path = lambda sid: str(tmp_path / "s.zarr")
    table = tmp_path / "s.zarr" / "tables" / "t1"
    (table / "X" / "data").mkdir(parents=True)
    (table / "zarr.json").write_text("{}")
    first = zarr.table_stamp("s", "t1")
    assert re.fullmatch(r"[0-9a-f]{12}", first) and zarr.table_stamp("s", "t1") == first
    x, top = zarr._identity((table / "X").stat()), zarr._identity(table.stat())
    data = table / "X" / "data"
    (data / "zarr.json").write_text("{}")                                   # an array of X written anew in place
    os.utime(data, ns=(data.stat().st_atime_ns, data.stat().st_mtime_ns + 10**9))   # (whatever the clock's step)
    assert zarr._identity((table / "X").stat()) == x and zarr._identity(table.stat()) == top   # X itself unchanged
    second = zarr.table_stamp("s", "t1")
    assert second != first
    data.rename(table / "X" / "indices")                                    # its arrays replaced
    third = zarr.table_stamp("s", "t1")
    assert third not in (first, second)
    (table / "obs").mkdir()                                                 # another part of it
    assert zarr.table_stamp("s", "t1") not in (first, second, third)
    assert zarr.table_stamp("s", "nope") is None


def test_the_listing_says_whose_count_compare_reads(tmp_path, monkeypatch):
    """Each listed segmentation says whose count Compare reads for it (count_source): proseg's own when a table
    annotating it carries proseg's run record (sopa's reading of proseg's output, or a proseg save of this app),
    Xenium's for Xenium's cells, else the transcripts inside its outlines (Cellpose, an import counted by its
    outlines elsewhere, a table with no record). A listing kept from before these were said is read again."""
    zarr, calls, folder = stood_in(tmp_path, monkeypatch)
    shapes = [{"key": key, "count": 4, "geometry": {"Polygon": 4}, "extent": [0, 0, 40, 40]}
              for key in ("cell_boundaries", "cellpose_boundaries_1", "proseg_boundaries", "sopa_proseg", "x_cells")]
    zarr.detail = lambda sid: {"points_key": "transcripts", "shapes": shapes}
    made_by = {"shapes": [{"key": "cellpose_boundaries_1", "app": {"kind": "stain", "method": "cellpose"}}],
               "tables": [{"key": "table", "region": ["cell_boundaries"], "proseg": False, "proseg_count": False,
                           "methods": {"cell_boundaries": {"Imported Cell Segmentation": 4}}},
                          {"key": "xenium_table", "region": ["x_cells"], "proseg": False, "proseg_count": False,
                           "methods": {"x_cells": {"Segmented by interior stain (18S)": 4}}},
                          {"key": "saved", "region": ["proseg_boundaries"], "proseg": True, "proseg_count": True,
                           "methods": None},
                          {"key": "sopa_table", "region": ["sopa_proseg"], "proseg": False, "proseg_count": True,
                           "methods": None}]}
    zarr._made_by = lambda sid: made_by
    listed = {e["key"]: (e["source"], e["count_by"], e["count_table"]) for e in zarr.segmentations("s")}
    assert listed == {"x_cells": ("xenium", "xenium", "xenium_table"),
                      "cell_boundaries": ("imported", "outlines", None),
                      "proseg_boundaries": ("proseg", "proseg", "saved"),
                      "cellpose_boundaries_1": ("cellpose", "outlines", None),
                      "sopa_proseg": ("other", "proseg", "sopa_table")}
    assert Z.count_source("x_cells", None) == {"count_by": "outlines", "count_table": None}
    # The listing the helper gave before (no proseg_count, another stamp) is asked of the helper again.
    del zarr._made_by
    sample = tmp_path / "s.zarr"
    (sample / "tables").mkdir(parents=True)
    cached = tmp_path / "cache" / "zarr" / "s" / "segmentations.json"
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps({**made_by, "stamp": [zarr._mtime(str(sample)), list(Z.PROVENANCE_KEYS)]}))
    asked = []
    zarr.helper = lambda command, args, **kwargs: asked.append(command) or made_by
    assert zarr._made_by("s") == made_by and asked == ["segmentations"]
    assert json.loads(cached.read_text())["stamp"] == [zarr._mtime(str(sample)), list(Z.PROVENANCE_KEYS),
                                                       list(PROSEG_TABLE_KEYS)]
    assert zarr._made_by("s")["tables"] == made_by["tables"] and asked == ["segmentations"]     # kept now
    assert PROSEG_TABLE_KEYS == ("proseg_run", "sopa_companion", "proseg_live")


# ---- an element whose folder does not answer is never a version --------------------------------------------------------
def test_an_unreadable_element_is_never_a_version(tmp_path, monkeypatch):
    zarr, calls, folder = stood_in(tmp_path, monkeypatch)
    whole = zarr.counts("s", "cells")
    outlines = tmp_path / "cache" / "zarr" / "s" / "segmentations" / f"cells-{STAMP}.v{VECTORS_VERSION}.bin.zst"
    outlines.parent.mkdir(parents=True)
    outlines.write_bytes(b"outlines")
    assert zarr.segmentation_outlines("s", "cells") == (outlines, STAMP)
    # The folder does not answer, twice: refused in plain words (503), nothing kept under "0", nothing pruned.
    zarr.element_stamp = reads(None)
    for call in (lambda: zarr.counts("s", "cells", rule={"min_qv": 30}), lambda: zarr.counts("s", "cells"),
                 lambda: zarr.counts("s", "cells", cached_only=True), lambda: zarr.segmentation_outlines("s", "cells")):
        with pytest.raises(SampleError, match="could not be read just now") as refused:
            call()
        assert refused.value.status == 503
    assert len(calls) == 1 and outlines.exists() and not list(outlines.parent.glob("cells-0.*"))
    assert sorted(p.name for p in folder.iterdir()) == [whole.name]
    assert [(e["key"], e["version"]) for e in zarr.segmentations("s")] == [("cells", Z.UNREAD)]   # listed, no version
    # It answers again a moment later: read once more, and the work goes on under the real stamp.
    zarr.element_stamp = reads(None, STAMP)
    strict = zarr.counts("s", "cells", rule={"min_qv": 30})
    assert strict.name == f"cells-qv30-{V}-{STAMP}.npz" and whole.exists() and len(calls) == 2
    zarr.element_stamp = reads(None, STAMP)
    assert zarr.segmentations("s")[0]["version"] == STAMP
    zarr.element_stamp = reads(None, STAMP)
    assert zarr.segmentation_outlines("s", "cells") == (outlines, STAMP)


# ---- a helper's failure is said in its own words, never its progress lines ---------------------------------------------
def test_a_helper_failure_never_says_its_progress_lines(tmp_path):
    script = tmp_path / "python"
    script.write_text(f"#!{sys.executable}\nimport json, sys\n"
                      "asked = sys.stdin.read()                 # the arguments, as the engine sends them\n"
                      "for i in range(3):\n"
                      "    sys.stderr.write(json.dumps({'progress': 'Reading', 'fraction': i / 4}) + '\\n')\n"
                      "if 'words' in asked:\n"
                      "    sys.stderr.write('MemoryError: out of memory\\n')\n"
                      "    sys.stderr.write(json.dumps({'progress': 'Counting', 'fraction': 0.9}) + '\\n')\n"
                      "sys.exit(1)\n")
    script.chmod(0o755)
    zarr = samples(tmp_path, str(script))
    with pytest.raises(Z.HelperFailed) as failed:                    # without a progress callback, as counts runs it
        zarr.helper("count", {})
    assert str(failed.value) == "helper exited with code 1" and failed.value.status == 422
    with pytest.raises(Z.HelperFailed) as failed:
        zarr.helper("count", {"words": True})
    assert str(failed.value) == "MemoryError: out of memory"
    heard = []
    with pytest.raises(Z.HelperFailed) as failed:
        zarr.helper("count", {"words": True}, progress=lambda text, fraction: heard.append(fraction))
    assert str(failed.value) == "MemoryError: out of memory" and heard == [0, 0.25, 0.5, 0.9]
