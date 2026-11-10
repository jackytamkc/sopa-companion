"""SpatialData helper for the SoPA companion. It runs in the sopa environment ([zarr] python in config.toml), started by
engine/zarr_samples.py:

    python -m sopa_companion.zarr_bridge.helper <command> '<json arguments>'
    python -m sopa_companion.zarr_bridge.helper <command> - [<path> ...]        (the JSON arguments on stdin)

The engine uses the second form: one command-line argument is limited to 128 KiB, which a save's provenance can pass.
The paths after "-" are where the command writes (out, output), named there only for other processes to see.

Progress goes to stderr as JSON lines {"progress": text, "fraction": 0..1}. The result is one JSON object on the last
line of stdout; failures print {"error": text} and exit with status 1.

Every coordinate this helper hands out is in the *working frame*: the intrinsic coordinate system of the chosen
transcripts (points) element, i.e. the x/y/z columns proseg segments (micrometres for Xenium). Shapes and images are
mapped into it with spatialdata's transformations, the way sopa's `to_intrinsic` aligns elements. A sample without
transcripts (images only: it opens for stain segmentation alone) works in one of its coordinate systems instead
(_image_frame): a micrometre one its main image reaches, else "global", else the one most of its images name.

Commands
  inspect  {path, points_key?}
           Elements, their transformations into the working frame and the candidate priors (cheap: parquet metadata,
           a small sample of rows and the shapes tables; no pixel or transcript bulk reads), and what the page says of
           them: the frame's unit (units), what z holds (z), the images left out and why (skipped_images), the gene
           column (gene_column), every points element (points_elements), each image as it is on disk (stamp:
           image_stamp) and notes. points_key null: no transcripts.
           A sample that cannot be opened fails with one plain line (HelperError).
  shapes   {path, key, points_key, out}
           A shapes element as compact outlines in the working frame: <out>.bin + <out>.json (layout in the json).
  prepare  {path, points_key, region, prior, out}
           The region's transcripts with the chosen prior, written as a Xenium-schema transcripts parquet for
           `proseg --xenium`. Shapes priors are assigned with sopa's own point-in-polygon join
           (sopa.spatial.join._get_cell_id, as sopa.make_transcript_patches(prior_shapes_key=...) does).
  save     {path, points_key, output, shapes_key, table_key, set_default, overwrite, provenance, kind?, keep?,
            view?, frame_from?, stage_to?, consolidate?}
           Proseg's results into the sample as sopa.segmentation.proseg builds them (sopa's _read_proseg with the
           transcripts' transformations, sopa's parse_table, sopa's set_boundaries_attrs), or a stain run's kept cells,
           written as one recoverable transaction: a lock per zarr shared by every process, names checked with
           SpatialData's own rules first, elements being replaced moved aside (atomic rename) with a write-ahead journal
           (each move listed before it happens), written with no-clobber writes, and rolled back on any failure. An
           interrupted save (killed process) is rolled back by the next save into that zarr (a rollback cut short too
           is safe to run again): its own journal only (the whole backup folder name and the zarr's path in the
           journal), and only once the process that wrote it has died; while one cannot be resolved (a live save,
           another computer's, or a rollback whose folder could not be marked) that save writes nothing and says why
           (`unfinished`: that folder); a rollback that left something in its folder is said in `warnings`. API mode:
           `path` may be null (no target), `view` (the engine's own zarr) takes the same transaction
           unconsolidated, `frame_from` places the result, `stage_to` receives it first for the notebook; answer:
           shapes_key, table_key, cells, genes, set_default, replaced, staged, written_to, view_written, view_error,
           warnings (save's docstring).
  count    {path, key, points_key, out, box?, min_qv?, exclude?, table?, by?}
           A shapes element's cells x genes counts for Compare, written to <out> (.npz): its own count when `table`
           names a table annotating it (its X: proseg's own count, `by` "proseg", or Xenium's, `by` "xenium"), each row
           given to the outline its instance id names; else, for a segmentation with no count of its own, the
           transcripts inside its outlines under the run's input rule (count_rule: every transcript of quality min_qv
           (20) or more of a real gene, none whose name the run's excluded-genes pattern `exclude` matches), each
           counted in the outline that holds its position (2-D; sopa's own join over every outline of the element:
           where outlines overlap, in the one its join finds first, never twice). With `box`, rows only for the
           outlines whose centre lies in it (their transcripts read whole, each one where a count of the whole element
           puts it).
  segmentations {path}
           What says how each shapes element was made (attributes and one obs column; no shapes or transcripts read):
           the app's stain saves (sopa_companion attrs on the shapes, or the previous version's proseg_live ones), the
           tables that annotate an element (their region, obs segmentation_method, uns sopa_companion or proseg_live of
           an app's proseg save and the run it came from, and whether they hold proseg's own count: PROSEG_TABLE_KEYS).
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
import glob
import hashlib
import shutil
import json
import math
import os
from pathlib import Path
import re
import socket
import sys
import time
import traceback
import warnings

import numpy as np

from sopa_companion import names, portable

UNASSIGNED = "UNASSIGNED"
KEY = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,99}")
# Compare's rule for a segmentation with no count of its own: proseg --xenium's own input rule (the run's min_qv,
# 20 by default; real genes; none its excluded-genes pattern matches), each transcript in the outline holding its
# position in the flat view, where a count of the whole element puts it. Counts are cached by the rule's name
# (count_rule; engine/zarr_samples.py counts); a new way of counting gets a new COUNT_VERSION, so no file counted the
# old way is ever reused (2: a box count joins every outline of the element, as a whole count does; 3: the control
# probes of CosMx and MERSCOPE are not genes either, NOT_GENES; 4: an element a table holds its own count of, proseg's
# or Xenium's, is read from it, and the file says whose count it holds: `count`).
COUNT_VERSION = 4
COUNT_MIN_QV = 20.0
TABLE_COUNTS = ("proseg", "xenium")      # whose own count a table holds (count's `by`); without a table: "outlines"
# A table holding proseg's own count (its X): its uns carries proseg's run record (proseg-output.zarr's proseg_run) or
# this app's proseg save's (names.PROVENANCE_KEYS). The mixing check's earlier saved line takes the same tables.
PROSEG_TABLE_KEYS = ("proseg_run", *names.PROVENANCE_KEYS)
# Not genes, for transcripts without an is_gene column: the runs' default excluded-genes pattern (engine/proseg_options.py
# keeps the same text), so runs and Compare leave out the same names: Xenium's controls, CosMx's (proseg's own CosMx
# preset: FalseCode, SystemControl, NegPrb, Negative) and MERSCOPE's Blank-*. Case-sensitive, as proseg reads it.
NOT_GENES_PATTERN = r"^(Deprecated|NegControl|Unassigned|Intergenic|FalseCode|SystemControl|NegPrb|Negative|Blank)"
NOT_GENES = re.compile(NOT_GENES_PATTERN)
PATTERN_MAX = 200          # the longest excluded-genes pattern taken (_excluded)
GENE_CHECK_S = 10.0        # the longest that pattern may take over a sample's gene names (_matching: microseconds each)
CENTRE_MARGIN_UM = 1.0     # `box` keeps outlines whose centre lies this close outside it too (the engine's own centres
                           # come from the exported float32 outlines; it picks its cells from these by `index`)
NO_TRANSCRIPTS = "This sample has no transcripts: only stain segmentation runs here"
# Other readers' gene columns, for transcripts whose points say no feature_key (_gene_column).
GENE_COLUMNS = ("gene", "target", "target_name", "gene_name")
NUCLEUS_FLAGS = ("overlaps_nucleus", "CellComp")     # a nucleus flag (CosMx's CellComp: Nuclear), never a cell id
BACKGROUND_SHARE = 0.05    # an id holding more of the transcripts than this is the "no cell" mark, not a cell
# A coordinate system in micrometres, by its name (whole name, any case): sopa's toy data and Visium HD say "microns".
MICRONS = re.compile(r"(?i)microns?|micrometers?|micrometres?|um|µm")
UNIT_SCALE = 0.01          # images within 1 % of 1 pixel per frame unit: the frame is in pixels (_units)
# z in whole numbers spanning at most this many values is a plane number, not micrometres (_z_kind): CosMx's 0-9,
# MERSCOPE's 0-6, Molecular Cartography's 0-25 slices. proseg would read them as µm (its thickness is max - min).
Z_PLANES_MAX = 64
# The app's own stain saves, by their default names (engine/zarr_samples.py save_defaults): never the suggested prior
# while the sample has other cell outlines (a stain save of one box is not the sample's cells). Its proseg saves are
# known by their table instead (_app_saves): sopa's own proseg writes proseg_boundaries too.
APP_SAVE = re.compile(r"(cellpose|instanseg)_boundaries_\d+")


def progress(text: str, fraction: float | None = None):
    """Progress for the engine. A vanished reader (engine stopped) must never abort the work in hand."""
    try:
        sys.stderr.write(json.dumps({"progress": text, "fraction": fraction}) + "\n")
        sys.stderr.flush()
    except (BrokenPipeError, OSError, ValueError):
        pass


class HelperError(Exception):
    """A failure in plain words, for the page. `details`: further fields of the answer ({"error": text, **details}):
    a name clash's {clash, existing}, so the Save dialog can offer Replace; a save stopped by an unfinished one's
    {unfinished: its backup folder} (_recover), nothing written."""
    def __init__(self, message: str, **details):
        super().__init__(message)
        self.details = details


class Skipped(HelperError):
    """An image left out of the sample's usable images, and why (`reason`: a short clause for skipped_images)."""
    def __init__(self, message: str, reason: str):
        super().__init__(message)
        self.reason = reason


def _min_qv(value) -> float:
    """A minimum transcript quality: a number of 0 or more (HelperError in plain words otherwise)."""
    try:
        quality = float(value) + 0.0                 # + 0.0: -0.0 is 0
    except (TypeError, ValueError):
        quality = math.nan
    if isinstance(value, bool) or not math.isfinite(quality) or quality < 0:
        raise HelperError("The minimum transcript quality must be a number of 0 or more")
    return quality


def _excluded(pattern) -> re.Pattern:
    """The run's excluded-genes pattern (proseg's --excluded-genes: a transcript whose gene name it matches anywhere is
    left out), compiled; HelperError in plain words when it is not a pattern, is longer than PATTERN_MAX, reads
    differently here than in proseg (Python only warns of a set inside a set, or of -- && ~~ || in one: proseg's
    [[:alpha:]] is a letter class, Python's a different set), or repeats a part that itself repeats (as in (a+)+ or
    (a{1,9}){1,9}) or a choice (as in (a|b)+): Python's matcher can then take minutes over one gene name, proseg's never
    does (_slow_repeat). What no rule here catches is held to a time limit where it is used (_matching)."""
    from re import _parser
    if not isinstance(pattern, str):
        raise HelperError("The excluded genes must be given as a text pattern")
    if len(pattern) > PATTERN_MAX:
        raise HelperError(f"The excluded-genes pattern is too long ({len(pattern)} characters): {PATTERN_MAX} at most")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            parsed = _parser.parse(pattern)          # parsed each time: re.compile's cache would hide the warning
    except FutureWarning:
        raise HelperError(f"The excluded-genes pattern {pattern!r} reads differently here than in proseg (a set inside "
                          "a set, or -- && ~~ || inside one): write it without them") from None
    except re.error as exc:
        raise HelperError(f"The excluded-genes pattern {pattern!r} is not a valid pattern ({exc.msg})") from None
    slow = _slow_repeat(parsed)
    if slow == "repeat":
        raise HelperError(f"The excluded-genes pattern {pattern!r} repeats a part that itself repeats (as in (a+)+), "
                          "which can take very long to check: write it without the inner repeat")
    if slow == "choice":
        raise HelperError(f"The excluded-genes pattern {pattern!r} repeats a choice (as in (a|b)+), which can take very "
                          "long to check: write the choice as a set ([ab]+) or without the repeat")
    return re.compile(pattern)


def _slow_repeat(parsed) -> str | None:
    """What makes a parsed pattern (re._parser) backtrack without end over one name, or None: "repeat", a part repeated
    a varying number of times (+, *, {1,9}: not ?, not {3}) that holds another such part; "choice", such a part that
    holds a choice (|), whose alternatives can match the same letters in many ways (as (\\w|\\w)+ does)."""
    from re import _constants as C, _parser
    repeats = {C.MAX_REPEAT, C.MIN_REPEAT, getattr(C, "POSSESSIVE_REPEAT", C.MAX_REPEAT)}

    def parts(value):
        """The sub-patterns in a parsed item's argument: groups, branches, assertions, conditionals."""
        if isinstance(value, _parser.SubPattern):
            yield value
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from parts(item)

    def walk(items, inside: bool) -> str | None:
        for op, value in items:
            if op in repeats:
                low, high = value[0], value[1]
                varies = high != low and (high == C.MAXREPEAT or high > 1)
                if varies and inside:
                    return "repeat"
                found = walk(value[2], inside or varies)
                if found:
                    return found
                continue
            if op == C.BRANCH and inside:
                return "choice"
            for part in parts(value):
                found = walk(part, inside)
                if found:
                    return found
        return None
    return walk(parsed, False)


def _matching(pattern: str, names: list[str]) -> np.ndarray:
    """Which of `names` the excluded-genes pattern matches anywhere (re.search, as proseg's --excluded-genes), worked
    out in a child Python held to GENE_CHECK_S: Python's matcher cannot be stopped from within, and a pattern that
    backtracks over one name in a way _excluded's rules do not see (many .* in a row) must fail the count in plain words,
    not hold this helper for an hour."""
    import subprocess
    code = ("import json, re, sys; pattern, names = json.load(sys.stdin); found = re.compile(pattern); "
            "json.dump([found.search(n) is not None for n in names], sys.stdout)")
    try:
        done = subprocess.run([sys.executable, "-I", "-c", code], input=json.dumps([pattern, list(names)]),
                              capture_output=True, text=True, timeout=GENE_CHECK_S, check=True,
                              **portable.no_window())
    except subprocess.TimeoutExpired:
        raise HelperError(f"The excluded-genes pattern {pattern!r} takes too long to check against this sample's gene "
                          f"names (over {GENE_CHECK_S:g} s): write it more simply") from None
    except subprocess.CalledProcessError as exc:
        raise HelperError(f"The excluded-genes pattern {pattern!r} could not be checked: "
                          f"{_one_line(exc.stderr or exc)}") from None
    return np.asarray(json.loads(done.stdout), dtype=bool).reshape(len(names))


def count_rule(min_qv: float = COUNT_MIN_QV, exclude: str | None = None) -> str:
    """The name of a counting rule, which names the counts made under it (engine/zarr_samples.py counts): qv<min_qv>
    (its shortest exact text, "e-" written "em"), then, with an excluded-genes pattern, -x<the first 8 hex digits of its
    sha1>, then -v<COUNT_VERSION>. The default (quality 20 or more, real genes) is "qv20-v4". HelperError in plain words
    for a quality below 0 or a pattern that is not one."""
    quality = repr(_min_qv(min_qv)).removesuffix(".0").replace("e-", "em").replace("e+", "ep")
    if exclude is None:
        return f"qv{quality}-v{COUNT_VERSION}"
    _excluded(exclude)
    return f"qv{quality}-x{hashlib.sha1(exclude.encode()).hexdigest()[:8]}-v{COUNT_VERSION}"


COUNT_RULE = count_rule()          # the default rule's name


# ---- reading ----------------------------------------------------------------------------------------------------
# What SpatialData says of an element it refuses (spatialdata._core.validation, spatialdata.models), in plain words;
# anything else by its own first sentence.
REFUSED = (("Name must contain only", "its name has characters SpatialData refuses (letters, digits, _ . - only)"),
           ("Name cannot start with", "its name starts with __, which SpatialData refuses"),
           ("Name cannot be", "its name is one SpatialData refuses"),
           ("case-variant", "its name differs from another's only by case"),
           ("`radius` not found", "its circles have no radius"),
           ("`radius` must be provided", "its circles have no radius"))


def _read(path: str, selection: tuple[str, ...]):
    """The sample, read by spatialdata. An element SpatialData refuses (a name it does not allow, circles without
    their radius, ...) stops the whole read in spatialdata itself: HelperError naming each one in one plain line,
    instead of spatialdata's text (whose advice, sanitize_table, is for tables only)."""
    import spatialdata as sd
    try:
        from spatialdata._core.validation import ValidationError
    except ImportError:                 # a spatialdata without it: nothing of the kind to name
        ValidationError = ()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return sd.read_zarr(path, selection=selection)
        except ValidationError as exc:
            raise HelperError(_refused(exc)) from None


def _refused(exc) -> str:
    """spatialdata's ValidationError as one line: "This sample cannot be opened: shapes 'cell boundaries': its name
    has characters SpatialData refuses (...). Fix or remove it, then open the sample again." """
    found = []
    for error in getattr(exc, "errors", None) or []:
        where = [str(p) for p in getattr(error, "location", ()) or ()]
        said = (str(getattr(error, "message", "")).strip().splitlines() or [""])[0]
        plain = next((words for start, words in REFUSED if start in said), said.split(". ")[0].rstrip(".") or "refused")
        name = f"{where[0]} '{'/'.join(where[1:])}'" if len(where) > 1 else (where[0] if where else "an element")
        found.append(f"{name}: {plain}")
    if not found:
        found.append(_one_line(exc))
    return (f"This sample cannot be opened: {'; '.join(found)}. Fix or remove {'it' if len(found) == 1 else 'them'}, "
            "then open the sample again")


def _one_line(exc) -> str:
    """An exception as one short line: its first line of text, after its type when that says something."""
    text = next((line.strip() for line in str(exc).splitlines() if line.strip()), "")
    line = text if isinstance(exc, HelperError) else f"{type(exc).__name__}: {text}" if text else type(exc).__name__
    return line if len(line) <= 300 else line[:299] + "…"


def _attributes(path: Path) -> dict:
    """Attributes of a zarr group (v3 zarr.json or v2 .zattrs)."""
    for name in ("zarr.json", ".zattrs"):
        try:
            data = json.loads((path / name).read_text())
        except (OSError, ValueError):
            continue
        return data.get("attributes", {}) if name == "zarr.json" else data
    return {}


def _points_files(path: str, key: str) -> list[str]:
    base = Path(path) / "points" / key / "points.parquet"
    if base.is_file():
        return [str(base)]
    files = sorted(glob.glob(str(base / "*.parquet")), key=lambda f: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", f)])
    if not files:
        raise HelperError(f"Points element '{key}' has no parquet data")
    return files


def _transformation(sdata, element, target):
    """`element`'s intrinsic frame -> the working frame `target`: another element's intrinsic frame (the transcripts),
    as sopa's to_intrinsic, or a coordinate system's name (a sample without transcripts: _to_system). None when
    nothing links them."""
    from spatialdata.transformations import Sequence, get_transformation, get_transformation_between_coordinate_systems
    if isinstance(target, str):
        return _to_system(sdata, element, target)
    try:
        return get_transformation_between_coordinate_systems(sdata, element, target)
    except Exception:
        first, second = get_transformation(element, get_all=True), get_transformation(target, get_all=True)
        common = set(first) & set(second)
        if not common:
            return None
        system = "global" if "global" in common else sorted(common)[0]
        return Sequence([first[system], second[system].inverse()])


def _matrix(transformation) -> np.ndarray:
    return np.asarray(transformation.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")), dtype=float)


def _affine(sdata, element, target) -> np.ndarray | None:
    """3x3 affine (x, y) of `element`'s intrinsic frame -> the working frame `target` (_transformation)."""
    transformation = _transformation(sdata, element, target)
    return None if transformation is None else _matrix(transformation)


def _elements_of(sdata):
    """(kind, key, element) of every element, in a fixed order."""
    for kind in ("shapes", "images", "labels", "points"):
        container = getattr(sdata, kind)
        for key in sorted(container):
            yield kind, key, container[key]


def _to_system(sdata, element, system: str):
    """`element`'s intrinsic frame -> the coordinate system `system`: its own transformation, else through another
    element naming both one of its systems and `system` (Visium HD's images reach its micrometre system only through
    its shapes; spatialdata's own path search refuses such samples: several equal paths). None when neither does."""
    from spatialdata.transformations import Sequence, get_transformation
    own = get_transformation(element, get_all=True)
    if system in own:
        return own[system]
    for _, _, other in _elements_of(sdata):
        theirs = get_transformation(other, get_all=True)
        if other is element or system not in theirs:
            continue
        for shared in sorted(set(own) & set(theirs)):
            try:
                return Sequence([own[shared], theirs[shared].inverse(), theirs[system]])
            except Exception:            # a transformation without an inverse: another path
                continue
    return None


def _image_frame(sdata) -> str | None:
    """The working frame of a sample without transcripts: the coordinate system its main image (the attrs'
    cell_segmentation_image, else the first by name) reaches (_to_system) that is in micrometres (MICRONS), else
    "global", else the one most images name, then by name. None without an image (nothing to open)."""
    from spatialdata.transformations import get_transformation
    images = dict(sdata.images)
    if not images:
        return None
    # `is None`, never `or`: a single-scale image is a DataArray, whose truth is its pixels' (read whole, then refused).
    main = images.get((sdata.attrs or {}).get("cell_segmentation_image"))
    if main is None:
        main = images[sorted(images)[0]]
    named = Counter(system for image in images.values() for system in get_transformation(image, get_all=True))
    reached = [system for system in sorted(sdata.coordinate_systems) if _to_system(sdata, main, system) is not None]
    if not reached:
        return None
    return min(reached, key=lambda s: (not MICRONS.fullmatch(s), s != "global", -named[s], s))


def _frame_of(sdata, points_key: str | None):
    """The working frame's target for _transformation: the transcripts element, or a sample without transcripts' frame
    (_image_frame); HelperError when it has neither."""
    if points_key:
        if points_key not in sdata.points:
            raise HelperError(f"The sample has no points element '{points_key}'")
        return sdata.points[points_key]
    frame = _image_frame(sdata)
    if frame is None:
        raise HelperError("This sample has no transcripts, and no image to work on")
    return frame


def _frame_transformations(sdata, points_key: str | None) -> dict:
    """The transformations of a new element drawn in the working frame: the transcripts' own (sopa's
    copy_transformations), or, without transcripts, Identity to the frame's coordinate system and, to each other system
    an element links with it, that element's way there."""
    from sopa.utils import copy_transformations
    from spatialdata.transformations import Affine, Identity, get_transformation
    if points_key:
        return copy_transformations(_frame_of(sdata, points_key))
    frame = _frame_of(sdata, None)
    found = {frame: Identity()}
    for system in sorted(sdata.coordinate_systems):
        for _, _, element in _elements_of(sdata):
            theirs = get_transformation(element, get_all=True)
            if system in found or frame not in theirs or system not in theirs:
                continue
            try:
                matrix = _matrix(theirs[system]) @ np.linalg.inv(_matrix(theirs[frame]))
            except np.linalg.LinAlgError:
                continue
            found[system] = Affine(matrix, input_axes=("x", "y"), output_axes=("x", "y"))
    return found


def _apply(matrix: np.ndarray, xy: np.ndarray) -> np.ndarray:
    return xy @ matrix[:2, :2].T + matrix[:2, 2]


def _default_points(sdata, attrs: dict) -> tuple[str | None, str | None]:
    """The transcripts element this app uses, and what chose it: "attrs" (transcripts_dataframe), "name" (the one named
    transcripts) or "first" (the first listed); (None, None) without one (a sample of images only)."""
    keys = list(sdata.points)
    if not keys:
        return None, None
    for candidate in (attrs.get("transcripts_dataframe"), attrs.get("transcripts")):
        if candidate in keys:
            return candidate, "attrs"
    return ("transcripts", "name") if "transcripts" in keys else (keys[0], "first")


def _gene_column(names, element_attrs: dict) -> str:
    """The transcripts' gene column: the points' feature_key, else feature_name, else the one column of GENE_COLUMNS
    they have (MERSCOPE's gene, CosMx's target, ...); HelperError in plain words otherwise, naming what they have."""
    names = [str(n) for n in names if not str(n).startswith("__")]
    feature = element_attrs.get("feature_key")
    if feature in names:
        return feature
    if "feature_name" in names:
        return "feature_name"
    found = [c for c in GENE_COLUMNS if c in names]
    if len(found) == 1:
        return found[0]
    if found:
        raise HelperError(f"The transcripts have several gene columns ({', '.join(found)}) and nothing says which one "
                          "to use: set the points' feature_key")
    shown = ", ".join(names[:20]) + (", …" if len(names) > 20 else "")
    raise HelperError(f"The transcripts have no gene column (they have {shown}): name it feature_name, or set the "
                      "points' feature_key")


def _column_stats(files: list[str], columns: list[str]) -> dict:
    """Row count and per-column min/max from parquet row-group statistics (no data read). `complete[c]` says whether
    every row group reported column c; when x/y are incomplete the extent is computed from the data instead."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    count, low, high, seen, groups = 0, {}, {}, {}, 0
    for file in files:
        meta = pq.ParquetFile(file).metadata
        count += meta.num_rows
        for r in range(meta.num_row_groups):
            group = meta.row_group(r)
            groups += 1
            for i in range(group.num_columns):
                column = group.column(i)
                name, stats = column.path_in_schema, column.statistics
                if name in columns and stats is not None and stats.has_min_max:
                    try:
                        lo, hi = float(stats.min), float(stats.max)
                    except (TypeError, ValueError):
                        continue
                    if not (math.isfinite(lo) and math.isfinite(hi)):
                        continue
                    low[name] = min(low.get(name, lo), lo)
                    high[name] = max(high.get(name, hi), hi)
                    seen[name] = seen.get(name, 0) + 1
    complete = {name: seen.get(name, 0) == groups for name in columns}
    for name in ("x", "y"):
        if name in columns and not complete[name]:
            import pyarrow.dataset as ds
            progress("Measuring the transcripts' extent (no parquet statistics)…")
            table = ds.dataset(files, format="parquet").to_table(columns=[name])
            values = pc.min_max(pc.drop_null(table.column(name)))
            if values["min"].is_valid:
                low[name], high[name] = float(values["min"].as_py()), float(values["max"].as_py())
                complete[name] = True
    return {"count": count, "min": low, "max": high, "complete": complete}


def _sample_rows(files: list[str], columns: list[str], rows: int = 600_000):
    """A spatially spread sample: the first row group of up to six evenly spaced files."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    picks = sorted(set(np.linspace(0, len(files) - 1, min(6, len(files))).round().astype(int).tolist()))
    tables = []
    for index in picks:
        file = pq.ParquetFile(files[index])
        table = file.read_row_group(0, columns=[c for c in columns if c in file.schema_arrow.names])
        tables.append(table.slice(0, rows // len(picks)))
    return pa.concat_tables(tables, promote_options="default") if tables else None


def _flags(array) -> np.ndarray:
    """A nucleus flag column as strings: booleans -> "1"/"0", numbers -> their integer text, nulls -> "0"."""
    import pyarrow as pa
    import pyarrow.compute as pc
    if hasattr(array, "combine_chunks"):
        array = array.combine_chunks()
    if pa.types.is_dictionary(array.type):
        array = array.dictionary_decode()
    if pa.types.is_boolean(array.type):
        values = np.asarray(pc.fill_null(array, False).to_numpy(zero_copy_only=False), dtype=bool)
        return np.where(values, "1", "0").astype(object)
    if pa.types.is_integer(array.type) or pa.types.is_floating(array.type):
        values = np.asarray(pc.fill_null(array, 0).to_numpy(zero_copy_only=False))
        return np.where(np.isfinite(values.astype(float)), values.astype(float).astype(np.int64).astype(str), "0").astype(object)
    return _as_strings(array)


def _castable(source, target) -> bool:
    """Whether a column of arrow type `source` becomes `target` without reading text as numbers: anything becomes text;
    numbers and booleans become numbers; only booleans and whole numbers become booleans."""
    import pyarrow as pa
    types = pa.types
    if types.is_string(target) or types.is_large_string(target):
        return True
    if types.is_boolean(target):
        return types.is_boolean(source) or types.is_integer(source)
    return types.is_integer(source) or types.is_floating(source) or types.is_boolean(source)


def _as_strings(array) -> np.ndarray:
    import pyarrow as pa
    import pyarrow.compute as pc
    if pa.types.is_dictionary(array.type):
        array = array.cast(array.type.value_type) if hasattr(array, "cast") else pc.cast(array, array.type.value_type)
    if not (pa.types.is_string(array.type) or pa.types.is_large_string(array.type)):
        array = pc.cast(array, pa.large_string())
    values = np.asarray(array.to_numpy(zero_copy_only=False), dtype=object)
    values[np.equal(values, None)] = None
    return values


def _name(path: str, attrs: dict) -> str:
    """A short sample name: the Xenium region from an output-<inst>-<slide>-<region>-<date>-<time> name, else the
    folder name without .zarr."""
    for source in (attrs.get("xenium_output_path") or "", Path(path).name):
        stem = re.sub(r"(_proseg|-proseg)?(\.zarr)?/?$", "", Path(source).name) if source else ""
        parts = re.split(r"__|-", stem)
        dates = [i for i, p in enumerate(parts) if re.fullmatch(r"20\d{6}", p)]
        if stem.startswith("output") and dates and dates[0] > 3:
            return "-".join(parts[3:dates[0]])
    return re.sub(r"\.zarr/?$", "", Path(path).name)


# ---- inspect ----------------------------------------------------------------------------------------------------
def inspect(args: dict) -> dict:
    """What the engine needs to open the sample (module docstring), schema 2: also points_elements and points_choice
    (what chose points_key: "asked", "attrs", "name", "first"; null without transcripts), frame ({kind: "points", key}
    or {kind: "system", name}: _image_frame), units (_units), z (_z_kind), gene_column (_gene_column), extent (the
    transcripts', else the usable images' together), skipped_images [{key, reason}], default_image (the attrs'
    cell_segmentation_image, else the first usable image that is not RGB, else the first usable), each image's `stamp`
    (image_stamp: the engine's image caches are kept by it), each shapes entry's count/geometry/extent over its
    non-empty geometries only (an element of empty ones has count 0 and no extent) and app_save (_app_saves), and notes
    [{code, text, detail}] for the page (a cell column's "no cell" mark read from the data: unassigned_token)."""
    import pyarrow.parquet as pq
    path = str(Path(args["path"]).resolve())
    if not (Path(path) / "zarr.json").exists() and not (Path(path) / ".zgroup").exists():
        raise HelperError("This folder is not a zarr store")
    # Before the read: an image written again while it is read is stamped as it was, so the engine opens it again.
    stamps = {folder.name: image_stamp(path, folder.name) for folder in _folders(Path(path), "images")}
    progress("Reading the SpatialData object…", 0.1)
    sdata = _read(path, ("points", "shapes", "images", "labels"))
    attrs = dict(sdata.attrs or {})
    if args.get("points_key"):
        if args["points_key"] not in sdata.points:
            raise HelperError(f"Unknown points element '{args['points_key']}'")
        points_key, choice = args["points_key"], "asked"
    else:
        points_key, choice = _default_points(sdata, attrs)
    frame = _frame_of(sdata, points_key)       # HelperError for a sample with neither transcripts nor an image

    points, chosen, notes = [], None, []
    for key, element in sdata.points.items():
        files = _points_files(path, key)
        schema = pq.read_schema(files[0])
        columns = {f.name: str(f.type) for f in schema if not f.name.startswith("__")}
        element_attrs = element.attrs.get("spatialdata_attrs", {})
        entry = {"key": key, "columns": columns, "feature_key": element_attrs.get("feature_key"),
                 "instance_key": element_attrs.get("instance_key"), "files": len(files)}
        if key == points_key:
            progress("Reading transcript statistics…", 0.3)
            entry["gene_column"] = _gene_column(columns, element_attrs)
            stats = _column_stats(files, ["x", "y", "z", "overlaps_nucleus"])
            entry.update(count=stats["count"],
                         extent=[stats["min"].get("x"), stats["min"].get("y"), stats["max"].get("x"), stats["max"].get("y")],
                         z_range=[stats["min"].get("z"), stats["max"].get("z")] if "z" in columns else None)
            if not stats["count"]:
                raise HelperError("The sample's transcripts are empty")
            if not all(v is not None and math.isfinite(v) for v in entry["extent"]):
                raise HelperError("The transcripts have no usable x/y coordinates")
            entry["prior_columns"] = _prior_columns(files, columns, element_attrs, attrs, stats)
            entry["z"] = _z_kind(files, columns, stats)
            notes += [c.pop("note") for c in entry["prior_columns"]["cells"] if c.get("note")]
            chosen = entry
        else:
            entry["count"] = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
        points.append(entry)

    progress("Reading images and shapes…", 0.6)
    images, skipped = [], []
    subject = "the transcripts" if points_key else "the other images"
    for key, image in sdata.images.items():
        try:
            entry = _image_entry(path, key, image, _affine(sdata, image, frame), subject)
        except Exception as exc:   # an image we cannot place is listed but not offered
            images.append({"key": key, "error": str(exc), "stamp": stamps.get(key)})
            skipped.append({"key": key, "reason": exc.reason if isinstance(exc, Skipped) else "could not be read"})
            continue
        entry["stamp"] = stamps.get(key)
        images.append(entry)
        if entry["to_frame"] is None:
            skipped.append({"key": key, "reason": f"no coordinate system in common with {subject}"})
    usable = [i for i in images if "error" not in i and i.get("to_frame")]
    if not points_key and not usable:
        raise HelperError("This sample has no transcripts, and none of its images can be placed: nothing here can be "
                          "opened")
    app_saves = _app_saves(Path(path), list(sdata.shapes))
    shapes = []
    for key, gdf in sdata.shapes.items():
        matrix = _affine(sdata, gdf, frame)
        # Only the geometries that are there: an element of empty ones (all-NaN bounds) has count 0 and no extent, so
        # it is neither offered nor listed, and the detail never carries NaN (the page's JSON refuses it).
        present = gdf[~(gdf.geometry.isna() | gdf.geometry.is_empty)]
        types = present.geometry.geom_type.value_counts().to_dict()
        entry = {"key": key, "count": int(len(present)), "geometry": {str(k): int(v) for k, v in types.items()},
                 "median_area_um2": None,
                 "columns": [c for c in gdf.columns if c != "geometry"][:40], "to_frame": matrix.tolist() if matrix is not None else None}
        if key in app_saves:
            entry["app_save"] = True
        bounds = present.total_bounds if len(present) else None
        if matrix is not None and bounds is not None and np.isfinite(bounds).all():
            x0, y0, x1, y1 = bounds
            corners = _apply(matrix, np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]], float))
            entry["extent"] = [float(corners[:, 0].min()), float(corners[:, 1].min()), float(corners[:, 0].max()), float(corners[:, 1].max())]
            areas = np.asarray(present.geometry.area, float) * abs(np.linalg.det(matrix[:2, :2]))
            areas = areas[np.isfinite(areas) & (areas > 0)]
            entry["median_area_um2"] = float(np.median(areas)) if len(areas) else None
        shapes.append(entry)
    tables = []
    for folder in sorted((Path(path) / "tables").glob("*")):
        if folder.is_dir():
            tables.append(_table_entry(folder))
    named = attrs.get("cell_segmentation_image")
    default_image = next((i["key"] for i in usable if i["key"] == named), None) or \
        next((i["key"] for i in usable if not _rgb(i)), None) or (usable[0]["key"] if usable else None)
    progress("Done", 1.0)
    return {"schema": 2, "path": path, "name": _name(path, attrs), "points_key": points_key,
            "points_elements": list(sdata.points), "points_choice": choice,
            "frame": {"kind": "points", "key": points_key} if points_key else {"kind": "system", "name": frame},
            "attrs": _jsonable(attrs), "points": points, "images": images, "skipped_images": skipped,
            "default_image": default_image, "shapes": shapes, "areas": 1, "labels": list(sdata.labels), "tables": tables,
            "suggested_prior": _suggest_prior(shapes, chosen),
            "units": _units(sdata, frame, usable), "z": chosen["z"] if chosen else {"kind": "none"},
            "gene_column": chosen["gene_column"] if chosen else None,
            "extent": chosen["extent"] if chosen else _images_extent(usable), "notes": notes,
            "mtime": _mtime(path)}


def _rgb(image: dict) -> bool:
    """An RGB picture (H&E): 3 channels of 8 bits."""
    return len(image.get("channels") or []) == 3 and image.get("dtype") == "uint8"


def _images_extent(images: list[dict]) -> list[float] | None:
    """The working-frame box the images cover together (a sample without transcripts: its extent)."""
    boxes = []
    for image in images:
        matrix, level = np.asarray(image["to_frame"], float), image["levels"][0]
        (ox, oy), (px, py) = level["origin"], level["pixel"]
        height, width = level["shape"][-2:]
        x0, y0 = matrix[0, 0] * ox + matrix[0, 2], matrix[1, 1] * oy + matrix[1, 2]
        boxes.append([x0, y0, x0 + matrix[0, 0] * px * width, y0 + matrix[1, 1] * py * height])
    if not boxes:
        return None
    boxes = np.asarray(boxes, float)
    return [float(boxes[:, 0].min()), float(boxes[:, 1].min()), float(boxes[:, 2].max()), float(boxes[:, 3].max())]


def _units(sdata, frame, images: list[dict]) -> dict:
    """{frame: "um" | "pixel" | "unknown", why}: what one unit of the working frame is. Every size the app states, limits
    and boxes and proseg's own, takes it as one micrometre. The transcripts' own coordinate system named in micrometres
    (MICRONS) decides: at scale 1 they are µm, at another scale they are not ("pixel"). Else the images do: every
    usable image within UNIT_SCALE of 1 pixel per unit means the frame is in pixels; images of other sizes fit
    micrometres (a sample without transcripts: unknown, unless its frame is a micrometre system)."""
    from spatialdata.transformations import get_transformation
    scales = [abs(float(np.asarray(i["to_frame"], float)[k, k])) * float(i["levels"][0]["pixel"][k]) for i in images for k in (0, 1)]
    pixels = bool(scales) and all(abs(s - 1.0) <= UNIT_SCALE for s in scales)
    if isinstance(frame, str):                  # a sample without transcripts: a coordinate system
        if MICRONS.fullmatch(frame):
            return {"frame": "um", "why": f"The images sit in the sample's coordinate system '{frame}', in micrometres"}
        if pixels:
            return {"frame": "pixel", "why": "The images say nothing of their pixel size: one unit is one pixel"}
        return {"frame": "unknown", "why": f"Nothing in the sample says what unit its coordinate system '{frame}' is in"}
    for system, transformation in sorted(get_transformation(frame, get_all=True).items()):
        if MICRONS.fullmatch(system):
            linear = _matrix(transformation)[:2, :2]
            if np.allclose(linear, np.eye(2), atol=UNIT_SCALE):
                return {"frame": "um", "why": f"The transcripts' coordinate system '{system}' says they are in micrometres"}
            scale = float(np.sqrt(abs(np.linalg.det(linear))))
            return {"frame": "pixel", "why": f"The transcripts reach micrometres ('{system}') at a scale of {scale:.4g}: "
                                             "their own unit is not one micrometre"}
    if pixels:
        return {"frame": "pixel", "why": "Every image has one pixel per unit of the transcripts' coordinates, as when "
                                         "both are in pixels"}
    if scales:
        return {"frame": "um", "why": "The images' pixel sizes fit transcripts in micrometres"}
    return {"frame": "unknown", "why": "Neither an image nor a coordinate system says what unit the transcripts are in"}


def _z_kind(files: list[str], columns: dict, stats: dict) -> dict:
    """What the transcripts' z holds: {kind: "um"} (a depth), {kind: "index", levels} (plane numbers: whole numbers
    spanning at most Z_PLANES_MAX values, which proseg reads as µm all the same) or {kind: "none"} (no z, or one value
    only). Read from the parquet statistics and the first row groups (_sample_rows)."""
    import pyarrow as pa
    import pyarrow.compute as pc
    if "z" not in columns:
        return {"kind": "none"}
    sample = _sample_rows(files, ["z"])
    if sample is None or "z" not in sample.column_names:
        return {"kind": "none"}
    values = pc.cast(sample.column("z"), pa.float64()).to_numpy(zero_copy_only=False)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"kind": "none"}
    complete = stats.get("complete", {}).get("z")
    low = float(stats["min"]["z"]) if complete and "z" in stats["min"] else float(values.min())
    high = float(stats["max"]["z"]) if complete and "z" in stats["max"] else float(values.max())
    if high <= low:
        return {"kind": "none"}
    whole = bool(np.all(values == np.round(values))) and low.is_integer() and high.is_integer()
    if whole and high - low + 1 <= Z_PLANES_MAX:
        return {"kind": "index", "levels": int(len(np.unique(values)))}
    return {"kind": "um"}


def _app_record(attrs: dict) -> dict | None:
    """The record a stain save of this app left on its shapes (their attrs): the first of names.PROVENANCE_KEYS that
    holds one (a sample keeps what either version saved into it), else None."""
    return next((attrs[key] for key in names.PROVENANCE_KEYS if isinstance(attrs.get(key), dict)), None)


def _app_table(folder: Path) -> Path | None:
    """The provenance group a proseg save of this app wrote into its table (uns/<the first of names.PROVENANCE_KEYS
    there is>), else None."""
    return next((folder / "uns" / key for key in names.PROVENANCE_KEYS if (folder / "uns" / key).is_dir()), None)


def _app_saves(path: Path, keys: list[str]) -> set[str]:
    """The shapes elements this app saved: its stain saves' default names (APP_SAVE) or its record on the shapes
    (_app_record), or a table its proseg save wrote (_app_table) annotating them (as `segmentations` reads)."""
    found = {key for key in keys if APP_SAVE.fullmatch(key)}
    found |= {key for key in keys if _app_record(_attributes(path / "shapes" / key)) is not None}
    for folder in sorted((path / "tables").glob("*")) if (path / "tables").is_dir() else []:
        if _app_table(folder) is not None:
            region = _attributes(folder).get("region")
            found |= {str(r) for r in (region if isinstance(region, list) else [region])} & set(keys)
    return found


def _mtime(path: str) -> float:
    stamps = []
    for name in ("zarr.json", ".zattrs", "points", "shapes", "images", "tables"):
        try:
            stamps.append(os.stat(Path(path) / name).st_mtime)
        except OSError:
            pass
    return max(stamps) if stamps else 0.0


IMAGE_METADATA = ("zarr.json", ".zattrs", ".zgroup", ".zarray")      # an image's and its levels' metadata files


def _identity(path: str) -> tuple:
    found = os.stat(path)                  # through a link (a view's linked image): the element it names
    return found.st_ino, found.st_size, found.st_mtime_ns


def image_stamp(path, key: str) -> str | None:
    """An image as it is on disk now: 12 hex digits of the identity, size and mtime of its folder (images/<key>), its
    metadata (zarr.json, or .zattrs/.zgroup) and each level's folder and array metadata. They change whenever the image
    is written again, its transformation, channels, shape or type alone included (a rewrite of its pixels alone, in
    place, is not seen); _mtime does not see the image's own metadata. None when its folder cannot be read now. inspect
    records it on each image (`stamp`), taken before the image is read; the engine keeps what it derives from an image
    (its overview, statistics, open arrays) by it and inspects the sample again when it differs. Standard library only:
    the engine imports it."""
    folder = os.path.join(str(path), "images", key)
    try:
        found = [("", *_identity(folder))]
        with os.scandir(folder) as entries:
            for entry in entries:
                if entry.name in IMAGE_METADATA:
                    found.append((entry.name, *_identity(entry.path)))
                elif entry.is_dir():                                  # a level (0, s0, ...): never its chunks
                    found.append((entry.name, *_identity(entry.path)))
                    for name in IMAGE_METADATA:
                        try:
                            found.append((f"{entry.name}/{name}", *_identity(os.path.join(entry.path, name))))
                        except FileNotFoundError:
                            pass
    except OSError:
        return None
    return hashlib.sha1(json.dumps(sorted(found)).encode()).hexdigest()[:12]


def _jsonable(value):
    try:
        json.dumps(value)
        return value
    except TypeError:
        return json.loads(json.dumps(value, default=str))


def _prior_columns(files, columns, element_attrs, attrs, stats) -> dict:
    """Columns that can act as proseg's prior: a cell assignment and a nuclear flag, with sampled coverage. When the
    attrs name the assignment column (transcript_to_cell_assignment) only it is offered, with the points' instance_key:
    other "cell" columns are not sample-wide cell ids (CosMx's cell_ID repeats in every FOV, its cell column glues each
    FOV's unassigned transcripts into one cell). A nucleus flag (NUCLEUS_FLAGS: CosMx's CellComp) is never a cell
    column. The attrs' "no cell" mark is checked against the data: when (almost) no transcript has it and -1 holds a
    background-sized share, -1 is used, and the entry's `note` says so (inspect's notes)."""
    instance = element_attrs.get("instance_key")

    def usable(name):
        return name in columns and name not in ("x", "y", "z") and name not in NUCLEUS_FLAGS and \
            any(t in columns[name] for t in ("string", "int", "dictionary"))
    assignment = attrs.get("transcript_to_cell_assignment") or []
    named = str(assignment[0]) if isinstance(assignment, (list, tuple)) and len(assignment) == 2 else None
    if usable(named):
        cell_candidates = list(dict.fromkeys(n for n in (named, instance) if usable(n)))
    else:
        cell_candidates = [n for n in columns if usable(n) and ("cell" in n.lower() or n == instance)]
    cell_candidates.sort(key=lambda n: (n != named, n != instance, n))
    nucleus_candidates = [n for n in columns if n in NUCLEUS_FLAGS]
    sample = _sample_rows(files, cell_candidates + nucleus_candidates)
    cells = []
    for name in cell_candidates:
        values = _as_strings(sample.column(name)) if sample is not None and name in sample.column_names else np.array([], object)
        share = (lambda token: float(np.mean(values == token)) if len(values) else 0.0)
        note = None
        if name == named:
            token = str(assignment[1])
            if token != "-1" and share(token) < 0.01 and share("-1") > BACKGROUND_SHARE:
                note = {"code": "unassigned_token", "text": f"In the transcripts' {name} column, -1 marks no cell",
                        "detail": f"The sample says {token} marks transcripts in no cell, but only {share(token):.1%} "
                                  f"have it, while -1 holds {share('-1'):.0%}, as transcripts in no cell do. This app "
                                  f"reads -1 as no cell."}
                token = "-1"
        elif re.fullmatch(r"u?int\d+", columns[name]) or re.match(r"dictionary<values=u?int", columns[name]):
            # Integer ids carry no marker: -1 when present, 0 only when it holds a background-sized share of the
            # transcripts (a real cell 0 holds ~1/ncells of them), otherwise every id is a cell. (Categorical text,
            # "dictionary<values=string, indices=int32>", is text: its marker is UNASSIGNED.)
            token = "-1" if np.any(values == "-1") else ("0" if share("0") > BACKGROUND_SHARE else "")
        else:
            token = UNASSIGNED if np.any(values == UNASSIGNED) else ""
        values = np.where(np.equal(values, None), token, values)
        assigned = float(np.mean(values != token)) if len(values) else None
        cells.append({"column": name, "unassigned": token, "assigned_fraction": assigned, **({"note": note} if note else {})})
    nuclei = []
    for name in nucleus_candidates:
        token = "Nuclear" if name == "CellComp" else "1"
        if name in stats["max"] and stats["max"][name] == 0 and stats.get("complete", {}).get(name):
            fraction = 0.0      # every row group says so: no transcript is flagged
        elif sample is not None and name in sample.column_names:
            fraction = float(np.mean(_flags(sample.column(name)) == token))
        else:
            fraction = None
        nuclei.append({"column": name, "nuclear": token, "nuclear_fraction": fraction})
    return {"cells": cells, "nuclei": nuclei}


def _suggest_prior(shapes: list[dict], points: dict | None) -> dict:
    """A sensible default: the instrument's own cell and nucleus shapes when present, else the transcripts' own
    assignment columns. The user can change both. Never an element the engine cannot serve (its KEY), and the app's
    own saves (app_save: a stain save may cover one box only) only when the sample has no other cell outlines. None
    for either without transcripts (`points` None: proseg does not run there)."""
    if points is None:
        return {"cells": None, "nuclei": None}
    usable = [s for s in shapes if KEY.fullmatch(s["key"]) and s.get("to_frame") and s["count"] > 1
              and any("Polygon" in g for g in s["geometry"])]
    rank = lambda s: (bool(s.get("app_save")), 0 if "xenium" in s["key"].lower() else 1, s["key"])
    nuclei = sorted((s for s in usable if "nucle" in s["key"].lower()), key=rank)
    cells = sorted((s for s in usable if "cell" in s["key"].lower() and "nucle" not in s["key"].lower()), key=rank)
    columns = (points or {}).get("prior_columns", {})
    cell_column = next((c for c in columns.get("cells", []) if (c["assigned_fraction"] or 0) > 0), None)
    nucleus_column = next((c for c in columns.get("nuclei", []) if (c["nuclear_fraction"] or 0) > 0), None)
    prior = {"cells": None, "nuclei": None}
    if cells:
        prior["cells"] = {"kind": "shapes", "key": cells[0]["key"]}
    elif cell_column:
        prior["cells"] = {"kind": "column", "column": cell_column["column"], "unassigned": cell_column["unassigned"]}
    if nuclei:
        prior["nuclei"] = {"kind": "shapes", "key": nuclei[0]["key"]}
    elif nucleus_column and prior["cells"] and prior["cells"]["kind"] == "column":
        prior["nuclei"] = {"kind": "column", "column": nucleus_column["column"], "nuclear": nucleus_column["nuclear"]}
    return prior


def _image_entry(path: str, key: str, image, matrix, subject: str = "the transcripts") -> dict:
    """Pyramid levels (zarr array path, shape, pixel origin/size in the image's intrinsic frame) and channels. Skipped
    (a HelperError with its reason) for one the app cannot place: rotated or mirrored relative to `subject` (what the
    working frame is), or whose levels do not match its metadata."""
    attrs = _attributes(Path(path) / "images" / key)
    multiscales = (attrs.get("ome") or attrs).get("multiscales") or []
    datasets = multiscales[0]["datasets"] if multiscales else []
    axes = [a["name"] if isinstance(a, dict) else a for a in (multiscales[0].get("axes", []) if multiscales else [])]
    arrays = [image[k]["image"] for k in sorted(image.keys(), key=lambda k: int(re.sub(r"\D", "", k) or 0))] \
        if hasattr(image, "keys") else [image]
    if len(datasets) != len(arrays):
        raise Skipped(f"Image '{key}': {len(arrays)} scales but {len(datasets)} datasets",
                      "its levels do not match its metadata")
    levels = []
    for dataset, array in zip(datasets, arrays):
        xs, ys = np.asarray(array.coords["x"].values, float), np.asarray(array.coords["y"].values, float)
        dx = float(xs[1] - xs[0]) if xs.size > 1 else 1.0
        dy = float(ys[1] - ys[0]) if ys.size > 1 else 1.0
        levels.append({"path": f"images/{key}/{dataset['path']}", "shape": [int(s) for s in array.shape],
                       "dims": list(array.dims), "origin": [float(xs[0] - dx / 2), float(ys[0] - dy / 2)], "pixel": [dx, dy]})
    first = arrays[0]
    channels = [str(c) for c in first.coords["c"].values] if "c" in first.dims else ["Channel 0"]
    if matrix is not None and abs(matrix[0, 1]) + abs(matrix[1, 0]) > 1e-9 * (abs(matrix[0, 0]) + abs(matrix[1, 1])):
        raise Skipped(f"Image '{key}' is rotated relative to {subject}: not supported", f"rotated relative to {subject}")
    if matrix is not None and (matrix[0, 0] <= 0 or matrix[1, 1] <= 0):
        raise Skipped(f"Image '{key}' is mirrored relative to {subject}: not supported", f"mirrored relative to {subject}")
    return {"key": key, "channels": channels, "dtype": str(first.dtype), "axes": axes or list(first.dims),
            "levels": levels, "to_frame": matrix.tolist() if matrix is not None else None}


def _table_entry(folder: Path) -> dict:
    """Name, size and annotation target of a table, from its zarr metadata (the AnnData itself is not loaded)."""
    import zarr
    entry = {"key": folder.name, "region": None, "instance_key": None, "region_key": None}
    try:
        group = zarr.open_group(str(folder), mode="r")
        for name in ("region", "region_key", "instance_key"):
            entry[name] = group.attrs.get(name)
        if "X" in group:
            x = group["X"]
            shape = x.attrs.get("shape") if not hasattr(x, "shape") else x.shape
            if shape is not None:
                entry["n_obs"], entry["n_vars"] = int(shape[0]), int(shape[1])
    except Exception as exc:     # listed by name only
        entry["error"] = f"{type(exc).__name__}: {exc}"
    return entry


# ---- shapes -----------------------------------------------------------------------------------------------------
def _aligned_shapes(sdata, key: str, points_key: str | None):
    """A shapes element in the working frame (sopa's to_intrinsic alignment; without transcripts, spatialdata's
    transform into the frame's coordinate system: _image_frame), circles (points with a radius) turned into polygons,
    empty geometries left out. `sdata` holds the images too when there are no transcripts (_selection)."""
    import geopandas as gpd
    import shapely
    import spatialdata
    from sopa.utils import to_intrinsic
    if key not in sdata.shapes:
        raise HelperError(f"Unknown shapes element '{key}'")
    if points_key:
        gdf = to_intrinsic(sdata, key, points_key)
    else:
        transformation = _transformation(sdata, sdata.shapes[key], _frame_of(sdata, None))
        if transformation is None:
            raise HelperError(f"The shapes '{key}' share no coordinate system with the sample's images")
        gdf = spatialdata.transform(sdata.shapes[key], transformation=transformation, maintain_positioning=True)
    circles = (gdf.geometry.geom_type == "Point").to_numpy()
    if "radius" in gdf.columns and circles.any():
        # Through shapely itself: geopandas' buffer passes its own quad_segs (from `resolution`), so a second one is a
        # TypeError (geopandas 1.1). 6 segments a quarter circle; other geometries are not grown.
        geometry = gdf.geometry.to_numpy().copy()
        radius = np.asarray(gdf["radius"], dtype=float)
        geometry[circles] = shapely.buffer(geometry[circles], radius[circles], quad_segs=6)
        gdf = gdf.copy()
        gdf.geometry = gpd.GeoSeries(geometry, index=gdf.index, crs=gdf.crs)
    keep = gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"]) & ~gdf.geometry.is_empty
    return gdf[keep]


def _selection(points_key: str | None, *kinds: str) -> tuple[str, ...]:
    """What to read for a command working in the frame: the transcripts' element, or, without transcripts, every kind
    inspect reads (shapes, images and labels): each can link the main image to a coordinate system (_to_system), so a
    command that read fewer than inspect did could settle on another frame (an image that reaches micrometres only
    through a labels element: its outlines and stain saves were placed in pixels)."""
    return ("points", *kinds) if points_key else tuple(dict.fromkeys((*kinds, "shapes", "images", "labels")))


def shapes(args: dict) -> dict:
    import shapely
    path, key, points_key, out = str(args["path"]), args["key"], args.get("points_key"), Path(args["out"])
    progress(f"Reading shapes '{key}'…", 0.2)
    sdata = _read(path, _selection(points_key, "shapes"))
    gdf = _aligned_shapes(sdata, key, points_key)
    progress("Converting outlines…", 0.6)
    geometries = gdf.geometry.to_numpy()
    parts, owner = shapely.get_parts(geometries, return_index=True)
    rings = shapely.get_exterior_ring(parts)
    coords, ring_index = shapely.get_coordinates(rings, return_index=True)
    counts = np.bincount(ring_index, minlength=len(parts))
    offsets = np.zeros(len(parts) + 1, np.uint32)
    np.cumsum(counts, out=offsets[1:])
    bounds = shapely.bounds(parts).astype(np.float32)
    ids = [str(i) for i in gdf.index]
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(out.name + f".{os.getpid()}.tmp")
    with open(temporary, "wb") as stream:
        for array in (bounds, offsets, owner.astype(np.uint32), coords.astype(np.float32)):
            stream.write(np.ascontiguousarray(array).astype(array.dtype.newbyteorder("<")).tobytes())
    os.replace(temporary, Path(str(out) + ".bin"))
    meta = {"schema": 1, "key": key, "polygons": len(ids), "parts": int(len(parts)), "vertices": int(len(coords)),
            "extent": [float(bounds[:, 0].min()), float(bounds[:, 1].min()), float(bounds[:, 2].max()), float(bounds[:, 3].max())] if len(parts) else None,
            "layout": "little-endian: bounds f32[parts*4] (x0,y0,x1,y1), offsets u32[parts+1], owner u32[parts] (polygon index), "
                      "coords f32[vertices*2] (x,y): exterior rings in the working frame",
            "ids": ids}
    temporary = out.with_name(out.name + f".{os.getpid()}.json.tmp")
    temporary.write_text(json.dumps(meta))
    os.replace(temporary, Path(str(out) + ".json"))
    progress("Done", 1.0)
    return {k: v for k, v in meta.items() if k != "ids"}


# ---- prepare ----------------------------------------------------------------------------------------------------
def _region(region: dict, extent: list[float] | None):
    """(bbox or None for everything, prepared shapely polygon or None)."""
    import shapely
    kind = (region or {}).get("kind", "full")
    if kind == "full":
        return None, None
    if kind == "box":
        x0, y0, x1, y1 = (float(v) for v in region["box"])
        return (x0, y0, x1, y1), None
    if kind == "polygon":
        polygon = shapely.make_valid(shapely.Polygon(np.asarray(region["points"], float)))
        if polygon.is_empty or polygon.area <= 0:
            raise HelperError("The drawn region is empty")
        shapely.prepare(polygon)
        return tuple(float(v) for v in polygon.bounds), polygon
    raise HelperError(f"Unknown region kind '{kind}'")


def _assign(gdf, x: np.ndarray, y: np.ndarray, label: str) -> np.ndarray:
    """1 + row position of the polygon containing each point (0 = none), with sopa's own join, in parallel chunks."""
    import pandas as pd
    from sopa.spatial.join import _get_cell_id
    gdf = gdf.reset_index(drop=True)
    gdf.index.name = "index_right"      # _get_cell_id sets this too; done once here so threads never race on it
    result = np.zeros(len(x), np.int64)
    step = 1_000_000
    chunks = [(a, min(a + step, len(x))) for a in range(0, len(x), step)]
    done = [0]

    def work(chunk):
        a, b = chunk
        frame = pd.DataFrame({"x": x[a:b], "y": y[a:b]})
        result[a:b] = _get_cell_id(gdf, frame, unassigned_value=0).to_numpy()
        done[0] += 1
        progress(f"Assigning transcripts to {label}… {min(100, round(100 * done[0] / len(chunks)))} %", done[0] / len(chunks))

    with ThreadPoolExecutor(max_workers=min(16, os.cpu_count() or 4)) as pool:
        list(pool.map(work, chunks))
    return result


def prepare(args: dict) -> dict:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq
    import shapely
    path, points_key, out = str(args["path"]), args.get("points_key"), Path(args["out"])
    if not points_key:
        raise HelperError(NO_TRANSCRIPTS)
    prior = args.get("prior") or {}
    cells, nuclei = prior.get("cells"), prior.get("nuclei")
    if not cells and not nuclei:
        raise HelperError("Choose a prior: proseg starts from cells (or nuclei) given by shapes or a transcript column")
    files = _points_files(path, points_key)
    dataset = ds.dataset(files, format="parquet")
    names = set(dataset.schema.names)
    if not {"x", "y"} <= names:
        raise HelperError("The transcripts need x and y columns")
    element_attrs = _attributes(Path(path) / "points" / points_key).get("spatialdata_attrs", {})
    feature = _gene_column(dataset.schema.names, element_attrs)     # as inspect chose it
    bbox, polygon = _region(args.get("region"), None)
    wanted = ["x", "y", "z", feature, "qv", "fov_name", "transcript_id", "nucleus_distance", "codeword_index",
              "codeword_category", "is_gene"]
    for spec in (cells, nuclei):
        if spec and spec.get("kind") == "column":
            if spec["column"] not in names:
                raise HelperError(f"The transcripts have no column '{spec['column']}'")
            wanted.append(spec["column"])
    wanted = list(dict.fromkeys(c for c in wanted if c in names))
    progress("Reading the region's transcripts…", 0.05)
    started = time.monotonic()
    expression = None
    if bbox:
        x0, y0, x1, y1 = bbox
        expression = (ds.field("x") >= x0) & (ds.field("x") < x1) & (ds.field("y") >= y0) & (ds.field("y") < y1)
    table = dataset.to_table(columns=wanted, filter=expression, use_threads=True)
    progress(f"Read {table.num_rows:,} transcripts in {time.monotonic() - started:.0f} s", 0.3)
    import pyarrow.compute as pc0
    usable = pc0.and_(pc0.is_finite(pc0.fill_null(pc0.cast(table.column("x"), pa.float64()), float("nan"))),
                      pc0.is_finite(pc0.fill_null(pc0.cast(table.column("y"), pa.float64()), float("nan"))))
    usable = pc0.and_(usable, pc0.is_valid(table.column(feature)))
    dropped = table.num_rows - pc0.sum(pc0.cast(usable, pa.int64())).as_py() if table.num_rows else 0
    if dropped:
        table = table.filter(usable)
        progress(f"Skipped {dropped:,} transcripts without coordinates or gene", 0.3)
    x = table.column("x").to_numpy().astype(np.float64)
    y = table.column("y").to_numpy().astype(np.float64)
    if polygon is not None:
        keep = shapely.contains_xy(polygon, x, y)
        table, x, y = table.filter(pa.array(keep)), x[keep], y[keep]
    if table.num_rows == 0:
        raise HelperError("There are no transcripts in this region")

    sdata = None
    if any(spec and spec.get("kind") == "shapes" for spec in (cells, nuclei)):
        progress("Reading the prior shapes…", 0.35)
        sdata = _read(path, _selection(points_key, "shapes"))

    def shapes_layer(spec, label):
        gdf = _aligned_shapes(sdata, spec["key"], points_key)
        if bbox:
            gdf = gdf.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        ids = np.asarray([str(i) for i in gdf.index], dtype=object)
        return _assign(gdf, x, y, label), ids

    n = table.num_rows
    cell_id = np.full(n, UNASSIGNED, dtype=object)
    overlaps = np.zeros(n, np.uint8)
    if cells and cells.get("kind") == "column":
        values = _as_strings(table.column(cells["column"]))
        token = str(cells.get("unassigned", UNASSIGNED))
        missing = np.equal(values, None) | (values == "") if token != "" else np.equal(values, None)
        cell_id = np.where(missing | (values == token), UNASSIGNED, values).astype(object)
    elif cells and cells.get("kind") == "shapes":
        index, ids = shapes_layer(cells, f"cells ({cells['key']})")
        inside = index > 0
        cell_id[inside] = ids[index[inside] - 1]
    if nuclei and nuclei.get("kind") == "column":
        overlaps = (_flags(table.column(nuclei["column"])) == str(nuclei.get("nuclear", "1"))).astype(np.uint8)
    elif nuclei and nuclei.get("kind") == "shapes":
        index, ids = shapes_layer(nuclei, f"nuclei ({nuclei['key']})")
        overlaps = (index > 0).astype(np.uint8)
        if not cells:   # a nucleus-only prior: every nucleus seeds a cell
            inside = index > 0
            cell_id[inside] = ids[index[inside] - 1]
    if not cells:
        overlaps &= (cell_id != UNASSIGNED).astype(np.uint8)
    elif not nuclei:
        # A cells-only prior (e.g. Cellpose shapes, no nuclei). proseg seeds its starting cells from nuclear transcripts
        # only, so with none flagged it starts with 0 cells and panics (clustering: remainder with a divisor of zero).
        # Without compartment information proseg's own rule is "use anything assigned to the cell"
        # (--use-cell-initialization, which its Xenium parquet reader cannot take): every transcript inside a prior
        # cell seeds that cell.
        overlaps = (cell_id != UNASSIGNED).astype(np.uint8)
    seeded_from_cells = bool(cells) and not nuclei
    if cells and nuclei and not (overlaps.astype(bool) & (cell_id != UNASSIGNED)).any():
        # A pairing whose nuclei flag nothing inside the prior's cells here (an all-0 flag column, or no nucleus inside
        # this region's cells) would reach proseg with no seed at all: 0 cells, and it panics. Seed from the cells.
        overlaps = (cell_id != UNASSIGNED).astype(np.uint8)
        seeded_from_cells = True

    progress("Writing proseg's input…", 0.9)

    def column(name, kind, default):
        """`name` as `kind`, or `default` on every row (a value, or one per row) when the transcripts have no such
        column, or hold what `kind` cannot take without reading text as numbers (_castable): MERSCOPE's transcript_id
        is Ensembl text, which a cast fails on (ArrowInvalid), so it is numbered as a missing one is."""
        if name in table.column_names:
            array = table.column(name).combine_chunks()
            if pa.types.is_dictionary(array.type):
                array = array.dictionary_decode()
            if _castable(array.type, kind):
                return pc.cast(array, kind, safe=False)
        return pa.array(default if isinstance(default, np.ndarray) else np.full(n, default), kind)

    z = column("z", pa.float32(), 0.0)
    xenium = pa.table({
        "transcript_id": column("transcript_id", pa.uint64(), np.arange(n, dtype=np.uint64)),
        "cell_id": pa.array(cell_id.tolist(), pa.large_string()),
        "overlaps_nucleus": pa.array(overlaps),
        "feature_name": column(feature, pa.large_string(), ""),
        "x_location": pa.array(x.astype(np.float32)), "y_location": pa.array(y.astype(np.float32)), "z_location": z,
        "qv": column("qv", pa.float32(), 40.0),
        "fov_name": column("fov_name", pa.large_string(), "0"),
        "nucleus_distance": column("nucleus_distance", pa.float32(), 0.0),
        "codeword_index": column("codeword_index", pa.int32(), 0),
        "codeword_category": column("codeword_category", pa.large_string(), "predesigned_gene"),
        "is_gene": column("is_gene", pa.bool_(), True),
    })
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(out.name + f".{os.getpid()}.tmp")
    pq.write_table(xenium, temporary, compression="zstd", row_group_size=1 << 20)
    os.replace(temporary, out)
    assigned = int(np.count_nonzero(cell_id != UNASSIGNED))
    progress("Done", 1.0)
    return {"transcripts": n, "assigned": assigned, "nuclear": int(overlaps.sum()), "skipped": int(dropped),
            "seeded_from": "cells" if seeded_from_cells else "nuclei",
            "prior_cells": int(len(set(cell_id[cell_id != UNASSIGNED].tolist()))) if assigned else 0,
            "bbox": [float(x.min()), float(y.min()), float(x.max()), float(y.max())],
            "seconds": round(time.monotonic() - started, 1)}


# ---- count (Compare) ------------------------------------------------------------
def _gene_codes(column) -> tuple[np.ndarray, list[str]]:
    """(code per transcript, names): a gene column as dictionary codes (the table's dictionaries unified first)."""
    import pyarrow as pa
    import pyarrow.compute as pc
    array = column.combine_chunks() if hasattr(column, "combine_chunks") else column
    if not pa.types.is_dictionary(array.type):
        array = pc.dictionary_encode(array)
    names = [str(v) for v in array.dictionary.to_pylist()]
    return np.asarray(array.indices.to_numpy(zero_copy_only=False), np.int64), names


def _table_counts(path: Path, table: str, key: str, names: list[str]):
    """A table's own counts of the shapes element `key` (its X): the rows of tables/<table> that hold the element's
    cells (all of them in a table of `key` alone; in one of several elements, those whose region_key names `key`), each
    given to the outline its instance id names (`names`: the outlines' ids as text, the element's index). -> (CSR of
    whole counts: one row per outline of `names` the table holds, in their order; those outlines' places in `names`; the
    table's gene names). HelperError in plain words: no such table, one that holds no cells of `key` or does not say
    which outline each of its rows is, no counts or counts that are not whole numbers of 0 or more, a cell named twice."""
    import pandas as pd
    import zarr
    from scipy import sparse
    try:
        from anndata.io import read_elem
    except ImportError:                  # anndata before 0.11
        from anndata.experimental import read_elem
    folder = path / "tables" / table
    if not KEY.fullmatch(table) or not folder.is_dir():
        raise HelperError(f"The sample has no table '{table}'")
    attrs = _attributes(folder)
    region = attrs.get("region")
    regions = [str(r) for r in (region if isinstance(region, list) else [] if region is None else [region])]
    if key not in regions:
        raise HelperError(f"The table '{table}' holds no cells of '{key}'")
    group = zarr.open_group(str(folder), mode="r")
    obs, instance, region_key = group["obs"], attrs.get("instance_key"), attrs.get("region_key")
    if not instance or instance not in obs or len(regions) > 1 and (not region_key or region_key not in obs):
        raise HelperError(f"The table '{table}' doesn't say which outline each of its cells is")
    if "X" not in group:
        raise HelperError(f"The table '{table}' holds no counts")
    ids = np.asarray(read_elem(obs[instance]), dtype=object).astype(str)
    rows = np.arange(len(ids))
    if len(regions) > 1:                 # a table of several elements: its rows of this one
        rows = np.flatnonzero(np.asarray(read_elem(obs[region_key]), dtype=object).astype(str) == key)
    held = pd.Index(ids[rows])
    if not held.is_unique:
        raise HelperError(f"The table '{table}' names a cell of '{key}' twice")
    found = held.get_indexer(pd.Index(names))          # each outline's row among `rows`, -1 when it holds none
    places = np.flatnonzero(found >= 0)
    matrix = sparse.csr_matrix(read_elem(group["X"]))[rows[found[places]]]
    matrix.eliminate_zeros()
    values = matrix.data
    if not np.isfinite(values).all() or (values < 0).any() or (values > np.iinfo(np.uint32).max).any() \
            or not np.equal(values, np.rint(values)).all():
        raise HelperError(f"The table '{table}' doesn't hold whole transcript counts")
    genes = [str(g) for g in read_elem(group["var"]).index]
    if len(genes) != matrix.shape[1]:
        raise HelperError(f"The table '{table}' names {len(genes)} genes for {matrix.shape[1]} columns of counts")
    return matrix, places, genes


def count(args: dict) -> dict:
    """The element's counts as <out>.npz (uncompressed, little-endian, no pickles): the cells x genes matrix as CSR
    `data` (u32 counts), `indices` (i32 gene), `indptr` (i64), `shape`; per row (cell) its `ids` (the element's index as
    text), `index` (i32: its position in the element, the exported outlines' order: helper `shapes`) and outline centre
    `x`, `y` (f32 µm, working frame); `genes` (the genes counted at least once, as the transcripts or the table name
    them), `count` (whose count it is: "proseg" or "xenium", a table's own; "outlines") and `rule` (count_rule, or
    "table:<its key>").
    With `table` (a table annotating the element that holds its own count: engine/zarr_samples.py count_source) the
    counts are that table's X, whole numbers (`by` says whose: TABLE_COUNTS), each of its rows the outline its instance
    id names (_table_counts): an outline it holds no row of gets none. Else, for a segmentation with no count of its own,
    the transcripts inside its outlines: those of quality `min_qv` (COUNT_MIN_QV) or more, of real genes (is_gene when
    the transcripts have it, else names NOT_GENES does not match) and, with `exclude` (the run's excluded-genes pattern),
    none whose gene name it matches anywhere (re.search, as proseg's --excluded-genes; under a time limit: _matching).
    Each is joined with EVERY outline of the element, so it goes where a count of the whole element puts it: where
    outlines overlap, sopa's join takes the one its spatial index finds first, an order made by every outline in the
    index (one joined alone would always win). An outline holding no transcript keeps an empty row.
    Rows are every outline of the element (with a table, every one it holds), or with `box` those whose centre lies in
    it (CENTRE_MARGIN_UM): a transcript an outline outside the box holds counts in no row."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.dataset as ds
    import shapely
    from scipy import sparse
    path, key, points_key, out = str(args["path"]), args["key"], args.get("points_key"), Path(args["out"])
    if not points_key:
        raise HelperError(NO_TRANSCRIPTS)
    box = args.get("box")
    table = args.get("table")
    by = args.get("by") or ("outlines" if table is None else None)
    if by not in (*TABLE_COUNTS, "outlines") or (by == "outlines") != (table is None):
        raise HelperError("A column's own count is a table's, proseg's or Xenium's; without one its outlines count it")
    if table is None:
        min_qv = _min_qv(COUNT_MIN_QV if args.get("min_qv") is None else args["min_qv"])
        exclude = args.get("exclude")
        excluded = None if exclude is None else _excluded(exclude)
        rule = count_rule(min_qv, exclude)
    else:
        rule = f"table:{table}"
    started = time.monotonic()
    seconds = {}
    progress(f"Reading shapes '{key}'…", 0.05)
    gdf = _aligned_shapes(_read(path, ("points", "shapes")), key, points_key)
    centres = shapely.centroid(gdf.geometry.to_numpy())
    cx, cy = shapely.get_x(centres), shapely.get_y(centres)
    index = np.arange(len(gdf))
    if box is not None:
        x0, y0, x1, y1 = (float(v) for v in box)
        m = CENTRE_MARGIN_UM
        index = np.flatnonzero((cx >= x0 - m) & (cx <= x1 + m) & (cy >= y0 - m) & (cy <= y1 + m))
    cells = gdf.iloc[index]
    seconds["shapes"] = round(time.monotonic() - started, 2)

    if table is not None:
        progress(f"Reading the counts of table '{table}'…", 0.3)
        read_at = time.monotonic()
        matrix, places, genes = _table_counts(Path(path), table, key, [str(i) for i in cells.index])
        index, cells = index[places], cells.iloc[places]
        used = np.bincount(matrix.indices, minlength=len(genes)) > 0 if len(genes) else np.zeros(0, bool)
        matrix = matrix[:, np.flatnonzero(used)]
        matrix.sort_indices()
        filters = None
        transcripts = assigned = int(round(float(matrix.sum())))
        seconds["read"] = round(time.monotonic() - read_at, 2)
    else:
        files = _points_files(path, points_key)
        dataset = ds.dataset(files, format="parquet")
        names = set(dataset.schema.names)
        if not {"x", "y"} <= names:
            raise HelperError("The transcripts need x and y columns")
        element_attrs = _attributes(Path(path) / "points" / points_key).get("spatialdata_attrs", {})
        feature = _gene_column(dataset.schema.names, element_attrs)     # as inspect chose it
        filters = {"qv": "qv" in names, "is_gene": "is_gene" in names, "min_qv": min_qv, "exclude": exclude}
        x = y = np.zeros(0)
        codes, genes = np.zeros(0, np.int64), []
        if len(cells):
            # Only the transcripts under these outlines' bounds, of quality min_qv or more and of real genes: parquet row
            # groups whose statistics say otherwise are never read.
            bx0, by0, bx1, by1 = (float(v) for v in cells.total_bounds)
            expression = (ds.field("x") >= bx0) & (ds.field("x") <= bx1) & (ds.field("y") >= by0) & (ds.field("y") <= by1)
            if filters["qv"]:
                expression &= ds.field("qv") >= min_qv
            if filters["is_gene"]:
                expression &= ds.field("is_gene")
            progress(f"Reading the transcripts under {len(cells):,} outlines…", 0.15)
            read_at = time.monotonic()
            arrow = dataset.to_table(columns=["x", "y", feature], filter=expression, use_threads=True)
            usable = pc.and_(pc.is_valid(arrow.column(feature)),
                             pc.and_(pc.is_finite(pc.cast(arrow.column("x"), pa.float64())),
                                     pc.is_finite(pc.cast(arrow.column("y"), pa.float64()))))
            arrow = arrow.filter(usable).unify_dictionaries()
            codes, genes = _gene_codes(arrow.column(feature))
            x = arrow.column("x").to_numpy().astype(np.float64)
            y = arrow.column("y").to_numpy().astype(np.float64)
            del arrow
            real = np.ones(len(genes), bool)
            if not filters["is_gene"]:       # no is_gene column: proseg --xenium's own exclusions by name
                real &= np.array([not NOT_GENES.match(g) for g in genes], bool)
            if excluded is not None and len(genes):   # the run's own pattern, as proseg leaves those genes out of its input
                real &= ~_matching(exclude, list(genes))
            if not real.all():
                keep = real[codes]
                x, y, codes = x[keep], y[keep], codes[keep]
            seconds["read"] = round(time.monotonic() - read_at, 2)
        progress(f"Read {len(x):,} transcripts", 0.3)
        assign_at = time.monotonic()
        owner = _assign(gdf, x, y, f"cells ({key})") if len(x) else np.zeros(0, np.int64)   # 1 + position, 0: none
        seconds["assign"] = round(time.monotonic() - assign_at, 2)

        progress("Counting…", 0.9)
        row_of = np.full(len(gdf) + 1, -1, np.int64)      # 1 + position in the element -> its row (-1: none, or not kept)
        row_of[index + 1] = np.arange(len(index))
        found = row_of[owner]
        hit = found >= 0
        rows, gene = found[hit].astype(np.int32), codes[hit]
        used = np.bincount(gene, minlength=len(genes)) > 0 if len(genes) else np.zeros(0, bool)
        remap = (np.cumsum(used) - 1).astype(np.int32)
        matrix = sparse.coo_matrix((np.ones(rows.size, np.uint32), (rows, remap[gene] if gene.size else gene.astype(np.int32))),
                                   shape=(len(cells), int(used.sum()))).tocsr()
        matrix.sum_duplicates()
        transcripts, assigned = int(len(x)), int(hit.sum())
    build_at = time.monotonic()
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(out.name + f".{os.getpid()}.tmp")
    with open(temporary, "wb") as stream:
        np.savez(stream, data=matrix.data.astype("<u4"), indices=matrix.indices.astype("<i4"),
                 indptr=matrix.indptr.astype("<i8"), shape=np.array(matrix.shape, "<i8"),
                 ids=np.array([str(i) for i in cells.index], dtype=str), index=index.astype("<i4"),
                 x=cx[index].astype("<f4"), y=cy[index].astype("<f4"),
                 genes=np.array([g for g, u in zip(genes, used) if u], dtype=str), rule=np.array(rule),
                 count=np.array(by))
    os.replace(temporary, out)
    seconds["build"] = round(time.monotonic() - build_at, 2)
    seconds["total"] = round(time.monotonic() - started, 2)
    progress("Done", 1.0)
    return {"key": key, "rule": rule, "count": by, "table": table, "box": box, "cells": int(len(cells)),
            "genes": int(used.sum()), "transcripts": transcripts, "assigned": assigned, "nnz": int(matrix.nnz),
            "filters": filters, "seconds": seconds}


# ---- segmentations (how each shapes element was made) ----------------------------------------------------------------
def _folders(path: Path, kind: str) -> list[Path]:
    folder = path / kind
    return sorted(p for p in folder.iterdir() if p.is_dir() and not p.name.startswith(".")) if folder.is_dir() else []


def _table_methods(folder: Path) -> dict:
    """A table's annotation target (region as a list), keys, whether an app's proseg save wrote it (_app_table), the
    run that save came from (`run`: its provenance's, None otherwise; the older runs' "saved already" check), whether
    it holds proseg's own count (`proseg_count`: its uns names one of PROSEG_TABLE_KEYS: an app's save, or proseg's own
    output read by sopa) and `methods` {region: {obs segmentation_method: cells}} (None without that column), read with
    anndata's own reader."""
    import pandas as pd
    import zarr
    attrs = _attributes(folder)
    region = attrs.get("region")
    regions = [str(r) for r in (region if isinstance(region, list) else [] if region is None else [region])]
    provenance = _app_table(folder)
    entry = {"key": folder.name, "region": regions, "region_key": attrs.get("region_key"),
             "instance_key": attrs.get("instance_key"), "proseg": provenance is not None,
             "proseg_count": any((folder / "uns" / key).is_dir() for key in PROSEG_TABLE_KEYS), "run": None,
             "methods": None}
    try:
        try:
            from anndata.io import read_elem
        except ImportError:                  # anndata before 0.11
            from anndata.experimental import read_elem
        if entry["proseg"]:
            try:
                live = zarr.open_group(str(provenance), mode="r")
                run = read_elem(live["run"]) if "run" in live else None
                entry["run"] = run if isinstance(run, str) else None
            except Exception:                # listed without its run
                pass
        obs = zarr.open_group(str(folder / "obs"), mode="r")
        if "segmentation_method" not in obs:
            return entry
        methods = np.asarray(read_elem(obs["segmentation_method"]), dtype=object)
        if len(regions) > 1 and entry["region_key"] in obs:
            where = np.asarray(read_elem(obs[entry["region_key"]]), dtype=object)
        else:
            where = np.full(len(methods), regions[0] if regions else "", dtype=object)
        found: dict = {}
        for place, method in zip(where.tolist(), methods.tolist()):
            if not pd.isna(method):
                tally = found.setdefault(str(place), {})
                tally[str(method)] = tally.get(str(method), 0) + 1
        entry["methods"] = found
    except Exception as exc:                 # listed without its methods
        entry["error"] = f"{type(exc).__name__}: {exc}"
    return entry


def segmentations(args: dict) -> dict:
    """{schema, shapes: [{key, app: {kind, method, model, run, complete, done, total} | None}], tables:
    [_table_methods]}: the zarr's own record of what made each element, for the engine to name it."""
    path = Path(args["path"]).resolve()
    shapes = []
    for folder in _folders(path, "shapes"):
        live = _app_record(_attributes(folder))
        app = None
        if live is not None:                 # an app's stain save (its record on the shapes group)
            done = live.get("progress") if isinstance(live.get("progress"), dict) else {}
            app = {**{k: _jsonable(live.get(k)) for k in ("kind", "method", "model", "run")},
                   "resolution": _jsonable(live.get("resolution")),
                   **{k: _jsonable(done.get(k)) for k in ("complete", "done", "total")}}
        shapes.append({"key": folder.name, "app": app})
    return {"schema": 1, "shapes": shapes, "tables": [_table_methods(folder) for folder in _folders(path, "tables")]}


# ---- save ----------------------------------------------------------------------------------------------------
BACKUPS = names.BACKUPS                  # where a save sets aside what it replaces (the previous version's name, kept)
CLOCK_SLACK_S = 5.0        # a process that started this long after a backup folder's stamp did not make that folder


LOCK_DIR_ENV = names.ZARR_LOCK_DIR_ENV   # a private folder (tests, the harness's engines): then the only one used
LOCK_STALE_S = 3600.0                    # a lock file nobody has opened for this long, and nobody holds, goes


def _lock_dirs(create: bool = True) -> list[Path]:
    """The folders of the save locks shared by every engine and helper of this user, in the order a save takes them
    (names.ZARR_LOCK_FOLDERS: the previous version's, then this version's), or the private folder LOCK_DIR_ENV names,
    alone. `create`: made when missing (a save); else only named (pruning and checks create nothing). RuntimeError for
    a private folder named the previous version's way, before anything is created."""
    if problem := names.retired_isolation():
        raise RuntimeError(problem)
    private = os.environ.get(LOCK_DIR_ENV)
    folders = [Path(private)] if private else [names.home_path(folder) for folder in names.ZARR_LOCK_FOLDERS]
    if create:
        for folder in folders:
            folder.mkdir(parents=True, exist_ok=True)
    return folders


def _lock_paths(path: Path, create: bool = True) -> list[Path]:
    """The zarr store's lock file in each lock folder (_lock_dirs), in the order a save takes them."""
    name = hashlib.sha1(str(path).encode()).hexdigest() + ".lock"
    return [folder / name for folder in _lock_dirs(create)]


@contextmanager
def _held(target: Path):
    """One lock file held exclusively (created, or truncated: the new mtime tells prune_locks it is in use)."""
    while True:
        handle = open(target, "w")
        try:
            try:
                portable.lock(handle, block=False)
            except BlockingIOError:
                progress("Waiting for another save into this sample to finish…")
                portable.lock(handle)
            # prune_locks may have removed the file between our open and our lock: a lock on that orphan excludes
            # nobody who opens the path now. Only the file still at the path counts.
            mine, current = os.fstat(handle.fileno()), os.stat(target)
            if (mine.st_dev, mine.st_ino) == (current.st_dev, current.st_ino):
                break
        except FileNotFoundError:
            pass
        except BaseException:
            handle.close()
            raise
        handle.close()
    try:
        yield
    finally:
        if portable.WINDOWS:
            portable.unlock(handle)        # Windows lets go of a lock at close only "when it can"
        handle.close()


@contextmanager
def _store_lock(path: Path):
    """The zarr store's lock files (_lock_paths), each held in turn for the whole block, let go in reverse order."""
    with ExitStack() as held:
        for target in _lock_paths(path):
            held.enter_context(_held(target))
        yield


def prune_locks(older_than_s: float = LOCK_STALE_S) -> list[str]:
    """Remove the lock files of stores that no save has opened for `older_than_s` (one per store ever saved into,
    tests' and fixtures' throwaway copies included) and that nobody holds now, in every lock folder (_lock_dirs, none
    created): taken without blocking, checked again under the lock, unlinked while held. A save that opened the file
    just before notices (_held) and retries; the previous version's saves do the same."""
    removed = []
    for folder in _lock_dirs(create=False):
        for path in folder.glob("*.lock"):
            try:
                if time.time() - path.stat().st_mtime < older_than_s:
                    continue
                fd = os.open(path, os.O_RDONLY | portable.O_NOFOLLOW)  # read-only: opening must not refresh its mtime
            except OSError:
                continue
            try:
                portable.lock(fd, block=False)
                mine, current = os.fstat(fd), os.stat(path)
                if (mine.st_dev, mine.st_ino) == (current.st_dev, current.st_ino) and \
                        time.time() - mine.st_mtime >= older_than_s:
                    path.unlink()
                    removed.append(path.name)
            except OSError:                 # held (BlockingIOError), or gone meanwhile
                pass
            finally:
                os.close(fd)
    return removed


def _elements(path: Path) -> dict[str, str]:
    """Element name -> kind, as on disk."""
    found = {}
    for kind in ("images", "labels", "points", "shapes", "tables"):
        for child in (path / kind).glob("*") if (path / kind).is_dir() else []:
            if child.is_dir():
                found[child.name] = kind
    return found


def _write_json(target: Path, value: dict):
    """`value` at `target`, whole or not at all (a temporary file renamed over it), on the disk before this returns:
    the save journal must outlive a power cut that follows the move it lists."""
    temporary = target.with_name(target.name + ".tmp")
    with open(temporary, "w") as stream:
        stream.write(json.dumps(value))
        stream.flush()
        os.fsync(stream.fileno())
    portable.replace(temporary, target)
    try:                                   # the rename itself (some filesystems cannot sync a folder: nothing to do)
        folder = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(folder)
        finally:
            os.close(folder)
    except OSError:
        pass


def _consolidate(path: Path):
    try:
        from spatialdata._io.io_zarr import _write_consolidated_metadata
        _write_consolidated_metadata(path)
    except Exception as exc:     # reading still works through the element folders
        progress(f"Could not refresh the consolidated metadata: {exc}")


def _consolidated(path: Path) -> bool:
    """Whether the zarr `path` holds consolidated metadata, as spatialdata's has_consolidated_metadata tells: its
    write_element then consolidates it again after every element, whatever a save's `consolidate` says."""
    try:
        import zarr
        return bool(getattr(zarr.open_group(str(path), mode="r").metadata, "consolidated_metadata", None))
    except Exception:
        return False


def _backup_folder(path: Path) -> re.Pattern:
    """A save's backup folder of the zarr `path`, matched whole: <zarr folder name>-<date>-<time>-<pid of the save>.
    Every zarr in a folder shares its .proseg-live-backups, so a zarr whose name begins like another's ("x.zarr" and
    "x.zarr-copy") must never take the other's journal for its own (re.escape: "s[1].zarr" is a name, not a glob)."""
    return re.compile(re.escape(path.name) + r"-(?P<stamp>\d{8}-\d{6})-(?P<pid>\d+)")


def _stamp_time(stamp: str) -> float | None:
    """A backup folder's stamp (time.strftime "%Y%m%d-%H%M%S", local time) in seconds since the epoch, read the later
    way where summer time could make it two (never earlier than the save began); None when it is not a stamp."""
    try:
        parts = time.strptime(stamp, "%Y%m%d-%H%M%S")
    except ValueError:
        return None
    readings = []
    for summer in (0, 1):
        try:
            readings.append(time.mktime((*parts[:8], summer)))
        except (OverflowError, ValueError):
            continue
    return max(readings) if readings else None


def _boot() -> str | None:
    return portable.boot_id()             # Linux: /proc/sys/kernel/random/boot_id, as before


def _process(pid: int) -> tuple[str, int] | None:
    """(state, start time in clock ticks after boot) of a process, from /proc/<pid>/stat; None when unreadable. macOS
    and Windows, which have no /proc: (state, start time in seconds since the epoch), from psutil (a journal's start is
    only ever compared on the computer that wrote it)."""
    if not portable.LINUX:
        try:
            import psutil
            process = psutil.Process(pid)
            status = process.status()
            return ("Z" if status == psutil.STATUS_ZOMBIE else "X" if status == psutil.STATUS_DEAD else "R",
                    process.create_time())
        except Exception:
            return None
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        fields = stat[stat.rindex(")") + 2:].split()       # after the command name, which may hold spaces
        return fields[0], int(fields[19])
    except (OSError, ValueError, IndexError):
        return None


def _leader(pid: int) -> bool | None:
    """Whether `pid` names a process, not one of another process's threads (a thread's id answers kill and /proc as a
    pid does); None when this cannot be read (on macOS and Windows a thread's id is never taken for a process's)."""
    if not portable.LINUX:
        return None
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("Tgid:"):
                return int(line.split()[1]) == pid
    except (OSError, ValueError, IndexError):
        return None
    return None


def _started_at(found: tuple[str, int]) -> float | None:
    """When a process (_process) started, in seconds since the epoch: the computer's boot time plus its start."""
    if not portable.LINUX:
        return float(found[1])
    try:
        boot = next(int(line.split()[1]) for line in Path("/proc/stat").read_text().splitlines()
                    if line.startswith("btime "))
        return boot + found[1] / os.sysconf("SC_CLK_TCK")
    except (OSError, StopIteration, ValueError, IndexError):
        return None


def _writer() -> dict:
    """The save process, as its journal names it: pid, computer, boot and start, so that a later process given the
    same pid (or a restarted computer's) is never taken for it (_alive)."""
    found = _process(os.getpid())
    return {"pid": os.getpid(), "host": socket.gethostname(), "boot": _boot(), "started": found[1] if found else None}


def _alive(writer: dict) -> bool | None:
    """Whether the process a journal names (_writer) still runs: True, False, or None when this computer cannot tell
    (another computer's, or no pid). A journal written before journals named their writer gives its folder's pid and
    stamp (`stamp`, _stamp_time): the save began before its folder was made, so a process given that pid since (one
    that started after the stamp, or a thread of another process) is not it, and the save is over."""
    if writer.get("host") not in (None, socket.gethostname()):
        return None
    pid = writer.get("pid")
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return None
    boot = _boot()
    if writer.get("boot") and boot and writer["boot"] != boot:
        return False                       # the computer restarted since: every process of then is gone
    try:
        portable.probe(pid)                # os.kill(pid, 0), which is no probe on Windows
    except ProcessLookupError:
        return False
    except PermissionError:                # another user's process with that pid: its start time decides
        pass
    except OSError:
        return None
    found = _process(pid)
    if found is not None and (found[0] in ("Z", "X") or
                              writer.get("started") is not None and found[1] != writer["started"]):
        return False                       # a dead one not reaped yet, or a later process given the same pid
    if _leader(pid) is False:
        return False                       # one of another process's threads: no process has that pid now
    if found is not None and writer.get("started") is None and writer.get("stamp") is not None:
        began = _started_at(found)
        if began is not None and began > writer["stamp"] + CLOCK_SLACK_S:
            return False                   # started after the folder was made: a later process given the same pid
    return True


def _remove_if_empty(folder: Path):
    try:
        folder.rmdir()
    except OSError:
        pass


UNMARKED = "its folder could not be marked as undone"      # _rollback: kept, yet not marked, so it would be undone again


def _leftovers(folder: Path) -> list[Path]:
    """What a backup folder holds besides its journal: files, and links (a view's linked element moved aside, API mode),
    never followed."""
    found = []
    for root, folders, files in os.walk(folder):
        found += [Path(root, name) for name in files
                  if not (Path(root) == folder and name in ("journal.json", "journal.json.tmp"))]
        found += [Path(root, name) for name in folders if Path(root, name).is_symlink()]
    return found


def _rollback(path: Path, journal_path: Path) -> list[str]:
    """Undo an unfinished save: remove what it wrote, put back what it moved aside (an element its journal lists as
    moved that is not in the backup was never moved, or an earlier rollback put it back: it is in place), restore the
    sample's attributes. Safe to run again when it was itself cut short (a second Ctrl+C, a kill): an element it replaced
    is removed only as its original is put back, while that original is still in the backup (the rename is atomic, so
    the backup is the progress mark); removing it among the written ones would delete the original once put back.
    Returns what could not be undone, in plain words. The backup folder goes only when nothing but its journal is left
    in it; otherwise it stays with its journal marked rolled_back (never rolled back again: by then the names it lists
    may hold new elements), so nothing moved aside is ever deleted (an earlier version moved before it journalled). A
    folder that can be neither removed nor marked is a problem too (UNMARKED): rolled back again later, it would delete
    what was saved since under the names it lists. Consolidates the zarr again when its journal says to, and when it
    holds consolidated metadata whatever the journal says (spatialdata's write_element refreshed that after each
    element: left as it is, it lists what was just removed, and the zarr no longer opens); a zarr without it, its
    journal saying to leave it so (an API session's view), stays without."""
    journal = json.loads(journal_path.read_text())
    problems = []
    moved = {(kind, name) for kind, name in journal.get("moved", [])}
    for kind, name in journal.get("written", []):
        if (kind, name) in moved:
            continue          # replaced: the loop below removes it as it puts the original back, if that is still aside
        try:
            shutil.rmtree(path / kind / name)
        except FileNotFoundError:
            pass
        except OSError as exc:
            problems.append(f"{kind}/{name} could not be removed ({exc.strerror or exc})")
    for kind, name in journal.get("moved", []):
        source = journal_path.parent / kind / name
        if not (source.exists() or source.is_symlink()):
            continue          # never moved, or put back already: in place
        try:
            (path / kind).mkdir(exist_ok=True)
            shutil.rmtree(path / kind / name, ignore_errors=True)        # what was written in its place, if anything
            portable.replace(source, path / kind / name)
        except OSError as exc:
            problems.append(f"{kind}/{name} could not be put back ({exc.strerror or exc})")
    if "attrs" in journal:
        try:
            import zarr
            zarr.open_group(str(path), mode="r+").attrs.put(journal["attrs"])
        except Exception as exc:
            problems.append(f"the sample's attributes could not be restored ({_one_line(exc)})")
    if journal.get("consolidate", True) or _consolidated(path):
        _consolidate(path)
    left = _leftovers(journal_path.parent)
    if not problems and not left:
        shutil.rmtree(journal_path.parent, ignore_errors=True)
        if not journal_path.exists():
            return []
    elif left and not problems:
        problems.append(f"{len(left)} file(s) its journal does not list were left there")
    journal["state"] = "rolled_back"
    try:
        _write_json(journal_path, journal)
    except OSError:
        problems.append(UNMARKED)
    if problems:
        progress(f"Kept {journal_path.parent}: {'; '.join(problems)}")
    return problems


def _recover(path: Path) -> list[str]:
    """Roll back this zarr's saves that were interrupted (process killed) before they committed. Only its own backup
    folders (_backup_folder, the whole name), only journals naming this zarr (one written before journals named their
    zarr is this zarr's by its folder name), and only once the process that wrote them has died (_alive). This save
    stops (HelperError, its `unfinished` naming that folder) before writing anything while one of them is unresolved: a
    live writer (another user's or install's engine saving into this zarr: each takes its lock in its own folder); an
    unfinished save of another computer (sharing the zarr's disk: only that computer can tell whether it still runs, and
    rolled back from here once it has moved on, it could delete what was saved since); and a rollback whose folder could
    not be marked rolled back (UNMARKED: rolled back again by a later save, it would delete what this one wrote under
    the names it lists). A rollback that left something, its folder marked (kept, never rolled back again), does not
    stop it: refusing would protect nothing, and a save refused while the app closes loses its run. Returns those, in
    plain words, one per folder, for the save's answer (warnings). Journals rolled back already are left as they are."""
    warned = []
    mine = _backup_folder(path)
    for root in (path.parent / name for name in names.BACKUP_FOLDERS):     # this version's, then a later one's
        if not root.is_dir():
            continue
        for folder in sorted(root.iterdir()):
            match = mine.fullmatch(folder.name)
            if match is None or not folder.is_dir():
                continue
            writer = {"pid": int(match.group("pid")), "stamp": _stamp_time(match.group("stamp"))}
            try:
                journal = json.loads((folder / "journal.json").read_text())
            except (OSError, ValueError):
                if not any(folder.iterdir()) and _alive(writer) is False:
                    _remove_if_empty(folder)              # killed before its journal was written: it moved nothing
                continue
            if journal.get("path", str(path)) != str(path) or journal.get("state") == "rolled_back":
                continue
            named = journal.get("writer") or writer
            alive = _alive(named)
            if journal.get("state") == "committed":
                if alive is False:                        # it finished; only its clean-up was cut short
                    shutil.rmtree(folder, ignore_errors=True)
                continue
            if alive:
                raise HelperError(f"Another save into this sample is under way (process {named['pid']}): "
                                  "try again once it has finished", unfinished=str(folder))
            if alive is None:
                host = named.get("host")
                if host and host != socket.gethostname():
                    raise HelperError(f"An unfinished save into this sample from {host} is in {folder}. If it is "
                                      f"still running, try again once it has finished; if not, save into this sample "
                                      f"once from {host} to undo it, or move that folder out of {root.name} (it holds "
                                      "what that save set aside) to save from here", unfinished=str(folder))
                raise HelperError(f"An unfinished save into this sample is in {folder}, and this computer cannot tell "
                                  "whether it is still running. If it is, try again once it has finished; if not, move "
                                  f"that folder out of {root.name} (it holds what that save set aside) to save from "
                                  "here", unfinished=str(folder))
            progress("Rolling back an interrupted save into this sample…")
            problems = _rollback(path, folder / "journal.json")
            if not problems:
                continue
            said = (f"An earlier save into this sample was interrupted and could not be fully undone: "
                    f"{'; '.join(problems)}. What it set aside is kept in {folder}")
            if UNMARKED in problems:
                raise HelperError(f"{said}. Move that folder out of {root.name} to save into this sample again",
                                  unfinished=str(folder))
            warned.append(said)
        _remove_if_empty(root)
    return warned


def _check_names(path: Path, shapes_key: str, table_key: str, overwrite: bool) -> list[tuple[str, str]]:
    """SpatialData's own naming rules, case-insensitive uniqueness, clashes, and tables that would be left pointing
    at replaced shapes. Returns the (kind, name) elements that will be replaced. A clash that Replace would settle
    names every clashing name and the sample's elements as they are now (HelperError details {clash, existing}), so
    the Save dialog can offer Replace even for names saved after its run ended."""
    from spatialdata._core.validation import check_valid_name
    for key in (shapes_key, table_key) if table_key is not None else (shapes_key,):
        try:
            check_valid_name(key)
        except (TypeError, ValueError) as exc:
            raise HelperError(f"'{key}': {exc}") from None
    if table_key is not None and shapes_key.lower() == table_key.lower():
        raise HelperError("The shapes layer and the table need different names (ignoring case)")
    existing = _elements(path)
    replaced, clashes = [], []
    for key, kind in ((shapes_key, "shapes"), (table_key, "tables")):
        if key is None:
            continue
        clash = next((name for name in existing if name.lower() == key.lower()), None)
        if clash is None:
            continue
        if clash != key:
            raise HelperError(f"'{key}' differs from the existing element '{clash}' only by case: choose another name")
        if existing[clash] != kind:
            raise HelperError(f"'{key}' already exists as {existing[clash]}: choose another name")
        if not overwrite:
            clashes.append(key)
        replaced.append((kind, key))
    if clashes:
        one = len(clashes) == 1
        raise HelperError(f"{' and '.join(clashes)} already exist{'s' if one else ''} in the sample: choose other names "
                          f"or allow replacing {'it' if one else 'them'}", clash=clashes, existing=sorted(existing))
    if ("shapes", shapes_key) in replaced:
        import zarr
        for name, kind in existing.items():
            if kind != "tables" or name == table_key:
                continue
            try:
                region = zarr.open_group(str(path / "tables" / name), mode="r").attrs.get("region")
            except Exception:
                continue
            regions = region if isinstance(region, list) else [region]
            if shapes_key in regions:
                raise HelperError(f"Table '{name}' annotates '{shapes_key}': replacing those shapes would break it. "
                                  f"Choose another name, or replace '{name}' instead of creating a new table.")
    return replaced


def _format_version(attrs: dict) -> str:
    """The SpatialData format a zarr's root attrs say it is in: spatialdata's reader takes one without versioned attrs
    as 0.1 (io_zarr read_zarr)."""
    return str((attrs.get("spatialdata_attrs") or {}).get("version") or "0.1")


def _store(args: dict, name: str) -> Path | None:
    """A folder `save` takes by the argument `name` (path, view, frame_from, stage_to), resolved; None when null."""
    value = args.get(name)
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise HelperError(f"'{name}' must be a folder")
    return Path(value).resolve()


def _built(args: dict, frame: Path, points_key: str | None, shapes_key: str, table_key: str | None):
    """(elements, provenance, genes): the run's results as SpatialData elements ((kind, name, element), shapes first),
    placed by the transformations of the zarr `frame` (its transcripts' own, or its images' frame system:
    _frame_transformations), what a stain save records on its shapes (`provenance`) and the table's gene count (0 for
    a stain run). Proseg's as sopa.segmentation.proseg builds them (sopa's _read_proseg and parse_table); a stain run's,
    its kept cells (`keep`, else the rules of an engine from before it)."""
    output = Path(args["output"])
    stain = args.get("kind") == "stain"
    progress("Reading the sample…", 0.1)
    sdata = _read(str(frame), _selection(points_key))
    transformations = _frame_transformations(sdata, points_key)   # the transcripts' own, or the frame system's
    provenance = dict(args.get("provenance") or {})
    if stain:
        import geopandas as gpd
        from spatialdata.models import ShapesModel
        from shapely.geometry import shape
        progress("Reading the cell shapes…", 0.3)
        features = json.loads((output / "boundaries.geojson").read_text())["features"]
        cells = [shape(f["geometry"]) for f in features]
        removed = 0             # the cells this save leaves out, by any rule (provenance: once each)
        kept = args.get("keep")
        if kept is not None:
            try:
                mask = np.load(kept, allow_pickle=False)
                valid = isinstance(mask, np.ndarray) and mask.dtype.kind == "b" and mask.ndim == 1 and len(mask) == len(cells)
                if isinstance(mask, np.lib.npyio.NpzFile):
                    mask.close()
            except (OSError, ValueError, EOFError):
                valid = False
            if not valid:
                raise HelperError("The cells changed since the save was asked for. Save again.")
            cells = [c for c, k in zip(cells, mask) if k]
            features = [f for f, k in zip(features, mask) if k]
            for key in ("removed_areas", "min_area_um2", "min_brightness", "removed_cells", "removed_version"):
                provenance.pop(key, None)    # the committed ranges are recorded together in filters
        if kept is None and args.get("exclude"):
            # Removed areas (lassos on the run): whole cells by centre, as the run showed them.
            import shapely
            area = shapely.union_all([shapely.make_valid(shapely.Polygon(r)) for r in args["exclude"]])
            centres = shapely.centroid(np.asarray(cells, dtype=object))
            # Inside or on an area's edge counts as inside, as in the engine's counts and the outlines shown.
            keep = ~shapely.intersects_xy(area, shapely.get_x(centres), shapely.get_y(centres))
            cells = [c for c, k in zip(cells, keep) if k]
            features = [f for f, k in zip(features, keep) if k]
            removed += int((~keep).sum())
        if kept is None and args.get("min_area_um2"):
            # The size rule: cells smaller than it stay out, as the run showed them.
            import shapely
            keep = shapely.area(np.asarray(cells, dtype=object)) >= args["min_area_um2"]
            cells = [c for c, k in zip(cells, keep) if k]
            features = [f for f, k in zip(features, keep) if k]
            removed += int((~keep).sum())
        if kept is None and args.get("min_brightness"):
            # The brightness rule: cells dimmer than it stay out, measured as the engine measures them.
            from sopa_companion.engine import stain_intensity
            rule = args["min_brightness"]
            progress(f"Measuring {rule['channel']} in each cell…", 0.35)
            read, frame_geometry, size = stain_intensity.job_reader(json.loads(Path(args["job"]).read_text()),
                                                                    rule["channel"])
            values = stain_intensity.brightness(read, frame_geometry, *stain_intensity.rings(cells), size)
            with np.errstate(invalid="ignore"):
                keep = ~(values < rule["value"])
            cells = [c for c, k in zip(cells, keep) if k]
            features = [f for f, k in zip(features, keep) if k]
            removed += int((~keep).sum())
        if kept is None and (args.get("exclude") or args.get("min_area_um2") or args.get("min_brightness")):
            # Counted here, from the cells left out: the engine's own count (passed in) could only add up twice.
            provenance["removed_cells"] = removed
        if not cells:          # spatialdata cannot write an empty shapes element (IndexError in ShapesModel.parse)
            raise HelperError("There are no cells to save: this run found none, or all were removed")
        geo_df = ShapesModel.parse(gpd.GeoDataFrame(geometry=cells), transformations=transformations)
        method = provenance.get("method", "cellpose")
        geo_df.index = [f"{method}-{f.get('id', i + 1)}" for i, f in enumerate(features)]
        return (("shapes", shapes_key, geo_df),), provenance, 0
    from spatialdata._core.validation import validate_table_attr_keys
    from sopa.aggregation.table import parse_table
    from sopa.segmentation.methods._proseg import _read_proseg
    progress("Reading proseg's results (sopa)…", 0.3)
    adata, geo_df = _read_proseg(output, transformations)
    if not len(geo_df):
        raise HelperError("There are no cells to save: this run found none")
    adata.uns[names.PROVENANCE] = {k: (v if isinstance(v, (str, int, float, bool)) else json.dumps(v)) for k, v in provenance.items()}
    table = parse_table(adata, geo_df, shapes_key)      # sopa's region / instance keys, centroids, areas
    validate_table_attr_keys(table)
    return (("shapes", shapes_key, geo_df), ("tables", table_key, table)), provenance, int(adata.n_vars)


def _stage(stage: Path, elements, provenance: dict | None):
    """The built elements as a new zarr at `stage` (API mode: the notebook reads them from there), never consolidated,
    a stain result's provenance on its shapes as in the sample. A folder already there must be empty (made for it)."""
    import spatialdata as sd
    import zarr
    if stage.exists() or stage.is_symlink():
        try:
            stage.rmdir()
        except OSError:
            raise HelperError(f"The folder for the result is in use already: {stage}") from None
    stage.parent.mkdir(parents=True, exist_ok=True)
    progress("Getting it ready for your notebook…", 0.5)
    kinds: dict = {}
    for kind, name, element in elements:
        kinds.setdefault(kind, {})[name] = element
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sd.SpatialData(**kinds).write(stage, consolidate_metadata=False)
        if provenance is not None:
            for kind, name, _ in elements:
                zarr.open_group(str(stage / kind / name), mode="r+").attrs[names.PROVENANCE] = provenance
        if os.environ.get("SOPA_COMPANION_TEST_FAIL_AFTER") == "stage":        # tests: a failure once it is staged
            raise RuntimeError("injected failure after staging")
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _commit(path: Path, elements, *, shapes_key: str, table_key: str | None, overwrite: bool, set_default: bool,
            provenance: dict | None, consolidate: bool, words: str, at: float, role: str = "") -> list[tuple[str, str]]:
    """`elements` written into the zarr `path` as one recoverable transaction (module docstring, save), under its lock
    and after its recovery (the caller's): names checked again, elements being replaced moved aside, each step listed
    in a write-ahead journal before it is taken, no-clobber writes (spatialdata's write_element), the sample's
    segmentation set from its attrs on disk now (their format kept), consolidated when `consolidate`, and rolled back on
    any failure. Returns the (kind, name) elements it replaced. `role` names the store for the tests' failure hooks
    ("" the target, "view": an API session's own zarr)."""
    import spatialdata as sd
    import zarr
    from sopa.utils import set_boundaries_attrs
    replaced = _check_names(path, shapes_key, table_key, overwrite)   # again, now that we are about to write
    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup = path.parent / BACKUPS / f"{path.name}-{stamp}-{os.getpid()}"
    backup.mkdir(parents=True)
    # The journal names its zarr and its process: recovery acts on this zarr's journals only, once that process has
    # died (_recover). Every step is listed before it is taken (write-ahead): a kill between a move and its entry would
    # otherwise leave the only copy of the old element in a folder recovery then deletes. `consolidate`: whether its
    # rollback may consolidate the zarr (an API session's view is never consolidated).
    journal = {"state": "writing", "path": str(path), "writer": _writer(), "shapes_key": shapes_key,
               "table_key": table_key, "consolidate": consolidate, "moved": [], "written": []}
    journal_path = backup / "journal.json"
    _write_json(journal_path, journal)
    hook = (lambda step: f"{role}:{step}") if role else (lambda step: step)
    holder = sd.SpatialData()          # the elements written into `path` through it, as sopa's own functions write them
    holder.path = path
    try:
        for kind, name in replaced:          # move the old elements aside (atomic, same filesystem)
            (backup / kind).mkdir(exist_ok=True)
            journal["moved"].append([kind, name])
            _write_json(journal_path, journal)
            portable.replace(path / kind / name, backup / kind / name)
            if os.environ.get("SOPA_COMPANION_TEST_DIE_AFTER") == hook("move"):      # tests: killed right after a move
                os._exit(137)
        progress(words, at)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for kind, name, element in elements:
                journal["written"].append([kind, name])
                _write_json(journal_path, journal)
                holder[name] = element
                holder.write_element(name, overwrite=False)
                if provenance is not None:
                    zarr.open_group(str(path / kind / name), mode="r+").attrs[names.PROVENANCE] = provenance
                if os.environ.get("SOPA_COMPANION_TEST_FAIL_AFTER") == hook(kind):     # tests: a failure mid-save
                    raise RuntimeError(f"injected failure after writing {kind}")
            if set_default:
                # sopa writes the whole attrs mapping: start from what is on disk now, not what was read earlier.
                fresh = dict(zarr.open_group(str(path), mode="r").attrs)
                journal["attrs"] = fresh.copy()
                _write_json(journal_path, journal)
                # spatialdata stamps that mapping with its current format (0.2) unless the mapping carries the
                # store's own: kept, a format-0.1 (zarr v2) store stays 0.1 (stamped 0.2, its images were read as
                # 0.2 and failed: KeyError 'ome'). A store without versioned attrs is 0.1 to spatialdata's reader,
                # and now says so (which also makes the boundaries attr visible to it).
                fresh.setdefault("spatialdata_attrs", {"version": _format_version(fresh)})
                holder.attrs = fresh
                set_boundaries_attrs(holder, shapes_key)
                if _format_version(dict(zarr.open_group(str(path), mode="r").attrs)) != _format_version(journal["attrs"]):
                    raise HelperError("Saving would have changed the sample's SpatialData format: nothing was saved")
        if consolidate:
            _consolidate(path)
        journal["state"] = "committed"
        _write_json(journal_path, journal)
    except BaseException as exc:
        progress("Saving failed: restoring the sample…")
        try:
            problems = _rollback(path, journal_path)
        except Exception as failed:                  # its journal unreadable: nothing is touched further
            problems = [f"the journal could not be read ({_one_line(failed)})"]
        _remove_if_empty(path.parent / BACKUPS)
        if problems:
            raise HelperError(f"Saving failed ({_one_line(exc)}), and the sample could not be fully restored: "
                              f"{'; '.join(problems)}. What was moved aside is kept in {backup}") from exc
        raise
    shutil.rmtree(backup, ignore_errors=True)
    _remove_if_empty(path.parent / BACKUPS)
    return replaced


def save(args: dict) -> dict:
    """A run's results into the sample (module docstring, save). API mode adds where they go:
    `path` null is no target; `view`, the engine's own zarr when it is not `path`, takes the same transaction, never
    consolidated, and its failure alone does not fail the save (view_written false, view_error: the app's layers do not
    show the result; it is still delivered), unless the view is its only place (no `path`, no `stage_to`): nothing
    would have the result then, so the save fails as a target's does; `frame_from` (default `path`, else `view`) is the
    zarr whose transformations place the result; `stage_to`, a new folder the built elements are written to first,
    unconsolidated, for the notebook (dropped again when the target's transaction fails); `consolidate` (default true,
    null too, as for the others) applies to `path`: false leaves a zarr without consolidated metadata so, while one
    holding it keeps it current all the same (spatialdata's write_element; a rollback too). The desktop passes none of
    them. `warnings`, in both: what an earlier interrupted save into the target or the view left, said by the save that
    rolled it back (_recover)."""
    target, view = _store(args, "path"), _store(args, "view")
    stage, frame = _store(args, "stage_to"), _store(args, "frame_from") or target or view
    consolidate = args.get("consolidate")
    consolidate = True if consolidate is None else consolidate
    if not isinstance(consolidate, bool):
        raise HelperError("'consolidate' must be true or false")
    in_place = view is not None and view == target         # an API session working in place: the target is its view
    if in_place:
        view = None
    if target is None and view is None and stage is None:
        raise HelperError("Nothing to save into: give the sample (path), the app's view or a folder for the result")
    if frame is None:
        raise HelperError("Nothing places the result: give the zarr it was made on (frame_from)")
    points_key, output = args.get("points_key"), Path(args["output"])
    shapes_key, table_key, overwrite = args["shapes_key"], args["table_key"], bool(args.get("overwrite"))
    stain = args.get("kind") == "stain"
    if stain:
        table_key = None
    elif not points_key:
        raise HelperError(NO_TRANSCRIPTS)
    set_default = bool(args.get("set_default", True))
    for key in (shapes_key, table_key) if not stain else (shapes_key,):
        if not isinstance(key, str) or not KEY.fullmatch(key or ""):
            raise HelperError(f"'{key}' is not a valid element name (letters, digits, _ . -)")
    result = "boundaries.geojson" if stain else "proseg-output.zarr"
    if not (output / result).exists():
        raise HelperError(f"This run has no {result} to save")
    try:
        prune_locks()                   # the other stores' stale locks: one file per store ever saved into
    except OSError:
        pass
    options = {"shapes_key": shapes_key, "table_key": table_key, "overwrite": overwrite, "set_default": set_default}
    replaced, staged, view_written, view_error = [], None, True if in_place else None, None
    warned = []
    only_view = target is None and stage is None          # the view its one place: its failure is the save's
    with ExitStack() as locks:
        # Both stores locked and recovered before the result is read from either (the view is usually its frame): the
        # target first, always in that order (a view belongs to one session).
        if target is not None:
            locks.enter_context(_store_lock(target))
            warned += _recover(target)
            _check_names(target, shapes_key, table_key, overwrite)       # before anything heavy: fail fast
        if view is not None:
            locks.enter_context(_store_lock(view))
            try:
                warned += _recover(view)
            except Exception as exc:          # not written, then (nor read through, but as the frame)
                if only_view:
                    raise
                view_written, view_error = False, _one_line(exc)
        elements, provenance, genes = _built(args, frame, points_key, shapes_key, table_key)
        provenance = provenance if stain else None       # a stain save's, on its shapes (proseg's is in its table)
        cells = int(len(elements[0][2]))
        if stage is not None:
            _stage(stage, elements, provenance)
            staged = str(stage)
        if target is not None:
            try:
                replaced = _commit(target, elements, provenance=provenance, consolidate=consolidate,
                                   words=f"Writing {cells:,} cells as '{shapes_key}'…", at=0.6, **options)
            except BaseException:
                if staged is not None:              # nothing was saved: nothing for the notebook either
                    shutil.rmtree(stage, ignore_errors=True)
                raise
        if view is not None and view_error is None:
            try:
                done = _commit(view, elements, provenance=provenance, consolidate=False, role="view",
                               words="Adding it to the app's layers…", at=0.9, **options)
                view_written = True
                replaced = replaced if target is not None else done
            except Exception as exc:      # saved (and delivered) all the same: only the app's layers do not show it
                if only_view:
                    raise
                view_written, view_error = False, _one_line(exc)
        if view_error is not None:
            progress(f"The app's layers do not show it: {view_error}")
    progress("Done", 1.0)
    return {"shapes_key": shapes_key, "table_key": table_key, "cells": cells, "genes": genes, "set_default": set_default,
            "replaced": [name for _, name in replaced], "staged": staged,
            "written_to": str(target) if target is not None else None, "view_written": view_written,
            "view_error": view_error, "warnings": warned}


def tissue(args: dict) -> dict:
    """The tissue as polygons with their holes ([exterior, hole, …] each, working frame): the sample's
    region_of_interest when `points_key` is given (null for a sample without transcripts: its frame), else sopa's
    tissue detection on the image's lowest level, in memory."""
    import shapely
    from .tissue import detect
    progress("Finding the tissue…", 0.2)
    if "points_key" in args:
        points_key = args["points_key"]
        gdf = _aligned_shapes(_read(args["path"], _selection(points_key, "shapes")), "region_of_interest", points_key)
        polygons = [p for g in gdf.geometry for p in shapely.get_parts(shapely.make_valid(g))
                    if p.geom_type == "Polygon" and not p.is_empty]
    else:
        polygons = detect(args["path"], args["image"])
    progress("Done", 1.0)

    def ring(r):
        return [[float(x), float(y)] for x, y in r.coords]
    return {"polygons": [[ring(p.exterior), *(ring(h) for h in p.interiors)] for p in polygons],
            "area_um2": float(sum(p.area for p in polygons))}


COMMANDS = {"inspect": inspect, "shapes": shapes, "prepare": prepare, "save": save, "tissue": tissue, "count": count,
            "segmentations": segmentations}


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[1] not in COMMANDS:
        print(json.dumps({"error": f"usage: helper {{{','.join(COMMANDS)}}} '<json>' (or - with the JSON on stdin)"}))
        return 2
    try:
        # "-" (or nothing): the JSON comes on stdin, as the engine sends it; what follows "-" only names where it
        # writes, for other processes to see (engine/leftovers.py named).
        given = argv[2] if len(argv) > 2 and argv[2] != "-" else None
        args = json.loads(given) if given is not None else json.loads(sys.stdin.read() or "{}")
        result = COMMANDS[argv[1]](args)
    except HelperError as exc:
        print(json.dumps({"error": str(exc), **_jsonable(exc.details)}))
        return 1
    except Exception as exc:              # the traceback for the engine's log; one line for the page
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({"error": _one_line(exc)}))
        return 1
    try:
        print(json.dumps(result), flush=True)
    except (BrokenPipeError, OSError):
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
