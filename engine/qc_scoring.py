"""Test a setting here: Go deeper's scoring as the page does it (frontend/src/qc-deeper.js), in Python, so that the
engine fixes a test's targets before any run exactly as the user saw them glow (engine/qc_pool.py).

A port, exact to the JavaScript: its rows and chips from a worker answer (normalise, explain's structure, rankOrder,
sourceRows), the where request they make (whereRequest), the where reply checked (normaliseWhere: version 3, each
point's proseg share w), each transcript's near/far likelihood pi and its P = w x pi (suspectScores, both stored as
float32 before any comparison, as the page's Float32Arrays are), how much of the glow lies near each source
(sourceSupport: a named source none of whose glowing points lies near it is not where they come from), the depth rank
of transcripts that share a place (samePlace) and the suspects a test keeps (keepSuspects, then the start's own list of
them). A point glows at P >= GLOW_MIN, whoever's outline it lies in; it is likely at pi >= LIKELY_MIN where proseg
counts at least LIKELY_W of it for the cell.
JavaScript's own rules are kept: Math.round rounds halves up (also below zero), toFixed rounds the double's exact decimal
value, Number.isFinite and isInteger refuse booleans, sorts are stable, and strings compare by UTF-16 code units. No
words of the page's decks: the structure only.

What it takes is what the page takes: any JSON value (as json.loads gives it; numpy scalars count as numbers), a key
that is absent being JavaScript's undefined (UNDEFINED where it shows: a neighbour's missing cell). The coercions the
page's code makes on odd values are made the same way (a cell number given as text, null as 0, an array as its text),
and whole numbers add up as doubles do (past 2^53 a sum loses its last bits: 2^53 + 1 is 2^53, so two neighbours' labels
can be one), so the port and the page agree on every input, not only on the worker's own. page_params and page_scores
return what the page's JSON would hold (NaN and infinities as None, undefined keys left out). A shared fixture
(frontend/test/fixtures/qc-scoring.json, made by tests/make_qc_scoring_inputs.py and frontend/test/make-qc-scoring.mjs)
is recomputed by both sides: node from qc-deeper.js, pytest from this module."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import functools
import math
import re
from typing import Callable

import numpy as np

# As frontend/src/qc-deeper.js has them.
RESULT_VERSION = 1                 # the worker answer's version taken
RANK_RATIO = 1.25                  # rows and chips change rank only when a value passes 1.25 x the one above
CHIPS, ROWS, ROW_MIN, SOURCES_MIN = 4, 5, 0.01, 5
SAME_KIND = 0.9                    # a neighbour of the cell's own kind (similar >= this)
WHERE_GENES, WHERE_OWN = 6, 3      # the where request: the chips' genes, then the cell's own genes
WHERE_V = (3,)                     # the where replies taken (3: each point's proseg share w, and what it came from)
GLOW_MIN, LIKELY_MIN, SUSPECTS_SENT = 0.2, 0.5, 400
LIKELY_W = 0.5                     # likely: pi >= LIKELY_MIN on a point proseg counts at least this much for the cell
MATCH_UM = 0.01                    # a place: x and y each within this
OWNER_CELL = 1                     # the where reply's owner codes: 0 none, 1 the cell, 2 + k source k, 255 another cell
HELD = ("exact", "approx", "none")     # the where reply's `held`: where proseg's shares came from
# The page's other rules the port needs (qc-deeper.js and qc-copy.mjs; the start's suspects: qc-panel.js testSuspects).
RESULT_STATES = ("ready", "too_few", "empty", "not_checked")
LEARNED = ("yes", "partly", "no")
DIRECTIONS = ("more", "fewer", "same")
START_BAND = 5                     # a start_change without a band: the worker's DEEP_START_BAND (qc-deeper-copy.mjs)
WHERE_ROLES = ("from", "own")
WHERE_WORDS = ("edge", "spread", "layer", "few")     # the where reply's word per gene (its keys only, never its words)
NOT_JUDGED = 255                   # the check's byte for a cell it did not judge (cellShare)
GENE_CHARS = 64                    # a suspect's gene name: 1 to this many UTF-16 code units
SPAN = 2.0 ** 26                   # samePlace: a place's number, x and y steps of MATCH_UM each within 2^25 of 0


class _Undefined:
    """JavaScript's undefined: what a missing key reads as. Falsy; never equal to None (null)."""
    __slots__ = ()

    def __repr__(self):
        return "UNDEFINED"

    def __bool__(self):
        return False


UNDEFINED = _Undefined()
_SAFE = 2 ** 53                    # a double holds every integer up to this
_WHITE = "".join(map(chr, (0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x20, 0xA0, 0x1680, *range(0x2000, 0x200B), 0x2028, 0x2029,
                           0x202F, 0x205F, 0x3000, 0xFEFF)))   # JavaScript's white space and line ends
_DECIMAL = re.compile(r"[+-]?(?:Infinity|(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?)")
_RADIX = re.compile(r"0(?:[xX]([0-9a-fA-F]+)|[oO]([0-7]+)|[bB]([01]+))")


# ---- JavaScript's values, as the page's code reads them ---------------------------------------------------------
def _is_bool(v) -> bool:
    return isinstance(v, (bool, np.bool_))


def _is_num(v) -> bool:
    """typeof v === 'number' (booleans are not numbers)."""
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_))


def _val(v):
    """A number as JavaScript holds it: an int while a double holds it exactly, else a float (rounded as JSON.parse
    rounds a long number; past the double's range an infinity). numpy scalars become Python numbers."""
    if isinstance(v, np.integer):
        v = int(v)
    if isinstance(v, int):
        if -_SAFE <= v <= _SAFE:
            return v
        try:
            return float(v)
        except OverflowError:
            return math.inf if v > 0 else -math.inf
    return float(v)


def _finite(v) -> bool:
    """Number.isFinite."""
    return _is_num(v) and math.isfinite(_val(v))


def _is_integer(v) -> bool:
    """Number.isInteger."""
    if not _finite(v):
        return False
    x = _val(v)
    return isinstance(x, int) or x.is_integer()


def _num(v, d=0):
    """The page's num(): a finite number >= 0, else `d`."""
    return _val(v) if _finite(v) and _val(v) >= 0 else d


def _maybe(v):
    """The page's maybe(): a finite number >= 0, else null."""
    return _num(v, None)


def _is_array(v) -> bool:
    return isinstance(v, (list, tuple, np.ndarray))


def _is_object(v) -> bool:
    """v && typeof v === 'object' (arrays are objects)."""
    return isinstance(v, dict) or _is_array(v)


def _get(obj, key: str):
    """obj.key for a JSON value: a dict's entry (UNDEFINED when absent); arrays, text and numbers have none of the keys
    read here. Reading a key of null or undefined is the page's TypeError."""
    if isinstance(obj, dict):
        return obj.get(key, UNDEFINED)
    if obj is None or obj is UNDEFINED:
        raise TypeError(f"Cannot read properties of {'null' if obj is None else 'undefined'} (reading '{key}')")
    return UNDEFINED


def _opt(obj, key: str):
    """obj?.key: UNDEFINED for null and undefined."""
    return UNDEFINED if obj is None or obj is UNDEFINED else _get(obj, key)


def _at(arr, i: int):
    """arr[i] (UNDEFINED past its end)."""
    return arr[i] if 0 <= i < len(arr) else UNDEFINED


def _nullish(v) -> bool:
    """v == null: null or undefined."""
    return v is None or v is UNDEFINED


def _truthy(v) -> bool:
    """JavaScript's truthiness: 0, NaN, '', null and undefined are false; every object and array is true."""
    if _nullish(v):
        return False
    if _is_bool(v):
        return bool(v)
    if _is_num(v):
        x = _val(v)
        return not math.isnan(x) and x != 0
    if isinstance(v, str):
        return v != ""
    return True


def _strict_eq(a, b) -> bool:
    """a === b on JSON values (objects by identity, NaN never equal, -0 equal to 0)."""
    if _nullish(a) or _nullish(b):
        return a is b
    if _is_bool(a) or _is_bool(b):
        return _is_bool(a) and _is_bool(b) and bool(a) == bool(b)
    if _is_num(a) or _is_num(b):
        return _is_num(a) and _is_num(b) and _val(a) == _val(b)
    if isinstance(a, str) or isinstance(b, str):
        return isinstance(a, str) and isinstance(b, str) and a == b
    return a is b


def _string_to_number(s: str) -> float:
    """Number(text): white space trimmed, '' 0, decimals, Infinity, 0x / 0o / 0b integers; anything else NaN."""
    t = s.strip(_WHITE)
    if not t:
        return 0
    if _DECIMAL.fullmatch(t):
        return float(t.replace("Infinity", "inf"))
    m = _RADIX.fullmatch(t)
    if m:
        digits, base = next((g, b) for g, b in zip(m.groups(), (16, 8, 2), strict=True) if g is not None)
        return _val(int(digits, base))
    return math.nan


def _to_number(v):
    """JavaScript's ToNumber on a JSON value: null 0, undefined NaN, booleans 0 / 1, text parsed, an array by its text,
    any other object NaN."""
    if v is None:
        return 0
    if v is UNDEFINED:
        return math.nan
    if _is_bool(v):
        return 1 if v else 0
    if _is_num(v):
        return _val(v)
    if isinstance(v, str):
        return _string_to_number(v)
    if _is_array(v):
        return _string_to_number(_to_string(v))
    return math.nan


def _number_to_string(x) -> str:
    """Number::toString: the shortest digits that read back as the same double (Python's repr finds the same ones),
    laid out as JavaScript does (no exponent from 1e-6 up to 1e21, else 1.5e-7 or 1e+21)."""
    if isinstance(x, int):
        return str(x)                                   # |x| <= 2^53: plain digits
    if math.isnan(x):
        return "NaN"
    if x == 0:
        return "0"
    if x < 0:
        return "-" + _number_to_string(-x)
    if x == math.inf:
        return "Infinity"
    _, digits, exponent = Decimal(repr(x)).as_tuple()
    digits = list(digits)
    while len(digits) > 1 and digits[-1] == 0:
        digits.pop()
        exponent += 1
    s = "".join(map(str, digits))
    k, n = len(s), exponent + len(s)                    # x = 0.s x 10^n, k digits
    if k <= n <= 21:
        return s + "0" * (n - k)
    if 0 < n <= 21:
        return s[:n] + "." + s[n:]
    if -6 < n <= 0:
        return "0." + "0" * -n + s
    e = n - 1
    return (s if k == 1 else s[0] + "." + s[1:]) + ("e+" if e >= 0 else "e-") + str(abs(e))


def _to_string(v) -> str:
    """String(v) on a JSON value (an array joined by commas, null and undefined in it as nothing)."""
    if isinstance(v, str):
        return v
    if v is None:
        return "null"
    if v is UNDEFINED:
        return "undefined"
    if _is_bool(v):
        return "true" if v else "false"
    if _is_num(v):
        return _number_to_string(_val(v))
    if _is_array(v):
        return ",".join("" if _nullish(e) else _to_string(e) for e in v)
    return "[object Object]"


def _plus(a, b):
    """a + b: text when either side is text (an array or object counts as its text), else numbers, their sum rounded
    to a double (two ints past 2^53: Python's exact sum would keep bits JavaScript's drops)."""
    a = _to_string(a) if _is_object(a) else a
    b = _to_string(b) if _is_object(b) else b
    if isinstance(a, str) or isinstance(b, str):
        return _to_string(a) + _to_string(b)
    return _val(_to_number(a) + _to_number(b))


def _minus_one(v):
    """v - 1 (an int stays an int)."""
    return _val(v) - 1 if _is_num(v) else _to_number(v) - 1


def _div(a, b):
    """a / b on doubles, never an exception: x / 0 is an infinity, 0 / 0 NaN, as in JavaScript."""
    try:
        return a / b
    except ZeroDivisionError:
        a = float(a)
        if math.isnan(a) or a == 0:
            return math.nan
        return math.copysign(math.inf, a) * math.copysign(1.0, float(b))
    except OverflowError:                               # an int quotient past the double's range
        return math.copysign(math.inf, a) * math.copysign(1.0, b)


def _max(a, b):
    """Math.max: NaN wins; +0 above -0."""
    if math.isnan(a) or math.isnan(b):
        return math.nan
    if a == b == 0:
        return 0 if not (math.copysign(1, a) < 0 and math.copysign(1, b) < 0) else -0.0
    return a if a > b else b


def _min(a, b):
    """Math.min: NaN wins; -0 below +0."""
    if math.isnan(a) or math.isnan(b):
        return math.nan
    if a == b == 0:
        return -0.0 if (math.copysign(1, a) < 0 or math.copysign(1, b) < 0) else 0
    return a if a < b else b


def _clamp(v, lo, hi):
    return _max(lo, _min(hi, v))


def _utf16(s: str) -> bytes:
    """A key that orders text as JavaScript's < does: by UTF-16 code units."""
    return s.encode("utf-16-be", "surrogatepass")


def _utf16_length(s: str) -> int:
    return len(_utf16(s)) // 2


def _quote(s: str) -> str:
    """JSON.stringify of text: quotes, backslashes and control characters escaped (lower-case hex), as are surrogates
    without their pair; everything else as it is."""
    out, units, i = ['"'], [ord(c) for c in s], 0
    named = {0x22: '\\"', 0x5C: "\\\\", 0x08: "\\b", 0x0C: "\\f", 0x0A: "\\n", 0x0D: "\\r", 0x09: "\\t"}
    while i < len(units):
        c = units[i]
        if c in named:
            out.append(named[c])
        elif c < 0x20:
            out.append(f"\\u{c:04x}")
        elif 0xD800 <= c <= 0xDBFF and i + 1 < len(units) and 0xDC00 <= units[i + 1] <= 0xDFFF:
            out.append(s[i:i + 2])                       # a pair held as two code points: one character to JavaScript
            i += 1
        elif 0xD800 <= c <= 0xDFFF:
            out.append(f"\\u{c:04x}")
        else:
            out.append(s[i])
        i += 1
    out.append('"')
    return "".join(out)


def _stringify(v) -> str:
    """JSON.stringify (no spaces): numbers as JavaScript writes them, NaN and infinities as null."""
    if _nullish(v):
        return "null"
    if _is_bool(v):
        return "true" if v else "false"
    if _is_num(v):
        x = _val(v)
        return _number_to_string(x) if not isinstance(x, float) or math.isfinite(x) else "null"
    if isinstance(v, str):
        return _quote(v)
    if _is_array(v):
        return "[" + ",".join(_stringify(e) for e in v) + "]"
    return "{" + ",".join(_quote(str(k)) + ":" + _stringify(x) for k, x in v.items() if x is not UNDEFINED) + "}"


def _json(v):
    """What JSON.parse(JSON.stringify(v)) gives back: NaN and infinities null, undefined keys left out (undefined in an
    array null), -0 as 0, numbers as JavaScript holds them."""
    if isinstance(v, dict):
        return {k: _json(x) for k, x in v.items() if x is not UNDEFINED}
    if _is_array(v):
        return [None if x is UNDEFINED else _json(x) for x in v]
    if _is_bool(v):
        return bool(v)
    if _is_num(v):
        x = _val(v)
        if isinstance(x, float) and (not math.isfinite(x) or x == 0):
            return None if not math.isfinite(x) else 0.0
        return x
    return v


def _stable_sort(items: list, compare: Callable) -> list:
    """Array.prototype.sort with a comparator: stable, and only "compare(a, b) < 0" puts a before b (NaN counts as 0),
    which is all Python's sort asks of cmp_to_key."""
    return sorted(items, key=functools.cmp_to_key(compare))


def _is_true(v) -> bool:
    """v === true."""
    return v is True or v is np.True_


# ---- Math.round and toFixed -------------------------------------------------------------------------------------
def js_round(v: float) -> int:
    """Math.round: the nearest whole number, halves up (also below zero: -2.5 -> -2), exact for every double (never
    floor(v + 0.5) in floating point, which turns 0.49999999999999994 into 1). NaN and infinities come back as they
    are."""
    x = _to_number(v)
    if isinstance(x, int):
        return x
    if not math.isfinite(x):
        return x
    r = math.floor(x)
    return r + 1 if x - r >= 0.5 else r                 # x - floor(x) is exact, or rounds only where both give 0


def _round_array(a: np.ndarray) -> np.ndarray:
    """js_round over a float64 array (as float64; NaN and infinities kept)."""
    r = np.floor(a)
    with np.errstate(invalid="ignore"):
        return r + (a - r >= 0.5)


def to_fixed(v: float, d: int) -> float:
    """+v.toFixed(d): the double's exact decimal value rounded half up to d places (0.3125 -> 0.313, -0.3125 -> -0.313),
    then read back as the nearest double. NaN, infinities and |v| >= 1e21 come back as they are."""
    if not 0 <= d <= 100:
        raise ValueError("toFixed() digits argument must be between 0 and 100")
    if not _is_num(v):
        raise TypeError("toFixed() is a number's")
    x = float(_val(v))
    if not math.isfinite(x) or abs(x) >= 1e21:
        return x
    if x == 0:
        return 0.0
    # An exact tie (x 10^d = n + 1/2 exactly: 2 x 10^d an odd whole number) takes the larger n; anything else has one
    # nearest, which Python's formatting finds from the same exact value.
    num, den = abs(x).as_integer_ratio()
    twice = 2 * num * 10 ** d
    if twice % den == 0 and (twice // den) % 2 == 1:
        return math.copysign(((twice // den + 1) // 2) / 10 ** d, x)
    return float(f"{x:.{d}f}")


# ---- the window's rows and chips ----------------------------------------------------------------------------------
def _cell_share(code):
    """qc-copy.mjs cellShare: the middle of a byte's range (null for a byte that is not a judged cell's)."""
    if not _is_integer(code) or not 0 <= _val(code) < NOT_JUDGED:
        return None
    return _min(1, (_val(code) + 0.5) / 250)


def _string_rows(rows) -> list:
    """Rows [name, ...] whose name is text."""
    return [g for g in rows if _is_array(g) and isinstance(_at(g, 0), str)] if _is_array(rows) else []


def normalise(raw: dict) -> dict | None:
    """normalise: a v 1 worker answer checked and normalised; None when it is not one."""
    if not _is_object(raw) or not isinstance(_get(raw, "state"), str) or _get(raw, "state") not in RESULT_STATES:
        return None
    v = _get(raw, "v")
    if not _nullish(v) and not _strict_eq(v, RESULT_VERSION):
        return None
    seq = _get(raw, "seq")
    seq = "start" if isinstance(seq, str) and seq == "start" else _val(seq) if _is_integer(seq) and _val(seq) >= 0 \
        else None
    if seq is None:
        return None
    model, iter_, code, ms = _get(raw, "model"), _get(raw, "iter"), _get(raw, "code"), _get(raw, "ms")
    cell = _get(raw, "cell")
    r = {"v": RESULT_VERSION, "state": _get(raw, "state"), "seq": seq, "iter": _val(iter_) if _finite(iter_) else None,
         "cell": None if cell is UNDEFINED else cell, "model": None if _nullish(model) else model,
         "transcripts": _num(_get(raw, "transcripts")), "outside": _num(_get(raw, "outside")),
         "code": _val(code) if _is_integer(code) else None, "ms": _val(ms) if _finite(ms) else None}
    if r["state"] == "too_few":
        r["genes"] = [[g[0], _num(_at(g, 1))] for g in _string_rows(_get(raw, "genes"))]
        return r
    if r["state"] != "ready":
        return r
    if r["code"] is None or not 0 <= r["code"] <= 250:
        return None
    parts = _get(raw, "parts")
    if _is_array(parts) and len(parts) == 3 and all(_finite(p) for p in parts):
        r["parts"] = [_max(0, _val(p)) for p in parts]
    else:
        share = _cell_share(r["code"])
        r["parts"] = [1 - share, share, 0]
    r["rough"] = _truthy(_get(raw, "rough"))
    genes = []
    for g in _string_rows(_get(raw, "genes")):
        if not _finite(_at(g, 1)):
            continue
        flag, giver = _at(g, 4), _at(g, 5)
        row = {"gene": g[0], "tx": _num(g[1]), "own": _num(_at(g, 2)), "nb": _num(_at(g, 3)),
               "flag": _is_true(flag) or (_is_num(flag) and _val(flag) == 1),
               "giver": _val(giver) if _is_integer(giver) else None}
        row["loose"] = _max(0, row["tx"] - row["own"] - row["nb"])
        genes.append(row)
    r["genes"] = genes
    kind = _get(raw, "kind")
    r["kind"] = [g for g in kind if isinstance(g, str)] if _is_array(kind) else []
    r["own_genes"] = [[g[0], _num(_at(g, 1)), _num(_at(g, 2))] for g in _string_rows(_get(raw, "own_genes"))]
    share = _get(raw, "same_kind_share")
    r["same_kind_share"] = _clamp(_val(share), 0, 1) if _finite(share) else 0
    neighbours = _get(raw, "neighbours")
    r["neighbours"] = []
    for n in (neighbours if _is_array(neighbours) else []):
        if not _is_object(n):
            continue
        um, same, similar, known = _get(n, "um"), _get(n, "same"), _get(n, "similar"), _get(n, "known")
        kind = _get(n, "kind")
        r["neighbours"].append({
            "cell": _get(n, "cell"), "um": _val(um) if _finite(um) else None, "tx": _num(_get(n, "tx")),
            "genes": [[g[0], _num(_at(g, 1))] for g in _string_rows(_get(n, "genes"))],
            "kind": [g for g in kind if isinstance(g, str)] if _is_array(kind) else [],
            # The worker's own class wins (it keeps a neighbour's class near the cut-off steady).
            "same": bool(same) if _is_bool(same) else
            _is_true(similar) or (_finite(similar) and _val(similar) >= SAME_KIND),
            "similar": None if similar is UNDEFINED else similar,
            "known": not (_is_bool(known) and not known)})
    learned = _get(raw, "learned")
    r["learned"] = learned if isinstance(learned, str) and learned in LEARNED else "yes"
    sc = _get(raw, "start_change")
    if _truthy(sc) and _is_object(sc) and _finite(_get(sc, "points")):
        band, draws, now_tx, start_tx = (_get(sc, k) for k in ("band", "draws", "now_tx", "start_tx"))
        now, start = _get(sc, "now"), _get(sc, "start")
        direction, held = _get(sc, "direction"), _get(sc, "held")
        r["start_change"] = {"points": _val(_get(sc, "points")), "band": _val(band) if _finite(band) else START_BAND,
                             "draws": _val(draws) if _finite(draws) else None,
                             "now_tx": _val(now_tx) if _finite(now_tx) else None,
                             "start_tx": _val(start_tx) if _finite(start_tx) else None,
                             "now": _val(now) if _finite(now) else None,
                             "start": _val(start) if _finite(start) else None,
                             "direction": direction if isinstance(direction, str) and direction in DIRECTIONS else None,
                             "held": _is_true(held)}
    else:
        r["start_change"] = None
    rest = _get(raw, "genes_rest")
    rest = [_val(v) for v in rest] if _is_array(rest) and len(rest) == 4 and all(_finite(v) for v in rest) else None
    r["rest"] = {"k": rest[0], "tx": rest[1], "own": rest[2], "nb": rest[3]} if rest and rest[0] > 0 else None
    total = _get(raw, "genes_total")
    r["genes_total"] = _val(total) if _is_integer(total) and _val(total) >= len(genes) else \
        _val(len(genes) + (r["rest"]["k"] if r["rest"] else 0))
    return r


def explain_core(r: dict, label: Callable = lambda c: _plus(c, 1)) -> dict:
    """The structural part of explain for a normalised result: {state, none (under SOURCES_MIN % from neighbours),
    chips: [{key, gene, nb, tx}] (the tagged genes by transcripts from neighbours, ties by name, each holding at least
    half its transcripts from them as a whole number), neighbours: [{key (the row's: its label, or "n<k>" without one),
    index, cell, label, share (of the cell's RNA), same, known, um, tx}], candidates (the neighbours with a share of at
    least ROW_MIN)}. `label(cell)`: a neighbour's label, as the page's lookups give it (Go deeper's: cell + 1)."""
    if r.get("state") != "ready":
        return {"state": r.get("state"), "none": False, "chips": [], "neighbours": [], "candidates": []}
    L, share = r["transcripts"], _cell_share(r["code"])
    nb_int = math.floor((share if share is not None else 0) * 100 + 1e-9)
    tagged = sorted((g for g in r["genes"] if g["flag"]), key=lambda g: (-g["nb"], _utf16(g["gene"])))
    # A chip's first number is rounded down: a tagged gene with under half its transcripts from neighbours as a whole
    # number stays off the chips.
    chips = [{"key": g["gene"], "gene": g["gene"], "nb": g["nb"], "tx": g["tx"]} for g in tagged
             if math.floor(g["nb"] + 1e-6) * 2.0 >= g["tx"]]
    neighbours = []
    for k, n in enumerate(r["neighbours"]):
        lab = label(n["cell"])
        neighbours.append({"key": lab if _truthy(lab) else f"n{k}", "index": k, "cell": n["cell"], "label": lab,
                           "share": _div(n["tx"], L) if L > 0 else 0, "same": n["same"], "known": n["known"],
                           "um": n["um"], "tx": n["tx"]})
    return {"state": "ready", "none": nb_int < SOURCES_MIN, "chips": chips, "neighbours": neighbours,
            "candidates": [n for n in neighbours if n["share"] >= ROW_MIN]}


def rank_order(previous: list, items: list, value: Callable, ratio: float = RANK_RATIO) -> list:
    """rankOrder: `items` (keys: text or numbers) ranked by `value(key)`, starting from the order shown before
    (`previous`): an item moves above the one over it only when its value passes `ratio` x that one's; new items enter
    where their value puts them."""
    items = list(items)
    wanted = set(items)
    order = [k for k in (previous or []) if k in wanted]
    known = set(order)
    for k in _stable_sort([k for k in items if k not in known], lambda a, b: value(b) - value(a)):
        i = len(order)
        while i > 0 and value(order[i - 1]) < value(k):
            i -= 1
        order.insert(i, k)
    moved, done = True, 0
    while moved and done < len(order):
        moved = False
        for i in range(1, len(order)):
            if value(order[i]) > ratio * value(order[i - 1]):
                order[i - 1], order[i] = order[i], order[i - 1]
                moved = True
        done += 1
    return order


def source_rows(model: dict, previous=()) -> list[dict]:
    """sourceRows' rows: the candidates (explain_core's, keyed by their row key; a key twice keeps its first place and
    its last neighbour, as a Map does) ranked from `previous`, at most ROWS."""
    by_key = {}
    for n in model["candidates"]:
        by_key[n["key"]] = n
    return [by_key[k] for k in rank_order(list(previous), list(by_key), lambda k: by_key[k]["share"])[:ROWS]]


def where_request(r: dict, *, cell: int = 0, none: bool = False, rows: list = (), chips: list = ()) -> dict | None:
    """whereRequest for the ready result `r` shown on the cell with label `cell` (cell number + 1), its source rows as
    painted and the chips' ranked genes: {key, params: {seq, cell (label - 1), sources: [label - 1], genes: [the chips'
    first WHERE_GENES], own: [its own genes not among them, the first WHERE_OWN]}, labels: [the rows' labels]}; None
    with nothing to show (no ready result, `none`, no row with a label)."""
    if not r or r.get("state") != "ready" or _truthy(none) or not _to_number(cell) > 0:
        return None
    shown = [row for row in rows if isinstance(row, dict) and _to_number(row.get("label", UNDEFINED)) > 0][:ROWS]
    if not shown:
        return None
    genes = list(dict.fromkeys(g for g in chips if isinstance(g, str) and g))[:WHERE_GENES]
    own = list(dict.fromkeys(g[0] for g in (r.get("own_genes") or [])
                             if isinstance(g[0], str) and g[0] and g[0] not in genes))[:WHERE_OWN]
    params = {"seq": r["seq"], "cell": _minus_one(cell), "sources": [_minus_one(row["label"]) for row in shown],
              "genes": genes, "own": own}
    return {"key": _stringify([params["seq"], params["cell"], params["sources"], genes, own]), "params": params,
            "labels": [row["label"] for row in shown]}


# ---- the where reply --------------------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class WhereReply:
    """A where reply as normaliseWhere returns it (the points as arrays)."""
    v: int
    seq: int | str
    iter: float | None
    cell: int
    basis: str | None
    box: list | None            # [x0, y0, x1, y1] µm, or None (nothing in that frame)
    genes: list[str]
    roles: list[str]            # "from" | "own", per gene
    count: int
    x: np.ndarray               # float32[count]
    y: np.ndarray               # float32[count]
    z: np.ndarray               # float32[count]: NaN where the reply gives no depth
    gene: np.ndarray            # uint8[count]: an index into genes
    owner: np.ndarray           # uint8[count]: OWNER_CELL, ... (where it lies: drawing only)
    near: np.ndarray            # uint8[count]: 0 or 1 (all 0 without them)
    w: np.ndarray               # float64[count]: proseg's share of the point for the cell, 0 to 1, as given
    held: str | None            # where the shares came from: one of HELD, None when the reply does not say
    truncated: bool
    cell_total: float | None
    stats: list[dict]           # gene_stats, normalised
    into: list[dict]


_PLAIN = (int, float)


def _numbers(values) -> np.ndarray | None:
    """A list of numbers as float64 (JavaScript's values: a long int rounded, past the range an infinity); None when one
    of them is not a number."""
    if all(type(v) in _PLAIN for v in values):
        try:
            return np.array(values, dtype=np.float64)
        except OverflowError:
            pass
    elif not all(_is_num(v) for v in values):
        return None
    return np.array([float(_val(v)) for v in values], dtype=np.float64)


def _depths(values) -> np.ndarray:
    """Each depth a finite number gives, else NaN (float64)."""
    out = np.array([float(_val(v)) if _is_num(v) else math.nan for v in values], dtype=np.float64)
    out[~np.isfinite(out)] = math.nan
    return out


def _f32(a: np.ndarray) -> np.ndarray:
    """A Float32Array's store: the nearest float32, ties to even, past its range an infinity."""
    with np.errstate(over="ignore"):
        return a.astype(np.float32)


def _pair(v):
    """The page's pair(): [a, b] of finite numbers >= 0, else null."""
    return [_val(v[0]), _val(v[1])] if _is_array(v) and len(v) == 2 and all(_finite(x) and _val(x) >= 0 for x in v) \
        else None


def normalise_where(raw: dict, req: dict | None = None) -> WhereReply | None:
    """normaliseWhere: a where reply (v 3: each point's proseg share w, 0 to 1) checked against its request `req`
    (where_request's, or None) and normalised; None when it is not one (any point that is not one refuses it all)."""
    if not _is_object(raw):
        return None
    v = _get(raw, "v")
    if not any(_strict_eq(v, w) for w in WHERE_V):
        return None
    if _truthy(req):
        params = _get(req, "params")
        if not _strict_eq(_get(raw, "cell"), _get(params, "cell")) or \
                _to_string(_get(raw, "seq")) != _to_string(_get(params, "seq")):
            return None
    genes, roles = _get(raw, "genes"), _get(raw, "roles")
    if not _is_array(genes) or not _is_array(roles) or len(genes) != len(roles) or \
            not all(isinstance(g, str) for g in genes) or not all(isinstance(r, str) and r in WHERE_ROLES for r in roles):
        return None
    box = _get(raw, "box")
    box = None if _nullish(box) else box                # null: neither the cell nor a source holds anything there
    if box is not None and (not _is_array(box) or len(box) != 4 or not all(_finite(b) for b in box)
                            or not (_val(box[2]) > _val(box[0]) and _val(box[3]) > _val(box[1]))):
        return None
    p = _get(raw, "points")
    p = p if _truthy(p) and _is_object(p) else None
    x = _opt(p, "x")
    if not _is_array(x):
        return None
    n = len(x)
    if not all(_is_array(_get(p, k)) and len(_get(p, k)) == n for k in ("y", "gene", "owner", "w")):
        return None
    near = _get(p, "near")
    if not _nullish(near) and (not _is_array(near) or len(near) != n):
        return None
    xs, ys, gs, os_, ws = (_numbers(_get(p, k)) for k in ("x", "y", "gene", "owner", "w"))
    if xs is None or ys is None or gs is None or os_ is None or ws is None:
        return None
    if not (np.isfinite(xs).all() and np.isfinite(ys).all()):
        return None
    with np.errstate(invalid="ignore"):
        if not (np.isfinite(ws).all() and (ws >= 0).all() and (ws <= 1).all()):    # a share: 0 to 1
            return None
    with np.errstate(invalid="ignore"):
        if not ((gs == np.floor(gs)).all() and (gs >= 0).all() and (gs < len(genes)).all()
                and (os_ == np.floor(os_)).all() and (os_ >= 0).all() and (os_ <= 255).all()):
            return None
    if _nullish(near):
        cs = np.zeros(n, np.float64)
    else:
        cs = _numbers(near)
        if cs is None or not ((cs == 0) | (cs == 1)).all():
            return None
    depth = _get(p, "z")
    z = _depths(depth) if _is_array(depth) and len(depth) == n else np.full(n, math.nan)
    stats = []
    for s in (_get(raw, "gene_stats") if _is_array(_get(raw, "gene_stats")) else []):
        if not (_truthy(s) and isinstance(_opt(s, "gene"), str)):
            continue
        role, giver, density, word = _get(s, "role"), _get(s, "giver"), _get(s, "density"), _get(s, "word")
        stats.append({"gene": s["gene"], "role": role if isinstance(role, str) and role in WHERE_ROLES else "from",
                      "giver": _val(giver) if _is_integer(giver) and _val(giver) >= 0 else None,
                      "in_cell": _num(_get(s, "in_cell")), "near_giver": _maybe(_get(s, "near_giver")),
                      "own_near": _pair(_get(s, "own_near")),
                      "density": {k: _maybe(_opt(density, k) if isinstance(density, dict) else UNDEFINED)
                                  for k in ("cell", "giver", "around")},
                      # Object.hasOwn(WHERE_WORDS, word): the word's text is one of its keys.
                      "word": word if _to_string(word) in WHERE_WORDS else None})
    into = []
    for t in (_get(raw, "into") if _is_array(_get(raw, "into")) else []):
        if not (_truthy(t) and isinstance(_opt(t, "gene"), str) and _is_integer(_get(t, "source"))
                and _val(_get(t, "source")) >= 0):
            continue
        into.append({"source": _val(t["source"]), "gene": t["gene"], "inside": _num(_get(t, "inside")),
                     "near_cell": _num(_get(t, "near_cell")), "source_own_near": _pair(_get(t, "source_own_near"))})
    iter_, cell_total, seq, cell, basis = (_get(raw, k) for k in ("iter", "cell_total", "seq", "cell", "basis"))
    held = _get(raw, "held")
    return WhereReply(
        v=int(_val(v)), seq=None if seq is UNDEFINED else seq,
        iter=_val(iter_) if _finite(iter_) else None, cell=None if cell is UNDEFINED else cell,
        basis=basis if isinstance(basis, str) else None, box=None if box is None else [_val(b) for b in box],
        genes=list(genes), roles=list(roles), count=n, x=_f32(xs), y=_f32(ys), z=_f32(z),
        gene=(gs.astype(np.int64) % 256).astype(np.uint8),           # a Uint8Array keeps an index modulo 256
        owner=os_.astype(np.uint8), near=cs.astype(np.uint8), w=ws, held=held if isinstance(held, str) and held in HELD
        else None, truncated=_is_true(_get(raw, "truncated")),
        cell_total=_val(cell_total) if _finite(cell_total) and _val(cell_total) >= 0 else None, stats=stats, into=into)


# ---- the suspects: each transcript's P, places shared, the list a test keeps -----------------------------------
def suspect_scores(reply: WhereReply, result: dict) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """suspectScores: which of the transcripts proseg counts for the cell likely came from a neighbour, from the
    normalised ready `result` the request was made for. Per "from" gene with giver k: s = nb / tx; f_r = (own_near[0] +
    1) / (own_near[1] + 2); f_g = near_giver / in_cell (proseg's count, as sums of w); f_s = clamp((f_g - (1 - s) f_r)
    / s, f_r, 1); a transcript of it near k gets pi = s f_s / (s f_s + (1 - s) f_r), one further pi = s (1 - f_s) / (s
    (1 - f_s) + (1 - s)(1 - f_r)) (0 for 0 / 0). Without a giver among the rows each gets s. Its P = w x pi, w its
    proseg share for the cell (a point with w 0 is no suspect). Returns (P per point as float32, 0 for a point that is
    no suspect; pi per point likewise; [{gene, giver, s, fr, fg, fs, near, far}] per gene scored)."""
    n = reply.count if reply is not None else 0
    p, pi = np.zeros(n, np.float32), np.zeros(n, np.float32)
    if not n or not isinstance(result, dict) or result.get("state") != "ready" or not _is_array(result.get("genes")):
        return p, pi, []
    rows = {g["gene"]: g for g in result["genes"]}
    stats = {s["gene"]: s for s in reply.stats if s["role"] == "from"}
    per = []
    for i, gene in enumerate(reply.genes):
        row = rows.get(gene) if reply.roles[i] == "from" else None
        if not row or not row["tx"] > 0:
            per.append(None)
            continue
        s, st = _clamp(_div(row["nb"], row["tx"]), 0, 1), stats.get(gene)
        k = st["giver"] if st is not None else None
        if k is None or not st["own_near"] or st["near_giver"] is None or not st["in_cell"] > 0:
            per.append({"gene": gene, "giver": None, "s": s, "near": s, "far": s})
            continue
        fr = _div(_val(st["own_near"][0] + 1), _val(st["own_near"][1] + 2))    # sums as doubles (past 2^53)
        fg = _clamp(_div(st["near_giver"], st["in_cell"]), 0, 1)
        if not s > 0:
            per.append({"gene": gene, "giver": k, "s": 0, "fr": fr, "fg": fg, "fs": fr, "near": 0, "far": 0})
            continue
        fs = _clamp(_div(fg - (1 - s) * fr, s), fr, 1)
        far_in = s * (1 - fs) + (1 - s) * (1 - fr)
        per.append({"gene": gene, "giver": k, "s": s, "fr": fr, "fg": fg, "fs": fs,
                    "near": _div(s * fs, s * fs + (1 - s) * fr), "far": _div(s * (1 - fs), far_in) if far_in > 0 else 0})
    scored = np.array([g is not None for g in per] + [False])          # the last: padding for a reply without genes
    near = np.array([g["near"] if g else 0.0 for g in per] + [0.0], np.float64)
    far = np.array([g["far"] if g else 0.0 for g in per] + [0.0], np.float64)
    gene = reply.gene.astype(np.int64)
    take = (reply.w > 0) & scored[gene]
    pi64 = np.where(reply.near != 0, near[gene], far[gene])          # each the double the page works out
    pi[take] = _f32(pi64)[take]
    p[take] = _f32(reply.w * pi64)[take]
    return p, pi, [g for g in per if g is not None]


def source_support(reply: WhereReply, p) -> dict:
    """sourceSupport: {source k (its place in the request's sources): proseg's share summed over the glowing points (P
    >= GLOW_MIN) that lie near it}: the points whose gene k gives and whose near flag is set, their w added up in point
    order as doubles; a source with none is left out (0). A source the glow names with no support (none of its glowing
    points lies near it) is not where they come from: the page then says they look like a cell with no outline there.
    `p`: P per point as float32 (another dtype is first stored as float32, as the page's Float32Array)."""
    out: dict = {}
    if reply is None or not reply.count:
        return out
    p = np.asarray(p)
    if p.dtype != np.float32:
        p = _f32(p.astype(np.float64))
    giver_of = {s["gene"]: s["giver"] for s in reply.stats if s["role"] == "from"}
    givers = [giver_of.get(g) for g in reply.genes]
    for i in np.flatnonzero((p.astype(np.float64) >= GLOW_MIN) & (reply.near != 0)).tolist():
        k = givers[int(reply.gene[i])]
        if k is not None:
            out[k] = out.get(k, 0.0) + float(reply.w[i])
    return out


def _place_key(k: float):
    """A place's number as a dict key: NaN (an infinite place's) as one key, as a Map holds it."""
    return "nan" if math.isnan(k) else k


def same_place(reply: WhereReply, indices) -> dict[int, tuple[int, int]]:
    """samePlace: for the points `indices`, those whose gene and place (x and y each within MATCH_UM: on its grid, at
    most a step away) other points of the reply share, any owner, itself included: {point: (rank among them by depth,
    z ascending, equal depths in point order; how many)}; left out where one of them has no depth."""
    out: dict[int, tuple[int, int]] = {}
    n = reply.count if reply is not None else 0
    indices = [int(i) for i in (indices if indices is not None else [])]
    if not n or not indices:
        return out
    # A place's number, in doubles as the page works it out: (round(x / MATCH_UM) + SPAN / 2) * SPAN + round(y /
    # MATCH_UM) + SPAN / 2 (a place past float32's range is infinite, and NaN where x and y are of opposite infinities).
    with np.errstate(invalid="ignore", over="ignore"):
        keys = (_round_array(reply.x.astype(np.float64) / MATCH_UM) + SPAN / 2) * SPAN \
            + _round_array(reply.y.astype(np.float64) / MATCH_UM) + SPAN / 2
    asked = {int(reply.gene[i]) for i in indices}
    grid: dict = {}
    for j in np.flatnonzero(np.isin(reply.gene, list(asked))).tolist():
        grid.setdefault((int(reply.gene[j]), _place_key(float(keys[j]))), []).append(j)
    z = reply.z.astype(np.float64)
    for i in indices:
        g, k, group = int(reply.gene[i]), float(keys[i]), []
        for d in range(9):
            group += grid.get((g, _place_key(k + ((d % 3) - 1) * SPAN + d // 3 - 1)), [])
        if len(group) < 2 or not np.isfinite(z[group]).all():
            continue
        group.sort(key=lambda j: (z[j], j))
        out[i] = (group.index(i) if i in group else -1, len(group))
    return out


def keep_suspects(reply: WhereReply, p: np.ndarray, *, cell_only: bool = True, max_n: int = SUSPECTS_SENT
                  ) -> list[dict]:
    """keepSuspects, then the list a test starts with (qc-panel.js testSuspects): the points at P >= GLOW_MIN (those
    proseg counts for the cell only, w > 0, with `cell_only`: P is 0 elsewhere anyway), P to 3 decimals, the likeliest
    first (ties in point order), at most `max_n` (both of the page's cuts of SUSPECTS_SENT); x and y to 2 decimals; a
    point that shares its gene and place with others carries its `rank` among them by depth and how many they are
    (`of`). [{x, y, gene, p, rank?, of?}]. `p`: P per point as float32 (another dtype is first stored as float32, as the
    page's Float32Array)."""
    p = np.asarray(p)
    if p.dtype != np.float32:
        p = _f32(p.astype(np.float64))
    glow = p.astype(np.float64) >= GLOW_MIN
    if cell_only:
        glow &= reply.w > 0
    found = [(i, to_fixed(float(p[i]), 3)) for i in np.flatnonzero(glow).tolist()]
    top = sorted(found, key=lambda s: -s[1])[:max(0, max_n)]
    shared = same_place(reply, [i for i, _ in top])
    kept = []
    for i, prob in top:
        s = {"x": to_fixed(float(reply.x[i]), 2), "y": to_fixed(float(reply.y[i]), 2),
             "gene": reply.genes[int(reply.gene[i])], "p": prob}
        if i in shared:
            s["rank"], s["of"] = shared[i]
        kept.append(s)
    # The start's own checks: finite places, P from 0 to 1, a gene of 1 to GENE_CHARS characters; then to 0.01 µm and
    # 0.001, rank and of only where they make sense.
    sent = [s for s in kept if math.isfinite(s["x"]) and math.isfinite(s["y"]) and 0 <= s["p"] <= 1
            and 1 <= _utf16_length(s["gene"]) <= GENE_CHARS]
    out = []
    for s in sorted(sent, key=lambda s: -s["p"])[:max(0, max_n)]:
        t = {"x": js_round(s["x"] * 100) / 100, "y": js_round(s["y"] * 100) / 100, "gene": s["gene"],
             "p": js_round(s["p"] * 1000) / 1000}
        if "of" in s and s["of"] > 1 and 0 <= s["rank"] < s["of"]:
            t["rank"], t["of"] = s["rank"], s["of"]
        out.append(t)
    return out


# ---- what the page makes of a cell, end to end ---------------------------------------------------------------------
def page_params(result_raw: dict, cell: int) -> dict:
    """What the page makes of a worker answer for `cell` (label - 1), with nothing shown before: {state, none, chips:
    [gene], rows: [{label, cell, share, same, key}], req: {key, params, labels} | None}. A ready answer with a cell
    that is not a cell number is refused (ValueError)."""
    r = normalise(result_raw)
    if r is None or r["state"] != "ready":
        return {"state": None if r is None else r["state"], "none": False, "chips": [], "rows": [], "req": None}
    if not _is_integer(cell) or _val(cell) < 0:
        raise ValueError("cell must be a cell number (label - 1)")
    m = explain_core(r)
    nb = {}
    for chip in m["chips"]:
        nb[chip["key"]] = chip["nb"]
    chips = rank_order([], [chip["key"] for chip in m["chips"]], lambda k: nb[k])
    rows = source_rows(m, [])
    req = where_request(r, cell=_plus(cell, 1), none=m["none"], rows=rows, chips=chips)
    return _json({"state": r["state"], "none": bool(m["none"]), "chips": chips,
                  "rows": [{"label": row["label"], "cell": row["cell"], "share": row["share"],
                            "same": _truthy(row["same"]), "key": row["key"]} for row in rows],
                  "req": None if req is None else {k: req[k] for k in ("key", "params", "labels")}})


def page_scores(result_raw: dict, req: dict | None, reply_raw: dict) -> dict | None:
    """What the page makes of the where reply to that request (page_params' req, or None): {count, points: [{i, x, y,
    gene, owner, near, w, pi, P, rank?, of?}] (the glowing points, P >= GLOW_MIN whoever's outline they lie in, in point
    order, x and y to 2 decimals, w as the reply gives it, pi and P the float32 values), genes: suspect_scores' per
    gene, support: [[k, source_support's sum]] by k}; None where the page takes no reply. A point is likely at pi >=
    LIKELY_MIN (as float32) with w >= LIKELY_W."""
    reply = normalise_where(reply_raw, req)
    if reply is None:
        return None
    p, pi, genes = suspect_scores(reply, normalise(result_raw))
    idx = np.flatnonzero(p.astype(np.float64) >= GLOW_MIN).tolist()
    same = same_place(reply, idx)
    points = []
    for i in idx:
        point = {"i": i, "x": to_fixed(float(reply.x[i]), 2), "y": to_fixed(float(reply.y[i]), 2),
                 "gene": reply.genes[int(reply.gene[i])], "owner": int(reply.owner[i]), "near": int(reply.near[i]),
                 "w": float(reply.w[i]), "pi": float(pi[i]), "P": float(p[i])}
        if i in same:
            point["rank"], point["of"] = same[i]
        points.append(point)
    support = source_support(reply, p)
    return _json({"count": reply.count, "points": points, "genes": genes,
                  "support": [[k, support[k]] for k in sorted(support)]})
