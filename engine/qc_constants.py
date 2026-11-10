"""The mixing check's calibrated constants, one entry per kind of run ("mode"), in one table: what engine/qc.py,
engine/qc_rules.py and engine/qc_areas.py read, by `value(mode_of(cells, tiled), key)`. Pure Python, importing nothing
of the engine's, so a calibration can read and write it on its own.

The modes: "region" (a dense region, at most [proseg] max_region_um across), "tiled" (a tiled region) and "whole" (a
whole sample: 20,000 cells or more). The values (CONSTANTS_ID) were measured again for regions and tiled regions on
repeat runs scored on proseg's own count (MEASURED; 500 µm squares and a 1.2 mm box run dense, the box run tiled), each
one measured or kept with its reason (TABLE). Whole samples keep the values measured on the transcripts inside the
drawn outlines until their own repeat runs on proseg's count re-measure them; until then a number shown with them
carries a caveat in its tooltip (`block`, which every summary carries), never a refusal. Go deeper's and the worker's
own constants (one value in every mode, measured on regions: CALIBRATION.md) are named per mode where they were kept,
not measured (KEPT: a caveat in their tooltips too).

Keys (every mode has every one):
  headline_weight, headline_ref_s  the headline's smoothing: weight headline_weight on the newest live result per
                                   headline_ref_s seconds of the run clock
  shown_margin                     the headline's whole number moves only past a rounding boundary by this much
  sampler_k                        the size-aware threshold 1.96 k 100 sqrt(p (1 - p) / n)
  matched_fixed                    the largest gap between repeat runs' like-for-like changes: the threshold's floor
  matched_model                    another learning of the model could reverse a change below this
  local_min_judged, local_min_paired   the minimum support for a verdict: judged and paired cells
  trial                            a knob change's band and verdict (engine/qc_rules.py Watch)
  areas                            the flagged areas' rules (engine/qc_areas.py Tracker): 100 µm bins for tiled runs
                                   and whole samples, 50 µm for dense regions
"""
from __future__ import annotations

import math

CONSTANTS_ID = "proseg-2026-10-31"        # measured on proseg's count (whole samples: kept, MEASURED); the values
                                          # before were "outline-2026-10", measured on outline counts
MODES = ("region", "tiled", "whole")
WHOLE_CELLS = 20_000                      # a run of this many cells is a whole sample (a 1.2 mm region holds a few
                                          # thousand)

_TRIAL = {"history_iterations": 50, "settle_after_burnin_iterations": 20, "min_history_results": 10, "mult": 2.0,
          "floor_per100": 1.0, "min_results": 5, "min_iterations": 20, "horizon_iterations": 300}
_AREAS = {"window": 3, "alpha": 0.5, "ref_dt": 1.0, "enter_results": 2, "stay_results": 3, "enter_s": 2.0,
          "stay_s": 10.0, "enter_min_m": 5.0, "enter_mult": 1.5, "enter_add": 0.04, "wilson_z": 1.645,
          "top_frac": 0.1, "stay_mult": 1.25, "stay_add": 0.02, "cap_swap_ratio": 2.0, "listed": 3,
          "letter_jaccard": 0.2, "letter_rest_s": 30.0, "rose_z": 2.0, "rose_points": 0.03, "tissue_min_j": 1.0}
_TILED_AREAS = {"bin_um": 100.0, **_AREAS, "enter_min_j": 40.0, "stay_min_j": 30.0, "cap": 8, "min_area_j": 300.0}
_DETAIL_AREAS = {"bin_um": 50.0, **_AREAS, "enter_min_j": 30.0, "stay_min_j": 20.0, "cap": 4, "min_area_j": 120.0}


def _entry(headline_weight: float, matched_fixed: float, matched_model: float, areas: dict) -> dict:
    return {"headline_weight": headline_weight, "headline_ref_s": 1.0, "shown_margin": 0.3, "sampler_k": 0.6,
            "matched_fixed": matched_fixed, "matched_model": matched_model, "local_min_judged": 300,
            "local_min_paired": 400, "trial": dict(_TRIAL), "areas": dict(areas)}


# Measured on proseg's count (2026-10-31): headline_weight, the largest weight whose arrow moves from one result to
# the next no more than the outline count's did at 0.5, at the app's pace (proseg's count moves more between results,
# and a tiled region's results come about half as often); matched_model, the widest range of a change since the start
# (or of drift's effect) across the learnings that score these runs (a sample's own, learned three times, and the own
# one of a crop cut from it, which read it with opposite signs: an arrow under it is drawn hollow), rounded up. The rest
# were measured again and kept: sampler_k (cells-only runs need 0.5, too thin a margin to lower it to), matched_fixed
# (the box's largest gap between repeat runs, 0.94), the support, the trial (false alarms 0-1.1 %) and the areas. Whole
# samples: the outline count's values, not re-measured yet.
TABLE: dict[str, dict] = {
    "region": _entry(0.2, 1.0, 2.3, _DETAIL_AREAS),
    "tiled": _entry(0.1, 1.0, 1.5, _TILED_AREAS),
    "whole": _entry(0.5, 0.3, 0.6, _TILED_AREAS),
}
MEASURED = {"region": True, "tiled": True, "whole": False}     # re-measured on proseg's own count
# Go deeper's and the worker's constants (one value in every mode) kept, not measured again, in each mode
# (CALIBRATION.md): "glow" the pink dots' thresholds (engine/qc_scoring.py GLOW_MIN, LIKELY_MIN, LIKELY_W), "where_words"
# the where words' (engine/deeper_where.py FEW_IN_CELL, EDGE_*, LAYER_RATIO), "start_band" the cell against itself at
# the start (analysis/worker.py DEEP_START_BAND, DEEP_START_KEEP), "min_counts" the UMAP's fewest transcripts per cell
# (its DEFAULTS), "like_depth" Compare's like-for-like depth (LIKE_DEPTH). The page adds a caveat to their tooltips.
KEPT = {"region": (), "tiled": ("glow", "where_words"),
        "whole": ("glow", "where_words", "start_band", "min_counts", "like_depth")}


def mode_of(cells, tiled: bool) -> str:
    """The mode of a run of `cells` cells (info.json "cells"; None while unknown): "whole" from WHOLE_CELLS on, else
    "tiled" for a tiled run, else "region"."""
    if isinstance(cells, (int, float)) and not isinstance(cells, bool) and math.isfinite(cells) \
            and cells >= WHOLE_CELLS:
        return "whole"
    return "tiled" if tiled else "region"


def run_mode(meta: dict, info: dict | None) -> str:
    """A run's mode, from its meta and info.json (None before it is written): mode_of its cells, tiled when the run
    is (meta mode "tiled") or its prior is stored in tiles (info prior format "tiles"): the one rule the check, the
    settings test and the page's caveats read."""
    info = info if isinstance(info, dict) else {}
    prior = info.get("prior") if isinstance(info.get("prior"), dict) else {}
    tiled = (meta or {}).get("mode") == "tiled" or prior.get("format") == "tiles"
    return mode_of(info.get("cells"), tiled)


def value(mode: str, key: str):
    """One constant of a mode ("region" for an unknown one); a dict comes as a copy. KeyError for an unknown key."""
    found = TABLE.get(mode, TABLE["region"])[key]
    return dict(found) if isinstance(found, dict) else found


def block(mode: str) -> dict:
    """What a summary (and a run, engine/runs.py Run.public) says of the constants it was judged with: {mode, id,
    measured (false: not re-measured on proseg's own count yet for runs like this one, a caveat for the page's tooltip),
    kept (the names of Go deeper's and the worker's constants kept, not measured, in this mode: KEPT)}."""
    mode = mode if mode in TABLE else "region"
    return {"mode": mode, "id": CONSTANTS_ID, "measured": bool(MEASURED.get(mode, False)), "kept": list(KEPT[mode])}
