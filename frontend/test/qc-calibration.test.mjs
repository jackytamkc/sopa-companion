// The calibration on proseg's own count (2026-10-31), plugged in on the page: its twins of the engine's measured values
// are the measurement's own (fixtures/qc-calibration.json; tests/test_qc_calibration.py holds the engine to the same
// file): the model's own threshold a change that carries none is judged with (a region's matched_model), drift's
// confidence line (likely, sized: about 1 in 100), Go deeper's start band where a reply has none (normalise's and
// startChange's), the floor of "too few to tell" and the pink dots' thresholds.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { MODEL_THRESHOLD, QC_TEXT, RULE_COPY, likeForLike } from '../src/qc-copy.mjs';
import { GLOW_MIN, LIKELY_MIN, LIKELY_W, WHERE_FEW, normalise } from '../src/qc-deeper.js';
import { START_BAND, startChange } from '../src/qc-deeper-copy.mjs';

const CAL = JSON.parse(readFileSync(new URL('./fixtures/qc-calibration.json', import.meta.url), 'utf8'));

test("the mixing check's twins: a change carrying no model threshold is judged with a region's; drift is likely, about 1 in 100", () => {
  assert.equal(MODEL_THRESHOLD, CAL.qc_constants.TABLE.region.matched_model);
  const change = { matched: -(MODEL_THRESHOLD - 0.1), threshold: 1.0, meaningful: true, direction: 'fewer' };
  assert.equal(likeForLike(change).light, true, 'under it: a hollow arrow');
  assert.equal(likeForLike({ ...change, matched: -MODEL_THRESHOLD }).light, false, 'from it on: a full one');
  assert.equal(RULE_COPY.R3.confidence, CAL.rule_R3.CONFIDENCE); assert.equal(RULE_COPY.R3.sized, CAL.rule_R3.sized);
  assert.equal(QC_TEXT[RULE_COPY.R3.sized], 'Likely to help: about 1 in 100');
});

test("Go deeper's twins: the start band where a reply has none, the floor of too few to tell, the pink dots' thresholds", () => {
  const band = CAL.worker.DEEP_START_BAND;
  assert.equal(START_BAND, band);
  assert.equal(startChange(116, 2, { points: band - 0.01 }).key, 'deepSameAsStart', 'inside the band: the same');
  assert.equal(startChange(116, 2, { points: band }).key, 'deepUpFromStart');
  assert.equal(startChange(2, 116, { points: -band }).key, 'deepDownFromStart');
  assert.equal(WHERE_FEW, CAL.where_words.FEW_IN_CELL);
  assert.deepEqual({ GLOW_MIN, LIKELY_MIN, LIKELY_W }, CAL.pink_dots);
});

test("a Go deeper reply without a band goes through normalise to startChange with the worker's band, as the window reads it", () => {
  // The window normalises every reply before startChange sees it: a start_change without a band (or with none above 0)
  // reads the worker's measured band on that path too, not an older one.
  const band = CAL.worker.DEEP_START_BAND, reply = change => ({ v: 1, state: 'ready', seq: 12, code: 116, start_change: change });
  const word = change => startChange(116, 2, normalise(reply(change)).start_change).key;
  assert.equal(normalise(reply({ points: 4 })).start_change.band, band);
  assert.equal(word({ points: band - 0.01 }), 'deepSameAsStart', 'inside the band: the same');
  assert.equal(word({ points: band }), 'deepUpFromStart');
  assert.equal(word({ points: -band }), 'deepDownFromStart');
  assert.equal(word({ points: band - 0.01, band: 0 }), 'deepSameAsStart', 'a band of 0 is none');
  // A reply's own band wins (the worker's, as it sent it).
  assert.equal(normalise(reply({ points: 4, band: 3 })).start_change.band, 3);
  assert.equal(word({ points: 4, band: 3 }), 'deepUpFromStart');
});
