import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, statSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import * as Q from '../src/qc-deeper.js';
import { FIXTURE, expectOf, isMain, keepSuspects, testSuspects, text, writes } from './make-qc-scoring.mjs';

// Go deeper's scoring, the page's and the engine's: engine/qc_scoring.py is the same scoring in Python (the engine fixes
// a test's suspects with it before any run, exactly as they glowed here). Both sides recompute every case of one shared
// fixture (fixtures/qc-scoring.json; tests/test_qc_scoring.py is the Python side): real worker answers and where
// replies (version 3: each point's proseg share w) from the engine's own test runs, the strip case, then edge
// cases at their exact values: P = w x pi on any owner, the likely rule, the sources' support. Here each case's expect is worked
// out again from qc-deeper.js, so a change to the page's scoring shows; the port's constants are the page's; the
// fixture's kept lists are its own copy's of the page's two steps, and theirs while the window still keeps a list of
// suspects (keepSuspects) and the panel still checks one (testSuspects); and the maker rewrites the fixture only when
// asked.
const CASES = JSON.parse(readFileSync(FIXTURE, 'utf8'));
const byName = new Map(CASES.map(c => [c.name, c]));

test('every case of the shared fixture is what the page makes of it', () => {
  assert.ok(CASES.length >= 100 && byName.size === CASES.length, 'cases named once each');
  for (const c of CASES) assert.deepEqual(expectOf(c), c.expect, c.name);
});

test('the fixture stays small, one case a line, as the maker writes it', () => {
  assert.ok(statSync(FIXTURE).size < 400 * 1000);
  assert.equal(readFileSync(FIXTURE, 'utf8'), text(CASES));
});

test('the fixture keeps the edges the port turns on', () => {
  const need = ['world_cell_0_ready', 'world_cell_8_too_few', 'world_cell_11_empty', 'hand_made_detail', 'hand_made_whole_sample',
    'chips_tied_by_name_utf16', 'chips_floor', 'chips_more_than_six_own_overlap', 'chips_names_escaped_in_the_key',
    'rows_more_than_five_and_ties', 'rows_share_exactly_one_percent', 'rows_same_kind_from_similar', 'rows_same_kind_classes_and_known',
    'rows_none_at_code_9', 'rows_shown_at_code_12', 'rows_seq_start', 'rows_numbers_written_as_javascript_does',
    'state_too_few', 'state_empty', 'state_not_checked', 'state_v2_refused',
    'scores_each_branch', 'scores_without_near', 'scores_w_and_the_likely_rule', 'scores_support_by_source',
    'scores_float32_and_to_fixed_ties', 'places_ranked_by_depth', 'a2_strip_far', 'a2_strip_beside',
    'refused_v2', 'refused_w_missing', 'taken_v_three_point_o',
    'kept_more_than_400', 'rows_labels_one_past_2_53', 'rows_cell_at_2_53', 'scores_own_near_past_2_53'];
  assert.deepEqual(need.filter(name => !byName.has(name)), []);
  const e = name => byName.get(name).expect;
  // Whole numbers past 2^53 add up as doubles: labels 2^53 and 2^53 + 1 are one row (its first place, the last
  // neighbour); own_near's + 1 and + 2 are lost.
  const big = 2 ** 53;
  assert.deepEqual(e('rows_labels_one_past_2_53').params.rows.map(r => [r.label, r.cell]), [[big, big], [8, 7]]);
  assert.equal(e('rows_cell_at_2_53').params.req.params.cell, big - 1);
  assert.deepEqual(e('scores_own_near_past_2_53').scores.genes.map(g => g.fr), [big / 3, 4 / big]);
  // P: float32 before the cut-offs; the kept list's ties half up; more than 400 glowing, 400 kept.
  const glowing = new Set(e('scores_float32_and_to_fixed_ties').scores.points.map(p => p.gene));
  assert.ok(glowing.has('F_0.199999999') && !glowing.has('F_0.19999998'));
  assert.deepEqual(e('scores_float32_and_to_fixed_ties').kept.filter(s => s.gene.startsWith('T_')).map(s => s.p), [0.563, 0.313]);
  assert.ok(e('kept_more_than_400').scores.points.length > 400 && e('kept_more_than_400').kept.length === Q.SUSPECTS_SENT);
  // P = w x pi on any owner (a point proseg moved in from a source's outline glows; one inside the cell's it does not
  // count does not); likely: pi >= 0.5 with half of it for the cell; the support of the sources near their glow, none
  // for cell-f, 30 µm from the strip of pink dots it would be named for.
  const each = e('scores_each_branch').scores.points;
  assert.deepEqual([...new Set(each.map(p => p.owner))].sort(), [0, 1, 2]); assert.ok(each.every(p => p.w > 0 && p.P >= Q.GLOW_MIN));
  assert.ok(each.every(p => Math.abs(p.P - p.w * p.pi) < 1e-6), 'P is the share times the likelihood');
  const likely = e('scores_w_and_the_likely_rule').scores.points.filter(p => p.pi >= Q.LIKELY_MIN && p.w >= Q.LIKELY_W).map(p => [p.gene, p.w]);
  assert.deepEqual(likely, [['GAVE', 1], ['GAVE', 0.5], ['HALF', 1]]);
  assert.deepEqual(e('scores_support_by_source').scores.support, [[0, 1.25]]);
  assert.deepEqual(e('a2_strip_far').scores.support, []); assert.deepEqual(e('a2_strip_beside').scores.support, [[0, 13 * 0.875]]);
  // Places shared, ranked by depth; every refusal answered with nothing.
  assert.ok(e('places_ranked_by_depth').scores.points.some(p => p.of > 1));
  const refused = CASES.filter(c => c.name.startsWith('refused_'));
  assert.ok(refused.length >= 40 && refused.every(c => c.expect.scores === null && c.expect.kept === null));
});

test("the port's constants are the page's", () => {
  // engine/qc_scoring.py writes them as NAME = value or NAME, NAME = value, value (# comments after).
  const source = readFileSync(new URL('../../engine/qc_scoring.py', import.meta.url), 'utf8');
  const port = new Map();
  for (const line of source.split('\n')) {
    const m = line.match(/^([A-Z_]+(?:, [A-Z_]+)*) = ([^#]+?)\s*(?:#.*)?$/);
    if (!m) continue;
    const names = m[1].split(', '), value = m[2].trim();
    const values = names.length > 1 ? value.split(/,\s*/) : [value];
    names.forEach((name, i) => port.set(name, values[i]));
  }
  const number = name => Number(port.get(name));
  for (const name of ['RESULT_VERSION', 'RANK_RATIO', 'CHIPS', 'ROWS', 'ROW_MIN', 'SOURCES_MIN', 'SAME_KIND', 'WHERE_GENES', 'WHERE_OWN',
    'GLOW_MIN', 'LIKELY_MIN', 'LIKELY_W', 'SUSPECTS_SENT', 'MATCH_UM']) assert.equal(number(name), Q[name], name);
  // A Python tuple: (3,) is [3].
  const tuple = text => JSON.parse(text.replace(/^\(/, '[').replace(/,?\)$/, ']').replace(/'/g, '"'));
  assert.deepEqual(tuple(port.get('WHERE_V')), [...Q.WHERE_V]);
  assert.deepEqual(tuple(port.get('HELD')), [...Q.HELD]);
  assert.equal(number('OWNER_CELL'), 1);
});

/** Every case's kept list checked: the fixture's own copy of both steps (make-qc-scoring.mjs) gives the fixture's list,
 *  and the window's keepSuspects (`method`) and the panel's testSuspects (`check`) give the same, each while the page
 *  still has it (null: it has not). How many cases were checked. */
function checkKept(method, check) {
  let compared = 0;
  for (const c of CASES) {
    const reply = Q.normaliseWhere(c.reply, c.expect.params.req ?? null);
    if (!reply) continue;
    const scores = Q.suspectScores(reply, Q.normalise(c.result)), list = keepSuspects(reply, scores);
    assert.deepEqual(JSON.parse(JSON.stringify(testSuspects(list))), c.expect.kept, c.name);
    if (method) {
      const window = { cell: 1 };
      method.call(window, reply, scores);
      assert.deepEqual(JSON.parse(JSON.stringify(window.kept.list)), JSON.parse(JSON.stringify(list)), c.name);
    }
    if (check) assert.deepEqual(JSON.parse(JSON.stringify(check(list))), c.expect.kept, c.name);
    compared++;
  }
  return compared;
}

test("the fixture's kept lists are its copy's, and the window's keepSuspects and the panel's testSuspects while the page has them", async () => {
  // The page stops sending suspects (the engine fixes them itself), so either may go: the fixture keeps its own copy of
  // both steps, checked here against the page's for as long as it has them. With neither the copy is still checked,
  // never skipped (the gate fails on a skip).
  const method = typeof Q.QcDeeper?.prototype?.keepSuspects === 'function' ? Q.QcDeeper.prototype.keepSuspects : null;
  let panel = null;
  try { panel = await import('../src/qc-panel.js'); } catch { panel = null; }
  const check = typeof panel?.testSuspects === 'function' ? panel.testSuspects : null;
  assert.ok(checkKept(method, check) >= 30);
  assert.ok(checkKept(null, null) >= 30);
});

test('the maker rewrites the fixture only when asked, never under node --test', () => {
  // A bare `node --test` runs every .mjs file under test/, the maker too: it must not rebuild the golden from the page.
  const maker = fileURLToPath(new URL('./make-qc-scoring.mjs', import.meta.url));
  assert.ok(isMain(['node', maker]) && !isMain(['node', fileURLToPath(import.meta.url)]) && !isMain(['node']));
  assert.equal(writes(['node', maker, '--write'], {}), true);
  assert.equal(writes(['node', maker], {}), false);
  assert.equal(writes(['node', maker, '--write'], { NODE_TEST_CONTEXT: 'child-v8' }), false);
  assert.equal(writes(['node', fileURLToPath(import.meta.url), '--write'], {}), false);
  assert.equal(writes(), false);                       // this run, the gate's or a bare node --test
});

test('rankOrder moves a row up only past a quarter more (the port checks the same table)', () => {
  const v = values => k => values[k];
  assert.deepEqual(Q.rankOrder(['a', 'b'], ['a', 'b'], v({ a: 1.0, b: 1.2 })), ['a', 'b']);
  assert.deepEqual(Q.rankOrder(['a', 'b'], ['a', 'b'], v({ a: 1.0, b: 1.3 })), ['b', 'a']);
  assert.deepEqual(Q.rankOrder(['a'], ['a', 'c', 'd'], v({ a: 5, c: 4, d: 3 })), ['a', 'c', 'd']);
  assert.deepEqual(Q.rankOrder(['a'], ['a', 'z'], v({ a: 1, z: 10 })), ['z', 'a']);
  assert.deepEqual(Q.rankOrder(['x', 'a'], ['a'], v({ a: 1 })), ['a']);
  assert.deepEqual(Q.rankOrder(['a', 'b', 'c'], ['a', 'b', 'c'], v({ a: 1, b: 2, c: 4 })), ['c', 'b', 'a']);
  assert.deepEqual(Q.rankOrder(null, ['e', 'f', 'a'], v({ e: 0, f: 0, a: 1 })), ['a', 'e', 'f']);
});
