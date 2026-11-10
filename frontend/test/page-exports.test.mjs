import test from 'node:test';
import assert from 'node:assert/strict';
import { readdirSync, readFileSync } from 'node:fs';
import { STAIN_TEXT } from '../src/stain.mjs';
import { EXCLUSION_TEXT } from '../src/stain-exclusions.mjs';

// Exports nothing in the page called, some kept alive only by their own tests, were
// taken out with those tests (a test of code the page never runs counts as coverage it does not give), or moved into
// them (a deck's keys as functions, the check's strings Go deeper reuses). They stay out, and nothing in the page names
// them; nor the mixing check's own copies of api.js's requests. The three the development demos use
// (qc-chart.mjs layoutTimeline and renderTimeline, umap.js setAutoRotate) stay for them. Left by the removed previous-run
// line, Runs list and multi-run close prompt: the chart's value at an iteration, the request deleting a run, and the
// session's word list only its test read.
const GONE = {
  'qc-chip.mjs': ['createQcChip'], 'compare.mjs': ['MAX_POINTS', 'STATES'], 'qc-chart.mjs': ['qcSnapshot', 'valueAtIteration'], 'stage.js': ['mixingShare'],
  'timeline.js': ['laneTop'], 'umap.js': ['layoutLabels'], 'qc-overlay.mjs': ['binColor'], 'qc-deeper.js': ['motionPlan'],
  'log-export.mjs': ['msLine'], 'qc-panel.js': ['createQcPanel'], 'qc-copy.mjs': ['QC_COPY'], 'qc-deeper-copy.mjs': ['REUSED'],
  'qc-client.mjs': ['QC_ENDPOINTS', 'qcRequest'], 'api.js': ['deleteRun'], 'session.mjs': ['sessionTexts'],
};
const src = new URL('../src/', import.meta.url);

test('exports the page never used are gone, and no page module names them', () => {
  const files = readdirSync(src).filter(name => /\.(m?js)$/.test(name));
  for (const [file, names] of Object.entries(GONE)) assert.ok(files.includes(file), file);
  for (const file of files) {
    const source = readFileSync(new URL(file, src), 'utf8');
    for (const name of Object.values(GONE).flat()) assert.doesNotMatch(source, new RegExp(`\\b${name}\\b`), `${file}: ${name}`);
  }
});

// The same parts left words, style rules and a name nothing uses: the previous run's key in the chart's legend, the old
// close prompt's run list, the pixel-size field the picker no longer has, the removal panel's copy of the cell values'
// error (stain-cells.mjs says it), and an alias of the chart's iterations.
test('words and style rules nothing on the page uses are gone', () => {
  const read = name => readFileSync(new URL(name, src), 'utf8');
  for (const [file, rule] of [['qc.css', 'qc-previous-key'], ['../styles/app.css', 'close-runs']])
    assert.doesNotMatch(read(file), new RegExp(`\\.${rule}\\b`), `${file}: .${rule}`);
  assert.equal('pixelSize' in STAIN_TEXT, false); assert.equal('valuesFailed' in EXCLUSION_TEXT, false);
  assert.doesNotMatch(read('qc-chart.mjs'), /\ballX\b/);
});
