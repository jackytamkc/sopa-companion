import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import { domFixture, deferred, settle } from './dom-fixture.mjs';
import { StainExclusions, EXCLUSION_TEXT } from '../src/stain-exclusions.mjs';
import { StainPatches } from '../src/stain-patches.mjs';
import { EXPLAIN_TEXT } from '../src/explain.mjs';
import { bannedWords } from '../src/qc-copy.mjs';
import { CellValues, CELL_VALUES_TEXT } from '../src/stain-cells.mjs';

// "Remove cells in an area" against the DOM fixture; explain.mjs writes its note into globalThis.document, the
// fixture's for the test.
const run = { id:'r', kind:'stain', state:'done', box:[0,0,200,200], progress:{done:1,revision:9}, exclusions:{version:0}, summary_model:{cells:100} };
const MARKUP = '<canvas id="stage"></canvas><section id="stain-exclusions" hidden><div id="stain-exclude-tools">' +
  '<button id="stain-exclude" type="button">Remove cells in an area</button><button id="stain-exclude-undo" type="button">' +
  'Undo</button><button id="stain-exclude-redo" type="button">Redo</button></div><p id="stain-exclude-help"></p>' +
  '<div id="stain-exclude-size-row"></div><p id="stain-exclude-sizes"></p>' +
  '<div id="stain-exclude-bright-row"><select id="stain-exclude-bright-channel"></select></div><p id="stain-exclude-brights"></p>' +
  '<p><span id="stain-exclude-total"></span><span id="stain-exclude-status"></span></p>' +
  '</section><input id="name" type="text"><textarea id="notes"></textarea><input id="tick" type="checkbox">';
const SQUARE = [[10,10],[80,10],[80,80],[10,80]], SMALL = [[100,100],[150,100],[150,150],[100,150]];
const LOW = [[20,120],[60,120],[60,160],[20,160]];
const area = (id, cells = 5, points = SQUARE) => ({ id, points, cells });
// `refresh`: what main.js does after a change (brings the run detail), here done at once unless given.
function panel(t, { refresh = async () => {},values=null } = {}) {
  const previous = globalThis.document, fixture = domFixture();
  globalThis.document = fixture.document;
  fixture.document.body.innerHTML = MARKUP;
  const canvas = fixture.$('stage');
  canvas.setPointerCapture = () => {}; canvas.hasPointerCapture = () => true; canvas.releasePointerCapture = () => {};
  // `spreads`: the reads of the cells' brightness (the brightness rule's line), apart from the areas' reads.
  const reads = [], writes = [], notices = [], draws = [], spreads = [], views = [];
  const pending = (list, op) => (...args) => { const d = deferred(); list.push({ op, args, ...d }); return d.promise; };
  // `view()`: the view moved (stage.js onChange: a pan or a zoom).
  const exclusions = new StainExclusions({ canvas, scale:1, toWorld:(x, y) => [x, y], toScreen:(x, y) => [x, y], onChange:fn => views.push(fn) },
    { stainExclusions:pending(reads, 'read'), stainExclude:pending(writes, 'post'), stainRestore:pending(writes, 'delete'),
      stainExcludeSize:pending(writes, 'size'), stainBrightness:pending(spreads, 'measure'),
      stainExcludeBrightness:pending(writes, 'bright') },
    { changed() { draws.push(1); }, refresh, notice:message => notices.push(message),values });
  t.after(() => { exclusions.reset(); globalThis.document = previous; });
  // A pointer event on the image (no `buttons`: none held, as for a hover, unless given); returns it, `stopped` when a
  // listener stopped it from reaching the page's own listeners.
  const fire = (type, x, y, extra = {}) => canvas.emit(type, { button:0, pointerId:1, offsetX:x, offsetY:y,
    preventDefault() {}, stopImmediatePropagation() { this.stopped = true; }, ...extra });
  const lasso = ([a, b, c, d] = SQUARE) => { fire('pointerdown', ...a); fire('pointermove', ...b); fire('pointermove', ...c); fire('pointerup', ...d); };
  // A click as the browser gives it: the pointer moves there, then a press, its release on the spot and the click.
  const click = (x, y) => { fire('pointermove', x, y); fire('pointerdown', x, y); const up = fire('pointerup', x, y); fire('click', x, y); return up; };
  // A key pressed with focus on `target` (the page's body unless given), through the page's keydown listeners; `stopped`
  // when a listener kept it from the others.
  const key = (name, init = {}) => {
    const e = { key:name, target:fixture.document.body, defaultPrevented:false, preventDefault() { this.defaultPrevented = true; },
      stopPropagation() { this.stopped = true; }, stopImmediatePropagation() { this.stopped = true; }, ...init };
    for (const fn of fixture.document.listeners.get('keydown') || []) fn(e);
    return e;
  };
  // The run finished and its areas read (none unless given).
  const ready = async (data = { version:0, areas:[], removed:0, min_area_um2:null, small:0 }) => {
    exclusions.update(run); reads.at(-1).resolve(data); await settle();
  };
  return { exclusions, button:exclusions.button, undo:exclusions.undoButton, redo:exclusions.redoButton, reads, writes, notices,
    draws, spreads, canvas, document:fixture.document, fire, lasso, click, key, ready, $:fixture.$,
    view:() => views.forEach(fn => fn()), history:() => { const s = exclusions.snapshot(); return [s.undo, s.redo, s.queued]; } };
}
const busyNow = button => button.getAttribute('aria-busy') === 'true';
// What explain.mjs showed beside a button ('' when nothing is shown).
const note = button => button.nextSibling?.className === 'missing-note' && !button.nextSibling.hidden ? button.nextSibling.textContent : '';
// A 2-D context that records what is drawn: [method, ...arguments] and [property, value]. Text measures 6 px a letter.
const recorder = () => {
  const calls = [];
  return new Proxy({}, { get: (_, name) => name === 'calls' ? calls : name === 'measureText' ? text => ({ width:text.length * 6 }) :
    (...args) => calls.push([name, ...args]),
    set: (_, name, value) => (calls.push([name, value]), true) });
};
const polygon = (g, region) => g.polygon(region.points);

test('Remove cells in an area is never greyed out: a press says why the areas cannot change yet', async t => {
  const h = panel(t), button = h.button;
  h.exclusions.update({ ...run, state:'running' });
  assert.equal(button.disabled, false);
  await button.click();
  assert.equal(h.exclusions.drawing, false);
  assert.match(note(button), /Pause the run or let it finish first\./);
  // Finished: the areas load first (busy, the label kept; a press meanwhile says so), then a press turns the mode on.
  h.exclusions.update(run);
  assert.equal(busyNow(button), true); assert.equal(button.textContent, EXCLUSION_TEXT.title);
  await button.click();
  assert.ok(note(button).includes(EXPLAIN_TEXT.busy)); assert.equal(h.exclusions.drawing, false);
  h.reads[0].resolve({ version:0, areas:[], removed:0 }); await settle();
  assert.equal(busyNow(button), false); assert.equal(note(button), '', 'ready now: the note goes');
  await button.click();
  assert.equal(h.exclusions.drawing, true); assert.equal(button.classList.contains('on'), true);
  assert.equal(button.textContent, EXCLUSION_TEXT.done);
  // Areas that could not be loaded: the press says so (and the notice did).
  h.exclusions.update({ ...run, id:'other' });
  h.reads[1].reject(Error('offline')); await settle();
  assert.deepEqual(h.notices, [EXCLUSION_TEXT.error]);
  await button.click();
  assert.equal(note(button).includes(EXCLUSION_TEXT.error), true); assert.equal(h.exclusions.drawing, false);
});

test('Undo and Redo are never greyed out: a press says what is missing, and clears what the others said', async t => {
  const h = panel(t);
  h.exclusions.update({ ...run, state:'running' });
  for (const button of [h.undo, h.redo]) {
    assert.equal(button.disabled, false); await button.click();
    assert.match(note(button), /Pause the run or let it finish first\./);
  }
  await h.ready();
  await h.undo.click(); assert.match(note(h.undo), /There is nothing to undo yet/);
  await h.redo.click(); assert.match(note(h.redo), /There is nothing to redo\./);
  assert.equal(note(h.undo), '', 'a press clears what the others said');
  assert.equal(h.writes.length, 0);
  // A change on its way holds nothing up either: Delete on an area clicked and Undo wait their turn.
  await h.button.click(); await h.redo.click(); assert.match(note(h.redo), /There is nothing to redo\./);
  h.lasso(); assert.equal(note(h.redo), '', 'a change clears what a press said');
  h.writes[0].resolve({ version:1, areas:[area(1)], removed:5 }); await settle();
  h.lasso(SMALL);
  h.click(40, 40); assert.equal(h.exclusions.snapshot().selected, 1);
  h.key('Delete'); await h.undo.click();
  assert.equal(h.writes.length, 2); assert.deepEqual(h.history(), [1, 0, 2]);
});

test('Undo with nothing to undo says what Undo covers and how an area\'s cells come back; so do the notices that clear it', async t => {
  const h = panel(t); await h.ready({ version:0, areas:[area(7)], removed:5 });
  await h.undo.click();
  assert.match(note(h.undo), /Undo covers the changes made in this window since the run was opened\. Click an area and press Delete to bring its cells back\./);
  for (const words of [EXCLUSION_TEXT.changed, EXCLUSION_TEXT.failed]) assert.match(words, /undo history was cleared; click an area and press Delete to bring its cells back\./);
  // The help line in the mode and out of it says so too; its tooltip has the details. No × is left to press.
  for (const words of [EXCLUSION_TEXT.idle, EXCLUSION_TEXT.hint]) assert.match(words, /Click an area and press Delete/);
  assert.equal(h.$('stain-exclude-help').title, EXCLUSION_TEXT.helpTip);
  assert.match(EXCLUSION_TEXT.helpTip, /where areas overlap, the one on top, and a click again on the same spot the one under it/);
  // Cells inside an area may also be left out for another reason: the tooltip says which stay out once it goes.
  assert.match(EXCLUSION_TEXT.helpTip, /apart from those also inside another area or outside the size or brightness range/);
  assert.equal(Object.values(EXCLUSION_TEXT).some(words => words.includes('×')), false);
});

test('a quick drag keeps its corners: every move the browser merged into one event is a point of the stretch', async t => {
  const h = panel(t); await h.ready();
  await h.button.click();
  const [a, b, c, d] = SQUARE;
  // The move to the second corner came merged into the move to the third (a busy frame): one pointermove carries both.
  h.fire('pointerdown', ...a);
  h.fire('pointermove', ...c, { clientX:c[0], clientY:c[1], getCoalescedEvents:() => [{ clientX:b[0], clientY:b[1] }, { clientX:c[0], clientY:c[1] }] });
  h.fire('pointerup', ...d);
  assert.equal(h.writes[0].op, 'post'); assert.deepEqual(h.writes[0].args, ['r', { version:0, points:SQUARE }]);
});

test('the mode stays on: several outlines in a row with one press of the button, until Done removing', async t => {
  const h = panel(t); await h.ready();
  await h.button.click();
  assert.equal(h.$('stain-exclude-help').textContent, EXCLUSION_TEXT.hint);
  h.lasso();
  assert.equal(h.writes[0].op, 'post'); assert.deepEqual(h.writes[0].args, ['r', { version:0, points:SQUARE }]);
  h.writes[0].resolve({ version:1, areas:[area(1, 12)], removed:12 }); await settle();
  h.lasso(SMALL);
  assert.equal(h.writes.length, 2, 'the next outline goes without another press');
  assert.deepEqual(h.writes[1].args, ['r', { version:1, points:SMALL }]);
  h.writes[1].resolve({ version:2, areas:[area(1, 12), area(2, 4, SMALL)], removed:16 }); await settle();
  assert.equal(h.exclusions.total.textContent, 'Keeps 84 of 100 cells');
  assert.deepEqual(h.exclusions.snapshot(), { active:true, drawing:true, busy:false, loading:false, queued:0, undo:2, redo:0,
    version:2, removed:16, areas_version:null, areas:2, selected:null, outline:0, size:null, small:null, brightness:{}, dim:null,
    kept:84, cells:100 });
  // Done removing: the mode ends and a drag on the image draws nothing.
  await h.button.click();
  assert.equal(h.button.textContent, EXCLUSION_TEXT.title); assert.equal(h.button.classList.contains('on'), false);
  assert.equal(h.$('stain-exclude-help').textContent, EXCLUSION_TEXT.idle);
  h.lasso(); assert.equal(h.writes.length, 2);
});

test('clicks place corners and a click on the first point removes the cells inside; a double-click or Enter closes too', async t => {
  const h = panel(t); await h.ready(); await h.button.click();
  for (const p of SQUARE) h.click(...p);
  assert.equal(h.writes.length, 0, 'corners only: nothing is sent yet'); assert.equal(h.exclusions.snapshot().outline, 4);
  const close = h.click(13, 12);                                   // 3.6 px from the first corner
  assert.equal(close.stopped, true, 'the press is the outline\'s, never the page\'s');
  assert.equal(h.writes[0].op, 'post'); assert.deepEqual(h.writes[0].args, ['r', { version:0, points:SQUARE }]);
  assert.equal(h.exclusions.snapshot().outline, 0); assert.equal(h.exclusions.drawing, true, 'the mode stays on');
  h.writes[0].resolve({ version:1, areas:[area(1)], removed:5 }); await settle();
  // A double-click: its two clicks place the last corner twice, and the browser's dblclick closes without the second.
  for (const p of SMALL) h.click(...p);
  h.click(...SMALL[3]);
  const double = h.canvas.emit('dblclick', { preventDefault() {}, stopImmediatePropagation() { this.stopped = true; } });
  assert.equal(double.stopped, true, 'in the mode a double-click never fits the view');
  assert.deepEqual(h.writes[1].args, ['r', { version:1, points:SMALL }]);
  h.writes[1].resolve({ version:2, areas:[area(1), area(2, 3, SMALL)], removed:8 }); await settle();
  // Enter: with fewer than three corners it says so; with three it closes. Enter on a focused button is the button's.
  h.click(20, 120); h.click(60, 120);
  assert.equal(h.key('Enter').defaultPrevented, true); assert.equal(h.notices.at(-1), EXCLUSION_TEXT.small);
  h.click(60, 160);
  assert.equal(h.key('Enter', { target:h.undo }).defaultPrevented, false); assert.equal(h.writes.length, 2);
  h.key('Enter');
  assert.deepEqual(h.writes[2].args, ['r', { version:2, points:[[20,120],[60,120],[60,160]] }]);
  // Out of the mode a double-click is the page's.
  await h.button.click();
  assert.equal(h.canvas.emit('dblclick', { stopImmediatePropagation() { this.stopped = true; } }).stopped, undefined);
});

test('a click on the first point of two corners says three are needed and places nothing: never a sliver', async t => {
  const h = panel(t); await h.ready(); await h.button.click();
  h.click(10, 10); h.click(80, 10);
  h.click(11, 10); h.click(11, 10);                                // two clicks on the first point (or a double-click)
  assert.deepEqual(h.notices, [EXCLUSION_TEXT.small, EXCLUSION_TEXT.small]);
  const double = h.canvas.emit('dblclick', { preventDefault() {}, stopImmediatePropagation() { this.stopped = true; } });
  assert.equal(double.stopped, true); assert.equal(h.notices.length, 3);
  assert.equal(h.writes.length, 0); assert.equal(h.exclusions.snapshot().outline, 2);
  h.click(80, 80); h.click(12, 11);
  assert.deepEqual(h.writes[0].args, ['r', { version:0, points:[[10,10],[80,10],[80,80]] }]);
});

test('in the mode a double-click with no outline being drawn is the page\'s (it fits the view) and leaves nothing', async t => {
  const h = panel(t); await h.ready(); await h.button.click();
  h.click(50, 50); h.click(51, 50);
  const double = h.canvas.emit('dblclick', { preventDefault() {}, stopImmediatePropagation() { this.stopped = true; } });
  assert.equal(double.stopped, undefined, 'it reaches the page');
  assert.equal(h.exclusions.snapshot().outline, 0); assert.deepEqual(h.notices, []); assert.equal(h.writes.length, 0);
});

test('Done removing with an outline open closes it first, so its cells go; one with too few corners goes with the mode', async t => {
  const h = panel(t); await h.ready(); await h.button.click();
  for (const p of SQUARE.slice(0, 3)) h.click(...p);
  await h.button.click();
  assert.equal(h.exclusions.drawing, false);
  assert.deepEqual(h.writes[0].args, ['r', { version:0, points:SQUARE.slice(0, 3) }]);
  h.writes[0].resolve({ version:1, areas:[area(1, 1, SQUARE.slice(0, 3))], removed:1 }); await settle();
  // One cell is said as one, on the Keeps line and beside the area clicked on.
  assert.equal(h.exclusions.total.textContent, 'Keeps 99 of 100 cells');
  h.click(60, 20); assert.equal(h.exclusions.snapshot().selected, 1);
  const g = recorder(); h.exclusions.draw(g, polygon);
  assert.deepEqual(g.calls.filter(([n]) => n === 'fillText').map(([, text]) => text), ['1 cell inside · Delete removes this area']);
  await h.button.click(); h.click(100, 150); h.click(150, 150);
  await h.button.click();
  assert.equal(h.exclusions.drawing, false); assert.equal(h.writes.length, 1); assert.equal(h.exclusions.snapshot().outline, 0);
});

test('a pan\'s moves, which the page makes after the outline sees them, tell the outline the view moved', async t => {
  const h = panel(t); await h.ready(); await h.button.click();
  h.click(10, 10); h.fire('pointermove', 30, 30);
  assert.equal(h.exclusions.draft.pointer.stale, undefined);
  h.fire('pointermove', 60, 30, { buttons:2 });                    // a right-drag pans
  assert.equal(h.exclusions.draft.pointer.stale, true);
  assert.equal(h.exclusions.snapshot().outline, 1);
});

test('a first read older than the run detail says is still shown, never an empty panel', async t => {
  const h = panel(t);
  h.exclusions.update({ ...run, exclusions:{ version:3, count:1, removed:5 } });
  h.reads[0].resolve({ version:2, areas:[area(1)], removed:5, min_area_um2:null, small:0 }); await settle();
  assert.equal(h.exclusions.snapshot().areas, 1); assert.equal(h.exclusions.total.textContent, 'Keeps 95 of 100 cells');
});

test('Backspace takes back a corner, Escape drops the outline and a second Escape leaves the mode; a lost press takes back only itself', async t => {
  const h = panel(t); await h.ready(); await h.button.click();
  h.click(10, 10); h.click(80, 10); h.click(80, 80);
  assert.equal(h.key('Backspace').defaultPrevented, true); assert.equal(h.exclusions.snapshot().outline, 2);
  // A drag after the corners, lost before its release: only the drag goes, and the release that may still come is no one's.
  h.fire('pointerdown', 80, 80); h.fire('pointermove', 60, 90); h.fire('pointermove', 40, 95);
  assert.equal(h.exclusions.snapshot().outline, 5);
  h.fire('pointercancel', 40, 95);
  assert.equal(h.exclusions.snapshot().outline, 2);
  assert.equal(h.fire('pointerup', 40, 95).stopped, undefined); assert.equal(h.exclusions.snapshot().outline, 2);
  // Escape in the middle of a drag: the whole outline goes, the rest of the press draws nothing, the mode stays on.
  h.fire('pointerdown', 10, 80); h.fire('pointermove', 30, 80);
  assert.equal(h.key('Escape').defaultPrevented, true);
  assert.equal(h.exclusions.snapshot().outline, 0); assert.equal(h.exclusions.drawing, true);
  h.fire('pointermove', 50, 80); h.fire('pointerup', 50, 80);
  assert.equal(h.exclusions.snapshot().outline, 0); assert.equal(h.writes.length, 0);
  // Escape with no outline leaves the mode; out of it Escape is the page's.
  assert.equal(h.key('Escape').defaultPrevented, true);
  assert.equal(h.exclusions.drawing, false); assert.equal(h.button.classList.contains('on'), false);
  assert.equal(h.key('Escape').defaultPrevented, false, 'out of the mode Escape is the page\'s');
  h.lasso(); h.click(10, 10); assert.equal(h.writes.length, 0); assert.equal(h.exclusions.snapshot().outline, 0);
});

test('the outline being drawn: its points, a dashed line on to the pointer, the first point larger where a press closes it', async t => {
  const h = panel(t); await h.ready(); await h.button.click();
  const draw = () => { const g = recorder(); h.exclusions.draw(g, polygon); return g.calls; };
  assert.equal(draw().some(([name]) => ['moveTo', 'fillRect'].includes(name)), false, 'no outline: none drawn');
  // Moves with no outline draw nothing again; with one, each move draws the dashed line again (drawing on demand).
  const before = h.draws.length;
  h.fire('pointermove', 30, 30); assert.equal(h.draws.length, before);
  h.click(10, 10); h.click(80, 10); h.click(80, 80);
  const placed = h.draws.length;
  h.fire('pointermove', 50, 60); assert.equal(h.draws.length, placed + 1);
  let calls = draw();
  const at = name => calls.findIndex(([n]) => n === name);
  assert.deepEqual(calls.filter(([n]) => n === 'moveTo' || n === 'lineTo'),
    [['moveTo', 10, 10], ['lineTo', 80, 10], ['lineTo', 80, 80], ['moveTo', 80, 80], ['lineTo', 50, 60]]);
  assert.deepEqual(calls[at('setLineDash')], ['setLineDash', [6, 4]], 'the line on to the pointer is dashed');
  assert.deepEqual(calls.filter(([n]) => n === 'setLineDash').at(-1), ['setLineDash', []], 'and the dash is put back');
  assert.deepEqual(calls.filter(([n]) => n === 'fillRect'), [['fillRect', 6, 6, 8, 8]]);
  // A right-drag pans (the page's): its moves are not the outline's, so the line stays where the pointer last was.
  assert.equal(h.fire('pointerdown', 50, 60, { button:2 }).stopped, undefined);
  assert.equal(h.fire('pointermove', 120, 120, { buttons:2 }).stopped, undefined);
  h.fire('pointerup', 120, 120, { button:2 });
  assert.deepEqual(draw().filter(([n]) => n === 'lineTo').at(-1), ['lineTo', 50, 60]);
  // Near the first point: it is white and larger, as a press there closes the outline.
  h.fire('pointermove', 13, 12);
  calls = draw();
  assert.deepEqual(calls.filter(([n]) => n === 'fillRect'), [['fillRect', 4, 4, 12, 12]]);
  assert.deepEqual(calls.filter(([n]) => n === 'fillStyle').at(-1), ['fillStyle', '#fff']);
});

test('the image shows a crosshair while the mode is on, and the button its colour, never a pressed state as well', async t => {
  const h = panel(t); await h.ready();
  assert.equal(h.canvas.classList.contains('removing'), false);
  await h.button.click();
  assert.equal(h.canvas.classList.contains('removing'), true); assert.equal(h.button.classList.contains('on'), true);
  assert.equal(h.button.getAttribute('aria-pressed'), null, 'the label says what a press does; no pressed state on top');
  h.key('Escape');
  assert.equal(h.canvas.classList.contains('removing'), false); assert.equal(h.button.classList.contains('on'), false);
  await h.button.click(); h.exclusions.reset();                         // another run or the picker: no crosshair left behind
  assert.equal(h.canvas.classList.contains('removing'), false);
  const css = readFileSync(new URL('../src/stain-exclusions.css', import.meta.url), 'utf8');
  assert.match(css, /\.stage\.removing\{cursor:crosshair\}/); assert.match(css, /#stain-exclude\.on\{/);
  const html = readFileSync(new URL('../index.html', import.meta.url), 'utf8');
  assert.doesNotMatch(html.match(/<button id="stain-exclude"[^>]*>/)[0], /aria-pressed/);
});

test("the mode's button keeps one width whatever its label: both labels are there for the style sheet to size it", async t => {
  const h = panel(t);
  assert.deepEqual({ ...h.button.dataset }, { off:EXCLUSION_TEXT.title, on:EXCLUSION_TEXT.done });
  const css = readFileSync(new URL('../src/stain-exclusions.css', import.meta.url), 'utf8');
  assert.match(css, /#stain-exclude\{display:inline-grid/);
  assert.match(css, /#stain-exclude::before,#stain-exclude::after\{visibility:hidden;height:0;overflow:hidden\}/);
  assert.match(css, /#stain-exclude::before\{content:attr\(data-off\)\}/); assert.match(css, /#stain-exclude::after\{content:attr\(data-on\)\}/);
});

test('an outline finished while one is on its way waits and goes after it with the new version, one at a time', async t => {
  const h = panel(t); await h.ready(); await h.button.click();
  h.lasso();
  assert.equal(h.exclusions.busy, true); assert.equal(busyNow(h.button), false, 'the mode is never held up');
  assert.equal(h.exclusions.status.textContent, EXCLUSION_TEXT.removing);
  h.lasso(SMALL);
  assert.equal(h.writes.length, 1, 'never two requests at once'); assert.equal(h.exclusions.queued, 1);
  assert.equal(h.exclusions.status.textContent, `${EXCLUSION_TEXT.removing} (1 waiting)`);
  // Done removing meanwhile ends the mode; what was drawn still goes.
  await h.button.click(); assert.equal(h.exclusions.drawing, false);
  h.writes[0].resolve({ version:1, areas:[area(1)], removed:5 }); await settle();
  assert.equal(h.writes.length, 2);
  assert.deepEqual(h.writes[1].args, ['r', { version:1, points:SMALL }], 'with the version the first change brought');
  assert.equal(h.exclusions.queued, 0);
  h.writes[1].resolve({ version:2, areas:[area(1), area(2, 3, SMALL)], removed:8 }); await settle();
  assert.equal(h.exclusions.busy, false); assert.equal(h.exclusions.status.textContent, '');
  assert.deepEqual(h.history(), [2, 0, 0]);
  // Another run while a change is on its way and one waits: everything lets go at once.
  await h.button.click(); h.lasso(); h.lasso(SMALL);
  h.exclusions.update({ ...run, id:'other' });
  assert.equal(h.exclusions.busy, false); assert.deepEqual(h.history(), [0, 0, 0]);
  h.writes[2].resolve({ version:3, areas:[], removed:0 }); await settle();
  assert.equal(h.writes.length, 3); assert.equal(h.exclusions.data, null);
});

test('a passive update from another window clears history and queued edits before another write',async t=>{
  const h=panel(t);await h.ready();await h.button.click();h.lasso();
  h.exclusions.update({...run,exclusions:{version:5}});h.lasso(SMALL);
  h.writes[0].resolve({version:1,areas:[area(1)],removed:5});await settle();
  h.reads[1].resolve({version:5,areas:[area(1),area(4)],removed:10});await settle();
  assert.equal(h.writes.length,1);assert.deepEqual(h.history(),[0,0,0]);
  assert.ok(h.notices.at(-1).includes(EXCLUSION_TEXT.changed));
});

test('Undo takes an outline back by its id; Redo posts its points again, and later steps follow the new id', async t => {
  const h = panel(t); await h.ready(); await h.button.click();
  h.lasso(); h.writes[0].resolve({ version:1, areas:[area(1)], removed:5 }); await settle();
  assert.equal(h.key('z', { ctrlKey:true }).defaultPrevented, true);
  assert.equal(h.writes[1].op, 'delete'); assert.deepEqual(h.writes[1].args, ['r', 1, 1]);
  assert.equal(h.exclusions.status.textContent, EXCLUSION_TEXT.restoring);
  h.writes[1].resolve({ version:2, areas:[], removed:0 }); await settle();
  assert.deepEqual(h.history(), [0, 1, 0]); assert.equal(h.exclusions.drawing, true);
  h.key('Z', { ctrlKey:true, shiftKey:true });
  assert.equal(h.writes[2].op, 'post'); assert.deepEqual(h.writes[2].args, ['r', { version:2, points:SQUARE }]);
  h.writes[2].resolve({ version:3, areas:[area(2)], removed:5 }); await settle();
  assert.deepEqual(h.history(), [1, 0, 0]);
  await h.undo.click();
  assert.deepEqual(h.writes[3].args, ['r', 2, 3], 'the area as the engine numbers it now');
  h.writes[3].resolve({ version:4, areas:[], removed:0 }); await settle();
  h.key('y', { ctrlKey:true });                                   // Ctrl+Y redoes too
  assert.deepEqual(h.writes[4].args, ['r', { version:4, points:SQUARE }]);
});

test('Undo of Delete on an area posts it again and Redo takes it away by its new id; an outline and a Delete on one area share it', async t => {
  const h = panel(t), points = [[20,20],[60,20],[60,60]];
  await h.ready({ version:0, areas:[area(7, 3, points)], removed:3 });
  h.click(50, 30); h.key('Delete');                                // Delete on an area from before this window
  assert.deepEqual(h.writes[0].args, ['r', 7, 0]);
  h.writes[0].resolve({ version:1, areas:[], removed:0 }); await settle();
  await h.undo.click();
  assert.deepEqual(h.writes[1].args, ['r', { version:1, points }]);
  h.writes[1].resolve({ version:2, areas:[area(8, 3, points)], removed:3 }); await settle();
  await h.redo.click();
  assert.deepEqual(h.writes[2].args, ['r', 8, 2]);
  h.writes[2].resolve({ version:3, areas:[], removed:0 }); await settle();
  assert.deepEqual(h.history(), [1, 0, 0]);
  // An outline, then Delete on its area: undoing both posts it again and then takes away the area posted, never the first id.
  await h.button.click(); h.lasso();
  h.writes[3].resolve({ version:4, areas:[area(9)], removed:5 }); await settle();
  h.click(40, 40); h.key('Delete');
  assert.deepEqual(h.writes[4].args, ['r', 9, 4]);
  h.writes[4].resolve({ version:5, areas:[], removed:0 }); await settle();
  await h.undo.click();
  assert.deepEqual(h.writes[5].args, ['r', { version:5, points:SQUARE }]);
  h.writes[5].resolve({ version:6, areas:[area(10)], removed:5 }); await settle();
  await h.undo.click();
  assert.deepEqual(h.writes[6].args, ['r', 10, 6]);
});

test('a new change after an undo clears what could be redone, also while it waits its turn', async t => {
  const h = panel(t); await h.ready(); await h.button.click();
  h.lasso(); h.writes[0].resolve({ version:1, areas:[area(1)], removed:5 }); await settle();
  await h.undo.click(); h.writes[1].resolve({ version:2, areas:[], removed:0 }); await settle();
  assert.deepEqual(h.history(), [0, 1, 0]);
  h.lasso(SMALL); h.writes[2].resolve({ version:3, areas:[area(2, 3, SMALL)], removed:3 }); await settle();
  assert.deepEqual(h.history(), [1, 0, 0]);
  await h.redo.click();
  assert.equal(h.writes.length, 3); assert.match(note(h.redo), /There is nothing to redo\./);
  // Counted through what is on its way and waiting: Undo while an outline is sent, then an outline; Redo has nothing left.
  h.lasso(); await h.undo.click(); h.lasso(SMALL);
  await h.redo.click();
  assert.match(note(h.redo), /There is nothing to redo\./); assert.deepEqual(h.history(), [1, 0, 2]);
});

test('a 409 with the areas at another version: read again, said, the history cleared; what waited is dropped, the mode stays on', async t => {
  const h = panel(t); await h.ready(); await h.button.click();
  h.lasso(); h.writes[0].resolve({ version:1, areas:[area(1)], removed:5 }); await settle();
  h.lasso(SMALL); h.writes[1].resolve({ version:2, areas:[area(1), area(2, 3, SMALL)], removed:8 }); await settle();
  await h.undo.click(); h.lasso();                                  // Undo on its way, an outline waiting
  assert.deepEqual(h.history(), [1, 0, 1]);
  h.writes[2].reject(Object.assign(Error('changed'), { status:409 })); await settle();
  assert.deepEqual(h.notices, [], 'what it was is known only once the areas are read');
  h.reads.at(-1).resolve({ version:4, areas:[area(1), area(5, 2, SMALL)], removed:7 }); await settle();
  assert.deepEqual(h.notices, [`${EXCLUSION_TEXT.changed} 1 later change was not sent.`]);
  assert.deepEqual(h.history(), [0, 0, 0]); assert.equal(h.writes.length, 3, 'the waiting outline was not sent');
  assert.equal(h.exclusions.drawing, true);
  assert.equal(h.exclusions.total.textContent, 'Keeps 93 of 100 cells', 'the areas read are shown');
  await h.undo.click(); assert.match(note(h.undo), /There is nothing to undo yet/); assert.equal(h.writes.length, 3);
  h.lasso(); assert.equal(h.writes[3].args[1].version, 4, 'the next outline goes with the version read');
});

test('a 409 with the areas still at the version sent is a refusal: the history stays and the engine\'s words are shown', async t => {
  const h = panel(t); await h.ready(); await h.button.click();
  h.lasso(); h.writes[0].resolve({ version:1, areas:[area(1)], removed:5 }); await settle();
  await h.undo.click(); h.lasso(SMALL); h.lasso(LOW);   // Undo on its way, two outlines waiting
  h.writes[1].reject(Object.assign(Error('Pause the run and wait for its cells before removing any'), { status:409 })); await settle();
  h.reads.at(-1).resolve({ version:1, areas:[area(1)], removed:5 }); await settle();
  // The engine's words, then what was dropped as a sentence of its own.
  assert.deepEqual(h.notices, ['Pause the run and wait for its cells before removing any. 2 later changes were not sent.']);
  assert.deepEqual(h.history(), [1, 0, 0], 'the history stays'); assert.equal(h.writes.length, 2, 'the waiting outlines were not sent');
  await h.undo.click(); assert.deepEqual(h.writes[2].args, ['r', 1, 1], 'Undo can try again');
  // A read after a 409 that fails tells nothing either way: a refusal, as far as is known.
  h.writes[2].reject(Object.assign(Error('Try again.'), { status:409 })); await settle();
  h.reads.at(-1).reject(Error('offline')); await settle();
  assert.equal(h.notices.at(-1), 'Try again.'); assert.deepEqual(h.history(), [1, 0, 0]);
});

test('Undo and Redo pressed while a change is on its way wait their turn and act on it', async t => {
  const h = panel(t); await h.ready(); await h.button.click();
  h.lasso();                                                       // nothing to undo until this outline has gone
  await h.undo.click();
  assert.equal(note(h.undo), ''); assert.deepEqual(h.history(), [0, 0, 1]); assert.equal(h.writes.length, 1);
  h.writes[0].resolve({ version:1, areas:[area(3)], removed:5 }); await settle();
  assert.deepEqual(h.writes[1].args, ['r', 3, 1], 'Undo takes back the area the outline made');
  await h.redo.click();                                            // Redo of the Undo on its way
  assert.equal(note(h.redo), ''); assert.deepEqual(h.history(), [0, 0, 1]);
  h.writes[1].resolve({ version:2, areas:[], removed:0 }); await settle();
  assert.deepEqual(h.writes[2].args, ['r', { version:2, points:SQUARE }]);
  h.writes[2].resolve({ version:3, areas:[area(4)], removed:5 }); await settle();
  assert.deepEqual(h.history(), [1, 0, 0]);
});

test('a refused change keeps the history; a change whose outcome is unknown clears it and reads the areas again', async t => {
  const h = panel(t); await h.ready(); await h.button.click();
  h.lasso(); h.writes[0].resolve({ version:1, areas:[area(1)], removed:5 }); await settle();
  await h.undo.click();
  h.writes[1].reject(Object.assign(Error('Remove at most 200 areas'), { status:422 })); await settle();
  assert.deepEqual(h.notices, ['Remove at most 200 areas']); assert.deepEqual(h.history(), [1, 0, 0]);
  assert.equal(h.reads.length, 1, 'nothing changed: nothing to read again');
  await h.undo.click(); assert.deepEqual(h.writes[2].args, ['r', 1, 1]);
  h.writes[2].reject(new TypeError('Failed to fetch')); await settle();
  assert.equal(h.notices[1], 'No answer came for the change, so it may or may not have gone through. The latest are ' +
    'shown and the undo history was cleared; click an area and press Delete to bring its cells back.');
  assert.deepEqual(h.history(), [0, 0, 0]);
  assert.equal(h.reads.length, 2); assert.equal(h.exclusions.data, null);
  h.writes.length = 0; h.reads[1].resolve({ version:1, areas:[area(1)], removed:5 }); await settle();
  await h.undo.click(); assert.match(note(h.undo), /There is nothing to undo yet/);
  // A server error is no answer either.
  h.lasso(SMALL); h.writes[0].reject(Object.assign(Error('Internal Server Error'), { status:500 })); await settle();
  assert.equal(h.notices[2], EXCLUSION_TEXT.failed);
});

test('while a change that went wrong is read again, a press says the last step is still running', async t => {
  const brought = deferred(), h = panel(t, { refresh:() => brought.promise });
  await h.ready(); await h.button.click();
  h.lasso(); h.writes[0].reject(new TypeError('Failed to fetch')); await settle();
  // The run detail is on its way (busy), then the areas are read (loading): neither is "could not be loaded".
  for (const button of [h.undo, h.redo]) { await button.click(); assert.equal(note(button), `Check these first:${EXPLAIN_TEXT.busy}`, button.id); }
  brought.resolve(); await settle();
  assert.equal(h.exclusions.loading, true);
  await h.undo.click(); assert.ok(note(h.undo).includes(EXPLAIN_TEXT.busy));
  h.reads.at(-1).resolve({ version:1, areas:[area(1)], removed:5 }); await settle();
  await h.undo.click(); assert.match(note(h.undo), /There is nothing to undo yet/);
});

test('the run starting again with changes waiting says how many were not sent', async t => {
  const h = panel(t); await h.ready(); await h.button.click();
  h.lasso(); h.lasso(SMALL); h.lasso(LOW);      // one on its way, two waiting
  h.click(10, 150);                                                     // and an outline being drawn
  h.exclusions.update({ ...run, state:'running' });
  assert.deepEqual(h.notices, ['The run started again, so 2 changes were not sent.']);
  assert.equal(h.exclusions.queued, 0); assert.equal(h.exclusions.drawing, false);
  assert.equal(h.exclusions.snapshot().outline, 0, 'the outline being drawn goes too');
  h.writes[0].resolve({ version:1, areas:[area(1)], removed:5 }); await settle();
  // Paused again: one waiting is said as one.
  h.exclusions.update({ ...run, exclusions:{ version:1 } }); h.reads.at(-1).resolve({ version:1, areas:[area(1)], removed:5 });
  await settle(); await h.button.click();
  h.lasso(SMALL); h.lasso(LOW);
  h.exclusions.update({ ...run, state:'running', exclusions:{ version:1 } });
  assert.equal(h.notices.at(-1), 'The run started again, so 1 change was not sent.');
});

test('a failed read while outlines wait drops them and draws the image again, so their outlines go', async t => {
  const h = panel(t); await h.ready(); await h.button.click();
  h.exclusions.update({ ...run, exclusions:{ version:2 } });          // a read starts
  h.lasso();                                                         // and the outline waits for it
  assert.equal(h.exclusions.queued, 1);
  const before = h.draws.length;
  h.reads[1].reject(Error('offline')); await settle();
  assert.equal(h.exclusions.queued, 0); assert.ok(h.draws.length > before, 'drawn again without the waiting outline');
  const g = recorder(); h.exclusions.draw(g, polygon);
  assert.equal(g.calls.filter(([n]) => n === 'polygon').length, 0);
  assert.deepEqual(h.notices, [EXCLUSION_TEXT.error]);
});

test('Ctrl+Z, Ctrl+Shift+Z and Ctrl+Y act while the mode is on or the focus is in the panel, never in a text field', async t => {
  const h = panel(t); await h.ready(); await h.button.click();
  h.lasso(); h.writes[0].resolve({ version:1, areas:[area(1)], removed:5 }); await settle();
  for (const id of ['name', 'notes', 'stain-exclude-size-min']) {
    assert.equal(h.key('z', { ctrlKey:true, target:h.$(id) }).defaultPrevented, false, id);
    h.key('Escape', { target:h.$(id) });
  }
  assert.equal(h.writes.length, 1); assert.equal(h.exclusions.drawing, true, 'Escape in a text field is the field\'s');
  h.key('z', { ctrlKey:true, altKey:true }); assert.equal(h.writes.length, 1);
  h.key('z', { ctrlKey:true, target:h.$('tick') });                // a checkbox is not typed into
  assert.deepEqual(h.writes[1].args, ['r', 1, 1]);
  h.writes[1].resolve({ version:2, areas:[], removed:0 }); await settle();
  // The mode off: the keys are the page's, unless the focus is in the panel (on Undo, say).
  h.key('Escape'); assert.equal(h.exclusions.drawing, false);
  assert.equal(h.key('z', { ctrlKey:true, shiftKey:true }).defaultPrevented, false); assert.equal(h.writes.length, 2);
  assert.equal(h.key('z', { ctrlKey:true, shiftKey:true, target:h.undo }).defaultPrevented, true);
  assert.deepEqual(h.writes[2].args, ['r', { version:2, points:SQUARE }]);
  h.writes[2].resolve({ version:3, areas:[area(2)], removed:5 }); await settle();
  // The run runs again: the panel is hidden and the keys are the page's.
  h.exclusions.update({ ...run, state:'running', exclusions:{version:3} });
  assert.equal(h.key('z', { ctrlKey:true, target:h.undo }).defaultPrevented, false); assert.equal(h.writes.length, 3);
});

// Areas are acted on where they are: a click on one selects it, Delete brings its cells back.
const TWO = { version:1, areas:[area(1, 12), area(2, 4, SMALL)], removed:16 };
const drawn = h => { const g = recorder(); h.exclusions.draw(g, polygon); return g.calls; };
const words = calls => calls.filter(([n]) => n === 'fillText').map(([, text]) => text);

test('in the mode a click inside an area selects it instead of placing a corner; it is drawn on top with its label', async t => {
  const h = panel(t); await h.ready(TWO); await h.button.click();
  const draws = h.draws.length;
  h.fire('pointermove', 40, 40); h.fire('pointerdown', 40, 40);
  assert.deepEqual(drawn(h).filter(([n]) => n === 'fillRect' || n === 'moveTo'), [], 'a press that may be a click shows no corner');
  const up = h.fire('pointerup', 40, 40);
  assert.equal(up.stopped, true, 'the press is the panel\'s, never a patch box\'s or the page\'s');
  assert.equal(h.exclusions.snapshot().selected, 1); assert.equal(h.exclusions.snapshot().outline, 0, 'no corner was placed');
  assert.equal(h.writes.length, 0); assert.ok(h.draws.length > draws, 'drawn again');
  const calls = drawn(h);
  // The other area as ever, then the selected one on top, more strongly, in the same colour; its label at the right of
  // its top: its cells and what Delete does.
  assert.deepEqual(calls.filter(([n]) => n === 'polygon').map(([, points]) => points), [SMALL, SQUARE]);
  const strokes = calls.filter(([n]) => n === 'strokeStyle').map(([, v]) => v);
  assert.deepEqual(strokes.slice(0, 2), ['rgba(255,180,140,.5)', '#ffb48c']);
  assert.deepEqual(calls.filter(([n]) => n === 'lineWidth').map(([, v]) => v).slice(0, 2), [1, 2]);
  const label = '12 cells inside · Delete removes this area', w = label.length * 6 + 12;
  assert.deepEqual(calls.filter(([n]) => n === 'fillRect'), [['fillRect', 88, 10, w, 20]]);
  assert.deepEqual(calls.filter(([n]) => n === 'fillText'), [['fillText', label, 94, 20.5]]);
  for (const text of [label, EXCLUSION_TEXT.selectedOne, EXCLUSION_TEXT.selectedNew]) assert.deepEqual(bannedWords(text), []);
  // A click outside every area lets it go and, in the mode, places the first corner as ever.
  h.click(180, 20);
  assert.equal(h.exclusions.snapshot().selected, null); assert.equal(h.exclusions.snapshot().outline, 1);
  assert.deepEqual(words(drawn(h)), []);
});

test('the label counts cells as the Keeps line does, and goes to the area\'s left where the image ends at its right', async t => {
  const right = [[300,20],[380,20],[380,60],[300,60]], h = panel(t);
  await h.ready({ version:1, areas:[area(1, 1), area(2, 40070, SMALL), area(3, 2, right)], removed:40073 });
  h.click(40, 40); assert.deepEqual(words(drawn(h)), ['1 cell inside · Delete removes this area']);
  h.click(120, 120); assert.deepEqual(words(drawn(h)), ['40,070 cells inside · Delete removes this area']);
  Object.assign(h.exclusions.stage, { cssWidth:400, cssHeight:300 });   // the image's size on screen
  h.click(340, 40);
  const text = '2 cells inside · Delete removes this area', w = text.length * 6 + 12;
  assert.deepEqual(drawn(h).filter(([n]) => n === 'fillRect'), [['fillRect', 300 - 8 - w, 20, w, 20]]);
  // An area as wide as the image: the label stays on it.
  h.exclusions.stage.cssWidth = 90; h.click(40, 40);
  assert.deepEqual(drawn(h).filter(([n]) => n === 'fillRect').map(([, x]) => x), [4]);
});

test('where areas overlap a click selects the one on top, the selected one included; a click again on the spot the one under it', async t => {
  const h = panel(t), over = [[50,50],[120,50],[120,120],[50,120]];
  await h.ready({ version:1, areas:[area(1, 12), area(2, 4, over), area(3, 2, LOW)], removed:18 });
  h.click(60, 60); assert.equal(h.exclusions.snapshot().selected, 2, 'inside 1 and 2: the later');
  h.click(20, 20); assert.equal(h.exclusions.snapshot().selected, 1);
  h.click(100, 100); assert.equal(h.exclusions.snapshot().selected, 2);
  h.click(40, 140); assert.equal(h.exclusions.snapshot().selected, 3);
  h.click(180, 180); assert.equal(h.exclusions.snapshot().selected, null, 'outside every area');
  // The selected area is drawn over the others, so a click inside it keeps it, also where another lies under it; a
  // click again on that spot (within the outline's close radius) steps to the one under it, and round again.
  h.click(20, 20); h.click(70, 70); assert.equal(h.exclusions.snapshot().selected, 1, 'drawn on top: kept');
  h.click(74, 66); assert.equal(h.exclusions.snapshot().selected, 2, 'again on the spot: the one under it');
  h.click(70, 70); assert.equal(h.exclusions.snapshot().selected, 1, 'and round again');
  // An area inside a later one is reached the same way; where only one lies, a click again keeps it.
  h.exclusions.apply({ version:2, areas:[area(4, 2, [[60,60],[70,60],[70,70],[60,70]]), area(5, 9, over)], removed:9 });
  assert.equal(h.exclusions.snapshot().selected, null, 'gone with its area');
  h.click(65, 65); assert.equal(h.exclusions.snapshot().selected, 5, 'the later lies on top');
  h.click(65, 65); assert.equal(h.exclusions.snapshot().selected, 4, 'the one inside it');
  h.click(100, 100); h.click(100, 100); assert.equal(h.exclusions.snapshot().selected, 5);
});

test('Delete or Backspace brings the selected area\'s cells back as one Undo step, and the key goes no further', async t => {
  const h = panel(t); await h.ready(TWO);
  h.click(120, 120); assert.equal(h.exclusions.snapshot().selected, 2);
  const del = h.key('Delete');
  assert.equal(del.defaultPrevented, true); assert.equal(del.stopped, true, 'no other handler on the page acts on it');
  assert.equal(h.writes[0].op, 'delete'); assert.deepEqual(h.writes[0].args, ['r', 2, 1]);
  assert.equal(h.exclusions.snapshot().selected, null, 'the selection goes with it');
  assert.equal(h.exclusions.status.textContent, EXCLUSION_TEXT.restoring);
  h.writes[0].resolve({ version:2, areas:[area(1, 12)], removed:12 }); await settle();
  assert.deepEqual(h.history(), [1, 0, 0]); assert.equal(h.exclusions.total.textContent, 'Keeps 88 of 100 cells');
  // Ctrl+Z puts the area back (posted again); Ctrl+Shift+Z takes it away again by the id it has now.
  h.key('z', { ctrlKey:true, target:h.undo });
  assert.deepEqual(h.writes[1].args, ['r', { version:2, points:SMALL }]);
  h.writes[1].resolve({ version:3, areas:[area(1, 12), area(5, 4, SMALL)], removed:16 }); await settle();
  assert.deepEqual(h.history(), [0, 1, 0]);
  h.key('z', { ctrlKey:true, shiftKey:true, target:h.undo });
  assert.deepEqual(h.writes[2].args, ['r', 5, 3]);
  h.writes[2].resolve({ version:4, areas:[area(1, 12)], removed:12 }); await settle();
  // Backspace does the same; with nothing selected neither key is taken.
  h.click(40, 40); const back = h.key('Backspace');
  assert.equal(back.defaultPrevented, true); assert.equal(back.stopped, true); assert.deepEqual(h.writes[3].args, ['r', 1, 4]);
  for (const name of ['Delete', 'Backspace']) assert.equal(h.key(name).defaultPrevented, false, name);
  assert.equal(h.writes.length, 4);
});

test('Delete in a text field is the field\'s, and with a modifier held the page\'s, even with an area selected', async t => {
  const h = panel(t); await h.ready(TWO);
  h.click(40, 40);
  for (const id of ['name', 'notes']) assert.equal(h.key('Delete', { target:h.$(id) }).defaultPrevented, false, id);
  assert.equal(h.key('Backspace', { target:h.$('name') }).defaultPrevented, false);
  assert.equal(h.key('Delete', { ctrlKey:true }).defaultPrevented, false);
  assert.equal(h.writes.length, 0); assert.equal(h.exclusions.snapshot().selected, 1);
});

test('while an outline is drawn a click inside an area adds a corner, and Backspace takes one back', async t => {
  const h = panel(t); await h.ready(TWO); await h.button.click();
  h.click(180, 20); h.click(40, 40);                                // a corner outside, then one inside area 1
  assert.equal(h.exclusions.snapshot().outline, 2); assert.equal(h.exclusions.snapshot().selected, null);
  const back = h.key('Backspace');
  assert.equal(back.defaultPrevented, true); assert.equal(h.exclusions.snapshot().outline, 1); assert.equal(h.writes.length, 0);
  // Clicked on before the mode, an area loses its selection to the first press that draws.
  h.key('Escape'); h.key('Escape'); assert.equal(h.exclusions.drawing, false);
  h.click(40, 40); assert.equal(h.exclusions.snapshot().selected, 1);
  await h.button.click(); h.click(180, 20);
  assert.equal(h.exclusions.snapshot().selected, null); assert.equal(h.exclusions.snapshot().outline, 1);
  assert.equal(h.key('Delete').defaultPrevented, false, 'nothing selected: Delete is not taken');
  // With an area selected, Backspace during a press that begins an outline takes back what the press placed, as ever.
  h.key('Escape'); h.click(40, 40); assert.equal(h.exclusions.snapshot().selected, 1);
  h.fire('pointerdown', 180, 20); assert.equal(h.exclusions.snapshot().outline, 1);
  assert.equal(h.key('Backspace').defaultPrevented, true);
  assert.equal(h.exclusions.snapshot().outline, 0); assert.equal(h.writes.length, 0, 'the area stays');
  h.fire('pointerup', 180, 20); assert.equal(h.exclusions.snapshot().selected, null);
});

test('in the mode a drag from inside an area draws an outline, as anywhere', async t => {
  const h = panel(t); await h.ready(TWO); await h.button.click();
  h.click(40, 40); assert.equal(h.exclusions.snapshot().selected, 1);
  h.fire('pointerdown', 40, 40); h.fire('pointermove', 60, 40);
  assert.ok(drawn(h).some(([n]) => n === 'moveTo'), 'once it moves, the stroke is drawn');
  h.fire('pointermove', 180, 40); h.fire('pointermove', 180, 180); h.fire('pointerup', 40, 180);
  assert.equal(h.writes[0].op, 'post'); assert.deepEqual(h.writes[0].args[1].points[0], [40, 40]);
  assert.equal(h.exclusions.snapshot().selected, null, 'a press that draws lets the selection go');
});

test('Escape lets the selection go first; then it drops the outline or leaves the mode, and out of it is the page\'s', async t => {
  const h = panel(t); await h.ready(TWO); await h.button.click();
  h.click(40, 40); assert.equal(h.exclusions.snapshot().selected, 1);
  const first = h.key('Escape');
  assert.equal(first.defaultPrevented, true); assert.equal(first.stopped, true);
  assert.equal(h.exclusions.snapshot().selected, null); assert.equal(h.exclusions.drawing, true, 'the mode stays on');
  const second = h.key('Escape');
  assert.equal(second.defaultPrevented, true); assert.equal(second.stopped, true); assert.equal(h.exclusions.drawing, false);
  // Out of the mode the first Escape is the selection's (the page's, which lets its cell go, waits), the next the page's.
  h.click(40, 40); assert.equal(h.exclusions.snapshot().selected, 1);
  assert.equal(h.key('Escape').defaultPrevented, true); assert.equal(h.exclusions.snapshot().selected, null);
  const page = h.key('Escape'); assert.equal(page.defaultPrevented, false); assert.equal(page.stopped, undefined);
});

test('out of the mode a drag (the page\'s pan) selects nothing, and a click on an area brings the keys to the image', async t => {
  const h = panel(t); await h.ready(TWO);
  h.$('name').focus();
  h.fire('pointerdown', 40, 40); h.fire('pointermove', 46, 40, { buttons:1 }); h.fire('pointerup', 40, 40); h.fire('click', 40, 40);
  assert.equal(h.exclusions.snapshot().selected, null, 'a pan that came back is no click');
  const right = { button:2 };
  h.fire('pointerdown', 40, 40, right); h.fire('pointerup', 40, 40, right); h.fire('click', 40, 40, right);
  assert.equal(h.exclusions.snapshot().selected, null, 'nor a press of another button');
  assert.equal(h.document.activeElement, h.$('name'));
  const tap = h.click(40, 40);
  assert.equal(tap.stopped, undefined, 'out of the mode the press goes on to the page');
  assert.equal(h.exclusions.snapshot().selected, 1); assert.equal(h.document.activeElement, h.canvas);
});

test('the pointer shows the hand over an area a click would select, never while an outline is drawn', async t => {
  const h = panel(t); await h.ready(TWO);
  const hand = () => h.canvas.classList.contains('over-area');
  h.fire('pointermove', 40, 40); assert.equal(hand(), true);
  h.fire('pointermove', 180, 40); assert.equal(hand(), false);
  h.fire('pointermove', 120, 120); h.fire('pointerleave'); assert.equal(hand(), false, 'gone with the pointer');
  await h.button.click();
  h.fire('pointermove', 40, 40); assert.equal(hand(), true, 'in the mode, with no outline: a click selects');
  h.click(180, 20); h.fire('pointermove', 40, 40); assert.equal(hand(), false, 'with an outline: a click adds a corner');
  const draws = h.draws.length; h.key('Escape');
  assert.equal(hand(), true, 'the outline dropped under the pointer at rest: the hand again');
  h.fire('pointermove', 41, 40);
  assert.equal(hand(), true); assert.equal(h.draws.length, draws + 1, 'the hand alone draws nothing again');
  h.exclusions.update({ ...run, state:'running' }); h.fire('pointermove', 40, 40);
  assert.equal(hand(), false, 'nothing to select while the run runs');
  const css = readFileSync(new URL('../src/stain-exclusions.css', import.meta.url), 'utf8');
  assert.match(css, /\.stage\.over-area\{cursor:pointer\}/);
  assert.ok(css.indexOf('.stage.over-area{') > css.indexOf('.stage.removing{'), 'over the mode\'s crosshair');
});

test('the selection goes with its area (Undo, another window) and with the run', async t => {
  const h = panel(t); await h.ready(); await h.button.click();
  h.lasso(); h.writes[0].resolve({ version:1, areas:[area(1)], removed:5 }); await settle();
  h.click(40, 40); assert.equal(h.exclusions.snapshot().selected, 1);
  await h.undo.click(); h.writes[1].resolve({ version:2, areas:[], removed:0 }); await settle();
  assert.equal(h.exclusions.snapshot().selected, null, 'Undo took the area back');
  await h.redo.click(); h.writes[2].resolve({ version:3, areas:[area(2)], removed:5 }); await settle();
  h.click(40, 40); assert.equal(h.exclusions.snapshot().selected, 2);
  // Another window: the area stays while it is there, and goes with it.
  h.exclusions.update({ ...run, exclusions:{ version:4 } }); h.reads.at(-1).resolve({ version:4, areas:[area(2), area(3, 4, SMALL)], removed:9 });
  await settle(); assert.equal(h.exclusions.snapshot().selected, 2);
  h.exclusions.update({ ...run, exclusions:{ version:5 } }); h.reads.at(-1).resolve({ version:5, areas:[area(3, 4, SMALL)], removed:4 });
  await settle(); assert.equal(h.exclusions.snapshot().selected, null); assert.deepEqual(words(drawn(h)), []);
  // The run runs again, or another run opens: nothing stays selected.
  h.click(120, 120); assert.equal(h.exclusions.snapshot().selected, 3);
  h.exclusions.update({ ...run, state:'running', exclusions:{ version:5 } });
  assert.equal(h.exclusions.snapshot().selected, null); assert.equal(h.key('Delete').defaultPrevented, false);
  // Nor does a click whose press began before the run started again.
  h.exclusions.update({ ...run, exclusions:{ version:5 } });
  h.fire('pointerdown', 120, 120); h.exclusions.update({ ...run, state:'running', exclusions:{ version:5 } });
  h.fire('pointerup', 120, 120); h.fire('click', 120, 120); assert.equal(h.exclusions.snapshot().selected, null);
});

test('the panel lists no areas: each is acted on where it is, on the image', async t => {
  const h = panel(t); await h.ready(TWO);
  const panelNode = h.$('stain-exclusions');
  assert.equal(panelNode.textContent.includes('Area'), false);
  assert.deepEqual(panelNode.all().filter(n => n.tagName === 'BUTTON').map(b => b.id), ['stain-exclude', 'stain-exclude-undo', 'stain-exclude-redo']);
  const html = readFileSync(new URL('../index.html', import.meta.url), 'utf8');
  assert.doesNotMatch(html, /stain-exclude-list/);
  assert.doesNotMatch(readFileSync(new URL('../src/stain-patches.css', import.meta.url), 'utf8'), /stain-exclude-list/, 'nor its style');
});

test('the area just drawn can be clicked and deleted while its cells are still being removed', async t => {
  const h = panel(t); await h.ready(); await h.button.click();
  h.lasso(); assert.equal(h.exclusions.busy, true);
  const points = h.writes[0].args[1].points;
  // A click inside the outline on its way selects it (no corner), drawn selected with what Delete does.
  h.click(40, 40);
  assert.equal(h.exclusions.snapshot().outline, 0, 'no corner was placed');
  assert.deepEqual(h.exclusions.selectedArea()?.points, points);
  assert.equal(h.exclusions.snapshot().selected, null, 'no id until the engine names it');
  const calls = drawn(h);
  assert.deepEqual(words(calls), [EXCLUSION_TEXT.selectedNew]);
  assert.deepEqual(calls.filter(([n]) => n === 'fill').length, 1, 'filled as the area clicked on');
  // Delete waits for the outline, then brings the cells back by the id the engine gave it: two Undo steps.
  const del = h.key('Delete');
  assert.equal(del.defaultPrevented, true); assert.equal(h.exclusions.queued, 1); assert.equal(h.exclusions.selectedArea(), null);
  h.writes[0].resolve({ version:1, areas:[area(7, 5, points)], removed:5 }); await settle();
  assert.equal(h.writes[1].op, 'delete'); assert.deepEqual(h.writes[1].args, ['r', 7, 1]);
  h.writes[1].resolve({ version:2, areas:[], removed:0 }); await settle();
  assert.deepEqual(h.history(), [2, 0, 0]); assert.equal(h.exclusions.snapshot().outline, 0);
  await h.undo.click(); assert.deepEqual(h.writes[2].args, ['r', { version:2, points }], 'Undo puts the area back');
});

test('out of the mode an outline on its way is clicked on as an area is, over the areas, and Delete brings back its own cells', async t => {
  const under = [[90,90],[130,90],[130,130],[90,130]];
  const h = panel(t); await h.ready({ version:0, areas:[area(9, 3, under)], removed:3 }); await h.button.click();
  h.lasso(); h.lasso(SMALL); await h.button.click();                 // one on its way, one waiting; the mode off
  assert.equal(h.exclusions.drawing, false); assert.equal(h.exclusions.queued, 1);
  h.click(120, 120); assert.deepEqual(h.exclusions.selectedArea()?.points, SMALL, 'drawn over the area under it');
  h.click(40, 40); assert.deepEqual(h.exclusions.selectedArea()?.points, SQUARE);
  assert.equal(h.key('Escape').defaultPrevented, true); assert.equal(h.exclusions.selectedArea(), null);
  // Delete on the one waiting, while the first is still on its way: it goes after it, as its own change.
  h.click(120, 120); assert.equal(h.key('Delete').defaultPrevented, true); assert.equal(h.exclusions.queued, 2);
  // Selected again, it stays selected while the first is named.
  h.click(120, 120); h.writes[0].resolve({ version:1, areas:[area(9, 3, under), area(3)], removed:8 }); await settle();
  assert.deepEqual(h.exclusions.selectedArea()?.points, SMALL);
  h.click(40, 40); assert.equal(h.exclusions.snapshot().selected, 3);
  h.writes[1].resolve({ version:2, areas:[area(9, 3, under), area(3), area(4, 2, SMALL)], removed:10 }); await settle();
  assert.deepEqual(h.writes[2].args, ['r', 4, 2], 'its cells come back by the id it was given');
});

test('an outline clicked on its way is, once named, that area: gone with Undo, not back with Redo; refused, nothing is selected', async t => {
  const h = panel(t); await h.ready(); await h.button.click();
  h.lasso(); h.click(40, 40);
  h.writes[0].resolve({ version:1, areas:[area(3)], removed:5 }); await settle();
  assert.equal(h.exclusions.snapshot().selected, 3);
  assert.deepEqual(words(drawn(h)), ['5 cells inside · Delete removes this area']);
  await h.undo.click(); h.writes[1].resolve({ version:2, areas:[], removed:0 }); await settle();
  assert.equal(h.exclusions.snapshot().selected, null);
  await h.redo.click(); h.writes[2].resolve({ version:3, areas:[area(4)], removed:5 }); await settle();
  assert.equal(h.exclusions.snapshot().selected, null, 'posted again it is another area');
  h.lasso(SMALL); h.click(120, 120); assert.deepEqual(h.exclusions.selectedArea()?.points, SMALL);
  h.writes[3].reject(Object.assign(Error('Remove at most 200 areas'), { status:422 })); await settle();
  assert.equal(h.exclusions.selectedArea(), null); assert.deepEqual(words(drawn(h)), []);
  assert.equal(h.key('Delete').defaultPrevented, false);
});

test('in the mode a short drag from inside an area is a stretch of an outline; with one drawn, a press inside an area shows on it', async t => {
  const h = panel(t); await h.ready(TWO); await h.button.click();
  const hand = () => h.canvas.classList.contains('over-area');
  h.fire('pointermove', 40, 40); assert.equal(hand(), true);
  h.fire('pointerdown', 40, 40); h.fire('pointermove', 45, 40, { buttons:1 }); h.fire('pointerup', 45, 40);
  assert.equal(h.exclusions.snapshot().selected, null); assert.equal(h.exclusions.snapshot().outline, 2, 'past 3 px: a stretch');
  assert.equal(hand(), false, 'an outline is being drawn: a click adds a corner');
  // With an outline being drawn, a press inside an area is a corner, drawn while the button is still down.
  h.fire('pointerdown', 120, 120);
  const calls = drawn(h);
  assert.ok(calls.some(([n]) => n === 'lineTo'), 'the outline on to it'); assert.ok(calls.some(([n]) => n === 'fillRect'), 'its first point');
  h.fire('pointerup', 120, 120);
  assert.equal(h.exclusions.snapshot().outline, 3); assert.equal(h.exclusions.snapshot().selected, null);
});

test('the snapshot names the area selected as shown: none while the areas are read again after a change went wrong', async t => {
  const h = panel(t); await h.ready(); await h.button.click();
  h.lasso(); h.writes[0].resolve({ version:1, areas:[area(1)], removed:5 }); await settle();
  await h.button.click(); h.click(40, 40); assert.equal(h.exclusions.snapshot().selected, 1);
  await h.undo.click(); h.writes[1].reject(new TypeError('Failed to fetch')); await settle();
  assert.equal(h.exclusions.data, null); assert.equal(h.exclusions.snapshot().selected, null, 'no areas shown, none selected');
  assert.equal(h.key('Delete').defaultPrevented, false);
  h.reads.at(-1).resolve({ version:1, areas:[area(1)], removed:5 }); await settle();
  assert.equal(h.exclusions.snapshot().selected, 1, 'still there, and still the one clicked on');
});

test('the hand follows the areas under a pointer at rest: gone with the area Delete took, back with Undo, and with the view', async t => {
  const h = panel(t); await h.ready(TWO);
  const hand = () => h.canvas.classList.contains('over-area');
  h.click(40, 40); assert.equal(hand(), true);
  h.key('Delete'); h.writes[0].resolve({ version:2, areas:[area(2, 4, SMALL)], removed:4 }); await settle();
  assert.equal(hand(), false, 'the area gone from under the pointer');
  await h.undo.click(); h.writes[1].resolve({ version:3, areas:[area(2, 4, SMALL), area(3, 12)], removed:16 }); await settle();
  assert.equal(hand(), true, 'back with Undo');
  h.fire('pointermove', 85, 85); assert.equal(hand(), false);
  h.exclusions.stage.toWorld = (x, y) => [x * 1.4, y * 1.4]; h.view();
  assert.equal(hand(), true, 'a zoom brought an area under it');
  h.fire('pointerleave'); h.view(); assert.equal(hand(), false, 'none once the pointer has left');
});

test('a screen reader hears the area clicked on: its label\'s words in a status that is not shown', async t => {
  const h = panel(t); await h.ready(TWO);
  const said = h.$('stain-exclude-selected');
  assert.equal(said.getAttribute('role'), 'status'); assert.equal(said.textContent, '');
  h.click(40, 40); assert.equal(said.textContent, '12 cells inside · Delete removes this area');
  h.key('Escape'); assert.equal(said.textContent, '');
  h.click(120, 120); h.key('Delete'); assert.equal(said.textContent, '', 'gone with Delete');
  h.click(40, 40); h.exclusions.apply({ version:2, areas:[area(2, 4, SMALL)], removed:4 });
  assert.equal(said.textContent, '', 'and with its area, gone in another window');
  const css = readFileSync(new URL('../src/stain-exclusions.css', import.meta.url), 'utf8');
  assert.match(css, /#stain-exclude-selected\{position:absolute;width:1px;height:1px;overflow:hidden;clip-path:inset\(50%\)/);
});

const boxes = { version:1, size:128, count:1, set:null, run_pixel_to_sample:[0,0,1,1], patches:[{ id:1, box:[10,10,138,138] }], exported:null };
// The whole page's markup with the areas' panel and the patch panel on one image, made as main.js makes them: the areas'
// panel first, so its capture listeners on the image come first. Both are loaded (no areas, one patch box).
async function both(t) {
  const previous = globalThis.document, previousWindow = globalThis.window, { document } = domFixture();
  globalThis.document = document; globalThis.window = { addEventListener() {} };
  document.body.innerHTML = readFileSync(new URL('../index.html', import.meta.url), 'utf8');
  document.createElementNS = (_, tag) => document.createElement(tag);
  const canvas = document.getElementById('stage');
  canvas.setPointerCapture = () => {}; canvas.hasPointerCapture = () => true; canvas.releasePointerCapture = () => {};
  canvas.width = 800; canvas.height = 400;
  const stage = { canvas, camera:{ x:400, y:200, zoom:1 }, scale:1, toWorld:(x, y) => [x, y], toScreen:(x, y) => [x, y], onChange() {} };
  const writes = [], reads = [];
  const read = () => { const d = deferred(); reads.push(d); return d.promise; };
  const write = op => (...args) => { const d = deferred(); writes.push({ op, args, ...d }); return d.promise; };
  const exclusions = new StainExclusions(stage, { stainExclusions:read, stainExclude:write('lasso'), stainRestore:write('restore'),
    stainExcludeSize:write('size'), stainBrightness:() => deferred().promise, stainExcludeBrightness:write('bright') },
    { changed() {}, refresh:async () => {}, notice() {} });
  const patches = new StainPatches(stage, { stainPatches:read, stainPatchBoxes:write('boxes') }, { notice() {} });
  t.after(() => { exclusions.reset(); patches.reset(); globalThis.document = previous; globalThis.window = previousWindow; });
  exclusions.update(run); patches.update(run);
  reads[0].resolve({ version:0, areas:[], removed:0 }); reads[1].resolve(boxes); await settle();
  return { document, canvas, exclusions, patches, writes };
}

test('in the mode a press on the image goes to the outline, never to a patch box; outside it the boxes work as before', async t => {
  const { canvas, exclusions, patches, writes } = await both(t);
  // An event stops where a listener stops it, as in a browser (the fixture's emit calls every listener).
  const fire = (type, x, y, extra) => {
    let stopped = false;
    const e = { button:0, pointerId:1, offsetX:x, offsetY:y, target:canvas, preventDefault() {}, stopImmediatePropagation() { stopped = true; }, ...extra };
    for (const fn of canvas.listeners.get(type) || []) { fn(e); if (stopped) break; }
  };
  const drag = extra => { fire('pointerdown', 50, 50, extra); fire('pointermove', 90, 50, extra); fire('pointermove', 90, 90, extra); fire('pointerup', 50, 90, extra); };
  drag();
  assert.deepEqual(writes.map(w => w.op), ['boxes'], 'outside the mode a press inside a box moves it, as before');
  writes[0].resolve({ ...boxes, version:2, patches:[{ id:1, box:[10,50,138,178] }] }); await settle();
  await exclusions.button.click();
  drag();
  assert.deepEqual(writes.map(w => w.op), ['boxes', 'lasso'], 'in the mode the same press draws an outline');
  drag({ shiftKey:true });                                          // also with Shift held, while a change is on its way
  assert.deepEqual(writes.map(w => w.op), ['boxes', 'lasso']); assert.equal(exclusions.queued, 1);
  // A click on a box is a corner of the outline, never the box's.
  fire('pointerdown', 60, 100); fire('pointerup', 60, 100);
  assert.equal(exclusions.snapshot().outline, 1); assert.equal(patches.drag, null);
});

test('out of the mode a press a patch box takes selects no area under it and lets the selected one go; a click beside the box selects', async t => {
  const { document, canvas, exclusions, patches, writes } = await both(t);
  exclusions.apply({ version:1, areas:[{ id:1, points:[[100,100],[200,100],[200,200],[100,200]], cells:3 }], removed:3 });
  // As in a browser: an event stops where a listener stops it, and preventDefault marks it.
  const fire = (type, x, y) => {
    let stopped = false;
    const e = { button:0, pointerId:1, offsetX:x, offsetY:y, target:canvas, defaultPrevented:false,
      preventDefault() { this.defaultPrevented = true; }, stopImmediatePropagation() { stopped = true; } };
    for (const fn of canvas.listeners.get(type) || []) { fn(e); if (stopped) break; }
    return e;
  };
  const click = (x, y) => { fire('pointerdown', x, y); const held = patches.drag; fire('pointerup', x, y); fire('click', x, y); return held; };
  assert.ok(click(120, 120), 'inside the box [10, 10, 138, 138] (and the area) the press is the box\'s');
  assert.equal(exclusions.snapshot().selected, null, 'so it selects no area');
  assert.equal(click(180, 180), null);
  assert.equal(exclusions.snapshot().selected, 1, 'beside the box a click selects the area');
  // A press on the box (on the area or beside it) is the box's: the area lets go, so Delete is never the area's then.
  for (const [x, y] of [[120, 120], [40, 40]]) {
    click(180, 180); assert.equal(exclusions.snapshot().selected, 1);
    assert.ok(click(x, y)); assert.equal(exclusions.snapshot().selected, null, `${x}, ${y}`);
  }
  const del = { key:'Delete', target:document.body, defaultPrevented:false, preventDefault() { this.defaultPrevented = true; },
    stopImmediatePropagation() {} };
  for (const fn of document.listeners.get('keydown') || []) fn(del);
  assert.equal(del.defaultPrevented, false); assert.deepEqual(writes, []);
});

test('main.js makes the areas panel before the patch panel, so its listeners on the image come first', () => {
  const source = readFileSync(new URL('../src/main.js', import.meta.url), 'utf8');
  const areas = source.indexOf('new StainExclusions('), boxes = source.indexOf('new StainPatches(');
  assert.ok(areas > 0 && boxes > areas);
});

test('one tool on the image at a time (main.js): the remove mode ends Add a box, and Add a box ends the mode', async t => {
  const { document, exclusions, patches } = await both(t);
  const main = readFileSync(new URL('../src/main.js', import.meta.url), 'utf8');
  const block = main.slice(main.indexOf('// One tool on the image at a time'), main.indexOf('const useCellposeSettings'));
  vm.runInNewContext(block, { $:id => document.getElementById(id), stainExclusions:exclusions, stainPatches:patches });
  // A click runs the button's own handler, then bubbles to the panel (the fixture's emit: onclick, then listeners).
  const click = (button, panel) => { button.emit('click'); if (panel) document.getElementById(panel).emit('click', { target:button }); };
  click(patches.add, 'stain-patches');
  assert.equal(patches.adding, true);
  click(exclusions.button);
  assert.equal(exclusions.drawing, true); assert.equal(patches.adding, false);
  assert.equal(patches.add.getAttribute('aria-pressed'), 'false');
  click(patches.add, 'stain-patches');
  assert.equal(patches.adding, true); assert.equal(exclusions.drawing, false);
  assert.equal(exclusions.button.classList.contains('on'), false);
  // Only Add a box ends the mode: the patch panel's other presses leave it on.
  click(patches.add, 'stain-patches'); click(exclusions.button);
  assert.equal(exclusions.drawing, true);
  click(patches.propose, 'stain-patches');
  assert.equal(exclusions.drawing, true); assert.equal(patches.adding, false);
});

test('the range and area words are plain, and the mounted controls have no Apply or Clear',async t=>{
  const h=panel(t);await h.ready();
  for(const words of Object.values(EXCLUSION_TEXT))assert.deepEqual(bannedWords(words),[],words);
  assert.equal(h.$('stain-exclude-size-apply'),null);assert.equal(h.$('stain-exclude-bright-clear'),null);
  assert.ok(h.exclusions.sizeChooser.root.textContent.includes(EXCLUSION_TEXT.sizeLabel));
  assert.equal(h.exclusions.sizeChooser.numbers.min.getAttribute('aria-label'),'Smallest cell size');
  assert.equal(h.exclusions.brightChooser.numbers.max.getAttribute('aria-label'),'Highest brightness');
});

const ranged = (version=0,size=null,brightness={}) => ({version,areas_version:0,areas:[],cells:5,kept:5,removed:0,size,brightness});
function typeRange(h,kind,end,value) {
  const chooser=kind==='size'?h.exclusions.sizeChooser:h.exclusions.brightChooser;
  const box=chooser.numbers[end];box.value=value===null?'':String(value);box.emit('blur');
}
function valuesForPanel() {
  const values=new CellValues();values.id='r';values.set='s';values.areasVersion=0;
  values.data={count:5,area:new Float32Array([5,10,20,40,80]),flags:new Uint8Array([0,0,0,1,0])};
  values.brightness.set('DAPI',new Float32Array([1,10,NaN,30,100]));
  values.brightness.set('Cy5',new Float32Array([50,10,0,20,0]));return values;
}
test('a held size handle updates Keeps and paint classes without a POST; release is one Undo step',async t=>{
  const h=panel(t);await h.ready(ranged());h.exclusions.values=valuesForPanel();
  const paint=[];h.exclusions.onClasses=value=>paint.push(value);h.exclusions.setVectors({set:'s',areas_version:0});
  const c=h.exclusions.sizeChooser;c.setValues(h.exclusions.values.data.area);const slider=c.sliders.max;
  slider.emit('pointerdown');slider.value=String(c.axis.position(20)*1000);slider.emit('input');
  assert.equal(h.exclusions.total.textContent,'Keeps 3 of 5 cells');assert.deepEqual([...paint.at(-1).array],[0,0,0,1,1]);
  assert.equal(h.writes.length,0);const readCount=h.reads.length;
  // A routine render while held must not be mistaken for an external edit.
  h.exclusions.render();assert.equal(h.reads.length,readCount);assert.equal(c.pending,undefined);
  slider.emit('change');slider.emit('pointerup');assert.equal(h.writes.length,1);
  assert.deepEqual(h.writes[0].args,['r',{version:0,min:null,max:20}]);
  h.writes[0].resolve(ranged(1,{min:null,max:20}));await settle();assert.deepEqual(h.history(),[1,0,0]);
  assert.equal(h.exclusions.total.textContent,'Keeps 3 of 5 cells','committed range remains hidden');
});
test('size Undo and Redo post both ends and request handle motion',async t=>{
  const h=panel(t);await h.ready(ranged());typeRange(h,'size','min',10);
  h.writes[0].resolve(ranged(1,{min:10,max:null}));await settle();
  const motions=[],original=h.exclusions.sizeChooser.setRange.bind(h.exclusions.sizeChooser);
  h.exclusions.sizeChooser.setRange=(value,options)=>{motions.push({value,options});return original(value,options);};
  await h.undo.click();assert.deepEqual(h.writes[1].args,['r',{version:1,min:null,max:null}]);
  h.writes[1].resolve(ranged(2));await settle();assert.deepEqual(h.history(),[0,1,0]);
  await h.redo.click();assert.deepEqual(h.writes[2].args,['r',{version:2,min:10,max:null}]);
  h.writes[2].resolve(ranged(3,{min:10,max:null}));await settle();assert.deepEqual(h.history(),[1,0,0]);
  assert.ok(motions.some(m=>m.value===null && m.options.animate));
});
test('Undo and Redo bring outlines back or take them as the handles pass them; rules loaded or another channel shown do not',async t=>{
  const frames=new Map();let id=0,now=0;
  const old=globalThis.requestAnimationFrame,oldCancel=globalThis.cancelAnimationFrame;
  globalThis.requestAnimationFrame=fn=>{frames.set(++id,fn);return id;};globalThis.cancelAnimationFrame=n=>frames.delete(n);
  t.mock.method(performance,'now',()=>now);
  t.after(()=>{globalThis.requestAnimationFrame=old;globalThis.cancelAnimationFrame=oldCancel;});
  const tick=time=>{now=time;for(const [n,fn] of [...frames]){frames.delete(n);fn(now);}};
  const h=panel(t),paint=[],kept=()=>h.exclusions.snapshot().kept;
  await h.ready(ranged(0,{min:null,max:10},{DAPI:{min:null,max:50}}));
  assert.equal(frames.size,1,'the handles glide to the rule read');assert.equal(h.exclusions.tweens.size,0,'a rule read changes none');
  tick(200);
  // Sizes 5, 10, 20, 40 (in an area), 80 µm² (DAPI 100: above 50) on a log axis from 0.1 to 80 µm².
  h.exclusions.onClasses=value=>paint.push(value?.kept);h.exclusions.values=valuesForPanel();
  h.exclusions.setVectors({set:'s',areas_version:0});h.exclusions.render();assert.equal(kept(),2);
  typeRange(h,'size','max',null);h.writes[0].resolve(ranged(1,null,{DAPI:{min:null,max:50}}));await settle();
  assert.equal(paint.at(-1),3,'no size rule: 20 µm² back');
  paint.length=0;now=0;await h.undo.click();assert.deepEqual(h.writes[1].args,['r',{version:1,min:null,max:10}]);
  assert.deepEqual([...new Set(paint)],[3],'Undo: the outlines first stay as they are');
  assert.equal(h.exclusions.total.textContent,'Keeps 3 of 5 cells');
  tick(100);assert.equal(paint.at(-1),3,'half way down the handle is still above 20 µm²');
  tick(170);assert.equal(paint.at(-1),2,'past 20 µm²: that cell goes');assert.equal(h.exclusions.total.textContent,'Keeps 2 of 5 cells');
  tick(200);assert.equal(paint.at(-1),2);assert.equal(frames.size,0);assert.equal(h.exclusions.tweens.size,0);
  h.writes[1].resolve(ranged(2,{min:null,max:10},{DAPI:{min:null,max:50}}));await settle();assert.equal(frames.size,0);
  paint.length=0;now=1000;await h.redo.click();assert.deepEqual([...new Set(paint)],[2]);
  tick(1050);assert.equal(paint.at(-1),2);tick(1100);assert.equal(paint.at(-1),3,'Redo: 20 µm² back as the handle passes it');
  tick(1200);assert.equal(h.exclusions.tweens.size,0);
  h.writes[2].resolve(ranged(3,null,{DAPI:{min:null,max:50}}));await settle();assert.equal(kept(),3);
  // Another channel shown: its handles move, no rule changes, and no outline comes or goes.
  h.exclusions.channels=['DAPI','Cy5'];h.exclusions.channel='DAPI';h.exclusions.render();tick(2000);
  paint.length=0;h.exclusions.channelField.value='Cy5';h.exclusions.channelField.emit('change');
  assert.ok(frames.size>0,'the handles move to Cy5\'s range');assert.equal(h.exclusions.tweens.size,0);
  tick(2100);assert.equal(kept(),3);tick(2200);assert.equal(kept(),3);assert.ok(paint.every(k=>k===3));
});
test('queued changes merge only with the last waiting range for the same channel',async t=>{
  const h=panel(t);await h.ready(ranged());typeRange(h,'size','min',5);
  typeRange(h,'size','min',10);typeRange(h,'size','min',15);assert.equal(h.exclusions.queued,1);
  h.exclusions.channel='DAPI';typeRange(h,'brightness','max',50);typeRange(h,'size','min',20);
  assert.equal(h.exclusions.queued,3);
  h.writes[0].resolve(ranged(1,{min:5,max:null}));await settle();assert.deepEqual(h.writes[1].args,['r',{version:1,min:15,max:null}]);
  h.writes[1].resolve(ranged(2,{min:15,max:null}));await settle();assert.equal(h.writes[2].op,'bright');
  h.writes[2].resolve(ranged(3,{min:15,max:null},{DAPI:{min:null,max:50}}));await settle();
  assert.deepEqual(h.writes[3].args,['r',{version:3,min:20,max:null}]);
});
test('a range typed again after queued Undo is applied after that Undo',async t=>{
  const h=panel(t);await h.ready(ranged());typeRange(h,'size','min',20);await h.undo.click();typeRange(h,'size','min',20);
  assert.equal(h.exclusions.queued,2);
  h.writes[0].resolve(ranged(1,{min:20,max:null}));await settle();assert.equal(h.writes[1].args[1].min,null);
  h.writes[1].resolve(ranged(2));await settle();assert.equal(h.writes[2].args[1].min,20);
});
test('brightness ranges persist across channel changes; missing measurements stay kept',async t=>{
  const h=panel(t);await h.ready(ranged(0,null,{DAPI:{min:5,max:50}}));h.exclusions.values=valuesForPanel();
  h.exclusions.channels=['DAPI','Cy5'];h.exclusions.channel='DAPI';h.exclusions.setVectors({set:'s',areas_version:0});h.exclusions.render();
  assert.equal(h.exclusions.total.textContent,'Keeps 2 of 5 cells','NaN stays kept');
  assert.equal(h.exclusions.channelField.children[0].textContent,'DAPI · in use');
  h.exclusions.channelField.value='Cy5';h.exclusions.channelField.emit('change');typeRange(h,'brightness','max',20);
  assert.deepEqual(h.writes[0].args,['r',{version:0,channel:'Cy5',min:null,max:20}]);
  h.writes[0].resolve(ranged(1,null,{DAPI:{min:5,max:50},Cy5:{min:null,max:20}}));await settle();
  h.exclusions.channelField.value='DAPI';h.exclusions.channelField.emit('change');
  assert.equal(h.exclusions.brightChooser.numbers.min.value,'5');assert.equal(h.exclusions.brightChooser.numbers.max.value,'50');
  await h.undo.click();assert.deepEqual(h.writes[1].args,['r',{version:1,channel:'Cy5',min:null,max:null}]);
});
test('brightness number boxes can commit while measurement is still pending',async t=>{
  const h=panel(t);await h.ready(ranged());h.exclusions.channel='DAPI';h.exclusions.render();
  typeRange(h,'brightness','max',100);assert.deepEqual(h.writes[0].args,['r',{version:0,channel:'DAPI',min:null,max:100}]);
});
test('a cancelled range reverts the paint and never posts',async t=>{
  const h=panel(t);await h.ready(ranged());h.exclusions.values=valuesForPanel();h.exclusions.setVectors({set:'s',areas_version:0});
  const c=h.exclusions.sizeChooser;c.setValues(h.exclusions.values.data.area);c.sliders.min.emit('pointerdown');
  c.sliders.min.value='900';c.sliders.min.emit('input');assert.notEqual(h.exclusions.total.textContent,'Keeps 4 of 5 cells');
  c.sliders.min.emit('pointercancel');assert.equal(h.exclusions.total.textContent,'Keeps 4 of 5 cells');assert.equal(h.writes.length,0);
});
test('a remote range waits for release, clears history and replaces the preview without posting it',async t=>{
  const h=panel(t);await h.ready(ranged());typeRange(h,'size','min',5);h.writes[0].resolve(ranged(1,{min:5,max:null}));await settle();
  const c=h.exclusions.sizeChooser;c.sliders.min.emit('pointerdown');c.sliders.min.value='500';c.sliders.min.emit('input');
  h.exclusions.apply(ranged(2,{min:20,max:50}));assert.equal(h.exclusions.data.version,1);
  c.sliders.min.emit('change');assert.equal(h.exclusions.data.version,2);assert.equal(c.numbers.min.value,'20');
  assert.deepEqual(h.history(),[0,0,0]);assert.equal(h.writes.length,1);assert.ok(h.notices.includes(EXCLUSION_TEXT.changed));
});
test('409 reloads ranges and clears history; a same-version refusal keeps Undo available',async t=>{
  const h=panel(t);await h.ready(ranged());typeRange(h,'size','min',5);h.writes[0].resolve(ranged(1,{min:5,max:null}));await settle();
  await h.undo.click();h.writes[1].reject(Object.assign(Error('Pause first.'),{status:409}));await settle();
  h.reads.at(-1).resolve(ranged(1,{min:5,max:null}));await settle();assert.deepEqual(h.history(),[1,0,0]);
  await h.undo.click();h.writes[2].reject(Object.assign(Error('Changed.'),{status:409}));await settle();
  h.reads.at(-1).resolve(ranged(2,{min:30,max:null}));await settle();assert.deepEqual(h.history(),[0,0,0]);
  assert.equal(h.exclusions.sizeChooser.numbers.min.value,'30');assert.ok(h.notices.includes(EXCLUSION_TEXT.changed));
});

test('a conflict displays the reloaded range while the run detail is still being refreshed',async t=>{
  let hold=false;const refresh=deferred(),h=panel(t,{refresh:()=>hold?refresh.promise:Promise.resolve()});
  await h.ready(ranged());typeRange(h,'size','min',10);h.writes[0].resolve(ranged(1,{min:10,max:null}));await settle();
  hold=true;typeRange(h,'size','min',20);h.writes[1].reject(Object.assign(Error('Changed'),{status:409}));await settle();
  h.reads.at(-1).resolve(ranged(2,{min:30,max:null}));await settle();
  assert.ok(h.exclusions.busy);assert.equal(h.exclusions.sizeChooser.numbers.min.value,'30');
  refresh.resolve();await settle();
});
test('Ctrl+Z on a range handle uses the shared Undo history',async t=>{
  const h=panel(t);await h.ready(ranged());typeRange(h,'size','min',10);h.writes[0].resolve(ranged(1,{min:10,max:null}));await settle();
  h.key('z',{ctrlKey:true,target:h.exclusions.sizeChooser.sliders.min});
  assert.deepEqual(h.writes[1].args,['r',{version:1,min:null,max:null}]);
});
test('whenSettled waits for the held handle and the entire queue before Save can read the state',async t=>{
  const h=panel(t);await h.ready(ranged());const c=h.exclusions.sizeChooser;
  c.sliders.min.emit('pointerdown');c.sliders.min.value='500';c.sliders.min.emit('input');
  let finished=false;const pending=h.exclusions.whenSettled().then(data=>{finished=true;return data;});
  await settle();assert.equal(finished,false);c.sliders.min.emit('change');typeRange(h,'size','max',80);await settle();
  assert.equal(finished,false);const min=h.writes[0].args[1].min;
  h.writes[0].resolve(ranged(1,{min,max:null}));await settle();assert.equal(finished,false);
  h.writes[1].resolve(ranged(2,{min,max:80}));assert.equal((await pending).version,2);
});
test('whenSettled reports a refused write instead of saving an older range',async t=>{
  const h=panel(t);await h.ready(ranged());typeRange(h,'size','min',10);
  const pending=assert.rejects(h.exclusions.whenSettled(),/Pause first/);
  h.writes[0].reject(Object.assign(Error('Pause first.'),{status:422}));await pending;
  assert.equal(h.exclusions.sizeChooser.numbers.min.value,'');
  assert.equal((await h.exclusions.whenSettled()).version,0,'the restored state can be saved after the failure was reported');
});
test('whenSettled has nothing to settle on a run that cannot be reviewed yet (running, failed, no tile): it resolves',async t=>{
  const h=panel(t);
  h.exclusions.update({...run,state:'running'});assert.equal(await h.exclusions.whenSettled(),null);
  h.exclusions.update({...run,id:'f',state:'failed',progress:{done:0,revision:0}});assert.equal(await h.exclusions.whenSettled(),null);
  assert.deepEqual(h.reads,[],'nothing was read');
  // Shown for review with its read failed: Save still hears it.
  h.exclusions.update(run);h.reads[0].reject(Error('offline'));await settle();
  await assert.rejects(h.exclusions.whenSettled(),/could not be loaded/);await assert.rejects(h.exclusions.whenSettled(),/could not be loaded/);
});
test('a read of the areas that failed and then answered never refuses the next Save or Proceed; one failing again is said again',async t=>{
  const h=panel(t);
  h.exclusions.update(run);h.reads[0].reject(Error('offline'));await settle();         // passing: the notice says so
  assert.deepEqual(h.notices,[EXCLUSION_TEXT.error]);
  h.exclusions.update({...run});assert.equal(h.reads.length,2,'the next run event reads them again');
  const pressed=h.exclusions.whenSettled();                                              // Save pressed while it reads
  h.reads[1].resolve({version:0,areas:[],removed:0});
  assert.equal((await pressed).version,0,'a press waiting for the read goes on');
  assert.equal((await h.exclusions.whenSettled()).version,0,'and so does the next');
  // Shown again for review (resumed and paused): that read fails, and the next press hears it once.
  h.exclusions.update({...run,state:'running'});h.exclusions.update(run);h.reads[2].reject(Error('offline'));await settle();
  await assert.rejects(h.exclusions.whenSettled(),/could not be loaded/);
  assert.equal((await h.exclusions.whenSettled()).version,0,'the areas shown are what Save goes with');
});
test('whenSettled rejects a switch of runs during a held range',async t=>{
  const h=panel(t);await h.ready(ranged());h.exclusions.sizeChooser.sliders.min.emit('pointerdown');
  const pending=assert.rejects(h.exclusions.whenSettled(),/run changed/);h.exclusions.reset();await pending;
});

test('cell values failing puts the reason under the chooser, keeps outlines and does not retry at idle',async t=>{
  const h=panel(t);await h.ready(ranged());let reads=0;
  h.exclusions.api.stainCells=async()=>{reads++;throw Error('offline');};
  const priors={};h.exclusions.priors=priors;h.exclusions.setVectors({set:'s',areas_version:0});await settle();
  assert.equal(h.exclusions.sizes.textContent,CELL_VALUES_TEXT.error);assert.equal(priors.classes,null);
  h.exclusions.render();h.exclusions.updateValues();await settle();assert.equal(reads,1);
});

test('the Keeps line in its own words, with grouped thousands; the snapshot counts what it says, live while held',async t=>{
  const h=panel(t);await h.ready({...ranged(),cells:40070,kept:40070});
  assert.equal(h.exclusions.total.textContent,'Keeps all 40,070 cells');
  assert.match(h.exclusions.total.title,/Widen a range to bring cells back\.$/);
  h.exclusions.apply({...ranged(1),cells:40070,kept:38512,removed:1558,by_size:1558});
  assert.equal(h.exclusions.total.textContent,'Keeps 38,512 of 40,070 cells');assert.match(h.exclusions.total.title,/^Left out: 1,558 for their size/);
  h.exclusions.apply({...ranged(2),cells:1,kept:0,removed:1});assert.equal(h.exclusions.total.textContent,'Keeps 0 of 1 cell');
  for(const words of [h.exclusions.total.textContent,h.exclusions.total.title])assert.deepEqual(bannedWords(words),[]);
  // The cells' values in: the live classes count, held or not; pending while held, until the change has gone through.
  h.exclusions.apply(ranged(3));h.exclusions.values=valuesForPanel();h.exclusions.setVectors({set:'s',areas_version:0});
  assert.deepEqual([h.exclusions.snapshot().kept,h.exclusions.snapshot().cells,h.exclusions.snapshot().areas_version],[4,5,0]);
  assert.equal(h.exclusions.pending,false);
  const c=h.exclusions.sizeChooser;c.setValues(h.exclusions.values.data.area);
  c.sliders.max.emit('pointerdown');c.sliders.max.value=String(c.axis.position(20)*1000);c.sliders.max.emit('input');
  assert.equal(h.exclusions.pending,true,'a held handle');
  assert.deepEqual([h.exclusions.snapshot().kept,h.exclusions.snapshot().cells],[3,5]);
  assert.equal(h.exclusions.total.textContent,'Keeps 3 of 5 cells');
  c.sliders.max.emit('change');assert.equal(h.exclusions.pending,true,'on its way');
  h.writes[0].resolve(ranged(4,{min:null,max:20}));await settle();
  assert.equal(h.exclusions.pending,false);assert.equal(h.exclusions.snapshot().kept,3);
});
