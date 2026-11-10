import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import { readFileSync } from 'node:fs';
import { domFixture } from './dom-fixture.mjs';

// Side-panel policies of main.js: per-frame updates rewrite only what changed, and nothing renders while hidden.
const source = readFileSync(new URL('../src/main.js', import.meta.url), 'utf8');
const html = readFileSync(new URL('../index.html', import.meta.url), 'utf8');
const slice = (from, to) => source.slice(source.indexOf(from), source.indexOf(to));

function harness() {
  const dom = domFixture(); dom.document.body.innerHTML = html;
  const panel = dom.$('now-panel'); panel.open = true; panel.hidden = false;
  const list = dom.document.createElement('div'); list.id = 'active-cells'; panel.append(list);
  const ranked = [];
  const context = vm.createContext({
    $: dom.$, document: dom.document, fmt: new Intl.NumberFormat('en-GB'), ACTIVE: ['preparing', 'running', 'paused', 'finishing'],
    state: { mode: 'run', run: { id: 'a' }, runs: [], current: 5, cells: [] },
    cellStats: { current: null }, analysis: { select() {} },
    mostActive: (stats) => { ranked.push(stats); return stats.top; },
    cellName: label => `cell ${label - 1}`,
  });
  vm.runInContext('const shown = { runs: "", activeRun: null, activeStats: null, activeNames: null, active: "" };\n' +
    slice('// ---- DOM writes only on change', 'function updateHud()') + slice('function facts(target, rows)', 'function updateFacts()') +
    slice('function updateActiveCells()', '// ---- region picking') + slice('function renderRuns()', 'let noticeTimer'), context);
  return { ...dom, context, ranked, list, panel, run: code => vm.runInContext(code, context) };
}

test('most active cells: ranked once per statistics set, kept while the next frame loads, untouched when collapsed', () => {
  const h = harness(), c = h.context, top = [{ id: 3, changed: 40 }, { id: 9, changed: 12 }];
  c.cellStats.current = { seq: 5, top };
  for (let k = 0; k < 3; k++) h.run('updateActiveCells()');
  assert.equal(h.ranked.length, 1, 'several calls per frame rank the cells once');
  assert.deepEqual(h.list.children.map(b => b.textContent), ['Most active cells', 'cell 3 · 40 changed', 'cell 9 · 12 changed']);
  const buttons = h.list.children;
  c.state.current = 6; c.cellStats.current = null; h.run('updateActiveCells()');
  assert.deepEqual(h.list.children, buttons, 'the list stays while the new frame’s statistics load');
  c.cellStats.current = { seq: 6, top: [...top] }; h.run('updateActiveCells()');
  assert.equal(h.ranked.length, 2);
  assert.equal(h.list.children[1], buttons[1], 'an unchanged ranking rebuilds nothing');
  const selected = [];
  c.analysis.select = id => selected.push(id);
  c.cellStats.current = { seq: 6, top: [{ id: 4, changed: 41 }, { id: 3, changed: 40 }] }; h.run('updateActiveCells()');
  assert.equal(h.list.children[1], buttons[1], 'a new ranking of the same length reuses the buttons');
  assert.deepEqual(h.list.children.map(b => b.textContent), ['Most active cells', 'cell 4 · 41 changed', 'cell 3 · 40 changed']);
  h.list.children[1].click(); assert.deepEqual(selected, [4], 'a reused button selects its current cell');
  h.panel.open = false; c.state.current = 7; c.cellStats.current = { seq: 7, top: [{ id: 1, changed: 5 }] }; h.run('updateActiveCells()');
  assert.equal(h.ranked.length, 3, 'a collapsed panel is not updated');
  h.panel.open = true; h.run('updateActiveCells()');
  assert.deepEqual(h.list.children.map(b => b.textContent), ['Most active cells', 'cell 1 · 5 changed']);
  c.state.run = { id: 'b' }; c.cellStats.current = null; h.run('updateActiveCells()');
  assert.deepEqual(h.list.children.map(b => b.textContent), ['Most active cells'], 'another run starts empty');
});

test('most active cells: a frame whose statistics cannot be loaded does not keep the previous ranking', () => {
  const h = harness(), c = h.context, top = [{ id: 3, changed: 40 }, { id: 9, changed: 12 }];
  c.cellStats.current = { seq: 5, top }; h.run('updateActiveCells()');
  assert.equal(h.list.children.length, 3);
  c.state.current = 6; c.cellStats.current = null; c.cellStats.failed = 5; h.run('updateActiveCells()');
  assert.equal(h.list.children.length, 3, 'loading (another frame failed earlier): the list stays');
  c.cellStats.failed = 6; h.run('updateActiveCells()');                // HTTP 404, a frame without statistics, ...
  assert.deepEqual(h.list.children.map(b => b.textContent), ['Most active cells']);
  c.state.current = 7; c.cellStats.failed = null; c.cellStats.current = { seq: 7, top }; h.run('updateActiveCells()');
  assert.deepEqual(h.list.children.map(b => b.textContent), ['Most active cells', 'cell 3 · 40 changed', 'cell 9 · 12 changed'],
    'the next statistics fill it again, even with the same ranking');
});

test('facts update their values in place when the terms are unchanged', () => {
  const h = harness(), dl = h.$('facts');
  h.run('facts($("facts"), [["Phase", "Burn-in"], ["Iteration", "12"], ["Log-likelihood", null]])');
  const nodes = dl.children;
  assert.deepEqual(nodes.map(n => n.textContent), ['Phase', 'Burn-in', 'Iteration', '12']);
  h.run('facts($("facts"), [["Phase", "Burn-in"], ["Iteration", "13"], ["Log-likelihood", ""]])');
  assert.deepEqual(dl.children, nodes); assert.equal(nodes[3].textContent, '13');
  h.run('facts($("facts"), [["Phase", "Sampling"], ["Iteration", "14"], ["Log-likelihood", "-1.2e6"]])');
  assert.deepEqual(dl.children.map(n => n.textContent), ['Phase', 'Sampling', 'Iteration', '14', 'Log-likelihood', '-1.2e6']);
});


test('the header has no run menu and the controller builds no run list',()=>{
  assert.doesNotMatch(html,/runs-button|runs-list|runs-count/);
  assert.doesNotMatch(source,/function renderRuns/);
});
