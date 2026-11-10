import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import { readFileSync } from 'node:fs';
import { domFixture, deferred, settle } from './dom-fixture.mjs';
import { EXPLAIN_TEXT, NEEDS_CLASS } from '../src/explain.mjs';
import { bannedWords } from '../src/qc-copy.mjs';
import { NOTE_TEXT } from '../src/sample-notes.mjs';

// The sample browser (sample-browser.js) against the DOM fixture. Its imports come from the context; explain.mjs is
// evaluated in the same context, so the document it writes notes into is the fixture's.
const moduleCode = file => readFileSync(new URL(`../src/${file}`, import.meta.url), 'utf8').replace(/^import .*;\n/gm, '').replace(/^export /gm, '');
function load(api, dom) {
  const context = vm.createContext({ document: dom.document, api });
  return vm.runInContext(moduleCode('explain.mjs') + moduleCode('sample-notes.mjs') + moduleCode('sample-browser.js') + '\nSampleBrowser', context);
}
const listing = { path:'/data', parent:'/', shortcuts:[{label:'Data',path:'/data'}], entries:[
  {kind:'dir',name:'nested',path:'/data/nested'}, {kind:'zarr',name:'sample.zarr',path:'/data/sample.zarr'},
] };
const busyNow = button => button.getAttribute('aria-busy') === 'true';

test('browser supports directories, keyboard navigation, selection without opening, and double-click', async () => {
  const dom = domFixture(), opened = [], pending = deferred(), paths = [];
  const Browser = load({
    browse: async path => { paths.push(path); return {...listing,path:path || listing.path}; },
    openSample: path => { paths.push(path); return pending.promise; },
  },dom);
  const browser = new Browser(sample => opened.push(sample)); browser.show(); await settle();
  assert.equal(browser.dialog.open,true);
  dom.$('browse-list').children[0].click(); await settle();
  assert.equal(dom.$('browse-path').value,'/data/nested');
  browser.keydown({key:'Backspace',target:dom.$('browse-list'),preventDefault(){}}); await settle();
  assert.equal(dom.$('browse-path').value,'/');
  browser.keydown({key:'ArrowDown',target:dom.$('browse-list'),preventDefault(){}});
  browser.keydown({key:'ArrowDown',target:dom.document.activeElement,preventDefault(){}});
  assert.equal(browser.selected,1);
  dom.$('browse-path').value='';dom.$('browse-path').oninput();
  assert.equal(dom.$('browse-open').disabled,false,'never greyed out');
  const row = dom.$('browse-list').children[1]; row.click();
  assert.equal(dom.$('browse-list').children[1],row,'first click retains DOM target for native double-click');
  assert.equal(opened.length,0);
  row.ondblclick();
  assert.equal(busyNow(dom.$('browse-open')),true); assert.equal(dom.$('browse-open').textContent,'Opening…');
  assert.equal(dom.$('browse-open').disabled,false); assert.equal(dom.$('browse-progress').hidden,false);
  assert.equal(dom.$('browse-path').disabled,true,'inputs are disabled only while busy');
  pending.resolve({id:'a',name:'sample'}); await settle();
  assert.equal(browser.dialog.open,false); assert.equal(opened[0].id,'a'); assert.equal(paths.at(-1),'/data/sample.zarr');
});

test('browser opens a typed zarr path, displays detail errors, and ignores navigation or opens cancelled by the user', async () => {
  const dom = domFixture(), old = deferred(), opening = deferred(), opened = [], paths = [];
  const Browser = load({
    browse: path => path === '/slow' ? old.promise : path === '/bad' ? Promise.reject(Error('Unreadable folder')) : Promise.resolve({...listing,path:path || '/data'}),
    openSample: path => { paths.push(path); return opening.promise; },
  },dom);
  const browser = new Browser(sample => opened.push(sample)); browser.show(); await settle();
  browser.navigate('/slow'); await browser.navigate('/new'); old.resolve({...listing,path:'/slow'}); await settle();
  assert.equal(dom.$('browse-path').value,'/new');
  await browser.navigate('/bad'); assert.equal(dom.$('browse-error').textContent,'Unreadable folder');
  dom.$('browse-path').value='/typed/sample.zarr'; dom.$('browse-path').oninput(); dom.$('browse-open').click();
  assert.equal(paths.at(-1),'/typed/sample.zarr');
  browser.dialog.close(); opening.resolve({id:'late'}); await settle(); assert.equal(opened.length,0);
  // A sample opened from the dialog closes it mid-step; showing it again starts with Open, not busy.
  browser.show('/data'); await settle();
  assert.equal(browser.dialog.open,false); assert.equal(opened.length,1);
  browser.show(); await settle();
  assert.equal(busyNow(dom.$('browse-open')),false); assert.equal(dom.$('browse-open').textContent,'Open');
});

test('Open is never greyed out: with nothing chosen a press says so at the path; listing a folder keeps it busy', async () => {
  const dom = domFixture(), lists = [], opened = [], tries = [];
  const Browser = load({
    browse: path => { const d = deferred(); lists.push(d); return d.promise.then(result => ({...result,path:path || listing.path})); },
    openSample: async path => { tries.push(path); return {id:'s'}; },
  },dom);
  const browser = new Browser(sample => opened.push(sample)), open = dom.$('browse-open'), path = dom.$('browse-path'), note = dom.$('browse-note');
  browser.show();
  // The start folder is listing: busy, its label unchanged; a press meanwhile says so and opens nothing.
  assert.equal(busyNow(open),true); assert.equal(open.textContent,'Open'); assert.equal(open.disabled,false);
  await open.click();
  assert.equal(note.hidden,false); assert.ok(note.textContent.includes(EXPLAIN_TEXT.busy)); assert.equal(tries.length,0);
  lists[0].resolve({...listing}); await settle();
  assert.equal(busyNow(open),false); assert.equal(note.hidden,true,'the listing arrived: the note goes');
  // Nothing chosen or typed: the press names what is missing, outlines the path and moves there.
  let focused = false; path.focus = () => { focused = true; };
  path.value = ''; path.oninput();
  await open.click();
  assert.equal(tries.length,0); assert.equal(note.hidden,false);
  assert.match(note.textContent,/Choose a folder or type a path\./);
  assert.ok(path.classList.contains(NEEDS_CLASS)); assert.equal(path.getAttribute('aria-invalid'),'true'); assert.equal(focused,true);
  path.value = '/data/sample.zarr'; path.emit('input');            // typing clears the outline and the note
  assert.ok(!path.classList.contains(NEEDS_CLASS)); assert.equal(note.hidden,true);
  path.value = ''; path.oninput(); await open.click(); assert.equal(note.hidden,false);
  dom.$('browse-list').children[1].click();                        // choosing a sample does too
  assert.equal(note.hidden,true);
  await open.click(); await settle();
  assert.deepEqual(tries,['/data/sample.zarr']); assert.equal(opened[0].id,'s'); assert.equal(browser.dialog.open,false);
  for (const words of [note.textContent,'Opening…','Choose a folder or type a path.']) assert.deepEqual(bannedWords(words),[]);
});

// ---- an open that fails says so plainly --------------------------------------------------------------------------------
test('a sample that cannot be opened is said in one plain line; the whole answer stays in its tooltip', async () => {
  const dom = domFixture(), opened = [];
  // spatialdata's own refusal, as the helper used to pass it on: its class, a link, a fix that does not apply, the element.
  const raw = 'ValidationError: Cannot construct SpatialData object, input contains invalid elements.\n' +
    'For renaming, please see the discussion here https://github.com/scverse/spatialdata/discussions/707 .\n' +
    'To fix, run `spatialdata.utils.sanitize_table(adata)`.\n  shapes/cell boundaries: Name must contain only alphanumeric characters, underscores, dots and hyphens.';
  const answers = [Object.assign(new Error(raw), { status: 422 }),
    Object.assign(new Error('Internal Server Error'), { status: 500 }),
    Object.assign(new TypeError('Failed to fetch')),
    Object.assign(new Error("shapes 'cell boundaries': its name has characters SpatialData refuses. Rename or remove it, then open it again."), { status: 422 })];
  const Browser = load({ browse: async path => ({ ...listing, path: path || listing.path }), openSample: async () => { throw answers.shift(); } }, dom);
  const browser = new Browser(sample => opened.push(sample)), error = dom.$('browse-error');
  browser.show(); await settle();
  const open = async () => { dom.$('browse-path').value = '/data/broken.zarr'; dom.$('browse-path').oninput(); await dom.$('browse-open').click(); await settle(); };
  await open();
  assert.equal(error.hidden, false);
  assert.equal(error.textContent, 'Cannot construct SpatialData object, input contains invalid elements.', 'its first line, without the class');
  assert.equal(error.title, raw, 'the whole answer, one hover away');
  assert.doesNotMatch(error.textContent, /sanitize_table|ValidationError|\n/);
  await open();
  assert.equal(error.textContent, NOTE_TEXT.engineFailed, 'an unexpected failure: what happened and where to look');
  await open();
  assert.equal(error.textContent, NOTE_TEXT.noAnswer);
  await open();
  assert.match(error.textContent, /^shapes 'cell boundaries': its name has characters SpatialData refuses/, 'a plain answer as it is');
  assert.equal(error.title, '', 'nothing more to say');
  assert.deepEqual(opened, []);
  for (const words of [NOTE_TEXT.engineFailed, NOTE_TEXT.noAnswer]) assert.deepEqual(bannedWords(words), [], words);
  // Choosing again clears it.
  dom.$('browse-path').value = '/data'; dom.$('browse-path').oninput();
  assert.equal(error.hidden, true); assert.equal(error.title, '');
});
