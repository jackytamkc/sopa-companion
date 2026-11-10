import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { PatchFolderDialog, PATCH_FOLDER_TEXT, holdsLine } from '../src/patch-folder.mjs';
import { EXPLAIN_TEXT, NEEDS_CLASS } from '../src/explain.mjs';
import { bannedWords } from '../src/qc-copy.mjs';
import { domFixture, deferred, settle } from './dom-fixture.mjs';

// Training patches' export dialog (patch-folder.mjs): every export asks where its patches go. The engine's listing is a
// double here; what it answers is engine/patch_folders.py's (tests/test_patch_folders.py).
const T = PATCH_FOLDER_TEXT;
const listing = (over = {}) => ({path: '/data/patches', exists: true, parent: '/data', entries: [
  {name: 'Alpha', path: '/data/patches/Alpha', patches: false}, {name: 'beta', path: '/data/patches/beta', patches: true}],
  shortcuts: [{kind: 'home', name: 'me', path: '/home/me'}, {kind: 'earlier', name: 'sample-2-scale0', path: '/data/sample-2-scale0'}],
  new: 'sample-1-scale0', holds: null, mixes: null, refused: null, ...over});

function fixture(t) {
  const previous = globalThis.document, {document} = domFixture();
  globalThis.document = document;
  t.after(() => { globalThis.document = previous; });
  const lists = [], exports = [], notices = [];
  const api = {stainPatchFolders: (id, path, signal) => { const d = deferred(); lists.push({id, path, signal, ...d}); return d.promise; }};
  const dialog = new PatchFolderDialog(api, {notice: words => notices.push(words)});
  const exportTo = (folder, version, onProgress) => { const d = deferred(); exports.push({folder, version, onProgress, ...d}); return d.promise; };
  const show = (version = 4) => dialog.show({run: {id: 'r1'}, version, exportTo, words: {changed: 'The boxes changed.', failed: 'It failed.'}});
  const $ = id => document.getElementById(id);
  const listed = async (over = {}, n = -1) => { lists.at(n).resolve(listing(over)); await settle(); };
  return {dialog, lists, exports, notices, show, $, listed, document};
}
const shown = node => !node.hidden;
const note = $ => $('patch-folder-note').hidden ? '' : $('patch-folder-note').textContent;
const rows = $ => $('patch-folder-list').children.filter(n => n.tagName === 'BUTTON');
const keydown = (dialog, target, name) => { for (const fn of dialog.dialog.listeners.get('keydown')) fn({key: name, target, preventDefault() {}}); };

test('every word is plain: the dialog\'s, and the engine\'s refusals it shows', () => {
  for (const [name, words] of Object.entries(T)) assert.deepEqual(bannedWords(words), [], name);
  const source = readFileSync(new URL('../../engine/patch_folders.py', import.meta.url), 'utf8');
  const start = source.indexOf('WORDS = {'), block = source.slice(start, source.indexOf('\n}\n', start));
  const found = [...block.matchAll(/^ {4}"(\w+)": "([^"]+)",$/gm)];
  assert.equal(found.length, 20, 'one line an entry');
  for (const [, name, words] of found) assert.deepEqual(bannedWords(words), [], name);
  assert.equal(holdsLine({patches: 1, sets: 1, samples: ['sample-1'], more: 0}), 'Holds 1 patch of sample-1.');
  assert.equal(holdsLine({patches: 1234, sets: 5, samples: ['a', 'b', 'c'], more: 2}), 'Holds 1,234 patches of a, b, c and 2 more.');
  assert.equal(holdsLine(null), '');
});

test('show lists the suggestion; while it lists the line says so, the list is busy and Export here says it is busy', async t => {
  const {dialog, lists, exports, show, $, listed} = fixture(t);
  assert.equal(dialog.dialog.open, false);
  show();
  assert.equal(dialog.dialog.open, true);
  assert.deepEqual(lists.map(l => [l.id, l.path]), [['r1', undefined]]);
  assert.ok(shown($('patch-folder-progress')) && $('patch-folder-progress').textContent === T.loading);
  assert.equal($('patch-folder-list').getAttribute('aria-busy'), 'true');
  $('patch-folder-export').click(); await settle();
  assert.equal(exports.length, 0); assert.match(note($), new RegExp(EXPLAIN_TEXT.busy));
  await listed({holds: {patches: 1234, sets: 2, samples: ['a', 'b', 'c'], more: 2}, mixes: 'channels'});
  assert.equal($('patch-folder-path').value, '/data/patches');
  assert.deepEqual(rows($).map(r => r.children[0].textContent), ['Alpha', 'beta']);
  assert.deepEqual(rows($).map(r => r.children.filter(c => c.className === 'badge').map(c => c.textContent)), [[], [T.holdsBadge]]);
  assert.equal($('patch-folder-status').textContent, 'Holds 1,234 patches of a, b, c and 2 more.');
  assert.ok(shown($('patch-folder-mix')) && $('patch-folder-mix').textContent === T.mixChannels);
  assert.ok(!shown($('patch-folder-error')) && !shown($('patch-folder-progress')) && note($) === '');
  assert.equal($('patch-folder-list').getAttribute('aria-busy'), 'false');
  assert.deepEqual($('patch-folder-shortcuts').children.map(b => [b.textContent, b.title]),
    [['Home', '/home/me'], ['Earlier: sample-2-scale0', '/data/sample-2-scale0']]);
  assert.equal($('patch-folder-lead').textContent.trim(), T.lead);
  assert.ok($('patch-folder-lead').all().some(n => n.title === T.leadTip), 'the tooltip of the lead line');
});

test('a new folder, a refused one and an empty one each read as they are', async t => {
  const {dialog, show, $, listed} = fixture(t);
  show();
  await listed({exists: false, entries: [], holds: null});
  assert.equal($('patch-folder-status').textContent, T.newHere);
  assert.equal($('patch-folder-list').children.at(-1).textContent, T.none);
  $('patch-folder-path').value = '/data/.hidden'; keydown(dialog, $('patch-folder-path'), 'Enter');
  await listed({path: '/data/.hidden', exists: false, entries: [], refused: '.hidden is a hidden folder.'});
  assert.ok(!shown($('patch-folder-status')), 'no "new folder" beside a refusal');
  assert.ok(shown($('patch-folder-error')) && $('patch-folder-error').textContent === '.hidden is a hidden folder.');
});

test('a row, the field, Backspace, the parent button, a shortcut and New folder each list the folder they name', async t => {
  const {dialog, lists, show, $, listed} = fixture(t);
  show(); await listed();
  rows($)[1].click(); assert.equal(lists.at(-1).path, '/data/patches/beta'); await listed();
  $('patch-folder-path').value = '  /typed/here '; keydown(dialog, $('patch-folder-path'), 'Enter');
  assert.equal(lists.at(-1).path, '/typed/here'); await listed();
  keydown(dialog, $('patch-folder-list'), 'Backspace'); assert.equal(lists.at(-1).path, '/data'); await listed();
  $('patch-folder-parent').click(); assert.equal(lists.at(-1).path, '/data'); await listed();
  $('patch-folder-shortcuts').children[1].click(); assert.equal(lists.at(-1).path, '/data/sample-2-scale0'); await listed();
  $('patch-folder-new').click(); assert.equal(lists.at(-1).path, '/data/patches/sample-1-scale0');
  await listed({path: '/', parent: null, new: 'sample-1-scale0'});
  assert.equal($('patch-folder-parent').hidden, true, 'nothing above /');
  $('patch-folder-new').click(); assert.equal(lists.at(-1).path, '/sample-1-scale0');
  const before = lists.length; await listed({path: '/', parent: null});
  keydown(dialog, $('patch-folder-list'), 'Backspace'); assert.equal(lists.length, before, 'Backspace at / asks nothing');
  // The arrow keys choose a row, Enter enters it.
  await listed();
  keydown(dialog, $('patch-folder-path'), 'ArrowDown'); keydown(dialog, $('patch-folder-list'), 'ArrowDown');
  assert.deepEqual(rows($).map(r => r.getAttribute('aria-selected')), ['false', 'true']);
  keydown(dialog, $('patch-folder-list'), 'Enter'); assert.equal(lists.at(-1).path, '/data/patches/beta');
});

test('a listing answered after a newer one, or after typing began, is never shown', async t => {
  const {dialog, lists, show, $} = fixture(t);
  show(); lists[0].resolve(listing()); await settle();
  rows($)[0].click(); rows($)[1].click();
  lists[2].resolve(listing({path: '/data/patches/beta', entries: []})); await settle();
  lists[1].resolve(listing({path: '/data/patches/Alpha'})); await settle();
  assert.equal($('patch-folder-path').value, '/data/patches/beta'); assert.equal(dialog.snapshot().path, '/data/patches/beta');
  $('patch-folder-new').click();                      // a listing on its way, then the user types: theirs stays
  $('patch-folder-path').value = '/being/typed'; $('patch-folder-path').oninput();
  lists.at(-1).resolve(listing({path: '/data/patches/beta/sample-1-scale0'})); await settle();
  assert.equal($('patch-folder-path').value, '/being/typed');
});

test('Export here explains an empty field or a refused folder and posts nothing; else it exports the field\'s folder with show()\'s version', async t => {
  const {dialog, exports, show, $, listed} = fixture(t);
  show(7); await listed({refused: 'data could not be read.'});
  $('patch-folder-export').click(); await settle();
  assert.equal(exports.length, 0); assert.match(note($), /Folder: data could not be read\./);
  assert.ok($('patch-folder-path').classList.contains(NEEDS_CLASS));
  $('patch-folder-path').value = ''; $('patch-folder-path').oninput();
  $('patch-folder-export').click(); await settle();
  assert.equal(exports.length, 0); assert.match(note($), new RegExp(T.choose));
  $('patch-folder-path').value = ' /data/patches/new '; $('patch-folder-path').oninput();
  $('patch-folder-export').click(); await settle();
  assert.deepEqual(exports.map(e => [e.folder, e.version]), [['/data/patches/new', 7]]);
  assert.equal($('patch-folder-export').textContent, T.exporting); assert.equal($('patch-folder-cancel').textContent, T.close);
  assert.equal($('patch-folder-progress').textContent, T.exportingLine);
  // How far it is, once the panel hears of a patch written (an export with the model's outlines: a minute a patch on a
  // CPU); before the first, the general words.
  exports[0].onProgress({done: 0, total: 8});
  assert.equal($('patch-folder-progress').textContent, T.exportingLine);
  exports[0].onProgress({done: 3, total: 8});
  assert.equal($('patch-folder-progress').textContent, 'Exporting patch 3 of 8… You can close this; the export goes on.');
  assert.deepEqual(dialog.snapshot(), {open: true, loading: false, exporting: true, path: '/data/patches', exists: true,
    refused: 'data could not be read.', mixes: null});
  for (const node of dialog.dialog.all()) assert.equal(node.disabled, false, node.id || node.tagName);
});

test('a folder whose real name ends in a space is listed again, checked and exported as shown; what is typed loses its spaces', async t => {
  const {dialog, lists, exports, show, $, listed} = fixture(t);
  show(); await listed({entries: [{name: 'shared ', path: '/data/patches/shared ', patches: false}]});
  rows($)[0].click(); assert.equal(lists.at(-1).path, '/data/patches/shared ');
  await listed({path: '/data/patches/shared ', entries: []});
  assert.equal($('patch-folder-path').value, '/data/patches/shared ');
  keydown(dialog, $('patch-folder-path'), 'Enter'); assert.equal(lists.at(-1).path, '/data/patches/shared ');
  await listed({path: '/data/patches/shared ', entries: [], refused: 'You cannot write to shared .'});
  $('patch-folder-export').click(); await settle();
  assert.equal(exports.length, 0, 'the refused folder shown is not exported into');
  assert.match(note($), /Folder: You cannot write to shared \./);
  keydown(dialog, $('patch-folder-path'), 'Enter');
  await listed({path: '/data/patches/shared ', entries: []});
  $('patch-folder-export').click(); await settle();
  assert.deepEqual(exports.map(e => e.folder), ['/data/patches/shared ']);
  exports[0].resolve({}); await settle();
  show(); await listed();
  $('patch-folder-path').value = ' /data/patches/shared '; $('patch-folder-path').oninput();
  keydown(dialog, $('patch-folder-path'), 'Enter'); assert.equal(lists.at(-1).path, '/data/patches/shared');
  await listed({path: '/data/patches/shared', entries: []});
  $('patch-folder-path').value = '/data/patches/shared '; $('patch-folder-path').oninput();
  $('patch-folder-export').click(); await settle();
  assert.deepEqual(exports.map(e => e.folder), ['/data/patches/shared ', '/data/patches/shared'], 'typed: trimmed');
});

test('while the export runs, every way to another folder says it is busy and asks nothing', async t => {
  const {dialog, lists, exports, show, $, listed} = fixture(t);
  show(); await listed();
  $('patch-folder-export').click(); await settle();
  const asked = lists.length;
  const presses = [() => rows($)[0].click(), () => $('patch-folder-shortcuts').children[0].click(), () => $('patch-folder-parent').click(),
    () => $('patch-folder-new').click(), () => keydown(dialog, $('patch-folder-path'), 'Enter'),
    () => keydown(dialog, $('patch-folder-list'), 'Backspace'), () => $('patch-folder-export').click()];
  for (const press of presses) {
    $('patch-folder-note').hidden = true;
    press(); await settle();
    assert.match(note($), new RegExp(EXPLAIN_TEXT.busy));
  }
  assert.equal(lists.length, asked); assert.equal(exports.length, 1);
  exports[0].resolve({}); await settle();
  assert.equal(dialog.dialog.open, false, 'done: the dialog closes'); assert.equal(dialog.snapshot().exporting, false);
});

test('a refusal of the folder keeps the dialog open with its words; boxes that changed and other failures close it with a notice', async t => {
  const {dialog, exports, notices, show, $, listed} = fixture(t);
  show(); await listed();
  $('patch-folder-export').click(); await settle();
  exports[0].reject(Object.assign(new Error('You cannot write to patches.'), {status: 422, data: {field: 'folder'}})); await settle();
  assert.equal(dialog.dialog.open, true); assert.equal($('patch-folder-error').textContent, 'You cannot write to patches.');
  assert.ok($('patch-folder-path').classList.contains(NEEDS_CLASS)); assert.equal($('patch-folder-path').getAttribute('aria-invalid'), 'true');
  assert.equal($('patch-folder-export').textContent, T.export); assert.deepEqual(notices, []);
  $('patch-folder-path').oninput();
  assert.ok(!$('patch-folder-path').classList.contains(NEEDS_CLASS) && !shown($('patch-folder-error')));
  $('patch-folder-export').click(); await settle();
  exports[1].reject(Object.assign(new Error('The boxes changed. Reload them and try again'), {status: 409, data: {reason: 'changed'}})); await settle();
  assert.equal(dialog.dialog.open, false); assert.deepEqual(notices, ['The boxes changed.']);
  show(); await listed();
  $('patch-folder-export').click(); await settle();
  exports[2].reject(Object.assign(new Error('These patches are being exported already. Wait for it to finish'), {status: 409, data: {}}));
  await settle();
  assert.equal(dialog.dialog.open, false); assert.equal(notices.at(-1), 'These patches are being exported already. Wait for it to finish');
  show(); await listed();
  $('patch-folder-export').click(); await settle();
  exports[3].reject(Object.assign(new Error(''), {status: 503})); await settle();
  assert.equal(notices.at(-1), 'It failed.');
});

test('closing never stops the export: its result reaches the notice; after the panel\'s reset nothing shows', async t => {
  const {dialog, exports, notices, show, $, listed} = fixture(t);
  show(); await listed();
  $('patch-folder-export').click(); await settle();
  $('patch-folder-cancel').click();
  assert.equal(dialog.dialog.open, false); assert.equal(dialog.snapshot().exporting, true, 'still exporting');
  exports[0].reject(Object.assign(new Error('You cannot write to patches.'), {status: 422, data: {field: 'folder'}})); await settle();
  assert.deepEqual(notices, ['You cannot write to patches.'], 'a folder refusal after closing: the notice says it');
  show(); await listed();
  $('patch-folder-export').click(); await settle();
  dialog.close(true);
  exports[1].reject(Object.assign(new Error('late'), {status: 503})); await settle();
  assert.deepEqual(notices, ['You cannot write to patches.']); assert.equal(dialog.snapshot().exporting, false);
  show();
  dialog.close(true);
  const asked = dialog.snapshot();
  assert.deepEqual([asked.open, asked.loading], [false, false]);
});

test('the dialog is built without innerHTML, and nothing in it is ever disabled', async t => {
  const source = readFileSync(new URL('../src/patch-folder.mjs', import.meta.url), 'utf8');
  assert.doesNotMatch(source, /innerHTML|\.disabled\s*=/);
  assert.doesNotMatch(source, /^import .* from '\.\/stain-patches\.mjs'/m);
  const {dialog, show, $, listed} = fixture(t);
  show(); await listed();
  assert.equal($('patch-folder').getAttribute('aria-labelledby'), 'patch-folder-title');
  assert.equal($('patch-folder-list').getAttribute('role'), 'listbox');
  assert.equal($('patch-folder-path').getAttribute('aria-label'), T.path);
  for (const node of dialog.dialog.all()) assert.equal(node.disabled, false, node.id || node.tagName);
  assert.deepEqual(Object.keys(dialog.snapshot()), ['open', 'loading', 'exporting', 'path', 'exists', 'refused', 'mixes']);
});
