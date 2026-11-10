import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, readdirSync } from 'node:fs';
import { COPIED, LEGACY, MemoryStorage, PREFIX, bootStorage, copyLegacyKeys } from '../src/storage-boot.mjs';

// The previous version of the app kept the page's memory under 'proseg-live:' and 'proseg-live-' keys: the desktop
// page carries them over to this version's 'sopa-companion' keys once, before anything reads storage.
const legacyStore = () => {
  const store = new MemoryStorage();
  store.setItem('proseg-live:layers:v1', '{"xenium":{"opacity":0.4}}');
  store.setItem('proseg-live:umap-width', '681');
  store.setItem('proseg-live:panel:qc-panel', 'false');
  store.setItem('proseg-live-stain-model', 'cpsam_20250102');
  store.setItem('proseg-live-instanseg-model', 'single_channel_nuclei');
  store.setItem('sopa-companion:umap-width', '640');          // set by this version already: kept as it is
  store.setItem('other-app:x', '1');
  return store;
};

test('the previous version\'s keys are copied once to their twins, never over a value this version set, and stay', () => {
  const store = legacyStore();
  assert.equal(copyLegacyKeys(store), 4);
  assert.equal(store.getItem('sopa-companion:layers:v1'), '{"xenium":{"opacity":0.4}}');
  assert.equal(store.getItem('sopa-companion:panel:qc-panel'), 'false');
  assert.equal(store.getItem('sopa-companion-stain-model'), 'cpsam_20250102');
  assert.equal(store.getItem('sopa-companion-instanseg-model'), 'single_channel_nuclei');
  assert.equal(store.getItem('sopa-companion:umap-width'), '640', 'a twin this version wrote wins');
  assert.equal(store.getItem('proseg-live:umap-width'), '681', 'the old keys stay');
  assert.equal(store.getItem('other-app:x'), '1'); assert.equal(store.getItem('sopa-companion:x'), null);
  assert.equal(store.getItem(COPIED), '1');
  // Once: what this version removes later is not brought back by the next start, nor is a key the old one adds.
  store.removeItem('sopa-companion:panel:qc-panel');
  store.setItem('proseg-live:qc:open', 'true');
  assert.equal(copyLegacyKeys(store), 0);
  assert.equal(store.getItem('sopa-companion:panel:qc-panel'), null); assert.equal(store.getItem('sopa-companion:qc:open'), null);
  assert.equal(`${PREFIX}${'proseg-live:a'.slice(LEGACY.length)}`, 'sopa-companion:a');
});

test('the desktop page copies at boot; a notebook session\'s tab copies nothing into the browser\'s storage', () => {
  const real = legacyStore();
  const desktop = { location: { search: '?run=r1' }, sessionStorage: new MemoryStorage(), localStorage: real };
  assert.equal(bootStorage(desktop), null);
  assert.equal(desktop.localStorage, real); assert.equal(real.getItem('sopa-companion:umap-width'), '640');
  assert.equal(real.getItem('sopa-companion:layers:v1'), '{"xenium":{"opacity":0.4}}');
  const browser = legacyStore();
  const tab = { location: { search: '?api=1f3a9c2e' }, sessionStorage: new MemoryStorage(), localStorage: browser };
  const store = bootStorage(tab);
  assert.ok(store); assert.equal(browser.getItem(COPIED), null); assert.equal(browser.getItem('sopa-companion:layers:v1'), null);
  assert.equal(store.getItem('sopa-companion:layers:v1'), null, 'the session\'s own storage starts empty');
});

test('storage that refuses never stops the boot, and what was not copied is copied at the next start', () => {
  const store = legacyStore();
  const real = store.setItem.bind(store);
  let refuse = 2;
  store.setItem = (key, value) => { if (key.startsWith('sopa-companion') && refuse-- <= 0) throw new Error('QuotaExceededError'); real(key, value); };
  assert.equal(copyLegacyKeys(store), -1);
  assert.equal(store.getItem(COPIED), null, 'not marked done while keys are left');
  store.setItem = real;
  assert.equal(copyLegacyKeys(store), 2, 'the rest, at the next start');
  for (const key of ['layers:v1', 'panel:qc-panel']) assert.notEqual(store.getItem(`sopa-companion:${key}`), null);
  assert.equal(store.getItem(COPIED), '1');
  const blocked = { location: { search: '' }, sessionStorage: new MemoryStorage(), get localStorage() { throw new Error('SecurityError'); } };
  assert.equal(bootStorage(blocked), null);
  assert.equal(copyLegacyKeys(null), 0);
  const throwing = { get length() { throw new Error('SecurityError'); }, getItem() { return null; } };
  assert.equal(copyLegacyKeys(throwing), -1);
});

test('every storage key the page names is this version\'s: the previous version\'s prefix only where it is carried over', () => {
  const src = new URL('../src/', import.meta.url);
  const named = [];
  for (const file of readdirSync(src).filter((name) => /\.(m?js)$/.test(name))) {
    const text = readFileSync(new URL(file, src), 'utf8');
    for (const match of text.matchAll(/(['"`])((?:proseg-live|sopa-companion)[:-][^'"`$]*)/g)) named.push([file, match[2]]);
  }
  const old = named.filter(([file, key]) => key.startsWith('proseg-live') && file !== 'storage-boot.mjs');
  assert.deepEqual(old, []);
  assert.ok(named.some(([, key]) => key.startsWith('sopa-companion:layers')), 'the scan finds the page\'s keys');
});
