import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readdirSync, readFileSync } from 'node:fs';

// The page and its tests are shared as they are, so none names a real sample, the tissue a sample came from, or a real
// cell: fixtures name made-up samples (sample-1) and cells whose ids are runs of one letter (ffaaaaaa-1), and comments say
// "a whole sample". Every page file (its code, styles and index.html) and every test is read. The patterns are built from
// parts, so this file never matches itself.
const FRONTEND = new URL('../', import.meta.url);
const listed = (folder, pattern) => readdirSync(new URL(folder, FRONTEND)).filter(name => pattern.test(name)).map(name => folder + name);
const PAGE = [...listed('src/', /\.(m?js|css)$/), ...listed('styles/', /\.css$/), 'index.html'];
const TESTS = listed('test/', /\.m?js$/);
const SAMPLE_CODE = new RegExp('\\b' + 'd' + 'u\\d{2}\\b', 'i');
const TISSUE = new RegExp('\\b(?:' + ['li' + 'ver', 'hepato' + 'cytes?', 'kupf' + 'fer'].join('|') + ')\\b', 'i');
const CELL_ID = /\b[a-p]{8}-1\b/g;
const madeUp = id => /^(?:([a-p])\1+)+-1$/.test(id);      // runs of one letter: ffaaaaaa-1, aabbccdd-1
// The genes of the panel the Go deeper fixtures once took their numbers from, kept as the first 12 hex digits of the
// sha256 of each upper-case name, so that this file never names them: fixtures name made-up genes (GXPN, GENE1).
const PANEL = new Set(('00d339a0772f 018db5050739 08de1840e0bd 0bec84246d4c 20954e310897 230d48cd45eb 23815676d2f5 ' +
  '2447320cd98e 3568d51588db 3c64c1960069 43c7b73b33d8 46bc5375d02e 4aa57950dc75 4cfc1db6dc7d 4de4c0643275 57d6d052856e ' +
  '5baa6d1e19b2 5ef5586ae976 5fa9b7315773 63934b49e161 7eb7ec3ee042 84147e6fcacc 90155a8b35a6 95d142ddd562 a232fe6d807a ' +
  'a5e27fe0befb a6709ecc4f25 aa5c877b2356 b0cdb95c9fea b3ac85909757 b4747a8f0dc3 b797acaa686d bd8874d7067d bedaf520cad8 ' +
  'da8718174046 e03180de9ca5 ec78455c8fea ec7d5935a4e5 f713e5e0e9df fd520c2e4f67 fdd1adda7ee5').split(' '));
const hashes = new Map();
const panelGene = word => {
  if (!hashes.has(word)) hashes.set(word, PANEL.has(createHash('sha256').update(word.toUpperCase()).digest('hex').slice(0, 12)));
  return hashes.get(word);
};

/** Every line of `files` that `found(line)` holds something for, as file:line: what. */
function lines(files, found) {
  return files.flatMap(file => readFileSync(new URL(file, FRONTEND), 'utf8').split('\n')
    .flatMap((line, n) => { const what = found(line); return what ? [`${file}:${n + 1}: ${what}`] : []; }));
}

test('no page or test file names a real sample by its code', () => {
  assert.ok(PAGE.length > 50 && TESTS.length > 50 && PAGE.includes('src/main.js') && TESTS.includes('test/page-privacy.test.mjs'));
  assert.deepEqual(lines([...PAGE, ...TESTS], line => line.match(SAMPLE_CODE)?.[0]), []);
});

test('no page file names the tissue a sample came from', () => {
  assert.deepEqual(lines(PAGE, line => line.match(TISSUE)?.[0]), []);
});

test("every cell id the page's files and tests name is a made-up one", () => {
  assert.ok(madeUp('ffaaaaaa-1') && madeUp('aabbccdd-1') && !madeUp('abcd' + 'efgh-1') && !madeUp('aaaa' + 'bcdd-1'));
  assert.deepEqual(lines([...PAGE, ...TESTS], line => (line.match(CELL_ID) || []).filter(id => !madeUp(id)).join(' ')), []);
});

test('no page or test file names a gene of that panel, in a fixture, a word or a variable\'s name', () => {
  assert.equal(PANEL.size, 41);
  assert.ok(!panelGene('GXPN') && !panelGene('gene1') && !panelGene('cell'));
  // Words, hyphenated ones whole and in parts (a gene's name may hold a hyphen).
  const words = line => (line.match(/[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*/g) || []).flatMap(word => [word, ...word.split('-')]);
  assert.deepEqual(lines([...PAGE, ...TESTS], line => words(line).filter(word => word.length > 2 && panelGene(word)).join(' ')), []);
});
