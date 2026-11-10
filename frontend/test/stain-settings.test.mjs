import test from 'node:test';
import assert from 'node:assert/strict';
import { domFixture, deferred, settle } from './dom-fixture.mjs';
import { CellposeSettings, fillCellposeSettings, SETTINGS_TEXT } from '../src/stain-settings.mjs';
import { EXPLAIN_TEXT } from '../src/explain.mjs';
import { bannedWords } from '../src/qc-copy.mjs';

// "Use settings from Cellpose" (the picker's and the patch panel's) against the DOM fixture; explain.mjs writes its
// note into globalThis.document, the fixture's for the test.
function page(t) {
  const previous = globalThis.document, fixture = domFixture();
  globalThis.document = fixture.document;
  fixture.document.body.innerHTML = '<div id="stain-tuning"><button id="stain-use-settings" type="button" disabled>Use settings from Cellpose</button>' +
    '<p id="stain-settings-reason"></p></div>';
  t.after(() => { globalThis.document = previous; });
  const reads = [], used = [];
  const api = { cellposeSettings: (sample, signal) => { const d = deferred(); reads.push({ sample, signal, ...d }); return d.promise; } };
  const settings = new CellposeSettings(api, fixture.$('stain-use-settings'), fixture.$('stain-settings-reason'), async data => { used.push(data); });
  return { ...fixture, settings, button: settings.button, reads, used, note: () => settings.button.nextSibling };
}
const saved = { found: true, from: 'patch-02_seg.npy', settings: { model: 'trained', diameter: 23 } };

test('Use settings from Cellpose is never greyed out: a press reads the newest file, then uses it or says the GUI saved none yet', async t => {
  const h = page(t), button = h.button;
  assert.equal(button.disabled, false, 'even where the page starts it greyed out');
  const passive = h.settings.refresh('s');
  assert.equal(button.disabled, false); assert.equal(button.getAttribute('aria-busy'), null, 'a read on its own does not hold the button');
  h.reads[0].resolve({ found: false }); await passive;
  assert.equal(h.settings.reason.textContent, '', 'nothing said until it is pressed');
  // A press reads again, busy with what it is doing; pressing again meanwhile says the last step is still running.
  const pressing = button.click();
  assert.equal(button.getAttribute('aria-busy'), 'true'); assert.equal(button.textContent, SETTINGS_TEXT.loading);
  await button.click();
  assert.ok(h.note().textContent.includes(EXPLAIN_TEXT.busy)); assert.equal(h.reads.length, 2);
  h.reads[1].resolve({ found: false }); await pressing;
  assert.equal(button.getAttribute('aria-busy'), null); assert.equal(button.textContent, SETTINGS_TEXT.use);
  assert.equal(h.note().hidden, false); assert.match(h.note().textContent, /No corrected patch saved by the Cellpose GUI yet for this sample/);
  assert.doesNotMatch(h.note().textContent, /Still working/, 'the busy note was for the press before');
  assert.deepEqual(h.used, []);
  // Saved in the GUI since: the next press uses the file and the note goes.
  const again = button.click(); h.reads[2].resolve(saved); await again;
  assert.deepEqual(h.used, [saved]); assert.equal(h.note().hidden, true);
  assert.equal(button.getAttribute('aria-busy'), null);
});

test('a press never imports the previous sample; another sample or no sample clears or explains; a failed read says so', async t => {
  const h = page(t), button = h.button;
  h.settings.refresh('a');
  const pressing = button.click();                     // reads 'a' again
  h.settings.refresh('b');                             // the picker moved on meanwhile
  assert.ok(h.reads[1].signal.aborted);
  h.reads[1].resolve(saved); await pressing;
  assert.deepEqual(h.used, [], "the previous sample's settings are never used");
  h.reads[2].resolve({ found: false }); await settle();
  const missing = button.click(); h.reads[3].resolve({ found: false }); await missing;
  assert.equal(h.note().hidden, false);
  h.settings.refresh('c');                             // a read for another sample: that note no longer applies
  assert.equal(h.note().hidden, true);
  h.reads[4].reject(Error('offline')); await settle();
  assert.equal(h.settings.reason.textContent, SETTINGS_TEXT.error);
  const failed = button.click(); h.reads[5].reject(Error('offline')); await failed;
  assert.equal(h.settings.reason.textContent, SETTINGS_TEXT.error); assert.equal(h.note().hidden, true); assert.deepEqual(h.used, []);
  h.settings.refresh(null);                            // no sample (the picker is opening one): said, nothing read
  await button.click();
  assert.match(h.note().textContent, /No corrected patch saved/); assert.equal(h.reads.length, 6);
});

test('the Cellpose settings words are plain', () => {
  for (const words of Object.values(SETTINGS_TEXT)) assert.deepEqual(bannedWords(words), [], words);
});

test('GUI values are copied once while Resolution matches the patches by physical pixel size',()=>{
  const fields=new Map(),get=id=>{if(!fields.has(id))fields.set(id,{});return fields.get(id);};
  const result={resolution:{level:2,um_per_px:.425},settings:{diameter:23,flow_threshold:.4,cellprob_threshold:0,
    normalize:{lowhigh:[-10,300],percentile:null,tile_norm_blocksize:48,sharpen_radius:2,smooth_radius:0,invert:false}}};
  const image={scales:[{level:0,factor:1,um_per_px:.2125},{level:1,factor:2,um_per_px:.425}]};
  const lines=fillCellposeSettings(get,result,image);
  assert.equal(get('stain-level').value,'1');assert.equal(get('stain-diameter').value,'23');
  assert.equal(get('stain-tile-norm').value,'48','engine already converted full-image pixel settings');
  assert.ok(lines.includes("Resolution: scale1 (the patches')"));
  for(const words of lines)assert.deepEqual(bannedWords(words),[]);
});
test('GUI settings retain the selected resolution if this image lacks a physical match',()=>{
  const fields=new Map([['stain-level',{value:'2'}]]),get=id=>{if(!fields.has(id))fields.set(id,{});return fields.get(id);};
  const result={resolution:{um_per_px:.425},settings:{level:1,diameter:null,flow_threshold:.4,cellprob_threshold:0,
    normalize:{percentile:[1,99],tile_norm_blocksize:0,sharpen_radius:0,smooth_radius:0}}};
  const lines=fillCellposeSettings(get,result,{scales:[{level:0,factor:1,um_per_px:1},{level:2,factor:4,um_per_px:4}]});
  assert.equal(get('stain-level').value,'2');assert.ok(lines.some(line=>line.includes('no matching copy')));
});
test('settings without a pixel size keep their own level, named as sdata names it',()=>{
  const fields=new Map(),get=id=>{if(!fields.has(id))fields.set(id,{});return fields.get(id);};
  const result={settings:{level:4,diameter:null,flow_threshold:.4,cellprob_threshold:0,normalize:{percentile:[1,99],
    tile_norm_blocksize:0,sharpen_radius:0,smooth_radius:0}}};
  const scales=[0,1,2,3,4].map(level=>({level,factor:2**level,um_per_px:.2125*2**level}));
  const lines=fillCellposeSettings(get,result,{scales});
  assert.equal(get('stain-level').value,'4');assert.ok(lines.includes("Resolution: scale4 (the patches')"));
  assert.ok(lines.includes('diameter: Auto'));assert.ok(lines.includes('normalize: percentile 1–99'));
});
