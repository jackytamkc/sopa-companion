import test from 'node:test';
import assert from 'node:assert/strict';
import { domFixture } from './dom-fixture.mjs';
import { NOTE_TEXT, sampleTools, sampleNotes, renderNotes, openFailure } from '../src/sample-notes.mjs';
import { bannedWords } from '../src/qc-copy.mjs';

// What the page says about a sample (sample-notes.mjs): the picker's notes on what an open found (the sample detail's
// `notes`), the tools a sample offers (`tools`), and an open that failed. The notes' words are the engine's; these
// are as an engine might write them.
const NOTES = [
  { code: 'pixel_frame', text: 'These transcripts seem to be in pixels, not micrometres: sizes shown as µm are pixels.',
    detail: 'Every image of the sample has 1.0 µm per pixel. Regions, sizes and the default box are in the transcripts\' units.' },
  { code: 'several_points', text: 'This sample has 2 transcript elements: this app uses a_points.', detail: 'Not used: b_points.' },
  { code: 'skipped_images', text: '2 images are left out.', detail: 'he_rotated: rotated. dapi_other: no coordinate system in common with the transcripts.' },
];

test('a sample offers the tools the engine lists; an engine that lists none offers both, unless the sample has no transcripts', () => {
  assert.deepEqual(sampleTools({ tools: ['stain'], transcripts: null }), ['stain']);
  assert.deepEqual(sampleTools({ tools: ['proseg', 'stain'], transcripts: 10 }), ['proseg', 'stain'], 'in the engine\'s order');
  assert.deepEqual(sampleTools({ tools: ['stain', 'other'] }), ['stain'], 'a tool this page does not have is left out');
  assert.deepEqual(sampleTools({ transcripts: 4000 }), ['stain', 'proseg'], 'an engine from before the list');
  assert.deepEqual(sampleTools({ transcripts: null }), ['stain']);
  assert.deepEqual(sampleTools({ tools: [], transcripts: null }), ['stain']);
  assert.deepEqual(sampleTools({ transcripts: 5 }, ['proseg']), ['proseg'], 'among the tools the page has');
});

test('the picker\'s notes are the engine\'s, in its order, and a sample that cannot run proseg always says why', () => {
  assert.deepEqual(sampleNotes({ transcripts: 10, notes: NOTES }), NOTES);
  assert.deepEqual(sampleNotes({ transcripts: 10 }), [], 'an engine from before the notes');
  assert.deepEqual(sampleNotes(null), []);
  assert.deepEqual(sampleNotes({ transcripts: 10, notes: [{ code: 'x', text: '  ' }, null, { text: ' Plain. ', detail: 7 }] }),
    [{ code: '', text: 'Plain.', detail: '' }], 'a note with no words is left out; a detail that is not words is none');
  // Images only: the engine's own note, else the page's, first.
  const own = { code: 'image_only', text: 'Images only here.', detail: 'No points element.' };
  assert.deepEqual(sampleNotes({ transcripts: null, tools: ['stain'], notes: [NOTES[2], own] }), [NOTES[2], own]);
  assert.deepEqual(sampleNotes({ transcripts: null, tools: ['stain'], notes: [NOTES[2]] }),
    [{ code: 'image_only', text: NOTE_TEXT.imageOnly, detail: NOTE_TEXT.imageOnlyDetail }, NOTES[2]]);
  assert.deepEqual(sampleNotes({ transcripts: null }).map(n => n.code), ['image_only'], 'also from an engine that lists no tools');
  // Transcripts but no image it can use (the engine lists proseg only): the picker opens it in Proseg Live, and says
  // why (it moved there without a word, from Stain segmentation).
  assert.deepEqual(sampleNotes({ transcripts: 10, tools: ['proseg'], notes: [NOTES[2]] }),
    [{ code: 'no_image', text: NOTE_TEXT.noImage, detail: NOTE_TEXT.noImageDetail }, NOTES[2]]);
  const said = { code: 'no_image', text: 'No image here.', detail: 'None.' };
  assert.deepEqual(sampleNotes({ transcripts: 10, tools: ['proseg'], notes: [said] }), [said], "the engine's own words first");
  assert.deepEqual(sampleNotes({ transcripts: 10, tools: ['proseg', 'stain'] }), [], 'both tools: nothing to say');
  for (const text of Object.values(NOTE_TEXT)) assert.deepEqual(bannedWords(text), [], text);
});

test('each note is one line with an ⓘ holding the rest (its tooltip and its name, reachable by the keyboard); none: hidden', () => {
  const dom = domFixture(), host = dom.document.createElement('div');
  host.hidden = true; dom.document.body.append(host);
  renderNotes(host, sampleNotes({ notes: NOTES }), dom.document);
  assert.equal(host.hidden, false);
  assert.deepEqual(host.children.map(line => [line.getAttribute('data-note'), line.children[0].textContent]), NOTES.map(n => [n.code, n.text]));
  for (const [i, line] of host.children.entries()) {
    const info = line.children[1];
    assert.equal(info.className, 'note-info'); assert.equal(info.title, NOTES[i].detail); assert.equal(info.getAttribute('aria-label'), NOTES[i].detail);
    assert.equal(info.tabIndex, 0); assert.equal(info.getAttribute('role'), 'img');
    assert.equal(info.children[0].tagName, 'SVG', 'the icon is drawn, not a character');
  }
  const first = host.children[0];
  renderNotes(host, sampleNotes({ notes: NOTES }), dom.document);
  assert.equal(host.children[0], first, 'the same notes: not drawn again');
  renderNotes(host, [{ code: 'z_index', text: 'z is a plane number here.', detail: '' }], dom.document);
  assert.deepEqual(host.children.map(line => line.children.length), [1], 'no detail: no ⓘ');
  renderNotes(host, [], dom.document);
  assert.equal(host.hidden, true); assert.equal(host.children.length, 0);
});

test('an open that failed is one plain line: its first, without a Python error\'s name; the rest only in its tooltip', () => {
  const raw = 'ValidationError: Cannot construct SpatialData object, input contains invalid elements.\nTo fix, run `x`.\n  shapes/a b: Name must contain only letters.';
  assert.deepEqual(openFailure(Object.assign(new Error(raw), { status: 422 })),
    { text: 'Cannot construct SpatialData object, input contains invalid elements.', detail: raw });
  assert.deepEqual(openFailure(Object.assign(new Error('This folder is not a SpatialData zarr'), { status: 422 })),
    { text: 'This folder is not a SpatialData zarr', detail: '' });
  assert.deepEqual(openFailure(Object.assign(new Error("KeyError: 'ome'"), { status: 422 })), { text: "KeyError: 'ome'", detail: '' },
    'a name alone after the error says nothing without it');
  assert.deepEqual(openFailure(Object.assign(new Error('Internal Server Error'), { status: 500 })), { text: NOTE_TEXT.engineFailed, detail: 'Internal Server Error' });
  assert.deepEqual(openFailure(Object.assign(new Error('Storage error: disk full'), { status: 503 })), { text: 'Storage error: disk full', detail: '' },
    'a failure with a reason says it');
  assert.equal(openFailure(new TypeError('Failed to fetch')).text, NOTE_TEXT.noAnswer);
  assert.equal(openFailure('Unreadable folder').text, 'Unreadable folder');
});
