import test from 'node:test';
import assert from 'node:assert/strict';
import { explain, clearExplanation, guard, busy, problem, EXPLAIN_TEXT, NEEDS_CLASS } from '../src/explain.mjs';
import { bannedWords } from '../src/qc-copy.mjs';
import { domFixture } from './dom-fixture.mjs';

function page() {
  const {document} = domFixture();
  document.body.innerHTML = '<label>Model <select id="model"></select></label><input id="size"><button id="go">Start</button>';
  return {document, button: document.getElementById('go')};
}

test('the words are plain', () => {
  for (const words of Object.values(EXPLAIN_TEXT)) assert.deepEqual(bannedWords(words), [], words);
});

test('a blur explanation marks the field without scrolling or taking focus',()=>{
  const {document,button}=page(),field=document.getElementById('size');
  field.scrollIntoView=field.focus=()=>assert.fail('blur must not steal focus');
  explain(button,[problem('size','Size','type a number')],{document,focus:false});
  assert.equal(field.getAttribute('aria-invalid'),'true');
});

test('a press with something missing names the fields, outlines them and moves to the first', () => {
  const {document, button} = page();
  const model = document.getElementById('model'), size = document.getElementById('size');
  let focused = null, scrolled = null;
  model.focus = () => { focused = 'model'; };
  model.scrollIntoView = () => { scrolled = 'model'; };
  const shown = explain(button, [problem('model', 'Model', 'choose one'), problem('size', 'Patch size', 'at most 1,320 pixels')],
    {document});
  assert.equal(shown, true);
  const note = button.nextSibling || button.parentElement.children.at(-1);
  assert.equal(note.hidden, false);
  assert.match(note.textContent, /Model: choose one/);
  assert.match(note.textContent, /Patch size: at most 1,320 pixels/);
  assert.ok(model.classList.contains(NEEDS_CLASS) && size.classList.contains(NEEDS_CLASS));
  assert.equal(model.getAttribute('aria-invalid'), 'true');
  assert.equal(focused, 'model'); assert.equal(scrolled, 'model');
  // The reason also sits right under each field, where the page jumps to.
  assert.equal(model.nextSibling.className, 'field-note'); assert.equal(model.nextSibling.textContent, 'choose one');
  assert.equal(size.nextSibling.textContent, 'at most 1,320 pixels');
  model.emit('change');                                  // fixing a field clears its own outline and reason
  assert.ok(!model.classList.contains(NEEDS_CLASS) && size.classList.contains(NEEDS_CLASS));
  assert.notEqual(model.nextSibling?.className, 'field-note');
  clearExplanation(button, {document});
  assert.ok(!size.classList.contains(NEEDS_CLASS) && note.hidden);
  assert.notEqual(size.nextSibling?.className, 'field-note');
  assert.equal(explain(button, [], {document}), false);
});

test('guard runs the action only when nothing is missing, and never disables the button', async () => {
  const {document, button} = page();
  let missing = [problem(null, '', 'Draw a region on the image.')], ran = 0;
  guard(button, () => missing, () => { ran++; }, {document});
  await button.onclick();
  assert.equal(ran, 0); assert.equal(button.disabled, false);
  missing = [];
  await button.onclick();
  assert.equal(ran, 1);
  const done = busy(button, 'Starting…');
  assert.equal(button.textContent, 'Starting…');
  await button.onclick();
  assert.equal(ran, 1);                                  // a press while busy says so instead of starting again
  done();
  assert.equal(button.textContent, 'Start'); assert.equal(button.getAttribute('aria-busy'), null);
});
