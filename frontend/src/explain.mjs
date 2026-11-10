// Buttons are never greyed out for something the user still has to choose or fill in: pressing one says which
// field is missing or not valid and why, outlines those fields and moves to the first. Only work already in progress
// keeps a button busy, and its label then says what it is doing (busy()).
//
// A problem is {field, label, message}: `field` is the element id to outline (null when there is no single field, e.g.
// "draw a region on the image"), `label` the field's name as the page shows it, `message` what it needs, in plain words.

export const EXPLAIN_TEXT = Object.freeze({
  lead: 'Check these first:',
  busy: 'Still working on the last step.',
});
export const NEEDS_CLASS = 'needs-input';

export const problem = (field, label, message) => ({field, label, message});

const notes = new WeakMap();       // button -> its note element
const marked = new WeakMap();      // button -> fields it outlined
const fieldNotes = new WeakMap();  // field -> the reason shown under it (the page jumps there, away from the button)

function unmark(field) {
  field.classList?.remove(NEEDS_CLASS);
  field.removeAttribute?.('aria-invalid');
  const under = fieldNotes.get(field);
  under?.remove();
  fieldNotes.delete(field);
}

function noteFor(button, note, doc) {
  if (note) return note;
  let el = notes.get(button);
  if (!el) {
    el = doc.createElement('div');
    el.className = 'missing-note';
    el.setAttribute('role', 'alert');
    el.hidden = true;
    button.after ? button.after(el) : button.parentElement?.append(el);
    notes.set(button, el);
  }
  return el;
}

/** Remove what explain() showed for this button: its note and the outlines it set. */
export function clearExplanation(button, {note, document: doc = globalThis.document} = {}) {
  const el = note || notes.get(button);
  if (el) { el.hidden = true; el.replaceChildren(); }
  for (const field of marked.get(button) || []) unmark(field);
  marked.delete(button);
}

/** Show `problems` beside `button`: a short list naming each field and what it needs; the fields are outlined, the first
 *  is scrolled into view and focused. A marked field loses its outline as soon as it changes. Returns whether there
 *  was anything to show. */
export function explain(button, problems, {note, document: doc = globalThis.document, focus = true} = {}) {
  clearExplanation(button, {note, document: doc});
  if (!problems?.length) return false;
  const el = noteFor(button, note, doc);
  const lead = doc.createElement('p');
  lead.textContent = EXPLAIN_TEXT.lead;
  const list = doc.createElement('ul');
  for (const p of problems) {
    const item = doc.createElement('li');
    item.textContent = p.label ? `${p.label}: ${p.message}` : p.message;
    list.append(item);
  }
  el.replaceChildren(lead, list);
  el.hidden = false;
  const fields = [];
  for (const p of problems) {
    const field = p.field ? doc.getElementById(p.field) : null;
    if (!field || fields.includes(field)) continue;
    field.classList?.add(NEEDS_CLASS);
    field.setAttribute?.('aria-invalid', 'true');
    const under = doc.createElement('div');           // the reason, right where the page jumps to
    under.className = 'field-note';
    under.textContent = p.message;
    field.after ? field.after(under) : field.parentElement?.append(under);
    fieldNotes.set(field, under);
    const clear = () => {
      unmark(field);
      field.removeEventListener?.('input', clear);
      field.removeEventListener?.('change', clear);
    };
    field.addEventListener?.('input', clear);
    field.addEventListener?.('change', clear);
    fields.push(field);
  }
  marked.set(button, fields);
  const first = fields[0];
  if (focus) {
    first?.scrollIntoView?.({block: 'center'});        // a jump, not smooth scrolling (page-quiet rule)
    first?.focus?.({preventScroll: true});
  }
  return true;
}

/** Wire `button` so a press runs check() first: problems are explained, otherwise the explanation is cleared and act()
 *  runs. The button itself is never disabled for missing input. Returns the click handler (also set as onclick). */
export function guard(button, check, act, options = {}) {
  const handler = async (event) => {
    if (button.getAttribute?.('aria-busy') === 'true') {
      explain(button, [problem(null, '', EXPLAIN_TEXT.busy)], options);
      return;
    }
    const problems = check() || [];
    if (explain(button, problems, options)) return;
    clearExplanation(button, options);
    return act(event);
  };
  button.onclick = handler;
  return handler;
}

/** Mark `button` busy with `label` (e.g. "Starting…") until the returned function is called; a press meanwhile says
 *  the last step is still running instead of starting another. */
export function busy(button, label) {
  const before = button.textContent;
  button.setAttribute('aria-busy', 'true');
  if (label) button.textContent = label;
  return () => {
    button.removeAttribute('aria-busy');
    if (label) button.textContent = before;
  };
}
