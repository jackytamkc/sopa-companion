// What the page says about a sample: as it opens, and when it cannot be opened. No imports: every page test evaluates it
// where it needs it.
//
// As it opens (the sample detail's `notes` and `tools`), the engine notes what the sample does not hold, or
// holds in a way the page cannot see for itself: transcripts that seem to be in pixels (code "pixel_frame"), several
// transcript elements of which one is used ("several_points"), images left out ("skipped_images"), z given as plane
// numbers ("z_index"), images without transcripts ("image_only"), and any other note an open finds. Each is one short
// line in the picker under the sample's name, with an ⓘ whose tooltip holds the rest. A sample without transcripts
// offers Stain segmentation only: the picker opens it there and says why.

export const NOTE_TEXT = Object.freeze({
  // The page's own words for a sample the engine offers Stain segmentation alone (its note, when it sends none).
  imageOnly: 'This sample has no transcripts: only stain segmentation runs here.',
  imageOnlyDetail: 'Proseg builds its cells from transcripts, and this sample holds images only. Stain segmentation finds cells in the images.',
  // ... and for one it offers Proseg Live alone (transcripts, but no image it can show).
  noImage: 'This sample has no image to segment: only proseg runs here.',
  noImageDetail: 'Stain segmentation finds cells in an image, and this sample has none the app can use. Proseg builds its cells from the transcripts.',
  noTranscripts: 'No transcripts',
  // A failed open whose answer gave no reason: the engine failed on something it did not expect (its log has it), or
  // it did not answer at all.
  engineFailed: 'The engine failed while reading this sample. Press Log to save what happened.',
  noAnswer: 'The engine did not answer. Check that it is still running, then try again.',
});

/** What the page says when a sample (or a folder) could not be read (the open dialog, the picker): the engine's reason
 *  as one plain line (its first, without the name of a Python error before a sentence), and the whole of what it said
 *  for the line's tooltip (empty when that is the line). An answer without a reason (an unexpected failure: "Internal
 *  Server Error") or no answer at all says so in plain words. Returns {text, detail}. */
export function openFailure(error) {
  const whole = String(error?.message ?? error ?? '').trim(), status = Number(error?.status);
  const unexpected = status >= 500 && (!whole || /^(internal server error|bad gateway|service unavailable|gateway timeout)$/i.test(whole));
  if (unexpected) return { text: NOTE_TEXT.engineFailed, detail: whole };
  if (!Number.isFinite(status) && (error?.name === 'TypeError' || !whole)) return { text: NOTE_TEXT.noAnswer, detail: whole };
  const first = whole.split('\n').map(line => line.trim()).find(Boolean) || '';
  const text = first.replace(/^(?:[A-Za-z_][\w.]*\.)?[A-Z]\w*(?:Error|Exception): +(?=\S+ \S)/, '');
  return { text, detail: text === whole ? '' : whole };
}

/** The ⓘ icon (an inline SVG drawn in the text's colour, one stroke weight: as the mixing check's). */
const INFO_ICON = '<svg viewBox="0 0 16 16" aria-hidden="true" focusable="false"><circle cx="8" cy="8" r="6.25"></circle>'
  + '<path d="M8 7.25v4M8 4.75v.01"></path></svg>';

/** The tools the sample can be used in, among `known` (the first screen's: shell.mjs TOOLS), in the engine's order
 *  (`tools`). An engine that does not list them: both, unless the sample has no transcripts (`transcripts` null). */
export function sampleTools(detail, known = ['stain', 'proseg']) {
  const listed = Array.isArray(detail?.tools) ? detail.tools.filter(tool => known.includes(tool)) : [];
  if (listed.length) return listed;
  return known.filter(tool => tool !== 'proseg' || detail?.transcripts !== null);
}

/** The sample's notes as the picker shows them, [{code, text, detail}]: the engine's in its order (one with no words is
 *  left out), and the page's own line for a sample that cannot run proseg, or stain segmentation, when the engine sent
 *  none for it (the picker opens such a sample in the tool it can run, and this says why). */
export function sampleNotes(detail) {
  const notes = (Array.isArray(detail?.notes) ? detail.notes : []).filter(note => typeof note?.text === 'string' && note.text.trim())
    .map(note => ({ code: String(note.code ?? ''), text: note.text.trim(), detail: typeof note.detail === 'string' ? note.detail.trim() : '' }));
  const tools = detail ? sampleTools(detail) : [];
  if (detail && !tools.includes('proseg') && !notes.some(note => note.code === 'image_only'))
    notes.unshift({ code: 'image_only', text: NOTE_TEXT.imageOnly, detail: NOTE_TEXT.imageOnlyDetail });
  if (detail && !tools.includes('stain') && !notes.some(note => note.code === 'no_image'))
    notes.unshift({ code: 'no_image', text: NOTE_TEXT.noImage, detail: NOTE_TEXT.noImageDetail });
  return notes;
}

/** `notes` (sampleNotes) into `host`, one line each: its words, then an ⓘ with its detail as tooltip and accessible name
 *  (focusable, so the keyboard can read it). The host is hidden while there are none. Drawn again only on a change. */
export function renderNotes(host, notes, doc = globalThis.document) {
  if (!host) return;
  const key = JSON.stringify(notes || []);
  if (host.notesKey === key) return;
  host.notesKey = key; host.replaceChildren();
  for (const note of notes || []) {
    const line = doc.createElement('p'); line.className = 'picker-note'; line.setAttribute('data-note', note.code);
    const words = doc.createElement('span'); words.textContent = note.text; line.append(words);
    if (note.detail) {
      const info = doc.createElement('span'); info.className = 'note-info'; info.tabIndex = 0; info.setAttribute('role', 'img');
      info.innerHTML = INFO_ICON; info.title = note.detail; info.setAttribute('aria-label', note.detail);
      line.append(info);
    }
    host.append(line);
  }
  host.hidden = !(notes || []).length;
}
