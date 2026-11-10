// The page in a notebook's session (API mode): `sopa_companion.launch(sdata)` binds one
// engine to one SpatialData. The engine says so in GET /health's `session` and pushes {ev: "session", ...} when it
// changes; a desktop engine sends {mode: "desktop"} or nothing. What changes on the page: no Open sample…, sample browser,
// Recent or ‹ Both tools; the two cards open the bound sample ("Choose a tool for sample-1"); the header names the sample
// on every screen; the first screen shows the SpatialData loading while it is prepared or opened (a card pressed
// meanwhile waits for it, then opens by itself); Save becomes Add to your SpatialData (the zarr_save states: saving,
// queued, saved, failed); Close ends the session (POST /closing with end: true); and an ended session says so in place
// of the page, with no Reload. Its words are here (SESSION_TEXT: the banned-words tests read them). The page never says
// "sdata": it cannot know the notebook's name for the object.

export const SESSION_TEXT = Object.freeze({
  choose: 'Choose a tool for {name}.',
  sampleZarr: 'From your notebook; results go into this object and {zarr}',
  sampleMemory: 'From your notebook; results go into this object (in memory only)',
  preparing: 'Preparing your SpatialData for the app: {done} of {total} GB',
  preparingStart: 'Preparing your SpatialData for the app…',
  opening: 'Opening {name}…', openingAny: 'Opening your SpatialData…',
  failed: 'Could not open your SpatialData: {reason}', failedAny: 'Could not open your SpatialData.',
  waiting: 'Opens when {name} is ready',
  ended: 'This session has ended. You can close this tab, and start a new one from your notebook.', endedTitle: 'Session ended',
  // A tab with no page key, or one the session's engine no longer takes; a link opened again, or too late.
  signedOutTitle: 'This page is not signed in',
  signedOut: 'Open this session again from your notebook: call .open() on what sopa_companion.launch returned.',
  linkUsedTitle: 'This link has been used',
  linkUsed: 'A link opens the app once, for a short time. For a new one, call .open() on what sopa_companion.launch returned.',
  closing: 'Closing…',
  windowTitle: 'SoPA companion · {name}',
  // Finish & save becomes Add.
  addTitle: 'Add to your SpatialData?', add: 'Add', addShapes: 'Add as shapes', adding: 'Adding…',
  defaultTick: 'Make it the default segmentation (as sopa does)',
  helpProseg: "Shapes and a table of proseg's counts, added as sopa's own functions add them.",
  helpStain: "Cell shapes, added as sopa's own functions add them.",
  alsoWritten: 'Also written to {zarr}.', memoryNoZarr: 'In memory only: this SpatialData has no zarr.',
  memoryAutosaveOff: "In memory only: sopa's auto-save is off.",
  clash: 'Already in your SpatialData: {names}. Adding will replace {them}.',
  queuedZarr: 'Written to {zarr} · in your notebook as soon as it is free', queuedMemory: 'Ready · in your notebook as soon as it is free',
  busy: 'Waiting for your notebook: a cell is running',
  added: 'Added {cells} cells as {names}', addedTo: 'Added to your SpatialData as {names}', addingAs: 'Adding to your SpatialData as {names}',
  viewProgress: 'View progress…',
  clashAtApply: "Not added to your notebook's object: it already has {names} (made after you pressed Add). It is in {zarr}.",
  clashAtApplyMemory: "Not added to your notebook's object: it already has {names} (made after you pressed Add).",
  addButton: 'Add to your SpatialData…', addAgain: 'Add again…', notAdded: 'Not added to your SpatialData yet', addedOutdated: 'Added before these changes: add again to keep them',
});
const fill = (text, values) => text.replace(/\{(\w+)\}/g, (_, key) => values[key] ?? '');
const gb = bytes => new Intl.NumberFormat('en-GB', { minimumFractionDigits: 1, maximumFractionDigits: 1 }).format(bytes / 1e9);
const listed = list => (list || []).filter(Boolean).join(', ');

/** The session as the page keeps it (`state.session`): a notebook engine's health.session, else null (the desktop app,
 *  an engine of an earlier version that sends no `session`). A push {ev: "session", ...} reads the same. */
export function apiSession(health) {
  const s = health && typeof health === 'object' && 'mode' in health && !('session' in health) ? health : health?.session;
  if (!s || typeof s !== 'object' || s.mode !== 'api') return null;
  const sample = s.sample && typeof s.sample === 'object' ? s.sample : {};
  return { mode: 'api', id: typeof s.id === 'string' ? s.id : null,
    sample: { id: sample.id ?? null, name: sample.name || null, backed: sample.backed === true, zarr_name: sample.zarr_name || null,
      view: sample.view ?? null, state: ['preparing', 'opening', 'ready', 'failed'].includes(sample.state) ? sample.state : 'opening',
      progress: { done: Number(sample.progress?.done) || 0, total: Number(sample.progress?.total) || 0 }, message: sample.message || null,
      memory_only: Array.isArray(sample.memory_only) ? sample.memory_only : [], tools: Array.isArray(sample.tools) ? sample.tools : [] },
    tool: ['proseg', 'stain'].includes(s.tool) ? s.tool : null,
    writes: ['zarr', 'memory', 'memory_autosave_off'].includes(s.writes) ? s.writes : 'memory',
    notebook: { busy: s.notebook?.busy === true }, patches: s.patches === true, link: s.link === true,
    state: s.state === 'ending' ? 'ending' : 'running' };
}
/** The bound sample's name for the page: the display name the notebook gave ("sample-1", "sample-1 (in memory)"). */
export const sessionName = session => session?.sample?.name || 'your SpatialData';
/** Where a result goes ({zarr}: the object's zarr when it is written too). */
export const sessionZarr = session => session?.writes === 'zarr' ? session.sample?.zarr_name || null : null;
/** The header's sample name and its tooltip (never a scratch path). */
export function sampleHeader(session) {
  const zarr = sessionZarr(session);
  return { name: sessionName(session), title: zarr ? fill(SESSION_TEXT.sampleZarr, { zarr }) : SESSION_TEXT.sampleMemory };
}
/** The window's title, so two sessions' windows can be told apart. */
export const windowTitle = session => session?.sample?.name ? fill(SESSION_TEXT.windowTitle, { name: session.sample.name }) : 'SoPA companion';
/** The first screen's line about the SpatialData: {text, state} while it is prepared (with `fraction`, the part done,
 *  once its size is known), opened or failed; null once ready. */
export function sampleStatus(session) {
  const sample = session?.sample;
  if (!sample || sample.state === 'ready') return null;
  if (sample.state === 'preparing') {
    const { done, total } = sample.progress;
    return total > 0 ? { state: 'preparing', text: fill(SESSION_TEXT.preparing, { done: gb(done), total: gb(total) }), fraction: Math.min(1, done / total) }
      : { state: 'preparing', text: SESSION_TEXT.preparingStart };
  }
  if (sample.state === 'failed') return { state: 'failed', text: sample.message ? fill(SESSION_TEXT.failed, { reason: sample.message }) : SESSION_TEXT.failedAny };
  return { state: 'opening', text: sample.name ? fill(SESSION_TEXT.opening, { name: sample.name }) : SESSION_TEXT.openingAny };
}
/** A card pressed while the SpatialData is still prepared or opened: "Opens when sample-1 is ready". */
export const waitingText = session => fill(SESSION_TEXT.waiting, { name: sessionName(session) });
/** The cards' heading: "Choose a tool for sample-1." */
export const chooseText = session => fill(SESSION_TEXT.choose, { name: sessionName(session) });

// ---- Add to your SpatialData (the Save dialog and the results card) ----------------------------------------------------
/** The Save dialog's words in a session: {title, submit, tick, help} for `run` (a stain run adds shapes only). */
export function addWords(session, run) {
  const stain = run?.kind === 'stain', zarr = sessionZarr(session);
  const where = session?.writes === 'zarr' && zarr ? fill(SESSION_TEXT.alsoWritten, { zarr })
    : session?.writes === 'memory_autosave_off' ? SESSION_TEXT.memoryAutosaveOff : SESSION_TEXT.memoryNoZarr;
  return { title: SESSION_TEXT.addTitle, submit: stain ? SESSION_TEXT.addShapes : SESSION_TEXT.add, tick: SESSION_TEXT.defaultTick,
    help: `${stain ? SESSION_TEXT.helpStain : SESSION_TEXT.helpProseg} ${where}`, saving: SESSION_TEXT.adding };
}
const keysOf = save => [save?.shapes_key, save?.table_key].filter(Boolean).join(' + ');
/** The line for a save in a session: saving (its own line), queued (in the object's zarr, or ready, until the
 *  notebook is free), saved, a clash at apply; '' otherwise. */
export function addLine(session, run) {
  const save = run?.zarr_save, zarr = save?.written_to ? String(save.written_to).split('/').filter(Boolean).at(-1) : sessionZarr(session);
  if (!save) return '';
  if (save.state === 'queued') return session?.notebook?.busy ? SESSION_TEXT.busy
    : save.written_to ? fill(SESSION_TEXT.queuedZarr, { zarr }) : SESSION_TEXT.queuedMemory;
  if (save.state === 'saved') return fill(SESSION_TEXT.added, { cells: new Intl.NumberFormat('en-GB').format(save.cells || 0), names: keysOf(save) });
  if (save.state === 'failed' && Array.isArray(save.clash) && save.clash.length)
    return save.written_to ? fill(SESSION_TEXT.clashAtApply, { names: listed(save.clash), zarr }) : fill(SESSION_TEXT.clashAtApplyMemory, { names: listed(save.clash) });
  return '';
}
/** The results card's save row in a session: {label, button}. `outdated`: saved before the run's cells changed. A queued
 *  result is on its way (in the object's zarr, or ready) until the notebook takes it: "Adding to your SpatialData as …",
 *  never "Added" before the notebook has it (addLine says where it is meanwhile). */
export function resultsRow(run, { outdated = false } = {}) {
  const save = run?.zarr_save, state = save?.state, done = ['saved', 'queued'].includes(state);
  return { label: outdated ? SESSION_TEXT.addedOutdated : state === 'saved' ? fill(SESSION_TEXT.addedTo, { names: keysOf(save) })
    : state === 'queued' ? fill(SESSION_TEXT.addingAs, { names: keysOf(save) }) : SESSION_TEXT.notAdded,
    button: state === 'saving' ? SESSION_TEXT.viewProgress : run?.kind === 'stain' ? SESSION_TEXT.addShapes : done ? SESSION_TEXT.addAgain : SESSION_TEXT.addButton };
}
