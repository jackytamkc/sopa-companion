// The one-run shell: the first screen's two tools, the header, and leaving a run (every way of leaving, Close among them)
// with one prompt. The engine owns saving and removal; the page waits for its gone event. In a notebook's session (API
// mode, session.mjs) the same prompt speaks of your SpatialData.
import { saveOutdated } from './zarr-save-state.mjs';

export const SHELL_TEXT = Object.freeze({
  name: 'SoPA companion', home: 'Back to the first screen', choose: 'Choose a tool to begin.',
  back: '‹ Both tools', backTitle: 'Back to the two tools',
  close: 'Close', closeTitle: 'Close SoPA companion',
  save: 'Save to the sample', discard: 'Discard', keep: 'Stay', saving: 'Saving…', discarding: 'Discarding…', retry: 'Try again',
  leaveTitle: 'Save this run to {sample}?', filteredTitle: 'Leave this run?',
  filtered: 'Your filters leave no cells, so there is nothing to save.',
  helpFinish: 'Save finishes it, then writes its cells to {sample} under new names. Discard deletes it.',
  helpWrite: 'Save writes its cells to {sample} under new names. Discard deletes it.',
  helpTile: 'Save stops it after this tile and writes the cells of its finished tiles to {sample} under new names. Discard deletes it.',
  helpNewest: 'Save writes its newest cells to {sample} under new names. Discard deletes it and keeps what you saved before.',
  savingTo: 'Saving to {sample}…', settingSaved: 'Saved, it can be compared with the new run in Compare.',
  waiting: 'It is being saved already: wait, or press Stay.',
  couldnt: "Couldn't {action}: {reason}", actionSave: 'save', actionDiscard: 'discard', actionStay: 'cancel', actionAdd: 'add',
  actionLeave: 'leave this run',
  closeAsk: 'Close SoPA companion?', closeLead: 'This run is not saved to {sample} yet:',
  closeHelp: 'Save and close saves it to the sample first (finishing it if it still samples). Discard and close deletes it.',
  closeSave: 'Save and close', closeDiscard: 'Discard and close', closeKeep: 'Keep working',
  closeDecided: 'Being saved already, whatever you choose:',
  closeDecidedHelp: 'Finish & save was pressed, so it is saved to {sample} under new names before the app closes.',
  closedElsewhere: 'This run was closed in another window.',
  startedElsewhere: 'A run was started in another window.',
  notStarted: 'The run that had not started was discarded.',
  patchesDiscarded: 'Your patch boxes were not exported, so they were discarded.',
  engineDown: 'Cannot reach the engine', engineLate: 'It did not answer within {seconds} seconds.', reload: 'Reload',
  // A tab with no page key, or one the engine no longer takes (it started again since): it asks the engine nothing.
  signedOutTitle: 'This page is not signed in',
  signedOut: 'Open SoPA companion again to sign in: run sopa-companion launch, or use its icon.',
  // A link opened again, or too late (a launch code is used once, within seconds or minutes).
  linkUsedTitle: 'This link no longer works',
  linkUsed: 'Each link works only once and expires quickly. To open SoPA companion again, run sopa-companion launch or use its icon.',
  // Close when the browser does not let the page close its own tab: the app has closed all the same (its page said
  // goodbye, so the engine shuts down), and the tab says so in place of the page; with the app still open in another
  // tab, the engine goes on for it, and only this tab is closed.
  closedTitle: 'SoPA companion has closed', closedTab: 'You can close this tab.',
  closedHereTitle: 'SoPA companion is closed in this tab', closedHere: 'It is still open in another tab. You can close this one.',
  // A run the engine paused while no window showed it (a crash, a lost connection, a tab the browser put away).
  awayPaused: 'This run was paused while no window showed it. Press Resume to carry on.',
  // The browser drew nothing: no WebGL2 (often a remote or virtual display, where Chrome blocks the graphics card).
  noWebglTitle: "This window can't draw the tissue",
  noWebglWhy: 'Its browser has WebGL2 turned off, which the app needs to draw. This happens on some remote or virtual displays.',
  noWebglHelp: 'Let the app\'s window use the graphics card: in its settings, under [desktop], chrome_flags = ["--use-angle=vulkan", '
    + '"--ignore-gpu-blocklist"] (from a notebook: ~/.config/sopa-companion/config.toml), then open it again. Or open its link in '
    + 'another browser.',
  // The same in any other browser, which has no chrome_flags (only the app's own Chrome window has them).
  noWebglHelpBrowser: "Turn on your browser's graphics acceleration in its settings, then reload. Or open the app in another browser, such as Chrome, Edge or Firefox.",
  engineFailed: 'The engine ran into a problem it did not expect. Press Log to save what happened.',
});
/** The same prompt in a notebook's session: the results go into the SpatialData passed in (never named "sdata": the
 *  page cannot know the variable's name). */
export const SHELL_API_TEXT = Object.freeze({
  save: 'Add to your SpatialData', saving: 'Adding…', leaveTitle: 'Add this run to your SpatialData?',
  helpFinish: 'Add finishes it, then adds its cells under new names. Discard deletes it.',
  helpWrite: 'Add adds its cells under new names. Discard deletes it.',
  helpTile: 'Add stops it after this tile and adds the cells of its finished tiles under new names. Discard deletes it.',
  helpNewest: 'Add adds its newest cells under new names. Discard deletes it and keeps what you added before.',
  savingTo: 'Adding to your SpatialData…', settingSaved: 'Added, it can be compared with the new run in Compare.',
  waiting: 'It is being added already: wait, or press Stay.',
  closeLead: 'This run is not added to your SpatialData yet:',
  closeHelp: 'Add finishes the run if it still samples, then adds its result under new names (nothing is replaced). Discard deletes it. Results you already added stay.',
  closeSave: 'Add to your SpatialData and close',
  closeDecided: 'Being added already, whatever you choose:',
  closeDecidedHelp: 'Finish & save was pressed, so it is added under new names before the session ends.',
});
const fill = (text, values) => text.replace(/\{(\w+)\}/g, (_, key) => values[key] ?? '');

/** The first screen's two tools, in their order (index.html holds the same words; a test keeps them equal). */
export const TOOLS = Object.freeze({
  stain: Object.freeze({ name: 'Stain segmentation',
    about: "Find cells in the sample's images with Cellpose, clean them up, and make training patches for the Cellpose GUI.",
    open: 'A .zarr written by spatialdata or sopa, with its stain images. It needs no transcripts.' }),
  proseg: Object.freeze({ name: 'Proseg Live',
    about: 'Watch proseg build cells from the transcripts, one iteration at a time, with the ResolVI mixing check and Go deeper.',
    open: 'A .zarr written by spatialdata or sopa. Proseg needs its transcripts and cell or nucleus shapes to start from.' }),
});

/** The tool a run type belongs to: Stain segmentation for stain runs and Training patches items, Proseg Live otherwise. */
export const toolOf = kind => ['stain', 'patches'].includes(kind) ? 'stain' : 'proseg';

const ACTIVE = ['preparing', 'running', 'paused', 'finishing'];
/** dict.get(key, fallback) as the engine reads its run.json (a key present with null stays null). */
const get = (o, key, fallback = undefined) => o && typeof o === 'object' && Object.hasOwn(o, key) ? o[key] : fallback;
/** Python's == for the JSON values compared here (None == None). */
const same = (a, b) => (a ?? null) === (b ?? null);
const truthy = v => !!v && !(Array.isArray(v) && !v.length) && !(typeof v === 'object' && !Array.isArray(v) && !Object.keys(v).length);

/** What leaving a run would lose: {kind: none | saved | saving | ask, can_add, why} (why: a stable reason, never shown).
 *  The engine's leave_kind (engine/runs.py) line for line; both are pinned by tests/data/leave_cases.json. */
export function leaveKind(run) {
  const result = (kind, why, can_add = false) => ({ kind, can_add, why });
  const meta = run || {};
  const test = get(meta, 'qc_test');
  if (truthy(test) && (typeof test !== 'object' || Array.isArray(test) || truthy(get(test, 'of')))) return result('none', 'test');
  const saved = truthy(get(meta, 'zarr_save')) ? meta.zarr_save : {};
  const state = get(saved, 'state');
  if (state === 'saving' || state === 'queued' && !truthy(get(saved, 'written_to'))) return result('saving', 'saving');
  const kind = get(meta, 'kind', 'proseg'), runState = get(meta, 'state');
  if (kind === 'patches') return result('none', 'patches');
  let current = ['saved', 'queued'].includes(state);
  if (kind === 'stain') {
    const progress = truthy(get(meta, 'progress')) ? meta.progress : {};
    const model = (truthy(get(meta, 'summary_model')) ? meta.summary_model : truthy(get(meta, 'summary')) ? meta.summary : null) || {};
    if (get(meta, 'stain_started') === false || !truthy(get(progress, 'done'))) return result('none', 'not_started');
    if (get(model, 'cells') === 0) return result('none', 'no_cells');
    if (get(truthy(get(meta, 'summary')) ? meta.summary : {}, 'cells') === 0) return result('ask', 'filtered');
    const exclusions = truthy(get(meta, 'exclusions')) ? meta.exclusions : {};
    const digest = get(saved, 'exclusions_digest'), now = get(exclusions, 'digest');
    const sameFilters = digest != null && now != null ? same(digest, now) : same(get(saved, 'exclusions_version', 0), get(exclusions, 'version', 0));
    current = current && same(get(saved, 'result_revision'), get(progress, 'revision')) && sameFilters;
    if (current) return result('saved', 'saved');
  } else {
    if (current) return result('saved', 'saved');
    if (['stopped', 'failed'].includes(runState)) return result('none', 'ended');
    if (runState === 'preparing' || get(meta, 'phase') === 'loading' || get(meta, 'frames') === 0 && ACTIVE.includes(runState))
      return result('none', 'not_started');
    const at = (truthy(get(meta, 'last')) ? meta.last : truthy(get(meta, 'ack')) ? meta.ack : null) || {};
    if (runState === 'paused' && get(meta, 'frames') === 1 && !truthy(get(meta, 'steps')) && !truthy(get(meta, 'finish_requested')) &&
        !truthy(get(meta, 'schedule_done')) && get(at, 'iter', 0) === 0) return result('none', 'not_started');
  }
  if (get(meta, 'sample_kind') === 'xenium') return result('ask', 'sample');
  return result('ask', ['saved', 'queued', 'outdated'].includes(state) ? 'changed' : 'unsaved', true);
}

/** A proseg run whose Finish & save was pressed, no save tried since and its Save dialog not cancelled, finishing or
 *  done: decided (closing waits for it and saves it under new names whatever the answer; a leave still asks). The
 *  engine's finish_and_save; pinned by each case's `decided` in tests/data/leave_cases.json. */
export function decidedAtClose(run) {
  const meta = run || {}, test = get(meta, 'qc_test');
  if (truthy(test) && (typeof test !== 'object' || Array.isArray(test) || truthy(get(test, 'of')))) return false;
  return get(meta, 'kind', 'proseg') === 'proseg' && truthy(get(meta, 'finish_requested')) && (get(meta, 'zarr_save') ?? null) === null &&
    !truthy(get(meta, 'save_cancelled')) && ['finishing', 'done'].includes(get(meta, 'state')) && get(meta, 'sample_kind', 'zarr') === 'zarr';
}

export const pending = runs => (runs || []).filter(run => ['ask', 'saving'].includes(leaveKind(run).kind));
const fmt = new Intl.NumberFormat('en-GB');
/** A stain run's method in a run's line: Cellpose's by name; another (a run of an earlier version's InstanSeg, dormant
 *  now) as what it is, a stain segmentation. */
const stainMethodWord = run => !run.stain?.method || run.stain.method === 'cellpose' ? 'Cellpose' : 'Stain segmentation';
export function runLine(run) {
  if (!run) return '';
  const iter = run.last?.iter ?? run.ack?.iter ?? run.iter, at = Number.isFinite(iter) ? ` at iteration ${fmt.format(iter)}` : '';
  let line;
  if (run.kind === 'stain') line = `${stainMethodWord(run)} · ${fmt.format(run.progress?.done ?? run.tiles?.done ?? 0)} of ${fmt.format(run.progress?.total ?? run.tiles?.of ?? 0)} tiles done`;
  else if (run.schedule_done) line = `At the end of proseg's standard run${at}`;
  else line = `${({running: 'Sampling', paused: 'Paused', preparing: 'Preparing', finishing: 'Finishing', done: 'Finished', failed: 'Stopped', stopped: 'Stopped'})[run.state] || 'Starting'}${at}`;
  if (run.zarr_save?.state === 'failed') line += ' · the last save failed';
  else if (saveOutdated(run)) line += ' · changed since you saved it';
  return line;
}

export const CLOSE_CHECK_MS = 1500;
let lastRevision = 0;
/** A strictly increasing number for this page's answers (leave, closing): the engine never lets an older answer
 *  override a newer one. Time-based, so the answers of two windows order by when they were given. */
export function nextRevision(now = Date.now()) {
  lastRevision = Math.max(lastRevision + 1, Math.floor(Number.isFinite(now) ? now : 0));
  return lastRevision;
}
const stainSampling = run => run.kind === 'stain' && ['running', 'paused', 'finishing'].includes(run.state);
const prosegSampling = run => run.kind !== 'stain' && ['running', 'paused', 'finishing'].includes(run.state);
/** The prompt's words for `run`: {title, lead, line, help, save, discard, stay, progress, saveTitle, waiting}.
 *  `reason`: what is being left ('close' for Close; 'setting' for New run with this setting); `api`: a notebook's session.
 *  Closing, a run whose Finish & save counts as answered is named as being saved whatever the answer. */
export function leaveWords(run, { reason = 'leave', api = false } = {}) {
  const T = api ? { ...SHELL_TEXT, ...SHELL_API_TEXT } : SHELL_TEXT, sample = run?.sample_name || run?.sample || 'the sample';
  const kind = leaveKind(run), filtered = kind.why === 'filtered', close = reason === 'close', decided = close && decidedAtClose(run);
  const help = filtered ? T.filtered : kind.why === 'changed' ? T.helpNewest : stainSampling(run) ? T.helpTile : prosegSampling(run) ? T.helpFinish : T.helpWrite;
  return {
    title: close ? T.closeAsk : filtered ? T.filteredTitle : fill(T.leaveTitle, { sample }),
    lead: decided ? T.closeDecided : close && !filtered ? fill(T.closeLead, { sample }) : '', line: runLine(run),
    help: decided ? fill(T.closeDecidedHelp, { sample }) : close && !filtered ? fill(T.closeHelp, { sample }) : fill(help, { sample }),
    save: close ? T.closeSave : T.save, discard: close ? T.closeDiscard : T.discard, stay: close ? T.closeKeep : T.keep,
    saving: T.saving, progress: fill(T.savingTo, { sample }), saveTitle: reason === 'setting' ? T.settingSaved : '', waiting: T.waiting,
    action: { add: api ? T.actionAdd : T.actionSave, discard: T.actionDiscard, stay: T.actionStay, leave: T.actionLeave },
  };
}
/** What the engine's last leave of `run` was doing when it failed (`leave_error` {answer, message}, Runs._leave): a
 *  discard, a save, or, for a run saved already, only leaving it (stopping and removing it failed). */
const failedAction = (run, answer) => answer === 'discard' ? 'discard' : leaveKind(run).kind === 'saved' ? 'leave' : 'add';
/** The engine's last leave of `run` failed: "Couldn't discard: The run has not stopped yet." ('' when it did not). The
 *  engine keeps it on the run until another leave or a save starts, or Stay, so the prompt says it too. */
export function leaveFailure(run, { api = false } = {}) {
  const failed = run?.leave_error;
  if (!failed?.message) return '';
  return fill(SHELL_TEXT.couldnt, { action: leaveWords(run, { api }).action[failedAction(run, failed.answer)], reason: failed.message });
}

/** One prompt for every way of leaving, Close included. answer(answer, run, revision) starts the engine's leave; the
 *  run's gone push ends it. While a Save or Discard is on its way to the engine, Stay (and Esc) wait for its reply, so
 *  a late Discard can never land after a Stay; a Discard the engine is carrying out is not taken back. */
export class LeavePrompt {
  constructor({ answer = null, log = null } = {}, doc = globalThis.document) {
    Object.assign(this, { answer, doc, log: log || (() => {}) });
    this.answered = null; this.sent = false; this.run = null; this.reason = null; this.api = false; this.revision = 0;
    const d = this.dialog = doc.createElement('dialog');
    d.className = 'modal leave-popup'; d.id = 'leave-popup'; d.setAttribute('aria-labelledby', 'leave-title');
    d.innerHTML = `<h2 id="leave-title"></h2><p id="leave-lead" class="leave-lead" hidden></p><p id="leave-run"></p><p id="leave-help" class="muted small"></p>
      <p id="leave-progress" role="status" hidden></p><p id="leave-error" class="error" role="alert" hidden></p>
      <div class="modal-actions"><button id="leave-stay" type="button">Stay</button><button id="leave-discard" type="button" class="danger">Discard</button><button id="leave-save" type="button" class="primary">Save to the sample</button></div>`;
    doc.body.append(d); this.$ = id => doc.getElementById(id);
    this.$('leave-save').onclick = () => this.choose('add');
    this.$('leave-discard').onclick = () => this.choose('discard');
    this.$('leave-stay').onclick = () => this.choose('stay');
    d.addEventListener('cancel', e => { e.preventDefault(); this.choose('stay'); });
  }
  get open() { return !!this.dialog.open; }
  /** A notebook's session (session.mjs): the words of your SpatialData. */
  setMode({ api = false } = {}) { this.api = !!api; if (this.open && this.run) this.update(this.run); }
  show(run, { reason = 'leave' } = {}) {
    this.reason = reason; this.error = ''; this.answered = null; this.lastAnswer = null; this.sent = false;
    this.update(run);
    if (!this.open) this.dialog.showModal();
    this.$(leaveKind(run).can_add ? 'leave-save' : 'leave-stay').focus();
  }
  /** A discard on its way, or carried out by the engine. */
  discarding(run = this.run) { return run?.leaving?.answer === 'discard' || this.sending && this.lastAnswer === 'discard'; }
  /** Closing a notebook's session: its answer ends the session, and the engine completes an Add already under way
   *  whatever the answer, so a save under way holds nothing back there. */
  get endsSession() { return this.api && this.reason === 'close'; }
  update(run) {
    this.run = run;
    const $ = this.$, kind = leaveKind(run), words = leaveWords(run, { reason: this.reason, api: this.api });
    const underWay = !!run.leaving || kind.kind === 'saving';
    const discarding = this.discarding(run), waiting = this.sending || underWay && !this.endsSession;
    $('leave-title').textContent = words.title;
    $('leave-lead').textContent = words.lead; $('leave-lead').hidden = !words.lead;
    $('leave-run').textContent = words.line;
    $('leave-help').textContent = words.help;
    // What went wrong last: this window's answer refused (this.error), else the engine's last leave of the run
    // (leave_error: a Discard whose run would not stop…), which says more than the failed save it may have left behind.
    const failed = !this.error && !waiting && run.leave_error?.message ? run.leave_error : null;
    const error = this.error || failed?.message || (run.zarr_save?.state === 'failed' && !waiting ? run.zarr_save.message : '');
    const action = failed ? failedAction(run, failed.answer) : this.lastAnswer === 'discard' ? 'discard' : this.lastAnswer === 'stay' ? 'stay' : 'add';
    $('leave-error').textContent = error ? fill(SHELL_TEXT.couldnt, { action: words.action[action], reason: error }) : ''; $('leave-error').hidden = !error;
    $('leave-save').hidden = kind.why === 'filtered';
    $('leave-save').textContent = waiting && !discarding ? words.saving : error && action === 'add' ? SHELL_TEXT.retry : words.save;
    $('leave-save').setAttribute('aria-busy', String(waiting && !discarding));
    $('leave-save').title = words.saveTitle;
    $('leave-discard').textContent = words.discard;
    $('leave-discard').setAttribute('aria-busy', String(discarding));
    $('leave-discard').title = waiting && !discarding ? words.waiting : '';
    $('leave-stay').textContent = words.stay;
    $('leave-progress').hidden = !waiting && !underWay;
    $('leave-progress').textContent = discarding ? SHELL_TEXT.discarding : run.leaving?.message || run.zarr_save?.message || words.progress;
  }
  async choose(answer) {
    if (!this.run) return;
    // Closing, a run whose Finish & save counts as answered is saved whatever is pressed (its words say so).
    if (answer === 'discard' && this.reason === 'close' && decidedAtClose(this.run)) answer = 'add';
    // Stay waits while a Save or Discard is on its way (its reply decides) and never takes back a Discard.
    if (answer === 'stay' && (this.sending && this.lastAnswer !== 'stay' || this.discarding())) return;
    if (answer !== 'stay' && (this.sending || !this.endsSession && (this.run.leaving || leaveKind(this.run).kind === 'saving'))) return;
    if (answer === 'add' && !leaveKind(this.run).can_add) return;
    const run = this.run, ticket = this.ticket = (this.ticket || 0) + 1, revision = this.revision = nextRevision();
    this.lastAnswer = answer; this.answered = answer === 'stay' ? null : answer; this.sent = false; this.error = ''; this.sending = true;
    this.log(`leave: ${answer}`); this.update(run);
    try {
      await this.answer?.(answer, run, revision);
      if (ticket !== this.ticket) return;
      this.sent = true;
      if (answer === 'stay') this.hide();
    } catch (error) {
      if (ticket === this.ticket) { this.error = error.message || String(error); this.answered = null; }
    } finally {
      if (ticket === this.ticket) { this.sending = false; if (this.open) this.update(this.run); }
    }
  }
  hide() { if (this.open) this.dialog.close(); }
  reset() { this.ticket = (this.ticket || 0) + 1; this.sending = false; this.answered = null; this.sent = false; this.hide(); }
  snapshot() { return { open: this.open, runs: this.run ? [this.run.id] : [], answered: this.answered, sent: this.sent,
    reason: this.reason, stage: this.run?.leaving?.stage ?? null }; }
}

/** Crash leftovers are reported once for each engine-start stamp, even after reloading the page; so are the runs of
 *  earlier versions the start removed because they were saved already or had nothing to save (`older_removed`). */
const seenLeftovers = new Set();
export function leftoversNotice(leftovers, storage = globalThis.localStorage) {
  const runs = Array.isArray(leftovers?.runs) ? leftovers.runs : [];
  const older = Number.isInteger(leftovers?.older_removed) && leftovers.older_removed > 0 ? leftovers.older_removed : 0;
  if (!leftovers?.stamp || !runs.length && !older) return '';
  const key = `sopa-companion:leftovers:${leftovers.stamp}`;
  if (seenLeftovers.has(key)) return '';
  try { if (storage?.getItem(key)) return ''; storage?.setItem(key, 'seen'); } catch { /* optional storage */ }
  seenLeftovers.add(key);
  const name = run => run.sample_name || run.sample, plain = runs.filter(run => !run.failed && !run.still_saving);
  const lines = [plain.length === 1 ? `A run on ${name(plain[0])} that was not saved to the sample was removed when the app started.` :
    plain.length > 1 ? `${plain.length} runs that were not saved to their samples were removed when the app started.` : ''];
  for (const run of runs.filter(run => run.failed)) lines.push(`Saving the run on ${name(run)} failed when the app closed${typeof run.failed === 'string' ? ` (${run.failed})` : ''}, so it was removed when the app started.`);
  for (const run of runs.filter(run => run.still_saving)) lines.push(`The run on ${name(run)} was still being saved by the last session when the app started: it is removed once that save ends.`);
  if (older) lines.push(older === 1 ? 'A run from an earlier version was removed: it was saved already or had nothing to save.' :
    `${older} runs from earlier versions were removed: they were saved already or had nothing to save.`);
  return lines.filter(Boolean).join(' ');
}

/** The window's ✕: while a run is pending the browser holds the window and the prompt asks; the window going anyway
 *  sends the answer given, else "discard" (nobody answered), with the runs it is for and a revision. */
export function watchClosing(win, popup, { sampling, hold = () => true, ask = runs => popup.show(runs[0], {reason: 'close'}), beacon }) {
  win.addEventListener('beforeunload', event => {
    if (popup.answered || !hold()) return;
    const runs = sampling(); if (!runs.length) return;
    ask(runs); event.preventDefault(); event.returnValue = '';
  });
  win.addEventListener('pagehide', () => {
    const runs = sampling();
    if (runs.length) beacon(popup.answered || 'discard', runs);
  });
}

export function closeWindow(win, stuck) {
  win.close();
  setTimeout(() => { if (!win.closed) stuck?.(); }, CLOSE_CHECK_MS);
}
