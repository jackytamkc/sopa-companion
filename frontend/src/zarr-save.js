import { api } from './api.js';
import { saveFields, saveModalState, shouldPromptSave, saveCompleted, saveable, defaultSegmentation } from './zarr-save-state.mjs';
import { STAIN_TEXT } from './stain.mjs';
import { guard, busy, clearExplanation, explain, problem } from './explain.mjs';
import { SESSION_TEXT, addWords, addLine } from './session.mjs';
const $ = id => document.getElementById(id);

export class ZarrSave {
  /** onRun(run): a run the save answered with; notice(text); onSaved(run): a save seen on its way has ended saved (its
   *  run's sample now holds what it wrote); whenSettled(): the cell changes settled before a save; onCancel(run): Not now
   *  (or Esc) closed the dialog with no save on its way (the page takes a Finish & save back). */
  constructor(onRun, notice, onSaved = null, whenSettled = null, onCancel = null) {
    this.whenSettled = whenSettled; this.onRun = onRun; this.notice = notice; this.onSaved = onSaved; this.onCancel = onCancel;
    this.finished = new Set(); this.pending = new Map(); this.queue = new Map();
    this.dialog = document.createElement('dialog'); this.dialog.className = 'modal'; this.dialog.setAttribute('aria-labelledby', 'zarr-title');
    this.dialog.innerHTML = `<form id="zarr-form"><h2 id="zarr-title"></h2>
      <label class="modal-field">Cell shapes layer<input id="zarr-shapes" type="text" autocomplete="off" spellcheck="false" aria-describedby="zarr-shapes-error"><span id="zarr-shapes-error" class="error"></span></label>
      <label class="modal-field">Table<input id="zarr-table" type="text" autocomplete="off" spellcheck="false" aria-describedby="zarr-table-error"><span id="zarr-table-error" class="error"></span></label>
      <label class="row toggle"><input id="zarr-default" type="checkbox" checked> <span id="zarr-default-text">Make it the sample's segmentation (as sopa does)</span></label>
      <p id="zarr-help" class="muted small"></p>
      <p id="zarr-warning" class="save-warning" hidden></p>
      <label id="zarr-overwrite-row" class="row toggle" hidden><input id="zarr-overwrite" type="checkbox"> Replace the existing layer</label>
      <p id="zarr-progress" class="modal-progress" role="status" hidden><span class="spinner"></span><span></span></p>
      <p id="zarr-error" class="error" role="alert" hidden></p>
      <div id="zarr-note" class="missing-note" role="alert" hidden></div>
      <div class="modal-actions"><button id="zarr-cancel" type="button">Not now</button><button id="zarr-submit" type="submit" class="primary">Save to zarr</button></div></form>`;
    document.body.append(this.dialog);
    this.note = { note: $('zarr-note') };
    $('zarr-cancel').onclick = () => { this.cancelled(); this.dialog.close(); };
    this.dialog.addEventListener('cancel', () => this.cancelled());      // Esc is Not now
    // Save is never greyed out: a press says which field to fix and why; saving keeps it busy. Enter in a name presses
    // it too (the browser clicks the form's submit button); the form's own submit only keeps the page where it is.
    $('zarr-form').onsubmit = e => e.preventDefault();
    guard($('zarr-submit'), () => this.whenSettled ? [] : this.problems(), () => this.submit(), this.note);
    // What the user changed by hand stays when the sample's names come back from the engine (ask).
    $('zarr-form').oninput = e => {
      this.error = ''; clearExplanation($('zarr-submit'), this.note);
      if (['zarr-shapes', 'zarr-table'].includes(e.target.id)) { $('zarr-overwrite').checked = false; (this.typed ??= new Set()).add(e.target.id); }
      if (e.target.id === 'zarr-default') this.ticked = true;
      this.render();
    };
    document.addEventListener('close', () => queueMicrotask(() => this.next()), true);
  }

  /** A notebook's session (session.mjs): Add to your SpatialData, in its words; null on the desktop. */
  setSession(session) { this.session = session || null; if (this.dialog.open && this.run) { this.words(this.run); this.render(); } }
  /** The dialog's words for `run`: the sample's (desktop) or your SpatialData's (a notebook's session). */
  words(run) {
    const api = this.session ? addWords(this.session, run) : null;
    $('zarr-title').textContent = api ? api.title : `Save to ${run.sample_name}?`;
    $('zarr-help').textContent = api ? api.help : run.kind === 'stain' ? STAIN_TEXT.saveHelp :
      "Written with sopa: shapes + an AnnData table of proseg's counts, linked to the shapes";
    $('zarr-submit').textContent = api ? api.submit : run.kind === 'stain' ? STAIN_TEXT.save : 'Save to zarr';
    $('zarr-default-text').textContent = api ? api.tick : "Make it the sample's segmentation (as sopa does)";
  }
  trackFinish(id) { this.finished.add(id); }
  cancelFinish(id) { this.finished.delete(id); this.queue.delete(id); }
  /** Not now (or Esc): with no save on its way, the page hears of it (onCancel). */
  cancelled() {
    const run = this.run;
    if (!run || this.submitting || this.pending.has(run.id) || run.zarr_save?.state === 'saving') return;
    try { this.onCancel?.(run); } catch { /* the page's follow-up only */ }
  }

  update(run) {
    const save = run.zarr_save;
    if (save?.state === 'saving' && !this.pending.has(run.id)) this.pending.set(run.id, null);
    const completed = this.pending.has(run.id) && saveCompleted(save, this.pending.get(run.id)) && this.pending.delete(run.id);
    if (this.run?.id === run.id) { this.run = run; if (completed) this.submitting = false; if (this.dialog.open) this.render(); }
    if (completed && ['saved', 'queued'].includes(save.state)) {
      if (this.dialog.open && this.run?.id === run.id) this.dialog.close();
      // A notebook's session: in the object's zarr (or ready) until the notebook is free, then added (session.mjs).
      this.notice(this.session ? addLine(this.session, run) || SESSION_TEXT.added.replace('{cells}', new Intl.NumberFormat('en-GB').format(save.cells || 0))
        .replace('{names}', [save.shapes_key, save.table_key].filter(Boolean).join(' + ')) :
        `Saved ${new Intl.NumberFormat('en-GB').format(save.cells)} cells to ${run.sample_name} as ${save.shapes_key}${save.table_key ? ` + ${save.table_key}` : ''}`);
      try { this.onSaved?.(run); } catch { /* the page's follow-up only */ }
    }
    if (shouldPromptSave(run, this.finished.has(run.id))) this.queue.set(run.id, run);
    if (['done', 'failed', 'stopped'].includes(run.state)) this.finished.delete(run.id);
    if (save) this.queue.delete(run.id);
    this.next();
  }

  next() {
    if (document.querySelector('dialog[open]') || !this.queue.size) return;
    const [id, run] = this.queue.entries().next().value; this.queue.delete(id); this.open(run);
  }

  open(run) {
    if (!saveable(run)) return;
    this.opened = (this.opened || 0) + 1;
    this.queue.delete(run.id); this.run = run; this.error = ''; this.submitting = false;
    this.known = null; this.typed = new Set(); this.ticked = false;      // the name fields typed in, by id
    this.done?.(); this.done = null; clearExplanation($('zarr-submit'), this.note);
    const fields = saveFields(run);
    this.words(run);
    $('zarr-shapes').value = fields.shapes_key; $('zarr-table').value = fields.table_key;
    $('zarr-table').parentElement.hidden = run.kind === 'stain';
    $('zarr-default').checked = fields.set_default; $('zarr-overwrite').checked = false;
    this.render(); if (!this.dialog.open) this.dialog.showModal();
    this.ask(run);
  }

  /** The sample's names as they are now: a save into the sample since this run ended (another run's, a
   *  stain run's) may have taken the names the run offers. The engine works the run's save defaults out again when it is
   *  asked for the run: each name field not typed in meanwhile takes its new name (one typed in keeps the user's, and
   *  the other its new one, not the stale one that may clash) and the tick its default, unless changed, and every
   *  name in the sample counts for a clash from then on (Replace is offered). A run saved before keeps its own names:
   *  saving it again replaces them. */
  async ask(run) {
    const ticket = this.asking = (this.asking || 0) + 1;
    let fresh = null;
    try { fresh = await api.run(run.id); } catch { /* the names as the run ended stand; a clash still answers 409 */ }
    const defaults = fresh?.save_defaults;
    if (ticket !== this.asking || this.run?.id !== run.id || !this.dialog.open || !defaults) return;
    this.known = { id: run.id, existing: [...(defaults.existing || [])] };
    if (!this.run.zarr_save && !this.view().saving) {
      if (!this.typed.has('zarr-shapes')) $('zarr-shapes').value = defaults.shapes_key || '';
      if (!this.typed.has('zarr-table')) $('zarr-table').value = defaults.table_key || '';
      if (!this.ticked) $('zarr-default').checked = defaultSegmentation({ ...this.run, save_defaults: defaults });
    }
    this.render();
  }

  fields() {
    return { shapes_key: $('zarr-shapes').value, table_key: this.run?.kind === 'stain' ? null : $('zarr-table').value,
      set_default: $('zarr-default').checked, overwrite: $('zarr-overwrite').checked };
  }

  /** The dialog as it stands (zarr-save-state.mjs), with the sample's names as last heard (ask, or a refused save). */
  view() {
    return saveModalState(this.run, this.fields(), { submitting: this.submitting || this.pending.has(this.run.id), error: this.error,
      existing: this.known?.id === this.run.id ? this.known.existing : null, api: !!this.session });
  }

  /** What stops a save now, by field label (explain.mjs): names to fix, a layer to replace, or a run still going. */
  problems() {
    const run = this.run, view = this.view();
    if (!saveable(run)) return [problem(null, '', run.kind === 'stain' ? 'Stop the run or let it finish first.' : 'Let the run finish first.')];
    // A stain run with no cells left (none found, or all removed): the engine refuses it (409); the press says so first.
    if (run.kind === 'stain' && run.summary?.cells === 0) return [problem(null, '', 'There are no cells to save: this run found none, or all were removed.')];
    const problems = [['zarr-shapes', 'shapes_key', 'Cell shapes layer'], ['zarr-table', 'table_key', 'Table']]
      .filter(([, key]) => view.errors[key]).map(([id, key, label]) => problem(id, label, view.errors[key]));
    if (view.conflicts.length && !$('zarr-overwrite').checked)
      problems.push(problem('zarr-overwrite', 'Replace the existing layer', `Tick it to replace ${view.conflicts.join(', ')}, or use a different name.`));
    return problems;
  }

  render() {
    const view = this.view();
    for (const [id, key] of [['zarr-shapes', 'shapes_key'], ['zarr-table', 'table_key']]) {
      $(id).disabled = view.saving; $(id).setAttribute('aria-invalid', String(!!view.errors[key]));
      $(id + '-error').textContent = view.errors[key] || '';
    }
    $('zarr-default').disabled = $('zarr-overwrite').disabled = view.saving;
    if (view.saving !== !!this.done) {
      if (view.saving) this.done = busy($('zarr-submit'), this.session ? SESSION_TEXT.adding : 'Saving…');
      else { this.done(); this.done = null; clearExplanation($('zarr-submit'), this.note); }
    }
    $('zarr-overwrite-row').hidden = $('zarr-warning').hidden = !view.conflicts.length;
    $('zarr-warning').textContent = this.session ? SESSION_TEXT.clash.replace('{names}', view.conflicts.join(', ')).replace('{them}', view.conflicts.length > 1 ? 'these layers' : 'this layer') :
      `Already in the sample: ${view.conflicts.join(', ')}. Saving will replace ${view.conflicts.length > 1 ? 'these layers' : 'this layer'}.`;
    $('zarr-progress').hidden = !view.saving; $('zarr-progress').lastElementChild.textContent = view.message;
    $('zarr-error').hidden = !view.error; $('zarr-error').textContent = view.error;
    $('zarr-cancel').textContent = view.saving ? 'Close' : 'Not now';
  }

  async submit() {
    if (this.settling) return;
    if (this.whenSettled) {
      this.settling = true; const id = this.run.id, opened = this.opened, done = busy($('zarr-submit'), 'Waiting for the cell changes…');
      try { await this.whenSettled(); }
      catch (error) { this.error = error.message; this.render(); return; }
      finally { this.settling = false; done(); }
      if (this.run?.id !== id || this.opened !== opened || !this.dialog.open) return;
      const problems = this.problems();
      if (problems.length) { explain($('zarr-submit'), problems, this.note); return; }
    }
    const fields = this.fields(), id = this.run.id;
    if (!this.view().canSave) return;
    this.submitting = true; this.error = ''; this.pending.set(id, this.run.zarr_save); this.render();
    let clash = false;
    try {
      const run = await api.saveZarr(id, fields);
      if (this.pending.has(id)) this.onRun(run);
    } catch (error) {
      if (this.pending.delete(id) && this.run?.id === id) {
        this.error = error.message;
        // A name clash (409) names the names taken and every name in the sample: from now on they count as clashes.
        const data = error.status === 409 ? error.data : null;
        if (Array.isArray(data?.clash) || Array.isArray(data?.existing)) {
          const before = this.known?.id === id ? this.known.existing : this.run.save_defaults?.existing || [];
          this.known = { id, existing: [...new Set([...before, ...(data.existing || []), ...(data.clash || [])])] };
          clash = true;
        }
      }
    } finally {
      if (this.run?.id === id) { this.submitting = false; if (this.dialog.open) this.render(); }
    }
    // The dialog then says what a press says for a clash it knew of (Replace, or other names), in place of the refusal.
    if (clash && this.dialog.open && this.run?.id === id && this.view().conflicts.length) {
      this.error = ''; this.render(); explain($('zarr-submit'), this.problems(), this.note);
    }
  }
}
