import { problem, guard, busy, explain, clearExplanation, EXPLAIN_TEXT, NEEDS_CLASS } from './explain.mjs';
import { renderNotes } from './sample-notes.mjs';

// Training patches' export dialog: every Export patches asks where this export's patches go, on the desktop and in a
// notebook. The folder chosen is a training folder: the Cellpose GUI trains one model on every patch in the folder it has
// open, so the patches go straight into it, beside what is there, under names of their own (engine/patch_folders.py).
// The engine suggests the folder (the one this sample's patches at this resolution went to last, else a new one named
// after the sample and its resolution), lists folders as the sample browser does, and says why one may not take patches;
// the page keeps nothing of it. Nothing is written before Export here, and no element here is ever disabled: a press
// that cannot go ahead says why (explain.mjs). Closing the dialog never stops an export: its result goes to the panel.
export const PATCH_FOLDER_TEXT = Object.freeze({
  title:'Export training patches',
  lead:'Choose a folder for these patches. The Cellpose GUI trains one model on all the patches in a folder.',
  leadTip:"Cellpose's Models > Train new model learns from every patch in the folder it has open, corrected or not. Keep one folder per model: a sample that needs a model of its own gets a folder of its own. Patches already in a folder are never replaced.",
  parent:'Parent folder (Backspace)', parentLabel:'Parent folder', path:'Folder for the patches', field:'Folder',
  newFolder:'New folder', newFolderTip:'A folder named after the sample and its resolution, inside the one shown. It is made when you export.',
  list:'Folders', holdsBadge:'Patches', none:'No folders here.', loading:'Loading folders…',
  exportingLine:'Exporting the patches… With model outlines this can take a few minutes. You can close this; the export goes on.',
  // How far an export with the model's outlines is (the panel asks while it runs: GET .../stain/patches "exporting").
  exportingPatch:'Exporting patch {done} of {total}… You can close this; the export goes on.',
  // The model was not on this computer yet ("exporting" downloading: true, until the first patch is written).
  exportingDownload:'Downloading the Cellpose model first (only needed once)… You can close this; the export goes on.',
  newHere:'A new folder: it is made when you export.',
  holds:'Holds {n} patches of {samples}.', holdsOne:'Holds 1 patch of {samples}.', more:'{names} and {n} more',
  mixResolution:'Patches cut at another resolution are here too: the Cellpose GUI will train one model on both.',
  mixChannels:'Patches of other channels are here too: the Cellpose GUI will train one model on both.',
  choose:'Choose a folder or type its path.',
  home:'Home', earlier:'Earlier: {name}',
  cancel:'Cancel', close:'Close', export:'Export here', exporting:'Exporting…',
  failed:'The folders could not be listed. Try again.',
});

const number = n => Number(n).toLocaleString('en-GB');
/** The samples a folder holds patches of: up to three names, then how many more. */
export function holdsLine(holds) {
  if (!holds?.patches) return '';
  const T = PATCH_FOLDER_TEXT, names = (holds.samples || []).join(', ');
  const samples = holds.more > 0 ? T.more.replace('{names}', names).replace('{n}', number(holds.more)) : names;
  return (holds.patches === 1 ? T.holdsOne : T.holds).replace('{n}', number(holds.patches)).replace('{samples}', samples);
}

export class PatchFolderDialog {
  /** `notice(words)`: the page's notice, for what an export says once the dialog is closed. */
  constructor(api, {notice} = {}) {
    Object.assign(this, {api, notice});
    this.ticket = 0; this.exportTicket = 0; this.selected = -1;
    this.listing = null; this.loading = false; this.exporting = false; this.failed = '';
    this.run = null; this.version = null; this.exportTo = null; this.words = {};
    const T = PATCH_FOLDER_TEXT;
    const make = (tag, id, className, text) => {
      const node = document.createElement(tag);
      if (id) node.id = id;
      if (className) node.className = className;
      if (text) node.textContent = text;
      if (tag === 'button') node.type = 'button';
      return node;
    };
    this.dialog = make('dialog', 'patch-folder', 'modal'); this.dialog.setAttribute('aria-labelledby', 'patch-folder-title');
    const title = make('h2', 'patch-folder-title', null, T.title);
    this.lead = make('div', 'patch-folder-lead');
    this.shortcuts = make('div', 'patch-folder-shortcuts', 'browse-shortcuts');
    this.parent = make('button', 'patch-folder-parent', null, '↑');
    this.parent.title = T.parent; this.parent.setAttribute('aria-label', T.parentLabel);
    this.path = make('input', 'patch-folder-path'); this.path.type = 'text';
    this.path.setAttribute('aria-label', T.path); this.path.setAttribute('placeholder', T.path);
    this.path.setAttribute('autocomplete', 'off'); this.path.setAttribute('spellcheck', 'false');
    this.newFolder = make('button', 'patch-folder-new', null, T.newFolder); this.newFolder.title = T.newFolderTip;
    const row = make('div', null, 'browse-path'); row.append(this.parent, this.path, this.newFolder);
    this.list = make('div', 'patch-folder-list', 'browse-list');
    this.list.setAttribute('role', 'listbox'); this.list.setAttribute('aria-label', T.list); this.list.setAttribute('tabindex', '0');
    this.status = make('p', 'patch-folder-status'); this.status.setAttribute('role', 'status');
    this.mix = make('p', 'patch-folder-mix', 'save-warning');
    this.error = make('p', 'patch-folder-error', 'error'); this.error.setAttribute('role', 'alert');
    this.progress = make('p', 'patch-folder-progress', 'modal-progress'); this.progress.setAttribute('role', 'status');
    this.progressText = make('span');
    this.progress.append(make('span', null, 'spinner'), this.progressText);
    this.note = make('div', 'patch-folder-note', 'missing-note'); this.note.setAttribute('role', 'alert'); this.note.hidden = true;
    this.cancel = make('button', 'patch-folder-cancel', null, T.cancel);
    this.exportButton = make('button', 'patch-folder-export', 'primary', T.export);
    const actions = make('div', null, 'modal-actions'); actions.append(this.cancel, this.exportButton);
    this.dialog.append(title, this.lead, this.shortcuts, row, this.list, this.status, this.mix, this.error, this.progress,
      this.note, actions);
    document.body.append(this.dialog);
    renderNotes(this.lead, [{code: 'lead', text: T.lead, detail: T.leadTip}]);
    this.events();
    this.render();
  }

  events() {
    const T = PATCH_FOLDER_TEXT;
    this.cancel.onclick = () => this.close();
    this.parent.onclick = () => this.go(this.listing?.parent);
    // New folder: a free name inside the folder shown, listed (made only by the export).
    this.newFolder.onclick = () => {
      const listing = this.listing;
      if (listing?.new) this.go(listing.path === '/' ? `/${listing.new}` : `${listing.path}/${listing.new}`);
      else if (this.exporting) this.go(null);
      else this.explainField(listing?.refused || T.choose);
    };
    // Export here is never greyed out: a press without a folder, or on the folder shown when it is refused, says why;
    // while a listing or the export runs it is busy, and a press says the last step is still running.
    guard(this.exportButton, () => this.problems(), () => this.exportHere(), {note: this.note});
    this.path.oninput = () => {
      if (this.loading) { this.ticket++; this.controller?.abort(); this.loading = false; }   // what is typed stays
      this.failed = ''; this.unmark(); this.render();
    };
    // Closing (Close, Escape) never stops an export; a listing on its way is left unread.
    this.dialog.addEventListener('close', () => { this.ticket++; this.controller?.abort(); this.loading = false; this.render(); });
    this.dialog.addEventListener('keydown', e => this.keydown(e));
  }

  /** Open on the run's suggestion. `version`: the boxes' version at this press, sent with every Export here;
   *  `exportTo(folder, version)`: the panel's export; `words`: {changed, failed}, its notices. */
  show({run, version, exportTo, words}) {
    if (this.dialog.open) return;
    Object.assign(this, {run, version, exportTo, words: words || {}});
    this.listing = null; this.selected = -1; this.failed = ''; this.path.value = ''; this.unmark();
    clearExplanation(this.exportButton, {note: this.note});
    this.dialog.showModal();
    this.navigate();
    this.path.focus();
  }

  /** `silent` (the panel's reset): an export's or a listing's late answer shows nothing. */
  close(silent = false) {
    if (silent) { this.exportTicket++; this.exporting = false; }
    this.ticket++; this.controller?.abort(); this.loading = false;
    if (this.dialog.open) this.dialog.close();
    this.render();
  }

  /** A folder pressed, typed or chosen: listed, unless the export runs (then the press says so and asks nothing). */
  go(path) {
    if (this.exporting) { explain(this.exportButton, [problem(null, '', EXPLAIN_TEXT.busy)], {note: this.note}); return; }
    if (path) this.navigate(path);
  }

  /** List `path` (none: the suggestion); a newer listing, or a close, leaves this one unread. */
  async navigate(path) {
    if (!this.run) return;
    const ticket = ++this.ticket;
    this.controller?.abort(); this.controller = new AbortController();
    this.loading = true; this.selected = -1; this.failed = ''; this.render();
    try {
      const listing = await this.api.stainPatchFolders(this.run.id, path, this.controller.signal);
      if (ticket !== this.ticket || !this.dialog.open) return;
      this.listing = listing; this.path.value = listing.path; this.unmark();
    } catch (e) {
      if (ticket === this.ticket && e?.name !== 'AbortError') this.failed = PATCH_FOLDER_TEXT.failed;
    } finally {
      if (ticket === this.ticket) { this.loading = false; this.render(); }
    }
  }

  /** The folder in the field: the folder listed, as the engine gave its real path, while the field still shows it (a
   *  folder's name may end in a space); else what is typed, without the spaces around it. */
  folder() {
    const value = this.path.value;
    return this.listing && value === this.listing.path ? value : value.trim();
  }

  problems() {
    const T = PATCH_FOLDER_TEXT, value = this.folder();
    if (!value) return [problem(this.path.id, T.field, T.choose)];
    if (this.listing?.refused && value === this.listing.path) return [problem(this.path.id, T.field, this.listing.refused)];
    return [];
  }

  explainField(words) { explain(this.exportButton, [problem(this.path.id, PATCH_FOLDER_TEXT.field, words)], {note: this.note}); }

  /** The export into the folder in the field, with the boxes' version from show(). A refusal of the folder keeps the
   *  dialog open with its words (the field outlined); anything else closes it and the panel's notice says it. */
  async exportHere() {
    const folder = this.folder(), ticket = ++this.exportTicket, words = this.words;
    this.exporting = true; this.exportProgress = null; this.failed = ''; this.unmark(); this.render();
    try {
      await this.exportTo(folder, this.version, progress => {
        if (ticket === this.exportTicket) { this.exportProgress = progress; this.render(); }
      });
      if (ticket === this.exportTicket && this.dialog.open) this.dialog.close();
    } catch (e) {
      if (ticket !== this.exportTicket) return;
      if (this.dialog.open && e?.data?.field === 'folder') {
        this.failed = e.message || words.failed || ''; this.mark();
      } else {
        if (this.dialog.open) this.dialog.close();
        this.notice?.(e?.data?.reason === 'changed' ? words.changed : e?.message || words.failed);
      }
    } finally {
      if (ticket === this.exportTicket) { this.exporting = false; this.exportProgress = null; this.render(); }
    }
  }

  mark() { this.path.classList.add(NEEDS_CLASS); this.path.setAttribute('aria-invalid', 'true'); }
  unmark() { this.path.classList.remove(NEEDS_CLASS); this.path.removeAttribute('aria-invalid'); }

  select(index, focus = false) {
    this.selected = index;
    for (const [i, row] of [...this.list.children].entries()) {
      row.setAttribute('aria-selected', String(i === index)); row.tabIndex = i === index ? 0 : -1;
    }
    const row = this.list.children[index];
    if (focus) row?.focus();
    row?.scrollIntoView?.({block: 'nearest'});
  }

  // As the sample browser: Enter in the field lists it, ↑ and ↓ choose a row, Enter on a row enters it, Backspace
  // outside the field goes up; Escape closes (the dialog's own).
  keydown(e) {
    if (e.key === 'Escape') return;
    const input = e.target === this.path, inList = this.list.contains(e.target);
    if (e.key === 'Enter' && input) {
      e.preventDefault();
      const path = this.folder();
      if (path || this.exporting) this.go(path || null); else this.explainField(PATCH_FOLDER_TEXT.choose);
    } else if (['ArrowUp', 'ArrowDown'].includes(e.key) && (input || inList)) {
      e.preventDefault();
      const n = this.listing?.entries?.length || 0;
      if (n) this.select(Math.max(0, Math.min(n - 1, this.selected + (e.key === 'ArrowDown' ? 1 : -1))), true);
    } else if (e.key === 'Enter' && inList) {
      e.preventDefault();
      const entry = this.listing?.entries?.[this.selected];
      if (entry || this.exporting) this.go(entry?.path);
    } else if (e.key === 'Backspace' && !input) {
      e.preventDefault();
      if (this.listing?.parent || this.exporting) this.go(this.listing?.parent);
    }
  }

  render() {
    const T = PATCH_FOLDER_TEXT, listing = this.listing, working = this.loading || this.exporting;
    const label = this.exporting ? T.exporting : this.loading ? '' : null;
    if (label !== this.busyLabel) {          // '': busy while a folder lists, its label unchanged
      this.done?.(); this.done = label === null ? null : busy(this.exportButton, label); this.busyLabel = label;
      if (label === null) clearExplanation(this.exportButton, {note: this.note});
    }
    this.cancel.textContent = this.exporting ? T.close : T.cancel;
    this.parent.hidden = !listing?.parent;
    this.progress.hidden = !working;
    const progress = this.exportProgress;              // once a patch is written, how many of how many
    this.progressText.textContent = !this.exporting ? T.loading : progress?.done >= 1 && progress.total >= progress.done ?
      T.exportingPatch.replace('{done}', progress.done).replace('{total}', progress.total) :
      progress?.downloading ? T.exportingDownload : T.exportingLine;
    this.shortcuts.replaceChildren(...(listing?.shortcuts || []).map(shortcut => {
      const button = document.createElement('button'); button.type = 'button';
      button.textContent = shortcut.kind === 'home' ? T.home : T.earlier.replace('{name}', shortcut.name);
      button.title = shortcut.path; button.onclick = () => this.go(shortcut.path);
      return button;
    }));
    this.list.replaceChildren(); this.list.setAttribute('aria-busy', String(this.loading));
    for (const [index, entry] of (listing?.entries || []).entries()) {
      const row = document.createElement('button'); row.type = 'button'; row.className = 'browse-entry';
      row.setAttribute('role', 'option'); row.setAttribute('aria-selected', String(index === this.selected));
      row.tabIndex = index === Math.max(0, this.selected) ? 0 : -1; row.title = entry.path;
      const name = document.createElement('span'); name.textContent = entry.name; row.append(name);
      if (entry.patches) { const badge = document.createElement('span'); badge.className = 'badge'; badge.textContent = T.holdsBadge; row.append(badge); }
      const arrow = document.createElement('span'); arrow.className = 'muted'; arrow.textContent = '›'; row.append(arrow);
      row.onclick = () => this.go(entry.path); row.onfocus = () => this.select(index);
      this.list.append(row);
    }
    if (!this.loading && !this.list.children.length) {
      const none = document.createElement('p'); none.className = 'muted'; none.textContent = T.none; this.list.append(none);
    }
    const status = !listing ? '' : listing.exists === false ? (listing.refused ? '' : T.newHere) : holdsLine(listing.holds);
    this.status.textContent = status; this.status.hidden = !status;
    const mix = listing?.mixes === 'resolution' ? T.mixResolution : listing?.mixes === 'channels' ? T.mixChannels : '';
    this.mix.textContent = mix; this.mix.hidden = !mix;
    const error = this.failed || listing?.refused || '';
    this.error.textContent = error; this.error.hidden = !error;
  }

  /** For the harness: {open, loading, exporting, path, exists, refused, mixes} of the folder shown. */
  snapshot() {
    const listing = this.listing;
    return {open: !!this.dialog.open, loading: this.loading, exporting: this.exporting, path: listing?.path ?? null,
      exists: listing?.exists ?? null, refused: listing?.refused ?? null, mixes: listing?.mixes ?? null};
  }
}
