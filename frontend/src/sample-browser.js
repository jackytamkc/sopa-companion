import { api } from './api.js';
import { guard, busy, clearExplanation, problem } from './explain.mjs';
import { openFailure } from './sample-notes.mjs';
const $ = id => document.getElementById(id);

export class SampleBrowser {
  constructor(onOpen) {
    this.onOpen = onOpen; this.ticket = 0; this.selected = -1;
    this.dialog = document.createElement('dialog'); this.dialog.className = 'modal';
    this.dialog.setAttribute('aria-labelledby', 'browse-title');
    this.dialog.innerHTML = `<h2 id="browse-title">Open a SpatialData sample</h2>
      <div id="browse-shortcuts" class="browse-shortcuts"></div>
      <div class="browse-path"><button id="browse-parent" type="button" title="Parent folder (Backspace)" aria-label="Parent folder">↑</button><input id="browse-path" type="text" aria-label="Folder or .zarr path" placeholder="Folder or .zarr path" autocomplete="off" spellcheck="false"></div>
      <div id="browse-list" class="browse-list" role="listbox" aria-label="Folders and SpatialData samples" tabindex="0"></div>
      <p id="browse-progress" class="modal-progress" role="status" hidden><span class="spinner"></span><span></span></p>
      <p id="browse-error" class="error" role="alert" hidden></p>
      <div id="browse-note" class="missing-note" role="alert" hidden></div>
      <div class="modal-actions"><button id="browse-cancel" type="button">Cancel</button><button id="browse-open" class="primary" type="button">Open</button></div>`;
    document.body.append(this.dialog);
    this.note = { note: $('browse-note') };
    $('browse-cancel').onclick = () => this.dialog.close();
    $('browse-parent').onclick = () => this.navigate(this.listing?.parent);
    // Open is never greyed out: a press with nothing chosen says so; listing or opening keeps it busy.
    guard($('browse-open'), () => this.listing?.entries[this.selected] || $('browse-path').value.trim() ? [] :
      [problem('browse-path', '', 'Choose a folder or type a path.')], () => this.openPath(), this.note);
    $('browse-path').oninput = () => {
      if (this.loading) { this.ticket++; this.loading = false; }
      this.selected = -1; this.error(''); this.render();
    };
    this.dialog.addEventListener('close', () => { this.ticket++; this.opening = this.loading = false; });
    this.dialog.addEventListener('keydown', e => this.keydown(e));
  }

  show(path) {
    if (this.dialog.open) return;
    this.listing = null; this.selected = -1; this.opening = this.loading = false;
    $('browse-path').value = path || ''; this.error(''); this.render();
    this.dialog.showModal();
    if (path) this.openPath(path); else this.navigate();
    $('browse-path').focus();
  }

  /** `message`: an error (said as openFailure says it), words, or '' to clear. */
  error(message) {
    clearExplanation($('browse-open'), this.note);
    const { text, detail } = message && typeof message === 'object' ? openFailure(message) : { text: message || '', detail: '' };
    $('browse-error').textContent = text; $('browse-error').title = detail; $('browse-error').hidden = !text;
  }

  async navigate(path) {
    if (this.opening) return;
    const ticket = ++this.ticket;
    this.loading = true; this.selected = -1; this.error(''); this.render();
    try {
      const listing = await api.browse(path);
      if (ticket !== this.ticket || !this.dialog.open) return;
      this.listing = listing; $('browse-path').value = listing.path;
    } catch (error) {
      if (ticket === this.ticket) this.error(error);
    } finally {
      if (ticket === this.ticket) { this.loading = false; this.render(); }
    }
  }

  async openPath(path) {
    if (this.opening) return;
    const entry = this.listing?.entries[this.selected];
    if (!path && entry?.kind === 'dir') { this.navigate(entry.path); return; }
    path ||= entry?.kind === 'zarr' ? entry.path : $('browse-path').value.trim();
    if (!path) { this.error('Choose a folder or type a path.'); return; }
    const ticket = ++this.ticket;
    this.opening = true; this.loading = false; this.error(''); this.render();
    try {
      const sample = await api.openSample(path);
      if (ticket !== this.ticket || !this.dialog.open) return;
      this.dialog.close(); this.onOpen(sample);
    } catch (error) {
      if (ticket === this.ticket) this.error(error);   // one plain line (openFailure), the whole reason in its tooltip
    } finally {
      if (ticket === this.ticket) { this.opening = false; this.render(); }
    }
  }

  select(index, focus = false) {
    this.selected = index; clearExplanation($('browse-open'), this.note);
    for (const [i, row] of [...$('browse-list').children].entries()) {
      row.setAttribute('aria-selected', String(i === index)); row.tabIndex = i === index ? 0 : -1;
    }
    const row = $('browse-list').children[index];
    if (focus) row?.focus();
    row?.scrollIntoView({ block: 'nearest' });
  }

  activate(index) {
    const entry = this.listing?.entries[index];
    if (!entry || this.loading || this.opening) return;
    if (entry.kind === 'dir') this.navigate(entry.path);
    else { this.selected = index; this.openPath(entry.path); }
  }

  keydown(e) {
    if (e.key === 'Escape') return;
    if (this.opening) return;
    const input = e.target === $('browse-path'), inList = $('browse-list').contains(e.target);
    if (e.key === 'Enter' && input) {
      e.preventDefault();
      const path = $('browse-path').value.trim();
      if (/\.zarr\/?$/i.test(path)) this.openPath(path); else this.navigate(path);
    } else if (['ArrowUp', 'ArrowDown'].includes(e.key) && (input || inList)) {
      e.preventDefault();
      const n = this.listing?.entries.length || 0;
      if (n && !this.loading) this.select(Math.max(0, Math.min(n - 1, this.selected + (e.key === 'ArrowDown' ? 1 : -1))), true);
    } else if (e.key === 'Enter' && inList) { e.preventDefault(); this.activate(this.selected); }
    else if (e.key === 'Backspace' && !input) { e.preventDefault(); if (this.listing?.parent) this.navigate(this.listing.parent); }
  }

  render() {
    const working = this.loading || this.opening, label = this.opening ? 'Opening…' : working ? '' : null;
    if (label !== this.busyLabel) {          // '' : busy while a folder lists, its label unchanged
      this.done?.(); this.done = label === null ? null : busy($('browse-open'), label); this.busyLabel = label;
      if (label === null) clearExplanation($('browse-open'), this.note);
    }
    $('browse-path').disabled = !!this.opening;
    $('browse-parent').disabled = working || !this.listing?.parent;
    const progress = $('browse-progress'); progress.hidden = !working;
    progress.lastElementChild.textContent = this.opening ? 'Opening sample… Reading transcripts, shapes and image.' : 'Loading folders…';
    const shortcuts = $('browse-shortcuts'); shortcuts.replaceChildren();
    for (const shortcut of this.listing?.shortcuts || []) {
      const button = document.createElement('button'); button.type = 'button'; button.textContent = shortcut.label;
      button.title = shortcut.path; button.disabled = !!this.opening; button.onclick = () => this.navigate(shortcut.path); shortcuts.append(button);
    }
    const list = $('browse-list'); list.replaceChildren(); list.setAttribute('aria-busy', String(!!working));
    for (const [index, entry] of (this.listing?.entries || []).entries()) {
      const row = document.createElement('button'); row.type = 'button'; row.className = 'browse-entry'; row.disabled = !!working;
      row.setAttribute('role', 'option'); row.setAttribute('aria-selected', String(index === this.selected)); row.tabIndex = index === Math.max(0, this.selected) ? 0 : -1;
      const name = document.createElement('span'); name.textContent = entry.name; row.append(name); row.title = entry.path;
      if (entry.kind === 'zarr') { const badge = document.createElement('span'); badge.className = 'badge'; badge.textContent = 'SpatialData'; row.append(badge); }
      else { const arrow = document.createElement('span'); arrow.textContent = '›'; arrow.className = 'muted'; row.append(arrow); }
      row.onclick = () => entry.kind === 'dir' ? this.navigate(entry.path) : this.select(index, true);
      row.onfocus = () => this.select(index);
      row.ondblclick = () => this.activate(index); list.append(row);
    }
    if (!working && !list.children.length) { const note = document.createElement('p'); note.className = 'muted'; note.textContent = 'No folders here. You can enter a .zarr path above.'; list.append(note); }
  }
}
