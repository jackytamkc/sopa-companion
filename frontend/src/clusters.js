// Fixed, colour-blind-aware families with alternating luminance. At 40 categories,
// colour alone cannot identify every pair: always retain the text labels.
export const CLUSTER_PALETTE = Object.freeze([
  '#0072b2', '#e69f00', '#009e73', '#cc79a7', '#56b4e9', '#f0e442', '#d55e00', '#999999',
  '#332288', '#88ccee', '#44aa99', '#117733', '#999933', '#ddcc77', '#cc6677', '#aa4499',
  '#661100', '#6699cc', '#aa4466', '#4477aa', '#228833', '#ee6677', '#bbbbbb', '#66ccee',
  '#aa3377', '#eecc66', '#004488', '#bb5566', '#ddaadd', '#77aadd', '#99ddff', '#44bb99',
  '#bbcc33', '#aaaa00', '#eedd88', '#ee8866', '#ffaabb', '#663399', '#aaccaa', '#dddddd',
]);
export function clusterColorCSS(index) {
  return Number.isInteger(index) && index >= 0 ? CLUSTER_PALETTE[index % 40] : '#66717e';
}
/** RGB array in 0..1; use clusterColorCSS(index) for CSS. */
export function clusterColor(index) {
  const hex = clusterColorCSS(index);
  return [1, 3, 5].map(i => parseInt(hex.slice(i, i + 2), 16) / 255);
}

const SIZE = new Intl.NumberFormat();
// The legend's height holds still while a run plays: its rows have one fixed height, it shows at most LEGEND_ROWS of
// them (more scroll inside it), and within a run it only grows, to the most rows it has shown, never shrinks. The
// clusters come and go with nearly every result (9-14 rows several times a second on a small region), and the legend
// sits in the middle of Layers: everything below it (the prior layers, the image channels, Colour by) used to jump with
// it.
export const LEGEND_ROWS = 8, ROW_PX = 28;
/** Legend that lives across results: update(summary) rewrites rows in place (sizes, labels, colours) and adds or
 * removes rows only when the set of cluster ids changes, so hover and focus survive live updates; its height only grows
 * (LEGEND_ROWS, ROW_PX). Mutates hidden before invoking onToggle(id) for each changed id. highlight(id) marks the row of
 * a cluster hovered elsewhere (a cell on the tissue) touching only two rows. Returns {update, highlight, dispose}. */
export function createLegend(container, { onHover = () => {}, onToggle = () => {}, hidden = new Set() } = {}) {
  const root = document.createElement('section'), heading = document.createElement('div'), list = document.createElement('div'), rows = new Map();
  root.style.cssText = 'font:12px/1.4 system-ui;color:inherit';
  heading.style.cssText = 'font-weight:600;padding:8px 4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis';
  list.className = 'cluster-legend-rows';
  list.style.cssText = `overflow-y:auto;overscroll-behavior:contain;max-height:${LEGEND_ROWS * ROW_PX}px`;
  root.append(heading, list); container.replaceChildren(root);
  let clusters = [], order = null, hovered = null, marked = null, tallest = 0;
  const paint = id => { const row = rows.get(id); if (row) row.button.style.background = id === marked ? '#ffffff1f' : id === hovered ? '#ffffff12' : 'transparent'; };
  const refresh = () => {
    for (const [id, row] of rows) {
      const shown = !hidden.has(id);
      if (row.shown !== shown) { row.shown = shown; row.button.style.opacity = shown ? '1' : '.4'; row.button.setAttribute('aria-pressed', String(shown)); }
    }
  };
  const hover = (id, active) => {
    if (!active && hovered !== id) return;          // leaving a row that is no longer the hovered one
    const was = hovered; hovered = active ? id : null;
    if (was !== id) paint(was);
    paint(id); onHover(hovered);
  };
  const make = id => {
    const button = document.createElement('button'), swatch = document.createElement('span'), label = document.createElement('span'), size = document.createElement('span');
    button.type = 'button';
    button.dataset.cluster = String(id);
    button.style.cssText = `display:flex;align-items:center;gap:8px;width:100%;height:${ROW_PX}px;box-sizing:border-box;border:0;border-radius:4px;padding:0 4px;background:transparent;color:inherit;text-align:left;cursor:pointer;font:inherit`;
    swatch.style.cssText = 'flex:0 0 10px;height:10px;border-radius:50%';
    label.style.cssText = 'flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap';
    size.style.cssText = 'font-variant-numeric:tabular-nums;opacity:.65';
    button.append(swatch, label, size);
    button.onpointerenter = button.onfocus = () => hover(id, true);
    button.onpointerleave = button.onblur = () => hover(id, false);
    button.onclick = event => {
      const changed = [];
      const set = (c, hide) => { if (hidden.has(c) !== hide) { hide ? hidden.add(c) : hidden.delete(c); changed.push(c); } };
      if (event.altKey || event.shiftKey) {
        const solo = clusters.every(c => hidden.has(c.id) === (c.id !== id));
        for (const c of clusters) set(c.id, !solo && c.id !== id);
      } else set(id, !hidden.has(id));
      refresh();
      for (const c of changed) onToggle(c);
    };
    return { button, swatch, label, size, shown: null, color: null, text: null, count: null };
  };
  const update = summary => {
    clusters = summary?.clusters || [];
    const title = `Clusters (provisional, iteration ${summary?.iter ?? summary?.iteration ?? '—'})`;
    if (heading.textContent !== title) heading.textContent = title;
    const ids = clusters.map(c => c.id), key = ids.join(',');
    if (key !== order) {
      // The clusters changed: keep the rows that remain (with their hover state), drop or add the others.
      order = key; const keep = new Set(ids), added = [];
      for (const id of [...rows.keys()]) if (!keep.has(id)) {
        rows.delete(id); if (marked === id) marked = null;
        if (hovered === id) { hovered = null; onHover(null); }
      }
      for (const id of ids) if (!rows.has(id)) { rows.set(id, make(id)); added.push(id); }
      list.replaceChildren(...ids.map(id => rows.get(id).button));
      added.forEach(paint);
      const height = Math.min(LEGEND_ROWS, Math.max(tallest, ids.length));
      if (height !== tallest) { tallest = height; list.style.minHeight = `${height * ROW_PX}px`; }   // grows only
    }
    // Only values that changed touch the DOM (a live update usually changes just the sizes).
    for (const c of clusters) {
      const row = rows.get(c.id), color = clusterColorCSS(c.color ?? c.id), count = SIZE.format(Number(c.size || 0));
      const text = c.label || (c.top || []).slice(0, 3).map(g => g.gene).join(' · ') || `Cluster ${c.id}`;
      if (row.color !== color) { row.color = color; row.swatch.style.background = color; }
      if (row.text !== text) { row.text = text; row.label.textContent = text; row.button.title = `${text}; click to hide/show, Alt/Shift-click to solo`; }
      if (row.count !== count) { row.count = count; row.size.textContent = count; }
    }
    refresh();
  };
  const highlight = id => { if (id === marked) return; const was = marked; marked = id; paint(was); paint(id); };
  const dispose = () => { root.remove(); rows.clear(); if (hovered !== null) { hovered = null; onHover(null); } };
  return { update, highlight, dispose };
}
/** One-off legend for a single summary; returns cleanup. Live views keep a createLegend() and update it. */
export function renderLegend(container, summary, options) {
  const legend = createLegend(container, options); legend.update(summary);
  return legend.dispose;
}
