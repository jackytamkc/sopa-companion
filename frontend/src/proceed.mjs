import { saveOutdated } from './zarr-save-state.mjs';

export const PROCEED_TEXT = Object.freeze({
  button: 'Proceed to proseg',
  tooltip: 'Opens the proseg setup with these cells as its starting cells, over the same region.',
  partialTooltip: 'Opens the proseg setup with these cells as its starting cells, over the part the stain run segmented.',
  outdated: 'Save as shapes again first: the cells changed since you saved them.',
  outdatedApi: 'Add as shapes again first: the cells changed since you added them.',
  waiting: 'Wait for the cell changes to finish.',
  missing: 'The saved cells are not offered yet. Try again in a moment.',
  small: 'Too little was segmented to start proseg from it.',
  noTranscripts: 'This sample has no transcripts for proseg.',
  limited: 'Limited to the part the stain run segmented.',
  // A notebook's session: the cells were added, but not where proseg reads its starting cells (the sample's zarr).
  memoryOnly: 'Proseg starts from cells in the zarr, and these are in memory only.',
  notInLayers: "Proseg starts from cells in the app's layers, and these could not be added to them.",
});

/** In a notebook's session, a save is in the zarr the app reads (where the proseg setup finds its starting cells) when the
 *  app's own copy took it (`view_written`), or, with no copy of its own (the object's zarr read in place), when it was
 *  written into the object's zarr (`written_to`). Not when it went to memory only (sopa's auto-save off) or the app's
 *  copy refused it (view_written false). */
const inZarrRead = save => save.view_written === true || save.view_written == null && !!save.written_to;

export function proceedOffer(run, { api = false, pending = false } = {}) {
  const save = run?.zarr_save;
  const hidden = reason => ({ shown: false, ready: false, reason });
  if (run?.kind !== 'stain' || run.sample_kind !== 'zarr') return hidden('');
  if (!run.sample_tools?.includes('proseg')) return hidden(run.sample_tools ? PROCEED_TEXT.noTranscripts : '');
  if (!['saved', 'outdated', ...(api ? ['queued'] : [])].includes(save?.state) || !(save.cells > 1) || !save.shapes_key) return hidden('');
  if (api && !inZarrRead(save)) return hidden(save.view_written === false ? PROCEED_TEXT.notInLayers : PROCEED_TEXT.memoryOnly);
  const partial = !!save.covered || run.progress?.complete === false;
  const reason = pending ? PROCEED_TEXT.waiting : saveOutdated(run) ? (api ? PROCEED_TEXT.outdatedApi : PROCEED_TEXT.outdated) : '';
  if (!reason && partial && !save.covered?.region) return hidden(PROCEED_TEXT.small);
  return { shown: true, ready: !reason, reason, title: partial ? PROCEED_TEXT.partialTooltip : PROCEED_TEXT.tooltip };
}

/** The saved shapes alone seed proseg. No nucleus pairing or settings are carried over. */
export function proceedPrefill(run) {
  const save = run.zarr_save, covered = save.covered;
  const region = covered?.region || run.region;
  const notes = covered ? [{ code: 'proceed_limited', text: PROCEED_TEXT.limited,
    detail: (covered.why === 'image' ? 'The image covers only this part of the region.' :
      `The stain run stopped after ${covered.done} of ${covered.of} tiles, so its cells cover only this part.`) +
      ' Draw another region to change it.' + (covered.pieces > 1 ? ' Only its largest part is used.' : '') }] : [];
  return { region: JSON.parse(JSON.stringify(region)), prior: { cells: { kind: 'shapes', key: save.shapes_key }, nuclei: null }, notes };
}
