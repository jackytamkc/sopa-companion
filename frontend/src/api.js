// HTTP and live-socket access to the local engine. The page's key travels in a header (page-key.mjs), never in a cookie:
// window.fetch, which every request here goes through, adds it, and the live socket offers it in its handshake.
import { pageKey } from './page-key.mjs';
import { pageSession, pageShowing } from './log-export.mjs';
const BASE = '/api/v1';
// Observe all fetches, including the independent transcript loader, without changing its API.
const nativeFetch = window.fetch.bind(window);
window.fetch = async (...args) => {
  const path = String(args[0]?.url || args[0]);
  // Telemetry is fire-and-forget: its failures (an older engine, a closed one) never reach the user. The mixing check
  // says in its own panel, in plain words, what it is waiting for or why it cannot run (a frame not checked yet is a
  // 404 by design): its requests never raise the general notice either. Nor do the Log button's (log-export.mjs): it
  // saves the page's part whatever the engine answers and its own notice says what was saved.
  // The remove panel's brightness spread says in its own line when the cells could not be measured, and the stain view
  // its outlines' (stain.mjs: a 409 while a run writes is read again, unsaid).
  // Leave and close failures are handled by the shell's prompt, the older runs' answers by their rows (older-runs.mjs).
  // Compare says in its own view what it waits for or could not do (compare.mjs), and Layers > Load Other Mask
  // in its own rows (seg-layers.mjs: a sample's list, a layer's outlines). The crash reports' requests are quiet too: the
  // popup says what failed (crash-report.mjs), and the page's log goes to the engine unseen (log-export.mjs). The
  // training patches' export dialog says in its own lines why a folder cannot be listed (patch-folder.mjs). A counted
  // frame's count of each cell (cellstats.mjs) is a frame's extra: without it the readouts keep the frame's
  // non-background count, in the same words.
  // Clusters & UMAP settings the engine refuses (422) are said in the settings' own line (analysis-settings.mjs), and a
  // result asked for by its settings generation that is gone (404) is one of earlier settings, dropped (analysis.js).
  // An image the page draws says itself when it could not be loaded (main.js: the sample's overview, a run's image).
  // The engine's health is asked where the page says in its own words what no answer means: as it starts (the first
  // screen: the engine cannot be reached), when a notebook's socket drops (the session's end) and as Close is pressed
  // (whether another tab keeps the app open).
  const quiet = /\/api\/v1\/health(?:\?|$)/.test(path) || /\/telemetry(?:\?|$)/.test(path) || /\/qc(?:[/?]|$)/.test(path) || /\/api\/v1\/(?:runs\/[^/?]+\/)?log(?:\?|$)/.test(path) ||
    /\/stain\/(?:brightness|cells|vectors)(?:\?|$)/.test(path) || /\/api\/v1\/closing$/.test(path) || /\/runs\/[^/?]+\/leave$/.test(path) ||
    /\/api\/v1\/runs\/older(?:\/[^/?]+)?$/.test(path) || /\/runs\/[^/?]+\/compare(?:[/?]|$)/.test(path) ||
    /\/samples\/[^/?]+\/segmentations(?:[/?]|$)/.test(path) || /\/api\/v1\/(?:crash(?:[/?]|$)|page-log(?:\?|$))/.test(path) ||
    /\/runs\/[^/?]+\/stain\/patches\/folders(?:\?|$)/.test(path) || /\/runs\/[^/?]+\/counted\/\d+$/.test(path) ||
    /\/samples\/[^/?]+\/overview\.png$|\/runs\/[^/?]+\/image\/[^/?]+\.png$/.test(path);
  // Start's refusals (e.g. no tissue in the region) are shown by the picker in its own words, not as this notice; so are
  // those of the Training patches preview the picker asks for meanwhile (the same checks).
  const creating = args[1]?.method === 'POST' && /\/api\/v1\/runs$/.test(path);
  const previewing = args[1]?.method === 'POST' && /\/api\/v1\/samples\/[^/?]+\/patches\/preview$/.test(path);
  // The Save dialog says a save's refusal for its names itself (the names taken, Replace offered: zarr-save.js): a name
  // clash (409) or a name differing from one only by case (422) is no failed request to retry.
  const saving = args[1]?.method === 'POST' && /\/api\/v1\/runs\/[^/?]+\/zarr$/.test(path);
  // The export dialog says an export's refusals itself: of its folder (422), its boxes changed or an export already on
  // its way (409), in its own line or the panel's notice (patch-folder.mjs), never as a failed request.
  const exporting = args[1]?.method === 'POST' && /\/api\/v1\/runs\/[^/?]+\/stain\/patches\/export$/.test(path);
  // Open in Cellpose GUI's refusals (a set no longer there, a GUI without its packages or one that could not start) are
  // the panel's notice and, for the packages, the command under its button (stain-patches.mjs).
  const opening = args[1]?.method === 'POST' && /\/api\/v1\/runs\/[^/?]+\/stain\/patches\/open$/.test(path);
  const settings = args[1]?.method === 'POST' && /\/api\/v1\/runs\/[^/?]+\/analysis$/.test(path);
  const outdated = /\/api\/v1\/runs\/[^/?]+\/analysis\/\d+(?:\/cells)?\?(?:[^#]*&)?generation=\d+(?:[&#]|$)/.test(path);
  try {
    const response = await nativeFetch(...args);
    // Signed out (page-key.mjs): the page says so on its own screen (main.js signedOut), never as a failed request.
    if (response.status === 401) return response;
    const viewMiss=(response.status===404 && /\/(?:frames\/\d+\/window|prior\/view|transcripts\/view)(?:\?|$)/.test(path)) ||
      (response.status===413 && /\/transcripts\/view(?:\?|$)/.test(path));
    const gone = response.status === 404 && /\/api\/v1\/runs\/[^/?]+(?:\/frames)?(?:\?|$)/.test(path);
    const refused = previewing && response.status === 422 || (saving || exporting) && [409, 422].includes(response.status) ||
      settings && response.status === 422 || outdated && response.status === 404 || opening && [404, 503].includes(response.status);
    if (!response.ok && !quiet && !creating && !refused && !viewMiss && !gone && !(response.status === 404 && /(?:\/transcripts|\/prior|\/xenium|\.u16)(?:\?|$)/.test(path))) {
      window.dispatchEvent(new CustomEvent('fetch-notice', { detail: `Request failed (${response.status}). Please try again.` }));
    }
    return response;
  } catch (error) {
    if (error.name !== 'AbortError' && !quiet) window.dispatchEvent(new CustomEvent('fetch-notice', { detail: 'Connection failed. Please try again.' }));
    throw error;
  }
};

/** A refusal or failure: its words, its HTTP status, and `data`, what the answer says besides its words (an object, else
 *  null): a save's name clash says which names are taken and every name in the sample ({error, clash, existing}). */
export class ApiError extends Error {
  constructor(message, status, data = null) { super(message); this.status = status; this.data = data; }
}

/** `keepalive`: the request goes on even when the page goes meanwhile (a small one: the browser allows 64 KB of them). */
async function request(method, path, body, signal, keepalive = false) {
  const response = await fetch(BASE + path, {
    method, credentials: 'same-origin', signal, ...(keepalive ? { keepalive: true } : {}),
    headers: body === undefined ? {} : { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!response.ok) {
    let answer = null;
    try { answer = await response.json(); } catch { /* not JSON */ }
    // The words are `detail` (the engine's problem answers), else `error`; an object `detail` holds the answer's data.
    const detail = answer?.detail, data = detail && typeof detail === 'object' && !Array.isArray(detail) ? detail :
      answer && typeof answer === 'object' && !Array.isArray(answer) ? answer : null;
    const words = typeof detail === 'string' && detail ? detail : typeof data?.error === 'string' && data.error ? data.error :
      detail ? JSON.stringify(detail) : response.statusText;
    throw new ApiError(words, response.status, data);
  }
  return response.status === 204 ? null : response.json();
}

export const api = {
  options: () => request('GET', '/options'),
  optionsGuide: () => request('GET', '/options/guide'),
  // The first three requests take a signal: the page gives up on them when the engine does not answer (main.js boot).
  health: (signal) => request('GET', '/health', undefined, signal),
  stainModels: (method = 'cellpose') => request('GET', `/stain/models?method=${encodeURIComponent(method)}`),
  tissue: (id,image,signal) => request('GET', `/samples/${encodeURIComponent(id)}/tissue?image=${encodeURIComponent(image)}`,undefined,signal),
  cellposeSettings: (id,signal) => request('GET', `/samples/${encodeURIComponent(id)}/cellpose-settings`,undefined,signal),
  async stainCells(id, set, channel = null, signal) {
    const query = new URLSearchParams({ set }); if (channel != null) query.set('channel', channel);
    const response = await fetch(`${BASE}/runs/${encodeURIComponent(id)}/stain/cells?${query}`, { credentials: 'same-origin', signal });
    if (!response.ok) throw new ApiError("The cells' values could not be read.", response.status);
    return response.arrayBuffer();
  },
  stainExclusions: (id,signal) => request('GET', `/runs/${encodeURIComponent(id)}/stain/exclusions`,undefined,signal),
  stainExclude: (id,body) => request('POST', `/runs/${encodeURIComponent(id)}/stain/exclusions`,body),
  stainRestore: (id,area,version) => request('DELETE', `/runs/${encodeURIComponent(id)}/stain/exclusions/${encodeURIComponent(area)}?version=${version}`),
  stainExcludeSize: (id,body) => request('POST', `/runs/${encodeURIComponent(id)}/stain/exclusions/size`,body),
  // The brightness rule: the cells' brightness in a channel (none named: the run's
  // default; the first read of a large run takes about a minute), and {version, channel, min, max}.
  stainBrightness: (id,channel,signal) => request('GET', `/runs/${encodeURIComponent(id)}/stain/brightness${channel ? `?channel=${encodeURIComponent(channel)}` : ''}`,undefined,signal),
  stainExcludeBrightness: (id,body) => request('POST', `/runs/${encodeURIComponent(id)}/stain/exclusions/brightness`,body),
  stainPatches: (id,signal) => request('GET', `/runs/${encodeURIComponent(id)}/stain/patches`,undefined,signal),
  stainPropose: (id,body) => request('POST', `/runs/${encodeURIComponent(id)}/stain/patches/propose`,body),
  stainPatchBoxes: (id,body) => request('POST', `/runs/${encodeURIComponent(id)}/stain/patches`,body),
  stainPatchExport: (id,body) => request('POST', `/runs/${encodeURIComponent(id)}/stain/patches/export`,body),
  stainPatchOpen: (id,body) => request('POST', `/runs/${encodeURIComponent(id)}/stain/patches/open`,body),
  /** The export dialog's view of folder `path` (none: the folder it suggests for this run's patches): {path, exists,
   *  parent, entries, shortcuts, new, holds, mixes, refused}; nothing is written (engine/patch_folders.py). */
  stainPatchFolders: (id,path,signal) => request('GET', `/runs/${encodeURIComponent(id)}/stain/patches/folders${path ? `?path=${encodeURIComponent(path)}` : ''}`,undefined,signal),
  // The recent samples: a list (a notebook's session answers {samples: [the bound one]}).
  samples: (signal) => request('GET', '/samples', undefined, signal).then(list => Array.isArray(list) ? list : Array.isArray(list?.samples) ? list.samples : []),
  browse: (path) => request('GET', '/browse' + (path ? `?path=${encodeURIComponent(path)}` : '')),
  openSample: (path) => request('POST', '/samples/open', { path }),
  forgetSample: (id) => request('DELETE', `/samples/${encodeURIComponent(id)}`),
  sample: (id) => request('GET', `/samples/${encodeURIComponent(id)}`),
  shapes: (id, key, box, signal) => request('GET', `/samples/${encodeURIComponent(id)}/shapes/${encodeURIComponent(key)}?box=${box.join(',')}&limit=20000`, undefined, signal),
  overviewUrl: (id) => `${BASE}/samples/${encodeURIComponent(id)}/overview.png`,
  /** An image from the engine (`url`: overviewUrl, imageUrl), decoded, ready to draw: fetched, since the page's key goes
   *  in a header, which an <img> cannot send. */
  async image(url, signal) {
    const response = await fetch(url, { signal });
    if (!response.ok) throw new ApiError(`HTTP ${response.status}`, response.status);
    const blob = await response.blob(), address = URL.createObjectURL(blob), img = new Image();
    try { img.src = address; await img.decode(); return img; } finally { URL.revokeObjectURL(address); }
  },
  runs: (signal) => request('GET', '/runs', undefined, signal),
  /** The runs older versions made and never saved: {runs: [{id, sample, sample_name, kind, method,
   *  state, iter, tiles, created, region, prior, options, stain, can_add, why, still_saving, answer, stage, message,
   *  zarr_save}]}, newest first. */
  olderRuns: (signal) => request('GET', '/runs/older', undefined, signal),
  /** Save to the sample ("add") or Discard an older run (`id`), or every row (`id` null; add skips those it cannot save):
   *  the engine answers at once and works on the rows one at a time. Later is `id` null with "later": the rows are set
   *  aside until the next start, and a Save all stops after the row under way. */
  answerOlder: (id, answer) => id == null ? request('POST', '/runs/older', { answer }) : request('POST', `/runs/older/${encodeURIComponent(id)}`, { answer }),
  // A run as it stands; its `save_defaults` are worked out against the sample's names as they are now (the Save dialog
  // asks each time it opens: zarr-save.js).
  run: (id) => request('GET', `/runs/${encodeURIComponent(id)}`),
  cells: (id) => request('GET', `/runs/${encodeURIComponent(id)}/cells`),
  frames: (id, since = 0) => request('GET', `/runs/${encodeURIComponent(id)}/frames?since=${since}`),
  createRun: (sample, region, prior, options = {}) => request('POST', '/runs', { sample, region, prior, options }),
  createStain: (sample, region, stain, kind = 'stain') => request('POST', '/runs', { kind, sample, region, stain }),
  /** The boxes a Training patches item would start with for this region and `stain` (as createStain takes it): {patches:
   *  [{id, box}], run_pixel_to_sample, message, extent: [width, height] of the region's image window (the boxes' frame),
   *  outline (a lasso only: [[x, y], …] in that frame)}. Nothing is written; refused (422) as creating the item would be. */
  previewPatches: (sample, region, stain, signal) =>
    request('POST', `/samples/${encodeURIComponent(sample)}/patches/preview`, { region, stain }, signal),
  /** 409 on a name clash, its error's `data` {error, clash: [names taken], existing: [every name in the sample]}. */
  saveZarr: (id, values) => request('POST', `/runs/${encodeURIComponent(id)}/zarr`, values),
  control: (id, action, speed, values) => request('POST', `/runs/${encodeURIComponent(id)}/control`, { action, speed, values }),
  /** Save to the sample ("add"), Discard or Stay for leaving the run; `revision` orders this page's answers (shell.mjs
   *  nextRevision), `page` names them as this page's: the engine takes an answer only over an older one. */
  leave: (id, answer, revision = null) => request('POST', `/runs/${encodeURIComponent(id)}/leave`,
    Number.isFinite(revision) ? { answer, revision, page: pageSession.session } : { answer, page: pageSession.session }),
  /** The answer to closing, "add" | "discard" (shell.mjs), for the runs it was given for (`runs`: their ids) with this
   *  page's `revision` (a newer revision, or a run started since, takes an older answer back), its name (`page`) and
   *  the showing that answers (`visit`: log-export.mjs pageShowing; one the page said before it was shown again does
   *  not count); `end`: a notebook's session ends once it is kept (Close). `beacon`: sent as the page goes, its goodbye
   *  (`bye`): a keepalive POST with the page's key, sent at once (page-key.mjs send), that never throws. */
  closing: (answer, { beacon = false, runs = [], revision = null, end = false } = {}) => {
    const body = { answer, runs: runs.slice(0, 64), ...(Number.isFinite(revision) ? { revision } : {}), page: pageSession.session,
      visit: pageShowing.visit, ...(end ? { end: true } : {}) };
    return beacon ? pageKey?.send(BASE + '/closing', JSON.stringify({ ...body, bye: true })) ?? false : request('POST', '/closing', body);
  },
  // A session that ended unexpectedly (crash-report.mjs; the engine writes and keeps the reports).
  /** The newest report not seen yet: {report: null | {id, at, reason, bytes, started}}. */
  crash: () => request('GET', '/crash'),
  /** The page found a page that did not unload ({session, at, found: 'tab' (the page before it in the same window) |
   *  'flag' (its flag alone)}): the engine writes that page's report, then answers as crash(). */
  crashPage: ({ session, at, found }) => request('POST', '/crash/page', found ? { at, session, found } : { at, session }),
  /** A report's text (text/plain). */
  async crashText(id) {
    const response = await fetch(`${BASE}/crash/${encodeURIComponent(id)}.txt`, { credentials: 'same-origin', cache: 'no-store' });
    if (!response.ok) throw new ApiError(`HTTP ${response.status}`, response.status);
    return response.text();
  },
  /** The report was offered (Save a copy… or Not now): never offered again. Sent even when the window closes or
   *  reloads right after the answer. */
  crashSeen: (id) => request('POST', `/crash/${encodeURIComponent(id)}/seen`, undefined, undefined, true),
  imageUrl: (id, file) => `${BASE}/runs/${encodeURIComponent(id)}/image/${file}`,
  // The mixing check: a sample's model, a run's switch, results and per-cell values.
  sampleQc: (id, signal) => request('GET', `/samples/${encodeURIComponent(id)}/qc`, undefined, signal),
  learnSampleQc: (id) => request('POST', `/samples/${encodeURIComponent(id)}/qc`, { learn: true }),
  runQc: (id, signal) => request('GET', `/runs/${encodeURIComponent(id)}/qc`, undefined, signal),
  /** {enabled?: bool, seq?: n}: the run's switch, or check that frame now. */
  setRunQc: (id, body) => request('POST', `/runs/${encodeURIComponent(id)}/qc`, body),
  qcSeries: (id, points = 600, signal) => request('GET', `/runs/${encodeURIComponent(id)}/qc/series?points=${points}`, undefined, signal),
  qcSummary: (id, seq, signal) => request('GET', `/runs/${encodeURIComponent(id)}/qc/${seq}`, undefined, signal),
  /** The start line with its own areas (where the starting cells already look mixed): {per100, judged, ..., areas}. */
  qcStart: (id, signal) => request('GET', `/runs/${encodeURIComponent(id)}/qc/start`, undefined, signal),
  /** {rule, area, flag, from, to}: the change the page has just set, to be watched (a trial) -> the trial. */
  qcTrial: (id, body) => request('POST', `/runs/${encodeURIComponent(id)}/qc/trial`, body),
  /** "kept" | "undone" -> the ended trial; with rule "U": the answer to the "Undo your last change?" card. */
  qcTrialEnd: (id, outcome, rule = null) => request('POST', `/runs/${encodeURIComponent(id)}/qc/trial/end`, rule ? { outcome, rule } : { outcome }),
  // "Test a setting here…" (quiet like every /qc request: the panel says what went wrong).
  /** The pool a test of `cell` (label - 1) would test, before any run: {v, state ('ready' | 'working': asked again,
   *  waiting up to `wait` seconds, 0 to 10 | 'failed'), run, cell, cell_name, seq, iter, model, constants, start,
   *  runs_per_side, k, area {rule: 'as_is' | 'square', box, inner} | null, offer, reasons [{code, words, ...data}],
   *  pattern, line, this_cell, pool {cells, suspects, need, reached, available, members} | null, estimate_s, ms,
   *  message}; 404 an unknown run or cell, 422 a bad cell or wait. `signal`: the page gives up on it. */
  qcTestPool: (id, cell, { wait = 0, signal } = {}) =>
    request('GET', `/runs/${encodeURIComponent(id)}/qc/test/pool?cell=${encodeURIComponent(cell)}&wait=${encodeURIComponent(wait)}`, undefined, signal),
  /** {cell (label - 1), flag: "cell-compactness" | "prior-weight" | "diffusion-probability", value} -> the test (202);
   *  409 a test of this run on its way or another run sampling, or the reasons a test can't answer there (its answer's
   *  `reasons`: ApiError data.reasons), 422 a bad cell, flag or value, the run's own value, far drift with drift off, 404
   *  no such cell. The engine fixes the suspects and their neighbours itself. */
  qcTestStart: (id, body) => request('POST', `/runs/${encodeURIComponent(id)}/qc/test`, body),
  /** The run's test {id, state, cell, cell_name, flag, name, value, current, tried, start ('cells' | 'paired'), area
   *  {rule, box, inner}, square (the area when it is a square around the cell) | null, runs_per_side, threads, runs {A1..,
   *  B1..}, progress {iter, of, runs, done, sides {A, B}, marks, phase, retry}, pool {cells, suspects, k, matched,
   *  uniform, unsure, not_found}, message, result (v 2, once done), was_running, seconds, hold {value, effective} | null,
   *  no_factorization, reruns, estimate_s}; 404 none. */
  qcTest: (id, signal) => request('GET', `/runs/${encodeURIComponent(id)}/qc/test`, undefined, signal),
  /** Cancels the run's test (all its runs stopped at once, then deleted); a test that has ended is forgotten. */
  qcTestCancel: (id) => request('DELETE', `/runs/${encodeURIComponent(id)}/qc/test`),
  /** The test's corner view: {state, box: [x0, y0, x1, y1] µm | null, suspects (the cell's, as the engine fixed them),
   *  runs: [{role ("A1": as now | "B1": with the value), side ("A" | "B"), seq, iter, voxel, x0, y0, w, h, labels
   *  (base64 of little-endian u32, w * h, row-major, y down), cell, sources, held (per suspect, proseg's share of it kept
   *  for the cell, or null), held_by ("exact" | "approx"), unsure, cell_total (proseg's count of the cell), parents
   *  ([label, its cell's label in your run, ...]), ms}]}; 404 when the run has no test (quiet: /qc). */
  qcTestView: (id, signal) => request('GET', `/runs/${encodeURIComponent(id)}/qc/test/view`, undefined, signal),
  // Go deeper: one cell's RNA, explained. `cell` is the label - 1 (the index
  // into GET /runs/{id}/cells), `seq` a frame or 'start'. Quiet like every /qc request (the window says what it waits for).
  /** {seq, cell, follow?} -> {state, reason?, seq, cell, model, deep, progress?, eta_s?, result?}; follow: the engine
   *  works the cell out with every analysed pass of the run from now on (its results come as {ev: "deeper"} pushes). */
  qcDeeper: (id, body) => request('POST', `/runs/${encodeURIComponent(id)}/qc/deeper`, body),
  /** The window closed: the engine stops following its cell. */
  qcDeeperStop: (id) => request('DELETE', `/runs/${encodeURIComponent(id)}/qc/deeper`),
  /** The cell's bytes over the run, for the window's small line: {model, start: code | null, points: [[iter, code]]}. */
  qcDeeperHistory: (id, cell, points = 200, signal) =>
    request('GET', `/runs/${encodeURIComponent(id)}/qc/deeper/history?cell=${encodeURIComponent(cell)}&points=${points}`, undefined, signal),
  /** "Show where on the image": {seq: n | 'start', cell, sources: [cell], genes:
   *  [name], own: [name]} (cells as label - 1) -> where the cell's and its sources' transcripts of those genes lie in
   *  that frame {v, seq, box, genes, roles, points, cell_total, gene_stats, into, ...}. */
  qcDeeperWhere: (id, { seq, cell, sources = [], genes = [], own = [] }, signal) => request('GET', `/runs/${encodeURIComponent(id)}/qc/deeper/where`
    + `?seq=${encodeURIComponent(seq)}&cell=${encodeURIComponent(cell)}&sources=${sources.map(encodeURIComponent).join(',')}`
    + `&genes=${genes.map(encodeURIComponent).join(',')}&own=${own.map(encodeURIComponent).join(',')}`, undefined, signal),
  /** "Learn the details" for a sample learned before Go deeper (its model keeps no details) -> as GET /samples/{id}/qc
   *  (with `model.deep` and `detail {state, progress, eta_s, reason}`). */
  learnSampleDetail: (id) => request('POST', `/samples/${encodeURIComponent(id)}/qc`, { detail: true }),
  /** One byte per cell (label - 1 order; share x 250, 255 = not judged) of frame `seq`, or of the start line ('start'),
   *  as checked by `model` on `count` (the result's: "averaged" | "sampled"; a frame counted again for the hold, a
   *  finish or the end is scored anew on the count proseg saves). The URL names both, so it is cached as immutable; a
   *  count the frame no longer has is a 404 (the result's summary is then asked for again). */
  async qcCells(id, seq, model, signal, count = null) {
    const where = seq === 'start' ? 'start/cells' : `${seq}/cells`;
    const counted = typeof count === 'string' && count && seq !== 'start' ? `&count=${encodeURIComponent(count)}` : '';
    const response = await fetch(`${BASE}/runs/${encodeURIComponent(id)}/qc/${where}?model=${encodeURIComponent(model)}${counted}`, { credentials: 'same-origin', signal });
    if (!response.ok) throw new ApiError(`Mixing values ${seq}: ${response.status}`, response.status);
    return new Uint8Array(await response.arrayBuffer());
  },
  // Segmentation layers and Compare (quiet: the view and the Layers group say what failed).
  /** Every cell-like shapes element of the sample: {segmentations: [{key, name, source, model, table, coverage, count,
   *  extent, version, outlines}]} (Layers > Load Other Mask reads it itself: seg-layers.mjs). */
  segmentations: (id, signal) => request('GET', `/samples/${encodeURIComponent(id)}/segmentations`, undefined, signal),
  /** Compare: start (or reuse) the comparison of `columns` ('run' and segmentation keys, 1-4, each once) at frame
   *  `seq` (the frame on screen; the engine's newest with cell statistics without one) -> the comparison as it stands
   *  (202). */
  compareStart: (id, columns, seq = null) => request('POST', `/runs/${encodeURIComponent(id)}/compare`,
    Number.isInteger(seq) ? { columns, seq } : { columns }),
  /** The run's comparison {job, state, step, progress, model, seq, iter, area, kinds, kinds_reason, columns, layout,
   *  bounds, layout_error, error, seconds}; 404 before it has one. */
  compareStatus: (id, signal) => request('GET', `/runs/${encodeURIComponent(id)}/compare`, undefined, signal),
  /** Column `index`'s UMAP points: 10 little-endian bytes each (f32 x, f32 y, u8 kind (255 none), u8 share (floor(share
   *  x 250), 255 not judged)); asked again when the comparison's `layout` grows. `job` and `layout` name the comparison
   *  and the layout the page read in its status: the engine refuses (409) points of another one (a second window may
   *  have started a new comparison of the run meanwhile). `what`: 'own' for its own map's points (the same records, its
   *  own x and y: ?own=1), 'clusters' for its own clusters (a little-endian u16 per point). */
  async comparePoints(id, index, job = null, layout = null, signal, what = 'points') {
    const query = [...(job != null ? [`job=${encodeURIComponent(job)}`] : []), ...(Number.isInteger(layout) ? [`layout=${layout}`] : []),
      ...(what === 'own' ? ['own=1'] : [])];
    const route = what === 'clusters' ? 'clusters' : 'points';
    const response = await fetch(`${BASE}/runs/${encodeURIComponent(id)}/compare/${route}/${index}.bin${query.length ? `?${query.join('&')}` : ''}`,
      { credentials: 'same-origin', signal });
    if (!response.ok) throw new ApiError(`Compare points ${index}: ${response.status}`, response.status);
    return response.arrayBuffer();
  },
  async frame(id, seq, signal) {
    // Stored gzip-compressed with Content-Encoding: gzip, so this is already the raw little-endian u32 raster.
    const response = await fetch(`${BASE}/runs/${encodeURIComponent(id)}/frames/${seq}`, { credentials: 'same-origin', signal });
    if (!response.ok) throw new ApiError(`Frame ${seq}: ${response.status}`, response.status);
    return response.arrayBuffer();
  },
};

// One socket for the whole page: run-list updates plus the frames of the run being watched. It signs in in its
// handshake: its address names the page (?page=, the page's log session) and it offers the page's key as a subprotocol
// (page-key.mjs protocols()), so nothing is said first and no key is ever in an address. Without a key it never
// connects; one the engine refuses closes with 4401, and the page is signed out. Reconnects by itself, until stop() (a
// notebook's session that ended: its engine is gone for good) or goodbye() (the page is going).
export class LiveSocket {
  constructor(handlers, { keys = pageKey, page = pageSession.session, showing = pageShowing } = {}) {
    this.handlers = handlers;
    this.keys = keys;
    this.page = page;
    this.showing = showing;
    this.watching = null;
    this.since = 0;
    this.delay = 300;
    this.stopped = false;
    this.leaving = false;
    this.open();
  }
  stop() { this.stopped = true; try { this.socket?.close(); } catch { /* closed already */ } }
  /** Whether its socket is open now: the engine counts it among the pages open (main.js Close asks how many). */
  connected() { return this.socket?.readyState === WebSocket.OPEN; }
  /** The page is going (pagehide, Close): the engine hears it ({op: "bye"}), and the socket does not come back. */
  goodbye() { this.leaving = true; this.send({ op: 'bye' }); }
  /** The page came back (the browser kept it aside: pageshow persisted): a new showing, which connects again. A socket
   *  still open is replaced too, so the engine hears the page is here again as that showing: a new connection takes a
   *  goodbye back, and a goodbye of the showing before that reaches the engine later changes nothing. */
  back() {
    const socket = this.socket;
    this.leaving = false; this.stopped = false;
    this.showing.visit += 1;
    this.socket = null; clearInterval(this.ping);          // what the old one says next is no longer heard
    if (socket && socket.readyState <= WebSocket.OPEN) { try { socket.close(); } catch { /* closed already */ } }
    this.open();
  }
  async open() {
    if (!await this.keys?.ready || this.stopped || this.leaving) return;
    if (this.socket && this.socket.readyState <= WebSocket.OPEN) return;      // one connecting or open already
    const protocols = this.keys.protocols();
    if (!protocols) return;                              // signed out meanwhile
    const url = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}${BASE}/ws?page=${encodeURIComponent(this.page)}&visit=${this.showing.visit}`;
    let socket;
    try { socket = this.socket = new WebSocket(url, protocols); } catch { this.stopped = true; return; }
    socket.onopen = () => {
      this.delay = 300;
      this.handlers.connection?.(true);
      if (this.watching) this.send({ op: 'watch', run: this.watching, since: this.since });
      clearInterval(this.ping);
      this.ping = setInterval(() => this.send({ op: 'ping' }), 20000);
    };
    socket.onmessage = (event) => {
      if (socket !== this.socket) return;
      let message;
      try { message = JSON.parse(event.data); } catch { return; }
      if (message.ev === 'frames' && message.run === this.watching) this.since = Math.max(this.since, message.from + message.frames.length);
      this.handlers.message?.(message);
    };
    socket.onclose = (event) => {
      if (socket !== this.socket) return;                // replaced already (back): the new one counts
      clearInterval(this.ping);
      // The engine refused the page's key (it started again since, with a new one): signed out, which the page says.
      if (event?.code === 4401) { this.stopped = true; this.keys.signedOut('refused'); }
      this.handlers.connection?.(false);
      if (this.stopped || this.leaving) return;
      setTimeout(() => { if (!this.stopped && !this.leaving) this.open(); }, this.delay);
      this.delay = Math.min(this.delay * 2, 5000);
    };
  }
  send(value) {
    if (this.socket?.readyState === WebSocket.OPEN) this.socket.send(JSON.stringify(value));
  }
  watch(runId, since) {
    this.watching = runId;
    this.since = since;
    this.send({ op: 'watch', run: runId, since });
  }
}
