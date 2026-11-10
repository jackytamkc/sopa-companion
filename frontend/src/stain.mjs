import { decodeVectors } from './vectors-decode.mjs';
import { ViewRequests } from './tiled-loader.mjs';
import { visibleBox, intersect, covers } from './tiled-view.mjs';
import { problem } from './explain.mjs';
import { boxInPolygon } from './region.mjs';

export const STAIN_TEXT = Object.freeze({
  outlines: 'Cell outlines',
  saveHelp: 'Writes the model’s cell shapes with the method and settings. You can use them as starting shapes for Proseg.',
  kind: 'Run type', proseg: 'Proseg', title: 'Stain segmentation', method: 'Method',
  cellpose: 'Cellpose', instanseg: 'InstanSeg', image: 'Image', nucleus: 'Nucleus channel',
  channels: 'Channels', channelsHelp: 'Choose one or more channels. For brightfield, the image channels must be ordered red, green, blue.',
  output: 'Find', nuclei: 'Nuclei', resolve: 'Match cells and nuclei',
  model: 'Model', yourModels: 'Your models', modelsLoading: 'Loading the model list…',
  modelsError: 'The model list could not be loaded. Try opening the picker again.',
  // Cellpose's settings go by Cellpose's own names; each one's plain words are its tooltip (CELLPOSE_WORDS).
  cytoplasm: 'Cell body or membrane channel', none: 'None', diameter: 'diameter',
  auto: 'Auto', autoHelp: 'Auto uses the model on the image as read, without resizing.',
  // The image's pyramid levels go by their SpatialData names (levelName): scale0, scale1, …
  resolution: 'Resolution', resolutionStored: 'the image as stored',
  resolutionHelp: 'scale0 is the image as stored; each scale after it is a smaller copy the sample already holds, '+
    'faster to segment but with less detail. Use the same resolution for training patches and for segmenting the '+
    'whole sample: if you train on scale1 patches, segment at scale1 too.',
  resolutionSingle: 'This image has scale0 only: no smaller copies.',
  resolutionSmall: 'Cells are about {n} px across here: too small to outline well.',
  resolutionSmallHelp: 'Below about 10 pixels across, cells lose their edges and small ones are missed. Choose a finer resolution.',
  resolutionDiameter: 'With a cell diameter set, Cellpose resizes the image so cells are 30 px across, so a smaller copy mostly saves reading time.',
  resolutionInstanseg: 'InstanSeg resizes every image to {n} µm per pixel itself, so a smaller copy mostly saves reading time.',
  resolutionEnlarged: 'Coarser than InstanSeg reads ({n} µm per pixel): it is enlarged back and loses detail.',
  diameterHelp: 'In pixels of scale0 (the image as stored). Cellpose gets it at the chosen resolution.',
  flow: 'flow_threshold', flowHelp: 'Lower values reject more uncertain shapes.',
  probability: 'cellprob_threshold', probabilityHelp: 'Lower values find more and larger cells.',
  tissue: 'Tissue only', more: 'More Cellpose settings',
  niter: 'niter', niterHelp: 'Leave blank for Cellpose to choose.',
  minSize: 'min_size', augment: 'augment',
  normalize: 'normalize', low: 'Low percentile', high: 'High percentile',
  fixedRange: 'lowhigh', intensityLow: 'Low intensity', intensityHigh: 'High intensity',
  tileNorm: 'tile_norm_blocksize', sharpen: 'sharpen_radius',
  smooth: 'smooth_radius', invert: 'invert',
  tiles: 'Finished tiles appear as outlines.', pause: 'Pause', resume: 'Resume', stop: 'Stop',
  pausing: 'Pausing after this tile…', resuming: 'Resuming…', stopping: 'Finishing this tile and joining shapes…',
  preparing: 'Loading the image and model…', failed: 'The image run failed. Details are in the Log.',
  unavailable: 'The image segmentation software is unavailable.',
  settingsError: 'Check the cell size and thresholds.', save: 'Save as shapes', writing: 'Writing cell shapes…',
  savedOutdated: 'Saved shapes are from an earlier result. Save again to include your changes.',
  cellOutlines: 'Cell outlines', sample: 'Sample', cells: 'Cells', progress: 'Tiles', sizes: 'Cell area',
  tileCells: 'Cells before joining',
  fullHelp: 'The whole image within the sample will be segmented in tiles.',
  previewError: 'The chosen image could not be shown. Details are in the Log.',
  lastTile: 'Last tile', state: 'State', partial: 'Stopped; Resume continues from the finished tiles.',
  interim: 'Shapes at tile edges are joined when the run stops or finishes.',
  complete: 'Finished', running: 'Finding cells…', paused: 'Paused',
  start: 'Start', waiting: 'Press Start to begin segmenting.',
  vectorsError: 'The cell outlines could not be loaded. Try opening the run again.',
  delete: 'Delete this run and its masks and shapes?',
});

// What a press of the picker's Start finds missing (main.js pickerProblems, explain.mjs): each message follows the
// field's own label ("Model: the list is still loading"); one without a field stands alone.
export const PICKER_TEXT = Object.freeze({
  starting: 'Starting…', suggesting: 'Suggesting boxes…', loading: 'The sample is still loading.', openSample: 'Open a sample first.',
  notBuilt: 'proseg-live is not built yet. Run native/build.sh, then reload.',
  notInstalled: "proseg-live is not installed in the app's Python environment. Install it as the README's Install section says, then reload.",
  notFound: 'The proseg-live that [proseg] binary names is not there. Correct that setting, then reload.',
  startFrom: 'Start from', firstCells: 'choose where proseg starts its cells',
  noFirstCells: 'this sample has no cells or nuclei to start from',
  region: 'Region', drawRegion: 'draw one on the image', drawLasso: 'draw a lasso on the image',
  closeLasso: 'close the outline first: click its first point, double-click or press Enter',
  lassoFew: 'Draw an area with at least three corners.',
  regionSmall: 'make it at least 20 µm across in both directions',
  patchRegion: 'less than 64 pixels across, too small for patches',
  count: 'a whole number from 1 to 64', size: 'a whole number from 64 to 4096',
  sizeFits: 'at most {largest}, so each patch fits inside the region',
  unavailable: 'not available on this computer', method: 'choose one from the list',
  modelsLoading: 'the list is still loading', modelsError: 'the list could not be loaded. Try opening the picker again.',
  model: 'choose one from the list', noModels: 'no models were found',
  image: 'choose one', noImage: 'this sample has no image', resolution: 'choose one this image has',
  resolutionRegion: 'the region is only {n} pixels across at {label}',
  channel: 'choose a channel of this image', otherChannel: 'choose a channel other than the nucleus channel, or None',
  channelOrNone: 'choose a channel of this image, or None', oneChannel: 'choose None, as this model reads one channel',
  channels: 'choose one or more', exactlyOne: 'choose exactly one for this model',
  threeChannels: 'choose three for this model (red, green and blue)',
  diameter: 'a number from 10 to 1000, or tick Auto', flow: 'a number from 0 to 10',
  probability: 'a number from −20 to 20', niter: 'a whole number of 1 or more, or leave it blank',
  minSize: 'a whole number from −1 to 100000', number: 'a number', aboveIntensity: 'more than the low intensity',
  percent: 'a number from 0 to 100', abovePercent: 'more than the low percentile',
  tileNorm: 'a whole number of 0 or more (0 is off)', zeroOrMore: 'a number of 0 or more',
  output: 'choose Cells or Nuclei', nucleiOnly: 'choose Nuclei for this model', resolve: 'untick it for this model',
  tick: 'tick or untick it', box: 'Box {n}', boxOutside: 'move it inside the region',
  // Under Start when the engine could not answer for its boxes (main.js previewedBoxes): nothing is made.
  boxesFailed: 'The boxes could not be suggested ({reason}). Try again.',
});

/** The stain methods the engine offers, in the picker's order (main.js offerStainMethods): Cellpose, and InstanSeg only
 *  where health's stain.methods lists it. InstanSeg is dormant: an engine lists it only while its internal switch is on,
 *  so the picker offers Cellpose alone and names no other method. */
export function stainMethods(health) {
  const listed = health?.stain?.methods;
  return ['cellpose', 'instanseg'].filter(method => method === 'cellpose' ||
    !!listed && typeof listed === 'object' && Object.hasOwn(listed, method));
}

/** Cellpose's own defaults as the picker's tuning fields hold them before they are changed (index.html): Training patches
 *  started from Empty send these, as their tuning is hidden then (main.js readStain), so a press of Start never names a
 *  field the user cannot see. */
export const CELLPOSE_DEFAULTS = Object.freeze({
  'stain-auto': true, 'stain-diameter': '30', 'stain-flow': '0.4', 'stain-probability': '0', 'stain-niter': '',
  'stain-min-size': '15', 'stain-augment': false, 'stain-normalize': true, 'stain-fixed-range': false,
  'stain-intensity-low': '', 'stain-intensity-high': '', 'stain-low': '1', 'stain-high': '99', 'stain-tile-norm': '0',
  'stain-sharpen': '0', 'stain-smooth': '0', 'stain-invert': false,
});

/** The plain words of each Cellpose setting the picker names by Cellpose's own name (STAIN_TEXT: diameter,
 *  flow_threshold, cellprob_threshold, …): the field's tooltip, with the setting's help where it has one. */
const CELLPOSE_WORDS = Object.freeze({
  diameter: 'Cell diameter', flow: 'Shape check', probability: 'Cell threshold', niter: 'Steps', minSize: 'Smallest cell',
  augment: 'Try flipped images', normalize: 'Adjust brightness', fixedRange: 'Fixed intensity range',
  tileNorm: 'Local brightness size', sharpen: 'Sharpening', smooth: 'Smoothing', invert: 'Reverse brightness',
});
/** A Cellpose setting's tooltip (its plain words, then its help); '' for any other STAIN_TEXT key. */
const cellposeTitle = key => Object.hasOwn(CELLPOSE_WORDS,key) ?
  [CELLPOSE_WORDS[key],STAIN_TEXT[`${key}Help`]].filter(Boolean).join('. ') : '';
/** The picker's words (index.html's [data-stain-copy] nodes, main.js setupControls): each node's text, and on a Cellpose
 *  setting's label (its name and its box together) the plain words as the tooltip, its field's own too (what a screen
 *  reader says with the name). */
export function stainCopy(nodes) {
  for(const node of nodes) {
    const key=node.dataset.stainCopy, title=cellposeTitle(key), label=node.closest('label');
    node.textContent=STAIN_TEXT[key];
    if(!title || !label)continue;
    label.title=title;
    for(const field of label.children)if(['INPUT','SELECT'].includes(field.tagName))field.title=title;
  }
}

/** Pyramid level k by its SpatialData name, the image's DataTree child: scale0 (the image as stored), scale1, … */
export const levelName = level => `scale${level}`;
/** The level name of a stored resolution record (a run's, a patch set's, a model's): from its level; a record of an
 *  earlier version with only a factor (or only a label such as "1/2") by the nearest power of two of it. '' when it
 *  says none of these (the engine's rule too: engine/stain_levels.py resolution_name). */
export function resolutionName(record) {
  if(Number.isInteger(record?.level) && record.level>=0)return levelName(record.level);
  let factor=record?.factor;
  if(!(Number.isFinite(factor) && factor>0)) {
    const label=typeof record?.label==='string'?record.label:'', old=/^1\/(\d+(?:\.\d+)?)$/.exec(label);
    factor=label==='Full'?1:old?Number(old[1]):NaN;
  }
  return factor>0?levelName(Math.max(0,Math.round(Math.log2(factor)))):'';
}
/** A level's tooltip: "scale1: 0.425 µm per pixel, 2x coarser than scale0" (scale0: "…, the image as stored"). */
export function resolutionTitle(scale) {
  const pixel=scale.um_per_px>0?`${Number(scale.um_per_px.toPrecision(3))} µm per pixel`:'';
  const size=!(scale.level>0)?STAIN_TEXT.resolutionStored:
    scale.factor>0?`${Number(scale.factor.toFixed(1))}x coarser than ${levelName(0)}`:'';
  return `${levelName(scale.level)}: ${[pixel,size].filter(Boolean).join(', ')}`;
}
/** A model's option: its name, its date and, trained on patches cut at a smaller copy, that copy's name. */
export function modelLabel(model) {
  const learned=resolutionName(model.resolution);
  return [model.name,model.date,learned && learned!==levelName(0)?`learned at ${learned}`:null].filter(Boolean).join(' · ');
}

const pixelNumber = value => Number(value.toFixed(3)).toString();
/** The engine's offered levels by their names, with scale0 alone for samples inspected before multiscale support.
 *  Never invent copies. */
export function resolutionChoices(image) {
  if(!image)return [];
  const scales=image.scales?.length?image.scales:[{level:0,factor:1,um_per_px:image.um_per_px,um_per_px_y:image.um_per_px_y}];
  return scales.filter(s=>Number.isInteger(s.level) && s.level>=0 && s.factor>0)
    .map(s=>({...s,label:levelName(s.level),title:resolutionTitle(s)}));
}
/** Keep physical resolution across images (within 5%); otherwise scale0. */
export function matchResolution(image, resolution) {
  if(!(resolution?.um_per_px>0))return 0;
  const choices=resolutionChoices(image).filter(s=>s.um_per_px>0)
    .sort((a,b)=>Math.abs(a.um_per_px/resolution.um_per_px-1)-Math.abs(b.um_per_px/resolution.um_per_px-1));
  return choices.length && Math.abs(choices[0].um_per_px/resolution.um_per_px-1)<=.05?choices[0].level:0;
}
/** Picker and run facts use the same readout. cellPxMedian is the probe's size at this resolution. */
export function resolutionReadout(image, settings={}, {sample=null,pixelFrame=false,cellPxMedian=null,modelPixelSize=.5}={}) {
  const choice=resolutionChoices(image).find(s=>s.level===(settings.level??0));
  if(!choice)return {text:'',warning:'',title:'',cellPx:null};
  const pixel=choice.um_per_px, factor=choice.factor;
  let cellPx=cellPxMedian>0?cellPxMedian:null;
  if(cellPx===null && settings.method!=='instanseg' && settings.diameter>0)cellPx=settings.diameter/factor;
  if(cellPx===null && pixel>0) {
    // What the run outlines: InstanSeg what Find asks for; Cellpose nuclei exactly when it has no cell body channel.
    const nuclei=settings.method==='instanseg'?settings.output==='nuclei':!settings.cytoplasm;
    const prior=sample?.suggested_prior?.[nuclei?'nuclei':'cells'],key=prior?.key ?? prior;
    const shapes=(sample?.shapes || []).filter(s=>s.median_area_um2>0);
    const candidates=sample?.prior_options?.[nuclei?'nuclei':'cells'] || [];
    const shape=shapes.find(s=>s.key===key) || shapes.find(s=>candidates.some(c=>c.key===s.key)) ||
      shapes.find(s=>nuclei?/nucle/i.test(s.key):/cell/i.test(s.key)) || shapes.find(s=>/cell|nucle/i.test(s.key));
    if(shape)cellPx=2*Math.sqrt(shape.median_area_um2/Math.PI)/pixel;
  }
  const pixels=pixelFrame || sample?.notes?.some(n=>n.code==='pixel_frame');
  const text=[!pixels && pixel>0?`${pixelNumber(pixel)} µm per pixel`:'',cellPx!==null?`cells about ${Math.round(cellPx)} px across`:''].filter(Boolean).join(' · ');
  const instanseg=settings.method==='instanseg', n=pixelNumber(modelPixelSize);
  const title=instanseg?STAIN_TEXT.resolutionInstanseg.replace('{n}',n):settings.diameter>0?STAIN_TEXT.resolutionDiameter:STAIN_TEXT.autoHelp;
  const warning=[cellPx!==null && cellPx<10?STAIN_TEXT.resolutionSmall.replace('{n}',Math.round(cellPx)):'',
    instanseg && !pixels && pixel>modelPixelSize?STAIN_TEXT.resolutionEnlarged.replace('{n}',n):''].filter(Boolean).join(' ');
  return {text,warning,title,warningTitle:STAIN_TEXT.resolutionSmallHelp,cellPx,label:choice.label};
}

export function resolutionRegionProblem(image, level, box) {
  const choice=resolutionChoices(image).find(s=>s.level===(level??0));
  if(!choice || !box)return null;
  const [x0,y0,x1,y1]=box, pixels=Math.min((x1-x0)/choice.um_per_px,(y1-y0)/(choice.um_per_px_y || choice.um_per_px));
  return pixels<32?problem('stain-level',STAIN_TEXT.resolution,
    PICKER_TEXT.resolutionRegion.replace('{n}',Math.floor(pixels)).replace('{label}',choice.label)):null;
}

export const stainProgress = run => `${run.progress?.done || 0} / ${run.progress?.total || 0}`;
export function stainSizes(run) {
  const a = run.summary?.area_um2;
  return a ? `${a.median.toFixed(1)} µm² middle; ${a.p10.toFixed(1)}–${a.p90.toFixed(1)} µm² for the middle 80%` : '—';
}
export function stainStatus(run) {
  if (run.stain_started === false && run.state === 'paused') return STAIN_TEXT.waiting;
  if (run.state === 'failed') return STAIN_TEXT.failed;
  if (run.state === 'finishing') return STAIN_TEXT.stopping;
  if (run.state === 'stopped') return STAIN_TEXT.partial;
  if (run.state === 'done') return STAIN_TEXT.complete;
  if (run.state === 'preparing') return STAIN_TEXT.preparing;
  if (run.message) return run.message;
  if (run.paused && run.ack?.state !== 'paused') return STAIN_TEXT.pausing;
  if (!run.paused && run.ack?.state === 'paused') return STAIN_TEXT.resuming;
  return run.paused ? STAIN_TEXT.paused : STAIN_TEXT.running;
}
export function stainValues(read) {
  if (read('stain-method') === 'instanseg') return {
    method: 'instanseg', model: read('stain-model') || 'fluorescence_nuclei_and_cells',
    image: read('stain-image'), level: Number(read('stain-level') || 0), channels: read('stain-channels'),
    output: read('stain-output'), resolve_cells: read('stain-resolve'), tissue_only: read('stain-tissue') !== false,
  };
  return { method: 'cellpose', model: read('stain-model') || 'cpsam', image: read('stain-image'), level: Number(read('stain-level') || 0), nucleus: read('stain-nucleus'),
    cytoplasm: read('stain-cytoplasm') || null, diameter: read('stain-auto') ? null : Number(read('stain-diameter')),
    flow_threshold: Number(read('stain-flow')), cellprob_threshold: Number(read('stain-probability')),
    tissue_only: read('stain-tissue') !== false,
    niter: read('stain-niter') ? Number(read('stain-niter')) : null,
    min_size: Number(read('stain-min-size') ?? 15), augment: !!read('stain-augment'),
    // A fixed range only counts while Adjust brightness is on (as the percentiles): off, its fields are disabled and
    // kept as typed, so a range left there never blocks Start (Cellpose ignores lowhigh without normalisation).
    normalize: {normalize: read('stain-normalize') !== false,
      lowhigh: read('stain-normalize') !== false && read('stain-fixed-range') ? ['stain-intensity-low','stain-intensity-high'].map(id=>
        read(id)?.trim() ? Number(read(id)) : NaN) : null,
      percentile: read('stain-normalize') === false || read('stain-fixed-range') ? null :
      [Number(read('stain-low') ?? 1), Number(read('stain-high') ?? 99)],
      tile_norm_blocksize: Number(read('stain-tile-norm') ?? 0),
      sharpen_radius: Number(read('stain-sharpen') ?? 0), smooth_radius: Number(read('stain-smooth') ?? 0),
      invert: !!read('stain-invert')} };
}
export function validStain(value, images, models) {
  const n=value.normalize, finite=v=>Number.isFinite(v) && v>=0;
  if(value.tissue_only!==undefined && typeof value.tissue_only!=='boolean')return false;
  if(value.method!=='instanseg' && (
    value.niter!=null && (!Number.isInteger(value.niter) || value.niter<1) ||
    value.min_size!==undefined && (!Number.isInteger(value.min_size) || value.min_size < -1 || value.min_size>100000) ||
    n && (n.normalize!==undefined && typeof n.normalize!=='boolean' ||
      n.lowhigh!=null && !(Array.isArray(n.lowhigh) && n.lowhigh.length===2 &&
        n.lowhigh.every(v=>Number.isFinite(v) && v>=-1e9 && v<=1e9) && n.lowhigh[0]<n.lowhigh[1]) ||
      !(n.percentile===null || Array.isArray(n.percentile) && n.percentile.length===2 &&
      n.percentile.every(finite) && n.percentile[0]<n.percentile[1] && n.percentile[1]<=100) ||
      !Number.isInteger(n.tile_norm_blocksize) || !finite(n.tile_norm_blocksize) ||
      !finite(n.sharpen_radius) || !finite(n.smooth_radius))))return false;
  const image = images?.find(i => i.key === value.image);
  if(image && !resolutionChoices(image).some(s=>s.level===(value.level??0)))return false;
  if (value.method === 'instanseg') {
    const dual = value.model === 'fluorescence_nuclei_and_cells', channels = value.channels;
    return (!models || models.some(m => m.name === value.model)) && !!image &&
      Array.isArray(channels) && channels.length > 0 && channels.every(c => image.channels.includes(c)) &&
      new Set(channels).size === channels.length &&
      (value.model !== 'single_channel_nuclei' || channels.length === 1) &&
      (value.model !== 'brightfield_nuclei' || channels.length === 3) &&
      ['cells','nuclei'].includes(value.output) && (dual || value.output === 'nuclei') &&
      typeof value.resolve_cells === 'boolean' && (dual || !value.resolve_cells);
  }
  if (value.method && value.method !== 'cellpose') return false;
  return (!models || models.some(m => m.name === value.model)) && !!image && image.channels.includes(value.nucleus) &&
    (value.cytoplasm === null || image.channels.includes(value.cytoplasm) && value.cytoplasm !== value.nucleus) &&
    (value.diameter === null || Number.isFinite(value.diameter) && value.diameter >= 10 && value.diameter <= 1000) &&
    Number.isFinite(value.flow_threshold) && value.flow_threshold >= 0 && value.flow_threshold <= 10 &&
    Number.isFinite(value.cellprob_threshold) && value.cellprob_threshold >= -20 && value.cellprob_threshold <= 20;
}

/** What validStain() rejects, field by field in the picker's order: explain.mjs problems naming each field by its label
 *  (empty exactly when validStain() accepts). `pair`: InstanSeg reads its channels from the nucleus and cell body fields
 *  (Training patches, except brightfield). */
export function stainProblems(value, images, models, {pair = false} = {}) {
  const T=STAIN_TEXT, P=PICKER_TEXT, out=[], add=(field,label,message)=>out.push(problem(field,label,message));
  const n=value.normalize, finite=v=>Number.isFinite(v) && v>=0, image=images?.find(i=>i.key===value.image);
  if(value.method && !['cellpose','instanseg'].includes(value.method)) { add('stain-method',T.method,P.method); return out; }
  if(models && !models.some(m=>m.name===value.model))add('stain-model',T.model,models.length?P.model:P.noModels);
  if(!image)add('stain-image',T.image,images?.length?P.image:P.noImage);
  if(image && !resolutionChoices(image).some(s=>s.level===(value.level??0)))add('stain-level',T.resolution,P.resolution);
  if(value.tissue_only!==undefined && typeof value.tissue_only!=='boolean')add('stain-tissue',T.tissue,P.tick);
  if(value.method==='instanseg') {
    const dual=value.model==='fluorescence_nuclei_and_cells', single=value.model==='single_channel_nuclei';
    const bright=value.model==='brightfield_nuclei', channels=Array.isArray(value.channels)?value.channels:[];
    const unknown=channels.findIndex(c=>!image?.channels.includes(c)), repeat=new Set(channels).size!==channels.length;
    if(image && pair && !bright) {
      if(!channels.length || unknown===0)add('stain-nucleus',T.nucleus,P.channel);
      else if(unknown>0)add('stain-cytoplasm',T.cytoplasm,P.channelOrNone);
      else if(repeat)add('stain-cytoplasm',T.cytoplasm,P.otherChannel);
      else if(single && channels.length!==1)add('stain-cytoplasm',T.cytoplasm,P.oneChannel);
    } else if(image) {
      if(!channels.length || unknown>=0 || repeat)add('stain-channels',T.channels,P.channels);
      else if(single && channels.length!==1)add('stain-channels',T.channels,P.exactlyOne);
      else if(bright && channels.length!==3)add('stain-channels',T.channels,P.threeChannels);
    }
    if(!['cells','nuclei'].includes(value.output) || !dual && value.output!=='nuclei')add('stain-output',T.output,dual?P.output:P.nucleiOnly);
    if(typeof value.resolve_cells!=='boolean' || !dual && value.resolve_cells)
      add('stain-resolve',T.resolve,typeof value.resolve_cells==='boolean'?P.resolve:P.tick);
    return out;
  }
  if(image && !image.channels.includes(value.nucleus))add('stain-nucleus',T.nucleus,P.channel);
  if(image && value.cytoplasm!==null && (!image.channels.includes(value.cytoplasm) || value.cytoplasm===value.nucleus))
    add('stain-cytoplasm',T.cytoplasm,value.cytoplasm===value.nucleus?P.otherChannel:P.channelOrNone);
  if(value.diameter!==null && !(Number.isFinite(value.diameter) && value.diameter>=10 && value.diameter<=1000))
    add('stain-diameter',T.diameter,P.diameter);
  if(!(Number.isFinite(value.flow_threshold) && value.flow_threshold>=0 && value.flow_threshold<=10))add('stain-flow',T.flow,P.flow);
  if(!(Number.isFinite(value.cellprob_threshold) && value.cellprob_threshold>=-20 && value.cellprob_threshold<=20))
    add('stain-probability',T.probability,P.probability);
  if(value.niter!=null && !(Number.isInteger(value.niter) && value.niter>=1))add('stain-niter',T.niter,P.niter);
  if(value.min_size!==undefined && !(Number.isInteger(value.min_size) && value.min_size>=-1 && value.min_size<=100000))
    add('stain-min-size',T.minSize,P.minSize);
  if(n) {
    if(n.normalize!==undefined && typeof n.normalize!=='boolean')add('stain-normalize',T.normalize,P.tick);
    if(n.lowhigh!=null) {
      const [low,high]=Array.isArray(n.lowhigh) && n.lowhigh.length===2?n.lowhigh:[NaN,NaN], ok=v=>Number.isFinite(v) && Math.abs(v)<=1e9;
      if(!ok(low))add('stain-intensity-low',T.intensityLow,P.number);
      if(!ok(high))add('stain-intensity-high',T.intensityHigh,P.number);
      else if(ok(low) && low>=high)add('stain-intensity-high',T.intensityHigh,P.aboveIntensity);
    }
    if(n.percentile!==null) {
      const [low,high]=Array.isArray(n.percentile) && n.percentile.length===2?n.percentile:[NaN,NaN];
      const lowBad=!(finite(low) && low<100), highBad=!(finite(high) && high>0 && high<=100);
      if(lowBad)add('stain-low',T.low,P.percent);
      if(highBad)add('stain-high',T.high,P.percent);
      else if(!lowBad && low>=high)add('stain-high',T.high,P.abovePercent);
    }
    if(!Number.isInteger(n.tile_norm_blocksize) || !finite(n.tile_norm_blocksize))add('stain-tile-norm',T.tileNorm,P.tileNorm);
    if(!finite(n.sharpen_radius))add('stain-sharpen',T.sharpen,P.zeroOrMore);
    if(!finite(n.smooth_radius))add('stain-smooth',T.smooth,P.zeroOrMore);
  }
  return out;
}

/** The largest patch side, in pixels of `image`, that fits a region w × h µm across (as the picker checks the size). */
export function largestPatch(w, h, image, level=0) {
  image=resolutionChoices(image).find(s=>s.level===level) || image;
  const sx=image?.um_per_px || 1, sy=image?.um_per_px_y || image?.um_per_px || 1;
  let side=Math.floor(Math.min(w/sx,h/sy));
  while(side>0 && (side*sx>w || side*sy>h))side--;
  return side>0?side:0;
}

/** Whether a Training patches box [x0, y0, x1, y1] lies inside its region, in the frame the picker's preview gave (main.js):
 *  within the region's image window `extent` [w, h] and, for a lasso, its `outline` ([[x, y], …] in the same pixels) by the
 *  rule the panel's gestures keep (region.mjs boxInPolygon: a corner outside the lasso at most a pixel from its edge, as
 *  the engine allows). A box equal to one of `suggested` (the engine's own boxes for this frame) is not checked again: the
 *  engine accepted it. A frame that is not known checks nothing (the engine still does). */
export function boxInRegion(box, {extent=null, outline=null, suggested=null}={}) {
  const [x0,y0,x1,y1]=box;
  if(suggested?.some(s=>s.length===4 && s.every((v,i)=>v===box[i])))return true;
  if(!(x0<x1 && y0<y1))return false;
  if(extent && !(x0>=0 && y0>=0 && x1<=extent[0] && y1<=extent[1]))return false;
  return !(outline?.length>=3) || boxInPolygon(box,outline,1);
}

export function stainPreviewImage(meta, buffer) {
  if (buffer.byteLength !== meta.width*meta.height*2) throw Error(STAIN_TEXT.previewError);
  const data = new Uint16Array(buffer), values=[];
  for(let i=0;i<data.length;i+=Math.max(1,Math.ceil(data.length/32768)))values.push(data[i]);
  values.sort((a,b)=>a-b);
  return { rect:{x0:meta.x0,y0:meta.y0,w:meta.x1-meta.x0,h:meta.y1-meta.y0},
    channels:[{source:{data,width:meta.width,height:meta.height},on:true,depth:16,colormap:'gray',blending:'additive',
      low:values[Math.floor(values.length*.01)]||0,high:Math.max(1,values[Math.floor(values.length*.99)]||1),gamma:1,opacity:1}] };
}

/** Viewport-sized binary outlines; stale requests cannot replace a later tile or another run. */
export const VECTORS_RETRY_MS = 500, VECTORS_RETRIES = 5;   // outlines read while their cells changed (409): asked again
export class StainView {
  constructor(stage, priors, tiled, error, onVectors=()=>{}) {
    this.stage = stage; this.priors = priors; this.tiled = tiled; this.error = error;
    this.onVectors=onVectors;
    this.requests = new ViewRequests();
  }
  reset(run = null) {
    this.requests.clear(); clearTimeout(this.timer); this.run = run; this.shown = null; this.failed = null;
    this.vectorsChanged(null);
  }
  /** The outlines' set (null: none) to onVectors, the range choosers' cells: a failure there is its own, never the
   *  outlines' (they stay shown and are not fetched again) nor the caller's. */
  vectorsChanged(data) {
    try { this.onVectors(data); } catch (error) { console.error(error); }
  }
  update(run) { this.run = run; this.refresh(); }
  visible() {
    return this.run && intersect(visibleBox(this.stage.camera, this.stage.canvas.width, this.stage.canvas.height), this.run.box);
  }
  move() {
    if (!this.run) return;
    this.requests.cancel('vectors');
    clearTimeout(this.timer); this.timer = setTimeout(() => this.refresh(), 120);
  }
  refresh() {
    const run = this.run;
    if (!run || typeof document!=='undefined' && document.hidden) return;
    this.tiled.run = run; this.tiled.imageView();
    if(run.kind==='patches')return;
    const view = this.visible();
    const exclusions=run.exclusions?.areas_version || 0;
    if (!view || this.shown?.revision === (run.progress?.revision || 0) && this.shown?.exclusions===exclusions && covers(this.shown.box, view)) {
      this.requests.cancel('vectors');
      return;
    }
    const mx = (view[2]-view[0])*.2, my = (view[3]-view[1])*.2;
    const box = intersect([view[0]-mx, view[1]-my, view[2]+mx, view[3]+my], run.box);
    const revision = run.progress?.revision || 0, key = JSON.stringify([run.id, revision, exclusions, box]);
    const current = () => this.run?.id === run.id && (this.run.progress?.revision || 0) === revision &&
      (this.run.exclusions?.areas_version || 0)===exclusions && covers(box, this.visible());
    if (this.requests.jobs.get('vectors')?.key === key || this.failed === key) return;
    this.requests.run('vectors', key, async signal => {
      const response = await fetch(`/api/v1/runs/${encodeURIComponent(run.id)}/stain/vectors?box=${box.join(',')}&revision=${revision}`, { signal });
      if (!response.ok) throw Object.assign(Error(STAIN_TEXT.vectorsError), { status: response.status });
      return decodeVectors(await response.arrayBuffer());
    }).then(data => {
      if (!current()) return;
      this.priors.vectors = data; this.priors.vectorsVisible = true; this.stage.dirty = true;
      this.shown = { revision, exclusions, areas_version: data.areas_version ?? exclusions, box };
      this.vectorsChanged(data);
    }).catch(error => {
      if (error.name === 'AbortError' || !current()) return;
      // 409: the cells changed while their outlines were read (a running run wrote a tile): read again shortly, unsaid;
      // only what does not settle within VECTORS_RETRIES tries is said.
      const tries = this.conflicts?.key === key ? this.conflicts.tries + 1 : 1;
      if (error.status === 409 && tries <= VECTORS_RETRIES) {
        this.conflicts = { key, tries }; clearTimeout(this.timer); this.timer = setTimeout(() => this.refresh(), VECTORS_RETRY_MS);
        return;
      }
      this.failed = key; this.error(STAIN_TEXT.vectorsError);
    });
  }
}
