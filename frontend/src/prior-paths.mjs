import { gridIndex, visibleCells } from './polygons.mjs';

// The path PriorLayers strokes for one vector layer: the polygons near the view, as one Path2D, stroked once.
//
// One stroke per layer is what the old code drew, and it has to stay one: a single path's stroke covers the union of its
// outlines, so the edge two neighbouring cells share is drawn once. Stroking each polygon on its own blends shared edges
// twice on the GPU canvas (0.3-1.0 % of the pixels of a detail view off by up to 58 levels); stroking the nuclei per
// polygon differs the same way on the software canvas. What made the old path slow was building it: Chrome's
// closePath() costs a copy of the whole path it is called on, so closing 41k subpaths of one growing path took ~26 s.
// Here closePath() is only ever called on a one-polygon Path2D, which is appended (linear) to its grid cell's tile;
// tiles are built once per PolygonSet, on first use, and a view appends the tiles it meets (~2 ms for 8k polygons).
//
// Tiles are native path memory that V8 does not see, so each set keeps at most TILE_BUDGET vertices of them, the least
// recently used going first (a dense whole sample's millions of vertices would otherwise be copied once more as the
// user pans across it), and PriorLayers releases a set's tiles as soon as the set is replaced (the picker makes a set
// per box).
//
// Far out, a tile whose polygons are all smaller than LOD_PX device px is drawn as its polygons' bounding boxes: the
// stroke of a sub-pixel polygon is a blob of about (size + line width)², which its box's stroke reproduces with 4
// vertices instead of ~25. No camera the app reaches with outlines shown gets there except a small run zoomed far out.
export const LOD_PX = 1;
export const TILE_BUDGET = 2_000_000;   // vertices per set (~20 MB of path memory)
export const FILL_SHARE = .3;           // a filled segmentation layer's inside: this share of its opacity, under its outlines
const TILES = new WeakMap();   // PolygonSet -> {paths: Map(tile key -> {path, vertices}) least recently used first, vertices, view}
const MARGIN_PX = 16;   // Polygons this close outside the canvas are stroked too: the GPU canvas rasterises strokes
                        // along the edge differently when the path ends just outside it (2 px left 74 px of a detail
                        // view off by up to 17 levels; 16 px none).

/** One closed subpath per polygon of grid cell k, in index order (box outlines with `boxes`): {path, vertices}. */
function tilePath(set, index, k, boxes, Path, classes) {
  const tile = new Path(), { offsets, xy, bbox } = set;
  let vertices = 0, polygons = 0;
  for (let m = index.starts[k]; m < index.starts[k + 1]; m++) {
    const i = index.members[m], a = offsets[i], b = offsets[i + 1];
    if(hidden(set,i,classes))continue;
    if (b <= a) continue;
    polygons++;
    const p = new Path();
    if (boxes) {
      const x0 = bbox[4 * i], y0 = bbox[4 * i + 1], x1 = bbox[4 * i + 2], y1 = bbox[4 * i + 3];
      p.moveTo(x0, y0); p.lineTo(x1, y0); p.lineTo(x1, y1); p.lineTo(x0, y1);
      vertices += 4;
    } else {
      p.moveTo(xy[2 * a], xy[2 * a + 1]);
      for (let j = a + 1; j < b; j++) p.lineTo(xy[2 * j], xy[2 * j + 1]);
      vertices += b - a;
    }
    p.closePath();
    tile.addPath(p);
  }
  return { path: tile, vertices, polygons };
}

// Stain ids are one-based over the entire cell set, even in a viewport or after area removal. Unknown ids stay shown.
const hidden = (set,i,classes) => {
  const id=set.ids?.[i];
  return Number.isInteger(id) && id>0 && id<=classes?.length && classes[id-1]!==0;
};
function signature(set,index,k,classes) {
  if(!classes)return '';
  let bits='';
  for(let m=index.starts[k];m<index.starts[k+1];m++)bits+=hidden(set,index.members[m],classes)?'1':'0';
  return bits.includes('1')?bits:'';
}

/** The tile of grid cell k (its box outlines with `far`), the set's tiles kept within `budget` vertices. */
function tile(set, index, tiles, k, far, Path, budget, classes) {
  const key = far ? -1 - k : k;
  let entry = tiles.paths.get(key);
  const sig=entry?.stamp===tiles.stamp?entry.signature:signature(set,index,k,classes);
  if(entry && entry.signature!==sig){tiles.vertices-=entry.vertices;tiles.paths.delete(key);entry=null;}
  if (entry) tiles.paths.delete(key);                     // used now: last in line to go
  else {
    entry = {...tilePath(set, index, k, far, Path, classes),signature:sig};
    tiles.vertices += entry.vertices;
    for (const [old, kept] of tiles.paths) {              // the least recently used first; never the one just made
      if (tiles.vertices <= budget) break;
      tiles.paths.delete(old); tiles.vertices -= kept.vertices;
    }
  }
  tiles.paths.set(key, entry);
  entry.stamp=tiles.stamp;
  return entry;
}

/** The path of the polygons of `set` that may show in the world box `box` at `zoom` device px per µm: { path, polygons
 *  (how many it holds), boxes (how many of those are drawn as their box) }. `Path`: the Path2D constructor. `reuse`
 *  (a drawing while the camera moves): the last path made for the same tiles is returned again instead of a copy (a
 *  path per animation frame would pile up native memory); a settled view always gets a path of its own. `budget`:
 *  the vertices of tiles the set keeps. */
export function layerPath(set, box, zoom, Path = globalThis.Path2D, reuse = false, budget = TILE_BUDGET, classes = null) {
  const index = gridIndex(set);
  let tiles = TILES.get(set);
  if (!tiles) TILES.set(set, tiles = { paths: new Map(), vertices: 0, view: null });
  if(tiles.classes!==classes){tiles.classes=classes;tiles.stamp=(tiles.stamp||0)+1;}
  const cells = visibleCells(index, box), far = cells.map(k => index.maxSide[k] * zoom < LOD_PX);
  const key = cells.map((k, i) => far[i] ? -1 - k : k).join();
  if (reuse && tiles.view?.key === key && tiles.view.Path === Path && tiles.view.classes===classes) return tiles.view.result;
  const path = new Path();
  let polygons = 0, boxes = 0;
  cells.forEach((k, i) => {
    const entry=tile(set, index, tiles, k, far[i], Path, budget, classes),n=entry.polygons;
    path.addPath(entry.path);
    polygons += n; if (far[i]) boxes += n;
  });
  const result = { path, polygons, boxes };
  tiles.view = { key, Path, result, classes };
  return result;
}

/** Forget a set's tiles (and its last path) now. */
export function releaseTiles(set) { TILES.delete(set); }

/** The same canvas commands on the main thread and in the picker worker. Each layer remains ONE union stroke,
 *  with the same path tiles, culling margin, transform, order and style. No readback or context options here: those
 *  can change the canvas rasteriser. paintMs covers path construction and stroke submission, as the harness did.
 *  `layers`: segmentation layers (seg-layers.mjs) [{set, color, opacity, width, fill}], front to back as Layers lists
 *  them, painted before the run's own (under them) and the last listed first; each is one path of its polygons near the
 *  view as well: filled first when `fill` (at FILL_SHARE of its opacity: one fill of the union, so a cell is tinted
 *  once), then its one union stroke. Without them the commands are exactly the run's own, as before. */
export function paintOutlines(canvas, view, vectors, settings, { picker = false, visible = true, moving = false, layers = [], classes = null } = {}) {
  canvas.width = view.width; canvas.height = view.height;
  const g = canvas.getContext('2d'), z = view.zoom;
  g.setTransform(z, 0, 0, z, canvas.width / 2 - view.x * z, canvas.height / 2 - view.y * z);
  const x0 = view.x - view.width / 2 / z, y0 = view.y - view.height / 2 / z;
  const x1 = view.x + view.width / 2 / z, y1 = view.y + view.height / 2 / z;
  const start = performance.now(), names = [];
  let polygons = 0;
  for (let k = layers.length - 1; k >= 0; k--) {
    const { set, color, opacity, width, fill } = layers[k];
    if (!set?.count) continue;
    const m = (width * view.ratio / 2 + MARGIN_PX) / z;
    const { path, polygons: n } = layerPath(set, [x0 - m, y0 - m, x1 + m, y1 + m], z, globalThis.Path2D, moving);
    if (n && fill) { g.fillStyle = color; g.globalAlpha = opacity * FILL_SHARE; g.fill(path); }
    g.strokeStyle = color; g.globalAlpha = opacity; g.lineWidth = width * view.ratio / z;
    if (n) g.stroke(path);
    polygons += n;
    if (!names.length) names.push('segmentations');
  }
  for (const [key, set] of [['xenium', vectors?.cells], ['nuclei', vectors?.nuclei]]) {
    if (!set?.count) continue;
    const o = settings[key];
    if (!visible || (!o.on && !picker)) continue;
    g.strokeStyle = o.color; g.globalAlpha = o.opacity; g.lineWidth = o.width * view.ratio / z;
    const m = (o.width * view.ratio / 2 + MARGIN_PX) / z;
    const array=classes?.set!=null && classes.set===vectors?.set?classes.array:null;
    const { path, polygons: n } = layerPath(set, [x0 - m, y0 - m, x1 + m, y1 + m], z, globalThis.Path2D, moving, TILE_BUDGET, array);
    if (n) g.stroke(path);
    polygons += n; names.push(key);
  }
  return { polygons, layer: names.join('+'), paintMs: performance.now() - start };
}
