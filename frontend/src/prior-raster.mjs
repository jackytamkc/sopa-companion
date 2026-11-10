import { paintOutlines, releaseTiles } from './prior-paths.mjs';
import { releaseIndex } from './polygons.mjs';

/** Worker-owned picker canvas. Keep the canvas and path tiles between views, just as PriorLayers does. The same worker
 *  draws the segmentation layers' raster (PriorLayers' SegRaster, one worker of its own): `layers` [{id, color, opacity,
 *  width, fill}] front to back, their sets by id; a set comes once, in `sets`, the first time it is drawn here, and one
 *  no longer listed is let go (its path tiles and grid with it). */
export class PriorRaster {
  constructor() {
    this.canvas = new OffscreenCanvas(1, 1);
    this.vectors = null;
    this.classes = null;
    this.sets = new Map();
  }
  async draw(request) {
    const start = performance.now();
    // Vectors are cloned once per replacement, never transferred away from the app or copied on every camera step.
    if ('vectors' in request) {
      for (const set of [this.vectors?.cells, this.vectors?.nuclei]) {
        if (set && set !== request.vectors?.cells && set !== request.vectors?.nuclei) { releaseTiles(set); releaseIndex(set); }
      }
      this.vectors = request.vectors;
    }
    if('classes' in request)this.classes=request.classes;
    if (request.layers) {
      const kept = new Map(request.layers.map(({ id }) => [id, request.sets?.[id] ?? this.sets.get(id)]));
      for (const [id, set] of this.sets) if (kept.get(id) !== set) { releaseTiles(set); releaseIndex(set); }
      this.sets = kept;
    }
    const layers = (request.layers || []).map(({ id, ...style }) => ({ ...style, set: this.sets.get(id) }));
    const painted = paintOutlines(this.canvas, request.view, this.vectors, request.settings,
      { picker: request.picker!==false, visible: request.visible, moving: request.moving, layers, classes:this.classes });
    // ImageBitmap uploads ignore WebGL's UNPACK_PREMULTIPLY_ALPHA_WEBGL and colour conversion flags. Prepare the
    // same straight-alpha, unflipped sRGB source as the old canvas upload, here in the worker. In particular, do not
    // transferToImageBitmap() with its default premultiplied alpha, or draw the bitmap through another 2-D canvas.
    const bitmap = await createImageBitmap(this.canvas, { premultiplyAlpha: 'none', colorSpaceConversion: 'none' });
    return { ...painted, bitmap, cost: performance.now() - start };
  }
}
