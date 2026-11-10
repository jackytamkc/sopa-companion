import { PriorRaster } from './prior-raster.mjs';

const raster = new PriorRaster();
// PriorLayers sends at most one job at a time. Camera/settings changes are coalesced on the main thread, so a pan
// cannot queue a series of expensive drawings or retain a series of vector sets and bitmaps in this worker.
self.onmessage = async ({ data }) => {
  let result;
  try {
    result = await raster.draw(data);
    self.postMessage({ id: data.id, ...result }, [result.bitmap]);
  } catch (error) {
    result?.bitmap.close();
    self.postMessage({ id: data.id, error: String(error?.message || error) });
  }
};
