export function normalizeBox([x0, y0, x1, y1]) {
  return [Math.min(x0, x1), Math.min(y0, y1), Math.max(x0, x1), Math.max(y0, y1)];
}

export function clampPoint([x, y], [x0, y0, x1, y1]) {
  return [Math.max(x0, Math.min(x1, x)), Math.max(y0, Math.min(y1, y))];
}

export function clampBox(box, extent, preserveSize = false) {
  const [x0, y0, x1, y1] = normalizeBox(box);
  if (!preserveSize) return [...clampPoint([x0, y0], extent), ...clampPoint([x1, y1], extent)];
  const w = Math.min(x1 - x0, extent[2] - extent[0]), h = Math.min(y1 - y0, extent[3] - extent[1]);
  const x = Math.max(extent[0], Math.min(extent[2] - w, x0)), y = Math.max(extent[1], Math.min(extent[3] - h, y0));
  return [x, y, x + w, y + h];
}

export function rectangleFromPoints(anchor, point, extent, square = false) {
  const a = clampPoint(anchor, extent), b = clampPoint(point, extent);
  if (square) {
    const sx = b[0] < a[0] ? -1 : 1, sy = b[1] < a[1] ? -1 : 1;
    const size = Math.min(Math.max(Math.abs(b[0] - a[0]), Math.abs(b[1] - a[1])),
      sx > 0 ? extent[2] - a[0] : a[0] - extent[0], sy > 0 ? extent[3] - a[1] : a[1] - extent[1]);
    b[0] = a[0] + sx * size; b[1] = a[1] + sy * size;
  }
  return normalizeBox([...a, ...b]);
}

export function polygonBBox(points) {
  if (!points?.length) return null;
  const box = [Infinity, Infinity, -Infinity, -Infinity];
  for (const [x, y] of points) {
    box[0] = Math.min(box[0], x); box[1] = Math.min(box[1], y);
    box[2] = Math.max(box[2], x); box[3] = Math.max(box[3], y);
  }
  return box;
}

export function polygonArea(points) {
  if (!points?.length) return 0;
  let area = 0;
  const [ox, oy] = points[0];
  for (let i = 0, j = points.length - 1; i < points.length; j = i++) {
    area += (points[j][0] - ox) * (points[i][1] - oy) - (points[i][0] - ox) * (points[j][1] - oy);
  }
  return Math.abs(area) / 2;
}

function distanceSquared(p, a, b) {
  const dx = b[0] - a[0], dy = b[1] - a[1], length = dx * dx + dy * dy;
  const t = length ? Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / length)) : 0;
  return (p[0] - a[0] - t * dx) ** 2 + (p[1] - a[1] - t * dy) ** 2;
}

export function pointInPolygon([x, y], points) {
  if (!points || points.length < 3) return false;
  let inside = false;
  for (let i = 0, j = points.length - 1; i < points.length; j = i++) {
    const a = points[j], b = points[i];
    if (distanceSquared([x, y], a, b) < 1e-18) return true;
    if ((a[1] > y) !== (b[1] > y) && x < (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]) + a[0]) inside = !inside;
  }
  return inside;
}

/** Whether the box [x0, y0, x1, y1] lies inside the polygon, `tolerance` units of slack at its edge allowed: each
 *  corner is inside or within `tolerance` of the outline, and no edge of the outline crosses the box deeper than
 *  `tolerance`. The engine's own test (shapely covers on the region grown by one pixel) agrees for tolerance 1. */
export function boxInPolygon([x0, y0, x1, y1], points, tolerance = 1) {
  if (!points || points.length < 3 || !(x0 < x1 && y0 < y1)) return false;
  const near = p => points.some((a, i) => distanceSquared(p, a, points[(i + 1) % points.length]) <= tolerance ** 2);
  for (const corner of [[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    if (!pointInPolygon(corner, points) && !near(corner)) return false;
  // An outline edge reaching into the box (shrunk by the tolerance) leaves part of it outside.
  const [a0, b0, a1, b1] = [x0 + tolerance, y0 + tolerance, x1 - tolerance, y1 - tolerance];
  if (!(a0 < a1 && b0 < b1)) return true;
  for (let i = 0; i < points.length; i++) {
    const [px, py] = points[i], [qx, qy] = points[(i + 1) % points.length];
    let lo = 0, hi = 1;                                    // Liang–Barsky: the part of this edge inside the box
    for (const [d, s] of [[-(qx - px), px - a0], [qx - px, a1 - px], [-(qy - py), py - b0], [qy - py, b1 - py]]) {
      if (d === 0) { if (s < 0) { lo = 1; hi = 0; break; } continue; }
      const r = s / d;
      if (d < 0) lo = Math.max(lo, r); else hi = Math.min(hi, r);
    }
    if (lo < hi) return false;
  }
  return true;
}

export function simplify(points, tolerance = 1.5) {
  if (points.length < 3) return points.slice();
  const keep = new Uint8Array(points.length), stack = [[0, points.length - 1]], limit = Math.max(0, tolerance) ** 2;
  keep[0] = keep[points.length - 1] = 1;
  while (stack.length) {
    const [first, last] = stack.pop();
    let best = limit, index = -1;
    for (let i = first + 1; i < last; i++) {
      const d = distanceSquared(points[i], points[first], points[last]);
      if (d > best) { best = d; index = i; }
    }
    if (index >= 0) { keep[index] = 1; stack.push([first, index], [index, last]); }
  }
  return points.filter((_, i) => keep[i]);
}

export function simplifyPolygon(points, tolerance = 1.5, maxVertices = 2000) {
  const ring = points.filter((p, i) => !i || p[0] !== points[i - 1][0] || p[1] !== points[i - 1][1]);
  if (ring.length > 1 && ring[0][0] === ring.at(-1)[0] && ring[0][1] === ring.at(-1)[1]) ring.pop();
  if (ring.length < 3) return ring;
  let split = 1;
  for (let i = 2; i < ring.length; i++) if (distanceSquared(ring[i], ring[0], ring[0]) > distanceSquared(ring[split], ring[0], ring[0])) split = i;
  const reduce = t => [...simplify(ring.slice(0, split + 1), t).slice(0, -1), ...simplify([...ring.slice(split), ring[0]], t).slice(0, -1)];
  let result = reduce(tolerance), t = Math.max(tolerance, .001);
  const cap = Math.max(3, maxVertices);
  while (result.length > cap) { t *= 2; result = reduce(t); }
  if (result.length < 3) return [ring[0], ring[Math.floor(ring.length / 3)], ring[Math.floor(2 * ring.length / 3)]];
  return result;
}

export function regionBBox(region, extent) {
  if (!region) return null;
  return region.kind === 'full' ? extent.slice() : region.kind === 'box' ? normalizeBox(region.box) : polygonBBox(region.points);
}

export function regionArea(region, extent) {
  if (region?.kind === 'polygon') return polygonArea(region.points);
  const box = regionBBox(region, extent);
  return box ? (box[2] - box[0]) * (box[3] - box[1]) : 0;
}

export function estimatedTranscripts(transcripts, region, extent) {
  const area = regionArea({ kind: 'full' }, extent);
  return area > 0 ? Math.round(Math.max(0, transcripts) * Math.min(1, regionArea(region, extent) / area)) : 0;
}

export function modeForBBox(box, detailUm = 1200) {
  return box && box[2] - box[0] <= detailUm && box[3] - box[1] <= detailUm ? 'detail' : 'tiled';
}
