// Where a flagged area of the mixing check lies. The engine still flags areas (their
// outlines in µm, at most 64 points, in each result's summary.areas): its suggestions and trials are about them. None
// is drawn: no outline, letter chip or motion on the stage. The page only needs where one lies: to move the view there
// (a card's "Look at its cells", R1's "Show the outlines of ... here") and to find its most mixed cell for Go deeper
// (main.js qcInspect).

/** Even-odd point in polygon. */
export function pointInOutline(x, y, points) {
  let inside = false;
  for (let i = 0, j = points.length - 1; i < points.length; j = i++) {
    const [xi, yi] = points[i], [xj, yj] = points[j];
    if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}
/** [x0, y0, x1, y1] of the outline. */
export function outlineBounds(points) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const [x, y] of points) { x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y); }
  return [x0, y0, x1, y1];
}
/** Where to fly the camera for an area (stage.fit's rect {x0, y0, w, h}): its outline's box, at least `minSide` µm
 *  across (enough to see its cells) and `pad` of that again around it. */
export function flyTarget(points, { minSide = 300, pad = .15 } = {}) {
  const [x0, y0, x1, y1] = outlineBounds(points), cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
  const w = Math.max(x1 - x0, minSide) * (1 + 2 * pad), h = Math.max(y1 - y0, minSide) * (1 + 2 * pad);
  return { x0: cx - w / 2, y0: cy - h / 2, w, h };
}
/** The outline of `area` ([[x, y] µm, ...]): its own, else that of the area of the same letter in `areas` (the result
 *  shown's summary.areas, as qc-client.mjs's view has them), its finite points only; null with fewer than 3. */
export function areaOutline(area, areas = []) {
  const own = Array.isArray(area?.outline) && area.outline.length ? area.outline
    : (Array.isArray(areas) ? areas : []).find(a => a && area?.L != null && a.L === area.L)?.outline;
  const points = (Array.isArray(own) ? own : []).filter(p => Number.isFinite(p?.[0]) && Number.isFinite(p?.[1]));
  return points.length >= 3 ? points : null;
}
