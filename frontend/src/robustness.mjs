// Shared, DOM-free rules for rendering limits and effective live settings.
export function fitSize(width, height, limit) {
  const scale = Math.min(1, limit / width, limit / height);
  return { width: Math.max(1, Math.floor(width * scale)), height: Math.max(1, Math.floor(height * scale)) };
}
export function liveValue(run, option) {
  const flag = option.flag;
  const effective = run?.last?.live?.[flag] ?? run?.options?.[flag] ?? option.default;
  const override = run?.live_values && Object.hasOwn(run.live_values, flag);
  const value = override ? run.live_values[flag] : effective;
  return { value, effective, pending: !!override && run?.last?.live?.[flag] !== value,
    automatic: flag === 'temperature' && !override && run?.options?.[flag] == null };
}
export function rasterSupported(frame, limits) {
  return frame.nx > 0 && frame.ny > 0 && frame.nx <= limits.texture && frame.ny <= limits.texture &&
    (frame.layers?.length || frame.nz || 0) <= limits.layers;
}
