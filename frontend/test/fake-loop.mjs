// A deterministic browser event loop for render-scheduling tests: virtual time, timers, and animation frames on a
// 60 Hz display. Browser globals are swapped in only while `run` executes synchronously, then restored.
export function fakeLoop({ hz = 60 } = {}) {
  const loop = { now: 0, period: 1000 / hz, rafs: [], timers: [], frames: 0, nextId: 1, hidden: false, listeners: new Map() };
  const document = {
    get hidden() { return loop.hidden; },
    addEventListener(type, fn) { loop.listeners.set(type, [...(loop.listeners.get(type) || []), fn]); },
    createElement: () => fakeElement(), body: { append() {} },
  };
  const globals = {
    performance: { now: () => loop.now },
    requestAnimationFrame: cb => { const id = loop.nextId++; loop.rafs.push({ id, cb }); return id; },
    cancelAnimationFrame: id => { loop.rafs = loop.rafs.filter(r => r.id !== id); },
    setTimeout: (cb, ms = 0) => { const id = loop.nextId++; loop.timers.push({ id, cb, at: loop.now + Math.max(0, ms) }); return id; },
    clearTimeout: id => { loop.timers = loop.timers.filter(t => t.id !== id); },
    document,
    window: { devicePixelRatio: 1, innerWidth: 1920, addEventListener() {} },
    ResizeObserver: class { constructor(cb) { loop.resize = cb; } observe() {} },
    matchMedia: () => ({ addEventListener() {} }),
  };
  /** Runs `fn` with the fake globals installed (everything stays synchronous). */
  loop.with = (fn) => {
    const saved = Object.fromEntries(Object.keys(globals).map(k => [k, Object.getOwnPropertyDescriptor(globalThis, k)]));
    for (const [k, v] of Object.entries(globals)) Object.defineProperty(globalThis, k, { value: v, configurable: true, writable: true });
    try { return fn(); } finally {
      for (const [k, d] of Object.entries(saved)) { if (d) Object.defineProperty(globalThis, k, d); else delete globalThis[k]; }
    }
  };
  /** `fn` runs at virtual time `t` (as a timer would). */
  loop.at = (t, fn) => { loop.timers.push({ id: loop.nextId++, cb: fn, at: t }); };
  /** Advance to `until`, firing due timers and, at each vsync, the animation frames requested before it. */
  loop.run = (until) => loop.with(() => {
    for (;;) {
      const timer = loop.timers.reduce((a, b) => (!a || b.at < a.at ? b : a), null);
      const vsync = loop.rafs.length ? (Math.floor(loop.now / loop.period + 1e-9) + 1) * loop.period : Infinity;
      const next = Math.min(timer ? timer.at : Infinity, vsync);
      if (next > until) { loop.now = until; return; }
      if (timer && timer.at <= vsync) {
        loop.now = Math.max(loop.now, timer.at);
        loop.timers = loop.timers.filter(t => t !== timer);
        timer.cb();
      } else {
        loop.now = vsync; loop.frames++;
        const due = loop.rafs; loop.rafs = [];
        for (const r of due) r.cb(vsync);
      }
    }
  });
  loop.setHidden = (hidden) => loop.with(() => { loop.hidden = hidden; for (const fn of loop.listeners.get('visibilitychange') || []) fn(); });
  loop.idle = () => !loop.rafs.length && !loop.timers.length;
  return loop;
}

/** Just enough of an element for the timeline's readout and canvas. */
export function fakeElement(extra = {}) {
  const listeners = {};
  return {
    hidden: false, style: {}, textContent: '', width: 0, height: 0, clientWidth: 0, clientHeight: 0, offsetWidth: 120, offsetHeight: 40,
    listeners, addEventListener(type, fn) { listeners[type] = fn; }, setPointerCapture() {},
    getContext: () => new Proxy({}, { get: (t, k) => k in t ? t[k] : k === 'measureText' ? () => ({ width: 50 }) :
      k === 'createLinearGradient' ? () => ({ addColorStop() {} }) : () => {} }),
    ...extra,
  };
}
