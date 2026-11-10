// Browser memory in a notebook's session opened as a link in the user's own browser. Loaded
// by index.html before the page, so nothing has read storage yet. A session's link carries `api` (its session id, or
// 1): from then on, in that tab, the page's `localStorage` is the tab's sessionStorage under keys of that session alone,
// so nothing persists past the tab and a later session that happens to get the same port sees nothing of this one. The
// tab remembers the session (a reload keeps it, though the page's own address no longer names it). Without `api` the
// browser's localStorage is the page's, as on the desktop (a session's own window has a throwaway profile anyway),
// where the previous version's memory is carried over first (copyLegacyKeys).
// No imports: it runs before anything else.

export const MARK = 'sopa-companion:api-session';
// The previous version of the app kept its memory under keys beginning 'proseg-live:' and 'proseg-live-', this one
// under 'sopa-companion' and the same rest. COPIED says the previous version's keys were carried over in this profile.
export const LEGACY = 'proseg-live';
export const PREFIX = 'sopa-companion';
export const COPIED = 'sopa-companion:storage-from-proseg-live';

/** The previous version's keys in `store` copied once to their twins (the value an old key holds goes to its twin when
 *  the twin has none; the old keys stay), so remembered panels, models and choices survive the app's new name. Marked
 *  done (COPIED) only once every key was taken: storage that refuses (full, blocked) leaves the rest for the next start.
 *  Returns how many keys were copied (-1: storage refused). */
export function copyLegacyKeys(store) {
  try {
    if (!store || store.getItem(COPIED) != null) return 0;
    const keys = [];
    for (let i = 0; i < store.length; i++) {
      const key = store.key(i);
      if (key?.startsWith(LEGACY + ':') || key?.startsWith(LEGACY + '-')) keys.push(key);
    }
    let copied = 0;
    for (const key of keys) {
      const twin = PREFIX + key.slice(LEGACY.length);
      const value = store.getItem(key);
      if (value != null && store.getItem(twin) == null) { store.setItem(twin, value); copied += 1; }
    }
    store.setItem(COPIED, '1');
    return copied;
  } catch { return -1; }
}

/** The session this tab belongs to: `?api=<id>` (1 for an unnamed one), else what the tab remembered; null otherwise. */
export function sessionId(search, store) {
  let id = null;
  try { id = new URLSearchParams(search || '').get('api'); } catch { id = null; }
  if (id != null && id !== '') return id === '1' || !/^[\w-]{1,64}$/.test(id) ? 'api' : id;
  try { return store?.getItem(MARK) || null; } catch { return null; }
}

/** Storage over `backing` whose keys all start with `prefix` (the Storage interface the page uses). */
export class PrefixedStorage {
  constructor(backing, prefix) { this.backing = backing; this.prefix = prefix; }
  own() { const keys = []; for (let i = 0; i < this.backing.length; i++) { const k = this.backing.key(i); if (k?.startsWith(this.prefix)) keys.push(k); } return keys; }
  get length() { return this.own().length; }
  key(i) { return this.own()[i]?.slice(this.prefix.length) ?? null; }
  getItem(key) { return this.backing.getItem(this.prefix + key); }
  setItem(key, value) { this.backing.setItem(this.prefix + key, String(value)); }
  removeItem(key) { this.backing.removeItem(this.prefix + key); }
  clear() { for (const key of this.own()) this.backing.removeItem(key); }
}

/** A Storage kept in memory (a tab whose sessionStorage refuses): the page loses nothing it needs. */
export class MemoryStorage {
  constructor() { this.map = new Map(); }
  get length() { return this.map.size; }
  key(i) { return [...this.map.keys()][i] ?? null; }
  getItem(key) { return this.map.has(key) ? this.map.get(key) : null; }
  setItem(key, value) { this.map.set(String(key), String(value)); }
  removeItem(key) { this.map.delete(key); }
  clear() { this.map.clear(); }
}

/** In a session's tab: `win.localStorage` becomes the session's own storage (returned); null elsewhere, where the
 *  previous version's keys are carried over to this version's (copyLegacyKeys). */
export function bootStorage(win = globalThis) {
  let tab = null;
  try { tab = win.sessionStorage; } catch { tab = null; }
  const id = sessionId(win.location?.search, tab);
  if (!id) {
    let local = null;
    try { local = win.localStorage; } catch { local = null; }
    copyLegacyKeys(local);
    return null;
  }
  let backing = tab;
  try { backing.setItem(MARK, id); } catch { backing = new MemoryStorage(); }
  const store = new PrefixedStorage(backing, `sc:${id}:`);
  try { Object.defineProperty(win, 'localStorage', { configurable: true, enumerable: true, get: () => store }); }
  catch { return null; }
  return store;
}

bootStorage();
