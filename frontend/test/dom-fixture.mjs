export function domFixture() {
  const document = { activeElement: null, listeners: new Map() };
  class Element {
    constructor(tag) {
      this.tagName = tag.toUpperCase(); this.childNodes = []; this.attributes = new Map(); this.listeners = new Map();
      this.value = ''; this.disabled = this.hidden = this.open = this.checked = false; this.style = {}; this.dataset = {};
      const classes = new Set();
      this.classList = { add: (...c) => c.forEach(x => classes.add(x)), remove: (...c) => c.forEach(x => classes.delete(x)),
        toggle: (c, on = !classes.has(c)) => (on ? classes.add(c) : classes.delete(c), on), contains: c => classes.has(c) };
    }
    get children() { return this.childNodes.filter(c => c.tagName !== '#TEXT'); }
    set textContent(value) { this.text = String(value); this.childNodes = []; }
    get textContent() { return (this.text || '') + this.childNodes.map(child => child.textContent).join(''); }
    set innerHTML(html) {
      this.replaceChildren(); const stack = [this];
      for (const part of html.match(/<[^>]+>|[^<]+/g) || []) {
        if (part.startsWith('</')) { stack.pop(); continue; }
        if (!part.startsWith('<')) { stack.at(-1).append(document.createTextNode(part)); continue; }
        const tag = part.match(/^<([\w-]+)/)?.[1]; if (!tag) continue;
        const node = new Element(tag);
        for (const match of part.matchAll(/([\w-]+)(?:="([^"]*)")?/g)) {
          const [_, key, value] = match; if (key === tag) continue;
          node.setAttribute(key, value ?? '');
          if (['value','type'].includes(key)) node[key] = value ?? '';
          if (['hidden','checked','disabled'].includes(key)) node[key] = true;
        }
        stack.at(-1).append(node);
        if (!['input','br','hr','img','meta','link'].includes(tag)) stack.push(node);
      }
    }
    append(...children) {
      for (let child of children) { if (typeof child !== 'object') child = document.createTextNode(child); child.parentElement = this; this.childNodes.push(child); }
    }
    replaceChildren(...children) { this.text = ''; this.childNodes = []; this.append(...children); }
    setAttribute(key, value) { this.attributes.set(key, String(value)); if (key === 'id') this.id = value; if (key === 'class') this.className = value; if (key === 'data-value') this.dataset.value = value; }
    getAttribute(key) { return this.attributes.has(key) ? this.attributes.get(key) : null; }
    removeAttribute(key) { this.attributes.delete(key); }
    removeEventListener(type, fn) { this.listeners.set(type, (this.listeners.get(type) || []).filter(f => f !== fn)); }
    after(...nodes) {
      const parent = this.parentElement; if (!parent) return;
      const i = parent.childNodes.indexOf(this);
      for (const node of nodes) node.parentElement = parent;
      parent.childNodes.splice(i + 1, 0, ...nodes);
    }
    remove() {
      const parent = this.parentElement; if (!parent) return;
      parent.childNodes.splice(parent.childNodes.indexOf(this), 1); this.parentElement = null;
    }
    get nextSibling() { const p = this.parentElement; return p ? p.childNodes[p.childNodes.indexOf(this) + 1] || null : null; }
    get lastElementChild() { return this.children.filter(c => c.tagName !== '#TEXT').at(-1); }
    get firstChild() { return this.childNodes[0]; }
    get lastChild() { return this.childNodes.at(-1); }
    addEventListener(type, fn) { const list = this.listeners.get(type) || []; list.push(fn); this.listeners.set(type, list); }
    emit(type, extra = {}) {
      const event = { target: this, preventDefault() {}, stopPropagation() {}, ...extra };
      this['on' + type]?.(event);
      for (const fn of this.listeners.get(type) || []) fn(event);
      return event;
    }
    focus() { document.activeElement = this; this.onfocus?.(); }
    scrollIntoView() {}
    contains(node) { return node === this || this.children.some(child => child.contains(node)); }
    closest(tag) { return this.tagName === tag.toUpperCase() ? this : this.parentElement?.closest(tag); }
    showModal() { this.open = true; }
    close() { this.open = false; this.emit('close'); for (const fn of document.listeners.get('close') || []) fn({ target: this }); }
    click() { if (!this.disabled) return this.onclick?.({ target: this, preventDefault() {}, stopPropagation() {} }); }
    all() { return [this, ...this.childNodes.flatMap(child => child.all())]; }
  }
  document.body = new Element('body');
  document.createElement = tag => new Element(tag);
  document.createTextNode = text => { const node = new Element('#text'); node.textContent = text; return node; };
  document.getElementById = id => document.body.all().find(node => node.id === id) || null;
  document.querySelector = selector => selector === 'dialog[open]' ? document.body.all().find(node => node.tagName === 'DIALOG' && node.open) || null : null;
  document.addEventListener = Element.prototype.addEventListener;
  return { document, $: document.getElementById, Element };
}

export function deferred() {
  let resolve, reject;
  const promise = new Promise((a,b) => { resolve = a; reject = b; });
  return { promise, resolve, reject };
}
export const settle = () => new Promise(resolve => setImmediate(resolve));
