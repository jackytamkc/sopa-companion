import test from 'node:test';
import assert from 'node:assert/strict';
import { readdirSync, readFileSync } from 'node:fs';

// The app's document has a strict Content-Security-Policy (server/security.py: scripts, styles and images from the
// engine alone, no inline script or style, no eval): the page must need none of what it refuses. Every module of the
// page and index.html is read whole. Styles set through the CSSOM (element.style.x = ..., a constructed sheet) are not
// refused, and stay allowed.
const SRC = new URL('../src/', import.meta.url);
const PAGE = [...readdirSync(SRC).filter(name => /\.m?js$/.test(name)).map(name => ['src/' + name, readFileSync(new URL(name, SRC), 'utf8')]),
  ['index.html', readFileSync(new URL('../index.html', import.meta.url), 'utf8')]];
const REFUSED = [
  ['an inline <script> (one without src)', /<script(?![^>]*\bsrc=)[^>]*>/i],
  ['an inline event handler attribute (on...=)', /<[a-z][^<>]*\son[a-z]+\s*=/i],
  ['a style attribute in markup or a template', /<[a-z][^<>]*\sstyle\s*=/i],
  ["setAttribute('style', ...)", /setAttribute\(\s*['"`]style['"`]/],
  ['a <style> element made by a script', /createElement\(\s*['"`]style['"`]\s*\)|<style[\s>]/i],
  ['eval', /(?<![\w.$])eval\s*\(/],
  ['new Function', /\bnew\s+Function\s*\(/],
  ['a javascript: address', /javascript:/i],
];

test('the page needs nothing its security policy refuses: no inline script, handler, style attribute or <style>, eval or javascript: address', () => {
  assert.ok(PAGE.length > 50 && PAGE.some(([name]) => name === 'src/main.js') && PAGE.some(([name]) => name === 'src/page-key.mjs'));
  // Whole files, not lines: a tag in a template may run over several lines.
  const found = [];
  for (const [name, text] of PAGE) {
    for (const [what, pattern] of REFUSED) {
      for (const match of text.matchAll(new RegExp(pattern.source, pattern.flags + 'g')))
        found.push(`${name}:${text.slice(0, match.index).split('\n').length}: ${what}`);
    }
  }
  assert.deepEqual(found, []);
});

test('the checks find what they look for', () => {
  const caught = sample => REFUSED.filter(([, pattern]) => pattern.test(sample)).map(([what]) => what);
  for (const sample of ['<script>go()</script>', '<button onclick="go()">', '<p style="color:red">', "el.setAttribute('style', 'x')",
    "document.createElement('style')", '<style>p{}</style>', 'eval(text)', 'new Function("x")', '<a href="javascript:go()">',
    '<button type="button"\n    onclick="go()">', '<div class="x"\n  style="display:none">'])
    assert.equal(caught(sample).length, 1, sample);
  for (const sample of ['<script type="module" src="./src/main.js"></script>', 'el.style.left = "3px"', 'button.onclick = () => go()',
    'const style = pick(saved)', 'retrieval(x)', 'evaluate(x)', 'this.evalCount', 'sheet.replaceSync(css)'])
    assert.deepEqual(caught(sample), [], sample);
});
