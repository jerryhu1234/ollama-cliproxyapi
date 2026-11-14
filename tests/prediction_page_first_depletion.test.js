const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const test = require('node:test');
const html = fs.readFileSync(path.join(__dirname, '../resources/prediction_page.html'), 'utf8');

class Element {
  constructor(tag) { this.tagName = tag; this.children = []; this.style = {}; this.className = ''; }
  append(...items) { for (const item of items) { if (item.parent) item.parent.children = item.parent.children.filter(n => n !== item); item.parent = this; this.children.push(item); } }
  get firstChild() { return this.children[0]; }
  appendChild(item) { this.append(item); return item; }
  set textContent(value) { this._text = String(value); this.children = []; }
  get textContent() { return (this._text || '') + this.children.map(n => n.textContent).join(''); }
  setAttribute(key, value) { this[key] = value; }
  scrollIntoView(options) { this.scrolled = options; }
  focus(options) { this.focused = options; }
  get classList() { return {
    add: value => { this.className += ' ' + value; },
    remove: value => { this.className = this.className.split(' ').filter(c => c !== value).join(' '); },
  }; }
}
function find(root, predicate) {
  if (predicate(root)) return root;
  for (const child of root.children) { const match = find(child, predicate); if (match) return match; }
}
function page(language = 'en', reduced = true) {
  const app = new Element('main'), timers = [];
  const context = { app, metaLine: null, document: { createElement: tag => new Element(tag) },
    window: { matchMedia: () => ({ matches: reduced }) }, setTimeout: fn => timers.push(fn),
    inDur: n => n > 0 ? `in ${n}s` : '—', fmtMoney: n => '$' + n,
    renderPersistBanner() {}, projectionChart: () => new Element('svg'),
    request: async () => {}, load() {}, REFRESH_MS: 60000, windowLabel: n => n,
  };
  const i18n = html.slice(html.indexOf('  const I18N ='), html.indexOf('  let lang ='));
  vm.createContext(context);
  vm.runInContext(i18n + `\nconst lang = '${language}'; function t(k) { return I18N[lang][k] || k; } function tf(k, vars) { return t(k).replace(/\\{(\\w+)\\}/g, (_, n) => vars[n]); }\n` +
    html.slice(html.indexOf('  function fleetStat('), html.indexOf('  // ---- P3b cross-type tab')), context);
  return { context, app, timers };
}
const accounts = [{ key_id: 'a', label: 'first@test', windows: [] }, { key_id: 'b', label: '<b>very-long-account@test.example</b>', windows: [] }];
function data(winner = 'b', seconds = 20) { return { accounts, by_type: { ollama: { first_depletion_key_id: winner, first_depletion_seconds: seconds } } }; }

test('names the winning account, marks only its card and preserves order', () => {
  const p = page(); p.context.render(data());
  const fleet = p.app.children[0], cards = p.app.children[1].children;
  assert.ok(fleet.textContent.includes(accounts[1].label), 'summary must name winner');
  assert.ok(fleet.textContent.includes('in 20s'));
  assert.ok(cards[0].textContent.includes(accounts[0].label));
  assert.ok(!cards[0].textContent.includes('First depletion'));
  assert.ok(cards[1].textContent.includes('First depletion · in 20s'));
  const label = find(fleet, n => n.className === 'first-account-name');
  assert.equal(label.textContent, accounts[1].label);
  assert.equal(label.title, accounts[1].label);
  assert.equal(label.children.length, 0, 'labels must not become HTML');
});
test('summary click focuses and temporarily highlights the actual card', () => {
  for (const reduced of [true, false]) {
    const p = page('en', reduced); p.context.render(data());
    const button = find(p.app.children[0], n => n.tagName === 'button');
    assert.ok(button, 'identified summary must be keyboard-accessible');
    assert.ok(button['aria-label'].includes('First depletion'));
    assert.ok(button['aria-label'].includes('in 20s'));
    button.onclick();
    const card = p.app.children[1].children[1];
    assert.equal(card.scrolled.behavior, reduced ? 'auto' : 'smooth');
    assert.equal(card.focused.preventScroll, true);
    assert.ok(card.className.includes('depletion-highlight'));
    p.timers.forEach(fn => fn());
    assert.ok(!card.className.includes('depletion-highlight'));
  }
});
test('missing identity, legacy responses and no prediction never invent a winner', () => {
  for (const input of [data('unknown'), data('b', 0), { accounts, by_type: { ollama: { first_depletion_seconds: 20 } } }]) {
    const p = page(); p.context.render(input);
    assert.equal(find(p.app.children[0], n => n.tagName === 'button'), undefined);
    assert.equal(find(p.app, n => n.className.includes('first-depletion-badge')), undefined);
  }
});
test('refresh moves the badge and Chinese translation without changing order', () => {
  const p = page('zh'); p.context.render(data()); p.context.render(data('a', 30));
  assert.ok(p.app.children[1].children[0].textContent.includes('最先耗尽 · in 30s'));
  assert.ok(!p.app.children[1].children[1].textContent.includes('最先耗尽'));
});
test('provider summary without local accounts never links to another provider', () => {
  const p = page(); const fleet = p.context.renderFleet(data().by_type.ollama, 'peer');
  assert.equal(find(fleet, n => n.tagName === 'button'), undefined);
});
