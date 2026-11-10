import { liveValue } from './robustness.mjs';
import { displayValue, validOptionValue, suggestionIssue, LiveOptionSession } from './option-guide.mjs';
import { guard, busy, clearExplanation, problem } from './explain.mjs';
// Schema-driven inputs shared by startup options and live controls.
const KEY = 'sopa-companion:options';
let helpId = 0;

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}
function button(text, className = 'text-button') {
  const node = element('button', className, text);
  node.type = 'button';
  return node;
}

export function optionHelp(option) {
  const toggle = button('?', 'option-help-toggle');
  const card = element('section', 'option-help');
  card.id = `option-help-${++helpId}`;
  card.hidden = true;
  card.setAttribute('aria-label', `About ${option.label}`);
  toggle.setAttribute('aria-label', `Explain ${option.label}`);
  toggle.setAttribute('aria-controls', card.id);
  toggle.setAttribute('aria-expanded', 'false');
  toggle.onclick = () => {
    card.hidden = !card.hidden;
    toggle.setAttribute('aria-expanded', String(!card.hidden));
  };
  const terms = element('dl');
  const directions = option.type === 'bool' ? ['Turn it on', 'Turn it off'] :
    option.type === 'str' ? ['Change the selection', 'Keep the default'] : ['Raise it →', 'Lower it →'];
  for (const [title, value] of [
    ['What it changes', option.what || option.help],
    [directions[0], option.up], [directions[1], option.down],
    ['Use when', option.use_when], ['Typical range and default', option.typical], ['Caution', option.caution],
  ]) {
    if (value) terms.append(element('dt', '', title), element('dd', '', value));
  }
  card.append(terms);
  return { toggle, card };
}

export function optionInput(option, value, change) {
  const input = document.createElement(option.choices?.length ? 'select' : 'input');
  if (option.choices?.length) {
    for (const choice of option.choices) {
      const item = document.createElement('option'); item.value = choice; item.textContent = choice; input.append(item);
    }
  } else {
    input.type = option.type === 'bool' ? 'checkbox' : ['int', 'float'].includes(option.type) ? 'number' : 'text';
    if (input.type === 'number') {
      input.step = option.type === 'int' ? '1' : 'any';
      if (option.min != null) input.min = option.min;
      if (option.max != null) input.max = option.max;
    }
    input.placeholder = String(option.default ?? '');
  }
  input.setAttribute('aria-label', option.label);
  input.title = option.what || option.help || '';
  // Proseg reports knobs as 32-bit floats (0.04 arrives as 0.03999999): show them rounded.
  const write = (v) => { if (option.type === 'bool') input.checked = !!v; else input.value = v == null || v === '' ? '' : option.type === 'float' && Number.isFinite(Number(v)) ? String(Number(Number(v).toPrecision(6))) : v; };
  write(value);
  input.addEventListener('input', () => {
    if (!input.checkValidity()) return;
    const value = option.type === 'bool' ? input.checked : input.value === '' ? option.default : ['int', 'float'].includes(option.type) ? Number(input.value) : input.value;
    change(value);
  });
  return { input, write };
}
// Every run follows proseg's own schedule, so each option's default is proseg's (the schema's). Values remembered while
// the picker offered schedule presets (older versions) were kept only where they differed from the preset's default:
// one that equals proseg's own default is dropped as it loads; any other stays the user's choice (Reset puts it back),
// unless proseg would refuse the run with it (refusedSchedule).
export class Options {
  constructor(container) { this.container = container; this.schema = []; this.values = {}; this.rows = new Map(); this.explicit = new Set(); }
  load(schema) {
    this.schema = schema; this.explicit = new Set();
    let saved = {};
    try { saved = JSON.parse(localStorage.getItem(KEY) || '{}') || {}; } catch { /* optional storage */ }
    this.values = Object.fromEntries(schema.filter(o => Object.hasOwn(saved, o.flag) && saved[o.flag] !== o.default).map(o => [o.flag, saved[o.flag]]));
    if (this.refusedSchedule()) { delete this.values.samples; delete this.values['recorded-samples']; }
    this.render();
  }
  /** More recorded samples than samples, which proseg refuses (the engine says "recorded-samples must be <= samples"):
   *  a Samples remembered while "Until I stop" recorded none (its preset: 60 samples, 0 recorded) is one such pair, and
   *  loads as proseg's own two defaults instead of a Start that fails. */
  refusedSchedule() {
    const samples = this.schema.find(o => o.flag === 'samples'), recorded = this.schema.find(o => o.flag === 'recorded-samples');
    return !!samples && !!recorded && Number(this.value(recorded)) > Number(this.value(samples));
  }
  save() { this.unsaved = false; try { localStorage.setItem(KEY, JSON.stringify(this.values)); } catch { /* optional storage */ } }
  /** Back to the remembered options when a fill-in (`setAll` with save: false) was left without starting a run. */
  restore() { if (this.unsaved) this.load(this.schema); }
  value(option) { return this.values[option.flag] ?? option.default; }
  setValue(flag, value) {
    const option = this.schema.find(o => o.flag === flag);
    if (!validOptionValue(option, value)) throw new Error('Check the suggested option value.');
    this.explicit.delete(flag);
    if (value === option.default) delete this.values[flag]; else this.values[flag] = value;
    this.rows.get(flag)?.write(value);
    this.save();
    this.onChange?.();
  }
  /** The rows holding a value the page cannot use (out of range, not a whole number), as explain.mjs problems that name
   *  the row and point at its input: for the picker's Start (main.js pickerProblems), which is never greyed out. */
  problems() {
    return this.schema.filter(option => this.rows.get(option.flag)?.input.checkValidity() === false).map(option => {
      const range = option.min != null && option.max != null ? ` from ${displayValue(option.min)} to ${displayValue(option.max)}` :
        option.min != null ? ` of at least ${displayValue(option.min)}` : option.max != null ? ` of at most ${displayValue(option.max)}` : '';
      return problem(`option-${option.flag}`, option.label, `${option.type === 'int' ? 'Use a whole number' : 'Use a number'}${range}.`);
    });
  }
  getValues() {
    for (const input of this.container.querySelectorAll('input, select')) if (!input.reportValidity()) throw new Error('Check the advanced option values.');
    return Object.fromEntries(this.schema.filter(o => Object.hasOwn(this.values, o.flag) && (this.values[o.flag] !== o.default || this.explicit.has(o.flag)))
      .map(o => [o.flag, this.values[o.flag]]));
  }
  /** Every advanced option at once ("New run with this setting": the run's own options plus the suggested one): the
   *  others go back to their defaults; a value the schema refuses is left out. `explicit`: flags the new run names even
   *  at their default (the suggested setting: the run records it), until their row is edited or reset. `save: false`:
   *  not remembered (until a row is edited or a run started with them: `save`; `restore` puts the remembered ones
   *  back). Returns the flags that were set. */
  setAll(values = {}, { explicit = [], save = true } = {}) {
    this.values = {}; this.explicit = new Set();
    for (const [flag, value] of Object.entries(values ?? {})) {
      const option = this.schema.find(o => o.flag === flag);
      if (!validOptionValue(option, value)) continue;
      if (explicit.includes(flag)) { this.values[flag] = value; this.explicit.add(flag); }
      else if (value !== option.default) this.values[flag] = value;
    }
    if (save) this.save(); else this.unsaved = true;
    this.render();
    return Object.keys(this.values);
  }
  render() {
    this.container.replaceChildren();
    this.rows.clear();
    const reset = button('Reset all');
    reset.onclick = () => { this.values = {}; this.explicit.clear(); this.save(); this.render(); };
    this.container.append(reset);
    const groups = new Map([['Main', []]]);
    for (const option of this.schema) {
      const group = option.advanced ? option.group || 'Other' : 'Main';
      if (!groups.has(group)) groups.set(group, []);
      groups.get(group).push(option);
    }
    for (const [group, options] of groups) {
      if (!options.length) continue;
      const heading = document.createElement('h3'); heading.textContent = group; this.container.append(heading);
      for (const option of options) {
        const row = document.createElement('div'); row.className = 'option-row';
        const label = document.createElement('label'); label.textContent = option.label;
        const { input, write } = optionInput(option, this.values[option.flag] ?? option.default, value => {
          this.explicit.delete(option.flag);
          if (value === option.default) delete this.values[option.flag]; else this.values[option.flag] = value;
          this.save();
          this.onChange?.();
        });
        input.id = `option-${option.flag}`;       // what a Start that cannot go ahead points at (problems)
        this.rows.set(option.flag, { input, write });
        label.append(input);
        const reset = button('Reset'); reset.setAttribute('aria-label', `Reset ${option.label}`);
        reset.onclick = () => { delete this.values[option.flag]; this.explicit.delete(option.flag); write(option.default); this.save(); this.onChange?.(); };
        const help = optionHelp(option);
        row.append(label, help.toggle, reset, help.card); this.container.append(row);
      }
    }
    this.onChange?.();
  }
}

export class KnobGuide {
  constructor(container, toggle, context) {
    this.container = container;
    this.context = context;
    this.messages = new Map();
    this.controls = [];
    this.generation = 0;
    toggle.setAttribute('aria-controls', container.id);
    toggle.setAttribute('aria-expanded', 'false');
    toggle.onclick = event => {
      event.preventDefault();
      event.stopPropagation();
      container.hidden = !container.hidden;
      toggle.setAttribute('aria-expanded', String(!container.hidden));
      const details = container.closest('details');
      if (!container.hidden && details) details.open = true;
      if (!container.hidden) this.select?.focus();
    };
    container.textContent = 'Loading knob guidance…';
  }
  load(guide, schema) {
    this.guide = guide;
    this.schema = new Map(schema.map(option => [option.flag, option]));
    this.container.replaceChildren();
    if (!guide.length) {
      this.container.append(element('p', '', 'Knob guidance is unavailable. Reload to retry.'));
      return;
    }
    const label = element('label', 'guide-symptoms', 'What do you see?');
    this.select = element('select');
    this.select.append(element('option', '', 'Choose a symptom…'));
    this.select.firstChild.value = '';
    guide.forEach((entry, index) => {
      const choice = element('option', '', entry.symptom);
      choice.value = String(index);
      this.select.append(choice);
    });
    label.append(this.select);
    this.detail = element('div', 'guide-detail');
    this.container.append(label, element('p', 'guide-note',
      'Try one change at a time and compare with the image. Suggested values are starting trials, not validated biological targets.'), this.detail);
    this.select.onchange = () => this.show(this.select.value);
  }
  reset() {
    this.generation++;
    this.messages.clear();
    if (this.select) { this.select.value = ''; this.show(''); }
  }
  show(index) {
    this.detail.replaceChildren();
    this.controls = [];
    if (index === '') return;
    const entry = this.guide[Number(index)];
    if (!entry) return;
    this.detail.append(element('h4', 'guide-symptom-title', entry.symptom), element('p', 'guide-explanation', entry.explanation));
    for (const action of entry.actions) {
      const option = this.schema.get(action.flag);
      if (!option) continue;
      const card = element('div', 'guide-action');
      const verb = { raise: 'Raise', lower: 'Lower', on: 'Turn on', off: 'Turn off', set: 'Set' }[action.direction];
      const title = option.type === 'bool' ? `${verb}: ${option.label}` : `${verb} ${option.label} to ${displayValue(action.value)}`;
      card.append(element('strong', '', title), element('p', '', action.why));
      const live = this.context.mode === 'live' && action.live && option.live;
      const timing = element('p', 'guide-timing', this.timing(live));
      card.append(timing);
      const status = element('p', 'guide-status');
      status.setAttribute('role', 'status');
      const current = element('p', 'guide-current');
      if (live || this.context.mode === 'picker') {
        card.append(current);
        const apply = button(live ? 'Try' : 'Use in Advanced options', 'guide-try');
        const undo = live ? button('Undo', 'guide-undo') : null;
        const controls = element('div', 'guide-buttons');
        controls.append(apply);
        if (undo) controls.append(undo);
        // Try and Undo are never greyed out: a press that cannot go ahead says why here, by the knob's name.
        const note = element('div', 'missing-note');
        note.setAttribute('role', 'alert');
        note.hidden = true;
        card.append(controls, note);
        const row = { action, option, apply, undo, status, current, live, timing };
        const act = async (pressed, isUndo) => {
          const generation = this.generation, done = live ? busy(pressed, 'Sending…') : null;
          try {
            const pending = isUndo ? this.context.undo(option) : this.context.apply(action, option);
            this.update();
            const previous = await pending;
            if (generation !== this.generation) return;
            this.messages.set(action.flag, isUndo ?
              `Restored ${displayValue(previous.value)}; applies from the next iteration.${previous.automatic ? ' Undo restores the number; automatic cooling needs a new run.' : ''}` :
              live ? { text: `Set to ${displayValue(action.value)}; applies from the next iteration.`, press: true } :
                'Added to Advanced options; applies to a new run.');
          } catch (error) {
            if (generation !== this.generation) return;
            this.messages.set(action.flag, error.message);
          } finally {
            done?.(); clearExplanation(pressed, { note });
          }
          this.update();
        };
        guard(apply, () => this.problems(row, false), () => act(apply, false), { note });
        if (undo) guard(undo, () => this.problems(row, true), () => act(undo, true), { note });
        this.controls.push(row);
      } else {
        card.append(element('p', 'guide-note', 'Set this in Advanced proseg options when choosing a region for a new run.'));
      }
      card.append(status);
      this.detail.append(card);
    }
    this.update();
  }
  /** Why a press of Try (or Undo) cannot go ahead now: the run's state or a change on its way (blocked), the suggestion
   *  against the current value, or no suggestion to undo; one problem named by the knob, or none. */
  problems({ action, option, live }, isUndo) {
    const reason = (live && this.context.blocked(option)) || (isUndo ?
      (this.context.canUndo(option) ? '' : 'No suggestion to undo yet.') : suggestionIssue(action, option, this.context.value(option), live));
    return reason ? [problem(null, option.label, reason)] : [];
  }
  /** Which of the run's buttons a press makes work now (the live guide's context: main.js runButtons), {} without one. */
  buttons() { return this.context.buttons?.() ?? {}; }
  /** A card's timing line: a live knob applies from the next iteration, naming the buttons that move the run on only
   *  while both work (Start or Step on the prior, before the first iteration; Resume or Step paused later); a startup
   *  knob applies to a new run. */
  timing(live) {
    if (!live) return 'Applies to a new run.';
    const b = this.buttons();
    return b.start && b.step ? 'Applies from the first iteration: press Start or Step.'
      : b.resume && b.step ? 'Applies from the next iteration: press Resume or Step.' : 'Applies from the next iteration.';
  }
  /** What a live Try's message adds while both buttons that move the run on work, else ''. */
  press() {
    const b = this.buttons();
    return b.start && b.step ? ' Press Start or Step.' : b.resume && b.step ? ' Press Resume or Step.' : '';
  }
  /** The cards' changing words (with every run render whose buttons changed: main.js renderRun): the current value,
   *  why a press cannot go ahead, the timing line and the last press's message, naming the run's buttons as they work
   *  now. */
  update() {
    for (const row of this.controls) {
      const { action, option, live } = row;
      const value = this.context.value(option);
      const blocked = live ? this.context.blocked(option) : '';
      row.current.textContent = `Current ${live ? 'requested ' : ''}value: ${displayValue(value)}`;
      row.apply.title = blocked || suggestionIssue(action, option, value, live);
      if (row.undo) row.undo.title = blocked || 'Restore the value before the last suggestion for this knob';
      const timing = this.timing(live);
      if (row.timing && row.timing.textContent !== timing) row.timing.textContent = timing;
      const message = this.messages.get(action.flag);
      row.status.textContent = typeof message === 'string' ? message : message ? message.text + (message.press ? this.press() : '') : '';
    }
  }
}

export class LiveKnobs {
  constructor(container, send, notice) { this.container = container; this.send = send; this.notice = notice; this.rows = new Map(); this.timers = new Map(); this.schema = []; }
  value(option) { return this.session?.value(option); }
  /** The live option of a proseg flag (from the schema of the last render), or null. */
  option(flag) { return this.schema.find(o => o.flag === flag && o.live) ?? null; }
  /** The mixing check's *Try it* (the try loop): the same session, validation (suggestionIssue,
   *  in the direction the suggestion goes, `from` -> `value`) and Undo history as the Live knobs rows and the knob guide,
   *  so the knob's row shows the change at once. `session`: the run's session it is set in, the one Live knobs has now
   *  by default; a test's Use passes the session of the run it was tested on, as it was when Use was pressed (its
   *  changes go to that run only, whatever the page shows meanwhile). Resolves with the value
   *  before ({value, automatic}: what Undo puts back); throws a plain message when the value cannot be applied now. */
  async trySetting(flag, value, from = undefined, session = this.session) {
    const option = this.option(flag);
    if (!option) throw new Error("This setting can't change during a run.");
    // A Live knobs edit still waiting to be sent (its 350 ms) is dropped: the value in effect is what Undo puts back.
    if (session === this.session && this.timers.has(flag)) { clearTimeout(this.timers.get(flag)); this.timers.delete(flag); session?.drafts.delete(flag); }
    const current = session?.value(option), before = from === undefined ? current : from;
    const direction = typeof value === 'number' && typeof before === 'number' ? (value < before ? 'lower' : 'raise') : 'set';
    // Proseg reports knobs as 32-bit floats: 0.02 comes back as 0.0199999995.
    if (typeof value === 'number' && typeof current === 'number' && Math.abs(value - current) <= 1e-6 * Math.max(1, Math.abs(value))) throw new Error('Already at this value.');
    const issue = suggestionIssue({ flag, value, direction }, option, current, true);
    if (issue) throw new Error(issue);
    return this.apply(option, value, true, false, session);
  }
  /** The mixing check's Undo: puts back `value` (the trial's old value) through the same session, or, without one, the
   *  value before the session's last remembered change of this knob (LiveOptionSession.undo). */
  async undoSetting(flag, value = undefined) {
    const option = this.option(flag);
    if (!option) throw new Error("This setting can't change during a run.");
    return value == null ? this.apply(option, undefined, false, true) : this.apply(option, value);
  }
  blocked(option) {
    if (!this.session?.editable) return 'Live knobs are available while a run is running or paused.';
    return this.session.busy.has(option.flag) ? 'Sending change…' : '';
  }
  /** Sets `option` through `session`: the Live knobs session, or the one trySetting was given (a row's edit waiting to be
   *  sent is the Live knobs session's, dropped only for that one). */
  async apply(option, value, remember = false, undo = false, session = this.session) {
    if (!session) throw new Error('Select an active run first.');
    if (session === this.session) { clearTimeout(this.timers.get(option.flag)); this.timers.delete(option.flag); }
    const pending = undo ? session.undo(option) : session.apply(option, value, remember);
    this.refresh();
    try { return await pending; }
    finally { if (this.session === session) this.refresh(); }
  }
  render(run, schema) {
    this.schema = schema ?? [];
    const options = this.schema.filter(o => o.live);
    this.container.parentElement.hidden = !run || !['preparing', 'running', 'paused', 'finishing'].includes(run.state) || !options.length;
    const key = run?.id + ':' + options.map(o => o.flag).join(',');
    if (key !== this.key) {
      for (const timer of this.timers.values()) clearTimeout(timer);
      this.timers.clear(); this.rows.clear(); this.container.replaceChildren(); this.key = key;
      this.session = run ? new LiveOptionSession(run, this.send) : null;
      this.guide?.reset();
      for (const option of options) {
        const wrapper = element('div', 'knob-row');
        const label = document.createElement('label'); label.className = 'knob'; label.textContent = option.label;
        const row = optionInput(option, liveValue(run, option).value, value => {
          this.session.stage(option, value);
          clearTimeout(this.timers.get(option.flag));
          this.timers.set(option.flag, setTimeout(async () => {
            try { await this.apply(option, value); } catch (e) { this.notice(e.message); }
          }, 350));
          this.guide?.messages.delete(option.flag);
          this.guide?.update();
        });
        row.option = option;
        row.status = element('small', 'knob-status');
        const help = optionHelp(option);
        label.append(row.input, row.status);
        wrapper.append(label, help.toggle, help.card);
        this.container.append(wrapper); this.rows.set(option.flag, row);
      }
    } else if (run) this.session.update(run);
    this.refresh();
  }
  refresh() {
    for (const row of this.rows.values()) {
      const option = row.option;
      const effective = liveValue(this.session?.run, option);
      row.status.textContent = this.session?.busy.has(option.flag) ? 'Sending…' : effective.pending ?
        `Pending until next iteration (effective: ${displayValue(effective.effective)})` : effective.automatic ? 'Automatic' : '';
      row.input.title = option.what || option.help || '';
      row.input.disabled = !!this.blocked(option);
      if (row.input.disabled) { clearTimeout(this.timers.get(option.flag)); this.timers.delete(option.flag); }
      if (document.activeElement !== row.input && !this.timers.has(option.flag)) row.write(this.value(option));
    }
    this.guide?.update();
  }
}
