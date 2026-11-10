import { liveValue } from './robustness.mjs';

export function displayValue(value) {
  // Numbers arrive from proseg as 32-bit floats (0.03999999): show them rounded.
  return typeof value === 'boolean' ? (value ? 'on' : 'off') : value == null ? 'automatic'
    : typeof value === 'number' && Number.isFinite(value) ? String(Number(value.toPrecision(6))) : String(value);
}

export function validOptionValue(option, value) {
  if (!option) return false;
  if (option.type === 'bool') return typeof value === 'boolean';
  if (option.type === 'str') return typeof value === 'string' && (!option.choices || option.choices.includes(value));
  return typeof value === 'number' && Number.isFinite(value) &&
    (option.type !== 'int' || Number.isInteger(value)) &&
    (option.min == null || value >= option.min) && (option.max == null || value <= option.max);
}

// Never let a fixed suggestion silently move a knob opposite to the advertised direction.
export function suggestionIssue(action, option, current, requirePrevious = true) {
  if (action.flag !== option?.flag || !validOptionValue(option, action.value)) return 'Suggestion unavailable.';
  if (current == null) return requirePrevious ? 'Waiting for the current value so Undo can restore it.' : '';
  if (current === action.value) return 'Already at this value.';
  if ((action.direction === 'raise' && current > action.value) ||
      (action.direction === 'lower' && current < action.value)) return 'Already beyond this suggestion in that direction.';
  return '';
}

// One session per run. Manual edits and guide trials use the same sender and pending values.
// Keeping this independent of the DOM lets us verify Try/Undo and races with snapshots.
export class LiveOptionSession {
  constructor(run, send) {
    this.run = run;
    this.send = send;
    this.drafts = new Map();
    this.busy = new Set();
    this.history = new Map();
  }
  update(run) {
    if (run?.id !== this.run?.id) throw new Error('Live settings belong to a different run.');
    this.run = run;
  }
  get editable() { return ['running', 'paused'].includes(this.run?.state); }
  value(option) { return this.drafts.has(option.flag) ? this.drafts.get(option.flag) : liveValue(this.run, option).value; }
  stage(option, value) {
    this.drafts.set(option.flag, value);
    this.history.delete(option.flag);
  }
  async apply(option, value, remember = false) {
    const flag = option.flag;
    if (!this.editable) throw new Error('Live knobs are available while a run is running or paused.');
    if (!option.live || !validOptionValue(option, value)) throw new Error('This value cannot be applied live.');
    if (this.busy.has(flag)) throw new Error('Wait for the pending change to this knob.');
    const previous = { value: this.value(option), automatic: !this.drafts.has(flag) && liveValue(this.run, option).automatic };
    if (remember && previous.value == null) throw new Error('Wait for the current value before trying this suggestion.');
    this.busy.add(flag);
    this.drafts.set(flag, value);
    try {
      await this.send(this.run.id, { [flag]: value });
      // The next snapshot can still report the old effective value. Retain the accepted request.
      this.run = { ...this.run, live_values: { ...this.run.live_values, [flag]: value } };
      if (remember) this.history.set(flag, previous);
      else this.history.delete(flag);
      return previous;
    } finally {
      this.drafts.delete(flag);
      this.busy.delete(flag);
    }
  }
  async undo(option) {
    const previous = this.history.get(option.flag);
    if (!previous) throw new Error('There is no suggestion to undo for this knob.');
    await this.apply(option, previous.value);
    return previous;
  }
}
