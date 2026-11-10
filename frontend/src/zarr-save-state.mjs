/** The engine's full-sample decision wins; otherwise only a full region (every stain tile done) is the default. */
export function defaultSegmentation(run) {
  if (typeof run.save_defaults?.set_default === 'boolean') return run.save_defaults.set_default;
  return run.region?.kind === 'full' && (run.kind !== 'stain' || run.progress?.complete === true);
}

/** The dialog's fields as it opens: the names and choice of the run's last save, else its save defaults. */
export function saveFields(run) {
  const values = run.zarr_save || run.save_defaults || {};
  return { shapes_key: values.shapes_key || '', table_key: values.table_key || '',
    set_default: run.zarr_save?.set_default ?? defaultSegmentation(run), overwrite: false };
}

/** The names' errors by field, the existing layers a save would replace (`conflicts`: exact names only) and whether it can
 *  go ahead. A name that differs from one in the sample only by case is an error of its field, never something to
 *  replace: the engine refuses it whatever Replace says (SpatialData's names must differ by more than case). `api`: a
 *  notebook's session, whose words name your SpatialData. */
export function validateSaveNames(fields, existing = [], shapesOnly = false, { api = false } = {}) {
  const errors = {}, pattern = /^[A-Za-z0-9_.-]+$/, keys = shapesOnly ? ['shapes_key'] : ['shapes_key', 'table_key'];
  for (const key of keys) {
    if (!fields[key]) errors[key] = 'Enter a name.';
    else if (!pattern.test(fields[key])) errors[key] = 'Use letters, digits, _ . - only.';
  }
  if (!shapesOnly && fields.shapes_key && fields.shapes_key.toLowerCase() === fields.table_key?.toLowerCase()) errors.table_key = 'Use a different name for the table.';
  const names = existing.filter(key => typeof key === 'string' && key);
  // As the engine checks it (Runs.save_to_zarr): another name equal ignoring case, even beside the exact one.
  const twin = name => !!name && names.find(key => key !== name && key.toLowerCase() === name.toLowerCase());
  for (const key of keys) {
    const other = !errors[key] && twin(fields[key]);
    if (other) errors[key] = `${other} is already in ${api ? 'your SpatialData' : 'the sample'}: use a name that differs by more than case.`;
  }
  const conflicts = [...new Set(keys.map(key => fields[key]).filter(name => name && names.includes(name)))];
  return { errors, conflicts, valid: !Object.keys(errors).length && (!conflicts.length || !!fields.overwrite) };
}

/** `existing`: the sample's names as the dialog last heard them (asked as it opened, or named by a refused save), in
 *  place of those the run's save defaults list (as the run ended: another save may have taken names since). `api`: a
 *  notebook's session (its words). */
export function saveModalState(run, fields, { submitting = false, error = '', existing: known = null, api = false } = {}) {
  const save = run.zarr_save;
  const existing = [...(known || run.save_defaults?.existing || []), ...(['saved', 'queued', 'outdated'].includes(save?.state) ? [save.shapes_key, save.table_key] : [])];
  const stain = run.kind === 'stain';
  const validation = validateSaveNames(fields, existing, stain, { api }), saving = submitting || save?.state === 'saving';
  return { ...validation, saving, canSave: saveable(run) && validation.valid && !saving,
    message: saving ? save?.state === 'saving' && save.message || (stain ? 'Writing cell shapes…' : 'Writing shapes and table…') : '',
    error: error || (!saving && save?.state === 'failed' ? save.message || 'Saving failed. Try again.' : '') };
}

export function saveable(run) {
  return run.kind !== 'patches' && run.sample_kind === 'zarr' && (run.kind === 'stain' ?
    !!run.progress?.resolved && (['done', 'stopped'].includes(run.state) || run.state === 'failed' && !!run.progress?.complete) :
    run.state === 'done');
}

export function saveOutdated(run) {
  const save = run.zarr_save;
  return save?.state === 'outdated' || run.kind === 'stain' && ['saved', 'queued'].includes(save?.state) &&
    (save.result_revision !== undefined && save.result_revision !== run.progress?.revision ||
      (save.exclusions_digest != null ? save.exclusions_digest !== run.exclusions?.digest :
        (save.exclusions_version ?? 0) !== (run.exclusions?.version ?? 0)));
}

export function shouldPromptSave(run, finishedHere) {
  return run.kind !== 'patches' && finishedHere && run.state === 'done' && run.sample_kind === 'zarr' && !run.zarr_save;
}

export function saveCompleted(save, initial) {
  return ['saved', 'queued', 'failed', 'outdated'].includes(save?.state) &&
    ['state', 'shapes_key', 'table_key', 'set_default', 'message', 'cells', 'at', 'result_revision', 'exclusions_version', 'exclusions_digest', 'version', 'base', 'log_hash'].some(key => save[key] !== initial?.[key]);
}
