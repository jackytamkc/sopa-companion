export function priorSpec(option) {
  if (!option) return null;
  if (option.kind === 'shapes') return { kind: 'shapes', key: option.key };
  const spec = { kind: 'column', column: option.column };
  for (const key of ['unassigned', 'nuclear']) if (key in option) spec[key] = option[key];
  return spec;
}

export function priorIndex(options, suggested) {
  if (!suggested) return -1;
  return options.findIndex(option => option.kind === suggested.kind &&
    (option.kind === 'shapes' ? option.key === suggested.key : option.column === suggested.column &&
      option.unassigned === suggested.unassigned && option.nuclear === suggested.nuclear));
}

const optionName = option => option.kind === 'shapes' ? option.key : option.column;

/** The picker's one prior list: the user picks exactly one, and proseg starts its cells from it.
 *  - First, when the sample suggests cells AND nuclei (the instrument's own pairing), that pairing: proseg's default,
 *    the cells seeded from the nuclei inside them.
 *  - Then every cell layer or assignment column on its own: every transcript inside a cell seeds that cell.
 *  - A nucleus layer is among the layers (a nucleus seeds a cell); a nucleus flag column cannot seed cells by itself,
 *    so it is only offered inside the pairing.
 *  Each choice: {label, prior: {cells, nuclei}} (the run's prior, as POST /runs takes it). */
export function priorChoices(options, suggested) {
  const cells = options?.cells || [], nuclei = options?.nuclei || [], out = [];
  const c = priorIndex(cells, suggested?.cells), n = priorIndex(nuclei, suggested?.nuclei);
  if (c >= 0 && n >= 0) {
    out.push({ label: `${optionName(cells[c])} cells, seeded from ${optionName(nuclei[n])} nuclei (proseg's default)`,
               prior: { cells: priorSpec(cells[c]), nuclei: priorSpec(nuclei[n]) } });
  }
  for (const option of cells) out.push({ label: option.label, prior: { cells: priorSpec(option), nuclei: null } });
  const layers = new Set(cells.filter(o => o.kind === 'shapes').map(o => o.key));
  for (const option of nuclei) {
    if (option.kind === 'shapes' && !layers.has(option.key)) {
      out.push({ label: option.label, prior: { cells: null, nuclei: priorSpec(option) } });
    }
  }
  return out;
}

/** Which of `choices` is the prior {cells, nuclei} (an earlier run's, or the sample's suggestion), or -1. */
export function choiceIndex(choices, prior) {
  const same = (a, b) => (!a && !b) || (!!a && !!b && priorIndex([a], b) === 0);
  return choices.findIndex(choice => same(choice.prior.cells, prior?.cells) && same(choice.prior.nuclei, prior?.nuclei));
}
