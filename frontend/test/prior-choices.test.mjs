import test from 'node:test';
import assert from 'node:assert/strict';
import { priorChoices, choiceIndex } from '../src/prior-options.mjs';
// The picker's one prior list (2026-10-19: the user chooses one prior, from one list): the
// instrument's own pairing first, then every layer or assignment column on its own; a nucleus flag only in the pairing.

const xenium = {
  cells: [{ kind: 'shapes', key: 'xenium_cell_boundaries', label: 'xenium_cell_boundaries (10 shapes)' },
          { kind: 'shapes', key: 'xenium_nucleus_boundaries', label: 'xenium_nucleus_boundaries (10 shapes)' },
          { kind: 'column', column: 'cell_id', unassigned: 'UNASSIGNED', label: "Transcripts' cell_id column (90% assigned)" }],
  nuclei: [{ kind: 'shapes', key: 'xenium_nucleus_boundaries', label: 'xenium_nucleus_boundaries (10 shapes)' },
           { kind: 'shapes', key: 'xenium_cell_boundaries', label: 'xenium_cell_boundaries (10 shapes)' },
           { kind: 'column', column: 'overlaps_nucleus', nuclear: '1', label: "Transcripts' overlaps_nucleus flag (30% flagged)" }],
};
const pairing = { cells: { kind: 'shapes', key: 'xenium_cell_boundaries' }, nuclei: { kind: 'shapes', key: 'xenium_nucleus_boundaries' } };

test('one list: the suggested cells-and-nuclei pairing first, then each layer or column alone, nothing twice', () => {
  const choices = priorChoices(xenium, pairing);
  assert.deepEqual(choices.map(c => c.label), [
    "xenium_cell_boundaries cells, seeded from xenium_nucleus_boundaries nuclei (proseg's default)",
    'xenium_cell_boundaries (10 shapes)', 'xenium_nucleus_boundaries (10 shapes)', "Transcripts' cell_id column (90% assigned)"]);
  assert.deepEqual(choices[0].prior, pairing);
  assert.deepEqual(choices[1].prior, { cells: { kind: 'shapes', key: 'xenium_cell_boundaries' }, nuclei: null });
  assert.deepEqual(choices[3].prior, { cells: { kind: 'column', column: 'cell_id', unassigned: 'UNASSIGNED' }, nuclei: null });
  assert.ok(choices.every(c => c.prior.nuclei?.column !== 'overlaps_nucleus' || c.prior.cells), 'a nucleus flag never alone');
  assert.equal(choiceIndex(choices, pairing), 0);
  assert.equal(choiceIndex(choices, { cells: { kind: 'column', column: 'cell_id', unassigned: 'UNASSIGNED' }, nuclei: null }), 3);
  assert.equal(choiceIndex(choices, { cells: { kind: 'shapes', key: 'gone' }, nuclei: null }), -1);
});

test('a sample with no nuclei (Cellpose shapes only, like B1): every layer is a choice on its own', () => {
  const cosmx = { cells: [{ kind: 'shapes', key: 'cellpose_boundaries_pruned', label: 'cellpose_boundaries_pruned (43,724 shapes)' },
                          { kind: 'column', column: 'cell_ID', unassigned: '0', label: "Transcripts' cell_ID column (92% assigned)" }],
                  nuclei: [{ kind: 'shapes', key: 'cellpose_boundaries_pruned', label: 'cellpose_boundaries_pruned (43,724 shapes)' },
                           { kind: 'column', column: 'CellComp', nuclear: 'Nuclear', label: "Transcripts' CellComp flag (40% flagged)" }] };
  const choices = priorChoices(cosmx, { cells: { kind: 'shapes', key: 'cellpose_boundaries_pruned' }, nuclei: null });
  assert.deepEqual(choices.map(c => c.prior), [
    { cells: { kind: 'shapes', key: 'cellpose_boundaries_pruned' }, nuclei: null },
    { cells: { kind: 'column', column: 'cell_ID', unassigned: '0' }, nuclei: null }]);
  const native = priorChoices(cosmx, { cells: { kind: 'column', column: 'cell_ID', unassigned: '0' },
                                       nuclei: { kind: 'column', column: 'CellComp', nuclear: 'Nuclear' } });
  assert.equal(native[0].label, "cell_ID cells, seeded from CellComp nuclei (proseg's default)");
  assert.equal(native.length, 3);
});
