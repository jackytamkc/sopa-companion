from pathlib import Path
import re
import subprocess

import pytest

from sopa_companion.engine.proseg_options import GUIDE, OPTIONS, live_lines, to_args

ROOT = Path(__file__).resolve().parents[1]


def test_flags_exist():
    binary = ROOT / 'native/bin/proseg-live'
    if not binary.exists():
        pytest.skip('proseg-live binary missing')
    help_text = subprocess.check_output([str(binary), '--help'], text=True)
    flags = set(re.findall(r'--([a-z][a-z0-9-]*)', help_text))
    assert {o['flag'] for o in OPTIONS} <= flags


def test_defaults_and_roundtrip():
    assert len({o['flag'] for o in OPTIONS}) == len(OPTIONS)
    values = {o['flag']: o['default'] for o in OPTIONS if o['default'] is not None}
    args = to_args(values)
    restored = {}
    i = 0
    schema = {o['flag']: o for o in OPTIONS}
    while i < len(args):
        flag = args[i][2:]
        typ = schema[flag]['type']
        if typ == 'bool':
            restored[flag] = True
            i += 1
        else:
            restored[flag] = {'int': int, 'float': float, 'str': str}[typ](args[i+1])
            i += 2
    assert restored == {k: v for k, v in values.items() if v is not False}
    for o in OPTIONS:
        assert isinstance(o['advanced'], bool) and isinstance(o['live'], bool)
        assert o['help'] and o['label']


@pytest.mark.parametrize('values,flag', [
    ({'bogus': 1}, 'bogus'), ({'samples': True}, 'samples'),
    ({'samples': '5'}, 'samples'), ({'samples': -1}, 'samples'),
    ({'samples': 2}, 'recorded-samples'),
    ({'voxel-size': 3.0}, 'voxel-size'),
    ({'voxel-size': 0.7}, 'voxel-size'),
    ({'voxel-layers': 257}, 'voxel-layers'),
    ({'density-bins': 17}, 'density-bins'),
    ({'cell-compactness': 0}, 'cell-compactness'),
    ({'diffusion-sigma-far': float('nan')}, 'diffusion-sigma-far'),
    ({'prior-weight': float('inf')}, 'prior-weight'),
    ({'no-diffusion': 1}, 'no-diffusion'),
    ({'output-counts-fmt': 'bad'}, 'output-counts-fmt'),
    ({'use-cell-initialization': True}, 'Xenium parquet'),
    ({'temperature': None}, 'temperature'),
])
def test_validation(values, flag):
    with pytest.raises(ValueError, match=flag):
        to_args(values)


def test_partial_and_live():
    assert to_args({}) == []
    assert to_args({'samples': 2, 'recorded-samples': 0}) == ['--samples', '2', '--recorded-samples', '0']
    assert live_lines({'no-diffusion': False, 'cell-compactness': .03}) == [
        'set cell-compactness 0.03', 'set no-diffusion false']
    with pytest.raises(ValueError, match='startup-only'):
        live_lines({'voxel-size': 1.0})
    for o in OPTIONS:
        if o['live']:
            value = o['default'] if o['default'] is not None else 1.0
            assert live_lines({o['flag']: value})


def test_schema_covers_source_or_documents_exclusion():
    main = ROOT / 'native/proseg/src/main.rs'
    if not main.exists():      # native/build.sh makes native/proseg from the crate and proseg-live.patch; never committed
        pytest.skip('proseg source not made yet: run native/build.sh')
    source = main.read_text().split('fn set_xenium_presets')[0]
    fields = {f.replace('_', '-') for f in re.findall(r'^    (\w+): ', source, re.M)}
    module = (ROOT / 'engine/proseg_options.py').read_text()
    excluded = set(re.findall(r'^# Excluded: ([\w-]+) — .+', module, re.M))
    assert fields == {o['flag'] for o in OPTIONS} | excluded


@pytest.mark.parametrize('option', OPTIONS, ids=lambda option: option['flag'])
def test_biology_explanations_are_complete(option):
    for field in ('what', 'up', 'down', 'use_when', 'typical'):
        assert isinstance(option[field], str) and option[field].strip(), (option['flag'], field)
    if 'caution' in option:
        assert isinstance(option['caution'], str) and option['caution'].strip()


def test_the_hold_says_what_it_does_for_every_start():
    """The hold's Live knobs label (its hint after the name, which the mixing
    check's Setting lines repeat) and its tooltip say what it does whatever the run started from, never "closer to
    Xenium" (a run started from Cellpose cells holds those); its typical range says where to aim for a start from cells
    only."""
    option = next(o for o in OPTIONS if o['flag'] == 'prior-weight')
    assert option['label'] == 'Prior weight (higher = the starting cells hold their transcripts harder)'
    for field in ('label', 'help', 'what'):
        assert 'closer to Xenium' not in option[field] and 'starting cell' in option[field], field
    assert option['help'] == option['what']
    assert 'cells only' in option['typical'] and 'about 1' in option['typical']
    assert 'transcripts per cell over 100' in option['caution']


def test_guide_actions_match_schema_and_can_be_applied():
    schema = {option['flag']: option for option in OPTIONS}
    assert len(GUIDE) >= 11
    assert len({entry['symptom'] for entry in GUIDE}) == len(GUIDE)
    for entry in GUIDE:
        assert entry['symptom'] and entry['explanation']
        assert isinstance(entry['actions'], list)
        for action in entry['actions']:
            option = schema[action['flag']]
            assert action['direction'] in {'raise', 'lower', 'on', 'off', 'set'}
            assert action['why']
            assert type(action['live']) is bool
            assert action['live'] == option['live']
            assert to_args({action['flag']: action['value']}) or action['value'] is False
            if action['live']:
                assert live_lines({action['flag']: action['value']})
            if action['direction'] in {'on', 'off'}:
                assert option['type'] == 'bool'
                assert action['value'] is (action['direction'] == 'on')
            elif action['direction'] in {'raise', 'lower'}:
                assert option['type'] in {'int', 'float'}
                if option['default'] is not None:
                    if action['direction'] == 'raise':
                        assert action['value'] > option['default']
                    else:
                        assert action['value'] < option['default']


def test_options_and_guide_endpoints(config):
    from fastapi.testclient import TestClient
    from sopa_companion.server import create_app

    application = create_app(config, token='t' * 40)
    headers = {'Authorization': 'Bearer ' + 't' * 40}
    with TestClient(application, base_url='http://127.0.0.1:8799') as client:
        assert client.get('/api/v1/options/guide').status_code == 401
        options_response = client.get('/api/v1/options', headers=headers)
        guide_response = client.get('/api/v1/options/guide', headers=headers)
        assert options_response.status_code == guide_response.status_code == 200
        assert options_response.json() == OPTIONS
        assert guide_response.json() == GUIDE


def test_the_default_excluded_genes_are_every_platforms_control_probes():
    """The option's default is what every run leaves out (input_args passes it unless the user set a pattern
    or let the controls in), and its help says which probes."""
    from sopa_companion.engine.proseg_options import DEFAULT_EXCLUDED_GENES, input_args
    option = next(o for o in OPTIONS if o['flag'] == 'excluded-genes')
    assert option['default'] == DEFAULT_EXCLUDED_GENES
    assert DEFAULT_EXCLUDED_GENES.startswith('^(Deprecated|NegControl|Unassigned|Intergenic|')   # proseg's own first
    for name in ('NegPrb', 'SystemControl', 'Blank', 'FalseCode', 'Negative'):
        assert name in DEFAULT_EXCLUDED_GENES
    for field in ('help', 'what'):
        assert "CosMx's NegPrb and SystemControl" in option[field] and "MERSCOPE's Blank" in option[field], field
    assert DEFAULT_EXCLUDED_GENES in option['typical']
    assert input_args({}) == ['--excluded-genes', DEFAULT_EXCLUDED_GENES]
    assert input_args({'samples': 2}) == ['--excluded-genes', DEFAULT_EXCLUDED_GENES]
    assert input_args({'excluded-genes': '^MALAT1$'}) == [] and input_args({'include-neg-ctrls': True}) == []
    assert input_args({'include-neg-ctrls': False}) == ['--excluded-genes', DEFAULT_EXCLUDED_GENES]
    # One rule for runs and Compare: Compare leaves out, for transcripts without is_gene, exactly the runs' default.
    from sopa_companion.engine.proseg_options import MAX_PATTERN
    from sopa_companion.zarr_bridge.helper import NOT_GENES_PATTERN, PATTERN_MAX
    assert DEFAULT_EXCLUDED_GENES == NOT_GENES_PATTERN and MAX_PATTERN == PATTERN_MAX


@pytest.mark.parametrize('pattern,error', [
    ('x' * 201, 'at most 200 characters'),
    ('^(NegPrb', 'not a valid pattern'),
    (r'\p{L}', 'not a valid pattern'),                    # Rust reads it, Python (Compare) does not
    ('[z-a]', 'not a valid pattern'),
    ('[[:alpha:]]', 'reads differently here than in proseg'),
    (r'(\w+)+-', 'repeats a part that itself repeats'),
    (r'(\w|\w|\w)+-', 'repeats a choice'),                 # minutes over one control probe's name in Python
])
def test_excluded_genes_must_be_a_short_pattern_python_reads(pattern, error):
    """Compare reads the run's excluded genes with Python's re, one gene name at a time; a pattern it
    cannot read, or reads differently, failed every comparison of the run, and a long or nested one could take minutes.
    Refused when the run starts, by Compare's own rule (zarr_bridge/helper.py _excluded) and in its words."""
    with pytest.raises(ValueError, match=error):
        to_args({'excluded-genes': pattern})
    assert to_args({'excluded-genes': '^(NegPrb|Blank)'}) == ['--excluded-genes', '^(NegPrb|Blank)']
    assert to_args({'excluded-genes': 'x' * 200})


def test_option_help_names_no_tissue_cell_type_or_sample():
    """The help shown on screen named the tissue and a cell type from earlier tests."""
    import json
    text = json.dumps([OPTIONS, GUIDE])
    for word in (r'\bliver\b', r'hepatocyte', r'\bDU\d\d\b', r'\bLIV\b'):
        assert not re.search(word, text, re.IGNORECASE), word
