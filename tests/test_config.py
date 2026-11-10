"""The settings files (engine/config.py): config.toml is this computer's own, config.example.toml the defaults that
ship and stand in when there is no config.toml (a fresh clone), and config.local.toml is merged over either; an
installed copy (no pyproject.toml beside the code) reads your own settings file after the example, a source tree and a
notebook session's engine never do; "auto" folders, environments and proseg-live are found as dirs.py and locate.py
say."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import shutil
import sys
import tomllib
import types

import pytest
from fastapi.testclient import TestClient

from sopa_companion import dirs, locate
from sopa_companion.engine import config as C

ROOT = C.ROOT
EXAMPLE = ROOT / "config.example.toml"
ENVIRONMENTS = {"analysis": ("python",), "zarr": ("python",), "qc": ("python",), "stain": ("python",),
                "api": ("engine_python",)}
# Settings this computer's config.toml may still hold that the code reads and ignores, so the example leaves them out.
RETIRED = {("qc", "notebook_dir"), ("qc", "gpu_busy_util"), ("qc", "gpu_quiet_s")}
# Settings only test engines use, which the engine still reads with its own defaults: keep_unsaved (keep every run
# folder, several runs, never save into a sample at close) and max_active_runs (the runs such an engine holds active at
# once). A user's engine holds one run, so the example leaves them out.
TEST_ENGINES = {("proseg", "keep_unsaved"), ("proseg", "max_active_runs")}
# A notebook's launch() reads these (api/launch.py), never the engine: this computer's config.toml need not hold them.
NOTEBOOK = {("api", "engine_python"), ("api", "scratch_dir"), ("api", "model_cache")}
# InstanSeg's settings: it is dormant behind the internal switch [stain] instanseg_enabled (off unless set), so the
# example, which users copy, names neither; this computer's config.toml may keep its environment for when it is back.
INTERNAL = {("stain", "instanseg_enabled"), ("stain", "instanseg_python")}
# Settings this computer's config.toml leaves to the defaults the code has for them, which are the example's: a line
# added to its [desktop] or [analysis] would be a difference a deploy ships only when told to (the harness's deploy
# --config-change).
DEFAULTED = {("desktop", "open_wait_s"), ("desktop", "close_grace_pending_s"), ("desktop", "away_stop_s"),
             ("analysis", "counts_spool_mb"), ("analysis", "counts_extract_share"), ("analysis", "counts_mb_s")}


def legacy_load_config(root: Path, path=None, overrides=None) -> dict:
    """load_config as it was before config.example.toml and config.local.toml: this computer's config.toml must still
    give exactly this (an engine already in use reads the same file)."""
    config = tomllib.loads((root / "config.toml").read_text())
    if path and Path(path).resolve() != root / "config.toml":
        config = C.merge(config, tomllib.loads(Path(path).read_text()))
    config = C.merge(config, overrides or {})
    data_dir = Path(config["paths"]["data_dir"])
    data_dir = (root / data_dir).resolve() if not data_dir.is_absolute() else data_dir.resolve()
    config["paths"]["data_dir"] = str(data_dir)
    config["paths"]["cache_dir"] = str(data_dir / "cache")
    config["paths"]["state_dir"] = str(Path(config["paths"].get("state_dir", "~/.local/state/sopa-companion")).expanduser().resolve())
    binary = Path(config["proseg"]["binary"]).expanduser()
    config["proseg"]["binary"] = str(binary if binary.is_absolute() else root / binary)
    # What is new, and changes nothing for a config.toml without it: "~" in environments, and threads = "auto".
    config["paths"]["sample_roots"] = [os.path.expanduser(r) for r in config["paths"].get("sample_roots", [])]
    for section, names in C.HOME_PATHS.items():
        for name in names:
            if isinstance(value := config.get(section, {}).get(name), str):
                config[section][name] = os.path.expanduser(value)
    for name, rule in C.AUTO_THREADS.items():
        if config["proseg"].get(name) == "auto":
            config["proseg"][name] = rule(C.cores())
    return config


def keys(table: dict) -> set[tuple[str, str]]:
    return {(section, key) for section, values in table.items() for key in values}


def without_locate(config: dict) -> dict:
    """The settings as the engine reads them, without what load_config found for the logs (config["locate"])."""
    return {key: value for key, value in config.items() if key != "locate"}


def has(*modules: str) -> bool:
    return all(importlib.util.find_spec(name) is not None for name in modules)


@pytest.fixture
def clone(tmp_path):
    """An app folder as a fresh clone has it: config.example.toml and no config.toml, and pyproject.toml beside the code
    (what makes a folder a source tree)."""
    root = tmp_path / "app"
    root.mkdir()
    shutil.copy(EXAMPLE, root / "config.example.toml")
    (root / "pyproject.toml").write_text('[project]\nname = "sopa-companion"\n')
    return root


@pytest.fixture
def installed(tmp_path, monkeypatch):
    """An installed copy's package folder (a wheel's): the example, no pyproject.toml, no config.toml; and a home folder
    of the test's own, where such a copy keeps its runs ("auto": ~/.local/share/sopa-companion)."""
    root = tmp_path / "env" / "site-packages" / "sopa_companion"
    root.mkdir(parents=True)
    shutil.copy(EXAMPLE, root / "config.example.toml")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    return root


@pytest.fixture
def mine():
    """Your own settings file (conftest.py points $XDG_CONFIG_HOME at a folder of the test's own)."""
    path = dirs.user_config_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture
def no_programs(tmp_path, monkeypatch):
    """No proseg-live wherever "auto" looks but an app folder's own native/bin: this Python's scripts folders stand in
    as an empty folder, and PATH is emptied."""
    empty = tmp_path / "no-programs"
    empty.mkdir()
    monkeypatch.setattr(locate, "sysconfig", types.SimpleNamespace(get_path=lambda name, scheme=None: str(empty),
                                                                   get_preferred_scheme=lambda kind: "posix_user"))
    monkeypatch.setenv("PATH", "")
    return empty


def test_this_computers_config_toml_reads_exactly_as_before(tmp_path):
    if not (ROOT / "config.toml").is_file():
        pytest.skip("no config.toml in this copy (a fresh clone): nothing of this computer's to keep")
    local = ROOT / "config.local.toml"
    # A local file merged over config.toml acts as the old --config did; there is none on a computer that never made one.
    expected = legacy_load_config(ROOT, local if local.is_file() else None)
    assert without_locate(C.load_config()) == expected                  # only what was found for the logs is new
    assert C.sources() == [ROOT / "config.toml", *([local] if local.is_file() else [])]
    # The same file in another folder (a source tree: pyproject.toml beside it), with a --config file and overrides (a
    # test engine, a second engine).
    copy = tmp_path / "copy"
    copy.mkdir()
    shutil.copy(ROOT / "config.toml", copy / "config.toml")
    (copy / "pyproject.toml").write_text('[project]\nname = "sopa-companion"\n')
    extra = tmp_path / "engine.toml"
    extra.write_text('[server]\nport = 8812\n[paths]\ndata_dir = "elsewhere"\n')
    overrides = {"desktop": {"exit_when_closed": False}, "proseg": {"threads": 3}}
    assert without_locate(C.load_config(extra, overrides, root=copy)) == legacy_load_config(copy, extra, overrides)
    assert without_locate(C.load_config(copy / "config.toml", root=copy)) == legacy_load_config(copy, copy / "config.toml")
    # And the engine's second load (create_app loads the settings again over the command's) changes nothing.
    once = C.load_config()
    assert C.load_config(overrides=once) == once


def test_a_clone_without_config_toml_runs_on_the_example(clone, no_programs):
    config = C.load_config(root=clone)
    assert C.sources(root=clone) == [clone / "config.example.toml"]
    # "auto": this Python when it has the feature's packages, else off (and the packages it lacks are said).
    for feature, (section, key, modules) in locate.FEATURES.items():
        found = config["locate"][feature]
        assert found["how"] == "auto", feature
        assert found["path"] == (os.path.abspath(sys.executable) if has(*modules) else ""), feature
        assert config[section].get(key, "") == found["path"], feature
        assert found["missing"] == [name for name in modules if not has(name)], feature
    assert config["api"]["engine_python"] == ""                             # a notebook's: the kernel's own
    assert config["paths"]["data_dir"] == str((clone / "var").resolve())
    # Not built yet: the engine keeps the place native/build.sh puts it, so a build shows without a restart.
    assert config["proseg"]["binary"] == str(clone / "native/bin/proseg-live")
    assert config["locate"]["proseg"]["how"] == "missing" and config["locate"]["proseg"]["fix"] == "build"
    assert type(config["proseg"]["threads"]) is int and type(config["proseg"]["tiled_threads"]) is int
    assert config["server"]["host"] == "127.0.0.1" and config["server"]["token"] == "auto"
    # Neither file at all: a plain message, not a traceback about config.toml.
    (clone / "config.example.toml").unlink()
    with pytest.raises(FileNotFoundError, match="neither config.toml nor config.example.toml"):
        C.load_config(root=clone)


def test_an_installed_copy_without_config_reads_your_file(installed, mine):
    """No --config: sources() and load_config() on an installed copy name your own file last (a notebook's session file
    is the only --config that keeps it out, and none is given)."""
    assert C.sources(root=installed) == [installed / "config.example.toml"]
    mine.write_text("[server]\nport = 8811\n")
    assert C.sources(root=installed) == [installed / "config.example.toml", mine]
    assert C.load_config(root=installed)["server"]["port"] == 8811
    assert C.api_session_file(None) is False and C.api_session_file("") is False


def test_an_installed_copy_reads_your_settings_file_after_its_own(installed, mine, tmp_path):
    mine.write_text('[server]\nport = 8811\n[desktop]\nchrome_flags = ["--password-store=basic"]\n[qc]\npython = "off"\n')
    config = C.load_config(root=installed)
    assert config["server"]["port"] == 8811 and config["desktop"]["chrome_flags"] == ["--password-store=basic"]
    assert config["server"]["host"] == "127.0.0.1"                         # the rest as the package's example
    assert config["locate"]["qc"]["how"] == "off" and config["qc"]["python"] == ""
    # The command line's --config goes over it, the caller's overrides over everything.
    extra = tmp_path / "engine.toml"
    extra.write_text("[server]\nport = 8812\n")
    assert C.sources(extra, root=installed) == [installed / "config.example.toml", mine, extra]
    assert C.load_config(extra, root=installed)["server"]["port"] == 8812
    assert C.load_config(extra, {"server": {"port": 8813}}, root=installed)["server"]["port"] == 8813
    assert C.sources(mine, root=installed) == [installed / "config.example.toml", mine]   # never merged twice


def test_a_source_tree_never_reads_your_settings_file(clone, mine):
    """Production and the dev copy are source trees: your notebook settings (another Python, GPUs, Chrome flags) never
    change their engines, nor the harness's."""
    mine.write_text("[server]\nport = 8811\n")
    assert C.sources(root=clone) == [clone / "config.example.toml"]
    assert C.load_config(root=clone)["server"]["port"] == 8777
    if (ROOT / "pyproject.toml").is_file():
        assert mine not in C.sources()
    assert C.sources(root=clone, user=True)[-1] == mine                    # only when asked


def test_a_notebook_sessions_engine_never_reads_your_settings_file(installed, mine, tmp_path):
    """A session's engine runs on what launch() read (your file included) and wrote into session.toml: reading your
    file again could hand it another Python, which launch() never checked."""
    mine.write_text('[server]\nport = 8811\n[analysis]\npython = "/opt/rapids/bin/python"\n')
    session = tmp_path / "session.toml"
    session.write_text('[session]\nmode = "api"\n[server]\nport = 0\n[analysis]\npython = "off"\n')
    assert C.api_session_file(session) and not C.api_session_file(tmp_path / "missing.toml")
    assert C.sources(session, root=installed) == [installed / "config.example.toml", session]
    config = C.load_config(session, root=installed)
    assert config["server"]["port"] == 0 and config["analysis"]["python"] == ""
    assert config["locate"]["analysis"]["how"] == "off"
    desktop = tmp_path / "desktop.toml"
    desktop.write_text('[session]\nmode = "desktop"\n')
    assert C.sources(desktop, root=installed) == [installed / "config.example.toml", mine, desktop]


def test_a_notebook_sessions_engine_loading_its_settings_again_never_reads_your_file(installed, mine, tmp_path,
                                                                                    monkeypatch):
    """The session's engine loads session.toml (server/api_serve.py main), and create_app loads the settings again over
    that: the second load must not bring your file in either. Whatever neither the example nor session.toml sets (a test
    engine's run limits, the Cellpose GUI's module) would otherwise reach the session's engine from your file."""
    from sopa_companion.api import protocol
    from sopa_companion.api.folders import write_config
    from sopa_companion.engine.session import Session, read_session_config
    from sopa_companion.server import create_app
    mine.write_text('[proseg]\nmax_active_runs = 7\nkeep_unsaved = true\n[stain]\ngui_module = "somebody.elses_gui"\n')
    session_toml = tmp_path / "session.toml"
    write_config(session_toml, protocol.build_session_config(
        id="1f3a9c2e", token="t" * 43, data_dir=str(tmp_path / "data"), state_dir=str(tmp_path / "state"),
        python=sys.executable, kernel_pid=os.getpid(), proseg_binary=sys.executable))
    monkeypatch.setattr(C, "ROOT", installed)                 # the app installed in the notebook's environment
    config = C.load_config(session_toml)
    again = C.load_config(overrides=config)                  # as create_app loads it
    assert "max_active_runs" not in again["proseg"] and "keep_unsaved" not in again["proseg"]
    assert "gui_module" not in again["stain"]
    # A desktop engine of the installed copy reads your file at both loads.
    desktop = C.load_config(overrides={"paths": {"data_dir": str(tmp_path / "d"), "state_dir": str(tmp_path / "s")}})
    assert desktop["proseg"]["max_active_runs"] == 7 and C.load_config(overrides=desktop)["proseg"]["max_active_runs"] == 7
    control_r, control_w = os.pipe()
    events_r, events_w = os.pipe()
    session = Session(read_session_config(session_toml), control_fd=control_r, events_fd=events_w)
    try:
        application = create_app(config, session=session)
        assert application.state.runs.max_active == 2 and application.state.runs.keep_unsaved is False
    finally:
        session.stop()
        for fd in (control_r, control_w, events_r, events_w):
            os.close(fd)


def test_a_notebook_sessions_engine_says_what_the_kernel_lacks(installed, tmp_path, monkeypatch):
    """launch() writes "off" for a feature whose "auto" found its modules missing in the kernel, with what it found
    beside it (session.toml's [locate]): the session's engine keeps that report, also when create_app loads the settings
    again, so Clusters & UMAP name the packages the kernel lacks (not an "off" nobody set), and the mixing check's probe
    still runs, to put a missing NVIDIA card first. A feature without a report stays off, as before."""
    from sopa_companion.api import protocol
    from sopa_companion.api.folders import write_config
    from sopa_companion.engine.analysis import AnalysisManager
    from sopa_companion.engine.qc import QcModels
    lacking = {"analysis": locate.Found("", "auto", ("igraph", "umap")).report(),
               "qc": locate.Found("", "auto", ("torch", "scvi", "scviva")).report()}
    session_toml = tmp_path / "session.toml"
    write_config(session_toml, protocol.build_session_config(
        id="1f3a9c2e", token="t" * 43, data_dir=str(tmp_path / "data"), state_dir=str(tmp_path / "state"),
        python=sys.executable, kernel_pid=os.getpid(), proseg_binary=sys.executable, locate=lacking))
    assert tomllib.loads(session_toml.read_text())["locate"] == lacking
    monkeypatch.setattr(C, "ROOT", installed)
    config = C.load_config(session_toml)
    for loaded in (config, C.load_config(overrides=config)):                  # the engine's load, then create_app's
        assert loaded["analysis"]["python"] == loaded["qc"]["python"] == ""
        assert {feature: loaded["locate"][feature] for feature in lacking} == lacking
        assert loaded["locate"]["stain"]["how"] == "off" and loaded["stain"]["python"] == ""
    analysis = AnalysisManager(config, runs=None, start=False)
    assert not analysis.available and analysis.missing == ["igraph", "umap-learn"]
    assert analysis.unavailable_why() == "this Python environment lacks igraph and umap-learn"
    models = QcModels(config, start=False, prober=lambda: {"ev": "probe", "missing": ["torch", "scvi-tools",
                                                                                        "scviva-tools"],
                                                             "torch": None, "cuda_build": None, "devices": None,
                                                             "nvidia": None, "error": None})
    try:
        assert models.probe_python == sys.executable
        assert models.capability()["why"] == "this Python environment lacks torch, scvi-tools and scviva-tools"
        models._probe()                                                       # a computer without an NVIDIA card
        assert (models.capability()["state"], models.capability()["reason"]) == ("unavailable", "no_gpu")
    finally:
        models.close()


def test_data_and_state_folders_auto(clone, installed, tmp_path, monkeypatch):
    """[paths] "auto" (or empty, or none): a source tree's var/, an installed copy's per-user data folder (never its
    environment's site-packages), and today's state folder whatever $XDG_STATE_HOME says."""
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "elsewhere"))
    data = home / ".local/share/sopa-companion"                  # whatever $XDG_DATA_HOME says (the conftest sets it)
    for value in ("auto", ""):
        paths = {"paths": {"data_dir": value, "state_dir": value}}
        assert C.load_config(root=clone, overrides=paths)["paths"]["data_dir"] == str((clone / "var").resolve())
        config = C.load_config(root=installed, overrides=paths)
        assert config["paths"]["data_dir"] == str(data.resolve()) and config["paths"]["cache_dir"] == str(data / "cache")
        assert config["paths"]["state_dir"] == str((home / ".local/state/sopa-companion").resolve())
    for value, folder in ((None, "var"), ("auto", "var"), ("", "var"), ("runs", "runs")):
        assert C.data_dir_of(value, clone) == (clone / folder).resolve(), value
    assert C.data_dir_of("~/runs", installed) == (home / "runs").resolve()
    assert C.data_dir_of(str(tmp_path / "abs"), installed) == (tmp_path / "abs").resolve()
    assert C.state_dir_of(None) == C.state_dir_of("auto") == (home / ".local/state/sopa-companion").resolve()
    assert C.state_dir_of("~/state") == (home / "state").resolve()


def test_environments_auto_off_and_paths_reach_the_engine(clone, tmp_path, monkeypatch, fake_binary):
    """Each environment's setting as the engine reads it: "auto" this Python when it has the feature's packages, "off"
    and a Python without them "" (off, as the code has always read it), a path as given (missing, still named). A
    loaded config handed back in (create_app loads the settings again over the command's) keeps its answers: "off"
    never turns into "auto"."""
    real = locate.python_for
    everything = lambda name: object()                      # this Python has every feature's packages
    monkeypatch.setattr(locate, "python_for", lambda feature, value=None: real(feature, value, find=everything))
    python = tmp_path / "other" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\n")
    python.chmod(0o755)
    settings = {"zarr": {"python": "auto"}, "stain": {"python": ""}, "qc": {"python": "off"},
                "analysis": {"python": str(python)}, "proseg": {"binary": str(fake_binary)}}
    config = C.load_config(root=clone, overrides=settings)
    here = os.path.abspath(sys.executable)
    assert (config["zarr"]["python"], config["stain"]["python"], config["qc"]["python"]) == (here, here, "")
    assert config["analysis"]["python"] == str(python) and config["stain"]["instanseg_python"] == here
    assert {feature: found["how"] for feature, found in config["locate"].items()} == {
        "zarr": "auto", "stain": "auto", "instanseg": "auto", "qc": "off", "analysis": "setting", "proseg": "setting"}
    again = C.load_config(root=clone, overrides=config)
    assert again["qc"]["python"] == "" and again["locate"] == config["locate"]
    # Missing packages: off, and said; a path that is not there: kept, its problem said.
    monkeypatch.setattr(locate, "python_for", real)
    gone = str(tmp_path / "gone" / "bin" / "python")
    config = C.load_config(root=clone, overrides={"analysis": {"python": gone}, "qc": {"python": "auto"},
                                                  "proseg": {"binary": str(fake_binary)}})
    assert config["analysis"]["python"] == gone and config["locate"]["analysis"]["problem"] == "missing"
    assert config["qc"]["python"] == ("" if not has("torch", "scvi", "scviva") else here)
    # They reach the engine: its parts read them where they always did.
    from sopa_companion.server import create_app
    monkeypatch.setattr(C, "ROOT", clone)
    overrides = {"server": {"port": 8799}, "desktop": {"exit_when_closed": False},
                 "paths": {"data_dir": str(tmp_path / "var"), "state_dir": str(tmp_path / "state")},
                 "proseg": {"binary": str(fake_binary)}, "zarr": {"python": str(python)}, "stain": {"python": "off"},
                 "qc": {"python": "off"}, "analysis": {"python": gone}}
    base = "http://127.0.0.1:8799"
    with TestClient(create_app(overrides, token="t" * 40), base_url=base) as client:
        health = client.get("/api/v1/health", headers={"Authorization": "Bearer " + "t" * 40, "Origin": base}).json()
    assert health["zarr"] and not health["analysis"] and not health["stain"]["available"]


def test_config_local_toml_is_merged_over_either_file(clone, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (clone / "config.local.toml").write_text('[server]\nport = 8800\n[analysis]\npython = "~/envs/rapids/bin/python"\n'
                                            '[zarr]\npython = "/opt/sopa/bin/python"\n')
    over_example = C.load_config(root=clone)
    assert C.sources(root=clone) == [clone / "config.example.toml", clone / "config.local.toml"]
    assert over_example["server"]["port"] == 8800 and over_example["server"]["host"] == "127.0.0.1"
    assert over_example["analysis"]["python"] == str(tmp_path / "home/envs/rapids/bin/python")
    assert over_example["zarr"]["python"] == "/opt/sopa/bin/python" and over_example["zarr"]["browse_start"] == ""
    assert over_example["locate"]["qc"]["how"] == "auto" and over_example["analysis"]["gpu"] == 0    # the rest as the example
    # With this computer's config.toml the local file goes over that one instead; the example is not read at all.
    mine = tomllib.loads(EXAMPLE.read_text())
    mine["server"]["port"], mine["stain"]["gpu"] = 8790, 1
    (clone / "config.toml").write_text(toml(mine))
    over_mine = C.load_config(root=clone)
    assert C.sources(root=clone) == [clone / "config.toml", clone / "config.local.toml"]
    assert over_mine["server"]["port"] == 8800 and over_mine["stain"]["gpu"] == 1
    # The command line's --config goes over both, the caller's overrides over everything.
    extra = tmp_path / "engine.toml"
    extra.write_text("[server]\nport = 8811\n")
    assert C.load_config(extra, root=clone)["server"]["port"] == 8811
    assert C.load_config(extra, {"server": {"port": 8899}}, root=clone)["server"]["port"] == 8899
    assert C.sources(extra, root=clone)[-1] == extra
    # A --config naming one of the two files is not merged twice.
    assert C.sources(clone / "config.local.toml", root=clone) == [clone / "config.toml", clone / "config.local.toml"]


def test_the_example_lists_every_setting_with_neutral_values():
    example = tomllib.loads(EXAMPLE.read_text())
    if (ROOT / "config.toml").is_file():
        mine = tomllib.loads((ROOT / "config.toml").read_text())
        assert keys(example) - NOTEBOOK - DEFAULTED <= keys(mine), \
            "the example sets something the engine's own config does not know"
        assert keys(mine) - RETIRED - TEST_ENGINES - INTERNAL <= keys(example), \
            "a setting of config.toml is missing from config.example.toml"
    assert not INTERNAL & keys(example) and "instanseg" not in EXAMPLE.read_text().lower()
    for section, names in ENVIRONMENTS.items():
        for name in names:
            # Each helper's Python is found ("auto"); a notebook's engine Python is the kernel's own (empty).
            assert example[section][name] == ("" if section == "api" else "auto"), (section, name)
    assert example["proseg"]["binary"] == example["paths"]["data_dir"] == example["paths"]["state_dir"] == "auto"
    # Clusters & UMAP on the GPU with RAPIDS where it is there, else on the CPU: the example chooses neither.
    assert example["analysis"]["backend"] == "auto"
    assert {example[s]["gpu"] for s in ("analysis", "qc", "stain")} == {0}             # every computer has a GPU 0
    assert example["proseg"]["threads"] == example["proseg"]["tiled_threads"] == "auto"
    assert example["zarr"]["browse_start"] == "" and example["paths"]["sample_roots"] == []
    # Nothing of one computer: no absolute path outside the app (only "~/..." examples), in values or comments.
    text = EXAMPLE.read_text()
    assert "/home/" not in text and "/mnt/" not in text and "/Users/" not in text
    strings = [v for values in example.values() for v in values.values() if isinstance(v, str)]
    strings += [v for values in example.values() for vs in values.values() if isinstance(vs, list) for v in vs]
    assert not [s for s in strings if s.startswith("/")]


def test_auto_threads_follow_the_cores_this_engine_may_use(clone, monkeypatch):
    for cores, threads, tiled in ((128, 32, 96), (8, 4, 6), (2, 1, 1), (1, 1, 1)):
        monkeypatch.setattr(C, "cores", lambda cores=cores: cores)
        proseg = C.load_config(root=clone)["proseg"]
        assert (proseg["threads"], proseg["tiled_threads"]) == (threads, tiled), cores
    assert C.load_config(root=clone, overrides={"proseg": {"threads": 5, "tiled_threads": 7}})["proseg"]["threads"] == 5
    with pytest.raises(ValueError, match="proseg.threads must be a positive integer"):
        C.load_config(root=clone, overrides={"proseg": {"threads": "many"}})
    monkeypatch.undo()
    assert C.cores() >= 1


def test_environments_given_from_the_home_folder_are_expanded_and_off_stays_off(clone, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    home = tmp_path / "home"
    config = C.load_config(root=clone, overrides={
        "analysis": {"python": "~/a/bin/python"}, "zarr": {"python": "~/s/bin/python", "browse_start": "~/samples"},
        "qc": {"python": "off"}, "stain": {"python": "~/s/bin/python", "instanseg_python": "~/i/bin/python"},
        "paths": {"sample_roots": ["~/bundles", "/data/bundles"], "data_dir": "~/runs"}})
    assert config["analysis"]["python"] == str(home / "a/bin/python")
    assert config["zarr"]["python"] == config["stain"]["python"] == str(home / "s/bin/python")
    assert config["zarr"]["browse_start"] == str(home / "samples")
    assert config["stain"]["instanseg_python"] == str(home / "i/bin/python")
    assert config["qc"]["python"] == ""                                               # not "." nor the home folder
    assert config["paths"]["sample_roots"] == [str(home / "bundles"), "/data/bundles"]
    assert config["paths"]["data_dir"] == str((home / "runs").resolve())


def test_the_app_starts_on_the_example_with_every_environment_off(clone, tmp_path, fake_binary, monkeypatch):
    """A fresh clone's engine (only config.example.toml) starts, and says that each environment is not set. It is a
    user's engine: one run at a time, without the test engines' settings."""
    from sopa_companion.server import create_app
    monkeypatch.setattr(C, "ROOT", clone)
    overrides = {"server": {"port": 8799}, "desktop": {"exit_when_closed": False},
                 "paths": {"data_dir": str(tmp_path / "var"), "state_dir": str(tmp_path / "state")},
                 "proseg": {"binary": str(fake_binary)},
                 **{section: {"python": "off"} for section in ("analysis", "zarr", "qc", "stain")}}
    base = "http://127.0.0.1:8799"
    with TestClient(create_app(overrides, token="t" * 40), base_url=base) as client:
        health = client.get("/api/v1/health", headers={"Authorization": "Bearer " + "t" * 40, "Origin": base}).json()
    assert health["ok"] and health["proseg"] and health["threads"] >= 1
    assert not health["zarr"] and not health["analysis"] and not health["qc"]["available"]
    assert health["stain"] == {"available": False, "methods": {"cellpose": False}}       # InstanSeg is dormant: not listed
    assert health["one_run"] is True and health["max_active_runs"] == 2


def test_the_settings_of_proseg_s_own_count_are_checked_and_left_as_given(clone):
    """[analysis] counts_spool_mb (a whole number of MB, 64 or more), counts_extract_share (above 0, at most 1) and
    counts_mb_s (0 or more; 0: no cap): none is an option on screen, the example lists each at the default its reader
    has, a value is never changed or added, and one out of range stops the engine at its start, named."""
    example = tomllib.loads(EXAMPLE.read_text())["analysis"]
    assert {key: example[key] for key in C.COUNTS_SETTINGS} == C.COUNTS_SETTINGS == \
        {"counts_spool_mb": 1024, "counts_extract_share": 0.10, "counts_mb_s": 0}
    for key, value in (("counts_spool_mb", 64), ("counts_spool_mb", 4096), ("counts_extract_share", 1),
                       ("counts_extract_share", 0.05), ("counts_mb_s", 0), ("counts_mb_s", 12.5)):
        assert C.load_config(root=clone, overrides={"analysis": {key: value}})["analysis"][key] == value
    for key, value in (("counts_spool_mb", 63), ("counts_spool_mb", 512.0), ("counts_spool_mb", True),
                       ("counts_spool_mb", "1024"), ("counts_extract_share", 0), ("counts_extract_share", 1.5),
                       ("counts_extract_share", float("nan")), ("counts_extract_share", "0.1"), ("counts_mb_s", -1),
                       ("counts_mb_s", float("inf")), ("counts_mb_s", False)):
        with pytest.raises(ValueError, match=f"analysis.{key} must be"):
            C.load_config(root=clone, overrides={"analysis": {key: value}})
    C.check_counts({})                                         # none given: each reader's default
    (clone / "config.example.toml").write_text(EXAMPLE.read_text().replace("counts_spool_mb = 1024\n", ""))
    assert "counts_spool_mb" not in C.load_config(root=clone)["analysis"]


def test_settings_only_test_engines_use_stay_out_of_the_example(clone):
    """keep_unsaved and max_active_runs are test engines' settings: a user who turned keep_unsaved on would get runs
    that are never saved at close and pile up unseen (the page lists none), and max_active_runs limits nothing on an
    engine that holds one run. The example, which users copy, leaves both out; the engine reads them with its
    defaults (a user's one-run engine), and a test engine's own settings still set them."""
    example = tomllib.loads(EXAMPLE.read_text())
    assert not TEST_ENGINES & keys(example)
    assert not {"keep_unsaved", "max_active_runs"} & C.load_config(root=clone)["proseg"].keys()
    test_engine = C.load_config(root=clone, overrides={"proseg": {"keep_unsaved": True, "max_active_runs": 3}})
    assert test_engine["proseg"]["keep_unsaved"] is True and test_engine["proseg"]["max_active_runs"] == 3


def test_the_example_says_what_a_notebook_reads(clone):
    """A notebook's launch() reads the example (an installed wheel's own settings file) and the user's own
    ~/.config/sopa-companion/config.toml: the example lists its [api] settings, neutral, and says which tables launch()
    takes and that "auto" is the kernel's own Python there. The engine never reads [api]."""
    example = tomllib.loads(EXAMPLE.read_text())
    assert {("api", key) for key in example["api"]} == NOTEBOOK
    assert all(value == "" for value in example["api"].values())
    text = " ".join(line.lstrip("# ").strip() for line in EXAMPLE.read_text().splitlines() if line.startswith("#"))
    assert "~/.config/sopa-companion/config.toml" in text and "$XDG_CONFIG_HOME" in text
    for table in ("[api]", "[desktop] browser and chrome_flags", "[proseg] binary", "[analysis] python and gpu",
                  "[stain] python and gpu", "[qc] python and gpu"):
        assert table in text, table
    assert "other settings there do not reach the session's engine" in text
    assert "kernel's own Python when it has the packages" in text
    # The engine's settings carry the table as it is (nothing of the engine reads it), so a clone still starts on it.
    assert C.load_config(root=clone)["api"] == {"engine_python": "", "scratch_dir": "", "model_cache": ""}


def toml(table: dict) -> str:
    """A small TOML writer for the test's own settings files (sections of scalars and lists)."""
    def value(v):
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, str):
            return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
        if isinstance(v, list):
            return "[" + ", ".join(value(x) for x in v) + "]"
        return repr(v)
    return "".join(f"[{section}]\n" + "".join(f"{k} = {value(v)}\n" for k, v in values.items()) + "\n"
                   for section, values in table.items())


def test_the_settings_files_stay_out_of_git_and_the_example_ships():
    rules = (ROOT / ".gitignore").read_text().splitlines()
    assert "/config.toml" in rules and "/config.local.toml" in rules
    assert not any(rule.strip("/") == "config.example.toml" for rule in rules)
    assert os.access(EXAMPLE, os.R_OK)
