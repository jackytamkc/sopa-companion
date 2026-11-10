"""The kernel/engine boundary's messages and settings: wire fixtures, malformed inputs and the import fence."""
from __future__ import annotations

from copy import deepcopy
import inspect
import io
import json
from pathlib import Path
import subprocess
import sys
import tomllib
from typing import get_type_hints

import pytest

from sopa_companion.api import protocol as P


TOKEN = "t" * 43
CONFIG_ARGS = {"id": "1f3a9c2e", "token": TOKEN, "data_dir": "/scratch/session", "state_dir": "/run/session",
               "python": "/opt/sopa/bin/python", "kernel_pid": 1234}
CONFIG = tomllib.loads(f'''
[server]
port = 0
token = "{TOKEN}"
[paths]
data_dir = "/scratch/session"
state_dir = "/run/session"
sample_roots = []
[desktop]
exit_when_closed = false
profile_dir = "auto"
[zarr]
python = "/opt/sopa/bin/python"
browse_start = ""
[qc]
python = "off"
enabled_by_default = false
gpu = 0
models_dir = ""
[stain]
python = "off"
gpu = 0
instanseg_python = "off"
instanseg_enabled = false
[analysis]
python = "off"
gpu = 0
[session]
mode = "api"
id = "1f3a9c2e"
tool = ""
open = "none"
patches_dir = ""
link_code_ttl_s = 600
kernel_pid = 1234
''')

CONTROLS = [
    {"op": "hello", "protocol": 1},
    {"op": "prepare", "done": 0, "total": 200},
    {"op": "prepare_failed", "message": "Could not prepare the sample"},
    {"op": "bind", "path": "/scratch/view.zarr", "view": "overlay", "name": "SMP02", "backed": True,
     "zarr": "/data/SMP02.zarr", "memory_only": ["extra_shapes"], "points_key": "transcripts",
     "linked": ["transcripts"], "written": ["extra_shapes"]},
    {"op": "object", "names": ["transcripts", "extra_shapes"], "zarr": "/data/SMP02.zarr", "auto_save": True},
    {"op": "notebook", "busy": False},
    {"op": "received", "id": "delivery-1"},
    {"op": "receive_failed", "id": "delivery-1", "message": "Could not read the result"},
    {"op": "applied", "id": "delivery-1", "state": "added", "keys": ["cells", "counts"], "clash": [],
     "message": None},
    {"op": "close", "pending": "discard"},
]
CONTROL_BY_OP = {message["op"]: message for message in CONTROLS}

DELIVERY = {"ev": "deliver", "id": "delivery-1", "run": "run-1", "kind": "proseg",
            "staged": "/scratch/deliveries/1/staged.zarr", "shapes_key": "cells", "table_key": "counts",
            "set_default": True, "overwrite": False, "replaced": [], "written_to": "/data/SMP02.zarr",
            "cells": 40_070, "genes": 500, "provenance": {"run": "run-1", "region": {"kind": "full"}}}
EVENTS = [
    {"ev": "ready", "port": 41237, "pid": 1235},
    {"ev": "session", "state": "ready", "message": None},
    DELIVERY,
    {"ev": "closing", "pending": "discard", "by": "window"},
    {"ev": "ended", "added": ["delivery-1"], "not_added": [
        {"run": "run-2", "reason": "discarded", "message": "Discarded at close"}]},
    {"ev": "link", "state": "used"},          # a link the handle showed signed a browser in
]
EVENT_BY_NAME = {message["ev"]: message for message in EVENTS}

HEALTH = {"mode": "api", "id": "1f3a9c2e",
          "sample": {"id": "z123", "name": "SMP02", "backed": True, "zarr_name": "SMP02.zarr",
                     "view": "in_place", "state": "ready", "progress": {"done": 0, "total": 0},
                     "message": None, "memory_only": [], "tools": ["proseg", "stain"]},
          "tool": None, "writes": "zarr", "notebook": {"busy": False}, "patches": False, "link": False,
          "state": "running"}
SAVES = [
    {"state": "saving", "stage": "building", "message": "Building the cells"},
    {"state": "queued", "written_to": "/data/SMP02.zarr"},
    {"state": "saved", "added": True, "cells": 40070, "shapes_key": "cells", "table_key": "counts",
     "written_to": "/data/SMP02.zarr", "at": "2026-10-23T12:00:00+00:00"},
    {"state": "failed", "message": "Could not save the cells"},
]
SAVE_BY_STATE = {message["state"]: message for message in SAVES}


def test_constants_match_the_interfaces_and_binding_close_answer():
    assert P.ENV_EVENTS_FD == "SOPA_COMPANION_EVENTS_FD"
    assert P.PROTOCOL_VERSION == 1
    assert P.SESSION_STATES == {"starting", "preparing", "ready", "closing", "closed", "failed"}
    assert P.SAMPLE_STATES == {"preparing", "opening", "ready", "failed"}
    assert P.HEALTH_STATES == {"running", "ending"}
    assert P.CLOSE_POLICIES == {"add", "discard"}
    assert P.DEFAULT_CLOSE_POLICY == "discard"
    assert P.VIEW_MODES == {"in_place", "overlay", "copy"}
    assert P.CONTROL_OPS == set(CONTROL_BY_OP)
    assert P.EVENTS == set(EVENT_BY_NAME)
    assert P.ZARR_SAVE_STATES == set(SAVE_BY_STATE)


def test_session_config_builds_the_exact_i1_tables_and_reads_toml():
    assert P.build_session_config(**CONFIG_ARGS) == CONFIG
    assert P.validate_session_config(CONFIG) == CONFIG


@pytest.mark.parametrize("open,exit_when_closed", [("window", True), ("link", False), ("none", False)])
def test_session_config_window_lifetime(open, exit_when_closed):
    config = P.build_session_config(**CONFIG_ARGS, open=open)
    assert config["desktop"]["exit_when_closed"] is exit_when_closed
    assert config["session"]["open"] == open


def test_session_config_optional_environments_and_paths():
    expected = deepcopy(CONFIG)
    expected["session"].update(tool="stain", patches_dir="/scratch/patches")
    expected["qc"].update(python="/opt/sopa/bin/python", enabled_by_default=True, gpu=2, models_dir="/scratch/models")
    expected["stain"].update(python="/opt/sopa/bin/python", gpu=1, instanseg_python="/opt/instanseg/bin/python",
                             instanseg_enabled=True)          # the dormant InstanSeg's internal switch, on
    expected["analysis"].update(python="/opt/rapids/bin/python", gpu=3)
    assert P.build_session_config(**CONFIG_ARGS, tool="stain", patches_dir="/scratch/patches",
                                  qc_python="/opt/sopa/bin/python", mixing_check=True, qc_gpu=2,
                                  model_cache="/scratch/models", stain_python="/opt/sopa/bin/python", stain_gpu=1,
                                  instanseg_python="/opt/instanseg/bin/python", instanseg_enabled=True,
                                  analysis_python="/opt/rapids/bin/python", analysis_gpu=3) == expected


@pytest.mark.parametrize("section", CONFIG)
def test_session_config_requires_each_table(section):
    value = deepcopy(CONFIG)
    del value[section]
    with pytest.raises(ValueError, match=section):
        P.validate_session_config(value)


@pytest.mark.parametrize("section,field", [(section, field) for section, table in CONFIG.items() for field in table])
def test_session_config_requires_each_field(section, field):
    value = deepcopy(CONFIG)
    del value[section][field]
    with pytest.raises(ValueError, match=field):
        P.validate_session_config(value)


@pytest.mark.parametrize("section,field,bad", [
    ("server", "port", 8777), ("server", "port", False), ("server", "token", "auto"),
    ("server", "token", "+" * 43), ("server", "token", "t" * 42), ("server", "token", "t" * 44),
    ("paths", "data_dir", ""), ("paths", "state_dir", None), ("paths", "sample_roots", ["/data"]),
    ("paths", "sample_roots", {}), ("desktop", "exit_when_closed", 1), ("desktop", "profile_dir", "/desktop"),
    ("zarr", "python", ""), ("zarr", "browse_start", "/data"), ("qc", "python", None),
    ("qc", "enabled_by_default", "false"), ("qc", "gpu", True), ("qc", "models_dir", None),
    ("stain", "python", None), ("stain", "gpu", -1), ("stain", "instanseg_python", None),
    ("stain", "instanseg_enabled", "false"), ("stain", "instanseg_enabled", 1),
    ("analysis", "python", None), ("analysis", "gpu", 0.0), ("session", "mode", "desktop"),
    ("session", "id", "abcdefgh"), ("session", "id", "1f3a9c2"), ("session", "id", "1f3a9c2ee"),
    ("session", "tool", "unknown"), ("session", "tool", None), ("session", "open", "auto"),
    ("session", "patches_dir", None),
    ("session", "link_code_ttl_s", 45), ("session", "link_code_ttl_s", 600.0),
    ("session", "kernel_pid", 0), ("session", "kernel_pid", True),
])
def test_session_config_rejects_bad_values_without_exposing_the_token(section, field, bad):
    value = deepcopy(CONFIG)
    value[section][field] = bad
    with pytest.raises(ValueError, match=field) as error:
        P.validate_session_config(value)
    assert TOKEN not in str(error.value)


@pytest.mark.parametrize("section,field", [("qc", "python"), ("stain", "python"), ("stain", "instanseg_python"),
                                           ("analysis", "python")])
def test_session_config_interpreters_are_off_or_a_full_path(section, field):
    """launch() resolves "auto" in the kernel: the session's engine, which may run in another interpreter, must never
    look again, so session.toml holds "off" or a full path, nothing else (a string check: no path is opened)."""
    for good in ("off", "/opt/sopa/bin/python", "/no/such/python"):
        value = deepcopy(CONFIG)
        value[section][field] = good
        assert P.validate_session_config(value)[section][field] == good
    for bad in ("", "auto", "Off", "bin/python", "~/envs/sopa/bin/python", 0, None):
        value = deepcopy(CONFIG)
        value[section][field] = bad
        with pytest.raises(ValueError, match=field):
            P.validate_session_config(value)
    built = P.build_session_config(**CONFIG_ARGS)
    assert built[section][field] == "off"                                # the builder's default: off


def test_session_config_carries_what_auto_found_missing_in_the_kernel():
    """[locate.<feature>]: for a feature whose "auto" found its modules missing in the kernel (its python "off"), what
    it found, so the session's engine can say what the kernel lacks; only the mixing check's, stain segmentation's and
    Clusters & UMAP's, only such a report, and only beside an "off"."""
    report = {"path": "", "how": "auto", "missing": ["igraph", "umap"], "problem": "", "fix": "", "passed": []}
    lacking = {"analysis": report, "qc": {**report, "missing": ["torch", "scvi", "scviva"]}}
    built = P.build_session_config(**CONFIG_ARGS, locate=lacking)
    assert built == {**CONFIG, "locate": lacking} and P.validate_session_config(built) == built
    assert "locate" not in P.build_session_config(**CONFIG_ARGS, locate={})
    for bad, words in (({"zarr": report}, "may name only"), ([report], "may name only"),
                       ({"analysis": {**report, "path": "/opt/sopa/bin/python"}}, "path"),
                       ({"analysis": {**report, "how": "setting"}}, "how"), ({"analysis": {**report, "missing": []}},
                                                                             "missing"),
                       ({"analysis": {**report, "missing": "igraph"}}, "missing")):
        with pytest.raises(ValueError, match=words):
            P.validate_session_config({**deepcopy(CONFIG), "locate": bad})
    value = deepcopy(CONFIG)
    value["analysis"]["python"] = "/opt/sopa/bin/python"
    with pytest.raises(ValueError, match="not off"):
        P.validate_session_config({**value, "locate": {"analysis": report}})


@pytest.mark.parametrize("changes", [{"open": "auto"}, {"tool": "other"}, {"token": "auto"}, {"kernel_pid": True},
                                    {"qc_gpu": -1}, {"mixing_check": 1}, {"model_cache": 5}])
def test_session_config_builder_validates(changes):
    with pytest.raises(ValueError):
        P.build_session_config(**{**CONFIG_ARGS, **changes})


@pytest.mark.parametrize("value", CONTROLS, ids=lambda value: value["op"])
def test_control_build_validate_and_json_round_trip(value):
    fields = {key: item for key, item in value.items() if key != "op"}
    assert P.build_control(value["op"], **fields) == value
    assert P.validate_control(value) == value
    assert P.decode_control_line(P.encode_line(value)) == value


@pytest.mark.parametrize("value", EVENTS, ids=lambda value: value["ev"])
def test_event_build_validate_and_json_round_trip(value):
    fields = {key: item for key, item in value.items() if key != "ev"}
    assert P.build_event(value["ev"], **fields) == value
    assert P.validate_event(value) == value
    assert P.decode_event_line(P.encode_line(value).encode()) == value


@pytest.mark.parametrize("tag,value,field", [(tag, value, field)
    for tag, messages in (("op", CONTROLS), ("ev", EVENTS)) for value in messages for field in value])
def test_messages_require_every_documented_field(tag, value, field):
    value = deepcopy(value)
    del value[field]
    with pytest.raises(ValueError, match=field):
        (P.validate_control if tag == "op" else P.validate_event)(value)


@pytest.mark.parametrize("op,field,bad", [
    ("hello", "protocol", 2), ("hello", "protocol", True), ("hello", "protocol", 1.0),
    ("prepare", "done", -1), ("prepare", "total", True), ("prepare", "done", 0.5),
    ("prepare_failed", "message", None), ("bind", "path", ""), ("bind", "view", "other"),
    ("bind", "name", ""), ("bind", "backed", 1), ("bind", "zarr", False),
    ("bind", "memory_only", "shapes"), ("bind", "points_key", []),
    ("bind", "linked", [False]), ("bind", "written", [""]),
    ("object", "names", [None]), ("object", "zarr", ""), ("object", "auto_save", "true"),
    ("notebook", "busy", 0), ("received", "id", ""), ("receive_failed", "id", None),
    ("receive_failed", "message", False), ("applied", "id", 1), ("applied", "state", "saved"),
    ("applied", "keys", {}), ("applied", "clash", "cells"), ("applied", "message", 2),
    ("close", "pending", "save"), ("close", "pending", None), ("close", "pending", "stay"),
])
def test_control_builders_and_readers_reject_malformed_known_ops(op, field, bad):
    value = deepcopy(CONTROL_BY_OP[op])
    value[field] = bad
    with pytest.raises(ValueError, match=field):
        P.build_control(**value)
    with pytest.raises(ValueError, match=field):
        P.decode_control_line(json.dumps(value))


@pytest.mark.parametrize("ev,field,bad", [
    ("ready", "port", 0), ("ready", "port", 65536), ("ready", "port", True),
    ("ready", "pid", 0), ("ready", "pid", True), ("session", "state", "running"),
    ("session", "message", 0), ("deliver", "id", ""), ("deliver", "run", None),
    ("deliver", "kind", "patches"), ("deliver", "staged", ""), ("deliver", "shapes_key", ""),
    ("deliver", "table_key", []), ("deliver", "set_default", 1), ("deliver", "overwrite", 0),
    ("deliver", "replaced", [1]), ("deliver", "written_to", False),
    ("deliver", "cells", True), ("deliver", "cells", -1), ("deliver", "genes", -1),
    ("deliver", "provenance", []), ("closing", "pending", "save"), ("closing", "by", "server"),
    ("ended", "added", [None]), ("ended", "not_added", {}), ("link", "state", "shown"), ("link", "state", None),
])
def test_event_builders_and_readers_reject_malformed_known_events(ev, field, bad):
    value = deepcopy(EVENT_BY_NAME[ev])
    value[field] = bad
    with pytest.raises(ValueError, match=field):
        P.build_event(**value)
    with pytest.raises(ValueError, match=field):
        P.decode_event_line(json.dumps(value))


@pytest.mark.parametrize("state", ["starting", "preparing", "ready", "closing", "closed", "failed"])
def test_every_handle_state_can_be_sent(state):
    assert P.build_event("session", state=state, message=None) == {"ev": "session", "state": state, "message": None}


@pytest.mark.parametrize("by", ["page", "handle", "window", "kernel"])
@pytest.mark.parametrize("pending", ["add", "discard"])
def test_close_policies_and_origins(by, pending):
    assert P.build_control("close", pending=pending) == {"op": "close", "pending": pending}
    assert P.build_event("closing", pending=pending, by=by) == {"ev": "closing", "pending": pending, "by": by}


@pytest.mark.parametrize("view", ["in_place", "overlay", "copy"])
def test_bind_without_transcripts_or_a_target(view):
    value = {**CONTROL_BY_OP["bind"], "view": view, "backed": False, "zarr": None, "points_key": None,
             "memory_only": [], "linked": [], "written": []}
    assert P.build_control(**value) == value
    value = {"op": "object", "names": [], "zarr": None, "auto_save": False}
    assert P.build_control(**value) == value


def test_stain_delivery_can_have_no_table_genes_or_written_target():
    value = {**DELIVERY, "kind": "stain", "table_key": None, "genes": None, "written_to": None,
             "set_default": False, "overwrite": True, "replaced": ["cells"],
             "provenance": {"resolution": {"level": 1, "factor": 2.0}, "filters": {}, "covered": None}}
    assert P.build_event(**value) == value
    assert P.decode_event_line(P.encode_line(value)) == value


def test_applied_failure_and_empty_ended_event():
    value = {"op": "applied", "id": "delivery-1", "state": "failed", "keys": [], "clash": ["cells"],
             "message": "The name was added in the notebook"}
    assert P.build_control(**value) == value
    assert P.build_event("ended", added=[], not_added=[]) == {"ev": "ended", "added": [], "not_added": []}


@pytest.mark.parametrize("reason", ["discarded", "failed", "clash", "incomplete"])
def test_ended_reasons(reason):
    value = {"ev": "ended", "added": [], "not_added": [{"run": "run-1", "reason": reason, "message": "Not added"}]}
    assert P.build_event(**value) == value


@pytest.mark.parametrize("item", [None, "run-1", {}, {"run": "run-1", "reason": "failed"},
    {"run": "", "reason": "failed", "message": "Failed"},
    {"run": "run-1", "reason": "closed", "message": "Closed"},
    {"run": "run-1", "reason": "failed", "message": None}])
def test_ended_validates_each_not_added_record(item):
    with pytest.raises(ValueError, match="not_added"):
        P.build_event("ended", added=[], not_added=[item])


@pytest.mark.parametrize("tag,builder,validator,reader", [
    ("op", P.build_control, P.validate_control, P.decode_control_line),
    ("ev", P.build_event, P.validate_event, P.decode_event_line),
])
def test_unknown_names_are_logged_and_ignored_by_readers_only(tag, builder, validator, reader, caplog):
    value = {tag: "future", "extra": {"values": [1, None]}}
    with pytest.raises(ValueError, match="Unknown"):
        builder("future", extra=value["extra"])
    with pytest.raises(ValueError, match="Unknown"):
        validator(value)
    assert validator(value, allow_unknown=True) is None
    assert reader(P.encode_line(value)) is None
    assert "Ignoring unknown API" in caplog.text and "future" in caplog.text


def test_a_reader_continues_after_an_unknown_op():
    stream = io.StringIO('{"op":"hello","protocol":1}\n{"op":"future"}\n{"op":"close","pending":"discard"}\n')
    messages = [value for line in stream if (value := P.decode_control_line(line)) is not None]
    assert messages == [{"op": "hello", "protocol": 1}, {"op": "close", "pending": "discard"}]


def test_json_lines_escape_text_and_accept_utf8_and_crlf():
    value = P.build_control("prepare_failed", message='A \"sample\"\nwith µm and \\ in its name')
    line = P.encode_line(value)
    assert isinstance(line, str) and line.endswith("\n") and line.count("\n") == 1
    assert P.decode_line(line) == value
    assert P.decode_line(line.rstrip("\n")) == value
    assert P.decode_control_line(json.dumps(value, ensure_ascii=False).encode() + b"\r\n") == value


@pytest.mark.parametrize("line", ["", " ", "[]", "null", "true", "1", '"hello"', "{", "{} {}", "{}\n{}\n",
    "{}\n\n", '{\n"op":"hello","protocol":1}', '{}\r', b"\xff", 5, None,
    '{"op":"hello","op":"close","protocol":1}', '{"nested":{"same":1,"same":2}}',
    '{"bad":NaN}', '{"bad":Infinity}', '{"bad":-Infinity}', '{"bad":1e999}'])
def test_decode_line_rejects_invalid_json_objects_and_framing(line):
    with pytest.raises(ValueError):
        P.decode_line(line)


@pytest.mark.parametrize("value", [None, [], "hello", {1: "key"}, {"bad": {1: "nested key"}},
    {"bad": float("nan")}, {"bad": [float("inf")]}, {"bad": float("-inf")},
    {"bad": (1, 2)}, {"bad": {1, 2}}, {"bad": object()}])
def test_encoder_rejects_non_json_and_non_finite_values(value):
    with pytest.raises(ValueError):
        P.encode_line(value)


def test_cycles_raise_value_error():
    value = {}
    value["cycle"] = value
    with pytest.raises(ValueError, match="cycle"):
        P.encode_line(value)


@pytest.mark.parametrize("value", [{}, {"op": ""}, {"op": []}, {"ev": "hello"}, {"op": "hello"}])
def test_unknown_op_tolerance_does_not_hide_malformed_controls(value):
    with pytest.raises(ValueError):
        P.decode_control_line(json.dumps(value))


@pytest.mark.parametrize("value", [{}, {"ev": ""}, {"ev": []}, {"op": "ready"}, {"ev": "ready"}])
def test_unknown_event_tolerance_does_not_hide_malformed_events(value):
    with pytest.raises(ValueError):
        P.decode_event_line(json.dumps(value))


def test_health_builds_the_i5_shape_and_desktop_shape():
    assert P.build_health_session(**HEALTH) == HEALTH
    assert P.validate_health_session(HEALTH) == HEALTH
    assert P.build_health_session("desktop") == {"mode": "desktop"}
    assert P.validate_health_session({"mode": "desktop"}) == {"mode": "desktop"}


@pytest.mark.parametrize("state", ["preparing", "opening", "ready", "failed"])
@pytest.mark.parametrize("writes", ["zarr", "memory", "memory_autosave_off"])
@pytest.mark.parametrize("session_state", ["running", "ending"])
def test_health_states_and_write_modes(state, writes, session_state):
    value = deepcopy(HEALTH)
    value.update(writes=writes, state=session_state, patches=True, link=True, tool="stain", notebook={"busy": True})
    value["sample"]["state"] = state
    assert P.build_health_session(**value) == value


def test_health_before_binding_an_unbacked_sample():
    value = deepcopy(HEALTH)
    value["sample"].update(id=None, name=None, zarr_name=None, view=None, backed=False, state="preparing", tools=[])
    value["writes"] = "memory"
    assert P.build_health_session(**value) == value
    value["ev"] = "session"
    assert P.validate_health_session(value) == value      # The page's push carries the same health.session fields.


@pytest.mark.parametrize("path", [(field,) for field in HEALTH] + [("sample", field) for field in HEALTH["sample"]]
    + [("sample", "progress", field) for field in ("done", "total")] + [("notebook", "busy")])
def test_health_requires_every_field(path):
    value = deepcopy(HEALTH)
    target = value
    for key in path[:-1]:
        target = target[key]
    del target[path[-1]]
    with pytest.raises(ValueError, match=path[-1]):
        P.validate_health_session(value)


@pytest.mark.parametrize("path,bad", [
    (("mode",), "other"), (("id",), "invalid!"), (("sample",), []), (("tool",), ""),
    (("writes",), "disk"), (("notebook",), False), (("patches",), 1), (("link",), "false"),
    (("state",), "closed"), (("notebook", "busy"), 0), (("sample", "id"), ""),
    (("sample", "name"), 2), (("sample", "backed"), "true"), (("sample", "zarr_name"), False),
    (("sample", "view"), "disk"), (("sample", "state"), "running"), (("sample", "progress"), []),
    (("sample", "progress", "done"), -1), (("sample", "progress", "total"), True),
    (("sample", "message"), 0), (("sample", "memory_only"), [None]), (("sample", "tools"), "stain"),
    (("sample", "tools"), ["unknown"]),
])
def test_health_builder_and_validator_reject_bad_fields(path, bad):
    value = deepcopy(HEALTH)
    target = value
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = bad
    with pytest.raises(ValueError, match=path[-1]):
        P.build_health_session(**value)
    with pytest.raises(ValueError, match=path[-1]):
        P.validate_health_session(value)


@pytest.mark.parametrize("value", SAVES, ids=lambda value: value["state"])
def test_zarr_save_builder_and_validator(value):
    assert P.build_zarr_save(**value) == value
    assert P.validate_zarr_save(value) == value


def test_zarr_save_stain_memory_and_failure_fields():
    value = {"state": "saving", "stage": "writing", "message": None}
    assert P.build_zarr_save(**value) == value
    value = {"state": "queued", "written_to": None}
    assert P.build_zarr_save(**value) == value
    value = {**SAVE_BY_STATE["saved"], "table_key": None, "written_to": None}
    assert P.build_zarr_save(**value) == value
    value = {"state": "failed", "message": "The name is already in the notebook", "written_to": "/data/SMP02.zarr",
             "clash": ["cells"]}
    assert P.build_zarr_save(**value) == value


@pytest.mark.parametrize("value,field", [(value, field) for value in SAVES for field in value])
def test_zarr_save_requires_each_field(value, field):
    value = deepcopy(value)
    del value[field]
    with pytest.raises(ValueError, match=field):
        P.validate_zarr_save(value)


@pytest.mark.parametrize("state,field,bad", [
    ("saving", "state", "done"), ("saving", "stage", "queued"), ("saving", "message", 1),
    ("queued", "written_to", ""), ("saved", "added", False), ("saved", "added", 1),
    ("saved", "cells", -1), ("saved", "cells", True), ("saved", "shapes_key", ""),
    ("saved", "table_key", False), ("saved", "written_to", []), ("saved", "at", None),
    ("failed", "message", None), ("failed", "written_to", 2), ("failed", "clash", "cells"),
])
def test_zarr_save_rejects_invalid_required_and_optional_fields(state, field, bad):
    value = {**SAVE_BY_STATE[state], field: bad}
    with pytest.raises(ValueError, match=field):
        P.build_zarr_save(**value)
    with pytest.raises(ValueError, match=field):
        P.validate_zarr_save(value)


@pytest.mark.parametrize("answer,expected", [("add", "add"), ("discard", "discard"), ("save", "add")])
@pytest.mark.parametrize("end", [None, False, True])
def test_closing_alias_and_optional_end(answer, expected, end):
    original = {"answer": answer, **({"end": end} if end is not None else {})}
    wanted = {**original, "answer": expected}
    assert P.build_closing(answer, end=end) == wanted
    assert P.validate_closing(original) == wanted
    assert original["answer"] == answer


@pytest.mark.parametrize("value", [{}, {"answer": "stay"}, {"answer": "ADD"}, {"answer": None},
    {"answer": "add", "end": None}, {"answer": "add", "end": 1}, {"answer": "discard", "end": "false"}])
def test_closing_rejects_missing_answers_and_invalid_fields(value):
    with pytest.raises(ValueError):
        P.validate_closing(value)


@pytest.mark.parametrize("fields", [{"answer": "stay"}, {"answer": None}, {"answer": "add", "end": 1}])
def test_closing_builder_checks_its_inputs(fields):
    with pytest.raises(ValueError):
        P.build_closing(**fields)


@pytest.mark.parametrize("validator,value", [(P.validate_session_config, CONFIG), (P.validate_control, CONTROLS[0]),
    (P.validate_event, DELIVERY), (P.validate_health_session, HEALTH), (P.validate_zarr_save, SAVES[0]),
    (P.validate_closing, {"answer": "save"})])
def test_validators_preserve_extensions_and_copy_nested_values(validator, value):
    value = {**deepcopy(value), "extra": {"items": ["kept"]}}
    original = deepcopy(value)
    checked = validator(value)
    assert value == original and checked["extra"] == value["extra"]
    checked["extra"]["items"].append("changed")
    assert value == original


@pytest.mark.parametrize("validator", [P.validate_session_config, P.validate_control, P.validate_event,
                                      P.validate_health_session, P.validate_zarr_save, P.validate_closing])
@pytest.mark.parametrize("bad", [None, [], "{}", {"extra": float("nan")}])
def test_validators_require_json_objects(validator, bad):
    with pytest.raises(ValueError):
        validator(bad)


def test_session_protocol_has_the_i8_names_and_signatures():
    assert get_type_hints(P.Session) == {"mode": str, "bound": str | None, "view_path": str | None,
                                        "notebook_busy": bool}
    for name, parameters, result in [
        ("target", ["self"], str | None), ("names", ["self"], set[str]),
        ("deliver", ["self", "event"], type(None)), ("received", ["self", "id", "timeout"], bool),
        ("applied", ["self", "id"], dict | None), ("closing_policy", ["self"], str | None),
    ]:
        method = getattr(P.Session, name)
        assert list(inspect.signature(method).parameters) == parameters
        assert get_type_hints(method)["return"] == result

    class EngineSession:
        mode = "api"
        bound = None
        view_path = None
        notebook_busy = False

        def target(self): return None
        def names(self): return set()
        def deliver(self, event): pass
        def received(self, id, timeout): return False
        def applied(self, id): return None
        def closing_policy(self): return None

    assert isinstance(EngineSession(), P.Session)
    assert not isinstance(object(), P.Session)
    with pytest.raises(TypeError):
        P.Session()


def test_protocol_imports_with_only_the_standard_library():
    script = '''
import sys
sys.path.insert(0, sys.argv[1])
import sopa_companion.api.protocol
unexpected = {name.split(".")[0] for name in sys.modules} - sys.stdlib_module_names - {"__main__", "sopa_companion"}
assert not unexpected, sorted(unexpected)
assert not any(name.startswith(("sopa_companion.engine", "sopa_companion.server")) for name in sys.modules)
# The api package's window module takes the desktop's browser opener and Chrome's command (standard library only), no
# more: never desktop/window.py, which brings psutil and the engine's settings.
desktop = {name for name in sys.modules if name.startswith("sopa_companion.desktop")}
assert desktop <= {"sopa_companion.desktop", "sopa_companion.desktop.chrome", "sopa_companion.desktop.browser"}, \
    sorted(desktop)
assert "psutil" not in sys.modules and "sopa_companion.desktop.window" not in sys.modules
'''
    result = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", script, str(Path(P.__file__).resolve().parents[2])],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
