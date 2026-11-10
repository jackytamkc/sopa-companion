"""The API mode's shared contract: session.toml, the control lines, the events, health.session and the closing answer.

Builders return ordinary dictionaries; validators return detached copies and raise ValueError on invalid input.
Extra fields are preserved for forward compatibility. build_session_config returns the tables for session.toml;
the handle owns writing that file and its permissions. No paths are opened or resolved here.

build_control / build_event and their validators check known messages strictly. decode_control_line and
decode_event_line read one JSON object, log unknown message names and return None for those names. Pipe I/O,
buffering, non-blocking writes and shutdown belong to the callers. encode_line returns text ending in one newline.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
from typing import Protocol, runtime_checkable


ENV_EVENTS_FD = "SOPA_COMPANION_EVENTS_FD"
PROTOCOL_VERSION = 1
SESSION_STATES = frozenset({"starting", "preparing", "ready", "closing", "closed", "failed"})
SAMPLE_STATES = frozenset({"preparing", "opening", "ready", "failed"})
HEALTH_STATES = frozenset({"running", "ending"})
CLOSE_POLICIES = frozenset({"add", "discard"})
DEFAULT_CLOSE_POLICY = "discard"       # No answer from the page or window, and CompanionApp.close()'s default.
MAX_EVENT_BYTES = 1024 * 1024          # The longest event line the handle reads; the engine never sends a longer one.
VIEW_MODES = frozenset({"in_place", "overlay", "copy"})
ZARR_SAVE_STATES = frozenset({"saving", "queued", "saved", "failed"})

_LOG = logging.getLogger(__name__)


def _text(value):
    return isinstance(value, str)


def _name(value):
    return isinstance(value, str) and bool(value)


def _bool(value):
    return type(value) is bool


def _uint(value):
    return type(value) is int and value >= 0


def _positive(value):
    return type(value) is int and value > 0


def _strings(value):
    return isinstance(value, list) and all(_name(item) for item in value)


def _nullable(check):
    return lambda value: value is None or check(value)


def _choice(values):
    return lambda value: isinstance(value, str) and value in values


def _interpreter(value):
    """An optional environment's Python in session.toml: "off", or a full path (launch() resolved "auto" already, so the
    session's engine never looks again). A string check only: nothing is opened here."""
    return isinstance(value, str) and (value == "off" or os.path.isabs(value))


def _session_id(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{8}", value) is not None


def _json_value(value, where):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    if isinstance(value, list):
        return [_json_value(item, f"{where}[{i}]") for i, item in enumerate(value)]
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError(f"{where} must have text keys")
        return {key: _json_value(item, f"{where}.{key}") for key, item in value.items()}
    raise ValueError(f"{where} must be a JSON value with finite numbers")


def _object(value, where):
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be an object")
    try:
        return _json_value(value, where)
    except RecursionError as exc:
        raise ValueError(f"{where} is nested too deeply or contains a cycle") from exc


def _fields(value, required, optional=None, *, where):
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be an object")
    for key, check in required.items():
        if key not in value:
            raise ValueError(f"{where}.{key} is required")
        if not check(value[key]):
            raise ValueError(f"Invalid {where}.{key}")
    for key, check in (optional or {}).items():
        if key in value and not check(value[key]):
            raise ValueError(f"Invalid {where}.{key}")


_CONFIG = {
    "server": {"port": lambda v: type(v) is int and v == 0,
               "token": lambda v: _text(v) and re.fullmatch(r"[A-Za-z0-9_-]{43}", v) is not None},
    "paths": {"data_dir": _name, "state_dir": _name,
              "sample_roots": lambda v: isinstance(v, list) and not v},
    "desktop": {"exit_when_closed": _bool, "profile_dir": _choice({"auto"})},
    "zarr": {"python": _name, "browse_start": _choice({""})},
    "qc": {"python": _interpreter, "enabled_by_default": _bool, "gpu": _uint, "models_dir": _text},
    "stain": {"python": _interpreter, "gpu": _uint, "instanseg_python": _interpreter, "instanseg_enabled": _bool},
    "analysis": {"python": _interpreter, "gpu": _uint},
    "session": {"mode": _choice({"api"}), "id": _session_id, "tool": _choice({"", "proseg", "stain"}),
                "open": _choice({"window", "link", "none"}), "patches_dir": _text,
                "link_code_ttl_s": lambda v: type(v) is int and v == 600, "kernel_pid": _positive},
}
# Tables written only when the handle has a value for them; the engine's own settings stand otherwise.
_OPTIONAL_CONFIG = {
    "proseg": {"binary": _name},
}
# [locate.<feature>]: what an "auto" found missing in the kernel (locate.py Found.report: no path, how "auto", the
# modules), for each feature launch() then wrote "off" for, so the session's engine can say what the kernel lacks
# without looking again. Only these features' interpreters ([<feature>] python) are resolved there and reported.
LOCATED = ("qc", "stain", "analysis")
_LACKING = {"path": lambda v: v == "", "how": _choice({"auto"}), "missing": lambda v: _strings(v) and bool(v)}


def build_session_config(*, id: str, token: str, data_dir: str, state_dir: str, python: str, kernel_pid: int,
                         tool: str | None = None, open: str = "none", qc_python: str = "off",
                         mixing_check: bool = False, qc_gpu: int = 0, model_cache: str | None = None,
                         stain_python: str = "off", stain_gpu: int = 0, instanseg_python: str = "off",
                         instanseg_enabled: bool = False, analysis_python: str = "off", analysis_gpu: int = 0,
                         patches_dir: str | None = None, proseg_binary: str | None = None,
                         locate: dict | None = None) -> dict:
    """session.toml's tables. Paths are strings; token is secrets.token_urlsafe(32), supplied by the handle.

    The caller resolves open="auto" and each optional environment ("auto") before building: an optional Python is
    "off" or a full path. python and kernel_pid are the kernel's sys.executable and os.getpid(). proseg_binary, the
    proseg-live program the handle checked, becomes [proseg] binary; None leaves the engine's own. instanseg_enabled is
    the internal switch of the dormant InstanSeg ([stain] instanseg_enabled), which the session's engine then has too.
    locate: {feature: its report} for the features ("qc", "stain", "analysis") whose "auto" found their modules missing
    in the kernel (their python "off"), written as [locate.<feature>] (LOCATED).
    """
    extra = {} if proseg_binary is None else {"proseg": {"binary": proseg_binary}}
    if locate:
        extra["locate"] = {feature: dict(report) for feature, report in locate.items()}
    return validate_session_config({
        "server": {"port": 0, "token": token},
        "paths": {"data_dir": data_dir, "state_dir": state_dir, "sample_roots": []},
        "desktop": {"exit_when_closed": open == "window", "profile_dir": "auto"},
        "zarr": {"python": python, "browse_start": ""},
        "qc": {"python": qc_python, "enabled_by_default": mixing_check, "gpu": qc_gpu,
               "models_dir": "" if model_cache is None else model_cache},
        "stain": {"python": stain_python, "gpu": stain_gpu, "instanseg_python": instanseg_python,
                  "instanseg_enabled": instanseg_enabled},
        "analysis": {"python": analysis_python, "gpu": analysis_gpu},
        "session": {"mode": "api", "id": id, "tool": "" if tool is None else tool, "open": open,
                    "patches_dir": "" if patches_dir is None else patches_dir,
                    "link_code_ttl_s": 600, "kernel_pid": kernel_pid},
        **extra,
    })


def validate_session_config(value: object) -> dict:
    """Check session.toml's tables (as tomllib.loads gives them), without reading paths or logging the token."""
    value = _object(value, "config")
    for section, fields in _CONFIG.items():
        if section not in value:
            raise ValueError(f"config.{section} is required")
        _fields(value[section], fields, where=f"config.{section}")
    for section, fields in _OPTIONAL_CONFIG.items():
        if section in value:
            _fields(value[section], fields, where=f"config.{section}")
    if "locate" in value:
        if not isinstance(value["locate"], dict) or set(value["locate"]) - set(LOCATED):
            raise ValueError("config.locate may name only " + ", ".join(LOCATED))
        for feature, report in value["locate"].items():
            _fields(report, _LACKING, where=f"config.locate.{feature}")
            if value[feature]["python"] != "off":
                raise ValueError(f"config.locate.{feature} reports a python that is not off")
    return value


_CONTROL = {
    "hello": {"protocol": lambda v: type(v) is int and v == PROTOCOL_VERSION},
    "prepare": {"done": _uint, "total": _uint},
    "prepare_failed": {"message": _text},
    "bind": {"path": _name, "view": _choice(VIEW_MODES), "name": _name, "backed": _bool,
             "zarr": _nullable(_name), "memory_only": _strings, "points_key": _nullable(_name),
             "linked": _strings, "written": _strings},
    "object": {"names": _strings, "zarr": _nullable(_name), "auto_save": _bool},
    "notebook": {"busy": _bool},
    "received": {"id": _name},
    "receive_failed": {"id": _name, "message": _text},
    "applied": {"id": _name, "state": _choice({"added", "failed"}), "keys": _strings, "clash": _strings,
                "message": _nullable(_text)},
    "close": {"pending": _choice(CLOSE_POLICIES)},
}
CONTROL_OPS = frozenset(_CONTROL)

_EVENTS = {
    "ready": {"port": lambda v: type(v) is int and 1 <= v <= 65535, "pid": _positive},
    "session": {"state": _choice(SESSION_STATES), "message": _nullable(_text)},
    "deliver": {"id": _name, "run": _name, "kind": _choice({"proseg", "stain"}), "staged": _name,
                "shapes_key": _name, "table_key": _nullable(_name), "set_default": _bool, "overwrite": _bool,
                "replaced": _strings, "written_to": _nullable(_name), "cells": _uint, "genes": _nullable(_uint),
                "provenance": lambda v: isinstance(v, dict)},
    "closing": {"pending": _choice(CLOSE_POLICIES), "by": _choice({"page", "handle", "window", "kernel"})},
    "ended": {"added": _strings, "not_added": lambda v: isinstance(v, list)},
    # A link the handle showed (or app.url) signed a browser in: the handle replaces the link it showed.
    "link": {"state": _choice({"used"})},
}
EVENTS = frozenset(_EVENTS)


def _message(value, tag, schemas, allow_unknown):
    value = _object(value, "message")
    _fields(value, {tag: _name}, where="message")
    name = value[tag]
    if name not in schemas:
        if allow_unknown:
            _LOG.warning("Ignoring unknown API %s %r", tag, name)
            return None
        raise ValueError(f"Unknown message.{tag}")
    _fields(value, schemas[name], where=f"message.{name}")
    return value


def build_control(op: str, **fields) -> dict:
    """Build a control message (kernel to engine) using its exact wire field names."""
    return validate_control({"op": op, **fields})


def validate_control(value: object, *, allow_unknown: bool = False) -> dict | None:
    """Validate a control message; allow_unknown logs and ignores an unknown op, but refuses malformed known ops."""
    return _message(value, "op", _CONTROL, allow_unknown)


def build_event(ev: str, **fields) -> dict:
    """Build an event (engine to kernel) using its exact wire field names (the discriminator is ev)."""
    return validate_event({"ev": ev, **fields})


def validate_event(value: object, *, allow_unknown: bool = False) -> dict | None:
    """Validate an event. A session event carries the handle's status, not the health.session object."""
    value = _message(value, "ev", _EVENTS, allow_unknown)
    if value is not None and value["ev"] == "ended":
        for i, item in enumerate(value["not_added"]):
            _fields(item, {"run": _name, "reason": _choice({"discarded", "failed", "clash", "incomplete"}),
                           "message": _text}, where=f"message.ended.not_added[{i}]")
    return value


def encode_line(value: dict) -> str:
    """Encode a JSON object as one line of text, including its final newline; use a message builder first."""
    return json.dumps(_object(value, "message"), ensure_ascii=True, allow_nan=False, separators=(",", ":")) + "\n"


def _pairs(items):
    value = {}
    for key, item in items:
        if key in value:
            raise ValueError("Repeated JSON field")
        value[key] = item
    return value


def _constant(value):
    raise ValueError("JSON numbers must be finite")


def decode_line(line: str | bytes) -> dict:
    """Read exactly one JSON object; accept UTF-8 bytes and an optional LF or CRLF terminator."""
    if isinstance(line, bytes):
        try:
            line = line.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("The message must be UTF-8") from exc
    if not isinstance(line, str):
        raise ValueError("The message must be text or UTF-8 bytes")
    if line.endswith("\n"):
        line = line[:-1].removesuffix("\r")
    if "\n" in line or "\r" in line:
        raise ValueError("Expected one JSON object per line")
    try:
        value = json.loads(line, object_pairs_hook=_pairs, parse_constant=_constant)
    except (ValueError, RecursionError) as exc:
        raise ValueError("Invalid JSON message") from exc
    return _object(value, "message")


def decode_control_line(line: str | bytes) -> dict | None:
    """Read a control line; unknown ops are logged and ignored without ending the reader."""
    return validate_control(decode_line(line), allow_unknown=True)


def decode_event_line(line: str | bytes) -> dict | None:
    """Read an event line; unknown events are logged and ignored for forward compatibility."""
    return validate_event(decode_line(line), allow_unknown=True)


def build_health_session(mode: str, **fields) -> dict:
    """Build health.session: the desktop needs only mode; API mode supplies the remaining fields."""
    return validate_health_session({"mode": mode, **fields})


def validate_health_session(value: object) -> dict:
    """Check health.session (also the fields beside ev in the page's session push)."""
    value = _object(value, "session")
    _fields(value, {"mode": _choice({"desktop", "api"})}, where="session")
    if value["mode"] == "desktop":
        return value
    _fields(value, {"id": _session_id, "sample": lambda v: isinstance(v, dict),
                    "tool": _nullable(_choice({"proseg", "stain"})),
                    "writes": _choice({"zarr", "memory", "memory_autosave_off"}),
                    "notebook": lambda v: isinstance(v, dict), "patches": _bool, "link": _bool,
                    "state": _choice(HEALTH_STATES)}, where="session")
    # The engine starts unbound: these sample identifiers may be null before bind/inspect finishes.
    _fields(value["sample"], {"id": _nullable(_name), "name": _nullable(_text), "backed": _bool,
                             "zarr_name": _nullable(_name), "view": _nullable(_choice(VIEW_MODES)),
                             "state": _choice(SAMPLE_STATES), "progress": lambda v: isinstance(v, dict),
                             "message": _nullable(_text), "memory_only": _strings,
                             "tools": lambda v: isinstance(v, list) and all(t in ("proseg", "stain") for t in v)},
            where="session.sample")
    _fields(value["sample"]["progress"], {"done": _uint, "total": _uint}, where="session.sample.progress")
    _fields(value["notebook"], {"busy": _bool}, where="session.notebook")
    return value


_ZARR_SAVE = {
    "saving": {"stage": _choice({"building", "writing"}), "message": _nullable(_text)},
    "queued": {"written_to": _nullable(_name)},
    "saved": {"added": lambda v: v is True, "cells": _uint, "shapes_key": _name, "table_key": _nullable(_name),
              "written_to": _nullable(_name), "at": _name},
    "failed": {"message": _text},
}


def build_zarr_save(state: str, **fields) -> dict:
    """Build an API-mode zarr_save state using health.session's field names."""
    return validate_zarr_save({"state": state, **fields})


def validate_zarr_save(value: object) -> dict:
    """Check an API-mode save state; a failed save may include written_to and clash."""
    value = _object(value, "zarr_save")
    _fields(value, {"state": _choice(ZARR_SAVE_STATES)}, where="zarr_save")
    optional = {"written_to": _nullable(_name), "clash": _strings} if value["state"] == "failed" else None
    _fields(value, _ZARR_SAVE[value["state"]], optional, where="zarr_save")
    return value


def build_closing(answer: str, *, end: bool | None = None) -> dict:
    """Build the POST /closing body; None omits end, and the legacy answer save becomes add."""
    return validate_closing({"answer": answer, **({"end": end} if end is not None else {})})


def validate_closing(value: object) -> dict:
    """Read POST /closing's save alias without mutating the caller's body; a missing answer is not a close decision."""
    value = _object(value, "closing")
    _fields(value, {"answer": _choice(CLOSE_POLICIES | {"save"})}, {"end": _bool}, where="closing")
    if value["answer"] == "save":
        value["answer"] = "add"
    return value


@runtime_checkable
class Session(Protocol):
    """The session interface engine.session supplies; engine callers use None on the desktop."""

    mode: str
    bound: str | None
    view_path: str | None
    notebook_busy: bool

    def target(self) -> str | None: ...

    def names(self) -> set[str]: ...

    def deliver(self, event: dict) -> None: ...

    def received(self, id: str, timeout: float) -> bool: ...

    def applied(self, id: str) -> dict | None: ...

    def closing_policy(self) -> str | None: ...
