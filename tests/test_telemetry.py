"""The page's telemetry v2 (frontend/src/telemetry.mjs) as the
engine keeps it: accepted and stored as sent, and summarised whole in the one-line engine log."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

BASE = "http://127.0.0.1:8799"

# A v2 report as the page sends it every 10 s while a run is open: the period of a 25.6 s Prior-shapes rebuild on a
# whole sample (the report came late: stretched), named by its long animation frame (the stage's own frame callback,
# no longer a telemetry.mjs wrapper) and by the app's own 'plv:' measures.
PAGE_REPORT_V2 = {
    "v": 2, "period_s": 29.16, "stretched": True, "hidden": False,
    "run": {"mode": "tiled", "state": "paused", "cells": 41190, "live": False, "umap": True, "view": "2d",
            "level": "detail", "stage": [871, 794]},
    "rates": {"stage": 0.1, "umap": 0.03, "timeline": 0, "batches": 0.07, "snapshots": 0, "frames": 0.14},
    "raf_lag_ms": {"p95": 16.7, "max": 25614.3},
    "long_tasks": {"count": 1, "max_ms": 25610.0},
    "loaf": {"count": 1, "max_ms": 25614.3, "blocking_max_ms": 25564.3,
             "script": {"ms": 25598.7, "type": "user-callback", "invoker": "FrameRequestCallback",
                        "source": "/src/stage.js", "fn": "", "pos": 10871}},
    "measures": {"plv:prior-rebuild": {"n": 1, "max_ms": 25590.2, "total_ms": 25590.2, "last_ms": 25590.2,
                                       "last": {"polygons": 41190, "layer": "xenium"}},
                 "plv:vectors-decode": {"n": 1, "max_ms": 830.5, "total_ms": 830.5, "last_ms": 830.5,
                                        "last": {"bytes": 9876543, "polygons": 82380}}},
    "input": {"count": 2, "delay_p95_ms": 25360.2, "delay_max_ms": 25360.2, "duration_p95_ms": 25648.0,
              "duration_max_ms": 25648.0},
    "heap_mb": 156.4,
    "display": {"dpr": 1, "window": [1854, 1011], "screen": [2560, 1369],
                "renderer": "ANGLE (NVIDIA, Vulkan 1.4.312 (NVIDIA Example GPU), NVIDIA)"},
    "fetch": {"vectors": {"n": 1, "bytes": 9876543, "wire": 9876543}, "view": {"n": 3, "bytes": 5700000, "wire": 600000}},
}


@pytest.fixture
def client(config):
    from sopa_companion.server import create_app
    application = create_app(config, token="t" * 40)
    with TestClient(application, base_url=BASE) as c:
        yield application, c


def test_a_v2_report_is_kept_as_sent(client):
    application, c = client
    # A page signs in with the engine's page key (new at each start), never the bearer (server/security.py).
    page = {"Origin": BASE, "X-SoPA-Page-Key": application.state.page_key, "Content-Type": "application/json"}
    body = json.dumps(PAGE_REPORT_V2, separators=(",", ":"))
    assert len(body.encode()) < 2048, "a period's report stays small"
    assert c.post("/api/v1/telemetry", content=body, headers=page).status_code == 204
    line = json.loads(application.state.telemetry.path.read_text())
    assert {k: v for k, v in line.items() if k != "t"} == PAGE_REPORT_V2


def test_the_log_line_holds_a_whole_v2_report():
    from sopa_companion.server.app import TELEMETRY_LINE_CHARS, telemetry_summary
    line = telemetry_summary({"t": "2026-10-17T04:00:00.000+00:00", **PAGE_REPORT_V2})
    assert not line.endswith("…") and len(line) <= TELEMETRY_LINE_CHARS and "\n" not in line, len(line)
    for part in ("v=2", "stretched=true", "rates.frames=0.14", "raf_lag_ms.max=25614.3", 'loaf.script.source="/src/stage.js"',
                 'loaf.script.invoker="FrameRequestCallback"', "loaf.script.pos=10871",
                 "measures.plv:prior-rebuild.max_ms=25590.2", "measures.plv:prior-rebuild.last={2}",
                 "measures.plv:vectors-decode.n=1", "fetch.vectors.bytes=9876543", 'display.renderer="ANGLE'):
        assert part in line, part
