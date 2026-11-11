"""LocalSecurity on its own (server/security.py): which credential opens what. A cookie goes to every port of its host,
so the page's key was sent to every other service on 127.0.0.1 the same browser visited, which could replay it. No
cookie is set or read now: the page trades a one-use launch code once, by POST, for its key, and sends the key in a
header; its WebSocket offers it in its handshake. The app's own files are public; everything else needs the bearer or
the page's key, checked before the app sees the request."""
from __future__ import annotations

import re

import pytest
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from sopa_companion.engine.config import ROOT
from sopa_companion.server.security import (AUTH, CODE_HEADER, KEY_PROTOCOL, LAUNCH_CODE_MAX_S, LINK_CODE_S, PAGE_HEADER,
                                            SIGN_IN, SOCKET_PROTOCOL, LaunchCodes, LocalSecurity, page_id)

BASE = "http://127.0.0.1:8799"
BEARER, PAGE_KEY = "b" * 43, "p" * 43
BEARING = {"Origin": BASE, "Authorization": "Bearer " + BEARER}
PAGE = {"Origin": BASE, "X-SoPA-Page-Key": PAGE_KEY}
HOST = {"Host": "127.0.0.1:8799"}


def guarded(**options):
    """The middleware over a probe app: what reached it (scope[AUTH], its headers) and which sockets it ran."""
    app = FastAPI()
    app.state.sockets = []

    @app.get("/api/v1/whoami")
    def whoami(request: Request):
        return {"auth": request.scope.get(AUTH), "headers": sorted({k.decode() for k, _ in request.scope["headers"]})}

    @app.post("/api/v1/shutdown")
    def shutdown():
        return {"stopped": True}

    @app.api_route("/", methods=["GET", "HEAD"])
    def root():
        return HTMLResponse("<!doctype html><title>SoPA companion</title>")

    @app.get("/src/main.js")
    def module(request: Request):
        return PlainTextResponse(f"// {request.scope.get(AUTH)}", media_type="text/javascript")

    @app.get("/dev/qc-demo.html")
    def dev():
        return HTMLResponse("<!doctype html><title>demo</title>")

    @app.websocket("/api/v1/ws")
    async def live(socket: WebSocket):
        app.state.sockets.append(dict(socket.query_params))
        offered = socket.scope.get("subprotocols") or ()
        await socket.accept(subprotocol=SOCKET_PROTOCOL if SOCKET_PROTOCOL in offered else None)
        await socket.send_json({"auth": socket.scope.get(AUTH), "page": page_id(socket.query_params.get("page"))})
        await socket.close()

    app.add_middleware(LocalSecurity, token=BEARER, port=8799, hosts=["127.0.0.1", "localhost", "[::1]"],
                       **{"page_key": PAGE_KEY, **options})
    client = TestClient(app, base_url=BASE)
    client.probe = app
    return client


def code(client, link=False) -> str:
    return client.post("/api/v1/session/launch-code" + ("?link=1" if link else ""), headers=BEARING).json()["code"]


def sign_in(client, value, **headers):
    return client.post(SIGN_IN, headers={"Origin": BASE, CODE_HEADER: value, **headers})


def page_socket(key=PAGE_KEY, page="page-1"):
    """What a page's `new WebSocket(url + '?page=<id>', ['sopa-companion', 'key.' + key])` sends."""
    return {"url": f"/api/v1/ws?page={page}", "subprotocols": [SOCKET_PROTOCOL, KEY_PROTOCOL + key],
            "headers": {**HOST, "Origin": BASE}}


def test_each_credential_opens_only_its_own_door():
    client = guarded()
    assert client.get("/api/v1/whoami", headers=BEARING).json()["auth"] == "bearer"
    assert client.get("/api/v1/whoami", headers=PAGE).json()["auth"] == "page"
    # Each value only where it belongs: the page's key as a bearer, the bearer as the page's key, either in a cookie or
    # an address, an empty one, a doubled one, one of another name: refused.
    for headers in ({"Origin": BASE, PAGE_HEADER: BEARER}, {"Origin": BASE, "Authorization": "Bearer " + PAGE_KEY},
                    {"Origin": BASE, "Cookie": f"sopa_companion_session={PAGE_KEY}"},
                    {"Origin": BASE, "Cookie": f"sc_1f3a9c2e={PAGE_KEY}; x-sopa-page-key={PAGE_KEY}"},
                    {"Origin": BASE, PAGE_HEADER: ""}, {"Origin": BASE, "X-SoPA-Page": PAGE_KEY}, {"Origin": BASE},
                    {"Origin": BASE, "Authorization": PAGE_KEY}, {"Origin": BASE, "Authorization": "Bearer " + BEARER + "x"}):
        refused = client.get("/api/v1/whoami", headers=headers)
        assert refused.status_code == 401 and refused.json()["code"] == "E_AUTH", headers
        assert refused.json()["detail"] == "A SoPA companion page key or bearer token is required"
    for url in (f"/api/v1/whoami?key={PAGE_KEY}", f"/api/v1/whoami?{PAGE_HEADER}={PAGE_KEY}"):
        assert client.get(url, headers={"Origin": BASE}).status_code == 401, url
    doubled = [("origin", BASE), (PAGE_HEADER, PAGE_KEY), (PAGE_HEADER, PAGE_KEY)]
    assert client.get("/api/v1/whoami", headers=doubled).status_code == 401
    assert client.get("/api/v1/whoami", headers=[("origin", BASE), ("authorization", "Bearer " + BEARER)] * 2).status_code == 401
    # Only the bearer issues a launch code: the page's key never does (it could not make itself new ones).
    assert client.post("/api/v1/session/launch-code", headers=PAGE).status_code == 401
    assert client.post("/api/v1/session/launch-code", headers=BEARING).json()["expires_in"] == 45


def test_a_launch_code_is_traded_once_by_post_for_the_key_and_a_get_spends_nothing():
    """A window or a link opens the app's address with ?code=: that GET only serves the page (a link preview, a
    prefetch, a reload spend nothing), and the page posts the code in a header, from the app's own origin, once."""
    client = guarded()
    first = code(client)
    page = client.get(f"/?code={first}&run=r1&harness=1", follow_redirects=False)
    assert page.status_code == 200 and page.text == "<!doctype html><title>SoPA companion</title>"
    assert "set-cookie" not in page.headers and "location" not in page.headers
    assert client.get(f"/?code={first}").status_code == 200, "spent nothing"
    traded = sign_in(client, first)
    assert traded.status_code == 200 and traded.json() == {"key": PAGE_KEY}
    assert traded.headers["cache-control"] == "no-store" and traded.headers["referrer-policy"] == "no-referrer"
    assert traded.headers["content-type"] == "application/json" and "set-cookie" not in traded.headers
    assert BEARER not in traded.text
    again = sign_in(client, first)
    assert again.status_code == 401 and again.json()["code"] == "E_AUTH" and "key" not in again.json()
    assert again.json()["detail"] == ("This launch code was used already or has expired. Relaunch SoPA companion from the icon "
                                      "or run python -m sopa_companion launch.")
    # Refused: no code, two codes (even two of the same), a code in the address, a code of another engine; no Origin, a
    # foreign one, another port of this computer, a page opened from a file.
    second = code(client)
    for headers in ({"Origin": BASE}, {"Origin": BASE, CODE_HEADER: ""}, {"Origin": BASE, CODE_HEADER: "x" * 43}):
        assert client.post(SIGN_IN, headers=headers).status_code == 401, headers
    assert client.post(SIGN_IN, headers=[("origin", BASE), (CODE_HEADER, second), (CODE_HEADER, second)]).status_code == 401
    # Only a POST trades: a GET or HEAD needs no Origin (any page can make one), so with the code in its header it is
    # refused like any request without a credential, and spends nothing.
    for method in ("GET", "HEAD"):
        assert client.request(method, SIGN_IN, headers={CODE_HEADER: second}).status_code == 401, method
        assert client.request(method, SIGN_IN, headers={"Origin": BASE, CODE_HEADER: second}).status_code == 401, method
    assert client.post(f"{SIGN_IN}?code={second}", headers={"Origin": BASE}).status_code == 401
    assert client.post(SIGN_IN, headers={CODE_HEADER: second}).status_code == 403
    for origin in ("http://evil.example", "http://127.0.0.1:8888", "null", "file://"):
        refused = sign_in(client, second, Origin=origin)
        assert refused.status_code == 403 and refused.json()["code"] == "E_ORIGIN", origin
    assert sign_in(client, second).json() == {"key": PAGE_KEY}, "none of those spent it"
    # The bearer is never traded for the page's key, nor the key for itself.
    assert sign_in(client, BEARER).status_code == 401 and sign_in(client, PAGE_KEY).status_code == 401


def test_launch_codes_live_at_most_a_minute_unless_told_otherwise_and_are_used_once():
    now = [0.0]
    codes = LaunchCodes(clock=lambda: now[0])
    assert codes.ttl == 45 and LAUNCH_CODE_MAX_S == 60
    with pytest.raises(ValueError, match="at most 60 seconds"):
        LaunchCodes(ttl=600)
    with pytest.raises(ValueError, match="at most 600 seconds"):
        LaunchCodes(ttl=601, limit=600)
    with pytest.raises(ValueError, match="A link's lifetime must be at most 600 seconds"):
        LaunchCodes(link_ttl=601)
    first, second = codes.issue()["code"], codes.issue()["code"]
    link = codes.issue(link=True)
    assert link["expires_in"] == LINK_CODE_S == 600
    assert codes.consume(first) == "window" and codes.consume(first) is None
    now[0] = 46
    assert codes.consume(second) is None
    now[0] = 599
    assert codes.consume(link["code"]) == "link" and codes.consume(link["code"]) is None    # a link: one use, 10 minutes
    assert codes.consume("never-issued") is None


def test_a_link_lives_ten_minutes_only_the_bearer_asks_for_one_and_each_sign_in_says_its_kind():
    """A link printed for a person to open by hand (launch --print-url, serve in a terminal, a notebook's link) is a
    launch code asked for with ?link=1: one use, 10 minutes, time to set up an SSH tunnel. Each sign-in is told with
    its code's kind (a notebook's session replaces its printed link once it was used)."""
    told = []
    client = guarded(on_sign_in=told.append)
    assert client.post("/api/v1/session/launch-code?link=1", headers=BEARING).json()["expires_in"] == 600
    assert client.post("/api/v1/session/launch-code?link=0", headers=BEARING).json()["expires_in"] == 45
    assert client.post("/api/v1/session/launch-code?link=1", headers=PAGE).status_code == 401
    link, window = code(client, link=True), code(client)
    assert sign_in(client, link).status_code == 200 and sign_in(client, window).status_code == 200
    assert sign_in(client, link).status_code == 401
    assert told == ["link", "window"], "told once each, only of a sign-in"

    def broken(kind):
        raise RuntimeError("the notebook went away")
    failing = guarded(on_sign_in=broken)
    assert sign_in(failing, code(failing)).json() == {"key": PAGE_KEY}, "a hook that fails never fails the sign-in"


def test_a_sessions_links_live_ten_minutes():
    client = guarded(launch_codes=LaunchCodes(ttl=600, limit=600), reopen="Open SoPA companion again from your notebook.")
    assert client.post("/api/v1/session/launch-code", headers=BEARING).json()["expires_in"] == 600
    assert sign_in(client, code(client)).json() == {"key": PAGE_KEY}
    assert sign_in(client, "used").json()["detail"].endswith("Open SoPA companion again from your notebook.")


def test_the_page_key_is_never_the_bearer_and_is_made_when_not_given():
    with pytest.raises(ValueError, match="must not be the bearer"):
        LocalSecurity(None, token=BEARER, port=8799, hosts=["127.0.0.1"], page_key=BEARER)
    made = LocalSecurity(None, token=BEARER, port=8799, hosts=["127.0.0.1"])
    assert len(made.page_key) >= 40 and made.page_key != BEARER
    assert made.page_key != LocalSecurity(None, token=BEARER, port=8799, hosts=["127.0.0.1"]).page_key


def test_the_apps_files_are_public_everything_else_is_not_and_the_host_is_checked_everywhere():
    """index.html and its modules need no credential (nothing in them is private; the page signs in once it runs).
    The Host check (DNS rebinding) holds for them too, a bearer in an address is refused anywhere, and the API answers
    a missing key in JSON: the page says it in its own words."""
    client = guarded()
    assert client.get("/").status_code == 200 and client.head("/").status_code == 200
    module = client.get("/src/main.js", headers={"Cookie": "other=1"})
    assert module.status_code == 200 and module.text == "// public"
    assert client.get("/src/main.js", headers=BEARING).text == "// public", "a file is public whoever asks"
    assert client.post("/", headers={"Origin": BASE}).status_code == 401
    for path in ("/api", "/api/v1/whoami", "/api/v1/unknown"):
        refused = client.get(path, headers={"Accept": "text/html"})
        assert refused.status_code == 401 and refused.headers["content-type"] == "application/problem+json", path
    for host in ("evil.example:8799", "127.0.0.1:8800", "127.0.0.1", "127.0.0.1:8799/x", "127.0.0.1:8799?a",
                 "u@127.0.0.1:8799", "127.0.0.1:8799 x"):
        assert client.get("/", headers={"Host": host}).status_code == 403, host
    assert client.get("/", headers=[("host", "127.0.0.1:8799"), ("host", "evil.example:8799")]).status_code == 403
    for url in (f"/?token={BEARER}", f"/?token={BEARER}&harness=1", f"/src/main.js?Token={PAGE_KEY}", "/?token=",
                "/api/v1/whoami?token=x"):
        refused = client.get(url, headers=BEARING, follow_redirects=False)
        assert refused.status_code == 401 and "set-cookie" not in refused.headers, url
        assert refused.json()["detail"].startswith("A token never goes in an address"), url
    assert client.get("/", headers={"Host": "localhost:8799"}).status_code == 200
    assert client.get("/", headers={"Host": "[::1]:8799"}).status_code == 200


def test_every_answer_is_neither_framed_nor_read_by_another_site_and_the_document_has_its_strict_policy():
    client = guarded()
    answers = {"document": client.get("/"), "index": client.get("/?code=x&run=r1"), "module": client.get("/src/main.js"),
               "api": client.get("/api/v1/whoami", headers=PAGE), "refused": client.get("/api/v1/whoami"),
               "host": client.get("/", headers={"Host": "evil.example:8799"}), "sign-in": sign_in(client, code(client)),
               "spent": sign_in(client, "spent"), "launch": client.post("/api/v1/session/launch-code", headers=BEARING),
               "dev": client.get("/dev/qc-demo.html")}
    for name, answer in answers.items():
        assert answer.headers["x-frame-options"] == "DENY", name
        assert answer.headers["cross-origin-resource-policy"] == "same-origin", name
        assert answer.headers["x-content-type-options"] == "nosniff", name
        assert answer.headers["referrer-policy"] == "no-referrer", name
        assert "set-cookie" not in answer.headers, name
    policy = answers["document"].headers["content-security-policy"]
    assert policy == ("default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data: blob:; font-src 'self'; "
                      "connect-src 'self' ws://127.0.0.1:8799; worker-src 'self'; object-src 'none'; base-uri 'none'; "
                      "form-action 'none'; frame-ancestors 'none'")
    assert "unsafe" not in policy and "nonce" not in policy
    assert answers["document"].headers["cross-origin-opener-policy"] == "same-origin"
    assert answers["index"].headers["content-security-policy"] == policy, "the address's query changes nothing"
    # The socket's address is the one the page was asked under, rebuilt from its parts.
    for host, address in (("localhost:8799", "ws://localhost:8799"), ("[::1]:8799", "ws://[::1]:8799"),
                          ("LOCALHOST:8799", "ws://localhost:8799")):
        assert f"connect-src 'self' {address};" in client.get("/", headers={"Host": host}).headers["content-security-policy"], host
    # Only the document: not its modules, its API, nor the development pages (never shipped).
    for name in ("module", "api", "refused", "dev", "sign-in"):
        assert "content-security-policy" not in answers[name].headers, name
        assert "cross-origin-opener-policy" not in answers[name].headers, name


def test_the_page_has_its_policy_whatever_address_it_is_asked_under():
    """The app's files resolve an address as a path on disk: //index.html, //, /index.html/ and /src%2F..%2Findex.html
    all serve index.html (a browser sends those as written). The page has its policy at each of them, as does any other
    page the engine serves; only the development pages go without (never shipped), and an address that merely passes
    through /dev/ to the app's page is the app's page."""
    from fastapi import FastAPI
    from sopa_companion.server.app import FreshStaticFiles
    app = FastAPI()

    @app.get("/api/v1/page.html")
    def page():
        return HTMLResponse("<!doctype html><title>a page</title>")

    app.mount("/", FreshStaticFiles(directory=ROOT / "frontend", html=True), name="frontend")
    app.add_middleware(LocalSecurity, token=BEARER, port=8799, hosts=["127.0.0.1"], page_key=PAGE_KEY)
    client = TestClient(app, base_url=BASE)
    index = (ROOT / "frontend/index.html").read_text()
    for path in ("/", "/index.html", "//index.html", "//", "/index.html/", "/src%2F..%2Findex.html", "/./index.html",
                 "/dev%2F..%2Findex.html", "/dev/..%2Findex.html", "/styles%2F..%2F"):
        answer = client.get(BASE + path)
        assert answer.status_code == 200 and answer.text == index, path
        assert answer.headers["content-security-policy"].startswith("default-src 'self'; script-src 'self';"), path
        assert answer.headers["cross-origin-opener-policy"] == "same-origin", path
        head = client.head(BASE + path)
        assert "content-security-policy" in head.headers and "cross-origin-opener-policy" in head.headers, path
    assert "content-security-policy" in client.get("/api/v1/page.html", headers=PAGE).headers, "any page the engine serves"
    for path in ("/src/main.js", "/styles/app.css"):
        assert "content-security-policy" not in client.get(BASE + path).headers, path
    if (ROOT / "frontend/dev/qc-demo.html").exists():           # development pages: in the development copy only
        demo = client.get(BASE + "/dev/qc-demo.html")
        assert demo.status_code == 200 and "content-security-policy" not in demo.headers


def test_the_page_s_files_have_their_own_types_whatever_the_computer_says(monkeypatch):
    """Python's mimetypes reads the computer's own table of file types (Windows' registry), where .js is often
    text/plain: a browser runs no module served as text, and the page showed its bare frame for ever (0.7.0 on Windows; fixed in 0.8.0).
    The app's files are served with the type their ending names, a 304 still naming none."""
    import mimetypes
    from fastapi import FastAPI
    from sopa_companion.server.app import PAGE_FILE_TYPES, FreshStaticFiles
    mimetypes.init()
    for table in mimetypes._db.types_map:          # both tables: restored when the test ends
        for ending in (".js", ".mjs", ".css", ".html", ".json"):
            monkeypatch.setitem(table, ending, "text/plain")
    assert mimetypes.guess_type("main.js")[0] == "text/plain", "the computer says a script is text"
    app = FastAPI()
    app.mount("/", FreshStaticFiles(directory=ROOT / "frontend", html=True), name="frontend")
    client = TestClient(app, base_url=BASE)
    for path, kind in (("/src/main.js", "text/javascript; charset=utf-8"), ("/src/page-key.mjs", "text/javascript; charset=utf-8"),
                       ("/src/prior-raster-worker.mjs", "text/javascript; charset=utf-8"),
                       ("/styles/app.css", "text/css; charset=utf-8"), ("/", "text/html; charset=utf-8"),
                       ("/index.html", "text/html; charset=utf-8")):
        answer = client.get(BASE + path)
        assert answer.status_code == 200 and answer.headers["content-type"] == kind, path
        assert answer.headers["cache-control"] == "no-cache", path
        again = client.get(BASE + path, headers={"If-None-Match": answer.headers["etag"]})
        assert again.status_code == 304 and "content-type" not in again.headers, path
    # Every kind of file the page is made of has its type here: a new kind is added to the table with the file.
    shipped = {file.suffix.lower() for folder in ("src", "styles") for file in (ROOT / "frontend" / folder).rglob("*") if file.is_file()}
    assert shipped | {".html"} <= set(PAGE_FILE_TYPES), sorted(shipped - set(PAGE_FILE_TYPES))


def test_no_cookie_ever_reaches_the_app_or_is_set():
    """No route reads a cookie, and another local site's cookies (a browser sends them to every port of 127.0.0.1)
    must never reach a log or a crash report: every Cookie header is dropped before the app."""
    client = guarded()
    seen = client.get("/api/v1/whoami", headers={**PAGE, "Cookie": "_xsrf=abc; notebook-session=xyz", "X-Other": "1"}).json()
    assert "cookie" not in seen["headers"] and "x-other" in seen["headers"] and PAGE_HEADER in seen["headers"]
    socket = page_socket()
    with client.websocket_connect(socket["url"], subprotocols=socket["subprotocols"],
                                  headers={**socket["headers"], "Cookie": "a=1"}) as live:
        assert live.receive_json()["auth"] == "page"
    # A sweep: nothing anywhere sets a cookie.
    for answer in (client.get("/"), client.get("/?code=" + code(client)), sign_in(client, code(client)), sign_in(client, "x"),
                   client.get("/api/v1/whoami", headers=PAGE), client.post("/api/v1/shutdown", headers=BEARING),
                   client.get("/api/v1/whoami"), client.post("/api/v1/session/launch-code", headers=BEARING)):
        assert "set-cookie" not in answer.headers


def test_the_websocket_takes_the_bearer_or_the_pages_key_in_its_handshake_and_says_which():
    client = guarded()
    with client.websocket_connect("/api/v1/ws", headers={**HOST, **BEARING}) as live:
        assert live.receive_json() == {"auth": "bearer", "page": None}
        assert live.accepted_subprotocol is None, "a tool offered none"
    with client.websocket_connect("/api/v1/ws", subprotocols=[SOCKET_PROTOCOL], headers={**HOST, **BEARING}) as live:
        assert live.accepted_subprotocol == SOCKET_PROTOCOL and live.receive_json()["auth"] == "bearer"
    socket = page_socket()
    with client.websocket_connect(socket["url"], subprotocols=socket["subprotocols"], headers=socket["headers"]) as live:
        assert live.accepted_subprotocol == SOCKET_PROTOCOL, "the page's own, never the key"
        assert live.receive_json() == {"auth": "page", "page": "page-1"}
    assert len(client.probe.state.sockets) == 3


@pytest.mark.parametrize("subprotocols", [
    [SOCKET_PROTOCOL, KEY_PROTOCOL + "x" * 43],                              # a key not this engine's (it started again)
    [SOCKET_PROTOCOL, KEY_PROTOCOL + BEARER],                                # the bearer where the key goes
    [SOCKET_PROTOCOL],                                                       # no key
    [SOCKET_PROTOCOL, KEY_PROTOCOL],                                         # an empty one
    [SOCKET_PROTOCOL, KEY_PROTOCOL + PAGE_KEY, KEY_PROTOCOL + PAGE_KEY],     # two
])
def test_a_pages_socket_without_this_engines_key_is_accepted_and_closed_at_once_so_the_page_knows_it_is_signed_out(subprotocols):
    """A handshake refused before accept reaches the page as 1006 whatever the reason, so it would retry for ever: a
    page's socket (it offers the page's subprotocol) is accepted, then closed with 4401 at once. The app never sees it,
    and nothing is left open or waiting."""
    client = guarded()
    with client.websocket_connect("/api/v1/ws?page=page-1", subprotocols=subprotocols, headers={**HOST, "Origin": BASE}) as live:
        assert live.accepted_subprotocol == SOCKET_PROTOCOL
        with pytest.raises(WebSocketDisconnect) as closed:
            live.receive_json()
    assert closed.value.code == 4401 and closed.value.reason == "This page is not signed in."
    assert client.probe.state.sockets == [], "the app never ran for it"


def test_any_other_socket_without_its_credential_is_refused_in_its_handshake():
    client = guarded()
    right = page_socket()
    for url, subprotocols, headers, refusal in (
            ("/api/v1/ws", [KEY_PROTOCOL + PAGE_KEY], {**HOST, "Origin": BASE}, 4401),        # the key, but no page's socket
            ("/api/v1/ws", None, {**HOST, "Origin": BASE}, 4401),                            # nothing at all
            (f"/api/v1/ws?key={PAGE_KEY}", None, {**HOST, "Origin": BASE}, 4401),            # the key in its address
            ("/api/v1/ws", None, {**HOST, "Origin": BASE, "Cookie": f"sopa_companion_session={PAGE_KEY}"}, 4401),
            ("/api/v1/ws", None, {**HOST, "Origin": BASE, PAGE_HEADER: PAGE_KEY}, 4401),     # a header no browser can set
            (right["url"], right["subprotocols"], {**HOST, "Origin": "http://evil.example"}, 4403),   # another site's page
            (right["url"], right["subprotocols"], {**HOST, "Origin": "http://127.0.0.1:8888"}, 4403),
            (right["url"], right["subprotocols"], {**HOST}, 4403),                           # no Origin
            (right["url"], right["subprotocols"], {"Host": "evil.example:8799", "Origin": BASE}, 4403),
            (f"{right['url']}&token={BEARER}", right["subprotocols"], {**HOST, "Origin": BASE}, 4401)):
        with pytest.raises(WebSocketDisconnect) as refused:
            with client.websocket_connect(url, subprotocols=subprotocols, headers=headers) as live:
                live.receive_json()
        assert refused.value.code == refusal, (url, subprotocols, headers)
    assert client.probe.state.sockets == [], "nothing reached the app, nothing left waiting"


def test_a_pages_id_is_a_short_plain_name_or_nothing():
    assert page_id("0b5e2c1a-7f3d-4c2e-9a41-3b2d5e6f7a8b") == "0b5e2c1a-7f3d-4c2e-9a41-3b2d5e6f7a8b"
    assert page_id("lxk2q-8f3k2j1a") == "lxk2q-8f3k2j1a"
    for bad in (None, "", "a" * 101, "a b", "a/b", "é", "a\n", 3, ["a"]):
        assert page_id(bad) is None, bad


def test_the_page_and_the_engine_name_the_same_headers_paths_and_protocols():
    """page-key.mjs and server/security.py agree: a header name, a path or a subprotocol said differently on one side
    would sign nobody in."""
    source = (ROOT / "frontend/src/page-key.mjs").read_text()
    said = dict(re.findall(r"\b([A-Z_]+) = '([^']*)'", source))
    assert said["PAGE_HEADER"].lower() == PAGE_HEADER and said["CODE_HEADER"].lower() == CODE_HEADER
    assert said["SIGN_IN"] == SIGN_IN and said["SOCKET_PROTOCOL"] == SOCKET_PROTOCOL and said["KEY_PROTOCOL"] == KEY_PROTOCOL
