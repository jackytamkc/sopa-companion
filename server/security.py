"""Local sign-in, DNS-rebinding and CSRF checks for HTTP and WS.

Two credentials. The bearer token is the engine's: the command line and tools read it from the private runtime file and
send it as `Authorization: Bearer`; only it issues launch codes and stops the engine (server/app.py /shutdown). The
page's key is another random value, made at each start and kept in memory only. A page gets it by trading a one-use
launch code once: the app's own page takes the code out of its address and posts it in a header to SIGN_IN, from the
app's own origin (frontend/src/page-key.mjs). A GET of the app's address never spends a code, so a link preview cannot
burn one. The page keeps the key in its tab and sends it in a header (PAGE_HEADER) with every request; its WebSocket
offers it in its handshake, as the `key.<key>` subprotocol beside SOCKET_PROTOCOL, and the engine answers with
SOCKET_PROTOCOL alone, never echoing the key. The page's key is never taken as a bearer, nor the bearer as the page's
key, and neither is read from an address or a cookie.

No cookie is ever set: a cookie goes to every port of its host, so another service on this computer that the same
browser visits (another user's included) would be sent it. Every Cookie header is dropped before the app sees a
request, so another local site's cookies never reach a log or a report either.

Every check is made here, before the app sees the request, and nothing waits for a client to say who it is: a socket
without its credential is refused in its handshake. The app's own files (a GET or HEAD outside /api/) are public: they
hold no secret, anyone on this computer can read them from disk, and the Host check still applies to them. Every
answer may be neither framed nor read by another site's page. The app's document, under whatever address it was asked
for, and any other page the engine serves (the development pages aside), also gets a strict Content-Security-Policy:
it limits where a script that should ever get into the page can send the key (its requests, images and forms reach
this engine alone), though not where it can take the whole tab (a navigation, a window it opens), which no policy
covers.
"""
from __future__ import annotations

import asyncio
import hmac
import json
import posixpath
import re
import secrets
import threading
import time
from urllib.parse import parse_qsl, urlsplit

PAGE_HEADER = "x-sopa-page-key"      # the page's key (frontend/src/page-key.mjs PAGE_HEADER)
CODE_HEADER = "x-sopa-launch-code"   # a launch code, traded once for the page's key
SIGN_IN = "/api/v1/session/sign-in"
LAUNCH = "/api/v1/session/launch-code"
SOCKET_PROTOCOL = "sopa-companion"   # the subprotocol a page's socket offers, and the engine answers with
KEY_PROTOCOL = "key."                # + the page's key: the socket's other offered entry (page-key.mjs KEY_PROTOCOL)
LAUNCH_CODE_S = 45               # a launch code's lifetime: the launcher opens its URL at once
LAUNCH_CODE_MAX_S = 60           # at most this, except an API session's link ([session] link_code_ttl_s = 600)
LINK_CODE_S = 600                # a link printed for a person to open (launch --print-url, serve in a terminal, an API
                                 # session's link): one use too, with time to set up an SSH tunnel
AUTH = "sopa_companion.auth"     # the scope key saying how a request signed in: "bearer", "page", or "public" (the
                                 # app's own files, which need no credential)
# How to open the app again, after a launch code that was used or expired, with the command as it is typed
# (cli.command_words; a source folder's by default). An API session says its own (from the notebook).
def reopen_words(command: str = "python -m sopa_companion") -> str:
    return f"Relaunch SoPA companion from the icon or run {command} launch."


REOPEN = reopen_words()
EXPIRED = "This launch code was used already or has expired."
SIGNED_OUT = "This page is not signed in."
# Answered for every request whose credential is missing or not this engine's.
REQUIRED = "A SoPA companion page key or bearer token is required"
IN_ADDRESS = ("A token never goes in an address: open the app with sopa-companion launch, or a link from "
              "sopa-companion launch --print-url.")
# Every answer: neither framed nor read by another site's page (a page of another local service included).
SHARED_HEADERS = [(b"x-content-type-options", b"nosniff"), (b"referrer-policy", b"no-referrer"),
                  (b"x-frame-options", b"DENY"), (b"cross-origin-resource-policy", b"same-origin")]
# The app's document (index.html): its own scripts, styles, images and workers only, and requests to this engine only,
# its socket's ws:// address named as well ('self' does not cover ws: in every browser). No inline script or style, no
# eval: the page has none (frontend/test/csp-ready.test.mjs).
POLICY = ("default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data: blob:; font-src 'self'; "
          "connect-src 'self' ws://{host}; worker-src 'self'; object-src 'none'; base-uri 'none'; form-action 'none'; "
          "frame-ancestors 'none'")
# The app's document, by the path its files resolve an address to (an answer that names no type, a 304, still is it);
# and what a browser shows as a page of its own, scripts and all, whatever its address.
DOCUMENTS = ("/", "/index.html")
PAGE_TYPES = (b"text/html", b"application/xhtml+xml", b"image/svg+xml")
DEVELOPMENT = "/dev"     # the development pages (never shipped: engine/shipping.py), which run inline scripts


class LaunchCodes:
    """Short-lived one-use credentials; never persisted or logged. `ttl`: a code's lifetime (at most `limit`);
    `link_ttl`: a link's, a code printed for a person to open (at most LINK_CODE_S). Each code is of a kind: "window"
    (opened at once) or "link" (printed for a person), which its sign-in reports."""
    def __init__(self, *, ttl: float = LAUNCH_CODE_S, limit: float = LAUNCH_CODE_MAX_S, link_ttl: float = LINK_CODE_S,
                 clock=time.monotonic):
        if not 0 < ttl <= limit:
            raise ValueError(f"Launch-code lifetime must be at most {limit:g} seconds")
        if not 0 < link_ttl <= LINK_CODE_S:
            raise ValueError(f"A link's lifetime must be at most {LINK_CODE_S:g} seconds")
        self.ttl, self.link_ttl, self.clock = ttl, link_ttl, clock
        self.codes = {}
        self.lock = threading.Lock()

    def issue(self, link: bool = False) -> dict:
        ttl = self.link_ttl if link else self.ttl
        with self.lock:
            now = self.clock()
            self.codes = {code: entry for code, entry in self.codes.items() if entry[0] > now}
            # Bound memory even if an authenticated client never redeems its codes.
            if len(self.codes) >= 1024:
                self.codes.pop(next(iter(self.codes)))
            code = secrets.token_urlsafe(32)
            self.codes[code] = (now + ttl, "link" if link else "window")
            return {"code": code, "expires_in": ttl}

    def consume(self, code: str) -> str | None:
        """The code's kind ("window" | "link"), using it up; None for a code used already, expired or never issued."""
        with self.lock:
            expiry, kind = self.codes.pop(code, (0, None))
            return kind if expiry > self.clock() else None


def _matches(value: str, expected: str) -> bool:
    return bool(value) and hmac.compare_digest(value.encode(), expected.encode())


def _resolved(path: str) -> str:
    """The path the app's files take an address's path for (Starlette's StaticFiles splits it at every / and
    normalises it, as a file path): //index.html, /index.html/ and /src/../index.html (a browser sends /src%2F..%2F
    as written) are all /index.html, and // is /."""
    return posixpath.normpath("/" + "/".join(part for part in path.split("/") if part))


def _shown_as_page(headers) -> bool:
    """Whether an answer's Content-Type is one a browser shows as a page of its own (PAGE_TYPES)."""
    return any(key.lower() == b"content-type" and value.split(b";", 1)[0].strip().lower() in PAGE_TYPES
               for key, value in headers)


def page_id(value) -> str | None:
    """A page's own id as its socket's address names it (?page=: frontend/src/log-export.mjs pageSession), when it is
    one: letters, digits, - and _, at most 100; else None. It is no secret, only a name."""
    return value if isinstance(value, str) and re.fullmatch(r"[\w-]{1,100}", value, re.ASCII) else None


async def _signed_out_socket(receive, send) -> None:
    """A page's socket whose key is missing or not this engine's: accepted (with the page's subprotocol, which a browser
    requires once it offered one), then closed at once with 4401. A handshake refused before accept reaches a page as
    1006, whatever the reason, and a page must tell "signed out" from a lost connection."""
    if (await receive()).get("type") != "websocket.connect":
        return
    await send({"type": "websocket.accept", "subprotocol": SOCKET_PROTOCOL})
    await send({"type": "websocket.close", "code": 4401, "reason": SIGNED_OUT})


class LocalSecurity:
    """`token`: the bearer (the runtime file's); `page_key`: the page's key (made here when not given), which a launch
    code is traded for; `reopen`: how to open the app again, said when a launch code was used or expired;
    `on_sign_in(kind)`: told of each sign-in, with the kind of the code traded ("window" | "link"), in a thread of its
    own once the page has its answer (it may wait: a notebook's session tells its kernel through a pipe); what it raises
    is dropped."""

    def __init__(self, app, *, token, port, hosts, launch_codes=None, page_key=None, reopen=REOPEN, on_sign_in=None):
        self.app, self.token, self.port, self.reopen, self.on_sign_in = app, token, port, reopen, on_sign_in
        self.page_key = page_key or secrets.token_urlsafe(32)
        if _matches(self.page_key, token):
            raise ValueError("The page's key must not be the bearer token")
        self.hosts = {h.strip("[]").lower() for h in hosts}
        self.launch_codes = launch_codes if launch_codes is not None else LaunchCodes()

    def local(self, authority, scheme="http"):
        try:
            parsed = urlsplit(scheme + "://" + authority)
            return parsed.hostname in self.hosts and (parsed.port or (443 if scheme == "https" else 80)) == self.port and not parsed.username and not parsed.password
        except ValueError:
            return False

    def local_host(self, host: str) -> bool:
        """A Host header naming this engine: an allowed loopback name and this port, and nothing else (no path, query or
        user), since the app's document puts it into its policy."""
        return bool(host) and not any(c in host for c in "/?#@\\ \t") and self.local(host)

    def socket_address(self, host: str) -> str:
        """The page's own ws:// address for its document's policy: the Host it was asked under (already checked as
        local), rebuilt from its parts so that nothing else of that header can reach the policy."""
        name = urlsplit("http://" + host).hostname
        return f"[{name}]:{self.port}" if ":" in name else f"{name}:{self.port}"

    async def __call__(self, scope, receive, send):
        started = time.perf_counter()
        if scope["type"] not in ("http", "websocket"):
            return await self.app(scope, receive, send)
        socket = scope["type"] == "websocket"
        method, path = scope.get("method"), scope.get("path", "")
        values: dict[str, list[str]] = {}
        for key, value in scope.get("headers", []):
            values.setdefault(key.decode("latin-1").lower(), []).append(value.decode("latin-1"))

        def one(name: str) -> str:
            """A header's value when it was sent once; "" when it was not, or more than once."""
            found = values.get(name, ())
            return found[0] if len(found) == 1 else ""

        async def secure_send(message, document=False, development=False):
            if message["type"] == "http.response.start":
                headers = message.setdefault("headers", [])
                took = (time.perf_counter() - started) * 1000
                headers.extend([*SHARED_HEADERS, (b"server-timing", ("app;dur=%.2f" % took).encode())])
                if not development and (document or _shown_as_page(headers)):
                    policy = POLICY.format(host=self.socket_address(host))
                    headers.extend([(b"content-security-policy", policy.encode()),
                                    (b"cross-origin-opener-policy", b"same-origin")])
            await send(message)

        async def answer(status: int, body: dict, content_type=b"application/json"):
            await secure_send({"type": "http.response.start", "status": status,
                               "headers": [(b"content-type", content_type), (b"cache-control", b"no-store")]})
            await send({"type": "http.response.body", "body": json.dumps(body).encode()})

        host = one("host")
        error, status = None, 403
        if not self.local_host(host):
            error = ("E_HOST", "Host is not an allowed loopback authority")
        origin = one("origin")
        needs_origin = socket or method not in ("GET", "HEAD", "OPTIONS")
        if not error and needs_origin:
            source = urlsplit(origin)
            if source.scheme not in ("http", "https") or not self.local(source.netloc, source.scheme) or source.path or source.query or source.fragment:
                error = ("E_ORIGIN", "A local same-port Origin is required")
        query = parse_qsl(scope.get("query_string", b"").decode("latin-1"), keep_blank_values=True)
        if not error and any(name.lower() == "token" for name, _ in query):
            # A browser keeps the addresses it opened (its history): the bearer never signs a browser in from one.
            error, status = ("E_AUTH", IN_ADDRESS), 401
        # Each credential where it belongs only: the bearer in Authorization, the page's key in its own header, or, on a
        # page's socket, in its handshake's subprotocols (a browser cannot set a socket's headers). A key in an address
        # or a cookie is never read.
        authorization = one("authorization")
        bearer = authorization[7:] if authorization.startswith("Bearer ") else ""
        offered = (scope.get("subprotocols") or []) if socket else []
        page_socket = socket and SOCKET_PROTOCOL in offered
        if socket:
            keys = [entry[len(KEY_PROTOCOL):] for entry in offered if entry.startswith(KEY_PROTOCOL)]
            page = keys[0] if page_socket and len(keys) == 1 else ""
        else:
            page = one(PAGE_HEADER)
        by_bearer = _matches(bearer, self.token)
        by_page = _matches(page, self.page_key)
        public = False
        if not error and not socket and method == "POST" and path == SIGN_IN:
            # A launch code traded for the page's key, once: from the app's own page (the Origin check above), the code
            # in a header, never an address. Nothing is logged either way.
            codes = values.get(CODE_HEADER, [])
            kind = self.launch_codes.consume(codes[0]) if len(codes) == 1 else None
            if kind is None:
                return await answer(401, {"type": "about:blank", "title": "E_AUTH", "code": "E_AUTH", "status": 401,
                                          "detail": f"{EXPIRED} {self.reopen}"}, b"application/problem+json")
            await answer(200, {"key": self.page_key})
            if self.on_sign_in is not None:
                try:
                    await asyncio.to_thread(self.on_sign_in, kind)      # never holding the other requests up
                except Exception:
                    pass
            return
        if not error and not socket and method == "POST" and path == LAUNCH:
            if not by_bearer:
                error, status = ("E_AUTH", "A bearer token is required to issue a launch code"), 401
            else:
                # ?link=1: a code for a link a person opens by hand (LaunchCodes.link_ttl), not a window opened now.
                return await answer(200, self.launch_codes.issue(link=("link", "1") in query))
        if not error and not socket and method in ("GET", "HEAD") and path != "/api" and not path.startswith("/api/"):
            public = True                       # the app's own files: index.html, src/, styles/
        elif not error and not (by_bearer or by_page):
            if page_socket:
                return await _signed_out_socket(receive, send)
            error, status = ("E_AUTH", REQUIRED), 401
        if error:
            if socket:
                code = 4401 if status == 401 else 4403
                return await send({"type": "websocket.close", "code": code, "reason": error[1]})
            return await answer(status, {"type": "about:blank", "title": error[0], "code": error[0], "status": status,
                                         "detail": error[1]}, b"application/problem+json")
        scope[AUTH] = "public" if public else "bearer" if by_bearer else "page"
        # No route reads a cookie: other local sites' cookies (a browser sends them to every port) go no further.
        scope["headers"] = [(key, value) for key, value in scope.get("headers", []) if key.lower() != b"cookie"]
        # The policy goes with the answer, not the address: the app's files resolve many addresses to the same page.
        where = _resolved(path)
        development = where == DEVELOPMENT or where.startswith(DEVELOPMENT + "/")
        document = not socket and where in DOCUMENTS
        await self.app(scope, receive, lambda message: secure_send(message, document, development))
