"""Command-line entry points: serve, launch, stop, status, install-desktop. The same as the `sopa-companion` command
(pip installs it) and as `python -m sopa_companion`; with no command, launch."""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import socket
import sys


def typed_command() -> bool:
    """Whether this process runs as the command pip installs (`sopa-companion ...`), not `python -m sopa_companion`."""
    return Path(sys.argv[0]).name in ("sopa-companion", "sopa-companion.exe")


def command_words() -> str:
    """How the hints write this copy's command: `sopa-companion` when that is what ran, or for an installed copy (its
    environment has the command; an engine its launch started runs as `python -m`), else `python -m sopa_companion` (a
    source folder)."""
    from . import dirs
    return "sopa-companion" if typed_command() or not dirs.source_tree() else "python -m sopa_companion"


def parser():
    # Named as the user typed it: the command pip installs, or the module.
    prog = "sopa-companion" if typed_command() else "python -m sopa_companion"
    top = argparse.ArgumentParser(prog=prog, description="SoPA companion: Stain segmentation "
                                  "(find cells in a sample's images) and Proseg Live (watch proseg segment cells, "
                                  "one iteration at a time)")
    top.add_argument("--config", help="Override configuration TOML")
    top.add_argument("--version", action="store_true", help="Print the app's version and the proseg-live it runs")
    commands = top.add_subparsers(dest="command")
    serve = commands.add_parser("serve", help="Serve the authenticated API and the app")
    serve.add_argument("--port", type=int, default=None)
    serve.add_argument("--host", choices=["127.0.0.1", "localhost", "::1"], default="127.0.0.1")
    # Also after `serve`: a notebook's session starts its engine as `serve --config <its session.toml>` (api/handle.py).
    serve.add_argument("--config", default=argparse.SUPPRESS, help="Override configuration TOML (a notebook "
                       "session's session.toml serves that session)")
    launch = commands.add_parser("launch", help="Start or reuse the engine and open the app")
    launch.add_argument("--no-browser", action="store_true")
    launch.add_argument("--print-url", action="store_true", help="Print a link that opens the app in one browser, once, "
                        "within 10 minutes (for an SSH tunnel), without opening a browser")
    launch.add_argument("--browser", choices=["auto", "chrome", "default"],
                        help="Where the app opens, this time only (default: [desktop] browser): default (your default "
                             "browser), chrome (an app window of its own), auto (= chrome)")
    launch.add_argument("--timeout", type=float, help="Startup timeout in seconds (default: desktop.start_timeout_s)")
    stop = commands.add_parser("stop", help="Stop the engine")
    stop.add_argument("--force", action="store_true",
                      help="Stop without asking: runs nobody saved or answered for are discarded, a save under way is "
                           "finished first; SIGKILL after the timeout")
    stop.add_argument("--timeout", type=float, default=None,
                      help="Seconds to wait for the engine to exit, saving included (default 900 with --force, else 30)")
    status = commands.add_parser("status", help="Report engine status without revealing credentials")
    status.add_argument("--json", action="store_true")
    desktop = commands.add_parser("install-desktop", help="Install or remove the per-user launcher and icon")
    desktop.add_argument("--uninstall", action="store_true")
    desktop.add_argument("--home", help="Install into this home directory (useful for isolated validation)")
    desktop.add_argument("--no-desktop-icon", action="store_true")
    desktop.add_argument("--dry-run", action="store_true")
    return top


DEFAULT_PORT = 8777                     # the app's own engine (config.toml [server] port)
FALLBACK_PORTS = range(8777, 8877)


def candidate_ports(port: int) -> list[int]:
    """The default port falls back to the next free one in 8777..8876, as it always has (the desktop launcher reads
    the port back from the runtime file). Any other port is exact: a test, harness or second engine must never end
    up on the default engine's port, nor take one it falls back to."""
    if port == DEFAULT_PORT:
        return [port, *[p for p in FALLBACK_PORTS if p != port]]
    return [port]


def listen(host: str, port: int) -> socket.socket:
    """A bound loopback socket on the first free candidate port."""
    for candidate in candidate_ports(port):
        sock = socket.socket(socket.AF_INET6 if host == "::1" else socket.AF_INET, socket.SOCK_STREAM)
        if os.name == "nt":
            # Windows' SO_REUSEADDR lets another process bind the same port and take the app's connections: the port is
            # this engine's alone instead (a closed engine's port is free again at once there anyway).
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        else:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, candidate))
            return sock
        except OSError:
            sock.close()
    if port == DEFAULT_PORT:
        raise RuntimeError("Cannot bind a loopback port in 8777–8876")
    raise RuntimeError(f"Port {port} on {host} is in use. Only the default port ({DEFAULT_PORT}) falls back to another "
                       "one: stop whatever holds it, or choose another --port (or [server] port in the --config file)")


def opening_lines(url: str, app, terminal: bool) -> str:
    """What `serve` prints as it starts. In a terminal, a link that signs one browser in, once, within 10 minutes (time
    for an SSH tunnel): never the bearer token, which a browser would keep in its history and which issues launch codes
    and stops the engine for as long as it runs. Into a log (the launcher's), the address alone."""
    if not terminal:
        return f"SoPA companion: {url}\nFor a private URL: {command_words()} launch --print-url"
    link = app.state.launch_codes.issue(link=True)
    return (f"SoPA companion: {url}?code={link['code']}\nThis link opens the app in one browser, within "
            f"{link['expires_in'] / 60:g} minutes. Another: {command_words()} launch --print-url")


def serve(config, host, port):
    from .names import retired_isolation
    if problem := retired_isolation():        # a test or harness engine meant to keep apart from the user's locks
        raise RuntimeError(problem)
    from .desktop import migrate
    moved = migrate.migrate(config)            # before anything makes the state folder (the default instance's first start)
    from .server.runtime_file import refuse_other
    refuse_other(config)                       # another copy's engine holds this state folder: exit 3, nothing touched
    from .desktop.logs import install_rotation
    install_rotation(config)
    if located := config.get("locate"):       # where each helper runs, and proseg-live (load_config found them)
        from . import locate
        print("SoPA companion: " + locate.summary(located), flush=True)
    if moved:
        print(moved, flush=True)
    from .server.runtime_file import acquire_engine_lock
    engine_lock = acquire_engine_lock(config)   # held (open) until the process exits
    import uvicorn
    from .server import create_app
    try:     # --port 0 is a port of its own (any free one), not "the configured port"
        listener = listen(host, config["server"]["port"] if port is None else port)
    except RuntimeError:
        os.close(engine_lock)
        raise
    port = listener.getsockname()[1]
    config["server"].update(host=host, port=port)
    try:
        app = create_app(config)
        from .server.runtime_file import install_lifecycle
        install_lifecycle(app, config, host, port)
        url = f"http://{'[::1]' if host == '::1' else host}:{port}/"
        print(opening_lines(url, app, sys.stdout.isatty()), flush=True)
        uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="info", access_log=False)).run(sockets=[listener])
    finally:
        listener.close()
        os.close(engine_lock)
    if (refused := getattr(app.state, "refused", None)) is not None:
        raise refused                          # another copy's engine took the state folder as this one started
    return 0


def api_session(path) -> bool:
    """Whether a --config file is a notebook session's session.toml ([session] mode = "api")."""
    from .engine.config import api_session_file
    return api_session_file(path)


# What to do when this Python lacks the engine's packages, by the copy's layout (engine_missing).
INSTALLED_FIX = "Install the app into this environment as the README's Install section says."
SOURCE_ENVIRONMENT = "This source folder has its own environment: env PYTHONPATH={entry} {root}/.venv/bin/python -m " \
                     "sopa_companion {command}"
SOURCE_FIX = 'Make this source folder\'s environment as docs/DEVELOPING.md\'s "From a source folder" section says.'


def engine_missing(command: str) -> str | None:
    """None when this Python has every module the engine imports (locate.ENGINE_MODULES, looked up without importing);
    else what is missing and how to fix it, in two lines. A source folder with its own .venv is given that
    environment's command with the PYTHONPATH that runs this very copy (the .venv may know another copy of the app
    through a .pth file)."""
    from . import dirs, locate
    missing = [name for name in locate.ENGINE_MODULES if importlib.util.find_spec(name) is None]
    if not missing:
        return None
    first = f"The app's engine needs {', '.join(missing)} in this Python ({os.path.abspath(sys.executable)})."
    root = dirs.ROOT
    if not dirs.source_tree(root):
        return first + "\n" + INSTALLED_FIX
    if (root / ".venv" / "bin" / "python").exists():
        from .engine.config import helper_path
        return first + "\n" + SOURCE_ENVIRONMENT.format(entry=helper_path(root), root=root, command=command)
    return first + "\n" + SOURCE_FIX


def version(config_path=None) -> int:
    """`--version`: the app's version, and the proseg-live it would run (the effective settings' [proseg] binary, else
    the automatic lookup alone); a program that is not there now says what makes it."""
    from . import __version__, locate
    try:
        from .engine.config import load_config
        found = load_config(config_path)["locate"]["proseg"]
    except Exception:                    # settings that do not load: the automatic lookup alone
        found = locate.proseg_live().report()
    print(f"sopa-companion {__version__}")
    if found.get("how") == "missing" or not Path(found.get("path") or "").is_file():
        print(f"proseg-live: missing ({locate.fix_words(found.get('fix') or '', found.get('path') or '')})")
    else:
        print(f"proseg-live: {found['path']}")
    return 0


def main(argv=None):
    args = parser().parse_args(argv)
    if args.version:
        return version(args.config)
    if args.command is None:                            # `sopa-companion` alone (or with --config): launch
        args = parser().parse_args([*(argv if argv is not None else sys.argv[1:]), "launch"])
    if args.command == "serve" and args.config and api_session(args.config):
        # A notebook's session: its engine runs in the kernel's own interpreter, on a port of its own, with its pipes
        # (server/api_serve.py).
        from .server import api_serve
        try:
            return api_serve.main(["--config", args.config])
        except Exception as exc:
            print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
    # No other interpreter is ever started instead: a Python without the engine's packages is told how to fix it.
    if problem := engine_missing(args.command):
        print(problem, file=sys.stderr)
        return 1
    from .engine.config import load_config
    config = load_config(args.config)
    try:
        if args.command == "serve":
            return serve(config, args.host, args.port)
        if args.command in ("launch", "stop", "status"):
            from .desktop.launcher import command
            return command(config, args)
        if args.command == "install-desktop":
            from .desktop.install import install
            return install(config, args)
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return getattr(exc, "exit_code", 1)       # the state folder's move refused: 3, as launch says it
    return 0
