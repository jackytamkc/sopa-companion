#!/usr/bin/env bash
# After the release's install (the README's lines, with a squatter's index beside PyPI: release.yml), the app and
# proseg-live as a tester gets them, run in a home folder of their own and without a display:
#   ci/install_smoke.sh BIN_DIR RELEASE_DIR
# BIN_DIR is the environment's bin folder (its python, sopa-companion and proseg-live), RELEASE_DIR the release folder
# the install took the two packages from.
# - the versions installed are the release folder's wheels', never a squatter's (ci/squat_index.py);
# - sopa-companion --version names BIN_DIR's proseg-live, which says the proseg it was built from;
# - import sopa_companion loads nothing of the engine (fastapi, uvicorn, httpx, psutil);
# - launch --no-browser starts the engine on the port of the user's own settings file (which an installed copy reads):
#   /health finds proseg-live and runs samples and Cellpose in this Python; status says healthy, with the installed
#   copy's data folder; stop --force stops it;
# - the console script's own serve is the app's engine: status finds it healthy, and stop --force stops it.
set -euo pipefail
[[ $# -eq 2 ]] || { echo "usage: ci/install_smoke.sh BIN_DIR RELEASE_DIR" >&2; exit 2; }
BIN="$(cd "$1" && pwd)"
RELEASE="$(cd "$2" && pwd)"
PY="$BIN/python"
APP="$BIN/sopa-companion"
fail() { echo "install_smoke.sh: $*" >&2; exit 1; }

# A home folder of its own: the user's settings, the state folder (its runtime file and logs), the data folder and the
# save locks all go there. No display, no GPU, none of the caller's XDG folders.
HOME="$(mktemp -d "${RUNNER_TEMP:-${TMPDIR:-/tmp}}/install-smoke-XXXXXX")"
export HOME CUDA_VISIBLE_DEVICES=""
unset DISPLAY WAYLAND_DISPLAY XDG_CONFIG_HOME XDG_DATA_HOME XDG_STATE_HOME XDG_CACHE_HOME XDG_RUNTIME_DIR PYTHONPATH
cd "$HOME"
SERVE=""
# Whatever happens, no engine of this check outlives it.
cleanup() {
  [[ -z "$SERVE" ]] || kill "$SERVE" 2>/dev/null || true
  "$APP" stop --force --timeout 30 >/dev/null 2>&1 || true
}
trap cleanup EXIT

# The one version of a package's wheels in the release folder ($1: the wheel's name before its version).
release_version() {
  local wheels=("$RELEASE/$1"-*.whl) versions
  [[ -e "${wheels[0]}" ]] || fail "the release folder holds no $1 wheel"
  versions="$(for wheel in "${wheels[@]}"; do wheel="${wheel##*/}"; wheel="${wheel#"$1"-}"; echo "${wheel%%-*}"; done | sort -u)"
  [[ "$(wc -l <<< "$versions")" -eq 1 ]] || fail "the release folder holds $1 wheels of several versions: $versions"
  echo "$versions"
}
app_version="$(release_version sopa_companion)"
program_version="$(release_version proseg_live)"

said="$("$APP" --version)" || fail "sopa-companion --version failed"
[[ "$(sed -n 1p <<< "$said")" == "sopa-companion $app_version" ]] \
  || fail "sopa-companion --version says '$(sed -n 1p <<< "$said")', not the release's $app_version"
installed="$("$PY" -m pip show proseg-live | sed -n 's/^Version: //p')"
[[ "$installed" == "$program_version" ]] || fail "proseg-live $installed is installed, not the release's $program_version"
named="$(sed -n 's/^proseg-live: //p' <<< "$said")"
[[ -n "$named" && "$(realpath "$named")" == "$(realpath "$BIN/proseg-live")" ]] \
  || fail "sopa-companion --version names '$named' for proseg-live, not $BIN/proseg-live"
proseg="proseg $(cut -d. -f1-3 <<< "$program_version")"
[[ "$("$BIN/proseg-live" --version)" == "$proseg" ]] || fail "proseg-live --version does not say $proseg"

"$PY" - "$BIN" <<'PY' || fail "import sopa_companion"
import os, sys
import sopa_companion
loaded = [name for name in ("fastapi", "uvicorn", "httpx", "psutil") if name in sys.modules]
assert not loaded, f"import sopa_companion loads the engine's {', '.join(loaded)}"
environment = os.path.dirname(os.path.realpath(sys.argv[1]))
assert os.path.realpath(sopa_companion.__file__).startswith(environment + os.sep), sopa_companion.__file__
PY

# The desktop route: an installed copy reads the user's own settings file, here only for a port of its own.
mkdir -p "$HOME/.config/sopa-companion"
printf '[server]\nport = 8811\n' > "$HOME/.config/sopa-companion/config.toml"
"$APP" launch --no-browser --timeout 180 || fail "launch --no-browser did not start the engine"
"$PY" - "$HOME/.local/state/sopa-companion/run/server.json" <<'PY' || fail "the engine's /health"
import json, sys, urllib.request
record = json.load(open(sys.argv[1]))            # the bearer token stays in this process, never on a command line
request = urllib.request.Request(record["base_url"] + "/api/v1/health", headers={"Authorization": "Bearer " + record["token"]})
health = json.load(urllib.request.urlopen(request, timeout=60))
wanted = {"ok": True, "proseg": True, "proseg_missing": None, "zarr": True}
wrong = {key: health.get(key, "absent") for key, value in wanted.items() if health.get(key, "absent") != value}
assert not wrong and health["stain"]["available"] is True, (wrong, health.get("stain"))
assert record["port"] == 8811, f"the engine took port {record['port']}, not the settings file's 8811"
PY
status="$("$APP" status --json)" || fail "status --json"
"$PY" - "$status" "$HOME/.local/share/sopa-companion" <<'PY' || fail "status --json: $status"
import json, os, sys
status = json.loads(sys.argv[1])
assert status["healthy"] is True and os.path.realpath(status["data_dir"]) == os.path.realpath(sys.argv[2]), status
PY
"$APP" stop --force --timeout 120 || fail "stop --force did not stop the engine launch started"

# The console script's engine: sopa-companion serve, recognised by status and stop as the app's own.
"$APP" serve --port 8812 > "$HOME/serve.log" 2>&1 &
SERVE=$!
healthy=""
for _ in $(seq 120); do
  if "$APP" status --json 2>/dev/null | "$PY" -c 'import json, sys; sys.exit(not json.load(sys.stdin).get("healthy"))'; then
    healthy=1
    break
  fi
  kill -0 "$SERVE" 2>/dev/null || fail "sopa-companion serve ended: $(tail -n 20 "$HOME/serve.log")"
  sleep 1
done
[[ -n "$healthy" ]] || fail "status never said healthy for sopa-companion serve within 120 s: $(tail -n 20 "$HOME/serve.log")"
"$APP" stop --force --timeout 120 || fail "stop --force did not stop sopa-companion serve"
for _ in $(seq 30); do kill -0 "$SERVE" 2>/dev/null || break; sleep 1; done
! kill -0 "$SERVE" 2>/dev/null || fail "sopa-companion serve still runs after stop --force"
wait "$SERVE" || true
SERVE=""
echo "install_smoke.sh: sopa-companion $app_version and proseg-live $program_version from $RELEASE: ok"
