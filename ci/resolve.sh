#!/usr/bin/env bash
# Weekly (resolve.yml): do a fresh sopa environment and the app still resolve together, with the mixing check's
# packages, and with RAPIDS? Metadata only (uv pip compile --no-build): nothing is built, installed or run.
#   ci/resolve.sh [OUT]          (default ci/resolve/out)
# OUT gets the inputs: companion.in (ci/requirements.py), base.in (sopa[cellpose] unpinned, so a new sopa shows within a
# week, and the app), mixing.in (base, scvi-tools and scviva-tools), gpu-cu13.in (mixing and RAPIDS for CUDA 13), and
# one answer per set, Python and platform, <set>-<python>-<platform>.txt (<...>.err in its place when it does not resolve):
# - base and mixing on Python 3.12 and 3.13, Linux x86-64 and ARM64: one that does not resolve fails the run (exit 1);
# - gpu-cu13 on 3.12 and 3.13, Linux x86-64, with NVIDIA's index: a warning (RAPIDS pins dask exactly);
# - base on macOS arm64 (15.0) and Windows x64: warnings (macOS and Windows come later).
# ci/resolve/expected/ holds the accepted answers, which the workflow compares OUT with: copy OUT there to accept one.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-$HERE/resolve/out}"
command -v uv >/dev/null || { echo "ci/resolve.sh needs uv on PATH" >&2; exit 2; }
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
# The release's dependencies.txt without its comment, which names the app's version.
python3 "$HERE/requirements.py" | grep -v '^#' > "$OUT/companion.in"
printf '%s\n' 'sopa[cellpose]' '-r companion.in' > "$OUT/base.in"
printf '%s\n' '-r base.in' 'scvi-tools>=1.5' 'scviva-tools>=0.1.7' > "$OUT/mixing.in"
printf '%s\n' '-r mixing.in' 'rapids-singlecell-cu13[rapids]' > "$OUT/gpu-cu13.in"
cp "$HERE/resolve/overrides.txt" "$OUT/overrides.txt"
# Every input by its name alone, from OUT: the answers' notes ("via -r base.in") then read the same wherever OUT is.
cd "$OUT"
failed=0

# resolve LEVEL SET PYTHON PLATFORM [uv options]: LEVEL is error (a failure fails the run) or warning.
resolve() {
  local level="$1" set="$2" python="$3" platform="$4" name="$2-$3-$4"
  shift 4
  if uv pip compile --quiet --no-build --no-header --python-version "$python" --python-platform "$platform" \
      --override overrides.txt "$@" "$set.in" -o "$name.txt" 2> "$name.err"; then
    rm -f "$name.err"
    echo "$name: $(grep -c '^[a-zA-Z0-9]' "$name.txt") packages"
  else
    rm -f "$name.txt"
    echo "::$level::$name does not resolve: $(tail -n 3 "$name.err" | tr '\n' ' ')"
    [[ "$level" == warning ]] || failed=1
  fi
}

for python in 3.12 3.13; do
  for platform in x86_64-manylinux_2_28 aarch64-manylinux_2_28; do
    resolve error base "$python" "$platform"
    resolve error mixing "$python" "$platform"
  done
  resolve warning gpu-cu13 "$python" x86_64-manylinux_2_28 --extra-index-url https://pypi.nvidia.com
  MACOSX_DEPLOYMENT_TARGET=15.0 resolve warning base "$python" aarch64-apple-darwin
  resolve warning base "$python" x86_64-pc-windows-msvc
done
exit "$failed"
