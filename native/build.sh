#!/usr/bin/env bash
# Build proseg-live (native/bin/proseg-live): proseg 3.2.0 plus per-iteration snapshots and pause/step control.
#
# One source: proseg-live.patch, applied to proseg 3.2.0 exactly as crates.io publishes it (the crate's checksum is
# pinned below). native/proseg is the tree this script makes from the two, kept on disk to build and to edit, and
# never committed. The crate brings its own Cargo.lock, which the patch updates: the build uses exactly those
# dependency versions (cargo --locked), from ~/.cargo when they are all there, otherwise downloaded at those versions;
# they are never resolved again.
#   native/build.sh               build native/proseg; make it first when it is missing, or remake it when it was made
#                                 from another patch and not edited since
#   native/build.sh --fresh       remake native/proseg from the crate and the patch, then build. A tree holding
#                                 changes the patch does not have is kept in native/.cache/, never deleted
#   native/build.sh --save-patch  after editing native/proseg: write its changes into proseg-live.patch (only once the
#                                 crate plus the new patch gives the tree back exactly); no build
#   native/build.sh --tree          make native/proseg (as the build does) and stop: no cargo
#   native/build.sh --package DIR   make DIR/proseg-live-<VERSION>: a fresh tree from the crate and the patch, with
#                                   native/package/'s pyproject.toml and PROSEG-LIVE.md and the patch itself (the
#                                   proseg-live package's source; native/proseg untouched); no cargo
#   native/build.sh --wheel DIR [-- MATURIN ARGS]   --package into native/.cache, then maturin build into DIR with
#                                   the folder names below replaced (the remaps); then native/package/check.py
# Every step works in native/.cache/ (ignored by git, never shipped) and only then moves the result in, so a failure
# (no network, a wrong checksum, a patch that does not apply, an interruption) leaves native/proseg and the patch as
# they were.
set -euo pipefail
CALLER="$PWD"                       # a folder given to --package or --wheel is the caller's, relative to where it runs
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

VERSION=3.2.0
# crates.io's checksum of proseg-3.2.0.crate (the cksum of its index entry): the only upstream code this build trusts.
SHA256=d6a3c12a8605fff159b9f7bcbeb3a0ccb364499fd1f9603ea704b52e767c018d
URL="https://static.crates.io/crates/proseg/proseg-$VERSION.crate"
CACHE="$HERE/.cache"
STAMP="$CACHE/proseg.made"          # what native/proseg was last made or saved from: crate, patch and tree checksums
WORK=""

die() { echo "native/build.sh: $*" >&2; exit 1; }
mode="${1:-}"
OUT=""
# maturin's own options for --wheel: by default a wheel for this computer's glibc. A release-grade one, from a build
# environment with ziglang: native/build.sh --wheel DIR -- --zig --compatibility manylinux_2_28
MATURIN_ARGS=(--compatibility linux)
case "$mode" in
  ""|--fresh|--save-patch|--tree) [[ $# -le 1 ]] || die "one option at most" ;;
  --package|--wheel)
    [[ $# -ge 2 && -n "$2" ]] || die "$mode needs a folder"
    [[ $# -le 2 || "$mode" == --wheel ]] || die "--package takes one folder"
    if [[ $# -gt 2 ]]; then
      [[ "$3" == -- ]] || die "--wheel takes one folder, then -- and maturin's own options"
      MATURIN_ARGS=("${@:4}")
    fi
    OUT="$2"
    [[ "$OUT" == /* ]] || OUT="$CALLER/$OUT" ;;
  *) die "unknown option $mode (use --fresh, --save-patch, --tree, --package DIR or --wheel DIR)" ;;
esac
trap '[[ -z "$WORK" ]] || rm -rf "$WORK"' EXIT
trap 'exit 1' INT TERM HUP          # an interrupted run cleans up too

# A CARGO_TARGET_DIR set by the caller wins (a build that keeps its own target); otherwise native/target, which may be
# a symlink to a target shared with other builds.
TARGET="${CARGO_TARGET_DIR:-$HERE/target}"

work() { [[ -n "$WORK" ]] || { mkdir -p "$CACHE"; WORK="$(mktemp -d "$CACHE/work-XXXXXX")"; }; }
sha() { sha256sum "$1" | cut -d' ' -f1; }
# One checksum over every file's path and content (a cargo target folder inside left out).
tree_sum() {
  (cd "$1" && find . -name target -prune -o -type f -print0 | LC_ALL=C sort -z | xargs -0r sha256sum) \
    | sha256sum | cut -d' ' -f1
}
stamped() { [[ -f "$STAMP" ]] && grep -qxF "$1 $2" "$STAMP"; }
stamp() {
  mkdir -p "$CACHE"
  printf 'crate %s\npatch %s\ntree %s\n' "$SHA256" "$(sha proseg-live.patch)" "$(tree_sum proseg)" > "$STAMP.new"
  mv -f "$STAMP.new" "$STAMP"
}

# CRATE: the pinned crate, checked. cargo's own download of it, else this script's from an earlier run, else crates.io.
crate() {
  local hit
  for hit in "${CARGO_HOME:-${HOME:-}/.cargo}"/registry/cache/*/"proseg-$VERSION.crate" "$CACHE/proseg-$VERSION.crate"; do
    if [[ -f "$hit" ]] && [[ "$(sha "$hit")" == "$SHA256" ]]; then CRATE="$hit"; return; fi
  done
  work
  echo "Downloading proseg $VERSION from crates.io" >&2
  if command -v curl >/dev/null; then
    curl -fsSL --retry 2 -o "$WORK/download.crate" "$URL" || die "cannot download $URL"
  elif command -v wget >/dev/null; then
    wget -q -O "$WORK/download.crate" "$URL" || die "cannot download $URL"
  else
    die "proseg-$VERSION.crate is not in ~/.cargo, and there is no curl or wget to download it"
  fi
  [[ "$(sha "$WORK/download.crate")" == "$SHA256" ]] || die "the downloaded crate does not have crates.io's checksum"
  mv -f "$WORK/download.crate" "$CACHE/proseg-$VERSION.crate"
  CRATE="$CACHE/proseg-$VERSION.crate"
}

# Unpack the pinned crate as folder $1 (inside $WORK), without cargo's own bookkeeping files (the patch never sees them).
pristine() {
  crate
  mkdir "$1.unpack"
  tar -xzf "$CRATE" -C "$1.unpack" --no-same-owner || die "cannot unpack $CRATE"
  mv "$1.unpack/proseg-$VERSION" "$1"
  rmdir "$1.unpack"
  rm -f "$1/.cargo-ok" "$1/.cargo_vcs_info.json"
  chmod -R u+w "$1"
}

# A new tree in $WORK/new: the crate plus the patch.
make_tree() {
  work
  pristine "$WORK/new"
  patch --batch --forward --no-backup-if-mismatch --quiet -p1 -d "$WORK/new" -i "$HERE/proseg-live.patch" >&2 \
    || die "proseg-live.patch does not apply to proseg $VERSION; native/proseg is unchanged"
}

# The new tree replaces native/proseg. The old one goes too when it holds nothing the new one lacks: it is identical,
# or the stamp shows it unedited since this script made it (so the patch it came from has it). Otherwise it is kept.
install_tree() {
  if [[ -e proseg ]]; then
    if stamped tree "$(tree_sum proseg)" || diff -rq -x target proseg "$WORK/new" >/dev/null 2>&1; then
      mv proseg "$WORK/old"
    else
      local kept
      kept="$CACHE/proseg-kept-$(date +%Y%m%d-%H%M%S)-$$"
      mv proseg "$kept"
      echo "native/proseg held changes proseg-live.patch does not have: kept in $kept" >&2
    fi
  fi
  mv "$WORK/new" proseg
  stamp
}

# Panic messages name the source file of the crate they come from, so a plain build writes every dependency's path
# under cargo's registry into the binary, and with it the builder's home folder and user name, which the wheel then
# gives to everyone who installs it. --remap-path-prefix writes neutral names in their place: the home folder, then
# this folder, the target and cargo's home (when several match, rustc takes the last, so the deeper ones come later).
# Nothing else in the binary changes. The flags go after any the caller gave in CARGO_ENCODED_RUSTFLAGS or RUSTFLAGS
# (rustflags in cargo's own config files are not read while either is set), one flag per \x1f-separated field, so a
# folder with spaces stays one flag.
encoded_rustflags() {
  local sep=$'\x1f' flags=() remap home="${HOME:-}" cargo_home="${CARGO_HOME:-${HOME:-}/.cargo}" target="$TARGET"
  if [[ -n "${CARGO_ENCODED_RUSTFLAGS+set}" ]]; then
    [[ -z "$CARGO_ENCODED_RUSTFLAGS" ]] || IFS="$sep" read -r -a flags <<< "$CARGO_ENCODED_RUSTFLAGS"
  elif [[ -n "${RUSTFLAGS:-}" ]]; then
    read -r -a flags <<< "$RUSTFLAGS"
  fi
  [[ "$target" == /* ]] || target="$HERE/$target"            # relative: from this folder, where cargo runs
  [[ "$cargo_home" == /* ]] || cargo_home="$HERE/$cargo_home"
  # A home folder of "/" (some containers) would rename every path: it is left out, as an unset one is.
  if [[ -n "${home%/}" ]]; then
    flags+=("--remap-path-prefix=${home%/}=~")
  fi
  for remap in "$HERE=/proseg-live" "${target%/}=/target" "${cargo_home%/}=/cargo"; do
    flags+=("--remap-path-prefix=$remap")
  done
  local IFS="$sep"
  printf '%s' "${flags[*]}"
}

# The proseg-live package's source, in $1/proseg-live-$VERSION (PACKAGE): a fresh tree from the crate and the patch
# (never native/proseg), with the package's pyproject.toml and page, and the patch itself so that its changes can be read
# on their own. maturin builds its sdist and wheels from this folder.
package_tree() {
  PACKAGE="$1/proseg-live-$VERSION"
  [[ ! -e "$PACKAGE" ]] || die "$PACKAGE is there already: remove it or give another folder"
  make_tree
  # Cargo.toml.orig is cargo's own record of the manifest a crate was published from: cargo refuses a source that holds
  # one (a reserved name, which cargo writes itself when it packages), so the package's source leaves it out.
  rm -f "$WORK/new/Cargo.toml.orig"
  cp package/pyproject.toml package/PROSEG-LIVE.md proseg-live.patch "$WORK/new/"
  mkdir -p "$1"
  [[ ! -e "$PACKAGE" ]] || die "$PACKAGE is there already: remove it or give another folder"
  mv "$WORK/new" "$PACKAGE"
}

if [[ "$mode" == "--package" ]]; then
  package_tree "$OUT"
  echo "Made $PACKAGE: proseg-live's source (maturin sdist and maturin build take its Cargo.toml)"
  exit 0
fi

if [[ "$mode" == "--wheel" ]]; then
  command -v maturin >/dev/null || die "--wheel needs maturin on PATH (a build environment: pip install maturin ziglang)"
  work
  # The tree sits in native/.cache, inside this folder, so the remap of this folder covers its paths as well.
  package_tree "$WORK"
  mkdir -p "$WORK/wheels"
  CARGO_ENCODED_RUSTFLAGS="$(encoded_rustflags)" CARGO_TARGET_DIR="$TARGET" \
    maturin build --release --locked --manifest-path "$PACKAGE/Cargo.toml" --out "$WORK/wheels" "${MATURIN_ARGS[@]}"
  # Checked before it reaches DIR: a wheel that names this computer, or holds more or less than the one program, never
  # does.
  CARGO_TARGET_DIR="$TARGET" python3 "$HERE/package/check.py" wheel "$WORK"/wheels/*.whl
  mkdir -p "$OUT"
  for wheel in "$WORK"/wheels/*.whl; do
    mv -f "$wheel" "$OUT/"
    echo "Built $OUT/${wheel##*/}"
  done
  exit 0
fi

if [[ "$mode" == "--save-patch" ]]; then
  [[ -f proseg/Cargo.toml ]] || die "there is no native/proseg to save"
  work
  pristine "$WORK/a"
  mkdir "$WORK/b"
  (cd proseg && tar -cf - --exclude=target .) | (cd "$WORK/b" && tar -xf -)
  # Fixed file times, C-locale order and UTC: the same tree always gives the same patch, byte for byte.
  find "$WORK/a" "$WORK/b" -exec touch -h -d '2000-01-01 00:00:00 UTC' {} +
  status=0
  (cd "$WORK" && LC_ALL=C TZ=UTC diff -ruN a b) > "$WORK/patch" || status=$?
  [[ $status -le 1 ]] \
    || die "diff could not compare native/proseg with the crate (exit $status): proseg-live.patch is unchanged"
  pristine "$WORK/check"
  if ! patch --batch --forward --no-backup-if-mismatch --quiet -p1 -d "$WORK/check" -i "$WORK/patch" >&2 \
      || ! diff -rq "$WORK/check" "$WORK/b" >&2; then
    die "the crate plus the new patch would not give native/proseg back (a binary file?): proseg-live.patch is unchanged"
  fi
  mv -f "$WORK/patch" proseg-live.patch
  stamp
  echo "Saved native/proseg's changes to proseg $VERSION in $HERE/proseg-live.patch"
  exit 0
fi

if [[ "$mode" == "--fresh" || ! -f proseg/Cargo.toml ]]; then
  make_tree
  install_tree
elif [[ -f "$STAMP" ]]; then
  # A tree this script made (without a stamp: one made by hand, which is built as it is).
  if ! stamped tree "$(tree_sum proseg)"; then
    echo "native/proseg has changes proseg-live.patch does not have (--save-patch saves them): building it as it is" >&2
  elif ! { stamped crate "$SHA256" && stamped patch "$(sha proseg-live.patch)"; }; then
    echo "proseg-live.patch changed since native/proseg was made from it: making it again" >&2
    make_tree
    install_tree
  fi
fi
if [[ ! -f proseg/Cargo.lock ]]; then
  # A tree without its lock file (as an older checkout of this repository had): the one the crate and patch give.
  make_tree
  cp "$WORK/new/Cargo.lock" proseg/Cargo.lock
  echo "native/proseg had no Cargo.lock: added the one the crate and the patch give" >&2
fi
if [[ "$mode" == "--tree" ]]; then
  echo "native/proseg is ready"
  exit 0
fi

host="$("${RUSTC:-rustc}" -vV | sed -n 's/^host: //p')"
# The locked dependencies of this computer's platform: from ~/.cargo when they are all there, otherwise downloaded at
# exactly the locked versions. The build itself never touches the network.
cargo fetch --locked --offline --target "$host" --manifest-path proseg/Cargo.toml 2>/dev/null \
  || cargo fetch --locked --target "$host" --manifest-path proseg/Cargo.toml

CARGO_ENCODED_RUSTFLAGS="$(encoded_rustflags)" CARGO_TARGET_DIR="$TARGET" \
  cargo build --release --locked --offline --manifest-path proseg/Cargo.toml --bin proseg
mkdir -p bin
# Replace by rename, never in place: a running proseg-live keeps its own copy of the old file.
cp "$TARGET/release/proseg" bin/.proseg-live.new
strip bin/.proseg-live.new
chmod 755 bin/.proseg-live.new
mv -f bin/.proseg-live.new bin/proseg-live
echo "Installed $(bin/proseg-live --version) as $HERE/bin/proseg-live"
