# SoPA's optional companion entry point

`sopa.patch` targets SoPA 2.2.11. It adds `companion = ["sopa-companion>=0.3"]` to
SoPA's extras and exports `sopa.companionapp(sdata, **kwargs)` from a small
`sopa/_companion.py`. The wrapper imports `sopa_companion` only when called and
returns `sopa_companion.launch(sdata, **kwargs)` unchanged. Importing the wrapper
loads only the standard library; SoPA's existing CLI help/version import guard
is preserved.

A missing companion gives the plain hint `pip install 'sopa[companion]'`. A
missing dependency *inside* the companion keeps its original exception. The
wrapper requires `sopa_companion.SOPA_API >= 1`, so an older companion gives an
upgrade hint before launch.

From a clean SoPA 2.2.11 checkout, with `COMPANION_CHECKOUT` pointing to this
repository:

```sh
git apply --check "$COMPANION_CHECKOUT/packaging/sopa-fork/sopa.patch"
git apply "$COMPANION_CHECKOUT/packaging/sopa-fork/sopa.patch"
```

The initializer hunk was made from the installed SoPA 2.2.11 source (SHA256
`5c499a3ed615312132782ae0e34bf227d1c7632b4759f605e08a15be5c0ae8a0`). Installed
wheels do not include SoPA's `pyproject.toml`: the extra hunk uses only the
`[project.optional-dependencies]` header and the existing `cellpose` and
`stardist` lines as context. The local tests apply the patch to that context and
a copy of the installed initializer; application to the full SoPA source
checkout remains a release check. The upstream release is
[v2.2.11](https://github.com/prism-oncology/sopa/releases/tag/v2.2.11).

The extra needs a published `sopa-companion>=0.3` wheel on the selected package
index (0.3 is the first without proseg-live inside it, which its own package
brings, so the extra also replaces a 0.2 or 0.1 wheel installed before). It
deliberately contains no Git URL: both `sopa-companion` and `proseg-live` are
on PyPI from the companion's 0.7.0 on, so `pip install 'sopa[companion]'`
resolves there. Merging this patch and updating SoPA's lock file are separate
release steps; this patch performs neither.

## Companion dependencies and wheels

The companion's `pyproject.toml` uses lower bounds, separate from the desktop's
validated `requirements.txt`. These admit an environment with
SoPA 2.2.11, NumPy 2.4.6, PyArrow 26.0.0 and Zarr 3.4.1. FastAPI,
`uvicorn[standard]`, orjson and zstandard are named by the companion's `api`
extra and are also base requirements: the plain `sopa-companion` dependency
above brings the engine. The `sopa`, `stain`, `qc`, `all` and `test` extras follow
the API design. GPU tools remain optional.

Release wheels come from CI (`.github/workflows/release.yml`); by hand,
`python -m build` makes the companion's pure-Python wheel and
`native/build.sh --wheel DIR` proseg-live's (in a build environment with maturin;
`native/package/check.py` checks the result). The companion's wheel carries one
package, `sopa_companion` (its `__init__.py` is the notebook's entry point), with
the page, example config and icon, and no program: proseg-live is a package of
its own (`native/package/`, GPL v3), whose source distribution is proseg's crate
with the patch applied and the patch beside it.

## Checks without installing

Point `SOPA_COMPANION_API_PYTHON` at the Python of a SoPA 2.2.11 environment made
as API mode's users make theirs, then run the companion's `tests/test_packaging.py`
with the test interpreter.
The environment probe uses only `importlib.metadata`; it imports neither SoPA
nor its numerical dependencies. It skips with a reason when the interpreter
is absent. Tests cover the ranges, desktop pin compatibility, package and
asset coverage, patch application, lazy forwarding,
missing-package hints, old API versions and unchanged dependency errors.
Wheel builds, installs and an actual app launch are checked by release CI
(`release.yml`); these unit tests do not perform them.
