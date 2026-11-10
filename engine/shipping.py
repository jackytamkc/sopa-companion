"""What a deploy ships, in one place: the Log button's code hash (engine/runlog.py) and a deploy's tree hash, rsync
and backup read these lists, so a deployed copy and the copy it came from give
the same hash.

rsync filter syntax: a bare name matches at any depth, a leading "/" anchors at the app folder.
"""
from __future__ import annotations

# What is not code: run data, environments, build products (cargo's targets, what pip install -e leaves, the wheels
# python -m build writes into dist/, the weekly resolver's answers in ci/resolve/out), caches, agent workspaces and
# stray outputs. Harness outputs live outside the tree; the scouts' import-check.mjs wrote telemetry-body.json next to
# itself. config.local.toml is each copy's own (engine/config.py): a deploy never carries one copy's over another's,
# and it is no part of the code's hash.
NOT_CODE = ("var", ".venv", ".cache", "__pycache__", ".pytest_cache", ".git", ".agents", ".aws", ".codex",
            ".codex-tasks", ".fuse_hidden*", "*membership_debug.txt", "/native/target*", "/native/testdata",
            "/frontend/test/telemetry-body.json", "/config.local.toml", "*.egg-info", "/build", "/dist",
            "/ci/resolve/out")
# The binary is bound by the build manifest instead (deploy rebuilds it in production and compares it).
TREE_EXCLUDES = NOT_CODE + ("/native/bin",)
# A deploy leaves the harness and the frontend's development pages behind.
SHIP_EXCLUDES = TREE_EXCLUDES + ("/harness", "/frontend/dev")
