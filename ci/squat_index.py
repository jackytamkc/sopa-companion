"""A squatter's index, for the release job to put beside PyPI while it installs with the README's lines.

    python ci/squat_index.py DIR

Neither of the app's two names is registered on PyPI during the alpha, so anyone could upload them there, and an index
in play gives pip the highest version it finds anywhere. DIR becomes a local simple index (PEP 503: DIR/index.html and
DIR/<name>/index.html beside that name's wheels) holding what such a squatter would upload: sopa-companion 99.0, and
proseg-live 99.0 and 3.2.0.99 (one the companion's pin, >=3.2.0.1,==3.2.0.*, admits), each a wheel of metadata only.
The release job passes it as PIP_EXTRA_INDEX_URL=file://DIR, and ci/install_smoke.sh then checks that none of them was
installed.
Standard library only.
"""
from __future__ import annotations

import base64
import hashlib
from pathlib import Path
import sys
import zipfile

SQUATS = {"sopa-companion": ("99.0",), "proseg-live": ("99.0", "3.2.0.99")}


def wheel(folder: Path, name: str, version: str) -> Path:
    """A wheel of `name` `version` that installs nothing but its own metadata."""
    dist = f"{name.replace('-', '_')}-{version}"
    files = {f"{dist}.dist-info/METADATA": f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
                                           "Summary: a squatter's upload, for the release job's install check\n",
             f"{dist}.dist-info/WHEEL": "Wheel-Version: 1.0\nGenerator: ci/squat_index.py\nRoot-Is-Purelib: true\n"
                                        "Tag: py3-none-any\n"}
    record = "".join(f"{path},sha256={base64.urlsafe_b64encode(hashlib.sha256(text.encode()).digest()).rstrip(b'=').decode()},"
                     f"{len(text.encode())}\n" for path, text in files.items())
    files[f"{dist}.dist-info/RECORD"] = record + f"{dist}.dist-info/RECORD,,\n"
    path = folder / f"{dist}-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as archive:
        for member, text in files.items():
            archive.writestr(member, text)
    return path


def page(title: str, links: list[str]) -> str:
    anchors = "\n".join(f'<a href="{href}">{text}</a><br>' for href, text in links)
    return f"<!DOCTYPE html>\n<html><head><title>{title}</title></head><body>\n{anchors}\n</body></html>\n"


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__.split("\n\n")[1], file=sys.stderr)
        return 2
    root = Path(argv[0]).resolve()
    for name, versions in SQUATS.items():
        folder = root / name
        folder.mkdir(parents=True, exist_ok=True)
        links = []
        for version in versions:
            built = wheel(folder, name, version)
            digest = hashlib.sha256(built.read_bytes()).hexdigest()
            links.append((f"{built.name}#sha256={digest}", built.name))
        (folder / "index.html").write_text(page(f"Links for {name}", links))
    (root / "index.html").write_text(page("Simple index", [(f"{name}/", name) for name in SQUATS]))
    print(f"squat_index.py: {root}: " + "; ".join(f"{name} {', '.join(versions)}" for name, versions in SQUATS.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
