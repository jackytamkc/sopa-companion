"""Deterministic process double: the same control/checkpoint/event contract, no Cellpose or GPU."""
import json
import hashlib
import os
from pathlib import Path
import sys
import time

import numpy as np

from sopa_companion.zarr_bridge.stain import atomic_json, checkpoint, emit, resolution_info


def models():
    return json.loads(os.environ["FAKE_STAIN_MODELS"]) if "FAKE_STAIN_MODELS" in os.environ else [
        dict(name=name, kind="builtin", path=None, date=None) for name in ("cpsam_v2", "cpdino", "cpdino-vitb", "cpsam")]


def arrays(rings):
    coords = np.array(rings, np.float32).reshape(-1, 2)
    return dict(coords=coords, offsets=np.arange(len(rings)+1, dtype=np.uint32)*5,
                bounds=np.array([[r[0][0], r[0][1], r[2][0], r[2][1]] for r in rings], np.float32).reshape(-1, 4))


def main():
    os.nice(19-os.getpriority(os.PRIO_PROCESS, 0))
    if sys.argv[1] == "--models":
        emit("models", models=models())
        return 0
    args = [a for a in sys.argv[1:] if a != "--join-only"]
    job = json.loads(Path(args[0]).read_text())
    join_only = "--join-only" in sys.argv[1:] or job.get("join_only", False)
    root = Path(job["out"]) / "stain"
    manifest = root / "tiles.json"
    records = json.loads(manifest.read_text()) if manifest.exists() else []
    total = 6
    progress = dict(done=len(records), total=total, revision=len(records), resolved=False, complete=False)
    record = dict(model=job["settings"]["model"])
    if not join_only and job["settings"].get("model_kind") == "custom":
        path = Path(job["settings"]["model_path"])
        record.update(model_path=str(path), model_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    if not join_only and os.environ.get("FAKE_STAIN_DOWNLOAD"):
        emit("progress", message="Downloading the model…")
        time.sleep(.2)
    if not join_only:
        emit("hello", **record, **resolution_info(job), device="cpu",
             cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
             nice=os.getpriority(os.PRIO_PROCESS, 0), display=os.environ.get("DISPLAY"))
    emit("plan", progress=progress)
    x, y, _, _ = job["region"]["bbox"]
    rings = [[[x+10+i*4, y+10], [x+12+i*4, y+10], [x+12+i*4, y+12], [x+10+i*4, y+12], [x+10+i*4, y+10]]
             for i in range(total)]
    for i in range(len(records), total):
        if join_only or not checkpoint(root, i):
            break
        time.sleep(float(os.environ.get("FAKE_STAIN_DELAY", ".15")))
        if os.environ.get("FAKE_STAIN_FAIL") == str(i):
            emit("error", message="Fake tile failure")
            return 1
        data = arrays([rings[i]])
        np.savez(root / f"tile-{i:06d}.npz", **data, box=data["bounds"][0])
        np.savez(root / f"tile-{i:06d}.masks.npz", masks=np.ones((2, 2), np.uint32), level=job["settings"].get("level", 0))
        records.append(dict(tile=i, box=data["bounds"][0].tolist(), cells=1, seconds=.15))
        atomic_json(manifest, records)
        progress = dict(progress, done=i+1, revision=i+1)
        emit("tile", progress=progress, tile=records[-1],
             summary=dict(cells=i+1, area_um2=dict(p10=4, median=4, p90=4)))
    emit("state", state="finishing", done=len(records))
    final = root / f"resolved-{len(records):06d}"
    final.mkdir(exist_ok=True)
    for k, v in arrays(rings[:len(records)]).items():
        np.save(final / f"{k}.npy", v)
    atomic_json(Path(job["out"]) / "output/boundaries.geojson", dict(type="FeatureCollection", features=[
        dict(type="Feature", id=i, properties={}, geometry=dict(type="Polygon", coordinates=[r]))
        for i, r in enumerate(rings[:len(records)])]))
    covered = None
    if len(records) < total:
        covered = dict(region=None, done=len(records), of=total, why="tiles", pieces=int(bool(records)))
        if records:
            bbox = [x, y, min(job["region"]["bbox"][2], x+max(20, 12+4*(len(records)-1))),
                    min(job["region"]["bbox"][3], y+20)]
            if min(bbox[2]-x, bbox[3]-y) >= 20:
                covered["region"] = dict(kind="box", box=bbox, bbox=bbox)
    result = dict(progress=dict(progress, done=len(records), revision=len(records)+1, resolved_dir=final.name,
                               resolved=True, complete=len(records) == total), covered=covered,
                  summary=dict(cells=len(records), area_um2=dict(p10=4, median=4, p90=4) if records else None))
    atomic_json(root / "result.json", result)
    emit("done", **result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
