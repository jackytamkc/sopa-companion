"""Run the actual installed Cellpose loaders without a window, a display, weights or a shared HOME."""
import importlib.metadata
import json
import logging
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

import numpy as np
import shapely
from shapely.geometry import Polygon, box, mapping

from sopa_companion.zarr_bridge import stain_patches as P
from sopa_companion.tests.fake_stain_patches import pixels


def setup(root, geometry=False, channels=None, direct=False):
    (root/"stain").mkdir()
    (root/"output").mkdir()
    image = {"key":"image","channels":["DAPI","Boundary"],"to_frame":np.eye(3).tolist(),"levels":[{
        "path":"unused","shape":[2,300,800],"dims":["c","y","x"],"origin":[0,0],"pixel":[2,3] if geometry else [1,1]}]}
    shapes = [box(x,20,x+20,40) for x in range(20,780,40)]
    if geometry:
        shapes = [Polygon([(10,12),(100,12),(100,150),(10,150)], [[(30,30),(50,30),(50,60),(30,60)]])]
    job = {"path":"not-opened","image":image,"settings":{"nucleus":"Boundary","cytoplasm":"DAPI","method":"other"},
           "region":{"kind":"full","bbox":[0,0,1600,900] if geometry else [0,0,800,300]}}
    if channels:
        from sopa_companion.zarr_bridge import stain
        image["channels"] = [f"channel-{i}" for i in range(channels)]
        image["levels"][0]["shape"][0] = channels
        job["settings"] = dict(method="instanseg", channels=image["channels"])
        yy, xx = np.mgrid[:300, :800]
        planes = np.stack([(xx % 23 + yy % 17 + 1)*(i+1) for i in range(channels)]).astype(np.float32)
        if channels > 3:
            planes[:2] = 0  # InstanSeg proposals can find image detail beyond the first two channels.
        stain.image_array = lambda path: planes
        P.image_pixels = stain.image_pixels
    else:
        P.image_pixels = pixels
    (root/"stain/job.json").write_text(json.dumps(job))
    (root/"output/boundaries.geojson").write_text(json.dumps({"features":[{"geometry":mapping(p)} for p in shapes]}))
    meta = {"id":"source","sample":"test","progress":{"resolved":True,"revision":1}}
    if direct:
        meta.update(kind="patches", start_from="empty")
        (root/"output/boundaries.geojson").unlink()
    result = P.Result(root,meta)
    folder = root/"export"
    folder.mkdir()
    manifest = result.export({"set":"a"*32,"patches":[{"box":[0,0,128,128]},{"box":[128,0,256,128]}]},folder,folder)
    return result, folder, manifest


class Count:
    def __init__(self): self.value=0
    def set(self, value): self.value=int(value)
    def get(self): return self.value
    def __gt__(self, value): return self.value>value
    def __eq__(self, value): return self.value==value


class Parent:
    """Only widgets/drawing are stand-ins. _load_seg, _initialize_images and _masks_to_gui run unchanged."""
    load_3D = False
    logger = logging.getLogger("loader-check")
    def __init__(self):
        self.colormap=np.ones((10000,3),np.uint8)*128
        self.ncells=Count()
        self.ViewDropDown=SimpleNamespace(setCurrentIndex=lambda i:None)
    def reset(self): pass
    def set_normalize_params(self, value): self.normalize=value
    def clear_all(self): pass
    def compute_saturation(self): pass
    def compute_scale(self): pass
    def draw_layer(self): pass
    def toggle_mask_ops(self): pass
    def enable_buttons(self): pass
    def update_layer(self): pass
    def update_plot(self): pass



def same_cells(a,b):
    """Equal label images up to renumbering: background matches and the labels map one to one."""
    if a.shape != b.shape or not np.array_equal(a > 0, b > 0):
        return False
    pairs = np.unique(np.stack([a[a > 0], b[b > 0]]), axis=1)
    return len(np.unique(pairs[0])) == len(np.unique(pairs[1])) == pairs.shape[1]


def refuses(call, message):
    try:
        call()
        raise AssertionError("invalid patches accepted")
    except ValueError as exc:
        assert message in str(exc), str(exc)

def compatibility(folder, manifest=None):
    """The installed Cellpose's loaders on an export in `folder`: `manifest` (a dict, or the path of a chosen folder's
    .sopa-companion/sets/<set>.json), else the set folder's own patches.json. In a chosen folder that holds other sets
    beside it (a model learns from them all) the counts are the folder's, so this set's files are looked for among them."""
    from cellpose import io
    from cellpose.gui import io as gui_io
    assert importlib.metadata.version("cellpose") == "4.2.1.1"
    if manifest is None:
        manifest = json.loads((folder/"patches.json").read_text())
    elif not isinstance(manifest, dict):
        manifest = json.loads(Path(manifest).read_text())
    alone = len(list((folder/".sopa-companion"/"sets").glob("*.json"))) <= 1
    files = io.get_image_files(str(folder), "_seg.npy")
    assert {str(folder/p["image"]) for p in manifest["patches"]} <= set(files)
    assert not alone or len(files) == len(manifest["patches"])
    for patch in manifest["patches"]:
        filename = folder/patch["mask"]
        data = np.load(filename,allow_pickle=True).item()
        mask = io.imread(str(filename))
        assert np.array_equal(mask,data["masks"])
        parent = Parent()
        gui_io._load_seg(parent,filename=str(filename))
        assert parent.loaded and parent.filename == str(folder/patch["image"])
        assert parent.stack.shape == (1,*mask.shape,1 if len(manifest["channels"]) == 1 else 3)
        assert np.array_equal(parent.cellpix[0],P.dense_mask(mask)[0])
        assert parent.ncells.get() == P.dense_mask(mask)[1]
        assert parent.normalize == data["normalize_params"]
    images, labels, names, *_ = io.load_train_test_data(str(folder), mask_filter="_seg.npy")
    assert names == files and len(images) == len(labels) == len(files)
    for image, label, name in zip(images, labels, names):
        patch = next((p for p in manifest["patches"] if p["image"] == Path(name).name), None)
        if patch is None:
            continue                                         # another set's, beside this one
        saved = np.load(folder/patch["mask"],allow_pickle=True).item()
        assert image.shape == (*label.shape,len(manifest["channels"])) and np.array_equal(label,saved["masks"])
    gui_images, gui_labels, gui_names, *_ = gui_io._get_train_set(files)
    assert gui_names == files and len(gui_images) == len(labels)
    # The GUI renumbers cells in raster order (fastremap.renumber): the same cells, possibly other numbers.
    assert all(same_cells(a,b) for a,b in zip(gui_labels,labels))
    return io


def check(root, case):
    if case == "ranges":
        from unittest.mock import patch
        result, folder, manifest = setup(root)
        first = np.arange(19, dtype=np.float32)
        first[6] = np.nan
        values = {"DAPI": first, "Boundary": np.arange(18, -1, -1, dtype=np.float32)}
        rules = dict(version=1, size=dict(min=400, max=400), brightness={
            "DAPI": dict(min=5, max=12), "Boundary": dict(min=10, max=13)})
        with patch.object(P.Result, "brightness", side_effect=lambda channel, progress: values[channel]):
            data = P.Result(root, {**result.meta, "exclusions": rules})
        assert [p.bounds[0] for p in data.shapes] == [220, 260, 300, 340]
        mask = data.masks([200, 0, 400, 128])
        assert mask.max() == 4
        manifest = data.export({"patches": [{"box": [200, 0, 400, 128]}]}, folder, folder)
        assert manifest["patches"][0]["cells"] == 4
        assert P.range_keep(np.array([99.99999237], np.float32), {}, [False],
                            {"size": {"min": 99.999995, "max": None}}).tolist() == [False]
        assert P.range_keep([400, 400], {"DAPI": np.array([np.nan, 5], np.float32)}, [False, True],
                            {"brightness": {"DAPI": {"min": 10, "max": None}}}).tolist() == [True, False]
        return
    if case in ("level", "level-model"):
        from unittest.mock import patch
        from sopa_companion.zarr_bridge import stain as S
        data, folder, manifest = setup(root)
        job = data.job
        job["settings"].update(level=1, diameter=24, min_size=64,
                               normalize=dict(sharpen_radius=8, smooth_radius=4, tile_norm_blocksize=128))
        job["image"]["levels"].append(dict(path="half", shape=[2, 150, 400], dims=["c", "y", "x"],
                                          origin=[0, 0], pixel=[2, 2]))
        job["out"] = str(root)
        (root / "stain/job.json").write_text(json.dumps(job))
        data = P.Result(root, data.meta)
        yy, xx = np.mgrid[:150, :400]
        planes = np.stack([xx, yy]).astype(np.float32)
        seen = []

        def read(path):
            seen.append(path)
            return planes

        class Model:
            def __init__(self, received, out):
                assert received == job and out == root
                assert S.at_level(received["settings"], S.job_level(received)[2])["diameter"] == 12

            def __call__(self, image):
                assert image.shape == (128, 128, 3)
                mask = np.zeros((128, 128), np.uint32)
                mask[20:40, 20:40] = 1
                return mask

        with patch.object(S, "image_array", side_effect=read), patch.object(P, "image_pixels", S.image_pixels):
            if case == "level-model":
                with patch.object(S, "Cellpose", Model):
                    P.export_model(dict(root=str(root), meta=data.meta, manifest=manifest,
                                        staging=str(folder), target=str(folder)))
                manifest = json.loads((folder / "patches.json").read_text())
            else:
                manifest = data.export(manifest, folder, folder)
        assert seen and all(p == "not-opened/half" for p in seen)
        assert [manifest[k] for k in ("level", "factor", "um_per_px")] == [1, 2, 2]
        assert manifest["run_pixel_to_sample"] == [0, 0, 2, 2]
        saved = np.load(folder / "patch-01_seg.npy", allow_pickle=True).item()
        assert saved["diameter"] == 12 and saved["normalize_params"]["sharpen_radius"] == 4
        assert saved["normalize_params"]["smooth_radius"] == 2
        assert saved["normalize_params"]["tile_norm_blocksize"] == 64
        assert manifest["settings"]["diameter"] == 24
        assert len(job["image"]["levels"]) == 2
        return
    if case == "stem":
        # A chosen folder's export: files named by the manifest's stem, written only into the staging folder, each
        # mask's file naming its image in the folder it is copied into; the GUI's loaders find them there.
        import shutil
        result, _, _ = setup(root)
        staging, target = root/"staging", root/"chosen"
        staging.mkdir()
        target.mkdir()
        stem = "Sample-1-20261024-101010"
        named = result.export({"set":"b"*32, "stem":stem, "patches":[{"box":[0,0,128,128]},{"box":[128,0,256,128]}]},
                              staging, target)
        assert [p["image"] for p in named["patches"]] == [f"{stem}-01.tif", f"{stem}-02.tif"]
        assert [p["mask"] for p in named["patches"]] == [f"{stem}-01_seg.npy", f"{stem}-02_seg.npy"]
        assert sorted(p.name for p in staging.iterdir()) == sorted(
            [f"{stem}-01.tif", f"{stem}-01_seg.npy", f"{stem}-02.tif", f"{stem}-02_seg.npy", "patches.json"])
        assert not list(target.iterdir())
        saved = np.load(staging/f"{stem}-01_seg.npy", allow_pickle=True).item()
        assert saved["filename"] == str(target/f"{stem}-01.tif")
        for bad in ("../up", ".hidden", "a/b", "x"*81, 7):
            refuses(lambda: result.export({"stem":bad, "patches":[{"box":[0,0,128,128]}]}, staging, target),
                    "The patch names could not be made")
        for path in staging.glob(f"{stem}-*"):
            shutil.copy(path, target/path.name)
        compatibility(target, named)
        return
    if case == "unsafe":
        class Unsafe:
            def __reduce__(self):
                return eval, ("__import__('pathlib').Path("+repr(str(root/"executed"))+").touch()",)
        path = root/"bad_seg.npy"
        np.save(path, {"masks":Unsafe()})
        try:
            P.read_seg(path,(128,128))
            raise AssertionError("unsafe pickle accepted")
        except ValueError:
            pass
        assert not (root/"executed").exists()
        return
    channels = int(case.split("-")[1]) if case.startswith("instanseg-") else None
    result, folder, manifest = setup(root, case=="geometry", channels, case in ("empty", "direct-proposal"))
    if case == "pixel-edges":
        job = result.job
        job["image"]["levels"][0].update(shape=[2, 1000, 1000], pixel=[.2125, .2125])
        job["region"] = {"kind": "rectangle", "bbox": [.05, .05, 100.05, 100.05]}
        (root/"stain/job.json").write_text(json.dumps(job))
        shapes = [box(x, y, x+2, y+2) for y in range(2, 98, 8) for x in range(2, 98, 8)]
        (root/"output/boundaries.geojson").write_text(json.dumps({"features":[{"geometry":mapping(p)} for p in shapes]}))
        for kind in ("patches", "stain"):
            crop = P.Result(root, {**result.meta, "kind": kind})
            assert crop.extent == [471, 471]
            for count, size, expected in ((3, 128, 3), (9, 128, 9), (1, 256, 1), (3, 256, 1)):
                boxes = crop.propose(count, size)
                assert len(boxes) == expected, (kind, count, size, boxes)
                assert boxes == crop.propose(count, size)
                assert all(not box(*a).intersection(box(*b)).area for i,a in enumerate(boxes) for b in boxes[i+1:])
            assert crop.validate([[0, 0, 128, 128], [215, 215, 471, 471]])
        print("471-pixel crop: both paths pack 3/9 x 128 and 1 x 256; edge boxes accepted")
        return
    if case == "model-pixels":
        import tifffile
        result.job["settings"]["normalize"] = {"normalize": True, "invert": True}
        before = result.pixels([0,0,128,128]).astype(np.float32)
        calls = []
        def segment(image):
            calls.append(image.shape)
            image[:] = 1-image/image.max()
            mask = np.zeros(image.shape[:2], np.uint32)
            mask[20:40,20:40] = 7
            return mask
        exported = result.export(manifest, folder, folder, segment)
        assert calls == [(128,128,3)]*2
        assert np.array_equal(tifffile.imread(folder/"patch-01.tif"), before)
        saved = np.load(folder/"patch-01_seg.npy", allow_pickle=True).item()
        assert saved["normalize_params"]["invert"] is True
        assert saved["masks"][25,25] == 1 and exported["patches"][0]["cells"] == 1
        return
    if case == "model-centroids":
        job = result.job
        job["image"]["levels"][0].update(origin=[100,200], pixel=[2,3])
        job["region"]["bbox"] = [140,260,1700,1100]
        tissue = list(box(184,350,264,380).exterior.coords)
        removed = list(box(216,355,248,375).exterior.coords)
        mask = np.zeros((64,64), np.uint32)
        for x, label in zip(range(4,53,8), [7,90,500,900,2000,2**31,2**32-1]):
            mask[8:12,x:x+4] = label
        for kind in ("patches", "stain"):
            for rings, exclusions, kept in (([tissue],[removed],[4,12,44]), ([tissue],[],[4,12,20,28,36,44]),
                                             ([[tissue,removed]],[],[4,12,20,36,44]),
                                             (None,[],[4,12,20,28,36,44,52]), ([],[],[4,12,20,28,36,44,52])):
                job["tissue"] = rings
                (root/"stain/job.json").write_text(json.dumps(job))
                data = P.Result(root, {**result.meta,"kind":kind}, exclusions)
                assert data.geometry == [140,260,2,3] and data.window[:2] == (20,20)
                exported = data.export({"patches":[{"box":[16,24,80,88]}]}, folder, folder,
                                       lambda image: mask.copy())
                actual = P.read_seg(folder/"patch-01_seg.npy", (64,64))
                expected = np.zeros_like(mask)
                for label, x in enumerate(kept, 1):
                    expected[8:12,x:x+4] = label
                assert np.array_equal(actual, expected), (kind, rings, exclusions, actual)
                assert exported["patches"][0]["cells"] == len(kept)
                assert exported["patches"][0]["export_hash"] == P.mask_hash(expected)
        return
    if case == "tissue":
        job = result.job
        job["image"]["levels"][0].update(origin=[100,200], pixel=[2,3])
        job["region"]["bbox"] = [100,200,1700,1100]
        tissue = shapely.union_all([box(x-2,390,x+2,394) for x in (228,484)])
        job["tissue"] = [list(p.exterior.coords) for p in tissue.geoms]
        shapes = [box(100+2*x,260,100+2*x+20,290) for x in range(20,780,40)]
        (root/"output/boundaries.geojson").write_text(json.dumps({"features":[{"geometry":mapping(p)} for p in shapes]}))
        def dark_tissue(job, window, step=1):
            image = pixels(job, window, step)
            image[:, np.arange(window[0],window[2],step)>=400, 0] += 1000
            return image
        P.image_pixels = dark_tissue
        for kind in ("patches", "stain"):
            (root/"stain/job.json").write_text(json.dumps(job))
            data = P.Result(root, {**result.meta, "kind":kind})
            boxes = data.propose(2,128)
            assert len(boxes) == 2
            assert all(tissue.covers(box(*data.world(b)).centroid) for b in boxes)
            assert all(b[2] <= 400 for b in boxes)  # These dark islands fail the old Otsu coverage test.
        job["tissue"] = None
        (root/"stain/job.json").write_text(json.dumps(job))
        fallback = {kind:P.Result(root, {**result.meta, "kind":kind}).propose(2,128)
                    for kind in ("patches", "stain")}
        assert all(b[2] > 400 for b in fallback["patches"])
        job["tissue"] = []
        (root/"stain/job.json").write_text(json.dumps(job))
        for kind in ("patches", "stain"):
            empty = P.Result(root, {**result.meta, "kind":kind})
            assert empty.tissue is None and empty.propose(2,128) == fallback[kind]
        return
    if case == "exclusions":
        rings = [list(box(29,29,31,31).exterior.coords), list(box(60,20,65,40).exterior.coords)]
        data = P.Result(root, result.meta, rings)
        mask = data.masks([0,0,128,128])
        assert not mask[20:40,20:40].any() and mask[25,62] and mask[25,105]
        assert not data.masks([16,16,32,48]).any()  # Use the original cell centroid, not the clipped crop's.
        boundary = P.Result(root, result.meta, [list(box(30,30,110,31).exterior.coords)])
        assert not boundary.masks([0,0,128,128]).any()  # Centroids on edges and corners are removed too.
        boundary.tissue = box(0,0,64,64)
        boundary.exclusions = box(64,0,65,64)
        assert boundary.allowed_centre([0,0,64,128])  # Tissue edge.
        assert not boundary.allowed_centre([0,0,128,128])  # Removed-area corner takes precedence.
        filtered = folder/"filtered"
        filtered.mkdir()
        data.export(manifest, filtered, filtered)
        assert not P.read_seg(filtered/"patch-01_seg.npy", (128,128))[20:40,20:40].any()
        compatibility(filtered)
        ring = list(box(0,0,400,300).exterior.coords)
        for kind in ("patches", "stain"):
            data = P.Result(root, {**result.meta, "kind":kind}, [ring])
            boxes = data.propose(2,128)
            assert len(boxes) == 2 and all(not Polygon(ring).covers(box(*data.world(b)).centroid)
                                         for b in boxes)
        return
    if case == "size-rule":
        # The run's size rule (engine/stain_exclusions.py) leaves the same small cells out of masks, exports and
        # proposals as it does out of the outlines and Save as shapes.
        cells = [box(20,20,40,40), box(60,20,70,30), box(100,20,140,60)]       # 400, 100 and 1600 µm²
        (root/"output/boundaries.geojson").write_text(json.dumps({"features":[{"geometry":mapping(p)} for p in cells]}))
        sized = P.Result(root, {**result.meta, "exclusions":{"version":1, "min_area_um2":200}})
        mask = sized.masks([0,0,160,128])
        assert mask[30,30] and not mask[25,65] and mask[40,120]
        assert not sized.tree.query(box(55,15,75,35), predicate="intersects").size
        assert len(P.Result(root, {**result.meta, "exclusions":{"version":2, "min_area_um2":None}}).shapes) == 3
        target = folder/"sized"
        target.mkdir()
        sized.export(manifest, target, target)
        saved = P.read_seg(target/"patch-01_seg.npy", (128,128))
        assert saved[30,30] and not saved[25,65] and saved[40,120]
        compatibility(target)
        return
    if case == "brightness-rule":
        # The run's brightness rule (engine/stain_intensity.py): cells dimmer than it in the run's image stay out of
        # proposals and masks, measured as the engine measures them.
        import zarr
        from sopa_companion.engine import stain_intensity
        level = result.job["image"]["levels"][0]
        planes = np.zeros(level["shape"], np.uint16)
        planes[0, 20:40, 20:420] = 1000                       # the first ten cells hold a nucleus, the other nine none
        planes[0, 20:40, 420:] = 50
        array = zarr.open_array(store=str(root/"image.zarr"/level["path"]), mode="w", shape=planes.shape,
                                dtype=planes.dtype, chunks=(1, 64, 64))
        array[:] = planes
        job = {**result.job, "path": str(root/"image.zarr")}
        (root/"stain/job.json").write_text(json.dumps(job))
        rule = {"version": 1, "min_brightness": {"channel": "DAPI", "value": 500}}
        dim = P.Result(root, {**result.meta, "exclusions": rule})
        assert len(dim.shapes) == 10 and all(p.bounds[2] <= 420 for p in dim.shapes)
        mask = dim.masks([360, 0, 488, 128])
        assert mask[25, 25] and not mask[25, 65]          # the last bright cell (380..400) is kept, the first dim not
        assert len(P.Result(root, {**result.meta, "exclusions": {**rule, "min_brightness": None}}).shapes) == 19
        # A finished run's values the engine stored for its final outlines (same cells, same order) are read, not
        # measured again: here stored as all dim, so every cell goes; stored for other outlines, they are not used.
        final = root/"stain/resolved-000003"
        final.mkdir()
        np.save(final/"offsets.npy", np.zeros(20, np.uint32))
        stamp = (final/"offsets.npy").stat().st_mtime_ns
        (root/"stain/brightness").mkdir()
        np.save(root/f"stain/brightness/resolved-000003-c0-{stain_intensity.CACHE_TAG}-{stamp}.npy", np.zeros(19, np.float32))
        meta = {**result.meta, "progress": {"resolved": True, "revision": 4, "done": 3}, "exclusions": rule}
        assert len(P.Result(root, meta).shapes) == 0
        np.save(root/f"stain/brightness/resolved-000003-c0-{stain_intensity.CACHE_TAG}-{stamp}.npy", np.zeros(5, np.float32))
        assert len(P.Result(root, {**meta, "progress": {**meta["progress"], "revision": 5}}).shapes) == 10
        offsets, coords = stain_intensity.rings(dim.shapes[:2])
        assert offsets.tolist() == [0, 5, 10] and coords[:5].tolist() == [list(map(float, c)) for c in
                                                                           dim.shapes[0].exterior.coords]
        return
    if case == "settings":
        from cellpose.models import normalize_default
        from cellpose.gui import io as gui_io
        from sopa_companion.engine.cellpose_settings import gui_settings
        assert P.NORMALIZE_DEFAULT == normalize_default
        for name, diameter in (("custom", 87.5), ("automatic", None), ("auto-string", "auto")):
            settings = {"method":"cellpose", "nucleus":"DAPI", "cytoplasm":"Boundary", "model":"cpsam",
                        "flow_threshold":.85, "cellprob_threshold":-2.25, "diameter":diameter,
                        "normalize":{"percentile":[2.,98.], "tile_norm_blocksize":64, "sharpen_radius":2,
                                     "smooth_radius":1, "invert":True}}
            if name == "custom":
                settings["model_path"] = str(root/"trained-model")
            result.job["settings"] = settings
            target = folder/name
            target.mkdir()
            result.export({"patches":[{"box":[0,0,96,64]}, {"box":[128,0,256,96]}]}, target, target)
            for path in target.glob("*_seg.npy"):
                saved = np.load(path,allow_pickle=True).item()
                assert saved["diameter"] == (87.5 if name == "custom" else 0)
                assert saved["flow_threshold"] == .85 and saved["cellprob_threshold"] == -2.25
                assert saved["normalize_params"] == {**normalize_default, **settings["normalize"]}
                assert saved["model_path"] == settings.get("model_path", "cpsam")
                assert "manual_changes" not in saved
            assert gui_settings(target, []) is None
            compatibility(target)
            parent = Parent()
            gui_io._load_seg(parent, filename=str(target/"patch-01_seg.npy"))
            parent.segmentation_settings = SimpleNamespace(diameter=saved["diameter"], flow_threshold=.85,
                                                           cellprob_threshold=-2.25)
            parent.current_model_path = saved["model_path"]
            parent.RGBDropDown = SimpleNamespace(name_map={"RGB":None})
            parent.color, parent.track_changes = "RGB", []
            parent.get_normalize_params = lambda:parent.normalize
            gui_io._save_sets(parent)
            read = gui_settings(target, [{"name":"cpsam"}, {"name":"trained-model"}])
            assert read and read["files"] == 1 and read["settings"]["model_path"] == saved["model_path"]
            assert read["settings"]["diameter"] == (87.5 if name == "custom" else None)
        return
    if case == "sizes":
        result.extent = [8192,8192]
        result.region = box(0,0,8192,8192)
        assert result.validate([[0,0,16,4096], [0,4096,4096,4112]])
        assert result.validate([[0,0,4096,4096]])  # Exactly the total pixel limit.
        large = root/"large_seg.npy"
        P.write_seg(large, None, np.zeros((4096,4096),np.uint32), root/"large.tif", result.job["settings"])
        assert large.stat().st_size > 64*1024*1024
        assert not P.read_seg(large, (4096,4096)).any()
        large.unlink()
        for boxes in ([[0,0,15,16]], [[0,0,16,4097]], [[0,0,4097,16]],
                      [[0,0,4096,4096], [4096,0,4112,16]]):
            refuses(lambda:result.validate(boxes), "patches" if len(boxes)>1 else "Box 1 must be 16 to 4096")
        for size in (63,4097,True,64.5):
            refuses(lambda:result.propose(1,size), "64 and 4096")
        result.extent = [63,8192]
        refuses(lambda:result.propose(1,64), "smaller patch size")
        result.extent = [8192,8192]
        # Sampled image reads keep the large-size bound check small and CPU-only.
        P.image_pixels = lambda job,b,step=1: pixels(job, [0,0,int(np.ceil((b[2]-b[0])/step)),
                                                        int(np.ceil((b[3]-b[1])/step))])
        assert len(result.propose(2,4096)) == 1  # The per-set pixel cap also applies to proposals.
        result.region = box(5,5,8192,8192)
        refuses(lambda:result.validate([[0,0,64,64]]), "inside the finished part")
        return
    if case == "preview":
        # An item before it exists: its job comes in memory, no run folder is read, and it has the frame, boxes and
        # checks of the item created from that job.
        meta = {**result.meta, "kind": "patches"}
        item = P.Result(root, meta)
        ahead = P.Result(root/"not-created", meta, job=json.loads((root/"stain/job.json").read_text()))
        assert not (root/"not-created").exists()
        assert ahead.geometry == item.geometry and ahead.window == item.window and ahead.extent == [800, 300]
        assert ahead.propose(3, 128) == item.propose(3, 128)
        assert ahead.validate([[0,0,128,128]]) == [[0,0,128,128]]
        refuses(lambda: ahead.propose(1, 512), "Choose a smaller patch size for this region")
        refuses(lambda: ahead.validate([[700,0,900,128]]), "Keep box 1 inside the region")
        refuses(lambda: ahead.validate([[0,0,128,128], [0,0,8,8]]), "Box 2 must be 16 to 4096")
        return
    if case == "empty":
        assert all(p["cells"] == 0 for p in manifest["patches"])
        compatibility(folder)
        print("Cellpose 4.2.1.1 all-zero masks accepted by GUI and training loaders")
        return
    if channels:
        assert manifest["method"] == "instanseg" and manifest["channels"] == result.job["settings"]["channels"]
        assert result.pixels([0,0,128,128],step=8).shape == (16,16,channels)
        assert result.propose(8,128) == result.propose(8,128)
        compatibility(folder)
        patch = folder/"patch-01_seg.npy"
        saved = np.load(patch,allow_pickle=True).item()
        saved["masks"][60:80,60:80] = 99
        np.save(patch,saved)
        assert P.inspect_patch(folder,manifest["patches"][0])["state"] == "corrected"
        return
    if case == "direct-proposal":
        # A sample-sized image with a real pyramid geometry; fail if level zero is requested.
        image = result.job["image"]
        image["levels"][0]["shape"] = [2, 64000, 96000]
        image["levels"].append({**image["levels"][0], "path":"low", "shape":[2,1000,1500], "pixel":[64,64]})
        result.job["region"]["bbox"] = [0,0,96000,64000]
        result.extent = [96000,64000]
        result.region = box(0,0,96000,64000)
        reads = []
        def low(job, window, step=1):
            assert job["image"]["levels"][0]["path"] == "low"
            reads.append(window)
            y,x = np.mgrid[:1000,:1500]
            plane = (10 + x%17 + y%13).astype(np.float32)
            plane[:, :300] = 0
            return plane[...,None]
        P.image_pixels = low
        import time
        start = time.perf_counter()
        a = result.propose(8,256)
        seconds = time.perf_counter()-start
        assert seconds < 3, seconds
        assert len(a) == 8 and len(reads) == 1 and a == result.propose(8,256)
        assert all(b[0] >= 300*64-256 for b in a)
        assert all(not box(*b).intersection(box(*c)).area for i,b in enumerate(a) for c in a[i+1:])
        result.region = Polygon([(0,0),(96000,0),(0,64000)])
        assert all(result.region.covers(box(*b)) for b in result.propose(8,256))
        def island(*args, **kwargs):
            plane = np.zeros((1000,1500,1),np.float32)
            y,x = np.mgrid[:12,:12]
            plane[300:312,600:612,0] = 100+x%7+y%5
            return plane
        P.image_pixels = island
        result.region = box(0,0,96000,64000)
        assert result.propose(1,256), "small islands between grid boxes must be found"
        P.image_pixels = lambda *a, **k: np.zeros((1000,1500,1),np.float32)
        try:
            result.propose(8,256)
            raise AssertionError("background accepted")
        except ValueError as exc:
            assert "No patches" in str(exc)
        print(f"Image-only 96k x 64k proposal via 1500 x 1000 pyramid: {seconds:.3f}s")
        return
    if case == "geometry":
        mask = P.read_seg(folder/"patch-01_seg.npy",(128,128))
        assert mask[5,6] == 1 and mask[12,16] == 0 and mask[60,60] == 0
        assert manifest["method"] == "other" and manifest["scale"] == [2,3]
        assert manifest["channels"] == ["Boundary","DAPI",None]
        assert np.array_equal(mask,result.masks([0,0,128,128]))
        # A paused run exports only committed coverage and uses the same joined model polygons.
        (root/"stain/tile-000000.geojson").write_bytes((root/"output/boundaries.geojson").read_bytes())
        (root/"stain/tiles.json").write_text(json.dumps([{"box":[0,0,256,384]}]))
        partial = P.Result(root,{**result.meta,"progress":{"resolved":False,"done":1,"revision":2}})
        assert np.array_equal(mask,partial.masks([0,0,128,128]))
        try:
            partial.validate([[128,0,256,128]])
            raise AssertionError("unfinished coverage accepted")
        except ValueError:
            pass
    elif case == "proposal":
        a = result.propose(8,128)
        # Cells occupy one thin row: only six non-overlapping 128-pixel boxes fit across it.
        assert a == result.propose(8,128) and len(a) == 6 and len(set(map(tuple,a))) == 6
        assert all(not box(*b).intersection(box(*c)).area for i,b in enumerate(a) for c in a[i+1:])
        assert all(len(result.tree.query(box(*result.world(b)),predicate="intersects")) for b in a)
        result.region=Polygon([(0,0),(800,0),(0,300)])
        assert all(result.region.covers(box(*result.world(b))) for b in result.propose(8,128))
        P.image_pixels=lambda job,b,step=1: np.zeros((b[3]-b[1],b[2]-b[0],3),np.float32)[::step,::step]
        try:
            result.propose(8,128)
            raise AssertionError("background accepted")
        except ValueError as exc:
            assert "No patches" in str(exc)
    elif case == "compatibility":
        io = compatibility(folder)
        target=folder/"patch-01_seg.npy"
        saved=np.load(target,allow_pickle=True).item()
        saved["masks"][60:80,60:80]=99
        np.save(target,saved)
        assert P.inspect_patch(folder,manifest["patches"][0])["state"] == "corrected"
        compatibility(folder)
        images,labels,names,*_ = io.load_train_test_data(str(folder),mask_filter="_seg.npy")
        assert names == io.get_image_files(str(folder),"_seg.npy")
        assert len(images) == len(labels) == 2 and images[0].shape == (128,128,3)
        assert labels[0][65,65] == 99 and labels[0][25,25] == 1
        # Execute the installed CLI entry point too, substituting only the function that creates its window. sopa's
        # guide installs Cellpose without its window's packages (the README adds them: pip install 'cellpose[gui]'),
        # and the app refuses to open the window then (gui_lacking): no entry point to run without them.
        import importlib.util
        if importlib.util.find_spec("qtpy") is None:
            print("Cellpose's window packages are not installed here: its entry point is not run")
        else:
            from cellpose.gui import gui
            from sopa_companion.zarr_bridge import stain_gui
            called, launch, argv = [], gui.run, sys.argv
            try:
                gui.run = lambda image=None: called.append(image)
                sys.argv = ["stain_gui",str(folder/"patch-01.tif")]
                stain_gui.main()
                assert called == [str(folder/"patch-01.tif")]
            finally:
                gui.run, sys.argv = launch, argv
        print("Cellpose 4.2.1.1 GUI _load_seg, io.imread, get_image_files and load_train_test_data: passed")


if __name__ == "__main__":
    if len(sys.argv) in (3, 4) and sys.argv[2] == "folder":
        compatibility(Path(sys.argv[1]), sys.argv[3] if len(sys.argv) == 4 else None)
    else:
        check(Path(sys.argv[1]),sys.argv[2])
