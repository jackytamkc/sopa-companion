"""Run in the sopa env: in-memory tissue detection against the sample's own region_of_interest (read-only).

    python tissue_check.py <zarr> <inspect.json>  -> prints JSON {iou, detected_area, roi_area, seconds}
    python tissue_check.py unit                   -> geometry rules (whole cells by centre, tiles skipped)
"""
import json
import sys
import time

import shapely


def main(path, inspect):
    from sopa_companion.zarr_bridge.tissue import detect, tissue_area
    image = next(i for i in json.load(open(inspect))["images"] if i.get("to_frame"))
    started = time.monotonic()
    found = tissue_area(detect(path, image))
    seconds = time.monotonic() - started
    # The region exactly as the app reads any shapes element (its own transform into the working frame).
    from sopa_companion.zarr_bridge import helper
    info = json.load(open(inspect))
    roi = helper._aligned_shapes(helper._read(path, ("points", "shapes")), "region_of_interest", info["points_key"])
    roi = shapely.union_all(list(roi.geometry))
    level = image["levels"][0]
    from sopa_companion.engine.zarr_samples import ZarrSamples
    ox, oy, sx, sy = ZarrSamples._geometry(image, level)
    bounds = shapely.box(ox, oy, ox + level["shape"][-1] * sx, oy + level["shape"][-2] * sy)
    roi, found = roi.intersection(bounds), found.intersection(bounds)
    inter, union = found.intersection(roi).area, found.union(roi).area
    print(json.dumps({"iou": inter / union, "detected_area": found.area, "roi_area": roi.area, "seconds": seconds}))


def unit():
    from shapely.geometry import Polygon, box
    from sopa_companion.zarr_bridge.tissue import keep_inside, tissue_area, touches
    area = tissue_area([[[0, 0], [10, 0], [10, 10], [0, 10]]])
    inside_edge = Polygon([(8, 4), (14, 4), (14, 6), (8, 6)])        # centroid (11, 5): outside, dropped whole
    across_edge = Polygon([(6, 4), (11, 4), (11, 6), (6, 6)])        # centroid (8.5, 5): inside, kept uncut
    outside = box(20, 20, 22, 22)
    kept = keep_inside([inside_edge, across_edge, outside], area)
    assert kept == [across_edge] and kept[0].area == across_edge.area
    assert keep_inside([outside], None) == [outside] and keep_inside([], area) == []
    area = tissue_area([box(0, 0, 10, 10), box(5, 5, 20, 20)])
    assert touches(area, [15, 15, 30, 30]) and not touches(area, [25, 25, 30, 30]) and touches(None, [99, 99, 100, 100])
    assert tissue_area([]) is None and tissue_area([Polygon()]) is None
    print(json.dumps({"unit": "ok"}))


if __name__ == "__main__":
    unit() if sys.argv[1:] == ["unit"] else main(*sys.argv[1:3])
