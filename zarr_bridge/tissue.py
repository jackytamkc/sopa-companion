"""Tissue for stain runs and training patches: sopa's `region_of_interest`, or sopa's own tissue detection in memory.

Every project sample has `region_of_interest` (from `sopa.segmentation.tissue`); the engine reads it through the
shapes export like any other shapes element. A sample without one gets the same detection here, on the image's lowest
pyramid level, using sopa's TissueSegmentation (its defaults, expand_radius 5 %, and its own choice of mode: "saturation"
for an RGB picture such as H&E, whose glass is its brightest part, else "staining") — in memory only, so nothing is
ever written into the user's zarr. Runs in the sopa environment.
"""
import numpy as np

# sopa.segmentation.tissue defaults (2.2.10).
DEFAULTS = dict(blur_kernel_size=5, open_kernel_size=5, close_kernel_size=5, drop_threshold=.01,
                clip_parameters=(.9, 5), allow_holes=True)
EXPAND_RADIUS_RATIO = .05
# The way tissue is found, in the engine's cache stamp (engine/zarr_samples.py tissue): a new way gets a new number, so
# nothing found the old way is used again (2: "saturation" for an 8-bit RGB picture; before, H&E gave no tissue).
DETECTION_VERSION = 2


def mode(data, channel=None) -> str:
    """sopa's tissue mode for an image (sopa.segmentation.tissue: "saturation" for 3 channels, "staining" otherwise,
    and always with a channel given), saturation only for 8-bit pictures, which its HSV conversion requires: on H&E,
    staining's brightest-channel rule finds the glass, not the tissue."""
    return "saturation" if channel is None and data.sizes["c"] == 3 and data.dtype == np.uint8 else "staining"


def detect(path, image, channel=None):
    """Tissue polygons in frame coordinates (µm), detected on the lowest level of `image` (the engine's image dict)."""
    import xarray as xr
    from shapely.affinity import affine_transform
    from sopa.segmentation._tissue import TissueSegmentation
    from sopa.shapes import expand_radius, to_valid_polygons
    import geopandas as gpd
    from sopa_companion.engine.zarr_samples import ZarrSamples
    from sopa_companion.zarr_bridge.stain import image_array

    level = image["levels"][-1]
    array = image_array(str(__import__("pathlib").Path(path) / level["path"]))
    planes = [ZarrSamples._plane(array, level, c, slice(None), slice(None)) for c in range(len(image["channels"]))]
    data = xr.DataArray(np.stack(planes), dims=("c", "y", "x"), coords={"c": list(image["channels"])})
    if not np.isfinite(data.values).all() or np.ptp(data.max(dim="c").values) == 0:
        return []                                   # a blank image has no tissue to find
    try:
        found = TissueSegmentation(image=data, channel=channel, **DEFAULTS).get_polygons(mode(data, channel))
    except (ValueError, ZeroDivisionError):         # e.g. Otsu on a single intensity after clipping
        return []
    found = found[~found.geometry.is_empty]
    if not len(found):
        return []
    found = expand_radius(found, EXPAND_RADIUS_RATIO)
    found = to_valid_polygons(found.explode(index_parts=False, ignore_index=True), simple_polygon=False)
    ox, oy, sx, sy = ZarrSamples._geometry(image, level)
    # sopa contours put pixel centres at integer coordinates; image origins describe pixel corners.
    return [affine_transform(p, [sx, 0, 0, sy, ox + .5 * sx, oy + .5 * sy])
            for p in gpd.GeoSeries(found.geometry) if not p.is_empty]


def tissue_area(polygons):
    """One valid geometry from tissue polygons — shapely objects, [exterior, hole, …] ring lists, or plain rings —
    or None."""
    import shapely
    from shapely.geometry import Polygon

    def polygon(p):
        if hasattr(p, "geom_type"):
            return p
        if len(p) and len(p[0]) and isinstance(p[0][0], (int, float)):
            return Polygon(p)                                  # a plain ring
        return Polygon(p[0], p[1:])                            # exterior and holes
    shapes = [polygon(p) for p in polygons]
    shapes = [shapely.make_valid(p) for p in shapes if not p.is_empty]
    if not shapes:
        return None
    area = shapely.union_all(shapes)
    return None if area.is_empty or area.area <= 0 else area


def keep_inside(polygons, area):
    """Cells whose centroid lies in the tissue (whole cells kept, never cut at the tissue edge), as sopa keeps cells
    of its region of interest."""
    import shapely
    if area is None or not len(polygons):
        return list(polygons)
    shapely.prepare(area)
    centres = shapely.centroid(np.asarray(polygons, dtype=object))
    # On the edge counts as inside, as for removed areas (engine/stain_exclusions.inside).
    inside = shapely.intersects_xy(area, shapely.get_x(centres), shapely.get_y(centres))
    return [p for p, keep in zip(polygons, inside) if keep]


def touches(area, box):
    """Whether a frame-coordinate box [x0, y0, x1, y1] touches the tissue (tiles that do not are skipped)."""
    from shapely.geometry import box as make_box
    return area is None or area.intersects(make_box(*box))
