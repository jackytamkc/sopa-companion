"""Synthetic SpatialData for the tests, run in the sopa environment (spatialdata + sopa):

    python make_spatialdata.py sample <path>   a 300 x 200 µm sample: transcripts, cell + nucleus shapes, 2-channel image
    python make_spatialdata.py output <path>   a proseg-output.zarr as proseg 3 writes it (table + cell_boundaries)
    python make_spatialdata.py crop <path>     a stand-in for the harness's crop of a Xenium sample (make_mini_harness.py)
"""
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")


def sample(path, pyramid=False):
    import dask.dataframe as dd
    import geopandas as gpd
    import pandas as pd
    import shapely
    import spatialdata as sd
    from spatialdata.models import Image2DModel, PointsModel, ShapesModel
    from spatialdata.transformations import Identity, Scale

    rng = np.random.default_rng(0)
    n = 3000
    x = rng.uniform(0, 300, n).astype(np.float32)
    y = rng.uniform(0, 200, n).astype(np.float32)
    cell = (x // 40).astype(int) * 10 + (y // 40).astype(int)
    inside = rng.random(n) < .7
    frame = pd.DataFrame({
        "x": x, "y": y, "z": rng.uniform(5, 15, n).astype(np.float32),
        "feature_name": pd.Categorical([f"G{i % 7}" for i in range(n)]),
        "cell_id": np.where(inside, [f"c{c}-1" for c in cell], "UNASSIGNED"),
        "overlaps_nucleus": (rng.random(n) < .3).astype(np.uint8),
        "qv": np.full(n, 40, np.float32), "fov_name": ["A1"] * n,
        "transcript_id": np.arange(n, dtype=np.uint64), "codeword_category": ["predesigned_gene"] * n,
        "is_gene": np.ones(n, bool), "nucleus_distance": np.zeros(n, np.float32), "codeword_index": np.zeros(n, np.int32),
    })
    points = PointsModel.parse(dd.from_pandas(frame, npartitions=2), coordinates={"x": "x", "y": "y", "z": "z"},
                               feature_key="feature_name", instance_key="cell_id", transformations={"global": Identity()})
    centres = [(20 + 40 * i, 20 + 40 * j) for i in range(7) for j in range(5)]
    cells = gpd.GeoDataFrame(geometry=[shapely.box(cx - 18, cy - 18, cx + 18, cy + 18) for cx, cy in centres],
                             index=[f"cell-{k}" for k in range(len(centres))])
    nuclei = gpd.GeoDataFrame(geometry=[shapely.box(cx - 6, cy - 6, cx + 6, cy + 6) for cx, cy in centres],
                              index=[f"nucleus-{k}" for k in range(len(centres))])
    image = rng.integers(0, 1000, (2, 403, 607) if pyramid else (2, 400, 600), dtype=np.uint16)
    image = Image2DModel.parse(image, dims=("c", "y", "x"), c_coords=["DAPI", "Boundary"],
                               scale_factors=[2, 2, 2] if pyramid else [2],
                               transformations={"global": Scale([0.5, 0.5], axes=("x", "y"))},
                               chunks=(1, 256, 256))
    data = sd.SpatialData(images={"morphology_focus": image}, points={"transcripts": points},
                          shapes={"cell_boundaries": ShapesModel.parse(cells, transformations={"global": Identity()}),
                                  "nucleus_boundaries": ShapesModel.parse(nuclei, transformations={"global": Identity()})},
                          attrs={"transcripts_dataframe": "transcripts", "cell_segmentation_image": "morphology_focus",
                                 "transcript_to_cell_assignment": ["cell_id", "UNASSIGNED"]})
    data.write(path, overwrite=True)


def output(path):
    import anndata
    import geopandas as gpd
    import pandas as pd
    import shapely
    import spatialdata as sd
    from spatialdata.models import ShapesModel, TableModel
    from spatialdata.transformations import Identity

    n, genes = 6, 7
    counts = np.arange(n * genes, dtype=np.float32).reshape(n, genes) % 5
    obs = pd.DataFrame({"cell": np.arange(n), "centroid_x": np.arange(n) * 40.0 + 20, "centroid_y": 20.0,
                        "region": pd.Categorical(["cell_boundaries"] * n)}, index=[str(i) for i in range(n)])
    table = anndata.AnnData(counts, obs=obs, var=pd.DataFrame(index=[f"G{i}" for i in range(genes)]))
    table.obs["instance"] = np.arange(n)
    table = TableModel.parse(table, region="cell_boundaries", region_key="region", instance_key="instance")
    shapes = gpd.GeoDataFrame(geometry=[shapely.box(40 * i + 2, 2, 40 * i + 38, 38) for i in range(n)])
    data = sd.SpatialData(tables={"table": table},
                          shapes={"cell_boundaries": ShapesModel.parse(shapes, transformations={"global": Identity()})})
    data.write(path, overwrite=True)


def crop(path):
    """What the measuring harness crops out of a Xenium sample, in small: tissue on glass, with sopa's region of interest
    stored as sopa stores it. A 3-channel 16-bit image, 1024 x 1024 px at 0.5 µm (512 µm across; levels down to 1/8), a
    bright disc of tissue (radius 400 px, 200 µm) on dark glass with noise; 20,000 transcripts of 12 genes in the disc,
    with Xenium's cell assignment; square cells on a grid in the disc twice (cell_boundaries, as a segmentation saved
    into the sample, and xenium_cell_boundaries), smaller squares for xenium_nucleus_boundaries, and a table of each
    cell's whole counts for each cell layer; region_of_interest from sopa's own tissue segmentation on the lowest level
    (sopa.segmentation.tissue, staining mode, as zarr_bridge/tissue.py detects it in memory)."""
    import anndata
    import dask.dataframe as dd
    import geopandas as gpd
    import pandas as pd
    import shapely
    import spatialdata as sd
    from spatialdata.models import Image2DModel, PointsModel, ShapesModel, TableModel
    from spatialdata.transformations import Identity, Scale
    import sopa

    rng = np.random.default_rng(0)
    centre, radius = 256.0, 200.0                       # µm: the disc in the 512 x 512 µm field
    yy, xx = np.mgrid[0:1024, 0:1024]
    disc = (xx - 512) ** 2 + (yy - 512) ** 2 <= 400 ** 2
    planes = [np.where(disc, level, 40) + rng.normal(0, 25, disc.shape) for level in (1800, 900, 600)]
    image = Image2DModel.parse(np.clip(planes, 0, 65535).astype(np.uint16), dims=("c", "y", "x"),
                               c_coords=["DAPI", "Boundary", "Interior"], scale_factors=[2, 2, 2],
                               transformations={"global": Scale([0.5, 0.5], axes=("x", "y"))}, chunks=(1, 256, 256))

    # Cells: squares of 16 µm (Xenium's 18 µm) every 24 µm, wholly inside the disc; nuclei of 8 µm at their centres.
    steps = np.arange(centre - radius, centre + radius + 1, 24.0)
    centres = [(x, y) for x in steps for y in steps if np.hypot(x - centre, y - centre) + 13 <= radius]

    def squares(half):
        return gpd.GeoDataFrame(geometry=[shapely.box(x - half, y - half, x + half, y + half) for x, y in centres])

    cells, xenium_cells, nuclei = squares(8), squares(9), squares(4)

    n, genes = 20000, [f"G{i}" for i in range(12)]
    angle, distance = rng.uniform(0, 2 * np.pi, n), radius * np.sqrt(rng.uniform(0, 1, n))
    x = (centre + distance * np.cos(angle)).astype(np.float32)
    y = (centre + distance * np.sin(angle)).astype(np.float32)
    points = gpd.GeoSeries(gpd.points_from_xy(x, y))

    def owner(shapes):
        """Each transcript's cell (its row in `shapes`), -1 outside every one."""
        found = gpd.sjoin(gpd.GeoDataFrame(geometry=points), shapes, predicate="within", how="left")["index_right"]
        return found[~found.index.duplicated()].fillna(-1).astype(int).to_numpy()

    in_cell, in_nucleus = owner(xenium_cells), owner(nuclei)
    gene = rng.integers(0, len(genes), n)
    frame = pd.DataFrame({
        "x": x, "y": y, "z": rng.uniform(5, 15, n).astype(np.float32),
        "feature_name": pd.Categorical([genes[g] for g in gene], categories=genes),
        "cell_id": np.where(in_cell >= 0, [f"c{k}-1" for k in in_cell], "UNASSIGNED"),
        "overlaps_nucleus": (in_nucleus >= 0).astype(np.uint8),
        "qv": np.full(n, 40, np.float32), "transcript_id": np.arange(n, dtype=np.uint64),
    })
    transcripts = PointsModel.parse(dd.from_pandas(frame, npartitions=2), coordinates={"x": "x", "y": "y", "z": "z"},
                                    feature_key="feature_name", instance_key="cell_id",
                                    transformations={"global": Identity()})

    def table(shapes, region):
        """Each cell's whole counts of each gene, annotating the shapes `region` (by their rows)."""
        inside = owner(shapes)
        counts = np.zeros((len(shapes), len(genes)), np.float32)
        np.add.at(counts, (inside[inside >= 0], gene[inside >= 0]), 1)
        obs = pd.DataFrame({"region": pd.Categorical([region] * len(shapes)), "cell": np.arange(len(shapes))},
                           index=[str(k) for k in range(len(shapes))])
        return TableModel.parse(anndata.AnnData(counts, obs=obs, var=pd.DataFrame(index=genes)), region=region,
                                region_key="region", instance_key="cell")

    data = sd.SpatialData(
        images={"morphology_focus": image}, points={"transcripts": transcripts},
        shapes={"cell_boundaries": ShapesModel.parse(cells, transformations={"global": Identity()}),
                "xenium_cell_boundaries": ShapesModel.parse(xenium_cells, transformations={"global": Identity()}),
                "xenium_nucleus_boundaries": ShapesModel.parse(nuclei, transformations={"global": Identity()})},
        tables={"table": table(cells, "cell_boundaries"), "xenium_table": table(xenium_cells, "xenium_cell_boundaries")},
        attrs={"transcripts_dataframe": "transcripts", "cell_segmentation_image": "morphology_focus",
               "transcript_to_cell_assignment": ["cell_id", "UNASSIGNED"]})
    # sopa's own region of interest, kept in the image's frame on its lowest level, as sopa writes it into a sample.
    sopa.segmentation.tissue(data, image_key="morphology_focus", level=-1, mode="staining")
    data.write(path, overwrite=True)


if __name__ == "__main__":
    {"sample": sample, "pyramid": lambda path: sample(path, pyramid=True), "output": output,
     "crop": crop}[sys.argv[1]](sys.argv[2])
