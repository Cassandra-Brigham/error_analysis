from __future__ import annotations

# Standard library
import os
import re
import json
import time
import uuid
import tempfile
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# Third-party
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import rasterio
from rasterio.mask import mask
from rasterio.fill import fillnodata
import rioxarray as rio
from osgeo import gdal
import geopandas as gpd
from shapely.geometry import shape, box, mapping, Polygon, MultiPolygon
from shapely.ops import unary_union, transform
from pyproj import Proj, Transformer, CRS
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info
from ipyleaflet import Map, GeomanDrawControl, GeoJSON, LegendControl, basemaps, ScaleControl
from scipy.interpolate import griddata, Rbf
from datetime import date, timedelta



# Optional PDAL
try:
    import pdal  # noqa: F401
    _PDAL_AVAILABLE = True
except ImportError:
    _PDAL_AVAILABLE = False

# GDAL config
gdal.UseExceptions()

# ----------------------------------------------------------------------
# Raster helpers
# ----------------------------------------------------------------------
@dataclass
class Raster:
    """Lazy wrapper around a raster file with convenience plotting."""

    path: Path | str
    _data: rio.DataArray | None = None

    def __post_init__(self):
        """Initialize the raster wrapper by validating the path.

        This method casts the provided ``path`` to a ``Path`` object and
        verifies that the file exists on disk.  A :class:`FileNotFoundError`
        is raised if the path does not point to an existing file.
        """
        self.path = Path(self.path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)

    # ---- data access -------------------------------------------------
    @property
    def data(self):
        """Return the raster data as a :class:`rioxarray.DataArray`.

        The raster contents are loaded lazily on first access and cached in
        the ``_data`` attribute.  Subsequent calls return the cached
        DataArray without re-reading from disk.
        """
        if self._data is None:
            self._data = rio.open_rasterio(self.path, masked=True)
        return self._data

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Return the (band, row, column) dimensions of the raster."""
        return self.data.shape

    @property
    def crs(self):
        """Return the coordinate reference system of the raster.

        This wraps the ``rioxarray`` CRS accessor to provide easy access
        to the underlying CRS object.
        """
        return self.data.rio.crs

    # -------------------------- utilities ----------------------------
    def reproject_to(self, target: "Raster") -> "Raster":
        """Reproject this raster to match the grid and CRS of another raster.

        Parameters
        ----------
        target : Raster
            A reference raster whose CRS and grid spacing should be matched.

        Returns
        -------
        Raster
            A new :class:`Raster` instance pointing to the reprojected file on
            disk.  The original raster on disk is not modified.
        """
        out = self.path.with_name(f"{self.path.stem}_reproj_{uuid.uuid4().hex[:6]}.tif")
        self.data.rio.reproject_match(target.data).rio.to_raster(out)
        return Raster(out)

    # --------------------------- plotting ----------------------------
    def plot(self, *, ax=None, cmap="viridis", vmin=None, vmax=None, title=None, **imshow_kw):
        """Display the raster as an image using Matplotlib.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes on which to draw the image.  If ``None``, a new figure and
            axes are created.
        cmap : str, default 'viridis'
            The matplotlib colormap name used to map data values to colors.
        vmin, vmax : float, optional
            Color scale limits.  If omitted, the full range of the data is
            used.
        title : str, optional
            Title for the subplot.  Omitted if ``None``.
        **imshow_kw : dict
            Additional keyword arguments passed to :func:`matplotlib.pyplot.imshow`.

        Returns
        -------
        matplotlib.axes.Axes
            The axes object containing the image plot.
        """
        if ax is None:
            _, ax = plt.subplots()
        arr = self.data.squeeze().values
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, **imshow_kw)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks([]); ax.set_yticks([])
        if title:
            ax.set_title(title)
        return ax
    
@dataclass
class RasterPair:
    """Pair two rasters, ensure alignment, and provide plotting."""

    raster1: Raster | str
    raster2: Raster | str
    _aligned: bool = False

    def __post_init__(self):
        """Coerce input rasters to :class:`Raster` instances.

        The constructor accepts either existing :class:`Raster` objects or
        file paths.  During initialization these are wrapped into
        :class:`Raster` objects so that subsequent methods can assume
        uniform behaviour.
        """
        self.raster1 = Raster(self.raster1) if not isinstance(self.raster1, Raster) else self.raster1
        self.raster2 = Raster(self.raster2) if not isinstance(self.raster2, Raster) else self.raster2

    # ---------------------- alignment checker ------------------------
    def _align(self):
        """Ensure the two rasters have matching grids and CRS.

        This helper method lazily reprojects the larger raster to match
        the smaller one if the underlying coordinate systems or
        transformations differ.  It caches the result to avoid repeated
        reproject operations when multiple properties are accessed.
        """
        if self._aligned:
            return
        def key(r: Raster):
            return (r.crs, r.data.rio.transform(), r.shape)
        if key(self.raster1) != key(self.raster2):
            larger, smaller = ((self.raster1, self.raster2) if np.prod(self.raster1.shape) > np.prod(self.raster2.shape) else (self.raster2, self.raster1))
            warnings.warn("Raster grid mismatch – reprojecting the larger raster to match the smaller.")
            reproj = larger.reproject_to(smaller)
            if larger is self.raster1:
                self.raster1 = reproj
            else:
                self.raster2 = reproj
        self._aligned = True

    raster1_data = property(lambda self: (self._align(), self.raster1.data)[1])
    raster2_data = property(lambda self: (self._align(), self.raster2.data)[1])

       # --------------------------- plotting ----------------------------
    def plot_pair(
        self,
        *,
        overlay: Raster | None = None,
        overlay_alpha: float = 0.5,
        vmin=None, 
        vmax=None,
        titles=("Raster 1", "Raster 2"),
        base_cmap="viridis",
        overlay_cmap="magma",
        legend: str | None = None,
    ) -> plt.Figure:
        """Plot two rasters side-by-side; optionally overlay a third raster
        and draw a single shared colorbar legend."""
        self._align()

        fig, axs = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

        # --- Plot base rasters manually, capture the first image for legend ---
        arr1 = self.raster1.data.squeeze().values
        arr2 = self.raster2.data.squeeze().values

        im = axs[0].imshow(arr1, cmap=base_cmap, vmin=vmin, vmax=vmax)
        axs[0].set_title(titles[0])
        axs[0].axis("off")

        axs[1].imshow(arr2, cmap=base_cmap, vmin=im.norm.vmin, vmax=im.norm.vmax)
        axs[1].set_title(titles[1])
        axs[1].axis("off")

        # --- Overlay, if present ---
        if overlay is not None:
            for base, ax in zip((self.raster1, self.raster2), axs):
                ov = overlay
                if ov.crs != base.crs or ov.shape[1:] != base.shape[1:]:
                    ov = ov.reproject_to(base)
                ax.imshow(ov.data.squeeze().values,
                          cmap=overlay_cmap,
                          alpha=overlay_alpha,
                          vmin=None, vmax=None)

        # --- Single shared colorbar ---
        if legend:
            cbar = fig.colorbar(im, ax=axs, fraction=0.046, pad=0.04)
            cbar.set_label(legend)

        return fig

# ----------------------------------------------------------------------
# DataAccess
# ----------------------------------------------------------------------
class DataAccess:
    """Wrapper for interactive or programmatic AOI definition."""

    # --------------------- static helpers ------------------------------
    @staticmethod
    def _coords_to_wkt(coords):
        """Convert a nested coordinate list into a WKT-compatible string.

        Parameters
        ----------
        coords : iterable
            A sequence of linear rings (each itself a sequence of ``(x, y)`` pairs).

        Returns
        -------
        str
            A comma-separated representation of coordinates suitable for
            inclusion in OpenTopography WKT polygon strings.
        """
        return ", ".join(
            ", ".join(f"{x}, {y}" for x, y in ring) for ring in coords
        )

    @classmethod
    def geojson_to_OTwkt(cls, gj):
        """Convert a GeoJSON polygon into an OpenTopography WKT string.

        Parameters
        ----------
        gj : dict
            A GeoJSON dictionary representing a polygon geometry.

        Returns
        -------
        str
            A comma-separated WKT representation of the polygon's coordinates.

        Raises
        ------
        ValueError
            If the provided GeoJSON does not represent a polygon.
        """
        if gj["type"] != "Polygon":
            raise ValueError("Input must be a Polygon GeoJSON")
        return cls._coords_to_wkt(gj["coordinates"])

    # --------------------- AOI via map draw ----------------------------
    def init_ot_catalog_map(
        self,
        center=(39.8283, -98.5795),
        zoom=3,
        layers=(
            ("3DEP", "https://raw.githubusercontent.com/OpenTopography/Data_Catalog_Spatial_Boundaries/main/usgs_3dep_boundaries.geojson", "#228B22"),
            ("NOAA", "https://raw.githubusercontent.com/OpenTopography/Data_Catalog_Spatial_Boundaries/main/noaa_coastal_lidar_boundaries.geojson", "#0000CD"),
            ("OpenTopography", "https://raw.githubusercontent.com/OpenTopography/Data_Catalog_Spatial_Boundaries/main/OT_PC_boundaries.geojson", "#fca45d"),
        ),
    ):
        """Return ipyleaflet Map with draw control and catalog layers."""
        self.bounds: dict[str, Any] = {}
        self.polygon: dict[str, Any] = {}

        def _on_draw(control, action, geo_json):
            feats = geo_json if isinstance(geo_json, list) else [geo_json]
            shapes = [shape(f["geometry"]) for f in feats]
            wkt_list = [self.geojson_to_OTwkt(f["geometry"]) for f in feats]
            merged = unary_union(shapes)
            minx, miny, maxx, maxy = merged.bounds
            self.bounds.update(dict(south=miny, west=minx, north=maxy, east=maxx, polygon_wkt=wkt_list))
            self.polygon.update(dict(merged_polygon=merged, all_polys=shapes))
            print("AOI bounds:", self.bounds)

        m = Map(center=center, zoom=zoom, basemap=basemaps.Esri.WorldTopoMap)
        dc = GeomanDrawControl(rectangle={"pathOptions": {"color": "#fca45d", "fillColor": "#fca45d"}},
                               polygon={"pathOptions": {"color": "#6be5c3", "fillColor": "#6be5c3"}})
        dc.polyline = dc.circlemarker = {}
        dc.on_draw(_on_draw)
        m.add_control(dc)

        for name, url, color in layers:
            gj = GeoJSON(data=requests.get(url).json(), name=name, style={"color": color})
            m.add_layer(gj)
        m.add_control(LegendControl({n: c for n, _, c in layers}, name="Legend"))
        m.add_control(ScaleControl(position="bottomleft", metric=True, imperial=False))
        return m

    # --------------------- AOI via manual bounds -----------------------
    def define_bounds_manual(self, south, north, west, east):
        """Define the area of interest (AOI) from numeric latitude/longitude bounds.

        Parameters
        ----------
        south, north, west, east : float
            Geographic coordinates delimiting the rectangular AOI in degrees.

        Returns
        -------
        dict
            A dictionary containing south, north, west, east and polygon WKT entries for the AOI.
        """
        poly = box(west, south, east, north)
        self.bounds = dict(south=south, north=north, west=west, east=east,
                           polygon_wkt=[self.geojson_to_OTwkt(mapping(poly))])
        self.polygon = dict(merged_polygon=poly, all_polys=[poly])
        return self.bounds

    # --------------------- AOI via uploaded vector ---------------------
    def define_bounds_from_file(self, vector_path: str, target_crs="EPSG:4326"):
        """Define the AOI using a vector file such as a shapefile or GeoJSON.

        The geometries in the provided file are merged into a single polygon
        in the target CRS.  The AOI bounds and WKT strings are stored on
        this ``DataAccess`` instance for later use in API queries.

        Parameters
        ----------
        vector_path : str
            Path to a vector file readable by GeoPandas.
        target_crs : str, default ``'EPSG:4326'``
            Desired output CRS for the AOI.

        Returns
        -------
        dict
            A dictionary of bounds (south, west, north, east) and polygon WKT strings.
        """
        gdf = gpd.read_file(vector_path)
        if gdf.empty:
            raise ValueError("No geometries in file")
        if gdf.crs is None:
            raise ValueError("Input CRS undefined")
        if gdf.crs.to_string() != target_crs:
            gdf = gdf.to_crs(target_crs)
        merged = unary_union(gdf.geometry)
        minx, miny, maxx, maxy = merged.bounds
        wkts = [self.geojson_to_OTwkt(mapping(geom)) for geom in gdf.geometry]
        self.bounds = dict(south=miny, west=minx, north=maxy, east=maxx, polygon_wkt=wkts)
        self.polygon = dict(merged_polygon=merged, all_polys=list(gdf.geometry))
        return self.bounds

# ----------------------------------------------------------------------
# OpenTopographyQuery
# ----------------------------------------------------------------------
class OpenTopographyQuery:
    def __init__(self, data_access: DataAccess):
        """Create a new query object tied to a particular AOI.

        Parameters
        ----------
        data_access : DataAccess
            An instance of :class:`DataAccess` holding the area of interest
            bounds and polygons.  This object is used to supply spatial
            parameters when querying the OpenTopography catalog.
        """
        self.da = data_access
        self.catalog_df: pd.DataFrame | None = None

    @staticmethod
    def _clean(name: str) -> str:
        """Sanitize dataset names by replacing non-word characters with underscores."""
        return re.sub(r"[^\w]+", "_", name)

    def query_catalog(
        self,
        product_format="PointCloud",
        include_federated=True,
        detail=False,
        save_as="results.json",
        url="https://portal.opentopography.org/API/otCatalog",
    ) -> pd.DataFrame:
        """Query the OpenTopography catalog for datasets within the current AOI.

        Parameters
        ----------
        product_format : str, default ``'PointCloud'``
            Desired data product format (e.g. ``'PointCloud'``).
        include_federated : bool, default ``True``
            Whether to include federated datasets in the search.
        detail : bool, default ``False``
            Request detailed metadata in the API response.
        save_as : str or None, default ``'results.json'``
            Optional filename to save the raw JSON response.  If ``None``,
            the response is not saved.
        url : str, default OpenTopography catalog API endpoint
            The API base URL to query.

        Returns
        -------
        pandas.DataFrame
            A dataframe listing datasets found within the AOI, sorted by
            start date.  The dataframe is also stored on this object as
            ``catalog_df`` for reuse.

        Raises
        ------
        ValueError
            If the area of interest has not been defined on the associated
            :class:`DataAccess` object.
        """
        # ... (no change to the request logic) ...
        if not getattr(self.da, "bounds", None):
            raise ValueError("AOI not defined")
        params = dict(
            productFormat=product_format,
            detail=str(detail).lower(),
            outputFormat="json",
            include_federated=str(include_federated).lower(),
        )
        if self.da.bounds.get("polygon_wkt"):
            params["polygon"] = self.da.bounds["polygon_wkt"]
        else:
            params.update(dict(minx=self.da.bounds["west"], miny=self.da.bounds["south"],
                        maxx=self.da.bounds["east"], maxy=self.da.bounds["north"]))
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        if save_as:
            Path(save_as).write_bytes(r.content)
            
        data = r.json()
        rows = []
        for ds in data["Datasets"]:
            meta = ds["Dataset"]
            
            start_date_str = None
            end_date_str = None
            coverage = meta.get("temporalCoverage")

            if isinstance(coverage, str):
                # Handle the string format, e.g., "2018-01-13 / 2018-06-11"
                if "/" in coverage:
                    parts = coverage.split("/")
                    start_date_str = parts[0].strip()
                    end_date_str = parts[1].strip()
                else:
                    # Handle a single date string, e.g., "2002-09-18"
                    start_date_str = coverage.strip()
                    end_date_str = coverage.strip()
            
            elif isinstance(coverage, dict):
                # Handle the dictionary format for robustness
                start_date_str = coverage.get("startDate")
                end_date_str = coverage.get("endDate")

            rows.append(
                {
                    "Name":           meta["name"],
                    "ID type":        meta["identifier"]["propertyID"],
                    "Data Source":    "usgs" if "USGS" in meta["identifier"]["propertyID"] or "usgs" in meta["identifier"]["propertyID"] else
                                    "noaa" if "NOAA" in meta["identifier"]["propertyID"] or "noaa" in meta["identifier"]["propertyID"] else "ot",
                    "Property ID":    meta["identifier"]["value"],
                    "Horizontal EPSG":
                        next((p["value"] for p in meta["spatialCoverage"]["additionalProperty"]
                            if p["name"] == "EPSG (Horizontal)"), None),
                    "Vertical Coordinates":
                        next((p["value"] for p in meta["spatialCoverage"]["additionalProperty"]
                            if p["name"] == "Vertical Coordinates"), None),
                    "Clean Name":     self._clean(meta["name"]),
                    "StartDate":      pd.to_datetime(start_date_str).date() if start_date_str else None,
                    "EndDate":        pd.to_datetime(end_date_str).date() if end_date_str else None,
                }
            )
        
        _catalog_df =  pd.DataFrame(rows)
        catalog_df = _catalog_df.sort_values(by="StartDate").reset_index(drop=True)
        self.catalog_df = catalog_df
        
        return self.catalog_df
    
    
        

    # shorthand to select compare / reference rows by DataFrame index
    def pick(self, idx_compare: int, idx_reference: int):
        """Select the compare and reference datasets by row index.

        After querying the catalog, datasets are presented in a dataframe.
        This method picks a pair of rows corresponding to the compare and
        reference datasets.  It stores various fields (names, CRS
        definitions, etc.) as attributes on the query object for later
        processing and prints the mid-point epoch for each dataset if
        available.

        Parameters
        ----------
        idx_compare : int
            Row index of the dataset to treat as the ``compare`` survey.
        idx_reference : int
            Row index of the dataset to treat as the ``reference`` survey.

        Returns
        -------
        tuple of pandas.Series
            The selected compare and reference rows from ``catalog_df``.
        """
        df = self.catalog_df
        self.compare = df.iloc[idx_compare]
        self.reference = df.iloc[idx_reference]
        self.compare_name = self.catalog_df["Name"].iloc[idx_compare]
        self.compare_data_source = self.catalog_df["Data Source"].iloc[idx_compare]
        self.compare_property_id = self.catalog_df["Property ID"].iloc[idx_compare]
        self.compare_horizontal_crs = self.catalog_df["Horizontal EPSG"].iloc[idx_compare]
        self.compare_vertical_crs = self.catalog_df["Vertical Coordinates"].iloc[idx_compare]
        self.compare_clean_name = self.catalog_df["Clean Name"].iloc[idx_compare]
        self.reference_name = self.catalog_df["Name"].iloc[idx_reference]
        self.reference_data_source = self.catalog_df["Data Source"].iloc[idx_reference]
        self.reference_property_id = self.catalog_df["Property ID"].iloc[idx_reference]
        self.reference_horizontal_crs = self.catalog_df["Horizontal EPSG"].iloc[idx_reference]
        self.reference_vertical_crs = self.catalog_df["Vertical Coordinates"].iloc[idx_reference]
        self.reference_clean_name = self.catalog_df["Clean Name"].iloc[idx_reference]
        
        compare_start = df["StartDate"].iloc[idx_compare]
        compare_end = df["EndDate"].iloc[idx_compare]
        if compare_start and compare_end:
            self.compare_epoch = compare_start + (compare_end - compare_start) / 2
        else:
            self.compare_epoch = None

        ref_start = df["StartDate"].iloc[idx_reference]
        ref_end = df["EndDate"].iloc[idx_reference]
        if ref_start and ref_end:
            self.reference_epoch = ref_start + (ref_end - ref_start) / 2
        else:
            self.reference_epoch = None
        
        if self.compare["Vertical Coordinates"] != self.reference["Vertical Coordinates"]:
            print("⚠️  Vertical CRSs differ between datasets")
        
        print(f"🔹 Compare Epoch: {self.compare_epoch}")
        print(f"🔹 Reference Epoch: {self.reference_epoch}")
        
        return self.compare, self.reference

# ------------------------------------------------------------------------------------------
# Download data and make DEMs
# ------------------------------------------------------------------------------------------
class GetDEMs:
    """Generate DEMs from local LAZ/point files or AWS EPT sources."""

    def __init__(self, data_access, ot_query):
        self.da = data_access
        self.ot = ot_query
                          
    # -----------------Raster gap‑fill utility--------------------------
    @staticmethod
    def fill_no_data(input_file, output_file, *, method="idw", nodata=-9999, max_dist=100, smooth_iter=0):
        """
        Fill missing values in a raster by interpolation and write to a new file.

        This utility reads the first band of ``input_file``, locates pixels equal
        to the supplied ``nodata`` value, and replaces those gaps using either
        GDAL's built-in inverse distance weighted (IDW) fill routine or a
        SciPy-based interpolator.  For SciPy methods, valid pixel coordinates
        are used to estimate missing values with nearest-neighbour, linear,
        cubic, or thin‑plate spline interpolation.  The resulting filled array
        is written to ``output_file`` with the same geotransform, projection
        and ``nodata`` metadata as the source raster.

        Parameters
        ----------
        input_file : str
            Path to the input raster file with gaps.
        output_file : str
            Destination filename for the filled raster.
        method : {"idw", "nearest", "linear", "cubic", "spline"}, optional
            Algorithm used to fill the gaps.  ``idw`` invokes GDAL's
            ``FillNodata`` function; other options use SciPy's
            :func:`scipy.interpolate.griddata` or radial basis function (Rbf)
            interpolation.  Default is ``"idw"``.
        nodata : float or int, default -9999
            Pixel value representing missing data in the input raster.
        max_dist : int, default 100
            Maximum search distance in pixels for the IDW fill method.
        smooth_iter : int, default 0
            Number of smoothing iterations applied by GDAL's ``FillNodata``.

        Returns
        -------
        None
            The function writes the filled raster to ``output_file`` and has
            no return value.
        """
        ds = gdal.Open(input_file, gdal.GA_ReadOnly)
        band = ds.GetRasterBand(1)
        arr = band.ReadAsArray()
        mask = arr == nodata

        def _interpolate(other):
            valid = np.where(~mask)
            nod = np.where(mask)
            coords = np.column_stack(valid)
            vals = arr[valid]
            if other == "nearest":
                return griddata(coords, vals, np.column_stack(nod), method="nearest")
            if other == "linear":
                return griddata(coords, vals, np.column_stack(nod), method="linear")
            if other == "cubic":
                return griddata(coords, vals, np.column_stack(nod), method="cubic")
            if other == "spline":
                rbf = Rbf(coords[:, 0], coords[:, 1], vals, function="thin_plate")
                return rbf(nod[0], nod[1])
            raise ValueError("Unknown method")

        if method == "idw":
            mem = gdal.GetDriverByName("MEM").CreateCopy("", ds, 0)
            gdal.FillNodata(mem.GetRasterBand(1), None, max_dist, smooth_iter)
            filled = mem.GetRasterBand(1).ReadAsArray()
        else:
            filled = arr.copy()
            filled_vals = _interpolate(method)
            filled[np.where(mask)] = filled_vals

        drv = gdal.GetDriverByName("GTiff")
        out = drv.Create(output_file, ds.RasterXSize, ds.RasterYSize, 1, gdal.GDT_Float32)
        out.SetGeoTransform(ds.GetGeoTransform())
        out.SetProjection(ds.GetProjection())
        out.GetRasterBand(1).WriteArray(filled)
        out.GetRasterBand(1).SetNoDataValue(nodata)
        out.FlushCache()

                    
    # --------------------Internal PDAL helpers-------------------------
    @staticmethod
    def _writer_gdal(filename, *, grid_method="idw", res=1.0, driver="GTiff"):
        """
        Construct a PDAL GDAL writer stage for gridding point cloud data.

        This helper builds a dictionary describing a ``writers.gdal`` stage
        compatible with PDAL pipelines.  It specifies output file name,
        gridding method (e.g. inverse distance weighted or maximum), pixel
        resolution and radius, nodata value, compression, and tiling options.

        Parameters
        ----------
        filename : str
            The destination filename for the output raster (including
            extension).
        grid_method : str, default 'idw'
            The gridding algorithm used to convert points to raster.  Typical
            values include ``'idw'`` for inverse distance weighting and
            ``'max'`` for maximum value gridding.
        res : float, default 1.0
            Output pixel size in map units (meters).
        driver : str, default 'GTiff'
            GDAL driver used to write the raster.  ``'GTiff'`` yields a
            GeoTIFF.

        Returns
        -------
        dict
            A dictionary suitable for inclusion in a PDAL pipeline,
            representing a GDAL writer stage.
        """
        return {
            "type": "writers.gdal",
            "filename": filename,
            "gdaldriver": driver,
            "nodata": -9999,
            "output_type": grid_method,
            "resolution": float(res),
            "radius": 2 * float(res),
            "gdalopts": "COMPRESS=LZW,TILED=YES,blockxsize=256,blockysize=256,COPY_SRC_OVERVIEWS=YES",
        }

    @staticmethod
    def _writer_las(name, ext):
        """
        Construct a PDAL LAS/LAZ writer stage for saving point clouds.

        Parameters
        ----------
        name : str
            Base filename (without extension) for the output point cloud.
        ext : {'las', 'laz'}
            Desired output format.  ``'las'`` produces an uncompressed LAS
            file while ``'laz'`` triggers LASzip compression.

        Returns
        -------
        dict
            A dictionary describing a ``writers.las`` stage for PDAL.

        Raises
        ------
        ValueError
            If ``ext`` is not one of ``'las'`` or ``'laz'``.
        """
        if ext not in {"las", "laz"}:
            raise ValueError("pc_outType must be 'las' or 'laz'")
        w = {"type": "writers.las", "filename": f"{name}.{ext}"}
        if ext == "laz":
            w["compression"] = "laszip"
        return w

    # ----------------------- Local file pipelines ---------------------
    @staticmethod
    def build_pdal_pipeline_from_file(filename, extent, filterNoise=False, reclassify=False, savePointCloud=True, outCRS='EPSG:3857', 
                            pc_outName='filter_test', pc_outType='laz'):
        """
        Construct a PDAL pipeline to read, optionally filter/reclassify, and
        reproject a local point cloud.

        This method builds a list of PDAL stages beginning with reading
        LAS/LAZ data from disk and cropping it to the supplied ``extent``.
        Optional stages remove noise classes 7 and 18, apply SMRF ground
        classification, reclassify ground points, and reproject the point
        cloud into a target CRS.  Finally, the pipeline may include a
        ``writers.las`` stage to save the filtered/reprojected cloud.

        Parameters
        ----------
        filename : str
            Path to a LAS or LAZ file to be processed.
        extent : shapely.geometry.Polygon
            Polygon in the same CRS as the input point cloud used to crop
            the data.
        filterNoise : bool, default False
            Remove noise points classified as 7 (low noise) or 18 (high
            noise).
        reclassify : bool, default False
            Apply SMRF ground classification and restrict the cloud to
            ground returns.
        savePointCloud : bool, default True
            When true, append a ``writers.las`` stage to write the processed
            point cloud.
        outCRS : str, default 'EPSG:3857'
            Desired output coordinate reference system.
        pc_outName : str, default 'filter_test'
            Base filename used when saving the filtered point cloud.
        pc_outType : {'las', 'laz'}, default 'laz'
            File format for the output point cloud.

        Returns
        -------
        list of dict
            A PDAL pipeline specification representing the requested
            operations.
        """
        # Initialize the pipeline with reading and cropping stages
        pointcloud_pipeline = [
            {
                "type": "readers.las",
                "filename": filename
            },
            {
                "type": "filters.crop",
                "polygon": extent.wkt
            }
        ]
        
        # Optionally add a noise filter stage
        if filterNoise:
            pointcloud_pipeline.append({
                "type": "filters.range",
                "limits": "Classification![7:7], Classification![18:18]"
            })
        
        # Optionally add reclassification stages
        if reclassify:
            pointcloud_pipeline += [
                {"type": "filters.assign", "value": "Classification = 0"},
                {"type": "filters.smrf"},
                {"type": "filters.range", "limits": "Classification[2:2]"}
            ]
        
        # Add reprojection stage
        pointcloud_pipeline.append({
            "type": "filters.reprojection",
            "out_srs": outCRS,
        })
        
        # Optionally add a save point cloud stage
        if savePointCloud:
            if pc_outType not in ['las', 'laz']:
                raise Exception("pc_outType must be 'las' or 'laz'.")
            
            writer_stage = {
                "type": "writers.las",
                "filename": f"{pc_outName}.{pc_outType}"
            }
            if pc_outType == 'laz':
                writer_stage["compression"] = "laszip"
            
            pointcloud_pipeline.append(writer_stage)
            
        return pointcloud_pipeline
    
    def make_DEM_pipeline_from_file(self, filename, extent, dem_resolution,
                        filterNoise=True, reclassify=False, savePointCloud=False, outCRS='EPSG:3857',
                        pc_outName='filter_test', pc_outType='laz', demType='dtm', gridMethod='idw', 
                        dem_outName='dem_test', dem_outExt='tif', driver="GTiff"):
        """
        Build a PDAL pipeline to convert a local point cloud into a DEM.

        This method wraps :meth:`build_pdal_pipeline_from_file` and then
        appends stages to generate a Digital Terrain Model (DTM) or
        Digital Surface Model (DSM) from the cropped/filtered point cloud.
        Ground points are optionally selected when ``demType`` is ``'dtm'``.

        Parameters
        ----------
        filename : str
            Path to the input LAS/LAZ file.
        extent : shapely.geometry.Polygon
            Clipping polygon in the input CRS.
        dem_resolution : float
            Pixel size in map units for the output raster.
        filterNoise : bool, default True
            Remove noise classes (7 and 18) prior to gridding.
        reclassify : bool, default False
            Reclassify points using SMRF and retain only ground returns.
        savePointCloud : bool, default False
            Write the intermediate point cloud to disk.
        outCRS : str, default 'EPSG:3857'
            Output CRS for the DEM.
        pc_outName : str, default 'filter_test'
            Base filename for the intermediate point cloud.
        pc_outType : {'las', 'laz'}, default 'laz'
            Format for the intermediate point cloud.
        demType : {'dtm', 'dsm'}, default 'dtm'
            Type of DEM to produce.  ``'dtm'`` filters ground points,
            whereas ``'dsm'`` uses all returns.
        gridMethod : str, default 'idw'
            Gridding algorithm (e.g. ``'idw'`` or ``'max'``) used by
            ``writers.gdal``.
        dem_outName : str, default 'dem_test'
            Base name for the output DEM.
        dem_outExt : str, default 'tif'
            File extension for the output raster.
        driver : str, default 'GTiff'
            GDAL driver used to create the DEM.

        Returns
        -------
        dict
            A PDAL pipeline dictionary describing the steps to generate the DEM.
        """
        # Build the base point cloud pipeline using the provided parameters
        pointcloud_pipeline = self.build_pdal_pipeline_from_file(filename, extent, filterNoise, reclassify, savePointCloud, outCRS, pc_outName, pc_outType)
        
        # Prepare the base pipeline dictionary
        dem_pipeline = {
            "pipeline": pointcloud_pipeline
        }

        # Add appropriate stages based on DEM type
        if demType == 'dsm':
            # Directly add the DSM writer stage
            dem_pipeline['pipeline'].append({
                "type": "writers.gdal",
                "filename": f"{dem_outName}.{dem_outExt}",
                "gdaldriver": driver,
                "nodata": -9999,
                "output_type": gridMethod,
                "resolution": float(dem_resolution),
                "radius": 2*float(dem_resolution),
                "gdalopts": "COMPRESS=LZW,TILED=YES,blockxsize=256,blockysize=256,COPY_SRC_OVERVIEWS=YES",
                "override_srs": outCRS
            })
        
        elif demType == 'dtm':
            # Add a filter to keep only ground points
            dem_pipeline['pipeline'].append({
                "type": "filters.range",
                "limits": "Classification[2:2]"
            })

            # Add the DTM writer stage
            dem_pipeline['pipeline'].append({
                "type": "writers.gdal",
                "filename": f"{dem_outName}.{dem_outExt}",
                "gdaldriver": driver,
                "nodata": -9999,
                "output_type": gridMethod,
                "resolution": float(dem_resolution),
                "radius": 2*float(dem_resolution),
                "gdalopts": "COMPRESS=LZW,TILED=YES,blockxsize=256,blockysize=256,COPY_SRC_OVERVIEWS=YES",
                "override_srs": outCRS
            })
        else:
            raise Exception("demType must be 'dsm' or 'dtm'.")
        
        return dem_pipeline

    # ----------------------- AWS EPT pipeline helpers -----------------
    @staticmethod
    def build_aws_pdal_pipeline(extent_epsg3857, property_ids, pc_resolution, data_source, filterNoise = False,
                            reclassify = False, savePointCloud = True, outCRS = 'EPSG:3857', pc_outName = 'filter_test', 
                            pc_outType = 'laz'):
        """
        Build a PDAL pipeline for downloading and processing AWS-hosted EPT data.

        Given a list of dataset identifiers, this function constructs a
        ``readers.ept`` stage for each ID, clips the point cloud to the
        provided polygon in EPSG:3857 coordinates, and optionally filters
        noise, reclassifies ground points, reprojects to a target CRS, and
        writes the resulting point cloud to disk.  The function supports
        USGS and NOAA data sources; other providers are not currently
        accepted.

        Parameters
        ----------
        extent_epsg3857 : shapely.geometry.Polygon
            AOI polygon expressed in Web Mercator (EPSG:3857) coordinates.
        property_ids : list of str
            Identifiers for the EPT datasets to read from S3 buckets or
            NOAA's Digital Coast STAC.
        pc_resolution : float
            Sampling resolution for the ``readers.ept`` stage.  Larger values
            will subsample the point cloud more aggressively.
        data_source : {'usgs', 'noaa'}
            Indicates which repository to fetch the data from.  For USGS,
            the EPT URL is constructed directly; for NOAA, the STAC
            catalog is queried to obtain the EPT link.
        filterNoise : bool, default False
            Remove noise classes 7 and 18 prior to further processing.
        reclassify : bool, default False
            Apply SMRF classification and select only ground points.
        savePointCloud : bool, default True
            Write the reprojected point cloud to disk.
        outCRS : str, default 'EPSG:3857'
            CRS to which the point cloud should be reprojected.
        pc_outName : str, default 'filter_test'
            Base filename used when saving the output point cloud.
        pc_outType : {'las', 'laz'}, default 'laz'
            File extension for the output point cloud.  ``'laz'`` triggers
            compression via LASzip.

        Returns
        -------
        dict
            A PDAL pipeline dictionary describing the EPT read and optional
            processing stages.
        """
        readers = []
        for id in property_ids:
            if data_source == 'usgs':
                url = f"https://s3-us-west-2.amazonaws.com/usgs-lidar-public/{id}/ept.json"
            elif data_source == 'noaa':
                stac_url = f"https://noaa-nos-coastal-lidar-pds.s3.amazonaws.com/entwine/stac/DigitalCoast_mission_{id}.json"
                response = requests.get(stac_url)
                data = response.json()
                url = data['assets']['ept']['href']
            else:
                raise ValueError("Invalid dataset source. Must be 'usgs' or 'noaa'.")

            reader = {
                "type": "readers.ept",
                "filename": str(url),
                "polygon": str(extent_epsg3857),
                "requests": 3,
                "resolution": pc_resolution
            }
            readers.append(reader)
            
        pointcloud_pipeline = {
                "pipeline":
                    readers
        }
        
        if filterNoise == True:
            
            filter_stage = {
                "type":"filters.range",
                "limits":"Classification![7:7], Classification![18:18]"
            }
            
            pointcloud_pipeline['pipeline'].append(filter_stage)
        
        if reclassify == True:
            
            remove_classes_stage = {
                "type":"filters.assign",
                "value":"Classification = 0"
            }
            
            classify_ground_stage = {
                "type":"filters.smrf"
            }
            
            reclass_stage = {
                "type":"filters.range",
                "limits":"Classification[2:2]"
            }

            
            pointcloud_pipeline['pipeline'].append(remove_classes_stage)
            pointcloud_pipeline['pipeline'].append(classify_ground_stage)
            pointcloud_pipeline['pipeline'].append(reclass_stage)
            
        reprojection_stage = {
            "type":"filters.reprojection",
            "out_srs":outCRS,
        }
        
        pointcloud_pipeline['pipeline'].append(reprojection_stage)
        
        if savePointCloud == True:
            
            if pc_outType == 'las':
                savePC_stage = {
                    "type": "writers.las",
                    "filename": str(pc_outName)+'.'+ str(pc_outType),
                }
            elif pc_outType == 'laz':    
                savePC_stage = {
                    "type": "writers.las",
                    "compression": "laszip",
                    "filename": str(pc_outName)+'.'+ str(pc_outType),
                }
            else:
                raise Exception("pc_outType must be 'las' or 'laz'.")

            pointcloud_pipeline['pipeline'].append(savePC_stage)
            
        return pointcloud_pipeline
    
    def make_DEM_pipeline_aws(self, extent_epsg3857, property_ids, pc_resolution, dem_resolution, data_source = "usgs",
                        filterNoise = True, reclassify = True, savePointCloud = False, outCRS = 'EPSG:3857',
                        pc_outName = 'filter_test', pc_outType = 'laz', demType = 'dtm', gridMethod = 'idw',
                        dem_outName = 'dem_test', dem_outExt = 'tif', driver = "GTiff"):
        """Build a PDAL pipeline to create a DEM from AWS-hosted point cloud data.

        This method wraps :func:`build_aws_pdal_pipeline` and appends additional
        stages to generate a Digital Terrain Model (DTM) or Digital Surface
        Model (DSM) from Entwine Point Tiles.  Depending on the ``demType``
        argument, it optionally filters ground returns before gridding.

        Parameters
        ----------
        extent_epsg3857 : shapely.geometry.Polygon
            Cropping polygon in EPSG:3857 coordinates.
        property_ids : list of str
            Identifiers for the point cloud datasets to read.
        pc_resolution : float
            Resolution used when selecting points from the EPT.
        dem_resolution : float
            Output pixel size for the DEM.
        data_source : {'usgs', 'noaa'}, default 'usgs'
            Which provider the EPT data originates from.
        filterNoise : bool, default True
            Remove noise classes (7 and 18) before gridding.
        reclassify : bool, default True
            Apply ground classification via SMRF before gridding.
        savePointCloud : bool, default False
            Save the intermediate point cloud to disk.
        outCRS : str, default 'EPSG:3857'
            Target CRS for the output DEM.
        pc_outName : str, default 'filter_test'
            Base filename for the intermediate point cloud.
        pc_outType : {'las', 'laz'}, default 'laz'
            Format for the intermediate point cloud.
        demType : {'dtm', 'dsm'}, default 'dtm'
            Type of DEM to generate: ``'dtm'`` for ground-only or ``'dsm'``
            for first-return surfaces.
        gridMethod : str, default 'idw'
            Gridding algorithm to use in the GDAL writer stage.
        dem_outName : str, default 'dem_test'
            Base name for the output DEM (without extension).
        dem_outExt : str, default 'tif'
            File extension for the output DEM.
        driver : str, default 'GTiff'
            GDAL driver used to write the DEM.

        Returns
        -------
        dict
            A PDAL pipeline dictionary describing the steps to produce the DEM.
        """

        dem_pipeline = self.build_aws_pdal_pipeline(extent_epsg3857, property_ids, pc_resolution, data_source,
                                                filterNoise, reclassify, savePointCloud, outCRS, pc_outName, pc_outType)
        
        
        if demType == 'dsm':
            dem_stage = {
                    "type":"writers.gdal",
                    "filename":str(dem_outName)+ '.' + str(dem_outExt),
                    "gdaldriver":driver,
                    "nodata":-9999,
                    "output_type":gridMethod,
                    "resolution":float(dem_resolution),
                    "gdalopts":"COMPRESS=LZW,TILED=YES,blockxsize=256,blockysize=256,COPY_SRC_OVERVIEWS=YES",
                    "override_srs": outCRS
            }
        
        elif demType == 'dtm':
            groundfilter_stage = {
                    "type":"filters.range",
                    "limits":"Classification[2:2]"
            }

            dem_pipeline['pipeline'].append(groundfilter_stage)

            dem_stage = {
                    "type":"writers.gdal",
                    "filename":str(dem_outName)+ '.' + str(dem_outExt),
                    "gdaldriver":driver,
                    "nodata":-9999,
                    "output_type":gridMethod,
                    "resolution":float(dem_resolution),
                    "gdalopts":"COMPRESS=LZW,TILED=YES,blockxsize=256,blockysize=256,COPY_SRC_OVERVIEWS=YES",
                    "override_srs": outCRS
            }
        
        else:
            raise Exception("demType must be 'dsm' or 'dtm'.")
            
            
        dem_pipeline['pipeline'].append(dem_stage)
        
        return dem_pipeline

    # --------------------- Native UTM from AOI bounds ------------------
    
    @staticmethod
    def native_utm_crs_from_aoi_bounds(bounds,datum):
        """
        Get the native UTM coordinate reference system from the 

        :param bounds: shapely Polygon of bounding box in EPSG:4326 CRS
        :param datum: string with datum name (e.g., "WGS84")
        :return: UTM CRS code
        """
        utm_crs_list = query_utm_crs_info(
            datum_name=datum,
            area_of_interest=AreaOfInterest(
                west_lon_degree=bounds["west"],
                south_lat_degree=bounds["south"],
                east_lon_degree=bounds["east"],
                north_lat_degree=bounds["north"],
            ),
        )
        utm_crs = CRS.from_epsg(utm_crs_list[0].code)
        return utm_crs

    
    # Add the transform_options parameter with a default value of None
    @staticmethod
    def reproject_polygon(
        polygon: Polygon | MultiPolygon,
        source_crs: int | str | CRS,
        target_crs: int | str | CRS,
        transform_options: dict = None
    ) -> Polygon | MultiPolygon:
        """Reproject a shapely Polygon or MultiPolygon to a new CRS."""
        
        # Helper function to handle the optional dictionary
        def _if_not_none(value, default):
            return default if value is None else value

        # Pass the unpacked options dictionary to the transformer
        proj_transformer = Transformer.from_crs(
            source_crs,
            target_crs,
            always_xy=True,
            **_if_not_none(transform_options, {})
        )
        from shapely.ops import transform
        return transform(proj_transformer.transform, polygon)
        
    # -------- End‑to‑end driver to download data & create DEMs---------

    def dem_download_workflow(
        self,
        folder,
        output_name,                    # Desired generic output name for files on user's local file system (w/o extension, modifiers like "_DTM", "_DSM" will be added depending on product created)
        API_Key,                        # OpenTopography Enterprise API key       
        dem_resolution = 1.0,           # Desired grid size (in meters) for output raster DEM
        dataset_type = "compare",       # Whether dataset is compare or reference dataset    
        epoch: Optional[date] = None,   # Optional epoch date for CRS, if not provided, will use current date
        filterNoise = True,             # Option to remove points from USGS Class 7 (Low Noise) and Class 18 (High Noise).
        reclassify = False,         
        savePointCloud = False,         
        pc_resolution = 0.1,            # The desired resolution of the pointcloud based on the following definition: 
                                        #        A point resolution limit to select, expressed as a grid cell edge length. 
                                        #        Units correspond to resource coordinate system units. For example, 
                                        #        for a coordinate system expressed in meters, a resolution value of 0.1 
                                        #        will select points up to a ground resolution of 100 points per square meter.
                                        #        The resulting resolution may not be exactly this value: the minimum possible 
                                        #        resolution that is at least as precise as the requested resolution will be selected. 
                                        #        Therefore the result may be a bit more precise than requested. 
                                        # Source: https://pdal.io/stages/readers.ept.html#readers-ept
        outCRS = "WGS84 UTM",           # Output coordinate reference systemt (CRS), specified by ESPG code (e.g., 3857 - Web Mercator)
        method="idw",                   # method for gap-filling
        nodata=-9999,                   # no data values
        max_dist=100,                   # max distance to consider for gap filling
        smooth_iter=0                   # number of smoothing iterations
        
    ):
        """Download point clouds and produce DEMs for a single dataset.

        This workflow coordinates the download of point cloud data (via the
        OpenTopography API, USGS, or NOAA), generation of DTM and DSM rasters
        using PDAL, and optional gap filling.  It handles coordinate
        reference system selection including epoch tags, writes outputs
        to disk, and stores the resulting file paths on the ``GetDEMs``
        instance.  Parameters mirror those of :func:`make_DEM_pipeline_from_file`
        and :func:`make_DEM_pipeline_aws` but additionally require an API
        key when fetching OT-hosted datasets.

        Parameters
        ----------
        folder : str
            Directory in which to save downloaded point clouds and generated DEMs.
        output_name : str
            Base filename used when saving outputs (suffixes ``_DTM`` and
            ``_DSM`` will be appended).
        API_Key : str
            Enterprise API key for accessing OpenTopography-hosted datasets.
        dem_resolution : float, default 1.0
            Grid spacing for the output DEMs in map units.
        dataset_type : {'compare', 'reference'}, default 'compare'
            Indicates whether the dataset being processed corresponds to the
            ``compare`` or ``reference`` survey.
        epoch : datetime.date or None, optional
            Epoch date to embed in the CRS of the output DEM.  If omitted,
            the current date is used.
        filterNoise : bool, default True
            Remove noise classes 7 and 18 when gridding.
        reclassify : bool, default False
            Apply SMRF ground classification prior to gridding.
        savePointCloud : bool, default False
            Save the filtered point cloud to disk.
        pc_resolution : float, default 0.1
            Resolution for sampling points from EPT sources (in units of the
            source CRS).  This influences the density of the downloaded
            point cloud.
        outCRS : str, default 'WGS84 UTM'
            Output CRS for the DEM.  If set to ``"WGS84 UTM"``, a native UTM
            zone is chosen based on the AOI.
        method : str, default 'idw'
            Gap-filling interpolation method (see :func:`fill_no_data`).
        nodata : int or float, default -9999
            NoData value to assign in the output rasters.
        max_dist : float, default 100
            Maximum search distance for gap filling.
        smooth_iter : int, default 0
            Number of smoothing iterations for gap filling.

        Returns
        -------
        None
            The function operates via side effects: rasters are written to
            disk and attributes on the ``GetDEMs`` object are updated to
            reference the generated DEMs.
        """

        self.initial_compare_dataset_crs = int(self.ot.compare_horizontal_crs)
        self.initial_reference_dataset_crs = int(self.ot.reference_horizontal_crs)

        self.target_compare_dataset_crs = self.native_utm_crs_from_aoi_bounds(self.da.bounds,"WGS84").to_epsg()
        self.target_reference_dataset_crs = self.native_utm_crs_from_aoi_bounds(self.da.bounds,"WGS84").to_epsg()

        self.bounds_polygon_epsg_initial_compare_crs = self.reproject_polygon(self.da.polygon["merged_polygon"], 4326, self.initial_compare_dataset_crs)
        self.bounds_polygon_epsg_initial_reference_crs = self.reproject_polygon(self.da.polygon["merged_polygon"], 4326, self.initial_reference_dataset_crs)
        
    
        if dataset_type == "compare":
            bounds_polygon_epsg_initial_crs = self.bounds_polygon_epsg_initial_compare_crs
            data_source_ = self.ot.compare_data_source
            dataset_id = self.ot.compare_property_id
            dataset_crs_ = self.target_compare_dataset_crs
            self.compare_dtm_path = folder+output_name+'_'+dataset_type+'_DTM.tif'
            self.compare_dsm_path = folder+output_name+'_'+dataset_type+'_DSM.tif'
               
        elif dataset_type == "reference":
            data_source_ = self.ot.reference_data_source
            dataset_id = self.ot.reference_property_id
            bounds_polygon_epsg_initial_crs = self.bounds_polygon_epsg_initial_reference_crs
            dataset_crs_ = self.target_reference_dataset_crs
            self.reference_dtm_path = folder+output_name+'_'+dataset_type+'_DTM.tif'
            self.reference_dsm_path = folder+output_name+'_'+dataset_type+'_DSM.tif'
            
        else:
            raise ValueError("dataset_type must be either 'compare' or 'reference'")

        if outCRS == "WGS84 UTM":
            base_crs = CRS.from_epsg(dataset_crs_)
        else:
            base_crs = CRS.from_user_input(outCRS)

        # Create a new CRS definition with the epoch
        if epoch:
            try:
                # Convert the base CRS to a PROJ string
                proj_string = base_crs.to_proj4()
                # Calculate the epoch as a decimal year
                decimal_year = epoch.year + (epoch.timetuple().tm_yday - 1) / 365.25
                # Append the epoch parameter to the PROJ string
                final_out_crs_wkt = f"{proj_string} +epoch={decimal_year:.4f}"
                print(f"✔️ Using epoch {epoch} for {dataset_type} dataset via PROJ string.")
            except Exception as e:
                warnings.warn(f"Could not convert CRS to PROJ string to add epoch: {e}. Proceeding without epoch.")
                final_out_crs_wkt = base_crs.to_wkt()
        else:
            final_out_crs_wkt = base_crs.to_wkt()
            print(f"⚠️ No epoch provided for {dataset_type} dataset.")
        
        if data_source_ == 'ot':
            # Use the OpenTopography Enterprise API to download the point cloud data 
        
            # Ensure base_url matches your API's base URL
            base_url = "https://portal.opentopography.org//API"
            endpoint = "/pointcloud"
            
            params = {
            "datasetName": dataset_id,
            "south" : self.da.bounds['south'], 
            "north" : self.da.bounds['north'], 
            "west" : self.da.bounds['west'], 
            "east" : self.da.bounds['east'], 
            "API_Key" : API_Key, #All OT hosted point cloud datasets require an enterprise partner API key for access. Please email info@opentopography.org for more information.   
        }
        
            # Make the GET request to the API
            response = requests.get(url=base_url + endpoint, params=params)

            # Check if the request was successful
            if response.status_code == 200:
                filename = folder + output_name+'_'+dataset_type+'.laz'
                with open(filename, 'wb') as file:
                    file.write(response.content)
            # Wait until the file is fully downloaded
                while not os.path.exists(filename):
                    time.sleep(1)
            
            
                ot_dtm_pipeline = self.make_DEM_pipeline_from_file(folder+output_name+'_'+dataset_type+'.laz', bounds_polygon_epsg_initial_crs, dem_resolution,
                                    filterNoise=False, reclassify=False, savePointCloud=False, outCRS=final_out_crs_wkt,
                                    pc_outName=folder+output_name, pc_outType='laz', demType='dtm', gridMethod='idw', 
                                    dem_outName=folder+output_name+'_'+dataset_type+'_DTM', dem_outExt='tif', driver="GTiff")
                ot_dtm_pipeline = pdal.Pipeline(json.dumps(ot_dtm_pipeline))
                ot_dtm_pipeline.execute_streaming(chunk_size=1000000)

                ot_dsm_pipeline = self.make_DEM_pipeline_from_file(folder+output_name+'_'+dataset_type+'.laz', bounds_polygon_epsg_initial_crs, dem_resolution,
                                        filterNoise=False, reclassify=False, savePointCloud=False, outCRS=final_out_crs_wkt,
                                        pc_outName=folder+output_name, pc_outType='laz', demType='dsm', gridMethod='max', 
                                        dem_outName=folder+output_name+'_'+dataset_type+'_DSM', dem_outExt='tif', driver="GTiff")
                ot_dsm_pipeline = pdal.Pipeline(json.dumps(ot_dsm_pipeline))
                ot_dsm_pipeline.execute_streaming(chunk_size=1000000)
                
            else:
                print(f"Error: {response.status_code}")
        
        elif data_source_ == "usgs":
            usgs_dtm_pipeline = self.make_DEM_pipeline_aws(bounds_polygon_epsg_initial_crs, [dataset_id], pc_resolution, dem_resolution, data_source = "usgs",
                    filterNoise = False, reclassify = False, savePointCloud = False, outCRS = final_out_crs_wkt,
                    pc_outName = folder+output_name+'_'+dataset_type, pc_outType = 'laz', demType = 'dtm', gridMethod = 'idw', 
                    dem_outName = folder+output_name+'_'+dataset_type+'_DTM', dem_outExt = 'tif', driver = "GTiff")

            usgs_dtm_pipeline = pdal.Pipeline(json.dumps(usgs_dtm_pipeline))
            usgs_dtm_pipeline.execute_streaming(chunk_size=1000000)
            
            usgs_dsm_pipeline = self.make_DEM_pipeline_aws(bounds_polygon_epsg_initial_crs, [dataset_id], pc_resolution, dem_resolution, data_source = "usgs",
                            filterNoise = False, reclassify = False, savePointCloud = False, outCRS = final_out_crs_wkt,
                            pc_outName = folder+output_name+'_'+dataset_type, pc_outType = 'laz', demType = 'dsm', gridMethod = 'max', 
                            dem_outName = folder+output_name+'_'+dataset_type+'_DSM', dem_outExt = 'tif', driver = "GTiff")
            
            usgs_dsm_pipeline = pdal.Pipeline(json.dumps(usgs_dsm_pipeline))
            usgs_dsm_pipeline.execute_streaming(chunk_size=1000000)
        
        elif data_source_ == "noaa":
            # Use AWS bucket and PDAL to download the point cloud data
            
            noaa_dtm_pipeline = self.make_DEM_pipeline_aws(bounds_polygon_epsg_initial_crs, [dataset_id], pc_resolution, dem_resolution, data_source = "noaa",
                            filterNoise = False, reclassify = False, savePointCloud = False, outCRS = final_out_crs_wkt,
                            pc_outName = folder+output_name+'_'+dataset_type, pc_outType = 'laz', demType = 'dtm', gridMethod = 'idw', 
                            dem_outName = folder+output_name+'_'+dataset_type+'_DTM', dem_outExt = 'tif', driver = "GTiff")
            noaa_dtm_pipeline = pdal.Pipeline(json.dumps(noaa_dtm_pipeline))
            noaa_dtm_pipeline.execute_streaming(chunk_size=1000000)
            
            noaa_dsm_pipeline = self.make_DEM_pipeline_aws(bounds_polygon_epsg_initial_crs, [dataset_id], pc_resolution, dem_resolution, data_source = "noaa",
                              filterNoise = False, reclassify = False, savePointCloud = False, outCRS = final_out_crs_wkt,
                              pc_outName = folder+output_name+'_'+dataset_type, pc_outType = 'laz', demType = 'dsm', gridMethod = 'max', 
                              dem_outName = folder+output_name+'_'+dataset_type+'_DSM', dem_outExt = 'tif', driver = "GTiff")
            noaa_dsm_pipeline = pdal.Pipeline(json.dumps(noaa_dsm_pipeline))
            noaa_dsm_pipeline.execute_streaming(chunk_size=1000000)
        
        else:
            raise ValueError("Data source must be either 'ot', 'usgs' or 'noaa'")
        
        # Determine which DEM paths to fill
        if dataset_type == "compare":
            dtm_path = self.compare_dtm_path
            dsm_path = self.compare_dsm_path
        else:
            dtm_path = self.reference_dtm_path
            dsm_path = self.reference_dsm_path

        # Fill NoData holes in the generated DEMs
        self.fill_no_data(dtm_path, dtm_path, method=method, nodata=nodata, max_dist=max_dist, smooth_iter=smooth_iter)
        self.fill_no_data(dsm_path, dsm_path, method=method, nodata=nodata, max_dist=max_dist, smooth_iter=smooth_iter)
    
    
    # -------------------Single-DEM utilities---------------------------
    @staticmethod
    def query_single_dem(da: DataAccess, product_format="PointCloud", include_federated=True, detail=False, save_as=None):
        """
        Query the OT catalog within AOI and return (OpenTopographyQuery, DataFrame).
        """
        otq = OpenTopographyQuery(da)
        df = otq.query_catalog(product_format=product_format, include_federated=include_federated, detail=detail, save_as=save_as)
        if df.empty:
            raise ValueError("No datasets found for AOI.")
        return otq, df

    @staticmethod
    def download_single_dem(
        da: DataAccess,
        folder: str,
        output_name: str,
        shapefile: str | None = None,
        index: int = 0,
        dem_type: str = "dtm",
        dem_resolution: float = 1.0,
        filterNoise: bool = True,
        reclassify: bool = False,
        pc_resolution: float = 0.1,
        outCRS: str = "WGS84 UTM",
        method: str = "idw",
        nodata: int = -9999,
        max_dist: float = 100,
        smooth_iter: int = 0,
    ) -> str:
        """
        Define AOI from shapefile or existing bounds, query OT, and download a single DEM.
        Returns DEM file path.
        """
        # Define AOI
        if shapefile:
            da.define_bounds_from_file(shapefile, target_crs="EPSG:4326")
        elif not getattr(da, "bounds", None):
            raise ValueError("AOI undefined: provide shapefile or set bounds.")

        # Query and pick
        otq, df = GetDEMs.query_single_dem(da, save_as=None)
        otq.pick(idx_compare=index, idx_reference=index)

        # Run DEM workflow
        gd = GetDEMs(da, otq)
        gd.dem_download_workflow(
            folder=folder,
            output_name=output_name,
            dem_resolution=dem_resolution,
            dataset_type="compare",
            epoch=epoch_to_use,
            filterNoise=filterNoise,
            reclassify=reclassify,
            savePointCloud=False,
            pc_resolution=pc_resolution,
            outCRS=outCRS,
            method=method,
            nodata=nodata,
            max_dist=max_dist,
            smooth_iter=smooth_iter,
        )
        # Return path
        #return raster

# ----------------------------------------------------------------------
# LocalDataManager  – handle USER-SUPPLIED point-clouds & DEMs
# ----------------------------------------------------------------------

class LocalDataManager:
    """
    Accept a user LA(S/Z) file **or** raster DEM, create a matching DTM/DSM
    (optionally gap-filled), and return a ``Raster`` ready for
    GeoidTransformer / RasterPair / TopoDifferencer.
    """

    # --- (No changes to __init__ or geometry helpers) ---
    def __init__(
        self,
        src_path: str | Path,
        out_dir: str | Path = "./user_data",
        *,
        dem_resolution : float = 1.0,
        dem_type       : str   = "dtm",
        filter_noise   : bool  = True,
        reclassify     : bool  = False,
        grid_method    : str   = "idw",
        gapfill        : bool  = True,
    ):
        self.src          = Path(src_path).expanduser()
        self.out_dir      = Path(out_dir).expanduser()
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # --- Parameters for point cloud processing ---
        self.dem_res      = dem_resolution
        self.dem_type     = dem_type.lower()
        self.filter_noise = filter_noise
        self.reclassify   = reclassify
        self.grid_method  = grid_method
        self.gapfill      = gapfill

        # --- Attributes to be populated from metadata ---
        self.orig_epsg_hz: int | None = None
        self.orig_epsg_vt: int | None = None
        self.resolution: Tuple[float, float] | None = None
        self.collection_start_date: date | None = None
        self.collection_end_date: date | None = None
        self.collection_midpoint_date: date | None = None

        if self.src.suffix.lower() in {".las", ".laz"}:
            self.kind = "pointcloud"
        elif self.src.suffix.lower() in {".tif", ".tiff"}:
            self.kind = "raster"
        else:
            raise ValueError(f"Unsupported file type: {self.src.suffix}")

        self._process()

    # ------------------------------------------------------------------
    # ------------------------- geometry helpers -----------------------
    # ------------------------------------------------------------------
    @staticmethod
    def _gps_seconds_to_date(gps_seconds: float) -> date | None:
        """Converts raw GPS time seconds to a Python date object."""
        if not gps_seconds or gps_seconds == 0:
            return None
        if gps_seconds < 1e9: # Handle LAS 1.4 adjustment
            gps_seconds += 1e9
        gps_epoch_start = date(1980, 1, 6)
        return (gps_epoch_start + timedelta(seconds=gps_seconds))

    @staticmethod
    def _get_acquisition_date_from_raster(raster_path: Path) -> date | None:
        """Extracts acquisition date by checking common metadata tags."""
        common_tags = ["ACQUISITION_DATE", "TIFFTAG_DATETIME"]
        with rasterio.open(raster_path) as src:
            tags = src.tags()
            for tag_name in common_tags:
                if tag_name in tags:
                    date_str = tags[tag_name]
                    try:
                        # Parse formats like '2018:01:18 12:00:00' or '2018-01-18'
                        return pd.to_datetime(date_str).date()
                    except (ValueError, TypeError):
                        continue
        warnings.warn(f"Could not find a valid acquisition date in the metadata for {raster_path.name}")
        return None
    
    @staticmethod
    def _get_crs_components(wkt: str) -> tuple[int | None, int | None]:
        """
        Extract horizontal and vertical EPSG codes from a WKT string.

        Returns:
            A tuple of (horizontal_epsg, vertical_epsg).
        """
        crs = CRS.from_wkt(wkt)
        hz_epsg, vt_epsg = None, None

        if crs.is_compound:
            for sub_crs in crs.sub_crs:
                if sub_crs.is_projected or sub_crs.is_geographic:
                    hz_epsg = sub_crs.to_epsg()
                elif sub_crs.is_vertical:
                    vt_epsg = sub_crs.to_epsg()
        else:
            # If not compound, it's likely just a horizontal CRS
            hz_epsg = crs.to_epsg()

        return hz_epsg, vt_epsg
    
    @staticmethod
    def _epsg_from_wkt(wkt: str) -> int | None:
        """Return EPSG code (or None) from a CRS WKT string."""
        return CRS.from_wkt(wkt).to_epsg(min_confidence=25)

    @staticmethod
    def _determine_utm_epsg(poly4326: Polygon) -> str:
        """Return the EPSG code string (e.g. '32611') of the centroid’s UTM zone."""
        lon, lat = poly4326.centroid.xy
        zone     = int((lon[0] + 180) / 6) + 1
        hemi     = "north" if lat[0] >= 0 else "south"
        proj     = Proj(f"+proj=utm +zone={zone} +{hemi} +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
        epsg     = CRS(proj.srs).to_epsg()
        return str(epsg or (32600 + zone if hemi == "north" else 32700 + zone))

    @staticmethod
    def _reproject_poly(poly: Polygon, src_epsg: str | int, dst_epsg: str | int) -> Polygon:
        """
        Reproject a polygon between coordinate reference systems.

        Parameters
        ----------
        poly : shapely.geometry.Polygon
            Geometry to reproject.
        src_epsg : int or str
            EPSG code of the source CRS.
        dst_epsg : int or str
            EPSG code of the destination CRS.

        Returns
        -------
        shapely.geometry.Polygon
            The polygon transformed into the target CRS.
        """
        tf = Transformer.from_crs(f"EPSG:{src_epsg}", f"EPSG:{dst_epsg}", always_xy=True)
        return transform(lambda x, y: tf.transform(x, y), poly)

    # ------------------------------------------------------------------
    # -------------------------- PDAL helpers --------------------------
    # ------------------------------------------------------------------
    @staticmethod
    def _build_pdal_pipeline(
        filename      : str,
        extent        : Polygon,
        *,
        filter_noise  : bool,
        reclassify    : bool,
        save_pc       : bool,
        out_crs_wkt   : str,
        pc_out_name   : str,
        pc_out_ext    : str = "laz",
    ) -> dict:
        """Return a readers → filters → writers PDAL pipeline dict."""
        pipe = [
            {"type": "readers.las", "filename": filename},
            {"type": "filters.crop", "polygon": extent.wkt},
        ]

        if filter_noise:
            pipe.append({"type": "filters.range",
                         "limits": "Classification![7:7], Classification![18:18]"})

        if reclassify:
            pipe.extend([
                {"type": "filters.assign", "value": "Classification = 0"},
                {"type": "filters.smrf"},
                {"type": "filters.range", "limits": "Classification[2:2]"},
            ])
        
        pipe.append({"type": "filters.reprojection", "out_srs": out_crs_wkt})

        if save_pc:
            # ... (no change here)
            writer = {"type": "writers.las", "filename": f"{pc_out_name}.{pc_out_ext}"}
            if pc_out_ext == "laz":
                writer["compression"] = "laszip"
            pipe.append(writer)

        return {"pipeline": pipe}

    def _build_dem_pipeline(
        self,
        filename       : str,
        extent         : Polygon,
        *,
        final_crs_wkt  : str,
        utm_crs_str    : str,
        dem_type       : str,
        grid_method    : str,
        dem_res        : float,
        dem_out_name   : str,
        filter_noise   : bool,
        reclassify     : bool,
    ) -> dict:
        """
        Construct a PDAL pipeline to grid point cloud data into a raster DEM.

        Parameters
        ----------
        filename : str
            Path to the LAS/LAZ file to be gridded.
        extent : shapely.geometry.Polygon
            Polygon defining the crop window in the input CRS.
        final_crs_wkt : str
            Well‑known text (WKT) representation of the desired output CRS,
            including any vertical component or epoch.
        utm_crs_str : str
            EPSG code string for the horizontal UTM CRS used during
            intermediate reprojection.
        dem_type : {'dtm', 'dsm'}
            The type of DEM to generate: ground-only DTM or first-return DSM.
        grid_method : str
            Gridding algorithm to use when rasterizing (e.g. ``'idw'``).
        dem_res : float
            Output pixel size for the DEM.
        dem_out_name : str
            Full pathname for the output raster file (including extension).
        filter_noise : bool
            Remove noise points prior to gridding.
        reclassify : bool
            Apply SMRF classification to isolate ground returns.

        Returns
        -------
        dict
            A PDAL pipeline dictionary with crop, optional filters, and a
            GDAL writer configured with the final CRS.
        """

        pipe = self._build_pdal_pipeline(
            filename     = filename,
            extent       = extent,
            filter_noise = filter_noise,
            reclassify   = reclassify,
            save_pc      = False,
            out_crs_wkt  = utm_crs_str,
            pc_out_name  = "dummy",
        )["pipeline"]

        if dem_type == "dtm":
            pipe.append({"type": "filters.range", "limits": "Classification[2:2]"})

        writer = {
            "type"      : "writers.gdal",
            "filename"  : dem_out_name,
            "gdaldriver": "GTiff",
            "nodata"    : -9999,
            "output_type": grid_method,
            "resolution": float(dem_res),
            "radius"    : 2 * float(dem_res),
            "gdalopts"  : "COMPRESS=LZW,TILED=YES,blockxsize=256,blockysize=256,COPY_SRC_OVERVIEWS=YES",
            "override_srs": final_crs_wkt,
        }
        
        pipe.append(writer)
        return {"pipeline": pipe}

    @staticmethod
    def _fill_nodata(
        src_tif        : str,
        *,
        method         : str = "idw",
        nodata_value   : float | int = -9999,
        max_dist       : int   = 100,
        smooth_iter    : int   = 0,
    ) -> None:
        """Fill NoData cells in-place."""
        ds   = rasterio.open(src_tif)
        band = ds.read(1)
        mask = band == nodata_value

        if not np.any(mask):
            ds.close()
            return  # nothing to fill

        if method == "idw":
            tmp = ds.read(1, masked=True)
            filled = fillnodata(tmp, max_search_distance=max_dist,
                                              smoothing_iterations=smooth_iter)
            
            
        else:
            # scipy-based fallback
            valid   = np.where(~mask)
            missing = np.where(mask)
            coords  = np.column_stack(valid)
            vals    = band[valid]
            if method in {"nearest", "linear", "cubic"}:
                filled_vals = griddata(coords, vals, np.column_stack(missing), method=method)
            elif method == "spline":
                rbf = Rbf(coords[:, 0], coords[:, 1], vals, function="thin_plate")
                filled_vals = rbf(missing[0], missing[1])
            else:
                raise ValueError("Unknown fill method.")
            filled = band.copy()
            filled[missing] = filled_vals

        with rasterio.open(src_tif, "r+") as dst:
            dst.write(filled, 1)

    @staticmethod
    def get_vertical_crs(raster_obj: Raster) -> CRS | None:
        """
        Extracts the vertical CRS from a Raster object.

        Args:
            raster_obj: An instance of your Raster class.

        Returns:
            A pyproj.CRS object for the vertical datum, or None if not found.
        """
        if not hasattr(raster_obj, 'crs'):
            print("Error: Raster object does not have a .crs attribute.")
            return None
        
        main_crs = CRS(raster_obj.crs) 

        if main_crs and main_crs.is_compound:
            for sub_crs in main_crs.sub_crs:
                if sub_crs.is_vertical:
                    return sub_crs  # Return the vertical CRS object
        
        return None # Return None if no vertical CRS is found
    # ------------------------------------------------------------------
    # ---------------------------- Processing Workflows ----------------
    # ------------------------------------------------------------------
    
    def _process(self):
        """Routes to the correct processing method based on file type."""
        if self.kind == "pointcloud":
            self._process_pointcloud()
        else:
            self._process_raster()

    def _process_raster(self):
        """Copies a raster and extracts its metadata."""
        print(f"📄 Processing raster file: {self.src.name}")
        # 1. Copy raster to ensure we work with a local copy
        dst = self.out_dir / self.src.name
        if dst != self.src:
            rio.open_rasterio(self.src, masked=True).rio.to_raster(dst)
        self.raster = Raster(dst)

        # 2. Extract metadata
        main_crs = CRS(self.raster.crs)
        if main_crs.is_compound:
            hz_crs = next((s for s in main_crs.sub_crs if s.is_projected or s.is_geographic), None)
            vt_crs = next((s for s in main_crs.sub_crs if s.is_vertical), None)
            self.orig_epsg_hz = hz_crs.to_epsg() if hz_crs else None
            self.orig_epsg_vt = vt_crs.to_epsg() if vt_crs else None
        else:
            self.orig_epsg_hz = main_crs.to_epsg()
            self.orig_epsg_vt = None
        
        self.resolution = self.raster.data.rio.resolution()
        
        # For consistency, set all date attributes from the single acquisition date
        acq_date = self._get_acquisition_date_from_raster(self.raster.path)
        if acq_date:
            self.collection_start_date = acq_date
            self.collection_end_date = acq_date
            self.collection_midpoint_date = acq_date
        
        print("   --- Raster Metadata ---")
        print(f"   - Horizontal EPSG: {self.orig_epsg_hz}")
        print(f"   - Vertical EPSG:   {self.orig_epsg_vt or 'Not Found'}")
        print(f"   - Resolution (X,Y): {self.resolution}")
        print(f"   - Acquisition Date: {self.collection_midpoint_date or 'Not Found'}")


    def _process_pointcloud(self):
        """Processes a point cloud file to create a DEM and extracts metadata."""
        print(f"☁️ Processing point cloud file: {self.src.name}")
        
        # 1. Derive CRS, dates, and bounding poly from the point cloud
        (
            epsg_orig_hz, epsg_orig_vt, epsg_utm, poly_utm, 
            self.collection_start_date, self.collection_end_date
        ) = self._get_pointcloud_metadata(self.src)

        midpoint_epoch = None
        if self.collection_start_date and self.collection_end_date:
            time_delta = self.collection_end_date - self.collection_start_date
            midpoint_epoch = self.collection_start_date + time_delta / 2
            self.collection_midpoint_date = midpoint_epoch
            print(f"   - Collection Dates: {self.collection_start_date} to {self.collection_end_date}")
            print(f"   - Midpoint Epoch:   {self.collection_midpoint_date.isoformat()}")

        self.orig_epsg_hz = epsg_orig_hz
        self.orig_epsg_vt = epsg_orig_vt
        self.utm_epsg = epsg_utm
        
        # 2. Build the final, epoch-aware, compound CRS
        utm_crs = CRS.from_epsg(epsg_utm)
        
        # Start with the horizontal UTM CRS
        final_crs_wkt = utm_crs.to_wkt()

        # If a vertical CRS is present, create a compound CRS
        if epsg_orig_vt:
            vert_crs = CRS.from_epsg(epsg_orig_vt)
            compound_crs = CRS.from_wkt(f'COMPOUNDCRS["{utm_crs.name} + {vert_crs.name}", {utm_crs.to_wkt()}, {vert_crs.to_wkt()}]')
            final_crs_wkt = compound_crs.to_wkt()
        
        # If an epoch is present, bind it to the current CRS
        if midpoint_epoch:
            try:
                base_crs_for_epoch = CRS.from_wkt(final_crs_wkt)
                # Suppress the expected UserWarning
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    proj_string = base_crs_for_epoch.to_proj4()
                
                decimal_year = midpoint_epoch.year + (midpoint_epoch.timetuple().tm_yday - 1) / 365.25
                # Overwrite the WKT string with the simpler PROJ string + epoch
                final_crs_wkt = f"{proj_string} +epoch={decimal_year:.4f}"
                print(f"✔️ Using epoch {midpoint_epoch.isoformat()} for CRS via PROJ string.")
            except Exception as e:
                warnings.warn(f"Could not convert CRS to PROJ string to add epoch: {e}. Proceeding without epoch.")
        
        # 3. Build and execute PDAL pipeline to create DEM
        dem_path = self.out_dir / f"{self.src.stem}_{self.dem_type.upper()}.tif"
        dem_pipe = self._build_dem_pipeline(
            filename=str(self.src),
            extent=poly_utm,
            final_crs_wkt=final_crs_wkt, # Pass the full final WKT
            utm_crs_str=f"EPSG:{epsg_utm}", # Pass the simple UTM for reprojection
            dem_type=self.dem_type,
            grid_method=self.grid_method,
            dem_res=self.dem_res,
            dem_out_name=dem_path.as_posix(),
            filter_noise=self.filter_noise,
            reclassify=self.reclassify,
        )
        pdal.Pipeline(json.dumps(dem_pipe)).execute()

        # 4. Gap-fill the new DEM
        if self.gapfill:
            self._fill_nodata(str(dem_path))
        self.raster = Raster(dem_path)

    # --- (No changes to _get_pointcloud_metadata or get_raster) ---
    def _get_pointcloud_metadata(self, pc_path: Path) -> Tuple[int | None, int | None, str, Polygon, date | None, date | None]:
        """Wrapper around a PDAL pipeline to extract key metadata."""
        meta_pipe = pdal.Pipeline(json.dumps({
            "pipeline":[
                {"type":"readers.las","filename":str(pc_path)},
                # Get stats for GpsTime in addition to X,Y,Z
                {"type":"filters.stats","dimensions":"X,Y,Z,GpsTime"},
                {"type":"filters.info"},
            ]
        }))
        meta_pipe.execute()
        md = meta_pipe.metadata["metadata"]
        
        # --- CRS Info ---
        wkt = md["filters.info"]["srs"]["wkt"]
        epsg_orig_hz, epsg_orig_vt = self._get_crs_components(wkt)

        # --- Bounding Poly Info ---
        coords = md["filters.stats"]["bbox"]["EPSG:4326"]["boundary"]["coordinates"][0]
        poly = Polygon([(float(pt[0]), float(pt[1])) for pt in coords])
        epsg_utm = self._determine_utm_epsg(poly)
        poly_utm = self._reproject_poly(poly, 4326, epsg_utm)
        
        # --- GPS Time Info ---
        stats = md.get("filters.stats", {}).get("statistic", [])
        min_time, max_time = None, None
        for stat in stats:
            if stat["name"] == "GpsTime":
                min_time = self._gps_seconds_to_date(stat["minimum"])
                max_time = self._gps_seconds_to_date(stat["maximum"])
                break
                
        return epsg_orig_hz, epsg_orig_vt, epsg_utm, poly_utm, min_time, max_time
    
    def get_raster(self) -> Raster:
        """Return a ready-to-use ``Raster`` instance."""
        return self.raster

class RasterPairProcessor:
    """
    Processes a pair of rasters by reprojecting them to a local UTM zone,
    aligning their grids, and cropping them to their precise area of
    overlapping data.
    """
    def __init__(self, raster_pair: RasterPair, out_dir: str | Path = "./processed_rasters"):
        self.raster_pair = raster_pair
        self.out_dir = Path(out_dir).expanduser()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.processed_raster1: Raster | None = None
        self.processed_raster2: Raster | None = None
        self.process()

    # NEW: Helper to find the appropriate UTM zone for the raster pair
    def _determine_utm_crs(self, raster1: Raster, raster2: Raster) -> CRS:
        """Determines the most suitable UTM CRS for the combined extent of two rasters."""
        # Get bounds of both rasters and reproject to WGS84 if necessary
        gdf1 = gpd.GeoDataFrame(geometry=[box(*raster1.data.rio.bounds())], crs=raster1.crs).to_crs("EPSG:4326")
        gdf2 = gpd.GeoDataFrame(geometry=[box(*raster2.data.rio.bounds())], crs=raster2.crs).to_crs("EPSG:4326")
        
        # Combine the geometries and find the centroid
        combined_bounds = unary_union([gdf1.geometry.iloc[0], gdf2.geometry.iloc[0]])
        lon, lat = combined_bounds.centroid.x, combined_bounds.centroid.y

        # Query for the best UTM zone based on the centroid
        utm_crs_list = query_utm_crs_info(
            datum_name="WGS 84",
            area_of_interest=AreaOfInterest(
                west_lon_degree=lon, south_lat_degree=lat,
                east_lon_degree=lon, north_lat_degree=lat,
            ),
        )
        return CRS.from_epsg(utm_crs_list[0].code)

    def _get_raster_overlap_poly(self, raster1: Raster, raster2: Raster) -> Polygon | None:
        """Calculates the intersection of valid data areas of two rasters."""
        print("   - Calculating valid data overlap...")
        # Get valid data shape for raster 1
        with rasterio.open(raster1.path) as r1_src:
            r1_mask = r1_src.read_masks(1) > 0
            r1_shapes = list(rasterio.features.shapes(r1_mask.astype('uint8'), mask=r1_mask, transform=r1_src.transform))
            r1_polys = [shape(geom) for geom, val in r1_shapes if val == 1]
            r1_valid_poly = unary_union(r1_polys)

        # Get valid data shape for raster 2
        with rasterio.open(raster2.path) as r2_src:
            r2_mask = r2_src.read_masks(1) > 0
            r2_shapes = list(rasterio.features.shapes(r2_mask.astype('uint8'), mask=r2_mask, transform=r2_src.transform))
            r2_polys = [shape(geom) for geom, val in r2_shapes if val == 1]
            r2_valid_poly = unary_union(r2_polys)

        # Intersect the two valid data polygons
        overlap_poly = r1_valid_poly.intersection(r2_valid_poly)

        if overlap_poly.is_empty:
            warnings.warn("Rasters do not have overlapping valid data areas.")
            return None
        return overlap_poly

    def _crop_raster_to_poly(self, raster: Raster, poly: Polygon, poly_crs, out_path: Path) -> Raster:
        """Crops a raster to a given polygon and saves it."""
        print(f"   - Cropping {raster.path.name}...")
        clipped_data = raster.data.rio.clip([poly], crs=poly_crs, from_disk=True)
        clipped_data.rio.to_raster(out_path)
        return Raster(out_path)

    def process(self):
        """Executes the full workflow: reproject, regrid, and crop."""
        r1_initial = self.raster_pair.raster1
        r2_initial = self.raster_pair.raster2
        print(f"Processing raster pair: {r1_initial.path.name} and {r2_initial.path.name}")
        
        # --- NEW STEP 1: Determine target UTM CRS and reproject both rasters ---
        target_utm_crs = self._determine_utm_crs(r1_initial, r2_initial)
        print(f"   - Determined target UTM CRS: {target_utm_crs.name} ({target_utm_crs.to_epsg()})")

        reprojected_rasters = []
        for i, r_initial in enumerate([r1_initial, r2_initial]):
            # Check for an epoch in the source raster's CRS
            wkt = CRS(r_initial.crs).to_wkt()
            epoch_match = re.search(r'FRAME_EPOCH\[(\d+\.\d+)\]', wkt)
            
            target_crs_wkt = target_utm_crs.to_wkt()
            # If an epoch exists, bind it to the target UTM CRS
            if epoch_match:
                epoch = float(epoch_match.group(1))
                print(f"   - Found epoch {epoch} for raster {i+1}. Preserving it in reprojection.")
                # Robustly get the base geographic CRS
                if target_utm_crs.is_projected:
                    base_geod_crs = target_utm_crs.source_crs
                else:
                    base_geod_crs = target_utm_crs
                target_crs_wkt = (
                    f'BOUNDCRS(SOURCECRS({target_utm_crs.to_wkt()}),'
                    f'TARGETCRS({base_geod_crs.to_wkt()}),'
                    f'ABRIDGEDTRANSFORMATION(METHOD["Time-dependent Helmert"],'
                    f'PARAMETER["Epoch",{epoch:.4f},TIMEUNIT["year"]]))'
                )
            
            # Reproject to the target CRS (with or without epoch)
            out_path = self.out_dir / f"{r_initial.path.stem}_utm.tif"
            print(f"   - Reprojecting raster {i+1} to UTM...")
            r_initial.data.rio.reproject(target_crs_wkt).rio.to_raster(out_path)
            reprojected_rasters.append(Raster(out_path))

        r1_utm, r2_utm = reprojected_rasters
        
        # --- STEP 2: Find overlap on the new UTM rasters ---
        overlap_poly = self._get_raster_overlap_poly(r1_utm, r2_utm)
        if not overlap_poly:
            return

        # --- STEP 3: Regrid raster 1 to match raster 2's grid ---
        print(f"   - Regridding {r1_utm.path.name} to match {r2_utm.path.name}'s grid...")
        r1_regridded_path = self.out_dir / f"{r1_utm.path.stem}_regridded.tif"
        r1_utm.data.rio.reproject_match(r2_utm.data).rio.to_raster(r1_regridded_path)
        r1_regridded = Raster(r1_regridded_path)
        
        # --- STEP 4: Crop both aligned rasters to the overlap area ---
        poly_crs = r2_utm.crs # The CRS of the polygon is the common UTM CRS
        r1_final_path = self.out_dir / f"{r1_regridded.path.stem}_final.tif"
        self.processed_raster1 = self._crop_raster_to_poly(r1_regridded, overlap_poly, poly_crs, r1_final_path)

        r2_final_path = self.out_dir / f"{r2_utm.path.stem}_final.tif"
        self.processed_raster2 = self._crop_raster_to_poly(r2_utm, overlap_poly, poly_crs, r2_final_path)
        
        print("✅ Raster pair processing complete.")

    def get_processed_pair(self) -> RasterPair | None:
        """
        Return a new :class:`RasterPair` of processed rasters or ``None``.

        After running :meth:`process`, this method wraps the two processed
        rasters (aligned, reprojected, and cropped to the overlap area) into
        a :class:`RasterPair`.  If processing has not yet been completed, or
        if no overlap exists, ``None`` is returned.

        Returns
        -------
        RasterPair or None
            A pair of processed rasters, or ``None`` if processing failed.
        """
        if self.processed_raster1 and self.processed_raster2:
            return RasterPair(self.processed_raster1, self.processed_raster2)
        return None

# ----------------------------------------------------------------------
# GeoidTransformer  – NEW CLASS
# ----------------------------------------------------------------------
class GeoidTransformer:
    """
    Handles vertical CRS transformations for a RasterPair.
    """
    _GEOID_MAP = {
        # --- Canonical Grid Filenames ---
        "geoid18_conus": "us_nga_geoid18_conus.tif", "geoid12b_conus": "g2012b_conus.gtx",
        "geoid12a_conus": "g2012a_conus.gtx", "geoid09_conus": "g2009_conus.gtx",
        "geoid06_conus": "g2006_conus.gtx", "geoid03_conus": "g2003_conus.gtx",
        "geoid99_conus": "g1999_conus.gtx", "geoid18_ak": "us_nga_geoid18_ak.tif",
        "geoid12b_ak": "g2012b_ak.gtx", "geoid12a_ak": "g2012a_ak.gtx",
        "geoid09_ak": "g2009_ak.gtx", "geoid06_ak": "g2006_ak.gtx",
        "geoid03_ak": "g2003_ak.gtx", "geoid99_ak": "g1999_ak.gtx",
        "geoid12b_hi": "g2012b_hi.gtx", "geoid12a_hi": "g2012a_hi.gtx",
        "geoid09_hi": "g2009_hi.gtx", "geoid06_hi": "g2006_hi.gtx",
        "geoid03_hi": "g2003_hi.gtx", "geoid99_hi": "g1999_hi.gtx",
        "geoid18_prvi": "us_nga_geoid18_prvi.tif", "geoid12b_prvi": "g2012b_prvi.gtx",
        "geoid12a_prvi": "g2012a_prvi.gtx", "geoid09_prvi": "g2009_prvi.gtx",
        "geoid03_prvi": "g2003_prvi.gtx", "geoid99_prvi": "g1999_prvi.gtx",
        "geoid18_guamnmi": "us_nga_geoid18_guamnmi.tif", "geoid03_guam": "g2003_guam.gtx",
        "geoid18_samoa": "us_nga_geoid18_samoa.tif", "geoid03_samoa": "g2003_samoa.gtx",
        "vertcon_conus": "vertcon_conus.gtx", "vertcon_ak": "vertcon_ak.gtx",
        "egm2008_1": "egm08_1.gtx", "egm2008": "egm08_25.gtx",
        "egm96": "egm96_15.gtx", "egm84": "egm84-30.gtx",
        "cgg2013a": "ca_geoid_CGG2013a.gtx", "ht2_2010v70": "ca_ht2_2010v70.gtx",
        "ausgeoid2020": "au_ga_AUSGeoid2020.gtx", "ausgeoid09": "au_ga_AUSGeoid09.gtx",
        "osgm15": "uk_osgm15.gtx", "egg2008": "eg_eur_eggm08.gtx",
        "nkg2015": "is_lmi_NKG2015.gtx", "gcg2016": "de_bkg_GCG2016.gtx",
        "raf20": "fr_ign_RAF20.gtx", "chgeo2004": "ch_ln02_CHGeo2004.gtx",
        "rednap": "es_ign_EGM08_REDNAP.gtx", "italiageo04": "it_gm_igmi2004.gtx",
        "nzgeoid2016": "nz_linz_nzgeoid2016.gtx", "gsigeo2011": "jp_gsi_gsigeo2011.gtx",
        "kngeo18": "kr_ngii_kngeo18.gtx", "mapgeo2015": "br_ibge_MAPGEO2015.gtx",
        "argeo-ar": "ar_ign_SR-GEO-AR.gtx", "sageoid2010": "za_geoid2010.gtx",
        # --- Aliases ---
        "geoid18": "us_nga_geoid18_conus.tif", "g18": "us_nga_geoid18_conus.tif",
        "geoid12b": "g2012b_conus.gtx", "g12b": "g2012b_conus.gtx",
        "geoid12a": "g2012a_conus.gtx", "g12a": "g2012a_conus.gtx",
        "geoid09": "g2009_conus.gtx", "g09": "g2009_conus.gtx",
        "geoid06": "g2006_conus.gtx", "g06": "g2006_conus.gtx",
        "geoid03": "g2003_conus.gtx", "g03": "g2003_conus.gtx",
        "geoid99": "g1999_conus.gtx", "g99": "g1999_conus.gtx",
        "egm08": "egm08_25.gtx", "geoid18alaska": "us_nga_geoid18_ak.tif",
        "g18ak": "us_nga_geoid18_ak.tif", "geoid12balaska": "g2012b_ak.gtx",
        "g12bak": "g2012b_ak.gtx", "geoid09alaska": "g2009_ak.gtx",
        "g09ak": "g2009_ak.gtx", "geoid12bhawaii": "g2012b_hi.gtx",
        "g12bhi": "g2012b_hi.gtx", "geoid09hawaii": "g2009_hi.gtx",
        "g09hi": "g2009_hi.gtx", "geoid18puertorico": "us_nga_geoid18_prvi.tif",
        "g18prvi": "us_nga_geoid18_prvi.tif", "geoid12bpuertorico": "g2012b_prvi.gtx",
        "g12bprvi": "g2012b_prvi.gtx", "geoid18guam": "us_nga_geoid18_guamnmi.tif",
        "g18guam": "us_nga_geoid18_guamnmi.tif", "geoid18samoa": "us_nga_geoid18_samoa.tif",
        "g18samoa": "us_nga_geoid18_samoa.tif", "vertcon": "vertcon_conus.gtx",
        "vertconalaska": "vertcon_ak.gtx", "ausgeoid20": "au_ga_AUSGeoid2020.gtx",
        "ausgeoid9": "au_ga_AUSGeoid09.gtx", "ukgeoid": "uk_osgm15.gtx",
        "britaingeoid": "uk_osgm15.gtx", "germanygeoid": "de_bkg_GCG2016.gtx",
        "francegeoid": "fr_ign_RAF20.gtx", "switzerlandgeoid": "ch_ln02_CHGeo2004.gtx",
        "spaingeoid": "es_ign_EGM08_REDNAP.gtx", "italygeoid": "it_gm_igmi2004.gtx",
        "japangeoid": "jp_gsi_gsigeo2011.gtx", "newzealandgeoid": "nz_linz_nzgeoid2016.gtx",
        "brazilgeoid": "br_ibge_MAPGEO2015.gtx"
    }

    def __init__(
        self,
        pair: RasterPair,
        compare_vcrs: str,
        reference_vcrs: str,
        *,
        compare_geoid: Optional[str] = None,
        reference_geoid: Optional[str] = None,
        out_dir: str | Path = "./vert_transformed",
    ):
        self.pair = pair
        self.compare_vcrs = compare_vcrs
        self.reference_vcrs = reference_vcrs
        self.compare_geoid = compare_geoid
        self.reference_geoid = reference_geoid
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.hcrs = self.pair.raster1.crs

    def _get_grid_filename(self, geoid_name: Optional[str]) -> Optional[str]:
        """Parses a user-friendly geoid name into a PROJ-compatible grid filename."""
        if geoid_name is None:
            return None
            
        # CORRECTED NORMALIZATION: Lowercase and remove spaces/underscores.
        norm_name = geoid_name.lower().replace(" ", "").replace("_", "")
        
        if norm_name in self._GEOID_MAP:
            return self._GEOID_MAP[norm_name]
        
        # Fallback for keys that might have spaces/underscores in them originally
        if geoid_name.lower() in self._GEOID_MAP:
            return self._GEOID_MAP[geoid_name.lower()]
            
        warnings.warn(
            f"⚠️ Geoid model '{geoid_name}' not found in the known list. "
            f"Assuming it is a valid PROJ grid filename."
        )
        return geoid_name

    def transform(self) -> RasterPair:
        """Performs the vertical transformation on the compare raster."""
        self.pair._align()

        if (self.compare_vcrs == self.reference_vcrs and
                self.compare_geoid == self.reference_geoid):
            print("✅ Source and target vertical systems are identical. No transformation needed.")
            return self.pair

        print("🚀 Starting vertical transformation...")

        # Get the full path for the output file
        out_path = self.out_dir / f"{self.pair.raster1.path.stem}_vtrans.tif"

        # --- Get the source and target CRS objects ---
        source_crs = CRS(self.pair.raster1.crs)
        # We assume the target HCRS is the same, but we define the target VCRS
        target_vcrs_wkt = CRS(self.reference_vcrs).to_wkt()
        
        # Build a compound WKT for the target CRS
        target_crs_wkt = (
            f'COMPOUNDCRS["{source_crs.name} + {CRS(self.reference_vcrs).name}",'
            f'{source_crs.to_wkt()},'
            f'{target_vcrs_wkt}]'
        )

        print(f"   - Transforming from VCRS: {self.compare_vcrs}")
        print(f"   - Transforming to VCRS:   {self.reference_vcrs}")

        # --- Use gdal.Warp for robust transformation ---
        gdal.Warp(
            destNameOrDestDS=str(out_path),
            srcDSOrSrcDSTab=str(self.pair.raster1.path),
            dstSRS=target_crs_wkt, # Target CRS
            # Let GDAL handle the source CRS detection from the file
        )

        print(f"✔️ Transformation complete. New raster saved to: {out_path}")
        return RasterPair(raster1=Raster(out_path), raster2=self.pair.raster2)

    # def _save_and_return_new_pair(self, transformed_da: rio.DataArray) -> RasterPair:
    #     """Saves a transformed DataArray and returns a new RasterPair."""
    #     out_path = self.out_dir / f"{self.pair.raster1.path.stem}_vtrans.tif"
    #     transformed_da.rio.to_raster(out_path)
    #     print(f"✔️ Transformation complete. New raster saved to: {out_path}")
    #     return RasterPair(raster1=Raster(out_path), raster2=self.pair.raster2)
# ----------------------------------------------------------------------
# Terrain derivatives                    
# ----------------------------------------------------------------------
class TerrainDerivatives:
    """Compute hillshade, slope, aspect, and roughness via GDAL DEMProcessing."""

    def __init__(self, out_dir: Path | str):
        """
        Initialize a directory for terrain derivative outputs.

        Parameters
        ----------
        out_dir : str or pathlib.Path
            Directory where derived raster products will be written.  The
            directory is created if it does not already exist.
        """
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _process(self, src: Path, name: str, mode: str, options=None, **kwargs) -> Path:
        """
        Internal helper to run GDAL DEMProcessing on a source DEM.

        Parameters
        ----------
        src : pathlib.Path
            Path to the source DEM.
        name : str
            Filename of the output raster relative to ``out_dir``.
        mode : str
            Processing mode passed to GDAL (e.g. ``'hillshade'``).
        options : gdal.DEMProcessingOptions, optional
            GDAL options controlling the processing (may be ``None``).
        **kwargs : dict
            Additional keyword arguments forwarded to ``gdal.DEMProcessing``.

        Returns
        -------
        pathlib.Path
            Path to the generated raster.  If the file already exists,
            processing is skipped.
        """
        dst = self.out_dir / name
        if not dst.exists():
            if options is not None:
                # only pass options if you actually have one
                gdal.DEMProcessing(str(dst), str(src), mode, options=options, **kwargs)
            else:
                # otherwise call without the options kwarg
                gdal.DEMProcessing(str(dst), str(src), mode, **kwargs)
        return dst

    def hillshade(self, dem: Path, azimuth: float = 315, altitude: float = 45) -> Path:
        """
        Compute a hillshade raster from a DEM.

        Parameters
        ----------
        dem : pathlib.Path
            Path to the input DEM.
        azimuth : float, default 315
            Sun azimuth angle in degrees.
        altitude : float, default 45
            Sun altitude angle in degrees.

        Returns
        -------
        pathlib.Path
            Path to the generated hillshade raster.
        """
        opts = gdal.DEMProcessingOptions(azimuth=azimuth, altitude=altitude)
        return self._process(dem, f"{Path(dem).stem}_hillshade.tif", "hillshade", options=opts)

    def slope(self, dem: Path) -> Path:
        """
        Derive a slope raster from a DEM.

        Parameters
        ----------
        dem : pathlib.Path
            Path to the input DEM.

        Returns
        -------
        pathlib.Path
            Path to the generated slope raster.
        """
        return self._process(dem, f"{Path(dem).stem}_slope.tif", "slope")

    def aspect(self, dem: Path) -> Path:
        """
        Derive an aspect raster from a DEM.

        Parameters
        ----------
        dem : pathlib.Path
            Path to the input DEM.

        Returns
        -------
        pathlib.Path
            Path to the generated aspect raster.  In very old versions of
            GDAL, the ``zeroForFlat`` option may not be available, in which
            case a warning is emitted and default behaviour is used.
        """
        try:
            opts = gdal.DEMProcessingOptions(zeroForFlat=True)
        except TypeError:  # very old GDAL
            warnings.warn("GDAL < 3.4 lacks zeroForFlat – flat areas will be 0")
            opts = None
        return self._process(dem, f"{Path(dem).stem}_aspect.tif", "aspect", options=opts)

    def roughness(self, dem: Path) -> Path:
        """
        Compute terrain roughness from a DEM.

        Parameters
        ----------
        dem : pathlib.Path
            Path to the input DEM.

        Returns
        -------
        pathlib.Path
            Path to the generated roughness raster.
        """
        return self._process(dem, f"{Path(dem).stem}_roughness.tif", "roughness")

# ----------------------------------------------------------------------
# TopoDifferencer                                                
# ----------------------------------------------------------------------
class TopoDifferencer:
    """Compute raster differences, NoData masks, and export/plot results."""

    def __init__(self, out_dir: Path | str):
        """
        Initialize an object to compute and visualize raster differences.

        Parameters
        ----------
        out_dir : str or pathlib.Path
            Directory where difference rasters and plots will be saved.
            The directory is created if it does not already exist.
        """
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _sym_range(arr: np.ndarray):
        """
        Compute symmetric colour bounds around zero for visualising differences.

        Given an array of signed differences, this helper finds the maximum
        absolute value and returns a tuple ``(-absmax, absmax)`` so that
        zero is centred in the colourmap.

        Parameters
        ----------
        arr : numpy.ndarray
            Array of numeric values, potentially containing NaNs.

        Returns
        -------
        tuple of float
            Minimum and maximum bounds symmetric about zero.
        """
        absmax = np.nanmax(np.abs(arr))
        return -absmax, absmax

    @staticmethod
    def _extent(raster: "Raster"):
        """
        Retrieve map extent from a :class:`Raster`.

        Parameters
        ----------
        raster : Raster
            Input raster wrapper from which to compute bounds.

        Returns
        -------
        list of float
            A list ``[left, right, bottom, top]`` representing the extent of
            the raster in its native CRS.
        """
        left, bottom, right, top = rasterio.open(raster.path).bounds
        return [left, right, bottom, top]

    def difference_da(self, pair: RasterPair):
        """Return an xarray DataArray of raster2 − raster1 (masked)."""
        return (pair.raster2_data - pair.raster1_data).compute()

    def save_difference_raster(self, pair: RasterPair, filename: str) -> Raster:
        """
        Compute and save a difference raster (raster2 minus raster1).

        Parameters
        ----------
        pair : RasterPair
            The aligned pair of rasters to be differenced.
        filename : str
            Base filename for the output raster (without extension).

        Returns
        -------
        Raster
            A :class:`Raster` pointing to the saved difference raster on disk.
        """
        diff_da = self.difference_da(pair)
        out = self.out_dir / f"{filename}.tif"
        diff_da.rio.to_raster(out)
        return Raster(out)

    def combined_mask(
        self,
        *,
        pair: RasterPair | None = None,
        a: Path | str | None = None,
        b: Path | str | None = None,
        name: str = "mask.tif",
    ) -> Raster:
        """
        Generate a combined NoData mask for a raster pair.

        This method produces a binary raster where pixels are set to 1 if
        either input raster contains NoData at that location.  The resulting
        mask is useful for consistent masking when visualising differences.

        Parameters
        ----------
        pair : RasterPair, optional
            Pre-aligned pair of rasters.  If provided, ``a`` and ``b``
            must be ``None``.
        a, b : str or pathlib.Path, optional
            Paths to individual raster files.  If ``pair`` is not given,
            both ``a`` and ``b`` must be provided.  The second raster
            will be reprojected to match the first if necessary.
        name : str, default 'mask.tif'
            Filename for the output mask within ``out_dir``.

        Returns
        -------
        Raster
            A :class:`Raster` representing the combined mask on disk.

        Raises
        ------
        ValueError
            If inputs are not provided correctly.
        """
        if pair is not None and (a or b):
            raise ValueError("Provide either 'pair' or 'a' & 'b', not both")
        if pair is None and None in (a, b):
            raise ValueError("Need both 'a' and 'b' paths when pair is not given")
        if pair is not None:
            r1, r2 = pair.raster1, pair.raster2
        else:
            r1, r2 = Raster(a), Raster(b)
            if r1.shape != r2.shape or r1.crs != r2.crs:
                r2 = r2.reproject_to(r1)
        mask = np.logical_or(r1.data.mask.squeeze(), r2.data.mask.squeeze())
        with rasterio.open(r1.path) as src:
            meta = src.meta.copy()
        meta.update(dtype="uint8", count=1, nodata=0)
        out = self.out_dir / name
        with rasterio.open(out, "w", **meta) as dst:
            dst.write(mask.astype("uint8"), 1)
        return Raster(out)

    def plot_difference(
        self,
        *,
        pair: RasterPair | None = None,
        diff_path: Path | str | None = None,
        overlay: Raster | None = None,
        mask_overlay: bool = True,
        cmap="RdBu_r",
        vmin: float | None = None,
        vmax: float | None = None,
        center_zero: bool = True,
        overlay_alpha: float = 0.4,
        title: str = "Difference",
        save_path: Path | str | None = None,
        dpi: int = 300,
    ) -> plt.Figure:
        """Plot (and optionally save) a difference raster with hillshade overlay.

        If *overlay* is provided it is drawn in grayscale beneath the diff map
        (alpha-blended). A combined NoData mask is applied when
        *mask_overlay* is True so only pixels valid in **both** rasters are
        shown.
        """
        if (pair is None) == (diff_path is None):
            raise ValueError("Provide exactly one of 'pair' or 'diff_path'")

        # Obtain diff DataArray and drop singleton band dimension
        if pair is not None:
            diff_da = self.difference_da(pair)
            diff_da = diff_da.squeeze(dim=[d for d in diff_da.dims if diff_da[d].size == 1], drop=True)
            base = pair.raster1
        else:
            diff_r = Raster(diff_path)
            # raster data has shape (bands, y, x) or (y, x)
            diff_da = diff_r.data.squeeze()
            base = diff_r

        # Convert to numpy and ensure 2D
        diff_arr = np.squeeze(diff_da.values)
        extent = self._extent(base)

        # Determine colour limits
        if center_zero and vmin is None and vmax is None:
            vmin, vmax = self._sym_range(diff_arr)

        # Prepare overlay alignment & masking
        if overlay is not None:
            ov = overlay
            if ov.shape[1:] != base.shape[1:] or ov.crs != base.crs:
                ov = ov.reproject_to(base)
            ov_arr = np.squeeze(ov.data.values)
            if mask_overlay:
                valid = ~np.logical_or(
                    np.isnan(diff_arr), np.isnan(ov_arr)
                )
                diff_arr = np.where(valid, diff_arr, np.nan)
                ov_arr = np.where(valid, ov_arr, np.nan)
        else:
            ov_arr = None

        # --- plotting ---
        fig, ax = plt.subplots(figsize=(10, 6))
        cmap_obj = plt.get_cmap(cmap)
        cmap_obj.set_bad(color="none")
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        im = ax.imshow(diff_arr, cmap=cmap_obj, norm=norm, extent=extent)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if ov_arr is not None:
            shade_cmap = plt.get_cmap("gray")
            shade_cmap.set_bad(color="none")
            ax.imshow(ov_arr, cmap=shade_cmap, alpha=overlay_alpha, extent=extent)
        ax.axis("off")
        ax.set_title(title)

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
        return fig
    
    from pathlib import Path


def get_colormap_bounds(arr: np.ndarray) -> tuple[float, float]:
    """
    Compute symmetric color bounds around zero for a difference array.
    """
    valid = ~np.isnan(arr)
    if not np.any(valid):
        return 0.0, 0.0
    v = np.abs(arr[valid])
    m = v.max()
    return -m, m


def reproject_for_map(src_path: Path | str, dst_path: Path | str, dst_crs: str = "EPSG:3857"):
    """
    Reproject a raster from its current CRS to a destination CRS.
    """
    with rio.open_rasterio(src_path, masked=True) as src:
        da = src.rio.reproject(dst_crs)
        da.rio.to_raster(dst_path)
        

