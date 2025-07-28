"""Utility classes for defining and analyzing stable and unstable areas in
topographic differencing workflows.

This module provides interactive mapping widgets for drawing polygons on
topographic difference rasters, functions for rasterizing those polygons
against a raster mask, and helpers for computing descriptive statistics
within stable or unstable regions.  The classes exposed here are designed
to be used in a Jupyter notebook environment and build upon the ``Raster``
class defined in ``differencing_functions_9.py``.
"""

from pathlib import Path
import statistics
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
import geopandas as gpd
from shapely.geometry import shape, mapping, Polygon
from shapely.ops import transform as shapely_transform
from pyproj import Transformer
import matplotlib.pyplot as plt
from ipyleaflet import Map, DrawControl, ImageOverlay, GeoJSON, LegendControl, WidgetControl
from ipywidgets import Button, HBox, Label
from scipy import stats

from differencing_functions import Raster

class TopoMapInteractor:
    """
    Interactive map for drawing 'stable' and 'unstable' polygons on a topo-difference raster,
    with pixel-count utility, two-layer legend, and labeled draw buttons.
    """
    def __init__(
        self,
        topo_diff_path: Path | str,
        hillshade_path: Path | str,
        output_dir: Path | str,
        zoom: int = 15,
        map_size: tuple[str, str] = ('800px', '1300px'),
        overlay_cmap: str = 'bwr_r',
        overlay_dpi: int = 300
    ):
        """Initialize an interactive map for delineating stable and unstable areas.

        Parameters
        ----------
        topo_diff_path : Path or str
            Path to the topographic difference raster that will be displayed as
            an overlay.  A ``Raster`` object is constructed lazily from this
            path.
        hillshade_path : Path or str
            Path to a hillshade raster used to provide terrain context on the
            map.
        output_dir : Path or str
            Directory where temporary overlay images and polygon shapefiles
            should be written.
        zoom : int, optional
            Initial zoom level for the map (default is 15).
        map_size : tuple of str, optional
            Height and width of the map widget expressed as CSS strings
            (default is ``('800px', '1300px')``).
        overlay_cmap : str, optional
            Colormap used to display the topographic difference values
            (default is ``'bwr_r'``).
        overlay_dpi : int, optional
            Resolution in dots per inch for the overlay PNG that is created
            for the map (default is 300).

        Notes
        -----
        When instantiated, the interactor will read the provided rasters,
        generate an image overlay, configure drawing controls for stable and
        unstable polygons, and attach them to an ``ipyleaflet`` map.  Drawn
        polygons are stored internally for later rasterization and analysis.
        """
        # Load rasters
        self.topo_diff = Raster(topo_diff_path)
        self.hillshade = Raster(hillshade_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Storage for geometries
        self.stable_geoms: list[Polygon] = []
        self.unstable_geoms: list[Polygon] = []
        self.current_category: str | None = None

        # Compute lat/lon bounds
        with rasterio.open(self.topo_diff.path) as ds:
            bounds = ds.bounds
            crs = ds.crs
        transformer = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)
        west, south = transformer.transform(bounds.left, bounds.bottom)
        east, north = transformer.transform(bounds.right, bounds.top)
        self.latlon_bounds = ((south, west), (north, east))

        # Generate overlay PNG
        png_path = self.output_dir / f"{Path(self.topo_diff.path).stem}.png"
        self._generate_overlay_png(png_path, cmap=overlay_cmap, dpi=overlay_dpi)

        # Initialize map
        center = ((north + south) / 2, (west + east) / 2)
        self.map = Map(
            center=center,
            zoom=zoom,
            layout={'height': map_size[0], 'width': map_size[1]}
        )

        # Add image overlay
        self.map.add_layer(ImageOverlay(url=str(png_path), bounds=self.latlon_bounds))

        # Legend
        legend_dict = {'Stable Area': 'green', 'Feature of Interest': 'red'}
        self.map.add_control(LegendControl(legend_dict, title='Legend'))

        # GeoJSON layers
        self.geojson_stable = GeoJSON(data={"type": "FeatureCollection", "features": []},
                                     style={"color": "green", "fillColor": "green", "fillOpacity": 0.3})
        self.geojson_unstable = GeoJSON(data={"type": "FeatureCollection", "features": []},
                                       style={"color": "red", "fillColor": "red", "fillOpacity": 0.3})
        self.map.add_layer(self.geojson_stable)
        self.map.add_layer(self.geojson_unstable)

        # Single DrawControl reused
        self.draw_control = DrawControl(
            polygon={"shapeOptions": {"weight": 2, "fillOpacity": 0.3}},
        )
        # disable other shapes
        for attr in ('circle', 'circlemarker', 'polyline', 'rectangle'):
            setattr(self.draw_control, attr, {})
        self.draw_control.on_draw(self._handle_draw)

        # Create labeled buttons
        self.btn_stable = Button(description='Stable', layout={'width': '80px'})
        self.btn_unstable = Button(description='Unstable', layout={'width': '80px'})
        # style colors
        self.btn_stable.style.button_color = 'lightgreen'
        self.btn_unstable.style.button_color = 'lightcoral'

        # Button callbacks
        self.btn_stable.on_click(lambda _: self._activate_category('stable'))
        self.btn_unstable.on_click(lambda _: self._activate_category('unstable'))

        # Place buttons above map
        btn_box = HBox([Label(' Draw mode:'), self.btn_stable, self.btn_unstable])
        self.map.add_control(WidgetControl(widget=btn_box, position='topright'))
        
        
    def _generate_overlay_png(self, png_path, cmap='bwr_r', dpi=300):
        """Render the topographic difference raster to a PNG for map display.

        The raster values are plotted with a symmetric colormap (centred
        around zero) and saved to the specified PNG path.  Pixels with
        ``NaN`` values are rendered as transparent.

        Parameters
        ----------
        png_path : Path or str
            Destination filename for the generated PNG.
        cmap : str, optional
            Name of the matplotlib colormap to use (default is ``'bwr_r'``).
        dpi : int, optional
            Resolution of the saved image in dots per inch (default is 300).
        """
        arr = np.squeeze(self.topo_diff.data.values)
        vmin, vmax = self._get_sym_bounds(arr)
        fig, ax = plt.subplots(frameon=False)
        ax.axis('off')
        cmap_obj = plt.get_cmap(cmap); cmap_obj.set_bad(color='none')
        ax.imshow(arr, cmap=cmap_obj, vmin=vmin, vmax=vmax)
        fig.savefig(png_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    @staticmethod
    def _get_sym_bounds(arr: np.ndarray) -> tuple[float, float]:
        """Compute symmetric bounds for diverging colormaps.

        Given a numeric array, this helper computes the negative and positive
        extremes (ignoring NaNs) such that the colour scale is symmetric
        about zero.  If the array contains no valid values, ``(-0.0, 0.0)``
        is returned.

        Parameters
        ----------
        arr : numpy.ndarray
            The input array of values.

        Returns
        -------
        tuple of float
            Minimum and maximum values for a symmetric colour scale.
        """
        flat = arr.flatten(); valid = flat[~np.isnan(flat)]
        return (-0.0, 0.0) if valid.size == 0 else (-np.max(np.abs(valid)), np.max(np.abs(valid)))

    def _activate_category(self, category: str):
        """Enable drawing for a specified polygon category.

        This sets the active category (either ``'stable'`` or ``'unstable'``)
        so that newly drawn polygons are assigned accordingly.  The draw
        control is added to the map if it is not already present, and the
        button styles are updated to reflect the active selection.

        Parameters
        ----------
        category : str
            The category name, either ``'stable'`` or ``'unstable'``.
        """
        self.current_category = category

        if self.draw_control not in self.map.controls:
            self.map.add_control(self.draw_control)

        # highlight active button
        self.btn_stable.style.font_weight = 'bold' if category == 'stable' else 'normal'
        self.btn_unstable.style.font_weight = 'bold' if category == 'unstable' else 'normal'

    def _handle_draw(self, control, action, geo_json):
        """Handle polygon drawing events and store them in the appropriate list.

        When the user finishes drawing a polygon on the map, this callback
        checks the current drawing category and appends the resulting
        geometry to either ``stable_geoms`` or ``unstable_geoms``.  The
        corresponding ``GeoJSON`` overlay is refreshed to display the newly
        drawn polygon.  After a polygon is accepted, the drawing mode
        automatically deactivates until the user clicks a category button
        again.

        Parameters
        ----------
        control : DrawControl
            The draw control emitting the event.
        action : str
            The type of draw action (we only process ``'created'``).
        geo_json : dict
            The GeoJSON representation of the drawn feature.
        """
        if action != 'created' or geo_json['geometry']['type'] != 'Polygon':
            return
            
        geom = shape(geo_json['geometry'])
        if self.current_category == 'stable':
            self.stable_geoms.append(geom)
            feats = [mapping(g) for g in self.stable_geoms]
            self.geojson_stable.data = {"type": "FeatureCollection", "features": feats}
        elif self.current_category == 'unstable':
            self.unstable_geoms.append(geom)
            feats = [mapping(g) for g in self.unstable_geoms]
            self.geojson_unstable.data = {"type": "FeatureCollection", "features": feats}

        self.map.remove_control(self.draw_control)
        self.current_category = None
        self.btn_stable.style.font_weight = 'normal'
        self.btn_unstable.style.font_weight = 'normal'

    def calculate_pixel_count(self, polygon: Polygon) -> int:
        """Count valid (non-nodata) raster pixels within a polygon.

        The input polygon is assumed to be defined in geographic
        coordinates (latitude and longitude).  It is reprojected into
        the raster's native coordinate reference system prior to
        masking.  Only pixels that are neither ``NaN`` nor equal to the
        ``nodata`` value are counted.

        Parameters
        ----------
        polygon : shapely.geometry.Polygon
            The polygon delineating the area over which to count valid
            pixels.

        Returns
        -------
        int
            The number of valid raster pixels intersecting the polygon.
        """
        with rasterio.open(self.topo_diff.path) as src:
            dst_crs = src.crs
            # transform polygon from EPSG:4326 to raster CRS
            transformer = Transformer.from_crs('EPSG:4326', dst_crs, always_xy=True)
            poly_proj = shapely_transform(transformer.transform, polygon)
            # mask with projected polygon
            out_img, _ = rasterio.mask.mask(src, [mapping(poly_proj)], crop=True)
            data = out_img[0]; nodata = src.nodata
            valid = (~np.isnan(data)) & (data != nodata)
            return int(np.sum(valid))

    def get_geodataframes(self) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """Return GeoDataFrames of the stable and unstable polygons.

        Two ``geopandas.GeoDataFrame`` objects are returned, each in the
        raster's coordinate reference system.  Each GeoDataFrame
        contains a ``'Name'`` column (e.g., ``'stable1'``, ``'unstable2'``),
        a ``'geometry'`` column with the polygon geometries, and a
        ``'pixels'`` column giving the count of valid raster cells
        intersecting each polygon.

        Returns
        -------
        tuple of GeoDataFrame
            A tuple ``(gdf_stable, gdf_unstable)`` of stable and unstable
            polygon data.
        """
        # Build GeoDataFrames in lat/lon CRS
        gdf_stable = gpd.GeoDataFrame(
            {'Name': [f'stable{i+1}' for i in range(len(self.stable_geoms))],
             'geometry': self.stable_geoms,
             'pixels': [self.calculate_pixel_count(g) for g in self.stable_geoms]
             },
            crs='EPSG:4326'
        )
        gdf_unstable = gpd.GeoDataFrame(
            {'Name': [f'unstable{i+1}' for i in range(len(self.unstable_geoms))],
             'geometry': self.unstable_geoms,
             'pixels': [self.calculate_pixel_count(g) for g in self.unstable_geoms]
             },
            crs='EPSG:4326'
        )
        # Reproject to raster CRS
        with rasterio.open(self.topo_diff.path) as src:
            raster_crs = src.crs
        gdf_stable = gdf_stable.to_crs(raster_crs)
        gdf_unstable = gdf_unstable.to_crs(raster_crs)
        return gdf_stable, gdf_unstable

def descriptive_stats(values: np.ndarray) -> pd.DataFrame:
    """
    Compute descriptive statistics for a 1D array of values.
    Returns a one-row DataFrame.
    """
    data = values[~np.isnan(values)]
    if data.size == 0:
        # return empty stats row
        cols = ['mean','median','mode','std','variance','min','max',
                'skewness','kurtosis','0.5_percentile','99.5_percentile']
        return pd.DataFrame([{c: np.nan for c in cols}])
    mean = np.mean(data)
    median = np.median(data)
    try:
        mode = statistics.mode(data)
    except statistics.StatisticsError:
        mode = np.nan
    std = np.std(data)
    var = np.var(data)
    minimum = np.min(data)
    maximum = np.max(data)
    skew = stats.skew(data)
    kurt = stats.kurtosis(data)
    p1, p99 = np.percentile(data, [0.5, 99.5])
    return pd.DataFrame([{
        'mean': mean,
        'median': median,
        'mode': mode,
        'std': std,
        'variance': var,
        'min': minimum,
        'max': maximum,
        'skewness': skew,
        'kurtosis': kurt,
        '0.5_percentile': p1,
        '99.5_percentile': p99
    }])

class StableAreaRasterizer:
    """
    Rasterize stable-area polygons to mask a topographic-difference raster:
    - `rasterize_all`: one output where inside polygons = original values, outside = nodata.
    - `rasterize_each`: separate rasters per polygon.
    """
    def __init__(self, topo_diff_path: Path | str, stable_gdf, nodata: float = -9999):
        """Prepare a rasterizer for stable-area polygons.

        This helper class takes a topographic-difference raster and a
        GeoDataFrame of stable polygons and provides methods to
        rasterize those polygons either all at once or individually.

        Parameters
        ----------
        topo_diff_path : Path or str
            Path to the topographic-difference raster file.
        stable_gdf : geopandas.GeoDataFrame
            GeoDataFrame containing polygons that delineate the stable
            areas.  This DataFrame should be in geographic (lat/lon)
            coordinates.
        nodata : float, optional
            The value that should be written outside polygon areas in
            the output rasters (default is ``-9999``).
        """
        self.topo_path = Path(topo_diff_path)
        self.gdf = stable_gdf.copy()
        self.nodata = nodata

    def rasterize_all(self, output_path: Path | str) -> Path:
        """Rasterize all stable polygons into one mask.

        A new raster is written to ``output_path`` where all cells that
        fall inside any of the stable-area polygons retain their
        original values from the input raster, and all cells outside
        those polygons are set to the ``nodata`` value specified when
        constructing this object.

        Parameters
        ----------
        output_path : Path or str
            Where to write the combined stable-area raster.

        Returns
        -------
        Path
            The path to the written GeoTIFF file.
        """
        out_path = Path(output_path)
        with rasterio.open(self.topo_path) as src:
            profile = src.profile.copy()
            profile.update(nodata=self.nodata)
            data = src.read(1)
            # rasterize mask of polygons
            mask = rasterize(
                [(geom, 1) for geom in self.gdf.geometry],
                out_shape=src.shape,
                transform=src.transform,
                fill=0,
                dtype='uint8'
            )
            # apply mask
            out = np.where(mask == 1, data, self.nodata)
            # write
            with rasterio.open(out_path, 'w', **profile) as dst:
                dst.write(out, 1)
        return out_path

    def rasterize_each(self, output_dir: Path | str) -> dict[int, Path]:
        """Rasterize each stable polygon to its own GeoTIFF.

        Each polygon in the GeoDataFrame is rasterized individually,
        preserving the original topographic-difference values within the
        polygon and writing ``nodata`` elsewhere.  The rasters are
        written to ``output_dir`` with filenames like
        ``stable_area_0.tif``, ``stable_area_1.tif``, etc.

        Parameters
        ----------
        output_dir : Path or str
            Directory where the individual rasters should be saved.

        Returns
        -------
        dict
            A dictionary mapping the polygon index to the path of the
            corresponding raster file.
        """
        outdir = Path(output_dir)
        outdir.mkdir(exist_ok=True, parents=True)
        paths = {}
        with rasterio.open(self.topo_path) as src:
            profile = src.profile.copy()
            profile.update(nodata=self.nodata)
            data = src.read(1)
            for idx, row in self.gdf.iterrows():
                geom = row.geometry
                mask = rasterize(
                    [(geom, 1)],
                    out_shape=src.shape,
                    transform=src.transform,
                    fill=0,
                    dtype='uint8'
                )
                out = np.where(mask == 1, data, self.nodata)
                path = outdir / f"stable_area_{idx}.tif"
                with rasterio.open(path, 'w', **profile) as dst:
                    dst.write(out, 1)
                paths[idx] = path
        return paths

class StableAreaAnalyzer:
    """
    Use rasters produced by StableAreaRasterizer to compute descriptive stats:
    - `stats_all`: stats on combined-area raster.
    - `stats_each`: stats on each individual-area raster.
    """
    def __init__(self, rasterizer: StableAreaRasterizer):
        """Analyse rasters produced by a ``StableAreaRasterizer``.

        This class provides helper methods to compute descriptive
        statistics for rasters that represent the stable areas defined
        by the user.  It works with both a combined mask and
        per-polygon rasters.

        Parameters
        ----------
        rasterizer : StableAreaRasterizer
            The rasterizer instance used to generate the stable-area
            rasters.
        """
        self.rasterizer = rasterizer

    def _stats_from_raster(self, path: Path | str) -> pd.DataFrame:
        """Calculate descriptive statistics for a single raster file.

        This internal helper reads a raster file, replaces the
        ``nodata`` values with ``NaN``, flattens the array and
        delegates to :func:`descriptive_stats` to compute a
        one-row DataFrame of descriptive measures.

        Parameters
        ----------
        path : Path or str
            Path to the raster from which to compute statistics.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing summary statistics for the raster values.
        """
        with rasterio.open(path) as src:
            arr = src.read(1)
            arr = np.where(arr == src.nodata, np.nan, arr)
            flat = arr.ravel()
        return descriptive_stats(flat)

    def stats_all(self, output_path: Path | str) -> pd.DataFrame:
        """Compute statistics for all stable polygons combined.

        This method calls ``rasterize_all`` on the underlying
        ``StableAreaRasterizer``, writes the combined raster to
        ``output_path``, then computes descriptive statistics on all
        valid cells in that raster.  The resulting DataFrame is
        indexed with a single row labelled ``'all_areas'``.

        Parameters
        ----------
        output_path : Path or str
            Path where the combined raster will be written.

        Returns
        -------
        pandas.DataFrame
            A one-row DataFrame containing descriptive statistics.
        """
        out = self.rasterizer.rasterize_all(output_path)
        df = self._stats_from_raster(out)
        df.index = ['all_areas']
        return df

    def stats_each(self, output_dir: Path | str) -> pd.DataFrame:
        """Compute statistics for each stable polygon individually.

        Each polygon is rasterized to its own file in ``output_dir``.
        Descriptive statistics are then computed for each raster and
        returned in a DataFrame indexed by the polygon ID.

        Parameters
        ----------
        output_dir : Path or str
            Directory where individual rasters will be written.

        Returns
        -------
        pandas.DataFrame
            A DataFrame where each row corresponds to a stable-area raster
            and the index is the polygon ID.
        """
        paths = self.rasterizer.rasterize_each(output_dir)
        records = []
        for area_id, path in paths.items():
            df = self._stats_from_raster(path)
            df['area_id'] = area_id
            records.append(df)
        result = pd.concat(records, ignore_index=True).set_index('area_id')
        return result
