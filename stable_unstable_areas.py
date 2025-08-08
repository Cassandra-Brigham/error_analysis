"""Utility classes for defining and analyzing stable and unstable areas in
topographic differencing workflows.

This module provides interactive mapping widgets for drawing polygons on
topographic difference rasters, functions for rasterizing those polygons
against a raster mask, and helpers for computing descriptive statistics
within stable or unstable regions. The classes exposed here are designed
to be used in a Jupyter notebook environment and build upon the ``Raster``
class defined in ``differencing_functions.py``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple, List, Optional, Iterable
import uuid
import statistics

import numpy as np
import pandas as pd
import rasterio
import rasterio.features as rfeatures
from rasterio.features import rasterize
import rasterio.mask
import geopandas as gpd
from shapely.geometry import shape, mapping, Polygon, MultiPolygon
from shapely.ops import transform as shapely_transform, unary_union
from pyproj import Transformer
import matplotlib.pyplot as plt
from ipyleaflet import Map, DrawControl, ImageOverlay, GeoJSON, WidgetControl
try:
    from ipyleaflet import LegendControl
    HAS_LEGEND = True
except Exception:
    HAS_LEGEND = False
from ipywidgets import Button, HBox, Label
from scipy import stats

from differencing_functions import Raster


class TopoMapInteractor:
    """
    Interactive map for drawing 'stable' and 'unstable' polygons on a topo-difference raster,
    with pixel-count utility, two-layer legend, labeled draw buttons, file loading, and
    optional auto-derivation of Stable = (valid raster) − (union of FOIs).
    """
    def __init__(
        self,
        topo_diff_path: Path | str,
        hillshade_path: Path | str,
        output_dir: Path | str,
        zoom: int = 15,
        map_size: tuple[str, str] = ('800px', '1300px'),
        overlay_cmap: str = 'bwr_r',
        overlay_dpi: int = 300,
        overlay_vmin: Optional[float] = None,
        overlay_vmax: Optional[float] = None,
        stable_path: Optional[Path | str] = None,
        unstable_path: Optional[Path | str] = None,
        stable_name_field: Optional[str] = None,
        unstable_name_field: Optional[str] = None,
        assume_input_crs: Optional[str] = 'EPSG:4326',
        auto_stable_from_unstable: bool = True,
        derive_min_area: Optional[float] = None,
        simplify_tolerance: Optional[float] = None,
    ):
        """
        Parameters
        ----------
        topo_diff_path, hillshade_path, output_dir : Path-like
            Input rasters and output directory for overlay PNGs.
        zoom, map_size : map appearance.
        overlay_cmap, overlay_dpi, overlay_vmin, overlay_vmax : overlay styling.
            If only one of vmin/vmax is provided, the other is inferred symmetrically.
        stable_path, unstable_path : Path-like or None
            Optional polygon files to preload (shapefile/.zip, GeoPackage, GeoJSON).
        stable_name_field, unstable_name_field : str or None
            Optional attribute name for feature labels when loading files.
        assume_input_crs : str or None
            CRS to assume when an input vector file lacks a CRS (default 'EPSG:4326').
        auto_stable_from_unstable : bool
            If True, automatically set Stable = valid raster minus FOIs whenever FOIs change.
        derive_min_area : float or None
            Minimum polygon area (in raster CRS units²) to keep when deriving Stable.
        simplify_tolerance : float or None
            Optional simplification tolerance (in raster CRS units) when deriving Stable.
        """
        # Load rasters
        self.topo_diff = Raster(topo_diff_path)
        self.hillshade = Raster(hillshade_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Storage for geometries and names
        self.stable_geoms: List[Polygon] = []
        self.unstable_geoms: List[Polygon] = []
        self.stable_names: List[str] = []
        self.unstable_names: List[str] = []
        self.current_category: Optional[str] = None

        # Auto-derive settings
        self.auto_stable_from_unstable = auto_stable_from_unstable
        self.derive_min_area = derive_min_area
        self.simplify_tolerance = simplify_tolerance

        # Compute lat/lon bounds
        with rasterio.open(self.topo_diff.path) as ds:
            bounds = ds.bounds
            crs = ds.crs
        transformer = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)
        west, south = transformer.transform(bounds.left, bounds.bottom)
        east, north = transformer.transform(bounds.right, bounds.top)
        self.latlon_bounds = ((south, west), (north, east))

        # Overlay style
        self.overlay_cmap = overlay_cmap
        self.overlay_dpi = overlay_dpi
        self.overlay_vmin = overlay_vmin
        self.overlay_vmax = overlay_vmax

        # Initial overlay PNG (cache-busted filename)
        self._overlay_png = self._new_overlay_png_path()
        self._generate_overlay_png(
            self._overlay_png,
            cmap=self.overlay_cmap,
            dpi=self.overlay_dpi,
            vmin=self.overlay_vmin,
            vmax=self.overlay_vmax,
        )

        # Initialize map
        center = ((north + south) / 2, (west + east) / 2)
        self.map = Map(center=center, zoom=zoom, layout={'height': map_size[0], 'width': map_size[1]})

        # Add image overlay
        self.overlay_layer = ImageOverlay(url=str(self._overlay_png), bounds=self.latlon_bounds)
        self.map.add_layer(self.overlay_layer)

        # Legend (safe)
        if HAS_LEGEND:
            legend_dict = {'Stable Area': 'green', 'Feature of Interest': 'red'}
            self.map.add_control(LegendControl(legend_dict, title='Legend'))

        # GeoJSON layers
        self.geojson_stable = GeoJSON(
            data={"type": "FeatureCollection", "features": []},
            style={"color": "green", "fillColor": "green", "fillOpacity": 0.3}
        )
        self.geojson_unstable = GeoJSON(
            data={"type": "FeatureCollection", "features": []},
            style={"color": "red", "fillColor": "red", "fillOpacity": 0.3}
        )
        self.map.add_layer(self.geojson_stable)
        self.map.add_layer(self.geojson_unstable)

        # Draw control
        self.draw_control = DrawControl(polygon={"shapeOptions": {"weight": 2, "fillOpacity": 0.3}})
        for attr in ('circle', 'circlemarker', 'polyline', 'rectangle'):
            setattr(self.draw_control, attr, {})
        self.draw_control.on_draw(self._handle_draw)

        # Buttons
        self.btn_stable = Button(description='Stable', layout={'width': '80px'})
        self.btn_unstable = Button(description='Unstable', layout={'width': '80px'})
        self.btn_stable.style.button_color = 'lightgreen'
        self.btn_unstable.style.button_color = 'lightcoral'
        self.btn_stable.on_click(lambda _: self._activate_category('stable'))
        self.btn_unstable.on_click(lambda _: self._activate_category('unstable'))
        btn_box = HBox([Label(' Draw mode:'), self.btn_stable, self.btn_unstable])
        self.map.add_control(WidgetControl(widget=btn_box, position='topright'))

        # Preload polygon files (if provided)
        if stable_path is not None:
            self.load_stable_polygons(stable_path, name_field=stable_name_field, assume_crs=assume_input_crs)
        if unstable_path is not None:
            self.load_unstable_polygons(unstable_path, name_field=unstable_name_field, assume_crs=assume_input_crs)
            if self.auto_stable_from_unstable:
                self.derive_stable_from_unstable(replace=True)

    # -------------------- Overlay helpers --------------------
    def _new_overlay_png_path(self) -> Path:
        stem = Path(self.topo_diff.path).stem
        return self.output_dir / f"{stem}-{uuid.uuid4().hex}.png"

    def _generate_overlay_png(
        self,
        png_path: Path | str,
        *,
        cmap: str = 'bwr_r',
        dpi: int = 300,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ):
        """Render the topographic difference raster to a PNG for map display.

        If vmin/vmax are both provided, they are used as-is.
        If only one is provided, the other is inferred symmetrically.
        If neither is provided, symmetric bounds are computed from the data.
        """
        arr = np.squeeze(self.topo_diff.data.values)

        # Resolve color limits
        if vmin is None and vmax is None:
            vmin_resolved, vmax_resolved = self._get_sym_bounds(arr)
        elif vmin is None and vmax is not None:
            vmax_resolved = float(vmax)
            vmin_resolved = -abs(vmax_resolved)
        elif vmin is not None and vmax is None:
            vmin_resolved = float(vmin)
            vmax_resolved = abs(vmin_resolved)
        else:
            vmin_resolved = float(vmin)
            vmax_resolved = float(vmax)

        # Pixel-perfect figure (no cropping, no resampling blur)
        h, w = arr.shape[-2], arr.shape[-1]
        fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1], frameon=False)  # fill full canvas
        ax.set_axis_off()

        cmap_obj = plt.get_cmap(cmap)
        cmap_obj.set_bad(color="none")  # NaNs transparent
        ax.imshow(arr, cmap=cmap_obj, vmin=vmin_resolved, vmax=vmax_resolved, interpolation="nearest")

        fig.savefig(png_path, dpi=dpi, transparent=True, pad_inches=0)
        plt.close(fig)

    @staticmethod
    def _get_sym_bounds(arr: np.ndarray) -> tuple[float, float]:
        flat = np.asarray(arr).ravel()
        valid = flat[np.isfinite(flat)]
        return (-0.0, 0.0) if valid.size == 0 else (-float(np.max(np.abs(valid))), float(np.max(np.abs(valid))))

    # -------------------- Draw controls --------------------
    def _activate_category(self, category: str):
        self.current_category = category
        if self.draw_control not in self.map.controls:
            self.map.add_control(self.draw_control)
        self.btn_stable.style.font_weight = 'bold' if category == 'stable' else 'normal'
        self.btn_unstable.style.font_weight = 'bold' if category == 'unstable' else 'normal'

    def _handle_draw(self, control, action, geo_json):
        if action != 'created' or geo_json['geometry']['type'] not in ('Polygon', 'MultiPolygon'):
            return
        geom = shape(geo_json['geometry'])
        geoms: Iterable[Polygon] = list(geom.geoms) if isinstance(geom, MultiPolygon) else [geom]  # type: ignore[attr-defined]

        if self.current_category == 'stable':
            self.stable_geoms.extend(geoms)  # type: ignore[arg-type]
            self.stable_names.extend([f"stable{i+1}" for i in range(len(self.stable_names), len(self.stable_geoms))])
            self._refresh_geojson_layer('stable')

        elif self.current_category == 'unstable':
            self.unstable_geoms.extend(geoms)  # type: ignore[arg-type]
            self.unstable_names.extend([f"unstable{i+1}" for i in range(len(self.unstable_names), len(self.unstable_geoms))])
            self._refresh_geojson_layer('unstable')
            if self.auto_stable_from_unstable:
                self.derive_stable_from_unstable(replace=True)

        if self.draw_control in self.map.controls:
            self.map.remove_control(self.draw_control)
        self.current_category = None
        self.btn_stable.style.font_weight = 'normal'
        self.btn_unstable.style.font_weight = 'normal'

    # -------------------- Load user polygon files --------------------
    def load_stable_polygons(self, path: Path | str, *, name_field: Optional[str] = None, assume_crs: Optional[str] = 'EPSG:4326'):
        gdf = self._read_polygon_file(path, assume_crs=assume_crs)
        self._append_polygons_from_gdf(gdf, category='stable', name_field=name_field)
        self._refresh_geojson_layer('stable')

    def load_unstable_polygons(self, path: Path | str, *, name_field: Optional[str] = None, assume_crs: Optional[str] = 'EPSG:4326'):
        gdf = self._read_polygon_file(path, assume_crs=assume_crs)
        self._append_polygons_from_gdf(gdf, category='unstable', name_field=name_field)
        self._refresh_geojson_layer('unstable')
        if self.auto_stable_from_unstable:
            self.derive_stable_from_unstable(replace=True)

    def _read_polygon_file(self, path: Path | str, *, assume_crs: Optional[str]) -> gpd.GeoDataFrame:
        p = Path(path)
        if p.suffix.lower() == '.zip':
            gdf = gpd.read_file(f"zip://{p}")
        else:
            gdf = gpd.read_file(p)

        gdf = gdf[~gdf.geometry.isna()]
        gdf = gdf[gdf.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])].copy()
        if gdf.empty:
            raise ValueError(f"No Polygon/MultiPolygon geometries found in {p}")

        if gdf.crs is None:
            if assume_crs is None:
                raise ValueError(f"Input file {p} has no CRS; provide assume_input_crs or fix dataset CRS.")
            gdf.set_crs(assume_crs, inplace=True)

        gdf = gdf.to_crs('EPSG:4326').explode(index_parts=False).reset_index(drop=True)
        return gdf

    def _append_polygons_from_gdf(self, gdf: gpd.GeoDataFrame, *, category: str, name_field: Optional[str]):
        base = 'stable' if category == 'stable' else 'unstable'
        start = len(self.stable_geoms) if category == 'stable' else len(self.unstable_geoms)

        # Names
        if name_field and (name_field in gdf.columns):
            raw_names = gdf[name_field].astype(str).replace({'': None}).tolist()
            names: List[str] = [(n if n not in (None, 'None') else f"{base}{start+i+1}") for i, n in enumerate(raw_names)]
        else:
            names = [f"{base}{start+i+1}" for i in range(len(gdf))]

        # Geometries
        polys: List[Polygon] = []
        for geom in gdf.geometry:
            if isinstance(geom, Polygon):
                polys.append(geom)
            elif isinstance(geom, MultiPolygon):
                polys.extend(list(geom.geoms))  # type: ignore[attr-defined]

        if category == 'stable':
            self.stable_geoms.extend(polys)
            self.stable_names.extend(names)
        else:
            self.unstable_geoms.extend(polys)
            self.unstable_names.extend(names)

    def _refresh_geojson_layer(self, category: str):
        if category == 'stable':
            feats = [
                {"type": "Feature", "properties": {"name": self.stable_names[i] if i < len(self.stable_names) else f"stable{i+1}"},
                 "geometry": mapping(g)}
                for i, g in enumerate(self.stable_geoms)
            ]
            self.geojson_stable.data = {"type": "FeatureCollection", "features": feats}
        else:
            feats = [
                {"type": "Feature", "properties": {"name": self.unstable_names[i] if i < len(self.unstable_names) else f"unstable{i+1}"},
                 "geometry": mapping(g)}
                for i, g in enumerate(self.unstable_geoms)
            ]
            self.geojson_unstable.data = {"type": "FeatureCollection", "features": feats}

    # -------------------- Derive STABLE = valid raster minus FOIs --------------------
    def derive_stable_from_unstable(
        self,
        *,
        replace: bool = True,
        exclude_nodata: bool = True,
        min_area: Optional[float] = None,
        simplify_tolerance: Optional[float] = None,
    ):
        """
        Compute Stable Areas as (valid raster area) MINUS (union of FOIs) in raster CRS,
        then reproject to EPSG:4326 for display.
        """
        if not self.unstable_geoms:
            stable_polys = self._valid_area_polygons_raster_crs(exclude_nodata=exclude_nodata)
        else:
            with rasterio.open(self.topo_diff.path) as src:
                raster_crs = src.crs
            to_raster = Transformer.from_crs('EPSG:4326', raster_crs, always_xy=True).transform
            foi_raster = [shapely_transform(to_raster, g) for g in self.unstable_geoms]
            foi_union = unary_union(foi_raster).buffer(0)

            valid_area = self._valid_area_polygons_raster_crs(exclude_nodata=exclude_nodata)
            if not valid_area:
                stable_polys = []
            else:
                base_union = unary_union(valid_area).buffer(0)
                diff = base_union.difference(foi_union)
                if diff.is_empty:
                    stable_polys = []
                else:
                    if isinstance(diff, Polygon):
                        stable_polys = [diff]
                    elif isinstance(diff, MultiPolygon):
                        stable_polys = list(diff.geoms)
                    else:
                        stable_polys = [g for g in getattr(diff, 'geoms', []) if isinstance(g, Polygon)]

        # Optional simplify & area filter in raster CRS
        tol = self.simplify_tolerance if simplify_tolerance is None else simplify_tolerance
        area_min = self.derive_min_area if min_area is None else min_area
        if tol is not None and tol > 0:
            stable_polys = [p.simplify(tol, preserve_topology=True) for p in stable_polys]
        if area_min is not None and area_min > 0:
            stable_polys = [p for p in stable_polys if p.area >= area_min]

        # Reproject to EPSG:4326 for display
        with rasterio.open(self.topo_diff.path) as src:
            raster_crs = src.crs
        to_ll = Transformer.from_crs(raster_crs, 'EPSG:4326', always_xy=True).transform
        stable_ll = [shapely_transform(to_ll, p) for p in stable_polys]

        if replace:
            self.stable_geoms = stable_ll
            self.stable_names = [f"stable{i+1}" for i in range(len(stable_ll))]
        else:
            start = len(self.stable_geoms)
            self.stable_geoms.extend(stable_ll)
            self.stable_names.extend([f"stable{start+i+1}" for i in range(len(stable_ll))])

        self._refresh_geojson_layer('stable')

    def _valid_area_polygons_raster_crs(self, *, exclude_nodata: bool) -> List[Polygon]:
        """
        Return polygons (in raster CRS) representing the valid analysis area.

        If exclude_nodata=True, polygonize cells that are finite and not equal to nodata.
        Else, return a single polygon of the raster bounds.
        """
        with rasterio.open(self.topo_diff.path) as src:
            if not exclude_nodata:
                x0, y0, x1, y1 = src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top
                return [Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])]

            arr = src.read(1, masked=True)
            data = np.array(arr.filled(np.nan), dtype=float)
            nodata = src.nodata
            valid = np.isfinite(data)
            if nodata is not None:
                valid &= (data != nodata)

            if not np.any(valid):
                return []

            # polygonize valid mask (value==1)
            shapes = rfeatures.shapes(valid.astype(np.uint8), transform=src.transform)
            polys = [shape(geom) for geom, val in shapes if val == 1]
            if not polys:
                return []
            merged = unary_union(polys).buffer(0)
            if isinstance(merged, MultiPolygon):
                return list(merged.geoms)
            return [merged]

    # -------------------- Analysis utilities --------------------
    def calculate_pixel_count(self, polygon: Polygon) -> int:
        """Count valid (non-nodata) raster pixels within a polygon."""
        with rasterio.open(self.topo_diff.path) as src:
            dst_crs = src.crs
            transformer = Transformer.from_crs('EPSG:4326', dst_crs, always_xy=True)
            poly_proj = shapely_transform(transformer.transform, polygon)
            out_img, _ = rasterio.mask.mask(src, [mapping(poly_proj)], crop=True)
            data = out_img[0]; nodata = src.nodata
            valid = np.isfinite(data) & (data != nodata)
            return int(np.sum(valid))

    def get_geodataframes(self) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """Return GeoDataFrames of the stable and unstable polygons (in raster CRS)."""
        gdf_stable = gpd.GeoDataFrame(
            {
                'Name': self.stable_names if self.stable_names else [f'stable{i+1}' for i in range(len(self.stable_geoms))],
                'geometry': self.stable_geoms,
                'pixels': [self.calculate_pixel_count(g) for g in self.stable_geoms]
            },
            crs='EPSG:4326'
        )
        gdf_unstable = gpd.GeoDataFrame(
            {
                'Name': self.unstable_names if self.unstable_names else [f'unstable{i+1}' for i in range(len(self.unstable_geoms))],
                'geometry': self.unstable_geoms,
                'pixels': [self.calculate_pixel_count(g) for g in self.unstable_geoms]
            },
            crs='EPSG:4326'
        )
        with rasterio.open(self.topo_diff.path) as src:
            raster_crs = src.crs
        return gdf_stable.to_crs(raster_crs), gdf_unstable.to_crs(raster_crs)

    # -------------------- Overlay refresh & swap --------------------
    def refresh_overlay(
        self,
        cmap: Optional[str] = None,
        dpi: Optional[int] = None,
        *,
        recalc_bounds: bool = False,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ):
        if cmap is not None:
            self.overlay_cmap = cmap
        if dpi is not None:
            self.overlay_dpi = dpi
        if vmin is not None or vmax is not None:
            if vmin is None and vmax is not None:
                self.overlay_vmax = float(vmax)
                self.overlay_vmin = -abs(self.overlay_vmax)
            elif vmin is not None and vmax is None:
                self.overlay_vmin = float(vmin)
                self.overlay_vmax = abs(self.overlay_vmin)
            else:
                self.overlay_vmin = vmin
                self.overlay_vmax = vmax

        if recalc_bounds:
            with rasterio.open(self.topo_diff.path) as ds:
                bounds = ds.bounds
                crs = ds.crs
            transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            west, south = transformer.transform(bounds.left, bounds.bottom)
            east, north = transformer.transform(bounds.right, bounds.top)
            self.latlon_bounds = ((south, west), (north, east))

        new_png = self._new_overlay_png_path()
        self._generate_overlay_png(
            new_png, cmap=self.overlay_cmap, dpi=self.overlay_dpi,
            vmin=self.overlay_vmin, vmax=self.overlay_vmax
        )

        try:
            self.map.remove_layer(self.overlay_layer)
        except Exception:
            pass
        self.overlay_layer = ImageOverlay(url=str(new_png), bounds=self.latlon_bounds)
        self.map.add_layer(self.overlay_layer)
        self._overlay_png = new_png

    def update_topo_diff(
        self,
        topo_diff_path: Path | str,
        *,
        cmap: Optional[str] = None,
        dpi: Optional[int] = None,
        recalc_bounds: bool = True,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ):
        self.topo_diff = Raster(topo_diff_path)
        self.refresh_overlay(
            cmap=cmap,
            dpi=dpi,
            recalc_bounds=recalc_bounds,
            vmin=vmin,
            vmax=vmax,
        )


def descriptive_stats(values: np.ndarray) -> pd.DataFrame:
    """
    Compute descriptive statistics for a 1D array of values.
    Returns a one-row DataFrame.
    """
    data = values[~np.isnan(values)]
    if data.size == 0:
        cols = ['mean','median','mode','std','variance','min','max',
                'skewness','kurtosis','0.5_percentile','99.5_percentile']
        return pd.DataFrame([{c: np.nan for c in cols}])
    mean = float(np.mean(data))
    median = float(np.median(data))
    try:
        mode = statistics.mode(data)
    except statistics.StatisticsError:
        mode = np.nan
    std = float(np.std(data))
    var = float(np.var(data))
    minimum = float(np.min(data))
    maximum = float(np.max(data))
    skew = float(stats.skew(data))
    kurt = float(stats.kurtosis(data))
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
        '0.5_percentile': float(p1),
        '99.5_percentile': float(p99)
    }])


class StableAreaRasterizer:
    """
    Rasterize stable-area polygons to mask a topographic-difference raster:
    - `rasterize_all`: one output where inside polygons = original values, outside = nodata.
    - `rasterize_each`: separate rasters per polygon.
    """
    def __init__(self, topo_diff_path: Path | str, stable_gdf, nodata: float = -9999):
        self.topo_path = Path(topo_diff_path)
        self.gdf = stable_gdf.copy()
        self.nodata = nodata

    def rasterize_all(self, output_path: Path | str) -> Path:
        out_path = Path(output_path)
        with rasterio.open(self.topo_path) as src:
            profile = src.profile.copy()
            profile.update(nodata=self.nodata)
            data = src.read(1)
            mask = rasterize(
                [(geom, 1) for geom in self.gdf.geometry],
                out_shape=src.shape,
                transform=src.transform,
                fill=0,
                dtype='uint8'
            )
            out = np.where(mask == 1, data, self.nodata)
            with rasterio.open(out_path, 'w', **profile) as dst:
                dst.write(out, 1)
        return out_path

    def rasterize_each(self, output_dir: Path | str) -> dict[int, Path]:
        outdir = Path(output_dir)
        outdir.mkdir(exist_ok=True, parents=True)
        paths: dict[int, Path] = {}
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
        self.rasterizer = rasterizer

    def _stats_from_raster(self, path: Path | str) -> pd.DataFrame:
        with rasterio.open(path) as src:
            arr = src.read(1)
            arr = np.where(arr == src.nodata, np.nan, arr)
            flat = arr.ravel()
        return descriptive_stats(flat)

    def stats_all(self, output_path: Path | str) -> pd.DataFrame:
        out = self.rasterizer.rasterize_all(output_path)
        df = self._stats_from_raster(out)
        df.index = ['all_areas']
        return df

    def stats_each(self, output_dir: Path | str) -> pd.DataFrame:
        paths = self.rasterizer.rasterize_each(output_dir)
        records: List[pd.DataFrame] = []
        for area_id, path in paths.items():
            df = self._stats_from_raster(path)
            df['area_id'] = area_id
            records.append(df)
        result = pd.concat(records, ignore_index=True).set_index('area_id')
        return result