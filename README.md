# Geostatistical error analysis in airborne lidar topographic differencing 

This repository contains a Jupyter notebook and supporting Python modules for a geostatistically-grounded workflow to quantify spatial uncertainty in lidar-derived topographic differencing. This toolkit provides a flexible environment for conducting customized uncertainty analysis for net change results.

## Scientific Background

The increasing availability of high-resolution, repeat lidar topographic datasets expands opportunities for detecting and quantifying landscape change. Accurately quantifying this change requires robust error modeling to account for uncertainties that are often spatially correlated and scale-dependent. This is crucial for a range of studies, from assessing landslide hazards and fluvial sediment budgets to monitoring tectonic activity and biomass changes.

Traditional uncertainty methods in topographic differencing often rely on techniques that may suppress real change or misrepresent spatially correlated errors. While geostatistical models have been adopted in fields like glaciology, their broader use has been limited by a steep learning curve and the need for specialized expertise.

This repository provides a reproducible and user-friendly approach to quantify and propagate the various sources of error in vertical topographic differencing. It dissects and quantifies common error components, including systematic vertical bias, spatially correlated random error at multiple scales, and uncorrelated noise, combining them into a total mean uncertainty. This enables more robust interpretations of topographic change and promotes best practices in lidar data analysis.

## Jupyter Notebook implementation

The Jupyter Notebook implementation guides users through a step-by-step workflow, offering a high degree of control and customization at each stage. The pipeline is structured as follows:

### Data Input and Pre-processing

The workflow begins with data input and pre-processing, managed by robust Raster and RasterPair classes that handle various data formats and sources.

- **Data Sources** : Users can upload pairs of point clouds or DEMs, or search the OpenTopography catalog for datasets. The toolkit supports local files as well as remote data via APIs.
- **Coordinate Reference System (CRS) Handling** : All inputs are reprojected into a common horizontal and vertical CRS (with common geoid model, when appropriate) to ensure alignment. The workflow supports dynamic CRS with explicit epochs for tectonically active regions.
- **Differencing** :The compare raster is subtracted from the reference raster to produce a DEM of difference (Δz).

### Delineation of Stable Areas and Features of Interest

A key step is the separation of stable ground from areas of active change.

- **Interactive Delineation** : An interactive map interface allows users to draw polygons to delineate geomorphically stable zones (e.g., bedrock, roads) and features of interest (e.g., landslides, riverbanks).
- **Error Calibration** : Stable areas serve as control zones for calibrating error and uncertainty, ensuring that the analysis is based on regions expected to have no significant change.

### Customizable Uncertainty Analysis

The core of the workflow is a comprehensive, customizable geostatistical analysis.
- **Vertical Bias Estimation** :  The median of elevation differences in stable areas is calculated to determine the systematic vertical bias.
- **Variogram Analysis** : The spatial structure of the error is characterized using variogram analysis, which includes optimized sampling, experimental variogram calculation, and nested model fitting to quantify spatial correlation at multiple scales.
- **Uncertainty Propagation** : The tool propagates correlated uncertainty using the parameters from the variogram analysis and uncorrelated uncertainty based on the dataset's RMS. These are summed in quadrature to estimate a total mean uncertainty for a specified area.
- **Advanced Options** : Users can select from multiple variogram models (e.g., Matérn, Gaussian), adjust model parameters, and iterate on the analysis for in-depth control.


## Repository Structure

- `error_analysis_notebook.ipynb` — primary analysis notebook.
- `differencing_functions.py` — helper functions for raster/array differencing.
- `stable_unstable_areas.py` — utilities to classify stable vs. unstable areas.
- `variography_uncertainty.py` — variography & uncertainty calculations.
- `requirements.txt` — Python dependencies.

## Environment & Installation

These tools rely on scientific and geospatial Python packages that include native dependencies. The recommended path is to create a Conda environment and then install the Python packages.

### Option A: Conda-first (recommended)

```bash
# Create and activate an environment
conda create -n geo-error python=3.11 -y
conda activate geo-error

# Core geospatial stack via conda-forge
conda install -c conda-forge gdal rasterio pyproj shapely geopandas rioxarray -y

# Remaining packages (also OK to conda install these)
pip install -r requirements.txt

# Enable Jupyter extensions (widgets + ipyleaflet)
jupyter nbextension enable --py widgetsnbextension
jupyter nbextension enable --py ipyleaflet
```

### Option B: Pure pip (may require system libs)

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

> If you encounter issues building GDAL/Rasterio/Shapely on your OS, prefer the Conda route above.

## Quick Start

1. **Install dependencies** (see above).
2. **Launch Jupyter**:
   ```bash
   jupyter lab  # or: jupyter notebook
   ```
3. Open **`error_analysis_notebook.ipynb`** and run cells from top to bottom, adjusting any file paths and parameters noted in the notebook.

## Data Inputs

The notebook and modules expect geospatial rasters/vectors commonly used in elevation and terrain workflows. Typical inputs include:
- Raster datasets (e.g., GeoTIFF) for **before/after** elevation models.
- Vector layers (e.g., GeoPackage/GeoJSON/Shapefile) for **stable/unstable** area delineations or evaluation polygons.
- Coordinate reference systems compatible with your area of interest (managed via `pyproj` / `rasterio` / `geopandas`).

> Review the first parameter/configuration cells in the notebook to point to your local data paths.

## Requirements

The project depends on the following external packages (see `requirements.txt`):  
`colormaps, geopandas, ipyleaflet, ipywidgets, matplotlib, numba, numpy, osgeo, pandas, pdal, pyproj, rasterio, requests, rioxarray, scipy, shapely`

If your environment manager needs package pins, test-known working versions and pin them in `requirements.txt` accordingly.

## Troubleshooting

- **GDAL/Rasterio/Shapely install errors**: Use Conda (`conda-forge` channel). Ensure system compilers and build tools are available if using pip.
- **PDAL / PDAL Python bindings**: Some systems require PDAL to be installed at the OS level. On Conda: `conda install -c conda-forge pdal`.
- **Jupyter widgets not rendering**: Ensure `ipywidgets` is installed and enabled. For Leaflet maps, install and enable `ipyleaflet`.
- **CRS or transform errors**: Confirm that source rasters and vectors share appropriate CRS or reproject using `rasterio` or `geopandas`.