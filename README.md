# Geostatistical error analysis in airborne lidar topographic differencing 

This repository contains a Jupyter notebook and supporting Python modules for geospatial error analysis, raster differencing, stability designation, and variography-based uncertainty estimation.

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


## Running the Analyses

- Use the helper modules by importing them within the notebook, for example:
  ```python
  import differencing_functions as df
  import stable_unstable_areas as sua
  import variography_uncertainty as vu
  ```
- Follow the notebook sections (headings) to execute:
  - _Run sections in order as organized in the notebook._

## Requirements

The project depends on the following external packages (see `requirements.txt`):  
`colormaps, geopandas, ipyleaflet, ipywidgets, matplotlib, numba, numpy, osgeo, pandas, pdal, pyproj, rasterio, requests, rioxarray, scipy, shapely`

If your environment manager needs package pins, test-known working versions and pin them in `requirements.txt` accordingly.

## Troubleshooting

- **GDAL/Rasterio/Shapely install errors**: Use Conda (`conda-forge` channel). Ensure system compilers and build tools are available if using pip.
- **PDAL / PDAL Python bindings**: Some systems require PDAL to be installed at the OS level. On Conda: `conda install -c conda-forge pdal`.
- **Jupyter widgets not rendering**: Ensure `ipywidgets` is installed and enabled. For Leaflet maps, install and enable `ipyleaflet`.
- **CRS or transform errors**: Confirm that source rasters and vectors share appropriate CRS or reproject using `rasterio` or `geopandas`.