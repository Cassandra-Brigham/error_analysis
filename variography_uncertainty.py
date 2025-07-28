"""Classes and functions for analyzing vertical differencing uncertainty.

This module provides a set of utilities for loading raster data,
sampling it for variogram analysis, fitting parametric semivariogram
models, and propagating uncertainty to user-defined areas.  It
implements numerically efficient algorithms via Numba for pairwise
distance calculations, supports bootstrap resampling for parameter
confidence intervals, and exposes high-level classes such as
``RasterDataHandler``, ``StatisticalAnalysis``, ``VariogramAnalysis``,
and ``RegionalUncertaintyEstimator`` to support a complete workflow
from raster preparation through uncertainty estimation.
"""

import math
from typing import Sequence, Optional, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
import rasterio
import rioxarray as rio
from numba import njit, prange
from pathlib import Path
from shapely.geometry import Polygon, MultiPolygon, Point, box, shape
import random
import geopandas as gpd
from rasterio.features import shapes
from shapely.ops import unary_union


def dropna(array):
    """
    Drops NaN values from a NumPy array.

    Parameters
    ----------
    array : np.ndarray
        The input NumPy array from which to drop NaN values.

    Returns
    -------
    np.ndarray
        A new array with NaN values removed.
    """
    return array[~np.isnan(array)]

class RasterDataHandler:
    """
    A class used for loading vertical differencing raster data, 
    subtracting a vertical systematic error from the raster,
    and randomly sampling the raster data for further analysis.

    Attributes
    ----------
    raster_path : str
        The file path to the raster data.
    unit : str
        The unit of measurement for the raster data (for plotting labels).
    resolution : float
        The resolution of the raster data.
    rioxarray_obj : rioxarray.DataArray or None
        The rioxarray object holding the raster data.
    data_array : numpy.ndarray or None
        The loaded raster data as a 1D NumPy array (NaNs removed).
    transformed_values : numpy.ndarray or None
        Placeholder for any future transformations if needed.
    samples : numpy.ndarray or None
        Sampled values from the raster data.
    coords : numpy.ndarray or None
        Coordinates of the sampled values.

    Methods
    -------
    load_raster(masked=True)
        Loads the raster data from the given path, optionally applying a mask to exclude NaN values.
    subtract_value_from_raster(output_raster_path, value_to_subtract)
        Subtracts a given value from the raster data and saves the result to a new file.
    plot_raster(plot_title)
        Plots a raster image using the loaded rioxarray object.
    sample_raster(area_side, samples_per_area, max_samples)
        Samples the raster data based on a given density and maximum number of samples.
    """
    def __init__(self, raster_path, unit, resolution):
        """
        Initialize the RasterDataHandler.

        Parameters
        ----------
        raster_path : str
            The file path to the raster data.
        unit : str
            The unit of measurement for the raster data.
        resolution : float
            The resolution of the raster data.
        """
        self.raster_path = raster_path
        self.unit = unit
        self.resolution = resolution
        self.rioxarray_obj = None
        self.data_array = None
        self.transformed_values = None
        self.samples = None
        self.coords = None
        self.shapely_geoms = None
        self.merged_geom = None
        self.detailed_area = None
        
        with rasterio.open(self.raster_path) as src:
            bounds = src.bounds
            self.bounds = (bounds.left, bounds.bottom, bounds.right, bounds.top)
            # create a Shapely geometry for the bbox
            self.bbox = box(*self.bounds)
    
    def get_detailed_area(self):  # long
        """Compute the precise area covered by valid data in the raster.

        This method reads the raster, identifies all pixels that are not
        nodata or NaN, converts the resulting mask into shapes, and
        merges them into a single geometry.  The merged geometry and
        its area are stored on the instance as ``self.merged_geom`` and
        ``self.detailed_area`` respectively.  The list of individual
        geometries is stored in ``self.shapely_geoms`` for further
        processing.
        """
        with rasterio.open(self.raster_path) as src:
            # read first band into a 2D array
            data = src.read(1)
            nodata = src.nodata
            mask = (data != nodata) if nodata is not None else ~np.isnan(data)
            geoms = shapes(data, mask=mask, transform=src.transform)
            # convert each (geom, value) pair into a Shapely object
        self.shapely_geoms = [shape(geom) for geom, value in geoms]
        self.merged_geom = unary_union(self.shapely_geoms)
        self.detailed_area = self.merged_geom.area

    def load_raster(self, masked=True):
        """
        Loads the raster data from the specified path, applying a mask to exclude NaN values if requested.

        Parameters
        ----------
        masked : bool, optional
            If True, NaN values in the raster data will be masked (default is True).
        """
        self.rioxarray_obj = rio.open_rasterio(self.raster_path, masked=masked)
        self.data_array = self.rioxarray_obj.data[~np.isnan(self.rioxarray_obj.data)].flatten()

    def subtract_value_from_raster(self, output_raster_path, value_to_subtract):
        """
        Subtracts a specified value from the raster data and saves the resulting raster to a new file.

        Parameters
        ----------
        output_raster_path : str
            The file path where the modified raster will be saved.
        value_to_subtract : float
            The value to be subtracted from each pixel in the raster data.
        """
        with rasterio.open(self.raster_path) as src:
            data = src.read()
            nodata = src.nodata

            # Create a mask of valid data
            if nodata is not None:
                mask = data != nodata
            else:
                mask = np.ones(data.shape, dtype=bool)

            data = data.astype(float)
            data[mask] -= value_to_subtract

            out_meta = src.meta.copy()
            out_meta.update({'dtype': 'float32', 'nodata': nodata})

            with rasterio.open(output_raster_path, 'w', **out_meta) as dst:
                dst.write(data)

    def plot_raster(self, plot_title):
        """
        Plots a raster image using the rioxarray object.

        Parameters
        ----------
        plot_title : str
            The title of the plot.

        Returns
        -------
        matplotlib.figure.Figure
            The figure object containing the plot.
        """
        rio_data = self.rioxarray_obj
        fig, ax = plt.subplots(figsize=(10, 6))
        rio_data.plot(cmap="bwr_r", ax=ax, robust=True)
        ax.set_title(plot_title, pad=30)
        ax.set_xlabel('Easting (m)')
        ax.set_ylabel('Northing (m)')
        ax.ticklabel_format(style="plain")
        ax.set_aspect('equal')
        return fig

    def sample_raster(self, area_side, samples_per_area, max_samples):
        """
        Samples the raster data based on a given density and a maximum number of samples,
        returning the sampled values and their coordinates.

        The parameter 'area_side' is used as a reference to convert the cell size into
        square km if 'area_side' is given in meters (e.g., area_side=1000 for 1 km side).

        Parameters
        ----------
        area_side : float
            The reference for converting pixel area into square km (if in meters).
        samples_per_area : float
            The number of samples to draw per square kilometer.
        max_samples : int
            The maximum total number of samples to draw.

        Returns
        -------
        None
            (But populates self.samples and self.coords with the drawn sample values
            and corresponding coordinates.)
        """
        with rasterio.open(self.raster_path) as src:
            data = src.read(1)
            nodata = src.nodata
            valid_data_mask = data != nodata if nodata is not None else ~np.isnan(data)

            cell_size = src.res[0]  # Pixel size in x-direction
            # Approx. area of each pixel in "km^2" if area_side = 1000.
            cell_area_sq_km = (cell_size ** 2) / (area_side ** 2)

            # Find valid data points
            valid_data_indices = np.where(valid_data_mask)
            valid_data_count = valid_data_indices[0].size

            # Estimate total samples based on area
            total_samples = min(int(cell_area_sq_km * samples_per_area * valid_data_count), max_samples)

            if total_samples > valid_data_count:
                raise ValueError("Requested samples exceed the number of valid data points.")

            # Randomly select valid data points
            chosen_indices = np.random.choice(valid_data_count, size=total_samples, replace=False)
            rows = valid_data_indices[0][chosen_indices]
            cols = valid_data_indices[1][chosen_indices]

            # Get sampled data values
            samples = data[rows, cols]

            # Compute coordinates all at once for efficiency
            x_coords, y_coords = src.xy(rows, cols)
            coords = np.vstack([x_coords, y_coords]).T

            # Remove any NaNs
            mask = ~np.isnan(samples)
            filtered_samples = samples[mask]
            filtered_coords = coords[mask]

            self.samples = filtered_samples
            self.coords = filtered_coords

class StatisticalAnalysis:
    """
    A class to perform statistical analysis on data, including plotting data statistics
    and estimating the uncertainty of the median value of the data using bootstrap 
    resampling with subsamples of the data.

    Attributes
    ----------
    raster_data_handler : RasterDataHandler
        An instance of RasterDataHandler to manage raster data operations.

    Methods
    -------
    plot_data_stats(filtered=True)
        Plots the histogram of the raster data with exploratory statistics.
    bootstrap_uncertainty_subsample(n_bootstrap=1000, subsample_proportion=0.1)
        Estimates the uncertainty of the median value of the data using bootstrap resampling.
    """
    def __init__(self, raster_data_handler):
        """
        Initialize the StatisticalAnalysis class.

        Parameters
        ----------
        raster_data_handler : RasterDataHandler
            An instance of RasterDataHandler to manage raster data operations.
        """
        self.raster_data_handler = raster_data_handler

    def plot_data_stats(self, filtered=True):
        """
        Plots the histogram of the raster data with exploratory statistics.

        Parameters
        ----------
        filtered : bool, optional
            If True, filter the data to exclude outliers (1st and 99th percentiles) 
            for better visualization. If False, use the unfiltered data. Default is True.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The matplotlib figure object containing the histogram and statistics.

        Notes
        -----
        - The function calculates and displays the mean, median, mode(s), minimum, maximum, 
          1st quartile, and 3rd quartile of the data.
        - The histogram is plotted with 60 bins (by default).
        - The mode(s) are displayed as vertical dashed lines on the histogram.
        - A text box with the calculated statistics is added to the plot.
        """
        data = self.raster_data_handler.data_array
        if data is None or len(data) == 0:
            raise ValueError("No data available to plot. Please load the raster first.")

        mean = np.mean(data)
        median = np.median(data)
        mode_result = stats.mode(data, nan_policy='omit')
        mode_val = mode_result.mode
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        p1 = np.percentile(data, 1)
        p99 = np.percentile(data, 99)
        minimum = np.min(data)
        maximum = np.max(data)

        # Optionally filter outliers for visualization
        if filtered:
            data = data[(data >= p1) & (data <= p99)]

        # Ensure mode_val is iterable
        if not isinstance(mode_val, np.ndarray):
            mode_val = np.array([mode_val])

        fig, ax = plt.subplots()
        # Plot histogram
        ax.hist(data, bins=60, density=False, alpha=0.6, color='g')
        ax.axvline(mean, color='r', linestyle='dashed', linewidth=1, label='Mean')
        ax.axvline(median, color='b', linestyle='dashed', linewidth=1, label='Median')
        for i, m in enumerate(mode_val):
            label = 'Mode' if i == 0 else "_nolegend_"
            ax.axvline(m, color='purple', linestyle='dashed', linewidth=1, label=label)

        # Prepare statistics text
        mode_str = ", ".join([f'{m:.3f}' for m in mode_val])
        textstr = '\n'.join((
            f'Mean: {mean:.3f}',
            f'Median: {median:.3f}',
            f'Mode(s): {mode_str}',
            f'Minimum: {minimum:.3f}',
            f'Maximum: {maximum:.3f}',
            f'1st Quartile: {q1:.3f}',
            f'3rd Quartile: {q3:.3f}'
        ))

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)

        ax.set_xlabel(f'Vertical Difference ({self.raster_data_handler.unit})')
        ax.set_ylabel('Count')
        ax.set_title('Histogram of differencing results with exploratory statistics')
        ax.legend()
        plt.tight_layout()
        return fig

    def bootstrap_uncertainty_subsample(self, n_bootstrap=1000, subsample_proportion=0.1):
        """
        Estimates the uncertainty of the median value of the data using bootstrap resampling.
        This method randomly samples subsets of the data, calculates their medians, and then 
        computes the standard deviation of these medians as a measure of uncertainty.

        Parameters
        ----------
        n_bootstrap : int, optional
            The number of bootstrap samples to generate (default is 1000).
        subsample_proportion : float, optional
            The proportion of the data to include in each subsample (default is 0.1).

        Returns
        -------
        uncertainty : float
            The standard deviation of the bootstrap medians, representing 
            the uncertainty of the median value.
        """
        if (self.raster_data_handler.data_array is None or 
            len(self.raster_data_handler.data_array) == 0):
            raise ValueError("No data available for bootstrap. Please load the raster first.")

        subsample_size = int(subsample_proportion * len(self.raster_data_handler.data_array))
        bootstrap_medians = np.zeros(n_bootstrap)
        for i in range(n_bootstrap):
            sample = np.random.choice(self.raster_data_handler.data_array,
                                      size=subsample_size,
                                      replace=True)
            bootstrap_medians[i] = np.median(sample)

        return np.std(bootstrap_medians)

class VariogramAnalysis:
    """
    A class to perform variogram analysis on raster data. It calculates mean variograms,
    fits spherical models (possibly with a nugget term) to the variogram data, and plots
    the results. The code supports multiple runs to produce a mean variogram and uses
    bootstrapping to estimate parameter uncertainties.
    """
    MIN_PAIRS = 10
    def __init__(self, raster_data_handler):
        """Initialize a ``VariogramAnalysis`` instance for a given raster.

        Parameters
        ----------
        raster_data_handler : RasterDataHandler
            An instance of :class:`RasterDataHandler` containing the raster
            data and sampling utilities required for variogram estimation.

        Notes
        -----
        This constructor merely stores the provided data handler and
        initializes various attributes used during variogram calculation
        and model fitting.  To compute an empirical variogram, call
        :meth:`calculate_mean_variogram_numba`.  To fit a parametric
        model, use :meth:`fit_best_spherical_model`.
        """
        self.raster_data_handler = raster_data_handler
        self.mean_variogram = None
        self.lags = None
        self.mean_count = None
        self.err_variogram = None
        self.best_model_config = None
        self.fitted_variogram = None
        self.rmse = None
        self.sills = None
        self.ranges = None
        self.ranges_min = None
        self.ranges_max = None
        self.ranges_median = None
        self.err_sills = None
        self.err_ranges = None
        self.sills_min = None
        self.sills_max = None
        self.sills_median = None
        self.best_nugget = None
        self.min_nugget = None
        self.max_nugget = None
        self.median_nugget = None
        self.best_aic = None
        self.best_params = None
        self.best_model_config = None
        self.cv_mean_error_best_aic = None
        self.param_samples = None
        self.n_bins = None
        self.sigma_variogram = None
        self.best_model_func = None
        self.best_guess = None
        self.best_bounds = None
        self.all_variograms = None
        self.all_counts = None
        self.param_samples = None
    
    
            
    @staticmethod
    @njit(parallel=True)
    def bin_distances_and_squared_differences(coords, values, bin_width, max_lag_multiplier, x_extent, y_extent):
        """
        Compute and bin pairwise distances and squared differences for Matheron estimation.

        Parameters:
        -----------
        coords : np.ndarray
            Array of coordinates of shape (M, 2).
        values : np.ndarray
            Array of values of shape (M,).
        bin_edges : np.ndarray
            Array of bin edges for distance binning.

        Returns:
        --------
        bin_counts : np.ndarray
            Counts of pairs in each bin.
        binned_sum_squared_diff : np.ndarray
            Sum of squared differences for each bin.
        """
        approx_max_distance = np.sqrt(x_extent**2 + y_extent**2)
        
        if max_lag_multiplier == "max":
            max_lag = approx_max_distance
        else:
            max_lag = int(approx_max_distance*max_lag_multiplier)
        
        #Determine bin edges using diagonal distance as maximum possible lag distance
        n_bins = int(np.ceil(max_lag / bin_width)) + 1
        bin_edges = np.arange(0, n_bins * bin_width, bin_width)
        
        M = coords.shape[0]
        max_distance = 0.0
        bin_counts = np.zeros(n_bins, dtype=np.int64)
        binned_sum_squared_diff = np.zeros(n_bins, dtype=np.float64)

        for i in prange(M):
            for j in range(i + 1, M):
                # Compute the pairwise distance
                d = 0.0
                for k in range(coords.shape[1]):
                    tmp = coords[i, k] - coords[j, k]
                    d += tmp * tmp
                dist = np.sqrt(d)
                max_distance = max(max_distance, dist)
                
                #Compute the difference
                diff = values[i] - values[j]
                
                # Compute the squared difference
                diff_squared = (diff) ** 2

                # Find the bin for this distance
                bin_idx = int(dist / bin_width)
                if 0 <= bin_idx < n_bins:
                    bin_counts[bin_idx] += 1
                    binned_sum_squared_diff[bin_idx] += diff_squared
        
        
        bin_edges = bin_edges[:n_bins]
        bin_counts = bin_counts[:n_bins]
        binned_sum_squared_diff = binned_sum_squared_diff[:n_bins]

        return n_bins, bin_counts, binned_sum_squared_diff, max_distance, max_lag
    
    @staticmethod
    def compute_matheron(bin_counts, ssd, min_pairs=8):
        """
        Compute Matheron’s semivariance estimator for binned pairwise differences.
        This function calculates the empirical semivariogram γ(h) for each lag bin
        using Matheron’s estimator:
            γ(h) = SSD(h) / (2 · N(h))
        where SSD(h) is the sum of squared differences of all point pairs within
        lag bin h, and N(h) is the number of pairs in that bin. Bins with fewer
        than `min_pairs` are assigned NaN.
        Args:
            bin_counts (array-like of int):
                Number of point‐pair observations in each lag bin.
            ssd (array-like of float):
                Sum of squared differences of measurements for each lag bin.
            min_pairs (int, optional):
                Minimum number of pairs required to compute a reliable estimate.
                Bins with counts below this threshold will be set to NaN.
                Defaults to 8.
        Returns:
            numpy.ndarray of float:
                Semivariance estimates for each lag bin. Bins with
                `bin_counts[i] < min_pairs` are returned as NaN.
        """
    
        γ = np.full_like(bin_counts, np.nan, dtype=float)
        for i,(cnt,sum_sq) in enumerate(zip(bin_counts,ssd)):
            if cnt >= min_pairs:
                γ[i] = sum_sq/(2*cnt)
        return γ
    
    def numba_variogram(self, area_side, samples_per_area, max_samples, bin_width, max_lag_multiplier):
        """
        Calculate the variogram using Numba for performance optimization.
        Parameters:
        -----------
        area_side : float
            The side length of the area to sample.
        samples_per_area : int
            The number of samples to take per area.
        max_samples : int
            The maximum number of samples to take.
        bin_width : float
            The width of the bins for distance binning.
        cell_size : float
            The size of the cell for declustering.
        n_offsets : int
            The number of offsets for declustering.
        max_lag_multiplier : str or float
            The multiplier for the maximum lag distance. Can be "median", "max", or a float value.
        normal_transform : bool
            Whether to apply a normal transformation to the data.
        weights : bool
            Whether to apply declustering weights.
        Returns:
        --------
        bin_counts : numpy.ndarray
            The counts of pairs in each bin.
        variogram_matheron : numpy.ndarray
            The calculated variogram values for each bin.
        n_bins : int
            The number of bins used.
        min_distance : float
            The minimum distance considered.
        max_distance : float
            The maximum distance considered.
        """
        
        self.raster_data_handler.sample_raster(area_side, samples_per_area, max_samples)
        
        min_distance = 0.0
        
        x_extent = self.raster_data_handler.rioxarray_obj.rio.width
        y_extent = self.raster_data_handler.rioxarray_obj.rio.height

        n_bins, bin_counts, binned_sum_squared_diff, max_distance, max_lag = self.bin_distances_and_squared_differences(self.raster_data_handler.coords, self.raster_data_handler.samples, bin_width, max_lag_multiplier, x_extent, y_extent)
        matheron_estimates = self.compute_matheron(bin_counts, binned_sum_squared_diff, min_pairs=10)
        
        return bin_counts, matheron_estimates, n_bins, min_distance, max_distance, max_lag
    
    def calculate_mean_variogram_numba(self,
                                   area_side,
                                   samples_per_area,
                                   max_samples,
                                   bin_width,
                                   max_n_bins,
                                   n_runs,
                                   max_lag_multiplier=1/3):
        """
        Calculate the mean variogram using numba over multiple runs.
        """

        # Prepare DataFrames to collect per‐run variograms and counts
        all_variograms = pd.DataFrame(np.nan, index=range(n_runs), columns=range(max_n_bins))
        counts = pd.DataFrame   (np.nan, index=range(n_runs), columns=range(max_n_bins))

        # Track how many bins each run actually produced
        all_n_bins = np.zeros(n_runs, dtype=int)

        for run in range(n_runs):
            count, variogram, n_bins, _, _, _ = self.numba_variogram(
                area_side, samples_per_area, max_samples, bin_width, max_lag_multiplier
            )
            # store semivariances and counts into our tables
            all_variograms.loc[run, :variogram.size-1] = variogram
            counts       .loc[run, :count.size-1]     = count
            all_n_bins[run] = n_bins

        # convert to numpy arrays for ease
        vario_arr  = all_variograms.values
        count_arr  = counts.values

        # compute mean & std, guarding empty‐slice warnings
        with np.errstate(all='ignore'):
            mean_variogram = np.nanmean(vario_arr, axis=0)
            err_variogram  = (np.nanpercentile(vario_arr, 97.5, axis=0)
                            -np.nanpercentile(vario_arr,  2.5,axis=0)) / 2.0
            mean_count     = np.nanmean(count_arr, axis=0)
            sigma_variogram = np.nanstd(vario_arr, axis=0)

        # replace any zero‐std with a tiny epsilon to avoid division‐by‐zero later
        sigma_filtered = sigma_variogram.copy()
        sigma_filtered[sigma_filtered == 0] = np.finfo(float).eps

        # now drop trailing NaNs so our lags array lines up
        valid = ~np.isnan(mean_variogram)
        self.mean_variogram = mean_variogram[valid]
        self.err_variogram  = err_variogram[valid]
        self.mean_count     = mean_count[valid]
        self.sigma_variogram= sigma_filtered[valid]

        # build lags at bin‐centers
        n_kept = self.mean_variogram.size
        self.lags = np.linspace(bin_width/2,
                                bin_width*n_kept - bin_width/2,
                                n_kept)

        # store raw run matrices if you still need them
        self.all_variograms = vario_arr
        self.all_counts     = count_arr
        self.n_bins         = int(np.nanmean(all_n_bins))
    
    @staticmethod
    def get_base_initial_guess(n, mean_variogram, lags, nugget=False):
        """Construct a naive initial guess for variogram model parameters.

        For a model with ``n`` spherical components, this helper builds a
        starting parameter vector by spreading the sills evenly and
        assigning ranges that increase linearly with the maximum lag.  If
        a nugget term is requested, an additional parameter is appended.

        Parameters
        ----------
        n : int
            Number of spherical components in the model.
        mean_variogram : array-like
            Empirical semivariogram values used to scale the sills.
        lags : array-like
            Lag distances corresponding to the empirical semivariogram.
        nugget : bool, optional
            Whether to include a nugget parameter at the end of the
            initial guess vector.

        Returns
        -------
        numpy.ndarray
            A 1-D array containing the initial guesses for the sills,
            ranges, and optional nugget.
        """
        max_semivariance = np.max(mean_variogram)*1.5
        half_max_lag = np.max(lags) / 2
        C = [max_semivariance / n]*n      # sills
        a = [((half_max_lag)/3)*(i+1) for i in range(n)]  # ranges
        p0 = C + a + ([max_semivariance / 4] if nugget else [])
        return np.array(p0)
    
    @staticmethod
    def get_randomized_guesses(base_guess, n_starts=5, perturb_factor=0.3):
        """
        Generate multiple initial guesses by perturbing the base guess.
        'perturb_factor' is a fraction of the base guess values.
        """
        p0s = []
        for _ in range(n_starts):
            rand_perturbation = (np.random.rand(len(base_guess)) - 0.5) * 2.0
            # Scale by base_guess * perturb_factor
            new_guess = base_guess * (1 + rand_perturbation * perturb_factor)
            # Ensure no negative sills or other invalid guesses
            new_guess = np.clip(new_guess, 1e-3, None)  
            p0s.append(new_guess)
        return p0s
    
    @staticmethod
    def pure_nugget_model(h, nugget):
        """Semivariogram model representing pure nugget variance.

        This model describes a process where there is no spatial
        correlation and the semivariance is constant for all lag
        distances ``h``.

        Parameters
        ----------
        h : array-like
            Distances at which to evaluate the model.
        nugget : float
            The variance (sill) of the process; this value is
            returned for all entries of ``h``.

        Returns
        -------
        numpy.ndarray
            An array of the same shape as ``h`` filled with ``nugget``.
        """
        return np.full_like(h, nugget)
    
    @staticmethod
    def spherical_model(h, *params):
        """
        Computes the spherical model for given distances and parameters.

        The spherical model is commonly used in geostatistics to describe spatial 
        correlation. It is defined piecewise, with different formulas for distances 
        less than or equal to the range parameter and for distances greater than the 
        range parameter.

        Parameters:
        h (array-like): Array of distances at which to evaluate the model.
        *params: Variable length argument list containing the sill and range parameters.
                    The first half of the parameters are the sills (C), and the second half 
                    are the ranges (a). The number of sills and ranges should be equal.

        Returns:
        numpy.ndarray: Array of model values corresponding to the input distances.
        """
        n = len(params) // 2
        C = params[:n]
        a = params[n:]
        model = np.zeros_like(h)
        for i in range(n):
            mask = h <= a[i]
            model[mask] += C[i] * (1.5 * (h[mask] / a[i]) - 0.5 * (h[mask] / a[i]) ** 3)
            #model[mask] += C[i] * (3 * h[mask] / (2 * a[i]) - (h[mask] ** 3) / (2 * a[i] ** 3))
            model[~mask] += C[i]
        return model
        
    def spherical_model_with_nugget(self,h, nugget, *params):
        """
        Computes the spherical model with a nugget effect.

        The spherical model is a type of variogram model used in geostatistics.
        This function adds a nugget effect to the spherical model.

        Parameters:
        h (array-like): Array of distances at which to evaluate the model.
        nugget (float): The nugget effect, representing the discontinuity at the origin.
        *params: Variable length argument list containing the sill and range parameters.
                    The first half of the parameters are the sills (C), and the second half 
                    are the ranges (a). The number of sills and ranges should be equal.

        Returns:
        numpy.ndarray: Array of model values corresponding to the input distances.
        """
        nugget = params[-1]
        # The sills and ranges are all parameters before the last one
        structural_params = params[:-1]
        
        return nugget + self.spherical_model(h, *structural_params)
        
    
    def cross_validate_variogram(self, model_func, p0, bounds, k=5):
        """
        Perform k-fold cross-validation on the binned variogram data.
        
        Returns a dictionary of average fold metrics:
            - 'rmse'
            - 'mae'
            - 'me'  (mean error)
            - 'mse'
        
        If no fold converges, returns None.
        """
        lags = self.lags
        mean_variogram = self.mean_variogram
        sigma_filtered = self.sigma_variogram
        
        n_bins = len(lags)
        indices = np.arange(n_bins)
        np.random.shuffle(indices)

        fold_size = max(1, n_bins // k)  # in case n_bins < k
        rmses = []
        maes = []
        mes  = []
        mses = []

        for i in range(k):
            valid_idx = indices[i*fold_size : (i+1)*fold_size]
            train_idx = np.setdiff1d(indices, valid_idx)

            # Subset train
            lags_train = lags[train_idx]
            vario_train = mean_variogram[train_idx]
            sigma_train = sigma_filtered[train_idx]

            # Fit on training fold
            try:
                popt, _ = curve_fit(model_func,
                                    lags_train, vario_train,
                                    p0=p0,
                                    bounds=bounds,
                                    sigma=sigma_train)
            except RuntimeError:
                continue

            # Predict on validation fold
            lags_valid = lags[valid_idx]
            vario_valid = mean_variogram[valid_idx]
            predictions = model_func(lags_valid, *popt)

            # Compute metrics
            errors = vario_valid - predictions
            rmse  = np.sqrt(np.mean(errors**2))
            mae   = np.mean(np.abs(errors))
            me    = np.mean(errors)
            mse   = np.mean(errors**2)

            rmses.append(rmse)
            maes.append(mae)
            mes.append(me)
            mses.append(mse)

        if len(rmses) == 0:
            return None  # indicates all folds failed to converge

        # Average across folds
        return {
            'rmse': np.mean(rmses),
            'mae':  np.mean(maes),
            'me':   np.mean(mes),
            'mse':  np.mean(mses)
        }
    
    @staticmethod
    def bootstrap_fit_variogram(lags, mean_vario,sigma_vario, model_func, p0,
                            bounds=None, n_boot=100, maxfev=10000):
        """
        Perform a parametric bootstrap to estimate parameter uncertainties for
        a variogram model, assuming `err_vario` represents the half-width of 
        a 95% confidence interval at each lag.

        Parameters
        ----------
        lags : np.ndarray
            Array of lag distances (length m).
        mean_vario : np.ndarray
            The mean variogram values across bins (length m).
        err_vario : np.ndarray 
            95% confidence interval for each bin (length m).
        model_func : callable
            The variogram model (e.g., spherical, exponential, etc.).
        p0 : np.ndarray
            An initial guess for the parameters [C1, C2, ..., a1, a2, ..., (nugget?)].
        bounds : tuple or None
            (lower_bounds, upper_bounds) for each parameter, if needed by curve_fit.
        n_boot : int
            Number of bootstrap replicates.
        maxfev : int
            Maximum function evaluations for curve_fit.

        Returns
        -------
        param_samples : np.ndarray
            Array of shape (n_successful, n_params) with fitted parameters from
            each bootstrap iteration. Some may fail to converge.
        """
        
        noise_array = np.zeros((n_boot, len(mean_vario)))
        rng = np.random.default_rng()
        for i,(v,s) in enumerate(zip(mean_vario,sigma_vario)):

            noise_temp = rng.normal(loc=v, scale=s, size=n_boot)
            noise_array[:,i] = noise_temp

        param_samples = []
        n_params = len(p0)

        for n in range(n_boot):

            # Create synthetic data
            synthetic_data = noise_array[n,:]

            # Fit the model
            try:
                popt_synth, pcov_synth = curve_fit(
                    model_func,
                    lags,
                    synthetic_data,
                    p0=p0,
                    sigma=None,  # or pass std_est if you want weighting
                    bounds=bounds if bounds is not None else (-np.inf, np.inf),
                    maxfev=maxfev
                )
                param_samples.append(popt_synth)
            except RuntimeError:
                # If the fit fails, store NaNs for the parameters
                param_samples.append([np.nan]*n_params)

        param_samples = np.array(param_samples)
        # Remove any failed fits (NaNs)
        valid = ~np.isnan(param_samples).any(axis=1)
        param_samples = param_samples[valid]

        return param_samples
    
    def fit_best_spherical_model(self,
                                 sigma_type: str = 'std',
                                 bounds: tuple = None,
                                 method: str = 'trf'):
        """Fit spherical variogram models and select the best configuration.

        This routine iterates over a set of candidate spherical models
        (with one to three components and optional nugget terms), fits
        each to the mean empirical semivariogram, and evaluates their
        Akaike Information Criterion (AIC).  The model with the lowest
        AIC is retained and stored along with its fitted parameters.

        Parameters
        ----------
        sigma_type : str, optional
            Determines the weighting applied during curve fitting.
            ``'std'`` uses the standard deviation of the semivariogram,
            while other options (``'linear'``, ``'exp'``, ``'sqrt'``,
            ``'sq'``) apply alternative weighting schemes.
        bounds : tuple, optional
            A tuple of lower and upper bounds for the parameters.  If
            ``None``, bounds are determined internally for each model.
        method : str, optional
            Optimization algorithm passed to ``scipy.optimize.curve_fit``.
            The default ``'trf'`` uses a trust-region method suitable for
            bounded problems.

        Raises
        ------
        RuntimeError
            If :meth:`calculate_mean_variogram_numba` has not been
            called prior to model fitting or if no valid model fit is
            found.
        """
        if self.all_variograms is None:
            raise RuntimeError("You must call calculate_mean_variogram_numba() before fitting.")

        # choose weights
        if sigma_type == 'std':
            sigma = self.sigma_variogram
        elif sigma_type == 'linear':
            sigma = 1.0 / (1.0 + self.lags)
        elif sigma_type == 'exp':
            sigma = np.exp(-self.lags)
        elif sigma_type == 'sqrt':
            sigma = 1.0 / np.sqrt(1.0 + self.lags)
        elif sigma_type == 'sq':
            sigma = 1.0 / (1.0 + self.lags**2)
        else:
            raise ValueError(f"Unknown sigma_type '{sigma_type}'")

        best_aic = np.inf
        best = None
        for config in (
            {'components': 1, 'nugget': False},
            {'components': 1, 'nugget': True},
            {'components': 2, 'nugget': False},
            {'components': 2, 'nugget': True},
            {'components': 3, 'nugget': False},
            {'components': 3, 'nugget': True},
        ):
            n = config['components']
            nugget = config['nugget']
            if n == 0:
                model = self.pure_nugget_model
                lower_bounds, upper_bounds = [0], [np.inf]
                p0s = [np.array([np.max(self.mean_variogram)])]
            else:
                model = self.spherical_model_with_nugget if nugget else self.spherical_model
                lower_bounds = [0]*n + [1]*n + ([0] if nugget else [])
                upper_bounds = [np.inf]*n + [np.max(self.lags)]*n + ([np.inf] if nugget else [])
                p0s = self.get_randomized_guesses(
                    self.get_base_initial_guess(n, self.mean_variogram, self.lags, nugget),
                    n_starts=5,
                    perturb_factor=0.5
                )
            bounds_tuple = (lower_bounds, upper_bounds)
            
            for p0 in p0s:
                try:
                    popt, _ = curve_fit(
                        model, self.lags, self.mean_variogram,
                        p0=p0, sigma=sigma, bounds=bounds_tuple,
                        method=method, maxfev=10000
                    )
                except RuntimeError:
                    continue

                # AIC
                resid = self.mean_variogram - model(self.lags, *popt)
                s2 = np.var(resid)
                ll = -0.5*len(resid)*(np.log(2*np.pi*s2)+1)
                aic = 2*len(popt) - 2*ll

                if aic < best_aic:
                    best_aic = aic
                    best_params = popt
                    best_model = config
                    best_func = model

        if best_params is None:
            raise RuntimeError("No valid model fit found.")

        self.best_params = best_params
        self.best_model_config = best_model
        self.best_model_func = best_func
        self.best_aic = best_aic
        self.fitted_variogram = (
            self.spherical_model_with_nugget if self.best_model_config['nugget']
            else self.spherical_model
        )(self.lags, *self.best_params)
        

    # extract sill & range point estimates
        n = self.best_model_config['components']
        nug = self.best_model_config['nugget']
        if n == 0:
            bounds_boot = ([0], [np.inf])
        else:
            # sill bounds = [0]*n, range bounds = [1]*n, optional nugget bound = [0]
            lb = [0]*n + [1]*n + ([0] if nug else [])
            ub = [np.inf]*n + [np.max(self.lags)]*n + ([np.inf] if nug else [])
            bounds_boot = (lb, ub)
        
        if self.best_model_config['nugget']:
            # [nug,C1..Cn,a1..an]
            self.sills  = self.best_params[1:1+n]
            self.ranges = self.best_params[1+n:1+2*n]
        else:
            # [C1..Cn,a1..an]
            self.sills  = self.best_params[:n]
            self.ranges = self.best_params[n:2*n]

        # parametric bootstrap
        samples = self.bootstrap_fit_variogram(
            self.lags,
            self.mean_variogram,
            self.err_variogram,
            self.best_model_func,
            self.best_params,
            bounds=bounds_boot,
            n_boot=500,
            maxfev=10000
        )

        # if any succeeded, compute percentiles
        if samples.size:
            # drop nugget column if present
            if self.best_model_config['nugget']:
                nug_samps = samples[:,0]
                samp = samples[:,1:]
            else:
                nug_samps = None
                samp = samples

            sill_samps  = samp[:,:n]
            range_samps = samp[:,n:2*n]

            self.sills_min    = np.percentile(sill_samps,  16, axis=0)
            self.sills_max    = np.percentile(sill_samps, 84, axis=0)
            self.sills_median = np.percentile(sill_samps, 50,   axis=0)

            self.ranges_min    = np.percentile(range_samps,  16, axis=0)
            self.ranges_max    = np.percentile(range_samps, 84, axis=0)
            self.ranges_median = np.percentile(range_samps, 50,   axis=0)

            if nug_samps is not None:
                self.min_nugget    = np.percentile(nug_samps,  16)
                self.max_nugget    = np.percentile(nug_samps, 84)
                self.median_nugget = np.percentile(nug_samps, 50)
            else:
                self.min_nugget = self.max_nugget = self.median_nugget = None
        else:
            # fallback to point estimates
            self.sills_min = self.sills_max = self.sills_median = np.array(self.sills)
            self.ranges_min = self.ranges_max = self.ranges_median = np.array(self.ranges)
            if self.best_model_config['nugget']:
                self.min_nugget = self.max_nugget = self.median_nugget = self.best_nugget
            else:
                self.min_nugget = self.max_nugget = self.median_nugget = None
    
    def plot_best_spherical_model(self):
        """
        Plots the best spherical model for the variogram analysis.
        This function generates a two-panel plot:
        - The top panel displays a histogram of semivariance counts.
        - The bottom panel shows the mean variogram with error bars, the fitted model,
          vertical lines at each range ± its error, and the nugget effect if applicable.
        The bottom title shows the RMSE from cross-validation.
        Returns:
            fig (matplotlib.figure.Figure): The figure object containing the plot.
        """
        # ensure everything is available
        if any(attr is None for attr in (self.mean_variogram,
                                         self.err_variogram,
                                         self.mean_count,
                                         self.lags,
                                         self.fitted_variogram)):
            raise RuntimeError("Must run calculate_mean_variogram_numba() and fit_best_spherical_model() first.")

        # truncate to common valid length
        n = min(len(self.lags),
                len(self.mean_variogram),
                len(self.err_variogram),
                len(self.fitted_variogram))
        lags   = self.lags[:n]
        gamma  = self.mean_variogram[:n]
        errs   = self.err_variogram[:n]
        model  = self.fitted_variogram[:n]
        counts = self.mean_count[:n]

        # drop any bins with zero or NaN counts
        valid_counts = (~np.isnan(counts)) & (counts > 0)
        count_lags   = lags[valid_counts]
        count_vals   = counts[valid_counts]

        # build figure
        fig, axs = plt.subplots(2, 1, 
                                gridspec_kw={'height_ratios': [1, 3]},
                                figsize=(10, 8),
                                sharex=True)

        # ─── top: histogram of mean pair-counts ────────────────────────────────
        bar_width = (lags[1] - lags[0]) * 0.9
        axs[0].bar(count_lags, count_vals, width=bar_width,
                   color='orange', alpha=0.5)
        axs[0].set_ylabel('Mean Count')
        axs[0].tick_params(labelbottom=False)

        # ─── bottom: variogram & model ────────────────────────────────────────
        axs[1].errorbar(lags, gamma, yerr=errs, fmt='o-', 
                        color='blue', label='Mean Variogram ± spread')
        axs[1].plot(lags, model, 'r-', label='Fitted Model')

        # draw range lines ± error
        colors = ['red','green','blue']
        if self.ranges is not None and \
           self.ranges_min is not None and \
           self.ranges_max is not None:
            ylim = axs[1].get_ylim()
            for i,(r, rmin, rmax) in enumerate(zip(self.ranges,
                                                   self.ranges_min,
                                                   self.ranges_max)):
                c = colors[i % len(colors)]
                axs[1].axvline(r, color=c, linestyle='--', linewidth=1,
                               label=f'Range {i+1}')
                axs[1].fill_betweenx(ylim, rmin, rmax,
                                     color=c, alpha=0.2)

        # draw nugget ± error
        if self.best_nugget is not None and \
           self.min_nugget is not None and \
           self.max_nugget is not None:
            axs[1].axhline(self.best_nugget, color='black',
                           linestyle='--', linewidth=1,
                           label='Nugget')
            axs[1].fill_between(lags, 
                                [self.min_nugget]*n, 
                                [self.max_nugget]*n,
                                color='gray', alpha=0.2)

        axs[1].set_xlabel('Lag Distance')
        axs[1].set_ylabel('Semivariance')
        axs[1].legend(loc='upper right')

        # show RMSE in title if available
        rmse_str = ""
        if isinstance(self.cv_mean_error_best_aic, dict):
            rmse = self.cv_mean_error_best_aic.get('rmse', None)
            if rmse is not None:
                rmse_str = f'RMSE: {rmse:.4f}'
        axs[1].set_title(rmse_str)

        plt.setp(axs[0].get_xticklabels(), visible=False)
        plt.tight_layout()
        return fig

class RegionalUncertaintyEstimator:
    # (unchanged __init__ and helper methods before estimate_monte_carlo_raster)
    """
    Estimate the regional uncertainty (standard deviation σ_A) for a given area (polygon)
    using multiple methods. The regional uncertainty is defined by the formula:
    
        σ_A^2 = (1 / A^2) ∬_A [σ_Δz^2 – γ(h)] dx dy,
    
    where A is the area of the polygon, σ_Δz^2 is the variance (sill) of elevation differences (a constant),
    and γ(h) is the semivariogram function giving the variance of differences at separation distance h.
    
    This class implements four methods to compute σ_A:
    
    1. **Analytical (Approximate):** Uses a theoretical approximation (e.g., treating the area as an 
       equivalent circle of the same area) to integrate the variogram analytically or semi-analytically.
    2. **Brute-force Numerical:** Performs a numerical double integration by discretizing the area 
       into a fine grid of points and summing contributions of all pairs of points.
    3. **Monte Carlo:** Estimates the double integral via random sampling of point pairs in the polygon.
    4. **FFT Convolution:** Uses Fast Fourier Transform convolution/correlation to efficiently compute 
       the double integral over the area by leveraging the convolution theorem.
    
    Each method returns the regional uncertainty σ_A (the standard deviation) for the area.
    
    **Parameters:**
    - `polygon`: The polygonal area of interest, defined as a GeoJSON dictionary/string or a Shapely Polygon/MultiPolygon. 
       Arbitrary polygonal shapes (including those with holes or multiple parts) are supported. 
       If a GeoJSON is provided, it will be converted to a Shapely geometry.
    - `gamma_func`: A callable function `gamma_func(h)` that returns the semivariogram γ(h) for a given 
       separation distance h. This models spatial correlation (γ(0) should be 0, and γ(h) approaches σ_Δz^2 as h increases to the range).
    - `sigma2_delta_z`: The fixed variance σ_Δz^2 (sill) corresponding to the semivariogram. This is the variance of differences 
       at zero separation (the total variance of the process). For a covariance model C(h), C(h) = σ_Δz^2 – γ(h).
    
    **Coordinate Units:** The polygon coordinates and the variogram function’s distance units must be consistent (e.g., both in meters). 
    If the polygon is provided in geographic coordinates (lat/long), it should be projected to a planar coordinate system 
    for meaningful distance calculations.
    """
    # ─── static helpers ─────────────────────────────────────────────────
    @staticmethod
    def arrange_params(*, sills, ranges, nugget=None):
        """
        Build a flat parameter list to feed the fitted variogram model.
        Returns (params1, params2, params3, all_params) where each paramsX
        corresponds to one of the spherical components (None if not needed).
        """
        if len(sills) != len(ranges):
            raise ValueError("sills and ranges must have the same length")

        all_params = []
        param_sets = []
        all_sills = []
        all_ranges = []

        for C, a in zip(sills, ranges):
            p = [C, a]
            param_sets.append(p)
            all_sills.append(C)
            all_ranges.append(a)
        all_params = all_sills + all_ranges

        if nugget is not None:
            all_params  = [nugget] + all_params
            param_sets  = [[nugget] + p for p in param_sets]

        # pad with None so we always return 3 items
        while len(param_sets) < 3:
            param_sets.append(None)

        return (*param_sets, all_params)

    @staticmethod
    def bind_gamma(model_func, params):
        """Return γ(h) with parameters ‘params’ baked in (or None)."""
        if params is None:
            return None
        else:
            
            return lambda h: model_func(np.asarray(h, dtype=float), *params)
    
    
    def __init__(self, raster_data_handler, variogram_analysis, area_of_interest):
        """Prepare a regional uncertainty estimator for a given area.

        Parameters
        ----------
        raster_data_handler : RasterDataHandler
            The data handler providing access to the underlying raster and its
            properties.
        variogram_analysis : VariogramAnalysis
            An instance with a fitted variogram model whose parameters
            (sills, ranges, nugget) will be used for uncertainty propagation.
        area_of_interest : str | Path | shapely.geometry.Polygon | shapely.geometry.MultiPolygon
            The spatial region over which to compute the mean uncertainty.  This
            may be supplied as a file path to a vector file (shapefile, GeoJSON,
            etc.), a ``Polygon`` or ``MultiPolygon``.  The geometry is
            dissolved into a single polygon internally.

        Raises
        ------
        ValueError
            If no geometries are found in a provided file or the resulting
            polygon is invalid.
        TypeError
            If the ``area_of_interest`` argument is of an unsupported type.
        """

        polygon = None
        if isinstance(area_of_interest, (str, Path)):
            # Handle file path input using geopandas
            gdf = gpd.read_file(area_of_interest)
            if gdf.empty:
                raise ValueError(f"No geometries found in file: {area_of_interest}")
            # Combine all geometries from the file into a single one
            polygon = gdf.unary_union
        elif isinstance(area_of_interest, (Polygon, MultiPolygon)):
            # Handle direct shapely geometry input
            polygon = area_of_interest
        else:
            raise TypeError(
                "area_of_interest must be a file path (str or Path) or a shapely Polygon/MultiPolygon."
            )

        # Ensure the final geometry is a single valid Polygon
        if isinstance(polygon, MultiPolygon):
            polygon = polygon.union()
        if not isinstance(polygon, Polygon) or polygon.is_empty or polygon.area <= 0:
            raise ValueError("Input must resolve to a single, valid Polygon with a non-zero area.")

        self.raster_data_handler = raster_data_handler
        self.variogram_analysis = variogram_analysis
        self.params = self.variogram_analysis.best_params
        self.polygon = polygon
        self.area = polygon.area
        self.sigma2 = (self.variogram_analysis.best_nugget or 0) +sum(self.variogram_analysis.sills)# σ_Δz^2 (sill variance)
        self.sigma2_min = (self.variogram_analysis.min_nugget or 0) + sum(self.variogram_analysis.sills_min)
        self.sigma2_max = (self.variogram_analysis.max_nugget or 0) + sum(self.variogram_analysis.sills_max)
        self.sills = self.variogram_analysis.sills
        self.sills_min = self.variogram_analysis.sills_min
        self.sills_max = self.variogram_analysis.sills_max
        self.ranges = self.variogram_analysis.ranges
        self.ranges_min = self.variogram_analysis.ranges_min
        self.ranges_max = self.variogram_analysis.ranges_max
        self.nugget = self.variogram_analysis.best_nugget
    
        self.best_model_config = self.variogram_analysis.best_model_config
        self.best_model_func = self.variogram_analysis.best_model_func
        
        # arrange parameters
        p1, p2, p3, p_tot = self.arrange_params(
            sills   = self.sills,
            ranges  = self.ranges,
            nugget  = self.nugget,
        )

        # store ready-to-use γ(h) callables
        self.gamma_func_total = self.bind_gamma(variogram_analysis.best_model_func, p_tot)
        self.gamma_func_1     = self.bind_gamma(variogram_analysis.best_model_func, p1)
        self.gamma_func_2     = self.bind_gamma(variogram_analysis.best_model_func, p2)
        self.gamma_func_3     = self.bind_gamma(variogram_analysis.best_model_func, p3)

        # and the min/max envelopes
        p1_min, p2_min, p3_min, p_tot_min = self.arrange_params(
            sills   = variogram_analysis.sills_min,
            ranges  = variogram_analysis.ranges_min,
            nugget  = variogram_analysis.min_nugget,
        )
        self.gamma_func_total_min = self.bind_gamma(variogram_analysis.best_model_func, p_tot_min)
        self.gamma_func_1_min     = self.bind_gamma(variogram_analysis.best_model_func, p1_min)
        self.gamma_func_2_min     = self.bind_gamma(variogram_analysis.best_model_func, p2_min)
        self.gamma_func_3_min     = self.bind_gamma(variogram_analysis.best_model_func, p3_min)
        
        p1_max, p2_max, p3_max, p_tot_max = self.arrange_params(
            sills   = variogram_analysis.sills_max,
            ranges  = variogram_analysis.ranges_max,
            nugget  = variogram_analysis.max_nugget,
        )
        self.gamma_func_total_max = self.bind_gamma(variogram_analysis.best_model_func, p_tot_max)
        self.gamma_func_1_max     = self.bind_gamma(variogram_analysis.best_model_func, p1_max)
        self.gamma_func_2_max     = self.bind_gamma(variogram_analysis.best_model_func, p2_max)
        self.gamma_func_3_max     = self.bind_gamma(variogram_analysis.best_model_func, p3_max)
        
        self.mean_random_correlated_1_max_polygon = None
        self.mean_random_correlated_2_max_polygon = None
        self.mean_random_correlated_3_max_polygon = None
        self.total_mean_correlated_uncertainty_polygon = None    
        self.mean_random_correlated_1_min_polygon = None
        self.mean_random_correlated_2_min_polygon = None    
        self.mean_random_correlated_3_min_polygon = None
        self.total_mean_correlated_uncertainty_min_polygon = None
        self.mean_random_correlated_1_polygon = None
        self.mean_random_correlated_2_polygon = None
        self.mean_random_correlated_3_polygon = None
        self.total_mean_correlated_uncertainty_max_polygon = None
        
        self.mean_random_correlated_1_raster = None
        self.mean_random_correlated_2_raster = None
        self.mean_random_correlated_3_raster = None
        self.total_mean_correlated_uncertainty_raster = None
        self.mean_random_correlated_1_min_raster = None
        self.mean_random_correlated_2_min_raster = None
        self.mean_random_correlated_3_min_raster = None
        self.total_mean_correlated_uncertainty_min_raster = None
        self.mean_random_correlated_1_max_raster = None
        self.mean_random_correlated_2_max_raster = None
        self.mean_random_correlated_3_max_raster = None
        self.total_mean_correlated_uncertainty_max_raster = None
        
        self.mean_random_uncorrelated = None
        
        self.total_mean_uncertainty_polygon = None
        self.total_mean_uncertainty_raster = None
        self.total_mean_uncertainty_min_polygon = None
        self.total_mean_uncertainty_min_raster = None
        self.total_mean_uncertainty_max_polygon = None
        self.total_mean_uncertainty_max_raster = None
        
    def estimate(self, func, method="analytical", **kwargs):
        """
        Estimate the regional uncertainty σ_A using the specified method.
        
        Parameters:
        - `method` (str): One of {"analytical", "brute_force", "monte_carlo", "fft"} (not case-sensitive).
        - `**kwargs`: Additional arguments specific to each method:
            * For `"analytical"`: `num_steps` (int, optional) – number of steps for radial integration.
            * For `"brute_force"`: `resolution` (float, optional) – grid cell size for discretization; or 
                                     `grid_points` (int, optional) – approximate number of grid cells along the longest polygon dimension.
            * For `"monte_carlo"`: `n_pairs` (int, optional) – number of random point pairs to sample.
            * For `"fft"`: `cell_size` (float, optional) – grid cell size for rasterization; or 
                            `grid_points` (int, optional) – number of cells along longest dimension if cell_size not provided.
        
        Returns:
        - `sigma_A` (float): The regional uncertainty (standard deviation) estimated by the chosen method.
        """
        method = method.lower()
        if method in ("analytical", "analytic"):
            return self.estimate_analytical(func, **kwargs)
        elif method in ("brute_force", "numerical"):
            return self.estimate_brute_force(func,**kwargs)
        elif method in ("monte_carlo", "montecarlo"):
            return self.estimate_monte_carlo(func,**kwargs)
        elif method in ("fft", "convolution"):
            return self.estimate_fft(func,**kwargs)
        elif method in ("hug", "hugonnet"):
            return self.estimate_hugonnet(func,**kwargs)
        elif method in ("raster"):
            return self.estimate_monte_carlo_raster(func,**kwargs)
        else:
            raise ValueError(f"Unknown method '{method}'. Valid options are 'analytical', 'brute_force', 'monte_carlo', or 'fft'.")

    def estimate_analytical(self,gamma_func, num_steps=1000):
        """
        Analytical (approximate) method for σ_A.
        
        Approximates the double integral by treating the area as an equivalent circle of the same area (Rolstad et al., 2009 approach). 
        The semivariogram is integrated radially from the center of this circle.
        
        **Note:** This method provides a fast closed-form or semi-analytical approximation, but it may overestimate σ_A for elongated 
        or irregular shapes (the disk approximation is conservative, as it assumes all points are as close as possible on average).
        
        Parameters:
        - `num_steps` (int): The number of radial steps for numerical integration (higher = more accurate integration of γ(h)).
        
        Returns:
        - `sigma_A` (float): The regional uncertainty (standard deviation) approximated analytically.
        """
        
    
        # --- geometry ---------------------------------------------------------
        R   = math.sqrt(self.area / math.pi)         # radius of equivalent disk
        dr  = R / num_steps

        # --- radial integral of the covariance -------------------------------
        #     I = ∫0^R r [σ² − γ(r)] dr   (mid-point rule)
        r_mid = (np.arange(num_steps) + 0.5) * dr
        cov   = self.sigma2 - gamma_func(r_mid)
        I     = np.sum(r_mid * cov) * dr            #  ∫ r C(r) dr

        # --- convert to σ_A² --------------------------------------------------
        sigma2_A = (2 * math.pi / self.area) * I    # Eq. above

        return math.sqrt(max(sigma2_A, 0.0))
    
    def estimate_brute_force(self, gamma_func, resolution=None, grid_points=100):
        """
        Brute-force numerical integration of σ_A.
        
        Discretizes the polygon area into a grid of points and explicitly sums the contributions 
        of all pairs of points within the polygon. This directly computes the double integral by summation.
        
        **Note:** This method is O(N^2) (where N is number of sample points inside the polygon) and can be very slow 
        for fine grids or large areas. It is primarily useful for validation or small areas.
        
        Parameters:
        - `resolution` (float, optional): Spacing between sample points (grid cell size). If not provided, 
                                          it will be determined such that the longest dimension of the polygon's 
                                          bounding box has `grid_points` cells.
        - `grid_points` (int, optional): Number of cells along the longest dimension (used to derive `resolution` if not given).
        
        Returns:
        - `sigma_A` (float): The regional uncertainty (standard deviation) from brute-force integration.
        """
        # Determine grid spacing (resolution)
        minx, miny, maxx, maxy = self.polygon.bounds
        width = maxx - minx
        height = maxy - miny
        if resolution is None:
            # Use grid_points to set resolution for the longest side of the bounding box
            if grid_points <= 0:
                raise ValueError("grid_points must be a positive integer.")
            longest_side = max(width, height)
            resolution = longest_side / grid_points
        if resolution <= 0:
            raise ValueError("resolution must be positive.")
        
        # Generate sample point coordinates (centers of grid cells) within the bounding box
        nx = int(math.ceil(width / resolution))
        ny = int(math.ceil(height / resolution))
        x_coords = [minx + (i + 0.5) * resolution for i in range(nx)]
        y_coords = [miny + (j + 0.5) * resolution for j in range(ny)]
        
        # Collect all points inside the polygon
        points = []
        for y in y_coords:
            for x in x_coords:
                if self.polygon.contains(Point(x, y)):
                    points.append((x, y))
        N = len(points)
        if N == 0:
            raise ValueError("No sample points found inside the polygon at the given resolution.")
        
        # Sum semivariogram values for all unique pairs of points (i<j)
        total_corr = 0.0
        for i in range(N):
            x1, y1 = points[i]
            for j in range(i + 1, N):
                x2, y2 = points[j]
                # Distance between point i and j
                dist = math.hypot(x2 - x1, y2 - y1)
                total_corr += (self.sigma2 - gamma_func(dist))
        # total_corr now is sum_{i<j} γ(|xi - xj|)
        # Each pair (i,j) corresponds to two ordered pairs in the double integral (i->j and j->i),
        # so multiply by 2 to account for both orientations.
        total_corr *= 2.0

        # Convert the summed semivariance to a continuous double integral value:
        # Each point represents an area element of size (resolution^2). For each pair, the contribution to ∬_A γ(h) is γ(dist) * (area_element_i) * (area_element_j).
        area_element = resolution ** 2
        integral_corr = total_corr * (area_element * area_element)

        # Now compute σ_A^2 = σ_Δz^2 – (1/A^2) * ∬_A γ(h) 
        sigma2_A = (integral_corr / (self.area ** 2))
        if sigma2_A < 0:
            sigma2_A = 0.0  # guard against slight negative due to numerical error
        return math.sqrt(sigma2_A)
    
    def estimate_monte_carlo(self, gamma_func, n_pairs=10_000, *, seed=None):
        """
        Monte-Carlo estimator of σ_A (area-averaged standard deviation).

        Parameters
        ----------
        gamma_func : callable
            Semivariogram γ(h).
        n_pairs : int, default 10000
            Number of independent point pairs.
        seed : int or None
            RNG seed for reproducibility.

        Returns
        -------
        float
            σ_A.
        """
        if n_pairs <= 0:
            raise ValueError("n_pairs must be positive.")
        if seed is not None:
            random.seed(seed)

        minx, miny, maxx, maxy = self.polygon.bounds
        get_point = lambda: next(
            p for p in iter(lambda: Point(random.uniform(minx, maxx),
                                        random.uniform(miny, maxy)), None)
            if self.polygon.contains(p)
        )

        total = 0.0
        for _ in range(n_pairs):
            p1, p2 = get_point(), get_point()
            h = p1.distance(p2)          # or math.hypot(...)
            total += self.sigma2 - gamma_func(h)

        sigma2_A = total / n_pairs
        return 0.0 if sigma2_A < 0 else math.sqrt(sigma2_A)
    
    def estimate_fft(self, gamma_func, cell_size=None, grid_points=200):
        """
        FFT convolution method for σ_A.
        
        Rasterizes the polygon onto a grid and computes the auto-correlation of the polygon shape 
        using FFTs. Then sums the covariance function over all pair separations weighted by the overlap area.
        
        This method converts the double integral into convolutions: 
        it computes the spatial overlap (auto-correlation) of the area for each separation distance 
        and multiplies by the covariance (σ_Δz^2 - γ) at that distance. Using FFT significantly speeds up 
        the calculation of the overlap for all possible shifts.
        
        **Note:** The polygon is discretized on a grid, so the accuracy depends on the grid resolution. 
        Ensure `cell_size` is sufficiently small relative to the spatial correlation range for accurate results. 
        
        Parameters:
        - `cell_size` (float, optional): The size of each grid cell in the same units as the polygon coordinates. 
                                         A smaller cell_size gives higher resolution (and more computation).
        - `grid_points` (int, optional): If `cell_size` is not specified, the number of grid cells along the longest 
                                         side of the polygon’s bounding box. This will determine `cell_size` as 
                                         (longest_side_length / grid_points).
        
        Returns:
        - `sigma_A` (float): The regional uncertainty (standard deviation) computed via FFT convolution.
        """
        # Determine grid resolution for rasterization
        minx, miny, maxx, maxy = self.polygon.bounds
        width = maxx - minx
        height = maxy - miny
        if cell_size is None:
            if grid_points <= 0:
                raise ValueError("grid_points must be a positive integer.")
            longest_side = max(width, height)
            cell_size = longest_side / grid_points
        if cell_size <= 0:
            raise ValueError("cell_size must be positive.")
        
        # Determine grid dimensions within bounding box
        nx = int(math.ceil(width / cell_size))
        ny = int(math.ceil(height / cell_size))
        # Pad the grid to avoid convolution wrap-around (pad to next power of 2 on each side)
        def next_power_of_two(n):
            """
            Return the next power of two greater than or equal to ``n``.

            Parameters
            ----------
            n : int
                Input integer for which to compute the power of two.

            Returns
            -------
            int
                The smallest power of two that is greater than or equal to ``n``.
            """
            return 1 << ((n - 1).bit_length())
        pad_x = next_power_of_two(2 * nx)
        pad_y = next_power_of_two(2 * ny)
        
        # Rasterize polygon onto grid (binary mask: 1 inside polygon, 0 outside)
        mask = np.zeros((pad_y, pad_x), dtype=float)
        # We'll place the polygon roughly centered in the padded grid to symmetrize the convolution space
        offset_x = (pad_x - nx) // 2
        offset_y = (pad_y - ny) // 2
        # Sample points at cell centers; mark mask cell as 1 if center is inside polygon
        y0 = miny + 0.5 * cell_size
        for j in range(ny):
            x0 = minx + 0.5 * cell_size
            y = y0 + j * cell_size
            for i in range(nx):
                x = x0 + i * cell_size
                if self.polygon.contains(Point(x, y)):
                    mask[offset_y + j, offset_x + i] = 1.0
        
        # Compute autocorrelation of mask using FFT (via convolution theorem)
        fft_mask = np.fft.rfft2(mask)  # real FFT for efficiency
        power_spectrum = fft_mask * np.conj(fft_mask)  # |FFT(mask)|^2
        corr = np.fft.irfft2(power_spectrum, s=mask.shape)  # inverse FFT to get convolution results
        corr = np.fft.fftshift(corr)  # shift zero-lag to center of array
        
        # Now 'corr' array holds the overlap area (in number of cells) of the polygon with itself 
        # at each displacement (i-offset in x, j-offset in y). We need to multiply this by covariance at that displacement.
        center_y, center_x = corr.shape[0] // 2, corr.shape[1] // 2
        # Loop over all displacements and accumulate covariance contribution
        total_covariance = 0.0
        for j in range(corr.shape[0]):
            dy = (j - center_y) * cell_size
            for i in range(corr.shape[1]):
                dx = (i - center_x) * cell_size
                # Distance for this displacement
                dist = math.hypot(dx, dy)
                # Covariance = σ_Δz^2 - γ(dist)
                cov = self.sigma2 - gamma_func(dist)
                # Overlap area for this displacement = corr[j, i] * (area of one cell)
                overlap_area = corr[j, i] * (cell_size ** 2)
                # Contribution to double integral: covariance * overlap_area
                total_covariance += cov * overlap_area
        # 'total_covariance' now approximates ∬_A [σ^2 - γ(h)] dx dy = σ_Δz^2 * A^2 - ∬_A γ(h) dx dy.
        # We want σ_A^2 = (1/A^2) * ∬_A [σ^2 - γ(h)] dx dy.
        sigma2_A = total_covariance / (self.area ** 2)
        if sigma2_A < 0:
            sigma2_A = 0.0
        return math.sqrt(sigma2_A)
    
    # ---------------------------------------------------------------------
    # Hugonnet et al. (2022) regional uncertainty – Monte-Carlo covariance
    # integration using K random “centre” samples inside the polygon.
    # Default K = 100 (the value recommended in the paper).
    # ---------------------------------------------------------------------
    def estimate_hugonnet(self,gamma_func, k: int = 100) -> float:
        """
        Hugonnet et al. (2022) regional uncertainty of the *mean* error
        inside the polygon.

        Parameters
        ----------
        k : int, default 100
            Number of random “centre” points (Hugonnet’s *K*).  Higher
            gives more accurate integration but scales linearly in
            runtime.

        Returns
        -------
        sigma_A : float
            One-sigma uncertainty (standard deviation) of the mean error
            over the polygon.
        """
        # ---- 1.  Build a list of all DEM-pixel centres that fall inside A
        #         (these represent the N points used in the double integral)
        rio_da = self.raster_data_handler.rioxarray_obj
        if rio_da is None:
            raise RuntimeError("Raster must be loaded before calling Hugonnet uncertainty")

        # pixel-centre coordinates
        xs = rio_da.x.values
        ys = rio_da.y.values
        Ncols, Nrows = len(xs), len(ys)

        # Generate a 2-D grid of all pixel centres, mask by polygon A
        pts   = []
        for j, y in enumerate(ys):
            for i, x in enumerate(xs):
                if self.polygon.contains(Point(float(x), float(y))):
                    pts.append((x, y))
        if not pts:
            raise RuntimeError("No DEM pixels found inside the polygon")
        pts = np.asarray(pts)                         # shape (N, 2)
        N   = pts.shape[0]

        # ---- 2.  Retrieve per-pixel vertical PRECISION σ²_dh for heteroscedasticity
        # If the user provided a per-pixel variance map, use it, else assume
        # a constant sill (conservative).
        if hasattr(self.raster_data_handler, "sigma2_map"):
            sigma2_map = self.raster_data_handler.sigma2_map  # same shape as raster
            # sample those pixel variances
            sig2_pixels = []
            for x, y in pts:
                col = np.argmin(np.abs(xs - x))
                row = np.argmin(np.abs(ys - y))
                sig2_pixels.append(float(sigma2_map[row, col]))
            sig2_pixels = np.asarray(sig2_pixels)
        else:
            # constant = total sill of variogram
            total_sill  = (self.nugget or 0.0) + np.sum(self.sills)
            sig2_pixels = np.full(N, total_sill, dtype=float)

        # average pixel variance σ²_dh|A (Eq. 19 in Hugonnet et al.)
        sigma2_dh_A = sig2_pixels.mean()

        # ---- 3.  Convenience: correlation ρ(d) = 1 – γ(d)/sill  (normalised)
        total_sill = (self.nugget or 0.0) + np.sum(self.sills)

        def rho(dist: np.ndarray) -> np.ndarray:
            """
            Compute the normalised correlation function ρ(d).

            Given an array of separation distances ``dist``, this function
            evaluates the variogram model at those distances, divides by the
            total sill (nugget plus sills), and returns ``1 – γ(d)/sill``.

            Parameters
            ----------
            dist : numpy.ndarray
                Array of separation distances at which to evaluate ρ.

            Returns
            -------
            numpy.ndarray
                The correlation values corresponding to each distance in ``dist``.
            """
            # γ(dist) from the fitted variogram model
            gamma_vals = gamma_func(dist)
            return 1.0 - (gamma_vals / total_sill)

        # ---- 4.  Draw K random “centre” points uniformly inside A
        rng = np.random.default_rng()
        centres = []
        minx, miny, maxx, maxy = self.polygon.bounds
        while len(centres) < k:
            cx = rng.uniform(minx, maxx)
            cy = rng.uniform(miny, maxy)
            if self.polygon.contains(Point(cx, cy)):
                centres.append((cx, cy))
        centres = np.asarray(centres)                 # (K, 2)

        # ---- 5.  Compute Hugonnet integral I = (1/K) Σ_k Σ_i ρ(d_ki)
        I_sum = 0.0
        for cx, cy in centres:
            # Euclidean distances to every pixel point
            dx = pts[:, 0] - cx
            dy = pts[:, 1] - cy
            d  = np.hypot(dx, dy)
            I_sum += rho(d).sum()
        I = I_sum / k                                # average over K

        # ---- 6.  regional variance σ²_A  (Eq. 18)
        sigma2_A = sigma2_dh_A * (I / N)

        # guard small numerical negatives
        if sigma2_A < 0:
            sigma2_A = 0.0

        return float(np.sqrt(sigma2_A))
    # --- FIX 1: robust level_of_detail selection ------------------------------
    def estimate_monte_carlo_raster(self, gamma_func, level_of_detail="detailed", n_pairs=10000):
        """Monte-Carlo estimator of σ_A (area-averaged standard deviation) on a raster."""
        if level_of_detail == "detailed":
            polygon = self.raster_data_handler.merged_geom
        elif level_of_detail in ("coarse", "box", "bbox"):
            polygon = self.raster_data_handler.bbox
        else:
            raise ValueError("level_of_detail must be 'detailed', 'coarse', 'box', or 'bbox'")

        if n_pairs <= 0:
            raise ValueError("n_pairs must be positive.")

        minx, miny, maxx, maxy = polygon.bounds
        get_point = lambda: next(
            p for p in iter(lambda: Point(random.uniform(minx, maxx),
                                           random.uniform(miny, maxy)), None)
            if polygon.contains(p)
        )

        total = 0.0
        for _ in range(n_pairs):
            p1, p2 = get_point(), get_point()
            h = p1.distance(p2)
            total += self.sigma2 - gamma_func(h)

        sigma2_A = total / n_pairs
        return 0.0 if sigma2_A < 0 else math.sqrt(sigma2_A)
    
    def calc_mean_random_uncorrelated(self):
        """
        Calculate the mean random uncorrelated uncertainty.
        
        Compute the mean random uncorrelated uncertainty by dividing the RMS by the square root of the length of the data array.
        The result is stored in the `mean_random_uncorrelated` attribute of the instance.
        """
        
        data = self.raster_data_handler.data_array

        def calculate_rms(values):
            """Calculate the Root Mean Square (RMS) of an array of numbers."""
            # Step 1: Square all the numbers
            squared_values = [x**2 for x in values]
            
            # Step 2: Calculate the mean of the squares
            mean_of_squares = sum(squared_values) / len(values)
            
            # Step 3: Take the square root of the mean
            rms = math.sqrt(mean_of_squares)
    
            return rms
        
        rms = calculate_rms(data)

        self.mean_random_uncorrelated = rms / np.sqrt(len(data))
    
    def calc_mean_random_correlated_polygon(self, n_pairs, seed):
        """Estimate mean correlated uncertainty within the polygon.

        This method computes the random correlated component of the mean
        uncertainty by Monte Carlo integration for each available
        semivariogram component (total, min, max, and up to three
        nested components).  The results are stored on the instance
        with attribute names reflecting the component and polygon.

        Parameters
        ----------
        n_pairs : int
            Number of random point pairs to sample for the Monte Carlo
            estimator.
        seed : int
            Random seed used to initialize the pseudo-random number
            generator for reproducible sampling.
        """
        if self.gamma_func_total:
            self.total_mean_correlated_uncertainty_polygon = self.estimate(self.gamma_func_total, method="monte_carlo", n_pairs=n_pairs, seed=seed)
        if self.gamma_func_total_min:
            self.total_mean_correlated_uncertainty_min_polygon = self.estimate(self.gamma_func_total_min, method="monte_carlo", n_pairs=n_pairs, seed=seed)
        if self.gamma_func_total_max:
            self.total_mean_correlated_uncertainty_max_polygon = self.estimate(self.gamma_func_total_max, method="monte_carlo", n_pairs=n_pairs, seed=seed)
        if self.gamma_func_1:
            self.mean_random_correlated_1_polygon = self.estimate(self.gamma_func_1, method="monte_carlo", n_pairs=n_pairs, seed=seed)
        if self.gamma_func_1_min:
            self.mean_random_correlated_1_min_polygon = self.estimate(self.gamma_func_1_min, method="monte_carlo", n_pairs=n_pairs, seed=seed)
        if self.gamma_func_1_max:
            self.mean_random_correlated_1_max_polygon = self.estimate(self.gamma_func_1_max, method="monte_carlo", n_pairs=n_pairs, seed=seed)
        if self.gamma_func_2:
            self.mean_random_correlated_2_polygon = self.estimate(self.gamma_func_2, method="monte_carlo", n_pairs=n_pairs, seed=seed)
        if self.gamma_func_2_min:
            self.mean_random_correlated_2_min_polygon = self.estimate(self.gamma_func_2_min, method="monte_carlo", n_pairs=n_pairs, seed=seed)
        if self.gamma_func_2_max:
            self.mean_random_correlated_2_max_polygon = self.estimate(self.gamma_func_2_max, method="monte_carlo", n_pairs=n_pairs, seed=seed)
        if self.gamma_func_3:
            self.mean_random_correlated_3_polygon = self.estimate(self.gamma_func_3, method="monte_carlo", n_pairs=n_pairs, seed=seed)
        if self.gamma_func_3_min:
            self.mean_random_correlated_3_min_polygon = self.estimate(self.gamma_func_3_min, method="monte_carlo", n_pairs=n_pairs, seed=seed)
        if self.gamma_func_3_max:
            self.mean_random_correlated_3_max_polygon = self.estimate(self.gamma_func_3_max, method="monte_carlo", n_pairs=n_pairs, seed=seed)

    # -------------------------------------------------------------------------
    # FIX 2: propagate n_pairs correctly & use central values in print_results
    # -------------------------------------------------------------------------
    def calc_mean_random_correlated_raster(self, level_of_detail, n_pairs):
        """Estimate mean correlated uncertainty via raster integration.

        Computes the mean random correlated component of the uncertainty
        over the raster's bounding box or a more detailed representation
        of the polygon, depending on ``level_of_detail``.  Results are
        stored as attributes on the estimator instance.

        Parameters
        ----------
        level_of_detail : str
            One of ``'detailed'``, ``'coarse'``, ``'box'`` or ``'bbox'``
            indicating how to approximate the polygon.  A detailed level
            uses the exact polygon shape; coarse/box/bbox successively
            simplify the geometry.
        n_pairs : int
            Number of random point pairs to use if Monte Carlo is
            employed internally (ignored for raster-based methods).
        """
        if self.gamma_func_total:
            self.total_mean_correlated_uncertainty_raster = self.estimate(self.gamma_func_total, method="raster", level_of_detail=level_of_detail, n_pairs=n_pairs)
        if self.gamma_func_total_min:
            self.total_mean_correlated_uncertainty_min_raster = self.estimate(self.gamma_func_total_min, method="raster", level_of_detail=level_of_detail, n_pairs=n_pairs)
        if self.gamma_func_total_max:
            self.total_mean_correlated_uncertainty_max_raster = self.estimate(self.gamma_func_total_max, method="raster", level_of_detail=level_of_detail, n_pairs=n_pairs)
        if self.gamma_func_1:
            self.mean_random_correlated_1_raster = self.estimate(self.gamma_func_1, method="raster", level_of_detail=level_of_detail, n_pairs=n_pairs)
        if self.gamma_func_1_min:
            self.mean_random_correlated_1_min_raster = self.estimate(self.gamma_func_1_min, method="raster", level_of_detail=level_of_detail, n_pairs=n_pairs)
        if self.gamma_func_1_max:
            self.mean_random_correlated_1_max_raster = self.estimate(self.gamma_func_1_max, method="raster", level_of_detail=level_of_detail, n_pairs=n_pairs)
        if self.gamma_func_2:
            self.mean_random_correlated_2_raster = self.estimate(self.gamma_func_2, method="raster", level_of_detail=level_of_detail, n_pairs=n_pairs)
        if self.gamma_func_2_min:
            self.mean_random_correlated_2_min_raster = self.estimate(self.gamma_func_2_min, method="raster", level_of_detail=level_of_detail, n_pairs=n_pairs)
        if self.gamma_func_2_max:
            self.mean_random_correlated_2_max_raster = self.estimate(self.gamma_func_2_max, method="raster", level_of_detail=level_of_detail, n_pairs=n_pairs)
        if self.gamma_func_3:
            self.mean_random_correlated_3_raster = self.estimate(self.gamma_func_3, method="raster", level_of_detail=level_of_detail, n_pairs=n_pairs)
        if self.gamma_func_3_min:
            self.mean_random_correlated_3_min_raster = self.estimate(self.gamma_func_3_min, method="raster", level_of_detail=level_of_detail, n_pairs=n_pairs)
        if self.gamma_func_3_max:
            self.mean_random_correlated_3_max_raster = self.estimate(self.gamma_func_3_max, method="raster", level_of_detail=level_of_detail, n_pairs=n_pairs)

    def calc_mean_uncertainty(self, n_pairs=100000, level_of_detail="detailed", seed=None):
        """Compute the total mean uncertainty for the area of interest.

        This convenience method invokes the uncorrelated, correlated
        polygon-based, and correlated raster-based estimators in turn,
        passing along the number of sample pairs and level of detail
        where appropriate.  After the component uncertainties are
        computed, the total mean uncertainty (and its minimum/maximum
        bounds if available) is assembled via quadrature and stored on
        the instance.  No values are returned directly.

        Parameters
        ----------
        n_pairs : int, optional
            Number of random point pairs to use in correlated uncertainty
            estimation (default is 100000).
        level_of_detail : str, optional
            Approximation level for raster-based uncertainty estimation
            (default is ``'detailed'``).  See
            :meth:`calc_mean_random_correlated_raster` for details.
        seed : int or None, optional
            Random seed for reproducible Monte Carlo sampling.  If
            ``None``, a random state will be used.
        """
        self.calc_mean_random_uncorrelated()
        self.calc_mean_random_correlated_polygon(n_pairs=n_pairs, seed=seed)
        # pass the same n_pairs to raster (was hard‑coded before)
        self.calc_mean_random_correlated_raster(level_of_detail=level_of_detail, n_pairs=n_pairs)
        if self.mean_random_uncorrelated and self.total_mean_correlated_uncertainty_polygon:
            self.total_mean_uncertainty_polygon = math.sqrt(
                self.mean_random_uncorrelated ** 2 +
                self.total_mean_correlated_uncertainty_polygon ** 2)
        if self.mean_random_uncorrelated and self.total_mean_correlated_uncertainty_min_polygon:
            self.total_mean_uncertainty_min_polygon = math.sqrt(
                self.mean_random_uncorrelated ** 2 +
                self.total_mean_correlated_uncertainty_min_polygon ** 2)
        if self.mean_random_uncorrelated and self.total_mean_correlated_uncertainty_max_polygon:
            self.total_mean_uncertainty_max_polygon = math.sqrt(
                self.mean_random_uncorrelated ** 2 +
                self.total_mean_correlated_uncertainty_max_polygon ** 2)
        if self.mean_random_uncorrelated and self.total_mean_correlated_uncertainty_raster:
            self.total_mean_uncertainty_raster = math.sqrt(
                self.mean_random_uncorrelated ** 2 +
                self.total_mean_correlated_uncertainty_raster ** 2)
        if self.mean_random_uncorrelated and self.total_mean_correlated_uncertainty_min_raster:
            self.total_mean_uncertainty_min_raster = math.sqrt(
                self.mean_random_uncorrelated ** 2 +
                self.total_mean_correlated_uncertainty_min_raster ** 2)
        if self.mean_random_uncorrelated and self.total_mean_correlated_uncertainty_max_raster:
            self.total_mean_uncertainty_max_raster = math.sqrt(
                self.mean_random_uncorrelated ** 2 +
                self.total_mean_correlated_uncertainty_max_raster ** 2)

    def print_results(self) -> None:
        """
        Nicely format all stored results.
        Lines whose key quantity is still None are omitted.
        """

        # ------------------------------------------------------------------ helpers
        def _fmt(v, f=".4f"):
            """Return formatted value or 'NA' if None (but we won’t print NA lines)."""
            return f"{v:{f}}" if v is not None else None

        def _triple(label, central, vmin, vmax, f=".4f"):
            """
            Print 'label central; min: …; max: …' if *any*
            of the three numbers is not-None.
            """
            pieces = []
            if central is not None:
                pieces.append(_fmt(central, f))
            if vmin is not None:
                pieces.append(f"min: {_fmt(vmin, f)}")
            if vmax is not None:
                pieces.append(f"max: {_fmt(vmax, f)}")
            if pieces:
                print(f"{label}{'; '.join(pieces)}")

        # ----------------------------------------------------------- header stuff
        print("Variogram Analysis Results:")
        if self.ranges is not None:
            print(
                f"Ranges: {self.ranges}"
                f"{'; min: ' + str(self.ranges_min) if self.ranges_min is not None else ''}"
                f"{'; max: ' + str(self.ranges_max) if self.ranges_max is not None else ''}"
            )
        if self.sills is not None:
            print(
                f"Sills: {self.sills}"
                f"{'; min: ' + str(self.sills_min) if self.sills_min is not None else ''}"
                f"{'; max: ' + str(self.sills_max) if self.sills_max is not None else ''}"
            )
        if self.nugget is not None:
            print(f"Nugget: {self.nugget:.4f}")
        print(f"Best Model Parameters: {self.variogram_analysis.best_params}")

        # --------------------------------------------------- polygon uncertainties
        print("\nUncertainty Results for Polygon of interest:")
        print(f"Polygon Area (m²): {self.area:.2f}")
        if self.mean_random_uncorrelated is not None:
            print(f"Mean Random Uncorrelated Uncertainty: {self.mean_random_uncorrelated:.4f}")

        _triple("Mean Random Correlated 1: ",
                self.mean_random_correlated_1_polygon,
                self.mean_random_correlated_1_min_polygon,
                self.mean_random_correlated_1_max_polygon)
        _triple("Mean Random Correlated 2: ",
                self.mean_random_correlated_2_polygon,
                self.mean_random_correlated_2_min_polygon,
                self.mean_random_correlated_2_max_polygon)
        _triple("Mean Random Correlated 3: ",
                self.mean_random_correlated_3_polygon,
                self.mean_random_correlated_3_min_polygon,
                self.mean_random_correlated_3_max_polygon)
        _triple("Total Mean Correlated Uncertainty (Polygon): ",
                self.total_mean_correlated_uncertainty_polygon,
                self.total_mean_correlated_uncertainty_min_polygon,
                self.total_mean_correlated_uncertainty_max_polygon)
        _triple("Total Mean Uncertainty (Polygon): ",
                self.total_mean_uncertainty_polygon,
                self.total_mean_uncertainty_min_polygon,
                self.total_mean_uncertainty_max_polygon)

        # ------------------------------------------------------- raster section
        print("\nUncertainty Results for Raster:")
        if getattr(self.raster_data_handler, "detailed_area", None) is not None:
            print(f"Detailed raster Area (m²): {self.raster_data_handler.detailed_area:.2f}")
        print(f"Coarse raster Area (m²): {self.raster_data_handler.bbox.area:.2f}")

        _triple("Mean Random Correlated 1 (Raster): ",
                self.mean_random_correlated_1_raster,
                self.mean_random_correlated_1_min_raster,
                self.mean_random_correlated_1_max_raster)
        _triple("Mean Random Correlated 2 (Raster): ",
                self.mean_random_correlated_2_raster,
                self.mean_random_correlated_2_min_raster,
                self.mean_random_correlated_2_max_raster)
        _triple("Mean Random Correlated 3 (Raster): ",
                self.mean_random_correlated_3_raster,
                self.mean_random_correlated_3_min_raster,
                self.mean_random_correlated_3_max_raster)
        _triple("Total Mean Correlated Uncertainty (Raster): ",
                self.total_mean_correlated_uncertainty_raster,
                self.total_mean_correlated_uncertainty_min_raster,
                self.total_mean_correlated_uncertainty_max_raster)
        _triple("Total Mean Uncertainty (Raster): ",
                self.total_mean_uncertainty_raster,
                self.total_mean_uncertainty_min_raster,
                self.total_mean_uncertainty_max_raster)

class ApplyUncertainty:
    """
    Compute spatial (correlated + uncorrelated) uncertainties from variogram parameters,
    and compute RMS from a GeoTIFF band.
    """

    @staticmethod
    def compute_spatial_uncertainties(
        ranges: Sequence[float],
        sills: Sequence[float],
        area: float,
        resolution: float,
        rms: Optional[float] = None,
        sills_min: Optional[Sequence[float]] = None,
        ranges_min: Optional[Sequence[float]] = None,
        sills_max: Optional[Sequence[float]] = None,
        ranges_max: Optional[Sequence[float]] = None
    ) -> Dict[str, Any]:
        """
        Compute mean uncorrelated & correlated uncertainty terms and their quadrature sum.

        Parameters
        ----------
        ranges : sequence of float
            Range parameters from variogram (same units as resolution).
        sills : sequence of float
            Sill parameters from variogram (variance units).
        area : float
            Total sampling area (in same linear units squared as ranges).
        resolution : float
            Raster cell size (same linear units as ranges).
        rms : float, optional
            Root‐mean‐square of your data array. If supplied, used for uncorrelated term.
        sills_min, ranges_min, sills_max, ranges_max : sequences, optional
            Percentile bounds for sills/ranges to get total_min/total_max.

        Returns
        -------
        dict
            {
              'uncorrelated': float or None,
              'correlated': List[float],
              'total': float,
              'total_min': float or None,
              'total_max': float or None
            }
        """
        # effective sample count
        n = area / (resolution ** 2)

        # uncorrelated term (if rms given)
        uncorr = (rms / math.sqrt(n)) if rms is not None else None

        # correlated terms
        corr = []
        for sill, rng in zip(sills, ranges):
            term = (math.sqrt(2 * sill) / math.sqrt(n)) * \
                   math.sqrt((math.pi * rng**2) / (5 * resolution**2))
            corr.append(term)

        # total quadrature sum
        total_sq = sum(c**2 for c in corr) + ((uncorr**2) if uncorr is not None else 0)
        total = math.sqrt(total_sq)

        # optional bounds
        total_min = total_max = None
        if sills_min and ranges_min:
            corr_min = [
                (math.sqrt(2 * smin) / math.sqrt(n)) * math.sqrt((math.pi * rmin**2) / (5 * resolution**2))
                for smin, rmin in zip(sills_min, ranges_min)
            ]
            total_min = math.sqrt(sum(c**2 for c in corr_min) + ((uncorr**2) if uncorr is not None else 0))

        if sills_max and ranges_max:
            corr_max = [
                (math.sqrt(2 * smax) / math.sqrt(n)) * math.sqrt((math.pi * rmax**2) / (5 * resolution**2))
                for smax, rmax in zip(sills_max, ranges_max)
            ]
            total_max = math.sqrt(sum(c**2 for c in corr_max) + ((uncorr**2) if uncorr is not None else 0))

        return {
            'uncorrelated': uncorr,
            'correlated': corr,
            'total': total,
            'total_min': total_min,
            'total_max': total_max
        }

    @staticmethod
    def compute_rms_from_tif(
        tif_path: str,
        band: int = 1
    ) -> float:
        """
        Compute the root‐mean‐square of a GeoTIFF band, ignoring nodata and NaNs.

        Parameters
        ----------
        tif_path : str
            Path to the input GeoTIFF.
        band : int, default 1
            Raster band to read.

        Returns
        -------
        float
            RMS of all valid (non‐nodata, non‐NaN) pixels.
        """
        with rasterio.open(tif_path) as src:
            arr = src.read(band).astype(float)
            nodata = src.nodata

        valid = ~np.isnan(arr)
        if nodata is not None:
            valid &= (arr != nodata)

        vals = arr[valid]
        if vals.size == 0:
            raise ValueError("No valid pixels found (all nodata or NaN).")
        return float(np.sqrt(np.mean(vals**2)))