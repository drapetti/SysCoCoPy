#!/usr/bin/env python
# coding: utf-8

"""
Calculates quaternions statistics and forms a design matrix. 
Based on stats.py and quaternion.py from Evan Tey and Chelsea Huang.
"""
import numpy as np
import scipy.stats as spstats
import warnings
import pandas as pd
from functools import partial
from astropy.io import fits
from multiprocessing import Pool,set_start_method
    
#warnings.filterwarnings('ignore')

#Based on stats.py from Evan Tey and Chelsea Huang
class BinnedStatistics(object):
    """An object for computing statistics within bins of time-series data."""

    def __init__(self, time_series, data_series, time_bins, pool=None):
        """An object for computing statistics within bins of time-series data.

        Arguments
        =========
        time_series: An array of sorted time series.
        data_series: A multi-dimensional array of data associated with the
            time series. The first dimension of data_series must match the
            length of time_series.
        time_bins: Bin boundaries for computing statistics.
        pool: an open multiprocessing Pool (optional)
        """
        self.time_series = time_series
        self.data_series = data_series
        self.time_bins = time_bins
        # Find the indices to split the quat_coeffs into bins.
        self.bin_index = np.searchsorted(time_series, time_bins, side="right")
        # Only keep the list of binned quat coeffs.
        # Discard the first and last bin, which are out of range.
        self.split_quat_coeffs = np.split(data_series, self.bin_index)[1:-1]
        self.pool = pool

    @property
    def average_time_bins(self):
        split_time_bins = np.split(self.time_series, self.bin_index)[1:-1]
        return np.array(list(map(np.average, split_time_bins)))

    def binned_statistic(self, stat_func):
        # Apply stat_func to each bin.
        if self.pool is not None:
            return np.array(self.pool.map(stat_func, self.split_quat_coeffs))
        else:
            binned_stat = list(map(stat_func, self.split_quat_coeffs))
            return np.array(list(binned_stat))

    def sum(self, *args, **kwargs):
        return self.binned_statistic(partial(np.nansum, *args, **kwargs))

    def average(self, *args, **kwargs):
        return self.binned_statistic(partial(np.nanmean, *args, **kwargs))

    def std(self, *args, **kwargs):
        return self.binned_statistic(partial(np.nanstd, *args, **kwargs))

    def skew(self, *args, **kwargs):
        return self.binned_statistic(partial(
            spstats.skew, nan_policy="omit", *args, **kwargs))

# Based on quaternion.py from Evan Tey and Chelsea Huang
def get_binned_quat_stats(quat_times, quats, bin_edges, pool=None):
    """Calculate binned quaternion statistics.

    Args:
        quat_times: [float], times
        quats: np.array with shape (3, ?), containing 3 quaternion time series.
               each must be the same length as quat_times
        bin_edges: [float], time bin edges. must be in same units as quat_times
        pool: (optional) an open multiprocessing Pool.

    Result:
        dict, holding binned quaternion statistics, e.g. q1_mean holds binned 
        means for q1 each binned statistic series will have length 
        len(bin_edges) - 1
    """
    quat_dict = {}
    for i, quat in enumerate(quats):
        binned_quat = BinnedStatistics(quat_times, quat / 5.0e-5, 
                                       bin_edges, pool=pool)

        quat_dict["q{}_mean".format(i + 1)] = binned_quat.average()
        quat_dict["q{}_std".format(i + 1)] = binned_quat.std()
        quat_dict["q{}_skew".format(i + 1)] = binned_quat.skew()

    return pd.DataFrame.from_dict(quat_dict)

def get_quat_flags(binned_quat_df):
    """Takes a dataframe of binned quaternion stats and returns quality flags.

    Args:
        binned_quat_df: pandas df with columns like q1_mean for quaternions in
                        [q1, q2, q3] and stats in [mean, std, skew]

    Returns:
        np.array([int]), quality flags based on the std of quaternions within
        a bin and whether or not the binned statistics are nans
    """
    index = np.zeros(binned_quat_df.shape[0]).astype(bool)
    for i in range(1, 4):
        index |= binned_quat_df["q{}_std".format(i)] > 0.16
        for stat in ["mean", "std", "skew"]:
            index |= np.isnan(binned_quat_df["q{}_{}".format(i, stat)])

    return index.astype(int)

# Quaternions reader script from Evan Tey and Chelsea Huang
def read_quat_file(fname, camera):
    # Read a mast quaternion fits file. Return time values 
    # and a (3, ?) np array of quaternions)
    quatdata = fits.open(fname)[camera].data
    columns=["TIME"]+["C{cam}_Q{i}".format(cam=camera, i=i) for i in [1, 2, 3]]
    quat_times = quatdata["TIME"]

    columns = ["C{cam}_Q{i}".format(cam=camera, i=i) for i in [1, 2, 3]]
    quaternions = quatdata.view(quatdata.dtype, np.ndarray)[columns]

    # Hack because FITS_rec doesn't support column 
    # selection: https://github.com/astropy/astropy/issues/6820
    return quat_times, np.array(quaternions.tolist()).T # slow, but works

def filtered_quat_dm(lc, quat_file, three_moments=True):
    """
    Returns a quaternion design matrix to capture TESS jitter with
    flagged entries filtered out.

    lc            : light curve
    quat_file     : quaternion fits file 
    three_moments : True  -> mean, standard deviation and skewness
                    False -> mean, standard deviation
    """

    from lightkurve.correctors.designmatrix import DesignMatrix
    
    # Quaternion statistics calculator script from Evan Tey and Chelsea Huang
    # Calculate binned quaternion statistics to match lc times
    qtimes, quats =  read_quat_file(quat_file, 4)

    texp = 30 / 60 / 24
    texp = lc.time.value[1] - lc.time.value[0] # assumes qtimes happens at 
                                               # regular intervals
    time_edges = list(lc.time.value - texp / 2) + [lc.time.value[-1] + texp / 2]

    set_start_method('fork',force=True)
    with Pool(None) as pool:
        orbit_quat_stat = get_binned_quat_stats(qtimes, quats, 
                                                np.array(time_edges), pool)
        orbit_quat_stat["flag"] = get_quat_flags(orbit_quat_stat)

    orbit_quat_stat.head()
    # End of quaternion statistics calculator script

    filter_mask = orbit_quat_stat.flag == 1
    filtered_orbit_quat_stat=orbit_quat_stat.where(~filter_mask,other=1)

    if three_moments==True:
        filtered_quat_dm = DesignMatrix(
        filtered_orbit_quat_stat,name="Filtered Quaternions")
    else:
        filtered_quat_dm = DesignMatrix(
        filtered_orbit_quat_stat[[
            "q1_mean", "q1_std", "q2_mean","q2_std", "q3_mean", "q3_std"]],
        name="Filtered Quaternions")

    return filtered_quat_dm

    