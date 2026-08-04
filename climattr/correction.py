import numpy as np
import xarray as xr

from datetime import datetime
from typing import Sequence, Union

from climattr.validator import validate_correction_method

def _distribution_dims(
    array: Union[xr.DataArray, xr.Dataset],
    sample_dims: Sequence[str]) -> list:
    """
    The subset of `sample_dims` actually present in `array`'s dims - the
    dimensions that represent draws from the same underlying distribution
    (e.g. repeated time steps, ensemble members) and so should be pooled
    together, rather than independent grid points (e.g. 'lat'/'lon').
    """
    return [d for d in sample_dims if d in array.dims]

###############################################################################

def _select_baseline(
    array: Union[xr.DataArray, xr.Dataset],
    idate: Union[datetime, None],
    edate: Union[datetime, None]) -> Union[xr.DataArray, xr.Dataset]:
    """
    Restrict `array` to `[idate, edate]` along 'time', if `array` actually
    has a 'time' dimension and both dates were given; otherwise return it
    unchanged (e.g. an ensemble-only array with no time axis to window).
    """
    if idate is not None and edate is not None and 'time' in array.dims:
        return array.sel(time=slice(idate, edate))

    return array

###############################################################################

def scaling(
    data: Union[xr.DataArray, xr.Dataset],
    clim: Union[xr.DataArray, xr.Dataset],
    idate: Union[datetime, None] = None,
    edate: Union[datetime, None] = None,
    method: str = 'add',
    sample_dims: Sequence[str] = ('time', 'ensemble')) -> Union[xr.DataArray, xr.Dataset]:
    """
    Scale the input data by adjusting it based on the climate mean over a
    specified period. The scaling can be done either by subtracting the mean
    (additive scaling) or by dividing by the mean (multiplicative scaling).

    The mean is computed per grid point: only the dims in `sample_dims`
    that `clim` actually has are reduced (e.g. an ensemble dimension is
    more draws from the same distribution, not an independent axis), so
    any other dims (e.g. 'lat'/'lon') are preserved and each grid point is
    corrected using its own climatology rather than a single domain-wide
    average.

    Parameters
    ----------
    data : Union[xr.DataArray, xr.Dataset]
        The data to be scaled, typically representing climate variables
        (e.g., temperature, precipitation).

    clim : Union[xr.DataArray, xr.Dataset]
        The climatology data used to calculate the mean for scaling. This
        should cover the same variable as 'data' over a baseline period.

    idate : datetime, optional
        The start date of the period over which the climatology mean is
        calculated. Ignored if `clim` has no 'time' dimension (e.g. it's
        ensemble-only). Default is None (use all of `clim`).

    edate : datetime, optional
        The end date of the period over which the climatology mean is
        calculated. See `idate`.

    method : str, optional, default = 'add'
        The method of scaling. If 'add', the climate mean is subtracted
        from the data (additive scaling). If 'mult', the data is divided
        by the climate mean (multiplicative scaling).

    sample_dims : Sequence[str], optional, default = ('time', 'ensemble')
        Dimension names that represent draws from the same underlying
        distribution and should be pooled together when computing the
        climatological mean, rather than treated as independent grid
        dimensions. Only the ones actually present in `clim` are used -
        pass e.g. `('time', 'realization')` if your ensemble dimension is
        named differently.

    Returns
    -------
    Union[xr.DataArray, xr.Dataset]
        The scaled data, same type/shape as `data`, with adjustments
        applied based on the specified method and (per grid point, if
        applicable) climatology mean.
    """
    # validate method
    validate_correction_method(method)

    clim = _select_baseline(clim, idate, edate)

    # pool every dim in `sample_dims` that's actually present into the
    # climatological mean per grid point, rather than collapsing to one
    # domain-wide scalar or leaving it unreduced as if it were an
    # independent broadcast dim
    reduce_dims = _distribution_dims(clim, sample_dims)
    if reduce_dims:
        clim = clim.mean(reduce_dims, skipna=True)

    if method == 'add':
        scaled_data = data - clim
    else:
        scaled_data = data / clim

    return scaled_data

###############################################################################

def quantile_mapping(
    data: Union[xr.DataArray, xr.Dataset],
    clim: Union[xr.DataArray, xr.Dataset],
    idate: Union[datetime, None] = None,
    edate: Union[datetime, None] = None,
    n_quantiles: int = 100,
    sample_dims: Sequence[str] = ('time', 'ensemble')) -> Union[xr.DataArray, xr.Dataset]:
    """
    Bias-correct the input data using empirical quantile mapping against a
    reference climatology.

    Builds an empirical quantile-quantile transfer function between `data`'s
    own distribution and `clim`'s (the reference/observed distribution)
    over a shared baseline period, then applies it to every value of `data`
    (baseline and beyond): each value is replaced by the value at the same
    percentile rank in the reference distribution. Unlike `scaling`, which
    only corrects the mean, this also corrects differences in spread and
    shape between the two distributions.

    Works with an `xr.DataArray` (a single time series, or gridded with
    e.g. 'time'/'lat'/'lon' dims - each grid point is corrected
    independently) or an `xr.Dataset` (each data variable is corrected
    independently, using the variable of the same name in `clim`). Any
    dimension in `sample_dims` (e.g. 'time' and/or an ensemble axis) is
    pooled together when building the transfer function and gets mapped
    elementwise through it; every other dimension (e.g. 'lat'/'lon') is
    corrected independently.

    Parameters
    ----------
    data : Union[xr.DataArray, xr.Dataset]
        The data to be bias-corrected, typically representing climate
        variables (e.g., temperature, precipitation).

    clim : Union[xr.DataArray, xr.Dataset]
        The reference/observed data used to build the target distribution
        for the mapping. Must share `data`'s non-sample dimensions (e.g.
        the same spatial grid) and, if a Dataset, its variable names.

    idate : datetime, optional
        The start date of the baseline period used to build the transfer
        function. Ignored if `data`/`clim` have no 'time' dimension (e.g.
        they're ensemble-only). Default is None (use the whole record).

    edate : datetime, optional
        The end date of the baseline period used to build the transfer
        function. See `idate`.

    n_quantiles : int, optional, default = 100
        The number of quantiles used to build the empirical transfer
        function between the two distributions.

    sample_dims : Sequence[str], optional, default = ('time', 'ensemble')
        Dimension names that represent draws from the same underlying
        distribution and should be pooled together when building the
        transfer function, rather than treated as independent grid
        dimensions. Only the ones actually present in `data`/`clim` are
        used - pass e.g. `('time', 'realization')` if your ensemble
        dimension is named differently.

    Returns
    -------
    Union[xr.DataArray, xr.Dataset]
        The bias-corrected data, same type as `data`, with every value
        mapped through the empirical quantile-quantile transfer function.
        Values outside the baseline period's range are clipped to the
        nearest quantile's correction (`numpy.interp`'s default
        extrapolation behaviour) rather than extrapolated.
    """
    data_base = _select_baseline(data, idate, edate)
    clim_base = _select_baseline(clim, idate, edate)

    quantiles = np.linspace(0, 1, n_quantiles)

    # dims to pool when building the transfer function (and later map
    # elementwise) - based on `data`/`clim` rather than the baseline-
    # windowed versions, since restricting 'time' to [idate, edate] doesn't
    # change which dimensions are present, only 'time''s own extent
    data_sample_dims = _distribution_dims(data, sample_dims)
    clim_sample_dims = _distribution_dims(clim, sample_dims)

    # reduce each baseline slice to its empirical quantiles first (a
    # 'quantile' dim, not 'time'/'ensemble'/...) - `data` (full length) and
    # the baseline slices (shorter, different time coverage) can't be
    # passed into a single apply_ufunc call directly, since xarray aligns
    # same-named dims (e.g. 'time') by coordinate label across all inputs
    # and the two extents don't match
    data_q = data_base.quantile(quantiles, dim=data_sample_dims, skipna=True)
    clim_q = clim_base.quantile(quantiles, dim=clim_sample_dims, skipna=True)

    def _quantile_map(values, data_q_values, clim_q_values):
        return np.interp(values, data_q_values, clim_q_values)

    corrected_data = xr.apply_ufunc(
        _quantile_map,
        data, data_q, clim_q,
        input_core_dims=[data_sample_dims, ['quantile'], ['quantile']],
        output_core_dims=[data_sample_dims],
        vectorize=True
    )

    # use .sizes (an ordered mapping on both DataArray and Dataset) rather
    # than .dims, whose return type differs/is being deprecated on Dataset
    return corrected_data.transpose(*data.sizes)

###############################################################################

