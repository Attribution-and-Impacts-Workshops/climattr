import numpy as np
import xarray as xr

from typing import List, Sequence, Union

from climattr.utils import get_percentiles_from_ci

# Helpers shared between climattr.attribution.risk_based (event-shifting +
# stationary refit) and climattr.attribution.wwa (direct nonstationary
# regression eval), so neither module depends on the other's internals.

def _calc_bootstrap_ensemble(
    data: np.ndarray,
    direction: str = "ascending",
    boot_size: int = 1000,
    seed: int = 42) -> np.ndarray:
    """
    Generates bootstrap ensembles from the input data and sorts them in the
    specified direction.

    This function creates multiple bootstrap samples from the input data, sorts
    each sample, and optionally reverses the order of the sorted samples based on
    the specified direction.

    Parameters
    ----------
    data : numpy.ndarray
        An array of shape (n_samples,) containing the input data from which
        bootstrap samples will be drawn.

    direction : str, optional
        The direction in which to sort the bootstrap samples. Can be either
        "ascending" (default) or "descending".

    boot_size : int, optional
        The number of bootstrap samples to generate. Default is 1000.

    Returns
    -------
    numpy.ndarray
        A 2D array of shape (boot_size, n_samples) where each row is a sorted
        bootstrap sample drawn from the input data. The samples are sorted in the
        specified direction.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.array([3.2, 1.5, 4.7, 2.8])
    >>> result = _calc_bootstrap_ensemble(data, direction="ascending", boot_size=3)
    >>> result
    array([[1.5, 2.8, 3.2, 4.7],
           [1.5, 2.8, 3.2, 4.7],
           [1.5, 3.2, 3.2, 4.7]])

    >>> result = _calc_bootstrap_ensemble(data, direction="descending", boot_size=2)
    >>> result
    array([[4.7, 3.2, 3.2, 1.5],
           [4.7, 3.2, 2.8, 1.5]])

    """
    np.random.seed(seed)

    # Flatten the input data
    n_samples = data.shape[0]

    # Generate the bootstrap samples using np.random.choice
    sample_store = np.random.choice(data, (int(boot_size), n_samples), replace=True)

    # Sort each row in the sample_store
    sample_store.sort(axis=1)

    # Reverse the rows if direction is "descending"
    if direction == "descending":
        sample_store = sample_store[:, ::-1]

    return sample_store

###############################################################################

def _calc_fitted_curve_confidence(
    data: np.ndarray,
    fit_function,
    prob_grid: np.ndarray,
    bootstrap_ci: int = 95,
    boot_size: int = 1000,
    seed: int = 42) -> np.ndarray:
    """
    Calculates a confidence band for the fitted curve itself, including the
    portion extrapolated beyond the observed record.

    The distribution is refit on `boot_size` resampled versions of `data`
    (parametric bootstrap), and each refit is evaluated at the same grid of
    cumulative probabilities (`prob_grid`) used for the point-estimate curve.
    Taking percentiles across replicates at every point in the grid therefore
    yields an uncertainty band that lines up with the fitted curve at every
    return period, including the extrapolated tail.

    Parameters
    ----------
    data : numpy.ndarray
        The observed data used to fit the distribution.

    fit_function : callable
        A scipy.stats-like distribution exposing `.fit()` and `.ppf()`.

    prob_grid : numpy.ndarray
        The cumulative probabilities at which to evaluate the fitted curve.

    bootstrap_ci : int, optional
        The confidence interval percentage. Default is 95.

    boot_size : int, optional
        The number of bootstrap replicates. Default is 1000.

    seed : int, optional
        Random seed for reproducibility. Default is 42.

    Returns
    -------
    numpy.ndarray
        A 2D array of shape (2, len(prob_grid)) with the lower and upper
        confidence bounds of the fitted curve at each point of `prob_grid`.
    """
    ci_inf, ci_sup = get_percentiles_from_ci(bootstrap_ci)

    rng = np.random.default_rng(seed)
    n_samples = data.shape[0]

    curves = np.empty((boot_size, prob_grid.shape[0]))
    for boot in range(boot_size):
        sample = rng.choice(data, size=n_samples, replace=True)
        params_boot = fit_function.fit(sample, loc=sample.mean(), scale=sample.std())
        curves[boot] = fit_function.ppf(prob_grid, *params_boot)

    return np.nanpercentile(curves, [ci_inf, ci_sup], axis=0)

###############################################################################

def _pr_calculation(
    all_array: np.ndarray,
    nat_array: np.ndarray,
    fit_function,
    thresh: int,
    direction: str = 'descending',
    params_all: tuple | None = None,
    params_nat: tuple | None = None) -> float:
    """
    Calculates the probability ratio (PR) between two datasets.

    This function fits the input datasets to a specified distribution using the
    provided fit function and calculates the probability ratio for a given threshold.

    Parameters
    ----------
    all_array : numpy.ndarray
        An array of shape (n_samples,) containing the "all" scenario data.

    nat_array : numpy.ndarray
        An array of shape (n_samples,) containing the "natural" scenario data.

    fit_function : callable
        A function that fits the input data to a distribution.

    thresh : int
        The threshold value for which the probability ratio will be calculated.

    direction : str, optional
        The direction in which to calculate the probability ratio. Default is
        "descending".

    Returns
    -------
    float
        The calculated probability ratio.
    """
    # if parameters are not specific then we fit the data to the distribution and get the parameters
    if not params_all:
        params_all = fit_function.fit(all_array, loc=all_array.mean(), scale=all_array.std())

    if not params_nat:
        params_nat = fit_function.fit(nat_array, loc=nat_array.mean(), scale=all_array.std())

    if direction == 'descending':
        probability_all = fit_function.sf(thresh, *params_all)
        probability_nat = fit_function.sf(thresh, *params_nat)
    else:
        probability_all = fit_function.cdf(thresh, *params_all)
        probability_nat = fit_function.cdf(thresh, *params_nat)

    pr = probability_all / probability_nat

    return pr

###############################################################################

def _rp_calculation(
    data: np.ndarray,
    fit_function,
    thresh: float,
    direction: str = 'descending',
    params: tuple | None = None) -> float:
    """
    Calculates the return period for a given threshold in the dataset.

    This function fits the input data to a specified distribution and calculates
    the return period for a given threshold value.

    Parameters
    ----------
    data : numpy.ndarray
        An array of shape (n_samples,) containing the input data.

    fit_function : callable
        A function that fits the input data to a distribution.

    thresh : float
        The threshold value for which the return period will be calculated.

    direction : str, optional
        The direction in which to calculate the return period. Default is "descending".

    Returns
    -------
    float
        The calculated return period for the given threshold.
    """
    if not params:
        params = fit_function.fit(data, loc=data.mean(), scale=data.std())

    if direction == 'descending':
        sf = fit_function.sf(thresh, *params)
        if sf == 0:
            rp = np.nan
        else:
            rp = 1 / sf
    else:
        cdf = fit_function.cdf(thresh, *params)
        if cdf == 0:
            rp = np.nan
        else:
            rp = 1 / cdf

    return rp

###############################################################################

def _rp_plot_data(
    data: np.ndarray,
    fit_function,
    color: str,
    label: str,
    ax,
    direction: str = 'descending',
    bootstrap_ci: int | None = 95,
    boot_size: int = 1000,
    params: tuple | None = None,
    max_return_period: float = 10000,
    thresh: float | None = None,
    ref_rp: float | None = None) -> dict:
    """
    Plots return period data along with its confidence intervals on a given axis.

    This function generates the return period data from the input data using a
    specified fit function. It also calculates and plots the confidence intervals
    for the return periods based on bootstrap sampling.

    Parameters
    ----------
    data : numpy.ndarray
        An array of shape (n_samples,) containing the input data for which return
        periods and confidence intervals will be calculated.

    fit_function : callable
        A function that fits the input data to a distribution and calculates the
        return period.

    color : str
        The color to use for plotting the return periods and confidence intervals.

    label : str
        The label to use for the plot legend.

    ax : matplotlib.axes.Axes
        The matplotlib axes object on which to plot the data.

    direction : str, optional
        The direction in which to sort the bootstrap samples. Can be either
        "ascending" or "descending" (default).

    bootstrap_ci : int, optional
        The confidence interval percentage to use for calculating the return time
        confidence intervals. Default is 95.

    boot_size : int, optional
        The number of bootstrap samples to generate. Default is 1000.

    max_return_period : float, optional
        The largest return period (in years) the fitted curve should be
        extrapolated out to. Default is 1000.

    thresh : float, optional
        A fixed event magnitude (e.g. the observed event) to find a
        return-period CI for, read directly off the plotted curve CI band
        (where it crosses `thresh`) so the two are consistent by
        construction. Default is None (skip this).

    ref_rp : float, optional
        A fixed return period (e.g. the other scenario's return period for
        `thresh`) to read a magnitude CI for, by evaluating this curve's
        already-plotted CI band at that return period. This is what backs
        the magnitude-change `axhspan` in `rp_plot`: NAT's magnitude CI at
        the return period ALL reaches `thresh`, so the vertical gap between
        the two CI bands visualises the intensity change. Default is None,
        which falls back to this curve's own `thresh_rp_point` (so calling
        with just `thresh` gives this curve's magnitude CI at its own
        thresh-crossing return period, with no separate `ref_rp` needed).

    Returns
    -------
    dict
        'thresh_rp_ci' : a 2-element array with the lower/upper confidence
        bounds of the return period of `thresh`, or (None, None) if
        `thresh`/`bootstrap_ci` was not given, or if `thresh` isn't within
        the range spanned by both edges of the curve CI band across
        `[1, max_return_period]` (i.e. no genuine crossing was found for
        one or both edges - only clipping at the grid boundary).

        'thresh_rp_point' : the point-estimate (non-bootstrapped) return
        period of `thresh`, or None if `thresh` was not given.

        'mag_ci' : a 2-element array with the lower/upper confidence bounds
        of this curve's magnitude at `ref_rp` (or `thresh_rp_point` if
        `ref_rp` wasn't given), or (None, None) if neither `ref_rp` nor
        `thresh` was given, or `bootstrap_ci` was not given.
    """
    if not params:
        params = fit_function.fit(data, loc=data.mean(), scale=data.std())

    return_period = np.array([
        _rp_calculation(data, fit_function, i, direction, params) for i in data]
    )

    ax.semilogx(
        return_period, data, marker='o', markersize=2,
        linestyle='None', mec=color, mfc=color,
        color=color, fillstyle='full',
        label=label, zorder=2
    )

    # extend the fitted curve out to max_return_period on both tails, on a
    # grid that is evenly spaced in log(return period) so it lines up with
    # the bootstrap CI band below at every point, including the extrapolated
    # tail
    tail_prob = 1 / max_return_period
    rp_grid = np.logspace(
        np.log10(1 / (1 - tail_prob)), np.log10(max_return_period), 700
    )
    if direction == 'descending':
        prob_grid = 1 - 1 / rp_grid
    else:
        prob_grid = 1 / rp_grid

    fitted_x = fit_function.ppf(prob_grid, *params)

    ax.semilogx(rp_grid, fitted_x, color=color, lw=2)

    thresh_rp_point = float(_rp_calculation(data, fit_function, thresh, direction, params)) \
        if thresh is not None else None

    thresh_rp_ci = (None, None)
    mag_ci = (None, None)

    # if bootstrap confidence interval is provided then we calculate
    # and plot the ci, extrapolated across the whole fitted curve
    if bootstrap_ci:
        fitted_lo, fitted_hi = _calc_fitted_curve_confidence(
            data, fit_function, prob_grid,
            bootstrap_ci=bootstrap_ci, boot_size=boot_size
        )

        ax.fill_between(
            rp_grid, fitted_lo, fitted_hi, color=color,
            alpha=0.2, linewidth=0., zorder=0
        )

        if thresh is not None:
            # read the return-period CI straight off the band that was just
            # drawn, i.e. where fitted_hi/fitted_lo cross `thresh` - this is
            # the quantity `axvspan` should show, and reading it directly
            # off the band guarantees the two are visually consistent.
            # fitted_lo/fitted_hi are monotonic in rp_grid (a percentile of
            # a family of monotonic curves is itself monotonic - increasing
            # for direction='descending', decreasing for 'ascending'), so
            # interpolation is well-defined once oriented increasing (as
            # np.interp requires).
            def _touches(curve):
                lo, hi = min(curve[0], curve[-1]), max(curve[0], curve[-1])
                return lo <= thresh <= hi

            def _rp_at(curve):
                if curve[0] > curve[-1]:
                    curve, grid = curve[::-1], rp_grid[::-1]
                else:
                    grid = rp_grid
                return np.interp(thresh, curve, grid)

            # only report a span if `thresh` genuinely falls within both the
            # lower and upper edges of the band across the plotted range -
            # otherwise the curve never actually reaches `thresh` within
            # max_return_period, and np.interp would silently clip to the
            # grid boundary instead of a real crossing, which is not a
            # meaningful CI to plot
            if _touches(fitted_hi) and _touches(fitted_lo):
                thresh_rp_ci = np.sort([_rp_at(fitted_hi), _rp_at(fitted_lo)])

        # default to this curve's own thresh-crossing return period if no
        # explicit ref_rp was given (e.g. NAT's magnitude is evaluated at
        # ALL's ref_rp, but ALL itself just uses its own)
        effective_ref_rp = ref_rp if ref_rp is not None else thresh_rp_point

        if effective_ref_rp is not None:
            # magnitude CI at a fixed return period: rp_grid is always
            # increasing by construction, so this interpolation (unlike the
            # thresh-crossing one above) never needs reorienting
            mag_ci = np.sort([
                np.interp(effective_ref_rp, rp_grid, fitted_lo),
                np.interp(effective_ref_rp, rp_grid, fitted_hi),
            ])

    return {
        'thresh_rp_ci': thresh_rp_ci,
        'thresh_rp_point': thresh_rp_point,
        'mag_ci': mag_ci,
    }

###############################################################################

# Helpers shared between climattr.attribution.analogues' public functions.

def _pairwise_distance(
    event_vec: np.ndarray,
    data_2d: np.ndarray,
    metric: str) -> np.ndarray:
    """
    Compute the distance from `event_vec` to every row of `data_2d`, using
    one of the named metrics ("euclidean", "manhattan", "chebyshev", "cosine", "correlation")


    Parameters
    ----------
    event_vec : np.ndarray
        1D array of shape (n_features,), the reference pattern.

    data_2d : np.ndarray
        2D array of shape (n_time, n_features), the candidate patterns.

    metric : str
        One of ("euclidean", "manhattan", "chebyshev", "cosine", "correlation")
.

    Returns
    -------
    np.ndarray
        1D array of shape (n_time,) with the distance of each row of
        `data_2d` to `event_vec`.
    """
    diff = data_2d - event_vec

    if metric == "euclidean":
        return np.sqrt(np.nansum(diff ** 2, axis=1))

    if metric == "manhattan":
        return np.nansum(np.abs(diff), axis=1)

    if metric == "chebyshev":
        return np.nanmax(np.abs(diff), axis=1)

    if metric == "cosine":
        a = np.nan_to_num(event_vec)
        b = np.nan_to_num(data_2d)
        num = b @ a
        denom = np.linalg.norm(b, axis=1) * np.linalg.norm(a) + 1e-12
        return 1.0 - num / denom

    if metric == "correlation":
        a = np.nan_to_num(event_vec)
        b = np.nan_to_num(data_2d)
        a_c = a - a.mean()
        b_c = b - b.mean(axis=1, keepdims=True)
        num = b_c @ a_c
        denom = np.linalg.norm(b_c, axis=1) * np.linalg.norm(a_c) + 1e-12
        return 1.0 - num / denom

    raise ValueError(f"Unknown distance metric {metric!r}; choose a valid distance metric or pass a callable")

###############################################################################

def _get_common_variables(
    event_pattern: xr.Dataset,
    all_events: xr.Dataset,
    variables: Union[Sequence[str], None] = None) -> List[str]:
    """
    Resolve which data variables to use from `event_pattern`/`all_events`.

    Parameters
    ----------
    event_pattern : xr.Dataset
        The reference pattern dataset.

    all_events : xr.Dataset
        The candidate dataset.

    variables : Sequence[str], optional
        Which data variables to use. Defaults to None (every variable
        present in both datasets).

    Returns
    -------
    List[str]
        The resolved list of shared variable names.
    """
    if variables is not None:
        missing = [v for v in variables if v not in event_pattern.data_vars or v not in all_events.data_vars]
        if missing:
            raise ValueError(f"variables {missing} not found in both datasets")
        return list(variables)

    common = [v for v in event_pattern.data_vars if v in all_events.data_vars]
    if not common:
        raise ValueError("event_pattern and all_events share no common data variables")

    return common

###############################################################################

def _dataset_to_matrix(ds: xr.Dataset, variables: Sequence[str]) -> np.ndarray:
    """
    Stack the given data variables of `ds` into a 2D array (time, features).

    Parameters
    ----------
    ds : xr.Dataset
        Dims (lat, lon), optionally also 'time'.

    variables : Sequence[str]
        Which data variables to stack.

    Returns
    -------
    np.ndarray
        A 2D array of shape (n_time, n_features), or (1, n_features) if
        `ds` has no 'time' dimension.
    """
    has_time = "time" in ds.dims

    arrays = []
    for var in variables:
        da = ds[var]
        if has_time:
            da = da.transpose("time", "lat", "lon")
            arrays.append(da.values.reshape(da.sizes["time"], -1))
        else:
            da = da.transpose("lat", "lon")
            arrays.append(da.values.reshape(1, -1))

    return np.concatenate(arrays, axis=1)

###############################################################################
