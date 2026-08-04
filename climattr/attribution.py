import numpy as np
import pandas as pd
import xarray as xr

from datetime import datetime
import statsmodels.api as sm
from joblib import Parallel, delayed
from scipy import stats

from climattr.minimization.fit import (
    fit_data,
    extrapolate_data
)
from climattr.utils import (
    get_percentiles_from_ci,
    get_fitted_percentiles
)
from climattr.validator import (
    validate_direction, 
    validate_ci
)

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
    # Constant to avoid division by a very small number when probabilities are too small
    # epsilon1 = 0.01
    # epsilon2 = 1e-3
    # epsilon3 = 1e-10

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

    # # If both probabilities are too small and the NAT probability is even lower
    # # we will end up with a very high PR value, which is not meaningful
    # if (probability_all < epsilon1) and (probability_nat < epsilon2):
    #     pr = np.nan
    # else:
    #     # just to avoid division by zero
    #     if probability_nat < epsilon3:
    #         probability_nat = epsilon3

    pr = probability_all / probability_nat

    return pr

###############################################################################

def _far_calculation(
    all_array: np.ndarray, 
    nat_array: np.ndarray, 
    fit_function, 
    thresh: float,
    direction: str = 'descending',
    params_all: tuple | None = None,
    params_nat: tuple | None = None) -> float:
    """
    Calculates the Fraction of Attributable Risk (FAR) between two datasets.

    This function computes the FAR, which is a measure of the fraction of risk 
    attributable to a specific factor, by comparing the probability ratio (PR) 
    between the "all" and "natural" scenario datasets.

    Parameters
    ----------
    all_array : numpy.ndarray
        An array of shape (n_samples,) containing the "all" scenario data.
        
    nat_array : numpy.ndarray
        An array of shape (n_samples,) containing the "natural" scenario data.
        
    fit_function : callable
        A function that fits the input data to a distribution.
        
    thresh : float
        The threshold value for which the FAR will be calculated.

    direction : str, optional
        The direction in which to calculate the return period. Default is "descending".

    Returns
    -------
    float
        The calculated Fraction of Attributable Risk (FAR).
    """
    epsilon = 1e-10  # Small constant to avoid division by a very small number    
    return 1 - (1 / (_pr_calculation(
        all_array, nat_array, fit_function, thresh, direction, params_all, params_nat
    ) + epsilon))

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

def attribution_metrics(
    all: xr.DataArray,
    nat: xr.DataArray,
    fit_function,
    thresh: float,
    direction: str = 'descending',
    bootstrap_ci: int = 95,
    boot_size: int = 1000,
    summary_statistics: bool = True,
    seed: int = 42,
    n_jobs: int = -1) -> pd.DataFrame:
    """
    Calculate attribution metrics (event magnitude, return period,
    probability ratio and intensity change) for "ALL" (observed) and "NAT"
    (natural) climate scenarios, in the same row/column layout as
    `climattr.attribution_wwa.attribution_metrics` - 'event_magnitude',
    'return_period', 'PR', 'dI_abs', 'dI_rel', 'n_obs', 'n_samp', 'n_failed'
    as rows, 'estimate'/'ci_inf'/'ci_sup' as columns.

    Unlike `attribution_wwa.attribution_metrics`, which refits a joint
    nonstationary regression per bootstrap replicate, this function
    bootstraps `all`/`nat` independently and refits a stationary
    distribution to each resampled array - so there is no single set of
    regression parameters (mu_0/sigma_0/c/alpha) to report here, only the
    derived quantities. `return_period` is the return period of `thresh`
    under the ALL climate; `dI_abs`/`dI_rel` compare `thresh` against the
    NAT climate's magnitude at that *same* return period (a matched-
    probability comparison), analogous to `attribution_wwa`'s intensity
    change at a fixed return period.

    Parameters
    ----------
    all : xr.DataArray
        Data array representing the "ALL" scenario, which includes
        human influences on climate.

    nat : xr.DataArray
        Data array representing the "NAT" scenario, which represents
        the natural climate without human influences.

    fit_function : callable
        A statistical distribution or fitting function used to model the data.

    thresh : float
        The threshold value for which the attribution metrics are calculated.

    direction : str, optional, default = 'descending'
        The direction in which to assess exceedance of the threshold.
        Can be 'descending' or 'ascending'.

    bootstrap_ci : int, optional, default = 95
        The confidence interval (CI) percentage for bootstrapping.

    boot_size : int, optional, default = 1000
        The number of bootstrap samples to generate.

    summary_statistics : bool, optional, default = True
        If True, return the 'estimate'/'ci_inf'/'ci_sup' summary table. If
        False, return one row per successful bootstrap replicate instead.

    seed : int, optional, default = 42
        Random seed for the ALL bootstrap resample. The NAT resample uses
        `seed + 1`, so the two are not position-matched draws of the same
        underlying random sequence (`all` and `nat` are typically an exact
        constant shift of one another - see Notes below - so reusing one
        seed for both would silently erase all bootstrap spread from any
        metric that compares them, such as dI_abs/dI_rel).

    Returns
    -------
    pd.DataFrame
        The attribution metrics table (see above), or the raw per-replicate
        bootstrap sample if `summary_statistics` is False.
    """
    validate_direction(direction)

    all_array = all.to_numpy().flatten()
    nat_array = nat.to_numpy().flatten()

    def compute_metrics(all_sample, nat_sample):
        """Fit ALL/NAT independently and derive event_magnitude/return_period/PR/dI."""
        params_all = fit_function.fit(all_sample, loc=all_sample.mean(), scale=all_sample.std())
        params_nat = fit_function.fit(nat_sample, loc=nat_sample.mean(), scale=nat_sample.std())

        rp = _rp_calculation(all_sample, fit_function, thresh, direction, params_all)
        pr = _pr_calculation(all_sample, nat_sample, fit_function, thresh, direction, params_all, params_nat)

        # NAT's effective magnitude at the same return period as `thresh`
        # under ALL, so dI reflects a matched-probability comparison
        u = 1 / rp
        rl_nat = fit_function.ppf(1 - u if direction == 'descending' else u, *params_nat)

        return {
            'event_magnitude': thresh,
            'return_period': rp,
            'PR': pr,
            'dI_abs': thresh - rl_nat,
            'dI_rel': (thresh - rl_nat) / rl_nat * 100,
        }

    metric_names = ['event_magnitude', 'return_period', 'PR', 'dI_abs', 'dI_rel']
    point_est = compute_metrics(all_array, nat_array)

    # different seeds: _calc_bootstrap_ensemble reseeds numpy's global RNG
    # internally, so reusing one seed for both would draw ALL and NAT at
    # the same positions every replicate - see the `seed` docstring above
    all_boot = _calc_bootstrap_ensemble(all_array, boot_size=boot_size, seed=seed)
    nat_boot = _calc_bootstrap_ensemble(nat_array, boot_size=boot_size, seed=seed + 1)

    def compute_boot(boot):
        try:
            return compute_metrics(all_boot[boot], nat_boot[boot])
        except Exception:
            return None

    # Run computations in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_boot)(boot) for boot in range(int(boot_size))
    )
    results = [r for r in results if r is not None]
    n_failed = int(boot_size) - len(results)

    if summary_statistics:
        ci_inf, ci_sup = get_percentiles_from_ci(bootstrap_ci)

        row_names = metric_names + ['n_obs', 'n_samp', 'n_failed']
        metrics_result = pd.DataFrame(
            np.full((len(row_names), 3), np.nan),
            index=row_names,
            columns=['estimate', 'ci_inf', 'ci_sup']
        )

        for metric_name in metric_names:
            values = np.array([r[metric_name] for r in results])
            values = values[~np.isnan(values)]

            metrics_result.loc[metric_name, 'estimate'] = point_est[metric_name]
            metrics_result.loc[metric_name, 'ci_inf'] = np.percentile(values, ci_inf)
            metrics_result.loc[metric_name, 'ci_sup'] = np.percentile(values, ci_sup)

        metrics_result.loc['n_obs', 'estimate'] = len(all_array)
        metrics_result.loc['n_samp', 'estimate'] = len(results)
        metrics_result.loc['n_failed', 'estimate'] = n_failed
    else:
        metrics_result = pd.DataFrame(results)

    return metrics_result
        
###############################################################################

def histogram_plot(
    ax,
    all: xr.DataArray,
    nat: xr.DataArray,
    fit_function,
    thresh: float,
    **kwargs) -> None:
    """
    Plot histograms of the "ALL" and "NAT" scenarios along with their 
    fitted probability density functions (PDFs).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object on which to draw the histogram.
    
    all : xr.DataArray
        Data array representing the "ALL" scenario, which includes 
        human influences on climate.
    
    nat : xr.DataArray
        Data array representing the "NAT" scenario, which represents 
        the natural climate without human influences.
    
    fit_function : callable
        A statistical distribution or fitting function used to model the data.
    
    thresh : float
        The threshold value, which is plotted as a vertical dashed line.
    
    Returns
    -------
    None
        This function does not return anything; it modifies the provided 
        axes object in-place.
    """
    all_array = all.to_numpy().flatten()
    nat_array = nat.to_numpy().flatten()

    params_all = fit_function.fit(all_array, loc=all_array.mean(), scale=all_array.std())
    params_nat = fit_function.fit(nat_array, loc=nat_array.mean(), scale=nat_array.std())

    # getting the kwargs
    all_color = kwargs.get('all_color', 'C1')
    nat_color = kwargs.get('nat_color', 'C0')
    alpha = kwargs.get('alpha', 0.5)

    ax.hist(all_array, color=all_color, alpha=alpha, density=True, label='ALL')
    ax.hist(nat_array, color=nat_color, alpha=alpha, density=True, label='NAT')

    # fit the requested distribution and plot it as a line
    percentiles = np.linspace(0.01, 99.9, 700)
    x_all = get_fitted_percentiles(percentiles, params_all, fit_function)
    x_nat = get_fitted_percentiles(percentiles, params_nat, fit_function)

    ax.plot(x_all, fit_function.pdf(x_all, *params_all), color=all_color, lw=2)
    ax.plot(x_nat, fit_function.pdf(x_nat, *params_nat), color=nat_color, lw=2)

    ax.axvline(thresh, color='k', ls='--')
    ax.legend()

###############################################################################

def rp_plot(
    ax,
    all: xr.DataArray,
    nat: xr.DataArray,
    fit_function,
    thresh: float,
    direction: str = 'descending',
    bootstrap_ci: int = 95,
    boot_size: int = 1000,
    max_return_period: float = 10000,
    span_size: float = 0.04,
    **kwargs) -> None:
    """
    Plot return periods for the "ALL" and "NAT" scenarios, including
    confidence intervals (CI) for the bootstrapped return periods.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object on which to draw the return period plot.
    
    all : xr.DataArray
        Data array representing the "ALL" scenario, which includes 
        human influences on climate.
    
    nat : xr.DataArray
        Data array representing the "NAT" scenario, which represents 
        the natural climate without human influences.
    
    fit_function : callable
        A statistical distribution or fitting function used to model the data.
    
    thresh : float
        The threshold value, which is plotted as a horizontal dashed line.
    
    direction : str, optional, default = 'descending'
        The direction in which to assess exceedance of the threshold. 
        Can be 'descending' or 'ascending'.
    
    bootstrap_ci : int, optional, default = 95
        The confidence interval (CI) percentage for bootstrapping.
    
    boot_size : int, optional, default = 1000
        The number of bootstrap samples to generate.

    max_return_period : float, optional, default = 1000
        The largest return period (in years) the fitted curves for ALL and
        NAT should be extrapolated out to.

    span_size : float, optional, default = 0.04
        Fraction of the axes the return-period `axvspan`s and magnitude
        `axhspan`s are drawn as, hugging the bottom/left edge respectively,
        instead of reaching all the way to the `thresh`/reference-return-
        period lines.

    Returns
    -------
    None
        This function does not return anything; it modifies the provided
        axes object in-place.
    """
    # validation steps
    validate_direction(direction)
    validate_ci(bootstrap_ci)

    all_array = np.sort(all.to_numpy().flatten())
    nat_array = np.sort(nat.to_numpy().flatten())

    if direction == 'descending':
        all_array = all_array[::-1]
        nat_array = nat_array[::-1]

    # getting the kwargs
    all_color = kwargs.get('all_color', 'C1')
    nat_color = kwargs.get('nat_color', 'C0')

    all_result = _rp_plot_data(
        all_array, fit_function, all_color, 'ALL', ax, direction, bootstrap_ci, boot_size,
        max_return_period=max_return_period, thresh=thresh
    )
    # NAT's magnitude is evaluated at the return period ALL reaches `thresh`
    # (its point estimate, not the CI, so there's a single well-defined x
    # position), so the axhspan below shows NAT's magnitude CI at a
    # same-probability event - the classic "intensity change" comparison
    nat_result = _rp_plot_data(
        nat_array, fit_function, nat_color, 'NAT', ax, direction, bootstrap_ci, boot_size,
        max_return_period=max_return_period, thresh=thresh, ref_rp=all_result['thresh_rp_point']
    )

    ax.axhline(thresh, color='k', ls='--')

    rp_point_all = all_result['thresh_rp_point']
    ax.axvline(rp_point_all, color='k', ls='--')

    # shade the CI of the return period of `thresh` itself, from refits of
    # the ALL/NAT data (not a rank-index lookup into an order-statistic
    # bootstrap of the per-point return_period array). Drawn regardless of
    # whether the *observed* ALL/NAT data reaches `thresh` - the fitted,
    # extrapolated curve is what matters here (e.g. NAT's observed range
    # will essentially never include the factual extreme event; that's the
    # point of the comparison), and `_rp_plot_data` already clips the span
    # to [1, max_return_period] if the curve never crosses `thresh` at all.
    conf_rp_inf_all, conf_rp_sup_all = all_result['thresh_rp_ci']
    conf_rp_inf_nat, conf_rp_sup_nat = nat_result['thresh_rp_ci']

    if conf_rp_inf_all is not None:
        ax.axvspan(
            conf_rp_inf_all, conf_rp_sup_all,
            ymin=0, ymax=span_size,
            facecolor='silver', edgecolor=all_color,
            linewidth=2., alpha=0.3, zorder=0
        )

    if conf_rp_inf_nat is not None:
        ax.axvspan(
            conf_rp_inf_nat, conf_rp_sup_nat,
            ymin=0, ymax=span_size,
            facecolor='silver', edgecolor=nat_color,
            linewidth=2., alpha=0.3, zorder=0
        )

    # shade the CI of the magnitude change: both ALL's and NAT's magnitude
    # at the return period ALL reaches `thresh` - ALL's own band brackets
    # `thresh` itself (since that's how rp_point_all was defined), and the
    # vertical gap between the two bands is the intensity change with its
    # uncertainty
    mag_lo_all, mag_hi_all = all_result['mag_ci']

    if mag_lo_all is not None:
        ax.axhspan(
            mag_lo_all, mag_hi_all,
            xmin=0, xmax=span_size,
            facecolor='silver', edgecolor=all_color,
            linewidth=2., alpha=0.3, zorder=0
        )

    mag_lo_nat, mag_hi_nat = nat_result['mag_ci']

    if mag_lo_nat is not None:
        ax.axhspan(
            mag_lo_nat, mag_hi_nat,
            xmin=0, xmax=span_size,
            facecolor='silver', edgecolor=nat_color,
            linewidth=2., alpha=0.3, zorder=0
        )

    ax.legend()

############################################################################### 