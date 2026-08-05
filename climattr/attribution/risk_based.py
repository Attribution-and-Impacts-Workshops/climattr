import numpy as np
import pandas as pd
import xarray as xr

from joblib import Parallel, delayed

from climattr.attribution.utils import (
    _calc_bootstrap_ensemble,
    _pr_calculation,
    _rp_calculation,
    _rp_plot_data
)
from climattr.utils import (
    get_percentiles_from_ci,
    get_fitted_percentiles
)
from climattr.validator import (
    validate_direction,
    validate_ci
)

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
    `climattr.attribution.wwa.attribution_metrics` - 'event_magnitude',
    'return_period', 'PR', 'dI_abs', 'dI_rel', 'n_obs', 'n_samp', 'n_failed'
    as rows, 'estimate'/'ci_inf'/'ci_sup' as columns.

    Unlike `wwa.attribution_metrics`, which refits a joint nonstationary
    regression per bootstrap replicate, this function bootstraps `all`/`nat`
    independently and refits a stationary distribution to each resampled
    array - so there is no single set of regression parameters
    (mu_0/sigma_0/c/alpha) to report here, only the derived quantities.
    `return_period` is the return period of `thresh` under the ALL climate;
    `dI_abs`/`dI_rel` compare `thresh` against the NAT climate's magnitude
    at that *same* return period (a matched-probability comparison),
    analogous to `wwa`'s intensity change at a fixed return period.

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
