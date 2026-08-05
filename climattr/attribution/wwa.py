import numpy as np
import pandas as pd
import xarray as xr

from datetime import datetime
from joblib import Parallel, delayed
from scipy import stats
from typing import Union

from climattr.attribution.utils import (
    _pr_calculation,
    _rp_calculation,
    _calc_bootstrap_ensemble,
    _rp_plot_data
)
from climattr.minimization import likelihood
from climattr.minimization.fit import (
    fit_data,
    extrapolate_data,
    dist_params_at_covariate,
    aic,
    bic,
    rsquared
)
from climattr.utils import (
    find_nearest,
    get_percentiles_from_ci,
    get_fitted_percentiles
)
from climattr.validator import (
    validate_ci
)


def fit_summary(
    all: xr.DataArray,
    covariates_df: pd.DataFrame,
    fit_function_name: str,
    strategy: str = 'linear',
    bootstrap_ci: int = 95,
    boot_size: int = 1000,
    seed: int = 42,
    n_jobs: int = -1,
    verbose: bool = False) -> pd.DataFrame:
    """
    Fit a WWA model and return a summary table with parameters,
    goodness-of-fit metrics (AIC, BIC, R²), and bootstrap confidence intervals.

    Parameters
    ----------
    all : xr.DataArray
        Data array with the climate variable to fit.
    covariates_df : pd.DataFrame
        DataFrame with one or more covariate columns indexed by time.
    fit_function_name : str
        Name of the distribution to fit ('genextreme', 'norm', 'gamma', 'genpareto').
    strategy : str, optional
        Covariate strategy ('linear' by default).
    bootstrap_ci : int, optional
        Confidence interval percentage (default 95).
    boot_size : int, optional
        Number of bootstrap iterations (default 1000).
    seed : int, optional
        Random seed for reproducibility (default 42).
    n_jobs : int, optional
        Number of parallel jobs (-1 for all cores, default -1).
    verbose : bool, optional
        If True, print fit details (default False).

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns 'estimate', 'ci_inf', 'ci_sup' for each
        parameter (mu_0, sigma_0, c, alpha_0, ..., alpha_n) and goodness-of-fit
        metric (log_likelihood, aic, bic, r2, n_obs).
    """
    import warnings

    all_dataframe = all.to_dataframe().reset_index()
    dataframe = all_dataframe.set_index('time').join(covariates_df).dropna()

    cov_columns = covariates_df.columns.tolist()

    # Point estimate on full dataset
    model = likelihood.FIT_FUNCTIONS[fit_function_name](
        dataframe[all.name], dataframe[cov_columns].values, strategy=strategy
    )
    result = model.fit(disp=verbose)

    n_covariates = len(cov_columns)
    alpha_names = [f'alpha_{col}' for col in cov_columns]
    param_names = ['mu_0', 'sigma_0', 'c'] + alpha_names
    k = len(result.params)
    n = int(result.nobs)

    estimates = {
        'mu_0': result.params[0],
        'sigma_0': result.params[1],
        'c': result.params[2],
    }
    for i, name in enumerate(alpha_names):
        estimates[name] = result.params[3 + i]
    estimates.update({
        'log_likelihood': result.llf,
        'aic': aic(- result.llf, k),
        'bic': bic(- result.llf, k, n),
        'r2': rsquared(result),
        'n_obs': n,
    })

    # Bootstrap
    indices = np.arange(len(dataframe))
    indices_boot = _calc_bootstrap_ensemble(indices, boot_size=boot_size, seed=seed)

    def compute_fit(boot):
        dataframe_boot = dataframe.iloc[indices_boot[boot]].sort_index().dropna()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_boot = likelihood.FIT_FUNCTIONS[fit_function_name](
                dataframe_boot[all.name], dataframe_boot[cov_columns].values, strategy=strategy
            )
            try:
                result_boot = model_boot.fit(disp=False)
            except Exception:
                return None

        k_boot = len(result_boot.params)
        n_boot = int(result_boot.nobs)

        boot_result = {
            'mu_0': result_boot.params[0],
            'sigma_0': result_boot.params[1],
            'c': result_boot.params[2],
        }
        for i, name in enumerate(alpha_names):
            boot_result[name] = result_boot.params[3 + i]
        boot_result.update({
            'log_likelihood': result_boot.llf,
            'aic': aic(- result_boot.llf, k_boot),
            'bic': bic(- result_boot.llf, k_boot, n_boot),
            'r2': rsquared(result_boot),
        })
        return boot_result

    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_fit)(boot) for boot in range(int(boot_size))
    )

    # Filter out failed fits
    results = [r for r in results if r is not None]

    ci_inf, ci_sup = get_percentiles_from_ci(bootstrap_ci)

    all_names = param_names + ['log_likelihood', 'aic', 'bic', 'r2', 'n_obs']
    summary = pd.DataFrame(
        np.zeros((len(all_names), 3)),
        index=all_names,
        columns=['estimate', 'ci_inf', 'ci_sup']
    )

    for name in all_names:
        summary.loc[name, 'estimate'] = estimates[name]

    for name in param_names + ['log_likelihood', 'aic', 'bic', 'r2']:
        values = np.array([r[name] for r in results])
        values = values[~np.isnan(values)]
        summary.loc[name, 'ci_inf'] = np.percentile(values, ci_inf)
        summary.loc[name, 'ci_sup'] = np.percentile(values, ci_sup)

    # n_obs has no CI
    summary.loc['n_obs', 'ci_inf'] = np.nan
    summary.loc['n_obs', 'ci_sup'] = np.nan

    return summary

###############################################################################

def histogram_plot(
    ax,
    all: xr.DataArray,
    covariates_df: pd.DataFrame,
    fit_function_name: str,
    all_date: Union[datetime, str] = '2015-11-30',
    nat_date: Union[datetime, str] = '1900-11-30',
    strategy: str = 'linear',
    verbose: bool = False,
    **kwargs) -> None:
    """
    Plot histograms and fitted PDFs for the factual (ALL) and
    counterfactual (NAT) climate scenarios.

    Fits a statistical distribution to the climate variable conditioned on
    one or more covariates, extrapolates to the ALL and NAT dates, and
    plots the resulting histograms and fitted PDFs. A vertical dashed line
    marks the observed event threshold.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object on which to draw the plot.
    all : xr.DataArray
        Data array with the climate variable to analyse.
    covariates_df : pd.DataFrame
        DataFrame with one or more covariate columns indexed by time.
    fit_function_name : str
        Name of the scipy.stats distribution to fit
        (e.g. 'genextreme', 'norm', 'gamma', 'genpareto').
    all_date : Union[datetime, str], optional
        Date representing the factual (ALL) climate (default '2015-11-30').
        Also used to determine the event threshold.
    nat_date : Union[datetime, str], optional
        Date representing the counterfactual (NAT) climate (default '1900-11-30').
    strategy : str, optional
        Covariate strategy for the fit (default 'linear').
    verbose : bool, optional
        If True, print fit details (default False).
    **kwargs
        Additional keyword arguments:
        - all_color : str, color for the ALL scenario (default 'C1').
        - nat_color : str, color for the NAT scenario (default 'C0').
        - alpha : float, histogram transparency (default 0.5).

    Returns
    -------
    None
        Modifies the provided axes object in-place.
    """
    all_dataframe = all.to_dataframe().reset_index()

    dataframe = all_dataframe.set_index('time').join(covariates_df).dropna()

    cov_columns = covariates_df.columns.tolist()

    # fit function using MLE
    params = fit_data(
        dataframe[all.name], dataframe[cov_columns], fit_function_name, strategy, verbose=verbose
    )

    # extrapolate data to the selected dates
    params_all, all_wwa = extrapolate_data(covariates_df, params, all_date, fit_function_name, strategy)
    params_nat, nat_wwa = extrapolate_data(covariates_df, params, nat_date, fit_function_name, strategy)

    fit_function = getattr(stats, fit_function_name)

    # get the threshold value for the selected date
    thresh = dataframe.loc[all_date, all.name]

    # getting the kwargs
    all_color = kwargs.get('all_color', 'C1')
    nat_color = kwargs.get('nat_color', 'C0')
    alpha = kwargs.get('alpha', 0.5)

    ax.hist(all_wwa, color=all_color, alpha=alpha, density=True, label='ALL')
    ax.hist(nat_wwa, color=nat_color, alpha=alpha, density=True, label='NAT')

    # fit the requested distribution and plot it as a line
    percentiles = np.linspace(0.01, 99.9, 700)
    x_all = get_fitted_percentiles(percentiles, params_all, fit_function)
    x_nat = get_fitted_percentiles(percentiles, params_nat, fit_function)

    ax.plot(x_all, fit_function.pdf(x_all, *params_all), color=all_color, lw=2)
    ax.plot(x_nat, fit_function.pdf(x_nat, *params_nat), color=nat_color, lw=2)

    ax.axvline(thresh, color='k', ls='--')
    ax.legend()

###############################################################################

def eff_return_level(
    params: np.ndarray,
    fit_function_name: str,
    strategy: str,
    covariate: Union[float, np.ndarray],
    rp: float,
    direction: str = 'descending') -> float:
    """
    Magnitude of the event with return period `rp` under the fitted
    nonstationary model evaluated at `covariate`. Equivalent to rwwa's
    `eff_return_level`.

    Parameters
    ----------
    params : np.ndarray
        The fitted parameters, as returned by `fit_data`.
    fit_function_name : str
        The distribution fitted ('norm', 'genextreme', 'gamma', 'genpareto').
    strategy : str
        The parameterization strategy used to fit `params`.
    covariate : Union[float, np.ndarray]
        The covariate value(s) at which to evaluate the distribution.
    rp : float
        The return period of interest.
    direction : str, optional
        The direction in which to assess exceedance. Default is 'descending'.

    Returns
    -------
    float
        The effective return level.
    """
    dist_params = dist_params_at_covariate(params, covariate, fit_function_name, strategy)
    fit_function = getattr(stats, fit_function_name)

    u = 1 / rp
    if direction == 'descending':
        return fit_function.ppf(1 - u, *dist_params)

    return fit_function.ppf(u, *dist_params)

###############################################################################

def int_change(
    params: np.ndarray,
    fit_function_name: str,
    strategy: str,
    cov_f: Union[float, np.ndarray],
    cov_cf: Union[float, np.ndarray],
    rp: float,
    direction: str = 'descending',
    relative: bool = False) -> float:
    """
    Change in the magnitude of the event with return period `rp` between the
    counterfactual and factual covariate values. Equivalent to rwwa's
    `int_change`.

    Parameters
    ----------
    params : np.ndarray
        The fitted parameters, as returned by `fit_data`.
    fit_function_name : str
        The distribution fitted ('norm', 'genextreme', 'gamma', 'genpareto').
    strategy : str
        The parameterization strategy used to fit `params`.
    cov_f : Union[float, np.ndarray]
        The covariate value(s) defining the factual climate.
    cov_cf : Union[float, np.ndarray]
        The covariate value(s) defining the counterfactual climate.
    rp : float
        The return period of interest.
    direction : str, optional
        The direction in which to assess exceedance. Default is 'descending'.
    relative : bool, optional
        If True, return the change as a percentage of the counterfactual
        return level. Default is False (absolute change).

    Returns
    -------
    float
        The intensity change (dI), absolute or relative.
    """
    rl_f = eff_return_level(params, fit_function_name, strategy, cov_f, rp, direction)
    rl_cf = eff_return_level(params, fit_function_name, strategy, cov_cf, rp, direction)

    if relative:
        return (rl_f - rl_cf) / rl_cf * 100

    return rl_f - rl_cf

###############################################################################

def attribution_metrics(
    all: xr.DataArray,
    covariates_df: pd.DataFrame,
    fit_function_name: str,
    cov_f: Union[float, np.ndarray],
    cov_cf: Union[float, np.ndarray],
    strategy: str = 'linear',
    ev: Union[float, None] = None,
    rp: Union[float, None] = None,
    direction: str = 'descending',
    bootstrap_ci: int = 95,
    boot_size: int = 500,
    seed: int = 42,
    n_jobs: int = -1,
    return_sample: bool = False) -> pd.DataFrame:
    """
    Bootstrapped confidence intervals for the probability ratio and
    intensity change of a nonstationary model, following the WWA 'rwwa'
    `boot_ci` methodology. Rows of (variable, covariates) are resampled with
    replacement and the whole regression model is refit on each resample via
    `fit_data`, so uncertainty in the fitted covariate relationship itself is
    propagated through to the estimates - the same row-resampling-and-refit
    pattern already used by `fit_summary` above (reuses
    `_calc_bootstrap_ensemble` for the resampling and `Parallel` for the
    replicate loop).

    `cov_f`/`cov_cf` are raw covariate values rather than dates, matching
    rwwa's convention of defining a counterfactual as an arbitrary offset
    (e.g. `cov_f - 1.2`) rather than requiring an actual historical date to
    be present in the record. Pass `covariates_df.loc[date]` for either if
    you want a date-based factual/counterfactual instead.

    Exactly one of `ev`/`rp` is held fixed across the point estimate and
    every bootstrap replicate; the other is re-derived from each replicate's
    own fitted parameters (e.g. if `ev` is fixed, the return period of that
    fixed magnitude will vary from replicate to replicate).

    Parameters
    ----------
    all : xr.DataArray
        Data array with the climate variable to analyse.
    covariates_df : pd.DataFrame
        DataFrame with one or more covariate columns indexed by time.
    fit_function_name : str
        The distribution to fit ('norm', 'genextreme', 'gamma', 'genpareto').
    cov_f : Union[float, np.ndarray]
        The covariate value(s) defining the factual climate.
    cov_cf : Union[float, np.ndarray]
        The covariate value(s) defining the counterfactual climate.
    strategy : str, optional
        Covariate strategy for the fit (default 'linear').
    ev : float, optional
        Magnitude of the event of interest. Defaults to the last observation
        of `all`, mirroring rwwa's default.
    rp : float, optional
        Fixed return period of interest. If given, `ev` is derived from it
        instead (the effective return level under `cov_f`) for the point
        estimate and every bootstrap replicate.
    direction : str, optional
        The direction in which to assess exceedance. Default is 'descending'.
    bootstrap_ci : int, optional
        Confidence interval percentage (default 95).
    boot_size : int, optional
        Number of bootstrap replicates (default 500).
    seed : int, optional
        Random seed for reproducibility (default 42).
    n_jobs : int, optional
        Number of parallel jobs (-1 for all cores, default -1).
    return_sample : bool, optional
        If True, return the full bootstrap sample instead of a summary.
        Default is False.

    Returns
    -------
    pd.DataFrame
        A table indexed by model parameter/metric name, with columns
        'estimate', 'ci_inf' and 'ci_sup' (matching `fit_summary`'s
        convention), plus trailing 'n_obs', 'n_samp' and 'n_failed' rows.
        If `return_sample` is True, a DataFrame with one row per successful
        bootstrap replicate instead.
    """
    all_dataframe = all.to_dataframe().reset_index()
    dataframe = all_dataframe.set_index('time').join(covariates_df).dropna()

    cov_columns = covariates_df.columns.tolist()
    fit_function = getattr(stats, fit_function_name)

    param_names = ['mu_0', 'sigma_0', 'c'] + [f'alpha_{cnm}' for cnm in cov_columns]
    metric_names = ['event_magnitude', 'return_period', 'PR', 'dI_abs', 'dI_rel']
    row_names = param_names + metric_names

    def _estimate(x, cov, fixed_ev, fixed_rp):
        params = fit_data(x, cov, fit_function_name, strategy, verbose=False)

        if fixed_rp is None:
            ev_i = x.iloc[-1] if fixed_ev is None else fixed_ev
            dist_params_f = dist_params_at_covariate(params, cov_f, fit_function_name, strategy)
            rp_i = float(_rp_calculation(None, fit_function, ev_i, direction, dist_params_f))
        else:
            rp_i = fixed_rp
            ev_i = eff_return_level(params, fit_function_name, strategy, cov_f, rp_i, direction)

        dist_params_f = dist_params_at_covariate(params, cov_f, fit_function_name, strategy)
        dist_params_cf = dist_params_at_covariate(params, cov_cf, fit_function_name, strategy)

        row = {
            'event_magnitude': ev_i,
            'return_period': rp_i,
            'PR': float(_pr_calculation(None, None, fit_function, ev_i, direction, dist_params_f, dist_params_cf)),
            'dI_abs': int_change(params, fit_function_name, strategy, cov_f, cov_cf, rp_i, direction, relative=False),
            'dI_rel': int_change(params, fit_function_name, strategy, cov_f, cov_cf, rp_i, direction, relative=True),
        }
        return np.concatenate([params, [row[m] for m in metric_names]])

    point_row = _estimate(dataframe[all.name], dataframe[cov_columns], ev, rp)
    point_est = dict(zip(row_names, point_row))

    # fix whichever of ev/rp was resolved for the point estimate, so it can
    # be held fixed across every bootstrap replicate too
    fixed_ev, fixed_rp = (None, point_est['return_period']) if rp is not None \
        else (point_est['event_magnitude'], None)

    indices = np.arange(len(dataframe))
    indices_boot = _calc_bootstrap_ensemble(indices, boot_size=boot_size, seed=seed)

    def compute_boot(boot):
        dataframe_boot = dataframe.iloc[indices_boot[boot]].sort_index().dropna()
        try:
            return _estimate(
                dataframe_boot[all.name], dataframe_boot[cov_columns], fixed_ev, fixed_rp
            )
        except Exception:
            return None

    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_boot)(boot) for boot in range(int(boot_size))
    )
    results = [r for r in results if r is not None]
    n_failed = boot_size - len(results)

    boot_matrix = np.array(results)

    if return_sample:
        return pd.DataFrame(boot_matrix, columns=row_names)

    ci_inf, ci_sup = get_percentiles_from_ci(bootstrap_ci)

    summary = pd.DataFrame({
        'estimate': point_row,
        'ci_inf': np.percentile(boot_matrix, ci_inf, axis=0),
        'ci_sup': np.percentile(boot_matrix, ci_sup, axis=0),
    }, index=row_names)

    summary.loc['n_obs'] = [len(dataframe), np.nan, np.nan]
    summary.loc['n_samp'] = [len(results), np.nan, np.nan]
    summary.loc['n_failed'] = [n_failed, np.nan, np.nan]

    return summary

###############################################################################

# def rp_plot(
#     ax,
#     all: xr.DataArray,
#     global_tas: pd.DataFrame,
#     fit_function_name: str,
#     all_date: Union[datetime, str] = '2015-11-30',
#     nat_date: Union[datetime, str] = '1900-11-30',
#     strategy: str = 'linear',
#     verbose: bool = False,
#     direction: str = 'descending',
#     bootstrap_ci: int = 95,
#     boot_size: int = 1000,
#     **kwargs) -> None:
#     """
#     Plot return periods for the "ALL" and "NAT" scenarios, including
#     confidence intervals (CI) for the bootstrapped return periods.

#     Parameters
#     ----------
#     ax : matplotlib.axes.Axes
#         The axes object on which to draw the return period plot.

#     all : xr.DataArray
#         Data array representing the "ALL" scenario, which includes
#         human influences on climate.

#     nat : xr.DataArray
#         Data array representing the "NAT" scenario, which represents
#         the natural climate without human influences.

#     fit_function : callable
#         A statistical distribution or fitting function used to model the data.

#     thresh : float
#         The threshold value, which is plotted as a horizontal dashed line.

#     direction : str, optional, default = 'descending'
#         The direction in which to assess exceedance of the threshold.
#         Can be 'descending' or 'ascending'.

#     bootstrap_ci : int, optional, default = 95
#         The confidence interval (CI) percentage for bootstrapping.

#     boot_size : int, optional, default = 1000
#         The number of bootstrap samples to generate.

#     Returns
#     -------
#     None
#         This function does not return anything; it modifies the provided
#         axes object in-place.
#     """
#     # validation steps
#     validate_direction(direction)
#     validate_ci(bootstrap_ci)

#     all_dataframe = all.to_dataframe().reset_index()

#     # global_tas['tas'] = global_tas['tas'].rolling(4, center=True).mean()
#     dataframe = all_dataframe.set_index('time').join(global_tas).dropna()

#     # fit function using MLE
#     params = fit_data(
#         dataframe[all.name], dataframe['tas'], fit_function_name, strategy, verbose=verbose
#     )

#     # extrapolate data to the selected dates
#     params_all, all_wwa = extrapolate_data(global_tas, params, all_date, fit_function_name, strategy)
#     params_nat, nat_wwa = extrapolate_data(global_tas, params, nat_date, fit_function_name, strategy)

#     fit_function = getattr(stats, fit_function_name)

#     # get the threshold value for the selected date
#     thresh = dataframe.loc[all_date, all.name]

#     if direction == 'descending':
#         all_wwa = all_wwa[::-1]
#         nat_wwa = nat_wwa[::-1]

#         all_span_checker = all_wwa.max() >= thresh
#         nat_span_checker = nat_wwa.max() >= thresh
#     else:
#         all_span_checker = all_wwa.min() <= thresh
#         nat_span_checker = nat_wwa.min() <= thresh

#     # getting the kwargs
#     all_color = kwargs.get('all_color', 'C1')
#     nat_color = kwargs.get('nat_color', 'C0')

#     conf_rp_inf_all, conf_rp_sup_all = _rp_plot_data(
#         all_wwa, fit_function, all_color, 'ALL', ax, direction, bootstrap_ci, boot_size, params_all
#     )
#     conf_rp_inf_nat, conf_rp_sup_nat = _rp_plot_data(
#         nat_wwa, fit_function, nat_color, 'NAT', ax, direction, bootstrap_ci, boot_size, params_nat
#     )

#     ax.axhline(thresh, color='k', ls='--')

#     # add return period estimate for ALL
#     idx = find_nearest(thresh, all_wwa)

#     ymin, ymax = ax.get_ylim()

#     if all_span_checker:
#         ax.axvspan(
#             conf_rp_inf_all[idx], conf_rp_sup_all[idx],
#             ymin=0, ymax=(thresh - ymin)/ (ymax - ymin),
#             facecolor='silver', edgecolor=all_color,
#             linewidth=2., alpha=0.3, zorder=0
#         )

#     # add return period estimate for NAT
#     idx = find_nearest(thresh, nat_wwa)

#     if nat_span_checker:
#         ax.axvspan(
#             conf_rp_inf_nat[idx], conf_rp_sup_nat[idx],
#             ymin=0, ymax=(thresh - ymin)/ (ymax - ymin),
#             facecolor='silver', edgecolor=nat_color,
#             linewidth=2., alpha=0.3, zorder=0
#         )

#     ax.legend()

# ###############################################################################
