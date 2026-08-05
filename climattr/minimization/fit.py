import warnings

import numpy as np
import pandas as pd
import xarray as xr

from datetime import datetime
from scipy import stats
from typing import List, Tuple, Union

from climattr.minimization import likelihood
from climattr.minimization.utils import choose_strategy, dist_args


def fit_data(
    x: pd.Series,
    covariates: Union[pd.Series, pd.DataFrame],
    fit_function_name: str,
    strategy: str,
    verbose: bool = True) -> np.ndarray:
    """
    Fits a statistical model to the provided data using a specified distribution
    and parameterization strategy.

    This function fits a statistical model to the observed data `x` by modeling
    the relationship between the data and one or more covariates.
    It supports fitting using different statistical distributions and
    parameterization strategies to account for potential changes in the
    distribution parameters with respect to the covariates.

    Parameters
    ----------
    x : pd.Series
        The observed data to be fitted.
    covariates : Union[pd.Series, pd.DataFrame]
        One or more covariate time series. A Series for a single covariate
        or a DataFrame with one column per covariate.
    fit_function_name : str
        The name of the distribution to fit. Supported options are:
        - `'genextreme'`: Generalized Extreme Value distribution.
        - `'norm'`: Normal (Gaussian) distribution.
        - `'gamma'`: Gamma distribution.
        - `'genpareto'`: Generalized Pareto distribution.
    strategy : str
        The parameterization strategy to use for modeling how the distribution
        parameters change with the covariates. Supported options are:
        - `'linear'`: Parameters change linearly with the covariates.
        - `'exponential'`: Parameters change exponentially with the covariates.
    verbose : bool, optional
        If `True`, prints a summary of the fit results. Default is `True`.

    Returns
    -------
    np.ndarray
        An array containing the fitted model parameters. The specific parameters
        returned depend on the chosen distribution and strategy.
    """
    # Ensure covariates is 2D
    if isinstance(covariates, pd.Series):
        covariates = covariates.values.reshape(-1, 1)
    elif isinstance(covariates, pd.DataFrame):
        covariates = covariates.values

    model = likelihood.FIT_FUNCTIONS[fit_function_name](x, covariates, strategy=strategy)

    # Fit the model
    if verbose:
        result = model.fit(disp=True)
        print(result.summary())
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.fit(disp=False)

    return result.params

###############################################################################

def shift_data(
    x: pd.Series,
    covariates: Union[pd.Series, pd.DataFrame],
    params: np.ndarray,
    strategy: str,
    reference_covariate: Union[float, np.ndarray]) -> xr.DataArray:
    """
    Shifts each observation in `x` to what it would have looked like under a
    fixed reference covariate value, following the WWA (World Weather
    Attribution) event-shifting approach: every observation is detrended by
    the fitted covariate relationship and re-centered at
    `reference_covariate`, preserving its own residual/variability.

    For the 'linear' strategy the shift is additive
    (x - mu(covariate) + mu(reference_covariate)), matching a constant
    location shift. For the 'exponential' strategy the shift is
    multiplicative (x * mu(reference_covariate) / mu(covariate)), matching a
    constant relative (percentage) change per unit of covariate.

    Parameters
    ----------
    x : pd.Series
        The observed data, indexed the same way as `covariates`.

    covariates : Union[pd.Series, pd.DataFrame]
        One or more covariate time series, aligned with `x`.

    params : np.ndarray
        The fitted parameters [mu_0, sigma_0, c, alpha_1, ..., alpha_n], as
        returned by `fit_data`.

    strategy : str
        The parameterization strategy used to fit `params`. Either 'linear'
        or 'exponential'.

    reference_covariate : Union[float, np.ndarray]
        The covariate value(s) to shift every observation to, e.g. the
        current GMST anomaly (for an "ALL" / factual series) or a
        counterfactual GMST anomaly (for a "NAT" / natural series).

    Returns
    -------
    xr.DataArray
        The shifted series, indexed the same way as `x`.
    """
    mu_0, sigma_0, _ = params[:3]
    alphas = params[3:]

    # Ensure covariates is 2D, same convention as fit_data
    if isinstance(covariates, pd.Series):
        covariates_arr = covariates.values.reshape(-1, 1)
    else:
        covariates_arr = covariates.values

    reference_arr = np.atleast_2d(reference_covariate)

    mu_obs, _ = choose_strategy(strategy, mu_0, sigma_0, alphas, covariates_arr)
    mu_ref, _ = choose_strategy(strategy, mu_0, sigma_0, alphas, reference_arr)
    mu_ref = mu_ref.item()

    if strategy == 'linear':
        shifted = x.values - mu_obs + mu_ref
    else:
        shifted = x.values * (mu_ref / mu_obs)

    return pd.Series(shifted, index=x.index, name=x.name).to_xarray()

###############################################################################

def get_wwa_series(
    x: pd.Series,
    covariates: Union[pd.Series, pd.DataFrame],
    params: np.ndarray,
    strategy: str,
    all_covariate: Union[float, np.ndarray],
    nat_covariate: Union[float, np.ndarray]) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Builds the "ALL" (factual) and "NAT" (counterfactual) shifted series used
    by WWA-style attribution plots, ready to be passed directly to
    `climattr.attribution.risk_based.histogram_plot` /
    `climattr.attribution.risk_based.rp_plot`.

    This is a convenience wrapper around `shift_data` that shifts `x` to
    both reference covariate values in one call.

    Parameters
    ----------
    x : pd.Series
        The observed data, indexed the same way as `covariates`.

    covariates : Union[pd.Series, pd.DataFrame]
        One or more covariate time series, aligned with `x`.

    params : np.ndarray
        The fitted parameters [mu_0, sigma_0, c, alpha_1, ..., alpha_n], as
        returned by `fit_data`.

    strategy : str
        The parameterization strategy used to fit `params`. Either 'linear'
        or 'exponential'.

    all_covariate : Union[float, np.ndarray]
        The covariate value(s) representing the current/factual climate,
        e.g. the GMST anomaly for the event's year.

    nat_covariate : Union[float, np.ndarray]
        The covariate value(s) representing the natural/counterfactual
        climate, e.g. a pre-industrial or reduced-warming GMST anomaly.

    Returns
    -------
    Tuple[xr.DataArray, xr.DataArray]
        The (all_wwa, nat_wwa) shifted series.
    """
    all_wwa = shift_data(x, covariates, params, strategy, all_covariate)
    nat_wwa = shift_data(x, covariates, params, strategy, nat_covariate)

    return all_wwa, nat_wwa

###############################################################################

def dist_params_at_covariate(
    params: np.ndarray,
    covariate: Union[float, np.ndarray],
    fit_function_name: str,
    strategy: str) -> List[float]:
    """
    Positional scipy.stats distribution arguments of the fitted nonstationary
    model at a fixed covariate value (a plain number or array, as opposed to
    `extrapolate_data`, which looks the covariate value up from a date).

    Parameters
    ----------
    params : np.ndarray
        The fitted parameters [mu_0, sigma_0, c, alpha_1, ..., alpha_n], as
        returned by `fit_data`.

    covariate : Union[float, np.ndarray]
        The covariate value(s) at which to evaluate the distribution.

    fit_function_name : str
        The name of the distribution ('genextreme', 'norm', 'gamma',
        'genpareto').

    strategy : str
        The parameterization strategy used to fit `params`. Either 'linear'
        or 'exponential'.

    Returns
    -------
    List[float]
        The positional distribution arguments at `covariate`.
    """
    mu_0, sigma_0, c = params[:3]
    alphas = params[3:]

    mu, sigma = choose_strategy(strategy, mu_0, sigma_0, alphas, np.atleast_2d(covariate))

    return dist_args(fit_function_name, mu.item(), sigma.item(), c)

###############################################################################

def extrapolate_data(
    covariates_df: pd.DataFrame,
    params: np.ndarray,
    date: Union[str, datetime],
    fit_function_name: str,
    strategy: str,
    size: int = 500) -> np.ndarray:
    """
    Extrapolates data for a future time point based on the fitted model parameters
    and covariate values.

    Parameters:
    -----------
    covariates_df : pd.DataFrame
        DataFrame with covariate time series indexed by time.
    params : np.ndarray
        The fitted parameters [mu_0, sigma_0, c, alpha_1, ..., alpha_n].
    date : Union[str, datetime]
        The date for which the extrapolation is to be made.
    fit_function_name : str
        The name of the distribution to use ('genextreme', 'norm', 'gamma', 'genpareto').
    strategy : str
        The strategy to use ('linear' or 'exponential') for computing the estimated
        parameters.
    size : int, optional
        The number of extrapolated data points to generate (default is 500).

    Returns:
    --------
    np.ndarray
        A NumPy array of extrapolated data points generated from the specified
        distribution.
    """
    # Get covariate values at the given date as a 1D array
    cov_values = covariates_df.loc[date].values

    function_params = dist_params_at_covariate(params, cov_values, fit_function_name, strategy)

    extrapolated_data = getattr(
        stats, fit_function_name
    ).rvs(*function_params, size=size)

    return function_params, extrapolated_data

###############################################################################

def aic(log_likelihood: float, k: int) -> float:
    """
    Calculate the Akaike Information Criterion (AIC).

    Parameters
    ----------
    log_likelihood : float
        The maximized log-likelihood of the fitted model.
    k : int
        The number of estimated parameters in the model.

    Returns
    -------
    float
        The AIC value. Lower values indicate a better trade-off
        between goodness-of-fit and model complexity.
    """
    return 2 * k - 2 * log_likelihood

###############################################################################

def bic(log_likelihood: float, k: int, n: int) -> float:
    """
    Calculate the Bayesian Information Criterion (BIC).

    Parameters
    ----------
    log_likelihood : float
        The maximized log-likelihood of the fitted model.
    k : int
        The number of estimated parameters in the model.
    n : int
        The number of observations in the dataset.

    Returns
    -------
    float
        The BIC value. Lower values indicate a better trade-off
        between goodness-of-fit and model complexity, with a stronger
        penalty for complexity than AIC.
    """
    return k * np.log(n) - 2 * log_likelihood

###############################################################################

def rsquared(result) -> float:
    """
    Calculate the R² (coefficient of determination) from a fitted WWA model.

    Computes R² as the proportion of variance in the observed data
    explained by the covariate-dependent mean (mu) of the fitted model.

    Parameters
    ----------
    result : statsmodels GenericLikelihoodModelResults
        The result object returned by model.fit().

    Returns
    -------
    float
        The R² value between 0 and 1, where 1 indicates a perfect fit.
    """
    mu_0, sigma_0, _ = result.params[:3]
    alphas = result.params[3:]
    model = result.model
    covariates = model.exog

    mu_fitted, _ = choose_strategy(model.strategy, mu_0, sigma_0, alphas, covariates)

    ss_res = np.sum((model.endog - mu_fitted) ** 2)
    ss_tot = np.sum((model.endog - np.mean(model.endog)) ** 2)

    return 1 - ss_res / ss_tot

###############################################################################
