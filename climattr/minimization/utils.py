import numpy as np

from typing import List, Tuple

def _as_2d_covariates(covariates: np.ndarray) -> np.ndarray:
    """
    Ensure covariates is a 2D array of shape (n_obs, n_covariates).

    Accepts a 1D array (single covariate) and reshapes it to (-1, 1) so
    callers don't need to remember to do this themselves.
    """
    covariates = np.asarray(covariates)

    if covariates.ndim == 1:
        covariates = covariates.reshape(-1, 1)

    return covariates

###############################################################################

def _as_1d_alphas(alphas: np.ndarray) -> np.ndarray:
    """
    Ensure alphas is at least a 1D array, so a bare scalar (e.g. a single
    coefficient pulled out of a fitted params array) is still valid for the
    single-covariate case.
    """
    return np.atleast_1d(alphas)

###############################################################################

def linear_function(
    mu_0: float,
    sigma_0: float,
    alphas: np.ndarray,
    covariates: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the estimated μ(tas) and σ(tas) using a linear model.

    Parameters:
    -----------
    mu_0 : float
        The baseline mean (μ_0) at the initial temperature.
    sigma_0 : float
        The baseline standard deviation (σ_0) at the initial temperature.
    alphas : np.ndarray
        Array of linear coefficients, one per covariate.
    covariates : np.ndarray
        Array of shape (n_obs, n_covariates) with covariate time series.

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing the estimated μ(tas) and σ(tas).
    """
    # Compute estimated μ(tas) and σ(tas)
    covariates = _as_2d_covariates(covariates)
    alphas = _as_1d_alphas(alphas)

    mu_tas_est = mu_0 + covariates @ alphas
    sigma_tas_est = sigma_0 * np.ones(covariates.shape[0])

    return mu_tas_est, sigma_tas_est

###############################################################################

def exponential_function(
    mu_0: float,
    sigma_0: float,
    alphas: np.ndarray,
    covariates: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the estimated μ(tas) and σ(tas) using an exponential model.

    Parameters:
    -----------
    mu_0 : float
        The baseline mean (μ_0) at the initial temperature.
    sigma_0 : float
        The baseline standard deviation (σ_0) at the initial temperature.
    alphas : np.ndarray
        Array of exponential coefficients, one per covariate.
    covariates : np.ndarray
        Array of shape (n_obs, n_covariates) with covariate time series.

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing the estimated μ(tas) and σ(tas) based on an
        exponential relationship.
    """
    # Compute estimated μ(tas) and σ(tas)
    covariates = _as_2d_covariates(covariates)
    alphas = _as_1d_alphas(alphas)

    exp_term = np.exp(covariates @ alphas / mu_0)
    mu_tas_est = mu_0 * exp_term
    sigma_tas_est = sigma_0 * exp_term

    return mu_tas_est, sigma_tas_est

###############################################################################

def dist_args(
    fit_function_name: str,
    mu: float,
    sigma: float,
    c: float) -> List[float]:
    """
    Build the positional [shape, loc, scale] arguments (or [loc, scale] for
    'norm', which has no shape parameter) that scipy.stats distribution
    methods (pdf, cdf, sf, ppf, rvs, ...) expect, from the fitted mu, sigma
    and shape (c) parameters.

    Parameters:
    -----------
    fit_function_name : str
        The name of the distribution ('norm', 'genextreme', 'gamma',
        'genpareto').
    mu : float
        The estimated location (μ) at some covariate value.
    sigma : float
        The estimated scale (σ) at some covariate value.
    c : float
        The fitted shape parameter (unused for 'norm').

    Returns:
    --------
    List[float]
        The positional arguments for the named scipy.stats distribution.
    """
    dist_params = {
        'norm': [mu, sigma],
        'genextreme': [c, mu, sigma],
        'gamma': [c, mu, sigma],
        'genpareto': [c, mu, sigma]
    }

    return dist_params[fit_function_name]

###############################################################################

def choose_strategy(
    strategy: str,
    mu_0: float,
    sigma_0: float,
    alphas: np.ndarray,
    covariates: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select and apply a model (linear or exponential) for computing μ(tas) and σ(tas).

    Parameters:
    -----------
    strategy : str
        The strategy to use ('linear' or 'exponential').
    mu_0 : float
        The baseline mean (μ_0) at the initial temperature.
    sigma_0 : float
        The baseline standard deviation (σ_0) at the initial temperature.
    alphas : np.ndarray
        Array of coefficients, one per covariate.
    covariates : np.ndarray
        Array of shape (n_obs, n_covariates) with covariate time series.

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing the estimated μ(tas) and σ(tas) based on the selected
        strategy.
    """
    # Compute estimated μ(tas) and σ(tas)
    if strategy == 'linear':
        mu_tas_est, sigma_tas_est = linear_function(
            mu_0, sigma_0, alphas, covariates
        )
    else:
        mu_tas_est, sigma_tas_est = exponential_function(
            mu_0, sigma_0, alphas, covariates
        )

    return mu_tas_est, sigma_tas_est

###############################################################################
