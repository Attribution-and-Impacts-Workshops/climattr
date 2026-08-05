import numpy as np

from scipy.special import gammaln
from statsmodels.base.model import GenericLikelihoodModel

from climattr.minimization.utils import choose_strategy

class GEVModel(GenericLikelihoodModel):

    def __init__(self, endog, exog, strategy='linear', **kwds):
        self.strategy = strategy

        super(GEVModel, self).__init__(endog, exog, **kwds)

    def nloglikeobs(self, params):

        mu_0, sigma_0, c = params[:3]
        alphas = params[3:]

        xi = - c

        covariates = self.exog
        x = self.endog

        mu_tas_est, sigma_tas_est = choose_strategy(
            self.strategy, mu_0, sigma_0, alphas, covariates
        )

        t = (x - mu_tas_est) / sigma_tas_est
        arg = 1 - xi * t
        if np.any(arg <= 0):
            return -np.inf * np.ones(covariates.shape[0])

        log_arg = np.log(arg)
        log_likelihood = -np.log(sigma_tas_est) - (1 - 1 / xi) * log_arg - arg ** (1 / xi)

        return - log_likelihood

    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwargs):
        # Provide starting values if not set
        if start_params is None:
            n_covariates = self.exog.shape[1]
            mu_0_start = np.mean(self.endog)
            sigma_0_start = np.std(self.endog)
            c_start = - 0.1
            start_params = np.array([mu_0_start, sigma_0_start, c_start, *np.zeros(n_covariates)])

        # Call the superclass fit method
        return super(GEVModel, self).fit(
            start_params=start_params, maxiter=maxiter, maxfun=maxfun, **kwargs
        )

###############################################################################

class NormModel(GenericLikelihoodModel):

    def __init__(self, endog, exog, strategy='linear', **kwds):
        self.strategy = strategy

        super(NormModel, self).__init__(endog, exog, **kwds)

    def nloglikeobs(self, params):

        mu_0, sigma_0, c = params[:3]
        alphas = params[3:]

        covariates = self.exog
        x = self.endog

        mu_tas_est, sigma_tas_est = choose_strategy(
            self.strategy, mu_0, sigma_0, alphas, covariates
        )

        log_likelihood = - np.log(sigma_tas_est ** 2) - (x - mu_tas_est) ** 2 / sigma_tas_est ** 2

        return - log_likelihood

    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwargs):
        # Provide starting values if not set
        if start_params is None:
            n_covariates = self.exog.shape[1]
            mu_0_start = np.mean(self.endog)
            sigma_0_start = np.std(self.endog)
            c_start = - 0.1
            start_params = np.array([mu_0_start, sigma_0_start, c_start, *np.zeros(n_covariates)])

        # Call the superclass fit method
        return super(NormModel, self).fit(
            start_params=start_params, maxiter=maxiter, maxfun=maxfun, **kwargs
        )

###############################################################################

class GammaModel(GenericLikelihoodModel):

    def __init__(self, endog, exog, strategy='linear', **kwds):
        self.strategy = strategy

        super(GammaModel, self).__init__(endog, exog, **kwds)

    def nloglikeobs(self, params):

        mu_0, sigma_0, c = params[:3]
        alphas = params[3:]

        xi = - c

        covariates = self.exog
        x = self.endog

        mu_tas_est, sigma_tas_est = choose_strategy(
            self.strategy, mu_0, sigma_0, alphas, covariates
        )

        if xi <= 0:
            return -np.inf * np.ones(covariates.shape[0])

        # Compute the log-likelihood
        log_likelihood = (xi - 1) * np.log(x) - \
            x / sigma_tas_est - \
            xi * np.log(sigma_tas_est) - \
            gammaln(xi)

        return - log_likelihood

    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwargs):
        # Provide starting values if not set
        if start_params is None:
            n_covariates = self.exog.shape[1]
            mu_0_start = np.mean(self.endog)
            sigma_0_start = np.std(self.endog)
            c_start = - 0.1
            start_params = np.array([mu_0_start, sigma_0_start, c_start, *np.zeros(n_covariates)])

        # Call the superclass fit method
        return super(GammaModel, self).fit(
            start_params=start_params, maxiter=maxiter, maxfun=maxfun, **kwargs
        )

###############################################################################

class GPDModel(GenericLikelihoodModel):

    def __init__(self, endog, exog, strategy='linear', **kwds):
        self.strategy = strategy

        super(GPDModel, self).__init__(endog, exog, **kwds)

    def nloglikeobs(self, params):

        mu_0, sigma_0, c = params[:3]
        alphas = params[3:]

        xi = - c

        covariates = self.exog
        x = self.endog

        mu_tas_est, sigma_tas_est = choose_strategy(
            self.strategy, mu_0, sigma_0, alphas, covariates
        )

        # Compute shifted data
        shifted_data = x - mu_tas_est

        if xi != 0:
            # Compute the term t = 1 + xi * (x_i - mu) / sigma
            t = 1 + xi * shifted_data / sigma_tas_est

            # Check the constraint t > 0 for all data points
            if np.any(t <= 0):
                return -np.inf * np.ones(covariates.shape[0])

            # Compute the log-likelihood
            log_likelihood = np.log(sigma_tas_est) - (1 / xi + 1) * t
        else:
            # When xi = 0, GPD reduces to exponential distribution
            log_likelihood = np.log(sigma_tas_est) - (1 / sigma_tas_est) * shifted_data

        return - log_likelihood

    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwargs):
        # Provide starting values if not set
        if start_params is None:
            n_covariates = self.exog.shape[1]
            mu_0_start = np.mean(self.endog)
            sigma_0_start = np.std(self.endog)
            c_start = - 0.1
            start_params = np.array([mu_0_start, sigma_0_start, c_start, *np.zeros(n_covariates)])

        # Call the superclass fit method
        return super(GPDModel, self).fit(
            start_params=start_params, maxiter=maxiter, maxfun=maxfun, **kwargs
        )

###############################################################################

class GumbelModel(GenericLikelihoodModel):

    def __init__(self, endog, exog, strategy='linear', **kwds):
        self.strategy = strategy

        super(GumbelModel, self).__init__(endog, exog, **kwds)

    def nloglikeobs(self, params):

        mu_0, sigma_0, c = params[:3]
        alphas = params[3:]

        covariates = self.exog
        x = self.endog

        mu_tas_est, sigma_tas_est = choose_strategy(
            self.strategy, mu_0, sigma_0, alphas, covariates
        )

        # standard (right-skewed) Gumbel, matching scipy's gumbel_r
        z = (x - mu_tas_est) / sigma_tas_est
        log_likelihood = -np.log(sigma_tas_est) - z - np.exp(-z)

        return - log_likelihood

    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwargs):
        # Provide starting values if not set
        if start_params is None:
            n_covariates = self.exog.shape[1]
            mu_0_start = np.mean(self.endog)
            sigma_0_start = np.std(self.endog)
            c_start = - 0.1
            start_params = np.array([mu_0_start, sigma_0_start, c_start, *np.zeros(n_covariates)])

        # Call the superclass fit method
        return super(GumbelModel, self).fit(
            start_params=start_params, maxiter=maxiter, maxfun=maxfun, **kwargs
        )

###############################################################################

class WeibullModel(GenericLikelihoodModel):

    def __init__(self, endog, exog, strategy='linear', **kwds):
        self.strategy = strategy

        super(WeibullModel, self).__init__(endog, exog, **kwds)

    def nloglikeobs(self, params):

        mu_0, sigma_0, c = params[:3]
        alphas = params[3:]

        covariates = self.exog
        x = self.endog

        mu_tas_est, sigma_tas_est = choose_strategy(
            self.strategy, mu_0, sigma_0, alphas, covariates
        )

        # matches scipy's weibull_min: shape c > 0, support x >= loc
        if c <= 0:
            return -np.inf * np.ones(covariates.shape[0])

        z = (x - mu_tas_est) / sigma_tas_est
        if np.any(z <= 0):
            return -np.inf * np.ones(covariates.shape[0])

        log_likelihood = np.log(c) - np.log(sigma_tas_est) + (c - 1) * np.log(z) - z ** c

        return - log_likelihood

    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwargs):
        # Provide starting values if not set
        if start_params is None:
            n_covariates = self.exog.shape[1]
            # loc must sit below the data (weibull_min's support is x >= loc)
            mu_0_start = np.min(self.endog) - 0.1 * np.std(self.endog)
            sigma_0_start = np.std(self.endog)
            c_start = 1.5
            start_params = np.array([mu_0_start, sigma_0_start, c_start, *np.zeros(n_covariates)])

        # Call the superclass fit method
        return super(WeibullModel, self).fit(
            start_params=start_params, maxiter=maxiter, maxfun=maxfun, **kwargs
        )

###############################################################################

class ExponentialModel(GenericLikelihoodModel):

    def __init__(self, endog, exog, strategy='linear', **kwds):
        self.strategy = strategy

        super(ExponentialModel, self).__init__(endog, exog, **kwds)

    def nloglikeobs(self, params):

        mu_0, sigma_0, c = params[:3]
        alphas = params[3:]

        covariates = self.exog
        x = self.endog

        mu_tas_est, sigma_tas_est = choose_strategy(
            self.strategy, mu_0, sigma_0, alphas, covariates
        )

        # matches scipy's expon: support x >= loc
        shifted_data = x - mu_tas_est
        if np.any(shifted_data < 0):
            return -np.inf * np.ones(covariates.shape[0])

        log_likelihood = -np.log(sigma_tas_est) - shifted_data / sigma_tas_est

        return - log_likelihood

    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwargs):
        # Provide starting values if not set
        if start_params is None:
            n_covariates = self.exog.shape[1]
            # loc must sit at/below the data (expon's support is x >= loc)
            mu_0_start = np.min(self.endog) - 0.1 * np.std(self.endog)
            sigma_0_start = np.std(self.endog)
            c_start = - 0.1
            start_params = np.array([mu_0_start, sigma_0_start, c_start, *np.zeros(n_covariates)])

        # Call the superclass fit method
        return super(ExponentialModel, self).fit(
            start_params=start_params, maxiter=maxiter, maxfun=maxfun, **kwargs
        )

###############################################################################

class LognormModel(GenericLikelihoodModel):

    def __init__(self, endog, exog, strategy='linear', **kwds):
        self.strategy = strategy

        super(LognormModel, self).__init__(endog, exog, **kwds)

    def nloglikeobs(self, params):

        mu_0, sigma_0, c = params[:3]
        alphas = params[3:]

        covariates = self.exog
        x = self.endog

        if np.any(x <= 0):
            return -np.inf * np.ones(covariates.shape[0])

        # mu_tas_est/sigma_tas_est model the mean/std of log(x), matching
        # scipy's lognorm(s=sigma, loc=0, scale=exp(mu)) convention
        mu_tas_est, sigma_tas_est = choose_strategy(
            self.strategy, mu_0, sigma_0, alphas, covariates
        )

        log_x = np.log(x)
        log_likelihood = -log_x - np.log(sigma_tas_est) - \
            (log_x - mu_tas_est) ** 2 / (2 * sigma_tas_est ** 2)

        return - log_likelihood

    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwargs):
        # Provide starting values if not set
        if start_params is None:
            n_covariates = self.exog.shape[1]
            log_endog = np.log(self.endog)
            mu_0_start = np.mean(log_endog)
            sigma_0_start = np.std(log_endog)
            c_start = - 0.1
            start_params = np.array([mu_0_start, sigma_0_start, c_start, *np.zeros(n_covariates)])

        # Call the superclass fit method
        return super(LognormModel, self).fit(
            start_params=start_params, maxiter=maxiter, maxfun=maxfun, **kwargs
        )

###############################################################################

# Single source of truth mapping a scipy.stats distribution name to its
# GenericLikelihoodModel subclass, shared by fit_data (climattr.minimization.
# fit) and fit_summary (climattr.attribution.wwa) rather than each keeping
# its own copy of this dict.
FIT_FUNCTIONS = {
    'genextreme': GEVModel,
    'norm': NormModel,
    'gamma': GammaModel,
    'genpareto': GPDModel,
    'gumbel_r': GumbelModel,
    'weibull_min': WeibullModel,
    'expon': ExponentialModel,
    'lognorm': LognormModel,
}

###############################################################################