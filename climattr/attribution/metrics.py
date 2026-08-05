import pandas as pd

from climattr.attribution import analogues, risk_based, wwa


def attribution_metrics(
    method: str = 'wwa',
    **kwargs) -> pd.DataFrame:
    """
    Compute attribution metrics, dispatching to one of climattr's
    attribution methodologies depending on `method`. This is a thin router:
    it does no data transformation itself, it just forwards `**kwargs` to
    whichever implementation is selected, so each method's own argument
    list applies unchanged.

    - method='wwa' (default): delegates to
      `climattr.attribution.wwa.attribution_metrics(all, covariates_df,
      fit_function_name, cov_f, cov_cf, strategy='linear', ev=None,
      rp=None, direction='descending', bootstrap_ci=95, boot_size=500,
      seed=42, n_jobs=-1, return_sample=False)`. Evaluates the fitted
      nonstationary regression directly at `cov_f`/`cov_cf`, bootstrapping
      by refitting the whole regression on resampled rows. Returns model
      parameters, event magnitude, return period, probability ratio (PR)
      and intensity change (dI_abs/dI_rel) - the R `rwwa` package's
      approach.

    - method='risk_based': delegates to
      `climattr.attribution.risk_based.attribution_metrics(all, nat,
      fit_function, thresh, direction='descending', bootstrap_ci=95,
      boot_size=1000, summary_statistics=True, n_jobs=-1)`. `all`/`nat`
      must already be a shifted series (e.g. via coupled model or atmosphere 
      only model) and `fit_function` a scipy.stats distribution object (not a name
      string). Bootstraps by resampling each shifted series independently
      and refitting a stationary distribution to it. Returns PR, FAR
      (Fraction of Attributable Risk) and return periods for the ALL/NAT
      climates - the classic WWA event-shifting approach.

    - method='analogues': delegates to
      `climattr.attribution.analogues.attribution_metrics(present_analogues,
      past_analogues, n_boot=300, bootstrap_ci=95, seed=42, n_jobs=-1,
      summary_statistics=True)`. `present_analogues`/`past_analogues` are
      pre-selected analogue groups (e.g. via
      `climattr.attribution.analogues.find_analogues`). Bootstraps the two
      groups independently and returns the area-averaged (spatial-mean)
      change in intensity between them, with a confidence interval - the
      analogue/composite-difference approach.

    Parameters
    ----------
    method : str, optional
        Which attribution methodology to use: 'wwa' (default), 'risk_based', or
        'analogues'.
    **kwargs
        Forwarded verbatim to the chosen implementation - see above for
        each one's argument list.

    Returns
    -------
    pd.DataFrame
        The attribution metrics table. Its shape and columns depend on
        `method` - see the underlying functions for details.
    """
    if method == 'wwa':
        return wwa.attribution_metrics(**kwargs)

    if method == 'risk_based':
        return risk_based.attribution_metrics(**kwargs)

    if method == 'analogues':
        return analogues.attribution_metrics(**kwargs)

    raise ValueError(f"Unknown method '{method}': expected 'wwa', 'risk_based', or 'analogues'")

###############################################################################
