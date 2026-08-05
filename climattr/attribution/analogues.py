import numpy as np
import pandas as pd
import xarray as xr

from typing import Callable, Sequence, Tuple, Union

from joblib import Parallel, delayed

from climattr.attribution.utils import (
    _calc_bootstrap_ensemble,
    _dataset_to_matrix,
    _get_common_variables,
    _pairwise_distance,
    _resolve_analogue_groups
)
from climattr.utils import get_percentiles_from_ci
from climattr.validator import validate_ci, validate_distance_metric

def compute_distances(
    event_pattern: xr.Dataset,
    all_events: xr.Dataset,
    distance: Union[str, Callable] = "euclidean",
    variables: Union[Sequence[str], None] = None) -> pd.DataFrame:
    """
    Compute the distance between `event_pattern` and every time step of
    `all_events`, using all data variables common to both datasets (or
    the ones given in `variables`) stacked over (lat, lon).

    Parameters
    ----------
    event_pattern : xr.Dataset
        Dims (lat, lon), optionally a length-1 'time' dim. The reference
        pattern (e.g. an extreme event, averaged over its duration).

    all_events : xr.Dataset
        Dims (time, lat, lon). The pool of candidate dates to compare
        against.

    distance : Union[str, Callable], optional, default = 'euclidean'
        One of 'euclidean', 'manhattan', 'chebyshev', 'cosine',
        'correlation', or a custom callable
        `f(event_vector, data_matrix) -> 1d array of distances`
        where `data_matrix` has shape (n_time, n_features).

    variables : Sequence[str], optional
        Which data variables to use. Defaults to all variables present
        in both datasets.

    Returns
    -------
    pd.DataFrame
        Columns 'time', 'distance' - one row per date in `all_events`,
        sorted by ascending distance.
    """
    validate_distance_metric(distance)

    if "time" not in all_events.dims:
        raise ValueError("all_events must have a 'time' dimension")

    variables = _get_common_variables(event_pattern, all_events, variables)

    if "time" in event_pattern.dims and event_pattern.sizes.get("time", 1) == 1:
        event_pattern = event_pattern.squeeze("time", drop=True)

    event_matrix = _dataset_to_matrix(event_pattern, variables)
    all_matrix = _dataset_to_matrix(all_events, variables)

    event_vec = event_matrix[0]

    if callable(distance):
        distances = np.asarray(distance(event_vec, all_matrix))
    else:
        distances = _pairwise_distance(event_vec, all_matrix, distance)

    df = pd.DataFrame({"time": np.asarray(all_events["time"].values), "distance": distances})

    return df.sort_values("distance", ascending=True, ignore_index=True)

###############################################################################

def find_analogues(
    event_pattern: xr.Dataset,
    all_events: xr.Dataset,
    n_analogues: int = 15,
    distance: Union[str, Callable] = "euclidean",
    variables: Union[Sequence[str], None] = None) -> Tuple[xr.Dataset, pd.DataFrame]:
    """
    Find the `n_analogues` dates in `all_events` most similar to
    `event_pattern` (smallest distance), using all shared data variables.

    See `compute_distances` for parameter details.

    Parameters
    ----------
    event_pattern : xr.Dataset
        Dims (lat, lon), optionally a length-1 'time' dim.

    all_events : xr.Dataset
        Dims (time, lat, lon).

    n_analogues : int, optional, default = 15
        The number of closest dates to select as analogues.

    distance : Union[str, Callable], optional, default = 'euclidean'
        The distance metric to use. See `compute_distances`.

    variables : Sequence[str], optional
        Which data variables to use. Defaults to all variables present
        in both datasets.

    Returns
    -------
    Tuple[xr.Dataset, pd.DataFrame]
        'analogues' : subset of `all_events` at the selected analogue
        times, sorted by increasing distance.

        'distances_df' : distance of every date in `all_events` to
        `event_pattern` (columns 'time', 'distance', 'is_analogue'),
        sorted ascending.
    """
    distances_df = compute_distances(event_pattern, all_events, distance, variables)
    distances_df["is_analogue"] = False
    distances_df.loc[distances_df.index[:n_analogues], "is_analogue"] = True

    analogue_times = distances_df.loc[distances_df["is_analogue"], "time"].to_numpy()
    analogues = all_events.sel(time=analogue_times)

    return analogues, distances_df

###############################################################################

def composite_differences(
    present_analogues: xr.Dataset,
    past_analogues: xr.Dataset,
    n_boot: int = 300,
    seed: Union[int, None] = None) -> Tuple[xr.Dataset, xr.Dataset]:
    """
    Compute present-minus-past composite differences from pre-selected
    analogue datasets, for every shared variable, and flag which grid
    points are significant via a bootstrap test.

    Parameters
    ----------
    present_analogues : xr.Dataset
        Dims (time, lat, lon) - the analogue dates for the present
        period, already containing whatever fields you want composited
        (anomalies, raw values, or a mix). Any preprocessing should
        already be applied; this function only composites and
        differences. Group sizes need not be equal.

    past_analogues : xr.Dataset
        Dims (time, lat, lon) - the analogue dates for the past period.
        See `present_analogues`.

    n_boot : int, optional, default = 300
        Number of bootstrap replicates.

    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    Tuple[xr.Dataset, xr.Dataset]
        'differences' : present-minus-past composite difference for
        every variable shared by `present_analogues` and
        `past_analogues`.

        'significant' : same variables/shape as 'differences', boolean:
        True where the observed difference exceeds two standard
        deviations of the bootstrap distribution of differences.
    """
    past_analogues, present_analogues, common_vars = _resolve_analogue_groups(past_analogues, present_analogues)

    past_composite = past_analogues.mean("time", skipna=True)
    present_composite = present_analogues.mean("time", skipna=True)
    observed_diff = present_composite - past_composite

    n_past = past_analogues.sizes["time"]
    n_present = present_analogues.sizes["time"]
    n_pool = n_past + n_present
    pooled = xr.concat([past_analogues, present_analogues], dim="time")

    rng = np.random.default_rng(seed)
    boot_diffs = []
    for _ in range(n_boot):
        perm = rng.permutation(n_pool)
        group_past = pooled.isel(time=perm[:n_past]).mean("time", skipna=True)
        group_present = pooled.isel(time=perm[n_past:n_past + n_present]).mean("time", skipna=True)
        boot_diffs.append(group_present - group_past)
    boot_diffs = xr.concat(boot_diffs, dim="bootstrap")

    boot_mean = boot_diffs.mean("bootstrap")
    boot_std = boot_diffs.std("bootstrap")

    significant = xr.Dataset(
        {
            var: np.abs(observed_diff[var] - boot_mean[var]) > 2 * boot_std[var]
            for var in common_vars
        }
    )

    return observed_diff, significant

###############################################################################

def attribution_metrics(
    present_analogues: xr.Dataset,
    past_analogues: xr.Dataset,
    n_boot: int = 300,
    bootstrap_ci: int = 95,
    seed: int = 42,
    n_jobs: int = -1,
    summary_statistics: bool = True) -> pd.DataFrame:
    """
    Calculate the area-averaged (spatial-mean) change in intensity between
    two pre-selected analogue groups, with a bootstrap confidence interval,
    for every variable shared by `present_analogues` and `past_analogues`.

    Builds on the same present-minus-past composite difference as
    `composite_differences`, but instead of a per-gridpoint permutation
    significance test, reduces each composite difference to a single
    spatial-mean value per variable (the "area difference") and resamples
    `present_analogues`/`past_analogues` independently with replacement
    (a case-resampling bootstrap, not the permutation test used for
    `composite_differences`' significance flag) to build a distribution of
    that area-mean difference - matching the estimate/ci_inf/ci_sup summary
    table convention of `climattr.attribution.risk_based.attribution_metrics`
    and `climattr.attribution.wwa.attribution_metrics`.

    Parameters
    ----------
    present_analogues : xr.Dataset
        Dims (time, lat, lon) - the analogue dates for the present period,
        already containing whatever fields you want composited (anomalies,
        raw values, or a mix). Group sizes need not be equal.

    past_analogues : xr.Dataset
        Dims (time, lat, lon) - the analogue dates for the past period.
        See `present_analogues`.

    n_boot : int, optional, default = 300
        Number of bootstrap replicates.

    bootstrap_ci : int, optional, default = 95
        The confidence interval (CI) percentage for bootstrapping.

    seed : int, optional, default = 42
        Random seed for the past-group bootstrap resample. The
        present-group resample uses `seed + 1`, so the two groups are not
        position-matched draws of the same underlying random sequence - see
        `climattr.attribution.risk_based.attribution_metrics`'s `seed`
        docstring for why that matters whenever the two groups could be
        correlated.

    n_jobs : int, optional, default = -1
        Number of parallel jobs (-1 for all cores).

    summary_statistics : bool, optional, default = True
        If True, return the 'estimate'/'ci_inf'/'ci_sup' summary table. If
        False, return one row per successful bootstrap replicate instead.

    Returns
    -------
    pd.DataFrame
        Indexed by variable name, columns 'estimate', 'ci_inf', 'ci_sup' -
        the observed area-averaged present-minus-past difference and its
        bootstrap confidence interval, plus trailing 'n_past', 'n_present',
        'n_boot', 'n_failed' rows. If `summary_statistics` is False, one row
        per successful bootstrap replicate (columns = variable names)
        instead.
    """
    validate_ci(bootstrap_ci)

    past_analogues, present_analogues, common_vars = _resolve_analogue_groups(past_analogues, present_analogues)

    def _area_diff(past, present):
        diff = present.mean("time", skipna=True) - past.mean("time", skipna=True)
        return {var: float(diff[var].mean(skipna=True)) for var in common_vars}

    point_est = _area_diff(past_analogues, present_analogues)

    n_past = past_analogues.sizes["time"]
    n_present = present_analogues.sizes["time"]

    # different seeds: _calc_bootstrap_ensemble reseeds numpy's global RNG
    # internally, so reusing one seed for both groups would draw past/present
    # at the same positions every replicate whenever the two groups are
    # correlated (see the `seed` docstring above)
    past_idx_boot = _calc_bootstrap_ensemble(np.arange(n_past), boot_size=n_boot, seed=seed)
    present_idx_boot = _calc_bootstrap_ensemble(np.arange(n_present), boot_size=n_boot, seed=seed + 1)

    def compute_boot(boot):
        try:
            past_boot = past_analogues.isel(time=past_idx_boot[boot])
            present_boot = present_analogues.isel(time=present_idx_boot[boot])
            return _area_diff(past_boot, present_boot)
        except Exception:
            return None

    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_boot)(boot) for boot in range(int(n_boot))
    )
    results = [r for r in results if r is not None]
    n_failed = int(n_boot) - len(results)

    if summary_statistics:
        ci_inf, ci_sup = get_percentiles_from_ci(bootstrap_ci)

        row_names = common_vars + ['n_past', 'n_present', 'n_boot', 'n_failed']
        metrics_result = pd.DataFrame(
            np.full((len(row_names), 3), np.nan),
            index=row_names,
            columns=['estimate', 'ci_inf', 'ci_sup']
        )

        for var in common_vars:
            values = np.array([r[var] for r in results])
            values = values[~np.isnan(values)]

            metrics_result.loc[var, 'estimate'] = point_est[var]
            metrics_result.loc[var, 'ci_inf'] = np.percentile(values, ci_inf)
            metrics_result.loc[var, 'ci_sup'] = np.percentile(values, ci_sup)

        metrics_result.loc['n_past', 'estimate'] = n_past
        metrics_result.loc['n_present', 'estimate'] = n_present
        metrics_result.loc['n_boot', 'estimate'] = len(results)
        metrics_result.loc['n_failed', 'estimate'] = n_failed
    else:
        metrics_result = pd.DataFrame(results)

    return metrics_result

###############################################################################
