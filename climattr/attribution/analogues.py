import numpy as np
import pandas as pd
import xarray as xr

from typing import Callable, Sequence, Tuple, Union

from climattr.attribution.utils import _dataset_to_matrix, _get_common_variables, _pairwise_distance
from climattr.validator import validate_distance_metric

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
    past_analogues: xr.Dataset,
    present_analogues: xr.Dataset,
    n_boot: int = 300,
    seed: Union[int, None] = None) -> Tuple[xr.Dataset, xr.Dataset]:
    """
    Compute present-minus-past composite differences from pre-selected
    analogue datasets, for every shared variable, and flag which grid
    points are significant via a bootstrap test.

    Parameters
    ----------
    past_analogues : xr.Dataset
        Dims (time, lat, lon) - the analogue dates for the past period,
        already containing whatever fields you want composited
        (anomalies, raw values, or a mix). Any preprocessing should
        already be applied; this function only composites and
        differences. Group sizes need not be equal.

    present_analogues : xr.Dataset
        Dims (time, lat, lon) - the analogue dates for the present
        period. See `past_analogues`.

    n_boot : int, optional, default = 300
        Number of bootstrap replicates.

    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    Tuple[xr.Dataset, xr.Dataset]
        'differences' : present-minus-past composite difference for
        every variable shared by `past_analogues` and
        `present_analogues`.

        'significant' : same variables/shape as 'differences', boolean:
        True where the observed difference exceeds two standard
        deviations of the bootstrap distribution of differences.
    """
    common_vars = [v for v in past_analogues.data_vars if v in present_analogues.data_vars]
    if not common_vars:
        raise ValueError("past_analogues and present_analogues share no data variables")

    past_analogues = past_analogues[common_vars]
    present_analogues = present_analogues[common_vars]

    if "time" not in past_analogues.dims or "time" not in present_analogues.dims:
        raise ValueError("both past_analogues and present_analogues must have a 'time' dimension")

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
            for var in observed_diff.data_vars
        }
    )

    return observed_diff, significant

###############################################################################
