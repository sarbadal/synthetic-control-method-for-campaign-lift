import numpy as np
from numpy.typing import ArrayLike, NDArray

type LiftResult = tuple[NDArray[np.float64], float, float]


def calculate_campaign_lift(actual: ArrayLike, synthetic: ArrayLike, dates: ArrayLike, campaign_start: str) -> LiftResult:
    """Calculate post-campaign absolute lift and aggregate lift metrics.

    Args:
        actual: 1D array-like of observed target sales, length T.
        synthetic: 1D array-like of synthetic control sales, length T.
        dates: 1D array-like of date values (parseable by NumPy), length T.
        campaign_start: Campaign start date as a string.

    Returns:
        A tuple containing:
        - absolute_lift_per_step: 1D NumPy array for post-campaign lift per step.
        - total_incremental_sales: Sum of post-campaign absolute lift.
        - percentage_lift: Percent lift versus post-campaign synthetic sales, or NaN
          when synthetic post-campaign sum is zero.
    """
    actual_arr = np.asarray(actual, dtype=float)
    synthetic_arr = np.asarray(synthetic, dtype=float)
    dates_arr = np.asarray(dates)

    if actual_arr.ndim != 1 or synthetic_arr.ndim != 1 or dates_arr.ndim != 1:
        raise ValueError("'actual', 'synthetic', and 'dates' must be 1D array-like inputs.")
    if actual_arr.shape[0] != synthetic_arr.shape[0]:
        raise ValueError("'actual' and 'synthetic' must have the same length.")
    if dates_arr.shape[0] != actual_arr.shape[0]:
        raise ValueError("'dates' must have the same length as 'actual' and 'synthetic'.")

    try:
        dates_dt = dates_arr.astype("datetime64[ns]")
    except (TypeError, ValueError) as exc:
        raise ValueError("'dates' must contain parseable date values.") from exc

    try:
        campaign_start_dt = np.datetime64(campaign_start, "ns")
    except (TypeError, ValueError) as exc:
        raise ValueError("'campaign_start' must be a parseable date string.") from exc

    start_matches = dates_dt == campaign_start_dt
    if not np.any(start_matches):
        raise ValueError("'campaign_start' date was not found in 'dates'.")

    campaign_start_idx = int(np.flatnonzero(start_matches)[0])

    post_actual = actual_arr[campaign_start_idx:]
    post_synthetic = synthetic_arr[campaign_start_idx:]

    absolute_lift_per_step: NDArray[np.float64] = post_actual - post_synthetic
    total_incremental_sales = float(np.sum(absolute_lift_per_step))
    total_synthetic_sales = float(np.sum(post_synthetic))

    percentage_lift = float(np.nan)
    if total_synthetic_sales != 0.0:
        percentage_lift = (total_incremental_sales / total_synthetic_sales) * 100.0

    return absolute_lift_per_step, total_incremental_sales, percentage_lift
