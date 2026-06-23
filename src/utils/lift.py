import pandas as pd
import numpy as np
from dataclasses import dataclass
from numpy.typing import ArrayLike, NDArray

from ..usb import UnivariateSyntheticBaseline

type LiftResult = tuple[NDArray[np.float64], float, float]


@dataclass
class LiftEvaluator:
    """Parameters for lift evaluation."""
    mode: str = "standard"
    seasonal_period: int = 7
    decay_periods: int = 14
    decay_rate: float = 0.75
    actual: ArrayLike | None = None
    synthetic: ArrayLike | None = None
    dates: ArrayLike | None = None
    campaign_start: str | None = None
    df: pd.DataFrame | None = None
    date_col: str | None = None
    kpi_col: str | None = None
    start_date: str | None = None
    end_date: str | None = None

    def __post_init__(self):
        self.mode = self.mode.strip().lower()
        if self.mode not in {"standard", "decay"}:
            raise ValueError("'mode' must be either 'standard' or 'decay'.")
        self.df = self.df.copy() if self.df is not None else None

    def _calculate_campaign_lift(self) -> LiftResult:
        """Calculate post-campaign absolute lift and aggregate lift metrics."""
        actual_arr = np.asarray(self.actual, dtype=float)
        synthetic_arr = np.asarray(self.synthetic, dtype=float)
        dates_arr = np.asarray(self.dates)

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
            campaign_start_dt = np.datetime64(self.campaign_start, "ns")
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

    def _evaluate_campaign_with_decay(self) -> LiftResult:
        """Evaluate campaign lift with residual decay and return aggregate LiftResult metrics."""
        df = self.df.sort_values(by=self.date_col).reset_index(drop=True)

        start_matches = df.index[df[self.date_col] == self.start_date]
        if start_matches.empty:
            raise ValueError("'start_date' date was not found in the provided DataFrame.")

        end_matches = df.index[df[self.date_col] == self.end_date]
        if end_matches.empty:
            raise ValueError("'end_date' date was not found in the provided DataFrame.")

        start_idx = int(start_matches[0])
        end_idx = int(end_matches[0])
        if end_idx < start_idx:
            raise ValueError("'end_date' must be on or after 'start_date'.")

        total_len = len(df)
        if start_idx == 0:
            raise ValueError("At least one pre-campaign data point is required before 'start_date'.")

        pre_kpi = df.loc[: start_idx - 1, self.kpi_col].values
        model = UnivariateSyntheticBaseline(seasonal_period=self.seasonal_period)
        model.fit(pre_kpi)

        full_baseline = model.predict(0, total_len - 1)
        df["Synthetic_Baseline"] = full_baseline

        actual_kpi = df[self.kpi_col].values
        calculated_lift = np.zeros(total_len, dtype=float)

        for t in range(start_idx, end_idx + 1):
            calculated_lift[t] = actual_kpi[t] - full_baseline[t]

        last_active_lift = calculated_lift[end_idx]
        for step in range(1, self.decay_periods + 1):
            t = end_idx + step
            if t >= total_len:
                break

            modeled_decay_lift = last_active_lift * (self.decay_rate ** step)
            observed_diff = actual_kpi[t] - full_baseline[t]
            calculated_lift[t] = min(modeled_decay_lift, max(0.0, observed_diff))

        df["Calculated_Lift"] = calculated_lift

        impact_end_idx = min(end_idx + self.decay_periods, total_len - 1)
        absolute_lift_per_step: NDArray[np.float64] = calculated_lift[start_idx : impact_end_idx + 1]

        total_incremental_sales = float(np.sum(absolute_lift_per_step))
        baseline_during_impact = float(np.sum(full_baseline[start_idx : impact_end_idx + 1]))
        percentage_lift = float(np.nan)
        if baseline_during_impact != 0.0:
            percentage_lift = (total_incremental_sales / baseline_during_impact) * 100.0

        return absolute_lift_per_step, total_incremental_sales, percentage_lift

    def calculate_lift(self) -> LiftResult:
        """Public method to calculate campaign lift based on the selected mode."""
        if self.mode == "standard":
            return self._calculate_campaign_lift()
        if self.mode == "decay":
            return self._evaluate_campaign_with_decay()
        raise ValueError("'mode' must be either 'standard' or 'decay'.")

