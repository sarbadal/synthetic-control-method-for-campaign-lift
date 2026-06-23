from typing import Protocol

import numpy as np
import pandas as pd

from ..usb import UnivariateSyntheticBaseline
from ..utils.lift import LiftEvaluator
from ..utils.format_results import format_results


class Control(Protocol):
    def fit(self, target: list[float], donors: list[list[float]]) -> None: ...

    def predict(self, donors: list[list[float]]) -> list[float]: ...


class UnivariateCampaignLiftCalculator:
    def __init__(self, df: pd.DataFrame, date_col: str, kpi_col: str, campaign_start: str, control_model: Control | None = None):
        self.df = df.copy()
        self.date_col = date_col
        self.kpi_col = kpi_col
        self.campaign_start = campaign_start
        self.control_model = control_model or UnivariateSyntheticBaseline()
        self._results_cache = None  # Cache for results to avoid recomputation

    def _campaign_start_index(self) -> int:
        dates_arr = np.asarray(self.df[self.date_col])
        if dates_arr.ndim != 1:
            raise ValueError(f"'{self.date_col}' must be a 1D array-like input.")

        try:
            dates_dt = pd.to_datetime(dates_arr)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"'{self.date_col}' must contain parseable date values.") from exc

        try:
            campaign_start_dt = pd.to_datetime(self.campaign_start)
        except (TypeError, ValueError) as exc:
            raise ValueError("'campaign_start' must be a parseable date string.") from exc

        # First try exact timestamp matching, then day-level matching as a fallback.
        start_matches = dates_dt == campaign_start_dt
        if not np.any(start_matches):
            start_matches = dates_dt.normalize() == campaign_start_dt.normalize()

        if not np.any(start_matches):
            min_date = dates_dt.min()
            max_date = dates_dt.max()
            raise ValueError(
                "'campaign_start' date was not found in the provided DataFrame. "
                f"Provided: {campaign_start_dt.date()}. Available date range: "
                f"{min_date.date()} to {max_date.date()}."
            )

        return int(np.flatnonzero(start_matches)[0])

    def run(self):
        if self._results_cache is not None:
            return self._results_cache

        start_idx = self._campaign_start_index()
        pre_campaign_data = self.df.iloc[:start_idx]
        post_campaign_data = self.df.iloc[start_idx:]

        # Fit the control model on pre-campaign data
        self.control_model.fit(pre_campaign_data[self.kpi_col].tolist())

        # Predict the synthetic baseline for the post-campaign period
        synthetic_baseline = self.control_model.predict(
            start_step=start_idx,
            end_step=len(self.df) - 1
        )

        # Evaluate lift using LiftEvaluator
        lift_evaluator = LiftEvaluator(
            mode="decay",
            seasonal_period=7,
            decay_periods=14,
            decay_rate=0.75,
            df=self.df,
            date_col=self.date_col,
            kpi_col=self.kpi_col,
            start_date=self.df[self.date_col].iloc[start_idx],
            end_date=self.df[self.date_col].iloc[-1],
        )

        absolute_lift_per_step, total_incremental_sales, percentage_lift = lift_evaluator.calculate_lift()

        # Format results for display
        results = (absolute_lift_per_step, total_incremental_sales, percentage_lift)

        self._results_cache = results
        return results

    def display_results(self) -> None:
        absolute_lift_per_step, total_incremental_sales, percentage_lift = self.run()

        print("--- Stepwise Lift ---")
        for i, step in enumerate(absolute_lift_per_step):
            print(f"Step {i + 1}: {step:.4f}")

        print("\n--- Campaign Evaluation Summary ---")
        print(f"Total Incremental Sales    : {total_incremental_sales:.2f}")
        print(f"Calculated Percentage Lift : {percentage_lift:.2f}%")
        