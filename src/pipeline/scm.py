from typing import Protocol

import numpy as np

from ..scm import SyntheticControl
from ..utils.lift import LiftEvaluator
from ..utils.format_results import format_results


class Control(Protocol):
    def fit(self, target: list[float], donors: list[list[float]]) -> None: ...

    def predict(self, donors: list[list[float]]) -> list[float]: ...


class CampaignLiftCalculator:
    def __init__(self, target_data: list[float], donor_data: list[list[float]], campaign_start: str, dates: list[str], control_model: Control | None = None):
        self.target_data = target_data
        self.donor_data = donor_data
        self.campaign_start = campaign_start
        self.dates = dates
        self.control_model = control_model or SyntheticControl()
        self._results_cache = None  # Cache for results to avoid recomputation

    def _donor_display_names(self, donor_count: int) -> list[str]:
        donor_columns = getattr(self.donor_data, "columns", None)
        if donor_columns is not None:
            donor_names = [str(name) for name in donor_columns]
            if len(donor_names) != donor_count:
                raise ValueError("'donor_data' columns length must match number of donor series.")
            return donor_names

        return [f"Donor {i + 1}" for i in range(donor_count)]

    def _campaign_start_index(self) -> int:
        dates_arr = np.asarray(self.dates)
        if dates_arr.ndim != 1:
            raise ValueError("'dates' must be a 1D array-like input.")

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

        return int(np.flatnonzero(start_matches)[0])

    def run(self):
        if self._results_cache is not None:
            return self._results_cache

        campaign_start_idx = self._campaign_start_index()

        target_arr = np.asarray(self.target_data, dtype=float).reshape(-1)
        donor_arr = np.asarray(self.donor_data, dtype=float)

        if donor_arr.ndim != 2:
            raise ValueError("'donor_data' must be 2D array-like with shape (T, J).")
        if target_arr.shape[0] != donor_arr.shape[0]:
            raise ValueError("'target_data' and 'donor_data' must have the same number of rows.")

        # Segment Data into Pre and Post Campaign Windows
        pre_target = target_arr[:campaign_start_idx]
        pre_donors = donor_arr[:campaign_start_idx, :]

        # Fit SCM Model
        self.control_model.fit(pre_target, pre_donors)

        # Generate Synthetic Counterfactual for all timelines
        synthetic_sales = self.control_model.predict(donor_arr)

        # Calculate Campaign Lift Metrics (Post-Campaign Window)
        evaluator = LiftEvaluator(
            mode="standard",
            actual=target_arr,
            synthetic=synthetic_sales,
            dates=self.dates,
            campaign_start=self.campaign_start
        )
        absolute_lift_per_step, total_incremental_sales, percentage_lift = evaluator.calculate_lift()

        # Format Results Output
        df_results = format_results(target_arr, synthetic_sales, campaign_start_idx)

        self._results_cache = (
            self.control_model.weights, 
            self.control_model.loss, 
            total_incremental_sales, 
            percentage_lift, 
            df_results
        )
        return self._results_cache

    def display_results(self) -> None:
        weights, loss, total_incremental_sales, percentage_lift, df_results = self.run()
        donor_names = self._donor_display_names(len(weights))

        print("--- Optimized Donor Weights ---")
        for i, weight in enumerate(weights):
            donor_label = f"Donor {i + 1}"
            donor_name = donor_names[i]
            print(f"{donor_label} ({donor_name}): {weight:.4f}")

        print("\n--- Campaign Evaluation Summary ---")
        print(f"Pre-Campaign MSPE Fit Loss : {loss:.4f}")
        print(f"Total Incremental Sales    : {total_incremental_sales:.2f} units")
        print(f"Calculated Percentage Lift : {percentage_lift:.2f}%")
        
        print("\n--- Detailed Results ---")
        print(df_results)