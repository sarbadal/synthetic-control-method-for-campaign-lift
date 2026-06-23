# Synthetic Control Method for Campaign Lift

This project estimates campaign lift using two pipelines:

- Multivariate Synthetic Control Method (SCM) using donor time series.
- Univariate synthetic baseline modeling using trend and seasonality.

Both pipelines estimate a no-campaign counterfactual and compare it with observed post-campaign outcomes to measure incremental impact.

## What This Repository Does

- Loads campaign test data from CSV files.
- Runs multivariate SCM with constrained donor weights.
- Runs univariate baseline modeling with trend + seasonal structure.
- Calculates post-campaign lift metrics (stepwise, total, percentage).
- Prints evaluation summaries to the console.

## Method Summary

Given observed sales or conversions \(Y_{it}\):

$$
Y_{it} = Y_{it}^N + \tau_{it} D_{it}
$$

For treated unit \(i=1\), post-campaign counterfactual is estimated by:

$$
\hat{Y}_{1t}^N = \sum_{j=2}^{J+1} w_j Y_{jt}
$$

Weights are found by minimizing pre-campaign mean squared prediction error (MSPE),
subject to:

$$
\sum_j w_j = 1, \quad w_j \ge 0
$$

Post-campaign lift at each step:

$$
tau_t = Y_{1t} - \hat{Y}_{1t}^N
$$

Total incremental lift and percent lift are then aggregated over the campaign period.

For the univariate workflow, the synthetic baseline is built from historical structure in a single KPI series, then projected into the campaign/post-campaign window.

## Project Layout

```
main.py
classic_cross_sectional.md
method.md
LICENSE
requirements.txt
src/
  scm.py
  usb.py
  data/
    product_sales_test_data.csv
    media_conversions_test_data.csv
    sales_test_data.csv
    univariate_test_data.csv
  pipeline/
    scm.py
    usb.py
  utils/
    data.py
    lift.py
    format_results.py
```

## Installation

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

Default run:

```bash
python main.py
```

`main.py` contains two entry paths:

- `main()` for multivariate SCM (`CampaignLiftCalculator`).
- `main_univariate()` for univariate baseline modeling (`UnivariateCampaignLiftCalculator`).

The currently enabled path is whichever function is called under `if __name__ == "__main__":`.

Current sample defaults:

- Multivariate: product sales data with campaign start `2026-01-18`.
- Univariate: media conversions KPI with campaign start `2026-03-22`.

## Data Interface

### Multivariate (`CampaignLiftCalculator`)

- `dates`: 1D date-like sequence parseable to `datetime64[ns]`
- `target_data`: 1D target values (or shape that can be flattened to 1D)
- `donor_data`: 2D donor matrix with shape `(T, J)`
- `campaign_start`: date string that must exist in `dates`

### Univariate (`UnivariateCampaignLiftCalculator`)

- `df`: pandas DataFrame with at least a date column and a KPI column
- `date_col`: date column name in `df`
- `kpi_col`: target KPI column name in `df`
- `campaign_start`: date string that must exist in `df[date_col]`

Important:

- `load_media_conversion_data()` returns `(dates, target_data, donor_data)`, not a DataFrame.
- `load_sample_univariate_data()` returns a DataFrame and is directly compatible with `UnivariateCampaignLiftCalculator`.

Validation in the code checks shape consistency and campaign date presence.

## Core Components

### `SyntheticControl` (`src/scm.py`)

- Optimizer: `scipy.optimize.minimize` with `SLSQP`
- Objective: pre-campaign MSPE
- Constraints: convex combination (sum-to-1, bounded `[0, 1]`)
- Outputs: `weights` and optimization `loss`

### `CampaignLiftCalculator` (`src/pipeline/scm.py`)

- Splits target/donors into pre- and post-campaign windows
- Fits SCM on pre-campaign data
- Predicts synthetic series for all time steps
- Computes:
  - absolute post-campaign lift per step
  - total incremental sales/conversions
  - percentage lift vs synthetic baseline
- Formats a report table with:
  - `Actual_Sales`
  - `Synthetic_Sales`
  - `Absolute_Lift` (set to `0` before campaign start for reporting)

### `UnivariateSyntheticBaseline` (`src/usb.py`)

- Fits deterministic trend + seasonality from pre-campaign KPI history
- Uses OLS closed-form estimation on a design matrix with intercept, trend, and seasonal dummies
- Projects an organic synthetic baseline forward for post-campaign evaluation

### `UnivariateCampaignLiftCalculator` (`src/pipeline/usb.py`)

- Accepts DataFrame input and identifies campaign start by date match
- Fits `UnivariateSyntheticBaseline` on pre-campaign KPI values
- Predicts synthetic baseline for campaign/post-campaign periods
- Uses `LiftEvaluator` in `decay` mode to compute:
  - stepwise absolute lift
  - total incremental lift
  - percentage lift

## Printed Output

Running `python main.py` prints output based on the selected entry path:

- Multivariate path:
  - optimized donor weights
  - pre-campaign MSPE fit loss
  - total incremental sales
  - calculated percentage lift
  - detailed results table
- Univariate path:
  - stepwise lift by period
  - total incremental sales
  - calculated percentage lift

## Switching Between Examples

1. Open `main.py`.
2. Choose either `main()` or `main_univariate()` in the `__main__` block.
3. Ensure `campaign_start` exists in the selected dataset.

### Univariate Data Choices

- Use `load_sample_univariate_data()` when you want a direct DataFrame input (`date`, `sales`).
- If using `load_media_conversion_data()`, convert its tuple output into a DataFrame before initializing `UnivariateCampaignLiftCalculator`.

## Assumptions and Good Practice

- Donors should be untreated by the campaign (no spillover) for multivariate SCM.
- Donors should share macro dynamics with the treated target.
- The pre-campaign period should be long enough to learn stable structure.
- Better pre-campaign fit improves counterfactual reliability.
- Verify that campaign start exists exactly in the date index.

## Notes

- The implementation prints tabular and summary outputs to console.
- No built-in plotting is currently included.
