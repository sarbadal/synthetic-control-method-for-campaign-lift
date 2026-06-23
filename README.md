# Synthetic Control Method for Campaign Lift

This project estimates campaign lift using a Synthetic Control Method (SCM) pipeline.
It builds a weighted combination of untreated donor series to approximate the target
series before campaign start, then uses that synthetic baseline to estimate post-campaign
incremental impact.

## What This Repository Does

- Loads target and donor time-series data from CSV files.
- Fits constrained donor weights (non-negative, sum to 1) on the pre-campaign period.
- Predicts a synthetic counterfactual across the full timeline.
- Calculates post-campaign incremental lift metrics.
- Prints optimized weights, fit loss, and a detailed row-level results table.

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
	au_t = Y_{1t} - \hat{Y}_{1t}^N
$$

Total incremental lift and percent lift are then aggregated over the campaign period.

## Project Layout

```
main.py
method.md
requirements.txt
src/
  pipeline.py
  scm.py
  data/
    product_sales_test_data.csv
    media_conversions_test_data.csv
    sales_test_data.csv
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

`main.py` currently uses:

- `load_product_sales_data()` from `src/utils/data.py`
- campaign start date: `2026-01-18`

## Data Interface

The pipeline expects:

- `dates`: 1D date-like sequence parseable to `datetime64[ns]`
- `target_data`: 1D target values (or shape that can be flattened to 1D)
- `donor_data`: 2D donor matrix with shape `(T, J)`
- `campaign_start`: date string that must exist in `dates`

Validation in the code ensures dimensional consistency and campaign date presence.

## Core Components

### `SyntheticControl` (`src/scm.py`)

- Optimizer: `scipy.optimize.minimize` with `SLSQP`
- Objective: pre-campaign MSPE
- Constraints: convex combination (sum-to-1, bounded [0, 1])
- Outputs: `weights` and optimization `loss`

### `CampaignLiftCalculator` (`src/pipeline.py`)

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
	- `Absolute_Lift` (set to 0 before campaign start for reporting)

## Printed Output

Running `python main.py` prints:

- optimized donor weights (with donor column names when available)
- pre-campaign MSPE fit loss
- total incremental sales
- calculated percentage lift
- full detailed results table

## Switching to Media Conversion Example

To run the media conversion test dataset, update `main.py`:

1. Replace `load_product_sales_data` with `load_media_conversion_data`.
2. Ensure `campaign_start` matches a date available in that dataset.

## Assumptions and Good Practice

- Donors must be untreated by the campaign (no spillover).
- Donors should share the same macro environment as the target.
- Target pre-period behavior should be representable by a convex combination of donors.
- Better pre-campaign fit generally improves counterfactual reliability.

## Notes

- `sales_test_data.csv` is present in `src/data/` but not used by `main.py` by default.
- The current implementation prints tabular results to console; it does not generate plots.
