# Interrupted Time Series Alternative (No Donor Markets)

When donor markets are unavailable, you can replace cross-sectional Synthetic Control with an Interrupted Time Series (ITS) counterfactual approach.

In this setup, the target series serves as its own control: the model is trained on pre-campaign history, forecasts an organic baseline into the campaign window, and attributes the gap between observed and forecast values to campaign impact.

## Workflow

```text
[Pre-Campaign Period] --> Train univariate baseline model
    |
[Campaign Active Window] --> Lift_t = Actual_t - Baseline_t
    |
[Post-Campaign Decay Window] --> Apply carryover decay for N periods
```

## Model Components

1. Baseline generation
Fit a univariate forecasting model on data from $t = 0$ to $T_{start} - 1$.
Good lightweight choices include:
    - Holt-Winters Exponential Smoothing
    - Deterministic trend + seasonality regression

2. Campaign-period lift
For each time step during the active campaign window $[T_{start}, T_{end}]$:

$$ Lift_t = Actual_t - \hat{Y}^{baseline}_t $$

3. Post-campaign decay (adstock)
Campaign effects may persist after $T_{end}$. Represent this carryover with a geometric decay factor $\lambda$ where $0 < \lambda < 1$.

$$ Carryover_{t+1} = \lambda \cdot Carryover_t $$

Apply decay for $N$ periods, then force remaining effect to zero so the observed series converges back to the organic baseline.

## Practical Notes

- This approach is useful when no valid untreated donor pool exists.
- Baseline quality depends strongly on stable pre-campaign history and correctly modeled seasonality.
- Structural breaks unrelated to campaign activity can bias lift estimates and should be checked before interpretation.