### Executive Summary

Synthetic Control Method (SCM) is a rigorous statistical framework. It estimates causal effects in observational studies. Traditional methods rely on a single control group. SCM creates an optimal, weighted combination of multiple unexposed units. This synthetic counterfactual simulates what would have happened without the intervention. 

In campaign evaluation, SCM predicts the baseline sales during the campaign period. Comparing this synthetic baseline to actual sales isolates the true campaign lift.

---

### Methodological Framework

The evaluation workflow splits into three distinct phases. 

#### 1. Pre-Campaign Formulation (Calibration)
*   **Data Aggregation**: Collect historical sales data for the target unit and a "donor pool" of unexposed control units (e.g., different regions, stores, or products).
*   **Weight Optimization**: Determine a vector of weights (W) assigned to the donor units. 
*   **Loss Minimization**: Minimize the Mean Squared Prediction Error (MSPE) between the target unit's actual sales and the synthetic combination during the pre-campaign window.
*   **Constraint Enforcement**: Weights must be non-negative (≥ 0) and sum to 1 to prevent extrapolation outside the support of the data.

#### 2. Campaign Phase (Projection)
*   **Counterfactual Generation**: Apply the frozen pre-campaign weights to the donor pool's actual sales during the campaign period.
*   **Zero-Influence Assumption**: Donor pool units must remain entirely unaffected by the campaign or marketing spillover.
*   **Baseline Isolation**: The resulting time series represents the clean, uninfluenced counterfactual sales trajectory.

#### 3. Post-Analysis (Quantification)
*   **Point-in-Time Treatment Effect**: Subtract the synthetic sales from the actual sales for each time increment during the campaign.
*   **Cumulative Lift**: Sum the point-in-time differences across the entire campaign duration to calculate total incremental volume or revenue.

---

### Mathematical Notation

Let $Y_{it}$ be the observed sales for unit $i$ at time $t$. 

#### Observed Outcomes
$$Y_{it} = Y_{it}^N + \tau_{it} D_{it}$$

*   $Y_{it}^N$: Potential sales without the campaign.
*   $\tau_{it}$: The treatment effect (campaign lift) for unit $i$ at time $t$.
*   $D_{it}$: Binary indicator (1 if unit $i$ is exposed to the campaign at time $t$, 0 otherwise).

#### Target Unit
Assume unit $i=1$ is the target campaign market, exposed at time $t > T_0$ (where $T_0$ is the campaign start date).

#### Synthetic Counterfactual
For $t > T_0$, estimate the unobserved counterfactual $Y_{1t}^N$ using a weighted average of $J$ donor units:

$$\hat{Y}_{1t}^N = \sum_{j=2}^{J+1} w_j Y_{jt}$$

#### Optimization Constraints
The weight vector $W = (w_2, \dots, w_{J+1})'$ is derived by minimizing pre-campaign discrepancy, subject to:
$$\sum_{j=2}^{J+1} w_j = 1 \quad \text{and} \quad w_j \ge 0$$

#### Lift Metrics
*   **Absolute Campaign Lift ($\tau_t$)**: 
    $$\tau_t = Y_{1t} - \hat{Y}_{1t}^N$$
*   **Total Incremental Effect ($\text{Lift}_{\text{total}}$)**: 
    $$\text{Lift}_{\text{total}} = \sum_{t=T_0+1}^{T} (Y_{1t} - \hat{Y}_{1t}^N)$$
*   **Percentage Lift**: 
    $$\%\text{Lift} = \left( \frac{\text{Lift}_{\text{total}}}{\sum_{t=T_0+1}^{T} \hat{Y}_{1t}^N} \right) \times 100$$

---

### Core Structural Advantages

*   **No Extrapolation**: Strict weight constraints prevent the model from projecting unrealistic sales figures outside the bounds of the donor pool.
*   **Time-Varying Confounding**: Unlike standard Difference-in-Differences (DiD), SCM accounts for changing unobserved variables by letting donor impacts evolve over time.
*   **Transparency**: The final weight vector explicitly reveals which control markets or products form the baseline, making the model highly auditable.
*   **Statistical Validation**: Placebo tests (applying the model to unexposed units) offer robust, non-parametric p-values to confirm significance.

## What is the "Donor Pool"?
In the Synthetic Control Method (SCM), the donor pool is a collection of unexposed units (such as regions, cities, stores, or products) that were not treated by the marketing campaign.
Instead of comparing your campaign market to a single, imperfect "control city" (like comparing Mumbai to Delhi), SCM blends multiple units from this donor pool together. The algorithm assigns a specific weight to each donor to create a custom, "synthetic" version of your target market.
------------------------------
## How the Algorithm Uses Donors
The script matches the target market’s historical patterns using the donor units.

```text
Donor 0 (e.g., Region A)  --[ Weight: 0.60 ]--> \
Donor 1 (e.g., Region B)  --[ Weight: 0.40 ]-->  ==>  Synthetic Target Market
Donor 2 (e.g., Region C)  --[ Weight: 0.00 ]--> /
```


   1. Pre-Campaign Calibration: The model looks at the pre-campaign data. It realizes that a combination of 60% of Donor 0 and 40% of Donor 1 perfectly mimics your target market's organic sales curves. It ignores Donor 2 completely by giving it a weight of 0.00.
   2. Post-Campaign Projection: During the campaign, the target market receives marketing spend, but the donor pool does not. The model takes the ongoing, real-time sales from Donor 0 and Donor 1, applies the same 0.60 and 0.40 weights, and constructs the baseline counterfactual (what would have happened).

------------------------------
## Strict Selection Rules for Your Donors
To get an accurate lift calculation, your donor pool units must meet three strict technical criteria:

* Total Isolation: Donors must have zero exposure to the campaign. If there is marketing spillover (e.g., a TV ad from the target market bleeds into a donor city's broadcast), the donor's sales will artificially rise, causing the model to underestimate your true campaign lift.
* Driven by the Same Macro Factors: The donors must share the same macroeconomic environment as the target market. If an unmapped economic shock (like a local lockdown or extreme weather) hits a donor but not the target market, the synthetic baseline will break down.
* No Extrapolation (The Convex Hull Rule): SCM restricts weights to be non-negative and sum to 1. This means the target market's sales must fall within the range of the donor pool's data. If your target market sells 10,000 units a week, and all your donors only sell 2,000 units a week, the model cannot properly scale up to match it.
