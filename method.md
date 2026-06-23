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