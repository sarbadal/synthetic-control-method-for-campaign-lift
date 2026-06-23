import numpy as np
import pandas as pd
from scipy.optimize import minimize


class SyntheticControl:
    def __init__(self):
        self.weights = None
        self.loss = None
        
    def _loss_function(self, weights, X_donors, y_target):
        """Calculates Mean Squared Prediction Error (MSPE)."""
        prediction = np.dot(X_donors, weights)
        return np.mean((y_target - prediction) ** 2)

    def fit(self, pre_target, pre_donors):
        """
        Optimizes weights using pre-campaign data.
        
        pre_target: 1D array-like of target unit sales (length T_pre)
        pre_donors: 2D array-like of donor pool sales (shape T_pre x J)
        """
        X = np.array(pre_donors)
        y = np.array(pre_target)
        num_donors = X.shape[1]
        
        # Initial guess: equal weights
        initial_weights = np.ones(num_donors) / num_donors
        
        # Constraint 1: Weights must sum to 1
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        
        # Constraint 2: Weights must be non-negative (0 <= w <= 1)
        bounds = [(0.0, 1.0) for _ in range(num_donors)]
        
        # Optimize
        result = minimize(
            fun=self._loss_function,
            x0=initial_weights,
            args=(X, y),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if not result.success:
            raise ValueError(f"Optimization failed: {result.message}")
            
        self.weights = result.x
        self.loss = result.fun
        return self

    def predict(self, donors):
        """Generates the synthetic counterfactual time series."""
        if self.weights is None:
            raise ValueError("Model must be fitted before making predictions.")
        return np.dot(np.array(donors), self.weights)


def calculate_campaign_lift(actual, synthetic, campaign_start):
    """
    Calculates absolute and percentage lift post-campaign.
    
    actual: 1D array-like of actual target sales (length T)
    synthetic: 1D array-like of synthetic control sales (length T)
    campaign_start: int, index where campaign starts
    """
    post_actual = actual[campaign_start:]
    post_synthetic = synthetic[campaign_start:]
    
    absolute_lift_per_step = post_actual - post_synthetic
    total_incremental_sales = np.sum(absolute_lift_per_step)
    total_synthetic_sales = np.sum(post_synthetic)
    percentage_lift = np.nan
    if total_synthetic_sales != 0:
        percentage_lift = (total_incremental_sales / total_synthetic_sales) * 100
        
    return absolute_lift_per_step, total_incremental_sales, percentage_lift


def format_results(actual, synthetic, campaign_start):
    """
    Formats results into a DataFrame for reporting.
    
    actual: 1D array-like of actual target sales (length T)
    synthetic: 1D array-like of synthetic control sales (length T)
    campaign_start: int, index where campaign starts
    """
    df = pd.DataFrame({
        'Actual_Sales': actual,
        'Synthetic_Sales': synthetic,
        'Absolute_Lift': actual - synthetic
    })
    df.loc[:campaign_start-1, 'Absolute_Lift'] = 0 # Zero out pre-treatment lift for reporting
    return df


def make_data(timesteps=30, num_donors=4, campaign_start=20, campaign_lift=25, random_seed=42):
    """
    Generates synthetic data for testing the SCM implementation.
    
    timesteps: int, total number of time steps
    num_donors: int, number of donor units
    campaign_start: int, index where campaign starts
    campaign_lift: float, magnitude of the campaign effect
    random_seed: int, seed for reproducibility
    """
    np.random.seed(random_seed)
    
    # Generate Base Trends (Shared organic market movements)
    base_trend = np.linspace(100, 150, timesteps) + np.sin(np.linspace(0, 10, timesteps)) * 10
    
    # Create Donor Pool Data (With unique variations)
    donor_data = np.zeros((timesteps, num_donors))
    for i in range(num_donors):
        donor_data[:, i] = base_trend + np.random.normal(0, 4, timesteps)
        
    # Create Target Unit Data
    # True composition: 60% Donor 0, 40% Donor 1
    target_data = 0.6 * donor_data[:, 0] + 0.4 * donor_data[:, 1] + np.random.normal(0, 1, timesteps)
    
    # Inject a Campaign Effect (Lift) post-campaign_start
    target_data[campaign_start:] += campaign_lift
    
    return target_data, donor_data


def run_scm_pipeline(target_data, donor_data, campaign_start):
    """
    Runs the full SCM pipeline: fitting, prediction, and lift calculation.
    
    target_data: 1D array-like of actual target sales (length T)
    donor_data: 2D array-like of donor pool sales (shape T x J)
    campaign_start: int, index where campaign starts
    """
    # Segment Data into Pre and Post Campaign Windows
    pre_target = target_data[:campaign_start]
    pre_donors = donor_data[:campaign_start, :]
    
    # Fit SCM Model
    scm = SyntheticControl()
    scm.fit(pre_target, pre_donors)
    
    # Generate Synthetic Counterfactual for all timelines
    synthetic_sales = scm.predict(donor_data)
    
    # Calculate Campaign Lift Metrics (Post-Campaign Window)
    absolute_lift_per_step, total_incremental_sales, percentage_lift = calculate_campaign_lift(target_data, synthetic_sales, campaign_start)
    
    # Format Results Output
    df_results = format_results(target_data, synthetic_sales, campaign_start)
    
    return scm.weights, scm.loss, total_incremental_sales, percentage_lift, df_results


def main():
    # 1. Setup Simulation Parameters
    timesteps = 30
    campaign_start = 20
    num_donors = 4
    campaign_lift = 25
    
    # 2. Generate Synthetic Data
    target_data, donor_data = make_data(timesteps, num_donors, campaign_start, campaign_lift)
    
    # 3. Run SCM Pipeline
    weights, loss, total_incremental_sales, percentage_lift, df_results = run_scm_pipeline(
        target_data, donor_data, campaign_start
    )
    
    # 4. Display Results
    print("--- Optimized Donor Weights ---")
    for idx, w in enumerate(weights):
        print(f"Donor {idx}: {w:.4f}")
        
    print("\n--- Campaign Evaluation Summary ---")
    print(f"Pre-Campaign MSPE Fit Loss : {loss:.4f}")
    print(f"Total Incremental Sales    : {total_incremental_sales:.2f} units")
    print(f"Calculated Percentage Lift : {percentage_lift:.2f}%")
    
    print("\n--- Detailed Results DataFrame ---")
    print(df_results)


if __name__ == "__main__":
    main()
