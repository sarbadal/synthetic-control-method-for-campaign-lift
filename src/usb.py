import numpy as np
import pandas as pd

class UnivariateSyntheticBaseline:
    def __init__(self, seasonal_period=7):
        self.seasonal_period = seasonal_period
        self.beta_trend = None
        self.beta_seasonal = None
        self.intercept = None

    def fit(self, pre_kpi):
        """Fits a deterministic Trend + Seasonality model on historical data."""
        y = np.array(pre_kpi)
        n = len(y)
        
        # Create structural time features
        time_steps = np.arange(n)
        seasons = np.array([t % self.seasonal_period for t in time_steps])
        
        # One-hot encode seasons (dropping first to avoid dummy variable trap)
        season_dummies = np.zeros((n, self.seasonal_period - 1))
        for i in range(1, self.seasonal_period):
            season_dummies[seasons == i, i - 1] = 1
            
        # Design Matrix X: [Intercept, Trend, Seasonal Dummies]
        X = np.hstack([np.ones((n, 1)), time_steps.reshape(-1, 1), season_dummies])
        
        # Ordinary Least Squares (OLS) closed form solution: (X'X)^(-1) X'y
        beta = np.linalg.inv(X.T @ X) @ X.T @ y
        
        self.intercept = beta[0]
        self.beta_trend = beta[1]
        self.beta_seasonal = beta[2:]
        return self

    def predict(self, start_step, end_step):
        """Generates the organic synthetic baseline out-of-sample forecast."""
        predictions = []
        for t in range(start_step, end_step + 1):
            season_idx = t % self.seasonal_period
            
            # Calculate seasonal component assignment
            if season_idx == 0:
                seasonal_val = 0
            else:
                seasonal_val = self.beta_seasonal[season_idx - 1]
                
            pred = self.intercept + (self.beta_trend * t) + seasonal_val
            predictions.append(pred)
        return np.array(predictions)


def evaluate_campaign_with_decay(df, date_col, kpi_col, start_date, end_date, decay_periods=14, decay_rate=0.75):
    """
    Parses time series, builds synthetic baseline, and evaluates lift accounting for decay.
    """
    df = df.sort_values(by=date_col).reset_index(drop=True)
    
    # Get index markers
    start_idx = df[df[date_col] == start_date].index[0]
    end_idx = df[df[date_col] == end_date].index[0]
    total_len = len(df)
    
    # 1. Fit synthetic baseline engine on completely clean historical data
    pre_kpi = df.loc[:start_idx - 1, kpi_col].values
    model = UnivariateSyntheticBaseline(seasonal_period=7)
    model.fit(pre_kpi)
    
    # 2. Project baseline for the full evaluation horizon (Campaign + Post-Campaign windows)
    full_baseline = model.predict(0, total_len - 1)
    df['Synthetic_Baseline'] = full_baseline
    
    # Initialize evaluation tracking arrays
    actual_kpi = df[kpi_col].values
    calculated_lift = np.zeros(total_len)
    
    # 3. Process Active Campaign Window
    for t in range(start_idx, end_idx + 1):
        calculated_lift[t] = actual_kpi[t] - full_baseline[t]
        
    # 4. Process Diminishing Post-Campaign Window (N periods)
    last_active_lift = calculated_lift[end_idx]
    
    for step in range(1, decay_periods + 1):
        t = end_idx + step
        if t >= total_len:
            break
            
    # Geometric Diminishing Effect formula
        modeled_decay_lift = last_active_lift * (decay_rate ** step)
        
        # Edge check: ensure actual data hasn't already tanked below baseline
        observed_diff = actual_kpi[t] - full_baseline[t]
        calculated_lift[t] = min(modeled_decay_lift, max(0, observed_diff))
        
    df['Calculated_Lift'] = calculated_lift
    
    # 5. Aggregate Financial Metrics
    active_lift_total = np.sum(calculated_lift[start_idx:end_idx + 1])
    decay_lift_total = np.sum(calculated_lift[end_idx + 1: end_idx + decay_periods + 1])
    total_incremental_lift = active_lift_total + decay_lift_total
    
    baseline_during_impact = np.sum(full_baseline[start_idx: end_idx + decay_periods + 1])
    percentage_lift = (total_incremental_lift / baseline_during_impact) * 100
    
    print("--- Campaign Multi-Phase Analytics ---")
    print(f"Active Phase Incremental Lift : {active_lift_total:.2f}")
    print(f"Post-Phase Decaying Residual  : {decay_lift_total:.2f}")
    print(f"Total Combined Campaign Lift  : {total_incremental_lift:.2f}")
    print(f"Combined Percentage Lift      : {percentage_lift:.2f}%")
    
    return df

# ==========================================
# Run Simulation Pipeline
# ==========================================
if __name__ == "__main__":
    # Generate 120 days of simulated sales data
    np.random.seed(42)
    dates = pd.date_range(start="2026-01-01", periods=120)
    
    # Base Organic Metrics: Trend + Weekly Seasonality + Random Noise
    base_trend = 500 + (np.arange(120) * 1.5)
    weekly_pattern = np.array([20, -40, -10, 30, 50, 10, -60]) # High weekends, low Mon/Tue
    seasonal_component = np.array([weekly_pattern[d.weekday()] for d in dates])
    noise = np.random.normal(0, 8, 120)
    
    sales_kpi = base_trend + seasonal_component + noise
    
    # Inject Active Campaign Lift (Days 80 to 95)
    # Day 80 is 2026-03-22, Day 95 is 2026-04-06
    sales_kpi[80:96] += 120  # Immediate flat injection spike
    
    # Inject Declining Residual Action (Days 96 onwards)
    last_lift = 120
    for idx, t in enumerate(range(96, 120)):
        decay_factor = last_lift * (0.80 ** (idx + 1))
        if decay_factor < 2: decay_factor = 0
        sales_kpi[t] += decay_factor
        
    df_sales = pd.DataFrame({'Date': dates, 'Sales_KPI': sales_kpi})
    
    # Execute Framework Calculation Pipeline
    df_evaluated = evaluate_campaign_with_decay(
        df=df_sales,
        date_col='Date',
        kpi_col='Sales_KPI',
        start_date=pd.Timestamp('2026-03-22'),
        end_date=pd.Timestamp('2026-04-06'),
        decay_periods=14,   # Look for trailing impacts for 14 days post campaign
        decay_rate=0.80     # Diminish impact by 20% every single day
    )
