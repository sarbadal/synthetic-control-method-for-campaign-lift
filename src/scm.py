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

