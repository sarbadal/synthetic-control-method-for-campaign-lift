from numpy.typing import ArrayLike
import numpy as np
import pandas as pd


def format_results(actual: ArrayLike, synthetic: ArrayLike, campaign_start: int) -> pd.DataFrame:
    """
    Formats results into a DataFrame for reporting.
    
    actual: 1D array-like of actual target sales (length T)
    synthetic: 1D array-like of synthetic control sales (length T)
    campaign_start: int, index where campaign starts
    """
    actual_arr = np.asarray(actual)
    synthetic_arr = np.asarray(synthetic)

    if actual_arr.shape != synthetic_arr.shape:
        raise ValueError("'actual' and 'synthetic' must have the same shape.")

    df = pd.DataFrame(
        {
            "Actual_Sales": actual_arr,
            "Synthetic_Sales": synthetic_arr,
            "Absolute_Lift": actual_arr - synthetic_arr,
        }
    )
    # Zero out pre-treatment lift for reporting.
    df.loc[: campaign_start - 1, "Absolute_Lift"] = 0
    return df