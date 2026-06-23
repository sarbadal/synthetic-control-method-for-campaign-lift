import pandas as pd
import numpy as np
from numpy.typing import NDArray
from pathlib import Path

type DataResult = tuple[NDArray[np.float64], NDArray[np.float64]]
type CsvDataResult = tuple[NDArray[np.datetime64], NDArray[np.float64], pd.DataFrame]


def make_data(timesteps: int = 30, num_donors: int = 4, campaign_start: int = 20, campaign_lift: float = 25.0, random_seed: int = 42) -> DataResult:
    """Generate synthetic target and donor series for SCM testing.

    Args:
        timesteps: Total number of time steps.
        num_donors: Number of donor units.
        campaign_start: Index where campaign starts.
        campaign_lift: Magnitude of the campaign effect.
        random_seed: Seed for reproducibility.

    Returns:
        A tuple containing:
        - target_data: 1D array of shape ``(timesteps,)``
        - donor_data: 2D array of shape ``(timesteps, num_donors)``
    """
    np.random.seed(random_seed)

    # Generate base trend from shared organic market movements.
    base_trend = np.linspace(100, 150, timesteps) + np.sin(np.linspace(0, 10, timesteps)) * 10

    # Create donor pool data with unit-specific noise.
    donor_data = np.zeros((timesteps, num_donors))
    for i in range(num_donors):
        donor_data[:, i] = base_trend + np.random.normal(0, 4, timesteps)

    # Target composition: 60% donor 0 and 40% donor 1, plus idiosyncratic noise.
    target_data = 0.6 * donor_data[:, 0] + 0.4 * donor_data[:, 1] + np.random.normal(0, 1, timesteps)

    # Inject campaign lift from campaign_start onward.
    target_data[campaign_start:] += campaign_lift

    return target_data, donor_data


def load_product_sales_data(filepath: str | Path = None) -> CsvDataResult:
    """Load product sales data from a CSV file."""
    if filepath is None:
        filepath = Path(__file__).parent.parent / "data" / "product_sales_test_data.csv"
    filepath = Path(filepath) if isinstance(filepath, str) else filepath
    df = pd.read_csv(filepath)
    target_col = ['Premium_Wireless_Audio']
    donor_cols = [
        'Wired_Audio_Baseline', 
        'Smart_Home_Accessories', 
        'Gaming_Peripherals', 
        'Power_Cables_Charging'
    ]
    dates = pd.to_datetime(df['date']).to_numpy(dtype='datetime64[ns]')
    target_data = df[target_col].values
    donor_data = df[donor_cols]
    return dates, target_data, donor_data


def load_media_conversion_data(filepath: str | Path = None) -> CsvDataResult:
    """Load media conversion data from a CSV file."""
    if filepath is None:
        filepath = Path(__file__).parent.parent / "data" / "media_conversions_test_data.csv"
    filepath = Path(filepath) if isinstance(filepath, str) else filepath
    df = pd.read_csv(filepath)
    target_col = ['Paid_Social_Conversions']
    donor_cols = [
        'Organic_Search_Baseline', 
        'Direct_Traffic_Baseline', 
        'Referral_Partner_Baseline', 
        'Email_Newsletter_Baseline'
    ]
    dates = pd.to_datetime(df['date']).to_numpy(dtype='datetime64[ns]')
    target_data = df[target_col].values
    donor_data = df[donor_cols]
    return dates, target_data, donor_data