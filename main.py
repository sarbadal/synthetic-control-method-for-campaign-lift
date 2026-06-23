from src.pipeline import CampaignLiftCalculator
from src.utils.data import load_media_conversion_data, load_product_sales_data


def main():
    dates, target_data, donor_data = load_product_sales_data()
    campaign_start = "2026-01-18"

    lift_calculator = CampaignLiftCalculator(
        dates=dates,
        target_data=target_data, 
        donor_data=donor_data, 
        campaign_start=campaign_start
    )

    lift_calculator.display_results()


if __name__ == "__main__":
    main()

