from src.pipeline.scm import CampaignLiftCalculator
from src.pipeline.usb import UnivariateCampaignLiftCalculator
from src.utils.data import load_media_conversion_data, load_product_sales_data, load_sample_univariate_data


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


def main_univariate():
    df = load_sample_univariate_data()
    date_col = "date"
    kpi_col = "sales"
    campaign_start = "2026-03-22"

    lift_calculator = UnivariateCampaignLiftCalculator(
        df=df,
        date_col=date_col,
        kpi_col=kpi_col,
        campaign_start=campaign_start
    )

    lift_calculator.display_results()


if __name__ == "__main__":
    # main()
    main_univariate()

