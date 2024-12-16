import json
import logging

import openai
import pandas as pd

from src.AssetAnalyzer.asset_analyzer import AssetAnalyzer

# Create a logger specific to this module
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    )

# Load in environment variables from config.json file
with open("config.json", encoding="utf-8") as config_file:
    config = json.load(config_file)

# OpenAI API access key
openai.api_key = config["OPENAI_KEY"]

# Set filepath variables
entities_fp = "data/isin_companies.csv"
assets_fp = "data/asset_data.csv"


def main() -> None:
    """Main function to map assets to companies.

    Applies fuzzy matching of asset data to company names to find shortlist
    of potential matches, and uses a LLM for exact matching.
    Company and ISIN matches for each asset are returned as CSV.
    """
    logger.info("Starting analysis...")

    # Assets to analyze
    assets_df = pd.read_csv(assets_fp, keep_default_na=False)

    # Find match for each asset in DataFrame
    for index, row in assets_df.iterrows():

        # Extract asset data
        asset_name = row["name"]
        asset_ownership_name = row["asset_ownership_name"]
        country = row["country"]

        # Initialise AssetAnalyzer class
        asset_analyzer = AssetAnalyzer(
            entities_fp,
            asset_name,
            asset_ownership_name,
            country,
        )

        # Data preprocessing
        asset_analyzer.format_country_names()
        asset_analyzer.remove_special_characters()

        # Get company matches shortlist from fuzzy match
        asset_analyzer.check_fuzzy_entity_matches("company_name", 60)

        # Store potential matches in dataframe
        assets_df.loc[index, "Potential matches"] = str(
            asset_analyzer.potential_matches,
            )

        # If potential matches are identified, use LLM to find closest match
        if asset_analyzer.potential_matches:
            match = asset_analyzer.check_llm_match()

            # Store match in DataFrame
            assets_df.loc[index, "LLM match"] = str(match)

            # Extract ISIN for company match
            isin = asset_analyzer.match_company_to_isin(
                "Entity ISIN",
                "company_name",
                match,
                )

            # Store ISIN in DataFrame
            assets_df.loc[index, "ISIN match"] = str(isin)
            assets_df.to_csv("data/assets_with_matches.csv", encoding="utf-8")

        logger.info(f"Analysed {index + 1} assets out of {len(assets_df)}")

    logger.info("Asset analysis complete.")


if __name__ == "__main__":
    main()
