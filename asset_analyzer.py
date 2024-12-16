import re
import json
import numpy as np
import openai
import pandas as pd
from fuzzywuzzy import fuzz
from fuzzywuzzy import process


class AssetAnalyzer:
    """A class to analyze asset data and match it against potential entities."""

    def __init__(
            self,
            entities_fp: str,
            asset_name: str,
            asset_ownership: str,
            asset_country: str,
            ) -> None:
        """Initialize the AssetAnalyzer with asset and entity data.

        Parameters
        ----------
            entities_fp (str): Filepath to the CSV containing company information.
            asset_name (str): The name of the asset being analyzed.
            asset_ownership (str): The ownership name of the asset.
            asset_country (str): The country where the asset is based.

        """
        self.entities_df = self.load_csv_as_pd(entities_fp)
        self.asset_name = str(asset_name)
        self.asset_ownership = str(asset_ownership)
        self.asset_country = str(asset_country)
        self.potential_matches = []

    def load_csv_as_pd(self, filepath: str):
        """Load csv file into pandas DataFrame.

        Parameters
        ----------
            filepath (str): The path to the CSV file.

        Returns
        -------
            pd.DataFrame: A pandas DataFrame containing the CSV data.

        """
        return pd.read_csv(filepath)

    def format_country_names(self) -> str:
        """Format the country name by swapping the order if it contains a comma.

        If the country name is in the format "Country, City", it changes it to 
        "City Country" (e.g., "United States, America" -> "America United States").

        Returns
        -------
            str: The formatted country name if a comma is present, otherwise None.

        """
        if "," in self.asset_country:
            parts = self.asset_country.split(", ")
            self.asset_country = f"{parts[1]} {parts[0]}"
            return self.asset_country
        return None

    def remove_special_characters(self) -> None:
        """Remove special characters from the asset name and asset ownership.

        This method uses a regular expression to remove any characters
        that are not alphanumeric or whitespace.
        """
        pattern = r"[^\w\s]"
        self.asset_name = re.sub(pattern, "", self.asset_name)
        self.asset_ownership = re.sub(pattern, "", self.asset_ownership)
        return None

    def check_fuzzy_entity_matches(self, entity_name_column: str, threshold: int) -> list:
        """Check if any entities match the asset name or ownership.

        This method compares the asset name and asset ownership with each entity's
        name from the specified column to find potential matches.

        Parameters
        ----------
            entity_name_column (str): The name of the column in the entity DataFrame
                                       to search for matches.

        Returns
        -------
            list: A list of potential matches found in the entity column.

        """
        for _entity_index, entity_row in self.entities_df.iterrows():
            entity_to_test = entity_row[entity_name_column]
            results = process.extract(
                entity_to_test,
                [self.asset_name, self.asset_ownership, self.asset_country],
                scorer=fuzz.token_set_ratio,
                )
            scores = [score for _, score in results]
            if any(score > threshold for score in scores):
                result_to_append = f"{entity_to_test}"
                self.potential_matches.append(result_to_append)

        return self.potential_matches

    def check_llm_match(self, model : str = "gpt-4o") -> dict:
        """Use an LLM to match asset data to the most relevant company.

        This method formats a user prompt with the asset name, ownership,
        and country and queries a LLM to select the most relevant company
        from a shortlist of potential matches.

        Parameters
        ----------
            model (str): The language model to use (default is "gpt-4o").

        Returns
        -------
            str: The response from the model in JSON format containing the "isin"
                 and "company_name" of the closest match.

        """
        system_prompt = """
        You are an expert analyzer of ESG Asset Data. Your role is to match asset data to the most relevant company name, from a given shortlist.

        Inputs:
        You will be given asset data which includes:
        1. Asset name
        2. Asset ownership name
        3. Country where the asset is based
        4. A list of possible companies this asset belongs to:

        Guidance:
        Consider all asset data given, and the full list of potential companies in the shortlist. 
        Return the company name to which the asset most likely belongs. Indicators of a potential match include:
        - Geographical information contained in the asset data and company
        - Acronyms of the company name are contained in the asset data
        - The name or ownership of the asset is close to the company name


        Output guidance:
        You must output only one company name from the shortlist, which provides a closest match to the asset data.
        Provide your output in JSON format with key "company_name". Never return None, unless there are no companies given in the shortlist. 
        Example output:

         {"company_name" : "<<company from shortlist>>"}

        """

        user_prompt = """

        Asset name: {asset_name}
        Asset ownership: {asset_ownership}
        Asset country: {asset_country}

        List of possible companies the asset belongs to: 
        {potential_matches}

        """
        try :
            response = openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt.format(
                        asset_name=self.asset_name,
                        asset_ownership=self.asset_ownership,
                        asset_country=self.asset_country,
                        potential_matches=self.potential_matches,
                        )},
                    ],
                response_format={"type": "json_object"},
            )

            llm_result = response.choices[0].message.content
            llm_result_json = json.loads(llm_result)

            return llm_result_json["company_name"]

        except Exception as e:
            print(e)

    def match_company_to_isin(
            self, 
            isin_column_name: str,
            entity_column_name: str,
            entity_match: str,
            ) -> str:
        """Match a company name to its corresponding ISIN.

        This method searches for a row in the DataFrame where the value 
        in the specified entity column matches the provided company name 
        (case-insensitively) and returns the ISIN value from the specified column.

        Parameters
        ----------
        isin_column_name : str
            The name of the column in the DataFrame containing ISIN values.
        entity_column_name : str
            The name of the column in the DataFrame containing company 
            names or entities to be matched.
        entity_match : str
            The company name or entity value to search for.

        Returns
        -------
        str
            The corresponding ISIN value if a match is found.

        """
        try :
            return self.entities_df.loc[
                self.entities_df[entity_column_name].str.lower() == entity_match, 
                isin_column_name,
                ].values[0]
        except Exception as e :
            print("Cannot find ISIN match. Error ", e)
