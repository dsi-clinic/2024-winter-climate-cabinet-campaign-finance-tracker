import os
import sys  # noqa

import pandas as pd
from constants import c_org_names, f_companies, f_org_names

script_dir = os.path.dirname(os.path.abspath(__file__))  # noqa
project_dir = os.path.dirname(script_dir)  # noqa
sys.path.append(project_dir)  # noqa


def classify_wrapper(
    individuals_df: pd.DataFrame, organizations_df: pd.DataFrame
):
    """Wrapper for classification in linkage pipeline

    Initialize the classify column in both dataframes and
    call sub-functions classifying individuals and organizations

    Args:
        individuals_df: cleaned and deduplicated dataframe of individuals
        organizations_df: cleaned and deduplicated dataframe of organizations

    Returns:
        individuals and organizations datfarames with a new
        'classification' column containing 'neutral', 'f', or 'c'.
        'neutral' status is the default for all entities, and those tagged
        as 'neutral' are entities which we could not confidently identify as
        either fossil fuel or clean energy organizations or affiliates.
        Classification is very conservative, and we are very confident that
        entities classified as one group or another are related to them.

    """

    individuals_df["classification"] = "neutral"
    organizations_df["classification"] = "neutral"

    classified_individuals = classify_individuals(individuals_df)
    classified_orgs = classify_orgs(organizations_df)

    return classified_individuals, classified_orgs


def matcher(df: pd.DataFrame, substring: str, column: str, category: str):
    """Applies a label to the classification column based on substrings

    We run through a given column containing strings in the dataframe. We
    seek out rows containing substrings, and apply a certain label to
    the classification column. We initialize using the 'neutral' label and
    use the 'f' and 'c' labels to denote fossil fuel and clean energy
    entities respectively.

    Args:
        df: a pandas dataframe
        substring: the string to search for
        column: the column name in which to search
        category: the category to assign the row, such as 'f' 'c' or 'neutral'

    Returns:
        A pandas dataframe in which rows matching the substring conditions in
        a certain column are marked with the appropriate category
    """

    bool_series = df[column].str.contains(substring, na=False)

    df.loc[bool_series, "classification"] = category

    return df


def classify_individuals(individuals_df: pd.DataFrame):
    """Part of the classification pipeline

    We check if individuals work for a known fossil fuel company
    and categorize them using the matcher() function.

    Args:
        individuals_df: a dataframe containing deduplicated
        standardized individuals data

    Returns:
        an individuals dataframe updated with the fossil fuels category
    """

    for i in f_companies:
        individuals_df = matcher(individuals_df, i, "company", "f")

    return individuals_df


def classify_orgs(organizations_df: pd.DataFrame):
    """Part of the classification pipeline

    We apply the matcher function to the organizations dataframe
    repeatedly, using a variety of substrings to identify fossil
    fuel and clean energy companies.

    Args:
        organizations_df: a dataframe containing deduplicated
        standardized organizations data

    Returns:
        an organizations dataframe updated with the fossil fuels
        and clean energy category
    """

    for i in f_org_names:
        organizations_df = matcher(organizations_df, i, "name", "f")

    for i in c_org_names:
        organizations_df = matcher(organizations_df, i, "name", "c")

    return organizations_df
