"""
Module for performing record linkage on state campaign finance dataset
"""
# import pandas as pd
import math

import numpy as np

import re

import textdistance as td
import usaddress

from utils.constants import COMPANY_TYPES


def get_address_line_1_from_full_address(address: str) -> str:
    """Given a full address, return the first line of the formatted address

    Address line 1 usually includes street address or PO Box information.

    Args:
        address: raw string representing full address
    Returns:
        address_line_1

    Sample Usage:
    >>> get_address_line_1_from_full_address('6727 W. Corrine Dr.  Peoria,AZ 85381')
    '6727 W. Corrine Dr.'
    >>> get_address_line_1_from_full_address('P.O. Box 5456  Sun City West ,AZ 85375')
    'P.O. Box 5456'
    >>> get_address_line_1_from_full_address('119 S 5th St  Niles,MI 49120')
    '119 S 5th St'
    >>> get_address_line_1_from_full_address(
    ...     '1415 PARKER STREET APT 251	DETROIT	MI	48214-0000'
    ... )
    '1415 PARKER STREET'
    """
    pass

    address_tuples = usaddress.parse(
        address
    )  # takes a string address and put them into value,key pairs as tuples
    line1_components = []
    for value, key in address_tuples:
        if key == "PlaceName":
            break
        elif key in (
            "AddressNumber",
            "StreetNamePreDirectional",
            "StreetName",
            "StreetNamePostType",
            "USPSBoxType",
            "USPSBoxID",
        ):
            line1_components.append(value)
    line1 = " ".join(line1_components)
    return line1


def calculate_string_similarity(string1: str, string2: str) -> float:
    """Returns how similar two strings are on a scale of 0 to 1
    
    This version utilizes Jaro-Winkler distance, which is a metric of
    edit distance. Jaro-Winkler specially prioritizes the early
    characters in a string.

    Since the ends of strings are often more valuable in matching names
    and addresses, we reverse the strings before matching them.

    https://en.wikipedia.org/wiki/Jaro%E2%80%93Winkler_distance
    https://github.com/Yomguithereal/talisman/blob/master/src/metrics/jaro-winkler.js

    The exact meaning of the metric is open, but the following must hold true:
    1. equivalent strings must return 1
    2. strings with no similar characters must return 0
    3. strings with higher intuitive similarity must return higher scores

    Args:
        string1: any string
        string2: any string
    Returns:
        similarity score

    Sample Usage:
    >>> calculate_string_similarity("exact match", "exact match")
    1.0
    >>> calculate_string_similarity("aaaaaa", "bbbbbbbbbbb")
    0.0
    >>> similar_score = calculate_string_similarity("very similar", "vary similar")
    >>> different_score = calculate_string_similarity("very similar", "very not close")
    >>> similar_score > different_score
    True
    """

    return float(td.jaro_winkler(string1.lower()[::-1], string2.lower()[::-1]))


def match_confidence(
    confidences: np.array(float), weights: np.array(float), weights_toggle: bool
) -> float:
    """Combine confidences for row matches into a final confidence

    This is a weighted log-odds based combination of row match confidences
    originating from various record linkage methods. Weights will be applied
    to the linkage methods in order and must be of the same length.

    weights_toggle allows one to turn weights on and off when calling the
    function. False cancels the use of weights.

    Since log-odds have undesirable behaviors at 0 and 1, we truncate at
    +-5, which corresponds to around half a percent probability or
    1 - the same.
    >>> match_confidence(np.array([.6, .9, .0001]), np.array([2,5.7,8]), True)
    2.627759082143462e-12
    >>> match_confidence(np.array([.6, .9, .0001]), np.array([2,5.7,8]), False)
    0.08337802853594725
    """

    if (min(confidences) < 0) or (max(confidences) > 1):
        raise ValueError("Probabilities must be bounded on [0, 1]")

    log_odds = []

    for c in confidences:
        l_o = np.log(c / (1 - c))

        if l_o > 5:
            l_o = 5

        elif l_o < -5:
            l_o = -5

        log_odds.append(l_o)

    if weights_toggle:
        log_odds = log_odds * weights

    l_o_sum = np.sum(log_odds)

    conf_sum = math.e ** (l_o_sum) / (1 + math.e ** (l_o_sum))
    return conf_sum


def cleaning_company_column(company_entry: str) -> str:
    """
    Given a string, check if it contains a variation of self employed, unemployed,
    or retired and return the standardized version.

    Args:
        company: string of inputted company names
    Returns:
        standardized for retired, self employed, and unemployed,
        or original string if no match or empty string

    >>> cleaning_company_column("Retireed")
    'Retired'
    >>> cleaning_company_column("self")
    'Self Employed'
    >>> cleaning_company_column("None")
    'Unemployed'
    >>> cleaning_company_column("N/A")
    'Unemployed'
    """

    if not company_entry:
        return company_entry

    company_edited = company_entry.lower()

    if company_edited == "n/a":
        return "Unemployed"

    company_edited = re.sub(r"[^\w\s]", "", company_edited)

    if (
        company_edited == "retired"
        or company_edited == "retiree"
        or company_edited == "retire"
        or "retiree" in company_edited
    ):
        return "Retired"

    elif (
        "self employe" in company_edited
        or "freelance" in company_edited
        or company_edited == "self"
        or company_edited == "independent contractor"
    ):
        return "Self Employed"
    elif (
        "unemploye" in company_edited
        or company_edited == "none"
        or company_edited == "not employed"
        or company_edited == "nan"
    ):
        return "Unemployed"

    else:
        return company_edited


def standardize_corp_names(company_name: str) -> str:
    """Given an employer name, return the standardized version

    Args:
        company_name: corporate name
    Returns:
        standardized company name

    >>> standardize_corp_names('MI BEER WINE WHOLESALERS ASSOC')
    'MI BEER WINE WHOLESALERS ASSOCIATION'

    >>> standardize_corp_names('MI COMMUNITY COLLEGE ASSOCIATION')
    'MI COMMUNITY COLLEGE ASSOCIATION'

    >>> standardize_corp_names('STEPHANIES CHANGEMAKER FUND')
    'STEPHANIES CHANGEMAKER FUND'

    """

    company_name_split = company_name.upper().split(" ")

    for i in range(len(company_name_split)):
        if company_name_split[i] in list(COMPANY_TYPES.keys()):
            hold = company_name_split[i]
            company_name_split[i] = COMPANY_TYPES[hold]

    new_company_name = " ".join(company_name_split)
    return new_company_name


def get_address_number_from_address_line_1(address_line_1: str) -> str:
    """Given an address line 1, return the building number or po box

    Args:
        address_line_1: either street information or PO box
    Returns:
        address or po box number

    Sample Usage:
    >>> get_address_number_from_address_line_1('6727 W. Corrine Dr.  Peoria,AZ 85381')
    '6727'
    >>> get_address_number_from_address_line_1('P.O. Box 5456  Sun City West ,AZ 85375')
    '5456'
    >>> get_address_number_from_address_line_1('119 S 5th St  Niles,MI 49120')
    '119'
    >>> get_address_number_from_address_line_1(
    ...     '1415 PARKER STREET APT 251	DETROIT	MI	48214-0000'
    ... )
    '1415'
    """

    address_line_1_components = usaddress.parse(address_line_1)

    for i in range(len(address_line_1_components)):
        if address_line_1_components[i][1] == "AddressNumber":
            return address_line_1_components[i][0]
        elif address_line_1_components[i][1] == "USPSBoxID":
            return address_line_1_components[i][0]
    raise ValueError("Can not find Address Number")
