import textdistance as td
import usaddress
from names_dataset import NameDataset

# Initialize the NameDataset class, takes too long to initialize within the function
nd = NameDataset()

"""
Module for performing record linkage on state campaign finance dataset
"""


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


def get_street_from_address_line_1(address_line_1: str) -> str:
    """Given an address line 1, return the street name

    Args:
        address_line_1: either street information or PO box
    Returns:
        street name
    Raises:
        ValueError: if string is malformed and no street can be reasonably
            found.

    >>> get_street_from_address_line_1("5645 N. UBER ST")
    'UBER ST'
    >>> get_street_from_address_line_1("")
    Traceback (most recent call last):
        ...
    ValueError: address_line_1 must have whitespace
    >>> get_street_from_address_line_1("PO Box 1111")
    Traceback (most recent call last):
        ...
    ValueError: address_line_1 is PO Box
    >>> get_street_from_address_line_1("300 59 St.")
    '59 St.'
    >>> get_street_from_address_line_1("Uber St.")
    'Uber St.'
    >>> get_street_from_address_line_1("3NW 59th St")
    '59th St'
    """
    if not address_line_1 or address_line_1.isspace():
        raise ValueError("address_line_1 must have whitespace")

    address_line_lower = address_line_1.lower()

    if "po box" in address_line_lower:
        raise ValueError("address_line_1 is PO Box")

    string = []
    address = usaddress.parse(address_line_1)
    for key, val in address:
        if val in ["StreetName", "StreetNamePostType"]:
            string.append(key)

    return " ".join(string)


def name_rank(first_name: str, last_name: str) -> list:
    """Returns a score for the rank of a given first name and last name
    https://github.com/philipperemy/name-dataset
    Args:
        first_name: any string
        last_name: any string
    Returns:
        name rank for first name and last names
        1 is the most common name, only for names in the United States
        First element in the list corresponds to the rank of the first name
        Second element in the list corresponds to the rank of the last name
        Empty or non string values will return None
        Names that are not found in the dataset will return 0

    >>> name_rank("John", "Smith")
    [5, 7]
    >>> name_rank("Adil", "Kassim")
    [0, 7392]
    >>> name_rank(None, 9)
    [None, None
    """
    first_name_rank = 0
    last_name_rank = 0
    if isinstance(first_name, str):
        first_name_result = nd.search(first_name)
        if first_name_result and isinstance(first_name_result, dict):
            first_name_data = first_name_result.get("first_name")
            if first_name_data and "rank" in first_name_data:
                first_name_rank = first_name_data["rank"].get(
                    "United States", 0
                )
    else:
        first_name_rank = None
    if isinstance(last_name, str):
        last_name_result = nd.search(last_name)
        if last_name_result and isinstance(last_name_result, dict):
            last_name_data = last_name_result.get("last_name")
            if last_name_data and "rank" in last_name_data:
                last_name_rank = last_name_data["rank"].get("United States", 0)
    else:
        last_name_rank = None
    return [first_name_rank, last_name_rank]
