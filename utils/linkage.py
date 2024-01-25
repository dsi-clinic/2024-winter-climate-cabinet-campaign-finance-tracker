import textdistance as td
import usaddress

"""
Module for performing record linkage on state campaign finance dataset
"""
import numpy as np
import pandas as pd


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


def calculate_row_similarity(
    row1: pd.DataFrame, row2: pd.DataFrame, weights: np.array, comparison_func
) -> float:
    """Find weighted similarity of two rows in a dataframe

    The length of the weights vector must be the same as
    the number of selected columns.

    This version is slow and not optimized, and will be
    revised in order to make it more efficient. It
    exists as to provide basic functionality. Once we have
    the comparison function locked in, using .apply will
    likely be easier and more efficient.
    """

    row_length = len(weights)
    if not (row1.shape[1] == row2.shape[1] == row_length):
        raise ValueError("Number of columns and weights must be the same")

    similarity = np.zeros(row_length)

    for i in range(row_length):
        similarity[i] = comparison_func(row1.iloc[:, i], row2.iloc[:, i])

    return sum(similarity * weights)


def row_matches(df: pd.DataFrame, weights: np.array, threshold: float, comparison_func) -> dict:
    """Get weighted similarity score of two rows

    Run through the rows using indices: if two rows have a comparison score
    greater than a threshold, we assign the later row to the former. Any
    row which is matched to any other row is not examined again. Matches are
    stored in a dictionary object, with each index appearing no more than once.

    This is not optimized
    """

    all_indices = np.array(list(df.index))

    index_dict = {}
    [index_dict.setdefault(x, []) for x in all_indices]

    discard_indices = []

    end = max(all_indices)
    for i in all_indices:
        # Skip indices that have been stored in the discard_indices list
        if i in discard_indices:
            continue

        # Iterate through the remaining numbers
        for j in range(i + 1, end):
            if j in discard_indices:
                continue

            # Our conditional
            if calculate_row_similarity(df.iloc[[i]], df.iloc[[j]], weights, comparison_func) > threshold:
                # Store the other index and mark it for skipping in future iterations
                discard_indices.append(j)
                index_dict[i].append[j]

    return index_dict
