

"""
Module for performing record linkage on state campaign finance dataset
"""
import numpy as np
# import pandas as pd
import math


def match_confidence(confidences: np.array(float), weights: np.array(float), weights_toggle: bool) -> float:
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
        l_o = np.log(c/(1-c))

        if l_o > 5:
            l_o = 5

        elif l_o < -5:
            l_o = -5

        log_odds.append(l_o)

    if weights_toggle:
        log_odds = log_odds * weights

    l_o_sum = np.sum(log_odds)

    conf_sum = math.e**(l_o_sum)/(1+math.e**(l_o_sum))

    return conf_sum
