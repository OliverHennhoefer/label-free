import math

from typing import Union, List
from scipy.stats import kendalltau, spearmanr
from itertools import combinations


def kendalls_tau(*lists_args: Union[List[int | float], List[List[int | float]]]):
    """Calculates the average Kendall's Tau correlation coefficient.

    This function computes Kendall's Tau for all unique pairs of score lists
    provided as input. It can accept multiple lists as separate arguments
    or a single list containing multiple lists of scores. If only a single
    list of scores is provided, it's treated as a list containing one score list.

    Handles cases where Kendall's Tau is NaN (e.g., due to identical lists or
    lists with insufficient variance). If NaN is returned by `scipy.stats.kendalltau`,
    this function appends 1.0 to the tau values if the lists are identical,
    and 0.0 otherwise.

    Args:
        *lists_args: A variable number of arguments.
            This can be either:
                1. Multiple lists, where each list contains scores
                   (e.g., `kendalls_tau([1, 2, 3], [1, 3, 2], [2, 1, 3])`).
                2. A single list that itself contains multiple lists of scores
                   (e.g., `kendalls_tau([[1, 2, 3], [1, 3, 2], [2, 1, 3]])`).
                3. A single list of scores (e.g., `kendalls_tau([1, 2, 3, 4])`).
                   This will be treated as a list containing one score list.

    Returns:
        float: The average Kendall's Tau correlation coefficient across all
        unique pairs of the input score lists.

    Raises:
        ZeroDivisionError: If `processed_score_lists` contains fewer than two lists,
            as `len(tau_values)` will be zero, leading to division by zero when
            calculating the average.

    """

    # Input processing to handle different ways of providing lists
    if len(lists_args) == 1 and isinstance(lists_args[0], list):
        first_arg_as_list = lists_args[0]
        if first_arg_as_list and all(
            isinstance(sub_item, list) for sub_item in first_arg_as_list
        ):
            processed_score_lists = first_arg_as_list
        else:
            processed_score_lists = [first_arg_as_list]
    else:
        processed_score_lists = list(lists_args)

    tau_values = []
    for list1, list2 in combinations(processed_score_lists, 2):
        tau_coefficient, _ = kendalltau(list1, list2)

        if math.isnan(tau_coefficient):
            # If scipy's kendalltau is NaN (e.g. constant arrays),
            # assign 1.0 if lists are identical, 0.0 otherwise.
            tau_values.append(1.0 if list1 == list2 else 0.0)
        else:
            tau_values.append(tau_coefficient)

    return sum(tau_values) / len(tau_values)


def spearmans_rho(*lists_args: Union[List[float | int], List[List[float | int]]]):
    """Calculates the average Spearman's Rank Correlation coefficient (rho).

    This function computes Spearman's Rho for all unique pairs of score lists
    provided as input. It can accept multiple lists as separate arguments
    or a single list containing multiple lists of scores. If only a single
    list of scores is provided, it's treated as a list containing one score list.

    Handles cases where Spearman's Rho is NaN (e.g., due to lists with
    insufficient variance, like constant arrays). If NaN is returned by
    `scipy.stats.spearmanr`, this function appends 1.0 to the rho values
    if the lists are identical, and 0.0 otherwise.

    Args:
        *lists_args: A variable number of arguments.
            This can be either:
                1. Multiple lists, where each list contains scores
                   (e.g., `spearmans_rho([1, 2, 3], [1, 3, 2], [2, 1, 3])`).
                2. A single list that itself contains multiple lists of scores
                   (e.g., `spearmans_rho([[1, 2, 3], [1, 3, 2], [2, 1, 3]])`).
                3. A single list of scores (e.g., `spearmans_rho([1, 2, 3, 4])`).
                   This will be treated as a list containing one score list.

    Returns:
        float: The average Spearman's Rho correlation coefficient across all
        unique pairs of the input score lists.

    Raises:
        ZeroDivisionError: If `processed_score_lists` contains fewer than two lists,
            as `len(rho_values)` will be zero, leading to division by zero when
            calculating the average.
        ValueError: If any input list has fewer than 2 elements, as Spearman's rho
            is not well-defined for such lists by scipy.stats.spearmanr.
    """

    if len(lists_args) == 1 and isinstance(lists_args[0], list):
        first_arg_as_list = lists_args[0]
        if first_arg_as_list and all(
            isinstance(sub_item, list) for sub_item in first_arg_as_list
        ):
            processed_score_lists = first_arg_as_list
        else:
            processed_score_lists = [first_arg_as_list]
    else:
        processed_score_lists = list(lists_args)

    rho_values = []
    for list1, list2 in combinations(processed_score_lists, 2):
        correlation_result = spearmanr(list1, list2)
        rho_coefficient = correlation_result.correlation

        if math.isnan(rho_coefficient):
            rho_values.append(1.0 if list1 == list2 else 0.0)
        else:
            rho_values.append(rho_coefficient)

    return sum(rho_values) / len(rho_values)
