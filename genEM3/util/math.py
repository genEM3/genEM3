"""
Set of utility functionality for mathematical operations
"""
from typing import Sequence


def minmax(ListofNumbers: Sequence[int]):
    """
    function to return the minimum and maximum of an array
    """
    return [min(ListofNumbers), max(ListofNumbers)]