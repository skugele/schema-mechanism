from typing import Protocol
from typing import runtime_checkable

import numpy as np


@runtime_checkable
class MatchStrategy(Protocol):
    def __call__(self, values: np.ndarray, ref: float) -> np.ndarray:
        """ Returns a Boolean value for each array element indicating whether it satisfies the match criteria.

        :param values: an array of values to test for match to ref
        :param ref: a reference value to test against

        :return: an array of Boolean values, where elements will be True if they match ref, False otherwise.
        """


class EqualityMatchStrategy:
    def __call__(self, values: np.ndarray, ref: float) -> np.ndarray:
        return values == ref


class AbsoluteDiffMatchStrategy:
    def __init__(self, max_diff: float):
        # the max absolute difference between values that counts as being identical
        self.max_diff = max_diff

    def __call__(self, values: np.ndarray, ref: float) -> np.ndarray:
        return values >= (ref - self.max_diff)
