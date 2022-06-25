from abc import ABC
from abc import abstractmethod
from typing import Sequence

import numpy as np

from schema_mechanism.util import get_number_of_digits_after_decimal


class WeightUpdateStrategy(ABC):

    @abstractmethod
    def update(self, weights: Sequence[float]) -> Sequence[float]:
        """ Updates weights based on this WeightUpdateStrategy.

        :param weights: a sequence containing the current weights

        :return: a sequence containing the updated weights
        """


class NoOpWeightUpdateStrategy(WeightUpdateStrategy):
    def __eq__(self, other) -> bool:
        if isinstance(other, WeightUpdateStrategy):
            return True
        return False if other is None else NotImplemented

    def update(self, weights: Sequence[float]) -> Sequence[float]:
        return weights


class CyclicWeightUpdateStrategy(WeightUpdateStrategy):
    MIN_STEP_SIZE = 1e-13

    def __init__(self, step_size: float = 1e-3) -> None:
        self.step_size = step_size

        if step_size <= 0.0 or step_size >= 1.0:
            raise ValueError('Step size must be between 0 and 1 (exclusive)')

        if step_size < CyclicWeightUpdateStrategy.MIN_STEP_SIZE:
            raise ValueError(f'Step size must be greater than {CyclicWeightUpdateStrategy.MIN_STEP_SIZE}')

        self.digits = get_number_of_digits_after_decimal(self.step_size)

    def __eq__(self, other) -> bool:
        if isinstance(other, CyclicWeightUpdateStrategy):
            return self.step_size == other.step_size
        return False if other is None else NotImplemented

    def update(self, weights: np.ndarray) -> np.ndarray:
        if len(weights) != 2:
            raise ValueError('Strategy currently only supports updates for weight arrays of length 2')

        step = self.step_size * np.array([-1.0, 1.0])
        new_weights = (weights + step) % (1.0 + CyclicWeightUpdateStrategy.MIN_STEP_SIZE)

        return np.round(new_weights, decimals=self.digits)
