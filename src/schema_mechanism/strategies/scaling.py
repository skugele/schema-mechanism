from abc import ABC
from abc import abstractmethod

import numpy as np


class ScalingStrategy(ABC):

    @abstractmethod
    def scale(self, values: np.ndarray) -> np.ndarray:
        """ Scales values based on the current strategy.

        :param values: an array of values to scale

        :return: an array containing the scaled values
        """


class SigmoidScalingStrategy(ScalingStrategy):

    def __init__(self,
                 range_scale: float = 2.0,
                 vertical_shift: float = 0.5,
                 intercept: float = 0.0):
        """

        :param range_scale:
        :param vertical_shift:
        :param intercept:
        """
        self.range_scale: float = range_scale
        self.vertical_shift: float = vertical_shift
        self.intercept: float = intercept

    def _sigmoid(self, values: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-values))

    def scale(self, values: np.ndarray) -> np.ndarray:
        return self.range_scale * (-self._sigmoid(values - self.intercept) + self.vertical_shift)
