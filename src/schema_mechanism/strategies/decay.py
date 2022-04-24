from typing import Optional
from typing import Protocol
from typing import runtime_checkable

import numpy as np


@runtime_checkable
class DecayStrategy(Protocol):
    def decay(self, value: float, count: int = 1) -> float:
        """ Decays a value based on a decay function.

        :param value: the value to be decayed
        :param count: the number of times decay will be applied

        :return: the decayed value
        """


#################################################
# implementations of the DecayStrategy protocol #
#################################################


class LinearDecayStrategy:
    def __init__(self, rate: float, minimum: Optional[float] = None) -> None:
        self.minimum = minimum if minimum is not None else -np.inf
        self.rate = rate

    @property
    def minimum(self) -> float:
        return self._minimum

    @minimum.setter
    def minimum(self, value: float) -> None:
        self._minimum = value

    @property
    def rate(self) -> float:
        return self._rate

    @rate.setter
    def rate(self, value: float) -> None:
        if value <= 0.0:
            raise ValueError('Decay strategy\'s rate value must be positive.')
        self._rate = value

    def decay(self, value: float, count: int = 1) -> float:
        return max(value - self.rate * count, self.minimum)


class GeometricDecayStrategy:
    def __init__(self, rate: float, minimum: Optional[float] = None) -> None:
        self.minimum = minimum if minimum is not None else -np.inf
        self.rate = rate

    @property
    def minimum(self) -> float:
        return self._minimum

    @minimum.setter
    def minimum(self, value: float) -> None:
        self._minimum = value

    @property
    def rate(self) -> float:
        return self._rate

    @rate.setter
    def rate(self, value: float) -> None:
        if not 0.0 < value < 1.0:
            raise ValueError('Decay strategy\'s rate value must be between 0.0 and 1.0 (exclusive).')
        self._rate = value

    def decay(self, value: float, count: int = 1) -> float:
        return max(value * self.rate ** count, self.minimum)
