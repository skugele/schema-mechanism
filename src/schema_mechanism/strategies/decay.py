from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Optional

import numpy as np


class DecayStrategy(ABC):

    @abstractmethod
    def decay(self, values: np.ndarray, step_size: float = 1.0) -> np.ndarray:
        """ Decays a value based on a decay function.

        :param values: an array of values to decayed
        :param step_size: a step size that indicates the magnitude of the decay

        :return: an array of decayed values
        """


#################################################
# implementations of the DecayStrategy protocol #
#################################################


class LinearDecayStrategy(DecayStrategy):
    def __init__(self, rate: float, minimum: Optional[float] = None) -> None:
        super().__init__()

        self.minimum = minimum if minimum is not None else -np.inf
        self.rate = rate

    def __eq__(self, other) -> bool:
        if isinstance(other, LinearDecayStrategy):
            return all(
                (
                    # generator expression for conditions to allow lazy evaluation
                    condition for condition in
                    [
                        self.minimum == other.minimum,
                        self.rate == other.rate
                    ]
                )
            )
        return False if other is None else NotImplemented

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

    def decay(self, values: np.ndarray, step_size: float = 1.0) -> np.ndarray:
        if step_size < 0.0:
            raise ValueError('Step size must be non-negative')

        return np.maximum(values - self.rate * step_size, self.minimum * np.ones_like(values))


class GeometricDecayStrategy(DecayStrategy):
    def __init__(self, rate: float) -> None:
        super().__init__()

        self.rate = rate

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, GeometricDecayStrategy):
            return self.rate == other.rate
        return False if other is None else NotImplemented

    @property
    def rate(self) -> float:
        return self._rate

    @rate.setter
    def rate(self, value: float) -> None:
        if not 0.0 < value < 1.0:
            raise ValueError('Decay strategy\'s rate value must be between 0.0 and 1.0 (exclusive).')
        self._rate = value

    def decay(self, values: np.ndarray, step_size: float = 1.0) -> np.ndarray:
        if step_size < 0.0:
            raise ValueError('Step size must be non-negative')

        return values * self.rate ** step_size


class ExponentialDecayStrategy(DecayStrategy):
    def __init__(self, rate: float, initial: Optional[float] = 1.0, minimum: Optional[float] = -np.inf) -> None:
        super().__init__()

        self.rate = rate
        self.initial: float = initial
        self.minimum: float = minimum

    def __eq__(self, other) -> bool:
        if isinstance(other, ExponentialDecayStrategy):
            return all(
                (
                    # generator expression for conditions to allow lazy evaluation
                    condition for condition in
                    [
                        self.rate == other.rate,
                        self.initial == other.initial,
                        self.minimum == other.minimum
                    ]
                )
            )
        return False if other is None else NotImplemented

    @property
    def rate(self) -> float:
        return self._rate

    @rate.setter
    def rate(self, value: float) -> None:
        if not value > 0.0:
            raise ValueError('Decay rate must be greater than 0.0.')
        self._rate = value

    def decay(self, values: np.ndarray, step_size: float = 1.0) -> np.ndarray:
        if step_size < 0.0:
            raise ValueError('Step size must be non-negative')

        x: np.ndarray = np.log(self.initial - values + 1.0) / np.log(self.rate + 1.0)
        y: np.ndarray = self.initial - (1.0 + self.rate) ** (x + step_size) + 1.0

        return np.maximum(y, self.minimum * np.ones_like(y))


class NoDecayStrategy(DecayStrategy):

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, NoDecayStrategy):
            return True
        return False if other is None else NotImplemented

    def decay(self, values: np.ndarray, step_size: float = 1.0) -> np.ndarray:
        if step_size < 0.0:
            raise ValueError('Step size must be non-negative')

        return values


class ImmediateDecayStrategy(DecayStrategy):
    def __init__(self, minimum: float = -np.inf):
        super().__init__()

        self.minimum: float = minimum

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, ImmediateDecayStrategy):
            return self.minimum == other.minimum
        return False if other is None else NotImplemented

    def decay(self, values: np.ndarray, step_size: float = 1.0) -> np.ndarray:
        if step_size < 0.0:
            raise ValueError('Step size must be non-negative')

        if step_size == 0:
            return values

        return self.minimum * np.ones_like(values)
