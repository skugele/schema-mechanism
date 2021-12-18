from dataclasses import dataclass
from numbers import Number
from typing import Any, Callable, Iterable, Optional

import numpy as np

from schema_mechanism.util import cosine_sims


@dataclass(frozen=True)
class State:
    discrete_values: Iterable[str] = None
    continuous_values: Iterable[np.ndarray] = None


@dataclass(frozen=True)
class Item:
    value: Any

    def is_on(self, state: Iterable, *args, **kwargs):
        pass


@dataclass(frozen=True)
class DiscreteItem(Item):
    """ A state element that can be thought as a proposition/feature. """

    value: str

    def is_on(self, state: State, *args, **kwargs) -> bool:
        if state is None or state.discrete_values is None:
            return False

        return self.value in state.discrete_values


@dataclass(frozen=True)
class ContinuousItem(Item):
    """ A state element that can be viewed as continuously comparable content. """

    DEFAULT_PRECISION = 2  # 2 decimal places of precision
    DEFAULT_ACTIVATION_THRESHOLD = 0.99

    value: np.ndarray

    similarity_measure: Callable[[np.ndarray, Iterable[np.ndarray]], np.ndarray] = cosine_sims

    def is_on(self, state: State, *args, **kwargs) -> bool:
        if state is None or state.continuous_values is None:
            return False

        threshold = kwargs['threshold'] if 'threshold' in kwargs else ContinuousItem.DEFAULT_ACTIVATION_THRESHOLD
        precision = kwargs['precision'] if 'precision' in kwargs else ContinuousItem.DEFAULT_PRECISION

        similarities = self.similarity_measure(self.value, state.continuous_values).round(precision)
        return np.any(similarities >= threshold)
