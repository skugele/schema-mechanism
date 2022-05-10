from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import Collection
from typing import Optional
from typing import TypeVar

import numpy as np

from schema_mechanism.strategies.decay import DecayStrategy
from schema_mechanism.util import AssociativeArrayList

T = TypeVar('T')


class Trace(ABC, AssociativeArrayList[T]):
    """

        See Sutton and Barto, 2018, Chapter 12.)
    """

    def __init__(self, decay_strategy: DecayStrategy, **kwargs):
        super().__init__(**kwargs)

        self.decay_strategy: DecayStrategy = decay_strategy

    @abstractmethod
    def update(self, active_set: Optional[Collection[T]] = None) -> None:
        pass


class AccumulatingTrace(Trace):
    """ Implements a generic-type accumulating trace.

        During update, accumulating traces decay all elements, and then increment all elements in their active sets by
        a specified value.

        See Sutton and Barto, 2018, Chapter 12.
    """

    def __init__(self, decay_strategy: DecayStrategy, active_increment: float = 1.0, **kwargs):
        super().__init__(decay_strategy=decay_strategy, **kwargs)

        self.active_increment: float = active_increment

    def update(self, active_set: Optional[Collection[T]] = None):
        active_set = np.array(active_set) if active_set is not None else np.array([])

        # add any new elements
        self.add(active_set)

        # decay all values
        self.values = self.decay_strategy.decay(self.values)

        # increase value of all elements in active_set
        indexes = self.indexes(active_set)
        if len(indexes) > 0:
            self.values[indexes] += self.active_increment


class ReplacingTrace(Trace):
    """ Implements a generic-type replacing trace.

        During update, replacing traces decay all elements and then set all elements in their active sets to a
        specified value (typically 1.0).

        See Sutton and Barto, 2018, Chapter 12.
    """

    def __init__(self, decay_strategy: DecayStrategy, active_value: float = 1.0, **kwargs):
        super().__init__(decay_strategy=decay_strategy, **kwargs)

        self.active_value = active_value

    def update(self, active_set: Optional[Collection[T]] = None):
        active_set = np.array(active_set) if active_set is not None else np.array([])

        # add any new elements
        self.add(active_set)

        # decay all values
        self.values = self.decay_strategy.decay(self.values)

        # set value of all elements in active_set to 1.0
        indexes = self.indexes(active_set)
        if len(indexes) > 0:
            self.values[indexes] = self.active_value
