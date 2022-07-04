from __future__ import annotations

import logging
from collections import Sequence
from typing import Protocol
from typing import runtime_checkable

import numpy as np

from schema_mechanism.core import Schema
from schema_mechanism.strategies.match import EqualityMatchStrategy
from schema_mechanism.strategies.match import MatchStrategy
from schema_mechanism.util import rng

logger = logging.getLogger('schema_mechanism.strategies.selection')


@runtime_checkable
class SelectionStrategy(Protocol):
    def __call__(self, schemas: Sequence[Schema], values: np.ndarray) -> tuple[Schema, float]:
        """ Selects a single schema based on the supplied ordered array of values.

        :param schemas: the schemas to be selected from
        :param values: an array containing the schemas' values

        :return: a single selected Schema and its effective selection value
        """


class RandomizeSelectionStrategy:
    def __call__(self, schemas: Sequence[Schema], values: np.ndarray) -> tuple[Schema, float]:
        selection_index = rng().uniform(0, len(schemas))
        return schemas[selection_index], values[selection_index]

    def __eq__(self, other) -> bool:
        if isinstance(other, RandomizeSelectionStrategy):
            return True
        return False if other is None else NotImplemented


class RandomizeBestSelectionStrategy:
    def __init__(self, match_strategy: MatchStrategy = None):
        self.match_strategy = match_strategy or EqualityMatchStrategy()

    def __call__(self, schemas: Sequence[Schema], values: np.ndarray) -> tuple[Schema, float]:
        max_value = np.max(values)

        # randomize selection if several schemas have values within sameness threshold
        best_schemas = np.argwhere(self.match_strategy(values, max_value)).flatten()
        selection_index = rng().choice(best_schemas, size=1)[0]

        return schemas[selection_index], values[selection_index]

    def __eq__(self, other) -> bool:
        if isinstance(other, RandomizeBestSelectionStrategy):
            return all(
                (
                    # generator expression for conditions to allow lazy evaluation
                    condition for condition in
                    [
                        self.match_strategy == other.match_strategy,
                    ]
                )
            )
        return False if other is None else NotImplemented
