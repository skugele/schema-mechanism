from typing import Any
from typing import Collection

import numpy as np

from schema_mechanism.data_structures import Item
from schema_mechanism.util import cosine_sims

ContinuousStateElement = np.ndarray


class ContinuousItem(Item):
    """ A state element that can be viewed as continuously comparable content. """

    DEFAULT_PRECISION = 2  # 2 decimal places of precision
    DEFAULT_ACTIVATION_THRESHOLD = 0.99
    DEFAULT_SIMILARITY_MEASURE = cosine_sims

    def __init__(self, state_element: ContinuousStateElement):
        super().__init__(state_element)

        # prevent modification of array values
        self.state_element.setflags(write=False)

    @property
    def state_element(self) -> ContinuousStateElement:
        return super().state_element

    def is_on(self, state: Collection[Any], *args, **kwargs) -> bool:
        threshold = kwargs['threshold'] if 'threshold' in kwargs else ContinuousItem.DEFAULT_ACTIVATION_THRESHOLD
        precision = kwargs['precision'] if 'precision' in kwargs else ContinuousItem.DEFAULT_PRECISION
        similarity_measure = (
            kwargs['similarity_measure']
            if 'similarity_measure' in kwargs
            else ContinuousItem.DEFAULT_SIMILARITY_MEASURE
        )

        continuous_state_elements = tuple(filter(lambda e: isinstance(e, ContinuousStateElement), state))
        if continuous_state_elements:
            similarities = similarity_measure(self.state_element, continuous_state_elements).round(precision)
            return np.any(similarities >= threshold)

        return False

    def __eq__(self, other: Item) -> bool:
        if isinstance(other, ContinuousItem):
            return np.array_equal(self.state_element, other.state_element)

        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.state_element.tostring())
