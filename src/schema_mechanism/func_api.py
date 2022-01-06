from typing import Union

from schema_mechanism.data_structures import ContinuousItem
from schema_mechanism.data_structures import ContinuousStateElement
from schema_mechanism.data_structures import DiscreteItem
from schema_mechanism.data_structures import DiscreteStateElement
from schema_mechanism.data_structures import ItemAssertion


def gen_assert(state_element: Union[DiscreteStateElement, ContinuousStateElement],
               negated: bool = False) -> ItemAssertion:
    if isinstance(state_element, DiscreteStateElement):
        return ItemAssertion(item=DiscreteItem(state_element), negated=negated)
    elif isinstance(state_element, ContinuousStateElement):
        return ItemAssertion(item=ContinuousItem(state_element), negated=negated)

    raise TypeError(f'State element type is invalid. Supported types: {DiscreteStateElement, ContinuousStateElement}')
