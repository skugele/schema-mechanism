from typing import Iterable
from typing import MutableSet

from schema_mechanism.data_structures import Item
from schema_mechanism.data_structures import ItemAssertion
from schema_mechanism.data_structures import StateElement
from schema_mechanism.data_structures import SymbolicItem


def create_item(state_element: StateElement) -> Item:
    return SymbolicItem(state_element)


def make_assertion(state_element: StateElement, negated: bool = False) -> ItemAssertion:
    return ItemAssertion(item=create_item(state_element), negated=negated)


def make_assertions(state_elements: Iterable[StateElement], negated: bool = False) -> MutableSet[ItemAssertion]:
    return set([ItemAssertion(item=create_item(se), negated=negated) for se in state_elements])

