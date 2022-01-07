from typing import Union

from schema_mechanism.data_structures import SymbolicItem
from schema_mechanism.data_structures import StateElement
from schema_mechanism.data_structures import Item
from schema_mechanism.data_structures import ItemAssertion


def create_item(state_element: StateElement) -> Item:
    return SymbolicItem(state_element)


def make_assertion(state_element: StateElement, negated: bool = False) -> ItemAssertion:
    return ItemAssertion(item=create_item(state_element), negated=negated)
