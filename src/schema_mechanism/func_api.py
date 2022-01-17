from typing import Collection
from typing import Iterable
from typing import MutableSet
from typing import Optional

from schema_mechanism.data_structures import Item
from schema_mechanism.data_structures import ItemAssertion
from schema_mechanism.data_structures import ItemPoolStateView
from schema_mechanism.data_structures import Schema
from schema_mechanism.data_structures import StateElement
from schema_mechanism.data_structures import SymbolicItem
from schema_mechanism.modules import lost_state
from schema_mechanism.modules import new_state


def create_item(state_element: StateElement) -> Item:
    return SymbolicItem(state_element)


def make_assertion(state_element: StateElement, negated: bool = False) -> ItemAssertion:
    return ItemAssertion(item=create_item(state_element), negated=negated)


def make_assertions(state_elements: Iterable[StateElement], negated: bool = False) -> MutableSet[ItemAssertion]:
    return set([ItemAssertion(item=create_item(se), negated=negated) for se in state_elements])


def update_schema(schema: Schema,
                  activated: bool,
                  s_prev: Optional[Collection[StateElement]],
                  s_curr: Collection[StateElement],
                  count: int = 1) -> Schema:
    """ Update the schema based on the previous and current state.

    :param schema: the schema to update
    :param activated: a bool indicated whether the schema was activated (explicitly or implicitly)
    :param s_prev: a collection containing the previous state elements
    :param s_curr: a collection containing the current state elements
    :param count: the number of times to perform this update

    :return: the updated schema
    """
    schema.update(activated=activated,
                  v_prev=ItemPoolStateView(s_prev),
                  v_curr=ItemPoolStateView(s_curr),
                  new=new_state(s_prev, s_curr),
                  lost=lost_state(s_prev, s_curr),
                  count=count)

    return schema
