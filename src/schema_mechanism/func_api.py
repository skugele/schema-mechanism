from typing import Collection
from typing import List
from typing import Optional
from typing import Tuple

from schema_mechanism.data_structures import Action
from schema_mechanism.data_structures import Item
from schema_mechanism.data_structures import ItemAssertion
from schema_mechanism.data_structures import ItemPoolStateView
from schema_mechanism.data_structures import Schema
from schema_mechanism.data_structures import StateAssertion
from schema_mechanism.data_structures import StateElement
from schema_mechanism.data_structures import SymbolicItem
from schema_mechanism.modules import SchemaTreeNode
from schema_mechanism.modules import lost_state
from schema_mechanism.modules import new_state


# TODO: Need to remove whitespace from all of these string representations?
def sym_state(str_repr: str) -> Collection[StateElement]:
    if not str_repr:
        return []

    # TODO: generalize the string -> int conversion to support other types
    return [int(se) for se in str_repr.split(',')]


def sym_item(str_repr: str) -> Item:
    # TODO: generalize the string -> int conversion to support other types
    return SymbolicItem(int(str_repr))


def sym_assert(str_repr: str) -> ItemAssertion:
    negated = False
    if '~' == str_repr[0]:
        negated = True
        str_repr = str_repr[1:]

    return ItemAssertion(sym_item(str_repr), negated)


def sym_asserts(str_repr: str) -> Collection[ItemAssertion]:
    return [sym_assert(token) for token in str_repr.split(',')]


def sym_state_assert(str_repr: str) -> StateAssertion:
    if not str_repr:
        return StateAssertion()
    tokens = str_repr.split(',')
    item_asserts = [sym_assert(ia_str) for ia_str in tokens]
    return StateAssertion(item_asserts)


def sym_schema_tree_node(str_repr: str) -> SchemaTreeNode:
    context_str, action_str, _ = str_repr.split('/')
    return SchemaTreeNode(
        context=sym_state_assert(context_str) if context_str else None,
        action=Action(action_str) if action_str else None, )


def sym_schema(str_repr: str) -> Schema:
    context_str, action_str, result_str = str_repr.split('/')
    if not action_str:
        raise ValueError('An action is required.')

    return Schema(context=sym_state_assert(context_str),
                  action=Action(action_str),
                  result=sym_state_assert(result_str))


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


def actions(n: Optional[int] = None, labels: Optional[List] = None) -> Collection[Action]:
    return (
        [Action(label) for label in labels] if labels else
        [Action(str(i)) for i in range(1, n + 1)] if n else
        []
    )


def primitive_schemas(actions: Collection[Action]) -> Tuple[Schema]:
    return tuple([Schema(action=a) for a in actions])
