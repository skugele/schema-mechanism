from collections import Collection
from typing import Optional

from schema_mechanism.data_structures import Action
from schema_mechanism.data_structures import Item
from schema_mechanism.data_structures import ItemAssertion
from schema_mechanism.data_structures import ItemPool
from schema_mechanism.data_structures import Schema
from schema_mechanism.data_structures import SchemaTreeNode
from schema_mechanism.data_structures import StateAssertion
from schema_mechanism.data_structures import StateElement
from schema_mechanism.data_structures import SymbolicItem


# TODO: generalize the current string -> int conversions in the sym_* methods to support other conversion types
def sym_state(str_repr: str) -> Collection[StateElement]:
    if not str_repr:
        return []

    return [se for se in str_repr.split(',')]


def sym_item(str_repr: str, primitive_value: float = None) -> Item:
    return ItemPool().get(state_element=str_repr,
                          item_type=SymbolicItem,
                          primitive_value=primitive_value)


def sym_items(str_repr: str, primitive_values: Collection[float] = None) -> Collection[Item]:
    state_elements = str_repr.split(',')
    if primitive_values:
        if len(state_elements) != len(primitive_values):
            raise ValueError('Primitive values must either be None or one must be given for each state element')
        return [sym_item(se, val) for se, val in zip(state_elements, primitive_values)]
    return [sym_item(token) for token in str_repr.split(',')]


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


def sym_schema_tree_node(str_repr: str, label: str = None) -> SchemaTreeNode:
    context_str, action_str, _ = str_repr.split('/')
    return SchemaTreeNode(
        context=sym_state_assert(context_str) if context_str else None,
        action=Action(action_str) if action_str else None,
        label=label
    )


def sym_schema(str_repr: str) -> Schema:
    context_str, action_str, result_str = str_repr.split('/')
    if not action_str:
        raise ValueError('An action is required.')

    return Schema(context=sym_state_assert(context_str),
                  action=Action(action_str),
                  result=sym_state_assert(result_str))


def actions(n: Optional[int] = None, labels: Optional[list] = None) -> Collection[Action]:
    return (
        [Action(label) for label in labels] if labels else
        [Action(str(i)) for i in range(1, n + 1)] if n else
        []
    )


def primitive_schemas(actions_: Collection[Action]) -> tuple[Schema]:
    return tuple([Schema(action=a) for a in actions_])
