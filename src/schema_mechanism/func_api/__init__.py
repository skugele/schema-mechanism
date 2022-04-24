from __future__ import annotations

from collections.abc import Collection
from typing import Optional

import lark.exceptions

from schema_mechanism.core import Action
from schema_mechanism.core import Assertion
from schema_mechanism.core import CompositeItem
from schema_mechanism.core import Item
from schema_mechanism.core import ItemAssertion
from schema_mechanism.core import NULL_STATE_ASSERT
from schema_mechanism.core import Schema
from schema_mechanism.core import SchemaPool
from schema_mechanism.core import SchemaTreeNode
from schema_mechanism.core import SchemaUniqueKey
from schema_mechanism.core import StateAssertion
from schema_mechanism.core import lost_state
from schema_mechanism.core import new_state
from schema_mechanism.func_api.parser import parse
from schema_mechanism.protocols import State


def sym_state(str_repr: str) -> State:
    if not str_repr:
        return tuple()
    return tuple(se for se in str_repr.split(','))


def sym_item(str_repr: str, **kwargs) -> Item:
    try:
        obj = parse(str_repr, **kwargs)
    except lark.exceptions.UnexpectedInput:
        raise ValueError(f'String representation for item is invalid: {str_repr}')
    assert isinstance(obj, Item)
    return obj


def sym_composite_item(str_repr: str, **kwargs) -> CompositeItem:
    obj = sym_item(str_repr, **kwargs)
    assert isinstance(obj, CompositeItem)
    return obj


def sym_items(str_repr: str, primitive_values: Collection[float] = None, **kwargs) -> Collection[Item]:
    state_elements = str_repr.split(',')
    if primitive_values:
        if len(state_elements) != len(primitive_values):
            raise ValueError('Primitive values must either be None or one must be given for each state element')
        return [sym_item(se, primitive_value=val, **kwargs) for se, val in zip(state_elements, primitive_values)]
    return [sym_item(token, **kwargs) for token in str_repr.split(',')]


def sym_item_assert(str_repr: str, **kwargs) -> ItemAssertion:
    obj = sym_assert(str_repr, **kwargs)
    if not isinstance(obj, ItemAssertion):
        raise ValueError(f'String representation for item assertion is invalid: {str_repr}')
    return obj


def sym_state_assert(str_repr: str, **kwargs) -> StateAssertion:
    # parser requires a trailing comma for one element state assertions
    if str_repr and not str_repr.endswith(','):
        str_repr += ','

    obj = sym_assert(str_repr, **kwargs)
    if not isinstance(obj, StateAssertion):
        raise ValueError(f'String representation for state assertion is invalid: {str_repr}')
    return obj


def sym_assert(str_repr: str, **kwargs) -> Assertion:
    try:
        obj = parse(str_repr, **kwargs)
    except lark.exceptions.UnexpectedInput:
        raise ValueError(f'String representation for assertion is invalid: {str_repr}')

    # workaround to a parsing ambiguity when single character strings are passed (parser currently returns an Item)
    if isinstance(obj, Item):
        obj = ItemAssertion(item=obj)

    assert isinstance(obj, Assertion)
    return obj


def sym_asserts(str_repr: str, **kwargs) -> Collection[Assertion]:
    return [sym_assert(token, **kwargs) for token in str_repr.split(',')]


def sym_schema_tree_node(str_repr: str, label: str = None) -> SchemaTreeNode:
    return SchemaTreeNode(
        context=sym_state_assert(str_repr) if str_repr else NULL_STATE_ASSERT,
        label=label
    )


def sym_schema(str_repr: str, **kwargs) -> Schema:
    try:
        obj = parse(str_repr, **kwargs)
    except lark.exceptions.UnexpectedInput:
        raise ValueError(f'String representation for schema is invalid: {str_repr}')

    assert isinstance(obj, Schema)
    return obj


def actions(n: Optional[int] = None, labels: Optional[list] = None) -> Collection[Action]:
    return (
        [Action(label) for label in labels] if labels else
        [Action(str(i)) for i in range(1, n + 1)] if n else
        []
    )


def primitive_schemas(actions_: Collection[Action]) -> tuple[Schema]:
    schemas: list[Schema] = [SchemaPool().get(SchemaUniqueKey(action=a)) for a in actions_]
    return tuple(schemas)


def update_schema(schema: Schema,
                  activated: bool,
                  succeeded: bool,
                  s_prev: Optional[State],
                  s_curr: State,
                  explained: Optional[bool] = None,
                  count: int = 1) -> Schema:
    """ Update the schema based on the previous and current state.

    :param schema: the schema to update
    :param activated: a bool indicated whether the schema was activated (explicitly or implicitly)
    :param s_prev: a collection containing the previous state elements
    :param s_curr: a collection containing the current state elements
    :param explained: True if a reliable schema was activated that "explained" the last state transition
    :param succeeded: True if the schema was successful in obtaining its result when activated; False otherwise.
    :param count: the number of times to perform this update

    :return: the updated schema
    """
    schema.update(activated=activated,
                  succeeded=succeeded,
                  selection_state=s_prev,
                  new=new_state(s_prev, s_curr),
                  lost=lost_state(s_prev, s_curr),
                  explained=explained,
                  count=count)

    return schema
