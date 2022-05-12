from __future__ import annotations

import logging
from collections.abc import Collection
from typing import Optional

import lark.exceptions

from schema_mechanism.core import Action
from schema_mechanism.core import CompositeItem
from schema_mechanism.core import Item
from schema_mechanism.core import NULL_STATE_ASSERT
from schema_mechanism.core import Schema
from schema_mechanism.core import SchemaPool
from schema_mechanism.core import SchemaTreeNode
from schema_mechanism.core import SchemaUniqueKey
from schema_mechanism.core import State
from schema_mechanism.core import StateAssertion
from schema_mechanism.core import lost_state
from schema_mechanism.core import new_state
from schema_mechanism.func_api.parser import parse

logger = logging.getLogger(__name__)


def sym_state(str_repr: str) -> State:
    if not str_repr:
        return frozenset()
    return frozenset(se for se in str_repr.split(','))


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
    sources = str_repr.split(';')
    if primitive_values is not None:
        if len(sources) != len(primitive_values):
            raise ValueError('Primitive values must either be None or one must be given for each state element')
        return [sym_item(se, primitive_value=val, **kwargs) for se, val in zip(sources, primitive_values)]
    return [sym_item(source, **kwargs) for source in sources]


# TODO: this can eventually be removed. It is kept as an alias for sym_state_assert while I deal with this massive
# TODO: redesign.
def sym_assert(str_repr: str, **kwargs) -> StateAssertion:
    return sym_state_assert(str_repr, **kwargs)


def sym_state_assert(str_repr: str, **kwargs) -> StateAssertion:
    try:
        obj = parse(str_repr, **kwargs)
    except (lark.exceptions.UnexpectedInput, lark.exceptions.UnexpectedCharacters) as e:
        raise ValueError(f'String representation for assertion is invalid: {e}') from None

    # workaround to a parsing ambiguity when single character strings are passed (parser currently returns an Item)
    if isinstance(obj, Item):
        obj = StateAssertion(items=[obj])

    if not isinstance(obj, StateAssertion):
        raise ValueError(f'String representation for assertion is invalid')

    return obj


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
