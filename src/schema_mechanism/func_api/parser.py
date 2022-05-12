import logging
from typing import Any

from lark import Lark
from lark import Transformer

from schema_mechanism.core import Action
from schema_mechanism.core import CompositeAction
from schema_mechanism.core import Item
from schema_mechanism.core import ItemPool
from schema_mechanism.core import NULL_STATE_ASSERT
from schema_mechanism.core import Schema
from schema_mechanism.core import SchemaPool
from schema_mechanism.core import SchemaUniqueKey
from schema_mechanism.core import StateAssertion

logger = logging.getLogger(__name__)

# Lark-based parser grammar (see https://lark-parser.readthedocs.io/en/latest/) used for parsing string representations
# into core objects (e.g., items, assertions, and schemas).
GRAMMAR = r"""
    ?object : item
            | composite_item
            | state_assertion
            | schema_with_primitive_action
            | schema_with_composite_action
            | primitive_action
            | composite_action

    STRING : /[^,\/)(]+/

    state_element: STRING
    item: state_element
    composite_item : "(" state_element ("," | ("," state_element)+ [","]) ")"
    state_assertion : (item | composite_item) ("," | ("," (item | composite_item))+ [","])
    primitive_action: STRING   
    composite_action: state_assertion
    schema_with_primitive_action : [state_assertion] "/" primitive_action "/" [state_assertion]
    schema_with_composite_action : [state_assertion] "/" composite_action "/" [state_assertion]

    %import common.WS
    %ignore WS
"""

parser = Lark(GRAMMAR, start='object')


class ObjectTransformer(Transformer):
    def __init__(self):
        super().__init__()

        self.opt_kwargs = dict()

    def state_element(self, tokens: list[Any]) -> str:
        (string,) = tokens
        return str(string)

    def item(self, tokens: list[Any]) -> Item:
        (state_element,) = tokens
        return ItemPool().get(str(state_element), **self.opt_kwargs)

    def composite_item(self, state_elements: list[Any]) -> Item:
        return ItemPool().get(frozenset(state_elements), **self.opt_kwargs)

    def state_assertion(self, items: list[Any]) -> StateAssertion:
        return StateAssertion(items=items) if items else NULL_STATE_ASSERT

    def schema_with_primitive_action(self, tokens: list[Any]) -> Schema:
        (context, primitive_action, result) = tokens
        key = SchemaUniqueKey(context=context, action=primitive_action, result=result)
        return SchemaPool().get(key, **self.opt_kwargs)

    def schema_with_composite_action(self, tokens: list[Any]) -> Schema:
        (context, composite_action, result) = tokens
        key = SchemaUniqueKey(context=context, action=composite_action, result=result)
        return SchemaPool().get(key, **self.opt_kwargs)

    def primitive_action(self, tokens: list[Any]) -> Action:
        (action,) = tokens
        return Action(label=str(action))

    def composite_action(self, tokens: list[Any]) -> CompositeAction:
        (state_assertion,) = tokens
        return CompositeAction(goal_state=state_assertion)


transformer = ObjectTransformer()


def parse(str_repr: str, **kwargs) -> Any:
    transformer.opt_kwargs = kwargs or dict()
    return transformer.transform(parser.parse(str_repr))
