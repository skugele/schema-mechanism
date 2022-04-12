from typing import Any

from lark import Lark
from lark import Transformer

from schema_mechanism.core import Action
from schema_mechanism.core import CompositeAction
from schema_mechanism.core import Item
from schema_mechanism.core import ItemAssertion
from schema_mechanism.core import ItemPool
from schema_mechanism.core import NULL_STATE_ASSERT
from schema_mechanism.core import Schema
from schema_mechanism.core import SchemaPool
from schema_mechanism.core import SchemaUniqueKey
from schema_mechanism.core import StateAssertion

# Lark-based parser grammar (see https://lark-parser.readthedocs.io/en/latest/) used for parsing string representations
# into core objects (e.g., items, assertions, and schemas).
GRAMMAR = r"""
    ?object : item
            | composite_item
            | item_assertion
            | state_assertion
            | schema_with_primitive_action
            | schema_with_composite_action
            | primitive_action
            | composite_action

    STRING : /[^~,\/)(]+/

    negated : "~"
    primitive_action: STRING   
    composite_action: state_assertion
    item: STRING
    item_assertion : [negated] (item | composite_item)
    state_assertion : item_assertion ("," | ("," item_assertion)+ [","])
    composite_item : "(" state_assertion ")"
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

    def item(self, tokens: list[Any]) -> Item:
        (state_element,) = tokens
        return ItemPool().get(str(state_element), **self.opt_kwargs)

    def composite_item(self, tokens: list[Any]) -> Item:
        (state_assertion,) = tokens
        return ItemPool().get(state_assertion, **self.opt_kwargs)

    def item_assertion(self, tokens: list[Any]) -> ItemAssertion:
        (is_negated, item) = tokens
        return ItemAssertion(item=item, negated=is_negated or False)

    def state_assertion(self, item_asserts: list[Any]) -> StateAssertion:
        return StateAssertion(asserts=item_asserts) if item_asserts else NULL_STATE_ASSERT

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

    def negated(self, _: Any):
        return True


transformer = ObjectTransformer()


def parse(str_repr: str, **kwargs) -> Any:
    transformer.opt_kwargs = kwargs or dict()
    return transformer.transform(parser.parse(str_repr))
