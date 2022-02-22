from typing import Any

from lark import Lark
from lark import Transformer

from schema_mechanism.core import Action
from schema_mechanism.core import GlobalParams
from schema_mechanism.core import Item
from schema_mechanism.core import ItemAssertion
from schema_mechanism.core import ItemPool
from schema_mechanism.core import Schema
from schema_mechanism.core import StateAssertion

# Lark-based parser grammar (see https://lark-parser.readthedocs.io/en/latest/) used for parsing string representations
# into core objects (e.g., items, assertions, and schemas).
GRAMMAR = r"""
    ?object : item
            | composite_item
            | item_assertion
            | state_assertion
            | schema
            | action

    STRING : /[^~,\/)(]+/

    negated : "~"
    action: STRING   
    item: STRING
    item_assertion : [negated] (item | composite_item)
    state_assertion : item_assertion ("," | ("," item_assertion)+)
    composite_item : "(" state_assertion ")"
    schema : [state_assertion] "/" action "/" [state_assertion]

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
        self.opt_kwargs['item_type'] = GlobalParams().DEFAULT_ITEM_TYPE
        return ItemPool().get(str(state_element), **self.opt_kwargs)

    def composite_item(self, tokens: list[Any]) -> Item:
        (state_assertion,) = tokens
        self.opt_kwargs['item_type'] = GlobalParams().DEFAULT_CONJUNCTIVE_ITEM_TYPE
        return ItemPool().get(source=state_assertion, **self.opt_kwargs)

    def item_assertion(self, tokens: list[Any]) -> ItemAssertion:
        (is_negated, item) = tokens
        return ItemAssertion(item=item, negated=is_negated or False)

    def state_assertion(self, item_asserts: list[Any]) -> StateAssertion:
        return StateAssertion(asserts=item_asserts)

    def schema(self, tokens: list[Any]) -> Schema:
        (context, action, result) = tokens
        return Schema(context=context, action=action, result=result)

    def action(self, tokens: list[Any]) -> Action:
        (action,) = tokens
        return Action(label=str(action))

    def negated(self, _: Any):
        return True


transformer = ObjectTransformer()


def parse(str_repr: str, **kwargs) -> Any:
    transformer.opt_kwargs = kwargs or dict()
    return transformer.transform(parser.parse(str_repr))
