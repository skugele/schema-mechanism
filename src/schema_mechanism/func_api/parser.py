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
from schema_mechanism.core import StateAssertion
from schema_mechanism.share import GlobalParams

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
        # self.opt_kwargs['item_type'] = GlobalParams().get('item_type') or SymbolicItem
        return ItemPool().get(str(state_element), **self.opt_kwargs)

    def composite_item(self, tokens: list[Any]) -> Item:
        (state_assertion,) = tokens
        # self.opt_kwargs['item_type'] = GlobalParams().get('composite_item_type') or CompositeItem
        return ItemPool().get(state_assertion, **self.opt_kwargs)

    def item_assertion(self, tokens: list[Any]) -> ItemAssertion:
        (is_negated, item) = tokens
        return ItemAssertion(item=item, negated=is_negated or False)

    def state_assertion(self, item_asserts: list[Any]) -> StateAssertion:
        return StateAssertion(asserts=item_asserts) if item_asserts else NULL_STATE_ASSERT

    def schema_with_primitive_action(self, tokens: list[Any]) -> Schema:
        (context, primitive_action, result) = tokens

        # TODO: the mixture of the ItemPool "factory" and this direct call to the schema initialize is strange. I
        #       should consider creating a schema factory...
        schema_type = GlobalParams().get('schema_type') or Schema
        return schema_type(context=context, action=primitive_action, result=result, **self.opt_kwargs)

    def schema_with_composite_action(self, tokens: list[Any]) -> Schema:
        (context, composite_action, result) = tokens
        schema_type = GlobalParams().get('schema_type') or Schema

        # TODO: the mixture of the ItemPool "factory" and this direct call to the schema initialize is strange. I
        #       should consider creating a schema factory...
        return schema_type(context=context,
                           action=composite_action,
                           result=result,
                           **self.opt_kwargs)

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
