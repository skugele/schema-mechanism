from __future__ import annotations

import itertools
from collections import Collection
from collections import Sequence
from collections import deque
from random import sample
from typing import FrozenSet
from typing import NamedTuple
from typing import Optional

from schema_mechanism.data_structures import Action
from schema_mechanism.data_structures import Item
from schema_mechanism.data_structures import ItemAssertion
from schema_mechanism.data_structures import ItemPoolStateView
from schema_mechanism.data_structures import NULL_STATE_ASSERT
from schema_mechanism.data_structures import Schema
from schema_mechanism.data_structures import SchemaTree
from schema_mechanism.data_structures import StateAssertion
from schema_mechanism.data_structures import StateElement
from schema_mechanism.util import Observable
from schema_mechanism.util import Observer


# TODO: Move into SchemaMemory as an inner class?
class SchemaMemoryStats:
    def __init__(self):
        self.n_updates = 0


class SchemaMemory(Observer):
    def __init__(self, primitives: Optional[Collection[Schema]] = None) -> None:
        super().__init__()

        self._schema_tree = SchemaTree(primitives)
        self._schema_tree.validate(raise_on_invalid=True)

        # register listeners for primitives
        if primitives:
            for schema in primitives:
                schema.register(self)

        self._stats: SchemaMemoryStats = SchemaMemoryStats()

        # TODO: Need to experiment with multiple (recent) updates. This was not in the original schema mechanism but
        # TODO: seems highly useful in many environments.
        self._as_selections: deque[ActionSelection.SelectionDetails] = deque([], maxlen=1)

    def __len__(self) -> int:
        return self._schema_tree.n_schemas

    def __contains__(self, schema: Schema) -> bool:
        return schema in self._schema_tree

    def __str__(self):
        return str(self._schema_tree)

    @property
    def schemas(self) -> Collection[Schema]:
        return list(itertools.chain.from_iterable([n.schemas for n in self._schema_tree]))

    @property
    def as_selections(self) -> Collection[ActionSelection.SelectionDetails]:
        return self._as_selections

    @staticmethod
    def from_tree(tree: SchemaTree) -> SchemaMemory:
        """ A factory method to initialize a SchemaMemory instance from a SchemaTree.

        Note: This method can be used to initialize SchemaMemory with arbitrary built-in schemas.

        :param tree: a SchemaTree pre-loaded with schemas.
        :return: a tree-initialized SchemaMemory instance
        """
        sm = SchemaMemory()

        sm._schema_tree = tree
        sm._schema_tree.validate(raise_on_invalid=True)

        # register listeners for schemas in tree
        for node in sm._schema_tree:
            for schema in node.schemas:
                schema.register(sm)

        return sm

    @property
    def stats(self) -> SchemaMemoryStats:
        return self._stats

    def update_all(self, result_state: Collection[StateElement]) -> None:
        """ Updates schemas based on results of previous action.

        :param result_state: the state following the execution of the schema's action

        :return: None
        """
        if len(self._as_selections) == 0:
            return

        # TODO: Change this when the mechanism can support updates based on multiple selections
        activation_state, applicable, schema = self._as_selections[0]

        # create previous and current state views
        v_act = ItemPoolStateView(activation_state)
        v_result = ItemPoolStateView(result_state)

        # create new and lost state element collections
        new = new_state(activation_state, result_state)
        lost = lost_state(activation_state, result_state)

        # update global statistics
        self._stats.n_updates += len(applicable)

        # I'm assuming that only APPLICABLE schemas should be updated. This is based on the belief that all
        # of the probabilities within a schema are really conditioned on the context being satisfied, even
        # though this fact is implicit.
        for app in applicable:
            # activated
            if app.action == schema.action:
                app.update(activated=True,
                           v_prev=v_act,
                           v_curr=v_result,
                           new=new,
                           lost=lost)

            # non-activated
            else:
                app.update(activated=False,
                           v_prev=v_act,
                           v_curr=v_result,
                           new=new,
                           lost=lost)

    def all_applicable(self, state: Collection[StateElement]) -> Sequence[Schema]:
        # TODO: Where do I add the items to the item pool???
        # TODO: Where do I create the item state view?

        return list(itertools.chain.from_iterable(n.schemas for n in self._schema_tree.find_all_satisfied(state)))

    def receive(self, *args, **kwargs) -> None:
        source: Schema = kwargs['source']

        if isinstance(source, Schema):
            self._receive_from_schema(schema=source, *args, **kwargs)
        elif isinstance(source, ActionSelection):
            self._receive_from_action_selection(act_select=source, *args, **kwargs)

    def _receive_from_schema(self, schema: Schema, *args, **kwargs) -> None:
        spin_off_type: Schema.SpinOffType = kwargs['spin_off_type']
        relevant_items: Collection[ItemAssertion] = kwargs['relevant_items']

        spin_offs = frozenset([create_spin_off(schema, spin_off_type, ia) for ia in relevant_items])

        # register listeners for spin-offs
        for s in spin_offs:
            s.register(self)

        self._schema_tree.add(schema, spin_offs, spin_off_type)

    def _receive_from_action_selection(self, act_select: ActionSelection, *args, **kwargs):
        selection: ActionSelection.SelectionDetails = kwargs['selection']

        self._as_selections.append(selection)


# TODO: Need to register SchemaMemory as a listener of ActionSelection
class ActionSelection(Observable):
    class SelectionDetails(NamedTuple):
        state: Collection[StateElement]
        applicable: Collection[Schema]
        schema: Schema

    """
        See Drescher, 1991, section 3.4
    """

    def select(self, sm: SchemaMemory, state: Collection[StateElement]) -> Schema:
        applicable = sm.all_applicable(state)

        # TODO: add logic to select a schema
        schema = sample(applicable, k=1)[0]

        # These details are needed by SchemaMemory (for example, to support learning)
        sd = self.SelectionDetails(state, applicable, schema)
        self.notify_all(source=ActionSelection, selection=sd)

        return schema


class SchemaMechanism(Observer, Observable):
    def __init__(self, primitive_actions: Collection[Action], primitive_items: Collection[Item]):
        super().__init__()

        self._primitive_actions = primitive_actions
        self._primitive_items = primitive_items

        self._primitive_schemas = [Schema(action=a) for a in self._primitive_actions]

        self._schema_memory = SchemaMemory(self._primitive_schemas)
        self._action_selection = ActionSelection()

    def receive(self, *args, **kwargs) -> None:
        state: Collection[StateElement] = kwargs['state']

        # learn from results of previous actions (if any)
        self._schema_memory.update_all(state)

        # select schema
        schema = self._action_selection.select(self._schema_memory, state)

        self.notify_all(source=self, selection=schema)


def held_state(s_prev: Collection[StateElement], s_curr: Collection[StateElement]) -> FrozenSet[StateElement]:
    """ Returns the set of state elements that are in both previous and current state

    :param s_prev: a collection of the previous state's elements
    :param s_curr: a collection of the current state's elements

    :return: a set containing state elements shared between current and previous state
    """
    if not all((s_prev, s_curr)):
        return frozenset()

    return frozenset([se for se in s_curr if se in s_prev])


def new_state(s_prev: Optional[Collection[StateElement]],
              s_curr: Optional[Collection[StateElement]]) -> FrozenSet[StateElement]:
    """ Returns the set of state elements that are in current state but not previous

    :param s_prev: a collection of the previous state's elements
    :param s_curr: a collection of the current state's elements

    :return: a set containing new state elements
    """
    if not all((s_prev, s_curr)):
        return frozenset()

    return frozenset([se for se in s_curr if se not in s_prev])


def lost_state(s_prev: Optional[Collection[StateElement]],
               s_curr: Optional[Collection[StateElement]]) -> FrozenSet[StateElement]:
    """ Returns the set of state elements that are in previous state but not current

    :param s_prev: a collection of the previous state's elements
    :param s_curr: a collection of the current state's elements

    :return: a set containing lost state elements
    """
    if not all((s_prev, s_curr)):
        return frozenset()

    return frozenset([se for se in s_prev if se not in s_curr])


def create_spin_off(schema: Schema, spin_off_type: Schema.SpinOffType, item_assert: ItemAssertion) -> Schema:
    """ Creates a context or result spin-off schema that includes the supplied item in its context or result.

    :param schema: the schema from which the new spin-off schema will be based
    :param spin_off_type: a supported Schema.SpinOffType
    :param item_assert: the item assertion to add to the context or result of a spin-off schema

    :return: a spin-off schema based on this one
    """
    if Schema.SpinOffType.CONTEXT == spin_off_type:
        new_context = (
            StateAssertion(item_asserts=(item_assert,))
            if schema.context is NULL_STATE_ASSERT
            else schema.context.replicate_with(item_assert)
        )
        return Schema(action=schema.action,
                      context=new_context,
                      result=schema.result)

    elif Schema.SpinOffType.RESULT == spin_off_type:
        new_result = (
            StateAssertion(item_asserts=(item_assert,))
            if schema.result is NULL_STATE_ASSERT
            else schema.result.replicate_with(item_assert)
        )
        return Schema(action=schema.action,
                      context=schema.context,
                      result=new_result)

    else:
        raise ValueError(f'Unsupported spin-off mode: {spin_off_type}')


def create_context_spin_off(source: Schema, item_assert: ItemAssertion) -> Schema:
    """ Creates a CONTEXT spin-off schema from the given source schema.

    :param source: the source schema
    :param item_assert: the new item assertion to include in the spin-off's context
    :return: a new context spin-off
    """
    return create_spin_off(source, Schema.SpinOffType.CONTEXT, item_assert)


def create_result_spin_off(source: Schema, item_assert: ItemAssertion) -> Schema:
    """ Creates a RESULT spin-off schema from the given source schema.

    :param source: the source schema
    :param item_assert: the new item assertion to include in the spin-off's result
    :return: a new result spin-off
    """
    return create_spin_off(source, Schema.SpinOffType.RESULT, item_assert)


# TODO: Is this function needed???
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
