# TODO: GlobalStatistics need to be updated from a method that triggers all ExtendedContext and ExtendedResult updates

# TODO: Need a method to determine all of the ON state items given a state (use item pool to cycle through all
# TODO: items, and keep track of the on ones)

# TODO: might be able to memoize this with a frozen set or something like that.

# TODO: Need a method that determines item relevance


# update schema-level stats (n_action_taken or n_action_not_taken) stats

# Update Procedure for Extended Result

# for each item in the extended context
# if item is ON:
# if item was already ON in previous state, do not update positive-transition statistics

# if item is OFF
# if item was already OFF in previous state, do not update negative-transition statistics


# Update procedure for Extended Context
from collections import Collection
from typing import FrozenSet

from schema_mechanism.data_structures import Context
from schema_mechanism.data_structures import ItemAssertion
from schema_mechanism.data_structures import Result
from schema_mechanism.data_structures import Schema
from schema_mechanism.data_structures import StateElement


class SchemaMemory:
    pass


class SchemaSelection:
    """
        See Drescher, 1991, section 3.4
    """
    pass


def held_state(s_prev: Collection[StateElement], s_curr: Collection[StateElement]) -> FrozenSet[StateElement]:
    """ Returns the set of state elements that are in both previous and current state

    :param s_prev: a collection of the previous state's elements
    :param s_curr: a collection of the current state's elements

    :return: a set containing state elements shared between current and previous state
    """
    if s_curr is None:
        return frozenset()

    return frozenset([se for se in s_curr if se in s_prev])


def new_state(s_prev: Collection[StateElement], s_curr: Collection[StateElement]) -> FrozenSet[StateElement]:
    """ Returns the set of state elements that are in current state but not previous

    :param s_prev: a collection of the previous state's elements
    :param s_curr: a collection of the current state's elements

    :return: a set containing new state elements
    """
    if s_curr is None:
        return frozenset()

    return frozenset([se for se in s_curr if se not in s_prev])


def lost_state(s_prev: Collection[StateElement], s_curr: Collection[StateElement]) -> FrozenSet[StateElement]:
    """ Returns the set of state elements that are in previous state but not current

    :param s_prev: a collection of the previous state's elements
    :param s_curr: a collection of the current state's elements

    :return: a set containing lost state elements
    """
    if s_prev is None:
        return frozenset()

    return frozenset([se for se in s_prev if se not in s_curr])


# TODO: Need a way to suppress the creation of a new spin-off schema when a new relevant item is detected, but that
# TODO: schema already exists. Seems like schema comparisons will be necessary, but maybe there is a better way. Some
# TODO: kind of a graph traversal may also be possible, where the graph contains the "family tree" of schemas


# TODO: My guess is that statistics updates should only occur for non-activated schemas that are applicable. This
# TODO: is based on the assumption that all of the probabilities within a schema are really conditioned on the context
# TODO: being satisfied, even though this fact is implicit.


def create_spin_off(schema: Schema, mode: str, item_assert: ItemAssertion) -> Schema:
    """ Creates a context or result spin-off schema that includes the supplied item in its context or result.

    :param schema: the schema from which the new spin-off schema will be based
    :param mode: "result" (see Drescher, 1991, p. 71) or "context" (see Drescher, 1991, p. 73)
    :param item_assert: the item assertion to add to the context or result of a spin-off schema
    :return: a spin-off schema based on this one
    """
    if "context" == mode:
        new_context = (
            Context(item_asserts=(item_assert,))
            if schema.context is None
            else schema.context.replicate_with(item_assert)
        )
        return Schema(action=schema.action,
                      context=new_context,
                      result=schema.result)

    elif "result" == mode:
        new_result = (
            Result(item_asserts=(item_assert,))
            if schema.result is None
            else schema.result.replicate_with(item_assert)
        )
        return Schema(action=schema.action,
                      context=schema.context,
                      result=new_result)

    else:
        raise ValueError(f'Unknown spin-off mode: {mode}')
