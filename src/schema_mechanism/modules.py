from __future__ import annotations

import itertools
from collections import Collection
from collections import Sequence
from typing import NamedTuple
from typing import Optional

import numpy as np

from schema_mechanism.data_structures import Action
from schema_mechanism.data_structures import Item
from schema_mechanism.data_structures import ItemAssertion
from schema_mechanism.data_structures import ItemPoolStateView
from schema_mechanism.data_structures import NULL_STATE_ASSERT
from schema_mechanism.data_structures import Schema
from schema_mechanism.data_structures import SchemaTree
from schema_mechanism.data_structures import StateAssertion
from schema_mechanism.data_structures import StateElement
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

    def __len__(self) -> int:
        return self._schema_tree.n_schemas

    def __contains__(self, schema: Schema) -> bool:
        return schema in self._schema_tree

    def __str__(self):
        return str(self._schema_tree)

    @property
    def schemas(self) -> Collection[Schema]:
        return list(itertools.chain.from_iterable([n.schemas for n in self._schema_tree]))

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

    def update_all(self,
                   selection_details: SchemaSelection.SelectionDetails,
                   selection_state: Collection[StateElement],
                   result_state: Collection[StateElement]) -> None:
        """ Updates schema statistics based on results of previously selected schema(s).

        Note: While the current implementation only supports updates based on the most recently selected schema, the
        interface supports a sequence of selection details. This is to

        :param selection_details: details corresponding to the most recently selected schema
        :param selection_state: the environment state that was previously used for schema selection
        :param result_state: the environment state resulting from the selected schema's executed action

        :return: None
        """
        applicable, schema = selection_details

        # create previous and current state views
        v_act = ItemPoolStateView(selection_state)
        v_result = ItemPoolStateView(result_state)

        # create new and lost state element collections
        new = new_state(selection_state, result_state)
        lost = lost_state(selection_state, result_state)

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

    def _receive_from_schema(self, schema: Schema, *args, **kwargs) -> None:
        spin_off_type: Schema.SpinOffType = kwargs['spin_off_type']
        relevant_items: Collection[ItemAssertion] = kwargs['relevant_items']

        spin_offs = frozenset([create_spin_off(schema, spin_off_type, ia) for ia in relevant_items])

        # register listeners for spin-offs
        for s in spin_offs:
            s.register(self)

        self._schema_tree.add(schema, spin_offs, spin_off_type)


class SchemaSelection:
    """ A module responsible for the selection of a schema from a set of applicable schemas.

        The module has the following high-level characteristics:

            1.) Schemas compete for activation at each time step (i.e., for each received environmental state)
            2.) Only one schema is activated at a time
            3.) Schemas are chosen based on their "activation importance"
            4.) Activation importance is based on "explicit goal pursuit" and "exploration"

        Explicit Goal Pursuit
        ---------------------
            * “The goal-pursuit criterion contributes to a schema’s importance to the extent that the schema’s
               activation helps chain to an explicit top-level goal” (Drescher, 1991, p. 61)
            * "The Schema Mechanism uses only reliable schemas to pursue goals." (Drescher 1987, p. 292)

        Exploration
        -----------
            * “The exploration criterion boosts the importance of a schema to promote its activation for the sake of
               what might be learned by that activation” (Drescher, 1991, p. 60)
            * “To strike a balance between goal-pursuit and exploration criteria, the [schema] mechanism alternates
               between emphasizing goal-pursuit criterion for a time, then emphasizing exploration criterion;
               currently, the exploration criterion is emphasized most often (about 90% of the time).”
               (Drescher, 1991, p. 61)

        “A new activation selection occurs at each time unit. Even if a chain of schemas leading to some goal is in
        progress, each next link in the chain must compete for activation. Thus, as with the execution of a composite
        action, control may shift to an unexpected, new, better path to the same goal. Top-level selection carries
        this opportunism one step further; here, control may even shift to a chain that leads instead to a different,
        more important goal.” (Drescher, 1991, p. 61)
    """

    class SelectionDetails(NamedTuple):
        applicable: Collection[Schema]
        selected: Schema

    """
        See Drescher, 1991, section 3.4
    """

    def __init__(self):
        # TODO: Set parameters.

        # TODO: goal/explore weight should be updated over time.
        # the relative weighting given to goal-pursuit vs exploration.
        self._goal_weight: float = 1.0
        self._explore_weight: float = 1.0 - self._goal_weight

        # the absolute difference from the max selection value that counts as being identical to the max
        self._absolute_max_value_diff: float = 0.05

    def select(self, applicable_schemas: Sequence[Schema]) -> SchemaSelection.SelectionDetails:
        if not applicable_schemas:
            raise ValueError('Collection of applicable schemas must contain at least one schema')

        # TODO: There are MANY factors that influence value, and it is not clear what their relative weights should be.
        # TODO: It seems that these will need to be parameterized and experimented with to determine the most beneficial
        # TODO: balance between them.

        #######################
        # Goal-Directed Value #
        #######################

        # Each explicit top-level goal is a state represented by some item, or conjunction of items. Items are
        # designated as corresponding to top-level goals based on their "value"

        # primitive value (associated with the mechanism's built-in primitive items)
        ############################################################################

        # TODO: Should this be limited to reliable schemas???
        # add total primitive value of schema results
        goal_values = np.array([s.result.total_primitive_value for s in applicable_schemas])

        # instrumental value
        ####################

        # TODO: Need to implement backward chain determination to find accessible schemas before
        # TODO: I can implement instrumental value. (See Drescher 1991, Sec. 5.1.2)

        # delegated value
        #################

        # TODO: Need to implement forward chain determination to find accessible schemas before
        # TODO: I can implement delegated value. (See Drescher 1991, Sec. 5.1.2)

        #   "For each item, the schema mechanism computes the value explicitly ACCESSIBLE from the current state--that
        #    is, the maximum value of any items that can be reached by a reliable CHAIN OF SCHEMAS starting with an
        #    applicable schema." (See Drescher 1991, p. 63)

        #   "The mechanism also keeps track of the average accessible value over an extended period of time." (See
        #    Drescher 1991, p. 63)

        #   "For each item, the mechanism keeps track of the average accessible value when the item is On, compared to
        #    when the item is Off. If the accessible value when On tends to exceed the value when Off, the item receives
        #    positive delegated value; if the accessible value when On is less than the value when Off, the item
        #    receives negative delegated value. The magnitude of the delegated value is proportional both to the size of
        #    the discrepancy of the On and Off values, and to the expected duration of the item's being On."
        #    (See Drescher 1991, p. 63)

        #   "For the purposes of the value-delegation comparison, accessible items of zero value count as having slight
        #    positive value, thus delegating more value to states that tend to offer a greater variety of accessible
        #    options." (See Drescher 1991, p. 63)

        # TODO: According to Drescher 1987, only RELIABLE schemas can serve as elements of a plan. Does this mean
        # TODO: that goal-directed behavior can only select from reliable schemas? Or something else??? How does this
        # TODO: affect the valuation of a schema?

        # "The Schema Mechanism uses only reliable schemas to pursue goals." (Drescher 1987, p. 292)

        #####################
        # Exploratory Value #
        #####################

        # TODO: Calculate this!
        explore_values = np.zeros_like(applicable_schemas)

        # TODO: Is selection limited to applicable schemas during exploratory selection?

        # TODO: Add a mechanism for tracking recently activated schemas.
        # hysteresis - "a recently activated schema is favored for activation, providing a focus of attention"

        # TODO: Add a mechanism for tracking frequency of schema activation.
        # "Other factors being equal, a more frequently used schema is favored for selection over a less used schema"
        # (See Drescher, 1991, p. 67)

        # TODO: Add a mechanism for suppressing schema selection for schemas that have been activated too often
        # TODO: recently.

        # TODO: What does "partly suppressed" mean in the quote below? A reduction in their activation importance?

        # habituation - "a schema that has recently been activated many times becomes partly suppressed, preventing
        #                a small number of schemas from persistently dominating the mechanism's activity"

        # TODO: Add a mechanism to track the frequency of activation over schema actions. (Is this purely based on
        # TODO: frequency, or recency as well???
        # "schemas with underrepresented actions receive enhanced exploration value." (See Drescher, 1991, p. 67)

        # TODO: It's not clear what this means? Does this apply to depth of spin-off, nesting of composite actions,
        # TODO: inclusion of synthetic items, etc.???
        # "a component of exploration value promotes underrepresented levels of actions, where a structure's level is
        # defined as follows: primitive items and actions are of level zero; any structure defined in terms of other
        # structures is of one greater level than the maximum of those structures' levels." (See Drescher, 1991, p. 67)

        #####################
        # Composite Actions #
        #####################

        # TODO: What does this mean?
        #   “A composite action is enabled when one of its components is applicable. If a schema is applicable but
        #    its action is not enabled, its selection for activation is inhibited; having a non-enabled action is,
        #    in this respect, similar to having an override condition obtain.” (Drescher, 1991, p.90)

        # 1.) Each link must compete for activation during each selection event
        #
        #   "Even if a chain of schemas leading to some goal is still in progress, each next link in the chain must
        #    compete for activation." (Drescher, 1991, p. 61)

        # 2.) Interruption/Abortion
        #
        #   “The mechanism also permits an executing composite action to be interrupted. Even if a schema with a
        #    composite action is in progress, the cycle of schema selection continues at each next time unit. If the
        #    pending schema is re-selected, its composite action proceeds to select and activate the next component
        #    schema (which may recursively invoke yet another composite action, etc.). If, on the other hand, a schema
        #    other than the pending schema is selected, the pending schema is aborted, its composite action terminated
        #    prematurely.” (Drescher, 1991, p. 61)

        # 3.) Enhanced importance of pending schemas (i.e., in-flight schemas with composite actions)
        #
        #   “The mechanism grants a pending schema enhanced importance for selection, so that the schema will likely
        #    be re-selected until its completion, unless some far more important opportunity arises. Hence, there is
        #    a kind of focus of attention that deters wild thrashing from one never-completed action to another, while
        #    still allowing interruption for a good reason.” (Drescher, 1991, p. 62)

        selection_values = self._goal_weight * goal_values + self._explore_weight * explore_values
        max_value = np.max(selection_values)

        best_schemas = np.argwhere(selection_values >= (max_value - self._absolute_max_value_diff)).flatten()
        selection_index = np.random.choice(best_schemas, size=1)[0]

        return self.SelectionDetails(applicable=applicable_schemas, selected=applicable_schemas[selection_index])


# FIXME: Need to rethink the interaction between modules. Should I use observers or let the SchemaMechanism
# FIXME: orchestrate those interactions???
class SchemaMechanism:
    def __init__(self, primitive_actions: Collection[Action], primitive_items: Collection[Item]):
        super().__init__()

        self._primitive_actions = primitive_actions
        self._primitive_items = primitive_items

        self._primitive_schemas = [Schema(action=a) for a in self._primitive_actions]

        self._schema_memory = SchemaMemory(self._primitive_schemas)
        self._schema_selection = SchemaSelection()

        self._selection_state: Optional[Collection[StateElement]] = None
        self._selection_details: Optional[SchemaSelection.SelectionDetails] = None

    def step(self, state: Collection[StateElement], *args, **kwargs) -> Schema:
        # learn from results of previous actions (if any)
        if self._selection_state and self._selection_details:
            self._schema_memory.update_all(
                selection_details=self._selection_details,
                selection_state=self._selection_state,
                result_state=state)

        # determine all schemas applicable to the current state
        applicable_schemas = self._schema_memory.all_applicable(state)

        # select a single schema from the applicable schemas
        self._selection_details = self._schema_selection.select(applicable_schemas)
        self._selection_state = state

        return self._selection_details.selected


def held_state(s_prev: Collection[StateElement], s_curr: Collection[StateElement]) -> frozenset[StateElement]:
    """ Returns the set of state elements that are in both previous and current state

    :param s_prev: a collection of the previous state's elements
    :param s_curr: a collection of the current state's elements

    :return: a set containing state elements shared between current and previous state
    """
    if not all((s_prev, s_curr)):
        return frozenset()

    return frozenset([se for se in s_curr if se in s_prev])


def new_state(s_prev: Optional[Collection[StateElement]],
              s_curr: Optional[Collection[StateElement]]) -> frozenset[StateElement]:
    """ Returns the set of state elements that are in current state but not previous

    :param s_prev: a collection of the previous state's elements
    :param s_curr: a collection of the current state's elements

    :return: a set containing new state elements
    """
    if not all((s_prev, s_curr)):
        return frozenset()

    return frozenset([se for se in s_curr if se not in s_prev])


def lost_state(s_prev: Optional[Collection[StateElement]],
               s_curr: Optional[Collection[StateElement]]) -> frozenset[StateElement]:
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
