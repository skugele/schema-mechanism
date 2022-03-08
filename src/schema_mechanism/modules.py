from __future__ import annotations

import itertools
from collections import deque
from collections.abc import Callable
from collections.abc import Collection
from collections.abc import Sequence
from typing import NamedTuple
from typing import Optional

import numpy as np

from schema_mechanism.core import Action
from schema_mechanism.core import Assertion
from schema_mechanism.core import CompositeAction
from schema_mechanism.core import CompositeItem
from schema_mechanism.core import GlobalParams
from schema_mechanism.core import GlobalStats
from schema_mechanism.core import Item
from schema_mechanism.core import ItemAssertion
from schema_mechanism.core import ItemPool
from schema_mechanism.core import NULL_STATE_ASSERT
from schema_mechanism.core import ReadOnlyItemPool
from schema_mechanism.core import Schema
from schema_mechanism.core import SchemaTree
from schema_mechanism.core import State
from schema_mechanism.core import StateAssertion
from schema_mechanism.core import SupportedFeature
from schema_mechanism.core import debug
from schema_mechanism.core import is_feature_enabled
from schema_mechanism.core import is_reliable
from schema_mechanism.core import lost_state
from schema_mechanism.core import new_state
from schema_mechanism.core import rng
from schema_mechanism.core import trace
from schema_mechanism.util import Observer


class SchemaMemoryStats:
    def __init__(self):
        self.n_updates = 0


class SchemaMemory(Observer):
    def __init__(self, primitives: Optional[Collection[Schema]] = None) -> None:
        super().__init__()

        self._schema_tree = SchemaTree(primitives) if primitives else None

        # register listeners for primitives
        if primitives:
            self._schema_tree.validate(raise_on_invalid=True)

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
    def schemas(self) -> list[Schema]:
        return list(itertools.chain.from_iterable([n.schemas_satisfied_by for n in self._schema_tree]))

    # TODO: Add something for the initialization of the SchemaResultTree??? Or should I just move it out
    # TODO: of SchemaMemory entirely?
    @staticmethod
    def from_tree(tree: SchemaTree) -> SchemaMemory:
        """ A factory method to initialize a SchemaMemory instance from a SchemaContextTree.

        Note: This method can be used to initialize SchemaMemory with arbitrary built-in schemas.

        :param tree: a SchemaContextTree pre-loaded with schemas.
        :return: a tree-initialized SchemaMemory instance
        """
        sm = SchemaMemory()

        sm._schema_tree = tree
        sm._schema_tree.validate(raise_on_invalid=True)

        # register listeners for schemas in tree
        for node in sm._schema_tree:
            for schema in node.schemas_satisfied_by:
                schema.register(sm)

        return sm

    @property
    def stats(self) -> SchemaMemoryStats:
        return self._stats

    def update_all(self,
                   selection_details: SchemaSelection.SelectionDetails,
                   selection_state: State,
                   result_state: State) -> None:
        """ Updates schema statistics based on results of previously selected schema(s).

        Note: While the current implementation only supports updates based on the most recently selected schema, the
        interface supports a sequence of selection details. This is to

        :param selection_details: details corresponding to the most recently selected schema
        :param selection_state: the environment state that was previously used for schema selection
        :param result_state: the environment state resulting from the selected schema's executed action

        :return: None
        """
        applicable, schema = selection_details

        # create new and lost state element collections
        new = new_state(selection_state, result_state)
        lost = lost_state(selection_state, result_state)

        # update global statistics
        self._stats.n_updates += len(applicable)

        # state transition explained by activated reliable schema (see SupportedFeature.ER_SUPPRESS_UPDATE_ON_EXPLAINED)
        explained = False

        # I'm assuming that only APPLICABLE schemas should be updated. This is based on the belief that all
        # of the probabilities within a schema are really conditioned on the context being satisfied, even
        # though this fact is implicit.
        activated_schemas = [s for s in applicable if s.action == schema.action]
        non_activated_schemas = [s for s in applicable if s.action != schema.action]

        # all activated schemas must be updated BEFORE non-activated. this supports ER_SUPPRESS_UPDATE_ON_EXPLAINED.
        for s in activated_schemas:
            if s.action == schema.action:
                explained |= True if is_reliable(schema) and schema.predicts_state(result_state) else False

                s.update(activated=True,
                         s_prev=selection_state,
                         s_curr=result_state,
                         new=new,
                         lost=lost,
                         explained=explained)

        for s in non_activated_schemas:
            s.update(activated=False,
                     s_prev=selection_state,
                     s_curr=result_state,
                     new=new,
                     lost=lost,
                     explained=explained)

    def all_applicable(self, state: State) -> Sequence[Schema]:
        return list(
            itertools.chain.from_iterable(n.schemas_satisfied_by for n in self._schema_tree.find_all_satisfied(state)))

    def receive(self, **kwargs) -> None:
        source: Schema = kwargs['source']

        if isinstance(source, Schema):
            self._receive_from_schema(schema=source, **kwargs)

    def _receive_from_schema(self, schema: Schema, **kwargs) -> None:
        spin_off_type: Schema.SpinOffType = kwargs['spin_off_type']
        relevant_items: Collection[ItemAssertion] = kwargs['relevant_items']

        spin_offs = frozenset([create_spin_off(schema, spin_off_type, ia) for ia in relevant_items])

        # register listeners for spin-offs
        for s in spin_offs:
            s.register(self)

        debug(f'creating spin-offs for schema {str(schema)}: {",".join([str(s) for s in spin_offs])}')

        self._schema_tree.add(schema, spin_offs, spin_off_type)

    def backward_chains(self,
                        goal_state: StateAssertion,
                        max_len: Optional[int] = None,
                        term_states: Optional[Collection[StateAssertion]] = None) -> list[Chain]:
        """ Returns a Collection of chains of reliable schemas leading away from the given goal state.

        Used for:
            (1) composite action controllers (see Drescher 1991, Sections 4.3 and 5.1.2)
            (2) propagating instrumental value (see Drescher 1991, Sections 3.4.1 and 5.1.2)

        :param goal_state: a state assertion describing the goal state
        :param max_len: an optional parameter that limits the maximum chain length
        :param term_states: terminal states

        :return: a Collection of Chains
        """
        if not goal_state or goal_state == NULL_STATE_ASSERT:
            return list()

        if max_len is not None and max_len <= 0:
            return list()

        term_states = term_states or set()

        chains = list()

        nodes = self._schema_tree.find_all_would_satisfy(goal_state)
        for n in nodes:
            more_chains = list()
            for s in n.schemas_would_satisfy:
                if (s.context != goal_state) and (s.context not in term_states) and (
                        s.result not in term_states) and is_reliable(s):
                    chains_ = self.backward_chains(
                        goal_state=s.context,
                        max_len=max_len - 1 if max_len else None,
                        term_states={s.result, goal_state, *term_states}
                    )
                    if chains_:
                        for c in chains_:
                            c.append(s)
                    else:
                        chains_.append(Chain([s]))
                    more_chains.extend(chains_)
            chains.extend(more_chains)
        return chains

        # "A chaining schema's result must include the entire context of the next schema in the chain."
        # (see Drescher, 1991, p. 100)

        # retrieve all schemas with result that matches goal state

        # if no matching schemas
        #     return empty collection
        # else
        #     for each goal schema in goal match list
        #         retrieve list of RELIABLE schemas with a result that satisfies the goal schema's context

        # X -> M -> J
        # X -> M -> K
        # X -> M -> L
        # ...
        # X -> N -> ...
        # X -> O -> ...
        # ...
        # Y -> ...
        # Z -> ...

        #    X (C1/A/G)       ...               Y (C2/A/G)       Z (C3/A/G)                [chain(goal=G, max_depth=N)]
        #    M (C11/A/C1)     ...               N (C12/A/C1)     O (C13/A/C1)              [chain(goal=C1, max_depth=N-1)]
        #    J (C111/A/C11)   ...               K (C112/A/C11)   L (C113/A/C11)            [chain(goal=C11, max_depth=N-2)]
        #    ...
        #    No Match For C111 or max_depth == 0                                           [chain(goal=C11...1, max_depth=0)]

        # [[J],[K],[L]]                                                   *** result from chain(goal=C11)
        # [[M,J],[M,K],[M,L]],... [[N,??],[N,??],[N,??]]                  *** result from chain(goal=C1)
        # [[X,M,J],[X,M,K],[X,M,L]],... [[X,N,??],[X,N,??],[X,N,??]]      *** result from chain(goal=G)


# Type aliases
SchemaEvaluationStrategy = Callable[[Sequence[Schema]], np.ndarray]
MatchStrategy = Callable[[np.ndarray, float], np.ndarray]
SelectionStrategy = Callable[[Sequence[Schema], np.ndarray], Schema]


class NoOpEvaluationStrategy:
    def __call__(self, schemas: Sequence[Schema]) -> np.ndarray:
        return np.zeros_like(schemas)


class EqualityMatchStrategy:
    def __call__(self, values: np.ndarray, ref: float) -> np.ndarray:
        return values == ref


class AbsoluteDiffMatchStrategy:
    def __init__(self, max_diff: float):
        # the max absolute difference between values that counts as being identical
        self.max_diff = max_diff

    def __call__(self, values: np.ndarray, ref: float) -> np.ndarray:
        return values >= (ref - self.max_diff)


class RandomizeSelectionStrategy:
    def __call__(self, schemas: Sequence[Schema], values: np.ndarray) -> Schema:
        return rng().choice(schemas, size=1)[0]


class RandomizeBestSelectionStrategy:
    def __init__(self, match: MatchStrategy):
        self.eq = match or EqualityMatchStrategy()

    def __call__(self, schemas: Sequence[Schema], values: np.ndarray) -> Schema:
        max_value = np.max(values)

        # randomize selection if several schemas have values within sameness threshold
        best_schemas = np.argwhere(self.eq(values, max_value)).flatten()
        selection_index = rng().choice(best_schemas, size=1)[0]

        return schemas[selection_index]


def primitive_values(schemas: Sequence[Schema]) -> np.ndarray:
    # Each explicit top-level goal is a state represented by some item, or conjunction of items. Items are
    # designated as corresponding to top-level goals based on their "value"

    # primitive value (associated with the mechanism's built-in primitive items)
    ############################################################################
    return (
        np.array([])
        if schemas is None or len(schemas) == 0
        else np.array([s.result.primitive_value for s in schemas])
    )


def delegated_values(schemas: Sequence[Schema]) -> np.ndarray:
    return (
        np.array([])
        if schemas is None or len(schemas) == 0
        else np.array([s.result.delegated_value for s in schemas])
    )


# TODO: implement this
def instrumental_values(schemas: Sequence[Schema]) -> np.ndarray:
    """

        "When the schema mechanism activates a schema as a link in some chain to a positively valued state, then that
         schema's result (or rather, the part of it that includes the next link's context) is said to have instrumental
         value.

         Instrumental value, unlike primitive (and delegated) value, is transient rather than persistent. As
         the state of the world changes, a given state may lie along a chain from the current state to a goal at one
         moment but not the next." (See Drescher 1991, p. 63)

    :param schemas:
    :return:
    """
    # instrumental value
    ####################

    # TODO: Need to implement backward chain determination to find accessible schemas before
    # TODO: I can implement instrumental value. (See Drescher 1991, Sec. 5.1.2)
    return np.zeros_like(schemas)


class GoalPursuitEvaluationStrategy:
    def __init__(self):
        # TODO: There are MANY factors that influence value, and it is not clear what their relative weights should be.
        # TODO: It seems that these will need to be parameterized and experimented with to determine the most beneficial
        # TODO: balance between them.
        pass

    def __call__(self, schemas: Sequence[Schema]) -> np.ndarray:
        pv = primitive_values(schemas)
        dv = delegated_values(schemas)
        iv = instrumental_values(schemas)

        return pv + dv + iv


class EpsilonGreedyExploratoryStrategy:
    def __init__(self, epsilon: float = None):
        self._epsilon = epsilon

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value: float) -> None:
        if value < 0.0 or value > 1.0:
            raise ValueError('Epsilon value must be between zero and one (exclusive).')
        self._epsilon = value

    def __call__(self, schemas: Sequence[Schema]) -> np.ndarray:
        if not schemas:
            return np.array([])

        values = np.zeros_like(schemas)

        # determine if taking exploratory or goal-directed action
        is_exploratory = rng().uniform(0.0, 1.0) < self._epsilon

        # randomly select winning schema if exploratory action and set value to np.inf
        if is_exploratory:
            values[rng().choice(len(schemas))] = np.inf
        return values


# TODO: implement this
class ExploratoryEvaluationStrategy:
    def __init__(self):
        # TODO: There are MANY factors that influence value, and it is not clear what their relative weights should be.
        # TODO: It seems that these will need to be parameterized and experimented with to determine the most beneficial
        # TODO: balance between them.
        pass

    def __call__(self, schemas: Sequence[Schema]) -> np.ndarray:
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

        # FIXME
        return rng().integers(0, 100, len(schemas))


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

    def __init__(self,
                 goal_pursuit: SchemaEvaluationStrategy = None,
                 explore: SchemaEvaluationStrategy = None,
                 select: SelectionStrategy = None,
                 goal_weight: float = None,
                 explore_weight: float = None):

        self._goal_pursuit = goal_pursuit or GoalPursuitEvaluationStrategy()
        self._explore = explore or EpsilonGreedyExploratoryStrategy(0.9)
        self._select = select or RandomizeBestSelectionStrategy(AbsoluteDiffMatchStrategy(1.0))

        self._goal_weight = goal_weight or GlobalParams().get('goal_weight')
        self._explore_weight = explore_weight or GlobalParams().get('explore_weight')

        if self._goal_weight + self._explore_weight != 1.0:
            raise ValueError('Goal-pursuit and exploratory weights must sum to 1.0.')

    @property
    def goal_weight(self) -> float:
        return self._goal_weight

    @property
    def explore_weight(self) -> float:
        return self._explore_weight

    def select(self, schemas: Sequence[Schema]) -> SchemaSelection.SelectionDetails:
        if not schemas:
            raise ValueError('Collection of applicable schemas must contain at least one schema')

        # TODO: update goal/explore weights.
        # "The schema mechanism maintains a cyclic balance between emphasizing goal-directed value and exploration
        #  value. The emphasis is achieved by changing the weights of the relative contributions of these components
        #  to the importance asserted by each schema. Goal-directed value is emphasized most of the time, but a
        #  significant part of the time, goal-directed value is diluted so that only very important goals take
        #  precedence over exploration criteria." (See Drescher, 1991, p. 66)

        goal_values = self._goal_pursuit(schemas)
        explore_values = self._explore(schemas)

        trace(f'selection applicable schemas: {", ".join([str(s) for s in schemas])}')
        trace(f'selection goal-pursuit values: {goal_values}')
        trace(f'selection exploratory values: {explore_values}')

        # TODO: Only reliable schemas should be used for goal pursuit. How do we do this??? Perhaps a large penalty
        # TODO: for unreliable schemas???
        selection_values = self.goal_weight * goal_values + self.explore_weight * explore_values

        trace(f'weighted selection values: {selection_values}')

        # debug('selection values: ')
        # for s, v in zip(schemas, selection_values):
        #     debug(f'{s}:{float(v):.2f}')

        # TODO: Need to increase the selection value for pending composite actions
        # “The mechanism grants a pending schema enhanced importance for selection, so that the schema will likely
        #  be re-selected until its completion, unless some far more important opportunity arises. Hence, there is
        #  a kind of focus of attention that deters wild thrashing from one never-completed action to another, while
        #  still allowing interruption for a good reason.” (Drescher, 1991, p. 62)

        # "A new activation selection occurs at each time unit. Even if a chain of schemas leading to some goal is
        #  still in progress, each next link in the chain must compete for activation." (Drescher, 1991, p. 61)
        selected_schema = self._select(schemas, selection_values)

        debug(f'selected schema: {selected_schema}')

        # TODO: if the selected schema contains a composite action, we must do some additional work here
        if isinstance(selected_schema.action, CompositeAction):
            selected_schema = self._process_composite_action(selected_schema)

        return self.SelectionDetails(applicable=schemas, selected=selected_schema)

    def _process_composite_action(self, schema: Schema) -> Schema:
        #####################
        # Composite Actions #
        #####################

        # TODO: What does this mean?
        #   “A composite action is enabled when one of its components is applicable. If a schema is applicable but
        #    its action is not enabled, its selection for activation is inhibited; having a non-enabled action is,
        #    in this respect, similar to having an override condition obtain.” (Drescher, 1991, p.90)

        # TODO: According to Drescher 1987, only RELIABLE schemas can serve as elements of a plan. I suspect this
        # TODO: relates to composite actions.
        # TODO: "The Schema Mechanism uses only reliable schemas to pursue goals." (Drescher 1987, p. 292)

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
        return schema


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

        self._selection_state: Optional[State] = None
        self._selection_details: Optional[SchemaSelection.SelectionDetails] = None

    @property
    def schema_memory(self) -> SchemaMemory:
        return self._schema_memory

    @property
    def schema_selection(self) -> SchemaSelection:
        return self._schema_selection

    @property
    def known_schemas(self) -> Collection[Schema]:
        return self._schema_memory.schemas

    def select(self, state: State, **kwargs) -> Schema:
        # TODO: Would it be better to lazy initialize items into item pool for non-primitive items?
        for se in state:
            _ = ItemPool().get(se)

        # learn from results of previous actions (if any)
        if self._selection_state and self._selection_details:
            self._schema_memory.update_all(
                selection_details=self._selection_details,
                selection_state=self._selection_state,
                result_state=state)

            for item in ReadOnlyItemPool():
                item.update_delegated_value(selection_state=self._selection_state,
                                            result_state=state)

        # updates unconditional state value average
        GlobalStats().update_baseline(state)

        # determine all schemas applicable to the current state
        applicable_schemas = self._schema_memory.all_applicable(state)

        # select a single schema from the applicable schemas
        self._selection_details = self._schema_selection.select(applicable_schemas)
        self._selection_state = state

        return self._selection_details.selected


def create_spin_off(schema: Schema, spin_off_type: Schema.SpinOffType, assertion: ItemAssertion) -> Schema:
    """ Creates a context or result spin-off schema that includes the supplied item in its context or result.

    :param schema: the schema from which the new spin-off schema will be based
    :param spin_off_type: a supported Schema.SpinOffType
    :param assertion: an assertion to add to the context or result of a spin-off schema

    :return: a spin-off schema based on this one
    """
    if not schema:
        ValueError('Schema must not be None')

    if not assertion or len(assertion) == 0:
        ValueError('Assertion must not be None or empty')

    if Schema.SpinOffType.CONTEXT == spin_off_type:
        new_context = (
            StateAssertion(asserts=(assertion,))
            if schema.context is NULL_STATE_ASSERT
            else Assertion.replicate_with(old=schema.context, new=assertion)
        )

        # add composite contexts to ItemPool to support learning of composite results
        if not is_feature_enabled(SupportedFeature.ER_INCREMENTAL_RESULTS):
            if len(new_context) > 1:
                _ = ItemPool().get(new_context, item_type=CompositeItem)

        return Schema(action=schema.action, context=new_context, result=schema.result)

    elif Schema.SpinOffType.RESULT == spin_off_type:
        if not is_feature_enabled(SupportedFeature.ER_INCREMENTAL_RESULTS) and not schema.is_primitive():
            raise ValueError('Result spin-off for primitive schemas only (unless ER_INCREMENTAL_RESULTS enabled)')

        new_result = (
            StateAssertion(asserts=(assertion,))
            if schema.result is NULL_STATE_ASSERT
            else Assertion.replicate_with(old=schema.result, new=assertion)
        )

        return Schema(action=schema.action, context=schema.context, result=new_result)

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


class ChainNode:
    def __init__(self, schema: Schema, proximity: int):
        self._schema = schema
        self._proximity = proximity

    @property
    def schema(self) -> Schema:
        return self._schema

    @property
    def proximity(self) -> int:
        return self._proximity


class Chain(deque):
    def __str__(self):
        return '->'.join([str(link) for link in self])

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(tuple(self))


# TODO: Implement this
def forward_chains(schema: Schema, depth: int, accept: Callable[[Schema], bool]) -> Collection[Chain]:
    # used to determine which states are ACCESSIBLE from the current state

    # ACCESSIBILITY determination:
    #    "To begin, each schema that is currently applicable broadcasts a message via its extended result
    #     to the items and conjunctions that are included in the schema's result. Any schema that has such an item or
    #     conjunction as its context broadcasts in turn via its own extended result, and so on, to some maximum depth
    #     of search. Any item or conjunction of items that receives a message by this process is currently accessible."
    #     (See Drescher 1991, p. 101)

    # TODO: seems like I need a breadth first graph traversal
    pass
