from __future__ import annotations

import itertools
from collections import deque
from collections.abc import Callable
from collections.abc import Collection
from collections.abc import Sequence
from copy import copy
from typing import NamedTuple
from typing import Optional

import numpy as np

from schema_mechanism.core import Action
from schema_mechanism.core import Assertion
from schema_mechanism.core import Chain
from schema_mechanism.core import CompositeAction
from schema_mechanism.core import CompositeItem
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
from schema_mechanism.core import is_reliable
from schema_mechanism.core import lost_state
from schema_mechanism.core import new_state
from schema_mechanism.share import SupportedFeature
from schema_mechanism.share import debug
from schema_mechanism.share import is_feature_enabled
from schema_mechanism.share import rng
from schema_mechanism.share import trace
from schema_mechanism.util import Observer
from schema_mechanism.util import equal_weights


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

    def __iter__(self) -> Schema:
        yield from itertools.chain.from_iterable([n.schemas_satisfied_by for n in self._schema_tree])

    @staticmethod
    def from_tree(tree: SchemaTree) -> SchemaMemory:
        """ A factory method to initialize a SchemaMemory instance from a SchemaTree.

        Note: This method can be used to initialize SchemaMemory with arbitrary built-in schemas.

        :param tree: a SchemaTree pre-loaded with schemas.
        :return: a tree-initialized SchemaMemory instance
        """
        sm = SchemaMemory()

        sm._schema_tree = copy(tree)
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
                   result_state: State) -> None:
        """ Updates schema statistics based on results of previously selected schema(s).

        Note: While the current implementation only supports updates based on the most recently selected schema, the
        interface supports a sequence of selection details. This is to

        :param selection_details: details corresponding to the most recently selected schema
        :param result_state: the environment state resulting from the selected schema's executed action

        :return: None
        """
        applicable, schema, selection_state, _ = selection_details

        # create new and lost state element collections
        new = new_state(selection_state, result_state)
        lost = lost_state(selection_state, result_state)

        # update global statistics
        self._stats.n_updates += len(applicable)

        # True if state transition explained by activated reliable schema
        #     (See SupportedFeature.ER_SUPPRESS_UPDATE_ON_EXPLAINED)
        explained = False

        # I'm assuming that only APPLICABLE schemas should be updated. This is based on the belief that all
        # of the probabilities within a schema are really conditioned on the context being satisfied, even
        # though this fact is implicit.
        activated_schemas = [s for s in applicable if s.is_activated(schema, result_state)]
        non_activated_schemas = [s for s in applicable if not s.is_activated(schema, result_state)]

        # all activated schemas must be updated BEFORE non-activated.
        for s in activated_schemas:
            # this supports SupportedFeature.ER_SUPPRESS_UPDATE_ON_EXPLAINED
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

        # update composite action controllers
        controllers = CompositeAction.all_satisfied_by(result_state)
        for c in controllers:
            chains = self.backward_chains(c.goal_state)
            c.update(chains)

    def all_applicable(self, state: State) -> Sequence[Schema]:
        satisfied = itertools.chain.from_iterable(
            n.schemas_satisfied_by for n in self._schema_tree.find_all_satisfied(state))

        return [schema for schema in satisfied if schema.is_applicable(state)]

    def receive(self, **kwargs) -> None:
        source: Schema = kwargs['source']

        if isinstance(source, Schema):
            self._receive_from_schema(schema=source, **kwargs)

    def _receive_from_schema(self, schema: Schema, **kwargs) -> None:
        spin_off_type: Schema.SpinOffType = kwargs['spin_off_type']
        relevant_items: Collection[ItemAssertion] = kwargs['relevant_items']

        spin_offs = frozenset([create_spin_off(schema, spin_off_type, ia) for ia in relevant_items])

        # "Whenever a bare schema spawns a spinoff schema, the mechanism determines whether the new schema's result is
        #  novel, as opposed to its already appearing as the result component of some other schema. If the result is
        #  novel, the schema mechanism defines a new composite action with that result as its goal state, it is the
        #  action of achieving that result. The schema mechanism also constructs a bare schema which has that action;
        #  that schema's extended result then can discover effects of achieving the action's goal state"
        #      (See Drescher 1991, p. 90)
        if schema.is_primitive() and (spin_off_type is Schema.SpinOffType.RESULT):
            for spin_off in spin_offs:
                if self.is_novel_result(spin_off.result):
                    trace(f'Novel result detected: {spin_off.result}. Creating new composite action.')

                    # creates and initializes a new composite action
                    ca = CompositeAction(goal_state=spin_off.result)
                    ca.controller.update(self.backward_chains(ca.goal_state))

                    # adds a new bare schema for the new composite action
                    ca_schema = Schema(action=ca)
                    ca_schema.register(self)

                    self._schema_tree.add_primitives([ca_schema])

        # register listeners for spin-offs
        for s in spin_offs:
            s.register(self)

        debug(f'creating spin-offs for schema {str(schema)}: {",".join([str(s) for s in spin_offs])}')

        self._schema_tree.add(schema, spin_offs, spin_off_type)

    def backward_chains(self,
                        goal_state: StateAssertion,
                        max_len: Optional[int] = None,
                        term_states: Optional[Collection[StateAssertion]] = None) -> list[Chain]:
        """ Returns a Collection of chains of RELIABLE schemas in proximity to the given goal state.

        "A chaining schema's result must include the entire context of the next schema in the chain."
            (see Drescher, 1991, p. 100)

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
                if ((s.context != goal_state)
                        and (s.context not in term_states)
                        and (s.result not in term_states)
                        and is_reliable(s)):
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

    def is_novel_result(self, result: StateAssertion) -> bool:
        return not any({result == s.result for s in self._schema_tree.root.schemas_satisfied_by})


# Type aliases
SchemaEvaluationStrategy = Callable[[Sequence[Schema], Optional[Schema]], np.ndarray]
MatchStrategy = Callable[[np.ndarray, float], np.ndarray]
SelectionStrategy = Callable[[Sequence[Schema], np.ndarray], tuple[Schema, float]]


class NoOpEvaluationStrategy:
    def __call__(self, schemas: Sequence[Schema], pending: Optional[Schema]) -> np.ndarray:
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
    def __call__(self, schemas: Sequence[Schema], values: np.ndarray) -> tuple[Schema, float]:
        selection_index = rng().uniform(0, len(schemas))
        return schemas[selection_index], values[selection_index]


class RandomizeBestSelectionStrategy:
    def __init__(self, match: MatchStrategy = None):
        self.eq = match or EqualityMatchStrategy()

    def __call__(self, schemas: Sequence[Schema], values: np.ndarray) -> tuple[Schema, float]:
        max_value = np.max(values)

        # randomize selection if several schemas have values within sameness threshold
        best_schemas = np.argwhere(self.eq(values, max_value)).flatten()
        selection_index = rng().choice(best_schemas, size=1)[0]

        return schemas[selection_index], values[selection_index]


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
def instrumental_values(schemas: Sequence[Schema], pending: Optional[Schema] = None) -> np.ndarray:
    """

        "When the schema mechanism activates a schema as a link in some chain to a positively valued state, then that
         schema's result (or rather, the part of it that includes the next link's context) is said to have instrumental
         value.

         Instrumental value, unlike primitive (and delegated) value, is transient rather than persistent. As
         the state of the world changes, a given state may lie along a chain from the current state to a goal at one
         moment but not the next." (See Drescher 1991, p. 63)

    :param schemas:
    :param pending:

    :return:
    """
    # instrumental value
    ####################

    # TODO: Need to implement backward chain determination to find accessible schemas before
    # TODO: I can implement instrumental value. (See Drescher 1991, Sec. 5.1.2)
    return np.zeros_like(schemas)


# TODO: Add a class description that mentions primitive, delegated, and instrumental value, as well as pending
# TODO: schema bias and reliability.
class GoalPursuitEvaluationStrategy:
    """
        * “The goal-pursuit criterion contributes to a schema’s importance to the extent that the schema’s
           activation helps chain to an explicit top-level goal” (Drescher, 1991, p. 61)
        * "The Schema Mechanism uses only reliable schemas to pursue goals." (Drescher 1987, p. 292)
    """

    def __init__(self):
        # TODO: There are MANY factors that influence value, and it is not clear what their relative weights should be.
        # TODO: It seems that these will need to be parameterized and experimented with to determine the most beneficial
        # TODO: balance between them.
        pass

    def __call__(self, schemas: Sequence[Schema], pending: Optional[Schema] = None) -> np.ndarray:
        pv = primitive_values(schemas)
        dv = delegated_values(schemas)
        iv = instrumental_values(schemas, pending)

        # TODO: Only reliable schemas should be used for goal pursuit. How do we do this??? Perhaps a large penalty
        # TODO: for unreliable schemas???

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

    def __call__(self, schemas: Sequence[Schema], pending: Optional[Schema] = None) -> np.ndarray:
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
    """
        * “The exploration criterion boosts the importance of a schema to promote its activation for the sake of
           what might be learned by that activation” (Drescher, 1991, p. 60)
        * “To strike a balance between goal-pursuit and exploration criteria, the [schema] mechanism alternates
           between emphasizing goal-pursuit criterion for a time, then emphasizing exploration criterion;
           currently, the exploration criterion is emphasized most often (about 90% of the time).”
           (Drescher, 1991, p. 61)
    """

    def __init__(self):
        # TODO: There are MANY factors that influence value, and it is not clear what their relative weights should be.
        # TODO: It seems that these will need to be parameterized and experimented with to determine the most beneficial
        # TODO: balance between them.
        pass

    def __call__(self, schemas: Sequence[Schema], pending: Optional[Schema]) -> np.ndarray:
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

            1.) Schemas compete for selection at each time step (e.g., for each received environmental state)
            2.) Only one schema is selected at a time
            3.) Schemas are chosen based on their "activation importance"
            4.) Activation importance is based on "explicit goal pursuit" and "exploration"
    """

    class SelectionDetails(NamedTuple):
        applicable: Collection[Schema]
        selected: Schema
        selection_state: State
        effective_value: float

    """
        See Drescher, 1991, section 3.4
    """

    def __init__(self,
                 select: Optional[SelectionStrategy] = None,
                 value_strategies: Optional[Collection[SchemaEvaluationStrategy]] = None,
                 weights: Optional[Collection[float]] = None,
                 **kwargs):

        self._select = select or RandomizeBestSelectionStrategy(AbsoluteDiffMatchStrategy(1.0))
        self._values = value_strategies or []
        self._weights = weights or equal_weights(len(self._values))

        if len(self._values) != len(self._weights):
            raise ValueError('Invalid weights. Must have a weight for each evaluation strategy.')

        if not np.isclose(1.0, sum(self._weights)):
            raise ValueError('Evaluation strategy weights must sum to 1.0.')

        # a stack of previously selected, non-terminated schemas with composite actions used for nested invocations
        # of controller components with composite actions
        self._pending_schemas: deque[Schema] = deque()

    @property
    def eval_strategies(self) -> Optional[Collection[SchemaEvaluationStrategy]]:
        return self._values

    @property
    def eval_weights(self) -> float:
        return self._weights

    def select(self, schemas: Sequence[Schema], state: State) -> SchemaSelection.SelectionDetails:
        """ Selects a schema for explicit activation from the supplied list of applicable schemas.

        Note: If a schema with a composite action was PREVIOUSLY selected, it will also compete for selection if any
        of its component schemas are applicable.

        (See Drescher 1991, Section 3.4 for details on the selection algorithm.)

        :param schemas: a list of applicable schemas that are candidates for selection
        :param state: the selection state

        :return: a SelectionDetails object that contains the selected schema and other selection-related information
        """
        if not schemas:
            raise ValueError('Collection of applicable schemas must contain at least one schema')

        # a schema from a composite action that was previously selected for execution
        pending = self._update_pending(state)

        # applicable schemas and their selection values
        candidates = schemas
        candidates_values = self.calc_effective_values(candidates, pending)

        # select a schema for execution. candidates will include components from any pending composite actions.
        selected_schema, value = self._select(candidates, candidates_values)
        debug(f'selected schema: {selected_schema} [eff. value: {value}]')

        # interruption of a pending schema (see Drescher, 1991, p. 61)
        if pending and (selected_schema not in pending.action.controller.components):
            trace(f'pending schema {pending} interrupted: alternate schema chosen {str(selected_schema)}')
            self._pending_schemas.clear()

        # "the [explicit] activation of a schema that has a composite action entails the immediate [explicit] activation
        #  of some component schema..." (See Drescher 1991, p. 60)
        while selected_schema.action.is_composite():
            # adds schema to list of pending composite action schemas
            trace(f'adding pending schema {selected_schema}')
            self._pending_schemas.appendleft(selected_schema)

            applicable_components = [s for s in selected_schema.action.controller.components if s.is_applicable(state)]

            # recursive call to select. (selecting from composite action's applicable components)
            trace(f'selecting component of {selected_schema} [recursive call to select]')
            sd = self.select(applicable_components, state)

            selected_schema = sd.selected
            value = sd.effective_value

        return self.SelectionDetails(applicable=schemas,
                                     selected=selected_schema,
                                     selection_state=state,
                                     effective_value=value)

    def calc_effective_values(self, schemas: Sequence[Schema], pending: Schema) -> np.ndarray:
        return np.sum([w * v(schemas, pending) for w, v in zip(self._weights, self._values)], axis=0)

    @property
    def pending_schema(self) -> Optional[Schema]:
        """ Returns the pending, previously selected, non-terminated schema with a composite action (if one exists).

        :return: the current pending Schema or None if one does not exist.
        """
        return self._pending_schemas[0] if self._pending_schemas else None

    def _update_pending(self, state: State) -> Optional[Schema]:
        next_pending = None
        while self._pending_schemas and not next_pending:
            pending_schema = self._pending_schemas[0]
            action = pending_schema.action
            goal_state = pending_schema.action.goal_state

            # sanity check
            assert action.is_composite()

            if not action.is_enabled(state=state):
                trace(f'pending schema {pending_schema} aborted: no applicable components for state "{state}"')
                self._pending_schemas.popleft()
            elif goal_state.is_satisfied(state=state):
                trace(f'pending schema {pending_schema} completed: goal state {goal_state} reached')
                self._pending_schemas.popleft()
            else:
                next_pending = pending_schema

        return next_pending


# FIXME: Need to rethink the interaction between modules. Should I use observers or let the SchemaMechanism
# FIXME: orchestrate those interactions???
class SchemaMechanism:
    def __init__(self, primitive_actions: Collection[Action], primitive_items: Collection[Item]):
        super().__init__()

        self._primitive_actions = primitive_actions
        self._primitive_items = primitive_items

        self._primitive_schemas = [Schema(action=a) for a in self._primitive_actions]

        self._schema_memory = SchemaMemory(self._primitive_schemas)
        self._schema_selection = SchemaSelection(
            select=RandomizeBestSelectionStrategy(AbsoluteDiffMatchStrategy(1.0)),
            value_strategies=[
                GoalPursuitEvaluationStrategy(),
                EpsilonGreedyExploratoryStrategy(0.9)
            ],
        )

        self._selection_details: Optional[SchemaSelection.SelectionDetails] = None

    @property
    def schema_memory(self) -> SchemaMemory:
        return self._schema_memory

    @property
    def schema_selection(self) -> SchemaSelection:
        return self._schema_selection

    def select(self, state: State, **kwargs) -> Schema:
        # TODO: Would it be better to lazy initialize items into item pool for non-primitive items?
        for se in state:
            _ = ItemPool().get(se)

        # learn from results of previous actions (if any)
        if self._selection_details:
            self._schema_memory.update_all(
                selection_details=self._selection_details,
                result_state=state)

            for item in ReadOnlyItemPool():
                item.update_delegated_value(selection_state=self._selection_details.selection_state,
                                            result_state=state)

        # updates unconditional state value average
        GlobalStats().update_baseline(state)

        # determine all schemas applicable to the current state
        applicable_schemas = self._schema_memory.all_applicable(state)

        # TODO: update goal/explore weights.
        # "The schema mechanism maintains a cyclic balance between emphasizing goal-directed value and exploration
        #  value. The emphasis is achieved by changing the weights of the relative contributions of these components
        #  to the importance asserted by each schema. Goal-directed value is emphasized most of the time, but a
        #  significant part of the time, goal-directed value is diluted so that only very important goals take
        #  precedence over exploration criteria." (See Drescher, 1991, p. 66)

        # select a single schema from the applicable schemas
        self._selection_details = self._schema_selection.select(applicable_schemas, state)

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
