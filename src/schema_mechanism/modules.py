from __future__ import annotations

import itertools
from collections import Iterable
from collections import deque
from collections.abc import Callable
from collections.abc import Collection
from collections.abc import Sequence
from copy import copy
from dataclasses import dataclass
from enum import Enum
from enum import auto
from typing import Any
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
from schema_mechanism.core import Schema
from schema_mechanism.core import SchemaPool
from schema_mechanism.core import SchemaTree
from schema_mechanism.core import SchemaUniqueKey
from schema_mechanism.core import State
from schema_mechanism.core import StateAssertion
from schema_mechanism.core import calc_delegated_value
from schema_mechanism.core import calc_primitive_value
from schema_mechanism.core import is_reliable
from schema_mechanism.core import lost_state
from schema_mechanism.core import new_state
from schema_mechanism.protocols import DecayStrategy
from schema_mechanism.share import GlobalParams
from schema_mechanism.share import SupportedFeature
from schema_mechanism.share import debug
from schema_mechanism.share import is_feature_enabled
from schema_mechanism.share import rng
from schema_mechanism.share import trace
from schema_mechanism.strategies import GeometricDecayStrategy
from schema_mechanism.util import Observer
from schema_mechanism.util import Trace
from schema_mechanism.util import equal_weights


class SchemaMemory(Observer):
    def __init__(self, schemas: Optional[Collection[Schema]] = None) -> None:
        """ Initializes SchemaMemory.

        :param schemas: an optional collection of built-in, bare (action-only) schemas.
        """
        super().__init__()

        # built-in schemas sent to initializer must be bare (action-only) schemas
        self._schema_tree = SchemaTree(schemas) if schemas else None

        # register listeners for built-in schemas
        if schemas:
            for schema in schemas:
                schema.register(self)

    def __len__(self) -> int:
        return self._schema_tree.n_schemas

    def __contains__(self, schema: Schema) -> bool:
        return schema in self._schema_tree

    def __str__(self):
        return str(self._schema_tree)

    def __iter__(self) -> Schema:
        yield from itertools.chain.from_iterable([n.schemas_satisfied_by for n in self._schema_tree])

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, SchemaMemory):
            return all({
                self._schema_tree == other._schema_tree,
            })
        return False if other is None else NotImplemented

    @classmethod
    def from_tree(cls, tree: SchemaTree) -> SchemaMemory:
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

    # TODO: this method is too complicated. I need to refactor it into multiple methods.
    def update_all(self, selection_details: SelectionDetails, result_state: State) -> None:
        """ Updates schema statistics based on results of previously selected schema(s).

        Note: While the current implementation only supports updates based on the most recently selected schema, the
        interface supports a sequence of selection details. This is to

        :param selection_details: details corresponding to the most recently selected schema
        :param result_state: the environment state resulting from the selected schema's executed action

        :return: None
        """
        applicable = selection_details.applicable
        schema = selection_details.selected
        selection_state = selection_details.selection_state

        # create new and lost state element collections
        new = new_state(selection_state, result_state)
        lost = lost_state(selection_state, result_state)

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
            succeeded = schema_succeeded(applicable=True,
                                         activated=True,
                                         satisfied=s.result.is_satisfied(result_state))

            s.update(activated=True,
                     succeeded=succeeded,
                     selection_state=selection_state,
                     new=new,
                     lost=lost,
                     explained=explained)

        for s in non_activated_schemas:
            s.update(activated=False,
                     succeeded=False,  # success requires implicit or explicit activation
                     selection_state=selection_state,
                     new=new,
                     lost=lost,
                     explained=explained)

        # process terminated_pending
        for pending_details in selection_details.terminated_pending:

            pending_schema = pending_details.schema
            pending_selection_state = pending_details.selection_state
            pending_status = pending_details.status

            if pending_status in [PendingStatus.ABORTED, PendingStatus.INTERRUPTED]:
                # TODO: I am not positive this is correct. I may need to revisit the correct way to update these
                # TODO: temporarily extended actions (i.e. composite actions).
                pending_schema.update(
                    activated=True,
                    succeeded=False,  # success requires implicit or explicit activation
                    selection_state=pending_selection_state,
                    new=new_state(pending_selection_state, result_state),
                    lost=lost_state(pending_selection_state, result_state),
                    explained=explained)

        # TODO: there must be a better way to choose when to perform an update than this randomized mess.
        # update composite action controllers
        controllers = CompositeAction.all_satisfied_by(result_state)
        for c in controllers:
            if rng().uniform(0.0, 1.0) < GlobalParams().get('backward_chains.update_frequency'):
                chains = self.backward_chains(c.goal_state, max_len=GlobalParams().get('backward_chains.max_len'))
                c.update(chains)

    def all_applicable(self, state: State) -> Sequence[Schema]:
        satisfied = itertools.chain.from_iterable(
            n.schemas_satisfied_by for n in self._schema_tree.find_all_satisfied(state))

        return [schema for schema in satisfied if schema.is_applicable(state)]

    def receive(self, **kwargs) -> None:
        source: Schema = kwargs['source']

        if isinstance(source, Schema):
            self._receive_from_schema(schema=source, **kwargs)

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

    def _receive_from_schema(self, schema: Schema, **kwargs) -> None:
        spin_off_type: Schema.SpinOffType = kwargs['spin_off_type']
        relevant_items: Collection[ItemAssertion] = kwargs['relevant_items']

        spin_offs = frozenset([create_spin_off(schema, spin_off_type, ia) for ia in relevant_items])

        if schema.is_bare() and (spin_off_type is Schema.SpinOffType.RESULT):
            self._create_composite_action_for_novel_spin_off_results(spin_offs)

        # register listeners for spin-offs
        for s in spin_offs:
            s.register(self)

        debug(f'creating spin-offs for schema {str(schema)}: {",".join([str(s) for s in spin_offs])}')

        self._schema_tree.add(schema, spin_offs, spin_off_type)

    def _create_composite_action_for_novel_spin_off_results(self, result_spin_offs: frozenset[Schema]) -> None:
        """ Creates new composite actions and bare schemas for novel results of result spin-offs.

        "Whenever a bare schema spawns a spinoff schema, the mechanism determines whether the new schema's result is
         novel, as opposed to its already appearing as the result component of some other schema. If the result is
         novel, the schema mechanism defines a new composite action with that result as its goal state, it is the
         action of achieving that result. The schema mechanism also constructs a bare schema which has that action;
         that schema's extended result then can discover effects of achieving the action's goal state"
             (See Drescher 1991, p. 90)

        :param result_spin_offs: a set of result spin-off schemas

        :return: None
        """
        if not is_feature_enabled(SupportedFeature.COMPOSITE_ACTIONS):
            return

        for spin_off in result_spin_offs:
            if self.is_novel_result(spin_off.result):

                # TODO: There must be a better way to limit composite action creation to high value states. This
                # TODO: solution is problematic because the result state's value will fluctuate over time, and
                # TODO: this will permanently prevent the creation of a composite action if the result state
                # TODO: is discovered very early. Note that allowing composite actions for all result states is
                # TODO: not tractable for most environments.
                min_adv = GlobalParams().get('composite_actions.learn.min_baseline_advantage')
                if calc_primitive_value(spin_off.result.as_state()) < GlobalStats().baseline_value + min_adv:
                    continue

                trace(f'Novel result detected: {spin_off.result}. Creating new composite action.')

                # creates and initializes a new composite action
                ca = CompositeAction(goal_state=spin_off.result)
                ca.controller.update(self.backward_chains(ca.goal_state))

                GlobalStats().action_trace.add([ca])

                # adds a new bare schema for the new composite action
                ca_schema = SchemaPool().get(SchemaUniqueKey(action=ca))
                ca_schema.register(self)

                self._schema_tree.add_bare_schemas([ca_schema])


def schema_succeeded(applicable: bool, activated: bool, satisfied: bool) -> bool:
    return applicable and activated and satisfied


# TODO: Move these into protocols?
# Type aliases
SchemaEvaluationStrategy = Callable[[Sequence[Schema], Optional[Schema]], np.ndarray]
MatchStrategy = Callable[[np.ndarray, float], np.ndarray]
SelectionStrategy = Callable[[Sequence[Schema], np.ndarray], tuple[Schema, float]]


# TODO: Move these into strategies?
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
        else np.array([calc_primitive_value(s.result) for s in schemas])
    )


def delegated_values(schemas: Sequence[Schema]) -> np.ndarray:
    return (
        np.array([])
        if schemas is None or len(schemas) == 0
        else np.array([calc_delegated_value(s.result) for s in schemas])
    )


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
    if not schemas and pending:
        raise ValueError('A non-empty sequence of schemas must be provided whenever a pending schema is provided.')

    values = np.zeros_like(schemas, dtype=np.float64)

    # no instrumental value if pending schema not active
    if not pending:
        return values

    if not pending.action.is_composite():
        raise ValueError('Pending schemas must have composite actions.')

    goal_state = pending.action.goal_state
    goal_state_value = calc_primitive_value(goal_state) + calc_delegated_value(goal_state)

    # goal state must have positive value to propagate instrumental value
    if goal_state_value <= 0:
        return values

    controller = pending.action.controller
    for i, schema in enumerate(schemas):
        if schema in controller.components:
            proximity = controller.proximity(schema)
            cost = controller.total_cost(schema)
            values[i] = max(proximity * goal_state_value - cost, 0.0)

    return values


class PendingFocusCalculator:
    """ A functor that returns values intended to provide a temporary selection focus on the current pending schema.
    This value quickly vanishes and becomes aversion if the pending schema's goal state fails to obtain after some
    number of component executions.
    """

    def __init__(self, max_focus: float = None, focus_exp: float = None) -> None:
        self._active_pending: Optional[Schema] = None
        self._n: int = 0

        # TODO: Add global parameters for range of focus values
        self.max_focus = max_focus or 100.0
        self.focus_exp = focus_exp or 1.5

    def __call__(self, schemas: Sequence[Schema], pending: Optional[Schema] = None) -> np.ndarray:
        if schemas is None or len(schemas) == 0:
            return np.array([])

        values = np.zeros(len(schemas), dtype=np.float64)
        if not pending:
            self._active_pending = None
            return values

        if pending != self._active_pending:
            self._active_pending = pending
            self._n = 0

        focus_value = self.max_focus - self.focus_exp ** self._n + 1

        # TODO: how can I do this more efficiently?
        for i, schema in enumerate(schemas):
            if schema in self._active_pending.action.controller.components:
                values[i] = focus_value

        self._n += 1

        return values


pending_focus_values = PendingFocusCalculator()


def reliability_values(schemas: Sequence[Schema],
                       pending: Optional[Schema] = None,
                       max_penalty: Optional[float] = 1.0) -> np.ndarray:
    """ Returns an array of values that serve to penalize unreliable schemas.

    :param schemas:
    :param pending:
    :param max_penalty:

    :return: an array of reliability penalty values
    """
    if max_penalty <= 0.0:
        raise ValueError('Max penalty must be > 0.0')

    # nans treated as 0.0 reliability
    reliabilities = np.array([0.0 if s.reliability is np.nan else s.reliability for s in schemas])
    return (
        np.array([])
        if schemas is None or len(schemas) == 0
        else max_penalty * (np.power(reliabilities, 2) - 1.0)
    )


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
        rv = reliability_values(schemas, pending,
                                max_penalty=GlobalParams().get('goal_pursuit_strategy.reliability.max_penalty'))
        fv = pending_focus_values(schemas, pending)

        trace('goal pursuit selection values:')
        trace(f'\tschemas: {[str(s) for s in schemas]}')
        trace(f'\tpending: {[str(pending)]}')
        trace(f'\tpv: {str(pv)}')
        trace(f'\tdv: {str(dv)}')
        trace(f'\tiv: {str(iv)}')
        trace(f'\trv: {str(rv)}')
        trace(f'\tfv: {str(fv)}')

        return pv + dv + iv + rv + fv


class EpsilonGreedyExploratoryStrategy:
    def __init__(self, epsilon: float = None, decay_strategy: Optional[DecayStrategy] = None):
        self._epsilon = epsilon
        self._decay_strategy = decay_strategy

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

        # bypass epsilon greedy when schemas selected by a composite action's controller
        if pending:
            return values

        # determine if taking exploratory or goal-directed action
        is_exploratory = rng().uniform(0.0, 1.0) < self.epsilon

        # randomly select winning schema if exploratory action and set value to np.inf
        if is_exploratory:
            values[rng().choice(len(schemas))] = np.inf

        # decay epsilon if decay strategy set
        if self._decay_strategy:
            self.epsilon = self._decay_strategy.decay(self.epsilon)

        return values


def habituation_exploratory_value(schemas: Sequence[Schema],
                                  trace: Trace[Action],
                                  multiplier: Optional[float] = None,
                                  pending: Optional[Schema] = None) -> np.ndarray:
    """ Modulates the selection value of actions based on the recency and frequency of their selection.

    The selection value of actions that have been chosen many times recently and/or frequently is decreased
    (implementing "habituation") and boosts the value of actions that have been under-represented in recent selections.

    See also:
        (1) "a schema that has recently been activated many times becomes partly suppressed (habituation), preventing
             a small number of schemas from persistently dominating the mechanism's activity."
             (See Drescher, 1991, p. 66.)

        (2) "schemas with underrepresented actions receive enhanced exploration value." (See Drescher, 1991, p. 67.)

    :param schemas:
    :param trace:
    :param multiplier:
    :param pending:

    :return:
    """
    if trace is None:
        raise ValueError('An action trace must be provided.')

    multiplier = multiplier or 1.0
    if multiplier <= 0.0:
        raise ValueError('Multiplier must be a positive number.')

    if not schemas:
        return np.array([])

    actions = [s.action for s in schemas]
    trace_values = trace.values[trace.indexes(actions)]
    median_trace_value = np.median(trace_values)

    values = -multiplier * (trace_values - median_trace_value)

    return np.round(values, 2)


class ExploratoryEvaluationStrategy:
    """
        * “The exploration criterion boosts the importance of a schema to promote its activation for the sake of
           what might be learned by that activation” (Drescher, 1991, p. 60)

    """

    def __init__(self):
        # TODO: There are MANY factors that influence value, and it is not clear what their relative weights should be.
        # TODO: It seems that these will need to be parameterized and experimented with to determine the most beneficial
        # TODO: balance between them.
        self._eps_greedy = EpsilonGreedyExploratoryStrategy(
            epsilon=GlobalParams().get('random_exploratory_strategy.epsilon.initial'),
            decay_strategy=GeometricDecayStrategy(
                rate=GlobalParams().get('random_exploratory_strategy.epsilon.decay.rate.initial'),
                minimum=GlobalParams().get('random_exploratory_strategy.epsilon.decay.rate.min')))

    def __call__(self, schemas: Sequence[Schema], pending: Optional[Schema]) -> np.ndarray:
        # TODO: Add a mechanism for tracking recently activated schemas.
        # hysteresis - "a recently activated schema is favored for activation, providing a focus of attention"

        # TODO: Add a mechanism for tracking frequency of schema activation.
        # "Other factors being equal, a more frequently used schema is favored for selection over a less used schema"
        # (See Drescher, 1991, p. 67)

        # TODO: It's not clear what this means? Does this apply to depth of spin-off, nesting of composite actions,
        # TODO: inclusion of synthetic items, etc.???
        # "a component of exploration value promotes underrepresented levels of actions, where a structure's level is
        # defined as follows: primitive items and actions are of level zero; any structure defined in terms of other
        # structures is of one greater level than the maximum of those structures' levels." (See Drescher, 1991, p. 67)

        hab_v = habituation_exploratory_value(schemas=schemas,
                                              trace=GlobalStats().action_trace,
                                              multiplier=GlobalParams().get(
                                                  'habituation_exploratory_strategy.multiplier'))

        eps_v = self._eps_greedy(schemas=schemas)

        trace('exploratory selection values:')
        trace(f'\tschemas: {[str(s) for s in schemas]}')
        trace(f'\tpending: {[str(pending)]}')
        trace(f'\thab_v: {str(hab_v)}')
        trace(f'\teps_v: {str(eps_v)}')

        debug(f'epsilon = {self._eps_greedy.epsilon}')

        return hab_v + eps_v


class PendingStatus(Enum):
    IN_PROGRESS = auto()  # indicates that the schema was selected and is still active
    INTERRUPTED = auto()  # indicates that a non-component schema was selected (despite having applicable components)
    ABORTED = auto()  # indicates that no component schemas were applicable
    COMPLETED = auto()  # indicates that the goal state was reached


@dataclass
class PendingDetails:
    schema: Schema
    selection_state: State
    status: PendingStatus = PendingStatus.IN_PROGRESS
    duration: int = 0


@dataclass
class SelectionDetails:
    applicable: Collection[Schema]
    selected: Schema
    terminated_pending: Optional[list[PendingDetails]]
    selection_state: State
    effective_value: float


class SchemaSelection:
    """ Responsible for the selection of schemas based on their situational applicability and value.

    The module has the following high-level characteristics:

        1.) Schemas compete for selection at each time step (e.g., for each received environmental state).
        2.) Only one schema is selected at a time.
        3.) Schemas are chosen based on their goal-pursuit and exploratory value.
        4.) Schemas with composite actions may be selected. If selected, one of their applicable component schemas will
            be immediately selected, and the composite action schema will be added to the pending schema list. Pending
            schemas are schemas with composite action that are previously selected, non-terminated, schemas whose
            action execution is still in progress.
        5.) Pending schemas may be terminated by reaching their goal state (i.e., action COMPLETED), by an alternate
            schema being selected (i.e., action INTERRUPTED), or by having no applicable component schemas (i.e.,
            action ABORTED).

    See Drescher, 1991, section 3.4 for additional details.

    Attributes:

    """

    def __init__(self,
                 select_strategy: Optional[SelectionStrategy] = None,
                 value_strategies: Optional[Collection[SchemaEvaluationStrategy]] = None,
                 weights: Optional[Collection[float]] = None) -> None:
        """ Initializes SchemaSelection based on a set of strategies that define its operation.

        :param select_strategy: a strategy for selecting a single, applicable schema.
        :param value_strategies: a collection of strategies for evaluating schemas.
        :param weights: an optional collection of weights, one for each value strategy (weights must sum to 1.0).
        """
        self.select_strategy: SelectionStrategy = (
                select_strategy or RandomizeBestSelectionStrategy(AbsoluteDiffMatchStrategy(1.0))
        )
        self.value_strategies: Collection[SchemaEvaluationStrategy] = value_strategies or []

        # TODO: Update goal/explore weights. Perhaps this should be a strategy that is passed into the class?
        # "The schema mechanism maintains a cyclic balance between emphasizing goal-directed value and exploration
        #  value. The emphasis is achieved by changing the weights of the relative contributions of these components
        #  to the importance asserted by each schema. Goal-directed value is emphasized most of the time, but a
        #  significant part of the time, goal-directed value is diluted so that only very important goals take
        #  precedence over exploration criteria." (See Drescher, 1991, p. 66)

        # “To strike a balance between goal-pursuit and exploration criteria, the [schema] mechanism alternates
        #  between emphasizing goal-pursuit criterion for a time, then emphasizing exploration criterion; currently,
        #  the exploration criterion is emphasized most often (about 90% of the time).” (See Drescher, 1991, p. 61)
        self.weights = weights or equal_weights(len(self._value_strategies))

        # a stack of previously selected, non-terminated schemas with composite actions used for nested invocations
        # of controller components with composite actions
        self._pending_details: deque[PendingDetails] = deque()

    @property
    def select_strategy(self) -> SelectionStrategy:
        return self._select_strategy

    @select_strategy.setter
    def select_strategy(self, value: SelectionStrategy) -> None:
        self._select_strategy = value

    @property
    def value_strategies(self) -> Collection[SchemaEvaluationStrategy]:
        return self._value_strategies

    @value_strategies.setter
    def value_strategies(self, values: Collection[SchemaEvaluationStrategy]) -> None:
        self._value_strategies = values

    @property
    def weights(self) -> np.ndarray:
        return self._weights

    @weights.setter
    def weights(self, value: np.ndarray) -> None:
        if len(self._value_strategies) != len(value):
            raise ValueError('Invalid weights. Must have a weight for each evaluation strategy.')
        if len(value) > 0 and not np.isclose(1.0, sum(value)):
            raise ValueError('Evaluation strategy weights must sum to 1.0.')
        if any({w < 0.0 or w > 1.0 for w in value}):
            raise ValueError('Evaluation strategy weights must be between 0.0 and 1.0 (inclusive).')

        self._weights = value

    # TODO: This method is way too complicated and needs to be refactored into multiple methods.
    def select(self, schemas: Sequence[Schema], state: State) -> SelectionDetails:
        """ Selects a schema for explicit activation from the supplied list of applicable schemas.

        Note: If a schema with a composite action was PREVIOUSLY selected, it will also compete for selection if any
        of its component schemas are applicable (even if its context is not currently satisfied).

        (See Drescher 1991, Section 3.4 for details on the selection algorithm.)

        :param schemas: a list of applicable schemas that are candidates for selection
        :param state: the selection state

        :return: a SelectionDetails object that contains the selected schema and other selection-related information
        """
        if not schemas:
            raise ValueError('Collection of applicable schemas must contain at least one schema')

        # updates the status of previously selected pending schemas in preparation for selection
        terminated_pending = self._update_pending(state)

        # applicable schemas and their selection values
        candidates = schemas
        candidates_values = self.calc_effective_values(candidates, self.pending_schema)

        debug(f'candidates [n: {len(candidates)}]')
        for c, v in zip(candidates, candidates_values):
            debug(f'\t{str(c)} -> {v:.2f}')

        # select a schema for execution. candidates will include components from any pending composite actions.
        selected_schema, value = self._select_strategy(candidates, candidates_values)
        debug(f'selected schema: {selected_schema} [eff. value: {value}]')

        # interruption of a pending schema (see Drescher, 1991, p. 61)
        if self.pending_schema and (selected_schema not in self.pending_schema.action.controller.components):
            trace(f'pending schema {self.pending_schema} interrupted: alternate schema chosen {selected_schema}')

            # add all interrupted schemas to the completed list
            terminated_pending.extend(self._interrupt_pending(selected_schema))

        # "the [explicit] activation of a schema that has a composite action entails the immediate [explicit] activation
        #  of some component schema..." (See Drescher 1991, p. 60)
        while selected_schema.action.is_composite():
            trace(f'adding pending schema {selected_schema}')

            # adds schema to list of pending composite action schemas
            self._pending_details.appendleft(
                PendingDetails(schema=selected_schema, selection_state=state))

            # TODO: This is ugly. Need to update the action trace for the pending composite action. Not sure where
            # TODO: to do this if not here....
            GlobalStats().action_trace.update([selected_schema.action])

            applicable_components = [s for s in selected_schema.action.controller.components if s.is_applicable(state)]

            # recursive call to select. (selecting from composite action's applicable components)
            trace(f'selecting component of {selected_schema} [recursive call to select]')
            sd = self.select(applicable_components, state)

            selected_schema = sd.selected
            value = sd.effective_value

        return SelectionDetails(applicable=schemas,
                                selected=selected_schema,
                                terminated_pending=terminated_pending,
                                selection_state=state,
                                effective_value=value)

    def calc_effective_values(self, schemas: Sequence[Schema], pending: Optional[Schema] = None) -> np.ndarray:
        if not self._value_strategies:
            return np.zeros_like(schemas)

        return np.sum([w * v(schemas, pending) for w, v in zip(self._weights, self._value_strategies)], axis=0)

    @property
    def pending_schema(self) -> Optional[Schema]:
        """ Returns the pending, previously selected, non-terminated schema with a composite action (if one exists).

        :return: the current pending Schema or None if one does not exist.
        """
        return self._pending_details[0].schema if self._pending_details else None

    def _update_pending(self, state: State) -> list[PendingDetails]:
        terminated_list: list[PendingDetails] = list()

        next_pending: Optional[Schema] = None
        while self._pending_details and not next_pending:

            details = self._pending_details[0]
            action = details.schema.action
            goal_state = action.goal_state

            # TODO: not sure if this is a good way to do this
            max_duration = 1.5 * GlobalParams().get('backward_chains.max_len')

            # sanity check
            assert action.is_composite()

            if not action.is_enabled(state=state):
                trace(f'pending schema {details.schema} aborted: no applicable components for state "{state}"')
                details.status = PendingStatus.ABORTED
                terminated_list.append(details)

                # remove from pending stack (no longer active)
                self._pending_details.popleft()

            elif details.duration > max_duration:
                trace(f'pending schema {details.schema} interrupted: max duration {max_duration} exceeded"')
                details.status = PendingStatus.INTERRUPTED
                terminated_list.append(details)

                # remove from pending stack (no longer active)
                self._pending_details.popleft()

            elif goal_state.is_satisfied(state=state):
                trace(f'pending schema {details.schema} completed: goal state {goal_state} reached')
                details.status = PendingStatus.COMPLETED
                terminated_list.append(details)

                # remove from pending stack (no longer active)
                self._pending_details.popleft()

            else:
                next_pending = details.schema

        # update counts for all active pending schemas
        for details in self._pending_details:
            details.duration += 1

        return terminated_list

    def _interrupt_pending(self, alternate_schema: Schema) -> list[PendingDetails]:
        interrupted_list: list[PendingDetails] = []

        while self._pending_details:
            details = self._pending_details[0]
            components = details.schema.action.controller.components

            # if the chosen (interrupting) schema is a component of a pending schema, continue execution of that
            # pending schema
            if alternate_schema in components:
                break

            # otherwise, remove pending schema from the stack and add it to the interrupted list
            else:
                self._pending_details.popleft()

                interrupted_list.append(details)
                details.status = PendingStatus.INTERRUPTED

        return interrupted_list


class SchemaMechanism:
    def __init__(self, items: Iterable[Item], schema_memory: SchemaMemory, schema_selection: SchemaSelection):
        super().__init__()

        self._schema_memory: SchemaMemory = schema_memory
        self._schema_selection: SchemaSelection = schema_selection
        self._params: GlobalParams = GlobalParams()
        self._stats: GlobalStats = GlobalStats()

        # initialize traces
        built_in_actions = {schema.action for schema in self.schema_memory}
        self._stats.action_trace.add(built_in_actions)

        # pool references (used primarily for serialization)
        self._item_pool: ItemPool = ItemPool()
        self._schema_pool: SchemaPool = SchemaPool()

        # ensure that all primitive items exist in the ItemPool
        for item in items:
            self._item_pool.get(item.source)

    @property
    def schema_memory(self) -> SchemaMemory:
        return self._schema_memory

    @property
    def schema_selection(self) -> SchemaSelection:
        return self._schema_selection

    @property
    def params(self) -> GlobalParams:
        """ Retrieves the global parameters.

        :return: a reference to the GlobalParams
        """
        return self._params

    @property
    def stats(self) -> GlobalStats:
        """ Retrieves the global statistics.

        :return: a reference to the GlobalStats
        """
        return self._stats

    def select(self, state: State, **kwargs) -> SelectionDetails:
        applicable_schemas = self.schema_memory.all_applicable(state)
        return self.schema_selection.select(applicable_schemas, state)

    def learn(self, selection_details: SelectionDetails, result_state: State, **kwargs) -> None:
        self.schema_memory.update_all(selection_details=selection_details, result_state=result_state)

        selected_schema = selection_details.selected
        selection_state = selection_details.selection_state

        self._stats.action_trace.update([selected_schema.action])
        self._stats.delegated_value_helper.update(selection_state=selection_state, result_state=result_state)

        # updates unconditional state value average
        self._stats.n += 1
        self._stats.update_baseline(result_state)


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

        return SchemaPool().get(SchemaUniqueKey(action=schema.action, context=new_context, result=schema.result))

    elif Schema.SpinOffType.RESULT == spin_off_type:
        if not is_feature_enabled(SupportedFeature.ER_INCREMENTAL_RESULTS) and not schema.is_bare():
            raise ValueError('Result spin-off for primitive schemas only (unless ER_INCREMENTAL_RESULTS enabled)')

        new_result = (
            StateAssertion(asserts=(assertion,))
            if schema.result is NULL_STATE_ASSERT
            else Assertion.replicate_with(old=schema.result, new=assertion)
        )

        return SchemaPool().get(SchemaUniqueKey(action=schema.action, context=schema.context, result=new_result))

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
