from __future__ import annotations

import itertools
from collections import Iterable
from collections import deque
from collections.abc import Collection
from collections.abc import Sequence
from copy import copy
from dataclasses import dataclass
from enum import Enum
from enum import auto
from typing import Any
from typing import Optional

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
from schema_mechanism.core import calc_primitive_value
from schema_mechanism.core import is_reliable
from schema_mechanism.core import lost_state
from schema_mechanism.core import new_state
from schema_mechanism.share import GlobalParams
from schema_mechanism.share import SupportedFeature
from schema_mechanism.share import debug
from schema_mechanism.share import is_feature_enabled
from schema_mechanism.share import rng
from schema_mechanism.share import trace
from schema_mechanism.strategies.evaluation import EvaluationStrategy
from schema_mechanism.strategies.evaluation import NoOpEvaluationStrategy
from schema_mechanism.strategies.selection import RandomizeBestSelectionStrategy
from schema_mechanism.strategies.selection import SelectionStrategy
from schema_mechanism.util import Observer


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
        2.) Only one schema is selected at a time (based on a selection strategy).
        3.) Schemas are chosen based on their selection importance (quantified by an evaluation strategy).
        4.) Schemas with composite actions may be selected. If selected, one of their applicable component schemas will
            be immediately selected and returned. Note that schema's containing composite actions are never returned
            from the select method, only their components schemas with non-composite actions.
        5.) Selected composite action schema whose execution has not yet terminated will be designed as "pending
            schemas" and added to the class's pending schema queue.
        6.) Pending schemas may be terminated by reaching their goal state (i.e., action COMPLETED), by an alternate
            schema being selected (i.e., action INTERRUPTED), or by having no applicable component schemas (i.e.,
            action ABORTED). Once terminated, they will be removed from the class's pending schema queue.

    See Drescher, 1991, section 3.4 for additional details.

    """

    def __init__(self,
                 select_strategy: SelectionStrategy = None,
                 evaluation_strategy: EvaluationStrategy = None) -> None:
        """ Initializes SchemaSelection based on a set of strategies that define its operation.

        :param select_strategy: a strategy for selecting a single schema from a set of applicable schemas
        :param evaluation_strategy: a strategy for calculating the selection importance of applicable schemas
        """
        self.select_strategy: SelectionStrategy = select_strategy or RandomizeBestSelectionStrategy()
        self.evaluation_strategy: EvaluationStrategy = evaluation_strategy or NoOpEvaluationStrategy()

        # a stack of previously selected, non-terminated schemas with composite actions used for nested invocations
        # of controller components with composite actions
        self._pending_schemas_stack: deque[PendingDetails] = deque()

    def select(self, schemas: Sequence[Schema], state: State) -> SelectionDetails:
        """ Selects a schema for explicit activation from the supplied list of applicable schemas.

        Note: If a schema with a composite action was PREVIOUSLY selected, it will also compete for selection if any
        of its component schemas are applicable (even if its context is not currently satisfied).

        See Drescher 1991, Section 3.4 for details on the selection algorithm.

        :param schemas: a list of applicable schemas that are candidates for selection
        :param state: the selection state

        :return: a SelectionDetails object that contains the selected schema and other selection-related information
        """
        if not schemas:
            raise ValueError('Collection of applicable schemas must contain at least one schema')

        # checks status of any pending (composite action) schemas that were previously selected
        terminated_pending_list: list[PendingDetails] = (
            []
            if not self.pending_schema
            else self._update_pending_schemas(state)
        )

        # evaluate and select exactly one schema
        # (note: applicable schemas may contain components of previously selected pending schemas)
        schemas_values = self.evaluation_strategy(schemas, self.pending_schema)
        selected_schema, selected_value = self.select_strategy(schemas, schemas_values)

        # TODO: I'd like to externalize this pending interrupted check, but it will require some work.
        # check if selected schema interrupts a pending schema's execution
        if self.is_pending_interrupted_by_selected_schema(selected_schema):
            interrupted_pending: list[PendingDetails] = self._interrupt_pending(selected_schema)
            terminated_pending_list.extend(interrupted_pending)

        # if the selected schema has a composite action, then immediate select one of its components
        if selected_schema.action.is_composite():
            selected_schema, selected_value = self.select_component_schema(selected_schema, selected_value, state)

        return SelectionDetails(applicable=schemas,
                                selected=selected_schema,
                                terminated_pending=terminated_pending_list,
                                selection_state=state,
                                effective_value=selected_value)

    def is_pending_interrupted_by_selected_schema(self, selected_schema: Schema) -> bool:
        """ Checks if the selected schema interrupts the current pending schema's execution.

        A pending schema is interrupted if a selected schema is not among the pending schema's action's components.
        (See Drescher, 1991, p. 61.)

        :param selected_schema: the selected schema

        :return: True if pending schema is interrupted; False otherwise.
        """
        return self.pending_schema and (selected_schema not in self.pending_schema.action.controller.components)

    def select_component_schema(self,
                                selected_schema: Schema,
                                selected_value: float,
                                selection_state: State) -> tuple[Schema, float]:
        """ Selects a component schema with a non-composite action from a selected, composite action schema.

        "the [explicit] activation of a schema that has a composite action entails the immediate [explicit] activation
         of some component schema" (See Drescher 1991, p. 60)

        :param selected_schema: a selected (composite action) schema
        :param selected_value: the current value of the selected (composite action) schema
        :param selection_state: the selected schema's selection state

        :return: a tuple containing the selected component schema and its value
        """
        schema: Schema = selected_schema
        value: float = selected_value

        # recursively select from composite action schema components until a schema with a primitive action is found.
        while schema and schema.action.is_composite():
            # TODO: I'd like to externalize this update to the pending schemas queue, but it will take some work.
            # adds this composite action schema to the list of pending composite action schemas
            self._pending_schemas_stack.appendleft(PendingDetails(schema=schema, selection_state=selection_state))

            applicable_components = [s for s in schema.action.controller.components if s.is_applicable(selection_state)]

            components_values = self.evaluation_strategy(applicable_components, self.pending_schema)
            schema, value = self.select_strategy(applicable_components, components_values)

        return schema, value

    @property
    def pending_schema(self) -> Optional[Schema]:
        """ Returns the pending, previously selected, non-terminated schema with a composite action (if one exists).

        :return: the current pending Schema or None if one does not exist.
        """
        return self._pending_schemas_stack[0].schema if self._pending_schemas_stack else None

    def _update_pending_schemas(self, selection_state: State) -> list[PendingDetails]:
        """ Updates the status of previously selected pending schemas and returns a list of terminated pending schemas.

        :param selection_state: the selection state

        :return: a list of PendingDetails for any terminated pending schemas
        """
        terminated_list: list[PendingDetails] = list()

        next_pending: Optional[Schema] = None
        while self._pending_schemas_stack and not next_pending:

            details = self._pending_schemas_stack[0]
            action = details.schema.action
            goal_state = action.goal_state

            # TODO: not sure if this is a good way to do this
            max_duration = 1.5 * GlobalParams().get('backward_chains.max_len')

            # sanity check
            assert action.is_composite()

            if not action.is_enabled(state=selection_state):
                trace(
                    f'pending schema {details.schema} aborted: no applicable components for state "{selection_state}"')
                details.status = PendingStatus.ABORTED
                terminated_list.append(details)

                # remove from pending stack (no longer active)
                self._pending_schemas_stack.popleft()

            elif details.duration > max_duration:
                trace(f'pending schema {details.schema} interrupted: max duration {max_duration} exceeded"')
                details.status = PendingStatus.INTERRUPTED
                terminated_list.append(details)

                # remove from pending stack (no longer active)
                self._pending_schemas_stack.popleft()

            elif goal_state.is_satisfied(state=selection_state):
                trace(f'pending schema {details.schema} completed: goal state {goal_state} reached')
                details.status = PendingStatus.COMPLETED
                terminated_list.append(details)

                # remove from pending stack (no longer active)
                self._pending_schemas_stack.popleft()

            else:
                next_pending = details.schema

        # update counts for all active pending schemas
        for details in self._pending_schemas_stack:
            details.duration += 1

        return terminated_list

    def _interrupt_pending(self, selected_schema: Schema) -> list[PendingDetails]:
        interrupted_list: list[PendingDetails] = []

        while self._pending_schemas_stack:
            details = self._pending_schemas_stack[0]
            components = details.schema.action.controller.components

            # if selected schema is a component of a pending schema, continue execution of that pending schema
            if selected_schema in components:
                break

            # otherwise, remove pending schema from the stack and add it to the interrupted list
            else:
                self._pending_schemas_stack.popleft()

                interrupted_list.append(details)
                details.status = PendingStatus.INTERRUPTED

        return interrupted_list


class SchemaMechanism:
    def __init__(self,
                 items: Iterable[Item],
                 schema_memory: SchemaMemory,
                 schema_selection: SchemaSelection,
                 global_params: GlobalParams = None,
                 global_stats: GlobalStats = None):
        super().__init__()

        self._schema_memory: SchemaMemory = schema_memory
        self._schema_selection: SchemaSelection = schema_selection
        self._params: GlobalParams = global_params or GlobalParams()
        self._stats: GlobalStats = global_stats or GlobalStats()

        # initialize traces
        # TODO: Move this initialization outside of SchemaMechanism
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

        # FIXME: all of these action trace updates could be moved into the action trace update method!
        self._stats.action_trace.update([selected_schema.action])

        # updates action trace to include the composite actions of previously selected, terminated pending schemas
        for pending_details in selection_details.terminated_pending:
            pending_schema = pending_details.schema
            self._stats.action_trace.update([pending_schema.action])

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
