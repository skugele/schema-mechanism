from collections import Sequence
from typing import Optional
from typing import Protocol
from typing import runtime_checkable

import numpy as np

from schema_mechanism.core import Action
from schema_mechanism.core import GlobalStats
from schema_mechanism.core import Schema
from schema_mechanism.core import calc_delegated_value
from schema_mechanism.core import calc_primitive_value
from schema_mechanism.share import GlobalParams
from schema_mechanism.share import debug
from schema_mechanism.share import rng
from schema_mechanism.share import trace
from schema_mechanism.strategies.decay import DecayStrategy
from schema_mechanism.strategies.decay import GeometricDecayStrategy
from schema_mechanism.util import Trace


@runtime_checkable
class EvaluationStrategy(Protocol):
    def __call__(self, schemas: Sequence[Schema], pending: Optional[Schema] = None) -> np.ndarray:
        """ Evaluates the given schema based on this strategy's evaluation criteria.

        :param schemas: the schemas to be evaluated
        :param pending: an optional pending schema

        :return: an array containing the values of the schemas (according to this strategy)
        """


# TODO: Move these into strategies?
class NoOpEvaluationStrategy:
    def __call__(self, schemas: Sequence[Schema], pending: Optional[Schema]) -> np.ndarray:
        return np.zeros_like(schemas)


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
