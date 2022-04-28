from collections import Sequence
from typing import Collection
from typing import Optional
from typing import Protocol
from typing import runtime_checkable

import numpy as np

from schema_mechanism.core import Action
from schema_mechanism.core import Schema
from schema_mechanism.core import calc_delegated_value
from schema_mechanism.core import calc_primitive_value
from schema_mechanism.share import rng
from schema_mechanism.strategies.decay import DecayStrategy
from schema_mechanism.strategies.decay import GeometricDecayStrategy
from schema_mechanism.util import AccumulatingTrace
from schema_mechanism.util import Trace
from schema_mechanism.util import equal_weights


@runtime_checkable
class EvaluationStrategy(Protocol):
    def __call__(self, schemas: Sequence[Schema], pending: Optional[Schema] = None, **kwargs) -> np.ndarray:
        """ Evaluates the given schema based on this strategy's evaluation criteria.

        :param schemas: the schemas to be evaluated
        :param pending: an optional pending schema

        :return: an array containing the values of the schemas (according to this strategy)
        """


class NoOpEvaluationStrategy:
    """ An no-op implementation of the evaluation strategy protocol. Always returns zeros."""

    def __call__(self, schemas: Sequence[Schema], pending: Optional[Schema] = None, **kwargs) -> np.ndarray:
        return np.zeros_like(schemas)


class PrimitiveValueEvaluationStrategy:
    """ An implementation of the evaluation strategy protocol based solely on the primitive values of schemas."""

    def __call__(self, schemas: Sequence[Schema], pending: Optional[Schema] = None, **kwargs) -> np.ndarray:
        return (
            np.array([])
            if schemas is None or len(schemas) == 0
            else np.array([calc_primitive_value(s.result) for s in schemas])
        )


class DelegatedValueEvaluationStrategy:
    """ An implementation of the evaluation strategy protocol based solely on the delegated values of schemas. """

    def __call__(self, schemas: Sequence[Schema], pending: Optional[Schema] = None, **kwargs) -> np.ndarray:
        return (
            np.array([])
            if schemas is None or len(schemas) == 0
            else np.array([calc_delegated_value(s.result) for s in schemas])
        )


class InstrumentalValueEvaluationStrategy:
    """ An implementation of the evaluation strategy protocol based solely on the instrumental values of schemas.

    Drescher on instrumental value:

    "When the schema mechanism activates a schema as a link in some chain to a positively valued state, then
     that schema's result (or rather, the part of it that includes the next link's context) is said to have
     instrumental value.

     Instrumental value, unlike primitive (and delegated) value, is transient rather than persistent. As
     the state of the world changes, a given state may lie along a chain from the current state to a goal at one
     moment but not the next." (See Drescher 1991, p. 63)

    """

    def __call__(self, schemas: Sequence[Schema], pending: Optional[Schema] = None, **kwargs) -> np.ndarray:
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


class PendingFocusEvaluationStrategy:
    """ An implementation of the EvaluationStrategy protocol.

     This implementation initially returns higher values for component schemas of the currently selected pending
     (composite action) schema. However, this value advantage quickly vanishes and becomes aversion if the pending
     schema's goal state fails to obtain after repeated, consecutive selection.
    """

    def __init__(self,
                 max_value: float = None,
                 focus_exp: float = None,
                 decay_strategy: Optional[DecayStrategy] = None) -> None:
        self.max_value = max_value or 1.0

        # TODO: change the implementation to use decay strategy and remove this
        self.focus_exp = focus_exp or 1.5

        # TODO: change the implementations to use this
        self.decay_strategy = decay_strategy

        self._active_pending: Optional[Schema] = None
        self._n: int = 0

    @property
    def max_value(self) -> float:
        return self._max_value

    @max_value.setter
    def max_value(self, value: float) -> None:
        self._max_value = value

    @property
    def decay_strategy(self) -> DecayStrategy:
        return self._decay_strategy

    @decay_strategy.setter
    def decay_strategy(self, value: DecayStrategy) -> None:
        self._decay_strategy = value

    def __call__(self, schemas: Sequence[Schema], pending: Optional[Schema] = None, **kwargs) -> np.ndarray:
        if schemas is None or len(schemas) == 0:
            return np.array([])

        values = np.zeros(len(schemas), dtype=np.float64)
        if not pending:
            self._active_pending = None
            return values

        if pending != self._active_pending:
            self._active_pending = pending
            self._n = 0

        # TODO: changes this to use a decay strategy?
        focus_value = self.max_value - self.focus_exp ** self._n + 1

        # TODO: how can I do this more efficiently?
        for i, schema in enumerate(schemas):
            if schema in self._active_pending.action.controller.components:
                values[i] = focus_value

        self._n += 1

        return values


class ReliabilityEvaluationStrategy:
    """ An implementation of the EvaluationStrategy protocol that penalizes unreliable schemas.

    The values returned will be in the range [0.0, -MAX_PENALTY].

    * Schemas with reliabilities of 1.0 will receive a value of 0.0.
    * Schemas with reliabilities of 0.0 will receive a value of -MAX_PENALTY.
    * Schemas with reliabilities between 0.0 and 1.0 will have negative values that become more negative as their
    reliability approaches 0.0.
    """

    def __init__(self, max_penalty: float = 1.0) -> None:
        self.max_penalty = max_penalty

    @property
    def max_penalty(self) -> float:
        return self._max_penalty

    @max_penalty.setter
    def max_penalty(self, value: float) -> None:
        if value <= 0.0:
            raise ValueError('Max penalty must be > 0.0')
        self._max_penalty = value

    def __call__(self,
                 schemas: Sequence[Schema],
                 pending: Optional[Schema] = None,
                 **kwargs) -> np.ndarray:
        # nans treated as 0.0 reliability
        reliabilities = np.array([0.0 if s.reliability is np.nan else s.reliability for s in schemas])
        return (
            np.array([])
            if schemas is None or len(schemas) == 0
            else self.max_penalty * (np.power(reliabilities, 2) - 1.0)
        )


# TODO: externalize the decay strategy and remove min/initial parameters
class EpsilonGreedyEvaluationStrategy:
    """ An implementation of the EvaluationStrategy protocol based on an epsilon-greedy mechanism.

    The values returned by this strategy will either be 0.0 or np.inf.
    * np.inf is given to schemas chosen at random. All other values returned will be 0.0.
    * At most one schema will be given the value of np.inf per strategy invocation.

    An epsilon parameter is used to determine the probability of random selection.
    * An epsilon of 1.0 indicates that values will always be randomly selected (i.e., one schema will always be given
    the value of np.inf per invocation.
    * An epsilon of 0.0 indicates that values will never be randomly selected (i.e., schemas will always be given
    zero values, resulting in this strategy functioning as a no-op for schema selection purposes).

    An optional decay strategy can be provided to this evaluation strategy to reduce the probability of random
    selections over time.

    See Sutton and Barto, 2018 for more details on epsilon-greedy selection.

    """

    def __init__(self,
                 epsilon: float = None,
                 epsilon_min: float = None,
                 decay_strategy: Optional[DecayStrategy] = None) -> None:
        self.epsilon = epsilon
        self.epsilon_min = 0.0 if epsilon_min is None else epsilon_min
        self.decay_strategy = decay_strategy

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value: float) -> None:
        if value < 0.0 or value > 1.0:
            raise ValueError('Epsilon value must be between 0.0 and 1.0 (exclusive).')
        self._epsilon = value

    @property
    def epsilon_min(self) -> float:
        return self._epsilon_min

    @epsilon_min.setter
    def epsilon_min(self, value: float) -> None:
        if value < 0.0 or value > 1.0:
            raise ValueError('Minimum epsilon value must be between 0.0 and 1.0 (exclusive).')
        self._epsilon_min = value

    @property
    def decay_strategy(self) -> DecayStrategy:
        return self._decay_strategy

    @decay_strategy.setter
    def decay_strategy(self, value: DecayStrategy) -> None:
        self._decay_strategy = value

    def __call__(self, schemas: Sequence[Schema], pending: Optional[Schema] = None) -> np.ndarray:
        if not schemas:
            return np.array([])

        values = np.zeros_like(schemas)

        # TODO: this may be a bad idea. Perhaps randomly selecting from the pending schemas applicable components
        # TODO: would be a better option???

        # bypass epsilon greedy when schemas selected by a composite action's controller
        if pending:
            return values

        # determine if taking exploratory or goal-directed action
        is_exploratory = rng().uniform(0.0, 1.0) < self.epsilon

        # randomly select winning schema if exploratory action and set value to np.inf
        if is_exploratory:
            values[rng().choice(len(schemas))] = np.inf

        # decay epsilon if decay strategy set
        if self.decay_strategy:
            if self.epsilon > self.epsilon_min:
                self.epsilon = max(self.decay_strategy.decay(self.epsilon), self.epsilon_min)

        return values


class HabituationEvaluationStrategy:
    """ An implementation of the EvaluationStrategy protocol.

     This implementation modulates the selection value of actions based on the recency and frequency of their selection.

     The value of schemas with actions that have been chosen many times recently and/or frequently is decreased. This
     implements a form of "habituation" (see Drescher, 1991, Section 3.4.2). The intent of this strategy is to boost
     the value of actions that have been under-represented in recent selections.

     See also:

        (1) "a schema that has recently been activated many times becomes partly suppressed (habituation), preventing
             a small number of schemas from persistently dominating the mechanism's activity."
             (See Drescher, 1991, p. 66.)

        (2) "schemas with underrepresented actions receive enhanced exploration value." (See Drescher, 1991, p. 67.)

    """

    def __init__(self, trace: Trace[Action] = None, multiplier: float = 1.0) -> None:
        self.trace = trace
        self.multiplier = multiplier

        if self.trace is None:
            raise ValueError('Trace cannot be None.')

    @property
    def multiplier(self) -> float:
        return self._multiplier

    @multiplier.setter
    def multiplier(self, value: float) -> None:
        if value <= 0.0:
            raise ValueError('Multiplier must be a positive number.')

        self._multiplier = value

    def __call__(self, schemas: Sequence[Schema], pending: Optional[Schema] = None) -> np.ndarray:
        if not schemas:
            return np.array([])

        actions = [s.action for s in schemas]
        trace_values = self.trace.values[self.trace.indexes(actions)]
        median_trace_value = np.median(trace_values)

        values = -self.multiplier * (trace_values - median_trace_value)

        # FIXME: Is this rounding operation really necessary???
        return np.round(values, 2)


class CompositeEvaluationStrategy:
    """ An implementation of the EvaluationStrategy protocol based on a weighted-sum over other evaluation strategies.
    """

    def __init__(self, strategies: Collection[EvaluationStrategy], weights: Collection[float] = None) -> None:
        self.strategies = strategies
        self.weights = np.array(weights) if weights else equal_weights(len(self.strategies))

    @property
    def strategies(self) -> Collection[EvaluationStrategy]:
        return self._strategies

    @strategies.setter
    def strategies(self, values: Collection[EvaluationStrategy]) -> None:
        self._strategies = values

    @property
    def weights(self) -> np.ndarray:
        return self._weights

    @weights.setter
    def weights(self, values: np.ndarray) -> None:
        if len(self.strategies) != len(values):
            raise ValueError('Invalid weights. Must have a weight for each strategy.')
        if len(values) > 0 and not np.isclose(1.0, sum(values)):
            raise ValueError('Strategy weights must sum to 1.0.')
        if any({w < 0.0 or w > 1.0 for w in values}):
            raise ValueError('Strategy weights must be between 0.0 and 1.0 (inclusive).')

        self._weights = values

    def __call__(self, schemas: Sequence[Schema], pending: Optional[Schema], **kwargs) -> np.ndarray:
        return sum(weight * strategy(schemas=schemas, pending=pending)
                   for weight, strategy in zip(self.weights, self.strategies))


class DefaultExploratoryEvaluationStrategy:
    """ The default exploratory evaluation strategy

        * “The exploration criterion boosts the importance of a schema to promote its activation for the sake of
           what might be learned by that activation” (Drescher, 1991, p. 60)

    """

    def __init__(self,
                 habituation_trace: Trace = None,
                 habituation_multiplier: float = 1.0,
                 epsilon_initial: float = 0.99,
                 epsilon_min: float = 0.2,
                 epsilon_decay_strategy: DecayStrategy = None
                 ):
        # TODO: add an evaluation strategy for the following
        # "a component of exploration value promotes underrepresented levels of actions, where a structure's level is
        # defined as follows: primitive items and actions are of level zero; any structure defined in terms of other
        # structures is of one greater level than the maximum of those structures' levels." (See Drescher, 1991, p. 67)

        self._value_strategy = CompositeEvaluationStrategy(
            strategies=[
                HabituationEvaluationStrategy(
                    trace=habituation_trace or AccumulatingTrace(decay_rate=0.2),
                    multiplier=habituation_multiplier
                ),
                EpsilonGreedyEvaluationStrategy(
                    epsilon=epsilon_initial,
                    decay_strategy=epsilon_decay_strategy or GeometricDecayStrategy(rate=0.95, minimum=epsilon_min)
                )
            ]
        )

    def __call__(self, schemas: Sequence[Schema], pending: Optional[Schema], **kwargs) -> np.ndarray:
        return self._value_strategy(schemas=schemas, pending=pending)


class DefaultGoalPursuitEvaluationStrategy:
    def __init__(self,
                 max_reliability_penalty: float = 1.0,
                 max_pending_focus: float = 1.0,
                 pending_focus_decay_strategy: DecayStrategy = None):
        self.value_strategy = CompositeEvaluationStrategy(
            strategies=[
                PrimitiveValueEvaluationStrategy(),
                DelegatedValueEvaluationStrategy(),
                InstrumentalValueEvaluationStrategy(),
                ReliabilityEvaluationStrategy(max_penalty=max_reliability_penalty),
                PendingFocusEvaluationStrategy(
                    max_value=max_pending_focus,
                    focus_exp=1.0,
                    decay_strategy=pending_focus_decay_strategy or GeometricDecayStrategy
                )
            ]
        )

    def __call__(self, schemas: Sequence[Schema], pending: Optional[Schema] = None, **kwargs) -> np.ndarray:
        return self.value_strategy(schemas=schemas, pending=pending)
