from abc import ABC
from abc import abstractmethod
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
from schema_mechanism.share import debug
from schema_mechanism.share import rng
from schema_mechanism.strategies.decay import DecayStrategy
from schema_mechanism.strategies.decay import ExponentialDecayStrategy
from schema_mechanism.strategies.trace import AccumulatingTrace
from schema_mechanism.strategies.trace import Trace
from schema_mechanism.util import equal_weights


@runtime_checkable
class EvaluationPostProcess(Protocol):
    def __call__(self, schemas: Sequence[Schema], values: np.ndarray) -> np.ndarray:
        """ A callable invoked by evaluation strategies that performs post-evaluation operations.

        Note: these callables can be used to perform diagnostic messaging, value modifications (e.g., scaling), etc.

        :param schemas: the schemas that were evaluated by the strategy
        :param values: the values of the evaluated schemas

        :return: the (potentially modified) schema values as a np.ndarray
        """
        pass


class EvaluationStrategy(ABC):

    def __call__(self,
                 schemas: Sequence[Schema],
                 pending: Optional[Schema] = None,
                 post_process: Sequence[EvaluationPostProcess] = None,
                 **kwargs) -> np.ndarray:
        """ Returns the schemas' values and invokes an optional sequence of post-process callables (if requested).

        :param schemas: the schemas to be evaluated
        :param pending: an optional pending schema
        :param post_process: an optional collection of callables that will be invoked after schema value determination

        :return: an array containing the values of the schemas (according to this strategy)
        """
        values = self.values(schemas=schemas, pending=pending, **kwargs)

        if post_process:
            for operation in post_process:
                values = operation(schemas=schemas, values=values)

        return values

    @abstractmethod
    def values(self, schemas: Sequence[Schema], pending: Optional[Schema] = None, **kwargs) -> np.ndarray:
        """ Evaluates the given schema based on this strategy's evaluation criteria.

        :param schemas: the schemas to be evaluated
        :param pending: an optional pending schema

        :return: an array containing the values of the schemas (according to this strategy)
        """
        pass


class NoOpEvaluationStrategy(EvaluationStrategy):
    """ An no-op implementation of the evaluation strategy protocol. Always returns zeros."""

    def values(self, schemas: Sequence[Schema], pending: Optional[Schema] = None, **kwargs) -> np.ndarray:
        return np.zeros_like(schemas)


class TotalPrimitiveValueEvaluationStrategy(EvaluationStrategy):
    """ An implementation of the evaluation strategy protocol based solely on the primitive values of schemas."""

    def __init__(self) -> None:
        self.calc_schema_values = np.vectorize(self._calc_schema_value, otypes=[float])

    def _calc_schema_value(self, schema: Schema) -> float:
        return sum(item.primitive_value for item in schema.result.items if schema.result)

    def values(self, schemas: Sequence[Schema], pending: Optional[Schema] = None, **kwargs) -> np.ndarray:
        return (
            np.array([])
            if schemas is None or len(schemas) == 0
            else self.calc_schema_values(schemas)
        )


class TotalDelegatedValueEvaluationStrategy(EvaluationStrategy):
    """ An implementation of the evaluation strategy protocol based solely on the delegated values of schemas. """

    def __init__(self) -> None:
        self.calc_schema_values = np.vectorize(self._calc_schema_value, otypes=[float])

    def _calc_schema_value(self, schema: Schema) -> float:
        return sum(item.delegated_value for item in schema.result.items if schema.result)

    def values(self, schemas: Sequence[Schema], pending: Optional[Schema] = None, **kwargs) -> np.ndarray:
        return (
            np.array([])
            if schemas is None or len(schemas) == 0
            else self.calc_schema_values(schemas)
        )


class MaxDelegatedValueEvaluationStrategy(EvaluationStrategy):
    """ An implementation of the evaluation strategy protocol based solely on the delegated values of schemas. """

    def __init__(self) -> None:
        self.calc_schema_values = np.vectorize(self._calc_schema_value, otypes=[float])

    def _calc_schema_value(self, schema: Schema) -> float:
        return max([item.delegated_value for item in schema.result.items if schema.result], default=0.0)

    def values(self, schemas: Sequence[Schema], pending: Optional[Schema] = None, **kwargs) -> np.ndarray:
        return (
            np.array([])
            if schemas is None or len(schemas) == 0
            else self.calc_schema_values(schemas)
        )


class InstrumentalValueEvaluationStrategy(EvaluationStrategy):
    """ An implementation of the evaluation strategy protocol based solely on the instrumental values of schemas.

    Drescher on instrumental value:

    "When the schema mechanism activates a schema as a link in some chain to a positively valued state, then
     that schema's result (or rather, the part of it that includes the next link's context) is said to have
     instrumental value.

     Instrumental value, unlike primitive (and delegated) value, is transient rather than persistent. As
     the state of the world changes, a given state may lie along a chain from the current state to a goal at one
     moment but not the next." (See Drescher 1991, p. 63)

    """

    def values(self, schemas: Sequence[Schema], pending: Optional[Schema] = None, **kwargs) -> np.ndarray:
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


class PendingFocusEvaluationStrategy(EvaluationStrategy):
    """ An implementation of the EvaluationStrategy protocol.

     This implementation initially returns higher values for component schemas of the currently selected pending
     (composite action) schema. However, this value advantage quickly vanishes and becomes aversion if the pending
     schema's goal state fails to obtain after repeated, consecutive selection.
    """

    def __init__(self,
                 max_value: float = 1.0,
                 decay_strategy: Optional[DecayStrategy] = None) -> None:
        self.max_value = max_value
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

    def values(self, schemas: Sequence[Schema], pending: Optional[Schema] = None, **kwargs) -> np.ndarray:
        if schemas is None or len(schemas) == 0:
            return np.array([])

        # zero values (i.e., no pending focus) if pending schema is not set
        if not pending:
            self._active_pending = None
            return np.zeros(len(schemas), dtype=np.float64)

        if pending != self._active_pending:
            self._active_pending = pending
            self._n = 0

        values = self.max_value * np.ones_like(schemas, dtype=np.float64)

        # decay focus value by steps equal to the number of repeated calls for this pending schema
        if self.decay_strategy:
            values = self._decay_strategy.decay(values=values, step_size=self._n)

        for i, schema in enumerate(schemas):
            if schema not in self._active_pending.action.controller.components:
                values[i] = 0.0

        self._n += 1

        return values


class ReliabilityEvaluationStrategy(EvaluationStrategy):
    """ An implementation of the EvaluationStrategy protocol that penalizes unreliable schemas.

    The values returned will be in the range [0.0, -MAX_PENALTY].

    * Schemas with reliabilities of 1.0 will receive a value of 0.0.
    * Schemas with reliabilities of 0.0 will receive a value of -MAX_PENALTY.
    * Schemas with reliabilities between 0.0 and 1.0 will have negative values that become more negative as their
    reliability approaches 0.0.
    """

    def __init__(self, max_penalty: float = 1.0, severity: float = 2.0) -> None:
        self.max_penalty = max_penalty
        self.severity = severity

    @property
    def max_penalty(self) -> float:
        return self._max_penalty

    @max_penalty.setter
    def max_penalty(self, value: float) -> None:
        if value <= 0.0:
            raise ValueError('Max penalty must be > 0.0')
        self._max_penalty = value

    @property
    def severity(self) -> float:
        return self._severity

    @severity.setter
    def severity(self, value: float) -> None:
        if value <= 0.0:
            raise ValueError('Severity must be > 0.0')
        self._severity = value

    def values(self, schemas: Sequence[Schema], pending: Optional[Schema] = None, **kwargs) -> np.ndarray:
        # nans treated as 0.0 reliability
        reliabilities = np.array([0.0 if s.reliability is np.nan else s.reliability for s in schemas])
        return (
            np.array([])
            if schemas is None or len(schemas) == 0
            else self.max_penalty * (np.power(reliabilities, self.severity) - 1.0)
        )


# TODO: externalize the decay strategy and remove min/initial parameters
class EpsilonGreedyEvaluationStrategy(EvaluationStrategy):
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
                 epsilon: float = 0.99,
                 epsilon_min: float = 0.0,
                 decay_strategy: Optional[DecayStrategy] = None) -> None:
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.decay_strategy = decay_strategy

        if self.epsilon_min > self.epsilon:
            raise ValueError('Epsilon min must be less than or equal to epsilon')

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

    def values(self, schemas: Sequence[Schema], pending: Optional[Schema] = None, **kwargs) -> np.ndarray:
        if not schemas:
            return np.array([])

        values = np.zeros_like(schemas)

        # determine if taking exploratory or goal-directed action
        is_exploratory = rng().uniform(0.0, 1.0) < self.epsilon

        if self.decay_strategy:
            self._decay_epsilon()

        # non-exploratory evaluation - all returned values are zeros
        if not is_exploratory:
            return values

        # randomly select winning schema, setting its value to np.inf
        if pending:
            pending_components = pending.action.controller.components
            pending_indexes = [schemas.index(schema) for schema in set(schemas).intersection(pending_components)]

            # if pending, limit choice to pending schema's components
            values[rng().choice(pending_indexes)] = np.inf
        else:
            # bypass epsilon greedy when schemas selected by a composite action's controller
            values[rng().choice(len(schemas))] = np.inf

        return values

    def _decay_epsilon(self) -> None:
        decayed_epsilon = self.decay_strategy.decay(np.array([self.epsilon]))
        self.epsilon = np.maximum(decayed_epsilon, self.epsilon_min)[0]


class HabituationEvaluationStrategy(EvaluationStrategy):
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

    def values(self, schemas: Sequence[Schema], pending: Optional[Schema] = None, **kwargs) -> np.ndarray:
        if not schemas:
            return np.array([])

        actions = [s.action for s in schemas]
        trace_values = self.trace.values[self.trace.indexes(actions)]
        median_trace_value = np.median(trace_values)

        return -self.multiplier * (trace_values - median_trace_value)


class CompositeEvaluationStrategy(EvaluationStrategy):
    """ An implementation of the EvaluationStrategy protocol based on a weighted-sum over other evaluation strategies.
    """

    def __init__(self,
                 strategies: Collection[EvaluationStrategy],
                 weights: Collection[float] = None,
                 strategy_alias: str = None) -> None:
        self.strategies = strategies
        self.weights = np.array(weights) if weights else equal_weights(len(self.strategies))
        self.strategy_alias = strategy_alias or ''

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

    def values(self, schemas: Sequence[Schema], pending: Optional[Schema] = None, **kwargs) -> np.ndarray:
        return sum(weight * strategy(schemas=schemas, pending=pending)
                   for weight, strategy in zip(self.weights, self.strategies))


class DefaultExploratoryEvaluationStrategy(EvaluationStrategy):
    """ The default exploratory evaluation strategy

        * “The exploration criterion boosts the importance of a schema to promote its activation for the sake of
           what might be learned by that activation” (Drescher, 1991, p. 60)

    """

    def __init__(self,
                 habituation_trace: Trace[Action] = None,
                 habituation_multiplier: float = 1.0,
                 epsilon_initial: float = 0.99,
                 epsilon_min: float = 0.2,
                 epsilon_decay_strategy: DecayStrategy = None
                 ):
        """

        :param habituation_trace:
        :param habituation_multiplier:
        :param epsilon_initial:
        :param epsilon_min:
        :param epsilon_decay_strategy:
        """
        # TODO: add an evaluation strategy for the following
        # "a component of exploration value promotes underrepresented levels of actions, where a structure's level is
        # defined as follows: primitive items and actions are of level zero; any structure defined in terms of other
        # structures is of one greater level than the maximum of those structures' levels." (See Drescher, 1991, p. 67)

        default_habituation_trace = (
            AccumulatingTrace(
                active_increment=0.1,
                decay_strategy=ExponentialDecayStrategy(
                    rate=0.5,
                    initial=0.1,
                    minimum=0.0
                )
            )
        ) if not habituation_trace else None

        self.habituation_value_strategy = (
            HabituationEvaluationStrategy(
                trace=habituation_trace or default_habituation_trace,
                multiplier=habituation_multiplier
            ),
        )

        default_epsilon_decay = (
            ExponentialDecayStrategy(rate=0.9999, minimum=epsilon_min)
        )

        self.epsilon_greedy_value_strategy = (
            EpsilonGreedyEvaluationStrategy(
                epsilon=epsilon_initial,
                decay_strategy=epsilon_decay_strategy or default_epsilon_decay
            )
        )

        self._value_strategy = CompositeEvaluationStrategy(
            strategies=[
                self.habituation_value_strategy,
                self.epsilon_greedy_value_strategy
            ]
        )

    def values(self, schemas: Sequence[Schema], pending: Optional[Schema] = None, **kwargs) -> np.ndarray:
        return self._value_strategy(schemas=schemas, pending=pending, **kwargs)


class DefaultGoalPursuitEvaluationStrategy(EvaluationStrategy):
    def __init__(self,
                 reliability_max_penalty: float = 1.0,
                 reliability_severity: float = 0.1,
                 pending_focus_max_value: float = 1.0,
                 pending_focus_decay_strategy: DecayStrategy = None):
        """

        :param reliability_max_penalty:
        :param reliability_severity:
        :param pending_focus_max_value:
        :param pending_focus_decay_strategy:
        """
        self.primitive_value_strategy = TotalPrimitiveValueEvaluationStrategy()
        self.delegated_value_strategy = TotalDelegatedValueEvaluationStrategy()
        self.instrumental_value_strategy = InstrumentalValueEvaluationStrategy()
        self.reliability_value_strategy = ReliabilityEvaluationStrategy(
            max_penalty=reliability_max_penalty,
            severity=reliability_severity)

        default_pending_focus_decay_strategy = ExponentialDecayStrategy(
            rate=0.5,
            initial=1.0,
            minimum=0.0
        )

        self.pending_focus_value_strategy = PendingFocusEvaluationStrategy(
            max_value=pending_focus_max_value,
            decay_strategy=(
                    pending_focus_decay_strategy or
                    default_pending_focus_decay_strategy
            )
        )

        default_pending_focus_decay_strategy = ExponentialDecayStrategy(
            rate=0.5,
            initial=1.0,
            minimum=0.0
        )

        self.value_strategy = CompositeEvaluationStrategy(
            strategies=[
                self.primitive_value_strategy,
                self.delegated_value_strategy,
                self.instrumental_value_strategy,
                self.reliability_value_strategy,
                self.pending_focus_value_strategy
            ]
        )

    def values(self, schemas: Sequence[Schema], pending: Optional[Schema] = None, **kwargs) -> np.ndarray:
        return self.value_strategy(schemas=schemas, pending=pending)


####################
# helper functions #
####################
def display_values(schemas: Sequence[Schema], values: np.ndarray) -> np.ndarray:
    for schema, value in zip(schemas, values):
        debug(f'{schema} [{value}]')

    return values


def display_minmax(schemas: Sequence[Schema], values: np.ndarray) -> np.ndarray:
    max_value = np.max(values)
    max_value_schema = schemas[np.flatnonzero(values == max_value)[0]]
    min_value = np.min(values)
    min_value_schema = schemas[np.flatnonzero(values == min_value)[0]]

    debug(f'MAX: {max_value_schema} [{max_value}]')
    debug(f'MIN: {min_value_schema} [{min_value}]')

    return values
