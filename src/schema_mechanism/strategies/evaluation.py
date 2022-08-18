import logging
from abc import ABC
from abc import abstractmethod
from collections import Sequence
from typing import Optional
from typing import Protocol
from typing import runtime_checkable

import numpy as np

from schema_mechanism.core import Schema
from schema_mechanism.core import calc_delegated_value
from schema_mechanism.core import calc_primitive_value
from schema_mechanism.core import get_action_trace
from schema_mechanism.strategies.decay import DecayStrategy
from schema_mechanism.strategies.decay import GeometricDecayStrategy
from schema_mechanism.strategies.scaling import ScalingStrategy
from schema_mechanism.strategies.scaling import SigmoidScalingStrategy
from schema_mechanism.strategies.weight_update import CyclicWeightUpdateStrategy
from schema_mechanism.strategies.weight_update import NoOpWeightUpdateStrategy
from schema_mechanism.strategies.weight_update import WeightUpdateStrategy
from schema_mechanism.util import equal_weights
from schema_mechanism.util import repr_str
from schema_mechanism.util import rng

logger = logging.getLogger(__name__)


@runtime_checkable
class EvaluationPostProcess(Protocol):
    def __call__(self,
                 schemas: Optional[Sequence[Schema]],
                 pending: Optional[Schema],
                 values: Optional[np.ndarray]) -> np.ndarray:
        """ A callable invoked by evaluation strategies that performs post-evaluation operations.

        Note: these callables can be used to perform diagnostic messaging, value modifications (e.g., scaling), etc.

        :param schemas: an optional sequence of schemas that were evaluated by the strategy
        :param pending: an optional pending schema
        :param values: an optional array of values for the evaluated schemas

        :return: the (potentially modified) schema values as a np.ndarray
        """
        pass


class EvaluationStrategy(ABC):

    def __call__(self,
                 schemas: Sequence[Schema],
                 pending: Optional[Schema] = None,
                 post_process: Sequence[EvaluationPostProcess] = None
                 ) -> np.ndarray:
        """ Returns the schemas' values and invokes an optional sequence of post-process callables (if requested).

        :param schemas: the schemas to be evaluated
        :param pending: an optional pending schema
        :param post_process: an optional collection of callables that will be invoked after schema value determination

        :return: an array containing the values of the schemas (according to this strategy)
        """

        # determines the values of the given schemas (may be based on internal state maintained by strategy)
        values = self.values(schemas=schemas, pending=pending)

        if post_process:
            class_name = self.__class__.__name__
            logger.debug(f'*** {class_name} Post Process Results ***')
            for i, operation in enumerate(post_process, start=1):
                operation_name = (
                    operation.__name__ or operation.__class__.__name__
                    if hasattr(operation, '__name__') or hasattr(operation.__class__, '__name')
                    else ''
                ).upper()
                logger.debug(f'Post-Process Callable {i} "{operation_name}":')
                values = operation(schemas=schemas, pending=pending, values=values)

        # updates strategy's internal state AFTER schema value determination. (This may influence later evaluations.)
        self.update(schemas=schemas, pending=pending)

        return values

    def __repr__(self):
        return repr_str(obj=self, attr_values=dict())

    @abstractmethod
    def values(self, schemas: Sequence[Schema], pending: Optional[Schema] = None) -> np.ndarray:
        """ Evaluates the given schema based on this strategy's evaluation criteria.

        Note: This method should be idempotent until the update method is invoked.

        :param schemas: the schemas to be evaluated
        :param pending: an optional pending schema

        :return: an array containing the values of the schemas (according to this strategy)
        """
        pass

    def update(self, schemas: Sequence[Schema], pending: Optional[Schema] = None) -> None:
        """ Updates any internal state maintained by the evaluation strategy, which may influence evaluation.

        :param schemas: the schemas to be evaluated
        :param pending: an optional pending schema

        :return: None
        """
        pass


class NoOpEvaluationStrategy(EvaluationStrategy):
    """ An EvaluationStrategy that always returns zero value. Intended to function as a NO-OP. """

    def __eq__(self, other) -> bool:
        if isinstance(other, NoOpEvaluationStrategy):
            return True
        return False if other is None else NotImplemented

    def values(self, schemas: Sequence[Schema], pending: Optional[Schema] = None) -> np.ndarray:
        return np.zeros_like(schemas)


class TotalPrimitiveValueEvaluationStrategy(EvaluationStrategy):
    """ An EvaluationStrategy based solely on the total primitive value over schemas' items. """

    def __init__(self) -> None:
        self.calc_schema_values = np.vectorize(self._calc_schema_value, otypes=[float])

    def __eq__(self, other) -> bool:
        if isinstance(other, TotalPrimitiveValueEvaluationStrategy):
            return True
        return False if other is None else NotImplemented

    def _calc_schema_value(self, schema: Schema) -> float:
        return sum(item.primitive_value for item in schema.result.items if schema.result)

    def values(self, schemas: Sequence[Schema], pending: Optional[Schema] = None) -> np.ndarray:
        return (
            np.array([])
            if schemas is None or len(schemas) == 0
            else self.calc_schema_values(schemas)
        )


class TotalDelegatedValueEvaluationStrategy(EvaluationStrategy):
    """ An EvaluationStrategy based solely on the total delegated value over schemas' items. """

    def __init__(self) -> None:
        self.calc_schema_values = np.vectorize(self._calc_schema_value, otypes=[float])

    def __eq__(self, other) -> bool:
        if isinstance(other, TotalDelegatedValueEvaluationStrategy):
            return True
        return False if other is None else NotImplemented

    def _calc_schema_value(self, schema: Schema) -> float:
        return sum(item.delegated_value for item in schema.result.items if schema.result)

    def values(self, schemas: Sequence[Schema], pending: Optional[Schema] = None) -> np.ndarray:
        return (
            np.array([])
            if schemas is None or len(schemas) == 0
            else self.calc_schema_values(schemas)
        )


class MaxDelegatedValueEvaluationStrategy(EvaluationStrategy):
    """ An EvaluationStrategy based solely on the maximum delegated value over schemas' items. """

    def __init__(self) -> None:
        self.calc_schema_values = np.vectorize(self._calc_schema_value, otypes=[float])

    def __eq__(self, other) -> bool:
        if isinstance(other, MaxDelegatedValueEvaluationStrategy):
            return True
        return False if other is None else NotImplemented

    def _calc_schema_value(self, schema: Schema) -> float:
        return max([item.delegated_value for item in schema.result.items if schema.result], default=0.0)

    def values(self, schemas: Sequence[Schema], pending: Optional[Schema] = None) -> np.ndarray:
        return (
            np.array([])
            if schemas is None or len(schemas) == 0
            else self.calc_schema_values(schemas)
        )


class InstrumentalValueEvaluationStrategy(EvaluationStrategy):
    """ An EvaluationStrategy based solely on the instrumental values of schemas.

    Drescher on instrumental value:

    "When the schema mechanism activates a schema as a link in some chain to a positively valued state, then
     that schema's result (or rather, the part of it that includes the next link's context) is said to have
     instrumental value.

     Instrumental value, unlike primitive (and delegated) value, is transient rather than persistent. As
     the state of the world changes, a given state may lie along a chain from the current state to a goal at one
     moment but not the next." (See Drescher 1991, p. 63)

    """

    def __eq__(self, other) -> bool:
        if isinstance(other, InstrumentalValueEvaluationStrategy):
            return True
        return False if other is None else NotImplemented

    def values(self, schemas: Sequence[Schema], pending: Optional[Schema] = None) -> np.ndarray:
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
    """ An EvaluationStrategy that initially returns higher values for component schemas of the currently selected
    pending (composite action) schema. However, this value advantage quickly vanishes and becomes aversion if the
    pending schema's goal state fails to obtain after repeated, consecutive selection.
    """

    def __init__(self,
                 max_value: float = 1.0,
                 decay_strategy: Optional[DecayStrategy] = None
                 ) -> None:
        self.max_value = max_value
        self.decay_strategy = decay_strategy

        self._active_pending: Optional[Schema] = None
        self._n: int = 0

    def __repr__(self):
        attr_values = {
            'max_value': self.max_value,
            'decay_strategy': self.decay_strategy
        }
        return repr_str(obj=self, attr_values=attr_values)

    def __eq__(self, other) -> bool:
        if isinstance(other, PendingFocusEvaluationStrategy):
            return all(
                (
                    # generator expression for conditions to allow lazy evaluation
                    condition for condition in
                    [
                        self.max_value == other.max_value,
                        self.decay_strategy == other.decay_strategy,
                    ]
                )
            )
        return False if other is None else NotImplemented

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

    def values(self, schemas: Sequence[Schema], pending: Optional[Schema] = None) -> np.ndarray:
        if schemas is None or len(schemas) == 0:
            return np.array([])

        # zero values (i.e., no pending focus) if pending schema is not set
        if not pending:
            return np.zeros(len(schemas), dtype=np.float64)

        values = self.max_value * np.ones_like(schemas, dtype=np.float64)
        step_size = self._n if pending == self._active_pending else 0

        # decay focus value by steps equal to the number of repeated calls for this pending schema
        if self.decay_strategy:
            values = self._decay_strategy.decay(values=values, step_size=step_size)

        for i, schema in enumerate(schemas):
            if schema not in pending.action.controller.components:
                values[i] = 0.0

        return values

    def update(self, schemas: Sequence[Schema], pending: Optional[Schema] = None) -> None:
        if not pending:
            self._active_pending = None

        # new active pending
        elif pending != self._active_pending:
            self._active_pending = pending

            # values would have been called once already with this pending, so next call should be with n == 1
            self._n = 1

        # previously set active pending occurred again
        else:
            self._n += 1


class ReliabilityEvaluationStrategy(EvaluationStrategy):
    """ An EvaluationStrategy that penalizes unreliable schemas.

    The values returned will be in the range [0.0, -MAX_PENALTY].

    * Schemas with reliabilities of 1.0 will receive a value of 0.0.
    * Schemas with reliabilities of 0.0 will receive a value of -MAX_PENALTY.
    * Schemas with reliabilities between 0.0 and 1.0 will have negative values that become more negative as their
    reliability approaches 0.0.
    """

    def __init__(self, max_penalty: float = 1.0, severity: float = 2.0, threshold: float = 0.0) -> None:
        self.max_penalty = max_penalty
        self.severity = severity
        self.threshold = threshold

    def __repr__(self):
        attr_values = {
            'max_penalty': self.max_penalty,
            'severity': self.severity,
            'threshold': self.threshold,
        }
        return repr_str(obj=self, attr_values=attr_values)

    def __eq__(self, other) -> bool:
        if isinstance(other, ReliabilityEvaluationStrategy):
            return all(
                (
                    # generator expression for conditions to allow lazy evaluation
                    condition for condition in
                    [
                        self.max_penalty == other.max_penalty,
                        self.severity == other.severity,
                        self.threshold == other.threshold
                    ]
                )
            )
        return False if other is None else NotImplemented

    @property
    def max_penalty(self) -> float:
        return self._max_penalty

    @max_penalty.setter
    def max_penalty(self, value: float) -> None:
        if not (value > 0.0):
            raise ValueError('Max penalty must be > 0.0')
        self._max_penalty = value

    @property
    def severity(self) -> float:
        return self._severity

    @severity.setter
    def severity(self, value: float) -> None:
        if not (value > 0.0):
            raise ValueError('Severity must be > 0.0')
        self._severity = value

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        if not (0.0 <= value < 1.0):
            raise ValueError('Threshold must be be >= 0.0 and < 1.0')
        self._threshold = value

    def values(self, schemas: Sequence[Schema], pending: Optional[Schema] = None) -> np.ndarray:
        if schemas is None or len(schemas) == 0:
            return np.array([])

        # nans treated as 0.0 reliability
        reliabilities = np.array([0.0 if s.reliability is np.nan else s.reliability for s in schemas])

        over_threshold = reliabilities > self.threshold

        # values <= threshold are set to max_penalty; otherwise, apply penalty function
        result = np.full_like(reliabilities, -self.max_penalty)

        if any(over_threshold):
            result[over_threshold] = self.max_penalty * (np.power(reliabilities[over_threshold], self.severity) - 1.0)

        return result


class EpsilonRandomEvaluationStrategy(EvaluationStrategy):
    """ An EvaluationStrategy based on epsilon-random selection of schemas.

    The values returned by this strategy will either be 0.0 or max_value.
    * At most one schema will be given the value of max_value per invocation. All other values will be 0.0.

    An epsilon parameter is used to determine the probability of a random selection.
    * An epsilon of 1.0 indicates that values will always be randomly selected (i.e., one schema will always be given
    the value of max_value per invocation.)
    * An epsilon of 0.0 indicates that values will never be randomly selected (i.e., schemas will always be given
    zero values, resulting in this strategy functioning as a no-op for schema selection purposes).

    An optional decay strategy can be provided to this evaluation strategy to reduce the probability of random
    selections over time.

    See Sutton and Barto, 2018 for more details on epsilon-greedy selection.

    """

    def __init__(self,
                 epsilon: float = 0.99,
                 epsilon_min: float = 0.0,
                 decay_strategy: Optional[DecayStrategy] = None,
                 max_value=1.0
                 ) -> None:
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.decay_strategy = decay_strategy
        self.max_value = max_value

        if self.epsilon_min > self.epsilon:
            raise ValueError('Epsilon min must be less than or equal to epsilon')

    def __repr__(self):
        attr_values = {
            'epsilon': self.epsilon,
            'epsilon_min': self.epsilon_min,
            'decay_strategy': self.decay_strategy,
            'max_value': self.max_value,
        }
        return repr_str(obj=self, attr_values=attr_values)

    def __eq__(self, other) -> bool:
        if isinstance(other, EpsilonRandomEvaluationStrategy):
            return all(
                (
                    # generator expression for conditions to allow lazy evaluation
                    condition for condition in
                    [
                        self.epsilon == other.epsilon,
                        self.epsilon_min == other.epsilon_min,
                        self.decay_strategy == other.decay_strategy,
                        self.max_value == other.max_value,
                    ]
                )
            )
        return False if other is None else NotImplemented

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

    def values(self, schemas: Sequence[Schema], pending: Optional[Schema] = None) -> np.ndarray:
        if not schemas:
            return np.array([])

        values = np.zeros_like(schemas)

        # determine if taking exploratory or goal-directed action
        is_exploratory = rng().uniform(0.0, 1.0) < self.epsilon

        # non-exploratory evaluation - all returned values are zeros
        if not is_exploratory:
            return values

        # randomly select winning schema, setting its value to max value
        if pending:
            pending_components = pending.action.controller.components
            pending_indexes = [schemas.index(schema) for schema in set(schemas).intersection(pending_components)]

            # if pending, limit choice to pending schema's components
            values[rng().choice(pending_indexes)] = self.max_value
        else:
            values[rng().choice(len(schemas))] = self.max_value

        return values

    def update(self, schemas: Sequence[Schema], pending: Optional[Schema] = None) -> None:
        if self.decay_strategy:
            decayed_epsilon = self.decay_strategy.decay(np.array([self.epsilon]))
            self.epsilon = np.maximum(decayed_epsilon, self.epsilon_min)[0]


class HabituationEvaluationStrategy(EvaluationStrategy):
    """ An EvaluationStrategy that modulates the selection value of schemas based on the recency and frequency of their
    action's selection.

     Internally, it uses the global action trace that is accessible by the get/set_action_trace functions in the
     core package.

     The value of schemas with actions that have been chosen many times recently and/or frequently is decreased. This
     implements a form of "habituation" (see Drescher, 1991, Section 3.4.2). The intent of this strategy is to boost
     the value of actions that have been under-represented in recent selections.

     See also:

        (1) "a schema that has recently been activated many times becomes partly suppressed (habituation), preventing
             a small number of schemas from persistently dominating the mechanism's activity."
             (See Drescher, 1991, p. 66.)

        (2) "schemas with underrepresented actions receive enhanced exploration value." (See Drescher, 1991, p. 67.)

    """

    def __init__(self, scaling_strategy: ScalingStrategy) -> None:
        self.scaling_strategy: ScalingStrategy = scaling_strategy

    def __repr__(self):
        attr_values = {
            'scaling_strategy': self.scaling_strategy,
        }
        return repr_str(obj=self, attr_values=attr_values)

    def __eq__(self, other) -> bool:
        if isinstance(other, HabituationEvaluationStrategy):
            return all(
                (
                    # generator expression for conditions to allow lazy evaluation
                    condition for condition in
                    [
                        self.scaling_strategy == other.scaling_strategy,
                    ]
                )
            )
        return False if other is None else NotImplemented

    def values(self, schemas: Sequence[Schema], pending: Optional[Schema] = None) -> np.ndarray:
        if not schemas:
            return np.array([])

        actions = [s.action for s in schemas]
        trace = get_action_trace()
        trace_values = trace.values[trace.indexes(actions)]
        median_trace_value = np.median(trace_values)

        values = self.scaling_strategy.scale(trace_values - median_trace_value)
        return values


class CompositeEvaluationStrategy(EvaluationStrategy):
    """ An implementation of the EvaluationStrategy protocol based on a weighted-sum over a collection of evaluation
    strategies. """

    def __init__(self,
                 strategies: Sequence[EvaluationStrategy],
                 weights: Sequence[float] = None,
                 post_process: Sequence[EvaluationPostProcess] = None,
                 strategy_alias: str = None
                 ) -> None:
        self.strategies: list[EvaluationStrategy] = list(strategies)
        self.weights: Sequence[float] = (
            np.array(weights)
            if weights is not None
            else equal_weights(len(self.strategies))
        )
        self.post_process: Sequence[EvaluationPostProcess] = (
            post_process
            if post_process
            else []
        )
        self.strategy_alias = strategy_alias or ''

    def __repr__(self):
        attr_values = {
            'strategies': self.strategies,
            'weights': self.weights,
            'post_process': self.post_process,
            'strategy_alias': self.strategy_alias,
        }
        return repr_str(obj=self, attr_values=attr_values)

    def __eq__(self, other) -> bool:
        if isinstance(other, CompositeEvaluationStrategy):
            return all(
                (
                    # generator expression for conditions to allow lazy evaluation
                    condition for condition in
                    [
                        self.strategies == other.strategies,
                        np.array_equal(self.weights, other.weights),
                        self.post_process == other.post_process,
                    ]
                )
            )
        return False if other is None else NotImplemented

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

    def __call__(self,
                 schemas: Sequence[Schema],
                 pending: Optional[Schema] = None,
                 post_process: Sequence[EvaluationPostProcess] = None,
                 ) -> np.ndarray:
        """ Returns the schemas' values as the weighted sum over each strategy's values and updates each strategy.

        Note: by default, the instance's post-process callables are used; however, they can be overridden by sending
        an alternate set of post-process callables as an argument to the method.

        :param schemas: the schemas to be evaluated
        :param pending: an optional pending schema
        :param post_process: an optional collection of callables that will be invoked after schema value determination

        :return: an array containing the values of the schemas (according to this strategy)
        """
        return super().__call__(
            schemas=schemas,
            pending=pending,
            post_process=post_process or self.post_process,
        )

    def values(self, schemas: Sequence[Schema], pending: Optional[Schema] = None) -> np.ndarray:
        values_list = [
            strategy.values(
                schemas=schemas,
                pending=pending,
            )
            for strategy in self.strategies
        ]

        return sum(weight * values for weight, values in zip(self.weights, values_list))

    def update(self, schemas: Sequence[Schema], pending: Optional[Schema] = None) -> None:
        for strategy in self.strategies:
            strategy.update(
                schemas=schemas,
                pending=pending,
            )


class DefaultExploratoryEvaluationStrategy(CompositeEvaluationStrategy):
    """ A default exploratory evaluation strategy that includes randomized exploration and action-balancing strategies.

    “The exploration criterion boosts the importance of a schema to promote its activation for the sake of what might
    be learned by that activation” (Drescher, 1991, p. 60)

    """

    def __init__(self,
                 epsilon: float = 0.99,
                 epsilon_min: float = 0.2,
                 epsilon_decay_rate: float = 0.9999,
                 post_process: Sequence[EvaluationPostProcess] = None,
                 ) -> None:
        # TODO: add an evaluation strategy for the following
        # "a component of exploration value promotes underrepresented levels of actions, where a structure's level is
        # defined as follows: primitive items and actions are of level zero; any structure defined in terms of other
        # structures is of one greater level than the maximum of those structures' levels." (See Drescher, 1991, p. 67)

        self._habituation_value_strategy = HabituationEvaluationStrategy(
            scaling_strategy=SigmoidScalingStrategy()
        )

        self._epsilon_decay_strategy = GeometricDecayStrategy(rate=epsilon_decay_rate)
        self._epsilon_random_value_strategy = EpsilonRandomEvaluationStrategy(
            epsilon=epsilon,
            epsilon_min=epsilon_min,
            decay_strategy=self._epsilon_decay_strategy
        )

        super().__init__(
            strategies=[
                self._habituation_value_strategy,
                self._epsilon_random_value_strategy
            ],
            post_process=post_process,
            strategy_alias='default-exploratory-strategy'
        )

    def __eq__(self, other) -> bool:
        if isinstance(other, DefaultExploratoryEvaluationStrategy):
            return all(
                (
                    # generator expression for conditions to allow lazy evaluation
                    condition for condition in
                    [
                        self.epsilon == other.epsilon,
                        self.epsilon_min == other.epsilon_min,
                        self.epsilon_decay_rate == other.epsilon_decay_rate,
                        self.post_process == other.post_process,
                    ]
                )
            )
        return False if other is None else NotImplemented

    @property
    def epsilon(self) -> float:
        return self._epsilon_random_value_strategy.epsilon

    @epsilon.setter
    def epsilon(self, value: float) -> None:
        self._epsilon_random_value_strategy.epsilon = value

    @property
    def epsilon_min(self) -> float:
        return self._epsilon_random_value_strategy.epsilon_min

    @epsilon_min.setter
    def epsilon_min(self, value: float) -> None:
        self._epsilon_random_value_strategy.epsilon_min = value

    @property
    def epsilon_decay_rate(self) -> float:
        return self._epsilon_decay_strategy.rate

    @epsilon_decay_rate.setter
    def epsilon_decay_rate(self, value: float) -> None:
        self._epsilon_decay_strategy.rate = value


class DefaultGoalPursuitEvaluationStrategy(CompositeEvaluationStrategy):
    def __init__(self,
                 reliability_max_penalty: float = 1.0,
                 reliability_threshold: float = 0.0,
                 pending_focus_max_value: float = 1.0,
                 pending_focus_decay_rate: float = 0.5,
                 weights: Sequence[float] = None,
                 post_process: Sequence[EvaluationPostProcess] = None,
                 ) -> None:
        # basic item value functions
        self._primitive_value_strategy = TotalPrimitiveValueEvaluationStrategy()
        self._delegated_value_strategy = TotalDelegatedValueEvaluationStrategy()

        # penalty to unreliable schemas
        self._reliability_value_strategy = ReliabilityEvaluationStrategy(
            max_penalty=reliability_max_penalty,
            threshold=reliability_threshold,
        )

        # composite-action-specific value functions
        self._instrumental_value_strategy = InstrumentalValueEvaluationStrategy()

        self._pending_focus_decay_strategy = GeometricDecayStrategy(rate=pending_focus_decay_rate)
        self._pending_focus_value_strategy = PendingFocusEvaluationStrategy(
            max_value=pending_focus_max_value,
            decay_strategy=self._pending_focus_decay_strategy
        )

        super().__init__(
            strategies=[
                self._delegated_value_strategy,
                self._instrumental_value_strategy,
                self._pending_focus_value_strategy,
                self._primitive_value_strategy,
                self._reliability_value_strategy,
            ],
            weights=weights,
            post_process=post_process,
            strategy_alias='default-goal-pursuit-strategy',
        )

    def __eq__(self, other) -> bool:
        if isinstance(other, DefaultGoalPursuitEvaluationStrategy):
            return all(
                (
                    # generator expression for conditions to allow lazy evaluation
                    condition for condition in
                    [
                        self.reliability_max_penalty == other.reliability_max_penalty,
                        self.pending_focus_max_value == other.pending_focus_max_value,
                        self.pending_focus_decay_rate == other.pending_focus_decay_rate,
                        self.post_process == other.post_process,
                    ]
                )
            )
        return False if other is None else NotImplemented

    @property
    def reliability_max_penalty(self) -> float:
        return self._reliability_value_strategy.max_penalty

    @reliability_max_penalty.setter
    def reliability_max_penalty(self, value: float) -> None:
        self._reliability_value_strategy.max_penalty = value

    @property
    def reliability_threshold(self) -> float:
        return self._reliability_value_strategy.threshold

    @reliability_threshold.setter
    def reliability_threshold(self, value: float) -> None:
        self._reliability_value_strategy.threshold = value

    @property
    def pending_focus_max_value(self) -> float:
        return self._pending_focus_value_strategy.max_value

    @pending_focus_max_value.setter
    def pending_focus_max_value(self, value: float) -> None:
        self._pending_focus_value_strategy.max_value = value

    @property
    def pending_focus_decay_rate(self) -> float:
        return self._pending_focus_decay_strategy.rate

    @pending_focus_decay_rate.setter
    def pending_focus_decay_rate(self, value: float) -> None:
        self._pending_focus_decay_strategy.rate = value


class DefaultEvaluationStrategy(CompositeEvaluationStrategy):
    def __init__(self,
                 goal_pursuit_strategy: EvaluationStrategy = None,
                 exploratory_strategy: EvaluationStrategy = None,
                 weights: Sequence[float] = None,
                 weight_update_strategy: WeightUpdateStrategy = None,
                 post_process: Sequence[EvaluationPostProcess] = None
                 ) -> None:
        self.weight_update_strategy = weight_update_strategy or CyclicWeightUpdateStrategy()

        # temporary objects. these are later accessed via the CompositeEvaluationStrategy
        goal_pursuit_strategy = goal_pursuit_strategy or DefaultGoalPursuitEvaluationStrategy()
        exploratory_strategy = exploratory_strategy or DefaultExploratoryEvaluationStrategy()
        weights = equal_weights(2) if weights is None else weights

        super().__init__(
            strategies=[goal_pursuit_strategy, exploratory_strategy],
            weights=weights,
            post_process=post_process,
            strategy_alias='default_evaluation_strategy'
        )

    def __eq__(self, other) -> bool:
        if isinstance(other, DefaultEvaluationStrategy):
            return all(
                (
                    # generator expression for conditions to allow lazy evaluation
                    condition for condition in
                    [
                        super().__eq__(other),
                        self.weight_update_strategy == other.weight_update_strategy,
                    ]
                )
            )
        return False if other is None else NotImplemented

    @property
    def goal_pursuit_strategy(self) -> EvaluationStrategy:
        return self.strategies[0]

    @goal_pursuit_strategy.setter
    def goal_pursuit_strategy(self, value: EvaluationStrategy) -> None:
        self.strategies[0] = value

    @property
    def exploratory_strategy(self) -> EvaluationStrategy:
        return self.strategies[1]

    @exploratory_strategy.setter
    def exploratory_strategy(self, value: EvaluationStrategy) -> None:
        self.strategies[1] = value

    def update(self, schemas: Sequence[Schema], pending: Optional[Schema] = None) -> None:
        # update weights used for next evaluation
        self.weights = self.weight_update_strategy.update(self.weights)

        # update any internal state maintained by strategies
        super().update(schemas=schemas, pending=pending)

    def configure_as_greedy(self) -> None:
        self.weights = [1.0, 0.0]
        self.weight_update_strategy = NoOpWeightUpdateStrategy()

        self.goal_pursuit_strategy.reliability_max_penalty = 1.5


class DefaultGreedyEvaluationStrategy(DefaultGoalPursuitEvaluationStrategy):
    def __init__(self,
                 reliability_threshold: float = 0.7,
                 reliability_max_penalty: float = 1.0,
                 weights: Sequence[float] = None,
                 post_process: Sequence[EvaluationPostProcess] = None,
                 ) -> None:
        default_weights = [
            0.35,  # delegated value
            0.2,  # instrumental value
            0.1,  # pending focus
            0.1,  # primitive value
            0.25,  # reliability
        ]

        super().__init__(
            reliability_max_penalty=reliability_max_penalty,
            reliability_threshold=reliability_threshold,
            weights=default_weights if weights is None else weights,
            post_process=post_process
        )


####################
# helper functions #
####################
def display_values(schemas: Optional[Sequence[Schema]],
                   pending: Optional[Schema],
                   values: Optional[np.ndarray]) -> np.ndarray:
    for schema, value in zip(schemas, values):
        logger.debug(f'{schema} [{value}]')

    return values


def display_minmax(schemas: Optional[Sequence[Schema]],
                   pending: Optional[Schema],
                   values: Optional[np.ndarray]) -> np.ndarray:
    if len(values) == 0:
        return values

    max_value = np.max(values)
    max_value_schema = schemas[np.flatnonzero(values == max_value)[0]]
    min_value = np.min(values)
    min_value_schema = schemas[np.flatnonzero(values == min_value)[0]]

    logger.debug(f'Schema with Maximum Value: {max_value_schema} [{max_value}]')
    logger.debug(f'Schema with Minimum Value: {min_value_schema} [{min_value}]')

    return values
