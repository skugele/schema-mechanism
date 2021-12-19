from typing import Any, Callable, Iterable, Collection, Optional

import numpy as np

from schema_mechanism.util import cosine_sims, get_unique_id


class State:
    def __init__(self,
                 discrete_values: Collection[str] = None,
                 continuous_values: Collection[np.ndarray] = None):
        self._continuous_values = continuous_values
        self._discrete_values = discrete_values

    @property
    def discrete_values(self) -> Collection[str]:
        return self._discrete_values

    @property
    def continuous_values(self) -> Collection[np.ndarray]:
        return self._continuous_values


class Item:
    def __init__(self, value: Any, negated: bool = False):
        self._value = value
        self._negated = negated

    @property
    def value(self) -> Any:
        return self._value

    @property
    def negated(self) -> bool:
        return self._negated

    def is_on(self, state: State, *args, **kwargs) -> bool:
        return NotImplemented

    def is_off(self, state: State, *args, **kwargs) -> bool:
        return not self.is_on(state, *args, **kwargs)

    def is_satisfied(self, state: State, *args, **kwargs) -> bool:
        if self._negated:
            return self.is_off(state, *args, **kwargs)
        else:
            return self.is_on(state, *args, **kwargs)


class DiscreteItem(Item):
    """ A state element that can be thought as a proposition/feature. """

    def __init__(self, value: str, negated: bool = False):
        super().__init__(value, negated)

    def is_on(self, state: State, *args, **kwargs) -> bool:
        if state is None or state.discrete_values is None:
            return False

        return self.value in state.discrete_values


class ContinuousItem(Item):
    """ A state element that can be viewed as continuously comparable content. """

    DEFAULT_PRECISION = 2  # 2 decimal places of precision
    DEFAULT_ACTIVATION_THRESHOLD = 0.99
    DEFAULT_SIMILARITY_MEASURE = cosine_sims

    def __init__(self,
                 value: np.ndarray,
                 negated: bool = False,
                 similarity_measure: Callable[[np.ndarray, Iterable[np.ndarray]], np.ndarray] = cosine_sims):
        super().__init__(value, negated)

        self.similarity_measure = similarity_measure

    def is_on(self, state: State, *args, **kwargs) -> bool:
        if state is None or state.continuous_values is None:
            return False

        threshold = kwargs['threshold'] if 'threshold' in kwargs else ContinuousItem.DEFAULT_ACTIVATION_THRESHOLD
        precision = kwargs['precision'] if 'precision' in kwargs else ContinuousItem.DEFAULT_PRECISION
        similarity_measure = (
            kwargs['similarity_measure']
            if 'similarity_measure' in kwargs
            else ContinuousItem.DEFAULT_SIMILARITY_MEASURE
        )

        similarities = similarity_measure(self.value, state.continuous_values).round(precision)
        return np.any(similarities >= threshold)

    def __eq__(self, other) -> bool:
        return np.array_equal(self.value, other)

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)


class Action:
    def __init__(self, id: Optional[str] = None):
        self._id = id or get_unique_id()

    @property
    def id(self) -> str:
        return self._id


class StateAssertion:

    def __init__(self, items: Optional[Collection[Item]] = None):
        self._items = items or frozenset()

    @property
    def items(self) -> Collection[Item]:
        return self._items

    def is_satisfied(self, state: State, *args, **kwargs) -> bool:
        """ Satisfied when all non-negated items are On, and all negated items are Off.

        :param state: the agent's current state
        :param args: optional positional arguments
        :param kwargs: optional keyword arguments
        :return: True if the context is satisfied given the current state; False otherwise.
        """
        return all(map(lambda i: i.is_satisfied(state, *args, **kwargs), self._items))

    def __len__(self):
        return len(self._items)


class Context(StateAssertion):
    pass


class Result(StateAssertion):
    pass


class Schema:
    """
    a three-component data structure used to express a prediction about the environmental state that
    will result from taking a particular action when in a given environmental state (i.e., context).

    Note: A schema is not a rule that says to take a particular action when its context is satisfied;
    the schema just says what might happen if that action were taken.
    """
    INITIAL_RELIABILITY = 0.0

    def __init__(self, action: Action, context: Optional[Context] = None, result: Optional[Result] = None):
        self._context = context
        self._action = action
        self._result = result

        # A unique identifier for this schema
        self._id = get_unique_id()

        # A schema's reliability is the likelihood with which the schema succeeds (i.e., its
        # result obtains) when activated
        self.reliability = Schema.INITIAL_RELIABILITY

        if self.action is None:
            raise ValueError('Action cannot be None')

    @property
    def context(self) -> Context:
        return self._context

    @property
    def action(self) -> Action:
        return self._action

    @property
    def result(self) -> Result:
        return self._result

    @property
    def id(self) -> str:
        return self._id

    # TODO: Should this be in action selection?
    def is_applicable(self) -> bool:
        """ “A schema is said to be applicable when its context is satisfied and no
             known overriding conditions obtain.” (Drescher, 1991, p.53)
        :return:
        """
        pass
