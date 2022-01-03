from __future__ import annotations

from typing import Any
from typing import Collection
from typing import Optional
from typing import Tuple

import numpy as np

from schema_mechanism.util import cosine_sims
from schema_mechanism.util import get_unique_id


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


# FIXME: Should be as immutable as possible
# TODO: How do we add modality to this? If we do, we need to add modality to the State as well.
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


# FIXME: Should be as immutable as possible
# TODO: This could be backed by an Object Pool to minimize the number of distinct objects.
# TODO: See Flyweight Pattern.
class DiscreteItem(Item):
    """ A state element that can be thought as a proposition/feature. """

    def __init__(self, value: str, negated: bool = False):
        super().__init__(value, negated)

    def is_on(self, state: State, *args, **kwargs) -> bool:
        if state is None or state.discrete_values is None:
            return False

        return self.value in state.discrete_values

    def __eq__(self, other) -> bool:
        if isinstance(other, DiscreteItem):
            return self.value == other.value
        return NotImplemented

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)


# FIXME: Should be as immutable as possible
class ContinuousItem(Item):
    """ A state element that can be viewed as continuously comparable content. """

    DEFAULT_PRECISION = 2  # 2 decimal places of precision
    DEFAULT_ACTIVATION_THRESHOLD = 0.99
    DEFAULT_SIMILARITY_MEASURE = cosine_sims

    def __init__(self, value: np.ndarray, negated: bool = False):
        super().__init__(value, negated)

        # prevent modification of array values
        self.value.setflags(write=False)

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
        return np.array_equal(self.value, other.value)

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)


# FIXME: Should be as immutable as possible
class Action:
    def __init__(self, uid: Optional[str] = None):
        self._uid = uid or get_unique_id()

    @property
    def uid(self) -> str:
        return self._uid

    def __eq__(self, other):
        if isinstance(other, Action):
            return self._uid == other._uid
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self._uid)


# FIXME: Should be as immutable as possible. Maybe we can use Tuples instead of Collections?
class StateAssertion:

    def __init__(self, items: Optional[Tuple[Item, ...]] = None):
        self._items = items or tuple()

    @property
    def items(self) -> Tuple[Item, ...]:
        return self._items

    def is_satisfied(self, state: State, *args, **kwargs) -> bool:
        """ Satisfied when all non-negated items are On, and all negated items are Off.

        :param state: the agent's current state
        :param args: optional positional arguments
        :param kwargs: optional keyword arguments
        :return: True if the context is satisfied given the current state; False otherwise.
        """
        return all(map(lambda i: i.is_satisfied(state, *args, **kwargs), self._items))

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, item: Item) -> bool:
        return item in self._items

    def replicate_with(self, item: Item) -> StateAssertion:
        if self.__contains__(item):
            raise ValueError('Item already exists in StateAssertion')

        return StateAssertion(items=(*self._items, item))


# TODO: This class can probably be removed. Existing Contexts can be changed to StateAssertions.
class Context(StateAssertion):
    pass


# TODO: This class can probably be removed. Existing Results can be changed to StateAssertions.
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

        self._overriding_conditions = None

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

    @property
    def overriding_conditions(self) -> StateAssertion:
        return self._overriding_conditions

    @overriding_conditions.setter
    def overriding_conditions(self, overriding_conditions: StateAssertion):
        self._overriding_conditions = overriding_conditions

    def is_applicable(self, state: State, *args, **kwargs) -> bool:
        """ A schema is applicable when its context is satisfied and there are no active overriding conditions.

            “A schema is said to be applicable when its context is satisfied and no
                 known overriding conditions obtain.” (Drescher, 1991, p.53)

        :param state: the agent's current state
        :param args: optional positional arguments
        :param kwargs: optional keyword arguments
        :return: True if the schema is applicable in this state; False, otherwise.
        """
        overridden = False
        if self.overriding_conditions is not None:
            overridden = self.overriding_conditions.is_satisfied(state, *args, **kwargs)
        return (not overridden) and self.context.is_satisfied(state, *args, **kwargs)

    def create_spin_off(self, mode: str, item: Item) -> Schema:
        """ Creates a context or result spin-off schema that includes the supplied item in its context or result.

        :param mode: "result" (see Drescher, 1991, p. 71) or "context" (see Drescher, 1991, p. 73)
        :param item: a relevant item to add to the context or result of a spin-off schema
        :return: a spin-off schema based on this one
        """
        if "context" == mode:
            new_context = (
                Context(items=(item,))
                if self.context is None
                else self.context.replicate_with(item)
            )
            return Schema(action=self.action,
                          context=new_context,
                          result=self.result)

        elif "result" == mode:
            new_result = (
                Result(items=(item,))
                if self.result is None
                else self.result.replicate_with(item)
            )
            return Schema(action=self.action,
                          context=self.context,
                          result=new_result)

        else:
            raise ValueError(f'Unknown spin-off mode: {mode}')
