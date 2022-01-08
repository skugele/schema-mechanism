from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from collections import defaultdict
from functools import cache
from functools import lru_cache
from typing import Any
from typing import Collection
from typing import Dict
from typing import Hashable
from typing import Optional
from typing import Tuple
from typing import Type

import numpy as np

from schema_mechanism.util import Singleton
from schema_mechanism.util import get_unique_id

# Type Aliases
##############
StateElement = Hashable


# Classes
#########
class Item(ABC):
    def __init__(self, state_element: StateElement):
        self._state_element = state_element

    @property
    def state_element(self) -> StateElement:
        return self._state_element

    @abstractmethod
    def is_on(self, state: Collection[StateElement], *args, **kwargs) -> bool:
        return NotImplemented

    def is_off(self, state: Collection[StateElement], *args, **kwargs) -> bool:
        return not self.is_on(state, *args, **kwargs)

    @abstractmethod
    def __eq__(self, other: Any):
        pass

    def __ne__(self, other: Any):
        return not self.__eq__(other)

    # TODO: Need to be really careful with the default hash implementations which produce different values between
    # TODO: runs. This will kill and direct serialization/deserialization of data structures that rely on hashes.
    @abstractmethod
    def __hash__(self):
        pass


class ItemPool(metaclass=Singleton):
    _items: Dict[StateElement, Item] = dict()

    def __contains__(self, state_element: StateElement):
        return state_element in self._items

    def __len__(self) -> int:
        return len(self._items)

    def items(self) -> Collection[Item]:
        return self._items.values()

    def clear(self):
        self._items.clear()

        # clears the lru cache on get method
        self.get.cache_clear()

    @lru_cache
    def get(self, state_element: StateElement, item_type: Type[Item], **kwargs) -> Item:
        obj = self._items.get(state_element)
        if obj is None and not kwargs.get('read_only', False):
            self._items[state_element] = obj = item_type(state_element)
        return obj


class ReadOnlyItemPool(ItemPool):
    def __init__(self, pool: ItemPool):
        self._pool = pool

    def get(self, state_element: StateElement, item_type: Type[Item], **kwargs) -> Item:
        kwargs['read_only'] = True
        return self._pool.get(state_element, item_type, **kwargs)


class SymbolicItem(Item):
    """ A state element that can be thought as a proposition/feature. """

    def __init__(self, state_element: StateElement):
        super().__init__(state_element)

    @property
    def state_element(self) -> StateElement:
        return super().state_element

    def is_on(self, state: Collection[StateElement], *args, **kwargs) -> bool:
        return self.state_element in state

    def __eq__(self, other: Any) -> bool:
        return self.state_element == other.state_element if isinstance(other, SymbolicItem) else NotImplemented

    def __hash__(self) -> int:
        return hash(self.state_element)


class ItemAssertion:
    def __init__(self, item: Item, negated: bool = False) -> None:
        self._item = item
        self._negated = negated

    @property
    def item(self) -> Item:
        return self._item

    @property
    def negated(self) -> bool:
        return self._negated

    def is_satisfied(self, state: Collection[StateElement], *args, **kwargs) -> bool:
        if self._negated:
            return self._item.is_off(state, *args, **kwargs)
        else:
            return self._item.is_on(state, *args, **kwargs)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, ItemAssertion):
            return self._item == other._item and self._negated == other._negated
        return NotImplemented

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)


# TODO: Memory usage can be reduced by using complementary relationships
class ItemStatistics:
    def __init__(self):
        self._n = 0
        self._n_on = 0
        self._n_off = 0
        self._n_action = 0
        self._n_not_action = 0

        self._n_on_with_action = 0
        self._n_on_without_action = 0
        self._n_off_with_action = 0
        self._n_off_without_action = 0

    # TODO: An external function is needed that will compare the corresponding item's state element
    # TODO: to the state before and after an action is taken. If the item assertion is satisfied before
    # TODO: the action is executed then NO UPDATE.

    # "a trial for which the result was already satisfied before the action was taken does not count as a
    # positive-transition trial; and one for which the result was already unsatisfied does not count
    # as a negative-transition trial" (see Drescher, 1991, p. 72)
    def update(self, item_on: bool, action_taken: bool, count: int = 1) -> None:
        self._n += count

        if item_on and action_taken:
            self._n_on += count
            self._n_action += count
            self._n_on_with_action += count

        elif item_on and not action_taken:
            self._n_on += count
            self._n_not_action += count
            self._n_on_without_action += count

        elif not item_on and action_taken:
            self._n_off += count
            self._n_action += count
            self._n_off_with_action += count

        elif not item_on and not action_taken:
            self._n_off += count
            self._n_not_action += count
            self._n_off_without_action += count

    @property
    def positive_transition_corr(self) -> float:
        """ Returns the positive-transition correlation for this item.

            "The positive-transition correlation is the ratio of the probability of the slot's item turning On when
            the schema's action has just been taken to the probability of its turning On when the schema's action
            is not being taken." (see Drescher, 1991, p. 71)
        :return: the positive-transition correlation
        """
        try:
            # calculate conditional probabilities
            p_on_and_action = 0.0 if self._n_action == 0 else self._n_on_with_action / self._n_action
            p_on_without_action = 0.0 if self._n_not_action == 0 else self._n_on_without_action / self._n_not_action

            # calculate the ratio p(on AND action) : p(on AND NOT action)
            positive_trans_corr = p_on_and_action / (p_on_and_action + p_on_without_action)
            return positive_trans_corr
        except ZeroDivisionError:
            return np.NAN

    @property
    def negative_transition_corr(self) -> float:
        """ Returns the negative-transition correlation for this item.

            "The negative-transition correlation is the ratio of the probability of the slot's item turning Off when
            the schema's action has just been taken to the probability of its turning Off when the schema's action
            is not being taken." (see Drescher, 1991, p. 72)
        :return: the negative-transition correlation
        """
        try:
            p_off_and_action = 0.0 if self._n_action == 0 else self._n_off_with_action / self._n_action
            p_off_without_action = 0.0 if self._n_not_action == 0 else self._n_off_without_action / self._n_not_action

            # calculate the ratio p(off AND action) : p(off AND NOT action)
            negative_trans_corr = p_off_and_action / (p_off_and_action + p_off_without_action)
            return negative_trans_corr
        except ZeroDivisionError:
            return np.NAN

    @property
    def n(self) -> int:
        return self._n_on + self._n_off

    @property
    def n_on(self) -> int:
        return self._n_on

    @property
    def n_off(self) -> int:
        return self._n_off

    @property
    def n_action(self) -> int:
        return self._n_action

    @property
    def n_not_action(self) -> int:
        return self._n_not_action

    @property
    def n_on_with_action(self) -> int:
        return self._n_on_with_action

    @property
    def n_on_without_action(self) -> int:
        return self._n_on_without_action

    @property
    def n_off_with_action(self) -> int:
        return self._n_off_with_action

    @property
    def n_off_without_action(self) -> int:
        return self._n_off_without_action


# noinspection PyMissingConstructor
class FrozenItemStatisticsDecorator(ItemStatistics):
    def __init__(self, stats: ItemStatistics):
        self._stats = stats

    def update(self, item_on: bool, action_taken: bool, count: int = 1) -> None:
        raise NotImplementedError('Updates not supported on frozen object')
    
    def negative_transition_corr(self) -> float:
        return self._stats.negative_transition_corr
        
    def positive_transition_corr(self) -> float:
        return self._stats.positive_transition_corr
    
    @property
    def n(self) -> int:
        return self._stats.n

    @property
    def n_on(self) -> int:
        return self._stats.n_on

    @property
    def n_off(self) -> int:
        return self._stats.n_off

    @property
    def n_action(self) -> int:
        return self._stats.n_action

    @property
    def n_not_action(self) -> int:
        return self._stats.n_not_action

    @property
    def n_on_with_action(self) -> int:
        return self._stats.n_on_with_action

    @property
    def n_on_without_action(self) -> int:
        return self._stats.n_on_without_action

    @property
    def n_off_with_action(self) -> int:
        return self._stats.n_off_with_action

    @property
    def n_off_without_action(self) -> int:
        return self._stats.n_off_without_action


_NULL_STATS = FrozenItemStatisticsDecorator(ItemStatistics())


# noinspection PyMissingConstructor
class ItemStatisticsDecorator(Item):
    def __init__(self, item: Item) -> None:
        self._item = item
        self._stats = ItemStatistics()

    @property
    def item(self) -> Item:
        return self._item

    @property
    def stats(self) -> ItemStatistics:
        return self._stats

    @property
    def state_element(self) -> Any:
        return self._item.state_element

    def is_on(self, state: Collection[Any], *args, **kwargs) -> bool:
        return self._item.is_on(state, *args, **kwargs)

    def is_off(self, state: Collection[Any], *args, **kwargs) -> bool:
        return not self.is_on(state, *args, **kwargs)

    def __eq__(self, other: Item) -> bool:
        return self._item.__eq__(other)

    def __hash__(self) -> int:
        return hash(self._item)


class Action:
    def __init__(self, uid: Optional[str] = None):
        self._uid = uid or get_unique_id()

    @property
    def uid(self) -> str:
        return self._uid

    def __eq__(self, other) -> bool:
        if isinstance(other, Action):
            return self._uid == other._uid
        return False

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return hash(self._uid)


class StateAssertion:

    def __init__(self, item_asserts: Optional[Tuple[ItemAssertion, ...]] = None):
        self._item_asserts = item_asserts or tuple()

    @property
    def items(self) -> Tuple[ItemAssertion, ...]:
        return self._item_asserts

    def is_satisfied(self, state: Collection[StateElement], *args, **kwargs) -> bool:
        """ Satisfied when all non-negated items are On, and all negated items are Off.

        :param state: the agent's current state
        :param args: optional positional arguments
        :param kwargs: optional keyword arguments
        :return: True if the context is satisfied given the current state; False otherwise.
        """
        return all(map(lambda i: i.is_satisfied(state, *args, **kwargs), self._item_asserts))

    def __len__(self) -> int:
        return len(self._item_asserts)

    def __contains__(self, item_assert: ItemAssertion) -> bool:
        return item_assert in self._item_asserts

    def replicate_with(self, item_assert: ItemAssertion) -> StateAssertion:
        if self.__contains__(item_assert):
            raise ValueError('ItemAssertion already exists in StateAssertion')

        return StateAssertion(item_asserts=(*self._item_asserts, item_assert))


# TODO: This class can probably be removed. Existing Contexts can be changed to StateAssertions.
class Context(StateAssertion):
    pass


# TODO: This class can probably be removed. Existing Results can be changed to StateAssertions.
class Result(StateAssertion):
    pass


class ExtendedContext:
    def __init__(self, item_pool: ItemPool):
        self._item_pool = item_pool
        self._stats = defaultdict(lambda: _NULL_STATS)


class ExtendedResult:
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

    def is_applicable(self, state: Collection[Any], *args, **kwargs) -> bool:
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

    def create_spin_off(self, mode: str, item_assert: ItemAssertion) -> Schema:
        """ Creates a context or result spin-off schema that includes the supplied item in its context or result.

        :param mode: "result" (see Drescher, 1991, p. 71) or "context" (see Drescher, 1991, p. 73)
        :param item_assert: the item assertion to add to the context or result of a spin-off schema
        :return: a spin-off schema based on this one
        """
        if "context" == mode:
            new_context = (
                Context(item_asserts=(item_assert,))
                if self.context is None
                else self.context.replicate_with(item_assert)
            )
            return Schema(action=self.action,
                          context=new_context,
                          result=self.result)

        elif "result" == mode:
            new_result = (
                Result(item_asserts=(item_assert,))
                if self.result is None
                else self.result.replicate_with(item_assert)
            )
            return Schema(action=self.action,
                          context=self.context,
                          result=new_result)

        else:
            raise ValueError(f'Unknown spin-off mode: {mode}')
