from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from collections import defaultdict
from functools import lru_cache
from typing import Any
from typing import Collection
from typing import Dict
from typing import Hashable
from typing import Optional
from typing import Tuple
from typing import Type

import numpy as np

from schema_mechanism.util import Observable
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

    @property
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
    def __init__(self):
        self._pool = ItemPool()

    def get(self, state_element: StateElement, item_type: Type[Item], **kwargs) -> Item:
        kwargs['read_only'] = True
        return self._pool.get(state_element, item_type, **kwargs)

    def clear(self):
        raise NotImplementedError('ReadOnlyItemPool does not support clear operation.')


# TODO: What to do if the underlying item pool changes?
class ItemPoolStateView:
    def __init__(self, pool: ReadOnlyItemPool, state: Collection[StateElement]):
        # The state to which this view corresponds
        self._state = state
        self._on_items = set([item for item in pool.items if item.is_on(state)])

    @property
    def state(self):
        return self._state

    def is_on(self, item: Item) -> bool:
        return item in self._on_items


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


class SchemaStats:
    def __init__(self):
        self._n = 0  # total number of update events
        self._n_activated = 0
        self._n_succeeded = 0

        # TODO: Does reliability require a variable, or can its value be derived from other variables?
        # A schema's reliability is the likelihood with which the schema succeeds (i.e., its result obtains)
        # when activated
        self._reliability: float = 0.0

    def update(self, activated: bool = False, success: bool = False, count: int = 1):
        self._n += count

        if activated:
            self._n_activated += count

            if success:
                self._n_succeeded += count

    @property
    def n_activated(self) -> int:
        """ Returns the number of times this schema was activated (explicitly or implicitly).

            “To activate a schema is to initiate its action when the schema is applicable.” (Drescher, 1991, p.53)

            :return: the number of activations (explicit or implicit)
        """
        return self._n_activated

    @property
    def n_not_activated(self) -> int:
        return self._n - self._n_activated

    @property
    def n_succeeded(self) -> int:
        """ Returns the number of times this schema succeeded on activation.

            “An activated schema is said to succeed if its predicted results all in fact obtain, and to
            fail otherwise.” (Drescher, 1991, p.53)

        :return: the number of successes
        """
        return self._n_succeeded

    @property
    def n_failed(self) -> int:
        return self._n - self._n_succeeded

    @property
    def reliability(self) -> float:
        """ Returns the schema's reliability (i.e., its probability of success) when activated.

            "A schema's reliability is the probability with which the schema succeeds when activated."
            (Drescher, 1991, p. 54)

        :return: the schema's reliability
        """
        return self._reliability


class ECItemStats:
    """ Extended context item-level statistics

        "each extended-context slot records the ratio of the probability that the schema will succeed (i.e., that
        its result will obtain) if the schema is activated when the slot's item is On, to the probability
        of success if that item is Off when the schema is activated." (Drescher, 1991, p. 73)
    """

    def __init__(self, schema_stats: SchemaStats):
        self._schema_stats = schema_stats

        self._n_success_and_on = 0
        self._n_success_and_off = 0
        self._n_fail_and_on = 0
        self._n_fail_and_off = 0

    def update(self, item_on: bool, success: bool, count: int = 1) -> None:
        if item_on and success:
            self._n_success_and_on += count

        elif item_on and not success:
            self._n_fail_and_on += count

        elif not item_on and success:
            self._n_success_and_off += count

        elif not item_on and not success:
            self._n_fail_and_off += count

    @property
    def success_corr(self) -> float:
        """ Returns the ratio p(success | item on) : p(success | item off)

        :return: the ratio as a float, or numpy.NAN if division by zero
        """
        try:
            # calculate conditional probabilities
            p_success_and_on = (0.0
                                if self.n_on == 0
                                else self._n_success_and_on / self.n_on)
            p_success_and_off = (0.0
                                 if self.n_off == 0
                                 else self._n_success_and_off / self.n_off)

            p_success = p_success_and_on + p_success_and_off

            # calculate the ratio p(success | on) : p(success | off)
            positive_success_corr = p_success_and_on / p_success
            return positive_success_corr
        except ZeroDivisionError:
            return np.NAN

    @property
    def failure_corr(self) -> float:
        """ Returns the ratio p(failure | item on) : p(failure | item off)

        :return: the ratio as a float, or numpy.NAN if division by zero
        """
        try:
            # calculate conditional probabilities
            p_fail_and_on = (0.0
                             if self.n_on == 0
                             else self._n_fail_and_on / self.n_on)
            p_fail_and_off = (0.0
                              if self.n_off == 0
                              else self._n_fail_and_off / self.n_off)

            p_fail = p_fail_and_on + p_fail_and_off

            # calculate the ratio p(failure | on) : p(failure | off)
            positive_failure_corr = p_fail_and_on / p_fail
            return positive_failure_corr
        except ZeroDivisionError:
            return np.NAN

    @property
    def n_on(self) -> int:
        return self.n_success_and_on + self.n_fail_and_on

    @property
    def n_off(self) -> int:
        return self.n_success_and_off + self.n_fail_and_off

    @property
    def n_success_and_on(self) -> int:
        return self._n_success_and_on

    @property
    def n_success_and_off(self) -> int:
        return self._n_success_and_off

    @property
    def n_fail_and_on(self) -> int:
        return self._n_fail_and_on

    @property
    def n_fail_and_off(self) -> int:
        return self._n_fail_and_off


class ERItemStats:
    """ Extended result item-level statistics """

    def __init__(self, schema_stats: SchemaStats):
        self._schema_stats = schema_stats

        self._n_on_and_activated = 0
        self._n_on_and_not_activated = 0
        self._n_off_and_activated = 0
        self._n_off_and_not_activated = 0

    # TODO: An external function is needed that will compare the corresponding item's state element
    # TODO: to the state before and after an action is taken. If the item assertion is satisfied before
    # TODO: the action is executed then NO UPDATE.

    # "a trial for which the result was already satisfied before the action was taken does not count as a
    # positive-transition trial; and one for which the result was already unsatisfied does not count
    # as a negative-transition trial" (see Drescher, 1991, p. 72)
    def update(self, item_on: bool, activated: bool, count: int = 1) -> None:
        if item_on and activated:
            self._n_on_and_activated += count

        elif item_on and not activated:
            self._n_on_and_not_activated += count

        elif not item_on and activated:
            self._n_off_and_activated += count

        elif not item_on and not activated:
            self._n_off_and_not_activated += count

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
            p_on_and_activated = (0.0
                                  if self._schema_stats.n_activated == 0
                                  else self.n_on_and_activated / self._schema_stats.n_activated)
            p_on_and_not_activated = (0.0
                                      if self._schema_stats.n_not_activated == 0
                                      else self.n_on_and_not_activated / self._schema_stats.n_not_activated)

            p_on = p_on_and_activated + p_on_and_not_activated

            # calculate the ratio p(on AND activated) : p(on AND NOT activated)
            positive_trans_corr = p_on_and_activated / p_on
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
            p_off_and_activated = (0.0
                                   if self._schema_stats.n_activated == 0
                                   else self.n_off_and_activated / self._schema_stats.n_activated)
            p_off_and_not_activated = (0.0
                                       if self._schema_stats.n_not_activated == 0
                                       else self.n_off_and_not_activated / self._schema_stats.n_not_activated)

            p_off = p_off_and_activated + p_off_and_not_activated

            # calculate the ratio p(off AND activated) : p(off AND NOT activated)
            negative_trans_corr = p_off_and_activated / p_off
            return negative_trans_corr
        except ZeroDivisionError:
            return np.NAN

    @property
    def n_on(self) -> int:
        return self._n_on_and_activated + self._n_on_and_not_activated

    @property
    def n_off(self) -> int:
        return self._n_off_and_activated + self._n_off_and_not_activated

    @property
    def n_on_and_activated(self) -> int:
        return self._n_on_and_activated

    @property
    def n_on_and_not_activated(self) -> int:
        return self._n_on_and_not_activated

    @property
    def n_off_and_activated(self) -> int:
        return self._n_off_and_activated

    @property
    def n_off_and_not_activated(self) -> int:
        return self._n_off_and_not_activated


class ReadOnlySchemaStats(SchemaStats):
    def update(self, *args, **kwargs):
        raise NotImplementedError('Update not implemented for readonly view.')


class ReadOnlyECItemStats(ECItemStats):
    def update(self, *args, **kwargs):
        raise NotImplementedError('Update not implemented for readonly view.')


class ReadOnlyERItemStats(ERItemStats):
    def update(self, *args, **kwargs):
        raise NotImplementedError('Update not implemented for readonly view.')


# A single immutable object that is meant to be used for all item instances that have never had stats updates
NULL_SCHEMA_STATS = ReadOnlySchemaStats()
NULL_EC_ITEM_STATS = ReadOnlyECItemStats(NULL_SCHEMA_STATS)
NULL_ER_ITEM_STATS = ReadOnlyERItemStats(NULL_SCHEMA_STATS)


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


# TODO: Need a mechanism to alert the associated schema when an item becomes relevant (AN OBSERVER)
class ExtendedResult(Observable):
    def __init__(self, schema_stats: SchemaStats) -> None:
        super().__init__()

        self._schema_stats = schema_stats

        self._item_pool = ReadOnlyItemPool()
        self._stats: Dict[Item, ERItemStats] = defaultdict(lambda: NULL_ER_ITEM_STATS)

    @property
    def stats(self) -> Dict[Item, ERItemStats]:
        return self._stats

    def update(self, item: Item, is_on: bool, action_taken=False) -> None:
        item_stats = self._stats[item]
        if item_stats is NULL_ER_ITEM_STATS:
            self._stats[item] = item_stats = ERItemStats(schema_stats=self._schema_stats)

        item_stats.update(is_on, action_taken)

    # TODO: Try to optimize this. The vast majority of the items in each extended context should have identical
    # TODO: statistics.
    def update_all(self, view: ItemPoolStateView, action_taken=False) -> None:
        for item in self._item_pool.items:
            self.update(item, view.is_on(item), action_taken)


class ExtendedContext(Observable):
    def __init__(self, schema_stats: SchemaStats) -> None:
        super().__init__()

        self._schema_stats = schema_stats

        self._item_pool = ReadOnlyItemPool()
        self._stats: Dict[Item, ECItemStats] = defaultdict(lambda: NULL_EC_ITEM_STATS)

    @property
    def stats(self) -> Dict[Item, ECItemStats]:
        return self._stats

    def update(self, item: Item, is_on: bool, success: bool) -> None:
        item_stats = self._stats[item]
        if item_stats is NULL_EC_ITEM_STATS:
            self._stats[item] = item_stats = ECItemStats(schema_stats=self._schema_stats)

        item_stats.update(is_on, success)

    # TODO: Try to optimize this. The vast majority of the items in each extended context should have identical
    # TODO: statistics.
    def update_all(self, view: ItemPoolStateView, success: bool) -> None:
        for item in self._item_pool.items:
            self.update(item, view.is_on(item), success)


class Schema:
    """
    a three-component data structure used to express a prediction about the environmental state that
    will result from taking a particular action when in a given environmental state (i.e., context).

    Note: A schema is not a rule that says to take a particular action when its context is satisfied;
    the schema just says what might happen if that action were taken.
    """

    def __init__(self, action: Action, context: Optional[Context] = None, result: Optional[Result] = None):
        super().__init__()

        self._context: Optional[Context] = context
        self._action: Action = action
        self._result: Optional[Result] = result

        if self.action is None:
            raise ValueError('Action cannot be None')

        self._stats: SchemaStats = SchemaStats()

        self._extended_context: ExtendedContext = ExtendedContext(self._stats)
        self._extended_result: ExtendedResult = ExtendedResult(self._stats)

        self._overriding_conditions: Optional[StateAssertion] = None

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
    def extended_context(self) -> ExtendedContext:
        return self._extended_context

    @property
    def extended_result(self) -> ExtendedResult:
        return self._extended_result

    @property
    def overriding_conditions(self) -> StateAssertion:
        return self._overriding_conditions

    @overriding_conditions.setter
    def overriding_conditions(self, overriding_conditions: StateAssertion):
        self._overriding_conditions = overriding_conditions

    @property
    def stats(self) -> SchemaStats:
        return self._stats

    def is_applicable(self, state: Collection[StateElement], *args, **kwargs) -> bool:
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

    def update(self, activated: bool, success: bool, state_diff: Collection[StateElement]) -> None:
        self._stats.update(activated=activated, success=success)

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
