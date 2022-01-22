from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from collections import defaultdict
from enum import Enum
from enum import auto
from enum import unique
from functools import lru_cache
from typing import Any
from typing import Collection
from typing import Dict
from typing import FrozenSet
from typing import Hashable
from typing import Iterator
from typing import List
from typing import MutableSet
from typing import Optional
from typing import Type

import numpy as np

from schema_mechanism.util import Observable
from schema_mechanism.util import Observer
from schema_mechanism.util import Singleton
from schema_mechanism.util import UniqueIdMixin
from schema_mechanism.util import repr_str

# Type Aliases
##############
StateElement = Hashable


# Classes
#########
class Item(ABC):
    def __init__(self, state_element: StateElement):
        self._state_element = state_element

        # TODO: Need to add value primitive values.
        # TODO: Need to add value delegated values.

        # TODO: Not clear whether these value should be in a separate decorator class, or in the base class since
        # TODO: they are only used in an action selection context.

    @property
    def state_element(self) -> StateElement:
        return self._state_element

    @abstractmethod
    def is_on(self, state: Collection[StateElement], *args, **kwargs) -> bool:
        return NotImplemented

    def is_off(self, state: Collection[StateElement], *args, **kwargs) -> bool:
        return not self.is_on(state, *args, **kwargs)

    @abstractmethod
    def copy(self) -> Item:
        return NotImplemented

    # TODO: Need to be really careful with the default hash implementations which produce different values between
    # TODO: runs. This will kill and direct serialization/deserialization of data structures that rely on hashes.

    def __str__(self) -> str:
        return str(self._state_element)

    def __repr__(self) -> str:
        return repr_str(self, {'state_element': str(self._state_element)})


class ItemPool(metaclass=Singleton):
    _items: Dict[StateElement, Item] = dict()

    def __contains__(self, state_element: StateElement):
        return state_element in self._items

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[Item]:
        return iter(self._items.values())

    def clear(self):
        self._items.clear()

        self.get.cache_clear()

    @lru_cache
    def get(self, state_element: StateElement, item_type: Optional[Type[Item]] = None, **kwargs) -> Item:
        obj = self._items.get(state_element)
        if obj is None and not kwargs.get('read_only', False):
            self._items[state_element] = obj = item_type(state_element) if item_type else SymbolicItem(state_element)
        return obj


class ReadOnlyItemPool(ItemPool):
    def __init__(self):
        self._pool = ItemPool()

    def get(self, state_element: StateElement, item_type: Optional[Type[Item]] = None, **kwargs) -> Item:
        kwargs['read_only'] = True
        return self._pool.get(state_element, item_type, **kwargs)

    def clear(self):
        raise NotImplementedError('ReadOnlyItemPool does not support clear operation.')


# TODO: What to do if the underlying item pool changes?
class ItemPoolStateView:
    def __init__(self, state: Optional[Collection[StateElement]]):
        # The state to which this view corresponds
        self._state = state
        self._on_items = set([item for item in ReadOnlyItemPool() if item.is_on(state)]) if state else set()

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

    def copy(self) -> Item:
        return SymbolicItem(self.state_element)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, SymbolicItem):
            return self.state_element == other.state_element
        return False if other is None else NotImplemented

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

    def is_on(self, state: Collection[StateElement], *args, **kwargs) -> bool:
        return self._item.is_on(state, *args, **kwargs)

    def is_satisfied(self, state: Collection[StateElement], *args, **kwargs) -> bool:
        if self._negated:
            return self._item.is_off(state, *args, **kwargs)
        else:
            return self._item.is_on(state, *args, **kwargs)

    def is_satisfied_in_view(self, view: ItemPoolStateView) -> bool:
        return self._negated and view.is_on(self._item)

    def copy(self) -> ItemAssertion:
        """ Performs a shallow copy of this ItemAssertion. """
        return ItemAssertion(item=self.item, negated=self.negated)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, ItemAssertion):
            return self._item == other._item and self._negated == other._negated
        elif isinstance(other, Item):
            return self._item == other

        return False if other is None else NotImplemented

    def __hash__(self) -> int:
        """ Returns a hash of this object.

        :return: an integer hash of this object
        """
        return hash((self._item, self._negated))

    def __str__(self) -> str:
        return f'{"~" if self._negated else ""}{self._item.state_element}'

    def __repr__(self) -> str:
        return repr_str(self, {'item': self.item,
                               'negated': self.negated})


class SchemaStats:
    def __init__(self):
        self._n = 0  # total number of update events
        self._n_activated = 0
        self._n_success = 0

    def update(self, activated: bool = False, success: bool = False, count: int = 1):
        self._n += count

        if activated:
            self._n_activated += count

            if success:
                self._n_success += count

    @property
    def n(self) -> int:
        """ Returns the number of times this schema's stats were updated.

            A schema is updated whenever its context is satisfied.

            :return: the total number of schema updates (regardless of activation)
        """
        return self._n

    @property
    def n_activated(self) -> int:
        """ Returns the number of times this schema was activated (explicitly or implicitly).

            “To activate a schema is to initiate its action when the schema is applicable.” (Drescher, 1991, p.53)

            :return: the number of activations (explicit or implicit)
        """
        return self._n_activated

    @property
    def n_not_activated(self) -> int:
        """ Returns the number of times this schema was not chosen for activation (explicitly or implicitly).

            “To activate a schema is to initiate its action when the schema is applicable.” (Drescher, 1991, p.53)

            :return: the number of times the schema was not chosen for activation (explicit or implicit)
        """
        return self._n - self._n_activated

    @property
    def n_success(self) -> int:
        """ Returns the number of times this schema succeeded on activation.

            “An activated schema is said to succeed if its predicted results all in fact obtain, and to
            fail otherwise.” (Drescher, 1991, p.53)

        :return: the number of successes
        """
        return self._n_success

    @property
    def n_fail(self) -> int:
        """ Returns the number of times this schema failed on activation.

            “An activated schema is said to succeed if its predicted results all in fact obtain, and to
            fail otherwise.” (Drescher, 1991, p.53)

        :return: the number of failures
        """
        return self._n_activated - self._n_success

    def __repr__(self):
        attr_values = {
            'n': f'{self.n:,}',
            'n_activated': f'{self.n_activated:,}',
            'n_success': f'{self.n_success:,}',
        }

        return repr_str(self, attr_values)


class ItemStats:
    pass


class ECItemStats(ItemStats):
    """ Extended context item-level statistics

        "each extended-context slot records the ratio of the probability that the schema will succeed (i.e., that
        its result will obtain) if the schema is activated when the slot's item is On, to the probability
        of success if that item is Off when the schema is activated." (Drescher, 1991, p. 73)
    """

    def __init__(self, schema_stats: SchemaStats):
        # FIXME: Not used at the moment. Kept in the interface for consistency with ERItemStats.
        self._schema_stats = schema_stats

        self._n_success_and_on = 0
        self._n_success_and_off = 0
        self._n_fail_and_on = 0
        self._n_fail_and_off = 0

    def update(self, on: bool, success: bool, count: int = 1) -> None:
        if on and success:
            self._n_success_and_on += count

        elif on and not success:
            self._n_fail_and_on += count

        elif not on and success:
            self._n_success_and_off += count

        elif not on and not success:
            self._n_fail_and_off += count

    @property
    def success_corr(self) -> float:
        """ Returns the ratio p(success | item on) : p(success | item off)

        :return: the ratio as a float, or numpy.NAN if division by zero
        """
        try:
            # calculate conditional probabilities
            p_success_given_on = self._n_success_and_on / self.n_on
            p_success_given_off = self._n_success_and_off / self.n_off

            # the part-to-part ratio between p(success | on) : p(success | off)
            return p_success_given_on / (p_success_given_on + p_success_given_off)
        except ZeroDivisionError:
            return np.NAN

    @property
    def failure_corr(self) -> float:
        """ Returns the ratio p(failure | item on) : p(failure | item off)

        :return: the ratio as a float, or numpy.NAN if division by zero
        """
        try:
            # calculate conditional probabilities
            p_fail_given_on = self._n_fail_and_on / self.n_on
            p_fail_given_off = self._n_fail_and_off / self.n_off

            # the part-to-part ratio between p(failure | on) : p(failure | off)
            return p_fail_given_on / (p_fail_given_on + p_fail_given_off)
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

    def copy(self) -> ECItemStats:
        new = ECItemStats(self._schema_stats)

        new._n_success_and_on = self._n_success_and_on
        new._n_success_and_off = self._n_success_and_off
        new._n_fail_and_on = self._n_fail_and_on
        new._n_fail_and_off = self._n_fail_and_off

        return new

    def __eq__(self, other) -> bool:
        if isinstance(other, ECItemStats):
            # the "s is o" check is to handle np.nan fields
            return all({s is o or s == o for s, o in
                        [[self._n_success_and_on, other._n_success_and_on],
                         [self._n_success_and_off, other._n_success_and_off],
                         [self._n_fail_and_on, other._n_fail_and_on],
                         [self._n_fail_and_off, other._n_fail_and_off]]})

        return False if other is None else NotImplemented

    def __hash__(self):
        return hash((self._n_success_and_on,
                     self._n_success_and_off,
                     self._n_fail_and_on,
                     self._n_fail_and_off))

    def __str__(self) -> str:
        attr_values = (
            f'sc: {self.success_corr:.2}',
            f'fc: {self.failure_corr:.2}',
        )

        return f'{self.__class__.__name__}[{"; ".join(attr_values)}]'

    def __repr__(self):
        attr_values = {
            'success_corr': f'{self.success_corr:.2}',
            'failure_corr': f'{self.failure_corr:.2}',
            'n_on': f'{self.n_on:,}',
            'n_off': f'{self.n_off:,}',
            'n_success_and_on': f'{self.n_success_and_on:,}',
            'n_success_and_off': f'{self.n_success_and_off:,}',
            'n_fail_and_on': f'{self.n_fail_and_on:,}',
            'n_fail_and_off': f'{self.n_fail_and_off:,}',
        }

        return repr_str(self, attr_values)


class ERItemStats(ItemStats):
    """ Extended result item-level statistics """

    def __init__(self, schema_stats: SchemaStats):
        self._schema_stats = schema_stats

        self._n_on_and_activated = 0
        self._n_on_and_not_activated = 0
        self._n_off_and_activated = 0
        self._n_off_and_not_activated = 0

    def update(self, on: bool, activated: bool, count: int = 1) -> None:
        if on and activated:
            self._n_on_and_activated += count

        elif on and not activated:
            self._n_on_and_not_activated += count

        elif not on and activated:
            self._n_off_and_activated += count

        elif not on and not activated:
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
            p_on_given_activated = self.n_on_and_activated / self._schema_stats.n_activated
            p_on_given_not_activated = self.n_on_and_not_activated / self._schema_stats.n_not_activated

            # the part-to-part ratio between p(on AND activated) : p(on AND NOT activated)
            positive_trans_corr = p_on_given_activated / (p_on_given_activated + p_on_given_not_activated)
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
            p_off_given_activated = self.n_off_and_activated / self._schema_stats.n_activated
            p_off_given_not_activated = self.n_off_and_not_activated / self._schema_stats.n_not_activated

            # the part-to-part ratio between p(off AND activated) : p(off AND NOT activated)
            negative_trans_corr = p_off_given_activated / (p_off_given_activated + p_off_given_not_activated)
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

    def copy(self) -> ERItemStats:
        new = ERItemStats(self._schema_stats)

        new._n_on_and_activated = self._n_on_and_activated
        new._n_on_and_not_activated = self._n_on_and_not_activated
        new._n_off_and_activated = self._n_off_and_activated
        new._n_off_and_not_activated = self._n_off_and_not_activated

        return new

    def __eq__(self, other) -> bool:
        if isinstance(other, ERItemStats):
            # the "s is o" check is to handle np.nan fields
            return all({s is o or s == o for s, o in
                        [[self._n_on_and_activated, other._n_on_and_activated],
                         [self._n_off_and_activated, other._n_off_and_activated],
                         [self._n_on_and_not_activated, other._n_on_and_not_activated],
                         [self._n_off_and_not_activated, other._n_off_and_not_activated]]})

        return False if other is None else NotImplemented

    def __hash__(self):
        return hash((self._n_on_and_activated,
                     self._n_off_and_activated,
                     self._n_on_and_not_activated,
                     self._n_off_and_not_activated))

    def __str__(self) -> str:
        attr_values = (
            f'ptc: {self.positive_transition_corr:.2}',
            f'ntc: {self.negative_transition_corr:.2}',
        )

        return f'{self.__class__.__name__}[{"; ".join(attr_values)}]'

    def __repr__(self) -> str:
        attr_values = {
            'positive_transition_corr': f'{self.positive_transition_corr:.2}',
            'negative_transition_corr': f'{self.negative_transition_corr:.2}',
            'n_on': f'{self.n_on:,}',
            'n_off': f'{self.n_off:,}',
            'n_on_and_activated': f'{self.n_on_and_activated:,}',
            'n_on_and_not_activated': f'{self.n_on_and_not_activated:,}',
            'n_off_and_activated': f'{self.n_off_and_activated:,}',
            '_n_off_and_not_activated': f'{self._n_off_and_not_activated:,}',
        }

        return repr_str(self, attr_values)


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


class Action(UniqueIdMixin):
    _last_uid: int = 0

    def __init__(self, label: Optional[str] = None):
        super().__init__()

        self._label = label

    @property
    def uid(self) -> int:
        """ a globally unique id for this action

        :return: returns this object's unique id
        """
        return self._uid

    @property
    def label(self) -> Optional[str]:
        """ A description of this action.

        :return: returns the Action's label
        """
        return self._label

    def copy(self) -> Action:
        # bypasses initializer to force reuse of uid
        copy = super().__new__(Action)
        copy._uid = self._uid
        copy._label = self._label

        return copy

    def __eq__(self, other):
        if isinstance(other, Action):
            if self._label and other._label:
                return self._label == other._label
            else:
                return self._uid == other._uid

        return False if other is None else NotImplemented

    def __hash__(self) -> int:
        return hash(self._label or self._uid)

    def __str__(self) -> str:
        return self._label or str(self.uid)

    def __repr__(self) -> str:
        return repr_str(self, {'uid': self.uid,
                               'label': self.label})


class StateAssertion:

    def __init__(self, item_asserts: Optional[Collection[ItemAssertion, ...]] = None):
        self._pos_asserts = frozenset(filter(lambda ia: not ia.negated, item_asserts)) if item_asserts else frozenset()
        self._neg_asserts = frozenset(filter(lambda ia: ia.negated, item_asserts)) if item_asserts else frozenset()

    def __iter__(self) -> Iterator[ItemAssertion]:
        yield from self._pos_asserts
        yield from self._neg_asserts

    def __len__(self) -> int:
        return len(self._pos_asserts) + len(self._neg_asserts)

    def __contains__(self, item_assert: ItemAssertion) -> bool:
        return (item_assert in self._neg_asserts if item_assert.negated
                else item_assert in self._pos_asserts)

    def __eq__(self, other) -> bool:
        if isinstance(other, StateAssertion):
            return (self._pos_asserts == other._pos_asserts
                    and self._neg_asserts == other._neg_asserts)

        return False if other is None else NotImplemented

    def __hash__(self) -> int:
        return hash((self._pos_asserts, self._neg_asserts))

    def __str__(self) -> str:
        return ','.join(map(str, self))

    def __repr__(self) -> str:
        return repr_str(self, {'item_asserts': str(self)})

    def copy(self) -> StateAssertion:
        """ Returns a shallow copy of this object.

        :return: the copy
        """
        new = super().__new__(StateAssertion)

        new._pos_asserts = self._pos_asserts
        new._neg_asserts = self._neg_asserts

        return new

    def is_satisfied(self, state: Collection[StateElement], *args, **kwargs) -> bool:
        """ Satisfied when all non-negated items are On, and all negated items are Off.

        :param state: the agent's current state
        :param args: optional positional arguments
        :param kwargs: optional keyword arguments
        :return: True if this state assertion is satisfied given the current state; False otherwise.
        """
        return all({ia.is_satisfied(state, *args, **kwargs) for ia in self})

    def is_satisfied_in_view(self, view: ItemPoolStateView) -> bool:
        """ Satisfied when all non-negated items are On, and all negated items are Off.

        :param view: an item pool view
        :return: True if this state assertion is satisfied given the current state; False otherwise.
        """
        return all({not ia.negated and view.is_on(ia.item) for ia in self})

    def replicate_with(self, item_assert: ItemAssertion) -> StateAssertion:
        if self.__contains__(item_assert):
            raise ValueError('ItemAssertion already exists in StateAssertion')

        return StateAssertion(item_asserts=(*self, item_assert))


NULL_STATE_ASSERT = StateAssertion()


# TODO: There will be many shared item assertions between extended contexts and extended results. It would
# TODO: be very beneficial to have a pool for these as well.
class ItemAssertionPool:
    pass


class ExtendedItemCollection(Observable):
    # thresholds for determining the relevance of items (possible that both should always have the same value)
    POS_CORR_RELEVANCE_THRESHOLD = 0.65
    NEG_CORR_RELEVANCE_THRESHOLD = 0.65

    def __init__(self, schema_stats: SchemaStats, null_member: ItemStats = None):
        super().__init__()

        self._schema_stats = schema_stats
        self._null_member = null_member

        self._stats: Dict[Item, ItemStats] = defaultdict(lambda: self._null_member)

        # TODO: Should this really be a mutable Item Pool??? What is in place upstream to guarantee that all
        # TODO: state elements in the current state have been added to the item pool? Wouldn't it be better
        # TODO: just to allow new items to be added "in passing"
        self._item_pool = ItemPool()

        self._relevant_items: MutableSet[ItemAssertion] = set()
        self._new_relevant_items: MutableSet[ItemAssertion] = set()

    @property
    def stats(self) -> Dict[Item, Any]:
        return self._stats

    @property
    def relevant_items(self) -> FrozenSet[ItemAssertion]:
        return frozenset(self._relevant_items)

    def update_relevant_items(self, item_assert: ItemAssertion):
        if item_assert not in self._relevant_items:
            self._relevant_items.add(item_assert)
            self._new_relevant_items.add(item_assert)

    def known_relevant_item(self, item_assert: ItemAssertion) -> bool:
        return item_assert in self._relevant_items

    def notify_all(self, *args, **kwargs) -> None:
        if 'source' not in kwargs:
            kwargs['source'] = self

        super().notify_all(*args, **kwargs)

        # clears the set
        self._new_relevant_items = set()

    @property
    def new_relevant_items(self) -> FrozenSet[ItemAssertion]:
        return frozenset(self._new_relevant_items)

    def __str__(self) -> str:
        name = self.__class__.__name__
        stats = '; '.join([f'{k} -> {v}' for k, v in self._stats.items()])

        return f'{name}[{stats}]'

    def __repr__(self) -> str:
        item_stats = ', '.join(['{} -> {}'.format(k, v) for k, v in self.stats.items()])
        relevant_items = ', '.join(str(i) for i in self.relevant_items)
        new_relevant_items = ', '.join(str(i) for i in self.new_relevant_items)

        attr_values = {
            'stats': f'[{item_stats}]',
            'relevant_items': f'[{relevant_items}]',
            'new_relevant_items': f'[{new_relevant_items}]',
        }

        return repr_str(self, attr_values)


# TODO: Need to suppress item relevance when those items appear in the parent schema's result (THIS MIGHT BE SOMEThING
# TODO: FOR THE SUPERCLASS)

# TODO: ExtendedContext and ExtendedResult test_share many attributes and a lot of behavior. It may be beneficial
# TODO: to create a shared parent class from which they both inherit.
class ExtendedResult(ExtendedItemCollection):

    def __init__(self, schema_stats: SchemaStats) -> None:
        super().__init__(schema_stats, null_member=NULL_ER_ITEM_STATS)

    @property
    def stats(self) -> Dict[Item, ERItemStats]:
        return super().stats

    def update(self, item: Item, on: bool, activated=False, count: int = 1) -> None:
        item_stats = self._stats[item]
        if item_stats is NULL_ER_ITEM_STATS:
            self._stats[item] = item_stats = ERItemStats(schema_stats=self._schema_stats)

        item_stats.update(on=on, activated=activated, count=count)

        if item_stats.positive_transition_corr > self.POS_CORR_RELEVANCE_THRESHOLD:
            item_assert = ItemAssertion(item)
            if not self.known_relevant_item(item_assert):
                self.update_relevant_items(item_assert)

        elif item_stats.negative_transition_corr > self.NEG_CORR_RELEVANCE_THRESHOLD:
            item_assert = ItemAssertion(item, negated=True)
            if not self.known_relevant_item(item_assert):
                self.update_relevant_items(item_assert)

    # TODO: Try to optimize this. The vast majority of the items in each extended context should have identical
    # TODO: statistics.
    def update_all(self, activated, new: Collection[StateElement], lost: Collection[StateElement],
                   count: int = 1) -> None:

        # "a trial for which the result was already satisfied before the action was taken does not count as a
        # positive-transition trial; and one for which the result was already unsatisfied does not count
        # as a negative-transition trial" (see Drescher, 1991, p. 72)
        for se in new:
            self.update(item=self._item_pool.get(se), on=True, activated=activated, count=count)

        for se in lost:
            self.update(item=self._item_pool.get(se), on=False, activated=activated, count=count)

        if self.new_relevant_items:
            self.notify_all(source=self)


# TODO: Need to suppress item relevance when those items appear in the parent schema's context (THIS MIGHT BE SOMEThING
# TODO: FOR THE SUPERCLASS)

class ExtendedContext(ExtendedItemCollection):
    """
        a schema’s extended context tried to identify conditions under which the result more
        reliably follows the action. Each extended context slot keeps track of whether the schema
        is significantly more reliable when the associated item is On (or Off). When the mechanism
        thus discovers an item whose state is relevant to the schema’s reliability, it adds that
        item (or its negation) to the context of a spin-off schema. (see Drescher, 1987, p. 291)

        Supports the discovery of:

            reliable schemas (see Drescher, 1991, Section 4.1.2)
            overriding conditions (see Drescher, 1991, Section 4.1.5)
            sustained context conditions (see Drescher, 1991, Section 4.1.6)
            conditions for turning Off a synthetic item (see Drescher, 1991, Section 4.2.2)
    """

    def __init__(self, schema_stats: SchemaStats) -> None:
        super().__init__(schema_stats, null_member=NULL_EC_ITEM_STATS)

    @property
    def stats(self) -> Dict[Item, ECItemStats]:
        return super().stats

    def update(self, item: Item, on: bool, success: bool, count: int = 1) -> None:
        item_stats = self._stats[item]
        if item_stats is NULL_EC_ITEM_STATS:
            self._stats[item] = item_stats = ECItemStats(schema_stats=self._schema_stats)

        item_stats.update(on, success, count)

        if item_stats.success_corr > self.POS_CORR_RELEVANCE_THRESHOLD:
            item_assert = ItemAssertion(item)
            if not self.known_relevant_item(item_assert):
                self.update_relevant_items(ItemAssertion(item))

        elif item_stats.failure_corr > self.NEG_CORR_RELEVANCE_THRESHOLD:
            item_assert = ItemAssertion(item, negated=True)
            if not self.known_relevant_item(item_assert):
                self.update_relevant_items(item_assert)

    # TODO: Try to optimize this. The vast majority of the items in each extended context should have identical
    # TODO: statistics.
    def update_all(self, view: ItemPoolStateView, success: bool, count: int = 1) -> None:
        for item in self._item_pool:
            self.update(item=item, on=view.is_on(item), success=success, count=count)

        if self.new_relevant_items:
            self.notify_all(source=self)


# TODO: Candidate for the flyweight pattern
class Schema(Observer, Observable, UniqueIdMixin):
    """
    a three-component data structure used to express a prediction about the environmental state that
    will result from taking a particular action when in a given environmental state (i.e., context).

    Note: A schema is not a rule that says to take a particular action when its context is satisfied;
    the schema just says what might happen if that action were taken. Furthermore, the schema's result is
    not an exhaustive list of all environmental consequences of the schema, neither is the schema's context
    an exhaustive list of all environmental conditions under which the schema's result may obtain if the
    action were taken.
    """

    @unique
    class SpinOffType(Enum):
        CONTEXT = auto(),  # (see Drescher, 1991, p. 73)
        RESULT = auto()  # (see Drescher, 1991, p. 71)

    def __init__(self,
                 action: Action,
                 context: Optional[StateAssertion] = None,
                 result: Optional[StateAssertion] = None):
        super().__init__()

        self._context: Optional[StateAssertion] = context or NULL_STATE_ASSERT
        self._action: Action = action
        self._result: Optional[StateAssertion] = result or NULL_STATE_ASSERT

        if self.action is None:
            raise ValueError('Action cannot be None')

        self._stats: SchemaStats = SchemaStats()

        self._extended_context: ExtendedContext = ExtendedContext(self._stats)
        self._extended_result: ExtendedResult = ExtendedResult(self._stats)

        # TODO: Need to update overriding conditions.
        self._overriding_conditions: Optional[StateAssertion] = None

        # This observer registration is used to notify the schema when a relevant item has been detected in its
        # extended context or extended result
        self._extended_context.register(self)
        self._extended_result.register(self)

        self._result_spinoffs: List[Schema] = list()

        # TODO: Are duration or cost needed?

        # The duration is the average time from the activation to the completion of an action.
        # self.duration = Schema.INITIAL_DURATION

        # The cost is the minimum (i.e., the greatest magnitude) of any negative-valued results
        # of schemas that are implicitly activated as a side effect of the given schema’s [explicit]
        # activation on that occasion (see Drescher, 1991, p.55).
        # self.cost = Schema.INITIAL_COST

    @property
    def context(self) -> StateAssertion:
        return self._context

    @property
    def action(self) -> Action:
        return self._action

    @property
    def result(self) -> StateAssertion:
        return self._result

    @property
    def extended_context(self) -> ExtendedContext:
        return self._extended_context

    @property
    def extended_result(self) -> ExtendedResult:
        return self._extended_result

    @property
    def overriding_conditions(self) -> StateAssertion:
        """

        :return: a StateAssertion containing the overriding conditions
        """
        return self._overriding_conditions

    # TODO: How do we set these overriding conditions???
    @overriding_conditions.setter
    def overriding_conditions(self, overriding_conditions: StateAssertion) -> None:
        self._overriding_conditions = overriding_conditions

    @property
    def reliability(self) -> float:
        """ Returns the schema's reliability (i.e., its probability of success) when activated.

            "A schema's reliability is the probability with which the schema succeeds when activated."
            (Drescher, 1991, p. 54)

        :return: the schema's reliability
        """
        return np.NAN if self.stats.n_activated == 0 else self.stats.n_success / self.stats.n_activated

    @property
    def stats(self) -> SchemaStats:
        return self._stats

    @property
    def result_spinoffs(self) -> List[Schema]:
        return self._result_spinoffs

    @result_spinoffs.setter
    def result_spinoffs(self, new_value) -> None:
        self._result_spinoffs = new_value

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

    def is_primitive(self) -> bool:
        """ Returns whether this instance is a primitive (action-only) schema.

        :return: True if this is a primitive schema; False otherwise.
        """
        return self.context is NULL_STATE_ASSERT and self.result is NULL_STATE_ASSERT

    def update(self,
               activated: bool,
               v_prev: Optional[ItemPoolStateView],
               v_curr: ItemPoolStateView,
               new: Collection[StateElement] = None,
               lost: Collection[StateElement] = None, count=1) -> None:
        """

            Note: As a result of these updates, one or more notifications may be generated by the schema's
            context and/or result. These will be received via schema's 'receive' method.

        :param activated: True if this schema was implicitly or explicitly activated; False otherwise
        :param v_prev: a view of the set of items that were On in the item pool for the previous state
        :param v_curr: a view of the set of items that are On in the item pool for the current state
        :param new: the state elements in current but not previous state
        :param lost: the state elements in previous but not current state
        :param count: the number of updates to perform

        :return: None
        """

        # True if this schema was activated AND its result obtained; False otherwise
        success: bool = activated and self.result.is_satisfied_in_view(v_curr)

        # update top-level stats
        self._stats.update(activated=activated, success=success, count=count)

        # update extended result stats
        if not all((new, lost)):
            self._extended_result.update_all(activated=activated, new=new, lost=lost, count=count)

        # update extended context stats
        if activated and v_prev:
            self._extended_context.update_all(view=v_prev, success=success, count=count)

    # invoked by the schema's extended context or extended result when one of their items is
    # determined to be relevant
    def receive(self, *args, **kwargs) -> None:

        # ext_source should be an ExtendedContext, ExtendedResult, or one of their subclasses
        ext_source: ExtendedItemCollection = kwargs['source']
        relevant_items: Collection[ItemAssertion] = ext_source.new_relevant_items

        self.notify_all(source=self,
                        ext_source=ext_source,
                        relevant_items=relevant_items)

    def copy(self) -> Schema:
        """ Returns a copy of this schema that is equal to its parent.

            Note: The copy returned from this method replicates the immutable components of this schema. As such,
            it is essentially the parent schema before any learning.

        :return: a copy of this Schema
        """
        new = Schema(context=self._context,
                     action=self._action,
                     result=self._result)

        new._uid = self._uid

        return new

    def __eq__(self, other) -> bool:
        if isinstance(other, Schema):
            return all({s == o for s, o in
                        [[self._context, other._context],
                         [self._action, other._action],
                         [self._result, other._result]]})

        return False if other is None else NotImplemented

    def __hash__(self) -> int:
        return hash((self._context,
                     self._action,
                     self._result,))

    def __str__(self) -> str:
        return f'{self.context}/{self.action}/{self.result}'

    def __repr__(self) -> str:
        return repr_str(self, {'uid': self.uid,
                               'context': self.context,
                               'action': self.action,
                               'result': self.result,
                               'overriding_conditions': self.overriding_conditions,
                               'reliability': self.reliability, })
