from __future__ import annotations

import itertools
import logging
from abc import ABC
from abc import ABCMeta
from abc import abstractmethod
from collections import defaultdict
from collections import deque
from collections.abc import Collection
from collections.abc import Iterator
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from functools import singledispatch
from functools import singledispatchmethod
from time import time
from typing import Any
from typing import Callable
from typing import Hashable
from typing import Iterable
from typing import NamedTuple
from typing import Optional
from typing import Protocol
from typing import Type
from typing import runtime_checkable

import numpy as np
from anytree import AsciiStyle
from anytree import LevelOrderIter
from anytree import NodeMixin
from anytree import RenderTree

from schema_mechanism.parameters import GlobalParams
from schema_mechanism.share import SupportedFeature
from schema_mechanism.strategies import correlation_test
from schema_mechanism.strategies.correlation_test import CorrelationTable
from schema_mechanism.strategies.correlation_test import ItemCorrelationTest
from schema_mechanism.strategies.decay import GeometricDecayStrategy
from schema_mechanism.strategies.trace import AccumulatingTrace
from schema_mechanism.strategies.trace import ReplacingTrace
from schema_mechanism.strategies.trace import Trace
from schema_mechanism.util import AssociativeArrayList
from schema_mechanism.util import DefaultDictWithKeyFactory
from schema_mechanism.util import Observable
from schema_mechanism.util import Observer
from schema_mechanism.util import Singleton
from schema_mechanism.util import UniqueIdMixin
from schema_mechanism.util import pairwise
from schema_mechanism.util import repr_str
from schema_mechanism.validate import RangeValidator
from schema_mechanism.validate import SupportedFeatureValidator
from schema_mechanism.validate import WhiteListValidator

logger = logging.getLogger(__name__)


@runtime_checkable
class StateElement(Hashable, Protocol, metaclass=ABCMeta):
    """
        This protocol is intended to enforce the hash-ability of state elements, and to allow for future required
        methods without demanding strict sub-classing.
    """
    pass


@runtime_checkable
class State(Collection[StateElement], Hashable, Protocol):
    """
    """
    pass


def held_state(s_prev: State, s_curr: State) -> frozenset[Item]:
    """ Returns the set of items that are On in both the current and previous states

    :param s_prev: a collection of the previous state's elements
    :param s_curr: a collection of the current state's elements

    :return: a (potentially empty) set of Items
    """
    if not all((s_prev, s_curr)):
        return frozenset()

    # singular
    held = [ItemPool().get(se) for se in s_curr if se in s_prev]

    # conjunctions
    for ci in ItemPool().composite_items:
        if ci.is_on(s_curr) and ci.is_on(s_prev):
            held.append(ci)

    return frozenset(held)


def new_state(s_prev: Optional[State], s_curr: Optional[State]) -> frozenset[Item]:
    """ Returns the set of items that are On in current state but not the previous state

    :param s_prev: a collection of the previous state's elements
    :param s_curr: a collection of the current state's elements

    :return: a (potentially empty) set of Items
    """
    if not s_curr:
        return frozenset()

    s_prev = [] if s_prev is None else s_prev

    # singular
    new = [ItemPool().get(se) for se in s_curr if se not in s_prev]

    # conjunctions
    for ci in ItemPool().composite_items:
        if ci.is_on(s_curr) and not ci.is_on(s_prev):
            new.append(ci)

    return frozenset(new)


def lost_state(s_prev: Optional[State], s_curr: Optional[State]) -> frozenset[Item]:
    """ Returns the set of items that are On in previous state but not current state

    :param s_prev: a collection of the previous state's elements
    :param s_curr: a collection of the current state's elements

    :return: a (potentially empty) set of Items
    """
    if not s_prev:
        return frozenset()

    s_curr = [] if s_curr is None else s_curr

    # singular
    lost = [ItemPool().get(se) for se in s_prev if se not in s_curr]

    # conjunctions
    for ci in ItemPool().composite_items:
        if ci.is_on(s_prev) and not ci.is_on(s_curr):
            lost.append(ci)

    return frozenset(lost)


class DelegatedValueHelper(ABC):
    """ An abstract base class for helper classes that track and derive an Item's delegated value.

    For a discussion of delegated value see Drescher 1991, p. 63.
    """

    @abstractmethod
    def get_delegated_value(self, item: Item) -> float:
        """ Gets the delegated value for the requested item.

        :param item: the item for which the delegated value will be returned.
        :return: the item's current delegated value
        """

    @abstractmethod
    def set_delegated_value(self, item: Item, value: float) -> None:
        """ Sets the delegated value for the requested item.

        (For a discussion of delegated value see Drescher 1991, p. 63.)

        :return: None
        """

    @abstractmethod
    def update(self, selection_state: State, result_state: State, **kwargs) -> None:
        """ Updates the item's delegated value.

        Delegated value should only be updated when states of non-zero value are encountered following the inclusion
        of the tracked item (i.e., the item being On) in a selection state. Note that an item's delegated value may be
        updated even if the item was On in a selection state that does not immediately precede a value-bearing result
        states. In other words, an item's delegated value may be updated even if there are many intermediate states
        between a selection state in which the tracked item was On and a value-bearing result state.

        :param selection_state: the most recent selection state.
        :param result_state: the state immediately following the selection state.
        :param kwargs: a dictionary of optional keyword arguments

        :return: None
        """

    @abstractmethod
    def reset(self) -> None:
        """ Resets the delegated value helper to its initial state prior to update.

        :return: None
        """


class EligibilityTraceDelegatedValueHelper(DelegatedValueHelper):
    """ An eligibility-trace-based implementation of the DelegatedValueHelper.

    Drescher stated that delegated value "accrues to states that generally tend to facilitate other things of
    value" (p. 63). This occurs without reference to a specific goal. In other words, delegated value is the average,
    goal-agnostic, utility of an Item being On. This implementation of the DelegatedValueHelper retains the spirit of
    Drescher's delegated value, but deviates in the specifics how delegated value is learned for performance reasons.

    In particular, Drescher's implementation of delegated values is based on forward chaining over learned schemas using
    "parallel broadcasts" (see Drescher 1991, p. 101). By contrast, the implementation provided in this class leverages
    the idea of eligibility traces from reinforcement learning to learn the goal-agnostic utility of items.

    An additional benefit of this implementation over Drescher's is that the value of more distant accessible states
    can be DISCOUNTED over that of more immediately accessible valuable states. This allows agents to discriminate
    between Items that lead to more immediate vs. more distantly valuable states. Drescher's implementation is
    undiscounted (see Drescher 1991, p. 63).

    Items are given delegated value based on the regularity with which they are On prior to the occurrence of
    (other) states of value; that is, based on whether they appear to facilitate the obtainment of those future states.
    The value delegated to those Items (i.e., the credit given to them as facilitators of future states of value) can
    be modulated by a discount factor and a trace decay parameter. The discount factor quantifies the attenuation of the
    contribution of state values based on their temporal distance from an Item's most recent On state. The trace decay
    parameter specifies a horizon over which state values can propagate back-in-time.

    For information on eligibility traces, see Chapter 12 of
        Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction. MIT press.
    """

    def __init__(self, discount_factor: float, eligibility_trace: Trace[Item]) -> None:
        """ Initializes the EligibilityTraceDelegatedValueHelper

        Note: a discount factor of 0.0 will give NO VALUE to future states when calculating delegated value (i.e.,
        delegated value will always be zero) while a discount factor of 1.0 will value more distant states EQUALLY with
        less distant states (i.e., state value will be undiscounted).

        :param discount_factor: quantifies the reduction in future state value w.r.t. an item's delegated value
        :param eligibility_trace: a trace used to determine the time horizon over which an item's delegated values
        receive credit for encountered state values.
        """
        self.discount_factor = discount_factor
        self.eligibility_trace: Trace[Item] = eligibility_trace

        self._delegated_values: AssociativeArrayList[Item] = AssociativeArrayList()

    def __eq__(self, other) -> bool:
        if isinstance(other, EligibilityTraceDelegatedValueHelper):
            return all(
                (
                    # generator expression for conditions to allow lazy evaluation
                    condition for condition in
                    [
                        self.discount_factor == other.discount_factor,
                        self.eligibility_trace == other.eligibility_trace,
                        self._delegated_values == other._delegated_values
                    ]
                )
            )
        return False if other is None else NotImplemented

    @property
    def discount_factor(self) -> float:
        """ A factor that quantifies the reduction in value per step for future rewards (0 <= Discount <= 1).

        A discount factor of 0.0 places no value on future rewards. As a result, delegated value will also be zero. A
        value of 1.0 values future and current rewards equally.
        """
        return self._discount_factor

    @discount_factor.setter
    def discount_factor(self, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f'Discount factor must be between 0.0 and 1.0 inclusive')

        self._discount_factor = value

    def get_delegated_value(self, item: Item) -> float:
        """ Returns the delegated value for the requested item.

        (For a discussion of delegated value see Drescher 1991, p. 63.)

        :return: the Item's current delegated value
        """
        index = self._delegated_values.indexes([item], add_missing=True)[0]
        return self._delegated_values.values[index]

    def set_delegated_value(self, item: Item, value: float) -> None:
        """ Sets the delegated value for this item.

        (For a discussion of delegated value see Drescher 1991, p. 63.)

        :return: None
        """
        index = self._delegated_values.indexes([item], add_missing=True)[0]
        self._delegated_values.values[index] = value

    # TODO: the determination of active items from states should be removed from this method
    # TODO:    * remove selection_state and result_state parameters
    # TODO:    * add active_items parameter
    # TODO: externalize effective_state_value calculation and pass that value in as a parameter?
    def update(self,
               selection_state: State,
               result_state: State,
               **kwargs) -> None:
        """ Updates delegated-value-related statistics based on the selection and result states.

        :param selection_state: the state from which the last schema was selected (i.e., an action taken)
        :param result_state: the state that immediately followed the provided selection state
        :param kwargs: optional keyword arguments
        :return: None
        """

        # FIXME: this is horribly inefficient. Must find a better way.
        active_items = [item for item in ReadOnlyItemPool() if item.is_on(state=selection_state, **kwargs)]

        # updating eligibility trace (decay all items, increase active items)
        self.eligibility_trace.update(active_items)

        # the value accessible from the current state
        target = self.effective_state_value(selection_state, result_state)

        # incremental update of delegated values towards target
        params = get_global_params()
        lr = params.get('learning_rate')
        for item in ReadOnlyItemPool():
            tv = self.eligibility_trace[item]
            if tv > 0.0:
                err = target - self._delegated_values[item]
                self._delegated_values[item] = self._delegated_values[item] + lr * err * tv

    def effective_state_value(self, selection_state: State, result_state: State) -> float:
        """ Calculates the effective value of the result state based on changes from previous state.

        :param selection_state: the state prior to the given result state
        :param result_state: the result state that will be evaluated

        :return: a float that quantifies the value of the result state
        """
        # only include the value of items that were not On in the selection state
        new_items = new_state(selection_state, result_state)
        if not new_items:
            return 0.0

        most_specific_new_items = reduce_to_most_specific_items(new_items)

        pv = sum(item.primitive_value for item in most_specific_new_items)
        dv = sum(item.delegated_value for item in most_specific_new_items)

        # the delegated value accessible from the current state
        return self.discount_factor * (pv + self.discount_factor * dv)

    def reset(self) -> None:
        self.eligibility_trace.clear()
        self._delegated_values.clear()


class Item(ABC, UniqueIdMixin):
    def __init__(self,
                 source: Any,
                 primitive_value: Optional[float] = None,
                 delegated_value: Optional[float] = None,
                 **kwargs) -> None:
        super().__init__()

        self._source = source
        self._primitive_value: float = 0.0 if primitive_value is None else primitive_value

        # initializes the delegated value. (The delegated value will change over time, unlike primitive value.)
        initial_delegated_value: float = 0.0 if delegated_value is None else delegated_value

        dv_helper = get_delegated_value_helper()
        dv_helper.set_delegated_value(item=self, value=initial_delegated_value)

    @abstractmethod
    def __hash__(self) -> int:
        """ Returns a hash code for this item """

    def __str__(self) -> str:
        return ','.join(sorted([str(element) for element in self.state_elements])) if self.source else ''

    def __repr__(self) -> str:
        return repr_str(self, {'source': str(self.source),
                               'pv': self.primitive_value,
                               'dv': self.delegated_value})

    @property
    def source(self) -> Any:
        return self._source

    @property
    @abstractmethod
    def state_elements(self) -> set[StateElement]:
        """ Returns the state elements contained in this item's source """

    @property
    def primitive_value(self) -> float:
        """ The primitive value for this item.

            “The schema mechanism explicitly designates an item as corresponding to a top-level goal by assigning the
             item a positive value; an item can also take on a negative value, indicating a state to be avoided.”
            (see Drescher, 1991, p. 61)
            (see also, Drescher, 1991, Section 3.4.1)

        :return: the primitive value as a float
        """
        return 0.0 if self._primitive_value is None else self._primitive_value

    @property
    def delegated_value(self) -> float:
        dv_helper = get_delegated_value_helper()
        return dv_helper.get_delegated_value(self) if dv_helper else 0.0

    @abstractmethod
    def is_on(self, state: State, **kwargs) -> bool:
        """ Returns whether this item is On (i.e, present) in the current state """

    def is_off(self, state: State, **kwargs) -> bool:
        """ Returns whether this item is Off (i.e, absent) in the current state """
        return not self.is_on(state, **kwargs)


class SymbolicItem(Item):
    """ A state element that can be thought as a proposition/feature. """

    def __init__(self,
                 source: str,
                 primitive_value: float = 0.0,
                 delegated_value: float = 0.0,
                 **kwargs):
        if not isinstance(source, str):
            raise ValueError(f'Source for symbolic item must be a str, not {type(source)}')

        super().__init__(
            source=source,
            primitive_value=primitive_value,
            delegated_value=delegated_value,
            **kwargs
        )

    def __eq__(self, other: Any) -> bool:
        # ItemPool should be used for all item creation, so this should be an optimization
        if self is other:
            return True

        if isinstance(other, SymbolicItem):
            return self.source == other.source
        return False if other is None else NotImplemented

    def __hash__(self) -> int:
        return hash(self.source)

    @property
    def source(self) -> str:
        return super().source

    @property
    def state_elements(self) -> set[str]:
        return {self.source}

    def is_on(self, state: State, **kwargs) -> bool:
        return self.source in state


def non_composite_items(items: Iterable[Item]) -> Collection[Item]:
    return list(filter(lambda i: not isinstance(i, CompositeItem), items))


def composite_items(items: Iterable[Item]) -> Collection[Item]:
    return list(filter(lambda i: isinstance(i, CompositeItem), items))


def items_from_state(state: State) -> Iterable[Item]:
    return [ItemPool().get(se) for se in state] if state else []


def item_contained_in(item_1: Item, item_2: Item) -> bool:
    set_1 = item_1.state_elements
    set_2 = item_2.state_elements

    return set_1.issubset(set_2)


def reduce_to_most_specific_items(items: Optional[Collection[Item]]) -> Collection[Item]:
    reduced_items = set(items) if items else set()

    items_to_keep = set()

    while reduced_items:
        item = reduced_items.pop()

        check_set = reduced_items.union(items_to_keep)
        contained_in = check_set and any((item_contained_in(item, o) for o in check_set))
        if not contained_in:
            items_to_keep.add(item)

    return items_to_keep


@dataclass
class GlobalStats:
    n: int = 0
    baseline_value: float = 0.0

    def update(self, selection_state: State, result_state: State) -> None:
        """ Updates the global statistics based on the selection and result states.

        :param selection_state: the selection state from the last selection event
        :param result_state: the result state from the last selection event

        :return: None
        """
        params = get_global_params()
        lr = params.get('learning_rate')

        self.n += 1

        # updates baseline
        self.baseline_value += lr * (calc_primitive_value(result_state) - self.baseline_value)

    def reset(self):
        self.baseline_value = 0.0
        self.n = 0


@dataclass
class SchemaStats:
    n: int = 0
    n_activated: int = 0
    n_success: int = 0

    def update(self, activated: bool = False, success: bool = False, count: int = 1):
        self.n += count

        if activated:
            self.n_activated += count

            if success:
                self.n_success += count

    @property
    def n_not_activated(self) -> int:
        """ Returns the number of times this schema was not chosen for activation (explicitly or implicitly).

            “To activate a schema is to initiate its action when the schema is applicable.” (Drescher, 1991, p.53)

            :return: the number of times the schema was not chosen for activation (explicit or implicit)
        """
        return self.n - self.n_activated

    @property
    def n_fail(self) -> int:
        """ Returns the number of times this schema failed on activation.

            “An activated schema is said to succeed if its predicted results all in fact obtain, and to
            fail otherwise.” (Drescher, 1991, p.53)

        :return: the number of failures
        """
        return self.n_activated - self.n_success


class ItemStats(ABC):

    @property
    @abstractmethod
    def correlation_test(self) -> ItemCorrelationTest:
        """ Returns a reference to the item correlation test used to generate item stats """

    @property
    @abstractmethod
    def positive_correlation_threshold(self) -> float:
        """ The threshold used to determine if positive correlation needed for relevant items designation """

    @property
    @abstractmethod
    def negative_correlation_threshold(self) -> float:
        """ The threshold used to determine if negative correlation needed for relevant items designation """

    @property
    def positive_correlation_stat(self) -> float:
        """ Returns a real number that quantifies this item's positive correlation value """
        return self.correlation_test.positive_corr_statistic(self.as_table())

    @property
    def negative_correlation_stat(self) -> float:
        """ Returns a real number that quantifies this item's negative correlation value """
        return self.correlation_test.negative_corr_statistic(self.as_table())

    @property
    def positive_correlation(self) -> bool:
        """ Returns True if this Item is positively correlated based on the current threshold; False otherwise """
        return self.correlation_test.positive_corr_statistic(self.as_table()) >= self.positive_correlation_threshold

    @property
    def negative_correlation(self) -> bool:
        """ Returns True if this Item is negatively correlated based on the current threshold; False otherwise """
        return self.correlation_test.negative_corr_statistic(self.as_table()) >= self.negative_correlation_threshold

    @abstractmethod
    def as_table(self) -> CorrelationTable:
        """ Returns the item statistics as a CorrelationTable """

    @abstractmethod
    def update(self, **kwargs) -> None:
        """ Updates the item statistics """


@dataclass
class ECItemStats(ItemStats):
    """ Extended context item-level statistics

        "each extended-context slot records the ratio of the probability that the schema will succeed (i.e., that
        its result will obtain) if the schema is activated when the slot's item is On, to the probability
        of success if that item is Off when the schema is activated." (Drescher, 1991, p. 73)
    """
    n_success_and_on: int = 0
    n_success_and_off: int = 0
    n_fail_and_on: int = 0
    n_fail_and_off: int = 0

    def __str__(self) -> str:
        attr_values = (
            f'sc: {self.positive_correlation_stat:.2}',
            f'fc: {self.negative_correlation_stat:.2}',
        )

        return f'{self.__class__.__name__}[{"; ".join(attr_values)}]'

    def __repr__(self):
        attr_values = {
            'success_corr': f'{self.positive_correlation_stat:.2}',
            'failure_corr': f'{self.negative_correlation_stat:.2}',
            'n_on': f'{self.n_on:,}',
            'n_off': f'{self.n_off:,}',
            'n_success_and_on': f'{self.n_success_and_on:,}',
            'n_success_and_off': f'{self.n_success_and_off:,}',
            'n_fail_and_on': f'{self.n_fail_and_on:,}',
            'n_fail_and_off': f'{self.n_fail_and_off:,}',
        }

        return repr_str(self, attr_values)

    def __hash__(self):
        return hash((self.n_success_and_on,
                     self.n_success_and_off,
                     self.n_fail_and_on,
                     self.n_fail_and_off))

    @property
    def correlation_test(self) -> ItemCorrelationTest:
        params = get_global_params()

        test_name = params.get('ext_context.correlation_test') or 'DrescherCorrelationTest'
        return correlation_test.name_to_type_map[test_name]

    @property
    def positive_correlation_threshold(self) -> float:
        params = get_global_params()
        return params.get('ext_context.positive_correlation_threshold')

    @property
    def negative_correlation_threshold(self) -> float:
        params = get_global_params()
        return params.get('ext_context.negative_correlation_threshold')

    def update(self, on: bool, success: bool, count: int = 1) -> None:
        if on and success:
            self.n_success_and_on += count

        elif on and not success:
            self.n_fail_and_on += count

        elif not on and success:
            self.n_success_and_off += count

        elif not on and not success:
            self.n_fail_and_off += count

    @property
    def n_on(self) -> int:
        return self.n_success_and_on + self.n_fail_and_on

    @property
    def n_off(self) -> int:
        return self.n_success_and_off + self.n_fail_and_off

    @property
    def n_success(self) -> int:
        return self.n_success_and_on + self.n_success_and_off

    @property
    def n_fail(self) -> int:
        return self.n_fail_and_on + self.n_fail_and_off

    @property
    def n(self) -> int:
        return self.n_on + self.n_off

    @property
    def specificity(self) -> float:
        """ Quantifies the item's specificity. Greater values imply greater specificity.

        "An item is considered more specific if it is On less frequently." (See Drescher, 1991, p.77)

        :return: a float between 0.0 and 1.0
        """
        try:
            return self.n_off / self.n
        except ZeroDivisionError:
            return np.NAN

    def as_table(self) -> CorrelationTable:
        return self.n_success_and_on, self.n_fail_and_on, self.n_success_and_off, self.n_fail_and_off

    def reset(self) -> None:
        self.n_success_and_on = 0
        self.n_fail_and_on = 0
        self.n_success_and_off = 0
        self.n_fail_and_off = 0


@dataclass
class ERItemStats(ItemStats):
    """ Extended result item-level statistics """

    n_on_and_activated: int = 0
    n_on_and_not_activated: int = 0
    n_off_and_activated: int = 0
    n_off_and_not_activated: int = 0

    @property
    def correlation_test(self) -> ItemCorrelationTest:
        params = get_global_params()
        test_name = params.get('ext_result.correlation_test') or 'DrescherCorrelationTest'
        return correlation_test.name_to_type_map[test_name]

    @property
    def positive_correlation_threshold(self) -> float:
        params = get_global_params()
        return params.get('ext_result.positive_correlation_threshold')

    @property
    def negative_correlation_threshold(self) -> float:
        params = get_global_params()
        return params.get('ext_result.negative_correlation_threshold')

    @property
    def n_on(self) -> int:
        return self.n_on_and_activated + self.n_on_and_not_activated

    @property
    def n(self) -> int:
        return self.n_on + self.n_off

    @property
    def n_off(self) -> int:
        return self.n_off_and_activated + self.n_off_and_not_activated

    @property
    def n_activated(self) -> int:
        return self.n_on_and_activated + self.n_off_and_activated

    @property
    def n_not_activated(self) -> int:
        return self.n_on_and_not_activated + self.n_off_and_not_activated

    def update(self, on: bool, activated: bool, count: int = 1) -> None:
        if on and activated:
            self.n_on_and_activated += count

        elif on and not activated:
            self.n_on_and_not_activated += count

        elif not on and activated:
            self.n_off_and_activated += count

        elif not on and not activated:
            self.n_off_and_not_activated += count

    def reset(self) -> None:
        self.n_on_and_activated = 0
        self.n_on_and_not_activated = 0
        self.n_off_and_activated = 0
        self.n_off_and_not_activated = 0

    def as_table(self) -> CorrelationTable:
        return (self.n_on_and_activated,
                self.n_off_and_activated,
                self.n_on_and_not_activated,
                self.n_off_and_not_activated)

    def __hash__(self):
        return hash((self.n_on_and_activated,
                     self.n_off_and_activated,
                     self.n_on_and_not_activated,
                     self.n_off_and_not_activated))

    def __str__(self) -> str:
        attr_values = (
            f'ptc: {self.positive_correlation_stat:.2}',
            f'ntc: {self.negative_correlation_stat:.2}',
        )

        return f'{self.__class__.__name__}[{"; ".join(attr_values)}]'

    def __repr__(self) -> str:
        attr_values = {
            'positive_transition_corr': f'{self.positive_correlation_stat:.2}',
            'negative_transition_corr': f'{self.negative_correlation_stat:.2}',
            'n_on': f'{self.n_on:,}',
            'n_off': f'{self.n_off:,}',
            'n_on_and_activated': f'{self.n_on_and_activated:,}',
            'n_on_and_not_activated': f'{self.n_on_and_not_activated:,}',
            'n_off_and_activated': f'{self.n_off_and_activated:,}',
            'n_off_and_not_activated': f'{self.n_off_and_not_activated:,}',
        }

        return repr_str(self, attr_values)


class ReadOnlySchemaStats(SchemaStats):
    def update(self, **kwargs):
        raise NotImplementedError('Update not implemented for readonly view.')


class ReadOnlyECItemStats(ECItemStats):
    def __init__(self):
        super().__init__()

    def update(self, on: bool, success: bool, count: int = 1) -> None:
        raise NotImplementedError('Update not implemented for readonly view.')


class ReadOnlyERItemStats(ERItemStats):
    def __init__(self):
        super().__init__()

    def update(self, on: bool, activated: bool, count: int = 1) -> None:
        raise NotImplementedError('Update not implemented for readonly view.')


class FrozenECItemStats(ECItemStats):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return 'FROZEN'

    def update(self, on: bool, success: bool, count: int = 1) -> None:
        pass


class FrozenERItemStats(ERItemStats):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return 'FROZEN'

    def update(self, on: bool, activated: bool, count: int = 1) -> None:
        pass


# A single immutable object that is meant to be used for all item instances that have never had stats updates
NULL_SCHEMA_STATS = ReadOnlySchemaStats()

NULL_EC_ITEM_STATS = ReadOnlyECItemStats()
NULL_ER_ITEM_STATS = ReadOnlyERItemStats()

FROZEN_EC_ITEM_STATS = FrozenECItemStats()
FROZEN_ER_ITEM_STATS = FrozenERItemStats()


class CompositeItem(Item):
    """ An Item whose source is a collection of state elements. """

    def __init__(self,
                 source: Collection[StateElement],
                 primitive_value: Optional[float] = None,
                 delegated_value: Optional[float] = None,
                 **kwargs) -> None:

        self._validate_source(source)

        self._items = frozenset([ItemPool().get(se) for se in source])

        # by default, primitive value is the sum over non-composite item primitive values, but this can be overridden
        primitive_value = (
            sum(item.primitive_value for item in self._items)
            if primitive_value is None
            else primitive_value
        )

        # by default, delegated value is initialized to the sum over non-composite item delegated values, but this
        # can be overridden. (Note that the delegated value may change over time as the agent learns from experience.)
        delegated_value = (
            sum(item.delegated_value for item in self._items)
            if delegated_value is None
            else delegated_value
        )

        super().__init__(
            source=frozenset(source),
            primitive_value=primitive_value,
            delegated_value=delegated_value,
        )

    def __contains__(self, element: StateElement) -> bool:
        if isinstance(element, StateElement):
            return element in self.source
        return False

    def __eq__(self, other) -> bool:
        if isinstance(other, CompositeItem):
            return self.source == other.source
        return False if other is None else NotImplemented

    def __hash__(self) -> int:
        return hash(self.source)

    def __repr__(self) -> str:
        return repr_str(self, {'source': ','.join(sorted([str(element) for element in self.source])),
                               'pv': self.primitive_value,
                               'dv': self.delegated_value})

    @property
    def state_elements(self) -> set[StateElement]:
        return self.source

    def is_on(self, state: State, **kwargs) -> bool:
        return all((item.is_on(state) for item in self._items))

    def _validate_source(self, source: Any) -> None:
        try:
            iter(source)
        except TypeError:
            raise TypeError('Source must be iterable') from None

        if len(source) < 2:
            raise ValueError('Source must contain at least two state elements.')


class StateAssertion(UniqueIdMixin):

    def __init__(self, items: Optional[Collection[Item, ...]] = None, **kwargs):
        self._items = frozenset(items) if items else frozenset()

        super().__init__()

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[Item]:
        yield from self._items

    def __contains__(self, item: Item) -> bool:
        return item in self._items

    def __str__(self) -> str:
        return ','.join(sorted(map(str, self._items)))

    def __repr__(self) -> str:
        return repr_str(self, {'items': str(self)})

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, StateAssertion):
            return self._items == other._items
        return False if other is None else NotImplemented

    def __hash__(self) -> int:
        return hash(self._items)

    # TODO: can this be replace by using the fact that this class is iterable???
    @property
    def items(self) -> frozenset[Item]:
        return self._items

    def union(self, items: Collection[Item, ...]) -> StateAssertion:
        return StateAssertion(items=[*self._items, *items])

    def issubset(self, other: StateAssertion) -> bool:
        return self._items.issubset(other._items)

    def difference(self, other: StateAssertion) -> Iterable[Item]:
        return list(self._items.difference(other._items))

    def is_satisfied(self, state: State, **kwargs) -> bool:
        """ Satisfied when all non-negated items are On, and all negated items are Off.

        :param state: the agent's current state
        :param kwargs: optional keyword arguments
        :return: True if this state assertion is satisfied given the current state; False otherwise.
        """
        return all((item.is_on(state, **kwargs) for item in self._items))

    def as_state(self) -> State:
        """ Returns a State consistent with this StateAssertion.

        :return: a State
        """
        return frozenset(itertools.chain.from_iterable([item.state_elements for item in self._items]))

    @staticmethod
    def from_state(state: State) -> StateAssertion:
        """ Factory method for creating state assertions that would be satisfied by the given State.

        :param state: a State

        :return: a StateAssertion
        """
        return StateAssertion([ReadOnlyItemPool().get(se) for se in state])

    def flatten(self) -> StateAssertion:
        """ Returns a version of this StateAssertion that is composed of non-composite items. """
        return StateAssertion.from_state(self.as_state())


NULL_STATE_ASSERT = StateAssertion()


class ExtendedItemCollection(Observable):
    def __init__(
            self,
            suppressed_items: Collection[Item] = None,
            relevant_items: Iterable[Item] = None,
            stats: dict[Item, ItemStats] = None,
            null_member: ItemStats = None,
    ) -> None:
        super().__init__()

        self._suppressed_items: frozenset[Item] = frozenset(suppressed_items) if suppressed_items else frozenset()
        self._relevant_items: set[Item] = set(relevant_items) if relevant_items else set()

        self._stats: dict[Any, ItemStats] = defaultdict(self._get_null_stats)
        if stats:
            self._stats.update(stats)

        self._null_member = null_member

        self._new_relevant_items: set[Item] = set()

    def __str__(self) -> str:
        name = self.__class__.__name__
        values = '; '.join([f'{k} -> {v}' for k, v in self._stats.items()])

        return f'{name}[{values}]'

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

    def __eq__(self, other) -> bool:
        if isinstance(other, ExtendedItemCollection):
            return all(
                (
                    # generator expression for conditions to allow lazy evaluation
                    condition for condition in
                    [
                        self._suppressed_items == other._suppressed_items,
                        self._relevant_items == other._relevant_items,
                        self._new_relevant_items == other._new_relevant_items,
                        self._stats == other._stats
                    ]
                )
            )
        return False if other is None else NotImplemented

    def __hash__(self) -> int:
        hash_str = str(self._suppressed_items) + str(self._stats)
        return hash(hash_str)

    @property
    def stats(self) -> dict[Any, Any]:
        return self._stats

    @property
    def suppressed_items(self) -> frozenset[Item]:
        return self._suppressed_items

    @property
    def relevant_items(self) -> frozenset[Item]:
        return frozenset(self._relevant_items)

    def update_relevant_items(self, item: Item, suppressed: bool = False):
        if item not in self._relevant_items:
            self._relevant_items.add(item)

            # if suppressed then no spin-offs will be created for this item
            if suppressed:
                logger.debug(f'suppressing spin-off for item {item}')
            else:
                self._new_relevant_items.add(item)

    def notify_all(self, spin_off_type: SchemaSpinOffType, **kwargs) -> None:
        super().notify_all(source=self, spin_off_type=spin_off_type, **kwargs)

        # clears the set
        self._new_relevant_items = set()

    @property
    def new_relevant_items(self) -> frozenset[Item]:
        return frozenset(self._new_relevant_items)

    @new_relevant_items.setter
    def new_relevant_items(self, value: Collection[Item]) -> None:
        self._new_relevant_items = frozenset(value)

    def _get_null_stats(self) -> ItemStats:
        return self._null_member


class ExtendedResult(ExtendedItemCollection):
    """
        a schema’s extended result is used to identify the consequences of an agent's actions. Each extended result
        slot keeps track of whether an item (or conjunction of items) is significantly more likely to occur when a
        schema is activated (positive correlation) or when it is NOT activated (negative correlation). When such
        significant items are found, result spin-offs are created.

        It's important to note that, unlike extended contexts, the schema mechanism is NOT designed to build composite
        results incrementally due to the combinatorial proliferation of results that would occur (see Drescher 1991,
        p. 78). Only a PRIMITIVE SCHEMA (i.e., a schema with an empty result) can spin-off a schema with a new
        result item. Furthermore, multi-item result spin-offs can be create ALL-AT-ONCE (as opposed to incrementally).

        The mechanism for doing this is the following (see Drescher 1991, Section 4.1.4):

        (1) In addition to containing a slot for each item, a schema's extended result also includes a slot for each
            SET of items (i.e., composite context) that appears as the context of a RELIABLE schema.
        (2) Extended results treat these SETS OF ITEMS like individual items, maintaining the same statistics.
        (3) When an item or set of items is found to be relevant, a spin-off schema is created with that item or
            conjunction of items.

        Supports the discovery of:

            reliable schemas
            chains of schemas
    """

    def __init__(
            self,
            suppressed_items: Collection[Item] = None,
            relevant_items: Iterable[Item] = None,
            stats: dict[Item, ItemStats] = None,
    ) -> None:
        super().__init__(
            suppressed_items=suppressed_items,
            relevant_items=relevant_items,
            stats=stats,
            null_member=NULL_ER_ITEM_STATS
        )

    @property
    def stats(self) -> dict[Item, ERItemStats]:
        return super().stats

    def update(self, item: Item, on: bool, activated=False, count: int = 1) -> None:
        item_stats = self._stats[item]
        if item_stats is FROZEN_ER_ITEM_STATS:
            return

        if item_stats is NULL_ER_ITEM_STATS:
            self._stats[item] = item_stats = ERItemStats()

        item_stats.update(on=on, activated=activated, count=count)

        positive_correlation_exists = item_stats.positive_correlation
        negative_correlation_exists = item_stats.negative_correlation
        correlation_exists = positive_correlation_exists or negative_correlation_exists
        no_progress = self._check_for_no_progress(item_stats)

        item_is_relevant = self._check_for_relevance(
            item=item,
            positive_correlation_exists=positive_correlation_exists
        )

        if item_is_relevant:
            self.update_relevant_items(item)

        should_freeze_item_stats = correlation_exists or no_progress
        if (is_feature_enabled(SupportedFeature.FREEZE_ITEM_STATS_UPDATES_ON_CORRELATION)
                and should_freeze_item_stats):
            self._stats[item] = FROZEN_ER_ITEM_STATS

    def update_all(self, activated: bool, new: Collection[Item], lost: Collection[Item], count: int = 1) -> None:

        # "a trial for which the result was already satisfied before the action was taken does not count as a
        # positive-transition trial; and one for which the result was already unsatisfied does not count
        # as a negative-transition trial" (see Drescher, 1991, p. 72)
        if new:
            for item in new:
                if item in lost:
                    raise ValueError(f'Item {item} is in both new and lost!')

                self.update(item=item, on=True, activated=activated, count=count)

        if lost:
            for item in lost:
                self.update(item=item, on=False, activated=activated, count=count)

        if self.new_relevant_items:
            self.notify_all(spin_off_type=SchemaSpinOffType.RESULT)

    def _check_for_relevance(self, item: Item, positive_correlation_exists: bool) -> bool:
        if item in self.suppressed_items or item in self.relevant_items:
            return False

        if positive_correlation_exists:
            return True

    def _check_for_no_progress(self, item_stats: ERItemStats) -> bool:
        return False


class ExtendedContext(ExtendedItemCollection):
    """
        a schema’s extended context is used to identify conditions under which the result more
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

    def __init__(
            self,
            suppressed_items: Collection[Item] = None,
            relevant_items: Iterable[Item] = None,
            stats: dict[Item, ItemStats] = None,
    ) -> None:
        super().__init__(
            suppressed_items=suppressed_items,
            relevant_items=relevant_items,
            stats=stats,
            null_member=NULL_EC_ITEM_STATS
        )

        self._pending_relevant_items = set()
        self._pending_max_specificity = -np.inf

    @property
    def stats(self) -> dict[Item, ECItemStats]:
        """ Returns a dictionary mapping items to statistics that are specific to this extended context.

        Note: Care should be used when saving references to the values of this dictionary, as the references can change.

        :return: a extended-context statistics dictionary
        """
        return super().stats

    def update(self, item: Item, on: bool, success: bool, count: int = 1) -> None:
        item_stats = self._stats[item]
        if item_stats is FROZEN_EC_ITEM_STATS:
            return

        if item_stats is NULL_EC_ITEM_STATS:
            self._stats[item] = item_stats = ECItemStats()

        item_stats.update(on=on, success=success, count=count)

        positive_correlation_exists = item_stats.positive_correlation
        negative_correlation_exists = item_stats.negative_correlation
        correlation_exists = positive_correlation_exists or negative_correlation_exists
        no_progress = self._check_for_no_progress(item_stats)

        specificity = item_stats.specificity

        item_is_relevant = self._check_for_relevance(
            item=item,
            positive_correlation_exists=positive_correlation_exists,
            specificity=specificity
        )

        if item_is_relevant:
            self._pending_relevant_items.add(item)

        should_freeze_item_stats = correlation_exists or no_progress
        if (is_feature_enabled(SupportedFeature.FREEZE_ITEM_STATS_UPDATES_ON_CORRELATION)
                and should_freeze_item_stats):
            self._stats[item] = FROZEN_EC_ITEM_STATS

    def update_all(self, selection_state: State, success: bool, count: int = 1) -> None:
        # bypass updates when a more specific spinoff schema exists
        if (is_feature_enabled(SupportedFeature.EC_DEFER_TO_MORE_SPECIFIC_SCHEMA)
                and self.defer_update_to_spin_offs(selection_state)):
            return

        for item in ItemPool().items:
            self.update(item=item, on=item.is_on(selection_state), success=success, count=count)

        self.check_pending_relevant_items()

        if self.new_relevant_items:
            self.notify_all(spin_off_type=SchemaSpinOffType.CONTEXT)

            # after spinoff, all stats are reset to remove the influence of relevant items in stats; tracking the
            # correlations for these items is deferred to the schema's context spin-offs
            if is_feature_enabled(SupportedFeature.EC_DEFER_TO_MORE_SPECIFIC_SCHEMA):
                self.stats.clear()

    def defer_update_to_spin_offs(self, state: State) -> bool:
        """ Checks if this state activates previously recognized relevant items. If so, defer to spin-offs.

        Note: This method supports the optional enhancement SupportedFeature.EC_DEFER_TO_MORE_SPECIFIC_SCHEMA

        :param state: a State

        :return: True if should defer stats updates for this state to this schema's spin-offs.
        """
        return any((item.is_on(state) for item in self.relevant_items))

    @property
    def pending_relevant_items(self) -> frozenset[Item]:
        return frozenset(self._pending_relevant_items)

    @property
    def pending_max_specificity(self) -> float:
        return self._pending_max_specificity

    # TODO: Rename this method. It is not a check, but more of a flush of the pending items to the super class.
    def check_pending_relevant_items(self) -> None:
        for item in self._pending_relevant_items:
            self.update_relevant_items(item=item)

        self.clear_pending_relevant_items()

    def clear_pending_relevant_items(self) -> None:
        self._pending_relevant_items.clear()
        self._pending_max_specificity = -np.inf

    def _check_for_relevance(self, item: Item, positive_correlation_exists: bool, specificity: float) -> bool:
        if (item in self.suppressed_items
                or item in self.relevant_items
                or not positive_correlation_exists):
            return False

        # if enabled, this enhancement allows only a single relevant item per update (the most "specific").
        if is_feature_enabled(SupportedFeature.EC_MOST_SPECIFIC_ON_MULTIPLE):
            # item is less specific than earlier item; suppress it on this update.
            if specificity < self._pending_max_specificity:
                return False

            # item is the most specific so far; replace previous pending item assertion.
            else:
                self._pending_relevant_items.clear()
                self._pending_max_specificity = max(self._pending_max_specificity, specificity)

        return True

    def _check_for_no_progress(self, item_stats: ECItemStats) -> bool:
        return False


class Controller:
    def __init__(
            self,
            goal_state: StateAssertion,
            proximity_map: dict[Schema, float] = None,
            total_cost_map: dict[Schema, float] = None,
            components: set[Schema] = None,
            descendants: set[Schema] = None,
    ) -> None:
        self._goal_state: StateAssertion = goal_state

        self._proximity_map: dict[Schema, float] = defaultdict(float)
        if proximity_map:
            self._proximity_map.update(proximity_map)

        self._total_cost_map: dict[Schema, float] = defaultdict(float)
        if total_cost_map:
            self._total_cost_map.update(total_cost_map)

        self._components: set[Schema] = set() if components is None else components
        self._descendants: set[Schema] = set() if descendants is None else descendants

    def __eq__(self, other):
        if self is other:
            return True
        if isinstance(other, Controller):
            return self.goal_state == other.goal_state
        return False if other is None else NotImplemented

    def __hash__(self):
        return hash(self._goal_state)

    @property
    def goal_state(self) -> StateAssertion:
        return self._goal_state

    @property
    def components(self) -> set[Schema]:
        """ The set of schemas that are selectable in pursuit of the controller's goal state.

        Note: Components are currently limited to RELIABLE schemas that chain to the controller's goal state.
        Drescher may have also allowed UNRELIABLE schemas as components. Components are also limited to be schemas
        with primitive actions (i.e., composite action schemas are not currently supported).

        :return: the set of component schemas
        """
        return self._components

    @property
    def descendants(self) -> set[Schema]:
        """ The set of schemas that are immediate components or their descendant components.

        :return: the set of descendant schemas
        """
        return self._descendants

    @property
    def proximity_map(self) -> dict[Schema, float]:
        return self._proximity_map

    def proximity(self, schema: Schema) -> float:
        """ Returns the proximity (i.e., closeness) of a schema to the composite action's goal state.

        "Proximity is inversely proportionate to the expected time to reach the goal state, derived from the
         expected activation time of the schemas in the relevant chain; proximity is also proportionate to those
         schemas' reliability, and inversely proportionate to their cost of activation." (See Drescher 1991, p. 60)

        :param schema: a component schema
        :return: a float that quantifies the schema's goal state proximity
        """
        return self._proximity_map[schema]

    @property
    def total_cost_map(self) -> dict[Schema, float]:
        return self._total_cost_map

    def total_cost(self, schema: Schema) -> float:
        """ Returns the total cost that would be incurred in order to achieve the composite action's goal state.

        :param schema: a component schema
        :return: a float that quantifies the total cost
        """
        return self._total_cost_map[schema]

    def update(self, chains: list[Chain[Schema]]) -> None:
        if not chains:
            return

        # FIXME: implement incremental learning or remove this variable
        # lr: float = GlobalParams().get('learning_rate')

        # FIXME: this should probably be removed
        self._components.clear()
        self._descendants.clear()

        for chain in chains:
            if not chain:
                continue

            # sanity check: chains must lead to the controller's goal state
            final_state = chain[-1].result.as_state()
            if not self.goal_state.is_satisfied(final_state):
                raise ValueError('Invalid chain: chain should result in state that satisfies the goal state')

            avg_duration_to_goal_state = 0.0
            total_cost_to_goal_state = 0.0

            for schema in reversed(chain):

                # FIXME: Current implementation is limited to controller component schemas with primitive actions only.
                # FIXME: Allowing components with composite actions results in difficult to correct RecursionErrors.

                # early terminate chain processing if composite action schema encountered
                if schema.action.is_composite():
                    break

                # FIXME: Uncomment this if/when components with composite actions are supported
                # prevents recursion (a controller should not have itself as a component)
                # if self.contained_in(schema):
                #     break

                self._components.add(schema)
                self._descendants.add(schema)

                # FIXME: Uncomment this if/when components with composite actions are supported
                # if schema.action.is_composite():
                #     self._descendants.update(schema.action.controller.descendants)

                avg_duration_to_goal_state += schema.avg_duration
                total_cost_to_goal_state += schema.cost

                # FIXME: incremental learning seems to be producing undesirable results... can I make this work?
                # self._proximity[schema] += lr * (1.0 / avg_duration_to_goal_state - self._proximity[schema])
                self._proximity_map[schema] = 1.0 / avg_duration_to_goal_state
                self._total_cost_map[schema] = total_cost_to_goal_state

        # TODO: Implement this...
        # "each time a composite action is explicitly initiated, the controller keeps track of which component
        #  schemas are actually activated and when.... If the action successfully culminates in the goal state,
        #  the actual cost and duration of execution from each entry point are compared with the proximity
        #  information stored in the slot of each component actually activated; in case of discrepancy, the stored
        #  information is adjusted in the direction of the actual data. If the action fails to reach its goal
        #  state, the proximity measures for the utilized components are degraded." (See Drescher 1991, p. 92)

    # uncomment this when composite action components are supported
    # def contained_in(self, schema: Schema) -> bool:
    #     if not schema.action.is_composite():
    #         return False
    #
    #     if self == schema.action.controller:
    #         return True
    #
    #     controller = schema.action.controller
    #     for component in itertools.chain.from_iterable([controller.components, controller.descendants]):
    #         if component.action.is_composite() and self == component.action.controller:
    #             return True
    #     return False


class DummyController(Controller):
    def __init__(self):
        super().__init__(goal_state=NULL_STATE_ASSERT)

        self._components = set()
        self._descendants = set()

    @property
    def components(self) -> set[Schema]:
        return self._components

    @property
    def descendants(self) -> set[Schema]:
        return self._descendants

    def update(self, *args, **kwargs) -> None:
        raise NotImplementedError()

    def proximity(self, schema: Schema) -> float:
        return -np.inf


NULL_CONTROLLER = DummyController()


class Action(UniqueIdMixin):

    def __init__(self, label: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)

        self._label = label

    def __eq__(self, other):
        if isinstance(other, Action):
            if self._label and other._label:
                return self._label == other._label
            else:
                return self.uid == other.uid

        return False if other is None else NotImplemented

    def __hash__(self) -> int:
        return hash(self._label or self.uid)

    def __str__(self) -> str:
        return self._label or str(self.uid)

    def __repr__(self) -> str:
        return repr_str(self, {'uid': self.uid,
                               'label': self.label})

    @property
    def label(self) -> Optional[str]:
        """ A description of this action.

        :return: returns the Action's label
        """
        return self._label

    @property
    def goal_state(self) -> StateAssertion:
        return NULL_CONTROLLER.goal_state

    @property
    def controller(self) -> Controller:
        return NULL_CONTROLLER

    def is_composite(self):
        return False

    def is_enabled(self, **kwargs) -> bool:
        """ Returns whether this action is enabled.

        Note: Schemas are inhibited for activation if their actions are disabled. This is primarily useful for
              composite actions.

        :param kwargs: optional keyword arguments

        :return: True if enabled; False otherwise.
        """
        return True


def default_for_controller_map(key: StateAssertion) -> Controller:
    return Controller(key)


class CompositeAction(Action):
    """ "A composite action is essentially a subroutine: it is defined to be the action of achieving the designated
     goal state, by whatever means is available. The means are given by chains of schemas that lead to the goal
     state..." (See Drescher 1991, p. 59)
    """

    def __init__(self, goal_state: StateAssertion, **kwargs):
        super().__init__(**kwargs)

        if not goal_state:
            raise ValueError('Goal state is not optional.')

        self._controller = get_controller_map()[goal_state]

    def __eq__(self, other) -> bool:
        if isinstance(other, CompositeAction):
            return self.goal_state == other.goal_state
        return False if other is None else NotImplemented

    def __hash__(self) -> int:
        return hash(self.goal_state)

    def __str__(self) -> str:
        return str(self.controller.goal_state)

    @property
    def goal_state(self) -> StateAssertion:
        return self._controller.goal_state

    @property
    def controller(self) -> Controller:
        return self._controller

    def is_composite(self):
        return True

    def is_enabled(self, state: State, **kwargs) -> bool:
        """

        "A composite action is enabled when one of its components is applicable." (See Drescher 1991, p. 90)

        :param state:
        :return:
        """
        return any({schema.is_applicable(state) for schema in self._controller.components})

    @classmethod
    def all_satisfied_by(cls, state: State) -> set[Controller]:
        """ Returns a collection containing all of the controllers satisfied by this state.

        :param state: a State

        :return: a collection of controllers satisfying this state
        """
        controller_map: ControllerMap = get_controller_map()

        satisfied: list[Controller] = []
        for goal_state, controller in controller_map.items():
            if goal_state.is_satisfied(state):
                satisfied.append(controller)
        return set(satisfied)


class SchemaSpinOffType(Enum):
    CONTEXT = 'CONTEXT'  # (see Drescher, 1991, p. 73)
    RESULT = 'RESULT'  # (see Drescher, 1991, p. 71)


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

    def __init__(self,
                 action: Action,
                 context: Optional[StateAssertion] = None,
                 result: Optional[StateAssertion] = None,
                 extended_context: Optional[ExtendedContext] = None,
                 extended_result: Optional[ExtendedResult] = None,
                 schema_stats: Optional[SchemaStats] = None,
                 overriding_conditions: Optional[StateAssertion] = None,
                 avg_duration: Optional[float] = None,
                 cost: Optional[float] = None,
                 creation_time: Optional[float] = None,
                 **kwargs):
        super().__init__()

        self._context: Optional[StateAssertion] = context or NULL_STATE_ASSERT
        self._action: Action = action
        self._result: Optional[StateAssertion] = result or NULL_STATE_ASSERT

        if self.action is None:
            raise ValueError('Action cannot be None')

        self._is_bare = not (context or result)

        # bare schemas do not have an extended context
        self._extended_context: ExtendedContext = (
            None
            if self._is_bare else
            extended_context or ExtendedContext(suppressed_items=self._context.items)
        )

        # only bare schemas have an extended result
        self._extended_result: ExtendedResult = (
            extended_result or ExtendedResult(suppressed_items=self._result.items)
            if self._is_bare else
            None
        )

        # This observer registration is used to notify the schema when a relevant item has been detected in its
        # extended context or extended result
        if self._extended_context:
            self._extended_context.register(self)

        if self._extended_result:
            self._extended_result.register(self)

        self._stats: SchemaStats = schema_stats or SchemaStats()

        # TODO: Need to update overriding conditions.
        self._overriding_conditions: Optional[StateAssertion] = overriding_conditions
        self._avg_duration: Optional[float] = 1.0 if avg_duration is None else avg_duration
        self._cost: Optional[float] = 1.0 if cost is None else cost
        self._creation_time = time() if creation_time is None else creation_time

    def __eq__(self, other) -> bool:
        # SchemaPool should be used for all item creation, so this should be an optimization
        if self is other:
            return True

        if isinstance(other, Schema):
            return all(
                (
                    # generator expression for conditions to allow lazy evaluation
                    condition for condition in
                    [
                        self._context == other._context,
                        self._action == other._action,
                        self._result == other._result,
                    ]
                )
            )
        return False if other is None else NotImplemented

    def __hash__(self) -> int:
        return hash((self._context,
                     self._action,
                     self._result,))

    def __str__(self) -> str:
        return (
                f'{self.context}/{self.action}/{self.result} ' +
                f'[rel: {self.reliability:.2}]'
        )

    def __repr__(self) -> str:
        attr_values = {
            'action': self.action,
            'context': self.context,
            'cost': self.cost,
            'creation_time': datetime.fromtimestamp(self.creation_time),
            'overriding_conditions': self.overriding_conditions,
            'reliability': self.reliability,
            'result': self.result,
            'uid': self.uid,
        }

        return repr_str(self, attr_values=attr_values)

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
        return (
            np.NAN if self.is_bare() or self.stats.n_activated == 0
            else self.stats.n_success / self.stats.n_activated
        )

    @property
    def stats(self) -> SchemaStats:
        return self._stats

    # TODO: All durations are currently fixed at 1.0. Need a way to calculate duration during action execution to
    # TODO: enable this update.
    @property
    def avg_duration(self) -> float:
        """ The average time from schema activation to the completion of the schema's action.

        :return:
        """
        return np.nan if self._avg_duration is None else self._avg_duration

    @avg_duration.setter
    def avg_duration(self, value: float) -> None:
        self._avg_duration = value

    # TODO: Cost still needs to be implemented; however, the specifics of its update are still unclear to me. Based
    # TODO: on Drescher's description (see Drescher 1991, p. 54) it seems like a WORST CASE activation result. This
    # TODO: seems strange unless we include something about schema reliability. Cost may be meant to counter-balance
    # TODO: delegated value, which seems like a BEST CASE activation result, but I'm not convinced there is a value to
    # TODO: separating these factors.
    @property
    def cost(self) -> float:
        """ The schema's activation cost.

        "[a schema's] cost is the minimum (i.e., the greatest magnitude) of any negative-valued results of schemas that
         are implicitly activated as a side-effect of the given schema's activation on that occasion."
             (See Drescher 1991, p. 54)

        :return: a float quantifying the schema's cost
        """
        return self._cost

    @cost.setter
    def cost(self, value: float) -> None:
        self._cost = value

    @property
    def creation_time(self) -> float:
        """ Returns this schema's creation timestamp (i.e., time in seconds since the epoch).

        :return: a float corresponding to this schema's creation time
        """
        return self._creation_time

    def is_applicable(self, state: State, **kwargs) -> bool:
        """ Returns whether this schema is "applicable" in the given State.

        A schema is applicable when:

        (1) its context is satisfied
        (2) its action is enabled
        (3) there are no overriding conditions

        (See Drescher 1991, pp. 53 and 90)

        :param state: the agent's current state
        :param kwargs: optional keyword arguments

        :return: True if the schema is applicable in this State; False, otherwise.
        """
        inhibited = not self.action.is_enabled(state=state)
        if self.overriding_conditions is not None:
            inhibited |= self.overriding_conditions.is_satisfied(state, **kwargs)

        return (not inhibited) and self.context.is_satisfied(state, **kwargs)

    def is_activated(self, schema: Schema, state: State, applicable: bool = True) -> bool:
        """ Returns whether this schema was (implicitly or explicitly) activated as a result of the last selection.

        "As a side-effect of an explicit activation, other schemas whose contexts are satisfied, but which are not
         themselves selected for activation, may have their actions initiated (if they happen to share the same
         action as the schema that was explicitly activated.) Such schemas are said to be implicitly activated."
         (See Drescher 1991, p. 54)

        “A composite action is considered to have been implicitly taken whenever its goal state becomes
         satisfied--that is, makes a transition from Off to On--even if that composite action was never initiated by
         an activated schema. Marginal attribution can thereby detect results caused by the goal state, even if the
         goal state obtains due to external events.” (See Drescher, 1991, p. 91)

        :param schema: the selected schema (explicitly activated)
        :param state: the state RESULTING from the last selected action
        :param applicable: whether this schema was applicable during the last selection event

        :return: True if this schema is activated; False otherwise.
        """
        if not applicable:
            return False

        # explicit activation
        if self == schema:
            return True

        # implicit activation
        if self.action.is_composite():
            goal_state: StateAssertion = self.action.goal_state
            return goal_state.is_satisfied(state)
        else:
            return self.action == schema.action

    def is_bare(self) -> bool:
        """ Returns whether this instance is a bare (action-only) schema.

        :return: True if this is a bare schema; False otherwise.
        """
        return self._is_bare

    def predicts_state(self, state: State) -> bool:
        if not self.result:
            return False

        return StateAssertion.from_state(state) == self.result.flatten()

    def update(self,
               activated: bool,
               succeeded: bool,
               selection_state: Optional[State],
               new: Collection[Item] = None,
               lost: Collection[Item] = None,
               explained: Optional[bool] = None,
               duration: float = 1.0,
               count=1) -> None:
        """

            Note: As a result of these updates, one or more notifications may be generated by the schema's
            context and/or result. These will be received via schema's 'receive' method.

        :param activated: True if this schema was implicitly or explicitly activated
        :param succeeded: True if the schema's result was achieved; False otherwise.
        :param selection_state: the environment state on which the most recent schema selection was based.
        :param new: the state elements in current but not previous state
        :param lost: the state elements in previous but not current state
        :param explained: True if a reliable schema was activated that "explained" the last state transition
        :param duration: the elapsed time between the schema's selection and its completion
        :param count: the number of updates to perform

        :return: None
        """

        # update top-level stats
        self._stats.update(activated=activated, success=succeeded, count=count)

        # update average duration
        self.avg_duration += (1.0 / float(self._stats.n)) * (duration - self.avg_duration)

        # update extended result stats
        if self._extended_result:
            if is_feature_enabled(SupportedFeature.ER_SUPPRESS_UPDATE_ON_EXPLAINED) and explained:
                logger.debug(
                    f'update suppressed for schema {self} because its result was explained by a reliable schema')
            else:
                self._extended_result.update_all(activated=activated, new=new, lost=lost, count=count)

        # update extended context stats
        if is_feature_enabled(SupportedFeature.EC_SUPPRESS_UPDATE_ON_RELIABLE) and is_reliable(self):
            logger.debug(f'update suppressed for schema {self} because it is reliable')
        else:
            if all((self._extended_context, activated, selection_state)):
                self._extended_context.update_all(selection_state=selection_state, success=succeeded, count=count)

    def receive(self, source: ExtendedItemCollection, spin_off_type: SchemaSpinOffType, **kwargs) -> None:
        """ Supports the Schema class's implementation of the observer pattern.

        This method is typically invoked by a Schema's extended context or extended result when relevant items are
        discovered, initiating the creation of spin-off schemas based on those relevant items

        :param source: the ExtendedItemCollection that invoked the receive method
        :param spin_off_type: the relevant spin-off type (e.g., CONTEXT or RESULT)
        :param kwargs: optional keyword arguments that will be sent to this schema's observers (via notify_all)
        """
        self.notify_all(source=self, spin_off_type=spin_off_type, relevant_items=source.new_relevant_items, **kwargs)

    @property
    def unique_key(self) -> SchemaUniqueKey:
        return SchemaUniqueKey(
            action=self.action,
            context=self.context,
            result=self.result
        )


class SchemaUniqueKey(NamedTuple):
    action: Action
    context: Optional[StateAssertion] = NULL_STATE_ASSERT
    result: Optional[StateAssertion] = NULL_STATE_ASSERT


class SchemaPool(metaclass=Singleton):
    """
    Implements a flyweight design pattern for Schema types.
    """
    _schemas: dict[SchemaUniqueKey, Schema] = dict()

    def __contains__(self, schema: Schema) -> bool:
        if not isinstance(schema, Schema):
            return False

        return schema.unique_key in SchemaPool._schemas

    def __len__(self) -> int:
        return len(SchemaPool._schemas)

    def __iter__(self) -> Iterator[Schema]:
        yield from SchemaPool._schemas.values()

    @property
    def schemas(self) -> Collection[Schema]:
        return SchemaPool._schemas.values()

    def clear(self):
        SchemaPool._schemas.clear()

    def get(self, key: SchemaUniqueKey, /, *, schema_type: Optional[Type[Schema]] = None, **kwargs) -> Optional[Schema]:
        read_only = kwargs.get('read_only', False)
        schema_type = schema_type or Schema

        obj = SchemaPool._schemas.get(key)

        # create new schema and add to pool if not found and not read_only
        if not obj and not read_only:
            obj = SchemaPool._schemas[key] = schema_type(
                context=key.context, action=key.action, result=key.result, **kwargs)

        return obj


class ReadOnlySchemaPool(SchemaPool):
    def __init__(self):
        self._pool = SchemaPool()

    def get(self, key: SchemaUniqueKey, /, *, schema_type: Optional[Type[Schema]] = None, **kwargs) -> Optional[Schema]:
        kwargs['read_only'] = True
        return self._pool.get(key, schema_type=schema_type, **kwargs)

    def clear(self):
        raise NotImplementedError('ReadOnlySchemaPool does not support clear operation.')


def is_reliable(schema: Schema, threshold: Optional[float] = None) -> bool:
    params = get_global_params()
    threshold = threshold or params.get('schema.reliability_threshold')
    return schema.reliability != np.NAN and schema.reliability >= threshold


class SchemaSearchCollection(ABC, Collection):

    @abstractmethod
    def __iter__(self) -> Iterator[Schema]:
        """ Returns an iterator over the schemas in this container. """

    @abstractmethod
    def __len__(self) -> int:
        """ Returns the number of contained schemas. """

    @abstractmethod
    def __contains__(self, element: Any) -> bool:
        """ Returns True if element is a Schema contained within this collection; False otherwise. """

    @abstractmethod
    def validate(self, **kwargs) -> None:
        """ Validates the container's schemas are legal. Raises an exception if invalid. """

    @abstractmethod
    def find(self, predicate: Callable[[Schema], bool], **kwargs) -> Collection[Schema]:
        """ Returns a collection of schemas that satisfy the provided predicate. """

    @abstractmethod
    def find_all_satisfied(self, state: State, **kwargs) -> Collection[Schema]:
        """ Returns a collection of schemas whose contexts are satisfied in the given state. """

    @abstractmethod
    def find_all_would_satisfy(self, assertion: StateAssertion, **kwargs) -> Collection[Schema]:
        """ Returns a collection of schemas whose results satisfy the given state assertion. """

    @abstractmethod
    def is_novel_result(self, result: StateAssertion) -> bool:
        """ Returns True if there are no schemas in tree with a result matching the given state assertion. """

    @abstractmethod
    def add(self, schemas: Collection[Schema], **kwargs) -> None:
        """ Adds a collection of schemas to this container. """


class SchemaTreeNode(NodeMixin, UniqueIdMixin):
    def __init__(self,
                 context: Optional[StateAssertion] = None,
                 schemas_satisfied_by: Optional[Iterable[Schema]] = None,
                 schemas_would_satisfy: Optional[Iterable[Schema]] = None,
                 label: str = None,
                 parent: SchemaTreeNode = None,
                 uid: int = None) -> None:
        """ Initializes this SchemaTreeNode.

        :param context: the state assertion that characterizes this node's location in the tree
        :param schemas_satisfied_by: the schemas whose contexts would be satisfied by this node's context's assertion
        :param schemas_would_satisfy: the schemas with results that would satisfy this node's context
        :param label: an optional label for this node (e.g., "root")
        :param parent: an optional parent node for this node
        :param uid: a unique identifier for this node (typically used when loading a learned schema mechanism)
        """
        self.context = context.flatten() if context else NULL_STATE_ASSERT

        self.schemas_satisfied_by: set[Schema] = (
            set()
            if schemas_satisfied_by is None
            else set(schemas_satisfied_by)
        )

        self.schemas_would_satisfy: set[Schema] = (
            set()
            if schemas_would_satisfy is None
            else set(schemas_would_satisfy)
        )

        self.label = label

        if parent is not None:
            self.parent = parent

        NodeMixin.__init__(self)
        UniqueIdMixin.__init__(self, uid=uid)

    def __hash__(self) -> int:
        return hash(self.context)

    def __eq__(self, other) -> bool:
        if isinstance(other, SchemaTreeNode):
            return all(
                (
                    # generator expression for conditions to allow lazy evaluation
                    condition for condition in
                    [
                        self.context == other.context,
                        self.label == other.label,
                        self.schemas_satisfied_by == other.schemas_satisfied_by,
                        self.schemas_would_satisfy == other.schemas_would_satisfy,
                        self.parent == other.parent,
                    ]
                )
            )
        return False if other is None else NotImplemented

    def __str__(self) -> str:
        return self.label if self.label else f'{self.context}'

    def __repr__(self) -> str:
        return repr_str(self, {'context': self.context,
                               'schemas_satisfied_by': self.schemas_satisfied_by,
                               'schemas_would_satisfy': self.schemas_would_satisfy,
                               'label': self.label})


class SchemaTree(SchemaSearchCollection):
    """ A search tree of SchemaTreeNodes """

    def __init__(self, root: Optional[SchemaTreeNode] = None) -> None:
        """ Initializes a SchemaTree.

        Optionally, a root node an be provided to initialize this SchemaTree. This would typically be used to initialize
        a SchemaTree from a previously learned SchemaTree instance.

        :param root: an optional root SchemaTreeNode to be used as this tree's root node
        """
        self._root = root or SchemaTreeNode(label='root')
        self._nodes_map: dict[StateAssertion, SchemaTreeNode] = self._generate_nodes_map_from_root(self._root)

        # raises a ValueError if tree is invalid
        self.validate()

    def _generate_nodes_map_from_root(self, root: SchemaTreeNode) -> dict[StateAssertion, SchemaTreeNode]:
        nodes_map: dict[StateAssertion, SchemaTreeNode] = dict()

        for node in LevelOrderIter(node=root):
            nodes_map[node.context] = node

        return nodes_map

    def __iter__(self) -> Iterator[Schema]:
        for node in LevelOrderIter(node=self.root):
            for schema in node.schemas_satisfied_by:
                yield schema

    def __len__(self) -> int:
        return len([schema for schema in self])

    def __contains__(self, schema: Schema) -> bool:
        if isinstance(schema, Schema):
            node = self._nodes_map.get(schema.context.flatten())
            return schema in node.schemas_satisfied_by if node else False

        return False

    def __str__(self) -> str:
        return RenderTree(self._root, style=AsciiStyle()).by_attr(lambda s: str(s))

    def __eq__(self, other) -> bool:
        if isinstance(other, SchemaTree):
            return all(
                (
                    # generator expression for conditions to allow lazy evaluation
                    condition for condition in
                    [
                        self._root == other._root,
                        self._nodes_map == other._nodes_map,
                    ]
                )
            )
        return False if other is None else NotImplemented

    def add(self, schemas: Collection[Schema], source: Optional[Schema] = None, **kwargs) -> None:
        if not schemas:
            raise ValueError('Spin-off schemas cannot be empty or None')

        logger.debug(f'adding schemas! [{[str(s) for s in schemas]}]')

        for schema in schemas:
            # ensure schema is in schema pool
            schema = SchemaPool().get(schema.unique_key)

            self._update_schemas_satisfied_by(schema, source)
            self._update_schemas_would_satisfy(schema)

    def _update_schemas_satisfied_by(self, schema: Schema, source: Schema) -> None:
        source_node = self.root if source is None else self.lookup_node(source.context)
        if not source_node:
            raise ValueError(f'Source schema \'{source}\' not found in schema tree!')

        # result spin-off
        if source_node.context == schema.context:
            source_node.schemas_satisfied_by.add(schema)

        # context spin-off
        else:
            schema_node = self.lookup_node(schema.context)

            # node already exists in tree (generated from different source)
            if schema_node:
                schema_node.schemas_satisfied_by.add(schema)

            # novel assertion - requires new tree node
            else:
                new_node = SchemaTreeNode(schema.context)
                new_node.schemas_satisfied_by.add(schema)

                source_node.children += (new_node,)

                # add new node to nodes map
                self.add_node(new_node)

    def _update_schemas_would_satisfy(self, schema: Schema) -> None:
        schema_node = self.lookup_node(schema.result)
        if schema_node:
            schema_node.schemas_would_satisfy.add(schema)

        else:
            new_node = SchemaTreeNode(schema.result)
            new_node.schemas_would_satisfy.add(schema)

            # must be a primitive schema's result spin-off - add new node as child of root
            self.root.children += (new_node,)

            # add new node to nodes map
            self.add_node(new_node)

    def find(self, predicate: Callable[[Schema], bool], **kwargs) -> Collection[Schema]:
        return {schema for schema in self if predicate(schema)}

    def find_all_satisfied(self, state: State, **kwargs) -> Collection[Schema]:
        matches: set[SchemaTreeNode] = set()

        nodes_to_process = [self._root]
        while nodes_to_process:
            node = nodes_to_process.pop()
            if node.context.is_satisfied(state, **kwargs):
                matches.add(node)
                if node.children:
                    nodes_to_process += node.children

        return set(itertools.chain.from_iterable([node.schemas_satisfied_by for node in matches]))

    def find_all_would_satisfy(self, assertion: StateAssertion, **kwargs) -> Collection[Schema]:
        matches: set[SchemaTreeNode] = set()

        nodes_to_process: list[SchemaTreeNode] = [*self._root.leaves]
        while nodes_to_process:
            node = nodes_to_process.pop()
            context_as_state = node.context.as_state()
            if assertion.is_satisfied(context_as_state, **kwargs):
                matches.add(node)
                if node.parent:
                    nodes_to_process += [node.parent]

        return set(itertools.chain.from_iterable([node.schemas_would_satisfy for node in matches]))

    def validate(self) -> None:
        for node in LevelOrderIter(self._root):
            self._is_valid_node(node)

    def is_novel_result(self, result: StateAssertion) -> bool:
        # a bare schema must exist for every result, so we can simplify this search by restricting it to the root node
        return not any({result == s.result for s in self.root.schemas_satisfied_by})

    @property
    def root(self) -> SchemaTreeNode:
        return self._root

    @property
    def nodes_map(self) -> dict[StateAssertion, SchemaTreeNode]:
        return self._nodes_map

    def lookup_node(self, assertion: StateAssertion) -> SchemaTreeNode:
        """ Retrieves a SchemaTreeNode matching the given state assertion (if it exists). """
        return self._nodes_map.get(assertion.flatten())

    def add_node(self, node: SchemaTreeNode) -> None:
        """ Adds a SchemaTreeNode. """
        self._nodes_map[node.context] = node

    def _is_valid_node(self, node: SchemaTreeNode) -> None:
        node_asserts = node.context.flatten()

        # 1. node has proper depth for context
        if len(node.context.flatten()) != node.depth:
            raise ValueError('invalid node: depth must equal the number of item assertion in context minus 1')

        if node is not self.root:

            # 2. node's context contains all of its parents item assertions
            parent_asserts = node.parent.context.flatten()

            if not parent_asserts.issubset(node_asserts):
                raise ValueError('invalid node: node\'s assertions should contain all of parent\'s assertions')

        # consistency checks between node and its schemas
        for schema in node.schemas_satisfied_by:
            # 3. item assertions should be identical across all contained schemas, and equal to node's assertions
            if node_asserts != schema.context.flatten():
                raise ValueError('invalid node: schemas in schemas_satisfied_by must have same assertions as node')

            # 4. composite results should exist as contexts in the tree
            if (len(schema.result.flatten()) > 1
                    and schema.result.flatten() not in self.nodes_map):
                raise ValueError('invalid node: composite results must exist as a context in tree')

        for schema in node.schemas_would_satisfy:
            schema_result_asserts = schema.result.flatten()

            # 5. node's context should be a subset of the schemas' results that would satisfy it
            if not node_asserts.issubset(schema_result_asserts):
                raise ValueError('invalid node: schemas in schemas_would_satisfy must have result = node context')

        # 6. node is in nodes map's values
        if node not in self.nodes_map.values():
            raise ValueError('invalid node: node not found in nodes map')


# helper methods for SchemaTree
def generate_nodes_for_assertion(assertion: StateAssertion,
                                 tree: SchemaTree,
                                 ) -> SchemaTreeNode:
    """ Directly adds a new leaf node, and all necessary internal nodes, to this schema tree for the given assertion.

    Note: this method is primarily used to support the addition of bare schemas for composite, primitive items. In
    these cases, the agent starts with a built-in, composite assertion, and can directly learn a no context schema
    that contains that assertion as its result. This deviates from the usual process of learning composite item
    results that requires the result to first exist as a context of some schema within the tree.

    :param assertion: the assertion for which a node will be added
    :param tree: the tree to which the node will be added

    :return:
    """
    if assertion is NULL_STATE_ASSERT:
        return tree.root

    matches = find_nodes_matching_assertion(
        assertion=assertion,
        tree=tree,
        starting_node=tree.root
    )

    # select one of the tree's nodes with a context that best matches the given assertion
    max_items_matched = max([key for key in matches.keys()]) if matches else 0
    best_match = tree.root if max_items_matched == 0 else matches[max_items_matched][0]
    node = best_match

    # add internal nodes between root and leaf node so tree is valid
    missing_items = list(assertion.flatten().difference(node.context.flatten()))
    for item in missing_items:
        new_node = SchemaTreeNode(context=node.context.union([item]))
        tree.add_node(new_node)

        node.children += (new_node,)
        node = new_node

    return node


def find_nodes_matching_assertion(assertion: StateAssertion,
                                  tree: SchemaTree,
                                  starting_node: Optional[SchemaTreeNode] = None) -> dict[int, list[SchemaTreeNode]]:
    """ Find all nodes with contexts that are subsets of the given state assertion.

    :param assertion: the state assertion to which nodes' contexts will be compared
    :param tree: the schema tree on which the search will be performed
    :param starting_node: an optional node from which the search will begin

    :return: a dictionary containing lists of matched nodes keyed by their match length
    """
    # attempt to find an exact match
    exact_match = tree.lookup_node(assertion)
    if exact_match:
        return {len(exact_match.context): [exact_match]}

    # no exact match - gather partial matches
    partial_matches: dict[int, list[SchemaTreeNode]] = defaultdict(list)

    starting_node: SchemaTreeNode = tree.root if starting_node is None else starting_node
    child: SchemaTreeNode

    for child in starting_node.descendants:
        if child.context.issubset(assertion):
            items_matched = len(child.context)
            partial_matches[items_matched].extend([child])

    return partial_matches


# TODO: should this be added into the existing ADD method for SchemaTree?
def add_bare_schema_to_tree(tree: SchemaTree, schema: Schema) -> SchemaTreeNode:
    node: SchemaTreeNode = tree.lookup_node(schema.result)
    if node:
        node.schemas_would_satisfy += (schema,)
    else:
        node = generate_nodes_for_assertion(tree=tree, assertion=schema.result)
        node.schemas_would_satisfy.add(schema)
    return node


class ItemPool(metaclass=Singleton):
    """
    Implements a flyweight design pattern for Item types.
    """
    _items: dict[StateElement, Item] = dict()
    _composite_items: dict[frozenset[StateElement], CompositeItem] = dict()

    def __contains__(self, source: Any) -> bool:
        return source in ItemPool._items or source in ItemPool._composite_items

    def __len__(self) -> int:
        return len(ItemPool._items) + len(ItemPool._composite_items)

    def __iter__(self) -> Iterator[Item]:
        yield from ItemPool._items.values()
        yield from ItemPool._composite_items.values()

    @property
    def items(self) -> Collection[Item]:
        return ItemPool._items.values()

    @property
    def composite_items(self) -> Collection[CompositeItem]:
        return ItemPool._composite_items.values()

    def clear(self):
        ItemPool._items.clear()
        ItemPool._composite_items.clear()

    @singledispatchmethod
    def get(
            self,
            source: Any, /, *,
            item_type: Optional[Type[Item]] = None,
            read_only: bool = False,
            **kwargs
    ) -> Optional[Item]:
        raise NotImplementedError(f'Source type is not supported.')

    @get.register
    def _(
            self,
            source: StateElement, /, *,
            item_type: Optional[Type[Item]] = SymbolicItem,
            read_only: bool = False,
            **kwargs
    ) -> Optional[Item]:
        item_type = item_type

        obj = ItemPool._items.get(source)

        # create new item and add to pool if not found and not read_only
        if not obj and not read_only:
            obj = ItemPool._items[source] = item_type(source=source, **kwargs)

        return obj

    @get.register
    def _(
            self,
            source: frozenset, /, *,
            item_type: Optional[Type[CompositeItem]] = None,
            read_only: bool = False,
            **kwargs
    ) -> Optional[CompositeItem]:
        item_type = item_type or CompositeItem

        key = frozenset(source)
        obj = ItemPool._composite_items.get(key)

        # create new item and add to pool if not found and not read_only
        if not obj and not read_only:
            obj = ItemPool._composite_items[key] = item_type(source=key, **kwargs)

        return obj


class ReadOnlyItemPool(ItemPool):
    def __init__(self):
        self._pool = ItemPool()

    def get(self, source: Any, item_type: Optional[Type[Item]] = None, **kwargs) -> Optional[Item]:
        kwargs['read_only'] = True
        return self._pool.get(source, item_type=item_type, **kwargs)

    def clear(self):
        raise NotImplementedError('ReadOnlyItemPool does not support clear operation.')


class Chain(deque):
    def __str__(self):
        return ' -> '.join([str(link) for link in self])

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(tuple(self))

    def is_valid(self, raise_on_invalid: bool = False) -> bool:
        if len(self) == 0:
            return True

        if not all({isinstance(s, Schema) for s in self}):
            if raise_on_invalid:
                raise ValueError(f'All elements of Chain must be Schemas.')
            return False

        s1: Schema
        s2: Schema
        for s1, s2 in pairwise(self):
            if not s2.context.is_satisfied(s1.result.as_state()):
                if raise_on_invalid:
                    raise ValueError(f'Schemas contexts must be satisfied by their predecessor\'s result')
                return False
        return True


@singledispatch
def calc_primitive_value(other: Optional[Any]) -> float:
    raise TypeError(f'Primitive value not supported for this type: {type(other)}')


@calc_primitive_value.register
def _(se: StateElement) -> float:
    item = ReadOnlyItemPool().get(se)
    return item.primitive_value if item else 0.0


@calc_primitive_value.register
def _(state: State) -> float:
    if len(state) == 0:
        return 0.0
    items = [ReadOnlyItemPool().get(se) for se in state]
    return sum(i.primitive_value for i in items if i)


@calc_primitive_value.register
def _(assertion: StateAssertion) -> float:
    return sum(item.primitive_value for item in assertion)


@calc_primitive_value.register
def _(item: Item) -> float:
    return item.primitive_value


@calc_primitive_value.register
def _(item: CompositeItem) -> float:
    return item.primitive_value


@singledispatch
def calc_delegated_value(other: Optional[Any]) -> float:
    raise TypeError(f'Delegated value not supported for this type: {type(other)}')


@calc_delegated_value.register
def _(state: State) -> float:
    if len(state) == 0:
        return 0.0

    items = [ReadOnlyItemPool().get(se) for se in state]
    return sum(i.delegated_value for i in items if i) if items else 0.0


@calc_delegated_value.register
def _(assertion: StateAssertion) -> float:
    return sum(item.delegated_value for item in assertion)


@calc_delegated_value.register
def _(se: StateElement) -> float:
    item = ReadOnlyItemPool().get(se)
    return calc_delegated_value(item) if item else 0.0


@calc_delegated_value.register
def _(item: Item) -> float:
    return item.delegated_value


@calc_delegated_value.register
def _(item: CompositeItem) -> float:
    return item.delegated_value


def calc_value(o: Any) -> float:
    return calc_primitive_value(o) + calc_delegated_value(o)


def advantage(o: Any) -> float:
    """ Calculates the additional value this object has over a learned baseline. """
    baseline_value = get_global_stats().baseline_value
    return calc_value(o) - baseline_value


ControllerMap = dict[StateAssertion, Controller]
_controller_map: ControllerMap = (
    DefaultDictWithKeyFactory(default_for_controller_map)
)


def get_controller_map() -> ControllerMap:
    return _controller_map


default_delegated_value_helper = EligibilityTraceDelegatedValueHelper(
    discount_factor=0.5,
    eligibility_trace=ReplacingTrace(
        decay_strategy=GeometricDecayStrategy(rate=0.5),
        active_value=1.0
    )
)

default_action_trace = AccumulatingTrace(
    decay_strategy=GeometricDecayStrategy(rate=0.75),
    active_increment=0.25
)


def get_default_global_params() -> GlobalParams:
    zero_to_one_range_validator = RangeValidator(0.0, 1.0)
    item_correlation_validator = WhiteListValidator([
        'BarnardExactCorrelationTest',
        'CorrelationOnEncounter',
        'DrescherCorrelationTest',
        'FisherExactCorrelationTest',
    ])

    params = GlobalParams()

    # determines step size for incremental updates (e.g., this is used for delegated value updates)
    params.set(name='learning_rate', value=0.01, validator=zero_to_one_range_validator)

    # item correlation test used for determining relevance of extended context items
    params.set(
        name='ext_context.correlation_test',
        value='DrescherCorrelationTest',
        validator=item_correlation_validator
    )

    # item correlation test used for determining relevance of extended result items
    params.set(
        name='ext_result.correlation_test',
        value='CorrelationOnEncounter',
        validator=item_correlation_validator
    )

    # thresholds for determining the relevance of extended context items
    #     from 0.0 [weakest correlation] to 1.0 [strongest correlation]
    params.set(name='ext_context.positive_correlation_threshold', value=0.95, validator=zero_to_one_range_validator)
    params.set(name='ext_context.negative_correlation_threshold', value=0.95, validator=zero_to_one_range_validator)

    # thresholds for determining the relevance of extended result items
    #     from 0.0 [weakest correlation] to 1.0 [strongest correlation]
    params.set(name='ext_result.positive_correlation_threshold', value=0.95, validator=zero_to_one_range_validator)
    params.set(name='ext_result.negative_correlation_threshold', value=0.95, validator=zero_to_one_range_validator)

    # success threshold used for determining that a schema is reliable
    #     from 0.0 [schema has never succeeded] to 1.0 [schema always succeeds]
    params.set(name='schema.reliability_threshold', value=0.95, validator=zero_to_one_range_validator)

    # used by backward_chains (supports composite action) - determines the maximum chain length
    params.set(
        name='composite_actions.backward_chains.max_length',
        value=4,
        validator=RangeValidator(low=0)
    )

    # the probability of updating (via backward chains) the components associated with a composite action controller
    params.set(
        name='composite_actions.update_frequency',
        value=0.01,
        validator=zero_to_one_range_validator
    )

    # composite actions are created for novel result states that have values that are greater than the baseline
    # value by AT LEAST this amount
    params.set(
        name='composite_actions.min_baseline_advantage',
        value=0.25
    )

    # set default features
    default_features = {
        SupportedFeature.COMPOSITE_ACTIONS,
        SupportedFeature.EC_DEFER_TO_MORE_SPECIFIC_SCHEMA,
        SupportedFeature.EC_MOST_SPECIFIC_ON_MULTIPLE,
        SupportedFeature.ER_SUPPRESS_UPDATE_ON_EXPLAINED,
        SupportedFeature.EC_SUPPRESS_UPDATE_ON_RELIABLE,
        SupportedFeature.FREEZE_ITEM_STATS_UPDATES_ON_CORRELATION,
    }

    params.set(
        name='features',
        value=default_features,
        validator=SupportedFeatureValidator()
    )

    return params


default_global_params = get_default_global_params()
default_global_stats = GlobalStats()

_delegated_value_helper: Optional[DelegatedValueHelper] = None
_action_trace: Optional[Trace[Action]] = None
_global_params: Optional[GlobalParams] = None
_global_stats: Optional[GlobalStats] = None


def set_delegated_value_helper(delegated_value_helper: DelegatedValueHelper) -> None:
    global _delegated_value_helper
    _delegated_value_helper = deepcopy(delegated_value_helper)


def get_delegated_value_helper() -> DelegatedValueHelper:
    return _delegated_value_helper


def set_action_trace(action_trace: Trace[Action]) -> None:
    global _action_trace
    _action_trace = deepcopy(action_trace)


def get_action_trace() -> Trace[Action]:
    return _action_trace


def set_global_stats(global_stats: GlobalStats) -> None:
    global _global_stats
    _global_stats = deepcopy(global_stats)


def get_global_stats() -> GlobalStats:
    return _global_stats


def set_global_params(global_params: GlobalParams) -> None:
    global _global_params
    _global_params = deepcopy(global_params)


def get_global_params() -> GlobalParams:
    return _global_params


if not _delegated_value_helper:
    set_delegated_value_helper(default_delegated_value_helper)

if not _action_trace:
    set_action_trace(default_action_trace)

if not _global_params:
    set_global_params(default_global_params)

if not _global_stats:
    set_global_stats(default_global_stats)


def is_feature_enabled(feature: SupportedFeature) -> bool:
    params = get_global_params()
    return feature in params.get('features')
