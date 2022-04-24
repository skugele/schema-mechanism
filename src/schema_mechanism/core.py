from __future__ import annotations

import itertools
from abc import ABC
from abc import ABCMeta
from abc import abstractmethod
from collections import defaultdict
from collections import deque
from collections.abc import Collection
from collections.abc import Iterator
from enum import Enum
from enum import auto
from enum import unique
from functools import singledispatch
from functools import singledispatchmethod
from typing import Any
from typing import Hashable
from typing import NamedTuple
from typing import Optional
from typing import Protocol
from typing import Type
from typing import Union
from typing import runtime_checkable

import numpy as np
from anytree import AsciiStyle
from anytree import LevelOrderIter
from anytree import NodeMixin
from anytree import RenderTree

from schema_mechanism.share import GlobalParams
from schema_mechanism.share import SupportedFeature
from schema_mechanism.share import debug
from schema_mechanism.share import is_feature_enabled
from schema_mechanism.share import trace
from schema_mechanism.stats import CorrelationTable
from schema_mechanism.stats import FisherExactCorrelationTest
from schema_mechanism.stats import ItemCorrelationTest
from schema_mechanism.util import AccumulatingTrace
from schema_mechanism.util import AssociativeArrayList
from schema_mechanism.util import DefaultDictWithKeyFactory
from schema_mechanism.util import Observable
from schema_mechanism.util import Observer
from schema_mechanism.util import ReplacingTrace
from schema_mechanism.util import Singleton
from schema_mechanism.util import Trace
from schema_mechanism.util import UniqueIdMixin
from schema_mechanism.util import pairwise
from schema_mechanism.util import repr_str


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
    def delegated_value(self, item: Item) -> float:
        """ Returns the delegated value for the requested item.

        :param item: the item for which the delegated value will be returned.
        :return: the item's current delegated value
        """
        pass

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
        pass


class EligibilityTraceDelegatedValueHelper(DelegatedValueHelper):
    """ An eligibility-trace-based implementation of the DelegatedValueHelper.

    Drescher stated that delegated value "accrues to states that generally tend to facilitate other things of
    value" (p. 63). This occurs without reference to a specific goal. In other words, delegated value is the average,
    goal-agnostic, utility of an Item being On. This implementation of the DelegatedValueHlper retains the spirit of
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

    def __init__(self, discount_factor: float, trace_decay: float) -> None:
        """ Initializes the EligibilityTraceDelegatedValueHelper

        Note: a discount factor of 0.0 will give NO VALUE to future states when calculating delegated value (i.e.,
        delegated value will always be zero) while a discount factor of 1.0 will value more distant states EQUALLY with
        less distant states (i.e., state value will be undiscounted).

        :param discount_factor: quantifies the reduction in future state value w.r.t. an item's delegated value
        :param trace_decay: quantifies the time horizon over which future states influence an item's delegated values
        """
        self.discount_factor = discount_factor

        # note: eligibility traces usually multiply the decay rate by the discount factor. At the moment, it seems
        #       cleaner to disentangle these factors, but I may want to change this in the future.
        self._eligibility_trace: ReplacingTrace[Item] = ReplacingTrace(decay_rate=trace_decay)
        self._delegated_values: AssociativeArrayList[Item] = AssociativeArrayList()

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

    @property
    def eligibility_trace(self) -> Trace:
        return self._eligibility_trace

    def delegated_value(self, item: Item) -> float:
        """ Returns the delegated value for the requested item.

        (For a discussion of delegated value see Drescher 1991, p. 63.)

        :return: the Item's current delegated value
        """
        index = self._delegated_values.indexes([item], add_missing=True)[0]
        return self._delegated_values.values[index]

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
        self._eligibility_trace.update(active_items)

        # the value accessible from the current state
        target = self.effective_state_value(selection_state, result_state)

        # incremental update of delegated values towards target
        lr = GlobalParams().get('learning_rate')
        for item in ReadOnlyItemPool():
            tv = self._eligibility_trace[item]
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


class Item(ABC):

    def __init__(self, source: Any, primitive_value: float = None, **kwargs) -> None:
        super().__init__()

        self._source = source
        self._primitive_value = primitive_value or 0.0

        # TODO: let this be set by argument to __init__
        self._delegated_value_helper: DelegatedValueHelper = GlobalStats().delegated_value_helper

    # TODO: Need to be really careful with the default hash implementations which produce different values between
    # TODO: runs. This will kill and direct serialization/deserialization of data structures that rely on hashes.
    @abstractmethod
    def __hash__(self) -> int:
        pass

    def __str__(self) -> str:
        return str(self.source)

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
        pass

    @property
    def primitive_value(self) -> float:
        return 0.0 if self._primitive_value is None else self._primitive_value

    @primitive_value.setter
    def primitive_value(self, value: float) -> None:
        """ Sets the primitive value for this item.

            “The schema mechanism explicitly designates an item as corresponding to a top-level goal by assigning the
             item a positive value; an item can also take on a negative value, indicating a state to be avoided.”
            (see Drescher, 1991, p. 61)
            (see also, Drescher, 1991, Section 3.4.1)

        :param value: a positive or negative float
        :return: None
        """
        self._primitive_value = value

    @property
    def delegated_value(self) -> float:
        return self._delegated_value_helper.delegated_value(self)

    @abstractmethod
    def is_on(self, state: State, **kwargs) -> bool:
        return NotImplemented

    def is_off(self, state: State, **kwargs) -> bool:
        return not self.is_on(state, **kwargs)


class SymbolicItem(Item):
    """ A state element that can be thought as a proposition/feature. """

    def __init__(self, source: str, primitive_value: float = None, **kwargs):
        if not isinstance(source, str):
            raise ValueError(f'Source for symbolic item must be a str, not {type(source)}')

        super().__init__(source=source, primitive_value=primitive_value, **kwargs)

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


def non_composite_items(items: Collection[Item]) -> Collection[Item]:
    return list(filter(lambda i: not isinstance(i, CompositeItem), items))


def composite_items(items: Collection[Item]) -> Collection[Item]:
    return list(filter(lambda i: isinstance(i, CompositeItem), items))


def items_from_state(state: State) -> Collection[Item]:
    return [ItemPool().get(se) for se in state] if state else []


def item_contained_in(item_1: Item, item_2: Item) -> bool:
    set_1 = item_1.state_elements
    set_2 = item_2.state_elements

    # all non-negated elements of item_1 in item_2's state elements
    if set_1.issubset(set_2):

        if isinstance(item_1, CompositeItem):
            # if both args are composite, check negated item assertions
            if isinstance(item_2, CompositeItem):
                set_1 = item_1.negated_asserts
                set_2 = item_2.negated_asserts

                return set_1.issubset(set_2)

            # if first argument is composite, but 2nd is non-composite return False
            else:
                return False
        else:
            return True

    return False


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


class GlobalStats(metaclass=Singleton):
    def __init__(self, baseline_value: Optional[float] = None):
        self._n: int = 0

        self._baseline_value = baseline_value or 0.0

        self._dv_helper = EligibilityTraceDelegatedValueHelper(
            discount_factor=GlobalParams().get('delegated_value_helper.discount_factor'),
            trace_decay=GlobalParams().get('delegated_value_helper.decay_rate'))

        self._action_trace: AccumulatingTrace[Action] = AccumulatingTrace(
            decay_rate=GlobalParams().get('habituation_exploratory_strategy.decay.rate'))

    @property
    def n(self) -> int:
        return self._n

    @n.setter
    def n(self, value: int) -> None:
        self._n = value

    @property
    def delegated_value_helper(self) -> EligibilityTraceDelegatedValueHelper:
        return self._dv_helper

    @property
    def action_trace(self) -> Trace[Action]:
        return self._action_trace

    @property
    def baseline_value(self) -> float:
        """ A running average of the primitive state value encountered over all visited states.

        :return: the empirical (primitive value) baseline
        """
        return self._baseline_value

    @baseline_value.setter
    def baseline_value(self, value: float) -> None:
        self._baseline_value = value

    def update_baseline(self, state: State) -> None:
        """ Updates an unconditional running average of the primitive values of states encountered.

        :param state: a state
        :return: None
        """
        lr = GlobalParams().get('learning_rate')
        self._baseline_value += lr * (calc_primitive_value(state) - self._baseline_value)

    # TODO: implement this, and replace update_baseline
    def update(self):
        pass

    def reset(self):
        self._baseline_value = 0.0
        self._n = 0

        # TODO: this code is redundant with initializer code
        self._dv_helper = EligibilityTraceDelegatedValueHelper(
            discount_factor=GlobalParams().get('delegated_value_helper.discount_factor'),
            trace_decay=GlobalParams().get('delegated_value_helper.decay_rate'))

        self._action_trace: AccumulatingTrace[Action] = AccumulatingTrace(
            decay_rate=GlobalParams().get('habituation_exploratory_strategy.decay.rate'))


class SchemaStats:
    def __init__(self):
        self._n = 0  # total number of update events
        self._n_activated = 0
        self._n_success = 0

    def __repr__(self):
        attr_values = {
            'n': f'{self.n:,}',
            'n_activated': f'{self.n_activated:,}',
            'n_success': f'{self.n_success:,}',
        }

        return repr_str(self, attr_values)

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


class ItemStats(ABC):

    @property
    @abstractmethod
    def correlation_test(self) -> ItemCorrelationTest:
        pass

    @property
    @abstractmethod
    def positive_correlation_threshold(self) -> float:
        pass

    @property
    @abstractmethod
    def negative_correlation_threshold(self) -> float:
        pass

    @property
    def positive_correlation_stat(self) -> float:
        return self.correlation_test.positive_corr_statistic(self.as_table())

    @property
    def negative_correlation_stat(self) -> float:
        return self.correlation_test.negative_corr_statistic(self.as_table())

    @property
    def positive_correlation(self) -> bool:
        return self.correlation_test.positive_corr_statistic(self.as_table()) >= self.positive_correlation_threshold

    @property
    def negative_correlation(self) -> bool:
        return self.correlation_test.negative_corr_statistic(self.as_table()) >= self.negative_correlation_threshold

    @abstractmethod
    def as_table(self) -> CorrelationTable:
        pass

    @abstractmethod
    def update(self, **kwargs) -> None:
        pass


class ECItemStats(ItemStats):
    """ Extended context item-level statistics

        "each extended-context slot records the ratio of the probability that the schema will succeed (i.e., that
        its result will obtain) if the schema is activated when the slot's item is On, to the probability
        of success if that item is Off when the schema is activated." (Drescher, 1991, p. 73)
    """

    def __init__(self):
        super().__init__()

        self._n_success_and_on = 0
        self._n_success_and_off = 0
        self._n_fail_and_on = 0
        self._n_fail_and_off = 0

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, ECItemStats):
            return all({
                self._n_success_and_on == other._n_success_and_on,
                self._n_success_and_off == other._n_success_and_off,
                self._n_fail_and_on == other._n_fail_and_on,
                self._n_fail_and_off == other._n_fail_and_off,
            })
        return False if other is None else NotImplemented

    @property
    def correlation_test(self) -> ItemCorrelationTest:
        return GlobalParams().get('ext_context.correlation_test') or FisherExactCorrelationTest()

    @property
    def positive_correlation_threshold(self) -> float:
        return GlobalParams().get('ext_context.positive_correlation_threshold')

    @property
    def negative_correlation_threshold(self) -> float:
        return GlobalParams().get('ext_context.negative_correlation_threshold')

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
    def n_on(self) -> int:
        return self.n_success_and_on + self.n_fail_and_on

    @property
    def n_off(self) -> int:
        return self.n_success_and_off + self.n_fail_and_off

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

    def as_table(self) -> CorrelationTable:
        return self.n_success_and_on, self.n_fail_and_on, self.n_success_and_off, self.n_fail_and_off

    def __hash__(self):
        return hash((self._n_success_and_on,
                     self._n_success_and_off,
                     self._n_fail_and_on,
                     self._n_fail_and_off))

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


class ERItemStats(ItemStats):
    """ Extended result item-level statistics """

    def __init__(self):
        super().__init__()

        self._n_on_and_activated = 0
        self._n_on_and_not_activated = 0
        self._n_off_and_activated = 0
        self._n_off_and_not_activated = 0

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, ERItemStats):
            return all({
                self._n_on_and_activated == other._n_on_and_activated,
                self._n_on_and_not_activated == other._n_on_and_not_activated,
                self._n_off_and_activated == other._n_off_and_activated,
                self._n_off_and_not_activated == other._n_off_and_not_activated,
            })
        return False if other is None else NotImplemented

    @property
    def correlation_test(self) -> ItemCorrelationTest:
        return GlobalParams().get('ext_result.correlation_test') or FisherExactCorrelationTest()

    @property
    def positive_correlation_threshold(self) -> float:
        return GlobalParams().get('ext_result.positive_correlation_threshold')

    @property
    def negative_correlation_threshold(self) -> float:
        return GlobalParams().get('ext_result.negative_correlation_threshold')

    @property
    def n_on(self) -> int:
        return self._n_on_and_activated + self._n_on_and_not_activated

    @property
    def n_off(self) -> int:
        return self._n_off_and_activated + self._n_off_and_not_activated

    @property
    def n_activated(self) -> int:
        return self._n_on_and_activated + self._n_off_and_activated

    @property
    def n_not_activated(self) -> int:
        return self._n_on_and_not_activated + self._n_off_and_not_activated

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

    def update(self, on: bool, activated: bool, count: int = 1) -> None:
        if on and activated:
            self._n_on_and_activated += count

        elif on and not activated:
            self._n_on_and_not_activated += count

        elif not on and activated:
            self._n_off_and_activated += count

        elif not on and not activated:
            self._n_off_and_not_activated += count

    def as_table(self) -> CorrelationTable:
        return (self.n_on_and_activated,
                self.n_off_and_activated,
                self.n_on_and_not_activated,
                self.n_off_and_not_activated)

    def __hash__(self):
        return hash((self._n_on_and_activated,
                     self._n_off_and_activated,
                     self._n_on_and_not_activated,
                     self._n_off_and_not_activated))

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
            'n_off_and_not_activated': f'{self._n_off_and_not_activated:,}',
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


# A single immutable object that is meant to be used for all item instances that have never had stats updates
NULL_SCHEMA_STATS = ReadOnlySchemaStats()
NULL_EC_ITEM_STATS = ReadOnlyECItemStats()
NULL_ER_ITEM_STATS = ReadOnlyERItemStats()


class Assertion(ABC):
    def __init__(self, negated: bool, **kwargs) -> None:
        super().__init__(**kwargs)

        self._is_negated = negated

    @abstractmethod
    def __iter__(self) -> Iterator[Assertion]:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __contains__(self, assertion: Assertion) -> bool:
        pass

    @abstractmethod
    def __hash__(self) -> int:
        pass

    @property
    def is_negated(self) -> bool:
        return self._is_negated

    @abstractmethod
    def is_satisfied(self, state: State) -> bool:
        pass

    # TODO: This should be removed
    @property
    @abstractmethod
    def items(self) -> frozenset[Item]:
        pass

    # TODO: This should be removed
    @staticmethod
    def replicate_with(old: Assertion, new: Assertion) -> StateAssertion:
        for ia in new:
            if ia in old:
                raise ValueError('New assertion already exists in old assertion')

        return StateAssertion(asserts=(*old, *new))


class ItemAssertion(Assertion):

    def __init__(self, item: Item, negated: bool = False, **kwargs) -> None:
        super().__init__(negated, **kwargs)

        self._item = item

    def __iter__(self) -> Iterator[Assertion]:
        yield from [self]

    # TODO: This is ugly. Need a better way of handling CompositeItems.
    def __len__(self) -> int:
        return len(self._item.source) if isinstance(self._item, CompositeItem) else 1

    def __contains__(self, assertion: Assertion) -> bool:
        return self == assertion

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, ItemAssertion):
            return self._item == other._item and self.is_negated == other.is_negated
        elif isinstance(other, Item):
            return self._item == other

        return False if other is None else NotImplemented

    def __hash__(self) -> int:
        """ Returns a hash of this object.

        :return: an integer hash of this object
        """
        return hash((self._item, self.is_negated))

    def __str__(self) -> str:
        return f'{"~" if self.is_negated else ""}{self._item}'

    def __repr__(self) -> str:
        return repr_str(self, {'item': self._item, 'negated': self.is_negated})

    @property
    def item(self) -> Item:
        """ Returns the Item on which this ItemAssertion is based.

        This method is unique to ItemAssertions (i.e., it does not appear in parent class).
        :return: an Item
        """
        return self._item

    # TODO: This should be removed
    @property
    def items(self) -> frozenset[Item]:
        return frozenset([self._item])

    def is_satisfied(self, state: State, **kwargs) -> bool:
        if self.is_negated:
            return self._item.is_off(state, **kwargs)
        else:
            return self._item.is_on(state, **kwargs)


class StateAssertion(Assertion):

    def __init__(self,
                 asserts: Optional[Collection[ItemAssertion, ...]] = None,
                 negated: bool = False,
                 **kwargs):
        super().__init__(negated, **kwargs)

        # item assertions must be expanded to flatten composite item assertions
        _asserts = set()
        if asserts:
            for ia in asserts:
                if isinstance(ia.item, CompositeItem):
                    _asserts.update(*ia.item.source)
                else:
                    _asserts.update(ia)

        self._asserts = frozenset(filter(lambda a: not a.is_negated, _asserts)) if _asserts else frozenset()
        self._neg_asserts = frozenset(filter(lambda a: a.is_negated, _asserts)) if _asserts else frozenset()

        self._items = frozenset(itertools.chain.from_iterable(a.items for a in asserts)) if asserts else frozenset()

    def __iter__(self) -> Iterator[ItemAssertion]:
        yield from self._asserts
        yield from self._neg_asserts

    def __len__(self) -> int:
        return sum(len(ia) for ia in self._asserts) + sum(len(ia) for ia in self._neg_asserts)

    def __contains__(self, item_assert: ItemAssertion) -> bool:
        if isinstance(item_assert.item, CompositeItem):
            return all({ia in self._neg_asserts if ia.is_negated else ia in self._asserts
                        for ia in item_assert.item.source})

        return (item_assert in self._neg_asserts if item_assert.is_negated
                else item_assert in self._asserts)

    def __eq__(self, other) -> bool:
        if isinstance(other, StateAssertion):
            return (self._asserts == other._asserts
                    and self._neg_asserts == other._neg_asserts)

        return False if other is None else NotImplemented

    def __hash__(self) -> int:
        return hash((self._asserts, self._neg_asserts))

    def __str__(self) -> str:
        return ','.join(sorted(map(str, self)))

    def __repr__(self) -> str:
        return repr_str(self, {'asserts': str(self)})

    # TODO: This should be removed
    @property
    def items(self) -> frozenset[Item]:
        return self._items

    @property
    def asserts(self) -> frozenset[ItemAssertion]:
        return self._asserts

    @property
    def negated_asserts(self) -> frozenset[ItemAssertion]:
        return self._neg_asserts

    def is_satisfied(self, state: State, **kwargs) -> bool:
        """ Satisfied when all non-negated items are On, and all negated items are Off.

        :param state: the agent's current state
        :param kwargs: optional keyword arguments
        :return: True if this state assertion is satisfied given the current state; False otherwise.
        """
        return all({ia.is_satisfied(state, **kwargs) for ia in self})

    def as_state(self) -> State:
        """ Returns a State consistent with this StateAssertion.

        :return: a State
        """
        return frozenset(itertools.chain.from_iterable([ia.item.state_elements for ia in self.asserts]))

    @staticmethod
    def from_state(state: State) -> StateAssertion:
        """ Factory method for creating state assertions that would be satisfied by the given State.

        :param state: a State
        :return: a StateAssertion
        """
        return StateAssertion(asserts=[ItemAssertion(ReadOnlyItemPool().get(se)) for se in state])


NULL_STATE_ASSERT = StateAssertion()


# TODO: There may be a subtle bug here when the StateAssertion is negated and a CompositeItem containing
# TODO: that negated assertion is included in a negated ItemAssertion or StateAssertion.
class CompositeItem(StateAssertion, Item):
    """ A StateAssertion wrapper that functions like an Item.

    This class is primarily used to support ExtendedResult statistics when the ER_INCREMENTAL_RESULTS is disabled.
    """

    def __init__(self, source: StateAssertion, primitive_value: float = None, **kwargs) -> None:
        if len(source) < 2:
            raise ValueError('Source assertion must have at least two elements')

        super().__init__(source=source,
                         primitive_value=primitive_value,
                         asserts=[*source.asserts, *source.negated_asserts])

    @property
    def source(self) -> StateAssertion:
        return super().source

    @property
    def state_elements(self) -> set[StateElement]:
        return set(itertools.chain.from_iterable([ia.item.state_elements for ia in self.source.asserts]))

    @property
    def primitive_value(self) -> float:
        return self._primitive_value if self._primitive_value else calc_primitive_value(self.source)

    @primitive_value.setter
    def primitive_value(self, value: float) -> None:
        self._primitive_value = value

    def is_on(self, state: State, **kwargs) -> bool:
        return self.source.is_satisfied(state)

    def __contains__(self, assertion: Assertion) -> bool:
        if isinstance(assertion, ItemAssertion):
            return assertion in self.source
        return False

    def __eq__(self, other) -> bool:
        return super().__eq__(other)

    def __hash__(self) -> int:
        return hash(self.source)

    def __str__(self) -> str:
        return Item.__str__(self)

    def __repr__(self) -> str:
        return Item.__repr__(self)


class ExtendedItemCollection(Observable):
    def __init__(self, suppressed_items: Collection[Item] = None, null_member: ItemStats = None):
        super().__init__()

        self._null_member = null_member

        self._suppressed_items: frozenset[Item] = frozenset(suppressed_items) or frozenset()

        # FIXME: these names are confusing... they are not items! They are item assertions.
        self._relevant_items: set[ItemAssertion] = set()
        self._new_relevant_items: set[ItemAssertion] = set()

        self._stats: dict[Any, ItemStats] = defaultdict(self._get_null_stats)

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

    def __eq__(self, other: Any):
        if isinstance(other, ExtendedItemCollection):
            return all({
                # self._null_member == other._null_member,
                self._suppressed_items == other._suppressed_items,
                self._relevant_items == other._relevant_items,
                self._new_relevant_items == other._new_relevant_items,
                self._stats == other._stats
            })
        return False if other is None else NotImplemented

    @property
    def stats(self) -> dict[Any, Any]:
        return self._stats

    @property
    def suppressed_items(self) -> frozenset[Item]:
        return self._suppressed_items

    # TODO: "relevant items" is confusing. these are item assertion. this terminology is a carry-over from Drescher.
    @property
    def relevant_items(self) -> frozenset[Assertion]:
        return frozenset(self._relevant_items)

    def update_relevant_items(self, assertion: ItemAssertion, suppressed: bool = False):
        if assertion not in self._relevant_items:
            self._relevant_items.add(assertion)

            # if suppressed then no spin-offs will be created for this item
            if suppressed:
                debug(f'suppressing spin-off for item assertion {assertion}')
            else:
                self._new_relevant_items.add(assertion)

    def notify_all(self, **kwargs) -> None:
        if 'source' not in kwargs:
            kwargs['source'] = self

        super().notify_all(**kwargs)

        # clears the set
        self._new_relevant_items = set()

    # TODO: this is an ugly design! there must be a better way....
    @property
    def new_relevant_items(self) -> frozenset[Assertion]:
        return frozenset(self._new_relevant_items)

    @new_relevant_items.setter
    def new_relevant_items(self, value: Collection[Assertion]) -> None:
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

    def __init__(self, result: StateAssertion) -> None:
        super().__init__(suppressed_items=result.items, null_member=NULL_ER_ITEM_STATS)

    @property
    def stats(self) -> dict[Item, ERItemStats]:
        return super().stats

    def update(self, item: Item, on: bool, activated=False, count: int = 1) -> None:
        item_stats = self._stats[item]
        if item_stats is NULL_ER_ITEM_STATS:
            self._stats[item] = item_stats = ERItemStats()

        item_stats.update(on=on, activated=activated, count=count)

        if item not in self.suppressed_items:
            self._check_for_relevance(item, item_stats)

    def update_all(self, activated: bool, new: Collection[Item], lost: Collection[Item], count: int = 1) -> None:

        # "a trial for which the result was already satisfied before the action was taken does not count as a
        # positive-transition trial; and one for which the result was already unsatisfied does not count
        # as a negative-transition trial" (see Drescher, 1991, p. 72)
        if new:
            for item in new:
                self.update(item=item, on=True, activated=activated, count=count)

        if lost:
            for item in lost:
                self.update(item=item, on=False, activated=activated, count=count)

        if self.new_relevant_items:
            self.notify_all(source=self)

    def _check_for_relevance(self, item: Item, item_stats: ERItemStats) -> None:
        if item_stats.positive_correlation:
            item_assert = ItemAssertion(item)
            if item_assert not in self.relevant_items:
                self.update_relevant_items(item_assert)

        elif item_stats.negative_correlation:
            item_assert = ItemAssertion(item, negated=True)
            if item_assert not in self.relevant_items:
                self.update_relevant_items(
                    item_assert,
                    suppressed=is_feature_enabled(SupportedFeature.ER_POSITIVE_ASSERTIONS_ONLY))


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

    def __init__(self, context: StateAssertion) -> None:
        super().__init__(suppressed_items=context.items, null_member=NULL_EC_ITEM_STATS)

        self._pending_relevant_items = set()
        self._pending_max_specificity = -np.inf

    @property
    def stats(self) -> dict[Item, ECItemStats]:
        return super().stats

    def update(self, item: Item, on: bool, success: bool, count: int = 1) -> None:
        item_stats = self._stats[item]
        if item_stats is NULL_EC_ITEM_STATS:
            self._stats[item] = item_stats = ECItemStats()

        item_stats.update(on, success, count)

        if item not in self.suppressed_items:
            self._check_for_relevance(item, item_stats)

    def update_all(self, state: State, success: bool, count: int = 1) -> None:
        # bypass updates when a more specific spinoff schema exists
        if (is_feature_enabled(SupportedFeature.EC_DEFER_TO_MORE_SPECIFIC_SCHEMA)
                and self.defer_update_to_spin_offs(state)):
            return

        for item in ItemPool().items:
            self.update(item=item, on=item.is_on(state), success=success, count=count)

        self.check_pending_relevant_items()

        if self.new_relevant_items:
            self.notify_all(source=self)

            # after spinoff, all stats are reset to remove the influence of relevant items in stats; tracking the
            # correlations for these items is deferred to the schema's context spin-offs
            if is_feature_enabled(SupportedFeature.EC_DEFER_TO_MORE_SPECIFIC_SCHEMA):
                self.stats.clear()

    def defer_update_to_spin_offs(self, state: State) -> bool:
        """ Checks if this state activates previously recognized non-negated relevant items. If so, defer to spin-offs.

        Note: This method supports the optional enhancement SupportedFeature.EC_DEFER_TO_MORE_SPECIFIC_SCHEMA

        :param state: a State
        :return: True if should defer stats updates for this state to this schema's spin-offs.
        """
        for ia in self.relevant_items:
            deferring = (
                not ia.is_negated and ia.is_satisfied(state)
                if is_feature_enabled(SupportedFeature.EC_POSITIVE_ASSERTIONS_ONLY) else
                ia.is_satisfied(state)
            )

            if deferring:
                trace(f'deferring update to spin_off for state {state} due to relevant assertion {ia}')
                return True
        return False

    @property
    def pending_relevant_items(self) -> frozenset[ItemAssertion]:
        return frozenset(self._pending_relevant_items)

    @property
    def pending_max_specificity(self) -> float:
        return self._pending_max_specificity

    # TODO: Rename this method. It is not a check, but more of a flush of the pending items to the super class.
    def check_pending_relevant_items(self) -> None:
        for ia in self._pending_relevant_items:
            self.update_relevant_items(
                assertion=ia,
                suppressed=is_feature_enabled(SupportedFeature.EC_POSITIVE_ASSERTIONS_ONLY) and ia.is_negated)

        self.clear_pending_relevant_items()

    def clear_pending_relevant_items(self) -> None:
        self._pending_relevant_items.clear()
        self._pending_max_specificity = -np.inf

    def _check_for_relevance(self, item: Item, item_stats: ECItemStats) -> None:

        # if item is relevant, a new item assertion is created
        item_assert = (
            ItemAssertion(item) if item_stats.positive_correlation else
            ItemAssertion(item, negated=True) if item_stats.negative_correlation else
            None
        )

        if item_assert and item_assert not in self.relevant_items:
            specificity = self.stats[item].specificity

            # if enabled, this enhancement allows only a single relevant item per update (the most "specific").
            if is_feature_enabled(SupportedFeature.EC_MOST_SPECIFIC_ON_MULTIPLE):
                # item is less specific than earlier item; suppress it on this update.
                if specificity < self._pending_max_specificity:
                    return

                # item is the most specific so far; replace previous pending item assertion.
                else:
                    self._pending_relevant_items.clear()
                    self._pending_max_specificity = max(self._pending_max_specificity, specificity)

            self._pending_relevant_items.add(item_assert)


class Controller:
    def __init__(self, goal_state: StateAssertion):
        self._goal_state = goal_state
        self._proximity: dict[Schema, float] = defaultdict(float)
        self._total_cost: dict[Schema, float] = defaultdict(float)
        self._components: set[Schema] = set()
        self._descendants: set[Schema] = set()

    def __eq__(self, other):
        if self is other:
            return True
        if isinstance(other, Controller):
            return self._goal_state == other.goal_state
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

    def proximity(self, schema: Schema) -> float:
        """ Returns the proximity (i.e., closeness) of a schema to the composite action's goal state.

        "Proximity is inversely proportionate to the expected time to reach the goal state, derived from the
         expected activation time of the schemas in the relevant chain; proximity is also proportionate to those
         schemas' reliability, and inversely proportionate to their cost of activation." (See Drescher 1991, p. 60)

        :param schema: a component schema
        :return: a float that quantifies the schema's goal state proximity
        """
        return self._proximity[schema]

    def total_cost(self, schema: Schema) -> float:
        """ Returns the total cost that would be incurred in order to achieve the composite action's goal state.

        :param schema: a component schema
        :return: a float that quantifies the total cost
        """
        return self._total_cost[schema]

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
                self._proximity[schema] = 1.0 / avg_duration_to_goal_state
                self._total_cost[schema] = total_cost_to_goal_state

        # TODO: Implement this...
        # "each time a composite action is explicitly initiated, the controller keeps track of which component
        #  schemas are actually activated and when.... If the action successfully culminates in the goal state,
        #  the actual cost and duration of execution from each entry point are compared with the proximity
        #  information stored in the slot of each component actually activated; in case of discrepancy, the stored
        #  information is adjusted in the direction of the actual data. If the action fails to reach its goal
        #  state, the proximity measures for the utilized components are degraded." (See Drescher 1991, p. 92)

    def contained_in(self, schema: Schema) -> bool:
        if not schema.action.is_composite():
            return False

        if self == schema.action.controller:
            return True

        controller = schema.action.controller
        for component in itertools.chain.from_iterable([controller.components, controller.descendants]):
            if component.action.is_composite() and self == component.action.controller:
                return True
        return False


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

    def contained_in(self, *args, **kwargs) -> bool:
        return False

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
                return self._uid == other._uid

        return False if other is None else NotImplemented

    def __hash__(self) -> int:
        return hash(self._label or self._uid)

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

    _controller_map: dict[StateAssertion, Controller] = (
        DefaultDictWithKeyFactory(default_for_controller_map)
    )

    def __init__(self, goal_state: StateAssertion, **kwargs):
        super().__init__(**kwargs)

        if not goal_state:
            raise ValueError('Goal state is not optional.')

        self._controller = CompositeAction._controller_map[goal_state]

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
    def all_satisfied_by(cls, state: State) -> Collection[Controller]:
        satisfied: list[Controller] = []
        for goal_state, controller in cls._controller_map.items():
            if goal_state.is_satisfied(state):
                satisfied.append(controller)
        return satisfied

    @classmethod
    def reset(cls) -> None:
        return cls._controller_map.clear()


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
                 result: Optional[StateAssertion] = None,
                 **kwargs):
        super().__init__()

        self._context: Optional[StateAssertion] = context or NULL_STATE_ASSERT
        self._action: Action = action
        self._result: Optional[StateAssertion] = result or NULL_STATE_ASSERT

        if self.action is None:
            raise ValueError('Action cannot be None')

        self._stats: SchemaStats = SchemaStats()

        self._is_bare = not (context or result)

        self._extended_context: ExtendedContext = (
            None
            if self._is_bare else
            ExtendedContext(self._context)
        )

        self._extended_result: ExtendedResult = (
            ExtendedResult(self._result)
            if self._is_bare or is_feature_enabled(SupportedFeature.ER_INCREMENTAL_RESULTS) else
            None
        )

        # TODO: Need to update overriding conditions.
        self._overriding_conditions: Optional[StateAssertion] = None

        # This observer registration is used to notify the schema when a relevant item has been detected in its
        # extended context or extended result
        if not self._is_bare:
            self._extended_context.register(self)

        if self._is_bare or is_feature_enabled(SupportedFeature.ER_INCREMENTAL_RESULTS):
            self._extended_result.register(self)

        self._avg_duration: Optional[float] = None
        self._cost: Optional[float] = 1.0

    def __eq__(self, other) -> bool:
        # SchemaPool should be used for all item creation, so this should be an optimization
        if self is other:
            return True

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
        return (
                f'{self.context}/{self.action}/{self.result} ' +
                f'[rel: {self.reliability:.2}]'
        )

    def __repr__(self) -> str:
        return repr_str(self, {'uid': self.uid,
                               'context': self.context,
                               'action': self.action,
                               'result': self.result,
                               'overriding_conditions': self.overriding_conditions,
                               'reliability': self.reliability, })

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

        # all (negated and non-negated) assertions in schema's result must be satisfied
        is_satisfied = self.result.is_satisfied(state)

        # all state elements must be accounted for by POSITIVE assertions
        state_items = set(ItemPool().get(se) for se in state)

        # only include non-negated assertions
        result_items = set()
        for ia in self.result.asserts:
            item = ia.item
            if isinstance(item, CompositeItem):
                for nested_ia in item.source.asserts:
                    result_items.add(nested_ia.item)
            else:
                result_items.add(item)

        return is_satisfied and state_items == result_items

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
        if self.avg_duration is np.nan:
            self.avg_duration = duration
        else:
            self.avg_duration += (1.0 / float(self._stats.n)) * (duration - self.avg_duration)

        # update extended result stats
        if self._extended_result:
            if is_feature_enabled(SupportedFeature.ER_SUPPRESS_UPDATE_ON_EXPLAINED) and explained:
                trace(f'update suppressed for schema {self} because its result was explained by a reliable schema')
            else:
                self._extended_result.update_all(activated=activated, new=new, lost=lost, count=count)

        # update extended context stats
        if all((self._extended_context, activated, selection_state)):
            self._extended_context.update_all(state=selection_state, success=succeeded, count=count)

    # invoked by a schema's extended context or extended result when a relevant item is discovered
    def receive(self, **kwargs) -> None:

        # ext_source should be an ExtendedContext, ExtendedResult, or one of their subclasses
        ext_source: ExtendedItemCollection = kwargs['source']
        relevant_items: Collection[Assertion] = ext_source.new_relevant_items

        spin_off_type = (
            Schema.SpinOffType.CONTEXT if isinstance(ext_source, ExtendedContext) else
            Schema.SpinOffType.RESULT if isinstance(ext_source, ExtendedResult) else
            None
        )

        if not spin_off_type:
            raise ValueError(f'Unrecognized source in receive: {type(ext_source)}')

        self.notify_all(source=self, spin_off_type=spin_off_type, relevant_items=relevant_items)


class SchemaUniqueKey(NamedTuple):
    action: Action
    context: Optional[StateAssertion] = None
    result: Optional[StateAssertion] = None


class SchemaPool(metaclass=Singleton):
    """
    Implements a flyweight design pattern for Schema types.
    """
    _schemas: dict[SchemaUniqueKey, Schema] = dict()

    def __contains__(self, key: SchemaUniqueKey) -> bool:
        return key in SchemaPool._schemas

    def __len__(self) -> int:
        return len(SchemaPool._schemas)

    def __iter__(self) -> Iterator[Schema]:
        yield from SchemaPool._schemas.values()

    def __getstate__(self) -> dict[str, Any]:
        return {'_schemas': SchemaPool._schemas}

    def __setstate__(self, state: dict[str:Any]) -> None:
        sp = SchemaPool()
        for key in state:
            setattr(sp, key, state[key])

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
    threshold = threshold or GlobalParams().get('reliability_threshold')
    return schema.reliability != np.NAN and schema.reliability >= threshold


class SchemaTreeNode(NodeMixin):
    def __init__(self,
                 context: Optional[StateAssertion] = None,
                 label: str = None) -> None:
        self._context = context or NULL_STATE_ASSERT
        self._label = label

        self._schemas_satisfied_by = set()
        self._schemas_would_satisfy = set()

    def __hash__(self) -> int:
        return hash(self._context)

    def __eq__(self, other) -> bool:
        if isinstance(other, SchemaTreeNode):
            return self._context == other._context

    def __str__(self) -> str:
        return self._label if self._label else f'{self._context}'

    def __repr__(self) -> str:
        return repr_str(self, {'context': self._context,
                               'label': self._label})

    @property
    def context(self) -> Optional[StateAssertion]:
        return self._context

    @property
    def label(self) -> str:
        return self._label

    @property
    def schemas_satisfied_by(self) -> set[Schema]:
        return self._schemas_satisfied_by

    @schemas_satisfied_by.setter
    def schemas_satisfied_by(self, value) -> None:
        self._schemas_satisfied_by = value

    @property
    def schemas_would_satisfy(self) -> set[Schema]:
        return self._schemas_would_satisfy

    @schemas_would_satisfy.setter
    def schemas_would_satisfy(self, value) -> None:
        self._schemas_would_satisfy = value


class SchemaTree:
    """ A search tree of SchemaTreeNodes with the following special properties:

    1. Each tree node contains a set of schemas with identical contexts and actions.
    2. Each tree node (except the root) has the same action as their descendants.
    3. Each tree node's depth equals the number of item assertion in its context plus one; for example, the
    tree nodes corresponding to primitive (action only) schemas would have a tree height of one.
    4. Each tree node's context contains all of the item assertion in their ancestors plus one new item assertion
    not found in ANY ancestor. For example, if a node's parent's context contains item assertion 1,2,3, then it
    will contain 1,2,and 3 plus a new item assertion (say 4).

    """

    def __init__(self, schemas: Collection[Schema]) -> None:
        """ Initializes this SchemaTree from a set of bare, primitive schemas for built-in, primitive actions.

        :param schemas: a collection of bare schemas (should contain one for each primitive action)
        """
        if not schemas:
            raise ValueError('SchemaTree must be initialized with a collection of bare schemas.')

        self._root = SchemaTreeNode(label='root')

        self._nodes: dict[StateAssertion, SchemaTreeNode] = dict()
        self._nodes[self._root.context] = self._root

        self._n_schemas: int = 0

        self.add_bare_schemas(schemas)

    def __iter__(self) -> Iterator[SchemaTreeNode]:
        return LevelOrderIter(node=self.root)

    def __len__(self) -> int:
        """ Returns the number of SchemaTreeNodes (not the number of schemas).

        :return: The number of SchemaTreeNodes in this tree.
        """
        return len(self._nodes)

    def __contains__(self, s: Union[SchemaTreeNode, Schema]) -> bool:
        if isinstance(s, SchemaTreeNode):
            return s.context in self._nodes
        elif isinstance(s, Schema):
            node = self._nodes.get(s.context)
            return s in node.schemas_satisfied_by if node else False

        return False

    def __str__(self) -> str:
        return RenderTree(self._root, style=AsciiStyle()).by_attr(lambda s: str(s))

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, SchemaTree):
            return all({
                self._root == other._root,
                self._nodes == other._nodes,
                self._n_schemas == other._n_schemas
            })
        return False if other is None else NotImplemented

    @property
    def root(self) -> SchemaTreeNode:
        return self._root

    @property
    def n_schemas(self) -> int:
        return self._n_schemas

    @property
    def height(self) -> int:
        return self.root.height

    def get(self, assertion: StateAssertion) -> SchemaTreeNode:
        """ Retrieves the SchemaTreeNode matching this schema's context and action (if it exists).

        :param assertion: the assertion on which this retrieval is based
        :return: a SchemaTreeNode (if found) or raises a KeyError
        """
        return self._nodes[assertion]

    def add_bare_schemas(self, schemas: Collection[Schema]) -> None:
        trace(f'adding bare schemas! [{[str(s) for s in schemas]}]')
        if not schemas:
            raise ValueError('Schemas cannot be empty or None')

        if any({not schema.is_bare() for schema in schemas}):
            raise ValueError('Schemas must be bare (action-only) schemas')

        # needed because schemas to add may already exist in set reducing total new count
        len_before_add = len(self.root.schemas_satisfied_by)
        self.root.schemas_satisfied_by |= set(schemas)
        self._n_schemas += len(self.root.schemas_satisfied_by) - len_before_add

    def add_context_spin_offs(self, source: Schema, spin_offs: Collection[Schema]) -> None:
        """ Adds context spin-off schemas to this tree.

        :param source: the source schema that resulted in these spin-off schemas.
        :param spin_offs: the spin-off schemas.
        :return: None
        """
        self.add(source, frozenset(spin_offs), Schema.SpinOffType.CONTEXT)

    def add_result_spin_offs(self, source: Schema, spin_offs: Collection[Schema]):
        """ Adds result spin-off schemas to this tree.

        :param source: the source schema that resulted in these spin-off schemas.
        :param spin_offs: the spin-off schemas.
        :return: None
        """
        self.add(source, frozenset(spin_offs), Schema.SpinOffType.RESULT)

    def find_all_satisfied(self, state: State, **kwargs) -> Collection[SchemaTreeNode]:
        """ Returns a collection of tree nodes containing schemas with contexts that are satisfied by this state.

        :param state: the state
        :return: a collection of schemas
        """
        matches: set[SchemaTreeNode] = set()

        nodes_to_process = [self._root]
        while nodes_to_process:
            node = nodes_to_process.pop()
            if node.context.is_satisfied(state, **kwargs):
                matches.add(node)
                if node.children:
                    nodes_to_process += node.children

        return matches

    def find_all_would_satisfy(self, assertion: StateAssertion, **kwargs) -> Collection[SchemaTreeNode]:
        matches: set[SchemaTreeNode] = set()

        nodes_to_process: list[SchemaTreeNode] = [*self._root.leaves]
        while nodes_to_process:
            node = nodes_to_process.pop()
            context_as_state = node.context.as_state()
            if assertion.is_satisfied(context_as_state, **kwargs):
                matches.add(node)
                if node.parent:
                    nodes_to_process += [node.parent]

        return matches

    def is_valid_node(self, node: SchemaTreeNode, raise_on_invalid: bool = False) -> bool:
        # 1. node is in tree (path from root to node)
        if node not in self:
            if raise_on_invalid:
                raise ValueError('invalid node: no path from node to root')
            return False

        # 2. node has proper depth for context
        if len(node.context) != node.depth:
            if raise_on_invalid:
                raise ValueError('invalid node: depth must equal the number of item assertion in context minus 1')
            return False

        if node is not self.root:

            # 3. node's context contains all of parents
            if not all({ia in node.context for ia in node.parent.context}):
                if raise_on_invalid:
                    raise ValueError('invalid node: context should contain all of parent\'s item assertion')
                return False

            # 4. node's context contains exactly one item assertion not in parent's context
            if len(node.parent.context) + 1 != len(node.context):
                if raise_on_invalid:
                    raise ValueError('invalid node: context must differ from parent in exactly one assertion.')
                return False

        # consistency checks between node and its schemas
        for s in node.schemas_satisfied_by:

            # 5. contexts should be identical across all contained schemas, and equal to node's context
            if node.context != s.context:
                if raise_on_invalid:
                    raise ValueError('invalid node: schemas in schemas_satisfied_by must have same context as node')
                return False

            # 6. composite results should exist as contexts in the tree
            if len(s.result) > 1 and SchemaTreeNode(context=s.result) not in self:
                if raise_on_invalid:
                    raise ValueError('invalid node: composite results must exist as a context in tree')
                return False

        for s in node.schemas_would_satisfy:

            # 7. results should be identical across all contained schemas, and equal to node's context
            if node.context != s.result:
                if raise_on_invalid:
                    raise ValueError('invalid node: schemas in schemas_would_satisfy must have result = node context')
                return False

        return True

    def validate(self, raise_on_invalid: bool = False) -> Collection[SchemaTreeNode]:
        """ Validates that all nodes in the tree comply with its invariant properties and returns invalid nodes.

        :return: A set of invalid nodes (if any).
        """
        return set([node for node in LevelOrderIter(self._root,
                                                    filter_=lambda n: not self.is_valid_node(n, raise_on_invalid))])

    def add(self,
            source: Union[Schema, SchemaTreeNode],
            spin_offs: frozenset[Schema],
            spin_off_type: Optional[Schema.SpinOffType] = None) -> SchemaTreeNode:
        """ Adds schemas to this schema tree.

        :param source: the "source" schema that generated the given (primitive or spin-off) schemas, or the previously
        added tree node containing that "source" schema.
        :param spin_offs: a collection of spin-off schemas
        :param spin_off_type: the schema spin-off type (CONTEXT or RESULT), or None when adding primitive schemas

        :return: the parent node for which the add operation occurred
        """
        trace(f'adding schemas! [parent: {source}, spin-offs: {[str(s) for s in spin_offs]}]')
        if not spin_offs:
            raise ValueError('Spin-off schemas cannot be empty or None')

        if any({source.action != s.action for s in spin_offs}):
            raise ValueError('Spin-off schemas must have the same action as their source.')

        try:
            node = source if isinstance(source, SchemaTreeNode) else self.get(source.context)

            if Schema.SpinOffType.RESULT is spin_off_type:
                # needed because schemas to add may already exist in set reducing total new count
                len_before_add = len(node.schemas_satisfied_by)
                node.schemas_satisfied_by |= spin_offs
                self._n_schemas += len(node.schemas_satisfied_by) - len_before_add

            # for context spin-offs
            else:
                for s in spin_offs:
                    match = self._nodes.get(s.context)

                    # node already exists in tree (generated from different source)
                    if match:
                        if s not in match.schemas_satisfied_by:
                            match.schemas_satisfied_by.add(s)
                            self._n_schemas += 1

                    else:
                        new_node = SchemaTreeNode(s.context)
                        new_node.schemas_satisfied_by.add(s)

                        node.children += (new_node,)

                        self._nodes[s.context] = new_node

                        self._n_schemas += len(new_node.schemas_satisfied_by)

            # updates schemas_would_satisfy, creating a new tree node if necessary
            for s in spin_offs:
                match = self._nodes.get(s.result)

                # node already exists in tree (should always be the case for composite results)
                if match:
                    if s not in match.schemas_would_satisfy:
                        match.schemas_would_satisfy.add(s)
                else:
                    if len(s.result) != 1:
                        raise ValueError(f'Encountered an illegal composite result spin-off: {s.result}')

                    new_node = SchemaTreeNode(s.result)
                    new_node.schemas_would_satisfy.add(s)

                    # must be a primitive schema's result spin-off - add new node as child of root
                    self.root.children += (new_node,)

                    self._nodes[s.result] = new_node

            return node
        except KeyError:
            raise ValueError('Source schema does not have a corresponding tree node.')


class ItemPool(metaclass=Singleton):
    """
    Implements a flyweight design pattern for Item types.
    """
    _items: dict[StateElement, Item] = dict()
    _composite_items: dict[StateAssertion, CompositeItem] = dict()

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
    def get(self, source: Any, /, *, item_type: Optional[Type[Item]] = None, **kwargs) -> Optional[Item]:
        raise NotImplementedError(f'Source type is not supported.')

    @get.register
    def _(self, source: StateElement, /, *, item_type: Optional[Type[Item]] = None, **kwargs) -> Optional[Item]:
        read_only = kwargs.get('read_only', False)
        item_type = item_type or SymbolicItem

        obj = ItemPool._items.get(source)

        # create new item and add to pool if not found and not read_only
        if not obj and not read_only:
            obj = ItemPool._items[source] = item_type(source, **kwargs)

        return obj

    @get.register
    def _(self,
          source: StateAssertion, /, *,
          item_type: Optional[Type[CompositeItem]] = None,
          **kwargs) -> Optional[CompositeItem]:
        read_only = kwargs.get('read_only', False)
        item_type = item_type or CompositeItem

        obj = ItemPool._composite_items.get(source)

        # create new item and add to pool if not found and not read_only
        if not obj and not read_only:
            obj = ItemPool._composite_items[source] = item_type(source, **kwargs)

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
def _(assertion: ItemAssertion) -> float:
    return 0.0 if assertion.is_negated else calc_primitive_value(assertion.item)


@calc_primitive_value.register
def _(assertion: StateAssertion) -> float:
    if assertion.is_negated:
        return 0.0
    items = [ia.item for ia in assertion.asserts]
    return sum(i.primitive_value for i in items)


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
def _(assertion: ItemAssertion) -> float:
    return 0.0 if assertion.is_negated else calc_delegated_value(assertion.item)


@calc_delegated_value.register
def _(assertion: StateAssertion) -> float:
    if assertion.is_negated:
        return 0.0
    items = [ia.item for ia in assertion.asserts if ia.item]
    return sum(i.delegated_value for i in items) if items else 0.0


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
