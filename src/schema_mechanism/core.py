from __future__ import annotations

import itertools
import sys
from abc import ABC
from abc import abstractmethod
from collections import Iterable
from collections import defaultdict
from collections.abc import Collection
from collections.abc import Hashable
from collections.abc import Iterator
from collections.abc import MutableSet
from datetime import datetime
from enum import Enum
from enum import IntEnum
from enum import auto
from enum import unique
from time import time
from typing import Any
from typing import Optional
from typing import TextIO
from typing import Type
from typing import Union

import numpy as np
import scipy.stats as stats
from anytree import AsciiStyle
from anytree import LevelOrderIter
from anytree import NodeMixin
from anytree import RenderTree

from schema_mechanism.util import Observable
from schema_mechanism.util import Observer
from schema_mechanism.util import Singleton
from schema_mechanism.util import UniqueIdMixin
from schema_mechanism.util import repr_str
from schema_mechanism.validate import MultiValidator
from schema_mechanism.validate import NULL_VALIDATOR
from schema_mechanism.validate import RangeValidator
from schema_mechanism.validate import SubClassValidator
from schema_mechanism.validate import TypeValidator
from schema_mechanism.validate import Validator

StateElement = Hashable


class Activatable(ABC):
    @abstractmethod
    def is_on(self, state: State, **kwargs) -> bool: ...

    def is_off(self, state: State, **kwargs) -> bool:
        return not self.is_on(state, **kwargs)


# TODO: Better name?
class ValueBearer(ABC):
    @property
    @abstractmethod
    def primitive_value(self) -> float: ...

    @property
    @abstractmethod
    def avg_accessible_value(self) -> float: ...

    @property
    @abstractmethod
    def delegated_value(self) -> float: ...


class State(ValueBearer):
    def __init__(self, elements: Collection[StateElement], label: Optional[str] = None) -> None:
        super().__init__()

        self._elements = frozenset(elements)
        self._label = label

    @property
    def elements(self) -> Collection[StateElement]:
        return self._elements

    @property
    def label(self) -> Optional[str]:
        return self._label

    @property
    def primitive_value(self) -> float:
        if not self._elements:
            return 0.0

        return sum(ReadOnlyItemPool().get(se).primitive_value for se in self._elements)

    @property
    def delegated_value(self) -> float:
        if not self._elements:
            return 0.0 - GlobalStats().baseline_value
        return np.max([ReadOnlyItemPool().get(se).delegated_value for se in self._elements])

    @property
    def avg_accessible_value(self) -> float:
        if not self._elements:
            return 0.0
        items = [ReadOnlyItemPool().get(se) for se in self._elements]
        return np.max([i.avg_accessible_value for i in items])

    def __len__(self) -> int:
        return len(self._elements)

    def __iter__(self) -> Iterator[StateElement]:
        yield from self._elements

    def __contains__(self, element: StateElement) -> bool:
        return element in self._elements

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, State):
            return self._elements == other._elements
        return False if other is None else NotImplemented

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return hash(self._elements)

    def __str__(self) -> str:
        e_str = ','.join([str(se) for se in self._elements])
        return f'{e_str} ({self._label})' if self._label else e_str


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
    if not all((s_prev, s_curr)):
        return frozenset()

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
    if not all((s_prev, s_curr)):
        return frozenset()

    # singular
    lost = [ItemPool().get(se) for se in s_prev if se not in s_curr]

    # conjunctions
    for ci in ItemPool().composite_items:
        if ci.is_on(s_prev) and not ci.is_on(s_curr):
            lost.append(ci)

    return frozenset(lost)


class DelegatedValueHelper:
    """ A helper class that performs calculations necessary to derive an Item's delegated value.

    Note: Drescher's implementation of delegated values uses forwarding chaining based on "parallel broadcasts"
    (see Drescher 1991, p. 101). The current implementation deviates from Drescher's for performance reasons (performant
    parallel broadcasts are not possible in Python). Instead of forward chaining, it uses a "backward" view that is
    similar in spirit to eligibility traces. These value "traces" are initiated whenever the tracked item is in an On
    state and terminate some configurable number of state transitions away from the state in which the Item was On. Upon
    termination the trace's value (i.e., the maximum state value encountered along the trace) is used to update a
    running average of the accessible value for the item. The trace's value also takes into account the average
    accessible values of the items encountered during the trace.

    Delegated value is calculated as the difference between the Item's average accessible value when On and a
    "baseline" state value maintained in the global statistics. When the average accessible value when On is greater
    than the baseline, the delegated value will be positive. Similarly, when it is less than the baseline, the
    delegated value will be negative.
    """

    def __init__(self, item: Item) -> None:
        self._item = item

        self._dv_trace_updates_remaining = 0
        self._dv_trace_value = -np.inf

        self._dv_avg_accessible_value = 0.0

    @property
    def item(self) -> Item:
        return self._item

    @property
    def trace_updates_remaining(self) -> int:
        return self._dv_trace_updates_remaining

    @property
    def trace_value(self) -> float:
        return self._dv_trace_value

    @property
    def avg_accessible_value(self) -> float:
        """ The average of the maximum state values accessible when this Item is On.

        "At each time unit, the schema mechanism computes the value explicitly accessible from the current state--that
         is, the maximum value of any items that can be reached by a reliable chain of schemas starting with an
         applicable schema." (See Drescher 1991, p. 63)

        :return: the average accessible value when this Item is On
        """
        return self._dv_avg_accessible_value

    @property
    def delegated_value(self) -> float:
        """ Returns the delegated value for this helper's item.

        (For a discussion of delegated value see Drescher 1991, p. 63.)

        :return: the Item's current delegated value
        """
        return self._dv_avg_accessible_value - GlobalStats().baseline_value

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
        # start trace if item is On in selection state
        if self._item.is_on(state=selection_state, **kwargs):
            self._dv_trace_updates_remaining = GlobalParams().get('dv_trace_max_len')

        # update trace value if updates remain
        if self._dv_trace_updates_remaining > 0:
            self._update_trace_val(result_state)

        # if another On state is encountered during trace, need to terminate previous trace immediately
        immediate_termination = self._item.is_on(state=result_state, **kwargs)
        if immediate_termination:
            self._dv_trace_updates_remaining = 0

        # update avg. accessible value based on trace value on final trace update
        if self._dv_trace_updates_remaining <= 0 and self._dv_trace_value != -np.inf:
            self._update_avg_accessible_val()

    def _update_trace_val(self, state: State) -> None:
        s_pv = state.primitive_value
        s_aav = state.avg_accessible_value

        self._dv_trace_value = np.max([self._dv_trace_value, s_pv, s_aav])
        self._dv_trace_updates_remaining -= 1

    def _update_avg_accessible_val(self) -> None:
        if self._dv_trace_value == -np.inf:
            raise ValueError('invalid trace value: -np.inf.')

        err = self._dv_trace_value - self._dv_avg_accessible_value
        learning_rate = GlobalParams().get('learning_rate')
        self._dv_avg_accessible_value += learning_rate * err

        # clear trace value
        self._dv_trace_value = -np.inf


class Item(Activatable, ValueBearer):

    def __init__(self, source: Any, primitive_value: float = None, **kwargs) -> None:
        super().__init__()

        self._source = source
        self._primitive_value = primitive_value or 0.0
        self._delegated_value_helper = DelegatedValueHelper(item=self)

    @property
    def source(self) -> Any:
        return self._source

    @property
    @abstractmethod
    def state_elements(self) -> set[StateElement]:
        pass

    @property
    def primitive_value(self) -> float:
        return self._primitive_value

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
    def avg_accessible_value(self) -> float:
        return self._delegated_value_helper.avg_accessible_value

    @property
    def delegated_value(self) -> float:
        return self._delegated_value_helper.delegated_value

    def update_delegated_value(self,
                               selection_state: State,
                               result_state: State,
                               **kwargs) -> None:
        """ Updates delegated value based on if item was On in selection and the value of items in the result state.

        :param selection_state: the state from which the last schema was selected (i.e., an action taken)
        :param result_state: the state that immediately followed the provided selection state
        :param kwargs: optional keyword arguments
        :return: None
        """
        self._delegated_value_helper.update(item=self,
                                            selection_state=selection_state,
                                            result_state=result_state,
                                            **kwargs)

    @abstractmethod
    def is_on(self, state: State, **kwargs) -> bool:
        return NotImplemented

    def is_off(self, state: State, **kwargs) -> bool:
        return not self.is_on(state, **kwargs)

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
                               'dv': self.delegated_value,
                               'aav': self.avg_accessible_value, })


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

    # TODO: can this be changed to a overloaded method for source=StateElement and source=StateAssertion using
    # TODO: the @singledispatch decorator?
    def get(self, source: Any, *, item_type: Optional[Type[Item]] = None, **kwargs) -> Optional[Item]:
        read_only = kwargs.get('read_only', False)
        item_type = item_type or GlobalParams().get('item_type')

        type_dict = (
            ItemPool._composite_items
            if item_type is GlobalParams().get('composite_item_type')
            else ItemPool._items
        )

        obj = type_dict.get(source)

        # create new item and add to pool if not found and not read_only
        if not obj and not read_only:
            obj = type_dict[source] = item_type(source, **kwargs)

        return obj


class ReadOnlyItemPool(ItemPool):
    def __init__(self):
        self._pool = ItemPool()

    def get(self, source: Any, item_type: Optional[Type[Item]] = None, **kwargs) -> Item:
        kwargs['read_only'] = True
        return self._pool.get(source, item_type=item_type, **kwargs)

    def clear(self):
        raise NotImplementedError('ReadOnlyItemPool does not support clear operation.')


class SymbolicItem(Item):
    """ A state element that can be thought as a proposition/feature. """

    def __init__(self, source: str, primitive_value: float = None, **kwargs):
        if not isinstance(source, str):
            raise ValueError('Source for symbolic item must be a string')

        super().__init__(source=source, primitive_value=primitive_value, **kwargs)

    @property
    def source(self) -> str:
        return super().source

    @property
    def state_elements(self) -> set[str]:
        return {super().source}

    def is_on(self, state: State, **kwargs) -> bool:
        return self.source in state

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, SymbolicItem):
            return self.source == other.source
        return False if other is None else NotImplemented

    def __hash__(self) -> int:
        return hash(self.source)


def non_composite_items(items: Collection[Item]) -> Collection[Item]:
    return list(filter(lambda i: not isinstance(i, CompositeItem), items))


def composite_items(items: Collection[Item]) -> Collection[Item]:
    return list(filter(lambda i: isinstance(i, CompositeItem), items))


class Verbosity(IntEnum):
    TRACE = auto()
    DEBUG = auto()
    INFO = auto()
    WARN = auto()
    ERROR = auto()
    FATAL = auto()
    NONE = auto()


class SupportedFeature(Enum):
    # "There is an embellishment of the marginal attribution algorithm--deferring to a more specific applicable schema--
    #  that often enables the discovery of an item whose relevance has been obscured." (see Drescher,1991, pp. 75-76)
    EC_DEFER_TO_MORE_SPECIFIC_SCHEMA = auto()

    # "[another] embellishment also reduces redundancy: when a schema's extended context simultaneously detects the
    # relevance of several items--that is, their statistics pass the significance threshold on the same trial--the most
    # specific is chosen as the one for inclusion in a spin-off from that schema." (see Drescher, 1991, p. 77)
    #
    #     Note: Requires that EC_DEFER_TO_MORE_SPECIFIC is also enabled.
    EC_MOST_SPECIFIC_ON_MULTIPLE = auto()

    # "The machinery's sensitivity to results is amplified by an embellishment of marginal attribution: when a given
    #  schema is idle (i.e., it has not just completed an activation), the updating of its extended result data is
    #  suppressed for any state transition which is explained--meaning that the transition is predicted as the result
    #  of a reliable schema whose activation has just completed." (see Drescher, 1991, p. 73)
    ER_SUPPRESS_UPDATE_ON_EXPLAINED = auto()

    # Supports the creation of result spin-off schemas incrementally. This was not supported in the original schema
    # mechanism because of the proliferation of composite results that result. It is allowed here to facilitate
    # comparison and experimentation.
    ER_INCREMENTAL_RESULTS = auto()

    # Modifies the schema mechanism to only create context spin-offs containing positive assertions.
    EC_POSITIVE_ASSERTIONS_ONLY = auto()

    # Modifies the schema mechanism to only create result spin-offs containing positive assertions.
    ER_POSITIVE_ASSERTIONS_ONLY = auto()


class SupportedFeatureValidator(Validator):
    def __call__(self, features: Optional[Collection[SupportedFeature]]) -> None:
        features = set(features)
        for value in features:
            if not isinstance(value, SupportedFeature):
                raise ValueError(f'Unsupported feature: {value}')

        if (SupportedFeature.EC_MOST_SPECIFIC_ON_MULTIPLE in features and
                SupportedFeature.EC_DEFER_TO_MORE_SPECIFIC_SCHEMA not in features):
            raise ValueError(f'The feature EC_MOST_SPECIFIC_ON_MULTIPLE requires EC_DEFER_TO_MORE_SPECIFIC_SCHEMA')


def is_feature_enabled(feature: SupportedFeature) -> bool:
    return feature in GlobalParams().get('features')


class GlobalParams(metaclass=Singleton):

    def __init__(self) -> None:
        self._defaults: dict[str, Any] = dict()
        self._validators: dict[str, Validator] = defaultdict(lambda: NULL_VALIDATOR)

        self._set_validators()
        self._set_defaults()

        self._params: dict[str, Any] = dict(self._defaults)

    @property
    def defaults(self) -> dict[str, Any]:
        return self._defaults

    def set(self, name: str, value: Any) -> None:
        if name not in self._params:
            warn(f'Parameter "{name}" does not exist. Creating new parameter.')

        # raises ValueError if new value is invalid
        self._validators[name](value)

        self._params[name] = value

    def get(self, name: str) -> Any:
        if name not in self._params:
            warn(f'Parameter "{name}" does not exist.')

        return self._params.get(name)

    def reset(self):
        self._params = dict(self._defaults)

    def _set_defaults(self):
        # verbosity used to determine the active print/warn statements
        self._defaults['verbosity'] = Verbosity.WARN

        # format string used by output functions (debug, info, warn, error, fatal)
        self._defaults['output_format'] = '{timestamp} [{severity}]\t{message}'

        # default seed for the random number generator
        self._defaults['rng_seed'] = int(time())

        # determines step size for incremental updates (e.g., this is used for delegated value updates)
        self._defaults['learning_rate'] = 0.01

        # method for determining statistical correlation
        self._defaults['correlation_method'] = FisherExactCorrelationTest()

        # thresholds for determining the relevance of items (1.0 -> correlation always occurs)
        self._defaults['positive_correlation_threshold'] = 0.95
        self._defaults['negative_correlation_threshold'] = 0.95

        # success threshold used for determining that a schema is reliable (1.0 -> schema always succeeds)
        self._defaults['reliability_threshold'] = 0.95

        # used by delegated value helper
        self._defaults['dv_trace_max_len'] = 5

        # schema selection weighting
        self._defaults['goal_weight'] = 0.6
        self._defaults['explore_weight'] = 0.4

        # default features
        self._defaults['features'] = {
            SupportedFeature.EC_DEFER_TO_MORE_SPECIFIC_SCHEMA,
            SupportedFeature.EC_MOST_SPECIFIC_ON_MULTIPLE,
            SupportedFeature.ER_POSITIVE_ASSERTIONS_ONLY,
            SupportedFeature.EC_POSITIVE_ASSERTIONS_ONLY,
            SupportedFeature.ER_SUPPRESS_UPDATE_ON_EXPLAINED,
        }

        # object factories
        self._defaults['schema_type'] = Schema
        self._defaults['item_type'] = SymbolicItem
        self._defaults['composite_item_type'] = CompositeItem

    def _set_validators(self):
        self._validators['features'] = SupportedFeatureValidator()
        self._validators['rng_seed'] = TypeValidator([int])
        self._validators['learning_rate'] = RangeValidator(0.0, 1.0)
        self._validators['correlation_method'] = TypeValidator([ItemCorrelationTest])
        self._validators['positive_correlation_threshold'] = RangeValidator(0.0, 1.0)
        self._validators['negative_correlation_threshold'] = RangeValidator(0.0, 1.0)
        self._validators['reliability_threshold'] = RangeValidator(0.0, 1.0)
        self._validators['goal_weight'] = RangeValidator(0.0, 1.0)
        self._validators['explore_weight'] = RangeValidator(0.0, 1.0)
        self._validators['verbosity'] = TypeValidator([Verbosity])
        self._validators['output_format'] = TypeValidator([str])
        self._validators['item_type'] = SubClassValidator([Item])
        self._validators['composite_item_type'] = SubClassValidator([CompositeItem])
        self._validators['dv_trace_max_len'] = MultiValidator([TypeValidator([int]), RangeValidator(low=0.0)])


class GlobalStats(metaclass=Singleton):
    def __init__(self, baseline_value: Optional[float] = None) -> None:
        self._baseline = baseline_value or 0.0

    @property
    def baseline_value(self) -> float:
        """ A running average of the primitive state value encountered over all visited states.

        :return: the empirical (primitive value) baseline
        """
        return self._baseline

    @baseline_value.setter
    def baseline_value(self, value: float) -> None:
        self._baseline = value

    def update_baseline(self, state: State) -> None:
        """ Updates an unconditional running average of the primitive values of states encountered.

        :param state: a state
        :return: None
        """
        learning_rate = GlobalParams().get('learning_rate')
        self._baseline += learning_rate * (state.primitive_value - self._baseline)

    def reset(self):
        self._baseline = 0.0


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


class ItemCorrelationTest(ABC):

    def positive_corr(self, table: Iterable) -> bool:
        """

        :param table:
        :return:
        """
        return self.positive_corr_statistic(table) >= GlobalParams().get('positive_correlation_threshold')

    def negative_corr(self, table: Iterable) -> bool:
        """

        :param table:
        :return:
        """
        return self.negative_corr_statistic(table) >= GlobalParams().get('negative_correlation_threshold')

    @abstractmethod
    def positive_corr_statistic(self, table: Iterable) -> float:
        pass

    @abstractmethod
    def negative_corr_statistic(self, table: Iterable) -> float:
        pass

    def validate_data(self, data: Iterable) -> np.ndarray:
        """ Raises a ValueError if data cannot be interpreted as a 2x2 array of integers.

        :param data: the iterable to validate
        :return: True if valid table; False otherwise.
        """
        table = np.array(data)
        if not (table.shape == (2, 2) and np.issubdtype(table.dtype, int)):
            raise ValueError('invalid data: must be interpretable as a 2x2 array of integers')
        return table


class DrescherCorrelationTest(ItemCorrelationTest):

    def positive_corr_statistic(self, table: Iterable) -> float:
        """ Returns the part-to-part ratio Pr(A | X) : Pr(A | not X)

        Input data should be a 2x2 table of the form: [[N(A,X), N(not A,X)], [N(A,not X), N(not A,not X)]],
        where N(A,X) is the number of events that are both A AND X

        :return: the ratio as a float, or numpy.NAN if division by zero
        """
        # raises ValueError
        table = self.validate_data(table)

        try:
            n_x = np.sum(table[0, :])
            n_not_x = np.sum(table[1, :])

            n_a_and_x = table[0, 0]
            n_a_and_not_x = table[1, 0]

            if n_x == 0 or n_not_x == 0:
                return 0.0

            # calculate conditional probabilities
            pr_a_given_x = n_a_and_x / n_x
            pr_a_given_not_x = n_a_and_not_x / n_not_x

            pr_a = pr_a_given_x + pr_a_given_not_x

            if pr_a == 0:
                return 0.0

            # the part-to-part ratio Pr(A | X) : Pr(A | not X)
            ratio = pr_a_given_x / pr_a
            return ratio
        except ZeroDivisionError:
            return 0.0

    def negative_corr_statistic(self, table: Iterable) -> float:
        """ Returns the part-to-part ratio Pr(not A | X) : Pr(not A | not X)

        Input data should be a 2x2 table of the form: [[N(A,X), N(not A,X)], [N(A,not X), N(not A,not X)]],
        where N(A,X) is the number of events that are both A AND X

        :return: the ratio as a float, or numpy.NAN if division by zero
        """
        # raises ValueError
        table = self.validate_data(table)

        try:
            n_x = np.sum(table[0, :])
            n_not_x = np.sum(table[1, :])

            n_not_a_and_x = table[0, 1]
            n_not_a_and_not_x = table[1, 1]

            if n_x == 0 or n_not_x == 0:
                return 0.0

            # calculate conditional probabilities
            pr_not_a_given_x = n_not_a_and_x / n_x
            pr_not_a_given_not_x = n_not_a_and_not_x / n_not_x

            pr_not_a = pr_not_a_given_x + pr_not_a_given_not_x

            if pr_not_a == 0:
                return 0.0

            # the part-to-part ratio between Pr(not A | X) : Pr(not A | not X)
            ratio = pr_not_a_given_x / pr_not_a
            return ratio
        except ZeroDivisionError:
            return 0.0


class BarnardExactCorrelationTest(ItemCorrelationTest):

    def positive_corr_statistic(self, table: Iterable) -> float:
        """

        :param table:
        :return:
        """
        # raises ValueError
        table = self.validate_data(table)

        return 1.0 - stats.barnard_exact(table, alternative='greater').pvalue

    def negative_corr_statistic(self, table: Iterable) -> float:
        """

        :param table:
        :return:
        """
        # raises ValueError
        table = self.validate_data(table)

        return 1.0 - stats.barnard_exact(table, alternative='less').pvalue


class FisherExactCorrelationTest(ItemCorrelationTest):

    def positive_corr_statistic(self, table: Iterable) -> float:
        """

        :param table:
        :return:
        """
        # raises ValueError
        table = self.validate_data(table)

        _, p_value = stats.fisher_exact(table, alternative='greater')
        return 1.0 - p_value

    def negative_corr_statistic(self, table: Iterable) -> float:
        """

        :param table:
        :return:
        """
        # raises ValueError
        table = self.validate_data(table)

        _, p_value = stats.fisher_exact(table, alternative='less')
        return 1.0 - p_value


class ItemStats(ABC):
    @abstractmethod
    def as_array(self) -> np.ndarray:
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
            # # calculate conditional probabilities
            # p_success_given_on = self._n_success_and_on / self.n_on
            # p_success_given_off = self._n_success_and_off / self.n_off
            #
            # # the part-to-part ratio between p(success | on) : p(success | off)
            # return p_success_given_on / (p_success_given_on + p_success_given_off)
            correlation_method: ItemCorrelationTest = GlobalParams().get('correlation_method')

            success_corr = correlation_method.positive_corr_statistic(self.as_array())
            return success_corr
        except ZeroDivisionError:
            return np.NAN

    @property
    def failure_corr(self) -> float:
        """ Returns the ratio p(failure | item on) : p(failure | item off)

        :return: the ratio as a float, or numpy.NAN if division by zero
        """
        try:
            # # calculate conditional probabilities
            # p_fail_given_on = self._n_fail_and_on / self.n_on
            # p_fail_given_off = self._n_fail_and_off / self.n_off
            #
            # # the part-to-part ratio between p(failure | on) : p(failure | off)
            # return p_fail_given_on / (p_fail_given_on + p_fail_given_off)
            correlation_method: ItemCorrelationTest = GlobalParams().get('correlation_method')

            failure_corr = correlation_method.negative_corr_statistic(self.as_array())
            return failure_corr
        except ZeroDivisionError:
            return np.NAN

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

    def as_array(self) -> np.ndarray:
        return np.array([
            [self.n_success_and_on, self.n_fail_and_on],
            [self.n_success_and_off, self.n_fail_and_off],
        ])

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

    def __init__(self):
        self._n_on_and_activated = 0
        self._n_on_and_not_activated = 0
        self._n_off_and_activated = 0
        self._n_off_and_not_activated = 0

    @property
    def positive_transition_corr(self) -> float:
        """ Returns the positive-transition correlation for this item.

            "The positive-transition correlation is the ratio of the probability of the slot's item turning On when
            the schema's action has just been taken to the probability of its turning On when the schema's action
            is not being taken." (see Drescher, 1991, p. 71)

        :return: the positive-transition correlation
        """
        try:
            correlation_method: ItemCorrelationTest = GlobalParams().get('correlation_method')
            return correlation_method.positive_corr_statistic(self.as_array())
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
            correlation_method: ItemCorrelationTest = GlobalParams().get('correlation_method')
            return correlation_method.negative_corr_statistic(self.as_array())
        except ZeroDivisionError:
            return np.NAN

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

    def as_array(self) -> np.ndarray:
        return np.array([
            [self.n_on_and_activated, self.n_off_and_activated],
            [self.n_on_and_not_activated, self.n_off_and_not_activated]
        ])

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
    def update(self, **kwargs):
        raise NotImplementedError('Update not implemented for readonly view.')


class ReadOnlyECItemStats(ECItemStats):
    def update(self, on: bool, success: bool, count: int = 1) -> None:
        raise NotImplementedError('Update not implemented for readonly view.')


class ReadOnlyERItemStats(ERItemStats):
    def update(self, on: bool, activated: bool, count: int = 1) -> None:
        raise NotImplementedError('Update not implemented for readonly view.')


# A single immutable object that is meant to be used for all item instances that have never had stats updates
NULL_SCHEMA_STATS = ReadOnlySchemaStats()
NULL_EC_ITEM_STATS = ReadOnlyECItemStats()
NULL_ER_ITEM_STATS = ReadOnlyERItemStats()


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


class CompositeAction(Action):
    """

    """

    class Controller:
        def __init__(self):
            self._components: Collection[Schema] = set()
            self._goal_proximity: dict[Schema, int] = defaultdict(lambda: np.inf)

        @property
        def components(self) -> Collection[Schema]:
            return self._components

        def update(self) -> None:
            pass

    def __init__(self, goal_state: StateAssertion, **kwargs):
        super().__init__(**kwargs)

        if not goal_state:
            raise ValueError('Goal state is not optional.')

        self._goal_state = goal_state
        self._controller = CompositeAction.Controller()

    @property
    def controller(self) -> CompositeAction.Controller:
        return self._controller

    def is_enabled(self, state: State) -> bool:
        return any({schema.is_applicable(state) for schema in self._controller.components})


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

    @property
    @abstractmethod
    def items(self) -> frozenset[Item]:
        pass

    @staticmethod
    def replicate_with(old: Assertion, new: Assertion) -> StateAssertion:
        for ia in new:
            if ia in old:
                raise ValueError('New assertion already exists in old assertion')

        return StateAssertion(asserts=(*old, *new))


class ItemAssertion(Assertion, ValueBearer):

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

    @property
    def items(self) -> frozenset[Item]:
        return frozenset([self._item])

    def is_satisfied(self, state: State, **kwargs) -> bool:
        if self.is_negated:
            return self._item.is_off(state, **kwargs)
        else:
            return self._item.is_on(state, **kwargs)

    # TODO: It's not clear how to handle negative assertions. Simply subtracting the values of those items seems
    # TODO: incorrect, as the thing that it is not may actually be more valuable. The safest course for now seems to be
    # TODO: to exclude negated asserts from these calculations.
    @property
    def primitive_value(self):
        return 0.0 if self.is_negated else self._item.primitive_value

    @property
    def delegated_value(self):
        return 0.0 if self.is_negated else self._item.delegated_value

    @property
    def avg_accessible_value(self) -> float:
        return 0.0 if self.is_negated else self._item.avg_accessible_value


class StateAssertion(Assertion, ValueBearer):

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
        return State(set(itertools.chain.from_iterable([ia.item.state_elements for ia in self.asserts])))

    @staticmethod
    def from_state(state: State) -> StateAssertion:
        """ Factory method for creating state assertions that would be satisfied by the given State.

        :param state: a State
        :return: a StateAssertion
        """
        return StateAssertion(asserts=[ItemAssertion(ReadOnlyItemPool().get(se)) for se in state])

    # TODO: It's not clear how to handle negative assertions. Simply subtracting the values of those items seems
    # TODO: incorrect, as the thing that it is not may actually be more valuable. The safest course for now seems to be
    # TODO: to exclude negated asserts from these calculations.
    @property
    def primitive_value(self):
        return sum(ia.item.primitive_value for ia in self._asserts)

    @property
    def delegated_value(self):
        return sum(ia.item.delegated_value for ia in self._asserts)

    @property
    def avg_accessible_value(self) -> float:
        return sum(ia.item.avg_accessible_value for ia in self._asserts)


NULL_STATE_ASSERT = StateAssertion()


# TODO: There may be a subtle bug here when the StateAssertion is negated and the CompositeItem containing
# TODO: that negated assertion is included in a negated ItemAssertion or StateAssertion.s
class CompositeItem(StateAssertion, Item):
    """ A StateAssertion wrapper that functions like an Item.

    This class is primarily used to support ExtendedResult statistics when the ER_INCREMENTAL_RESULTS is disabled.
    """

    def __init__(self, source: StateAssertion, **kwargs) -> None:
        if len(source) < 2:
            raise ValueError('Source assertion must have at least two elements')

        super().__init__(source=source,
                         primitive_value=source.primitive_value,
                         asserts=[*source.asserts, *source.negated_asserts])

    @property
    def source(self) -> StateAssertion:
        return super().source

    @property
    def state_elements(self) -> set[StateElement]:
        return set(itertools.chain.from_iterable([ia.item.state_elements for ia in self.source.asserts]))

    def is_on(self, state: State, **kwargs) -> bool:
        return self.source.is_satisfied(state)

    def __contains__(self, assertion: Assertion) -> bool:
        if isinstance(assertion, ItemAssertion):
            return assertion in self.source
        return False

    def __eq__(self, other) -> bool:
        return super().__eq__(other)
        # if isinstance(other, Item):
        #     return self.source == other.source
        # if isinstance(other, StateAssertion):
        #     return super().__eq__()
        # return False if other is None else NotImplemented

    def __hash__(self) -> int:
        return hash(self.source)

    def __str__(self) -> str:
        return f'({str(self.source)})'


class ExtendedItemCollection(Observable):
    def __init__(self, suppressed_items: Collection[Item] = None, null_member: ItemStats = None):
        super().__init__()

        self._null_member = null_member

        self._suppressed_items: frozenset[Item] = frozenset(suppressed_items) or frozenset()

        # TODO: these names are confusing... they are not items!
        self._relevant_items: MutableSet[Assertion] = set()
        self._new_relevant_items: MutableSet[Assertion] = set()

        self._stats: dict[Any, ItemStats] = defaultdict(lambda: self._null_member)

        self._item_pool = ItemPool()

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

    def update_relevant_items(self, assertion: Assertion, suppressed: bool = False):
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

    @property
    def correlation_method(self) -> ItemCorrelationTest:
        return GlobalParams().get('correlation_method')

    @property
    def positive_correlation_threshold(self) -> float:
        return GlobalParams().get('positive_correlation_threshold')

    @property
    def negative_correlation_threshold(self) -> float:
        return GlobalParams().get('negative_correlation_threshold')

    # TODO: this is an ugly design! there must be a better way....
    @property
    def new_relevant_items(self) -> frozenset[Assertion]:
        return frozenset(self._new_relevant_items)

    @new_relevant_items.setter
    def new_relevant_items(self, value: Collection[Assertion]) -> None:
        self._new_relevant_items = frozenset(value)


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

    # TODO: Try to optimize this. The vast majority of the items in each extended context should have identical
    # TODO: statistics.
    def update_all(self, activated: bool, new: Collection[Item], lost: Collection[Item],
                   count: int = 1) -> None:

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
        if item_stats.positive_transition_corr > self.positive_correlation_threshold:
            item_assert = ItemAssertion(item)
            if item_assert not in self.relevant_items:
                self.update_relevant_items(item_assert)

        elif item_stats.negative_transition_corr > self.negative_correlation_threshold:
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

    # TODO: Try to optimize this. The vast majority of the items in each extended context should have identical
    # TODO: statistics.
    def update_all(self, state: State, success: bool, count: int = 1) -> None:
        # bypass updates when a more specific spinoff schema exists
        if is_feature_enabled(SupportedFeature.EC_DEFER_TO_MORE_SPECIFIC_SCHEMA) and self.defer_update_to_spin_offs(
                state):
            return

        for item in self._item_pool.items:
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
                debug(f'deferring update to spin_off for state {state} due to relevant assertion {ia}')
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

        # TODO: The check against thresholds should be done in the correlation test classes
        # if item is relevant, a new item assertion is created
        item_assert = (
            ItemAssertion(item) if item_stats.success_corr > self.positive_correlation_threshold else
            ItemAssertion(item, negated=True) if item_stats.failure_corr > self.negative_correlation_threshold else
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

        self._is_primitive = not (context or result)

        self._extended_context: ExtendedContext = (
            None
            if self._is_primitive else
            ExtendedContext(self._context)
        )

        self._extended_result: ExtendedResult = (
            ExtendedResult(self._result)
            if self._is_primitive or is_feature_enabled(SupportedFeature.ER_INCREMENTAL_RESULTS) else
            None
        )

        # TODO: Need to update overriding conditions.
        self._overriding_conditions: Optional[StateAssertion] = None

        # This observer registration is used to notify the schema when a relevant item has been detected in its
        # extended context or extended result
        if not self._is_primitive:
            self._extended_context.register(self)

        if self._is_primitive or is_feature_enabled(SupportedFeature.ER_INCREMENTAL_RESULTS):
            self._extended_result.register(self)

        # TODO: Is duration or cost needed?

        # The duration is the average time from the activation to the completion of an action.
        # self.duration = Schema.INITIAL_DURATION

        # TODO: This may be necessary to properly balance delegated and instrumental values against the cost of
        # TODO: obtaining that future reward. (An alternative might be to use discounted eligibility traces in the
        # TODO: delegated value calculations.)
        # The cost is the minimum (i.e., the greatest magnitude) of any negative-valued results
        # of schemas that are implicitly activated as a side effect of the given schema’s [explicit]
        # activation on that occasion (see Drescher, 1991, p.55).
        # self.cost = Schema.INITIAL_COST

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
            np.NAN if self.is_primitive() or self.stats.n_activated == 0
            else self.stats.n_success / self.stats.n_activated
        )

    @property
    def stats(self) -> SchemaStats:
        return self._stats

    def is_applicable(self, state: State, **kwargs) -> bool:
        """ A schema is applicable when its context is satisfied and there are no active overriding conditions.

            “A schema is said to be applicable when its context is satisfied and no
                 known overriding conditions obtain.” (Drescher, 1991, p.53)

        :param state: the agent's current state
        :param kwargs: optional keyword arguments
        :return: True if the schema is applicable in this state; False, otherwise.
        """
        overridden = False
        if self.overriding_conditions is not None:
            overridden = self.overriding_conditions.is_satisfied(state, **kwargs)
        return (not overridden) and self.context.is_satisfied(state, **kwargs)

    def is_primitive(self) -> bool:
        """ Returns whether this instance is a primitive (action-only) schema.

        :return: True if this is a primitive schema; False otherwise.
        """
        return self._is_primitive

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
               s_prev: Optional[State],
               s_curr: State,
               new: Collection[Item] = None,
               lost: Collection[Item] = None,
               explained: Optional[bool] = None,
               count=1) -> None:
        """

            Note: As a result of these updates, one or more notifications may be generated by the schema's
            context and/or result. These will be received via schema's 'receive' method.

        :param activated: True if this schema was implicitly or explicitly activated
        :param s_prev: the previous state
        :param s_curr: the current state
        :param new: the state elements in current but not previous state
        :param lost: the state elements in previous but not current state
        :param explained: True if a reliable schema was activated that "explained" the last state transition
        :param count: the number of updates to perform

        :return: None
        """

        # True if this schema was activated AND its result obtained; False otherwise
        success: bool = activated and self.result.is_satisfied(s_curr)

        # update top-level stats
        self._stats.update(activated=activated, success=success, count=count)

        # update extended result stats
        if self._extended_result:
            if is_feature_enabled(SupportedFeature.ER_SUPPRESS_UPDATE_ON_EXPLAINED) and explained:
                debug(f'update suppressed for schema {self} because its result was explained by a reliable schema')
            else:
                self._extended_result.update_all(activated=activated, new=new, lost=lost, count=count)

        # update extended context stats
        if all((self._extended_context, activated, s_prev)):
            self._extended_context.update_all(state=s_prev, success=success, count=count)

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

    def __init__(self, primitives: Collection[Schema]) -> None:
        if not primitives:
            raise ValueError('SchemaTree must be initialized with a collection of primitive schemas.')

        self._root = SchemaTreeNode(label='root')
        self._root.schemas_satisfied_by.update(primitives)

        self._nodes: dict[StateAssertion, SchemaTreeNode] = dict()
        self._nodes[self._root.context] = self._root

        self._n_schemas = len(primitives)

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
        matches: MutableSet[SchemaTreeNode] = set()

        nodes_to_process = [self._root]
        while nodes_to_process:
            node = nodes_to_process.pop()
            if node.context.is_satisfied(state, **kwargs):
                matches.add(node)
                if node.children:
                    nodes_to_process += node.children

        return matches

    def find_all_would_satisfy(self, assertion: StateAssertion, **kwargs) -> Collection[SchemaTreeNode]:
        matches: MutableSet[SchemaTreeNode] = set()

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


def _output_fd(level: Verbosity) -> TextIO:
    return sys.stdout if level < Verbosity.WARN else sys.stderr


def _timestamp() -> str:
    return datetime.now().isoformat()


def display_message(message: str, level: Verbosity) -> None:
    verbosity = GlobalParams().get('verbosity')
    output_format = GlobalParams().get('output_format')

    if level >= verbosity:
        out = output_format.format(timestamp=_timestamp(), severity=level.name, message=message)
        print(out, file=_output_fd(level), flush=True)


def trace(message):
    display_message(message=message, level=Verbosity.TRACE)


def debug(message):
    display_message(message=message, level=Verbosity.DEBUG)


def info(message):
    display_message(message=message, level=Verbosity.INFO)


def warn(message):
    display_message(message=message, level=Verbosity.WARN)


def error(message):
    display_message(message=message, level=Verbosity.ERROR)


def fatal(message):
    display_message(message=message, level=Verbosity.FATAL)


_rng = None
_seed = None


def rng():
    global _rng
    global _seed

    new_seed = GlobalParams().get('rng_seed')
    if new_seed != _seed:
        warn(f'(Re-)initializing random number generator using seed="{new_seed}".')
        warn(f'For reproducibility, you should also set "PYTHONHASHSEED={new_seed}" in your environment variables.')

        # setting globals
        _rng = np.random.default_rng(new_seed)
        _seed = new_seed

    return _rng
