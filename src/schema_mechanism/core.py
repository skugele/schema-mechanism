from __future__ import annotations

import itertools
import sys
from abc import ABC
from abc import abstractmethod
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
from typing import Any
from typing import Optional
from typing import TextIO
from typing import Type
from typing import Union

import numpy as np
from anytree import AsciiStyle
from anytree import LevelOrderIter
from anytree import NodeMixin
from anytree import RenderTree

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
class Activatable(ABC):
    @abstractmethod
    def is_on(self, state: State, *args, **kwargs) -> bool: ...

    def is_off(self, state: State, *args, **kwargs) -> bool:
        return not self.is_on(state, *args, **kwargs)


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
               *args, **kwargs) -> None:
        """ Updates delegated-value-related statistics based on the selection and result states.

        :param selection_state: the state from which the last schema was selected (i.e., an action taken)
        :param result_state: the state that immediately followed the provided selection state
        :param args: optional positional arguments
        :param kwargs: optional keyword arguments
        :return: None
        """
        # start trace if item is On in selection state
        if self._item.is_on(state=selection_state, *args, **kwargs):
            self._dv_trace_updates_remaining = GlobalParams().dv_trace_max_len

        # update trace value if updates remain
        if self._dv_trace_updates_remaining > 0:
            self._update_trace_val(result_state)

        # if another On state is encountered during trace, need to terminate previous trace immediately
        immediate_termination = self._item.is_on(state=result_state, *args, **kwargs)
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
        self._dv_avg_accessible_value += GlobalParams().learn_rate * err

        # clear trace value
        self._dv_trace_value = -np.inf


class Item(Activatable, ValueBearer):

    def __init__(self, source: Any, primitive_value: float = None, *args, **kwargs) -> None:
        self._source = source
        self._primitive_value = primitive_value or 0.0
        self._delegated_value_helper = DelegatedValueHelper(item=self)

    @property
    def source(self) -> Any:
        return self._source

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
                               *args,
                               **kwargs) -> None:
        """ Updates delegated value based on if item was On in selection and the value of items in the result state.

        :param selection_state: the state from which the last schema was selected (i.e., an action taken)
        :param result_state: the state that immediately followed the provided selection state
        :param args: optional positional arguments
        :param kwargs: optional keyword arguments
        :return: None
        """
        self._delegated_value_helper.update(item=self,
                                            selection_state=selection_state,
                                            result_state=result_state,
                                            *args, **kwargs)

    @abstractmethod
    def is_on(self, state: State, *args, **kwargs) -> bool:
        return NotImplemented

    def is_off(self, state: State, *args, **kwargs) -> bool:
        return not self.is_on(state, *args, **kwargs)

    @abstractmethod
    def copy(self) -> Item:
        return NotImplemented

    # TODO: Need to be really careful with the default hash implementations which produce different values between
    # TODO: runs. This will kill and direct serialization/deserialization of data structures that rely on hashes.
    @abstractmethod
    def __hash__(self) -> int:
        pass


# TODO: rename to ItemFactory???
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

    def get(self, source: Any, *, item_type: Optional[Type[Item]] = None, **kwargs) -> Optional[Item]:
        read_only = kwargs.get('read_only', False)
        item_type = item_type or GlobalParams().item_type

        type_dict = (
            ItemPool._composite_items
            if item_type is GlobalParams().composite_item_type
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


# class ItemPoolStateView:
#     def __init__(self, state: Optional[State]):
#         # The state to which this view corresponds
#         self._state = state
#         self._on_items = set([item for item in ReadOnlyItemPool() if item.is_on(state)]) if state else set()
#
#     @property
#     def state(self):
#         return self._state
#
#     def is_on(self, item: Item) -> bool:
#         return item in self._on_items
#
#     def is_off(self, item: Item) -> bool:
#         return item not in self._on_items


class SymbolicItem(Item):
    """ A state element that can be thought as a proposition/feature. """

    def __init__(self, source: str, primitive_value: float = None, *args, **kwargs):
        if not isinstance(source, str):
            raise ValueError('Source for symbolic item must be a string')

        super().__init__(source=source, primitive_value=primitive_value, *args, **kwargs)

    @property
    def source(self) -> str:
        return super().source

    def is_on(self, state: State, *args, **kwargs) -> bool:
        return self.source in state

    def copy(self) -> Item:
        return SymbolicItem(self.source)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, SymbolicItem):
            return self.source == other.source
        return False if other is None else NotImplemented

    def __hash__(self) -> int:
        return hash(self.source)

    def __str__(self) -> str:
        return str(self.source)

    def __repr__(self) -> str:
        return repr_str(self, {'source': str(self.source),
                               'pv': self.primitive_value,
                               'dv': self.delegated_value,
                               'aav': self.avg_accessible_value, })


# TODO: There may be a subtle bug here when the StateAssertion is negated and the CompositeItem containing
# TODO: that negated assertion is included in a negated ItemAssertion or StateAssertion.s
class CompositeItem(Item, ValueBearer):
    """ A StateAssertion wrapper that functions like an Item.

    This class is primarily used to support ExtendedResult statistics when the ER_INCREMENTAL_RESULTS is disabled.
    """

    def __init__(self, source: StateAssertion) -> None:
        if len(source) < 2:
            raise ValueError('Source assertion must have at least two elements')

        super().__init__(source=source, primitive_value=source.primitive_value)

    @property
    def source(self) -> StateAssertion:
        return super().source

    def is_on(self, state: State, *args, **kwargs) -> bool:
        return self.source.is_satisfied(state)

    def copy(self) -> Item:
        return CompositeItem(self.source.copy())

    def __contains__(self, assertion: Assertion) -> bool:
        if isinstance(assertion, ItemAssertion):
            return assertion in self.source
        return False

    def __eq__(self, other) -> bool:
        if isinstance(other, CompositeItem):
            return self.source == other.source
        return False if other is None else NotImplemented

    def __hash__(self) -> int:
        return hash(self.source)

    def __str__(self) -> str:
        return f'{"~" if self.source.is_negated else ""}({",".join(str(ia) for ia in self.source)})'

    def __repr__(self) -> str:
        return repr_str(self, {'assertion': self.source, 'negated': self.source.is_negated})


def non_composite_items(items: Collection[Item]) -> Collection[Item]:
    return list(filter(lambda i: not isinstance(i, CompositeItem), items))


def composite_items(items: Collection[Item]) -> Collection[Item]:
    return list(filter(lambda i: isinstance(i, CompositeItem), items))


class Verbosity(IntEnum):
    DEBUG = 0
    INFO = 1
    WARN = 2
    ERROR = 3
    FATAL = 4
    NONE = 5


class GlobalOption(Enum):
    # "There is an embellishment of the marginal attribution algorithm--deferring to a more specific applicable schema--
    #  that often enables the discovery of an item who relevance has been obscured." (see Drescher,1991, pp. 75-76)
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


class GlobalParams(metaclass=Singleton):
    # verbosity used to determine the active print/warn statements
    DEFAULT_VERBOSITY = Verbosity.WARN

    # format string used by output functions (debug, info, warn, error, fatal)
    DEFAULT_OUTPUT_FORMAT = '{timestamp} [{severity}] - "{message}"'

    # determines step size for incremental updates (e.g., this is used for delegated value updates)
    DEFAULT_LEARN_RATE = 0.01

    # thresholds for determining the relevance of items (1.0 -> correlation always occurs)
    DEFAULT_POS_CORR_THRESHOLD = 0.75
    DEFAULT_NEG_CORR_THRESHOLD = 0.75

    # success threshold used for determining that a schema is reliable (1.0 -> schema always succeeds)
    DEFAULT_RELIABILITY_THRESHOLD = 0.95

    # TODO: What default options make sense?
    DEFAULT_OPTIONS = frozenset((

    ))

    # used by ItemFactory
    DEFAULT_ITEM_TYPE = SymbolicItem
    DEFAULT_COMPOSITE_ITEM_TYPE = CompositeItem

    # used by delegated value helper
    DEFAULT_DV_TRACE_MAX_LEN = 5

    def __init__(self) -> None:
        self._verbosity: Verbosity = Verbosity.NONE
        self._output_format: str = str()
        self._learn_rate: float = float()
        self._pos_corr_threshold: float = float()
        self._neg_corr_threshold: float = float()
        self._reliability: float = float()
        self._item_type: Type[Item] = Item
        self._composite_item_type: Type[CompositeItem] = CompositeItem
        self._dv_trace_max_len: int = int()
        self._options: Optional[MutableSet[GlobalOption]] = set()

        self._set_default_values()

    @property
    def learn_rate(self) -> float:
        return self._learn_rate

    @learn_rate.setter
    def learn_rate(self, value: float) -> None:
        if value < 0.0 or value > 1.0:
            raise ValueError('Learning rate must be >= zero and <= to one.')
        self._learn_rate = value

    @property
    def pos_corr_threshold(self) -> float:
        """ Gets the positive correlation threshold for relevance.

        :return: a float between 0.0 and 1.0 (inclusive)
        """
        return self._pos_corr_threshold

    @pos_corr_threshold.setter
    def pos_corr_threshold(self, value: float) -> None:
        """ Sets the positive correlation threshold for relevance.

        :param value: the new positive correlation threshold--a float between 0.0 and 1.0 (inclusive)
        :return: None
        """
        if value < 0.0 or value > 1.0:
            raise ValueError('positive correlation threshold must be between zero and one (inclusive).')
        self._pos_corr_threshold = value

    @property
    def neg_corr_threshold(self) -> float:
        """ Gets the negative correlation threshold for relevance.

        :return: a float between 0.0 and 1.0 (inclusive)
        """
        return self._pos_corr_threshold

    @neg_corr_threshold.setter
    def neg_corr_threshold(self, value: float) -> None:
        """ Sets the negative correlation threshold for relevance.

        :param value: the new negative correlation threshold--a float between 0.0 and 1.0 (inclusive)
        :return: None
        """
        if value < 0.0 or value > 1.0:
            raise ValueError('negative correlation threshold must be between zero and one (inclusive).')
        self._pos_corr_threshold = value

    @property
    def reliability_threshold(self) -> float:
        """ Gets the threshold for schema reliability.

        :return: a float between 0.0 and 1.0 (inclusive)
        """
        return self._reliability

    @reliability_threshold.setter
    def reliability_threshold(self, value: float) -> None:
        """ Sets the threshold for schema reliability.

        :param value: the new schema reliability threshold--a float between 0.0 and 1.0 (inclusive)
        :return: None
        """
        if value < 0.0 or value > 1.0:
            raise ValueError('schema reliability threshold must be between zero and one (inclusive).')
        self._pos_corr_threshold = value

    @property
    def verbosity(self) -> Verbosity:
        """ Gets the current output verbosity.

        :return: a Verbosity value
        """
        return self._verbosity

    @verbosity.setter
    def verbosity(self, value: Verbosity) -> None:
        """ Sets the output verbosity.

        :param value: the new output verbosity
        :return: None
        """
        if value not in Verbosity:
            raise ValueError(f'Invalid verbosity. Supported values include: {[v for v in Verbosity]}')
        self._verbosity = value

    @property
    def output_format(self) -> str:
        """ Gets the format string used for diagnostic and informational messages.

        :return: a Verbosity value
        """
        return self._output_format

    @output_format.setter
    def output_format(self, value: Verbosity) -> None:
        """ Sets the format string used for diagnostic and informational messages.

        Supported variables in format include: {message}, {timestamp}, and {severity}.

        :param value: the new format string
        :return: None
        """
        self._output_format = value

    @property
    def item_type(self) -> Type[Item]:
        """ Gets 5he Item type used for non-composite Item creation.

        :return: an Item type
        """
        return self._item_type

    @item_type.setter
    def item_type(self, value: Type[Item]) -> None:
        """ Sets the Item type used for non-composite Item creation.

        :param value: an Item type
        :return: None
        """
        self._item_type = value

    @property
    def composite_item_type(self) -> Type[CompositeItem]:
        """ Gets 5he Item type used for non-composite Item creation.

        :return: an Item type
        """
        return self._composite_item_type

    @composite_item_type.setter
    def composite_item_type(self, value: Type[CompositeItem]) -> None:
        """ Sets the Item type used for non-composite Item creation.

        :param value: an Item type
        :return: None
        """
        self._composite_item_type = value

    @property
    def options(self) -> MutableSet[GlobalOption]:
        return self._options

    @options.setter
    def options(self, value: Optional[Collection[GlobalOption]]) -> None:
        self._options = self._set_options(value)

    @property
    def dv_trace_max_len(self) -> int:
        """ Gets the maximum trace length for delegated value calculations

        :return: a positive int
        """
        return self._dv_trace_max_len

    @dv_trace_max_len.setter
    def dv_trace_max_len(self, value: int) -> None:
        """ Sets the maximum trace length for delegated value calculations

        :param value: a positive int
        :return: None
        """
        if value <= 0:
            raise ValueError('the max delegated value trace length must be positive')
        self._dv_trace_max_len = value

    def _set_options(self, enhancements: Optional[Collection[GlobalOption]]) -> MutableSet[GlobalOption]:
        options = set(enhancements)
        if GlobalOption.EC_MOST_SPECIFIC_ON_MULTIPLE in enhancements:
            if GlobalOption.EC_DEFER_TO_MORE_SPECIFIC_SCHEMA not in enhancements:
                warn('Option EC_MOST_SPECIFIC_ON_MULTIPLE requires EC_DEFER_TO_MORE_SPECIFIC!')
                warn('Option EC_DEFER_TO_MORE_SPECIFIC_SCHEMA was added automatically.')
                options.add(GlobalOption.EC_DEFER_TO_MORE_SPECIFIC_SCHEMA)

        return options

    def is_enabled(self, option: GlobalOption) -> bool:
        return option in self._options

    def reset(self):
        self._set_default_values()

    def _set_default_values(self) -> None:
        self.verbosity = GlobalParams.DEFAULT_VERBOSITY
        self.output_format = GlobalParams.DEFAULT_OUTPUT_FORMAT
        self.learn_rate = GlobalParams.DEFAULT_LEARN_RATE
        self.pos_corr_threshold = GlobalParams.DEFAULT_POS_CORR_THRESHOLD
        self.neg_corr_threshold = GlobalParams.DEFAULT_NEG_CORR_THRESHOLD
        self.reliability_threshold = GlobalParams.DEFAULT_RELIABILITY_THRESHOLD
        self.dv_trace_max_len = GlobalParams.DEFAULT_DV_TRACE_MAX_LEN
        self.item_type = GlobalParams.DEFAULT_ITEM_TYPE
        self.composite_item_type = GlobalParams.DEFAULT_COMPOSITE_ITEM_TYPE
        self.options = GlobalParams.DEFAULT_OPTIONS


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
        """ Updates an unconditional running average of the primitive values of states.

        :param state: a state
        :return: None
        """
        self._baseline += GlobalParams().learn_rate * (state.primitive_value - self._baseline)

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


class ItemStats:
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

    def copy(self) -> ECItemStats:
        new = ECItemStats()

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

    def __init__(self):
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
            p_on_given_activated = self.n_on_and_activated / self.n_activated
            p_on_given_not_activated = self.n_on_and_not_activated / self.n_not_activated

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
            p_off_given_activated = self.n_off_and_activated / self.n_activated
            p_off_given_not_activated = self.n_off_and_not_activated / self.n_not_activated

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

    def copy(self) -> ERItemStats:
        new = ERItemStats()

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
NULL_EC_ITEM_STATS = ReadOnlyECItemStats()
NULL_ER_ITEM_STATS = ReadOnlyERItemStats()


class Action(UniqueIdMixin):
    _last_uid: int = 0

    def __init__(self, label: Optional[str] = None):
        super().__init__()

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

    def copy(self) -> Action:
        # bypasses initializer to force reuse of uid
        copy = super().__new__(Action)
        copy._uid = self._uid
        copy._label = self._label

        return copy


# TODO: Implement composite actions.
class CompositeAction:
    """

     “A composite action is considered to have been implicitly taken whenever its goal state becomes satisfied--that
     is, makes a transition from Off to On--even if that composite action was never initiated by an activated schema.
     Marginal attribution can thereby detect results caused by the goal state, even if the goal state obtains due
     to external events.” (See Drescher, 1991, p. 91)

    """
    pass


class Assertion(ABC):
    def __init__(self, negated: bool) -> None:
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

    def __init__(self, item: Item, negated: bool = False) -> None:
        super().__init__(negated)

        self._item = item

    def __iter__(self) -> Iterator[Assertion]:
        yield from [self]

    def __len__(self) -> int:
        return 1

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
        return self._item

    @property
    def items(self) -> frozenset[Item]:
        return frozenset([self._item])

    def is_satisfied(self, state: State, *args, **kwargs) -> bool:
        if self.is_negated:
            return self._item.is_off(state, *args, **kwargs)
        else:
            return self._item.is_on(state, *args, **kwargs)

    # TODO: Can this be removed???
    def copy(self) -> ItemAssertion:
        """ Performs a shallow copy of this ItemAssertion. """
        return ItemAssertion(item=self._item, negated=self.is_negated)

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

    def __init__(self, asserts: Optional[Collection[ItemAssertion, ...]] = None, negated: bool = False):
        super().__init__(negated)

        self._asserts = frozenset(filter(lambda ia: not ia.is_negated, asserts)) if asserts else frozenset()
        self._neg_asserts = frozenset(filter(lambda ia: ia.is_negated, asserts)) if asserts else frozenset()

        self._items = frozenset(itertools.chain.from_iterable(a.items for a in asserts)) if asserts else frozenset()

    def __iter__(self) -> Iterator[ItemAssertion]:
        yield from self._asserts
        yield from self._neg_asserts

    def __len__(self) -> int:
        return len(self._asserts) + len(self._neg_asserts)

    def __contains__(self, item_assert: ItemAssertion) -> bool:
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
        return ','.join(map(str, self))

    def __repr__(self) -> str:
        return repr_str(self, {'asserts': str(self)})

    @property
    def items(self) -> frozenset[Item]:
        return self._items

    def copy(self) -> StateAssertion:
        """ Returns a shallow copy of this object.

        :return: the copy
        """
        new = super().__new__(StateAssertion)

        new._asserts = self._asserts
        new._neg_asserts = self._neg_asserts

        return new

    def is_satisfied(self, state: State, *args, **kwargs) -> bool:
        """ Satisfied when all non-negated items are On, and all negated items are Off.

        :param state: the agent's current state
        :param args: optional positional arguments
        :param kwargs: optional keyword arguments
        :return: True if this state assertion is satisfied given the current state; False otherwise.
        """
        return all({ia.is_satisfied(state, *args, **kwargs) for ia in self})

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


class ExtendedItemCollection(Observable):
    def __init__(self, suppressed_items: Collection[Item] = None, null_member: ItemStats = None):
        super().__init__()

        self._null_member = null_member

        # TODO: why is this not a set like the relevant item containers???
        self._suppressed_items: list[Item] = list(suppressed_items) or list()

        # TODO: these names are confusing... they are not items!
        self._relevant_items: MutableSet[Assertion] = set()
        self._new_relevant_items: MutableSet[Assertion] = set()

        self._stats: dict[Any, ItemStats] = defaultdict(lambda: self._null_member)

        self._item_pool = ItemPool()

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

    @property
    def stats(self) -> dict[Any, Any]:
        return self._stats

    @property
    def suppressed_items(self) -> Collection[Item]:
        return self._suppressed_items

    def update_suppressed_items(self, item) -> None:
        if item not in self._suppressed_items:
            self._suppressed_items.append(item)

    # TODO: "relevant items" is confusing. these are item assertion. this terminology is a carry-over from Drescher.
    @property
    def relevant_items(self) -> frozenset[Assertion]:
        return frozenset(self._relevant_items)

    def update_relevant_items(self, assertion: Assertion):
        if assertion not in self._relevant_items:
            self._relevant_items.add(assertion)
            self._new_relevant_items.add(assertion)

    def notify_all(self, *args, **kwargs) -> None:
        if 'source' not in kwargs:
            kwargs['source'] = self

        super().notify_all(*args, **kwargs)

        # clears the set
        self._new_relevant_items = set()

    # TODO: this is an ugly design! there must be a better way....
    @property
    def new_relevant_items(self) -> frozenset[Assertion]:
        return frozenset(self._new_relevant_items)

    @new_relevant_items.setter
    def new_relevant_items(self, value: Collection[Assertion]) -> None:
        self._new_relevant_items = frozenset(value)


class AssertionRegistry(metaclass=Singleton):
    """ Stores learned assertion.

    Note: this class is primarily used by extended results to store learned contexts. It supports the creation of
    "composite" result spin-off schemas. (See Drescher 1991, Sec. 4.1.4 for details on result conjunctions.)

    """

    def __init__(self) -> None:
        self._assertions: MutableSet[Assertion] = set()

    def __len__(self) -> int:
        return len(self._assertions)

    def __iter__(self) -> Iterator[Assertion]:
        yield from self._assertions

    def __contains__(self, assertion: Assertion) -> bool:
        return assertion in self._assertions

    def add(self, assertion: Assertion) -> None:
        self._assertions.add(assertion)


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
            choins of schemas
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
        if item_stats.positive_transition_corr > GlobalParams().pos_corr_threshold:
            item_assert = ItemAssertion(item)
            if item_assert not in self.relevant_items:
                self.update_relevant_items(item_assert)

        elif item_stats.negative_transition_corr > GlobalParams().neg_corr_threshold:
            item_assert = ItemAssertion(item, negated=True)
            if item_assert not in self.relevant_items:
                self.update_relevant_items(item_assert)

    def update_relevant_items(self, assertion: Assertion):
        # TODO: need to guarantee that only item assertions are being added here....
        if GlobalParams().is_enabled(GlobalOption.ER_POSITIVE_ASSERTIONS_ONLY) and assertion.is_negated:
            # TODO: need to verify that this is a SINGLE item or composite item!
            self.update_suppressed_items(assertion.items)
        else:
            super().update_relevant_items(assertion)


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
        if (GlobalParams().is_enabled(GlobalOption.EC_DEFER_TO_MORE_SPECIFIC_SCHEMA)
                and self.defer_update_to_spin_offs(state)):
            return

        for item in self._item_pool.items:
            self.update(item=item, on=item.is_on(state), success=success, count=count)

        self.check_pending_relevant_items()

        if self.new_relevant_items:
            self.notify_all(source=self)

            # after spinoff, all stats are reset to remove the influence of relevant items in stats; tracking the
            # correlations for these items is deferred to the schema's context spin-offs
            if GlobalParams().is_enabled(GlobalOption.EC_DEFER_TO_MORE_SPECIFIC_SCHEMA):
                self.stats.clear()

    def defer_update_to_spin_offs(self, state: State) -> bool:
        """ Checks if this state activates previously recognized non-negated relevant items. If so, defer to spin-offs.

        Note: This method supports the optional enhancement GlobalOption.EC_DEFER_TO_MORE_SPECIFIC_SCHEMA

        :param state: a State
        :return: True if should defer stats updates for this state to this schema's spin-offs.
        """
        for ia in self.relevant_items:
            if not ia.is_negated and ia.is_satisfied(state):
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
            self.update_relevant_items(ia)

        self.clear_pending_relevant_items()

    def clear_pending_relevant_items(self) -> None:
        self._pending_relevant_items.clear()
        self._pending_max_specificity = -np.inf

    def update_relevant_items(self, assertion: Assertion):
        # TODO: need to guarantee that only item assertions are being added here....
        if GlobalParams().is_enabled(GlobalOption.EC_POSITIVE_ASSERTIONS_ONLY) and assertion.is_negated:
            # TODO: need to verify that this is a SINGLE item or composite item!
            self.update_suppressed_items(assertion.items)
        else:
            super().update_relevant_items(assertion)

    def _check_for_relevance(self, item: Item, item_stats: ECItemStats) -> None:
        # if item is relevant, a new item assertion is created
        item_assert = (
            ItemAssertion(item) if item_stats.success_corr > GlobalParams().pos_corr_threshold else
            ItemAssertion(item, negated=True) if item_stats.failure_corr > GlobalParams().neg_corr_threshold else
            None
        )

        if item_assert and item_assert not in self.relevant_items:
            specificity = self.stats[item].specificity

            # if enabled, this enhancement allows only a single relevant item per update (the most "specific").
            if GlobalParams().is_enabled(GlobalOption.EC_MOST_SPECIFIC_ON_MULTIPLE):
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
                 result: Optional[StateAssertion] = None):
        super().__init__()

        self._context: Optional[StateAssertion] = context or NULL_STATE_ASSERT
        self._action: Action = action
        self._result: Optional[StateAssertion] = result or NULL_STATE_ASSERT

        if self.action is None:
            raise ValueError('Action cannot be None')

        self._stats: SchemaStats = SchemaStats()

        self._is_primitive = not (context or result)

        self._extended_context: ExtendedContext = (
            ExtendedContext(self._context)
            if not self._is_primitive else
            None
        )
        self._extended_result: ExtendedResult = (
            ExtendedResult(self._result)
            if self._is_primitive or GlobalParams().is_enabled(GlobalOption.ER_INCREMENTAL_RESULTS) else
            None
        )

        # TODO: Need to update overriding conditions.
        self._overriding_conditions: Optional[StateAssertion] = None

        # This observer registration is used to notify the schema when a relevant item has been detected in its
        # extended context or extended result
        if not self._is_primitive:
            self._extended_context.register(self)

        if self._is_primitive or GlobalParams().is_enabled(GlobalOption.ER_INCREMENTAL_RESULTS):
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

    def is_primitive(self) -> bool:
        """ Returns whether this instance is a primitive (action-only) schema.

        :return: True if this is a primitive schema; False otherwise.
        """
        return self._is_primitive

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
            if not GlobalParams().is_enabled(GlobalOption.ER_SUPPRESS_UPDATE_ON_EXPLAINED) or not explained:
                self._extended_result.update_all(activated=activated, new=new, lost=lost, count=count)

        # update extended context stats
        if all((self._extended_context, activated, s_prev)):
            self._extended_context.update_all(state=s_prev, success=success, count=count)

    # invoked by a schema's extended context or extended result when a relevant item is discovered
    def receive(self, *args, **kwargs) -> None:

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


def is_reliable(schema: Schema, threshold: Optional[float] = None) -> bool:
    threshold = threshold or GlobalParams().reliability_threshold
    return schema.reliability != np.NAN and schema.reliability >= threshold


class SchemaTreeNode(NodeMixin):
    def __init__(self,
                 context: Optional[StateAssertion] = None,
                 action: Optional[Action] = None,
                 label: str = None) -> None:
        self._context = context
        self._action = action

        self._schemas = set()

        self.label = label

    def __hash__(self) -> int:
        return hash((self._context, self._action))

    def __eq__(self, other) -> bool:
        if isinstance(other, SchemaTreeNode):
            return self._action == other._action and self._context == other._context

    def __str__(self) -> str:
        return self.label if self.label else f'{self._context}/{self._action}/'

    def __repr__(self) -> str:
        return repr_str(self, {'context': self._context,
                               'action': self._action, })

    @property
    def context(self) -> Optional[StateAssertion]:
        return self._context

    @property
    def action(self) -> Optional[Action]:
        return self._action

    @property
    def schemas(self) -> MutableSet[Schema]:
        return self._schemas

    @schemas.setter
    def schemas(self, value) -> None:
        self._schemas = value

    def copy(self) -> SchemaTreeNode:
        """ Returns a new SchemaTreeNode with the same context and action.

        Note: any schemas associated with the original will be lost.

        :return: A SchemaTreeNode with the same context and action as this instance.
        """
        return SchemaTreeNode(context=self.context,
                              action=self.action)


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

    def __init__(self, primitives: Optional[Collection[Schema]] = None) -> None:
        self._root = SchemaTreeNode(label='root')
        self._nodes: dict[tuple[StateAssertion, Action], SchemaTreeNode] = dict()

        self._n_schemas = 0

        if primitives:
            self.add_primitives(primitives)

    def __iter__(self) -> Iterator[SchemaTreeNode]:
        iter_ = LevelOrderIter(node=self.root)
        next(iter_)  # skips root
        return iter_

    def __len__(self) -> int:
        """ Returns the number of SchemaTreeNodes (not the number of schemas).

        :return: The number of SchemaTreeNodes in this tree.
        """
        return len(self._nodes)

    def __contains__(self, s: Union[SchemaTreeNode, Schema]) -> bool:
        if isinstance(s, SchemaTreeNode):
            return (s.context, s.action) in self._nodes
        elif isinstance(s, Schema):
            node = self._nodes.get((s.context, s.action))
            return s in node.schemas if node else False

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

    def get(self, schema: Schema) -> SchemaTreeNode:
        """ Retrieves the SchemaTreeNode matching this schema's context and action (if it exists).

        :param schema: the schema (in particular, the context and action) on which this retrieval is based
        :return: a SchemaTreeNode (if found) or raises a KeyError
        """
        return self._nodes[(schema.context, schema.action)]

    def add_primitives(self, primitives: Collection[Schema]) -> None:
        """ Adds primitive schemas to this tree.

        :param primitives: a collection of primitive schemas
        :return: None
        """
        self.add(self.root, frozenset(primitives))

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

    def find_all_satisfied(self, state: State, *args, **kwargs) -> Collection[SchemaTreeNode]:
        """ Returns a collection of tree nodes containing schemas with contexts that are satisfied by this state.

        :param state: the state
        :return: a collection of schemas
        """
        matches: MutableSet[SchemaTreeNode] = set()

        nodes_to_process = list(self._root.children)
        while nodes_to_process:
            node = nodes_to_process.pop()
            if node.context.is_satisfied(state, *args, **kwargs):
                matches.add(node)
                if node.children:
                    nodes_to_process += node.children

        return matches

    def is_valid_node(self, node: SchemaTreeNode, raise_on_invalid: bool = False) -> bool:
        if node is self._root:
            return True

        # 1. node is in tree (path from root to node)
        if node not in self:
            if raise_on_invalid:
                raise ValueError('invalid node: no path from node to root')
            return False

        # 2. node has proper depth for context
        if len(node.context) != node.depth - 1:
            if raise_on_invalid:
                raise ValueError('invalid node: depth must equal the number of item assertion in context minus 1')
            return False

        # checks that apply to nodes that contain non-primitive schemas (context + action)
        if node.parent is not self.root:

            # 3. node has same action as parent
            if node.action != node.parent.action:
                if raise_on_invalid:
                    raise ValueError('invalid node: must have same action as parent')
                return False

            # 4. node's context contains all of parents
            if not all({ia in node.context for ia in node.parent.context}):
                if raise_on_invalid:
                    raise ValueError('invalid node: context should contain all of parent\'s item assertion')
                return False

            # 5. node's context contains exactly one item assertion not in parent's context
            if len(node.parent.context) + 1 != len(node.context):
                if raise_on_invalid:
                    raise ValueError('invalid node: context must differ from parent in exactly one assertion.')
                return False

        # consistency checks between node and its schemas
        if node.schemas:

            # 6. actions should be identical across all contained schemas, and equal to node's action
            if not all({node.action == s.action for s in node.schemas}):
                if raise_on_invalid:
                    raise ValueError('invalid node: all schemas must have the same action')
                return False

            # 7. contexts should be identical across all contained schemas, and equal to node's context
            if not all({node.context == s.context for s in node.schemas}):
                if raise_on_invalid:
                    raise ValueError('invalid node: all schemas must have the same context')
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
            schemas: frozenset[Schema],
            spin_off_type: Optional[Schema.SpinOffType] = None) -> SchemaTreeNode:
        """ Adds schemas to this schema tree.

        :param source: the "source" schema that generated the given (primitive or spin-off) schemas, or the previously
        added tree node containing that "source" schema.
        :param schemas: a collection of (primitive or spin-off) schemas
        :param spin_off_type: the schema spin-off type (CONTEXT or RESULT), or None when adding primitive schemas

        Note: The source can also be the tree root. This can be used for adding primitive schemas to the tree. In
        this case, the spin_off_type should be None or CONTEXT.

        :return: the parent node for which the add operation occurred
        """
        if not schemas:
            raise ValueError('Schemas to add cannot be empty or None')

        try:
            node = source if isinstance(source, SchemaTreeNode) else self.get(source)
            if Schema.SpinOffType.RESULT is spin_off_type:
                # needed because schemas to add may already exist in set reducing total new count
                len_before_add = len(node.schemas)
                node.schemas |= schemas
                self._n_schemas += len(node.schemas) - len_before_add
            # for context spin-offs and primitive schemas
            else:
                for s in schemas:
                    key = (s.context, s.action)

                    # node already exists in tree (generated from different source)
                    if key in self._nodes:
                        continue

                    new_node = SchemaTreeNode(s.context, s.action)
                    new_node.schemas.add(s)

                    node.children += (new_node,)

                    self._nodes[key] = new_node

                    self._n_schemas += len(new_node.schemas)
            return node
        except KeyError:
            raise ValueError('Source schema does not have a corresponding tree node.')


def _output_fd(level: Verbosity) -> TextIO:
    return sys.stdout if level < Verbosity.WARN else sys.stderr


def _timestamp() -> str:
    return datetime.now().isoformat()


def _display_message(message: str, level: Verbosity) -> None:
    if level >= GlobalParams().verbosity:
        out = GlobalParams().output_format.format(timestamp=_timestamp(), severity=level.name, message=message)
        print(out, file=_output_fd(level))


def debug(message):
    _display_message(message=message, level=Verbosity.DEBUG)


def info(message):
    _display_message(message=message, level=Verbosity.INFO)


def warn(message):
    _display_message(message=message, level=Verbosity.WARN)


def error(message):
    _display_message(message=message, level=Verbosity.ERROR)


def fatal(message):
    _display_message(message=message, level=Verbosity.FATAL)