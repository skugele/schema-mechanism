from __future__ import annotations

import itertools
import logging
import random
from abc import ABC
from abc import ABCMeta
from abc import abstractmethod
from collections import Collection
from collections import defaultdict
from collections.abc import Iterable
from itertools import tee
from pathlib import Path
from typing import Any
from typing import Generic
from typing import Optional
from typing import TypeVar
from typing import Union

import numpy as np

logger = logging.getLogger(__name__)


class UniqueIdMixin:
    _last_uid: int = 0

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self._uid = self._gen_uid()

    @property
    def uid(self) -> int:
        return self._uid

    @classmethod
    def _gen_uid(cls) -> int:
        cls._last_uid += 1
        return cls._last_uid


class Observer(ABC):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    @abstractmethod
    def receive(self, **kwargs) -> None:
        """ Receive update from observer.

        :return: None
        """


class Observable(ABC):
    """ An observable subject. """

    def __init__(self, **kwargs) -> None:
        self._observers: list[Observer] = list()

        super().__init__(**kwargs)

    @property
    def observers(self) -> list[Observer]:
        return self._observers

    def register(self, observer: Observer) -> None:
        """ Registers an observer with this observable.

        :param observer: the observer to register
        :return: None
        """
        self._observers.append(observer)

    def unregister(self, observer: Observer) -> None:
        """ Unregisters an observer that was previously registered with this observable.

        :param observer: the observer to unregister
        :return: None
        """
        self._observers.remove(observer)

    def notify_all(self, **kwargs) -> None:
        """ Notify all registered observers.

        :param kwargs: keyword args to broadcast to observers
        :return: None
        """
        for obs in self._observers:
            obs.receive(**kwargs)


class Singleton(type, metaclass=ABCMeta):
    _instances = {}

    def __call__(cls, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(**kwargs)
        return cls._instances[cls]


class BoundedSet(set):
    def __init__(self, values: Optional[Iterable[Any]] = None, accepted_values: Optional[Collection[Any]] = None):
        self._accepted_values = frozenset(accepted_values) if accepted_values else frozenset()

        values = values or []
        self.check_values(values)
        super().__init__(values)

    def update(self, *s: Iterable[Any]) -> None:
        self.check_values(itertools.chain.from_iterable(s))
        super().update(*s)

    def add(self, element: Any) -> None:
        self.check_values([element])
        super().add(element)

    def is_legal_value(self, value: Any) -> bool:
        return value in self._accepted_values

    def check_values(self, values: Iterable[Any]) -> None:
        illegal_values = [v for v in values if not self.is_legal_value(v)]
        if illegal_values:
            raise ValueError(f'illegal values: {illegal_values}')


T = TypeVar('T')


class AssociativeArrayList(Generic[T]):
    def __init__(self, pre_allocated: int = 1000, block_size: int = 100):
        self._pre_allocated = pre_allocated
        self._block_size = block_size

        # map of hashable objects to their values array index
        self._indexes: dict[T, int] = dict()

        self._values = np.zeros(pre_allocated, dtype=np.float64)

        self._last_index = 0

    def __iter__(self) -> T:
        yield from self._indexes.keys()

    def __len__(self):
        return self._last_index

    def __contains__(self, item: T) -> bool:
        return item in self._indexes

    def __getitem__(self, key: T) -> np.ndarray:
        if key not in self._indexes:
            self.add([key])

        index = self._indexes[key]
        return np.array([self._values[index]])

    def __setitem__(self, key: T, value: float) -> None:
        if key not in self._indexes:
            self.add([key])

        index = self._indexes[key]
        self._values[index] = value

    def __delitem__(self, key: T):
        if key not in self._indexes:
            raise IndexError(f'Key does not exist: {key}')

        index = self._indexes[key]
        del self._indexes[key]

        self._values = np.append(self._values[:index], self._values[index + 1:])
        self._last_index -= 1

    # overrides the default behavior of using __len__ to determine truthiness
    def __bool__(self) -> bool:
        return True

    def __eq__(self, other) -> bool:
        if isinstance(other, AssociativeArrayList):
            return all(
                (
                    # generator expression for conditions to allow lazy evaluation
                    condition for condition in
                    [
                        self._pre_allocated == other._pre_allocated,
                        self._block_size == other._block_size,
                        self._indexes == other._indexes,
                        self._last_index == other._last_index,
                        np.array_equal(self._values, other._values),
                    ]
                )
            )
        return False if other is None else NotImplemented

    @property
    def block_size(self) -> int:
        return self._block_size

    @property
    def n_allocated(self) -> int:
        return len(self._values)

    def keys(self) -> np.ndarray:
        keys = list(self._indexes.keys())
        return np.array(keys)

    @property
    def values(self) -> np.ndarray:
        """ Returns the active portion of the values array. """
        return self._values[:self._last_index]

    @values.setter
    def values(self, values: np.ndarray) -> None:
        self._values = values

    def items(self) -> (T, float):
        """ An iterator over registered objects and their values. """
        for key in self.keys():
            index = self._indexes[key]
            value = self._values[index]

            yield key, value

    def add(self, elements: Collection[T]):
        new_elements = [e for e in elements if e not in self._indexes]
        if not new_elements:
            return

        indexes = range(self._last_index, self._last_index + len(new_elements))
        self._indexes.update({k: v for k, v in zip(new_elements, indexes)})

        # allocate a new block of elements (as necessary)
        overflow = len(self._indexes) - len(self._values)
        blocks_needed = overflow // self._block_size + 1 if overflow else 0
        if blocks_needed > 0:
            self._values = np.append(self._values, np.zeros(blocks_needed * self._block_size))

        self._last_index += len(new_elements)

    def update(self, other: Union[AssociativeArrayList, dict]) -> None:
        for key in other:
            if key not in self:
                self.add(key)

            index = self._indexes[key]
            self._values[index] = other[key]

    def clear(self) -> None:
        self._indexes.clear()
        self._values = np.zeros(self._pre_allocated, dtype=np.float64)

        self._last_index = 0

    def indexes(self, keys: Collection[T], add_missing: bool = False) -> np.ndarray:
        if keys is None:
            return np.array([])

        unknown_keys = [k for k in keys if k not in self._indexes]
        if unknown_keys:
            if not add_missing:
                raise ValueError(f'Encountered unknown keys: {unknown_keys}')

            self.add(unknown_keys)

        indexes = np.array([self._indexes[k] for k in keys if k in self._indexes])
        return indexes


class DefaultDictWithKeyFactory(defaultdict):
    """ A subclass of defaultdict that supports defaults parameterized by the missing keys.

    """

    # noinspection PyArgumentList
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret


def repr_str(obj: Any, attr_values: dict[str, Any]) -> str:
    type_name = type(obj).__name__

    # attribute values sorted by dictionary key
    sorted_attr_values = {key: attr_values[key] for key in sorted(attr_values)}

    return f'{type_name}({sorted_attr_values or str()})'


# note: this is available in standard Python starting in 3.10
def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def equal_weights(n: int) -> np.ndarray:
    if n < 0:
        raise ValueError('n must be a positive integer')

    return np.ones(n) / n


_seed: Optional[int] = None


def set_random_seed(seed: int) -> None:
    global _seed
    _seed = seed

    random.seed(_seed)


def get_random_seed() -> Optional[int]:
    return _seed


_rng: Optional[np.random.Generator] = None


def rng():
    global _rng
    global _seed

    seed = get_random_seed()
    if not _rng:
        logger.warning(f'Initializing random number generator using seed="{seed}".')
        logger.warning(
            f'For reproducibility, you should also set "PYTHONHASHSEED={seed}" in your environment variables.')

        # setting globals
        _rng = np.random.default_rng(seed)
        _seed = seed

    return _rng


def check_writable(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise ValueError(f'File {path} already exists. Use overwrite to replace.')


def check_readable(path: Path) -> None:
    if not (path.exists() and path.is_file()):
        raise ValueError(f'Unable to deserialize non-existent or unreadable file: {path}')
