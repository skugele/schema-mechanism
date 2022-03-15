import importlib
import itertools
from abc import ABCMeta
from collections import Collection
from collections.abc import Iterable
from itertools import tee
from typing import Any
from typing import Optional
from typing import Type

import numpy as np
import sklearn.metrics as sk_metrics


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
        # FIXME: not thread safe
        cls._last_uid += 1
        return cls._last_uid


class Observer:
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def receive(self, **kwargs) -> None:
        """ Receive update from observer.

        :return: None
        """
        raise NotImplementedError('Observer\'s receive method is missing an implementation.')


class Observable:
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


def cosine_sims(v: np.ndarray, state: Iterable[np.ndarray]) -> np.ndarray:
    """ Calculates the cosine similarities between a vector v and a set of state vectors. """
    return sk_metrics.pairwise.cosine_similarity(v.reshape(1, -1), state)


def repr_str(obj: Any, attr_values: dict[str, Any]) -> str:
    type_name = type(obj).__name__

    return f'{type_name}({attr_values})'


# note: this is available in standard Python starting in 3.10
def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def dynamic_type(module_name: str, class_name: str) -> Type[Any]:
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def equal_weights(n: int) -> np.array:
    if n < 0:
        raise ValueError('n must be a positive integer')

    return np.ones(n) / n
