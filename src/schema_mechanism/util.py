import sys
from abc import ABCMeta
from collections.abc import Iterable
from itertools import tee
from typing import Any

import numpy as np
import sklearn.metrics as sk_metrics


class UniqueIdMixin:
    _last_uid: int = 0

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

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
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def receive(self, *args, **kwargs) -> None:
        """ Receive update from observer.

        :return: None
        """
        raise NotImplementedError('Observer\'s receive method is missing an implementation.')


class Observable:
    """ An observable subject. """

    def __init__(self, *args, **kwargs) -> None:
        self._observers: list[Observer] = list()

        super().__init__(*args, **kwargs)

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

    def notify_all(self, *args, **kwargs) -> None:
        """ Notify all registered observers.

        :param args: positional args to broadcast to observers
        :param kwargs: keyword args to broadcast to observers
        :return: None
        """
        for obs in self._observers:
            obs.receive(*args, **kwargs)


class Singleton(type, metaclass=ABCMeta):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


def cosine_sims(v: np.ndarray, state: Iterable[np.ndarray]) -> np.ndarray:
    """ Calculates the cosine similarities between a vector v and a set of state vectors. """
    return sk_metrics.pairwise.cosine_similarity(v.reshape(1, -1), state)


def get_orthogonal_vector(v: np.ndarray):
    """ Applies Gram-Schmidt to make a new vector that is orthogonal to v """
    v_orth = np.random.rand(*v.shape)
    v_orth -= v_orth.dot(v) / np.linalg.norm(v) ** 2
    return v_orth


def repr_str(obj: Any, attr_values: dict[str, Any]) -> str:
    type_name = type(obj).__name__

    return f'{type_name}({attr_values})'


# note: this is available in standard Python starting in 3.10
def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def warn(message):
    print(message, file=sys.stderr)
