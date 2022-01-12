from abc import ABCMeta
from typing import Iterable
from typing import List
from uuid import uuid4

import numpy as np
import sklearn.metrics as sk_metrics


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
        self._observers: List[Observer] = list()

        super().__init__(*args, **kwargs)

    @property
    def observers(self) -> List[Observer]:
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


def get_unique_id():
    return uuid4()
