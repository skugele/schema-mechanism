from abc import ABCMeta
from typing import Iterable
from uuid import uuid4

import numpy as np
import sklearn.metrics as sk_metrics


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
