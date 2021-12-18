from numbers import Number
from typing import Iterable

import numpy as np
import sklearn.metrics as sk_metrics


def cosine_sims(v: np.ndarray, state: Iterable[np.ndarray]) -> np.ndarray:
    """ Calculates the cosine similarities between a vector v and a set of state vectors. """
    return sk_metrics.pairwise.cosine_similarity(v.reshape(1, -1), state)


def get_orthogonal_vector(v: np.ndarray):
    """ Applies Gram-Schmidt to make a new vector that is orthogonal to v """
    v_orth = np.random.rand(*v.shape)
    v_orth -= v_orth.dot(v) / np.linalg.norm(v) ** 2
    return v_orth
