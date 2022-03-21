from __future__ import annotations

from abc import ABCMeta
from collections import Collection
from collections import Hashable
from typing import Protocol
from typing import runtime_checkable


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
