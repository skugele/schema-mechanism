from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any
from typing import Iterable

from schema_mechanism.core import Action
from schema_mechanism.core import State


@dataclass
class EpisodeSummary:
    steps: int = 0


# TODO: Make this consistent with OpenAI gym environment interface.
class Environment(ABC):
    @property
    @abstractmethod
    def id(self) -> str:
        """ Returns a string identifier for this environment.

        :return: a constant str identifier
        """

    @property
    @abstractmethod
    def actions(self) -> Iterable[Action]:
        """ Returns an iterable containing the primitive actions supported for agents to execute in this environment.

        :return: an iterable of primitive actions
        """

    @property
    @abstractmethod
    def episode_summary(self) -> Any:
        """ Returns a summary of the current episode. """

    @abstractmethod
    def step(self, action: Action) -> tuple[State, bool]:
        """ Updates the environment based on the given action.

        :param action: an Action to execute against this environment

        :return: a tuple containing the updated State, and a bool that indicates if current state is terminal
        """

    @abstractmethod
    def reset(self) -> tuple[State, bool]:
        """ Resets the environment to its initial state and returns that state.

        :return: a tuple containing the initial State, and a bool that indicates if initial state is terminal
        """

    @abstractmethod
    def render(self) -> Any:
        """ Renders the environment in a format suitable for human consumption. """

    @abstractmethod
    def is_terminal(self) -> bool:
        """ Returns a bool signifying whether the environment is in a terminal state. """
