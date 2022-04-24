from __future__ import annotations

import copy
import re
from random import choice
from typing import NamedTuple
from typing import Optional

import numpy as np

from schema_mechanism.core import Action
from schema_mechanism.protocols import State
from schema_mechanism.share import debug

DIRECTIONS = 'NSEW'

dir_trans = str.maketrans(DIRECTIONS, '0123')
dir_rev_trans = str.maketrans('0123', DIRECTIONS)

dir_left_of = {'N': 'W', 'S': 'E', 'E': 'N', 'W': 'S'}
dir_right_of = {'N': 'E', 'S': 'W', 'E': 'S', 'W': 'N'}


def pos_in_front_of(agent, steps=1) -> tuple[int, int]:
    return forward_move_dict[agent.direction](agent.position, steps)


# Update rules for agent position based on direction, number of steps, and move forward action
forward_move_dict = {'N': lambda pos, steps: tuple(pos + np.array((-steps, 0))),
                     'S': lambda pos, steps: tuple(pos + np.array((steps, 0))),
                     'E': lambda pos, steps: tuple(pos + np.array((0, steps))),
                     'W': lambda pos, steps: tuple(pos + np.array((0, -steps)))}


def direction_as_int(direction) -> int:
    return int(direction.translate(dir_trans))


def is_alive(actor) -> bool:
    return actor.health > 0


def agent_in_front_of_wumpus(agent, wumpus) -> bool:
    return pos_in_front_of(agent) == wumpus.position


class WumpusWorldAgent:

    def __init__(self, position: tuple[int, int], direction: str, health: int = 1, n_arrows: int = 0) -> None:
        """ Initialize an agent with an initial position and the direction the agent is facing.

        :param position: a tuple (x,y) coordinate of the agent's current WumpusWorld position
        :param direction: a character with compass direction 'N', 'S', 'E', 'W'
        :param health: an integer number of health points
        :param n_arrows: an integer number of arrows
        """
        self.position = position
        self.direction = direction
        self.health = health
        self.n_arrows = n_arrows
        self.n_gold = 0

        self.has_escaped = False

    def move_forward(self, steps: int = 1) -> None:
        """ Update agent position by applying move forward action """
        self.position = pos_in_front_of(self, steps)

    def turn_left(self) -> None:
        """ Update agent direction by applying turn left action """
        self.direction = dir_left_of[self.direction]

    def turn_right(self) -> None:
        """ Update agent direction by applying turn right action """
        self.direction = dir_right_of[self.direction]

    @property
    def as_array(self) -> np.ndarray:
        return np.array([self.position[0], self.position[1], direction_as_int(self.direction),
                         self.health, self.n_arrows, self.n_gold], dtype=np.int)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return 'position: {}\ndirection: {}\nhealth: {}\narrows: {}\ngold: {}'.format(self.position, self.direction,
                                                                                      self.health, self.n_arrows,
                                                                                      self.n_gold)


class Wumpus:
    def __init__(self, position: tuple[int, int], health: int = 1) -> None:
        """ Initialize the wumpus with a position in WumpusWorld and a number of health points.

        :param position: a tuple (x,y) coordinate of the Wumpus's current WumpusWorld position
        :param health: an integer number of health points
        """
        self.position = position
        self.health = health

    @property
    def as_array(self) -> np.ndarray:
        return np.array([self.position[0], self.position[1], self.health], dtype=np.int)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return 'position: {}\nhealth: {}'.format(self.position, self.health)


# symbols
ARROW = 'a'
WALL = 'w'
PIT = 'p'
EXIT = 'e'
GOLD = 'g'
FREE = '.'
AGENT = 'A'
WUMPUS = 'W'


class WumpusWorld:
    def __init__(self, world_map: str) -> None:
        """A Wumpus Grid World environment

        :param world_map: a string representations that depicts the wumpus world and its objects
        """
        self.cells = np.array([list(row) for row in re.sub(' ', '', world_map).split('\n') if len(row) != 0])
        self.symbols = self._unique_world_symbols

        # Assumes rectangular grid world
        self.n_rows = len(self.cells)
        self.n_columns = len(self.cells[0])
        self.n_cells = self.n_rows * self.n_columns

    def clear(self, location: tuple[int, int]) -> None:
        self.cells[location] = FREE

    def locations_of(self, obj):
        if obj not in self.symbols:
            return list()

        return list(zip(*np.where(self.cells == obj)))

    @property
    def as_array(self):
        """Return a state vector (1d numpy ndarray) that summarizes the environment state

        :return: A state vector
        """
        return np.reshape(np.vectorize(self.symbols.get)(self.cells), (-1))

    @property
    def _unique_world_symbols(self):
        symbols = {}

        symbol_id = 0
        for row in self.cells:
            for symbol in row:
                if symbol not in symbols:
                    symbols[symbol] = symbol_id
                    symbol_id += 1

        return symbols


def agent_takes_damage(mdp):
    if mdp.agent.position in mdp.env.locations_of(PIT):
        debug("Agent fell in pit!")
        return True

    elif mdp.wumpus and is_alive(mdp.wumpus) and mdp.agent.position == mdp.wumpus.position:
        debug("Agent attacked by Wumpus!")
        return True

    return False


def move_forward(mdp):
    # Illegal move (into a wall)
    if not pos_in_front_of(mdp.agent) in mdp.env.locations_of(WALL):

        # Update agent position
        mdp.agent.move_forward()

        # Update agent health based on new position
        if agent_takes_damage(mdp):
            mdp.agent.health -= 1


def pickup_gold(mdp):
    if mdp.agent.position in mdp.env.locations_of(GOLD):
        debug("Gold Found!")

        mdp.agent.n_gold += 1

        # Remove gold from environment at agent position
        mdp.env.clear(mdp.agent.position)


def pickup_arrow(mdp):
    if mdp.agent.position in mdp.env.locations_of(ARROW):
        debug("Arrow Found!")

        mdp.agent.n_arrows += 1

        # Remove arrow from environment at agent position
        mdp.env.clear(mdp.agent.position)


def fire_arrow(mdp) -> None:
    if mdp.agent.n_arrows <= 0:
        return

    debug("Arrow Fired!")
    mdp.agent.n_arrows -= 1

    if pos_in_front_of(mdp.agent) == mdp.wumpus.position and is_alive(mdp.wumpus):
        mdp.wumpus.health -= 1

        if is_alive(mdp.wumpus):
            debug("Wumpus Hit!")
        else:
            debug("Wumpus Killed!")


def use_exit(mdp):
    if mdp.agent.position in mdp.env.locations_of(EXIT):
        mdp.agent.has_escaped = True


# Actions Definitions
MOVE_FORWARD = Action('MOVE[FORWARD]')
TURN_LEFT = Action('TURN[LEFT]')
TURN_RIGHT = Action('TURN[RIGHT]')
# PICKUP_GOLD = Action('PICKUP[GOLD]')
# PICKUP_ARROW = Action('PICKUP[ARROW]')
# FIRE_ARROW = Action('FIRE[ARROW]')
USE_EXIT = Action('USE[EXIT]')

actions_dict = {
    MOVE_FORWARD: move_forward,
    TURN_LEFT: lambda mdp: mdp.agent.turn_left(),
    TURN_RIGHT: lambda mdp: mdp.agent.turn_right(),
    USE_EXIT: use_exit
}

actions_avail_dict = {
    MOVE_FORWARD: lambda mdp: not pos_in_front_of(mdp.agent) in mdp.env.locations_of(WALL),
    TURN_LEFT: lambda mdp: True,
    TURN_RIGHT: lambda mdp: True,
    USE_EXIT: lambda mdp: mdp.agent.position in mdp.env.locations_of(EXIT)
}


# actions_dict = {
#     MOVE_FORWARD: move_forward,
#     TURN_LEFT: lambda mdp: mdp.agent.turn_left(),
#     TURN_RIGHT: lambda mdp: mdp.agent.turn_right(),
#     PICKUP_GOLD: pickup_gold,
#     PICKUP_ARROW: pickup_arrow,
#     FIRE_ARROW: fire_arrow,
#     USE_EXIT: use_exit
# }
#
# actions_avail_dict = {
#     MOVE_FORWARD: lambda mdp: not pos_in_front_of(mdp.agent) in mdp.env.locations_of(WALL),
#     TURN_LEFT: lambda mdp: True,
#     TURN_RIGHT: lambda mdp: True,
#     PICKUP_GOLD: lambda mdp: mdp.agent.position in mdp.env.locations_of(GOLD),
#     PICKUP_ARROW: lambda mdp: mdp.agent.position in mdp.env.locations_of(ARROW),
#     FIRE_ARROW: lambda mdp: mdp.agent.n_arrows > 0,
#     USE_EXIT: lambda mdp: mdp.agent.position in mdp.env.locations_of(EXIT)
# }


def available_actions(obs):
    return [a for a in actions_dict.keys() if actions_avail_dict[a](obs)]


class Position(NamedTuple):
    x: int  # a non-negative integer indicating horizontal world position
    y: int  # a non-negative integer indicating vertical world position


class PhysicalState(NamedTuple):
    direction: str  # N,S,E,W
    position: Position
    health: int  # a non-negative integer quantifying the health of an entity or object


class AgentState(NamedTuple):
    physical_state: PhysicalState
    possessions: dict


class WumpusState(NamedTuple):
    physical_state: PhysicalState


class WorldState(NamedTuple):
    map: str
    agent_state: AgentState
    wumpus_state: WumpusState


def get_agent_observation(world: WumpusWorld, agent: WumpusWorldAgent, wumpus: Wumpus) -> State:
    state_elements = [
        f'AGENT.POSITION={agent.position[0]};{agent.position[1]}',
        f'AGENT.DIR={agent.direction}',
        # f'AGENT.HEALTH={agent.health}',

        # agent's possessions
        # f'AGENT.HAS[ARROWS:{agent.n_arrows}]',
        # f'AGENT.HAS[GOLD:{agent.n_gold}]',
    ]

    # objects/entities in agent's cell
    for obj in world.cells[agent.position]:
        state_elements.append(f'IN_CELL_WITH_AGENT={obj}')

    # objects/entities in front of agent
    for obj in world.cells[pos_in_front_of(agent)]:
        state_elements.append(f'IN_FRONT_OF_AGENT={obj}')

    # other events
    ##############

    # Wumpus is dead
    if wumpus and not is_alive(wumpus):
        state_elements.append('EVENT[WUMPUS DEAD]')

    # agent has escaped
    if agent.has_escaped:
        state_elements.append('EVENT[AGENT ESCAPED]')

    return tuple(state_elements)


class WumpusWorldMDP:

    def __init__(self,
                 worldmap: str,
                 agent: WumpusWorldAgent,
                 wumpus: Optional[Wumpus] = None,
                 randomized_start: bool = False) -> None:
        """Initialize a WumpusWorldMDP for the specified WumpusWorld and MDP attributes.

        :param worldmap: a string representations that depicts the wumpus world and its objects
        :param agent: a WumpusWorldAgent object to serve as an agent specification
        :param wumpus: a Wumpus object to serve as a Wumpus specification
        """
        self.worldmap = worldmap
        self.agent_spec = agent
        self.wumpus_spec = wumpus
        self.randomized_start = randomized_start

        self.env: Optional[WumpusWorld] = None
        self.agent: Optional[WumpusWorldAgent] = None
        self.wumpus: Optional[Wumpus] = None

    @property
    def state(self) -> State:
        return get_agent_observation(world=self.env, agent=self.agent, wumpus=self.wumpus)

    @property
    def actions(self) -> list[Action]:
        return list(actions_dict.keys())

    @property
    def is_terminal(self) -> bool:
        if not is_alive(self.agent) or self.agent.has_escaped:
            return True

        return False

    def render(self) -> str:
        displayed_cells = np.copy(self.env.cells)

        # Add agent and wumpus
        displayed_cells[self.agent.position] = AGENT

        if self.wumpus:
            displayed_cells[self.wumpus.position] = WUMPUS

        return str(displayed_cells)

    def step(self, action: Action) -> tuple[State, bool]:
        """Update the mdp based on supplied agent action."""
        if self.env is None:
            raise ValueError("Reset must be called to initialize the environment.")

        if self.is_terminal:
            raise ValueError("MDP is in a terminal state. reset() must be called to reinitialize the mdp.")
            # return state, 0.0, done

        # Update MDP from action
        actions_dict[action](self)

        # Return reward, next_state, and is_terminal value
        return self.state, self.is_terminal

    def reset(self) -> tuple[State, bool]:
        self.env = WumpusWorld(self.worldmap)
        self.agent = copy.deepcopy(self.agent_spec)

        if self.randomized_start:
            self.agent.position = choice(self.env.locations_of(FREE))
            self.agent.direction = choice(DIRECTIONS)

        if self.wumpus_spec:
            self.wumpus = copy.deepcopy(self.wumpus_spec)

        return self.state, self.is_terminal
