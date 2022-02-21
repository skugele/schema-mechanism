import argparse
import random
from collections.abc import Sequence
from typing import Optional

from schema_mechanism.core import Action
from schema_mechanism.core import GlobalOption
from schema_mechanism.core import GlobalParams
from schema_mechanism.core import State
from schema_mechanism.core import StateElement
from schema_mechanism.func_api import sym_item
from schema_mechanism.func_api import sym_state
from schema_mechanism.modules import SchemaMechanism
from schema_mechanism.util import Observable

DEFAULT_N_MACHINES = 5
DEFAULT_N_STEPS = 5000


# state elements
class Machine:
    def __init__(self, id_: str, p_win: float) -> None:
        self._id = id_
        self._p_win = p_win
        self._outcomes = ['L', 'W']
        self._weights = [1.0 - p_win, p_win]

    def play(self, count: int = 1) -> Sequence[str]:
        return random.choices(self._outcomes, self._weights, k=count)

    @property
    def id(self) -> str:
        return self._id

    def __str__(self) -> str:
        return f'M{self._id}'

    def __repr__(self) -> str:
        return f'{self}[p_win={self._p_win:.2}]'


class BanditEnvironment(Observable):
    DEFAULT_INIT_STATE = sym_state('S')  # standing

    def __init__(self, machines: Sequence[Machine], init_state: Optional[State] = None) -> None:
        super().__init__()

        if not machines:
            raise ValueError('Must supply at least one machines must be a positive number')

        self._machines = machines

        self._actions = [Action(a_str) for a_str in ['deposit', 'stand', 'play']]
        self._sit_actions = [Action(f'sit({m})') for m in self._machines]
        self._actions.extend(self._sit_actions)

        self._state_elements = ['W', 'L', 'S', 'P'] + [str(m) for m in self._machines]
        self._states = [sym_state('W'), sym_state('L'), sym_state('S'), sym_state('P')]

        # add machine related states
        self._machine_base_states = [sym_state(str(m)) for m in self._machines]
        self._machine_play_states = [sym_state(f'{m},P') for m in self._machines]
        self._machine_win_states = [sym_state(f'{m},W') for m in self._machines]
        self._machine_lose_states = [sym_state(f'{m},L') for m in self._machines]

        self._states.extend([
            *self._machine_base_states,
            *self._machine_play_states,
            *self._machine_win_states,
            *self._machine_lose_states
        ])

        self._init_state = init_state or sym_state('S')
        if self._init_state not in self._states:
            raise ValueError(f'init_state is invalid: {self._init_state}')

        self._current_state = self._init_state

    @property
    def machines(self) -> Sequence[Machine]:
        return self._machines

    @property
    def actions(self) -> Sequence[Action]:
        return self._actions

    @property
    def state_elements(self) -> Sequence[StateElement]:
        return self._state_elements

    @property
    def states(self) -> Sequence[State]:
        return self._states

    @property
    def current_state(self) -> State:
        return self._current_state

    def step(self, action: Action) -> State:
        if action not in self.actions:
            raise ValueError(f'Invalid action: {action}')

        # standing
        if self._current_state == sym_state('S'):
            # sit at machine
            if action in self._sit_actions:
                m_ndx = self._sit_actions.index(action)
                self._current_state = self._machine_base_states[m_ndx]

        # at machine
        elif self._current_state in self._machine_base_states:
            if action == Action('deposit'):
                m_ndx = self._machine_base_states.index(self._current_state)
                self._current_state = self._machine_play_states[m_ndx]
            elif action == Action('stand'):
                self._current_state = sym_state('S')

        # at machine, coin deposited
        elif self._current_state in self._machine_play_states:
            if action == Action('play'):
                m_ndx = self._machine_play_states.index(self._current_state)
                if self._machines[m_ndx].play()[0] == 'W':
                    self._current_state = self._machine_win_states[m_ndx]
                else:
                    self._current_state = self._machine_lose_states[m_ndx]
            elif action == Action('stand'):
                self._current_state = sym_state('S')

        # at machine, just won
        elif self._current_state in self._machine_win_states:
            m_ndx = self._machine_win_states.index(self._current_state)
            if action == Action('deposit'):
                self._current_state = self._machine_play_states[m_ndx]
            elif action == Action('stand'):
                self._current_state = sym_state('S')
            else:
                self._current_state = self._machine_base_states[m_ndx]

        # at machine, just lost
        elif self._current_state in self._machine_lose_states:
            m_ndx = self._machine_lose_states.index(self._current_state)
            if action == Action('deposit'):
                self._current_state = self._machine_play_states[m_ndx]
            elif action == Action('stand'):
                self._current_state = sym_state('S')
            else:
                self._current_state = self._machine_base_states[m_ndx]

        return self._current_state

    def reset(self) -> State:
        self._current_state = self._init_state
        return self._current_state


def parse_args():
    """ Parses command line arguments.
    :return: argparse parser with parsed command line args
    """
    parser = argparse.ArgumentParser(description='Godot AI Bridge (GAB) - DEMO Environment Action Client')

    parser.add_argument('--n_machines', type=int, required=False, default=DEFAULT_N_MACHINES,
                        help=f'the id of the agent to which this action will be sent (default: {DEFAULT_N_MACHINES})')
    parser.add_argument('--steps', type=int, required=False, default=DEFAULT_N_STEPS,
                        help=f'the id of the agent to which this action will be sent (default: {DEFAULT_N_STEPS})')

    return parser.parse_args()


if __name__ == '__main__':
    GlobalParams().options.add(GlobalOption.EC_MOST_SPECIFIC_ON_MULTIPLE)
    GlobalParams().options.add(GlobalOption.EC_DEFER_TO_MORE_SPECIFIC_SCHEMA)
    # GlobalParams().options.add(GlobalOption.ER_POSITIVE_ASSERTIONS_ONLY)
    GlobalParams().options.add(GlobalOption.EC_POSITIVE_ASSERTIONS_ONLY)
    # GlobalParams().options.add(GlobalOption.ER_INCREMENTAL_RESULTS)

    args = parse_args()

    machines = [Machine(str(id_), p_win=random.uniform(0, 1)) for id_ in range(args.n_machines)]
    env = BanditEnvironment(machines)

    # primitive items
    i_win = sym_item('W', primitive_value=10.0)
    i_lose = sym_item('L', primitive_value=-2.0)
    i_pay = sym_item('P', primitive_value=-1.0)

    sm = SchemaMechanism(primitive_actions=env.actions,
                         primitive_items=[i_win, i_lose, i_pay])

    for n in range(args.steps):
        # print(f'State[{n}]: ', env.current_state)

        # action = random.choice(env.actions)
        schema = sm.select(env.current_state)
        print(f'Selected Schema[{n}]: ', schema)
        # print(schema.extended_context)
        # print(schema.extended_result)
        # print(f'Action[{n}]: {schema.action}')
        result = env.step(schema.action)

    print(f'n_schema: {len(sm.known_schemas)}')
    for s in sm.known_schemas:
        print(s)
    print(f'---------------------------------')

    for m in machines:
        print(repr(m))
