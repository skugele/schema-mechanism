import argparse
from collections.abc import Sequence
from typing import Optional

from schema_mechanism.core import Action
from schema_mechanism.core import GlobalOption
from schema_mechanism.core import GlobalParams
from schema_mechanism.core import GlobalStats
from schema_mechanism.core import ItemPool
from schema_mechanism.core import Schema
from schema_mechanism.core import State
from schema_mechanism.core import StateElement
from schema_mechanism.core import Verbosity
from schema_mechanism.core import debug
from schema_mechanism.core import info
from schema_mechanism.examples import RANDOM_SEED
from schema_mechanism.func_api import sym_item
from schema_mechanism.func_api import sym_schema
from schema_mechanism.func_api import sym_state
from schema_mechanism.modules import SchemaMechanism
from schema_mechanism.util import Observable
from schema_mechanism.util import get_rand_gen


class Machine:
    def __init__(self, id_: str, p_win: float) -> None:
        self._id = id_
        self._p_win = p_win
        self._outcomes = ['L', 'W']
        self._weights = [1.0 - p_win, p_win]

        self._rng = get_rand_gen(GlobalParams().rng_seed)

    def play(self, count: int = 1) -> Sequence[str]:
        return self._rng.choice(self._outcomes, size=count, p=self._weights)

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
        self._sit_actions = [Action(f'sit({m_})') for m_ in self._machines]
        self._actions.extend(self._sit_actions)

        self._state_elements = ['W', 'L', 'S', 'P'] + [str(m_) for m_ in self._machines]
        self._states = [sym_state('W'), sym_state('L'), sym_state('S'), sym_state('P')]

        # add machine related states
        self._machine_base_states = [sym_state(str(m_)) for m_ in self._machines]
        self._machine_play_states = [sym_state(f'{m_},P') for m_ in self._machines]
        self._machine_win_states = [sym_state(f'{m_},W') for m_ in self._machines]
        self._machine_lose_states = [sym_state(f'{m_},L') for m_ in self._machines]

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

    parser.add_argument('--n_machines', type=int, required=False, default=N_MACHINES,
                        help=f'the id of the agent to which this action will be sent (default: {N_MACHINES})')
    parser.add_argument('--steps', type=int, required=False, default=N_STEPS,
                        help=f'the id of the agent to which this action will be sent (default: {N_STEPS})')

    return parser.parse_args()


def display_known_schemas(sm: SchemaMechanism) -> None:
    info(f'n schemas ({len(sm.known_schemas)})')
    for s in sm.known_schemas:
        info(f'\t{str(s)}')
        # display_schema_info(s, sm)


def display_item_values() -> None:
    info(f'n items: ({len(ItemPool())})')
    for i in sorted(ItemPool(), key=lambda i: -i.delegated_value):
        info(repr(i))


def display_schema_info(schema: Schema, sm: SchemaMechanism) -> None:
    schemas = sm.schema_memory.schemas
    try:
        ndx = schemas.index(schema)
        s = schemas[ndx]

        info(f'schema: {schema}')
        if s.extended_context:
            info('EXTENDED_CONTEXT')
            info(f'relevant items: {s.extended_context.relevant_items}')
            for k, v in s.extended_context.stats.items():
                info(f'item: {k} -> {repr(v)}')

        if s.extended_result:
            info('EXTENDED_RESULT')
            info(f'relevant items: {s.extended_result.relevant_items}')
            for k, v in s.extended_result.stats.items():
                info(f'item: {k} -> {repr(v)}')
    except ValueError:
        return


# global constants
N_MACHINES = 2
N_STEPS = 10000


def run():
    GlobalParams().learn_rate = 0.05
    GlobalParams().dv_trace_max_len = 2

    GlobalParams().verbosity = Verbosity.DEBUG
    GlobalParams().output_format = '{message}'
    GlobalParams().rng_seed = RANDOM_SEED

    GlobalParams().options.add(GlobalOption.EC_DEFER_TO_MORE_SPECIFIC_SCHEMA)
    GlobalParams().options.add(GlobalOption.EC_MOST_SPECIFIC_ON_MULTIPLE)
    GlobalParams().options.add(GlobalOption.ER_POSITIVE_ASSERTIONS_ONLY)
    GlobalParams().options.add(GlobalOption.EC_POSITIVE_ASSERTIONS_ONLY)
    GlobalParams().options.add(GlobalOption.ER_SUPPRESS_UPDATE_ON_EXPLAINED)
    # GlobalParams().options.add(GlobalOption.ER_INCREMENTAL_RESULTS)

    args = parse_args()
    rng = get_rand_gen(GlobalParams().rng_seed)

    machines = [Machine(str(id_), p_win=rng.uniform(0, 1)) for id_ in range(args.n_machines)]
    env = BanditEnvironment(machines)

    # primitive items
    i_win = sym_item('W', primitive_value=100.0)
    i_lose = sym_item('L', primitive_value=-20.0)
    i_pay = sym_item('P', primitive_value=-5.0)

    sm = SchemaMechanism(primitive_actions=env.actions,
                         primitive_items=[i_win, i_lose, i_pay])

    for n in range(args.steps):
        # display_item_values()
        curr_state = env.current_state
        info(f'State[{n}]: {curr_state}')
        schema = sm.select(curr_state)
        # info(f'Selected Schema[{n}]: {schema}')
        result_state = env.step(schema.action)
        # info(f'Result[{n}]: {result_state}')
        # display_schema_info(sym_schema('/play/W,'), sm)
        display_schema_info(sym_schema('P,/play/W,'), sm)
        # display_schema_info(sym_schema('M0,/play/W,'), sm)
        # display_schema_info(sym_schema('P,/play/L,'), sm)
        # display_schema_info(sym_schema('P,/play/L,'), sm)
        # display_schema_info(sym_schema('M0,P/play/W,'), sm)
        # display_schema_info(sym_schema('P,/play/W,'), sm)

        # info(f'n_schema: {len(sm.known_schemas)}')
        display_known_schemas(sm)

    for m in machines:
        info(repr(m))

    debug(f'baseline: {GlobalStats().baseline_value}')
    display_item_values()


if __name__ == '__main__':
    run()
