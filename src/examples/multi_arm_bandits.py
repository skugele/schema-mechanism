import argparse
from collections.abc import Sequence
from time import time
from typing import Optional

from examples import display_summary
from schema_mechanism.core import Action
from schema_mechanism.core import State
from schema_mechanism.core import StateElement
from schema_mechanism.func_api import sym_item
from schema_mechanism.func_api import sym_state
from schema_mechanism.modules import SchemaMechanism
from schema_mechanism.share import GlobalParams
from schema_mechanism.share import info
from schema_mechanism.share import rng
from schema_mechanism.stats import CorrelationOnEncounter
from schema_mechanism.stats import FisherExactCorrelationTest
from schema_mechanism.util import Observable


class Machine:
    def __init__(self, id_: str, p_win: float) -> None:
        self._id = id_
        self._p_win = p_win
        self._outcomes = ['L', 'W']
        self._weights = [1.0 - p_win, p_win]
        self._rng = rng()

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
    parser = argparse.ArgumentParser(description='Multi-Armed Bandit Example Environment (Schema Mechanism)')

    parser.add_argument('--n_machines', type=int, required=False, default=N_MACHINES,
                        help=f'the id of the agent to which this action will be sent (default: {N_MACHINES})')
    parser.add_argument('--steps', type=int, required=False, default=N_STEPS,
                        help=f'the id of the agent to which this action will be sent (default: {N_STEPS})')

    return parser.parse_args()


# global constants
N_MACHINES = 4
N_STEPS = 1000


def run():
    args = parse_args()

    GlobalParams().set('composite_action_min_baseline_advantage', 25.0)
    GlobalParams().set('delegated_value_helper.decay_rate', 0.2)
    GlobalParams().set('delegated_value_helper.discount_factor', 0.5)
    GlobalParams().set('random_exploratory_strategy.epsilon.decay.rate.initial', 0.999)
    GlobalParams().set('random_exploratory_strategy.epsilon.decay.rate.min', 0.1)
    GlobalParams().set('schema_selection.weights.explore_weight', 0.9)
    GlobalParams().set('schema_selection.weights.goal_weight', 0.1)
    GlobalParams().set('habituation_exploratory_strategy.decay.rate', 0.9)
    GlobalParams().set('habituation_exploratory_strategy.multiplier', 10.0)
    GlobalParams().set('learning_rate', 0.05)
    GlobalParams().set('goal_pursuit_strategy.reliability.max_penalty', 10.0)
    GlobalParams().set('reliability_threshold', 0.7)

    # item correlation test used for determining relevance of extended context items
    GlobalParams().set('ext_context.correlation_test', FisherExactCorrelationTest)

    # thresholds for determining the relevance of extended result items
    #     from 0.0 [weakest correlation] to 1.0 [strongest correlation]
    GlobalParams().set('ext_context.positive_correlation_threshold', 0.95)

    # item correlation test used for determining relevance of extended result items
    GlobalParams().set('ext_result.correlation_test', CorrelationOnEncounter)

    # thresholds for determining the relevance of extended result items
    #     from 0.0 [weakest correlation] to 1.0 [strongest correlation]
    GlobalParams().set('ext_result.positive_correlation_threshold', 0.95)

    machines = [Machine(str(id_), p_win=rng().uniform(0, 1)) for id_ in range(args.n_machines)]
    env = BanditEnvironment(machines)

    # primitive items
    i_win = sym_item('W', primitive_value=100.0)
    i_lose = sym_item('L', primitive_value=-100.0)
    i_pay = sym_item('P', primitive_value=-5.0)

    sm = SchemaMechanism(primitive_actions=env.actions,
                         primitive_items=[i_win, i_lose, i_pay])

    start_time = time()

    for n in range(args.steps):
        info(f'state[{n}]: {env.current_state}')

        selection_details = sm.select(env.current_state)

        current_composite_schema = sm.schema_selection.pending_schema
        if current_composite_schema:
            info(f'active composite action schema: {current_composite_schema} ')

        terminated_composite_schemas = selection_details.terminated_pending

        if terminated_composite_schemas:
            i = 0
            for pending_details in terminated_composite_schemas:
                info(f'terminated schema [{i}]: {pending_details.schema}')
                info(f'termination status [{i}]: {pending_details.status}')
                info(f'selection state [{i}]: {pending_details.selection_state}')
                info(f'duration [{i}]: {pending_details.duration}')
            i += 1

        schema = selection_details.selected
        info(f'selected schema[{n}]: {schema}')

        state = env.step(schema.action)

        sm.learn(selection_details, result_state=state)

    end_time = time()

    info(f'elapsed time: {end_time - start_time}s')

    # displays machine properties
    info(f'machines ({len(machines)}):')
    for m in machines:
        info(f'\t{repr(m)}')

    display_summary(sm)


if __name__ == '__main__':
    run()
