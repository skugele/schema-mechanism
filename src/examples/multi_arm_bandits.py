import argparse
from collections.abc import Sequence
from time import sleep
from typing import Optional

from examples import display_summary
from examples import is_paused
from examples import run_decorator
from schema_mechanism.core import Action
from schema_mechanism.core import GlobalStats
from schema_mechanism.core import Schema
from schema_mechanism.core import State
from schema_mechanism.func_api import sym_item
from schema_mechanism.func_api import sym_state
from schema_mechanism.modules import AbsoluteDiffMatchStrategy
from schema_mechanism.modules import RandomizeBestSelectionStrategy
from schema_mechanism.modules import SchemaMechanism
from schema_mechanism.modules import SchemaMemory
from schema_mechanism.modules import SchemaSelection
from schema_mechanism.share import GlobalParams
from schema_mechanism.share import display_params
from schema_mechanism.share import info
from schema_mechanism.share import rng
from schema_mechanism.stats import CorrelationOnEncounter
from schema_mechanism.stats import FisherExactCorrelationTest
from schema_mechanism.strategies.decay import GeometricDecayStrategy
from schema_mechanism.strategies.evaluation import DefaultGoalPursuitEvaluationStrategy
from schema_mechanism.strategies.evaluation import EpsilonGreedyEvaluationStrategy
from schema_mechanism.strategies.evaluation import ReliabilityEvaluationStrategy
from schema_mechanism.util import Observable

# global constants

N_MACHINES = 2
N_STEPS = 5000


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

    def __init__(self,
                 machines: Sequence[Machine],
                 currency_to_play: int = 50,
                 currency_on_win: int = 100,
                 init_state: Optional[State] = None) -> None:
        super().__init__()

        if not machines:
            raise ValueError('Must supply at least one machine')

        self._machines = machines
        self._currency_to_play = currency_to_play
        self._currency_on_win = currency_on_win

        self._actions = [Action(a_str) for a_str in ['deposit', 'stand', 'play', 'withdraw']]

        self._sit_actions = [Action(f'sit({m_})') for m_ in self._machines]
        self._actions.extend(self._sit_actions)

        self._states = [
            sym_state('W'),  # agent's last play won
            sym_state('L'),  # agent's last play lost
            sym_state('S'),  # agent is standing
            sym_state('P'),  # money is deposited in current machine
            sym_state('B'),  # agent is broke (not enough money to play),
        ]

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
        self._current_state = self._init_state

        if self._init_state not in self._states:
            raise ValueError(f'initial state is an invalid state: {self._init_state}')

        self._winnings: int = 0

    @property
    def winnings(self) -> int:
        return self._winnings

    @winnings.setter
    def winnings(self, value: int) -> None:
        self._winnings = value

    @property
    def currency_to_play(self) -> int:
        return self._currency_to_play

    @currency_to_play.setter
    def currency_to_play(self, value: int) -> None:
        self._currency_to_play = value

    @property
    def currency_on_win(self) -> int:
        return self._currency_on_win

    @currency_on_win.setter
    def currency_on_win(self, value: int) -> None:
        self._currency_on_win = value

    @property
    def machines(self) -> Sequence[Machine]:
        return self._machines

    @property
    def actions(self) -> Sequence[Action]:
        return self._actions

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
                self.winnings -= self._currency_to_play
                m_ndx = self._machine_base_states.index(self._current_state)
                self._current_state = self._machine_play_states[m_ndx]
            elif action == Action('stand'):
                self._current_state = sym_state('S')

        # at machine, coin deposited
        elif self._current_state in self._machine_play_states:
            if action == Action('play'):
                m_ndx = self._machine_play_states.index(self._current_state)
                if self._machines[m_ndx].play()[0] == 'W':
                    self._winnings += self._currency_on_win
                    self._current_state = self._machine_win_states[m_ndx]
                else:
                    self._current_state = self._machine_lose_states[m_ndx]
            elif action == Action('stand'):
                self._current_state = sym_state('S')

        # at machine, just won
        elif self._current_state in self._machine_win_states:
            m_ndx = self._machine_win_states.index(self._current_state)
            if action == Action('deposit'):
                self.winnings -= self._currency_to_play
                self._current_state = self._machine_play_states[m_ndx]
            elif action == Action('stand'):
                self._current_state = sym_state('S')
            else:
                self._current_state = self._machine_base_states[m_ndx]

        # at machine, just lost
        elif self._current_state in self._machine_lose_states:
            m_ndx = self._machine_lose_states.index(self._current_state)
            if action == Action('deposit'):
                self.winnings -= self._currency_to_play
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

    parser.add_argument('--machines', type=int, required=False, default=N_MACHINES,
                        help=f'the id of the agent to which this action will be sent (default: {N_MACHINES})')
    parser.add_argument('--steps', type=int, required=False, default=N_STEPS,
                        help=f'the id of the agent to which this action will be sent (default: {N_STEPS})')

    return parser.parse_args()


def create_schema_mechanism(env: BanditEnvironment) -> SchemaMechanism:
    primitive_items = [
        sym_item('W', primitive_value=1.0),
        sym_item('L', primitive_value=-1.0),
        sym_item('P', primitive_value=-0.5),
    ]
    bare_schemas = [Schema(action=a) for a in env.actions]
    schema_memory = SchemaMemory(bare_schemas)
    schema_selection = SchemaSelection(
        select_strategy=RandomizeBestSelectionStrategy(AbsoluteDiffMatchStrategy(0.0)),
        value_strategies=[
            DefaultGoalPursuitEvaluationStrategy(),
            ReliabilityEvaluationStrategy(max_penalty=0.005),
            EpsilonGreedyEvaluationStrategy(epsilon=0.999,
                                            epsilon_min=0.05,
                                            decay_strategy=GeometricDecayStrategy(rate=0.999))
        ],
    )

    sm: SchemaMechanism = SchemaMechanism(
        items=primitive_items,
        schema_memory=schema_memory,
        schema_selection=schema_selection,
        global_params=GlobalParams(),
        global_stats=GlobalStats()
    )

    sm.params.set('backward_chains.update_frequency', 0.01)
    sm.params.set('backward_chains.max_len', 3)
    sm.params.set('delegated_value_helper.decay_rate', 0.0)
    sm.params.set('delegated_value_helper.discount_factor', 0.5)

    sm.params.set('learning_rate', 0.01)
    sm.params.set('reliability_threshold', 0.6)
    sm.params.set('composite_actions.learn.min_baseline_advantage', 1.0)

    # item correlation test used for determining relevance of extended context items
    sm.params.set('ext_context.correlation_test', FisherExactCorrelationTest)

    # thresholds for determining the relevance of extended result items
    #     from 0.0 [weakest correlation] to 1.0 [strongest correlation]
    sm.params.set('ext_context.positive_correlation_threshold', 0.95)

    # item correlation test used for determining relevance of extended result items
    sm.params.set('ext_result.correlation_test', CorrelationOnEncounter)

    # thresholds for determining the relevance of extended result items
    #     from 0.0 [weakest correlation] to 1.0 [strongest correlation]
    sm.params.set('ext_result.positive_correlation_threshold', 0.95)

    display_params()

    return sm


# TODO: create a general purpose run with args:
#       (1) environment
#       (2) schema mechanism
#       (3) pause callback
#       (4) start of episode callback
#       (5) end of episode callback
#       (6) start of step callback
#       (7) end of step callback

@run_decorator
def run():
    args = parse_args()

    machines = [Machine(str(id_), p_win=rng().uniform(0, 1)) for id_ in range(args.machines)]
    env = BanditEnvironment(machines)

    sm = create_schema_mechanism(env)

    for n in range(args.steps):
        info(f'winnings: {env.winnings}')
        info(f'state[{n}]: {env.current_state}')

        current_composite_schema = sm.schema_selection.pending_schema
        if current_composite_schema:
            info(f'active composite action schema: {current_composite_schema} ')

        selection_details = sm.select(env.current_state)

        schema = selection_details.selected
        action = schema.action
        effective_value = selection_details.effective_value
        terminated_composite_schemas = selection_details.terminated_pending

        if terminated_composite_schemas:
            i = 0
            for pending_details in terminated_composite_schemas:
                info(f'terminated schema [{i}]: {pending_details.schema}')
                info(f'termination status [{i}]: {pending_details.status}')
                info(f'selection state [{i}]: {pending_details.selection_state}')
                info(f'duration [{i}]: {pending_details.duration}')
            i += 1

        if is_paused():
            display_machine_info(machines)
            display_summary(sm)

            try:
                while is_paused():
                    sleep(0.1)
            except KeyboardInterrupt:
                pass

        info(f'selected schema[{n}]: {schema} [eff. value: {effective_value}]')

        state = env.step(action)

        sm.learn(selection_details, result_state=state)

    display_machine_info(machines)
    display_summary(sm)


def display_machine_info(machines):
    info(f'machines ({len(machines)}):')
    for m in machines:
        info(f'\t{repr(m)}')


if __name__ == '__main__':
    run()
