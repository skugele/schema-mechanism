import argparse
import logging.config
from collections.abc import Sequence
from pathlib import Path
from time import sleep
from time import time
from typing import Optional

import optuna
from optuna.integration import SkoptSampler
from optuna.pruners import MedianPruner
from optuna.pruners import SuccessiveHalvingPruner
from optuna.samplers import RandomSampler
from optuna.samplers import TPESampler
from pandas import DataFrame

from examples import RANDOM_SEED
from examples import display_schema_info
from examples import display_summary
from examples import is_paused
from examples import parse_optimizer_args
from examples import parser_serialization_args
from examples import run_decorator
from examples.optimizer.multi_arm_bandits import get_objective_function
from schema_mechanism.core import Action
from schema_mechanism.core import State
from schema_mechanism.core import get_global_params
from schema_mechanism.func_api import sym_item
from schema_mechanism.func_api import sym_schema
from schema_mechanism.func_api import sym_state
from schema_mechanism.modules import SchemaMechanism
from schema_mechanism.modules import init
from schema_mechanism.modules import load
from schema_mechanism.modules import save
from schema_mechanism.parameters import GlobalParams
from schema_mechanism.serialization import DEFAULT_ENCODING
from schema_mechanism.strategies.evaluation import DefaultEvaluationStrategy
from schema_mechanism.util import Observable
from schema_mechanism.util import rng
from schema_mechanism.util import set_random_seed

logger = logging.getLogger('examples.environments.multi_arm_bandits')

# For reproducibility, we we also need to set PYTHONHASHSEED=RANDOM_SEED in the environment
set_random_seed(RANDOM_SEED)

# global constants
N_MACHINES = 16
N_STEPS = 8000


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

        self.machines = machines
        self.currency_to_play = currency_to_play
        self.currency_on_win = currency_on_win

        self.actions = [Action(a_str) for a_str in ['deposit', 'stand', 'play']]

        self._sit_actions = [Action(f'sit({machine})') for machine in self.machines]
        self.actions.extend(self._sit_actions)

        self.states = [
            sym_state('W'),  # agent's last play won
            sym_state('L'),  # agent's last play lost
            sym_state('S'),  # agent is standing
            sym_state('P'),  # money is deposited in current machine
        ]

        # add machine related states
        self._machine_base_states = [sym_state(str(machine)) for machine in self.machines]
        self._machine_play_states = [sym_state(f'{machine},P') for machine in self.machines]
        self._machine_win_states = [sym_state(f'{machine},W') for machine in self.machines]
        self._machine_lose_states = [sym_state(f'{machine},L') for machine in self.machines]

        self.states.extend([
            *self._machine_base_states,
            *self._machine_play_states,
            *self._machine_win_states,
            *self._machine_lose_states
        ])

        self._init_state = init_state or sym_state('S')
        self._current_state = self._init_state

        if self._init_state not in self.states:
            raise ValueError(f'initial state is an invalid state: {self._init_state}')

        self.winnings: int = 0

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
                self.winnings -= self.currency_to_play
                m_ndx = self._machine_base_states.index(self._current_state)
                self._current_state = self._machine_play_states[m_ndx]
            elif action == Action('stand'):
                self._current_state = sym_state('S')

        # at machine, coin deposited
        elif self._current_state in self._machine_play_states:
            if action == Action('play'):
                m_ndx = self._machine_play_states.index(self._current_state)
                if self.machines[m_ndx].play()[0] == 'W':
                    self.winnings += self.currency_on_win
                    self._current_state = self._machine_win_states[m_ndx]
                else:
                    self._current_state = self._machine_lose_states[m_ndx]
            elif action == Action('stand'):
                self._current_state = sym_state('S')

        # at machine, just won
        elif self._current_state in self._machine_win_states:
            m_ndx = self._machine_win_states.index(self._current_state)
            if action == Action('deposit'):
                self.winnings -= self.currency_to_play
                self._current_state = self._machine_play_states[m_ndx]
            elif action == Action('stand'):
                self._current_state = sym_state('S')
            else:
                self._current_state = self._machine_base_states[m_ndx]

        # at machine, just lost
        elif self._current_state in self._machine_lose_states:
            m_ndx = self._machine_lose_states.index(self._current_state)
            if action == Action('deposit'):
                self.winnings -= self.currency_to_play
                self._current_state = self._machine_play_states[m_ndx]
            elif action == Action('stand'):
                self._current_state = sym_state('S')
            else:
                self._current_state = self._machine_base_states[m_ndx]

        return self._current_state

    def reset(self) -> State:
        self.winnings = 0
        self._current_state = self._init_state
        return self._current_state


def parse_env_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--machines', type=int, required=False, default=N_MACHINES,
                        help=f'the id of the agent to which this action will be sent (default: {N_MACHINES})')
    parser.add_argument('--steps', type=int, required=False, default=N_STEPS,
                        help=f'the id of the agent to which this action will be sent (default: {N_STEPS})')

    return parser


def parse_args():
    """ Parses command line arguments.
    :return: argparse parser with parsed command line args
    """
    parser = argparse.ArgumentParser(description='Multi-Armed Bandit Example Environment (Schema Mechanism)')

    parser = parse_env_args(parser)
    parser = parse_optimizer_args(parser)
    parser = parser_serialization_args(parser)

    return parser.parse_args()


def init_params() -> GlobalParams:
    params = get_global_params()

    params.set('learning_rate', 0.001)

    params.set('composite_actions.update_frequency', 0.01)
    params.set('composite_actions.backward_chains.max_length', 2)
    params.set('composite_actions.min_baseline_advantage', 0.4)

    params.set('schema.reliability_threshold', 0.8)

    # item correlation test used for determining relevance of extended context items
    params.set('ext_context.correlation_test', 'DrescherCorrelationTest')

    # thresholds for determining the relevance of extended result items
    #     from 0.0 [weakest correlation] to 1.0 [strongest correlation]
    params.set('ext_context.positive_correlation_threshold', 0.9)

    # item correlation test used for determining relevance of extended result items
    params.set('ext_result.correlation_test', 'CorrelationOnEncounter')

    # thresholds for determining the relevance of extended result items
    #     from 0.0 [weakest correlation] to 1.0 [strongest correlation]
    params.set('ext_result.positive_correlation_threshold', 1.0)

    return params


def load_schema_mechanism(model_filepath: Path) -> SchemaMechanism:
    schema_mechanism: Optional[SchemaMechanism] = None
    try:
        schema_mechanism = load(model_filepath)
        logger.info(f'Successfully schema mechanism from {str(model_filepath)}')
    except FileNotFoundError as e:
        logger.error(e)

    return schema_mechanism


def create_schema_mechanism(env: BanditEnvironment) -> SchemaMechanism:
    primitive_items = [
        sym_item('W', primitive_value=1.0),
        sym_item('L', primitive_value=-1.0),
        sym_item('P', primitive_value=-0.5),
    ]

    schema_mechanism = init(
        items=primitive_items,
        actions=env.actions,
        global_params=init_params()
    )

    schema_mechanism.schema_selection.evaluation_strategy = DefaultEvaluationStrategy(
        epsilon=0.9999,
        epsilon_min=0.025,
        epsilon_decay_rate=0.9999,
        reliability_max_penality=0.6,
        pending_focus_max_value=0.9,
        pending_focus_decay_rate=0.4
    )

    return schema_mechanism


def calculate_score(env: BanditEnvironment) -> float:
    return env.winnings


@run_decorator
def run(env: BanditEnvironment,
        schema_mechanism: SchemaMechanism,
        n_steps: int,
        save_path: Path = None) -> float:
    env.reset()

    for n in range(n_steps):
        logger.debug(f'winnings: {env.winnings}')
        logger.debug(f'state[{n}]: {env.current_state}')

        current_composite_schema = schema_mechanism.schema_selection.pending_schema
        if current_composite_schema:
            logger.debug(f'active composite action schema: {current_composite_schema} ')

        selection_details = schema_mechanism.select(env.current_state)

        schema = selection_details.selected
        action = schema.action
        effective_value = selection_details.effective_value
        terminated_composite_schemas = selection_details.terminated_pending

        if terminated_composite_schemas:
            i = 0
            for pending_details in terminated_composite_schemas:
                logger.debug(f'terminated schema [{i}]: {pending_details.schema}')
                logger.debug(f'termination status [{i}]: {pending_details.status}')
                logger.debug(f'selection state [{i}]: {pending_details.selection_state}')
                logger.debug(f'duration [{i}]: {pending_details.duration}')
            i += 1

        if is_paused():
            display_machine_info(env.machines)
            display_summary(schema_mechanism)
            display_schema_info(sym_schema('/play/'))

            try:
                while is_paused():
                    sleep(0.1)
            except KeyboardInterrupt:
                pass

        logger.debug(f'selected schema[{n}]: {schema} [eff. value: {effective_value}]')

        state = env.step(action)

        schema_mechanism.learn(selection_details=selection_details, result_state=state)

    display_machine_info(env.machines)
    display_summary(schema_mechanism)

    if save_path:
        save(
            modules=[schema_mechanism],
            path=save_path,
            encoding=DEFAULT_ENCODING,
        )

    return calculate_score(env)


def optimize(env: BanditEnvironment,
             sampler: str,
             pruner: str,
             n_trials: int,
             n_steps_per_trial: int,
             n_runs_per_trial: int) -> DataFrame:
    seed = int(time())

    if sampler == 'random':
        sampler = RandomSampler(seed=seed)
    elif sampler == 'tpe':
        sampler = TPESampler(n_startup_trials=5, seed=seed)
    elif sampler == 'skopt':
        sampler = SkoptSampler(skopt_kwargs={'base_estimator': "GP", 'acq_func': 'gp_hedge'})
    else:
        raise ValueError('Unknown sampler: {}'.format(sampler))

    if pruner == 'halving':
        pruner = SuccessiveHalvingPruner(min_resource=1, reduction_factor=4, min_early_stopping_rate=0)
    elif pruner == 'median':
        pruner = MedianPruner(n_startup_trials=5)
    elif pruner == 'none':
        # Do not prune
        pruner = MedianPruner(n_startup_trials=n_trials)
    else:
        raise ValueError('Unknown pruner: {}'.format(pruner))

    env_id = 'multi-arm_bandits'
    session_id = int(time())
    study_name = f'{env_id}-{session_id}-optimizer_study'
    storage = f'sqlite:///{study_name}.db'

    study = optuna.create_study(study_name=study_name,
                                storage=storage,
                                load_if_exists=True,
                                sampler=sampler,
                                pruner=pruner)

    try:
        objective = get_objective_function(
            env=env,
            run=run,
            n_steps_per_trial=n_steps_per_trial,
            n_runs_per_trial=n_runs_per_trial
        )
        study.optimize(objective, n_trials)
    except KeyboardInterrupt:
        pass

    logger.info(f'Number of finished trials: {len(study.trials)}')
    logger.info('Best trial:')
    best_trial = study.best_trial

    logger.info(f'\tValue: {-best_trial.value}')
    logger.info('\tParams: ')
    for key, value in best_trial.params.items():
        logger.info(f'\t\t{key}: {value}')

    return study.trials_dataframe()


def display_machine_info(machines):
    logger.info(f'machines ({len(machines)}):')
    for m in machines:
        logger.info(f'\t{repr(m)}')


def main():
    # configure logger
    logging.config.fileConfig('config/logging.conf')

    args = parse_args()

    machines = [Machine(str(id_), p_win=rng().uniform(0, 1)) for id_ in range(args.machines)]
    env = BanditEnvironment(machines)

    if args.optimize:
        data_frame = optimize(
            env=env,
            sampler=args.optimizer_sampler,
            pruner=args.optimizer_pruner,
            n_trials=args.optimizer_trials,
            n_steps_per_trial=args.steps,
            n_runs_per_trial=args.optimizer_runs_per_trial,
        )

        report_name = f'optimizer_results_{int(time())}-{args.optimizer_sampler}-{args.optimizer_pruner}.csv'
        report_path = f'local/optimize/multi_arm_bandits/{report_name}'
        logger.info("Writing report to {}".format(report_path))
        data_frame.to_csv(report_path)
    else:
        load_path: Path = args.load_path
        save_path: Path = args.save_path

        schema_mechanism = load_schema_mechanism(load_path) if load_path else None
        if not schema_mechanism:
            schema_mechanism = create_schema_mechanism(env)

        winnings = run(env=env, schema_mechanism=schema_mechanism, n_steps=args.steps, save_path=save_path)
        logger.info(f'Agent\'s winnings: {winnings}')


if __name__ == '__main__':
    main()
