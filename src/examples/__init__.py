import argparse
import logging.config
from functools import partial
from functools import wraps
from pathlib import Path
from time import sleep
from time import time
from typing import Any
from typing import Callable
from typing import Optional
from typing import Protocol
from typing import Sequence

from pynput import keyboard

from examples.environments import Environment
from examples.environments import EpisodeSummary
from schema_mechanism.core import GlobalStats
from schema_mechanism.core import ReadOnlyItemPool
from schema_mechanism.core import Schema
from schema_mechanism.core import State
from schema_mechanism.core import get_global_stats
from schema_mechanism.modules import SchemaMechanism
from schema_mechanism.modules import SelectionDetails
from schema_mechanism.modules import save
from schema_mechanism.serialization import DEFAULT_ENCODING
from schema_mechanism.share import Predicate

logger = logging.getLogger(__name__)

# configure random seed
RANDOM_SEED = int(time())


def display_known_schemas(sm: SchemaMechanism, composite_only: bool = False) -> None:
    if composite_only:
        logger.info(f'composite schemas: (n = {sum(1 for s in sm.schema_memory if s.action.is_composite())})')
    else:
        logger.info(f'known schemas: (n = {len(sm.schema_memory)})')

    schemas = sorted([schema for schema in sm.schema_memory], key=lambda s: int(s.uid))
    for i, schema in enumerate(schemas):
        is_composite = schema.action.is_composite()

        if composite_only and not is_composite:
            continue

        logger.info(f'\t{i} ->  {str(schema)} [composite? {is_composite}][uid = {schema.uid}]')
        if is_composite:
            display_components(schema)


def display_components(schema: Schema) -> None:
    if not schema.action.is_composite():
        return

    logger.info(f'\t\tcontroller components: (n = {len(schema.action.controller.components)})')
    for component in schema.action.controller.components:
        logger.info(f'\t\t\t{str(component)} [proximity: {schema.action.controller.proximity(component)}]')


def display_item_values() -> None:
    global_stats: GlobalStats = get_global_stats()
    logger.info(f'baseline value: {global_stats.baseline_value}')

    pool = ReadOnlyItemPool()
    logger.info(f'known items: (n = {len(pool)})')
    for item in sorted(pool, key=lambda i: -i.delegated_value):
        logger.info(f'\t{repr(item)}')


def display_schema_info(schema: Schema) -> None:
    try:
        logger.info(f'schema [uid = {schema.uid}]: {schema}')
        logger.info(f'schema stats: {schema.stats}')
        if schema.extended_context:
            logger.info('EXTENDED_CONTEXT')
            logger.info(f'relevant items: {schema.extended_context.relevant_items}')
            for k, v in schema.extended_context.stats.items():
                logger.info(f'item: {k} -> {repr(v)}')

        if schema.extended_result:
            logger.info('EXTENDED_RESULT')
            logger.info(f'relevant items: {schema.extended_result.relevant_items}')
            for k, v in schema.extended_result.stats.items():
                logger.info(f'item: {k} -> {repr(v)}')

    except ValueError:
        return


def display_schema_memory(sm: SchemaMechanism) -> None:
    logger.debug(f'schema memory:')
    for line in str(sm.schema_memory).split('\n'):
        logger.info(line)


def display_summary(sm: SchemaMechanism) -> None:
    display_item_values()
    display_known_schemas(sm)
    display_schema_memory(sm)


_paused = False
_running = True


def is_paused() -> bool:
    return _paused


def toggle_paused() -> None:
    global _paused
    _paused = not _paused


def request_stop() -> None:
    global _running
    _running = False


def is_running() -> bool:
    return _running


ESC_KEY = keyboard.Key.esc
MINUS_KEY = keyboard.KeyCode(char='-')
UNDERSCORE_KEY = keyboard.KeyCode(char='_')
PLUS_KEY = keyboard.KeyCode(char='+')
EQUAL_KEY = keyboard.KeyCode(char='=')
SPACE_KEY = keyboard.Key.space


def default_on_press(key: keyboard.Key):
    global _paused, _running

    if key == SPACE_KEY:
        toggle_paused()
    if key == ESC_KEY:
        request_stop()


class RunnerCallback(Protocol):
    def __call__(self,
                 env: Environment,
                 schema_mechanism: SchemaMechanism,
                 step: int,
                 episode: int,
                 state: State,
                 selection_details: Optional[SelectionDetails] = None,
                 running: bool = True,
                 render_env: bool = True,
                 **kwargs,
                 ) -> None: ...


class SaveCallback:
    def __init__(self,
                 path: Path,
                 conditions: Sequence[Predicate],
                 encoding: str = DEFAULT_ENCODING,
                 ) -> None:
        self.path = path
        self.conditions = conditions
        self.encoding = encoding

    def __call__(self, schema_mechanism: SchemaMechanism, **kwargs) -> None:
        if any(condition(**kwargs) for condition in self.conditions):
            save(
                modules=[schema_mechanism],
                path=self.path,
                encoding=self.encoding,
            )


class SaveWhenTerminal:
    def __call__(self, env: Environment, **kwargs) -> bool:
        return True if env.is_terminal() else False


class SaveEveryNSteps(Predicate):
    def __init__(self, n: int):
        self.n = n
        self.cumulative_steps: int = 0

    def __call__(self, **kwargs) -> bool:
        self.cumulative_steps += 1
        return self.cumulative_steps % self.n == 0


class SaveEveryNEpisodes(Predicate):
    def __init__(self, n: int):
        self.n = n
        self.cumulative_episodes: int = 0

    def __call__(self, env: Environment, **kwargs) -> bool:
        if env.is_terminal():
            self.cumulative_episodes += 1

        return self.cumulative_steps % self.n == 0 if self.cumulative_episodes > 0 else False


save_when_terminal: Predicate = SaveWhenTerminal()


def save_every_n_steps(n: int) -> Predicate:
    return SaveEveryNSteps(n)


def save_every_n_episodes(n: int) -> Predicate:
    return SaveEveryNSteps(n)


def no_op_callback(*_unused_args, **_unused_kwargs) -> None:
    pass


def default_display_on_step(env: Environment,
                            schema_mechanism: SchemaMechanism,
                            step: int,
                            episode: int,
                            state: State,
                            selection_details: Optional[SelectionDetails] = None,
                            render_env: bool = True,
                            *_unused_args,
                            **_unused_kwargs,
                            ) -> None:
    progress_indicator: str = f'{episode}:{step}'

    if render_env:
        print(env.render())

    schema: Schema = selection_details.selected
    effective_value = selection_details.effective_value

    logger.debug(f'selection state [{progress_indicator}]: {selection_details.selection_state}')
    logger.debug(f'selected schema [{progress_indicator}]: {schema} [eff. value: {effective_value}]')
    logger.debug(f'resulting state [{progress_indicator}]: {state}')

    current_composite_schema = schema_mechanism.schema_selection.pending_schema
    if current_composite_schema:
        logger.debug(f'active composite action schema [{progress_indicator}]: {current_composite_schema} ')

    terminated_composite_schemas = selection_details.terminated_pending
    if terminated_composite_schemas:
        logger.debug(f'terminated schemas:')
        for i, pending_details in enumerate(terminated_composite_schemas):
            logger.debug(f'schema [{i}]: {pending_details.schema}')
            logger.debug(f'selection state [{i}]: {pending_details.selection_state}')
            logger.debug(f'status [{i}]: {pending_details.status}')
            logger.debug(f'duration [{i}]: {pending_details.duration}')


def default_display_on_pause(env: Environment,
                             schema_mechanism: SchemaMechanism,
                             step: int,
                             state: State,
                             selection_details: Optional[SelectionDetails] = None,
                             render_env: bool = True,
                             *_unused_args,
                             **_unused_kwargs,
                             ) -> None:
    display_item_values()
    display_known_schemas(schema_mechanism)
    display_schema_memory(schema_mechanism)

    if render_env:
        logger.info(env.render())

    schema: Schema = selection_details.selected
    effective_value = selection_details.effective_value

    logger.info(f'selection state [{step}]: {selection_details.selection_state}')
    logger.info(f'selected schema [{step}]: {schema} [eff. value: {effective_value}]')
    logger.info(f'resulting state [{step}]: {state}')

    # wait for keypress to resume operation
    logger.warning('Environment paused!')
    try:
        while is_paused():
            sleep(0.01)
    except KeyboardInterrupt:
        pass
    logger.warning('Environment resumed by keypress!')


class Runner(Protocol):
    def __call__(self,
                 env: Environment,
                 schema_mechanism: SchemaMechanism,
                 max_steps: int,
                 episode: int = 0,
                 on_step: Sequence[RunnerCallback] = None,
                 on_pause: Sequence[RunnerCallback] = None,
                 render_env: bool = True
                 ) -> EpisodeSummary: ...


def _runner(_run: Runner, on_press: RunnerCallback = None) -> Callable[..., Runner]:
    @wraps(_run)
    def _run_wrapper(**kwargs) -> Any:
        listener = keyboard.Listener(on_press=on_press)
        listener.start()

        start = time()
        results = _run(**kwargs)
        end = time()

        logger.info(f'elapsed time: {end - start}s')

        return results

    return _run_wrapper


runner: Callable[..., Runner] = partial(_runner, on_press=default_on_press)


@runner
def run_episode(env: Environment,
                schema_mechanism: SchemaMechanism,
                max_steps: int,
                episode: int = 0,
                on_step: Sequence[RunnerCallback] = None,
                on_pause: Sequence[RunnerCallback] = None,
                render_env: bool = True
                ) -> Any:
    on_step = on_step if on_step else [default_display_on_step]
    on_pause = on_pause if on_pause else [default_display_on_pause]

    # initialize the world
    state, is_terminal = env.reset()

    # episode loop
    for step in range(max_steps):
        if is_terminal or not is_running():
            break

        if render_env:
            print(env.render())

        selection_details = schema_mechanism.select(state)
        state, is_terminal = env.step(selection_details.selected.action)
        schema_mechanism.learn(selection_details, result_state=state)

        for on_step_callback in on_step:
            on_step_callback(
                env=env,
                schema_mechanism=schema_mechanism,
                step=step,
                episode=episode,
                state=state,
                running=is_running(),
                selection_details=selection_details,
                render_env=render_env,
            )

        if is_paused():
            for on_pause_callback in on_pause:
                on_pause_callback(
                    env=env,
                    schema_mechanism=schema_mechanism,
                    step=step,
                    episode=episode,
                    state=state,
                    running=is_running(),
                    selection_details=selection_details,
                    render_env=render_env
                )

    return env.episode_summary


DEFAULT_OPTIMIZER_PRUNER = 'median'
DEFAULT_OPTIMIZER_SAMPLER = 'tpe'
DEFAULT_OPTIMIZER_TRIALS = 50
DEFAULT_OPTIMIZER_RUNS_PER_TRIAL = 10
DEFAULT_SHOW_PROGRESS_BAR = False
DEFAULT_OPTIMIZER_USE_DATABASE = False

DEFAULT_EPISODES = 1
DEFAULT_STEPS_PER_EPISODE = 500
DEFAULT_SAVE_FREQUENCY = 2500


def parse_run_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        '--steps', type=int, required=False, default=DEFAULT_STEPS_PER_EPISODE,
        help=f'the number of time steps per episode (default: {DEFAULT_STEPS_PER_EPISODE})'
    )
    parser.add_argument(
        '--episodes', type=int, required=False, default=DEFAULT_EPISODES,
        help=f'the number of episodes (default: {DEFAULT_EPISODES})'
    )
    parser.add_argument(
        '--greedy', required=False, action="store_true", default=False,
        help=f'use a greedy evaluation strategy for schema selection (default: False)'
    )

    return parser


def parse_optimizer_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        '--optimize',
        required=False,
        action="store_true",
        default=False,
        help=f'execute a study to optimize hyper-parameters (default: {False})'
    )
    parser.add_argument(
        '--optimizer_sampler',
        type=str,
        default=DEFAULT_OPTIMIZER_SAMPLER,
        choices=['random', 'tpe', 'skopt'],
        help=f'sampler to use when optimizing hyper-parameters (default: {DEFAULT_OPTIMIZER_SAMPLER})'
    )
    parser.add_argument(
        '--optimizer_pruner',
        type=str,
        default=DEFAULT_OPTIMIZER_PRUNER,
        choices=['halving', 'median', 'none'],
        help=f'pruner to use when optimizing hyper-parameters (default: {DEFAULT_OPTIMIZER_PRUNER})'
    )
    parser.add_argument(
        '--optimizer_trials',
        metavar='N',
        type=int,
        required=False,
        default=DEFAULT_OPTIMIZER_TRIALS,
        help=f'the number of optimizer trials to run (default: {DEFAULT_OPTIMIZER_TRIALS})'
    )
    parser.add_argument(
        '--optimizer_runs_per_trial',
        metavar='N',
        type=int,
        required=False,
        default=DEFAULT_OPTIMIZER_RUNS_PER_TRIAL,
        help=f'the number of separate environment runs executed per optimizer trial '
             + f'(default: {DEFAULT_OPTIMIZER_RUNS_PER_TRIAL})'
    )
    parser.add_argument(
        '--optimizer_study_name',
        type=str,
        default=None,
        help=f'name used for the optimizer study (default: {None})'
    )
    parser.add_argument(
        '--optimizer_use_db',
        action="store_true",
        default=DEFAULT_OPTIMIZER_USE_DATABASE,
        help=f'saves results from optimizer study to a database to allow resuming an interrupted study ' +
             f'(default: {DEFAULT_OPTIMIZER_USE_DATABASE})'
    )
    parser.add_argument(
        '--show_progress_bar',
        action="store_true",
        default=DEFAULT_SHOW_PROGRESS_BAR,
        help=f'show a progress bar for optimizer on standard error (default: {DEFAULT_SHOW_PROGRESS_BAR})'
    )

    return parser


def parser_serialization_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        '--save_path',
        metavar='FILE',
        type=Path,
        required=False,
        default=None,
        help='the filepath to a directory that will be used to save the learned schema mechanism'
    )
    parser.add_argument(
        '--save_frequency',
        metavar='STEPS',
        type=int,
        required=False,
        default=DEFAULT_SAVE_FREQUENCY,
        help='the save frequency in number of steps taken by agent'
    )
    parser.add_argument(
        '--load_path',
        metavar='FILE',
        type=Path,
        required=False,
        default=None,
        help='the filepath to the manifest of a learned schema mechanism'
    )

    return parser
