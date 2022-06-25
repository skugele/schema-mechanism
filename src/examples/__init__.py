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

logger = logging.getLogger(__name__)

# configure random seed
RANDOM_SEED = 8675309


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


# TODO: fix this!
def decrease_log_level() -> None:
    pass


# TODO: fix this!
def increase_log_level() -> None:
    pass


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
    if key in (MINUS_KEY, UNDERSCORE_KEY):
        decrease_log_level()
    if key in (PLUS_KEY, EQUAL_KEY):
        increase_log_level()


class RunInformationDisplayer(Protocol):
    def __call__(self,
                 env: Environment,
                 schema_mechanism: SchemaMechanism,
                 step: int,
                 state: State,
                 selection_details: Optional[SelectionDetails] = None,
                 render_env: bool = True,
                 ) -> None: ...


def default_display_on_step(env: Environment,
                            schema_mechanism: SchemaMechanism,
                            step: int,
                            state: State,
                            selection_details: Optional[SelectionDetails] = None,
                            ) -> None:
    schema: Schema = selection_details.selected
    effective_value = selection_details.effective_value

    logger.debug(f'selection state [{step}]: {selection_details.selection_state}')
    logger.debug(f'selected schema [{step}]: {schema} [eff. value: {effective_value}]')
    logger.debug(f'resulting state [{step}]: {state}')

    current_composite_schema = schema_mechanism.schema_selection.pending_schema
    if current_composite_schema:
        logger.debug(f'active composite action schema [{step}]: {current_composite_schema} ')

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


class Runner(Protocol):
    def __call__(self,
                 env: Environment,
                 schema_mechanism: SchemaMechanism,
                 max_steps: int,
                 on_step: RunInformationDisplayer,
                 on_pause: RunInformationDisplayer,
                 render_env: bool = True
                 ) -> EpisodeSummary: ...


def _runnable(run: Runner,
              on_step: RunInformationDisplayer,
              on_pause: RunInformationDisplayer,
              on_press: Callable[[keyboard.Key], None]):
    @wraps(run)
    def _run_wrapper(on_step=on_step, on_pause=on_pause, **kwargs) -> Any:
        listener = keyboard.Listener(on_press=on_press)
        listener.start()

        start = time()
        results = run(**kwargs, on_step=on_step, on_pause=on_pause)
        end = time()

        logger.info(f'elapsed time: {end - start}s')

        return results

    return _run_wrapper


runner = partial(
    _runnable,
    on_step=default_display_on_step,
    on_pause=default_display_on_pause,
    on_press=default_on_press
)


@runner
def run(env: Environment,
        schema_mechanism: SchemaMechanism,
        max_steps: int,
        on_step: RunInformationDisplayer = None,
        on_pause: RunInformationDisplayer = None,
        render_env: bool = True
        ) -> Any:
    # initialize the world
    state, is_terminal = env.reset()

    # episode loop
    for step in range(1, max_steps + 1):
        if is_terminal or not is_running():
            break

        if render_env:
            print(env.render())

        selection_details = schema_mechanism.select(state)
        state, is_terminal = env.step(selection_details.selected.action)
        schema_mechanism.learn(selection_details, result_state=state)

        if is_paused():
            # displays information about this step
            if on_pause:
                on_pause(
                    env=env,
                    schema_mechanism=schema_mechanism,
                    step=step,
                    state=state,
                    selection_details=selection_details,
                )

            try:
                while is_paused():
                    sleep(0.1)
            except KeyboardInterrupt:
                pass

        # displays status information at the beginning of each step
        if on_step:
            on_step(
                env=env,
                schema_mechanism=schema_mechanism,
                step=step,
                state=state,
                selection_details=selection_details,
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


def parse_run_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        '--steps', type=int, required=False, default=DEFAULT_STEPS_PER_EPISODE,
        help=f'the id of the agent to which this action will be sent (default: {DEFAULT_STEPS_PER_EPISODE})'
    )
    parser.add_argument(
        '--episodes', type=int, required=False, default=DEFAULT_EPISODES,
        help=f'the number of episodes (default: {DEFAULT_EPISODES})'
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
        '--load_path',
        metavar='FILE',
        type=Path,
        required=False,
        default=None,
        help='the filepath to the manifest of a learned schema mechanism'
    )

    return parser
