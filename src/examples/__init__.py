import argparse
import logging.config
from pathlib import Path
from time import time
from typing import Callable

from pynput import keyboard

from schema_mechanism.core import GlobalStats
from schema_mechanism.core import ReadOnlyItemPool
from schema_mechanism.core import Schema
from schema_mechanism.core import get_global_stats
from schema_mechanism.modules import SchemaMechanism

logger = logging.getLogger(__name__)

# configure random seed
RANDOM_SEED = 8675309


def display_known_schemas(sm: SchemaMechanism, composite_only: bool = False) -> None:
    if composite_only:
        logger.debug(f'composite schemas: (n = {sum(1 for s in sm.schema_memory if s.action.is_composite())})')
    else:
        logger.debug(f'known schemas: (n = {len(sm.schema_memory)})')

    schemas = sorted([schema for schema in sm.schema_memory], key=lambda s: int(s.uid))
    for i, schema in enumerate(schemas):
        is_composite = schema.action.is_composite()

        if composite_only and not is_composite:
            continue

        logger.debug(f'\t{i} ->  {str(schema)} [composite? {is_composite}][uid = {schema.uid}]')
        if is_composite:
            display_components(schema)


def display_components(schema: Schema) -> None:
    if not schema.action.is_composite():
        return

    logger.debug(f'\t\tcontroller components: (n = {len(schema.action.controller.components)})')
    for component in schema.action.controller.components:
        logger.debug(f'\t\t\t{str(component)} [proximity: {schema.action.controller.proximity(component)}]')


def display_item_values() -> None:
    global_stats: GlobalStats = get_global_stats()
    logger.debug(f'baseline value: {global_stats.baseline_value}')

    pool = ReadOnlyItemPool()
    logger.debug(f'known items: (n = {len(pool)})')
    for item in sorted(pool, key=lambda i: -i.delegated_value):
        logger.debug(f'\t{repr(item)}')


def display_schema_info(schema: Schema) -> None:
    try:
        logger.debug(f'schema [uid = {schema.uid}]: {schema}')
        logger.debug(f'schema stats: {schema.stats}')
        if schema.extended_context:
            logger.debug('EXTENDED_CONTEXT')
            logger.debug(f'relevant items: {schema.extended_context.relevant_items}')
            for k, v in schema.extended_context.stats.items():
                logger.debug(f'item: {k} -> {repr(v)}')

        if schema.extended_result:
            logger.debug('EXTENDED_RESULT')
            logger.debug(f'relevant items: {schema.extended_result.relevant_items}')
            for k, v in schema.extended_result.stats.items():
                logger.debug(f'item: {k} -> {repr(v)}')
    except ValueError:
        return


def display_schema_memory(sm: SchemaMechanism) -> None:
    logger.debug(f'schema memory:')
    for line in str(sm.schema_memory).split('\n'):
        logger.debug(line)


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


def on_press(key):
    global _paused, _running

    if key == SPACE_KEY:
        toggle_paused()
    if key == ESC_KEY:
        request_stop()
    if key in (MINUS_KEY, UNDERSCORE_KEY):
        decrease_log_level()
    if key in (PLUS_KEY, EQUAL_KEY):
        increase_log_level()


def run_decorator(run: Callable):
    def _run_wrapper(*args, **kwargs):
        listener = keyboard.Listener(on_press=on_press)
        listener.start()

        start = time()
        results = run(*args, **kwargs)
        end = time()

        logger.info(f'elapsed time: {end - start}s')

        return results

    return _run_wrapper


DEFAULT_OPTIMIZER_PRUNER = 'median'
DEFAULT_OPTIMIZER_SAMPLER = 'tpe'
DEFAULT_OPTIMIZER_TRIALS = 50
DEFAULT_OPTIMIZER_RUNS_PER_TRIAL = 10
DEFAULT_SHOW_PROGRESS_BAR = False
DEFAULT_OPTIMIZER_USE_DATABASE = False


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
