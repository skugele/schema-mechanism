import logging.config
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
        logger.info(f'composite schemas: (n = {sum(1 for s in sm.schema_memory if s.action.is_composite())})')
    else:
        logger.info(f'known schemas: (n = {len(sm.schema_memory)})')

    for s in sm.schema_memory:
        is_composite = s.action.is_composite()

        if composite_only and not is_composite:
            continue

        logger.info(f'\t{str(s)} [composite? {is_composite}]')
        if is_composite:
            display_components(s)


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
        logger.info(f'schema: {schema}')
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
    logger.info(f'schema memory:')
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
        run(*args, **kwargs)
        end = time()

        logger.info(f'elapsed time: {end - start}s')

    return _run_wrapper
