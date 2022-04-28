import random
from time import time
from typing import Callable

from pynput import keyboard

from schema_mechanism.core import GlobalStats
from schema_mechanism.core import ReadOnlyItemPool
from schema_mechanism.core import Schema
from schema_mechanism.modules import SchemaMechanism
from schema_mechanism.share import GlobalParams
from schema_mechanism.share import Verbosity
from schema_mechanism.share import info
from schema_mechanism.share import log_level

RANDOM_SEED = 8675309

# For reproducibility, we we also need to set PYTHONHASHSEED=RANDOM_SEED in the environment
random.seed(RANDOM_SEED)

# set random seed for reproducibility
params = GlobalParams()
params.set('rng_seed', RANDOM_SEED)
params.set('verbosity', Verbosity.INFO)


def display_known_schemas(sm: SchemaMechanism, composite_only: bool = False) -> None:
    if composite_only:
        info(f'composite schemas: (n = {sum(1 for s in sm.schema_memory if s.action.is_composite())})')
    else:
        info(f'known schemas: (n = {len(sm.schema_memory)})')

    for s in sm.schema_memory:
        is_composite = s.action.is_composite()

        if composite_only and not is_composite:
            continue

        info(f'\t{str(s)} [composite? {is_composite}]')
        if is_composite:
            display_components(s)


def display_components(schema: Schema) -> None:
    if not schema.action.is_composite():
        return

    info(f'\t\tcontroller components: (n = {len(schema.action.controller.components)})')
    for component in schema.action.controller.components:
        info(f'\t\t\t{str(component)} [proximity: {schema.action.controller.proximity(component)}]')


def display_item_values() -> None:
    info(f'baseline value: {GlobalStats().baseline_value}')

    pool = ReadOnlyItemPool()
    info(f'known items: (n = {len(pool)})')
    for item in sorted(pool, key=lambda i: -i.delegated_value):
        info(f'\t{repr(item)}')


def display_schema_info(schema: Schema) -> None:
    try:
        info(f'schema: {schema}')
        info(f'schema stats: {schema.stats}')
        if schema.extended_context:
            info('EXTENDED_CONTEXT')
            info(f'relevant items: {schema.extended_context.relevant_items}')
            for k, v in schema.extended_context.stats.items():
                info(f'item: {k} -> {repr(v)}')

        if schema.extended_result:
            info('EXTENDED_RESULT')
            info(f'relevant items: {schema.extended_result.relevant_items}')
            for k, v in schema.extended_result.stats.items():
                info(f'item: {k} -> {repr(v)}')
    except ValueError:
        return


def display_schema_memory(sm: SchemaMechanism) -> None:
    info(f'schema memory:')
    for line in str(sm.schema_memory).split('\n'):
        info(line)


def display_summary(sm: SchemaMechanism) -> None:
    display_item_values()
    display_known_schemas(sm)
    display_schema_memory(sm)


def decrease_log_level() -> None:
    try:
        verbosity = Verbosity(log_level() - 1)
        GlobalParams().set('verbosity', verbosity)
    except ValueError:
        pass


def increase_log_level() -> None:
    try:
        verbosity = Verbosity(log_level() + 1)
        GlobalParams().set('verbosity', verbosity)
    except ValueError:
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

        info(f'elapsed time: {end - start}s')

    return _run_wrapper
