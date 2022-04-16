import random

from schema_mechanism.core import GlobalStats
from schema_mechanism.core import ReadOnlyItemPool
from schema_mechanism.core import Schema
from schema_mechanism.modules import SchemaMechanism
from schema_mechanism.share import GlobalParams
from schema_mechanism.share import Verbosity
from schema_mechanism.share import info

RANDOM_SEED = 8675309

# For reproducibility, we we also need to set PYTHONHASHSEED=RANDOM_SEED in the environment
random.seed(RANDOM_SEED)

# set random seed for reproducibility
params = GlobalParams()
params.set('rng_seed', RANDOM_SEED)
params.set('verbosity', Verbosity.INFO)


def display_known_schemas(sm: SchemaMechanism) -> None:
    info(f'known schemas (n = {len(sm.schema_memory)})')
    for s in sm.schema_memory:
        composite = s.action.is_composite()
        info(f'\t{str(s)} [composite? {composite}]')
        if composite:
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
