import random

from schema_mechanism.core import GlobalParams
from schema_mechanism.core import GlobalStats
from schema_mechanism.core import ReadOnlyItemPool
from schema_mechanism.core import Schema
from schema_mechanism.core import Verbosity
from schema_mechanism.core import info
from schema_mechanism.modules import SchemaMechanism

RANDOM_SEED = 8675309

# For reproducibility, we we also need to set PYTHONHASHSEED=8675309 in the environment
random.seed(RANDOM_SEED)

# set random seed for reproducibility
params = GlobalParams()
params.set('rng_seed', RANDOM_SEED)
params.set('verbosity', Verbosity.INFO)


def display_known_schemas(sm: SchemaMechanism) -> None:
    info(f'number of known schemas ({len(sm.known_schemas)})')
    for s in sm.known_schemas:
        info(f'\t{str(s)}')


def display_item_values() -> None:
    info(f'baseline value: {GlobalStats().baseline_value}')

    pool = ReadOnlyItemPool()
    info(f'number of known items: ({len(pool)})')
    for i in sorted(pool, key=lambda i: -i.delegated_value):
        info(f'\t{repr(i)}')


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


def display_schema_memory(sm: SchemaMechanism) -> None:
    info(f'schema memory:')
    for line in str(sm.schema_memory).split('\n'):
        info(line)


def display_summary(sm: SchemaMechanism) -> None:
    display_item_values()
    display_known_schemas(sm)
    display_schema_memory(sm)
