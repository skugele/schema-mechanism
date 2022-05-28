import json
from typing import Any
from typing import Callable

from schema_mechanism.core import Action
from schema_mechanism.core import CompositeAction
from schema_mechanism.core import CompositeItem
from schema_mechanism.core import Controller
from schema_mechanism.core import ECItemStats
from schema_mechanism.core import ERItemStats
from schema_mechanism.core import ExtendedContext
from schema_mechanism.core import ExtendedResult
from schema_mechanism.core import FROZEN_EC_ITEM_STATS
from schema_mechanism.core import FROZEN_ER_ITEM_STATS
from schema_mechanism.core import FrozenECItemStats
from schema_mechanism.core import FrozenERItemStats
from schema_mechanism.core import GlobalStats
from schema_mechanism.core import Item
from schema_mechanism.core import ItemPool
from schema_mechanism.core import Schema
from schema_mechanism.core import SchemaStats
from schema_mechanism.core import StateAssertion
from schema_mechanism.core import SymbolicItem


def decode_action(obj_dict: dict) -> Action:
    return Action(**obj_dict)


def decode_composite_action(obj_dict: dict) -> CompositeAction:
    return CompositeAction(**obj_dict)


def decode_controller(obj_dict: dict) -> Controller:
    proximity_map: dict[Schema, float] = dict()
    if 'proximity_map' in obj_dict:
        encoded_proximity_map: dict[str, float] = obj_dict.pop('proximity_map')
        for encoded_key, value in encoded_proximity_map.items():
            decoded_key_as_str: str = encoded_key.replace('\\', '')

            schema: Schema = decode(decoded_key_as_str)
            proximity_map[schema] = value

    total_cost_map: dict[Schema, float] = dict()
    if 'total_cost_map' in obj_dict:
        encoded_total_cost_map: dict[str, float] = obj_dict.pop('total_cost_map')
        for encoded_key, value in encoded_total_cost_map.items():
            decoded_key_as_str: str = encoded_key.replace('\\', '')

            schema: Schema = decode(decoded_key_as_str)
            total_cost_map[schema] = value

    return Controller(
        proximity_map=proximity_map,
        total_cost_map=total_cost_map,
        **obj_dict
    )


def decode_symbolic_item(obj_dict: dict) -> SymbolicItem:
    source = obj_dict.pop('source')
    return ItemPool().get(source, item_type=SymbolicItem, **obj_dict)


def decode_composite_item(obj_dict: dict) -> CompositeItem:
    source = obj_dict.pop('source')
    return ItemPool().get(frozenset(source), item_type=CompositeItem, **obj_dict)


def decode_state_assertion(obj_dict: dict) -> StateAssertion:
    items = obj_dict.pop('items')
    return StateAssertion(items=items, **obj_dict)


def decode_global_stats(obj_dict: dict) -> GlobalStats:
    return GlobalStats(**obj_dict)


def decode_schema_stats(obj_dict: dict) -> SchemaStats:
    return SchemaStats(**obj_dict)


def decode_ec_item_stats(obj_dict: dict) -> ECItemStats:
    return ECItemStats(**obj_dict)


def decode_er_item_stats(obj_dict: dict) -> ERItemStats:
    return ERItemStats(**obj_dict)


def decode_frozen_ec_item_stats(obj_dict: dict) -> FrozenECItemStats:
    return FROZEN_EC_ITEM_STATS


def decode_frozen_er_item_stats(obj_dict: dict) -> FrozenERItemStats:
    return FROZEN_ER_ITEM_STATS


def decode_extended_context(obj_dict: dict) -> ExtendedContext:
    stats: dict[Item, ECItemStats] = dict()
    if 'stats' in obj_dict:
        encoded_stats: dict[str, ECItemStats] = obj_dict.pop('stats')
        for encoded_key, item_stats in encoded_stats.items():
            decoded_key_as_str: str = encoded_key.replace('\\', '')

            item: Item = decode(decoded_key_as_str)
            stats[item] = item_stats

    return ExtendedContext(stats=stats, **obj_dict)


def decode_extended_result(obj_dict: dict) -> ExtendedResult:
    stats: dict[Item, ERItemStats] = dict()
    if 'stats' in obj_dict:
        encoded_stats: dict[str, ERItemStats] = obj_dict.pop('stats')
        for encoded_key, item_stats in encoded_stats.items():
            decoded_key_as_str: str = encoded_key.replace('\\', '')

            item: Item = decode(decoded_key_as_str)
            stats[item] = item_stats

    return ExtendedResult(stats=stats, **obj_dict)


def decode_schema(obj_dict: dict) -> Schema:
    return Schema(**obj_dict)


decoder_map: dict[str, Callable] = {
    'Action': decode_action,
    'CompositeAction': decode_composite_action,
    'Controller': decode_controller,
    'SymbolicItem': decode_symbolic_item,
    'CompositeItem': decode_composite_item,
    'StateAssertion': decode_state_assertion,
    'GlobalStats': decode_global_stats,
    'SchemaStats': decode_schema_stats,
    'ECItemStats': decode_ec_item_stats,
    'ERItemStats': decode_er_item_stats,
    'FrozenECItemStats': decode_frozen_ec_item_stats,
    'FrozenERItemStats': decode_frozen_er_item_stats,
    'ExtendedContext': decode_extended_context,
    'ExtendedResult': decode_extended_result,
    'Schema': decode_schema,
}


def _decode(obj_dict: dict) -> Any:
    # the __type__ element is added to all objects serialized using a custom encoder
    obj_type = obj_dict.pop('__type__') if '__type__' in obj_dict else None

    decoder = decoder_map.get(obj_type)

    # use custom decoder if one exists; otherwise, return object dictionary for default json decoder to attempt decoding
    return decoder(obj_dict) if decoder else obj_dict


def decode(obj: Any, /) -> Any:
    return json.loads(obj, object_hook=_decode)
