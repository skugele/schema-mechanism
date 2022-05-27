import json
from typing import Any
from typing import Callable
from typing import Type

from schema_mechanism.core import Action
from schema_mechanism.core import CompositeItem
from schema_mechanism.core import GlobalStats
from schema_mechanism.core import StateAssertion
from schema_mechanism.core import SymbolicItem


def encode_action(action: Action) -> dict:
    if not isinstance(action, Action):
        TypeError(f"Object of type '{type(action)}' is not JSON serializable")

    attrs = {
        '__type__': 'Action',
        'label': action.label,
    }
    return attrs


def encode_symbolic_item(item: SymbolicItem) -> dict:
    if not isinstance(item, SymbolicItem):
        raise TypeError(f"Object of type '{type(item)}' is not JSON serializable")

    attrs = {
        '__type__': 'SymbolicItem',
        'source': item.source,
        'primitive_value': item.primitive_value,
        'delegated_value': item.delegated_value,
    }
    return attrs


def encode_composite_item(item: CompositeItem) -> dict:
    if not isinstance(item, CompositeItem):
        raise TypeError(f"Object of type '{type(item)}' is not JSON serializable")

    attrs = {
        '__type__': 'CompositeItem',
        'source': list(item.source),
        'primitive_value': item.primitive_value,
        'delegated_value': item.delegated_value
    }
    return attrs


def encode_state_assertion(state_assertion: StateAssertion) -> dict:
    if not isinstance(state_assertion, StateAssertion):
        raise TypeError(f"Object of type '{type(state_assertion)}' is not JSON serializable")

    attrs = {
        '__type__': 'StateAssertion',
        'items': list(state_assertion.items),
    }
    return attrs


def encode_global_stats(global_stats: GlobalStats) -> dict:
    if not isinstance(global_stats, GlobalStats):
        raise TypeError(f"Object of type '{type(global_stats)}' is not JSON serializable")

    attrs = {
        '__type__': 'GlobalStats',
        'n': global_stats.n,
        'baseline_value': global_stats.baseline_value,
    }
    return attrs


# def encode_schema(schema: Schema):
#     if not isinstance(schema, Schema):
#         raise TypeError(f"Object of type '{type(schema)}' is not JSON serializable")
#
#     attrs = {
#         '__type__': 'Schema',
#         'source': list(item.source),
#         'primitive_value': item.primitive_value,
#         'delegated_value': item.delegated_value
#     }
#     return attrs


encoder_map: dict[Type, Callable] = {
    Action: encode_action,
    SymbolicItem: encode_symbolic_item,
    CompositeItem: encode_composite_item,
    StateAssertion: encode_state_assertion,
    GlobalStats: encode_global_stats,
}


def _encode(obj: Any):
    encoder = encoder_map.get(type(obj))
    if not encoder:
        raise TypeError(f"Object of type '{type(obj)}' is not JSON serializable")
    return encoder(obj)


def encode(obj: Any, /) -> str:
    return json.dumps(obj, default=_encode)
