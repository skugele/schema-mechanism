import json
from typing import Any
from typing import Callable
from typing import Type
from typing import Union

from schema_mechanism.core import Action
from schema_mechanism.core import CompositeItem
from schema_mechanism.core import SymbolicItem


def encode_action(action: Action):
    if not isinstance(action, Action):
        TypeError(f"Object of type '{type(action)}' is not JSON serializable")

    attrs = {
        '__type__': 'Action',
        'label': action.label,
    }
    return attrs


def encode_symbolic_item(item: SymbolicItem):
    if not isinstance(item, SymbolicItem):
        raise TypeError(f"Object of type '{type(item)}' is not JSON serializable")

    attrs = {
        '__type__': 'SymbolicItem',
        'source': item.source,
        'primitive_value': item.primitive_value,
        'delegated_value': item.delegated_value,
    }
    return attrs


def encode_composite_item(item: CompositeItem):
    if not isinstance(item, CompositeItem):
        raise TypeError(f"Object of type '{type(item)}' is not JSON serializable")

    attrs = {
        '__type__': 'CompositeItem',
        'source': list(item.source),
        'primitive_value': item.primitive_value,
        'delegated_value': item.delegated_value
    }
    return attrs


def encode_set_as_list(set_: Union[set, frozenset]) -> list:
    if not (isinstance(set_, set) or isinstance(set_, frozenset)):
        raise TypeError(f"Object of type '{type(set_)}' is not JSON serializable")

    return list(set_)


encoder_map: dict[Type, Callable] = {
    Action: encode_action,
    SymbolicItem: encode_symbolic_item,
    CompositeItem: encode_composite_item,
}


def _encode(obj: Any):
    encoder = encoder_map.get(type(obj))
    if not encoder:
        raise TypeError(f"Object of type '{type(obj)}' is not JSON serializable")
    return encoder(obj)


def encode(obj: Any, /) -> str:
    return json.dumps(obj, default=_encode)
