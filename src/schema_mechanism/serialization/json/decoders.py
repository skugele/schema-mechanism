import json
from typing import Any
from typing import Callable

from schema_mechanism.core import Action
from schema_mechanism.core import CompositeItem
from schema_mechanism.core import ItemPool
from schema_mechanism.core import SymbolicItem


def decode_action(obj_dict: dict) -> Any:
    return Action(**obj_dict)


def decode_symbolic_item(obj_dict: dict) -> Any:
    return ItemPool().get(obj_dict['source'], item_type=SymbolicItem, **obj_dict)


def decode_composite_item(obj_dict: dict) -> Any:
    return ItemPool().get(frozenset(obj_dict['source']), item_type=CompositeItem, **obj_dict)


decoder_map: dict[str, Callable] = {
    'Action': decode_action,
    'SymbolicItem': decode_symbolic_item,
    'CompositeItem': decode_composite_item
}


def _decode(obj_dict: dict) -> Any:
    # the __type__ element is added to all objects serialized using a custom encoder
    obj_type = obj_dict.pop('__type__') if '__type__' in obj_dict else None

    decoder = decoder_map.get(obj_type)

    # use custom decoder if one exists; otherwise, return object dictionary for default json decoder to attempt decoding
    return decoder(obj_dict) if decoder else obj_dict


def decode(obj: Any, /) -> Any:
    return json.loads(obj, object_hook=_decode)
