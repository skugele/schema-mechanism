import functools
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
from schema_mechanism.core import ItemPool
from schema_mechanism.core import Schema
from schema_mechanism.core import SchemaStats
from schema_mechanism.core import SchemaTree
from schema_mechanism.core import SchemaTreeNode
from schema_mechanism.core import StateAssertion
from schema_mechanism.core import SymbolicItem
from schema_mechanism.modules import SchemaMemory


def decode_action(obj_dict: dict, object_registry: dict[int, Any], **kwargs) -> Action:
    return Action(**obj_dict)


def decode_composite_action(obj_dict: dict, object_registry: dict[int, Any], **kwargs) -> CompositeAction:
    return CompositeAction(**obj_dict)


def decode_controller(obj_dict: dict, object_registry: dict[int, Any], **kwargs) -> Controller:
    components: set[Schema] = set()
    if 'components' in obj_dict:
        encoded_components: list[int] = obj_dict.pop('components')
        components = set(
            map_list_from_unique_ids_to_list_of_objects(
                encoded_components,
                object_registry=object_registry)
        )

    descendants: set[Schema] = set()
    if 'descendants' in obj_dict:
        encoded_descendants: list[int] = obj_dict.pop('descendants')
        descendants = set(
            map_list_from_unique_ids_to_list_of_objects(
                encoded_descendants,
                object_registry=object_registry)
        )

    proximity_map: dict[Schema, float] = dict()
    if 'proximity_map' in obj_dict:
        encoded_proximity_map = obj_dict.pop('proximity_map')
        proximity_map = map_dict_from_unique_id_to_object_keys(
            encoded_proximity_map,
            object_registry=object_registry
        )

    total_cost_map: dict[Schema, float] = dict()
    if 'total_cost_map' in obj_dict:
        encoded_total_cost_map: dict[int, float] = obj_dict.pop('total_cost_map')
        total_cost_map = map_dict_from_unique_id_to_object_keys(
            encoded_total_cost_map,
            object_registry=object_registry
        )

    return Controller(
        components=components,
        descendants=descendants,
        proximity_map=proximity_map,
        total_cost_map=total_cost_map,
        **obj_dict
    )


def decode_symbolic_item(obj_dict: dict, object_registry: dict[int, Any], **kwargs) -> SymbolicItem:
    source = obj_dict.pop('source')
    return ItemPool().get(source, item_type=SymbolicItem, **obj_dict)


def decode_composite_item(obj_dict: dict, object_registry: dict[int, Any], **kwargs) -> CompositeItem:
    source = obj_dict.pop('source')
    return ItemPool().get(frozenset(source), item_type=CompositeItem, **obj_dict)


def decode_state_assertion(obj_dict: dict, object_registry: dict[int, Any], **kwargs) -> StateAssertion:
    encoded_items = obj_dict.pop('items')
    items = map_list_from_unique_ids_to_list_of_objects(
        encoded_items,
        object_registry=object_registry
    )
    return StateAssertion(items=items, **obj_dict)


def decode_global_stats(obj_dict: dict, object_registry: dict[int, Any], **kwargs) -> GlobalStats:
    return GlobalStats(**obj_dict)


def decode_schema_stats(obj_dict: dict, object_registry: dict[int, Any], **kwargs) -> SchemaStats:
    return SchemaStats(**obj_dict)


def decode_ec_item_stats(obj_dict: dict, object_registry: dict[int, Any], **kwargs) -> ECItemStats:
    return ECItemStats(**obj_dict)


def decode_er_item_stats(obj_dict: dict, object_registry: dict[int, Any], **kwargs) -> ERItemStats:
    return ERItemStats(**obj_dict)


def decode_frozen_ec_item_stats(obj_dict: dict, object_registry: dict[int, Any], **kwargs) -> FrozenECItemStats:
    return FROZEN_EC_ITEM_STATS


def decode_frozen_er_item_stats(obj_dict: dict, object_registry: dict[int, Any], **kwargs) -> FrozenERItemStats:
    return FROZEN_ER_ITEM_STATS


def decode_extended_context(obj_dict: dict, object_registry: dict[int, Any], **kwargs) -> ExtendedContext:
    stats = map_dict_from_unique_id_to_object_keys(
        obj_dict.pop('stats'),
        object_registry=object_registry
    )

    suppressed_items = map_list_from_unique_ids_to_list_of_objects(
        obj_dict.pop('suppressed_items'),
        object_registry=object_registry
    )

    relevant_items = map_list_from_unique_ids_to_list_of_objects(
        obj_dict.pop('relevant_items'),
        object_registry=object_registry
    )

    return ExtendedContext(
        stats=stats,
        suppressed_items=suppressed_items,
        relevant_items=relevant_items,
    )


def decode_extended_result(obj_dict: dict, object_registry: dict[int, Any], **kwargs) -> ExtendedResult:
    stats = map_dict_from_unique_id_to_object_keys(
        obj_dict.pop('stats'),
        object_registry=object_registry
    )

    suppressed_items = map_list_from_unique_ids_to_list_of_objects(
        obj_dict.pop('suppressed_items'),
        object_registry=object_registry
    )

    relevant_items = map_list_from_unique_ids_to_list_of_objects(
        obj_dict.pop('relevant_items'),
        object_registry=object_registry
    )

    return ExtendedResult(
        stats=stats,
        suppressed_items=suppressed_items,
        relevant_items=relevant_items,
    )


def decode_schema(obj_dict: dict, object_registry: dict[int, Any], **kwargs) -> Schema:
    return Schema(**obj_dict)


def decode_schema_tree_node(obj_dict: dict, object_registry: dict[int, Any], **kwargs) -> SchemaTreeNode:
    schemas_satisfied_by = map_list_from_unique_ids_to_list_of_objects(
        obj_dict.pop('schemas_satisfied_by'),
        object_registry=object_registry
    )

    schemas_would_satisfy = map_list_from_unique_ids_to_list_of_objects(
        obj_dict.pop('schemas_would_satisfy'),
        object_registry=object_registry
    )

    parent: SchemaTreeNode = object_registry[obj_dict.pop('parent')]
    children: list[SchemaTreeNode] = map_list_from_unique_ids_to_list_of_objects(
        obj_dict.pop('children'),
        object_registry=object_registry
    )

    schema_tree_node = SchemaTreeNode(
        schemas_satisfied_by=schemas_satisfied_by,
        schemas_would_satisfy=schemas_would_satisfy,
        **obj_dict
    )

    schema_tree_node.parent = parent
    schema_tree_node.children = tuple(children)

    return schema_tree_node


def decode_schema_tree(obj_dict: dict, object_registry: dict[int, Any], **kwargs) -> SchemaTree:
    root: SchemaTreeNode = object_registry[obj_dict.pop('root')]
    nodes_map: dict[StateAssertion, SchemaTreeNode] = dict()
    if 'nodes_map' in obj_dict:
        encoded_nodes_map: dict[int, int] = obj_dict.pop('nodes_map')
        for encoded_state_assertion, encoded_node in encoded_nodes_map.items():
            state_assertion: StateAssertion = object_registry[int(encoded_state_assertion)]
            node: SchemaTreeNode = object_registry[int(encoded_node)]

            nodes_map[state_assertion] = node

    return SchemaTree(root=root, nodes_map=nodes_map)


def decode_schema_memory(obj_dict: dict, object_registry: dict[int, Any], **kwargs) -> SchemaMemory:
    return SchemaMemory(**obj_dict)


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
    'SchemaTreeNode': decode_schema_tree_node,
    'SchemaTree': decode_schema_tree,
    'SchemaMemory': decode_schema_memory,
}


def map_list_from_unique_ids_to_list_of_objects(
        unique_ids: list[int],
        object_registry: dict[int, Any]) -> list[Any]:
    return [object_registry[int(uid)] for uid in unique_ids]


def map_dict_from_unique_id_to_object_keys(
        unique_id_dict: dict[int, Any],
        object_registry: dict[int, Any]) -> dict[Any, Any]:
    return {object_registry[int(uid)]: value for uid, value in unique_id_dict.items()}


def _decode(obj_dict: dict, **kwargs) -> Any:
    # the __type__ element is added to all objects serialized using a custom encoder
    obj_type = obj_dict.pop('__type__') if '__type__' in obj_dict else None

    decoder = decoder_map.get(obj_type)

    # use custom decoder if one exists; otherwise, return object dictionary for default json decoder to attempt decoding
    return decoder(obj_dict, **kwargs) if decoder else obj_dict


def decode(obj: Any, /, **kwargs) -> Any:
    return json.loads(obj, object_hook=functools.partial(_decode, **kwargs))
