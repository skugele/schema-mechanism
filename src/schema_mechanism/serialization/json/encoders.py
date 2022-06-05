import functools
import json
from typing import Any
from typing import Callable
from typing import Type

from schema_mechanism.core import Action
from schema_mechanism.core import CompositeAction
from schema_mechanism.core import CompositeItem
from schema_mechanism.core import Controller
from schema_mechanism.core import ECItemStats
from schema_mechanism.core import ERItemStats
from schema_mechanism.core import ExtendedContext
from schema_mechanism.core import ExtendedResult
from schema_mechanism.core import FrozenECItemStats
from schema_mechanism.core import FrozenERItemStats
from schema_mechanism.core import GlobalStats
from schema_mechanism.core import Schema
from schema_mechanism.core import SchemaStats
from schema_mechanism.core import SchemaTree
from schema_mechanism.core import SchemaTreeNode
from schema_mechanism.core import StateAssertion
from schema_mechanism.core import SymbolicItem
from schema_mechanism.modules import SchemaMemory


def encode_action(
        action: Action,
        object_registry: dict[int, Any] = None,
        **kwargs) -> dict:
    if not isinstance(action, Action):
        TypeError(f"Object of type '{type(action)}' is not JSON serializable")

    attrs = {
        '__type__': 'Action',
        'label': action.label,
    }
    return attrs


def encode_composite_action(
        composite_action: CompositeAction,
        object_registry: dict[int, Any] = None,
        **kwargs) -> dict:
    if not isinstance(composite_action, CompositeAction):
        TypeError(f"Object of type '{type(composite_action)}' is not JSON serializable")

    attrs = {
        '__type__': 'CompositeAction',
        'label': composite_action.label,
        'goal_state': composite_action.goal_state,
    }
    return attrs


def encode_controller(
        controller: Controller,
        object_registry: dict[int, Any] = None,
        **kwargs) -> dict:
    if not isinstance(controller, Controller):
        TypeError(f"Object of type '{type(controller)}' is not JSON serializable")

    components = list(controller.components)
    descendants = list(controller.descendants)
    proximity_map = controller.proximity_map
    total_cost_map = controller.total_cost_map

    if object_registry is not None:
        for schema in components:
            object_registry[schema.uid] = schema

        for schema in descendants:
            object_registry[schema.uid] = schema

        components = map_list_to_unique_ids(components)
        descendants = map_list_to_unique_ids(descendants)
        proximity_map = map_dict_keys_to_unique_ids(proximity_map)
        total_cost_map = map_dict_keys_to_unique_ids(total_cost_map)

    attrs = {
        '__type__': 'Controller',
        'goal_state': controller.goal_state,
        'proximity_map': proximity_map,
        'total_cost_map': total_cost_map,
        'components': components,
        'descendants': descendants,
    }
    return attrs


def encode_symbolic_item(
        item: SymbolicItem,
        object_registry: dict[int, Any] = None,
        **kwargs) -> dict:
    if not isinstance(item, SymbolicItem):
        raise TypeError(f"Object of type '{type(item)}' is not JSON serializable")

    attrs = {
        '__type__': 'SymbolicItem',
        'source': item.source,
        'primitive_value': item.primitive_value,
        'delegated_value': item.delegated_value,
    }
    return attrs


def encode_composite_item(
        item: CompositeItem,
        object_registry: dict[int, Any] = None,
        **kwargs) -> dict:
    if not isinstance(item, CompositeItem):
        raise TypeError(f"Object of type '{type(item)}' is not JSON serializable")

    attrs = {
        '__type__': 'CompositeItem',
        'source': list(item.source),
        'primitive_value': item.primitive_value,
        'delegated_value': item.delegated_value
    }
    return attrs


def encode_state_assertion(
        state_assertion: StateAssertion,
        object_registry: dict[int, Any] = None,
        **kwargs) -> dict:
    if not isinstance(state_assertion, StateAssertion):
        raise TypeError(f"Object of type '{type(state_assertion)}' is not JSON serializable")

    items = list(state_assertion.items)

    if object_registry is not None:
        for item in items:
            object_registry[item.uid] = item

        items = map_list_to_unique_ids(items)

    attrs = {
        '__type__': 'StateAssertion',
        'items': items,
    }
    return attrs


def encode_global_stats(
        global_stats: GlobalStats,
        object_registry: dict[int, Any] = None,
        **kwargs) -> dict:
    if not isinstance(global_stats, GlobalStats):
        raise TypeError(f"Object of type '{type(global_stats)}' is not JSON serializable")

    attrs = {
        '__type__': 'GlobalStats',
        'n': global_stats.n,
        'baseline_value': global_stats.baseline_value,
    }
    return attrs


def encode_schema_stats(
        schema_stats: SchemaStats,
        object_registry: dict[int, Any] = None,
        **kwargs) -> dict:
    if not isinstance(schema_stats, SchemaStats):
        raise TypeError(f"Object of type '{type(schema_stats)}' is not JSON serializable")

    attrs = {
        '__type__': 'SchemaStats',
        'n': schema_stats.n,
        'n_success': schema_stats.n_success,
        'n_activated': schema_stats.n_activated,
    }
    return attrs


def encode_ec_item_stats(
        ec_item_stats: ECItemStats,
        object_registry: dict[int, Any] = None,
        **kwargs) -> dict:
    if not isinstance(ec_item_stats, ECItemStats):
        raise TypeError(f"Object of type '{type(ec_item_stats)}' is not JSON serializable")

    attrs = {
        '__type__': 'ECItemStats',
        'n_success_and_on': ec_item_stats.n_success_and_on,
        'n_success_and_off': ec_item_stats.n_success_and_off,
        'n_fail_and_on': ec_item_stats.n_fail_and_on,
        'n_fail_and_off': ec_item_stats.n_fail_and_off,
    }
    return attrs


def encode_er_item_stats(
        er_item_stats: ERItemStats,
        object_registry: dict[int, Any] = None,
        **kwargs) -> dict:
    if not isinstance(er_item_stats, ERItemStats):
        raise TypeError(f"Object of type '{type(er_item_stats)}' is not JSON serializable")

    attrs = {
        '__type__': 'ERItemStats',
        'n_on_and_activated': er_item_stats.n_on_and_activated,
        'n_on_and_not_activated': er_item_stats.n_on_and_not_activated,
        'n_off_and_activated': er_item_stats.n_off_and_activated,
        'n_off_and_not_activated': er_item_stats.n_off_and_not_activated,
    }
    return attrs


def encode_frozen_ec_item_stats(
        ec_item_stats: FrozenECItemStats,
        object_registry: dict[int, Any] = None,
        **kwargs) -> dict:
    if not isinstance(ec_item_stats, FrozenECItemStats):
        raise TypeError(f"Object of type '{type(ec_item_stats)}' is not JSON serializable")

    attrs = {
        '__type__': 'FrozenECItemStats',
    }
    return attrs


def encode_frozen_er_item_stats(
        er_item_stats: FrozenERItemStats,
        object_registry: dict[int, Any] = None,
        **kwargs) -> dict:
    if not isinstance(er_item_stats, FrozenERItemStats):
        raise TypeError(f"Object of type '{type(er_item_stats)}' is not JSON serializable")

    attrs = {
        '__type__': 'FrozenERItemStats',
    }
    return attrs


def encode_extended_context(
        extended_context: ExtendedContext,
        object_registry: dict[int, Any] = None,
        **kwargs) -> dict:
    if not isinstance(extended_context, ExtendedContext):
        raise TypeError(f"Object of type '{type(extended_context)}' is not JSON serializable")

    suppressed_items = list(extended_context.suppressed_items)
    relevant_items = list(extended_context.relevant_items)

    if object_registry is not None:
        for item in suppressed_items:
            object_registry[item.uid] = item

        for item in relevant_items:
            object_registry[item.uid] = item

        suppressed_items = map_list_to_unique_ids(suppressed_items)
        relevant_items = map_list_to_unique_ids(relevant_items)

    stats: dict[int, Any] = map_dict_keys_to_unique_ids(extended_context.stats)

    attrs = {
        '__type__': 'ExtendedContext',
        'suppressed_items': suppressed_items,
        'relevant_items': relevant_items,
        'stats': stats
    }
    return attrs


def encode_extended_result(
        extended_result: ExtendedResult,
        object_registry: dict[int, Any] = None,
        **kwargs) -> dict:
    if not isinstance(extended_result, ExtendedResult):
        raise TypeError(f"Object of type '{type(extended_result)}' is not JSON serializable")

    suppressed_items = list(extended_result.suppressed_items)
    relevant_items = list(extended_result.relevant_items)

    if object_registry is not None:
        for item in suppressed_items:
            object_registry[item.uid] = item

        for item in relevant_items:
            object_registry[item.uid] = item

        suppressed_items = map_list_to_unique_ids(suppressed_items)
        relevant_items = map_list_to_unique_ids(relevant_items)

    stats: dict[int, Any] = map_dict_keys_to_unique_ids(extended_result.stats)

    attrs = {
        '__type__': 'ExtendedResult',
        'suppressed_items': suppressed_items,
        'relevant_items': relevant_items,
        'stats': stats
    }
    return attrs


def encode_schema(
        schema: Schema,
        object_registry: dict[int, Any] = None,
        **kwargs) -> dict:
    if not isinstance(schema, Schema):
        raise TypeError(f"Object of type '{type(schema)}' is not JSON serializable")

    if object_registry:
        object_registry[schema.uid] = schema

    attrs = {
        '__type__': 'Schema',
        'action': schema.action,
        'context': schema.context,
        'result': schema.result,
        'extended_context': schema.extended_context,
        'extended_result': schema.extended_result,
        'schema_stats': schema.stats,
        'overriding_conditions': schema.overriding_conditions,
        'avg_duration': schema.avg_duration,
        'cost': schema.cost,
        'creation_time': schema.creation_time,

    }
    return attrs


def encode_schema_tree_node(
        schema_tree_node: SchemaTreeNode,
        object_registry: dict[int, Any] = None,
        **kwargs) -> dict:
    if not isinstance(schema_tree_node, SchemaTreeNode):
        raise TypeError(f"Object of type '{type(schema_tree_node)}' is not JSON serializable")

    schemas_satisfied_by = list(schema_tree_node.schemas_satisfied_by)
    schemas_would_satisfy = list(schema_tree_node.schemas_would_satisfy)
    parent = schema_tree_node.parent
    children = list(schema_tree_node.children)

    if object_registry is not None:
        for schema in schema_tree_node.schemas_satisfied_by:
            object_registry[schema.uid] = schema

        for schema in schema_tree_node.schemas_would_satisfy:
            object_registry[schema.uid] = schema

        for node in schema_tree_node.children:
            object_registry[node.uid] = node

        if parent:
            object_registry[parent.uid] = parent

        schemas_satisfied_by = map_list_to_unique_ids(schemas_satisfied_by)
        schemas_would_satisfy = map_list_to_unique_ids(schemas_would_satisfy)
        children = map_list_to_unique_ids(children)
        parent = parent.uid if parent else None

    attrs = {
        '__type__': 'SchemaTreeNode',
        'context': schema_tree_node.context,
        'schemas_satisfied_by': schemas_satisfied_by,
        'schemas_would_satisfy': schemas_would_satisfy,
        'parent': parent,
        'children': children,
        'label': schema_tree_node.label,
    }
    return attrs


def encode_schema_tree(
        schema_tree: SchemaTree,
        object_registry: dict[int, Any] = None,
        **kwargs) -> dict:
    if not isinstance(schema_tree, SchemaTree):
        raise TypeError(f"Object of type '{type(schema_tree)}' is not JSON serializable")

    root = schema_tree.root

    object_registry[root.uid] = root

    encoded_root = schema_tree.root.uid
    encoded_nodes_map: dict[int, int] = dict()

    for state_assertion, node in schema_tree.nodes_map.items():
        object_registry[state_assertion.uid] = state_assertion
        object_registry[node.uid] = node

        encoded_nodes_map[state_assertion.uid] = node.uid

    attrs = {
        '__type__': 'SchemaTree',
        'root': encoded_root,
        'nodes_map': encoded_nodes_map,
    }
    return attrs


def encode_schema_memory(
        schema_memory: SchemaMemory,
        object_registry: dict[int, Any] = None,
        **kwargs) -> dict:
    schema_collection = schema_memory.schema_collection

    attrs = {
        '__type__': 'SchemaMemory',
        'schema_collection': schema_collection,
    }
    return attrs


encoder_map: dict[Type, Callable] = {
    # built-ins
    int: int,
    float: float,
    set: list,
    frozenset: list,

    # custom objects
    Action: encode_action,
    CompositeAction: encode_composite_action,
    Controller: encode_controller,
    SymbolicItem: encode_symbolic_item,
    CompositeItem: encode_composite_item,
    StateAssertion: encode_state_assertion,
    GlobalStats: encode_global_stats,
    SchemaStats: encode_schema_stats,
    ECItemStats: encode_ec_item_stats,
    ERItemStats: encode_er_item_stats,
    FrozenECItemStats: encode_frozen_ec_item_stats,
    FrozenERItemStats: encode_frozen_er_item_stats,
    ExtendedContext: encode_extended_context,
    ExtendedResult: encode_extended_result,
    Schema: encode_schema,
    SchemaTreeNode: encode_schema_tree_node,
    SchemaTree: encode_schema_tree,
    SchemaMemory: encode_schema_memory,
}


def map_list_to_unique_ids(objects: list[Any]) -> list[int]:
    return [o.uid for o in objects]


def map_dict_keys_to_unique_ids(object_dict: dict) -> dict[int, Any]:
    return {key.uid: value for key, value in object_dict.items()}


def _encode(obj: Any, **kwargs):
    encoder = encoder_map.get(type(obj))
    if not encoder:
        raise TypeError(f"Object of type '{type(obj)}' is not JSON serializable")
    return encoder(obj, **kwargs)


def encode(obj: Any, /, **kwargs) -> str:
    return json.dumps(obj, default=functools.partial(_encode, **kwargs))
