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
from schema_mechanism.core import SchemaTreeNode
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


def encode_composite_action(composite_action: CompositeAction) -> dict:
    if not isinstance(composite_action, CompositeAction):
        TypeError(f"Object of type '{type(composite_action)}' is not JSON serializable")

    attrs = {
        '__type__': 'CompositeAction',
        'label': composite_action.label,
        'goal_state': composite_action.goal_state,
    }
    return attrs


def encode_controller(controller: Controller) -> dict:
    if not isinstance(controller, Controller):
        TypeError(f"Object of type '{type(controller)}' is not JSON serializable")

    encoded_proximity_map: dict[str, float] = dict()
    if controller.proximity_map:
        for key, value in controller.proximity_map.items():
            encoded_key = str(encode(key))
            encoded_proximity_map[encoded_key] = value

    encoded_total_cost_map: dict[str, float] = dict()
    if controller.total_cost_map:
        for key, value in controller.total_cost_map.items():
            encoded_key = str(encode(key))
            encoded_total_cost_map[encoded_key] = value

    attrs = {
        '__type__': 'Controller',
        'goal_state': controller.goal_state,
        'proximity_map': encoded_proximity_map,
        'total_cost_map': encoded_total_cost_map,
        'components': list(controller.components),
        'descendants': list(controller.descendants),
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


def encode_schema_stats(schema_stats: SchemaStats) -> dict:
    if not isinstance(schema_stats, SchemaStats):
        raise TypeError(f"Object of type '{type(schema_stats)}' is not JSON serializable")

    attrs = {
        '__type__': 'SchemaStats',
        'n': schema_stats.n,
        'n_success': schema_stats.n_success,
        'n_activated': schema_stats.n_activated,
    }
    return attrs


def encode_ec_item_stats(ec_item_stats: ECItemStats) -> dict:
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


def encode_er_item_stats(er_item_stats: ERItemStats) -> dict:
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


def encode_frozen_ec_item_stats(ec_item_stats: FrozenECItemStats) -> dict:
    if not isinstance(ec_item_stats, FrozenECItemStats):
        raise TypeError(f"Object of type '{type(ec_item_stats)}' is not JSON serializable")

    attrs = {
        '__type__': 'FrozenECItemStats',
    }
    return attrs


def encode_frozen_er_item_stats(er_item_stats: FrozenERItemStats) -> dict:
    if not isinstance(er_item_stats, FrozenERItemStats):
        raise TypeError(f"Object of type '{type(er_item_stats)}' is not JSON serializable")

    attrs = {
        '__type__': 'FrozenERItemStats',
    }
    return attrs


def encode_extended_context(extended_context: ExtendedContext) -> dict:
    if not isinstance(extended_context, ExtendedContext):
        raise TypeError(f"Object of type '{type(extended_context)}' is not JSON serializable")

    encoded_stats = dict()
    if extended_context.stats:
        for key, value in extended_context.stats.items():
            encoded_key = str(encode(key))
            encoded_stats[encoded_key] = value

    attrs = {
        '__type__': 'ExtendedContext',
        'suppressed_items': list(extended_context.suppressed_items),
        'relevant_items': list(extended_context.relevant_items),
        'stats': encoded_stats
    }
    return attrs


def encode_extended_result(extended_result: ExtendedResult) -> dict:
    if not isinstance(extended_result, ExtendedResult):
        raise TypeError(f"Object of type '{type(extended_result)}' is not JSON serializable")

    encoded_stats = dict()
    if extended_result.stats:
        for key, value in extended_result.stats.items():
            encoded_key = str(encode(key))
            encoded_stats[encoded_key] = value

    attrs = {
        '__type__': 'ExtendedResult',
        'suppressed_items': list(extended_result.suppressed_items),
        'relevant_items': list(extended_result.relevant_items),
        'stats': encoded_stats
    }
    return attrs


def encode_schema(schema: Schema):
    if not isinstance(schema, Schema):
        raise TypeError(f"Object of type '{type(schema)}' is not JSON serializable")

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


def encode_schema_tree_node(schema_tree_node: SchemaTreeNode) -> dict:
    if not isinstance(schema_tree_node, SchemaTreeNode):
        raise TypeError(f"Object of type '{type(schema_tree_node)}' is not JSON serializable")

    attrs = {
        '__type__': 'SchemaTreeNode',
        'context': schema_tree_node.context,
        'schemas_satisfied_by': list(schema_tree_node.schemas_satisfied_by),
        'schemas_would_satisfy': list(schema_tree_node.schemas_would_satisfy),
        'label': schema_tree_node.label,
    }
    return attrs


encoder_map: dict[Type, Callable] = {
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
}


def _encode(obj: Any):
    encoder = encoder_map.get(type(obj))
    if not encoder:
        raise TypeError(f"Object of type '{type(obj)}' is not JSON serializable")
    return encoder(obj)


def encode(obj: Any, /) -> str:
    return json.dumps(obj, default=_encode)
