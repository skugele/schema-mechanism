import functools
import json
from typing import Any
from typing import Callable
from typing import Type

from anytree.exporter import DictExporter

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
from schema_mechanism.core import StateAssertion
from schema_mechanism.core import SymbolicItem
from schema_mechanism.parameters import GlobalParams
from schema_mechanism.share import SupportedFeature
from schema_mechanism.strategies.decay import ExponentialDecayStrategy
from schema_mechanism.strategies.decay import GeometricDecayStrategy
from schema_mechanism.strategies.decay import ImmediateDecayStrategy
from schema_mechanism.strategies.decay import LinearDecayStrategy
from schema_mechanism.strategies.decay import NoDecayStrategy
from schema_mechanism.strategies.evaluation import CompositeEvaluationStrategy
from schema_mechanism.strategies.evaluation import DefaultEvaluationStrategy
from schema_mechanism.strategies.evaluation import DefaultExploratoryEvaluationStrategy
from schema_mechanism.strategies.evaluation import DefaultGoalPursuitEvaluationStrategy
from schema_mechanism.strategies.evaluation import EpsilonGreedyEvaluationStrategy
from schema_mechanism.strategies.evaluation import HabituationEvaluationStrategy
from schema_mechanism.strategies.evaluation import InstrumentalValueEvaluationStrategy
from schema_mechanism.strategies.evaluation import MaxDelegatedValueEvaluationStrategy
from schema_mechanism.strategies.evaluation import NoOpEvaluationStrategy
from schema_mechanism.strategies.evaluation import PendingFocusEvaluationStrategy
from schema_mechanism.strategies.evaluation import ReliabilityEvaluationStrategy
from schema_mechanism.strategies.evaluation import TotalDelegatedValueEvaluationStrategy
from schema_mechanism.strategies.evaluation import TotalPrimitiveValueEvaluationStrategy
from schema_mechanism.strategies.match import AbsoluteDiffMatchStrategy
from schema_mechanism.strategies.match import EqualityMatchStrategy
from schema_mechanism.strategies.scaling import SigmoidScalingStrategy
from schema_mechanism.strategies.selection import RandomizeBestSelectionStrategy
from schema_mechanism.strategies.selection import RandomizeSelectionStrategy
from schema_mechanism.strategies.trace import AccumulatingTrace
from schema_mechanism.strategies.trace import ReplacingTrace
from schema_mechanism.validate import AcceptAllValidator
from schema_mechanism.validate import BlackListValidator
from schema_mechanism.validate import ElementWiseValidator
from schema_mechanism.validate import MultiValidator
from schema_mechanism.validate import RangeValidator
from schema_mechanism.validate import SupportedFeatureValidator
from schema_mechanism.validate import WhiteListValidator


def encode_action(
        action: Action,
        object_registry: dict[int, Any] = None,
        read_only_object_registry: bool = False,
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
        read_only_object_registry: bool = False,
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
        read_only_object_registry: bool = False,
        **kwargs) -> dict:
    if not isinstance(controller, Controller):
        TypeError(f"Object of type '{type(controller)}' is not JSON serializable")

    components = list(controller.components)
    descendants = list(controller.descendants)
    proximity_map = controller.proximity_map
    total_cost_map = controller.total_cost_map

    if object_registry is not None and not read_only_object_registry:
        # add objects to object registry
        if not read_only_object_registry:
            for schema in components:
                object_registry[schema.uid] = schema

            for schema in descendants:
                object_registry[schema.uid] = schema

        # encode object references in json
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
        read_only_object_registry: bool = False,
        **kwargs) -> dict:
    if not isinstance(item, SymbolicItem):
        raise TypeError(f"Object of type '{type(item)}' is not JSON serializable")

    if object_registry is not None and not read_only_object_registry:
        object_registry[item.uid] = item

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
        read_only_object_registry: bool = False,
        **kwargs) -> dict:
    if not isinstance(item, CompositeItem):
        raise TypeError(f"Object of type '{type(item)}' is not JSON serializable")

    if object_registry is not None and not read_only_object_registry:
        object_registry[item.uid] = item

    attrs = {
        '__type__': 'CompositeItem',
        'source': list(item.source),
        'primitive_value': item.primitive_value,
        'delegated_value': item.delegated_value
    }
    return attrs


def encode_state_assertion(
        state_assertion: StateAssertion,
        object_registry: dict[str, Any] = None,
        read_only_object_registry: bool = False,
        **kwargs) -> dict:
    if not isinstance(state_assertion, StateAssertion):
        raise TypeError(f"Object of type '{type(state_assertion)}' is not JSON serializable")

    items = list(state_assertion.items)

    if object_registry is not None:
        # add objects to object registry
        if not read_only_object_registry:
            for item in items:
                object_registry[item.uid] = item

        # encode object references in json
        items = map_list_to_unique_ids(items)

    attrs = {
        '__type__': 'StateAssertion',
        'items': items,
    }
    return attrs


def encode_global_params(
        global_params: GlobalParams,
        object_registry: dict[str, Any] = None,
        read_only_object_registry: bool = False,
        **kwargs) -> dict:
    if not isinstance(global_params, GlobalParams):
        raise TypeError(f"Object of type '{type(global_params)}' is not JSON serializable")

    attrs = {
        '__type__': 'GlobalParams',
        'parameters': global_params.parameters,
        'validators': global_params.validators
    }
    return attrs


def encode_global_stats(
        global_stats: GlobalStats,
        object_registry: dict[str, Any] = None,
        read_only_object_registry: bool = False,
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
        object_registry: dict[str, Any] = None,
        read_only_object_registry: bool = False,
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
        object_registry: dict[str, Any] = None,
        read_only_object_registry: bool = False,
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
        object_registry: dict[str, Any] = None,
        read_only_object_registry: bool = False,
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
        object_registry: dict[str, Any] = None,
        read_only_object_registry: bool = False,
        **kwargs) -> dict:
    if not isinstance(ec_item_stats, FrozenECItemStats):
        raise TypeError(f"Object of type '{type(ec_item_stats)}' is not JSON serializable")

    attrs = {
        '__type__': 'FrozenECItemStats',
    }
    return attrs


def encode_frozen_er_item_stats(
        er_item_stats: FrozenERItemStats,
        object_registry: dict[str, Any] = None,
        read_only_object_registry: bool = False,
        **kwargs) -> dict:
    if not isinstance(er_item_stats, FrozenERItemStats):
        raise TypeError(f"Object of type '{type(er_item_stats)}' is not JSON serializable")

    attrs = {
        '__type__': 'FrozenERItemStats',
    }
    return attrs


def encode_extended_context(
        extended_context: ExtendedContext,
        object_registry: dict[str, Any] = None,
        read_only_object_registry: bool = False,
        **kwargs) -> dict:
    if not isinstance(extended_context, ExtendedContext):
        raise TypeError(f"Object of type '{type(extended_context)}' is not JSON serializable")

    suppressed_items = list(extended_context.suppressed_items)
    relevant_items = list(extended_context.relevant_items)
    stats = extended_context.stats

    if object_registry is not None:
        # add objects to object registry
        if not read_only_object_registry:
            for item in suppressed_items:
                object_registry[item.uid] = item

            for item in relevant_items:
                object_registry[item.uid] = item

            for item in stats.keys():
                object_registry[item.uid] = item

        # encode object references in json
        suppressed_items = map_list_to_unique_ids(suppressed_items)
        relevant_items = map_list_to_unique_ids(relevant_items)
        stats = map_dict_keys_to_unique_ids(extended_context.stats)

    attrs = {
        '__type__': 'ExtendedContext',
        'suppressed_items': suppressed_items,
        'relevant_items': relevant_items,
        'stats': stats
    }
    return attrs


def encode_extended_result(
        extended_result: ExtendedResult,
        object_registry: dict[str, Any] = None,
        read_only_object_registry: bool = False,
        **kwargs) -> dict:
    if not isinstance(extended_result, ExtendedResult):
        raise TypeError(f"Object of type '{type(extended_result)}' is not JSON serializable")

    suppressed_items = list(extended_result.suppressed_items)
    relevant_items = list(extended_result.relevant_items)
    stats = extended_result.stats

    if object_registry is not None:
        # add objects to object registry
        if not read_only_object_registry:
            for item in suppressed_items:
                object_registry[item.uid] = item

            for item in relevant_items:
                object_registry[item.uid] = item

            for item in stats.keys():
                object_registry[item.uid] = item

        # encode object references in json
        suppressed_items = map_list_to_unique_ids(suppressed_items)
        relevant_items = map_list_to_unique_ids(relevant_items)
        stats = map_dict_keys_to_unique_ids(extended_result.stats)

    attrs = {
        '__type__': 'ExtendedResult',
        'suppressed_items': suppressed_items,
        'relevant_items': relevant_items,
        'stats': stats
    }
    return attrs


def encode_schema(
        schema: Schema,
        object_registry: dict[str, Any] = None,
        read_only_object_registry: bool = False,
        **kwargs) -> dict:
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


def encode_schema_tree(
        schema_tree: SchemaTree,
        object_registry: dict[str, Any] = None,
        read_only_object_registry: bool = False,
        **kwargs) -> dict:
    if not isinstance(schema_tree, SchemaTree):
        raise TypeError(f"Object of type '{type(schema_tree)}' is not JSON serializable")

    attrs = {
        '__type__': 'SchemaTree',
        'root': DictExporter().export(schema_tree.root),
    }
    return attrs


def encode_set(obj: set, **kwargs) -> dict:
    attrs = {
        '__type__': 'set',
        'values': list(obj),
    }
    return attrs


def encode_frozen_set(obj: frozenset, **kwargs) -> dict:
    attrs = {
        '__type__': 'frozenset',
        'values': list(obj),
    }
    return attrs


def encode_supported_feature(
        obj: SupportedFeature,
        **kwargs) -> dict:
    attrs = {
        '__type__': 'SupportedFeature',
        'value': obj.value
    }
    return attrs


def encode_range_validator(
        validator: RangeValidator,
        object_registry: dict[str, Any] = None,
        read_only_object_registry: bool = False,
        **kwargs) -> dict:
    attrs = {
        '__type__': 'RangeValidator',
        'low': validator.low,
        'high': validator.high,
        'exclude': list(validator.exclude)
    }
    return attrs


def encode_black_list_validator(
        validator: BlackListValidator,
        object_registry: dict[str, Any] = None,
        read_only_object_registry: bool = False,
        **kwargs) -> dict:
    attrs = {
        '__type__': 'BlackListValidator',
        'reject_set': list(validator.reject_set),
    }
    return attrs


def encode_white_list_validator(
        validator: WhiteListValidator,
        object_registry: dict[str, Any] = None,
        read_only_object_registry: bool = False,
        **kwargs) -> dict:
    attrs = {
        '__type__': 'WhiteListValidator',
        'accept_set': list(validator.accept_set),
    }
    return attrs


def encode_multi_validator(
        validator: MultiValidator,
        object_registry: dict[str, Any] = None,
        read_only_object_registry: bool = False,
        **kwargs) -> dict:
    attrs = {
        '__type__': 'MultiValidator',
        'validators': list(validator.validators),
    }
    return attrs


def encode_element_wise_validator(
        validator: ElementWiseValidator,
        object_registry: dict[str, Any] = None,
        read_only_object_registry: bool = False,
        **kwargs) -> dict:
    attrs = {
        '__type__': 'ElementWiseValidator',
        'validator': validator.validator
    }
    return attrs


def encode_accept_all_validator(
        validator: AcceptAllValidator,
        object_registry: dict[str, Any] = None,
        read_only_object_registry: bool = False,
        **kwargs) -> dict:
    attrs = {
        '__type__': 'AcceptAllValidator',
    }
    return attrs


def encode_supported_feature_validator(
        validator: SupportedFeatureValidator,
        object_registry: dict[str, Any] = None,
        read_only_object_registry: bool = False,
        **kwargs) -> dict:
    attrs = {
        '__type__': 'SupportedFeatureValidator',
    }
    return attrs


def encode_linear_decay_strategy(
        strategy: LinearDecayStrategy,
        object_registry: dict[str, Any] = None,
        read_only_object_registry: bool = False,
        **kwargs) -> dict:
    attrs = {
        '__type__': 'LinearDecayStrategy',
        'rate': strategy.rate,
        'minimum': strategy.minimum
    }
    return attrs


def encode_geometric_decay_strategy(
        strategy: GeometricDecayStrategy,
        object_registry: dict[str, Any] = None,
        read_only_object_registry: bool = False,
        **kwargs) -> dict:
    attrs = {
        '__type__': 'GeometricDecayStrategy',
        'rate': strategy.rate
    }
    return attrs


def encode_exponential_decay_strategy(
        strategy: ExponentialDecayStrategy,
        object_registry: dict[str, Any] = None,
        read_only_object_registry: bool = False,
        **kwargs) -> dict:
    attrs = {
        '__type__': 'ExponentialDecayStrategy',
        'rate': strategy.rate,
        'minimum': strategy.minimum,
        'initial': strategy.initial
    }
    return attrs


def encode_no_decay_strategy(
        strategy: NoDecayStrategy,
        object_registry: dict[str, Any] = None,
        read_only_object_registry: bool = False,
        **kwargs) -> dict:
    attrs = {
        '__type__': 'NoDecayStrategy'
    }
    return attrs


def encode_immediate_decay_strategy(
        strategy: ImmediateDecayStrategy,
        object_registry: dict[str, Any] = None,
        read_only_object_registry: bool = False,
        **kwargs) -> dict:
    attrs = {
        '__type__': 'ImmediateDecayStrategy',
        'minimum': strategy.minimum
    }
    return attrs


def encode_no_op_evaluation_strategy(
        strategy: NoOpEvaluationStrategy,
        object_registry: dict[str, Any] = None,
        read_only_object_registry: bool = False,
        **kwargs) -> dict:
    attrs = {
        '__type__': 'NoOpEvaluationStrategy'
    }
    return attrs


def encode_total_primitive_value_evaluation_strategy(
        strategy: TotalPrimitiveValueEvaluationStrategy,
        object_registry: dict[str, Any] = None,
        read_only_object_registry: bool = False,
        **kwargs) -> dict:
    attrs = {
        '__type__': 'TotalPrimitiveValueEvaluationStrategy'
    }
    return attrs


def encode_total_delegated_value_evaluation_strategy(
        strategy: TotalDelegatedValueEvaluationStrategy,
        object_registry: dict[str, Any] = None,
        read_only_object_registry: bool = False,
        **kwargs) -> dict:
    attrs = {
        '__type__': 'TotalDelegatedValueEvaluationStrategy'
    }
    return attrs


def encode_max_delegated_value_evaluation_strategy(
        strategy: MaxDelegatedValueEvaluationStrategy,
        object_registry: dict[str, Any] = None,
        read_only_object_registry: bool = False,
        **kwargs) -> dict:
    attrs = {
        '__type__': 'MaxDelegatedValueEvaluationStrategy'
    }
    return attrs


def encode_instrumental_value_evaluation_strategy(
        strategy: InstrumentalValueEvaluationStrategy,
        object_registry: dict[str, Any] = None,
        read_only_object_registry: bool = False,
        **kwargs) -> dict:
    attrs = {
        '__type__': 'InstrumentalValueEvaluationStrategy'
    }
    return attrs


def encode_pending_focus_evaluation_strategy(
        strategy: PendingFocusEvaluationStrategy,
        object_registry: dict[str, Any] = None,
        read_only_object_registry: bool = False,
        **kwargs) -> dict:
    attrs = {
        '__type__': 'PendingFocusEvaluationStrategy',
        'max_value': strategy.max_value,
        'decay_strategy': strategy.decay_strategy
    }
    return attrs


def encode_reliability_evaluation_strategy(
        strategy: ReliabilityEvaluationStrategy,
        object_registry: dict[str, Any] = None,
        read_only_object_registry: bool = False,
        **kwargs) -> dict:
    attrs = {
        '__type__': 'ReliabilityEvaluationStrategy',
        'max_penalty': strategy.max_penalty,
        'severity': strategy.severity,
    }
    return attrs


def encode_epsilon_greedy_evaluation_strategy(
        strategy: EpsilonGreedyEvaluationStrategy,
        object_registry: dict[str, Any] = None,
        read_only_object_registry: bool = False,
        **kwargs) -> dict:
    attrs = {
        '__type__': 'EpsilonGreedyEvaluationStrategy',
        'epsilon': strategy.epsilon,
        'epsilon_min': strategy.epsilon_min,
        'max_value': strategy.max_value,
        'decay_strategy': strategy.decay_strategy,
    }
    return attrs


def encode_habituation_evaluation_strategy(
        strategy: HabituationEvaluationStrategy,
        object_registry: dict[str, Any] = None,
        read_only_object_registry: bool = False,
        **kwargs) -> dict:
    attrs = {
        '__type__': 'HabituationEvaluationStrategy',
        'scaling_strategy': strategy.scaling_strategy,
    }
    return attrs


def encode_composite_evaluation_strategy(
        strategy: CompositeEvaluationStrategy,
        object_registry: dict[str, Any] = None,
        read_only_object_registry: bool = False,
        **kwargs) -> dict:
    attrs = {
        '__type__': 'CompositeEvaluationStrategy',
        'strategies': list(strategy.strategies),
        'strategy_alias': strategy.strategy_alias,
        'weights': list(strategy.weights),
    }
    return attrs


def encode_default_exploratory_evaluation_strategy(
        strategy: DefaultExploratoryEvaluationStrategy,
        object_registry: dict[str, Any] = None,
        read_only_object_registry: bool = False,
        **kwargs) -> dict:
    attrs = {
        '__type__': 'DefaultExploratoryEvaluationStrategy',
        'epsilon': strategy.epsilon,
        'epsilon_min': strategy.epsilon_min,
        'epsilon_decay_rate': strategy.epsilon_decay_rate,
        'post_process': list(strategy.post_process),
    }
    return attrs


def encode_default_global_pursuit_evaluation_strategy(
        strategy: DefaultGoalPursuitEvaluationStrategy,
        object_registry: dict[str, Any] = None,
        read_only_object_registry: bool = False,
        **kwargs) -> dict:
    attrs = {
        '__type__': 'DefaultGoalPursuitEvaluationStrategy',
        'reliability_max_penalty': strategy.reliability_max_penalty,
        'pending_focus_max_value': strategy.pending_focus_max_value,
        'pending_focus_decay_rate': strategy.pending_focus_decay_rate,
        'post_process': list(strategy.post_process),
    }
    return attrs


def encode_default_evaluation_strategy(
        strategy: DefaultEvaluationStrategy,
        object_registry: dict[str, Any] = None,
        read_only_object_registry: bool = False,
        **kwargs) -> dict:
    attrs = {
        '__type__': 'DefaultEvaluationStrategy',
        'goal_pursuit_strategy': strategy.goal_pursuit_strategy,
        'exploratory_strategy': strategy.exploratory_strategy,
    }
    return attrs


def encode_equality_match_strategy(
        strategy: EqualityMatchStrategy,
        object_registry: dict[str, Any] = None,
        read_only_object_registry: bool = False,
        **kwargs) -> dict:
    attrs = {
        '__type__': 'EqualityMatchStrategy',
    }
    return attrs


def encode_absolute_diff_match_strategy(
        strategy: AbsoluteDiffMatchStrategy,
        object_registry: dict[str, Any] = None,
        read_only_object_registry: bool = False,
        **kwargs) -> dict:
    attrs = {
        '__type__': 'AbsoluteDiffMatchStrategy',
        'max_diff': strategy.max_diff,
    }
    return attrs


def encode_sigmoid_scaling_strategy(
        strategy: SigmoidScalingStrategy,
        object_registry: dict[str, Any] = None,
        read_only_object_registry: bool = False,
        **kwargs) -> dict:
    attrs = {
        '__type__': 'SigmoidScalingStrategy',
        'range_scale': strategy.range_scale,
        'vertical_shift': strategy.vertical_shift,
        'intercept': strategy.intercept,
    }
    return attrs


def encode_randomize_selection_strategy(
        strategy: RandomizeSelectionStrategy,
        object_registry: dict[str, Any] = None,
        read_only_object_registry: bool = False,
        **kwargs) -> dict:
    attrs = {
        '__type__': 'RandomizeSelectionStrategy',
    }
    return attrs


def encode_randomize_best_selection_strategy(
        strategy: RandomizeBestSelectionStrategy,
        object_registry: dict[str, Any] = None,
        read_only_object_registry: bool = False,
        **kwargs) -> dict:
    attrs = {
        '__type__': 'RandomizeBestSelectionStrategy',
        'match_strategy': strategy.match_strategy,
    }
    return attrs


def encode_accumulating_trace(
        strategy: AccumulatingTrace,
        object_registry: dict[str, Any] = None,
        read_only_object_registry: bool = False,
        **kwargs) -> dict:
    attrs = {
        '__type__': 'AccumulatingTrace',
        'decay_strategy': strategy.decay_strategy,
        'active_increment': strategy.active_increment,
    }
    return attrs


def encode_replacing_trace(
        strategy: ReplacingTrace,
        object_registry: dict[str, Any] = None,
        read_only_object_registry: bool = False,
        **kwargs) -> dict:
    attrs = {
        '__type__': 'ReplacingTrace',
        'decay_strategy': strategy.decay_strategy,
        'active_value': strategy.active_value,
    }
    return attrs


encoder_map: dict[Type, Callable] = {
    # built-ins
    int: int,
    set: encode_set,
    frozenset: encode_frozen_set,

    # custom objects
    AbsoluteDiffMatchStrategy: encode_absolute_diff_match_strategy,
    AcceptAllValidator: encode_accept_all_validator,
    AccumulatingTrace: encode_accumulating_trace,
    Action: encode_action,
    BlackListValidator: encode_black_list_validator,
    CompositeAction: encode_composite_action,
    CompositeEvaluationStrategy: encode_composite_evaluation_strategy,
    CompositeItem: encode_composite_item,
    Controller: encode_controller,
    DefaultExploratoryEvaluationStrategy: encode_default_exploratory_evaluation_strategy,
    DefaultGoalPursuitEvaluationStrategy: encode_default_global_pursuit_evaluation_strategy,
    DefaultEvaluationStrategy: encode_default_evaluation_strategy,
    ECItemStats: encode_ec_item_stats,
    ERItemStats: encode_er_item_stats,
    ElementWiseValidator: encode_element_wise_validator,
    EpsilonGreedyEvaluationStrategy: encode_epsilon_greedy_evaluation_strategy,
    EqualityMatchStrategy: encode_equality_match_strategy,
    ExponentialDecayStrategy: encode_exponential_decay_strategy,
    ExtendedContext: encode_extended_context,
    ExtendedResult: encode_extended_result,
    FrozenECItemStats: encode_frozen_ec_item_stats,
    FrozenERItemStats: encode_frozen_er_item_stats,
    GeometricDecayStrategy: encode_geometric_decay_strategy,
    GlobalParams: encode_global_params,
    GlobalStats: encode_global_stats,
    HabituationEvaluationStrategy: encode_habituation_evaluation_strategy,
    ImmediateDecayStrategy: encode_immediate_decay_strategy,
    InstrumentalValueEvaluationStrategy: encode_instrumental_value_evaluation_strategy,
    LinearDecayStrategy: encode_linear_decay_strategy,
    MaxDelegatedValueEvaluationStrategy: encode_max_delegated_value_evaluation_strategy,
    MultiValidator: encode_multi_validator,
    NoDecayStrategy: encode_no_decay_strategy,
    NoOpEvaluationStrategy: encode_no_op_evaluation_strategy,
    PendingFocusEvaluationStrategy: encode_pending_focus_evaluation_strategy,
    RandomizeBestSelectionStrategy: encode_randomize_best_selection_strategy,
    RandomizeSelectionStrategy: encode_randomize_selection_strategy,
    RangeValidator: encode_range_validator,
    ReliabilityEvaluationStrategy: encode_reliability_evaluation_strategy,
    ReplacingTrace: encode_replacing_trace,
    Schema: encode_schema,
    SchemaStats: encode_schema_stats,
    SchemaTree: encode_schema_tree,
    SigmoidScalingStrategy: encode_sigmoid_scaling_strategy,
    StateAssertion: encode_state_assertion,
    SupportedFeature: encode_supported_feature,
    SupportedFeatureValidator: encode_supported_feature_validator,
    SymbolicItem: encode_symbolic_item,
    TotalDelegatedValueEvaluationStrategy: encode_total_delegated_value_evaluation_strategy,
    TotalPrimitiveValueEvaluationStrategy: encode_total_primitive_value_evaluation_strategy,
    WhiteListValidator: encode_white_list_validator,
}


def map_list_to_unique_ids(objects: list[Any]) -> list[int]:
    return [o.uid for o in objects]


def map_dict_keys_to_unique_ids(object_dict: dict) -> dict[str, Any]:
    return {key.uid: value for key, value in object_dict.items()}


def _encode(obj: Any, **kwargs):
    encoder = encoder_map.get(type(obj))

    if not encoder:
        raise TypeError(f"Object of type '{type(obj)}' is not JSON serializable")
    return encoder(obj, **kwargs)


def encode(obj: Any, /, **kwargs) -> str:
    return json.dumps(obj, default=functools.partial(_encode, **kwargs))
