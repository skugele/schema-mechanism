import functools
import json
from typing import Any
from typing import Callable
from typing import Optional

from anytree.importer import DictImporter

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
from schema_mechanism.core import NULL_STATE_ASSERT
from schema_mechanism.core import Schema
from schema_mechanism.core import SchemaPool
from schema_mechanism.core import SchemaStats
from schema_mechanism.core import SchemaTree
from schema_mechanism.core import SchemaTreeNode
from schema_mechanism.core import SchemaUniqueKey
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
from schema_mechanism.strategies.evaluation import EpsilonRandomEvaluationStrategy
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
from schema_mechanism.strategies.weight_update import CyclicWeightUpdateStrategy
from schema_mechanism.strategies.weight_update import NoOpWeightUpdateStrategy
from schema_mechanism.validate import AcceptAllValidator
from schema_mechanism.validate import BlackListValidator
from schema_mechanism.validate import ElementWiseValidator
from schema_mechanism.validate import MultiValidator
from schema_mechanism.validate import RangeValidator
from schema_mechanism.validate import SupportedFeatureValidator
from schema_mechanism.validate import WhiteListValidator


def decode_action(obj_dict: dict, object_registry: Optional[dict[str, Any]] = None, **kwargs) -> Action:
    return Action(**obj_dict)


def decode_composite_action(obj_dict: dict, object_registry: Optional[dict[str, Any]] = None,
                            **kwargs) -> CompositeAction:
    return CompositeAction(**obj_dict)


def decode_controller(obj_dict: dict, object_registry: Optional[dict[str, Any]] = None, **kwargs) -> Controller:
    components = obj_dict.pop('components')
    descendants = obj_dict.pop('descendants')
    proximity_map = obj_dict.pop('proximity_map')
    total_cost_map = obj_dict.pop('total_cost_map')

    if object_registry is not None:
        components = set(
            map_list_from_unique_ids_to_list_of_objects(
                components,
                object_registry=object_registry)
        )

        descendants = set(
            map_list_from_unique_ids_to_list_of_objects(
                descendants,
                object_registry=object_registry)
        )

        proximity_map = map_dict_from_unique_id_to_object_keys(
            proximity_map,
            object_registry=object_registry
        )

        total_cost_map = map_dict_from_unique_id_to_object_keys(
            total_cost_map,
            object_registry=object_registry
        )

    return Controller(
        components=components,
        descendants=descendants,
        proximity_map=proximity_map,
        total_cost_map=total_cost_map,
        **obj_dict
    )


def decode_symbolic_item(obj_dict: dict, object_registry: Optional[dict[str, Any]] = None, **kwargs) -> SymbolicItem:
    source = obj_dict.pop('source')
    return ItemPool().get(source, item_type=SymbolicItem, **obj_dict)


def decode_composite_item(obj_dict: dict, object_registry: Optional[dict[str, Any]] = None, **kwargs) -> CompositeItem:
    source = obj_dict.pop('source')
    return ItemPool().get(frozenset(source), item_type=CompositeItem, **obj_dict)


def decode_state_assertion(obj_dict: dict, object_registry: Optional[dict[str, Any]] = None,
                           **kwargs) -> StateAssertion:
    items = obj_dict.pop('items')
    if object_registry is not None:
        items = map_list_from_unique_ids_to_list_of_objects(items, object_registry=object_registry)
    return NULL_STATE_ASSERT if items is None else StateAssertion(items=items)


def decode_global_params(obj_dict: dict, object_registry: Optional[dict[str, Any]] = None, **kwargs) -> GlobalParams:
    return GlobalParams(**obj_dict)


def decode_global_stats(obj_dict: dict, object_registry: Optional[dict[str, Any]] = None, **kwargs) -> GlobalStats:
    return GlobalStats(**obj_dict)


def decode_schema_stats(obj_dict: dict, object_registry: Optional[dict[str, Any]] = None, **kwargs) -> SchemaStats:
    return SchemaStats(**obj_dict)


def decode_ec_item_stats(obj_dict: dict, object_registry: Optional[dict[str, Any]] = None, **kwargs) -> ECItemStats:
    return ECItemStats(**obj_dict)


def decode_er_item_stats(obj_dict: dict, object_registry: Optional[dict[str, Any]] = None, **kwargs) -> ERItemStats:
    return ERItemStats(**obj_dict)


def decode_frozen_ec_item_stats(obj_dict: dict, object_registry: Optional[dict[str, Any]] = None,
                                **kwargs) -> FrozenECItemStats:
    return FROZEN_EC_ITEM_STATS


def decode_frozen_er_item_stats(obj_dict: dict, object_registry: Optional[dict[str, Any]] = None,
                                **kwargs) -> FrozenERItemStats:
    return FROZEN_ER_ITEM_STATS


def decode_extended_context(obj_dict: dict, object_registry: Optional[dict[str, Any]] = None,
                            **kwargs) -> ExtendedContext:
    stats = obj_dict.pop('stats')
    suppressed_items = obj_dict.pop('suppressed_items')
    relevant_items = obj_dict.pop('relevant_items')

    if object_registry is not None:
        stats = map_dict_from_unique_id_to_object_keys(
            stats,
            object_registry=object_registry
        )

        suppressed_items = map_list_from_unique_ids_to_list_of_objects(
            suppressed_items,
            object_registry=object_registry
        )

        relevant_items = map_list_from_unique_ids_to_list_of_objects(
            relevant_items,
            object_registry=object_registry
        )

    return ExtendedContext(
        stats=stats,
        suppressed_items=suppressed_items,
        relevant_items=relevant_items,
    )


def decode_extended_result(obj_dict: dict, object_registry: Optional[dict[str, Any]] = None,
                           **kwargs) -> ExtendedResult:
    stats = obj_dict.pop('stats')
    suppressed_items = obj_dict.pop('suppressed_items')
    relevant_items = obj_dict.pop('relevant_items')

    if object_registry is not None:
        stats = map_dict_from_unique_id_to_object_keys(
            stats,
            object_registry=object_registry
        )

        suppressed_items = map_list_from_unique_ids_to_list_of_objects(
            suppressed_items,
            object_registry=object_registry
        )

        relevant_items = map_list_from_unique_ids_to_list_of_objects(
            relevant_items,
            object_registry=object_registry
        )

    return ExtendedResult(
        stats=stats,
        suppressed_items=suppressed_items,
        relevant_items=relevant_items,
    )


def decode_schema(obj_dict: dict, object_registry: Optional[dict[str, Any]] = None, **kwargs) -> Schema:
    key = SchemaUniqueKey(
        context=obj_dict.pop('context'),
        action=obj_dict.pop('action'),
        result=obj_dict.pop('result'),
    )

    return SchemaPool().get(key, **obj_dict)


def decode_schema_tree(obj_dict: dict, object_registry: Optional[dict[str, Any]] = None, **kwargs) -> SchemaTree:
    root = DictImporter(nodecls=SchemaTreeNode).import_(obj_dict.pop('root'))
    return SchemaTree(root=root)


def decode_supported_feature(obj_dict: dict, **kwargs) -> SupportedFeature:
    return SupportedFeature(obj_dict['value'])


def decode_range_validator(obj_dict: dict, **kwargs) -> RangeValidator:
    return RangeValidator(**obj_dict)


def decode_black_list_validator(obj_dict: dict, **kwargs) -> BlackListValidator:
    return BlackListValidator(**obj_dict)


def decode_white_list_validator(obj_dict: dict, **kwargs) -> WhiteListValidator:
    return WhiteListValidator(**obj_dict)


def decode_accept_all_validator(obj_dict: dict, **kwargs) -> AcceptAllValidator:
    return AcceptAllValidator()


def decode_multi_validator(obj_dict: dict, **kwargs) -> MultiValidator:
    return MultiValidator(**obj_dict)


def decode_element_wise_validator(obj_dict: dict, **kwargs) -> ElementWiseValidator:
    return ElementWiseValidator(**obj_dict)


def decode_supported_feature_validator(obj_dict: dict, **kwargs) -> SupportedFeatureValidator:
    return SupportedFeatureValidator()


def decode_linear_decay_strategy(obj_dict: dict, **kwargs) -> LinearDecayStrategy:
    return LinearDecayStrategy(**obj_dict)


def decode_geometric_decay_strategy(obj_dict: dict, **kwargs) -> GeometricDecayStrategy:
    return GeometricDecayStrategy(**obj_dict)


def decode_exponential_decay_strategy(obj_dict: dict, **kwargs) -> ExponentialDecayStrategy:
    return ExponentialDecayStrategy(**obj_dict)


def decode_no_decay_strategy(obj_dict: dict, **kwargs) -> NoDecayStrategy:
    return NoDecayStrategy()


def decode_immediate_decay_strategy(obj_dict: dict, **kwargs) -> ImmediateDecayStrategy:
    return ImmediateDecayStrategy(**obj_dict)


def decode_no_op_evaluation_strategy(obj_dict: dict, **kwargs) -> NoOpEvaluationStrategy:
    return NoOpEvaluationStrategy()


def decode_total_primitive_value_evaluation_strategy(obj_dict: dict, **kwargs) -> TotalPrimitiveValueEvaluationStrategy:
    return TotalPrimitiveValueEvaluationStrategy()


def decode_total_delegated_value_evaluation_strategy(obj_dict: dict, **kwargs) -> TotalDelegatedValueEvaluationStrategy:
    return TotalDelegatedValueEvaluationStrategy()


def decode_max_delegated_value_evaluation_strategy(obj_dict: dict, **kwargs) -> MaxDelegatedValueEvaluationStrategy:
    return MaxDelegatedValueEvaluationStrategy()


def decode_instrumental_value_evaluation_strategy(obj_dict: dict, **kwargs) -> InstrumentalValueEvaluationStrategy:
    return InstrumentalValueEvaluationStrategy()


def decode_pending_focus_evaluation_strategy(obj_dict: dict, **kwargs) -> PendingFocusEvaluationStrategy:
    return PendingFocusEvaluationStrategy()


def decode_reliability_evaluation_strategy(obj_dict: dict, **kwargs) -> ReliabilityEvaluationStrategy:
    return ReliabilityEvaluationStrategy(**obj_dict)


def decode_epsilon_random_evaluation_strategy(obj_dict: dict, **kwargs) -> EpsilonRandomEvaluationStrategy:
    return EpsilonRandomEvaluationStrategy(**obj_dict)


def decode_habituation_evaluation_strategy(obj_dict: dict, **kwargs) -> HabituationEvaluationStrategy:
    return HabituationEvaluationStrategy(**obj_dict)


def decode_composite_evaluation_strategy(obj_dict: dict, **kwargs) -> CompositeEvaluationStrategy:
    return CompositeEvaluationStrategy(**obj_dict)


def decode_default_exploratory_evaluation_strategy(obj_dict: dict, **kwargs) -> DefaultExploratoryEvaluationStrategy:
    return DefaultExploratoryEvaluationStrategy(**obj_dict)


def decode_default_global_pursuit_evaluation_strategy(obj_dict: dict, **kwargs) -> DefaultGoalPursuitEvaluationStrategy:
    return DefaultGoalPursuitEvaluationStrategy(**obj_dict)


def decode_default_evaluation_strategy(obj_dict: dict, **kwargs) -> DefaultEvaluationStrategy:
    return DefaultEvaluationStrategy(**obj_dict)


def decode_equality_match_strategy(obj_dict: dict, **kwargs) -> EqualityMatchStrategy:
    return EqualityMatchStrategy()


def decode_absolute_diff_match_strategy(obj_dict: dict, **kwargs) -> AbsoluteDiffMatchStrategy:
    return AbsoluteDiffMatchStrategy(**obj_dict)


def decode_sigmoid_scaling_strategy(obj_dict: dict, **kwargs) -> SigmoidScalingStrategy:
    return SigmoidScalingStrategy(**obj_dict)


def decode_randomize_selection_strategy(obj_dict: dict, **kwargs) -> RandomizeSelectionStrategy:
    return RandomizeSelectionStrategy()


def decode_randomize_best_selection_strategy(obj_dict: dict, **kwargs) -> RandomizeBestSelectionStrategy:
    return RandomizeBestSelectionStrategy(**obj_dict)


def decode_no_op_weight_update_strategy(obj_dict: dict, **kwargs) -> NoOpWeightUpdateStrategy:
    return NoOpWeightUpdateStrategy()


def decode_cyclic_weight_update_strategy(obj_dict: dict, **kwargs) -> CyclicWeightUpdateStrategy:
    return CyclicWeightUpdateStrategy(**obj_dict)


def decode_accumulating_trace(obj_dict: dict, **kwargs) -> AccumulatingTrace:
    return AccumulatingTrace(**obj_dict)


def decode_replacing_trace(obj_dict: dict, **kwargs) -> ReplacingTrace:
    return ReplacingTrace(**obj_dict)


def decode_set(obj_dict: dict, **kwargs) -> set:
    return set(obj_dict['values'])


def decode_frozenset(obj_dict: dict, **kwargs) -> frozenset:
    return frozenset(obj_dict['values'])


decoder_map: dict[str, Callable] = {
    # built-ins
    'set': decode_set,
    'frozenset': decode_frozenset,

    # custom objects
    'AbsoluteDiffMatchStrategy': decode_absolute_diff_match_strategy,
    'AcceptAllValidator': decode_accept_all_validator,
    'AccumulatingTrace': decode_accumulating_trace,
    'Action': decode_action,
    'BlackListValidator': decode_black_list_validator,
    'CompositeAction': decode_composite_action,
    'CompositeEvaluationStrategy': decode_composite_evaluation_strategy,
    'CompositeItem': decode_composite_item,
    'Controller': decode_controller,
    'CyclicWeightUpdateStrategy': decode_cyclic_weight_update_strategy,
    'DefaultEvaluationStrategy': decode_default_evaluation_strategy,
    'DefaultExploratoryEvaluationStrategy': decode_default_exploratory_evaluation_strategy,
    'DefaultGoalPursuitEvaluationStrategy': decode_default_global_pursuit_evaluation_strategy,
    'ECItemStats': decode_ec_item_stats,
    'ERItemStats': decode_er_item_stats,
    'ElementWiseValidator': decode_element_wise_validator,
    'EpsilonRandomEvaluationStrategy': decode_epsilon_random_evaluation_strategy,
    'EqualityMatchStrategy': decode_equality_match_strategy,
    'ExponentialDecayStrategy': decode_exponential_decay_strategy,
    'ExtendedContext': decode_extended_context,
    'ExtendedResult': decode_extended_result,
    'FrozenECItemStats': decode_frozen_ec_item_stats,
    'FrozenERItemStats': decode_frozen_er_item_stats,
    'GeometricDecayStrategy': decode_geometric_decay_strategy,
    'GlobalParams': decode_global_params,
    'GlobalStats': decode_global_stats,
    'HabituationEvaluationStrategy': decode_habituation_evaluation_strategy,
    'ImmediateDecayStrategy': decode_immediate_decay_strategy,
    'InstrumentalValueEvaluationStrategy': decode_instrumental_value_evaluation_strategy,
    'LinearDecayStrategy': decode_linear_decay_strategy,
    'MaxDelegatedValueEvaluationStrategy': decode_max_delegated_value_evaluation_strategy,
    'MultiValidator': decode_multi_validator,
    'NoDecayStrategy': decode_no_decay_strategy,
    'NoOpEvaluationStrategy': decode_no_op_evaluation_strategy,
    'NoOpWeightUpdateStrategy': decode_no_op_weight_update_strategy,
    'PendingFocusEvaluationStrategy': decode_pending_focus_evaluation_strategy,
    'RandomizeBestSelectionStrategy': decode_randomize_best_selection_strategy,
    'RandomizeSelectionStrategy': decode_randomize_selection_strategy,
    'RangeValidator': decode_range_validator,
    'ReliabilityEvaluationStrategy': decode_reliability_evaluation_strategy,
    'ReplacingTrace': decode_replacing_trace,
    'Schema': decode_schema,
    'SchemaStats': decode_schema_stats,
    'SchemaTree': decode_schema_tree,
    'SigmoidScalingStrategy': decode_sigmoid_scaling_strategy,
    'StateAssertion': decode_state_assertion,
    'SupportedFeature': decode_supported_feature,
    'SupportedFeatureValidator': decode_supported_feature_validator,
    'SymbolicItem': decode_symbolic_item,
    'TotalDelegatedValueEvaluationStrategy': decode_total_delegated_value_evaluation_strategy,
    'TotalPrimitiveValueEvaluationStrategy': decode_total_primitive_value_evaluation_strategy,
    'WhiteListValidator': decode_white_list_validator,
}


def map_list_from_unique_ids_to_list_of_objects(
        unique_ids: list[str],
        object_registry: dict[str, Any]) -> list[Any]:
    return [object_registry[uid] for uid in unique_ids]


def map_dict_from_unique_id_to_object_keys(
        unique_id_dict: dict[str, Any],
        object_registry: dict[str, Any]) -> dict[Any, Any]:
    return {object_registry[uid]: value for uid, value in unique_id_dict.items()}


def _decode(obj_dict: dict, **kwargs) -> Any:
    # the __type__ element is added to all objects serialized using a custom encoder
    obj_type = obj_dict.pop('__type__') if '__type__' in obj_dict else None

    decoder = decoder_map.get(obj_type)

    # use custom decoder if one exists; otherwise, return object dictionary for default json decoder to attempt decoding
    return decoder(obj_dict, **kwargs) if decoder else obj_dict


def decode(obj: Any, /, **kwargs) -> Any:
    return json.loads(obj, object_hook=functools.partial(_decode, **kwargs))
