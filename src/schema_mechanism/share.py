from __future__ import annotations

import sys
from collections import Collection
from collections import ItemsView
from collections import defaultdict
from datetime import datetime
from enum import Enum
from enum import IntEnum
from enum import auto
from time import time
from typing import Any
from typing import Optional
from typing import TextIO

import numpy as np

from schema_mechanism.stats import ItemCorrelationTest
from schema_mechanism.util import Singleton
from schema_mechanism.validate import MultiValidator
from schema_mechanism.validate import NULL_VALIDATOR
from schema_mechanism.validate import RangeValidator
from schema_mechanism.validate import TypeValidator
from schema_mechanism.validate import Validator


class Verbosity(IntEnum):
    TRACE = auto()
    DEBUG = auto()
    INFO = auto()
    WARN = auto()
    ERROR = auto()
    FATAL = auto()
    NONE = auto()


class SupportedFeature(Enum):
    COMPOSITE_ACTIONS = auto()

    # "There is an embellishment of the marginal attribution algorithm--deferring to a more specific applicable schema--
    #  that often enables the discovery of an item whose relevance has been obscured." (see Drescher,1991, pp. 75-76)
    EC_DEFER_TO_MORE_SPECIFIC_SCHEMA = auto()

    # "[another] embellishment also reduces redundancy: when a schema's extended context simultaneously detects the
    # relevance of several items--that is, their statistics pass the significance threshold on the same trial--the most
    # specific is chosen as the one for inclusion in a spin-off from that schema." (see Drescher, 1991, p. 77)
    #
    #     Note: Requires that EC_DEFER_TO_MORE_SPECIFIC is also enabled.
    EC_MOST_SPECIFIC_ON_MULTIPLE = auto()

    # "The machinery's sensitivity to results is amplified by an embellishment of marginal attribution: when a given
    #  schema is idle (i.e., it has not just completed an activation), the updating of its extended result data is
    #  suppressed for any state transition which is explained--meaning that the transition is predicted as the result
    #  of a reliable schema whose activation has just completed." (see Drescher, 1991, p. 73)
    ER_SUPPRESS_UPDATE_ON_EXPLAINED = auto()

    # Supports the creation of result spin-off schemas incrementally. This was not supported in the original schema
    # mechanism because of the proliferation of composite results that result. It is allowed here to facilitate
    # comparison and experimentation.
    ER_INCREMENTAL_RESULTS = auto()

    # Modifies the schema mechanism to only create context spin-offs containing positive assertions.
    EC_POSITIVE_ASSERTIONS_ONLY = auto()

    # Modifies the schema mechanism to only create result spin-offs containing positive assertions.
    ER_POSITIVE_ASSERTIONS_ONLY = auto()


class SupportedFeatureValidator(Validator):
    def __call__(self, features: Optional[Collection[SupportedFeature]]) -> None:
        features = set(features)
        for value in features:
            if not isinstance(value, SupportedFeature):
                raise ValueError(f'Unsupported feature: {value}')

        if (SupportedFeature.EC_MOST_SPECIFIC_ON_MULTIPLE in features and
                SupportedFeature.EC_DEFER_TO_MORE_SPECIFIC_SCHEMA not in features):
            raise ValueError(f'The feature EC_MOST_SPECIFIC_ON_MULTIPLE requires EC_DEFER_TO_MORE_SPECIFIC_SCHEMA')


def is_feature_enabled(feature: SupportedFeature) -> bool:
    return feature in GlobalParams().get('features')


class GlobalParams(metaclass=Singleton):

    def __init__(self) -> None:
        self._defaults: dict[str, Any] = dict()
        self._validators: dict[str, Validator] = defaultdict(lambda: NULL_VALIDATOR)
        self._params: dict[str, Any] = dict(self._defaults)

        self._set_validators()
        self._set_defaults()

        self.reset()

    def __iter__(self) -> ItemsView[str, Any]:
        yield from self._params.items()

    @property
    def defaults(self) -> dict[str, Any]:
        return self._defaults

    @property
    def validators(self) -> dict[str, Any]:
        return self._validators

    def set(self, name: str, value: Any) -> None:
        if name not in self._params:
            warn(f'Parameter "{name}" does not exist. Creating new parameter.')

        # raises ValueError if new value is invalid
        self._validators[name](value)
        self._params[name] = value

    def get(self, name: str) -> Any:
        if name not in self._params:
            warn(f'Parameter "{name}" does not exist.')

        return self._params.get(name)

    def _set_defaults(self) -> None:

        # verbosity used to determine the active print/warn statements
        self._defaults['verbosity'] = Verbosity.WARN

        # format string used by output functions (debug, info, warn, error, fatal)
        self._defaults['output_format'] = '{timestamp} [{severity}]\t{message}'

        # default seed for the random number generator
        self._defaults['rng_seed'] = int(time())

        # determines step size for incremental updates (e.g., this is used for delegated value updates)
        self._defaults['learning_rate'] = 0.01

        # item correlation test used for determining relevance of extended context items
        self._defaults['ext_context.correlation_test'] = None

        # thresholds for determining the relevance of extended result items
        #     from 0.0 [weakest correlation] to 1.0 [strongest correlation]
        self._defaults['ext_context.positive_correlation_threshold'] = 0.95
        self._defaults['ext_context.negative_correlation_threshold'] = 0.95

        # item correlation test used for determining relevance of extended result items
        self._defaults['ext_result.correlation_test'] = None

        # thresholds for determining the relevance of extended result items
        #     from 0.0 [weakest correlation] to 1.0 [strongest correlation]
        self._defaults['ext_result.positive_correlation_threshold'] = 0.95
        self._defaults['ext_result.negative_correlation_threshold'] = 0.95

        # success threshold used for determining that a schema is reliable
        #     from 0.0 [schema has never succeeded] to 1.0 [schema always succeeds]
        self._defaults['reliability_threshold'] = 0.95

        # used by delegated value helper
        self._defaults['delegated_value_helper.discount_factor'] = 0.9
        self._defaults['delegated_value_helper.decay_rate'] = 0.2

        # used by backward_chains (supports composite action)
        self._defaults['backward_chains.max_len'] = 5
        self._defaults['backward_chains.update_frequency'] = 0.01

        # composite actions are created for novel result states that have values that are greater than the baseline
        # value by AT LEAST this amount
        self._defaults['composite_action_min_baseline_advantage'] = 1.0

        # used by reliability_values
        self._defaults['goal_pursuit_strategy.reliability.max_penalty'] = 10.0

        # used by reliability_values
        self._defaults['habituation_exploratory_strategy.decay.rate'] = 0.95
        self._defaults['habituation_exploratory_strategy.multiplier'] = 10.0

        # used by epsilon greedy exploratory
        self._defaults['random_exploratory_strategy.epsilon.initial'] = 1.0
        self._defaults['random_exploratory_strategy.epsilon.decay.rate'] = 0.999
        self._defaults['random_exploratory_strategy.epsilon.decay.min'] = 0.01

        # schema selection weighting (set in SchemaMechanism)
        self._defaults['schema_selection.weights.goal_weight'] = 0.6
        self._defaults['schema_selection.weights.explore_weight'] = 0.4

        # default features
        self._defaults['features'] = {
            SupportedFeature.COMPOSITE_ACTIONS,
            SupportedFeature.EC_DEFER_TO_MORE_SPECIFIC_SCHEMA,
            SupportedFeature.EC_MOST_SPECIFIC_ON_MULTIPLE,
            SupportedFeature.ER_POSITIVE_ASSERTIONS_ONLY,
            SupportedFeature.EC_POSITIVE_ASSERTIONS_ONLY,
            SupportedFeature.ER_SUPPRESS_UPDATE_ON_EXPLAINED,
        }

        # TODO: this registers the key names to prevent warning messages, but I need to update this with the names
        # TODO: of default classes or some other indicators later. (NOTE: circular dependency issues will ensue if
        # TODO: I initialize these with the types from schema_mechanism.core).
        self._defaults['item_type'] = None
        self._defaults['composite_item_type'] = None
        self._defaults['schema_type'] = None

    def _set_validators(self):

        self._validators['backward_chains.max_len'] = MultiValidator([TypeValidator([int]), RangeValidator(low=0)])
        self._validators['backward_chains.update_frequency'] = RangeValidator(0.0, 1.0)
        self._validators['composite_action_min_baseline_advantage'] = TypeValidator([float])
        self._validators['delegated_value_helper.decay_rate'] = RangeValidator(0.0, 1.0)
        self._validators['delegated_value_helper.discount_factor'] = RangeValidator(0.0, 1.0)
        self._validators['ext_context.correlation_test'] = TypeValidator([ItemCorrelationTest])
        self._validators['ext_context.negative_correlation_threshold'] = RangeValidator(0.0, 1.0)
        self._validators['ext_context.positive_correlation_threshold'] = RangeValidator(0.0, 1.0)
        self._validators['ext_result.correlation_test'] = TypeValidator([ItemCorrelationTest])
        self._validators['ext_result.negative_correlation_threshold'] = RangeValidator(0.0, 1.0)
        self._validators['ext_result.positive_correlation_threshold'] = RangeValidator(0.0, 1.0)
        self._validators['features'] = SupportedFeatureValidator()
        self._validators['goal_pursuit_strategy.reliability.max_penalty'] = RangeValidator(0.0, exclude=[0.0])
        self._validators['learning_rate'] = RangeValidator(0.0, 1.0)
        self._validators['output_format'] = TypeValidator([str])
        self._validators['random_exploratory_strategy.epsilon.decay.min'] = RangeValidator(low=0.0)
        self._validators['random_exploratory_strategy.epsilon.decay.rate'] = RangeValidator(0.0, 1.0,
                                                                                            exclude=[0.0, 1.0])
        self._validators['random_exploratory_strategy.epsilon.initial'] = RangeValidator(0.0, 1.0)
        self._validators['reliability_threshold'] = RangeValidator(0.0, 1.0)
        self._validators['rng_seed'] = TypeValidator([int])
        self._validators['schema_selection.weights.explore_weight'] = RangeValidator(0.0, 1.0)
        self._validators['schema_selection.weights.goal_weight'] = RangeValidator(0.0, 1.0)
        self._validators['verbosity'] = TypeValidator([Verbosity])

        # used by reliability_values
        self._defaults['habituation_exploratory_strategy.decay.rate'] = 0.95
        self._defaults['habituation_exploratory_strategy.multiplier'] = 10.0

        # TODO: is there any way to do validators for?
        # TODO:    correlation_test,
        # TODO:    item_type,
        # TODO:    composite_item_type,
        # TODO:    schema_type

    def reset(self):
        self._params = dict(self._defaults)


global_params: GlobalParams = GlobalParams()


def display_params() -> None:
    info(f'Global Parameters:')
    for param, value in global_params:
        is_default_value = value == global_params.defaults.get(param, None)
        info(f'\t{param} = \'{value}\' [DEFAULT: {is_default_value}]')


def _output_fd(level: Verbosity) -> TextIO:
    return sys.stdout if level < Verbosity.WARN else sys.stderr


def _timestamp() -> str:
    return datetime.now().isoformat()


def display_message(message: str, level: Verbosity) -> None:
    verbosity = GlobalParams().get('verbosity')
    output_format = GlobalParams().get('output_format')

    if level >= verbosity:
        out = output_format.format(timestamp=_timestamp(), severity=level.name, message=message)
        print(out, file=_output_fd(level), flush=True)


def trace(message):
    display_message(message=message, level=Verbosity.TRACE)


def debug(message):
    display_message(message=message, level=Verbosity.DEBUG)


def info(message):
    display_message(message=message, level=Verbosity.INFO)


def warn(message):
    display_message(message=message, level=Verbosity.WARN)


def error(message):
    display_message(message=message, level=Verbosity.ERROR)


def fatal(message):
    display_message(message=message, level=Verbosity.FATAL)


_rng = None
_seed = None


def rng():
    global _rng
    global _seed

    new_seed = GlobalParams().get('rng_seed')
    if new_seed != _seed:
        warn(f'(Re-)initializing random number generator using seed="{new_seed}".')
        warn(f'For reproducibility, you should also set "PYTHONHASHSEED={new_seed}" in your environment variables.')

        # setting globals
        _rng = np.random.default_rng(new_seed)
        _seed = new_seed

    return _rng
