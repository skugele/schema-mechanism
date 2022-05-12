from __future__ import annotations

import logging
from collections import Collection
from collections import ItemsView
from collections import defaultdict
from enum import Enum
from enum import auto
from time import time
from typing import Any
from typing import Optional

import numpy as np

from schema_mechanism.util import Singleton
from schema_mechanism.validate import MultiValidator
from schema_mechanism.validate import NULL_VALIDATOR
from schema_mechanism.validate import RangeValidator
from schema_mechanism.validate import TypeValidator
from schema_mechanism.validate import Validator

logger = logging.getLogger(__name__)


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

    def __getstate__(self) -> dict[str, Any]:
        return {'_params': self._params}

    def __setstate__(self, state: dict[str:Any]) -> None:
        gp = GlobalParams()
        for key in state:
            setattr(gp, key, state[key])

    @property
    def defaults(self) -> dict[str, Any]:
        return self._defaults

    @property
    def validators(self) -> dict[str, Any]:
        return self._validators

    def set(self, name: str, value: Any) -> None:
        if name not in self._params:
            logger.warning(f'Parameter "{name}" does not exist. Creating new parameter.')

        # raises ValueError if new value is invalid
        self._validators[name](value)
        self._params[name] = value

    def get(self, name: str) -> Any:
        if name not in self._params:
            logger.warning(f'Parameter "{name}" does not exist.')

        return self._params.get(name)

    def _set_defaults(self) -> None:
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

        # used by backward_chains (supports composite action)
        self._defaults['backward_chains.max_len'] = 5
        self._defaults['backward_chains.update_frequency'] = 0.01

        # composite actions are created for novel result states that have values that are greater than the baseline
        # value by AT LEAST this amount
        self._defaults['composite_actions.learn.min_baseline_advantage'] = 1.0

        # default features
        self._defaults['features'] = {
            SupportedFeature.COMPOSITE_ACTIONS,
            SupportedFeature.EC_DEFER_TO_MORE_SPECIFIC_SCHEMA,
            SupportedFeature.EC_MOST_SPECIFIC_ON_MULTIPLE,
            SupportedFeature.ER_POSITIVE_ASSERTIONS_ONLY,
            SupportedFeature.EC_POSITIVE_ASSERTIONS_ONLY,
            SupportedFeature.ER_SUPPRESS_UPDATE_ON_EXPLAINED,
        }

    def _set_validators(self):
        self._validators['backward_chains.max_len'] = MultiValidator([TypeValidator([int]), RangeValidator(low=0)])
        self._validators['backward_chains.update_frequency'] = RangeValidator(0.0, 1.0)
        self._validators['composite_actions.learn.min_baseline_advantage'] = TypeValidator([float])
        self._validators['ext_context.negative_correlation_threshold'] = RangeValidator(0.0, 1.0)
        self._validators['ext_context.positive_correlation_threshold'] = RangeValidator(0.0, 1.0)
        self._validators['ext_result.negative_correlation_threshold'] = RangeValidator(0.0, 1.0)
        self._validators['ext_result.positive_correlation_threshold'] = RangeValidator(0.0, 1.0)
        self._validators['features'] = SupportedFeatureValidator()
        self._validators['learning_rate'] = RangeValidator(0.0, 1.0)
        self._validators['reliability_threshold'] = RangeValidator(0.0, 1.0)
        self._validators['rng_seed'] = TypeValidator([int])

    def reset(self):
        self._params = dict(self._defaults)


global_params: GlobalParams = GlobalParams()


def display_params() -> None:
    logger.info(f'Global Parameters:')
    for param, value in global_params:
        is_default_value = value == global_params.defaults.get(param, None)
        logger.info(f'\t{param} = \'{value}\' [DEFAULT: {is_default_value}]')


_rng = None
_seed = None


def rng():
    global _rng
    global _seed

    new_seed = GlobalParams().get('rng_seed')
    if new_seed != _seed:
        logger.warning(f'(Re-)initializing random number generator using seed="{new_seed}".')
        logger.warning(
            f'For reproducibility, you should also set "PYTHONHASHSEED={new_seed}" in your environment variables.')

        # setting globals
        _rng = np.random.default_rng(new_seed)
        _seed = new_seed

    return _rng
