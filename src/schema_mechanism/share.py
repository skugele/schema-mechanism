from __future__ import annotations

import logging
import random
from collections import Collection
from collections import ItemsView
from collections import defaultdict
from enum import Enum
from typing import Any
from typing import Optional

import numpy as np

from schema_mechanism.validate import MultiValidator
from schema_mechanism.validate import NULL_VALIDATOR
from schema_mechanism.validate import RangeValidator
from schema_mechanism.validate import TypeValidator
from schema_mechanism.validate import Validator

logger = logging.getLogger(__name__)


class SupportedFeature(Enum):
    COMPOSITE_ACTIONS = 'COMPOSITE_ACTIONS'

    # "There is an embellishment of the marginal attribution algorithm--deferring to a more specific applicable schema--
    #  that often enables the discovery of an item whose relevance has been obscured." (see Drescher,1991, pp. 75-76)
    EC_DEFER_TO_MORE_SPECIFIC_SCHEMA = 'EC_DEFER_TO_MORE_SPECIFIC_SCHEMA'

    # "[another] embellishment also reduces redundancy: when a schema's extended context simultaneously detects the
    # relevance of several items--that is, their statistics pass the significance threshold on the same trial--the most
    # specific is chosen as the one for inclusion in a spin-off from that schema." (see Drescher, 1991, p. 77)
    #
    #     Note: Requires that EC_DEFER_TO_MORE_SPECIFIC is also enabled.
    EC_MOST_SPECIFIC_ON_MULTIPLE = 'EC_MOST_SPECIFIC_ON_MULTIPLE'

    # "The machinery's sensitivity to results is amplified by an embellishment of marginal attribution: when a given
    #  schema is idle (i.e., it has not just completed an activation), the updating of its extended result data is
    #  suppressed for any state transition which is explained--meaning that the transition is predicted as the result
    #  of a reliable schema whose activation has just completed." (see Drescher, 1991, p. 73)
    ER_SUPPRESS_UPDATE_ON_EXPLAINED = 'ER_SUPPRESS_UPDATE_ON_EXPLAINED'

    # Supports the creation of result spin-off schemas incrementally. This was not supported in the original schema
    # mechanism because of the proliferation of composite results that result. It is allowed here to facilitate
    # comparison and experimentation.
    ER_INCREMENTAL_RESULTS = 'ER_INCREMENTAL_RESULTS'


class SupportedFeatureValidator(Validator):
    def __call__(self, features: Optional[Collection[SupportedFeature]]) -> None:
        features = set(features)
        for value in features:
            if not isinstance(value, SupportedFeature):
                raise ValueError(f'Unsupported feature: {value}')

        if (SupportedFeature.EC_MOST_SPECIFIC_ON_MULTIPLE in features and
                SupportedFeature.EC_DEFER_TO_MORE_SPECIFIC_SCHEMA not in features):
            raise ValueError(f'The feature EC_MOST_SPECIFIC_ON_MULTIPLE requires EC_DEFER_TO_MORE_SPECIFIC_SCHEMA')

    def __eq__(self, other) -> bool:
        return isinstance(other, SupportedFeatureValidator)

    def __hash__(self) -> int:
        return hash(self.__class__.__name__)


class GlobalParams:

    def __init__(self) -> None:
        self._defaults: dict[str, Any] = dict()
        self._validators: dict[str, Validator] = defaultdict(lambda: NULL_VALIDATOR)
        self._params: dict[str, Any] = dict(self._defaults)

        self._set_validators()
        self._set_defaults()

        self.reset()

    def __iter__(self) -> ItemsView[str, Any]:
        yield from self._params.items()

    def __eq__(self, other) -> bool:
        if isinstance(other, GlobalParams):
            return all((other._params == self._params,
                        other._defaults == self._defaults,
                        other._validators == self._validators))
        return False if other is None else NotImplemented

    def __str__(self) -> str:
        parameter_details = []
        for param, value in self:
            is_default_value = value == self.defaults.get(param, None)
            parameter_details.append(f'{param} = \'{value}\' [DEFAULT: {is_default_value}]')
        return 'Global Parameters: ' + '; '.join(parameter_details)

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

    def reset(self):
        self._params = dict(self._defaults)

    def _set_defaults(self) -> None:
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


_rng: Optional[np.random.Generator] = None
_seed: Optional[int] = None


def set_random_seed(seed: int) -> None:
    global _seed
    _seed = seed

    random.seed(_seed)


def get_random_seed() -> Optional[int]:
    return _seed


def rng():
    global _rng
    global _seed

    seed = get_random_seed()
    if not _rng:
        logger.warning(f'Initializing random number generator using seed="{seed}".')
        logger.warning(
            f'For reproducibility, you should also set "PYTHONHASHSEED={seed}" in your environment variables.')

        # setting globals
        _rng = np.random.default_rng(seed)
        _seed = seed

    return _rng
