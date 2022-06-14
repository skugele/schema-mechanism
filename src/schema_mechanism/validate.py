import logging
from abc import ABC
from abc import abstractmethod
from collections import Iterable
from typing import Any
from typing import Optional

import numpy as np

from schema_mechanism.share import SupportedFeature

logger = logging.getLogger(__name__)


class Validator(ABC):
    @abstractmethod
    def __call__(self, value: Any) -> None:
        """ Generates a ValueError if validation fails. """


class AcceptAllValidator(Validator):
    def __call__(self, value: Any):
        pass

    def __eq__(self, other) -> bool:
        if isinstance(other, AcceptAllValidator):
            return True
        return False if None else NotImplemented

    def __hash__(self) -> int:
        return hash(self.__class__.__name__)


# TODO: This class could be enhanced to support any object type that is orderable
class RangeValidator(Validator):

    def __init__(self,
                 low: Optional[float] = None,
                 high: Optional[float] = None,
                 exclude: Optional[Iterable[float]] = None) -> None:
        self.low = -np.inf if low is None else low
        self.high = np.inf if high is None else high
        self.exclude = frozenset(exclude) if exclude else frozenset()

        if self.low >= self.high:
            raise ValueError('RangeValidator\'s low value must be less than high value.')

    def __call__(self, value: Any) -> None:
        try:
            if (value < self.low) or (value > self.high) or (value in self.exclude):
                raise ValueError(f'Value must be between {self.low} and {self.high} excluding {self.exclude or None}.')
        except TypeError:
            raise ValueError(f'Value not supported: {value}')

    def __eq__(self, other) -> bool:
        if isinstance(other, RangeValidator):
            return all(
                (
                    # generator expression for conditions to allow lazy evaluation
                    condition for condition in
                    [
                        other.low == self.low,
                        other.high == self.high,
                        other.exclude == self.exclude
                    ]
                )
            )
        return False if other is None else NotImplemented

    def __hash__(self) -> int:
        arg_dict = {'low': hash(self.low), 'high': hash(self.high), 'exclude': hash(self.exclude)}
        return hash(self.__class__.__name__ + str(arg_dict))


class BlackListValidator(Validator):
    def __init__(self, reject_set: Iterable[Any]) -> None:
        self.reject_set = frozenset(reject_set) if reject_set else frozenset()
        if not self.reject_set:
            raise ValueError('BlackListValidator\'s reject_set cannot be empty.')

    def __call__(self, value: Any) -> None:
        # TypeError will be raised if value is not hashable
        if value in self.reject_set:
            raise ValueError(f'Value not supported: {value}.')

    def __eq__(self, other) -> bool:
        if isinstance(other, BlackListValidator):
            return self.reject_set == other.reject_set
        return False if other is None else NotImplemented

    def __hash__(self) -> int:
        return hash(self.__class__.__name__ + str(self.reject_set))


class WhiteListValidator(Validator):
    def __init__(self, accept_set: Iterable[Any]) -> None:
        self.accept_set = frozenset(accept_set) if accept_set else frozenset()
        if not self.accept_set:
            raise ValueError('WhiteListValidator\'s accept_set cannot be empty.')

    def __call__(self, value: Any) -> None:
        # TypeError will be raised if value is not hashable
        if value not in self.accept_set:
            raise ValueError(f'Value not supported: {value}.')

    def __eq__(self, other) -> bool:
        if isinstance(other, WhiteListValidator):
            return self.accept_set == other.accept_set
        return False if other is None else NotImplemented

    def __hash__(self) -> int:
        return hash(self.__class__.__name__ + str(self.accept_set))


class MultiValidator(Validator):
    def __init__(self, validators: Iterable[Validator]) -> None:
        self.validators = frozenset(validators) if validators else frozenset()
        if not self.validators:
            raise ValueError('MultiValidator\'s list of validator_list cannot be empty.')

    def __call__(self, value: Any) -> None:
        for v in self.validators:
            v(value)

    def __eq__(self, other) -> bool:
        if isinstance(other, MultiValidator):
            return self.validators == other.validators
        return False if other is None else NotImplemented

    def __hash__(self) -> int:
        return hash(self.__class__.__name__ + str(self.validators))


class ElementWiseValidator(Validator):
    """ Applies a validator to each element of an iterable. """

    def __init__(self, validator: Validator) -> None:
        self.validator = validator
        if not self.validator:
            raise ValueError('ElementValidator\'s validator must be defined in initializer.')

    def __call__(self, value: Iterable[Any]) -> None:

        try:
            for v in value:
                self.validator(v)

        # this will occur if the value is not iterable
        except TypeError:
            raise ValueError(f'Value not supported: {value}.')

    def __eq__(self, other) -> bool:
        if isinstance(other, ElementWiseValidator):
            return self.validator == other.validator
        return False if None else NotImplemented

    def __hash__(self) -> int:
        return hash(self.__class__.__name__ + str(self.validator))


class SupportedFeatureValidator(Validator):
    def __call__(self, features: Optional[Iterable[SupportedFeature]]) -> None:
        features = set(features)
        for value in features:
            if not isinstance(value, SupportedFeature):
                raise ValueError(f'Unsupported feature: {value}')

        if (SupportedFeature.EC_MOST_SPECIFIC_ON_MULTIPLE in features and
                SupportedFeature.EC_DEFER_TO_MORE_SPECIFIC_SCHEMA not in features):
            raise ValueError(f'The feature EC_MOST_SPECIFIC_ON_MULTIPLE requires EC_DEFER_TO_MORE_SPECIFIC_SCHEMA')

    def __eq__(self, other) -> bool:
        if isinstance(other, SupportedFeatureValidator):
            return True
        return False if None else NotImplemented

    def __hash__(self) -> int:
        return hash(self.__class__.__name__)


NULL_VALIDATOR = AcceptAllValidator()
