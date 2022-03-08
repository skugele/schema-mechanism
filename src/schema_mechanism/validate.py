from abc import ABC
from abc import abstractmethod
from collections import Callable
from collections import Iterable
from typing import Any
from typing import Optional
from typing import Type

import numpy as np


class Validator(ABC):
    @abstractmethod
    def __call__(self, value: Any) -> None:
        """ Generates a ValueError if validation fails. """
        pass


class AcceptAllValidator(Validator):
    def __call__(self, value: Any):
        pass


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
        if (value < self.low) or (value > self.high) or (value in self.exclude):
            raise ValueError(f'Value must be between {self.low} and {self.high} excluding {self.exclude or None}.')


class TypeValidator(Validator):
    def __init__(self, accept_set: Iterable[Type]) -> None:
        self.accept_set = list(accept_set) if accept_set else list()
        if not self.accept_set:
            raise ValueError('TypeValidator\'s accept_set cannot be empty.')

    def __call__(self, value: Any) -> None:
        if value is None:
            return

        accepted = any({isinstance(value, type_) for type_ in self.accept_set})
        if not accepted:
            raise ValueError(f'Type not supported: {type(value)}.')


class SubClassValidator(Validator):
    def __init__(self, accept_set: Iterable[Type]) -> None:
        self.accept_set = list(accept_set) if accept_set else list()
        if not self.accept_set:
            raise ValueError('SubClassValidator\'s accept_set cannot be empty.')

    def __call__(self, value: Any) -> None:
        if value is None:
            return

        accepted = any({issubclass(value, type_) for type_ in self.accept_set})
        if not accepted:
            raise ValueError(f'Subclass not supported: {value}.')


class BlackListValidator(Validator):
    def __init__(self, reject_set: Iterable[Any]) -> None:
        self.reject_set = list(reject_set) if reject_set else list()
        if not self.reject_set:
            raise ValueError('BlackListValidator\'s reject_set cannot be empty.')

    def __call__(self, value: Any) -> None:
        if value in self.reject_set:
            raise ValueError(f'Value not supported: {value}.')


class WhiteListValidator(Validator):
    def __init__(self, accept_set: Iterable[Any]) -> None:
        self.accept_set = list(accept_set) if accept_set else list()
        if not self.accept_set:
            raise ValueError('WhiteListValidator\'s accept_set cannot be empty.')

    def __call__(self, value: Any) -> None:
        if value not in self.accept_set:
            raise ValueError(f'Value not supported: {value}.')


class MultiValidator(Validator):
    def __init__(self, validators: Iterable[Validator]) -> None:
        self.validators = list(validators) if validators else list()
        if not self.validators:
            raise ValueError('MultiValidator\'s list of validator_list cannot be empty.')

    def __call__(self, value: Any) -> None:
        for v in self.validators:
            v(value)


class ElementValidator(Validator):
    def __init__(self, validator: Validator) -> None:
        self.validator = validator
        if not self.validator:
            raise ValueError('ElementValidator\'s validator must be defined in initializer.')

    def __call__(self, value: Iterable[Any]) -> None:
        for v in value:
            self.validator(v)


class CustomValidator(Validator):
    def __init__(self, validator: Callable[[Any], None]):
        self.validator = validator
        if not self.validator:
            raise ValueError('CustomValidator\'s validator must be defined in initializer.')

    def __call__(self, value: Any) -> None:
        self.validator(value)


NULL_VALIDATOR = AcceptAllValidator()
POSITIVE_INTEGER_VALIDATOR = MultiValidator([TypeValidator([int]), RangeValidator(low=1)])
