import logging
from collections import defaultdict
from typing import Any
from typing import ItemsView
from typing import Optional

from schema_mechanism.validate import NULL_VALIDATOR
from schema_mechanism.validate import Validator

logger = logging.getLogger(__name__)


class GlobalParams:

    def __init__(self, parameters: dict[str, Any] = None, validators: dict[str, Validator] = None) -> None:
        self.parameters: dict[str, Any] = dict()
        self.validators: dict[str, Validator] = defaultdict(lambda: NULL_VALIDATOR)

        if parameters:
            self.parameters.update(parameters)

        if validators:
            self.validators.update(validators)

    def __iter__(self) -> ItemsView[str, Any]:
        yield from self.parameters.items()

    def __eq__(self, other) -> bool:
        if isinstance(other, GlobalParams):
            return all(
                (
                    # generator expression for conditions to allow lazy evaluation
                    condition for condition in
                    [
                        other.parameters == self.parameters,
                        other.validators == self.validators
                    ]
                )
            )
        return False if other is None else NotImplemented

    def __str__(self) -> str:
        parameter_details = []
        for param, value in self:
            parameter_details.append(f'{param} = \'{value}\'')
        return '; '.join(parameter_details)

    def __contains__(self, key: str) -> bool:
        return key in self.parameters

    def set(self, name: str, value: Any, validator: Optional[Validator] = None) -> None:
        if validator:
            self.validators[name] = validator

        # raises ValueError if new value is invalid
        self.validators[name](value)
        self.parameters[name] = value

        logger.info(f'Setting parameter "{name}" to value "{value}".')

    def get(self, name: str) -> Any:
        if name not in self.parameters:
            logger.warning(f'Parameter "{name}" does not exist.')

        return self.parameters.get(name)

    def reset(self):
        self.parameters.clear()
        self.validators.clear()
