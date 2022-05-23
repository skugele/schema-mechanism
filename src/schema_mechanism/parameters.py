import logging
from collections import defaultdict
from typing import Any
from typing import ItemsView
from typing import Optional

from schema_mechanism.validate import NULL_VALIDATOR
from schema_mechanism.validate import Validator

logger = logging.getLogger(__name__)


class GlobalParams:

    def __init__(self) -> None:
        self._validators: dict[str, Validator] = defaultdict(lambda: NULL_VALIDATOR)
        self._params: dict[str, Any] = dict()

    def __iter__(self) -> ItemsView[str, Any]:
        yield from self._params.items()

    def __eq__(self, other) -> bool:
        if isinstance(other, GlobalParams):
            return all(
                (
                    # generator expression for conditions to allow lazy evaluation
                    condition for condition in
                    [
                        other._params == self._params,
                        other._validators == self._validators
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
        return key in self._params

    @property
    def parameters(self) -> dict[str, Any]:
        return self._params

    @property
    def validators(self) -> dict[str, Any]:
        return self._validators

    def set(self, name: str, value: Any, validator: Optional[Validator] = None) -> None:
        if validator:
            self._validators[name] = validator

        # raises ValueError if new value is invalid
        self._validators[name](value)
        self._params[name] = value

        logger.info(f'Setting parameter "{name}" to value "{value}".')

    def get(self, name: str) -> Any:
        if name not in self._params:
            logger.warning(f'Parameter "{name}" does not exist.')

        return self._params.get(name)

    def reset(self):
        self._params.clear()
        self._validators.clear()
