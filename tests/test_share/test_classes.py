from typing import Any

from schema_mechanism.core import GlobalStats
from schema_mechanism.core import SymbolicItem
from schema_mechanism.util import Observable
from schema_mechanism.util import Observer


class MockObserver(Observer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.messages = []

    @property
    def n_received(self) -> int:
        return len(self.messages)

    @property
    def last_message(self) -> dict[str, Any]:
        return {} if not self.messages else self.messages[-1]

    def receive(self, *args, **kwargs) -> None:
        self.messages.append({'args': args, 'kwargs': kwargs})


class MockObservable(Observable):
    pass


class MockSymbolicItem(SymbolicItem):
    def __init__(self,
                 source: str,
                 primitive_value: float,
                 avg_accessible_value: float):
        super().__init__(source, primitive_value)

        self._avg_accessible_value = avg_accessible_value

    @property
    def avg_accessible_value(self) -> float:
        return self._avg_accessible_value

    @avg_accessible_value.setter
    def avg_accessible_value(self, value: float) -> None:
        self._avg_accessible_value = value

    @property
    def delegated_value(self) -> float:
        return self._avg_accessible_value - GlobalStats().baseline_value
