from typing import Any
from typing import Optional

from schema_mechanism.core import Action
from schema_mechanism.core import CompositeItem
from schema_mechanism.core import GlobalStats
from schema_mechanism.core import Schema
from schema_mechanism.core import StateAssertion
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


class MockCompositeItem(CompositeItem):
    def __init__(self,
                 source: StateAssertion,
                 avg_accessible_value: float):
        super().__init__(source)

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


class MockSchema(Schema):
    def __init__(self,
                 action: Action,
                 context: Optional[StateAssertion] = None,
                 result: Optional[StateAssertion] = None,
                 reliability: float = None):
        super().__init__(action=action, context=context, result=result)

        self._reliability = reliability

    @property
    def reliability(self) -> float:
        return super().reliability if self._reliability is None else self._reliability
