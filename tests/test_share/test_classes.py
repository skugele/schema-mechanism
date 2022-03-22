from collections import deque
from typing import Any
from typing import Optional

from schema_mechanism.core import Action
from schema_mechanism.core import CompositeItem
from schema_mechanism.core import GlobalStats
from schema_mechanism.core import Schema
from schema_mechanism.core import StateAssertion
from schema_mechanism.core import SymbolicItem
from schema_mechanism.modules import SchemaSelection
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
                 primitive_value: Optional[float] = None,
                 avg_accessible_value: Optional[float] = None,
                 **kwargs):
        super().__init__(source, primitive_value)

        self._mock_avg_accessible_value = avg_accessible_value

    @property
    def avg_accessible_value(self) -> float:
        return (
            super().avg_accessible_value
            if self._mock_avg_accessible_value is None
            else self._mock_avg_accessible_value
        )

    @avg_accessible_value.setter
    def avg_accessible_value(self, value: float) -> None:
        self._mock_avg_accessible_value = value

    @property
    def delegated_value(self) -> float:
        return self.avg_accessible_value - GlobalStats().baseline_value


class MockCompositeItem(CompositeItem):
    def __init__(self,
                 source: StateAssertion,
                 avg_accessible_value: float,
                 **kwargs):
        super().__init__(source)

        self._mock_avg_accessible_value = avg_accessible_value

    @property
    def avg_accessible_value(self) -> float:
        return (
            super().avg_accessible_value
            if self._mock_avg_accessible_value is None
            else self._mock_avg_accessible_value
        )

    @avg_accessible_value.setter
    def avg_accessible_value(self, value: float) -> None:
        self._mock_avg_accessible_value = value

    @property
    def delegated_value(self) -> float:
        return self.avg_accessible_value - GlobalStats().baseline_value


class MockSchema(Schema):
    def __init__(self,
                 action: Action,
                 context: Optional[StateAssertion] = None,
                 result: Optional[StateAssertion] = None,
                 reliability: float = None,
                 avg_duration: float = None,
                 cost: float = None,
                 **kwargs):
        super().__init__(action=action, context=context, result=result, **kwargs)

        self._mock_reliability = reliability
        self._mock_avg_duration = avg_duration
        self._mock_cost = cost

    @property
    def reliability(self) -> float:
        return (
            super().reliability
            if self._mock_reliability is None
            else self._mock_reliability
        )

    @property
    def avg_duration(self) -> float:
        return (
            super().avg_duration
            if self._mock_avg_duration is None
            else self._mock_avg_duration
        )

    @property
    def cost(self) -> float:
        return (
            super().cost
            if self._mock_cost is None
            else self._mock_cost
        )


class MockSchemaSelection(SchemaSelection):
    def __init__(self, pending_schemas: Optional[deque[Schema]] = None, **kwargs):
        super().__init__(**kwargs)

        self._pending_schemas = pending_schemas
