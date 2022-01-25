from typing import Any
from typing import Dict

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
    def last_message(self) -> Dict[str, Any]:
        return {} if not self.messages else self.messages[-1]

    def receive(self, *args, **kwargs) -> None:
        self.messages.append({'args': args, 'kwargs': kwargs})


class MockObservable(Observable):
    pass
