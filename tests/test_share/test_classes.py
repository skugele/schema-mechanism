from collections import Collection
from typing import Any
from typing import Dict

from schema_mechanism.data_structures import ECItemStats
from schema_mechanism.data_structures import ERItemStats
from schema_mechanism.data_structures import ExtendedContext
from schema_mechanism.data_structures import ExtendedResult
from schema_mechanism.data_structures import Item
from schema_mechanism.data_structures import ItemPoolStateView
from schema_mechanism.data_structures import SchemaStats
from schema_mechanism.data_structures import StateElement
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


class ECItemStatsTestWrapper(ECItemStats):
    def __init__(self, schema_stats: SchemaStats = None):
        ss = schema_stats or SchemaStats()
        super().__init__(schema_stats=ss)

    def update(self, on: bool, success: bool, count: int = 1) -> None:
        # success is not used in ECItemStats statistical calculations
        self._schema_stats.update(activated=True, success=success, count=count)
        super().update(on=on, success=success, count=count)


class ERItemStatsTestWrapper(ERItemStats):
    def __init__(self, schema_stats: SchemaStats = None):
        ss = schema_stats or SchemaStats()
        super().__init__(schema_stats=ss)

    def update(self, on: bool, activated: bool, count: int = 1) -> None:
        # success is not used in ERItemStats statistical calculations
        self._schema_stats.update(activated=activated, success=True, count=count)
        super().update(on=on, activated=activated, count=count)


class ExtendedContextTestWrapper(ExtendedContext):
    def __init__(self, schema_stats: SchemaStats = None):
        ss = schema_stats or SchemaStats()
        super().__init__(schema_stats=SchemaStats())

    def update(self, item: Item, on: bool, success: bool, count: int = 1) -> None:
        # activated is not used in ExtendedResult statistical calculations
        self._schema_stats.update(activated=True, success=success, count=count)
        super().update(item=item, on=on, success=success, count=count)

    def update_all(self, view: ItemPoolStateView, success: bool, count: int = 1) -> None:
        # activated is not used in ExtendedResult statistical calculations
        self._schema_stats.update(activated=True, success=success, count=count)
        super().update_all(view=view, success=success, count=count)


class ExtendedResultTestWrapper(ExtendedResult):
    def __init__(self, schema_stats: SchemaStats = None):
        ss = schema_stats or SchemaStats()
        super().__init__(schema_stats=ss)

    def update(self, item: Item, on: bool, activated=False, count: int = 1) -> None:
        # success is not used in ExtendedResult statistical calculations
        self._schema_stats.update(activated=activated, success=True, count=count)
        super().update(item=item, on=on, activated=activated, count=count)

    def update_all(self, activated=False, new: Collection[StateElement] = None, lost: Collection[StateElement] = None,
                   count: int = 1) -> None:
        # success is not used in ExtendedResult statistical calculations
        self._schema_stats.update(activated=activated, success=True, count=count)
        super().update_all(activated=activated, new=new, lost=lost, count=count)
