from schema_mechanism.data_structures import ExtendedContext
from schema_mechanism.data_structures import ExtendedResult
from schema_mechanism.data_structures import Item
from schema_mechanism.util import Observable
from schema_mechanism.util import Observer


class MockObserver(Observer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_received = 0
        self.last_message = None

    def receive(self, *args, **kwargs) -> None:
        self.n_received += 1
        self.last_message = {'args': args, 'kwargs': kwargs}


class MockObservable(Observable):
    pass


class ExtendedContextTestWrapper(ExtendedContext):
    def update(self, item: Item, on: bool, success: bool, count: int = 1) -> None:
        # activated is not used in ExtendedResult statistical calculations
        self._schema_stats.update(activated=True, success=success, count=count)
        super().update(item=item, on=on, success=success, count=count)


class ExtendedResultTestWrapper(ExtendedResult):
    def update(self, item: Item, on: bool, activated=False, count: int = 1) -> None:
        # success is not used in ExtendedResult statistical calculations
        self._schema_stats.update(activated=activated, success=True, count=count)
        super().update(item=item, on=on, activated=activated, count=count)