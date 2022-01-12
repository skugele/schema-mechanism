from schema_mechanism.util import Observable
from schema_mechanism.util import Observer


class TestObserver(Observer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_received = 0
        self.last_message = None

    def receive(self, *args, **kwargs) -> None:
        self.n_received += 1
        self.last_message = {'args': args, 'kwargs': kwargs}


class TestObservable(Observable):
    pass
