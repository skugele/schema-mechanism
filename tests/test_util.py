from unittest import TestCase

from schema_mechanism.util import Observable
from schema_mechanism.util import Observer
from schema_mechanism.util import get_unique_id


class Test(TestCase):
    def test_get_unique_id(self):
        ids = []
        for _ in range(1000):
            uid = get_unique_id()
            self.assertNotIn(uid, ids)

            ids.append(uid)


class TestObserverAndObservables(TestCase):
    class ConcreteObserver(Observer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.n_received = 0
            self.last_message = None

        def receive(self, *args, **kwargs) -> None:
            self.n_received += 1
            self.last_message = {'args': args, 'kwargs': kwargs}

    class ConcreteObservable(Observable):
        pass

    def test(self):
        observable = self.ConcreteObservable()

        # test register
        n_observers = 10
        observers = [self.ConcreteObserver() for _ in range(n_observers)]

        n_registered = 0
        for obs in observers:
            observable.register(obs)
            n_registered += 1

            self.assertEqual(n_registered, len(observable.observers))
            self.assertTrue(obs in observable.observers)

        # test notify
        observable.notify_all('args', keyword='kwargs')
        for obs in observers:
            self.assertEqual(1, obs.n_received)
            self.assertEqual('args', obs.last_message['args'][0])
            self.assertEqual('kwargs', obs.last_message['kwargs']['keyword'])

        # test unregister
        removed_obs = observers.pop()
        observable.unregister(removed_obs)

        self.assertEqual(len(observers), len(observable.observers))
        self.assertNotIn(removed_obs, observable.observers)
