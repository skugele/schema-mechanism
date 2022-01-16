from unittest import TestCase

from test_share.test_classes import MockObservable
from test_share.test_classes import MockObserver


class TestObserverAndObservables(TestCase):
    def test(self):
        observable = MockObservable()

        # test register
        n_observers = 10
        observers = [MockObserver() for _ in range(n_observers)]

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

    def ConcreteObserver(self):
        pass
