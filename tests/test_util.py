from unittest import TestCase

from schema_mechanism.util import BoundedSet
from test_share.test_classes import MockObservable
from test_share.test_classes import MockObserver
from test_share.test_func import common_test_setup


class TestObserverAndObservables(TestCase):
    def setUp(self) -> None:
        common_test_setup()

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
        observable.notify_all(keyword='kwargs')
        for obs in observers:
            self.assertEqual(1, obs.n_received)
            self.assertEqual('kwargs', obs.last_message['kwargs']['keyword'])

        # test unregister
        removed_obs = observers.pop()
        observable.unregister(removed_obs)

        self.assertEqual(len(observers), len(observable.observers))
        self.assertNotIn(removed_obs, observable.observers)

    def ConcreteObserver(self):
        pass


class TestBoundedSet(TestCase):
    def setUp(self):
        common_test_setup()

        self.legal_values = set(range(10))
        self.illegal_values = {-1, 11, 'illegal', frozenset()}
        self.s = BoundedSet(accepted_values=self.legal_values)

    def test_init(self):
        # test: empty bounded set supported
        s = BoundedSet()
        self.assertSetEqual(set(), s)

        # test: bounded set initialized with non-empty iterable should include all of its values
        s = BoundedSet(self.legal_values, accepted_values=range(10))
        self.assertSetEqual(self.legal_values, s)

        # test: bounded set objects are instances of set
        self.assertIsInstance(s, set)

    def test_is_legal_value(self):
        # test: all legal values should return True
        for v in self.legal_values:
            self.assertTrue(self.s.is_legal_value(v))

        # test: illegal values should return False
        for v in self.illegal_values:
            self.assertFalse(self.s.is_legal_value(v))

    def test_check_values(self):
        # test: legal values should not raise a ValueError
        try:
            self.s.check_values(self.legal_values)
        except ValueError as e:
            self.fail(f'Unexpected ValueError encountered: {str(e)}')

        # test: illegal values should raise a ValueError
        for value in self.illegal_values:
            self.assertRaises(ValueError, lambda: self.s.check_values([value]))

    def test_add(self):
        # test: adding legal values should be included in set and not raise a ValueError
        for v in self.legal_values:
            self.s.add(v)
            self.assertIn(v, self.s)

        # test: illegal values should raise a ValueError on add
        for v in self.illegal_values:
            self.assertRaises(ValueError, lambda: self.s.add(v))

    def test_update(self):
        # test: adding legal values should be included in set and not raise a ValueError
        self.s.update(self.legal_values)
        self.assertSetEqual(self.legal_values, self.s)

        # test: illegal values should raise a ValueError on add
        self.assertRaises(ValueError, lambda: self.s.add(self.illegal_values))
