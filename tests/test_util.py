import unittest
from unittest import TestCase

import numpy as np

from schema_mechanism.util import BoundedSet
from schema_mechanism.util import DefaultDictWithKeyFactory
from schema_mechanism.util import equal_weights
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


class TestFunctions(unittest.TestCase):
    def setUp(self) -> None:
        common_test_setup()

    def test_equal_weights(self):
        # test: n = 0 should return an empty array
        weights = equal_weights(0)

        self.assertIsInstance(weights, np.ndarray)
        self.assertEqual(0, len(weights))

        # test: n < 0 should raise a ValueError
        self.assertRaises(ValueError, lambda: equal_weights(-1))

        # test: n >= 1 should return numpy ndarray containing n elements that sum close to 1.0
        try:
            for n in range(1, 100):
                weights = equal_weights(n)
                self.assertIsInstance(weights, np.ndarray)
                self.assertEqual(n, len(weights))
                self.assertAlmostEqual(1.0, sum(weights))
        except ValueError as e:
            self.fail(f'Unexpected exception: {str(e)}')


# a default factory method used for testing DefaultDictWithKeyFactory
def default_key_factory(key):
    return f'missing {key}'


class TestDefaultDictWithKeyFactory(TestCase):
    def setUp(self) -> None:
        common_test_setup()

    def test_init(self):
        dict_with_factory = DefaultDictWithKeyFactory(default_key_factory)

        # test: default factory should have been set properly by initializer
        self.assertIs(default_key_factory, dict_with_factory.default_factory)

        # test: dictionary should be initially empty
        self.assertEqual(0, len(dict_with_factory))

    def test_default_factory(self):
        dict_with_factory = DefaultDictWithKeyFactory(default_key_factory)

        # test: non-missing keys should return previously set values
        dict_with_factory['key1'] = 'value1'
        dict_with_factory['key2'] = 'value2'
        dict_with_factory['key3'] = 'value3'

        self.assertEqual('value1', dict_with_factory['key1'])
        self.assertEqual('value2', dict_with_factory['key2'])
        self.assertEqual('value3', dict_with_factory['key3'])

        # test: missing values should be supplied by key factory (in this case, the return value of default_key_factory
        for key in ['key4', 'key5', 'key6']:
            len_before_lookup = len(dict_with_factory)
            self.assertEqual(default_key_factory(key), dict_with_factory[key])
            len_after_lookup = len(dict_with_factory)

            # test: the size of the dictionary should have been increased by one
            self.assertEqual(len_before_lookup + 1, len_after_lookup)

    def test_without_default_factory(self):
        # test: KeyError should be raised if a default factory is not set
        dict_without_factory = DefaultDictWithKeyFactory()

        self.assertRaises(KeyError, lambda: dict_without_factory['missing'])
