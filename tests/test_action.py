from unittest import TestCase

from schema_mechanism.core import Action
from test_share.test_func import common_test_setup
from test_share.test_func import satisfies_equality_checks
from test_share.test_func import satisfies_hash_checks


class TestAction(TestCase):
    def setUp(self) -> None:
        common_test_setup()

        self.a = Action(label='label')

    def test_init(self):
        self.assertEqual('label', self.a.label)

        # a globally unique id (uid) should be assigned for each action
        self.assertIsNotNone(self.a.uid)

        # uses a set to test uniqueness of a large number of actions (set size should equal to the number of actions)
        n_actions = 100_000
        self.assertTrue(n_actions, len(set([Action() for _ in range(n_actions)])))

    def test_eq(self):
        self.assertTrue(satisfies_equality_checks(obj=self.a, other=Action('other')))

        # test: labels should be used to determine equality (if they exist)
        self.assertEqual(Action('label 1'), Action('label 1'))
        self.assertNotEqual(Action('label 1'), Action('label 2'))
        self.assertNotEqual(Action(), Action())

    def test_hash(self):
        self.assertTrue(satisfies_hash_checks(obj=self.a))
