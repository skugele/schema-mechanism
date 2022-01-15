from unittest import TestCase

from schema_mechanism.data_structures import Action
from test_share.test_func import is_eq_with_null_is_false
from test_share.test_func import is_hash_consistent
from test_share.test_func import is_hash_same_for_equal_objects


class TestAction(TestCase):
    def setUp(self) -> None:
        self.a = Action(label='label')

    def test_init(self):
        self.assertEqual('label', self.a.label)

        # a globally unique id (uid) should be assigned for each action
        self.assertIsNotNone(self.a.uid)

        # uses a set to test uniqueness of a large number of actions (set size should equal to the number of actions)
        n_actions = 100_000
        self.assertTrue(n_actions, len(set([Action() for _ in range(n_actions)])))

    def test_eq(self):
        a1 = Action()
        a2 = Action()

        self.assertEqual(a1, a1)
        self.assertNotEqual(a1, a2)

        # labels should not be used for equality
        self.assertNotEqual(Action('label 1'), Action('label 1'))

        self.assertTrue(is_eq_with_null_is_false(a1))

    def test_hash(self):
        self.assertEqual(int, type(hash(Action)))
        self.assertTrue(is_hash_consistent(Action()))

        self.assertTrue(is_hash_same_for_equal_objects(self.a, self.a))
        self.assertTrue(is_hash_same_for_equal_objects(self.a, self.a.copy()))

    def test_copy(self):
        self.assertIsNot(self.a, self.a.copy())
        self.assertEqual(self.a, self.a.copy())
