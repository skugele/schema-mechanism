from unittest import TestCase

from schema_mechanism.data_structures import State
from test_share.test_func import is_eq_consistent
from test_share.test_func import is_eq_reflexive
from test_share.test_func import is_eq_symmetric
from test_share.test_func import is_eq_transitive
from test_share.test_func import is_eq_with_null_is_false
from test_share.test_func import is_hash_consistent
from test_share.test_func import is_hash_same_for_equal_objects


class TestState(TestCase):
    def setUp(self) -> None:
        self.s = State(elements=['1', '2', '3'])
        self.s_copy = State(elements=['1', '2', '3'])
        self.s_copy_copy = State(elements=['1', '2', '3'])

        self.s_disjoint = State(elements=['4', '5'])
        self.s_conjoint = State(elements=['1', '2', '4'])
        self.s_contained = State(elements=[*self.s.elements, 100])

        self.s_empty = State(elements=[], label='empty')

    def test_init(self):
        self.assertEqual(0, len(self.s_empty.elements))
        self.assertEqual(frozenset([]), self.s_empty.elements)
        self.assertEqual('empty', self.s_empty.label)

        self.assertEqual(3, len(self.s.elements))
        self.assertEqual(frozenset(['1', '2', '3']), self.s.elements)
        self.assertEqual(None, self.s.label)

    def test_eq(self):
        self.assertEqual(self.s, self.s)
        self.assertEqual(self.s, self.s_copy)
        self.assertNotEqual(self.s, self.s_disjoint)
        self.assertNotEqual(self.s, self.s_conjoint)
        self.assertNotEqual(self.s, self.s_contained)

        self.assertTrue(is_eq_reflexive(self.s))
        self.assertTrue(is_eq_symmetric(x=self.s, y=self.s_copy))
        self.assertTrue(is_eq_transitive(x=self.s, y=self.s_copy, z=self.s_copy_copy))
        self.assertTrue(is_eq_consistent(x=self.s, y=self.s_copy))
        self.assertTrue(is_eq_with_null_is_false(self.s))

    def test_hash(self):
        self.assertIsInstance(hash(self.s), int)
        self.assertTrue(is_hash_consistent(self.s))
        self.assertTrue(is_hash_same_for_equal_objects(x=self.s, y=self.s_copy))
