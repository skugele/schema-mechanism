from unittest import TestCase

from schema_mechanism.core import CompositeItem
from schema_mechanism.core import GlobalStats
from schema_mechanism.core import ItemPool
from schema_mechanism.core import State
from schema_mechanism.core import composite_items
from schema_mechanism.core import held_state
from schema_mechanism.core import lost_state
from schema_mechanism.core import new_state
from schema_mechanism.core import non_composite_items
from schema_mechanism.func_api import sym_items
from schema_mechanism.func_api import sym_state
from schema_mechanism.func_api import sym_state_assert
from test_share.test_classes import MockSymbolicItem
from test_share.test_func import common_test_setup
from test_share.test_func import is_eq_consistent
from test_share.test_func import is_eq_reflexive
from test_share.test_func import is_eq_symmetric
from test_share.test_func import is_eq_transitive
from test_share.test_func import is_eq_with_null_is_false
from test_share.test_func import is_hash_consistent
from test_share.test_func import is_hash_same_for_equal_objects


class TestState(TestCase):
    def setUp(self) -> None:
        common_test_setup()

        # adds items to item pool
        i1 = ItemPool().get('1', primitive_value=-1.0, avg_accessible_value=3.0, item_type=MockSymbolicItem)
        i2 = ItemPool().get('2', primitive_value=0.0, avg_accessible_value=1.0, item_type=MockSymbolicItem)
        i3 = ItemPool().get('3', primitive_value=1.0, avg_accessible_value=-3.0, item_type=MockSymbolicItem)
        i4 = ItemPool().get('4', primitive_value=-1.0, avg_accessible_value=0.0, item_type=MockSymbolicItem)
        i5 = ItemPool().get('5', primitive_value=0.0, avg_accessible_value=-1.0, item_type=MockSymbolicItem)

        self.s = State(elements=['1', '2', '3'])
        self.s_copy = State(elements=['1', '2', '3'])
        self.s_copy_copy = State(elements=['1', '2', '3'])

        self.s_disjoint = State(elements=['4', '5'])
        self.s_conjoint = State(elements=['1', '2', '4'])
        self.s_contained = State(elements=[*self.s.elements, 100])

        self.s_empty = State(elements=[], label='empty')

        GlobalStats(baseline_value=-1.0)

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

    def test_len(self):
        self.assertEqual(0, len(self.s_empty))
        self.assertEqual(3, len(self.s))

    def test_contains(self):
        for se in ['1', '2', '3']:
            self.assertIn(se, self.s)

        for se in ['4', '5', '6']:
            self.assertNotIn(se, self.s)
            self.assertNotIn(se, self.s_empty)

    def test_primitive_value(self):
        # empty state
        self.assertEqual(0.0, self.s_empty.primitive_value)

        # single element state (negative primitive value)
        self.assertEqual(-1.0, sym_state('1').primitive_value)

        # single element state (zero primitive value)
        self.assertEqual(0.0, sym_state('2').primitive_value)

        # single element state (positive primitive value)
        self.assertEqual(1.0, sym_state('3').primitive_value)

        # multiple element state (negative, zero, and positive primitive values)
        self.assertEqual(-1.0, sym_state('1,2').primitive_value)
        self.assertEqual(1.0, sym_state('2,3').primitive_value)
        self.assertEqual(0.0, sym_state('1,3').primitive_value)
        self.assertEqual(0.0, sym_state('1,2,3').primitive_value)

    def test_avg_accessible_value(self):
        # empty state
        self.assertEqual(0.0, self.s_empty.avg_accessible_value)

        # single element state (negative avg accessible value)
        self.assertEqual(-1.0, sym_state('5').avg_accessible_value)

        # single element state (zero avg accessible value)
        self.assertEqual(0.0, sym_state('4').avg_accessible_value)

        # single element state (positive avg accessible value)
        self.assertEqual(1.0, sym_state('2').avg_accessible_value)

        # multiple element state (negative, zero, and positive avg accessible value)
        self.assertEqual(3.0, sym_state('1,2').avg_accessible_value)
        self.assertEqual(1.0, sym_state('2,3').avg_accessible_value)
        self.assertEqual(0.0, sym_state('3,4').avg_accessible_value)
        self.assertEqual(-1.0, sym_state('3,5').avg_accessible_value)

    def test_delegated_value(self):
        # empty state
        self.assertEqual(0.0 - GlobalStats().baseline_value, self.s_empty.delegated_value)

        # single element state (negative avg accessible value)
        self.assertEqual(-1.0 - GlobalStats().baseline_value, sym_state('5').delegated_value)

        # single element state (zero avg accessible value)
        self.assertEqual(0.0 - GlobalStats().baseline_value, sym_state('4').delegated_value)

        # single element state (positive avg accessible value)
        self.assertEqual(1.0 - GlobalStats().baseline_value, sym_state('2').delegated_value)

        # multiple element state (negative, zero, and positive avg accessible value)
        self.assertEqual(3.0 - GlobalStats().baseline_value, sym_state('1,2').delegated_value)
        self.assertEqual(1.0 - GlobalStats().baseline_value, sym_state('2,3').delegated_value)
        self.assertEqual(0.0 - GlobalStats().baseline_value, sym_state('3,4').delegated_value)
        self.assertEqual(-1.0 - GlobalStats().baseline_value, sym_state('3,5').delegated_value)


class TestStateFunctions(TestCase):
    def setUp(self) -> None:
        common_test_setup()

        self.s_1 = sym_state('1,2,3,4,5')
        self.s_2 = sym_state('4,5,6,7,8')
        self.s_empty = sym_state('')
        self.s_none = None

        # add conjunctive items to pool
        pool = ItemPool()

        self.ci1 = pool.get(sym_state_assert('1,2,3'), item_type=CompositeItem)  # On in s_1
        self.ci2 = pool.get(sym_state_assert('4,5'), item_type=CompositeItem)  # On in s_1 and s_2
        self.ci3 = pool.get(sym_state_assert('6,7,8'), item_type=CompositeItem)  # On in s_2
        self.ci4 = pool.get(sym_state_assert('~2,~3'), item_type=CompositeItem)  # On in s_2

    def test_held_state_1(self):
        self.assertEqual(0, len(held_state(s_prev=self.s_empty, s_curr=self.s_empty)))
        self.assertEqual(0, len(held_state(s_prev=self.s_none, s_curr=self.s_empty)))
        self.assertEqual(0, len(held_state(s_prev=self.s_empty, s_curr=self.s_none)))
        self.assertEqual(0, len(held_state(s_prev=self.s_none, s_curr=self.s_none)))

    def test_held_state_2(self):
        # basic (non-conjunctive) items
        self.assertSetEqual(set(sym_items('4,5')),
                            set(non_composite_items(held_state(s_prev=self.s_1, s_curr=self.s_2))))
        self.assertSetEqual(set(sym_items('4,5')),
                            set(non_composite_items(held_state(s_prev=self.s_2, s_curr=self.s_1))))

    def test_held_state_3(self):
        # conjunctive items
        self.assertSetEqual({self.ci2}, set(composite_items(held_state(s_prev=self.s_1, s_curr=self.s_2))))
        self.assertSetEqual({self.ci2}, set(composite_items(held_state(s_prev=self.s_2, s_curr=self.s_1))))

    def test_lost_state_1(self):
        self.assertEqual(0, len(lost_state(s_prev=self.s_empty, s_curr=self.s_empty)))
        self.assertEqual(0, len(lost_state(s_prev=self.s_none, s_curr=self.s_empty)))
        self.assertEqual(0, len(lost_state(s_prev=self.s_empty, s_curr=self.s_none)))
        self.assertEqual(0, len(lost_state(s_prev=self.s_none, s_curr=self.s_none)))

    def test_lost_state_2(self):
        # basic (non-conjunctive) items
        self.assertSetEqual(set(sym_items('1,2,3')),
                            set(non_composite_items(lost_state(s_prev=self.s_1, s_curr=self.s_2))))
        self.assertSetEqual(set(sym_items('6,7,8')),
                            set(non_composite_items(lost_state(s_prev=self.s_2, s_curr=self.s_1))))

    def test_lost_state_3(self):
        # conjunctive items
        self.assertSetEqual({self.ci1}, set(composite_items(lost_state(s_prev=self.s_1, s_curr=self.s_2))))
        self.assertSetEqual({self.ci3, self.ci4}, set(composite_items(lost_state(s_prev=self.s_2, s_curr=self.s_1))))

    def test_new_state_1(self):
        self.assertEqual(0, len(new_state(s_prev=self.s_empty, s_curr=self.s_empty)))
        self.assertEqual(0, len(new_state(s_prev=self.s_none, s_curr=self.s_empty)))
        self.assertEqual(0, len(new_state(s_prev=self.s_empty, s_curr=self.s_none)))
        self.assertEqual(0, len(new_state(s_prev=self.s_none, s_curr=self.s_none)))

    def test_new_state_2(self):
        # basic (non-conjunctive) items
        self.assertSetEqual(set(sym_items('6,7,8')),
                            set(non_composite_items(new_state(s_prev=self.s_1, s_curr=self.s_2))))
        self.assertSetEqual(set(sym_items('1,2,3')),
                            set(non_composite_items(new_state(s_prev=self.s_2, s_curr=self.s_1))))

    def test_new_state_3(self):
        # conjunctive items
        self.assertSetEqual({self.ci3, self.ci4}, set(composite_items(new_state(s_prev=self.s_1, s_curr=self.s_2))))
        self.assertSetEqual({self.ci1}, set(composite_items(new_state(s_prev=self.s_2, s_curr=self.s_1))))
