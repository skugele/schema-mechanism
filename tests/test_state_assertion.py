from random import sample
from time import time
from unittest import TestCase

import test_share
from schema_mechanism.core import Assertion
from schema_mechanism.core import CompositeItem
from schema_mechanism.core import ItemAssertion
from schema_mechanism.core import ItemPool
from schema_mechanism.core import State
from schema_mechanism.core import StateAssertion
from schema_mechanism.func_api import sym_item_assert
from schema_mechanism.func_api import sym_state
from schema_mechanism.func_api import sym_state_assert
from test_share.test_func import common_test_setup
from test_share.test_func import is_eq_consistent
from test_share.test_func import is_eq_reflexive
from test_share.test_func import is_eq_symmetric
from test_share.test_func import is_eq_transitive
from test_share.test_func import is_eq_with_null_is_false
from test_share.test_func import is_hash_consistent
from test_share.test_func import is_hash_same_for_equal_objects


class TestStateAssertion(TestCase):
    def setUp(self) -> None:
        common_test_setup()

        self.pool = ItemPool()

        # fill ItemPool
        _ = self.pool.get('1', primitive_value=-1.0)
        _ = self.pool.get('2', primitive_value=1.0)
        _ = self.pool.get('3', primitive_value=3.0)
        _ = self.pool.get('4', primitive_value=-3.0)
        _ = self.pool.get('5', primitive_value=0.0)

        self.asserts = (
            sym_item_assert('~1'), sym_item_assert('~2'), sym_item_assert('3'), sym_item_assert('4'),
            sym_item_assert('5'))
        self.sa = StateAssertion(asserts=self.asserts)

    def test_init(self):
        # test: empty StateAssertion (no item assertion) SHOULD be allowed
        self.assertEqual(0, len(StateAssertion()))

        # test: multiple item assertion SHOULD be allowed
        self.assertEqual(1, len(StateAssertion(asserts=(sym_item_assert('1'),))))

        # test: multiple item assertion SHOULD be allowed
        self.assertEqual(len(self.asserts), len(self.sa))

        for ia in self.sa:
            self.assertIn(ia, self.sa)

    def test_is_satisfied(self):
        state = sym_state('1,2')

        # test: an empty StateAssertion should always be satisfied
        c = StateAssertion()
        self.assertTrue(c.is_satisfied(state))

        # single discrete item
        #######################

        # test: the following are expected to be satisfied
        c = StateAssertion((sym_item_assert('1'),))
        self.assertTrue(c.is_satisfied(state))

        c = StateAssertion((sym_item_assert('2'),))
        self.assertTrue(c.is_satisfied(state))

        c = StateAssertion((sym_item_assert('~3'),))
        self.assertTrue(c.is_satisfied(state))

        # test: the following are expected to NOT be satisfied
        c = StateAssertion((sym_item_assert('3'),))
        self.assertFalse(c.is_satisfied(state))

        # multiple discrete items (all must be matched)
        ###############################################

        # test: the following are expected to be satisfied
        c = StateAssertion((sym_item_assert('1'), sym_item_assert('2')))
        self.assertTrue(c.is_satisfied(state))

        c = StateAssertion((sym_item_assert('1'), sym_item_assert('~3')))
        self.assertTrue(c.is_satisfied(state))

        c = StateAssertion((sym_item_assert('2'), sym_item_assert('~3')))
        self.assertTrue(c.is_satisfied(state))

        # test: the following are expected to NOT be satisfied
        c = StateAssertion((sym_item_assert('1'), sym_item_assert('~2')))
        self.assertFalse(c.is_satisfied(state))

        c = StateAssertion((sym_item_assert('1'),
                            sym_item_assert('3')))
        self.assertFalse(c.is_satisfied(state))

        c = StateAssertion((sym_item_assert('1'),
                            sym_item_assert('2'),
                            sym_item_assert('3')))
        self.assertFalse(c.is_satisfied(state))

    def test_is_contained(self):
        # test: the following are expected to be contained in the state assertion
        for ia in self.asserts:
            self.assertIn(ia, self.sa)

        # test: the following are expected to NOT be contained in the state assertion (negated versions)
        for ia in self.asserts:
            self.assertNotIn(ItemAssertion(ia.item, negated=not ia.is_negated), self.sa)

        self.assertNotIn(sym_item_assert('7'), self.sa)
        self.assertNotIn(sym_item_assert('~8'), self.sa)

    def test_replicate_with(self):
        sa = StateAssertion()

        # 1st discrete item should be added
        sa1 = Assertion.replicate_with(sa, sym_item_assert('1'))
        self.assertIsNot(sa, sa1)
        self.assertTrue(sym_item_assert('1') in sa1)
        self.assertEqual(1, len(sa1))

        # 2nd discrete item should be added
        sa2 = Assertion.replicate_with(sa1, sym_item_assert('2'))
        self.assertIsNot(sa1, sa2)
        self.assertTrue(sym_item_assert('2') in sa2)
        self.assertEqual(2, len(sa2))

        # identical discrete item should NOT be added
        self.assertRaises(ValueError, lambda: Assertion.replicate_with(sa2, sym_item_assert('2')))

    def test_primitive_value_1(self):
        # test: empty assertion should have zero total value
        self.assertEqual(0.0, StateAssertion().primitive_value)

    def test_primitive_value_2(self):
        # test: single NON-NEGATED item assertion in state assertion should have (prim. value == item's prim. value)
        ia = self.asserts[2]

        # sanity checks
        self.assertTrue(ia.item.primitive_value > 0.0)
        self.assertFalse(ia.is_negated)

        sa = StateAssertion((ia,))
        self.assertEqual(ia.item.primitive_value, sa.primitive_value)

    def test_primitive_value_3(self):
        # test: state assertion with single NEGATED item assertion should have prim. value == 0.0
        ia = self.asserts[1]

        # sanity checks
        self.assertTrue(ia.item.primitive_value > 0.0)
        self.assertTrue(ia.is_negated)

        sa = StateAssertion((ia,))
        self.assertEqual(0.0, sa.primitive_value)

    def test_primitive_value_4(self):
        # test: state assertion with multiple NON-NEGATED item assertion should have prim. value == sum of item
        # assertion' prim. values
        ias_pos = list(filter(lambda ia_: not ia_.is_negated, self.asserts))

        sa = StateAssertion(ias_pos)

        self.assertEqual(sum(ia.item.primitive_value for ia in ias_pos), sa.primitive_value)

    def test_primitive_value_5(self):
        # test: state assertion with multiple NEGATED item assertion should have prim. value == -(sum of item
        # assertion' prim. values)
        ias_neg = list(filter(lambda ia_: ia_.is_negated, self.asserts))

        sa = StateAssertion(ias_neg)

        self.assertEqual(sum(ia.item.primitive_value for ia in ias_neg), sa.primitive_value)

    def test_primitive_value_6(self):
        # test: state assertion with negated AND non-negated item assertion should have prim. value == sum(prim. value
        # of item assertion' with NON-NEGATED prim. values) - sum(prim. values of item assertion with NEGATED prim.
        # values)
        ias_pos = list(filter(lambda ia_: not ia_.is_negated, self.asserts))
        ias_neg = list(filter(lambda ia_: ia_.is_negated, self.asserts))

        sa = StateAssertion({*ias_pos, *ias_neg})
        expected_value = sum(ia.item.primitive_value for ia in ias_pos) - sum(ia.item.primitive_value for ia in ias_neg)
        self.assertEqual(expected_value, sa.primitive_value)

    def test_primitive_value_7(self):
        # test: NEGATED STATE ASSERTION with negated AND non-negated item assertion should have prim. value ==
        # -[sum(prim. value of item assertion' with NON-NEGATED prim. values) -
        #   sum(prim. values of item assertion with NEGATED prim. values)]
        ias_pos = list(filter(lambda ia_: not ia_.is_negated, self.asserts))
        ias_neg = list(filter(lambda ia_: ia_.is_negated, self.asserts))

        sa = StateAssertion({*ias_pos, *ias_neg})
        expected_value = -(
                sum(ia.item.primitive_value for ia in ias_pos) -
                sum(ia.item.primitive_value for ia in ias_neg)
        )
        self.assertEqual(expected_value, sa.primitive_value)

    def test_iterable(self):
        count = 0
        for _ in self.sa:
            count += 1

        self.assertEqual(len(self.asserts), count)

    def test_len(self):
        self.assertEqual(0, len(StateAssertion()))
        self.assertEqual(len(self.asserts), len(self.sa))

    def test_equal(self):
        copy = self.sa.copy()
        other = sym_state_assert('4,5,6')

        self.assertEqual(self.sa, self.sa)
        self.assertEqual(self.sa, copy)
        self.assertNotEqual(self.sa, other)

        self.assertTrue(is_eq_reflexive(self.sa))
        self.assertTrue(is_eq_symmetric(x=self.sa, y=copy))
        self.assertTrue(is_eq_transitive(x=self.sa, y=copy, z=copy.copy()))
        self.assertTrue(is_eq_consistent(x=self.sa, y=copy))
        self.assertTrue(is_eq_with_null_is_false(self.sa))

    def test_hash(self):
        self.assertIsInstance(hash(self.sa), int)
        self.assertTrue(is_hash_consistent(self.sa))
        self.assertTrue(is_hash_same_for_equal_objects(x=self.sa, y=self.sa.copy()))

    @test_share.performance_test
    def test_performance_1(self):
        n_items = 100

        pool = ItemPool()

        for state_element in range(n_items):
            pool.get(state_element)

        n_iters = 10_000

        elapsed_time = 0
        elapsed_time_2 = 0
        for _ in range(n_iters):
            start = time()
            state = State(sample(range(n_items), k=5))
            pos_asserts = [ItemAssertion(item=ItemPool().get(str(i))) for i in sample(range(0, 50), k=5)]
            neg_asserts = [ItemAssertion(item=ItemPool().get(str(i)), negated=True) for i in
                           sample(range(50, 100), k=3)]
            stop = time()
            elapsed_time_2 += stop - start

            sa = StateAssertion((*pos_asserts, *neg_asserts))
            start = time()
            sa.is_satisfied(state)
            end = time()
            elapsed_time += end - start

        print(f'Time to call StateAssertion.is_satisfied {n_iters:,} times: {elapsed_time}s')

        print(f'other shit: {elapsed_time_2}')

    def test_replicate_with_2(self):
        # test: OLD state assertion with NEW non-composite item assertion
        old = sym_state_assert('1,2,3')
        new = sym_item_assert('4')

        sa = Assertion.replicate_with(old, new)
        self.assertEqual(sym_state_assert('1,2,3,4'), sa)

        # test: OLD state assertion with NEW composite item assertion
        old = sym_state_assert('1,2,3')
        new = ItemAssertion(CompositeItem(sym_state_assert('4,~5')), negated=True)

        sa = Assertion.replicate_with(old, new)
        self.assertEqual(sym_state_assert('1,2,3,~(4,~5)'), sa)

        # test: OLD state assertion with NEW state assertion
        old = sym_state_assert('1,2,3')
        new = sym_state_assert('4,')

        sa = Assertion.replicate_with(old, new)
        self.assertEqual(sym_state_assert('1,2,3,4'), sa)

        # test: OLD state assertion with NEW state assertion
        old = sym_state_assert('1,2,3')
        new = sym_state_assert('4,')

        sa = Assertion.replicate_with(old, new)
        self.assertEqual(sym_state_assert('1,2,3,4'), sa)
