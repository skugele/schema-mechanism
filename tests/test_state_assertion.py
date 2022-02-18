import itertools
from random import sample
from time import time
from unittest import TestCase

import test_share
from schema_mechanism.data_structures import ItemPool
from schema_mechanism.data_structures import State
from schema_mechanism.data_structures import StateAssertion
from schema_mechanism.func_api import sym_assert
from schema_mechanism.func_api import sym_state
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

        self.sa = StateAssertion((sym_assert('1'), sym_assert('~2'), sym_assert('3')))

    def test_init(self):
        # Allow empty StateAssertion (no items)
        sa = StateAssertion()
        self.assertEqual(0, len(sa))

        # Support multiple assertions
        sa = StateAssertion((sym_assert('1'), sym_assert('~2'), sym_assert('3')))
        self.assertEqual(3, len(sa))
        self.assertIn(sym_assert('1'), sa)
        self.assertIn(sym_assert('~2'), sa)
        self.assertIn(sym_assert('3'), sa)

    def test_is_satisfied(self):
        state = sym_state('1,2')

        # an empty StateAssertion should always be satisfied
        c = StateAssertion()
        self.assertTrue(c.is_satisfied(state))

        # single discrete item
        #######################

        # expected to be satisfied
        c = StateAssertion((sym_assert('1'),))
        self.assertTrue(c.is_satisfied(state))

        c = StateAssertion((sym_assert('2'),))
        self.assertTrue(c.is_satisfied(state))

        c = StateAssertion((sym_assert('~3'),))
        self.assertTrue(c.is_satisfied(state))

        # expected to NOT be satisfied
        c = StateAssertion((sym_assert('3'),))
        self.assertFalse(c.is_satisfied(state))

        # multiple discrete items (all must be matched)
        ###############################################

        # expected to be satisfied
        c = StateAssertion((sym_assert('1'), sym_assert('2')))
        self.assertTrue(c.is_satisfied(state))

        c = StateAssertion((sym_assert('1'), sym_assert('~3')))
        self.assertTrue(c.is_satisfied(state))

        c = StateAssertion((sym_assert('2'), sym_assert('~3')))
        self.assertTrue(c.is_satisfied(state))

        # expected to NOT be satisfied
        c = StateAssertion((sym_assert('1'), sym_assert('~2')))
        self.assertFalse(c.is_satisfied(state))

        c = StateAssertion((sym_assert('1'),
                            sym_assert('3')))
        self.assertFalse(c.is_satisfied(state))

        c = StateAssertion((sym_assert('1'),
                            sym_assert('2'),
                            sym_assert('3')))
        self.assertFalse(c.is_satisfied(state))

    def test_is_contained(self):
        # expected to be contained (discrete)
        self.assertTrue(sym_assert('1') in self.sa)
        self.assertTrue(sym_assert('~2') in self.sa)

        # expected to be NOT contained (discrete)
        self.assertFalse(sym_assert('2') in self.sa)
        self.assertFalse(sym_assert('4') in self.sa)

    def test_replicate_with(self):
        sa = StateAssertion()

        # 1st discrete item should be added
        sa1 = sa.replicate_with(sym_assert('1'))
        self.assertIsNot(sa, sa1)
        self.assertTrue(sym_assert('1') in sa1)
        self.assertEqual(1, len(sa1))

        # 2nd discrete item should be added
        sa2 = sa1.replicate_with(sym_assert('2'))
        self.assertIsNot(sa1, sa2)
        self.assertTrue(sym_assert('2') in sa2)
        self.assertEqual(2, len(sa2))

        # identical discrete item should NOT be added
        try:
            sa2.replicate_with(sym_assert('2'))
            self.fail('Did\'t raise ValueError as expected!')
        except ValueError as e:
            self.assertEqual(str(e), 'ItemAssertion already exists in StateAssertion')

    def test_total_primitive_value_1(self):
        # empty assertion should have zero total value
        self.assertEqual(0.0, StateAssertion().total_primitive_value)

    def test_total_primitive_value_2(self):
        # single (non-negated) item state assertion should have total value equal to the item's primitive value
        ia1 = sym_assert('1', 3.0)
        sa = StateAssertion([ia1])
        self.assertEqual(ia1.item.primitive_value, sa.total_primitive_value)

    def test_total_primitive_value_3(self):
        # single negated item state assertion should have total value of 0.0
        ia2 = sym_assert('~2', 100.0)
        sa = StateAssertion([ia2])
        self.assertEqual(0.0, sa.total_primitive_value)

    def test_total_primitive_value_4(self):
        ItemPool().clear()

        # multiple (non-negated) item state assertion should have total value equal to the sum of item primitive values
        sa = StateAssertion([sym_assert(str(i), 1.0) for i in range(10)])
        self.assertEqual(10.0, sa.total_primitive_value)

    def test_total_primitive_value_5(self):
        # multiple mixed negated and non-negated item state assertion should have a total value equal to the sum of
        # the non-negated items' primitive values
        negated_ias = [sym_assert(f'~{i}', primitive_value=1.0) for i in range(1, 10, 2)]
        non_negated_ias = [sym_assert(f'{i}', primitive_value=1.0) for i in range(0, 10, 2)]

        sa = StateAssertion(list(itertools.chain.from_iterable([negated_ias, non_negated_ias])))
        self.assertEqual(sum(ia.item.primitive_value for ia in non_negated_ias), sa.total_primitive_value)

    def test_iterable(self):
        count = 0
        for _ in self.sa:
            count += 1

        self.assertEqual(3, count)

    def test_len(self):
        self.assertEqual(0, len(StateAssertion()))
        self.assertEqual(3, len(self.sa))

    def test_equal(self):
        copy = self.sa.copy()
        other = StateAssertion((sym_assert('4'),
                                sym_assert('5'),
                                sym_assert('6')))

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
        for _ in range(n_iters):
            state = State(sample(range(100), k=5))
            pos_asserts = [sym_assert(f'{i}') for i in sample(range(0, 50), k=5)]
            neg_asserts = [sym_assert(f'~{i}') for i in sample(range(50, 100), k=3)]

            sa = StateAssertion((*pos_asserts, *neg_asserts))
            start = time()
            sa.is_satisfied(state)
            end = time()
            elapsed_time += end - start

        print(f'Time to call StateAssertion.is_satisfied {n_iters:,} times: {elapsed_time}s')
