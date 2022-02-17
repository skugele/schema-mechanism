import itertools
from random import sample
from time import time
from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np

import test_share
from schema_mechanism.data_structures import Action
from schema_mechanism.data_structures import ItemPool
from schema_mechanism.data_structures import ItemPoolStateView
from schema_mechanism.data_structures import NULL_STATE_ASSERT
from schema_mechanism.data_structures import Schema
from schema_mechanism.data_structures import State
from schema_mechanism.data_structures import StateAssertion
from schema_mechanism.data_structures import SymbolicItem
from schema_mechanism.func_api import sym_schema
from schema_mechanism.func_api import sym_state
from schema_mechanism.func_api import sym_state_assert
from schema_mechanism.modules import lost_state
from schema_mechanism.modules import new_state
from schema_mechanism.modules import update_schema
from test_share.test_classes import MockObserver
from test_share.test_func import common_test_setup
from test_share.test_func import is_eq_consistent
from test_share.test_func import is_eq_reflexive
from test_share.test_func import is_eq_symmetric
from test_share.test_func import is_eq_transitive
from test_share.test_func import is_eq_with_null_is_false
from test_share.test_func import is_hash_consistent
from test_share.test_func import is_hash_same_for_equal_objects


class TestSchema(TestCase):
    def setUp(self) -> None:
        common_test_setup()

        self._item_pool = ItemPool()

        # populate pool
        for i in range(10):
            _ = self._item_pool.get(str(i), primitive_value=1.0, item_type=SymbolicItem)

        self.schema = sym_schema('0,1/A1/2,3,4')

        self.obs = MockObserver()
        self.schema.register(self.obs)

    def test_init(self):
        # Action CANNOT be None
        try:
            # noinspection PyTypeChecker
            Schema(action=None)
            self.fail('Action=None should generate a ValueError')
        except ValueError as e:
            self.assertEqual('Action cannot be None', str(e))

        # Context and Result CAN be None
        try:
            s = Schema(action=Action('My Action'))
            self.assertIs(s.context, NULL_STATE_ASSERT)
            self.assertIs(s.result, NULL_STATE_ASSERT)
        except Exception as e:
            self.fail(f'Unexpected exception raised: {e}')

        # Verify immutability
        s = Schema(context=sym_state_assert('1'),
                   action=Action(),
                   result=sym_state_assert('2'))

        try:
            # noinspection PyPropertyAccess
            s.context = StateAssertion()
            self.fail('Schema\'s context is not immutable as expected')
        except AttributeError:
            pass

        try:
            # noinspection PyPropertyAccess
            s.result = StateAssertion()
            self.fail('Schema\'s result is not immutable as expected')
        except AttributeError:
            pass

        try:
            # noinspection PyPropertyAccess
            s.action = Action()
            self.fail('Schema\'s action is not immutable as expected')
        except AttributeError:
            pass

    def test_items_in_context_and_result(self):
        for item in itertools.chain.from_iterable([self.schema.context.items, self.schema.result.items]):
            self.assertIsInstance(item, SymbolicItem)
            self.assertEqual(1.0, item.primitive_value)

        # all schemas should share the same underlying items
        s1 = sym_schema('1,2/A1/3,4')
        s2 = sym_schema('1,2/A2/3,4')

        for i1, i2 in zip(s1.context.items, s2.context.items):
            self.assertIs(i1, i2)

        for i1, i2 in zip(s1.result.items, s2.result.items):
            self.assertIs(i1, i2)

    def test_is_context_satisfied(self):
        c = sym_state_assert('1,~2,3')
        schema = Schema(context=c, action=Action(), result=None)

        # expected to be satisfied
        ##########################
        self.assertTrue(schema.context.is_satisfied(state=sym_state('1,3')))
        self.assertTrue(schema.context.is_satisfied(state=sym_state('1,3,4')))

        # expected to NOT be satisfied
        ##############################
        # case 1: present negated item
        self.assertFalse(schema.context.is_satisfied(state=sym_state('1,2,3')))

        # case 2: missing non-negated item
        self.assertFalse(schema.context.is_satisfied(state=sym_state('1')))
        self.assertFalse(schema.context.is_satisfied(state=sym_state('3')))

        # case 3 : both present negated item and missing non-negated item
        self.assertFalse(schema.context.is_satisfied(state=sym_state('1,2')))
        self.assertFalse(schema.context.is_satisfied(state=sym_state('2,3')))

    def test_is_applicable(self):
        # primitive schemas should always be applicable
        schema = Schema(action=Action())
        self.assertTrue(schema.is_applicable(state=sym_state('1')))

        c = sym_state_assert('1,~2,3')

        schema = Schema(context=c, action=Action(), result=None)

        # expected to be applicable
        ##########################
        self.assertTrue(schema.is_applicable(state=sym_state('1,3')))
        self.assertTrue(schema.is_applicable(state=sym_state('1,3,4')))

        # expected to NOT be applicable
        ###############################

        # case 1: present negated item
        self.assertFalse(schema.is_applicable(state=sym_state('1,2,3')))

        # case 2: missing non-negated item
        self.assertFalse(schema.is_applicable(state=sym_state('1')))
        self.assertFalse(schema.is_applicable(state=sym_state('3')))

        # case 3 : both present negated item and missing non-negated item
        self.assertFalse(schema.is_applicable(state=sym_state('1,2')))
        self.assertFalse(schema.is_applicable(state=sym_state('2,3')))

        # Tests overriding conditions
        #############################
        schema.overriding_conditions = sym_state_assert('5')

        # expected to be applicable
        self.assertTrue(schema.is_applicable(state=sym_state('1,3,4')))

        # expected to NOT be applicable (due to overriding condition)
        self.assertFalse(schema.is_applicable(state=sym_state('1,3,4,5')))

    def test_update_1(self):
        # activated + success
        s_prev = sym_state('0,1,2,3')
        s_curr = sym_state('2,3,4,5')

        v_prev = ItemPoolStateView(s_prev)
        v_curr = ItemPoolStateView(s_curr)

        new = new_state(s_prev, s_curr)
        lost = lost_state(s_prev, s_curr)

        self.schema.update(activated=True, v_prev=v_prev, v_curr=v_curr, new=new, lost=lost)

        # check schema level statistics
        self.assertEqual(1, self.schema.stats.n)
        self.assertEqual(1, self.schema.stats.n_success)
        self.assertEqual(0, self.schema.stats.n_fail)
        self.assertEqual(1, self.schema.stats.n_activated)

        # check item level statistics
        for i in self._item_pool:
            ec_stats = self.schema.extended_context.stats[i]

            if v_prev.is_on(i):
                self.assertEqual(1, ec_stats.n_success_and_on)
                self.assertEqual(0, ec_stats.n_success_and_off)
                self.assertEqual(0, ec_stats.n_fail_and_on)
                self.assertEqual(0, ec_stats.n_fail_and_off)
            else:
                self.assertEqual(0, ec_stats.n_success_and_on)
                self.assertEqual(1, ec_stats.n_success_and_off)
                self.assertEqual(0, ec_stats.n_fail_and_on)
                self.assertEqual(0, ec_stats.n_fail_and_off)

            er_stats = self.schema.extended_result.stats.get(i)

            if i in new:
                self.assertEqual(1, er_stats.n_on_and_activated)
                self.assertEqual(0, er_stats.n_on_and_not_activated)
                self.assertEqual(0, er_stats.n_off_and_activated)
                self.assertEqual(0, er_stats.n_off_and_not_activated)
            elif i in lost:
                self.assertEqual(0, er_stats.n_on_and_activated)
                self.assertEqual(0, er_stats.n_on_and_not_activated)
                self.assertEqual(1, er_stats.n_off_and_activated)
                self.assertEqual(0, er_stats.n_off_and_not_activated)

    def test_update_2(self):
        # activated and not success
        s_prev = sym_state('0,1,2,3')
        s_curr = sym_state('2,5')

        v_prev = ItemPoolStateView(s_prev)
        v_curr = ItemPoolStateView(s_curr)

        new = new_state(s_prev, s_curr)
        lost = lost_state(s_prev, s_curr)

        self.schema.update(activated=True, v_prev=v_prev, v_curr=v_curr, new=new, lost=lost)

        # check schema level statistics
        self.assertEqual(1, self.schema.stats.n)
        self.assertEqual(0, self.schema.stats.n_success)
        self.assertEqual(1, self.schema.stats.n_fail)
        self.assertEqual(1, self.schema.stats.n_activated)

        # check item level statistics
        for i in self._item_pool:
            ec_stats = self.schema.extended_context.stats[i]

            if v_prev.is_on(i):
                self.assertEqual(0, ec_stats.n_success_and_on)
                self.assertEqual(0, ec_stats.n_success_and_off)
                self.assertEqual(1, ec_stats.n_fail_and_on)
                self.assertEqual(0, ec_stats.n_fail_and_off)
            else:
                self.assertEqual(0, ec_stats.n_success_and_on)
                self.assertEqual(0, ec_stats.n_success_and_off)
                self.assertEqual(0, ec_stats.n_fail_and_on)
                self.assertEqual(1, ec_stats.n_fail_and_off)

            er_stats = self.schema.extended_result.stats.get(i)

            if i in new:
                self.assertEqual(1, er_stats.n_on_and_activated)
                self.assertEqual(0, er_stats.n_on_and_not_activated)
                self.assertEqual(0, er_stats.n_off_and_activated)
                self.assertEqual(0, er_stats.n_off_and_not_activated)
            elif i in lost:
                self.assertEqual(0, er_stats.n_on_and_activated)
                self.assertEqual(0, er_stats.n_on_and_not_activated)
                self.assertEqual(1, er_stats.n_off_and_activated)
                self.assertEqual(0, er_stats.n_off_and_not_activated)

    def test_update_3(self):
        # not activated
        s_prev = sym_state('0,3')
        s_curr = sym_state('2,4,5')

        v_prev = ItemPoolStateView(s_prev)
        v_curr = ItemPoolStateView(s_curr)

        new = new_state(s_prev, s_curr)
        lost = lost_state(s_prev, s_curr)

        self.schema.update(activated=False, v_prev=v_prev, v_curr=v_curr, new=new, lost=lost)

        # check schema level statistics
        self.assertEqual(1, self.schema.stats.n)
        self.assertEqual(0, self.schema.stats.n_success)
        self.assertEqual(0, self.schema.stats.n_fail)
        self.assertEqual(0, self.schema.stats.n_activated)

        # check item level statistics
        for i in self._item_pool:
            ec_stats = self.schema.extended_context.stats[i]

            if v_prev.is_on(i):
                self.assertEqual(0, ec_stats.n_success_and_on)
                self.assertEqual(0, ec_stats.n_success_and_off)
                self.assertEqual(0, ec_stats.n_fail_and_on)
                self.assertEqual(0, ec_stats.n_fail_and_off)
            else:
                self.assertEqual(0, ec_stats.n_success_and_on)
                self.assertEqual(0, ec_stats.n_success_and_off)
                self.assertEqual(0, ec_stats.n_fail_and_on)
                self.assertEqual(0, ec_stats.n_fail_and_off)

            er_stats = self.schema.extended_result.stats.get(i)

            if i in new:
                self.assertEqual(0, er_stats.n_on_and_activated)
                self.assertEqual(1, er_stats.n_on_and_not_activated)
                self.assertEqual(0, er_stats.n_off_and_activated)
                self.assertEqual(0, er_stats.n_off_and_not_activated)
            elif i in lost:
                self.assertEqual(0, er_stats.n_on_and_activated)
                self.assertEqual(0, er_stats.n_on_and_not_activated)
                self.assertEqual(0, er_stats.n_off_and_activated)
                self.assertEqual(1, er_stats.n_off_and_not_activated)

    def test_update_4(self):

        s_prev = None
        s_curr = sym_state('2,4,5')

        v_prev = ItemPoolStateView(s_prev)
        v_curr = ItemPoolStateView(s_curr)

        new = new_state(s_prev, s_curr)
        lost = lost_state(s_prev, s_curr)

        self.schema.update(activated=False, v_prev=v_prev, v_curr=v_curr, new=new, lost=lost)

        # check schema level statistics
        self.assertEqual(1, self.schema.stats.n)
        self.assertEqual(0, self.schema.stats.n_success)
        self.assertEqual(0, self.schema.stats.n_fail)
        self.assertEqual(0, self.schema.stats.n_activated)

        # check item level statistics
        for i in self._item_pool:
            ec_stats = self.schema.extended_context.stats[i]

            if v_prev.is_on(i):
                self.assertEqual(0, ec_stats.n_success_and_on)
                self.assertEqual(0, ec_stats.n_success_and_off)
                self.assertEqual(0, ec_stats.n_fail_and_on)
                self.assertEqual(0, ec_stats.n_fail_and_off)
            else:
                self.assertEqual(0, ec_stats.n_success_and_on)
                self.assertEqual(0, ec_stats.n_success_and_off)
                self.assertEqual(0, ec_stats.n_fail_and_on)
                self.assertEqual(0, ec_stats.n_fail_and_off)

            er_stats = self.schema.extended_result.stats.get(i)

            if i in new:
                self.assertEqual(0, er_stats.n_on_and_activated)
                self.assertEqual(1, er_stats.n_on_and_not_activated)
                self.assertEqual(0, er_stats.n_off_and_activated)
                self.assertEqual(0, er_stats.n_off_and_not_activated)
            elif i in lost:
                self.assertEqual(0, er_stats.n_on_and_activated)
                self.assertEqual(0, er_stats.n_on_and_not_activated)
                self.assertEqual(0, er_stats.n_off_and_activated)
                self.assertEqual(1, er_stats.n_off_and_not_activated)

    def test_reliability_0(self):
        # reliability before update should be NAN
        self.assertIs(np.NAN, self.schema.reliability)

    def test_reliability_1(self):
        # success update
        update_schema(self.schema, activated=True, s_prev=sym_state('0,1'), s_curr=sym_state('2,3,4,5'), count=1)
        self.assertEqual(1.0, self.schema.reliability)

        # failure update
        update_schema(self.schema, activated=True, s_prev=sym_state('0,1'), s_curr=sym_state('2,3,5'), count=1)
        self.assertEqual(0.5, self.schema.reliability)

        # failure update
        update_schema(self.schema, activated=True, s_prev=sym_state('0,1'), s_curr=sym_state('2,3,5'), count=2)
        self.assertEqual(0.25, self.schema.reliability)

    def test_reliability_2(self):
        # failure update
        update_schema(self.schema, activated=True, s_prev=sym_state('0,1'), s_curr=sym_state('2,3,5'), count=1)
        self.assertEqual(0.0, self.schema.reliability)

        # success update
        update_schema(self.schema, activated=True, s_prev=sym_state('0,1'), s_curr=sym_state('2,3,4,5'), count=1)
        self.assertEqual(0.5, self.schema.reliability)

        # success update
        update_schema(self.schema, activated=True, s_prev=sym_state('0,1'), s_curr=sym_state('2,3,4,5'), count=2)
        self.assertEqual(0.75, self.schema.reliability)

    def test_reliability_3(self):
        # reliability stats SHOULD NOT affected when schema not activated

        # failure update WITHOUT activation
        update_schema(self.schema, activated=False, s_prev=sym_state('0,1'), s_curr=sym_state('2,3,5'), count=1)
        self.assertIs(np.NAN, self.schema.reliability)

        # success update WITHOUT activation
        update_schema(self.schema, activated=False, s_prev=sym_state('0,1'), s_curr=sym_state('2,3,4,5'), count=1)
        self.assertIs(np.NAN, self.schema.reliability)

        # failure update WITHOUT activation
        update_schema(self.schema, activated=False, s_prev=sym_state('0,1'), s_curr=sym_state('2,3,5'), count=1)
        self.assertIs(np.NAN, self.schema.reliability)

        # success update WITHOUT activation
        update_schema(self.schema, activated=False, s_prev=sym_state('0,1'), s_curr=sym_state('2,3,4,5'), count=1)
        self.assertIs(np.NAN, self.schema.reliability)

        # success update WITH activation
        update_schema(self.schema, activated=True, s_prev=sym_state('0,1'), s_curr=sym_state('2,3,4,5'), count=1)
        self.assertEqual(1.0, self.schema.reliability)

        # failure update WITHOUT activation
        update_schema(self.schema, activated=False, s_prev=sym_state('0,1'), s_curr=sym_state('2,3,5'), count=1)
        self.assertEqual(1.0, self.schema.reliability)

    def test_notify_all(self):
        self.schema.notify_all = MagicMock()

        act_state = sym_state('0,1,8')
        result_state = sym_state('2,3,4,7')

        # activated + success
        update_schema(self.schema, activated=True, s_prev=act_state, s_curr=result_state, count=1)

        act_state = sym_state('0,1')
        result_state = sym_state('2,3,7')

        # activated + not success
        update_schema(self.schema, activated=True, s_prev=act_state, s_curr=result_state, count=1)
        self.schema.notify_all.assert_called()

        act_state = sym_state('0,1,7')
        result_state = sym_state('2')

        # not activated and not success
        update_schema(self.schema, activated=False, s_prev=act_state, s_curr=result_state, count=1)
        self.schema.notify_all.assert_called()

    def test_copy(self):
        copy = self.schema.copy()

        self.assertEqual(self.schema, copy)
        self.assertIsNot(self.schema, copy)

    def test_equal(self):
        copy = self.schema.copy()
        other = Schema(
            context=sym_state_assert('1,2'),
            action=Action(),
            result=sym_state_assert('3,4,5'),
        )

        self.assertEqual(self.schema, self.schema)
        self.assertEqual(self.schema, copy)
        self.assertNotEqual(self.schema, other)

        self.assertTrue(is_eq_reflexive(self.schema))
        self.assertTrue(is_eq_symmetric(x=self.schema, y=copy))
        self.assertTrue(is_eq_transitive(x=self.schema, y=copy, z=copy.copy()))
        self.assertTrue(is_eq_consistent(x=self.schema, y=copy))
        self.assertTrue(is_eq_with_null_is_false(self.schema))

    def test_hash(self):
        self.assertIsInstance(hash(self.schema), int)
        self.assertTrue(is_hash_consistent(self.schema))
        self.assertTrue(is_hash_same_for_equal_objects(x=self.schema, y=self.schema.copy()))

    @test_share.performance_test
    def test_performance_1(self):
        n_items = 10_000
        n_state_elements = 25

        # populate item pool
        self._item_pool.clear()
        for i in range(n_items):
            self._item_pool.get(i)

        s_prev = State(sample(range(n_items), k=n_state_elements))
        s_curr = State(sample(range(n_items), k=n_state_elements))

        v_prev = ItemPoolStateView(s_prev)
        v_curr = ItemPoolStateView(s_curr)

        new = new_state(s_prev, s_curr)
        lost = lost_state(s_prev, s_curr)

        # TODO: This is WAY too slow...
        start = time()
        self.schema.update(activated=True,
                           v_prev=v_prev,
                           v_curr=v_curr,
                           new=new,
                           lost=lost)
        end = time()
        elapsed_time = end - start

        print(f'Time updating a schema with {n_items:,} items : {elapsed_time}s ')

    @test_share.performance_test
    def test_performance_2(self):
        # NOTE: n_items does not seem to impact performance much
        n_items = 100_000

        # NOTE: elapsed time seems to scale linearly with n_context_elements
        n_context_elements = 3

        # NOTE: elapsed time seems to scale linearly with n_schemas
        n_schemas = 10_000

        # populate item pool
        self._item_pool.clear()
        for i in range(n_items):
            self._item_pool.get(i)

        schemas = [
            Schema(context=sym_state_assert(','.join(str(i) for i in sample(range(n_items), k=n_context_elements))),
                   action=Action()) for _ in range(n_schemas)
        ]

        state = State(sample(range(n_items), k=n_context_elements))

        start = time()
        for s in schemas:
            s.is_applicable(state)
        end = time()
        elapsed_time = end - start

        print(f'Time determining if {n_schemas:,} schemas are applicable: {elapsed_time}s ')

    @test_share.performance_test
    def test_performance_3(self):
        n_iters = 100_000

        copy = self.schema.copy()
        other = Schema(
            context=sym_state_assert('1,2'),
            action=Action(),
            result=sym_state_assert('3,4,5'),
        )

        start = time()
        for _ in range(n_iters):
            _ = self.schema == other
        end = time()
        elapsed_time = end - start

        print(f'Time for {n_iters:,} calls to Schema.__eq__ comparing unequal objects: {elapsed_time}s ')

        start = time()
        for _ in range(n_iters):
            _ = self.schema == copy
        end = time()
        elapsed_time = end - start

        print(f'Time for {n_iters:,} calls to Schema.__eq__ comparing equal objects: {elapsed_time}s ')

    @test_share.performance_test
    def test_performance_4(self):
        n_iters = 100_000

        start = time()
        for _ in range(n_iters):
            _ = hash(self.schema)
        end = time()
        elapsed_time = end - start

        print(f'Time for {n_iters:,} calls to Schema.__hash__: {elapsed_time}s ')

    @test_share.string_test
    def test_str(self):
        schema = Schema(action=Action('A1'))
        print(schema)
