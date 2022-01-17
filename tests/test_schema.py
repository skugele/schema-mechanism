from random import sample
from time import time
from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np
from anytree import AsciiStyle
from anytree import RenderTree

import test_share
from schema_mechanism.data_structures import Action
from schema_mechanism.data_structures import Context
from schema_mechanism.data_structures import ItemPool
from schema_mechanism.data_structures import ItemPoolStateView
from schema_mechanism.data_structures import Result
from schema_mechanism.data_structures import Schema
from schema_mechanism.data_structures import StateAssertion
from schema_mechanism.data_structures import SymbolicItem
from schema_mechanism.func_api import make_assertion
from schema_mechanism.func_api import make_assertions
from schema_mechanism.func_api import update_schema
from schema_mechanism.modules import create_spin_off
from schema_mechanism.modules import lost_state
from schema_mechanism.modules import new_state
from test_share.test_classes import MockObserver
from test_share.test_func import is_eq_consistent
from test_share.test_func import is_eq_reflexive
from test_share.test_func import is_eq_symmetric
from test_share.test_func import is_eq_transitive
from test_share.test_func import is_eq_with_null_is_false
from test_share.test_func import is_hash_consistent
from test_share.test_func import is_hash_same_for_equal_objects


class TestSchema(TestCase):
    def setUp(self) -> None:
        self._item_pool = ItemPool()

        # populate pool
        for i in range(10):
            _ = self._item_pool.get(i, SymbolicItem)

        self.context = Context(make_assertions((0, 1)))
        self.action = Action()
        self.result = Result(make_assertions((2, 3, 4)))

        self.schema = Schema(
            context=self.context,
            action=self.action,
            result=self.result
        )

        self.obs = MockObserver()
        self.schema.register(self.obs)

    def test_init(self):
        # Action CANNOT be None
        try:
            Schema(action=None)
            self.fail('Action=None should generate a ValueError')
        except ValueError as e:
            self.assertEqual('Action cannot be None', str(e))

        # Context and Result CAN be None
        try:
            s = Schema(action=Action('My Action'))
            self.assertIsNone(s.context)
            self.assertIsNone(s.result)
        except Exception as e:
            self.fail(f'Unexpected exception raised: {e}')

        # Verify immutability
        s = Schema(context=Context(item_asserts=(make_assertion('1'),)),
                   action=Action(),
                   result=Result(item_asserts=(make_assertion('2'),)))

        try:
            s.context = Context()
            self.fail('Schema\'s context is not immutable as expected')
        except Exception as e:
            pass

        try:
            s.result = Result()
            self.fail('Schema\'s result is not immutable as expected')
        except Exception as e:
            pass

        try:
            s.action = Action()
            self.fail('Schema\'s action is not immutable as expected')
        except Exception as e:
            pass

        self.assertEqual(None, self.schema.spin_off_type)

    def test_is_context_satisfied(self):
        c = Context((
            make_assertion('1'),
            make_assertion('2', negated=True),
            make_assertion('3')
        ))

        schema = Schema(context=c, action=Action(), result=None)

        # expected to be satisfied
        ##########################
        self.assertTrue(schema.context.is_satisfied(state=['1', '3']))
        self.assertTrue(schema.context.is_satisfied(state=['1', '3', '4']))

        # expected to NOT be satisfied
        ##############################
        # case 1: present negated item
        self.assertFalse(schema.context.is_satisfied(state=['1', '2', '3']))

        # case 2: missing non-negated item
        self.assertFalse(schema.context.is_satisfied(state=['1']))
        self.assertFalse(schema.context.is_satisfied(state=['3']))

        # case 3 : both present negated item and missing non-negated item
        self.assertFalse(schema.context.is_satisfied(state=['1', '2']))
        self.assertFalse(schema.context.is_satisfied(state=['2', '3']))

    def test_is_applicable(self):
        c = Context(
            item_asserts=(
                make_assertion('1'),
                make_assertion('2', negated=True),
                make_assertion('3'),
            ))

        schema = Schema(context=c, action=Action(), result=None)

        # expected to be applicable
        ##########################
        self.assertTrue(schema.is_applicable(state=['1', '3']))
        self.assertTrue(schema.is_applicable(state=['1', '3', '4']))

        # expected to NOT be applicable
        ###############################

        # case 1: present negated item
        self.assertFalse(schema.is_applicable(state=['1', '2', '3']))

        # case 2: missing non-negated item
        self.assertFalse(schema.is_applicable(state=['1']))
        self.assertFalse(schema.is_applicable(state=['3']))

        # case 3 : both present negated item and missing non-negated item
        self.assertFalse(schema.is_applicable(state=['1', '2']))
        self.assertFalse(schema.is_applicable(state=['2', '3']))

        # Tests overriding conditions
        #############################
        schema.overriding_conditions = StateAssertion((make_assertion('5'),))

        # expected to be applicable
        self.assertTrue(schema.is_applicable(state=['1', '3', '4']))

        # expected to NOT be applicable (due to overriding condition)
        self.assertFalse(schema.is_applicable(state=['1', '3', '4', '5']))

    def test_update_1(self):
        # activated + success
        s_prev = [0, 1, 2, 3]
        s_curr = [2, 3, 4, 5]

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
        s_prev = [0, 1, 2, 3]
        s_curr = [2, 5]

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
        s_prev = [0, 3]
        s_curr = [2, 4, 5]

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
        s_curr = [2, 4, 5]

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
        update_schema(self.schema, activated=True, s_prev=[0, 1], s_curr=[2, 3, 4, 5], count=1)
        self.assertEqual(1.0, self.schema.reliability)

        # failure update
        update_schema(self.schema, activated=True, s_prev=[0, 1], s_curr=[2, 3, 5], count=1)
        self.assertEqual(0.5, self.schema.reliability)

        # failure update
        update_schema(self.schema, activated=True, s_prev=[0, 1], s_curr=[2, 3, 5], count=2)
        self.assertEqual(0.25, self.schema.reliability)

    def test_reliability_2(self):
        # failure update
        update_schema(self.schema, activated=True, s_prev=[0, 1], s_curr=[2, 3, 5], count=1)
        self.assertEqual(0.0, self.schema.reliability)

        # success update
        update_schema(self.schema, activated=True, s_prev=[0, 1], s_curr=[2, 3, 4, 5], count=1)
        self.assertEqual(0.5, self.schema.reliability)

        # success update
        update_schema(self.schema, activated=True, s_prev=[0, 1], s_curr=[2, 3, 4, 5], count=2)
        self.assertEqual(0.75, self.schema.reliability)

    def test_reliability_3(self):
        # reliability stats SHOULD NOT affected when schema not activated

        # failure update WITHOUT activation
        update_schema(self.schema, activated=False, s_prev=[0, 1], s_curr=[2, 3, 5], count=1)
        self.assertIs(np.NAN, self.schema.reliability)

        # success update WITHOUT activation
        update_schema(self.schema, activated=False, s_prev=[0, 1], s_curr=[2, 3, 4, 5], count=1)
        self.assertIs(np.NAN, self.schema.reliability)

        # failure update WITHOUT activation
        update_schema(self.schema, activated=False, s_prev=[0, 1], s_curr=[2, 3, 5], count=1)
        self.assertIs(np.NAN, self.schema.reliability)

        # success update WITHOUT activation
        update_schema(self.schema, activated=False, s_prev=[0, 1], s_curr=[2, 3, 4, 5], count=1)
        self.assertIs(np.NAN, self.schema.reliability)

        # success update WITH activation
        update_schema(self.schema, activated=True, s_prev=[0, 1], s_curr=[2, 3, 4, 5], count=1)
        self.assertEqual(1.0, self.schema.reliability)

        # failure update WITHOUT activation
        update_schema(self.schema, activated=False, s_prev=[0, 1], s_curr=[2, 3, 5], count=1)
        self.assertEqual(1.0, self.schema.reliability)

    def test_notify_all(self):
        self.schema.notify_all = MagicMock()

        # activated + success
        update_schema(self.schema, activated=True, s_prev=[0, 1], s_curr=[2, 3, 4, 5], count=10)
        self.schema.notify_all.assert_not_called()

        # activated and not success
        update_schema(self.schema, activated=True, s_prev=[0, 1, 7], s_curr=[3, 4, 7], count=10)
        self.schema.notify_all.assert_called()

        # not activated
        update_schema(self.schema, activated=False, s_prev=[0, 3], s_curr=[4, 7, 8], count=10)
        self.schema.notify_all.assert_called()

    def test_copy(self):
        copy = self.schema.copy()

        self.assertEqual(self.schema, copy)
        self.assertIsNot(self.schema, copy)

    def test_equal(self):
        copy = self.schema.copy()
        other = Schema(
            context=Context(make_assertions((1, 2))),
            action=Action(),
            result=Result(make_assertions((3, 4, 5)))
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

    @test_share.string_test
    def test_graph(self):
        parent = Schema(action=Action())

        child1 = create_spin_off(parent, mode=Schema.SpinOffType.CONTEXT, item_assert=make_assertion(1))
        child2 = create_spin_off(parent, mode=Schema.SpinOffType.CONTEXT, item_assert=make_assertion(2))
        child3 = create_spin_off(parent, mode=Schema.SpinOffType.RESULT, item_assert=make_assertion(3))
        child4 = create_spin_off(parent, mode=Schema.SpinOffType.RESULT, item_assert=make_assertion(4))

        parent.children += (child1, child2, child3, child4)

        child1_1 = create_spin_off(child1, mode=Schema.SpinOffType.CONTEXT, item_assert=make_assertion(2))
        child1_2 = create_spin_off(child1, mode=Schema.SpinOffType.CONTEXT, item_assert=make_assertion(3))

        child1.children += (child1_1, child1_2)

        child2_1 = create_spin_off(child2, mode=Schema.SpinOffType.CONTEXT, item_assert=make_assertion(1))
        child2_2 = create_spin_off(child2, mode=Schema.SpinOffType.CONTEXT, item_assert=make_assertion(3))

        child2.children += (child2_1, child2_2)

        child3_1 = create_spin_off(parent, mode=Schema.SpinOffType.RESULT, item_assert=make_assertion(1))

        child3.children += (child3_1,)

        print(RenderTree(parent, style=AsciiStyle()).by_attr(lambda s: str(s)))

    @test_share.performance_test
    def test_performance_1(self):
        n_items = 10_000
        n_state_elements = 25

        # populate item pool
        self._item_pool.clear()
        for i in range(n_items):
            self._item_pool.get(i)

        s_prev = sample(range(n_items), k=n_state_elements)
        s_curr = sample(range(n_items), k=n_state_elements)

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

        schemas = [Schema(context=Context(make_assertions(sample(range(n_items), k=n_context_elements))),
                          action=Action()) for _ in range(n_schemas)]

        state = sample(range(n_items), k=n_context_elements)

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
            context=Context(make_assertions((1, 2))),
            action=Action(),
            result=Result(make_assertions((3, 4, 5)))
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
